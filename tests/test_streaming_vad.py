"""Behavioral tests for the MLX Venn port and causal streaming/feedback contracts.

Real Metal execution requires an unsandboxed Apple Silicon test process. The
optional model test uses a previously downloaded, checksum-verified reader.
"""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", exc_type=ImportError)

from venn.audio.cli import command, open_wav  # noqa: E402
from venn.audio.reader import DIMENSIONS, FRAME, PROFILE, SileroReader  # noqa: E402
from venn.audio.session import Boundaries, BoundaryConfig, VADSession  # noqa: E402
from venn.heads.mlx_choice import MLXLinearChoiceHead  # noqa: E402


class Reader:
    def encode(self, samples):
        x = np.zeros(DIMENSIONS, np.float32)
        x[0] = float(np.mean(samples)) * 3
        x[128] = 0.1  # Deliberately imperfect constant baseline, p ~= .69.
        return x, float(1 / (1 + np.exp(-0.8)))


def pcm(value, frames=1):
    return np.full(FRAME * frames, value, dtype="<i2").tobytes()


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_mlx_matches_actual_venn_linear_choice_head(device):
    import torch
    from venn.heads.linear_choice import LinearChoiceHead

    rng = np.random.default_rng(123)
    W = rng.normal(size=(3, 768)).astype(np.float32)
    b = rng.normal(size=3).astype(np.float32)
    O = rng.normal(size=(3, 768)).astype(np.float32)
    X = rng.normal(size=(7, 768)).astype(np.float32)
    reference = LinearChoiceHead(W, b, O)
    expected = reference(torch.tensor(X), None, torch.tensor(np.broadcast_to(O, (7, *O.shape)).copy()),
                         torch.ones((7, 3), dtype=torch.bool)).detach().numpy()
    actual = MLXLinearChoiceHead(W, b, ["a", "b", "c"], device=device, temperature=1.7)
    np.testing.assert_allclose(actual.logits(X), expected[:, :-1], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual.probabilities(X), torch.softmax(torch.tensor(expected[:, :-1]) / 1.7, -1),
                               rtol=2e-5, atol=2e-5)
    assert np.all(expected[:, -1] == -1e4)


def test_chunking_preserves_every_sample_and_never_pads():
    source = pcm(4000, 3) + pcm(-5000, 2) + b'\x00'
    whole = VADSession(Reader(), device="cpu")
    expected = whole.push(source)
    split = VADSession(Reader(), device="cpu")
    actual = []
    for start in range(0, len(source), 127):
        actual.extend(split.push(source[start:start + 127]))
    assert len(actual) == len(expected) == 5
    for a, b in zip(actual, expected):
        for key in ("start_sample", "end_sample", "decision", "baseline_probability", "boundary"):
            assert a[key] == b[key]
    assert split.finish()["discarded_tail_bytes"] == 1
    assert split.finish()["discarded_tail_bytes"] == 0
    assert split.samples == 5 * FRAME


def test_boundaries_have_hysteresis_and_preroll():
    state = Boundaries(BoundaryConfig(pre_roll_ms=32))
    assert state.push(.9, 512) is None
    event = state.push(.9, 1024)
    assert event == {"type": "speech_start", "sample": 0, "seconds": 0.0}
    for end in (1536, 2048, 2560, 3072):
        assert state.push(.1, end) is None
        assert state.active
    assert state.push(.1, 3584) == {"type": "speech_end", "sample": 1024, "seconds": .064}
    assert not state.active
    assert state.push(.9, 4096) is None
    assert state.push(.4, 4608) is None
    assert state.push(.9, 5120) is None  # Interrupted onset run starts again.


def test_online_corrections_learn_and_checkpoint_roundtrip(tmp_path):
    state = VADSession(Reader(), device="cpu")
    background = state.push(pcm(-10000, 16))
    speech = state.push(pcm(10000, 16))
    assert all(event["version"] == 0 for event in background + speech)
    state.feedback(0, .512, "background")
    assert not state.learn()["accepted"]  # Both classes are required.
    state.feedback(.512, 1.024, "speech")
    heldout = np.array([Reader().encode(np.full(FRAME, v, np.float32))[0] for v in (-.28, .28)])
    before = state.head.probabilities(heldout)
    for _ in range(10):
        result = state.learn()
        assert result["accepted"] and result["loss_after"] < result["loss_before"]
    after = state.head.probabilities(heldout)
    assert after[0, 1] < before[0, 1] - .25
    assert after[1, 1] > before[1, 1] + .1
    path = tmp_path / "learned.npz"
    state.save(path)
    restored = VADSession(Reader(), device="cpu", checkpoint=path)
    np.testing.assert_array_equal(restored.head.probabilities(heldout), after)
    assert restored.version == state.version
    assert [len(m) for m in restored.replay] == [16, 16]
    assert restored.samples == 0 and len(restored.history) == 0
    restored.reset_learning()
    np.testing.assert_allclose(restored.head.probabilities(heldout), before)
    assert VADSession(Reader(), device="cpu", checkpoint=path).version == 0


def test_feedback_relabels_instead_of_duplicating_and_rejects_future():
    state = VADSession(Reader(), device="cpu")
    state.push(pcm(10, 16))
    assert state.feedback(0, .512, "speech")["class_counts"] == [0, 16]
    assert state.feedback(0, .512, "speech")["class_counts"] == [0, 16]
    assert state.feedback(0, .256, "background")["class_counts"] == [8, 8]
    with pytest.raises(ValueError, match="retained"):
        state.feedback(0, 10, "speech")
    with pytest.raises(ValueError, match="finite"):
        state.feedback(float('nan'), 1, "speech")
    with pytest.raises(ValueError, match="complete"):
        state.feedback(.001, .002, "speech")
    assert command(state, 'background 0.256')[0]["class_counts"] == [16, 0]


def test_history_and_replay_are_bounded():
    state = VADSession(Reader(), device="cpu")
    for _ in range(32):
        state.push(pcm(10, 60))
    assert len(state.history) == 1875
    with pytest.raises(ValueError, match="retained"):
        state.feedback(0, .5, "speech")
    state.feedback(50, 60, "background")
    assert len(state.replay[0]) == 256
    with pytest.raises(ValueError, match="two seconds"):
        state.push(bytes(64001))


def test_bad_checkpoint_never_changes_active_model(tmp_path):
    state = VADSession(Reader(), device="cpu")
    path = tmp_path / "bad.npz"
    state.save(path)
    with np.load(path, allow_pickle=False) as data:
        values = dict(data)
    metadata = json.loads(str(values["metadata"].item()))
    metadata["profile"]["representation"] = "incompatible"
    values["metadata"] = json.dumps(metadata)
    np.savez(path, **values)
    previous = state.head
    with pytest.raises(ValueError, match="profile"):
        state.load(path)
    assert state.head is previous


def test_audio_import_does_not_import_torch_or_server():
    subprocess.run([sys.executable, "-c",
                    "from venn.audio.session import VADSession; import sys; "
                    "assert 'torch' not in sys.modules; assert 'venn.server' not in sys.modules"], check=True)


def test_wav_rejects_wrong_sample_rate(tmp_path):
    import wave
    path = tmp_path / "wrong.wav"
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
        output.writeframes(bytes(2048))
    with pytest.raises(ValueError, match="16 kHz"):
        open_wav(path)


def test_real_reader_initial_venn_decision_matches_frozen_model():
    path = Path('.venn-audio/models/silero-v6.2.onnx')
    if not path.exists():
        pytest.skip("Run python -m venn.audio download for real-weight validation")
    state = VADSession(SileroReader(path), device="cpu")
    rng = np.random.default_rng(100)
    audio = np.clip(rng.normal(0, 500, FRAME * 20), -32768, 32767).astype('<i2').tobytes()
    events = state.push(audio)
    for event in events:
        assert event["decision"]["probabilities"]["speech"] == pytest.approx(event["baseline_probability"], abs=2e-6)
    assert state.info()["profile"] == PROFILE


def test_nonfinite_weights_and_inputs_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        MLXLinearChoiceHead([[float('nan')], [0]], [0, 0], ["a", "b"])
    head = MLXLinearChoiceHead([[0], [0]], [0, 0], ["a", "b"])
    with pytest.raises(ValueError, match="finite"):
        head.probabilities([[float('nan')]])


@pytest.mark.parametrize("overflow", [False, True])
def test_microphone_callback_handoff_and_overflow_stop(monkeypatch, capsys, overflow):
    import io
    from types import SimpleNamespace
    import sounddevice
    from venn.audio import cli

    state = VADSession(Reader(), device="cpu")
    monkeypatch.setattr(cli, "session", lambda args: state)
    monkeypatch.setattr(sys, "stdin", io.StringIO('quit\n'))

    class Capture:
        def __init__(self, **kwargs):
            assert kwargs['samplerate'] == 16000 and kwargs['blocksize'] == 512
            assert kwargs['channels'] == 1 and kwargs['dtype'] == 'int16'
            self.callback = kwargs['callback']

        def __enter__(self):
            for _ in range(9 if overflow else 1):
                self.callback(pcm(100), FRAME, None, None)
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(sounddevice, 'RawInputStream', Capture)
    # emit's default stream is bound at import, so capture events directly.
    events = []
    monkeypatch.setattr(cli, 'emit', lambda value, **kwargs: events.append(value))
    if overflow:
        with pytest.raises(RuntimeError, match="overflow"):
            cli.microphone(SimpleNamespace(input_device=None))
        assert not any(e['type'] == 'frame' for e in events)
    else:
        cli.microphone(SimpleNamespace(input_device=None))
        assert sum(e['type'] == 'frame' for e in events) == 1
    assert any(e['type'] == 'end' for e in events)


def test_feedback_changes_memory_decision_before_any_training():
    state = VADSession(Reader(), device="cpu")
    first = state.push(pcm(-10000, 16))[-1]
    assert first["decision"]["source"] == "baseline"
    assert first["decision"]["memory"]["abstained"]
    assert first["decision"]["answer"] == "speech"
    weights_before = state.head.arrays()
    state.feedback(0, .512, "background")
    next_frame = state.push(pcm(-10000))[0]
    assert next_frame["decision"]["source"] == "memory"
    assert next_frame["decision"]["answer"] == "background"
    assert next_frame["decision"]["probabilities"] is None  # Neighborhood agreement is not a probability.
    assert next_frame["decision"]["cases"]
    assert next_frame["version"] == 0
    for before, after in zip(weights_before, state.head.arrays()):
        np.testing.assert_array_equal(before, after)


def test_multiple_questions_and_requery_reuse_the_same_audio_vectors(tmp_path):
    class CountingReader(Reader):
        calls = 0

        def encode(self, samples):
            self.calls += 1
            return super().encode(samples)

    reader = CountingReader()
    state = VADSession(reader, device="cpu", include_vectors=True)
    frames = state.push(pcm(-10000, 16))
    state.register_question("sound", ["fan", "keyboard"])
    state.feedback(0, .512, "background")
    state.feedback(0, .512, "keyboard", question="sound")
    answers = state.ask(0, .512)
    assert reader.calls == 16
    assert answers[0]["answers"]["vad"]["answer"] == "background"
    assert answers[0]["answers"]["sound"]["answer"] == "keyboard"
    assert len(frames[0]["state"]["vector"]) == DIMENSIONS
    assert frames[0]["state"]["tag"]["fields"] == ["audio"]
    path = tmp_path / "questions.npz"
    state.save(path)
    restored = VADSession(Reader(), device="cpu", checkpoint=path)
    event = restored.push(pcm(-10000))[0]
    assert event["questions"]["sound"]["answer"] == "keyboard"
    assert event["questions"]["vad"]["answer"] == "background"
    assert {case["id"] for case in event["decision"]["cases"]} <= {f["state"]["id"] for f in frames}


def test_compiled_policy_is_invalidated_by_new_feedback():
    state = VADSession(Reader(), device="cpu", policy="compiled")
    state.push(pcm(-10000, 16))
    state.push(pcm(10000, 16))
    state.feedback(0, .512, "background")
    state.feedback(.512, 1.024, "speech")
    result = state.learn()
    assert result["accepted"] and result["objective"] == "compile_memory_policy"
    assert result["head_revision"] == result["memory_revision"]
    assert state.push(pcm(-10000))[0]["decision"]["source"] == "compiled"
    state.feedback(0, .512, "speech")
    corrected = state.push(pcm(-10000))[0]["decision"]
    assert corrected["source"] == "memory" and corrected["answer"] == "speech"
    assert not corrected["head_current"]


def test_custom_question_abstains_and_feedback_command_targets_it():
    state = VADSession(Reader(), device="cpu")
    state.register_question("sound", ["fan", "keyboard"])
    event = state.push(pcm(-10000, 16))[-1]
    assert event["questions"]["sound"]["answer"] == "none"
    feedback = command(state, 'label sound keyboard 0.512')
    assert feedback[0]["question"] == "sound"
    assert command(state, 'ask sound')[0]["results"][0]["answers"]["sound"]["answer"] == "keyboard"
    assert state.questions["vad"].store.counts() == [0, 0]
    with pytest.raises(ValueError, match="labels"):
        state.register_question("sound", ["different", "labels"])


def test_legacy_checkpoint_migrates_labels_but_not_a_false_compilation_claim(tmp_path):
    state = VADSession(Reader(), device="cpu")
    x, _ = Reader().encode(np.full(FRAME, -.3, np.float32))
    W, b = state.head.arrays()
    path = tmp_path / 'legacy.npz'
    np.savez(path, W=W, b=b, X=x[None], y=np.array([0], np.int32), keys=np.array(['legacy:512']),
             metadata=json.dumps({'schema': 1, 'profile': PROFILE, 'labels': ['background', 'speech'],
                                  'version': 3, 'temperature': 1.0}))
    restored = VADSession(Reader(), device="cpu", checkpoint=path, policy="compiled")
    event = restored.push(pcm(-10000))[0]
    assert restored.version == 3
    assert event["decision"]["source"] == "memory"
    assert event["decision"]["answer"] == "background"
    assert restored.questions["vad"].head_revision is None
    restored.save(path)
    with np.load(path, allow_pickle=False) as saved:
        assert json.loads(str(saved['metadata'].item()))['schema'] == 3


def test_compiled_head_runs_without_querying_the_live_store(tmp_path, monkeypatch):
    state = VADSession(Reader(), device="cpu", policy="compiled")
    state.push(pcm(-10000, 16))
    state.push(pcm(10000, 16))
    state.feedback(0, .512, "background")
    state.feedback(.512, 1.024, "speech")
    assert state.learn()["accepted"]
    path = tmp_path / 'compiled.npz'
    state.save(path)
    restored = VADSession(Reader(), device="cpu", policy="compiled", checkpoint=path)
    monkeypatch.setattr(restored.questions['vad'].store, 'search', lambda *a, **kw: pytest.fail('Live store queried'))
    result = restored.push(pcm(-10000))[0]['decision']
    assert result['source'] == 'compiled'
    assert result['memory'] is None and result['evidence'] is None
    assert result['compilation']['cases'] == 32
    # Unknown vectors use the frozen detector; stale store statistics are never invented.
    unseen = np.zeros(DIMENSIONS, np.float32)
    unseen[5] = 1
    result = restored._answers(unseen, .2)['vad']
    assert result['source'] == 'baseline'
    assert result['compiled']['reason'] == 'outside_compiled_coverage'
