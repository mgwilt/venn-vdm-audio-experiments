"""Independent benchmark references, metric arithmetic, and held-out isolation."""
import json

import numpy as np
import pytest

from venn.audio import evaluate
from venn.audio.reader import FRAME


def test_metrics_exclude_uncertain_frames_and_count_errors():
    metrics = evaluate.classification_metrics([1, 1, 0, 0, -1], [1, 0, 1, 0, 1])
    assert [metrics[k] for k in ('tp', 'fn', 'fp', 'tn')] == [1, 1, 1, 1]
    assert metrics['f1'] == metrics['balanced_accuracy'] == .5
    assert metrics['false_positive_seconds'] == metrics['missed_speech_seconds'] == .032
    assert metrics['ignored_frames'] == 1
    empty = evaluate.classification_metrics([-1], [1])
    assert empty['f1'] is empty['balanced_accuracy'] is None
    with pytest.raises(ValueError):
        evaluate.classification_metrics([1, 0], [1])


def test_partial_boundaries_are_ignored_and_overlapping_intervals_form_a_union():
    spec = {'speech': [[.04, .10], [.08, .128]], 'ignore': [[.09, .095]]}
    assert evaluate.reference_frames(spec, 5).tolist() == [0, -1, -1, 1, 0]
    assert evaluate.reference_frames({'speech': [[.032, .096]]}, 4).tolist() == [0, 1, 1, 0]


@pytest.mark.parametrize('regions', [[[0, 1]], [[.1, .05]], [[0, float('nan')]], ['bad']])
def test_invalid_annotations_fail(regions):
    with pytest.raises(ValueError, match='interval'):
        evaluate.reference_frames({'speech': regions}, 4)


def test_noise_and_attenuation_do_not_change_reference_labels():
    clip = np.full(16 * FRAME, .1, np.float32)
    clip[6 * FRAME:8 * FRAME] = 0  # A pause inside an utterance is uncertain.
    clean, reference = evaluate.mix_fixture([clip], gain=1., snr=None, seed=1)
    noisy, noisy_reference = evaluate.mix_fixture([clip], gain=.03, snr=0, seed=1)
    repeated, _ = evaluate.mix_fixture([clip], gain=.03, snr=0, seed=1)
    np.testing.assert_array_equal(reference, noisy_reference)
    np.testing.assert_array_equal(noisy, repeated)
    assert np.all(reference[128 + 6:128 + 8] == -1)
    assert np.any(noisy.reshape(-1, FRAME)[reference == 0] != 0)
    assert np.all(clean.reshape(-1, FRAME)[reference == 0] == 0)


def make_manifest(tmp_path):
    evaluate.write_audio(tmp_path / 'train.wav', np.zeros(FRAME * 4))
    evaluate.write_audio(tmp_path / 'test.wav', np.full(FRAME * 4, .1))
    manifest = {'schema': 1, 'reference_kind': 'test_annotations', 'cases': [
        {'name': 'case', 'train': {'wav': 'train.wav'},
         'test': {'wav': 'test.wav', 'speech': [[.032, .096]]}}]}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_manifest_rejects_train_test_leakage_and_corrupt_audio(tmp_path):
    path, manifest = make_manifest(tmp_path)
    _, cases = evaluate.load_suite(path)
    assert cases[0]['test']['truth'].tolist() == [0, 1, 1, 0]
    test = manifest['cases'][0]['test']
    test['sha256'] = 'incorrect'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='checksum'):
        evaluate.load_suite(path)
    del test['sha256']
    test['wav'] = 'train.wav'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='must differ'):
        evaluate.load_suite(path)


def test_missing_test_annotations_cannot_silently_score_everything_as_background(tmp_path):
    path, manifest = make_manifest(tmp_path)
    del manifest['cases'][0]['test']['speech']
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='speech intervals'):
        evaluate.load_suite(path)


def test_evaluation_cannot_learn_or_save_and_uses_identical_audio_per_mode(monkeypatch, tmp_path):
    pytest.importorskip('mlx.core', exc_type=ImportError)
    from venn.audio import session
    checkpoint = tmp_path / 'fixed.npz'
    checkpoint.write_bytes(b'fixed training state')
    encoded, sessions = [], []

    class Reader:
        def __init__(self, path):
            encoded.append([])
            self.seen = encoded[-1]

        def encode(self, samples):
            self.seen.append(samples.copy())
            return None, .9

    class FixedSession:
        def __init__(self, reader, *, device, checkpoint, policy):
            assert checkpoint.read_bytes() == b'fixed training state'
            self.reader, self.policy = reader, policy
            sessions.append(self)

        def push(self, frame):
            _, score = self.reader.encode(np.frombuffer(frame, '<i2').astype(np.float32) / 32768)
            return [{'decision': {'scores': {'speech': score}, 'source': self.policy}, 'speaking': True}]

        def feedback(self, *args, **kwargs):
            pytest.fail('Test audio must never provide feedback')

        learn = save = feedback

    monkeypatch.setattr(evaluate, 'SileroReader', Reader)
    monkeypatch.setattr(session, 'VADSession', FixedSession)
    pcm = np.arange(FRAME * 3, dtype='<i2').tobytes()
    for mode in ('baseline', 'memory', 'compiled'):
        scores, gates, timings, sources = evaluate.score_clip(
            pcm, model_dir=tmp_path, device='cpu', checkpoint=checkpoint, mode=mode)
        assert len(scores) == len(gates) == len(timings) == 3
        assert sources == {mode: 3}
    assert len(sessions) == 2
    np.testing.assert_array_equal(encoded[0], encoded[1])
    np.testing.assert_array_equal(encoded[1], encoded[2])
    assert checkpoint.read_bytes() == b'fixed training state'


def test_evaluate_cli_never_opens_microphone(monkeypatch):
    from venn.audio import cli
    seen = []
    monkeypatch.setattr(evaluate, 'run', lambda args: seen.append(args))
    monkeypatch.setattr(cli, 'microphone', lambda args: pytest.fail('No microphone allowed'))
    cli.main(['evaluate', '--seed', '23'])
    assert seen[0].seed == 23 and seen[0].manifest is None
