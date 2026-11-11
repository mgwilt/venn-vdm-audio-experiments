"""Conservative automatic supervision, provenance, and nonblocking handoff."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from venn.audio.reader import FRAME, RATE, DIMENSIONS
from venn.audio.teacher import AutoTeacher, label_window, SOURCE, SILENCE_SOURCE, WINDOW_FRAMES


def pcm(value=1000, frames=WINDOW_FRAMES):
    return np.full(FRAME * frames, value, dtype='<i2').tobytes()


def token(start=.5, end=1.3, confidence=.99, text="word"):
    return dict(start=start, end=end, confidence=confidence, text=text)


def test_empty_transcript_is_not_background():
    assert label_window(pcm(), []) == []


def test_speech_uses_only_confident_token_interiors_with_signal():
    spans = label_window(pcm(), [token()])
    assert len(spans) == 1 and spans[0]['label'] == 'speech'
    assert spans[0]['start'] >= .532 and spans[0]['end'] <= 1.268
    assert spans[0]['source'] == SOURCE
    assert label_window(pcm(), [token(confidence=.7)]) == []
    assert label_window(pcm(), [token(text='...')]) == []
    assert not any(s['label'] == 'speech' for s in label_window(pcm(0), [token()]))
    assert label_window(pcm(32767), [token()]) == []


def test_sustained_near_silence_is_independent_negative_evidence():
    spans = label_window(pcm(0), [])
    assert spans == [{'start': .16, 'end': 3.936, 'label': 'background',
                      'source': SILENCE_SOURCE, 'teacher_score': None}]
    # Even uncertain or punctuation tokens protect neighboring audio from negatives.
    spans = label_window(pcm(0), [token(confidence=.1)])
    assert all(s['end'] <= .34 or s['start'] >= 1.46 for s in spans)
    assert label_window(pcm(0, frames=16), []) == []  # Short quiet runs are skipped.


@pytest.mark.parametrize('bad', [token(start=-1), token(end=20), token(confidence=float('nan')),
                                 token(confidence=2), token(start=2, end=1)])
def test_bad_teacher_output_cannot_become_labels(bad):
    with pytest.raises(ValueError, match='token'):
        label_window(pcm(), [bad])


class Worker:
    busy = False
    failed = False

    def __init__(self):
        self.jobs = []
        self.result = None

    def submit(self, job):
        if self.busy:
            return False
        self.jobs.append(job)
        self.busy = True
        return True

    def complete(self, spans=None):
        self.result = {k: v for k, v in self.jobs[-1].items() if k != 'pcm'}
        self.result.update(type='teacher_result', spans=spans or [])

    def poll(self):
        if self.result is None:
            return None
        result, self.result, self.busy = self.result, None, False
        return result

    def close(self):
        self.busy = False


def test_windows_overlap_and_backpressure_never_blocks_or_grows_audio_memory():
    worker = Worker()
    teacher = AutoTeacher(worker)
    state = SimpleNamespace(session_id='s', learning_epoch=0, samples=0)
    for _ in range(1000):
        state.samples += FRAME
        teacher.feed(pcm(frames=1), state)
        assert len(teacher.buffer) < WINDOW_FRAMES * FRAME * 2
    assert len(worker.jobs) == 1 and teacher.skipped_windows > 0
    assert worker.jobs[0]['start_sample'] == 0
    assert worker.jobs[0]['end_sample'] == WINDOW_FRAMES * FRAME
    worker.busy = False
    for _ in range(64):
        state.samples += FRAME
        teacher.feed(pcm(frames=1), state)
    assert len(worker.jobs) == 2
    assert worker.jobs[-1]['start_sample'] > worker.jobs[0]['end_sample']


@pytest.fixture
def state():
    pytest.importorskip('mlx.core', exc_type=ImportError)
    from venn.audio.session import VADSession

    class Reader:
        def encode(self, samples):
            vector = np.zeros(DIMENSIONS, np.float32)
            vector[0], vector[128] = np.mean(samples), .1
            return vector, .1  # Deliberately rejects speech: never gate teacher windows on this.

    return VADSession(Reader(), device='cpu')


def feed_window(state, teacher):
    for _ in range(WINDOW_FRAMES):
        audio = pcm(frames=1)
        state.push(audio)
        teacher.feed(audio, state)


def speech_span():
    return {'start': .512, 'end': 1.024, 'label': 'speech', 'source': SOURCE}


def test_delayed_teacher_teaches_even_when_current_vad_rejects_speech(state):
    teacher = AutoTeacher(Worker())
    feed_window(state, teacher)
    assert state.ask(.512, 1.024)[0]['answers']['vad']['answer'] == 'background'
    teacher.worker.complete([speech_span()])
    updates = teacher.poll(state)
    assert updates[0]['changed_frames'] == 16
    assert state.ask(.512, 1.024)[0]['answers']['vad']['answer'] == 'speech'
    assert all(c['source'] == SOURCE for c in state.questions['vad'].store.cases.values())
    assert state.samples == WINDOW_FRAMES * FRAME  # No resampling or timeline shift.


def test_manual_feedback_wins_over_pending_teacher_and_survives_checkpoint(state, tmp_path):
    teacher = AutoTeacher(Worker())
    feed_window(state, teacher)
    state.feedback(.512, 1.024, 'background')
    teacher.worker.complete([speech_span()])
    assert teacher.poll(state)[0]['changed_frames'] == 0
    assert state.questions['vad'].store.counts() == [16, 0]
    state.feedback(1.024, 1.536, 'speech', source=SOURCE)
    path = tmp_path / 'teacher.npz'
    state.save(path)
    state.load(path)
    assert {c['source'] for c in state.questions['vad'].store.cases.values()} == {'manual', SOURCE}
    # Automatic evidence can be promoted to a human correction even with the same label.
    assert state.feedback(1.024, 1.536, 'speech')['changed_frames'] == 16
    assert all(c['source'] == 'manual' for c in state.questions['vad'].store.cases.values())


def test_reset_discards_pending_results_and_starts_new_audio_window(state):
    teacher = AutoTeacher(Worker())
    feed_window(state, teacher)
    state.reset_learning()
    teacher.worker.complete([speech_span()])
    assert teacher.poll(state) == [{'type': 'teacher_discarded', 'reason': 'session_or_reset_changed'}]
    assert not state.questions['vad'].store.cases
    feed_window(state, teacher)
    assert teacher.worker.jobs[-1]['start_sample'] == WINDOW_FRAMES * FRAME
    assert teacher.worker.jobs[-1]['epoch'] == 1


def test_expired_teacher_evidence_is_not_shifted_into_new_audio(state):
    teacher = AutoTeacher(Worker())
    feed_window(state, teacher)
    for _ in range(32):
        state.push(pcm(frames=60))
    teacher.worker.complete([speech_span()])
    assert teacher.poll(state)[0]['changed_frames'] == 0
    assert not state.questions['vad'].store.cases


def test_teacher_error_does_not_stop_vad(state):
    teacher = AutoTeacher(Worker())
    teacher.worker.result = {'type': 'teacher_error', 'message': 'test failure'}
    assert teacher.poll(state)[0]['type'] == 'teacher_error'
    assert state.push(pcm(frames=1))[0]['decision']['source'] == 'baseline'


def test_finalize_compiles_new_memory_without_waiting_for_periodic_deadline(state):
    teacher = AutoTeacher(Worker())
    feed_window(state, teacher)
    teacher.worker.complete([speech_span()])
    teacher.poll(state)
    state.feedback(1.536, 2.048, 'background', source=SILENCE_SOURCE)
    assert teacher.compile(state) == []
    assert teacher.compile(state, force=True)[0]['type'] == 'learning'
    assert teacher.compile(state, force=True) == []


def test_teacher_drain_has_a_deadline(state):
    teacher = AutoTeacher(Worker())
    feed_window(state, teacher)
    with pytest.raises(RuntimeError, match='timed out'):
        teacher.drain(state, timeout=0)


def test_schema_two_memory_migrates_as_manual_evidence(state, tmp_path):
    state.push(pcm(frames=16))
    state.feedback(0, .512, 'speech')
    path = tmp_path / 'old.npz'
    state.save(path)
    with np.load(path, allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in saved.files if not k.endswith('_sources')}
    meta = json.loads(str(arrays['metadata'].item()))
    meta['schema'] = 2
    arrays['metadata'] = json.dumps(meta)
    np.savez(path, **arrays)
    state.load(path)
    assert all(c['source'] == 'manual' for c in state.questions['vad'].store.cases.values())


def test_automatic_cases_do_not_evict_manual_cases(state):
    for _ in range(5):
        state.push(pcm(frames=60))
    state.feedback(0, 8.192, 'background')
    assert state.feedback(8.192, 9.6, 'background', source=SILENCE_SOURCE)['changed_frames'] == 0
    assert all(c['source'] == 'manual' for c in state.questions['vad'].store.cases.values())


def test_file_cli_teaches_without_manual_annotations(state, monkeypatch, tmp_path):
    import wave
    from venn.audio import cli

    class CompletingWorker(Worker):
        def submit(self, job):
            accepted = super().submit(job)
            if accepted:
                self.complete([speech_span()])
            return accepted

    path = tmp_path / 'input.wav'
    with wave.open(str(path), 'wb') as output:
        output.setparams((1, 2, RATE, 0, 'NONE', 'not compressed'))
        output.writeframes(pcm(frames=WINDOW_FRAMES))
    monkeypatch.setattr(cli, 'session', lambda args: state)
    monkeypatch.setattr(cli, 'automatic_teacher', lambda args: AutoTeacher(CompletingWorker()))
    events = []
    monkeypatch.setattr(cli, 'emit', lambda value, **kw: events.append(value))
    cli.replay(SimpleNamespace(wav=path, labels=None, realtime=False))
    assert any(e['type'] == 'feedback' and e['source'] == SOURCE for e in events)
    assert len([e for e in events if e['type'] == 'frame']) == WINDOW_FRAMES
    assert state.questions['vad'].store.counts() == [0, 16]


@pytest.mark.parametrize('mode,extra', [('mic', []), ('file', ['recording.wav'])])
def test_automatic_teacher_is_the_cli_default(monkeypatch, mode, extra):
    from venn.audio import cli
    seen = []
    monkeypatch.setattr(cli, 'microphone' if mode == 'mic' else 'replay', lambda args: seen.append(args.teacher))
    cli.main([mode, *extra])
    cli.main([mode, *extra, '--teacher', 'none'])
    assert seen == ['parakeet', 'none']
