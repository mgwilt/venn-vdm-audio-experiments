"""Microphone-free VAD evaluation with separate adaptation and held-out audio.

The built-in suite is synthetic. Its positive reference selects audible frames
from the clean TTS source before attenuation/noise; quiet frames inside utterances
are ignored. It is an engineering regression test, not human-labeled VAD accuracy.
Custom manifests can supply independent annotations on real recordings.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import wave

import numpy as np

from venn.audio.reader import FRAME, RATE, PROFILE, SileroReader

FIXTURE_VERSION = 1
TRAIN_TEXT = (
    "Please open the window and bring the blue notebook to the table.",
    "We are testing a system that learns to recognize speech during a conversation.",
)
TEST_TEXT = (
    "Tomorrow morning the delivery driver will leave a package beside the door.",
    "Remember to check the weather before you start your afternoon walk.",
)
CONDITIONS = (
    ("clean", 1., None, "Samantha"),
    ("quiet_speech", .03, None, "Samantha"),
    ("noise_10db", 1., 10., "Samantha"),
    ("noise_0db", 1., 0., "Samantha"),
    ("new_voice", 1., None, "Daniel"),
)


def digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read_audio(path):
    with wave.open(str(path), "rb") as file:
        if (file.getnchannels(), file.getsampwidth(), file.getframerate(), file.getcomptype()) != (1, 2, RATE, "NONE"):
            raise ValueError(f"Expected mono 16 kHz PCM16 WAV: {path}")
        if not FRAME <= file.getnframes() <= RATE * 300:
            raise ValueError("Evaluation clips must contain 32 ms to 300 seconds of audio")
        pcm = file.readframes(file.getnframes())
    # Never pad recordings or give models access to annotation arrays.
    return pcm[:len(pcm) // (FRAME * 2) * FRAME * 2]


def write_audio(path, samples):
    if np.max(np.abs(samples), initial=0) >= 1:
        raise ValueError("Synthetic mix clips; reduce the fixture amplitude")
    with wave.open(str(path), "wb") as file:
        file.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        file.writeframes(np.rint(samples * 32767).astype('<i2').tobytes())


def intervals(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [[int(a) * FRAME / RATE, int(b) * FRAME / RATE] for a, b in zip(edges[::2], edges[1::2])]


def synthesize(path, text, voice):
    if not path.exists():
        try:
            subprocess.run(["/usr/bin/say", "-v", voice, "-r", "175", "--file-format=WAVE",
                            "--data-format=LEI16@16000", "-o", str(path), text], check=True, capture_output=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Speech synthesis failed for {voice}: {error.stderr.decode(errors='replace')}") from error
    pcm = read_audio(path)
    values = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768
    peak = float(np.max(np.abs(values)))
    if peak <= 1e-6:
        raise ValueError(f"TTS produced no signal: {voice}")
    return values * (.3 / peak)


def mix_fixture(clips, *, gain, snr, seed):
    """Reference from clean source; no model prediction enters these labels."""
    rng = np.random.default_rng(seed)
    segments, targets = [], []
    for clip in clips:
        segments.extend([np.zeros(RATE * 2 + 768, np.float32), clip])  # 2.048 s gap.
        clean_rms = np.sqrt(np.mean(clip.reshape(-1, FRAME) ** 2, axis=1))
        # Low-energy source frames include pauses and uncertain consonant edges.
        audible = clean_rms >= max(float(clean_rms.max()) * .016, 1e-4)
        reference = np.where(audible, 1, -1).astype(np.int8)
        reference[:2] = reference[-2:] = -1
        targets.extend([np.zeros(64, np.int8), reference])
    segments.append(np.zeros(RATE * 2 + 768, np.float32))
    targets.append(np.zeros(64, np.int8))
    clean = np.concatenate(segments)
    truth = np.concatenate(targets)
    signal = clean * gain
    if snr is not None:
        # Synthetic broad-band fan-like noise, with a stable hum. Includes noise-only gaps.
        raw = rng.normal(size=len(signal)).astype(np.float32)
        noise = np.convolve(raw, np.ones(9) / 9, mode='same')
        noise += .15 * np.sin(2 * np.pi * 120 * np.arange(len(signal)) / RATE)
        speech_rms = np.sqrt(np.mean(signal.reshape(-1, FRAME)[truth == 1] ** 2))
        noise *= speech_rms / (10 ** (snr / 20) * np.sqrt(np.mean(noise ** 2)))
        signal += noise
    # Calibration silence is available during adaptation; it is not a noise label.
    signal = np.r_[np.zeros(64 * FRAME), signal]
    truth = np.r_[np.zeros(64, np.int8), truth]
    return signal.astype(np.float32), truth


def build_suite(directory, seed=17):
    directory.mkdir(parents=True, exist_ok=True)
    sources = {}
    for split, texts in (("train", TRAIN_TEXT), ("test", TEST_TEXT)):
        for voice in ("Samantha", "Daniel") if split == "test" else ("Samantha",):
            for i, text in enumerate(texts):
                path = directory / f"source-{split}-{voice}-{i}.wav"
                sources[split, voice, i] = synthesize(path, text, voice)
    cases = []
    for index, (name, gain, snr, voice) in enumerate(CONDITIONS):
        case = {"name": name, "gain": gain, "snr_db": snr}
        for split, texts in (("train", TRAIN_TEXT), ("test", TEST_TEXT)):
            selected_voice = "Samantha" if split == "train" else voice
            clips = [sources[split, selected_voice, i] for i in range(len(texts))]
            audio, truth = mix_fixture(clips, gain=gain, snr=snr, seed=seed + index * 2 + (split == "test"))
            path = directory / f"{name}-{split}.wav"
            write_audio(path, audio)
            case[split] = {"wav": path.name, "sha256": digest(path), "speech": intervals(truth == 1),
                           "ignore": intervals(truth == -1), "voice": selected_voice}
        cases.append(case)
    manifest = {"schema": 1, "reference_kind": "synthetic_clean_source_energy",
                "limitations": "TTS speech and synthetic noise; quiet source frames ignored; not human VAD ground truth",
                "fixture_version": FIXTURE_VERSION, "seed": seed, "macos": platform.mac_ver()[0],
                "train_text": TRAIN_TEXT, "test_text": TEST_TEXT, "cases": cases}
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    return path


def reference_frames(spec, count):
    """Unannotated regions are background; ignore regions override speech.

    A frame straddling an annotation boundary is ignored rather than assigning
    a partial speech frame a false exact label.
    """
    truth = np.zeros(count, np.int8)
    starts, ends = np.arange(count) * FRAME / RATE, (np.arange(count) + 1) * FRAME / RATE
    for field, label in (("speech", 1), ("ignore", -1)):
        regions = spec.get(field, [])
        if not isinstance(regions, list):
            raise ValueError("Annotations must be lists of [start, end] seconds")
        for region in regions:
            if (not isinstance(region, list) or len(region) != 2
                    or not all(isinstance(t, (int, float)) and np.isfinite(t) for t in region)
                    or not 0 <= region[0] < region[1] <= count * FRAME / RATE + 1e-8):
                raise ValueError("Invalid annotation interval")
        merged = []
        for a, b in sorted(regions):
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(b, merged[-1][1])
            else:
                merged.append([a, b])
        for a, b in merged:
            overlaps = (ends > a + 1e-8) & (starts < b - 1e-8)
            inside = (starts >= a - 1e-8) & (ends <= b + 1e-8)
            truth[overlaps] = -1
            if label == 1:
                truth[inside] = 1
    return truth


def load_suite(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if (not isinstance(manifest, dict) or manifest.get('schema') != 1
            or not isinstance(manifest.get('reference_kind'), str)
            or not isinstance(manifest.get('cases'), list) or not 1 <= len(manifest['cases']) <= 32):
        raise ValueError("Expected schema 1 manifest with reference_kind and 1..32 cases")
    cases, names = [], set()
    for row in manifest['cases']:
        if not isinstance(row, dict):
            raise ValueError("Cases must be objects with name, train, and test")
        name = row.get('name')
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Case names must be unique strings")
        names.add(name)
        case = {'name': name}
        for split in ('train', 'test'):
            spec = row.get(split)
            if not isinstance(spec, dict) or not isinstance(spec.get('wav'), str):
                raise ValueError(f"{split} must specify a WAV path")
            if split == 'test' and 'speech' not in spec:
                raise ValueError("Test audio must specify speech intervals (an empty list means no speech)")
            audio_path = (path.parent / spec['wav']).resolve()
            pcm = read_audio(audio_path)
            sha = digest(audio_path)
            if spec.get('sha256', sha) != sha:
                raise ValueError(f"Audio checksum mismatch: {audio_path}")
            case[split] = {'pcm': pcm, 'truth': reference_frames(spec, len(pcm) // (FRAME * 2)),
                           'path': str(audio_path), 'sha256': sha}
        if case['train']['sha256'] == case['test']['sha256']:
            raise ValueError("Training and test recordings must differ")
        cases.append(case)
    return manifest, cases


def classification_metrics(truth, predicted):
    truth, predicted = np.asarray(truth), np.asarray(predicted, bool)
    if truth.shape != predicted.shape or not np.isin(truth, [-1, 0, 1]).all():
        raise ValueError("Metrics require matching predictions and {-1, 0, 1} references")
    tp = int(np.sum((truth == 1) & predicted))
    fn = int(np.sum((truth == 1) & ~predicted))
    fp = int(np.sum((truth == 0) & predicted))
    tn = int(np.sum((truth == 0) & ~predicted))
    ratio = lambda a, b: a / b if b else None
    recall, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
    return {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn, 'ignored_frames': int(np.sum(truth < 0)),
            'precision': ratio(tp, tp + fp), 'recall': recall, 'f1': ratio(2 * tp, 2 * tp + fp + fn),
            'false_positive_rate': ratio(fp, fp + tn), 'miss_rate': ratio(fn, fn + tp),
            'balanced_accuracy': (recall + specificity) / 2 if recall is not None and specificity is not None else None,
            'false_positive_seconds': fp * FRAME / RATE, 'missed_speech_seconds': fn * FRAME / RATE}


def timing_metrics(values):
    return {'p50_ms': float(np.median(values)), 'p95_ms': float(np.percentile(values, 95)),
            'max_ms': float(max(values)), 'frames_over_32ms': int(np.sum(np.asarray(values) > 32)),
            'compute_realtime_factor': float(np.mean(values) / (FRAME / RATE * 1000))}


def adapt(pcm, state, teacher):
    """Causal offline adaptation. No evaluation reference is accepted here."""
    events = []
    for start in range(0, len(pcm), FRAME * 2):
        frame = pcm[start:start + FRAME * 2]
        state.push(frame)
        events.extend(teacher.poll(state))
        if teacher.feed(frame, state):
            events.extend(teacher.drain(state))
    events.extend(teacher.drain(state))
    teacher.flush(state)
    events.extend(teacher.drain(state))
    events.extend(teacher.compile(state, force=True))
    errors = [event for event in events if event['type'] == 'teacher_error']
    if errors:
        raise RuntimeError(f"Teacher failed during evaluation: {errors[0]['message']}")
    return events


def score_clip(pcm, *, model_dir, device, checkpoint=None, mode='baseline'):
    from venn.audio.session import VADSession, Boundaries

    reader = SileroReader(Path(model_dir) / 'silero-v6.2.onnx')
    state = VADSession(reader, device=device, checkpoint=checkpoint, policy=mode) if mode != 'baseline' else None
    boundaries = Boundaries()
    scores, gates, timings = [], [], []
    sources = Counter()
    for start in range(0, len(pcm), FRAME * 2):
        frame = pcm[start:start + FRAME * 2]
        started = perf_counter()
        if state is not None:
            event = state.push(frame)[0]
            score = event['decision']['scores']['speech']
            active, source = event['speaking'], event['decision']['source']
        else:
            _, score = reader.encode(np.frombuffer(frame, '<i2').astype(np.float32) / 32768)
            boundaries.push(score, (start + len(frame)) // 2)
            active, source = boundaries.active, 'baseline'
        timings.append((perf_counter() - started) * 1000)
        scores.append(score)
        gates.append(active)
        sources[source] += 1
    # Evaluation never calls feedback/learn/save. Each mode starts with reset reader state.
    return np.asarray(scores), np.asarray(gates), timings, dict(sources)


def run(args):
    from dataclasses import asdict
    from venn.audio.session import VADSession, BoundaryConfig
    from venn.audio.teacher import AutoTeacher, TeacherProcess, MODEL, REVISION

    output = Path(args.output or Path('.venn-audio/evaluations') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if (output / 'report.json').exists():
        raise ValueError("This output already has a report; choose a new --output directory")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else build_suite(output / 'fixtures', args.seed)
    manifest, cases = load_suite(manifest_path)
    started = perf_counter()
    worker = TeacherProcess(args.model_dir)
    report = {'schema': 1, 'manifest': str(manifest_path.resolve()), 'reference_kind': manifest['reference_kind'],
              'limitations': manifest.get('limitations', 'Reference quality depends on supplied annotations'),
              'protocol': 'Separate training audio; teacher disabled on held-out test audio; fixed thresholds; reset readers per mode',
              'timing_scope': 'Per-frame inference including first frame; excludes model loading, teacher, capture, and audio playback',
              'gate_scoring': 'Live speaking flag at frame end; no retrospective timestamp pre-roll credit',
              'boundaries': asdict(BoundaryConfig()), 'frame_threshold': .5,
              'reader': PROFILE, 'teacher': {'model': MODEL, 'revision': REVISION},
              'teacher_startup_ms': (perf_counter() - started) * 1000,
              'platform': platform.platform(), 'cases': []}
    try:
        for index, case in enumerate(cases):
            print(f"Evaluating {case['name']} ({index + 1}/{len(cases)})…", file=sys.stderr, flush=True)
            folder = output / f'case-{index:02d}'
            folder.mkdir(exist_ok=True)
            checkpoint = folder / 'adapted.npz'
            state = VADSession(SileroReader(Path(args.model_dir) / 'silero-v6.2.onnx'), device=args.device)
            teacher = AutoTeacher(worker)
            begin = perf_counter()
            updates = adapt(case['train']['pcm'], state, teacher)
            state.save(checkpoint)
            checkpoint_hash = digest(checkpoint)
            q = state.questions['vad']
            row = {'name': case['name'], 'device': state.device,
                   'train': {k: v for k, v in case['train'].items() if k not in ('pcm', 'truth')},
                   'test': {k: v for k, v in case['test'].items() if k not in ('pcm', 'truth')},
                   'adaptation_seconds': (perf_counter() - begin), 'counts': dict(zip(q.store.options, q.store.counts())),
                   'compiled_head_current': q.head_revision == q.store.revision and q.compilation is not None,
                   'teacher_windows': sum(e['type'] == 'teacher_result' for e in updates),
                   'teacher_processing_ms': [e['processing_ms'] for e in updates if e['type'] == 'teacher_result'],
                   'modes': {}}
            (folder / 'teacher-events.json').write_text(json.dumps(updates, indent=2, allow_nan=False) + '\n')
            for mode in ('baseline', 'memory', 'compiled'):
                scores, gates, timings, sources = score_clip(case['test']['pcm'], model_dir=args.model_dir,
                    device=state.device, checkpoint=checkpoint if mode != 'baseline' else None, mode=mode)
                truth = case['test']['truth']
                row['modes'][mode] = {'frame': classification_metrics(truth, scores >= .5),
                                      'gate': classification_metrics(truth, gates),
                                      'timing': timing_metrics(timings), 'sources': sources}
                np.savez(folder / f'{mode}-predictions.npz', scores=scores, speaking=gates, reference=truth,
                         processing_ms=timings)
            if digest(checkpoint) != checkpoint_hash:
                raise RuntimeError("Held-out evaluation modified its training checkpoint")
            report['cases'].append(row)
    finally:
        worker.close()
    report['summary'] = {}
    for mode in ('baseline', 'memory', 'compiled'):
        report['summary'][mode] = {}
        for metric in ('f1', 'balanced_accuracy', 'false_positive_rate', 'miss_rate'):
            values = [case['modes'][mode]['gate'][metric] for case in report['cases']]
            values = [value for value in values if value is not None]
            report['summary'][mode][f'macro_{metric}'] = float(np.mean(values)) if values else None
    destination = output / 'report.json'
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'report': str(destination.resolve()), 'reference_kind': report['reference_kind'],
                      'summary': report['summary']}, allow_nan=False))
    return report
