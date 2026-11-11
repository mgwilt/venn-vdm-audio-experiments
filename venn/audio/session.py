"""Single-owner streaming Venn classifier, feedback replay, and atomic checkpoints."""
from collections import OrderedDict, deque
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from time import perf_counter
from uuid import uuid4

import numpy as np

from venn.audio.reader import DIMENSIONS, FRAME, PROFILE, RATE
from venn.heads.mlx_choice import MLXLinearChoiceHead

LABELS = ("background", "speech")
HISTORY_FRAMES = 1875  # 60 seconds; only feature vectors, never raw audio.
REPLAY_PER_CLASS = 256


@dataclass(frozen=True)
class BoundaryConfig:
    start_threshold: float = 0.6
    stop_threshold: float = 0.35
    start_frames: int = 2
    stop_frames: int = 5
    pre_roll_ms: int = 160

    def __post_init__(self):
        if not 0 <= self.stop_threshold < self.start_threshold <= 1:
            raise ValueError("Require 0 <= stop threshold < start threshold <= 1")
        if any(type(n) is not int or n < 1 for n in (self.start_frames, self.stop_frames)):
            raise ValueError("Boundary durations must be positive frame counts")
        if not 0 <= self.pre_roll_ms <= 1000:
            raise ValueError("Pre-roll must be between 0 and 1000 ms")


class Boundaries:
    def __init__(self, config=None):
        self.config = config or BoundaryConfig()
        self.active = False
        self.run = 0

    def push(self, probability, end_sample):
        cfg = self.config
        if not self.active:
            self.run = self.run + 1 if probability >= cfg.start_threshold else 0
            if self.run >= cfg.start_frames:
                self.active, self.run = True, 0
                start = max(0, end_sample - cfg.start_frames * FRAME - cfg.pre_roll_ms * RATE // 1000)
                return {"type": "speech_start", "sample": start, "seconds": start / RATE}
        else:
            self.run = self.run + 1 if probability < cfg.stop_threshold else 0
            if self.run >= cfg.stop_frames:
                self.active, self.run = False, 0
                stop = end_sample - cfg.stop_frames * FRAME
                return {"type": "speech_end", "sample": stop, "seconds": stop / RATE}
        return None


class VADSession:
    """Feed PCM16 mono/16 kHz bytes; call all methods on one processing thread.

    Feedback labels completed frames. Predictions are emitted before learning.
    The frozen reader provides features; the final decision always comes from
    Venn's MLX linear Choice head. No model updates occur without explicit labels.
    """
    def __init__(self, reader, *, device="auto", checkpoint=None, boundaries=None):
        self.reader = reader
        W = np.zeros((2, DIMENSIONS), np.float32)
        W[1, 128] = 8  # Initial Choice logits exactly recover the frozen speech estimate.
        self.anchor = (W, np.zeros(2, np.float32))
        self.head = MLXLinearChoiceHead(*self.anchor, LABELS, device=device)
        self.session_id = uuid4().hex
        self.pending = bytearray()
        self.samples = 0
        self.history = deque(maxlen=HISTORY_FRAMES)
        self.replay = [OrderedDict(), OrderedDict()]
        self.version = 0
        self.feedback_frames = 0
        self.boundaries = Boundaries(boundaries)
        self.checkpoint = Path(checkpoint) if checkpoint else None
        if self.checkpoint and self.checkpoint.exists():
            self.load(self.checkpoint)

    def info(self):
        return {"type": "session", "session_id": self.session_id, "profile": PROFILE,
                "classifier": "venn-linear-choice-mlx", "labels": LABELS,
                "device": self.head.device, "device_median_ms": self.head.device_timings,
                "version": self.version, "learning": "explicit-feedback",
                "calibrated": False, "frame_ms": FRAME * 1000 / RATE}

    def push(self, pcm):
        if not isinstance(pcm, bytes) or len(pcm) > RATE * 2 * 2:
            raise ValueError("Supply at most two seconds of PCM16 bytes per push")
        self.pending.extend(pcm)
        events = []
        while len(self.pending) >= FRAME * 2:
            started = perf_counter()
            audio = np.frombuffer(self.pending[:FRAME * 2], dtype="<i2").astype(np.float32) / 32768
            del self.pending[:FRAME * 2]
            features, baseline = self.reader.encode(audio)
            probabilities = self.head.probabilities(features[None])[0]
            self.samples += FRAME
            self.history.append((self.samples, features.copy()))
            p = float(probabilities[1])
            boundary = self.boundaries.push(p, self.samples)
            events.append({
                "type": "frame", "session_id": self.session_id,
                "start_sample": self.samples - FRAME, "end_sample": self.samples,
                "start": (self.samples - FRAME) / RATE, "end": self.samples / RATE,
                "decision": {"type": "Choice", "answer": LABELS[int(np.argmax(probabilities))],
                             "probabilities": {"background": float(probabilities[0]), "speech": p, "none": 0.0},
                             "abstained": False, "calibrated": False},
                "baseline_probability": baseline, "speaking": self.boundaries.active,
                "boundary": boundary, "version": self.version,
                "processing_ms": (perf_counter() - started) * 1000,
            })
        return events

    def feedback(self, start, end, label):
        if label not in LABELS:
            raise ValueError("Feedback label must be 'speech' or 'background'")
        if not all(isinstance(t, (int, float)) and math.isfinite(t) for t in (start, end)) or not 0 <= start < end:
            raise ValueError("Feedback requires finite 0 <= start < end seconds")
        if not self.history:
            raise ValueError("No completed audio frames to label")
        first_sample = self.history[0][0] - FRAME
        if start * RATE < first_sample - 1e-5 or end * RATE > self.samples + 1e-5:
            raise ValueError("Feedback must lie within the retained, completed 60 seconds")
        selected = [(sample, x) for sample, x in self.history
                    if sample - FRAME >= start * RATE - 1e-5 and sample <= end * RATE + 1e-5]
        if not selected:
            raise ValueError("Feedback interval must contain at least one complete 32 ms frame")
        target = LABELS.index(label)
        for sample, x in selected:
            key = f"{self.session_id}:{sample}"
            for memory in self.replay:
                memory.pop(key, None)  # Corrections replace previous labels for a frame.
            self.replay[target][key] = x.copy()
            while len(self.replay[target]) > REPLAY_PER_CLASS:
                self.replay[target].popitem(last=False)
        self.feedback_frames += len(selected)
        return {"type": "feedback", "label": label, "frames": len(selected),
                "class_counts": [len(memory) for memory in self.replay]}

    def learn(self):
        started = perf_counter()
        counts = [len(memory) for memory in self.replay]
        if min(counts) < 8:
            return {"type": "learning", "accepted": False, "reason": "Need at least 8 labeled frames per class",
                    "class_counts": counts, "version": self.version}
        X = np.array([x for memory in self.replay for x in memory.values()], np.float32)
        y = np.repeat(np.arange(2), counts)
        result = self.head.update(X, y, anchor=self.anchor)
        if result["accepted"]:
            self.version += 1
        if self.checkpoint:
            self.save(self.checkpoint)
        return {"type": "learning", **result, "version": self.version,
                "class_counts": counts, "processing_ms": (perf_counter() - started) * 1000}

    def reset_learning(self):
        """Return to the frozen baseline without disrupting the audio timeline."""
        self.head = MLXLinearChoiceHead(*self.anchor, LABELS, device=self.head.device)
        self.replay = [OrderedDict(), OrderedDict()]
        self.version = 0
        if self.checkpoint:
            self.save(self.checkpoint)

    def finish(self):
        """Do not pad incomplete audio or interpret stream closure as natural silence."""
        trailing = len(self.pending)
        self.pending.clear()
        return {"type": "end", "seconds": self.samples / RATE, "discarded_tail_bytes": trailing,
                "speech_open": self.boundaries.active, "version": self.version}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        W, b = self.head.arrays()
        rows = [(key, x, label) for label, memory in enumerate(self.replay) for key, x in memory.items()]
        metadata = {"schema": 1, "profile": PROFILE, "labels": LABELS, "version": self.version,
                    "temperature": self.head.temperature}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
                temporary = Path(file.name)
                np.savez(file, W=W, b=b, metadata=json.dumps(metadata),
                         X=np.array([row[1] for row in rows], np.float32).reshape(-1, DIMENSIONS),
                         y=np.array([row[2] for row in rows], np.int32),
                         keys=np.array([row[0] for row in rows], dtype=str))
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)

    def load(self, path):
        # Validate everything before changing the active model or memory.
        if Path(path).stat().st_size > 2_000_000:
            raise ValueError("Checkpoint exceeds size limit")
        with np.load(path, allow_pickle=False) as saved:
            meta = json.loads(str(saved["metadata"].item()))
            if meta.get("schema") != 1 or meta.get("profile") != PROFILE or meta.get("labels") != list(LABELS):
                raise ValueError("Checkpoint audio profile or labels do not match")
            if type(meta.get("version")) is not int or meta["version"] < 0:
                raise ValueError("Invalid checkpoint version")
            X, y, keys = saved["X"], saved["y"], saved["keys"]
            if (X.ndim != 2 or X.shape[1] != DIMENSIONS or len(X) > 2 * REPLAY_PER_CLASS
                    or y.shape != (len(X),) or keys.shape != (len(X),)
                    or not np.isfinite(X).all() or not np.issubdtype(y.dtype, np.integer)
                    or not np.isin(y, [0, 1]).all() or len(set(keys.tolist())) != len(keys)
                    or keys.dtype.kind != 'U' or np.max(np.bincount(y, minlength=2)) > REPLAY_PER_CLASS):
                raise ValueError("Invalid checkpoint replay data")
            if saved["W"].shape != (2, DIMENSIONS) or meta.get("temperature") != 1.0:
                raise ValueError("Invalid checkpoint classifier")
            head = MLXLinearChoiceHead(saved["W"], saved["b"], LABELS, device=self.head.device)
            replay = [OrderedDict(), OrderedDict()]
            for key, x, label in zip(keys, X, y):
                replay[int(label)][str(key)] = x.copy()
        self.head, self.replay, self.version = head, replay, meta["version"]
