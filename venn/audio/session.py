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

from venn.audio.reader import AUDIO_TAG, DIMENSIONS, FRAME, PROFILE, RATE
from venn.audio.memory import AudioQuestion, MAX_PER_LABEL, unit
from venn.heads.mlx_choice import MLXLinearChoiceHead

LABELS = ("background", "speech")
HISTORY_FRAMES = 1875  # 60 seconds; only feature vectors, never raw audio.
REPLAY_PER_CLASS = MAX_PER_LABEL


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
    """One causal reader, reusable tagged states, named Venn questions and memory.

    All methods have one processing-thread owner. Feedback immediately changes
    memory answers; learn() compiles that memory policy into an MLX Choice head.
    The default policy consults memory. Compiled mode uses a current head and a
    frozen coverage reference without querying the store. Only VAD has a fallback.
    """
    def __init__(self, reader, *, device="auto", checkpoint=None, boundaries=None,
                 policy="memory", include_vectors=False):
        if policy not in ("memory", "compiled"):
            raise ValueError("Policy must be memory or compiled")
        self.reader, self.policy, self.include_vectors = reader, policy, include_vectors
        W = np.zeros((2, DIMENSIONS), np.float32)
        W[1, 128] = 8
        self.anchor = (W, np.zeros(2, np.float32))
        initial = MLXLinearChoiceHead(*self.anchor, LABELS, device=device)
        self.device = initial.device
        self.device_timings = initial.device_timings
        self.questions = {}
        vad = self.register_question("vad", LABELS)
        vad.head, vad.anchor = initial, self.anchor
        self.session_id = uuid4().hex
        self.pending = bytearray()
        self.samples = 0
        self.history = deque(maxlen=HISTORY_FRAMES)
        self.version = 0
        self.learning_epoch = 0
        self.manual_frames = {}
        self.boundaries = Boundaries(boundaries)
        self.checkpoint = Path(checkpoint) if checkpoint else None
        if self.checkpoint and self.checkpoint.exists():
            self.load(self.checkpoint)

    @property
    def head(self):
        return self.questions["vad"].head

    @property
    def replay(self):
        """Compatibility view of the VAD question's labeled memory."""
        cases = self.questions["vad"].store.cases
        return [OrderedDict((key, case["vector"]) for key, case in cases.items() if case["label"] == label)
                for label in LABELS]

    def register_question(self, name, options):
        previous = self.questions.get(name)
        if previous is not None:
            if tuple(options) != previous.store.options:
                raise ValueError("A registered question's labels cannot be changed")
            return previous
        if len(self.questions) >= 8:
            raise ValueError("At most eight audio questions may be registered")
        question = AudioQuestion(name, AUDIO_TAG, options)
        self.questions[name] = question
        return question

    def info(self):
        return {"type": "session", "protocol_version": 2, "session_id": self.session_id,
                "profile": PROFILE, "vector_tag": AUDIO_TAG.to_dict(),
                "classifier": "venn-memory-and-compiled-choice", "labels": LABELS,
                "policy": self.policy, "device": self.device, "device_median_ms": self.device_timings,
                "version": self.version, "learning": "labeled-evidence-to-memory",
                "calibrated": False, "frame_ms": FRAME * 1000 / RATE,
                "questions": {name: {"options": q.store.options, "counts": q.store.counts(),
                                     "memory_revision": q.store.revision, "head_revision": q.head_revision}
                              for name, q in self.questions.items()}}

    def _answers(self, features, baseline, names=None):
        answers = {}
        for name in self.questions if names is None else names:
            if name not in self.questions:
                raise ValueError(f"Unknown audio question: {name}")
            q = self.questions[name]
            fresh = q.head is not None and q.head_revision == q.store.revision and q.compilation is not None
            compiled = self.policy == "compiled" and fresh
            evidence = q.compiled_answer(features) if compiled else q.ask(features[None])[0]
            decision = dict(evidence)
            if evidence["abstained"] and name == "vad":
                # No arbitrary exploratory action from OutcomeService controls the gate.
                scores = {"background": 1 - baseline, "speech": baseline}
                decision.update(answer=LABELS[int(baseline >= .5)], source="baseline", abstained=False,
                                scores=scores, probabilities={**scores, "none": 0.0}, score_kind="model_probability")
            decision["memory"] = None if compiled else evidence
            decision["compiled"] = evidence if compiled else None
            decision["head_revision"] = q.head_revision
            decision["head_current"] = fresh
            answers[name] = decision
        return answers

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
            features = np.asarray(features, np.float32)
            if features.shape != (DIMENSIONS,) or not np.isfinite(baseline) or not 0 <= baseline <= 1:
                raise ValueError("Reader returned an invalid audio state")
            answers = self._answers(features, baseline)
            self.samples += FRAME
            self.history.append((self.samples, features.copy(), baseline))
            decision = answers["vad"]
            boundary = self.boundaries.push(decision["scores"]["speech"], self.samples)
            state = {"id": f"{self.session_id}:{self.samples}", "tag": AUDIO_TAG.to_dict()}
            if self.include_vectors:
                state["vector"] = features.tolist()
            events.append({
                "type": "frame", "protocol_version": 2, "session_id": self.session_id, "state": state,
                "start_sample": self.samples - FRAME, "end_sample": self.samples,
                "start": (self.samples - FRAME) / RATE, "end": self.samples / RATE,
                "questions": answers, "decision": decision,
                "baseline_probability": baseline, "speaking": self.boundaries.active,
                "boundary": boundary, "version": self.version,
                "processing_ms": (perf_counter() - started) * 1000,
            })
        return events

    def _select(self, start, end):
        if not all(isinstance(t, (int, float)) and math.isfinite(t) for t in (start, end)) or not 0 <= start < end:
            raise ValueError("Intervals require finite 0 <= start < end seconds")
        if not self.history:
            raise ValueError("No completed audio frames available")
        if start * RATE < self.history[0][0] - FRAME - 1e-5 or end * RATE > self.samples + 1e-5:
            raise ValueError("Interval must lie within the retained, completed 60 seconds")
        selected = [(sample, x, p) for sample, x, p in self.history
                    if sample - FRAME >= start * RATE - 1e-5 and sample <= end * RATE + 1e-5]
        if not selected:
            raise ValueError("Interval must contain at least one complete 32 ms frame")
        return selected

    def ask(self, start, end, *, questions=None):
        """Re-query retained vectors under current memory/heads; never re-encode audio."""
        names = list(self.questions) if questions is None else list(questions)
        if not names or len(set(names)) != len(names) or any(name not in self.questions for name in names):
            raise ValueError("Provide distinct registered question names")
        return [{"state_id": f"{self.session_id}:{sample}", "end": sample / RATE,
                 "answers": self._answers(x, baseline, names)} for sample, x, baseline in self._select(start, end)]

    def feedback(self, start, end, label, *, question="vad", source="manual", group=None):
        q = self.questions.get(question)
        if q is None or label not in q.store.options:
            raise ValueError("Feedback requires a registered question and option label")
        selected = self._select(start, end)
        group = group or uuid4().hex
        protected = self.manual_frames.setdefault(question, set())
        protected.intersection_update(sample for sample, _, _ in self.history)
        if source == "manual":
            protected.update(sample for sample, _, _ in selected)
        changed = sum(q.store.put(f"{self.session_id}:{sample}", x, label, group=group, source=source)
                      for sample, x, _ in selected if source == "manual" or sample not in protected)
        return {"type": "feedback", "question": question, "label": label, "frames": len(selected),
                "changed_frames": changed, "class_counts": q.store.counts(), "memory_revision": q.store.revision,
                "source": source, "feedback_group": group}

    def learn(self, *, question="vad"):
        """Compile the live Venn memory policy into a small MLX head."""
        if question not in self.questions:
            raise ValueError("Unknown audio question")
        q = self.questions[question]
        started = perf_counter()
        X, targets, keys = q.compilation_targets()
        counts = np.bincount(targets, minlength=len(q.store.options)).tolist()
        if min(counts) < 8:
            result = {"accepted": False, "reason": "Need at least 8 supported memory-policy targets per class"}
        else:
            if q.head is None:
                q.anchor = (np.zeros((len(q.store.options), DIMENSIONS), np.float32), np.zeros(len(q.store.options), np.float32))
                q.head = MLXLinearChoiceHead(*q.anchor, q.store.options, device=self.device)
            result = q.head.update(X, targets, anchor=q.anchor)
            if result["accepted"]:
                q.head_revision = q.store.revision
                q.coverage = unit(X).copy()
                self.version += 1
            result["memory_agreement"] = float(np.mean(q.head.probabilities(X).argmax(1) == targets))
            if result["accepted"]:
                q.compilation = {"memory_revision": q.head_revision, "cases": len(X),
                                 "memory_agreement": result["memory_agreement"],
                                 "evaluation": "training-memory agreement, not held-out accuracy"}
        if self.checkpoint:
            self.save(self.checkpoint)
        return {"type": "learning", "question": question, "objective": "compile_memory_policy",
                **result, "version": self.version, "class_counts": counts,
                "memory_revision": q.store.revision, "head_revision": q.head_revision,
                "target_case_ids": keys, "evaluation": "training-memory agreement, not held-out accuracy",
                "processing_ms": (perf_counter() - started) * 1000}

    def reset_learning(self):
        options = {name: q.store.options for name, q in self.questions.items()}
        self.questions = {}
        for name, labels in options.items():
            self.register_question(name, labels)
        vad = self.questions["vad"]
        vad.anchor = self.anchor
        vad.head = MLXLinearChoiceHead(*self.anchor, LABELS, device=self.device)
        self.version = 0
        self.learning_epoch += 1  # Discard delayed teacher results from before reset.
        self.manual_frames.clear()
        if self.checkpoint:
            self.save(self.checkpoint)

    def finish(self):
        trailing = len(self.pending)
        self.pending.clear()
        return {"type": "end", "seconds": self.samples / RATE, "discarded_tail_bytes": trailing,
                "speech_open": self.boundaries.active, "version": self.version}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"schema": 3, "profile": PROFILE, "vector_tag": AUDIO_TAG.to_dict(),
                    "version": self.version, "questions": []}
        arrays = {}
        for index, (name, q) in enumerate(self.questions.items()):
            keys, X, _, y = q.store.arrays()
            prefix = f"q{index}_"
            arrays.update({prefix + "X": X, prefix + "y": y, prefix + "keys": np.array(keys, dtype=str),
                           prefix + "groups": np.array([c["group"] for c in q.store.cases.values()], dtype=str),
                           prefix + "sources": np.array([c["source"] for c in q.store.cases.values()], dtype=str)})
            metadata["questions"].append({"name": name, "options": q.store.options,
                                          "revision": q.store.revision, "head_revision": q.head_revision,
                                          "has_head": q.head is not None,
                                          "compilation": q.compilation,
                                          "max_distance": q.store.max_distance, "margin": q.margin})
            if q.head is not None:
                arrays[prefix + "W"], arrays[prefix + "b"] = q.head.arrays()
            arrays[prefix + "coverage"] = q.coverage
        arrays["metadata"] = json.dumps(metadata)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
                temporary = Path(file.name)
                np.savez(file, **arrays)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)

    def load(self, path):
        """Validate before publishing; v1 labeled buffers migrate into Venn memory.

        V1 heads remain available but are marked stale until compiled from memory.
        """
        # Covers eight bounded questions, compiled coverage, and 128-character IDs.
        if Path(path).stat().st_size > 48_000_000:
            raise ValueError("Checkpoint exceeds size limit")
        restored = {}
        with np.load(path, allow_pickle=False) as saved:
            meta = json.loads(str(saved["metadata"].item()))
            if not isinstance(meta, dict) or meta.get("profile") != PROFILE or meta.get("schema") not in (1, 2, 3):
                raise ValueError("Checkpoint audio profile or schema does not match")
            if type(meta.get("version")) is not int or meta["version"] < 0:
                raise ValueError("Invalid checkpoint version")
            legacy = meta["schema"] == 1
            if legacy:
                if meta.get("labels") != list(LABELS) or meta.get("temperature") != 1.0:
                    raise ValueError("Invalid legacy classifier")
                specs = [{"name": "vad", "options": LABELS, "revision": len(saved["X"]),
                          "head_revision": None, "has_head": True, "max_distance": .15, "margin": .2}]
            else:
                if meta.get("vector_tag") != AUDIO_TAG.to_dict():
                    raise ValueError("Checkpoint vector tag does not match")
                specs = meta.get("questions")
            if not isinstance(specs, list) or not 1 <= len(specs) <= 8:
                raise ValueError("Invalid checkpoint questions")
            for index, spec in enumerate(specs):
                if not isinstance(spec, dict) or not {"name", "options", "revision", "head_revision", "has_head", "max_distance", "margin"} <= set(spec):
                    raise ValueError("Invalid checkpoint question metadata")
                name, options = spec["name"], spec["options"]
                if name in restored or name == "vad" and tuple(options) != LABELS:
                    raise ValueError("Invalid checkpoint question labels")
                q = AudioQuestion(name, AUDIO_TAG, options, max_distance=spec["max_distance"], margin=spec["margin"])
                prefix = "" if legacy else f"q{index}_"
                X, y, keys = saved[prefix + "X"], saved[prefix + "y"], saved[prefix + "keys"]
                groups = np.array(["legacy"] * len(X)) if legacy else saved[prefix + "groups"]
                sources = saved[prefix + "sources"] if meta["schema"] == 3 else np.full(len(X), "manual")
                if (X.ndim != 2 or X.shape[1] != DIMENSIONS or len(X) > MAX_PER_LABEL * len(options)
                        or y.shape != (len(X),) or keys.shape != (len(X),) or groups.shape != (len(X),)
                        or not np.isfinite(X).all() or not np.issubdtype(y.dtype, np.integer)
                        or not np.isin(y, np.arange(len(options))).all()
                        or keys.dtype.kind != 'U' or groups.dtype.kind != 'U'
                        or sources.shape != (len(X),) or sources.dtype.kind != 'U'
                        or len(set(keys.tolist())) != len(keys)
                        or np.max(np.bincount(y, minlength=len(options))) > MAX_PER_LABEL):
                    raise ValueError("Invalid checkpoint replay data")
                revision, head_revision = spec["revision"], spec["head_revision"]
                if (type(revision) is not int or revision < len(X)
                        or head_revision is not None and (type(head_revision) is not int or not 0 <= head_revision <= revision)
                        or type(spec["has_head"]) is not bool or head_revision is not None and not spec["has_head"]):
                    raise ValueError("Invalid checkpoint memory revision")
                for key, x, label, group, source in zip(keys, X, y, groups, sources):
                    q.store.put(str(key), x, options[int(label)], group=str(group), source=str(source))
                q.store.revision = revision
                q.head_revision = head_revision
                if not legacy and spec.get("compilation") is not None:
                    coverage = saved[prefix + "coverage"]
                    report = spec["compilation"]
                    if (head_revision is None or not isinstance(report, dict) or report.get("memory_revision") != head_revision
                            or coverage.ndim != 2 or coverage.shape[1] != DIMENSIONS
                            or not 1 <= len(coverage) <= MAX_PER_LABEL * len(options)
                            or report.get("cases") != len(coverage)
                            or not isinstance(report.get("memory_agreement"), (int, float))
                            or not 0 <= report["memory_agreement"] <= 1
                            or not np.isfinite(coverage).all()
                            or not np.allclose(np.linalg.norm(coverage, axis=1), 1, atol=1e-5)):
                        raise ValueError("Invalid compiled coverage or report")
                    q.coverage, q.compilation = coverage.copy(), report
                else:
                    q.head_revision = None
                q.anchor = self.anchor if name == "vad" else (np.zeros((len(options), DIMENSIONS), np.float32), np.zeros(len(options), np.float32))
                if spec["has_head"]:
                    W, b = saved[prefix + "W"], saved[prefix + "b"]
                    if W.shape != (len(options), DIMENSIONS):
                        raise ValueError("Invalid checkpoint classifier dimensions")
                    q.head = MLXLinearChoiceHead(W, b, options, device=self.device)
                if name == "vad" and q.head is None:
                    raise ValueError("Checkpoint is missing the VAD head")
                restored[name] = q
        if "vad" not in restored:
            raise ValueError("Checkpoint is missing the VAD question")
        self.questions, self.version = restored, meta["version"]
