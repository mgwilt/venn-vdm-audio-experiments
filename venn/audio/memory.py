"""App-owned audio cases queried by Venn's existing OutcomeService.

A categorical correction defines a known label-agreement utility for every
option (1 for the corrected label, 0 otherwise). These are supervised utilities,
not observed rewards from taking actions and not calibrated probabilities.
"""
from collections import OrderedDict

import numpy as np

from venn.outcomes import OutcomeService
from venn.profiles import VectorTag

MAX_PER_LABEL = 256


def unit(vectors):
    values = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms <= 1e-12):
        raise ValueError("Audio memory vectors must be finite and nonzero")
    return values / norms


class AudioCaseStore:
    """Bounded exact cosine search with the standard Venn outcome-adapter contract.

    The distance cutoff is applied before top-k. Unfamiliar regions return no
    evidence rather than arbitrarily distant neighbors. Call on one owner thread.
    """
    read_after_write = "immediate in-process; caller serializes reads and writes"
    distance = "cosine"

    def __init__(self, tag, options, *, max_distance=0.15):
        self.tag = VectorTag.parse(tag)
        if (not isinstance(options, (list, tuple)) or not 2 <= len(options) <= 8
                or not all(isinstance(v, str) and 0 < len(v) <= 64 and v != "none" for v in options)
                or len(set(options)) != len(options)):
            raise ValueError("Provide 2..8 unique option labels excluding 'none'")
        if not np.isfinite(max_distance) or not 0 <= max_distance <= 2:
            raise ValueError("Cosine distance cutoff must be in [0, 2]")
        self.options = tuple(options)
        self.max_distance = float(max_distance)
        self.cases = OrderedDict()
        self.revision = 0
        self._cache = None

    def put(self, key, vector, label, *, group):
        x = np.asarray(vector, dtype=np.float32)
        if (x.shape != (self.tag.dimensions,) or label not in self.options
                or not isinstance(key, str) or not 0 < len(key) <= 128
                or not isinstance(group, str) or not 0 < len(group) <= 128):
            raise ValueError("Invalid labeled audio case")
        unit(x)
        old = self.cases.get(key)
        if old is not None and old["label"] == label and np.array_equal(old["vector"], x):
            return False
        self.cases.pop(key, None)
        self.cases[key] = {"vector": x.copy(), "label": label, "group": group}
        same = [identity for identity, case in self.cases.items() if case["label"] == label]
        for identity in same[:-MAX_PER_LABEL]:
            del self.cases[identity]
        self.revision += 1
        self._cache = None
        return True

    def arrays(self):
        if self._cache is None:
            keys = list(self.cases)
            raw = np.array([c["vector"] for c in self.cases.values()], np.float32).reshape(-1, self.tag.dimensions)
            labels = np.array([self.options.index(c["label"]) for c in self.cases.values()], np.int32)
            self._cache = keys, raw, unit(raw), labels
        return self._cache

    def search(self, queries, options, k, *, filters, exact_match=False):
        if exact_match:
            raise ValueError("Audio memory uses distance-bounded evidence, not exact-key overrides")
        if not set(options) <= set(self.options) or type(k) is not int or not 1 <= k <= 1024:
            raise ValueError("Invalid audio memory options or neighbor count")
        q = np.asarray(queries, np.float32)
        if q.ndim != 2 or q.shape[1] != self.tag.dimensions:
            raise ValueError("Queries do not match the audio tag")
        q = unit(q)
        keys, _, vectors, labels = self.arrays()
        distances = np.clip(1 - q @ vectors.T, 0, 2)
        allowed = np.ones(len(keys), dtype=bool)
        if filters:
            allowed = np.array([all(self.cases[key].get(field) == value for field, value in filters.items())
                                for key in keys], dtype=bool)
        result = {option: [] for option in options}
        for row in distances:
            eligible = np.flatnonzero(allowed & (row <= self.max_distance))
            eligible = eligible[np.argsort(row[eligible], kind="stable")[:k]]
            for option in options:
                target = self.options.index(option)
                result[option].append([
                    {"id": keys[i], "distance": float(row[i]), "outcome": float(labels[i] == target)}
                    for i in eligible
                ])
        return result

    def export(self):
        _, raw, _, labels = self.arrays()
        return {option: (raw.copy(), (labels == i).astype(np.float32)) for i, option in enumerate(self.options)}

    def counts(self):
        return [sum(case["label"] == option for case in self.cases.values()) for option in self.options]


class AudioQuestion:
    """One named decision over a shared audio representation, backed by Venn itself."""
    def __init__(self, name, tag, options, *, max_distance=0.15, margin=0.2):
        if not isinstance(name, str) or not 0 < len(name) <= 64:
            raise ValueError("Question name must be 1..64 characters")
        if not np.isfinite(margin) or not 0 <= margin <= 1:
            raise ValueError("Evidence margin must be in [0, 1]")
        self.name, self.margin = name, float(margin)
        self.store = AudioCaseStore(tag, options, max_distance=max_distance)
        self.service = OutcomeService()
        self.handle = self.service.register_store(self.store)
        self.rng = np.random.default_rng(0)
        self.head = None
        self.head_revision = None
        self.anchor = None
        self.coverage = np.empty((0, self.store.tag.dimensions), np.float32)
        self.compilation = None

    def compiled_answer(self, vector):
        """Answer without the live store using the head and its frozen coverage reference.

        Coverage vectors belong to the compiled artifact. They carry no live
        outcomes; attribution is to the compilation revision, not retrieved cases.
        """
        covered = bool(len(self.coverage)) and float(np.min(
            np.clip(1 - unit(vector) @ self.coverage.T, 0, 2))) <= self.store.max_distance
        probabilities = self.head.probabilities(np.asarray(vector, np.float32)[None])[0] if covered else None
        scores = {option: float(probabilities[i]) if covered else None for i, option in enumerate(self.store.options)}
        return {"type": "Choice", "answer": self.store.options[int(np.argmax(probabilities))] if covered else "none",
                "source": "compiled", "abstained": not bool(covered),
                "scores": scores, "score_kind": "model_probability",
                "probabilities": {**scores, "none": 0.0} if covered else None, "calibrated": False,
                "reason": None if covered else "outside_compiled_coverage", "evidence": None,
                "cases": [], "evidence_groups": 0, "memory_revision": self.store.revision,
                "compilation": self.compilation}

    def ask(self, vectors):
        """Use the public nearest_outcome API; retain evidence and explicit abstention."""
        values = np.asarray(vectors, np.float32)
        result = self.service.decide_batch(
            [{"vector": vector, "tag": self.store.tag.to_dict()} for vector in values],
            {self.name: {"type": "nearest_outcome", "store": self.handle,
                         "options": list(self.store.options), "k": 5, "min_support": 1}},
            rng=self.rng,
        )
        answers = []
        for row in result["results"]:
            evidence = row["answers"][self.name]
            scores = {label: details["mean"] for label, details in evidence["options"].items()}
            ranked = sorted((score for score in scores.values() if score is not None), reverse=True)
            reason = "no_nearby_cases" if len(ranked) < 2 else None
            if reason is None and ranked[0] - ranked[1] + 1e-12 < self.margin:
                reason = "conflicting_cases"
            ids = sorted({key for item in evidence["options"].values() for key in item["evidence_ids"]})
            answers.append({
                "type": "Choice", "answer": "none" if reason else evidence["answer"],
                "source": "memory", "abstained": reason is not None,
                "probabilities": None, "calibrated": False,
                "scores": scores, "score_kind": "neighbor_label_agreement",
                "reason": reason, "memory_revision": self.store.revision,
                "evidence": evidence["options"],
                "cases": [{"id": key, "label": self.store.cases[key]["label"],
                           "feedback_group": self.store.cases[key]["group"]} for key in ids],
                "evidence_groups": len({self.store.cases[key]["group"] for key in ids}),
            })
        return answers

    def compilation_targets(self):
        """Distill the memory's policy, not the original feedback labels.

        Conflicted/unsupported cases are excluded. Targets use exactly the same
        Venn query as live inference. Self-neighbors match Venn's existing store
        compilation semantics; agreement here is not held-out accuracy.
        """
        keys, raw, _, _ = self.store.arrays()
        rows, targets, evidence_ids = [], [], []
        for begin in range(0, len(raw), 128):
            for offset, answer in enumerate(self.ask(raw[begin:begin + 128])):
                if not answer["abstained"]:
                    rows.append(raw[begin + offset])
                    targets.append(self.store.options.index(answer["answer"]))
                    evidence_ids.append(keys[begin + offset])
        return (np.asarray(rows, np.float32).reshape(-1, self.store.tag.dimensions),
                np.asarray(targets, np.int32), evidence_ids)
