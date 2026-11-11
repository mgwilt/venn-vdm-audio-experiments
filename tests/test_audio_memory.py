"""Venn outcome semantics for reusable audio states, without any neural runtime."""
import sys
import subprocess

import numpy as np
import pytest

from venn.audio.memory import AudioCaseStore, AudioQuestion
from venn.audio.reader import AUDIO_TAG, DIMENSIONS
from venn.outcomes import OutcomeService
from venn.profiles import VectorTag


def vector(axis=0):
    x = np.zeros(DIMENSIONS, np.float32)
    x[axis] = 1
    return x


def test_same_store_is_usable_by_venns_public_outcome_service():
    store = AudioCaseStore(AUDIO_TAG, ["quiet", "speech"])
    store.put("correction-1", vector(), "speech", group="feedback-1")
    service = OutcomeService()
    handle = service.register_store(store)
    answer = service.decide_batch(
        [{"vector": vector(), "tag": AUDIO_TAG.to_dict()}],
        {"voice": {"type": "nearest_outcome", "store": handle, "options": ["quiet", "speech"]}},
    )["results"][0]["answers"]["voice"]
    assert answer["answer"] == "speech"
    assert answer["options"]["speech"]["mean"] == 1
    assert answer["options"]["quiet"]["mean"] == 0
    assert answer["options"]["speech"]["evidence_ids"] == ["correction-1"]
    assert answer["probabilities"] is None
    changed = AUDIO_TAG.to_dict()
    changed["revision"] = "0" * 40
    with pytest.raises(ValueError, match="tag mismatch"):
        service.decide_batch([{"vector": vector(), "tag": changed}],
                             {"voice": {"type": "nearest_outcome", "store": handle, "options": ["quiet", "speech"]}})


def test_memory_abstains_on_unknown_and_conflicting_audio():
    question = AudioQuestion("voice", AUDIO_TAG, ["quiet", "speech"])
    assert question.ask(vector()[None])[0]["reason"] == "no_nearby_cases"
    question.store.put("speech-1", vector(), "speech", group="a")
    known = question.ask(vector()[None])[0]
    assert known["answer"] == "speech" and known["evidence_groups"] == 1
    assert question.ask(vector(1)[None])[0]["reason"] == "no_nearby_cases"
    question.store.put("quiet-1", vector(), "quiet", group="b")
    conflict = question.ask(vector()[None])[0]
    assert conflict["answer"] == "none" and conflict["reason"] == "conflicting_cases"
    assert conflict["evidence_groups"] == 2


def test_compilation_targets_are_the_memory_policy_not_raw_labels():
    question = AudioQuestion("voice", AUDIO_TAG, ["quiet", "speech"])
    for i, label in enumerate(["speech", "speech", "speech", "quiet"]):
        question.store.put(str(i), vector(), label, group=str(i))
    X, targets, ids = question.compilation_targets()
    assert X.shape == (4, DIMENSIONS) and len(ids) == 4
    assert list(targets) == [1, 1, 1, 1]  # Even the contradictory case learns the memory policy.
    assert question.store.counts() == [1, 3]


def test_three_of_five_votes_meet_the_default_evidence_margin():
    question = AudioQuestion("voice", AUDIO_TAG, ["quiet", "speech"])
    for i, label in enumerate(["speech", "speech", "speech", "quiet", "quiet"]):
        question.store.put(str(i), vector(), label, group=str(i))
    answer = question.ask(vector()[None])[0]
    assert answer["answer"] == "speech" and not answer["abstained"]


def test_relabeling_and_capacity_have_immediate_search_visibility():
    store = AudioCaseStore(AUDIO_TAG, ["a", "b"])
    store.put("same", vector(), "a", group="first")
    revision = store.revision
    assert not store.put("same", vector(), "a", group="duplicate")
    assert store.revision == revision
    assert store.put("same", vector(), "b", group="correction")
    assert store.search(vector()[None], ["a", "b"], 1, filters={})["b"][0][0]["outcome"] == 1
    for i in range(300):
        store.put(f"b-{i}", vector(), "b", group="long-interval")
    assert len(store.cases) == 256 and "same" not in store.cases
    assert not store.search(vector()[None], ["b"], 5, filters={"group": "missing"})["b"][0]
    exported = store.export()
    assert exported["b"][0].shape == (256, DIMENSIONS)
    assert np.all(exported["b"][1] == 1)


def test_vector_tags_and_geometry_reject_incompatible_inputs():
    store = AudioCaseStore(AUDIO_TAG, ["a", "b"])
    with pytest.raises(ValueError, match="nonzero"):
        store.put("zero", np.zeros(DIMENSIONS), "a", group="bad")
    with pytest.raises(ValueError, match="finite"):
        store.put("nan", np.full(DIMENSIONS, np.nan), "a", group="bad")
    with pytest.raises(ValueError, match="Invalid"):
        store.put("wrong", vector()[:2], "a", group="bad")
    assert VectorTag.parse(AUDIO_TAG.to_dict()) == AUDIO_TAG


def test_memory_has_no_torch_faiss_or_mlx_runtime_dependency():
    subprocess.run([sys.executable, "-c", "from venn.audio.memory import AudioQuestion; import sys; "
                    "assert not {'torch', 'faiss', 'mlx.core'} & sys.modules.keys()"], check=True)
