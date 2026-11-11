"""Venn: typed decisions over embeddings.

Public entry points:

- :class:`OutcomeService`: decisions from an application-owned store of past cases
  and their outcomes; runs in-process with no trained head, and can compile a store
  into a small head that answers the same question without a lookup.
- :class:`DecisionService` and :func:`create_app`: the Choice/Score/Noul decision
  engine and its HTTP application, backed by the ``venn-1`` head.
- :class:`VectorTag`, :data:`TEXT_TAG`, :data:`VENN_1`: the profile and tag types
  that every vector must carry.

The engine lives in :mod:`venn.server`, :mod:`venn.heads`, :mod:`venn.index`, :mod:`venn.reader`
and :mod:`venn.runtime`.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("venn-vdm")
except PackageNotFoundError:
    __version__ = "0+source"
__all__ = ["DecisionService", "OutcomeService", "create_app", "VectorTag", "TEXT_TAG", "IMAGE_TAGS", "VENN_1"]


def __getattr__(name):
    """Keep audio-only processes from loading the Torch/text service."""
    from importlib import import_module

    modules = {
        "DecisionService": "venn.server", "create_app": "venn.server",
        "OutcomeService": "venn.outcomes", "VectorTag": "venn.profiles",
        "TEXT_TAG": "venn.profiles", "IMAGE_TAGS": "venn.profiles", "VENN_1": "venn.profiles",
    }
    if name not in modules:
        raise AttributeError(f"module 'venn' has no attribute {name!r}")
    value = getattr(import_module(modules[name]), name)
    globals()[name] = value
    return value
