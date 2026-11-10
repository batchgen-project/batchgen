"""Resolve a ``--model`` value to the registered model id it names.

Per-model tables (host-KV profiles, host weight envelopes) are keyed by the
released Hugging Face id. ``--model`` may also be a local checkpoint
directory named after that id, or a release variant whose name only adds a
variant suffix (``-Base``, ``-Instruct``, ``-Chat`` or a date). Every table
lookup goes through :func:`lookup`, so one rule decides what an identifier
names, and an identifier that names no registered model fails at the table
that needs it instead of being matched by substring to a neighbour (an
unlisted ``Kimi-K3.2`` must not resolve to ``Kimi-K3``).
"""

from __future__ import annotations

import re
from typing import Dict, Iterator, Mapping, Optional, TypeVar

V = TypeVar("V")

# A trailing release-variant suffix that does not change the model's
# geometry: instruction or base tuning, or a date stamp (-2507, -20250101).
_VARIANT_SUFFIX = re.compile(r"-(?:instruct|base|chat|\d{4}|\d{6}|\d{8})$")


def strip_variant(key: str) -> str:
    """``key`` without its trailing release-variant suffixes."""
    while True:
        match = _VARIANT_SUFFIX.search(key)
        if match is None:
            return key
        key = key[: match.start()]


def candidate_keys(model_name: str) -> Iterator[str]:
    """Normalized keys ``model_name`` may be registered under, most exact
    first: the whole value, then ``org/name`` and ``name`` from its last
    path components (a local directory named after the release), each
    without its variant suffixes."""
    if not isinstance(model_name, str):
        raise ValueError("model_name must be a string")
    name = model_name.strip().lower().rstrip("/")
    parts = [part for part in name.split("/") if part]
    seen = set()
    for candidate in (name, "/".join(parts[-2:]), parts[-1] if parts else ""):
        key = strip_variant(candidate)
        if key and key not in seen:
            seen.add(key)
            yield key


def build_index(table: Mapping[str, V]) -> Dict[str, V]:
    """``table`` re-keyed by variant-stripped lowercase id, plus the bare
    release name of each ``org/name`` id.

    Fails at import when two registered ids that differ only by a variant
    suffix map to different values: the suffix would then matter, and
    stripping it would merge two models.
    """
    index: Dict[str, V] = {}
    for alias, value in table.items():
        key = strip_variant(alias.strip().lower())
        if key in index and index[key] != value:
            raise ValueError(
                f"registered ids {alias!r} and another variant of {key!r} map "
                "to different entries; a variant suffix cannot be ignored here"
            )
        index[key] = value
    # The bare release name of an ``org/name`` id names it too (a directory
    # called GLM-5.3-FP8), unless two registered ids share that name with
    # different entries; such a name stays unresolved rather than guessed.
    names: Dict[str, V] = {}
    ambiguous = set()
    for key, value in index.items():
        if "/" not in key:
            continue
        name = key.rsplit("/", 1)[1]
        if name in names and names[name] != value:
            ambiguous.add(name)
        names.setdefault(name, value)
    for name, value in names.items():
        if name not in ambiguous and name not in index:
            index[name] = value
    return index


def lookup(index: Mapping[str, V], model_name: str) -> Optional[V]:
    """The entry ``model_name`` names in a :func:`build_index` index."""
    for key in candidate_keys(model_name):
        if key in index:
            return index[key]
    return None


__all__ = ["build_index", "candidate_keys", "lookup", "strip_variant"]
