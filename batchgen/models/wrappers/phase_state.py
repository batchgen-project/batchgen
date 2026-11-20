# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                            #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
#                                                                               #
#  you may obtain a copy of the license at                                      #
#                                                                               #
#                  http://www.apache.org/licenses/license-2.0                   #
#                                                                               #
#  unless required by applicable law or agreed to in writing, software          #
#  distributed under the license is distributed on an "as is" basis,            #
#  without warranties or conditions of any kind, either express or implied.     #
#  see the license for the specific language governing permissions and          #
#  limitations under the license.                                               #
# ---------------------------------------------------------------------------- #

"""Process-wide model phase state: its single owner and its single writer.

The model wrappers keep their phase state (prefill/decode context) at process
scope, so it is only well defined while exactly one ``BatchGenWorker`` drives
it. ``claim_process_phase_state`` enforces that: a second live worker in the
same process fails loudly at construction instead of silently sharing, and
overwriting, the first worker's state.

The claim holds its owner weakly, so a worker that has been garbage collected
no longer blocks a successor. Workers are constructed once per process, on the
process's main thread, so the claim needs no lock.

This module is also the ONLY place that assigns the per-phase class attributes
on the wrapper classes. The attributes
are grouped by how they are written together; one dataclass per group names
the group's fields, and one ``bind_*`` function per group performs the
assignments. A binder assigns exactly the fields the caller passes, in the
order the caller passes them, to exactly the class object the caller names;
the value object passed is the object assigned. Readers still read the class
attributes. A later migration can convert them, group by group, to read these
dataclasses instead.
"""

from __future__ import annotations

import dataclasses
import weakref
from typing import Any, Mapping, Optional

__all__ = [
    "DebugState",
    "DecodeEntryBinding",
    "KVAppendBinding",
    "PhaseFlag",
    "PrefillPoolBinding",
    "PrepackMetadata",
    "StepMetadata",
    "PHASE_STATE_FIELDS",
    "bind_debug",
    "bind_decode_entry",
    "bind_kv_append",
    "bind_phase",
    "bind_prefill_pool",
    "bind_prepack",
    "bind_step",
    "claim_process_phase_state",
    "release_process_phase_state",
    "restore_saved_state",
]

_owner_ref: Optional["weakref.ReferenceType[Any]"] = None


def _current_owner() -> Any:
    return None if _owner_ref is None else _owner_ref()


def claim_process_phase_state(owner: Any) -> None:
    """Record ``owner`` as the one worker driving this process's phase state.

    Raises ``RuntimeError`` if a different owner still holds the claim. The
    same owner may claim again.
    """
    global _owner_ref
    current = _current_owner()
    if current is not None and current is not owner:
        raise RuntimeError(
            "a BatchGenWorker already owns the model phase state of this process"
        )
    _owner_ref = weakref.ref(owner)


def release_process_phase_state(owner: Any) -> None:
    """Drop the claim if ``owner`` holds it; otherwise do nothing."""
    global _owner_ref
    if _current_owner() is owner:
        _owner_ref = None


# --------------------------------------------------------------------------- #
# Phase-state groups. Field names equal the wrapper class-attribute
# names; each attribute belongs to exactly one group.
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class PhaseFlag:
    """The phase routing flag; selects the wrappers' forward branch."""

    phase: Optional[str] = None


@dataclasses.dataclass
class DebugState:
    """The batch-level debug dict active for the running step."""

    batchgen_debug: Optional[dict] = None


@dataclasses.dataclass
class StepMetadata:
    """Per-step batch metadata (prefill and decode)."""

    attention_mask: Optional[Any] = None
    position_ids: Optional[Any] = None
    cur_batch: Optional[Any] = None
    cache_seqlens: Optional[Any] = None
    max_seqlen: Optional[Any] = None


@dataclasses.dataclass
class DecodeEntryBinding:
    """KV managers, host views and cached-state handles bound at decode
    entry; the host views are also bound, alone, for the prefill offload."""

    gpu_paged_kv_manager: Optional[Any] = None
    gpu_paged_kv_manager_aux: Optional[Any] = None
    host_paged_kv_worker_view: Optional[Any] = None
    host_paged_kv_worker_view_aux: Optional[Any] = None
    past_key_states: Optional[Any] = None
    scale: Optional[Any] = None


@dataclasses.dataclass
class KVAppendBinding:
    """The per-layer KV append callbacks for the decode forward."""

    kv_append_callback: Optional[Any] = None
    kv_append_callback_aux: Optional[Any] = None


@dataclasses.dataclass
class PrepackMetadata:
    """One prepacked (varlen) prefill micro-batch's packed-row metadata."""

    prepack_mode: Optional[bool] = None
    prepack_cu_seqlens: Optional[Any] = None
    prepack_max_seqlen: Optional[Any] = None
    prepack_num_sequences: Optional[Any] = None
    prepack_seq_lengths: Optional[Any] = None
    prepack_full_seq_lengths: Optional[Any] = None
    prepack_compute_cached_tokens: Optional[Any] = None
    prepack_attached_tokens: Optional[Any] = None
    prefill_prefix_materialization: Optional[Any] = None


@dataclasses.dataclass
class PrefillPoolBinding:
    """One prefill pool wave's bindings."""

    prefill_pool: Optional[Any] = None
    prefill_pool_batch: Optional[Any] = None
    prefill_pool_row_starts: Optional[Any] = None
    prefill_pool_segment_sids: Optional[Any] = None
    prefill_pool_materialized_segments: Optional[Any] = None
    prefill_pool_layer_map: Optional[Any] = None
    prefill_pool_offload_recorder: Optional[Any] = None


def _group_fields(group: type) -> frozenset:
    return frozenset(f.name for f in dataclasses.fields(group))


_PHASE_FLAG_FIELDS = _group_fields(PhaseFlag)
_DEBUG_FIELDS = _group_fields(DebugState)
_STEP_FIELDS = _group_fields(StepMetadata)
_DECODE_ENTRY_FIELDS = _group_fields(DecodeEntryBinding)
_KV_APPEND_FIELDS = _group_fields(KVAppendBinding)
_PREPACK_FIELDS = _group_fields(PrepackMetadata)
_PREFILL_POOL_FIELDS = _group_fields(PrefillPoolBinding)

#: Every class attribute this module owns (the single-writer test reads this).
PHASE_STATE_FIELDS = frozenset().union(
    _PHASE_FLAG_FIELDS,
    _DEBUG_FIELDS,
    _STEP_FIELDS,
    _DECODE_ENTRY_FIELDS,
    _KV_APPEND_FIELDS,
    _PREPACK_FIELDS,
    _PREFILL_POOL_FIELDS,
)


def _bind(target: type, fields: dict, allowed: frozenset, binder: str) -> None:
    """Assign ``fields`` on the class object ``target`` in caller order.

    Keyword arguments preserve the caller's order, so the assignments happen
    exactly in the order the call site lists them; a fixed parameter list
    would impose the signature's order instead. Only field names of the
    binder's group are accepted; an unknown name is a programming error and
    raises ``TypeError`` before anything is assigned. The value object passed
    is the object assigned: no copying, no conversion, no ``None`` filtering.
    """
    for name in fields:
        if name not in allowed:
            raise TypeError(
                f"{binder}() got an unexpected phase-state field {name!r}"
            )
    for name, value in fields.items():
        setattr(target, name, value)


def bind_phase(target: type, **fields: Any) -> None:
    """Assign the phase flag (``PhaseFlag`` group) on ``target``."""
    _bind(target, fields, _PHASE_FLAG_FIELDS, "bind_phase")


def bind_debug(target: type, **fields: Any) -> None:
    """Assign the debug dict (``DebugState`` group) on ``target``."""
    _bind(target, fields, _DEBUG_FIELDS, "bind_debug")


def bind_step(target: type, **fields: Any) -> None:
    """Assign step metadata (``StepMetadata`` group) on ``target``."""
    _bind(target, fields, _STEP_FIELDS, "bind_step")


def bind_decode_entry(target: type, **fields: Any) -> None:
    """Assign decode-entry bindings (``DecodeEntryBinding`` group)."""
    _bind(target, fields, _DECODE_ENTRY_FIELDS, "bind_decode_entry")


def bind_kv_append(target: type, **fields: Any) -> None:
    """Assign KV append callbacks (``KVAppendBinding`` group)."""
    _bind(target, fields, _KV_APPEND_FIELDS, "bind_kv_append")


def bind_prepack(target: type, **fields: Any) -> None:
    """Assign prepack metadata (``PrepackMetadata`` group)."""
    _bind(target, fields, _PREPACK_FIELDS, "bind_prepack")


def bind_prefill_pool(target: type, **fields: Any) -> None:
    """Assign prefill-pool bindings (``PrefillPoolBinding`` group)."""
    _bind(target, fields, _PREFILL_POOL_FIELDS, "bind_prefill_pool")


def restore_saved_state(target: type, saved: Mapping[str, Any]) -> None:
    """Restore a saved attribute mapping on ``target`` in its iteration order.

    The one writer for the models' ``startup_decode_probe_step`` restore
    loops: the save side reads the attributes into a mapping and stays at the
    call site; this assigns them back in the mapping's own iteration order.
    """
    for name in saved:
        if name not in PHASE_STATE_FIELDS:
            raise TypeError(
                f"restore_saved_state() got an unexpected phase-state field "
                f"{name!r}"
            )
    for name, value in saved.items():
        setattr(target, name, value)
