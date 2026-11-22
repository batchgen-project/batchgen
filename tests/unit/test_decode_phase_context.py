"""CPU checks for the frozen decode admission context."""

from __future__ import annotations

import dataclasses
import inspect
from types import SimpleNamespace

import pytest

from batchgen.worker.admission.context import (
    DecodePhaseContext,
    RankFacts,
    SchedulingState,
)
from tests.worker_method_source import load_worker_methods


_build_context = load_worker_methods(
    ["_decode_phase_context"],
    {
        "DecodePhaseContext": DecodePhaseContext,
        "RankFacts": RankFacts,
        "SchedulingState": SchedulingState,
    },
)["_decode_phase_context"]


class _Worker:
    rank = 2
    world_size = 4
    local_rank = 0
    local_world_size = 2
    rank_ctx = SimpleNamespace()
    torch_device = "cpu"
    PAGE_SIZE = 64
    DECISION_INTERVAL = 128
    enable_prefix_cache = True
    enable_host_kv_eviction = True
    host_kv_chunk_size = 256
    prefix_cache_debug_stats = False
    _kv_stats_native = True
    _kv_stats_compare = False
    global_batch = object()
    seq_table = object()
    _traj = object()
    _prefix = object()
    gpu_paged_kv_cache_manager = object()
    host_paged_kv_worker_view = object()
    core_engine = object()
    _sequences_with_gpu_kv = set()
    prefix_cache_coordinator = object()
    prefix_cache_runtime_config = object()
    adaptive_chunk_sizer = None
    parallel_manager = SimpleNamespace(attn_tp_size=2)

    def _decode_attn_tp_size(self):
        return self.parallel_manager.attn_tp_size


def _all_field_values(value):
    if dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            field_value = getattr(value, field.name)
            yield field_value
            yield from _all_field_values(field_value)


def test_context_is_frozen_and_does_not_capture_worker_or_bound_methods():
    worker = _Worker()
    context = _build_context(worker)

    assert isinstance(context, DecodePhaseContext)
    assert isinstance(context.facts, RankFacts)
    assert isinstance(context.sched, SchedulingState)
    for dataclass_type in (RankFacts, SchedulingState, DecodePhaseContext):
        assert dataclasses.is_dataclass(dataclass_type)
        assert dataclass_type.__dataclass_params__.frozen
        instance = context if dataclass_type is DecodePhaseContext else getattr(
            context, "facts" if dataclass_type is RankFacts else "sched"
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            instance.__setattr__(dataclasses.fields(dataclass_type)[0].name, None)

    for value in _all_field_values(context):
        assert value is not worker
        assert not (inspect.ismethod(value) and value.__self__ is worker)


def test_context_carries_the_prefix_state_object():
    """The pool-segment release reads the prefix-reuse state from the context."""
    worker = _Worker()
    context = _build_context(worker)
    assert context.sched.prefix is worker._prefix


def test_context_carries_the_kv_stats_flags_as_plain_facts():
    """The stats-tier flags are set once at worker init and never rebound."""
    worker = _Worker()
    context = _build_context(worker)
    assert context.facts.kv_stats_native is True
    assert context.facts.kv_stats_compare is False
