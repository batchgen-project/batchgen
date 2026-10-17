# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3                                                           #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""The pooled-prefill probe seam: free when off, honest about the gather.

Four diagnostics ride the same batch-level ``batchgen_debug`` seam - the
epilogue trace, the Host offload byte check, the kernel-argument dump and the
pad-poison probe. Two properties of that seam are load-bearing and neither is
visible from a probe's own unit tests:

1. OFF IS FREE. A production prefill arms none of them, so deciding that must
   not build a list of sequences, build a merged debug dict, or import a probe
   module - and it must walk the wave's sequences ONCE for all four probes, not
   once each. This file counts the walk and counts the imports.

2. THE GATHER RECORD IS NOT HAND-WRITTEN WHERE IT COULD BE WRONG. The trace
   line carries ``tp=<rank>/<size>`` and an owned mask. Writing ``0/1, owned``
   from the caller is only true where the rank holds every row; on the
   streamed-SP8 path it holds one contiguous row shard and a hand-written
   record would report a BROKEN multi-rank gather as healthy - a false accept,
   which is the one failure an acceptance probe may not have. So the tracer is
   passed INTO the real gather on both prefill paths, and the single-rank
   record is only written where the manager has no gather at all, asserted.

``batchgen_worker`` imports the whole engine, so - as in
``tests/unit/test_prefix_drain_metric.py`` - the seam methods are compiled in
isolation against a fake worker, and the two call sites that cannot be compiled
in isolation (they sit inside the full prefill loops) are pinned structurally
against the module's own AST.
"""

from __future__ import annotations

import ast
import builtins
import copy
import logging
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from batchgen.prefix_reuse import epilogue_trace as _TRACE
from batchgen.prefix_reuse.wave_runtime import PoolWave

WORKER = Path(__file__).resolve().parents[2] / "batchgen" / "batchgen_worker.py"
TREE = ast.parse(WORKER.read_text())


def _method(name: str) -> ast.FunctionDef:
    return next(
        node for node in ast.walk(TREE)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _module_strings() -> dict:
    """Every module-level ``NAME = "literal"`` of the worker."""

    out = {}
    for node in TREE.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                out[target.id] = node.value.value
    return out


def _module_key_tuple() -> tuple:
    """``_POOL_PROBE_DEBUG_KEYS`` resolved through those literals."""

    strings = _module_strings()
    assign = next(
        node for node in TREE.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "_POOL_PROBE_DEBUG_KEYS"
    )
    return tuple(strings[element.id] for element in assign.value.elts)


_PROBE_KEYS = _module_key_tuple()

_SEAM = (
    "_pool_probe_debug",
    "_debug_flag_enabled",
    "_new_pool_epilogue_tracer",
    "_new_pool_kernel_dump_recorder",
    "_pool_pad_poison_enabled",
    "_new_pool_offload_check_recorder",
)
_NAMESPACE = {
    "logging": logging,
    "_POOL_PROBE_DEBUG_KEYS": _PROBE_KEYS,
}
exec(compile(ast.fix_missing_locations(ast.Module(
    body=[copy.deepcopy(_method(name)) for name in _SEAM],
    type_ignores=[],
)), "<worker-probe-seam>", "exec"), _NAMESPACE)


# --------------------------------------------------------------------------- #
#  A fake worker and wave: only what the seam reads
# --------------------------------------------------------------------------- #
class _GlobalBatch:
    """Counts every sequence lookup, which is what "walked once" means."""

    def __init__(self, sequences):
        self._sequences = sequences
        self.lookups = 0

    def get_sequence(self, uuid):
        self.lookups += 1
        return self._sequences[uuid]


class _Worker:
    def __init__(self, debug_by_uuid, *, server_debug=None, rank=3):
        self.rank = rank
        self._batchgen_debug = server_debug
        self.global_batch = _GlobalBatch({
            uuid: SimpleNamespace(batchgen_debug=debug)
            for uuid, debug in debug_by_uuid.items()
        })

    for _name in _SEAM:
        locals()[_name] = _NAMESPACE[_name]
    del _name


def _wave(uuids):
    return SimpleNamespace(
        uuids=list(uuids),
        global_ids=list(range(64, 64 + len(uuids))),
        page_tokens=64,
        plan=SimpleNamespace(segments=(), chunks=()),
        probe_debug=None,
        probe_debug_read=False,
    )


def _setups(worker, wave):
    """Every probe setup the pooled wave runs, in the order it runs them."""

    return (
        worker._new_pool_kernel_dump_recorder(wave),
        worker._pool_pad_poison_enabled(wave),
        worker._new_pool_epilogue_tracer(wave),
        worker._new_pool_offload_check_recorder(wave),
    )


def _with_import_log(call):
    """Run ``call``, returning (result, probe modules it imported)."""

    imported: list[str] = []
    real = builtins.__import__

    def counting(name, *args, **kwargs):
        if name.startswith("batchgen.prefix_reuse"):
            imported.append(name)
        return real(name, *args, **kwargs)

    builtins.__import__ = counting
    try:
        return call(), imported
    finally:
        builtins.__import__ = real


_EIGHT = tuple(f"u{index}" for index in range(8))


# --------------------------------------------------------------------------- #
#  (1) Off is free
# --------------------------------------------------------------------------- #
def test_a_wave_that_armed_nothing_walks_its_sequences_once_and_imports_nothing():
    worker = _Worker({uuid: None for uuid in _EIGHT})
    wave = _wave(_EIGHT)

    results, imported = _with_import_log(lambda: _setups(worker, wave))

    assert results == (None, False, None, None)
    # ONE pass over the eight members for all four probes, not four passes.
    assert worker.global_batch.lookups == len(_EIGHT)
    assert imported == []
    # No dict was built for the answer: None IS the answer, and it is cached.
    assert wave.probe_debug is None and wave.probe_debug_read is True


def test_unrelated_debug_keys_leave_every_pooled_probe_off():
    """A batch debugging something else must not arm a pooled probe."""

    worker = _Worker({
        uuid: {"glm5_dispatch_trace": True, "k3_prefill_profile": 1}
        for uuid in _EIGHT
    })
    wave = _wave(_EIGHT)

    results, imported = _with_import_log(lambda: _setups(worker, wave))

    assert results == (None, False, None, None)
    assert imported == []
    assert wave.probe_debug is None


def test_the_cached_answer_is_read_once_however_many_probes_ask():
    worker = _Worker({
        uuid: ({"prefix_pool_offload_check": True} if uuid == "u5" else None)
        for uuid in _EIGHT
    })
    wave = _wave(_EIGHT)

    _setups(worker, wave)
    after_first_pass = worker.global_batch.lookups
    _setups(worker, wave)

    assert after_first_pass == len(_EIGHT)
    assert worker.global_batch.lookups == len(_EIGHT), "the wave was re-walked"
    assert wave.probe_debug == {"prefix_pool_offload_check": True}


def test_only_the_pooled_probe_keys_are_kept_and_the_first_member_wins():
    worker = _Worker({
        "u0": {"prefix_pool_pad_poison": True, "glm5_dispatch_trace": True},
        "u1": {"prefix_pool_pad_poison": False, "prefix_pool_dump_dir": "/d"},
    })
    wave = _wave(("u0", "u1"))

    assert worker._pool_probe_debug(wave) == {
        "prefix_pool_pad_poison": True, "prefix_pool_dump_dir": "/d",
    }
    assert worker._pool_pad_poison_enabled(wave) is True


def test_the_server_level_debug_wins_outright_and_reads_no_sequence():
    worker = _Worker(
        {uuid: None for uuid in _EIGHT},
        server_debug={"prefix_pool_pad_poison": "on", "unrelated": 1},
    )
    wave = _wave(_EIGHT)

    assert worker._pool_probe_debug(wave) == {"prefix_pool_pad_poison": "on"}
    assert worker.global_batch.lookups == 0
    assert worker._pool_pad_poison_enabled(wave) is True


def test_a_malformed_probe_flag_costs_the_probe_and_not_the_prefill(caplog):
    """The parse refuses the value; the prefill still gets its tracer slot."""

    worker = _Worker({"u0": {"prefix_pool_epilogue_trace": "maybe"}})
    wave = _wave(("u0",))

    with caplog.at_level(logging.WARNING):
        assert worker._new_pool_epilogue_tracer(wave) is None

    line = next(
        record.getMessage() for record in caplog.records
        if "PROBE_ERROR" in record.getMessage()
    )
    assert "probe=epilogue_trace.setup" in line and "rank=3" in line
    assert "prefix_pool_epilogue_trace" in line


def test_the_worker_key_list_is_exactly_the_probe_modules_own_flags():
    """The literals that keep the off-path import-free cannot drift."""

    from batchgen.prefix_reuse import executor, kernel_dump, offload_check

    assert set(_PROBE_KEYS) == {
        _TRACE.DEBUG_FLAG,
        _TRACE.PROBE_FLAG,
        offload_check.DEBUG_FLAG,
        kernel_dump.DEBUG_FLAG,
        kernel_dump.DUMP_DIR_FLAG,
        executor.PAD_POISON_DEBUG_FLAG,
    }
    assert len(_PROBE_KEYS) == len(set(_PROBE_KEYS))


def test_the_wave_carries_the_once_per_wave_cache():
    declared = {field.name: field.default for field in fields(PoolWave)}

    assert declared["probe_debug"] is None
    assert declared["probe_debug_read"] is False


# --------------------------------------------------------------------------- #
#  (2) The gather record is never hand-written where it could be wrong
# --------------------------------------------------------------------------- #
def _gather_calls(function_name: str) -> list:
    return [
        node for node in ast.walk(_method(function_name))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "gather_last_token_hidden"
    ]


def _record_gather_calls(function_name: str) -> list:
    return [
        node for node in ast.walk(_method(function_name))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "record_gather"
    ]


@pytest.mark.parametrize(
    "function_name", ["_prefill_prepacked_pool", "prefill_prepacked"]
)
def test_both_prefill_paths_forward_the_tracer_into_the_real_gather(
    function_name,
):
    calls = _gather_calls(function_name)
    assert len(calls) == 1, f"{function_name}: {len(calls)} gather calls"
    expansions = [
        ast.unparse(keyword.value) for keyword in calls[0].keywords
        if keyword.arg is None
    ]
    assert len(expansions) == 1, f"{function_name}: {expansions}"
    assert "'trace'" in expansions[0]
    assert "_trace" in expansions[0]


@pytest.mark.parametrize(
    "function_name", ["_prefill_prepacked_pool", "prefill_prepacked"]
)
def test_neither_prefill_path_hand_writes_a_gather_record(function_name):
    """A hand-written ``tp=0/1, owned`` record is only legal behind the
    asserted helper, never inline in a prefill loop."""

    assert _record_gather_calls(function_name) == []


def test_record_unsharded_gather_refuses_a_sharded_manager():
    tracer = _TRACE.EpilogueTracer(rank=0)
    manager = SimpleNamespace(
        prefill_uses_streamed_sp8=lambda: True,
        gather_prefill_last_token_hidden=lambda *a, **k: None,
    )

    with pytest.raises(RuntimeError, match="streamed-SP8"):
        _TRACE.record_unsharded_gather(
            tracer, torch.ones(2, 4), num_global_rows=2, manager=manager
        )
    assert tracer.owned is None and tracer.attn_tp_size == 1


def test_record_unsharded_gather_refuses_an_unclassifiable_manager():
    """A manager with a gather but no shard predicate is not asserted safe."""

    tracer = _TRACE.EpilogueTracer(rank=0)
    manager = SimpleNamespace(
        gather_prefill_last_token_hidden=lambda *a, **k: None
    )

    with pytest.raises(RuntimeError, match="no streamed-SP8 predicate"):
        _TRACE.record_unsharded_gather(
            tracer, torch.ones(2, 4), num_global_rows=2, manager=manager
        )


def test_record_unsharded_gather_accepts_a_manager_with_no_gather_at_all():
    tracer = _TRACE.EpilogueTracer(rank=0)
    hidden = torch.tensor([[3.0, 4.0], [0.0, 0.0]])

    _TRACE.record_unsharded_gather(
        tracer, hidden, num_global_rows=7, manager=SimpleNamespace()
    )

    assert (tracer.attn_tp_rank, tracer.attn_tp_size) == (0, 1)
    assert tracer.shard == (0, 7) and tracer.owned == [True, True]
    assert tracer.pre_norms == pytest.approx([5.0, 0.0])
    assert tracer.post_norms == tracer.pre_norms


def test_the_real_gather_records_through_the_tracer_on_its_unsharded_branch():
    """The production function, not a transcription of it."""

    from batchgen.models.moonshotai.kimi_linear.Parallel_Strategy_Manager import (
        KimiLinearParallelStrategyManager,
    )

    tracer = _TRACE.EpilogueTracer(rank=0)
    manager = SimpleNamespace(
        prefill_uses_streamed_sp8=lambda: False,
        global_rank=0,
        gather_prefill_last_token_hidden=(
            KimiLinearParallelStrategyManager.gather_prefill_last_token_hidden
        ),
    )
    hidden = torch.arange(12.0).reshape(1, 6, 2)

    selected = manager.gather_prefill_last_token_hidden(
        manager, hidden, torch.tensor([1, 5]), 6, trace=tracer
    )

    assert torch.equal(selected, hidden[0, torch.tensor([1, 5]), :])
    # The REAL rows the gather kept, recorded from inside it.
    assert (tracer.attn_tp_rank, tracer.attn_tp_size) == (0, 1)
    assert tracer.shard == (0, 6) and tracer.owned == [True, True]
    assert tracer.pre_norms == pytest.approx(
        hidden[0, torch.tensor([1, 5]), :].norm(dim=-1).tolist()
    )


def test_the_real_gather_is_unchanged_when_no_tracer_is_passed():
    from batchgen.models.moonshotai.kimi_linear.Parallel_Strategy_Manager import (
        KimiLinearParallelStrategyManager,
    )

    manager = SimpleNamespace(
        prefill_uses_streamed_sp8=lambda: False, global_rank=0
    )
    hidden = torch.arange(12.0).reshape(1, 6, 2)

    selected = KimiLinearParallelStrategyManager.gather_prefill_last_token_hidden(
        manager, hidden, torch.tensor([0, 3]), 6
    )

    assert torch.equal(selected, hidden[0, torch.tensor([0, 3]), :])
