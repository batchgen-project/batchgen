"""Audit of the page-boundary flow: which orderings are real dependencies.

The boundary settles the previous boundary's async work, gathers a snapshot,
lets rank 0 decide, and then carries the decision out. The question audited
here is whether the carry-out step has dependencies that force it to be
split, or to change sequence status at several points.

Two kinds of evidence, both CPU-only:

* Part A drives the real rank-0 decision (`BoundaryHandler.compute_decisions`)
  and shows, from its page arithmetic, which acquisitions are paid for by
  which releases. Those are the orderings no refactor can remove.
* Part B reads the real source of `_page_boundary_fast` and pins the facts
  that are only incidental: where status is written, who reads status between
  those writes, and where the worker-only calls sit.

Part B covers `batchgen_worker.py`, `batchgen/worker/*_engine.py` and
`admission/plan.py`, following calls by method name. Helper classes outside
those files are not followed.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Dict, List, Set, Tuple

from batchgen.worker.boundary import (
    BoundaryDecisionRequest,
    BoundaryHandler,
    BoundarySeqMeta,
)

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_GPN = 8


# ---------------------------------------------------------------- Part A

def _state(*, rank=0, completed=False, decoded=10, extra_gpu=0, gpu_pages=4,
           host_pages=2, growth=0):
    return {
        "completed": completed,
        "assigned_rank": rank,
        "decode_dp_group": None,
        "decoded_length": decoded,
        "additional_pages_needed": extra_gpu,
        "gpu_pages_allocated": gpu_pages,
        "host_pages_allocated": host_pages,
        "host_owned_pages": host_pages,
        "needs_host_growth": growth > 0,
        "host_growth_pages": growth,
        "current_context_length": 100,
        "host_token_capacity": 4096,
    }


def _decide(state, *, gpu_free, host_free=None, candidates=None, evict=False):
    uuids = list(state)
    stats = None
    if host_free is not None:
        stats = ({"node_id": 0, "num_total_pages": 1000,
                  "num_free_pages": host_free, "num_reclaimable_pages": 0},)
    return BoundaryHandler.compute_decisions(BoundaryDecisionRequest(
        decode_uuids=tuple(uuids),
        global_seq_state=state,
        global_candidate_info=candidates or {},
        per_rank_free=(gpu_free,),
        chunk_size=64,
        per_node_host_stats=stats,
        seq_meta={
            u: BoundarySeqMeta(i, 0, 100, 4096, state[u]["host_pages_allocated"])
            for i, u in enumerate(uuids)
        },
        world_size=1,
        num_gpus_per_node=_GPN,
        enable_host_kv_eviction=evict,
    ))


def test_every_decode_row_gets_exactly_one_outcome():
    """The plan is a partition, so all status changes are known at broadcast."""
    state = {
        "done": _state(completed=True),
        "grow": _state(growth=1),
        "ext_a": _state(extra_gpu=2, decoded=1),
        "ext_b": _state(extra_gpu=2, decoded=2),
        "idle": _state(),
    }
    plan = _decide(state, gpu_free=1, host_free=0, evict=True)
    groups = [plan.completed_uuids, plan.host_evicted_uuids,
              plan.onhold_uuids, plan.decode_uuids_final]
    flat = [u for g in groups for u in g]
    assert sorted(flat) == sorted(state)          # covers every row once
    assert len(flat) == len(set(flat))
    assert set(plan.host_growth_uuids) <= set(plan.decode_uuids_final)
    survivors_extending = set(plan.seqs_needing_extension) - set(plan.onhold_uuids)
    assert survivors_extending <= set(plan.decode_uuids_final)


def test_host_growth_is_paid_for_by_completed_rows_pages():
    """Real dependency: completed rows release host pages before growth."""
    state = {"done": _state(completed=True, host_pages=2), "grow": _state(growth=2)}
    plan = _decide(state, gpu_free=100, host_free=0)
    assert plan.growth_feasible and plan.host_growth_uuids == ["grow"]
    assert plan.host_evicted_uuids == []
    # Zero pages were free at the snapshot; the growth can only succeed after
    # the completed row's two pages are back in the pool.
    assert sum(plan.host_growth_pages) == 2


def test_host_growth_is_paid_for_by_evicted_rows_pages():
    """Real dependency: evicted rows release host pages before growth."""
    state = {"grow": _state(growth=1, host_pages=1, decoded=50),
             "victim": _state(host_pages=8, decoded=1)}
    plan = _decide(state, gpu_free=100, host_free=0, evict=True)
    assert plan.host_evicted_uuids == ["victim"]
    assert plan.growth_feasible and plan.host_growth_uuids == ["grow"]


def test_gpu_extension_is_paid_for_by_on_hold_rows_pages():
    """Real dependency: on-hold rows free GPU pages before the extension."""
    state = {"a": _state(extra_gpu=2, gpu_pages=4, decoded=1),
             "b": _state(extra_gpu=2, gpu_pages=4, decoded=2)}
    plan = _decide(state, gpu_free=1)
    assert plan.onhold_uuids == ["a"]
    extending = [u for u in plan.seqs_needing_extension if u not in plan.onhold_uuids]
    need = sum(state[u]["additional_pages_needed"] for u in extending)
    freed = sum(state[u]["gpu_pages_allocated"] for u in plan.onhold_uuids)
    assert need > 1                # more than was free at the snapshot
    assert need <= 1 + freed       # covered once the on-hold row's pages return


def test_gpu_extension_and_load_do_not_use_completed_rows_gpu_pages():
    """Not a dependency: the GPU budget ignores pages completed rows free."""
    candidates = {"waiting": {"pages_needed": 2, "assigned_rank": 0,
                              "decode_dp_group": None, "status": "PREFILLED",
                              "decoded_length": 0}}
    state = {"done": _state(completed=True, gpu_pages=4), "stay": _state()}
    plan = _decide(state, gpu_free=0, candidates=candidates)
    assert plan.new_load_uuids == []      # 4 pages are about to be freed, unused
    roomy = _decide(state, gpu_free=2, candidates=candidates)
    assert roomy.new_load_uuids == ["waiting"]


def test_load_budget_is_free_pages_minus_extension_only():
    """Loads come after extensions and never count on-hold frees."""
    candidates = {"waiting": {"pages_needed": 3, "assigned_rank": 0,
                              "decode_dp_group": None, "status": "PREFILLED",
                              "decoded_length": 0}}
    state = {"a": _state(extra_gpu=2, gpu_pages=4, decoded=1),
             "b": _state(extra_gpu=2, gpu_pages=4, decoded=2)}
    plan = _decide(state, gpu_free=1, candidates=candidates)
    # After execution 1 + 4 - 2 = 3 pages are free, enough for the candidate,
    # but the plan budgets 1 - 2 and loads nothing.
    assert plan.onhold_uuids == ["a"] and plan.new_load_uuids == []


# ---------------------------------------------------------------- Part B

def _functions() -> Dict[str, List[Tuple[str, ast.FunctionDef]]]:
    files = [_ROOT / "batchgen" / "batchgen_worker.py"]
    files += sorted((_ROOT / "batchgen" / "worker").glob("*_engine.py"))
    files.append(_ROOT / "batchgen" / "worker" / "admission" / "plan.py")
    index: Dict[str, List[Tuple[str, ast.FunctionDef]]] = {}
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.FunctionDef):
                index.setdefault(node.name, []).append((path.name, node))
    return index


_INDEX = _functions()
_BOUNDARY = next(
    node for name, node in _INDEX["_page_boundary_fast"]
    if name == "batchgen_worker.py"
)


def _calls(node: ast.AST, name: str) -> List[ast.Call]:
    return sorted(
        (n for n in ast.walk(node)
         if isinstance(n, ast.Call)
         and ((isinstance(n.func, ast.Attribute) and n.func.attr == name)
              or (isinstance(n.func, ast.Name) and n.func.id == name))),
        key=lambda n: n.lineno,
    )


def _line(name: str, which: int = 0) -> int:
    return _calls(_BOUNDARY, name)[which].lineno


def _status_reads(node: ast.AST) -> Set[Tuple[str, str]]:
    """(what is read, 'log' or 'logic') for every status read under ``node``."""
    parent = {c: p for p in ast.walk(node) for c in ast.iter_child_nodes(p)}
    found = set()
    for n in ast.walk(node):
        if (isinstance(n, ast.Attribute) and n.attr == "status"
                and isinstance(n.ctx, ast.Load)):
            what = "seq.status"
        elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr == "get_sequences_by_status"):
            what = ast.unparse(n.args[0]).split(".")[-1]
        else:
            continue
        log, p = False, n
        while p in parent:
            p = parent[p]
            if isinstance(p, ast.JoinedStr):
                log = True
        found.add((what, "log" if log else "logic"))
    return found


def _status_writes() -> List[Tuple[int, str]]:
    writes = []
    for call in _calls(_BOUNDARY, "_update_batch_status"):
        writes.append((call.lineno, ast.unparse(call.args[1]).split(".")[-1]))
    for call in _calls(_BOUNDARY, "apply_or_raise"):
        status = next(
            n.attr for n in ast.walk(call)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == "SequenceStatus"
        )
        writes.append((call.lineno, status))
    return sorted(writes)


def test_status_is_written_four_times_in_this_order():
    assert [s for _, s in _status_writes()] == [
        "COMPLETED", "EVICTED", "ON_HOLD", "ON_HOLD",
    ]


def test_no_step_between_the_status_writes_acts_on_a_status():
    """What stands between today's four writes and one write.

    Between the first and the last status write, the boundary and everything
    it calls read a status in exactly two places, and neither would be
    affected by moving the EVICTED and ON_HOLD writes earlier: one counts
    COMPLETED rows, one prints a status in a log line. The third reader this
    test used to pin — `_release_host_kv_pages_for_batch` rebuilding the GPU
    page table from the rows still IN_DECODE — is gone: the caller now states
    those rows itself (`remaining_decode_uuids=decisions.active_uuids`), so
    the release makes no status read at all.
    """
    writes = _status_writes()
    lo, hi = writes[0][0], writes[-1][0]
    span = [n for n in ast.walk(_BOUNDARY)
            if getattr(n, "lineno", None) is not None and lo <= n.lineno <= hi]

    readers: Dict[str, Set[Tuple[str, str]]] = {}
    direct = set()
    for n in span:
        if isinstance(n, (ast.Attribute, ast.Call)):
            direct |= _status_reads(n)
    if direct:
        readers["_page_boundary_fast"] = direct

    seen: Set[str] = set()

    def follow(name: str) -> None:
        if name in seen or name not in _INDEX:
            return
        seen.add(name)
        for _, node in _INDEX[name]:
            reads = _status_reads(node)
            if reads:
                readers.setdefault(name, set()).update(reads)
            for call in ast.walk(node):
                if (isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Attribute)
                        and call.func.attr.startswith("_")):
                    follow(call.func.attr)

    for n in span:
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr.startswith("_")
                and n.func.attr != "_update_batch_status"):
            follow(n.func.attr)

    # B2 completion split: the LOCALMAP-POP log line (the one status read on
    # the completion report path) moved with the slot/index release into
    # `KVCapacityEngine._release_completed_slot_index`; `_report_completion`
    # itself no longer reads a status. Same read, new location.
    assert readers == {
        "_page_boundary_fast": {("COMPLETED", "logic")},
        "_release_completed_slot_index": {("seq.status", "log")},
    }


def test_only_the_group_mirror_sits_inside_the_carry_out_step():
    """Of the three worker-only calls, two are already outside.

    The carry-out step runs from the decision broadcast to the launch of the
    async loads. The page-table rebuild and the watermark trigger are called
    before the snapshot or after the loads, never inside. Only the mirror of
    grown host page tables to a group's followers is inside.
    """
    start = _line("broadcast_object_list")
    end = max(n.lineno for n in ast.walk(_BOUNDARY)
              if isinstance(n, ast.Attribute) and n.attr == "num_loaded")

    def inside(name: str) -> List[bool]:
        return [start < c.lineno < end for c in _calls(_BOUNDARY, name)]

    assert inside("_mirror_group_host_kv") == [True]
    assert inside("_rebuild_page_table_for_batch") == [False, False, False]
    assert inside("_check_host_kv_watermark_trigger") == [False]


def test_the_boundary_mirror_call_uses_none_of_the_trajectory_work():
    """Why the mirror is worker-only today, and why that is not inherent.

    `_mirror_group_host_kv` serves prefill and the boundary. Every use of the
    trajectory machinery in it is guarded by its `trajectory` flag, and the
    boundary calls it without the flag and with no prefix states.
    """
    call = _calls(_BOUNDARY, "_mirror_group_host_kv")[0]
    assert not call.keywords                      # trajectory stays False
    assert ast.unparse(call.args[1]) == "{}"      # no prefix states

    mirror = next(node for name, node in _INDEX["_mirror_group_host_kv"]
                  if name == "batchgen_worker.py")
    parent = {c: p for p in ast.walk(mirror) for c in ast.iter_child_nodes(p)}

    def guarded(node: ast.AST) -> bool:
        p = node
        while p in parent:
            p = parent[p]
            test = getattr(p, "test", None)
            if test is not None and "trajectory" in ast.unparse(test):
                return True
        return False

    trajectory_calls = [
        n for n in ast.walk(mirror)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and "trajectory" in n.func.attr
    ]
    assert len(trajectory_calls) == 8
    assert all(guarded(n) for n in trajectory_calls)


def test_releases_finish_on_every_rank_before_host_growth():
    """The one barrier a real dependency demands is where it should be."""
    release = _line("_release_host_kv_pages_for_batch")
    evict_release = next(
        n.lineno for n in ast.walk(_BOUNDARY)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "release_sequence_pages"
    )
    grow = next(
        n.lineno for n in ast.walk(_BOUNDARY)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "grow_pages_for_sequences"
    )
    barriers = [c.lineno for c in _calls(_BOUNDARY, "barrier")]
    assert any(max(release, evict_release) < b < grow for b in barriers)


def test_on_hold_frees_precede_extension_and_extension_precedes_load():
    free_calls = [c.lineno for c in _calls(_BOUNDARY, "free_pages_for_sequences")]
    extend = _line("_extend_gpu_kv_allocation")
    load = _line("_allocate_gpu_pages_with_host_identity")
    assert free_calls[0] < extend            # planned on-hold rows
    assert extend < free_calls[1] < load     # rows whose extension failed
