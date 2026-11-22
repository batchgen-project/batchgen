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
`admission/plan.py`, following calls by method name. The whole-package
closure test at the end widens that to every module under `batchgen/`,
resolving calls by name (an over-approximation) and pinning every status
reader the carry-out can reach.
"""

from __future__ import annotations

import ast
import copy
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
def _worker_function(name: str) -> ast.FunctionDef:
    return next(
        node for file_name, node in _INDEX[name] if file_name == "batchgen_worker.py"
    )


_EXECUTOR = _ROOT / "batchgen" / "worker" / "admission" / "boundary_executor.py"


def _executor_apply() -> ast.FunctionDef:
    tree = ast.parse(_EXECUTOR.read_text(), filename=str(_EXECUTOR))
    executor = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BoundaryExecutor"
    )
    return next(
        node for node in executor.body
        if isinstance(node, ast.FunctionDef) and node.name == "apply"
    )


def _boundary_with_carry_out_inlined() -> ast.FunctionDef:
    """The page boundary as one function again, in execution order.

    The carry-out of the decisions is ``BoundaryExecutor.apply``; the boundary
    reaches it through the worker's ``_apply_boundary_decisions``, which it
    calls once. The facts
    below are about the order of statements across the whole boundary, so the
    carry-out's statements are put back where the call is, with line numbers
    shifted so that source order equals execution order: everything before
    the call, then the carry-out, then everything after it.
    """
    boundary = copy.deepcopy(_worker_function("_page_boundary_fast"))
    carry_out = copy.deepcopy(_executor_apply())
    calls = [
        index for index, statement in enumerate(boundary.body)
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_apply_boundary_decisions"
            for node in ast.walk(statement)
        )
    ]
    assert len(calls) == 1, "the boundary calls its carry-out exactly once"
    at = calls[0]
    # The statement after the call returns early when no decode row is left;
    # in the carry-out that is its own early return, so it is dropped here.
    after = boundary.body[at + 2:]
    for offset, statements in ((1_000_000, carry_out.body), (2_000_000, after)):
        for statement in statements:
            for node in ast.walk(statement):
                if hasattr(node, "lineno"):
                    node.lineno += offset
                if getattr(node, "end_lineno", None) is not None:
                    node.end_lineno += offset
    boundary.body = boundary.body[:at] + carry_out.body + after
    return boundary


_BOUNDARY = _boundary_with_carry_out_inlined()


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


# The executor reaches the completion outlet through its CompletionIO argument;
# these are the worker methods the caller binds to its two fields.
_COMPLETION_OUTLET = {
    "gather_completed_tokens": "_gather_completed_tokens",
    "report_completion": "_report_completion",
}


def _status_writes() -> List[Tuple[int, str]]:
    writes = []
    # update_batch_status(pool, rank, uuids, status, edge)
    for call in _calls(_BOUNDARY, "update_batch_status"):
        writes.append((call.lineno, ast.unparse(call.args[3]).split(".")[-1]))
    for call in _calls(_BOUNDARY, "apply_or_raise"):
        status = next(
            n.attr for n in ast.walk(call)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == "SequenceStatus"
        )
        writes.append((call.lineno, status))
    return sorted(writes)


def test_planned_statuses_apply_at_one_point_then_at_most_the_amendment():
    """All PLANNED status changes apply at one point; one amendment later.

    The decision is a partition (Part A), so the three planned groups are
    known when the decision arrives: COMPLETED through `update_batch_status`
    (rows the per-step sync already completed must be skipped), then EVICTED
    and ON_HOLD through `apply_or_raise` with the boundary edge, with no other
    call between them. The only later status write is the ON_HOLD amendment
    for rows whose GPU page extension failed.
    """
    writes = _status_writes()
    assert [s for _, s in writes] == [
        "COMPLETED", "EVICTED", "ON_HOLD", "ON_HOLD",
    ]

    # One point: between the first and third write, the only calls are the
    # write mechanism itself.
    lo, hi = writes[0][0], writes[2][0]
    mechanism = {"update_batch_status", "apply_or_raise",
                 "TransitionPlan", "Transition", "tuple"}
    for node in ast.walk(_BOUNDARY):
        if (isinstance(node, ast.Call)
                and getattr(node, "lineno", None) is not None
                and lo <= node.lineno <= hi):
            name = (node.func.attr if isinstance(node.func, ast.Attribute)
                    else node.func.id if isinstance(node.func, ast.Name)
                    else ast.unparse(node.func))
            assert name in mechanism, (node.lineno, name)

    # The fourth write is the extension-failure amendment: it sits under the
    # branch taken when the collective extension did not succeed.
    parent = {c: p for p in ast.walk(_BOUNDARY) for c in ast.iter_child_nodes(p)}
    amendment = _calls(_BOUNDARY, "apply_or_raise")[-1]
    assert amendment.lineno == writes[3][0]
    guards = []
    node = amendment
    while node in parent:
        node = parent[node]
        if isinstance(node, ast.If):
            guards.append(ast.unparse(node.test))
    assert any("local_extension_success" in g for g in guards), guards


def test_the_decode_index_snapshot_precedes_every_status_write():
    """The completion release depends on the live IN_DECODE index.

    Rows can be IN_DECODE without being in this call's decode set, so the
    release's remaining rows must come from the index — read BEFORE any of
    the boundary's status writes (the planned writes move evicted and
    on-hold rows out of IN_DECODE) and narrowed only by the completed rows,
    which is exactly what a live read returned when only the completion
    update had been applied.
    """
    writes = _status_writes()
    snapshots = [
        c for c in _calls(_BOUNDARY, "get_sequences_by_status")
        if "IN_DECODE" in ast.unparse(c)
    ]
    assert len(snapshots) == 1
    assert snapshots[0].lineno < writes[0][0]

    release = _calls(_BOUNDARY, "_release_host_kv_pages_for_batch")[0]
    remaining = next(
        k.value for k in release.keywords if k.arg == "remaining_decode_uuids"
    )
    assert "in_decode_before" in ast.unparse(remaining)
    assert "completed_set" in ast.unparse(remaining)


def test_no_step_between_the_status_writes_acts_on_a_status():
    """Between the first write and the amendment, status is read twice.

    Between the first status write and the last (the amendment), the boundary
    and everything it calls read a status in exactly two places, and neither
    is affected by the planned EVICTED and ON_HOLD writes applying at the
    front: one counts COMPLETED rows, one prints a status in a log line. The
    release's GPU page-table rebuild depends on the live IN_DECODE index, so
    the boundary snapshots that index BEFORE its first status write and the
    release takes the remaining rows as an argument instead of reading status
    itself (pinned by test_the_decode_index_snapshot_precedes_every_status_write).
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
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            if n.func.attr.startswith("_"):
                follow(n.func.attr)
            elif n.func.attr in _COMPLETION_OUTLET:
                # completion.report_completion(...) is the worker's
                # _report_completion, handed to the executor by the caller.
                follow(_COMPLETION_OUTLET[n.func.attr])

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

    assert inside("_mirror_group_host_page_tables") == [True]
    assert inside("_rebuild_page_table_for_batch") == [False, False, False]
    assert inside("_check_host_kv_watermark_trigger") == [False]


def test_the_boundary_mirror_copies_page_tables_and_nothing_else():
    """The boundary's mirror is its own function, limited to page tables.

    Prefill keeps `_mirror_group_host_kv` (prefix states, pool segments,
    trajectory arenas). After a decode-time host growth the followers only
    need the leader's grown page tables, so the boundary calls
    `_mirror_group_host_page_tables`: one collective, built from the decode
    context, with no prefix, pool-wave or trajectory state in it.
    """
    assert _calls(_BOUNDARY, "_mirror_group_host_kv") == []
    call = _calls(_BOUNDARY, "_mirror_group_host_page_tables")[0]
    assert len(call.args) == 2 and not call.keywords
    assert ast.unparse(call.args[0]) == "ctx"

    mirror = next(node for name, node in _INDEX["_mirror_group_host_page_tables"]
                  if name == "kv_capacity_engine.py")
    assert [a.arg for a in mirror.args.args] == ["self", "ctx", "uuids"]
    collectives = [
        n for n in ast.walk(mirror)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr in {"all_gather_object", "broadcast_object_list",
                            "all_reduce", "all_gather", "broadcast"}
    ]
    assert [c.func.attr for c in collectives] == ["all_gather_object"]
    body = ast.unparse(ast.Module(body=mirror.body[1:], type_ignores=[]))
    for foreign in ("prefix", "trajectory", "_traj", "segment", "wave"):
        assert foreign not in body, foreign


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


# ------------------------------------------- Part B: whole-package closure

_PKG = _ROOT / "batchgen"

#: Names never defined in `batchgen/` (builtins, stdlib, torch, logging).
#: Kept small on purpose: a name NOT listed here that happens to match a
#: `batchgen` def is walked anyway (the closure over-approximates).
_SKIP = set("""append items keys values get set list dict len sorted sum max
min int str bool float tuple zip enumerate isinstance getattr setattr hasattr
range add discard pop update join split format warning info debug error
exception perf_counter tensor item barrier all_reduce all_gather_object
broadcast_object_list synchronize RuntimeError ValueError KeyError TypeError
Transition TransitionPlan BoundaryDecisions next iter startswith endswith copy
extend remove clear count wait print type repr abs any all ceil""".split())

#: Known name collisions: the carry-out's call resolves to exactly this file.
#: `update_batch_status` is imported into `batchgen_worker.py` from the
#: admission package; the server-process `BatchStorage` method of the same
#: name is a different object on a different process.
_PINNED = {"update_batch_status": "batchgen/worker/admission/plan.py"}

#: Status-keyed accessors and status-dependent methods of
#: `SequenceEntry`/`SequenceBatch`; a call to one is a status read.
_STATUS_METHODS = {
    "get_sequences_by_status", "status_transition", "plan_transitions",
    "count_by_status", "get_sequences_for_rank_with_status",
    "get_resumable_sequences", "get_resumable_sequences_for_rank",
    "has_queueing", "has_prefilled", "has_in_decode", "has_on_hold",
    "has_resumable", "has_evicted", "all_completed", "is_finished",
    "is_active_in_decode", "is_resumable", "validate_metadata",
    "get_my_sequences_by_status",
}


def _package_index() -> Dict[str, List[Tuple[str, ast.AST]]]:
    index: Dict[str, List[Tuple[str, ast.AST]]] = {}
    for path in sorted(_PKG.rglob("*.py")):
        rel = str(path.relative_to(_ROOT))
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                index.setdefault(node.name, []).append((rel, node))
    return index


def _package_defs(index, name):
    # BoundaryExecutor.apply is the carry-out itself (the root of the closure),
    # not a callee: `apply` by name means SequenceBatch.apply here.
    defs = [
        (rel, node) for rel, node in index.get(name, [])
        if not (name == "apply" and rel.endswith("admission/boundary_executor.py"))
    ]
    if name in _PINNED:
        defs = [(rel, node) for rel, node in defs if rel == _PINNED[name]]
    return defs


def _called_names(fn_node) -> List[str]:
    """Called names in ``fn_node``."""
    names = []
    for call in ast.walk(fn_node):
        if not isinstance(call, ast.Call):
            continue
        if isinstance(call.func, ast.Attribute):
            names.append(call.func.attr)
        elif isinstance(call.func, ast.Name):
            names.append(call.func.id)
    return names


def _package_status_reads(node) -> Set[str]:
    reads = set()
    for n in ast.walk(node):
        if (isinstance(n, ast.Attribute) and n.attr in ("status", "_status_index")
                and isinstance(n.ctx, ast.Load)):
            reads.add(n.attr)
        elif isinstance(n, ast.Call):
            name = (n.func.attr if isinstance(n.func, ast.Attribute)
                    else n.func.id if isinstance(n.func, ast.Name) else None)
            if name in _STATUS_METHODS:
                reads.add(f"call:{name}")
    return reads


def test_whole_package_closure_reads_status_only_in_pinned_places():
    """Every status read reachable from the carry-out, package-wide.

    From the single planned write point to the amendment, the boundary's
    carry-out and everything it can reach across all of `batchgen/` —
    following calls by NAME, so helper classes (`SequenceEntry`,
    `SequenceBatch`, the index helpers, `decode_dp_group`, the trajectory
    book, the host/GPU KV managers, prefix-reuse releases, the group mirror)
    are all included, and unrelated same-named defs over-approximate the
    closure — reads a sequence status only through the write mechanism
    itself, two COMPLETED-count checks, and one debug log line. None of
    these can change when the planned EVICTED and ON_HOLD rows already carry
    their new status, because none distinguishes IN_DECODE from EVICTED or
    ON_HOLD:

    * the mechanism reads the status of its own target rows at apply time;
    * `get_sequences_by_status(COMPLETED)` and `all_completed()` count only
      COMPLETED rows, and the planned rows are non-COMPLETED under either
      ordering;
    * the `[LOCALMAP-POP]` debug line prints the status of a row the same
      carry-out just completed.

    `to_json` is an over-approximation artifact: the generic name matches
    `LaunchReadinessReport.to_json`, which reads that report's own `status`
    field, not a sequence's.
    """
    index = _package_index()
    writes = _status_writes()
    lo, hi = writes[0][0], writes[-1][0]

    seen: Set[str] = set()
    queue: List[str] = []
    for node in ast.walk(_BOUNDARY):
        if (isinstance(node, ast.Call)
                and getattr(node, "lineno", None) is not None
                and lo <= node.lineno <= hi):
            name = (node.func.attr if isinstance(node.func, ast.Attribute)
                    else node.func.id if isinstance(node.func, ast.Name)
                    else None)
            # The completion outlet is reached through the CompletionIO
            # argument; follow the worker methods bound to it.
            name = _COMPLETION_OUTLET.get(name, name)
            if name and name not in _SKIP and name not in seen:
                seen.add(name)
                queue.append(name)

    while queue:
        name = queue.pop()
        if name == "_page_boundary_fast":   # the carry-out is the root
            continue
        for _, node in _package_defs(index, name):
            for called in _called_names(node):
                if called not in _SKIP and called not in seen:
                    seen.add(called)
                    queue.append(called)

    # The closure must actually cover the package, not fizzle at the seeds.
    files = {rel for name in seen for rel, _ in _package_defs(index, name)}
    assert len(files) > 50, sorted(files)

    readers: Dict[str, Set[str]] = {}
    for name in sorted(seen):
        for _, node in _package_defs(index, name):
            reads = _package_status_reads(node)
            if reads:
                readers.setdefault(name, set()).update(reads)

    assert readers == {
        # the write mechanism itself
        "update_batch_status": {"call:plan_transitions"},
        "plan_transitions": {"status"},
        "apply": {"_status_index", "call:status_transition", "status"},
        "status_transition": {"status"},
        # COMPLETED-count checks (planned rows are non-COMPLETED either way)
        "get_sequences_by_status": {"_status_index"},
        "all_completed": {"_status_index"},
        "_maybe_emit_prefix_drain_metric": {"call:all_completed"},
        # one debug log line on a just-completed row
        "_release_completed_slot_index": {"status"},
        # name-collision artifact: LaunchReadinessReport's own field
        "to_json": {"status"},
    }


def test_the_carry_out_is_one_method_with_a_declared_result():
    """The carry-out is ``BoundaryExecutor.apply``; the worker only wires it.

    ``apply`` takes the decode context and the completion outlet, never the
    worker. It ends in exactly two places: early, when no decode row is left
    after the releases and host growth, and at the end. Both return the same
    eight values; the seventh tells the boundary which of the two it was and
    the eighth is the pointer bundle of a load launched here, or None.
    """
    carry_out = _executor_apply()
    assert [arg.arg for arg in carry_out.args.args] == [
        "self", "ctx", "completion", "decisions", "decode_uuids", "batch",
        "gpu_manager", "worker_view", "global_seq_state", "chunk_size",
        "timing", "boundary_start", "t0",
    ]
    names = {node.id for node in ast.walk(carry_out) if isinstance(node, ast.Name)}
    assert "worker" not in names
    assert not any(
        isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        for node in ast.walk(carry_out)
    ), "the executor is stateless and holds no worker"
    returns = [node for node in ast.walk(carry_out) if isinstance(node, ast.Return)]
    assert len(returns) == 2
    for node in returns:
        assert isinstance(node.value, ast.Tuple) and len(node.value.elts) == 8
    assert sorted(ast.unparse(node.value.elts[6]) for node in returns) == ["False", "True"]
    assert sorted(ast.unparse(node.value.elts[7]) for node in returns) == [
        "None", "async_load_tensors",
    ]

    # The worker's method builds the context and the outlet, calls apply once,
    # and keeps the pointer bundle.
    wiring = _worker_function("_apply_boundary_decisions")
    applies = _calls(wiring, "apply")
    assert len(applies) == 1
    assert ast.unparse(applies[0].args[0]) == "self._decode_phase_context()"
    outlet = applies[0].args[1]
    assert {kw.arg: ast.unparse(kw.value) for kw in outlet.keywords} == {
        "gather_completed_tokens": "self._gather_completed_tokens",
        "report_completion": "self._report_completion",
    }
    assert "self._async_load_tensors = async_load_tensors" in ast.unparse(wiring)

    boundary = _worker_function("_page_boundary_fast")
    assert len(_calls(boundary, "_apply_boundary_decisions")) == 1
