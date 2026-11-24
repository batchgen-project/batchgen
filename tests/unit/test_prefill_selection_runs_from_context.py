"""Run the prefill admission selection seam on CPU, with no live worker.

Two helpers moved off the worker into ``admission.prefill_selection``; this
file EXECUTES them from a hand-built ``PrefillPhaseContext`` — a real
``SequenceBatch`` pool, a real ``SequenceTable``, a real ``TrajectoryStore``
and a real ``PrefixReuseState`` — with no stand-in worker object anywhere.

Two prefill-admission methods stayed on the worker
(``_reject_unsatisfiable_prefill_requests``: its final capacity publication is
gated on a condition its return value does not carry;
``_prefill_admission_guard_state``: a two-line pure read). Their branches are
still CPU-only — the GPU acceptance legs run with the prefix cache off and
ample host memory — so this file executes them too, compiled from the real
worker source (``load_worker_methods``) onto a recording stub, exactly like
the other worker-method CPU tests.

Covered, one test each:

* ``_prefill_candidate_digest``: deterministic, ORDER-sensitive, and
  field-sensitive on a field the selection request reads;
* ``_prefill_admission_guard_state``: ``None`` stays ``None``; a snapshot has
  the reservation outcome appended;
* ``_reject_unsatisfiable_prefill_requests``: fits (incl. exactly-fitting),
  EVICTED exemption, unknown pool total, the full rank-0 rejection (error
  completion, trajectory release, QueryBook unbind, pool removal, capacity
  publication), the non-zero rank, the missing-sequence row and the absent
  response queue;
* ``_make_prefill_selection_request``: the G==1 snapshot, the shared-segment
  credit latch both ways, the G>1 node mapping through ``decode_dp_group``,
  the missing-group error and the EVICTED flag.

Plus the source-level contract: the moved helpers take ``ctx`` first and
never reference a worker; ``PrefillPhaseContext`` carries no callable field;
the delegators call their same-name helper with
``self._prefill_phase_context()``, and the ``_make_prefill_selection_request``
delegator resolves the shared-prefix segments BEFORE the module call, in the
original first-statement order.
"""

from __future__ import annotations

import ast
import logging
import pathlib
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import pytest

from batchgen.query_book import release_query_book_slot
from batchgen.sequence import SequenceBatch, SequenceEntry, SequenceStatus
import batchgen.sequence as sequence_module
from batchgen.worker.admission.context import (
    PrefillPhaseContext,
    RankFacts,
    SchedulingState,
)
from batchgen.worker.admission import prefill_selection
from batchgen.worker.admission.plan import (
    StatusEdge,
    Transition,
    TransitionPlan,
    apply_or_raise,
)
from batchgen.worker.prefill import PrefillCandidate
from batchgen.worker.prefix_reuse_state import PrefixReuseState
from batchgen.worker.rank_context import RankContext
from batchgen.worker.sequence_table import SequenceTable
from batchgen.worker.trajectory_store import TrajectoryStore
from tests.worker_method_source import load_worker_methods

_ROOT = pathlib.Path(__file__).resolve().parents[2]
PAGE = SequenceEntry.PAGE_SIZE

MOVED = (
    "_prefill_candidate_digest",
    "_make_prefill_selection_request",
)


# --------------------------------------------------------------------------- #
#  Fakes: recording collaborators only                                        #
# --------------------------------------------------------------------------- #


class RecordingQueue:
    """Stands in for the rank-0 response queue."""

    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


class RecordingTrajectoryBook:
    """Stands in for the unified trajectory book: slots by uuid, releases
    recorded."""

    def __init__(self, slots):
        self.slots = dict(slots)
        self.released = []

    def has_sequence(self, uuid):
        return uuid in self.slots

    def slot_for(self, uuid):
        return self.slots[uuid]

    def release(self, slot):
        self.released.append(slot)


# --------------------------------------------------------------------------- #
#  Context and stub-worker assembly                                           #
# --------------------------------------------------------------------------- #


def _add_seq(pool, uuid, global_idx, *, status=SequenceStatus.QUEUEING,
             prompt_length=PAGE, max_decode_length=PAGE,
             assigned_rank=0, decode_dp_group=None, prefill_order=None):
    seq = SequenceEntry(
        uuid=uuid,
        global_idx=global_idx,
        prompt_length=prompt_length,
        max_decode_length=max_decode_length,
    )
    pool.add_sequence(seq)
    pool.assign_rank(uuid, assigned_rank)
    walk = {
        SequenceStatus.QUEUEING: (),
        SequenceStatus.EVICTED: (
            SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED,
            SequenceStatus.IN_DECODE, SequenceStatus.EVICTED,
        ),
    }[status]
    for step in walk:
        apply_or_raise(
            pool,
            TransitionPlan(StatusEdge.DECODE_ENTRY, (Transition(uuid, step),)),
        )
    seq.decode_dp_group = decode_dp_group
    seq.prefill_order = prefill_order
    return seq


def _facts(*, rank=0, world_size=8, local_world_size=8, attn_tp_size=1):
    return RankFacts(
        rank=rank,
        world_size=world_size,
        local_rank=rank % local_world_size,
        local_world_size=local_world_size,
        rank_ctx=RankContext(
            rank=rank, world_size=world_size, local_rank=rank % local_world_size,
            local_world_size=local_world_size, decode_group_size=attn_tp_size,
        ),
        attn_tp_size=attn_tp_size,
        device="cpu",
        page_size=PAGE,
        decision_interval=8,
        enable_prefix_cache=False,
        enable_host_kv_eviction=False,
        host_kv_chunk_size=PAGE,
    )


def _ctx(pool, *, prefix=None, **facts_kwargs):
    return PrefillPhaseContext(
        facts=_facts(**facts_kwargs),
        sched=SchedulingState(
            pool=pool, seq_table=SequenceTable(), trajectories=TrajectoryStore()
        ),
        prefix=prefix if prefix is not None else PrefixReuseState(),
    )


# The two worker-bodied methods, compiled from the real worker source onto a
# recording stub (no live worker).
_WORKER_METHODS = load_worker_methods(
    ["_reject_unsatisfiable_prefill_requests", "_prefill_admission_guard_state"],
    {
        "List": List, "Dict": Dict, "Tuple": Tuple, "Optional": Optional,
        "logging": logging,
        "release_query_book_slot": release_query_book_slot,
    },
)


def _stub_worker(pool, *, rank=0, table=None, query_book=None,
                 response_queue=None, trajectory_book=None,
                 admission_state=None, reservation_outcome=None):
    traj = TrajectoryStore()
    traj.trajectory_book = trajectory_book
    worker = SimpleNamespace(
        rank=rank,
        global_batch=pool,
        seq_table=table if table is not None else SequenceTable(),
        query_book=query_book if query_book is not None else {},
        _traj=traj,
        _response_queue=response_queue,
        _prefill_admission_state=admission_state,
        _prefill_reservation_outcome=reservation_outcome,
        publishes=[],
    )
    worker._publish_trajectory_pool_capacity = (
        lambda: worker.publishes.append(1)
    )
    for name, function in _WORKER_METHODS.items():
        setattr(worker, name, function.__get__(worker))
    return worker


def _candidate(uuid, *, node_id=0, is_evicted=False):
    return PrefillCandidate(
        uuid=uuid,
        assigned_rank=0,
        node_id=node_id,
        is_evicted=is_evicted,
        global_idx=0,
        total_decoded_before_eviction=0,
        prompt_length=PAGE,
        kv_token_budget=2 * PAGE,
        page_size=PAGE,
    )


def _selection(*candidates):
    return SimpleNamespace(candidates=tuple(candidates))


# --------------------------------------------------------------------------- #
#  1. Candidate digest (moved; runs from the context)                         #
# --------------------------------------------------------------------------- #


def test_candidate_digest_is_deterministic_order_and_field_sensitive():
    pool = SequenceBatch()
    _add_seq(pool, "a", 0)
    _add_seq(pool, "b", 1)
    ctx = _ctx(pool)

    first = prefill_selection._prefill_candidate_digest(ctx, ["a", "b"])
    assert first == prefill_selection._prefill_candidate_digest(ctx, ["a", "b"])
    assert isinstance(first, int)

    # ORDER is part of the digest: a rank enumerating the same candidates in
    # a different order must be named by the admission gather.
    assert first != prefill_selection._prefill_candidate_digest(ctx, ["b", "a"])

    # A field the selection request reads (prefill_order decides the queueing
    # admission order) must change the digest.
    pool.get_sequence("a").prefill_order = 7
    assert first != prefill_selection._prefill_candidate_digest(ctx, ["a", "b"])


# --------------------------------------------------------------------------- #
#  2. Guard state (worker-bodied; compiled from the real source)              #
# --------------------------------------------------------------------------- #


def test_guard_state_none_resets_and_snapshot_carries_the_outcome():
    pool = SequenceBatch()
    worker = _stub_worker(pool)
    assert worker._prefill_admission_guard_state() is None

    worker = _stub_worker(
        pool,
        admission_state=(1, 2, 3),
        reservation_outcome="rejected_by_capacity_snapshot",
    )
    assert worker._prefill_admission_guard_state() == (
        1, 2, 3, "rejected_by_capacity_snapshot",
    )
    # "not attempted" is a state of its own: the outcome slot is None, which
    # is still distinct from the no-candidates None above.
    worker = _stub_worker(pool, admission_state=(1, 2, 3))
    assert worker._prefill_admission_guard_state() == (1, 2, 3, None)


# --------------------------------------------------------------------------- #
#  3. Unsatisfiable-request rejection (worker-bodied; compiled from source)   #
# --------------------------------------------------------------------------- #


def test_fitting_requests_are_kept_and_nothing_is_published():
    pool = SequenceBatch()
    _add_seq(pool, "fits", 0)
    _add_seq(pool, "exact", 1)
    worker = _stub_worker(pool)

    rejected = worker._reject_unsatisfiable_prefill_requests(
        _selection(_candidate("fits"), _candidate("exact")),
        [100],
        # An exactly-fitting reservation (needed == total) is NOT impossible:
        # the strict > is what spares it.
        {"fits": 10, "exact": 100},
    )

    assert rejected == []
    assert pool.get_sequence("fits") is not None
    assert pool.get_sequence("exact") is not None
    assert worker.publishes == []


def test_evicted_and_unknown_pool_candidates_are_never_rejected():
    pool = SequenceBatch()
    _add_seq(pool, "evicted", 0, status=SequenceStatus.EVICTED)
    _add_seq(pool, "unknown-node", 1)
    worker = _stub_worker(pool)

    rejected = worker._reject_unsatisfiable_prefill_requests(
        _selection(
            # An EVICTED sequence larger than its whole pool is an engine
            # invariant violation and must keep failing by name.
            _candidate("evicted", node_id=0, is_evicted=True),
            # Node 1 reported no total (-1): "can never fit" is undecidable.
            _candidate("unknown-node", node_id=1),
        ),
        [10, -1],
        {"evicted": 999, "unknown-node": 999},
    )

    assert rejected == []
    assert pool.get_sequence("evicted") is not None
    assert pool.get_sequence("unknown-node") is not None
    assert worker.publishes == []


def test_rank0_rejection_reports_releases_and_removes():
    pool = SequenceBatch()
    seq = _add_seq(pool, "too-big", 0)
    _add_seq(pool, "fits", 1)
    seq._buffer_slot = 5
    table = SequenceTable()
    local_idx = table.bind("too-big")
    query_book = {local_idx: "entry"}
    book = RecordingTrajectoryBook({"too-big": 5})
    queue = RecordingQueue()
    worker = _stub_worker(
        pool, table=table, query_book=query_book, response_queue=queue,
        trajectory_book=book,
    )

    rejected = worker._reject_unsatisfiable_prefill_requests(
        _selection(_candidate("too-big"), _candidate("fits")),
        [100],
        {"too-big": 101, "fits": 1},
    )

    assert rejected == ["too-big"]
    # The error completion names the same channel admission uses.
    assert [item["type"] for item in queue.items] == ["completion"]
    assert queue.items[0]["request_id"] == "too-big"
    assert queue.items[0]["error"]["code"] == "host_kv_capacity_exceeded"
    assert "101" in queue.items[0]["error"]["message"]
    # Trajectory reservation released exactly as completion releases it.
    assert book.released == [5]
    assert seq._buffer_slot == -1
    # QueryBook slot unbound and dropped.
    assert "too-big" not in table.uuid_to_local
    assert query_book == {}
    # Dropped from the pool; the fitting row stays.
    assert pool.get_sequence("too-big") is None
    assert pool.get_sequence("fits") is not None
    # The freed trajectory capacity is published once, after the loop.
    assert worker.publishes == [1]


def test_nonzero_rank_rejects_without_reporting():
    pool = SequenceBatch()
    _add_seq(pool, "too-big", 0)
    queue = RecordingQueue()
    worker = _stub_worker(pool, rank=3, response_queue=queue)

    rejected = worker._reject_unsatisfiable_prefill_requests(
        _selection(_candidate("too-big")), [100], {"too-big": 101},
    )

    assert rejected == ["too-big"]
    assert queue.items == []
    assert pool.get_sequence("too-big") is None
    assert worker.publishes == [1]


def test_missing_sequence_and_absent_queue_do_not_break_the_wave():
    pool = SequenceBatch()
    _add_seq(pool, "present", 0)
    # rank 0 with NO response queue: the rejection still releases and removes.
    worker = _stub_worker(pool, response_queue=None)

    rejected = worker._reject_unsatisfiable_prefill_requests(
        _selection(_candidate("ghost"), _candidate("present")),
        [100],
        {"ghost": 200, "present": 150},
    )

    # The ghost row (already gone from the pool) is skipped silently; the
    # present row is still rejected. The publish still runs: it is gated on
    # "some request was impossible", not on the returned list.
    assert rejected == ["present"]
    assert pool.get_sequence("present") is None
    assert worker.publishes == [1]


def test_all_ghost_rejection_still_publishes():
    """The publish gate is ``impossible``, not the returned list.

    Every impossible uuid missing from the pool returns ``[]`` THROUGH the
    publish — the one shape that distinguishes the final return from the
    no-impossible early return, and the reason the publish cannot move to a
    delegator on the return value alone.
    """
    pool = SequenceBatch()
    worker = _stub_worker(pool)

    rejected = worker._reject_unsatisfiable_prefill_requests(
        _selection(_candidate("ghost")), [100], {"ghost": 200},
    )

    assert rejected == []
    assert worker.publishes == [1]


# --------------------------------------------------------------------------- #
#  4. Selection request snapshot (moved; runs from the context)               #
# --------------------------------------------------------------------------- #


def test_selection_request_snapshots_the_candidates_at_g1():
    pool = SequenceBatch()
    _add_seq(pool, "q", 3, assigned_rank=2, prefill_order=9)
    _add_seq(pool, "e", 4, status=SequenceStatus.EVICTED, assigned_rank=1)
    prefix = PrefixReuseState()
    ctx = _ctx(pool, prefix=prefix)

    request = prefill_selection._make_prefill_selection_request(
        ctx, ["q", "e"], {}, [50], 1, PAGE,
        per_node_sequence_free=[6],
    )

    assert prefix.shared_prefix_credit_applied is False
    q, e = request.candidates
    assert (q.uuid, q.assigned_rank, q.node_id, q.is_evicted) == ("q", 2, 0, False)
    assert q.global_idx == 3 and q.prefill_order == 9
    assert q.prompt_length == PAGE and q.kv_token_budget == 2 * PAGE
    assert q.page_size == PAGE and q.host_kv_replication_factor == 1
    assert q.shared_prefix_segments == ()
    assert (e.uuid, e.is_evicted) == ("e", True)
    assert request.per_node_host_free == (50,)
    assert request.chunk_size == PAGE and request.num_nodes == 1
    assert request.gpus_per_node == 8
    assert request.initial_gpu_page_buffer == sequence_module.INITIAL_GPU_PAGE_BUFFER
    assert request.per_rank_sequence_free is None
    assert request.per_node_sequence_free == (6,)
    assert request.per_node_host_reclaimable is None


def test_shared_segments_set_the_credit_latch_and_reach_their_candidate():
    pool = SequenceBatch()
    _add_seq(pool, "hit", 0)
    _add_seq(pool, "miss", 1)
    prefix = PrefixReuseState()
    ctx = _ctx(pool, prefix=prefix)

    request = prefill_selection._make_prefill_selection_request(
        ctx, ["hit", "miss"], {"hit": ((12, 3),)}, [50], 1, PAGE,
    )

    assert prefix.shared_prefix_credit_applied is True
    hit, miss = request.candidates
    assert hit.shared_prefix_segments == ((12, 3),)
    assert miss.shared_prefix_segments == ()


def test_g_gt_1_maps_the_node_through_the_decode_dp_group():
    pool = SequenceBatch()
    # assigned_rank 0 sits on node 0; the sequence's group leader
    # (host_kv_owner_rank(1, 8) == 8) sits on node 1. The group must win.
    _add_seq(pool, "grouped", 0, assigned_rank=0, decode_dp_group=1)
    ctx = _ctx(pool, world_size=16, local_world_size=8, attn_tp_size=8)

    request = prefill_selection._make_prefill_selection_request(
        ctx, ["grouped"], {}, [50, 50], 2, PAGE,
    )

    assert request.candidates[0].node_id == 1


def test_g_gt_1_without_a_group_raises_before_selection():
    pool = SequenceBatch()
    _add_seq(pool, "ungrouped", 0, decode_dp_group=None)
    ctx = _ctx(pool, world_size=16, local_world_size=8, attn_tp_size=8)

    with pytest.raises(
        RuntimeError, match="has no decode_dp_group before"
    ):
        prefill_selection._make_prefill_selection_request(
            ctx, ["ungrouped"], {}, [50, 50], 2, PAGE,
        )


# --------------------------------------------------------------------------- #
#  5. Source-level contract                                                   #
# --------------------------------------------------------------------------- #


def _module_function(path, name):
    tree = ast.parse(path.read_text(), filename=str(path))
    return next(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def _class_node(path, class_name):
    tree = ast.parse(path.read_text(), filename=str(path))
    return next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )


def _class_method(path, class_name, method_name):
    return next(
        node for node in _class_node(path, class_name).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )


def test_moved_helpers_take_the_context_and_the_worker_only_wires_them():
    module = _ROOT / "batchgen" / "worker" / "admission" / "prefill_selection.py"
    worker = _ROOT / "batchgen" / "batchgen_worker.py"
    for name in MOVED:
        helper = _module_function(module, name)
        assert helper.args.args[0].arg == "ctx", name
        names = {n.id for n in ast.walk(helper) if isinstance(n, ast.Name)}
        assert "worker" not in names, name
        assert "self" not in names, (
            f"{name} is stateless and holds no worker"
        )

        delegator = _class_method(worker, "BatchGenWorker", name)
        calls = [
            node for node in ast.walk(delegator)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == name
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "prefill_selection"
        ]
        assert len(calls) == 1, name
        assert ast.unparse(calls[0].args[0]) == "self._prefill_phase_context()", name


def test_the_context_carries_state_only_never_a_callable():
    context = _ROOT / "batchgen" / "worker" / "admission" / "context.py"
    cls = _class_node(context, "PrefillPhaseContext")
    fields = [
        node for node in cls.body if isinstance(node, ast.AnnAssign)
    ]
    assert [f.target.id for f in fields] == ["facts", "sched", "prefix"]
    for field in fields:
        assert "Callable" not in ast.unparse(field.annotation), (
            f"{field.target.id}: the context must not carry a bound worker "
            "method"
        )


def test_the_selection_delegator_resolves_segments_before_the_module_call():
    """The resolver call keeps its original first-statement position.

    The worker resolves the shared-prefix segments (worker state) and passes
    the VALUE; the module call comes after, so the statement order matches
    the pre-move method: resolve, then latch, then the candidate walk.
    """
    worker = _ROOT / "batchgen" / "batchgen_worker.py"
    delegator = _class_method(
        worker, "BatchGenWorker", "_make_prefill_selection_request"
    )
    resolve = [
        node for node in ast.walk(delegator)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_admission_shared_prefix_segments"
    ]
    module_call = [
        node for node in ast.walk(delegator)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_make_prefill_selection_request"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "prefill_selection"
    ]
    assert len(resolve) == 1 and len(module_call) == 1
    assert resolve[0].lineno < module_call[0].lineno
    args = [ast.unparse(arg) for arg in module_call[0].args]
    assert args[:3] == [
        "self._prefill_phase_context()", "all_candidates", "shared_segments",
    ]
