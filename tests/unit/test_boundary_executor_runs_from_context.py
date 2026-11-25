"""Run ``BoundaryExecutor.apply`` on CPU from a hand-built decode context.

Every earlier test of the boundary carry-out is a source-level (AST) check.
This file is the first that EXECUTES ``apply``: a real ``DecodePhaseContext``
is assembled by hand — a real ``SequenceBatch`` pool, a real ``SequenceTable``
and a real ``TrajectoryStore`` — and the executor runs against it with no
stand-in worker object anywhere. Only the device-side collaborators are
recording fakes: the GPU paged-KV manager, the host paged-KV worker view and
the ``torch.distributed`` namespace (single rank, world_size 1, so no
collective may ever be reached — the fake records any call so the tests can
assert none happened).

Covered boundary outcomes, one test each:

* completion: tokens gathered and reported once, GPU pages freed, host pages
  released, the row leaves the returned decode set;
* on-hold: GPU pages freed, host pages kept, scalars zeroed;
* GPU page extension failure after a passing preflight (the branch a GPU run
  cannot produce by configuration): the row goes ON_HOLD with an
  ``extension_failed`` event and its pages freed;
* all rows completed: the early-return tuple shape.
"""

from __future__ import annotations

import importlib
import sys
import time
import types
from types import SimpleNamespace

from tests.triton_cpu_stub import install_triton_cpu_stub

install_triton_cpu_stub()


def _install_native_loader_stub() -> None:
    """Serve ``batchgen.native_loader`` when the compiled runtime is absent.

    ``kv_capacity_engine`` imports the dual host-KV coordinator, which imports
    the native core engine at module scope. On a CPU-only box without the
    compiled extension that import chain needs a stand-in; no native call ever
    runs in these tests. Nothing is installed when the real module imports.
    """
    if "batchgen.native_loader" in sys.modules:
        return
    try:
        importlib.import_module("batchgen.native_loader")
    except Exception:
        stub = types.ModuleType("batchgen.native_loader")
        stub.core_engine = SimpleNamespace()
        sys.modules["batchgen.native_loader"] = stub


_install_native_loader_stub()

import batchgen.lifespan as lifespan  # noqa: E402
import batchgen.worker.admission.boundary_executor as boundary_executor_module  # noqa: E402
import batchgen.worker.kv_capacity_engine as kv_capacity_engine_module  # noqa: E402
from batchgen.continuous_batching import (  # noqa: E402
    BoundaryDecisions,
    FastBoundaryTimingStats,
)
from batchgen.lifespan import SeqEvent  # noqa: E402
from batchgen.sequence import SequenceBatch, SequenceEntry, SequenceStatus  # noqa: E402
from batchgen.worker.admission.boundary_executor import (  # noqa: E402
    BoundaryExecutor,
    CompletionIO,
)
from batchgen.worker.gpu_residency import GpuResidencySet  # noqa: E402
from batchgen.worker.admission.context import (  # noqa: E402
    DecodePhaseContext,
    RankFacts,
    SchedulingState,
)
from batchgen.worker.admission.plan import (  # noqa: E402
    StatusEdge,
    Transition,
    TransitionPlan,
    apply_or_raise,
)
from batchgen.worker.rank_context import RankContext  # noqa: E402
from batchgen.worker.sequence_table import SequenceTable  # noqa: E402
from batchgen.worker.trajectory_store import TrajectoryStore  # noqa: E402

PAGE = SequenceEntry.PAGE_SIZE

UUID_A = "seq-a"
UUID_B = "seq-b"
GID_A = 100
GID_B = 101


def _residency(*uuids):
    """A seeded owner, exactly as the worker's allocation path builds one."""
    owner = GpuResidencySet()
    for uuid in uuids:
        owner.mark_resident(uuid)
    return owner


# --------------------------------------------------------------------------- #
#  Fakes: device-side collaborators only, each recording its calls            #
# --------------------------------------------------------------------------- #


class FakeGpuManager:
    """Stands in for the GPU paged-KV cache manager (``ctx.gpu_kv`` and the
    ``gpu_manager`` argument — one object in the real worker)."""

    is_initialized = True

    def __init__(self, num_free_pages=100, extend_raises=False):
        self.num_free_pages = num_free_pages
        self.extend_raises = extend_raises
        self.freed = []      # one list of global ids per free_pages_for_sequences
        self.rebuilds = []   # one list of global ids per rebuild_page_table
        self.extends = []    # (global_id, total_tokens) per extend_pages_for_sequence

    def get_stats(self):
        return SimpleNamespace(num_free_pages=self.num_free_pages)

    def free_pages_for_sequences(self, global_ids):
        self.freed.append(list(global_ids))

    def rebuild_page_table(self, global_ids):
        self.rebuilds.append(list(global_ids))

    def extend_pages_for_sequence(self, global_id, total_tokens):
        self.extends.append((int(global_id), int(total_tokens)))
        if self.extend_raises:
            raise RuntimeError("allocator rejected the extension")


class FakeHostView:
    """Stands in for the host paged-KV worker view (``ctx.host_view`` and the
    ``worker_view`` argument)."""

    def __init__(self):
        self.released = []       # one list of global ids per release_sequence_pages
        self.unregistered = []   # one list of global ids per unregister_sequences

    def release_sequence_pages(self, global_ids):
        self.released.append(list(global_ids))

    def unregister_sequences(self, global_ids):
        self.unregistered.append(list(global_ids))


class FakeDist:
    """Stands in for ``torch.distributed`` in the executor and the KV engine.

    world_size is 1 in every test, so no collective may be reached; each
    method records so the tests can assert the single-rank path stayed
    collective-free.
    """

    def __init__(self):
        self.calls = []

    def is_initialized(self):
        self.calls.append("is_initialized")
        return False

    def all_reduce(self, *args, **kwargs):
        self.calls.append("all_reduce")

    def barrier(self, *args, **kwargs):
        self.calls.append("barrier")

    def all_gather_object(self, *args, **kwargs):
        self.calls.append("all_gather_object")

    def broadcast_object_list(self, *args, **kwargs):
        self.calls.append("broadcast_object_list")


class RecordingCompletion:
    """The two ``CompletionIO`` callables, recording their calls."""

    def __init__(self, texts):
        self._texts = texts
        self.gather_calls = []   # one list of uuids per gather_completed_tokens
        self.reports = []        # (uuid, gathered_text) per report_completion

    def gather(self, completed_uuids):
        self.gather_calls.append(list(completed_uuids))
        return {uuid: self._texts[uuid] for uuid in completed_uuids}

    def report(self, uuid, gathered_text=None):
        self.reports.append((uuid, gathered_text))


# --------------------------------------------------------------------------- #
#  Context assembly                                                           #
# --------------------------------------------------------------------------- #


def _make_decoding_seq(pool, table, uuid, global_idx):
    """A real pool row walked to IN_DECODE with one decoded page on GPU."""
    seq = SequenceEntry(
        uuid=uuid,
        global_idx=global_idx,
        prompt_length=PAGE,
        max_decode_length=3 * PAGE,
    )
    pool.add_sequence(seq)
    pool.assign_rank(uuid, 0)
    for status in (
        SequenceStatus.IN_PREFILL,
        SequenceStatus.PREFILLED,
        SequenceStatus.IN_DECODE,
    ):
        apply_or_raise(
            pool, TransitionPlan(StatusEdge.DECODE_ENTRY, (Transition(uuid, status),))
        )
    seq.decoded_length = PAGE
    seq.current_context_length = 2 * PAGE
    seq.gpu_pages_allocated = 2
    seq.had_initial_gpu_reservation = True
    seq.host_pages_allocated = 4
    seq.host_owned_pages = 4
    seq.host_token_capacity = 4 * PAGE
    table.bind(uuid)
    return seq


def _harness(monkeypatch, *, gpu_free=100, extend_raises=False):
    """Two decoding rows on rank 0 of a world of 1, all collaborators fakes."""
    fake_dist = FakeDist()
    monkeypatch.setattr(boundary_executor_module, "dist", fake_dist)
    monkeypatch.setattr(kv_capacity_engine_module, "dist", fake_dist)

    pool = SequenceBatch()
    table = SequenceTable()
    seq_a = _make_decoding_seq(pool, table, UUID_A, GID_A)
    seq_b = _make_decoding_seq(pool, table, UUID_B, GID_B)

    gpu = FakeGpuManager(num_free_pages=gpu_free, extend_raises=extend_raises)
    view = FakeHostView()
    facts = RankFacts(
        rank=0,
        world_size=1,
        local_rank=0,
        local_world_size=1,
        rank_ctx=RankContext(
            rank=0, world_size=1, local_rank=0, local_world_size=1,
            decode_group_size=1,
        ),
        attn_tp_size=1,
        device="cpu",
        page_size=PAGE,
        decision_interval=8,
        enable_prefix_cache=False,
        enable_host_kv_eviction=False,
        host_kv_chunk_size=PAGE,
    )
    sched = SchedulingState(
        pool=pool, seq_table=table, trajectories=TrajectoryStore(), prefix=None
    )
    ctx = DecodePhaseContext(
        facts=facts,
        sched=sched,
        gpu_kv=gpu,
        host_view=view,
        core_engine=None,
        gpu_resident=_residency(UUID_A, UUID_B),
        prefix_coordinator=None,
        prefix_runtime=None,
        chunk_sizer=None,
    )
    completion = RecordingCompletion({UUID_A: "text-a", UUID_B: "text-b"})
    return SimpleNamespace(
        pool=pool, table=table, ctx=ctx, gpu=gpu, view=view, dist=fake_dist,
        completion=completion, seq_a=seq_a, seq_b=seq_b,
    )


def _decisions(**overrides):
    base = dict(
        completed_uuids=[],
        active_uuids=[],
        host_growth_uuids=[],
        host_growth_pages=[],
        growth_feasible=False,
        host_evicted_uuids=[],
        onhold_uuids=[],
        seqs_needing_extension=[],
        new_load_uuids=[],
        decode_uuids_final=[],
    )
    base.update(overrides)
    return BoundaryDecisions(**base)


def _apply(h, decisions):
    """Call ``apply`` the way the boundary does: pre-decision decode set in."""
    decode_uuids = [
        uuid for uuid in (UUID_A, UUID_B)
        if h.pool.get_sequence(uuid).status == SequenceStatus.IN_DECODE
    ]
    batch = [h.table.uuid_to_local[uuid] for uuid in decode_uuids]
    timing = FastBoundaryTimingStats()
    now = time.perf_counter()
    result = BoundaryExecutor().apply(
        h.ctx,
        CompletionIO(
            gather_completed_tokens=h.completion.gather,
            report_completion=h.completion.report,
        ),
        decisions,
        decode_uuids,
        batch,
        gpu_manager=h.gpu,
        worker_view=h.view,
        global_seq_state={},
        chunk_size=PAGE,
        timing=timing,
        boundary_start=now,
        t0=now,
    )
    return result, timing


# --------------------------------------------------------------------------- #
#  1. Completion                                                              #
# --------------------------------------------------------------------------- #


def test_completion_releases_pages_and_reports_once(monkeypatch):
    h = _harness(monkeypatch)
    result, timing = _apply(h, _decisions(
        completed_uuids=[UUID_A],
        active_uuids=[UUID_B],
        decode_uuids_final=[UUID_B],
    ))

    # Completion outlet: one gather of the completed set, one report per row,
    # carrying the gathered text.
    assert h.completion.gather_calls == [[UUID_A]]
    assert h.completion.reports == [(UUID_A, "text-a")]

    # Pool statuses: the completed row moved, the other stayed decoding.
    assert h.seq_a.status == SequenceStatus.COMPLETED
    assert h.seq_b.status == SequenceStatus.IN_DECODE

    # GPU pages freed on the fake manager; host pages released on the fake
    # view; the survivor's page table rebuilt from the remaining decode set.
    assert h.gpu.freed == [[GID_A]]
    assert h.view.released == [[GID_A]]
    assert h.gpu.rebuilds == [[GID_B]]

    # Per-sequence scalars zeroed on the completed row only.
    assert h.seq_a.gpu_pages_allocated == 0
    assert h.seq_a.host_pages_allocated == 0
    assert h.seq_a.host_owned_pages == 0
    assert h.seq_a.host_token_capacity == 0
    assert h.seq_b.gpu_pages_allocated == 2
    assert h.seq_b.host_pages_allocated == 4
    assert h.ctx.gpu_resident == {UUID_B}

    # Returned tuple: the completed row left the decode set; no load launched.
    decode_uuids, batch, task, loads, ll, lg, no_rows_left, tensors = result
    assert decode_uuids == [UUID_B]
    assert batch == [h.table.uuid_to_local[UUID_B]]
    assert (task, loads, ll, lg, no_rows_left, tensors) == (
        None, [], [], [], False, None
    )
    assert timing.total_completed_cumulative == 1

    # Single rank: no collective was reached.
    assert h.dist.calls == []


# --------------------------------------------------------------------------- #
#  2. On-hold                                                                 #
# --------------------------------------------------------------------------- #


def test_onhold_frees_gpu_pages_but_keeps_host_pages(monkeypatch):
    h = _harness(monkeypatch)
    result, _ = _apply(h, _decisions(
        active_uuids=[UUID_A, UUID_B],
        onhold_uuids=[UUID_A],
        decode_uuids_final=[UUID_B],
    ))

    assert h.seq_a.status == SequenceStatus.ON_HOLD
    assert h.seq_a.gpu_pages_allocated == 0
    # GPU pages freed through the manager's bulk path...
    assert h.gpu.freed == [[GID_A]]
    # ...but the host-side KV stays: nothing released or unregistered.
    assert h.view.released == []
    assert h.view.unregistered == []
    assert h.seq_a.host_pages_allocated == 4
    assert h.seq_a.host_owned_pages == 4
    # Off the GPU-resident set; the survivor remains.
    assert h.ctx.gpu_resident == {UUID_B}
    assert h.seq_b.status == SequenceStatus.IN_DECODE

    decode_uuids, batch, task, loads, ll, lg, no_rows_left, tensors = result
    assert decode_uuids == [UUID_B]
    assert batch == [h.table.uuid_to_local[UUID_B]]
    assert no_rows_left is False
    assert (task, loads, ll, lg, tensors) == (None, [], [], [], None)
    assert h.completion.reports == []
    assert h.dist.calls == []


# --------------------------------------------------------------------------- #
#  3. GPU page extension failure                                              #
# --------------------------------------------------------------------------- #


def test_extension_failure_moves_sequence_on_hold(monkeypatch):
    """The allocator rejects after a passing stats preflight.

    The stats snapshot shows enough free pages, so the collective preflight
    passes; ``extend_pages_for_sequence`` then raises. The executor must
    convert that into ON_HOLD with pages freed, never leave the row decoding
    with ``cache_seqlens`` beyond its allocation.
    """
    monkeypatch.setattr(lifespan, "ENABLED", True)
    h = _harness(monkeypatch, extend_raises=True)
    result, _ = _apply(h, _decisions(
        active_uuids=[UUID_A, UUID_B],
        seqs_needing_extension=[UUID_A],
        decode_uuids_final=[UUID_A, UUID_B],
    ))

    # The allocator was really asked for the extension: 2 more pages on top of
    # the 2 allocated, so 4 pages * PAGE tokens total.
    assert h.gpu.extends == [(GID_A, 4 * PAGE)]

    # The failed row is ON_HOLD with zeroed GPU scalars and freed pages.
    assert h.seq_a.status == SequenceStatus.ON_HOLD
    assert h.seq_a.gpu_pages_allocated == 0
    assert h.gpu.freed == [[GID_A]]
    assert UUID_A not in h.ctx.gpu_resident

    # The ON_HOLD event carries the extension-failure trigger.
    onhold_events = [
        record for record in h.seq_a._lifespan_log
        if record.event == SeqEvent.ON_HOLD
    ]
    assert [record.detail for record in onhold_events] == [
        "trigger=extension_failed"
    ]

    # Host pages stay; the row can re-enter decode from host KV later.
    assert h.view.released == []
    assert h.seq_a.host_pages_allocated == 4

    # The other sequence keeps decoding and the failed row left the batch.
    assert h.seq_b.status == SequenceStatus.IN_DECODE
    decode_uuids, batch, task, loads, ll, lg, no_rows_left, tensors = result
    assert decode_uuids == [UUID_B]
    assert batch == [h.table.uuid_to_local[UUID_B]]
    assert no_rows_left is False
    assert (task, loads, ll, lg, tensors) == (None, [], [], [], None)
    assert h.dist.calls == []


# --------------------------------------------------------------------------- #
#  4. No decode rows left                                                     #
# --------------------------------------------------------------------------- #


def test_all_completed_takes_early_return_shape(monkeypatch):
    h = _harness(monkeypatch)
    result, timing = _apply(h, _decisions(
        completed_uuids=[UUID_A, UUID_B],
        active_uuids=[],
        decode_uuids_final=[],
    ))

    assert result == ([], [], None, [], [], [], True, None)
    assert timing.total_completed_cumulative == 2

    # Both rows fully completed and released.
    assert h.seq_a.status == SequenceStatus.COMPLETED
    assert h.seq_b.status == SequenceStatus.COMPLETED
    assert h.gpu.freed == [[GID_A, GID_B]]
    assert h.view.released == [[GID_A, GID_B]]
    # No remaining rows: no page-table rebuild.
    assert h.gpu.rebuilds == []
    assert h.ctx.gpu_resident == set()
    assert h.completion.gather_calls == [[UUID_A, UUID_B]]
    assert sorted(h.completion.reports) == [
        (UUID_A, "text-a"), (UUID_B, "text-b"),
    ]
    assert h.dist.calls == []
