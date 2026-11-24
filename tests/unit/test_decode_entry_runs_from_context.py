"""Run ``DecodeEntry.select`` on CPU from a hand-built decode context.

The decode-entry selection moved off the worker; this file EXECUTES it: a real
``DecodePhaseContext`` is assembled by hand — a real ``SequenceBatch`` pool, a
real ``SequenceTable`` and a real ``TrajectoryStore`` — and ``select`` runs
against it with no stand-in worker object anywhere. Only the device-side
collaborators are fakes: the GPU paged-KV manager (recording its capacity
reads) and the ``torch.distributed`` namespace.

Covered outcomes, one test each:

* no candidates: ``[]``, and the collective capacity snapshot was still taken
  first, exactly as on the worker;
* PREFILLED + ON_HOLD candidates within capacity are selected in
  ``global_idx`` order, not in enumeration order;
* a candidate whose reservation does not fit the free pages is left out while
  a later, smaller candidate still enters;
* an IN_DECODE sequence with ``attn_tp_size > 1`` and no ``decode_dp_group``
  raises the existing ``ValueError``.

Plus the source-level contract: ``select`` takes ``(self, ctx, max_rank_bsz)``
and never references a worker, and the worker's ``_prepare_decode_batch``
calls it exactly once, from ``self._decode_phase_context()``.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import importlib
import pathlib
import sys
import types
from types import SimpleNamespace

import pytest

from tests.triton_cpu_stub import install_triton_cpu_stub

install_triton_cpu_stub()


def _install_native_loader_stub() -> None:
    """Serve ``batchgen.native_loader`` when the compiled runtime is absent."""
    if "batchgen.native_loader" in sys.modules:
        return
    try:
        importlib.import_module("batchgen.native_loader")
    except Exception:
        stub = types.ModuleType("batchgen.native_loader")
        stub.core_engine = SimpleNamespace()
        sys.modules["batchgen.native_loader"] = stub


_install_native_loader_stub()

import batchgen.worker.admission.decode_entry as decode_entry_module  # noqa: E402
import batchgen.worker.kv_capacity_engine as kv_capacity_engine_module  # noqa: E402
from batchgen.sequence import SequenceBatch, SequenceEntry, SequenceStatus  # noqa: E402
from batchgen.worker.admission.context import (  # noqa: E402
    DecodePhaseContext,
    RankFacts,
    SchedulingState,
)
from batchgen.worker.admission.decode_entry import DecodeEntry  # noqa: E402
from batchgen.worker.gpu_residency import GpuResidencySet  # noqa: E402
from batchgen.worker.admission.plan import (  # noqa: E402
    StatusEdge,
    Transition,
    TransitionPlan,
    apply_or_raise,
)
from batchgen.worker.rank_context import RankContext  # noqa: E402
from batchgen.worker.sequence_table import SequenceTable  # noqa: E402
from batchgen.worker.trajectory_store import TrajectoryStore  # noqa: E402

_ROOT = pathlib.Path(__file__).resolve().parents[2]
PAGE = SequenceEntry.PAGE_SIZE


# --------------------------------------------------------------------------- #
#  Fakes: device-side collaborators only, each recording its calls            #
# --------------------------------------------------------------------------- #


class FakeGpuManager:
    """Stands in for the GPU paged-KV cache manager (``ctx.gpu_kv``)."""

    is_initialized = True

    def __init__(self, num_free_pages=100, num_total_pages=100):
        self.num_free_pages = num_free_pages
        self.num_total_pages = num_total_pages
        self.snapshots = 0     # capacity_snapshot entries
        self.stats_reads = 0   # get_stats calls

    @contextlib.contextmanager
    def capacity_snapshot(self):
        self.snapshots += 1
        yield self

    def get_stats(self):
        self.stats_reads += 1
        return SimpleNamespace(
            num_free_pages=self.num_free_pages,
            num_total_pages=self.num_total_pages,
        )


class FakeDist:
    """Stands in for ``torch.distributed`` in decode entry and the KV engine.

    ``all_gather`` fills every slot with this rank's own stats, which is the
    correct single-node fixture: every fake rank reports the same pool.
    """

    def __init__(self):
        self.calls = []

    def is_initialized(self):
        self.calls.append("is_initialized")
        return True

    def all_gather(self, out, local):
        self.calls.append("all_gather")
        for item in out:
            item.copy_(local)

    def all_gather_object(self, out, local):
        self.calls.append("all_gather_object")
        for index in range(len(out)):
            out[index] = local


# --------------------------------------------------------------------------- #
#  Context assembly                                                           #
# --------------------------------------------------------------------------- #


def _add_seq(pool, table, uuid, global_idx, *, status, req_pages,
             decode_dp_group=None):
    """A real pool row walked to ``status`` whose reservation is ``req_pages``.

    ``get_gpu_pages_for_two_page_buffer`` caps at ``host_pages_allocated``, so
    setting that scalar pins the candidate's GPU reservation exactly.
    """
    seq = SequenceEntry(
        uuid=uuid,
        global_idx=global_idx,
        prompt_length=PAGE,
        max_decode_length=63 * PAGE,
    )
    pool.add_sequence(seq)
    pool.assign_rank(uuid, 0)
    walk = {
        SequenceStatus.PREFILLED: (
            SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED,
        ),
        SequenceStatus.ON_HOLD: (
            SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED,
            SequenceStatus.IN_DECODE, SequenceStatus.ON_HOLD,
        ),
        SequenceStatus.IN_DECODE: (
            SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED,
            SequenceStatus.IN_DECODE,
        ),
    }[status]
    for step in walk:
        apply_or_raise(
            pool, TransitionPlan(StatusEdge.DECODE_ENTRY, (Transition(uuid, step),))
        )
    seq.host_pages_allocated = req_pages
    seq.host_owned_pages = req_pages
    seq.host_token_capacity = req_pages * PAGE
    seq.decode_dp_group = decode_dp_group
    table.bind(uuid)
    return seq


def _context(monkeypatch, gpu, *, world_size=1, attn_tp_size=1):
    fake_dist = FakeDist()
    monkeypatch.setattr(decode_entry_module, "dist", fake_dist)
    monkeypatch.setattr(kv_capacity_engine_module, "dist", fake_dist)

    pool = SequenceBatch()
    table = SequenceTable()
    facts = RankFacts(
        rank=0,
        world_size=world_size,
        local_rank=0,
        local_world_size=world_size,
        rank_ctx=RankContext(
            rank=0, world_size=world_size, local_rank=0,
            local_world_size=world_size, decode_group_size=attn_tp_size,
        ),
        attn_tp_size=attn_tp_size,
        device="cpu",
        page_size=PAGE,
        decision_interval=8,
        enable_prefix_cache=False,
        enable_host_kv_eviction=False,
        host_kv_chunk_size=PAGE,
    )
    sched = SchedulingState(
        pool=pool, seq_table=table, trajectories=TrajectoryStore()
    )
    ctx = DecodePhaseContext(
        facts=facts,
        sched=sched,
        gpu_kv=gpu,
        host_view=None,
        core_engine=None,
        gpu_resident=GpuResidencySet(),
        prefix_coordinator=None,
        prefix_runtime=None,
        chunk_sizer=None,
    )
    return SimpleNamespace(pool=pool, table=table, ctx=ctx, dist=fake_dist)


# --------------------------------------------------------------------------- #
#  1. No candidates                                                           #
# --------------------------------------------------------------------------- #


def test_no_candidates_returns_empty_after_the_capacity_snapshot(monkeypatch):
    gpu = FakeGpuManager()
    h = _context(monkeypatch, gpu)

    assert DecodeEntry().select(h.ctx, 0) == []

    # The collective snapshot is taken BEFORE the empty-candidate return, as
    # on the worker: the snapshot holds the pool mapped and keeps the ranks'
    # read pattern identical whether or not candidates exist.
    assert gpu.snapshots == 1
    assert gpu.stats_reads == 1


# --------------------------------------------------------------------------- #
#  2. PREFILLED + ON_HOLD, in global_idx order                                #
# --------------------------------------------------------------------------- #


def test_prefilled_and_onhold_select_in_global_idx_order(monkeypatch):
    gpu = FakeGpuManager(num_free_pages=100, num_total_pages=100)
    h = _context(monkeypatch, gpu)
    # The ON_HOLD row has the LOWER global_idx: the selection must order by
    # global_idx, not by the PREFILLED-then-ON_HOLD enumeration.
    _add_seq(h.pool, h.table, "seq-onhold", 100,
             status=SequenceStatus.ON_HOLD, req_pages=2)
    _add_seq(h.pool, h.table, "seq-prefilled", 101,
             status=SequenceStatus.PREFILLED, req_pages=2)

    assert DecodeEntry().select(h.ctx, 0) == ["seq-onhold", "seq-prefilled"]
    assert gpu.snapshots == 1


# --------------------------------------------------------------------------- #
#  3. A candidate that does not fit is left out                               #
# --------------------------------------------------------------------------- #


def test_candidate_beyond_free_pages_is_left_out(monkeypatch):
    gpu = FakeGpuManager(num_free_pages=4, num_total_pages=100)
    h = _context(monkeypatch, gpu)
    _add_seq(h.pool, h.table, "seq-first", 100,
             status=SequenceStatus.PREFILLED, req_pages=2)
    _add_seq(h.pool, h.table, "seq-too-big", 101,
             status=SequenceStatus.PREFILLED, req_pages=4)
    _add_seq(h.pool, h.table, "seq-last", 102,
             status=SequenceStatus.PREFILLED, req_pages=2)

    # 2 + 4 > 4 free pages: the middle row is left out; the greedy walk still
    # admits the later row that fits the remaining 2 pages.
    assert DecodeEntry().select(h.ctx, 0) == ["seq-first", "seq-last"]


# --------------------------------------------------------------------------- #
#  4. IN_DECODE without a decode_dp_group under TP decode                     #
# --------------------------------------------------------------------------- #


def test_in_decode_without_group_raises_under_tp_decode(monkeypatch):
    gpu = FakeGpuManager()
    h = _context(monkeypatch, gpu, world_size=2, attn_tp_size=2)
    _add_seq(h.pool, h.table, "seq-candidate", 100,
             status=SequenceStatus.PREFILLED, req_pages=2, decode_dp_group=0)
    _add_seq(h.pool, h.table, "seq-decoding", 101,
             status=SequenceStatus.IN_DECODE, req_pages=2,
             decode_dp_group=None)

    with pytest.raises(ValueError, match="has no decode_dp_group for"):
        DecodeEntry().select(h.ctx, 0)


# --------------------------------------------------------------------------- #
#  Shared-prefix runs (prefix cache on)                                       #
# --------------------------------------------------------------------------- #


class FakeHostView:
    """Stands in for the host paged-KV view the shared runs are read from."""

    def __init__(self, tables):
        self.tables = tables
        self.requests = []

    def build_page_table(self, global_ids):
        self.requests.append(list(global_ids))
        return [list(self.tables[int(g)]) for g in global_ids]


def _shared_prefix_harness(monkeypatch, *, world_size=1):
    """Two PREFILLED rows of one book whose four Host pages are all shared."""
    gpu = FakeGpuManager()
    gpu.is_initialized = True
    h = _context(monkeypatch, gpu, world_size=world_size)
    facts = dataclasses.replace(h.ctx.facts, enable_prefix_cache=True)
    h.ctx = dataclasses.replace(h.ctx, facts=facts)
    _add_seq(h.pool, h.table, "a", 0, status=SequenceStatus.PREFILLED, req_pages=4)
    _add_seq(h.pool, h.table, "b", 1, status=SequenceStatus.PREFILLED, req_pages=4)
    h.view = FakeHostView({0: [7, 8, 9, 10], 1: [7, 8, 9, 10]})
    monkeypatch.setattr(
        kv_capacity_engine_module.KVCapacityEngine,
        "_shared_page_identity_sources",
        lambda self, ctx, manager: [(None, None, h.view)],
    )
    return h


def test_local_shared_runs_are_read_from_the_host_page_ids(monkeypatch):
    h = _shared_prefix_harness(monkeypatch)
    runs = decode_entry_module._local_decode_shared_prefix_runs(h.ctx, ["a", "b"])
    # One page of each reservation stays private even though all four Host
    # pages are shared, so three pages are offered per row; the expectation
    # is the pure planner on exactly those tables.
    planned = decode_entry_module.plan_shared_runs({0: [7, 8, 9], 1: [7, 8, 9]})
    assert planned, "the fixture must produce a shared run"
    assert runs == {
        uuid: ((int(planned[idx][0]), int(planned[idx][1])),)
        for uuid, idx in (("a", 0), ("b", 1)) if idx in planned
    }
    assert runs
    assert h.view.requests == [[0, 1]]


def test_a_single_owned_candidate_has_no_shared_run(monkeypatch):
    h = _shared_prefix_harness(monkeypatch)
    assert decode_entry_module._local_decode_shared_prefix_runs(h.ctx, ["a"]) == {}
    assert h.view.requests == []


def test_shared_run_gather_is_skipped_without_the_prefix_cache(monkeypatch):
    h = _shared_prefix_harness(monkeypatch)
    off = dataclasses.replace(
        h.ctx, facts=dataclasses.replace(h.ctx.facts, enable_prefix_cache=False)
    )
    assert decode_entry_module._gather_decode_shared_prefix_segments(off, ["a", "b"]) == {}
    assert h.view.requests == [] and h.dist.calls == []


def test_shared_run_gather_is_local_on_one_rank_and_collective_on_two(monkeypatch):
    one = _shared_prefix_harness(monkeypatch)
    local = decode_entry_module._local_decode_shared_prefix_runs(one.ctx, ["a", "b"])
    assert decode_entry_module._gather_decode_shared_prefix_segments(
        one.ctx, ["a", "b"]
    ) == local
    assert "all_gather_object" not in one.dist.calls

    two = _shared_prefix_harness(monkeypatch, world_size=2)
    merged = decode_entry_module._gather_decode_shared_prefix_segments(
        two.ctx, ["a", "b"]
    )
    assert merged == local
    assert two.dist.calls.count("all_gather_object") == 1


def test_select_passes_the_shared_runs_to_the_pure_selection(monkeypatch):
    h = _shared_prefix_harness(monkeypatch)
    seen = {}
    real = decode_entry_module.DecodeScheduler.select_decode_batch

    def recording(request):
        seen["segments"] = {
            c.uuid: c.shared_prefix_segments for c in request.candidates
        }
        return real(request)

    monkeypatch.setattr(
        decode_entry_module.DecodeScheduler, "select_decode_batch",
        staticmethod(recording),
    )
    selected = DecodeEntry().select(h.ctx, 0)
    local = decode_entry_module._local_decode_shared_prefix_runs(h.ctx, ["a", "b"])
    assert seen["segments"] == {u: local.get(u, ()) for u in ("a", "b")}
    assert selected == ["a", "b"]


# --------------------------------------------------------------------------- #
#  5. Source-level contract                                                   #
# --------------------------------------------------------------------------- #


def _class_method(path, class_name, method_name):
    tree = ast.parse(path.read_text(), filename=str(path))
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )


def test_select_runs_from_the_context_and_the_worker_only_wires_it():
    select = _class_method(
        _ROOT / "batchgen" / "worker" / "admission" / "decode_entry.py",
        "DecodeEntry",
        "select",
    )
    assert [arg.arg for arg in select.args.args] == ["self", "ctx", "max_rank_bsz"]
    names = {node.id for node in ast.walk(select) if isinstance(node, ast.Name)}
    assert "worker" not in names
    assert not any(
        isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        for node in ast.walk(select)
    ), "the selection is stateless and holds no worker"

    prepare = _class_method(
        _ROOT / "batchgen" / "batchgen_worker.py",
        "BatchGenWorker",
        "_prepare_decode_batch",
    )
    select_calls = [
        node for node in ast.walk(prepare)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "select"
        and isinstance(node.func.value, ast.Call)
        and isinstance(node.func.value.func, ast.Name)
        and node.func.value.func.id == "DecodeEntry"
    ]
    assert len(select_calls) == 1
    assert ast.unparse(select_calls[0].args[0]) == "self._decode_phase_context()"
