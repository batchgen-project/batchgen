"""The in-wave prefix pool across one serve group (G>1, Kimi-K3).

At G>1 a sequence is replicated onto all G ranks of its serve group and every
one of them prefills it, so the pool wave must be IDENTICAL on all G ranks: the
page and recurrent-state budgets are reduced to the group MIN, the plan digest
is compared inside the group, and the Host pages a segment lands on are the
group leader's alone (the followers mirror its page table). These tests pin that
split without importing the worker: its methods are compiled straight out of the
module's AST onto a namespace of fakes, so nothing here needs CUDA or NCCL. The
pure prefix-reuse helpers (planner, pool, executor, wave runtime) are the real
ones, so the plans and digests under test are the real ones too.

Ranks are stepped one at a time in this process, so a real rendezvous is
impossible: ``FakeDist`` reduces or gathers the caller's contribution against
peer values the test configures, which is what the collective would produce.
"""

import ast
import copy
import dataclasses
import hashlib
import logging
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence

import pytest
import torch

from batchgen.prefix_reuse import wave_runtime
from batchgen.prefix_reuse.config import PrefixKVGroupSemantic
from batchgen.prefix_reuse.executor import PoolChunkExecutor
from batchgen.prefix_reuse.pool import PrefillPrefixPool
from batchgen.prefix_reuse.prefill import PrefixCacheSequenceState
from batchgen.prefix_reuse.wave_plan import plan_wave_prefix_sharing


WORKER = Path(__file__).resolve().parents[2] / "batchgen" / "batchgen_worker.py"

PAGE_TOKENS = 4
# Two prompts sharing exactly two blocks, so the wave has ONE pooled segment.
SHARED = list(range(100, 100 + 2 * PAGE_TOKENS))
PROMPTS = [SHARED + [1, 2, 3, 4, 5], SHARED + [6, 7, 8, 9, 10]]
UUIDS = ["a", "b"]
GLOBAL_IDS = [100, 101]

# What the Host geometry says (one 8-wide K/V head over 2 layers) and what the
# K3 PSM says instead (2 local heads of 6-wide K and 4-wide V over 2 layers).
GEOMETRY = SimpleNamespace(
    num_layers=2, num_k_heads=1, k_head_dim=8, kv_dtype=torch.bfloat16,
)
K3_LAYOUT = {
    "num_layers": 2,
    "layer_map": (0, -1, 1),
    "num_kv_heads": 2,
    "k_head_dim": 6,
    "v_head_dim": 4,
}
K3_PAGE_BYTES = 2 * PAGE_TOKENS * 2 * (6 + 4) * 2
TOKEN_CAP = 64


def _worker_class():
    tree = ast.parse(WORKER.read_text())
    return next(node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker")


def _worker_method_node(name):
    return next(node for node in _worker_class().body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def _worker_methods(names, globals_=None):
    """Compile the named worker methods onto one shared namespace of fakes."""
    wanted = [copy.deepcopy(_worker_method_node(name)) for name in names]
    module = ast.fix_missing_locations(ast.Module(body=wanted, type_ignores=[]))
    namespace = {
        "List": List, "Optional": Optional, "Sequence": Sequence,
        "hashlib": hashlib, "logging": logging, "math": math,
        "replace": dataclasses.replace,
        **(globals_ or {}),
    }
    exec(compile(module, str(WORKER), "exec"), namespace)
    return {name: namespace[name] for name in names}


def _bind(worker, methods):
    for name, function in methods.items():
        setattr(worker, name, types.MethodType(function, worker))
    return worker


# --------------------------------------------------------------------------- #
#  Fakes
# --------------------------------------------------------------------------- #


class FakeCuda:
    """Just enough of ``torch.cuda`` for the planner's memory accounting."""

    def __init__(self, free_bytes):
        self.free_bytes = int(free_bytes)

    def synchronize(self, device=None):
        pass

    def empty_cache(self):
        pass

    def mem_get_info(self, device=None):
        return self.free_bytes, 1 << 40

    def memory_reserved(self, device=None):
        return 0

    def memory_allocated(self, device=None):
        return 0

    def reset_peak_memory_stats(self, device=None):
        pass


def _fake_torch(free_bytes):
    return SimpleNamespace(
        empty=torch.empty, tensor=torch.tensor, zeros_like=torch.zeros_like,
        int64=torch.int64, cuda=FakeCuda(free_bytes),
    )


class FakeDist:
    """MIN all-reduce / all-gather against configured peer contributions."""

    ReduceOp = SimpleNamespace(MIN="min", MAX="max")

    def __init__(self, *, peer_budget=None, peer_digest=None, world_size=2):
        self.peer_budget = peer_budget
        self.peer_digest = peer_digest
        self.slots = [None] * world_size
        self.rank = 0
        self.reduced = []
        self.gathered = []
        self.object_calls = 0

    def all_reduce(self, tensor, op=None, group=None):
        assert op == self.ReduceOp.MIN, op
        assert group is not None, "the pool budget must reduce over the group"
        self.reduced.append([int(value) for value in tensor.tolist()])
        if self.peer_budget is not None:
            # The planner reduces [pages, slots, materialization vote]; pool
            # construction reduces a 1-wide [pages] tensor over the same
            # group. A scripted peer that says nothing about the trailing
            # values contributes a value the MIN ignores -- for the vote that
            # is 1, "this peer confirms".
            values = list(self.peer_budget[: tensor.numel()])
            while len(values) < tensor.numel():
                values.append(1)
            peer = torch.tensor(values, dtype=tensor.dtype)
            torch.minimum(tensor, peer, out=tensor)

    def all_gather(self, out, tensor, group=None):
        assert group is not None, "the plan digest must gather over the group"
        self.gathered.append(int(tensor.item()))
        for slot in out:
            slot.copy_(tensor)
        if self.peer_digest is not None:
            out[-1].fill_(int(self.peer_digest))

    def all_gather_object(self, out, payload):
        # Ranks run sequentially, so a slot holds whatever that rank published.
        self.object_calls += 1
        self.slots[self.rank] = payload
        for index, value in enumerate(self.slots):
            out[index] = value


class FakeBackend:
    """One host-KV region of one KV group, shared by every rank of the node."""

    def __init__(self, first_page):
        self.pages = {}
        self.released = []
        self.unregistered = []
        self.next_page = first_page


class FakeView:
    """A worker view: a LOCAL page table over a shared backend region."""

    def __init__(self, backend, rank):
        self.backend = backend
        self.rank = rank
        self.table = {}

    def register_sequences(self, global_ids):
        for global_id in global_ids:
            self.table[global_id] = list(self.backend.pages.get(global_id, ()))

    def allocate_pages_for_sequences(self, seq_token_pairs):
        for global_id, tokens in seq_token_pairs:
            new = [
                self.backend.next_page + offset
                for offset in range(-(-int(tokens) // PAGE_TOKENS))
            ]
            self.backend.next_page += len(new)
            self.backend.pages.setdefault(int(global_id), []).extend(new)
            self.table.setdefault(int(global_id), []).extend(new)

    def build_page_table(self, global_ids):
        return [list(self.table.get(global_id, ())) for global_id in global_ids]

    def set_sequence_pages(self, global_id, page_ids):
        self.table[int(global_id)] = [int(page) for page in page_ids]

    def release_sequence_pages(self, global_ids):
        self.backend.released.append((self.rank, list(global_ids)))
        for global_id in global_ids:
            self.backend.pages.pop(int(global_id), None)
            self.table.pop(int(global_id), None)

    def unregister_sequences(self, global_ids):
        # Local-only: drops this view's record, never the shared pages.
        self.backend.unregistered.append((self.rank, list(global_ids)))
        for global_id in global_ids:
            self.table.pop(int(global_id), None)


class RecordingCoordinator:
    def __init__(self):
        self.released = []

    def release_attachment(self, handle):
        self.released.append(int(handle))


def _seqs():
    return {
        uuid: SimpleNamespace(
            uuid=uuid,
            global_idx=global_id,
            assigned_rank=0,
            decode_dp_group=0,
            prompt_length=len(prompt),
            input_ids=torch.tensor([prompt], dtype=torch.long),
        )
        for uuid, global_id, prompt in zip(UUIDS, GLOBAL_IDS, PROMPTS)
    }


def _stub_heavy_modules(monkeypatch, *, free_state_slots):
    """Serve the imports whose real modules need the compiled engine/CUDA."""
    monkeypatch.setitem(
        sys.modules, "batchgen.kv_cache.host_kv_mananger_config",
        SimpleNamespace(build_gpu_kv_config=lambda *args, **kwargs: GEOMETRY),
    )
    monkeypatch.setitem(
        sys.modules, "batchgen.decode_dp_group",
        SimpleNamespace(host_kv_owner_rank=lambda group, size: int(group) * size),
    )
    state_manager = (
        None if free_state_slots is None else SimpleNamespace(
            get_stats=lambda: SimpleNamespace(
                num_free_state_items=int(free_state_slots)
            )
        )
    )
    monkeypatch.setitem(
        sys.modules, "batchgen.models.moonshotai.kimi_linear.wrappers",
        SimpleNamespace(KimiLinearKDAWrapper=SimpleNamespace(
            state_manager=state_manager,
            slot_manager=None if state_manager is None else object(),
        )),
    )


def _plan_worker(
    *, free_pages, fake_dist, attn_tp_size=2, layout=K3_LAYOUT, rank=0,
    token_cap=TOKEN_CAP,
):
    """A fake worker holding only what the pool planner reads."""
    methods = _worker_methods(
        [
            "_plan_prefill_pool_wave",
            "_check_pool_plan_identical",
            "_prefill_pool_layout",
            "_prefill_pool_psm_layout",
            "_prefill_pool_page_dims",
            "_prefill_pool_group",
            "_prefill_pool_free_state_slots",
            "_replace_prefix_lookup_hits",
            # M3 Slice 3 plan input: with the trajectory flag off these
            # return "no hits" and the plan is the pre-Slice-3 one.
            "_materialized_prefix_hits",
            "_unmaterializable_frontier",
            "_trajectory_commit_requested",
            "_owns_host_kv",
            "_decode_attn_tp_size",
            "_construct_prefill_pool",
            "_prefill_pool_page_bytes",
            # The executor and the pool are built with the batch-level probes,
            # read through the shared batchgen_debug seam.
            "_pool_pad_poison_enabled",
            "_new_pool_kernel_dump_recorder",
            "_active_batchgen_debug_for_sequences",
            "_debug_flag_enabled",
        ],
        {
            "torch": _fake_torch(free_pages * K3_PAGE_BYTES),
            "dist": fake_dist,
            # The pool layout branches on the DSA dual-region coordinator; a
            # single-region model (GPT-OSS, K3) is not an instance of it.
            "DualHostKVCoordinator": type("DualHostKVCoordinator", (), {}),
        },
    )
    parallel_manager = SimpleNamespace(
        attn_tp_size=attn_tp_size,
        attn_tp_group="group" if attn_tp_size > 1 else None,
    )
    if layout is not None:
        parallel_manager.prefill_pool_layout = lambda: dict(layout)
        parallel_manager.supports_group_pool_prefill = lambda: True
    worker = SimpleNamespace(
        rank=rank,
        local_rank=0,
        world_size=attn_tp_size,
        torch_device="cpu",
        huggingface_ckpt_name="moonshotai/Kimi-K3",
        parallel_manager=parallel_manager,
        engine_config=SimpleNamespace(Module_Batching_Config=SimpleNamespace(
            prefill_micro_batch_token_cap=token_cap,
            prefill_prefix_pool_workspace_bytes=0,
        )),
        prefix_cache_runtime_config=SimpleNamespace(
            group_specs=(SimpleNamespace(raw_page_tokens=PAGE_TOKENS),),
        ),
        host_paged_kv_worker_view=object(),
        global_batch=SimpleNamespace(get_sequence=_seqs().get),
        prefix_cache_coordinator=RecordingCoordinator(),
        _prefix_sequence_states={},
        _prefill_pool_base=(0, 0, 0),
        # No trajectory frontier in play: no attachment, no mirrored index.
        _trajectory_attachments={},
        _trajectory_session_groups=None,
        # No worker-wide batchgen_debug: the batch-level flags are the only
        # source, and none of these sequences sets one.
        _batchgen_debug=None,
        # Request-level flags ARE the source here, so the per-sequence merge
        # must stay reachable (the worker latches this at admission).
        _any_sequence_batchgen_debug=True,
        # The planner reads the cached prefill-phase page budget (pages) and
        # prompts through the trajectory-book contract.
        _prefill_pool_page_budget=free_pages,
    )
    worker._trajectory_tokens = (
        lambda seq, length=None, **_: seq.input_ids[0, :length]
    )
    return _bind(worker, methods)


# --------------------------------------------------------------------------- #
#  (i) The page and state budgets are the group MIN, and the pool is sized
#      from the PSM layout, not from the Host geometry
# --------------------------------------------------------------------------- #


def _seqs_with_debug(debug):
    """The wave's sequences, each carrying one batch-level debug dict."""
    seqs = _seqs()
    for seq in seqs.values():
        seq.batchgen_debug = dict(debug)
    return seqs


def _pool_with_debug(monkeypatch, debug):
    _stub_heavy_modules(monkeypatch, free_state_slots=32)
    worker = _plan_worker(free_pages=20, fake_dist=FakeDist(peer_budget=[16, 32]))
    if debug is not None:
        worker.global_batch = SimpleNamespace(
            get_sequence=_seqs_with_debug(debug).get
        )
    wave = worker._plan_prefill_pool_wave(UUIDS, {})
    assert wave is not None
    worker._construct_prefill_pool(wave)
    return wave


def test_the_executor_is_built_without_the_pad_poison_probe(monkeypatch):
    """Production default: no batch asked, so the page table pads with 0."""
    wave = _pool_with_debug(monkeypatch, None)

    assert wave.executor.pad_poison is False


def test_a_batch_can_arm_the_pad_poison_probe(monkeypatch):
    """``batchgen_debug.prefix_pool_pad_poison`` reaches the chunk executor."""
    wave = _pool_with_debug(monkeypatch, {"prefix_pool_pad_poison": True})

    assert wave.executor.pad_poison is True


def test_another_batch_flag_does_not_arm_the_pad_poison_probe(monkeypatch):
    """Only its own flag arms it; an unrelated debug dict leaves it off."""
    wave = _pool_with_debug(monkeypatch, {"prefix_pool_offload_check": True})

    assert wave.executor.pad_poison is False


@pytest.mark.parametrize("mode", ["sync", "nosync"])
def test_a_batch_can_arm_the_kernel_dump_in_either_mode(
    monkeypatch, tmp_path, mode
):
    """``prefix_pool_dump_kernel_args`` reaches the pool, with its dump dir."""
    wave = _pool_with_debug(monkeypatch, {
        "prefix_pool_dump_kernel_args": mode,
        "prefix_pool_dump_dir": str(tmp_path),
    })

    recorder = wave.pool.kernel_dump
    assert recorder is not None
    assert recorder.mode == mode
    assert recorder.rank == 0
    assert recorder.directory == str(tmp_path / "pool_dump_rank0")
    assert recorder.page_tokens == wave.page_tokens
    # Row metadata can name a live member only if the global ids came along.
    assert list(recorder.global_ids) == list(wave.global_ids)


def test_the_kernel_dump_is_off_by_default(monkeypatch):
    assert _pool_with_debug(monkeypatch, None).pool.kernel_dump is None


def test_another_batch_flag_does_not_arm_the_kernel_dump(monkeypatch):
    wave = _pool_with_debug(monkeypatch, {"prefix_pool_pad_poison": True})

    assert wave.pool.kernel_dump is None


def test_a_mistyped_dump_mode_is_refused_at_pool_construction(monkeypatch):
    """Not silently off: a typo must fail the wave, not look like a clean run."""
    with pytest.raises(ValueError, match="prefix_pool_dump_kernel_args"):
        _pool_with_debug(
            monkeypatch, {"prefix_pool_dump_kernel_args": "synch"}
        )


def test_pool_budget_is_the_group_minimum(monkeypatch):
    _stub_heavy_modules(monkeypatch, free_state_slots=32)
    # This rank could hold 20 pages; the tightest rank of its group only 16.
    fake_dist = FakeDist(peer_budget=[16, 32])
    worker = _plan_worker(free_pages=20, fake_dist=fake_dist)

    wave = worker._plan_prefill_pool_wave(UUIDS, {})

    # Pages, state slots and the M3 Slice 3 materialization vote (1 = this
    # rank has nothing to object to) travel in the one reduce.
    assert fake_dist.reduced == [[20, 32, 1]], (
        "the rank must contribute its own budget"
    )
    assert wave is not None
    assert wave.total_pages == 16
    # The pool is built in the prefill phase; the group's 16 pages are each the
    # PSM's expanded 192/128-style page (320 B here), not the Host latent page.
    worker._construct_prefill_pool(wave)
    assert wave.pool.num_pages == 16
    assert wave.pool.page_bytes == K3_PAGE_BYTES
    assert wave.pool.k.shape == (2, 16, PAGE_TOKENS, 2, 6)
    assert wave.pool.v.shape == (2, 16, PAGE_TOKENS, 2, 4)
    # Every chunk token lives in the pool, so the chunk is capped by the pages
    # left over for the segments those tails read: (16 - 2) // 2 pages.
    assert wave.plan.chunk_tokens == 28 < TOKEN_CAP
    assert len(wave.plan.segments) == 1
    assert wave.plan.segments[0].token_end == len(SHARED)


def test_host_geometry_still_sizes_a_model_without_a_psm_layout(monkeypatch):
    # GPT-OSS/GLM declare no layout, so the pool stays the Host geometry's and
    # no chunk cap is applied: byte-identical to the pre-S3 planner.
    _stub_heavy_modules(monkeypatch, free_state_slots=None)
    fake_dist = FakeDist()
    worker = _plan_worker(
        free_pages=20, fake_dist=fake_dist, attn_tp_size=1, layout=None,
    )
    # The cached budget is already in pool pages; without a PSM layout the
    # page is the Host geometry's.
    host_page_bytes = 2 * PAGE_TOKENS * 1 * (8 + 8) * 2

    wave = worker._plan_prefill_pool_wave(UUIDS, {})

    assert fake_dist.reduced == [] and fake_dist.gathered == []
    assert wave.total_pages == 20
    worker._construct_prefill_pool(wave)
    assert wave.pool.page_bytes == host_page_bytes
    assert wave.pool.num_pages == 20
    assert wave.pool.k.shape[-2:] == (1, 8)
    assert wave.plan.chunk_tokens == TOKEN_CAP


# --------------------------------------------------------------------------- #
#  (ii) A plan that is not identical across the group is named, not run
# --------------------------------------------------------------------------- #


def test_diverging_plan_digest_raises(monkeypatch):
    _stub_heavy_modules(monkeypatch, free_state_slots=32)
    fake_dist = FakeDist(peer_budget=[16, 32], peer_digest=1234567)
    worker = _plan_worker(free_pages=20, fake_dist=fake_dist)

    with pytest.raises(RuntimeError, match="diverged inside the serve group"):
        worker._plan_prefill_pool_wave(UUIDS, {})

    assert len(fake_dist.gathered) == 1


def test_identical_plans_pass_the_digest_check(monkeypatch):
    _stub_heavy_modules(monkeypatch, free_state_slots=32)
    fake_dist = FakeDist(peer_budget=[16, 32])
    worker = _plan_worker(free_pages=20, fake_dist=fake_dist)

    assert worker._plan_prefill_pool_wave(UUIDS, {}) is not None
    assert len(fake_dist.gathered) == 1


# --------------------------------------------------------------------------- #
#  (iii) The recurrent-state budget falls back uniformly over the group
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("rank", [0, 1])
def test_too_few_state_slots_drops_the_pool_on_every_rank(monkeypatch, rank):
    # One segment plus two sequences need three slots; the tightest rank of the
    # group has two, so EVERY rank must take the existing path.
    _stub_heavy_modules(monkeypatch, free_state_slots=32 if rank else 2)
    fake_dist = FakeDist(peer_budget=[16, 2])
    worker = _plan_worker(free_pages=20, fake_dist=fake_dist, rank=rank)

    assert worker._plan_prefill_pool_wave(UUIDS, {}) is None
    # A declined plan leaves the lookup states untouched; the CALLER releases
    # the hits for a pool-only model (K3), uniformly across the group.
    assert worker._prefix_sequence_states == {}


def test_enough_state_slots_keeps_the_pool(monkeypatch):
    _stub_heavy_modules(monkeypatch, free_state_slots=3)
    worker = _plan_worker(
        free_pages=20, fake_dist=FakeDist(peer_budget=[16, 3]),
    )

    wave = worker._plan_prefill_pool_wave(UUIDS, {})

    assert wave is not None
    assert len(wave.plan.segments) + len(PROMPTS) == 3


def test_a_model_without_recurrent_state_never_checks_slots(monkeypatch):
    # -1 means "no such state"; the comparison must not fire on it.
    _stub_heavy_modules(monkeypatch, free_state_slots=None)
    worker = _plan_worker(free_pages=20, fake_dist=FakeDist(peer_budget=[16, -1]))

    assert worker._prefill_pool_free_state_slots() == -1
    assert worker._plan_prefill_pool_wave(UUIDS, {}) is not None


# --------------------------------------------------------------------------- #
#  (iv) Segment Host pages: the leader allocates, the followers mirror
# --------------------------------------------------------------------------- #


def _real_wave():
    plan = plan_wave_prefix_sharing(
        PROMPTS, block_tokens=PAGE_TOKENS, pool_pages=8, chunk_tokens=28,
    )
    assert len(plan.segments) == 1
    pool = PrefillPrefixPool(
        num_layers=2, num_pages=16, page_tokens=PAGE_TOKENS, num_kv_heads=2,
        head_dim=6, dtype=torch.bfloat16, device="cpu", v_head_dim=4,
    )
    return wave_runtime.PoolWave(
        plan=plan, prompts=PROMPTS, uuids=list(UUIDS), global_ids=list(GLOBAL_IDS),
        page_tokens=pool.page_tokens, total_pages=pool.num_pages,
        pool=pool, executor=PoolChunkExecutor(plan, PROMPTS, pool),
    )


def _mirror_worker(rank, *, views, wave, fake_dist, states=None):
    methods = _worker_methods(
        [
            "_mirror_group_host_kv",
            "_allocate_prefill_pool_host_pages",
            "_prefill_pool_segment_host_pages",
            "_next_pool_segment_sid",
            "_owns_host_kv",
            "_decode_attn_tp_size",
        ],
        {"dist": fake_dist},
    )
    worker = SimpleNamespace(
        rank=rank,
        world_size=2,
        local_world_size=8,
        parallel_manager=SimpleNamespace(attn_tp_size=2),
        global_batch=SimpleNamespace(get_sequence=_seqs().get),
        host_paged_kv_worker_view=views[0],
        _prefix_sequence_states=states if states is not None else {},
        _host_views_by_group=lambda: views,
        _prefill_pool_wave=wave,
        _pool_member_sids={},
        _pool_sid_members={},
        _pool_sid_counter=0,
    )
    return _bind(worker, methods)


def _fresh_states():
    return {
        global_id: PrefixCacheSequenceState(
            lookup_result=None, attached_tokens=len(SHARED),
            compute_cached_tokens=len(SHARED),
        )
        for global_id in GLOBAL_IDS
    }


def test_follower_installs_the_leaders_segment_pages(monkeypatch):
    _stub_heavy_modules(monkeypatch, free_state_slots=32)
    backends = {0: FakeBackend(1000)}
    views_by_rank = {
        rank: {group: FakeView(backend, rank) for group, backend in backends.items()}
        for rank in (0, 1)
    }
    fake_dist = FakeDist(world_size=2)
    waves, workers, states = {}, {}, {}

    for rank in (0, 1):
        fake_dist.rank = rank
        waves[rank] = _real_wave()
        states[rank] = _fresh_states()
        worker = _mirror_worker(
            rank, views=views_by_rank[rank], wave=waves[rank],
            fake_dist=fake_dist, states=dict(states[rank]),
        )
        workers[rank] = worker
        local_states = dict(states[rank])
        if worker._owns_host_kv(_seqs()["a"]):
            worker._allocate_prefill_pool_host_pages(
                views_by_rank[rank], local_states
            )
        worker._mirror_group_host_kv(UUIDS, local_states)
        worker.states = local_states

    leader, follower = waves[0], waves[1]
    # The pseudo sequence ids and their pages are the leader's on both ranks.
    assert leader.segment_sids and follower.segment_sids == leader.segment_sids
    assert follower.segment_host_pages == leader.segment_host_pages
    sid = leader.segment_sids[0]
    assert sid < 0
    assert views_by_rank[1][0].table[sid] == views_by_rank[0][0].table[sid]
    assert len(views_by_rank[0][0].table[sid]) == len(SHARED) // PAGE_TOKENS
    # Nothing was allocated a second time: the follower only set a page table.
    assert views_by_rank[1][0].backend is views_by_rank[0][0].backend
    assert workers[1]._pool_member_sids == {}
    # The follower also took the leader's chain for each member sequence.
    for global_id in GLOBAL_IDS:
        assert workers[1].states[global_id].shared_page_ids_for_group(0) == tuple(
            leader.segment_host_pages[0][0]
        )


def test_follower_refuses_a_leader_that_planned_no_wave(monkeypatch):
    # A leader without a wave while this rank has one is a plan divergence the
    # digest check cannot see (it ran before the early returns), so name it.
    _stub_heavy_modules(monkeypatch, free_state_slots=32)
    backends = {0: FakeBackend(1000)}
    fake_dist = FakeDist(world_size=2)

    fake_dist.rank = 0
    leader = _mirror_worker(
        0, views={0: FakeView(backends[0], 0)}, wave=None, fake_dist=fake_dist,
    )
    leader._mirror_group_host_kv(UUIDS, _fresh_states())

    fake_dist.rank = 1
    follower = _mirror_worker(
        1, views={0: FakeView(backends[0], 1)}, wave=_real_wave(),
        fake_dist=fake_dist,
    )
    with pytest.raises(RuntimeError, match="planned no prefix pool wave"):
        follower._mirror_group_host_kv(UUIDS, {})


# --------------------------------------------------------------------------- #
#  (v) Commit and release are the leader's; a follower drops local state only
# --------------------------------------------------------------------------- #


def _finish_worker(rank, *, views, wave, states, coordinator):
    methods = _worker_methods(
        [
            "_finish_prefill_pool_wave",
            "_release_pool_segment_refs",
            "_owns_host_kv",
            "_decode_attn_tp_size",
        ],
    )
    worker = SimpleNamespace(
        rank=rank,
        parallel_manager=SimpleNamespace(attn_tp_size=2),
        global_batch=SimpleNamespace(get_sequence=_seqs().get),
        prefix_cache_coordinator=coordinator,
        prefix_cache_runtime_config=SimpleNamespace(
            namespace_digest=(1, 2, 3, 4), publish_boundary_tokens=PAGE_TOKENS,
            max_nodes=64,
        ),
        _prefix_sequence_states=states,
        _prefill_pool_wave=wave,
        _pool_member_sids={},
        _pool_sid_members={},
        _host_views_by_group=lambda: views,
    )
    return _bind(worker, methods)


@pytest.fixture
def recorded_commit(monkeypatch):
    """Record the coordinator side of the wave finish instead of running it."""
    calls = {"commit": [], "attach": []}
    monkeypatch.setattr(
        wave_runtime, "commit_pool_segments",
        lambda wave, **kwargs: calls["commit"].append(wave) or [7],
    )
    monkeypatch.setattr(
        wave_runtime, "attach_pool_members",
        lambda wave, **kwargs: calls["attach"].append(wave) or {0: 11, 1: 12},
    )
    return calls


def test_leader_commits_the_wave_and_keeps_its_pages(monkeypatch, recorded_commit):
    _stub_heavy_modules(monkeypatch, free_state_slots=32)
    backend = FakeBackend(1000)
    views = {0: FakeView(backend, 0)}
    wave = _real_wave()
    wave.segment_sids = {0: -1}
    wave.segment_host_pages = {0: {0: [1000, 1001]}}
    states = _fresh_states()
    coordinator = RecordingCoordinator()
    worker = _finish_worker(
        0, views=views, wave=wave, states=states, coordinator=coordinator,
    )

    worker._finish_prefill_pool_wave()

    assert recorded_commit["commit"] == [wave] and recorded_commit["attach"] == [wave]
    assert coordinator.released == [7]  # the chain handles the commit pinned
    assert backend.unregistered == [] and backend.released == []
    for index, global_id in enumerate(GLOBAL_IDS):
        assert states[global_id].commit_attachment_handles == (11 + index,)
    assert worker._prefill_pool_wave is None


def test_follower_finish_drops_its_mirror_without_committing(
    monkeypatch, recorded_commit,
):
    _stub_heavy_modules(monkeypatch, free_state_slots=32)
    backend = FakeBackend(1000)
    views = {0: FakeView(backend, 1)}
    views[0].set_sequence_pages(-1, [1000, 1001])
    wave = _real_wave()
    wave.segment_sids = {0: -1}
    wave.segment_host_pages = {0: {0: [1000, 1001]}}
    states = _fresh_states()
    coordinator = RecordingCoordinator()
    worker = _finish_worker(
        1, views=views, wave=wave, states=states, coordinator=coordinator,
    )

    worker._finish_prefill_pool_wave()

    assert recorded_commit == {"commit": [], "attach": []}
    assert coordinator.released == []
    # Local record gone, shared pages untouched.
    assert backend.unregistered == [(1, [-1])]
    assert backend.released == []
    assert views[0].table == {}
    assert worker._prefill_pool_wave is None
    # And a later member release frees nothing: the refcount is the leader's.
    worker._release_pool_segment_refs(GLOBAL_IDS[0])
    assert backend.released == []


def test_leader_release_frees_segment_pages_after_its_last_member(monkeypatch):
    _stub_heavy_modules(monkeypatch, free_state_slots=32)
    backend = FakeBackend(1000)
    views = {0: FakeView(backend, 0)}
    worker = _finish_worker(
        0, views=views, wave=None, states={}, coordinator=RecordingCoordinator(),
    )
    worker._pool_member_sids = {gid: (-1,) for gid in GLOBAL_IDS}
    worker._pool_sid_members = {-1: 2}

    worker._release_pool_segment_refs(GLOBAL_IDS[0])
    assert backend.released == []  # one member still reads the segment

    worker._release_pool_segment_refs(GLOBAL_IDS[1])
    assert backend.released == [(0, [-1])]


# --------------------------------------------------------------------------- #
#  (vi) A pool wave runs one model pass per planned chunk
# --------------------------------------------------------------------------- #


def _pass_count_worker(wave, *, enable_prepack=False, micro_batch_size=4):
    methods = _worker_methods(["_prefill_model_pass_count"])
    worker = SimpleNamespace(
        rank=0,
        enable_prepack=enable_prepack,
        engine_config=SimpleNamespace(Module_Batching_Config=SimpleNamespace(
            MoE_prefill_micro_batch_size=micro_batch_size,
        )),
        _prefill_pool_wave=wave,
    )
    return _bind(worker, methods)


def test_pass_count_equals_the_planned_chunk_count():
    wave = _real_wave()
    worker = _pass_count_worker(wave)

    # Not the number of whole prompts: a chunk packs pooled segments and tails.
    assert worker._prefill_model_pass_count([0, 1]) == len(wave.plan.chunks)
    assert len(wave.plan.chunks) >= 2
    assert worker._prefill_model_pass_count([]) == 0


def test_pass_count_without_a_wave_is_unchanged():
    worker = _pass_count_worker(None, micro_batch_size=2)

    assert worker._prefill_model_pass_count([0, 1, 2]) == 2


# --------------------------------------------------------------------------- #
#  (vii) The gate: G>1 only for a PSM that declares a group-safe pool prefill
# --------------------------------------------------------------------------- #


def _gate_worker(
    *, attn_tp_size, semantic=PrefixKVGroupSemantic.FULL_KV, workspace=1 << 20,
    group_safe=None, resident_ep=None, enable_prefix_cache=True, integrity=False,
):
    methods = _worker_methods(
        ["_prefill_prefix_pool_enabled", "_pool_prefill_group_safe",
         "_decode_attn_tp_size"],
    )
    parallel_manager = SimpleNamespace(attn_tp_size=attn_tp_size)
    if group_safe is not None:
        parallel_manager.supports_group_pool_prefill = lambda: group_safe
    if resident_ep is not None:
        parallel_manager.prefill_uses_resident_ep = lambda: resident_ep
    worker = SimpleNamespace(
        enable_prefix_cache=enable_prefix_cache,
        _prefix_integrity=integrity,
        parallel_manager=parallel_manager,
        engine_config=SimpleNamespace(Module_Batching_Config=SimpleNamespace(
            prefill_prefix_pool_workspace_bytes=workspace,
        )),
        prefix_cache_runtime_config=SimpleNamespace(
            group_specs=(SimpleNamespace(semantic=semantic),),
        ),
    )
    return _bind(worker, methods)


def test_g1_gate_is_unchanged():
    # GPT-OSS at G==1: a planner workspace turns the pool on, none leaves it off.
    assert _gate_worker(attn_tp_size=1)._prefill_prefix_pool_enabled()
    assert not _gate_worker(
        attn_tp_size=1, workspace=0
    )._prefill_prefix_pool_enabled()
    assert not _gate_worker(
        attn_tp_size=1, enable_prefix_cache=False
    )._prefill_prefix_pool_enabled()
    # An MLA-latent model is pool-only, so no workspace is a configuration bug.
    with pytest.raises(RuntimeError, match="needs the prefill prefix pool"):
        _gate_worker(
            attn_tp_size=1, semantic=PrefixKVGroupSemantic.MLA_LATENT, workspace=0,
        )._prefill_prefix_pool_enabled()


def test_g_gt_1_is_refused_unless_the_psm_declares_it():
    for kwargs in ({}, {"group_safe": False}):
        worker = _gate_worker(attn_tp_size=8, **kwargs)
        with pytest.raises(RuntimeError, match="runs per DP rank"):
            worker._prefill_prefix_pool_enabled()


def test_g_gt_1_is_admitted_for_a_group_safe_psm():
    worker = _gate_worker(
        attn_tp_size=8, semantic=PrefixKVGroupSemantic.MLA_LATENT,
        group_safe=True, resident_ep=False,
    )
    assert worker._pool_prefill_group_safe()
    assert worker._prefill_prefix_pool_enabled()


def test_resident_ep_prefill_is_still_refused():
    worker = _gate_worker(
        attn_tp_size=8, semantic=PrefixKVGroupSemantic.MLA_LATENT,
        group_safe=True, resident_ep=True,
    )
    with pytest.raises(RuntimeError, match="joins cross-rank collectives"):
        worker._prefill_prefix_pool_enabled()


# --------------------------------------------------------------------------- #
#  (viii) The wave is planned outside the leader gate, and no collective of
#         the pool sits under a rank-dependent condition
# --------------------------------------------------------------------------- #


def _parents(method):
    parents = {}
    for node in ast.walk(method):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def test_wave_planning_is_reached_outside_the_leader_gate():
    # Planning now lives in the reservation-time attach: every SERVING rank
    # plans (the G>1 plan is a group collective); only the host-KV leader
    # looks up and attaches.
    method = _worker_method_node("_attach_prefix_cache_for_prefill")
    parents = _parents(method)
    calls = [
        node for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_plan_prefill_pool_wave"
    ]
    assert len(calls) == 1
    node = calls[0]
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.If):
            test = ast.unparse(node.test)
            assert "owns_host_kv" not in test, (
                f"the pool wave is planned under {test!r}, so only the serve "
                "group's leader would reach its budget reduction"
            )


def test_pool_collectives_are_not_rank_conditional():
    for name, collective in (
        ("_plan_prefill_pool_wave", "all_reduce"),
        ("_check_pool_plan_identical", "all_gather"),
    ):
        method = _worker_method_node(name)
        parents = _parents(method)
        calls = [
            node for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == collective
        ]
        assert len(calls) == 1, name
        node = calls[0]
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.If):
                test = ast.unparse(node.test)
                for rank_dependent in ("self.rank", "_owns_host_kv", "local_rank"):
                    assert rank_dependent not in test, (
                        f"{name}: {collective} sits under {test!r}"
                    )
