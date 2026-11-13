"""A prefix-hit micro-batch's temporary full-length GPU KV fits its budget.

The cross-wave prefix-hit prefill materializes temporary GPU KV for the FULL
prompt length of every member of a micro-batch, while the token cap bounds
only the computed (suffix) tokens. The micro-batch planner therefore caps the
transient pages of any micro-batch holding a hit against the prefill-phase
prefix KV budget, and the temporary manager is sized by the same function.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional, Tuple

import pytest
import torch

from batchgen.prefill.prepack import (
    PrefixMaterializationExceedsBudget,
    build_prefill_micro_batches,
)
from batchgen.prefix_reuse.materialization import (
    gpu_kv_page_bytes,
    materialize_gpt_oss_prefixes,
    temporary_gpu_kv_config,
    temporary_kv_page_tokens,
    temporary_kv_pages,
)
from tests.worker_method_source import WORKER_SOURCE, load_worker_methods


@dataclass(frozen=True)
class _KVConfig:
    """The GPUPagedKVConfig fields the sizing reads (the real one needs CUDA
    extension imports)."""

    num_layers: int
    num_pages: int
    page_size_tokens: int
    num_k_heads: int
    k_head_dim: int
    num_v_heads: Optional[int]
    v_head_dim: Optional[int]
    kv_dtype: torch.dtype

    @property
    def has_v_cache(self) -> bool:
        return bool(self.num_v_heads and self.v_head_dim)


PROFILE = _KVConfig(
    num_layers=2, num_pages=1, page_size_tokens=64, num_k_heads=1,
    k_head_dim=4, num_v_heads=1, v_head_dim=4, kv_dtype=torch.bfloat16,
)


# ---------------------------------------------------------------- sizing


def test_temporary_config_is_fa_aligned_and_holds_every_full_length():
    config = temporary_gpu_kv_config(PROFILE, [1, 256, 257, 600])
    assert config.page_size_tokens == 256
    assert config.num_pages == 1 + 1 + 2 + 3
    assert temporary_kv_page_tokens(300) == 512
    assert temporary_kv_pages(0, 256) == 1


def test_page_bytes_match_the_tensors_initialize_allocates():
    config = temporary_gpu_kv_config(PROFILE, [512])
    k = torch.empty(
        (config.num_layers, config.num_pages, config.page_size_tokens,
         config.num_k_heads, config.k_head_dim), dtype=config.kv_dtype,
    )
    v = torch.empty(
        (config.num_layers, config.num_pages, config.page_size_tokens,
         config.num_v_heads, config.v_head_dim), dtype=config.kv_dtype,
    )
    total = k.numel() * k.element_size() + v.numel() * v.element_size()
    assert gpu_kv_page_bytes(config) * config.num_pages == total


def test_page_bytes_without_v_cache_count_k_only():
    config = _KVConfig(
        num_layers=3, num_pages=1, page_size_tokens=256, num_k_heads=1,
        k_head_dim=8, num_v_heads=0, v_head_dim=0, kv_dtype=torch.bfloat16,
    )
    assert gpu_kv_page_bytes(config) == 3 * 256 * 8 * 2


# --------------------------------------------------------------- planner


def _transient_pages(micro_batches, hits, pages):
    return [
        sum(pages[start:end]) if any(hits[start:end]) else 0
        for start, end in micro_batches
    ]


def test_hits_that_fit_keep_the_token_plan():
    lengths = [10, 20, 30]
    plain, _ = build_prefill_micro_batches(lengths, 100, l2_balance=False)
    planned, _ = build_prefill_micro_batches(
        lengths, 100, l2_balance=False,
        prefix_hits=[True, False, True], kv_pages=[3, 2, 4], kv_page_cap=9,
    )
    assert planned == plain == [(0, 3)]


def test_hits_split_until_every_transient_fits():
    # Short suffixes, long prompts: the token cap alone packs all six rows in
    # one micro-batch whose full-length KV is 6 * 5 = 30 pages.
    lengths = [8] * 6
    hits = [True, False, True, True, False, True]
    pages = [5] * 6
    plain, _ = build_prefill_micro_batches(lengths, 1000, l2_balance=False)
    assert plain == [(0, 6)]
    planned, _ = build_prefill_micro_batches(
        lengths, 1000, l2_balance=False,
        prefix_hits=hits, kv_pages=pages, kv_page_cap=12,
    )
    assert planned == [(0, 2), (2, 4), (4, 6)]
    assert max(_transient_pages(planned, hits, pages)) <= 12
    # Contiguous, covering, order-preserving.
    assert planned[0][0] == 0 and planned[-1][1] == 6
    assert all(a[1] == b[0] for a, b in zip(planned, planned[1:]))


def test_miss_rows_count_only_inside_a_hit_micro_batch():
    # A miss row larger than the cap is fine on its own: a micro-batch with
    # no hit materializes nothing. Next to a hit it must split off.
    planned, _ = build_prefill_micro_batches(
        [4, 4, 4], 1000, l2_balance=False,
        prefix_hits=[False, False, True], kv_pages=[50, 50, 3], kv_page_cap=10,
    )
    assert planned == [(0, 2), (2, 3)]
    assert _transient_pages(planned, [False, False, True], [50, 50, 3]) == [0, 3]


def test_single_hit_sequence_above_the_budget_is_a_named_error():
    with pytest.raises(PrefixMaterializationExceedsBudget, match="sequence 1"):
        build_prefill_micro_batches(
            [4, 4], 1000, l2_balance=False,
            prefix_hits=[False, True], kv_pages=[1, 11], kv_page_cap=10,
        )
    with pytest.raises(PrefixMaterializationExceedsBudget):
        build_prefill_micro_batches(
            [4, 4], 1000, single_sequence_only=True,
            prefix_hits=[True, False], kv_pages=[11, 1], kv_page_cap=10,
        )


def test_no_hits_leave_the_plan_untouched():
    lengths = [100, 3000, 50, 7000, 20]
    for l2 in (False, True):
        plain = build_prefill_micro_batches(lengths, 4096, l2_balance=l2)
        planned = build_prefill_micro_batches(
            lengths, 4096, l2_balance=l2,
            prefix_hits=[False] * 5, kv_pages=[10**6] * 5, kv_page_cap=0,
        )
        assert planned == plain


def test_hits_and_pages_must_come_together_with_one_entry_per_sequence():
    with pytest.raises(ValueError, match="together"):
        build_prefill_micro_batches([4], 10, prefix_hits=[True])
    with pytest.raises(ValueError, match="one entry per sequence"):
        build_prefill_micro_batches(
            [4, 4], 10, prefix_hits=[True], kv_pages=[1], kv_page_cap=4,
        )


# ------------------------------------------------- worker planning seam


_PLAN = load_worker_methods(
    [
        "_plan_prefill_micro_batches",
        "_prefill_model_pass_count",
        "_measure_prefill_prefix_kv_budget",
    ],
    {
        "os": os,
        "List": List,
        "Tuple": Tuple,
        "build_prefill_micro_batches": build_prefill_micro_batches,
        "logging": SimpleNamespace(info=lambda *a, **k: None),
        "math": __import__("math"),
    },
)


class _PlanWorker:
    def __init__(self, *, budget, attn_tp=1, token_cap=10**6):
        self.engine_config = SimpleNamespace(
            Module_Batching_Config=SimpleNamespace(
                prefill_micro_batch_token_cap=token_cap,
                prefill_prefix_pool_workspace_bytes=0,
            )
        )
        self.parallel_manager = SimpleNamespace()
        self._prefill_prefix_kv_budget_bytes = budget
        self._attn_tp = attn_tp
        self.kv_config_calls = 0

    def _decode_attn_tp_size(self):
        return self._attn_tp

    def _prefix_materialization_kv_config(self, full_seq_lengths):
        self.kv_config_calls += 1
        return temporary_gpu_kv_config(PROFILE, full_seq_lengths)

    def _plan_prefill_micro_batches(self, *args):
        return _PLAN["_plan_prefill_micro_batches"](self, *args)


PAGE_BYTES = gpu_kv_page_bytes(temporary_gpu_kv_config(PROFILE, [1]))


def test_worker_caps_hit_micro_batches_by_budget_over_page_bytes():
    worker = _PlanWorker(budget=12 * PAGE_BYTES + PAGE_BYTES - 1)
    full = [1280] * 6  # 5 pages of 256 tokens each
    cached = [1272, 0, 1272, 1272, 0, 1272]
    suffix = [f - c for f, c in zip(full, cached)]
    micro_batches, _, _ = worker._plan_prefill_micro_batches(suffix, full, cached)
    assert micro_batches == [(0, 2), (2, 4), (4, 6)]
    for start, end in micro_batches:
        config = temporary_gpu_kv_config(PROFILE, full[start:end])
        assert config.num_pages * gpu_kv_page_bytes(config) <= (
            worker._prefill_prefix_kv_budget_bytes
        )


def test_worker_without_hits_never_reads_the_budget_or_geometry():
    worker = _PlanWorker(budget=None)
    micro_batches, _, _ = worker._plan_prefill_micro_batches(
        [400, 400, 400], [400, 400, 400], [0, 0, 0]
    )
    assert micro_batches == build_prefill_micro_batches(
        [400, 400, 400], 10**6, l2_balance=True
    )[0]
    assert worker.kv_config_calls == 0


def test_worker_hit_without_budget_fails_loud():
    worker = _PlanWorker(budget=None)
    with pytest.raises(RuntimeError, match="no prefill-phase prefix KV budget"):
        worker._plan_prefill_micro_batches([8], [1280], [1272])


def test_worker_hit_on_a_tp_serve_group_fails_loud():
    worker = _PlanWorker(budget=10**12, attn_tp=4)
    with pytest.raises(RuntimeError, match="rank-local budget"):
        worker._plan_prefill_micro_batches([8], [1280], [1272])


def test_worker_single_sequence_above_budget_is_named_before_any_forward():
    worker = _PlanWorker(budget=4 * PAGE_BYTES)
    with pytest.raises(PrefixMaterializationExceedsBudget):
        worker._plan_prefill_micro_batches([8], [1280], [1272])


def test_pass_count_plans_with_the_inputs_execution_plans_with():
    worker = _PlanWorker(budget=12 * PAGE_BYTES)
    full = [1280] * 6
    cached = [1272, 0, 1272, 1272, 0, 1272]
    seqs = {
        f"u{i}": SimpleNamespace(global_idx=100 + i, prompt_length=full[i])
        for i in range(6)
    }
    worker._prefill_pool_wave = None
    worker.enable_prepack = True
    worker._local_to_uuid_map = {i: f"u{i}" for i in range(6)}
    worker.global_batch = SimpleNamespace(get_sequence=seqs.__getitem__)
    worker._prefix_sequence_states = {
        100 + i: SimpleNamespace(compute_cached_tokens=cached[i])
        for i in range(6)
        if cached[i]
    }
    count = _PLAN["_prefill_model_pass_count"](worker, list(range(6)))
    executed, _, _ = worker._plan_prefill_micro_batches(
        [f - c for f, c in zip(full, cached)], full, cached
    )
    assert count == len(executed) == 3


class _FakeCuda:
    def __init__(self, free):
        self.free = free

    def synchronize(self, device):
        pass

    def empty_cache(self):
        pass

    def mem_get_info(self, device):
        return self.free, 4 * self.free


@pytest.mark.parametrize("pool_enabled", [False, True])
def test_budget_is_free_memory_beside_the_workspace(pool_enabled):
    measure = load_worker_methods(
        ["_measure_prefill_prefix_kv_budget"],
        {
            "torch": SimpleNamespace(cuda=_FakeCuda(free=100 * 2**30)),
            "logging": SimpleNamespace(info=lambda *a, **k: None),
        },
    )["_measure_prefill_prefix_kv_budget"]
    worker = SimpleNamespace(
        rank=0, torch_device="cuda:0", local_rank=0,
        engine_config=SimpleNamespace(
            Module_Batching_Config=SimpleNamespace(
                prefill_prefix_pool_workspace_bytes=16 * 2**30,
            )
        ),
        _prefill_prefix_kv_budget_bytes=None,
        _prefill_pool_page_budget=None,
        _prefill_prefix_pool_enabled=lambda: pool_enabled,
        _prefill_pool_page_bytes=lambda: 2**30,
    )
    measure(worker)
    assert worker._prefill_prefix_kv_budget_bytes == 84 * 2**30
    assert worker._prefill_pool_page_budget == (84 if pool_enabled else None)


def test_no_workspace_means_no_budget():
    measure = load_worker_methods(
        ["_measure_prefill_prefix_kv_budget"],
        {
            "torch": SimpleNamespace(cuda=_FakeCuda(free=100 * 2**30)),
            "logging": SimpleNamespace(info=lambda *a, **k: None),
        },
    )["_measure_prefill_prefix_kv_budget"]
    worker = SimpleNamespace(
        rank=0, torch_device="cuda:0", local_rank=0,
        engine_config=SimpleNamespace(
            Module_Batching_Config=SimpleNamespace(
                prefill_prefix_pool_workspace_bytes=0,
            )
        ),
        _prefill_prefix_kv_budget_bytes=None,
        _prefill_pool_page_budget=None,
        _prefill_prefix_pool_enabled=lambda: False,
    )
    measure(worker)
    assert worker._prefill_prefix_kv_budget_bytes is None


# ------------------------------- chunked == unchunked materialization


class _RecordingManager:
    """A temporary GPU KV manager that enforces its own page capacity."""

    def __init__(self, config):
        self.config = config
        self.used_pages = 0
        self.rows = {}
        self.destroyed = False

    def allocate_pages_for_sequences(self, sequence_ids, lengths):
        need = sum(
            temporary_kv_pages(length, self.config.page_size_tokens)
            for length in lengths
        )
        if self.used_pages + need > self.config.num_pages:
            raise MemoryError("temporary manager over capacity")
        self.used_pages += need
        for sid, length in zip(sequence_ids, lengths):
            self.rows[int(sid)] = {"full": int(length)}

    def rebuild_page_table(self, sequence_ids):
        pass

    def prepare_prefill_suffix_append(
        self, *, sequence_ids, prefix_lens, suffix_lens, rebuild_page_table
    ):
        for sid, prefix, suffix in zip(sequence_ids, prefix_lens, suffix_lens):
            self.rows[int(sid)].update(prefix=int(prefix), suffix=int(suffix))
        return SimpleNamespace(sequence_ids=list(sequence_ids))

    def get_padded_3d_page_pointers(self):
        width = max(
            temporary_kv_pages(row["full"], self.config.page_size_tokens)
            for row in self.rows.values()
        )
        shape = (self.config.num_layers, len(self.rows), width)
        return torch.zeros(shape, dtype=torch.int64), torch.zeros(
            shape, dtype=torch.int64
        )

    def destroy(self, *, empty_cuda_cache=False):
        self.destroyed = True


class _Host:
    def __init__(self):
        self.loads = []

    def async_load_prefix_pages_to_device(self, *, host_page_ids, active_page_counts, **_):
        self.loads.append((host_page_ids.tolist(), active_page_counts.tolist()))
        return SimpleNamespace(done=lambda: True, wait=lambda: None)


class _Coordinator:
    def __init__(self):
        self.begun = []
        self.ended = []

    def begin_attachment_load(self, handle):
        self.begun.append(int(handle))

    def end_attachment_load(self, handle):
        self.ended.append(int(handle))


def _lookup(handle, pages):
    return SimpleNamespace(
        attachment_handle=handle,
        materialization_spans=[
            SimpleNamespace(
                group_id=0, pages=[SimpleNamespace(page_id=p) for p in pages]
            )
        ] if pages else [],
    )


def _materialize(micro_batches, sids, full, cached, lookups, raw_page_tokens):
    """Run each micro-batch's materialization, then close it as prefill does."""
    host, coordinator = _Host(), _Coordinator()
    state, peaks = {}, []
    for start, end in micro_batches:
        if not any(cached[start:end]):
            continue
        manager = _RecordingManager(
            temporary_gpu_kv_config(PROFILE, full[start:end])
        )
        materialization = materialize_gpt_oss_prefixes(
            gpu_manager=manager,
            host_worker_view=host,
            coordinator=coordinator,
            lookup_results=lookups[start:end],
            sequence_ids=sids[start:end],
            prompt_lengths=full[start:end],
            compute_cached_tokens=cached[start:end],
            raw_page_tokens=raw_page_tokens,
        )
        peaks.append(manager.config.num_pages)
        page_ids, counts = host.loads[-1] if materialization.load_task else ([], [])
        for offset, sid in enumerate(sids[start:end]):
            row = dict(manager.rows[sid])
            row["host_pages"] = (
                page_ids[offset][: counts[offset]] if counts else []
            )
            state[sid] = row
        materialization.close()
        assert manager.destroyed
    return state, sorted(coordinator.begun), sorted(coordinator.ended), peaks


def test_split_materialization_reaches_the_same_final_state():
    raw_page_tokens = 256  # host page == FA page: no pointer expansion
    sids = [11, 12, 13, 14, 15, 16]
    full = [1280, 1024, 1280, 900, 1280, 1300]
    cached = [1024, 0, 768, 512, 0, 1024]
    lookups = [
        _lookup(91, [1, 2, 3, 4]), _lookup(0, []), _lookup(93, [5, 6, 7]),
        _lookup(94, [8, 9]), _lookup(0, []), _lookup(96, [10, 11, 12, 13]),
    ]
    suffix = [f - c for f, c in zip(full, cached)]
    hits = [c > 0 for c in cached]
    pages = [temporary_kv_pages(f, 256) for f in full]

    whole, _ = build_prefill_micro_batches(suffix, 10**6, l2_balance=False)
    split, _ = build_prefill_micro_batches(
        suffix, 10**6, l2_balance=False,
        prefix_hits=hits, kv_pages=pages, kv_page_cap=11,
    )
    assert whole == [(0, 6)] and len(split) > 1

    one = _materialize(whole, sids, full, cached, lookups, raw_page_tokens)
    many = _materialize(split, sids, full, cached, lookups, raw_page_tokens)
    # Same per-sequence KV: full length, prefix, suffix and loaded Host pages.
    assert one[0] == many[0]
    assert one[0][11]["host_pages"] == [1, 2, 3, 4]
    assert one[0][12]["host_pages"] == []
    # Every attachment taken once and released once, either way.
    assert one[1] == many[1] == one[2] == many[2] == [91, 93, 94, 96]
    # The transient is what changed: one 29-page manager vs. ones that fit.
    assert one[3] == [sum(pages)] == [29]
    assert split == [(0, 2), (2, 4), (4, 6)] and max(many[3]) <= 11


# ------------------------------------------------ structural wiring


def _method(name):
    tree = ast.parse(WORKER_SOURCE.read_text())
    cls = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "BatchGenWorker"
    )
    return next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
    )


def test_prefill_sizes_the_temporary_manager_with_the_planned_function():
    body = ast.unparse(_method("prefill_prepacked"))
    assert "self._prefix_materialization_kv_config(batch_full_seq_lengths)" in body
    assert (
        "self._plan_prefill_micro_batches(seq_lengths_list, full_seq_lengths, "
        "compute_cached_tokens)" in body
    )
    # The old inline full-length sizing must not come back beside it.
    assert "fa_page_tokens" not in body


def test_configure_prefill_measures_the_budget_whenever_prefix_cache_is_on():
    body = ast.unparse(_method("_config_prefill_for_batch"))
    assert "if self.enable_prefix_cache:\n" in body
    assert "self._measure_prefill_prefix_kv_budget()" in body
