"""Host-KV release returns only a sequence's own chain, not prefix pages.

Pins the bookkeeping rule behind ``SequenceEntry.host_owned_pages``:
owned = allocated_total - attached - retained, against the real backend, and
shows why the growth planner must credit it instead of host_pages_allocated.
"""

import ctypes
import os
import uuid

import pytest

from batchgen.continuous_batching import plan_host_kv_growth_evictions
from batchgen.models.engine_loader import core_engine as bg
from batchgen.prefix_reuse.commit import (
    build_prefix_commit_request,
    collect_group_pages_for_commit,
    retain_inserted_prefix_pages,
)
from batchgen.prefix_reuse.config import (
    build_prefix_cache_runtime_config,
    create_host_prefix_cache_coordinator,
    unlink_prefix_cache_shared_memory,
)
from batchgen.prefix_reuse.prefill import (
    PrefixCacheSequenceState,
    lookup_prefix_cache_for_prefill,
)

PAGE_TOKENS = 4


def _shm_unlink(name: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.shm_unlink(name.encode()) != 0 and ctypes.get_errno() != 2:
        raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))


def _host_config(shm_name: str):
    config = bg.HostPagedKVConfig()
    config.shm_name = shm_name
    config.num_layers = 1
    config.num_pages = 16
    config.page_size_tokens = PAGE_TOKENS
    config.num_k_heads = 1
    config.k_head_dim = 8
    config.num_v_heads = 1
    config.v_head_dim = 8
    config.k_element_size_bytes = 2
    config.v_element_size_bytes = 2
    config.sequence_table_capacity = 16
    return config


def _growth_plan(*, free_pages, completed_pages):
    return plan_host_kv_growth_evictions(
        active_uuids=["grower"],
        completed_uuids=["seed", "hit"],
        host_growth_uuids=["grower"],
        host_growth_pages=[8],
        eviction_candidates=[],
        free_pages=free_pages,
        total_pages=16,
        completed_pages=completed_pages,
        watermark_percent=0,
        safety_margin=0,
    )


def test_release_returns_owned_pages_and_planner_must_credit_them():
    host_shm = f"/prefix_growth_credit_host_{uuid.uuid4().hex}"
    host_config = _host_config(host_shm)
    runtime = build_prefix_cache_runtime_config(
        model_name="openai/gpt-oss-120b",
        kv_dtype="bfloat16",
        host_kv_config=host_config,
    )
    unlink_prefix_cache_shared_memory(runtime)
    host = bg.DefaultHostPagedKVWorkerView(host_config)
    owner = None
    worker = None
    try:
        host.initialize(0, True)
        owner = create_host_prefix_cache_coordinator(
            core_engine_module=bg,
            runtime_config=runtime,
            create_region=True,
        )
        worker = create_host_prefix_cache_coordinator(
            core_engine_module=bg,
            runtime_config=runtime,
            create_region=False,
        )

        # Seed S: 4 pages reserved, 12 prompt tokens (3 pages) committed.
        seed = 101
        tokens = list(range(12))
        seed_allocated = 4
        host.register_sequences([seed])
        host.allocate_pages_for_sequences([(seed, seed_allocated * PAGE_TOKENS)])
        pages = collect_group_pages_for_commit(
            worker_views_by_group={0: host},
            sequence_id=seed,
            commit_tokens=len(tokens),
            raw_page_tokens_by_group={0: PAGE_TOKENS},
        )
        request = build_prefix_commit_request(
            namespace_digest=runtime.namespace_digest,
            token_ids=tokens,
            publish_boundary_tokens=runtime.publish_boundary_tokens,
            pages_by_group=pages,
        )
        assert request is not None
        result = request.commit(worker)
        retained = retain_inserted_prefix_pages(
            commit_result=result,
            request=request,
            worker_views_by_group={0: host},
            sequence_id=seed,
        )
        seed_retained = sum(len(value) for value in retained.values())
        assert seed_retained == 3
        seed_owned = seed_allocated - seed_retained

        # Hit T: attaches the 3 shared pages, 1 private page (4 reserved).
        hit = 202
        hit_allocated = 4
        lookup = lookup_prefix_cache_for_prefill(
            coordinator=worker,
            namespace_digest=runtime.namespace_digest,
            prompt_token_ids=[tokens],
            page_size_tokens=PAGE_TOKENS,
        )
        hit_state = PrefixCacheSequenceState(
            lookup_result=lookup.lookup_results[0],
            attached_tokens=int(lookup.attached_tokens[0]),
            compute_cached_tokens=int(lookup.compute_cached_tokens[0]),
        )
        hit_attached = hit_state.attached_tokens // PAGE_TOKENS
        assert hit_attached == 3
        host.register_sequences([hit])
        host.attach_shared_prefix_pages(hit, list(hit_state.shared_page_ids))
        host.allocate_pages_for_sequences(
            [(hit, (hit_allocated - hit_attached) * PAGE_TOKENS)]
        )
        hit_owned = hit_allocated - hit_attached

        # Grower G: 8 private pages.
        grower = 303
        host.register_sequences([grower])
        host.allocate_pages_for_sequences([(grower, 8 * PAGE_TOKENS)])

        free_before = host.get_stats().num_free_pages
        assert free_before == 3

        # Rank-0 plan at the boundary: completed S and T are credited before
        # they release. host_pages_allocated (4 + 4) over-credits by the
        # attached + retained pages; host_owned_pages (1 + 1) does not.
        old_plan = _growth_plan(
            free_pages=free_before,
            completed_pages=seed_allocated + hit_allocated,
        )
        owned_plan = _growth_plan(
            free_pages=free_before,
            completed_pages=seed_owned + hit_owned,
        )
        assert old_plan.growth_feasible_after_eviction is True
        assert owned_plan.growth_feasible_after_eviction is False

        host.release_sequence_pages([seed, hit])
        free_after = host.get_stats().num_free_pages
        assert free_after - free_before == seed_owned + hit_owned == 2
        assert free_after == owned_plan.expected_free_pages == 5

        with pytest.raises(RuntimeError, match="Insufficient free pages"):
            host.grow_pages_for_sequences([(grower, 8)])
    finally:
        worker = None
        owner = None
        try:
            host.shutdown()
        except Exception:
            pass
        del host
        unlink_prefix_cache_shared_memory(runtime)
        _shm_unlink(host_shm)
