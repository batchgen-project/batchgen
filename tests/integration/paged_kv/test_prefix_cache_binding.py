from unittest import SkipTest

from batchgen.models.engine_loader import core_engine as bg


def _make_prefix_cache(
    *,
    num_pages: int = 64,
    radix_nodes: int = 256,
    radix_edges: int = 512,
    prefix_entries: int = 64,
    prefix_page_refs: int = 256,
    enable_prefix_reuse: bool = True,
    prefix_min_reuse_pages: int = 1,
    prefix_min_store_pages: int = 1,
    prefix_page_budget: int = 32,
):
    if not hasattr(bg, "HostKVPrefixCacheHarness"):
        raise SkipTest("HostKVPrefixCacheHarness binding is not available")
    return bg.HostKVPrefixCacheHarness(
        num_pages,
        radix_nodes,
        radix_edges,
        prefix_entries,
        prefix_page_refs,
        enable_prefix_reuse,
        prefix_min_reuse_pages,
        prefix_min_store_pages,
        prefix_page_budget,
    )


def test_prefix_cache_harness_commit_and_lookup_hit() -> None:
    cache = _make_prefix_cache()
    tokens = list(range(128))

    pages, reused = cache.lookup(tokens, 2)
    assert pages == []
    assert reused == 0
    assert cache.get_stats().prefix_miss_count == 1

    assert cache.commit(tokens, [3, 4]) is True
    pages, reused = cache.lookup(tokens, 2)
    assert pages == [3, 4]
    assert reused == 2

    stats = cache.get_stats()
    assert stats.prefix_entry_count == 1
    assert stats.prefix_used_pages == 2
    assert stats.prefix_hit_count >= 1
    assert cache.page_refcount(3) == 1
    assert cache.page_refcount(4) == 1


def test_prefix_cache_harness_respects_min_store_and_reuse_pages() -> None:
    cache = _make_prefix_cache(prefix_min_reuse_pages=2, prefix_min_store_pages=2)

    one_page_tokens = list(range(64))
    assert cache.commit(one_page_tokens, [5]) is False
    assert cache.get_stats().prefix_entry_count == 0

    two_page_tokens = list(range(128))
    assert cache.commit(two_page_tokens, [5, 6]) is True

    pages, reused = cache.lookup(two_page_tokens, 1)
    assert pages == []
    assert reused == 0

    pages, reused = cache.lookup(two_page_tokens, 2)
    assert pages == [5, 6]
    assert reused == 2


def test_prefix_cache_harness_lru_evict_releases_page_refs() -> None:
    cache = _make_prefix_cache(num_pages=8, prefix_page_budget=2)
    tokens_a = [1000 + i for i in range(128)]
    tokens_b = [2000 + i for i in range(128)]

    assert cache.commit(tokens_a, [0, 1]) is True
    assert cache.page_refcount(0) == 1
    assert cache.page_refcount(1) == 1

    assert cache.commit(tokens_b, [2, 3]) is True
    stats = cache.get_stats()
    assert stats.prefix_evict_count >= 1
    assert stats.prefix_entry_count == 1
    assert stats.prefix_used_pages <= 2

    assert cache.page_refcount(0) == 0
    assert cache.page_refcount(1) == 0
    assert cache.page_refcount(2) == 1
    assert cache.page_refcount(3) == 1

    pages, reused = cache.lookup(tokens_a, 2)
    assert pages == []
    assert reused == 0

    pages, reused = cache.lookup(tokens_b, 2)
    assert pages == [2, 3]
    assert reused == 2


def test_prefix_cache_harness_supports_decode_extension_commit() -> None:
    cache = _make_prefix_cache(prefix_min_store_pages=1, prefix_page_budget=16)
    prompt_tokens = [3000 + i for i in range(64)]
    decode_tokens = [4000 + i for i in range(64)]

    assert cache.commit(prompt_tokens, [9]) is True
    assert cache.commit(prompt_tokens + decode_tokens, [9, 10]) is True

    pages, reused = cache.lookup(prompt_tokens, 1)
    assert pages == [9]
    assert reused == 1

    pages, reused = cache.lookup(prompt_tokens + decode_tokens, 2)
    assert pages == [9, 10]
    assert reused == 2


def test_prefix_cache_harness_duplicate_commit_avoids_spurious_evict() -> None:
    cache = _make_prefix_cache(num_pages=8, prefix_page_budget=2)
    tokens = [5000 + i for i in range(128)]

    assert cache.commit(tokens, [0, 1]) is True
    stats_before = cache.get_stats()
    assert stats_before.prefix_entry_count == 1
    assert stats_before.prefix_evict_count == 0
    assert cache.page_refcount(0) == 1
    assert cache.page_refcount(1) == 1

    # Re-committing the exact same prefix should only touch recency metadata.
    assert cache.commit(tokens, [0, 1]) is True
    stats_after = cache.get_stats()
    assert stats_after.prefix_entry_count == 1
    assert stats_after.prefix_evict_count == 0
    assert cache.page_refcount(0) == 1
    assert cache.page_refcount(1) == 1


def test_prefix_cache_harness_lookup_refreshes_lru_recency() -> None:
    cache = _make_prefix_cache(num_pages=12, prefix_page_budget=4)
    tokens_a = [6000 + i for i in range(128)]
    tokens_b = [7000 + i for i in range(128)]
    tokens_c = [8000 + i for i in range(128)]

    assert cache.commit(tokens_a, [0, 1]) is True
    assert cache.commit(tokens_b, [2, 3]) is True

    pages, reused = cache.lookup(tokens_a, 2)
    assert pages == [0, 1]
    assert reused == 2

    assert cache.commit(tokens_c, [4, 5]) is True
    stats = cache.get_stats()
    assert stats.prefix_evict_count >= 1
    assert stats.prefix_entry_count == 2
    assert stats.prefix_used_pages == 4

    pages, reused = cache.lookup(tokens_a, 2)
    assert pages == [0, 1]
    assert reused == 2

    pages, reused = cache.lookup(tokens_b, 2)
    assert pages == []
    assert reused == 0

    pages, reused = cache.lookup(tokens_c, 2)
    assert pages == [4, 5]
    assert reused == 2

    assert cache.page_refcount(0) == 1
    assert cache.page_refcount(1) == 1
    assert cache.page_refcount(2) == 0
    assert cache.page_refcount(3) == 0
    assert cache.page_refcount(4) == 1
    assert cache.page_refcount(5) == 1
