"""A paged-KV token update must not requantize its source on the way in.

``tl.store`` converts to the destination pointer's dtype, so a bf16 token
written into an FP8 (or byte-carrier) cache used to be silently requantized
per element: the page then matched neither the producer's values nor the
layout its reader expects — and for a page-split FP8 cache it would also
overwrite the scale region's bytes with K values. The update is a byte move by
contract, so the dtypes must be equal and a mismatch is named.

The guard is a precondition, raised before ANY kernel launch -- including
for a K+V update whose V side is the mismatched one: validating V only after
the K launch would leave a half-written cache (K updated, V not). The K+V
rejection tests therefore assert the K cache is byte-for-byte untouched, and
run on CUDA too, where a regressed K-first launch would really write K (on
CPU any launch fails on its own, so CPU alone cannot catch the ordering).
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")  # the kv-cache update module is Triton-backed

from batchgen_kernels.triton.kv_cache import (  # noqa: E402
    run_paged_kv_token_update,
    run_paged_kv_token_update_fused,
)


def _page_table() -> torch.Tensor:
    return torch.zeros(2, 4, dtype=torch.int32)


def test_fused_update_rejects_a_dtype_mismatch():
    cache = torch.zeros(4, 64, 1, 132, dtype=torch.float8_e4m3fn)
    tokens = torch.zeros(2, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="source dtype to equal the cache dtype"):
        run_paged_kv_token_update_fused(
            k_cache=cache,
            k_tokens=tokens,
            page_table=_page_table(),
            slot_indices=torch.zeros(2, dtype=torch.int32),
            token_indices=torch.zeros(2, dtype=torch.int32),
            page_size_tokens=64,
        )


_DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA required"
        ),
    ),
]


def _kv_operands(device: str, v_dtype: torch.dtype):
    # A non-zero sentinel K cache and non-zero tokens: any K write is visible.
    k_cache = torch.full((4, 64, 8, 64), 7.0, dtype=torch.bfloat16, device=device)
    v_cache = torch.zeros(4, 64, 8, 64, dtype=v_dtype, device=device)
    tokens = torch.full((2, 512), 3.0, dtype=torch.bfloat16, device=device)
    return k_cache, v_cache, tokens


@pytest.mark.parametrize("device", _DEVICES)
def test_fused_update_rejects_a_v_cache_dtype_mismatch(device):
    k_cache, v_cache, tokens = _kv_operands(device, torch.float16)
    k_before = k_cache.clone()
    with pytest.raises(ValueError, match="source dtype to equal the cache dtype"):
        run_paged_kv_token_update_fused(
            k_cache=k_cache,
            k_tokens=tokens,
            page_table=_page_table().to(device),
            slot_indices=torch.zeros(2, dtype=torch.int32, device=device),
            token_indices=torch.zeros(2, dtype=torch.int32, device=device),
            page_size_tokens=64,
            v_cache=v_cache,
            v_tokens=tokens,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    # Rejected before the FIRST launch: the K cache must be untouched.
    assert torch.equal(k_cache, k_before)


@pytest.mark.parametrize("device", _DEVICES)
def test_unfused_update_rejects_a_v_cache_dtype_mismatch(device):
    k_cache, v_cache, tokens = _kv_operands(device, torch.float16)
    k_before = k_cache.clone()
    with pytest.raises(ValueError, match="source dtype to equal the cache dtype"):
        run_paged_kv_token_update(
            k_cache=k_cache,
            k_tokens=tokens,
            page_indices=torch.zeros(2, dtype=torch.int32, device=device),
            token_offsets=torch.zeros(2, dtype=torch.int32, device=device),
            v_cache=v_cache,
            v_tokens=tokens,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    assert torch.equal(k_cache, k_before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_update_with_matching_dtypes_writes_both_caches():
    # Positive control for the rejection tests above: with matching dtypes
    # the same operands DO write K and V, so "K untouched" there is a real
    # observation, not an artifact of a launch that could never write.
    k_cache, v_cache, tokens = _kv_operands("cuda", torch.bfloat16)
    k_before = k_cache.clone()
    run_paged_kv_token_update_fused(
        k_cache=k_cache,
        k_tokens=tokens,
        page_table=_page_table().cuda(),
        slot_indices=torch.zeros(2, dtype=torch.int32, device="cuda"),
        token_indices=torch.zeros(2, dtype=torch.int32, device="cuda"),
        page_size_tokens=64,
        v_cache=v_cache,
        v_tokens=tokens,
    )
    torch.cuda.synchronize()
    assert not torch.equal(k_cache, k_before)
    assert torch.equal(k_cache[0, 0].reshape(-1), tokens[0])
    assert torch.equal(v_cache[0, 0].reshape(-1), tokens[0])


def test_unfused_update_rejects_a_dtype_mismatch():
    cache = torch.zeros(4, 64, 1, 132, dtype=torch.float8_e4m3fn)
    tokens = torch.zeros(2, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="source dtype to equal the cache dtype"):
        run_paged_kv_token_update(
            k_cache=cache,
            k_tokens=tokens,
            page_indices=torch.zeros(2, dtype=torch.int32),
            token_offsets=torch.zeros(2, dtype=torch.int32),
        )
