"""A paged-KV token update must not requantize its source on the way in.

``tl.store`` converts to the destination pointer's dtype, so a bf16 token
written into an FP8 (or byte-carrier) cache used to be silently requantized
per element: the page then matched neither the producer's values nor the
layout its reader expects — and for a page-split FP8 cache it would also
overwrite the scale region's bytes with K values. The update is a byte move by
contract, so the dtypes must be equal and a mismatch is named.

CPU test: the guard is a precondition, raised before any kernel launch.
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


def test_fused_update_rejects_a_v_cache_dtype_mismatch():
    cache = torch.zeros(4, 64, 8, 64, dtype=torch.bfloat16)
    v_cache = torch.zeros(4, 64, 8, 64, dtype=torch.float16)
    tokens = torch.zeros(2, 512, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="source dtype to equal the cache dtype"):
        run_paged_kv_token_update_fused(
            k_cache=cache,
            k_tokens=tokens,
            page_table=_page_table(),
            slot_indices=torch.zeros(2, dtype=torch.int32),
            token_indices=torch.zeros(2, dtype=torch.int32),
            page_size_tokens=64,
            v_cache=v_cache,
            v_tokens=tokens,
        )


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
