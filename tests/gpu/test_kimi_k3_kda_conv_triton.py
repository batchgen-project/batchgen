# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3                                                          #
#  copyright (c) EfficientMoE team 2025                                        #
#  Licensed under the Apache License, Version 2.0                              #
# ---------------------------------------------------------------------------- #
"""Bit-exactness of the token-major KDA causal conv against the CUDA kernel."""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import pytest
import torch


if os.environ.get("K3_GPU_STAGE") == "1" and not torch.cuda.is_available():
    raise RuntimeError("K3_GPU_STAGE=1 requires a visible CUDA device")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="staged Kimi-K3 CUDA test"
)

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT / "batchgen" / "models" / "moonshotai" / "kimi_linear"
ALIAS = "_k3_kda_conv_gpu"
package = types.ModuleType(ALIAS)
package.__path__ = [str(PACKAGE_DIR)]
sys.modules.setdefault(ALIAS, package)
kda_conv_triton = importlib.import_module(f"{ALIAS}.kda_conv_triton")


def _case(cu, dim, seed, init_mask, num_slots=16, width=4):
    from batchgen_kernels.conv1d import causal_conv1d_fwd

    gen = torch.Generator(device="cuda").manual_seed(seed)
    total = cu[-1]
    x = torch.randn((total, dim), generator=gen, device="cuda", dtype=torch.bfloat16)
    w = torch.randn((dim, width), generator=gen, device="cuda", dtype=torch.bfloat16) * 0.3
    b = torch.randn((dim,), generator=gen, device="cuda", dtype=torch.bfloat16) * 0.1
    pool = torch.randn((num_slots, dim, width - 1), generator=gen, device="cuda", dtype=torch.bfloat16)
    nseq = len(cu) - 1
    slots = torch.randperm(num_slots, generator=gen, device="cuda")[:nseq].to(torch.int32)
    cu_t = torch.tensor(cu, dtype=torch.int32, device="cuda")
    init = torch.tensor(init_mask, dtype=torch.bool, device="cuda") if init_mask is not None else None

    x_ref, pool_ref = x.clone(), pool.clone()
    ref = causal_conv1d_fwd(
        x_ref, w, bias=b, conv_states=pool_ref, query_start_loc=cu_t,
        cache_indices=slots, has_initial_state=init, overwrite_x=True,
    )
    x_new, pool_new = x.clone(), pool.clone()
    # Production hands the depthwise Conv1d parameter, i.e. (dim, 1, W).
    got = kda_conv_triton.kda_causal_conv1d_triton(
        x_new, w.view(dim, 1, width), b, pool_new, cu_t, slots, init,
    )
    torch.cuda.synchronize()
    return ref, got, pool_ref, pool_new


@pytest.mark.parametrize(
    "cu,init_mask",
    [
        ([0, 137, 4233, 4300, 9000], None),
        ([0, 137, 4233, 4300, 9000], [True, False, True, True]),
        ([0, 64, 128, 8192], [False, True, False]),
        ([0, 3, 70, 71, 200], [True, True, False, True]),
        ([0, 1, 4, 200], [True, True, True]),
    ],
)
@pytest.mark.parametrize("dim", [1536, 64])
def test_token_major_conv_matches_cuda_kernel_bitwise(cu, init_mask, dim):
    ref, got, pool_ref, pool_new = _case(cu, dim, hash((tuple(cu), dim)) & 0xFFFF, init_mask)
    assert torch.equal(got, ref), (
        f"output mismatch max_abs={(got.float() - ref.float()).abs().max().item():.3e}"
    )
    assert torch.equal(pool_new, pool_ref), "final conv states differ"


def test_padded_slot_leaves_rows_untouched():
    cu = [0, 100, 300, 400]
    dim = 128
    from batchgen_kernels.conv1d import causal_conv1d_fwd

    gen = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn((400, dim), generator=gen, device="cuda", dtype=torch.bfloat16)
    w = torch.randn((dim, 4), generator=gen, device="cuda", dtype=torch.bfloat16) * 0.3
    b = torch.zeros((dim,), device="cuda", dtype=torch.bfloat16)
    pool = torch.zeros((8, dim, 3), device="cuda", dtype=torch.bfloat16)
    slots = torch.tensor([0, -1, 2], dtype=torch.int32, device="cuda")
    cu_t = torch.tensor(cu, dtype=torch.int32, device="cuda")
    x_ref, pool_ref = x.clone(), pool.clone()
    ref = causal_conv1d_fwd(
        x_ref, w, bias=b, conv_states=pool_ref, query_start_loc=cu_t,
        cache_indices=slots, has_initial_state=None, overwrite_x=True, pad_slot_id=-1,
    )
    x_new, pool_new = x.clone(), pool.clone()
    got = kda_conv_triton.kda_causal_conv1d_triton(x_new, w, b, pool_new, cu_t, slots, None, pad_slot_id=-1)
    torch.cuda.synchronize()
    assert torch.equal(got[100:300], x[100:300])
    assert torch.equal(got, ref)
    assert torch.equal(pool_new, pool_ref)
