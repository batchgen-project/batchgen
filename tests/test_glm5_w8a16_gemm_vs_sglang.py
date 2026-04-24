"""Byte-exact BatchGen vs SGLang ``w8a16_gemm`` end-to-end comparison.

Feeds identical ``(weight_fp8, scale, activation_bf16)`` through:

  1. BatchGen's ``w8a16_gemm`` (fa3_backend.py:679) — which does
     ``act_quant(x)`` + ``deep_gemm.fp8_gemm_nt(lhs, rhs, out)`` internally.
  2. An inlined SGLang-equivalent: SGLang's ``_act_quant_kernel`` (from
     ``sglang/srt/layers/attention/nsa/triton_kernel.py``, copy-pasted)
     followed by ``deep_gemm.fp8_gemm_nt(lhs, rhs, out)`` — matching
     SGLang's pass-through wrapper at
     ``sglang/srt/layers/deep_gemm_wrapper/entrypoint.py:84-102``.

Compares BF16 output byte-for-byte via ``.view(torch.uint16)`` plus
FP32-cast max|Δ|/mean|Δ|/rel_max for diagnostic.

Baseline: commit ``5ff7f577`` (CUDA ``act_quant_3d`` byte-exact fix).
Covers 6 GLM-5 layer-0 projection shapes at ``M ∈ {1, 128, 256, 1024}``
to exercise both BatchGen act_quant branches (CUDA for M ≤ 128 and
Triton for M > 128).

If ALL shapes byte-exact match: DeepGEMM invocation is deterministic
and BatchGen's argument passing is equivalent to SGLang's — rules out
``w8a16_gemm`` as a prefill FP8 divergence source. If any diverge, the
test output pins which shape + by how many bytes so we can investigate
stride/alignment/scale-layout.
"""
import pytest
import torch
import triton
import triton.language as tl

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FP8 kernels require CUDA (SM90+)",
)


# =========================================================================
# SGLang ``_act_quant_kernel`` — verbatim from
# sglang/srt/layers/attention/nsa/triton_kernel.py:9-84 (Apache 2.0)
# =========================================================================
@triton.jit
def _sglang_act_quant_kernel(
    X_ptr, Y_ptr, S_ptr, M, N,
    group_size: tl.constexpr, round_scale: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max

    row_start = pid_m * BLOCK_M
    col_start = pid_n * group_size

    rows = row_start + tl.arange(0, BLOCK_M)
    cols = col_start + tl.arange(0, BLOCK_N)

    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]

    x_ptrs = X_ptr + rows[:, None] * N + cols[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
    if round_scale:
        scale = tl.exp2(tl.ceil(tl.log2(amax * fp8_max_inv)))
    else:
        scale = amax * fp8_max_inv

    y = tl.minimum(tl.maximum(x / scale[:, None], fp8_min), fp8_max)
    y_ptrs = Y_ptr + rows[:, None] * N + cols[None, :]
    tl.store(y_ptrs, y, mask=mask)

    s_ptrs = S_ptr + rows * (N // group_size) + pid_n
    tl.store(s_ptrs, scale, mask=row_mask)


def sglang_act_quant(x, block_size=128):
    assert x.is_contiguous(), "input must be contiguous"
    assert x.size(-1) % block_size == 0
    N = x.size(-1)
    x_flat = x.view(-1, N)
    M = x_flat.size(0)
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn).view(-1, N)
    s = x.new_empty(M, N // block_size, dtype=torch.float32)
    grid = (triton.cdiv(M, 32), triton.cdiv(N, block_size))
    _sglang_act_quant_kernel[grid](
        x_flat, y, s, M, N,
        group_size=block_size, round_scale=False,
        BLOCK_M=32, BLOCK_N=block_size, num_stages=2,
    )
    return y, s


def _build_weight(n, k, device, seed):
    """Synthetic per-128-block FP8 weight + FP32 scale."""
    assert n % 128 == 0 and k % 128 == 0, f"n={n} k={k} must be multiples of 128"
    g = torch.Generator(device=device).manual_seed(seed)
    w_bf16 = torch.randn((n, k), dtype=torch.bfloat16, device=device, generator=g) * 0.02
    w_fp32 = w_bf16.float()
    n_b, k_b = n // 128, k // 128
    w_fp8 = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=device)
    scale = torch.empty((n_b, k_b), dtype=torch.float32, device=device)
    FP8_MAX = 448.0
    for i in range(n_b):
        for j in range(k_b):
            tile = w_fp32[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128]
            s = tile.abs().max().clamp(min=1e-12) / FP8_MAX
            scale[i, j] = s
            w_fp8[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128] = (
                (tile / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            )
    return w_fp8, scale


# GLM-5 layer-0 projection shapes. kv_a_proj_with_mqa (N=576, not x128) is
# skipped — DeepGEMM rejects non-128-multiple N on SM90.
SHAPES = [
    ("q_a_proj",         2048,   6144),
    ("q_b_proj",        16384,   2048),
    ("kv_b_proj",       28672,    512),
    ("o_proj",           6144,  16384),
    ("mlp_gate_up",     12288,   6144),
    ("mlp_down",         6144,  12288),
]


@pytest.mark.parametrize("name,n,k", SHAPES)
@pytest.mark.parametrize("m", [1, 128, 256, 1024])
def test_w8a16_gemm_byte_exact_vs_sglang(name, n, k, m):
    import deep_gemm
    from batchgen.attention.mla.fa3_backend import w8a16_gemm

    device = "cuda"
    torch.manual_seed(0xB65 + hash(name) % 1000 + m)
    w_fp8, scale = _build_weight(n, k, device, seed=0x77 + hash(name) % 500)
    x = torch.randn((m, k), dtype=torch.bfloat16, device=device) * 0.05

    # BatchGen path: w8a16_gemm = act_quant(x) + deep_gemm.fp8_gemm_nt
    bg_out = w8a16_gemm(w_fp8, scale, x)

    # SGLang path: inlined act_quant + deep_gemm.fp8_gemm_nt (same wrapper)
    sg_lhs = sglang_act_quant(x.contiguous(), block_size=128)
    sg_out = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    deep_gemm.fp8_gemm_nt(sg_lhs, (w_fp8, scale), sg_out)

    assert bg_out.shape == sg_out.shape == (m, n)

    # Byte-exact BF16 comparison (reinterpret as uint16)
    bg_u16 = bg_out.view(torch.uint16)
    sg_u16 = sg_out.view(torch.uint16)
    mismatched = (bg_u16 != sg_u16).sum().item()
    total = bg_u16.numel()

    diff = (bg_out.float() - sg_out.float()).abs()
    ref = sg_out.float().abs().clamp(min=1e-6)
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    rel_max = (diff / ref).max().item()

    print(
        f"[{name:12s} m={m:5d} n={n:5d} k={k:5d}] "
        f"byte_mismatch={mismatched:>6}/{total:<9} "
        f"({100.0 * mismatched / max(1, total):.4f}%) "
        f"max|Δ|={max_abs:.4e} mean|Δ|={mean_abs:.4e} "
        f"rel_max={rel_max:.4e}"
    )

    assert mismatched == 0, (
        f"[{name}] BatchGen and SGLang w8a16_gemm diverge byte-level: "
        f"{mismatched}/{total} BF16 elements differ "
        f"(max|Δ|={max_abs:.4e}, rel_max={rel_max:.4e}). "
        f"This is the prefill FP8 divergence source at the DeepGEMM boundary."
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
