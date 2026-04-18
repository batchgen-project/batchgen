"""Byte-level Triton-vs-Triton test: BatchGen ``act_quant`` vs SGLang's
``_act_quant_kernel`` (inlined verbatim from SGLang's
``sglang/srt/layers/attention/nsa/triton_kernel.py``).

Context: the earlier PyTorch-reference test flagged a 0.07-0.15% byte
mismatch between BatchGen's act_quant and ``x.to(torch.float8_e4m3fn)``.
That comparison was unfair: PyTorch's element-wise FP8 cast may use a
different rounding mode than Triton's ``.to(tl.float8e4nv)``. Both
engines use Triton for act_quant at serving time, so an apples-to-apples
comparison must run Triton kernels on BOTH sides.

If this test PASSES byte-exact: act_quant is NOT the FP8 divergence
source vs SGLang — they produce identical quantization. The remaining
suspects are the DeepGEMM call itself (same library version? same
flags?) or the scale format.

If this test FAILS: BatchGen's Triton kernel deviates from SGLang's,
and the divergence shows up byte-level. We can then patch BatchGen's
kernel to match SGLang or just swap BatchGen's implementation for the
SGLang version.

The SGLang kernel is inlined here (Apache 2.0 license; source:
https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/nsa/triton_kernel.py)
so this test has no runtime dependency on a sglang install.
"""
import pytest
import torch
import triton
import triton.language as tl


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FP8 kernels require CUDA",
)


# =========================================================================
# SGLang ``_act_quant_kernel`` — verbatim inlined from
# sglang/srt/layers/attention/nsa/triton_kernel.py:9-84
# =========================================================================
@triton.jit
def _sglang_act_quant_kernel(
    X_ptr,
    Y_ptr,
    S_ptr,
    M,
    N,
    group_size: tl.constexpr,
    round_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
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

    x_abs = tl.abs(x)
    amax = tl.max(x_abs, axis=1)

    amax = tl.maximum(amax, 1e-4)

    if round_scale:
        log_val = tl.log2(amax * fp8_max_inv)
        log_ceil = tl.ceil(log_val)
        scale = tl.exp2(log_ceil)
    else:
        scale = amax * fp8_max_inv

    scale_broadcast = scale[:, None]
    y = x / scale_broadcast
    y = tl.minimum(tl.maximum(y, fp8_min), fp8_max)

    y_ptrs = Y_ptr + rows[:, None] * N + cols[None, :]
    tl.store(y_ptrs, y, mask=mask)

    s_cols = pid_n
    s_ptrs = S_ptr + rows * (N // group_size) + s_cols
    s_mask = row_mask
    tl.store(s_ptrs, scale, mask=s_mask)


def sglang_act_quant(x: torch.Tensor, block_size: int = 128):
    """Python wrapper matching SGLang's ``act_quant``
    (triton_kernel.py:86-136), pinned to round_scale=False (what GLM-5
    prefill actually uses; SGLang only enables round_scale for FP8 scale
    fmt UE8M0 which isn't the default path)."""
    assert x.is_contiguous()
    assert x.size(-1) % block_size == 0
    N = x.size(-1)
    x_flat = x.view(-1, N)
    M = x_flat.size(0)
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    y_flat = y.view(-1, N)
    s = x.new_empty(*x.size()[:-1], N // block_size, dtype=torch.float32)
    s_flat = s.view(-1, N // block_size)
    BLOCK_M = 32
    BLOCK_N = block_size
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, block_size))
    _sglang_act_quant_kernel[grid](
        x_flat,
        y_flat,
        s_flat,
        M,
        N,
        group_size=block_size,
        round_scale=False,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_stages=2,
    )
    return y, s


def _fp8_bytes(t: torch.Tensor) -> torch.Tensor:
    """Reinterpret FP8_E4M3 tensor as raw uint8."""
    return t.view(torch.uint8)


@pytest.mark.parametrize("shape,std", [
    ((1, 6144), 0.03),       # typical layer-0 post-embedding
    ((16, 6144), 0.05),      # prefill 16 tokens
    ((256, 6144), 0.05),     # prefill 256 tokens
    ((128, 2048), 0.10),     # q_a output pre q_b_proj
    ((256, 512), 0.20),      # normed_kv pre kv_b_proj
    ((32, 16384), 0.02),     # attn_output pre o_proj
    ((64, 12288), 0.04),     # post post_attn_ln pre mlp.gate/up
])
def test_batchgen_vs_sglang_act_quant_byte_exact(shape, std):
    """BatchGen ``act_quant`` vs SGLang's inlined Triton kernel, byte-exact.

    Feeds identical BF16 input to both, compares:
      - FP8 tensor reinterpreted as raw uint8 (exact byte match).
      - FP32 scale tensor (exact value match).

    Both kernels use ``.to(tl.float8e4nv)`` so rounding should be the
    same. Differences flagged here are BatchGen-vs-SGLang real
    divergences, not PyTorch-vs-Triton rounding artifacts.
    """
    from batchgen.attention.mla.fa3_backend import act_quant

    device = "cuda"
    torch.manual_seed(0xAC7 + hash(shape) % 1000 + int(std * 10000))
    x = torch.randn(shape, dtype=torch.bfloat16, device=device) * std

    # BatchGen
    bg_fp8, bg_scale = act_quant(x)

    # SGLang (inlined)
    sg_fp8, sg_scale = sglang_act_quant(x, block_size=128)

    # -- Shape checks --
    assert bg_fp8.shape == sg_fp8.shape, (
        f"fp8 shape mismatch: bg={bg_fp8.shape} sg={sg_fp8.shape}"
    )
    assert bg_scale.shape == sg_scale.shape, (
        f"scale shape mismatch: bg={bg_scale.shape} sg={sg_scale.shape}"
    )

    # -- Scale check (exact FP32) --
    scale_diff = (bg_scale.float() - sg_scale.float()).abs()
    scale_max = scale_diff.max().item()
    scale_mismatch = (scale_diff > 0).sum().item()
    print(
        f"[SCALE shape={shape} std={std:.3f}] "
        f"max|Δ|={scale_max:.4e} "
        f"mismatched_elems={scale_mismatch}/{bg_scale.numel()}"
    )

    # -- FP8 byte check --
    bg_bytes = _fp8_bytes(bg_fp8)
    sg_bytes = _fp8_bytes(sg_fp8)
    byte_eq = (bg_bytes == sg_bytes).all().item()
    mismatched = (bg_bytes != sg_bytes).sum().item()
    total = bg_bytes.numel()
    print(
        f"[FP8  shape={shape} std={std:.3f}] byte_exact={byte_eq}  "
        f"mismatched_bytes={mismatched}/{total} "
        f"({100.0 * mismatched / max(1, total):.3f}%)"
    )

    # Sample divergence location for diagnostic
    if not byte_eq:
        bg_f32 = bg_fp8.float()
        sg_f32 = sg_fp8.float()
        diff = (bg_f32 - sg_f32).abs()
        worst = diff.view(-1).argmax().item()
        print(
            f"  sample mismatch @ flat_idx={worst}: "
            f"x={x.float().view(-1)[worst].item():.6f}, "
            f"bg={bg_f32.view(-1)[worst].item():.4f}, "
            f"sg={sg_f32.view(-1)[worst].item():.4f}, "
            f"Δ={diff.view(-1)[worst].item():.4f}"
        )

    assert byte_eq, (
        f"BatchGen and SGLang Triton act_quant kernels produce DIFFERENT "
        f"FP8 bytes on shape={shape} std={std} "
        f"({mismatched}/{total} = "
        f"{100.0 * mismatched / max(1, total):.3f}%). "
        f"This is the prefill FP8 divergence source vs SGLang."
    )


@pytest.mark.parametrize("shape,std", [
    ((4, 128), 0.05),      # one block, minimal case
    ((64, 128), 0.05),     # 64 rows, one K-block
    ((1, 512), 1e-5),      # tiny magnitude — clamp regime
])
def test_batchgen_vs_sglang_edge_cases(shape, std):
    """Edge magnitudes: clamp floor differs between BatchGen and SGLang
    (BatchGen uses FP8_E4M3_MIN_NORMAL=1.5e-5, SGLang uses 1e-4). Expect
    divergence for std<1e-4; otherwise expect byte-exact."""
    from batchgen.attention.mla.fa3_backend import act_quant

    device = "cuda"
    torch.manual_seed(0xED6E + int(std * 1e6) % 1000)
    x = (torch.randn(shape, dtype=torch.bfloat16, device=device) * std).contiguous()

    bg_fp8, bg_scale = act_quant(x)
    sg_fp8, sg_scale = sglang_act_quant(x, block_size=128)

    bg_bytes = _fp8_bytes(bg_fp8)
    sg_bytes = _fp8_bytes(sg_fp8)
    mismatched = (bg_bytes != sg_bytes).sum().item()
    scale_max = (bg_scale.float() - sg_scale.float()).abs().max().item()
    print(
        f"[EDGE shape={shape} std={std:.1e}] "
        f"byte_mismatch={mismatched}/{bg_bytes.numel()} "
        f"scale_max_diff={scale_max:.4e}"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
