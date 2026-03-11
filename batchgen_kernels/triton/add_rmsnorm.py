"""
Fused Residual Add + RMSNorm kernel.

Single kernel launch replacing ~5 PyTorch ops:
  residual = residual + hidden
  hidden_normed = rmsnorm(residual)

This pattern appears twice per transformer layer:
  1. Post-attention: residual += attn_output, then RMSNorm for MoE input
  2. Post-MoE: residual += moe_output, then RMSNorm for next layer's input

Usage:
    from fused_add_rmsnorm import fused_add_rmsnorm

    # In-place residual update + normalized output
    normed, residual = fused_add_rmsnorm(residual, hidden, weight, eps=1e-5)

Design:
    - Single pass: load residual + hidden → add → accumulate variance →
      normalize → store both residual (updated) and normed output
    - FP32 accumulation for variance (matches PyTorch reference)
    - Outputs: (normed, residual_updated)
    - One program per row — optimal for decode [B, hidden_size]
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _add_rmsnorm_kernel(
    Residual_ptr,    # [num_rows, hidden_size] residual (read + write)
    Hidden_ptr,      # [num_rows, hidden_size] hidden states to add
    W_ptr,           # [hidden_size] norm weight
    Normed_ptr,      # [num_rows, hidden_size] normalized output
    stride_r,
    stride_h,
    stride_o,
    hidden_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused residual add + RMSNorm.

    Computes:
        residual = residual + hidden
        normed = rmsnorm(residual, weight, eps)

    Single pass for hidden_size <= BLOCK_SIZE:
        1. Load residual and hidden
        2. Add in FP32
        3. Accumulate sum of squares
        4. Compute inv_rms
        5. Normalize with weight
        6. Store updated residual and normed output
    """
    row_idx = tl.program_id(0)
    row_r = row_idx * stride_r
    row_h = row_idx * stride_h
    row_o = row_idx * stride_o

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size

    # Load residual and hidden, add in FP32
    r = tl.load(Residual_ptr + row_r + cols, mask=mask, other=0.0)
    h = tl.load(Hidden_ptr + row_h + cols, mask=mask, other=0.0)

    # Add in FP32 for precision
    sum_fp32 = r.to(tl.float32) + h.to(tl.float32)

    # Compute variance
    sum_sq = sum_fp32 * sum_fp32
    variance = tl.sum(sum_sq, axis=0) / hidden_size
    inv_rms = 1.0 / tl.sqrt(variance + eps)

    # Load weight and normalize
    w = tl.load(W_ptr + cols, mask=mask, other=1.0)
    w_fp32 = w.to(tl.float32)
    normed = sum_fp32 * inv_rms * w_fp32

    # Store updated residual (in original dtype) and normed output
    tl.store(Residual_ptr + row_r + cols, sum_fp32.to(r.dtype), mask=mask)
    tl.store(Normed_ptr + row_o + cols, normed.to(r.dtype), mask=mask)


@triton.jit
def _add_rmsnorm_kernel_large(
    Residual_ptr,
    Hidden_ptr,
    W_ptr,
    Normed_ptr,
    stride_r,
    stride_h,
    stride_o,
    hidden_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Two-pass variant for hidden_size > max BLOCK_SIZE."""
    row_idx = tl.program_id(0)
    row_r = row_idx * stride_r
    row_h = row_idx * stride_h
    row_o = row_idx * stride_o

    # Pass 1: add and accumulate sum of squares
    sum_sq_acc = 0.0
    for block_start in range(0, hidden_size, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        r = tl.load(Residual_ptr + row_r + cols, mask=mask, other=0.0)
        h = tl.load(Hidden_ptr + row_h + cols, mask=mask, other=0.0)
        sum_fp32 = r.to(tl.float32) + h.to(tl.float32)
        # Store updated residual immediately
        tl.store(Residual_ptr + row_r + cols, sum_fp32.to(r.dtype), mask=mask)
        sum_sq_acc += tl.sum(sum_fp32 * sum_fp32, axis=0)

    variance = sum_sq_acc / hidden_size
    inv_rms = 1.0 / tl.sqrt(variance + eps)

    # Pass 2: reload residual, normalize, store
    for block_start in range(0, hidden_size, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        r = tl.load(Residual_ptr + row_r + cols, mask=mask, other=0.0)
        w = tl.load(W_ptr + cols, mask=mask, other=1.0)
        r_fp32 = r.to(tl.float32)
        w_fp32 = w.to(tl.float32)
        normed = r_fp32 * inv_rms * w_fp32
        tl.store(Normed_ptr + row_o + cols, normed.to(r.dtype), mask=mask)


_MAX_SINGLE_PASS_HIDDEN = 8192


def fused_add_rmsnorm(
    residual: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    normed_out: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused residual add + RMSNorm.

    Args:
        residual: Residual tensor [..., hidden_size]. Modified IN-PLACE.
        hidden: Hidden states to add [..., hidden_size].
        weight: Norm weight [hidden_size].
        eps: Epsilon for numerical stability.
        normed_out: Optional pre-allocated output for normalized result.

    Returns:
        (normed, residual) — normed is the normalized output,
        residual is updated in-place (residual = residual + hidden).
    """
    assert residual.is_cuda and hidden.is_cuda and weight.is_cuda
    orig_shape = residual.shape
    hidden_size = orig_shape[-1]

    r_2d = residual.reshape(-1, hidden_size)
    h_2d = hidden.reshape(-1, hidden_size)
    num_rows = r_2d.shape[0]

    if normed_out is None:
        normed_out = torch.empty_like(residual)
    o_2d = normed_out.reshape(-1, hidden_size)

    assert weight.shape == (hidden_size,)

    if hidden_size <= _MAX_SINGLE_PASS_HIDDEN:
        BLOCK_SIZE = triton.next_power_of_2(hidden_size)
        _add_rmsnorm_kernel[(num_rows,)](
            r_2d, h_2d, weight, o_2d,
            r_2d.stride(0), h_2d.stride(0), o_2d.stride(0),
            hidden_size,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        BLOCK_SIZE = 4096
        _add_rmsnorm_kernel_large[(num_rows,)](
            r_2d, h_2d, weight, o_2d,
            r_2d.stride(0), h_2d.stride(0), o_2d.stride(0),
            hidden_size,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return normed_out.reshape(orig_shape), residual


# ============================================================================
# Validation
# ============================================================================

def _pytorch_add_rmsnorm(residual, hidden, weight, eps=1e-5):
    """PyTorch reference: add then RMSNorm."""
    residual = residual + hidden
    x = residual.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    normed = (weight * x).to(residual.dtype)
    return normed, residual


def validate(
    hidden_size: int = 4096,
    batch_sizes=(1, 2, 3, 4, 7, 8, 11, 16, 32, 61, 64, 128, 256),
    dtype=torch.bfloat16,
    eps: float = 1e-5,
    atol: float = 1e-5,
    rtol: float = 1.6e-2,
):
    """Validate fused add+RMSNorm against PyTorch reference."""
    device = torch.device("cuda")
    weight = torch.randn(hidden_size, device=device, dtype=dtype)
    all_passed = True

    for bs in batch_sizes:
        residual = torch.randn(bs, hidden_size, device=device, dtype=dtype)
        hidden = torch.randn(bs, hidden_size, device=device, dtype=dtype)

        # Reference (clone residual since fused modifies in-place)
        ref_normed, ref_residual = _pytorch_add_rmsnorm(
            residual.clone(), hidden, weight, eps
        )

        # Fused
        fused_normed, fused_residual = fused_add_rmsnorm(
            residual.clone(), hidden, weight, eps
        )

        # Check normed output
        diff_n = (fused_normed.float() - ref_normed.float()).abs()
        tol_n = atol + rtol * ref_normed.float().abs()
        n_fail_n = (diff_n > tol_n).sum().item()

        # Check residual
        diff_r = (fused_residual.float() - ref_residual.float()).abs()
        max_diff_r = diff_r.max().item()

        n_total = diff_n.numel()
        max_diff_n = diff_n.max().item()
        passed = ((n_fail_n == 0) or (n_fail_n / n_total < 1e-4)) and max_diff_r < 1e-3

        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(
            f"  [{status}] batch={bs:>3d}, hidden={hidden_size}, "
            f"normed_max_diff={max_diff_n:.2e}, residual_max_diff={max_diff_r:.2e}, "
            f"fail={n_fail_n}/{n_total}"
        )

    return all_passed


def benchmark(
    hidden_size: int = 4096,
    batch_sizes=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
    dtype=torch.bfloat16,
    warmup: int = 50,
    iters: int = 200,
):
    """
    Bandwidth benchmark for fused add+RMSNorm.

    Memory traffic per row:
        Read:  residual [H*e] + hidden [H*e] + weight [H*e]
        Write: normed [H*e] + residual_updated [H*e]
        Total: (3*bs + 1)*H*e read + 2*bs*H*e write = (5*bs + 1)*H*e
        (weight read once, amortized across rows)
    """
    device = torch.device("cuda")
    elem_size = torch.tensor([], dtype=dtype).element_size()
    weight = torch.randn(hidden_size, device=device, dtype=dtype)

    print(f"{'Batch':>6s}  {'Time(us)':>10s}  {'BW(TB/s)':>10s}  {'BW%_of_4TB':>10s}")
    print("-" * 45)

    for bs in batch_sizes:
        residual = torch.randn(bs, hidden_size, device=device, dtype=dtype)
        hidden = torch.randn(bs, hidden_size, device=device, dtype=dtype)
        normed_out = torch.empty_like(residual)

        # Warmup
        for _ in range(warmup):
            r_copy = residual.clone()
            fused_add_rmsnorm(r_copy, hidden, weight, normed_out=normed_out)

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iters):
            # Reset residual each iter (in-place modified)
            residual.copy_(hidden)  # cheap copy to reset
            fused_add_rmsnorm(residual, hidden, weight, normed_out=normed_out)
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end) / iters
        elapsed_s = elapsed_ms / 1000.0

        # Bytes: read(residual + hidden + weight) + write(residual + normed)
        bytes_moved = (2 * bs * hidden_size + hidden_size + 2 * bs * hidden_size) * elem_size
        bw_tb = bytes_moved / elapsed_s / 1e12

        print(f"{bs:>6d}  {elapsed_ms*1000:>10.1f}  {bw_tb:>10.3f}  {bw_tb/4.0*100:>9.1f}%")


def benchmark_vs_pytorch(
    hidden_size: int = 4096,
    batch_sizes=(1, 2, 4, 8, 16, 32, 64, 128, 256),
    dtype=torch.bfloat16,
    warmup: int = 50,
    iters: int = 200,
):
    """Compare fused vs PyTorch add+RMSNorm latency."""
    device = torch.device("cuda")
    weight = torch.randn(hidden_size, device=device, dtype=dtype)

    print(f"{'Batch':>6s}  {'Fused(us)':>10s}  {'PyTorch(us)':>12s}  {'Speedup':>8s}")
    print("-" * 45)

    for bs in batch_sizes:
        residual = torch.randn(bs, hidden_size, device=device, dtype=dtype)
        hidden = torch.randn(bs, hidden_size, device=device, dtype=dtype)

        # Warmup
        for _ in range(warmup):
            fused_add_rmsnorm(residual.clone(), hidden, weight)
            _pytorch_add_rmsnorm(residual.clone(), hidden, weight)

        torch.cuda.synchronize()

        # Fused
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fused_add_rmsnorm(residual.clone(), hidden, weight)
        end.record()
        torch.cuda.synchronize()
        fused_us = start.elapsed_time(end) / iters * 1000

        # PyTorch
        start.record()
        for _ in range(iters):
            _pytorch_add_rmsnorm(residual.clone(), hidden, weight)
        end.record()
        torch.cuda.synchronize()
        pytorch_us = start.elapsed_time(end) / iters * 1000

        speedup = pytorch_us / fused_us
        print(f"{bs:>6d}  {fused_us:>10.1f}  {pytorch_us:>12.1f}  {speedup:>7.2f}x")


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Fused Add+RMSNorm — Validation + Benchmark")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    # === Correctness ===
    print("--- Correctness Validation ---")
    for hidden in [4096]:
        print(f"Hidden size: {hidden}")
        for dt in [torch.bfloat16, torch.float16]:
            print(f"  dtype: {dt}")
            passed = validate(hidden_size=hidden, dtype=dt)
            if not passed:
                print("  ** VALIDATION FAILED **")
                sys.exit(1)
        print()

    # === Bandwidth Benchmark ===
    print("--- Bandwidth Benchmark (BF16, hidden=4096) ---")
    print("Target: 80% of H20 HBM3E (~4 TB/s) = 3.2 TB/s")
    print()
    benchmark(hidden_size=4096, dtype=torch.bfloat16)
    print()

    # === Latency Comparison ===
    print("--- Latency: Fused vs PyTorch (decode batch sizes) ---")
    benchmark_vs_pytorch(hidden_size=4096, dtype=torch.bfloat16)
    print()

    print("All validations passed.")
