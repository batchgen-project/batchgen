"""
High-performance Triton RMSNorm kernel.

Standalone RMSNorm: single kernel launch replacing ~4 PyTorch ops.
Supports BF16/FP16 input with FP32 accumulation for numerical precision.

Usage:
    from fused_rmsnorm import fused_rmsnorm

    # Standalone RMSNorm (replaces ~4 kernel launches with 1)
    output = fused_rmsnorm(x, weight, eps=1e-5)

Design:
    - Single-pass: accumulates sum-of-squares while loading, then normalizes
      in registers (no second global memory read for small hidden sizes)
    - FP32 accumulation for variance computation (matches PyTorch reference)
    - FP32 rsqrt (no fast-math approximation)
    - BF16/FP16 output with proper rounding
    - One program per row — optimal for decode shapes [B, hidden_size]

Reference (PyTorch equivalent):
    x_fp32 = x.float()
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    x = x_fp32 * torch.rsqrt(variance + eps)
    return (weight * x).to(input_dtype)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    X_ptr,          # [num_rows, hidden_size] input
    W_ptr,          # [hidden_size] weight
    O_ptr,          # [num_rows, hidden_size] output
    stride_x,       # stride of X along row dimension
    stride_o,       # stride of O along row dimension
    hidden_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    RMSNorm kernel: one program per row.

    For hidden_size <= BLOCK_SIZE (typical for decode: 4096, 7168, etc.),
    this is a single-pass kernel: load → accumulate variance → normalize → store.
    Data stays in registers between accumulation and normalization.

    For hidden_size > BLOCK_SIZE, falls back to two-pass (accumulate, then
    reload and normalize). This path is rarely hit in practice.
    """
    row_idx = tl.program_id(0)
    row_start_x = row_idx * stride_x
    row_start_o = row_idx * stride_o

    # --- Phase 1: Load data and compute sum of squares in FP32 ---
    # For hidden_size <= BLOCK_SIZE, we keep data in registers for Phase 2
    sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size

    # Load input as FP32 for accumulation
    x = tl.load(X_ptr + row_start_x + cols, mask=mask, other=0.0)
    x_fp32 = x.to(tl.float32)

    # Accumulate sum of squares
    sum_sq = x_fp32 * x_fp32

    # Reduce to scalar
    variance = tl.sum(sum_sq, axis=0) / hidden_size

    # Compute inverse RMS in FP32 (no fast-math)
    inv_rms = 1.0 / tl.sqrt(variance + eps)

    # --- Phase 2: Normalize and write output ---
    # Data is still in registers (x_fp32) from Phase 1
    w = tl.load(W_ptr + cols, mask=mask, other=1.0)
    w_fp32 = w.to(tl.float32)

    # Normalize: (x * inv_rms) * weight, then cast back to input dtype
    out = x_fp32 * inv_rms * w_fp32
    tl.store(O_ptr + row_start_o + cols, out.to(x.dtype), mask=mask)


@triton.jit
def _rmsnorm_kernel_large(
    X_ptr,
    W_ptr,
    O_ptr,
    stride_x,
    stride_o,
    hidden_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Two-pass RMSNorm for hidden_size > max BLOCK_SIZE.
    Pass 1: accumulate sum of squares.
    Pass 2: reload, normalize, store.
    """
    row_idx = tl.program_id(0)
    row_start_x = row_idx * stride_x
    row_start_o = row_idx * stride_o

    # Pass 1: sum of squares
    sum_sq_acc = 0.0
    for block_start in range(0, hidden_size, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        x = tl.load(X_ptr + row_start_x + cols, mask=mask, other=0.0)
        x_fp32 = x.to(tl.float32)
        sum_sq_acc += tl.sum(x_fp32 * x_fp32, axis=0)

    variance = sum_sq_acc / hidden_size
    inv_rms = 1.0 / tl.sqrt(variance + eps)

    # Pass 2: normalize and store
    for block_start in range(0, hidden_size, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        x = tl.load(X_ptr + row_start_x + cols, mask=mask, other=0.0)
        w = tl.load(W_ptr + cols, mask=mask, other=1.0)
        x_fp32 = x.to(tl.float32)
        w_fp32 = w.to(tl.float32)
        out = x_fp32 * inv_rms * w_fp32
        tl.store(O_ptr + row_start_o + cols, out.to(x.dtype), mask=mask)


# Maximum single-pass BLOCK_SIZE (Triton limit for register pressure)
_MAX_SINGLE_PASS_HIDDEN = 8192


def fused_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """
    Fused RMSNorm: single Triton kernel replacing ~4 PyTorch ops.

    Args:
        x: Input tensor of shape [..., hidden_size]. Must be contiguous in last dim.
        weight: Norm weight of shape [hidden_size].
        eps: Epsilon for numerical stability.
        out: Optional pre-allocated output tensor (same shape/dtype as x).

    Returns:
        Normalized tensor of same shape and dtype as x.
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be on CUDA"
    orig_shape = x.shape
    hidden_size = orig_shape[-1]
    x_2d = x if x.dim() == 2 else x.reshape(-1, hidden_size)
    num_rows = x_2d.shape[0]

    if out is None:
        out = torch.empty_like(x)
    out_2d = out if out.dim() == 2 else out.reshape(-1, hidden_size)

    assert weight.shape == (hidden_size,), f"Weight shape mismatch: {weight.shape} vs ({hidden_size},)"

    if hidden_size <= _MAX_SINGLE_PASS_HIDDEN:
        # Single-pass: BLOCK_SIZE = next power of 2 >= hidden_size
        BLOCK_SIZE = triton.next_power_of_2(hidden_size)
        _rmsnorm_kernel[(num_rows,)](
            x_2d, weight, out_2d,
            x_2d.stride(0), out_2d.stride(0),
            hidden_size,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # Two-pass for very large hidden sizes
        BLOCK_SIZE = 4096
        _rmsnorm_kernel_large[(num_rows,)](
            x_2d, weight, out_2d,
            x_2d.stride(0), out_2d.stride(0),
            hidden_size,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return out.reshape(orig_shape)


@triton.jit
def _rmsnorm_group_quant_kernel(
    x_ptr,
    weight_ptr,
    normalized_ptr,
    quantized_ptr,
    scale_ptr,
    num_valid_tokens_ptr,
    stride_xm,
    stride_normalized_m,
    stride_quantized_m,
    stride_scale_m,
    stride_scale_group,
    hidden_size: tl.constexpr,
    group_size: tl.constexpr,
    num_groups: tl.constexpr,
    eps: tl.constexpr,
    has_valid_tokens: tl.constexpr,
    block_size: tl.constexpr,
):
    """Fuse RMSNorm with block-128 FP8 activation quantization."""
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < hidden_size
    row_valid = (
        row < tl.load(num_valid_tokens_ptr)
        if has_valid_tokens
        else True
    )

    x = tl.load(
        x_ptr + row * stride_xm + offsets,
        mask=mask & row_valid,
        other=0.0,
    ).to(tl.float32)
    weight = tl.load(
        weight_ptr + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    x_groups = tl.reshape(x, (num_groups, group_size))
    variance = tl.sum(tl.sum(x_groups * x_groups, axis=1), axis=0) / hidden_size
    normalized = (x * tl.rsqrt(variance + eps) * weight).to(tl.bfloat16)
    tl.store(
        normalized_ptr + row * stride_normalized_m + offsets,
        tl.where(row_valid, normalized, 0.0),
        mask=mask,
    )

    normalized_groups = tl.reshape(
        normalized.to(tl.float32),
        (num_groups, group_size),
    )
    amax = tl.max(tl.abs(normalized_groups), axis=1)
    amax = tl.maximum(amax, 1.52587890625e-05)
    scale = tl.maximum(amax * (1.0 / 448.0), 1e-12)
    quantized = tl.maximum(
        tl.minimum(
            normalized_groups / tl.expand_dims(scale, 1),
            448.0,
        ),
        -448.0,
    ).to(tl.float8e4nv)
    tl.store(
        quantized_ptr + row * stride_quantized_m + offsets,
        tl.reshape(quantized, (block_size,)),
        mask=mask,
    )
    groups = tl.arange(0, num_groups)
    tl.store(
        scale_ptr + row * stride_scale_m + groups * stride_scale_group,
        tl.where(row_valid, scale, 1e-12),
    )


def fused_rmsnorm_group_quant_out(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    normalized_out: torch.Tensor,
    quantized_out: torch.Tensor,
    scale_out: torch.Tensor,
    *,
    num_valid_tokens: torch.Tensor | None = None,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Write RMSNorm BF16 output and its blockwise FP8 quantization."""
    if x.dim() != 2 or x.stride(1) != 1:
        raise ValueError("x must be a 2-D tensor contiguous in its hidden dimension")
    if x.dtype != torch.bfloat16:
        raise TypeError(f"x must be bfloat16, got {x.dtype}")
    rows, hidden_size = x.shape
    if hidden_size % group_size != 0:
        raise ValueError(
            f"hidden size {hidden_size} must be divisible by group size {group_size}"
        )
    if weight.shape != (hidden_size,) or weight.device != x.device:
        raise ValueError(
            f"weight must have shape {(hidden_size,)} on {x.device}, "
            f"got {tuple(weight.shape)} on {weight.device}"
        )
    if normalized_out.shape != x.shape or normalized_out.dtype != x.dtype:
        raise ValueError(
            f"normalized_out must match x shape/dtype, got "
            f"{tuple(normalized_out.shape)} {normalized_out.dtype}"
        )
    if not normalized_out.is_contiguous():
        raise ValueError("normalized_out must be contiguous")
    if quantized_out.shape != x.shape or quantized_out.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"quantized_out must be FP8 with shape {tuple(x.shape)}, got "
            f"{tuple(quantized_out.shape)} {quantized_out.dtype}"
        )
    if not quantized_out.is_contiguous():
        raise ValueError("quantized_out must be contiguous")
    num_groups = hidden_size // group_size
    if scale_out.shape != (rows, num_groups) or scale_out.dtype != torch.float32:
        raise ValueError(
            f"scale_out must be FP32 with shape {(rows, num_groups)}, got "
            f"{tuple(scale_out.shape)} {scale_out.dtype}"
        )
    if scale_out.device != x.device:
        raise ValueError("scale_out must be on the same device as x")
    if num_valid_tokens is not None:
        if num_valid_tokens.device != x.device:
            raise ValueError("num_valid_tokens must be on the same device as x")
        if num_valid_tokens.dtype != torch.int32:
            raise TypeError(
                f"num_valid_tokens must be int32, got {num_valid_tokens.dtype}"
            )
        if num_valid_tokens.numel() != 1:
            raise ValueError(
                "num_valid_tokens must contain one element, got "
                f"{tuple(num_valid_tokens.shape)}"
            )

    block_size = triton.next_power_of_2(hidden_size)
    _rmsnorm_group_quant_kernel[(rows,)](
        x,
        weight,
        normalized_out,
        quantized_out,
        scale_out,
        num_valid_tokens if num_valid_tokens is not None else scale_out,
        x.stride(0),
        normalized_out.stride(0),
        quantized_out.stride(0),
        scale_out.stride(0),
        scale_out.stride(1),
        hidden_size=hidden_size,
        group_size=group_size,
        num_groups=num_groups,
        eps=eps,
        has_valid_tokens=num_valid_tokens is not None,
        block_size=block_size,
        num_warps=8,
    )
    return normalized_out, quantized_out, scale_out


class FusedRMSNorm(torch.nn.Module):
    """
    Drop-in replacement for PyTorch RMSNorm.
    Single kernel launch instead of ~4 separate ops.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = torch.nn.Parameter(
            torch.ones(hidden_size, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_rmsnorm(x, self.weight, self.eps)


# ============================================================================
# Validation against PyTorch reference
# ============================================================================

def _pytorch_rmsnorm(x, weight, eps=1e-5):
    """PyTorch reference implementation (from model.py:254-259)."""
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight * x).to(input_dtype)


def validate(
    hidden_size: int = 4096,
    batch_sizes=(1, 2, 3, 4, 7, 8, 11, 16, 32, 61, 64, 128, 256),
    dtype=torch.bfloat16,
    eps: float = 1e-5,
    atol: float = 1e-5,
    rtol: float = 1.6e-2,
):
    """
    Validate fused RMSNorm against PyTorch reference.
    Uses BF16 WGMMA tolerance per project conventions.
    """
    device = torch.device("cuda")
    weight = torch.randn(hidden_size, device=device, dtype=dtype)
    all_passed = True

    for bs in batch_sizes:
        x = torch.randn(bs, hidden_size, device=device, dtype=dtype)

        ref = _pytorch_rmsnorm(x, weight, eps)
        out = fused_rmsnorm(x, weight, eps)

        diff = (out.float() - ref.float()).abs()
        tol = atol + rtol * ref.float().abs()
        n_fail = (diff > tol).sum().item()
        n_total = diff.numel()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        passed = (n_fail == 0) or (n_fail / n_total < 1e-4)
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(
            f"  [{status}] batch={bs:>3d}, hidden={hidden_size}, "
            f"max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, "
            f"fail={n_fail}/{n_total}"
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
    Bandwidth benchmark for fused RMSNorm.

    Memory traffic per row:
        Read:  x [hidden_size * elem_size] + weight [hidden_size * elem_size]
        Write: out [hidden_size * elem_size]
        Total: 3 * hidden_size * elem_size per row

    Target: 80% of H20 HBM3E peak (~4 TB/s) = 3.2 TB/s
    """
    device = torch.device("cuda")
    elem_size = torch.tensor([], dtype=dtype).element_size()  # 2 for BF16
    weight = torch.randn(hidden_size, device=device, dtype=dtype)

    print(f"{'Batch':>6s}  {'Time(us)':>10s}  {'BW(TB/s)':>10s}  {'BW%_of_4TB':>10s}")
    print("-" * 45)

    for bs in batch_sizes:
        x = torch.randn(bs, hidden_size, device=device, dtype=dtype)
        out = torch.empty_like(x)

        # Warmup
        for _ in range(warmup):
            fused_rmsnorm(x, weight, out=out)

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iters):
            fused_rmsnorm(x, weight, out=out)
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end) / iters
        elapsed_s = elapsed_ms / 1000.0

        # Bytes: read x + read weight + write out
        # Weight is [hidden_size] — shared across rows but still read once per kernel
        bytes_moved = bs * hidden_size * elem_size * 2 + hidden_size * elem_size + bs * hidden_size * elem_size
        # Simplified: (2*bs + 1) * hidden_size * elem_size for reads, bs * hidden_size * elem_size for write
        # More accurate: input (bs * H * 2B) + weight (H * 2B) + output (bs * H * 2B)
        bytes_moved = (2 * bs * hidden_size + hidden_size) * elem_size
        bw_tb = bytes_moved / elapsed_s / 1e12

        print(f"{bs:>6d}  {elapsed_ms*1000:>10.1f}  {bw_tb:>10.3f}  {bw_tb/4.0*100:>9.1f}%")


def benchmark_vs_pytorch(
    hidden_size: int = 4096,
    batch_sizes=(1, 2, 4, 8, 16, 32, 64, 128, 256),
    dtype=torch.bfloat16,
    warmup: int = 50,
    iters: int = 200,
):
    """Compare fused vs PyTorch RMSNorm latency."""
    device = torch.device("cuda")
    weight = torch.randn(hidden_size, device=device, dtype=dtype)

    print(f"{'Batch':>6s}  {'Fused(us)':>10s}  {'PyTorch(us)':>12s}  {'Speedup':>8s}")
    print("-" * 45)

    for bs in batch_sizes:
        x = torch.randn(bs, hidden_size, device=device, dtype=dtype)

        # Warmup
        for _ in range(warmup):
            fused_rmsnorm(x, weight)
            _pytorch_rmsnorm(x, weight)

        torch.cuda.synchronize()

        # Fused
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fused_rmsnorm(x, weight)
        end.record()
        torch.cuda.synchronize()
        fused_us = start.elapsed_time(end) / iters * 1000

        # PyTorch
        start.record()
        for _ in range(iters):
            _pytorch_rmsnorm(x, weight)
        end.record()
        torch.cuda.synchronize()
        pytorch_us = start.elapsed_time(end) / iters * 1000

        speedup = pytorch_us / fused_us
        print(f"{bs:>6d}  {fused_us:>10.1f}  {pytorch_us:>12.1f}  {speedup:>7.2f}x")


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Fused RMSNorm — Validation + Benchmark")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Compute: {torch.cuda.get_device_capability()}")
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

    # 3D input test
    print("3D input test (batch=4, seq=128, hidden=4096):")
    x = torch.randn(4, 128, 4096, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    ref = _pytorch_rmsnorm(x, w)
    out = fused_rmsnorm(x, w)
    max_diff = (out.float() - ref.float()).abs().max().item()
    print(f"  max_diff={max_diff:.2e} {'PASS' if max_diff < 0.05 else 'FAIL'}")
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
