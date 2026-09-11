"""Triton RMSNorm for Kimi-K3 decode: one program per row, one pass.

``batchgen_kernels.attention._C_fused_ops.rmsnorm_forward`` is one launch but
walks each row with 256 scalar-load threads over two passes; at the K3 decode
shapes ([bucket, 7168] / [ntp, 3584] rows) it is latency-bound at ~12 us per
call, and the whole-model graph replays ~327 norms per step. This kernel
loads the row once with vectorized accesses (BLOCK = next pow2 of the hidden
size), does the fp32 math and writes the bf16 result: ~2-3 us per call.
Numerics match the CUDA kernel (fp32 sum of squares, fp32 scale, one rounding).
"""
import torch

try:  # pragma: no cover - environment dependent
    import triton
    import triton.language as tl
    _TRITON_OK = True
except Exception:  # noqa: BLE001
    _TRITON_OK = False


if _TRITON_OK:

    @triton.jit
    def _rmsnorm_row_kernel(
        x_ptr, w_ptr, y_ptr,
        hidden: tl.constexpr, eps,
        stride_row,
        BLOCK: tl.constexpr,
    ):
        # int64 row offset: the streamed-SP8 prefill norms up to 512K token
        # rows x 7168; int32 ``row * stride`` wraps at row 299,593 (s5: the
        # last 7 of 256 sequences in a 307K-token prefill sampled token 0).
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < hidden
        base = row * stride_row
        x = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        ss = tl.sum(x * x, axis=0)
        inv = 1.0 / tl.sqrt(ss / hidden + eps)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * inv * w
        tl.store(y_ptr + base + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


def triton_rmsnorm_available() -> bool:
    return _TRITON_OK


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last dim of a CUDA tensor; same dtype in and out."""
    hidden = x.shape[-1]
    x2 = x.reshape(-1, hidden)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    rows = x2.shape[0]
    y = torch.empty_like(x2)
    if rows == 0:
        return y.view_as(x)
    block = triton.next_power_of_2(hidden)
    num_warps = 4 if block <= 2048 else (8 if block <= 8192 else 16)
    _rmsnorm_row_kernel[(rows,)](
        x2, weight, y, hidden, float(eps), x2.stride(0),
        BLOCK=block, num_warps=num_warps,
    )
    return y.view_as(x)
