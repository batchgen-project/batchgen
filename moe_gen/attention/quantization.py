import torch
import triton
import triton.language as tl

@triton.jit
def _dequant_kernel(
    q_ptr, scale_ptr, x_ptr,
    M, dim, num_blocks,
    stride_qm, stride_qd,
    stride_sm, stride_sb,
    stride_xm, stride_xd,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)    # which row (0 … M)
    pid_b = tl.program_id(1)    # which block in that row

    # compute start index of this block in the row
    start = pid_b * BLOCK_SIZE

    # vector of element‐indices within this block
    offs = start + tl.arange(0, BLOCK_SIZE)
    # mask out any lanes that run past dim
    mask = offs < dim

    # load FP8_e4m3fn values, cast to FP32
    q_off = q_ptr + pid_m * stride_qm + offs * stride_qd
    q_vals = tl.load(q_off, mask=mask)
    q_fp32 = tl.cast(q_vals, tl.float32)

    # load per‐block scale scalar - no need to broadcast explicitly
    s_off = scale_ptr + pid_m * stride_sm + pid_b * stride_sb
    s = tl.load(s_off)  # float32

    # Scalar-vector multiplication is handled implicitly
    # No need for explicit broadcasting
    out = tl.cast(q_fp32 * s, tl.bfloat16)

    # store back
    x_off = x_ptr + pid_m * stride_xm + offs * stride_xd
    tl.store(x_off, out, mask=mask)

def dequant_per_token_triton(q: torch.Tensor, scale: torch.Tensor, BLOCK_SIZE: int = 128):
    """Dequantize FP8 tensor with given block size"""
    assert q.is_cuda and scale.is_cuda
    assert q.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32

    bsz, seq_len, dim = q.shape
    M = bsz * seq_len
    num_blocks = (dim + BLOCK_SIZE - 1) // BLOCK_SIZE

    q_flat     = q.view(M, dim)
    scale_flat = scale.view(M, num_blocks)
    x_flat     = torch.empty((M, dim), device=q.device, dtype=torch.bfloat16)

    grid = (M, num_blocks)
    _dequant_kernel[grid](
        q_flat, scale_flat, x_flat,
        M, dim, num_blocks,
        q_flat.stride(0), q_flat.stride(1),
        scale_flat.stride(0), scale_flat.stride(1),
        x_flat.stride(0), x_flat.stride(1),
        BLOCK_SIZE
    )
    return x_flat.view(bsz, seq_len, dim)