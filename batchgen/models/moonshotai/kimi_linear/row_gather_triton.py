"""Fused row gathers for the Kimi-K3 whole-model decode graph.

The balanced-split MoE routing permutes rows twice per MoE layer:
``padded = hidden[idx] * valid`` before the MoE and
``hidden = residual + moe_out[idx] * valid`` after it. Each is one Triton
kernel here instead of gather + mul (+ add) launches (3 fewer graph nodes per
layer). Rows with ``valid == 0`` are written as zeros without reading the
source row, so the index of an invalid row may be anything in range.
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
    def _gather_rows_masked_kernel(
        src_ptr, idx_ptr, valid_ptr, out_ptr, hidden, stride_src, stride_out,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cb = tl.program_id(1)
        cols = cb * BLOCK + tl.arange(0, BLOCK)
        mask = cols < hidden
        valid = tl.load(valid_ptr + row).to(tl.int32)
        src_row = tl.load(idx_ptr + row).to(tl.int64)
        x = tl.load(src_ptr + src_row * stride_src + cols, mask=mask & (valid != 0), other=0.0)
        tl.store(out_ptr + row.to(tl.int64) * stride_out + cols, x, mask=mask)

    @triton.jit
    def _add_gathered_rows_masked_kernel(
        res_ptr, src_ptr, idx_ptr, valid_ptr, out_ptr, hidden,
        stride_res, stride_src, stride_out,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cb = tl.program_id(1)
        cols = cb * BLOCK + tl.arange(0, BLOCK)
        mask = cols < hidden
        valid = tl.load(valid_ptr + row).to(tl.int32)
        src_row = tl.load(idx_ptr + row).to(tl.int64)
        r = tl.load(res_ptr + row.to(tl.int64) * stride_res + cols, mask=mask, other=0.0)
        x = tl.load(src_ptr + src_row * stride_src + cols, mask=mask & (valid != 0), other=0.0)
        # same arithmetic as ``residual + gathered * valid`` in the row dtype
        out = r + x
        tl.store(out_ptr + row.to(tl.int64) * stride_out + cols, out, mask=mask)


def row_gather_triton_available() -> bool:
    return _TRITON_OK


def _check(t: torch.Tensor, name: str) -> None:
    if not t.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if t.dim() != 2 or t.stride(1) != 1:
        raise ValueError(f"{name} must be 2-D with a contiguous last dim")


def gather_rows_masked(src: torch.Tensor, idx: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """out[i] = src[idx[i]] if valid[i] else 0   ([rows, H], src.dtype)."""
    _check(src, "src")
    rows = idx.shape[0]
    hidden = src.shape[1]
    out = torch.empty((rows, hidden), dtype=src.dtype, device=src.device)
    if rows == 0:
        return out
    block = 1024
    _gather_rows_masked_kernel[(rows, triton.cdiv(hidden, block))](
        src, idx, valid, out, hidden, src.stride(0), out.stride(0),
        BLOCK=block, num_warps=4,
    )
    return out


def add_gathered_rows_masked(residual: torch.Tensor, src: torch.Tensor,
                             idx: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """out[i] = residual[i] + (src[idx[i]] if valid[i] else 0)."""
    _check(residual, "residual")
    _check(src, "src")
    rows, hidden = residual.shape
    if src.shape[1] != hidden:
        raise ValueError(f"hidden mismatch: residual {hidden} vs src {src.shape[1]}")
    out = torch.empty_like(residual)
    if rows == 0:
        return out
    block = 1024
    _add_gathered_rows_masked_kernel[(rows, triton.cdiv(hidden, block))](
        residual, src, idx, valid, out, hidden,
        residual.stride(0), src.stride(0), out.stride(0),
        BLOCK=block, num_warps=4,
    )
    return out
