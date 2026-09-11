"""Triton SiTU-and-multiply for Kimi-K3 (shared expert / dense MLP).

``SituAndMul.forward`` is ~10 elementwise launches (two fp32 casts, tanh,
divides, sigmoid, products, the cast back); in the decode graph that is
~12 us per MoE layer of tiny kernels. This kernel reads ``[gate, up]`` once
and writes the product: the same fp32 math and one rounding to the output
dtype. ``tanh`` is evaluated as ``1 - 2 / (exp(2x) + 1)``, which agrees with
libdevice to fp32 rounding; the row layout ([rows, 2d] -> [rows, d]) is the
module's contract.
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
    def _tanh(x):
        # numerically safe: exp(2x) overflows to inf -> 1 - 0 = 1; exp(-large) -> 1 - 2/1 = -1
        return 1.0 - 2.0 / (tl.exp(2.0 * x) + 1.0)

    @triton.jit
    def _situ_mul_kernel(
        x_ptr, y_ptr, d, stride_x, stride_y,
        beta, inv_beta, linear_beta, inv_linear_beta,
        HAS_LINEAR_BETA: tl.constexpr, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cb = tl.program_id(1)
        cols = cb * BLOCK + tl.arange(0, BLOCK)
        mask = cols < d
        base = x_ptr + row * stride_x
        gate = tl.load(base + cols, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(base + d + cols, mask=mask, other=0.0).to(tl.float32)
        situ_a = beta * _tanh(gate * inv_beta) * tl.sigmoid(gate)
        if HAS_LINEAR_BETA:
            up = linear_beta * _tanh(up * inv_linear_beta)
        out = situ_a * up
        tl.store(y_ptr + row * stride_y + cols, out.to(y_ptr.dtype.element_ty), mask=mask)


def situ_triton_available() -> bool:
    return _TRITON_OK


def situ_and_mul_triton(x: torch.Tensor, beta: float, linear_beta) -> torch.Tensor:
    """``x[..., :d]`` = gate, ``x[..., d:]`` = up -> ``[..., d]`` in x.dtype."""
    d = x.shape[-1] // 2
    x2 = x.reshape(-1, 2 * d)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    rows = x2.shape[0]
    y = torch.empty((rows, d), dtype=x.dtype, device=x.device)
    if rows == 0:
        return y.reshape(*x.shape[:-1], d)
    block = 1024
    grid = (rows, triton.cdiv(d, block))
    has_lb = linear_beta is not None
    lb = float(linear_beta) if has_lb else 1.0
    _situ_mul_kernel[grid](
        x2, y, d, x2.stride(0), y.stride(0),
        float(beta), 1.0 / float(beta), lb, 1.0 / lb,
        HAS_LINEAR_BETA=has_lb, BLOCK=block, num_warps=4,
    )
    return y.reshape(*x.shape[:-1], d)
