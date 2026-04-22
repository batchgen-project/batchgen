"""FP8 Absorb Kernels for GLM-5 MLA Decode — H20 (SM90a) Only

Replaces BF16 torch.einsum for q_absorb and out_absorb with FP8 kernels.

q_absorb:  einsum("bhd,hdc->bhc") = [B, H, 192] @ [H, 192, 512] → [B, H, 512]
           Per-head: [B, 192] @ [192, 512] → [B, 512]. At B=32: M=32, K=192, N=512.

out_absorb: einsum("bqhc,hdc->bqhd") = [B, 1, H, 512] @ [H, 256, 512] → [B, 1, H, 256]
            Per-head: [B, 512] @ [512, 256] → [B, 256]. At B=32: M=32, K=512, N=256.

Two approaches:
  1. BF16 GEMV (Triton): Tiled dot product, BW-optimized for tiny M.
  2. FP8 WGMMA (Triton): On-the-fly FP8 quantization in kernel, tl.dot with FP8 → WGMMA.
     No materialized FP8 tensors. Weights pre-quantized offline.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple, Optional


# ============================================================================
# Approach 1: BF16 Triton GEMV — tiled dot product, BW-optimized
# ============================================================================

@triton.jit
def _absorb_gemv_kernel(
    X_ptr,       # [B, H, K] BF16 — input activations
    W_ptr,       # [H, K, N] BF16 — weight (contiguous)
    OUT_ptr,     # [B, H, N] BF16 — output
    stride_xb,   # B stride for X
    stride_xh,   # H stride for X
    stride_wh,   # H stride for W
    stride_ob,   # B stride for OUT
    stride_oh,   # H stride for OUT
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Each program computes [BLOCK_B, BLOCK_N] output for one head.

    Grid: (cdiv(B, BLOCK_B), H, cdiv(N, BLOCK_N))
    Inner loop tiles over K with tl.dot for register-level matmul.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)

    b_start = pid_b * BLOCK_B
    b_offs = b_start + tl.arange(0, BLOCK_B)
    b_mask = b_offs < B

    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    x_base = X_ptr + pid_h * stride_xh
    w_base = W_ptr + pid_h * stride_wh

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        x_tile = tl.load(
            x_base + b_offs[:, None] * stride_xb + k_offs[None, :],
            mask=b_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        w_tile = tl.load(
            w_base + k_offs[:, None] * N + n_offs[None, :],
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(x_tile, w_tile)

    out_base = OUT_ptr + pid_h * stride_oh
    tl.store(
        out_base + b_offs[:, None] * stride_ob + n_offs[None, :],
        acc.to(tl.bfloat16),
        mask=b_mask[:, None] & n_mask[None, :],
    )


def triton_absorb_gemv(
    x_bhk: torch.Tensor,      # [B, H, K] BF16
    w_hkn: torch.Tensor,      # [H, K, N] BF16
) -> torch.Tensor:
    """Triton BF16 GEMV for absorb — tiled dot product."""
    assert x_bhk.is_contiguous() and w_hkn.is_contiguous()
    B, H, K = x_bhk.shape
    N = w_hkn.shape[2]

    out = torch.empty(B, H, N, dtype=torch.bfloat16, device=x_bhk.device)

    BLOCK_B = min(32, triton.next_power_of_2(B))
    BLOCK_K = min(64, triton.next_power_of_2(K))
    BLOCK_N = 64

    grid = (triton.cdiv(B, BLOCK_B), H, triton.cdiv(N, BLOCK_N))

    _absorb_gemv_kernel[grid](
        x_bhk, w_hkn, out,
        x_bhk.stride(0), x_bhk.stride(1),
        w_hkn.stride(0),
        out.stride(0), out.stride(1),
        B=B, H=H, K=K, N=N,
        BLOCK_B=BLOCK_B, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
    )

    return out


# ============================================================================
# Approach 2: FP8 WGMMA — on-the-fly quantization, no materialized tensors
# ============================================================================

@triton.jit
def _absorb_fp8_wgmma_kernel(
    X_ptr,       # [B, H, K] BF16 — input activations
    W_ptr,       # [H, N, K] FP8 E4M3 — pre-quantized weight (row-major: each row is K)
    W_scale_ptr, # [H, N_scale_blocks, K_scale_blocks] FP32 — per-block weight scales
    OUT_ptr,     # [B, H, N] BF16 — output
    stride_xb,   # strides
    stride_xh,
    stride_wh,   # stride for W along H dim = N * K
    stride_wsh,  # stride for W_scale along H dim
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_B: tl.constexpr,   # tile M (batch) — 32 or 64
    BLOCK_K: tl.constexpr,   # tile K for FP8 block quantization — 128
    BLOCK_N: tl.constexpr,   # tile N — 64 or 128
    FP8_MAX: tl.constexpr,   # 448.0 for E4M3
):
    """FP8 WGMMA absorb kernel — quantizes activations on-the-fly.

    Each program computes [BLOCK_B, BLOCK_N] output for one head.
    Grid: (cdiv(B, BLOCK_B), H, cdiv(N, BLOCK_N))

    Inner loop over K in BLOCK_K chunks:
      1. Load BF16 activation tile [BLOCK_B, BLOCK_K]
      2. Per-row absmax → scale → quantize to FP8 in registers
      3. Load FP8 weight tile [BLOCK_K, BLOCK_N] (already FP8)
      4. tl.dot(x_fp8, w_fp8) → accumulate in FP32
      5. Apply combined scale (x_scale * w_scale) to accumulator
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)

    b_start = pid_b * BLOCK_B
    b_offs = b_start + tl.arange(0, BLOCK_B)
    b_mask = b_offs < B

    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    x_base = X_ptr + pid_h * stride_xh
    # W is [H, N, K] row-major, but we need [K, N] tiles for matmul
    # W[h, n, k] is at W_ptr + h * N * K + n * K + k
    w_h_base = W_ptr + pid_h * stride_wh
    ws_h_base = W_scale_ptr + pid_h * stride_wsh

    num_k_blocks = (K + BLOCK_K - 1) // BLOCK_K

    for k_block_idx in range(num_k_blocks):
        k_start = k_block_idx * BLOCK_K
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        # 1. Load BF16 activation tile [BLOCK_B, BLOCK_K]
        x_bf16 = tl.load(
            x_base + b_offs[:, None] * stride_xb + k_offs[None, :],
            mask=b_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # 2. Per-row absmax quantization to FP8
        x_abs = tl.abs(x_bf16)
        x_absmax = tl.max(x_abs, axis=1)  # [BLOCK_B]
        x_scale = tl.maximum(x_absmax, 1e-12) / FP8_MAX  # [BLOCK_B]
        x_scaled = x_bf16 / x_scale[:, None]
        x_scaled = tl.minimum(tl.maximum(x_scaled, -FP8_MAX), FP8_MAX)
        x_fp8 = x_scaled.to(tl.float8e4nv)

        # 3. Load FP8 weight tile: W[h, n_offs, k_offs] → need [BLOCK_K, BLOCK_N]
        # W is [H, N, K]: W[h, n, k] = w_h_base + n * K + k
        # We need W_tile[k, n] = W[h, n, k] — transposed load
        w_fp8 = tl.load(
            w_h_base + n_offs[None, :] * K + k_offs[:, None],
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float8e4nv)

        # 4. FP8 dot product → FP32 accumulator
        # tl.dot with FP8 inputs maps to WGMMA on SM90a
        dot_result = tl.dot(x_fp8, w_fp8)  # [BLOCK_B, BLOCK_N], FP32

        # 5. Apply scales: result = x_scale * w_scale * dot_result
        # w_scale: per-(n_block, k_block). Shape: [N_scale_blocks, K_scale_blocks]
        # For this k_block and n_tile, load the relevant w_scale entries
        # w_scale layout: [H, ceil(N/BLOCK_K_SCALE), ceil(K/BLOCK_K_SCALE)]
        # But we used BLOCK_K=128 for quantization, so k_block_idx is the scale index
        # w_scale[h, n_scale_idx, k_block_idx]

        # For per-row weight scale (act_quant style): w_scale is [H, N, ceil(K/128)]
        # Load w_scale[h, n_offs, k_block_idx] → [BLOCK_N]
        n_scale_blocks = (K + BLOCK_K - 1) // BLOCK_K
        w_scale = tl.load(
            ws_h_base + n_offs * n_scale_blocks + k_block_idx,
            mask=n_mask,
            other=1.0,
        )  # [BLOCK_N]

        # Combined scale: x_scale[b] * w_scale[n] → [BLOCK_B, BLOCK_N]
        combined_scale = x_scale[:, None] * w_scale[None, :]
        acc += dot_result * combined_scale

    # Store output
    out_base = OUT_ptr + pid_h * (B * N)  # assuming out is [B, H, N] contiguous
    # Actually: out[b, h, n] = OUT_ptr + b * H * N + h * N + n
    out_base2 = OUT_ptr
    tl.store(
        out_base2 + b_offs[:, None] * (H * N) + pid_h * N + n_offs[None, :],
        acc.to(tl.bfloat16),
        mask=b_mask[:, None] & n_mask[None, :],
    )


class FP8AbsorbWeights:
    """Pre-quantized absorb weights for FP8 WGMMA kernel.

    Quantize once at model load time, reuse across all decode steps.
    No deep_gemm dependency — uses simple per-row block quantization.
    """
    def __init__(
        self,
        q_absorb_bf16: torch.Tensor,    # [H, K=192, N=512] BF16
        out_absorb_bf16: torch.Tensor,   # [H, N=256, K=512] BF16
        block_k: int = 128,
    ):
        self.block_k = block_k

        # q_absorb: [H, 192, 512] → need [H, N=512, K=192] for NT-style
        # Weight layout: [H, N, K] — each row of N has K elements
        self._quantize_weight(
            q_absorb_bf16.transpose(1, 2).contiguous(),  # [H, N=512, K=192]
            'q_absorb',
            block_k,
        )

        # out_absorb: [H, 256, 512] — einsum "bqhc,hdc->bqhd"
        # Per-head: [B, C=512] @ [K=512, N=256] where W is [H, N=256, K=512]
        # Already in [H, N, K] layout
        self._quantize_weight(
            out_absorb_bf16.contiguous(),  # [H, N=256, K=512]
            'out_absorb',
            block_k,
        )

    def _quantize_weight(self, w_hnk: torch.Tensor, prefix: str, block_k: int):
        """Quantize [H, N, K] BF16 weight to FP8 with per-row-block scales."""
        H, N, K = w_hnk.shape
        fp8_max = 448.0
        num_k_blocks = (K + block_k - 1) // block_k

        w_fp8 = torch.empty(H, N, K, dtype=torch.float8_e4m3fn, device=w_hnk.device)
        w_scale = torch.empty(H, N, num_k_blocks, dtype=torch.float32, device=w_hnk.device)

        for kb in range(num_k_blocks):
            k_start = kb * block_k
            k_end = min(k_start + block_k, K)
            block = w_hnk[:, :, k_start:k_end].float()  # [H, N, block_len]
            absmax = block.abs().amax(dim=-1)  # [H, N]
            scale = absmax.clamp(min=1e-12) / fp8_max  # [H, N]
            scaled = block / scale.unsqueeze(-1)
            scaled = scaled.clamp(-fp8_max, fp8_max)
            w_fp8[:, :, k_start:k_end] = scaled.to(torch.float8_e4m3fn)
            w_scale[:, :, kb] = scale

        setattr(self, f'{prefix}_fp8', w_fp8)
        setattr(self, f'{prefix}_scale', w_scale)


def triton_absorb_fp8(
    x_bhk: torch.Tensor,           # [B, H, K] BF16
    w_fp8: torch.Tensor,           # [H, N, K] FP8
    w_scale: torch.Tensor,         # [H, N, ceil(K/128)] FP32
) -> torch.Tensor:
    """Triton FP8 WGMMA absorb — on-the-fly activation quantization."""
    x_bhk = x_bhk.contiguous()
    B, H, K = x_bhk.shape
    N = w_fp8.shape[1]

    out = torch.empty(B, H, N, dtype=torch.bfloat16, device=x_bhk.device)

    BLOCK_B = min(64, triton.next_power_of_2(B))
    BLOCK_K = 128  # match weight quantization block
    # Ensure BLOCK_K doesn't exceed K — pad up to power of 2
    if K < BLOCK_K:
        BLOCK_K = triton.next_power_of_2(K)
    BLOCK_N = 64

    grid = (triton.cdiv(B, BLOCK_B), H, triton.cdiv(N, BLOCK_N))

    _absorb_fp8_wgmma_kernel[grid](
        x_bhk, w_fp8, w_scale, out,
        x_bhk.stride(0), x_bhk.stride(1),
        w_fp8.shape[1] * w_fp8.shape[2],   # stride_wh = N * K
        w_scale.shape[1] * w_scale.shape[2],  # stride_wsh = N * num_k_blocks
        B=B, H=H, K=K, N=N,
        BLOCK_B=BLOCK_B, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
        FP8_MAX=448.0,
    )

    return out


# ============================================================================
# High-level API
# ============================================================================

def fp8_q_absorb_gemv(q_nope: torch.Tensor, q_absorb_w: torch.Tensor) -> torch.Tensor:
    """BF16 Triton GEMV q_absorb: [B, H, 192] @ [H, 192, 512] → [B, H, 512]"""
    return triton_absorb_gemv(q_nope.contiguous(), q_absorb_w.contiguous())


def fp8_out_absorb_gemv(
    attn_out: torch.Tensor,          # [B, 1, H, 512] or [B, H, 512]
    out_absorb_w: torch.Tensor,      # [H, 256, 512]
) -> torch.Tensor:
    """BF16 Triton GEMV out_absorb: [B, H, 512] @ [H, 512, 256] → [B, H, 256]"""
    squeeze = False
    if attn_out.dim() == 4:
        B, Q, H, C = attn_out.shape
        assert Q == 1
        attn_out = attn_out.squeeze(1)
        squeeze = True

    w_hkn = out_absorb_w.transpose(1, 2).contiguous()
    out = triton_absorb_gemv(attn_out.contiguous(), w_hkn)

    if squeeze:
        out = out.unsqueeze(1)
    return out


def fp8_q_absorb(q_nope: torch.Tensor, weights: FP8AbsorbWeights) -> torch.Tensor:
    """FP8 WGMMA q_absorb: [B, H, 192] @ [H, 192, 512] → [B, H, 512]"""
    return triton_absorb_fp8(q_nope, weights.q_absorb_fp8, weights.q_absorb_scale)


def fp8_out_absorb(
    attn_out: torch.Tensor,
    weights: FP8AbsorbWeights,
) -> torch.Tensor:
    """FP8 WGMMA out_absorb: [B, H, 512] @ [H, 512, 256] → [B, H, 256]"""
    squeeze = False
    if attn_out.dim() == 4:
        attn_out = attn_out.squeeze(1)
        squeeze = True

    out = triton_absorb_fp8(attn_out, weights.out_absorb_fp8, weights.out_absorb_scale)

    if squeeze:
        out = out.unsqueeze(1)
    return out


# ============================================================================
# PyTorch reference (for validation)
# ============================================================================

def q_absorb_reference(q_nope: torch.Tensor, q_absorb: torch.Tensor) -> torch.Tensor:
    """Reference: torch.einsum("bhd,hdc->bhc", q_nope, q_absorb)"""
    return torch.einsum("bhd,hdc->bhc", q_nope, q_absorb)


def out_absorb_reference(attn_out: torch.Tensor, out_absorb: torch.Tensor) -> torch.Tensor:
    """Reference: torch.einsum("bqhc,hdc->bqhd", attn_out, out_absorb)"""
    return torch.einsum("bqhc,hdc->bqhd", attn_out, out_absorb)
