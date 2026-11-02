"""Fused Indexer KV Projection — CUDA WGMMA (H20 SM90a)

Version: v3 — Single-WG TMA-both + separate act_quant
Hypothesis: v2 2-WG producer/math pipeline has too much overhead for tiny M.
            Single-WG with TMA for both A (pre-quantized) and B should be faster.
Result: POSITIVE — 1.47-1.62× over Torch, 1.12-2.09× over Triton (42-47µs)

Pipeline: hidden_states [B, 6144] BF16 → act_quant [B, 6144] FP8 + scale
          → WGMMA kernel (TMA both A+B) → [B, 128] BF16
          → RMSNorm [B, 128] BF16

Dimensions (GLM-5 indexer):
  M = B ≤ 64 (decode batch, padded to BLOCK_M=64)
  K = 6144 (hidden_size)
  N = 128 (index_head_dim)
  num_k_blocks = 48 (6144 / 128)

Architecture: 1 warpgroup (128 threads) per CTA
  TMA loads both A (act FP8) and B (weight FP8) with 128B swizzle
  WGMMA m64n32k32 FP8×FP8→FP32 + post-scale from quant scales
  4-stage pipeline for K dimension

Grid: (cdiv(B, 64), cdiv(N, 32)) = typically (1, 4)

The kernel is the AOT extension
``batchgen_kernels.attention._C_indexer_kv_proj_wgmma`` (source
``batchgen_kernels/src/attention/indexer_kv_proj_wgmma.cu``, built with
``-DNUM_STAGES=4``), declared in ``setup.py`` and mirrored in
``_jit_registry.py``. It is imported on first use and never compiled at
runtime.
"""

import torch

_module_cache = {}

_BLOCK_M = 64
_BLOCK_N = 32
_BLOCK_K = 128
_NUM_STAGES = 4

_EXTENSION = "batchgen_kernels.attention._C_indexer_kv_proj_wgmma"


# ──────────────────────────────────────────────────────────────────────────────
# Build
# ──────────────────────────────────────────────────────────────────────────────

def build_module(num_stages=4):
    """Return the AOT indexer KV projection extension.

    Only the ``NUM_STAGES=4`` pipeline is built ahead of time; any other depth
    would need a runtime compile, which this path no longer performs.
    """
    if num_stages != _NUM_STAGES:
        raise ValueError(
            f"indexer KV projection is built ahead of time with "
            f"NUM_STAGES={_NUM_STAGES} only, got num_stages={num_stages!r}"
        )
    cache_key = ("indexer_kv_v3", num_stages)
    if cache_key in _module_cache:
        return _module_cache[cache_key]

    # Resolved at call time so tests can substitute the loader.
    from batchgen_kernels import load_extension

    module = load_extension(_EXTENSION, allow_dev_jit=False)
    _module_cache[cache_key] = module
    return module


# ──────────────────────────────────────────────────────────────────────────────
# Weight preparation
# ──────────────────────────────────────────────────────────────────────────────

class FP8IndexerWeightsCUDA:
    """Pre-quantized indexer wk weights for CUDA WGMMA kernel."""

    def __init__(self, wk_weight_bf16: torch.Tensor, module, block_k: int = 128):
        N, K = wk_weight_bf16.shape
        fp8_max = 448.0
        num_k_blocks = (K + block_k - 1) // block_k

        self.w_fp8 = torch.empty(N, K, dtype=torch.float8_e4m3fn,
                                 device=wk_weight_bf16.device)
        self.w_scale = torch.empty(N // 32, num_k_blocks, dtype=torch.float32,
                                   device=wk_weight_bf16.device)

        for n_tile in range(N // 32):
            n_start = n_tile * 32
            n_end = n_start + 32
            for kb in range(num_k_blocks):
                k_start = kb * block_k
                k_end = min(k_start + block_k, K)
                block = wk_weight_bf16[n_start:n_end, k_start:k_end].float()
                absmax = block.abs().amax()
                scale = absmax.clamp(min=1e-12) / fp8_max
                self.w_fp8[n_start:n_end, k_start:k_end] = (block / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
                self.w_scale[n_tile, kb] = scale

        self.block_k = block_k
        self.N = N
        self.K = K

        # Create TMA descriptor for weight [N, K] with box (32, 128)
        self.tma_desc = module.create_tma_desc(self.w_fp8, N, K, 32, 128)


# ──────────────────────────────────────────────────────────────────────────────
# High-level API
# ──────────────────────────────────────────────────────────────────────────────

def cuda_wk_proj_rmsnorm(
    hidden_states: torch.Tensor,
    weights: FP8IndexerWeightsCUDA,
    rmsnorm_weight: torch.Tensor,
    module,
    eps: float = 1e-6,
) -> torch.Tensor:
    hidden_states = hidden_states.contiguous()
    B, K = hidden_states.shape
    N = weights.N

    # Act quant
    x_fp8 = torch.empty(B, K, dtype=torch.float8_e4m3fn, device=hidden_states.device)
    x_scale = torch.empty(B, dtype=torch.float32, device=hidden_states.device)
    module.run_act_quant(hidden_states, x_fp8, x_scale)

    # Create TMA desc for activation — need padded to BLOCK_M=64
    B_padded = max(B, 64)
    if B < 64:
        x_fp8_padded = torch.zeros(B_padded, K, dtype=torch.float8_e4m3fn, device=x_fp8.device)
        x_fp8_padded[:B] = x_fp8
        x_fp8 = x_fp8_padded
    a_tma_desc = module.create_tma_desc(x_fp8, B_padded, K, 64, 128)

    return module.indexer_kv_proj_forward(
        a_tma_desc, weights.tma_desc,
        weights.w_scale, x_scale,
        rmsnorm_weight,
        B, N, K, eps,
    )


def cuda_wk_proj_rmsnorm_out(
    hidden_states: torch.Tensor,
    weights: FP8IndexerWeightsCUDA,
    rmsnorm_weight: torch.Tensor,
    module,
    x_fp8_padded: torch.Tensor,
    x_scale: torch.Tensor,
    a_tma_desc: torch.Tensor,
    out_gemm: torch.Tensor,
    out: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be contiguous for graph-captured WK projection")
    B, K = hidden_states.shape
    N = weights.N
    _validate_projection_out_buffers(B, K, N, x_fp8_padded, x_scale, out)
    if out_gemm.shape != (B, N) or out_gemm.dtype != torch.bfloat16:
        raise ValueError(f"out_gemm must be BF16 with shape {(B, N)}, got {out_gemm.shape} {out_gemm.dtype}")

    module.run_act_quant(hidden_states, x_fp8_padded[:B], x_scale)
    module.indexer_kv_proj_forward_out(
        a_tma_desc,
        weights.tma_desc,
        weights.w_scale,
        x_scale,
        rmsnorm_weight,
        out_gemm,
        out,
        B,
        N,
        K,
        eps,
    )
    return out


def cuda_wk_proj_gemm_only(
    hidden_states: torch.Tensor,
    weights: FP8IndexerWeightsCUDA,
    module,
) -> torch.Tensor:
    hidden_states = hidden_states.contiguous()
    B, K = hidden_states.shape
    N = weights.N

    x_fp8 = torch.empty(B, K, dtype=torch.float8_e4m3fn, device=hidden_states.device)
    x_scale = torch.empty(B, dtype=torch.float32, device=hidden_states.device)
    module.run_act_quant(hidden_states, x_fp8, x_scale)

    B_padded = max(B, 64)
    if B < 64:
        x_fp8_padded = torch.zeros(B_padded, K, dtype=torch.float8_e4m3fn, device=x_fp8.device)
        x_fp8_padded[:B] = x_fp8
        x_fp8 = x_fp8_padded
    a_tma_desc = module.create_tma_desc(x_fp8, B_padded, K, 64, 128)

    return module.indexer_kv_proj_gemm_only(
        a_tma_desc, weights.tma_desc,
        weights.w_scale, x_scale,
        B, N, K,
    )


def cuda_wk_proj_gemm_only_out(
    hidden_states: torch.Tensor,
    weights: FP8IndexerWeightsCUDA,
    module,
    x_fp8_padded: torch.Tensor,
    x_scale: torch.Tensor,
    a_tma_desc: torch.Tensor,
    out: torch.Tensor,
    num_valid_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    if not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be contiguous for graph-captured WK projection")
    B, K = hidden_states.shape
    N = weights.N
    _validate_projection_out_buffers(B, K, N, x_fp8_padded, x_scale, out)

    if num_valid_tokens is None:
        module.run_act_quant(hidden_states, x_fp8_padded[:B], x_scale)
    else:
        module.run_act_quant_valid(hidden_states, x_fp8_padded[:B], x_scale, num_valid_tokens)
    if num_valid_tokens is None:
        module.indexer_kv_proj_gemm_only_out(
            a_tma_desc,
            weights.tma_desc,
            weights.w_scale,
            x_scale,
            out,
            B,
            N,
            K,
        )
    else:
        module.indexer_kv_proj_gemm_only_valid_m_out(
            a_tma_desc,
            weights.tma_desc,
            weights.w_scale,
            x_scale,
            out,
            num_valid_tokens,
            B,
            N,
            K,
        )
    return out


def make_fp8_activation_scratch(
    batch_size: int,
    hidden_size: int,
    module,
    *,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    padded_batch = max(batch_size, _BLOCK_M)
    x_fp8_padded = torch.empty(
        padded_batch,
        hidden_size,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    x_scale = torch.empty(batch_size, dtype=torch.float32, device=device)
    a_tma_desc = module.create_tma_desc(
        x_fp8_padded,
        padded_batch,
        hidden_size,
        _BLOCK_M,
        _BLOCK_K,
    )
    return x_fp8_padded, x_scale, a_tma_desc


def _validate_projection_out_buffers(
    B: int,
    K: int,
    N: int,
    x_fp8_padded: torch.Tensor,
    x_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    padded_batch = max(B, _BLOCK_M)
    if x_fp8_padded.shape != (padded_batch, K) or x_fp8_padded.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"x_fp8_padded must be FP8 with shape {(padded_batch, K)}, "
            f"got {x_fp8_padded.shape} {x_fp8_padded.dtype}"
        )
    if x_scale.shape != (B,) or x_scale.dtype != torch.float32:
        raise ValueError(f"x_scale must be FP32 with shape {(B,)}, got {x_scale.shape} {x_scale.dtype}")
    if out.shape != (B, N) or out.dtype != torch.bfloat16:
        raise ValueError(f"out must be BF16 with shape {(B, N)}, got {out.shape} {out.dtype}")


# ──────────────────────────────────────────────────────────────────────────────
# Reference
# ──────────────────────────────────────────────────────────────────────────────

def wk_proj_rmsnorm_reference(
    hidden_states: torch.Tensor,
    wk_weight: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    k = torch.nn.functional.linear(hidden_states, wk_weight)
    variance = k.float().pow(2).mean(-1, keepdim=True)
    k_normed = k * torch.rsqrt(variance + eps)
    return (k_normed * rmsnorm_weight).to(torch.bfloat16)
