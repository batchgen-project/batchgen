"""
Fused grouped MoE kernels using WGMMA for GPT-OSS-120B decode.

Ported from the validated TMA kernel in batchgen_kernels/moe/gptoss/grouped_mxfp4_tma.py
with bias support added from the production manual-load kernel.

Two CUDA kernels (TMA for A-matrix, Phase 10a byte-LUT for MXFP4 weights):
- Stage 1: gate + up + SwiGLU (3D grid: experts × N-tiles × M-tiles)
- Stage 2: down projection (3D grid: experts × K-tiles × M-tiles)

Pipeline: dispatch -> WGMMA S1 -> WGMMA S2 -> reduce (4 kernel launches)

Usage:
    from batchgen.moe.fused_wgmma_grouped import (
        fused_mxfp4_grouped_moe_forward_cuda_routing,
        is_grouped_wgmma_available,
    )

    if is_grouped_wgmma_available():
        output = fused_mxfp4_grouped_moe_forward_cuda_routing(
            hidden_states, topk_indices, topk_weights, ...)
"""

import os
import logging

import torch


# Module-level state
_grouped_wgmma_available = None
_grouped_module = None


# ──────────────────────────────────────────────────────────────────────────────
# CUDA Source Code (external .cu file)
# ──────────────────────────────────────────────────────────────────────────────
# Ported from batchgen_kernels/moe/gptoss/grouped_mxfp4_tma.py
# with bias support from the validated production kernel
# Source: batchgen_kernels/src/moe/grouped_mxfp4_wgmma.cu

# ──────────────────────────────────────────────────────────────────────────────
# Module Loading
# ──────────────────────────────────────────────────────────────────────────────

def _check_wgmma_support() -> bool:
    """Check if WGMMA (SM90) is supported on this system."""
    if not torch.cuda.is_available():
        return False
    device = torch.cuda.current_device()
    cc = torch.cuda.get_device_capability(device)
    if cc[0] < 9:
        logging.debug(f"WGMMA requires SM90+, found SM{cc[0]}{cc[1]}")
        return False
    return True


def _load_grouped_module():
    """Load the grouped WGMMA CUDA module (pre-compiled via pip install)."""
    global _grouped_module

    if _grouped_module is not None:
        return _grouped_module

    try:
        import batchgen_kernels
        _grouped_module = batchgen_kernels.load_extension("batchgen_kernels.moe._C_grouped_mxfp4_wgmma")
        logging.info("Loaded pre-compiled WGMMA fused grouped MXFP4 MoE kernels")
        return _grouped_module
    except Exception as e:
        logging.warning(f"Failed to load WGMMA grouped MoE kernels: {e}")
        return None


def is_grouped_wgmma_available() -> bool:
    """Check if grouped WGMMA fused kernels are available."""
    global _grouped_wgmma_available

    if _grouped_wgmma_available is not None:
        return _grouped_wgmma_available

    if not _check_wgmma_support():
        _grouped_wgmma_available = False
        return False

    if os.environ.get("BATCHGEN_DISABLE_WGMMA_GROUPED", "0") == "1":
        logging.info("WGMMA grouped kernels disabled by BATCHGEN_DISABLE_WGMMA_GROUPED")
        _grouped_wgmma_available = False
        return False

    mod = _load_grouped_module()
    _grouped_wgmma_available = mod is not None
    return _grouped_wgmma_available


# ──────────────────────────────────────────────────────────────────────────────
# Low-Level Python Wrappers
# ──────────────────────────────────────────────────────────────────────────────

def fused_mxfp4_grouped_stage1(
    sorted_hidden: torch.Tensor,       # [total_tokens, K] BF16
    expert_offsets: torch.Tensor,       # [num_experts + 1] int32
    gate_ptrs: torch.Tensor,           # [num_experts] int64
    gate_scale_ptrs: torch.Tensor,     # [num_experts] int64
    up_ptrs: torch.Tensor,             # [num_experts] int64
    up_scale_ptrs: torch.Tensor,       # [num_experts] int64
    N: int,                            # intermediate_size
    stride_weight_n: int,              # K // 2
    stride_scale_n: int,               # K // 32
    gate_bias_ptrs: torch.Tensor = None,  # [num_experts] int64 or None
    up_bias_ptrs: torch.Tensor = None,    # [num_experts] int64 or None
) -> torch.Tensor:                     # [total_tokens, N] BF16
    """Grouped MXFP4 Stage 1: gate + up + SwiGLU via WGMMA with TMA.

    Operates on 1D+offsets layout directly from CUDA dispatch.
    3D grid (num_experts, N-tiles, max_m_tiles) with TMA for A-matrix loads.

    Args:
        sorted_hidden: Dispatched tokens [total_tokens, K] BF16
        expert_offsets: Cumulative offsets [num_experts + 1] int32
        gate_ptrs/gate_scale_ptrs: Pointer arrays [num_experts] int64
        up_ptrs/up_scale_ptrs: Pointer arrays [num_experts] int64
        N: Intermediate dimension (gate/up output width)
        stride_weight_n: Weight stride along N (= K // 2 for MXFP4)
        stride_scale_n: Scale stride along N (= K // 32 for MXFP4)
        gate_bias_ptrs: Pointer array for gate biases [num_experts] int64, or None
        up_bias_ptrs: Pointer array for up biases [num_experts] int64, or None

    Returns:
        Intermediate activations [total_tokens, N] BF16 after SwiGLU
    """
    mod = _load_grouped_module()
    assert mod is not None, "WGMMA grouped module not available"

    total_tokens = sorted_hidden.shape[0]
    max_m_tiles = (total_tokens + 63) // 64 if total_tokens > 0 else 1

    empty_bias = torch.empty(0, dtype=torch.int64, device=sorted_hidden.device)
    gb = gate_bias_ptrs if gate_bias_ptrs is not None else empty_bias
    ub = up_bias_ptrs if up_bias_ptrs is not None else empty_bias

    return mod.grouped_mxfp4_moe_stage1(
        sorted_hidden, expert_offsets,
        gate_ptrs, gate_scale_ptrs,
        up_ptrs, up_scale_ptrs,
        gb, ub,
        N, stride_weight_n, stride_scale_n,
        max_m_tiles,
    )


def fused_mxfp4_grouped_stage2(
    intermediate: torch.Tensor,        # [total_tokens, N] BF16
    expert_offsets: torch.Tensor,       # [num_experts + 1] int32
    down_ptrs: torch.Tensor,           # [num_experts] int64
    down_scale_ptrs: torch.Tensor,     # [num_experts] int64
    K: int,                            # hidden_size (output width)
    stride_weight_n: int,              # N // 2
    stride_scale_n: int,               # N // 32
    down_bias_ptrs: torch.Tensor = None,  # [num_experts] int64 or None
) -> torch.Tensor:                     # [total_tokens, K] BF16
    """Grouped MXFP4 Stage 2: down projection via WGMMA with TMA.

    3D grid (num_experts, K-tiles, max_m_tiles) with TMA for A-matrix loads.

    Args:
        intermediate: Stage 1 output [total_tokens, N] BF16
        expert_offsets: Cumulative offsets [num_experts + 1] int32
        down_ptrs/down_scale_ptrs: Pointer arrays [num_experts] int64
        K: Hidden size (output width)
        stride_weight_n: Weight stride along N (= N // 2 for MXFP4)
        stride_scale_n: Scale stride along N (= N // 32 for MXFP4)
        down_bias_ptrs: Pointer array for down biases [num_experts] int64, or None

    Returns:
        Output activations [total_tokens, K] BF16
    """
    mod = _load_grouped_module()
    assert mod is not None, "WGMMA grouped module not available"

    empty_bias = torch.empty(0, dtype=torch.int64, device=intermediate.device)
    db = down_bias_ptrs if down_bias_ptrs is not None else empty_bias

    total_tokens = intermediate.shape[0]
    max_m_tiles = (total_tokens + 63) // 64 if total_tokens > 0 else 1

    return mod.grouped_mxfp4_moe_stage2(
        intermediate, expert_offsets,
        down_ptrs, down_scale_ptrs,
        db,
        K, stride_weight_n, stride_scale_n,
        max_m_tiles,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CUDA Graph Compatible: In-place Wrappers + TMA Descriptor Creation
# ──────────────────────────────────────────────────────────────────────────────

_BLOCK_M = 64
_BLOCK_K = 64

# Cached empty bias tensor per device (avoids allocation during graph capture)
_empty_bias_cache = {}


def _get_empty_bias(device: torch.device) -> torch.Tensor:
    """Return a cached zero-size int64 tensor for use as empty bias placeholder."""
    key = str(device)
    if key not in _empty_bias_cache:
        _empty_bias_cache[key] = torch.empty(0, dtype=torch.int64, device=device)
    return _empty_bias_cache[key]


def create_tma_descriptor(tensor: torch.Tensor, block_m: int = _BLOCK_M, block_k: int = _BLOCK_K) -> torch.Tensor:
    """Create a TMA descriptor for a 2D BF16 tensor.

    Returns a CPU uint8 tensor of 128 bytes encoding the CUtensorMap.
    Must be called BEFORE CUDA graph capture — TMA descriptor creation
    is a CPU-side driver API call and is not capturable.

    Args:
        tensor: 2D BF16 CUDA tensor (the global memory source for TMA loads)
        block_m: Tile height (default 64)
        block_k: Tile width (default 64)

    Returns:
        torch.Tensor of shape [128], dtype uint8, on CPU
    """
    mod = _load_grouped_module()
    assert mod is not None, "WGMMA grouped module not available"
    return mod.create_tma_desc_bf16(tensor, block_m, block_k)


def fused_mxfp4_grouped_stage1_inplace(
    sorted_hidden: torch.Tensor,
    output: torch.Tensor,
    tma_desc: torch.Tensor,
    expert_offsets: torch.Tensor,
    gate_ptrs: torch.Tensor,
    gate_scale_ptrs: torch.Tensor,
    up_ptrs: torch.Tensor,
    up_scale_ptrs: torch.Tensor,
    N: int,
    stride_weight_n: int,
    stride_scale_n: int,
    gate_bias_ptrs: torch.Tensor = None,
    up_bias_ptrs: torch.Tensor = None,
) -> None:
    """In-place Stage 1 with pre-built TMA descriptor. CUDA graph compatible."""
    mod = _load_grouped_module()
    assert mod is not None, "WGMMA grouped module not available"

    total_tokens = sorted_hidden.shape[0]
    max_m_tiles = (total_tokens + 63) // 64 if total_tokens > 0 else 1

    gb = gate_bias_ptrs if gate_bias_ptrs is not None else _get_empty_bias(sorted_hidden.device)
    ub = up_bias_ptrs if up_bias_ptrs is not None else _get_empty_bias(sorted_hidden.device)

    mod.grouped_mxfp4_moe_stage1_inplace(
        sorted_hidden, output, tma_desc, expert_offsets,
        gate_ptrs, gate_scale_ptrs,
        up_ptrs, up_scale_ptrs,
        gb, ub,
        N, stride_weight_n, stride_scale_n,
        max_m_tiles,
    )


def fused_mxfp4_grouped_stage2_inplace(
    intermediate: torch.Tensor,
    output: torch.Tensor,
    tma_desc: torch.Tensor,
    expert_offsets: torch.Tensor,
    down_ptrs: torch.Tensor,
    down_scale_ptrs: torch.Tensor,
    K: int,
    stride_weight_n: int,
    stride_scale_n: int,
    down_bias_ptrs: torch.Tensor = None,
) -> None:
    """In-place Stage 2 with pre-built TMA descriptor. CUDA graph compatible."""
    mod = _load_grouped_module()
    assert mod is not None, "WGMMA grouped module not available"

    total_tokens = intermediate.shape[0]
    max_m_tiles = (total_tokens + 63) // 64 if total_tokens > 0 else 1

    db = down_bias_ptrs if down_bias_ptrs is not None else _get_empty_bias(intermediate.device)

    mod.grouped_mxfp4_moe_stage2_inplace(
        intermediate, output, tma_desc, expert_offsets,
        down_ptrs, down_scale_ptrs,
        db,
        K, stride_weight_n, stride_scale_n,
        max_m_tiles,
    )


# ──────────────────────────────────────────────────────────────────────────────
# End-to-End API with CUDA Routing
# ──────────────────────────────────────────────────────────────────────────────

def fused_mxfp4_grouped_moe_forward_cuda_routing(
    hidden_states: torch.Tensor,       # [batch*seq, hidden] BF16
    topk_indices: torch.Tensor,        # [batch*seq, K] int32
    topk_weights: torch.Tensor,        # [batch*seq, K] FP32
    # Pre-computed pointer arrays
    gate_ptrs: torch.Tensor,
    gate_scale_ptrs: torch.Tensor,
    up_ptrs: torch.Tensor,
    up_scale_ptrs: torch.Tensor,
    down_ptrs: torch.Tensor,
    down_scale_ptrs: torch.Tensor,
    # Reference weights for stride computation
    gate_weight_ref: torch.Tensor,
    gate_scale_ref: torch.Tensor,
    down_weight_ref: torch.Tensor,
    down_scale_ref: torch.Tensor,
    num_experts: int = 128,
    expert_start: int = 0,
    num_local_experts: int = 128,
    # Bias pointer arrays (None = no biases)
    gate_bias_ptrs: torch.Tensor = None,
    up_bias_ptrs: torch.Tensor = None,
    down_bias_ptrs: torch.Tensor = None,
    output: torch.Tensor = None,
) -> torch.Tensor:
    """End-to-end grouped MXFP4 MoE forward using WGMMA + CUDA routing.

    Full pipeline: dispatch -> WGMMA S1 -> WGMMA S2 -> reduce (4 kernel launches).
    Drop-in replacement for grouped_mxfp4_moe_forward_cuda_routing in
    mxfp4_grouped_gemm.py which uses 9+ launches.

    Args:
        hidden_states: Input [batch*seq, hidden] BF16
        topk_indices: Expert indices [batch*seq, K] int32
        topk_weights: Routing weights [batch*seq, K] FP32
        gate_ptrs, gate_scale_ptrs: Gate weight/scale pointer arrays
        up_ptrs, up_scale_ptrs: Up weight/scale pointer arrays
        down_ptrs, down_scale_ptrs: Down weight/scale pointer arrays
        gate_weight_ref, gate_scale_ref: Reference tensors for stride computation
        down_weight_ref, down_scale_ref: Reference tensors for stride computation
        num_experts: Total number of experts
        expert_start: First local expert index (for EP)
        num_local_experts: Number of local experts
        gate_bias_ptrs: Gate bias pointer array [num_experts] int64, or None
        up_bias_ptrs: Up bias pointer array [num_experts] int64, or None
        down_bias_ptrs: Down bias pointer array [num_experts] int64, or None
        output: Optional pre-allocated output buffer [batch*seq, hidden] BF16

    Returns:
        Output [batch*seq, hidden] BF16
    """
    from batchgen.moe.routing import dispatch_count_gather_cuda, reduce_weighted_scatter_cuda

    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    K_topk = topk_indices.shape[1]
    N_intermediate = gate_weight_ref.shape[0]  # intermediate_size

    # Step 1: CUDA dispatch (count + prefix_sum + gather)
    dispatched_x, expert_counts, expert_offsets, topk_pos = dispatch_count_gather_cuda(
        hidden_states, topk_indices,
        expert_start, num_local_experts,
    )

    # No CPU-GPU sync: pass dispatched_x at full allocated size.
    # The WGMMA kernels bound per-expert work via expert_offsets and early-return
    # for excess M-tiles (line 367: if (m_tile >= num_m_tiles) return).

    # TMA descriptor requires gmem_rows >= BLOCK_M (64). When total dispatched
    # tokens < 64 (e.g. BS=1 with 8 EP ranks → 32 dispatched), pad to BLOCK_M.
    # Extra rows are zeros, never referenced by expert_offsets, so they don't
    # affect correctness. reduce_weighted_scatter_cuda uses num_tokens (original).
    _BLOCK_M = 64
    if dispatched_x.shape[0] < _BLOCK_M:
        padded_dx = torch.zeros(
            _BLOCK_M, hidden_size, dtype=dispatched_x.dtype, device=dispatched_x.device
        )
        padded_dx[:dispatched_x.shape[0]] = dispatched_x
        dispatched_x = padded_dx

    # Compute strides from reference weights
    # Stage 1 (gate/up): weight is [N, K//2], scale is [N, K//32]
    s1_stride_weight_n = gate_weight_ref.shape[1]   # K // 2
    s1_stride_scale_n = gate_scale_ref.shape[1]     # K // 32

    # Stage 2 (down): weight is [K_hidden, N//2], scale is [K_hidden, N//32]
    s2_stride_weight_n = down_weight_ref.shape[1]   # N // 2
    s2_stride_scale_n = down_scale_ref.shape[1]     # N // 32

    # Step 2: WGMMA Stage 1 (gate + up + SwiGLU)
    intermediate = fused_mxfp4_grouped_stage1(
        dispatched_x, expert_offsets,
        gate_ptrs, gate_scale_ptrs,
        up_ptrs, up_scale_ptrs,
        N_intermediate, s1_stride_weight_n, s1_stride_scale_n,
        gate_bias_ptrs=gate_bias_ptrs,
        up_bias_ptrs=up_bias_ptrs,
    )

    # Step 3: WGMMA Stage 2 (down projection)
    sorted_output = fused_mxfp4_grouped_stage2(
        intermediate, expert_offsets,
        down_ptrs, down_scale_ptrs,
        hidden_size, s2_stride_weight_n, s2_stride_scale_n,
        down_bias_ptrs=down_bias_ptrs,
    )

    # Step 4: Reduce (weighted scatter-add back to original order)
    output = reduce_weighted_scatter_cuda(
        sorted_output, topk_pos, topk_weights,
        num_tokens, hidden_size, K_topk,
        output=output,
    )

    return output
