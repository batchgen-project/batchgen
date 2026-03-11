"""
Fused grouped MoE kernels using WGMMA for Kimi K2.5 INT4 W4A16 decode.

Adapted from fused_wgmma_grouped.py (MXFP4 GPT-OSS) with INT4 dequantization:
- INT4 offset decode (nibble - 8) instead of MXFP4 byte-LUT
- BF16 scales (direct load) instead of E8M0 exponent construction
- Standard SiLU activation instead of GPT-OSS SwiGLU (alpha=1.702, clamp ±7)
- No byte-LUT needed in shared memory (saves 1024 bytes)

Two CUDA kernels (TMA for A-matrix, inline INT4 decode for weights):
- Stage 1: gate + up + SiLU (3D grid: experts × N-tiles × M-tiles)
- Stage 2: down projection (3D grid: experts × K-tiles × M-tiles)

Pipeline: dispatch -> WGMMA S1 -> WGMMA S2 -> reduce (4 kernel launches)

INT4 W4A16 Format (compressed-tensors, symmetric):
- 2 INT4 values per byte (low nibble = even position, high nibble = odd position)
- Offset encoding: signed_value = nibble - 8, range [-8, +7]
- Group size 32: one BF16 scale per 32 INT4 values
- Works with both uint8 and int32 packed tensors (same byte layout on little-endian)

Usage:
    from batchgen.moe.fused_int4_wgmma_grouped import (
        fused_int4_grouped_moe_forward_cuda_routing,
        is_int4_grouped_wgmma_available,
    )

    if is_int4_grouped_wgmma_available():
        output = fused_int4_grouped_moe_forward_cuda_routing(
            hidden_states, topk_indices, topk_weights, ...)
"""

import os
import logging

import torch


# Module-level state
_int4_grouped_wgmma_available = None
_int4_grouped_module = None


# ──────────────────────────────────────────────────────────────────────────────
# CUDA Source Code (external .cu file)
# ──────────────────────────────────────────────────────────────────────────────
# Source: batchgen_kernels/src/moe/fused_int4_wgmma_grouped.cu

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


def _load_int4_grouped_module():
    """Load the pre-compiled grouped INT4 WGMMA CUDA module (Stage 1 + Stage 2)."""
    global _int4_grouped_module

    if _int4_grouped_module is not None:
        return _int4_grouped_module

    try:
        import batchgen_kernels
        _int4_grouped_module = batchgen_kernels.load_extension(
            "batchgen_kernels.moe._C_fused_int4_wgmma_grouped"
        )
        logging.info("Loaded pre-compiled WGMMA fused grouped INT4 MoE kernels")
        return _int4_grouped_module
    except Exception as e:
        logging.warning(f"Failed to load WGMMA grouped INT4 MoE kernels: {e}")
        return None


def is_int4_grouped_wgmma_available() -> bool:
    """Check if grouped INT4 WGMMA fused kernels are available."""
    global _int4_grouped_wgmma_available

    if _int4_grouped_wgmma_available is not None:
        return _int4_grouped_wgmma_available

    if not _check_wgmma_support():
        _int4_grouped_wgmma_available = False
        return False

    if os.environ.get("BATCHGEN_DISABLE_WGMMA_INT4_GROUPED", "0") == "1":
        logging.info("WGMMA INT4 grouped kernels disabled by BATCHGEN_DISABLE_WGMMA_INT4_GROUPED")
        _int4_grouped_wgmma_available = False
        return False

    mod = _load_int4_grouped_module()
    _int4_grouped_wgmma_available = mod is not None
    return _int4_grouped_wgmma_available


# ──────────────────────────────────────────────────────────────────────────────
# Low-Level Python Wrappers
# ──────────────────────────────────────────────────────────────────────────────

def fused_int4_grouped_stage1(
    sorted_hidden: torch.Tensor,       # [E*mtp, K] BF16 (3D strided layout)
    tokens_per_expert: torch.Tensor,   # [num_experts] int32
    gate_ptrs: torch.Tensor,           # [num_experts] int64
    gate_scale_ptrs: torch.Tensor,     # [num_experts] int64
    up_ptrs: torch.Tensor,             # [num_experts] int64
    up_scale_ptrs: torch.Tensor,       # [num_experts] int64
    N: int,                            # intermediate_size
    stride_weight_n: int,              # K // 2 (bytes per row in packed weight)
    stride_scale_n: int,               # K // 32 (BF16 elements per row in scale)
    max_tokens_padded: int,            # stride per expert in the 3D buffer
    gate_bias_ptrs: torch.Tensor = None,
    up_bias_ptrs: torch.Tensor = None,
) -> torch.Tensor:
    """Grouped INT4 Stage 1: gate + up + SiLU via WGMMA with TMA.

    Uses 3D strided buffer layout [E, max_tokens_padded, K].
    Each expert's tokens start at expert_idx * max_tokens_padded.
    """
    mod = _load_int4_grouped_module()
    assert mod is not None, "INT4 WGMMA grouped module not available"

    max_m_tiles = (max_tokens_padded + 63) // 64 if max_tokens_padded > 0 else 1

    empty_bias = torch.empty(0, dtype=torch.int64, device=sorted_hidden.device)
    gb = gate_bias_ptrs if gate_bias_ptrs is not None else empty_bias
    ub = up_bias_ptrs if up_bias_ptrs is not None else empty_bias

    return mod.grouped_int4_moe_stage1(
        sorted_hidden, tokens_per_expert,
        gate_ptrs, gate_scale_ptrs,
        up_ptrs, up_scale_ptrs,
        gb, ub,
        N, stride_weight_n, stride_scale_n,
        max_m_tiles, max_tokens_padded,
    )


def fused_int4_grouped_stage2(
    intermediate: torch.Tensor,        # [E*mtp, N] BF16 (3D strided layout)
    tokens_per_expert: torch.Tensor,   # [num_experts] int32
    down_ptrs: torch.Tensor,           # [num_experts] int64
    down_scale_ptrs: torch.Tensor,     # [num_experts] int64
    K: int,                            # hidden_size (output width)
    stride_weight_n: int,              # N // 2 (bytes per row in packed weight)
    stride_scale_n: int,               # N // 32 (BF16 elements per row in scale)
    max_tokens_padded: int,            # stride per expert in the 3D buffer
    down_bias_ptrs: torch.Tensor = None,
) -> torch.Tensor:
    """Grouped INT4 Stage 2: down projection via WGMMA with TMA.

    Uses 3D strided buffer layout [E, max_tokens_padded, N].
    Each expert's tokens start at expert_idx * max_tokens_padded.
    """
    mod = _load_int4_grouped_module()
    assert mod is not None, "INT4 WGMMA grouped module not available"

    empty_bias = torch.empty(0, dtype=torch.int64, device=intermediate.device)
    db = down_bias_ptrs if down_bias_ptrs is not None else empty_bias

    max_m_tiles = (max_tokens_padded + 63) // 64 if max_tokens_padded > 0 else 1

    return mod.grouped_int4_moe_stage2(
        intermediate, tokens_per_expert,
        down_ptrs, down_scale_ptrs,
        db,
        K, stride_weight_n, stride_scale_n,
        max_m_tiles, max_tokens_padded,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CUDA Graph Compatible: In-place Wrappers + TMA Descriptor Creation
# ──────────────────────────────────────────────────────────────────────────────

_BLOCK_M = 64
_BLOCK_K = 64

_empty_bias_cache = {}


def _get_empty_bias(device: torch.device) -> torch.Tensor:
    """Return a cached zero-size int64 tensor for use as empty bias placeholder."""
    key = str(device)
    if key not in _empty_bias_cache:
        _empty_bias_cache[key] = torch.empty(0, dtype=torch.int64, device=device)
    return _empty_bias_cache[key]


def create_tma_descriptor(tensor: torch.Tensor, block_m: int = _BLOCK_M, block_k: int = _BLOCK_K) -> torch.Tensor:
    """Create a TMA descriptor for a 2D BF16 tensor.

    Must be called BEFORE CUDA graph capture.
    """
    mod = _load_int4_grouped_module()
    assert mod is not None, "INT4 WGMMA grouped module not available"
    return mod.create_tma_desc_bf16(tensor, block_m, block_k)


def fused_int4_grouped_stage1_inplace(
    sorted_hidden: torch.Tensor,
    output: torch.Tensor,
    tma_desc: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    gate_ptrs: torch.Tensor,
    gate_scale_ptrs: torch.Tensor,
    up_ptrs: torch.Tensor,
    up_scale_ptrs: torch.Tensor,
    N: int,
    stride_weight_n: int,
    stride_scale_n: int,
    max_tokens_padded: int,
    gate_bias_ptrs: torch.Tensor = None,
    up_bias_ptrs: torch.Tensor = None,
) -> None:
    """In-place Stage 1 with pre-built TMA descriptor and 3D strided layout."""
    mod = _load_int4_grouped_module()
    assert mod is not None, "INT4 WGMMA grouped module not available"

    max_m_tiles = (max_tokens_padded + 63) // 64 if max_tokens_padded > 0 else 1

    gb = gate_bias_ptrs if gate_bias_ptrs is not None else _get_empty_bias(sorted_hidden.device)
    ub = up_bias_ptrs if up_bias_ptrs is not None else _get_empty_bias(sorted_hidden.device)

    mod.grouped_int4_moe_stage1_inplace(
        sorted_hidden, output, tma_desc, tokens_per_expert,
        gate_ptrs, gate_scale_ptrs,
        up_ptrs, up_scale_ptrs,
        gb, ub,
        N, stride_weight_n, stride_scale_n,
        max_m_tiles, max_tokens_padded,
    )


def fused_int4_grouped_stage2_inplace(
    intermediate: torch.Tensor,
    output: torch.Tensor,
    tma_desc: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    down_ptrs: torch.Tensor,
    down_scale_ptrs: torch.Tensor,
    K: int,
    stride_weight_n: int,
    stride_scale_n: int,
    max_tokens_padded: int,
    down_bias_ptrs: torch.Tensor = None,
) -> None:
    """In-place Stage 2 with pre-built TMA descriptor and 3D strided layout."""
    mod = _load_int4_grouped_module()
    assert mod is not None, "INT4 WGMMA grouped module not available"

    max_m_tiles = (max_tokens_padded + 63) // 64 if max_tokens_padded > 0 else 1

    db = down_bias_ptrs if down_bias_ptrs is not None else _get_empty_bias(intermediate.device)

    mod.grouped_int4_moe_stage2_inplace(
        intermediate, output, tma_desc, tokens_per_expert,
        down_ptrs, down_scale_ptrs,
        db,
        K, stride_weight_n, stride_scale_n,
        max_m_tiles, max_tokens_padded,
    )


# ──────────────────────────────────────────────────────────────────────────────
# End-to-End API with CUDA Routing
# ──────────────────────────────────────────────────────────────────────────────

def fused_int4_grouped_moe_forward_cuda_routing(
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
    # Dimensions
    N_intermediate: int,
    hidden_size: int,
    # Strides (in bytes for weight, in BF16 elements for scale)
    s1_stride_weight_n: int,           # K_hidden // 2
    s1_stride_scale_n: int,            # K_hidden // 32
    s2_stride_weight_n: int,           # N_inter // 2
    s2_stride_scale_n: int,            # N_inter // 32
    num_experts: int = 384,
    expert_start: int = 0,
    num_local_experts: int = 384,
    # Bias pointer arrays (None = no biases)
    gate_bias_ptrs: torch.Tensor = None,
    up_bias_ptrs: torch.Tensor = None,
    down_bias_ptrs: torch.Tensor = None,
) -> torch.Tensor:
    """End-to-end grouped INT4 MoE forward using WGMMA + CUDA routing.

    Full pipeline: dispatch -> WGMMA S1 -> WGMMA S2 -> reduce (4 kernel launches).

    Args:
        hidden_states: Input [batch*seq, hidden] BF16
        topk_indices: Expert indices [batch*seq, K] int32
        topk_weights: Routing weights [batch*seq, K] FP32
        gate_ptrs, gate_scale_ptrs: Gate weight/scale pointer arrays
        up_ptrs, up_scale_ptrs: Up weight/scale pointer arrays
        down_ptrs, down_scale_ptrs: Down weight/scale pointer arrays
        N_intermediate: Intermediate dimension (gate/up output width)
        hidden_size: Hidden dimension
        s1_stride_weight_n: Stage 1 weight stride (K_hidden // 2)
        s1_stride_scale_n: Stage 1 scale stride (K_hidden // 32)
        s2_stride_weight_n: Stage 2 weight stride (N_inter // 2)
        s2_stride_scale_n: Stage 2 scale stride (N_inter // 32)
        num_experts: Total number of experts
        expert_start: First local expert index (for EP)
        num_local_experts: Number of local experts

    Returns:
        Output [batch*seq, hidden] BF16
    """
    from batchgen.moe.routing import dispatch_count_gather_cuda, reduce_weighted_scatter_cuda

    num_tokens = hidden_states.shape[0]
    K_topk = topk_indices.shape[1]

    # Step 1: CUDA dispatch
    dispatched_x, expert_counts, expert_offsets, topk_pos = dispatch_count_gather_cuda(
        hidden_states, topk_indices,
        expert_start, num_local_experts,
    )

    # Pad to BLOCK_M for TMA descriptor
    if dispatched_x.shape[0] < _BLOCK_M:
        padded_dx = torch.zeros(
            _BLOCK_M, hidden_size, dtype=dispatched_x.dtype, device=dispatched_x.device
        )
        padded_dx[:dispatched_x.shape[0]] = dispatched_x
        dispatched_x = padded_dx

    # Step 2: WGMMA Stage 1 (gate + up + SiLU)
    intermediate = fused_int4_grouped_stage1(
        dispatched_x, expert_offsets,
        gate_ptrs, gate_scale_ptrs,
        up_ptrs, up_scale_ptrs,
        N_intermediate, s1_stride_weight_n, s1_stride_scale_n,
        gate_bias_ptrs=gate_bias_ptrs,
        up_bias_ptrs=up_bias_ptrs,
    )

    # Step 3: WGMMA Stage 2 (down projection)
    sorted_output = fused_int4_grouped_stage2(
        intermediate, expert_offsets,
        down_ptrs, down_scale_ptrs,
        hidden_size, s2_stride_weight_n, s2_stride_scale_n,
        down_bias_ptrs=down_bias_ptrs,
    )

    # Step 4: Reduce (weighted scatter-add back to original order)
    output = reduce_weighted_scatter_cuda(
        sorted_output, topk_pos, topk_weights,
        num_tokens, hidden_size, K_topk,
    )

    return output
