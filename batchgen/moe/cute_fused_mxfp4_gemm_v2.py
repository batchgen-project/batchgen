"""CuTe-style fused MXFP4 grouped GEMM V2 - Routed Experts Only.

This kernel improves on V1 by:
1. Only processing routed experts (not all 128)
2. Vectorized dequantization (64-bit loads, 128-bit stores like cute_simple)
3. Proper staging: global -> smem_packed -> decode -> smem_rhs -> WMMA

Key optimizations:
- Vectorized 64-bit loads for packed FP4 to shared memory
- Vectorized decode from smem_packed to smem_rhs
- Shared memory LUT for FP4 decode (no ALU)
- Hardware ldexpf() for scale application
- WMMA tensor cores for BF16 GEMM

Target: 15-20ms for 128 experts with 4 tokens each (vs Triton's 27ms)
"""

import torch
from torch.utils.cpp_extension import load_inline
import os

# CuTe-style fused MXFP4 GEMM V2 CUDA source
CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>

using namespace nvcuda;

// ============================================================================
// FP4 E2M1 Lookup Table (16 values)
// ============================================================================
__constant__ float FP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// ============================================================================
// Configuration
// ============================================================================
#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 64
#define WARP_SIZE 32
#define NUM_WARPS 8
#define THREADS_PER_BLOCK (NUM_WARPS * WARP_SIZE)  // 256

// WMMA tile sizes
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Tiles per block dimension
#define TILES_M (BLOCK_M / WMMA_M)  // 4
#define TILES_N (BLOCK_N / WMMA_N)  // 4
#define TILES_K (BLOCK_K / WMMA_K)  // 4

// Shared memory sizes
#define SMEM_LHS_SIZE (BLOCK_M * (BLOCK_K + 8))  // Padding to avoid bank conflicts
#define SMEM_RHS_SIZE (BLOCK_N * (BLOCK_K + 8))
#define SMEM_PACKED_SIZE (BLOCK_N * BLOCK_K / 2)  // [64, 32] packed bytes

// ============================================================================
// Helper Functions
// ============================================================================
__device__ __forceinline__ float hw_ldexp(float x, int exp) {
    return ldexpf(x, exp);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float x) {
    return __float2bfloat16(x);
}

// ============================================================================
// V2 Kernel: Vectorized Dequant + WMMA GEMM for Routed Experts
// ============================================================================
/*
 * Grid: (num_routed_experts, cdiv(N, BLOCK_N))
 * Block: 256 threads
 *
 * This kernel only processes experts that have tokens routed to them.
 * Uses CSR-style indexing with expert_ids and token_offsets.
 *
 * Key improvements over V1:
 * 1. Vectorized 64-bit loads for packed FP4 -> smem_packed
 * 2. Vectorized decode from smem_packed -> smem_rhs
 * 3. Vectorized 128-bit stores for decoded BF16
 */
__global__ void cute_fused_mxfp4_gemm_v2_kernel(
    // Input tokens [total_tokens, K] BF16
    const __nv_bfloat16* __restrict__ hidden_states,
    // Routed expert info
    const int* __restrict__ expert_ids,      // [num_routed] - which experts
    const int* __restrict__ token_offsets,   // [num_routed + 1] - CSR offsets
    // Weight/scale pointer arrays [total_experts] int64
    const int64_t* __restrict__ weight_ptrs,
    const int64_t* __restrict__ scale_ptrs,
    // Output [total_tokens, N] BF16
    __nv_bfloat16* __restrict__ output,
    // Dimensions
    const int total_tokens,
    const int N,
    const int K,
    // Strides for weights [N, K//2]
    const int64_t stride_weight_n,
    const int64_t stride_weight_k,
    // Strides for scales [N, K//32]
    const int64_t stride_scale_n,
    const int64_t stride_scale_k
) {
    // Shared memory layout
    extern __shared__ char smem[];
    float* lut = reinterpret_cast<float*>(smem);  // 16 floats = 64 bytes
    __nv_bfloat16* smem_lhs = reinterpret_cast<__nv_bfloat16*>(smem + 64);  // [64, 72]
    __nv_bfloat16* smem_rhs = reinterpret_cast<__nv_bfloat16*>(smem + 64 + SMEM_LHS_SIZE * sizeof(__nv_bfloat16));  // [64, 72]
    uint8_t* smem_packed = reinterpret_cast<uint8_t*>(smem + 64 + (SMEM_LHS_SIZE + SMEM_RHS_SIZE) * sizeof(__nv_bfloat16));  // [64, 32]
    float* smem_acc_temp = reinterpret_cast<float*>(smem + 64 + (SMEM_LHS_SIZE + SMEM_RHS_SIZE) * sizeof(__nv_bfloat16) + SMEM_PACKED_SIZE);  // For WMMA output

    // Load FP4 LUT to shared memory
    if (threadIdx.x < 16) {
        lut[threadIdx.x] = FP4_LUT[threadIdx.x];
    }

    // Block indices
    const int routed_expert_idx = blockIdx.x;  // Which routed expert (0 to num_routed-1)
    const int n_block = blockIdx.y;            // N-tile index

    // Get actual expert ID and token range
    const int expert_id = expert_ids[routed_expert_idx];
    const int token_start = token_offsets[routed_expert_idx];
    const int token_end = token_offsets[routed_expert_idx + 1];
    const int num_tokens = token_end - token_start;

    // Early exit if no tokens
    if (num_tokens == 0) return;

    // Get weight/scale base pointers for this expert
    const uint8_t* weight_base = reinterpret_cast<const uint8_t*>(weight_ptrs[expert_id]);
    const uint8_t* scale_base = reinterpret_cast<const uint8_t*>(scale_ptrs[expert_id]);

    // N-block range
    const int n_start = n_block * BLOCK_N;
    if (n_start >= N) return;

    // Thread indices
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // Warp tile assignment for WMMA: 8 warps cover 4x4 = 16 tiles (each warp does 2)
    const int warp_m = (warp_id / 2) % 4;
    const int warp_n = (warp_id % 2) * 2;

    // WMMA fragments for accumulation
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag[2];

    // Process M in blocks (handle variable num_tokens)
    const int num_m_blocks = (num_tokens + BLOCK_M - 1) / BLOCK_M;

    for (int m_block = 0; m_block < num_m_blocks; m_block++) {
        const int m_start = m_block * BLOCK_M;
        const int m_end = min(m_start + BLOCK_M, num_tokens);
        const int actual_m = m_end - m_start;

        // Initialize accumulators
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            wmma::fill_fragment(acc_frag[i], 0.0f);
        }

        // K-loop
        const int num_k_blocks = K / BLOCK_K;

        for (int k_block = 0; k_block < num_k_blocks; k_block++) {
            const int k_start = k_block * BLOCK_K;

            __syncthreads();

            // =========================================================
            // PHASE 1: Vectorized load of packed FP4 to smem_packed
            // =========================================================
            // Tile: [BLOCK_N, BLOCK_K/2] = [64, 32] = 2048 bytes
            // 256 threads, each loads 8 bytes using uint2
            {
                const int bytes_per_thread = SMEM_PACKED_SIZE / THREADS_PER_BLOCK;  // 8
                const int byte_offset = tid * bytes_per_thread;
                const int n_local = byte_offset / (BLOCK_K / 2);
                const int k_packed_local = byte_offset % (BLOCK_K / 2);

                const int n_global = n_start + n_local;
                const int k_packed_global = k_start / 2 + k_packed_local;

                if (n_global < N) {
                    const int64_t packed_offset = n_global * stride_weight_n + k_packed_global * stride_weight_k;

                    // Vectorized 64-bit load
                    if (stride_weight_k == 1 && k_packed_local + 8 <= BLOCK_K / 2) {
                        uint2 packed_vec = *reinterpret_cast<const uint2*>(weight_base + packed_offset);
                        *reinterpret_cast<uint2*>(&smem_packed[byte_offset]) = packed_vec;
                    } else {
                        // Fallback to scalar
                        #pragma unroll
                        for (int i = 0; i < bytes_per_thread; i++) {
                            smem_packed[byte_offset + i] = weight_base[packed_offset + i * stride_weight_k];
                        }
                    }
                } else {
                    // Zero padding for out-of-bounds
                    #pragma unroll
                    for (int i = 0; i < bytes_per_thread; i++) {
                        smem_packed[byte_offset + i] = 0;
                    }
                }
            }

            // =========================================================
            // PHASE 2: Vectorized load of LHS to smem_lhs
            // =========================================================
            // Tile: [BLOCK_M, BLOCK_K] = [64, 64] = 8192 bytes (BF16)
            // 256 threads, each loads 16 BF16 (32 bytes)
            {
                const int elements_per_thread = (BLOCK_M * BLOCK_K) / THREADS_PER_BLOCK;  // 16
                const int elem_start = tid * elements_per_thread;

                #pragma unroll
                for (int i = 0; i < elements_per_thread; i++) {
                    const int elem_idx = elem_start + i;
                    const int m_local = elem_idx / BLOCK_K;
                    const int k_local = elem_idx % BLOCK_K;
                    const int m_global = token_start + m_start + m_local;
                    const int k_global = k_start + k_local;

                    __nv_bfloat16 val;
                    if (m_local < actual_m && k_global < K) {
                        val = hidden_states[m_global * K + k_global];
                    } else {
                        val = __float2bfloat16(0.0f);
                    }
                    smem_lhs[m_local * (BLOCK_K + 8) + k_local] = val;
                }
            }

            __syncthreads();

            // =========================================================
            // PHASE 3: Vectorized decode from smem_packed to smem_rhs
            // =========================================================
            // Each thread decodes 16 FP4 values -> 16 BF16 values
            // 256 threads * 16 = 4096 values = [64, 64] tile
            {
                // Thread mapping: process 16 consecutive K values for one N row
                const int values_per_thread = 16;
                const int total_values = BLOCK_N * BLOCK_K;  // 4096
                const int values_per_thread_actual = total_values / THREADS_PER_BLOCK;  // 16

                const int start_idx = tid * values_per_thread_actual;
                const int n_local = start_idx / BLOCK_K;
                const int k_local_start = start_idx % BLOCK_K;

                const int n_global = n_start + n_local;

                // Load scale (2 scales per BLOCK_K=64: one for K[0:31], one for K[32:63])
                int scale_lo = 0, scale_hi = 0;
                if (n_global < N) {
                    const int scale_k_lo = k_block * 2;
                    const int scale_k_hi = k_block * 2 + 1;
                    const int64_t scale_offset_lo = n_global * stride_scale_n + scale_k_lo * stride_scale_k;
                    const int64_t scale_offset_hi = n_global * stride_scale_n + scale_k_hi * stride_scale_k;
                    scale_lo = static_cast<int>(scale_base[scale_offset_lo]) - 127;
                    scale_hi = static_cast<int>(scale_base[scale_offset_hi]) - 127;
                }

                // Read 8 packed bytes from smem (16 FP4 values)
                const int packed_offset = n_local * (BLOCK_K / 2) + k_local_start / 2;
                uint8_t bytes[8];

                // Vectorized read from smem
                *reinterpret_cast<uint2*>(bytes) = *reinterpret_cast<uint2*>(&smem_packed[packed_offset]);

                // Decode and scale
                __nv_bfloat16 bf16_vals[16];
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    const uint8_t byte = bytes[i];
                    const int idx_lo = byte & 0x0F;
                    const int idx_hi = (byte >> 4) & 0x0F;

                    // Determine which scale to use based on K position
                    const int k_pos = k_local_start + i * 2;
                    const int scale = (k_pos < 32) ? scale_lo : scale_hi;

                    // LUT lookup + ldexpf
                    float v_lo = hw_ldexp(lut[idx_lo], scale);
                    float v_hi = hw_ldexp(lut[idx_hi], scale);

                    bf16_vals[i * 2] = float_to_bf16(v_lo);
                    bf16_vals[i * 2 + 1] = float_to_bf16(v_hi);
                }

                // Vectorized write to smem_rhs (2x uint4 = 32 bytes = 16 BF16)
                const int rhs_offset = n_local * (BLOCK_K + 8) + k_local_start;
                uint4* out_ptr = reinterpret_cast<uint4*>(&smem_rhs[rhs_offset]);
                uint4* val_ptr = reinterpret_cast<uint4*>(bf16_vals);
                out_ptr[0] = val_ptr[0];
                out_ptr[1] = val_ptr[1];
            }

            __syncthreads();

            // =========================================================
            // PHASE 4: WMMA Tensor Core GEMM
            // =========================================================
            #pragma unroll
            for (int k_tile = 0; k_tile < TILES_K; k_tile++) {
                const int k_tile_start = k_tile * WMMA_K;

                // Load LHS fragment
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::load_matrix_sync(a_frag, &smem_lhs[warp_m * WMMA_M * (BLOCK_K + 8) + k_tile_start], BLOCK_K + 8);

                // Process 2 N-tiles per warp
                #pragma unroll
                for (int n_tile_offset = 0; n_tile_offset < 2; n_tile_offset++) {
                    const int n_tile = warp_n + n_tile_offset;
                    const int n_tile_start = n_tile * WMMA_N;

                    // Load RHS fragment (col_major for transpose)
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
                    wmma::load_matrix_sync(b_frag, &smem_rhs[n_tile_start * (BLOCK_K + 8) + k_tile_start], BLOCK_K + 8);

                    // Matrix multiply accumulate
                    wmma::mma_sync(acc_frag[n_tile_offset], a_frag, b_frag, acc_frag[n_tile_offset]);
                }
            }
        }  // K-loop

        // =========================================================
        // Store output
        // =========================================================
        __syncthreads();

        #pragma unroll
        for (int n_tile_offset = 0; n_tile_offset < 2; n_tile_offset++) {
            const int n_tile = warp_n + n_tile_offset;
            const int m_out_start = m_start + warp_m * WMMA_M;
            const int n_out_start = n_start + n_tile * WMMA_N;

            // Store to temp smem buffer
            float* temp_ptr = &smem_acc_temp[warp_id * WMMA_M * WMMA_N];
            wmma::store_matrix_sync(temp_ptr, acc_frag[n_tile_offset], WMMA_N, wmma::mem_row_major);
            __syncwarp();

            // Convert and store to global memory
            const int elements_per_lane = (WMMA_M * WMMA_N) / WARP_SIZE;  // 8
            #pragma unroll
            for (int i = 0; i < elements_per_lane; i++) {
                const int elem_idx = lane_id + i * WARP_SIZE;
                const int row = elem_idx / WMMA_N;
                const int col = elem_idx % WMMA_N;
                const int m_global = token_start + m_out_start + row;
                const int n_global = n_out_start + col;

                if (m_out_start + row < actual_m && n_global < N) {
                    output[m_global * N + n_global] = __float2bfloat16(temp_ptr[elem_idx]);
                }
            }
        }
    }  // M-loop
}


// ============================================================================
// Launch wrapper
// ============================================================================
void cute_fused_mxfp4_gemm_v2_impl(
    torch::Tensor hidden_states,    // [total_tokens, K] BF16
    torch::Tensor expert_ids,       // [num_routed] int32
    torch::Tensor token_offsets,    // [num_routed + 1] int32
    torch::Tensor weight_ptrs,      // [total_experts] int64
    torch::Tensor scale_ptrs,       // [total_experts] int64
    torch::Tensor output,           // [total_tokens, N] BF16
    int N,
    int64_t stride_weight_n,
    int64_t stride_weight_k,
    int64_t stride_scale_n,
    int64_t stride_scale_k
) {
    const int num_routed = expert_ids.size(0);
    const int total_tokens = hidden_states.size(0);
    const int K = hidden_states.size(1);

    // Grid: (num_routed_experts, cdiv(N, BLOCK_N))
    dim3 grid(num_routed, (N + BLOCK_N - 1) / BLOCK_N);
    dim3 block(THREADS_PER_BLOCK);

    // Shared memory size
    const size_t smem_size = 64 +  // LUT
                             SMEM_LHS_SIZE * sizeof(__nv_bfloat16) +
                             SMEM_RHS_SIZE * sizeof(__nv_bfloat16) +
                             SMEM_PACKED_SIZE +
                             NUM_WARPS * WMMA_M * WMMA_N * sizeof(float);  // Temp for WMMA output

    cute_fused_mxfp4_gemm_v2_kernel<<<grid, block, smem_size>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr()),
        expert_ids.data_ptr<int>(),
        token_offsets.data_ptr<int>(),
        weight_ptrs.data_ptr<int64_t>(),
        scale_ptrs.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        total_tokens, N, K,
        stride_weight_n, stride_weight_k,
        stride_scale_n, stride_scale_k
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cute_fused_mxfp4_gemm_v2_impl", &cute_fused_mxfp4_gemm_v2_impl,
          "CuTe-style fused MXFP4 GEMM V2 - routed experts only");
}
'''

# Compile and cache
_cute_gemm_v2_module = None


def _get_cute_gemm_v2_module():
    """Lazy-load and compile the CuTe GEMM V2 module."""
    global _cute_gemm_v2_module
    if _cute_gemm_v2_module is None:
        _cute_gemm_v2_module = load_inline(
            name='cute_fused_mxfp4_gemm_v2',
            cpp_sources=[],
            cuda_sources=[CUDA_SOURCE],
            extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo',
                              '-arch=sm_80'],  # Ampere+ for WMMA BF16
            verbose=os.environ.get('CUDA_DEBUG', '0') == '1',
        )
    return _cute_gemm_v2_module


def cute_routed_mxfp4_gemm(
    hidden_states: torch.Tensor,      # [total_tokens, K] BF16
    expert_ids: torch.Tensor,         # [num_routed] int32 - which experts have tokens
    token_offsets: torch.Tensor,      # [num_routed + 1] int32 - CSR offsets
    weight_ptrs: torch.Tensor,        # [total_experts] int64
    scale_ptrs: torch.Tensor,         # [total_experts] int64
    N: int,                           # Output dimension
    weight_ref: torch.Tensor,         # Reference weight for strides [N, K//2]
    scale_ref: torch.Tensor,          # Reference scale for strides [N, K//32]
) -> torch.Tensor:
    """CuTe-style fused MXFP4 GEMM V2 - only processes routed experts.

    This is the optimized version that:
    1. Only launches kernels for experts with tokens
    2. Uses vectorized dequantization (like cute_simple)
    3. Uses WMMA tensor cores for GEMM

    Args:
        hidden_states: Input tokens [total_tokens, K] in BF16
        expert_ids: Which experts have tokens [num_routed] in int32
        token_offsets: CSR-style offsets [num_routed + 1] in int32
        weight_ptrs: Pointer array to all expert weights [total_experts]
        scale_ptrs: Pointer array to all expert scales [total_experts]
        N: Output dimension
        weight_ref: Reference weight tensor for strides
        scale_ref: Reference scale tensor for strides

    Returns:
        output: [total_tokens, N] in BF16

    Example:
        # 4 experts activated with varying tokens
        expert_ids = torch.tensor([3, 17, 45, 99], dtype=torch.int32, device='cuda')
        token_offsets = torch.tensor([0, 2, 5, 7, 8], dtype=torch.int32, device='cuda')
        # Expert 3: tokens 0-1, Expert 17: tokens 2-4, Expert 45: tokens 5-6, Expert 99: token 7
        hidden_states = torch.randn(8, K, dtype=torch.bfloat16, device='cuda')

        output = cute_routed_mxfp4_gemm(
            hidden_states, expert_ids, token_offsets,
            weight_ptrs, scale_ptrs, N, weight_ref, scale_ref
        )
    """
    total_tokens = hidden_states.shape[0]
    device = hidden_states.device

    # Ensure correct dtypes
    if expert_ids.dtype != torch.int32:
        expert_ids = expert_ids.to(torch.int32)
    if token_offsets.dtype != torch.int32:
        token_offsets = token_offsets.to(torch.int32)

    # Allocate output
    output = torch.empty(total_tokens, N, dtype=torch.bfloat16, device=device)

    # Get module and launch
    mod = _get_cute_gemm_v2_module()
    mod.cute_fused_mxfp4_gemm_v2_impl(
        hidden_states,
        expert_ids,
        token_offsets,
        weight_ptrs,
        scale_ptrs,
        output,
        N,
        weight_ref.stride(0),
        weight_ref.stride(1),
        scale_ref.stride(0),
        scale_ref.stride(1),
    )

    return output


def cute_grouped_mxfp4_gemm_3d_v2(
    hidden_3d: torch.Tensor,          # [E, M_max, K] BF16
    weight_ptrs: torch.Tensor,        # [num_experts] int64
    scale_ptrs: torch.Tensor,         # [num_experts] int64
    expert_counts: torch.Tensor,      # [num_experts] int32
    N: int,                           # Output dimension
    weight_ref: torch.Tensor,         # Reference weight for strides [N, K//2]
    scale_ref: torch.Tensor,          # Reference scale for strides [N, K//32]
) -> torch.Tensor:
    """CuTe-style fused MXFP4 GEMM V2 with 3D interface (compatible with V1).

    This wrapper converts the 3D interface to the routed-only API.

    Args:
        hidden_3d: Input tensor [E, M_max, K] in BF16
        weight_ptrs: Pointer array [num_experts] to weight tensors
        scale_ptrs: Pointer array [num_experts] to scale tensors
        expert_counts: Token counts per expert [num_experts]
        N: Output dimension
        weight_ref: Reference weight tensor for strides
        scale_ref: Reference scale tensor for strides

    Returns:
        output_3d: [E, M_max, N] in BF16
    """
    num_experts = hidden_3d.shape[0]
    M_max = hidden_3d.shape[1]
    K = hidden_3d.shape[2]
    device = hidden_3d.device

    # Find routed experts (those with tokens)
    expert_counts_cpu = expert_counts.cpu()
    routed_mask = expert_counts_cpu > 0
    routed_indices = torch.where(routed_mask)[0]
    num_routed = len(routed_indices)

    if num_routed == 0:
        # No tokens routed, return zeros
        return torch.zeros(num_experts, M_max, N, dtype=torch.bfloat16, device=device)

    # Build CSR-style offsets for routed experts
    expert_ids = routed_indices.to(torch.int32).to(device)
    routed_counts = expert_counts_cpu[routed_mask]
    token_offsets = torch.zeros(num_routed + 1, dtype=torch.int32, device=device)
    token_offsets[1:] = torch.cumsum(routed_counts, dim=0).to(torch.int32).to(device)

    # Flatten hidden states for routed experts only
    total_tokens = token_offsets[-1].item()
    hidden_flat = torch.empty(total_tokens, K, dtype=torch.bfloat16, device=device)

    offset = 0
    for i, expert_idx in enumerate(routed_indices):
        count = expert_counts_cpu[expert_idx].item()
        if count > 0:
            hidden_flat[offset:offset+count] = hidden_3d[expert_idx, :count]
            offset += count

    # Call the routed-only kernel
    output_flat = cute_routed_mxfp4_gemm(
        hidden_flat, expert_ids, token_offsets,
        weight_ptrs, scale_ptrs, N, weight_ref, scale_ref
    )

    # Unflatten output back to 3D
    output_3d = torch.zeros(num_experts, M_max, N, dtype=torch.bfloat16, device=device)
    offset = 0
    for i, expert_idx in enumerate(routed_indices):
        count = expert_counts_cpu[expert_idx].item()
        if count > 0:
            output_3d[expert_idx, :count] = output_flat[offset:offset+count]
            offset += count

    return output_3d


if __name__ == "__main__":
    import time

    print("CuTe-Style Fused MXFP4 Grouped GEMM V2")
    print("=" * 60)

    # Compile
    print("Compiling CuTe GEMM V2 kernel...")
    _get_cute_gemm_v2_module()
    print("Compilation successful!")

    # Test parameters (GPT-OSS-120B, simulating production with few routed experts)
    total_experts = 128
    num_routed = 4  # Only 4 experts activated (typical for batch=1)
    tokens_per_expert = 2
    total_tokens = num_routed * tokens_per_expert
    N = 13824  # Intermediate size
    K = 5120   # Hidden size
    device = "cuda"

    print(f"\nConfig: {total_experts} total experts, {num_routed} routed, "
          f"{tokens_per_expert} tokens/expert, N={N}, K={K}")

    # Create test data
    hidden_states = torch.randn(total_tokens, K, dtype=torch.bfloat16, device=device)

    # Create expert weights and scales for all experts
    weights = [torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)
               for _ in range(total_experts)]
    scales = [torch.randint(120, 134, (N, K // 32), dtype=torch.uint8, device=device)
              for _ in range(total_experts)]

    # Create pointer arrays
    weight_ptrs = torch.tensor([w.data_ptr() for w in weights],
                               dtype=torch.int64, device=device)
    scale_ptrs = torch.tensor([s.data_ptr() for s in scales],
                              dtype=torch.int64, device=device)

    # Routed expert info (simulate 4 experts with 2 tokens each)
    expert_ids = torch.tensor([3, 17, 45, 99], dtype=torch.int32, device=device)
    token_offsets = torch.tensor([0, 2, 4, 6, 8], dtype=torch.int32, device=device)

    # Benchmark
    warmup_iters = 5
    bench_iters = 20

    print("\nBenchmarking routed-only API...")

    # Warmup
    for _ in range(warmup_iters):
        output = cute_routed_mxfp4_gemm(
            hidden_states, expert_ids, token_offsets,
            weight_ptrs, scale_ptrs, N, weights[0], scales[0]
        )
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(bench_iters):
        output = cute_routed_mxfp4_gemm(
            hidden_states, expert_ids, token_offsets,
            weight_ptrs, scale_ptrs, N, weights[0], scales[0]
        )
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) / bench_iters * 1000

    print(f"\nV2 kernel (routed-only):")
    print(f"  Routed experts: {num_routed} (of {total_experts})")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Time: {elapsed_ms:.3f} ms")
    print(f"  Output shape: {output.shape}")

    # Also test 3D interface for comparison
    print("\n" + "=" * 60)
    print("Testing 3D interface (for comparison with V1)...")

    hidden_3d = torch.randn(total_experts, tokens_per_expert, K,
                            dtype=torch.bfloat16, device=device)
    expert_counts = torch.zeros(total_experts, dtype=torch.int32, device=device)
    expert_counts[3] = tokens_per_expert
    expert_counts[17] = tokens_per_expert
    expert_counts[45] = tokens_per_expert
    expert_counts[99] = tokens_per_expert

    # Warmup
    for _ in range(warmup_iters):
        output_3d = cute_grouped_mxfp4_gemm_3d_v2(
            hidden_3d, weight_ptrs, scale_ptrs, expert_counts,
            N, weights[0], scales[0]
        )
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(bench_iters):
        output_3d = cute_grouped_mxfp4_gemm_3d_v2(
            hidden_3d, weight_ptrs, scale_ptrs, expert_counts,
            N, weights[0], scales[0]
        )
    torch.cuda.synchronize()
    elapsed_ms_3d = (time.perf_counter() - start) / bench_iters * 1000

    print(f"\nV2 kernel (3D interface):")
    print(f"  Time: {elapsed_ms_3d:.3f} ms")
    print(f"  Output shape: {output_3d.shape}")
