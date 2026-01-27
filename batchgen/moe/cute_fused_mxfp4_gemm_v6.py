"""CuTe-style fused MXFP4 grouped GEMM V6 - Simple CTAs, No Double Buffering.

V6 philosophy: Let the hardware do the work!
- Launch enough CTAs to saturate the GPU
- Each CTA is simple: load tile → compute → store
- No manual double-buffering or ping-pong complexity
- Rely on the GPU scheduler to overlap different CTAs

Key insight: Modern GPUs can run many CTAs concurrently. Instead of complex
software pipelining within a CTA, we let different CTAs naturally overlap
their memory accesses with each other's compute.

Architecture:
- Grid: (num_routed × num_m_blocks, num_n_blocks)  -- more CTAs!
- Each CTA handles ONE (expert, m_block, n_block) triplet
- Simple K-loop with ONE sync per iteration
- No double-buffered smem (smaller footprint = more occupancy)

Shared memory layout (~20KB):
- LUT: 64 bytes (16 floats)
- smem_lhs: 9216 bytes [64 × 72] BF16
- smem_rhs: 9216 bytes [64 × 72] BF16
- smem_acc_temp: 8192 bytes [8 × 16 × 16] FP32
Total: ~27KB per CTA

Target: With 144KB smem/SM on H20, we can fit 5 CTAs/SM → better occupancy!
"""

import torch
from torch.utils.cpp_extension import load_inline
import os

CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>

using namespace nvcuda;

// FP4 E2M1 LUT
__constant__ float FP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Configuration
#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 64
#define WARP_SIZE 32
#define NUM_WARPS 8
#define THREADS_PER_BLOCK (NUM_WARPS * WARP_SIZE)  // 256

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define TILES_K (BLOCK_K / WMMA_K)  // 4

// Single-buffered smem (no double buffering!)
#define LHS_STRIDE (BLOCK_K + 8)  // 72
#define RHS_STRIDE (BLOCK_K + 8)  // 72
#define SMEM_LHS_SIZE (BLOCK_M * LHS_STRIDE)  // 4608
#define SMEM_RHS_SIZE (BLOCK_N * RHS_STRIDE)  // 4608

// ============================================================================
// Simple LHS load (no double buffering)
// ============================================================================
__device__ __forceinline__ void load_lhs_simple(
    __nv_bfloat16* smem_lhs,
    const __nv_bfloat16* hidden_states,
    int token_start, int m_start, int actual_m, int k_start, int K, int tid
) {
    const int elements_per_thread = (BLOCK_M * BLOCK_K) / THREADS_PER_BLOCK;  // 16

    #pragma unroll
    for (int i = 0; i < elements_per_thread; i++) {
        const int elem_idx = tid * elements_per_thread + i;
        const int m_local = elem_idx / BLOCK_K;
        const int k_local = elem_idx % BLOCK_K;
        const int m_global = token_start + m_start + m_local;
        const int k_global = k_start + k_local;

        __nv_bfloat16 val = (m_local < actual_m && k_global < K)
            ? hidden_states[m_global * K + k_global]
            : __float2bfloat16(0.0f);
        smem_lhs[m_local * LHS_STRIDE + k_local] = val;
    }
}

// ============================================================================
// Fused load+decode RHS (same as V5, no double buffering)
// ============================================================================
__device__ __forceinline__ void load_decode_rhs_simple(
    __nv_bfloat16* smem_rhs,
    const float* lut,
    const uint8_t* weight_base,
    const uint8_t* scale_base,
    int n_start, int k_start, int N, int K,
    int64_t stride_weight_n, int64_t stride_weight_k,
    int64_t stride_scale_n, int64_t stride_scale_k,
    int tid
) {
    const int values_per_thread = (BLOCK_N * BLOCK_K) / THREADS_PER_BLOCK;  // 16
    const int start_idx = tid * values_per_thread;
    const int n_local = start_idx / BLOCK_K;
    const int k_local_start = start_idx % BLOCK_K;
    const int n_global = n_start + n_local;

    if (n_global >= N) {
        #pragma unroll
        for (int i = 0; i < values_per_thread; i++) {
            smem_rhs[n_local * RHS_STRIDE + k_local_start + i] = __float2bfloat16(0.0f);
        }
        return;
    }

    // Load packed FP4 directly from global
    const int k_packed_start = k_start / 2 + k_local_start / 2;
    const int64_t packed_offset = n_global * stride_weight_n + k_packed_start * stride_weight_k;

    uint2 packed_vec;
    if (stride_weight_k == 1) {
        packed_vec = *reinterpret_cast<const uint2*>(weight_base + packed_offset);
    } else {
        uint8_t* bytes = reinterpret_cast<uint8_t*>(&packed_vec);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            bytes[i] = weight_base[packed_offset + i * stride_weight_k];
        }
    }

    // Load scales
    const int k_block_idx = k_start / 32;
    int scale_lo = static_cast<int>(scale_base[n_global * stride_scale_n + k_block_idx * stride_scale_k]) - 127;
    int scale_hi = static_cast<int>(scale_base[n_global * stride_scale_n + (k_block_idx + 1) * stride_scale_k]) - 127;

    // Decode in registers
    uint8_t* bytes = reinterpret_cast<uint8_t*>(&packed_vec);
    __nv_bfloat16 bf16_vals[16];

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const uint8_t byte = bytes[i];
        const int k_pos = k_local_start + i * 2;
        const int scale = (k_pos < 32) ? scale_lo : scale_hi;

        float v_lo = ldexpf(lut[byte & 0x0F], scale);
        float v_hi = ldexpf(lut[(byte >> 4) & 0x0F], scale);

        bf16_vals[i * 2] = __float2bfloat16(v_lo);
        bf16_vals[i * 2 + 1] = __float2bfloat16(v_hi);
    }

    // Vectorized write to smem
    uint4* out_ptr = reinterpret_cast<uint4*>(&smem_rhs[n_local * RHS_STRIDE + k_local_start]);
    uint4* val_ptr = reinterpret_cast<uint4*>(bf16_vals);
    out_ptr[0] = val_ptr[0];
    out_ptr[1] = val_ptr[1];
}

// ============================================================================
// V6 Kernel: Simple CTAs, No Double Buffering
// ============================================================================
__global__ void cute_fused_mxfp4_gemm_v6_kernel(
    const __nv_bfloat16* __restrict__ hidden_states,
    const int* __restrict__ expert_ids,
    const int* __restrict__ token_offsets,
    const int* __restrict__ m_block_to_expert,  // Maps m_block_idx to expert
    const int* __restrict__ m_block_to_local,   // Maps m_block_idx to local m_block
    const int64_t* __restrict__ weight_ptrs,
    const int64_t* __restrict__ scale_ptrs,
    __nv_bfloat16* __restrict__ output,
    const int total_m_blocks,
    const int N, const int K,
    const int64_t stride_weight_n, const int64_t stride_weight_k,
    const int64_t stride_scale_n, const int64_t stride_scale_k
) {
    // =========================================================================
    // Single-buffered shared memory (~27KB)
    // =========================================================================
    extern __shared__ char smem[];

    float* lut = reinterpret_cast<float*>(smem);
    __nv_bfloat16* smem_lhs = reinterpret_cast<__nv_bfloat16*>(smem + 64);
    __nv_bfloat16* smem_rhs = reinterpret_cast<__nv_bfloat16*>(smem + 64 + SMEM_LHS_SIZE * sizeof(__nv_bfloat16));
    float* smem_acc_temp = reinterpret_cast<float*>(smem + 64 + (SMEM_LHS_SIZE + SMEM_RHS_SIZE) * sizeof(__nv_bfloat16));

    // Load LUT
    if (threadIdx.x < 16) {
        lut[threadIdx.x] = FP4_LUT[threadIdx.x];
    }

    // Block indices - each CTA handles ONE (expert, m_block, n_block)
    const int m_block_idx = blockIdx.x;  // Combined expert + m_block
    const int n_block = blockIdx.y;

    if (m_block_idx >= total_m_blocks) return;

    // Decode which expert and local m_block this CTA handles
    const int routed_idx = m_block_to_expert[m_block_idx];
    const int local_m_block = m_block_to_local[m_block_idx];

    const int expert_id = expert_ids[routed_idx];
    const int token_start = token_offsets[routed_idx];
    const int token_end = token_offsets[routed_idx + 1];
    const int num_tokens = token_end - token_start;

    if (num_tokens == 0) return;

    const int m_start = local_m_block * BLOCK_M;
    const int actual_m = min(BLOCK_M, num_tokens - m_start);
    if (actual_m <= 0) return;

    const uint8_t* weight_base = reinterpret_cast<const uint8_t*>(weight_ptrs[expert_id]);
    const uint8_t* scale_base = reinterpret_cast<const uint8_t*>(scale_ptrs[expert_id]);

    const int n_start = n_block * BLOCK_N;
    if (n_start >= N) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warp_m = (warp_id / 2) % 4;
    const int warp_n = (warp_id % 2) * 2;

    // Initialize accumulators
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag[2];
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        wmma::fill_fragment(acc_frag[i], 0.0f);
    }

    const int num_k_blocks = K / BLOCK_K;

    // =========================================================================
    // SIMPLE K-LOOP: load → sync → compute → sync
    // No double buffering! Let different CTAs overlap naturally.
    // =========================================================================
    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        const int k_start = k_block * BLOCK_K;

        // Load tiles
        load_lhs_simple(smem_lhs, hidden_states, token_start, m_start, actual_m, k_start, K, tid);
        load_decode_rhs_simple(smem_rhs, lut, weight_base, scale_base,
                               n_start, k_start, N, K,
                               stride_weight_n, stride_weight_k,
                               stride_scale_n, stride_scale_k, tid);

        __syncthreads();  // Wait for loads

        // WMMA compute
        #pragma unroll
        for (int k_tile = 0; k_tile < TILES_K; k_tile++) {
            const int k_tile_start = k_tile * WMMA_K;

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, &smem_lhs[warp_m * WMMA_M * LHS_STRIDE + k_tile_start], LHS_STRIDE);

            #pragma unroll
            for (int n_tile_offset = 0; n_tile_offset < 2; n_tile_offset++) {
                const int n_tile = warp_n + n_tile_offset;

                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
                wmma::load_matrix_sync(b_frag, &smem_rhs[n_tile * WMMA_N * RHS_STRIDE + k_tile_start], RHS_STRIDE);

                wmma::mma_sync(acc_frag[n_tile_offset], a_frag, b_frag, acc_frag[n_tile_offset]);
            }
        }

        __syncthreads();  // Wait before next iteration overwrites smem
    }

    // =========================================================================
    // Store output (vectorized)
    // =========================================================================
    #pragma unroll
    for (int n_tile_offset = 0; n_tile_offset < 2; n_tile_offset++) {
        const int n_tile = warp_n + n_tile_offset;
        const int m_out_start = m_start + warp_m * WMMA_M;
        const int n_out_start = n_start + n_tile * WMMA_N;

        float* temp_ptr = &smem_acc_temp[warp_id * WMMA_M * WMMA_N];
        wmma::store_matrix_sync(temp_ptr, acc_frag[n_tile_offset], WMMA_N, wmma::mem_row_major);

        #pragma unroll
        for (int chunk = 0; chunk < 2; chunk++) {
            const int base_elem = lane_id * 4 + chunk * 128;
            if (base_elem < WMMA_M * WMMA_N) {
                const int local_row = base_elem / WMMA_N;
                const int local_col = base_elem % WMMA_N;
                const int m_global = token_start + m_out_start + local_row;
                const int n_global = n_out_start + local_col;

                float4 f4 = {temp_ptr[base_elem], temp_ptr[base_elem+1],
                             temp_ptr[base_elem+2], temp_ptr[base_elem+3]};
                __nv_bfloat162 bf2_lo = __floats2bfloat162_rn(f4.x, f4.y);
                __nv_bfloat162 bf2_hi = __floats2bfloat162_rn(f4.z, f4.w);

                if (m_out_start + local_row < actual_m && n_global + 3 < N) {
                    uint2 store_val = {*reinterpret_cast<uint32_t*>(&bf2_lo),
                                       *reinterpret_cast<uint32_t*>(&bf2_hi)};
                    *reinterpret_cast<uint2*>(&output[m_global * N + n_global]) = store_val;
                } else if (m_out_start + local_row < actual_m) {
                    if (n_global < N) output[m_global * N + n_global] = __low2bfloat16(bf2_lo);
                    if (n_global + 1 < N) output[m_global * N + n_global + 1] = __high2bfloat16(bf2_lo);
                    if (n_global + 2 < N) output[m_global * N + n_global + 2] = __low2bfloat16(bf2_hi);
                    if (n_global + 3 < N) output[m_global * N + n_global + 3] = __high2bfloat16(bf2_hi);
                }
            }
        }
    }
}


// ============================================================================
// Launch wrapper
// ============================================================================
void cute_fused_mxfp4_gemm_v6_impl(
    torch::Tensor hidden_states,
    torch::Tensor expert_ids,
    torch::Tensor token_offsets,
    torch::Tensor weight_ptrs,
    torch::Tensor scale_ptrs,
    torch::Tensor output,
    int N,
    int64_t stride_weight_n, int64_t stride_weight_k,
    int64_t stride_scale_n, int64_t stride_scale_k
) {
    const int num_routed = expert_ids.size(0);
    const int K = hidden_states.size(1);

    // Build m_block mapping on CPU
    auto token_offsets_cpu = token_offsets.cpu();
    int* offsets = token_offsets_cpu.data_ptr<int>();

    std::vector<int> m_block_to_expert_vec;
    std::vector<int> m_block_to_local_vec;

    for (int r = 0; r < num_routed; r++) {
        int num_tokens = offsets[r + 1] - offsets[r];
        int num_m_blocks = (num_tokens + BLOCK_M - 1) / BLOCK_M;
        for (int mb = 0; mb < num_m_blocks; mb++) {
            m_block_to_expert_vec.push_back(r);
            m_block_to_local_vec.push_back(mb);
        }
    }

    int total_m_blocks = m_block_to_expert_vec.size();
    if (total_m_blocks == 0) return;

    // Copy mappings to GPU
    auto m_block_to_expert = torch::tensor(m_block_to_expert_vec,
        torch::TensorOptions().dtype(torch::kInt32).device(hidden_states.device()));
    auto m_block_to_local = torch::tensor(m_block_to_local_vec,
        torch::TensorOptions().dtype(torch::kInt32).device(hidden_states.device()));

    // Grid: (total_m_blocks, cdiv(N, BLOCK_N))
    dim3 grid(total_m_blocks, (N + BLOCK_N - 1) / BLOCK_N);
    dim3 block(THREADS_PER_BLOCK);

    // Smem: ~27KB single-buffered
    const size_t smem_size = 64 +
                             SMEM_LHS_SIZE * sizeof(__nv_bfloat16) +
                             SMEM_RHS_SIZE * sizeof(__nv_bfloat16) +
                             NUM_WARPS * WMMA_M * WMMA_N * sizeof(float);

    cute_fused_mxfp4_gemm_v6_kernel<<<grid, block, smem_size>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr()),
        expert_ids.data_ptr<int>(),
        token_offsets.data_ptr<int>(),
        m_block_to_expert.data_ptr<int>(),
        m_block_to_local.data_ptr<int>(),
        weight_ptrs.data_ptr<int64_t>(),
        scale_ptrs.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        total_m_blocks, N, K,
        stride_weight_n, stride_weight_k,
        stride_scale_n, stride_scale_k
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cute_fused_mxfp4_gemm_v6_impl", &cute_fused_mxfp4_gemm_v6_impl,
          "CuTe fused MXFP4 GEMM V6 - Simple CTAs, no double buffering");
}
'''

_cute_gemm_v6_module = None

def _get_cute_gemm_v6_module():
    global _cute_gemm_v6_module
    if _cute_gemm_v6_module is None:
        _cute_gemm_v6_module = load_inline(
            name='cute_fused_mxfp4_gemm_v6',
            cpp_sources=[],
            cuda_sources=[CUDA_SOURCE],
            extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo', '-arch=sm_90a'],
            verbose=os.environ.get('CUDA_DEBUG', '0') == '1',
        )
    return _cute_gemm_v6_module


def cute_routed_mxfp4_gemm_v6(
    hidden_states: torch.Tensor,
    expert_ids: torch.Tensor,
    token_offsets: torch.Tensor,
    weight_ptrs: torch.Tensor,
    scale_ptrs: torch.Tensor,
    N: int,
    weight_ref: torch.Tensor,
    scale_ref: torch.Tensor,
) -> torch.Tensor:
    """V6: Simple CTAs, no double buffering - let hardware overlap CTAs."""
    total_tokens = hidden_states.shape[0]
    device = hidden_states.device

    if expert_ids.dtype != torch.int32:
        expert_ids = expert_ids.to(torch.int32)
    if token_offsets.dtype != torch.int32:
        token_offsets = token_offsets.to(torch.int32)

    output = torch.empty(total_tokens, N, dtype=torch.bfloat16, device=device)

    mod = _get_cute_gemm_v6_module()
    mod.cute_fused_mxfp4_gemm_v6_impl(
        hidden_states, expert_ids, token_offsets,
        weight_ptrs, scale_ptrs, output, N,
        weight_ref.stride(0), weight_ref.stride(1),
        scale_ref.stride(0), scale_ref.stride(1),
    )
    return output


def cute_grouped_mxfp4_gemm_3d_v6(
    hidden_3d: torch.Tensor,
    weight_ptrs: torch.Tensor,
    scale_ptrs: torch.Tensor,
    expert_counts: torch.Tensor,
    N: int,
    weight_ref: torch.Tensor,
    scale_ref: torch.Tensor,
) -> torch.Tensor:
    """V6 with 3D interface."""
    num_experts = hidden_3d.shape[0]
    M_max = hidden_3d.shape[1]
    K = hidden_3d.shape[2]
    device = hidden_3d.device

    expert_counts_cpu = expert_counts.cpu()
    routed_mask = expert_counts_cpu > 0
    routed_indices = torch.where(routed_mask)[0]
    num_routed = len(routed_indices)

    if num_routed == 0:
        return torch.zeros(num_experts, M_max, N, dtype=torch.bfloat16, device=device)

    expert_ids = routed_indices.to(torch.int32).to(device)
    routed_counts = expert_counts_cpu[routed_mask]
    token_offsets = torch.zeros(num_routed + 1, dtype=torch.int32, device=device)
    token_offsets[1:] = torch.cumsum(routed_counts, dim=0).to(torch.int32).to(device)

    total_tokens = token_offsets[-1].item()
    hidden_flat = torch.empty(total_tokens, K, dtype=torch.bfloat16, device=device)

    offset = 0
    for i, expert_idx in enumerate(routed_indices):
        count = expert_counts_cpu[expert_idx].item()
        if count > 0:
            hidden_flat[offset:offset+count] = hidden_3d[expert_idx, :count]
            offset += count

    output_flat = cute_routed_mxfp4_gemm_v6(
        hidden_flat, expert_ids, token_offsets,
        weight_ptrs, scale_ptrs, N, weight_ref, scale_ref
    )

    output_3d = torch.zeros(num_experts, M_max, N, dtype=torch.bfloat16, device=device)
    offset = 0
    for i, expert_idx in enumerate(routed_indices):
        count = expert_counts_cpu[expert_idx].item()
        if count > 0:
            output_3d[expert_idx, :count] = output_flat[offset:offset+count]
            offset += count

    return output_3d
