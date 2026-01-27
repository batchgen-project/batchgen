"""CuTe-style fused MXFP4 grouped GEMM V7 - Warp Specialization + Persistent CTA.

V7 philosophy: Software pipelining with warp specialization
- Persistent CTAs that loop over multiple work items
- Warp specialization: producer warps (load) vs consumer warps (compute)
- True overlap: producers load k+1 while consumers compute k
- Barrier-based synchronization between warp groups

Architecture:
- Grid: (num_SMs, 1) -- persistent, each SM processes multiple tiles
- 8 warps total: 2 producer warps + 6 consumer warps
- Producers: Load LHS and RHS tiles to double-buffered smem
- Consumers: WMMA compute using data prepared by producers
- Named barriers for producer-consumer synchronization

Warp roles:
- Warp 0-1: PRODUCERS - load LHS and decode RHS
- Warp 2-7: CONSUMERS - WMMA tensor core compute

Pipeline:
1. Producers load k=0 to buffer[0]
2. Barrier: Wait for producers
3. Loop k=0 to K-1:
   - Producers: Load k+1 to buffer[1-ping_pong] (if not last)
   - Consumers: Compute using buffer[ping_pong]
   - Barrier: Wait for both to finish
   - Swap buffers
4. Store output

This achieves true load/compute overlap through warp specialization.
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

// Warp specialization: 2 producers + 6 consumers
#define NUM_PRODUCER_WARPS 2
#define NUM_CONSUMER_WARPS 6
#define PRODUCER_THREADS (NUM_PRODUCER_WARPS * WARP_SIZE)  // 64

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define TILES_K (BLOCK_K / WMMA_K)  // 4

#define LHS_STRIDE (BLOCK_K + 8)  // 72
#define RHS_STRIDE (BLOCK_K + 8)  // 72
#define SMEM_LHS_SIZE (BLOCK_M * LHS_STRIDE)  // 4608
#define SMEM_RHS_SIZE (BLOCK_N * RHS_STRIDE)  // 4608

// Barrier IDs
#define BAR_PRODUCER_DONE 0
#define BAR_CONSUMER_DONE 1

// ============================================================================
// Producer functions (warps 0-1)
// ============================================================================
__device__ __forceinline__ void producer_load_lhs(
    __nv_bfloat16* smem_lhs,
    const __nv_bfloat16* hidden_states,
    int token_start, int m_start, int actual_m, int k_start, int K,
    int producer_tid  // 0-63
) {
    // 64 producer threads load 64*64 = 4096 elements, each loads 64 elements
    const int elements_per_thread = (BLOCK_M * BLOCK_K) / PRODUCER_THREADS;  // 64

    #pragma unroll 4
    for (int i = 0; i < elements_per_thread; i++) {
        const int elem_idx = producer_tid * elements_per_thread + i;
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

__device__ __forceinline__ void producer_load_decode_rhs(
    __nv_bfloat16* smem_rhs,
    const float* lut,
    const uint8_t* weight_base,
    const uint8_t* scale_base,
    int n_start, int k_start, int N, int K,
    int64_t stride_weight_n, int64_t stride_weight_k,
    int64_t stride_scale_n, int64_t stride_scale_k,
    int producer_tid  // 0-63
) {
    // 64 producer threads process 64*64 = 4096 values, each processes 64 values
    const int values_per_thread = (BLOCK_N * BLOCK_K) / PRODUCER_THREADS;  // 64
    const int start_idx = producer_tid * values_per_thread;

    // Process in chunks of 16 (like other versions)
    for (int chunk = 0; chunk < values_per_thread; chunk += 16) {
        const int idx = start_idx + chunk;
        const int n_local = idx / BLOCK_K;
        const int k_local_start = idx % BLOCK_K;
        const int n_global = n_start + n_local;

        if (n_global >= N) {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                smem_rhs[n_local * RHS_STRIDE + k_local_start + i] = __float2bfloat16(0.0f);
            }
            continue;
        }

        // Load packed FP4
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
        const int actual_k_pos = k_local_start;
        const int scale_idx = k_block_idx + (actual_k_pos >= 32 ? 1 : 0);
        int scale_lo = static_cast<int>(scale_base[n_global * stride_scale_n + k_block_idx * stride_scale_k]) - 127;
        int scale_hi = static_cast<int>(scale_base[n_global * stride_scale_n + (k_block_idx + 1) * stride_scale_k]) - 127;

        // Decode
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

        // Store
        uint4* out_ptr = reinterpret_cast<uint4*>(&smem_rhs[n_local * RHS_STRIDE + k_local_start]);
        uint4* val_ptr = reinterpret_cast<uint4*>(bf16_vals);
        out_ptr[0] = val_ptr[0];
        out_ptr[1] = val_ptr[1];
    }
}

// ============================================================================
// V7 Kernel: Warp Specialization + Persistent CTA
// ============================================================================
__global__ void cute_fused_mxfp4_gemm_v7_kernel(
    const __nv_bfloat16* __restrict__ hidden_states,
    const int* __restrict__ expert_ids,
    const int* __restrict__ token_offsets,
    const int* __restrict__ work_tiles,  // [num_tiles, 3]: (routed_idx, m_block, n_block)
    const int64_t* __restrict__ weight_ptrs,
    const int64_t* __restrict__ scale_ptrs,
    __nv_bfloat16* __restrict__ output,
    const int num_tiles,
    const int N, const int K,
    const int64_t stride_weight_n, const int64_t stride_weight_k,
    const int64_t stride_scale_n, const int64_t stride_scale_k
) {
    // =========================================================================
    // Double-buffered shared memory
    // =========================================================================
    extern __shared__ char smem[];

    float* lut = reinterpret_cast<float*>(smem);
    __nv_bfloat16* smem_lhs_0 = reinterpret_cast<__nv_bfloat16*>(smem + 64);
    __nv_bfloat16* smem_lhs_1 = reinterpret_cast<__nv_bfloat16*>(smem + 64 + SMEM_LHS_SIZE * sizeof(__nv_bfloat16));
    __nv_bfloat16* smem_rhs_0 = reinterpret_cast<__nv_bfloat16*>(smem + 64 + 2 * SMEM_LHS_SIZE * sizeof(__nv_bfloat16));
    __nv_bfloat16* smem_rhs_1 = reinterpret_cast<__nv_bfloat16*>(smem + 64 + 2 * SMEM_LHS_SIZE * sizeof(__nv_bfloat16) + SMEM_RHS_SIZE * sizeof(__nv_bfloat16));
    float* smem_acc_temp = reinterpret_cast<float*>(smem + 64 + 2 * (SMEM_LHS_SIZE + SMEM_RHS_SIZE) * sizeof(__nv_bfloat16));

    __nv_bfloat16* smem_lhs[2] = {smem_lhs_0, smem_lhs_1};
    __nv_bfloat16* smem_rhs[2] = {smem_rhs_0, smem_rhs_1};

    // Load LUT
    if (threadIdx.x < 16) {
        lut[threadIdx.x] = FP4_LUT[threadIdx.x];
    }

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // Warp roles
    const bool is_producer = (warp_id < NUM_PRODUCER_WARPS);
    const int producer_tid = tid;  // 0-63 for producers
    const int consumer_warp_id = warp_id - NUM_PRODUCER_WARPS;  // 0-5 for consumers

    // Consumer warp tile assignment (6 consumer warps for 64x64 output)
    // Each consumer warp handles a 16x32 region (2 adjacent 16x16 WMMA tiles)
    const int consumer_warp_m = consumer_warp_id / 2;  // 0, 0, 1, 1, 2, 2
    const int consumer_warp_n = (consumer_warp_id % 2) * 2;  // 0, 2, 0, 2, 0, 2

    // Consumer accumulators
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag[2];

    // =========================================================================
    // PERSISTENT CTA: Loop over assigned tiles
    // =========================================================================
    for (int tile_idx = blockIdx.x; tile_idx < num_tiles; tile_idx += gridDim.x) {
        // Decode work tile
        const int routed_idx = work_tiles[tile_idx * 3 + 0];
        const int local_m_block = work_tiles[tile_idx * 3 + 1];
        const int n_block = work_tiles[tile_idx * 3 + 2];

        const int expert_id = expert_ids[routed_idx];
        const int token_start = token_offsets[routed_idx];
        const int token_end = token_offsets[routed_idx + 1];
        const int num_tokens = token_end - token_start;

        if (num_tokens == 0) continue;

        const int m_start = local_m_block * BLOCK_M;
        const int actual_m = min(BLOCK_M, num_tokens - m_start);
        if (actual_m <= 0) continue;

        const uint8_t* weight_base = reinterpret_cast<const uint8_t*>(weight_ptrs[expert_id]);
        const uint8_t* scale_base = reinterpret_cast<const uint8_t*>(scale_ptrs[expert_id]);

        const int n_start = n_block * BLOCK_N;
        if (n_start >= N) continue;

        // Initialize consumer accumulators
        if (!is_producer) {
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                wmma::fill_fragment(acc_frag[i], 0.0f);
            }
        }

        const int num_k_blocks = K / BLOCK_K;
        int ping_pong = 0;

        // =====================================================================
        // PREFETCH k=0 (producers only)
        // =====================================================================
        if (is_producer) {
            producer_load_lhs(smem_lhs[0], hidden_states, token_start, m_start, actual_m, 0, K, producer_tid);
            producer_load_decode_rhs(smem_rhs[0], lut, weight_base, scale_base,
                                     n_start, 0, N, K,
                                     stride_weight_n, stride_weight_k,
                                     stride_scale_n, stride_scale_k, producer_tid);
        }
        __syncthreads();  // All threads wait for prefetch

        // =====================================================================
        // MAIN K-LOOP with warp specialization
        // =====================================================================
        for (int k_block = 0; k_block < num_k_blocks; k_block++) {
            const int next_buf = 1 - ping_pong;

            // PRODUCERS: Load k+1 (while consumers compute k)
            if (is_producer && k_block < num_k_blocks - 1) {
                const int k_start_next = (k_block + 1) * BLOCK_K;
                producer_load_lhs(smem_lhs[next_buf], hidden_states, token_start, m_start,
                                  actual_m, k_start_next, K, producer_tid);
                producer_load_decode_rhs(smem_rhs[next_buf], lut, weight_base, scale_base,
                                         n_start, k_start_next, N, K,
                                         stride_weight_n, stride_weight_k,
                                         stride_scale_n, stride_scale_k, producer_tid);
            }

            // CONSUMERS: Compute using current buffer (while producers load next)
            if (!is_producer) {
                #pragma unroll
                for (int k_tile = 0; k_tile < TILES_K; k_tile++) {
                    const int k_tile_start = k_tile * WMMA_K;

                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
                    wmma::load_matrix_sync(a_frag, &smem_lhs[ping_pong][consumer_warp_m * WMMA_M * LHS_STRIDE + k_tile_start], LHS_STRIDE);

                    #pragma unroll
                    for (int n_tile_offset = 0; n_tile_offset < 2; n_tile_offset++) {
                        const int n_tile = consumer_warp_n + n_tile_offset;

                        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
                        wmma::load_matrix_sync(b_frag, &smem_rhs[ping_pong][n_tile * WMMA_N * RHS_STRIDE + k_tile_start], RHS_STRIDE);

                        wmma::mma_sync(acc_frag[n_tile_offset], a_frag, b_frag, acc_frag[n_tile_offset]);
                    }
                }
            }

            // Sync: Wait for both producers and consumers
            __syncthreads();

            ping_pong = next_buf;
        }

        // =====================================================================
        // STORE OUTPUT (consumers only)
        // =====================================================================
        if (!is_producer) {
            #pragma unroll
            for (int n_tile_offset = 0; n_tile_offset < 2; n_tile_offset++) {
                const int n_tile = consumer_warp_n + n_tile_offset;
                const int m_out_start = m_start + consumer_warp_m * WMMA_M;
                const int n_out_start = n_start + n_tile * WMMA_N;

                float* temp_ptr = &smem_acc_temp[consumer_warp_id * WMMA_M * WMMA_N];
                wmma::store_matrix_sync(temp_ptr, acc_frag[n_tile_offset], WMMA_N, wmma::mem_row_major);

                __syncwarp();

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

        __syncthreads();  // Sync before next tile
    }
}


// ============================================================================
// Launch wrapper
// ============================================================================
void cute_fused_mxfp4_gemm_v7_impl(
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

    // Build work tiles on CPU
    auto token_offsets_cpu = token_offsets.cpu();
    int* offsets = token_offsets_cpu.data_ptr<int>();

    std::vector<int> work_tiles_vec;

    for (int r = 0; r < num_routed; r++) {
        int num_tokens = offsets[r + 1] - offsets[r];
        int num_m_blocks = (num_tokens + BLOCK_M - 1) / BLOCK_M;
        int num_n_blocks = (N + BLOCK_N - 1) / BLOCK_N;

        for (int mb = 0; mb < num_m_blocks; mb++) {
            for (int nb = 0; nb < num_n_blocks; nb++) {
                work_tiles_vec.push_back(r);   // routed_idx
                work_tiles_vec.push_back(mb);  // m_block
                work_tiles_vec.push_back(nb);  // n_block
            }
        }
    }

    int num_tiles = work_tiles_vec.size() / 3;
    if (num_tiles == 0) return;

    // Copy work tiles to GPU
    auto work_tiles = torch::tensor(work_tiles_vec,
        torch::TensorOptions().dtype(torch::kInt32).device(hidden_states.device()));

    // Get device properties for persistent grid sizing
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // Persistent grid: min(num_tiles, num_SMs * occupancy)
    const size_t smem_size = 64 +
                             2 * SMEM_LHS_SIZE * sizeof(__nv_bfloat16) +
                             2 * SMEM_RHS_SIZE * sizeof(__nv_bfloat16) +
                             NUM_CONSUMER_WARPS * WMMA_M * WMMA_N * sizeof(float);

    int max_blocks_per_sm = prop.sharedMemPerMultiprocessor / smem_size;
    max_blocks_per_sm = min(max_blocks_per_sm, prop.maxBlocksPerMultiProcessor);

    int num_sms = prop.multiProcessorCount;
    int grid_size = min(num_tiles, num_sms * max_blocks_per_sm);

    dim3 grid(grid_size);
    dim3 block(THREADS_PER_BLOCK);

    cute_fused_mxfp4_gemm_v7_kernel<<<grid, block, smem_size>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr()),
        expert_ids.data_ptr<int>(),
        token_offsets.data_ptr<int>(),
        work_tiles.data_ptr<int>(),
        weight_ptrs.data_ptr<int64_t>(),
        scale_ptrs.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        num_tiles, N, K,
        stride_weight_n, stride_weight_k,
        stride_scale_n, stride_scale_k
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cute_fused_mxfp4_gemm_v7_impl", &cute_fused_mxfp4_gemm_v7_impl,
          "CuTe fused MXFP4 GEMM V7 - Warp specialization + persistent CTA");
}
'''

_cute_gemm_v7_module = None

def _get_cute_gemm_v7_module():
    global _cute_gemm_v7_module
    if _cute_gemm_v7_module is None:
        _cute_gemm_v7_module = load_inline(
            name='cute_fused_mxfp4_gemm_v7',
            cpp_sources=[],
            cuda_sources=[CUDA_SOURCE],
            extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo', '-arch=sm_90a'],
            verbose=os.environ.get('CUDA_DEBUG', '0') == '1',
        )
    return _cute_gemm_v7_module


def cute_routed_mxfp4_gemm_v7(
    hidden_states: torch.Tensor,
    expert_ids: torch.Tensor,
    token_offsets: torch.Tensor,
    weight_ptrs: torch.Tensor,
    scale_ptrs: torch.Tensor,
    N: int,
    weight_ref: torch.Tensor,
    scale_ref: torch.Tensor,
) -> torch.Tensor:
    """V7: Warp specialization + persistent CTA - producer/consumer pattern."""
    total_tokens = hidden_states.shape[0]
    device = hidden_states.device

    if expert_ids.dtype != torch.int32:
        expert_ids = expert_ids.to(torch.int32)
    if token_offsets.dtype != torch.int32:
        token_offsets = token_offsets.to(torch.int32)

    output = torch.empty(total_tokens, N, dtype=torch.bfloat16, device=device)

    mod = _get_cute_gemm_v7_module()
    mod.cute_fused_mxfp4_gemm_v7_impl(
        hidden_states, expert_ids, token_offsets,
        weight_ptrs, scale_ptrs, output, N,
        weight_ref.stride(0), weight_ref.stride(1),
        scale_ref.stride(0), scale_ref.stride(1),
    )
    return output


def cute_grouped_mxfp4_gemm_3d_v7(
    hidden_3d: torch.Tensor,
    weight_ptrs: torch.Tensor,
    scale_ptrs: torch.Tensor,
    expert_counts: torch.Tensor,
    N: int,
    weight_ref: torch.Tensor,
    scale_ref: torch.Tensor,
) -> torch.Tensor:
    """V7 with 3D interface."""
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

    output_flat = cute_routed_mxfp4_gemm_v7(
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
