// Grouped MXFP4 MoE WGMMA kernels — Stage 1 (gate+up+SwiGLU) + Stage 2 (down)
// Extracted from batchgen/moe/fused_wgmma_grouped.py
// SM90+ (Hopper) — WGMMA m64n64k16, TMA, mbarrier

#include <torch/python.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#if CUDA_VERSION >= 13000
// CUDA 13 drops the unversioned alias; the v12000 signature is unchanged.
typedef PFN_cuTensorMapEncodeTiled_v12000 PFN_cuTensorMapEncodeTiled;
#endif
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <utility>

// ============================================================================
// Configuration
// ============================================================================
#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 64
#define WARP_SIZE 32
#define WGMMA_K 16
#define TILES_K (BLOCK_K / WGMMA_K)          // 4
#define WGMMA_NUM_ACCUM 32                    // 64*64/128
#define NUM_STAGES 2
#define TILE_ELEMS (BLOCK_M * BLOCK_K)        // 4096
#define TILE_BYTES (TILE_ELEMS * 2)           // 8192 bytes (BF16)
#define TOTAL_THREADS 256                     // 2 WG
#define PRODUCER_THREADS 128                  // WG0
#define PRODUCER_BAR_ID 1                     // named barrier for producer sync
#define LUT_ENTRIES 256                       // byte-level LUT
#define LUT_BYTES (LUT_ENTRIES * 4)           // 256 * sizeof(__nv_bfloat162) = 1024

// SwiGLU activation parameters
#define SWIGLU_ALPHA 1.702f
#define SWIGLU_LIMIT 7.0f

// FP4 E2M1 values
__constant__ float FP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// ============================================================================
// GmmaDescriptor — 128B swizzle (layout_type=1)
// ============================================================================
union GmmaDescriptor {
    uint64_t desc_;
    struct {
        uint16_t start_address_: 14, : 2;
        uint16_t leading_byte_offset_: 14, : 2;
        uint16_t stride_byte_offset_: 14, : 2;
        uint8_t : 1, base_offset_: 3, : 4;
        uint8_t : 6, layout_type_: 2;
    } bitfield;
};

template <class T>
__device__ __forceinline__ GmmaDescriptor make_smem_desc(T* smem_ptr) {
    GmmaDescriptor desc;
    desc.desc_ = 0;
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    desc.bitfield.start_address_ = addr >> 4;
    desc.bitfield.layout_type_ = 1;
    desc.bitfield.leading_byte_offset_ = 0;
    desc.bitfield.stride_byte_offset_ = 1024 >> 4;
    desc.bitfield.base_offset_ = 0;
    return desc;
}

// ============================================================================
// Warpgroup synchronization primitives
// ============================================================================
__device__ __forceinline__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void warpgroup_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void warpgroup_fence_operand(float& reg) {
    asm volatile("" : "+f"(reg) :: "memory");
}

// ============================================================================
// WGMMA m64n64k16 BF16 SS
// ============================================================================
__device__ __forceinline__ void wgmma_m64n64k16_f32_bf16_bf16_ss(
    uint64_t const& desc_a,
    uint64_t const& desc_b,
    float& d00, float& d01, float& d02, float& d03,
    float& d04, float& d05, float& d06, float& d07,
    float& d08, float& d09, float& d10, float& d11,
    float& d12, float& d13, float& d14, float& d15,
    float& d16, float& d17, float& d18, float& d19,
    float& d20, float& d21, float& d22, float& d23,
    float& d24, float& d25, float& d26, float& d27,
    float& d28, float& d29, float& d30, float& d31,
    int scale_d
) {
    asm volatile(
    "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
      " %8,  %9,  %10, %11, %12, %13, %14, %15, "
      " %16, %17, %18, %19, %20, %21, %22, %23, "
      " %24, %25, %26, %27, %28, %29, %30, %31},"
      " %32,"
      " %33,"
      " p,   1, 1, 0, 0;\n"
    "}\n"
      : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03),
        "+f"(d04), "+f"(d05), "+f"(d06), "+f"(d07),
        "+f"(d08), "+f"(d09), "+f"(d10), "+f"(d11),
        "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15),
        "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19),
        "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23),
        "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27),
        "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31)
      :  "l"(desc_a),
         "l"(desc_b),
         "r"(scale_d));
}

template <size_t... Idx>
__device__ __forceinline__ void wgmma_bf16_ss_impl(
    uint64_t const& desc_a, uint64_t const& desc_b,
    float* d, int scale_d,
    std::index_sequence<Idx...>
) {
    wgmma_m64n64k16_f32_bf16_bf16_ss(desc_a, desc_b, d[Idx]..., scale_d);
}

__device__ __forceinline__ void wgmma_bf16_ss(
    uint64_t const& desc_a, uint64_t const& desc_b,
    float* d, int scale_d
) {
    wgmma_bf16_ss_impl(desc_a, desc_b, d, scale_d,
                        std::make_index_sequence<32>{});
}

// ============================================================================
// reg_to_mn: accumulator register -> (m, n) position
// ============================================================================
__device__ __forceinline__ void reg_to_mn(
    int reg_idx, int warp_in_wg, int lane_id, int& m, int& n
) {
    const int group   = reg_idx / 4;
    const int sub_idx = reg_idx % 4;
    const int m_half  = sub_idx / 2;
    const int n_bit   = sub_idx % 2;
    m = warp_in_wg * 16 + (lane_id / 4) + m_half * 8;
    n = group * 8 + (lane_id % 4) * 2 + n_bit;
}

// ============================================================================
// mbarrier helpers
// ============================================================================
__device__ __forceinline__ void mbarrier_init(uint64_t* mbar, uint32_t count) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :: "r"(smem_addr), "r"(count));
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t* mbar) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "{\n"
        ".reg .b64 state;\n"
        "mbarrier.arrive.shared::cta.b64 state, [%0];\n"
        "}\n"
        :: "r"(smem_addr));
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(
    uint64_t* mbar, uint32_t tx_bytes
) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"(smem_addr), "r"(tx_bytes));
}

__device__ __forceinline__ void mbarrier_wait_parity(
    uint64_t* mbar, uint32_t phase_parity
) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "GROUPED_MXFP4_MBAR_WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
        "@P bra GROUPED_MXFP4_MBAR_DONE_%=;\n"
        "bra GROUPED_MXFP4_MBAR_WAIT_%=;\n"
        "GROUPED_MXFP4_MBAR_DONE_%=:\n"
        "}\n"
        :: "r"(smem_addr), "r"(phase_parity));
}

// ============================================================================
// TMA load 2D
// ============================================================================
__device__ __forceinline__ void tma_load_2d(
    const void* desc, uint64_t* mbar, void* smem_ptr,
    int32_t coord_0, int32_t coord_1
) {
    uint64_t desc_addr = reinterpret_cast<uint64_t>(desc);
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4}], [%2];"
        :: "r"(smem_addr), "l"(desc_addr), "r"(mbar_addr),
           "r"(coord_0), "r"(coord_1)
        : "memory");
}

// ============================================================================
// Named barrier sync (producer WG internal)
// ============================================================================
__device__ __forceinline__ void bar_sync(uint32_t bar_id, uint32_t num_threads) {
    asm volatile("bar.sync %0, %1;" :: "r"(bar_id), "r"(num_threads));
}

// ============================================================================
// Batched Byte-level BF16 LUT decode
// ============================================================================
__device__ __forceinline__ void load_decode_rhs_swizzled_batched(
    __nv_bfloat16* smem_rhs,
    const uint8_t* weight_base,
    const uint8_t* scale_base,
    int n_start, int k_start,
    int N, int K,
    int64_t stride_weight_n,
    int64_t stride_scale_n,
    int tid,
    const __nv_bfloat162* byte_lut
) {
    const int n_local = tid / 2;
    const int k_local_start = (tid & 1) * 32;
    const int n_global = n_start + n_local;

    if (n_global >= N) return;

    const int k_packed_start = k_start / 2 + k_local_start / 2;
    uint4 packed_vec = *reinterpret_cast<const uint4*>(
        weight_base + n_global * stride_weight_n + k_packed_start);
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&packed_vec);

    int raw_scale = static_cast<int>(
        scale_base[n_global * stride_scale_n + (k_start + k_local_start) / 32]);
    int exp_bits = max(1, min(254, raw_scale));
    uint16_t sbits = static_cast<uint16_t>(exp_bits << 7);
    uint32_t spair = (static_cast<uint32_t>(sbits) << 16) | sbits;
    __nv_bfloat162 scale2 = *reinterpret_cast<__nv_bfloat162*>(&spair);

    __nv_bfloat162 raw[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        raw[i] = byte_lut[bytes[i]];
    }

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        raw[i] = __hmul2(raw[i], scale2);
    }

    const int n_mod8 = n_local & 7;
    const int n_base = n_local << 6;

    #pragma unroll
    for (int g = 0; g < 4; g++) {
        const int grp = (k_local_start + g * 8) >> 3;
        const int swz = grp ^ n_mod8;
        const int addr = n_base + (swz << 3);

        uint4 wide;
        wide.x = *reinterpret_cast<const uint32_t*>(&raw[g * 4 + 0]);
        wide.y = *reinterpret_cast<const uint32_t*>(&raw[g * 4 + 1]);
        wide.z = *reinterpret_cast<const uint32_t*>(&raw[g * 4 + 2]);
        wide.w = *reinterpret_cast<const uint32_t*>(&raw[g * 4 + 3]);
        *reinterpret_cast<uint4*>(&smem_rhs[addr]) = wide;
    }
}

// ============================================================================
// Stage 1 Kernel: Gate + Up + SwiGLU (TMA-based, 3D grid, with bias)
//   Grid: (num_experts, num_n_tiles, max_m_tiles)
// ============================================================================
__global__ void __launch_bounds__(TOTAL_THREADS, 1)
grouped_mxfp4_moe_stage1_kernel(
    const __grid_constant__ CUtensorMap desc_a,
    const int64_t* __restrict__ gate_ptrs,
    const int64_t* __restrict__ gate_scale_ptrs,
    const int64_t* __restrict__ up_ptrs,
    const int64_t* __restrict__ up_scale_ptrs,
    const int64_t* __restrict__ gate_bias_ptrs,
    const int64_t* __restrict__ up_bias_ptrs,
    const int32_t* __restrict__ expert_offsets,
    __nv_bfloat16* __restrict__ C,
    int64_t stride_c_m,
    int total_tokens, int N, int K, int num_experts,
    int64_t stride_weight_n,
    int64_t stride_scale_n,
    int has_gate_bias, int has_up_bias
) {
    const int tid = threadIdx.x;
    const int wg_id = tid / 128;
    const int wg_tid = tid % 128;

    const int expert_idx = blockIdx.x;
    const int n_tile = blockIdx.y;
    const int m_tile = blockIdx.z;
    const int n_start = n_tile * BLOCK_N;

    if (n_start >= N) return;

    const int expert_start = expert_offsets[expert_idx];
    const int expert_end = expert_offsets[expert_idx + 1];
    const int expert_tokens = expert_end - expert_start;

    const int num_m_tiles = (expert_tokens + BLOCK_M - 1) / BLOCK_M;
    if (m_tile >= num_m_tiles) return;

    const int m_start = expert_start + m_tile * BLOCK_M;
    const int m_size = min(BLOCK_M, expert_end - m_start);

    const uint8_t* gate_weight = reinterpret_cast<const uint8_t*>(gate_ptrs[expert_idx]);
    const uint8_t* gate_scale = reinterpret_cast<const uint8_t*>(gate_scale_ptrs[expert_idx]);
    const uint8_t* up_weight = reinterpret_cast<const uint8_t*>(up_ptrs[expert_idx]);
    const uint8_t* up_scale = reinterpret_cast<const uint8_t*>(up_scale_ptrs[expert_idx]);

    const __nv_bfloat16* gate_bias = has_gate_bias ?
        reinterpret_cast<const __nv_bfloat16*>(gate_bias_ptrs[expert_idx]) : nullptr;
    const __nv_bfloat16* up_bias = has_up_bias ?
        reinterpret_cast<const __nv_bfloat16*>(up_bias_ptrs[expert_idx]) : nullptr;

    extern __shared__ __align__(128) char smem_buf[];
    __nv_bfloat16* smem_a[NUM_STAGES];
    __nv_bfloat16* smem_b[NUM_STAGES];
    for (int s = 0; s < NUM_STAGES; s++) {
        smem_a[s] = reinterpret_cast<__nv_bfloat16*>(smem_buf + (2*s)     * TILE_BYTES);
        smem_b[s] = reinterpret_cast<__nv_bfloat16*>(smem_buf + (2*s + 1) * TILE_BYTES);
    }
    const int bar_offset = 2 * NUM_STAGES * TILE_BYTES;
    uint64_t* gate_full_barriers  = reinterpret_cast<uint64_t*>(smem_buf + bar_offset);
    uint64_t* gate_empty_barriers = gate_full_barriers + NUM_STAGES;
    uint64_t* up_full_barriers    = gate_empty_barriers + NUM_STAGES;
    uint64_t* up_empty_barriers   = up_full_barriers + NUM_STAGES;
    __nv_bfloat162* byte_lut = reinterpret_cast<__nv_bfloat162*>(
        smem_buf + bar_offset + 4 * NUM_STAGES * sizeof(uint64_t));

    const int num_k_blocks = (K + BLOCK_K - 1) / BLOCK_K;

    // Init barriers with arrival count = 2 (TMA + decode)
    if (tid == 0) {
        for (int s = 0; s < NUM_STAGES; s++) {
            mbarrier_init(&gate_full_barriers[s], 2);
            mbarrier_init(&gate_empty_barriers[s], 1);
            mbarrier_init(&up_full_barriers[s], 2);
            mbarrier_init(&up_empty_barriers[s], 1);
        }
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();

    if (wg_id == 0) {
        // PRODUCER WG: TMA A + decode B

        for (int i = wg_tid; i < LUT_ENTRIES; i += PRODUCER_THREADS) {
            __nv_bfloat16 lo = __float2bfloat16(FP4_LUT[i & 0xF]);
            __nv_bfloat16 hi = __float2bfloat16(FP4_LUT[i >> 4]);
            byte_lut[i] = __halves2bfloat162(lo, hi);
        }
        bar_sync(PRODUCER_BAR_ID, PRODUCER_THREADS);

        if (wg_tid == 0) {
            asm volatile("prefetch.tensormap [%0];" :: "l"(&desc_a) : "memory");
        }

        // GATE K-LOOP
        for (int kb = 0; kb < num_k_blocks; kb++) {
            const int s = kb % NUM_STAGES;
            const int empty_phase = ((kb / NUM_STAGES) + 1) & 1;

            mbarrier_wait_parity(&gate_empty_barriers[s], empty_phase);

            if (wg_tid == 0) {
                mbarrier_arrive_expect_tx(&gate_full_barriers[s], TILE_BYTES);
                tma_load_2d(&desc_a, &gate_full_barriers[s], smem_a[s],
                            kb * BLOCK_K, m_start);
            }

            load_decode_rhs_swizzled_batched(
                smem_b[s], gate_weight, gate_scale,
                n_start, kb * BLOCK_K, N, K,
                stride_weight_n, stride_scale_n, wg_tid, byte_lut);

            bar_sync(PRODUCER_BAR_ID, PRODUCER_THREADS);
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            if (wg_tid == 0) {
                mbarrier_arrive(&gate_full_barriers[s]);
            }
        }

        __syncthreads();

        // UP K-LOOP
        for (int kb = 0; kb < num_k_blocks; kb++) {
            const int s = kb % NUM_STAGES;
            const int empty_phase = ((kb / NUM_STAGES) + 1) & 1;

            mbarrier_wait_parity(&up_empty_barriers[s], empty_phase);

            if (wg_tid == 0) {
                mbarrier_arrive_expect_tx(&up_full_barriers[s], TILE_BYTES);
                tma_load_2d(&desc_a, &up_full_barriers[s], smem_a[s],
                            kb * BLOCK_K, m_start);
            }

            load_decode_rhs_swizzled_batched(
                smem_b[s], up_weight, up_scale,
                n_start, kb * BLOCK_K, N, K,
                stride_weight_n, stride_scale_n, wg_tid, byte_lut);

            bar_sync(PRODUCER_BAR_ID, PRODUCER_THREADS);
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            if (wg_tid == 0) {
                mbarrier_arrive(&up_full_barriers[s]);
            }
        }

    } else {
        // MATH WG: gate GEMM -> up GEMM -> SwiGLU
        float gate_acc[WGMMA_NUM_ACCUM];
        float up_acc[WGMMA_NUM_ACCUM];
        for (int i = 0; i < WGMMA_NUM_ACCUM; i++) gate_acc[i] = 0.0f;
        for (int i = 0; i < WGMMA_NUM_ACCUM; i++) up_acc[i] = 0.0f;

        const int warp_in_wg = wg_tid / WARP_SIZE;
        const int lane_id = wg_tid % WARP_SIZE;

        // GATE K-LOOP
        for (int kb = 0; kb < num_k_blocks; kb++) {
            const int s = kb % NUM_STAGES;
            const int full_phase = (kb / NUM_STAGES) & 1;

            mbarrier_wait_parity(&gate_full_barriers[s], full_phase);

            #pragma unroll
            for (int i = 0; i < WGMMA_NUM_ACCUM; i++)
                warpgroup_fence_operand(gate_acc[i]);

            warpgroup_arrive();

            #pragma unroll
            for (int t = 0; t < TILES_K; t++) {
                GmmaDescriptor da = make_smem_desc(smem_a[s] + t * WGMMA_K);
                GmmaDescriptor db = make_smem_desc(smem_b[s] + t * WGMMA_K);
                wgmma_bf16_ss(da.desc_, db.desc_, gate_acc, 1);
            }

            warpgroup_commit_batch();

            #pragma unroll
            for (int i = 0; i < WGMMA_NUM_ACCUM; i++)
                warpgroup_fence_operand(gate_acc[i]);

            warpgroup_wait<0>();

            if (wg_tid == 0) {
                mbarrier_arrive(&gate_empty_barriers[s]);
            }
        }

        __syncthreads();

        // UP K-LOOP
        for (int kb = 0; kb < num_k_blocks; kb++) {
            const int s = kb % NUM_STAGES;
            const int full_phase = (kb / NUM_STAGES) & 1;

            mbarrier_wait_parity(&up_full_barriers[s], full_phase);

            #pragma unroll
            for (int i = 0; i < WGMMA_NUM_ACCUM; i++)
                warpgroup_fence_operand(up_acc[i]);

            warpgroup_arrive();

            #pragma unroll
            for (int t = 0; t < TILES_K; t++) {
                GmmaDescriptor da = make_smem_desc(smem_a[s] + t * WGMMA_K);
                GmmaDescriptor db = make_smem_desc(smem_b[s] + t * WGMMA_K);
                wgmma_bf16_ss(da.desc_, db.desc_, up_acc, 1);
            }

            warpgroup_commit_batch();

            #pragma unroll
            for (int i = 0; i < WGMMA_NUM_ACCUM; i++)
                warpgroup_fence_operand(up_acc[i]);

            warpgroup_wait<0>();

            if (wg_tid == 0) {
                mbarrier_arrive(&up_empty_barriers[s]);
            }
        }

        // SwiGLU EPILOGUE with bias
        #pragma unroll 1
        for (int i = 0; i < WGMMA_NUM_ACCUM; i++) {
            int m, n;
            reg_to_mn(i, warp_in_wg, lane_id, m, n);
            const int m_global = m_start + m;
            const int n_global = n_start + n;

            if (m_global >= expert_end || n_global >= N) continue;

            __nv_bfloat16 g_bf16 = __float2bfloat16(gate_acc[i]);
            __nv_bfloat16 u_bf16 = __float2bfloat16(up_acc[i]);

            if (has_gate_bias && gate_bias != nullptr) {
                g_bf16 = __hadd(g_bf16, gate_bias[n_global]);
            }
            if (has_up_bias && up_bias != nullptr) {
                u_bf16 = __hadd(u_bf16, up_bias[n_global]);
            }

            float g = __bfloat162float(g_bf16);
            float u = __bfloat162float(u_bf16);

            float g_c = fminf(g, SWIGLU_LIMIT);
            float u_c = fmaxf(fminf(u, SWIGLU_LIMIT), -SWIGLU_LIMIT);
            float sig = __frcp_rn(1.0f + expf(-SWIGLU_ALPHA * g_c));
            float result = g_c * sig * (u_c + 1.0f);

            C[m_global * stride_c_m + n_global] = __float2bfloat16(result);
        }
    }
}

// ============================================================================
// Stage 2 Kernel: Down projection (TMA-based, 3D grid, with bias)
//   Grid: (num_experts, num_k_tiles, max_m_tiles)
// ============================================================================
__global__ void __launch_bounds__(TOTAL_THREADS, 1)
grouped_mxfp4_moe_stage2_kernel(
    const __grid_constant__ CUtensorMap desc_input,
    const int64_t* __restrict__ down_ptrs,
    const int64_t* __restrict__ down_scale_ptrs,
    const int64_t* __restrict__ down_bias_ptrs,
    const int32_t* __restrict__ expert_offsets,
    __nv_bfloat16* __restrict__ C,
    int64_t stride_c_m,
    int total_tokens, int N, int K, int num_experts,
    int64_t stride_weight_n,
    int64_t stride_scale_n,
    int has_down_bias
) {
    const int tid = threadIdx.x;
    const int wg_id = tid / 128;
    const int wg_tid = tid % 128;

    const int expert_idx = blockIdx.x;
    const int k_tile = blockIdx.y;
    const int m_tile = blockIdx.z;
    const int k_start = k_tile * BLOCK_N;

    if (k_start >= K) return;

    const int expert_start = expert_offsets[expert_idx];
    const int expert_end = expert_offsets[expert_idx + 1];
    const int expert_tokens = expert_end - expert_start;

    const int num_m_tiles = (expert_tokens + BLOCK_M - 1) / BLOCK_M;
    if (m_tile >= num_m_tiles) return;

    const int m_start = expert_start + m_tile * BLOCK_M;
    const int m_size = min(BLOCK_M, expert_end - m_start);

    const uint8_t* down_weight = reinterpret_cast<const uint8_t*>(down_ptrs[expert_idx]);
    const uint8_t* down_scale = reinterpret_cast<const uint8_t*>(down_scale_ptrs[expert_idx]);
    const __nv_bfloat16* down_bias = has_down_bias ?
        reinterpret_cast<const __nv_bfloat16*>(down_bias_ptrs[expert_idx]) : nullptr;

    extern __shared__ __align__(128) char smem_buf[];
    __nv_bfloat16* smem_a[NUM_STAGES];
    __nv_bfloat16* smem_b[NUM_STAGES];
    for (int s = 0; s < NUM_STAGES; s++) {
        smem_a[s] = reinterpret_cast<__nv_bfloat16*>(smem_buf + (2*s)     * TILE_BYTES);
        smem_b[s] = reinterpret_cast<__nv_bfloat16*>(smem_buf + (2*s + 1) * TILE_BYTES);
    }
    const int bar_offset = 2 * NUM_STAGES * TILE_BYTES;
    uint64_t* full_barriers  = reinterpret_cast<uint64_t*>(smem_buf + bar_offset);
    uint64_t* empty_barriers = full_barriers + NUM_STAGES;
    __nv_bfloat162* byte_lut = reinterpret_cast<__nv_bfloat162*>(
        smem_buf + bar_offset + 2 * NUM_STAGES * sizeof(uint64_t));

    const int num_n_blocks = (N + BLOCK_K - 1) / BLOCK_K;

    // Init barriers with arrival count = 2 (TMA + decode)
    if (tid == 0) {
        for (int s = 0; s < NUM_STAGES; s++) {
            mbarrier_init(&full_barriers[s], 2);
            mbarrier_init(&empty_barriers[s], 1);
        }
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();

    if (wg_id == 0) {
        // PRODUCER WG

        for (int i = wg_tid; i < LUT_ENTRIES; i += PRODUCER_THREADS) {
            __nv_bfloat16 lo = __float2bfloat16(FP4_LUT[i & 0xF]);
            __nv_bfloat16 hi = __float2bfloat16(FP4_LUT[i >> 4]);
            byte_lut[i] = __halves2bfloat162(lo, hi);
        }
        bar_sync(PRODUCER_BAR_ID, PRODUCER_THREADS);

        if (wg_tid == 0) {
            asm volatile("prefetch.tensormap [%0];" :: "l"(&desc_input) : "memory");
        }

        for (int nb = 0; nb < num_n_blocks; nb++) {
            const int s = nb % NUM_STAGES;
            const int empty_phase = ((nb / NUM_STAGES) + 1) & 1;

            mbarrier_wait_parity(&empty_barriers[s], empty_phase);

            if (wg_tid == 0) {
                mbarrier_arrive_expect_tx(&full_barriers[s], TILE_BYTES);
                tma_load_2d(&desc_input, &full_barriers[s], smem_a[s],
                            nb * BLOCK_K, m_start);
            }

            load_decode_rhs_swizzled_batched(
                smem_b[s], down_weight, down_scale,
                k_start, nb * BLOCK_K, K, N,
                stride_weight_n, stride_scale_n, wg_tid, byte_lut);

            bar_sync(PRODUCER_BAR_ID, PRODUCER_THREADS);
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            if (wg_tid == 0) {
                mbarrier_arrive(&full_barriers[s]);
            }
        }

    } else {
        // MATH WG
        float acc[WGMMA_NUM_ACCUM];
        for (int i = 0; i < WGMMA_NUM_ACCUM; i++) acc[i] = 0.0f;

        const int warp_in_wg = wg_tid / WARP_SIZE;
        const int lane_id = wg_tid % WARP_SIZE;

        for (int nb = 0; nb < num_n_blocks; nb++) {
            const int s = nb % NUM_STAGES;
            const int full_phase = (nb / NUM_STAGES) & 1;

            mbarrier_wait_parity(&full_barriers[s], full_phase);

            #pragma unroll
            for (int i = 0; i < WGMMA_NUM_ACCUM; i++)
                warpgroup_fence_operand(acc[i]);

            warpgroup_arrive();

            #pragma unroll
            for (int t = 0; t < TILES_K; t++) {
                GmmaDescriptor da = make_smem_desc(smem_a[s] + t * WGMMA_K);
                GmmaDescriptor db = make_smem_desc(smem_b[s] + t * WGMMA_K);
                wgmma_bf16_ss(da.desc_, db.desc_, acc, 1);
            }

            warpgroup_commit_batch();

            #pragma unroll
            for (int i = 0; i < WGMMA_NUM_ACCUM; i++)
                warpgroup_fence_operand(acc[i]);

            warpgroup_wait<0>();

            if (wg_tid == 0) {
                mbarrier_arrive(&empty_barriers[s]);
            }
        }

        // EPILOGUE: store with optional bias
        #pragma unroll 1
        for (int i = 0; i < WGMMA_NUM_ACCUM; i++) {
            int m, n;
            reg_to_mn(i, warp_in_wg, lane_id, m, n);
            const int m_global = m_start + m;
            const int k_global = k_start + n;

            if (m_global >= expert_end || k_global >= K) continue;

            float result = acc[i];

            if (has_down_bias && down_bias != nullptr) {
                result += __bfloat162float(down_bias[k_global]);
            }

            C[m_global * stride_c_m + k_global] = __float2bfloat16(result);
        }
    }
}

// ============================================================================
// Host utilities: TMA descriptor creation
// ============================================================================
static PFN_cuTensorMapEncodeTiled get_cuTensorMapEncodeTiled() {
    cudaDriverEntryPointQueryResult driver_status;
    void* ptr = nullptr;
#if CUDA_VERSION >= 12050
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &ptr, 12000,
                                     cudaEnableDefault, &driver_status);
#else
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &ptr,
                            cudaEnableDefault, &driver_status);
#endif
    if (driver_status != cudaDriverEntryPointSuccess)
        throw std::runtime_error("Failed to get cuTensorMapEncodeTiled");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled>(ptr);
}

static CUtensorMap make_2d_tma_desc_bf16(
    __nv_bfloat16* global_address,
    uint64_t gmem_rows, uint64_t gmem_cols,
    uint32_t smem_rows, uint32_t smem_cols,
    PFN_cuTensorMapEncodeTiled encode_func
) {
    CUtensorMap tensor_map = {};
    uint64_t gmem_dim[2] = {gmem_cols, gmem_rows};
    uint64_t global_stride[1] = {gmem_cols * sizeof(__nv_bfloat16)};
    uint32_t smem_dim[2] = {smem_cols, smem_rows};
    uint32_t elem_strides[2] = {1, 1};

    auto result = encode_func(
        &tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        global_address,
        gmem_dim,
        global_stride,
        smem_dim,
        elem_strides,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
    if (result != CUDA_SUCCESS)
        throw std::runtime_error("cuTensorMapEncodeTiled failed");
    return tensor_map;
}

// ============================================================================
// C++ wrapper: Stage 1 (gate + up + SwiGLU) with TMA + bias
// ============================================================================
torch::Tensor grouped_mxfp4_moe_stage1(
    torch::Tensor A,
    torch::Tensor expert_offsets,
    torch::Tensor gate_ptrs,
    torch::Tensor gate_scale_ptrs,
    torch::Tensor up_ptrs,
    torch::Tensor up_scale_ptrs,
    torch::Tensor gate_bias_ptrs,
    torch::Tensor up_bias_ptrs,
    int N,
    int64_t stride_weight_n,
    int64_t stride_scale_n,
    int max_m_tiles
) {
    TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kBFloat16);
    TORCH_CHECK(expert_offsets.is_cuda() && expert_offsets.dtype() == torch::kInt32);

    const int total_tokens = A.size(0);
    const int K = A.size(1);
    const int num_experts = expert_offsets.size(0) - 1;

    auto C = torch::empty({total_tokens, N}, A.options());

    auto encode_func = get_cuTensorMapEncodeTiled();
    CUtensorMap desc_a = make_2d_tma_desc_bf16(
        reinterpret_cast<__nv_bfloat16*>(A.data_ptr()),
        total_tokens, K, BLOCK_M, BLOCK_K, encode_func);

    const int num_n_tiles = (N + BLOCK_N - 1) / BLOCK_N;
    dim3 grid(num_experts, num_n_tiles, max_m_tiles);
    dim3 block(TOTAL_THREADS);

    constexpr int smem_bytes = 2 * NUM_STAGES * TILE_BYTES +
                               4 * NUM_STAGES * sizeof(uint64_t) + LUT_BYTES;

    cudaFuncSetAttribute(grouped_mxfp4_moe_stage1_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    int has_gate_bias = gate_bias_ptrs.numel() > 0 ? 1 : 0;
    int has_up_bias = up_bias_ptrs.numel() > 0 ? 1 : 0;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grouped_mxfp4_moe_stage1_kernel<<<grid, block, smem_bytes, stream>>>(
        desc_a,
        gate_ptrs.data_ptr<int64_t>(),
        gate_scale_ptrs.data_ptr<int64_t>(),
        up_ptrs.data_ptr<int64_t>(),
        up_scale_ptrs.data_ptr<int64_t>(),
        has_gate_bias ? gate_bias_ptrs.data_ptr<int64_t>() : nullptr,
        has_up_bias ? up_bias_ptrs.data_ptr<int64_t>() : nullptr,
        expert_offsets.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        C.stride(0),
        total_tokens, N, K, num_experts,
        stride_weight_n, stride_scale_n,
        has_gate_bias, has_up_bias);

    return C;
}

// ============================================================================
// C++ wrapper: Stage 2 (down projection) with TMA + bias
// ============================================================================
torch::Tensor grouped_mxfp4_moe_stage2(
    torch::Tensor input,
    torch::Tensor expert_offsets,
    torch::Tensor down_ptrs,
    torch::Tensor down_scale_ptrs,
    torch::Tensor down_bias_ptrs,
    int K,
    int64_t stride_weight_n,
    int64_t stride_scale_n,
    int max_m_tiles
) {
    TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kBFloat16);
    TORCH_CHECK(expert_offsets.is_cuda() && expert_offsets.dtype() == torch::kInt32);

    const int total_tokens = input.size(0);
    const int N = input.size(1);
    const int num_experts = expert_offsets.size(0) - 1;

    auto C = torch::empty({total_tokens, K}, input.options());

    auto encode_func = get_cuTensorMapEncodeTiled();
    CUtensorMap desc_input = make_2d_tma_desc_bf16(
        reinterpret_cast<__nv_bfloat16*>(input.data_ptr()),
        total_tokens, N, BLOCK_M, BLOCK_K, encode_func);

    const int num_k_tiles = (K + BLOCK_N - 1) / BLOCK_N;
    dim3 grid(num_experts, num_k_tiles, max_m_tiles);
    dim3 block(TOTAL_THREADS);

    constexpr int smem_bytes = 2 * NUM_STAGES * TILE_BYTES +
                               2 * NUM_STAGES * sizeof(uint64_t) + LUT_BYTES;

    cudaFuncSetAttribute(grouped_mxfp4_moe_stage2_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    int has_down_bias = down_bias_ptrs.numel() > 0 ? 1 : 0;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grouped_mxfp4_moe_stage2_kernel<<<grid, block, smem_bytes, stream>>>(
        desc_input,
        down_ptrs.data_ptr<int64_t>(),
        down_scale_ptrs.data_ptr<int64_t>(),
        has_down_bias ? down_bias_ptrs.data_ptr<int64_t>() : nullptr,
        expert_offsets.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        C.stride(0),
        total_tokens, N, K, num_experts,
        stride_weight_n, stride_scale_n,
        has_down_bias);

    return C;
}

// ============================================================================
// TMA descriptor creation (callable from Python for pre-building descriptors)
// ============================================================================
torch::Tensor create_tma_desc_bf16(
    torch::Tensor tensor,
    int block_m,
    int block_k
) {
    TORCH_CHECK(tensor.is_cuda() && tensor.dtype() == torch::kBFloat16);
    TORCH_CHECK(tensor.dim() == 2, "Expected 2D tensor");

    auto encode_func = get_cuTensorMapEncodeTiled();
    CUtensorMap desc = make_2d_tma_desc_bf16(
        reinterpret_cast<__nv_bfloat16*>(tensor.data_ptr()),
        tensor.size(0), tensor.size(1), block_m, block_k, encode_func);

    // Copy the 128-byte descriptor into a uint8 tensor
    auto desc_tensor = torch::empty({128}, torch::dtype(torch::kUInt8).device(torch::kCPU));
    std::memcpy(desc_tensor.data_ptr(), &desc, sizeof(CUtensorMap));
    return desc_tensor;
}

// ============================================================================
// C++ wrapper: Stage 1 inplace (pre-allocated output + pre-built TMA desc)
// ============================================================================
void grouped_mxfp4_moe_stage1_inplace(
    torch::Tensor A,
    torch::Tensor C,
    torch::Tensor tma_desc_bytes,
    torch::Tensor expert_offsets,
    torch::Tensor gate_ptrs,
    torch::Tensor gate_scale_ptrs,
    torch::Tensor up_ptrs,
    torch::Tensor up_scale_ptrs,
    torch::Tensor gate_bias_ptrs,
    torch::Tensor up_bias_ptrs,
    int N,
    int64_t stride_weight_n,
    int64_t stride_scale_n,
    int max_m_tiles
) {
    TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kBFloat16);
    TORCH_CHECK(C.is_cuda() && C.dtype() == torch::kBFloat16);
    TORCH_CHECK(tma_desc_bytes.numel() == 128, "TMA descriptor must be 128 bytes");

    const int total_tokens = A.size(0);
    const int K = A.size(1);
    const int num_experts = expert_offsets.size(0) - 1;

    // Copy pre-built TMA descriptor from CPU tensor
    CUtensorMap desc_a;
    std::memcpy(&desc_a, tma_desc_bytes.data_ptr(), sizeof(CUtensorMap));

    const int num_n_tiles = (N + BLOCK_N - 1) / BLOCK_N;
    dim3 grid(num_experts, num_n_tiles, max_m_tiles);
    dim3 block(TOTAL_THREADS);

    constexpr int smem_bytes = 2 * NUM_STAGES * TILE_BYTES +
                               4 * NUM_STAGES * sizeof(uint64_t) + LUT_BYTES;

    cudaFuncSetAttribute(grouped_mxfp4_moe_stage1_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    int has_gate_bias = gate_bias_ptrs.numel() > 0 ? 1 : 0;
    int has_up_bias = up_bias_ptrs.numel() > 0 ? 1 : 0;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grouped_mxfp4_moe_stage1_kernel<<<grid, block, smem_bytes, stream>>>(
        desc_a,
        gate_ptrs.data_ptr<int64_t>(),
        gate_scale_ptrs.data_ptr<int64_t>(),
        up_ptrs.data_ptr<int64_t>(),
        up_scale_ptrs.data_ptr<int64_t>(),
        has_gate_bias ? gate_bias_ptrs.data_ptr<int64_t>() : nullptr,
        has_up_bias ? up_bias_ptrs.data_ptr<int64_t>() : nullptr,
        expert_offsets.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        C.stride(0),
        total_tokens, N, K, num_experts,
        stride_weight_n, stride_scale_n,
        has_gate_bias, has_up_bias);
}

// ============================================================================
// C++ wrapper: Stage 2 inplace (pre-allocated output + pre-built TMA desc)
// ============================================================================
void grouped_mxfp4_moe_stage2_inplace(
    torch::Tensor input,
    torch::Tensor C,
    torch::Tensor tma_desc_bytes,
    torch::Tensor expert_offsets,
    torch::Tensor down_ptrs,
    torch::Tensor down_scale_ptrs,
    torch::Tensor down_bias_ptrs,
    int K,
    int64_t stride_weight_n,
    int64_t stride_scale_n,
    int max_m_tiles
) {
    TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kBFloat16);
    TORCH_CHECK(C.is_cuda() && C.dtype() == torch::kBFloat16);
    TORCH_CHECK(tma_desc_bytes.numel() == 128, "TMA descriptor must be 128 bytes");

    const int total_tokens = input.size(0);
    const int N = input.size(1);
    const int num_experts = expert_offsets.size(0) - 1;

    // Copy pre-built TMA descriptor from CPU tensor
    CUtensorMap desc_input;
    std::memcpy(&desc_input, tma_desc_bytes.data_ptr(), sizeof(CUtensorMap));

    const int num_k_tiles = (K + BLOCK_N - 1) / BLOCK_N;
    dim3 grid(num_experts, num_k_tiles, max_m_tiles);
    dim3 block(TOTAL_THREADS);

    constexpr int smem_bytes = 2 * NUM_STAGES * TILE_BYTES +
                               2 * NUM_STAGES * sizeof(uint64_t) + LUT_BYTES;

    cudaFuncSetAttribute(grouped_mxfp4_moe_stage2_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    int has_down_bias = down_bias_ptrs.numel() > 0 ? 1 : 0;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grouped_mxfp4_moe_stage2_kernel<<<grid, block, smem_bytes, stream>>>(
        desc_input,
        down_ptrs.data_ptr<int64_t>(),
        down_scale_ptrs.data_ptr<int64_t>(),
        has_down_bias ? down_bias_ptrs.data_ptr<int64_t>() : nullptr,
        expert_offsets.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        C.stride(0),
        total_tokens, N, K, num_experts,
        stride_weight_n, stride_scale_n,
        has_down_bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grouped_mxfp4_moe_stage1", &grouped_mxfp4_moe_stage1,
          "Grouped MXFP4 Stage 1 (gate+up+SwiGLU) with TMA + bias");
    m.def("grouped_mxfp4_moe_stage2", &grouped_mxfp4_moe_stage2,
          "Grouped MXFP4 Stage 2 (down projection) with TMA + bias");
    m.def("create_tma_desc_bf16", &create_tma_desc_bf16,
          "Create TMA descriptor for a 2D BF16 tensor");
    m.def("grouped_mxfp4_moe_stage1_inplace", &grouped_mxfp4_moe_stage1_inplace,
          "Grouped MXFP4 Stage 1 inplace (pre-allocated output + TMA desc)");
    m.def("grouped_mxfp4_moe_stage2_inplace", &grouped_mxfp4_moe_stage2_inplace,
          "Grouped MXFP4 Stage 2 inplace (pre-allocated output + TMA desc)");
}
