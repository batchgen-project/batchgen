#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <ATen/cuda/CUDAContext.h>
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
    desc.bitfield.stride_byte_offset_ = BLOCK_K;
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
// bf16_mul_rn: BF16 multiply with forced rounding (prevents FMA fusion)
// ============================================================================
__device__ __forceinline__ __nv_bfloat16 bf16_mul_rn(__nv_bfloat16 a, __nv_bfloat16 b) {
    float fa = __bfloat162float(a);
    float fb = __bfloat162float(b);
    float fc;
    asm volatile("mul.rn.f32 %0, %1, %2;" : "=f"(fc) : "f"(fa), "f"(fb));
    return __float2bfloat16(fc);
}

// ============================================================================
// reg_to_mn: accumulator register -> (m, n) position in 64x64 tile
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
        "QKV_WGMMA_MBAR_WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
        "@P bra QKV_WGMMA_MBAR_DONE_%=;\n"
        "bra QKV_WGMMA_MBAR_WAIT_%=;\n"
        "QKV_WGMMA_MBAR_DONE_%=:\n"
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
// QKV Projection WGMMA Kernel (with optional RoPE fusion)
//   Output = input [M, K] @ W_qkv [N, K].T + bias [N]
//   Split store: Q [M, q_size], K [M, kv_size], V [M, kv_size]
//   Optional: RoPE rotation on Q and K in epilogue
//   Grid: (ceil(M/64), ceil(N/64))
// ============================================================================
__global__ void __launch_bounds__(TOTAL_THREADS, 1)
qkv_wgmma_kernel(
    const __grid_constant__ CUtensorMap desc_a,   // input [M, K]
    const __grid_constant__ CUtensorMap desc_b,   // W_qkv [N, K]
    const __nv_bfloat16* __restrict__ bias,       // [N] or nullptr
    __nv_bfloat16* __restrict__ Q_out,            // [M, q_size]
    __nv_bfloat16* __restrict__ K_out,            // [M, kv_size]
    __nv_bfloat16* __restrict__ V_out,            // [M, kv_size]
    int64_t stride_q, int64_t stride_k, int64_t stride_v,
    const __nv_bfloat16* __restrict__ rope_cos,   // [M, head_dim] or nullptr
    const __nv_bfloat16* __restrict__ rope_sin,   // [M, head_dim] or nullptr
    int head_dim,                                 // 64
    const int* __restrict__ num_valid_ptr,        // 1-element int32 device tensor, or nullptr
    int M, int N, int K,
    int q_size, int kv_size, int has_bias
) {
    const int tid = threadIdx.x;
    const int wg_id = tid / 128;
    const int wg_tid = tid % 128;

    // 2D grid: (m_tile, n_tile)
    const int m_tile = blockIdx.x;
    const int n_tile = blockIdx.y;
    const int m_start = m_tile * BLOCK_M;
    const int n_start = n_tile * BLOCK_N;

    // num_valid_tokens guard for CUDA graph bucketing
    if (num_valid_ptr != nullptr && m_start >= *num_valid_ptr) return;
    if (m_start >= M) return;

    const int m_size = min(BLOCK_M, M - m_start);

    // -- Shared memory layout --
    extern __shared__ __align__(128) char smem_buf[];
    __nv_bfloat16* smem_a[NUM_STAGES];
    __nv_bfloat16* smem_b[NUM_STAGES];
    for (int s = 0; s < NUM_STAGES; s++) {
        smem_a[s] = reinterpret_cast<__nv_bfloat16*>(smem_buf + (2*s)     * TILE_BYTES);
        smem_b[s] = reinterpret_cast<__nv_bfloat16*>(smem_buf + (2*s + 1) * TILE_BYTES);
    }
    uint64_t* full_barriers  = reinterpret_cast<uint64_t*>(
        smem_buf + 2 * NUM_STAGES * TILE_BYTES);
    uint64_t* empty_barriers = full_barriers + NUM_STAGES;

    const int num_k_blocks = (K + BLOCK_K - 1) / BLOCK_K;

    // -- Init barriers --
    if (tid == 0) {
        for (int s = 0; s < NUM_STAGES; s++) {
            mbarrier_init(&full_barriers[s], 1);
            mbarrier_init(&empty_barriers[s], 1);
        }
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();

    if (wg_id == 0) {
        // ================================================================
        // PRODUCER WG: TMA loads for A and B
        // ================================================================
        if (wg_tid == 0) {
            asm volatile("prefetch.tensormap [%0];" :: "l"(&desc_a) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"(&desc_b) : "memory");
        }

        for (int kb = 0; kb < num_k_blocks; kb++) {
            const int s = kb % NUM_STAGES;
            const int empty_phase = ((kb / NUM_STAGES) + 1) & 1;

            mbarrier_wait_parity(&empty_barriers[s], empty_phase);

            if (wg_tid == 0) {
                mbarrier_arrive_expect_tx(&full_barriers[s], 2 * TILE_BYTES);
                tma_load_2d(&desc_a, &full_barriers[s], smem_a[s],
                            kb * BLOCK_K, m_start);
                tma_load_2d(&desc_b, &full_barriers[s], smem_b[s],
                            kb * BLOCK_K, n_start);
            }
        }

    } else {
        // ================================================================
        // MATH WG: WGMMA K-loop + split epilogue (with optional RoPE)
        // ================================================================
        float acc[WGMMA_NUM_ACCUM];
        for (int i = 0; i < WGMMA_NUM_ACCUM; i++) acc[i] = 0.0f;

        const int warp_in_wg = wg_tid / WARP_SIZE;
        const int lane_id = wg_tid % WARP_SIZE;

        // === K-LOOP ===
        for (int kb = 0; kb < num_k_blocks; kb++) {
            const int s = kb % NUM_STAGES;
            const int full_phase = (kb / NUM_STAGES) & 1;

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

        // === SPLIT EPILOGUE (with optional RoPE) ===
        const int kv_start = q_size;
        const int v_start  = q_size + kv_size;

        __nv_bfloat16* out_ptr;
        int64_t out_stride;
        int n_offset;
        int is_qk = 0;

        if (n_start < kv_start) {
            out_ptr = Q_out;
            out_stride = stride_q;
            n_offset = 0;
            is_qk = 1;
        } else if (n_start < v_start) {
            out_ptr = K_out;
            out_stride = stride_k;
            n_offset = kv_start;
            is_qk = 1;
        } else {
            out_ptr = V_out;
            out_stride = stride_v;
            n_offset = v_start;
        }

        if (is_qk && rope_cos != nullptr) {
            // -- RoPE-fused epilogue for Q/K tiles --
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int m1, n1, m2, n2;
                reg_to_mn(i, warp_in_wg, lane_id, m1, n1);
                reg_to_mn(i + 16, warp_in_wg, lane_id, m2, n2);

                const int m_global = m_start + m1;
                const int n1_local = n_start + n1 - n_offset;
                const int n2_local = n_start + n2 - n_offset;

                if (m1 < m_size) {
                    __nv_bfloat16 x1 = __float2bfloat16(acc[i]);
                    __nv_bfloat16 x2 = __float2bfloat16(acc[i + 16]);

                    if (has_bias) {
                        x1 = __hadd(x1, bias[n_start + n1]);
                        x2 = __hadd(x2, bias[n_start + n2]);
                    }

                    __nv_bfloat16 c = rope_cos[m_global * head_dim + n1];
                    __nv_bfloat16 s = rope_sin[m_global * head_dim + n1];

                    __nv_bfloat16 x1c = bf16_mul_rn(x1, c);
                    __nv_bfloat16 x2s = bf16_mul_rn(x2, s);
                    __nv_bfloat16 x2c = bf16_mul_rn(x2, c);
                    __nv_bfloat16 x1s = bf16_mul_rn(x1, s);

                    out_ptr[m_global * out_stride + n1_local] = __hsub(x1c, x2s);
                    out_ptr[m_global * out_stride + n2_local] = __hadd(x2c, x1s);
                }
            }
        } else {
            // -- Standard epilogue (no RoPE) --
            #pragma unroll
            for (int i = 0; i < WGMMA_NUM_ACCUM; i++) {
                int m, n;
                reg_to_mn(i, warp_in_wg, lane_id, m, n);
                const int m_global = m_start + m;
                const int n_global = n_start + n;
                const int n_local = n_global - n_offset;

                if (m < m_size && n_global < N) {
                    __nv_bfloat16 result = __float2bfloat16(acc[i]);

                    if (has_bias) {
                        result = __hadd(result, bias[n_global]);
                    }

                    out_ptr[m_global * out_stride + n_local] = result;
                }
            }
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
// Shared kernel launch helper (used by both eager and inplace wrappers)
// ============================================================================
static bool _smem_configured = false;

static void launch_qkv_wgmma(
    CUtensorMap& desc_a, CUtensorMap& desc_b,
    const __nv_bfloat16* bias_ptr, int has_bias,
    __nv_bfloat16* Q_ptr, __nv_bfloat16* K_ptr, __nv_bfloat16* V_ptr,
    int64_t stride_q, int64_t stride_k, int64_t stride_v,
    const __nv_bfloat16* rope_cos_ptr, const __nv_bfloat16* rope_sin_ptr,
    int head_dim, const int* num_valid_ptr,
    int M, int N, int K, int q_size, int kv_size
) {
    const int num_m_tiles = (M + BLOCK_M - 1) / BLOCK_M;
    const int num_n_tiles = (N + BLOCK_N - 1) / BLOCK_N;
    dim3 grid(num_m_tiles, num_n_tiles);
    dim3 block(TOTAL_THREADS);

    constexpr int smem_bytes = 2 * NUM_STAGES * TILE_BYTES + 2 * NUM_STAGES * sizeof(uint64_t);

    if (!_smem_configured) {
        cudaFuncSetAttribute(qkv_wgmma_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        _smem_configured = true;
    }

    // Must use PyTorch's current stream — default stream (<<<grid, block, smem>>>)
    // is NOT captured during CUDA graph recording.
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    qkv_wgmma_kernel<<<grid, block, smem_bytes, stream>>>(
        desc_a, desc_b,
        bias_ptr, Q_ptr, K_ptr, V_ptr,
        stride_q, stride_k, stride_v,
        rope_cos_ptr, rope_sin_ptr, head_dim,
        num_valid_ptr, M, N, K, q_size, kv_size, has_bias);
}

// ============================================================================
// C++ wrapper: eager mode (creates TMA descs + allocates output)
// NOT safe during CUDA graph capture — use qkv_wgmma_forward_inplace instead.
// ============================================================================
std::vector<torch::Tensor> qkv_wgmma_forward(
    torch::Tensor input,        // [M, K] BF16
    torch::Tensor weight,       // [N, K] BF16 (row-major, nn.Linear convention)
    torch::Tensor bias,         // [N] BF16 or empty tensor (size 0)
    int q_size,                 // 4096
    int kv_size,                // 512
    c10::optional<torch::Tensor> num_valid_tokens,  // 1-element int32 device tensor
    torch::Tensor rope_cos,     // [M, head_dim] BF16 or empty tensor (size 0)
    torch::Tensor rope_sin,     // [M, head_dim] BF16 or empty tensor (size 0)
    int head_dim                // 64
) {
    TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kBFloat16);
    TORCH_CHECK(weight.is_cuda() && weight.dtype() == torch::kBFloat16);
    TORCH_CHECK(input.is_contiguous());
    TORCH_CHECK(weight.is_contiguous());

    const int M = input.size(0);
    const int K = input.size(1);
    const int N = weight.size(0);

    TORCH_CHECK(weight.size(1) == K, "Weight K dimension mismatch");
    TORCH_CHECK(q_size + 2 * kv_size == N, "q_size + 2*kv_size must equal N");

    int has_bias = 0;
    if (bias.numel() > 0) {
        TORCH_CHECK(bias.is_cuda() && bias.dtype() == torch::kBFloat16);
        TORCH_CHECK(bias.size(0) == N, "Bias size must equal N");
        has_bias = 1;
    }

    const __nv_bfloat16* rope_cos_ptr = nullptr;
    const __nv_bfloat16* rope_sin_ptr = nullptr;
    if (rope_cos.numel() > 0) {
        TORCH_CHECK(rope_cos.is_cuda() && rope_cos.dtype() == torch::kBFloat16);
        TORCH_CHECK(rope_sin.is_cuda() && rope_sin.dtype() == torch::kBFloat16);
        TORCH_CHECK(rope_cos.is_contiguous() && rope_sin.is_contiguous());
        TORCH_CHECK(rope_cos.size(0) == M && rope_cos.size(1) == head_dim,
                     "rope_cos must be [M, head_dim]");
        TORCH_CHECK(rope_sin.size(0) == M && rope_sin.size(1) == head_dim,
                     "rope_sin must be [M, head_dim]");
        rope_cos_ptr = reinterpret_cast<const __nv_bfloat16*>(rope_cos.data_ptr());
        rope_sin_ptr = reinterpret_cast<const __nv_bfloat16*>(rope_sin.data_ptr());
    }

    const int* num_valid_ptr = nullptr;
    if (num_valid_tokens.has_value() && num_valid_tokens->defined()) {
        num_valid_ptr = num_valid_tokens->data_ptr<int>();
    }

    auto Q_out = torch::empty({M, q_size}, input.options());
    auto K_out = torch::empty({M, kv_size}, input.options());
    auto V_out = torch::empty({M, kv_size}, input.options());

    if (M == 0) return {Q_out, K_out, V_out};

    auto encode_func = get_cuTensorMapEncodeTiled();

    CUtensorMap desc_a = make_2d_tma_desc_bf16(
        reinterpret_cast<__nv_bfloat16*>(input.data_ptr()),
        M, K, BLOCK_M, BLOCK_K, encode_func);

    CUtensorMap desc_b = make_2d_tma_desc_bf16(
        reinterpret_cast<__nv_bfloat16*>(weight.data_ptr()),
        N, K, BLOCK_N, BLOCK_K, encode_func);

    launch_qkv_wgmma(
        desc_a, desc_b,
        has_bias ? reinterpret_cast<__nv_bfloat16*>(bias.data_ptr()) : nullptr, has_bias,
        reinterpret_cast<__nv_bfloat16*>(Q_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(K_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(V_out.data_ptr()),
        Q_out.stride(0), K_out.stride(0), V_out.stride(0),
        rope_cos_ptr, rope_sin_ptr, head_dim, num_valid_ptr,
        M, N, K, q_size, kv_size);

    return {Q_out, K_out, V_out};
}

// ============================================================================
// Create TMA descriptor (callable from Python, BEFORE graph capture)
// Returns CPU uint8[128] encoding the CUtensorMap struct.
// ============================================================================
torch::Tensor create_qkv_tma_desc(
    torch::Tensor tensor,       // [rows, cols] BF16, CUDA, contiguous
    int block_rows,             // BLOCK_M or BLOCK_N
    int block_cols              // BLOCK_K
) {
    TORCH_CHECK(tensor.is_cuda() && tensor.dtype() == torch::kBFloat16);
    TORCH_CHECK(tensor.dim() == 2, "Expected 2D tensor");
    TORCH_CHECK(tensor.is_contiguous());

    auto encode_func = get_cuTensorMapEncodeTiled();
    CUtensorMap desc = make_2d_tma_desc_bf16(
        reinterpret_cast<__nv_bfloat16*>(tensor.data_ptr()),
        tensor.size(0), tensor.size(1),
        block_rows, block_cols, encode_func);

    auto desc_tensor = torch::empty({128}, torch::dtype(torch::kUInt8).device(torch::kCPU));
    std::memcpy(desc_tensor.data_ptr(), &desc, sizeof(CUtensorMap));
    return desc_tensor;
}

// ============================================================================
// C++ wrapper: inplace mode (pre-allocated output + pre-built TMA descs)
// Safe during CUDA graph capture — no cuTensorMapEncodeTiled, no torch::empty.
// ============================================================================
void qkv_wgmma_forward_inplace(
    torch::Tensor Q_out,              // [max_M, q_size] BF16, pre-allocated
    torch::Tensor K_out,              // [max_M, kv_size] BF16, pre-allocated
    torch::Tensor V_out,              // [max_M, kv_size] BF16, pre-allocated
    torch::Tensor tma_desc_a_bytes,   // CPU uint8[128], pre-built for input
    torch::Tensor tma_desc_b_bytes,   // CPU uint8[128], pre-built for weight
    torch::Tensor bias,               // [N] BF16 or empty tensor (size 0)
    torch::Tensor rope_cos,           // [M, head_dim] BF16 or empty tensor (size 0)
    torch::Tensor rope_sin,           // [M, head_dim] BF16 or empty tensor (size 0)
    int head_dim,
    c10::optional<torch::Tensor> num_valid_tokens,
    int M, int N, int K, int q_size, int kv_size
) {
    TORCH_CHECK(tma_desc_a_bytes.numel() == 128, "TMA desc A must be 128 bytes");
    TORCH_CHECK(tma_desc_b_bytes.numel() == 128, "TMA desc B must be 128 bytes");
    TORCH_CHECK(Q_out.is_cuda() && Q_out.dtype() == torch::kBFloat16);
    TORCH_CHECK(K_out.is_cuda() && K_out.dtype() == torch::kBFloat16);
    TORCH_CHECK(V_out.is_cuda() && V_out.dtype() == torch::kBFloat16);

    if (M == 0) return;

    // Reconstruct TMA descriptors from pre-built bytes (host memcpy, no driver call)
    CUtensorMap desc_a, desc_b;
    std::memcpy(&desc_a, tma_desc_a_bytes.data_ptr(), sizeof(CUtensorMap));
    std::memcpy(&desc_b, tma_desc_b_bytes.data_ptr(), sizeof(CUtensorMap));

    int has_bias = 0;
    if (bias.numel() > 0) {
        has_bias = 1;
    }

    const __nv_bfloat16* rope_cos_ptr = nullptr;
    const __nv_bfloat16* rope_sin_ptr = nullptr;
    if (rope_cos.numel() > 0) {
        rope_cos_ptr = reinterpret_cast<const __nv_bfloat16*>(rope_cos.data_ptr());
        rope_sin_ptr = reinterpret_cast<const __nv_bfloat16*>(rope_sin.data_ptr());
    }

    const int* num_valid_ptr = nullptr;
    if (num_valid_tokens.has_value() && num_valid_tokens->defined()) {
        num_valid_ptr = num_valid_tokens->data_ptr<int>();
    }

    launch_qkv_wgmma(
        desc_a, desc_b,
        has_bias ? reinterpret_cast<__nv_bfloat16*>(bias.data_ptr()) : nullptr, has_bias,
        reinterpret_cast<__nv_bfloat16*>(Q_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(K_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(V_out.data_ptr()),
        Q_out.stride(0), K_out.stride(0), V_out.stride(0),
        rope_cos_ptr, rope_sin_ptr, head_dim, num_valid_ptr,
        M, N, K, q_size, kv_size);
}

// ============================================================================
// Module registration (manual PYBIND11, matches MoE WGMMA pattern)
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qkv_wgmma_forward", &qkv_wgmma_forward,
          "QKV WGMMA fused projection (eager mode)");
    m.def("qkv_wgmma_forward_inplace", &qkv_wgmma_forward_inplace,
          "QKV WGMMA fused projection (graph-safe inplace mode)");
    m.def("create_qkv_tma_desc", &create_qkv_tma_desc,
          "Create TMA descriptor for QKV WGMMA (call before graph capture)");
}
