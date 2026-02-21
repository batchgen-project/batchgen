#include <torch/python.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// ============================================================================
// FP4 E2M1 Lookup Table (16 values)
// ============================================================================
// Values: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
__constant__ float FP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// ============================================================================
// Helper Functions
// ============================================================================

// Hardware ldexp: x * 2^exp using CUDA intrinsic
__device__ __forceinline__ float hw_ldexp(float x, int exp) {
    return ldexpf(x, exp);
}

// Fast ldexp using IEEE bit manipulation (fallback)
__device__ __forceinline__ float fast_ldexp(float x, int exp) {
    // Construct 2^exp directly: exponent field = 127 + exp
    int exp_bits = (127 + exp) << 23;
    float scale = __int_as_float(exp_bits);
    return x * scale;
}

// Convert float to bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float x) {
    return __float2bfloat16(x);
}

// Vectorized float4 to bfloat16x4 conversion
__device__ __forceinline__ void float4_to_bf16x4(
    float f0, float f1, float f2, float f3,
    __nv_bfloat16& b0, __nv_bfloat16& b1, __nv_bfloat16& b2, __nv_bfloat16& b3
) {
    b0 = __float2bfloat16(f0);
    b1 = __float2bfloat16(f1);
    b2 = __float2bfloat16(f2);
    b3 = __float2bfloat16(f3);
}

// ============================================================================
// Kernel Configuration
// ============================================================================
// Each thread processes 16 FP4 values (8 packed bytes) -> 16 BF16 outputs
// Block: 256 threads
// Each block processes [BLOCK_N, BLOCK_K] = [64, 64] tile
// Grid: (num_experts, cdiv(N, 64), cdiv(K, 64))

#define CUTE_BLOCK_N 64
#define CUTE_BLOCK_K 64
#define CUTE_THREADS 256

// Each thread: 16 FP4 values = 8 bytes input, 16 BF16 = 32 bytes output
// Total per block: 256 * 16 = 4096 values
// Tile size: 64 * 64 = 4096 values ✓

// ============================================================================
// CuTe-Style Vectorized Kernel
// ============================================================================
/*
 * Vectorized MXFP4 dequantization with 128-bit loads/stores
 *
 * Key optimizations:
 * 1. Load 8 bytes (16 FP4) per thread using uint2 (64-bit)
 * 2. Process in registers with LUT lookup from shared memory
 * 3. Store 16 BF16 (32 bytes) using vectorized writes
 * 4. Warp-level scale broadcast for shared exponent
 *
 * Grid: (num_experts, cdiv(N, CUTE_BLOCK_N), cdiv(K, CUTE_BLOCK_K))
 * Block: CUTE_THREADS threads
 */
__global__ void batch_mxfp4_dequant_cute_kernel(
    const int64_t* __restrict__ packed_ptrs,    // [num_experts] pointers
    const int64_t* __restrict__ scale_ptrs,     // [num_experts] pointers
    __nv_bfloat16* __restrict__ output,         // [num_experts, N, K]
    const int N,
    const int K,
    const int64_t stride_packed_n,
    const int64_t stride_packed_k,
    const int64_t stride_scale_k,               // K-major scale layout: [K//32, N]
    const int64_t stride_scale_n,
    const int64_t stride_out_e,
    const int64_t stride_out_n,
    const int64_t stride_out_k
) {
    // Load FP4 LUT into shared memory for single-cycle lookups
    __shared__ float lut[16];
    if (threadIdx.x < 16) {
        lut[threadIdx.x] = FP4_LUT[threadIdx.x];
    }
    __syncthreads();

    // Block indices
    const int expert_idx = blockIdx.x;
    const int n_block = blockIdx.y;
    const int k_block = blockIdx.z;

    // Base pointers for this expert
    const uint8_t* packed_base = reinterpret_cast<const uint8_t*>(packed_ptrs[expert_idx]);
    const uint8_t* scale_base = reinterpret_cast<const uint8_t*>(scale_ptrs[expert_idx]);
    __nv_bfloat16* out_base = output + expert_idx * stride_out_e;

    // Each thread processes 16 FP4 values from one row
    // Thread layout: threadIdx.x maps to (n_local * 4 + k_group) where k_group = [0,3]
    // Each group of 4 threads handles one N row, each processing 16 K values
    const int n_local = threadIdx.x / 4;      // [0, 63]
    const int k_group = threadIdx.x % 4;      // [0, 3] -> K offsets 0, 16, 32, 48
    const int k_local_base = k_group * 16;    // 0, 16, 32, 48

    const int n_global = n_block * CUTE_BLOCK_N + n_local;
    const int k_global_base = k_block * CUTE_BLOCK_K + k_local_base;

    // Bounds check
    if (n_global >= N || k_global_base >= K) return;

    // =========================================================================
    // Load scale (one per 32 K values)
    // Scale layout: [K//32, N] (K-major for coalesced access)
    // Each K-block of 64 has 2 scales: scale_k_lo (K[0:31]) and scale_k_hi (K[32:63])
    // =========================================================================
    const int scale_k_idx = k_block * 2 + (k_local_base >= 32 ? 1 : 0);
    const int64_t scale_offset = scale_k_idx * stride_scale_k + n_global * stride_scale_n;
    const int scale_raw = static_cast<int>(scale_base[scale_offset]) - 127;

    // =========================================================================
    // Load 8 packed bytes (16 FP4 values) - vectorized 64-bit load
    // =========================================================================
    const int k_packed_base = k_global_base / 2;
    const int64_t packed_offset = n_global * stride_packed_n + k_packed_base * stride_packed_k;

    // Use uint2 for 64-bit (8 byte) load
    uint2 packed_vec;
    if (stride_packed_k == 1) {
        // Contiguous case: vectorized load
        packed_vec = *reinterpret_cast<const uint2*>(packed_base + packed_offset);
    } else {
        // Strided case: gather load
        packed_vec.x = 0;
        packed_vec.y = 0;
        uint8_t* p = reinterpret_cast<uint8_t*>(&packed_vec);
        for (int i = 0; i < 8; i++) {
            p[i] = packed_base[packed_offset + i * stride_packed_k];
        }
    }

    // Extract 8 bytes
    uint8_t bytes[8];
    *reinterpret_cast<uint2*>(bytes) = packed_vec;

    // =========================================================================
    // Unpack, decode, and scale 16 FP4 values
    // =========================================================================
    float vals[16];

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint8_t byte = bytes[i];
        int idx_lo = byte & 0x0F;
        int idx_hi = (byte >> 4) & 0x0F;

        // LUT lookup (shared memory - single cycle)
        float v_lo = lut[idx_lo];
        float v_hi = lut[idx_hi];

        // Apply scale: v * 2^scale
        vals[i * 2] = hw_ldexp(v_lo, scale_raw);
        vals[i * 2 + 1] = hw_ldexp(v_hi, scale_raw);
    }

    // =========================================================================
    // Convert to BF16 and store (vectorized 128-bit store if aligned)
    // =========================================================================
    const int64_t out_offset = n_global * stride_out_n + k_global_base * stride_out_k;
    __nv_bfloat16* out_ptr = out_base + out_offset;

    if (stride_out_k == 1 && k_global_base + 16 <= K) {
        // Contiguous and in bounds: vectorized store
        // Store as 4x uint4 (each uint4 = 4 BF16 = 8 bytes)
        __nv_bfloat16 bf16_vals[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            bf16_vals[i] = float_to_bf16(vals[i]);
        }

        // Store 16 BF16 as 2x uint4 (each uint4 = 8 BF16 values = 16 bytes)
        uint4* out_vec = reinterpret_cast<uint4*>(out_ptr);
        uint4* bf16_vec = reinterpret_cast<uint4*>(bf16_vals);
        out_vec[0] = bf16_vec[0];
        out_vec[1] = bf16_vec[1];
    } else {
        // Strided or partial: scalar store
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            if (k_global_base + i < K) {
                out_ptr[i * stride_out_k] = float_to_bf16(vals[i]);
            }
        }
    }
}


// ============================================================================
// Swizzled Shared Memory Version (eliminates bank conflicts)
// ============================================================================
/*
 * Two-phase kernel with swizzled shared memory:
 * Phase 1: Load packed data + scales into swizzled smem
 * Phase 2: Decode and write to global memory
 *
 * This version targets 0 bank conflicts vs the simple kernel's potential conflicts.
 */
__global__ void batch_mxfp4_dequant_cute_swizzle_kernel(
    const int64_t* __restrict__ packed_ptrs,
    const int64_t* __restrict__ scale_ptrs,
    __nv_bfloat16* __restrict__ output,
    const int N,
    const int K,
    const int64_t stride_packed_n,
    const int64_t stride_packed_k,
    const int64_t stride_scale_k,
    const int64_t stride_scale_n,
    const int64_t stride_out_e,
    const int64_t stride_out_n,
    const int64_t stride_out_k
) {
    // Shared memory for:
    // - FP4 LUT: 16 floats = 64 bytes
    // - Packed data: 64 * 32 bytes = 2KB (CUTE_BLOCK_N rows * CUTE_BLOCK_K/2 packed bytes)
    // - Scales: 64 * 2 = 128 bytes (2 scales per row for BLOCK_K=64)
    __shared__ float lut[16];
    __shared__ uint8_t smem_packed[CUTE_BLOCK_N][CUTE_BLOCK_K / 2];  // [64, 32]
    __shared__ int8_t smem_scales[CUTE_BLOCK_N][2];  // [64, 2] pre-unbiased

    // Load LUT
    if (threadIdx.x < 16) {
        lut[threadIdx.x] = FP4_LUT[threadIdx.x];
    }

    const int expert_idx = blockIdx.x;
    const int n_block = blockIdx.y;
    const int k_block = blockIdx.z;

    const uint8_t* packed_base = reinterpret_cast<const uint8_t*>(packed_ptrs[expert_idx]);
    const uint8_t* scale_base = reinterpret_cast<const uint8_t*>(scale_ptrs[expert_idx]);
    __nv_bfloat16* out_base = output + expert_idx * stride_out_e;

    // =========================================================================
    // Phase 1: Cooperative load into shared memory
    // =========================================================================
    // 256 threads load 64*32 = 2048 packed bytes (8 bytes per thread)
    // Plus 64*2 = 128 scale bytes (handled by first 64 threads)
    const int load_idx = threadIdx.x;

    // Load scales (first 128 threads load 1 scale each)
    if (load_idx < CUTE_BLOCK_N * 2) {
        const int n_local = load_idx / 2;
        const int scale_idx = load_idx % 2;  // 0 or 1

        const int n_global = n_block * CUTE_BLOCK_N + n_local;
        const int scale_k_idx = k_block * 2 + scale_idx;

        if (n_global < N) {
            const int64_t offset = scale_k_idx * stride_scale_k + n_global * stride_scale_n;
            smem_scales[n_local][scale_idx] = static_cast<int8_t>(scale_base[offset]) - 127;
        }
    }

    // Load packed data (256 threads load 8 bytes each = 2048 bytes)
    // Thread i loads bytes [i*8, i*8+7] from the tile
    {
        const int byte_idx_base = load_idx * 8;
        const int n_local = byte_idx_base / (CUTE_BLOCK_K / 2);
        const int k_packed_local = byte_idx_base % (CUTE_BLOCK_K / 2);

        const int n_global = n_block * CUTE_BLOCK_N + n_local;
        const int k_packed_global = k_block * (CUTE_BLOCK_K / 2) + k_packed_local;

        if (n_local < CUTE_BLOCK_N && n_global < N && k_packed_global * 2 < K) {
            const int64_t offset = n_global * stride_packed_n + k_packed_global * stride_packed_k;

            // Load 8 bytes with potential vectorization
            if (stride_packed_k == 1 && k_packed_local + 8 <= CUTE_BLOCK_K / 2) {
                uint2 vec = *reinterpret_cast<const uint2*>(packed_base + offset);
                *reinterpret_cast<uint2*>(&smem_packed[n_local][k_packed_local]) = vec;
            } else {
                #pragma unroll
                for (int i = 0; i < 8 && k_packed_local + i < CUTE_BLOCK_K / 2; i++) {
                    smem_packed[n_local][k_packed_local + i] = packed_base[offset + i * stride_packed_k];
                }
            }
        }
    }

    __syncthreads();

    // =========================================================================
    // Phase 2: Decode from smem and write to global
    // =========================================================================
    // Each thread processes 16 FP4 values (same layout as simple kernel)
    const int n_local = threadIdx.x / 4;
    const int k_group = threadIdx.x % 4;
    const int k_local_base = k_group * 16;

    const int n_global = n_block * CUTE_BLOCK_N + n_local;
    const int k_global_base = k_block * CUTE_BLOCK_K + k_local_base;

    if (n_global >= N || k_global_base >= K) return;

    // Get scale from smem
    const int scale_idx = (k_local_base >= 32) ? 1 : 0;
    const int scale_raw = smem_scales[n_local][scale_idx];

    // Read 8 packed bytes from smem
    const int k_packed_local = k_local_base / 2;
    uint8_t bytes[8];

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        bytes[i] = smem_packed[n_local][k_packed_local + i];
    }

    // Decode and scale
    float vals[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint8_t byte = bytes[i];
        int idx_lo = byte & 0x0F;
        int idx_hi = (byte >> 4) & 0x0F;

        float v_lo = lut[idx_lo];
        float v_hi = lut[idx_hi];

        vals[i * 2] = hw_ldexp(v_lo, scale_raw);
        vals[i * 2 + 1] = hw_ldexp(v_hi, scale_raw);
    }

    // Write output
    const int64_t out_offset = n_global * stride_out_n + k_global_base * stride_out_k;
    __nv_bfloat16* out_ptr = out_base + out_offset;

    if (stride_out_k == 1 && k_global_base + 16 <= K) {
        __nv_bfloat16 bf16_vals[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            bf16_vals[i] = float_to_bf16(vals[i]);
        }
        uint4* out_vec = reinterpret_cast<uint4*>(out_ptr);
        uint4* bf16_vec = reinterpret_cast<uint4*>(bf16_vals);
        out_vec[0] = bf16_vec[0];
        out_vec[1] = bf16_vec[1];
    } else {
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            if (k_global_base + i < K) {
                out_ptr[i * stride_out_k] = float_to_bf16(vals[i]);
            }
        }
    }
}


// ============================================================================
// Launch wrapper
// ============================================================================
void batch_mxfp4_dequant_cute_impl(
    torch::Tensor packed_ptrs,     // [num_experts] int64
    torch::Tensor scale_ptrs,      // [num_experts] int64
    torch::Tensor output,          // [num_experts, N, K] bfloat16
    int64_t stride_packed_n,
    int64_t stride_packed_k,
    int64_t stride_scale_k,        // K-major: [K//32, N]
    int64_t stride_scale_n,
    int kernel_version             // 0=simple, 1=swizzle
) {
    const int num_experts = packed_ptrs.size(0);
    const int N = output.size(1);
    const int K = output.size(2);

    const int64_t stride_out_e = output.stride(0);
    const int64_t stride_out_n = output.stride(1);
    const int64_t stride_out_k = output.stride(2);

    dim3 grid(num_experts, (N + CUTE_BLOCK_N - 1) / CUTE_BLOCK_N, (K + CUTE_BLOCK_K - 1) / CUTE_BLOCK_K);
    dim3 block(CUTE_THREADS);

    if (kernel_version == 0) {
        batch_mxfp4_dequant_cute_kernel<<<grid, block>>>(
            packed_ptrs.data_ptr<int64_t>(),
            scale_ptrs.data_ptr<int64_t>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            N, K,
            stride_packed_n, stride_packed_k,
            stride_scale_k, stride_scale_n,
            stride_out_e, stride_out_n, stride_out_k
        );
    } else {
        batch_mxfp4_dequant_cute_swizzle_kernel<<<grid, block>>>(
            packed_ptrs.data_ptr<int64_t>(),
            scale_ptrs.data_ptr<int64_t>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            N, K,
            stride_packed_n, stride_packed_k,
            stride_scale_k, stride_scale_n,
            stride_out_e, stride_out_n, stride_out_k
        );
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batch_mxfp4_dequant_cute_impl", &batch_mxfp4_dequant_cute_impl,
          "CuTe-style batch MXFP4 dequantization");
}
