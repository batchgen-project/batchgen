"""CUDA kernel: fused RMSNorm + RoPE on KV and Q + cache write.

Replaces the Triton `fused_rmsnorm_rope_cache_update_with_q_return_new_kv` kernel.

Per batch element (grid = bsz):
  1. RMSNorm on KV lora slice [512] → normalized KV [512]
  2. RoPE on KV rope slice [64] → rotated k_pe [64]
  3. RoPE on all 64 Q heads' rope slices [64 × 64] — modifies q_pe in-place
  4. Write normalized+rotated KV [576] to flat cache at position
  5. Return offload_kv [bsz, 1, 576] for downstream (q_absorb etc.)

Block: 256 threads
  - Warp 0-3 (128 threads): handle RMSNorm + KV RoPE + cache write
  - Warp 4-7 (128 threads): help with Q head RoPE (64 heads × 64 dims)
"""

import torch
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Warp-level reduction for RMSNorm variance
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void fused_kv_norm_rope_cache_kernel(
    const __nv_bfloat16* __restrict__ new_kv_ptr,     // [B, 1, total_dim=576]
    __nv_bfloat16* __restrict__ cache_ptr,             // [B, max_seq_len, total_dim]
    __nv_bfloat16* __restrict__ offload_ptr,           // [B, 1, total_dim]
    __nv_bfloat16* __restrict__ q_pe_ptr,              // [B, H, 1, rope_dim] — modified in-place
    const __nv_bfloat16* __restrict__ cos_ptr,         // [max_pos, rope_dim]
    const __nv_bfloat16* __restrict__ sin_ptr,         // [max_pos, rope_dim]
    const int64_t* __restrict__ position_ids_ptr,      // [B, 1]
    const __nv_bfloat16* __restrict__ norm_weight_ptr, // [kv_lora_rank]
    int B, int H, int max_seq_len,
    int kv_lora_rank,    // 512
    int rope_dim,        // 64
    float eps
) {
    int batch = blockIdx.x;
    if (batch >= B) return;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;  // 256

    int total_dim = kv_lora_rank + rope_dim;  // 576
    int half_rope = rope_dim / 2;  // 32
    int64_t pos_id = position_ids_ptr[batch];

    // Shared memory: [total_dim] for processed KV + [1] for inv_rms
    extern __shared__ char smem_raw[];
    float* smem_float = reinterpret_cast<float*>(smem_raw);  // for reduction
    __nv_bfloat16* smem_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 256 * sizeof(float));

    // Input pointer for this batch
    const __nv_bfloat16* kv_in = new_kv_ptr + batch * total_dim;

    // ══════════════ Stage 1: RMSNorm on KV lora slice [0:kv_lora_rank] ══════════════

    // Compute sum of squares (parallel reduction across threads)
    float local_sq = 0.0f;
    for (int i = tid; i < kv_lora_rank; i += nthreads) {
        float val = __bfloat162float(kv_in[i]);
        local_sq += val * val;
    }

    // Block-level reduction via shared memory
    smem_float[tid] = local_sq;
    __syncthreads();

    // Tree reduction
    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem_float[tid] += smem_float[tid + stride];
        }
        __syncthreads();
    }

    float variance = smem_float[0] / kv_lora_rank;
    float inv_rms = rsqrtf(variance + eps);

    // Apply normalization and write to smem_kv
    for (int i = tid; i < kv_lora_rank; i += nthreads) {
        float val = __bfloat162float(kv_in[i]);
        float w = __bfloat162float(norm_weight_ptr[i]);
        smem_kv[i] = __float2bfloat16(val * inv_rms * w);
    }

    // ══════════════ Stage 2: RoPE on KV rope slice [kv_lora_rank:total_dim] ══════════════

    // cos/sin for this position
    int cos_sin_base = pos_id * rope_dim;

    if (tid < half_rope) {
        int even_idx = tid * 2;
        int odd_idx = tid * 2 + 1;

        float kv_even = __bfloat162float(kv_in[kv_lora_rank + even_idx]);
        float kv_odd  = __bfloat162float(kv_in[kv_lora_rank + odd_idx]);

        float cos_first  = __bfloat162float(cos_ptr[cos_sin_base + tid]);
        float sin_first  = __bfloat162float(sin_ptr[cos_sin_base + tid]);
        float cos_second = __bfloat162float(cos_ptr[cos_sin_base + half_rope + tid]);
        float sin_second = __bfloat162float(sin_ptr[cos_sin_base + half_rope + tid]);

        // Rotary: first half and second half
        smem_kv[kv_lora_rank + tid]            = __float2bfloat16(kv_even * cos_first - kv_odd * sin_first);
        smem_kv[kv_lora_rank + half_rope + tid] = __float2bfloat16(kv_odd * cos_second + kv_even * sin_second);
    }
    __syncthreads();

    // ══════════════ Stage 3: Write processed KV to cache + offload ══════════════

    // Cache write: cache[batch, pos_id, :] = smem_kv[:]
    __nv_bfloat16* cache_dst = cache_ptr + batch * max_seq_len * total_dim + pos_id * total_dim;
    __nv_bfloat16* offload_dst = offload_ptr + batch * total_dim;

    for (int i = tid; i < total_dim; i += nthreads) {
        cache_dst[i] = smem_kv[i];
        offload_dst[i] = smem_kv[i];
    }

    // ══════════════ Stage 4: RoPE on all Q heads (in-place) ══════════════

    // q_pe layout: [B, H, 1, rope_dim] contiguous
    // Each head has rope_dim values. We have H=64 heads × rope_dim=64 = 4096 values total.
    // With 256 threads: 4096/256 = 16 values per thread

    int q_pe_batch_stride = H * rope_dim;

    // Process all heads
    for (int idx = tid; idx < H * half_rope; idx += nthreads) {
        int head = idx / half_rope;
        int r = idx % half_rope;

        int even_idx = r * 2;
        int odd_idx = r * 2 + 1;

        int q_base = batch * q_pe_batch_stride + head * rope_dim;
        float q_even = __bfloat162float(q_pe_ptr[q_base + even_idx]);
        float q_odd  = __bfloat162float(q_pe_ptr[q_base + odd_idx]);

        float cos_first  = __bfloat162float(cos_ptr[cos_sin_base + r]);
        float sin_first  = __bfloat162float(sin_ptr[cos_sin_base + r]);
        float cos_second = __bfloat162float(cos_ptr[cos_sin_base + half_rope + r]);
        float sin_second = __bfloat162float(sin_ptr[cos_sin_base + half_rope + r]);

        q_pe_ptr[q_base + r]            = __float2bfloat16(q_even * cos_first - q_odd * sin_first);
        q_pe_ptr[q_base + half_rope + r] = __float2bfloat16(q_odd * cos_second + q_even * sin_second);
    }
}

torch::Tensor fused_kv_norm_rope_cache_forward(
    torch::Tensor new_kv,           // [B, 1, 576]
    torch::Tensor cache,            // [B, max_seq_len, 576]
    torch::Tensor q_pe,             // [B, H, 1, rope_dim]
    torch::Tensor cos_cache,        // [max_pos, rope_dim]
    torch::Tensor sin_cache,        // [max_pos, rope_dim]
    torch::Tensor position_ids,     // [B, 1]
    torch::Tensor norm_weight,      // [kv_lora_rank]
    int kv_lora_rank,
    int rope_dim,
    float eps
) {
    int B = new_kv.size(0);
    int total_dim = kv_lora_rank + rope_dim;
    int H = q_pe.size(1);
    int max_seq_len = cache.size(1);

    auto offload = torch::empty({B, 1, total_dim}, new_kv.options());

    int threads = 256;
    // Shared memory: [256] floats for reduction + [total_dim] bf16 for KV
    int smem_bytes = threads * sizeof(float) + total_dim * sizeof(__nv_bfloat16);

    fused_kv_norm_rope_cache_kernel<<<B, threads, smem_bytes>>>(
        reinterpret_cast<const __nv_bfloat16*>(new_kv.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(cache.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(offload.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(cos_cache.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(sin_cache.data_ptr<at::BFloat16>()),
        position_ids.data_ptr<int64_t>(),
        reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
        B, H, max_seq_len, kv_lora_rank, rope_dim, eps
    );

    return offload;
}
"""

_CPP_SRC = r"""
torch::Tensor fused_kv_norm_rope_cache_forward(
    torch::Tensor new_kv,
    torch::Tensor cache,
    torch::Tensor q_pe,
    torch::Tensor cos_cache,
    torch::Tensor sin_cache,
    torch::Tensor position_ids,
    torch::Tensor norm_weight,
    int kv_lora_rank,
    int rope_dim,
    float eps
);
"""

_module = None


def _load():
    global _module
    if _module is not None:
        return _module
    _module = load_inline(
        name="fused_kv_norm_rope_cache_cuda",
        cpp_sources=[_CPP_SRC],
        cuda_sources=[_CUDA_SRC],
        functions=["fused_kv_norm_rope_cache_forward"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        verbose=False,
    )
    return _module


def fused_kv_norm_rope_cache_cuda(
    new_compressed_kv: torch.Tensor,  # [bsz, 1, 576]
    flat_cache: torch.Tensor,         # [bsz, max_seq_len, 576] — flat KV cache
    q_pe: torch.Tensor,               # [bsz, H, 1, rope_dim] — modified in-place
    cos: torch.Tensor,                # [max_pos, rope_dim]
    sin: torch.Tensor,                # [max_pos, rope_dim]
    position_ids: torch.Tensor,       # [bsz, 1]
    norm_weight: torch.Tensor,        # [kv_lora_rank]
    kv_lora_rank: int = 512,
    rope_dim: int = 64,
    eps: float = 1e-6,
) -> torch.Tensor:
    """CUDA fused RMSNorm + RoPE + cache write.

    Equivalent to fused_rmsnorm_rope_cache_update_with_q_return_new_kv (Triton).
    Returns offload_kv [bsz, 1, 576].
    """
    mod = _load()
    return mod.fused_kv_norm_rope_cache_forward(
        new_compressed_kv.contiguous(),
        flat_cache.contiguous(),
        q_pe.contiguous(),
        cos.contiguous(),
        sin.contiguous(),
        position_ids.contiguous(),
        norm_weight.contiguous(),
        kv_lora_rank,
        rope_dim,
        eps,
    )
