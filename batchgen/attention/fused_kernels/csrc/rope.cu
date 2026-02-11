#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "attention_ops.h"

// ============================================================================
// RoPE kernel: one thread per half_dim element, one block per head-vector
// ============================================================================

template <typename T>
__global__ void rope_kernel(
    const T* __restrict__ query,    // [total_q, head_dim]
    const T* __restrict__ key,      // [total_k, head_dim]
    const T* __restrict__ cos,      // [B * S, head_dim]
    const T* __restrict__ sin,      // [B * S, head_dim]
    T* __restrict__ q_out,
    T* __restrict__ k_out,
    int total_q,                    // B * S * num_q_heads
    int total_k,                    // B * S * num_kv_heads
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int half_dim,
    int head_dim)
{
    int vec_idx = blockIdx.x;
    bool is_q = vec_idx < total_q;

    const T* x_ptr;
    T* o_ptr;
    int local_idx;
    int num_heads;

    if (is_q) {
        x_ptr = query;
        o_ptr = q_out;
        local_idx = vec_idx;
        num_heads = num_q_heads;
    } else {
        x_ptr = key;
        o_ptr = k_out;
        local_idx = vec_idx - total_q;
        num_heads = num_kv_heads;
    }

    // batch_idx from local_idx: local_idx = batch * seq * heads + seq_pos * heads + head
    int batch_seq_idx = local_idx / num_heads;  // = batch * seq + seq_pos

    // Load cos/sin for this batch+seq position (only half_dim elements)
    int cos_base = batch_seq_idx * head_dim;

    int tid = threadIdx.x;
    if (tid >= half_dim) return;

    // Load x1 (first half) and x2 (second half)
    int x_base = local_idx * head_dim;
    T x1 = x_ptr[x_base + tid];
    T x2 = x_ptr[x_base + half_dim + tid];

    T c = cos[cos_base + tid];
    T s = sin[cos_base + tid];

    // YaRN half-dim rotation.
    // Match PyTorch precision: each multiply rounds to T independently,
    // then subtract/add in T. This avoids FP32-intermediate precision
    // differences vs PyTorch's elementwise ops.
    //
    // out[..., :half] = x1*cos - x2*sin
    // out[..., half:] = x2*cos + x1*sin
    T x1c = static_cast<T>(static_cast<float>(x1) * static_cast<float>(c));
    T x2s = static_cast<T>(static_cast<float>(x2) * static_cast<float>(s));
    T x2c = static_cast<T>(static_cast<float>(x2) * static_cast<float>(c));
    T x1s = static_cast<T>(static_cast<float>(x1) * static_cast<float>(s));

    o_ptr[x_base + tid] = static_cast<T>(static_cast<float>(x1c) - static_cast<float>(x2s));
    o_ptr[x_base + half_dim + tid] = static_cast<T>(static_cast<float>(x2c) + static_cast<float>(x1s));
}

// ============================================================================
// Host function
// ============================================================================

std::vector<torch::Tensor> rope_forward(
    torch::Tensor query,    // [B, S, num_heads, head_dim]
    torch::Tensor key,      // [B, S, num_kv_heads, head_dim]
    torch::Tensor cos,      // [B, S, head_dim]
    torch::Tensor sin,      // [B, S, head_dim]
    int half_dim)
{
    TORCH_CHECK(query.is_cuda(), "query must be CUDA");
    TORCH_CHECK(key.is_cuda(), "key must be CUDA");

    query = query.contiguous();
    key = key.contiguous();
    cos = cos.contiguous();
    sin = sin.contiguous();

    int B = query.size(0);
    int S = query.size(1);
    int num_q_heads = query.size(2);
    int head_dim = query.size(3);
    int num_kv_heads = key.size(2);

    int total_q = B * S * num_q_heads;
    int total_k = B * S * num_kv_heads;
    int total = total_q + total_k;

    // Flatten to [total_vectors, head_dim]
    auto q_flat = query.reshape({total_q, head_dim});
    auto k_flat = key.reshape({total_k, head_dim});
    auto cos_flat = cos.reshape({-1, head_dim});
    auto sin_flat = sin.reshape({-1, head_dim});

    auto q_out = torch::empty_like(q_flat);
    auto k_out = torch::empty_like(k_flat);

    // One block per vector, half_dim threads per block
    // For head_dim=64, half_dim=32 → 32 threads per block
    const int threads = half_dim;
    const int blocks = total;

    AT_DISPATCH_SWITCH(query.scalar_type(),
        "rope_forward",
        AT_DISPATCH_CASE(at::ScalarType::BFloat16,
            [&] { rope_kernel<at::BFloat16><<<blocks, threads>>>(
                q_flat.data_ptr<at::BFloat16>(),
                k_flat.data_ptr<at::BFloat16>(),
                cos_flat.data_ptr<at::BFloat16>(),
                sin_flat.data_ptr<at::BFloat16>(),
                q_out.data_ptr<at::BFloat16>(),
                k_out.data_ptr<at::BFloat16>(),
                total_q, total_k,
                num_q_heads, num_kv_heads,
                S, half_dim, head_dim); })
        AT_DISPATCH_CASE(at::ScalarType::Half,
            [&] { rope_kernel<at::Half><<<blocks, threads>>>(
                q_flat.data_ptr<at::Half>(),
                k_flat.data_ptr<at::Half>(),
                cos_flat.data_ptr<at::Half>(),
                sin_flat.data_ptr<at::Half>(),
                q_out.data_ptr<at::Half>(),
                k_out.data_ptr<at::Half>(),
                total_q, total_k,
                num_q_heads, num_kv_heads,
                S, half_dim, head_dim); })
    );

    return {
        q_out.reshape_as(query),
        k_out.reshape_as(key)
    };
}
