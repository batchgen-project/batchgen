#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "attention_ops.h"

// ============================================================================
// QKV Split kernel: read packed QKV once, write Q/K/V to pre-allocated tensors
// Uses vectorized float4 loads/stores (8 BF16 per transaction)
// ============================================================================

template <typename T>
__global__ void qkv_split_kernel(
    const T* __restrict__ qkv,     // [M, total_dim]
    T* __restrict__ q_out,         // [M, q_size]
    T* __restrict__ k_out,         // [M, kv_size]
    T* __restrict__ v_out,         // [M, kv_size]
    int M,
    int q_size,
    int kv_size,
    int total_dim)
{
    int row = blockIdx.x;
    if (row >= M) return;

    const T* qkv_row = qkv + row * total_dim;
    T* q_row = q_out + row * q_size;
    T* k_row = k_out + row * kv_size;
    T* v_row = v_out + row * kv_size;

    // Vectorized copy: 8 BF16 = 16 bytes = float4
    // q_size=4096, kv_size=512 are all divisible by 8
    constexpr int VEC_SIZE = 8;  // elements per float4 for BF16/FP16
    int q_vecs = q_size / VEC_SIZE;
    int kv_vecs = kv_size / VEC_SIZE;

    // Copy Q portion vectorized
    const float4* qkv_vec = reinterpret_cast<const float4*>(qkv_row);
    float4* q_vec = reinterpret_cast<float4*>(q_row);
    for (int i = threadIdx.x; i < q_vecs; i += blockDim.x) {
        q_vec[i] = qkv_vec[i];
    }

    // Copy K portion vectorized
    const float4* k_src = reinterpret_cast<const float4*>(qkv_row + q_size);
    float4* k_vec = reinterpret_cast<float4*>(k_row);
    for (int i = threadIdx.x; i < kv_vecs; i += blockDim.x) {
        k_vec[i] = k_src[i];
    }

    // Copy V portion vectorized
    const float4* v_src = reinterpret_cast<const float4*>(qkv_row + q_size + kv_size);
    float4* v_vec = reinterpret_cast<float4*>(v_row);
    for (int i = threadIdx.x; i < kv_vecs; i += blockDim.x) {
        v_vec[i] = v_src[i];
    }
}

// ============================================================================
// Host function: in-place version (caller provides output tensors)
// Zero allocation overhead — just launch the kernel
// ============================================================================

void qkv_split_inplace(
    torch::Tensor qkv,     // [M, total_dim]
    torch::Tensor q_out,   // [M, q_size] pre-allocated
    torch::Tensor k_out,   // [M, kv_size] pre-allocated
    torch::Tensor v_out,   // [M, kv_size] pre-allocated
    int q_size,
    int kv_size)
{
    int total_dim = qkv.size(-1);
    int M = qkv.numel() / total_dim;

    if (M == 0) return;

    const int threads = 256;
    const int blocks = M;

    AT_DISPATCH_SWITCH(qkv.scalar_type(),
        "qkv_split_inplace",
        AT_DISPATCH_CASE(at::ScalarType::BFloat16,
            [&] { qkv_split_kernel<at::BFloat16><<<blocks, threads>>>(
                qkv.data_ptr<at::BFloat16>(),
                q_out.data_ptr<at::BFloat16>(),
                k_out.data_ptr<at::BFloat16>(),
                v_out.data_ptr<at::BFloat16>(),
                M, q_size, kv_size, total_dim); })
        AT_DISPATCH_CASE(at::ScalarType::Half,
            [&] { qkv_split_kernel<at::Half><<<blocks, threads>>>(
                qkv.data_ptr<at::Half>(),
                q_out.data_ptr<at::Half>(),
                k_out.data_ptr<at::Half>(),
                v_out.data_ptr<at::Half>(),
                M, q_size, kv_size, total_dim); })
    );
}

// ============================================================================
// Host function: allocating version (for convenience/testing)
// ============================================================================

std::vector<torch::Tensor> qkv_split_forward(
    torch::Tensor qkv,
    int q_size,
    int kv_size)
{
    TORCH_CHECK(qkv.is_cuda(), "qkv must be CUDA");

    auto orig_shape = qkv.sizes().vec();
    int total_dim = orig_shape.back();
    TORCH_CHECK(total_dim == q_size + 2 * kv_size,
                "total_dim must equal q_size + 2*kv_size");

    int M = qkv.numel() / total_dim;

    if (M == 0) {
        auto q_shape = orig_shape;
        auto kv_shape = orig_shape;
        q_shape.back() = q_size;
        kv_shape.back() = kv_size;
        return {
            torch::empty(q_shape, qkv.options()),
            torch::empty(kv_shape, qkv.options()),
            torch::empty(kv_shape, qkv.options()),
        };
    }

    auto qkv_flat = qkv.reshape({M, total_dim});

    auto q_out = torch::empty({M, q_size}, qkv.options());
    auto k_out = torch::empty({M, kv_size}, qkv.options());
    auto v_out = torch::empty({M, kv_size}, qkv.options());

    qkv_split_inplace(qkv_flat, q_out, k_out, v_out, q_size, kv_size);

    auto q_shape = orig_shape;
    auto kv_shape = orig_shape;
    q_shape.back() = q_size;
    kv_shape.back() = kv_size;

    return {
        q_out.reshape(q_shape),
        k_out.reshape(kv_shape),
        v_out.reshape(kv_shape),
    };
}
