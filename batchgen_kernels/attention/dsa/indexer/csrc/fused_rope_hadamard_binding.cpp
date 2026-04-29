/******************************************************************************
 * C++ binding for fused interleaved RoPE + Hadamard transform.
 * Specialized for dim=128, bf16.
 ******************************************************************************/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

// Forward declaration (defined in fused_rope_hadamard.cu)
struct FusedRopeHadamardParams {
    using index_t = int64_t;
    void *__restrict__ x_ptr;
    void *__restrict__ out_ptr;
    const float *__restrict__ cos_ptr;
    const float *__restrict__ sin_ptr;
    const int64_t *__restrict__ pos_ptr;
    int batch;
    int cos_stride;
    float scale;
};

void fused_rope_hadamard_launch(FusedRopeHadamardParams &params, cudaStream_t stream);

at::Tensor fused_rope_hadamard(
    at::Tensor &x,          // [batch, 128] bf16, after LayerNorm
    at::Tensor &cos_cache,  // [max_seq, 64] float32
    at::Tensor &sin_cache,  // [max_seq, 64] float32
    at::Tensor &positions,  // [batch] int64
    float scale             // 1/sqrt(128)
) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::ScalarType::BFloat16);
    TORCH_CHECK(x.size(-1) == 128, "fused_rope_hadamard requires dim=128");
    TORCH_CHECK(cos_cache.scalar_type() == at::ScalarType::Float, "cos must be float32");
    TORCH_CHECK(sin_cache.scalar_type() == at::ScalarType::Float, "sin must be float32");
    TORCH_CHECK(positions.scalar_type() == at::ScalarType::Long, "positions must be int64");

    const auto shapes_og = x.sizes();
    auto x_2d = x.reshape({-1, 128});
    if (x_2d.stride(-1) != 1) { x_2d = x_2d.contiguous(); }
    const int batch_size = x_2d.size(0);

    // Ensure cos/sin are contiguous
    auto cos_contig = cos_cache.contiguous();
    auto sin_contig = sin_cache.contiguous();
    auto pos_contig = positions.reshape({-1}).contiguous();

    at::Tensor out = torch::empty_like(x_2d);

    FusedRopeHadamardParams params;
    memset(&params, 0, sizeof(params));
    params.x_ptr = x_2d.data_ptr();
    params.out_ptr = out.data_ptr();
    params.cos_ptr = cos_contig.data_ptr<float>();
    params.sin_ptr = sin_contig.data_ptr<float>();
    params.pos_ptr = pos_contig.data_ptr<int64_t>();
    params.batch = batch_size;
    params.cos_stride = cos_contig.stride(0);
    params.scale = scale;

    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    fused_rope_hadamard_launch(params, stream);

    return out.reshape(shapes_og);
}

void fused_rope_hadamard_out(
    at::Tensor &x,          // [batch, 128] bf16, after LayerNorm
    at::Tensor &cos_cache,  // [max_seq, 64] float32
    at::Tensor &sin_cache,  // [max_seq, 64] float32
    at::Tensor &positions,  // [batch] int64
    at::Tensor &out,        // [batch, 128] bf16
    float scale             // 1/sqrt(128)
) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::ScalarType::BFloat16);
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::ScalarType::BFloat16);
    TORCH_CHECK(x.sizes() == out.sizes(), "out must match x shape");
    TORCH_CHECK(x.size(-1) == 128, "fused_rope_hadamard requires dim=128");
    TORCH_CHECK(cos_cache.scalar_type() == at::ScalarType::Float, "cos must be float32");
    TORCH_CHECK(sin_cache.scalar_type() == at::ScalarType::Float, "sin must be float32");
    TORCH_CHECK(positions.scalar_type() == at::ScalarType::Long, "positions must be int64");
    TORCH_CHECK(cos_cache.is_contiguous(), "cos_cache must be contiguous for graph out path");
    TORCH_CHECK(sin_cache.is_contiguous(), "sin_cache must be contiguous for graph out path");
    TORCH_CHECK(positions.is_contiguous(), "positions must be contiguous for graph out path");

    auto x_2d = x.reshape({-1, 128});
    auto out_2d = out.reshape({-1, 128});
    TORCH_CHECK(x_2d.stride(-1) == 1, "x last dimension must be contiguous");
    TORCH_CHECK(out_2d.stride(-1) == 1, "out last dimension must be contiguous");
    const int batch_size = x_2d.size(0);
    TORCH_CHECK(positions.numel() == batch_size, "positions length must match flattened batch");

    FusedRopeHadamardParams params;
    memset(&params, 0, sizeof(params));
    params.x_ptr = x_2d.data_ptr();
    params.out_ptr = out_2d.data_ptr();
    params.cos_ptr = cos_cache.data_ptr<float>();
    params.sin_ptr = sin_cache.data_ptr<float>();
    params.pos_ptr = positions.data_ptr<int64_t>();
    params.batch = batch_size;
    params.cos_stride = cos_cache.stride(0);
    params.scale = scale;

    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    fused_rope_hadamard_launch(params, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_rope_hadamard", &fused_rope_hadamard,
          "Fused interleaved RoPE + Hadamard transform (dim=128, bf16)");
    m.def("fused_rope_hadamard_out", &fused_rope_hadamard_out,
          "Out-buffer fused interleaved RoPE + Hadamard transform (dim=128, bf16)");
}
