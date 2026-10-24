/******************************************************************************
 * C++ binding for fused interleaved RoPE + Hadamard transform.
 * Specialized for dim=128, bf16.
 ******************************************************************************/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <optional>

// Forward declaration (defined in fused_rope_hadamard.cu)
struct FusedRopeHadamardParams {
    using index_t = int64_t;
    void *__restrict__ x_ptr;
    void *__restrict__ out_ptr;
    void *__restrict__ out_fp8_ptr;
    const float *__restrict__ cos_ptr;
    const float *__restrict__ sin_ptr;
    const int64_t *__restrict__ pos_ptr;
    int batch;
    int cos_stride;
    float scale;
};

void fused_rope_hadamard_launch(FusedRopeHadamardParams &params,
                               bool apply_hadamard, cudaStream_t stream);

void fused_rope_hadamard_strided_launch(FusedRopeHadamardParams &params,
                                        int64_t x_row_stride,
                                        int x_heads_per_row,
                                        bool apply_hadamard,
                                        cudaStream_t stream);

at::Tensor fused_rope_hadamard(
    at::Tensor &x,          // [batch, 128] bf16, after LayerNorm
    at::Tensor &cos_cache,  // [max_seq, 64] float32
    at::Tensor &sin_cache,  // [max_seq, 64] float32
    at::Tensor &positions,  // [batch] int64
    float scale,            // 1/sqrt(128)
    bool apply_hadamard = true
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
    fused_rope_hadamard_launch(params, apply_hadamard, stream);

    return out.reshape(shapes_og);
}

void fused_rope_hadamard_out(
    at::Tensor &x,          // [batch, 128] bf16, after LayerNorm
    at::Tensor &cos_cache,  // [max_seq, 64] float32
    at::Tensor &sin_cache,  // [max_seq, 64] float32
    at::Tensor &positions,  // [batch] int64
    at::Tensor &out,        // [batch, 128] bf16
    float scale,            // 1/sqrt(128)
    std::optional<at::Tensor> out_fp8 = std::nullopt,  // [batch, 128] e4m3
    bool apply_hadamard = true
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
    if (out_fp8.has_value()) {
        const at::Tensor &q_fp8 = out_fp8.value();
        TORCH_CHECK(q_fp8.is_cuda(), "out_fp8 must be a CUDA tensor");
        TORCH_CHECK(q_fp8.scalar_type() == at::ScalarType::Float8_e4m3fn,
                    "out_fp8 must be float8_e4m3fn");
        TORCH_CHECK(q_fp8.is_contiguous(), "out_fp8 must be contiguous");
        TORCH_CHECK(q_fp8.numel() == x_2d.numel(),
                    "out_fp8 must hold exactly as many elements as x");
        params.out_fp8_ptr = q_fp8.data_ptr();
    }
    params.cos_ptr = cos_cache.data_ptr<float>();
    params.sin_ptr = sin_cache.data_ptr<float>();
    params.pos_ptr = positions.data_ptr<int64_t>();
    params.batch = batch_size;
    params.cos_stride = cos_cache.stride(0);
    params.scale = scale;

    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    fused_rope_hadamard_launch(params, apply_hadamard, stream);
}

// Strided-source version of ``fused_rope_hadamard_out`` (kernel version
// management: the function above is unchanged). ``x`` is [rows, heads, 128]
// with its heads packed (stride 128) and an explicit row stride, e.g. the
// trailing column slice of the merged [q_b | wq_b] GEMM output. It is read in
// place -- no reshape, no copy. ``out`` / ``out_fp8`` stay contiguous.
void fused_rope_hadamard_strided_out(
    at::Tensor &x,          // [rows, heads, 128] bf16, strides (x_row_stride, 128, 1)
    at::Tensor &cos_cache,  // [max_seq, 64] float32
    at::Tensor &sin_cache,  // [max_seq, 64] float32
    at::Tensor &positions,  // [rows * heads] int64
    at::Tensor &out,        // [rows, heads, 128] bf16, contiguous
    int64_t x_row_stride,   // elements between consecutive rows of x
    float scale,            // 1/sqrt(128)
    std::optional<at::Tensor> out_fp8 = std::nullopt,  // [rows * heads * 128] e4m3
    bool apply_hadamard = true
) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::ScalarType::BFloat16,
                "x must be a CUDA bf16 tensor");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::ScalarType::BFloat16,
                "out must be a CUDA bf16 tensor");
    TORCH_CHECK(x.dim() == 3, "strided x must be [rows, heads, 128]");
    TORCH_CHECK(x.size(2) == 128, "fused_rope_hadamard requires dim=128");
    TORCH_CHECK(x.sizes() == out.sizes(), "out must match x shape");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(x.stride(2) == 1 && x.stride(1) == 128,
                "strided x must have packed heads: strides (row, 128, 1)");
    TORCH_CHECK(x.stride(0) == x_row_stride,
                "x_row_stride does not match x.stride(0)");
    TORCH_CHECK(x_row_stride >= x.size(1) * 128,
                "x_row_stride is smaller than one row of heads");
    // 16-byte vector loads of 8 bf16: every row start must stay aligned.
    TORCH_CHECK(x_row_stride % 8 == 0, "x_row_stride must be a multiple of 8");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
                "x must be 16-byte aligned");
    TORCH_CHECK(cos_cache.scalar_type() == at::ScalarType::Float, "cos must be float32");
    TORCH_CHECK(sin_cache.scalar_type() == at::ScalarType::Float, "sin must be float32");
    TORCH_CHECK(positions.scalar_type() == at::ScalarType::Long, "positions must be int64");
    TORCH_CHECK(cos_cache.is_contiguous(), "cos_cache must be contiguous for graph out path");
    TORCH_CHECK(sin_cache.is_contiguous(), "sin_cache must be contiguous for graph out path");
    TORCH_CHECK(positions.is_contiguous(), "positions must be contiguous for graph out path");

    const int64_t flat_rows = x.size(0) * x.size(1);
    TORCH_CHECK(positions.numel() == flat_rows, "positions length must match rows * heads");

    FusedRopeHadamardParams params;
    memset(&params, 0, sizeof(params));
    params.x_ptr = x.data_ptr();
    params.out_ptr = out.data_ptr();
    if (out_fp8.has_value()) {
        const at::Tensor &q_fp8 = out_fp8.value();
        TORCH_CHECK(q_fp8.is_cuda(), "out_fp8 must be a CUDA tensor");
        TORCH_CHECK(q_fp8.scalar_type() == at::ScalarType::Float8_e4m3fn,
                    "out_fp8 must be float8_e4m3fn");
        TORCH_CHECK(q_fp8.is_contiguous(), "out_fp8 must be contiguous");
        TORCH_CHECK(q_fp8.numel() == out.numel(),
                    "out_fp8 must hold exactly as many elements as x");
        params.out_fp8_ptr = q_fp8.data_ptr();
    }
    params.cos_ptr = cos_cache.data_ptr<float>();
    params.sin_ptr = sin_cache.data_ptr<float>();
    params.pos_ptr = positions.data_ptr<int64_t>();
    params.batch = static_cast<int>(flat_rows);
    params.cos_stride = cos_cache.stride(0);
    params.scale = scale;

    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    fused_rope_hadamard_strided_launch(
        params, x_row_stride, static_cast<int>(x.size(1)), apply_hadamard, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_rope_hadamard", &fused_rope_hadamard,
          "Fused interleaved RoPE + optional Hadamard transform (dim=128, bf16)",
          pybind11::arg("x"), pybind11::arg("cos_cache"), pybind11::arg("sin_cache"),
          pybind11::arg("positions"), pybind11::arg("scale"),
          pybind11::arg("apply_hadamard") = true);
    m.def("fused_rope_hadamard_out", &fused_rope_hadamard_out,
          "Out-buffer fused interleaved RoPE + Hadamard transform (dim=128, bf16)",
          pybind11::arg("x"), pybind11::arg("cos_cache"), pybind11::arg("sin_cache"),
          pybind11::arg("positions"), pybind11::arg("out"), pybind11::arg("scale"),
          pybind11::arg("out_fp8") = std::nullopt,
          pybind11::arg("apply_hadamard") = true);
    m.def("fused_rope_hadamard_strided_out", &fused_rope_hadamard_strided_out,
          "Out-buffer fused RoPE + Hadamard over a strided [rows, heads, 128] source",
          pybind11::arg("x"), pybind11::arg("cos_cache"), pybind11::arg("sin_cache"),
          pybind11::arg("positions"), pybind11::arg("out"),
          pybind11::arg("x_row_stride"), pybind11::arg("scale"),
          pybind11::arg("out_fp8") = std::nullopt,
          pybind11::arg("apply_hadamard") = true);
}
