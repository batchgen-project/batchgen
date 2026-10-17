// Fused quantize (bf16 -> e4m3) + page-split write of the DSA indexer K cache.
//
// One page of a page-split cache is the SoA arrangement
// deep_gemm.fp8_paged_mqa_logits reads:
//
//     [ page_size * data_dim  e4m3 K bytes ][ page_size * 4  fp32 scales ]
//
// Token `o` of a page owns [o * data_dim, (o + 1) * data_dim) of the K region
// and its single fp32 scale at page_size * data_dim + o * 4.
//
// One warp per token: warp-reduce amax over the head dim, scale =
// max(amax, 1e-4) / 448, e4m3 = cvt(k * (1 / scale)). The reciprocal MULTIPLY
// (not a divide) and the amax floor are the byte-exactness contract with
// `quantize_indexer_k_fp8` and with DeepGEMM's own cast.
//
// The kernel resolves each token's physical page through the GPU page table,
// exactly like `run_paged_kv_token_update_fused`, so it is a drop-in
// replacement for the bf16 AoS writer inside a captured CUDA graph (no host
// index math, no extra launch).
//
// It optionally also emits the host-append sources of the SAME quantization:
// `k_fp8_out` [N, data_dim] e4m3 and `scales_out` [N] fp32. The host KV region
// takes those two tensors (see HostPagedKVWorkerView's two-tensor convention),
// so the D2H offload reuses this kernel's bytes instead of re-quantizing.

#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kWarpSize = 32;
constexpr int kScaleBytesPerToken = 4;
// data_dim <= 256 and data_dim % 32 == 0, so a lane holds at most 8 elements.
constexpr int kMaxValsPerLane = 8;
constexpr int kMaxDataDim = kWarpSize * kMaxValsPerLane;
constexpr float kAmaxFloor = 1e-4f;
constexpr float kFp8Max = 448.0f;

__global__ void fused_indexer_k_write_fp8_kernel(
    const __nv_bfloat16* __restrict__ k_src,        // [N, data_dim]
    const int32_t* __restrict__ page_table,         // [num_slots, table_cols]
    const int32_t* __restrict__ slot_indices,       // [N]
    const int32_t* __restrict__ token_indices,      // [N]
    const int32_t* __restrict__ num_valid_tokens,   // [1], may be nullptr
    uint8_t* __restrict__ cache,                    // [num_pages, page_bytes]
    uint8_t* __restrict__ k_fp8_out,                // [N, data_dim], may be nullptr
    float* __restrict__ scales_out,                 // [N], may be nullptr
    int num_tokens,
    int data_dim,
    int page_size,
    int page_table_cols,
    int num_pages) {
    const int token = blockIdx.x;
    if (token >= num_tokens) {
        return;
    }
    const int lane = threadIdx.x;
    const int per_lane = data_dim / kWarpSize;
    const __nv_bfloat16* krow =
        k_src + static_cast<long long>(token) * data_dim;

    float vals[kMaxValsPerLane];
    float local_max = 0.0f;
#pragma unroll
    for (int i = 0; i < kMaxValsPerLane; ++i) {
        if (i < per_lane) {
            const float v = __bfloat162float(krow[lane * per_lane + i]);
            vals[i] = v;
            // fmaxf ignores a NaN operand, so a stale/garbage row cannot turn
            // the scale itself into NaN; it degenerates to the amax floor.
            local_max = fmaxf(local_max, fabsf(v));
        }
    }
#pragma unroll
    for (int off = kWarpSize / 2; off >= 1; off >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, off));
    }

    const float scale = fmaxf(local_max, kAmaxFloor) / kFp8Max;
    const float inv = 1.0f / scale;

    // Host-append sources are row-local and are written for EVERY row of the
    // launch, matching what the bf16 writer staged (a plain copy of all rows):
    // row validity is decided later, by the sequence set the host append is
    // called with, not here.
    if (k_fp8_out != nullptr) {
        uint8_t* dst = k_fp8_out + static_cast<long long>(token) * data_dim;
#pragma unroll
        for (int i = 0; i < kMaxValsPerLane; ++i) {
            if (i < per_lane) {
                dst[lane * per_lane + i] = __nv_cvt_float_to_fp8(
                    vals[i] * inv, __NV_SATFINITE, __NV_E4M3);
            }
        }
    }
    if (scales_out != nullptr && lane == 0) {
        scales_out[token] = scale;
    }

    // Paged-cache write: same row gating as the bf16 page-table writer --
    // rows beyond num_valid_tokens and the -1 slot sentinel (CUDA graph
    // bucket padding) write nothing.
    if (num_valid_tokens != nullptr && token >= *num_valid_tokens) {
        return;
    }
    const int slot = slot_indices[token];
    if (slot < 0) {
        return;
    }
    const int token_index = token_indices[token];
    const int page_slot = token_index / page_size;
    const int offset = token_index - page_slot * page_size;
    if (page_slot >= page_table_cols) {
        return;
    }
    const int page =
        page_table[static_cast<long long>(slot) * page_table_cols + page_slot];
    // An unallocated page-table entry (-1) for an otherwise valid row is a
    // caller bug; the row is dropped rather than written through a negative
    // page base.
    if (page < 0 || page >= num_pages) {
        return;
    }

    const long long page_bytes =
        static_cast<long long>(page_size) * (data_dim + kScaleBytesPerToken);
    uint8_t* page_base = cache + page * page_bytes;
    uint8_t* kdst = page_base + static_cast<long long>(offset) * data_dim;
#pragma unroll
    for (int i = 0; i < kMaxValsPerLane; ++i) {
        if (i < per_lane) {
            kdst[lane * per_lane + i] = __nv_cvt_float_to_fp8(
                vals[i] * inv, __NV_SATFINITE, __NV_E4M3);
        }
    }
    if (lane == 0) {
        uint8_t* sdst = page_base +
                        static_cast<long long>(page_size) * data_dim +
                        static_cast<long long>(offset) * kScaleBytesPerToken;
        *reinterpret_cast<float*>(sdst) = scale;
    }
}

void CheckCudaInt32(const torch::Tensor& t, const char* name,
                    const torch::Tensor& ref) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.device() == ref.device(), name,
                " must share the device of k_src");
    TORCH_CHECK(t.scalar_type() == torch::kInt32, name, " must be int32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

}  // namespace

void fused_indexer_k_write_fp8(
    torch::Tensor k_src,         // [N, data_dim] bf16
    torch::Tensor page_table,    // [num_slots, table_cols] int32
    torch::Tensor slot_indices,  // [N] int32
    torch::Tensor token_indices, // [N] int32
    torch::Tensor cache,         // [num_pages, page_bytes] 1-byte
    int64_t page_size_tokens,
    c10::optional<torch::Tensor> k_fp8_out,
    c10::optional<torch::Tensor> scales_out,
    c10::optional<torch::Tensor> num_valid_tokens) {
    TORCH_CHECK(k_src.is_cuda(), "k_src must be a CUDA tensor");
    TORCH_CHECK(k_src.scalar_type() == torch::kBFloat16,
                "k_src must be bfloat16");
    TORCH_CHECK(k_src.is_contiguous(), "k_src must be contiguous");
    TORCH_CHECK(k_src.dim() == 2, "k_src must be [N, data_dim]");

    const int num_tokens = static_cast<int>(k_src.size(0));
    const int data_dim = static_cast<int>(k_src.size(1));
    TORCH_CHECK(data_dim % kWarpSize == 0 && data_dim > 0 &&
                    data_dim <= kMaxDataDim,
                "data_dim must be a positive multiple of 32 and <= 256, got ",
                data_dim);
    const int page_size = static_cast<int>(page_size_tokens);
    TORCH_CHECK(page_size > 0, "page_size_tokens must be positive");

    CheckCudaInt32(page_table, "page_table", k_src);
    CheckCudaInt32(slot_indices, "slot_indices", k_src);
    CheckCudaInt32(token_indices, "token_indices", k_src);
    TORCH_CHECK(page_table.dim() == 2,
                "page_table must be [num_slots, max_pages_per_sequence]");
    TORCH_CHECK(slot_indices.size(0) == num_tokens,
                "slot_indices must hold one entry per token");
    TORCH_CHECK(token_indices.size(0) == num_tokens,
                "token_indices must hold one entry per token");

    TORCH_CHECK(cache.is_cuda(), "cache must be a CUDA tensor");
    TORCH_CHECK(cache.device() == k_src.device(),
                "cache must share the device of k_src");
    TORCH_CHECK(cache.is_contiguous(), "cache must be contiguous");
    TORCH_CHECK(cache.element_size() == 1,
                "cache must have a 1-byte element type");
    TORCH_CHECK(cache.dim() == 2, "cache must be [num_pages, page_bytes]");
    const int64_t expected_page_bytes =
        static_cast<int64_t>(page_size) * (data_dim + kScaleBytesPerToken);
    TORCH_CHECK(cache.size(1) == expected_page_bytes,
                "cache page bytes ", cache.size(1), " do not match page_size=",
                page_size, " and data_dim=", data_dim);
    const int num_pages = static_cast<int>(cache.size(0));

    uint8_t* k_fp8_ptr = nullptr;
    if (k_fp8_out.has_value()) {
        const torch::Tensor& out = *k_fp8_out;
        TORCH_CHECK(out.is_cuda() && out.device() == k_src.device(),
                    "k_fp8_out must be a CUDA tensor on k_src's device");
        TORCH_CHECK(out.is_contiguous(), "k_fp8_out must be contiguous");
        TORCH_CHECK(out.element_size() == 1,
                    "k_fp8_out must have a 1-byte element type");
        TORCH_CHECK(out.numel() == static_cast<int64_t>(num_tokens) * data_dim,
                    "k_fp8_out must hold N * data_dim elements");
        k_fp8_ptr = reinterpret_cast<uint8_t*>(out.data_ptr());
    }
    float* scales_ptr = nullptr;
    if (scales_out.has_value()) {
        const torch::Tensor& out = *scales_out;
        TORCH_CHECK(out.is_cuda() && out.device() == k_src.device(),
                    "scales_out must be a CUDA tensor on k_src's device");
        TORCH_CHECK(out.is_contiguous(), "scales_out must be contiguous");
        TORCH_CHECK(out.scalar_type() == torch::kFloat32,
                    "scales_out must be float32");
        TORCH_CHECK(out.numel() == num_tokens,
                    "scales_out must hold one scale per token");
        scales_ptr = out.data_ptr<float>();
    }
    const int32_t* valid_ptr = nullptr;
    if (num_valid_tokens.has_value()) {
        const torch::Tensor& nv = *num_valid_tokens;
        CheckCudaInt32(nv, "num_valid_tokens", k_src);
        TORCH_CHECK(nv.numel() == 1,
                    "num_valid_tokens must contain one element");
        valid_ptr = nv.data_ptr<int32_t>();
    }

    if (num_tokens == 0) {
        return;
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    fused_indexer_k_write_fp8_kernel<<<num_tokens, kWarpSize, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(k_src.data_ptr()),
        page_table.data_ptr<int32_t>(),
        slot_indices.data_ptr<int32_t>(),
        token_indices.data_ptr<int32_t>(),
        valid_ptr,
        reinterpret_cast<uint8_t*>(cache.data_ptr()),
        k_fp8_ptr,
        scales_ptr,
        num_tokens,
        data_dim,
        page_size,
        static_cast<int>(page_table.size(1)),
        num_pages);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_indexer_k_write_fp8", &fused_indexer_k_write_fp8,
          "Quantize indexer K to e4m3 and write it into a page-split paged "
          "cache, optionally emitting the host-append (k_fp8, scales) sources",
          py::arg("k_src"), py::arg("page_table"), py::arg("slot_indices"),
          py::arg("token_indices"), py::arg("cache"),
          py::arg("page_size_tokens"),
          py::arg("k_fp8_out") = c10::nullopt,
          py::arg("scales_out") = c10::nullopt,
          py::arg("num_valid_tokens") = c10::nullopt);
}
