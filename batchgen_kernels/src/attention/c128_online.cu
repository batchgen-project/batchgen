// -------------------------------------------------------------------------- //
//  c128_online.cu — Streaming HCA compress-128 (ring_size=1, online softmax) //
//                                                                            //
//  Ported from sglang deepseek_v4/c128_online.cuh.                           //
//  Algorithm identical; sglang infrastructure (TVM FFI, sgl_kernel headers,   //
//  PDL) replaced with torch C++ extension / raw CUDA.                        //
// -------------------------------------------------------------------------- //

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

namespace {

// -------------------------------------------------------------------------- //
// Decode kernel — one token per batch element, per-element online softmax.
//
// kv_score_buffer layout (per slot):  [max(D) | sum(D) | kv(D)]   (3*D floats)
// kv_score_input  layout (per token): [kv(D)  | score(D)]         (2*D floats)
//
// When old_sum == 0 the slot is uninitialised → first-token init.
// The kernel ALWAYS writes both buffer and output; the caller decides when a
// 128-chunk is complete and should zero the buffer for the next chunk.
// -------------------------------------------------------------------------- //

template <int kHeadDim>
__global__ void c128_online_step_kernel(
    float* __restrict__ kv_score_buffer,
    const float* __restrict__ kv_score_input,
    float* __restrict__ output,
    const int32_t* __restrict__ indices,
    uint32_t batch_size) {

  constexpr int kVecSize = 4;
  constexpr int kBlockSize = kHeadDim / kVecSize;

  const uint32_t batch_id = blockIdx.x;
  if (batch_id >= batch_size) return;

  const uint32_t tid = threadIdx.x;
  if (tid >= static_cast<uint32_t>(kBlockSize)) return;

  const int32_t index = indices[batch_id];
  const uint32_t base = tid * kVecSize;

  // Pointers ------------------------------------------------------------------
  float* buf = kv_score_buffer + static_cast<int64_t>(index) * kHeadDim * 3;
  const float* inp =
      kv_score_input + static_cast<int64_t>(batch_id) * kHeadDim * 2;
  float* out = output + static_cast<int64_t>(batch_id) * kHeadDim;

  // Per-element online softmax ------------------------------------------------
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) {
    const uint32_t idx = base + i;
    const float old_max = buf[idx];
    const float old_sum = buf[kHeadDim + idx];
    const float old_kv = buf[2 * kHeadDim + idx];
    const float nkv = inp[idx];
    const float nscore = inp[kHeadDim + idx];

    float r_kv, r_max, r_sum;
    if (old_sum == 0.0f) {
      // First token of this chunk — initialise.
      r_kv = nkv;
      r_max = nscore;
      r_sum = 1.0f;
    } else {
      // Mid-chunk — combine prior partial state via online softmax.
      r_max = fmaxf(old_max, nscore);
      const float resc = old_sum * expf(old_max - r_max);
      const float nexp = expf(nscore - r_max);
      r_sum = resc + nexp;
      r_kv = (old_kv * resc + nkv * nexp) / r_sum;
    }

    // Persist running state.
    buf[idx] = r_max;
    buf[kHeadDim + idx] = r_sum;
    buf[2 * kHeadDim + idx] = r_kv;
    // Always emit current weighted average.
    out[idx] = r_kv;
  }
}

// -------------------------------------------------------------------------- //
// Launch helpers                                                              //
// -------------------------------------------------------------------------- //

template <int kHeadDim>
void launch_c128_online_step(
    torch::Tensor kv_score_buffer,
    torch::Tensor kv_score_input,
    torch::Tensor output,
    torch::Tensor indices) {

  const uint32_t batch_size =
      static_cast<uint32_t>(kv_score_input.size(0));
  if (batch_size == 0) return;

  constexpr int kBlockSize = kHeadDim / 4;
  c10::cuda::CUDAGuard device_guard(kv_score_input.device());
  const auto stream = at::cuda::getCurrentCUDAStream().stream();

  c128_online_step_kernel<kHeadDim>
      <<<batch_size, kBlockSize, 0, stream>>>(
          kv_score_buffer.data_ptr<float>(),
          kv_score_input.data_ptr<float>(),
          output.data_ptr<float>(),
          indices.data_ptr<int32_t>(),
          batch_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// -------------------------------------------------------------------------- //
// Entrypoint (dispatches on head_dim)                                         //
// -------------------------------------------------------------------------- //

void c128_online_step(
    torch::Tensor kv_score_buffer,
    torch::Tensor kv_score_input,
    torch::Tensor output,
    torch::Tensor indices) {

  TORCH_CHECK(kv_score_buffer.is_cuda(), "kv_score_buffer must be CUDA");
  TORCH_CHECK(kv_score_input.is_cuda(), "kv_score_input must be CUDA");
  TORCH_CHECK(output.is_cuda(), "output must be CUDA");
  TORCH_CHECK(indices.is_cuda(), "indices must be CUDA");
  TORCH_CHECK(kv_score_buffer.dtype() == torch::kFloat32,
              "kv_score_buffer must be float32");
  TORCH_CHECK(kv_score_input.dtype() == torch::kFloat32,
              "kv_score_input must be float32");
  TORCH_CHECK(output.dtype() == torch::kFloat32,
              "output must be float32");
  TORCH_CHECK(indices.dtype() == torch::kInt32,
              "indices must be int32");
  TORCH_CHECK(kv_score_input.is_contiguous(), "kv_score_input must be contiguous");
  TORCH_CHECK(kv_score_buffer.is_contiguous(), "kv_score_buffer must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");

  const int64_t head_dim = kv_score_input.size(-1) / 2;
  TORCH_CHECK(head_dim > 0 && head_dim % 4 == 0,
              "head_dim must be positive and a multiple of 4, got ", head_dim);

  switch (head_dim) {
    case 128:
      launch_c128_online_step<128>(kv_score_buffer, kv_score_input,
                                   output, indices);
      break;
    case 256:
      launch_c128_online_step<256>(kv_score_buffer, kv_score_input,
                                   output, indices);
      break;
    case 512:
      launch_c128_online_step<512>(kv_score_buffer, kv_score_input,
                                   output, indices);
      break;
    default:
      TORCH_CHECK(false,
                  "Unsupported head_dim=", head_dim,
                  "; supported: {128, 256, 512}");
  }
}
