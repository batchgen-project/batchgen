#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int TopK = 2048;
constexpr int kThreadsPerBlock = 1024;
constexpr size_t kSmem = 8 * 1024 * sizeof(uint32_t);

struct FastTopKParams {
  const float* __restrict__ input;
  int32_t* __restrict__ indices;
  const int32_t* __restrict__ lengths;
  const int32_t* __restrict__ num_valid_tokens;
  int64_t input_stride;
  bool has_valid_tokens;
};

__device__ void dense_prefix_topk(int32_t* __restrict__ indices, int32_t length) {
  const int tid = threadIdx.x;
  for (int i = tid; i < TopK; i += kThreadsPerBlock) {
    indices[i] = (i < length) ? i : -1;
  }
}

__device__ __forceinline__ uint8_t score_to_u8_key(float x) {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ uint32_t score_to_u32_key(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ void radix_topk_2048(
    const float* __restrict__ input,
    int32_t* __restrict__ indices,
    int32_t length) {
  int topk = TopK;
  constexpr int BLOCK_SIZE = kThreadsPerBlock;
  constexpr int RADIX = 256;
  constexpr int SMEM_INPUT_SIZE = kSmem / (2 * sizeof(int));

  alignas(128) __shared__ int histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int counter;
  alignas(128) __shared__ int threshold_bin_id;
  alignas(128) __shared__ int num_input[2];
  auto& histogram = histogram_buf[0];
  extern __shared__ int input_idx[][SMEM_INPUT_SIZE];

  const int tx = threadIdx.x;

  if (tx < RADIX + 1) {
    histogram[tx] = 0;
  }
  __syncthreads();

  for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
    const int bin = static_cast<int>(score_to_u8_key(input[idx]));
    atomicAdd(&histogram[bin], 1);
  }
  __syncthreads();

  auto run_cumsum = [&] {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      if (tx < RADIX) {
        const int j = 1 << i;
        const int k = i & 1;
        int value = histogram_buf[k][tx];
        if (tx < RADIX - j) {
          value += histogram_buf[k][tx + j];
        }
        histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  run_cumsum();
  if (tx < RADIX && histogram[tx] > topk && histogram[tx + 1] <= topk) {
    threshold_bin_id = tx;
    num_input[0] = 0;
    counter = 0;
  }
  __syncthreads();

  int threshold_bin = threshold_bin_id;
  topk -= histogram[threshold_bin + 1];

  if (topk == 0) {
    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const int bin = static_cast<int>(score_to_u8_key(input[idx]));
      if (bin > threshold_bin) {
        const int pos = atomicAdd(&counter, 1);
        indices[pos] = idx;
      }
    }
    __syncthreads();
    return;
  }

  __syncthreads();
  if (tx < RADIX + 1) {
    histogram[tx] = 0;
  }
  __syncthreads();

  for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
    const float raw = input[idx];
    const int bin = static_cast<int>(score_to_u8_key(raw));
    if (bin > threshold_bin) {
      const int pos = atomicAdd(&counter, 1);
      indices[pos] = idx;
    } else if (bin == threshold_bin) {
      const int pos = atomicAdd(&num_input[0], 1);
      if (pos < SMEM_INPUT_SIZE) {
        input_idx[0][pos] = idx;
        const int sub_bin = (score_to_u32_key(raw) >> 24) & 0xFF;
        atomicAdd(&histogram[sub_bin], 1);
      }
    }
  }
  __syncthreads();

#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    __shared__ int last_remain;
    const int r_idx = round & 1;
    const int raw_num_input = num_input[r_idx];
    const int clipped_num_input = raw_num_input < SMEM_INPUT_SIZE ? raw_num_input : SMEM_INPUT_SIZE;

    run_cumsum();
    if (tx < RADIX && histogram[tx] > topk && histogram[tx + 1] <= topk) {
      threshold_bin_id = tx;
      num_input[r_idx ^ 1] = 0;
      last_remain = topk - histogram[tx + 1];
    }
    __syncthreads();

    threshold_bin = threshold_bin_id;
    topk -= histogram[threshold_bin + 1];

    if (topk == 0) {
      for (int i = tx; i < clipped_num_input; i += BLOCK_SIZE) {
        const int idx = input_idx[r_idx][i];
        const int offset = 24 - round * 8;
        const int bin = (score_to_u32_key(input[idx]) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const int pos = atomicAdd(&counter, 1);
          indices[pos] = idx;
        }
      }
      __syncthreads();
      break;
    }

    __syncthreads();
    if (tx < RADIX + 1) {
      histogram[tx] = 0;
    }
    __syncthreads();

    for (int i = tx; i < clipped_num_input; i += BLOCK_SIZE) {
      const int idx = input_idx[r_idx][i];
      const float raw = input[idx];
      const int offset = 24 - round * 8;
      const int bin = (score_to_u32_key(raw) >> offset) & 0xFF;
      if (bin > threshold_bin) {
        const int pos = atomicAdd(&counter, 1);
        indices[pos] = idx;
      } else if (bin == threshold_bin) {
        if (round == 3) {
          const int pos = atomicAdd(&last_remain, -1);
          if (pos > 0) {
            indices[TopK - pos] = idx;
          }
        } else {
          const int pos = atomicAdd(&num_input[r_idx ^ 1], 1);
          if (pos < SMEM_INPUT_SIZE) {
            input_idx[r_idx ^ 1][pos] = idx;
            const int sub_bin = (score_to_u32_key(raw) >> (offset - 8)) & 0xFF;
            atomicAdd(&histogram[sub_bin], 1);
          }
        }
      }
    }
    __syncthreads();
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)
void topk_2048_kernel(FastTopKParams params) {
  const uint64_t bid = static_cast<uint64_t>(blockIdx.x);
  int32_t* row_indices = params.indices + bid * TopK;
  if (params.has_valid_tokens && bid >= static_cast<uint64_t>(params.num_valid_tokens[0])) {
    dense_prefix_topk(row_indices, 0);
    return;
  }
  const int32_t length = params.lengths[bid];
  const float* row_scores = params.input + bid * params.input_stride;
  if (length <= TopK) {
    dense_prefix_topk(row_indices, length);
  } else {
    radix_topk_2048(row_scores, row_indices, length);
  }
}

}  // namespace

void fast_topk_2048_out(
    torch::Tensor score,
    torch::Tensor lengths,
    torch::Tensor indices,
    c10::optional<torch::Tensor> num_valid_tokens) {
  TORCH_CHECK(score.is_cuda(), "score must be CUDA");
  TORCH_CHECK(lengths.is_cuda(), "lengths must be CUDA");
  TORCH_CHECK(indices.is_cuda(), "indices must be CUDA");
  TORCH_CHECK(score.dtype() == torch::kFloat32, "score must be float32");
  TORCH_CHECK(lengths.dtype() == torch::kInt32, "lengths must be int32");
  TORCH_CHECK(indices.dtype() == torch::kInt32, "indices must be int32");
  TORCH_CHECK(score.dim() == 2 && score.is_contiguous(), "score must be contiguous [B, N]");
  TORCH_CHECK(lengths.dim() == 1 && lengths.is_contiguous(), "lengths must be contiguous [B]");
  TORCH_CHECK(indices.dim() == 2 && indices.is_contiguous(), "indices must be contiguous [B, 2048]");
  TORCH_CHECK(score.size(0) == lengths.size(0), "score and lengths batch mismatch");
  TORCH_CHECK(indices.size(0) == score.size(0) && indices.size(1) == TopK, "indices shape must be [B, 2048]");

  const int32_t* num_valid_ptr = nullptr;
  const bool has_valid_tokens = num_valid_tokens.has_value() && num_valid_tokens->defined();
  if (has_valid_tokens) {
    TORCH_CHECK(num_valid_tokens->is_cuda(), "num_valid_tokens must be CUDA");
    TORCH_CHECK(num_valid_tokens->dtype() == torch::kInt32, "num_valid_tokens must be int32");
    TORCH_CHECK(num_valid_tokens->numel() == 1, "num_valid_tokens must contain one element");
    num_valid_ptr = num_valid_tokens->data_ptr<int32_t>();
  }

  c10::cuda::CUDAGuard device_guard(score.device());
  FastTopKParams params{
      score.data_ptr<float>(),
      indices.data_ptr<int32_t>(),
      lengths.data_ptr<int32_t>(),
      num_valid_ptr,
      score.stride(0),
      has_valid_tokens,
  };
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const dim3 grid(static_cast<uint32_t>(score.size(0)));
  const dim3 block(kThreadsPerBlock);
  topk_2048_kernel<<<grid, block, kSmem, stream>>>(params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fast_topk_2048_out", torch::wrap_pybind_function(fast_topk_2048_out),
        "fast_topk_2048_out");
}
