#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kBlockM = 16;
constexpr int kBlockN = 128;
constexpr int kBlockI = 128;

__device__ __constant__ float kFp4Lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

__device__ __forceinline__ float decode_fp4(uint8_t packed, int k_idx) {
  uint8_t nibble = (k_idx & 1) == 0 ? (packed & 0x0F) : (packed >> 4);
  return kFp4Lut[nibble & 0x0F];
}

__device__ __forceinline__ float scaled_fp4_value(
    const uint8_t* weight_base,
    const uint8_t* scale_base,
    int64_t stride_weight_k,
    int64_t stride_weight_n,
    int64_t stride_scale_n,
    int64_t stride_scale_k,
    int k_idx,
    int n_idx) {
  uint8_t packed = weight_base[(k_idx >> 1) * stride_weight_k + static_cast<int64_t>(n_idx) * stride_weight_n];
  int exp = static_cast<int>(scale_base[static_cast<int64_t>(n_idx) * stride_scale_n + (k_idx >> 5) * stride_scale_k]) - 127;
  return ldexpf(decode_fp4(packed, k_idx), exp);
}

__global__ void mega_moe_sm120_kernel(
    const __nv_bfloat16* __restrict__ hidden_states,
    const int64_t* __restrict__ slot_token_ids,
    const float* __restrict__ slot_weights,
    const int32_t* __restrict__ block_experts,
    const int32_t* __restrict__ block_slot_starts,
    const int32_t* __restrict__ block_rows,
    const int32_t* __restrict__ num_blocks_ptr,
    const int32_t* __restrict__ expt_hist,
    const uint8_t* __restrict__ stage1_weight,
    const uint8_t* __restrict__ stage1_scale,
    const uint8_t* __restrict__ stage2_weight,
    const uint8_t* __restrict__ stage2_scale,
    float* __restrict__ output,
    int hidden,
    int intermediate,
    int64_t stride_hidden_m,
    int64_t stride_hidden_k,
    int64_t stride_stage1_e,
    int64_t stride_stage1_k,
    int64_t stride_stage1_n,
    int64_t stride_stage1_se,
    int64_t stride_stage1_sn,
    int64_t stride_stage1_sk,
    int64_t stride_stage2_e,
    int64_t stride_stage2_k,
    int64_t stride_stage2_n,
    int64_t stride_stage2_se,
    int64_t stride_stage2_sn,
    int64_t stride_stage2_sk,
    int64_t stride_output_m,
    int64_t stride_output_n,
    float swiglu_limit) {
  __shared__ __nv_bfloat16 activated[kBlockM * kBlockI];

  int lane_n = threadIdx.x;
  int block_idx = blockIdx.x;
  int n_start = static_cast<int>(blockIdx.y) * kBlockN;

  int active_blocks = num_blocks_ptr[0];
  if (block_idx >= active_blocks || lane_n >= kBlockN) {
    return;
  }

  int expert = block_experts[block_idx];
  int slot_start = block_slot_starts[block_idx];
  int row_start = block_rows[block_idx];
  int rows_in_block = expt_hist[expert] - row_start;
  if (rows_in_block <= 0) {
    return;
  }
  rows_in_block = min(rows_in_block, kBlockM);

  int out_col = n_start + lane_n;
  float acc_out[kBlockM] = {0.0f};

  const uint8_t* stage1_weight_base = stage1_weight + static_cast<int64_t>(expert) * stride_stage1_e;
  const uint8_t* stage1_scale_base = stage1_scale + static_cast<int64_t>(expert) * stride_stage1_se;
  const uint8_t* stage2_weight_base = stage2_weight + static_cast<int64_t>(expert) * stride_stage2_e;
  const uint8_t* stage2_scale_base = stage2_scale + static_cast<int64_t>(expert) * stride_stage2_se;

  for (int i0 = 0; i0 < intermediate; i0 += kBlockI) {
    int i_col = i0 + lane_n;
    bool valid_i = i_col < intermediate;
    int up_i_col = intermediate + i_col;

    float gate_acc[kBlockM] = {0.0f};
    float up_acc[kBlockM] = {0.0f};

    if (valid_i) {
      for (int m = 0; m < rows_in_block; ++m) {
        int64_t token_id = slot_token_ids[slot_start + m];
        const __nv_bfloat16* x_row = hidden_states + token_id * stride_hidden_m;

        float gate_sum = 0.0f;
        float up_sum = 0.0f;
        for (int k = 0; k < hidden; ++k) {
          float x = __bfloat162float(x_row[k * stride_hidden_k]);
          float gate_w = scaled_fp4_value(
              stage1_weight_base,
              stage1_scale_base,
              stride_stage1_k,
              stride_stage1_n,
              stride_stage1_sn,
              stride_stage1_sk,
              k,
              i_col);
          float up_w = scaled_fp4_value(
              stage1_weight_base,
              stage1_scale_base,
              stride_stage1_k,
              stride_stage1_n,
              stride_stage1_sn,
              stride_stage1_sk,
              k,
              up_i_col);
          gate_sum += x * gate_w;
          up_sum += x * up_w;
        }
        if (swiglu_limit > 0.0f) {
          gate_sum = fminf(gate_sum, swiglu_limit);
          up_sum = fmaxf(fminf(up_sum, swiglu_limit), -swiglu_limit);
        }
        gate_acc[m] = gate_sum;
        up_acc[m] = up_sum;
      }
    }

    for (int m = 0; m < kBlockM; ++m) {
      float activated_val = 0.0f;
      if (valid_i && m < rows_in_block) {
        float gate_val = gate_acc[m];
        activated_val = (gate_val / (1.0f + expf(-gate_val))) * up_acc[m];
      }
      activated[m * kBlockI + lane_n] = __float2bfloat16(activated_val);
    }
    __syncthreads();

    if (out_col < hidden) {
      for (int m = 0; m < rows_in_block; ++m) {
        float partial = 0.0f;
        for (int ii = 0; ii < kBlockI; ++ii) {
          int inter_idx = i0 + ii;
          if (inter_idx >= intermediate) {
            break;
          }
          float act = __bfloat162float(activated[m * kBlockI + ii]);
          float w = scaled_fp4_value(
              stage2_weight_base,
              stage2_scale_base,
              stride_stage2_k,
              stride_stage2_n,
              stride_stage2_sn,
              stride_stage2_sk,
              inter_idx,
              out_col);
          partial += act * w;
        }
        acc_out[m] += partial;
      }
    }
    __syncthreads();
  }

  if (out_col < hidden) {
    for (int m = 0; m < rows_in_block; ++m) {
      int64_t token_id = slot_token_ids[slot_start + m];
      float routed = acc_out[m] * slot_weights[slot_start + m];
      atomicAdd(output + token_id * stride_output_m + static_cast<int64_t>(out_col) * stride_output_n, routed);
    }
  }
}

void check_cuda_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

void mega_moe_sm120_forward_cuda(
    torch::Tensor hidden_states,
    torch::Tensor slot_token_ids,
    torch::Tensor slot_weights,
    torch::Tensor block_experts,
    torch::Tensor block_slot_starts,
    torch::Tensor block_rows,
    torch::Tensor num_blocks,
    torch::Tensor expt_hist,
    torch::Tensor stage1_weight,
    torch::Tensor stage1_scale,
    torch::Tensor stage2_weight,
    torch::Tensor stage2_scale,
    torch::Tensor output,
    double swiglu_limit) {
  check_cuda_tensor(hidden_states, "hidden_states");
  check_cuda_tensor(slot_token_ids, "slot_token_ids");
  check_cuda_tensor(slot_weights, "slot_weights");
  check_cuda_tensor(block_experts, "block_experts");
  check_cuda_tensor(block_slot_starts, "block_slot_starts");
  check_cuda_tensor(block_rows, "block_rows");
  check_cuda_tensor(num_blocks, "num_blocks");
  check_cuda_tensor(expt_hist, "expt_hist");
  check_cuda_tensor(stage1_weight, "stage1_weight");
  check_cuda_tensor(stage1_scale, "stage1_scale");
  check_cuda_tensor(stage2_weight, "stage2_weight");
  check_cuda_tensor(stage2_scale, "stage2_scale");
  check_cuda_tensor(output, "output");

  TORCH_CHECK(hidden_states.scalar_type() == torch::kBFloat16, "hidden_states must be bfloat16");
  TORCH_CHECK(slot_token_ids.scalar_type() == torch::kInt64, "slot_token_ids must be int64");
  TORCH_CHECK(slot_weights.scalar_type() == torch::kFloat32, "slot_weights must be float32");
  TORCH_CHECK(block_experts.scalar_type() == torch::kInt32, "block_experts must be int32");
  TORCH_CHECK(block_slot_starts.scalar_type() == torch::kInt32, "block_slot_starts must be int32");
  TORCH_CHECK(block_rows.scalar_type() == torch::kInt32, "block_rows must be int32");
  TORCH_CHECK(num_blocks.scalar_type() == torch::kInt32, "num_blocks must be int32");
  TORCH_CHECK(expt_hist.scalar_type() == torch::kInt32, "expt_hist must be int32");
  TORCH_CHECK(stage1_weight.scalar_type() == torch::kUInt8, "stage1_weight must be uint8");
  TORCH_CHECK(stage1_scale.scalar_type() == torch::kUInt8, "stage1_scale must be uint8");
  TORCH_CHECK(stage2_weight.scalar_type() == torch::kUInt8, "stage2_weight must be uint8");
  TORCH_CHECK(stage2_scale.scalar_type() == torch::kUInt8, "stage2_scale must be uint8");
  TORCH_CHECK(output.scalar_type() == torch::kFloat32, "output must be float32");

  TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be rank-2");
  TORCH_CHECK(output.dim() == 2, "output must be rank-2");
  TORCH_CHECK(stage1_weight.dim() == 3 && stage1_scale.dim() == 3, "stage1 tensors must be rank-3");
  TORCH_CHECK(stage2_weight.dim() == 3 && stage2_scale.dim() == 3, "stage2 tensors must be rank-3");
  TORCH_CHECK(num_blocks.numel() == 1, "num_blocks must contain exactly one element");

  auto cc = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(cc->major == 12, "mega_moe_sm120_forward_cuda requires sm120/cc12.x device");

  int hidden = static_cast<int>(hidden_states.size(1));
  int intermediate = static_cast<int>(stage2_weight.size(1) * 2);

  TORCH_CHECK(stage1_weight.size(0) == stage1_scale.size(0), "stage1 expert dimension mismatch");
  TORCH_CHECK(stage2_weight.size(0) == stage2_scale.size(0), "stage2 expert dimension mismatch");
  TORCH_CHECK(stage1_weight.size(1) * 2 == hidden, "stage1 packed-K mismatch with hidden size");
  TORCH_CHECK(stage2_weight.size(2) == hidden, "stage2 N/output mismatch with hidden size");
  TORCH_CHECK(stage1_weight.size(2) == intermediate * 2, "stage1 output width must equal 2 * intermediate");
  TORCH_CHECK(stage2_scale.size(1) == hidden, "stage2 scale N dimension mismatch");

  c10::cuda::CUDAGuard device_guard(hidden_states.device());
  int blocks_x = static_cast<int>(block_experts.size(0));
  if (blocks_x < 1) {
    blocks_x = 1;
  }
  int blocks_y = (hidden + kBlockN - 1) / kBlockN;
  dim3 grid(blocks_x, blocks_y, 1);
  dim3 block(kBlockN, 1, 1);
  auto stream = at::cuda::getCurrentCUDAStream(hidden_states.device().index());
  mega_moe_sm120_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr<at::BFloat16>()),
      slot_token_ids.data_ptr<int64_t>(),
      slot_weights.data_ptr<float>(),
      block_experts.data_ptr<int32_t>(),
      block_slot_starts.data_ptr<int32_t>(),
      block_rows.data_ptr<int32_t>(),
      num_blocks.data_ptr<int32_t>(),
      expt_hist.data_ptr<int32_t>(),
      stage1_weight.data_ptr<uint8_t>(),
      stage1_scale.data_ptr<uint8_t>(),
      stage2_weight.data_ptr<uint8_t>(),
      stage2_scale.data_ptr<uint8_t>(),
      output.data_ptr<float>(),
      hidden,
      intermediate,
      hidden_states.stride(0),
      hidden_states.stride(1),
      stage1_weight.stride(0),
      stage1_weight.stride(1),
      stage1_weight.stride(2),
      stage1_scale.stride(0),
      stage1_scale.stride(1),
      stage1_scale.stride(2),
      stage2_weight.stride(0),
      stage2_weight.stride(1),
      stage2_weight.stride(2),
      stage2_scale.stride(0),
      stage2_scale.stride(1),
      stage2_scale.stride(2),
      output.stride(0),
      output.stride(1),
      static_cast<float>(swiglu_limit));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mega_moe_sm120_forward_cuda", &mega_moe_sm120_forward_cuda, "Native sm120 mega MoE forward");
}
