#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kHeadDim = 128;
constexpr int kKernelWidth = 4;
constexpr int kConvStateWidth = kKernelWidth - 1;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kChunkV = 32;
constexpr int kNumChunks = kHeadDim / kChunkV;
constexpr int kStateChunkElements = kChunkV * kHeadDim;

__device__ __forceinline__ float bf16_to_float(const __nv_bfloat16* ptr,
                                                int64_t index) {
    return __bfloat162float(ptr[index]);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float value) {
    return __float2bfloat16(value);
}

__device__ __forceinline__ float warp_sum(float value) {
    constexpr unsigned mask = 0xffffffffu;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(mask, value, offset);
    }
    return value;
}

__device__ float block_sum(float value, float* scratch) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const float total = warp_sum(value);
    if (lane == 0) {
        scratch[warp] = total;
    }
    __syncthreads();

    float result = 0.0f;
    if (warp == 0) {
        result = lane < kWarps ? scratch[lane] : 0.0f;
        result = warp_sum(result);
        if (lane == 0) {
            scratch[0] = result;
        }
    }
    __syncthreads();
    return scratch[0];
}

__device__ __forceinline__ uint32_t shared_address(void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void cp_async_16b(void* shared_dst,
                                               const void* global_src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :: "r"(shared_address(shared_dst)), "l"(global_src)
                 : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;" ::: "memory");
}

// The BatchGen KDA manager stores each convolution pool as
// [slots, channels, width-1].  The fused kernel deliberately consumes that
// native layout instead of making the [slots, width-1, channels] staging copy
// used by the upstream reference kernel.
__device__ void load_state_chunk(float* shared_state,
                                 const float* state,
                                 int slot,
                                 int head,
                                 int64_t state_slot_stride,
                                 int chunk) {
    const int tid = threadIdx.x;
    const int stage = chunk & 1;
    const int v_base = chunk * kChunkV;
    const int64_t slot_base = static_cast<int64_t>(slot) * state_slot_stride;
    for (int linear4 = tid;
         linear4 < kStateChunkElements / 4;
         linear4 += kThreads) {
        const int element = linear4 * 4;
        const int row = element / kHeadDim;
        const int key = element - row * kHeadDim;
        float* dst = shared_state + stage * kStateChunkElements
                   + row * kHeadDim + key;
        const float* src = state + slot_base
                         + ((head * kHeadDim + v_base + row) * kHeadDim + key);
        cp_async_16b(dst, src);
    }
    cp_async_commit();
}

template <typename T>
__device__ __forceinline__ float load_weight(const T* weight,
                                              int64_t channel,
                                              int tap,
                                              int64_t channel_stride,
                                              int64_t tap_stride);

template <>
__device__ __forceinline__ float load_weight<float>(const float* weight,
                                                     int64_t channel,
                                                     int tap,
                                                     int64_t channel_stride,
                                                     int64_t tap_stride) {
    return weight[channel * channel_stride + tap * tap_stride];
}

template <bool kUseLowerBound>
__global__ __launch_bounds__(kThreads, 2) void kda_fused_decode_kernel(
    const __nv_bfloat16* __restrict__ mixed_qkv,
    const __nv_bfloat16* __restrict__ forget_gate,
    const __nv_bfloat16* __restrict__ beta,
    __nv_bfloat16* __restrict__ conv_q,
    __nv_bfloat16* __restrict__ conv_k,
    __nv_bfloat16* __restrict__ conv_v,
    const float* __restrict__ weight_q,
    const float* __restrict__ weight_k,
    const float* __restrict__ weight_v,
    const float* __restrict__ bias_q,
    const float* __restrict__ bias_k,
    const float* __restrict__ bias_v,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ onorm_gate,
    const float* __restrict__ onorm_weight,
    float* __restrict__ state,
    const int32_t* __restrict__ state_indices,
    __nv_bfloat16* __restrict__ output,
    int batch,
    int heads,
    float scale,
    float onorm_eps,
    float lower_bound,
    int64_t mixed_stride,
    int64_t forget_stride,
    int64_t beta_stride,
    int64_t conv_slot_stride,
    int64_t conv_channel_stride,
    int64_t conv_tap_stride,
    int64_t weight_channel_stride,
    int64_t weight_tap_stride,
    int64_t state_slot_stride,
    int64_t onorm_gate_stride) {
    const int token = blockIdx.x;
    const int head = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (token >= batch || head >= heads) {
        return;
    }

    const int segment = heads * kHeadDim;
    const int channel_base = head * kHeadDim;
    const int32_t slot = state_indices[token];
    __shared__ float q[kHeadDim];
    __shared__ float k[kHeadDim];
    __shared__ float decay[kHeadDim];
    __shared__ float value[kHeadDim];
    __shared__ float output_value[kHeadDim];
    __shared__ float reduce[kWarps];
    __shared__ float beta_value;
    extern __shared__ float state_tile[];

    if (slot < 0) {
        if (tid < kHeadDim) {
            output[token * segment + channel_base + tid] =
                float_to_bf16(0.0f);
        }
        return;
    }

    const int64_t mixed_base = static_cast<int64_t>(token) * mixed_stride;
    const int64_t forget_base = static_cast<int64_t>(token) * forget_stride;
    const int64_t beta_base = static_cast<int64_t>(token) * beta_stride;
    const int64_t conv_base = static_cast<int64_t>(slot) * conv_slot_stride;

    // Begin the first state transfer before doing the independent convolution
    // update.  The second stage is issued while stage zero is consumed below.
    load_state_chunk(state_tile, state, slot, head, state_slot_stride, 0);

    if (tid < kHeadDim) {
        const int channel = channel_base + tid;
        float q_acc = bias_q == nullptr ? 0.0f : bias_q[channel];
        float k_acc = bias_k == nullptr ? 0.0f : bias_k[channel];
        float v_acc = bias_v == nullptr ? 0.0f : bias_v[channel];

#pragma unroll
        for (int tap = 0; tap < kConvStateWidth; ++tap) {
            const int64_t conv_offset = conv_base
                                      + channel * conv_channel_stride
                                      + tap * conv_tap_stride;
            const float q_state = bf16_to_float(conv_q, conv_offset);
            const float k_state = bf16_to_float(conv_k, conv_offset);
            const float v_state = bf16_to_float(conv_v, conv_offset);
            q_acc += q_state * load_weight(
                weight_q, channel, tap, weight_channel_stride, weight_tap_stride);
            k_acc += k_state * load_weight(
                weight_k, channel, tap, weight_channel_stride, weight_tap_stride);
            v_acc += v_state * load_weight(
                weight_v, channel, tap, weight_channel_stride, weight_tap_stride);
        }

        const __nv_bfloat16 q_new = mixed_qkv[mixed_base + channel];
        const __nv_bfloat16 k_new = mixed_qkv[mixed_base + segment + channel];
        const __nv_bfloat16 v_new = mixed_qkv[mixed_base + 2 * segment + channel];
        q_acc += __bfloat162float(q_new) * load_weight(
            weight_q, channel, kKernelWidth - 1,
            weight_channel_stride, weight_tap_stride);
        k_acc += __bfloat162float(k_new) * load_weight(
            weight_k, channel, kKernelWidth - 1,
            weight_channel_stride, weight_tap_stride);
        v_acc += __bfloat162float(v_new) * load_weight(
            weight_v, channel, kKernelWidth - 1,
            weight_channel_stride, weight_tap_stride);

        // Keep the exact ShortConvolution shift-register order: old taps 1/2
        // become new taps 0/1, and the raw projection is tap 2.
        const int64_t conv_tap0 = conv_base + channel * conv_channel_stride;
        const int64_t conv_tap1 = conv_tap0 + conv_tap_stride;
        const int64_t conv_tap2 = conv_tap1 + conv_tap_stride;
        const __nv_bfloat16 q_shift1 = conv_q[conv_tap1];
        const __nv_bfloat16 q_shift2 = conv_q[conv_tap2];
        const __nv_bfloat16 k_shift1 = conv_k[conv_tap1];
        const __nv_bfloat16 k_shift2 = conv_k[conv_tap2];
        const __nv_bfloat16 v_shift1 = conv_v[conv_tap1];
        const __nv_bfloat16 v_shift2 = conv_v[conv_tap2];
        conv_q[conv_tap0] = q_shift1;
        conv_q[conv_tap1] = q_shift2;
        conv_q[conv_tap2] = q_new;
        conv_k[conv_tap0] = k_shift1;
        conv_k[conv_tap1] = k_shift2;
        conv_k[conv_tap2] = k_new;
        conv_v[conv_tap0] = v_shift1;
        conv_v[conv_tap1] = v_shift2;
        conv_v[conv_tap2] = v_new;

        // This matches the fast-math path used by the fused recurrence
        // kernel.  Numerical acceptance remains an end-to-end K3 gate.
        q[tid] = q_acc / (1.0f + __expf(-q_acc));
        k[tid] = k_acc / (1.0f + __expf(-k_acc));
        value[tid] = v_acc / (1.0f + __expf(-v_acc));

        const float gate = bf16_to_float(
            forget_gate, forget_base + channel) + dt_bias[channel];
        const float exp_a = __expf(a_log[head]);
        if constexpr (kUseLowerBound) {
            const float sigmoid = 1.0f / (1.0f + __expf(-exp_a * gate));
            decay[tid] = __expf(lower_bound * sigmoid);
        } else {
            // Keep the fallback path aligned with SGLang's softplus_fast:
            // log1pf is materially better behaved for small gates, and the
            // asymptotic branch is strict (> 20), not >= 20.
            const float softplus = gate > 20.0f
                ? gate : log1pf(__expf(gate));
            decay[tid] = __expf(-exp_a * softplus);
        }
    }

    if (tid == 0) {
        const float raw_beta = bf16_to_float(beta, beta_base + head);
        beta_value = 1.0f / (1.0f + __expf(-raw_beta));
    }

    const float q_sq = tid < kHeadDim ? q[tid] * q[tid] : 0.0f;
    const float k_sq = tid < kHeadDim ? k[tid] * k[tid] : 0.0f;
    const float q_norm = rsqrtf(block_sum(q_sq, reduce) + 1.0e-6f);
    const float k_norm = rsqrtf(block_sum(k_sq, reduce) + 1.0e-6f);
    if (tid < kHeadDim) {
        q[tid] *= q_norm * scale;
        k[tid] *= k_norm;
    }
    __syncthreads();
    __syncthreads();  // beta_value is published after the projection update.

    const int key_base = lane * 4;
    float q_reg[4];
    float k_reg[4];
    float decay_reg[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        q_reg[i] = q[key_base + i];
        k_reg[i] = k[key_base + i];
        decay_reg[i] = decay[key_base + i];
    }

#pragma unroll
    for (int chunk = 0; chunk < kNumChunks; ++chunk) {
        cp_async_wait_all();
        __syncthreads();
        if (chunk + 1 < kNumChunks) {
            load_state_chunk(state_tile, state, slot, head,
                             state_slot_stride, chunk + 1);
        }

        const float* current_state = state_tile
            + (chunk & 1) * kStateChunkElements;
#pragma unroll
        for (int row = 0; row < 4; row += 2) {
            const int value_row0 = warp + row * kWarps;
            const int value_row1 = warp + (row + 1) * kWarps;
            const int output_row0 = chunk * kChunkV + value_row0;
            const int output_row1 = chunk * kChunkV + value_row1;

            const float4 raw0 = *reinterpret_cast<const float4*>(
                current_state + value_row0 * kHeadDim + key_base);
            const float4 raw1 = *reinterpret_cast<const float4*>(
                current_state + value_row1 * kHeadDim + key_base);
            float h0[4] = {
                raw0.x * decay_reg[0], raw0.y * decay_reg[1],
                raw0.z * decay_reg[2], raw0.w * decay_reg[3]};
            float h1[4] = {
                raw1.x * decay_reg[0], raw1.y * decay_reg[1],
                raw1.z * decay_reg[2], raw1.w * decay_reg[3]};
            const float dot_k0 = warp_sum(
                h0[0] * k_reg[0] + h0[1] * k_reg[1]
                + h0[2] * k_reg[2] + h0[3] * k_reg[3]);
            const float dot_k1 = warp_sum(
                h1[0] * k_reg[0] + h1[1] * k_reg[1]
                + h1[2] * k_reg[2] + h1[3] * k_reg[3]);
            const float update0 = (value[output_row0] - dot_k0) * beta_value;
            const float update1 = (value[output_row1] - dot_k1) * beta_value;

            float dot_q0 = 0.0f;
            float dot_q1 = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                h0[i] += k_reg[i] * update0;
                h1[i] += k_reg[i] * update1;
                dot_q0 += h0[i] * q_reg[i];
                dot_q1 += h1[i] * q_reg[i];
            }
            dot_q0 = warp_sum(dot_q0);
            dot_q1 = warp_sum(dot_q1);

            const int64_t state_base = static_cast<int64_t>(slot)
                * state_slot_stride
                + (static_cast<int64_t>(head) * kHeadDim * kHeadDim
                   + static_cast<int64_t>(output_row0) * kHeadDim + key_base);
            *reinterpret_cast<float4*>(state + state_base) =
                make_float4(h0[0], h0[1], h0[2], h0[3]);
            *reinterpret_cast<float4*>(state + state_base
                + static_cast<int64_t>(kWarps) * kHeadDim) =
                make_float4(h1[0], h1[1], h1[2], h1[3]);
            if (lane == 0) {
                output_value[output_row0] = dot_q0;
                output_value[output_row1] = dot_q1;
            }
        }
        __syncthreads();
    }

    const float sumsq = block_sum(
        tid < kHeadDim ? output_value[tid] * output_value[tid] : 0.0f,
        reduce);
    if (tid < kHeadDim) {
        const float norm = rsqrtf(sumsq / static_cast<float>(kHeadDim)
                                  + onorm_eps);
        const float gate = 1.0f / (1.0f + __expf(-bf16_to_float(
            onorm_gate, static_cast<int64_t>(token) * onorm_gate_stride
                + channel_base + tid)));
        const float y = output_value[tid] * norm * onorm_weight[tid] * gate;
        output[token * segment + channel_base + tid] = float_to_bf16(y);
    }
}

template <bool kUseLowerBound>
void launch_kda_fused_decode(
    const torch::Tensor& mixed_qkv,
    const torch::Tensor& forget_gate,
    const torch::Tensor& beta,
    const torch::Tensor& conv_q,
    const torch::Tensor& conv_k,
    const torch::Tensor& conv_v,
    const torch::Tensor& weight_q,
    const torch::Tensor& weight_k,
    const torch::Tensor& weight_v,
    const c10::optional<torch::Tensor>& bias_q,
    const c10::optional<torch::Tensor>& bias_k,
    const c10::optional<torch::Tensor>& bias_v,
    const torch::Tensor& a_log,
    const torch::Tensor& dt_bias,
    const torch::Tensor& onorm_gate,
    const torch::Tensor& onorm_weight,
    const torch::Tensor& state,
    const torch::Tensor& state_indices,
    torch::Tensor& output,
    float scale,
    float onorm_eps,
    float lower_bound) {
    const int batch = static_cast<int>(mixed_qkv.size(0));
    const int heads = static_cast<int>(state.size(1));
    const int segment = heads * kHeadDim;
    const size_t shared_bytes = static_cast<size_t>(2)
        * kStateChunkElements * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream(mixed_qkv.device().index());

    const float* bq = bias_q.has_value() ? bias_q->data_ptr<float>() : nullptr;
    const float* bk = bias_k.has_value() ? bias_k->data_ptr<float>() : nullptr;
    const float* bv = bias_v.has_value() ? bias_v->data_ptr<float>() : nullptr;

    kda_fused_decode_kernel<kUseLowerBound><<<
        dim3(batch, heads), dim3(kThreads), shared_bytes, stream.stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(mixed_qkv.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(forget_gate.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(beta.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(conv_q.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(conv_k.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(conv_v.data_ptr()),
        weight_q.data_ptr<float>(), weight_k.data_ptr<float>(),
        weight_v.data_ptr<float>(), bq, bk, bv,
        a_log.data_ptr<float>(), dt_bias.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(onorm_gate.data_ptr()),
        onorm_weight.data_ptr<float>(), state.data_ptr<float>(),
        state_indices.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        batch, heads, scale, onorm_eps, lower_bound,
        mixed_qkv.stride(0), forget_gate.stride(0), beta.stride(0),
        conv_q.stride(0), conv_q.stride(1), conv_q.stride(2),
        weight_q.stride(0), weight_q.stride(1), state.stride(0),
        onorm_gate.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_common(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
}

torch::Tensor kda_fused_decode_forward(
    const torch::Tensor& mixed_qkv,
    const torch::Tensor& forget_gate,
    const torch::Tensor& beta,
    const torch::Tensor& conv_q,
    const torch::Tensor& conv_k,
    const torch::Tensor& conv_v,
    const torch::Tensor& weight_q,
    const torch::Tensor& weight_k,
    const torch::Tensor& weight_v,
    const c10::optional<torch::Tensor>& bias_q,
    const c10::optional<torch::Tensor>& bias_k,
    const c10::optional<torch::Tensor>& bias_v,
    const torch::Tensor& a_log,
    const torch::Tensor& dt_bias,
    const torch::Tensor& onorm_gate,
    const torch::Tensor& onorm_weight,
    const torch::Tensor& state,
    const torch::Tensor& state_indices,
    double scale,
    double onorm_eps,
    double lower_bound,
    bool use_lower_bound) {
    check_common(mixed_qkv, "mixed_qkv");
    check_common(forget_gate, "forget_gate");
    check_common(beta, "beta");
    check_common(conv_q, "conv_q");
    check_common(conv_k, "conv_k");
    check_common(conv_v, "conv_v");
    check_common(weight_q, "weight_q");
    check_common(weight_k, "weight_k");
    check_common(weight_v, "weight_v");
    check_common(a_log, "a_log");
    check_common(dt_bias, "dt_bias");
    check_common(onorm_gate, "onorm_gate");
    check_common(onorm_weight, "onorm_weight");
    check_common(state, "state");
    check_common(state_indices, "state_indices");
    if (bias_q.has_value()) check_common(*bias_q, "bias_q");
    if (bias_k.has_value()) check_common(*bias_k, "bias_k");
    if (bias_v.has_value()) check_common(*bias_v, "bias_v");

    TORCH_CHECK(mixed_qkv.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(forget_gate.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(beta.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(conv_q.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(conv_k.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(conv_v.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(weight_q.scalar_type() == torch::kFloat);
    TORCH_CHECK(weight_k.scalar_type() == torch::kFloat);
    TORCH_CHECK(weight_v.scalar_type() == torch::kFloat);
    TORCH_CHECK(a_log.scalar_type() == torch::kFloat);
    TORCH_CHECK(dt_bias.scalar_type() == torch::kFloat);
    TORCH_CHECK(onorm_gate.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(onorm_weight.scalar_type() == torch::kFloat);
    TORCH_CHECK(state.scalar_type() == torch::kFloat);
    TORCH_CHECK(state_indices.scalar_type() == torch::kInt);
    for (const auto& bias : {bias_q, bias_k, bias_v}) {
        if (bias.has_value()) {
            TORCH_CHECK(bias->scalar_type() == torch::kFloat);
        }
    }

    TORCH_CHECK(mixed_qkv.dim() == 2 && mixed_qkv.size(1) % (3 * kHeadDim) == 0,
                "mixed_qkv must be [B, 3*H*128]");
    const int64_t batch = mixed_qkv.size(0);
    const int64_t heads = mixed_qkv.size(1) / (3 * kHeadDim);
    TORCH_CHECK(heads == 3 || heads == 6 || heads == 12,
                "unsupported local KDA head count: ", heads);
    const int64_t segment = heads * kHeadDim;
    TORCH_CHECK(forget_gate.dim() == 2 && forget_gate.size(0) == batch
                && forget_gate.size(1) == segment);
    TORCH_CHECK(beta.dim() == 2 && beta.size(0) == batch
                && beta.size(1) == heads);
    for (const auto& conv : {conv_q, conv_k, conv_v}) {
        TORCH_CHECK(conv.dim() == 3 && conv.size(1) == segment
                    && conv.size(2) == kConvStateWidth);
    }
    for (const auto& weight : {weight_q, weight_k, weight_v}) {
        TORCH_CHECK(weight.dim() == 2 && weight.size(0) == segment
                    && weight.size(1) == kKernelWidth);
    }
    for (const auto& bias : {bias_q, bias_k, bias_v}) {
        if (bias.has_value()) {
            TORCH_CHECK(bias->dim() == 1 && bias->size(0) == segment);
        }
    }
    TORCH_CHECK(a_log.dim() == 1 && a_log.size(0) >= heads);
    TORCH_CHECK(dt_bias.dim() == 1 && dt_bias.size(0) == segment);
    TORCH_CHECK(onorm_gate.dim() == 2 && onorm_gate.size(0) == batch
                && onorm_gate.size(1) == segment);
    TORCH_CHECK(onorm_weight.dim() == 1 && onorm_weight.size(0) == kHeadDim);
    TORCH_CHECK(state.dim() == 4 && state.size(0) <= conv_q.size(0)
                && state.size(1) == heads && state.size(2) == kHeadDim
                && state.size(3) == kHeadDim);
    TORCH_CHECK(state_indices.dim() == 1 && state_indices.size(0) == batch);
    TORCH_CHECK(mixed_qkv.stride(1) == 1 && forget_gate.stride(1) == 1
                && beta.stride(1) == 1 && onorm_gate.stride(1) == 1);
    TORCH_CHECK(conv_q.stride(2) == 1 && conv_k.stride(2) == 1
                && conv_v.stride(2) == 1);
    TORCH_CHECK(weight_q.stride(1) == 1 && weight_k.stride(1) == 1
                && weight_v.stride(1) == 1);
    TORCH_CHECK(state.stride(3) == 1 && state.stride(2) == kHeadDim
                && state.stride(1) == kHeadDim * kHeadDim);
    TORCH_CHECK(state_indices.is_contiguous());

    auto output = torch::empty(
        {batch, segment}, mixed_qkv.options().dtype(torch::kBFloat16));
    if (batch == 0) {
        return output;
    }

    c10::cuda::CUDAGuard device_guard(mixed_qkv.device());
    if (use_lower_bound) {
        launch_kda_fused_decode<true>(
            mixed_qkv, forget_gate, beta, conv_q, conv_k, conv_v,
            weight_q, weight_k, weight_v, bias_q, bias_k, bias_v,
            a_log, dt_bias, onorm_gate, onorm_weight, state, state_indices,
            output, static_cast<float>(scale), static_cast<float>(onorm_eps),
            static_cast<float>(lower_bound));
    } else {
        launch_kda_fused_decode<false>(
            mixed_qkv, forget_gate, beta, conv_q, conv_k, conv_v,
            weight_q, weight_k, weight_v, bias_q, bias_k, bias_v,
            a_log, dt_bias, onorm_gate, onorm_weight, state, state_indices,
            output, static_cast<float>(scale), static_cast<float>(onorm_eps),
            static_cast<float>(lower_bound));
    }
    return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kda_fused_decode_forward", &kda_fused_decode_forward,
          "K3 fused KDA decode: conv + recurrence + gated RMSNorm");
}
