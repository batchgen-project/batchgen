// causal_conv1d for BatchGen — varlen prefill (state write to pool) + decode update.
//
// Ported from sgl-kernel csrc/mamba/causal_conv1d.cu, itself adapted from
// Dao-AILab causal-conv1d (fwd + update). Attribution preserved below.
//
// BatchGen adaptations vs the sgl-kernel version:
//   1. Decode update is functional: reads x, writes a fresh `out` (x untouched).
//   2. Conv-state pool layout is (num_slots, dim, width - 1), slot-indexed via
//      cache_indices (fwd) / conv_state_indices (update); entries equal to
//      pad_slot_id are skipped (state untouched).
//   3. Fwd keeps the sgl/vLLM convention: channel-major x (dim, total) varlen,
//      written in place (the python wrapper stages the transpose copy and
//      returns a token-major view).
//
// State semantics (width W, state width W-1): after prefill, a slot holds the
// last W-1 inputs [x_{T-W+1} .. x_{T-1}] at positions [0 .. W-2]. The decode
// update shifts left and appends the new token, matching fla's
// ShortConvolution cache [N, D, W] via the mapping fla_state[..., 1:] ==
// cuda_state[..., 0:W-1].
//
// clang-format off
// adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/csrc/causal_conv1d_fwd.cu
// and https://github.com/Dao-AILab/causal-conv1d/blob/main/csrc/causal_conv1d_update.cu
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define BOOL_SWITCH(COND, CONST_NAME, ...)                                           \
    [&] {                                                                            \
        if (COND) {                                                                  \
            static constexpr bool CONST_NAME = true;                                 \
            return __VA_ARGS__();                                                    \
        } else {                                                                     \
            static constexpr bool CONST_NAME = false;                                \
            return __VA_ARGS__();                                                    \
        }                                                                            \
    }()

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

#define DISPATCH_WTYPE_ITYPE_FLOAT_AND_HALF_AND_BF16(ITYPE, NAME, ...)              \
    if (ITYPE == at::ScalarType::Half) {                                            \
        using input_t = at::Half;                                                   \
        using weight_t = at::Half;                                                  \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::BFloat16) {                                 \
        using input_t = at::BFloat16;                                               \
        using weight_t = at::BFloat16;                                              \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::Float)  {                                   \
        using input_t = float;                                                      \
        using weight_t = float;                                                     \
        __VA_ARGS__();                                                              \
    } else {                                                                        \
        AT_ERROR(#NAME, " not implemented for input type '", toString(ITYPE), "'"); \
    }

////////////////////////////////////////////////////////////////////////////////////////////////////

struct ConvParamsBase {
    using index_t = uint32_t;

    int batch, dim, seqlen, width;
    int64_t pad_slot_id;
    bool silu_activation;

    index_t x_batch_stride;
    index_t x_c_stride;
    index_t x_l_stride;
    index_t weight_c_stride;
    index_t weight_width_stride;
    index_t out_batch_stride;
    index_t out_c_stride;
    index_t out_l_stride;

    int conv_state_len;
    index_t conv_state_batch_stride;
    index_t conv_state_c_stride;
    index_t conv_state_l_stride;

    // Common data pointers.
    void *__restrict__ x_ptr;
    void *__restrict__ weight_ptr;
    void *__restrict__ bias_ptr;
    void *__restrict__ out_ptr;

    void *__restrict__ conv_state_ptr;
    void *__restrict__ query_start_loc_ptr;
    void *__restrict__ has_initial_state_ptr;
    void *__restrict__ cache_indices_ptr;
    int32_t *__restrict__ cache_seqlens;

    // For the continuous batching case. Makes it so that the mamba state for
    // the current batch doesn't need to be a contiguous tensor.
    int32_t *__restrict__ conv_state_indices_ptr;

    // Final-state pool write (prefill): slot-indexed via cache_indices.
    void *  conv_states_ptr;
    index_t conv_states_batch_stride;
    index_t conv_states_l_stride;
    index_t conv_states_c_stride;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__device__ inline T shuffle_xor(T val, int offset) {
    return __shfl_xor_sync(uint32_t(-1), val, offset);
}

constexpr size_t custom_max(std::initializer_list<size_t> ilist)
{
    return std::max(ilist);
}

template<typename T>
constexpr T constexpr_min(T a, T b) {
    return std::min(a, b);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int BYTES> struct BytesToType {};

template<> struct BytesToType<16> {
    using Type = uint4;
    static_assert(sizeof(Type) == 16);
};

template<> struct BytesToType<8> {
    using Type = uint64_t;
    static_assert(sizeof(Type) == 8);
};

template<> struct BytesToType<4> {
    using Type = uint32_t;
    static_assert(sizeof(Type) == 4);
};

template<> struct BytesToType<2> {
    using Type = uint16_t;
    static_assert(sizeof(Type) == 2);
};

template<> struct BytesToType<1> {
    using Type = uint8_t;
    static_assert(sizeof(Type) == 1);
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename input_t, typename weight_t>
void causal_conv1d_fwd_cuda(ConvParamsBase &params, cudaStream_t stream);

template<typename input_t, typename weight_t>
void causal_conv1d_update_cuda(ConvParamsBase &params, cudaStream_t stream);

void set_conv_params_fwd(ConvParamsBase &params,
                         // sizes
                         const size_t batch,
                         const size_t dim,
                         const size_t seqlen,
                         const size_t width,
                         // device pointers
                         const at::Tensor &x,
                         const at::Tensor &weight,
                         const at::Tensor &out,
                         const std::optional<at::Tensor>& bias,
                         bool silu_activation,
                         int64_t pad_slot_id,
                         const std::optional<at::Tensor>& query_start_loc = std::nullopt,
                         const std::optional<at::Tensor>& cache_indices = std::nullopt,
                         const std::optional<at::Tensor>& has_initial_state = std::nullopt) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.seqlen = seqlen;
    params.width = width;
    params.pad_slot_id = pad_slot_id;

    params.silu_activation = silu_activation;

    // Set the pointers and strides.
    params.x_ptr = x.data_ptr();
    params.weight_ptr = weight.data_ptr();
    params.bias_ptr = bias.has_value() ? bias.value().data_ptr() : nullptr;
    params.out_ptr = out.data_ptr();
    // All strides are in elements, not bytes.
    params.query_start_loc_ptr = query_start_loc.has_value() ? query_start_loc.value().data_ptr() : nullptr;
    params.cache_indices_ptr = cache_indices.has_value() ? cache_indices.value().data_ptr() : nullptr;
    params.has_initial_state_ptr = has_initial_state.has_value() ? has_initial_state.value().data_ptr() : nullptr;
    const bool varlen = params.query_start_loc_ptr != nullptr;
    // All strides are in elements, not bytes.
    // varlen: x is (dim, total_tokens) channel-major — batch/c stride from
    // dims (0, 1); the sequence dim is required contiguous (stride 1).
    // batched: x is (batch, dim, seqlen); sequence dim required contiguous.
    params.x_batch_stride = x.stride(varlen ? 1 : 0);
    params.x_c_stride = x.stride(varlen ? 0 : 1);
    params.x_l_stride = x.stride(varlen ? 1 : 2);
    params.out_batch_stride = out.stride(varlen ? 1 : 0);
    params.out_c_stride = out.stride(varlen ? 0 : 1);
    params.out_l_stride = out.stride(varlen ? 1 : 2);
    params.weight_c_stride = weight.stride(0);
    params.weight_width_stride = weight.stride(1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Host bindings
////////////////////////////////////////////////////////////////////////////////////////////////////

void causal_conv1d_fwd(
    const at::Tensor &x,                                   // varlen: (dim, total_tokens) channel-major, written in place;
                                                           // batched: (batch, dim, seqlen), written in place
    const at::Tensor &weight,                              // (dim, width)
    const std::optional<at::Tensor> &bias_,                // (dim,) or None
    const std::optional<at::Tensor> &conv_states,          // (num_slots, dim, width-1) pool; final state written in place
    const std::optional<at::Tensor> &query_start_loc,      // (batch+1,) int32 — enables varlen mode
    const std::optional<at::Tensor> &cache_indices,        // (batch,) int32 — pool slot per sequence
    const std::optional<at::Tensor> &has_initial_state,    // (batch,) bool — read initial state from pool slot
    bool silu_activation,
    int64_t pad_slot_id) {
    auto input_type = x.scalar_type();
    auto weight_type = weight.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(weight_type == input_type, "weight type must equal input type");

    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(weight.is_cuda());

    const bool varlen = query_start_loc.has_value();
    const auto sizes = x.sizes();
    const int batch_size = varlen ? query_start_loc.value().sizes()[0] - 1 : sizes[0];
    const int dim = varlen ? sizes[0] : sizes[1];
    const int seqlen = varlen ? sizes[1] : sizes[2];
    const int width = weight.size(-1);
    TORCH_CHECK(width >= 2 && width <= 4, "causal_conv1d only supports width between 2 and 4");
    if (varlen) {
        CHECK_SHAPE(x, dim, seqlen);
    } else {
        CHECK_SHAPE(x, batch_size, dim, seqlen);
    }
    CHECK_SHAPE(weight, dim, width);
    // The block-load kernel path requires the sequence dim to be contiguous.
    TORCH_CHECK(x.stride(-1) == 1, "causal_conv1d_fwd requires x.stride(-1) == 1 (stage a contiguous copy)");

    if (bias_.has_value()) {
        auto bias = bias_.value();
        TORCH_CHECK(bias.scalar_type() == weight_type);
        TORCH_CHECK(bias.is_cuda());
        TORCH_CHECK(bias.stride(-1) == 1);
        CHECK_SHAPE(bias, dim);
    }

    if (has_initial_state.has_value()) {
        auto has_initial_state_ = has_initial_state.value();
        TORCH_CHECK(has_initial_state_.scalar_type() == at::ScalarType::Bool);
        TORCH_CHECK(has_initial_state_.is_cuda());
        CHECK_SHAPE(has_initial_state_, batch_size);
    }

    if (query_start_loc.has_value()) {
        auto query_start_loc_ = query_start_loc.value();
        TORCH_CHECK(query_start_loc_.scalar_type() == at::ScalarType::Int);
        TORCH_CHECK(query_start_loc_.is_cuda());
        TORCH_CHECK(query_start_loc_.stride(-1) == 1);
    }

    if (cache_indices.has_value()) {
        auto cache_indices_ = cache_indices.value();
        TORCH_CHECK(cache_indices_.scalar_type() == at::ScalarType::Int);
        TORCH_CHECK(cache_indices_.is_cuda());
        CHECK_SHAPE(cache_indices_, batch_size);
    }

    at::Tensor out = x;  // in-place, following the sgl/vLLM convention

    ConvParamsBase params;
    set_conv_params_fwd(params, batch_size, dim, seqlen, width, x, weight, out,
                        bias_,
                        silu_activation,
                        pad_slot_id,
                        query_start_loc,
                        cache_indices,
                        has_initial_state
                        );

    if (conv_states.has_value()) {
        auto conv_states_ = conv_states.value();
        TORCH_CHECK(conv_states_.scalar_type() == input_type);
        TORCH_CHECK(conv_states_.is_cuda());
        TORCH_CHECK(conv_states_.dim() == 3, "conv_states must be (num_slots, dim, width-1)");
        TORCH_CHECK(conv_states_.size(1) == dim);
        TORCH_CHECK(conv_states_.size(2) == width - 1, "conv_states last dim must be width-1");
        params.conv_states_ptr = conv_states_.data_ptr();
        params.conv_states_batch_stride = conv_states_.stride(0);
        params.conv_states_c_stride = conv_states_.stride(1);
        params.conv_states_l_stride = conv_states_.stride(2);
    } else {
        params.conv_states_ptr = nullptr;
    }

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_WTYPE_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "causal_conv1d_fwd", [&] {
            causal_conv1d_fwd_cuda<input_t, weight_t>(params, stream);
    });
}


at::Tensor causal_conv1d_update(
    const at::Tensor &x,               // (batch, dim) or (batch, dim, seqlen)
    const at::Tensor &conv_state,      // (num_slots, dim, state_len), updated in place; state_len >= width-1
    const at::Tensor &weight,          // (dim, width)
    const std::optional<at::Tensor> &bias_,
    bool silu_activation,
    const std::optional<at::Tensor> &cache_seqlens_,        // (batch,) int32 — circular-buffer mode (optional)
    const std::optional<at::Tensor> &conv_state_indices_,   // (batch,) int32 — pool slot per request
    int64_t pad_slot_id) {
    auto input_type = x.scalar_type();
    auto weight_type = weight.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(weight_type == input_type, "weight type must equal input type");
    TORCH_CHECK(conv_state.scalar_type() == input_type);

    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(conv_state.is_cuda());
    TORCH_CHECK(weight.is_cuda());

    at::Tensor x_ = x.dim() == 2 ? x.unsqueeze(-1) : x;
    TORCH_CHECK(x_.dim() == 3, "x must be (batch, dim) or (batch, dim, seqlen)");

    const auto sizes = x_.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];
    const int width = weight.size(-1);
    const int conv_state_len = conv_state.size(2);
    TORCH_CHECK(conv_state_len >= width - 1);

    CHECK_SHAPE(x_, batch_size, dim, seqlen);
    CHECK_SHAPE(weight, dim, width);

    TORCH_CHECK(width >= 2 && width <= 4, "causal_conv1d only supports width between 2 and 4");

    if (bias_.has_value()) {
        auto bias = bias_.value();
        TORCH_CHECK(bias.scalar_type() == weight_type);
        TORCH_CHECK(bias.is_cuda());
        TORCH_CHECK(bias.stride(-1) == 1);
        CHECK_SHAPE(bias, dim);
    }

    at::Tensor out = at::zeros_like(x_);

    ConvParamsBase params;
    set_conv_params_fwd(params, batch_size, dim, seqlen, width, x_, weight, out,
                        bias_,
                        silu_activation,
                        pad_slot_id);
    params.conv_state_ptr = conv_state.data_ptr();
    params.conv_state_len = conv_state_len;
    // All strides are in elements, not bytes.
    params.conv_state_batch_stride = conv_state.stride(0);
    params.conv_state_c_stride = conv_state.stride(1);
    params.conv_state_l_stride = conv_state.stride(2);

    if (cache_seqlens_.has_value()) {
        auto cache_seqlens = cache_seqlens_.value();
        TORCH_CHECK(cache_seqlens.scalar_type() == torch::kInt32);
        TORCH_CHECK(cache_seqlens.is_cuda());
        TORCH_CHECK(cache_seqlens.stride(-1) == 1);
        CHECK_SHAPE(cache_seqlens, batch_size);
        params.cache_seqlens = cache_seqlens.data_ptr<int32_t>();
    } else {
        params.cache_seqlens = nullptr;
    }

    if (conv_state_indices_.has_value()) {
        auto conv_state_indices = conv_state_indices_.value();
        TORCH_CHECK(conv_state_indices.scalar_type() == torch::kInt32)
        TORCH_CHECK(conv_state_indices.is_cuda());
        TORCH_CHECK(conv_state_indices.stride(0) == 1)
        CHECK_SHAPE(conv_state_indices, batch_size);

        int conv_state_entries = conv_state.size(0);
        CHECK_SHAPE(conv_state, conv_state_entries, dim, conv_state_len);

        params.conv_state_indices_ptr = conv_state_indices.data_ptr<int32_t>();
    } else {
        TORCH_CHECK(conv_state.size(0) == batch_size, "without conv_state_indices, conv_state batch must equal x batch");
        params.conv_state_indices_ptr = nullptr;
    }

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_WTYPE_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "causal_conv1d_update", [&] {
            causal_conv1d_update_cuda<input_t, weight_t>(params, stream);
    });
    return x.dim() == 2 ? out.squeeze(-1) : out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Forward kernel (prefill)
////////////////////////////////////////////////////////////////////////////////////////////////////

template<int kNThreads_, int kWidth_, bool kIsVecLoad_, typename input_t_, typename weight_t_>
struct Causal_conv1d_fwd_kernel_traits {
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kWidth = kWidth_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : 8;
    static_assert(kWidth <= kNElts);
    static constexpr bool kIsVecLoad = kIsVecLoad_;
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNElts, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, 1, cub::BLOCK_LOAD_DIRECT>;
    using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNElts, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, 1, cub::BLOCK_STORE_DIRECT>;
    static constexpr int kSmemIOSize = kIsVecLoad
        ? 0
        : custom_max({sizeof(typename BlockLoadT::TempStorage), sizeof(typename BlockStoreT::TempStorage)});
    static constexpr int kSmemExchangeSize = kNThreads * kNBytes * kNElts;
    static constexpr int kSmemSize = kSmemIOSize + kSmemExchangeSize;
};

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads)
void causal_conv1d_fwd_kernel(ConvParamsBase params) {
    constexpr int kWidth = Ktraits::kWidth;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNElts = Ktraits::kNElts;
    constexpr bool kIsVecLoad = Ktraits::kIsVecLoad;
    using input_t = typename Ktraits::input_t;
    using vec_t = typename Ktraits::vec_t;
    using weight_t = typename Ktraits::weight_t;

    // Shared memory.
    extern __shared__ char smem_[];
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_load_vec = reinterpret_cast<typename Ktraits::BlockLoadVecT::TempStorage&>(smem_);
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_store_vec = reinterpret_cast<typename Ktraits::BlockStoreVecT::TempStorage&>(smem_);
    vec_t *smem_exchange = reinterpret_cast<vec_t *>(smem_ + Ktraits::kSmemIOSize);

    const bool kVarlen = params.query_start_loc_ptr != nullptr;
    const int tidx = threadIdx.x;
    const int batch_id = blockIdx.x;
    const int channel_id = blockIdx.y;
    const int *query_start_loc = kVarlen ? reinterpret_cast<int *>(params.query_start_loc_ptr) : nullptr;
    const int sequence_start_index = kVarlen ? query_start_loc[batch_id] : batch_id;
    const int seqlen = kVarlen ? query_start_loc[batch_id + 1] - sequence_start_index : params.seqlen;

    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + sequence_start_index * params.x_batch_stride
        + channel_id * params.x_c_stride;
    weight_t *weight = reinterpret_cast<weight_t *>(params.weight_ptr) + channel_id * params.weight_c_stride;
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + sequence_start_index * params.out_batch_stride
        + channel_id * params.out_c_stride;
    float bias_val = params.bias_ptr == nullptr ? 0.f : float(reinterpret_cast<weight_t *>(params.bias_ptr)[channel_id]);

    bool has_initial_state = params.has_initial_state_ptr == nullptr ? false
        : reinterpret_cast<bool *>(params.has_initial_state_ptr)[batch_id];

    int* cache_indices = params.cache_indices_ptr == nullptr ? nullptr
        : reinterpret_cast<int *>(params.cache_indices_ptr);
    int cache_index = cache_indices == nullptr ? batch_id : cache_indices[batch_id];
    // cache_index == params.pad_slot_id is defined as padding, so we exit early
    if (cache_index == params.pad_slot_id){
        return;
    }
    input_t *conv_states = params.conv_states_ptr == nullptr ? nullptr
        : reinterpret_cast<input_t *>(params.conv_states_ptr) + cache_index * params.conv_states_batch_stride + channel_id * params.conv_states_c_stride;

    // Thread 0 will load the last elements of the previous chunk, so we initialize those to 0.
    if (tidx == 0) {
        input_t initial_state[kNElts] = {0};
        if (has_initial_state) {
            #pragma unroll
            for (int w = 0; w < kWidth - 1; ++w){ initial_state[kNElts - 1 - (kWidth - 2) + w ] = conv_states[w * params.conv_states_l_stride]; }
        }
        smem_exchange[kNThreads - 1] = reinterpret_cast<vec_t *>(initial_state)[0];
    }

    float weight_vals[kWidth];
    #pragma unroll
    for (int i = 0; i < kWidth; ++i) { weight_vals[i] = float(weight[i * params.weight_width_stride]); }

    constexpr int kChunkSize = kNThreads * kNElts;
    const int n_chunks = (seqlen + kChunkSize - 1) / kChunkSize;
    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        input_t x_vals_load[2 * kNElts] = {0};
        if constexpr(kIsVecLoad) {
            typename Ktraits::BlockLoadVecT(smem_load_vec).Load(reinterpret_cast<vec_t*>(x), *reinterpret_cast<vec_t (*)[1]>(&x_vals_load[kNElts]), (seqlen - chunk * kChunkSize) / kNElts);
        } else {
            __syncthreads();
            typename Ktraits::BlockLoadT(smem_load).Load(x, *reinterpret_cast<input_t (*)[kNElts]>(&x_vals_load[kNElts]), seqlen - chunk * kChunkSize);
        }
        x += kChunkSize * params.x_l_stride;
        __syncthreads();
        // Thread kNThreads - 1 don't write yet, so that thread 0 can read
        // the last elements of the previous chunk.
        if (tidx < kNThreads - 1) { smem_exchange[tidx] = reinterpret_cast<vec_t *>(x_vals_load)[1]; }
        __syncthreads();
        reinterpret_cast<vec_t *>(x_vals_load)[0] = smem_exchange[tidx > 0 ? tidx - 1 : kNThreads - 1];
        __syncthreads();
        // Now thread kNThreads - 1 can write the last elements of the current chunk.
        if (tidx == kNThreads - 1) { smem_exchange[tidx] = reinterpret_cast<vec_t *>(x_vals_load)[1]; }

        float x_vals[2 * kNElts];
        #pragma unroll
        for (int i = 0; i < 2 * kNElts; ++i) { x_vals[i] = float(x_vals_load[i]); }

        float out_vals[kNElts];
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) {
            out_vals[i] = bias_val;
            #pragma unroll
            for (int w = 0; w < kWidth; ++w) {
                out_vals[i] += weight_vals[w] * x_vals[kNElts + i - (kWidth - w - 1)];
            }
        }

        if (params.silu_activation) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) {
                out_vals[i] = out_vals[i] / (1 + expf(-out_vals[i]));
            }
        }

        input_t out_vals_store[kNElts];
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) { out_vals_store[i] = out_vals[i]; }
        if constexpr(kIsVecLoad) {
            typename Ktraits::BlockStoreVecT(smem_store_vec).Store(reinterpret_cast<vec_t*>(out), reinterpret_cast<vec_t (&)[1]>(out_vals_store), (seqlen - chunk * kChunkSize) / kNElts);
        } else {
            typename Ktraits::BlockStoreT(smem_store).Store(out, out_vals_store, seqlen - chunk * kChunkSize);
        }
        out += kChunkSize;

        int final_state_position =  ((seqlen - (kWidth - 1)) - (n_chunks - 1) * kChunkSize);
        // in case the final state is separated between the last "smem_exchange" and
        // and the one before it (chunk = n_chunks - 1 and chunk = n_chunks - 2),
        // (which occurs when `final_state_position` is a non-positivie index)
        // we load the correct data from smem_exchange from both chunks, the last chunk iteration and the one before it
        if (conv_states != nullptr && final_state_position < 0 && seqlen > kWidth){
            input_t vals_load[kNElts] = {0};
            if ((chunk == n_chunks - 2) && (tidx == kNThreads - 1)){
                // chunk = n_chunks - 2, a segment of the final state sits in the last index
                reinterpret_cast<vec_t *>(vals_load)[0] = smem_exchange[kNThreads - 1];
                #pragma unroll
                for (int w = 0; w < -final_state_position; ++w){
                    conv_states[w * params.conv_states_l_stride] = vals_load[kNElts + final_state_position + w];
                }
            }
            if ((chunk == n_chunks - 1) && tidx == 0){
                // chunk = n_chunks - 1, the second segment of the final state first positions
                reinterpret_cast<vec_t *>(vals_load)[0] = smem_exchange[0];
                for (int w = -final_state_position; w < kWidth - 1; ++w){
                    conv_states[w * params.conv_states_l_stride] = vals_load[w + final_state_position];
                }
                return;
            }
        }
    }
    // Final state is stored in the smem_exchange last token slot,
    // in case seqlen < kWidth, we would need to take the final state from the
    // initial state which is stored in conv_states
    // in case seqlen > kWidth, we would need to load the last kWidth - 1 data
    // and load it into conv_state accordingly
    int last_thread =  ((seqlen - (kWidth - 1)) - (n_chunks - 1) * kChunkSize) / kNElts;
    if (conv_states != nullptr && tidx == last_thread) {
        input_t x_vals_load[kNElts * 2] = {0};
        // in case we are on the first kWidth tokens
        if (last_thread == 0 && seqlen < kWidth){
            // Need to take the initial state
            reinterpret_cast<vec_t *>(x_vals_load)[0] = smem_exchange[0];
            const int offset = seqlen - (kWidth - 1);
            #pragma unroll
            for (int w = 0; w < kWidth - 1; ++w){
                // pad the existing state
                if ((w - seqlen) >= 0 && has_initial_state) { conv_states[(w - seqlen) * params.conv_states_l_stride] = conv_states[w * params.conv_states_l_stride]; }
                else if ((w - seqlen) >= 0 && !has_initial_state) { conv_states[(w - seqlen) * params.conv_states_l_stride] = input_t(0.0f); }
            }
            #pragma unroll
            for (int w = 0; w < kWidth - 1; ++w){
                if (offset + w >= 0)
                    conv_states[w * params.conv_states_l_stride] = x_vals_load[offset + w ];
            }
        }
        else {
            // in case the final state is in between the threads data
            const int offset = ((seqlen - (kWidth - 1)) % (kNElts));
            if ((offset + kWidth - 2) >= kNElts && (last_thread + 1 < kNThreads)){
                // In Case last_thread == kNThreads - 1, accessing last_thread + 1 will result in a
                // illegal access error on H100.
                // Therefore, we access last_thread + 1, only if the final state data sits there
                reinterpret_cast<vec_t *>(x_vals_load)[1] = smem_exchange[last_thread + 1];
            }
            reinterpret_cast<vec_t *>(x_vals_load)[0] = smem_exchange[last_thread];
            #pragma unroll
            for (int w = 0; w < kWidth - 1; ++w){
                conv_states[w * params.conv_states_l_stride] = x_vals_load[offset + w ];
            }
        }

    }
}


template<int kNThreads, int kWidth, typename input_t, typename weight_t>
void causal_conv1d_fwd_launch(ConvParamsBase &params, cudaStream_t stream) {
    static constexpr int kNElts = sizeof(input_t) == 4 ? 4 : 8;
    const bool kVarlen = params.query_start_loc_ptr != nullptr;
    BOOL_SWITCH(params.seqlen % kNElts == 0 && !kVarlen, kIsVecLoad, [&] {
        using Ktraits = Causal_conv1d_fwd_kernel_traits<kNThreads, kWidth, kIsVecLoad, input_t, weight_t>;
        constexpr int kSmemSize = Ktraits::kSmemSize;
        dim3 grid(params.batch, params.dim);

        auto kernel = &causal_conv1d_fwd_kernel<Ktraits>;

        if (kSmemSize >= 48 * 1024) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
        }
        kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

template<typename input_t, typename weight_t>
void causal_conv1d_fwd_cuda(ConvParamsBase &params, cudaStream_t stream) {
    if (params.width == 2) {
        causal_conv1d_fwd_launch<128, 2, input_t, weight_t>(params, stream);
    } else if (params.width == 3) {
        causal_conv1d_fwd_launch<128, 3, input_t, weight_t>(params, stream);
    } else if (params.width == 4) {
        causal_conv1d_fwd_launch<128, 4, input_t, weight_t>(params, stream);
    }
}

template void causal_conv1d_fwd_cuda<float, float>(ConvParamsBase &params, cudaStream_t stream);
template void causal_conv1d_fwd_cuda<at::Half, at::Half>(ConvParamsBase &params, cudaStream_t stream);
template void causal_conv1d_fwd_cuda<at::BFloat16, at::BFloat16>(ConvParamsBase &params, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////////////////////////
// Update kernel (decode)
////////////////////////////////////////////////////////////////////////////////////////////////////

template<int kNThreads_, int kWidth_, typename input_t_, typename weight_t_>
struct Causal_conv1d_update_kernel_traits {
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kWidth = kWidth_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
};

template<typename Ktraits, bool kIsCircularBuffer>
__global__ __launch_bounds__(Ktraits::kNThreads)
void causal_conv1d_update_kernel(ConvParamsBase params) {
    constexpr int kWidth = Ktraits::kWidth;
    constexpr int kNThreads = Ktraits::kNThreads;
    using input_t = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;

    const int tidx = threadIdx.x;
    const int batch_id = blockIdx.x;
    const int channel_id = blockIdx.y * kNThreads + tidx;
    if (channel_id >= params.dim) return;

    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride
        + channel_id * params.x_c_stride;

    // If params.conv_state_batch_indices is set, then the conv state is gathered from the conv state tensor
    // along the batch axis. Otherwise, the conv state coordinate is the same as the batch id.
    const int conv_state_batch_coord = params.conv_state_indices_ptr == nullptr
        ? batch_id
        : params.conv_state_indices_ptr[batch_id];
    // conv_state_batch_coord == params.pad_slot_id is defined as padding so we exit early
    if (conv_state_batch_coord == params.pad_slot_id){
        return;
    }
    input_t *conv_state = reinterpret_cast<input_t *>(params.conv_state_ptr)
        + conv_state_batch_coord * params.conv_state_batch_stride
        + channel_id * params.conv_state_c_stride;

    weight_t *weight = reinterpret_cast<weight_t *>(params.weight_ptr) + channel_id * params.weight_c_stride;
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
        + channel_id * params.out_c_stride;
    float bias_val = params.bias_ptr == nullptr ? 0.f : float(reinterpret_cast<weight_t *>(params.bias_ptr)[channel_id]);

    int state_len = params.conv_state_len;
    int advance_len = params.seqlen;
    int cache_seqlen = kIsCircularBuffer ? params.cache_seqlens[batch_id] % state_len : 0;
    int update_idx = cache_seqlen - (kWidth - 1);
    update_idx = update_idx < 0 ? update_idx + state_len : update_idx;

    float weight_vals[kWidth] = {0};
    #pragma unroll
    for (int i = 0; i < kWidth; ++i) { weight_vals[i] = float(weight[i * params.weight_width_stride]); }

    float x_vals[kWidth] = {0};
    if constexpr (!kIsCircularBuffer) {
        #pragma unroll 2
        for (int i = 0; i < state_len - advance_len - (kWidth - 1); ++i) {
            conv_state[i * params.conv_state_l_stride] = conv_state[(i + advance_len) * params.conv_state_l_stride];
        }
        #pragma unroll
        for (int i = 0; i < kWidth - 1; ++i) {
            input_t state_val = conv_state[(state_len - (kWidth - 1) + i) * params.conv_state_l_stride];
            if (i < advance_len + (kWidth - 1) && state_len - advance_len - (kWidth - 1) + i >= 0) {
                conv_state[(state_len - advance_len - (kWidth - 1) + i) * params.conv_state_l_stride] = state_val;
            }
            x_vals[i] = float(state_val);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < kWidth - 1; ++i, update_idx = update_idx + 1 >= state_len ? update_idx + 1 - state_len : update_idx + 1) {
            input_t state_val = conv_state[update_idx * params.conv_state_l_stride];
            x_vals[i] = float(state_val);
        }
    }
    #pragma unroll 2
    for (int i = 0; i < params.seqlen; ++i) {
        input_t x_val = x[i * params.x_l_stride];
        if constexpr (!kIsCircularBuffer) {
            if (i < advance_len && state_len - advance_len + i >= 0) {
                conv_state[(state_len - advance_len + i) * params.conv_state_l_stride] = x_val;
            }
        } else {
            conv_state[update_idx * params.conv_state_l_stride] = x_val;
            ++update_idx;
            update_idx = update_idx >= state_len ? update_idx - state_len : update_idx;
        }
        x_vals[kWidth - 1] = float(x_val);
        float out_val = bias_val;
        #pragma unroll
        for (int j = 0; j < kWidth; ++j) { out_val += weight_vals[j] * x_vals[j]; }
        if (params.silu_activation) { out_val = out_val / (1 + expf(-out_val)); }
        out[i * params.out_l_stride] = input_t(out_val);
        // Shift the input buffer by 1
        #pragma unroll
        for (int i = 0; i < kWidth - 1; ++i) { x_vals[i] = x_vals[i + 1]; }
    }
}

template<int kNThreads, int kWidth, typename input_t, typename weight_t>
void causal_conv1d_update_launch(ConvParamsBase &params, cudaStream_t stream) {
    using Ktraits = Causal_conv1d_update_kernel_traits<kNThreads, kWidth, input_t, weight_t>;
    dim3 grid(params.batch, (params.dim + kNThreads - 1) / kNThreads);
    auto kernel = params.cache_seqlens == nullptr
        ? &causal_conv1d_update_kernel<Ktraits, false>
        : &causal_conv1d_update_kernel<Ktraits, true>;
    kernel<<<grid, Ktraits::kNThreads, 0, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename input_t, typename weight_t>
void causal_conv1d_update_cuda(ConvParamsBase &params, cudaStream_t stream) {
    if (params.width == 2) {
        causal_conv1d_update_launch<64, 2, input_t, weight_t>(params, stream);
    } else if (params.width == 3) {
        causal_conv1d_update_launch<64, 3, input_t, weight_t>(params, stream);
    } else if (params.width == 4) {
        causal_conv1d_update_launch<64, 4, input_t, weight_t>(params, stream);
    }
}

template void causal_conv1d_update_cuda<float, float>(ConvParamsBase &params, cudaStream_t stream);
template void causal_conv1d_update_cuda<at::Half, at::Half>(ConvParamsBase &params, cudaStream_t stream);
template void causal_conv1d_update_cuda<at::BFloat16, at::BFloat16>(ConvParamsBase &params, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("causal_conv1d_fwd", &causal_conv1d_fwd,
          "Causal conv1d forward (prefill), in place. x: (dim, total_tokens) "
          "channel-major varlen with query_start_loc, or (batch, dim, seqlen). "
          "Final conv state written into the pool slot given by cache_indices.",
          py::arg("x"), py::arg("weight"), py::arg("bias"),
          py::arg("conv_states"), py::arg("query_start_loc"),
          py::arg("cache_indices"), py::arg("has_initial_state"),
          py::arg("silu_activation"), py::arg("pad_slot_id"));
    m.def("causal_conv1d_update", &causal_conv1d_update,
          "Causal conv1d decode update. x: (batch, dim[, seqlen]); conv_state "
          "pool (num_slots, dim, state_len) updated in place at "
          "conv_state_indices.",
          py::arg("x"), py::arg("conv_state"), py::arg("weight"),
          py::arg("bias"), py::arg("silu_activation"),
          py::arg("cache_seqlens"), py::arg("conv_state_indices"),
          py::arg("pad_slot_id"));
}
