/*
 * Fused Gate: WGMMA Router GEMM + Bias + TopK + Softmax (SM90a).
 *
 * 2-kernel design replacing cuBLAS mm + router_bias_cast + gate_topk_softmax:
 *   Kernel A: WGMMA GEMM [N, K_dim] × [E, K_dim]^T → [N, E] FP32 + BF16 bias
 *   Kernel B: TopK + Softmax (1 block per token, 256 threads)
 *
 * Requires FusedGateContext for cached weight transpose + TMA descriptors.
 * All kernel launches use getCurrentCUDAStream() for CUDA graph compatibility.
 *
 * Supports num_valid_tokens to skip padding tokens in CUDA graph capture.
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <float.h>
#include "wgmma_common.cuh"

#define MAX_EXPERTS 128
#define MAX_TOPK 8
#define GATE_THREADS 256


// ============================================================================
// Kernel A: WGMMA Router GEMM + BF16 Bias Epilogue
// ============================================================================

__global__ void __launch_bounds__(TOTAL_THREADS, 1)
wgmma_router_gemm_bias_kernel(
    const __grid_constant__ CUtensorMap tma_desc_a,   // [N_padded, K_dim]
    const __grid_constant__ CUtensorMap tma_desc_b,   // [E, K_dim]
    const __nv_bfloat16* __restrict__ router_bias,    // [E] BF16 (nullptr if none)
    float* __restrict__ output,                        // [N, E] FP32
    int N,
    int K_dim,
    int E
) {
    const int tid = threadIdx.x;
    const int wg_id = tid / 128;
    const int wg_tid = tid % 128;
    const int warp_in_wg = wg_tid / WARP_SIZE;
    const int lane_id = wg_tid % WARP_SIZE;

    const int m_tile = blockIdx.x;
    const int n_tile = blockIdx.y;
    const int m_start = m_tile * BLOCK_M;
    const int n_start = n_tile * BLOCK_N;

    if (m_start >= N || n_start >= E) return;

    // ── Shared memory layout ──
    extern __shared__ __align__(128) char smem_buf[];
    __nv_bfloat16* smem_a[NUM_STAGES];
    __nv_bfloat16* smem_b[NUM_STAGES];
    for (int s = 0; s < NUM_STAGES; s++) {
        smem_a[s] = reinterpret_cast<__nv_bfloat16*>(smem_buf + (2*s)     * TILE_BYTES_A);
        smem_b[s] = reinterpret_cast<__nv_bfloat16*>(smem_buf + (2*s + 1) * TILE_BYTES_B);
    }
    const int bar_offset = 2 * NUM_STAGES * TILE_BYTES_A;
    uint64_t* full_barriers  = reinterpret_cast<uint64_t*>(smem_buf + bar_offset);
    uint64_t* empty_barriers = full_barriers + NUM_STAGES;

    const int num_k_blocks = (K_dim + BLOCK_K - 1) / BLOCK_K;

    // ── Init barriers ──
    if (tid == 0) {
        for (int s = 0; s < NUM_STAGES; s++) {
            mbarrier_init(&full_barriers[s], 1);
            mbarrier_init(&empty_barriers[s], 1);
        }
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();

    if (wg_id == 0) {
        // ── PRODUCER WG ──
        if (wg_tid == 0) {
            asm volatile("prefetch.tensormap [%0];" :: "l"(&tma_desc_a) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"(&tma_desc_b) : "memory");
        }
        bar_sync(PRODUCER_BAR_ID, PRODUCER_THREADS);

        for (int kb = 0; kb < num_k_blocks; kb++) {
            const int s = kb % NUM_STAGES;
            const int empty_phase = ((kb / NUM_STAGES) + 1) & 1;

            mbarrier_wait_parity(&empty_barriers[s], empty_phase);

            if (wg_tid == 0) {
                uint32_t tx_bytes = TILE_BYTES_A + TILE_BYTES_B;
                mbarrier_arrive_expect_tx(&full_barriers[s], tx_bytes);
                tma_load_2d(&tma_desc_a, &full_barriers[s], smem_a[s],
                            kb * BLOCK_K, m_start);
                tma_load_2d(&tma_desc_b, &full_barriers[s], smem_b[s],
                            kb * BLOCK_K, n_start);
            }
            bar_sync(PRODUCER_BAR_ID, PRODUCER_THREADS);
        }
    } else {
        // ── CONSUMER WG ──
        float acc[WGMMA_NUM_ACCUM];
        for (int i = 0; i < WGMMA_NUM_ACCUM; i++) acc[i] = 0.0f;

        for (int kb = 0; kb < num_k_blocks; kb++) {
            const int s = kb % NUM_STAGES;
            const int full_phase = (kb / NUM_STAGES) & 1;

            mbarrier_wait_parity(&full_barriers[s], full_phase);

            #pragma unroll
            for (int i = 0; i < WGMMA_NUM_ACCUM; i++)
                warpgroup_fence_operand(acc[i]);

            warpgroup_arrive();

            #pragma unroll
            for (int t = 0; t < TILES_K; t++) {
                GmmaDescriptor da = make_smem_desc(smem_a[s] + t * WGMMA_K);
                GmmaDescriptor db = make_smem_desc(smem_b[s] + t * WGMMA_K);
                wgmma_bf16_ss(da.desc_, db.desc_, acc, 1);
            }

            warpgroup_commit_batch();

            #pragma unroll
            for (int i = 0; i < WGMMA_NUM_ACCUM; i++)
                warpgroup_fence_operand(acc[i]);

            warpgroup_wait<0>();

            if (wg_tid == 0) {
                mbarrier_arrive(&empty_barriers[s]);
            }
        }

        // ── Epilogue: write FP32 output + bias ──
        #pragma unroll 1
        for (int i = 0; i < WGMMA_NUM_ACCUM; i++) {
            int m, n;
            reg_to_mn(i, warp_in_wg, lane_id, m, n);
            int m_global = m_start + m;
            int n_global = n_start + n;
            if (m_global < N && n_global < E) {
                float val = acc[i];
                if (router_bias != nullptr) {
                    val += __bfloat162float(router_bias[n_global]);
                }
                output[m_global * E + n_global] = val;
            }
        }
    }
}


// ============================================================================
// Kernel B: TopK + Softmax (1 block per token)
// ============================================================================

template <int K>
__global__ void fused_gate_topk_softmax_kernel(
    const float* __restrict__ router_logits,  // [N, E]
    int32_t* __restrict__ topk_indices,       // [N, K]
    float* __restrict__ topk_weights,         // [N, K]
    int N, int E
) {
    const int token_id = blockIdx.x;
    if (token_id >= N) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = GATE_THREADS / WARP_SIZE;

    __shared__ float s_logits[MAX_EXPERTS];
    __shared__ float s_warp_vals[8];
    __shared__ int s_warp_idxs[8];
    __shared__ float s_topk_vals[8];
    __shared__ int s_topk_idxs[8];

    // Load logits to shared memory (coalesced)
    const float* token_logits = router_logits + token_id * E;
    for (int i = tid; i < E; i += GATE_THREADS) {
        s_logits[i] = token_logits[i];
    }
    __syncthreads();

    // Iterative argmax: find top-K by successive max + masking
    #pragma unroll
    for (int k = 0; k < K; k++) {
        float local_max = -FLT_MAX;
        int local_idx = -1;
        for (int i = tid; i < E; i += GATE_THREADS) {
            float v = s_logits[i];
            if (v > local_max || (v == local_max && i < local_idx)) {
                local_max = v;
                local_idx = i;
            }
        }

        warp_reduce_argmax(local_max, local_idx);

        if (lane_id == 0) {
            s_warp_vals[warp_id] = local_max;
            s_warp_idxs[warp_id] = local_idx;
        }
        __syncthreads();

        if (warp_id == 0) {
            float val = (lane_id < num_warps) ? s_warp_vals[lane_id] : -FLT_MAX;
            int idx = (lane_id < num_warps) ? s_warp_idxs[lane_id] : -1;
            warp_reduce_argmax(val, idx);

            if (lane_id == 0) {
                s_topk_vals[k] = val;
                s_topk_idxs[k] = idx;
                s_logits[idx] = -FLT_MAX;
            }
        }
        __syncthreads();
    }

    // Numerically stable softmax over K selected values (thread 0 only)
    if (tid == 0) {
        float max_val = s_topk_vals[0];
        #pragma unroll
        for (int k = 1; k < K; k++)
            max_val = fmaxf(max_val, s_topk_vals[k]);

        float sum_exp = 0.0f;
        float exp_vals[MAX_TOPK];
        #pragma unroll
        for (int k = 0; k < K; k++) {
            exp_vals[k] = expf(s_topk_vals[k] - max_val);
            sum_exp += exp_vals[k];
        }

        float inv_sum = 1.0f / sum_exp;
        int32_t* out_idx = topk_indices + token_id * K;
        float* out_w = topk_weights + token_id * K;
        #pragma unroll
        for (int k = 0; k < K; k++) {
            out_idx[k] = s_topk_idxs[k];
            out_w[k] = exp_vals[k] * inv_sum;
        }
    }
}


// ============================================================================
// FusedGateContext: cache weight_t, TMA desc B, encode_func at init time
// ============================================================================

struct FusedGateContext {
    torch::Tensor weight_t;          // [E, K_dim] BF16, cached transpose
    torch::Tensor router_bias;       // [E] BF16, kept alive
    CUtensorMap tma_desc_b;          // cached TMA descriptor for weight
    CUtensorMap tma_desc_a;          // cached TMA descriptor for input (graph mode)
    PFN_cuTensorMapEncodeTiled encode_func;
    int K_dim;
    int E;
    int topk;
    int smem_bytes;
    bool has_bias;
    bool smem_attr_set;
    bool has_cached_tma_a;           // true if tma_desc_a is valid
    void* cached_input_ptr;          // data_ptr of cached input (for validation)
    int cached_input_N;              // N dimension of cached input
};

int64_t create_fused_gate_context(
    torch::Tensor router_weight,    // [K_dim, E] BF16
    torch::Tensor router_bias,      // [E] BF16 (or empty)
    int topk
) {
    TORCH_CHECK(router_weight.is_cuda() && router_weight.dtype() == torch::kBFloat16,
                "router_weight must be CUDA BF16");

    auto ctx = new FusedGateContext();
    ctx->K_dim = router_weight.size(0);
    ctx->E = router_weight.size(1);
    ctx->topk = topk;

    // Cache weight transpose (done once at model init, not per call)
    ctx->weight_t = router_weight.t().contiguous();  // [E, K_dim]

    // Cache encode function (dlsym lookup done once)
    ctx->encode_func = get_cuTensorMapEncodeTiled();

    // Cache TMA descriptor for B (weight is fixed, pointer must remain valid)
    ctx->tma_desc_b = make_2d_tma_desc_bf16(
        reinterpret_cast<__nv_bfloat16*>(ctx->weight_t.data_ptr()),
        ctx->E, ctx->K_dim, BLOCK_N, BLOCK_K, ctx->encode_func);

    // Cache bias
    ctx->has_bias = router_bias.defined() && router_bias.numel() > 0;
    if (ctx->has_bias) {
        ctx->router_bias = router_bias;
    }

    // Cache smem bytes
    ctx->smem_bytes = 2 * NUM_STAGES * TILE_BYTES_A + NUM_STAGES * 2 * sizeof(uint64_t);
    ctx->smem_attr_set = false;
    ctx->has_cached_tma_a = false;
    ctx->cached_input_ptr = nullptr;
    ctx->cached_input_N = 0;

    return reinterpret_cast<int64_t>(ctx);
}

void destroy_fused_gate_context(int64_t ctx_ptr) {
    delete reinterpret_cast<FusedGateContext*>(ctx_ptr);
}

std::vector<torch::Tensor> fused_gate_forward(
    int64_t ctx_ptr,
    torch::Tensor hidden_states,     // [N, K_dim] BF16
    torch::Tensor logits,            // [N, E] FP32 pre-allocated (optional)
    torch::Tensor topk_indices,      // [N, topk] int32 pre-allocated (optional)
    torch::Tensor topk_weights,      // [N, topk] FP32 pre-allocated (optional)
    int64_t num_valid_tokens
) {
    auto* ctx = reinterpret_cast<FusedGateContext*>(ctx_ptr);
    TORCH_CHECK(hidden_states.is_cuda() && hidden_states.dtype() == torch::kBFloat16,
                "hidden_states must be CUDA BF16");

    const int N = hidden_states.size(0);
    const int K_dim = ctx->K_dim;
    const int E = ctx->E;
    const int topk = ctx->topk;
    auto device = hidden_states.device();

    // Effective token count for CUDA graph compatibility
    const int N_eff = (num_valid_tokens > 0 && num_valid_tokens < N)
                      ? static_cast<int>(num_valid_tokens) : N;

    // Allocate outputs if not pre-allocated
    if (!logits.defined() || logits.numel() == 0) {
        logits = torch::empty({N, E}, torch::dtype(torch::kFloat32).device(device));
    }
    if (!topk_indices.defined() || topk_indices.numel() == 0) {
        topk_indices = torch::empty({N, topk}, torch::dtype(torch::kInt32).device(device));
    }
    if (!topk_weights.defined() || topk_weights.numel() == 0) {
        topk_weights = torch::empty({N, topk}, torch::dtype(torch::kFloat32).device(device));
    }

    // TMA descriptor for A (input): cached when input buffer address is stable
    // (CUDA graph mode uses fixed GPU addresses from SharedMoEBufferPool).
    // TMA with CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA auto-fills
    // OOB reads with zero, so partial M-tiles at the boundary are handled.
    void* input_ptr = hidden_states.data_ptr();
    CUtensorMap tma_a;
    if (ctx->has_cached_tma_a && ctx->cached_input_ptr == input_ptr && ctx->cached_input_N == N) {
        tma_a = ctx->tma_desc_a;
    } else {
        tma_a = make_2d_tma_desc_bf16(
            reinterpret_cast<__nv_bfloat16*>(input_ptr),
            N, K_dim, BLOCK_M, BLOCK_K, ctx->encode_func);
        ctx->tma_desc_a = tma_a;
        ctx->has_cached_tma_a = true;
        ctx->cached_input_ptr = input_ptr;
        ctx->cached_input_N = N;
    }

    const __nv_bfloat16* bias_ptr = ctx->has_bias
        ? reinterpret_cast<const __nv_bfloat16*>(ctx->router_bias.data_ptr())
        : nullptr;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Set smem attribute once (lazy init, not per-call)
    if (!ctx->smem_attr_set) {
        cudaFuncSetAttribute(
            wgmma_router_gemm_bias_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            ctx->smem_bytes);
        ctx->smem_attr_set = true;
    }

    // Kernel A: WGMMA GEMM + bias (only process valid token tiles)
    const int num_m_tiles = (N_eff + BLOCK_M - 1) / BLOCK_M;
    const int num_n_tiles = (E + BLOCK_N - 1) / BLOCK_N;

    wgmma_router_gemm_bias_kernel<<<dim3(num_m_tiles, num_n_tiles), TOTAL_THREADS, ctx->smem_bytes, stream>>>(
        tma_a, ctx->tma_desc_b, bias_ptr,
        logits.data_ptr<float>(),
        N_eff, K_dim, E);

    // Kernel B: TopK + Softmax (only valid tokens)
    switch (topk) {
        case 2:
            fused_gate_topk_softmax_kernel<2><<<N_eff, GATE_THREADS, 0, stream>>>(
                logits.data_ptr<float>(),
                topk_indices.data_ptr<int32_t>(),
                topk_weights.data_ptr<float>(),
                N_eff, E);
            break;
        case 4:
            fused_gate_topk_softmax_kernel<4><<<N_eff, GATE_THREADS, 0, stream>>>(
                logits.data_ptr<float>(),
                topk_indices.data_ptr<int32_t>(),
                topk_weights.data_ptr<float>(),
                N_eff, E);
            break;
        case 8:
            fused_gate_topk_softmax_kernel<8><<<N_eff, GATE_THREADS, 0, stream>>>(
                logits.data_ptr<float>(),
                topk_indices.data_ptr<int32_t>(),
                topk_weights.data_ptr<float>(),
                N_eff, E);
            break;
        default:
            TORCH_CHECK(false, "Unsupported topk=", topk, ". Supported: 2, 4, 8");
    }

    return {topk_indices, topk_weights};
}
