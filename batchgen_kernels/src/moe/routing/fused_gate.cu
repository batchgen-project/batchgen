/*
 * Fused Gate: WGMMA Router GEMM + Bias + TopK + Softmax (SM90a).
 *
 * 2-kernel design replacing cuBLAS mm + router_bias_cast + gate_topk_softmax:
 *   Kernel A: WGMMA GEMM [N, K_dim] × [E, K_dim]^T → [N, E] FP32 + BF16 bias
 *   Kernel B: TopK + Softmax (1 block per token, 256 threads)
 *
 * Requires FusedGateContext for cached weight + TMA descriptors.
 * All kernel launches use getCurrentCUDAStream() for CUDA graph compatibility.
 *
 * Supports num_valid_tokens to skip padding tokens in CUDA graph capture.
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <float.h>
#include "wgmma_common.cuh"
#include "routing_ops.h"


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
// FusedGateContext: cache weight_t, TMA desc B, encode_func at init time
// ============================================================================

struct FusedGateContext {
    torch::Tensor weight_t;          // [E, K_dim] BF16, kept contiguous (no transpose)
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
    int cached_input_rows;           // declared rows in cached TMA desc (full buffer extent)
};

int64_t create_fused_gate_context(
    torch::Tensor router_weight,    // [E, K_dim] BF16 (nn.Linear weight)
    torch::Tensor router_bias,      // [E] BF16 (or empty)
    int topk
) {
    TORCH_CHECK(router_weight.is_cuda() && router_weight.dtype() == torch::kBFloat16,
                "router_weight must be CUDA BF16");

    auto ctx = new FusedGateContext();
    // nn.Linear weight is [out_features, in_features] = [E, K_dim]
    ctx->E = router_weight.size(0);
    ctx->K_dim = router_weight.size(1);
    ctx->topk = topk;

    // Keep weight contiguous as [E, K_dim] — WGMMA B operand needs [N, K] layout
    // (WGMMA internally computes A × B^T, so B=[E, K] gives the correct matmul)
    ctx->weight_t = router_weight.contiguous();  // [E, K_dim]

    // Cache encode function (dlsym lookup done once)
    ctx->encode_func = get_cuTensorMapEncodeTiled();

    // Cache TMA descriptor for B: [E, K_dim] with gmem_rows=E, gmem_cols=K_dim
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

    // Set smem attribute eagerly (not during graph capture)
    cudaFuncSetAttribute(
        wgmma_router_gemm_bias_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        ctx->smem_bytes);
    ctx->smem_attr_set = true;
    ctx->has_cached_tma_a = false;
    ctx->cached_input_ptr = nullptr;
    ctx->cached_input_rows = 0;

    return reinterpret_cast<int64_t>(ctx);
}

void destroy_fused_gate_context(int64_t ctx_ptr) {
    delete reinterpret_cast<FusedGateContext*>(ctx_ptr);
}

void fused_gate_warmup(int64_t ctx_ptr, torch::Tensor base_buffer) {
    // Create TMA descriptor for A against the full base buffer.
    // Call once with the base buffer (e.g., SharedMoEBufferPool's b["all_tokens"]).
    // All per-bucket views share the same data_ptr, so one TMA desc covers all buckets.
    // The kernel grid controls which rows are actually processed per bucket.
    auto* ctx = reinterpret_cast<FusedGateContext*>(ctx_ptr);
    void* ptr = base_buffer.data_ptr();
    int rows = base_buffer.size(0);
    ctx->tma_desc_a = make_2d_tma_desc_bf16(
        reinterpret_cast<__nv_bfloat16*>(ptr),
        rows, ctx->K_dim, BLOCK_M, BLOCK_K, ctx->encode_func);
    ctx->has_cached_tma_a = true;
    ctx->cached_input_ptr = ptr;
    ctx->cached_input_rows = rows;
}

torch::Tensor fused_router_forward(
    int64_t ctx_ptr,
    torch::Tensor hidden_states,
    torch::Tensor logits,
    int64_t num_valid_tokens
) {
    auto* ctx = reinterpret_cast<FusedGateContext*>(ctx_ptr);
    TORCH_CHECK(hidden_states.is_cuda() && hidden_states.dtype() == torch::kBFloat16,
                "hidden_states must be CUDA BF16");

    const int N = hidden_states.size(0);
    const int K_dim = ctx->K_dim;
    const int E = ctx->E;
    auto device = hidden_states.device();
    const int N_eff = (num_valid_tokens > 0 && num_valid_tokens < N)
                      ? static_cast<int>(num_valid_tokens) : N;

    if (!logits.defined() || logits.numel() == 0) {
        logits = torch::empty({N, E}, torch::dtype(torch::kFloat32).device(device));
    }

    void* input_ptr = hidden_states.data_ptr();
    const int input_rows = hidden_states.size(0);
    if (!ctx->has_cached_tma_a ||
        ctx->cached_input_ptr != input_ptr ||
        input_rows > ctx->cached_input_rows) {
        ctx->tma_desc_a = make_2d_tma_desc_bf16(
            reinterpret_cast<__nv_bfloat16*>(input_ptr),
            input_rows, ctx->K_dim, BLOCK_M, BLOCK_K, ctx->encode_func);
        ctx->has_cached_tma_a = true;
        ctx->cached_input_ptr = input_ptr;
        ctx->cached_input_rows = input_rows;
    }

    const __nv_bfloat16* bias_ptr = ctx->has_bias
        ? reinterpret_cast<const __nv_bfloat16*>(ctx->router_bias.data_ptr())
        : nullptr;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int num_m_tiles = (N_eff + BLOCK_M - 1) / BLOCK_M;
    const int num_n_tiles = (E + BLOCK_N - 1) / BLOCK_N;

    wgmma_router_gemm_bias_kernel<<<
        dim3(num_m_tiles, num_n_tiles),
        TOTAL_THREADS,
        ctx->smem_bytes,
        stream>>>(
            ctx->tma_desc_a,
            ctx->tma_desc_b,
            bias_ptr,
            logits.data_ptr<float>(),
            N_eff,
            K_dim,
            E);
    return logits;
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
    const int N = hidden_states.size(0);
    const int topk = ctx->topk;
    auto device = hidden_states.device();

    // Effective token count for CUDA graph compatibility
    const int N_eff = (num_valid_tokens > 0 && num_valid_tokens < N)
                      ? static_cast<int>(num_valid_tokens) : N;

    if (!topk_indices.defined() || topk_indices.numel() == 0) {
        topk_indices = torch::empty({N, topk}, torch::dtype(torch::kInt32).device(device));
    }
    if (!topk_weights.defined() || topk_weights.numel() == 0) {
        topk_weights = torch::empty({N, topk}, torch::dtype(torch::kFloat32).device(device));
    }

    logits = fused_router_forward(
        ctx_ptr,
        hidden_states,
        logits,
        num_valid_tokens);

    // Kernel B: TopK + Softmax (reuse standalone gate kernel)
    {
        auto result = gate_topk_softmax_cuda(
            logits, topk, topk_indices, topk_weights, N_eff);
        topk_indices = result[0];
        topk_weights = result[1];
    }

    return {topk_indices, topk_weights};
}
