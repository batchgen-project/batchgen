#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>

namespace {

constexpr int WARP_SIZE        = 32;
constexpr int WARPS_PER_CTA    = 4;
constexpr int THREADS_PER_CTA  = WARP_SIZE * WARPS_PER_CTA;
constexpr int VEC_BYTES        = 16;
constexpr int ELEMS_PER_LOAD   = VEC_BYTES / sizeof(__nv_bfloat16);


template <int N_HEADS, int HEAD_DIM, int KV_DIM>
__global__ __launch_bounds__(THREADS_PER_CTA, 16)
void fused_qk_rmsnorm_kernel(
    const __nv_bfloat16* __restrict__ qr,
    __nv_bfloat16*       __restrict__ qr_out,
    int64_t                            qr_stride_t,
    int64_t                            qr_stride_h,
    int64_t                            qr_out_stride_t,
    int64_t                            qr_out_stride_h,
    const __nv_bfloat16* __restrict__ kv,
    __nv_bfloat16*       __restrict__ kv_out,
    int64_t                            kv_stride_t,
    int64_t                            kv_out_stride_t,
    const float*         __restrict__ kv_weight,
    float                              eps)
{
    static_assert(HEAD_DIM % (ELEMS_PER_LOAD * WARP_SIZE) == 0,
                  "HEAD_DIM must be a multiple of 256");
    static_assert(KV_DIM   % (ELEMS_PER_LOAD * WARP_SIZE) == 0,
                  "KV_DIM must be a multiple of 256");

    constexpr int Q_LOADS_PER_THREAD  = HEAD_DIM / (ELEMS_PER_LOAD * WARP_SIZE);
    constexpr int KV_LOADS_PER_THREAD = KV_DIM   / (ELEMS_PER_LOAD * WARP_SIZE);

    const int warp_in_cta = threadIdx.y;
    const int lane        = threadIdx.x;
    const int token       = blockIdx.x;
    const int task_base   = blockIdx.y * WARPS_PER_CTA;
    const int task        = task_base + warp_in_cta;

    if (task > N_HEADS) return;

    if (task < N_HEADS) {
        const int head = task;
        const __nv_bfloat16* in_row =
            qr     + token * qr_stride_t     + head * qr_stride_h;
        __nv_bfloat16*       out_row =
            qr_out + token * qr_out_stride_t + head * qr_out_stride_h;

        uint4 vecs[Q_LOADS_PER_THREAD];
        float local_sum_sq = 0.f;

        #pragma unroll
        for (int i = 0; i < Q_LOADS_PER_THREAD; ++i) {
            const int base_elem = i * WARP_SIZE * ELEMS_PER_LOAD
                                + lane * ELEMS_PER_LOAD;
            vecs[i] = *reinterpret_cast<const uint4*>(in_row + base_elem);
            const __nv_bfloat162* pairs =
                reinterpret_cast<const __nv_bfloat162*>(&vecs[i]);
            #pragma unroll
            for (int j = 0; j < ELEMS_PER_LOAD / 2; ++j) {
                float2 v = __bfloat1622float2(pairs[j]);
                local_sum_sq += v.x * v.x + v.y * v.y;
            }
        }

        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            local_sum_sq += __shfl_xor_sync(0xffffffff, local_sum_sq, offset);
        }
        const float inv_rms = rsqrtf(local_sum_sq / HEAD_DIM + eps);

        #pragma unroll
        for (int i = 0; i < Q_LOADS_PER_THREAD; ++i) {
            __nv_bfloat162* pairs =
                reinterpret_cast<__nv_bfloat162*>(&vecs[i]);
            #pragma unroll
            for (int j = 0; j < ELEMS_PER_LOAD / 2; ++j) {
                float2 v = __bfloat1622float2(pairs[j]);
                v.x *= inv_rms;
                v.y *= inv_rms;
                pairs[j] = __float22bfloat162_rn(v);
            }
            const int base_elem = i * WARP_SIZE * ELEMS_PER_LOAD
                                + lane * ELEMS_PER_LOAD;
            *reinterpret_cast<uint4*>(out_row + base_elem) = vecs[i];
        }
    } else {
        const __nv_bfloat16* in_row  = kv     + token * kv_stride_t;
        __nv_bfloat16*       out_row = kv_out + token * kv_out_stride_t;

        uint4 vecs[KV_LOADS_PER_THREAD];
        float local_sum_sq = 0.f;

        #pragma unroll
        for (int i = 0; i < KV_LOADS_PER_THREAD; ++i) {
            const int base_elem = i * WARP_SIZE * ELEMS_PER_LOAD
                                + lane * ELEMS_PER_LOAD;
            vecs[i] = *reinterpret_cast<const uint4*>(in_row + base_elem);
            const __nv_bfloat162* pairs =
                reinterpret_cast<const __nv_bfloat162*>(&vecs[i]);
            #pragma unroll
            for (int j = 0; j < ELEMS_PER_LOAD / 2; ++j) {
                float2 v = __bfloat1622float2(pairs[j]);
                local_sum_sq += v.x * v.x + v.y * v.y;
            }
        }

        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            local_sum_sq += __shfl_xor_sync(0xffffffff, local_sum_sq, offset);
        }
        const float inv_rms = rsqrtf(local_sum_sq / KV_DIM + eps);

        #pragma unroll
        for (int i = 0; i < KV_LOADS_PER_THREAD; ++i) {
            const int base_elem = i * WARP_SIZE * ELEMS_PER_LOAD
                                + lane * ELEMS_PER_LOAD;
            const float4 w_lo =
                *reinterpret_cast<const float4*>(kv_weight + base_elem);
            const float4 w_hi =
                *reinterpret_cast<const float4*>(kv_weight + base_elem + 4);
            __nv_bfloat162* pairs =
                reinterpret_cast<__nv_bfloat162*>(&vecs[i]);
            float2 v0 = __bfloat1622float2(pairs[0]);
            float2 v1 = __bfloat1622float2(pairs[1]);
            float2 v2 = __bfloat1622float2(pairs[2]);
            float2 v3 = __bfloat1622float2(pairs[3]);
            v0.x = v0.x * inv_rms * w_lo.x;
            v0.y = v0.y * inv_rms * w_lo.y;
            v1.x = v1.x * inv_rms * w_lo.z;
            v1.y = v1.y * inv_rms * w_lo.w;
            v2.x = v2.x * inv_rms * w_hi.x;
            v2.y = v2.y * inv_rms * w_hi.y;
            v3.x = v3.x * inv_rms * w_hi.z;
            v3.y = v3.y * inv_rms * w_hi.w;
            pairs[0] = __float22bfloat162_rn(v0);
            pairs[1] = __float22bfloat162_rn(v1);
            pairs[2] = __float22bfloat162_rn(v2);
            pairs[3] = __float22bfloat162_rn(v3);
            *reinterpret_cast<uint4*>(out_row + base_elem) = vecs[i];
        }
    }
}


std::vector<torch::Tensor> fused_qk_rmsnorm_forward(
    torch::Tensor qr,
    torch::Tensor kv,
    torch::Tensor kv_weight,
    double eps)
{
    TORCH_CHECK(qr.is_cuda() && kv.is_cuda() && kv_weight.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(qr.dtype() == torch::kBFloat16, "qr must be bf16");
    TORCH_CHECK(kv.dtype() == torch::kBFloat16, "kv must be bf16");
    TORCH_CHECK(kv_weight.dtype() == torch::kFloat32, "kv_weight must be fp32");
    TORCH_CHECK(qr.dim() == 3, "qr must be [T, n_heads, head_dim]");
    TORCH_CHECK(kv.dim() == 2, "kv must be [T, kv_dim]");
    TORCH_CHECK(qr.stride(-1) == 1, "qr inner dim must be contiguous");
    TORCH_CHECK(kv.stride(-1) == 1, "kv inner dim must be contiguous");
    TORCH_CHECK(kv_weight.is_contiguous(), "kv_weight must be contiguous");
    TORCH_CHECK(qr.size(0) == kv.size(0), "token dim mismatch");
    TORCH_CHECK(kv.size(1) == kv_weight.size(0), "kv_weight dim mismatch");

    const int T        = qr.size(0);
    const int n_heads  = qr.size(1);
    const int head_dim = qr.size(2);
    const int kv_dim   = kv.size(1);

    auto qr_out = torch::empty_like(qr);
    auto kv_out = torch::empty_like(kv);
    if (T == 0) {
        return {qr_out, kv_out};
    }

    TORCH_CHECK(head_dim == 512 && kv_dim == 512 &&
                (n_heads == 64 || n_heads == 128),
                "fused_qk_rmsnorm_cuda: unsupported shape (n_heads=", n_heads,
                ", head_dim=", head_dim, ", kv_dim=", kv_dim,
                "). Supported: (n_heads=64|128, head_dim=512, kv_dim=512).");

    auto stream = at::cuda::getCurrentCUDAStream();
    auto qr_ptr     = reinterpret_cast<const __nv_bfloat16*>(qr.data_ptr());
    auto qr_out_ptr = reinterpret_cast<__nv_bfloat16*>(qr_out.data_ptr());
    auto kv_ptr     = reinterpret_cast<const __nv_bfloat16*>(kv.data_ptr());
    auto kv_out_ptr = reinterpret_cast<__nv_bfloat16*>(kv_out.data_ptr());
    auto kvw_ptr    = kv_weight.data_ptr<float>();

    const int tasks_total = n_heads + 1;
    const int cta_tasks   = (tasks_total + WARPS_PER_CTA - 1) / WARPS_PER_CTA;
    dim3 grid(T, cta_tasks);
    dim3 block(WARP_SIZE, WARPS_PER_CTA);

    if (n_heads == 64) {
        fused_qk_rmsnorm_kernel<64, 512, 512>
            <<<grid, block, 0, stream>>>(
                qr_ptr, qr_out_ptr,
                qr.stride(0), qr.stride(1),
                qr_out.stride(0), qr_out.stride(1),
                kv_ptr, kv_out_ptr,
                kv.stride(0), kv_out.stride(0),
                kvw_ptr, static_cast<float>(eps));
    } else {  // n_heads == 128 (validated above)
        fused_qk_rmsnorm_kernel<128, 512, 512>
            <<<grid, block, 0, stream>>>(
                qr_ptr, qr_out_ptr,
                qr.stride(0), qr.stride(1),
                qr_out.stride(0), qr_out.stride(1),
                kv_ptr, kv_out_ptr,
                kv.stride(0), kv_out.stride(0),
                kvw_ptr, static_cast<float>(eps));
    }

    return {qr_out, kv_out};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_qk_rmsnorm_forward", &fused_qk_rmsnorm_forward,
          "Fused per-head Q + global KV RMSNorm (CUDA, bf16)");
}
