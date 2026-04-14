#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

__global__ void fused_q_split_kernel(
    const __nv_bfloat16* __restrict__ q_flat,  // [B, H * q_head_dim]
    __nv_bfloat16* __restrict__ q_nope,        // [B, H, nope_dim]
    __nv_bfloat16* __restrict__ q_pe,          // [B, H, 1, rope_dim]
    int B, int H, int q_head_dim, int nope_dim, int rope_dim
) {
    int head = blockIdx.x;
    int batch = blockIdx.y;
    if (batch >= B) return;

    int tid = threadIdx.x;

    // Source offset: q_flat[batch, head * q_head_dim + ...]
    int src_base = batch * H * q_head_dim + head * q_head_dim;

    // Copy nope part: q_flat[..., 0:nope_dim] -> q_nope[batch, head, :]
    if (tid < nope_dim) {
        int dst_idx = batch * H * nope_dim + head * nope_dim + tid;
        q_nope[dst_idx] = q_flat[src_base + tid];
    }

    // Copy rope part: q_flat[..., nope_dim:nope_dim+rope_dim] -> q_pe[batch, head, 0, :]
    if (tid < rope_dim) {
        int dst_idx = batch * H * rope_dim + head * rope_dim + tid;
        q_pe[dst_idx] = q_flat[src_base + nope_dim + tid];
    }
}

void fused_q_split_forward(
    torch::Tensor q_flat,    // [B, H * q_head_dim]
    torch::Tensor q_nope,    // [B, H, nope_dim]
    torch::Tensor q_pe       // [B, H, 1, rope_dim]
) {
    int B = q_flat.size(0);
    int total_dim = q_flat.size(1);
    int nope_dim = q_nope.size(2);
    int rope_dim = q_pe.size(3);
    int H = total_dim / (nope_dim + rope_dim);

    dim3 grid(H, B);
    int threads = (nope_dim > rope_dim) ? nope_dim : rope_dim;
    threads = ((threads + 31) / 32) * 32;

    fused_q_split_kernel<<<grid, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_flat.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
        B, H, nope_dim + rope_dim, nope_dim, rope_dim
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_q_split_forward", &fused_q_split_forward,
          "Fused q_b split into q_nope + q_pe");
}
