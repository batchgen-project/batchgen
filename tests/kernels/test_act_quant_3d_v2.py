"""Sanity + perf unit test for the 2D-grid act_quant_3d kernel (v2).

v1 (production, batchgen_kernels/src/moe/fp8_blockwise/fp8_blockwise_ops.cu):
    grid = (E,), each CTA iterates `for m in 0..tokens_per_expert[e]`.
    At GLM-5 EP=16 this launches only 16 CTAs on H20's 132 SMs → ~12%
    occupancy; observed ~500 µs/call.

v2 (proposed):
    grid = (E, mtp), one CTA per (expert, token). Early-exits for
    padded slots (`token_idx >= tokens_per_expert[expert]`). Same
    numerics — per-k-block FP32 absmax → FP8 cast sequence unchanged.

Run on H20:
    bash scripts/remote/kernel_test.sh tests/kernels/test_act_quant_3d_v2.py
"""
from __future__ import annotations

import argparse
import time

import torch
from torch.utils.cpp_extension import load_inline


_V1_V2_SRC = r"""
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <cstdint>
#include <tuple>

// Match prod: batchgen_kernels/src/moe/fp8_blockwise/fp8_blockwise_ops.cu
#define FP8_MAX_VAL 448.0f
#define QUANT_EPS   1e-12f
#define BLOCK_SIZE_QUANT 128

// ------------------------------------------------------------------
// v1 — grid=(E,), serial over tokens (copy of prod kernel)
// ------------------------------------------------------------------
__global__ void act_quant_3d_kernel_v1(
    const __nv_bfloat16* __restrict__ x,
    uint8_t* __restrict__ y,
    float* __restrict__ scale,
    const int32_t* __restrict__ tokens_per_expert,
    int mtp, int K, int num_k_blocks
) {
    const int expert = blockIdx.x;
    const int valid_tokens = tokens_per_expert[expert];
    if (valid_tokens == 0) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;

    const __nv_bfloat16* x_expert = x + (int64_t)expert * mtp * K;
    uint8_t* y_expert = y + (int64_t)expert * mtp * K;
    float* scale_expert = scale + (int64_t)expert * mtp * num_k_blocks;

    for (int m = 0; m < valid_tokens; m++) {
        const __nv_bfloat16* x_row = x_expert + (int64_t)m * K;
        uint8_t* y_row = y_expert + (int64_t)m * K;
        float* scale_row = scale_expert + (int64_t)m * num_k_blocks;

        for (int kb = warp_id; kb < num_k_blocks; kb += num_warps) {
            int col_base = kb * BLOCK_SIZE_QUANT;
            float vals[4]; float local_max = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int col = col_base + lane_id * 4 + i;
                vals[i] = (col < K) ? __bfloat162float(x_row[col]) : 0.0f;
                local_max = fmaxf(local_max, fabsf(vals[i]));
            }
            #pragma unroll
            for (int off = 16; off >= 1; off >>= 1) {
                local_max = fmaxf(local_max,
                                  __shfl_xor_sync(0xffffffff, local_max, off));
            }
            constexpr float FP8_MAX_VAL_INV = 1.0f / FP8_MAX_VAL;
            float s = fmaxf(local_max, QUANT_EPS) * FP8_MAX_VAL_INV;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int col = col_base + lane_id * 4 + i;
                if (col < K) {
                    float sc = vals[i] / s;
                    sc = fmaxf(fminf(sc, FP8_MAX_VAL), -FP8_MAX_VAL);
                    y_row[col] = __nv_cvt_float_to_fp8(sc, __NV_SATFINITE, __NV_E4M3);
                }
            }
            if (lane_id == 0) scale_row[kb] = s;
        }
    }
}

// ------------------------------------------------------------------
// v2 — grid=(E, mtp), one CTA per (expert, token)
// ------------------------------------------------------------------
__global__ void act_quant_3d_kernel_v2(
    const __nv_bfloat16* __restrict__ x,
    uint8_t* __restrict__ y,
    float* __restrict__ scale,
    const int32_t* __restrict__ tokens_per_expert,
    int mtp, int K, int num_k_blocks
) {
    const int expert = blockIdx.x;
    const int token  = blockIdx.y;
    const int valid_tokens = tokens_per_expert[expert];
    if (token >= valid_tokens) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;

    const __nv_bfloat16* x_row =
        x + ((int64_t)expert * mtp + token) * K;
    uint8_t* y_row =
        y + ((int64_t)expert * mtp + token) * K;
    float* scale_row =
        scale + ((int64_t)expert * mtp + token) * num_k_blocks;

    for (int kb = warp_id; kb < num_k_blocks; kb += num_warps) {
        int col_base = kb * BLOCK_SIZE_QUANT;
        float vals[4]; float local_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int col = col_base + lane_id * 4 + i;
            vals[i] = (col < K) ? __bfloat162float(x_row[col]) : 0.0f;
            local_max = fmaxf(local_max, fabsf(vals[i]));
        }
        #pragma unroll
        for (int off = 16; off >= 1; off >>= 1) {
            local_max = fmaxf(local_max,
                              __shfl_xor_sync(0xffffffff, local_max, off));
        }
        constexpr float FP8_MAX_VAL_INV = 1.0f / FP8_MAX_VAL;
        float s = fmaxf(local_max, QUANT_EPS) * FP8_MAX_VAL_INV;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int col = col_base + lane_id * 4 + i;
            if (col < K) {
                float sc = vals[i] / s;
                sc = fmaxf(fminf(sc, FP8_MAX_VAL), -FP8_MAX_VAL);
                y_row[col] = __nv_cvt_float_to_fp8(sc, __NV_SATFINITE, __NV_E4M3);
            }
        }
        if (lane_id == 0) scale_row[kb] = s;
    }
}

// ------------------------------------------------------------------
// Python wrappers
// ------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> run_v1(
    torch::Tensor x, torch::Tensor tokens_per_expert
) {
    int E = x.size(0), mtp = x.size(1), K = x.size(2);
    int num_k_blocks = (K + BLOCK_SIZE_QUANT - 1) / BLOCK_SIZE_QUANT;
    auto y = torch::empty({E, mtp, K},
                          torch::dtype(torch::kUInt8).device(x.device()));
    auto scale = torch::empty({E, mtp, num_k_blocks},
                              torch::dtype(torch::kFloat32).device(x.device()));
    cudaStream_t s = at::cuda::getCurrentCUDAStream();
    act_quant_3d_kernel_v1<<<E, 128, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        y.data_ptr<uint8_t>(), scale.data_ptr<float>(),
        tokens_per_expert.data_ptr<int32_t>(), mtp, K, num_k_blocks);
    return std::make_tuple(y, scale);
}

std::tuple<torch::Tensor, torch::Tensor> run_v2(
    torch::Tensor x, torch::Tensor tokens_per_expert
) {
    int E = x.size(0), mtp = x.size(1), K = x.size(2);
    int num_k_blocks = (K + BLOCK_SIZE_QUANT - 1) / BLOCK_SIZE_QUANT;
    auto y = torch::empty({E, mtp, K},
                          torch::dtype(torch::kUInt8).device(x.device()));
    auto scale = torch::empty({E, mtp, num_k_blocks},
                              torch::dtype(torch::kFloat32).device(x.device()));
    cudaStream_t s = at::cuda::getCurrentCUDAStream();
    dim3 grid(E, mtp);
    act_quant_3d_kernel_v2<<<grid, 128, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        y.data_ptr<uint8_t>(), scale.data_ptr<float>(),
        tokens_per_expert.data_ptr<int32_t>(), mtp, K, num_k_blocks);
    return std::make_tuple(y, scale);
}
"""


def _build():
    return load_inline(
        name="act_quant_3d_v1v2_test",
        cpp_sources=[
            "#include <tuple>\n#include <torch/extension.h>\n"
            "std::tuple<torch::Tensor, torch::Tensor> run_v1(torch::Tensor, torch::Tensor);\n"
            "std::tuple<torch::Tensor, torch::Tensor> run_v2(torch::Tensor, torch::Tensor);\n",
        ],
        cuda_sources=[_V1_V2_SRC],
        functions=["run_v1", "run_v2"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


def _make_input(E, mtp, K, valid_pattern="random", seed=0):
    torch.manual_seed(seed)
    x = (torch.randn(E, mtp, K, device="cuda", dtype=torch.bfloat16) * 5.0)
    if valid_pattern == "random":
        tokens_per_expert = torch.randint(0, mtp + 1, (E,),
                                          device="cuda", dtype=torch.int32)
    elif valid_pattern == "full":
        tokens_per_expert = torch.full((E,), mtp,
                                       device="cuda", dtype=torch.int32)
    elif valid_pattern == "empty":
        tokens_per_expert = torch.zeros((E,),
                                        device="cuda", dtype=torch.int32)
    return x, tokens_per_expert


def correctness(ext, E, mtp, K, pattern):
    x, tpe = _make_input(E, mtp, K, valid_pattern=pattern)
    y1, s1 = ext.run_v1(x, tpe)
    y2, s2 = ext.run_v2(x, tpe)
    torch.cuda.synchronize()

    # Only compare valid slots per expert; padded slots are undefined for both.
    tpe_cpu = tpe.cpu().tolist()
    total_valid = sum(tpe_cpu)
    mismatches = 0
    for e, n in enumerate(tpe_cpu):
        if n == 0:
            continue
        eq_y = torch.equal(y1[e, :n], y2[e, :n])
        eq_s = torch.equal(s1[e, :n], s2[e, :n])
        if not (eq_y and eq_s):
            mismatches += 1
            if mismatches <= 3:
                diff_y = (y1[e, :n].int() - y2[e, :n].int()).abs().sum().item()
                diff_s = (s1[e, :n] - s2[e, :n]).abs().sum().item()
                print(f"  expert {e} (n={n}): y byte-diff={diff_y} scale-diff={diff_s:.3e}")
    if mismatches:
        raise AssertionError(f"{mismatches}/{E} experts mismatched")
    print(f"[CORRECT] E={E} mtp={mtp} K={K} pattern={pattern}: "
          f"{total_valid} valid slots — bit-exact v1==v2 ✓")


def perf(ext, E, mtp, K, warmup=20, iters=100):
    x, tpe = _make_input(E, mtp, K, valid_pattern="full")

    for _ in range(warmup):
        ext.run_v1(x, tpe)
        ext.run_v2(x, tpe)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        ext.run_v1(x, tpe)
    torch.cuda.synchronize()
    v1_us = (time.perf_counter() - t0) * 1e6 / iters

    t0 = time.perf_counter()
    for _ in range(iters):
        ext.run_v2(x, tpe)
    torch.cuda.synchronize()
    v2_us = (time.perf_counter() - t0) * 1e6 / iters

    speedup = v1_us / v2_us if v2_us > 0 else float("inf")
    print(f"[PERF]   E={E} mtp={mtp} K={K}: "
          f"v1={v1_us:8.2f}µs  v2={v2_us:8.2f}µs  speedup={speedup:5.2f}×")
    return speedup


def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA unavailable")
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--e", type=int, default=16)
    parser.add_argument("--mtp", type=int, default=128)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    print("Building inline CUDA extension...")
    ext = _build()
    print("Built.\n")

    print("== Correctness ==")
    correctness(ext, args.e, args.mtp, K=6144, pattern="full")
    correctness(ext, args.e, args.mtp, K=6144, pattern="random")
    correctness(ext, args.e, args.mtp, K=8704, pattern="full")
    correctness(ext, args.e, args.mtp, K=8704, pattern="random")
    correctness(ext, args.e, args.mtp, K=6144, pattern="empty")
    print()

    print("== Perf ==")
    s1 = perf(ext, args.e, args.mtp, K=6144, iters=args.iters)
    s2 = perf(ext, args.e, args.mtp, K=8704, iters=args.iters)
    if min(s1, s2) < 5.0:
        raise AssertionError(
            f"v2 speedup below 5× threshold: K=6144 {s1:.2f}×, K=8704 {s2:.2f}×")
    print(f"\n[PASS] v2 ≥ 5× v1 on both shapes")


if __name__ == "__main__":
    main()
