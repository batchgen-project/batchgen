# CUTLASS/CuTe MoE notes for sm120

## What CUTLASS 4.0 does provide

- `cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl` builds a grouped/ptr-array dispatch policy for NVFP4/MXFP4 blockscaled GEMM on `arch::Sm120`.
- The builder selects `MainloopSm120ArrayTmaWarpSpecializedBlockScaled<...>` when the A/B stride types imply grouped/ptr-array GEMM.
- `cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp` and `sm120_mma_array_tma_blockwise_scaling.hpp` define the array mainloops and their argument plumbing.

## Why the CUTLASS array path is not usable here today

### 1. The array mainloop is hard-wired to TMA/tensormap machinery

Evidence from `sm120_blockscaled_mma_array_tma.hpp`:

- `static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD>, ...)`
- `static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD>, ...)`
- `cute::TmaDescriptor smem_tensormap_A/B/SFA/SFB`
- `make_tma_copy(...)` for A/B/SFA/SFB
- runtime descriptor mutation via `tma_descriptor_replace_*` and `tma_desc_commit_group()`

So the provided grouped sm120 blockscaled path is not a manual-global-load mma.sync path; it is a TMA-driven path.

### 2. CUTLASS exposes sm120 ptr-array dispatch tags, but the matching kernel specialization is missing

Evidence:

- `cutlass/gemm/dispatch_policy.hpp` defines
  - `KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaledSm120`
  - `KernelPtrArrayTmaWarpSpecializedPingpongBlockScaledSm120`
  - `KernelPtrArrayTmaWarpSpecializedCooperativeBlockwiseScalingSm120`
  - `KernelPtrArrayTmaWarpSpecializedPingpongBlockwiseScalingSm120`
- `sm120_blockscaled_mma_builder.inl` selects those tags for grouped GEMM.

But under `cutlass/gemm/kernel/` there is no `sm120_gemm_array_tma_warpspecialized*.hpp` equivalent to the SM100 implementation in `sm100_gemm_array_tma_warpspecialized.hpp`.

The only sm120 kernel specialization in this tree is:

- `cutlass/gemm/kernel/sm120_gemm_tma_warpspecialized_cooperative_asymmetric_dma.hpp`

and that specialization is for sparse/asymmetric DMA schedules, not the dense/blockscaled ptr-array schedules needed by the MoE grouped GEMM.

### 3. Project constraint mismatch

This task targets RTX PRO 6000 Blackwell (`cc 12.0`) with the constraint set:

- use `mma.sync`
- no WGMMA
- no TMEM
- no TMA descriptors in the intended implementation path

The available CUTLASS grouped sm120 blockscaled path conflicts with that requirement because it assumes TMA/tensormap-based mainloops.

## Resulting implementation choice

For this iteration, the heavy fused MoE compute is moved out of Triton into a native CUDA extension:

- new file: `batchgen_kernels/src/moe/mega_moe_sm120.cu`
- wrapper: `batchgen_kernels/moe/mega_moe_sm120.py`
- default sm120 call path wired through `batchgen/moe/v4_mega3_moe_sm120.py`

The route-pack metadata construction remains in the existing Triton helper because the main blocker is Triton's fused GEMM/activation/scatter IR bloat, not the lightweight routing pack kernel.

## Current state of the native path

- Native kernel fuses:
  - gather by `slot_token_ids`
  - stage-1 MXFP4 gate/up matmuls
  - SwiGLU
  - stage-2 MXFP4 down matmul
  - scatter via `atomicAdd`
- It is intentionally conservative and correctness-oriented.
- It is **not yet** a tensor-core-optimized `mma.sync`/CuTe kernel.

## Validation performed

### Build

Built successfully with:

```bash
CUDA_HOME=/usr/local/cuda-13.1 BUILD_ARCH=sm120 MAX_JOBS=4 python setup.py build_ext --inplace
```

The new extension `batchgen_kernels.moe._C_mega_moe_sm120` compiled and loaded successfully on sm120.

### Numerical comparison

Feasible validation was done by comparing the native sm120 path against the existing ragged Triton path on synthetic DeepSeek-V4-shaped MXFP4 weights/routing for batch sizes `{1, 8, 64, 256}`.

Observed absolute error:

| B | max_abs | mean_abs |
|---|---------|----------|
| 1 | 0.001541 | 0.000357 |
| 8 | 0.003984 | 0.000517 |
| 64 | 0.003822 | 0.000471 |
| 256 | 0.004905 | 0.000477 |

This is evidence that the native path is numerically close to the ragged reference on the exercised synthetic cases.

### Kernel-only timing

At `B=64`, timing only the native fused kernel body (routing metadata already prepared) gave:

| Case | kernel-only time |
|------|------------------|
| native sm120 fused kernel | `11803.02 us` |

So the current implementation **does not meet** the `< 2000 us` target.

## Next step if performance is insufficient

Implement a custom CuTe or PTX `mma.sync` kernel using the same route-pack metadata shape used by `v4_mega_moe_sm120.py`, rather than trying to force the unavailable CUTLASS grouped sm120 array path.
