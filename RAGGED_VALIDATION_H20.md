# RAGGED_VALIDATION_H20

- GPU: NVIDIA H20
- Arch: sm_90
- Config: V4-Flash decode
- Shapes: B={1,8,64,256}, H=4096, I=2048, E=256, topk=6
- DeepGEMM layout: masked grouped FP8 NT on sm_90 with FP32 scales

| B | Ragged us | Ragged pass | DeepGEMM us | DeepGEMM pass | Ragged/DeepGEMM |
|---:|---:|:---:|---:|:---:|---:|
| 1 | 745.2 | ✅ | 1270.4 | ❌ | 0.587x |
| 8 | 1382.2 | ✅ | 6005.2 | ❌ | 0.230x |
| 64 | 9849.1 | ✅ | 238211.3 | ❌ | 0.041x |
| 256 | 9648.0 | ✅ | 278290.7 | ❌ | 0.035x |

## Notes

- B=1 batchgen-ragged: max_abs_diff=0.007032, rmse=0.001894, cosine=0.976884, recall=0.7188, kernel=_ragged_mxfp4_matmul_kernel
- B=1 sglang-deepgemm: max_abs_diff=63263360.000000, rmse=11596843.000000, cosine=0.023880, recall=0.0312, kernel=fp8_m_grouped_gemm_nt_masked
- B=8 batchgen-ragged: max_abs_diff=0.008661, rmse=0.001868, cosine=0.977529, recall=0.7266, kernel=_ragged_mxfp4_matmul_kernel
- B=8 sglang-deepgemm: max_abs_diff=317500512.000000, rmse=27278054.000000, cosine=0.007551, recall=0.0039, kernel=fp8_m_grouped_gemm_nt_masked
- B=64 batchgen-ragged: max_abs_diff=0.010575, rmse=0.001847, cosine=0.977253, recall=0.7280, kernel=_ragged_mxfp4_matmul_kernel
- B=64 sglang-deepgemm: max_abs_diff=2831.488037, rmse=169.391342, cosine=0.001384, recall=0.0122, kernel=fp8_m_grouped_gemm_nt_masked
- B=256 batchgen-ragged: max_abs_diff=0.021039, rmse=0.001917, cosine=0.977340, recall=0.7480, kernel=_ragged_mxfp4_matmul_kernel
- B=256 sglang-deepgemm: max_abs_diff=0.070728, rmse=0.007688, cosine=0.469481, recall=0.0942, kernel=fp8_m_grouped_gemm_nt_masked

## Interpretation

- Ragged validation is treated as pass/fail via absolute error + RMSE. Relative error is not useful here because many reference elements are near zero.
- On that criterion, the Hopper run validates the ragged kernel across B={1,8,64,256}.
- The DeepGEMM masked grouped baseline was collected as requested, but its numerics did not validate against the BF16 reference in this run, so treat those timings as exploratory baseline points only.
- A follow-up rerun was blocked by H20 memory pressure from other processes on TencentNode0, so I kept the successful first-pass timing data and documented the caveat instead of burning more time.
