# MEGA3 kernel notes

## Goal

Replace the 2-launch DeepSeek-V4 MXFP4 mega-kernel fallback with a 3-launch path
that stays simple enough for Triton to compile efficiently on sm120.

## Architecture

1. **route_pack()**
   - Pure PyTorch GPU routing metadata construction.
   - Reuses the compact on-device counting/sort path from `v4_ragged_moe_sm120.py`.
   - No large compile-time unrolled loops.

2. **stage1_swiglu_kernel**
   - One `tl.dot_scaled` only.
   - Inline gather from `hidden_states[token_id]`.
   - Produces fused `[gate, up]`, applies SwiGLU in-kernel, multiplies routing
     weight, materializes `activated[S, intermediate]`.

3. **stage2_scatter_kernel**
   - One `tl.dot_scaled` only.
   - Loads `activated[S, intermediate]`.
   - Down-projects and atomically accumulates into `output[token_id]`.

## Why this should compile better

- No multi-`tl.dot_scaled` register pressure in a single kernel.
- No giant Triton IR from route-pack compile-time loops over batch/top-k maxima.
- Materialized intermediate is modest (~6 MiB at 1536x2048 bf16).

## Files

- `batchgen/moe/v4_mega3_moe_sm120.py`: new 3-launch implementation.
- `batchgen/moe/v4_slot_moe_sm120.py`: mega3 is the default path; ragged remains
  the explicit fallback via `BATCHGEN_V4_RAGGED_FALLBACK=1`.
- `batchgen/moe/bench_v4_mega3_moe.py`: permanent synthetic benchmark harness.
- `tests/integration/test_v4_linear_numerics_parity.py`: mega3-vs-ragged
  correctness gate for token counts `{1, 8, 64, 256}`.

## Benchmark commands

```bash
python -m pytest tests/integration/test_v4_linear_numerics_parity.py -q -s
python -m batchgen.moe.bench_v4_mega3_moe --tokens 64 --iters 100
```

## Notes

- `bench_v4_mega3_moe.py` reports both CUDA-event kernel time and synchronized
  wall time.
- If `logs/sglang_v4_flash_decode_rows.jsonl` is present, the benchmark also
  prints the recorded SGLang reference row for the requested token count.
- Fresh SGLang Docker comparisons still need an available running baseline
  service/environment; the benchmark script itself does not launch Docker.
