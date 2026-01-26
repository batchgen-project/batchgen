# Triton Kernels Tests & Benchmarks

Tests and benchmarks for MXFP4 (Microscaling FP4) quantization kernels used in MoE (Mixture of Experts) layers.

## Benchmarks

### `bench_mxfp4_moe.py` (Unified - Recommended)

**Primary benchmark script** that compares all MXFP4 MoE GEMM approaches:

```bash
# Quick A/B: Fused vs Decoupled comparison
python bench_mxfp4_moe.py --quick --tokens 4

# Full comparison of all approaches
python bench_mxfp4_moe.py --compare-all --tokens 4

# FP4 decode version comparison (v1-v6)
python bench_mxfp4_moe.py --compare-fp4

# GEMM hyperparameter tuning (grid search)
python bench_mxfp4_moe.py --tune-gemm --tokens 4

# Numerical validation only
python bench_mxfp4_moe.py --validate

# Export results to CSV
python bench_mxfp4_moe.py --compare-all --output results.csv
```

**Approaches compared:**
| Approach | Description |
|----------|-------------|
| Fused MXFP4 Grouped GEMM | Inline dequantization during GEMM |
| Decoupled Dequant + BF16 GEMM | Separate dequant kernel + BF16 GEMM |
| FP4 decode v1-v6 | Dequant kernel variants (v6_scale_transpose is fastest) |
| Unfused baselines | Reference implementations |

### `bench_mxfp4_grouped_gemm.py` (Deprecated)

Legacy benchmark for fused GEMM hyperparameter tuning. Kept for reference.
Use `bench_mxfp4_moe.py --tune-gemm` instead.

### `bench_decoupled_mxfp4_moe.py` (Deprecated)

Legacy benchmark for decoupled approach and FP4 decode versions. Kept for reference.
Use `bench_mxfp4_moe.py --compare-fp4` instead.

## Tests

### `test_grouped_mxfp4_moe.py`

Correctness tests for grouped MXFP4 MoE operations:
- Token dispatch/undispatch
- MLP forward pass
- Grouped MoE forward (per-expert and true grouped 3D)
- Single expert optimized kernel
- Performance benchmarks

```bash
python test_grouped_mxfp4_moe.py
```

### `test_fused_mxfp4_gemm.py`

Tests for fused MXFP4 GEMM kernel correctness.

### `test_triton_kernels_wrapper.py`

Tests for Triton kernel wrapper functions.

### `debug_mxfp4_dequant.py`

Debug utilities for MXFP4 dequantization issues.

## FP4 Decode Versions

The dequantization kernel has multiple decode implementations with different performance:

| Version | Time (ms)* | Method | Notes |
|---------|-----------|--------|-------|
| v1_sequential | 39.3 | 16 tl.where() | Baseline (slowest) |
| v2_e2m1 | 28.1 | E2M1 arithmetic | 5-6 tl.where() |
| v3_binary_tree | 37.7 | Binary tree | 4 tl.where() |
| v4_branchless | 25.8 | IEEE bitcast | 2 tl.where() |
| v5_memopt | 21.0 | BLOCK_K=64 | 2x fewer K-blocks |
| v6_scale_transpose | 20.2 | K-major scales | Coalesced loads (BEST) |

*Benchmarked on H20 with 128 experts, 4 tokens/expert, GPT-OSS-120B dimensions

## GPT-OSS-120B Dimensions

Default benchmark configuration:
- **Experts**: 128
- **Hidden size**: 5120 (K dimension)
- **Intermediate size**: 13824 (N dimension)
- **Data movement**: ~23 GB (4.5 GB packed + 0.3 GB scales + 18 GB output)

## Requirements

- CUDA-capable GPU (H100/H20/A100 recommended)
- PyTorch with CUDA support
- Triton (`pip install triton`)
- BatchGen package installed
