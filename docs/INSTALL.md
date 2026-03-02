# BatchGen Installation Guide

## Prerequisites
- **GPU**: Hopper (H100/H20) or newer, SM90+
- **CUDA**: 12.8+ toolkit installed
- **Python**: 3.11+
- **OS**: Ubuntu 22.04 (tested)

## Option A: Docker (Recommended for Production)

```bash
cd BatchGen
docker build -t batchgen -f docker/Dockerfile .
```

This builds everything in order: PyTorch → flash-attn 3 → FlashMLA → DeepGEMM → batchgen_kernels → batchgen. Takes ~40-60 min.

## Option B: install_deps.sh (Bare Metal / Conda)

```bash
# 1. Create conda env
conda create -n batchgen python=3.11 -y
conda activate batchgen

# 2. Clone repo
git clone --recursive https://github.com/EfficientMoE/BatchGen.git
cd BatchGen

# 3. Install everything
./scripts/install_deps.sh --all
```

This installs (in order):
1. **PyTorch 2.9.0+cu128** — from official wheels (~2 min)
2. **flash-attention 3** — built from source, Hopper only (~15-20 min)
3. **FlashMLA** — built from source (~5-10 min)
4. **DeepGEMM** — built from source (~5-10 min)
5. **batchgen_kernels** — AOT-compiled CUDA extensions, 14 kernels (~7 min)
6. **batchgen** — main package via `pip install .` (~1 min)

Total: ~40-50 min on first install.

## Option C: Pre-built Wheels (Skip Source Compilation)

If you have pre-built wheels for flash-attn/FlashMLA/DeepGEMM:

```bash
./scripts/install_deps.sh --all --wheel-dir /path/to/wheels
```

This skips the ~30 min source compilation of the three Hopper deps.

## Key Notes

| Item | Detail |
|------|--------|
| **Install mode** | `pip install .` (non-editable) — required for ray/production |
| **batchgen_kernels** | Must use `--no-build-isolation` (needs installed PyTorch headers) |
| **H20 GPUs** | Set `TORCH_CUDA_ARCH_LIST=9.0a` before building kernels |
| **Core engine** | JIT-compiled at first server launch via ninja (automatic, ~5s) |
| **No JIT for compute kernels** | All 14 CUDA extensions are AOT-compiled in `batchgen_kernels` |

## Verification

```bash
python -c "
import batchgen_kernels
for ext in [
    'batchgen_kernels.moe._C_expert_mxfp4_wgmma',
    'batchgen_kernels.moe._C_grouped_mxfp4_wgmma',
    'batchgen_kernels.moe._C_grouped_int4_wgmma',
    'batchgen_kernels.moe._C_single_expert_int4_wgmma',
    'batchgen_kernels.moe._C_fused_int4_wgmma_grouped',
    'batchgen_kernels.attention._C_qkv_wgmma',
    'batchgen_kernels.moe._C_routing',
    'batchgen_kernels.moe._C_dispatch_scatter_3d',
    'batchgen_kernels.attention._C_fused_ops',
    'batchgen_kernels.moe._C_mxfp4_dequant_cute',
    'batchgen_kernels.moe._C_mxfp4_dequant',
    'batchgen_kernels.common._C_rmsnorm',
    'batchgen_kernels.common._C_cuda_rmsnorm',
    'batchgen_kernels.common._C_mgn_ops',
]:
    batchgen_kernels.load_extension(ext)
    print(f'  {ext}: OK')
import flash_attn_3, flash_mla, deep_gemm
print('All dependencies verified.')
"
```

## Install Dependency Graph

```
PyTorch 2.9.0+cu128
├── flash-attention 3  (--no-build-isolation)
├── FlashMLA           (--no-build-isolation)
├── DeepGEMM           (--no-build-isolation)
├── batchgen_kernels   (--no-build-isolation, 14 CUDAExtensions)
│   ├── SM90a: WGMMA kernels (MoE, QKV, routing)
│   └── SM80+: fused ops (RMSNorm, RoPE, dequant, MGN)
└── batchgen           (pip install .)
    └── core_engine    (JIT at first launch, automatic)
```

## Troubleshooting

### flash-attention build hangs or fails with torch 2.9.0
flash-attention's `setup.py` tries to download a pre-built wheel matching your
torch version before building from source. No pre-built wheel exists for torch 2.9,
so the download hangs or times out. The install scripts already handle this by setting
`FLASH_ATTENTION_FORCE_BUILD=TRUE`, which skips the wheel download and builds from
source directly. If building manually:

```bash
cd flash-attention/hopper
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install . --no-build-isolation
```

### nvcc: "A single input file is required"
Ensure `TORCH_CUDA_ARCH_LIST` is set correctly (e.g., `9.0a` for H20).

### CUTLASS headers not found
The `batchgen_kernels` build requires the CUTLASS submodule. Ensure you cloned with
`--recursive`, or run `git submodule update --init --recursive`.
