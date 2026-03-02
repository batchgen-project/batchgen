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

## Option C: Pre-built Wheels (Fastest, ~2 min)

Pre-built wheels for Hopper GPUs (CUDA 12.8, PyTorch 2.9, Python 3.11) are
available on the [GitHub Releases](https://github.com/EfficientMoE/BatchGen/releases) page.

```bash
# 1. Create conda env + install PyTorch
conda create -n batchgen python=3.11 -y
conda activate batchgen
pip install torch==2.9.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# 2. Download and install all wheels from the latest release
RELEASE_URL="https://github.com/EfficientMoE/BatchGen/releases/download/v0.1.0"
pip install \
  "${RELEASE_URL}/flash_attn_3-3.0.0b1-cp311-cp311-linux_x86_64.whl" \
  "${RELEASE_URL}/flash_mla-1.0.0-cp311-cp311-linux_x86_64.whl" \
  "${RELEASE_URL}/deep_gemm-2.1.1-cp311-cp311-linux_x86_64.whl" \
  "${RELEASE_URL}/batchgen_kernels-0.1.0-cp311-cp311-linux_x86_64.whl" \
  "${RELEASE_URL}/batchgen-0.1-cp311-cp311-linux_x86_64.whl"
```

Or with `install_deps.sh` using a local wheel directory:

```bash
# Download wheels to a local directory first
mkdir -p wheels && cd wheels
RELEASE_URL="https://github.com/EfficientMoE/BatchGen/releases/download/v0.1.0"
for whl in flash_attn_3 flash_mla deep_gemm batchgen_kernels batchgen; do
    wget "${RELEASE_URL}/${whl}"-*.whl
done
cd ..

# Install using the script
git clone https://github.com/EfficientMoE/BatchGen.git && cd BatchGen
./scripts/install_deps.sh --all --wheel-dir /path/to/wheels
```

No source compilation needed — total install time ~2 min.

### Building wheels yourself

If you need to build wheels for a different environment:

```bash
bash scripts/build_wheels.sh --output-dir /path/to/wheels
```

Then upload to a GitHub Release:

```bash
gh release create v0.1.0 --title "BatchGen v0.1.0"
gh release upload v0.1.0 /path/to/wheels/*.whl
```

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

### Git clone fails (network issues)
If GitHub is unreliable, use Option C (pre-built wheels) which only requires
downloading `.whl` files. Alternatively, clone repos on a machine with better
connectivity and copy them over.
