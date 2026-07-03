# Manual Installation Guide

This guide provides step-by-step instructions for manually installing BatchGen and its dependencies.

For quick installation, use the automated script instead:
```bash
./scripts/install_deps.sh
```

---

## Prerequisites

- Python 3.11+
- CUDA 12.8+ toolkit
- Git

Create and activate a virtual environment:
```bash
conda create --name batchgen python=3.11
conda activate batchgen
```

---

## Step 1: Install PyTorch with CUDA Support

```bash
pip install torch==2.9.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

---

## Step 2: Install Flash-Attention

### For Ampere GPUs (A100, A5000, RTX 4090, etc.)

```bash
pip install flash-attn --no-build-isolation
```

### For Hopper GPUs (H100, H20, etc.)

Install flash-attention 3 from the Hopper-optimized branch:

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention && git checkout v2.8.2
cd hopper && FLASH_ATTENTION_FORCE_BUILD=TRUE pip install . --no-build-isolation
```

> `FLASH_ATTENTION_FORCE_BUILD=TRUE` skips the prebuilt-wheel download attempt:
> no wheel exists for PyTorch 2.9, and depending on network conditions the
> download can hang or time out before falling back to a source build.

See https://github.com/Dao-AILab/flash-attention for more details.

---

## Step 3: Install FlashMLA (Hopper Only)

FlashMLA provides optimized Multi-head Latent Attention for DeepSeek models on Hopper GPUs.

```bash
FLASH_MLA_DISABLE_SM100=1 pip install "git+https://github.com/deepseek-ai/FlashMLA.git@1408756a88e52a25196b759eaf8db89d2b51b5a1" --no-build-isolation
```

> Pin the commit and set `FLASH_MLA_DISABLE_SM100=1` (same as
> `scripts/install_deps.sh`): FlashMLA HEAD enables SM100 (Blackwell) kernels
> that require nvcc >= 12.9, so the unpinned command fails on a CUDA 12.8
> toolchain. For Blackwell builds use nvcc 12.9+ and drop the env var.

See https://github.com/deepseek-ai/FlashMLA for more details.

---

## Step 4: Install DeepGEMM (Hopper Only)

DeepGEMM provides optimized FP8 GEMM kernels for Hopper GPUs.

```bash
git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git
cd DeepGEMM && git checkout v2.1.1.post3 && git submodule update --init --recursive
pip install . --no-build-isolation
```

> Check out `v2.1.1.post3` (the version `scripts/install_deps.sh` pins):
> DeepGEMM HEAD is not guaranteed to build against this stack.

See https://github.com/deepseek-ai/DeepGEMM for more details.

---

## Step 5: Reinstall PyTorch (Important)

Building flash-attention, FlashMLA, or DeepGEMM from source may downgrade PyTorch or install conflicting versions of triton. Reinstall to ensure the correct versions:

```bash
pip install torch==2.9.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

---

## Step 6: Clone BatchGen

```bash
git clone https://github.com/batchgen-project/batchgen.git
cd batchgen
```

---

## Step 7: Build batchgen_kernels (AOT CUDA extensions)

`batchgen_kernels` is a separate package containing the pre-compiled CUDA
kernels. It must be built before installing BatchGen, otherwise the compiled
extensions are missing at runtime. On H20, export `TORCH_CUDA_ARCH_LIST=9.0a`
first; for Blackwell (B200) export `BUILD_ARCH=sm100`.

```bash
cd batchgen_kernels
pip install . --no-build-isolation
cd ..
```

---

## Step 8: Install BatchGen

Install non-editable. An editable install (`pip install -e .`) leaves the source
tree on `sys.path`, which shadows the installed package and breaks ray/production
launches — see [INSTALL.md](INSTALL.md#important-do-not-run-from-the-source-directory).

```bash
pip install . --no-build-isolation
```

---

## Verify Installation

```python
import torch
import batchgen

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
```

To also confirm the compiled `batchgen_kernels` extensions and the
flash-attention / FlashMLA / DeepGEMM dependencies, run the fuller verification
snippet in [INSTALL.md](INSTALL.md#verification).

---

## Troubleshooting

### CUDA Version Mismatch

If you see CUDA version errors, ensure your CUDA toolkit matches the PyTorch CUDA version:
```bash
nvcc --version  # Should show CUDA 12.8+
```

### Build Failures

For build failures with flash-attention or DeepGEMM:
1. Ensure `ninja` is installed: `pip install ninja`
2. Check that CUDA headers are accessible
3. Try building with verbose output: `pip install . -v --no-build-isolation`

### Import Errors

If imports fail after installation:
1. Restart your Python interpreter
2. Check that you're in the correct virtual environment
3. Verify the package is installed: `pip list | grep batchgen`

---

## See Also

- [README](../README.md) - Quick installation and overview
- [Deployment Guide](deploy-deepseek-r1-h20.md) - Multi-node deployment
- [Server Flags Reference](server-flags.md) - All server configuration options
