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
pip install torch==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128
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
git clone git@github.com:Dao-AILab/flash-attention.git
cd flash-attention && git checkout v2.8.2
cd hopper && pip install . --no-build-isolation
```

See https://github.com/Dao-AILab/flash-attention for more details.

---

## Step 3: Install FlashMLA (Hopper Only)

FlashMLA provides optimized Multi-head Latent Attention for DeepSeek models on Hopper GPUs.

```bash
pip install git+https://github.com/deepseek-ai/FlashMLA.git --no-build-isolation
```

See https://github.com/deepseek-ai/FlashMLA for more details.

---

## Step 4: Install DeepGEMM (Hopper Only)

DeepGEMM provides optimized FP8 GEMM kernels for Hopper GPUs.

```bash
git clone --recursive git@github.com:deepseek-ai/DeepGEMM.git
cd DeepGEMM && pip install . --no-build-isolation
```

See https://github.com/deepseek-ai/DeepGEMM for more details.

---

## Step 5: Install BatchGen

```bash
git clone git@github.com:EfficientMoE/BatchGen.git
cd BatchGen
pip install -e .
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
