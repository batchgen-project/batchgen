# BatchGen Installation Guide

## Prerequisites
- **GPU**: Hopper (H100/H20), Blackwell (B200), or newer — SM90+
- **CUDA**: 12.8+ toolkit installed (Blackwell/B200 requires **12.9+**, see below)
- **Python**: 3.11+
- **OS**: Ubuntu 22.04 (tested)
- **GitHub access**: the repository is currently **private** — anonymous
  `git clone` and raw release-asset URLs fail (404). Authenticate first
  (e.g. `gh auth login`), clone via `gh repo clone batchgen-project/batchgen`,
  and fetch release wheels with `gh release download` (see Option C).

All three options below have been validated end-to-end on a fresh 2-node (16×H20) cluster
with Kimi-K2.5. Pick whichever fits your workflow. For Blackwell (B200), see the
[Blackwell (B200) section](#blackwell-b200--sm_100) for the additional `BUILD_ARCH=sm100` knob.

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
git clone https://github.com/batchgen-project/batchgen.git
cd batchgen

# 3. Install everything
./scripts/install_deps.sh --all
```

This installs (in order):
1. **PyTorch 2.9.0+cu128** — from official wheels (~2 min)
2. **flash-attention 3** — built from source, Hopper only (~15-20 min)
3. **FlashMLA** — built from source (~5-10 min)
4. **DeepGEMM** — built from source (~5-10 min)
5. **batchgen_kernels** — AOT-compiled CUDA extensions, 22 kernels (~7 min)
6. **batchgen** — main package via `pip install .` (~1 min)

Total: ~40-50 min on first install.

## Option C: Pre-built Wheels (Fastest, ~2 min)

Pre-built wheels for Hopper GPUs (CUDA 12.8, PyTorch 2.9, Python 3.11) are
available on the [GitHub Releases](https://github.com/batchgen-project/batchgen/releases) page.
The wheel set below is pinned to `v1.0.10.post2` — the most recent release that
ships the full dependency wheel set (later releases only ship the batchgen
wheels). While the repository is private, plain `pip install <URL>` returns
404; download the wheels with an authenticated client first:

```bash
gh release download v1.0.10.post2 -R batchgen-project/batchgen -p '*.whl' -D ./wheels
pip install ./wheels/*.whl
```

Or, with direct URLs once the repository is public:

```bash
# 1. Create conda env + install PyTorch
conda create -n batchgen python=3.11 -y
conda activate batchgen
pip install torch==2.9.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# 2. Install all wheels from the pinned release
RELEASE_URL="https://github.com/batchgen-project/batchgen/releases/download/v1.0.10.post2"
pip install \
  "${RELEASE_URL}/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl" \
  "${RELEASE_URL}/flash_mla-1.0.0+1408756-cp311-cp311-linux_x86_64.whl" \
  "${RELEASE_URL}/deep_gemm-2.1.1+c9f8b34-cp311-cp311-linux_x86_64.whl" \
  "${RELEASE_URL}/batchgen_kernels-0.3.2+sm90a-cp311-cp311-linux_x86_64.whl" \
  "${RELEASE_URL}/batchgen-1.0.10.post2-py3-none-any.whl"
```

No source compilation needed — pip auto-installs all remaining Python dependencies
(transformers, fastapi, etc.) from PyPI when installing the batchgen wheel.

### Building wheels yourself

If you need to build wheels for a different environment:

```bash
bash scripts/build_wheels.sh --output-dir /path/to/wheels
```

Then upload to a GitHub Release:

```bash
gh release create v1.0.4.post1 --title "BatchGen v1.0.4.post1"
gh release upload v1.0.4.post1 /path/to/wheels/*.whl
```

## Blackwell (B200 / sm_100)

BatchGen runs on NVIDIA Blackwell (B200, compute capability `10.0`). The engine
auto-detects the architecture (`detect_gpu_arch()` returns `blackwell`) and the
MLA models (DeepSeek-R1 / GLM-5 / Kimi-K2.5) route attention through the
FlashAttention-3 + FlashMLA path, the same as Hopper.

### Requirements
- **CUDA toolkit / `nvcc` ≥ 12.9** — required to build FlashMLA's SM100 kernels.
  The Docker image (Option A) uses a CUDA 12.9 base for this reason.
- PyTorch stays `2.9.0+cu128`; only the build-time `nvcc` needs to be 12.9+.

### Install
Set `BUILD_ARCH=sm100` so the Blackwell build paths are selected. This both
enables FlashMLA's SM100 kernels and builds `batchgen_kernels` for `sm_100`:

```bash
# Bare metal / conda (Option B):
BUILD_ARCH=sm100 ./scripts/install_deps.sh --all

# Docker (Option A) — pass it as a build arg:
docker build --build-arg BUILD_ARCH=sm100 -t batchgen:b200 -f docker/Dockerfile .
```

With the default `BUILD_ARCH=sm90a`, the scripts/Dockerfile behave exactly as
the Hopper instructions above (FlashMLA SM100 disabled).

### Run
Use the B200 engine config, which sets `"gpu_arch": "blackwell"`:

```bash
cd /root   # not the source dir (see below)
python -m batchgen.launch_http_server \
  --engine-config configurations/DeepSeek-R1/engine_config_B200_8.json \
  ...
```

> **Note (kernel coverage):** The dedicated SM100 port of the `batchgen_kernels`
> WGMMA MoE/attention kernels is tracked as a separate effort. Until it lands,
> `BUILD_ARCH=sm100` builds the SM80-class fused ops retargeted to `sm_100`; the
> WGMMA-only kernels are not yet built for Blackwell, and code paths that probe
> for them (e.g. `is_qkv_wgmma_available()`) fall back automatically. The MLA
> attention path (FlashAttention-3 / FlashMLA) is the intended Blackwell route.

## Important: Do Not Run from the Source Directory

After installing with `pip install .` (non-editable), you **must not** launch the
server from the BatchGen source directory. Python will find the source `batchgen/`
and `batchgen_kernels/` directories before the installed site-packages, causing
import errors (e.g., missing compiled CUDA extensions).

```bash
# WRONG — source dir shadows installed packages
cd /path/to/BatchGen
python -m batchgen.launch_http_server ...

# CORRECT — run from any other directory
cd /root   # or /tmp, or ~, etc.
python -m batchgen.launch_http_server ...
```

This does not apply to Docker (Option A), where the source is the install target.

## Key Notes

| Item | Detail |
|------|--------|
| **Install mode** | `pip install .` (non-editable) — required for ray/production |
| **Do not run from source dir** | Source tree shadows installed packages (see above) |
| **batchgen_kernels** | Must use `--no-build-isolation` (needs installed PyTorch headers) |
| **H20 GPUs** | Set `TORCH_CUDA_ARCH_LIST=9.0a` before building kernels |
| **Core engine** | JIT-compiled at first server launch via ninja (automatic, ~5s) |
| **No JIT for compute kernels** | All 22 CUDA extensions are AOT-compiled in `batchgen_kernels` |

## Verification

This discovers and imports every compiled `_C_*` extension shipped in the
installed `batchgen_kernels` package, so it stays correct as kernels are added:

```bash
python -c "
import glob, os, importlib
import batchgen_kernels
pkg_dir = os.path.dirname(batchgen_kernels.__file__)
exts = sorted(
    f'batchgen_kernels.{os.path.basename(os.path.dirname(p))}.'
    + os.path.basename(p).split('.')[0]
    for p in glob.glob(os.path.join(pkg_dir, '*', '_C_*.so'))
)
assert exts, 'No compiled batchgen_kernels extensions found'
for ext in exts:
    importlib.import_module(ext)
    print(f'  {ext}: OK')
print(f'{len(exts)} batchgen_kernels extensions verified.')
import flash_attn_interface, flash_mla, deep_gemm
print('All dependencies verified.')
"
```

## Install Dependency Graph

```
PyTorch 2.9.0+cu128
├── flash-attention 3  (--no-build-isolation)
├── FlashMLA           (--no-build-isolation)
├── DeepGEMM           (--no-build-isolation)
├── batchgen_kernels   (--no-build-isolation, 22 CUDAExtensions)
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

### FlashMLA build fails with SM100 errors
FlashMLA's SM100 (Blackwell) codepath requires NVCC 12.9+. If you are building for
Hopper (the default, `BUILD_ARCH=sm90a`) the install scripts disable SM100 automatically:

```bash
FLASH_MLA_DISABLE_SM100=1 pip install . --no-build-isolation
```

For Blackwell, build with `BUILD_ARCH=sm100` (which leaves SM100 enabled) and a CUDA
12.9+ toolkit. See the [Blackwell (B200) section](#blackwell-b200--sm_100).

### nvcc: "A single input file is required"
Ensure `TORCH_CUDA_ARCH_LIST` is set correctly (e.g., `9.0a` for H20).

### CUTLASS headers not found
The `batchgen_kernels` build requires CUTLASS headers, which are **vendored** in the
repository at `batchgen_kernels/3rd/cutlass/` (not a git submodule). If they are
missing, you likely have a shallow or partial checkout — re-clone the full repository.

### Git clone fails (network issues)
If GitHub is unreliable, use Option C (pre-built wheels) which only requires
downloading `.whl` files. Alternatively, clone repos on a machine with better
connectivity and copy them over.
