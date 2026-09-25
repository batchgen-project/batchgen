#!/bin/bash
# ---------------------------------------------------------------------------- #
#  BatchGen Dependency Installation Script                                      #
#  Copyright (c) EfficientMoE team 2025                                         #
#                                                                               #
#  Supports Hopper (SM90) and Blackwell (SM100/B200) GPUs:                      #
#  Hopper:    flash-attention 3, FlashMLA, DeepGEMM                             #
#  Blackwell: flash-attention 4 (CuTeDSL/JIT), FlashMLA (SM100), DeepGEMM      #
# ---------------------------------------------------------------------------- #

set -e  # Exit on error

# Resolve the repo layout ONCE, before anything can `cd` away: BASH_SOURCE[0] is
# relative when invoked as ./scripts/install_deps.sh, and the dependency
# installers cd into $INSTALL_DIR, which used to break the late lookups.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCHGEN_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration - pinned versions for reproducibility
FLASH_ATTN_VERSION="v2.8.2"
FLASHMLA_COMMIT="1408756a88e52a25196b759eaf8db89d2b51b5a1"
# DeepGEMM comes from the sgl-project fork, which ships the supported
# build_sgl_deep_gemm.sh path and the fp8_mqa_logits(max_seqlen_k=...) runtime
# contract BatchGen calls. It publishes the `sgl-deep-gemm` distribution while
# still importing as `deep_gemm`. Its build needs apache-tvm-ffi present first.
DEEPGEMM_REPO_URL="https://github.com/sgl-project/DeepGEMM.git"
DEEPGEMM_SRC_DIR="DeepGEMM-sgl"
DEEPGEMM_VERSION="v0.1.5.post3"
DEEPGEMM_DIST="sgl-deep-gemm"
DEEPGEMM_DIST_VERSION="0.1.5.post3"
TVM_FFI_VERSION="0.1.11"
WHEEL_VERSION="0.45.1"
DEEPGEMM_PLATFORM_TAG="linux_x86_64"

# Build target architecture for batchgen_kernels and FlashMLA.
#   sm90a (default) -> Hopper; FlashMLA SM100 kernels are disabled.
#   sm100 / all     -> Blackwell (B200); FlashMLA SM100 kernels are enabled
#                      (requires nvcc >= 12.9).
# This is consumed unchanged by batchgen_kernels/setup.py via the environment.
BUILD_ARCH="${BUILD_ARCH:-sm90a}"

# FlashMLA: only disable its SM100 (Blackwell) kernels for the Hopper build.
FLASH_MLA_ENV=()
if [[ "$BUILD_ARCH" == "sm90a" ]]; then
    FLASH_MLA_ENV=(FLASH_MLA_DISABLE_SM100=1)
fi

# Installation directory (defaults to temp, can be overridden)
INSTALL_DIR="${BATCHGEN_INSTALL_DIR:-/tmp/batchgen_deps}"

# --- Pre-built wheel auto-download (Hopper fast path) -------------------------
# By default the installer first tries to fetch matching pre-built wheels
# (flash-attn 3, FlashMLA, DeepGEMM, batchgen_kernels) from the public GitHub
# release and install them (no compilation, ~2 min). If any required wheel is
# missing for this Python/GPU-arch it transparently falls back to building from
# source. `--from-source` forces the build; `--release-tag TAG` (or
# BATCHGEN_RELEASE_TAG) pins the release; `--wheel-dir DIR` uses local wheels.
BATCHGEN_REPO="${BATCHGEN_REPO:-batchgen-project/batchgen}"
RELEASE_TAG="${BATCHGEN_RELEASE_TAG:-}"   # empty => latest release
FROM_SOURCE=0

print_step() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# System (OS) packages required to BUILD the JIT-compiled `core_engine` extension.
# The core_engine sources (core/Parameter_Server/posix_shm.cpp) `#include <numa.h>`
# and link `-lnuma`, so the NUMA development headers must be present or the first
# server launch fails with:
#     fatal error: numa.h: No such file or directory
# The runtime-only `numactl`/`numactl-libs` packages do NOT ship numa.h — the
# *-devel (rpm) / *-dev (deb) package is required.
install_system_deps() {
    print_step "Installing system packages (NUMA dev headers for core_engine)..."

    if [[ -f /usr/include/numa.h ]]; then
        print_success "numa.h already present"
        return 0
    fi

    local SUDO=""
    [[ "$(id -u)" != "0" ]] && command -v sudo &>/dev/null && SUDO="sudo"

    if command -v dnf &>/dev/null; then
        $SUDO dnf install -y numactl-devel || print_warning "dnf install numactl-devel failed"
    elif command -v yum &>/dev/null; then
        $SUDO yum install -y numactl-devel || print_warning "yum install numactl-devel failed"
    elif command -v apt-get &>/dev/null; then
        $SUDO apt-get update -y && $SUDO apt-get install -y libnuma-dev || print_warning "apt-get install libnuma-dev failed"
    elif command -v zypper &>/dev/null; then
        $SUDO zypper install -y libnuma-devel || print_warning "zypper install libnuma-devel failed"
    else
        print_warning "No supported package manager found; install the NUMA dev headers manually"
        print_warning "  (numactl-devel on RHEL/TencentOS, libnuma-dev on Debian/Ubuntu)."
    fi

    if [[ -f /usr/include/numa.h ]]; then
        print_success "NUMA dev headers installed (numa.h present)"
    else
        print_warning "numa.h still missing — the core_engine JIT build will fail until installed."
    fi
}

check_prerequisites() {
    print_step "Checking prerequisites..."

    # System headers needed by the core_engine JIT build (numa.h -> numactl-devel).
    # UCX (for the core_engine distributed-weight daemon's UCP memory-handle API) is
    # provided by the `libucx-cu12` pip wheel declared in requirements.txt (installed
    # with the batchgen package), so no system UCX or source build is needed here.
    install_system_deps

    # Check Python
    if ! command -v python &> /dev/null; then
        print_error "Python not found. Please install Python 3.11+."
        exit 1
    fi

    PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    # Pure-python comparison (avoids a hard dependency on `bc`, which is absent on
    # many minimal images).
    if ! python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
        print_error "Python 3.11+ required. Found: $PYTHON_VERSION"
        exit 1
    fi
    print_success "Python $PYTHON_VERSION found"

    # Check git
    if ! command -v git &> /dev/null; then
        print_error "git not found. Please install git."
        exit 1
    fi
    print_success "git found"

    # Check CUDA. An explicitly-set CUDA_HOME wins: it lets a user point at a
    # toolkit provisioned to a non-standard prefix, or override a mismatched
    # system nvcc (e.g. a CUDA 13 fleet image) with a CUDA 12.8 toolkit. Else
    # look for nvcc on PATH, then scan common locations, preferring a 12.x
    # toolkit (the reference build is CUDA 12.8).
    if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
        export PATH="${CUDA_HOME}/bin:$PATH"
        print_step "Using nvcc from CUDA_HOME=${CUDA_HOME}"
    elif ! command -v nvcc &> /dev/null; then
        for _cudadir in /usr/local/cuda-12*/bin /usr/local/cuda/bin; do
            if [[ -x "$_cudadir/nvcc" ]]; then
                export PATH="$_cudadir:$PATH"
                print_step "Added $_cudadir to PATH (nvcc)"
                break
            fi
        done
    fi
    if ! command -v nvcc &> /dev/null; then
        print_warning "nvcc not found (no CUDA toolkit on PATH, under /usr/local/cuda*, or via CUDA_HOME)."
        print_warning "A source build needs the CUDA 12.8 toolkit; install it and re-run with"
        print_warning "  CUDA_HOME=/path/to/cuda-12.8 bash scripts/install_deps.sh   (wheel fast-path needs no toolkit)."
    else
        CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
        print_success "CUDA $CUDA_VERSION found (nvcc: $(command -v nvcc))"
        if [[ "${CUDA_VERSION%%.*}" != "12" ]]; then
            print_warning "Reference build targets CUDA 12.8 (torch 2.9.0+cu128); found CUDA $CUDA_VERSION."
            print_warning "A different major CUDA will likely fail the FA3/FlashMLA/DeepGEMM builds or mismatch the pinned torch."
            print_warning "Point CUDA_HOME at a 12.8 toolkit to override: CUDA_HOME=/path/to/cuda-12.8 bash scripts/install_deps.sh"
        fi
    fi

    # Check ninja (for fast builds)
    if ! command -v ninja &> /dev/null; then
        print_step "Installing ninja for faster builds..."
        pip install ninja
    fi
    print_success "ninja found"
}

check_gpu_arch() {
    print_step "Detecting GPU architecture..."

    GPU_GENERATION=$(python -c "
import torch
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    if major >= 10:
        print('blackwell')
    elif major >= 9:
        print('hopper')
    elif major >= 8:
        print('ampere')
    else:
        print('other')
else:
    print('no_gpu')
" 2>/dev/null || echo "unknown")

    case "$GPU_GENERATION" in
        blackwell)
            print_success "Blackwell GPU detected (SM100+) — will use FA4 + FlashMLA SM100 + DeepGEMM"
            ;;
        hopper)
            print_success "Hopper GPU detected (SM90)"
            ;;
        ampere)
            print_warning "Ampere GPU detected. Hopper/Blackwell-specific optimizations will be skipped."
            ;;
        no_gpu)
            print_warning "No GPU detected. Installing dependencies anyway (for CPU-only builds)."
            GPU_GENERATION="hopper"  # default to Hopper path for CI/CPU boxes
            ;;
        *)
            print_warning "Could not detect GPU architecture. Assuming Hopper."
            GPU_GENERATION="hopper"
            ;;
    esac
}

torch_contract_ok() {
    python - "$1" "$2" <<'PY'
import sys
import torch

expected, channel = sys.argv[1:]
digits = channel[2:]
expected_cuda = f"{digits[:-1]}.{digits[-1]}" if digits.isdigit() else ""
if torch.__version__ != expected or torch.version.cuda != expected_cuda:
    raise SystemExit(1)
PY
}

install_torch() {
    print_step "Checking PyTorch installation..."

    # Resolve the CUDA channel: honour TORCH_CUDA_CHANNEL env var (set by
    # provision_machine.sh based on GPU arch), default to cu128 (Hopper).
    local cuda_channel="${TORCH_CUDA_CHANNEL:-cu128}"
    local torch_ver="2.9.0+${cuda_channel}"

    if python -c "import torch; print(torch.__version__)" &> /dev/null; then
        TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
        if torch_contract_ok "$torch_ver" "$cuda_channel"
        then
            print_success "PyTorch $TORCH_VERSION / CUDA ${cuda_channel} ABI verified"
        else
            print_warning "PyTorch $TORCH_VERSION does not match required ${torch_ver}; reinstalling"
            pip install "torch==${torch_ver}" --index-url "https://download.pytorch.org/whl/${cuda_channel}"
            TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
            if ! torch_contract_ok "$torch_ver" "$cuda_channel"
            then
                print_error "PyTorch install did not produce the required ${torch_ver}/${cuda_channel} ABI"
                return 1
            fi
            print_success "PyTorch $TORCH_VERSION / CUDA ${cuda_channel} ABI verified"
        fi

        # Check CUDA availability
        CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
        if [[ "$CUDA_AVAILABLE" == "False" ]]; then
            print_warning "PyTorch installed but CUDA not available. Consider reinstalling with CUDA support."
        fi
    else
        print_step "Installing PyTorch ${torch_ver} (cuda_channel=${cuda_channel})..."
        pip install "torch==${torch_ver}" --index-url "https://download.pytorch.org/whl/${cuda_channel}"
        print_success "PyTorch installed"
    fi
}

install_flash_attention() {
    print_step "Installing flash-attention 3 (Hopper)..."

    # Check if already installed (hopper build exposes the `flash_attn_interface` module)
    if python -c "import flash_attn_interface" &> /dev/null 2>&1; then
        print_success "flash-attention 3 already installed"
        return 0
    fi

    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"

    if [[ -d "flash-attention" ]]; then
        print_step "Updating existing flash-attention repository..."
        cd flash-attention
        git fetch origin
        git checkout "$FLASH_ATTN_VERSION"
    else
        print_step "Cloning flash-attention repository..."
        # Hopper setup initializes its CUTLASS submodule itself. Clone the
        # pinned tag directly so we do not download unrelated ROCm submodules
        # from the repository's default branch first.
        git clone --branch "$FLASH_ATTN_VERSION" --single-branch --depth 1 \
            https://github.com/Dao-AILab/flash-attention.git
        cd flash-attention
    fi

    print_step "Building flash-attention 3 (this may take 10-20 minutes)..."
    cd hopper
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    FLASH_ATTENTION_DISABLE_SM80=TRUE \
        pip install . --no-build-isolation

    print_success "flash-attention 3 installed"
}

install_flash_attention_4() {
    print_step "Installing flash-attention 4 (Blackwell/CuTeDSL JIT)..."

    if python -c "from flash_attn.cute import flash_attn_func" &>/dev/null 2>&1; then
        print_success "flash-attention 4 already installed"
        return 0
    fi

    pip install flash-attn-4 \
        "nvidia-cutlass-dsl>=4.4.2" quack-kernels torch-c-dlpack-ext cuda-python \
        --extra-index-url https://pypi.nvidia.com --no-build-isolation

    print_success "flash-attention 4 installed"
}

install_flashmla() {
    print_step "Installing FlashMLA..."

    # Check if already installed
    if python -c "import flash_mla" &> /dev/null 2>&1; then
        print_success "FlashMLA already installed"
        return 0
    fi

    print_step "Installing FlashMLA from git (this may take 5-10 minutes)..."
    env "${FLASH_MLA_ENV[@]}" pip install "git+https://github.com/deepseek-ai/FlashMLA.git@${FLASHMLA_COMMIT}" --no-build-isolation

    print_success "FlashMLA installed"
}

# The DeepGEMM runtime contract: the pinned `sgl-deep-gemm` distribution AND the
# fp8_mqa_logits signature BatchGen calls. An `import deep_gemm` alone is not
# enough — the upstream deepseek-ai package imports under the same name but has
# no max_seqlen_k parameter, so it must not satisfy the skip gate.
deepgemm_contract_ok() {
    DG_DIST="$DEEPGEMM_DIST" DG_VERSION="$DEEPGEMM_DIST_VERSION" python - <<'PY' &> /dev/null
import inspect
import os
from importlib.metadata import version

import deep_gemm

assert version(os.environ["DG_DIST"]) == os.environ["DG_VERSION"]
assert "max_seqlen_k" in inspect.signature(deep_gemm.fp8_mqa_logits).parameters
PY
}

# Locate the wheel produced by build_sgl_deep_gemm.sh under a DeepGEMM checkout.
# Exactly one must match, otherwise the caller fails instead of guessing. A
# py3-none-any wheel is retagged as Linux/x86_64 before installation. Do not
# claim a manylinux baseline here: the actual glibc floor depends on the build
# image and must be checked separately before publishing a manylinux tag.
find_deepgemm_wheel() {
    local repo="$1" found count whl
    found="$(find "$repo/dist" -maxdepth 1 -type f -name 'sgl_deep_gemm-*.whl' | sort)"
    count="$(printf '%s' "$found" | grep -c . || true)"
    if [[ "$count" != "1" ]]; then
        print_error "expected exactly 1 sgl_deep_gemm wheel under $repo, found $count" >&2
        return 1
    fi
    whl="$found"
    if [[ "$whl" == *-py3-none-any.whl ]]; then
        if [[ "$(uname -m)" != "x86_64" ]]; then
            print_error "DeepGEMM release wheel retagging requires an x86_64 build host" >&2
            return 1
        fi
        python -m wheel tags --platform-tag "$DEEPGEMM_PLATFORM_TAG" --remove "$whl" >&2
        whl="${whl%-py3-none-any.whl}-py3-none-${DEEPGEMM_PLATFORM_TAG}.whl"
    fi
    printf '%s\n' "$whl"
}

clean_deepgemm_wheels() {
    local repo="$1"
    [[ -d "$repo/dist" ]] || return 0
    find "$repo/dist" -maxdepth 1 -type f -name 'sgl_deep_gemm-*.whl' -delete
}

remove_upstream_deepgemm() {
    if python -c 'from importlib.metadata import version; version("deep-gemm")' &>/dev/null; then
        print_step "Removing incompatible upstream deep-gemm distribution..."
        pip uninstall -y deep-gemm
        # Both distributions own the same import package. If a previous
        # partial upgrade left both metadata records behind, uninstalling the
        # upstream RECORD may remove files from the SGL package too. Remove its
        # metadata as well so the following install cannot be skipped as
        # already satisfied.
        if python -c 'from importlib.metadata import version; version("sgl-deep-gemm")' &>/dev/null; then
            pip uninstall -y sgl-deep-gemm
        fi
    fi
}

install_deepgemm() {
    print_step "Installing DeepGEMM (${DEEPGEMM_DIST} ${DEEPGEMM_VERSION})..."

    # Check if already installed (distribution + runtime signature)
    if deepgemm_contract_ok; then
        print_success "DeepGEMM already installed (${DEEPGEMM_DIST}==${DEEPGEMM_DIST_VERSION})"
        return 0
    fi

    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"

    if [[ -d "$DEEPGEMM_SRC_DIR" ]]; then
        print_step "Updating existing DeepGEMM repository..."
        cd "$DEEPGEMM_SRC_DIR"
        git fetch origin --tags
        git checkout "$DEEPGEMM_VERSION"
        git submodule update --init --recursive
    else
        print_step "Cloning DeepGEMM repository ($DEEPGEMM_REPO_URL)..."
        git clone --recursive "$DEEPGEMM_REPO_URL" "$DEEPGEMM_SRC_DIR"
        cd "$DEEPGEMM_SRC_DIR"
        git checkout "$DEEPGEMM_VERSION"
        git submodule update --init --recursive
    fi

    print_step "Building DeepGEMM (this may take 5-10 minutes)..."
    pip install "apache-tvm-ffi==${TVM_FFI_VERSION}" "wheel==${WHEEL_VERSION}"
    clean_deepgemm_wheels "$PWD"
    bash ./build_sgl_deep_gemm.sh

    local dg_wheel
    dg_wheel="$(find_deepgemm_wheel "$PWD")" || exit 1
    remove_upstream_deepgemm
    pip install "$dg_wheel"

    if ! deepgemm_contract_ok; then
        print_error "DeepGEMM installed but the runtime contract failed"
        print_error "  (expected ${DEEPGEMM_DIST}==${DEEPGEMM_DIST_VERSION} with fp8_mqa_logits(max_seqlen_k=...))"
        exit 1
    fi

    print_success "DeepGEMM installed"
}

batchgen_kernels_contract_ok() {
    # Importing the namespace package is not sufficient: a source checkout or an
    # incomplete wheel can satisfy that check while the first GPT-OSS request
    # later fails when it lazily imports the fused attention extension.
    python - <<'PY' &>/dev/null
import importlib
import os
import site
import sysconfig

import batchgen_kernels

module_path = getattr(batchgen_kernels, "__file__", None)
site_roots = {
    os.path.realpath(path)
    for path in (*site.getsitepackages(), sysconfig.get_paths()["purelib"])
}
if not module_path or not any(
    os.path.commonpath((os.path.realpath(module_path), root)) == root
    for root in site_roots
):
    raise ImportError(
        "batchgen_kernels resolves outside site-packages "
        f"({module_path!r}); refusing an editable/source shadow"
    )

for module_name in (
    "batchgen_kernels.attention._C_fused_ops",
    "batchgen_kernels.attention._C_gqa_mha_decode_bf16",
):
    importlib.import_module(module_name)
PY
}

install_batchgen_kernels() {
    # A pre-built wheel is usable only when the runtime extensions needed by
    # the model path are present.  If a partial/source-only package is visible,
    # rebuild from the checked-out sources instead of silently skipping it.
    if batchgen_kernels_contract_ok; then
        print_success "batchgen_kernels AOT runtime contract satisfied; skipping compilation"
        return 0
    fi

    print_warning "batchgen_kernels is missing or its AOT runtime contract is incomplete; rebuilding"

    print_step "Installing batchgen_kernels (AOT-compiled CUDA kernel extensions)..."

    if [[ -f "$BATCHGEN_DIR/batchgen_kernels/setup.py" ]]; then
        cd "$BATCHGEN_DIR/batchgen_kernels"
        pip install . --no-build-isolation --no-deps --force-reinstall
        if ! batchgen_kernels_contract_ok; then
            print_error "batchgen_kernels installed but the AOT runtime contract is still incomplete"
            return 1
        fi
        print_success "batchgen_kernels installed and AOT runtime contract verified"
    else
        print_warning "batchgen_kernels/setup.py not found, skipping kernel compilation"
    fi
}

install_batchgen() {
    print_step "Installing BatchGen..."

    # Find BatchGen directory (script is in scripts/, BatchGen is parent)
    if [[ -f "$BATCHGEN_DIR/setup.py" ]]; then
        cd "$BATCHGEN_DIR"
        # Replace stale editable/direct-url installs from another worktree
        # without allowing dependency resolution to change the verified ABI.
        pip install . --no-build-isolation --no-deps --force-reinstall
        print_success "BatchGen installed"
    else
        print_error "Could not find BatchGen setup.py at $BATCHGEN_DIR"
        print_error "Please run this script from the BatchGen/scripts directory"
        exit 1
    fi
}

cleanup() {
    if [[ "$KEEP_BUILD_DIR" != "1" ]]; then
        print_step "Cleaning up build directory..."
        rm -rf "$INSTALL_DIR"
        print_success "Cleanup complete"
    else
        print_warning "Build directory kept at: $INSTALL_DIR"
    fi
}

# Hopper fast path: stage matching pre-built wheels (flash-attn 3, FlashMLA,
# DeepGEMM, batchgen_kernels) from the public GitHub release into a local dir so
# the caller installs them instead of compiling (~2 min vs ~40-60 min). All four
# plus the apache-tvm-ffi runtime wheel must be present for this Python/GPU-arch;
# otherwise WHEEL_DIR is left empty and
# the caller falls back to building from source. Never fatal.
try_download_wheels() {
    [[ -n "$WHEEL_DIR" ]] && return 0          # explicit local --wheel-dir wins
    [[ $FROM_SOURCE -eq 1 ]] && return 0        # user forced a source build
    command -v curl &>/dev/null || { print_warning "curl not found; building from source."; return 0; }

    local pytag api assets tmp w
    pytag="cp$(python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")' 2>/dev/null || echo cp311)"
    if [[ -n "$RELEASE_TAG" ]]; then
        api="https://api.github.com/repos/${BATCHGEN_REPO}/releases/tags/${RELEASE_TAG}"
    else
        api="https://api.github.com/repos/${BATCHGEN_REPO}/releases/latest"
    fi
    print_step "Looking for pre-built Hopper wheels on ${BATCHGEN_REPO} (${RELEASE_TAG:-latest}, ${pytag}/${BUILD_ARCH})..."

    # Select matching asset names and download URLs (python-ABI + GPU-arch aware).
    assets="$(curl -fsSL "$api" 2>/dev/null | PYTAG="$pytag" WANT_ARCH="$BUILD_ARCH" python3 -c '
import sys, os, json
pytag = os.environ["PYTAG"]; arch = os.environ["WANT_ARCH"]
try:
    assets = json.load(sys.stdin).get("assets", [])
except Exception:
    sys.exit(0)
want = ("flash_attn_3", "flash_mla", "sgl_deep_gemm", "apache_tvm_ffi", "batchgen_kernels")
for a in assets:
    n = a.get("name", "")
    if not n.endswith(".whl") or not any(n.startswith(w) for w in want):
        continue
    if ("abi3" not in n) and ("py3-none" not in n) and (pytag not in n):
        continue                                    # python-ABI mismatch
    if n.startswith("batchgen_kernels") and (arch not in n):
        continue                                    # GPU-arch mismatch
    u = a.get("browser_download_url", "")
    if u:
        print(f"{n}\t{u}")
' || true)"

    # Require all five wheels; otherwise fall back to a full source build.
    for w in flash_attn_3 flash_mla sgl_deep_gemm apache_tvm_ffi batchgen_kernels; do
        if ! printf '%s\n' "$assets" | grep -q "^${w}.*\\.whl[[:space:]]"; then
            print_warning "no pre-built '${w}' wheel for this env (${pytag}/${BUILD_ARCH}) — building from source."
            return 0
        fi
    done

    tmp="$INSTALL_DIR/prebuilt_wheels"          # under INSTALL_DIR so cleanup() removes it
    rm -rf "$tmp"; mkdir -p "$tmp"
    ( cd "$tmp" && printf '%s\n' "$assets" | while IFS=$'\t' read -r n u; do
        [[ -n "$u" ]] && { curl -fsSL -o "$n" "$u" || print_warning "download failed: $u"; }
      done )

    # Accept only a COMPLETE set; a partial download falls back to source.
    for w in flash_attn_3 flash_mla sgl_deep_gemm apache_tvm_ffi batchgen_kernels; do
        if ! ls "$tmp/${w}"*.whl &>/dev/null 2>&1; then
            print_warning "incomplete wheel set (missing ${w}); building from source."
            rm -rf "$tmp"
            return 0
        fi
    done
    WHEEL_DIR="$tmp"
    print_success "Fetched pre-built wheels into $WHEEL_DIR (skipping ~40-60 min compilation)"
}

show_help() {
    echo "BatchGen Dependency Installation Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --all             Install all dependencies (default for Hopper GPUs)"
    echo "  --flash-attn      Install flash-attention 3 only"
    echo "  --flashmla        Install FlashMLA only"
    echo "  --deepgemm        Install DeepGEMM only"
    echo "  --batchgen        Install BatchGen only"
    echo "  --wheel-dir DIR   Use pre-built wheels from a LOCAL dir (offline; auto-detects arch)"
    echo "  --release-tag TAG Fetch pre-built wheels from this GitHub release tag (default: latest)"
    echo "  --from-source     Force building all deps from source (skip the wheel fast path)"
    echo "  --skip-gpu-check  Skip GPU architecture detection"
    echo "  --keep-build      Keep build directory after installation"
    echo "  --help            Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  BATCHGEN_INSTALL_DIR  Directory for cloning repos (default: /tmp/batchgen_deps)"
    echo "  WHEEL_DIR             Pre-built wheel directory (same as --wheel-dir)"
    echo "  BATCHGEN_RELEASE_TAG  Release tag to fetch pre-built wheels from (default: latest)"
    echo "  KEEP_BUILD_DIR        Set to 1 to keep build directory"
    echo ""
    echo "Examples:"
    echo "  $0                                  # Install everything (auto-detect GPU)"
    echo "  $0 --flash-attn                     # Install only flash-attention 3"
    echo "  $0 --wheel-dir /path/to/wheels      # Install deps from pre-built wheels"
    echo "  $0 --skip-gpu-check                 # Install all deps without GPU check"
}

# Pre-build (warm) the core_engine JIT extension now, while the CUDA toolkit is on
# hand from the source build, so the FIRST server launch does not require CUDA_HOME
# / nvcc on PATH. The compile is CPU-only (no GPU needed). Non-fatal: on failure the
# engine JIT-builds at first launch instead (which then needs CUDA_HOME set).
warm_core_engine() {
    python -c "import batchgen" &> /dev/null || return 0   # batchgen not installed; skip
    print_step "Warming the core_engine JIT build (so the first server launch needs no CUDA toolkit)..."
    if python -c "from batchgen.models.engine_loader import core_engine" > /tmp/batchgen_core_engine_warm.log 2>&1; then
        print_success "core_engine JIT built and cached"
    else
        print_warning "core_engine warm build failed; it will JIT-build on first server launch (set CUDA_HOME then). Log: /tmp/batchgen_core_engine_warm.log"
    fi
}

main() {
    echo "========================================"
    echo "  BatchGen Dependency Installer"
    echo "========================================"
    echo ""

    # Parse arguments
    INSTALL_ALL=0
    INSTALL_FLASH_ATTN=0
    INSTALL_FLASHMLA=0
    INSTALL_DEEPGEMM=0
    INSTALL_BATCHGEN=0
    SKIP_GPU_CHECK=0
    WHEEL_DIR="${WHEEL_DIR:-}"  # honour env var; overridden by --wheel-dir

    while [[ $# -gt 0 ]]; do
        case $1 in
            --all)
                INSTALL_ALL=1
                shift
                ;;
            --flash-attn)
                INSTALL_FLASH_ATTN=1
                shift
                ;;
            --flashmla)
                INSTALL_FLASHMLA=1
                shift
                ;;
            --deepgemm)
                INSTALL_DEEPGEMM=1
                shift
                ;;
            --batchgen)
                INSTALL_BATCHGEN=1
                shift
                ;;
            --skip-gpu-check)
                SKIP_GPU_CHECK=1
                shift
                ;;
            --wheel-dir)
                WHEEL_DIR="$2"
                shift 2
                ;;
            --from-source)
                FROM_SOURCE=1
                shift
                ;;
            --release-tag)
                RELEASE_TAG="$2"
                shift 2
                ;;
            --keep-build)
                export KEEP_BUILD_DIR=1
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Default to a full install unless the user explicitly narrowed the targets.
    # Modifier-only invocations (--from-source, --skip-gpu-check, --wheel-dir,
    # --release-tag, --keep-build) must still install everything, not silently
    # no-op into a misleading "Installation complete!".
    if [[ $INSTALL_ALL -eq 0 && $INSTALL_FLASH_ATTN -eq 0 && $INSTALL_FLASHMLA -eq 0 \
          && $INSTALL_DEEPGEMM -eq 0 && $INSTALL_BATCHGEN -eq 0 ]]; then
        INSTALL_ALL=1
    fi

    # Check prerequisites
    check_prerequisites

    # Install torch if needed
    install_torch

    # Check GPU architecture
    GPU_GENERATION="hopper"  # default
    IS_HOPPER=0
    IS_BLACKWELL=0
    if [[ $SKIP_GPU_CHECK -eq 0 ]]; then
        check_gpu_arch  # sets GPU_GENERATION
    else
        print_warning "GPU check skipped, assuming Hopper architecture"
        GPU_GENERATION="hopper"
    fi
    [[ "$GPU_GENERATION" == "hopper"    ]] && IS_HOPPER=1
    [[ "$GPU_GENERATION" == "blackwell" ]] && IS_BLACKWELL=1

    # Install dependencies based on options
    if [[ $INSTALL_ALL -eq 1 ]]; then
        if [[ $IS_HOPPER -eq 1 ]]; then
            try_download_wheels   # populate WHEEL_DIR from the public release; else source-build below
            if [[ -n "$WHEEL_DIR" && -d "$WHEEL_DIR" ]]; then
                print_step "Installing Hopper dependencies from pre-built wheels: $WHEEL_DIR"
                remove_upstream_deepgemm
                if ! pip install --find-links "$WHEEL_DIR" --no-index \
                    flash-attn-3 flash-mla sgl-deep-gemm batchgen-kernels \
                    "apache-tvm-ffi==${TVM_FFI_VERSION}"; then
                    print_warning "pre-built wheel installation failed; validating each dependency and source-building only what is missing"
                fi
                # Validate the runtime interfaces rather than trusting wheel
                # metadata. Each installer skips a dependency that imports and
                # source-builds only a missing or incorrectly packaged one.
                install_flash_attention
                install_flashmla
                install_deepgemm
                print_success "Hopper dependencies installed and verified"
            else
                install_flash_attention
                install_flashmla
                install_deepgemm
                # Reinstall PyTorch — building deps from source may downgrade torch or triton
                print_step "Reinstalling PyTorch to ensure correct version after dependency builds..."
                local cuda_channel="${TORCH_CUDA_CHANNEL:-cu128}"
                pip install "torch==2.9.0+${cuda_channel}" --index-url "https://download.pytorch.org/whl/${cuda_channel}"
                print_success "PyTorch reinstalled"
            fi
        elif [[ $IS_BLACKWELL -eq 1 ]]; then
            if [[ -n "$WHEEL_DIR" && -d "$WHEEL_DIR" ]]; then
                print_step "Installing Blackwell dependencies from pre-built wheels: $WHEEL_DIR"
                remove_upstream_deepgemm
                for whl in "$WHEEL_DIR"/*.whl; do
                    [[ -f "$whl" ]] || continue
                    pip install "$whl" --no-deps \
                      || { SITE=$(python -c "import site; print(site.getsitepackages()[0])"); unzip -oq "$whl" -d "$SITE/"; }
                done
                # FA4 runtime deps (pure-Python, not in wheel cache)
                pip install "nvidia-cutlass-dsl>=4.4.2" quack-kernels torch-c-dlpack-ext cuda-python \
                    --extra-index-url https://pypi.nvidia.com -q
                install_deepgemm
                print_success "Blackwell dependencies installed from wheels"
            else
                install_flash_attention_4
                install_flashmla
                install_deepgemm
            fi
        else
            print_warning "Skipping GPU-specific dependencies (Ampere or unknown GPU detected)"
            print_warning "Use --skip-gpu-check to force Hopper installation"
        fi
        install_batchgen_kernels
        install_batchgen
    else
        if [[ $INSTALL_FLASH_ATTN -eq 1 ]]; then
            install_flash_attention
        fi
        if [[ $INSTALL_FLASHMLA -eq 1 ]]; then
            install_flashmla
        fi
        if [[ $INSTALL_DEEPGEMM -eq 1 ]]; then
            install_deepgemm
        fi
        if [[ $INSTALL_BATCHGEN -eq 1 ]]; then
            install_batchgen_kernels
            install_batchgen
        fi
    fi

    # Warm the core_engine JIT now (CUDA toolkit is available during install) so
    # the first server launch needs no CUDA_HOME / nvcc on PATH.
    warm_core_engine

    # Cleanup
    cleanup

    echo ""
    echo "========================================"
    print_success "Installation complete!"
    echo "========================================"
    echo ""
    echo "You can now use BatchGen. Example:"
    echo "  python -m batchgen.parameter_server --model deepseek-ai/DeepSeek-R1"
}

main "$@"
