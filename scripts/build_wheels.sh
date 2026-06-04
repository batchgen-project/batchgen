#!/bin/bash
# ---------------------------------------------------------------------------- #
#  BatchGen Wheel Builder                                                       #
#  Copyright (c) EfficientMoE team 2025                                         #
#                                                                               #
#  Builds pre-compiled wheels for all CUDA dependencies + batchgen_kernels.     #
#  Users can install these wheels with pip install *.whl — no compilation.      #
#                                                                               #
#  Target: CUDA 12.8 + PyTorch 2.9 + Python 3.11 + Hopper (SM90)              #
#                                                                               #
#  Usage:                                                                       #
#    bash scripts/build_wheels.sh --output-dir /path/to/wheels                  #
#                                                                               #
#  Options:                                                                     #
#    --output-dir DIR     Where to put built wheels (required)                  #
#    --deps-dir DIR       Where to clone/find dep repos (default: /tmp/...)     #
#    --skip-flash-attn    Skip flash-attention 3 wheel                          #
#    --skip-flashmla      Skip FlashMLA wheel                                  #
#    --skip-deepgemm      Skip DeepGEMM wheel                                  #
#    --skip-kernels       Skip batchgen_kernels wheel                           #
#    --only-kernels       Only build batchgen_kernels wheel                     #
# ---------------------------------------------------------------------------- #

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

step()    { echo -e "${BLUE}==>${NC} $1"; }
ok()      { echo -e "${GREEN}[OK]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail()    { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

# ── Pinned versions (must match install_deps.sh / Dockerfile) ──
FLASH_ATTN_VERSION="v2.8.2"
FLASHMLA_COMMIT="1408756a88e52a25196b759eaf8db89d2b51b5a1"
DEEPGEMM_VERSION="v2.1.1.post3"

# ── Build target arch (sm90a default / sm100 / all) ──
# Hopper (sm90a) disables FlashMLA SM100 kernels; sm100/all enable them
# (requires nvcc >= 12.9).
BUILD_ARCH="${BUILD_ARCH:-sm90a}"
FLASH_MLA_ENV=()
if [[ "$BUILD_ARCH" == "sm90a" ]]; then
    FLASH_MLA_ENV=(FLASH_MLA_DISABLE_SM100=1)
fi

# ── Defaults ──
OUTPUT_DIR=""
DEPS_DIR="${BATCHGEN_DEPS_DIR:-/tmp/batchgen_wheel_build}"
SKIP_FLASH_ATTN=0
SKIP_FLASHMLA=0
SKIP_DEEPGEMM=0
SKIP_KERNELS=0
ONLY_KERNELS=0

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)    OUTPUT_DIR="$2";    shift 2 ;;
        --deps-dir)      DEPS_DIR="$2";      shift 2 ;;
        --skip-flash-attn) SKIP_FLASH_ATTN=1; shift ;;
        --skip-flashmla)   SKIP_FLASHMLA=1;   shift ;;
        --skip-deepgemm)   SKIP_DEEPGEMM=1;   shift ;;
        --skip-kernels)    SKIP_KERNELS=1;     shift ;;
        --only-kernels)    ONLY_KERNELS=1;     shift ;;
        --help) echo "Usage: $0 --output-dir DIR [OPTIONS]"; exit 0 ;;
        *) fail "Unknown option: $1" ;;
    esac
done

[[ -z "$OUTPUT_DIR" ]] && fail "--output-dir is required"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$DEPS_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCHGEN_DIR="$(dirname "$SCRIPT_DIR")"

# ── Verify environment ──
step "Verifying build environment..."

PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
TORCH_VERSION=$(python -c 'import torch; print(torch.__version__)')
CUDA_VERSION=$(python -c 'import torch; print(torch.version.cuda)')

echo "  Python:  $PYTHON_VERSION"
echo "  PyTorch: $TORCH_VERSION"
echo "  CUDA:    $CUDA_VERSION"

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    fail "CUDA not available. Wheels must be built on a GPU machine."
fi

GPU_ARCH=$(python -c "import torch; print(torch.cuda.get_device_capability())")
echo "  GPU:     $(python -c 'import torch; print(torch.cuda.get_device_name())')"
echo "  SM:      $GPU_ARCH"
ok "Build environment verified"

# ── Helper: clone or update repo ──
clone_or_update() {
    local name="$1" url="$2" ref="$3"
    cd "$DEPS_DIR"
    if [[ -d "$name" ]]; then
        step "Updating $name..."
        cd "$name"
        git fetch origin 2>/dev/null || warn "git fetch failed (network?), using existing checkout"
        git checkout "$ref" 2>/dev/null || true
    else
        step "Cloning $name..."
        git clone --recursive "$url" "$name" || fail "Failed to clone $name. Check network."
        cd "$name"
        git checkout "$ref"
    fi
    git submodule update --init --recursive 2>/dev/null || warn "Submodule update failed, may need manual fix"
}

# ── Build flash-attention 3 ──
if [[ $ONLY_KERNELS -eq 0 && $SKIP_FLASH_ATTN -eq 0 ]]; then
    step "Building flash-attention 3 wheel ($FLASH_ATTN_VERSION)..."
    clone_or_update "flash-attention" \
        "https://github.com/Dao-AILab/flash-attention.git" \
        "$FLASH_ATTN_VERSION"
    cd "$DEPS_DIR/flash-attention/hopper"
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
        pip wheel . --no-build-isolation --no-deps -w "$OUTPUT_DIR"
    ok "flash-attention 3 wheel built"
else
    warn "Skipping flash-attention 3"
fi

# ── Build FlashMLA ──
if [[ $ONLY_KERNELS -eq 0 && $SKIP_FLASHMLA -eq 0 ]]; then
    step "Building FlashMLA wheel ($FLASHMLA_COMMIT)..."
    clone_or_update "FlashMLA" \
        "https://github.com/deepseek-ai/FlashMLA.git" \
        "$FLASHMLA_COMMIT"
    cd "$DEPS_DIR/FlashMLA"
    env "${FLASH_MLA_ENV[@]}" pip wheel . --no-build-isolation --no-deps -w "$OUTPUT_DIR"
    ok "FlashMLA wheel built"
else
    warn "Skipping FlashMLA"
fi

# ── Build DeepGEMM ──
if [[ $ONLY_KERNELS -eq 0 && $SKIP_DEEPGEMM -eq 0 ]]; then
    step "Building DeepGEMM wheel ($DEEPGEMM_VERSION)..."
    clone_or_update "DeepGEMM" \
        "https://github.com/deepseek-ai/DeepGEMM.git" \
        "$DEEPGEMM_VERSION"
    cd "$DEPS_DIR/DeepGEMM"
    pip wheel . --no-build-isolation --no-deps -w "$OUTPUT_DIR"
    ok "DeepGEMM wheel built"
else
    warn "Skipping DeepGEMM"
fi

# ── Build batchgen_kernels ──
if [[ $SKIP_KERNELS -eq 0 ]]; then
    step "Building batchgen_kernels wheel..."
    cd "$BATCHGEN_DIR/batchgen_kernels"
    pip wheel . --no-build-isolation --no-deps -w "$OUTPUT_DIR"
    ok "batchgen_kernels wheel built"
else
    warn "Skipping batchgen_kernels"
fi

# ── Build batchgen ──
step "Building batchgen wheel..."
cd "$BATCHGEN_DIR"
pip wheel . --no-deps -w "$OUTPUT_DIR"
ok "batchgen wheel built"

# ── Summary ──
echo ""
echo -e "${BLUE}$(printf '=%.0s' {1..60})${NC}"
echo -e "${GREEN}Wheels built successfully:${NC}"
echo ""
ls -lh "$OUTPUT_DIR"/*.whl 2>/dev/null
echo ""
echo "Install on a matching machine (CUDA $CUDA_VERSION, PyTorch $TORCH_VERSION, Python $PYTHON_VERSION):"
echo "  pip install $OUTPUT_DIR/*.whl"
echo ""
echo "Upload to GitHub Release:"
echo "  gh release create vX.Y.Z --title 'BatchGen vX.Y.Z'"
echo "  gh release upload vX.Y.Z $OUTPUT_DIR/*.whl"
