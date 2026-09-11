#!/bin/bash
# ---------------------------------------------------------------------------- #
#  BatchGen release-wheel builder — the "CI we run ourselves"                   #
#                                                                               #
#  Builds the COMPLETE pre-built wheel set for the Hopper reference environment #
#  and (optionally) uploads it to a GitHub release tag, so that                 #
#  `scripts/install_deps.sh` can auto-download it (fast path) instead of        #
#  compiling on the user's machine.                                             #
#                                                                               #
#  There is intentionally NO tag-triggered GitHub Actions workflow (no stable   #
#  GPU runner yet). Run this MANUALLY on a Hopper node (e.g. h200-instance-3),  #
#  inside the batchgen conda env / container, from the repo root, BEFORE        #
#  cutting a new release:                                                        #
#                                                                               #
#      bash scripts/build_release_wheels.sh                 # build into ./dist  #
#      bash scripts/build_release_wheels.sh --tag v1.1.0    # build + upload     #
#                                                                               #
#  Reference env: Python 3.11, torch 2.9.0+cu128, CUDA 12.8, sm90a (Hopper).    #
#  Produces wheels: flash_attn (FA3), flash_mla, deep_gemm,                     #
#  batchgen_kernels (sm90a), batchgen. Version pins MIRROR install_deps.sh —    #
#  keep the two in sync.                                                        #
# ---------------------------------------------------------------------------- #
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCHGEN_DIR="$(dirname "$SCRIPT_DIR")"

# --- Pins (keep in sync with scripts/install_deps.sh) ------------------------
FLASH_ATTN_VERSION="v2.8.2"
FLASHMLA_COMMIT="1408756a88e52a25196b759eaf8db89d2b51b5a1"
DEEPGEMM_VERSION="v2.1.1.post3"
BUILD_ARCH="${BUILD_ARCH:-sm90a}"                 # sm90a (Hopper) reference build

# --- Options -----------------------------------------------------------------
OUT_DIR="${BATCHGEN_DIST_DIR:-$BATCHGEN_DIR/dist}"
BUILD_DIR="${BATCHGEN_BUILD_DIR:-/tmp/batchgen_wheel_build}"
REPO="${BATCHGEN_REPO:-batchgen-project/batchgen}"
RELEASE_TAG=""

BLUE='\033[0;34m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${BLUE}==>${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARNING]${NC} $*"; }

usage() {
    cat <<EOF
BatchGen release-wheel builder (run on a Hopper node, e.g. h200-instance-3)

Usage: bash scripts/build_release_wheels.sh [OPTIONS]

Options:
  --tag TAG         Upload the built wheels to GitHub release TAG (gh required)
  --out DIR         Output directory for wheels (default: ./dist)
  --build-arch A    Kernel arch: sm90a (Hopper, default) | sm100 (Blackwell)
  --repo OWNER/NAME Target repo for --tag upload (default: batchgen-project/batchgen)
  --help            Show this help

Environment: BATCHGEN_DIST_DIR, BATCHGEN_BUILD_DIR, BATCHGEN_REPO, BUILD_ARCH
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)        RELEASE_TAG="$2"; shift 2 ;;
        --out)        OUT_DIR="$2"; shift 2 ;;
        --build-arch) BUILD_ARCH="$2"; shift 2 ;;
        --repo)       REPO="$2"; shift 2 ;;
        --help)       usage; exit 0 ;;
        *)            echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# --- Preflight ---------------------------------------------------------------
command -v python >/dev/null || { echo "python not found"; exit 1; }
PYV=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
[[ "$PYV" == "3.11" ]] || warn "expected Python 3.11 (reference env); found $PYV — wheel tags will differ."
command -v nvcc >/dev/null || warn "nvcc not found — CUDA extension builds will fail without the toolkit."
python -c "import torch; assert torch.version.cuda" 2>/dev/null \
    || warn "torch with CUDA not importable — FA3/FlashMLA/DeepGEMM/batchgen_kernels builds need it."

mkdir -p "$OUT_DIR" "$BUILD_DIR"
log "Building the Hopper release wheel set (${BUILD_ARCH}, py${PYV}) into $OUT_DIR"

# 1) flash-attention 3 (Hopper) — clone recursively, build a wheel from source
log "flash-attention 3 (${FLASH_ATTN_VERSION})..."
cd "$BUILD_DIR"
[[ -d flash-attention ]] || git clone --recursive https://github.com/Dao-AILab/flash-attention.git
( cd flash-attention \
    && git fetch --tags --quiet \
    && git checkout "$FLASH_ATTN_VERSION" \
    && git submodule update --init --recursive \
    && FLASH_ATTENTION_FORCE_BUILD=TRUE pip wheel . --no-build-isolation --no-deps -w "$OUT_DIR" )

# 2) FlashMLA — Hopper build disables the SM100 kernels
log "FlashMLA (${FLASHMLA_COMMIT})..."
FLASH_MLA_ENV=()
[[ "$BUILD_ARCH" == "sm90a" ]] && FLASH_MLA_ENV=(FLASH_MLA_DISABLE_SM100=1)
env "${FLASH_MLA_ENV[@]}" pip wheel \
    "git+https://github.com/deepseek-ai/FlashMLA.git@${FLASHMLA_COMMIT}" \
    --no-build-isolation --no-deps -w "$OUT_DIR"

# 3) DeepGEMM — needs submodules (cutlass), so clone explicitly
log "DeepGEMM (${DEEPGEMM_VERSION})..."
cd "$BUILD_DIR"
[[ -d DeepGEMM ]] || git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git
( cd DeepGEMM \
    && git fetch --tags --quiet \
    && git checkout "$DEEPGEMM_VERSION" \
    && git submodule update --init --recursive \
    && pip wheel . --no-build-isolation --no-deps -w "$OUT_DIR" )

# 4) batchgen_kernels — arch-specific AOT CUDA extensions
log "batchgen_kernels (${BUILD_ARCH})..."
BUILD_ARCH="$BUILD_ARCH" pip wheel "$BATCHGEN_DIR/batchgen_kernels" \
    --no-build-isolation --no-deps -w "$OUT_DIR"

# 5) batchgen — pure-Python package (kernels ship separately, above)
log "batchgen (Python package)..."
pip wheel "$BATCHGEN_DIR" --no-deps -w "$OUT_DIR"

echo
ok "Wheels in $OUT_DIR:"
ls -1 "$OUT_DIR"/*.whl | sed 's/^/  /'

# --- Optional upload ---------------------------------------------------------
if [[ -n "$RELEASE_TAG" ]]; then
    command -v gh >/dev/null || { warn "gh not found; skipping upload of $RELEASE_TAG."; exit 1; }
    log "Uploading wheels to release ${RELEASE_TAG} on ${REPO} (--clobber)..."
    gh release upload "$RELEASE_TAG" "$OUT_DIR"/*.whl --repo "$REPO" --clobber
    ok "Uploaded the complete wheel set to ${RELEASE_TAG}."
else
    echo
    log "Build-only (no --tag). To publish: gh release upload <tag> $OUT_DIR/*.whl --repo $REPO --clobber"
fi
