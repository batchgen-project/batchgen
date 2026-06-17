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

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration - pinned versions for reproducibility
FLASH_ATTN_VERSION="v2.8.2"
FLASHMLA_COMMIT="1408756a88e52a25196b759eaf8db89d2b51b5a1"
DEEPGEMM_VERSION="v2.1.1.post3"

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

check_prerequisites() {
    print_step "Checking prerequisites..."

    # Check Python
    if ! command -v python &> /dev/null; then
        print_error "Python not found. Please install Python 3.11+."
        exit 1
    fi

    PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [[ $(echo "$PYTHON_VERSION < 3.11" | bc -l) -eq 1 ]]; then
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

    # Check CUDA
    if ! command -v nvcc &> /dev/null; then
        print_warning "nvcc not found. CUDA may not be properly configured."
        print_warning "Make sure CUDA toolkit is installed and in PATH."
    else
        CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
        print_success "CUDA $CUDA_VERSION found"
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

install_torch() {
    print_step "Checking PyTorch installation..."

    if python -c "import torch; print(torch.__version__)" &> /dev/null; then
        TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
        print_success "PyTorch $TORCH_VERSION already installed"

        # Check CUDA availability
        CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
        if [[ "$CUDA_AVAILABLE" == "False" ]]; then
            print_warning "PyTorch installed but CUDA not available. Consider reinstalling with CUDA support."
        fi
    else
        print_step "Installing PyTorch with CUDA 12.8 support..."
        pip install torch==2.9.0+cu128 --index-url https://download.pytorch.org/whl/cu128
        print_success "PyTorch installed"
    fi
}

install_flash_attention() {
    print_step "Installing flash-attention 3 (Hopper)..."

    # Check if already installed
    if python -c "import flash_attn_hopper" &> /dev/null 2>&1; then
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
        git clone --recursive https://github.com/Dao-AILab/flash-attention.git
        cd flash-attention
        git checkout "$FLASH_ATTN_VERSION"
    fi

    print_step "Building flash-attention 3 (this may take 10-20 minutes)..."
    cd hopper
    FLASH_ATTENTION_FORCE_BUILD=TRUE pip install . --no-build-isolation

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

install_deepgemm() {
    print_step "Installing DeepGEMM..."

    # Check if already installed
    if python -c "import deep_gemm" &> /dev/null 2>&1; then
        print_success "DeepGEMM already installed"
        return 0
    fi

    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"

    if [[ -d "DeepGEMM" ]]; then
        print_step "Updating existing DeepGEMM repository..."
        cd DeepGEMM
        git fetch origin
        git checkout "$DEEPGEMM_VERSION"
        git submodule update --init --recursive
    else
        print_step "Cloning DeepGEMM repository..."
        git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git
        cd DeepGEMM
        git checkout "$DEEPGEMM_VERSION"
        git submodule update --init --recursive
    fi

    print_step "Building DeepGEMM (this may take 5-10 minutes)..."
    pip install . --no-build-isolation

    print_success "DeepGEMM installed"
}

install_batchgen_kernels() {
    print_step "Installing batchgen_kernels (AOT-compiled CUDA kernel extensions)..."

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    BATCHGEN_DIR="$(dirname "$SCRIPT_DIR")"

    if [[ -f "$BATCHGEN_DIR/batchgen_kernels/setup.py" ]]; then
        cd "$BATCHGEN_DIR/batchgen_kernels"
        pip install . --no-build-isolation
        print_success "batchgen_kernels installed"
    else
        print_warning "batchgen_kernels/setup.py not found, skipping kernel compilation"
    fi
}

install_batchgen() {
    print_step "Installing BatchGen..."

    # Find BatchGen directory (script is in scripts/, BatchGen is parent)
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    BATCHGEN_DIR="$(dirname "$SCRIPT_DIR")"

    if [[ -f "$BATCHGEN_DIR/setup.py" ]]; then
        cd "$BATCHGEN_DIR"
        pip install .
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
    echo "  --wheel-dir DIR   Use pre-built wheels for FA3/FA4/FlashMLA/DeepGEMM (auto-detects arch)"
    echo "  --skip-gpu-check  Skip GPU architecture detection"
    echo "  --keep-build      Keep build directory after installation"
    echo "  --help            Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  BATCHGEN_INSTALL_DIR  Directory for cloning repos (default: /tmp/batchgen_deps)"
    echo "  WHEEL_DIR             Pre-built wheel directory (same as --wheel-dir)"
    echo "  KEEP_BUILD_DIR        Set to 1 to keep build directory"
    echo ""
    echo "Examples:"
    echo "  $0                                  # Install everything (auto-detect GPU)"
    echo "  $0 --flash-attn                     # Install only flash-attention 3"
    echo "  $0 --wheel-dir /path/to/wheels      # Install deps from pre-built wheels"
    echo "  $0 --skip-gpu-check                 # Install all deps without GPU check"
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

    if [[ $# -eq 0 ]]; then
        INSTALL_ALL=1
    fi

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
            if [[ -n "$WHEEL_DIR" && -d "$WHEEL_DIR" ]]; then
                print_step "Installing Hopper dependencies from pre-built wheels: $WHEEL_DIR"
                pip install --find-links "$WHEEL_DIR" --no-index \
                    flash-attn-hopper flash-mla deep-gemm 2>/dev/null || \
                    pip install "$WHEEL_DIR"/*.whl
                print_success "Hopper dependencies installed from wheels"
            else
                install_flash_attention
                install_flashmla
                install_deepgemm
                # Reinstall PyTorch — building deps from source may downgrade torch or triton
                print_step "Reinstalling PyTorch to ensure correct version after dependency builds..."
                pip install torch==2.9.0+cu128 --index-url https://download.pytorch.org/whl/cu128
                print_success "PyTorch reinstalled"
            fi
        elif [[ $IS_BLACKWELL -eq 1 ]]; then
            if [[ -n "$WHEEL_DIR" && -d "$WHEEL_DIR" ]]; then
                print_step "Installing Blackwell dependencies from pre-built wheels: $WHEEL_DIR"
                for whl in "$WHEEL_DIR"/*.whl; do
                    [[ -f "$whl" ]] || continue
                    pip install "$whl" --no-deps \
                      || { SITE=$(python -c "import site; print(site.getsitepackages()[0])"); unzip -oq "$whl" -d "$SITE/"; }
                done
                # FA4 runtime deps (pure-Python, not in wheel cache)
                pip install "nvidia-cutlass-dsl>=4.4.2" quack-kernels torch-c-dlpack-ext cuda-python \
                    --extra-index-url https://pypi.nvidia.com -q
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
