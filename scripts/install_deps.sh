#!/bin/bash
# ---------------------------------------------------------------------------- #
#  BatchGen Dependency Installation Script                                      #
#  Copyright (c) EfficientMoE team 2025                                         #
#                                                                               #
#  This script installs the required dependencies for BatchGen on Hopper GPUs:  #
#  - flash-attention 3 (Hopper optimized)                                       #
#  - FlashMLA                                                                   #
#  - DeepGEMM                                                                   #
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

    GPU_ARCH=$(python -c "
import torch
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    if major >= 9:
        print('hopper')
    elif major >= 8:
        print('ampere')
    else:
        print('other')
else:
    print('no_gpu')
" 2>/dev/null || echo "unknown")

    if [[ "$GPU_ARCH" == "hopper" ]]; then
        print_success "Hopper GPU detected (SM90+)"
        return 0
    elif [[ "$GPU_ARCH" == "ampere" ]]; then
        print_warning "Ampere GPU detected. Hopper-specific optimizations will be skipped."
        return 1
    elif [[ "$GPU_ARCH" == "no_gpu" ]]; then
        print_warning "No GPU detected. Installing dependencies anyway (for CPU-only builds)."
        return 0
    else
        print_warning "Could not detect GPU architecture. Assuming Hopper."
        return 0
    fi
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

install_flashmla() {
    print_step "Installing FlashMLA..."

    # Check if already installed
    if python -c "import flash_mla" &> /dev/null 2>&1; then
        print_success "FlashMLA already installed"
        return 0
    fi

    print_step "Installing FlashMLA from git (this may take 5-10 minutes)..."
    FLASH_MLA_DISABLE_SM100=1 pip install "git+https://github.com/deepseek-ai/FlashMLA.git@${FLASHMLA_COMMIT}" --no-build-isolation

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
    echo "  --wheel-dir DIR   Use pre-built wheels for flash-attn/FlashMLA/DeepGEMM"
    echo "  --skip-gpu-check  Skip GPU architecture detection"
    echo "  --keep-build      Keep build directory after installation"
    echo "  --help            Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  BATCHGEN_INSTALL_DIR  Directory for cloning repos (default: /tmp/batchgen_deps)"
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
    WHEEL_DIR=""

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
    IS_HOPPER=0
    if [[ $SKIP_GPU_CHECK -eq 0 ]]; then
        if check_gpu_arch; then
            IS_HOPPER=1
        fi
    else
        IS_HOPPER=1
        print_warning "GPU check skipped, assuming Hopper architecture"
    fi

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
        else
            print_warning "Skipping Hopper-specific dependencies (non-Hopper GPU detected)"
            print_warning "Use --skip-gpu-check to force installation"
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
