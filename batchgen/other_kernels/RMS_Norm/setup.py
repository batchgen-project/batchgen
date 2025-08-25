# setup.py
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os

# Check if CUDA is available
def cuda_is_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

# CUDA extension setup
def get_cuda_extension():
    if not cuda_is_available():
        raise RuntimeError("CUDA is not available. Please install PyTorch with CUDA support.")
    
    # Get CUDA compute capabilities
    cuda_flags = [
        '-O3',
        '-std=c++17',
        '--use_fast_math',
        '-U__CUDA_NO_HALF_OPERATORS__',
        '-U__CUDA_NO_HALF_CONVERSIONS__',
        '-U__CUDA_NO_HALF2_OPERATORS__',
        '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
        '--expt-relaxed-constexpr',
        '--expt-extended-lambda',
    ]
    
    # Add architecture-specific flags
    # Support for major GPU architectures
    arch_flags = [
        '-gencode=arch=compute_70,code=sm_70',  # V100
        '-gencode=arch=compute_75,code=sm_75',  # T4, RTX 20xx
        '-gencode=arch=compute_80,code=sm_80',  # A100
        '-gencode=arch=compute_86,code=sm_86',  # RTX 30xx
        '-gencode=arch=compute_89,code=sm_89',  # RTX 40xx
        '-gencode=arch=compute_90,code=sm_90',  # H100
    ]
    
    cuda_flags.extend(arch_flags)
    
    # Include directories
    include_dirs = [
        # CUB library (usually included with CUDA toolkit)
        os.path.join(torch.utils.cpp_extension.CUDA_HOME, 'include'),
    ]
    
    extension = CUDAExtension(
        name='fused_rmsnorm_cuda',
        sources=[
            'fused_rmsnorm_kernel.cu',
        ],
        include_dirs=include_dirs,
        extra_compile_args={
            'cxx': ['-O3', '-std=c++17'],
            'nvcc': cuda_flags
        },
        libraries=['cublas', 'curand'],
        verbose=True
    )
    
    return extension

# Read requirements
def read_requirements():
    requirements = []
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return requirements

# Read long description
def read_long_description():
    if os.path.exists('README.md'):
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    return "Fused CUDA implementation of RMSNorm for efficient deep learning"

# Main setup
def main():
    ext_modules = []
    cmdclass = {}
    
    # Only build CUDA extension if CUDA is available
    if cuda_is_available():
        try:
            ext_modules = [get_cuda_extension()]
            cmdclass = {'build_ext': BuildExtension}
            print("CUDA extension will be built")
        except Exception as e:
            print(f"Warning: Could not build CUDA extension: {e}")
            print("Installing without CUDA acceleration")
    else:
        print("CUDA not available. Installing without CUDA acceleration")
    
    setup(
        name='fused-rmsnorm'
    )

if __name__ == '__main__':
    main()