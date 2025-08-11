from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# Check if CUDA is available
if torch.cuda.is_available():
    ext_modules = [
        CUDAExtension(
            name='fused_moe_cuda',
            sources=[
                'fused_moe_dispatch.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-Xptxas=-O3',
                    '-Xcompiler=-O3',
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',  # For bfloat16 support
                ]
            }
        )
    ]
    cmdclass = {'build_ext': BuildExtension}
else:
    print("CUDA not available, installing CPU-only version")
    ext_modules = []
    cmdclass = {}

setup(
    name='fused_moe_permutation'
)