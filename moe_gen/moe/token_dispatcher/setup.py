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
    name='fused_moe_dispatch',
    version='0.1.0',
    description='Fused MoE Token Dispatch for efficient expert routing',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=[
        'torch>=1.12.0',
    ],
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)