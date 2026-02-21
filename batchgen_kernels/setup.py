"""Install batchgen_kernels — pre-compiled CUDA kernels for BatchGen.

The attention decode kernel ships as a pre-compiled .so (SM90a).
MoE WGMMA kernels are compiled from source at install time.

Usage:
    pip install -e batchgen_kernels/
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_sm90a_flags = ["-std=c++17", "-arch=sm_90a", "-O3", "--ptxas-options=-v", "-lineinfo"]

setup(
    name="batchgen_kernels",
    version="0.1.0",
    description="Pre-compiled CUDA kernels for BatchGen inference",
    package_dir={
        "batchgen_kernels": ".",
        "batchgen_kernels.attention": "attention",
        "batchgen_kernels.moe": "moe",
        "batchgen_kernels.common": "common",
    },
    packages=["batchgen_kernels", "batchgen_kernels.attention",
              "batchgen_kernels.moe", "batchgen_kernels.common"],
    package_data={"batchgen_kernels.attention": ["_C_gqa_mha_decode*.so"]},
    ext_modules=[
        CUDAExtension(
            name="batchgen_kernels.moe._C_expert_mxfp4_wgmma",
            sources=["src/moe/expert_mxfp4_wgmma.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": _sm90a_flags},
        ),
        CUDAExtension(
            name="batchgen_kernels.moe._C_grouped_mxfp4_wgmma",
            sources=["src/moe/grouped_mxfp4_wgmma.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": _sm90a_flags},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
    install_requires=["torch>=2.9.0"],
)
