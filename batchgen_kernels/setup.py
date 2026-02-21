"""Install batchgen_kernels — pre-compiled CUDA kernels for BatchGen.

The attention decode kernel ships as a pre-compiled .so (SM90a).
Other kernels (MoE, routing, etc.) will be added as source in future.

Usage:
    pip install -e batchgen_kernels/
"""

from setuptools import setup

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
    python_requires=">=3.10",
    install_requires=["torch>=2.9.0"],
)
