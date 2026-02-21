"""Install batchgen_kernels — pre-compiled CUDA kernels for BatchGen.

The attention decode kernel ships as a pre-compiled .so (SM90a).
All other kernels are compiled from source at install time.

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
        # MoE WGMMA kernels (SM90a)
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
        # Routing kernels (SM90a — uses WGMMA for fused gate)
        CUDAExtension(
            name="batchgen_kernels.moe._C_routing",
            sources=[
                "src/moe/routing/routing_extension.cc",
                "src/moe/routing/gate_topk_softmax.cu",
                "src/moe/routing/dispatch_count_gather.cu",
                "src/moe/routing/reduce_weighted_scatter.cu",
                "src/moe/routing/router_epilogue.cu",
                "src/moe/routing/gate_sigmoid_topk.cu",
                "src/moe/routing/fused_gate.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-std=c++17",
                         "-gencode", "arch=compute_90a,code=sm_90a"],
            },
        ),
        # Attention fused ops (RMSNorm, RoPE, QKV split)
        CUDAExtension(
            name="batchgen_kernels.attention._C_fused_ops",
            sources=[
                "src/attention/csrc/attention_extension.cc",
                "src/attention/csrc/rmsnorm.cu",
                "src/attention/csrc/rope.cu",
                "src/attention/csrc/qkv_split.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-std=c++17", "--expt-relaxed-constexpr",
                         "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"],
            },
        ),
        # CuTe MXFP4 dequantization
        CUDAExtension(
            name="batchgen_kernels.moe._C_mxfp4_dequant_cute",
            sources=["src/moe/mxfp4_dequant_cute.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
    install_requires=["torch>=2.9.0"],
)
