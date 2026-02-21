"""Install batchgen_kernels — pre-compiled CUDA kernels for BatchGen.

The attention decode kernel ships as a pre-compiled .so (SM90a).
All other kernels are compiled from source at install time.

IMPORTANT: Must use --no-build-isolation to compile against the
installed PyTorch (build isolation installs a different torch version
whose headers may not match the runtime library).

Usage:
    pip install -e batchgen_kernels/ --no-build-isolation
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# Force CXX11 ABI to match PyTorch (same approach as flash-attention)
torch._C._GLIBCXX_USE_CXX11_ABI = True

# Parallel compilation: MAX_JOBS controls number of files compiled in parallel
if not os.environ.get("MAX_JOBS"):
    os.environ["MAX_JOBS"] = str(max(1, os.cpu_count() // 2))

# nvcc --threads controls parallelism within a single .cu file
_nvcc_threads = os.getenv("NVCC_THREADS", "4")

_sm90a_flags = ["-std=c++17", "-arch=sm_90a", "-O3", "--ptxas-options=-v",
                "-lineinfo", "--threads", _nvcc_threads]

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
    package_data={"batchgen_kernels.attention": ["_C_gqa_mha_decode_bf16*.so"]},
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
                         "-gencode", "arch=compute_90a,code=sm_90a",
                         "--threads", _nvcc_threads],
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
                         "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                         "--threads", _nvcc_threads],
            },
        ),
        # CuTe MXFP4 dequantization
        CUDAExtension(
            name="batchgen_kernels.moe._C_mxfp4_dequant_cute",
            sources=["src/moe/mxfp4_dequant_cute.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo",
                         "--threads", _nvcc_threads],
            },
        ),
        # QKV WGMMA fused projection (SM90a)
        CUDAExtension(
            name="batchgen_kernels.attention._C_qkv_wgmma",
            sources=["src/attention/qkv_wgmma.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": _sm90a_flags},
        ),
        # MXFP4 dequant with shared memory LUT
        CUDAExtension(
            name="batchgen_kernels.moe._C_mxfp4_dequant",
            sources=["src/moe/mxfp4_dequant.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo",
                         "--threads", _nvcc_threads],
            },
        ),
        # RMSNorm (multi-dtype: BF16/FP16/FP32)
        CUDAExtension(
            name="batchgen_kernels.common._C_rmsnorm",
            sources=["src/common/rmsnorm.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-std=c++17",
                         "-U__CUDA_NO_HALF_OPERATORS__",
                         "-U__CUDA_NO_HALF_CONVERSIONS__",
                         "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                         "--expt-relaxed-constexpr",
                         "--threads", _nvcc_threads],
            },
        ),
        # Grouped INT4 WGMMA for K2.5 decode (SM90a)
        CUDAExtension(
            name="batchgen_kernels.moe._C_grouped_int4_wgmma",
            sources=["src/moe/grouped_int4_wgmma_ext.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": _sm90a_flags},
        ),
        # Single-expert INT4 WGMMA (SM90a)
        CUDAExtension(
            name="batchgen_kernels.moe._C_single_expert_int4_wgmma",
            sources=["src/moe/single_expert_int4_wgmma.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": _sm90a_flags},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
    install_requires=["torch>=2.9.0"],
)
