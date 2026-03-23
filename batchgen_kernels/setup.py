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

_this_dir = os.path.dirname(os.path.abspath(__file__))
import torch

# Force CXX11 ABI to match PyTorch (same approach as flash-attention)
torch._C._GLIBCXX_USE_CXX11_ABI = True

# Parallel compilation: MAX_JOBS controls number of files compiled in parallel
if not os.environ.get("MAX_JOBS"):
    os.environ["MAX_JOBS"] = str(max(1, os.cpu_count() // 2))

# nvcc --threads controls parallelism within a single .cu file
_nvcc_threads = os.getenv("NVCC_THREADS", "4")

# ── Multi-architecture flag sets ──

_sm90a_flags = ["-std=c++17", "-arch=sm_90a", "-O3", "--ptxas-options=-v",
                "-lineinfo", "--threads", _nvcc_threads]

_sm80_gencode = ["-gencode", "arch=compute_80,code=sm_80",
                 "-gencode", "arch=compute_90,code=sm_90"]

# Add SM100 gencode if CUDA toolkit >= 12.8
_cuda_version = getattr(torch.version, "cuda", None)
if _cuda_version:
    _cuda_major, _cuda_minor = (int(x) for x in _cuda_version.split(".")[:2])
    if (_cuda_major, _cuda_minor) >= (12, 8):
        _sm80_gencode += ["-gencode", "arch=compute_100,code=sm_100"]

_sm80_flags = ["-std=c++17", "-O3", "--threads", _nvcc_threads] + _sm80_gencode

setup(
    name="batchgen_kernels",
    version="0.1.0",
    description="Pre-compiled CUDA kernels for BatchGen inference",
    package_dir={
        "batchgen_kernels": ".",
        "batchgen_kernels.attention": "attention",
        "batchgen_kernels.moe": "moe",
        "batchgen_kernels.common": "common",
        "batchgen_kernels.triton": "triton",
    },
    packages=["batchgen_kernels", "batchgen_kernels.attention",
              "batchgen_kernels.moe", "batchgen_kernels.common",
              "batchgen_kernels.triton"],
    package_data={"batchgen_kernels.attention": ["_C_gqa_mha_decode_bf16*.so"]},
    ext_modules=[
        # ── SM90a WGMMA kernels ──

        # MoE WGMMA kernels (MXFP4)
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
        # Grouped INT4 WGMMA for K2.5 decode
        CUDAExtension(
            name="batchgen_kernels.moe._C_grouped_int4_wgmma",
            sources=["src/moe/grouped_int4_wgmma_ext.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": _sm90a_flags},
        ),
        # Single-expert INT4 WGMMA
        CUDAExtension(
            name="batchgen_kernels.moe._C_single_expert_int4_wgmma",
            sources=["src/moe/single_expert_int4_wgmma.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": _sm90a_flags},
        ),
        # Fused INT4 WGMMA grouped (K2.5 TMA-based, gate+up+SiLU + down)
        CUDAExtension(
            name="batchgen_kernels.moe._C_fused_int4_wgmma_grouped",
            sources=["src/moe/fused_int4_wgmma_grouped.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": _sm90a_flags},
        ),
        # Marlin W4A16 grouped GEMM (gate+up+SiLU fused + down)
        CUDAExtension(
            name="batchgen_kernels.moe._C_marlin_grouped_gemm",
            sources=["src/moe/marlin_grouped_gemm.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-std=c++17", "-arch=sm_90a",
                         "--use_fast_math", "-lineinfo",
                         "-DUSE_BF16_COMPUTE",
                         "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                         "--threads", _nvcc_threads],
            },
        ),
        # Marlin <-> WGMMA weight transform
        CUDAExtension(
            name="batchgen_kernels.moe._C_marlin_transform",
            sources=["src/moe/marlin_transform_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-std=c++17", "-arch=sm_90a",
                         "--use_fast_math",
                         "--threads", _nvcc_threads],
            },
        ),
        # QKV WGMMA fused projection
        CUDAExtension(
            name="batchgen_kernels.attention._C_qkv_wgmma",
            sources=["src/attention/qkv_wgmma.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": _sm90a_flags},
        ),
        # Routing kernels (fused_gate uses WGMMA)
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
        # 3D dispatch scatter + reduce (strided MoE buffer)
        CUDAExtension(
            name="batchgen_kernels.moe._C_dispatch_scatter_3d",
            sources=["src/moe/dispatch_scatter_3d.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-std=c++17",
                         "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                         "--threads", _nvcc_threads],
            },
        ),

        # ── SM80+ universal kernels ──

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
                         "--threads", _nvcc_threads] + _sm80_gencode,
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
        # RMSNorm (multi-dtype: BF16/FP16/FP32) — common
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
        # CUDA RMSNorm + Add+RMSNorm (from cuda_rmsnorm.py)
        CUDAExtension(
            name="batchgen_kernels.common._C_cuda_rmsnorm",
            sources=["src/common/cuda_rmsnorm.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-std=c++17",
                         "-U__CUDA_NO_HALF_OPERATORS__",
                         "-U__CUDA_NO_HALF_CONVERSIONS__",
                         "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                         "--expt-relaxed-constexpr",
                         "--threads", _nvcc_threads],
            },
        ),
        # MGN (MoE General Native) ops — token dispatch, fused gate, bincount, rmsnorm
        CUDAExtension(
            name="batchgen_kernels.common._C_mgn_ops",
            sources=[
                "src/moe/mgn/mgn_extension.cc",
                "src/moe/mgn/fused_moe_token_dispatch.cu",
                "src/moe/mgn/moe_fused_gate.cu",
                "src/moe/mgn/expert_bin_count.cu",
                "src/moe/mgn/rmsnorm.cu",
            ],
            include_dirs=[os.path.join(_this_dir, "src/moe/mgn"),
                         os.path.join(_this_dir, "3rd/cutlass/include")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-std=c++17",
                         "-U__CUDA_NO_HALF_OPERATORS__",
                         "-U__CUDA_NO_HALF_CONVERSIONS__",
                         "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                         "--threads", _nvcc_threads] + _sm80_gencode,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
    # torch must be pre-installed (with correct CUDA variant, e.g. cu128).
    # Do NOT list it here — pip would pull the CPU-only version from PyPI.
    install_requires=[],
)
