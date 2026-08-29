"""Install batchgen_kernels — pre-compiled CUDA kernels for BatchGen.

The attention decode kernel ships as a pre-compiled .so (SM90a).
All other kernels are compiled from source at install time.

IMPORTANT: Must use --no-build-isolation to compile against the
installed PyTorch (build isolation installs a different torch version
whose headers may not match the runtime library).

Usage:
    pip install -e batchgen_kernels/ --no-build-isolation

Environment variables:
    MAX_JOBS        — parallel file compilation (default: cpu_count/2)
    NVCC_THREADS    — parallelism within a single .cu file (default: 4)
    BUILD_ARCH      — "sm90a" (default), "sm100", or "all". Controls which arch kernels to build
    BATCHGEN_KERNELS_DEV — "1" enables JIT fallback at runtime (not build-time)
"""

import os
import shutil
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_this_dir = os.path.dirname(os.path.abspath(__file__))
import torch


# ── Version (single source of truth: _version.py) ──

def _get_version():
    version_file = os.path.join(_this_dir, "_version.py")
    ns = {"__file__": version_file}
    with open(version_file) as f:
        exec(f.read(), ns)
    return ns["__version_full__"]


# Force CXX11 ABI to match PyTorch (same approach as flash-attention)
torch._C._GLIBCXX_USE_CXX11_ABI = True

# Parallel compilation: MAX_JOBS controls number of files compiled in parallel
if not os.environ.get("MAX_JOBS"):
    os.environ["MAX_JOBS"] = str(max(1, os.cpu_count() // 2))

# nvcc --threads controls parallelism within a single .cu file
_nvcc_threads = os.getenv("NVCC_THREADS", "4")


# ── ccache / sccache integration ──

def _setup_ccache():
    """Detect and configure ccache/sccache for faster incremental builds."""
    for tool in ("sccache", "ccache"):
        if shutil.which(tool):
            os.environ.setdefault("CC", f"{tool} gcc")
            os.environ.setdefault("CXX", f"{tool} g++")
            if tool == "ccache":
                os.environ.setdefault("CCACHE_NVCC", "1")
            print(f"[batchgen_kernels] Using {tool} for compilation cache")
            return tool
    return None

_cache_tool = _setup_ccache()


# ── CUDA_HOME auto-detection (for external users) ──

if not os.environ.get("CUDA_HOME"):
    _nvcc_path = shutil.which("nvcc")
    if _nvcc_path:
        os.environ["CUDA_HOME"] = os.path.dirname(os.path.dirname(_nvcc_path))
    else:
        raise RuntimeError(
            "CUDA toolkit not found. Set CUDA_HOME or ensure nvcc is on PATH.\n"
            "Example: export CUDA_HOME=/usr/local/cuda-12.8"
        )


# ── Architecture build gating ──
# BUILD_ARCH: "sm90a" (default), "sm100", "all"

_build_arch = os.environ.get("BUILD_ARCH", "sm90a")
_build_sm90a = _build_arch in ("sm90a", "all")
_build_sm100 = _build_arch in ("sm100", "all")

# ── Architecture flag sets ──

# Explicit -gencode (not bare -arch=sm_90a): bare -arch makes torch cpp_extension
# append its own arch flags (GPU-detection/TORCH_CUDA_ARCH_LIST based), which crashes
# on no-GPU hosts and adds a plain compute_90 pass that ptxas rejects for wgmma.*.
_sm90a_flags = ["-std=c++17", "-gencode", "arch=compute_90a,code=sm_90a", "-O3", "--ptxas-options=-v",
                "-lineinfo", "--threads", _nvcc_threads]

if _build_arch == "sm90a":
    _sm80_gencode = ["-gencode", "arch=compute_90a,code=sm_90a"]
elif _build_arch == "sm100":
    _sm80_gencode = ["-gencode", "arch=compute_100,code=sm_100"]
elif _build_arch == "all":
    _sm80_gencode = [
        "-gencode", "arch=compute_80,code=sm_80",
        "-gencode", "arch=compute_90,code=sm_90",
    ]
    # Add SM100 gencode if CUDA toolkit >= 12.8
    _cuda_version = getattr(torch.version, "cuda", None)
    if _cuda_version:
        _cuda_major, _cuda_minor = (int(x) for x in _cuda_version.split(".")[:2])
        if (_cuda_major, _cuda_minor) >= (12, 8):
            _sm80_gencode += ["-gencode", "arch=compute_100,code=sm_100"]
else:
    raise RuntimeError(
        f"Unsupported BUILD_ARCH={_build_arch!r}; expected sm90a, sm100, or all"
    )

_sm80_flags = ["-std=c++17", "-O3", "--threads", _nvcc_threads] + _sm80_gencode

# ── Build extension list ──

_sm90a_extensions = [
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
            "nvcc": ["-O3", "-std=c++17", "-gencode", "arch=compute_90a,code=sm_90a",
                     "--use_fast_math", "-lineinfo",
                     "-DUSE_BF16_COMPUTE",
                     "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                     "--threads", _nvcc_threads],
        },
    ),
    # FP8 blockwise grouped GEMM (CuTe persistent, adaptive TileM)
    CUDAExtension(
        name="batchgen_kernels.moe._C_fp8_blockwise_gemm",
        sources=["src/moe/fp8_blockwise/fp8_blockwise_gemm.cu"],
        include_dirs=[os.path.join(_this_dir, "3rd/cutlass/include"),
                     _this_dir],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3", "-std=c++17", "-gencode", "arch=compute_90a,code=sm_90a",
                     "-lineinfo", "--expt-relaxed-constexpr",
                     "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
                     "-DNDEBUG",
                     "-Xptxas=-v",
                     "--threads", _nvcc_threads],
        },
    ),
    # FP8 blockwise MoE pipeline ops (act_quant_3d, silu_mul_3d, fused_silu_quant_3d)
    CUDAExtension(
        name="batchgen_kernels.moe._C_fp8_blockwise_ops",
        sources=["src/moe/fp8_blockwise/fp8_blockwise_ops.cu"],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3", "-std=c++17", "-gencode", "arch=compute_90a,code=sm_90a",
                     "-lineinfo",
                     "--threads", _nvcc_threads],
        },
    ),
    # Marlin <-> WGMMA weight transform
    CUDAExtension(
        name="batchgen_kernels.moe._C_marlin_transform",
        sources=["src/moe/marlin_transform_kernel.cu"],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3", "-std=c++17", "-gencode", "arch=compute_90a,code=sm_90a",
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
            "src/moe/routing/glm5_router_gemm.cu",
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
                     "--threads", _nvcc_threads] + _sm80_gencode,
        },
    ),
    # ── AOT MLA attention kernels (SM90a, BF16-only) ──

    # Fused RMSNorm + RoPE + cache write (KV + Q)
    CUDAExtension(
        name="batchgen_kernels.attention._C_fused_kv_norm_rope",
        sources=["src/attention/fused_kv_norm_rope_cache.cu"],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3", "--use_fast_math", "-std=c++17",
                     "-gencode", "arch=compute_90a,code=sm_90a",
                     "--threads", _nvcc_threads],
        },
    ),
    # Fused q_absorb GEMV + q_pe copy
    CUDAExtension(
        name="batchgen_kernels.attention._C_fused_q_absorb",
        sources=["src/attention/fused_q_absorb.cu"],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3", "--use_fast_math", "-std=c++17",
                     "-gencode", "arch=compute_90a,code=sm_90a",
                     "--threads", _nvcc_threads],
        },
    ),
    # Fused q_b split into q_nope + q_pe
    CUDAExtension(
        name="batchgen_kernels.attention._C_fused_q_split",
        sources=["src/attention/fused_q_split.cu"],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3", "--use_fast_math", "-std=c++17",
                     "-gencode", "arch=compute_90a,code=sm_90a",
                     "--threads", _nvcc_threads],
        },
    ),
    # Fused K3 KDA decode: short-conv update + recurrent delta rule + gated
    # RMSNorm.  The kernel is K3/SM90a-specific and is AOT-only at runtime.
    CUDAExtension(
        name="batchgen_kernels.attention._C_kda_fused_decode",
        sources=["src/attention/kda_fused_decode.cu"],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3", "--use_fast_math", "-std=c++17",
                     "-gencode", "arch=compute_90a,code=sm_90a",
                     "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                     "--threads", _nvcc_threads],
        },
    ),
]

_sm80_extensions = [
    # ── SM80+ universal kernels ──

    # Causal conv1d (KDA / mamba): varlen prefill + pooled-state decode update
    CUDAExtension(
        name="batchgen_kernels.conv1d._C_causal_conv1d",
        sources=["src/conv1d/causal_conv1d.cu"],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3", "-std=c++17", "--expt-relaxed-constexpr",
                     "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                     "--threads", _nvcc_threads] + _sm80_gencode,
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
                     "--threads", _nvcc_threads] + _sm80_gencode,
        },
    ),
    # MXFP4 dequant with shared memory LUT
    CUDAExtension(
        name="batchgen_kernels.moe._C_mxfp4_dequant",
        sources=["src/moe/mxfp4_dequant.cu"],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3", "--use_fast_math", "-lineinfo",
                     "--threads", _nvcc_threads] + _sm80_gencode,
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
                     "--threads", _nvcc_threads] + _sm80_gencode,
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
                     "--threads", _nvcc_threads] + _sm80_gencode,
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

    # ── AOT MoE token permutation (SM80+, multi-dtype) ──

    CUDAExtension(
        name="batchgen_kernels.moe._C_fused_moe_token_permutation",
        sources=["src/moe/fused_moe_token_permutation.cu"],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3", "--use_fast_math", "-std=c++17",
                     "--threads", _nvcc_threads] + _sm80_gencode,
        },
    ),
]

# Assemble final extension list based on build flags
_ext_modules = []
if _build_sm90a:
    _ext_modules.extend(_sm90a_extensions)
else:
    print(f"[batchgen_kernels] BUILD_ARCH={_build_arch}: skipping SM90a-only kernels")
_ext_modules.extend(_sm80_extensions)

setup(
    name="batchgen_kernels",
    version=_get_version(),
    description="Pre-compiled CUDA kernels for BatchGen inference",
    package_dir={
        "batchgen_kernels": ".",
        "batchgen_kernels.attention": "attention",
        "batchgen_kernels.attention.dsa": "attention/dsa",
        "batchgen_kernels.attention.dsa.indexer": "attention/dsa/indexer",
        "batchgen_kernels.moe": "moe",
        "batchgen_kernels.common": "common",
        "batchgen_kernels.conv1d": "conv1d",
        "batchgen_kernels.triton": "triton",
    },
    packages=["batchgen_kernels", "batchgen_kernels.attention",
              "batchgen_kernels.attention.dsa",
              "batchgen_kernels.attention.dsa.indexer",
              "batchgen_kernels.moe", "batchgen_kernels.common",
              "batchgen_kernels.conv1d", "batchgen_kernels.triton"],
    package_data={
        "batchgen_kernels.attention": ["_C_gqa_mha_decode_bf16*.so"],
        "batchgen_kernels.attention.dsa.indexer": [
            "csrc/*.cpp",
            "csrc/*.cu",
            "csrc/*.h",
        ],
    },
    ext_modules=_ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
    # torch must be pre-installed (with correct CUDA variant, e.g. cu128).
    # Do NOT list it here — pip would pull the CPU-only version from PyPI.
    install_requires=[],
)
