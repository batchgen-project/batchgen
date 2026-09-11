"""JIT compilation registry for dev mode.

Maps module_name -> {sources, nvcc_flags, cxx_flags, include_dirs}
with paths relative to the batchgen_kernels package root.

This mirrors setup.py ext_modules. When adding a new CUDAExtension in
setup.py, add a corresponding entry here for JIT dev mode support.
"""

# Common flag sets (mirror setup.py)
_SM90A_FLAGS = [
    "-std=c++17", "-arch=sm_90a", "-O3",
    "--ptxas-options=-v", "-lineinfo", "--threads", "4",
]

_SM80_GENCODE = [
    "-gencode", "arch=compute_80,code=sm_80",
    "-gencode", "arch=compute_90,code=sm_90",
]

_SM80_FLAGS = ["-std=c++17", "-O3", "--threads", "4"] + _SM80_GENCODE


def get_registry():
    """Return JIT compilation config for all CUDA extensions."""
    return {
        # ── SM90a WGMMA kernels ──

        "batchgen_kernels.moe._C_expert_mxfp4_wgmma": {
            "sources": ["src/moe/expert_mxfp4_wgmma.cu"],
            "nvcc_flags": _SM90A_FLAGS,
        },
        "batchgen_kernels.moe._C_grouped_mxfp4_wgmma": {
            "sources": ["src/moe/grouped_mxfp4_wgmma.cu"],
            "nvcc_flags": _SM90A_FLAGS,
        },
        "batchgen_kernels.moe._C_grouped_int4_wgmma": {
            "sources": ["src/moe/grouped_int4_wgmma_ext.cu"],
            "nvcc_flags": _SM90A_FLAGS,
        },
        "batchgen_kernels.moe._C_single_expert_int4_wgmma": {
            "sources": ["src/moe/single_expert_int4_wgmma.cu"],
            "nvcc_flags": _SM90A_FLAGS,
        },
        "batchgen_kernels.moe._C_fused_int4_wgmma_grouped": {
            "sources": ["src/moe/fused_int4_wgmma_grouped.cu"],
            "nvcc_flags": _SM90A_FLAGS,
        },
        "batchgen_kernels.moe._C_marlin_grouped_gemm": {
            "sources": ["src/moe/marlin_grouped_gemm.cu"],
            "nvcc_flags": [
                "-O3", "-std=c++17", "-arch=sm_90a",
                "--use_fast_math", "-lineinfo",
                "-DUSE_BF16_COMPUTE",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--threads", "4",
            ],
        },
        "batchgen_kernels.moe._C_fp8_blockwise_gemm": {
            "sources": ["src/moe/fp8_blockwise/fp8_blockwise_gemm.cu"],
            "nvcc_flags": [
                "-O3", "-std=c++17", "-arch=sm_90a",
                "-lineinfo", "--expt-relaxed-constexpr",
                "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
                "-DNDEBUG", "-Xptxas=-v",
                "--threads", "4",
            ],
            "include_dirs": ["3rd/cutlass/include", "."],
        },
        "batchgen_kernels.moe._C_fp8_blockwise_ops": {
            "sources": ["src/moe/fp8_blockwise/fp8_blockwise_ops.cu"],
            "nvcc_flags": [
                "-O3", "-std=c++17", "-arch=sm_90a",
                "-lineinfo", "--threads", "4",
            ],
        },
        "batchgen_kernels.moe._C_marlin_transform": {
            "sources": ["src/moe/marlin_transform_kernel.cu"],
            "nvcc_flags": [
                "-O3", "-std=c++17", "-arch=sm_90a",
                "--use_fast_math", "--threads", "4",
            ],
        },
        "batchgen_kernels.attention._C_qkv_wgmma": {
            "sources": ["src/attention/qkv_wgmma.cu"],
            "nvcc_flags": _SM90A_FLAGS,
        },
        "batchgen_kernels.moe._C_routing": {
            "sources": [
                "src/moe/routing/routing_extension.cc",
                "src/moe/routing/gate_topk_softmax.cu",
                "src/moe/routing/dispatch_count_gather.cu",
                "src/moe/routing/reduce_weighted_scatter.cu",
                "src/moe/routing/router_epilogue.cu",
                "src/moe/routing/gate_sigmoid_topk.cu",
                "src/moe/routing/glm5_router_gemm.cu",
                "src/moe/routing/fused_gate.cu",
            ],
            "nvcc_flags": [
                "-O3", "--use_fast_math", "-std=c++17",
                "-gencode", "arch=compute_90a,code=sm_90a",
                "--threads", "4",
            ],
        },
        "batchgen_kernels.moe._C_dispatch_scatter_3d": {
            "sources": ["src/moe/dispatch_scatter_3d.cu"],
            "nvcc_flags": [
                "-O3", "-std=c++17",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--threads", "4",
            ],
        },

        # ── SM80+ universal kernels ──

        "batchgen_kernels.attention._C_fused_ops": {
            "sources": [
                "src/attention/csrc/attention_extension.cc",
                "src/attention/csrc/rmsnorm.cu",
                "src/attention/csrc/rope.cu",
                "src/attention/csrc/qkv_split.cu",
            ],
            "nvcc_flags": [
                "-O3", "-std=c++17", "--expt-relaxed-constexpr",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--threads", "4",
            ] + _SM80_GENCODE,
        },
        "batchgen_kernels.conv1d._C_causal_conv1d": {
            "sources": ["src/conv1d/causal_conv1d.cu"],
            "nvcc_flags": [
                "-O3", "-std=c++17", "--expt-relaxed-constexpr",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--threads", "4",
            ] + _SM80_GENCODE,
        },
        "batchgen_kernels.moe._C_mxfp4_dequant_cute": {
            "sources": ["src/moe/mxfp4_dequant_cute.cu"],
            "nvcc_flags": [
                "-O3", "--use_fast_math", "-lineinfo",
                "--threads", "4",
            ],
        },
        "batchgen_kernels.moe._C_mxfp4_dequant": {
            "sources": ["src/moe/mxfp4_dequant.cu"],
            "nvcc_flags": [
                "-O3", "--use_fast_math", "-lineinfo",
                "--threads", "4",
            ],
        },
        "batchgen_kernels.common._C_rmsnorm": {
            "sources": ["src/common/rmsnorm.cu"],
            "nvcc_flags": [
                "-O3", "--use_fast_math", "-std=c++17",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--threads", "4",
            ],
        },
        "batchgen_kernels.common._C_cuda_rmsnorm": {
            "sources": ["src/common/cuda_rmsnorm.cu"],
            "nvcc_flags": [
                "-O3", "-std=c++17",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--threads", "4",
            ],
        },
        "batchgen_kernels.common._C_mgn_ops": {
            "sources": [
                "src/moe/mgn/mgn_extension.cc",
                "src/moe/mgn/fused_moe_token_dispatch.cu",
                "src/moe/mgn/moe_fused_gate.cu",
                "src/moe/mgn/expert_bin_count.cu",
                "src/moe/mgn/rmsnorm.cu",
            ],
            "nvcc_flags": [
                "-O3", "-std=c++17",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--threads", "4",
            ] + _SM80_GENCODE,
            "include_dirs": ["src/moe/mgn", "3rd/cutlass/include"],
        },

        # ── AOT MLA attention kernels (SM90a) ──

        "batchgen_kernels.attention._C_fused_kv_norm_rope": {
            "sources": ["src/attention/fused_kv_norm_rope_cache.cu"],
            "nvcc_flags": [
                "-O3", "--use_fast_math", "-std=c++17",
                "-gencode", "arch=compute_90a,code=sm_90a",
                "--threads", "4",
            ],
        },
        "batchgen_kernels.attention._C_fused_q_absorb": {
            "sources": ["src/attention/fused_q_absorb.cu"],
            "nvcc_flags": [
                "-O3", "--use_fast_math", "-std=c++17",
                "-gencode", "arch=compute_90a,code=sm_90a",
                "--threads", "4",
            ],
        },
        "batchgen_kernels.attention._C_fused_q_split": {
            "sources": ["src/attention/fused_q_split.cu"],
            "nvcc_flags": [
                "-O3", "--use_fast_math", "-std=c++17",
                "-gencode", "arch=compute_90a,code=sm_90a",
                "--threads", "4",
            ],
        },
        "batchgen_kernels.attention._C_kda_fused_decode": {
            "sources": ["src/attention/kda_fused_decode.cu"],
            "nvcc_flags": [
                "-O3", "--use_fast_math", "-std=c++17",
                "-gencode", "arch=compute_90a,code=sm_90a",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--threads", "4",
            ],
        },

        # ── AOT MoE token permutation (SM80+) ──

        "batchgen_kernels.moe._C_fused_moe_token_permutation": {
            "sources": ["src/moe/fused_moe_token_permutation.cu"],
            "nvcc_flags": [
                "-O3", "--use_fast_math", "-std=c++17",
                "--threads", "4",
            ] + _SM80_GENCODE,
        },
    }
