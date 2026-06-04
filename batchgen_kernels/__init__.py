"""batchgen_kernels — Pre-compiled CUDA kernels for BatchGen inference.

Provides attention decode, MoE WGMMA, routing, and utility kernels.
Requires SM90+ GPU (H100/H20/H800) or SM100+ (Blackwell).

Usage:
    from batchgen_kernels.attention import attention_decode_bf16
    from batchgen_kernels.moe.grouped_mxfp4 import grouped_mxfp4_stage1_swiglu
"""

from batchgen_kernels._version import (
    __version__,
    __version_full__,
    version_info,
)

import os
import importlib
import logging

import torch

logger = logging.getLogger(__name__)

_DEV_MODE = os.environ.get("BATCHGEN_KERNELS_DEV", "0") == "1"


def load_extension(module_name: str):
    """Import a pre-compiled CUDA extension by module name.

    Extensions are compiled at pip install time via CUDAExtension with
    torch/python.h + CXX11 ABI, so standard import works directly.

    With BATCHGEN_KERNELS_DEV=1, falls back to JIT compilation from source
    if the AOT module import fails.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        if not _DEV_MODE:
            raise

    logger.warning(
        f"[DEV] AOT import failed for {module_name}, attempting JIT..."
    )
    return _jit_compile(module_name)


def _jit_compile(module_name: str):
    """JIT compile a CUDA extension from source (dev mode only)."""
    from torch.utils.cpp_extension import load as jit_load
    from batchgen_kernels._jit_registry import get_registry

    registry = get_registry()
    if module_name not in registry:
        raise ImportError(
            f"No JIT config for {module_name}. "
            f"Available: {list(registry.keys())}"
        )

    cfg = registry[module_name]
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    sources = [os.path.join(pkg_dir, s) for s in cfg["sources"]]
    include_dirs = [
        os.path.join(pkg_dir, d) for d in cfg.get("include_dirs", [])
    ]

    short_name = module_name.rsplit(".", 1)[-1]

    return jit_load(
        name=short_name,
        sources=sources,
        extra_cuda_cflags=cfg.get("nvcc_flags", []),
        extra_cflags=cfg.get("cxx_flags", ["-O3"]),
        extra_include_paths=include_dirs,
        verbose=True,
    )


def get_device_arch() -> str:
    """Detect GPU architecture for kernel selection."""
    if not torch.cuda.is_available():
        raise RuntimeError("batchgen_kernels requires CUDA")
    cc = torch.cuda.get_device_capability()
    if cc[0] == 12:
        return "sm120"
    elif cc[0] >= 10:
        return "sm100"
    elif cc[0] >= 9:
        return "sm90a"
    raise RuntimeError(
        f"batchgen_kernels requires SM90+ GPU, got compute capability {cc[0]}.{cc[1]}"
    )
