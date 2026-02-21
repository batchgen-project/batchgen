"""batchgen_kernels — Pre-compiled CUDA kernels for BatchGen inference.

Provides attention decode, MoE WGMMA, routing, and utility kernels.
Requires SM90+ GPU (H100/H20/H800) or SM100+ (Blackwell).

Usage:
    from batchgen_kernels.attention import attention_decode_bf16
    from batchgen_kernels.moe.grouped_mxfp4 import grouped_mxfp4_stage1_swiglu
"""

__version__ = "0.1.0"

import torch
from pathlib import Path


def get_src_dir() -> Path:
    """Return the path to batchgen_kernels/src/ for JIT compilation."""
    return Path(__file__).parent / "src"


def get_device_arch() -> str:
    """Detect GPU architecture for kernel selection."""
    if not torch.cuda.is_available():
        raise RuntimeError("batchgen_kernels requires CUDA")
    cc = torch.cuda.get_device_capability()
    if cc[0] >= 10:
        return "sm100"
    elif cc[0] >= 9:
        return "sm90a"
    raise RuntimeError(
        f"batchgen_kernels requires SM90+ GPU, got compute capability {cc[0]}.{cc[1]}"
    )
