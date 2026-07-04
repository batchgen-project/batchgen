"""MoE kernels: WGMMA grouped/expert GEMM, routing, dequantization."""

from .mega_moe_sm120 import (
    is_mega_moe_sm120_available,
    mega_moe_sm120_forward,
)

__all__ = ["is_mega_moe_sm120_available", "mega_moe_sm120_forward"]
