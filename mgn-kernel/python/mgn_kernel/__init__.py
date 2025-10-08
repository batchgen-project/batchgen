import ctypes
import os
import platform

import torch

SYSTEM_ARCH = platform.machine()

cuda_path = f"/usr/local/cuda/targets/{SYSTEM_ARCH}-linux/lib/libcudart.so.12"
if os.path.exists(cuda_path):
    ctypes.CDLL(cuda_path, mode=ctypes.RTLD_GLOBAL)

from . import common_ops
from .moe import (
    moe_fused_gate,
    expert_bincount,
    fused_moe_token_dispatch,
)

from .elementwise import fused_rmsnorm

build_tree_kernel = (
    None
)
