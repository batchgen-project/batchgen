"""GPU architecture detection.

Kept in its own lightweight module so callers (tests, planners, CLI bridges)
can import :func:`detect_gpu_arch` without paying for the heavy
``batchgen.server.worker_manager`` import chain (the worker module pulls in
``batchgen.batchgen_worker``, the FastAPI server, NCCL, the native engine
loader, etc.). ``worker_manager`` re-exports the symbol for backward
compatibility.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def detect_gpu_arch() -> str:
    """Auto-detect GPU architecture based on CUDA compute capability.

    Returns:
        'blackwell' for compute capability 10.x (B200, GB200, etc.)
        'hopper' for compute capability 9.x (H100, H20, H200, etc.)
        'ampere' for compute capability 8.x (A100, A5000, RTX 4090, etc.)

    Raises:
        RuntimeError: If no CUDA devices found or unsupported architecture.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA devices available for GPU architecture detection"
        )

    major, minor = torch.cuda.get_device_capability(0)
    device_name = torch.cuda.get_device_name(0)

    if major == 10:
        arch = "blackwell"
    elif major == 9:
        arch = "hopper"
    elif major == 8:
        arch = "ampere"
    else:
        raise RuntimeError(
            f"Unsupported GPU architecture: compute capability {major}.{minor} "
            f"({device_name}). BatchGen requires Blackwell (sm_100), "
            f"Hopper (sm_90), or Ampere (sm_80)."
        )

    logger.info(
        "Auto-detected GPU architecture: %s (compute capability %d.%d, %s)",
        arch, major, minor, device_name,
    )
    return arch
