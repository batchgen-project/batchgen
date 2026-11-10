"""Back-compat shim: detect_gpu_arch moved to batchgen.config.config so the
scheduler and worker can share it without importing the server package."""

from batchgen.config.config import detect_gpu_arch

__all__ = ["detect_gpu_arch"]
