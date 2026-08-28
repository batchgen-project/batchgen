"""batchgen_kernels version — single source of truth.

setup.py reads this via exec() at build time (no torch/CUDA import needed).
__init__.py re-exports __version__ and version_info at runtime.

BUILD_ARCH env var controls the arch suffix in wheel names:
  "sm90a" (default) -> batchgen_kernels-0.4.0+sm90a
  "sm100"           -> batchgen_kernels-0.4.0+sm100
  "all"             -> batchgen_kernels-0.4.0 (no suffix)
"""

import os

__version__ = "0.4.0"
version_info = (0, 4, 0)

# Arch suffix for wheel naming (PEP 440 local version)
_build_arch = os.environ.get("BUILD_ARCH", "sm90a")
__version_full__ = __version__ if _build_arch == "all" else f"{__version__}+{_build_arch}"
