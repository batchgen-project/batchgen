"""batchgen_kernels version — single source of truth.

setup.py reads this via exec() at build time (no torch/CUDA import needed).
__init__.py re-exports __version__ and version_info at runtime.

BUILD_ARCH env var controls the arch suffix in wheel names. setup.py detects
the visible CUDA device when BUILD_ARCH is unset and then writes the resolved
value back to the environment before reading this file:
  "sm90a" -> batchgen_kernels-0.3.1.post3+sm90a
  "sm100" -> batchgen_kernels-0.3.1.post3+sm100
  "all"   -> batchgen_kernels-0.3.1.post3 (no suffix)
"""

import os

__version__ = "0.3.1.post3"
version_info = (0, 3, 1, "post", 3)

# Arch suffix for wheel naming (PEP 440 local version)
_build_arch = os.environ.get("BUILD_ARCH", "sm90a")
__version_full__ = __version__ if _build_arch == "all" else f"{__version__}+{_build_arch}"
