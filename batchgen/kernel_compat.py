"""Runtime compatibility check for batchgen_kernels.

batchgen_kernels is not on PyPI, so pip cannot enforce version constraints.
This module checks the installed version at import time and raises a clear
error if the kernel package is too old.
"""

import logging

logger = logging.getLogger(__name__)

# Minimum batchgen_kernels version required by this batchgen release.
# Bump this when batchgen starts using new kernel APIs.
MIN_KERNELS_VERSION = (0, 4, 5)


def check_kernels_version():
    """Check batchgen_kernels version compatibility. Called at import batchgen."""
    try:
        import batchgen_kernels
    except ImportError as exc:
        raise RuntimeError(
            "batchgen_kernels is required; install the matching AOT wheel before "
            "starting BatchGen"
        ) from exc

    if not hasattr(batchgen_kernels, "version_info"):
        raise RuntimeError(
            "batchgen_kernels has no version_info attribute; refusing an "
            "unverifiable/source-only kernel package"
        )

    installed = batchgen_kernels.version_info
    if installed < MIN_KERNELS_VERSION:
        min_str = ".".join(str(x) for x in MIN_KERNELS_VERSION)
        cur_str = batchgen_kernels.__version__
        raise RuntimeError(
            f"batchgen requires batchgen_kernels >= {min_str}, "
            f"found {cur_str}. Rebuild: pip install -e batchgen_kernels/ --no-build-isolation"
        )
