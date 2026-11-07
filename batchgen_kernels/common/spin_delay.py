"""Python entry point for the run-time-duration spin delay CUDA kernel.

The kernel lives in ``src/common/spin_delay.cu`` (a NEW kernel; nothing is
replaced). A captured CUDA graph bakes the delay table's pointer and the site
index, and the kernel loads the duration when it runs, so an async copy into
the table between two replays changes the delay without a recapture. See the
``.cu`` header for why ``torch.cuda._sleep`` cannot serve this purpose.
"""

import torch

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        import batchgen_kernels
        _ext = batchgen_kernels.load_extension(
            "batchgen_kernels.common._C_spin_delay"
        )
    return _ext


def preload_spin_delay_kernel():
    """Resolve the extension before capture, so no JIT happens inside a graph."""
    return _get_ext()


def spin_delay(delay_ns: torch.Tensor, site: int) -> None:
    """Spin on the current stream for ``delay_ns[site]`` nanoseconds.

    ``delay_ns`` is a contiguous 1-D int64 CUDA tensor; the entry is read on
    the device at kernel run time, and a value ``<= 0`` returns at once.
    ``site`` is a host int fixed at launch (and therefore at capture).
    """
    _get_ext().spin_delay(delay_ns, int(site))
