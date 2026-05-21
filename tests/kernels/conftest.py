# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
# ---------------------------------------------------------------------------- #

import pytest
import torch

ATOL = 1e-5
RTOL = 1.6e-2
OUTLIER_THRESHOLD = 1e-4

FP8_ATOL = 0.05
FP8_RTOL = 0.05
FP8_OUTLIER_THRESHOLD = 1e-3


def _assert_bf16_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = ATOL,
    rtol: float = RTOL,
    outlier_threshold: float = OUTLIER_THRESHOLD,
    msg: str = "",
) -> None:
    """ATOL + RTOL * |expected| tolerance with outlier threshold (batchgen convention)."""
    diff = (actual.float() - expected.float()).abs()
    tol = atol + rtol * expected.float().abs()
    n_fail = int((diff > tol).sum().item())
    n_total = diff.numel()
    if n_fail and n_fail / n_total >= outlier_threshold:
        tag = f" [{msg}]" if msg else ""
        raise AssertionError(
            f"V4 kernel mismatch{tag}: "
            f"max_abs={float(diff.max().item()):.6g}  "
            f"failures={n_fail}/{n_total}"
        )


def _assert_fp8_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    msg: str = "",
) -> None:
    _assert_bf16_close(
        actual,
        expected,
        atol=FP8_ATOL,
        rtol=FP8_RTOL,
        outlier_threshold=FP8_OUTLIER_THRESHOLD,
        msg=msg,
    )


class disable_tf32:
    def __enter__(self):
        self._old_matmul = torch.backends.cuda.matmul.allow_tf32
        self._old_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return self

    def __exit__(self, *exc):
        torch.backends.cuda.matmul.allow_tf32 = self._old_matmul
        torch.backends.cudnn.allow_tf32 = self._old_cudnn


def _bench(fn, *args, warmup: int = 5, iters: int = 20) -> float:
    """Returns average kernel time in ms via CUDA events."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters
