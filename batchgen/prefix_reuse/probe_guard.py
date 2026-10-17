"""One failure contract for every pooled-prefill diagnostic probe.

Observability must not take down what it observes. The epilogue trace, the
offload byte-equality check and the kernel dump all read tensors, resolve Host
page tables and write files from inside a live serving forward, so each of them
has shapes, APIs and filesystems it can be wrong about. Letting any of those
raise turns a diagnostic into an outage, and - worse on a multi-rank path - a
probe that raises between a rank's local work and a collective leaves the other
ranks waiting in the collective forever.

So every probe entry point is wrapped: a diagnostics failure logs ONE
structured ``probe_error`` WARNING naming the probe, the method and the
exception, and the caller gets a neutral value back. The line is greppable and
counted, so "the probe was armed and silent" can never be confused with "the
probe died".

Strict mode is the inverse contract and exists for the TESTS: a unit test that
pins an input the probe must refuse needs the exception, not a warning. It is
opt-in per probe object (``strict=True``) and is never set on a serving path.
"""

from __future__ import annotations

import functools
import logging

PREFIX = "PROBE_ERROR"


def probe_error(probe: str, exc: BaseException, *, rank=None, logger=None) -> str:
    """Log and return the one structured line a failed probe emits.

    ``probe`` is ``<module>.<method>`` so the line localizes the failure
    without a traceback; the traceback itself goes to DEBUG, where a developer
    chasing the probe can turn it on without widening a serving log.
    """

    line = (
        f"[{PREFIX}] probe={probe} rank={'na' if rank is None else rank} "
        f"error={type(exc).__name__}: {exc}"
    )
    log = logging if logger is None else logger
    log.warning(line)
    log.debug("[%s] probe=%s traceback", PREFIX, probe, exc_info=exc)
    return line


def probe_safe(probe: str, *, default=None):
    """Make a probe METHOD fail-safe, unless its object asked to be strict.

    ``default`` is either the neutral value to return or a zero-argument
    factory for it, so a method whose contract is "returns a list" never hands
    a shared mutable back to two callers.
    """

    def decorate(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            try:
                return method(self, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - that is the contract
                if getattr(self, "strict", False):
                    raise
                probe_error(
                    f"{probe}.{method.__name__}",
                    exc,
                    rank=getattr(self, "rank", None),
                )
                return default() if callable(default) else default

        return wrapper

    return decorate


def run_probe(probe: str, call, *args, rank=None, default=None, logger=None,
              **kwargs):
    """Call a probe SETUP or VERIFY step fail-safe; ``default`` on failure.

    Used where there is no probe object to carry ``strict``: building a
    recorder from a batch's debug keys (a mistyped flag must not kill the
    batch) and the end-of-wave verification step.
    """

    try:
        return call(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - that is the contract
        probe_error(probe, exc, rank=rank, logger=logger)
        return default() if callable(default) else default
