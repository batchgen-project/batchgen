# ---------------------------------------------------------------------------- #
#  BatchGen                                                                     #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #

"""Model-neutral Triton prewarm core: compile recorder, late-compile guard,
and the two-phase prewarm runner.

Triton compiles one binary per specialisation key of a ``@triton.jit``
function: the constexpr values, every argument's dtype, and per-argument
attributes (an int equal to 1 becomes a constant, an int or a pointer address
divisible by 16 carries the divisibility attribute, unless the kernel lists
the argument in ``do_not_specialize``). A served request that reaches a key no
earlier launch produced compiles inside the request. A model's prewarm drives
its serving entry points over a table of shape classes before the worker
reports ready; this module supplies the parts that do not depend on the model:

  * ``int_class``, ``class_representatives`` and ``partition``: the integer
    specialisation classes and the deterministic rank split of the entries;
  * ``TritonCompileRecorder``: Triton's ``jit_cache_hook`` and
    ``jit_post_compile_hook``. It counts compiles and disk-cache loads per
    phase and, once its ``LateCompileGuard`` is armed, logs every compilation
    after the prewarm with the full specialisation, or raises when the batch
    sets one of the guard's strict ``batchgen_debug`` keys;
  * ``run_prewarm``: phase 1 runs this rank's share of the entries, which
    the node-local ranks partition because they share one Triton cache
    directory; after a barrier, phase 2 runs every entry on every rank,
    loading from that cache into each process's in-memory kernel cache.

Importable without Triton or torch: both are imported lazily. This module
sits in the kernels layer and must not import the ``batchgen`` package;
callers pass their own batch-debug source to ``LateCompileGuard``.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple


logger = logging.getLogger(__name__)

DEFAULT_LOG_TAG = "[TRITON_PREWARM]"

# Batch-level ``batchgen_debug`` keys that turn a late compile into an error.
DEFAULT_STRICT_KEYS = ("triton_strict",)

# A post-compile hook that fires this long after the cache hook is counted as
# a compilation; anything faster was a load from the on-disk cache.
COMPILE_SECONDS = 0.5

DISARMED = "disarmed"
LOG = "log"
STRICT = "strict"


# ============================================================================
#  Integer classes and partitioning (pure)
# ============================================================================


def int_class(value: int) -> str:
    """Triton's integer specialisation class of ``value``."""
    value = int(value)
    if value == 1:
        return "1"
    if value % 16 == 0:
        return "d16"
    return "any"


def class_representatives(values) -> List[int]:
    """The smallest value of each ``int_class`` present in ``values``."""
    chosen: Dict[str, int] = {}
    for value in sorted({int(v) for v in values if int(v) > 0}):
        chosen.setdefault(int_class(value), value)
    return sorted(chosen.values())


def partition(items: Sequence, index: int, count: int) -> list:
    """Items ``i`` with ``i % count == index``: a deterministic split that
    covers every item exactly once across ``index = 0 .. count - 1``."""
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"invalid partition index {index} of {count}")
    return [item for i, item in enumerate(items) if i % count == index]


# ============================================================================
#  Compile recorder and late-compile guard
# ============================================================================


class LateTritonCompileError(RuntimeError):
    """A Triton kernel compiled after the prewarm under strict mode."""


def _normalise_keys(keys) -> Tuple[str, ...]:
    if isinstance(keys, str):
        return (keys,)
    return tuple(keys)


def _flag_set(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "strict")
    if isinstance(value, (bool, int)):
        return bool(value)
    return False


def strict_key(debug, keys=DEFAULT_STRICT_KEYS) -> Optional[str]:
    """The first of ``keys`` that a ``batchgen_debug`` dict sets truthy."""
    if not isinstance(debug, dict):
        return None
    for key in _normalise_keys(keys):
        if _flag_set(debug.get(key)):
            return key
    return None


def strict_requested(debug, keys=DEFAULT_STRICT_KEYS) -> bool:
    """Whether a ``batchgen_debug`` dict asks for strict mode."""
    return strict_key(debug, keys) is not None


def _no_debug() -> dict:
    return {}


class LateCompileGuard:
    """Disarmed during the prewarm; armed in log mode afterwards.

    Armed, each compilation is logged at WARNING, and raises instead when the
    current batch's debug dict (from ``debug_source``) sets any of
    ``strict_keys`` truthy, or the guard was armed strict. The raise happens
    inside the model forward, which has no per-batch exception boundary, so
    it takes down the worker pool, and the flags of every sequence in the
    step are merged, so a co-scheduled batch's shape can trigger it. Strict
    is therefore for acceptance runs only, never the default.
    """

    def __init__(self, debug_source: Optional[Callable[[], dict]] = None,
                 strict_keys=DEFAULT_STRICT_KEYS):
        self._mode = DISARMED
        self._debug_source = debug_source or _no_debug
        self.strict_keys = _normalise_keys(strict_keys)

    def arm(self, strict: bool = False) -> None:
        self._mode = STRICT if strict else LOG

    def disarm(self) -> None:
        self._mode = DISARMED

    @property
    def armed_mode(self) -> str:
        return self._mode

    def state(self) -> str:
        if self._mode != LOG:
            return self._mode
        if strict_requested(self._debug_source(), self.strict_keys):
            return STRICT
        return LOG

    def strict_reason(self) -> str:
        """What made the guard strict, for the error message."""
        if self._mode == STRICT:
            return "a guard armed strict"
        key = strict_key(self._debug_source(), self.strict_keys)
        if key is not None:
            return f"batchgen_debug.{key}"
        return "strict mode"


@dataclass
class CompileEvent:
    """One Triton compilation (or disk-cache load) as the hooks saw it."""

    kernel: str
    constexprs: str
    signature: str
    attrs: str
    options: str
    wall: Optional[float] = None
    source: str = "pending"
    late: bool = False

    def describe(self) -> str:
        wall = "-" if self.wall is None else f"{self.wall:.3f}s"
        return (
            f"kernel={self.kernel} source={self.source} wall={wall} "
            f"options={self.options} constexprs={self.constexprs} "
            f"signature={self.signature} attrs={self.attrs}"
        )


def _arg_label(path, arg_names: Sequence[str]) -> str:
    """Name a Triton argument path such as ``(3,)`` after the kernel arg."""
    if isinstance(path, int):
        path = (path,)
    if (
        isinstance(path, tuple) and path and isinstance(path[0], int)
        and 0 <= path[0] < len(arg_names)
    ):
        return arg_names[path[0]] + "".join(f"[{i}]" for i in path[1:])
    return str(path)


def _format_constants(constants, arg_names) -> str:
    if not isinstance(constants, dict):
        return "{}" if constants is None else str(constants)
    return "{" + ",".join(
        f"{_arg_label(path, arg_names)}={value!r}"
        for path, value in constants.items()
    ) + "}"


def _format_signature(signature) -> str:
    if not isinstance(signature, dict):
        return "{}" if signature is None else str(signature)
    return "{" + ",".join(f"{name}:{ty}" for name, ty in signature.items()) + "}"


def _format_attrs(attrs, arg_names) -> str:
    if not isinstance(attrs, dict):
        return "{}" if attrs is None else str(attrs)
    parts = []
    for path, values in attrs.items():
        label = _arg_label(path, arg_names)
        if isinstance(values, (list, tuple)):
            for item in values:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    parts.append(f"{label}:{item[0]}={item[1]!r}")
                else:
                    parts.append(f"{label}:{item!r}")
        else:
            parts.append(f"{label}:{values!r}")
    return "{" + ",".join(parts) + "}"


def parse_hook_payload(payload: dict) -> CompileEvent:
    """Build a ``CompileEvent`` from Triton 3.5's hook keyword arguments.

    Triton calls both hooks as ``hook(key=, repr=, fn=JitFunctionInfo,
    compile={"key", "signature", "constants", "configs", "num_warps", ...},
    is_manual_warmup=, already_compiled=)``. Missing fields are tolerated so
    a Triton upgrade degrades the log, not the launch.
    """
    fn = payload.get("fn")
    info = payload.get("compile") or {}
    jit = getattr(fn, "jit_function", None)
    arg_names = list(getattr(jit, "arg_names", None) or [])
    module = getattr(fn, "module", None)
    name = getattr(fn, "name", None)
    if name:
        kernel = f"{module}.{name}" if module else str(name)
    else:
        kernel = str(payload.get("repr", "?")).split("[", 1)[0]
    configs = info.get("configs") or ()
    options = ",".join(
        f"{option}={info.get(option)}"
        for option in ("num_warps", "num_stages", "num_ctas")
    )
    return CompileEvent(
        kernel=kernel,
        constexprs=_format_constants(info.get("constants"), arg_names),
        signature=_format_signature(info.get("signature")),
        attrs=_format_attrs(configs[0] if configs else None, arg_names),
        options=options,
    )


def _pairing_token(payload: dict):
    fn = payload.get("fn")
    owner = getattr(fn, "jit_function", fn)
    return id(owner), str(payload.get("key")), threading.get_ident()


class TritonCompileRecorder:
    """Owns Triton's two process-wide compile hooks.

    ``jit_cache_hook`` fires on an in-memory kernel cache miss, before
    Triton compiles or loads the binary from disk; ``jit_post_compile_hook``
    fires once the kernel is ready. The time between them separates a real
    compile from a disk-cache load. Any hooks already installed are chained.
    The cache hook never returns a truthy value of its own: Triton would
    take that as "skip the compile" and drop the launch.
    """

    def __init__(self, guard: Optional[LateCompileGuard] = None,
                 compile_seconds: float = COMPILE_SECONDS,
                 clock: Callable[[], float] = time.perf_counter,
                 log: Optional[logging.Logger] = None,
                 log_tag: str = DEFAULT_LOG_TAG):
        self.guard = guard or LateCompileGuard()
        self.compile_seconds = float(compile_seconds)
        self.log_tag = str(log_tag)
        self._clock = clock
        self._log = log or logger
        self._lock = threading.Lock()
        self._pending: Dict[tuple, Tuple[float, CompileEvent]] = {}
        self.events: List[CompileEvent] = []
        self.installed = False
        self._runtime = None
        self._previous_cache_hook = None
        self._previous_post_hook = None

    def install(self, runtime=None) -> bool:
        if self.installed:
            return True
        if runtime is None:
            try:
                from triton import knobs
                runtime = knobs.runtime
            except (ImportError, AttributeError) as exc:
                self._log.warning(
                    "%s Triton compile hooks unavailable (%s); the prewarm "
                    "runs without the recorder and the late-compile guard",
                    self.log_tag, exc,
                )
                return False
        if not (
            hasattr(runtime, "jit_cache_hook")
            and hasattr(runtime, "jit_post_compile_hook")
        ):
            self._log.warning(
                "%s this Triton has no jit_cache_hook/jit_post_compile_hook; "
                "the prewarm runs without the recorder and the late-compile "
                "guard", self.log_tag,
            )
            return False
        self._previous_cache_hook = runtime.jit_cache_hook
        self._previous_post_hook = runtime.jit_post_compile_hook
        runtime.jit_cache_hook = self.cache_hook
        runtime.jit_post_compile_hook = self.post_compile_hook
        self._runtime = runtime
        self.installed = True
        return True

    def uninstall(self) -> None:
        if not self.installed:
            return
        runtime = self._runtime
        if runtime.jit_cache_hook == self.cache_hook:
            runtime.jit_cache_hook = self._previous_cache_hook
        if runtime.jit_post_compile_hook == self.post_compile_hook:
            runtime.jit_post_compile_hook = self._previous_post_hook
        self.installed = False
        self._runtime = None

    def cache_hook(self, **payload):
        if self._previous_cache_hook is not None:
            result = self._previous_cache_hook(**payload)
            if result:
                return result
        event = parse_hook_payload(payload)
        state = self.guard.state()
        event.late = state != DISARMED
        if state == STRICT:
            self._log.warning(
                "%s late compile (strict) %s", self.log_tag, event.describe()
            )
            raise LateTritonCompileError(
                f"Triton kernel compiled after the prewarm under "
                f"{self.guard.strict_reason()}: {event.describe()}"
            )
        with self._lock:
            self._pending[_pairing_token(payload)] = (self._clock(), event)
        return None

    def post_compile_hook(self, **payload):
        now = self._clock()
        with self._lock:
            entry = self._pending.pop(_pairing_token(payload), None)
        if entry is None:
            event = parse_hook_payload(payload)
            event.late = self.guard.state() != DISARMED
        else:
            start, event = entry
            event.wall = now - start
            event.source = (
                "compile" if event.wall >= self.compile_seconds else "load"
            )
        with self._lock:
            self.events.append(event)
        if event.late:
            self._log.warning(
                "%s late compile %s", self.log_tag, event.describe()
            )
        else:
            self._log.debug("%s %s", self.log_tag, event.describe())
        if self._previous_post_hook is not None:
            return self._previous_post_hook(**payload)
        return None

    def mark(self) -> int:
        with self._lock:
            return len(self.events)

    def counts(self, since: int = 0) -> Dict[str, int]:
        with self._lock:
            events = self.events[since:]
        return {
            "compiled": sum(1 for e in events if e.source == "compile"),
            "loaded": sum(1 for e in events if e.source == "load"),
        }

    def per_kernel(self, since: int = 0) -> Dict[str, Tuple[int, int]]:
        """``kernel -> (compiled, loaded)`` over events from ``since``."""
        with self._lock:
            events = self.events[since:]
        table: Dict[str, List[int]] = {}
        for event in events:
            row = table.setdefault(event.kernel, [0, 0])
            if event.source == "compile":
                row[0] += 1
            elif event.source == "load":
                row[1] += 1
        return {kernel: (row[0], row[1]) for kernel, row in table.items()}


_RECORDER: Optional[TritonCompileRecorder] = None
_RECORDER_LOCK = threading.Lock()


def get_recorder(
    log_tag: str = DEFAULT_LOG_TAG,
    debug_source: Optional[Callable[[], dict]] = None,
    strict_keys: Optional[Sequence[str]] = None,
) -> TritonCompileRecorder:
    """The process-wide recorder (Triton's hooks are process-wide).

    The first caller creates it and fixes its log tag, batch-debug source and
    strict keys; later callers get the same recorder whatever they pass.
    """
    global _RECORDER
    with _RECORDER_LOCK:
        if _RECORDER is None:
            guard_kwargs = {"debug_source": debug_source}
            if strict_keys is not None:
                guard_kwargs["strict_keys"] = tuple(strict_keys)
            _RECORDER = TritonCompileRecorder(
                guard=LateCompileGuard(**guard_kwargs), log_tag=log_tag,
            )
        return _RECORDER


# ============================================================================
#  Phases
# ============================================================================


@dataclass(frozen=True)
class PrewarmEntry:
    name: str
    run: Callable[[], object]


@dataclass
class PhaseReport:
    name: str
    entries: int
    failed: int
    compiled: int
    loaded: int
    wall: float


def _run_phase(name, entries, recorder, rank, sync,
               log_tag: str = DEFAULT_LOG_TAG) -> PhaseReport:
    start = time.perf_counter()
    mark = recorder.mark() if recorder is not None else 0
    failed = 0
    for entry in entries:
        try:
            entry.run()
            sync()
        except Exception:  # noqa: BLE001 - one bad entry must not stop startup
            failed += 1
            logger.warning(
                "%s rank=%s %s entry=%s failed", log_tag, rank, name,
                entry.name, exc_info=True,
            )
    # -1: no recorder (this Triton exposes no compile hooks).
    counts = (
        recorder.counts(mark) if recorder is not None
        else {"compiled": -1, "loaded": -1}
    )
    report = PhaseReport(
        name=name, entries=len(entries), failed=failed,
        compiled=counts["compiled"], loaded=counts["loaded"],
        wall=time.perf_counter() - start,
    )
    logger.info(
        "%s rank=%s %s compiled=%d loaded=%d entries=%d failed=%d wall=%.1fs",
        log_tag, rank, name, report.compiled, report.loaded, report.entries,
        report.failed, report.wall,
    )
    return report


def _cuda_synchronize() -> None:
    import torch

    torch.cuda.synchronize()


def run_prewarm(entries, *, recorder=None, part_index=0, part_count=1,
                barrier=None, rank=0, sync=None,
                log_tag: Optional[str] = None) -> List[PhaseReport]:
    """Phase 1 on this rank's share, barrier, phase 2 on every entry.

    With a single participant phase 1 already ran everything in this
    process, so phase 2 is skipped. ``log_tag`` defaults to the recorder's
    tag, else ``DEFAULT_LOG_TAG``.
    """
    if sync is None:
        sync = _cuda_synchronize
    if log_tag is None:
        log_tag = recorder.log_tag if recorder is not None else DEFAULT_LOG_TAG
    reports = [_run_phase(
        "phase1", partition(entries, part_index, part_count), recorder, rank,
        sync, log_tag,
    )]
    if part_count > 1:
        if barrier is not None:
            barrier()
        reports.append(_run_phase("phase2", entries, recorder, rank, sync,
                                  log_tag))
    return reports
