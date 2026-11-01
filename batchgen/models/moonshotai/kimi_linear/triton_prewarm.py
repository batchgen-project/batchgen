# ---------------------------------------------------------------------------- #
#  BatchGen                                                                     #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #

"""Kimi-K3 Triton prewarm, run before the worker reports ready.

Triton compiles one binary per specialisation key of a ``@triton.jit``
function: the constexpr values, every argument's dtype, and per-argument
attributes (an int equal to 1 becomes a constant, an int or a pointer address
divisible by 16 carries the divisibility attribute, unless the kernel lists
the argument in ``do_not_specialize``). A served request that reaches a key no
earlier launch produced compiles inside the request, and fla's KDA kernels
alone have hundreds of reachable keys.

The prewarm drives the serving path's own Python entry points
(``kda_prefill_serving``, ``kda_decode_serving``, the module norms and
activations, the attention-residual mixer) over a table of shape classes, so
every key is produced by the same wrappers, views and kwargs that serving
uses. Replaying recorded ``triton.compile`` calls was rejected: the attribute
format is Triton-version specific and the replay would break on every Triton
or fla upgrade.

Parts:
  * ``KDA_PREFILL_SHAPES`` and the pure class model (``kda_call_classes``)
    that the CPU tests check the table against;
  * ``TritonCompileRecorder``: Triton's ``jit_cache_hook`` and
    ``jit_post_compile_hook``. It counts compiles and disk-cache loads per
    phase and, once its ``LateCompileGuard`` is armed, logs every compilation
    after the prewarm with the full specialisation, or raises when the batch
    sets ``batchgen_debug.k3_triton_strict``;
  * ``prewarm_k3_triton_kernels``: phase 1 partitions the entries over the
    node-local attention-TP ranks, which share one Triton cache directory;
    after a barrier, phase 2 runs every entry on every rank, loading from
    that cache into each process's in-memory kernel cache.

Importable without Triton or fla: every GPU dependency is imported lazily.
"""

import logging
import threading
import time
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch


logger = logging.getLogger(__name__)

LOG_TAG = "[K3_TRITON_WARMUP]"

# Batch-level ``batchgen_debug`` key that turns a late compile into an error.
STRICT_DEBUG_KEY = "k3_triton_strict"

# A post-compile hook that fires this long after the cache hook is counted as
# a compilation; anything faster was a load from the on-disk cache.
COMPILE_SECONDS = 0.5

# Row counts for kernels whose row count only sizes the launch grid. Two
# values guard against a row argument being added to one of them later.
ROW_COUNTS = (1, 17)

# ``num_bank_rows`` is a runtime int of the attention-residual kernels and is
# always below 16, so its classes are "== 1" and "other".
ATTN_RESIDUAL_BANK_ROWS = (1, 2)

DISARMED = "disarmed"
LOG = "log"
STRICT = "strict"


# ============================================================================
#  Shape classes (pure)
# ============================================================================


@dataclass(frozen=True)
class KDAPrefillShape:
    """One packed prefill micro-batch through a single KDA layer.

    ``lengths`` holds one ``(a, b)`` pair per sequence, meaning
    ``a * segment_tokens + b`` tokens, so an entry that must cross the segment
    cap does so whichever cap the prefill mode selects. The fixed lengths
    assume a cap of at least 1,024 tokens (the serving caps are 4,096 and
    16,384); a smaller cap would segment them and change the class they
    witness.

    ``slot_offset`` > 0 hands the conv kernel a slot-id view that starts
    ``slot_offset`` int32 elements into its storage. That is the pointer
    alignment of a prefix-pool level's ``slots[row_start:row_end]`` slice;
    the plain packed path always passes a fresh, aligned tensor.
    """

    name: str
    lengths: Tuple[Tuple[int, int], ...]
    slot_offset: int = 0

    def resolve(self, segment_tokens: int) -> List[int]:
        lengths = [a * int(segment_tokens) + b for a, b in self.lengths]
        if not lengths or any(n <= 0 for n in lengths):
            raise ValueError(
                f"KDA prewarm shape {self.name} resolves to {lengths} at "
                f"segment_tokens={segment_tokens}"
            )
        return lengths


def _tokens(*counts: int) -> Tuple[Tuple[int, int], ...]:
    return tuple((0, int(count)) for count in counts)


def _over_cap(extra: int) -> Tuple[Tuple[int, int], ...]:
    return ((1, int(extra)),)


# Every reachable (T, N, NT, varlen) integer class of a ``chunk_kda`` call, one
# witness each. T is the call's token count, N its sequence count, NT its total
# chunk count (sum of ceil(len / 64)); each is classed "1", "d16" (divisible by
# 16) or "any". A call is varlen whenever cu_seqlens is passed: always on the
# single-call path, and on a segment that overlaps more than one sequence. A
# segment inside one sequence is a batch-1 call whose N is not an argument
# ("-"). Extend the table, not the code, when the recorder reports a miss.
KDA_PREFILL_SHAPES = (
    # Single varlen call (total <= segment cap).        (T, N, NT)
    KDAPrefillShape("t1", _tokens(1)),                    # (1, 1, 1)
    KDAPrefillShape("t17", _tokens(17)),                  # (any, 1, 1)
    KDAPrefillShape("t1000", _tokens(1000)),              # (any, 1, d16)
    KDAPrefillShape("t65", _tokens(65)),                  # (any, 1, any)
    KDAPrefillShape("t64", _tokens(64)),                  # (d16, 1, 1)
    KDAPrefillShape("t1024", _tokens(1024)),              # (d16, 1, d16)
    KDAPrefillShape("t128", _tokens(128)),                # (d16, 1, any)
    KDAPrefillShape("n2_t3", _tokens(1, 2)),              # (any, any, any)
    KDAPrefillShape("n2_t961", _tokens(960, 1)),          # (any, any, d16)
    KDAPrefillShape("n2_t16", _tokens(8, 8)),             # (d16, any, any)
    KDAPrefillShape("n2_t1024", _tokens(960, 64)),        # (d16, any, d16)
    KDAPrefillShape("n16_t16", _tokens(*([1] * 16))),     # (d16, d16, d16)
    KDAPrefillShape("n16_t17", _tokens(*([1] * 15), 2)),  # (any, d16, d16)
    KDAPrefillShape("n16_t80", _tokens(*([1] * 15), 65)),  # (d16, d16, any)
    KDAPrefillShape("n16_t81", _tokens(*([1] * 15), 66)),  # (any, d16, any)
    # Segmented sweep: a cap-sized batch-1 call, then a batch-1 tail.
    KDAPrefillShape("cap_tail1", _over_cap(1)),           # tail (1, -, 1)
    KDAPrefillShape("cap_tail17", _over_cap(17)),         # tail (any, -, 1)
    KDAPrefillShape("cap_tail1000", _over_cap(1000)),     # tail (any, -, d16)
    KDAPrefillShape("cap_tail65", _over_cap(65)),         # tail (any, -, any)
    KDAPrefillShape("cap_tail64", _over_cap(64)),         # tail (d16, -, 1)
    KDAPrefillShape("cap_tail1024", _over_cap(1024)),     # tail (d16, -, d16)
    KDAPrefillShape("cap_tail128", _over_cap(128)),       # tail (d16, -, any)
    # Conv slot ids at a 4-byte offset (prefix-pool level slice).
    KDAPrefillShape("n2_slot_offset", _tokens(17, 65), slot_offset=1),
)


def int_class(value: int) -> str:
    """Triton's integer specialisation class of ``value``."""
    value = int(value)
    if value == 1:
        return "1"
    if value % 16 == 0:
        return "d16"
    return "any"


def kda_call_classes(lengths: Sequence[int], segment_tokens: int, plan_fn,
                     chunk_size: int = 64) -> List[Tuple[str, str, str, bool]]:
    """``(T, N, NT, varlen)`` classes of every ``chunk_kda`` call for one
    packed micro-batch, mirroring ``serving_modules._kda_chunk_segments``.

    ``plan_fn`` is ``serving_modules._kda_segment_plan``, passed in so the
    model is testable without the serving module's imports.
    """
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + int(n))
    total = cu[-1]
    if total <= segment_tokens:
        calls = [(total, list(lengths), True)]
    else:
        calls = []
        for start, end, lo, hi, bounds in plan_fn(cu, segment_tokens):
            seqs = [b - a for a, b in zip(bounds, bounds[1:])]
            calls.append((end - start, seqs, hi - lo > 1))
    classes = []
    for tokens, seqs, varlen in calls:
        chunks = sum(-(-int(n) // chunk_size) for n in seqs)
        classes.append((
            int_class(tokens),
            int_class(len(seqs)) if varlen else "-",
            int_class(chunks),
            varlen,
        ))
    return classes


def planned_kda_call_classes(shapes, segment_tokens: int, plan_fn,
                             chunk_size: int = 64) -> set:
    """Union of the call classes the shape table produces at this cap."""
    planned = set()
    for shape in shapes:
        planned.update(kda_call_classes(
            shape.resolve(segment_tokens), segment_tokens, plan_fn, chunk_size
        ))
    return planned


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


def strict_requested(debug) -> bool:
    """Whether a ``batchgen_debug`` dict asks for strict mode."""
    if not isinstance(debug, dict):
        return False
    value = debug.get(STRICT_DEBUG_KEY)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "strict")
    if isinstance(value, (bool, int)):
        return bool(value)
    return False


def _batch_debug() -> dict:
    """The served batch's ``batchgen_debug``, through the class attribute
    that ``batchgen_debug.kimi_decode_graph_mode`` is read from."""
    try:
        from .cuda_graph_segments import _debug_dict
    except ImportError:
        return {}
    return _debug_dict()


class LateCompileGuard:
    """Disarmed during the prewarm; armed in log mode afterwards.

    Armed, each compilation is logged at WARNING, and raises instead when the
    current batch sets ``batchgen_debug.k3_triton_strict`` or the guard was
    armed strict. Strict is never the default: an unforeseen key would turn a
    slow request into a failed one.
    """

    def __init__(self, debug_source: Optional[Callable[[], dict]] = None):
        self._mode = DISARMED
        self._debug_source = debug_source or _batch_debug

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
        return STRICT if strict_requested(self._debug_source()) else LOG


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
                 log: Optional[logging.Logger] = None):
        self.guard = guard or LateCompileGuard()
        self.compile_seconds = float(compile_seconds)
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
                    LOG_TAG, exc,
                )
                return False
        if not (
            hasattr(runtime, "jit_cache_hook")
            and hasattr(runtime, "jit_post_compile_hook")
        ):
            self._log.warning(
                "%s this Triton has no jit_cache_hook/jit_post_compile_hook; "
                "the prewarm runs without the recorder and the late-compile "
                "guard", LOG_TAG,
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
                "%s late compile (strict) %s", LOG_TAG, event.describe()
            )
            raise LateTritonCompileError(
                f"Triton kernel compiled after the K3 prewarm under "
                f"batchgen_debug.{STRICT_DEBUG_KEY}: {event.describe()}"
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
            self._log.warning("%s late compile %s", LOG_TAG, event.describe())
        else:
            self._log.debug("%s %s", LOG_TAG, event.describe())
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


def get_recorder() -> TritonCompileRecorder:
    """The process-wide recorder (Triton's hooks are process-wide)."""
    global _RECORDER
    if _RECORDER is None:
        _RECORDER = TritonCompileRecorder()
    return _RECORDER


# ============================================================================
#  Prewarm entries
# ============================================================================


@dataclass(frozen=True)
class PrewarmEntry:
    name: str
    run: Callable[[], object]


def _scratch_kda_state(template, num_slots: int):
    """Zeroed pools with the live layer's dtypes and per-slot layout.

    The live per-layer pools are contiguous slices of the state manager's
    tensors, so a contiguous pool with the same trailing shape presents the
    same strides to every kernel, and no live slot is read or written.
    """

    def like(pool):
        return torch.zeros(
            (int(num_slots), *pool.shape[1:]), dtype=pool.dtype,
            device=pool.device,
        )

    return SimpleNamespace(
        conv_q=like(template.conv_q),
        conv_k=like(template.conv_k),
        conv_v=like(template.conv_v),
        recurrent_pool=like(template.recurrent_pool),
        cur_decode_slots=None,
        has_initial_state=None,
    )


def _run_kda_prefill(module, template, shape: KDAPrefillShape,
                     segment_tokens: int) -> None:
    weight = module.q_proj.weight
    device, dtype = weight.device, weight.dtype
    lengths = shape.resolve(segment_tokens)
    num_slots = shape.slot_offset + len(lengths)
    state = _scratch_kda_state(template, num_slots)
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    # int32 cu_seqlens and bool has_initial_state, as the worker builds them.
    cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=device)
    slot_ids = torch.arange(
        num_slots, dtype=torch.int32, device=device
    )[shape.slot_offset:]
    has_initial_state = torch.zeros(len(lengths), dtype=torch.bool,
                                    device=device)
    hidden = torch.zeros(cu[-1], weight.shape[1], dtype=dtype, device=device)
    module.kda_prefill_serving(
        hidden, cu_seqlens, slot_ids, has_initial_state, state,
        segment_tokens=segment_tokens, reduce_output=False,
    )


def _run_kda_decode(module, template, rows: int) -> None:
    weight = module.q_proj.weight
    state = _scratch_kda_state(template, rows)
    state.cur_decode_slots = torch.arange(
        rows, dtype=torch.int32, device=weight.device
    )
    hidden = torch.zeros(rows, 1, weight.shape[1], dtype=weight.dtype,
                         device=weight.device)
    module.kda_decode_serving(hidden, state, reduce_output=False)


# fla's gated norm sizes its constexpr block count as ceil(rows / this), with
# rows = tokens x heads of the whole prefill micro-batch.
O_NORM_ROWS_PER_BLOCK = 2048 * 32


def o_norm_token_counts(heads: int, max_tokens: int) -> List[int]:
    """Token counts that reach every o_norm block count up to
    ``max_tokens``, each with a row count divisible by 16 and one that is not."""
    max_blocks = -(-int(max_tokens) * int(heads) // O_NORM_ROWS_PER_BLOCK)
    counts = []
    for blocks in range(1, max_blocks + 1):
        tokens = min(blocks * O_NORM_ROWS_PER_BLOCK // heads, int(max_tokens))
        wanted = {True, False}
        while wanted and tokens >= 1 and (
                (blocks - 1) * O_NORM_ROWS_PER_BLOCK < tokens * heads):
            aligned = (tokens * heads) % 16 == 0
            if aligned in wanted:
                wanted.discard(aligned)
                counts.append(tokens)
            tokens -= 1
    return sorted(counts)


def _run_o_norm(o_norm, heads: int, head_dim: int, tokens_list, dtype,
                device) -> None:
    for tokens in tokens_list:
        o = torch.empty(tokens, heads, head_dim, dtype=dtype, device=device)
        o_norm(o, torch.empty_like(o))
        del o


def kda_prefill_entries(module, template, segment_tokens: int,
                        shapes=KDA_PREFILL_SHAPES,
                        max_prefill_tokens: int = 0) -> List[PrewarmEntry]:
    entries = [
        PrewarmEntry(
            f"kda_prefill/{shape.name}",
            partial(_run_kda_prefill, module, template, shape,
                    int(segment_tokens)),
        )
        for shape in shapes
    ]
    weight = module.q_proj.weight
    entries.extend(beta_sigmoid_entries(module.num_heads, weight.dtype,
                                        weight.device))
    if max_prefill_tokens > 0:
        heads, head_dim = int(module.num_heads), int(module.head_dim)
        tokens_list = o_norm_token_counts(heads, max_prefill_tokens)
        entries.append(PrewarmEntry(
            f"kda_o_norm/max{int(max_prefill_tokens)}",
            partial(_run_o_norm, module.o_norm, heads, head_dim, tokens_list,
                    weight.dtype, weight.device),
        ))
    return entries


def kda_decode_entries(module, template, buckets) -> List[PrewarmEntry]:
    return [
        PrewarmEntry(f"kda_decode/b{int(rows)}",
                     partial(_run_kda_decode, module, template, int(rows)))
        for rows in sorted({int(b) for b in buckets if int(b) > 0})
    ]


def _run_attn_residual(proj, norm, bank_rows: int) -> None:
    from .attn_residual_triton import mix_attn_residual_triton

    weight = proj.weight
    hidden = weight.shape[1]
    prefix = torch.zeros((1, hidden), dtype=weight.dtype, device=weight.device)
    bank = torch.zeros((1, bank_rows, hidden), dtype=weight.dtype,
                       device=weight.device)
    mix_attn_residual_triton(prefix, bank, proj, norm)


def attn_residual_entries(model) -> List[PrewarmEntry]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if not layers:
        return []
    proj = getattr(layers[0], "self_attention_res_proj", None)
    norm = getattr(layers[0], "self_attention_res_norm", None)
    if proj is None or norm is None:
        return []
    return [
        PrewarmEntry(f"attn_residual/rows{rows}",
                     partial(_run_attn_residual, proj, norm, rows))
        for rows in ATTN_RESIDUAL_BANK_ROWS
    ]


def _run_rows(fn, width: int, dtype, device) -> None:
    for rows in ROW_COUNTS:
        fn(torch.zeros(rows, width, dtype=dtype, device=device))


def _run_situ_gated(situ_gated, n: int, dtype, device) -> None:
    for rows in ROW_COUNTS:
        gate_up = torch.zeros(rows, 2 * n, dtype=dtype, device=device)
        out = torch.empty(rows, n, dtype=dtype, device=device)
        situ_gated(gate_up, out)


def _run_row_gathers(hidden: int, dtype, device) -> None:
    from .row_gather_triton import add_gathered_rows_masked, gather_rows_masked

    for rows in ROW_COUNTS:
        src = torch.zeros(rows, hidden, dtype=dtype, device=device)
        idx = torch.arange(rows, dtype=torch.long, device=device)
        valid = torch.ones(rows, dtype=torch.int32, device=device)
        gather_rows_masked(src, idx, valid)
        add_gathered_rows_masked(src, src, idx, valid)


def _run_beta_sigmoid(beta_sigmoid_fwd, heads: int, dtype, device) -> None:
    # KDA prefill applies the beta sigmoid to ``beta[lo:hi]`` of a contiguous
    # (tokens, heads) tensor, so the input address is 16-byte aligned only for
    # some level offsets ``lo``; the element count is tokens * heads.
    for element_offset in (0, 1):
        for tokens in (1, 16):
            count = tokens * heads
            storage = torch.zeros(count + element_offset, dtype=dtype,
                                  device=device)
            beta_sigmoid_fwd(storage[element_offset:])


def beta_sigmoid_entries(heads: int, dtype, device) -> List[PrewarmEntry]:
    try:
        from fla.ops.common.gate import fused_beta_sigmoid_fwd
    except ImportError as exc:
        logger.debug("%s beta sigmoid not prewarmed: %s", LOG_TAG, exc)
        return []
    return [PrewarmEntry(
        f"beta_sigmoid/h{int(heads)}",
        partial(_run_beta_sigmoid, fused_beta_sigmoid_fwd, int(heads), dtype,
                device),
    )]


def _resident(weight) -> bool:
    return weight is not None and not weight.is_meta and weight.numel() > 0


def row_kernel_entries(model, config, act_dtype, device) -> List[PrewarmEntry]:
    """Norms, SiTU activations and decode row gathers: one launch-grid-only
    row axis each, so one entry per distinct width suffices."""
    from .model import KimiMLP, KimiRMSNorm, SituAndMul

    norms = {}
    situs = {}
    for module in model.modules():
        if isinstance(module, KimiRMSNorm):
            weight = module.weight
            # The Triton norm runs only when the activation dtype matches.
            if _resident(weight) and weight.dtype == act_dtype:
                norms.setdefault(int(weight.shape[0]), module)
        elif isinstance(module, KimiMLP) and isinstance(
            module.act_fn, SituAndMul
        ):
            weight = module.gate_proj.weight
            if _resident(weight):
                key = (int(weight.shape[0]),
                       module.act_fn.linear_beta is not None)
                situs.setdefault(key, module.act_fn)

    entries = [
        PrewarmEntry(f"rmsnorm/h{hidden}",
                     partial(_run_rows, norm, hidden, act_dtype, device))
        for hidden, norm in sorted(norms.items())
    ]
    entries.extend(
        PrewarmEntry(
            f"situ/d{width}" + ("_lb" if linear_beta else ""),
            partial(_run_rows, act, 2 * width, act_dtype, device),
        )
        for (width, linear_beta), act in sorted(situs.items())
    )

    routed = getattr(config, "moe_intermediate_size", None)
    if routed and getattr(config, "routed_expert_hidden_size", None):
        try:
            from batchgen.moe.k3_prefill_dequant_once import situ_gated
        except ImportError as exc:
            logger.debug("%s situ_gated not prewarmed: %s", LOG_TAG, exc)
        else:
            entries.append(PrewarmEntry(
                f"situ_gated/n{int(routed)}",
                partial(_run_situ_gated, situ_gated, int(routed), act_dtype,
                        device),
            ))

    from .row_gather_triton import row_gather_triton_available

    hidden_size = int(getattr(config, "hidden_size", 0) or 0)
    if hidden_size > 0 and row_gather_triton_available():
        entries.append(PrewarmEntry(
            f"row_gather/h{hidden_size}",
            partial(_run_row_gathers, hidden_size, act_dtype, device),
        ))
    return entries


def build_prewarm_entries(model, config, decode_buckets,
                          max_prefill_tokens: int = 0) -> List[PrewarmEntry]:
    """The deterministic entry list; identical on every rank of a node."""
    from .serving_modules import KDA_PREFILL_SEGMENT_TOKENS
    from .wrappers import KimiLinearKDAWrapper

    entries: List[PrewarmEntry] = []
    act_dtype = torch.bfloat16
    device = None
    kda_index = next(
        (i for i in range(int(config.num_hidden_layers))
         if config.is_kda_layer(i)),
        None,
    )
    if kda_index is not None:
        wrapper = model.model.layers[kda_index].self_attn
        module = getattr(wrapper, "module", wrapper)
        weight = getattr(getattr(module, "q_proj", None), "weight", None)
        template = KimiLinearKDAWrapper.layer_pools.get(kda_index)
        if template is None or not _resident(weight):
            logger.warning(
                "%s KDA layer %d has no resident weights or state pools; "
                "KDA kernels are not prewarmed", LOG_TAG, kda_index,
            )
        else:
            act_dtype, device = weight.dtype, weight.device
            # The prefill mode's cap (_set_prefill_memory_tiling), else the
            # serving default the wrapper falls back to.
            segment_tokens = (
                getattr(wrapper, "_resident_prefill_segment_tokens", None)
                or KDA_PREFILL_SEGMENT_TOKENS
            )
            entries.extend(kda_prefill_entries(
                module, template, segment_tokens,
                max_prefill_tokens=max_prefill_tokens))
            entries.extend(kda_decode_entries(module, template,
                                              decode_buckets))
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    entries.extend(attn_residual_entries(model))
    entries.extend(row_kernel_entries(model, config, act_dtype, device))
    return entries


# ============================================================================
#  Phases
# ============================================================================


@dataclass
class PhaseReport:
    name: str
    entries: int
    failed: int
    compiled: int
    loaded: int
    wall: float


def _run_phase(name, entries, recorder, rank, sync) -> PhaseReport:
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
                "%s rank=%s %s entry=%s failed", LOG_TAG, rank, name,
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
        LOG_TAG, rank, name, report.compiled, report.loaded, report.entries,
        report.failed, report.wall,
    )
    return report


def run_prewarm(entries, *, recorder=None, part_index=0, part_count=1,
                barrier=None, rank=0, sync=None) -> List[PhaseReport]:
    """Phase 1 on this rank's share, barrier, phase 2 on every entry.

    With a single participant phase 1 already ran everything in this
    process, so phase 2 is skipped.
    """
    if sync is None:
        sync = torch.cuda.synchronize
    reports = [_run_phase(
        "phase1", partition(entries, part_index, part_count), recorder, rank,
        sync,
    )]
    if part_count > 1:
        if barrier is not None:
            barrier()
        reports.append(_run_phase("phase2", entries, recorder, rank, sync))
    return reports


def _clear_plan_caches() -> None:
    """Drop the warmup's cu_seqlens from the one-entry plan caches."""
    from . import kda_conv_triton, serving_modules

    serving_modules._KDA_SEGMENT_PLAN_CACHE.clear()
    kda_conv_triton._PLAN_CACHE.clear()


_PREWARM_DONE = False


def prewarm_k3_triton_kernels(model, config, *, decode_buckets,
                              max_prefill_tokens=0, part_index=0,
                              part_count=1, barrier=None, rank=0) -> bool:
    """Compile every planned K3 Triton key, then arm the late-compile guard.

    Once per process: the in-memory kernel caches outlive model rebuilds,
    and a worker restart is a new process that runs it again. Returns
    whether it ran.
    """
    global _PREWARM_DONE
    if _PREWARM_DONE:
        return False
    recorder = get_recorder()
    active = recorder if recorder.install() else None
    with torch.inference_mode():
        entries = build_prewarm_entries(model, config, decode_buckets,
                                        max_prefill_tokens=max_prefill_tokens)
        run_prewarm(
            entries, recorder=active, part_index=part_index,
            part_count=part_count, barrier=barrier, rank=rank,
        )
    _clear_plan_caches()
    torch.cuda.empty_cache()
    if active is not None:
        for kernel, (compiled, loaded) in sorted(active.per_kernel().items()):
            logger.debug(
                "%s rank=%s variants kernel=%s compiled=%d loaded=%d",
                LOG_TAG, rank, kernel, compiled, loaded,
            )
        active.guard.arm()
        logger.info(
            "%s rank=%s late-compile guard armed: later compiles log at "
            "WARNING; batchgen_debug.%s makes them raise",
            LOG_TAG, rank, STRICT_DEBUG_KEY,
        )
    _PREWARM_DONE = True
    return True
