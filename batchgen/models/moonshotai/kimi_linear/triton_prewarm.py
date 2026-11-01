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
  * the compile recorder, the late-compile guard and the two-phase runner
    from ``batchgen_kernels.triton_prewarm``, bound here to the
    ``[K3_TRITON_WARMUP]`` log tag and ``batchgen_debug.k3_triton_strict``;
  * ``prewarm_k3_triton_kernels``: phase 1 partitions the entries over the
    node-local attention-TP ranks, which share one Triton cache directory;
    after a barrier, phase 2 runs every entry on every rank, loading from
    that cache into each process's in-memory kernel cache.

Importable without Triton or fla: every GPU dependency is imported lazily.
"""

import logging
import time
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from batchgen_kernels import triton_prewarm as _core
from batchgen_kernels.triton_prewarm import (  # noqa: F401 - re-exported
    COMPILE_SECONDS,
    DISARMED,
    LOG,
    STRICT,
    CompileEvent,
    LateTritonCompileError,
    PhaseReport,
    PrewarmEntry,
    class_representatives,
    int_class,
    parse_hook_payload,
    partition,
)


logger = logging.getLogger(__name__)

LOG_TAG = "[K3_TRITON_WARMUP]"

# Batch-level ``batchgen_debug`` key that turns a late compile into an error.
STRICT_DEBUG_KEY = "k3_triton_strict"

# Row counts for kernels whose row count only sizes the launch grid. Two
# values guard against a row argument being added to one of them later.
ROW_COUNTS = (1, 17)

# ``num_bank_rows`` is a runtime int of the attention-residual kernels and is
# always below 16, so its classes are "== 1" and "other".
ATTN_RESIDUAL_BANK_ROWS = (1, 2)

# Width classes of the decode graph's static page table (max pages per
# sequence), which the paged KV write takes as a runtime int.
PAGE_TABLE_COLS = (16, 17)

# Page size when no paged-KV manager exists yet at prewarm time; the serving
# page sizes are powers of two from 16 up, so all share this int class.
DEFAULT_PAGE_SIZE_TOKENS = 64


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


# ============================================================================
#  Compile recorder and late-compile guard, bound to K3
# ============================================================================


def strict_requested(debug) -> bool:
    """Whether a ``batchgen_debug`` dict sets ``k3_triton_strict``."""
    return _core.strict_requested(debug, (STRICT_DEBUG_KEY,))


def _batch_debug() -> dict:
    """The served batch's ``batchgen_debug``, through the class attribute
    that ``batchgen_debug.kimi_decode_graph_mode`` is read from."""
    try:
        from .cuda_graph_segments import _debug_dict
    except ImportError:
        return {}
    return _debug_dict()


class LateCompileGuard(_core.LateCompileGuard):
    """``batchgen_kernels.triton_prewarm.LateCompileGuard`` whose defaults
    read the served batch's ``batchgen_debug`` and its ``k3_triton_strict``
    key."""

    def __init__(self, debug_source: Optional[Callable[[], dict]] = None,
                 strict_keys=(STRICT_DEBUG_KEY,)):
        super().__init__(debug_source=debug_source or _batch_debug,
                         strict_keys=strict_keys)


class TritonCompileRecorder(_core.TritonCompileRecorder):
    """``batchgen_kernels.triton_prewarm.TritonCompileRecorder`` whose
    defaults are the K3 guard, this module's logger and ``LOG_TAG``."""

    def __init__(self, guard: Optional[_core.LateCompileGuard] = None,
                 compile_seconds: float = COMPILE_SECONDS,
                 clock: Callable[[], float] = time.perf_counter,
                 log: Optional[logging.Logger] = None,
                 log_tag: str = LOG_TAG):
        super().__init__(guard=guard or LateCompileGuard(),
                         compile_seconds=compile_seconds, clock=clock,
                         log=log or logger, log_tag=log_tag)


def get_recorder() -> _core.TritonCompileRecorder:
    """The process-wide recorder (Triton's hooks are process-wide)."""
    return _core.get_recorder(
        log_tag=LOG_TAG, debug_source=_batch_debug,
        strict_keys=(STRICT_DEBUG_KEY,),
    )


def run_prewarm(entries, *, log_tag: str = LOG_TAG, **kwargs):
    """``batchgen_kernels.triton_prewarm.run_prewarm`` under ``LOG_TAG``."""
    return _core.run_prewarm(entries, log_tag=log_tag, **kwargs)


# ============================================================================
#  Prewarm entries
# ============================================================================


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


def _run_marlin_dequant(dequant_marlin_bf16, device) -> None:
    # Marlin packs K in 16-row blocks and N in 64-column tiles, so every
    # routed weight has N and K divisible by 16: a single key.
    n, k = 64, 32
    qw = torch.zeros(k // 16, n * 2, dtype=torch.int32, device=device)
    scales = torch.full((k // 32, n), 127, dtype=torch.uint8, device=device)
    out = torch.empty(n, k, dtype=torch.bfloat16, device=device)
    dequant_marlin_bf16(qw, scales, out)


def _run_mla_kv_write(update, elements: int, page_size: int, rows_list,
                      dtype, device) -> None:
    """The decode graph's paged write of each new MLA latent, captured with
    ``num_valid_tokens``; the eager fused write passes none."""
    for rows in rows_list:
        tokens = torch.zeros(rows, elements, dtype=dtype, device=device)
        index = torch.zeros(rows, dtype=torch.int32, device=device)
        valid = torch.full((1,), rows, dtype=torch.int32, device=device)
        for cols in PAGE_TABLE_COLS:
            cache = torch.zeros(1, page_size, elements, dtype=dtype,
                                device=device)
            page_table = torch.zeros(rows, cols, dtype=torch.int32,
                                     device=device)
            for num_valid in (valid, None):
                update(
                    k_cache=cache, k_tokens=tokens, page_table=page_table,
                    slot_indices=index, token_indices=index,
                    page_size_tokens=page_size, num_valid_tokens=num_valid,
                )


def _page_size_tokens() -> int:
    from batchgen.models.wrappers import AttnWrapperBase

    manager = getattr(AttnWrapperBase, "gpu_paged_kv_manager", None)
    size = getattr(getattr(manager, "config", None), "page_size_tokens", None)
    return int(size) if size else DEFAULT_PAGE_SIZE_TOKENS


def mla_kv_write_entries(model, config, decode_buckets,
                         device) -> List[PrewarmEntry]:
    mla_index = next(
        (i for i in range(int(config.num_hidden_layers))
         if not config.is_kda_layer(i)),
        None,
    )
    if mla_index is None:
        return []
    wrapper = model.model.layers[mla_index].self_attn
    attn = getattr(wrapper, "module", wrapper)
    weight = getattr(getattr(attn, "kv_a_proj_with_mqa", None), "weight",
                     None)
    rank = getattr(attn, "kv_lora_rank", None)
    rope = getattr(attn, "qk_rope_head_dim", None)
    if not _resident(weight) or rank is None or rope is None:
        return []
    try:
        from batchgen_kernels.triton.kv_cache import (
            run_paged_kv_token_update_fused,
        )
    except ImportError as exc:
        logger.debug("%s MLA KV write not prewarmed: %s", LOG_TAG, exc)
        return []
    rows_list = class_representatives(decode_buckets)
    elements = int(rank) + int(rope)
    return [PrewarmEntry(
        f"mla_kv_write/e{elements}",
        partial(_run_mla_kv_write, run_paged_kv_token_update_fused, elements,
                _page_size_tokens(), rows_list, weight.dtype, device),
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
            from batchgen.moe.k3_prefill_dequant_once import (
                dequant_marlin_bf16,
                situ_gated,
            )
        except ImportError as exc:
            logger.debug("%s dequant-once prefill not prewarmed: %s",
                         LOG_TAG, exc)
        else:
            entries.append(PrewarmEntry(
                f"situ_gated/n{int(routed)}",
                partial(_run_situ_gated, situ_gated, int(routed), act_dtype,
                        device),
            ))
            entries.append(PrewarmEntry(
                "marlin_dequant",
                partial(_run_marlin_dequant, dequant_marlin_bf16, device),
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
    entries.extend(mla_kv_write_entries(model, config, decode_buckets,
                                        device))
    return entries


# ============================================================================
#  Prewarm
# ============================================================================


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
