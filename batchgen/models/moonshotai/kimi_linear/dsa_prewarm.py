# ---------------------------------------------------------------------------- #
#  BatchGen                                                                     #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #

"""Kimi-K3.1 DSA prewarm: the indexer's JIT kernels, compiled before ready.

The DSA layers of Kimi-K3.1 reach JIT kernels no Kimi-K3 layer does
(``jit_coverage.K31_DSA_JIT_KERNELS``). Their entries join the K3 Triton
prewarm (``triton_prewarm.build_prewarm_entries``), which runs on every rank
inside ``configure_prefill`` during the pre-readiness K3 startup, so they share
its node-local partition, barrier and every-rank load pass. Every entry
launches the serving call (the same function, dtypes, layouts and kwargs the
eager and graph steps hand it) on fresh zero tensors; nothing reads a live KV
pool, page table or weight.

Key model (what each kernel's binary is selected by):

==================================  ===========================================
kernel                              key -> witnesses
==================================  ===========================================
``_per_token_blocked_quantize_1d``  dim (``index_head_dim``), plain row-major
(``act_quant``, prefill scorer)     scales (stride 1), the row count's class.
                                    ``_score_and_select_prefill`` at rows
                                    ``PREFILL_ROW_WITNESSES``: the keys cast
                                    ``rows`` rows, the queries ``rows * H``.
``sm90_fp8_mqa_logits``             heads, head_dim, compressed logits: one
                                    key; the same prefill scorer call.
``_transform_selected_positions``   ``index_topk`` width, primary page size,
                                    ``has_valid_tokens``, int64 indices (both
                                    steps clamp to int64). Eager: width
                                    ``index_topk``, no valid count. Graph:
                                    width ``min(index_topk, W * page)`` for
                                    every aux graph width W, valid count.
``_prepare_paged_mqa_inputs``       aux page size, ``BLOCK_P = min(256,
(graph scorer prologue)             next_pow2(W))`` and the class of W (the
                                    table strides and ``max_pages``): one
                                    witness per ``(BLOCK_P, class)`` over W in
                                    ``[1, W_max]``.
``sm90_paged_mqa_logits_metadata``  ``align(batch, 32)``: every multiple of 32
``sm90_fp8_paged_mqa_logits``       up to the per-rank decode row cap (graph
                                    buckets and eager batches alike), through
                                    the scorer's own ``warm_paged_mqa_logits``.
==================================  ===========================================

W (the aux graph page-table width) is ``ceil(L / page)`` for the deployment
context L, which ``prepare_deployment`` verifies only after this prewarm runs;
the witnesses therefore cover every width up to ``ceil(L_max / page)`` rather
than one exact width, so a verified L at or below the bound is covered.

The aux cache is DeepGEMM's page-split FP8 layout, whose scorer accepts only
``PAGED_MQA_BLOCK_KV``-token pages, and the dual coordinator asserts equal
primary and aux page sizes; a deployment page size other than that is refused
here (``KimiDsaPrewarmError``), before readiness, rather than at the first
long decode step.

Importable without Triton, DeepGEMM or CUDA: every GPU dependency is imported
inside the runner that launches it, and every runner takes the serving
function as an injectable ``fn`` for the CPU tests.
"""

from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Optional, Tuple

import torch

from batchgen_kernels.triton_prewarm import PrewarmEntry, int_class
from batchgen.models.moonshotai.kimi_linear.jit_coverage import (
    SITE_DSA_PAGED_MQA,
    SITE_DSA_PAGED_MQA_PREPARE,
    SITE_DSA_PREFILL_SELECT,
    SITE_DSA_SELECTED_POSITIONS,
)

# Row counts of the prefill scorer call: "1", "d16" and "any" for the keys'
# row count (``rows``); the queries' row count ``rows * H`` follows.
PREFILL_ROW_WITNESSES = (1, 16, 17)

# ``paged_mqa_logits.PAGED_MQA_BLOCK_KV`` / ``PAGED_MQA_NEXT_N`` /
# ``indexer_k_quant.SCALE_BYTES_PER_TOKEN`` (the CPU tests pin the three to
# the kernel modules, which import Triton).
PAGED_MQA_BLOCK_KV = 64
PAGED_MQA_NEXT_N = 1
PAGE_SPLIT_SCALE_BYTES = 4

# ``prepare_paged_mqa_inputs_out``'s gather tile cap.
PREPARE_MAX_BLOCK_P = 256

# DeepGEMM's paged-MQA metadata template rounds the batch up to this.
PAGED_MQA_BATCH_ALIGN = 32


class KimiDsaPrewarmError(RuntimeError):
    """The deployment cannot run the Kimi-K3.1 DSA kernels being prewarmed."""


# ============================================================================
#  Shapes and key classes (pure)
# ============================================================================


def next_power_of_2(value: int) -> int:
    """``triton.next_power_of_2`` for positive ints."""
    value = int(value)
    if value < 1:
        raise ValueError(f"next_power_of_2 needs a positive int, got {value}")
    return 1 << (value - 1).bit_length()


@dataclass(frozen=True)
class KimiDsaPrewarmShapes:
    """What selects the DSA kernels' keys: the indexer geometry from the
    model config and the decode deployment's page size, context bound and
    per-rank decode row cap."""

    index_n_heads: int
    index_head_dim: int
    index_topk: int
    page_size: int
    max_model_len: int
    max_decode_rows: int

    @classmethod
    def from_config(cls, config, *, page_size: int, max_model_len: int,
                    max_decode_rows: int) -> "KimiDsaPrewarmShapes":
        """``config`` is a ``KimiLinearConfig`` whose ``dsa()`` is not None.

        The indexer is replicated across attention TP
        (``tp_weight_sharding``), so its head count is the config's.
        """
        dsa = config.dsa()
        if dsa is None:
            raise KimiDsaPrewarmError(
                "Kimi-K3.1 DSA prewarm needs a config with dsa_config"
            )
        shapes = cls(
            index_n_heads=int(dsa.index_num_heads),
            index_head_dim=int(dsa.index_head_dim),
            index_topk=int(dsa.index_topk),
            page_size=int(page_size),
            max_model_len=int(max_model_len),
            max_decode_rows=int(max_decode_rows),
        )
        shapes.validate()
        return shapes

    def validate(self) -> None:
        if self.page_size != PAGED_MQA_BLOCK_KV:
            raise KimiDsaPrewarmError(
                f"Kimi-K3.1 DSA decode scores a page-split FP8 aux cache "
                f"whose scorer takes {PAGED_MQA_BLOCK_KV}-token pages only, "
                f"and the aux page size equals the primary one; this "
                f"deployment's page size is {self.page_size}"
            )
        for name in ("index_n_heads", "index_head_dim", "index_topk",
                     "max_model_len", "max_decode_rows"):
            if int(getattr(self, name)) <= 0:
                raise KimiDsaPrewarmError(
                    f"Kimi-K3.1 DSA prewarm: {name}={getattr(self, name)} "
                    "must be positive"
                )

    @property
    def max_aux_pages(self) -> int:
        """Upper bound of the aux graph page-table width W (the KV page
        table's ``ceil(max_model_len / page_size)``)."""
        return -(-self.max_model_len // self.page_size)

    @property
    def aux_carrier_bytes(self) -> int:
        return self.index_head_dim + PAGE_SPLIT_SCALE_BYTES


def prepare_key(width: int, page_size: int) -> tuple:
    """What selects a ``_prepare_paged_mqa_inputs_kernel`` binary (graph
    scorer, valid count always given): the page size (and the stage
    alignment it fixes), the tile ``BLOCK_P`` and the class of the width,
    which is the page-table and block-table row stride and ``max_pages``."""
    return (int(page_size), min(PREPARE_MAX_BLOCK_P, next_power_of_2(width)),
            int_class(width))


def aux_width_witnesses(max_width: int, page_size: int) -> List[int]:
    """The smallest aux graph width of each ``prepare_key`` in
    ``[1, max_width]``."""
    chosen = {}
    for width in range(1, int(max_width) + 1):
        chosen.setdefault(prepare_key(width, page_size), width)
    return sorted(chosen.values())


def graph_score_widths(index_topk: int, page_size: int,
                       max_width: int) -> List[int]:
    """Every top-k width a captured DSA step can run the selection
    transform at: ``dsa_decode.graph_score_width(index_topk, W * page)`` for
    W in ``[1, max_width]`` (the graph's context capacity is the aux graph
    table's ``W * page`` tokens)."""
    topk = int(index_topk)
    widths = set()
    for width in range(1, int(max_width) + 1):
        capacity = width * int(page_size)
        widths.add(min(topk, capacity))
        if capacity >= topk:
            break
    return sorted(widths)


def transform_key(width: int, page_size: int, has_valid: bool) -> tuple:
    """``_transform_selected_positions_kernel``: the constexprs (the top-k
    width, the primary page size, the two ``has_*`` flags) and the index
    dtype (int64 on both Kimi steps); the table width is not specialised."""
    return (int(width), int(page_size), False, bool(has_valid), "int64")


def paged_mqa_batches(max_rows: int) -> List[int]:
    """One batch per ``align(batch, 32)`` value up to ``max_rows``."""
    align = PAGED_MQA_BATCH_ALIGN
    top = (int(max_rows) + align - 1) // align * align
    return list(range(align, top + 1, align))


def act_quant_key(rows: int, dim: int) -> tuple:
    """``_per_token_blocked_quantize_1d_kernel`` for a plain ``act_quant``
    of a contiguous ``[rows, dim]`` input: the width fixes the constexprs and
    strides, the scales are ``[rows, ceil(dim / 128)]`` (their row stride is
    1 at ``dim <= 128``), and the row count's class selects the rest."""
    blocks = -(-int(dim) // 128)
    return (int(dim), int_class(rows), int_class(blocks), "1")


def prefill_select_quant_keys(shapes: KimiDsaPrewarmShapes, rows: int) -> set:
    """The two casts of one ``_score_and_select_prefill`` call at ``rows``
    query tokens (one score chunk): the keys ``[rows, D]`` and the queries
    ``[rows, H, D]`` viewed as ``[rows * H, D]``."""
    dim = shapes.index_head_dim
    return {act_quant_key(rows, dim),
            act_quant_key(rows * shapes.index_n_heads, dim)}


# ============================================================================
#  Entry runners: fresh tensors through the serving functions
# ============================================================================


def _valid_count(rows: int, device):
    return torch.full((1,), int(rows), dtype=torch.int32, device=device)


def _run_prefill_select(shapes: KimiDsaPrewarmShapes, rows: int, device,
                        fn: Optional[Callable] = None) -> None:
    """``dsa_sparse_prefill._score_and_select_prefill`` as the packed sparse
    prefill calls it: bf16 ``q_idx [T, H, D]`` and ``index_keys [T, D]``,
    fp32 head gates ``[T, H]``, int32 packed causal ranges, ``max_seqlen`` the
    packed length. Reaches both ``act_quant`` casts, ``fp8_mqa_logits`` and
    ``fast_topk_2048_out`` (AOT)."""
    if fn is None:
        from batchgen.models.moonshotai.kimi_linear.dsa_sparse_prefill import (
            _prefill_index_topk,
        )
        fn = _prefill_index_topk()
    rows = int(rows)
    heads, dim = shapes.index_n_heads, shapes.index_head_dim
    fn(
        q_idx=torch.zeros((rows, heads, dim), dtype=torch.bfloat16,
                          device=device),
        index_keys=torch.zeros((rows, dim), dtype=torch.bfloat16,
                               device=device),
        head_gates=torch.zeros((rows, heads), dtype=torch.float32,
                               device=device),
        causal_starts=torch.zeros((rows,), dtype=torch.int32, device=device),
        causal_ends=torch.arange(1, rows + 1, dtype=torch.int32,
                                 device=device),
        max_seqlen=rows,
        topk=shapes.index_topk,
    )


def _run_selected_positions(width: int, page_size: int, has_valid: bool,
                            device, fn: Optional[Callable] = None) -> None:
    """``transform_selected_positions_out`` as ``dsa_decode.
    sparse_mla_attention`` calls it: int32 page table and lengths, int64
    selected positions, no slot indices; the graph step adds the valid
    count. One row, one table column (the width is not specialised)."""
    if fn is None:
        from batchgen_kernels.attention.dsa.selected_page_table import (
            transform_selected_positions_out as fn,
        )
    fn(
        torch.zeros((1, 1), dtype=torch.int32, device=device),
        torch.zeros((1,), dtype=torch.int32, device=device),
        torch.zeros((1, int(width)), dtype=torch.int64, device=device),
        torch.empty((1, int(width)), dtype=torch.int32, device=device),
        torch.empty((1,), dtype=torch.int32, device=device),
        page_size=int(page_size),
        num_valid_tokens=_valid_count(1, device) if has_valid else None,
    )


def _run_paged_mqa_prepare(width: int, page_size: int, device,
                           fn: Optional[Callable] = None) -> None:
    """The graph scorer's prologue launch (``prepare_paged_mqa_inputs_out``
    inside ``paged_mqa_score_and_topk_out``) at aux table width ``width``:
    the static aux table, the identity slot map, the valid count."""
    if fn is None:
        from batchgen_kernels.attention.dsa.paged_mqa_scorer import (
            prepare_paged_mqa_inputs_out as fn,
        )
    width = int(width)
    fn(
        torch.zeros((1, width), dtype=torch.int32, device=device),
        torch.zeros((1,), dtype=torch.int32, device=device),
        torch.ones((1,), dtype=torch.int32, device=device),
        torch.empty((1, width), dtype=torch.int32, device=device),
        torch.empty((1, PAGED_MQA_NEXT_N), dtype=torch.int32, device=device),
        num_valid_tokens=_valid_count(1, device),
        deep_gemm_context_lens_out=torch.empty(
            (1, PAGED_MQA_NEXT_N), dtype=torch.int32, device=device),
        page_size=int(page_size),
    )


def _run_paged_mqa(shapes: KimiDsaPrewarmShapes, batch: int, device,
                   fn: Optional[Callable] = None) -> None:
    """``get_paged_mqa_logits_metadata`` and ``fp8_paged_mqa_logits`` at one
    ``align(batch, 32)`` through the scorer's own warm helper, on a one-page
    page-split scratch cache (the templates key on the batch alignment and
    the model's heads/head_dim/page size, not on the cache or context)."""
    if fn is None:
        from batchgen_kernels.attention.dsa.paged_mqa_scorer import (
            warm_paged_mqa_logits as fn,
        )
    page = shapes.page_size
    cache = torch.zeros((1, page, 1, shapes.aux_carrier_bytes),
                        dtype=torch.uint8, device=device)
    fn(aux_blocked_k=cache, batch=int(batch),
       num_heads=shapes.index_n_heads, index_head_dim=shapes.index_head_dim,
       max_pages_per_seq=1, max_seqlen=page, page_size=page)


# ============================================================================
#  Entry lists (deterministic: identical on every rank)
# ============================================================================


def prefill_entries(shapes: KimiDsaPrewarmShapes, device) -> List[PrewarmEntry]:
    return [
        PrewarmEntry(f"{SITE_DSA_PREFILL_SELECT}/t{rows}",
                     partial(_run_prefill_select, shapes, rows, device))
        for rows in PREFILL_ROW_WITNESSES
    ]


def decode_entries(shapes: KimiDsaPrewarmShapes, device) -> List[PrewarmEntry]:
    """Eager and graph decode: the selection transform, the graph scorer's
    prologue and the DeepGEMM paged scorer."""
    page, topk = shapes.page_size, shapes.index_topk
    entries = [PrewarmEntry(
        f"{SITE_DSA_SELECTED_POSITIONS}/eager/k{topk}",
        partial(_run_selected_positions, topk, page, False, device),
    )]
    entries.extend(
        PrewarmEntry(f"{SITE_DSA_SELECTED_POSITIONS}/graph/k{width}",
                     partial(_run_selected_positions, width, page, True,
                             device))
        for width in graph_score_widths(topk, page, shapes.max_aux_pages)
    )
    entries.extend(
        PrewarmEntry(f"{SITE_DSA_PAGED_MQA_PREPARE}/w{width}",
                     partial(_run_paged_mqa_prepare, width, page, device))
        for width in aux_width_witnesses(shapes.max_aux_pages, page)
    )
    entries.extend(
        PrewarmEntry(f"{SITE_DSA_PAGED_MQA}/b{batch}",
                     partial(_run_paged_mqa, shapes, batch, device))
        for batch in paged_mqa_batches(shapes.max_decode_rows)
    )
    return entries


def dsa_entries(shapes: KimiDsaPrewarmShapes, device) -> List[PrewarmEntry]:
    return prefill_entries(shapes, device) + decode_entries(shapes, device)


@dataclass(frozen=True)
class KimiDsaDeployment:
    """The decode deployment facts the DSA keys depend on, as the strategy
    manager reads them before readiness."""

    page_size: int
    max_model_len: int
    max_decode_rows: int


def planned_kernel_sites(entries) -> Tuple[str, ...]:
    """The entry-name prefixes (``jit_coverage`` site names) of a list."""
    return tuple(sorted({entry.name.split("/", 1)[0] for entry in entries}))
