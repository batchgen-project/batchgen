"""Contract of the DeepGEMM paged-MQA indexer scorer.

``batchgen_kernels/attention/dsa/paged_mqa_logits.py`` is the torch-only half
of the swap from the bf16 Triton scorer to
``deep_gemm.fp8_paged_mqa_logits``: it holds the shape/dtype/stride contract,
the logits row-stride trap, the top-k masking contract and the torch
references the GPU kernels are measured against. All of that is checked here on
CPU, so a dev run catches a contract regression without a GPU, Triton or
DeepGEMM.

The module's PACKAGE (``batchgen_kernels.attention.dsa``) imports Triton-backed
kernels unguarded on purpose, so on a Triton-less host the torch-only module is
loaded from its file with a temporary package stub, and the stub is removed
again so nothing else in the session sees it.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

import pytest
import torch


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_DSA_DIR = _REPO_ROOT / "batchgen_kernels" / "attention" / "dsa"
_PKG = "batchgen_kernels.attention.dsa"
_MODULE_PATH = _DSA_DIR / "paged_mqa_logits.py"


def _load_contract_module():
    name = "_paged_mqa_logits_contract"
    if name in sys.modules:
        return sys.modules[name]
    stub_installed = False
    if _PKG not in sys.modules:
        try:
            importlib.import_module(_PKG)
        except Exception:
            stub = types.ModuleType(_PKG)
            stub.__path__ = [str(_DSA_DIR)]
            sys.modules[_PKG] = stub
            stub_installed = True
    try:
        spec = importlib.util.spec_from_file_location(name, _MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        # Registered before exec because @dataclass resolves its own module.
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        if stub_installed:
            sys.modules.pop(_PKG, None)
            sys.modules.pop(f"{_PKG}.indexer_k_quant", None)
    return module


_pm = _load_contract_module()

HEADS = 32
HEAD_DIM = 128
PAGE_SIZE = 64
SCALE_BYTES = 4
CARRIER = HEAD_DIM + SCALE_BYTES


def _page_split_cache(num_pages: int = 8) -> torch.Tensor:
    return torch.zeros(
        num_pages, PAGE_SIZE, 1, CARRIER, dtype=torch.float8_e4m3fn
    )


def _aos_cache(num_pages: int = 8) -> torch.Tensor:
    return torch.zeros(num_pages, PAGE_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)


def _plan_kwargs(batch: int = 4, max_seqlen: int = 4096, **overrides):
    kwargs = dict(
        q=torch.zeros(batch, HEADS, HEAD_DIM, dtype=torch.bfloat16),
        aux_blocked_k=_page_split_cache(),
        head_gates=torch.zeros(batch, HEADS, dtype=torch.float32),
        cache_seqlens=torch.zeros(batch, dtype=torch.int32),
        max_pages_per_seq=(max_seqlen + PAGE_SIZE - 1) // PAGE_SIZE,
        topk=2048,
        max_seqlen=max_seqlen,
        index_head_dim=HEAD_DIM,
        page_size=PAGE_SIZE,
    )
    kwargs.update(overrides)
    return kwargs


# --------------------------------------------------------------------------- #
# The row-stride trap
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "max_seqlen,expected",
    [(256, 256), (257, 512), (4096, 4096), (10_240, 10_240), (10_241, 10_496)],
)
def test_align_context_len_rounds_up_to_the_logits_split(max_seqlen, expected):
    assert _pm.align_context_len(max_seqlen) == expected
    assert _pm.align_context_len(max_seqlen) % _pm.PAGED_MQA_LOGITS_ALIGN == 0
    assert _pm.align_context_len(max_seqlen) >= max_seqlen


def test_align_context_len_rejects_non_positive():
    with pytest.raises(ValueError, match="must be positive"):
        _pm.align_context_len(0)


def test_check_logits_layout_accepts_the_aligned_slice():
    """DeepGEMM's own allocation shape, for a 256-aligned max_context_len."""
    plan = _pm.plan_paged_mqa_logits(**_plan_kwargs(batch=4, max_seqlen=4096))
    assert plan.max_context_len == 4096
    logits = torch.empty(4, 4096, dtype=torch.float32)
    _pm.check_logits_layout(logits, plan)


def test_check_logits_layout_catches_a_slice_whose_row_stride_is_the_aligned_value():
    """The trap: DeepGEMM slices a 256-aligned allocation, so a non-aligned
    ``max_context_len`` leaves rows strided by the ALIGNED width. The top-k
    reads row b at ``stride(0) * b``, so this must fail rather than misread."""
    plan = _pm.plan_paged_mqa_logits(**_plan_kwargs(batch=4, max_seqlen=4096))
    sliced = torch.empty(4, 4352, dtype=torch.float32)[:, :4096]
    assert sliced.shape == plan.logits_shape
    assert sliced.stride(0) == 4352
    with pytest.raises(RuntimeError, match="row stride"):
        _pm.check_logits_layout(sliced, plan)


def test_check_logits_layout_catches_a_wrong_shape_and_dtype():
    plan = _pm.plan_paged_mqa_logits(**_plan_kwargs(batch=4, max_seqlen=4096))
    with pytest.raises(RuntimeError, match="shape"):
        _pm.check_logits_layout(torch.empty(4, 2048, dtype=torch.float32), plan)
    with pytest.raises(RuntimeError, match="float32"):
        _pm.check_logits_layout(torch.empty(4, 4096, dtype=torch.bfloat16), plan)


# --------------------------------------------------------------------------- #
# The format gate
# --------------------------------------------------------------------------- #

def test_require_page_split_aux_cache_accepts_the_glm52_carrier():
    _pm.require_page_split_aux_cache(_page_split_cache(), HEAD_DIM)


def test_require_page_split_aux_cache_names_an_aos_cache():
    with pytest.raises(RuntimeError, match="page-split FP8 aux indexer cache"):
        _pm.require_page_split_aux_cache(_aos_cache(), HEAD_DIM)


def test_require_page_split_aux_cache_names_a_missized_one_byte_cache():
    missized = torch.zeros(4, PAGE_SIZE, 1, HEAD_DIM, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="must be page-split"):
        _pm.require_page_split_aux_cache(missized, HEAD_DIM)


# --------------------------------------------------------------------------- #
# plan_paged_mqa_logits: DeepGEMM's host asserts, up front and named
# --------------------------------------------------------------------------- #

def test_plan_accepts_the_glm52_geometry():
    plan = _pm.plan_paged_mqa_logits(**_plan_kwargs(batch=8, max_seqlen=10_240))
    assert plan.batch == 8
    assert plan.next_n == 1
    assert plan.num_heads == HEADS
    assert plan.head_dim == HEAD_DIM
    assert plan.page_size == _pm.PAGED_MQA_BLOCK_KV == 64
    assert plan.max_context_len == 10_240
    assert plan.q_shape == (8, 1, HEADS, HEAD_DIM)
    assert plan.logits_shape == (8, 10_240)


def test_plan_rejects_an_aos_cache():
    with pytest.raises(RuntimeError, match="page-split FP8 aux indexer cache"):
        _pm.plan_paged_mqa_logits(**_plan_kwargs(aux_blocked_k=_aos_cache()))


def test_plan_rejects_a_page_size_other_than_64():
    cache = torch.zeros(4, 32, 1, CARRIER, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="block_kv == 64"):
        _pm.plan_paged_mqa_logits(
            **_plan_kwargs(aux_blocked_k=cache, page_size=32)
        )


def test_plan_rejects_a_head_count_that_does_not_divide_128():
    with pytest.raises(ValueError, match="128 % num_heads"):
        _pm.plan_paged_mqa_logits(
            **_plan_kwargs(
                q=torch.zeros(4, 24, HEAD_DIM, dtype=torch.bfloat16),
                head_gates=torch.zeros(4, 24, dtype=torch.float32),
            )
        )


def test_plan_rejects_a_head_dim_mismatch_against_the_cache():
    with pytest.raises(ValueError, match="head_dim"):
        _pm.plan_paged_mqa_logits(
            **_plan_kwargs(q=torch.zeros(4, HEADS, 64, dtype=torch.bfloat16))
        )


def test_plan_rejects_non_fp32_or_strided_head_gates():
    with pytest.raises(TypeError, match="float32"):
        _pm.plan_paged_mqa_logits(
            **_plan_kwargs(head_gates=torch.zeros(4, HEADS, dtype=torch.bfloat16))
        )
    strided = torch.zeros(4, 2 * HEADS, dtype=torch.float32)[:, :HEADS]
    with pytest.raises(ValueError, match="contiguous"):
        _pm.plan_paged_mqa_logits(**_plan_kwargs(head_gates=strided))


def test_plan_rejects_non_int32_cache_seqlens():
    with pytest.raises(TypeError, match="int32"):
        _pm.plan_paged_mqa_logits(
            **_plan_kwargs(cache_seqlens=torch.zeros(4, dtype=torch.int64))
        )


def test_plan_rejects_topk_above_max_seqlen():
    with pytest.raises(ValueError, match="exceeds max_seqlen"):
        _pm.plan_paged_mqa_logits(**_plan_kwargs(max_seqlen=1024, topk=2048))


def test_plan_rejects_a_page_table_too_narrow_for_max_seqlen():
    with pytest.raises(ValueError, match="page table has only"):
        _pm.plan_paged_mqa_logits(
            **_plan_kwargs(max_seqlen=4096, max_pages_per_seq=8)
        )


def test_plan_rejects_a_cache_whose_token_records_are_strided():
    """DeepGEMM re-views the cache as (e4m3 values, fp32 scales) from
    ``stride(0)``, so the two inner strides must be the per-token carrier."""
    wide = torch.zeros(4, PAGE_SIZE, 1, 2 * CARRIER, dtype=torch.float8_e4m3fn)
    strided = wide[..., :CARRIER]
    assert strided.shape[-1] == CARRIER
    with pytest.raises(ValueError, match="strides"):
        _pm.plan_paged_mqa_logits(**_plan_kwargs(aux_blocked_k=strided))


def test_plan_rejects_more_than_one_kv_head():
    cache = torch.zeros(4, PAGE_SIZE, 2, CARRIER, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="one KV head"):
        _pm.plan_paged_mqa_logits(**_plan_kwargs(aux_blocked_k=cache))


# --------------------------------------------------------------------------- #
# q quantization
# --------------------------------------------------------------------------- #

def test_quantize_q_e4m3_reference_shape_is_deepgemms_next_n_layout():
    q = torch.randn(5, HEADS, HEAD_DIM, dtype=torch.bfloat16)
    q_fp8 = _pm.quantize_q_e4m3_reference(q)
    assert q_fp8.shape == (5, 1, HEADS, HEAD_DIM)
    assert q_fp8.dtype == torch.float8_e4m3fn
    assert q_fp8.is_contiguous()


def test_quantize_q_e4m3_reference_is_an_explicitly_saturating_cast():
    """No scale channel exists on DeepGEMM's q, so the conversion is the plain
    cast its own reference test feeds the kernel -- with the out-of-range
    behaviour made explicit: a bare ``.to(float8_e4m3fn)`` gives NaN for
    ``|x| > 448`` on torch 2.9 CPU, so the reference clamps to +-448 first
    (matching the .cu K writer's SATFINITE) and must never produce NaN/inf."""
    q = torch.tensor(
        [[[0.5, -0.25, 1000.0, 1e-5, -1000.0, float("inf"), float("-inf"), 448.0]]],
        dtype=torch.bfloat16,
    )
    out = _pm.quantize_q_e4m3_reference(q).float().reshape(-1)
    assert out[0] == 0.5
    assert out[1] == -0.25
    assert out[2] == 448.0  # saturates: neither NaN nor inf
    assert out[3] == 0.0  # below the smallest subnormal
    assert out[4] == -448.0
    assert out[5] == 448.0
    assert out[6] == -448.0
    assert out[7] == 448.0  # the boundary itself is exact
    assert torch.isfinite(out).all()


def test_saturate_q_to_e4m3_range_is_the_shared_production_clamp():
    """The production q-casts must go through the same helper the reference
    uses, so the two cannot drift on out-of-range inputs."""
    src = (
        pathlib.Path(__file__).resolve().parents[1]
        / "batchgen_kernels" / "attention" / "dsa" / "paged_mqa_scorer.py"
    ).read_text()
    # Every e4m3 q materialization in the scorer is clamped first: the eager
    # cast and the graph path's copy into the static buffer.
    assert src.count("saturate_q_to_e4m3_range(q)") == 2
    assert "q.to(torch.float8_e4m3fn)" not in src
    assert ".copy_(q)" not in src


# --------------------------------------------------------------------------- #
# Per-step input preparation
# --------------------------------------------------------------------------- #

def test_prepare_reference_gathers_page_table_rows_into_batch_order():
    page_table = torch.tensor(
        [[10, 11, -1], [20, 21, 22], [30, -1, -1], [40, 41, -1]],
        dtype=torch.int32,
    )
    slots = torch.tensor([2, 0, 3], dtype=torch.int32)
    seqlens = torch.tensor([65, 130, 70], dtype=torch.int32)
    block_table, context_lens = _pm.prepare_paged_mqa_inputs_reference(
        page_table, slots, seqlens
    )
    assert torch.equal(
        block_table,
        torch.tensor([[30, -1, -1], [10, 11, -1], [40, 41, -1]], dtype=torch.int32),
    )
    assert torch.equal(context_lens, torch.tensor([[65], [130], [70]], dtype=torch.int32))
    assert block_table.stride(1) == 1
    assert context_lens.is_contiguous()


def test_prepare_reference_zeroes_context_len_for_a_slotless_row():
    """A row with no cache slot must read nothing: DeepGEMM visits no block at
    context_len 0, and the top-k emits -1 padding for it."""
    page_table = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
    slots = torch.tensor([1, -1], dtype=torch.int32)
    seqlens = torch.tensor([128, 999], dtype=torch.int32)
    block_table, context_lens = _pm.prepare_paged_mqa_inputs_reference(
        page_table, slots, seqlens
    )
    assert torch.equal(context_lens, torch.tensor([[128], [0]], dtype=torch.int32))
    # A row that reads nothing gets the defined all-page-0 row, never a
    # borrowed (clamped-slot) row of another sequence.
    assert torch.equal(block_table[1], torch.zeros(2, dtype=torch.int32))
    assert torch.equal(block_table[0], page_table[1])


def test_prepare_reference_zeroes_context_len_past_num_valid_tokens():
    page_table = torch.tensor([[10, 11], [20, 21], [30, 31]], dtype=torch.int32)
    slots = torch.tensor([0, 1, 2], dtype=torch.int32)
    seqlens = torch.tensor([100, 200, 300], dtype=torch.int32)
    _, context_lens = _pm.prepare_paged_mqa_inputs_reference(
        page_table, slots, seqlens, torch.tensor([2], dtype=torch.int32)
    )
    assert torch.equal(
        context_lens, torch.tensor([[100], [200], [0]], dtype=torch.int32)
    )


def test_prepare_reference_passes_unmapped_page_sentinels_through():
    """``-1`` is the page table's 'not allocated' fill. It is unreachable
    inside a row's context_len, and DeepGEMM's TMA zero-fills an out-of-range
    page -- the same value the retired Triton scorer produced."""
    page_table = torch.tensor([[7, -1, -1]], dtype=torch.int32)
    block_table, _ = _pm.prepare_paged_mqa_inputs_reference(
        page_table,
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([64], dtype=torch.int32),
    )
    assert torch.equal(block_table, page_table)


def test_prepare_reference_writes_page_zero_rows_for_every_row_that_reads_nothing():
    """Slotless, past-num_valid_tokens and live-but-empty rows all get a
    DEFINED all-page-0 table row, and a padded row's slot -- which may be any
    value, even past the page table -- is never dereferenced."""
    page_table = torch.tensor(
        [[10, 11, -1], [20, 21, 22], [30, -1, -1]], dtype=torch.int32
    )
    slots = torch.tensor([1, -1, 2, 0, 10**6], dtype=torch.int32)
    seqlens = torch.tensor([130, 64, 0, 65, 64], dtype=torch.int32)
    block_table, context_lens = _pm.prepare_paged_mqa_inputs_reference(
        page_table, slots, seqlens, torch.tensor([4], dtype=torch.int32)
    )
    assert torch.equal(
        context_lens, torch.tensor([[130], [0], [0], [65], [0]], dtype=torch.int32)
    )
    zeros = torch.zeros(3, dtype=torch.int32)
    assert torch.equal(block_table[0], page_table[1])  # live, reads
    assert torch.equal(block_table[1], zeros)  # slotless
    assert torch.equal(block_table[2], zeros)  # live, seqlen 0
    assert torch.equal(block_table[3], page_table[0])  # live, reads
    assert torch.equal(block_table[4], zeros)  # past num_valid, garbage slot


def test_prepare_reference_rejects_a_non_int32_page_table():
    with pytest.raises(TypeError, match="int32"):
        _pm.prepare_paged_mqa_inputs_reference(
            torch.zeros(2, 2, dtype=torch.int64),
            torch.zeros(2, dtype=torch.int32),
            torch.zeros(2, dtype=torch.int32),
        )


# --------------------------------------------------------------------------- #
# The never-all-zero guard (DeepGEMM sm90 metadata scheduler defect)
# --------------------------------------------------------------------------- #

def _sm90_metadata_reads_in_bounds(
    lens: list[int], num_sms: int, split_kv: int = 256
) -> bool:
    """CPU model of DeepGEMM's sm90 NON-varlen ``paged_mqa_logits_metadata``.

    Transcribed from the scheduler's work-distribution branch: a 32-lane
    prefix sum over ``align(batch, 32)`` shared entries (padding rows count 0
    segments), then per SM a binary search for its segment start over
    ``[0, batch)`` followed by reads of ``prefix_sum[q_idx]`` and
    ``prefix_sum[q_idx - 1]``. Returns whether every shared read stays inside
    the ``align(batch, 32)`` buffer. Only the zero/non-zero segment count of a
    row matters, so ``split_kv`` and ``num_sms`` do not change the verdict.
    """

    batch = len(lens)
    aligned = (batch + 31) // 32 * 32
    segs = [(int(n) + split_kv - 1) // split_kv for n in lens] + [0] * (aligned - batch)
    prefix, running = [], 0
    for s in segs:
        running += s
        prefix.append(running)
    q, r = divmod(running, num_sms)
    pivot = num_sms - r
    reads: set[int] = set()
    for sm_idx in range(num_sms):
        seg_starts = sm_idx * q + (sm_idx - pivot if sm_idx > pivot else 0)
        lo, hi = 0, batch
        while lo < hi:
            mid = (lo + hi) // 2
            reads.add(mid)
            if prefix[mid] <= seg_starts:
                lo = mid + 1
            else:
                hi = mid
        reads.add(lo)  # num_segs_q = prefix_sum[q_idx] - prefix_sum[q_idx - 1]
        if lo > 0:
            reads.add(lo - 1)
    return max(reads) < aligned


# The node repro's CASE table (2026-10-17, sgl deep_gemm 0.1.5.post3, sm90):
# (batch, fill, rows set to 64, node verdict).
_NODE_CASES = [
    (128, 0, 0, False),
    (127, 0, 0, True),
    (96, 0, 0, False),
    (32, 0, 0, False),
    (128, 64, 0, True),
    (128, 0, 1, True),
    (127, 0, 1, True),
]


@pytest.mark.parametrize("num_sms", [78, 132])
@pytest.mark.parametrize("batch,fill,nonzero,node_ok", _NODE_CASES)
def test_scheduler_model_reproduces_the_node_case_table(
    num_sms, batch, fill, nonzero, node_ok
):
    """The CPU model must agree with every node-measured case, or the guard
    tests below would be checking the wrong defect."""
    lens = [fill] * batch
    for b in range(nonzero):
        lens[b] = 64
    assert _sm90_metadata_reads_in_bounds(lens, num_sms) is node_ok


_GUARD_BATCHES = [32, 64, 96, 127, 128]


def _prepared(batch: int, case: str):
    """Reference-prepared inputs for one guard case."""
    page_table = torch.arange(1, (batch + 1) * 4 + 1, dtype=torch.int32).reshape(
        batch + 1, 4
    )
    slots = torch.arange(batch, dtype=torch.int32)
    seqlens = torch.full((batch,), 200, dtype=torch.int32)
    nvt = None
    if case == "all_invalid_num_valid_0":  # a rank with nothing to decode
        nvt = torch.tensor([0], dtype=torch.int32)
    elif case == "all_invalid_slotless":
        slots = torch.full((batch,), -1, dtype=torch.int32)
    elif case == "all_zero_inputs":  # whole-model capture warmup
        slots = torch.zeros(batch, dtype=torch.int32)
        seqlens = torch.zeros(batch, dtype=torch.int32)
    elif case == "one_valid_last_row":
        slots[:-1] = -1
    elif case == "one_valid_row0":
        nvt = torch.tensor([1], dtype=torch.int32)
    else:
        raise AssertionError(case)
    block_table, true_lens = _pm.prepare_paged_mqa_inputs_reference(
        page_table, slots, seqlens, nvt
    )
    return block_table, true_lens


_GUARD_CASES = [
    "all_invalid_num_valid_0",
    "all_invalid_slotless",
    "all_zero_inputs",
    "one_valid_last_row",
    "one_valid_row0",
]


@pytest.mark.parametrize("num_sms", [78, 132])
@pytest.mark.parametrize("case", _GUARD_CASES)
@pytest.mark.parametrize("batch", _GUARD_BATCHES)
def test_lens_handed_to_deep_gemm_are_never_all_zero(batch, case, num_sms):
    block_table, true_lens = _prepared(batch, case)
    guarded = _pm.guard_deep_gemm_context_lens(true_lens)

    assert guarded.shape == true_lens.shape
    assert guarded.dtype == torch.int32 and guarded.is_contiguous()
    assert int(guarded.sum()) > 0
    assert int(guarded[0, 0]) == max(int(true_lens[0, 0]), 1)
    # Only row 0 may differ, and only upward: the top-k's TRUE bound never
    # exceeds what DeepGEMM computed.
    assert torch.equal(guarded[1:], true_lens[1:])
    assert bool((true_lens <= guarded).all())
    assert _sm90_metadata_reads_in_bounds(guarded.reshape(-1).tolist(), num_sms)

    if int(true_lens.sum()) == 0:
        # The case really is the defect's trigger whenever the batch is
        # 32-aligned -- the guard is what makes it safe.
        unguarded_ok = _sm90_metadata_reads_in_bounds(
            true_lens.reshape(-1).tolist(), num_sms
        )
        assert unguarded_ok is (batch % 32 != 0)
    if int(true_lens[0, 0]) == 0:
        # The forced one-token read of row 0 lands on page 0.
        assert int(block_table[0, 0]) == 0


def test_guard_returns_a_copy_and_leaves_the_true_lengths_alone():
    true_lens = torch.zeros(32, 1, dtype=torch.int32)
    guarded = _pm.guard_deep_gemm_context_lens(true_lens)
    assert int(true_lens.sum()) == 0
    assert guarded.data_ptr() != true_lens.data_ptr()
    live = torch.tensor([[5], [0], [7]], dtype=torch.int32)
    assert torch.equal(_pm.guard_deep_gemm_context_lens(live), live)


def test_guard_rejects_a_one_dimensional_or_non_int32_input():
    with pytest.raises(TypeError, match="2-D"):
        _pm.guard_deep_gemm_context_lens(torch.zeros(32, dtype=torch.int32))
    with pytest.raises(TypeError, match="int32"):
        _pm.guard_deep_gemm_context_lens(torch.zeros(32, 1, dtype=torch.int64))


@pytest.mark.parametrize("batch", _GUARD_BATCHES)
def test_topk_masks_by_the_true_lengths_so_a_forced_row_selects_nothing(batch):
    """DeepGEMM computed a score for row 0's forced token; the top-k must not
    select it, because it bounds rows by the TRUE lengths."""
    torch.manual_seed(2)
    topk = 16
    _, true_lens = _prepared(batch, "one_valid_last_row")
    guarded = _pm.guard_deep_gemm_context_lens(true_lens)
    assert int(true_lens[0, 0]) == 0 and int(guarded[0, 0]) == 1
    logits = torch.randn(batch, 256, dtype=torch.float32)
    logits[0, 0] = float("inf")  # whatever the forced read produced

    picked = _pm.topk_from_logits_reference(logits, true_lens.view(batch), topk)
    assert picked[0].tolist() == [-1] * topk
    assert bool((picked[1:-1] == -1).all())
    expected_last = torch.topk(logits[-1, :200], topk).indices.to(torch.int32)
    assert torch.equal(picked[-1], expected_last)
    # Masking by the guarded copy instead WOULD leak the forced token.
    leaked = _pm.topk_from_logits_reference(logits, guarded.view(batch), topk)
    assert leaked[0, 0].item() == 0


# --------------------------------------------------------------------------- #
# THE masking contract: DeepGEMM does not mask, the top-k does
# --------------------------------------------------------------------------- #

def test_topk_reference_never_reads_past_a_rows_length():
    """The load-bearing property of the whole swap.

    ``clean_logits=True`` is unreachable for the 2-D ``context_lens`` form, so
    everything past a row's ``context_len`` is whatever the allocator handed
    DeepGEMM. Filling that tail with the largest possible scores must not
    change a single selected index.
    """
    torch.manual_seed(0)
    batch, width, topk = 3, 4096, 2048
    lengths = torch.tensor([3000, 2500, 4096], dtype=torch.int32)
    clean = torch.randn(batch, width, dtype=torch.float32)
    for b in range(batch):
        clean[b, lengths[b]:] = float("-inf")
    poisoned = clean.clone()
    for b in range(batch):
        poisoned[b, lengths[b]:] = float("inf")

    picked_clean = _pm.topk_from_logits_reference(clean, lengths, topk)
    picked_poisoned = _pm.topk_from_logits_reference(poisoned, lengths, topk)
    assert torch.equal(picked_clean, picked_poisoned)
    # And nothing selected is out of range for its row.
    for b in range(batch):
        assert int(picked_poisoned[b].max()) < int(lengths[b])


def test_topk_reference_emits_the_dense_prefix_for_a_short_row():
    """A row at or under topk selects its whole context by position and reads
    no score at all -- which is why NaN garbage in a short row is harmless."""
    logits = torch.full((1, 64), float("nan"), dtype=torch.float32)
    picked = _pm.topk_from_logits_reference(
        logits, torch.tensor([10], dtype=torch.int32), 16
    )
    assert picked[0, :10].tolist() == list(range(10))
    assert picked[0, 10:].tolist() == [-1] * 6


def test_topk_reference_blanks_a_zero_length_row():
    logits = torch.randn(2, 128, dtype=torch.float32)
    picked = _pm.topk_from_logits_reference(
        logits, torch.tensor([0, 4], dtype=torch.int32), 4
    )
    assert picked[0].tolist() == [-1, -1, -1, -1]
    # length == topk is still the dense prefix, not a selection.
    assert picked[1].tolist() == [0, 1, 2, 3]


def test_topk_reference_blanks_rows_past_num_valid_tokens():
    logits = torch.randn(3, 128, dtype=torch.float32)
    picked = _pm.topk_from_logits_reference(
        logits,
        torch.tensor([64, 64, 64], dtype=torch.int32),
        4,
        num_valid_tokens=torch.tensor([2], dtype=torch.int32),
    )
    assert picked[2].tolist() == [-1, -1, -1, -1]
    assert picked[0].tolist() != [-1, -1, -1, -1]


def test_topk_reference_selects_the_exact_set_for_a_long_row():
    torch.manual_seed(1)
    logits = torch.randn(1, 512, dtype=torch.float32)
    picked = _pm.topk_from_logits_reference(
        logits, torch.tensor([300], dtype=torch.int32), 8
    )
    expected = torch.topk(logits[0, :300], 8).indices.to(torch.int32)
    assert torch.equal(picked[0], expected)


# --------------------------------------------------------------------------- #
# The Triton preparation pass must match its torch reference (needs a GPU)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU for Triton")
@pytest.mark.parametrize("num_valid", [None, 3])
def test_triton_prepare_matches_the_reference(num_valid):
    from batchgen_kernels.attention.dsa.paged_mqa_scorer import (
        prepare_paged_mqa_inputs_out,
    )

    device = torch.device("cuda")
    torch.manual_seed(0)
    num_slots, max_pages, batch = 9, 300, 5
    page_table = torch.randint(
        -1, 64, (num_slots, max_pages), dtype=torch.int32, device=device
    )
    slots = torch.tensor([4, -1, 0, 8, 2], dtype=torch.int32, device=device)
    seqlens = torch.tensor(
        [4096, 777, 64, 9999, 1], dtype=torch.int32, device=device
    )
    nvt = (
        None
        if num_valid is None
        else torch.tensor([num_valid], dtype=torch.int32, device=device)
    )

    # Poisoned outputs: every element must be WRITTEN, never left behind.
    block_table = torch.full((batch, max_pages), 12345, dtype=torch.int32, device=device)
    context_lens = torch.full((batch, 1), 12345, dtype=torch.int32, device=device)
    dg_lens = torch.full((batch, 1), 12345, dtype=torch.int32, device=device)
    prepare_paged_mqa_inputs_out(
        page_table,
        slots,
        seqlens,
        block_table,
        context_lens,
        num_valid_tokens=nvt,
        deep_gemm_context_lens_out=dg_lens,
    )
    ref_bt, ref_ctx = _pm.prepare_paged_mqa_inputs_reference(
        page_table, slots, seqlens, nvt
    )
    assert torch.equal(block_table, ref_bt)
    assert torch.equal(context_lens, ref_ctx)
    assert torch.equal(dg_lens, _pm.guard_deep_gemm_context_lens(ref_ctx))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU for Triton")
@pytest.mark.parametrize("case", _GUARD_CASES)
@pytest.mark.parametrize("batch", _GUARD_BATCHES)
def test_triton_prepare_guards_the_deep_gemm_lens(batch, case):
    """The fused guard in the Triton pass matches the torch guard, defines
    page-0 rows for rows that read nothing, and never hands DeepGEMM an
    all-zero length vector -- including at the 32-aligned buckets."""
    from batchgen_kernels.attention.dsa.paged_mqa_scorer import (
        prepare_paged_mqa_inputs_out,
    )

    device = torch.device("cuda")
    page_table = torch.arange(1, (batch + 1) * 4 + 1, dtype=torch.int32).reshape(
        batch + 1, 4
    )
    slots = torch.arange(batch, dtype=torch.int32)
    seqlens = torch.full((batch,), 200, dtype=torch.int32)
    nvt = None
    if case == "all_invalid_num_valid_0":
        nvt = torch.tensor([0], dtype=torch.int32)
        slots[:] = 10**6  # padded rows: the slot must never be dereferenced
    elif case == "all_invalid_slotless":
        slots[:] = -1
    elif case == "all_zero_inputs":
        slots.zero_()
        seqlens.zero_()
    elif case == "one_valid_last_row":
        slots[:-1] = -1
    elif case == "one_valid_row0":
        nvt = torch.tensor([1], dtype=torch.int32)
        slots[1:] = 10**6
    # The reference never dereferences a non-reading row's slot either.
    ref_bt, ref_ctx = _pm.prepare_paged_mqa_inputs_reference(
        page_table, slots, seqlens, nvt
    )

    block_table = torch.full((batch, 4), 12345, dtype=torch.int32, device=device)
    context_lens = torch.full((batch, 1), 12345, dtype=torch.int32, device=device)
    dg_lens = torch.full((batch, 1), 12345, dtype=torch.int32, device=device)
    prepare_paged_mqa_inputs_out(
        page_table.to(device),
        slots.to(device),
        seqlens.to(device),
        block_table,
        context_lens,
        num_valid_tokens=None if nvt is None else nvt.to(device),
        deep_gemm_context_lens_out=dg_lens,
    )
    assert torch.equal(block_table.cpu(), ref_bt)
    assert torch.equal(context_lens.cpu(), ref_ctx)
    assert torch.equal(dg_lens.cpu(), _pm.guard_deep_gemm_context_lens(ref_ctx))
    assert int(dg_lens.sum()) > 0
