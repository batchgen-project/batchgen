"""ReLU-gated fp32 oracle and the acceptance metrics for the indexer scorer.

Swapping the bf16 Triton scorer for ``deep_gemm.fp8_paged_mqa_logits`` changes
what the indexer COMPUTES -- ``sum_h w_h * relu(q_h . k_t)`` instead of
``sum_h w_h * (q_h . k_t)`` -- so exact top-k set equality is not the bar. Two
independent effects separate the new production path from the GLM-5
training-time formulation:

1. the formula change itself (decided: the ReLU form is the ground truth), and
2. FP8: DeepGEMM reads e4m3 ``q`` and the cache's per-token-scaled e4m3 ``K``,
   while the oracle is fp32. Only this second effect is the integration's to
   defend, and it is what the metrics here measure.

This file owns:

* ``relu_gated_oracle`` -- the fp32 ReLU-gated aggregate, masked structurally
  (a row's scores exist only over ``[0, context_len)``), which is also the
  masking contract the production top-k implements;
* ``no_relu_oracle`` -- the retired formula, so the size of the SEMANTICS
  change can be re-measured rather than quoted;
* ``selection_metrics`` -- top-k overlap, retained oracle score mass, and the
  near-tie concentration that distinguishes "the misses were near the
  selection threshold" from "the selection degraded";
* CPU tests of all three, which is what makes the metrics trustworthy before a
  GPU ever runs;
* a CUDA-gated harness test that drives the real production scorer and reports
  the metrics. Its assertion is gated on ``ACCEPTANCE``, which stays ``None``
  until the main loop approves the bound (see the proposal in the docstring of
  ``ACCEPTANCE_PROPOSAL``).
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys
import types
from dataclasses import dataclass

import pytest
import torch


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_DSA_DIR = _REPO_ROOT / "batchgen_kernels" / "attention" / "dsa"
_PKG = "batchgen_kernels.attention.dsa"

FP8_MAX = 448.0
AMAX_FLOOR = 1e-4
HEADS = 32
HEAD_DIM = 128
PAGE_SIZE = 64


def _load_torch_only(module_name: str, filename: str):
    """Load a torch-only DSA module without executing its Triton-importing package."""
    alias = f"_oracle_{module_name}"
    if alias in sys.modules:
        return sys.modules[alias]
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
        spec = importlib.util.spec_from_file_location(alias, _DSA_DIR / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
    finally:
        if stub_installed:
            sys.modules.pop(_PKG, None)
            sys.modules.pop(f"{_PKG}.indexer_k_quant", None)
    return module


_quant = _load_torch_only("indexer_k_quant", "indexer_k_quant.py")


# --------------------------------------------------------------------------- #
# Oracles
# --------------------------------------------------------------------------- #

def relu_gated_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    head_gates: torch.Tensor,
    context_lens: torch.Tensor,
) -> list[torch.Tensor]:
    """fp64 ``sum_h gates[b,h] * relu(q[b,h] . k[b,t])`` for ``t < len[b]``.

    Returns one 1-D tensor per row, of that row's own length. Rows are returned
    separately on purpose: a masked position must not exist at all. Padding
    with ``-inf`` and then applying ReLU would turn it into a score of exactly
    ``0``, which can outrank a real position whose aggregate is negative (head
    gates are unconstrained in sign), and the production top-k never reads past
    ``context_len`` either.

    ``k`` is ``[batch, max_len, head_dim]`` -- already gathered, because the
    oracle is about the FORMULA, not about paging.
    """

    rows: list[torch.Tensor] = []
    q64 = q.double()
    k64 = k.double()
    g64 = head_gates.double()
    for b in range(q.shape[0]):
        length = int(context_lens[b].item())
        if length <= 0:
            rows.append(torch.zeros(0, dtype=torch.float64, device=q.device))
            continue
        scores = torch.relu(q64[b] @ k64[b, :length].T)  # [heads, length]
        rows.append(g64[b] @ scores)
    return rows


def no_relu_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    head_gates: torch.Tensor,
    context_lens: torch.Tensor,
) -> list[torch.Tensor]:
    """The retired formula: the same sum WITHOUT the per-head ReLU."""

    rows: list[torch.Tensor] = []
    q64, k64, g64 = q.double(), k.double(), head_gates.double()
    for b in range(q.shape[0]):
        length = int(context_lens[b].item())
        if length <= 0:
            rows.append(torch.zeros(0, dtype=torch.float64, device=q.device))
            continue
        rows.append(g64[b] @ (q64[b] @ k64[b, :length].T))
    return rows


def quantize_k_like_the_cache(k: torch.Tensor) -> torch.Tensor:
    """Round-trip ``k`` through the aux cache's own e4m3 + per-token fp32 scale.

    This is what DeepGEMM actually reads, so an oracle comparison that skips it
    would measure a difference the production path does not have.
    """

    k_fp8, scales = _quant.quantize_indexer_k_fp8(k)
    return k_fp8.float() * scales.unsqueeze(-1).float()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SelectionMetrics:
    """Per-batch summary of a selection against the oracle.

    ``overlap``: ``|picked ∩ oracle_topk| / topk``, averaged over rows.

    ``score_mass``: ``sum(oracle[picked]) / sum(oracle[oracle_topk])``. This is
    the decision-relevant number: it says how much of the oracle's selected
    relevance the production selection actually captured.

    ``near_tie_concentration``: ``(1 - score_mass) / (1 - overlap)``. If the
    differing indices were ordinary members of the oracle's top-k, losing a
    fraction ``f`` of them would cost about ``f`` of the mass and this ratio
    would be ~1. A ratio near 0 is the quantitative statement that the
    disagreements sit at the selection threshold -- which is the only honest
    form of "near-tie analysis" when FP8 makes set equality unreachable.

    ``margin_fraction_within``: for each disagreeing pick, its distance below
    the oracle's k-th value normalized by the oracle's top-k spread
    (``best - threshold``); the dict gives the fraction of disagreements inside
    each relative band.
    """

    overlap_mean: float
    overlap_min: float
    score_mass_mean: float
    score_mass_min: float
    near_tie_concentration: float
    disagreements_per_row: float
    threshold_gap_mean: float
    margin_fraction_within: dict


def selection_metrics(
    picked: torch.Tensor,
    oracle_rows: list[torch.Tensor],
    topk: int,
    bands: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1),
) -> SelectionMetrics:
    """Compare a production selection against the oracle, row by row.

    ``picked`` is ``[batch, topk]`` int32/int64 with ``-1`` padding (the
    production top-k's own convention for short and invalid rows).
    """

    overlaps: list[float] = []
    masses: list[float] = []
    gaps: list[float] = []
    in_band = {b: 0 for b in bands}
    total_disagreements = 0
    rows_counted = 0

    for b, oracle in enumerate(oracle_rows):
        length = int(oracle.numel())
        if length <= topk:
            # Dense-prefix row: the selection is positional, so there is
            # nothing to compare -- any correct implementation agrees exactly.
            continue
        rows_counted += 1
        picks = picked[b]
        picks = picks[picks >= 0]
        sorted_oracle = torch.sort(oracle, descending=True)
        oracle_top = sorted_oracle.indices[:topk]
        threshold = float(sorted_oracle.values[topk - 1])
        nxt = float(sorted_oracle.values[topk])
        best = float(sorted_oracle.values[0])
        gaps.append(threshold - nxt)

        picked_set = set(int(i) for i in picks.tolist())
        oracle_set = set(int(i) for i in oracle_top.tolist())
        overlaps.append(len(picked_set & oracle_set) / topk)
        masses.append(
            float(oracle[list(picked_set)].sum() / oracle[oracle_top].sum())
        )

        spread = (best - threshold) or 1.0
        for idx in picked_set - oracle_set:
            total_disagreements += 1
            relative = (threshold - float(oracle[idx])) / spread
            for band in bands:
                if relative <= band:
                    in_band[band] += 1

    if rows_counted == 0:
        raise ValueError(
            "no row longer than topk: the oracle comparison needs rows that "
            "actually select, not dense-prefix rows"
        )
    overlap_mean = sum(overlaps) / len(overlaps)
    mass_mean = sum(masses) / len(masses)
    concentration = (
        (1.0 - mass_mean) / (1.0 - overlap_mean) if overlap_mean < 1.0 else 0.0
    )
    return SelectionMetrics(
        overlap_mean=overlap_mean,
        overlap_min=min(overlaps),
        score_mass_mean=mass_mean,
        score_mass_min=min(masses),
        near_tie_concentration=concentration,
        disagreements_per_row=total_disagreements / rows_counted,
        threshold_gap_mean=sum(gaps) / len(gaps),
        margin_fraction_within={
            band: (in_band[band] / total_disagreements if total_disagreements else 1.0)
            for band in bands
        },
    )


ACCEPTANCE_PROPOSAL = """Proposed acceptance bound (PENDING main-loop approval).

Measured on the dev machine (CPU, fp64 oracle, isotropic synthetic q/K/gates,
batch 4, topk 2048), FP8 inputs vs the fp32 ReLU-gated oracle:

    L= 4096  overlap mean 0.9858 min 0.9834  mass mean 0.99940 min 0.99907
    L= 8192  overlap mean 0.9789 min 0.9746  mass mean 0.99971 min 0.99912
    L=16384  overlap mean 0.9696 min 0.9673  mass mean 0.99793 min 0.99537
    L=32768  overlap mean 0.9680 min 0.9658  mass mean 0.99893 min 0.99849

    near-tie concentration (1-mass)/(1-overlap): 0.014 .. 0.069

The same harness reproduces the notes' no-ReLU-vs-ReLU figures
(83.8 / 72.5 / 64.9 / 60.0 % at the four lengths vs 83.0 / 71.9 / 66.2 / 61.6 %
in notes section 3.3), which is the independent check that it measures what it
claims to.

Proposal, to be applied against the fp32 ReLU-gated oracle on REAL GLM-5.2
decode activations:

    overlap_mean  >= 0.95      (headroom ~2 pts under the synthetic measurement)
    overlap_min   >= 0.92      (per row)
    score_mass_mean >= 0.995
    score_mass_min  >= 0.99    (per row)
    near_tie_concentration <= 0.15

Rationale: the overlap bound is the honest statement that FP8 cannot reproduce
the set, with enough headroom for real activations being less isotropic than
the synthetic study. The score-mass bound is what actually protects answer
quality, and the concentration ratio is the near-tie evidence: at overlap 0.97
with mass 0.998 the differing indices carry ~20x less oracle weight than an
average top-k member, i.e. they sit at the threshold. A regression that
dropped real candidates would move mass and concentration long before it moved
overlap.
"""

# Set by the main loop once the bound above is approved; until then the
# CUDA-gated test reports the metrics and skips the assertion.
ACCEPTANCE: dict | None = None


# --------------------------------------------------------------------------- #
# CPU tests of the oracle
# --------------------------------------------------------------------------- #

def test_relu_gated_oracle_matches_the_deepgemm_epilogue_formula():
    """``relu`` before head weighting, which is what DeepGEMM hardcodes."""
    torch.manual_seed(0)
    q = torch.randn(2, 4, 8, dtype=torch.float32)
    k = torch.randn(2, 16, 8, dtype=torch.float32)
    gates = torch.randn(2, 4, dtype=torch.float32)
    lens = torch.tensor([16, 10], dtype=torch.int32)
    rows = relu_gated_oracle(q, k, gates, lens)
    for b, length in enumerate((16, 10)):
        expected = torch.einsum(
            "ht,h->t",
            torch.relu(torch.einsum("hd,td->ht", q[b].double(), k[b, :length].double())),
            gates[b].double(),
        )
        assert rows[b].shape == (length,)
        torch.testing.assert_close(rows[b], expected)


def test_relu_gated_oracle_row_is_exactly_the_context_length():
    """A masked position must not exist: ReLU would turn a -inf pad into 0,
    which can outrank a real position whose aggregate is negative."""
    q = torch.zeros(1, 2, 4)
    k = torch.zeros(1, 8, 4)
    gates = torch.full((1, 2), -1.0)
    rows = relu_gated_oracle(q, k, gates, torch.tensor([3], dtype=torch.int32))
    assert rows[0].numel() == 3


def test_no_relu_oracle_differs_only_by_the_head_relu():
    torch.manual_seed(1)
    q = torch.randn(1, 4, 8)
    k = torch.randn(1, 32, 8)
    gates = torch.ones(1, 4)
    lens = torch.tensor([32], dtype=torch.int32)
    relu_row = relu_gated_oracle(q, k, gates, lens)[0]
    plain_row = no_relu_oracle(q, k, gates, lens)[0]
    assert not torch.allclose(relu_row, plain_row)
    # With all-positive scores the two formulas coincide.
    k_pos = k.abs()
    q_pos = q.abs()
    torch.testing.assert_close(
        relu_gated_oracle(q_pos, k_pos, gates, lens)[0],
        no_relu_oracle(q_pos, k_pos, gates, lens)[0],
    )


def test_relu_gated_oracle_is_invariant_to_a_positive_q_rescale():
    """``relu(c*x) == c*relu(x)`` for ``c > 0``, so a positive per-row rescale
    of q only scales the row's logits and cannot change its top-k. This is the
    algebraic licence for folding ``softmax_scale`` into the head gates."""
    torch.manual_seed(2)
    q = torch.randn(1, 8, 16)
    k = torch.randn(1, 64, 16)
    gates = torch.randn(1, 8)
    lens = torch.tensor([64], dtype=torch.int32)
    base = relu_gated_oracle(q, k, gates, lens)[0]
    scaled = relu_gated_oracle(q * 512.0, k, gates, lens)[0]
    torch.testing.assert_close(scaled / 512.0, base)
    assert torch.equal(
        torch.topk(base, 8).indices, torch.topk(scaled, 8).indices
    )


def test_quantize_k_like_the_cache_round_trips_within_the_fp8_bar():
    torch.manual_seed(3)
    k = torch.randn(256, HEAD_DIM, dtype=torch.bfloat16)
    deq = quantize_k_like_the_cache(k)
    x, y = k.double(), deq.double()
    calc_diff = 1 - 2 * (x * y).sum() / (x * x + y * y).sum()
    assert float(calc_diff) < 1e-3


# --------------------------------------------------------------------------- #
# CPU tests of the metrics
# --------------------------------------------------------------------------- #

def _oracle_rows(values: torch.Tensor) -> list[torch.Tensor]:
    return [row.double() for row in values]


def test_metrics_are_perfect_for_the_oracles_own_selection():
    torch.manual_seed(4)
    oracle = _oracle_rows(torch.randn(2, 64))
    topk = 8
    picked = torch.stack(
        [torch.topk(row, topk).indices for row in oracle]
    ).to(torch.int32)
    m = selection_metrics(picked, oracle, topk)
    assert m.overlap_mean == 1.0
    assert m.score_mass_mean == pytest.approx(1.0)
    assert m.near_tie_concentration == 0.0
    assert m.disagreements_per_row == 0.0


def test_metrics_catch_a_selection_that_drops_real_candidates():
    """Swapping the top picks for the worst positions must move BOTH overlap
    and score mass, and must push the concentration ratio towards 1."""
    oracle = [torch.linspace(1.0, 0.0, 64, dtype=torch.float64)]
    topk = 8
    picked = torch.arange(56, 64, dtype=torch.int32).unsqueeze(0)
    m = selection_metrics(picked, oracle, topk)
    assert m.overlap_mean == 0.0
    assert m.score_mass_mean < 0.2
    assert m.near_tie_concentration > 0.8


def test_metrics_near_tie_concentration_is_small_for_threshold_swaps():
    """The diagnostic that matters: trading the k-th pick for the (k+1)-th
    changes the SET but almost none of the score mass."""
    oracle = [torch.linspace(1.0, 0.0, 2048, dtype=torch.float64)]
    topk = 64
    picked = torch.cat(
        [torch.arange(63), torch.tensor([64])]
    ).to(torch.int32).unsqueeze(0)
    m = selection_metrics(picked, oracle, topk)
    assert m.overlap_mean == pytest.approx(63 / 64)
    assert m.score_mass_mean > 0.999
    assert m.near_tie_concentration < 0.1


def test_metrics_ignore_dense_prefix_rows_and_minus_one_padding():
    oracle = [torch.randn(4, dtype=torch.float64), torch.randn(64, dtype=torch.float64)]
    topk = 8
    picked = torch.full((2, topk), -1, dtype=torch.int32)
    picked[1] = torch.topk(oracle[1], topk).indices.to(torch.int32)
    m = selection_metrics(picked, oracle, topk)
    assert m.overlap_mean == 1.0


def test_metrics_reject_a_comparison_with_no_selecting_row():
    oracle = [torch.randn(4, dtype=torch.float64)]
    with pytest.raises(ValueError, match="no row longer than topk"):
        selection_metrics(torch.full((1, 8), -1, dtype=torch.int32), oracle, 8)


def test_metrics_reproduce_the_semantics_change_the_notes_measured():
    """End-to-end harness check: the no-ReLU formula must disagree with the
    ReLU oracle by the order the layout study reported (notes section 3.3),
    not by a tie-level amount. This is what makes the harness's FP8 numbers
    credible."""
    torch.manual_seed(5)
    length, topk = 4096, 512
    q = (torch.randn(1, HEADS, HEAD_DIM) / math.sqrt(HEAD_DIM)).to(torch.bfloat16)
    k = torch.randn(1, length, HEAD_DIM).to(torch.bfloat16)
    gates = torch.randn(1, HEADS)
    lens = torch.tensor([length], dtype=torch.int32)
    oracle = relu_gated_oracle(q, k, gates, lens)
    retired = no_relu_oracle(q, k, gates, lens)
    picked = torch.stack(
        [torch.topk(row, topk).indices for row in retired]
    ).to(torch.int32)
    m = selection_metrics(picked, oracle, topk)
    assert 0.5 < m.overlap_mean < 0.95, m.overlap_mean


def test_metrics_quantify_the_fp8_effect_on_the_relu_formula():
    """The delta this PR is responsible for: same formula, FP8 inputs.

    Asserted loosely on purpose -- the numbers that matter come from real
    activations on the node. What is pinned here is the SHAPE of the result:
    a few percent of indices move, almost no score mass does.
    """
    torch.manual_seed(6)
    length, topk = 4096, 512
    q = (torch.randn(1, HEADS, HEAD_DIM) / math.sqrt(HEAD_DIM)).to(torch.bfloat16)
    k = torch.randn(1, length, HEAD_DIM).to(torch.bfloat16)
    gates = torch.randn(1, HEADS)
    lens = torch.tensor([length], dtype=torch.int32)

    oracle = relu_gated_oracle(q, k, gates, lens)
    fp8_rows = relu_gated_oracle(
        q.to(torch.float8_e4m3fn).float(),
        quantize_k_like_the_cache(k),
        gates,
        lens,
    )
    picked = torch.stack(
        [torch.topk(row, topk).indices for row in fp8_rows]
    ).to(torch.int32)
    m = selection_metrics(picked, oracle, topk)
    assert m.overlap_mean > 0.90, m.overlap_mean
    assert m.score_mass_mean > 0.99, m.score_mass_mean
    assert m.near_tie_concentration < 0.3, m.near_tie_concentration


# --------------------------------------------------------------------------- #
# The node harness (CUDA-gated); the bound drops into ACCEPTANCE
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("length", [4096, 8192, 16384])
def test_production_scorer_against_the_relu_gated_oracle(length):
    """Drive the real page-split scorer and report the acceptance metrics.

    Builds a page-split FP8 aux cache from known bf16 keys, scores it with
    ``paged_mqa_score_and_topk`` (the eager production entry point), and
    compares the selection with the fp32 ReLU-gated oracle over the SAME
    dequantized keys.
    """

    from batchgen_kernels.attention.dsa.indexer_k_quant import (
        quantize_indexer_k_fp8,
    )
    from batchgen_kernels.attention.dsa.paged_mqa_scorer import (
        paged_mqa_score_and_topk,
    )

    device = torch.device("cuda")
    torch.manual_seed(7)
    batch, topk = 4, 2048
    pages_per_row = length // PAGE_SIZE
    num_pages = batch * pages_per_row

    keys = torch.randn(num_pages, PAGE_SIZE, HEAD_DIM, device=device).to(torch.bfloat16)
    k_fp8, scales = quantize_indexer_k_fp8(keys)
    cache = torch.zeros(
        num_pages, PAGE_SIZE, 1, HEAD_DIM + 4, dtype=torch.float8_e4m3fn, device=device
    )
    pages = cache.view(torch.uint8).reshape(num_pages, -1)
    pages[:, : PAGE_SIZE * HEAD_DIM] = k_fp8.reshape(num_pages, -1).view(torch.uint8)
    pages[:, PAGE_SIZE * HEAD_DIM :] = (
        scales.reshape(num_pages, PAGE_SIZE).contiguous().view(torch.uint8)
    )

    block_table = torch.arange(num_pages, dtype=torch.int32, device=device).reshape(
        batch, pages_per_row
    )
    cache_seqlens = torch.full((batch,), length, dtype=torch.int32, device=device)
    q = (torch.randn(batch, HEADS, HEAD_DIM, device=device) / math.sqrt(HEAD_DIM)).to(
        torch.bfloat16
    )
    head_gates = torch.randn(batch, HEADS, dtype=torch.float32, device=device)

    picked = paged_mqa_score_and_topk(
        q=q,
        aux_blocked_k=cache,
        block_table=block_table,
        head_gates=head_gates,
        cache_seqlens=cache_seqlens,
        topk=topk,
        max_seqlen=length,
        index_head_dim=HEAD_DIM,
        page_size=PAGE_SIZE,
    )

    dequantized = (k_fp8.float() * scales.unsqueeze(-1).float()).reshape(
        batch, length, HEAD_DIM
    )
    oracle = relu_gated_oracle(q.float(), dequantized, head_gates, cache_seqlens.cpu())
    metrics = selection_metrics(picked.cpu(), oracle, topk)
    print(f"\nL={length} {metrics}")

    if ACCEPTANCE is None:
        pytest.skip(
            "top-k overlap bound not yet approved; metrics reported above. "
            f"{ACCEPTANCE_PROPOSAL.splitlines()[0]}"
        )
    assert metrics.overlap_mean >= ACCEPTANCE["overlap_mean"]
    assert metrics.overlap_min >= ACCEPTANCE["overlap_min"]
    assert metrics.score_mass_mean >= ACCEPTANCE["score_mass_mean"]
    assert metrics.score_mass_min >= ACCEPTANCE["score_mass_min"]
    assert metrics.near_tie_concentration <= ACCEPTANCE["near_tie_concentration"]
