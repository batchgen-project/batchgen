"""CPU contract for the GLM-5 router GEMM split-K path.

Runs without CUDA and without Triton: the policy module is loaded by path so
the Triton-importing package ``__init__`` stays out of the way. Two things are
checked here that a GPU test cannot check as sharply:

* the split count and the K partition are a function of K alone across the
  whole row range the graphed router path captures, which is what makes a
  row's result independent of its bucket's padding; and
* the float32 budget the GPU test asserts, derived from the block-level
  summation orders the two kernels actually perform.
"""

import importlib.util
import inspect
from pathlib import Path

import pytest
import torch


def _load_policy():
    path = (
        Path(__file__).resolve().parents[1]
        / "batchgen_kernels"
        / "triton"
        / "glm5_router_gemm_config.py"
    )
    spec = importlib.util.spec_from_file_location(
        "glm5_router_gemm_policy_under_test", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


POLICY = _load_policy()

GLM5_K = 6144
GLM5_EXPERTS = 256
GLM5_TOPK = 8
GRAPH_ROWS = (192, 256, 384, 512)

# Budget asserted for the serial-K vs split-K difference, as a multiple of
# float32 eps scaled by the magnitude of the logits. The two kernels sum the
# same block_k chunk partials and differ only in how they group them, so the
# difference is a float32 reassociation, bounded by the accumulator precision
# times the result magnitude -- not by K, M, or the activation distribution.
# The block-level model below measures a worst case of roughly 4 eps across
# three activation regimes; 32 eps is asserted, leaving room for the tensor
# core's own intra-chunk order on the device.
ULP_BUDGET = 32
F32_EPS = 2.0**-23


def _activations(regime, rows, k, experts, seed):
    generator = torch.Generator().manual_seed(seed)
    weight = (
        torch.randn(experts, k, generator=generator, dtype=torch.float32) * 0.05
    ).to(torch.bfloat16)
    if regime == "unit":
        hidden = torch.randn(rows, k, generator=generator, dtype=torch.float32) * 0.1
    elif regime == "wide":
        hidden = torch.randn(rows, k, generator=generator, dtype=torch.float32) * 2.0
    elif regime == "outlier":
        # A handful of dominant channels, the shape post-norm decode hidden
        # states actually take, which is the worst case for a regrouped sum.
        hidden = torch.randn(rows, k, generator=generator, dtype=torch.float32) * 0.5
        hidden[:, ::512] *= 40.0
    else:  # pragma: no cover - guarded by parametrize
        raise AssertionError(f"unknown regime {regime}")
    return hidden.to(torch.bfloat16), weight


def _chunk_partials(hidden, weight):
    """float32 partial product of every block_k chunk, in increasing K order."""
    block_k = POLICY.ROUTER_BLOCK_K
    chunks = hidden.shape[1] // block_k
    out = torch.empty(
        chunks, hidden.shape[0], weight.shape[0], dtype=torch.float32
    )
    for c in range(chunks):
        span = slice(c * block_k, (c + 1) * block_k)
        out[c] = hidden[:, span].float().matmul(weight[:, span].float().t())
    return out


def _ordered_sum(partials, order):
    acc = partials[order[0]].clone()
    for index in order[1:]:
        acc = acc + partials[index]
    return acc


def _serial_k_model(partials):
    """What the serial-K kernel accumulates: one chain over every chunk."""
    return _ordered_sum(partials, range(partials.shape[0]))


def _split_k_model(partials, split_k):
    """What the split-K pair accumulates: per-split chains, then split order."""
    block_k = POLICY.ROUTER_BLOCK_K
    k = partials.shape[0] * block_k
    per_split = []
    for start, stop in POLICY.router_split_k_bounds(k, split_k):
        per_split.append(
            _ordered_sum(partials, range(start // block_k, stop // block_k))
        )
    acc = per_split[0].clone()
    for partial in per_split[1:]:
        acc = acc + partial
    return acc


def _bitwise_equal(left, right):
    return bool(
        torch.equal(left.view(torch.int32), right.view(torch.int32))
    )


def test_graph_row_range_uses_a_single_split_count():
    # Every captured bucket whose global row count lands in this range must
    # resolve to the same split count, or the same row would be summed in a
    # different order depending on which bucket carried it.
    counts = {POLICY.select_router_split_k(m, GLM5_K) for m in range(192, 513)}
    assert counts == {POLICY.ROUTER_SPLIT_K}
    assert POLICY.ROUTER_SPLIT_K > 1


@pytest.mark.parametrize("m", [1, 8, 64, 128, 191, 513, 1024, 4096])
def test_rows_outside_the_graph_row_range_stay_serial(m):
    assert POLICY.select_router_split_k(m, GLM5_K) == 1


@pytest.mark.parametrize("m", GRAPH_ROWS)
def test_production_shape_selects_split_k(m):
    assert POLICY.select_router_split_k(m, GLM5_K) == POLICY.ROUTER_SPLIT_K


def test_partition_is_exact_contiguous_and_ordered():
    bounds = POLICY.router_split_k_bounds(GLM5_K, POLICY.ROUTER_SPLIT_K)
    assert len(bounds) == POLICY.ROUTER_SPLIT_K
    assert bounds[0][0] == 0
    assert bounds[-1][1] == GLM5_K
    widths = set()
    for index, (start, stop) in enumerate(bounds):
        assert start < stop
        if index:
            assert start == bounds[index - 1][1]
        assert start % POLICY.ROUTER_BLOCK_K == 0
        assert (stop - start) % POLICY.ROUTER_BLOCK_K == 0
        widths.add(stop - start)
    assert len(widths) == 1


def test_partition_cannot_depend_on_the_row_count():
    # The invariant stated as a signature: neither the partition nor its order
    # is allowed to take the row count as an input.
    assert list(
        inspect.signature(POLICY.router_split_k_bounds).parameters
    ) == ["k", "split_k"]


@pytest.mark.parametrize("split_k", [0, -1, 5, 7, 128, 6145])
def test_inexact_split_counts_are_rejected(split_k):
    with pytest.raises(POLICY.RouterGemmShapeError):
        POLICY.validate_router_split_k(GLM5_K, split_k)


@pytest.mark.parametrize("split_k", [1, 2, 4, 8, 16, 32])
def test_exact_split_counts_are_accepted_for_the_production_k(split_k):
    POLICY.validate_router_split_k(GLM5_K, split_k)
    assert POLICY.router_split_k_is_exact(GLM5_K, split_k)


@pytest.mark.parametrize("k", [100, 6143, 96])
def test_rows_in_range_stay_serial_when_k_cannot_be_partitioned(k):
    assert POLICY.select_router_split_k(256, k) == 1


@pytest.mark.parametrize("regime", ["unit", "wide", "outlier"])
def test_split_k_regrouping_stays_within_the_asserted_budget(regime):
    rows = 128
    hidden, weight = _activations(regime, rows, GLM5_K, GLM5_EXPERTS, seed=20261016)
    partials = _chunk_partials(hidden, weight)
    serial = _serial_k_model(partials)
    split = _split_k_model(partials, POLICY.ROUTER_SPLIT_K)

    scale = float(serial.abs().max())
    budget = ULP_BUDGET * F32_EPS * scale
    observed = float((split - serial).abs().max())
    assert observed <= budget, (
        f"{regime}: regrouping diff {observed:.3e} exceeds {budget:.3e} "
        f"(logit scale {scale:.3f})"
    )

    serial_ids = serial.topk(GLM5_TOPK, dim=-1).indices.sort(dim=-1).values
    split_ids = split.topk(GLM5_TOPK, dim=-1).indices.sort(dim=-1).values
    assert torch.equal(serial_ids, split_ids), f"{regime}: top-k set disagreement"


def test_split_k_regrouping_is_row_local():
    # Same rows, two row counts: the chunk partials and every grouped sum are
    # bitwise identical, so nothing in the split-K order can make a row's
    # result depend on how many rows shared its launch.
    hidden, weight = _activations("outlier", 512, GLM5_K, GLM5_EXPERTS, seed=4321)
    wide = _split_k_model(_chunk_partials(hidden, weight), POLICY.ROUTER_SPLIT_K)
    narrow = _split_k_model(
        _chunk_partials(hidden[:192], weight), POLICY.ROUTER_SPLIT_K
    )
    assert _bitwise_equal(wide[:192], narrow)


@pytest.mark.parametrize("split_k", [4, 8, 16])
def test_every_candidate_split_count_shares_the_same_budget(split_k):
    # The split count is a tuning knob; none of the candidates may widen the
    # numerical envelope the production path is allowed to move inside.
    hidden, weight = _activations("wide", 128, GLM5_K, GLM5_EXPERTS, seed=777)
    partials = _chunk_partials(hidden, weight)
    serial = _serial_k_model(partials)
    split = _split_k_model(partials, split_k)
    budget = ULP_BUDGET * F32_EPS * float(serial.abs().max())
    assert float((split - serial).abs().max()) <= budget
