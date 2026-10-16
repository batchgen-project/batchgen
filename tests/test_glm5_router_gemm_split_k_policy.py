"""CPU contract for the GLM-5 router GEMM split-K path.

Runs without CUDA and without Triton: the policy module is loaded by path so
the Triton-importing package ``__init__`` stays out of the way. What is checked
here is what a GPU test cannot check as sharply -- that the split count every
row in the captured range resolves to is one the measured table states, that it
is a function of the row count and K alone, and that the K partition it implies
is a function of K alone, which together are what make a row's result
independent of its bucket's padding and of every other row.

The count itself is per bucket, not global: the sweep in the policy module
measures S=8 fastest only at the lowest captured bucket and S=4 fastest from
256 rows up. So the invariant this file pins is not "one count everywhere" but
"one count per bucket, taken from the measurement".

The numerical bound is NOT derived here, and the CPU chain models below are not
a model of either kernel's error. They materialise every block_k chunk partial
as a separately rounded float32 value and then reassociate those values, which
is the one thing the device does not do: both kernels hold a float32 register
accumulator across their chunks (``acc += tl.dot(...)``, which the compiler is
free to fold into the MMA's own accumulator), so no per-chunk value is ever
rounded on its own and the order in which the block_k products are summed
inside a chunk is hardware defined. A difference measured between these two CPU
models therefore measures the chunk-level REGROUPING alone; it is neither an
upper bound on a device kernel's distance from truth nor a budget the device has
to meet, and the earlier version of this file was wrong to derive one from it.

What survives on the CPU side is the shape of the bound rather than its value:
these models do execute exactly the chunk-level chains the depth formula
describes (96 chained chunk additions for serial-K; 12 then 8 at S=8, 24 then 4
at S=4), so the formula can be checked against them here for every count the
policy can select, and this file asserts that the formula and the headroom match
the GPU suite's. The bound itself is asserted
against a float64 oracle in tests/test_glm5_router_gemm_cuda.py, which is the
only place the real chains run.
"""

import importlib.util
import inspect
from pathlib import Path

import pytest
import torch


def _load_by_path(name, *parts):
    path = Path(__file__).resolve().parents[1].joinpath(*parts)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


POLICY = _load_by_path(
    "glm5_router_gemm_policy_under_test",
    "batchgen_kernels",
    "triton",
    "glm5_router_gemm_config.py",
)
# Imported for its bound constants and depth formula only: the module body is
# CUDA-free, and every test in it is skipped without a device.
GPU_SUITE = _load_by_path(
    "glm5_router_gemm_cuda_suite_under_test",
    "tests",
    "test_glm5_router_gemm_cuda.py",
)

GLM5_K = 6144
GLM5_EXPERTS = 256
GLM5_TOPK = 8
GRAPH_ROWS = (192, 256, 384, 512)

# The measured split-count table, restated so this file asserts the policy
# against the measurement rather than against whatever the policy now holds. The
# sweep behind it is the one the policy module cites: internal campaign
# glm52-decode-step-gap-vs-sglang, slice A.
MEASURED_SPLITS = ((192, 8), (256, 4), (384, 4), (512, 4))

# The per-element error bound both suites use, restated here so this file can
# check it against the chains it runs:
#
#     |result - truth| <= HEADROOM * depth * eps * sum_c |chunk_c|
#
# with depth = chunks for the serial-K chain and chunks // split_k + split_k for
# the split-K pair. Scaling by sum_c|chunk_c| rather than by |result| is the
# whole correction: the error of a summation chain tracks the terms it sums, and
# the chunks partially cancel, so they are larger than the logit they produce.
# The headroom covers the intra-chunk tensor-core order, which is identical in
# both kernels and not derivable from either side. Derivation and calibration
# live in the GPU suite's header; test_bound_constants_match_the_gpu_suite keeps
# the two copies from drifting.
CHAIN_BOUND_HEADROOM = 4.0
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
    """float32 partial product of every block_k chunk, in increasing K order.

    A CPU stand-in, not what the device holds: here each chunk becomes a
    separately rounded float32 value, while on the device the chunk is
    accumulated straight into a register accumulator.
    """
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
    """The serial-K kernel's chunk-level GROUPING: one chain over every chunk.

    It reproduces which chunks are summed together and in what order, which is
    all this file claims. It is not the kernel's arithmetic: the device never
    rounds a chunk on its own, and its intra-chunk order is hardware defined.
    """
    return _ordered_sum(partials, range(partials.shape[0]))


def _split_k_model(partials, split_k):
    """The split-K pair's chunk-level grouping: per-split chains, split order.

    Same standing as :func:`_serial_k_model` -- a model of the grouping, not of
    the error.
    """
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


def _fp64_truth(hidden, weight):
    """The oracle: the same bf16 inputs contracted in float64.

    Its own error is bounded by ``k * eps64 * sum_d |h_d w_d|``, eight orders of
    magnitude under the float32 bounds below.
    """
    return hidden.to(torch.float64).matmul(weight.to(torch.float64).t())


def _chunk_magnitude(partials):
    """Per-element ``sum_c |chunk_c|``, the magnitude the bounds scale with."""
    return partials.to(torch.float64).abs().sum(dim=0)


def _chain_bound(partials, split_k):
    depth = GPU_SUITE.chain_addition_depth(partials.shape[0], split_k)
    return CHAIN_BOUND_HEADROOM * depth * F32_EPS * _chunk_magnitude(partials)


def _assert_inside(result, truth, bound, what):
    error = (result.to(torch.float64) - truth).abs()
    n_over = int((error > bound).sum())
    assert not n_over, (
        f"{what}: {n_over}/{error.numel()} elements exceed the bound, worst "
        f"ratio {float(error.div(bound.clamp_min(1e-300)).max()):.3f}"
    )


def test_every_row_in_the_graph_range_resolves_to_a_measured_split():
    # The split count now varies across buckets, so what has to hold for every
    # row in the captured range is weaker but still sharp: a real split count,
    # and one that appears in the measured table rather than an interpolation
    # between two of its entries.
    counts = {POLICY.select_router_split_k(m, GLM5_K) for m in range(192, 513)}
    assert counts <= set(POLICY.ROUTER_SPLIT_K_VALUES)
    assert all(count > 1 for count in counts)
    assert counts == set(POLICY.ROUTER_SPLIT_K_VALUES), (
        "a table entry is unreachable from the captured row range"
    )


def test_the_split_count_takes_only_the_shape_as_input():
    # Bucket-M independence stated as a signature, the same way the partition's
    # is below. The count may differ between captured buckets, but inside one
    # bucket m is world_size * bucket_size and therefore fixed, so a count that
    # takes nothing but m and k is a count that is constant for that bucket's
    # whole graph -- and in particular cannot see how many rows are live.
    assert list(
        inspect.signature(POLICY.select_router_split_k).parameters
    ) == ["m", "k"]
    assert list(
        inspect.signature(POLICY.router_measured_split_k).parameters
    ) == ["m"]
    assert all(
        POLICY.select_router_split_k(m, GLM5_K)
        == POLICY.select_router_split_k(m, GLM5_K)
        for m in range(192, 513)
    )


@pytest.mark.parametrize("m", [1, 8, 64, 128, 191, 513, 1024, 4096])
def test_rows_outside_the_graph_row_range_stay_serial(m):
    assert POLICY.select_router_split_k(m, GLM5_K) == 1


@pytest.mark.parametrize("m,expected", MEASURED_SPLITS)
def test_measured_rows_select_their_measured_split(m, expected):
    # The table is the measurement, restated as the assertion. 192 is the one
    # captured bucket where the extra parallelism still pays; 256 and above
    # spend more in the reduction pass than the shorter K walk saves.
    assert POLICY.select_router_split_k(m, GLM5_K) == expected


def test_the_table_covers_exactly_the_measured_rows():
    assert [row for row, _ in POLICY.ROUTER_SPLIT_K_MEASURED] == list(GRAPH_ROWS)
    assert POLICY.ROUTER_SPLIT_K_MEASURED == MEASURED_SPLITS
    assert POLICY.ROUTER_SPLIT_K_VALUES == (4, 8)


@pytest.mark.parametrize(
    "m,expected",
    [
        # Unmeasured buckets generate_bucket_sizes can place: 14/20/28 at
        # world_size 16. Each takes its nearest measured point, and 224 -- 32
        # rows from both 192 and 256 -- takes the higher row count by the tie
        # rule the policy documents.
        (200, 8),
        (208, 8),
        (223, 8),
        (224, 4),
        (225, 4),
        (240, 4),
        (320, 4),
        (448, 4),
    ],
)
def test_unmeasured_rows_take_the_nearest_measured_split(m, expected):
    assert POLICY.select_router_split_k(m, GLM5_K) == expected
    assert POLICY.router_measured_split_k(m) == expected


def test_the_split_count_changes_exactly_once_over_the_range():
    # One monotone step, not a sawtooth: the table has a single sign change, so
    # the policy must too, or some bucket is getting a count no measurement
    # supports.
    counts = [POLICY.select_router_split_k(m, GLM5_K) for m in range(192, 513)]
    changes = [i for i in range(1, len(counts)) if counts[i] != counts[i - 1]]
    assert len(changes) == 1, f"split count changes at {changes}"
    assert counts[0] == 8
    assert counts[-1] == 4


@pytest.mark.parametrize(
    "table",
    [
        pytest.param((), id="empty"),
        pytest.param(((192, 8), (256, 4)), id="stops-below-the-max"),
        pytest.param(((256, 4), (512, 4)), id="starts-above-the-min"),
        pytest.param(((256, 4), (192, 8), (512, 4)), id="unsorted"),
        pytest.param(((192, 8), (256, 4), (256, 4), (512, 4)), id="duplicate-rows"),
        pytest.param(((192, 8), (256, 1), (512, 4)), id="not-a-split"),
    ],
)
def test_a_table_that_stops_covering_the_dispatched_range_is_rejected(
    table, monkeypatch
):
    # The table is the only source of the count, so a table that does not span
    # the dispatched rows has to fail loudly rather than extrapolate.
    monkeypatch.setattr(POLICY, "ROUTER_SPLIT_K_MEASURED", table)
    with pytest.raises(POLICY.RouterGemmPolicyError):
        POLICY._validate_measured_table()


def test_the_dispatched_range_matches_the_measured_rows():
    assert POLICY.ROUTER_SPLIT_K_MIN_ROWS == GRAPH_ROWS[0]
    assert POLICY.ROUTER_SPLIT_K_MAX_ROWS == GRAPH_ROWS[-1]
    POLICY._validate_measured_table()


@pytest.mark.parametrize("split_k", [4, 8])
def test_partition_is_exact_contiguous_and_ordered(split_k):
    bounds = POLICY.router_split_k_bounds(GLM5_K, split_k)
    assert len(bounds) == split_k
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


def test_bound_constants_match_the_gpu_suite():
    # The bound is asserted on the device and restated here; if the two copies
    # drift, the file that claims to pin the formula stops pinning anything.
    assert CHAIN_BOUND_HEADROOM == GPU_SUITE.CHAIN_BOUND_HEADROOM
    assert F32_EPS == GPU_SUITE.F32_EPS
    chunks = GLM5_K // POLICY.ROUTER_BLOCK_K
    assert GPU_SUITE.chain_addition_depth(chunks, 1) == chunks
    for split_k in POLICY.ROUTER_SPLIT_K_VALUES:
        assert GPU_SUITE.chain_addition_depth(chunks, split_k) == (
            chunks // split_k + split_k
        )
    with pytest.raises(ValueError):
        GPU_SUITE.chain_addition_depth(chunks, 7)
    with pytest.raises(ValueError):
        GPU_SUITE.chain_addition_depth(chunks, 0)


@pytest.mark.parametrize("split_k", [4, 8])
@pytest.mark.parametrize("regime", ["unit", "wide", "outlier"])
def test_chunk_level_regrouping_stays_inside_the_chain_bound(regime, split_k):
    # The two models share one set of materialised chunk partials, so their
    # difference carries no intra-chunk error at all: it is exactly the
    # chunk-level reassociation, bounded by the two chains' depths together.
    # Run for both counts the table can select, since each regroups the same
    # chunks differently: depth 20 at S=8 and 28 at S=4, against 96 serial.
    rows = 128
    hidden, weight = _activations(regime, rows, GLM5_K, GLM5_EXPERTS, seed=20261016)
    partials = _chunk_partials(hidden, weight)
    serial = _serial_k_model(partials)
    split = _split_k_model(partials, split_k)

    bound = _chain_bound(partials, 1) + _chain_bound(partials, split_k)
    _assert_inside(split, serial.to(torch.float64), bound, f"{regime}: regrouping")

    # Each model is also inside the bound for the chain it runs, and the shorter
    # chains are not the further ones from truth -- the same two properties the
    # GPU suite asserts on the real kernels.
    truth = _fp64_truth(hidden, weight)
    _assert_inside(serial, truth, _chain_bound(partials, 1), f"{regime}: serial-K")
    _assert_inside(
        split,
        truth,
        _chain_bound(partials, split_k),
        f"{regime}: split-K={split_k}",
    )
    d_serial = float((serial.to(torch.float64) - truth).abs().max())
    d_split = float((split.to(torch.float64) - truth).abs().max())
    assert d_split <= d_serial, (
        f"{regime}: split-K={split_k} grouping is further from float64 truth "
        f"than serial-K ({d_split:.3e} > {d_serial:.3e})"
    )

    serial_ids = serial.topk(GLM5_TOPK, dim=-1).indices.sort(dim=-1).values
    split_ids = split.topk(GLM5_TOPK, dim=-1).indices.sort(dim=-1).values
    assert torch.equal(serial_ids, split_ids), f"{regime}: top-k set disagreement"


@pytest.mark.parametrize("regime", ["unit", "wide", "outlier"])
def test_the_table_counts_group_to_the_same_routed_expert_set(regime):
    # The per-bucket table lets one row be summed at S=8 in one bucket and S=4
    # in another, so the groupings must agree on the routed expert SET -- the
    # only part of the logits the MoE dispatch reads. Asserted here on the
    # chunk-level models and on the real kernels in the GPU suite.
    hidden, weight = _activations(regime, 128, GLM5_K, GLM5_EXPERTS, seed=515151)
    partials = _chunk_partials(hidden, weight)
    ids = {}
    for split_k in POLICY.ROUTER_SPLIT_K_VALUES:
        grouped = _split_k_model(partials, split_k)
        ids[split_k] = grouped.topk(GLM5_TOPK, dim=-1).indices.sort(dim=-1).values
    reference = POLICY.ROUTER_SPLIT_K_VALUES[0]
    for split_k, value in ids.items():
        assert torch.equal(value, ids[reference]), (
            f"{regime}: split_k={split_k} and split_k={reference} route to "
            f"different expert sets"
        )


@pytest.mark.parametrize("split_k", [4, 8])
def test_split_k_regrouping_is_row_local(split_k):
    # Same rows, two row counts, one split count -- which is the within-bucket
    # case, since a bucket fixes the count. The chunk partials and every grouped
    # sum are bitwise identical, so nothing in the split-K order can make a
    # row's result depend on how many rows shared its launch.
    hidden, weight = _activations("outlier", 512, GLM5_K, GLM5_EXPERTS, seed=4321)
    wide = _split_k_model(_chunk_partials(hidden, weight), split_k)
    narrow = _split_k_model(_chunk_partials(hidden[:192], weight), split_k)
    assert _bitwise_equal(wide[:192], narrow)


@pytest.mark.parametrize("split_k", [4, 8, 16])
def test_every_candidate_split_count_stays_inside_its_own_chain_bound(split_k):
    # The split count is a tuning knob. Each candidate is held to the bound for
    # the chain it would execute -- depth chunks // split_k + split_k, which is
    # shorter than the serial chain for every candidate here -- and none of them
    # may be further from truth than the serial chain they replace.
    hidden, weight = _activations("wide", 128, GLM5_K, GLM5_EXPERTS, seed=777)
    partials = _chunk_partials(hidden, weight)
    truth = _fp64_truth(hidden, weight)
    serial = _serial_k_model(partials)
    split = _split_k_model(partials, split_k)

    depth = GPU_SUITE.chain_addition_depth(partials.shape[0], split_k)
    assert depth < partials.shape[0], (
        f"split_k={split_k} does not shorten the chain (depth {depth})"
    )
    _assert_inside(split, truth, _chain_bound(partials, split_k), f"split_k={split_k}")
    assert float((split.to(torch.float64) - truth).abs().max()) <= float(
        (serial.to(torch.float64) - truth).abs().max()
    )
