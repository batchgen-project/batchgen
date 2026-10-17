"""CPU contract for the GLM-5 router GEMM split-K path.

Runs without CUDA and without Triton: the policy module is loaded by path so
the Triton-importing package ``__init__`` stays out of the way. What is checked
here is what a GPU test cannot check as sharply -- that the split count every
row in the captured range resolves to is one the measured table states, that it
is a function of the row count and K alone, and that the K partition it implies
is a function of K alone, which together are what make a row's result
independent of its bucket's padding and of every other row.

The count is one constant over the whole dispatched range. The sweep in the
policy module does measure a different count as marginally faster at the very
top of that range, but a count that changes with the bucket changes the float32
reduction order with it, and that breaks the bitwise cross-bucket
M-independence contract the GPU suite asserts. So the invariant this file pins
is "one count everywhere in the dispatched range, taken from the measurement",
and the table is held to a single entry rather than trusted to be constant by
inspection.

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
# Row counts the graphed router is dispatched at, bottom to top of the range.
# The bottom two are the per-rank row counts the DeepEP exchange routes on; the
# rest are all-gathered row spaces.
GRAPH_ROWS = (64, 128, 192, 256, 384, 512)

# The measured split-count table, restated so this file asserts the policy
# against the measurement rather than against whatever the policy now holds. The
# sweep behind it is the one the policy module cites: internal campaign
# glm52-decode-step-gap-vs-sglang, slice F, whose cold-weight harness supersedes
# the earlier L2-warm one. One entry, because one count covers the dispatched
# range by ruling (same campaign, 2026-10-17).
MEASURED_SPLITS = ((64, 8),)

# Counts the kernel can still be FORCED to run through ``split_k=``, which is
# how the sweep measures them, even though the policy selects only one. The
# second count is the one the cold sweep finds marginally ahead at the very top
# of the range, and the one a withdrawn per-bucket table would have selected
# there; the chunk-level models below keep it covered so the next sweep is not
# measuring an unasserted path.
FORCED_SPLITS = (4, 8)

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


def test_every_row_in_the_graph_range_resolves_to_the_one_measured_split():
    # Every row in the captured range resolves to a real split count, the same
    # one, and the one the table states -- never an interpolation and never a
    # second count that some other bucket could pick instead.
    counts = {POLICY.select_router_split_k(m, GLM5_K) for m in range(64, 513)}
    assert len(counts) == 1, f"the captured range resolves to {sorted(counts)}"
    assert all(count > 1 for count in counts)
    assert counts == set(POLICY.ROUTER_SPLIT_K_VALUES), (
        "a table entry is unreachable from the captured row range"
    )


def test_the_split_count_takes_only_the_shape_as_input():
    # Bucket-M independence stated as a signature, the same way the partition's
    # is below. The count is the same for every bucket, and a count that takes
    # nothing but m and k cannot become bucket-dependent by accident either: in
    # particular it cannot see how many rows of a launch are live.
    assert list(
        inspect.signature(POLICY.select_router_split_k).parameters
    ) == ["m", "k"]
    assert list(
        inspect.signature(POLICY.router_measured_split_k).parameters
    ) == ["m"]
    assert all(
        POLICY.select_router_split_k(m, GLM5_K)
        == POLICY.select_router_split_k(m, GLM5_K)
        for m in range(64, 513)
    )


@pytest.mark.parametrize("m", [1, 8, 16, 32, 48, 63, 513, 1024, 4096])
def test_rows_outside_the_graph_row_range_stay_serial(m):
    assert POLICY.select_router_split_k(m, GLM5_K) == 1


@pytest.mark.parametrize("m,expected", MEASURED_SPLITS)
def test_measured_rows_select_their_measured_split(m, expected):
    # The table is the measurement, restated as the assertion, at the row the
    # measurement was taken at.
    assert POLICY.select_router_split_k(m, GLM5_K) == expected


@pytest.mark.parametrize("m", GRAPH_ROWS)
def test_every_captured_bucket_selects_the_same_split(m):
    # The captured buckets are the ones whose graphs bake the count in, so each
    # of them has to bake in the same one; otherwise the same row reduces in two
    # different orders depending on which bucket carried it.
    assert POLICY.select_router_split_k(m, GLM5_K) == MEASURED_SPLITS[0][1]


def test_the_table_states_one_count_for_the_whole_range():
    assert POLICY.ROUTER_SPLIT_K_MEASURED == MEASURED_SPLITS
    assert len(POLICY.ROUTER_SPLIT_K_MEASURED) == 1
    assert [row for row, _ in POLICY.ROUTER_SPLIT_K_MEASURED] == [GRAPH_ROWS[0]]
    assert POLICY.ROUTER_SPLIT_K_VALUES == (8,)


@pytest.mark.parametrize(
    "m",
    [
        # Unmeasured per-rank row counts the DeepEP path routes on, between the
        # bottom of the range and the largest decode bucket.
        65,
        96,
        112,
        127,
        160,
        176,
        # Unmeasured buckets generate_bucket_sizes can place: 14/20/28 at
        # world_size 16. 200 is one of the two row counts (with 192, covered
        # above) where the withdrawn per-bucket table resolved to a different
        # count than the padded 512-row launch it has to stay bitwise equal to.
        200,
        208,
        223,
        224,
        225,
        240,
        320,
        448,
    ],
)
def test_unmeasured_rows_take_the_one_measured_split(m):
    expected = MEASURED_SPLITS[0][1]
    assert POLICY.select_router_split_k(m, GLM5_K) == expected
    assert POLICY.router_measured_split_k(m) == expected


def test_the_split_count_never_changes_over_the_range():
    # The contract the per-bucket table broke: the count fixes the float32
    # reduction order, so a single change anywhere inside the dispatched range
    # makes some row's logits depend on which bucket carried it.
    counts = [POLICY.select_router_split_k(m, GLM5_K) for m in range(64, 513)]
    changes = [64 + i for i in range(1, len(counts)) if counts[i] != counts[i - 1]]
    assert not changes, f"split count changes at rows {changes}"
    assert set(counts) == {MEASURED_SPLITS[0][1]}


@pytest.mark.parametrize(
    "table",
    [
        pytest.param((), id="empty"),
        pytest.param(((64, 8), (192, 4), (384, 4), (512, 4)), id="bucket-dependent"),
        pytest.param(((64, 8), (256, 4)), id="two-counts"),
        pytest.param(((192, 8),), id="starts-above-the-min"),
        pytest.param(((64, 8), (1024, 8)), id="measures-above-the-max"),
        pytest.param(((64, 8), (512, 8), (256, 8)), id="unsorted"),
        pytest.param(((64, 8), (256, 8), (256, 8)), id="duplicate-rows"),
        pytest.param(((64, 1),), id="not-a-split"),
    ],
)
def test_a_table_that_does_not_state_one_count_for_the_range_is_rejected(
    table, monkeypatch
):
    # The table is the only source of the count, so it has to fail loudly rather
    # than extrapolate off the bottom of the dispatched range or hand two
    # buckets two different reduction orders. "bucket-dependent" is the exact
    # table this file's contract replaced.
    monkeypatch.setattr(POLICY, "ROUTER_SPLIT_K_MEASURED", table)
    with pytest.raises(POLICY.RouterGemmPolicyError):
        POLICY._validate_measured_table()


def test_the_dispatched_range_matches_the_captured_buckets():
    # The range the split-K path serves is the range of captured buckets, which
    # is what makes one count over that range enough.
    assert POLICY.ROUTER_SPLIT_K_MIN_ROWS == GRAPH_ROWS[0]
    assert POLICY.ROUTER_SPLIT_K_MAX_ROWS == GRAPH_ROWS[-1]
    POLICY._validate_measured_table()


@pytest.mark.parametrize("split_k", FORCED_SPLITS)
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
    # The selected count is one of the counts the sweep can force, so the two
    # files' constants cannot drift apart either.
    assert set(POLICY.ROUTER_SPLIT_K_VALUES) <= set(FORCED_SPLITS)
    assert FORCED_SPLITS == GPU_SUITE.FORCED_SPLIT_COUNTS
    for split_k in FORCED_SPLITS:
        assert GPU_SUITE.chain_addition_depth(chunks, split_k) == (
            chunks // split_k + split_k
        )
    with pytest.raises(ValueError):
        GPU_SUITE.chain_addition_depth(chunks, 7)
    with pytest.raises(ValueError):
        GPU_SUITE.chain_addition_depth(chunks, 0)


@pytest.mark.parametrize("split_k", FORCED_SPLITS)
@pytest.mark.parametrize("regime", ["unit", "wide", "outlier"])
def test_chunk_level_regrouping_stays_inside_the_chain_bound(regime, split_k):
    # The two models share one set of materialised chunk partials, so their
    # difference carries no intra-chunk error at all: it is exactly the
    # chunk-level reassociation, bounded by the two chains' depths together.
    # Run for every count the sweep can force, not only the selected one, since
    # each regroups the same chunks differently: depth 20 at S=8 and 28 at S=4,
    # against 96 serial.
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
def test_the_forced_counts_group_to_the_same_routed_expert_set(regime):
    # The policy now selects one count, so no two buckets disagree -- but the
    # counts the sweep forces still have to agree on the routed expert SET, the
    # only part of the logits the MoE dispatch reads, or the next measurement
    # would be choosing between kernels that route differently. Asserted here on
    # the chunk-level models and on the real kernels in the GPU suite.
    hidden, weight = _activations(regime, 128, GLM5_K, GLM5_EXPERTS, seed=515151)
    partials = _chunk_partials(hidden, weight)
    ids = {}
    for split_k in FORCED_SPLITS:
        grouped = _split_k_model(partials, split_k)
        ids[split_k] = grouped.topk(GLM5_TOPK, dim=-1).indices.sort(dim=-1).values
    reference = FORCED_SPLITS[0]
    for split_k, value in ids.items():
        assert torch.equal(value, ids[reference]), (
            f"{regime}: split_k={split_k} and split_k={reference} route to "
            f"different expert sets"
        )


@pytest.mark.parametrize("split_k", FORCED_SPLITS)
def test_split_k_regrouping_is_row_local(split_k):
    # Same rows, two row counts, one split count -- which is now every pair of
    # launches in the dispatched range, since the count no longer varies with
    # the row count. The chunk partials and every grouped sum are bitwise
    # identical, so nothing in the split-K order can make a row's result depend
    # on how many rows shared its launch.
    hidden, weight = _activations("outlier", 512, GLM5_K, GLM5_EXPERTS, seed=4321)
    narrow_rows = GRAPH_ROWS[0]
    wide = _split_k_model(_chunk_partials(hidden, weight), split_k)
    narrow = _split_k_model(
        _chunk_partials(hidden[:narrow_rows], weight), split_k
    )
    assert _bitwise_equal(wide[:narrow_rows], narrow)


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
