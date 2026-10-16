import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GLM-5 router GEMM tests require CUDA",
)


ATOL = 1e-5
RTOL = 1.6e-2
OUTLIER_THRESHOLD = 1e-4


def _assert_bf16_router_close(actual: torch.Tensor, ref: torch.Tensor) -> None:
    diff = (actual.float() - ref.float()).abs()
    tol = ATOL + RTOL * ref.float().abs()
    failures = diff > tol
    n_fail = int(failures.sum().item())
    n_total = failures.numel()
    if n_fail and n_fail / n_total >= OUTLIER_THRESHOLD:
        raise AssertionError(
            f"router GEMM mismatch: max_abs={float(diff.max().item())} "
            f"failures={n_fail}/{n_total}"
        )


def _reference_router(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return hidden.matmul(weight.t()).float()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


@pytest.mark.parametrize("m", [1, 3, 7, 8, 11, 16, 32, 61, 64, 128, 256, 512, 1024])
def test_glm5_router_gemm_shape_sweep_matches_reference(m):
    from batchgen.moe.routing import glm5_router_gemm_cuda

    torch.manual_seed(1234 + m)
    device = torch.device("cuda")
    hidden = (torch.randn(m, 64, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    weight = (torch.randn(32, 64, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    actual = glm5_router_gemm_cuda(hidden, weight)
    ref = _reference_router(hidden, weight)
    _assert_bf16_router_close(actual, ref)


def test_glm5_router_gemm_glm_shape_smoke_matches_reference():
    from batchgen.moe.routing import glm5_router_gemm_cuda

    torch.manual_seed(2026)
    device = torch.device("cuda")
    hidden = (torch.randn(4, 6144, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    weight = (torch.randn(256, 6144, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    actual = glm5_router_gemm_cuda(hidden, weight)
    ref = _reference_router(hidden, weight)
    _assert_bf16_router_close(actual, ref)


def test_glm5_router_gemm_rank_counts_zero_invalid_rows():
    from batchgen.moe.routing import glm5_router_gemm_cuda

    torch.manual_seed(99)
    device = torch.device("cuda")
    world_size = 4
    bucket = 8
    hidden = (torch.randn(world_size * bucket, 128, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    weight = (torch.randn(64, 128, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    rank_counts = torch.tensor([3, 8, 0, 5], device=device, dtype=torch.int64)

    rows = torch.arange(world_size * bucket, device=device)
    valid = (rows % bucket) < rank_counts[rows // bucket]
    hidden_poisoned = hidden.clone()
    hidden_poisoned[~valid] = torch.tensor(float("nan"), device=device, dtype=torch.bfloat16)

    actual = glm5_router_gemm_cuda(
        hidden_poisoned,
        weight,
        rank_token_counts=rank_counts,
        bucket_size=bucket,
        world_size=world_size,
    )
    ref_valid = _reference_router(hidden[valid].contiguous(), weight)

    torch.testing.assert_close(actual[~valid], torch.zeros_like(actual[~valid]), atol=0, rtol=0)
    _assert_bf16_router_close(actual[valid], ref_valid)


def test_glm5_router_gemm_valid_rows_are_bucket_stable():
    from batchgen.moe.routing import glm5_router_gemm_cuda

    torch.manual_seed(100)
    device = torch.device("cuda")
    world_size = 3
    bucket = 7
    hidden = (torch.randn(world_size * bucket, 96, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    weight = (torch.randn(48, 96, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    rank_counts = torch.tensor([7, 3, 5], device=device, dtype=torch.int64)
    rows = torch.arange(world_size * bucket, device=device)
    valid = (rows % bucket) < rank_counts[rows // bucket]

    padded_logits = glm5_router_gemm_cuda(
        hidden,
        weight,
        rank_token_counts=rank_counts,
        bucket_size=bucket,
        world_size=world_size,
    )
    compact_logits = glm5_router_gemm_cuda(hidden[valid].contiguous(), weight)
    torch.testing.assert_close(padded_logits[valid], compact_logits, atol=0, rtol=0)


@pytest.mark.parametrize("world_size,bucket", [(16, 1), (16, 8), (16, 32)])
def test_glm5_router_gemm_production_graph_envelope_rank_counts(world_size, bucket):
    from batchgen.moe.routing import glm5_router_gemm_cuda

    torch.manual_seed(20260511 + bucket)
    device = torch.device("cuda")
    hidden = (
        torch.randn(world_size * bucket, 6144, device=device, dtype=torch.float32) * 0.1
    ).to(torch.bfloat16)
    weight = (
        torch.randn(256, 6144, device=device, dtype=torch.float32) * 0.1
    ).to(torch.bfloat16)
    rank_counts = (
        (torch.arange(world_size, device=device, dtype=torch.int64) * 7) % (bucket + 1)
    )
    rank_counts[0] = bucket
    if world_size > 1:
        rank_counts[1] = 0

    rows = torch.arange(world_size * bucket, device=device)
    valid = (rows % bucket) < rank_counts[rows // bucket]
    hidden_poisoned = hidden.clone()
    hidden_poisoned[~valid] = torch.tensor(float("nan"), device=device, dtype=torch.bfloat16)

    actual = glm5_router_gemm_cuda(
        hidden_poisoned,
        weight,
        rank_token_counts=rank_counts,
        bucket_size=bucket,
        world_size=world_size,
    )
    ref_valid = _reference_router(hidden[valid].contiguous(), weight)

    torch.testing.assert_close(actual[~valid], torch.zeros_like(actual[~valid]), atol=0, rtol=0)
    _assert_bf16_router_close(actual[valid], ref_valid)


def test_glm5_router_gemm_cuda_graph_replay_uses_device_rank_counts():
    from batchgen.moe.routing import glm5_router_gemm_cuda

    torch.manual_seed(777)
    device = torch.device("cuda")
    world_size = 2
    bucket = 4
    hidden = (torch.randn(world_size * bucket, 64, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    weight = (torch.randn(32, 64, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    rank_counts = torch.full((world_size,), bucket, device=device, dtype=torch.int64)
    graph_out = torch.empty(world_size * bucket, 32, device=device, dtype=torch.float32)

    glm5_router_gemm_cuda(
        hidden,
        weight,
        router_logits=graph_out,
        rank_token_counts=rank_counts,
        bucket_size=bucket,
        world_size=world_size,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        glm5_router_gemm_cuda(
            hidden,
            weight,
            router_logits=graph_out,
            rank_token_counts=rank_counts,
            bucket_size=bucket,
            world_size=world_size,
        )

    new_hidden = (torch.randn_like(hidden.float()) * 0.1).to(torch.bfloat16)
    new_counts = torch.tensor([1, 3], device=device, dtype=torch.int64)
    rows = torch.arange(world_size * bucket, device=device)
    valid = (rows % bucket) < new_counts[rows // bucket]
    new_hidden[~valid] = torch.tensor(float("nan"), device=device, dtype=torch.bfloat16)
    hidden.copy_(new_hidden)
    rank_counts.copy_(new_counts)

    graph.replay()
    expected = glm5_router_gemm_cuda(
        new_hidden,
        weight,
        rank_token_counts=new_counts,
        bucket_size=bucket,
        world_size=world_size,
    )
    torch.testing.assert_close(graph_out, expected, atol=0, rtol=0)


@pytest.mark.parametrize("m", [160, 320, 600])
def test_glm5_tensorcore_router_matches_bf16_reference_and_graph_replay(m):
    from batchgen.moe.routing import FusedGateContext

    torch.manual_seed(20260818 + m)
    device = torch.device("cuda")
    hidden_base = (
        torch.randn(600, 6144, device=device, dtype=torch.float32) * 0.1
    ).to(torch.bfloat16)
    weight = (
        torch.randn(256, 6144, device=device, dtype=torch.float32) * 0.1
    ).to(torch.bfloat16)
    output = torch.empty(m, 256, device=device, dtype=torch.float32)
    context = FusedGateContext(weight, router_bias=None, topk=8)
    context.warmup(hidden_base)

    actual = context.router_forward(hidden_base[:m], logits=output)
    ref = _reference_router(hidden_base[:m], weight)
    _assert_bf16_router_close(actual, ref)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        context.router_forward(hidden_base[:m], logits=output)

    replacement = (
        torch.randn_like(hidden_base[:m].float()) * 0.1
    ).to(torch.bfloat16)
    hidden_base[:m].copy_(replacement)
    graph.replay()
    ref_replay = _reference_router(replacement, weight)
    _assert_bf16_router_close(output, ref_replay)


@pytest.mark.parametrize("m", [200, 240, 280, 320, 400])
def test_glm5_triton_router_matches_bf16_reference_and_graph_replay(m):
    from batchgen_kernels.triton.glm5_router_gemm import glm5_router_gemm

    torch.manual_seed(20260819 + m)
    device = torch.device("cuda")
    hidden = (
        torch.randn(m, 6144, device=device, dtype=torch.float32) * 0.1
    ).to(torch.bfloat16)
    weight = (
        torch.randn(256, 6144, device=device, dtype=torch.float32) * 0.1
    ).to(torch.bfloat16)
    output = torch.empty(m, 256, device=device, dtype=torch.float32)

    glm5_router_gemm(hidden, weight, output)
    _assert_bf16_router_close(output, _reference_router(hidden, weight))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        glm5_router_gemm(hidden, weight, output)

    replacement = (
        torch.randn_like(hidden.float()) * 0.1
    ).to(torch.bfloat16)
    hidden.copy_(replacement)
    graph.replay()
    _assert_bf16_router_close(output, _reference_router(replacement, weight))


# --- split-K router GEMM -----------------------------------------------------
#
# Numerical contract, derived from the arithmetic each kernel actually runs.
#
# Both kernels take bf16 inputs and accumulate in float32. A bf16 value carries
# an 8-bit significand, so a product of two of them needs at most 16 significand
# bits and is *exact* in float32, and bf16 shares float32's exponent range, so
# nothing over- or underflows at these magnitudes. Every bit of error therefore
# comes from the ADDITIONS, and what separates the two kernels is only how deep
# a chain of additions a term has to survive:
#
#   serial-K   one register accumulator walks all C = K / block_k chunks, so a
#              chunk is rounded into the accumulator and then carried through
#              every later chunk -- depth C.
#   split-K    S programs each walk C / S chunks into their own accumulator,
#              and the reduce kernel sums the S partials in a fixed chain,
#              split 0 upwards -- depth C / S + S.
#
# For a float32 summation of exact terms in which term t passes through D_t
# rounded additions, the first-order error bound is eps * sum_t D_t |t| (Higham,
# "Accuracy and Stability of Numerical Algorithms", 2nd ed., sec. 3.1). With the
# per-block_k chunk partials as the terms, that gives, per output element,
#
#     |kernel - truth|  <=  CHAIN_BOUND_HEADROOM * depth * eps * sum_c |chunk_c|
#
# The scale is the sum of the CHUNK MAGNITUDES, not max|logit|. A chain's error
# tracks the terms it sums, and the chunks partly cancel against each other, so
# they are larger than the logit they add up to -- which is why a bound written
# against max|logit| is not a bound on either kernel at all: measured against
# float64, the pre-split serial-K kernel overruns a 32 * eps * max|logit| budget
# by 1.92x-2.35x across the eight regimes below.
#
# CHAIN_BOUND_HEADROOM = 4 covers the one depth the model above does not
# enumerate: the order in which the tensor core sums the block_k products inside
# a single chunk is hardware defined, and it is identical for the two kernels
# (both hand the same block_k tile to the same tl.dot), so it cannot be derived
# here. The factor is calibrated rather than guessed. Measured against a float64
# oracle over these eight regimes, each kernel's actual error reaches at most
# 0.36 (serial-K) and 0.21 (split-K) of its own depth * eps * sum_c|chunk_c|, so
# headroom 1 already passes with 2.8x margin and headroom 4 keeps about 11x. The
# bound still sits orders of magnitude below the defects this gate exists to
# catch. Injected into a CPU model of the split-K chain on the realistic m=192
# activations, dropping one chunk overshoots the headroom-4 bound by 5.8e3x and
# rounding the chunks to bf16 before accumulating them overshoots it by 1.4e2x,
# while the undamaged chains sit at 0.007x (serial-K) and 0.018x of their bounds.
#
# "truth" is a float64 matmul of the same bf16 inputs. Its own error is bounded
# by K * eps64 * sum_d |h_d w_d|, about 1e-8 of the float32 bounds above, so it
# is an oracle here and not a third kernel whose rounding has to be reproduced.
#
# This file owns the numerical bound because only a GPU can execute the chains
# the bound is about. tests/test_glm5_router_gemm_split_k_policy.py pins the same
# depth formula and headroom on the CPU side and asserts the two files agree, so
# the suites still move together.

GLM5_ROUTER_K = 6144
GLM5_ROUTER_EXPERTS = 256
GLM5_ROUTER_TOPK = 8
F32_EPS = 2.0**-23
CHAIN_BOUND_HEADROOM = 4.0


def chain_addition_depth(chunks: int, split_k: int) -> int:
    """Chunk-level rounded additions the deepest term of a chunk survives.

    ``split_k == 1`` is the serial-K kernel's single accumulator over every
    chunk; otherwise each of ``split_k`` accumulators walks ``chunks //
    split_k`` chunks and the reduce kernel chains the ``split_k`` partials.
    """
    if chunks < 1 or split_k < 1:
        raise ValueError(f"chunks={chunks} split_k={split_k} must both be >= 1")
    if split_k == 1:
        return chunks
    if chunks % split_k:
        raise ValueError(
            f"split_k={split_k} does not partition {chunks} chunks equally"
        )
    return chunks // split_k + split_k


def _fp64_truth_and_chunk_magnitude(hidden: torch.Tensor, weight: torch.Tensor):
    """float64 truth, per-element ``sum_c |chunk_c|``, and the chunk count."""
    from batchgen_kernels.triton.glm5_router_gemm_config import ROUTER_BLOCK_K

    k = hidden.shape[1]
    assert k % ROUTER_BLOCK_K == 0, (
        f"k={k} is not a whole number of block_k={ROUTER_BLOCK_K} chunks"
    )
    h64 = hidden.to(torch.float64)
    w64 = weight.to(torch.float64)
    truth = h64.matmul(w64.t())
    chunk_magnitude = torch.zeros_like(truth)
    for start in range(0, k, ROUTER_BLOCK_K):
        span = slice(start, start + ROUTER_BLOCK_K)
        chunk_magnitude += h64[:, span].matmul(w64[:, span].t()).abs()
    return truth, chunk_magnitude, k // ROUTER_BLOCK_K


def _chain_error_bound(
    chunk_magnitude: torch.Tensor, chunks: int, split_k: int
) -> torch.Tensor:
    depth = chain_addition_depth(chunks, split_k)
    return CHAIN_BOUND_HEADROOM * depth * F32_EPS * chunk_magnitude


def _assert_inside_chain_bound(
    actual: torch.Tensor,
    truth: torch.Tensor,
    chunk_magnitude: torch.Tensor,
    chunks: int,
    split_k: int,
    what: str,
) -> None:
    bound = _chain_error_bound(chunk_magnitude, chunks, split_k)
    error = (actual.to(torch.float64) - truth).abs()
    over = error > bound
    n_over = int(over.sum().item())
    if n_over:
        worst = int(error.div(bound.clamp_min(1e-300)).argmax().item())
        row, col = worst // error.shape[1], worst % error.shape[1]
        raise AssertionError(
            f"{what}: {n_over}/{error.numel()} elements exceed the "
            f"depth-{chain_addition_depth(chunks, split_k)} chain bound; worst "
            f"at ({row},{col}) error={float(error[row, col]):.3e} "
            f"bound={float(bound[row, col]):.3e} "
            f"sum_c|chunk_c|={float(chunk_magnitude[row, col]):.3e}"
        )


def _router_activations(m: int, regime: str, seed: int):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    weight = (
        torch.randn(GLM5_ROUTER_EXPERTS, GLM5_ROUTER_K, device=device, dtype=torch.float32)
        * 0.05
    ).to(torch.bfloat16)
    if regime == "random":
        hidden = (
            torch.randn(m, GLM5_ROUTER_K, device=device, dtype=torch.float32) * 0.1
        )
    elif regime == "realistic":
        # Post-norm decode hidden states: a few dominant channels and a
        # per-row gain spread, which is the worst case for a regrouped sum.
        hidden = torch.randn(m, GLM5_ROUTER_K, device=device, dtype=torch.float32) * 0.5
        hidden[:, ::512] *= 40.0
        hidden *= (
            1.0 + torch.rand(m, 1, device=device, dtype=torch.float32) * 3.0
        )
    else:  # pragma: no cover - guarded by parametrize
        raise AssertionError(f"unknown regime {regime}")
    return hidden.to(torch.bfloat16), weight


def _run_router(kernel, hidden, weight, split_k=None, partials=None):
    output = torch.empty(
        hidden.shape[0], weight.shape[0], device=hidden.device, dtype=torch.float32
    )
    kernel(hidden, weight, output, split_k=split_k, partials=partials)
    return output


def _assert_bitwise(actual: torch.Tensor, expected: torch.Tensor, what: str) -> None:
    mismatch = int((actual.view(torch.int32) != expected.view(torch.int32)).sum().item())
    if mismatch:
        raise AssertionError(
            f"{what}: {mismatch}/{expected.numel()} elements differ bitwise, "
            f"max_abs={float((actual - expected).abs().max().item())}"
        )


def _assert_split_k_selected(m: int) -> None:
    from batchgen_kernels.triton.glm5_router_gemm_config import (
        ROUTER_SPLIT_K,
        select_router_split_k,
    )

    assert select_router_split_k(m, GLM5_ROUTER_K) == ROUTER_SPLIT_K > 1, (
        f"m={m} is expected to take the split-K path"
    )


@pytest.mark.parametrize("m", [192, 256, 384, 512])
@pytest.mark.parametrize("regime", ["random", "realistic"])
def test_glm5_router_each_kernel_stays_inside_its_own_chain_bound(m, regime):
    # Each kernel is asserted against float64 truth with the bound derived from
    # the addition chain it executes -- not against the other kernel, whose own
    # rounding error is not a specification either of them has to reproduce.
    from batchgen_kernels.triton.glm5_router_gemm import glm5_router_gemm
    from batchgen_kernels.triton.glm5_router_gemm_config import ROUTER_SPLIT_K

    _assert_split_k_selected(m)
    hidden, weight = _router_activations(m, regime, seed=20261016 + m)

    serial = _run_router(glm5_router_gemm, hidden, weight, split_k=1)
    split = _run_router(glm5_router_gemm, hidden, weight)
    truth, chunk_magnitude, chunks = _fp64_truth_and_chunk_magnitude(hidden, weight)

    _assert_inside_chain_bound(
        serial, truth, chunk_magnitude, chunks, 1, f"m={m} {regime} serial-K"
    )
    _assert_inside_chain_bound(
        split,
        truth,
        chunk_magnitude,
        chunks,
        ROUTER_SPLIT_K,
        f"m={m} {regime} split-K",
    )


@pytest.mark.parametrize("m", [192, 256, 384, 512])
@pytest.mark.parametrize("regime", ["random", "realistic"])
def test_glm5_router_split_k_is_no_less_accurate_than_serial_k(m, regime):
    # Accuracy non-regression, the assert that actually protects the swap: the
    # shorter chains must not move the logits further from truth than the single
    # long chain they replace. Shorter chains should and do win here.
    from batchgen_kernels.triton.glm5_router_gemm import glm5_router_gemm

    _assert_split_k_selected(m)
    hidden, weight = _router_activations(m, regime, seed=20261016 + m)

    serial = _run_router(glm5_router_gemm, hidden, weight, split_k=1)
    split = _run_router(glm5_router_gemm, hidden, weight)
    truth, _, _ = _fp64_truth_and_chunk_magnitude(hidden, weight)

    d_serial = float((serial.to(torch.float64) - truth).abs().max().item())
    d_split = float((split.to(torch.float64) - truth).abs().max().item())
    assert d_split <= d_serial, (
        f"m={m} {regime}: split-K is further from float64 truth than serial-K "
        f"({d_split:.3e} > {d_serial:.3e})"
    )


@pytest.mark.parametrize("m", [192, 256, 384, 512])
@pytest.mark.parametrize("regime", ["random", "realistic"])
def test_glm5_router_split_k_topk_set_matches_and_reorders_only_near_ties(m, regime):
    # The routed expert SET is the contract the MoE path consumes: dispatch
    # selects by expert id, so a permutation of the slots cannot change which
    # experts run. Set agreement is therefore mandatory and exact.
    #
    # Slot ORDER is not a well-posed float32 contract. Two logits closer
    # together than a kernel's own error bound cannot be ranked by that kernel,
    # and requiring split-K to reproduce serial-K's ranking of such a pair would
    # require reproducing serial-K's rounding error. The order is allowed to
    # differ only where float64 says the pair is unresolvable at this precision,
    # and there split-K must land on the float64 order -- so a reordering is
    # excused only when split-K is the kernel that is right.
    from batchgen_kernels.triton.glm5_router_gemm import glm5_router_gemm

    _assert_split_k_selected(m)
    hidden, weight = _router_activations(m, regime, seed=20261016 + m)

    serial = _run_router(glm5_router_gemm, hidden, weight, split_k=1)
    split = _run_router(glm5_router_gemm, hidden, weight)
    truth, chunk_magnitude, chunks = _fp64_truth_and_chunk_magnitude(hidden, weight)

    serial_topk = serial.topk(GLM5_ROUTER_TOPK, dim=-1).indices
    split_topk = split.topk(GLM5_ROUTER_TOPK, dim=-1).indices
    truth_topk = truth.topk(GLM5_ROUTER_TOPK, dim=-1).indices

    assert torch.equal(
        split_topk.sort(dim=-1).values, serial_topk.sort(dim=-1).values
    ), f"m={m} {regime}: top-k expert sets disagree"

    # The serial-K bound is the wider of the two, so it is the resolution below
    # which neither kernel can be held to a ranking.
    resolution = _chain_error_bound(chunk_magnitude, chunks, 1)
    reordered = (~(split_topk == serial_topk).all(dim=-1)).nonzero().flatten()
    for row in reordered.tolist():
        ranked = truth[row].topk(GLM5_ROUTER_TOPK).values
        gap = float((ranked[:-1] - ranked[1:]).min().item())
        bound = float(resolution[row, serial_topk[row]].max().item())
        assert gap <= bound, (
            f"m={m} {regime} row {row}: top-k order differs although the "
            f"float64 adjacent gap {gap:.3e} is resolvable at this precision "
            f"(chain bound {bound:.3e})"
        )
        assert torch.equal(split_topk[row], truth_topk[row]), (
            f"m={m} {regime} row {row}: top-k order differs on a float64 "
            f"near-tie (gap {gap:.3e} <= bound {bound:.3e}) and split-K does "
            f"not match the float64 order {truth_topk[row].tolist()} "
            f"(split-K {split_topk[row].tolist()})"
        )


@pytest.mark.parametrize("valid_rows", [192, 200, 256, 384])
def test_glm5_router_split_k_valid_rows_are_bucket_m_independent(valid_rows):
    # The graphed router runs on a padded bucket. A valid row's logits must be
    # bitwise identical whether the launch carried 512 rows or only its own,
    # and the padded rows carry poison here to prove they stay inert.
    from batchgen_kernels.triton.glm5_router_gemm import glm5_router_gemm

    _assert_split_k_selected(valid_rows)
    hidden, weight = _router_activations(512, "realistic", seed=5120 + valid_rows)
    hidden[valid_rows:] = torch.tensor(
        float("nan"), device=hidden.device, dtype=torch.bfloat16
    )

    padded = _run_router(glm5_router_gemm, hidden, weight)
    compact = _run_router(glm5_router_gemm, hidden[:valid_rows].contiguous(), weight)

    _assert_bitwise(
        padded[:valid_rows], compact, f"valid rows at bucket 512 vs {valid_rows}"
    )


def test_glm5_router_split_k_is_launch_to_launch_deterministic():
    from batchgen_kernels.triton.glm5_router_gemm import glm5_router_gemm

    hidden, weight = _router_activations(512, "realistic", seed=909)
    first = _run_router(glm5_router_gemm, hidden, weight)
    second = _run_router(glm5_router_gemm, hidden, weight)
    _assert_bitwise(second, first, "repeat launch")


def test_glm5_router_split_k_graph_replay_matches_eager():
    from batchgen_kernels.triton.glm5_router_gemm import glm5_router_gemm
    from batchgen_kernels.triton.glm5_router_gemm_config import ROUTER_SPLIT_K

    hidden, weight = _router_activations(512, "realistic", seed=20260819)
    output = torch.empty(512, GLM5_ROUTER_EXPERTS, device=hidden.device, dtype=torch.float32)

    glm5_router_gemm(hidden, weight, output)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        glm5_router_gemm(hidden, weight, output)

    replacement, _ = _router_activations(512, "random", seed=4242)
    hidden.copy_(replacement)
    graph.replay()
    torch.cuda.synchronize()

    eager = _run_router(glm5_router_gemm, hidden, weight)
    _assert_bitwise(output, eager, "graph replay vs eager split-K")

    truth, chunk_magnitude, chunks = _fp64_truth_and_chunk_magnitude(hidden, weight)
    _assert_inside_chain_bound(
        output, truth, chunk_magnitude, chunks, ROUTER_SPLIT_K, "graph replay output"
    )


def test_glm5_router_split_k_preallocated_partials_match_internal_buffer():
    from batchgen_kernels.triton.glm5_router_gemm import glm5_router_gemm
    from batchgen_kernels.triton.glm5_router_gemm_config import ROUTER_SPLIT_K

    hidden, weight = _router_activations(512, "random", seed=31337)
    internal = _run_router(glm5_router_gemm, hidden, weight)
    scratch = torch.empty(
        (ROUTER_SPLIT_K, 512, GLM5_ROUTER_EXPERTS),
        device=hidden.device,
        dtype=torch.float32,
    )
    supplied = _run_router(glm5_router_gemm, hidden, weight, partials=scratch)
    _assert_bitwise(supplied, internal, "supplied partials buffer")


@pytest.mark.parametrize("split_k", [0, -1, 5, 7, 128])
def test_glm5_router_split_k_rejects_an_inexact_split_count(split_k):
    from batchgen_kernels.triton.glm5_router_gemm import glm5_router_gemm
    from batchgen_kernels.triton.glm5_router_gemm_config import RouterGemmShapeError

    hidden, weight = _router_activations(256, "random", seed=11)
    output = torch.empty(256, GLM5_ROUTER_EXPERTS, device=hidden.device, dtype=torch.float32)
    with pytest.raises(RouterGemmShapeError):
        glm5_router_gemm(hidden, weight, output, split_k=split_k)


def test_glm5_router_split_k_rejects_a_mismatched_partials_buffer():
    from batchgen_kernels.triton.glm5_router_gemm import glm5_router_gemm
    from batchgen_kernels.triton.glm5_router_gemm_config import (
        ROUTER_SPLIT_K,
        RouterGemmShapeError,
    )

    hidden, weight = _router_activations(256, "random", seed=12)
    output = torch.empty(256, GLM5_ROUTER_EXPERTS, device=hidden.device, dtype=torch.float32)
    wrong_shape = torch.empty(
        (ROUTER_SPLIT_K - 1, 256, GLM5_ROUTER_EXPERTS),
        device=hidden.device,
        dtype=torch.float32,
    )
    with pytest.raises(RouterGemmShapeError):
        glm5_router_gemm(hidden, weight, output, partials=wrong_shape)

    wrong_dtype = torch.empty(
        (ROUTER_SPLIT_K, 256, GLM5_ROUTER_EXPERTS),
        device=hidden.device,
        dtype=torch.bfloat16,
    )
    with pytest.raises(RouterGemmShapeError):
        glm5_router_gemm(hidden, weight, output, partials=wrong_dtype)


@pytest.mark.parametrize("m", [192, 256, 384, 512])
def test_glm5_router_split_k_still_matches_the_bf16_reference(m):
    from batchgen_kernels.triton.glm5_router_gemm import glm5_router_gemm

    _assert_split_k_selected(m)
    hidden, weight = _router_activations(m, "random", seed=818 + m)
    split = _run_router(glm5_router_gemm, hidden, weight)
    _assert_bf16_router_close(split, _reference_router(hidden, weight))
