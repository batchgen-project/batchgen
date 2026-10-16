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
# The split-K path sums the same block_k chunk partials as the serial-K kernel
# and differs only in how it groups them, so the two results differ by a
# float32 reassociation. The budget below is a multiple of float32 eps scaled
# by the magnitude of the logits; tests/test_glm5_router_gemm_split_k_policy.py
# measures the block-level worst case the budget is derived from and asserts
# the same number, so the two suites move together.

GLM5_ROUTER_K = 6144
GLM5_ROUTER_EXPERTS = 256
GLM5_ROUTER_TOPK = 8
SPLIT_K_ULP_BUDGET = 32
F32_EPS = 2.0**-23


def _split_k_budget(reference: torch.Tensor) -> float:
    return SPLIT_K_ULP_BUDGET * F32_EPS * float(reference.abs().max())


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
def test_glm5_router_split_k_agrees_with_serial_k_on_topk_and_logits(m, regime):
    from batchgen_kernels.triton.glm5_router_gemm import glm5_router_gemm

    _assert_split_k_selected(m)
    hidden, weight = _router_activations(m, regime, seed=20261016 + m)

    serial = _run_router(glm5_router_gemm, hidden, weight, split_k=1)
    split = _run_router(glm5_router_gemm, hidden, weight)

    budget = _split_k_budget(serial)
    observed = float((split - serial).abs().max().item())
    assert observed <= budget, (
        f"m={m} {regime}: logits max_abs_diff {observed:.3e} exceeds the "
        f"float32 reassociation budget {budget:.3e}"
    )

    serial_topk = serial.topk(GLM5_ROUTER_TOPK, dim=-1).indices
    split_topk = split.topk(GLM5_ROUTER_TOPK, dim=-1).indices
    assert torch.equal(split_topk, serial_topk), (
        f"m={m} {regime}: top-k expert ids disagree in position"
    )
    assert torch.equal(
        split_topk.sort(dim=-1).values, serial_topk.sort(dim=-1).values
    ), f"m={m} {regime}: top-k expert sets disagree"


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

    serial = _run_router(glm5_router_gemm, hidden, weight, split_k=1)
    assert float((output - serial).abs().max().item()) <= _split_k_budget(serial)


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
