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
