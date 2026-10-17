"""GLM-5 DeepEP low-latency: assumptions that only the real extension can
confirm. Inert without CUDA + a usable deep_ep build + an EP process group.

These pin the three facts the CPU contract tests have to take on faith:

1. the RDMA buffer size the build reports equals the derived identity, so the
   KV-page charge computed on paper is the charge actually paid;
2. ``low_latency_dispatch(use_fp8=True)`` returns scales whose last two
   dimensions are a transposed view of a contiguous
   ``[E_local, hidden/128, stride]`` allocation -- the repack in
   ``Glm5MoEGraphSegment._forward_deepep`` is wrong if this ever changes;
3. dispatch + combine replay inside a captured CUDA graph.

Run on the node with the EP world already initialized, e.g.
``torchrun --nproc_per_node 8 -m pytest tests/test_glm52_deepep_ll_gpu.py``.
"""
import os

import pytest
import torch

from batchgen.moe.deepep_ll import (
    DeepEPLowLatencyExchange,
    deepep_unavailable_reason,
)

GLM52_HIDDEN = 6144
GLM52_TOPK = 8
GLM52_EXPERTS = 256
TOKENS_PER_RANK = 64

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="DeepEP LL requires CUDA")


def _require_exchange():
    import torch.distributed as dist

    if not dist.is_initialized():
        pytest.skip("DeepEP LL requires an initialized process group")
    world = dist.get_world_size()
    reason = deepep_unavailable_reason(
        hidden=GLM52_HIDDEN, topk=GLM52_TOPK, num_experts=GLM52_EXPERTS,
        num_ranks=world, max_tokens_per_rank=TOKENS_PER_RANK, use_fp8=True)
    if reason is not None:
        pytest.skip(f"DeepEP unusable here: {reason}")
    if GLM52_EXPERTS % world:
        pytest.skip(f"world {world} does not divide {GLM52_EXPERTS} experts")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    return DeepEPLowLatencyExchange(
        dist.group.WORLD,
        max_tokens_per_rank=TOKENS_PER_RANK,
        hidden=GLM52_HIDDEN,
        num_experts=GLM52_EXPERTS,
        num_local_experts=GLM52_EXPERTS // world,
    ), world


def _derived_rdma_bytes(*, tokens_per_rank, hidden, num_experts):
    """csrc/config.hpp LowLatencyLayout + get_low_latency_rdma_size_hint."""
    num_scales = hidden // 128
    dispatch_msg = 16 + max(hidden * 2, hidden + num_scales * 4)
    combine_msg = num_scales * 4 + hidden * 2
    send = max(tokens_per_rank * dispatch_msg,
               num_experts * tokens_per_rank * combine_msg)
    recv = max(num_experts * tokens_per_rank * dispatch_msg,
               num_experts * tokens_per_rank * combine_msg)
    signaling = ((num_experts * 4 + 127) // 128) * 128
    total = send * 2 + recv * 2 + signaling * 2
    return ((total + 128) // 128) * 128


def test_rdma_buffer_matches_the_derived_identity():
    exchange, _world = _require_exchange()
    expected = _derived_rdma_bytes(
        tokens_per_rank=TOKENS_PER_RANK, hidden=GLM52_HIDDEN,
        num_experts=GLM52_EXPERTS)
    assert exchange.rdma_bytes == expected, (
        f"DeepEP reports {exchange.rdma_bytes} B, derived {expected} B; the "
        "KV-page charge was computed from the derived identity")


def test_fp8_dispatch_layout_matches_the_repack_assumption():
    exchange, world = _require_exchange()
    e = exchange.num_local_experts
    stride = exchange.expert_stride
    device = torch.device("cuda", torch.cuda.current_device())

    x = torch.randn(TOKENS_PER_RANK, GLM52_HIDDEN,
                    dtype=torch.bfloat16, device=device)
    topk = torch.randint(0, GLM52_EXPERTS, (TOKENS_PER_RANK, GLM52_TOPK),
                         dtype=torch.int64, device=device)
    recv_x, recv_scales, recv_count, _handle = exchange.dispatch_fp8(x, topk)

    assert recv_x.shape == (e, stride, GLM52_HIDDEN)
    assert recv_x.dtype == torch.float8_e4m3fn
    assert recv_count.shape == (e,) and recv_count.dtype == torch.int32
    assert recv_scales.shape == (e, stride, GLM52_HIDDEN // 128)
    assert recv_scales.dtype == torch.float32
    # Column-major in the last two dims: the (1, 2) transpose must be the
    # contiguous [e, hidden/128, stride] allocation the repack reads.
    assert not recv_scales.is_contiguous()
    assert recv_scales.transpose(1, 2).is_contiguous()
    assert recv_scales.stride() == (
        stride * (GLM52_HIDDEN // 128), 1, stride)


def test_dispatch_combine_replay_inside_a_captured_graph():
    exchange, _world = _require_exchange()
    e = exchange.num_local_experts
    stride = exchange.expert_stride
    device = torch.device("cuda", torch.cuda.current_device())

    x = torch.randn(TOKENS_PER_RANK, GLM52_HIDDEN,
                    dtype=torch.bfloat16, device=device)
    topk = torch.randint(0, GLM52_EXPERTS, (TOKENS_PER_RANK, GLM52_TOPK),
                         dtype=torch.int64, device=device)
    weights = torch.rand(TOKENS_PER_RANK, GLM52_TOPK,
                         dtype=torch.float32, device=device)
    out = torch.empty(TOKENS_PER_RANK, GLM52_HIDDEN,
                      dtype=torch.bfloat16, device=device)
    scale_buf = torch.zeros(GLM52_HIDDEN // 128, e * stride,
                            dtype=torch.float32, device=device)

    # warm the kernels outside capture
    recv_x, recv_scales, _count, handle = exchange.dispatch_fp8(x, topk)
    exchange.combine(
        torch.zeros(e, stride, GLM52_HIDDEN, dtype=torch.bfloat16,
                    device=device),
        topk, weights, handle, out=out)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        recv_x, recv_scales, _count, handle = exchange.dispatch_fp8(x, topk)
        scale_buf.view(GLM52_HIDDEN // 128, e, stride).copy_(
            recv_scales.transpose(1, 2).permute(1, 0, 2))
        exchange.combine(
            recv_x.view(e, stride, GLM52_HIDDEN).to(torch.bfloat16),
            topk, weights, handle, out=out)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
