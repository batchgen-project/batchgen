"""GLM-5 DeepEP low-latency: assumptions that only the real extension can
confirm. Inert without CUDA + a usable deep_ep build + an EP process group.

These pin the three facts the CPU contract tests have to take on faith:

1. the RDMA buffer size the build reports equals the derived identity, so the
   KV-page charge computed on paper is the charge actually paid;
2. ``low_latency_dispatch(use_fp8=True)`` returns scales whose last two
   dimensions are a transposed view of a contiguous
   ``[E_local, hidden/128, stride]`` allocation -- the repack in
   ``Glm5MoEGraphSegment._forward_deepep`` is wrong if this ever changes;
3. dispatch + combine replay inside a captured CUDA graph;
4. the send / receive split (``split_phases=True``, DeepEP's
   ``return_recv_hook=True`` with the hook issued at once) delivers exactly
   the single-kernel exchange's bytes -- receive rows, scales, per-expert and
   per-source counts, and combine output -- eager and in graph replay.

Run on the node with the EP world already initialized, e.g.
``torchrun --nproc_per_node 8 -m pytest tests/test_glm52_deepep_ll_gpu.py``.
"""
import os

import pytest
import torch

from batchgen.moe.deepep_ll import (
    DeepEPLowLatencyExchange,
    deepep_unavailable_reason,
    get_low_latency_exchange,
)

GLM52_HIDDEN = 6144
GLM52_TOPK = 8
GLM52_EXPERTS = 256
TOKENS_PER_RANK = 64

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="DeepEP LL requires CUDA")


def _require_deepep_world():
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
    return world


def _require_exchange():
    import torch.distributed as dist

    world = _require_deepep_world()
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


# --------------------------------------------------------------------------
# send / receive split == single kernel
# --------------------------------------------------------------------------
def _split_case_inputs(device, live_rows):
    """Deterministic per-rank inputs; rows >= live_rows are unrouted (-1)."""
    import torch.distributed as dist

    g = torch.Generator().manual_seed(20261021 + 97 * dist.get_rank()
                                      + live_rows)
    x = torch.randn(TOKENS_PER_RANK, GLM52_HIDDEN, generator=g).to(
        torch.bfloat16)
    # distinct experts per token, as the router produces
    topk = torch.stack([torch.randperm(GLM52_EXPERTS, generator=g)[:GLM52_TOPK]
                        for _ in range(TOKENS_PER_RANK)]).to(torch.int64)
    weights = torch.rand(TOKENS_PER_RANK, GLM52_TOPK, generator=g)
    topk[live_rows:] = -1
    weights[live_rows:] = 0.0
    return x.to(device), topk.to(device), weights.to(device)


def _canonical_receive(recv_x, recv_scales, recv_count, handle, world):
    """The receive space in a run-independent order.

    DeepEP packs each local expert's rows per source rank, and both the slot a
    sender claims and the block a source rank lands in come from atomics, so
    the ROW ORDER is not reproducible run to run while the row SET is. Rows are
    keyed by (source rank, source token index) -- ``layout_range`` holds
    (count, begin) per source rank as two int32 halves, ``src_info`` the
    source token index -- and returned in key order with their bytes."""
    src_info, layout = handle[0].cpu(), handle[1].cpu()
    count = recv_count.cpu()
    xs = recv_x.view(torch.uint8).cpu()
    sc = recv_scales.cpu()
    out = []
    for e in range(xs.shape[0]):
        lay = layout[e]
        n = (lay & 0xFFFFFFFF).to(torch.int64)
        begin = (lay >> 32).to(torch.int64)
        assert int(n.sum()) == int(count[e])
        keys, rows = [], []
        for r in range(world):
            idx = torch.arange(int(begin[r]), int(begin[r]) + int(n[r]))
            rows.append(idx)
            keys.append(r * TOKENS_PER_RANK + src_info[e, idx].to(torch.int64))
        rows = torch.cat(rows)
        keys = torch.cat(keys)
        order = torch.argsort(keys)
        out.append((n, keys[order], xs[e, rows[order]], sc[e, rows[order]]))
    return out


def _expert_outputs(recv_x, recv_scales):
    """A deterministic function of each received row's content only (its
    dequantized value), so the combine sees the same per-(expert, token)
    operand whatever row order the dispatch chose."""
    e, stride, h = recv_x.shape
    deq = (recv_x.float().view(e, stride, h // 128, 128)
           * recv_scales.unsqueeze(-1)).view(e, stride, h)
    return deq.to(torch.bfloat16)


def _split_round_trip(exchange, x, topk, weights, out, split):
    recv_x, recv_scales, recv_count, handle = exchange.dispatch_fp8(
        x, topk, split_phases=split)
    exchange.combine(_expert_outputs(recv_x, recv_scales), topk, weights,
                     handle, out=out, split_phases=split)
    return recv_x, recv_scales, recv_count, handle


def _assert_same_exchange(a, b, world, what):
    ca = _canonical_receive(*a[:4], world)
    cb = _canonical_receive(*b[:4], world)
    assert torch.equal(a[2].cpu(), b[2].cpu()), f"{what}: recv_count"
    for e, (ea, eb) in enumerate(zip(ca, cb)):
        for name, ta, tb in zip(("per-source counts", "source keys",
                                 "rows", "scales"), ea, eb):
            assert torch.equal(ta, tb), f"{what}: expert {e} {name}"


@pytest.mark.parametrize("live_rows", [TOKENS_PER_RANK, 40])
def test_split_phases_deliver_the_single_kernel_bytes(live_rows):
    """Bit identity of the two launch modes on the real extension, eager and
    CUDA-graph replay, against each other and against themselves."""
    import torch.distributed as dist

    world = _require_deepep_world()
    # the process-lifetime exchange the model uses, one buffer for both modes
    exchange = get_low_latency_exchange(
        dist.group.WORLD, max_tokens_per_rank=TOKENS_PER_RANK,
        hidden=GLM52_HIDDEN, num_experts=GLM52_EXPERTS,
        num_local_experts=GLM52_EXPERTS // world)
    device = torch.device("cuda", torch.cuda.current_device())
    x, topk, weights = _split_case_inputs(device, live_rows)

    eager = {}
    for split in (False, True):
        out = torch.empty(TOKENS_PER_RANK, GLM52_HIDDEN,
                          dtype=torch.bfloat16, device=device)
        res = _split_round_trip(exchange, x, topk, weights, out, split)
        torch.cuda.synchronize()
        eager[split] = res + (out.clone(),)
    _assert_same_exchange(eager[False], eager[True], world, "eager")
    assert torch.equal(eager[False][4], eager[True][4]), "eager combine"
    assert torch.isfinite(eager[True][4][:live_rows].float()).all()

    graphs = {}
    for split in (False, True):
        out = torch.empty(TOKENS_PER_RANK, GLM52_HIDDEN,
                          dtype=torch.bfloat16, device=device)
        # warm outside capture, as the graph manager does
        _split_round_trip(exchange, x, topk, weights, out, split)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            res = _split_round_trip(exchange, x, topk, weights, out, split)
        graphs[split] = (graph, res, out)
    for rep in range(2):
        for split in (False, True):
            graph, res, out = graphs[split]
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            _assert_same_exchange(eager[False], res + (out,), world,
                                  f"graph split={split} replay {rep}")
            assert torch.equal(out, eager[False][4]), (
                f"graph split={split} replay {rep}: combine")
