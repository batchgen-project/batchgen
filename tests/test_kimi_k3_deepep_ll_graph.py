"""K3 decode graph, DeepEP low-latency EP exchange (slice 3): the graph
segment's own dataflow around the exchange, with CPU stand-ins for the
kernels, the TP collectives and DeepEP itself.

Contract under test:
* the TP gather of the column-sliced latent assembles this rank's own
  rows' FULL latent in column order (the send buffer of the dispatch);
* dispatch output is consumed in place as the strided [E * mtp, k] Marlin
  activation with mtp = world * max_lt (the pool's fixed LL stride), with
  DeepEP's recv_count as the expert counts and the LL pointer tensors;
* the LL combine (which applies the top-k weights) writes this rank's
  [lt, k] rows into the static buffer that feeds norm -> up_proj.
"""
import contextlib
from types import SimpleNamespace

import pytest
import torch

import batchgen.models.moonshotai.kimi_linear.moe_cuda_graph_segments as k3_moe_graph
from batchgen.models.moonshotai.kimi_linear.moe_cuda_graph_segments import (
    K3MoEGraphBufferPool,
    K3MoEGraphSegment,
)


class _FakeGroup:
    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


class _FakeLowLatencyExchange:
    """Single-process stand-in: routes this rank's tokens to its own local
    experts the way DeepEP packs them (expert-major, rows [0, count)), and
    combines with the top-k weights like the LL combine does."""

    def __init__(self, group, *, max_tokens_per_rank, hidden, num_experts,
                 num_local_experts):
        self.num_ranks = group.size()
        self.max_tokens_per_rank = int(max_tokens_per_rank)
        self.hidden = int(hidden)
        self.num_experts = int(num_experts)
        self.num_local = int(num_local_experts)
        self.rdma_bytes = 0
        self.expert_start = 0
        self._routes = None

    def dispatch(self, x, topk_idx):
        assert x.dtype == torch.bfloat16 and topk_idx.dtype == torch.int64
        assert x.shape[1] == self.hidden
        mtp = self.num_ranks * self.max_tokens_per_rank
        recv = torch.zeros(self.num_local, mtp, self.hidden, dtype=torch.bfloat16)
        counts = torch.zeros(self.num_local, dtype=torch.int32)
        routes = []
        for t in range(x.shape[0]):
            for kk in range(topk_idx.shape[1]):
                e = int(topk_idx[t, kk])
                if not (self.expert_start <= e < self.expert_start + self.num_local):
                    continue
                le = e - self.expert_start
                recv[le, int(counts[le])] = x[t]
                routes.append((t, kk, le, int(counts[le])))
                counts[le] += 1
        self._routes = routes
        return recv, counts, ("handle",)

    def combine(self, x, topk_idx, topk_weights, handle, out):
        assert handle == ("handle",)
        assert x.shape == (self.num_local, self.num_ranks * self.max_tokens_per_rank, self.hidden)
        acc = torch.zeros(out.shape, dtype=torch.float32)
        for t, kk, le, pos in self._routes:
            acc[t] += float(topk_weights[t, kk]) * x[le, pos].float()
        out.copy_(acc.to(out.dtype))
        return out


def _build(monkeypatch, *, tp_size=4, tp_rank=1, world=8, bucket=8, latent=128,
           hidden=16, intermediate=64, num_local=2):
    monkeypatch.setattr(
        "batchgen.moe.deepep_ll.DeepEPLowLatencyExchange", _FakeLowLatencyExchange
    )
    monkeypatch.setattr("batchgen.moe.deepep_ll._EXCHANGES", {})
    pool = K3MoEGraphBufferPool(
        world_size=world, tp_size=tp_size, num_local_experts=num_local,
        intermediate_size=intermediate, latent_size=latent, hidden_size=hidden,
        top_k=16, expert_buckets=[bucket], device=torch.device("cpu"),
        deepep_group=_FakeGroup(world),
    )
    pool.setup()
    bufs = pool.get(bucket)
    lt = bufs.local_tokens
    cols = latent // tp_size

    segment = K3MoEGraphSegment.__new__(K3MoEGraphSegment)
    segment.pool = pool
    segment.device = torch.device("cpu")
    segment.hidden_size = hidden
    segment.latent_size = latent
    segment.intermediate_size = intermediate
    segment.top_k = 16
    segment.num_experts = world * num_local
    segment.world_size = world
    segment.rank = tp_rank
    segment.expert_start = 0
    segment.num_local_experts = num_local
    segment.tp_size = tp_size
    segment.tp_rank = tp_rank
    segment.tp_group = object()
    segment.latent_sharded = True
    segment.latent_cols = cols
    segment.shared_stream = None
    segment.fused_front = None
    segment.fused_gate_kernel = False

    # every token routes to local experts 0 and 1 (weights 0.25 / 0.5) plus
    # 14 remote experts; the last row is padding (-1, weight 0)
    idx = torch.full((lt, 16), 5, dtype=torch.int64)
    idx[:, 0] = 0
    idx[:, 1] = 1
    w = torch.full((lt, 16), 0.125, dtype=torch.float32)
    w[:, 0] = 0.25
    w[:, 1] = 0.5
    idx[-1] = -1
    w[-1] = 0.0
    gate_out = (idx.clone(), w.clone())
    segment.moe = SimpleNamespace(
        gate=lambda hidden_states: gate_out,
        shared_experts=SimpleNamespace(
            _ffn=lambda x: torch.zeros(x.shape[0], hidden, dtype=torch.bfloat16)),
    )
    # the column slice this rank holds of every group row: row r, col c = r + 100*c
    slice_rows = torch.arange(tp_size * lt, dtype=torch.float32).unsqueeze(1)
    slice_cols = torch.arange(cols, dtype=torch.float32).unsqueeze(0) * 100
    latent_slice_value = (slice_rows + slice_cols).to(torch.bfloat16)
    segment.resident = SimpleNamespace(
        shard=SimpleNamespace(
            gate_B_ptrs=None, gate_scales_ptrs=None, up_B_ptrs=None,
            up_scales_ptrs=None, down_B_ptrs=None, down_scales_ptrs=None,
        ),
        down_proj=lambda x: latent_slice_value.clone(),
        norm=None,
        up_proj=lambda x: x[:, :hidden].clone(),
        latent_tp_size=tp_size,
        layer_idx=7,
    )

    calls = {}

    def fake_s1(dispatched, intermediate_buf, counts, starts, *rest):
        calls["s1"] = (dispatched, counts, starts, rest)

    def fake_s3(intermediate_buf, down_B, c_ptrs, down_scales, starts, counts, *rest):
        calls["s3"] = (c_ptrs, starts, counts)
        # identity experts: the received rows come back unchanged
        pool.ll_expert_output.copy_(calls["s1"][0])

    monkeypatch.setattr(k3_moe_graph, "marlin_grouped_stage1_fused_mxfp4_situ", fake_s1)
    monkeypatch.setattr(k3_moe_graph, "marlin_grouped_m16_mxfp4", fake_s3)
    monkeypatch.setattr(k3_moe_graph.torch.cuda, "device",
                        lambda device: contextlib.nullcontext())

    def fake_gather(out, src, group=None):
        # every rank holds the same values, so the gather is a broadcast copy:
        # the slice gather lands [tp, tp*lt, cols], the up_proj gather [tp*lt, k]
        if out.dim() == 3:
            assert out.shape == (tp_size, tp_size * lt, cols) and src.shape == (tp_size * lt, cols)
            out.copy_(src.unsqueeze(0).expand(tp_size, tp_size * lt, cols))
        else:
            out.view(tp_size, lt, latent).copy_(src.unsqueeze(0).expand(tp_size, lt, latent))

    monkeypatch.setattr(k3_moe_graph.dist, "all_gather_into_tensor", fake_gather)
    monkeypatch.setattr(k3_moe_graph.dist, "all_reduce", lambda t, group=None: None)

    def refuse(*args, **kwargs):
        raise AssertionError("the DeepEP path must not run the NCCL EP exchange")

    monkeypatch.setattr(k3_moe_graph, "dispatch_scatter_3d", refuse)
    monkeypatch.setattr(k3_moe_graph, "reduce_weighted_scatter_fp32", refuse)
    return segment, pool, bufs, calls, latent_slice_value, w


def test_pool_fixed_ll_stride_and_buffers(monkeypatch):
    _, pool, bufs, _, _, _ = _build(monkeypatch)
    assert bufs.ll_gathered.is_contiguous()
    max_lt = bufs.local_tokens
    assert pool.deepep.max_tokens_per_rank == max_lt
    assert pool.deepep.hidden == 128 and pool.deepep.num_experts == 16
    mtp = pool.ll_max_tokens_padded
    assert mtp == 8 * max_lt
    assert torch.equal(pool.ll_expert_starts, torch.arange(2, dtype=torch.int32) * mtp)
    assert pool.ll_intermediate.shape[0] == 2 * mtp
    assert pool.ll_expert_output.shape[0] == 2 * mtp
    assert bufs.ll_send.shape == (max_lt, 128)
    assert bufs.ll_topk_idx.dtype == torch.int64
    assert bufs.ll_gathered.shape == (4, 4 * max_lt, 32)
    assert bufs.ll_combined.shape == (max_lt, 128)


def test_deepep_branch_assembles_own_rows_and_combines(monkeypatch):
    segment, pool, bufs, calls, latent_slice_value, w = _build(monkeypatch)
    lt, tp, cols, latent = bufs.local_tokens, 4, 32, 128
    padded = torch.zeros(tp * lt, 16, dtype=torch.bfloat16)
    out = segment.forward(
        padded=padded, local=padded[:lt], num_valid_tokens=torch.tensor(lt - 1),
    )
    assert out["moe_output"].shape == (tp * lt, 16)

    # own rows' full latent: row i = the tp column slices of group row
    # tp_rank*lt + i side by side
    mine = latent_slice_value[1 * lt:2 * lt]
    expected_send = mine.unsqueeze(1).expand(lt, tp, cols).reshape(lt, latent)
    assert torch.equal(bufs.ll_send, expected_send)
    assert torch.equal(bufs.ll_topk_idx[:, :2], torch.tensor([[0, 1]] * (lt - 1) + [[-1, -1]]))

    # S1/S3 ran on the dispatch output in the fixed LL layout
    dispatched, counts, starts, rest = calls["s1"]
    mtp = pool.ll_max_tokens_padded
    assert dispatched.shape == (2 * mtp, latent)
    assert counts.tolist() == [lt - 1, lt - 1]           # padding row unrouted
    assert starts is pool.ll_expert_starts
    assert rest[4] is pool.ll_s1_C_ptrs and rest[-3] == mtp
    assert torch.equal(dispatched[:lt - 1], expected_send[:lt - 1])          # expert 0 rows
    assert torch.equal(dispatched[mtp:mtp + lt - 1], expected_send[:lt - 1])  # expert 1 rows
    assert calls["s3"][0] is pool.ll_s3_C_ptrs and calls["s3"][2] is counts

    # combine = sum over this rank's experts of weight * identity(x)
    expected = (expected_send.float() * (0.25 + 0.5)).to(torch.bfloat16)
    expected[-1] = 0
    assert torch.equal(bufs.ll_combined, expected)
    assert torch.equal(out["moe_output"][lt:2 * lt], expected[:, :16])


def test_planner_gate_requires_sharded_latent(monkeypatch):
    try:
        from batchgen.models.moonshotai.kimi_linear.planner import k3_deepep_low_latency
    except Exception as exc:  # noqa: BLE001 - planner needs the JIT core_engine
        pytest.skip(f"planner import unavailable here: {exc}")
    assert k3_deepep_low_latency(sharded_latent=False) is False
    monkeypatch.setattr("batchgen.moe.deepep_ll.deepep_available", lambda: True)
    assert k3_deepep_low_latency(sharded_latent=True) is True
    # --k3-moe-exchange nccl forces the NCCL path even with the build present
    assert k3_deepep_low_latency(sharded_latent=True, moe_exchange="nccl") is False
    assert k3_deepep_low_latency(sharded_latent=True, moe_exchange="deepep") is True
    monkeypatch.setattr("batchgen.moe.deepep_ll.deepep_available", lambda: False)
    assert k3_deepep_low_latency(sharded_latent=True) is False
    with pytest.raises(RuntimeError, match="k3_moe_exchange=deepep"):
        k3_deepep_low_latency(sharded_latent=True, moe_exchange="deepep")
    with pytest.raises(ValueError):
        k3_deepep_low_latency(sharded_latent=True, moe_exchange="bogus")


def test_smaller_bucket_views_stay_contiguous(monkeypatch):
    monkeypatch.setattr(
        "batchgen.moe.deepep_ll.DeepEPLowLatencyExchange", _FakeLowLatencyExchange
    )
    monkeypatch.setattr("batchgen.moe.deepep_ll._EXCHANGES", {})
    pool = K3MoEGraphBufferPool(
        world_size=8, tp_size=4, num_local_experts=2, intermediate_size=64,
        latent_size=128, hidden_size=16, top_k=16, expert_buckets=[4, 8, 16],
        device=torch.device("cpu"), deepep_group=_FakeGroup(8),
    )
    pool.setup()
    for bucket in (4, 8, 16):
        bufs = pool.get(bucket)
        lt = bucket // 4
        assert bufs.ll_gathered.shape == (4, 4 * lt, 32) and bufs.ll_gathered.is_contiguous()
        assert bufs.ll_send.shape == (lt, 128) and bufs.ll_combined.shape == (lt, 128)


def test_exchange_is_process_lifetime(monkeypatch):
    monkeypatch.setattr(
        "batchgen.moe.deepep_ll.DeepEPLowLatencyExchange", _FakeLowLatencyExchange
    )
    monkeypatch.setattr("batchgen.moe.deepep_ll._EXCHANGES", {})
    group = _FakeGroup(8)
    kw = dict(world_size=8, tp_size=4, num_local_experts=2, intermediate_size=64,
              latent_size=128, hidden_size=16, top_k=16, expert_buckets=[8],
              device=torch.device("cpu"), deepep_group=group)
    a = K3MoEGraphBufferPool(**kw); a.setup(); a.release()
    b = K3MoEGraphBufferPool(**kw); b.setup()
    assert b.deepep is a.deepep   # rebuilt pool reuses the exchange, no re-init


def test_pool_rejects_stride_mismatch(monkeypatch):
    monkeypatch.setattr(
        "batchgen.moe.deepep_ll.DeepEPLowLatencyExchange", _FakeLowLatencyExchange
    )
    monkeypatch.setattr("batchgen.moe.deepep_ll._EXCHANGES", {})
    # world*max_lt not a multiple of 16 -> the graph's rounded stride differs
    pool = K3MoEGraphBufferPool(
        world_size=3, tp_size=4, num_local_experts=2, intermediate_size=64,
        latent_size=128, hidden_size=16, top_k=16, expert_buckets=[8],
        device=torch.device("cpu"), deepep_group=_FakeGroup(3),
    )
    with pytest.raises(ValueError, match="expert stride"):
        pool.setup()
