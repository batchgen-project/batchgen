"""GLM-5 decode MoE over the DeepEP low-latency exchange (CPU contract tests).

The GPU numerics are a node gate; what is checkable here is the dataflow the
graph segment captures. Contract under test:

* buffer sizing — the DeepEP receive space is expert-major with a uniform
  per-expert stride ``num_ranks * tokens/rank``, bucket-independent, and the
  stride must stay TileM-aligned for the grouped FP8 GEMM;
* the uniform-stride segment table replaces the device-written ragged one, and
  the GEMMs' live row counts come from DeepEP's ``recv_count``;
* the FP8 cast is folded into the dispatch, and the per-expert column-major
  scale blocks are repacked into the single ``[hidden/128, rows]`` tensor the
  GEMM requires, element for element;
* the LL combine applies the top-k weights and writes this rank's rows, so
  ``reduce_weighted_scatter`` and the NCCL pair never run;
* the router kernel dispatch is resolved on the row count THIS path routes —
  one rank's rows, not the all-gathered space — and reads its range from the
  kernel policy rather than from literals repeated here;
* M-independence — stale rows past ``recv_count[e]`` cannot reach the output;
* the shared-expert side-stream overlap — the fork opens right after the router
  GEMM (before the top-k and the exchange), the join closes after the combine,
  the accumulate consumes the pool-owned staging buffer, and the stream/event
  are process-lifetime resources that are never minted inside a capture;
* the exchange runs split into send and receive launches (``split_phases``) for
  both dispatch and combine;
* the top-k is the one-warp-per-row gate writing the exchange's int64 ids
  straight into ``ll_topk_idx`` -- no int32 index buffer and no copy kernel.
"""
import contextlib
import inspect
import sys
from types import SimpleNamespace

import pytest
import torch

import batchgen.moe.deepep_ll as deepep_ll
import batchgen.models.glm.glm5.moe_cuda_graph_segments as glm5_moe_graph
import torch.distributed as dist
from batchgen.models.glm.glm5.moe_cuda_graph_segments import (
    Glm5MoEGraphBufferPool,
    Glm5MoEGraphSegment,
)
from batchgen.moe.grouped_fp8_blockwise_moe import grouped_fp8_tile_m

H = 256          # hidden (multiple of 128 for the scale blocks)
N_INTER = 128
TOPK = 4
E_LOCAL = 2
WORLD = 4
BUCKET = 16
STRIDE = WORLD * BUCKET          # DeepEP rows per local expert
ROWS = E_LOCAL * STRIDE


class _FakeGroup:
    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


class _FakeExchange:
    """Single-process stand-in for ``DeepEPLowLatencyExchange``.

    Routes this rank's rows to its own local experts the way DeepEP packs them
    (expert-major, rows [0, count)), emits the FP8 pair with DeepEP's
    column-major scale layout, and combines with the top-k weights reading only
    the live rows.
    """

    def __init__(self, group, *, max_tokens_per_rank, hidden, num_experts,
                 num_local_experts):
        self.num_ranks = group.size()
        self.max_tokens_per_rank = int(max_tokens_per_rank)
        self.hidden = int(hidden)
        self.num_experts = int(num_experts)
        self.num_local_experts = int(num_local_experts)
        self.rdma_bytes = 0
        self.expert_start = 0
        self._routes = None
        self.dispatch_calls = 0
        self.combine_calls = 0
        # ``split_phases`` of every call, in call order
        self.dispatch_split = []
        self.combine_split = []

    @property
    def expert_stride(self):
        return self.num_ranks * self.max_tokens_per_rank

    @property
    def recv_rows(self):
        return self.num_local_experts * self.expert_stride

    def dispatch_fp8(self, x, topk_idx, *, split_phases=False):
        assert x.dtype == torch.bfloat16, x.dtype
        assert topk_idx.dtype == torch.int64, topk_idx.dtype
        assert x.shape[1] == self.hidden
        assert x.shape[0] <= self.max_tokens_per_rank
        self.dispatch_calls += 1
        self.dispatch_split.append(bool(split_phases))
        stride, e, h = self.expert_stride, self.num_local_experts, self.hidden
        recv = torch.zeros(e, stride, h, dtype=torch.float8_e4m3fn)
        # DeepEP allocates [e, h/128, stride] contiguous, then hands back the
        # (1, 2) transpose -- scales are column-major in the last two dims.
        scales_base = torch.zeros(e, h // 128, stride, dtype=torch.float32)
        counts = torch.zeros(e, dtype=torch.int32)
        routes = []
        for t in range(x.shape[0]):
            for kk in range(topk_idx.shape[1]):
                g = int(topk_idx[t, kk])
                if not (self.expert_start <= g
                        < self.expert_start + self.num_local_experts):
                    continue
                le = g - self.expert_start
                pos = int(counts[le])
                recv[le, pos] = x[t].to(torch.float8_e4m3fn)
                # a value that identifies (expert, row, scale block) exactly
                for blk in range(h // 128):
                    scales_base[le, blk, pos] = 1000 * (le + 1) + 10 * pos + blk
                routes.append((t, kk, le, pos))
                counts[le] += 1
        self._routes = routes
        self.scales_base = scales_base
        return recv, scales_base.transpose(1, 2), counts, ("handle",)

    def combine(self, x, topk_idx, topk_weights, handle, out, *,
                split_phases=False):
        assert handle == ("handle",)
        assert x.shape == (self.num_local_experts, self.expert_stride,
                           self.hidden), x.shape
        self.combine_calls += 1
        self.combine_split.append(bool(split_phases))
        acc = torch.zeros(out.shape, dtype=torch.float32)
        # only the live rows named by the handle are read -- this is what makes
        # the padded extent inert
        for t, kk, le, pos in self._routes:
            acc[t] += float(topk_weights[t, kk]) * x[le, pos].float()
        out.copy_(acc.to(out.dtype))
        return out


class _RecordingOverlap:
    """Stand-in for the shared-expert side stream.

    The real fork/join is a pair of CUDA dependency edges, which a CPU test
    cannot execute. What it CAN pin is the dataflow those edges exist for: the
    fork opens before anything touches the exchange, the join closes after the
    combine, and the accumulate reads a staging buffer that is only valid once
    the join has run.

    The last part is made observable rather than assumed: between fork and join
    the staged rows are WITHHELD (overwritten with a sentinel) and restored only
    by ``join``. An accumulate issued before the join therefore reads the
    sentinel and fails a value assertion, instead of passing because the CPU
    happened to run the two in the right order anyway.
    """

    WITHHELD = -31.0

    def __init__(self, order, staging):
        self.order = order
        self._staging = staging
        self._staged = None
        self.forks = 0
        self.joins = 0

    @contextlib.contextmanager
    def fork(self):
        self.forks += 1
        self.order.append("fork_enter")
        yield
        self._staged = self._staging.clone()
        self._staging.fill_(self.WITHHELD)
        self.order.append("fork_exit")

    def join(self):
        self.joins += 1
        if self._staged is None:
            raise AssertionError("join ran without a preceding fork")
        self._staging.copy_(self._staged)
        self.order.append("join")


def _install_fake_exchange(monkeypatch):
    monkeypatch.setattr(
        deepep_ll, "DeepEPLowLatencyExchange", _FakeExchange)
    monkeypatch.setattr(deepep_ll, "_EXCHANGES", {})


def _make_pool(monkeypatch, *, buckets=(BUCKET,), world=WORLD,
               num_local=E_LOCAL, mtpr=None, configure=True):
    _install_fake_exchange(monkeypatch)
    pool = Glm5MoEGraphBufferPool(
        world_size=world,
        hidden_size=H,
        num_experts_per_tok=TOPK,
        num_local_experts=num_local,
        intermediate_size=N_INTER,
        device=torch.device("cpu"),
        bucket_sizes=list(buckets),
    )
    if configure:
        pool.configure_deepep(
            _FakeGroup(world),
            max_tokens_per_rank=max(buckets) if mtpr is None else mtpr,
            num_experts_global=world * num_local,
        )
    return pool


# --------------------------------------------------------------------------
# Buffer sizing / shape resolution
# --------------------------------------------------------------------------
def test_pool_allocates_the_expert_major_receive_space(monkeypatch):
    pool = _make_pool(monkeypatch)
    pool.setup()
    bufs = pool.get(BUCKET)
    assert pool.ll_stride == STRIDE
    assert pool.deepep.recv_rows == ROWS
    assert bufs.capacity == ROWS
    assert bufs.intermediate.shape == (ROWS, N_INTER)
    assert bufs.expert_out.shape == (ROWS, H)
    assert bufs.inter_fp8.shape == (ROWS, N_INTER)
    # scales are the transposed [dim/128, rows] the grouped GEMM takes
    assert bufs.ll_x_scale.shape == (H // 128, ROWS)
    assert bufs.inter_scale.shape == (N_INTER // 128, ROWS)
    assert bufs.ll_x_scale.is_contiguous() and bufs.inter_scale.is_contiguous()
    # the router runs on LOCAL rows, not the all-gathered row space
    assert bufs.router_logits.shape == (BUCKET, 256)
    # the gate writes int64 ids into ll_topk_idx; no int32 index buffer exists
    assert bufs.topk_indices.numel() == 0
    assert bufs.ll_topk_idx.shape == (BUCKET, TOPK)
    assert bufs.ll_topk_idx.dtype == torch.int64
    assert bufs.local_moe_output.shape == (BUCKET, H)
    # the NCCL-only staging buffers are gone, not merely unused
    assert bufs.all_tokens.shape == (BUCKET, H)
    assert bufs.dispatched_x.shape[0] == 0
    assert bufs.x_fp8.shape[0] == 0
    assert bufs.routed_global_output.shape[0] == 0
    assert bufs.topk_pos.numel() == 0


def test_segment_table_is_the_constant_uniform_stride(monkeypatch):
    pool = _make_pool(monkeypatch)
    pool.setup()
    bufs = pool.get(BUCKET)
    expected = torch.arange(E_LOCAL + 1, dtype=torch.int32) * STRIDE
    assert torch.equal(bufs.ll_cu_seqlens, expected)
    # the ragged path's device-written table is replaced, not shadowed
    assert bufs.cu_seqlens is bufs.ll_cu_seqlens


def test_receive_space_is_bucket_independent(monkeypatch):
    pool = _make_pool(monkeypatch, buckets=(4, 8, 16))
    pool.setup()
    stride = 4 * 16         # sized by the LARGEST bucket (= tokens/rank)
    for bucket in (4, 8, 16):
        bufs = pool.get(bucket)
        assert pool.ll_stride == stride
        assert bufs.capacity == E_LOCAL * stride
        assert bufs.expert_out.shape == (E_LOCAL * stride, H)
        assert bufs.ll_x_scale.shape == (H // 128, E_LOCAL * stride)
        # only the per-rank row space follows the bucket
        assert bufs.padded.shape == (bucket, H)
        assert bufs.ll_topk_idx.shape == (bucket, TOPK)
        assert bufs.local_moe_output.shape == (bucket, H)


def test_stride_must_stay_tilem_aligned(monkeypatch):
    """The grouped FP8 GEMM addresses each expert's scale tile at
    cu_seqlens[e] / TileM and traps on device if a start is unaligned."""
    pool = _make_pool(monkeypatch, buckets=(6,), world=3, mtpr=6)
    assert (3 * 6) % glm5_moe_graph.ROW_ALIGN != 0
    with pytest.raises(RuntimeError, match="not a multiple of 64"):
        pool.setup()


def test_configure_rejects_inconsistent_requests(monkeypatch):
    _install_fake_exchange(monkeypatch)
    kw = dict(world_size=WORLD, hidden_size=H, num_experts_per_tok=TOPK,
              num_local_experts=E_LOCAL, intermediate_size=N_INTER,
              device=torch.device("cpu"), bucket_sizes=[BUCKET])
    pool = Glm5MoEGraphBufferPool(**kw)
    with pytest.raises(ValueError, match="below the largest decode bucket"):
        pool.configure_deepep(_FakeGroup(WORLD), max_tokens_per_rank=BUCKET - 1,
                              num_experts_global=WORLD * E_LOCAL)
    with pytest.raises(ValueError, match="num_experts_global"):
        pool.configure_deepep(_FakeGroup(WORLD), max_tokens_per_rank=BUCKET,
                              num_experts_global=WORLD * E_LOCAL + 1)
    pool.configure_deepep(_FakeGroup(WORLD), max_tokens_per_rank=BUCKET,
                          num_experts_global=WORLD * E_LOCAL)
    with pytest.raises(RuntimeError, match="already configured"):
        pool.configure_deepep(_FakeGroup(WORLD), max_tokens_per_rank=BUCKET * 2,
                              num_experts_global=WORLD * E_LOCAL)
    pool.setup()
    with pytest.raises(RuntimeError, match="must run before setup"):
        pool.configure_deepep(_FakeGroup(WORLD), max_tokens_per_rank=BUCKET,
                              num_experts_global=WORLD * E_LOCAL)


def test_nccl_pool_is_unchanged_and_keeps_ragged_capacity(monkeypatch):
    """The default path must not inherit the DeepEP row space."""
    pool = _make_pool(monkeypatch, configure=False)
    assert pool.deepep_enabled is False
    pool.setup()
    bufs = pool.get(BUCKET)
    assert pool.deepep is None and pool.ll_stride == 0
    assert bufs.capacity == pool._capacity_for(BUCKET)
    assert bufs.all_tokens.shape == (WORLD * BUCKET, H)
    assert bufs.dispatched_x.shape == (bufs.capacity, H)
    assert bufs.ll_topk_idx.numel() == 0 and bufs.ll_cu_seqlens.numel() == 0


def test_exchange_is_process_lifetime(monkeypatch):
    """A mid-run prefill wave drops and rebuilds the graph pool; the NVSHMEM
    buffer must survive it (re-init would repeat the handshake and, with
    explicitly_destroy, risk the exit-barrier hang)."""
    _install_fake_exchange(monkeypatch)
    group = _FakeGroup(WORLD)
    kw = dict(world_size=WORLD, hidden_size=H, num_experts_per_tok=TOPK,
              num_local_experts=E_LOCAL, intermediate_size=N_INTER,
              device=torch.device("cpu"), bucket_sizes=[BUCKET])
    pool_a = Glm5MoEGraphBufferPool(**kw)
    pool_a.configure_deepep(group, max_tokens_per_rank=BUCKET,
                            num_experts_global=WORLD * E_LOCAL)
    pool_a.setup()
    first = pool_a.deepep
    pool_a.release()
    assert pool_a.deepep is first

    pool_b = Glm5MoEGraphBufferPool(**kw)
    pool_b.configure_deepep(group, max_tokens_per_rank=BUCKET,
                            num_experts_global=WORLD * E_LOCAL)
    pool_b.setup()
    assert pool_b.deepep is first


# --------------------------------------------------------------------------
# Segment dataflow
# --------------------------------------------------------------------------
def _build_segment(monkeypatch, *, rank=1, bucket=BUCKET, valid_rows=None,
                   split_k_rows=(), shared_value=0.0, shared_dtype=None,
                   shared_rows=None, shared_foreign=False):
    pool = _make_pool(monkeypatch, buckets=(bucket,))
    pool.setup()
    bufs = pool.get(bucket)
    valid = bucket - 1 if valid_rows is None else valid_rows

    segment = Glm5MoEGraphSegment.__new__(Glm5MoEGraphSegment)
    segment.pool = pool
    segment.device = torch.device("cpu")
    segment.hidden_size = H
    segment.intermediate_size = N_INTER
    segment.num_experts_per_tok = TOPK
    segment.num_local_experts = E_LOCAL
    segment.world_size = WORLD
    segment.rank = rank
    segment.expert_start = rank * E_LOCAL
    # DeepEP owns the same expert window as GLM's rank: local expert i is the
    # global id rank * E_local + i (asserted by _validate_deepep_graph_safe).
    pool.deepep.expert_start = segment.expert_start
    segment.routed_scaling_factor = 1.0
    segment.deepep_ll = True
    segment.comm = SimpleNamespace(disabled=False)
    segment.gate_bias_fp32 = torch.zeros(256, dtype=torch.float32)
    segment.gate_weight_bf16 = torch.zeros(256, H, dtype=torch.bfloat16)

    # Which router kernel ran, and on how many rows. Both stand-ins record
    # rather than compute: the logits are zeroed either way, and ``fake_gate``
    # below writes the routed ids this suite asserts on.
    router_calls = {"wgmma": [], "split_k": []}

    def fake_router_forward(x, logits=None):
        router_calls["wgmma"].append(tuple(x.shape))
        return logits.fill_(0.0)

    def fake_split_k_router(hidden, weight, out):
        router_calls["split_k"].append(tuple(hidden.shape))
        return out.fill_(0.0)

    segment.router_context = SimpleNamespace(router_forward=fake_router_forward)

    # The real dispatch reads the split-K range from
    # ``batchgen_kernels.triton``, whose package import pulls in Triton, so the
    # CPU contract patches that one seam and names the row counts standing for
    # "inside the kernel policy's range". That the real seam reads the policy's
    # own bounds is asserted separately, below.
    wants_split_k = set(int(rows) for rows in split_k_rows)

    def fake_kernel_for(rows):
        return fake_split_k_router if int(rows) in wants_split_k else None

    monkeypatch.setattr(
        glm5_moe_graph, "_graphed_router_split_k_kernel", fake_kernel_for)
    # Call order across the three things the overlap has to interleave
    # correctly: the shared expert, the exchange, and the join.
    order = []

    # The real shared expert writes its down projection into the ``out=``
    # staging buffer and returns it. ``shared_dtype`` / ``shared_rows`` /
    # ``shared_foreign`` instead return a tensor of its own, which the segment
    # must refuse: the accumulate would read a buffer nothing wrote.
    def fake_shared_expert(x, out=None):
        order.append("shared_expert")
        assert out is not None, "the DeepEP path must pass the staging buffer"
        if shared_dtype is None and shared_rows is None and not shared_foreign:
            out.fill_(shared_value)
            return out
        rows = x.shape[0] if shared_rows is None else shared_rows
        return torch.full((rows, H), shared_value,
                          dtype=shared_dtype or torch.bfloat16)

    segment.moe = SimpleNamespace(
        layer_idx=5,
        fp8_gate_w3d=torch.zeros(E_LOCAL, N_INTER, H, dtype=torch.uint8),
        fp8_up_w3d=torch.zeros(E_LOCAL, N_INTER, H, dtype=torch.uint8),
        fp8_gate_ws3d=torch.zeros(1), fp8_up_ws3d=torch.zeros(1),
        fp8_down_w3d=torch.zeros(E_LOCAL, H, N_INTER, dtype=torch.uint8),
        fp8_down_ws3d=torch.zeros(1),
        shared_expert_forward=fake_shared_expert,
    )

    for name in ("dispatch_fp8", "combine"):
        real = getattr(pool.deepep, name)

        def tracing(*a, _real=real, _name=name, **kw):
            order.append("dispatch" if _name == "dispatch_fp8" else "combine")
            return _real(*a, **kw)

        setattr(pool.deepep, name, tracing)

    overlap = _RecordingOverlap(order, bufs.shared_out)
    segment._shared_overlap = overlap

    # The gate writes global expert ids: every live row picks this rank's two
    # local experts plus two remote ones; row `valid` onwards is padding. It
    # writes the exchange's int64 buffer itself -- there is no int32 result to
    # copy.
    def fake_gate(logits, bias, *, k, routed_scaling_factor, topk_indices,
                  topk_weights, rank_token_counts=None, rows_per_rank=None,
                  num_valid_tokens=None):
        assert topk_indices is bufs.ll_topk_idx
        assert topk_indices.dtype == torch.int64
        assert rank_token_counts is not None and rank_token_counts.numel() == 1
        assert rows_per_rank == bucket
        assert num_valid_tokens is None
        live = int(rank_token_counts[0])
        topk_indices.fill_(0)
        topk_indices[:, 0] = segment.expert_start
        topk_indices[:, 1] = segment.expert_start + 1
        topk_indices[:, 2] = 200
        topk_indices[:, 3] = 201
        topk_weights.fill_(0.125)
        topk_weights[:, 0] = 0.25
        topk_weights[:, 1] = 0.5
        topk_indices[live:] = -1
        topk_weights[live:] = 0.0
        return topk_indices, topk_weights

    calls = {}
    # Launch order of the routed chain's local kernels (the exchange/overlap
    # order above is asserted separately and stays free of these).
    chain = []

    def fake_prepare(s1_x, s1_output, s3_x, s3_output, seqlens, cu_seqlens,
                     avg, *, s1_tma_desc, s3_tma_desc, tiles, cu_tiles):
        chain.append("prepare")
        plan = SimpleNamespace(
            tile_m=grouped_fp8_tile_m(avg), s1_tma_desc=s1_tma_desc,
            s3_tma_desc=s3_tma_desc, tiles=tiles, cu_tiles=cu_tiles)
        calls["prepare"] = dict(
            s1_x=s1_x, s1_output=s1_output, s3_x=s3_x, s3_output=s3_output,
            seqlens=seqlens, cu_seqlens=cu_seqlens, avg=avg, plan=plan)
        return plan

    def fake_repack(recv_scales, recv_count, out, span_align):
        """The kernel's contract on CPU: per expert, copy exactly the
        span_align-rounded live columns of DeepEP's block; leave the rest."""
        chain.append("repack")
        calls["repack"] = dict(recv_scales=recv_scales, recv_count=recv_count,
                               out=out, span_align=span_align)
        e_, s_, kb_ = recv_scales.shape
        blocks = recv_scales.transpose(1, 2)        # [e, kb, s] contiguous
        for i in range(e_):
            rc = int(recv_count[i])
            span = -(-rc // span_align) * span_align
            out[:, i * s_:i * s_ + span] = blocks[i, :, :span]

    def fake_s1(x_fp8, x_scale, gate_w, up_w, gate_ws, up_ws, seqlens,
                cu_seqlens, avg, output=None, tma_plan=None):
        chain.append("s1")
        calls["s1"] = dict(x_fp8=x_fp8, x_scale=x_scale, seqlens=seqlens,
                           cu_seqlens=cu_seqlens, avg=avg, output=output,
                           tma_plan=tma_plan)
        return output

    def fake_quant(x, seqlens, cu_seqlens, out, scale):
        chain.append("quant")
        calls["quant"] = dict(x=x, seqlens=seqlens, cu_seqlens=cu_seqlens)

    def fake_s3(x_fp8, x_scale, down_w, down_ws, seqlens, cu_seqlens, avg,
                output=None, tma_plan=None):
        chain.append("s3")
        calls["s3"] = dict(x_fp8=x_fp8, seqlens=seqlens, cu_seqlens=cu_seqlens,
                           avg=avg, output=output, tma_plan=tma_plan)
        # identity experts: hand back the rows the dispatch delivered
        output.copy_(calls["s1"]["x_fp8"].to(torch.bfloat16))
        return output

    def refuse(*a, **kw):
        raise AssertionError(
            "the DeepEP path must not run the NCCL-era MoE kernels")

    monkeypatch.setattr(glm5_moe_graph, "gate_sigmoid_topk_warp_cuda", fake_gate)
    # the DeepEP path routes with the one-warp-per-row gate only
    monkeypatch.setattr(glm5_moe_graph, "gate_sigmoid_topk_cuda", refuse)
    monkeypatch.setattr(
        glm5_moe_graph, "grouped_fp8_blockwise_fused_s1", fake_s1)
    monkeypatch.setattr(glm5_moe_graph, "grouped_fp8_blockwise_s3", fake_s3)
    monkeypatch.setattr(glm5_moe_graph, "act_quant_ragged", fake_quant)
    monkeypatch.setattr(
        glm5_moe_graph, "prepare_grouped_fp8_s1_s3_tma", fake_prepare)
    monkeypatch.setattr(glm5_moe_graph, "deepep_ll_scale_repack", fake_repack)
    monkeypatch.setattr(glm5_moe_graph, "dispatch_scatter_ragged", refuse)
    monkeypatch.setattr(glm5_moe_graph, "reduce_weighted_scatter", refuse)

    calls["router"] = router_calls
    calls["order"] = order
    calls["chain"] = chain
    calls["overlap"] = overlap
    rank_counts = torch.full((WORLD,), bucket, dtype=torch.int64)
    rank_counts[rank] = valid
    return segment, pool, bufs, calls, rank_counts, valid


def test_forward_runs_the_exchange_and_skips_the_nccl_pair(monkeypatch):
    segment, pool, bufs, calls, rank_counts, valid = _build_segment(monkeypatch)
    padded = (torch.arange(BUCKET, dtype=torch.float32).unsqueeze(1)
              + torch.zeros(1, H)).to(torch.bfloat16)
    out = segment.forward(padded=padded, rank_token_counts=rank_counts)

    assert out["moe_output"].shape == (BUCKET, H)
    assert pool.deepep.dispatch_calls == 1 and pool.deepep.combine_calls == 1
    # the gate's int64 ids reach the exchange as written, padding rows unrouted
    assert bufs.ll_topk_idx.dtype == torch.int64
    assert torch.equal(bufs.ll_topk_idx[:valid, 0],
                       torch.full((valid,), segment.expert_start,
                                  dtype=torch.int64))
    assert torch.all(bufs.ll_topk_idx[valid:] == -1)

    # both GEMMs are driven by DeepEP's recv_count on the uniform table
    for stage in ("s1", "s3", "quant"):
        assert calls[stage]["cu_seqlens"] is bufs.ll_cu_seqlens
        assert calls[stage]["seqlens"].tolist() == [valid, valid]
    assert calls["s1"]["x_fp8"].shape == (ROWS, H)
    assert calls["s1"]["x_scale"] is bufs.ll_x_scale
    assert calls["s1"]["output"] is bufs.intermediate
    assert calls["s3"]["output"] is bufs.expert_out


def test_one_prepare_launch_serves_both_grouped_gemms(monkeypatch):
    """U1: the S1 and S3 descriptor/tile preparation is ONE launch per layer,
    issued before either GEMM, for exactly the buffers, counts, segment table
    and TileM hint both GEMMs then run with; both consume the same plan."""
    segment, pool, bufs, calls, rank_counts, valid = _build_segment(monkeypatch)
    padded = torch.ones(BUCKET, H, dtype=torch.bfloat16)
    segment.forward(padded=padded, rank_token_counts=rank_counts)

    assert calls["chain"] == ["prepare", "repack", "s1", "quant", "s3"]
    prep = calls["prepare"]
    plan = prep["plan"]
    assert calls["s1"]["tma_plan"] is plan and calls["s3"]["tma_plan"] is plan
    # the plan describes the very tensors each GEMM is handed
    assert prep["s1_x"].data_ptr() == calls["s1"]["x_fp8"].data_ptr()
    assert prep["s1_output"] is calls["s1"]["output"] is bufs.intermediate
    assert prep["s3_x"].data_ptr() == calls["s3"]["x_fp8"].data_ptr()
    assert prep["s3_x"].data_ptr() == bufs.inter_fp8.data_ptr()
    assert prep["s3_output"] is calls["s3"]["output"] is bufs.expert_out
    for stage in ("s1", "s3"):
        assert calls[stage]["seqlens"] is prep["seqlens"]
        assert calls[stage]["cu_seqlens"] is prep["cu_seqlens"]
        assert calls[stage]["avg"] == prep["avg"]
    # into the pool's one descriptor/tile set, never a per-call allocation
    assert plan.s1_tma_desc is bufs.ll_s1_tma and plan.s3_tma_desc is bufs.ll_s3_tma
    assert plan.tiles is bufs.ll_tiles and plan.cu_tiles is bufs.ll_cu_tiles


def test_the_descriptor_buffers_are_pool_owned_and_layer_shared(monkeypatch):
    pool = _make_pool(monkeypatch, buckets=(4, 8, 16))
    pool.setup()
    for name, shape, dtype in (
            ("ll_s1_tma", (2 * E_LOCAL, 128), torch.uint8),
            ("ll_s3_tma", (2 * E_LOCAL, 128), torch.uint8),
            ("ll_tiles", (E_LOCAL,), torch.int32),
            ("ll_cu_tiles", (E_LOCAL + 1,), torch.int32)):
        base = pool._base[name]
        assert tuple(base.shape) == shape and base.dtype == dtype, name
        for bucket in (4, 8, 16):
            assert getattr(pool.get(bucket), name) is base, (name, bucket)
    nccl = _make_pool(monkeypatch, configure=False)
    nccl.setup()
    for name in ("ll_s1_tma", "ll_s3_tma", "ll_tiles", "ll_cu_tiles"):
        assert nccl._base[name].shape[0] == 0, name


# --------------------------------------------------------------------------
# Router kernel dispatch
# --------------------------------------------------------------------------
@pytest.mark.parametrize("local_rows", [64, 128])
def test_the_deepep_router_takes_the_split_k_kernel_on_local_rows(
        monkeypatch, local_rows):
    """The gate that the all-gathered path already had, resolved on the row
    count THIS path routes.

    Dropping the all_gather divides the router's rows by the world size, so a
    bucket whose all-gathered space would have taken the split-K router has to
    still take it on one rank's rows. Without the gate every decode bucket fell
    through to the WGMMA router, whose tile shape is the wrong one at a single
    rank's row count — the one regression this path carried against the NCCL
    path it replaces.
    """
    segment, pool, bufs, calls, rank_counts, valid = _build_segment(
        monkeypatch, bucket=local_rows, split_k_rows=(local_rows,))
    padded = torch.ones(local_rows, H, dtype=torch.bfloat16)
    segment.forward(padded=padded, rank_token_counts=rank_counts)

    # One router call, the split-K kernel, on the LOCAL rows -- not
    # world_size * local_rows, which is what the all-gathered path routes.
    assert calls["router"]["split_k"] == [(local_rows, H)]
    assert calls["router"]["wgmma"] == []


def test_the_deepep_router_keeps_the_wgmma_kernel_outside_the_range(
        monkeypatch):
    """Row counts the split-K policy does not serve keep the WGMMA router, and
    the dispatch stays a single call either way."""
    segment, pool, bufs, calls, rank_counts, valid = _build_segment(
        monkeypatch, split_k_rows=())
    padded = torch.ones(BUCKET, H, dtype=torch.bfloat16)
    segment.forward(padded=padded, rank_token_counts=rank_counts)

    assert calls["router"]["wgmma"] == [(BUCKET, H)]
    assert calls["router"]["split_k"] == []


def test_the_router_dispatch_reads_the_kernel_policy_bounds(monkeypatch):
    """The window is the kernel policy's own constants, not literals repeated
    in the model layer: move the bounds and the dispatch moves with them.

    This exercises the real seam, with the kernel package stood in for because
    importing it needs Triton. Outside the range it must not even reach the
    kernel import, which is what returning before it asserts.
    """
    sentinel = object()
    monkeypatch.setitem(
        sys.modules,
        "batchgen_kernels.triton.glm5_router_gemm_config",
        SimpleNamespace(ROUTER_SPLIT_K_MIN_ROWS=100,
                        ROUTER_SPLIT_K_MAX_ROWS=200),
    )
    monkeypatch.setitem(
        sys.modules,
        "batchgen_kernels.triton.glm5_router_gemm",
        SimpleNamespace(glm5_router_gemm=sentinel),
    )

    select = glm5_moe_graph._graphed_router_split_k_kernel
    assert select(99) is None
    assert select(100) is sentinel
    assert select(137) is sentinel
    assert select(200) is sentinel
    assert select(201) is None


def test_dispatch_scale_repack_is_elementwise_exact(monkeypatch):
    """R1: per expert, exactly the columns of the GEMM's ceil(count / TileM)
    row tiles are repacked from DeepEP's block, element for element; every
    column past them keeps what the buffer held (here a sentinel)."""
    segment, pool, bufs, calls, rank_counts, valid = _build_segment(monkeypatch)
    sentinel = -7.5
    bufs.ll_x_scale.fill_(sentinel)
    padded = torch.ones(BUCKET, H, dtype=torch.bfloat16)
    segment.forward(padded=padded, rank_token_counts=rank_counts)

    rep = calls["repack"]
    plan = calls["prepare"]["plan"]
    # the exchange's own view, its counts, the GEMM's buffer and TileM
    assert rep["out"] is bufs.ll_x_scale is calls["s1"]["x_scale"]
    assert rep["recv_count"] is calls["s1"]["seqlens"]
    assert rep["recv_scales"].stride() == (STRIDE * (H // 128), 1, STRIDE)
    assert rep["span_align"] == plan.tile_m == grouped_fp8_tile_m(
        calls["s1"]["avg"])
    span = -(-valid // plan.tile_m) * plan.tile_m
    assert valid < span < STRIDE, "the geometry must leave columns uncopied"

    # DeepEP's contiguous allocation is [e, h/128, stride]; the GEMM needs one
    # [h/128, e * stride] tensor with expert e's columns at e * stride.
    base = pool.deepep.scales_base
    expected = torch.full((H // 128, ROWS), sentinel, dtype=torch.float32)
    for e in range(E_LOCAL):
        expected[:, e * STRIDE:e * STRIDE + span] = base[e][:, :span]
    assert torch.equal(bufs.ll_x_scale, expected)
    # spot-check the encoding so a transposed repack cannot pass
    for e in range(E_LOCAL):
        for pos in range(valid):
            for blk in range(H // 128):
                assert bufs.ll_x_scale[blk, e * STRIDE + pos].item() == (
                    1000 * (e + 1) + 10 * pos + blk)


def test_combine_applies_topk_weights_on_the_owning_rank(monkeypatch):
    segment, pool, bufs, calls, rank_counts, valid = _build_segment(monkeypatch)
    padded = (torch.arange(1, BUCKET + 1, dtype=torch.float32).unsqueeze(1)
              .expand(BUCKET, H).contiguous()).to(torch.bfloat16)
    out = segment.forward(padded=padded, rank_token_counts=rank_counts)
    # identity experts: row t gets (w0 + w1) * x[t] from this rank's two
    # experts; the two remote ids contribute nothing in a single-process stand-in
    expected = (padded.float() * (0.25 + 0.5)).to(torch.bfloat16)
    expected[valid:] = 0
    assert torch.equal(out["moe_output"], expected)
    assert out["moe_output"] is bufs.local_moe_output


def test_padded_rows_stay_inert(monkeypatch):
    """M-independence: rows at or beyond recv_count[e] hold whatever the last
    step left behind. Neither GEMM writes them (both are seqlens-driven) and
    the combine reads only the handle's live rows, so the output must not move
    when they are poisoned."""
    segment, pool, bufs, calls, rank_counts, valid = _build_segment(monkeypatch)
    padded = torch.ones(BUCKET, H, dtype=torch.bfloat16)
    clean = segment.forward(
        padded=padded, rank_token_counts=rank_counts)["moe_output"].clone()

    # poison every row the live counts do not cover, in both GEMM outputs
    for e in range(E_LOCAL):
        lo = e * STRIDE + valid
        hi = (e + 1) * STRIDE
        bufs.expert_out[lo:hi] = -999.0
        bufs.intermediate[lo:hi] = -999.0
        bufs.ll_x_scale[:, lo:hi] = float("nan")
    # ...and the shared-expert staging buffer, which the overlap added. It is a
    # process-lifetime buffer now rather than a tensor minted per step, so it
    # must carry no state between steps: the forked shared expert rewrites
    # every row of the bucket slice before the join.
    bufs.shared_out.fill_(float("nan"))
    pool._base["shared_out"].fill_(float("nan"))
    poisoned = segment.forward(
        padded=padded, rank_token_counts=rank_counts)["moe_output"]
    assert torch.equal(poisoned, clean)


# --------------------------------------------------------------------------
# Shared-expert side-stream overlap
# --------------------------------------------------------------------------
def test_the_shared_expert_forks_before_the_exchange_and_joins_after_combine(
        monkeypatch):
    """The whole point of the slice: the shared expert is issued BEFORE the
    dispatch and collected AFTER the combine, so the side stream has the entire
    routed chain to run against rather than being appended to it."""
    segment, pool, bufs, calls, rank_counts, valid = _build_segment(monkeypatch)
    padded = torch.ones(BUCKET, H, dtype=torch.bfloat16)
    segment.forward(padded=padded, rank_token_counts=rank_counts)

    assert calls["order"] == [
        "fork_enter", "shared_expert", "fork_exit",
        "dispatch", "combine", "join",
    ]
    # exactly one fork/join pair per layer per step -- not per bucket, not per
    # expert, and never a fork left unjoined
    assert calls["overlap"].forks == 1 and calls["overlap"].joins == 1


def test_the_accumulate_reads_the_joined_staging_buffer(monkeypatch):
    """Join-before-accumulate, as a value assertion.

    The stand-in withholds the staged rows until ``join`` runs, so an
    accumulate issued before the join would add the sentinel instead of the
    shared-expert output. Comparing against the same forward with a zero shared
    expert isolates exactly the staged contribution.
    """
    segment, pool, bufs, calls, rank_counts, valid = _build_segment(
        monkeypatch, shared_value=0.0)
    padded = torch.ones(BUCKET, H, dtype=torch.bfloat16)
    routed = segment.forward(
        padded=padded, rank_token_counts=rank_counts)["moe_output"].clone()

    segment, pool, bufs, calls, rank_counts, valid = _build_segment(
        monkeypatch, shared_value=0.5)
    out = segment.forward(padded=padded, rank_token_counts=rank_counts)
    assert torch.equal(out["moe_output"], routed + 0.5)
    assert not torch.any(out["moe_output"] == _RecordingOverlap.WITHHELD)
    # the accumulate consumes the POOL buffer, not a per-step temporary
    assert out["moe_output"] is bufs.local_moe_output


def test_the_staging_buffer_is_pool_owned_and_bucket_sliced(monkeypatch):
    pool = _make_pool(monkeypatch, buckets=(4, 8, 16))
    pool.setup()
    base = pool._base["shared_out"]
    assert base.shape == (16, H) and base.dtype == torch.bfloat16
    for bucket in (4, 8, 16):
        bufs = pool.get(bucket)
        # the accumulate's operand must match local_moe_output exactly
        assert bufs.shared_out.shape == bufs.local_moe_output.shape
        assert bufs.shared_out.dtype == bufs.local_moe_output.dtype
        # one allocation for every bucket AND every MoE layer: the layers are
        # sequential and each joins before the next forks
        assert bufs.shared_out.data_ptr() == base.data_ptr()


def test_the_nccl_pool_carries_no_staging_buffer(monkeypatch):
    """The side-stream fork is fenced to the DeepEP path, so the default path
    must not pay for a buffer it never reads."""
    pool = _make_pool(monkeypatch, configure=False)
    pool.setup()
    assert pool.get(BUCKET).shared_out.numel() == 0
    assert pool._base["shared_out"].numel() == 0


def test_the_nccl_forward_keeps_the_shared_expert_in_line():
    """The NCCL path is the default and the merged baseline; a side stream
    there would be a second unvalidated variable in the same measurement."""
    nccl = inspect.getsource(Glm5MoEGraphSegment.forward)
    deepep = inspect.getsource(Glm5MoEGraphSegment._forward_deepep)
    assert "_shared_overlap" not in nccl
    assert (
        "bufs.local_moe_output.add_(self.moe.shared_expert_forward(padded))"
        in nccl
    )
    assert "_shared_overlap.fork()" in deepep
    assert "_shared_overlap.join()" in deepep


def test_a_shared_output_outside_the_staging_buffer_fails_loud(monkeypatch):
    """The shared expert writes its output INTO the staging buffer (no staging
    copy), so any other returned tensor -- wrong dtype, wrong rows, or even a
    same-shaped stranger -- means the accumulate would read a buffer the side
    stream never wrote. That must raise, not be copied or ignored."""
    padded = torch.ones(BUCKET, H, dtype=torch.bfloat16)
    for kw in (dict(shared_dtype=torch.float32), dict(shared_rows=BUCKET - 1),
               dict(shared_foreign=True)):
        segment, pool, bufs, calls, rank_counts, valid = _build_segment(
            monkeypatch, **kw)
        with pytest.raises(RuntimeError, match="shared expert returned"):
            segment.forward(padded=padded, rank_token_counts=rank_counts)


def test_the_shared_expert_writes_straight_into_the_staging_buffer():
    """The staging copy is gone from the side stream: the segment passes the
    pool buffer as ``out=`` and checks identity, and no ``copy_`` into it is
    left in the DeepEP forward."""
    deepep = inspect.getsource(Glm5MoEGraphSegment._forward_deepep)
    assert "out=bufs.shared_out)" in deepep
    assert "shared_out is not bufs.shared_out" in deepep
    assert "bufs.shared_out.copy_(" not in deepep


def test_the_gate_writes_the_exchange_ids_without_a_copy():
    """The int32 -> int64 copy kernel is gone: the DeepEP forward hands the
    one-warp-per-row gate the exchange's int64 buffer and never copies into
    it."""
    deepep = inspect.getsource(Glm5MoEGraphSegment._forward_deepep)
    assert "gate_sigmoid_topk_warp_cuda(" in deepep
    assert "topk_indices=bufs.ll_topk_idx" in deepep
    assert "ll_topk_idx.copy_(" not in deepep
    assert "gate_sigmoid_topk_cuda(" not in deepep


# --- the stream/event resources themselves -------------------------------
class _FakeStream:
    def __init__(self, device):
        self.device = device


def _fake_cuda(monkeypatch, *, capturing=False, current_device=0):
    monkeypatch.setattr(glm5_moe_graph, "_SIDE_STREAMS", {})
    monkeypatch.setattr(torch.cuda, "Stream", _FakeStream)
    monkeypatch.setattr(torch.cuda, "Event", lambda *a, **kw: object())
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current_device)


def test_the_side_stream_is_one_process_resource_per_device(monkeypatch):
    """Created once and shared by every MoE layer. The graph pool is dropped
    and rebuilt around a mid-run prefill wave, so a stream owned by the pool
    would leak one handle per rebuild."""
    _fake_cuda(monkeypatch)
    first = glm5_moe_graph._glm5_moe_side_stream(torch.device("cuda", 0))
    assert glm5_moe_graph._glm5_moe_side_stream(torch.device("cuda", 0)) is first
    # a device-less "cuda" resolves through the current device, same stream
    assert glm5_moe_graph._glm5_moe_side_stream(torch.device("cuda")) is first
    # a second device gets its own
    other = glm5_moe_graph._glm5_moe_side_stream(torch.device("cuda", 1))
    assert other is not first


def test_the_side_stream_refuses_a_non_cuda_device(monkeypatch):
    _fake_cuda(monkeypatch)
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        glm5_moe_graph._glm5_moe_side_stream(torch.device("cpu"))


def test_the_overlap_resources_are_never_minted_inside_a_capture(monkeypatch):
    """A stream or event created during ``torch.cuda.graph`` is a capture-time
    resource the replay cannot reproduce. The fork/join CALLS belong inside the
    capture -- they are the dependency edges -- but the handles must predate
    it, and the graph manager builds segments before warmup and capture."""
    _fake_cuda(monkeypatch, capturing=True)
    with pytest.raises(RuntimeError, match="must exist before CUDA"):
        glm5_moe_graph._glm5_moe_side_stream(torch.device("cuda", 0))

    # with the stream already cached, the per-layer event is the next thing
    # that must not be minted under capture
    _fake_cuda(monkeypatch, capturing=False)
    glm5_moe_graph._glm5_moe_side_stream(torch.device("cuda", 0))
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="must be created before CUDA"):
        glm5_moe_graph._SharedExpertOverlap(torch.device("cuda", 0), 5)


def test_the_overlap_is_built_once_at_segment_construction(monkeypatch):
    """Per forward there is no construction at all: the handles are attached to
    the segment, so a replay issues only the two dependency edges."""
    src = inspect.getsource(Glm5MoEGraphSegment._forward_deepep)
    assert "_SharedExpertOverlap(" not in src
    assert "torch.cuda.Stream(" not in src
    assert "torch.cuda.Event(" not in src
    ctor = inspect.getsource(Glm5MoEGraphSegment.__init__)
    assert "_SharedExpertOverlap(" in ctor


def _validation_segment(monkeypatch, *, rank=1, world=WORLD,
                        expert_start=None, offloading=False):
    pool = _make_pool(monkeypatch, buckets=(BUCKET,), world=world)
    segment = Glm5MoEGraphSegment.__new__(Glm5MoEGraphSegment)
    segment.pool = pool
    segment.hidden_size = 6144
    segment.num_experts_per_tok = 8
    segment.num_local_experts = E_LOCAL
    segment.world_size = world
    segment.rank = rank
    segment.expert_start = (rank * E_LOCAL if expert_start is None
                            else expert_start)
    segment.moe = SimpleNamespace(layer_idx=5,
                                 enable_ep_offloading=offloading)
    return segment


def test_segment_gate_fails_loud_without_a_usable_build(monkeypatch):
    """--enable-deepep is explicit: an unusable build is an error at capture,
    never a silent fall back to the NCCL pair."""
    segment = _validation_segment(monkeypatch)
    monkeypatch.setattr(deepep_ll, "deepep_unavailable_reason",
                        lambda **kw: "deep_ep is not importable: test")
    with pytest.raises(RuntimeError, match="not importable"):
        segment._validate_deepep_graph_safe()


def test_segment_gate_passes_the_glm_shape_to_the_availability_check(monkeypatch):
    segment = _validation_segment(monkeypatch)
    seen = {}

    def spy(**kw):
        seen.update(kw)
        return None

    monkeypatch.setattr(
        deepep_ll, "deepep_unavailable_reason", spy)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: WORLD)
    segment._validate_deepep_graph_safe()
    assert seen == dict(hidden=6144, topk=8,
                        num_experts=WORLD * E_LOCAL, num_ranks=WORLD,
                        max_tokens_per_rank=BUCKET, use_fp8=True)


def test_segment_gate_requires_the_rank_aligned_expert_window(monkeypatch):
    """DeepEP returns recv_count for experts [rank*E_local, ...). A different
    GLM expert window would GEMM every routed row against the wrong weights."""
    segment = _validation_segment(monkeypatch, rank=1, expert_start=0)
    monkeypatch.setattr(
        deepep_ll, "deepep_unavailable_reason", lambda **kw: None)
    with pytest.raises(RuntimeError, match="rank \\* experts_per_rank"):
        segment._validate_deepep_graph_safe()


@pytest.mark.parametrize("field,value,match", [
    ("num_local_experts", 65, "at most 256 routed experts"),
    ("num_experts_per_tok", 6, "top-k 2, 4, 8 or 16"),
])
def test_segment_gate_fences_shapes_the_warp_gate_cannot_route(
        monkeypatch, field, value, match):
    """The DeepEP gate holds a row's experts in one warp's registers; a model
    outside that shape fails at validation, not inside a capture."""
    segment = _validation_segment(monkeypatch)
    setattr(segment, field, value)
    segment.expert_start = segment.rank * segment.num_local_experts
    monkeypatch.setattr(
        deepep_ll, "deepep_unavailable_reason", lambda **kw: None)
    with pytest.raises(RuntimeError, match=match):
        segment._validate_deepep_graph_safe()


def test_segment_gate_fences_expert_offloading(monkeypatch):
    segment = _validation_segment(monkeypatch, offloading=True)
    monkeypatch.setattr(
        deepep_ll, "deepep_unavailable_reason", lambda **kw: None)
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        segment._validate_deepep_graph_safe()


def test_segment_gate_requires_dist_world_equal_to_ep_world(monkeypatch):
    segment = _validation_segment(monkeypatch)
    monkeypatch.setattr(
        deepep_ll, "deepep_unavailable_reason", lambda **kw: None)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="initialized torch.distributed"):
        segment._validate_deepep_graph_safe()
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: WORLD + 1)
    with pytest.raises(RuntimeError, match="EP world"):
        segment._validate_deepep_graph_safe()


def test_bucket_switch_reuses_one_receive_space(monkeypatch):
    """Every bucket replays against the same expert-major rows, so no bucket
    switch needs a DeepEP re-clean (one clean at construction)."""
    segment, pool, bufs, calls, rank_counts, valid = _build_segment(
        monkeypatch, bucket=BUCKET)
    small = pool.get(BUCKET)
    assert small.expert_out.data_ptr() == pool._base["expert_out"].data_ptr()
    assert small.ll_x_scale.data_ptr() == pool._base["ll_x_scale"].data_ptr()
    assert pool.deepep.max_tokens_per_rank == BUCKET


def test_the_shared_expert_forks_between_the_router_gemm_and_the_top_k(monkeypatch):
    """Placement: the fork is issued after the router GEMM and before the
    top-k, the earliest point its input (``padded``) allows past the router,
    and nothing joins it before the combine."""
    segment, pool, bufs, calls, rank_counts, valid = _build_segment(monkeypatch)
    order = calls["order"]
    real_router = segment.router_context.router_forward
    real_gate = glm5_moe_graph.gate_sigmoid_topk_warp_cuda
    real_s1 = glm5_moe_graph.grouped_fp8_blockwise_fused_s1

    def traced_router(*a, **kw):
        order.append("router_gemm")
        return real_router(*a, **kw)

    def traced_gate(*a, **kw):
        order.append("topk")
        return real_gate(*a, **kw)

    def traced_s1(*a, **kw):
        order.append("s1")
        return real_s1(*a, **kw)

    segment.router_context.router_forward = traced_router
    monkeypatch.setattr(glm5_moe_graph, "gate_sigmoid_topk_warp_cuda", traced_gate)
    monkeypatch.setattr(
        glm5_moe_graph, "grouped_fp8_blockwise_fused_s1", traced_s1)
    padded = torch.ones(BUCKET, H, dtype=torch.bfloat16)
    segment.forward(padded=padded, rank_token_counts=rank_counts)
    assert order == [
        "router_gemm", "fork_enter", "shared_expert", "fork_exit", "topk",
        "dispatch", "s1", "combine", "join",
    ]


def test_dispatch_and_combine_run_split_into_send_and_receive(monkeypatch):
    """Both exchanges ask for the send / receive split, once per layer."""
    segment, pool, bufs, calls, rank_counts, valid = _build_segment(monkeypatch)
    padded = torch.ones(BUCKET, H, dtype=torch.bfloat16)
    segment.forward(padded=padded, rank_token_counts=rank_counts)
    assert pool.deepep.dispatch_split == [True]
    assert pool.deepep.combine_split == [True]

