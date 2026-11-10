"""K3 DP-attention rank size G and the streamed-prefill span S.

Attention, KDA and the shared expert run TP-G inside one DP rank; the routed
experts run EP over every GPU of the job. Streamed prefill holds one copy of a
layer's experts per span of S ranks (S a multiple of G). These tests drive the
REAL ``KimiLinearParallelStrategyManager`` and ``StreamedSP8LayerBuffer`` on
CPU with ``torch.distributed`` replaced by recorders, for the two shapes the
profile rows plan:

* G = S = 8 on world 16 (the validated 8-GPU-host layout): byte-identical
  group creation and expert ranges;
* G = 4, S = 8 on world 8 (two GB300 trays, or TP4 on one B300): one span
  over both DP ranks, global pass alignment, and the server prefill-mode
  option for resident EP.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from batchgen.models.moonshotai.kimi_linear.Parallel_Strategy_Manager import (
    KimiLinearParallelStrategyManager,
)
from batchgen.moe import streamed_sp8_mxfp4 as sp8


def _loaded_model_config():
    cfg = SimpleNamespace(
        model_type="kimi_k3",
        n_routed_experts=896,
        kda_num_heads=32,
        kda_head_dim=64,
        kda_conv_size=4,
        num_hidden_layers=4,
        hidden_size=16,
    )
    cfg.is_kda_layer = lambda layer_idx: layer_idx % 2 == 0
    return cfg


def _psm(*, world, group, span, rank=0, latent=True, mode=None, **basic):
    fields = dict(
        distributed_weight_sharded=False,
        distributed_weight_transport="host_rdma",
        attention_group_size=group,
        stream_all_modules=False,
        device_torch="cpu",
        prefill_moe_mode=mode,
    )
    fields.update(basic)
    engine = SimpleNamespace(
        Basic_Config=SimpleNamespace(**fields),
        GPU_Buffer_Config=SimpleNamespace(
            kda_state_slots=8, kda_prefix_segment_slots=0
        ),
        Module_Batching_Config=SimpleNamespace(
            k3_shard_latent_projections=latent,
            k3_prefill_stream_span=span,
        ),
    )
    return KimiLinearParallelStrategyManager(
        loaded_model_config=_loaded_model_config(),
        engine_config=engine,
        model_config=SimpleNamespace(),
        core_engine=SimpleNamespace(),
        skeleton_state_dict={},
        rank=rank % group,
        global_rank=rank,
        world_size=world,
    )


@pytest.fixture
def groups(monkeypatch):
    created = []

    def new_group(ranks):
        created.append(tuple(ranks))
        return ("group", tuple(ranks))

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "new_group", new_group)
    return created


# --------------------------------------------------------------------------- #
#  Groups and expert ownership                                                 #
# --------------------------------------------------------------------------- #


def test_g8_span8_creates_exactly_the_validated_groups(groups):
    psm = _psm(world=16, group=8, span=8, rank=9)
    psm._build_attn_tp_group()
    # attention family, then the node-local weight family: no span family,
    # because the span IS the DP rank and reuses the weight group.
    assert groups == [
        tuple(range(0, 8)), tuple(range(8, 16)),
        tuple(range(0, 8)), tuple(range(8, 16)),
    ]
    assert psm._stream_span_group is psm._weight_tp_group
    assert psm._stream_span_rank == psm._attn_tp_rank == 1


def test_g4_span8_adds_one_span_group_over_both_dp_ranks(groups):
    psm = _psm(world=8, group=4, span=8, rank=6)
    psm._build_attn_tp_group()
    assert groups == [
        (0, 1, 2, 3), (4, 5, 6, 7),
        (0, 1, 2, 3), (4, 5, 6, 7),
        tuple(range(8)),
    ]
    assert psm._attn_tp_group == ("group", (4, 5, 6, 7))
    assert psm._stream_span_group == ("group", tuple(range(8)))
    # rank 6 streams the 112 experts it owns as an EP8 rank
    assert psm._stream_span_rank == 6
    assert psm._attn_tp_rank == 2
    assert psm.experts_per_rank == 112 and psm.local_expert_start == 672


@pytest.mark.parametrize(
    "world, group, span", [(8, 4, 6), (8, 4, 16), (8, 4, 2), (12, 4, 12)],
)
def test_a_span_that_is_not_whole_dp_ranks_dividing_the_world_is_refused(
    world, group, span,
):
    with pytest.raises((ValueError, AssertionError)):
        _psm(world=world, group=group, span=span)


# --------------------------------------------------------------------------- #
#  Prefill mode: streaming default, server option, alignment                   #
# --------------------------------------------------------------------------- #


def test_streaming_is_the_default_and_aligns_passes_globally_when_s_exceeds_g():
    gb300 = _psm(world=8, group=4, span=8)
    assert gb300.default_prefill_moe_mode() == "streamed_sp8"
    assert gb300.prefill_uses_streamed_sp8()
    assert gb300.streamed_sp8_requires_global_pass_alignment()
    node_local = _psm(world=16, group=8, span=8)
    assert node_local.prefill_uses_streamed_sp8()
    # unchanged: only hierarchical GDR aligns the node-local layout
    assert not node_local.streamed_sp8_requires_global_pass_alignment()


def test_server_option_resident_ep_is_the_mode_of_every_wave():
    psm = _psm(world=8, group=4, span=8, mode="resident_ep")
    assert psm.default_prefill_moe_mode() == "resident_ep"
    assert psm.prefill_uses_resident_ep()
    assert not psm.streamed_sp8_requires_global_pass_alignment()


def test_server_option_resident_ep_needs_dp_attention():
    with pytest.raises(ValueError, match="resident_ep needs"):
        _psm(world=8, group=1, span=0, latent=False, mode="resident_ep")


def test_server_option_streamed_with_sharded_latent_is_refused_at_construction():
    with pytest.raises(ValueError, match="unsharded LatentMoE"):
        _psm(world=8, group=4, span=8, mode="streamed")


def test_distributed_weights_accept_resident_ep_from_startup():
    psm = _psm(
        world=8, group=4, span=8, mode="resident_ep",
        distributed_weight_sharded=True, distributed_weight_transport="host_rdma",
    )
    assert psm.prefill_uses_resident_ep()
    with pytest.raises(ValueError, match="streamed_sp8' or 'resident_ep"):
        psm.set_prefill_moe_mode("streamed")


# --------------------------------------------------------------------------- #
#  Transport-only passes join every MoE layer's span collectives               #
# --------------------------------------------------------------------------- #


class _RecordingStreamedMoE:
    def __init__(self, calls, layer_idx):
        self.calls = calls
        self.layer_idx = layer_idx

    def forward(self, x, gate, num_rows):
        self.calls.append(("forward", self.layer_idx, tuple(x.shape), num_rows, gate))
        return x


class _RecordingBuffer:
    def __init__(self, calls):
        self.calls = calls

    def has_pending(self):
        return False

    def allow_cross_launch(self):
        self.calls.append(("allow_cross_launch",))


def test_transport_only_pass_runs_each_moe_layer_with_zero_rows():
    psm = _psm(world=8, group=4, span=8)
    calls = []
    layers = []
    for layer_idx in range(3):
        if layer_idx == 0:
            layers.append(SimpleNamespace(block_sparse_moe=None))
            continue
        moe = SimpleNamespace(
            gate=f"gate{layer_idx}",
            _streamed_sp8_moe=_RecordingStreamedMoE(calls, layer_idx),
        )
        layers.append(SimpleNamespace(block_sparse_moe=moe))
    psm.model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    psm._streamed_sp8_buffer = _RecordingBuffer(calls)
    psm.run_streamed_sp8_transport_only_prefill(2)
    one_pass = [
        ("forward", 1, (0, 16), 0, "gate1"), ("allow_cross_launch",),
        ("forward", 2, (0, 16), 0, "gate2"), ("allow_cross_launch",),
    ]
    assert calls == one_pass * 2


def test_transport_only_pass_is_refused_when_passes_stay_inside_the_dp_rank():
    psm = _psm(world=16, group=8, span=8)
    psm.model = SimpleNamespace(model=SimpleNamespace(layers=[]))
    psm._streamed_sp8_buffer = _RecordingBuffer([])
    with pytest.raises(RuntimeError, match="beyond the DP rank"):
        psm.run_streamed_sp8_transport_only_prefill(1)


# --------------------------------------------------------------------------- #
#  Streamed buffer: expert range and the span row stride                       #
# --------------------------------------------------------------------------- #


@pytest.fixture
def cpu_buffer_env(monkeypatch):
    monkeypatch.setattr(torch.cuda, "Stream", lambda device=None: object())
    monkeypatch.setattr(torch.cuda, "Event", lambda *a, **k: object())


def _buffer(*, span_rank, span_size, tp_size, layers=(1, 2, 3)):
    return sp8.StreamedSP8LayerBuffer(
        core_engine=None,
        device="cpu",
        span_group=("group", "span"),
        span_rank=span_rank,
        span_size=span_size,
        tp_size=tp_size,
        num_experts=896,
        intermediate_size=64,
        latent_size=32,
        acquire_batch_size=8,
        layer_indices=list(layers),
    )


def test_buffer_streams_the_span_slot_experts(cpu_buffer_env):
    gb300 = _buffer(span_rank=6, span_size=8, tp_size=4)
    assert (gb300.expert_start, gb300.experts_per_rank) == (672, 112)
    node = _buffer(span_rank=3, span_size=8, tp_size=8)
    assert (node.expert_start, node.experts_per_rank) == (336, 112)


@pytest.mark.parametrize(
    "span_rank, span_size, tp_size", [(0, 6, 4), (0, 8, 1), (8, 8, 4), (0, 12, 4)],
)
def test_buffer_refuses_a_bad_span(cpu_buffer_env, span_rank, span_size, tp_size):
    with pytest.raises(ValueError, match="streamed SP8 requires"):
        _buffer(span_rank=span_rank, span_size=span_size, tp_size=tp_size)


def test_span_equal_to_tp_needs_no_collective(cpu_buffer_env, monkeypatch):
    monkeypatch.setattr(
        dist, "all_reduce",
        lambda *a, **k: pytest.fail("S == G must not issue a collective"),
    )
    buffer = _buffer(span_rank=0, span_size=8, tp_size=8)
    assert buffer.span_rows_per_rank(1, 17) == 3
    assert buffer.span_rows_per_rank(2, 0) == 0


def test_span_stride_is_the_span_maximum_taken_once_per_pass(
    cpu_buffer_env, monkeypatch,
):
    reduced = []
    other_dp_rank_stride = {"value": 5}

    def all_reduce(tensor, op=None, group=None):
        reduced.append((int(tensor.item()), op, group))
        tensor.fill_(max(int(tensor.item()), other_dp_rank_stride["value"]))

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    buffer = _buffer(span_rank=1, span_size=8, tp_size=4)
    # first MoE layer of a pass: 9 local rows -> 3 per rank, span max 5
    assert buffer.span_rows_per_rank(1, 9) == 5
    assert buffer.span_rows_per_rank(2, 9) == 5
    assert buffer.span_rows_per_rank(3, 9) == 5
    assert reduced == [(3, dist.ReduceOp.MAX, ("group", "span"))]
    # next pass, this DP rank has no rows; the other still has some
    other_dp_rank_stride["value"] = 2
    assert buffer.span_rows_per_rank(1, 0) == 2
    assert len(reduced) == 2
    # a whole span without rows takes the empty branch on every rank
    other_dp_rank_stride["value"] = 0
    assert buffer.span_rows_per_rank(1, 0) == 0


def test_a_pass_whose_rows_grow_between_layers_is_refused(
    cpu_buffer_env, monkeypatch,
):
    monkeypatch.setattr(dist, "all_reduce", lambda tensor, op=None, group=None: None)
    buffer = _buffer(span_rank=0, span_size=8, tp_size=4)
    assert buffer.span_rows_per_rank(1, 4) == 1
    with pytest.raises(RuntimeError, match="row count changed"):
        buffer.span_rows_per_rank(2, 40)
