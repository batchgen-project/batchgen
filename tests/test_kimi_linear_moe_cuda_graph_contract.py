"""CPU contracts for the K3 resident-MXFP4 CUDA-graph MoE seam.

These tests deliberately do not launch CUDA.  They pin the fixed geometry and
the rank-major <-> balanced-row mapping that the remote CUDA gate exercises.
"""

import contextlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import batchgen.models.moonshotai.kimi_linear.moe_cuda_graph_segments as k3_moe_graph

from batchgen.models.moonshotai.kimi_linear.cuda_graph_segments import (
    KimiLinearDecodeGraph,
)
from batchgen.models.moonshotai.kimi_linear.moe_cuda_graph_segments import (
    K3MoEGraphBufferPool,
    K3MoEGraphSegment,
    fuse_router_and_down_proj,
    fused_gate_kernel_eligible,
    k3_moe_graph_buckets,
    split_fused_front,
)
from batchgen.models.moonshotai.kimi_linear.moe_tp_reshard import (
    balanced_row_split,
)
from batchgen.models.moonshotai.kimi_linear.whole_model_cuda_graph_segments import (
    KimiLinearWholeModelSegment,
)


def _make_gate(
    *,
    num_experts=8,
    hidden=16,
    top_k=2,
    n_group=1,
    topk_group=1,
    scoring_func="sigmoid",
    norm_topk_prob=True,
):
    """A real ``KimiMoEGate`` on CPU; the model module pulls in fla/einops."""
    model = pytest.importorskip(
        "batchgen.models.moonshotai.kimi_linear.model",
        reason="KimiMoEGate needs the fla/einops model dependencies",
    )
    config = SimpleNamespace(
        hidden_size=hidden,
        num_experts_per_tok=top_k,
        n_routed_experts=num_experts,
        routed_scaling_factor=2.5,
        scoring_func=scoring_func,
        n_group=n_group,
        topk_group=topk_group,
        norm_topk_prob=norm_topk_prob,
    )
    gate = model.KimiMoEGate(config)
    with torch.no_grad():
        gate.weight.copy_(torch.randn(num_experts, hidden))
        gate.e_score_correction_bias.copy_(torch.randn(num_experts))
    return gate


def _make_fusable(*, num_experts=4, latent=6, hidden=8, gate_dtype=torch.bfloat16):
    gate = nn.Linear(hidden, num_experts, bias=False).to(gate_dtype)
    down_proj = nn.Linear(hidden, latent, bias=False).to(torch.bfloat16)
    return SimpleNamespace(gate=gate), down_proj


def test_k3_moe_graph_buckets_use_full_tp_group_geometry():
    assert k3_moe_graph_buckets([1, 2, 4, 8, 16, 24, 32], 8) == [8, 16, 24, 32]


def test_k3_graph_pool_has_one_local_reduce_scatter_output():
    pool = K3MoEGraphBufferPool(
        world_size=16,
        tp_size=8,
        num_local_experts=2,
        intermediate_size=256,
        latent_size=128,
        hidden_size=512,
        top_k=16,
        expert_buckets=[8, 16, 24, 32],
        device=torch.device("cpu"),
    )
    view = pool.get(24)

    assert view.local_tokens == 3
    assert view.global_tokens == 48
    assert view.max_tokens_padded == 48
    assert tuple(view.combined.shape) == (48, 128)
    assert view.combined.dtype == torch.float32
    assert tuple(view.local_combined.shape) == (3, 128)
    assert view.local_combined.dtype == torch.float32
    # The route-capacity bound is one route per token per expert at most; it is
    # global_tokens, not global_tokens * top_k, because top-k indices are unique.
    assert view.max_tokens_padded < view.global_tokens * 16
    # The routed rows are folded into the shared expert's own TP partial, so
    # the pool no longer carries a separate group-sized all-gather output.
    assert not hasattr(view, "tp_output")


def test_k3_graph_pool_preallocates_native_gate_kernel_outputs():
    pool = K3MoEGraphBufferPool(
        world_size=16,
        tp_size=8,
        num_local_experts=2,
        intermediate_size=256,
        latent_size=128,
        hidden_size=512,
        top_k=16,
        expert_buckets=[8, 16, 24, 32],
        device=torch.device("cpu"),
    )
    view = pool.get(24)

    # The gate kernel writes int32 indices and FP32 weights directly into these,
    # so the eligible graph path needs no cast and no padding-mask ``where``.
    assert tuple(view.local_indices.shape) == (3, 16)
    assert view.local_indices.dtype == torch.int32
    assert tuple(view.local_weights.shape) == (3, 16)
    assert view.local_weights.dtype == torch.float32
    # The same kernel casts the fused row's latent suffix into this buffer, so
    # the graph needs no separate strided FP32->BF16 copy either.
    assert tuple(view.local_latent.shape) == (3, 128)
    assert view.local_latent.dtype == torch.bfloat16
    # They must be the EP all-gather sources, hence contiguous rows.
    assert view.local_indices.is_contiguous()
    assert view.local_weights.is_contiguous()
    assert view.local_latent.is_contiguous()


def test_k3_graph_segment_inputs_keep_group_and_local_rows_separate():
    segment = K3MoEGraphSegment.__new__(K3MoEGraphSegment)
    segment.hidden_size = 512
    specs = segment.get_static_input_specs(24)

    assert specs["padded"].resolve_shape(24) == (24, 512)
    assert specs["local"].resolve_shape(24) == (24, 512)
    assert specs["num_valid_tokens"].resolve_shape(24) == (1,)


def test_k3_graph_combine_passes_preallocated_fp32_output(monkeypatch):
    segment = K3MoEGraphSegment.__new__(K3MoEGraphSegment)
    segment.top_k = 16
    segment.latent_size = 128
    output = torch.empty((3, 128), dtype=torch.float32)
    called = {}

    def fake_combine(expert_output, topk_pos, topk_weights, n, h, k, out):
        called["args"] = (expert_output, topk_pos, topk_weights, n, h, k, out)
        out.fill_(7.0)
        return out

    monkeypatch.setattr(k3_moe_graph, "reduce_weighted_scatter_fp32", fake_combine)
    bufs = SimpleNamespace(
        expert_output=torch.empty((48, 128), dtype=torch.bfloat16),
        topk_pos=torch.empty((3 * 16,), dtype=torch.int32),
        all_weights=torch.empty((3, 16), dtype=torch.float32),
        global_tokens=3,
        combined=output,
    )

    segment._combine_fp32(bufs)

    args = called["args"]
    assert args[3] == 3
    assert args[4] == 128
    assert args[5] == 16
    assert args[6] is output
    assert torch.equal(output, torch.full_like(output, 7.0))


# ---------------------------------------------------------------------------
#  Folded shared/routed TP reduction
# ---------------------------------------------------------------------------

_TP_GROUP = object()


class _FakeEPComm:
    """Records the EP collective order and fills the gather buffers in place."""

    def __init__(self):
        self.calls = []

    @contextlib.contextmanager
    def change_state(self, enable=True):
        yield

    def group_start(self):
        self.calls.append("group_start")

    def group_end(self):
        self.calls.append("group_end")

    def all_gather(self, out, src):
        self.calls.append("all_gather")
        out.zero_()
        out[: src.shape[0]].copy_(src)

    def reduce_scatter(self, out, src, op=None):
        self.calls.append("reduce_scatter")
        out.copy_(src[: out.shape[0]])


def _make_cpu_pool(*, hidden, tp_size=4, world_size=4, bucket=8, latent=128):
    return K3MoEGraphBufferPool(
        world_size=world_size,
        tp_size=tp_size,
        num_local_experts=2,
        intermediate_size=256,
        latent_size=latent,
        hidden_size=hidden,
        top_k=16,
        expert_buckets=[bucket],
        device=torch.device("cpu"),
    )


def _make_cpu_forward_segment(monkeypatch, *, tp_rank, shared_ffn, routed_value):
    """A K3 MoE segment whose kernels/collectives are CPU stand-ins.

    Only the graph's own dataflow is real: the routing seam, the buffer views,
    the rank-major fold, and the TP collective the fold replaces.
    """
    tp_size, hidden, latent, bucket = 4, 16, 128, 8
    pool = _make_cpu_pool(hidden=hidden, tp_size=tp_size, latent=latent,
                          bucket=bucket)
    bufs = pool.get(bucket)

    segment = K3MoEGraphSegment.__new__(K3MoEGraphSegment)
    segment.pool = pool
    segment.device = torch.device("cpu")
    segment.hidden_size = hidden
    segment.latent_size = latent
    segment.intermediate_size = 256
    segment.top_k = 16
    segment.num_experts = 8
    segment.world_size = tp_size
    segment.rank = tp_rank
    segment.expert_start = 0
    segment.num_local_experts = 2
    segment.tp_size = tp_size
    segment.tp_rank = tp_rank
    segment.tp_group = _TP_GROUP
    segment.shared_stream = None
    segment.fused_front = None
    segment.fused_gate_kernel = False

    gate_out = (
        torch.zeros(bufs.local_tokens, 16, dtype=torch.int64),
        torch.ones(bufs.local_tokens, 16, dtype=torch.float32),
    )
    segment.moe = SimpleNamespace(
        gate=lambda hidden_states: gate_out,
        shared_experts=SimpleNamespace(_ffn=shared_ffn),
    )
    segment.resident = SimpleNamespace(
        shard=SimpleNamespace(
            gate_B_ptrs=None, gate_scales_ptrs=None,
            up_B_ptrs=None, up_scales_ptrs=None,
            down_B_ptrs=None, down_scales_ptrs=None,
        ),
        comm=_FakeEPComm(),
        down_proj=lambda x: torch.zeros(
            x.shape[0], latent, dtype=torch.bfloat16
        ),
        norm=None,
        up_proj=lambda x: torch.full(
            (x.shape[0], hidden), routed_value, dtype=torch.bfloat16
        ),
    )

    monkeypatch.setattr(
        k3_moe_graph, "dispatch_scatter_3d", lambda *args: (args[6], args[8])
    )
    monkeypatch.setattr(
        k3_moe_graph, "marlin_grouped_stage1_fused_mxfp4_situ",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        k3_moe_graph, "marlin_grouped_m16_mxfp4", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        k3_moe_graph, "reduce_weighted_scatter_fp32",
        lambda expert_out, pos, weights, n, h, k, out: out.zero_(),
    )
    # The graph pins the NCCL device around its TP collective; on CPU that
    # context manager is the only piece that cannot run as written.
    monkeypatch.setattr(
        k3_moe_graph.torch.cuda, "device",
        lambda device: contextlib.nullcontext(),
    )

    def _refuse_all_gather(*args, **kwargs):
        raise AssertionError("the folded path must not all-gather routed rows")

    monkeypatch.setattr(
        k3_moe_graph.dist, "all_gather_into_tensor", _refuse_all_gather
    )
    reduces = []
    monkeypatch.setattr(
        k3_moe_graph.dist, "all_reduce",
        lambda tensor, group=None: reduces.append(
            (tensor, tensor.clone(), group)
        ),
    )
    return segment, bufs, reduces


@pytest.mark.parametrize("tp_rank", [0, 3])
def test_k3_graph_folds_routed_rows_into_one_tp_all_reduce(monkeypatch, tp_rank):
    seen = {}

    def shared_ffn(x):
        seen["padded"] = x
        return torch.zeros(x.shape[0], x.shape[1], dtype=torch.bfloat16)

    segment, bufs, reduces = _make_cpu_forward_segment(
        monkeypatch, tp_rank=tp_rank, shared_ffn=shared_ffn, routed_value=3.0
    )
    bucket = bufs.local_tokens * segment.tp_size
    padded = torch.zeros(bucket, segment.hidden_size, dtype=torch.bfloat16)

    moe_output = segment.forward(
        padded=padded,
        local=padded,
        num_valid_tokens=torch.tensor([bufs.local_tokens], dtype=torch.int32),
    )["moe_output"]

    # The unreduced FFN body runs over the whole replicated group batch, ...
    assert seen["padded"] is padded
    # ... exactly one TP collective carries both halves, ...
    assert len(reduces) == 1
    reduced, before_reduce, group = reduces[0]
    assert group is segment.tp_group
    assert reduced is moe_output
    # ... and this rank's routed rows land only in its rank-major slice.
    expected = torch.zeros_like(padded)
    start = tp_rank * bufs.local_tokens
    expected[start : start + bufs.local_tokens] = 3.0
    assert torch.equal(before_reduce, expected)


def test_folded_tp_reduction_equals_gather_plus_reduced_shared():
    """The fold is an algebraic identity, not an approximation."""
    torch.manual_seed(0)
    tp_size, local_tokens, hidden = 4, 3, 5
    partials = [
        torch.randn(tp_size * local_tokens, hidden) for _ in range(tp_size)
    ]
    routed = [torch.randn(local_tokens, hidden) for _ in range(tp_size)]

    # Old: all_gather(routed rows) + all_reduce(shared partials).
    old = torch.cat(routed, dim=0) + sum(partials)

    # New: each rank folds its routed rows into its own shared partial first,
    # then one all-reduce over the TP group.
    folded = [p.clone() for p in partials]
    for rank in range(tp_size):
        start = rank * local_tokens
        folded[rank][start : start + local_tokens] += routed[rank]
    new = sum(folded)

    assert torch.allclose(new, old, atol=1e-6)


def _make_segment_ctor_inputs(shared):
    hidden = 16
    moe = SimpleNamespace(
        hidden_dim=hidden,
        top_k=16,
        attn_tp_size=4,
        attn_tp_rank=1,
        attn_tp_group=_TP_GROUP,
        shared_experts=shared,
        gate=SimpleNamespace(weight=torch.zeros(8, hidden)),
    )
    resident = SimpleNamespace(
        shard=SimpleNamespace(K_latent=128, N=256, num_local=2),
        world_size=4,
        rank=1,
        expert_start=0,
        comm=object(),
    )
    return moe, resident, _make_cpu_pool(hidden=hidden)


def test_k3_graph_segment_accepts_a_tp_matched_shared_expert():
    shared = SimpleNamespace(_ffn=lambda x: x, _tp_size=4, _tp_group=_TP_GROUP)
    moe, resident, pool = _make_segment_ctor_inputs(shared)

    segment = K3MoEGraphSegment(moe, resident, pool, device=torch.device("cpu"))

    assert segment.tp_rank == 1
    assert segment.tp_group is _TP_GROUP


def test_k3_graph_segment_refuses_a_wrapped_shared_expert():
    # A streamed KimiLinearExpertWrapper exposes only __call__, so its TP
    # all-reduce cannot be deferred and folded.
    moe, resident, pool = _make_segment_ctor_inputs(
        SimpleNamespace(_tp_size=4, _tp_group=_TP_GROUP)
    )

    with pytest.raises(RuntimeError, match="_ffn"):
        K3MoEGraphSegment(moe, resident, pool, device=torch.device("cpu"))


def test_k3_graph_segment_refuses_a_shared_expert_outside_the_tp_group():
    moe, resident, pool = _make_segment_ctor_inputs(
        SimpleNamespace(_ffn=lambda x: x, _tp_size=1, _tp_group=None)
    )

    with pytest.raises(RuntimeError, match="TP group"):
        K3MoEGraphSegment(moe, resident, pool, device=torch.device("cpu"))


def test_nondivisible_tp_split_round_trips_through_graph_output():
    num_rows = 17
    group_size = 8
    hidden = 3
    original = torch.arange(num_rows * hidden).view(num_rows, hidden)
    ntp = (num_rows + group_size - 1) // group_size
    gathered = torch.zeros(group_size * ntp, hidden, dtype=original.dtype)

    for rank, (start, end) in enumerate(balanced_row_split(num_rows, group_size)):
        gathered[rank * ntp:rank * ntp + end - start].copy_(original[start:end])

    restored = KimiLinearDecodeGraph._reassemble_moe_rows(
        gathered, num_rows, group_size
    )
    assert torch.equal(restored, original)


def test_nondivisible_tp_scatter_uses_balanced_rows():
    rows = torch.arange(17)
    expected = [rows[start:end].tolist() for start, end in balanced_row_split(17, 8)]

    got = [
        KimiLinearDecodeGraph._scatter_moe_rows(rows, 8, rank).tolist()
        for rank in range(8)
    ]
    assert got == expected


def test_whole_model_row_maps_round_trip_nondivisible_batch_on_cpu():
    segment = KimiLinearWholeModelSegment.__new__(KimiLinearWholeModelSegment)
    segment.tp_size = 8
    segment.tp_rank = 3
    segment.device = torch.device("cpu")

    maps = segment._make_bucket_maps(17)
    splits = balanced_row_split(17, 8)
    assert maps.group_bucket == 24
    assert maps.local_bucket == 3
    assert maps.local_counts[17].item() == splits[3][1] - splits[3][0]

    rank_major_rows = maps.padded_indices[17].tolist()
    expected_rank_major = [
        row for start, end in splits for row in range(start, end)
    ]
    assert [
        row for row, valid in zip(rank_major_rows, maps.padded_valid[17].tolist())
        if valid
    ] == expected_rank_major

    original_to_rank_major = maps.original_indices[17]
    restored = [rank_major_rows[int(original_to_rank_major[row])] for row in range(17)]
    assert restored == list(range(17))


def test_whole_model_row_maps_have_fixed_device_tables():
    segment = KimiLinearWholeModelSegment.__new__(KimiLinearWholeModelSegment)
    segment.tp_size = 8
    segment.tp_rank = 0
    segment.device = torch.device("cpu")

    maps = segment._make_bucket_maps(8)
    for table in (
        maps.padded_indices,
        maps.padded_valid,
        maps.local_indices,
        maps.local_valid,
        maps.local_counts,
        maps.original_indices,
        maps.original_valid,
    ):
        assert table.device.type == "cpu"


def test_k3_whole_graph_kv_staging_is_compact_and_logically_mapped():
    segment = KimiLinearWholeModelSegment.__new__(KimiLinearWholeModelSegment)
    segment.num_layers = 4
    segment._primary_kv_layers = (1, 3)
    segment._logical_to_physical_kv = (-1, 0, -1, 1)
    segment._primary_kv_dim = 3
    segment._kv_key_buffer = torch.zeros(2, 2, 1, 1, 3)

    source = torch.arange(6, dtype=torch.float32).view(2, 1, 1, 3)
    segment._copy_primary_kv(3, source)

    assert tuple(segment._kv_key_buffer.shape) == (2, 2, 1, 1, 3)
    assert torch.equal(segment._kv_key_buffer[1, :2], source)
    with pytest.raises(KeyError, match="logical KDA layer 2"):
        segment._copy_primary_kv(2, source)


def test_whole_model_setup_and_release_reach_inline_moe_segments(monkeypatch):
    calls = []

    class Child:
        def __init__(self, name, *, fused_front=None):
            self.name = name
            self.fused_front = fused_front

        def setup_static_buffers(self, bucket):
            calls.append(("setup", self.name, bucket))

        def release_static_buffers(self, bucket):
            calls.append(("release", self.name, bucket))

    segment = KimiLinearWholeModelSegment.__new__(KimiLinearWholeModelSegment)
    segment.max_bucket_size = 8
    segment._bucket_maps = {}
    segment.layer_segments = [Child("attention")]
    segment.moe_segments = {1: Child("moe", fused_front=torch.empty(0))}
    segment._kv_key_buffer = None
    segment._primary_kv_dim = 0
    segment._kda_cu_seqlens = {}
    segment.device = torch.device("cpu")
    monkeypatch.setattr(segment, "_make_bucket_maps", lambda bucket: object())

    segment.setup_static_buffers(8)
    segment.release_static_buffers(8)

    assert calls == [
        ("setup", "attention", 8),
        ("setup", "moe", 8),
        ("release", "attention", 8),
        ("release", "moe", 8),
    ]
    assert 8 not in segment._bucket_maps


def test_gate_forward_equals_router_logits_plus_selector():
    gate = _make_gate()
    hidden = torch.randn(5, 1, gate.gating_dim)

    ref_idx, ref_weight = gate(hidden)
    logits = gate.router_logits(hidden.view(-1, gate.gating_dim))
    idx, weight = gate.select_experts(logits)

    assert logits.dtype == torch.float32
    assert tuple(logits.shape) == (5, gate.num_experts)
    assert torch.equal(ref_idx, idx)
    assert torch.equal(ref_weight, weight)


def test_gate_selector_accepts_fp32_logits_from_a_bf16_activation():
    gate = _make_gate()
    hidden = torch.randn(4, 1, gate.gating_dim, dtype=torch.bfloat16)

    ref_idx, ref_weight = gate(hidden)
    logits = gate.router_logits(hidden.view(-1, gate.gating_dim))
    idx, weight = gate.select_experts(logits)

    assert logits.dtype == torch.float32
    assert weight.dtype == torch.float32
    assert tuple(idx.shape) == (4, gate.top_k)
    assert torch.equal(ref_idx, idx)
    assert torch.equal(ref_weight, weight)


def test_gate_group_topk_branch_still_masks_non_selected_groups():
    gate = _make_gate(num_experts=8, top_k=2, n_group=4, topk_group=1)
    hidden = torch.randn(6, 1, gate.gating_dim)

    idx, _ = gate(hidden)

    # topk_group=1 of 4 groups leaves exactly one 2-expert group live, so both
    # selected experts must come from that group.
    groups = idx // (gate.num_experts // gate.num_expert_group)
    assert torch.equal(groups, groups[:, :1].expand_as(groups))


def test_fuse_router_and_down_proj_rebinds_both_weights_onto_one_slab():
    moe, down_proj = _make_fusable(num_experts=4, latent=6, hidden=8)
    gate_ref = moe.gate.weight.detach().clone()
    down_ref = down_proj.weight.detach().clone()

    fused = fuse_router_and_down_proj(moe, down_proj)

    assert fused is not None
    assert fused.dtype == torch.bfloat16
    assert tuple(fused.shape) == (4 + 6, 8)
    assert torch.equal(fused[:4], gate_ref)
    assert torch.equal(fused[4:], down_ref)
    # Views, not copies: no duplicate weight storage is retained.
    assert moe.gate.weight.data_ptr() == fused.data_ptr()
    assert down_proj.weight.data_ptr() == fused[4:].data_ptr()


def test_fuse_router_and_down_proj_is_idempotent():
    moe, down_proj = _make_fusable()

    fused = fuse_router_and_down_proj(moe, down_proj)
    gate_ptr = moe.gate.weight.data_ptr()
    again = fuse_router_and_down_proj(moe, down_proj)

    assert again is fused
    assert moe.gate.weight.data_ptr() == gate_ptr


def test_fuse_router_and_down_proj_declines_a_non_bf16_router():
    moe, down_proj = _make_fusable(gate_dtype=torch.float32)
    gate_ptr = moe.gate.weight.data_ptr()
    down_ptr = down_proj.weight.data_ptr()

    assert fuse_router_and_down_proj(moe, down_proj) is None
    assert moe.gate.weight.data_ptr() == gate_ptr
    assert down_proj.weight.data_ptr() == down_ptr


def test_fuse_router_and_down_proj_declines_mismatched_hidden_sizes():
    moe, _ = _make_fusable(hidden=8)
    down_proj = nn.Linear(16, 6, bias=False).to(torch.bfloat16)
    gate_ptr = moe.gate.weight.data_ptr()
    down_ptr = down_proj.weight.data_ptr()

    assert fuse_router_and_down_proj(moe, down_proj) is None
    assert moe.gate.weight.data_ptr() == gate_ptr
    assert down_proj.weight.data_ptr() == down_ptr


def test_split_fused_front_keeps_fp32_logits_and_casts_latent_once():
    fused_out = torch.randn(3, 10, dtype=torch.float32)

    logits, latent = split_fused_front(fused_out, 4)

    assert logits.dtype == torch.float32
    assert torch.equal(logits, fused_out[:, :4])
    assert latent.dtype == torch.bfloat16
    assert torch.equal(latent, fused_out[:, 4:].to(torch.bfloat16))


_CUDA = torch.device("cuda")
_SLAB = torch.empty(0, dtype=torch.bfloat16)


def test_fused_gate_kernel_eligible_accepts_the_k3_router():
    gate = _make_gate(num_experts=32, top_k=16)

    assert fused_gate_kernel_eligible(gate, fused_front=_SLAB, device=_CUDA)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"top_k": 8}, id="top_k_not_16"),
        pytest.param({"scoring_func": "softmax", "top_k": 16}, id="softmax"),
        pytest.param({"norm_topk_prob": False, "top_k": 16}, id="no_renormalize"),
        pytest.param(
            {"top_k": 16, "n_group": 4, "topk_group": 1}, id="group_routing"
        ),
    ],
)
def test_fused_gate_kernel_declines_recipes_the_kernel_cannot_reproduce(kwargs):
    gate = _make_gate(num_experts=32, **kwargs)

    assert not fused_gate_kernel_eligible(gate, fused_front=_SLAB, device=_CUDA)


def test_fused_gate_kernel_declines_without_cuda_or_a_fused_front():
    gate = _make_gate(num_experts=32, top_k=16)

    assert not fused_gate_kernel_eligible(
        gate, fused_front=_SLAB, device=torch.device("cpu")
    )
    assert not fused_gate_kernel_eligible(gate, fused_front=None, device=_CUDA)


def test_fused_gate_kernel_declines_a_non_fp32_correction_bias():
    gate = _make_gate(num_experts=32, top_k=16)
    gate.e_score_correction_bias = nn.Parameter(
        gate.e_score_correction_bias.detach().to(torch.bfloat16)
    )

    assert not fused_gate_kernel_eligible(gate, fused_front=_SLAB, device=_CUDA)


def test_k3_graph_fused_front_routes_into_preallocated_native_buffers(monkeypatch):
    class GateCalled(RuntimeError):
        pass

    local_tokens, hidden, experts, latent = 2, 4, 8, 6
    local_indices = torch.empty(local_tokens, 16, dtype=torch.int32)
    local_weights = torch.empty(local_tokens, 16, dtype=torch.float32)
    local_latent = torch.empty(local_tokens, latent, dtype=torch.bfloat16)
    bufs = SimpleNamespace(
        local_tokens=local_tokens,
        global_tokens=local_tokens,
        local_indices=local_indices,
        local_weights=local_weights,
        local_latent=local_latent,
    )
    valid = torch.tensor([1], dtype=torch.int32)
    called = {}

    segment = K3MoEGraphSegment.__new__(K3MoEGraphSegment)
    segment.pool = SimpleNamespace(get=lambda bucket: bufs)
    segment.hidden_size = hidden
    segment.num_experts = experts
    segment.top_k = 16
    segment.fused_front = torch.empty(experts + latent, hidden)
    segment.fused_gate_kernel = True
    segment.moe = SimpleNamespace(
        gate=SimpleNamespace(
            e_score_correction_bias=torch.randn(experts),
            routed_scaling_factor=2.5,
        )
    )

    fused_out = torch.randn(local_tokens, experts + latent)
    monkeypatch.setattr(k3_moe_graph.torch, "mm", lambda *args, **kwargs: fused_out)

    def fake_gate(logits, bias, **kwargs):
        called.update(logits=logits, bias=bias, **kwargs)
        raise GateCalled

    monkeypatch.setattr(k3_moe_graph, "gate_sigmoid_topk_cuda", fake_gate)

    with pytest.raises(GateCalled):
        segment.forward(
            padded=torch.empty(local_tokens, hidden),
            local=torch.empty(local_tokens, hidden),
            num_valid_tokens=valid,
        )

    assert tuple(called["logits"].shape) == (local_tokens, experts)
    assert called["logits"].stride(0) == experts + latent
    assert called["topk_indices"] is local_indices
    assert called["topk_weights"] is local_weights
    assert called["num_valid_tokens"] is valid
    assert called["k"] == 16
    # The latent half of the very same fused rows is cast in place, so no
    # separate strided FP32->BF16 copy is left in the graph.
    assert called["latent_out"] is local_latent
    assert called["latent_offset"] == experts


def test_fused_slab_gemm_reproduces_the_router_and_down_projections():
    num_experts, latent, hidden = 4, 6, 8
    moe, down_proj = _make_fusable(
        num_experts=num_experts, latent=latent, hidden=hidden
    )
    x = torch.randn(3, hidden, dtype=torch.bfloat16)
    ref_logits = F.linear(x.float(), moe.gate.weight.float(), None)
    ref_latent = torch.mm(x.float(), down_proj.weight.float().t())

    fused = fuse_router_and_down_proj(moe, down_proj)
    # CPU stand-in for the captured ``torch.mm(..., out_dtype=torch.float32)``.
    logits, got_latent = split_fused_front(
        torch.mm(x.float(), fused.float().t()), num_experts
    )

    assert torch.allclose(logits, ref_logits, atol=1e-5)
    # The latent half is cast to BF16 exactly once, so it matches the separate
    # projection to within one BF16 ulp.
    assert torch.allclose(got_latent.float(), ref_latent, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
#  Whole-model replay telemetry
# ---------------------------------------------------------------------------


class _FakeEvent:
    def __init__(self, ready=True, elapsed=0.0):
        self.ready = ready
        self.elapsed = elapsed
        self.records = 0

    def record(self):
        self.records += 1

    def query(self):
        return self.ready

    def elapsed_time(self, other):
        return self.elapsed


def _make_telemetry_graph():
    graph = KimiLinearDecodeGraph.__new__(KimiLinearDecodeGraph)
    graph.rank = 0
    graph._whole_bucket = 8
    graph._whole_replay_events = None
    graph._whole_replay_pending = False
    graph._whole_replays = 0
    graph._whole_replay_samples = 0
    graph._whole_replay_ms = 0.0
    graph._whole_kv_stage_samples = 0
    graph._whole_kv_stage_host_ms = 0.0
    return graph


def test_whole_replay_telemetry_reads_only_completed_events():
    graph = _make_telemetry_graph()
    start, end = _FakeEvent(elapsed=4.0), _FakeEvent(ready=False)
    graph._whole_replay_events = (start, end)
    graph._whole_replay_pending = True

    graph._collect_whole_replay_sample()

    # A replay still in flight stays pending: the measured path is never
    # synchronized just to read its own timing.
    assert graph._whole_replay_pending
    assert graph._whole_replay_samples == 0

    end.ready = True
    graph._collect_whole_replay_sample()

    assert not graph._whole_replay_pending
    assert graph._whole_replay_samples == 1
    assert graph._whole_replay_ms == 4.0


def test_get_capture_stats_reports_whole_model_replay_telemetry():
    graph = _make_telemetry_graph()
    graph.manager = SimpleNamespace(get_capture_stats=lambda: {})
    graph._capture_stats = {}
    graph.whole_manager = SimpleNamespace(get_capture_stats=lambda: {})
    graph._whole_capture_stats = {}
    graph.moe_manager = None
    graph._whole_replays = 4
    graph._whole_replay_samples = 2
    graph._whole_replay_ms = 5.0
    graph._whole_kv_stage_samples = 4
    graph._whole_kv_stage_host_ms = 2.0

    telemetry = graph.get_capture_stats()["kimi_whole_replay"]

    assert telemetry["replays"] == 4
    assert telemetry["replay_samples"] == 2
    assert telemetry["replay_ms_avg"] == 2.5
    assert telemetry["kv_stage_host_ms_avg"] == 0.5
