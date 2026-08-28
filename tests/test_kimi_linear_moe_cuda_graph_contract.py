"""CPU contracts for the K3 resident-MXFP4 CUDA-graph MoE seam.

These tests deliberately do not launch CUDA.  They pin the fixed geometry and
the rank-major <-> balanced-row mapping that the remote CUDA gate exercises.
"""

from types import SimpleNamespace

import pytest
import torch

import batchgen.models.moonshotai.kimi_linear.moe_cuda_graph_segments as k3_moe_graph

from batchgen.models.moonshotai.kimi_linear.cuda_graph_segments import (
    KimiLinearDecodeGraph,
)
from batchgen.models.moonshotai.kimi_linear.moe_cuda_graph_segments import (
    K3MoEGraphBufferPool,
    K3MoEGraphSegment,
    k3_moe_graph_buckets,
)
from batchgen.models.moonshotai.kimi_linear.moe_tp_reshard import (
    balanced_row_split,
)
from batchgen.models.moonshotai.kimi_linear.whole_model_cuda_graph_segments import (
    KimiLinearWholeModelSegment,
)


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
