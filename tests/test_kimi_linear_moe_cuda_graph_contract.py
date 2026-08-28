"""CPU contracts for the K3 resident-MXFP4 CUDA-graph MoE seam.

These tests deliberately do not launch CUDA.  They pin the fixed geometry and
the rank-major <-> balanced-row mapping that the remote CUDA gate exercises.
"""

import torch

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

