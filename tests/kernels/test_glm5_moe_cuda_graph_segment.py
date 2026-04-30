import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class _RankMaskSegment:
    def __init__(self, pool, *, world_size: int, hidden_size: int, topk: int, device: torch.device):
        from batchgen.cuda_graph.graph_manager import TensorSpec

        self.pool = pool
        self.world_size = world_size
        self.hidden_size = hidden_size
        self.topk = topk
        self.device = device
        self.TensorSpec = TensorSpec
        max_bucket = max(pool.bucket_sizes)
        rows = world_size * max_bucket
        self.static_indices = (
            torch.arange(rows * topk, dtype=torch.int32, device=device).view(rows, topk)
        )

    def setup_static_buffers(self, bucket_size: int) -> None:
        self.pool.setup()

    def get_static_input_specs(self, bucket_size: int):
        return {
            "padded": self.TensorSpec(("batch_size", self.hidden_size), torch.bfloat16),
            "rank_token_counts": self.TensorSpec((self.world_size,), torch.int64),
        }

    def get_static_output_specs(self, bucket_size: int):
        return {
            "masked": self.TensorSpec((self.world_size * bucket_size, self.topk), torch.int32),
        }

    def forward(self, *, padded: torch.Tensor, rank_token_counts: torch.Tensor):
        bucket_size = padded.shape[0]
        bufs = self.pool.get(bucket_size)
        rows = self.world_size * bucket_size

        bufs.topk_indices.copy_(self.static_indices[:rows])
        valid_per_row = rank_token_counts[bufs.rank_ids]
        padding_mask = bufs.local_pos >= valid_per_row
        padding_mask_2d = padding_mask.unsqueeze(1).expand_as(bufs.topk_indices)
        torch.where(
            padding_mask_2d,
            bufs.topk_negative_ones,
            bufs.topk_indices,
            out=bufs.topk_masked_indices,
        )
        return {"masked": bufs.topk_masked_indices}


def test_glm5_moe_rank_padding_mask_replays_with_new_counts():
    from batchgen.cuda_graph import BatchSizeBucketing, CUDAGraphManager
    from batchgen.models.glm.glm5.moe_cuda_graph_segments import Glm5MoEGraphBufferPool

    device = torch.device("cuda")
    world_size = 3
    hidden_size = 16
    topk = 4
    bucket = 2
    pool = Glm5MoEGraphBufferPool(
        world_size=world_size,
        hidden_size=hidden_size,
        num_experts_per_tok=topk,
        num_local_experts=2,
        intermediate_size=8,
        device=device,
        bucket_sizes=[bucket],
        base_mtp=8,
    )
    manager = CUDAGraphManager(BatchSizeBucketing([bucket]), device=device)
    segment = _RankMaskSegment(
        pool,
        world_size=world_size,
        hidden_size=hidden_size,
        topk=topk,
        device=device,
    )
    manager.register_segment("mask", segment)
    manager.warmup_and_capture_buckets([bucket])

    padded = torch.zeros(bucket, hidden_size, dtype=torch.bfloat16, device=device)
    counts_a = torch.tensor([1, 0, 2], dtype=torch.int64, device=device)
    out_a = manager.replay("mask", bucket, padded=padded, rank_token_counts=counts_a)["masked"]
    torch.cuda.synchronize()
    expected_a = torch.tensor(
        [
            [0, 1, 2, 3],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [16, 17, 18, 19],
            [20, 21, 22, 23],
        ],
        dtype=torch.int32,
        device=device,
    )
    torch.testing.assert_close(out_a, expected_a, rtol=0, atol=0)

    counts_b = torch.tensor([2, 1, 0], dtype=torch.int64, device=device)
    out_b = manager.replay("mask", bucket, padded=padded, rank_token_counts=counts_b)["masked"]
    torch.cuda.synchronize()
    expected_b = torch.tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
        ],
        dtype=torch.int32,
        device=device,
    )
    torch.testing.assert_close(out_b, expected_b, rtol=0, atol=0)


def test_glm5_moe_graph_buffer_pool_shapes():
    from batchgen.models.glm.glm5.moe_cuda_graph_segments import Glm5MoEGraphBufferPool

    device = torch.device("cuda")
    pool = Glm5MoEGraphBufferPool(
        world_size=2,
        hidden_size=32,
        num_experts_per_tok=8,
        num_local_experts=4,
        intermediate_size=16,
        device=device,
        bucket_sizes=[1, 2],
        base_mtp=8,
    )
    bufs = pool.get(2)

    assert bufs.padded.shape == (2, 32)
    assert bufs.all_tokens.shape == (4, 32)
    assert bufs.topk_indices.shape == (4, 8)
    assert bufs.dispatched_x.shape == (4 * bufs.max_tokens_padded, 32)
    assert bufs.intermediate.shape == (4 * bufs.max_tokens_padded, 16)
    assert bufs.cu_seqlens.shape == (5,)
