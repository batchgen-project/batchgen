import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _reference(expert_output, topk_pos, topk_indices, topk_weights):
    # Ascending expert-id order (stable), with BF16 rounding of every weighted
    # contribution and of every accumulation step; invalid slots (pos < 0) skip.
    order = torch.sort(topk_indices.long(), dim=1, stable=True).indices
    pos = torch.gather(topk_pos.long(), 1, order)
    weight = torch.gather(topk_weights, 1, order).to(torch.bfloat16).float()
    acc = torch.zeros(
        topk_pos.shape[0],
        expert_output.shape[1],
        dtype=torch.bfloat16,
        device=expert_output.device,
    )
    for k in range(topk_pos.shape[1]):
        valid = (pos[:, k] >= 0)[:, None]
        rows = expert_output[pos[:, k].clamp(min=0)].float()
        weighted = (rows * weight[:, k : k + 1]).to(torch.bfloat16)
        updated = (acc.float() + weighted.float()).to(torch.bfloat16)
        acc = torch.where(valid, updated, acc)
    return acc


def _inputs(n, k, h, experts=256, invalid_frac=0.1, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rows = n * k + 5
    expert_output = (
        torch.randn(rows, h, device="cuda", generator=gen)
        * torch.logspace(-3, 1, h, device="cuda")
    ).to(torch.bfloat16)
    topk_pos = torch.randperm(rows, device="cuda", generator=gen)[: n * k]
    topk_pos = topk_pos.view(n, k).to(torch.int32)
    drop = torch.rand(n, k, device="cuda", generator=gen) < invalid_frac
    topk_pos[drop] = -1
    topk_indices = torch.stack(
        [torch.randperm(experts, device="cuda", generator=gen)[:k] for _ in range(n)]
    ).to(torch.int32)
    topk_weights = torch.rand(n, k, device="cuda", generator=gen) * 2.5
    return expert_output, topk_pos, topk_indices, topk_weights


@pytest.mark.parametrize("n,k,h", [(37, 8, 6144), (1024, 8, 6144), (19, 4, 128), (7, 2, 256)])
def test_ordered_reduce_is_bitwise_equal_to_reference(n, k, h):
    from batchgen.moe.dispatch_scatter_3d import reduce_weighted_scatter_bf16_ordered

    expert_output, topk_pos, topk_indices, topk_weights = _inputs(n, k, h)
    # Mirror production: the output is a row slice of a larger [T, H] buffer.
    backing = torch.full((n + 64, h), float("nan"), dtype=torch.bfloat16, device="cuda")
    out = reduce_weighted_scatter_bf16_ordered(
        expert_output, topk_pos, topk_indices, topk_weights,
        n, h, k, output=backing[32 : 32 + n],
    )
    ref = _reference(expert_output, topk_pos, topk_indices, topk_weights)
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
    assert torch.isnan(backing[:32]).all() and torch.isnan(backing[32 + n :]).all()
