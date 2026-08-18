import torch
import torch.nn.functional as F

from batchgen.models.moonshotai.kimi_linear.Parallel_Strategy_Manager import (
    shard_mla_tensor,
    shard_shared_expert_tensor,
)
from batchgen.models.moonshotai.kimi_linear.model import SituAndMul


def test_shared_expert_tp_shards_reconstruct_full_situ_mlp():
    torch.manual_seed(7)
    tokens, hidden, intermediate, tp_size = 5, 12, 16, 4
    x = torch.randn(tokens, hidden)
    gate = torch.randn(intermediate, hidden)
    up = torch.randn(intermediate, hidden)
    down = torch.randn(hidden, intermediate)
    activation = SituAndMul(beta=4.0, linear_beta=25.0)

    full = F.linear(
        activation(torch.cat([F.linear(x, gate), F.linear(x, up)], dim=-1)),
        down,
    )

    partials = []
    for rank in range(tp_size):
        gate_local = shard_shared_expert_tensor(
            gate, "gate_proj.weight", tp_size, rank
        )
        up_local = shard_shared_expert_tensor(
            up, "up_proj.weight", tp_size, rank
        )
        down_local = shard_shared_expert_tensor(
            down, "down_proj.weight", tp_size, rank
        )
        partials.append(
            F.linear(
                activation(
                    torch.cat(
                        [
                            F.linear(x, gate_local),
                            F.linear(x, up_local),
                        ],
                        dim=-1,
                    )
                ),
                down_local,
            )
        )

    torch.testing.assert_close(sum(partials), full, rtol=1e-5, atol=1e-5)


def test_shared_expert_tp_size_one_is_identity():
    tensor = torch.randn(8, 6)
    assert shard_shared_expert_tensor(
        tensor, "gate_proj.weight", 1, 0
    ) is tensor


def test_mla_tp_shards_reconstruct_head_outputs_and_o_projection():
    torch.manual_seed(8)
    tokens, hidden, heads, q_dim, v_dim, tp_size = 4, 12, 8, 6, 4, 4
    q_lora, kv_lora = 5, 3
    q_a = torch.randn(q_lora, hidden)
    q_b = torch.randn(heads * q_dim, q_lora)
    kv_a = torch.randn(kv_lora + 2, hidden)
    kv_b = torch.randn(heads * (q_dim - 2 + v_dim), kv_lora)
    gate = torch.randn(heads * v_dim, hidden)
    o_proj = torch.randn(hidden, heads * v_dim)
    x = torch.randn(tokens, hidden)
    local_values = torch.randn(tokens, heads, v_dim)

    q_full = F.linear(F.linear(x, q_a), q_b)
    kv_full = F.linear(F.linear(x, kv_a)[:, :kv_lora], kv_b)
    gate_full = F.linear(x, gate).view(tokens, heads, v_dim).sigmoid()
    out_full = F.linear(
        (local_values * gate_full).reshape(tokens, heads * v_dim),
        o_proj,
    )

    q_parts = []
    kv_parts = []
    out_parts = []
    local_heads = heads // tp_size
    for rank in range(tp_size):
        q_a_local = shard_mla_tensor(q_a, "q_a_proj.weight", tp_size, rank)
        kv_a_local = shard_mla_tensor(
            kv_a, "kv_a_proj_with_mqa.weight", tp_size, rank
        )
        q_b_local = shard_mla_tensor(
            q_b, "q_b_proj.weight", tp_size, rank
        )
        kv_b_local = shard_mla_tensor(
            kv_b, "kv_b_proj.weight", tp_size, rank
        )
        gate_local = shard_mla_tensor(
            gate, "g_proj.weight", tp_size, rank
        )
        o_local = shard_mla_tensor(
            o_proj, "o_proj.weight", tp_size, rank
        )
        q_parts.append(F.linear(F.linear(x, q_a_local), q_b_local))
        kv_parts.append(
            F.linear(F.linear(x, kv_a_local)[:, :kv_lora], kv_b_local)
        )
        head_start = rank * local_heads
        values = local_values[:, head_start:head_start + local_heads]
        gates = F.linear(x, gate_local).view(
            tokens, local_heads, v_dim
        ).sigmoid()
        out_parts.append(
            F.linear(
                (values * gates).reshape(tokens, local_heads * v_dim),
                o_local,
            )
        )

    torch.testing.assert_close(
        torch.cat(q_parts, dim=-1), q_full, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        torch.cat(kv_parts, dim=-1), kv_full, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(sum(out_parts), out_full, rtol=1e-5, atol=1e-5)
