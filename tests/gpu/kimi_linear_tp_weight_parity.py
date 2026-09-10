"""8-rank GPU parity gate for Kimi-K3 replicated-weight TP sharding.

Run with:
  torchrun --standalone --nproc-per-node=8 \
    tests/gpu/kimi_linear_tp_weight_parity.py
"""

import importlib.util
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

_SHARDING_PATH = (
    Path(__file__).resolve().parents[2]
    / "batchgen/models/moonshotai/kimi_linear/tp_weight_sharding.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "kimi_linear_tp_weight_sharding", _SHARDING_PATH
)
_SHARDING = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SHARDING)
shard_mla_tensor = _SHARDING.shard_mla_tensor
shard_shared_expert_tensor = _SHARDING.shard_shared_expert_tensor


def _err_ratio(actual, expected):
    diff = (actual.float() - expected.float()).pow(2).mean().sqrt()
    ref = expected.float().pow(2).mean().sqrt().clamp_min(1e-12)
    return float((diff / ref).item())


def _broadcast(tensor, src=0):
    dist.broadcast(tensor, src=src)
    return tensor


def _situ_and_mul(x, beta=4.0, linear_beta=25.0):
    gate, up = x.chunk(2, dim=-1)
    gate = gate.float()
    up = up.float()
    situ = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    up = linear_beta * torch.tanh(up / linear_beta)
    return (situ * up).to(x.dtype)


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 8:
        raise RuntimeError(f"expected TP world=8, got {world}")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    torch.manual_seed(20260818)
    hidden, shared_intermediate = 128, 256
    heads, q_dim, v_dim = 16, 12, 8
    q_lora, kv_lora, tokens = 32, 16, 7
    local_heads = heads // world

    x = _broadcast(torch.randn(tokens, hidden, device=device))

    # Shared expert: gate/up column-parallel + down row-parallel + all-reduce.
    gate = _broadcast(
        torch.randn(shared_intermediate, hidden, device=device)
    )
    up = _broadcast(torch.randn(shared_intermediate, hidden, device=device))
    down = _broadcast(
        torch.randn(hidden, shared_intermediate, device=device)
    )
    shared_full = F.linear(
        _situ_and_mul(
            torch.cat([F.linear(x, gate), F.linear(x, up)], dim=-1)
        ),
        down,
    )
    shared_local = F.linear(
        _situ_and_mul(
            torch.cat(
                [
                    F.linear(
                        x,
                        shard_shared_expert_tensor(
                            gate, "gate_proj.weight", world, rank
                        ),
                    ),
                    F.linear(
                        x,
                        shard_shared_expert_tensor(
                            up, "up_proj.weight", world, rank
                        ),
                    ),
                ],
                dim=-1,
            )
        ),
        shard_shared_expert_tensor(
            down, "down_proj.weight", world, rank
        ),
    )
    dist.all_reduce(shared_local)
    shared_err = _err_ratio(shared_local, shared_full)

    # MLA: low-rank A projections replicated, head projections column-sharded,
    # output projection row-sharded, and output partials reduced.
    q_a = _broadcast(torch.randn(q_lora, hidden, device=device))
    q_b = _broadcast(torch.randn(heads * q_dim, q_lora, device=device))
    kv_a = _broadcast(torch.randn(kv_lora + 4, hidden, device=device))
    kv_b = _broadcast(
        torch.randn(heads * (q_dim - 4 + v_dim), kv_lora, device=device)
    )
    gate_mla = _broadcast(torch.randn(heads * v_dim, hidden, device=device))
    o_proj = _broadcast(torch.randn(hidden, heads * v_dim, device=device))
    values = _broadcast(torch.randn(tokens, heads, v_dim, device=device))

    q_full = F.linear(F.linear(x, q_a), q_b)
    kv_full = F.linear(F.linear(x, kv_a)[:, :kv_lora], kv_b)
    gate_full = F.linear(x, gate_mla).view(
        tokens, heads, v_dim
    ).sigmoid()
    mla_full = F.linear(
        (values * gate_full).reshape(tokens, heads * v_dim),
        o_proj,
    )

    q_local = F.linear(
        F.linear(
            x,
            shard_mla_tensor(q_a, "q_a_proj.weight", world, rank),
        ),
        shard_mla_tensor(q_b, "q_b_proj.weight", world, rank),
    )
    kv_local = F.linear(
        F.linear(
            x,
            shard_mla_tensor(
                kv_a, "kv_a_proj_with_mqa.weight", world, rank
            ),
        )[:, :kv_lora],
        shard_mla_tensor(kv_b, "kv_b_proj.weight", world, rank),
    )
    gate_local = F.linear(
        x,
        shard_mla_tensor(gate_mla, "g_proj.weight", world, rank),
    ).view(tokens, local_heads, v_dim).sigmoid()
    value_local = values[
        :, rank * local_heads:(rank + 1) * local_heads
    ]
    mla_local = F.linear(
        (value_local * gate_local).reshape(
            tokens, local_heads * v_dim
        ),
        shard_mla_tensor(o_proj, "o_proj.weight", world, rank),
    )
    dist.all_reduce(mla_local)

    q_parts = [torch.empty_like(q_local) for _ in range(world)]
    kv_parts = [torch.empty_like(kv_local) for _ in range(world)]
    dist.all_gather(q_parts, q_local)
    dist.all_gather(kv_parts, kv_local)
    q_err = _err_ratio(torch.cat(q_parts, dim=-1), q_full)
    kv_err = _err_ratio(torch.cat(kv_parts, dim=-1), kv_full)
    mla_err = _err_ratio(mla_local, mla_full)

    errors = torch.tensor(
        [shared_err, q_err, kv_err, mla_err],
        device=device,
        dtype=torch.float32,
    )
    dist.all_reduce(errors, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(
            "shared_err={:.3e} q_err={:.3e} kv_err={:.3e} "
            "mla_err={:.3e}".format(*errors.cpu().tolist())
        )
    if float(errors.max().item()) > 1e-5:
        raise AssertionError(f"TP parity error too large: {errors.cpu().tolist()}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
