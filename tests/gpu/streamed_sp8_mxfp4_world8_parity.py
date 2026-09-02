"""World-8 parity gate for Kimi-K3 TP8-attention + SP8-MoE prefill.

Run on one 8-GPU H20 node. Each rank:

1. owns one contiguous 1/8 expert shard;
2. node-locally all-gathers the six MXFP4 tensors into one full layer;
3. computes only its contiguous 1/8 token-row slice with grouped Marlin;
4. node-locally gathers rows.

The reconstructed output is compared with a world-1 all-expert resident
forward on the same synthetic K3 block. No cross-node process group exists in
this harness, so the only communication is the intended TP8-local weight and
row gathering.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _err_ratio(actual, reference):
    actual = actual.float()
    reference = reference.float()
    rms = reference.pow(2).mean().sqrt()
    return float(
        (actual - reference).pow(2).mean().sqrt() / (rms + 1e-8)
    )


def _bf16_gemm_metrics(actual, reference):
    actual = actual.float().reshape(-1)
    reference = reference.float().reshape(-1)
    diff = actual - reference
    rel_l2 = float(diff.norm() / reference.norm().clamp_min(1e-12))
    cosine = float(torch.nn.functional.cosine_similarity(
        actual.unsqueeze(0), reference.unsqueeze(0)
    ))
    max_abs_over_std = float(
        diff.abs().max() / reference.std().clamp_min(1e-12)
    )
    return {
        "rel_l2": rel_l2,
        "cosine": cosine,
        "max_abs_over_std": max_abs_over_std,
    }


def _install_core_engine_from_cache():
    """Prevent synthetic model imports from rebuilding the unrelated core."""
    if "batchgen.core_engine" in sys.modules:
        return
    cache = os.environ.get("TORCH_EXTENSIONS_DIR")
    if not cache:
        raise RuntimeError(
            "TORCH_EXTENSIONS_DIR is required for the world-8 parity gate"
        )
    so_path = Path(cache) / "core_engine" / "core_engine.so"
    if not so_path.is_file():
        raise FileNotFoundError(so_path)
    spec = importlib.util.spec_from_file_location(
        "batchgen.core_engine", so_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["batchgen.core_engine"] = module
    spec.loader.exec_module(module)


def _sources(experts):
    return [
        {
            "w1": (expert.w1.weight_packed.data, expert.w1.weight_scale.data),
            "w3": (expert.w3.weight_packed.data, expert.w3.weight_scale.data),
            "w2": (expert.w2.weight_packed.data, expert.w2.weight_scale.data),
        }
        for expert in experts
    ]


def _gather_sources(block, rank, world, device):
    experts = list(block.experts)
    assert len(experts) % world == 0
    per_rank = len(experts) // world
    start = rank * per_rank
    local = experts[start:start + per_rank]
    gathered = {}
    for name in (
        "w1.weight_packed",
        "w1.weight_scale",
        "w3.weight_packed",
        "w3.weight_scale",
        "w2.weight_packed",
        "w2.weight_scale",
    ):
        projection, tensor_name = name.split(".", 1)
        local_tensor = torch.stack(
            [getattr(getattr(expert, projection), tensor_name).data
             for expert in local],
            dim=0,
        ).contiguous()
        full = torch.empty(
            (len(experts), *local_tensor.shape[1:]),
            dtype=local_tensor.dtype,
            device=device,
        )
        dist.all_gather_into_tensor(full, local_tensor)
        gathered[name] = full

    sources = []
    for expert_idx in range(len(experts)):
        sources.append({
            "w1": (
                gathered["w1.weight_packed"][expert_idx],
                gathered["w1.weight_scale"][expert_idx],
            ),
            "w3": (
                gathered["w3.weight_packed"][expert_idx],
                gathered["w3.weight_scale"][expert_idx],
            ),
            "w2": (
                gathered["w2.weight_packed"][expert_idx],
                gathered["w2.weight_scale"][expert_idx],
            ),
        })
    return sources, per_rank


def _layer(block, shard, ResidentEPMXFP4MoELayer):
    return ResidentEPMXFP4MoELayer(
        layer_idx=0,
        shard=shard,
        down_proj=block.routed_expert_down_proj,
        norm=block.routed_expert_norm if block.latent_moe_use_norm else None,
        up_proj=block.routed_expert_up_proj,
        world_size=1,
        expert_start=0,
    )


class _StaticSP8Buffer:
    """Minimal weight-buffer contract for the activation-path parity gate."""

    def __init__(self, shard, rank, world):
        self._shard = shard
        self.tp_group = dist.group.WORLD
        self.tp_rank = rank
        self.tp_size = world
        self.experts_per_rank = shard.num_local
        self.expert_start = rank * shard.num_local

    def load(self, _layer_idx):
        return self._shard

    def begin_prefetch_next(self, _layer_idx):
        return None

    def allow_full_overwrite(self):
        return None


def worker(rank, world, out_path, master_port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    device = torch.device("cuda", rank)

    _install_core_engine_from_cache()
    import kimi_k3_harness as H
    import test_kimi_linear_mxfp4_latent_moe_serving as TT
    from batchgen.moe.fused_moe_mxfp4_resident import (
        ResidentEPMXFP4MoELayer,
        build_layer_shard,
    )
    from batchgen.moe.streamed_sp8_mxfp4 import (
        StreamedSP8MXFP4MoELayer,
    )
    from batchgen.models.moonshotai.kimi_linear.moe_tp_reshard import (
        all_gather_rows,
        all_gather_rows_add_,
        all_gather_rows_into,
        reduce_scatter_rows,
        scatter_rows,
    )
    from batchgen.models.moonshotai.kimi_linear.tp_weight_sharding import (
        shard_shared_expert_tensor,
    )
    from batchgen.models.moonshotai.kimi_linear.block_residual import (
        apply_attn_res,
    )

    block, cfg, _ = TT._build_mxfp4_serving_block("syn25_mxfp4")
    block = block.to(device)
    num_rows = 64
    hidden = cfg.hidden_size
    x = H.seeded_input(
        "streamed-sp8-world8", 1, num_rows, hidden, dtype=torch.bfloat16
    ).to(device).reshape(num_rows, hidden)

    experts_per_rank = len(block.experts) // world
    expert_start = rank * experts_per_rank
    local_shard = build_layer_shard(
        _sources(block.experts[expert_start:expert_start + experts_per_rank]),
        device,
    )
    buffer = _StaticSP8Buffer(local_shard, rank, world)
    common = dict(
        layer_idx=0,
        buffer=buffer,
        down_proj=block.routed_expert_down_proj,
        norm=(
            block.routed_expert_norm
            if block.latent_moe_use_norm
            else None
        ),
        up_proj=block.routed_expert_up_proj,
    )
    wide_layer = StreamedSP8MXFP4MoELayer(
        **common,
        chunk_rows=4096,
    )
    striped_layer = StreamedSP8MXFP4MoELayer(
        **common,
        chunk_rows=2048,
        collective_chunk_rows=2,
        collective_stripe_threshold_rows=1,
    )
    x_local = scatter_rows(x, world, rank)
    with torch.no_grad():
        wide_local = wide_layer.forward(x_local, block.gate, num_rows)
        striped_local = striped_layer.forward(x_local, block.gate, num_rows)
        wide = all_gather_rows(
            wide_local, num_rows, world, rank, dist.group.WORLD
        )
        striped = all_gather_rows(
            striped_local, num_rows, world, rank, dist.group.WORLD
        )

        reference_shard = build_layer_shard(_sources(block.experts), device)
        reference = _layer(
            block, reference_shard, ResidentEPMXFP4MoELayer
        ).forward(x, block.gate)

        shared = block.shared_experts(x)
        # Use a production-scale collective payload for the all-reduce versus
        # reduce-scatter comparison. At the tiny 64-row routed-kernel shape,
        # NCCL selects a different reduction algorithm for the two APIs and
        # measures that topology seam rather than the exact64 prefill path.
        shared_rows = 65_536
        shared_x = x.repeat(shared_rows // num_rows, 1)
        shared_world1 = block.shared_experts(shared_x)
        shared_tp = copy.deepcopy(block.shared_experts)
        for name in ("gate_proj", "up_proj", "down_proj"):
            projection = getattr(shared_tp, name)
            projection.weight = torch.nn.Parameter(
                shard_shared_expert_tensor(
                    projection.weight.detach(),
                    name + ".weight",
                    world,
                    rank,
                ),
                requires_grad=False,
            )
        shared_tp.intermediate_size = shared_tp.gate_proj.weight.shape[0]
        shared_tp.gate_proj.out_features = shared_tp.intermediate_size
        shared_tp.up_proj.out_features = shared_tp.intermediate_size
        shared_tp.down_proj.in_features = shared_tp.intermediate_size
        shared_local_reference = shared_tp._ffn(shared_x)
        shared_tp_reference = shared_local_reference.clone()
        dist.all_reduce(shared_tp_reference, group=dist.group.WORLD)
        shared_tp_vs_world1 = _err_ratio(
            shared_tp_reference, shared_world1
        )
        shared_x_local = scatter_rows(shared_x, world, rank)
        shared_input = all_gather_rows(
            shared_x_local,
            shared_rows,
            world,
            rank,
            dist.group.WORLD,
        )
        shared_input_exact = torch.equal(shared_input, shared_x)
        shared_partial = shared_tp._ffn_into(
            shared_input, shared_input
        )
        shared_local_exact = torch.equal(
            shared_partial, shared_local_reference
        )
        shared_sharded = reduce_scatter_rows(
            shared_partial, world, rank, dist.group.WORLD
        )
        shared_sharded_repeat = reduce_scatter_rows(
            shared_partial.clone(), world, rank, dist.group.WORLD
        )
        shared_reference_local = scatter_rows(
            shared_tp_reference, world, rank
        )
        shared_metrics = _bf16_gemm_metrics(
            shared_sharded, shared_reference_local
        )
        shared_noise = _bf16_gemm_metrics(
            shared_sharded_repeat, shared_sharded
        )["rel_l2"]
        routed_full = shared_x.mul(0.125)
        routed_local = scatter_rows(routed_full, world, rank)
        combined_sharded = shared_sharded + routed_local
        combined_sharded_error = _err_ratio(
            combined_sharded,
            scatter_rows(shared_tp_reference + routed_full, world, rank),
        )
        combined_reference = shared + wide
        combined = x.clone()
        block.shared_experts.forward_into(combined, combined)
        shared_alias_error = _err_ratio(combined, shared)
        all_gather_rows_add_(
            combined,
            wide_local,
            num_rows,
            world,
            rank,
            dist.group.WORLD,
            chunk_rows=2,
        )
        combined_error = _err_ratio(combined, combined_reference)

        uneven_rows = world * 3 + 1
        uneven = torch.arange(
            uneven_rows * 7, dtype=torch.float32, device=device
        ).view(uneven_rows, 7)
        uneven_local = scatter_rows(uneven, world, rank)
        uneven_output = torch.ones_like(uneven)
        all_gather_rows_add_(
            uneven_output,
            uneven_local,
            uneven_rows,
            world,
            rank,
            dist.group.WORLD,
            chunk_rows=2,
        )
        uneven_exact = torch.equal(uneven_output, uneven + 1)
        uneven_copy = torch.empty_like(uneven)
        all_gather_rows_into(
            uneven_copy,
            uneven_local,
            uneven_rows,
            world,
            rank,
            dist.group.WORLD,
            chunk_rows=2,
        )
        uneven_copy_exact = torch.equal(uneven_copy, uneven)
        uneven_partial = uneven + rank * 1000
        uneven_sum = uneven_partial.clone()
        dist.all_reduce(uneven_sum, group=dist.group.WORLD)
        uneven_reduced_local = reduce_scatter_rows(
            uneven_partial,
            world,
            rank,
            dist.group.WORLD,
        )
        uneven_reduce_scatter_exact = torch.equal(
            uneven_reduced_local,
            scatter_rows(uneven_sum, world, rank),
        )

        # The K3 block-attention residual mixer is token-independent. The
        # streamed-SP8 strategy computes one contiguous eighth per rank and
        # restores the replicated rows for the following TP attention/MLP.
        depth_gen = torch.Generator().manual_seed(260902)
        # Use K3's production H so both the replicated reference and every
        # token shard exercise the Triton mixer rather than its shape fallback.
        depth_tokens, depth_blocks, depth_hidden = 64, 5, 7168
        depth_prefix = torch.randn(
            depth_tokens,
            depth_hidden,
            generator=depth_gen,
            dtype=torch.bfloat16,
        ).to(device)
        depth_residual = torch.randn(
            depth_tokens,
            depth_blocks,
            depth_hidden,
            generator=depth_gen,
            dtype=torch.bfloat16,
        ).to(device)
        depth_proj = torch.nn.Linear(
            depth_hidden, 1, bias=False, dtype=torch.float32
        ).to(device)
        depth_proj.weight.copy_(torch.randn(
            1, depth_hidden, generator=depth_gen, dtype=torch.float32
        ).to(device))
        depth_norm = SimpleNamespace(
            weight=torch.randn(
                depth_hidden, generator=depth_gen, dtype=torch.float32
            ).to(device),
            variance_epsilon=1e-6,
        )
        depth_reference = apply_attn_res(
            depth_prefix,
            depth_residual,
            depth_proj,
            depth_norm,
            chunk_size=depth_tokens // world,
        )
        depth_norm._streamed_sp8_row_group = (
            world,
            rank,
            dist.group.WORLD,
        )
        depth_actual = apply_attn_res(
            depth_prefix,
            depth_residual,
            depth_proj,
            depth_norm,
            chunk_size=depth_tokens // world,
        )
        depth_mix_exact = torch.equal(depth_actual, depth_reference)
        depth_mix_max_abs = float(
            (depth_actual.float() - depth_reference.float()).abs().max()
        )
        depth_norm._streamed_sp8_keep_sharded = True
        depth_local = apply_attn_res(
            scatter_rows(depth_prefix, world, rank),
            depth_residual,
            depth_proj,
            depth_norm,
            chunk_size=depth_tokens // world,
        )
        depth_local_exact = torch.equal(
            depth_local, scatter_rows(depth_reference, world, rank)
        )
        del depth_norm._streamed_sp8_keep_sharded
        depth_regathered = apply_attn_res(
            depth_local,
            depth_residual,
            depth_proj,
            depth_norm,
            chunk_size=depth_tokens // world,
        )
        depth_regather_exact = torch.equal(
            depth_regathered, depth_reference
        )

    wide_error = _err_ratio(wide, reference)
    striped_error = _err_ratio(striped, reference)
    striped_vs_wide = _err_ratio(striped, wide)
    max_abs = float((striped.float() - reference.float()).abs().max())
    gate_out = block.gate(x_local.view(x_local.shape[0], 1, hidden))
    local_assignments = torch.tensor(
        [gate_out[0].numel()], dtype=torch.int64, device=device
    )
    dist.all_reduce(local_assignments)
    expected_assignments = num_rows * gate_out[0].shape[-1]

    if wide_error >= 3e-3 or striped_error >= 3e-3:
        raise AssertionError(
            "SP8 row parity failed: "
            f"wide={wide_error} striped={striped_error} max_abs={max_abs}"
        )
    if striped_vs_wide >= 3e-3:
        raise AssertionError(
            f"striped/wide parity failed: err_ratio={striped_vs_wide}"
        )
    if shared_alias_error != 0.0 or combined_error >= 3e-3:
        raise AssertionError(
            "bounded output assembly parity failed: "
            f"shared_alias={shared_alias_error} combined={combined_error}"
        )
    if not shared_input_exact or not shared_local_exact:
        raise AssertionError(
            "shared-expert sharded input/local FFN parity failed: "
            f"input={shared_input_exact} local={shared_local_exact}"
        )
    if (
        shared_noise > 1.25e-3
        or shared_metrics["rel_l2"] > 5e-3
        or shared_metrics["cosine"] < 0.9999
        or shared_metrics["max_abs_over_std"] > 3e-2
        or combined_sharded_error >= 3e-3
    ):
        raise AssertionError(
            "sharded shared/routed merge parity failed: "
            f"shared={shared_metrics} noise={shared_noise} "
            f"combined={combined_sharded_error}"
        )
    if not uneven_exact:
        raise AssertionError("uneven chunked row gather-add parity failed")
    if not uneven_copy_exact:
        raise AssertionError("uneven chunked row gather-copy parity failed")
    if not uneven_reduce_scatter_exact:
        raise AssertionError("uneven row reduce-scatter parity failed")
    if not depth_mix_exact:
        raise AssertionError(
            "row-sharded depth-mix parity failed: "
            f"max_abs={depth_mix_max_abs}"
        )
    if not depth_local_exact or not depth_regather_exact:
        raise AssertionError(
            "sharded depth carry parity failed: "
            f"local={depth_local_exact} regather={depth_regather_exact}"
        )
    if int(local_assignments.item()) != expected_assignments:
        raise AssertionError(
            f"duplicated assignments: got {int(local_assignments.item())}, "
            f"expected {expected_assignments}"
        )

    if rank == 0:
        result = {
            "world_size": world,
            "num_rows": num_rows,
            "rows_per_rank": num_rows // world,
            "num_experts": len(block.experts),
            "experts_per_ingress_rank": experts_per_rank,
            "top_k": gate_out[0].shape[-1],
            "routed_assignments": int(local_assignments.item()),
            "expected_assignments": expected_assignments,
            "wide_err_ratio_vs_world1": wide_error,
            "striped_err_ratio_vs_world1": striped_error,
            "striped_err_ratio_vs_wide": striped_vs_wide,
            "max_abs_vs_world1": max_abs,
            "shared_forward_into_err_ratio": shared_alias_error,
            "shared_sharded_metrics": shared_metrics,
            "shared_sharded_noise_rel_l2": shared_noise,
            "shared_tp_err_ratio_vs_world1": shared_tp_vs_world1,
            "shared_collective_rows": shared_rows,
            "shared_input_exact": shared_input_exact,
            "shared_local_exact": shared_local_exact,
            "combined_sharded_err_ratio": combined_sharded_error,
            "bounded_gather_add_err_ratio": combined_error,
            "uneven_gather_add_exact": uneven_exact,
            "uneven_gather_copy_exact": uneven_copy_exact,
            "uneven_reduce_scatter_exact": uneven_reduce_scatter_exact,
            "depth_mix_exact": depth_mix_exact,
            "depth_mix_max_abs": depth_mix_max_abs,
            "depth_local_exact": depth_local_exact,
            "depth_regather_exact": depth_regather_exact,
            "verdict": "PASS",
        }
        Path(out_path).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)

    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--port", type=int, default=29617)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
        raise RuntimeError(
            f"world-8 SP8 parity needs 8 GPUs; saw {torch.cuda.device_count()}"
        )
    mp.spawn(worker, args=(8, args.out, args.port), nprocs=8, join=True)


if __name__ == "__main__":
    main()
