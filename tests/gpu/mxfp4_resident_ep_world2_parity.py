# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3                                                           #
#  copyright (c) EfficientMoE team 2025                                         #
#  Licensed under the Apache License, Version 2.0                              #
# ---------------------------------------------------------------------------- #
"""World=2 EP parity harness for the resident MXFP4-LatentMoE decode layer
(M3.1b, decisions A13/A16) — single node, 2 GPUs, NOT a pytest test.

Proves ``ResidentEPMXFP4MoELayer.forward``'s world>=2 EP path
(``_forward_ep``: all_gather the 3584-d LATENT, shard-experts on the global
tokens, all_reduce(SUM) the combined latent, slice the local rows) reproduces
the SINGLE-RANK oracle on rank 0's tokens. Two GPUs hold a 2-way split of the
64 synthetic experts (32/rank); the global decode batch is DP-sharded across
ranks. rank 0's local-token EP output is gated against three references on the
SAME tokens/weights:

  * world=1 resident (SAME marlin kernels)  -> expect ~0 (EP correctness)
  * streamed oracle expert path (block(x))  -> expect ~0 (kernel + repack id)
  * fp32 dequant oracle (M3.1a reference)   -> expect ~1.3e-3 (== M3.1a gate)

Reuses the M3.1a fixture + oracle verbatim from
``tests/gpu/test_kimi_linear_mxfp4_latent_moe_serving.py`` so the parity is
against the identical weights, router, and dequant reference the world=1 gate
uses. Deterministic name-keyed seeding builds a BIT-IDENTICAL block on every
rank, so each rank's DP-replicated down_proj/router/norm/up + its expert slice
are consistent without broadcasting weights.

Run ON h20-instance-2 (2 free GPUs):

    BATCHGEN_KERNELS_DEV=1 K3_MXFP4_GPU=1 CUDA_VISIBLE_DEVICES=0,1 \
    PYTHONPATH=<repo>:<fla-src> \
    python tests/gpu/mxfp4_resident_ep_world2_parity.py --out <results.json>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# tests/gpu on the path so the M3.1a test module (fixture + oracle) imports by
# name; importing it also puts tests/ on the path (kimi_k3_harness).
sys.path.insert(0, str(Path(__file__).resolve().parent))


class DistComm:
    """Thin torch.distributed shim exposing exactly the pynccl surface the
    resident layer calls (change_state / all_gather(out, in) / all_reduce).
    The production path uses batchgen's PyNcclCommunicator; the collective
    LAYOUT (rank r at rows [r*ntp:(r+1)*ntp]) is identical."""

    def change_state(self, enable=True, stream=None):
        return nullcontext()

    def all_gather(self, output_tensor, input_tensor, stream=None):
        dist.all_gather_into_tensor(output_tensor, input_tensor.contiguous())

    def all_reduce(self, tensor, op=dist.ReduceOp.SUM, stream=None):
        dist.all_reduce(tensor, op=op)


def _err_ratio(a, r):
    a = a.float()
    r = r.float()
    err = (a - r).abs()
    rms = float(r.pow(2).mean().sqrt())
    return float(err.pow(2).mean().sqrt() / (rms + 1e-8)), rms


# (cfg_name, T0 rows on rank 0, T1 rows on rank 1)
SCENARIOS = [
    ("syn25_mxfp4", 64, 48),   # both ranks non-empty, T1<T0 (padding on rank 1)
    ("skew10_mxfp4", 64, 48),  # latent != hidden/2 config
    ("syn25_mxfp4", 64, 0),    # empty rank 1 (still runs every collective)
]


def _build_layer(T, block, cfg, dev, world, rank, ResidentEPMXFP4MoELayer,
                 build_layer_shard):
    E = len(block.experts)
    assert E % world == 0, f"num_experts {E} not divisible by world {world}"
    experts_per_rank = E // world
    expert_start = rank * experts_per_rank
    local = block.experts[expert_start:expert_start + experts_per_rank]
    shard = build_layer_shard(
        [{"w1": (e.w1.weight_packed.data, e.w1.weight_scale.data),
          "w3": (e.w3.weight_packed.data, e.w3.weight_scale.data),
          "w2": (e.w2.weight_packed.data, e.w2.weight_scale.data)}
         for e in local],
        device=dev)
    layer = ResidentEPMXFP4MoELayer(
        layer_idx=0, shard=shard,
        down_proj=block.routed_expert_down_proj,
        norm=block.routed_expert_norm if block.latent_moe_use_norm else None,
        up_proj=block.routed_expert_up_proj,
        comm=DistComm(), world_size=world, rank=rank,
        expert_start=expert_start)
    return layer, expert_start, experts_per_rank


def worker(rank, world, out_path, master_port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    dev = f"cuda:{rank}"

    import test_kimi_linear_mxfp4_latent_moe_serving as TT
    from batchgen.moe.fused_moe_mxfp4_resident import (
        ResidentEPMXFP4MoELayer, build_layer_shard,
    )
    import kimi_k3_harness as H

    results = []
    for cfg_name, T0, T1 in SCENARIOS:
        T_local = T0 if rank == 0 else T1
        T_total = T0 + T1
        ntp = max(T0, T1)
        ResidentEPMXFP4MoELayer.set_num_tokens_per_rank(ntp)

        block, cfg, _ = TT._build_mxfp4_serving_block(cfg_name)
        hidden = cfg.hidden_size
        x_global = H.seeded_input(
            f"ep2:{cfg_name}:{T0}:{T1}", 1, T_total, hidden,
            dtype=torch.bfloat16).to(dev).reshape(T_total, hidden)
        # DP shard: rank 0 owns [0:T0], rank 1 owns [T0:T0+T1].
        x_local = (x_global[:T0] if rank == 0
                   else x_global[T0:T0 + T1]).contiguous()

        layer, e_start, e_per = _build_layer(
            T_local, block, cfg, dev, world, rank,
            ResidentEPMXFP4MoELayer, build_layer_shard)

        with torch.no_grad():
            ep_out = layer.forward(x_local, block.gate)   # [T_local, hidden]

        assert list(ep_out.shape) == [T_local, hidden], ep_out.shape
        assert torch.isfinite(ep_out).all(), f"{cfg_name} EP output non-finite"

        if rank == 0:
            with torch.no_grad():
                # ref A: world=1 resident (all experts, SAME kernels) on rank0 tokens
                w1_layer, _, _ = _build_layer(
                    T0, block, cfg, dev, 1, 0,
                    ResidentEPMXFP4MoELayer, build_layer_shard)
                ref_res = w1_layer.forward(x_local, block.gate)
                # ref B: streamed oracle expert path (block == streamed path)
                x3d = x_local.reshape(1, T0, hidden)
                streamed = block(x3d)
                streamed_ep = (streamed - block.shared_experts(x3d)
                               ).reshape(T0, hidden)
                # ref C: fp32 dequant oracle expert path (the M3.1a reference)
                _full, ref_expert_path, _idx, _w = \
                    TT._dequant_fp32_reference(block, x3d)
                ref_oracle = ref_expert_path.reshape(T0, hidden)

            er_res, _ = _err_ratio(ep_out, ref_res)
            er_streamed, _ = _err_ratio(ep_out, streamed_ep)
            er_oracle, rms = _err_ratio(ep_out, ref_oracle)
            rec = dict(cfg=cfg_name, T0=T0, T1=T1, ntp=ntp,
                       experts_per_rank=e_per, num_experts=len(block.experts),
                       err_vs_world1_resident=er_res,
                       err_vs_streamed=er_streamed,
                       err_vs_fp32_oracle=er_oracle, rms=rms)
            results.append(rec)
            print("[ep2] {:14s} T0={} T1={} ntp={} E/rank={} | "
                  "vs_world1={:.3e} vs_streamed={:.3e} vs_oracle={:.3e}".format(
                      cfg_name, T0, T1, ntp, e_per, er_res, er_streamed,
                      er_oracle), flush=True)
        dist.barrier()

    if rank == 0:
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[ep2] wrote {out_path}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=29591)
    args = ap.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError(
            "world=2 EP parity needs 2 GPUs; saw "
            f"{torch.cuda.device_count()} (CUDA_VISIBLE_DEVICES="
            f"{os.environ.get('CUDA_VISIBLE_DEVICES')})")
    mp.spawn(worker, args=(2, args.out, args.port), nprocs=2, join=True)
    print("[ep2] DONE")


if __name__ == "__main__":
    main()
