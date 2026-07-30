# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #

"""Resident-EP decode MoE seam: stacked BF16 shards + fused grouped GEMM +
cross-rank combine (Kimi-Linear M4 P0.3).

DECODE-ONLY companion to the streamed pure-DP MoE path. Each rank materializes
the stacked BF16 weights of its EP shard (num_experts / world_size experts per
MoE layer) ONCE at configure_decoding — from the same host copy-engine tensors
the streamed path consumes (core_engine.get_tensor) — and then computes every
decode step entirely from HBM (no per-step H2D):

    pad local rows to the synced per-rank layout -> comm.all_gather ->
    router on the gathered GLOBAL tokens -> fused_moe_bf16 over the LOCAL
    shard with non-local expert slots weight-masked -> comm.all_reduce(SUM)
    over the global layout -> extract the local slice.

Cross-rank layout contract (mirrors KimiK25MoE._forward_decode):
  - ``num_tokens_per_rank`` (ntp) is the per-step MAX decode rows over all
    ranks, synced by the worker's ``_sync_decode_moe_rank_counts()`` collective
    and delivered via ``PSM.set_num_tokens_per_rank()`` BEFORE the forward.
    Every rank — including an EMPTY one — therefore knows the global buffer
    shape ``(world_size * ntp, H)`` from the synced scalar alone; no extra
    communication is needed to size the collectives.
  - An empty rank (0 local decode rows) skips NOTHING except the final local
    slice: it contributes zero-padded rows to the all_gather and MUST run the
    router + kernel + all_reduce, because its resident experts serve the OTHER
    ranks' tokens. Zero-padded rows are harmless — a bias-free expert MLP maps
    x = 0 to exactly 0 (w1·0 = 0, silu(0)·0 = 0, w2·0 = 0), so padding adds
    nothing to the reduced sum and the padded slice is discarded on extract.
  - Decode-only safety: all ranks always step decode together (worker :9746
    no-skip invariant), so the collectives cannot deadlock. Never call this
    from prefill — prefill ranks may run different module sequences.
"""

import logging
import time

import torch
from torch.distributed import ReduceOp

from batchgen_kernels.triton.fused_moe_bf16 import fused_moe_bf16


def build_layer_shard(get_tensor, layer_idx, expert_start, num_local,
                      hidden_size, intermediate_size, device,
                      dtype=torch.bfloat16):
    """Materialize one MoE layer's local expert shard as stacked GPU tensors.

    Copies each local expert's BF16 weights host->GPU exactly once, straight
    from the copy-engine host source (``core_engine.get_tensor``, keys
    ``routed_expert_{layer}_{expert}`` -> {"w1.weight", "w2.weight",
    "w3.weight"}; w1 = gate, w3 = up, w2 = down).

    HBM budget (per rank, H20 96 GB, server --gpu-memory-frac 0.90 -> 86.4 GB
    ceiling; num_local = 256/8 = 32, 26 MoE layers, H = 2304, I = 1024):
      - resident EP shards: 26 x 32 x (2*1024*2304 + 2304*1024) el x 2 B
                            = 26 x 453.5 MiB ~= 11.8 GB
      - KDA state pools:    kda_state_slots(256) x 20 layers x
                            (n_h*d_h^2 x 4 B fp32 recurrent
                             + 3 x 4096 x (W-1) x 2 B conv)
                            ~= 5.5 GB (d_h = 64) - 11.1 GB (d_h = 128 worst
                            case); scales linearly with kda_state_slots
      - resident non-expert (attn/KDA/shared/skeleton/lm_head): ~4 GB
      -> ~21-27 GB static; leaves > 55 GB under the 86.4 GB ceiling for the
      paged-KV pool, staging buffers, NCCL and activations — fits.

    Returns:
        w13: (num_local, 2*I, H) stacked [gate; up] weights (gate first —
             the packing fused_moe_bf16 expects).
        w2:  (num_local, H, I) stacked down-projection weights.
    """
    I, H = intermediate_size, hidden_size
    w13 = torch.empty((num_local, 2 * I, H), dtype=dtype, device=device)
    w2 = torch.empty((num_local, H, I), dtype=dtype, device=device)
    for i in range(num_local):
        tensors = get_tensor(f"routed_expert_{layer_idx}_{expert_start + i}")
        w13[i, :I].copy_(tensors["w1.weight"])
        w13[i, I:].copy_(tensors["w3.weight"])
        w2[i].copy_(tensors["w2.weight"])
    return w13, w2


class ResidentEPMoELayer:
    """Per-layer resident-EP decode MoE forward (thin: kernel + collectives).

    Holds the layer's stacked shard and the NCCL communicator; the router
    (KimiMoEGate) stays owned by the model and is passed into ``forward`` so
    it runs — unchanged — on the gathered global tokens (K2.5 pattern; gate
    weights are DP-replicated, so all ranks compute identical routing).
    """

    # Per-step padded rows per rank; synced across ranks by the worker
    # (dist.all_gather_into_tensor over local batch sizes) and written here
    # through PSM.set_num_tokens_per_rank before every decode forward.
    num_tokens_per_rank = None

    def __init__(self, layer_idx, w13, w2, comm, world_size, rank,
                 expert_start):
        self.layer_idx = layer_idx
        self.w13 = w13
        self.w2 = w2
        self.comm = comm
        self.world_size = world_size
        self.rank = rank
        self.expert_start = expert_start
        self.num_local_experts = w13.shape[0]

    @classmethod
    def set_num_tokens_per_rank(cls, num_tokens_per_rank):
        cls.num_tokens_per_rank = int(num_tokens_per_rank)

    def forward(self, x_local, gate):
        """Resident-EP decode MoE over DP-sharded tokens.

        Args:
            x_local: (num_local_tokens, H) this rank's decode rows; may be
                (0, H) on an empty rank — the collectives still run.
            gate: router module; called on the gathered (num_global, 1, H)
                tokens, returns (topk_idx, topk_weight[, ...]).

        Returns:
            (num_local_tokens, H) summed routed-expert output for the local
            rows (all 256 experts, combined across ranks).
        """
        ntp = ResidentEPMoELayer.num_tokens_per_rank
        num_tokens, H = x_local.shape
        assert ntp is not None and ntp > 0, (
            "resident-EP decode requires the worker rank-count sync "
            "(PSM.set_num_tokens_per_rank) before the MoE forward"
        )
        assert num_tokens <= ntp, (
            f"local decode rows {num_tokens} exceed synced ntp {ntp}"
        )
        num_global = self.world_size * ntp

        # Fixed per-rank layout: zero-pad local rows to ntp, gather globally.
        padded = x_local.new_zeros((ntp, H))
        if num_tokens > 0:
            padded[:num_tokens].copy_(x_local)
        all_tokens = x_local.new_empty((num_global, H))
        with self.comm.change_state(enable=True):
            self.comm.all_gather(all_tokens, padded)

        # Router on the global tokens (identical on every rank).
        gate_out = gate(all_tokens.view(num_global, 1, H))
        topk_idx, topk_weight = gate_out[0], gate_out[1]

        # Mask routing to the local shard: non-local slots keep weight 0 and
        # are pointed at local expert 0 — they contribute exactly 0 through
        # the weighted top-k reduction, and the extra rows cost no additional
        # weight traffic in the BW-bound decode regime (same experts read).
        local_ids = topk_idx - self.expert_start
        local_mask = (local_ids >= 0) & (local_ids < self.num_local_experts)
        local_ids = torch.where(local_mask, local_ids,
                                torch.zeros_like(local_ids))
        masked_weight = topk_weight * local_mask.to(topk_weight.dtype)

        partial = fused_moe_bf16(all_tokens, self.w13, self.w2,
                                 masked_weight, local_ids)

        # Combine expert shards: SUM over ranks in the global layout (BF16
        # over NCCL, K2.5-class numerics; per-rank top-k accumulation above
        # is fp32 inside moe_weighted_sum).
        with self.comm.change_state(enable=True):
            self.comm.all_reduce(partial, op=ReduceOp.SUM)

        start = self.rank * ntp
        return partial[start:start + num_tokens]


def build_resident_ep_layers(model_layers, get_tensor, comm, world_size, rank,
                             expert_start, num_local, intermediate_size,
                             device):
    """Materialize shards for every MoE layer and attach a ResidentEPMoELayer
    to each ``block_sparse_moe`` as ``_resident_ep_moe`` (consumed by
    ``moe_forward_serving``'s decode dispatch). Returns total bytes resident.
    """
    start_t = time.perf_counter()
    total_bytes = 0
    num_layers = 0
    for layer_idx, layer in enumerate(model_layers):
        moe = getattr(layer, "block_sparse_moe", None)
        if moe is None or moe.experts is None:
            continue
        w13, w2 = build_layer_shard(
            get_tensor, layer_idx, expert_start, num_local,
            moe.moe_hidden_size, intermediate_size, device,
        )
        moe._resident_ep_moe = ResidentEPMoELayer(
            layer_idx, w13, w2, comm, world_size, rank, expert_start,
        )
        total_bytes += (w13.numel() * w13.element_size()
                        + w2.numel() * w2.element_size())
        num_layers += 1
    logging.info(
        f"Rank {rank}: resident EP shards materialized — {num_layers} MoE "
        f"layers x {num_local} experts, {total_bytes / (1024**3):.2f} GiB "
        f"({time.perf_counter() - start_t:.1f}s, one-time H2D)"
    )
    return total_bytes
