# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #

"""Resident-EP decode MoE seam for Kimi-K3 MXFP4 LatentMoE (M3.1a, decision A13).

DECODE-ONLY companion to the STREAMED MXFP4-LatentMoE path
(``kimi_linear/serving_modules.py::moe_forward_serving``). Each rank repacks the
MXFP4 weights of its EP shard into marlin tile order ONCE at load — the packed
E2M1 nibbles + E8M0 scales the streamed path repacks EVERY forward
(``k3/mxfp4_expert.py::K3MXFP4Projection.marlin`` -> ``repack_mxfp4_to_marlin_device``,
MEASURED ~1068 us/expert) — and then computes every decode step from HBM with no
per-step repack and (M3.1b) no per-step H2D.

Latent dataflow, EXACTLY the streamed oracle's op order (the resident layer
reproduces the expert PATH; the caller adds the DP-local shared expert):

    router(hidden)                         # topk on the PRE-down 7168 hidden
    x = routed_expert_down_proj(hidden)    # 7168 -> 3584, ONCE per token
    dispatch x(latent) -> [E*mtp, latent]  # dispatch_scatter_3d (any top_k)
    grouped S1 (w1 gate + w3 up + SiTU)    # marlin MXFP4, resident shard
    grouped S3 (w2 down)                    # marlin MXFP4, resident shard
    y = sum_k weight_k * expert_out_k       # FP32 accumulate -> bf16 (see below)
    y = routed_expert_norm(y)               # ONCE, post-combine, pre-up
    y = routed_expert_up_proj(y)            # 3584 -> 7168

Because the grouped S1/S3 kernel entries are the SAME ones the streamed
single-expert path drives at E=1, and both sides repack with the identical
``repack_mxfp4_to_marlin_device``, the resident marlin weights are bit-identical
to the streamed path's per-forward repack — so resident == streamed to within
combine-order noise, and resident gates against the M3.0 fp32-dequant oracle at
the same err_ratio the streamed path already clears.

THE FP32 COMBINE (A13's flagged risk — read this before touching it)
--------------------------------------------------------------------
The spec named ``dispatch_scatter_3d.reduce_weighted_scatter`` for the fp32
combine, but that kernel is ``template<int K>`` with cases ONLY for K in
{2, 4, 8} (``dispatch_scatter_3d.cu`` switch; ``default`` is
``TORCH_CHECK(false, "Unsupported K=", K)``). K3's ``num_experts_per_token`` is
16, so the named kernel HARD-FAILS on K3 — its only production caller (K2.5)
runs top_k=8. The combine here is therefore done with the STREAMED ORACLE'S OWN
arithmetic: the per-token top-k contributions are summed in FP32 and cast to
bf16 exactly where ``moe_forward_serving`` downcasts (``results`` fp32 ->
``.to(identity.dtype)``). This is not a different combine — it is bit-for-bit
the reference reduction — so it cannot introduce combine error; it merely does
not use the fused kernel. Enabling the fused kernel on K3 requires a
``case 16:`` in ``dispatch_scatter_3d.cu`` + a kernel rebuild (a kernel-PR
change, and relevant mainly to the CUDA-graph decode path where a data-indexed
``index_add`` is less capture-friendly than the fused scatter). NAMED FOLLOW-UP
for M3.1b; flagged to the planning loop.

WORLD>=2 EP (M3.1b, decisions A13/A16). ``world_size == 1`` keeps the M3.1a
path bit-for-bit (no collectives). ``world_size >= 2`` routes ``forward`` to
``_forward_ep``: router + down-proj on the LOCAL padded hidden (DP-replicated),
all_gather the 3584-d LATENT (NOT the 7168 hidden — halves comm, A13) plus the
per-token routing, run this rank's expert shard on the global tokens, fp32
combine, all_reduce(SUM) the combined LATENT across ranks, slice this rank's
rows, then routed_expert_norm + routed_expert_up_proj. See ``_forward_ep`` for
the ntp layout contract.
"""

import logging
import time

import torch
from torch.distributed import ReduceOp

from batchgen.moe.dispatch_scatter_3d import (
    dispatch_scatter_3d,
    reduce_weighted_scatter_fp32,
)
from batchgen.moe.marlin_grouped_moe import (
    marlin_grouped_m16_mxfp4,
    marlin_grouped_stage1_fused_mxfp4_situ,
)
from batchgen.moe.routing import dispatch_count_gather_cuda


def compact_prefill_chunk_rows(num_global, configured_rows):
    """Choose the bounded resident-prefill expert chunk size."""
    rows = int(configured_rows)
    if int(num_global) > 16384:
        # W2 carries ~0.98 GiB more unavoidable full-batch latent/y state than
        # W1. Live telemetry after reserving y and tiling RMSNorm left only
        # 88–96 MiB at the first expert chunk; the 512-row BF16 expert_out is
        # 98 MiB by itself. A 256-row chunk halves dispatched/intermediate/
        # expert_out and the FP32 combine temporaries without changing their
        # per-token arithmetic or the single final all-reduce.
        rows = min(rows, 256)
    if rows <= 0:
        raise ValueError("compact_chunk_rows must be positive")
    return rows


def compact_dispatch_route_stats_by_chunk(
    topk_idx_i32,
    expert_start,
    num_local_experts,
    chunk_rows,
):
    """Read packed-dispatch capacities and tile bounds in one host transfer.

    Grouped Marlin needs a host-static upper bound on rows per expert. Build
    exact per-chunk counts on the GPU, then transfer only the maximum and sum
    per chunk. This keeps the expert loop asynchronous instead of synchronizing
    once for every chunk.
    """
    rows = int(topk_idx_i32.shape[0])
    top_k = int(topk_idx_i32.shape[1])
    chunk_rows = int(chunk_rows)
    num_local_experts = int(num_local_experts)
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if num_local_experts <= 0:
        raise ValueError("num_local_experts must be positive")
    if rows == 0:
        return []

    num_chunks = (rows + chunk_rows - 1) // chunk_rows
    padded_rows = num_chunks * chunk_rows
    if padded_rows == rows:
        padded = topk_idx_i32
    else:
        padded = torch.full(
            (padded_rows, top_k),
            -1,
            dtype=topk_idx_i32.dtype,
            device=topk_idx_i32.device,
        )
        padded[:rows].copy_(topk_idx_i32)

    local = padded - int(expert_start)
    local.masked_fill_(
        (local < 0) | (local >= num_local_experts),
        num_local_experts,
    )
    bins_per_chunk = num_local_experts + 1
    chunk_base = (
        torch.arange(
            num_chunks,
            dtype=torch.int64,
            device=topk_idx_i32.device,
        )
        * bins_per_chunk
    ).view(num_chunks, 1, 1)
    bins = local.to(torch.int64).view(
        num_chunks, chunk_rows, top_k
    ) + chunk_base
    counts = torch.bincount(
        bins.reshape(-1),
        minlength=num_chunks * bins_per_chunk,
    ).view(num_chunks, bins_per_chunk)
    local_counts = counts[:, :num_local_experts]
    return torch.stack(
        (local_counts.amax(dim=1), local_counts.sum(dim=1)),
        dim=1,
    ).tolist()


class MXFP4LayerShard:
    """One MoE layer's local expert shard, repacked into marlin tile order once.

    Holds the resident (marlin_qw int32, marlin_s uint8 E8M0) tensors for w1/w3/w2 of
    every local expert (kept alive so their ``data_ptr()`` stays valid) plus the
    six per-expert int64 pointer arrays the grouped marlin kernels consume.
    """

    def __init__(self, num_local, N, K_latent, device):
        self.num_local = int(num_local)
        self.N = int(N)                  # moe_intermediate_size
        self.K_latent = int(K_latent)    # routed_expert_hidden_size (latent)
        self.device = device
        # resident weight tensors, indexed [local_expert] -> dict(proj -> (qw, s))
        self._tensors = []
        # pointer arrays, filled by build_layer_shard
        self.gate_B_ptrs = None
        self.gate_scales_ptrs = None
        self.up_B_ptrs = None
        self.up_scales_ptrs = None
        self.down_B_ptrs = None
        self.down_scales_ptrs = None

    def nbytes(self):
        total = 0
        for t in self._tensors:
            for qw, s in t.values():
                total += qw.numel() * qw.element_size()
                total += s.numel() * s.element_size()
        return total


def _stage_registration_boundary_packed(packed, scale):
    """Return ``packed``, or a pageable copy of it when it straddles the compact
    store's ``cudaHostRegister`` right edge.

    MEASURED (world16 hierarchical GDR; global rank 14 registers routed experts
    [672, 784)): the first resident decode tensor of expert 784,
    ``w1.weight_packed``, begins exactly at that right edge.
    ``tensor.is_pinned()`` is true because it reports on the START address, so
    Torch takes the pinned H2D path, but the rest of the tensor lies outside the
    registered range and the direct ``packed.to(cuda)`` raises
    ``cudaErrorInvalidValue``. The immediately following scale — the next bytes
    of the store — reports ``is_pinned()`` false, which is exactly what the
    boundary looks like: a pinned packed whose adjacent neighbour is already
    unregistered. A pageable ``clone()`` copies successfully and byte-identically
    (the fully pinned expert 783 before it, the fully pageable expert 785 after
    it, and expert 840 all direct-copy correctly).

    This runs once per projection during the startup repack, and the CPU /
    contiguous / exact-adjacency tests keep every other tensor on the direct
    path.
    """
    if packed.device.type != "cpu" or scale.device.type != "cpu":
        return packed
    if not packed.is_contiguous() or not scale.is_contiguous():
        return packed
    if not packed.is_pinned() or scale.is_pinned():
        return packed
    packed_end = packed.data_ptr() + packed.numel() * packed.element_size()
    if packed_end != scale.data_ptr():
        return packed
    return packed.clone()


def _repack_projection(packed, scale, device):
    """Repack one MXFP4 projection ([n_out, k_in//pack] uint8 + [n_out, k_in//gs]
    uint8 E8M0) into (marlin_qw int32, marlin_s uint8 E8M0). Shapes are inferred from
    ``packed`` so the same helper serves w1/w3 (n_out=N, k_in=K) and w2
    (n_out=K, k_in=N)."""
    from batchgen.models.moonshotai.kimi_linear.k3.mxfp4_expert import (
        repack_mxfp4_to_marlin_device,
    )
    from batchgen.models.moonshotai.kimi_linear.k3.mxfp4_layout import (
        MXFP4_PACK_FACTOR,
    )

    if packed.dtype == torch.int32:
        # Stored marlin (task #53 offline): packed IS marlin_qw int32; scale
        # IS the marlin-order uint8 E8M0 consumed directly by the K3 kernel.
        packed = _stage_registration_boundary_packed(packed, scale)
        return packed.to(device), scale.to(device)
    n_out = int(packed.shape[0])
    k_in = int(packed.shape[1]) * MXFP4_PACK_FACTOR
    return repack_mxfp4_to_marlin_device(
        packed.to(device), scale.to(device), k_in, n_out, scale_bf16=False
    )


def build_layer_shard(expert_sources, device):
    """Repack one layer's local experts into a resident marlin shard, ONCE.

    Args:
        expert_sources: list over local experts; each item is a dict
            ``{"w1": (packed, scale), "w3": (packed, scale),
               "w2": (packed, scale)}`` where ``packed`` is [n_out, k_in//2]
            uint8 (low nibble = even k index) and ``scale`` is [n_out, k_in//32]
            uint8 E8M0 — exactly what ``K3MXFP4Projection`` holds and what
            ``core_engine.get_tensor`` serves (``w{1,3,2}.weight_{packed,scale}``).
        device: CUDA device for the resident tensors.

    Returns:
        MXFP4LayerShard with the six int64 pointer arrays populated.
    """
    num_local = len(expert_sources)
    if num_local == 0:
        raise ValueError("build_layer_shard: no local experts")
    # Infer N (=w1.n_out) and K_latent (=w1.k_in) from expert 0's gate projection.
    from batchgen.models.moonshotai.kimi_linear.k3.mxfp4_layout import (
        MXFP4_PACK_FACTOR,
    )
    w1_packed0 = expert_sources[0]["w1"][0]
    if w1_packed0.dtype == torch.int32:
        # Stored marlin: w1 marlin_qw is [K_latent // _MARLIN_TILE, N * 2].
        from batchgen.models.moonshotai.kimi_linear.k3.mxfp4_expert import (
            _MARLIN_TILE,
        )
        K_latent = int(w1_packed0.shape[0]) * _MARLIN_TILE
        N = int(w1_packed0.shape[1]) // 2
    else:
        N = int(w1_packed0.shape[0])
        K_latent = int(w1_packed0.shape[1]) * MXFP4_PACK_FACTOR

    shard = MXFP4LayerShard(num_local, N, K_latent, device)
    gate_B, gate_s, up_B, up_s, down_B, down_s = ([] for _ in range(6))
    for src in expert_sources:
        w1_qw, w1_s = _repack_projection(src["w1"][0], src["w1"][1], device)  # gate
        w3_qw, w3_s = _repack_projection(src["w3"][0], src["w3"][1], device)  # up
        w2_qw, w2_s = _repack_projection(src["w2"][0], src["w2"][1], device)  # down
        shard._tensors.append(
            {"w1": (w1_qw, w1_s), "w3": (w3_qw, w3_s), "w2": (w2_qw, w2_s)}
        )
        gate_B.append(w1_qw.data_ptr()); gate_s.append(w1_s.data_ptr())
        up_B.append(w3_qw.data_ptr()); up_s.append(w3_s.data_ptr())
        down_B.append(w2_qw.data_ptr()); down_s.append(w2_s.data_ptr())

    def _ptrs(vals):
        return torch.tensor(vals, dtype=torch.int64, device=device)

    shard.gate_B_ptrs = _ptrs(gate_B)
    shard.gate_scales_ptrs = _ptrs(gate_s)
    shard.up_B_ptrs = _ptrs(up_B)
    shard.up_scales_ptrs = _ptrs(up_s)
    shard.down_B_ptrs = _ptrs(down_B)
    shard.down_scales_ptrs = _ptrs(down_s)
    return shard


class ResidentEPMXFP4MoELayer:
    """Per-layer resident-EP decode MoE forward for K3 MXFP4 LatentMoE.

    Holds the layer's repacked marlin shard and references to the block's own
    (BF16, resident) latent seam modules ``routed_expert_down_proj`` /
    ``routed_expert_norm`` / ``routed_expert_up_proj``; the router is passed into
    ``forward`` (K2.5 pattern). ``forward`` returns the routed expert PATH; the
    caller adds the shared expert (mirrors the BF16
    ``moe_forward_resident_ep_decode`` seam).
    """

    # Per-step padded rows per rank; set by the worker for the M3.1b EP layout.
    # Unused at world=1 (all_gather / all_reduce are identity).
    num_tokens_per_rank = None
    # Set alongside ``num_tokens_per_rank`` at the worker's rank-count
    # synchronization boundary.  A K3 CUDA-graph MoE replay is collective
    # global, so it is enabled only when every EP rank has a live row; an empty
    # rank remains on the eager path and all ranks must make the same choice.
    decode_all_ranks_nonempty = False
    # Reused by every layer during resident prefill. W2 needs 896 MiB for this
    # exact FP32 global output; reserving it before the resident expert shards
    # prevents the late first-layer allocation from failing in a fragmented
    # allocator arena.
    _prefill_y = None

    def __init__(self, layer_idx, shard, down_proj, norm, up_proj,
                 comm=None, world_size=1, rank=0, expert_start=0):
        self.layer_idx = layer_idx
        self.shard = shard
        self.down_proj = down_proj
        self.norm = norm                 # may be None (latent_moe_use_norm False)
        self.up_proj = up_proj
        self.comm = comm
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.expert_start = int(expert_start)
        # Decode keeps the original graph-stable worst-case stride. R5 prefill
        # opts into exact route-count sizing because num_global can be 65K+
        # tokens and reserving that capacity for every local expert would OOM.
        self.compact_dispatch = False
        self.compact_chunk_rows = 2048

    @classmethod
    def set_num_tokens_per_rank(cls, num_tokens_per_rank):
        cls.num_tokens_per_rank = int(num_tokens_per_rank)

    @classmethod
    def set_rank_token_counts(cls, rank_token_counts):
        """Publish the boundary-synchronized empty-rank decision.

        ``rank_token_counts`` is a GPU tensor already produced by the worker's
        decode all-gather.  This is intentionally the only host read here; the
        graph path reuses the resulting Python boolean on every steady-state
        token rather than calling ``.item()`` from the model forward.
        """
        cls.decode_all_ranks_nonempty = not bool(
            (rank_token_counts == 0).any().item()
        )

    @classmethod
    def prepare_prefill_output(cls, num_global, hidden_size, device):
        cls._prefill_y = None
        cls._prefill_y = torch.empty(
            (int(num_global), int(hidden_size)),
            dtype=torch.float32,
            device=device,
        )
        return cls._prefill_y

    @classmethod
    def release_prefill_output(cls):
        cls._prefill_y = None

    def _combine_fp32(self, expert_out, topk_pos, topk_weight, T, H, K):
        """Weighted top-k combine in FP32 — the streamed oracle's exact reduction
        (``results`` fp32; the caller casts to bf16 exactly where
        ``moe_forward_serving`` downcasts). Substitutes for
        ``reduce_weighted_scatter``, which is K in {2,4,8} only (K3 is 16).

        Returns FP32 (not bf16) so the EP path can ``all_reduce`` the per-rank
        partial sums BEFORE the single bf16 downcast — summing disjoint expert
        subsets in fp32 then adding across ranks equals the world=1 fp32 sum to
        bf16-ULP. The world=1 caller downcasts immediately (M3.1a bit-identical).

        ``topk_pos`` [T*K] int32: absolute row of this (token, k) assignment in
        the strided expert_out buffer, or -1 (non-local / padding).
        """
        if self.compact_dispatch and K == 16:
            weight = topk_weight.reshape(T, K).float().contiguous()
            combined = torch.empty(
                (T, H), dtype=torch.float32, device=expert_out.device
            )
            return reduce_weighted_scatter_fp32(
                expert_out,
                topk_pos,
                weight,
                T,
                H,
                K,
                combined,
            )
        pos = topk_pos.view(T, K).long()
        weight = topk_weight.reshape(T, K).float()
        if self.compact_dispatch:
            # R5 prefill can have 65K+ global rows. Materializing
            # [T, K, H] twice (gather + contribution) would require ~28 GiB
            # at W2. Chunk only the independent token dimension; each chunk
            # keeps the original [rows, K, H] FP32 reduction unchanged, so
            # arithmetic order over top-k is identical to the decode path.
            combined = torch.empty(
                (T, H), dtype=torch.float32, device=expert_out.device
            )
            chunk_rows = 2048
            for start in range(0, T, chunk_rows):
                end = min(start + chunk_rows, T)
                chunk_pos = pos[start:end]
                valid = chunk_pos >= 0
                rows = chunk_pos.clamp(min=0).reshape(-1)
                gathered = expert_out.index_select(0, rows).float().view(
                    end - start, K, H
                )
                gathered.masked_fill_(
                    ~valid.unsqueeze(-1),
                    0.0,
                )
                w = weight[start:end].unsqueeze(-1)
                contribution = gathered * w
                combined[start:end].copy_(contribution.sum(dim=1))
            return combined
        valid = (pos >= 0)
        rows = pos.clamp(min=0).reshape(-1)
        gathered = expert_out.index_select(0, rows).float().view(T, K, H)
        # ``-1`` marks a route owned by another EP rank.  Its clamped row 0
        # can be unwritten (and therefore NaN/Inf); multiplying that value by
        # a zero validity mask would still propagate NaN.  Zero invalid lanes
        # before arithmetic, matching the compact-prefill branch above.
        gathered.masked_fill_(~valid.unsqueeze(-1), 0.0)
        w = weight.unsqueeze(-1)
        contrib = gathered * w
        return contrib.sum(dim=1)

    def _expert_path(
        self,
        x_latent,
        topk_idx_i32,
        num_rows,
        dispatch_capacity=None,
        packed_capacity=None,
        packed_max_rows=None,
    ):
        """Dispatch ``num_rows`` latent rows into this rank's local expert shard
        and run grouped marlin S1(gate+up+SiTU) -> S3(down); returns
        (expert_out [E*mtp, K_latent] bf16, topk_pos [num_rows*K] int32).

        The SAME kernel entries and strided buffer layout the M3.1a world=1 path
        used — factored so the EP path (``num_rows`` = world*ntp global tokens)
        drives bit-identical arithmetic. ``dispatch_scatter_3d`` masks non-owned
        experts to topk_pos = -1 via ``self.expert_start`` (world=1: start 0, all
        experts local; EP: this rank's shard only).
        """
        device = x_latent.device
        shard = self.shard
        E, N, K_latent = shard.num_local, shard.N, shard.K_latent
        K = topk_idx_i32.shape[-1]

        # --- dispatch latent rows into the expert-ordered activation buffer ---
        if self.compact_dispatch and dispatch_capacity is None:
            if packed_capacity is None or packed_max_rows is None:
                if packed_capacity is not None or packed_max_rows is not None:
                    raise ValueError(
                        "packed_capacity and packed_max_rows must be supplied "
                        "together"
                    )
                planned = compact_dispatch_route_stats_by_chunk(
                    topk_idx_i32,
                    self.expert_start,
                    E,
                    max(num_rows, 1),
                )
                packed_max_rows, packed_capacity = (
                    planned[0] if planned else (0, 0)
                )
            packed_capacity = int(packed_capacity)
            packed_max_rows = int(packed_max_rows)
            if packed_capacity < 0 or packed_capacity > num_rows * K:
                raise ValueError(
                    f"packed_capacity={packed_capacity} is outside "
                    f"[0, {num_rows * K}]"
                )
            if packed_max_rows < 0 or packed_max_rows > num_rows:
                raise ValueError(
                    f"packed_max_rows={packed_max_rows} is outside "
                    f"[0, {num_rows}]"
                )
            capacity = max(packed_capacity, 1)
            dispatched = torch.empty(
                capacity,
                K_latent,
                dtype=torch.bfloat16,
                device=device,
            )
            expert_counts = torch.empty(E, dtype=torch.int32, device=device)
            expert_offsets = torch.empty(
                E + 1, dtype=torch.int32, device=device
            )
            expert_counters = torch.empty(
                E, dtype=torch.int32, device=device
            )
            topk_pos = torch.empty(
                num_rows * K, dtype=torch.int32, device=device
            )
            dispatched, expert_counts, expert_offsets, topk_pos = (
                dispatch_count_gather_cuda(
                    x_latent,
                    topk_idx_i32,
                    self.expert_start,
                    E,
                    expert_counts=expert_counts,
                    expert_offsets=expert_offsets,
                    expert_counters=expert_counters,
                    dispatched_x=dispatched,
                    topk_pos=topk_pos,
                )
            )
            expert_starts = expert_offsets[:-1]
            max_m_tiles = max((packed_max_rows + 15) // 16, 1)
            # The wrapper uses ``mtp`` only to prove max_m_tiles covers every
            # expert. Packed storage has no shared stride, so its admissible
            # per-expert bound is the precomputed exact maximum.
            mtp = max(packed_max_rows, 1)
        elif dispatch_capacity is not None:
            if not self.compact_dispatch:
                raise ValueError(
                    "dispatch_capacity is only valid for compact dispatch"
                )
            if dispatch_capacity < num_rows:
                raise ValueError(
                    "dispatch_capacity cannot be smaller than num_rows"
                )
            mtp = max(((dispatch_capacity + 15) // 16) * 16, 16)
        else:
            mtp = max(((num_rows + 15) // 16) * 16, 16)
        if not (self.compact_dispatch and dispatch_capacity is None):
            dispatched = torch.zeros(
                E * mtp, K_latent, dtype=torch.bfloat16, device=device
            )
            expert_counts = torch.zeros(E, dtype=torch.int32, device=device)
            expert_counters = torch.zeros(E, dtype=torch.int32, device=device)
            topk_pos = torch.full(
                (num_rows * K,), -1, dtype=torch.int32, device=device
            )
            expert_counts, topk_pos = dispatch_scatter_3d(
                x_latent,
                topk_idx_i32,
                dispatched,
                self.expert_start,
                E,
                mtp,
                expert_counts,
                expert_counters,
                topk_pos,
            )
            expert_starts = (
                torch.arange(E, dtype=torch.int32, device=device) * mtp
            )
            max_m_tiles = (min(num_rows, mtp) + 15) // 16

        # --- grouped S1: gate(w1) + up(w3) + SiTU -> intermediate [E*mtp, N] ---
        activation_rows = dispatched.shape[0]
        intermediate = torch.empty(
            activation_rows, N, dtype=torch.bfloat16, device=device
        )
        row_N = N * intermediate.element_size()
        if self.compact_dispatch and dispatch_capacity is None:
            s1_C_ptrs = expert_starts.to(torch.int64).mul_(row_N).add_(
                intermediate.data_ptr()
            )
        else:
            s1_C_ptrs = torch.tensor(
                [intermediate.data_ptr() + e * mtp * row_N for e in range(E)],
                dtype=torch.int64,
                device=device,
            )
        s1_ws = torch.zeros(E * (N // 256 + 17), dtype=torch.int32, device=device)
        marlin_grouped_stage1_fused_mxfp4_situ(
            dispatched, intermediate, expert_counts, expert_starts,
            shard.gate_B_ptrs, shard.gate_scales_ptrs,
            shard.up_B_ptrs, shard.up_scales_ptrs, s1_C_ptrs,
            N, K_latent, s1_ws, max_m_tiles, mtp, E, num_rows,
        )

        # --- grouped S3: down(w2) -> expert_out [E*mtp, K_latent] ---
        expert_out = torch.empty(
            activation_rows, K_latent, dtype=torch.bfloat16, device=device
        )
        row_K = K_latent * expert_out.element_size()
        if self.compact_dispatch and dispatch_capacity is None:
            s3_C_ptrs = expert_starts.to(torch.int64).mul_(row_K).add_(
                expert_out.data_ptr()
            )
        else:
            s3_C_ptrs = torch.tensor(
                [expert_out.data_ptr() + e * mtp * row_K for e in range(E)],
                dtype=torch.int64,
                device=device,
            )
        s3_ws = torch.zeros(E * (K_latent // 256 + 17), dtype=torch.int32,
                            device=device)
        marlin_grouped_m16_mxfp4(
            intermediate, shard.down_B_ptrs, s3_C_ptrs, shard.down_scales_ptrs,
            expert_starts, expert_counts, E, K_latent, N, s3_ws, E,
            K_latent // 256, max_m_tiles,
        )
        return expert_out, topk_pos

    def forward(self, x, gate):
        """Resident-EP decode MoE over this rank's decode rows.

        Args:
            x: (T, H) hidden decode rows (bf16).
            gate: router module; called on the (T, 1, H) hidden, returns
                (topk_idx, topk_weight[, ...]).

        Returns:
            (T, H) routed expert-path output (down->experts->combine->norm->up),
            WITHOUT the shared expert (added by the caller).
        """
        if self.world_size != 1:
            return self._forward_ep(x, gate)
        T, H = x.shape
        if T == 0:
            return x.new_zeros((0, H))
        K_latent = self.shard.K_latent

        # --- router on the PRE-down hidden ---
        gate_out = gate(x.view(T, 1, H))
        topk_idx, topk_weight = gate_out[0], gate_out[1]
        K = topk_idx.shape[-1]

        # --- down-proj once per token: hidden -> latent ---
        x_latent = self.down_proj(x).contiguous()          # [T, K_latent] bf16

        # --- local shard (all experts at world=1) -> expert_out, topk_pos ---
        expert_out, topk_pos = self._expert_path(
            x_latent, topk_idx.to(torch.int32), T)

        # --- FP32 weighted combine (oracle-identical), then bf16 downcast ---
        y = self._combine_fp32(
            expert_out, topk_pos, topk_weight, T, K_latent, K).to(torch.bfloat16)

        # --- post-combine norm + up-proj: latent -> hidden ---
        if self.norm is not None:
            y = self.norm(y)
        y = self.up_proj(y)                                # [T, H] bf16
        return y

    def _forward_ep(self, x, gate):
        """world>=2 EP path (M3.1b, decisions A13/A16). Mirrors the BF16 resident
        collective structure (``fused_moe_bf16_resident``) but for the LATENT
        dataflow: gather the 3584-d LATENT (NOT the 7168 hidden — halves comm),
        route + shard-experts on the global tokens, reduce the combined LATENT,
        then slice this rank's rows and run norm + up-proj.

            router + down_proj on LOCAL padded hidden (DP-replicated, pre-gather)
            -> all_gather(latent) + all_gather(routing) over EP ranks
            -> this rank's expert shard on the global tokens (non-owned -> 0)
            -> fp32 combine -> all_reduce(SUM) over ranks -> local slice
            -> bf16 -> routed_expert_norm -> routed_expert_up_proj

        The per-rank layout scalar ntp (max decode rows over ranks) is delivered
        by the worker's rank-count sync via ``set_num_tokens_per_rank`` before
        the forward; every rank sizes the global buffer from ntp alone, so no
        extra communication is needed. Empty ranks (T == 0) still run every
        collective and the shard kernels in lockstep — their resident experts
        serve the OTHER ranks' tokens, and a bias-free expert MLP maps padded
        zero rows to exactly zero, so padding contributes nothing to the reduced
        sum and the padded slice is discarded on the local slice.

        Routing note: the router is DP-replicated, so running it on this rank's
        LOCAL hidden and all_gathering the per-token (topk_idx, topk_weight) is
        bit-identical to running it on the gathered global hidden — but gathers
        K int32 + K weights per token instead of the 7168-d hidden.
        """
        ntp = ResidentEPMXFP4MoELayer.num_tokens_per_rank
        T, H = x.shape
        assert ntp is not None and ntp > 0, (
            "resident-EP MXFP4 decode requires the worker rank-count sync "
            "(set_num_tokens_per_rank) before the MoE forward")
        assert T <= ntp, f"local decode rows {T} exceed synced ntp {ntp}"
        world = self.world_size
        num_global = world * ntp
        K_latent = self.shard.K_latent

        # --- router + down-proj on LOCAL (padded) hidden; DP-replicated ---
        padded = x.new_zeros((ntp, H))
        if T > 0:
            padded[:T].copy_(x)
        gate_out = gate(padded.view(ntp, 1, H))
        topk_idx = gate_out[0].reshape(ntp, -1).to(torch.int32)    # [ntp, K]
        topk_weight = gate_out[1].reshape(ntp, -1)                 # [ntp, K]
        K = topk_idx.shape[-1]
        x_latent = self.down_proj(padded).contiguous()            # [ntp, K_latent]

        # --- all_gather the LATENT + routing across EP ranks (NOT the hidden) ---
        all_latent = x_latent.new_empty((num_global, K_latent))
        all_idx = topk_idx.new_empty((num_global, K))
        all_weight = topk_weight.new_empty((num_global, K))
        with self.comm.change_state(enable=True):
            self.comm.all_gather(all_latent, x_latent)
            self.comm.all_gather(all_idx, topk_idx)
            self.comm.all_gather(all_weight, topk_weight)

        # --- this rank's expert shard on the global tokens (non-owned -> -1) ---
        if self.compact_dispatch:
            # Resident weights leave only a few GiB of HBM headroom. Even with
            # exact route-count sizing, a skewed prefill can send every row to
            # one local expert, making the three padded BF16 expert buffers
            # scale with all global rows (8.75 GiB at W1, 35 GiB at W2).
            # Chunk only the independent token dimension. Each token still
            # sees the same expert kernels and FP32 top-k reduction, and the
            # completed global latent is all-reduced exactly once below.
            prefill_y = ResidentEPMXFP4MoELayer._prefill_y
            if (
                prefill_y is None
                or prefill_y.shape[0] < num_global
                or prefill_y.shape[1] != K_latent
                or prefill_y.device != x.device
            ):
                raise RuntimeError(
                    "resident-EP prefill output was not preallocated for "
                    f"{num_global}x{K_latent} FP32 rows on {x.device}"
                )
            y = prefill_y[:num_global]
            chunk_rows = compact_prefill_chunk_rows(
                num_global, self.compact_chunk_rows
            )
            for chunk_start in range(0, num_global, chunk_rows):
                chunk_end = min(chunk_start + chunk_rows, num_global)
                chunk_count = chunk_end - chunk_start
                expert_out, topk_pos = self._expert_path(
                    all_latent[chunk_start:chunk_end],
                    all_idx[chunk_start:chunk_end],
                    chunk_count,
                    dispatch_capacity=chunk_rows,
                )
                y[chunk_start:chunk_end].copy_(
                    self._combine_fp32(
                        expert_out,
                        topk_pos,
                        all_weight[chunk_start:chunk_end],
                        chunk_count,
                        K_latent,
                        K,
                    )
                )
        else:
            expert_out, topk_pos = self._expert_path(
                all_latent, all_idx, num_global
            )
            y = self._combine_fp32(
                expert_out,
                topk_pos,
                all_weight,
                num_global,
                K_latent,
                K,
            )

        # --- SUM the completed combined LATENT across ranks ---
        with self.comm.change_state(enable=True):
            self.comm.all_reduce(y, op=ReduceOp.SUM)

        # --- local slice -> bf16 -> post-combine norm + up-proj: latent->hidden ---
        start = self.rank * ntp
        y = y[start:start + T].to(torch.bfloat16)
        if self.norm is not None:
            y = self.norm(y)
        y = self.up_proj(y)                                        # [T, H] bf16
        return y


def build_resident_ep_mxfp4_layers(model_layers, get_tensor, comm, world_size,
                                   rank, expert_start, num_local, device):
    """Materialize MXFP4 marlin shards for every MoE layer and attach a
    ``ResidentEPMXFP4MoELayer`` to each ``block_sparse_moe`` as
    ``_resident_ep_moe`` (consumed by ``moe_forward_serving``'s decode dispatch).

    Source is ``core_engine.get_tensor`` (the host copy-engine store), keys
    ``routed_expert_{layer}_{expert}`` -> ``w{1,3,2}.weight_{packed,scale}``
    (``k3/mxfp4_layout.routed_expert_tensor_names``). Returns total resident
    bytes.

    NOTE (M3.1a): the pytest gate builds the layer directly from the block's
    experts at world=1; this server-side build path is NOT exercised by that
    gate and is verified by the main loop's server run.
    """
    start_t = time.perf_counter()
    total_bytes = 0
    num_layers = 0
    for layer_idx, layer in enumerate(model_layers):
        moe = getattr(layer, "block_sparse_moe", None)
        if moe is None or moe.experts is None:
            continue
        expert_sources = []
        for i in range(num_local):
            t = get_tensor(f"routed_expert_{layer_idx}_{expert_start + i}")
            expert_sources.append({
                "w1": (t["w1.weight_packed"], t["w1.weight_scale"]),
                "w3": (t["w3.weight_packed"], t["w3.weight_scale"]),
                "w2": (t["w2.weight_packed"], t["w2.weight_scale"]),
            })
        shard = build_layer_shard(expert_sources, device)
        moe._resident_ep_moe = ResidentEPMXFP4MoELayer(
            layer_idx, shard,
            moe.routed_expert_down_proj,
            moe.routed_expert_norm if getattr(moe, "latent_moe_use_norm", False)
            else None,
            moe.routed_expert_up_proj,
            comm=comm, world_size=world_size, rank=rank,
            expert_start=expert_start,
        )
        total_bytes += shard.nbytes()
        num_layers += 1
    logging.info(
        f"Rank {rank}: resident EP MXFP4 marlin shards materialized — "
        f"{num_layers} MoE layers x {num_local} experts, "
        f"{total_bytes / (1024**3):.2f} GiB "
        f"({time.perf_counter() - start_t:.1f}s, one-time repack)"
    )
    return total_bytes
