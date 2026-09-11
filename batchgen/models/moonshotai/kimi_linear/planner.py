# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
#  see the license at                                                           #
#                                                                               #
#                  http://www.apache.org/licenses/license-2.0                   #
# ---------------------------------------------------------------------------- #

"""Kimi-Linear specific planner for BatchGen configuration."""

from batchgen.planner.base_planner import BasePlanner


_GIB = 1024 ** 3
_K3_H200_MIN_MEMORY_BYTES = 120 * _GIB
_K3_H20_PREFILL_TOKEN_CAP = 16_384
_K3_H200_PREFILL_TOKEN_CAP = 524_288
# Admission waves after the first decode phase prefill through the resident
# EP shard (kept on the GPU) in chunks of this many tokens — SGLang's
# max-prefill-tokens for the matched baseline. The 84 GiB shard leaves
# ~25 GB for prefill scratch on H200; the global FP32 MoE output alone is
# world_size x cap x hidden x 4 B = 7.5 GB at 16K.
_K3_H200_RESIDENT_PREFILL_TOKEN_CAP = 16_384
# Resident-prefill expert chunk for passes above 16K global rows. 2048 rows
# is ~3-4 GiB of dispatched/intermediate/expert_out + FP32 combine scratch
# against the ~18 GB free at an H200 handoff prefill; the 256-row default
# was a tight-HBM rule and costs 8x the grouped-Marlin launches.
_K3_H200_RESIDENT_PREFILL_CHUNK_ROWS = 2048
_K3_H20_PREFILL_COLLECTIVE_STRIPE_THRESHOLD_ROWS = 32_768
_K3_H200_PREFILL_COLLECTIVE_STRIPE_THRESHOLD_ROWS = 524_288
_K3_H20_PREFILL_GROUPED_CHUNK_ROWS = 2_048
_K3_H200_PREFILL_GROUPED_CHUNK_ROWS = 32_768
# Per-node decode admission cap on H200 (= KDA state slots). See
# k3_kda_state_slots for the HBM budget behind this number.
#
# MEASURED (2026-09-05, world16 resident-EP decode, 143.8 GB/GPU): configure_
# decoding grows allocation 44.94 -> 136.45 GB (+91.5 GB for the resident expert
# shard), so only ~20 GB is left for KDA state + GPU KV. The cap must be sized
# against BOTH, because slots and KV compete for the same remainder:
#   slot cost 53.57 MiB/seq; MLA KV ~56 MB/seq at the matched decode corpus
#   256 -> 13.8 GB slots + 14.3 GB KV = 28 GB > 20  -> KV starved; the engine
#          admitted only 20 sequences and decode collapsed under eviction
#   128 ->  6.7 GB slots +  7.2 GB KV = 14 GB < 20  -> fits with headroom
# Raising this number does NOT raise concurrency once KV is the binding
# constraint; it lowers it.
#
# But it must ALSO exceed the concurrency GPU KV can fund, because a slot is
# held by every DECODING sequence AND by every sequence in an in-flight prefill
# wave. At 128 a 120-seq prefill wave arriving beside 136 decoding sequences
# overran the pool and killed the batch with "Insufficient free KDA state
# items" (a FATAL allocator error), instead of being throttled by KV
# back-pressure (graceful). Size slots so the graceful limiter binds first:
#   at gpu_memory_frac 0.97 the budget is 135.6 GB and the non-slot base is
#   119.66 GB, so KV = 15.94 - 0.0523*S GB and KV concurrency ~= KV / 0.056 GB.
#   S = 192 -> slots 10.05 GB, KV 5.89 GB -> ~105 KV-funded concurrent, leaving
#   ~87 slots of headroom for prefill waves. Slots > KV concurrency by design.
#
# MEASURED at S=192 (r14): that headroom is WASTED. A slot is held from prefill
# until completion, so 192 slots parked 289 PREFILLED sequences that GPU KV
# could never decode, while in_decode COLLAPSED 98 -> 68 and node throughput
# fell 382 -> 219 tok/s. The earlier balance used the corpus AVERAGE (~56 MB/seq
# of KV); sequences grow toward max_tokens, so solve it at FULL length instead
# (4096 tok ~ 110 MB/seq):
#   KV = 15.94 - 0.0523*S GB ; concurrency = KV / 0.110
#   S = (15.94 - 0.0523*S)/0.110  ->  S ~= 98
# S = 96 -> slots 5.02 GB, KV 10.9 GB -> ~99 concurrent SUSTAINED to full
# length, versus 192 slots decaying to 68. Slots much above the KV-funded
# concurrency buy parked sequences, not throughput.
_K3_H200_KDA_STATE_SLOTS = 96
# With the LatentMoE projections TP8-sharded (DECODE_CONCURRENCY_PLAN.md,
# -8.3 GiB/rank) the same KV-funded solve gives S ~ 150 at full length; 160
# is a multiple of the TP size and leaves a ~15.8 GB KV pool (slots are
# 53.57 MiB each on TP8).
_K3_H200_KDA_STATE_SLOTS_SHARDED = 160
# With the vocab-parallel embed/lm_head as well (-3.84 GiB/rank) the planner
# funds 256 persistent slots per node: every D512 request in flight from
# wave 1 (no mid-run waves, long requests admitted at t=0). Slot cost ~36 MB.
_K3_H200_KDA_STATE_SLOTS_VOCAB_SHARDED = 256


def k3_shard_latent_projections(
    *, gpu_total_memory_bytes: int | None, attention_group_size: int
) -> bool:
    """TP-shard routed_expert_down/up_proj across the attention TP group on
    H200 TP8 (the replicated pair is 9.5 GiB/rank of the 27.6 GiB that
    competes with the KV pool and KDA slots)."""
    group_size = int(attention_group_size)
    if group_size <= 0:
        raise ValueError("attention_group_size must be positive")
    return bool(
        group_size == 8
        and gpu_total_memory_bytes is not None
        and int(gpu_total_memory_bytes) >= _K3_H200_MIN_MEMORY_BYTES
    )


def k3_deepep_low_latency(*, sharded_latent: bool, enable_deepep: bool = False) -> bool:
    """DeepEP LL replaces the 16-rank NCCL gather + reduce_scatter of the
    resident-EP decode graph (slice 3). ``enable_deepep`` is the server
    ``--enable-deepep`` flag: False (default) uses the NCCL path (the
    SGLang-baseline-equivalent exchange); True requires the DeepEP low-latency
    exchange and raises when the sharded-latent gate (H200 TP8) or the K3 DeepEP
    build is unavailable (no silent NCCL fallback). The graph build fails fast
    if the exchange cannot be constructed."""
    if not enable_deepep:
        return False
    from batchgen.moe.deepep_ll import deepep_available
    if not (sharded_latent and deepep_available()):
        raise RuntimeError(
            "--enable-deepep requires the TP-sharded latent (H200 TP8) and the "
            "K3 DeepEP build on PYTHONPATH (deep_ep/K3_BUILD marker)")
    return True


def k3_kda_state_slots(
    *, gpu_total_memory_bytes: int | None, attention_group_size: int,
    sharded_latent: bool = False,
    sharded_vocab: bool = False,
) -> int:
    """Return K3's user-visible KDA sequence capacity for this GPU class.

    The distributed K3 topology uses TP8 KDA, so each rank stores only 1/8
    of every sequence's recurrent/conv state (53.57 MiB per slot). H20 keeps
    the validated four-slot plan. The planner allocates one additional
    physical item for CUDA-graph scratch.

    H200 budget (2026-09-04). This value IS the per-node decode admission cap,
    so it sets decode concurrency and therefore decode throughput: every step
    re-reads the whole 84 GiB resident expert shard regardless of how many
    sequences are in flight, so each extra admitted sequence rides on weight
    traffic already paid for. At 256 slots the routed rows per expert are still
    only 256*16/896 = 4.6, far below the M~16 efficiency shoulder measured in
    the Phase-0 cost model, so the added concurrency is close to free.

        resident expert shard   84.0 GiB   (measured, released during prefill)
        KDA state, 256 slots    13.4 GiB   (256 x 53.57 MiB)
        skeleton/dense/MLA/etc  ~15   GiB
                               ---------
                                ~112  GiB  against 143 GB of H200 HBM

    The prior value was 32 (1.674 GiB/rank), chosen only to lift the H20
    four-sequence ceiling "without consuming the H200 KV-cache headroom" —
    conservative, never tuned for throughput. K3 offloads its MLA KV to host
    paged memory, so GPU KV headroom is not the binding constraint that
    reasoning assumed.

    Unknown memory and non-TP8 configurations fail safe to the validated
    four-slot plan.
    """
    group_size = int(attention_group_size)
    if group_size <= 0:
        raise ValueError("attention_group_size must be positive")
    if gpu_total_memory_bytes is None:
        return 4
    memory_bytes = int(gpu_total_memory_bytes)
    if memory_bytes <= 0:
        raise ValueError("gpu_total_memory_bytes must be positive")
    if group_size == 8 and memory_bytes >= _K3_H200_MIN_MEMORY_BYTES:
        if sharded_latent and sharded_vocab:
            return _K3_H200_KDA_STATE_SLOTS_VOCAB_SHARDED
        if sharded_latent:
            return _K3_H200_KDA_STATE_SLOTS_SHARDED
        return _K3_H200_KDA_STATE_SLOTS
    return 4


def k3_resident_prefill_token_cap(
    *, gpu_total_memory_bytes: int | None, attention_group_size: int
) -> int:
    """Token cap for resident-EP prefill of admission waves after the first
    decode phase; 0 disables the resident handoff (H20 keeps the validated
    release-and-stream cycle, its shard is small and its HBM is not)."""
    group_size = int(attention_group_size)
    if group_size <= 0:
        raise ValueError("attention_group_size must be positive")
    if (
        group_size == 8
        and gpu_total_memory_bytes is not None
        and int(gpu_total_memory_bytes) >= _K3_H200_MIN_MEMORY_BYTES
    ):
        return _K3_H200_RESIDENT_PREFILL_TOKEN_CAP
    return 0


def k3_prefill_micro_batch_token_cap(
    *, gpu_total_memory_bytes: int | None, attention_group_size: int
) -> int:
    """Return the TP8 K3 prefill token cap for the current GPU class.

    H20 keeps the validated single-long-sequence guard. H200 can release the
    84-GiB resident decode expert shard while streamed-SP8 prefill is active,
    leaving enough HBM to combine all eight 64K node-local prompts into one
    layer-wise model pass. Unknown memory and non-TP8 layouts fail safe to the
    H20 cap.
    """
    group_size = int(attention_group_size)
    if group_size <= 0:
        raise ValueError("attention_group_size must be positive")
    if gpu_total_memory_bytes is None:
        return _K3_H20_PREFILL_TOKEN_CAP
    memory_bytes = int(gpu_total_memory_bytes)
    if memory_bytes <= 0:
        raise ValueError("gpu_total_memory_bytes must be positive")
    if group_size == 8 and memory_bytes >= _K3_H200_MIN_MEMORY_BYTES:
        return _K3_H200_PREFILL_TOKEN_CAP
    return _K3_H20_PREFILL_TOKEN_CAP


def k3_prefill_collective_stripe_threshold_rows(
    *, gpu_total_memory_bytes: int | None, attention_group_size: int
) -> int:
    """Return the node-row threshold above which streamed-SP8 is striped.

    H20 keeps the validated 256-row stripe path. H200 streamed prefill first
    releases the 84-GiB resident decode shard, so the exact64 node batch can
    use one wide latent gather/reduce instead of issuing four collectives for
    each 256-row stripe. Unknown memory and non-TP8 layouts fail safe to the
    H20 threshold.
    """
    group_size = int(attention_group_size)
    if group_size <= 0:
        raise ValueError("attention_group_size must be positive")
    if gpu_total_memory_bytes is None:
        return _K3_H20_PREFILL_COLLECTIVE_STRIPE_THRESHOLD_ROWS
    memory_bytes = int(gpu_total_memory_bytes)
    if memory_bytes <= 0:
        raise ValueError("gpu_total_memory_bytes must be positive")
    if group_size == 8 and memory_bytes >= _K3_H200_MIN_MEMORY_BYTES:
        return _K3_H200_PREFILL_COLLECTIVE_STRIPE_THRESHOLD_ROWS
    return _K3_H20_PREFILL_COLLECTIVE_STRIPE_THRESHOLD_ROWS


def k3_prefill_grouped_chunk_rows(
    *, gpu_total_memory_bytes: int | None, attention_group_size: int
) -> int:
    """Return the routed-expert token chunk for streamed-SP8 prefill.

    H20 retains the validated 2K-row bound. H200 has enough temporary HBM for
    32K node rows, which gives each expert hundreds rather than tens of rows
    per grouped Marlin launch and cuts the exact64 loop from 256 to 16 chunks
    per layer. Unknown memory and non-TP8 layouts fail safe to the H20 bound.
    """
    group_size = int(attention_group_size)
    if group_size <= 0:
        raise ValueError("attention_group_size must be positive")
    if gpu_total_memory_bytes is None:
        return _K3_H20_PREFILL_GROUPED_CHUNK_ROWS
    memory_bytes = int(gpu_total_memory_bytes)
    if memory_bytes <= 0:
        raise ValueError("gpu_total_memory_bytes must be positive")
    if group_size == 8 and memory_bytes >= _K3_H200_MIN_MEMORY_BYTES:
        return _K3_H200_PREFILL_GROUPED_CHUNK_ROWS
    return _K3_H20_PREFILL_GROUPED_CHUNK_ROWS


class KimiLinearPlanner(BasePlanner):
    """Planner specialized for Kimi-Linear (48B-A3B, KDA + NoPE-MLA hybrid).

    - 27 layers: 20 KDA (linear attention, no KV) + 7 NoPE-MLA (paged KV,
      kv_lora_rank 512 + qk_rope_head_dim 64 per token, BF16).
    - 256 BF16 routed experts (moe_intermediate 1024) + 1 shared, top-8.
    - KV is only stored for MLA layers; KDA state lives in a separate
      Python-side pool (KDAStateGPUManager), not in KV_Storage.
    """

    # 256 experts * 3 projections * 1024 moe_intermediate / 8 top-k —
    # lighter per-token compute than K2.5; keep H20 MoE magic num.
    MAGIC_NUM = 672_000
    DEFAULT_MEM_FRAC = 0.85
    NUM_EXPERTS = 256

    def __init__(self, is_k3: bool = False, stream_all_modules: bool = None,
                 attention_group_size: int = 1,
                 gpu_total_memory_bytes: int | None = None,
                 enable_deepep: bool = False):
        """``is_k3`` selects the K3 (2.8T, 93L/896E) branch of the plan.

        Passed by ``KimiLinearInitializer`` from its own ``is_k3`` (which is
        derived from the checkpoint's ``config.json``, not from the model
        name). Default False keeps the validated 48B plan byte-identical.

        ``stream_all_modules`` defaults to ``is_k3``: OFF for the 48B, ON for
        K3, which cannot fit attn/kda/shared resident. Passing it True with
        ``is_k3=False`` is the PREFILL_PLAN M3 rehearsal — force the 48B to
        stream all four rings and check its logits are unchanged, on a model
        whose right answer is already known.

        ``attention_group_size`` (G, default 1) is the head-parallel TP degree
        for KDA (M2a). G=1 is the validated single-shard path, byte-identical
        to before. G>1 slices the ``kda_num_heads`` projections across G ranks
        (attn_tp sub-group), each rank owning ``kda_num_heads // G`` heads and
        summing the o_proj partials with an all_reduce; the PSM derives the
        sub-group layout from this value.
        """
        super().__init__()
        self.is_k3 = is_k3
        self.attention_group_size = int(attention_group_size)
        # Kimi-K3 decode MoE exchange (server --enable-deepep): False=NCCL, True=DeepEP
        self.enable_deepep = bool(enable_deepep)
        self.gpu_total_memory_bytes = (
            None
            if gpu_total_memory_bytes is None
            else int(gpu_total_memory_bytes)
        )
        # M2b: G>1 runs head-parallel KDA, whose RESIDENT load path (M2a)
        # requires stream_all_modules OFF -- the streamed-KDA head-shard seam
        # is unwired (PSM raises NotImplementedError). K3 streams only at G=1
        # (where 90 GiB/rank attn+kda+shared cannot fit resident); G>1 shards
        # the KDA (~7 GiB/rank) and goes fully resident. Explicit arg wins.
        if stream_all_modules is None:
            self.stream_all_modules = is_k3 and self.attention_group_size == 1
            if is_k3 and self.attention_group_size > 1:
                import logging
                logging.getLogger(__name__).warning(
                    "[K3] attention_group_size=%d>1 -> stream_all_modules=False "
                    "(resident head-parallel KDA)", self.attention_group_size)
        else:
            self.stream_all_modules = bool(stream_all_modules)
        if is_k3:
            # `_compute_batch_configs` runs before `_adjust_config_for_model`
            # and asserts NUM_EXPERTS // world_size > 0; K3 has 896, not 256.
            # Every value it derives from this is overwritten below (F6).
            self.NUM_EXPERTS = 896

    def __version__(self):
        return "0.1.0"

    def _adjust_config_for_model(self):
        """Kimi-Linear specific config adjustments."""
        # Modern paged-KV decode path (decoding_continuous), same as K2.5.
        self.config.Basic_Config.attn_mode = 3

        # Kimi-Linear context: model_max_length=1048576. The TP8 K3 path
        # keeps only four KDA state slots on H20 (each slot is ~428.6 MiB).
        # The admission scheduler enforces that persistent sequence limit;
        # this token cap is a separate bound on temporary prefill scratch and
        # must not be mistaken for a KDA-slot limit.
        self.config.Module_Batching_Config.prefill_micro_batch_token_cap = (
            k3_prefill_micro_batch_token_cap(
                gpu_total_memory_bytes=self.gpu_total_memory_bytes,
                attention_group_size=self.attention_group_size,
            )
            if self.is_k3 and self.attention_group_size > 1
            else 262_144
        )
        self.config.Module_Batching_Config.k3_resident_prefill_token_cap = (
            k3_resident_prefill_token_cap(
                gpu_total_memory_bytes=self.gpu_total_memory_bytes,
                attention_group_size=self.attention_group_size,
            )
            if self.is_k3
            else 0
        )
        self.config.Module_Batching_Config.k3_resident_prefill_chunk_rows = (
            _K3_H200_RESIDENT_PREFILL_CHUNK_ROWS
            if self.config.Module_Batching_Config.k3_resident_prefill_token_cap > 0
            else 0
        )
        self.config.Module_Batching_Config.k3_prefill_collective_stripe_threshold_rows = (
            k3_prefill_collective_stripe_threshold_rows(
                gpu_total_memory_bytes=self.gpu_total_memory_bytes,
                attention_group_size=self.attention_group_size,
            )
            if self.is_k3 and self.attention_group_size > 1
            else _K3_H20_PREFILL_COLLECTIVE_STRIPE_THRESHOLD_ROWS
        )
        self.config.Module_Batching_Config.k3_prefill_grouped_chunk_rows = (
            k3_prefill_grouped_chunk_rows(
                gpu_total_memory_bytes=self.gpu_total_memory_bytes,
                attention_group_size=self.attention_group_size,
            )
            if self.is_k3 and self.attention_group_size > 1
            else _K3_H20_PREFILL_GROUPED_CHUNK_ROWS
        )

        # BF16 KV (no kv quantization). Canonical spelling is "bfloat16":
        # base_planner's kv_element_size check and the engine dtype map
        # (core/utils.cpp) both match on "bfloat16", never "bf16".
        self.config.Basic_Config.kv_dtype = "bfloat16"

        # F6: _compute_batch_configs() ran before this method with the base
        # default attn_mode=1 (our attn_mode=3 above lands after buffer
        # sizing), so its expert-residency math is wrong for this model.
        # Override all three results here — this method runs last, so the
        # values stick:
        # (a) Decode expert staging buffers: normalize to the prefill ring (8).
        #     The memory-based split left 7-9 buffers. Kept non-zero under
        #     resident-EP decode (which streams nothing — the worker only
        #     starts the decode H2D streamer when the PSM returns a non-empty
        #     routed_expert task): 8 slots ~ 113 MB keep the
        #     decode_moe_mode="streamed" fallback launchable without
        #     re-planning.
        self.config.GPU_Buffer_Config.num_decoding_module_buffer["routed_expert"] = 8
        # (b) Zero persistent experts FOR THE ENGINE: this field is consumed
        #     Python-side only (base_planner sizing, other models' PSMs; the
        #     C++ engine reads model_config.num_local_experts in
        #     core/utils.cpp:357, never EP_Config) — it does NOT size the
        #     resident-EP decode shards, which the Kimi PSM keeps fully
        #     Python-side. 0 states the engine-facing contract: all streamed
        #     (prefill streams all 256 experts/rank, persistent=False).
        self.config.EP_Config.num_local_expert_per_layer = 0
        # (c) Decode admission cap — consumed by the worker as the per-rank
        #     max decode sequences (batchgen_worker rank_counts checks); the
        #     base leaves it None -> TypeError on first decode admission.
        #     16/rank x 8 ranks = single-wave admission of a 128-seq batch.
        #     Also bounds the resident-EP decode collective layout
        #     (num_tokens_per_rank <= 16 -> <= 128-row global buffer).
        self.config.Module_Batching_Config.MoE_decoding_micro_batch_size = 16

        # M4 P0.3 — decode MoE execution mode, read by the PSM at
        # configure_decoding (config-driven, no env vars):
        #   "resident_ep": stacked EP-8 BF16 shards resident on GPU
        #                  (~11.8 GB/rank) + fused_moe_bf16 + all_reduce;
        #   "streamed":    legacy per-expert host streaming (fallback;
        #                  ~94 GB H2D/step -> ~0.5 tok/s floor).
        self.config.EP_Config.decode_moe_mode = "resident_ep"

        # KDA conv/recurrent state-pool slots (peak concurrent sequences the
        # PSM pools can hold). Config-driven — replaces the retired
        # BATCHGEN_KDA_STATE_SLOTS env var (M1-C). One slot is reserved by the
        # Phase-A decode graph as the padding/warmup scratch slot.
        self.config.GPU_Buffer_Config.kda_state_slots = 256

        # M5.2 Phase-A CUDA-graph decode (per-layer attention spans; MoE stays
        # eager because its resident-EP forward runs NCCL collectives). Read by
        # the PSM at configure_decoding — config-driven, no env vars.
        #   "eager":   no graphs (default until the M5.5 gates pass);
        #   "graph":   replay the captured spans;
        #   "compare": replay AND run the eager span, logging max|delta|.
        # A batch can override per request with
        # batchgen_debug.kimi_decode_graph_mode.
        self.config.Basic_Config.decode_graph_mode = "eager"
        # Per-rank decode buckets; the top bucket must cover
        # MoE_decoding_micro_batch_size (16 rows/rank above).
        self.config.Basic_Config.decode_graph_buckets = [1, 2, 4, 8, 16]
        # Steps between graph-vs-eager comparisons in "compare" mode.
        self.config.Basic_Config.decode_graph_compare_every = 64

        # M-PR-6 — stream attn / kda_attn / shared_expert as well as the
        # routed experts. Read by the PSM at configure_prefill. OFF for the
        # 48B so its validated path keeps every weight resident, bit-for-bit.
        self.config.Basic_Config.stream_all_modules = self.stream_all_modules

        # M2a — head-parallel TP degree for KDA (G). Read by the PSM
        # (__init__) to derive the attn_tp sub-group layout. 1 = single-shard
        # (unchanged); >1 shards kda_num_heads across G ranks with an o_proj
        # all_reduce.
        self.config.Basic_Config.attention_group_size = self.attention_group_size

        if self.is_k3:
            self._adjust_config_for_k3()
        if self.stream_all_modules:
            self._configure_streamed_rings()

    def _adjust_config_for_k3(self):
        """K3 (93 layers, 896 experts, MXFP4) overrides. Runs last.

        The 48B keeps attn/kda_attn/shared_expert resident. K3 cannot: at
        measured per-module sizes (k3_module_shapes, cross-checked against the
        released index) residency alone is

            69 KDA  x 846.67 MiB = 57.05 GiB
            24 MLA  x 442.88 MiB = 10.38 GiB
            92 shrd x 252.00 MiB = 22.64 GiB
                                 = 90.07 GiB

        against 95.58 GiB of HBM, before the skeleton, the KDA state pools, the
        expert ring, activations or the CUDA context. Streaming all four module
        types is what makes 8xH20 possible at all.
        """
        # KDA conv/recurrent state pools. At K3 dimensions, an unsharded slot
        # is 428.6 MiB. Distributed K3 uses TP8 KDA, making it 53.57 MiB/rank.
        # Preserve 4 user sequences on H20 and 32 on H200, then allocate one
        # separate physical item for CUDA-graph padding/warmup. The graph
        # reserves that item before readiness, so admission still sees exactly
        # the user capacity instead of silently losing one sequence.
        shard_latent = k3_shard_latent_projections(
            gpu_total_memory_bytes=self.gpu_total_memory_bytes,
            attention_group_size=self.attention_group_size,
        )
        self.config.Module_Batching_Config.k3_shard_latent_projections = shard_latent
        # slice 2 rides the same gate: vocab-parallel embed_tokens / lm_head
        self.config.Module_Batching_Config.k3_shard_vocab_parallel = shard_latent
        self.config.Module_Batching_Config.k3_deepep_ll = k3_deepep_low_latency(
            sharded_latent=shard_latent, enable_deepep=self.enable_deepep,
        )
        sequence_slots = k3_kda_state_slots(
            gpu_total_memory_bytes=self.gpu_total_memory_bytes,
            attention_group_size=self.attention_group_size,
            sharded_latent=shard_latent,
            sharded_vocab=shard_latent,
        )
        self.config.GPU_Buffer_Config.kda_state_slots = sequence_slots + 1

        # This is a per-node sequence cap under TP8: every rank in the node
        # holds the same sequence set. Keep the graph top bucket equal to that
        # cap. The 24 bucket avoids padding a 17--24 row TP8 group to 32; that
        # distinction becomes load-bearing when the grouped MoE joins capture
        # because ceil(24/8)=3 while ceil(32/8)=4 rows/rank.
        self.config.Module_Batching_Config.MoE_decoding_micro_batch_size = (
            sequence_slots
        )
        # The cap itself must be a bucket: at 96 slots the candidate tuple
        # tops out at 64, so a full 96-row decode batch had no bucket and the
        # graph path fell back to eager for the whole run.
        self.config.Basic_Config.decode_graph_buckets = sorted({
            bucket
            for bucket in (1, 2, 4, 8, 16, 24, 32, 64, 96, 128, 160, 192, 256)
            if bucket <= sequence_slots
        } | {int(sequence_slots)})
        # Whole-model capture (collectives included) -- the analogue of
        # SGLang's --cuda-graph-backend-decode full. Eager K3 decode on H200
        # is latency-bound: ~0.3 s/step at 192 rows AND at 72 rows (93 layers
        # of launches + per-layer NCCL), so the long-generation tail of a
        # 512x4096 run cost r20 ~20 min at 72/192 occupancy. Falls back to
        # eager per step when any EP rank is empty. Batch-level override:
        # batchgen_debug.kimi_decode_graph_mode. H200 TP8 only: the H20
        # four-slot plan keeps its validated eager path.
        if (
            self.attention_group_size == 8
            and self.gpu_total_memory_bytes is not None
            and int(self.gpu_total_memory_bytes) >= _K3_H200_MIN_MEMORY_BYTES
        ):
            self.config.Basic_Config.decode_graph_mode = "graph"

    def _configure_streamed_rings(self):
        """Ring depths for the four streamed module types.

        Sizes below are K3's (k3_module_shapes); the 48B's modules are ~13x
        smaller, so these depths are cheaper still on the M3 rehearsal.
        """
        # Ring depths. Cost = depth x per-instance module size:
        #   attn           442.88 MiB x 2 =   885.8 MiB
        #   kda_attn       846.67 MiB x 2 =  1653.3 MiB
        #   shared_expert  252.00 MiB x 2 =   504.0 MiB
        #   routed_expert   16.73 MiB x112 = 1873.9 MiB
        #                                  = 4.80 GiB total
        # Depth 2 for the three big types is the MINIMUM that overlaps at all:
        # one slot being consumed while the producer fills the next. Each layer
        # requests exactly one attn OR one kda_attn and one shared_expert, so a
        # third slot buys prefetch cushion at 0.83 GiB per slot on the KDA ring
        # — the worst HBM-per-benefit trade of the four. Depth 112 on the routed
        # experts is one TP8 shard of a K3 layer (896/8): streamed-SP8 acquires
        # the whole shard before the node-local weight all-gather. The legacy
        # streamed path also tolerates the deeper producer lead.
        #
        # kda_attn has no entry in base_planner's default map
        # (base_planner.py:90-94), and GPU_Weight_Buffer::Init() iterates that
        # map (GPU_Weight_Buffer.cpp:95-125) — an absent key means zero slots,
        # acquireEmptyBuffer returns nullopt forever and the consumer dies on
        # the 2 s throw. Declaring the whole map here fixes that model-side.
        self.config.GPU_Buffer_Config.num_prefill_module_buffer = {
            "attn": 2,
            "kda_attn": 2,
            "shared_expert": 2,
            "routed_expert": 112,
        }

    def get_module_shapes(self) -> dict:
        """Return Kimi-Linear specific tensor shapes."""
        return {
            "hidden_size": 2304,
            "num_attention_heads": 32,
            "num_kv_heads": 1,  # MLA — all heads compress KV
            "num_layers": 27,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "expert_hidden_size": 1024,  # moe_intermediate_size
            "kv_lora_rank": 512,
            "q_lora_rank": None,  # direct q_proj (no q lora)
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
        }
