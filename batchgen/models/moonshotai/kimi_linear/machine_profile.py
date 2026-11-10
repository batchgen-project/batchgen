# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                         #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
#  see the license at                                                           #
#                                                                               #
#                  http://www.apache.org/licenses/license-2.0                   #
# ---------------------------------------------------------------------------- #

"""Per-machine Kimi engine profiles (K3 DP-attention rows, Kimi-Linear-48B reserve).

Every machine-dependent K3 knob lives in one row keyed by machine family
(``batchgen.hardware.machine_family``), world size and the DP-attention rank
size G (``attention_group_size``: attention, KDA and the shared expert run
TP-G inside one DP rank, the routed experts run EP over every GPU of the
job). Each value carries its source in ``sources``: a dated measurement, a
stated derivation, or "provisional". The planner reads the row and nothing
else; a (family, world size, G) triple without a row is refused by name
instead of borrowing another machine's constants.

The physical inputs behind the rows (``K3_GEOMETRY``, ``K3_FAMILY_RATES``
and each row's ``decode_budget``) are recorded next to the rows so the unit
tests recompute the derivations: the streamed prefill cap against the
weight-stream crossover and the HBM budget, the resident prefill cap against
the HBM beside the resident shard, and the KDA slot count against the GPU
KV solve.

Every row prefills in a streamed mode by default: all rows shard the
LatentMoE projections, so the parallel strategy manager's server prefill
mode is ``streamed_sp8`` (layer-wise expert streaming over the row's
``prefill_stream_span``); resident-EP prefill for every wave is an explicit
server option (``--prefill-moe-mode resident_ep``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Mapping

from batchgen.hardware.machine_family import MachineFamily, signature_for

_GIB = 1024 ** 3
_MIB = 1024 ** 2

class UnsupportedK3TopologyError(ValueError):
    """No K3 profile row exists for this machine family and topology."""


@dataclass(frozen=True)
class K3Geometry:
    """K3 checkpoint geometry the derivations use (read from the layout)."""

    routed_expert_bytes: int = 17_547_264  # one MXFP4 expert incl. scales
    routed_experts: int = 896
    moe_layers: int = 92  # 93 layers, first one dense
    mla_layers: int = 24
    mla_latent_width: int = 576  # kv_lora_rank 512 + rope 64, BF16
    hidden_size: int = 7168
    #: KDA conv + recurrent state of one sequence, all heads (MEASURED at TP8
    #: as 53.57 MiB per rank; the state is head-sharded, so it scales 1/G).
    kda_slot_bytes_unsharded: int = 8 * int(53.57 * _MIB)
    #: Decode parameters that are TP-sharded across the DP-attention rank,
    #: summed over the rank (module sizes from the released index): 69 KDA x
    #: 846.67 MiB + 24 MLA x 442.88 MiB + 92 shared experts x 252.00 MiB +
    #: 92 LatentMoE down/up projections x 98.00 MiB + vocab-parallel
    #: embed/lm_head 2 x 163,840 x 7,168 x 2 B.
    tp_sharded_decode_bytes: int = int(
        (69 * 846.67 + 24 * 442.88 + 92 * 252.00 + 92 * 98.00) * _MIB
        + 2 * 163_840 * 7_168 * 2
    )

    @property
    def kv_bytes_per_token(self) -> int:
        """GPU MLA KV per token per rank (the latent is replicated under TP-G)."""
        return self.mla_layers * self.mla_latent_width * 2

    def kda_slot_bytes(self, group_size: int) -> int:
        """KDA state of one sequence per rank at DP-attention rank size G."""
        return self.kda_slot_bytes_unsharded // int(group_size)

    def tp_sharded_decode_gib(self, group_size: int) -> float:
        """Per-rank share of the TP-sharded decode parameters at size G."""
        return self.tp_sharded_decode_bytes / int(group_size) / _GIB

    def resident_shard_bytes(self, world_size: int) -> int:
        """Resident EP shard of the routed experts per rank."""
        return self.moe_layers * (self.routed_experts // world_size) * self.routed_expert_bytes

    def stream_bytes_per_layer(self, stream_span: int = 8) -> int:
        """Routed-expert bytes one rank streams per MoE layer in streamed_sp8.

        The ``stream_span`` ranks of one span hold one copy of the layer's
        experts between them, so each streams ``experts / stream_span``.
        """
        return (self.routed_experts // stream_span) * self.routed_expert_bytes


K3_GEOMETRY = K3Geometry()

#: Activation bytes per node row per rank of a streamed K3 prefill pass at
#: G=8. ESTIMATED lower bound from the H200 out-of-memory record of a
#: 518,718-row pass: 134.10 GiB in use + 6.93 GiB requested against 42.91 GB
#: at the first iteration.
K3_PREFILL_ACTIVATION_BYTES_PER_ROW = (
    (134.10 + 6.93) * _GIB - 42.91e9
) / 518_718


def k3_prefill_activation_bytes_per_row(group_size: int) -> float:
    """Activation bytes per DP-rank row per rank at DP-attention size G.

    ESTIMATED: the measured G=8 figure scaled by 8/G, because each rank of a
    TP-G rank carries 1/G of the rank's rows through the row-sharded body and
    1/G of the heads through attention.
    """
    return K3_PREFILL_ACTIVATION_BYTES_PER_ROW * 8 / int(group_size)


@dataclass(frozen=True)
class K3FamilyRates:
    """Physical rates of one machine family for the prefill derivation.

    ``stream_seconds_per_layer``: time to stream one rank's 112 experts of one
    MoE layer. ``compute_seconds_per_row``: streamed-prefill compute per node
    row per layer per rank (a1). The crossover ``C* = B / a1`` is the wave
    size above which compute hides the weight stream.
    """

    stream_seconds_per_layer: float
    compute_seconds_per_row: float
    source: str


_A1_H200 = 77.6 / 93 / 524_288

K3_FAMILY_RATES: Mapping[MachineFamily, K3FamilyRates] = MappingProxyType({
    MachineFamily.H20: K3FamilyRates(
        stream_seconds_per_layer=0.090,
        compute_seconds_per_row=0.160 / 32_768,
        source=(
            "MEASURED: 84-90 ms layer supply floor on the distributed store "
            "(upper end used); ~160 ms per layer at 32,768 node rows "
            "(2026-11-01)"
        ),
    ),
    MachineFamily.H200: K3FamilyRates(
        stream_seconds_per_layer=K3_GEOMETRY.stream_bytes_per_layer() / 38e9,
        compute_seconds_per_row=_A1_H200,
        source=(
            "MEASURED: 38 GB/s effective host-to-device rate in serving "
            "(the slower of the measured rates); 77.6 s per 93-layer pass at "
            "524,288 node rows (2026-09-03)"
        ),
    ),
    MachineFamily.B300: K3FamilyRates(
        stream_seconds_per_layer=K3_GEOMETRY.stream_bytes_per_layer() / 50e9,
        compute_seconds_per_row=_A1_H200 / 2.2,
        source=(
            "ESTIMATED: host root ports run PCIe Gen5 x16 (link measured), "
            "rate 50 GB/s assumed; compute = H200 a1 / 2.2 (BF16 spec ratio)"
        ),
    ),
    MachineFamily.GB300: K3FamilyRates(
        stream_seconds_per_layer=K3_GEOMETRY.stream_bytes_per_layer() / 126e9,
        compute_seconds_per_row=_A1_H200 / 2.4,
        source=(
            "MEASURED rate: 126 GB/s cross-NUMA C2C (the slower of 212/126); "
            "ESTIMATED compute = H200 a1 / 2.4 (BF16 spec ratio)"
        ),
    ),
})


@dataclass(frozen=True)
class K3DecodeBudget:
    """Device-memory inputs of one row's decode-time HBM accounting.

    ``non_slot_base_gib`` is the device memory in use after decode
    configuration minus the KDA state pool (resident expert shard, decode
    parameters, CUDA context, collectives), before the GPU KV pool.
    ``prefill_non_expert_gib`` is the memory in use at the first iteration
    of a streamed prefill pass (resident shard released).
    """

    device_total_gib: float
    gpu_memory_frac: float
    non_slot_base_gib: float
    prefill_non_expert_gib: float
    graph_reserve_gib: float
    full_length_tokens: int
    source: str

    @property
    def budget_gib(self) -> float:
        return self.device_total_gib * self.gpu_memory_frac


def solve_kda_user_slots(
    budget: K3DecodeBudget,
    *,
    extra_physical_slots: int,
    attention_group_size: int,
    geometry: K3Geometry = K3_GEOMETRY,
) -> float:
    """Sequences the decode budget sustains to full length.

    Slots and GPU KV compete for the memory above the non-slot base. A slot
    is held from prefill until completion, so the count is solved at FULL
    length: ``S * (slot + kv(full_length)) = budget - base - graph reserve -
    extra physical slots``. The KDA slot is head-sharded over the
    DP-attention rank of size ``attention_group_size``. Returns the
    real-valued solution; a row takes a G multiple at or below it.
    """
    slot_gib = geometry.kda_slot_bytes(attention_group_size) / _GIB
    kv_gib = geometry.kv_bytes_per_token * budget.full_length_tokens / _GIB
    free_gib = (
        budget.budget_gib
        - budget.non_slot_base_gib
        - budget.graph_reserve_gib
        - extra_physical_slots * slot_gib
    )
    return free_gib / (slot_gib + kv_gib)


@dataclass(frozen=True)
class K3EngineProfile:
    """Every machine-dependent K3 knob for one (family, node count, G)."""

    family: MachineFamily
    num_nodes: int
    #: DP-attention rank size G: attention, KDA and the shared expert run
    #: TP-G inside one DP rank; the routed experts run EP over the world.
    attention_group_size: int
    shard_latent_projections: bool
    shard_vocab_parallel: bool
    #: Per-DP-rank decode admission cap (= user-visible KDA state slots).
    kda_user_slots: int
    #: How ``kda_user_slots`` was chosen: "kv-solve" (``solve_kda_user_slots``
    #: on ``decode_budget``) or "measured: ..." / "inherited: ...".
    kda_slots_basis: str
    prefill_micro_batch_token_cap: int
    resident_ep_prefill_token_cap: int
    resident_prefill_chunk_rows: int
    prefill_grouped_chunk_rows: int
    prefill_collective_stripe_threshold_rows: int
    #: Ranks that hold one copy of a layer's routed experts between them in
    #: streamed prefill (a multiple of G dividing the world): each streams
    #: ``experts / span`` and the layer's latent rows are gathered over it.
    prefill_stream_span: int
    prefix_pool_workspace_bytes: int
    decode_budget: K3DecodeBudget
    #: Source of every knob above, keyed by field name.
    sources: Mapping[str, str]
    #: Knobs whose value knowingly violates its derivation, with the reason
    #: (a "stream-bound" cap, or a tracked overrun awaiting measurement).
    derivation_exceptions: Mapping[str, str] = field(default_factory=dict)

    @property
    def world_size(self) -> int:
        return self.num_nodes * signature_for(self.family).gpus_per_node


def k3_profile_knob_names() -> tuple[str, ...]:
    """The K3 knobs every row must source."""
    skip = {"family", "num_nodes", "attention_group_size", "decode_budget",
            "sources", "derivation_exceptions", "kda_slots_basis"}
    return tuple(f.name for f in fields(K3EngineProfile) if f.name not in skip)


# --------------------------------------------------------------------------- #
# Values shared by rows (each row still states its own source).               #
# --------------------------------------------------------------------------- #

_H200_STREAMED_CAP = 524_288
_H200_RESIDENT_CAP = 16_384
_H200_RESIDENT_CHUNK_ROWS = 2_048
_H200_GROUPED_CHUNK_ROWS = 32_768
_H200_STRIPE_THRESHOLD_ROWS = 524_288
_H200_PREFIX_POOL_BYTES = 40 * _GIB

_NODE_STREAM_SPAN = (
    "validated streamed-SP8 layout: the node's eight ranks hold one copy of "
    "each layer's experts (112 each) and gather latent rows inside the node"
)
_JOB_STREAM_SPAN = (
    "POIS 2026-11-10: streaming over the EP span; every rank of the job "
    "streams the 112 experts it owns as an EP8 rank and the latent rows of "
    "every DP rank are gathered over the eight ranks"
)

_INHERITED_H200 = (
    "inherited from the h200 row (the value this machine ran under the "
    "retired memory tiers); provisional until measured on this machine"
)

# H20, 4 nodes x 8 (world 32), distributed host weights.
#
# KDA slots MEASURED 2026-10-29 (4-node world32 diagnostic run): the
# unsharded resident decode footprint was 71.7 GiB/rank (resident expert
# shard ~42 GiB + replicated latent projections 9.5 GiB + vocab 3.84 GiB +
# dense/KDA/skeleton ~15 GiB) and funded only four slots; TP8-sharded latent
# projections (-8.3 GiB) leave ~21 GiB of headroom against the ~84.6 GiB
# budget at frac 0.95. A 512-request batch prefilled in one wave and decoded
# in a single resident phase at 128 rows/node with 15.5 GiB of driver
# headroom after configure. Vocab-parallel stays off: on this topology it
# corrupted decode outputs (sub-project
# batchgen-k3-vocab-parallel-h20-corruption).
_H20_W32 = K3EngineProfile(
    family=MachineFamily.H20,
    num_nodes=4,
    attention_group_size=8,
    shard_latent_projections=True,
    shard_vocab_parallel=False,
    kda_user_slots=128,
    kda_slots_basis=(
        "measured: 4-node world32 run 2026-10-29, 128 rows/node resident "
        "decode with 15.5 GiB driver headroom after configure"
    ),
    prefill_micro_batch_token_cap=16_384,
    resident_ep_prefill_token_cap=0,
    resident_prefill_chunk_rows=0,
    prefill_grouped_chunk_rows=2_048,
    prefill_collective_stripe_threshold_rows=32_768,
    prefill_stream_span=8,
    prefix_pool_workspace_bytes=24 * _GIB,
    decode_budget=K3DecodeBudget(
        device_total_gib=89.05,
        gpu_memory_frac=0.95,
        non_slot_base_gib=63.0,
        prefill_non_expert_gib=21.4,
        graph_reserve_gib=2.0,
        full_length_tokens=4096,
        source=(
            "MEASURED 2026-10-29: 71.7 GiB unsharded decode residency, "
            "-8.3 GiB sharded latent -> ~63 GiB; prefill non-expert "
            "ESTIMATED as that residency minus the ~42 GiB resident shard"
        ),
    ),
    sources=MappingProxyType({
        "shard_latent_projections": "measured 2026-10-29: funds the 128-slot plan",
        "shard_vocab_parallel": (
            "measured 2026-10-29: vocab-parallel corrupts decode on this "
            "topology; off until that seam is fixed"
        ),
        "kda_user_slots": "measured 2026-10-29 (see kda_slots_basis)",
        "prefill_micro_batch_token_cap": "validated single-long-sequence guard",
        "resident_ep_prefill_token_cap": "0: resident-EP prefill is not planned on this topology",
        "resident_prefill_chunk_rows": "0: no resident-EP prefill planned",
        "prefill_grouped_chunk_rows": "validated 2K-row bound",
        "prefill_collective_stripe_threshold_rows": "validated 256-row stripe path above 32K rows",
        "prefill_stream_span": _NODE_STREAM_SPAN,
        "prefix_pool_workspace_bytes": (
            "measured: full in-wave pool matrix on a 4-node serve group, "
            "workspace peak plateaus under 13.5 GiB; 24 GiB keeps >= 1.5x"
        ),
    }),
    derivation_exceptions=MappingProxyType({
        "prefill_micro_batch_token_cap": (
            "stream-bound: 16,384 rows sit just below the 17K-21K crossover "
            "of the distributed store; kept as the validated guard"
        ),
    }),
)

# H200, 2 nodes x 8 (world 16).
#
# KDA slots: TP8-sharded latent projections (-8.3 GiB/rank) plus the
# vocab-parallel embed/lm_head (-3.84 GiB/rank) fund 256 persistent slots per
# node: every request of the measured short-decode corpus is in flight from
# the first wave. The full-length solve (4,096 tokens) funds fewer; history
# (MEASURED 2026-09-05, world16 resident-EP decode): at 192 unsharded slots,
# parked prefilled sequences that GPU KV could not decode cut node throughput
# from 382 to 219 tok/s, which is why the solve is taken at full length.
_H200_W16 = K3EngineProfile(
    family=MachineFamily.H200,
    num_nodes=2,
    attention_group_size=8,
    shard_latent_projections=True,
    shard_vocab_parallel=True,
    kda_user_slots=256,
    kda_slots_basis=(
        "measured: 2-node world16 resident-EP decode with sharded latent and "
        "vocab, short-decode corpus fully admitted from the first wave"
    ),
    prefill_micro_batch_token_cap=_H200_STREAMED_CAP,
    resident_ep_prefill_token_cap=_H200_RESIDENT_CAP,
    resident_prefill_chunk_rows=_H200_RESIDENT_CHUNK_ROWS,
    prefill_grouped_chunk_rows=_H200_GROUPED_CHUNK_ROWS,
    prefill_collective_stripe_threshold_rows=_H200_STRIPE_THRESHOLD_ROWS,
    prefill_stream_span=8,
    prefix_pool_workspace_bytes=_H200_PREFIX_POOL_BYTES,
    decode_budget=K3DecodeBudget(
        device_total_gib=139.81,
        gpu_memory_frac=0.97,
        non_slot_base_gib=106.6,
        prefill_non_expert_gib=39.96,
        graph_reserve_gib=2.0,
        full_length_tokens=4096,
        source=(
            "MEASURED 2026-09-05: non-slot base 119.66 GB unsharded at frac "
            "0.97, minus 8.3 + 3.84 GiB of sharding; prefill non-expert = "
            "42.91 GB in use at the first iteration of a streamed pass"
        ),
    ),
    sources=MappingProxyType({
        "shard_latent_projections": "measured: TP8-sharded latent validated on 2-node H200",
        "shard_vocab_parallel": "measured: vocab-parallel validated on 2-node H200",
        "kda_user_slots": "measured (see kda_slots_basis)",
        "prefill_micro_batch_token_cap": (
            "eight 64K node-local prompts in one layer-wise pass with the "
            "resident shard released"
        ),
        "resident_ep_prefill_token_cap": (
            "~25 GB beside the 84 GiB resident shard; the global FP32 MoE "
            "output alone is world x cap x hidden x 4 B = 7.5 GB at 16K"
        ),
        "resident_prefill_chunk_rows": (
            "measured probe: 2,048 rows is ~3-4 GiB of expert scratch against "
            "~18 GB free; the 256-row clamp cost 8x the grouped launches"
        ),
        "prefill_grouped_chunk_rows": "32K node rows: 16 grouped chunks per layer instead of 256",
        "prefill_collective_stripe_threshold_rows": (
            "one wide latent gather/reduce with the resident shard released"
        ),
        "prefill_stream_span": _NODE_STREAM_SPAN,
        "prefix_pool_workspace_bytes": (
            "provisional: segmented KDA chunk-sweep peaks plus the streamed "
            "grouped MoE FP32 output at the chunk cap, plus margin"
        ),
    }),
    derivation_exceptions=MappingProxyType({
        "prefill_micro_batch_token_cap": (
            "tracked HBM overrun: a 518,718-row streamed pass ran out of "
            "memory (sub-project "
            "k3-streamed-prefill-microbatch-cap-exceeds-activation-budget); "
            "the cap stays until that sub-project measures the limit"
        ),
    }),
)

# B300, 1 node x 8 (world 8): the whole EP8 expert set resident on one node.
#
# KDA slots (PROVISIONAL, kv-solve): MEASURED 2026-11-09 on the single-node
# deployment at frac 0.85 with 273 physical slots (256 user + 1 graph
# scratch + 16 prefix segments): device total 267.68 GiB, 208.13 GiB in use
# after decode configure (199.80 GiB allocated: resident EP shard 168.39 GiB,
# decode parameters 16.06 GiB, KDA state 273 x 53.57 MiB = 14.28 GiB). The
# non-slot base is 208.13 - 14.28 = 193.85 GiB. At frac 0.95 the budget is
# 254.30 GiB; minus the base, a 2 GiB graph reserve and 17 non-user slots
# (0.89 GiB), 57.56 GiB remain for user slots and GPU KV. One sequence held
# to 4,096 tokens costs 53.57 MiB of KDA state + 4,096 x 27,648 B of MLA KV
# = 0.1578 GiB, so S = 57.56 / 0.1578 = 364.8; the largest TP-size multiple
# below it is 360. A K3.1 run at frac 0.95 peaked at 254.65 GiB in use
# (2026-11-10), so the 0.95 budget leaves ~13 GiB of the device untouched.
# Every prefill knob keeps the value this machine ran under the retired
# memory tiers; streamed waves of 448K tokens prefilled without running out
# of memory (2026-11-10).
_B300_W8 = K3EngineProfile(
    family=MachineFamily.B300,
    num_nodes=1,
    attention_group_size=8,
    shard_latent_projections=True,
    shard_vocab_parallel=True,
    kda_user_slots=360,
    kda_slots_basis="kv-solve",
    prefill_micro_batch_token_cap=_H200_STREAMED_CAP,
    resident_ep_prefill_token_cap=_H200_RESIDENT_CAP,
    resident_prefill_chunk_rows=_H200_RESIDENT_CHUNK_ROWS,
    prefill_grouped_chunk_rows=_H200_GROUPED_CHUNK_ROWS,
    prefill_collective_stripe_threshold_rows=_H200_STRIPE_THRESHOLD_ROWS,
    prefill_stream_span=8,
    prefix_pool_workspace_bytes=_H200_PREFIX_POOL_BYTES,
    decode_budget=K3DecodeBudget(
        device_total_gib=267.68,
        gpu_memory_frac=0.95,
        non_slot_base_gib=208.13 - 273 * 53.57 / 1024,
        prefill_non_expert_gib=39.96 + (360 + 17 - 273) * 53.57 / 1024,
        graph_reserve_gib=2.0,
        full_length_tokens=4096,
        source=(
            "MEASURED 2026-11-09 decode configure (see the row comment); "
            "prefill non-expert ESTIMATED as the measured H200 first-iteration "
            "use plus the extra KDA slots of this row"
        ),
    ),
    sources=MappingProxyType({
        "shard_latent_projections": (
            "measured 2026-11-09/10: single-node runs with sharded latent "
            "passed the accuracy gate"
        ),
        "shard_vocab_parallel": "measured 2026-11-09/10: same runs, vocab-parallel on",
        "kda_user_slots": "provisional kv-solve on measured memory (see the row comment)",
        "prefill_micro_batch_token_cap": (
            _INHERITED_H200 + "; 448K-token streamed waves ran 2026-11-10"
        ),
        "resident_ep_prefill_token_cap": _INHERITED_H200,
        "resident_prefill_chunk_rows": _INHERITED_H200,
        "prefill_grouped_chunk_rows": _INHERITED_H200,
        "prefill_collective_stripe_threshold_rows": _INHERITED_H200,
        "prefill_stream_span": _NODE_STREAM_SPAN,
        "prefix_pool_workspace_bytes": _INHERITED_H200,
    }),
)

# DP-attention rank size 4 (TP4 attention/KDA/shared inside a DP rank, EP8
# routed experts over the world). Shared derivation for the G=4 rows:
#
# KDA state is head-sharded, so a slot is 8 x 53.57 / 4 = 107.14 MiB per rank.
# The TP-sharded decode parameters (``K3Geometry.tp_sharded_decode_bytes``,
# 103.25 GiB summed over the rank) cost 25.81 GiB per rank at G=4 against
# 12.91 GiB at G=8, so the non-slot base is the MEASURED B300 G=8 base
# (193.85 GiB: resident EP8 shard 168.39 + decode parameters 16.06 + CUDA
# context and collectives) plus 12.91 GiB. The MLA latent stays replicated, so
# GPU KV per token is unchanged. Streamed prefill activations per DP-rank row
# per rank scale 8/G (``k3_prefill_activation_bytes_per_row``), so the
# streamed cap is the G=8 value halved and the resident cap is the largest
# power of two whose activations and FP32 output fit beside the shard.
_G4_BASE_DELTA_GIB = (
    K3_GEOMETRY.tp_sharded_decode_gib(4) - K3_GEOMETRY.tp_sharded_decode_gib(8)
)
_B300_G8_MEASURED_BASE_GIB = 208.13 - 273 * 53.57 / 1024
_G4_SLOT_GIB = K3_GEOMETRY.kda_slot_bytes(4) / _GIB


def _g4_prefill_non_expert_gib(user_slots: int) -> float:
    """First-iteration streamed-prefill use at G=4 (shard released).

    ESTIMATED: the measured H200 G=8 first-iteration use (39.96 GiB) plus the
    extra TP-sharded parameters of G=4 plus this row's KDA state pool.
    """
    return 39.96 + _G4_BASE_DELTA_GIB + (user_slots + 17) * _G4_SLOT_GIB


# GB300, 2 trays x 4 GPUs (world 8), one DP rank per tray. One tray (4 x
# 276.6 GiB) cannot hold the routed expert set, so the routed experts run EP8
# over both trays. Device total: nvidia-smi reports 284,208 MiB (MEASURED
# 2026-11-09); CUDA reports 938 MiB less on this generation (the B300 offset),
# 283,270 MiB = 276.63 GiB. At frac 0.95 the budget is 262.80 GiB; minus the
# 206.76 GiB G=4 base, a 2 GiB graph reserve and 17 non-user slots (1.78 GiB),
# 52.26 GiB remain. One sequence held to 4,096 tokens costs 107.14 MiB of KDA
# state + 4,096 x 27,648 B of MLA KV = 0.2101 GiB, so S = 248.7; the largest
# multiple of G=4 below it is 248. PROVISIONAL until measured on GB300.
_GB300_W8_G4 = K3EngineProfile(
    family=MachineFamily.GB300,
    num_nodes=2,
    attention_group_size=4,
    shard_latent_projections=True,
    shard_vocab_parallel=True,
    kda_user_slots=248,
    kda_slots_basis="kv-solve",
    prefill_micro_batch_token_cap=_H200_STREAMED_CAP // 2,
    resident_ep_prefill_token_cap=_H200_RESIDENT_CAP,
    resident_prefill_chunk_rows=_H200_RESIDENT_CHUNK_ROWS,
    prefill_grouped_chunk_rows=_H200_GROUPED_CHUNK_ROWS,
    prefill_collective_stripe_threshold_rows=_H200_STRIPE_THRESHOLD_ROWS,
    prefill_stream_span=8,
    prefix_pool_workspace_bytes=_H200_PREFIX_POOL_BYTES,
    decode_budget=K3DecodeBudget(
        device_total_gib=283_270 / 1024,
        gpu_memory_frac=0.95,
        non_slot_base_gib=_B300_G8_MEASURED_BASE_GIB + _G4_BASE_DELTA_GIB,
        prefill_non_expert_gib=_g4_prefill_non_expert_gib(248),
        graph_reserve_gib=2.0,
        full_length_tokens=4096,
        source=(
            "nvidia-smi total MEASURED 2026-11-09 (CUDA total ESTIMATED with "
            "the B300 offset); base ESTIMATED as the measured B300 G=8 base "
            "plus the G=4 share of the TP-sharded decode parameters"
        ),
    ),
    sources=MappingProxyType({
        "shard_latent_projections": (
            "derivation: every DP-attention row shards the LatentMoE "
            "projections over its TP-G rank (streamed default)"
        ),
        "shard_vocab_parallel": "provisional: vocab-parallel over the TP4 rank",
        "kda_user_slots": "provisional kv-solve (see the row comment)",
        "prefill_micro_batch_token_cap": (
            "provisional: the H200 cap scaled by G/8 for the per-rank "
            "activation of a TP4 rank"
        ),
        "resident_ep_prefill_token_cap": (
            "provisional: largest power of two whose activations and FP32 "
            "output fit beside the resident EP8 shard"
        ),
        "resident_prefill_chunk_rows": _INHERITED_H200,
        "prefill_grouped_chunk_rows": _INHERITED_H200,
        "prefill_collective_stripe_threshold_rows": _INHERITED_H200,
        "prefill_stream_span": _JOB_STREAM_SPAN,
        "prefix_pool_workspace_bytes": _INHERITED_H200,
    }),
)

# B300, 1 node x 8 as two TP4 DP ranks (world 8). Same G=4 derivation on the
# measured B300 device total: budget 254.30 GiB; minus the 206.76 GiB base, the
# graph reserve and 17 non-user slots, 43.76 GiB remain, S = 208.3 -> 208.
# The resident cap halves (8,192): the 16,384-row activations plus FP32
# output do not fit beside the shard in this budget. PROVISIONAL.
_B300_W8_G4 = K3EngineProfile(
    family=MachineFamily.B300,
    num_nodes=1,
    attention_group_size=4,
    shard_latent_projections=True,
    shard_vocab_parallel=True,
    kda_user_slots=208,
    kda_slots_basis="kv-solve",
    prefill_micro_batch_token_cap=_H200_STREAMED_CAP // 2,
    resident_ep_prefill_token_cap=_H200_RESIDENT_CAP // 2,
    resident_prefill_chunk_rows=_H200_RESIDENT_CHUNK_ROWS,
    prefill_grouped_chunk_rows=_H200_GROUPED_CHUNK_ROWS,
    prefill_collective_stripe_threshold_rows=_H200_STRIPE_THRESHOLD_ROWS,
    prefill_stream_span=8,
    prefix_pool_workspace_bytes=_H200_PREFIX_POOL_BYTES,
    decode_budget=K3DecodeBudget(
        device_total_gib=267.68,
        gpu_memory_frac=0.95,
        non_slot_base_gib=_B300_G8_MEASURED_BASE_GIB + _G4_BASE_DELTA_GIB,
        prefill_non_expert_gib=_g4_prefill_non_expert_gib(208),
        graph_reserve_gib=2.0,
        full_length_tokens=4096,
        source=(
            "ESTIMATED: the measured B300 G=8 base plus the G=4 share of the "
            "TP-sharded decode parameters"
        ),
    ),
    sources=MappingProxyType({
        "shard_latent_projections": (
            "derivation: every DP-attention row shards the LatentMoE "
            "projections over its TP-G rank (streamed default)"
        ),
        "shard_vocab_parallel": "provisional: vocab-parallel over the TP4 rank",
        "kda_user_slots": "provisional kv-solve (see the row comment)",
        "prefill_micro_batch_token_cap": (
            "provisional: the H200 cap scaled by G/8 for the per-rank "
            "activation of a TP4 rank"
        ),
        "resident_ep_prefill_token_cap": (
            "provisional: largest power of two whose activations and FP32 "
            "output fit beside the resident EP8 shard"
        ),
        "resident_prefill_chunk_rows": _INHERITED_H200,
        "prefill_grouped_chunk_rows": _INHERITED_H200,
        "prefill_collective_stripe_threshold_rows": _INHERITED_H200,
        "prefill_stream_span": _JOB_STREAM_SPAN,
        "prefix_pool_workspace_bytes": _INHERITED_H200,
    }),
)

K3ProfileKey = tuple[MachineFamily, int, int]

K3_PROFILES: Mapping[K3ProfileKey, K3EngineProfile] = MappingProxyType({
    (row.family, row.world_size, row.attention_group_size): row
    for row in (_H20_W32, _H200_W16, _B300_W8, _B300_W8_G4, _GB300_W8_G4)
})


def k3_engine_profile(
    family: MachineFamily, world_size: int, attention_group_size: int,
) -> K3EngineProfile:
    """The K3 profile row of ``family`` at ``world_size`` GPUs and DP-attention
    rank size ``attention_group_size``.

    Raises ``UnsupportedK3TopologyError`` for a triple without a row.
    """
    key = (MachineFamily(family), int(world_size), int(attention_group_size))
    row = K3_PROFILES.get(key)
    if row is None:
        supported = ", ".join(
            f"{fam.value} x {world} G={group}" for fam, world, group in sorted(
                K3_PROFILES, key=lambda k: (k[0].value, k[1], k[2]))
        )
        raise UnsupportedK3TopologyError(
            f"no K3 profile for machine family {key[0].value!r} at "
            f"world_size={key[1]} with attention_group_size={key[2]}; "
            f"supported: {supported}"
        )
    return row


# Kimi-Linear-48B in-wave prefix-pool reserve, keyed by (machine family,
# DP-attention rank size G). The pool is an optional optimization: a layout
# without an entry runs with the pool off (reserve 0), never with another
# layout's reserve. The 48B runs the same resident head-parallel KDA and
# NoPE-MLA seams as K3 with identical per-head widths and strictly smaller
# temporaries (27 vs 93 layers, resident experts), so the K3 reserve of the
# same machine bounds it. The h20 G=8 entry was validated live on a 2x8 serve
# group (full in-wave pool + trajectory materialization matrix, zero
# out-of-memory, exact per-wave token accounting). The Blackwell entries carry
# the value they ran under the retired memory tiers and are provisional.
KL48_PREFIX_POOL_WORKSPACE_BYTES: Mapping[tuple[MachineFamily, int], int] = MappingProxyType({
    (MachineFamily.H20, 8): 24 * _GIB,
    (MachineFamily.H200, 8): 40 * _GIB,
    (MachineFamily.B200, 8): 40 * _GIB,
    (MachineFamily.B300, 8): 40 * _GIB,
    (MachineFamily.GB300, 8): 40 * _GIB,
})
