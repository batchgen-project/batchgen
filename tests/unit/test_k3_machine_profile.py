"""K3 machine-profile rows: completeness, sources and the derivations.

The derivation tests recompute, from the recorded rates and memory inputs,
the weight-stream crossover and HBM use of every prefill cap and the GPU KV
solve behind kv-solved KDA slot counts. A knob that violates its derivation
must carry an explicit exception reason, and an exception on a knob that
satisfies its derivation is stale and fails too.
"""

import pytest

from batchgen.hardware.machine_family import MachineFamily, signature_for
from batchgen.models.moonshotai.kimi_linear.machine_profile import (
    K3_FAMILY_RATES,
    K3_GEOMETRY,
    K3_PREFILL_ACTIVATION_BYTES_PER_ROW,
    K3_PROFILES,
    UnsupportedK3TopologyError,
    k3_engine_profile,
    k3_prefill_activation_bytes_per_row,
    k3_profile_knob_names,
    solve_kda_user_slots,
)

GIB = 1024 ** 3
ROWS = list(K3_PROFILES.values())
IDS = [
    f"{row.family.value}-w{row.world_size}-g{row.attention_group_size}"
    for row in ROWS
]


def test_the_supported_topologies_are_exactly_these():
    assert sorted((fam.value, world, g) for fam, world, g in K3_PROFILES) == [
        ("b300", 8, 4), ("b300", 8, 8), ("gb300", 8, 4),
        ("h20", 32, 8), ("h200", 16, 8),
    ]


@pytest.mark.parametrize("row", ROWS, ids=IDS)
def test_row_is_complete_and_every_knob_is_sourced(row):
    assert K3_PROFILES[(row.family, row.world_size, row.attention_group_size)] is row
    assert row.world_size == row.num_nodes * signature_for(row.family).gpus_per_node
    # A DP rank never spans hosts; the stream span is whole DP ranks.
    assert signature_for(row.family).gpus_per_node % row.attention_group_size == 0
    assert row.prefill_stream_span % row.attention_group_size == 0
    assert row.world_size % row.prefill_stream_span == 0
    assert K3_GEOMETRY.routed_experts % row.prefill_stream_span == 0
    knobs = set(k3_profile_knob_names())
    assert set(row.sources) == knobs
    assert all(isinstance(text, str) and text.strip() for text in row.sources.values())
    assert set(row.derivation_exceptions) <= knobs
    for name in knobs:
        value = getattr(row, name)
        assert isinstance(value, (bool, int)), name
        if not isinstance(value, bool):
            assert value >= 0, name
    assert row.decode_budget.source.strip()


@pytest.mark.parametrize("row", ROWS, ids=IDS)
def test_every_row_prefills_streamed_by_default(row):
    # The parallel strategy manager's server prefill mode is streamed_sp8
    # whenever the LatentMoE projections are TP-sharded (POIS 2026-11-09:
    # every prefill default is streaming).
    assert row.shard_latent_projections is True


@pytest.mark.parametrize("row", ROWS, ids=IDS)
def test_resident_chunk_rows_exist_exactly_with_a_resident_cap(row):
    assert (row.resident_prefill_chunk_rows > 0) == (
        row.resident_ep_prefill_token_cap > 0
    )


def _exception(row, knob, needed, keyword):
    reason = row.derivation_exceptions.get(knob)
    if needed:
        assert reason is not None and keyword in reason, (
            f"{row.family.value}: {knob} violates its derivation without a "
            f"{keyword!r} exception"
        )
    return reason


@pytest.mark.parametrize("row", ROWS, ids=IDS)
def test_streamed_cap_against_crossover_and_hbm(row):
    rates = K3_FAMILY_RATES[row.family]
    crossover = rates.stream_seconds_per_layer / rates.compute_seconds_per_row
    cap = row.prefill_micro_batch_token_cap
    hbm_gib = (
        cap * k3_prefill_activation_bytes_per_row(row.attention_group_size) / GIB
    )
    budget = row.decode_budget
    free_gib = budget.budget_gib - budget.prefill_non_expert_gib

    stream_bound = cap < crossover
    overrun = hbm_gib > free_gib
    reason = row.derivation_exceptions.get("prefill_micro_batch_token_cap")
    if stream_bound:
        _exception(row, "prefill_micro_batch_token_cap", True, "stream-bound")
    if overrun:
        _exception(row, "prefill_micro_batch_token_cap", True, "tracked")
    if not (stream_bound or overrun):
        assert reason is None, f"stale exception on {row.family.value}: {reason}"


@pytest.mark.parametrize("row", ROWS, ids=IDS)
def test_resident_cap_fits_beside_the_resident_shard(row):
    cap = row.resident_ep_prefill_token_cap
    if cap == 0:
        assert "resident_ep_prefill_token_cap" not in row.derivation_exceptions
        return
    activation = cap * k3_prefill_activation_bytes_per_row(row.attention_group_size)
    fp32_output = row.world_size * cap * K3_GEOMETRY.hidden_size * 4
    need_gib = (activation + fp32_output) / GIB
    budget = row.decode_budget
    free_gib = (
        budget.budget_gib
        - K3_GEOMETRY.resident_shard_bytes(row.world_size) / GIB
        - budget.prefill_non_expert_gib
    )
    assert need_gib <= free_gib or "tracked" in row.derivation_exceptions.get(
        "resident_ep_prefill_token_cap", ""
    )


@pytest.mark.parametrize("row", ROWS, ids=IDS)
def test_kda_slots_follow_their_stated_basis(row):
    basis = row.kda_slots_basis
    if basis == "kv-solve":
        solved = solve_kda_user_slots(
            row.decode_budget,
            extra_physical_slots=1 + 16,
            attention_group_size=row.attention_group_size,
        )
        group = row.attention_group_size
        assert row.kda_user_slots <= solved
        # the largest G multiple at or below the solve
        assert row.kda_user_slots % group == 0
        assert row.kda_user_slots + group > solved
    else:
        assert basis.startswith(("measured:", "inherited:")), basis


def test_resident_shard_geometry_matches_the_measured_shards():
    # measured: 168.39 GiB/rank single-node EP8, 84.19 GiB/rank two-node EP16
    assert K3_GEOMETRY.resident_shard_bytes(8) / GIB == pytest.approx(168.39, abs=0.01)
    assert K3_GEOMETRY.resident_shard_bytes(16) / GIB == pytest.approx(84.19, abs=0.01)


def test_b300_slot_solve_reproduces_the_stated_derivation():
    row = k3_engine_profile(MachineFamily.B300, 8, 8)
    budget = row.decode_budget
    # measured at frac 0.85 with 273 physical slots: 208.13 GiB in use
    assert budget.non_slot_base_gib == pytest.approx(193.85, abs=0.01)
    assert budget.budget_gib == pytest.approx(254.30, abs=0.01)
    assert K3_GEOMETRY.kv_bytes_per_token * 4096 / GIB == pytest.approx(0.1055, abs=1e-4)
    solved = solve_kda_user_slots(
        budget, extra_physical_slots=17, attention_group_size=8
    )
    assert solved == pytest.approx(364.8, abs=0.1)
    assert row.kda_user_slots == 360
    assert "provisional" in row.sources["kda_user_slots"]


def test_g4_rows_reproduce_the_stated_derivation():
    # TP-sharded decode parameters summed over a DP rank: 103.25 GiB, so G=4
    # costs 12.91 GiB per rank more than G=8.
    assert K3_GEOMETRY.tp_sharded_decode_bytes / GIB == pytest.approx(103.25, abs=0.01)
    delta = K3_GEOMETRY.tp_sharded_decode_gib(4) - K3_GEOMETRY.tp_sharded_decode_gib(8)
    assert delta == pytest.approx(12.91, abs=0.01)
    assert K3_GEOMETRY.kda_slot_bytes(4) == 2 * K3_GEOMETRY.kda_slot_bytes(8)
    gb300 = k3_engine_profile(MachineFamily.GB300, 8, 4)
    assert gb300.decode_budget.non_slot_base_gib == pytest.approx(206.76, abs=0.01)
    assert gb300.decode_budget.budget_gib == pytest.approx(262.80, abs=0.01)
    assert solve_kda_user_slots(
        gb300.decode_budget, extra_physical_slots=17, attention_group_size=4
    ) == pytest.approx(248.7, abs=0.1)
    assert gb300.kda_user_slots == 248
    assert gb300.prefill_stream_span == 8
    b300 = k3_engine_profile(MachineFamily.B300, 8, 4)
    assert solve_kda_user_slots(
        b300.decode_budget, extra_physical_slots=17, attention_group_size=4
    ) == pytest.approx(208.3, abs=0.1)
    assert b300.kda_user_slots == 208
    assert "provisional" in gb300.sources["kda_user_slots"]
    assert "provisional" in b300.sources["kda_user_slots"]


def test_g8_activation_estimate_is_the_measured_one():
    assert k3_prefill_activation_bytes_per_row(8) == K3_PREFILL_ACTIVATION_BYTES_PER_ROW
    assert k3_prefill_activation_bytes_per_row(4) == 2 * K3_PREFILL_ACTIVATION_BYTES_PER_ROW


@pytest.mark.parametrize(
    "family, world, group",
    [
        (MachineFamily.H20, 32, 8),
        (MachineFamily.H200, 16, 8),
        (MachineFamily.B300, 8, 8),
    ],
)
def test_hopper_and_b300_tp8_rows_stream_over_their_node(family, world, group):
    # Byte-identical to the validated streamed-SP8 layout: one copy of the
    # experts per eight-GPU node.
    assert k3_engine_profile(family, world, group).prefill_stream_span == 8


def test_profile_lookup_refuses_unknown_triples_by_name():
    with pytest.raises(UnsupportedK3TopologyError, match="b200"):
        k3_engine_profile(MachineFamily.B200, 8, 8)
    with pytest.raises(UnsupportedK3TopologyError, match="world_size=4"):
        k3_engine_profile(MachineFamily.GB300, 4, 4)
    # A DP rank of 8 would span the two GB300 trays.
    with pytest.raises(UnsupportedK3TopologyError, match="attention_group_size=8"):
        k3_engine_profile(MachineFamily.GB300, 8, 8)
    with pytest.raises(UnsupportedK3TopologyError, match="attention_group_size=2"):
        k3_engine_profile(MachineFamily.H200, 16, 2)
