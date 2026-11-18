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
    K3_PROFILE_ATTENTION_GROUP_SIZE,
    K3_PROFILES,
    K3_STREAMED_PREFILL_CAP_QUANTUM,
    K3_STREAMED_PREFILL_HEADROOM_GIB,
    K3DecodeBudget,
    UnsupportedK3TopologyError,
    k3_engine_profile,
    k3_profile_knob_names,
    solve_kda_user_slots,
    solve_streamed_prefill_cap,
)

GIB = 1024 ** 3
ROWS = list(K3_PROFILES.values())
IDS = [f"{row.family.value}-w{row.world_size}" for row in ROWS]


def test_the_supported_topologies_are_exactly_these():
    assert sorted((fam.value, world) for fam, world in K3_PROFILES) == [
        ("b300", 8), ("gb300", 8), ("h20", 32), ("h200", 16), ("h200", 32),
    ]


@pytest.mark.parametrize("row", ROWS, ids=IDS)
def test_row_is_complete_and_every_knob_is_sourced(row):
    assert K3_PROFILES[(row.family, row.world_size)] is row
    assert row.world_size == row.num_nodes * signature_for(row.family).gpus_per_node
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
    assert K3_PROFILE_ATTENTION_GROUP_SIZE == 8


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
    hbm_gib = cap * K3_PREFILL_ACTIVATION_BYTES_PER_ROW / GIB
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


def _budget(**overrides):
    values = dict(
        device_total_gib=100.0, gpu_memory_frac=1.0, non_slot_base_gib=50.0,
        prefill_non_expert_gib=20.0, graph_reserve_gib=2.0,
        full_length_tokens=4096, source="test",
    )
    values.update(overrides)
    return K3DecodeBudget(**values)


def test_streamed_cap_solve_is_the_budget_identity():
    budget = _budget()
    cap = solve_streamed_prefill_cap(budget)
    free_bytes = (100.0 - 20.0 - K3_STREAMED_PREFILL_HEADROOM_GIB) * GIB
    # whole KDA segments, the largest count that fits, and not one more
    assert cap % K3_STREAMED_PREFILL_CAP_QUANTUM == 0
    assert cap * K3_PREFILL_ACTIVATION_BYTES_PER_ROW <= free_bytes
    assert (
        (cap + K3_STREAMED_PREFILL_CAP_QUANTUM)
        * K3_PREFILL_ACTIVATION_BYTES_PER_ROW
        > free_bytes
    )
    # more memory in use before the pass -> a smaller cap, never the same one
    assert solve_streamed_prefill_cap(_budget(prefill_non_expert_gib=40.0)) < cap
    assert solve_streamed_prefill_cap(_budget(gpu_memory_frac=0.9)) < cap


def test_streamed_cap_solve_refuses_a_budget_below_one_segment():
    with pytest.raises(ValueError, match="less than one KDA segment"):
        solve_streamed_prefill_cap(_budget(prefill_non_expert_gib=96.0))


@pytest.mark.parametrize("world_size", [16, 32])
def test_h200_rows_solve_their_streamed_cap_from_the_budget(world_size):
    row = k3_engine_profile(MachineFamily.H200, world_size)
    assert row.prefill_micro_batch_token_cap == solve_streamed_prefill_cap(
        row.decode_budget
    )
    # The cap that ran out of memory on both topologies is no longer planned,
    # and no exception is needed to carry a solved cap.
    assert row.prefill_micro_batch_token_cap < 518_718
    assert "prefill_micro_batch_token_cap" not in row.derivation_exceptions
    assert "solve_streamed_prefill_cap" in row.sources["prefill_micro_batch_token_cap"]


def test_h200_four_node_row_states_what_is_measured_and_what_is_inherited():
    row = k3_engine_profile(MachineFamily.H200, 32)
    two_node = k3_engine_profile(MachineFamily.H200, 16)
    assert row.num_nodes == 4
    assert row.decode_budget.prefill_non_expert_gib == pytest.approx(39.55, abs=0.01)
    # an EP32 rank holds half the EP16 resident shard
    assert two_node.decode_budget.non_slot_base_gib - row.decode_budget.non_slot_base_gib == (
        pytest.approx(42.10, abs=0.01)
    )
    assert row.kda_slots_basis.startswith("inherited:")
    for knob in ("kda_user_slots", "resident_ep_prefill_token_cap",
                 "prefix_pool_workspace_bytes"):
        assert getattr(row, knob) == getattr(two_node, knob)
        assert "provisional" in row.sources[knob]


@pytest.mark.parametrize("row", ROWS, ids=IDS)
def test_resident_cap_fits_beside_the_resident_shard(row):
    cap = row.resident_ep_prefill_token_cap
    if cap == 0:
        assert "resident_ep_prefill_token_cap" not in row.derivation_exceptions
        return
    activation = cap * K3_PREFILL_ACTIVATION_BYTES_PER_ROW
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
        )
        assert row.kda_user_slots <= solved
        # the largest TP-size multiple at or below the solve
        assert row.kda_user_slots % 8 == 0
        assert row.kda_user_slots + 8 > solved
    else:
        assert basis.startswith(("measured:", "inherited:")), basis


def test_resident_shard_geometry_matches_the_measured_shards():
    # measured: 168.39 GiB/rank single-node EP8, 84.19 GiB/rank two-node EP16
    assert K3_GEOMETRY.resident_shard_bytes(8) / GIB == pytest.approx(168.39, abs=0.01)
    assert K3_GEOMETRY.resident_shard_bytes(16) / GIB == pytest.approx(84.19, abs=0.01)


def test_b300_slot_solve_reproduces_the_stated_derivation():
    row = k3_engine_profile(MachineFamily.B300, 8)
    budget = row.decode_budget
    # measured at frac 0.85 with 273 physical slots: 208.13 GiB in use
    assert budget.non_slot_base_gib == pytest.approx(193.85, abs=0.01)
    assert budget.budget_gib == pytest.approx(254.30, abs=0.01)
    assert K3_GEOMETRY.kv_bytes_per_token * 4096 / GIB == pytest.approx(0.1055, abs=1e-4)
    solved = solve_kda_user_slots(budget, extra_physical_slots=17)
    assert solved == pytest.approx(364.8, abs=0.1)
    assert row.kda_user_slots == 360
    assert "provisional" in row.sources["kda_user_slots"]


def test_profile_lookup_refuses_unknown_pairs_by_name():
    with pytest.raises(UnsupportedK3TopologyError, match="b200"):
        k3_engine_profile(MachineFamily.B200, 8)
    with pytest.raises(UnsupportedK3TopologyError, match="world_size=4"):
        k3_engine_profile(MachineFamily.GB300, 4)
