from collections import Counter

import pytest

from batchgen.weight_replica_group import (
    CrossNodeTopology,
    derive_cross_node_owner_map,
    derive_frame_plan,
    derive_intra_node_owner_map,
    derive_owner_map,
    compute_cross_node_groups,
    install_cross_record_kinds,
    install_record_kinds,
    stagger_task_for_member,
)


def _source_task(local_slot: int, *, layers: int = 3, num_experts: int = 16):
    experts_per_slot = num_experts // 8
    expert_start = local_slot * experts_per_slot
    experts = range(expert_start, expert_start + experts_per_slot)
    return {
        "attn": ["attn_0", "attn_2"],
        "kda_attn": ["kda_attn_1"],
        "shared_expert": [f"shared_expert_{layer}" for layer in range(layers)],
        "routed_expert": [
            f"routed_expert_{layer}_{expert}"
            for layer in range(layers)
            for expert in experts
        ],
    }


def test_cross_node_groups_cover_world_and_use_source_root_rule():
    topology = CrossNodeTopology(num_nodes=4, local_world=8)
    groups = compute_cross_node_groups(topology)

    assert set(groups) == set(range(32))
    expected_roots = (0, 1, 10, 11, 20, 21, 30, 31)
    for local_slot, expected_root in enumerate(expected_roots):
        group = groups[local_slot]
        assert group.local_slot == local_slot
        assert group.ranks == (
            local_slot,
            local_slot + 8,
            local_slot + 16,
            local_slot + 24,
        )
        assert group.source_root == expected_root
        assert all(groups[rank] == group for rank in group.ranks)


def test_k31_source_task_derives_frames_and_positions():
    task = _source_task(local_slot=3)
    plan = derive_frame_plan(task)

    assert plan is not None
    assert plan.layer_sequence == (0, 1, 2)
    assert plan.frame_size == {
        "attn": 1,
        "kda_attn": 1,
        "shared_expert": 1,
        "routed_expert": 2,
    }
    assert plan.position["attn_2"] == (2, None)
    assert plan.position["kda_attn_1"] == (1, None)
    assert plan.position["shared_expert_0"] == (0, None)
    assert plan.position["routed_expert_2_6"] == (2, 6)
    assert plan.position["routed_expert_2_7"] == (2, 7)


def test_cross_node_owner_map_covers_every_source_record():
    topology = CrossNodeTopology(num_nodes=4, local_world=8)
    groups = compute_cross_node_groups(topology)

    for local_slot in range(8):
        task = _source_task(local_slot)
        plan = derive_frame_plan(task)
        group = groups[local_slot]
        owners = derive_cross_node_owner_map(task, plan, group)
        expected_names = {
            name for names in task.values() for name in names
        }

        assert set(owners) == expected_names
        assert set(owners.values()) == {group.source_root}


def test_intra_node_owner_map_uses_contiguous_expert_blocks():
    for local_slot in range(8):
        task = _source_task(local_slot)
        owners = derive_intra_node_owner_map(
            task["routed_expert"], num_experts=16, local_world=8
        )
        assert derive_intra_node_owner_map(task, 16, 8) == owners
        assert set(owners.values()) == {local_slot}
        assert owners[f"routed_expert_0_{local_slot * 2}"] == local_slot
        assert owners[f"routed_expert_2_{local_slot * 2 + 1}"] == local_slot


def test_non_source_declaration_uses_source_task_and_matches_source():
    topology = CrossNodeTopology(num_nodes=4, local_world=8)
    groups = compute_cross_node_groups(topology)
    local_slot = 5
    group = groups[local_slot]
    source_rank = group.source_root
    non_source_rank = next(rank for rank in group.ranks if rank != source_rank)
    source_task = _source_task(local_slot)

    source_group = groups[source_rank]
    source_plan = derive_frame_plan(source_task)
    source_owners = derive_cross_node_owner_map(
        source_task, source_plan, source_group
    )

    rank_local_task = {
        module_type: (
            [] if module_type == "routed_expert" else list(names)
        )
        for module_type, names in source_task.items()
    }
    rank_local_plan = derive_frame_plan(rank_local_task)
    assert rank_local_task["routed_expert"] == []
    assert rank_local_plan != source_plan
    assert derive_cross_node_owner_map(
        rank_local_task, rank_local_plan, groups[non_source_rank]
    ) != source_owners

    declaration_plan = derive_frame_plan(source_task)
    declaration_owners = derive_cross_node_owner_map(
        source_task, declaration_plan, groups[non_source_rank]
    )
    assert groups[non_source_rank] == source_group
    assert declaration_plan == source_plan
    assert declaration_owners == source_owners


def test_stagger_is_a_permutation_and_source_records_are_first():
    topology = CrossNodeTopology(num_nodes=4, local_world=8)
    groups = compute_cross_node_groups(topology)
    task = _source_task(local_slot=5)
    plan = derive_frame_plan(task)
    group = groups[5]
    owners = derive_cross_node_owner_map(task, plan, group)

    for rank in group.ranks:
        staggered = stagger_task_for_member(task, plan, owners, group.ranks, rank)
        for module_type, names in task.items():
            assert Counter(staggered[module_type]) == Counter(names)
        assert derive_frame_plan(staggered) == plan

        if rank == group.source_root:
            for names in staggered.values():
                assert names
                assert owners[names[0]] == rank

    # The owner map is group-level, so a non-source member sees the same map.
    for rank in group.ranks:
        if rank == group.source_root:
            continue
        assert derive_cross_node_owner_map(task, plan, group) == owners


def test_existing_record_kinds_keep_source_host_and_non_source_replica():
    topology = CrossNodeTopology(num_nodes=4, local_world=8)
    group = compute_cross_node_groups(topology)[2]
    task = _source_task(local_slot=2)
    plan = derive_frame_plan(task)
    owners = derive_cross_node_owner_map(task, plan, group)

    source_kinds = install_record_kinds(group.source_root, group.ranks, owners)
    assert set(source_kinds.values()) == {"HostShared"}
    for rank in group.ranks:
        if rank == group.source_root:
            continue
        assert set(install_record_kinds(rank, group.ranks, owners).values()) == {
            "Replica"
        }


def test_cross_record_kinds_keep_source_local_rules_and_mark_remote_members():
    topology = CrossNodeTopology(num_nodes=4, local_world=8)
    groups = compute_cross_node_groups(topology)
    group = groups[0]
    task = _source_task(local_slot=0)
    plan = derive_frame_plan(task)
    intra_owners = derive_owner_map(
        tuple(range(8)),
        {
            module_type: names
            for module_type, names in task.items()
            if module_type != "routed_expert"
        },
        plan,
    )
    intra_owners.update(
        derive_intra_node_owner_map(task["routed_expert"], 16, 8)
    )

    source_kinds = install_cross_record_kinds(
        group.source_root, group, tuple(range(8)), intra_owners
    )
    assert source_kinds == install_record_kinds(
        group.source_root, tuple(range(8)), intra_owners
    )
    assert set(source_kinds.values()) == {"HostShared", "Replica"}

    for rank in (8, 16, 24):
        kinds = install_cross_record_kinds(
            rank, group, tuple(range(8)), intra_owners
        )
        assert set(kinds) == set(intra_owners)
        assert set(kinds.values()) == {"CrossReplica"}


def test_cross_record_kinds_rejects_rank_outside_group():
    topology = CrossNodeTopology(num_nodes=4, local_world=8)
    group = compute_cross_node_groups(topology)[0]
    with pytest.raises(ValueError, match="cross-node group"):
        install_cross_record_kinds(1, group, tuple(range(8)), {"attn_0": 0})


def test_cross_node_owner_map_rejects_unknown_names_and_types():
    topology = CrossNodeTopology(num_nodes=4, local_world=8)
    group = compute_cross_node_groups(topology)[0]
    task = _source_task(local_slot=0)
    plan = derive_frame_plan(task)

    malformed_name = dict(task)
    malformed_name["routed_expert"] = ["routed_expert_x_y"]
    malformed_plan = derive_frame_plan(malformed_name)
    with pytest.raises(ValueError):
        derive_cross_node_owner_map(malformed_name, malformed_plan, group)

    unknown_type = {"mystery": ["mystery_0"]}
    unknown_plan = derive_frame_plan(unknown_type)
    with pytest.raises(ValueError):
        derive_cross_node_owner_map(unknown_type, unknown_plan, group)


def test_cross_node_topology_rejects_empty_dimensions():
    with pytest.raises(ValueError):
        CrossNodeTopology(num_nodes=0, local_world=8)
    with pytest.raises(ValueError):
        CrossNodeTopology(num_nodes=4, local_world=0)
