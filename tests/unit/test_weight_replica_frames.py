"""Frame derivation from installed module names (spec v4, decision 2026-11-10).

The worker derives the frame layout from the installed weight-copy task's
module names instead of a new manager declaration. A task that does not fit
the layer grammar yields None, and the caller forms single-member groups.
"""

import pytest

from batchgen.weight_replica_group import (
    derive_frame_plan,
    derive_owner_map,
    install_record_kinds,
)


def _glm_task(layers=4, experts=8):
    return {
        "attn": [f"attn_{l}" for l in range(layers)],
        "shared_expert": [f"shared_expert_{l}" for l in range(layers)],
        "routed_expert": [
            f"routed_expert_{l}_{e}" for l in range(layers) for e in range(experts)
        ],
    }


def test_glm_shaped_task_derives_frames():
    plan = derive_frame_plan(_glm_task())
    assert plan is not None
    assert plan.layer_sequence == (0, 1, 2, 3)
    assert plan.frame_size == {
        "attn": 1,
        "shared_expert": 1,
        "routed_expert": 8,
    }
    assert plan.position["routed_expert_2_5"] == (2, 5)
    assert plan.position["attn_3"] == (3, None)


def test_partial_coverage_unions_the_layer_sequence():
    # GLM's first dense layers carry attention but no experts: the pass is
    # the ordered union and dense frames simply hold fewer records.
    task = _glm_task()
    task["attn"] = ["attn_0", "attn_1", "attn_2", "attn_9"]
    plan = derive_frame_plan(task)
    assert plan is not None
    assert plan.layer_sequence == (0, 1, 2, 3, 9)
    assert plan.position["attn_9"] == (4, None)
    assert plan.position["routed_expert_3_0"] == (3, 0)


def test_glm_dense_prefix_shape_derives_frames():
    task = {
        "attn": [f"attn_{l}" for l in range(4)],
        "shared_expert": ["shared_expert_3"],
        "routed_expert": [f"routed_expert_3_{e}" for e in range(8)],
    }
    plan = derive_frame_plan(task)
    assert plan is not None
    assert plan.layer_sequence == (0, 1, 2, 3)
    assert plan.frame_size == {
        "attn": 1,
        "shared_expert": 1,
        "routed_expert": 8,
    }
    assert plan.position["routed_expert_3_5"] == (3, 5)


def test_out_of_order_layers_yield_none():
    task = _glm_task()
    task["attn"] = ["attn_1", "attn_0", "attn_2", "attn_3"]
    assert derive_frame_plan(task) is None


def test_non_grammar_names_yield_none():
    assert derive_frame_plan({"routed_expert": ["weird-name"]}) is None


def test_ragged_frames_yield_none():
    task = {
        "routed_expert": [
            "routed_expert_0_0",
            "routed_expert_0_1",
            "routed_expert_1_0",
        ]
    }
    assert derive_frame_plan(task) is None


def test_owner_map_blocks_and_round_robin():
    task = _glm_task(layers=2, experts=8)
    plan = derive_frame_plan(task)
    owners = derive_owner_map([10, 11], task, plan)
    # routed experts: contiguous blocks of 4 per member
    assert owners["routed_expert_0_0"] == 10
    assert owners["routed_expert_0_3"] == 10
    assert owners["routed_expert_0_4"] == 11
    assert owners["routed_expert_1_7"] == 11
    # single-module types: layer round-robin
    assert owners["attn_0"] == 10 and owners["attn_1"] == 11
    assert owners["shared_expert_0"] == 10 and owners["shared_expert_1"] == 11


def test_owner_map_rejects_indivisible_experts():
    task = _glm_task(layers=1, experts=9)
    plan = derive_frame_plan(task)
    with pytest.raises(ValueError, match="not divisible"):
        derive_owner_map([0, 1], task, plan)


def test_record_kinds_single_member_is_all_host_private():
    task = _glm_task(layers=2, experts=8)
    plan = derive_frame_plan(task)
    owners = derive_owner_map([3], task, plan)
    kinds = install_record_kinds(3, [3], owners)
    assert set(kinds.values()) == {"HostPrivate"}


def test_record_kinds_group_of_two():
    task = _glm_task(layers=2, experts=8)
    plan = derive_frame_plan(task)
    owners = derive_owner_map([0, 1], task, plan)
    kinds = install_record_kinds(0, [0, 1], owners)
    assert kinds["routed_expert_0_0"] == "HostShared"
    assert kinds["routed_expert_0_4"] == "Replica"
    assert kinds["attn_1"] == "Replica"


def test_stagger_puts_own_block_first_and_rotates():
    from batchgen.weight_replica_group import stagger_task_for_member

    task = _glm_task(layers=2, experts=8)
    plan = derive_frame_plan(task)
    group = [0, 1, 2, 3]
    owners = derive_owner_map(group, task, plan)

    for rank in group:
        staggered = stagger_task_for_member(task, plan, owners, group, rank)
        # Same record set per type; grammar (and the canonical plan) intact.
        for module_type in task:
            assert sorted(staggered[module_type]) == sorted(task[module_type])
        assert derive_frame_plan(staggered) == plan
        # Within each layer, the member's own block leads and the owners
        # follow round-robin from the next member.
        routed = staggered["routed_expert"]
        per_layer = len(routed) // 2
        for layer_slice in (routed[:per_layer], routed[per_layer:]):
            owner_walk = [group.index(owners[name]) for name in layer_slice]
            me = group.index(rank)
            expected = [
                (me + step) % len(group)
                for step in range(len(group))
                for _ in range(per_layer // len(group))
            ]
            assert owner_walk == expected
        # Idempotent: staggering a staggered task changes nothing.
        assert stagger_task_for_member(
            staggered, plan, owners, group, rank
        ) == staggered

    # Members disagree on walk order (that is the point).
    assert stagger_task_for_member(
        task, plan, owners, group, 0
    ) != stagger_task_for_member(task, plan, owners, group, 1)


def test_stagger_leaves_single_record_layers_alone():
    from batchgen.weight_replica_group import stagger_task_for_member

    task = _glm_task(layers=3, experts=8)
    plan = derive_frame_plan(task)
    group = [0, 1]
    owners = derive_owner_map(group, task, plan)
    staggered = stagger_task_for_member(task, plan, owners, group, 1)
    assert staggered["attn"] == task["attn"]
    assert staggered["shared_expert"] == task["shared_expert"]
