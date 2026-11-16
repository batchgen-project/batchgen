from batchgen.weight_replica_group import derive_frame_plan
from batchgen.weight_replica_transport import (
    CrossReplicaLandingBook,
    plan_broadcast_frames,
    submit_broadcast_frames,
    mark_completed_submission,
)


def _task():
    return {
        "attn": ["attn_0", "attn_1"],
        "routed_expert": [
            "routed_expert_0_0",
            "routed_expert_0_1",
            "routed_expert_1_0",
            "routed_expert_1_1",
        ],
    }


def test_broadcast_frames_are_layer_major_and_type_stable():
    task = _task()
    plan = derive_frame_plan(task)
    assert plan is not None

    planned = plan_broadcast_frames(
        task, plan, rank=9, source_root=1, pass_id=2
    )

    assert planned.source_root == 1
    assert not planned.is_source
    assert [frame.frame for frame in planned.frames] == [2, 3]
    assert [
        (record.module, record.record_index)
        for record in planned.frames[0].records
    ] == [
        ("attn_0", 0),
        ("routed_expert_0_0", 1),
        ("routed_expert_0_1", 2),
    ]
    assert [
        (record.module, record.record_index)
        for record in planned.frames[1].records
    ] == [
        ("attn_1", 0),
        ("routed_expert_1_0", 1),
        ("routed_expert_1_1", 2),
    ]


def test_synthetic_transport_receives_every_frame_in_order():
    task = _task()
    plan = derive_frame_plan(task)
    assert plan is not None
    planned = plan_broadcast_frames(task, plan, rank=1, source_root=1)
    calls = []

    def transport(records, group, stream):
        calls.append((group, stream, [record.module for record in records]))
        return {"frame": len(calls)}

    submissions = submit_broadcast_frames(
        planned, transport=transport, group="group", stream="stream"
    )

    assert [call[2] for call in calls] == [
        ["attn_0", "routed_expert_0_0", "routed_expert_0_1"],
        ["attn_1", "routed_expert_1_0", "routed_expert_1_1"],
    ]
    assert all(call[:2] == ("group", "stream") for call in calls)
    assert [submission.handle for submission in submissions] == [
        {"frame": 1},
        {"frame": 2},
    ]


def test_landing_book_accepts_completion_before_frame_registration():
    landing = CrossReplicaLandingBook()
    landing.record_landing(7, 2)
    landing.record_landing(7, 1)
    landing.register_frame(7, [1, 2])

    assert landing.frame_landed(7)
    assert landing.landed_records(7) == frozenset({1, 2})


def test_completion_marks_every_cross_record_and_rejects_duplicates():
    task = _task()
    plan = derive_frame_plan(task)
    assert plan is not None
    planned = plan_broadcast_frames(task, plan, rank=9, source_root=1)
    submission = submit_broadcast_frames(
        planned,
        transport=lambda records, group, stream: None,
        group=None,
        stream=None,
    )[0]
    landing = CrossReplicaLandingBook()
    mark_completed_submission(submission, landing)

    assert landing.frame_landed(submission.frame.frame)



def _slot_task(layers=2, experts_per_rank=2, local_world=4):
    experts = experts_per_rank * local_world
    return {
        "attn": [f"attn_{l}" for l in range(layers)],
        "routed_expert": [
            f"routed_expert_{l}_{e}" for l in range(layers) for e in range(experts)
        ],
    }


def test_build_cross_node_plan_source_root_off_node_zero():
    from batchgen.weight_replica_transport import build_cross_node_plan

    # N=4 nodes x G=8 slots: local slot 2's source root is rank 10 (node 1).
    task = _slot_task(layers=2, experts_per_rank=2, local_world=8)
    source = build_cross_node_plan(
        task, num_nodes=4, local_world=8, rank=10, num_experts=16
    )
    assert source.group.local_slot == 2
    assert source.group.source_root == 10
    assert source.group.ranks == (2, 10, 18, 26)
    # The source root keeps intra-node kinds (owners live on ITS node, 8..15).
    assert set(source.kinds.values()) <= {"HostShared", "HostPrivate", "Replica"}
    assert any(kind in ("HostShared", "HostPrivate")
               for kind in source.kinds.values())
    assert source.broadcast.is_source

    member = build_cross_node_plan(
        task, num_nodes=4, local_world=8, rank=26, num_experts=16
    )
    assert set(member.kinds.values()) == {"CrossReplica"}
    assert not member.broadcast.is_source
    assert [frame.frame for frame in member.broadcast.frames] == [0, 1]
    # Landing book covers exactly the frame's record indices.
    frame0 = member.broadcast.frames[0]
    member.landing.register_frame(
        frame0.frame, [record.record_index for record in frame0.records]
    )
    for record in frame0.records:
        member.landing.record_landing(frame0.frame, record.record_index)
    assert member.landing.frame_landed(frame0.frame)


def test_build_cross_node_plan_rejects_bad_grammar():
    import pytest
    from batchgen.weight_replica_transport import build_cross_node_plan

    with pytest.raises(ValueError, match="layer grammar"):
        build_cross_node_plan(
            {"attn": ["not_a_layer_name"]}, num_nodes=2, local_world=4, rank=0
        )


def test_broadcast_driver_sequences_frames_across_passes():
    import pytest
    from batchgen.weight_replica_transport import BroadcastDriver

    task = _task()
    plan = derive_frame_plan(task)
    planned = plan_broadcast_frames(task, plan, rank=0, source_root=0)
    landing = CrossReplicaLandingBook()
    driver = BroadcastDriver(planned, landing, layers_per_pass=2)
    assert driver.source_root == 0

    # Pass 1: the planned frames, in order.
    first = driver.next_frame()
    assert first.frame == 0
    driver.complete(first)
    assert landing.frame_landed(0)
    second = driver.next_frame()
    assert second.frame == 1
    driver.complete(second)

    # Pass 2: the same canonical records, frame ids offset by a whole pass.
    third = driver.next_frame()
    assert third.frame == 2
    assert [r.record_index for r in third.records] == [
        r.record_index for r in first.records
    ]
    assert {r.module for r in third.records} == {r.module for r in first.records}
    assert all(r.frame == 2 for r in third.records)
    driver.complete(third)
    assert landing.frame_landed(2)
    # A double completion of one frame instance is a terminal-record error.
    with pytest.raises(ValueError, match="already registered"):
        driver.complete(third)


def test_broadcast_driver_rejects_short_pass_span():
    import pytest
    from batchgen.weight_replica_transport import BroadcastDriver

    task = _task()
    plan = derive_frame_plan(task)
    planned = plan_broadcast_frames(task, plan, rank=0, source_root=0)
    with pytest.raises(ValueError, match="layers_per_pass"):
        BroadcastDriver(planned, CrossReplicaLandingBook(), layers_per_pass=1)


def test_build_cross_node_plan_slot_shard_uses_global_expert_count():
    import pytest
    from batchgen.weight_replica_transport import build_cross_node_plan

    # A slot-2 shard of a 16-expert, 8-slot model: global experts 4..5 only.
    task = {
        "attn": ["attn_0", "attn_1"],
        "routed_expert": [
            f"routed_expert_{l}_{e}" for l in range(2) for e in (4, 5)
        ],
    }
    plan = build_cross_node_plan(
        task, num_nodes=4, local_world=8, rank=2, num_experts=16
    )
    assert plan.broadcast.frames
    # Without the whole-model count the owner rule cannot be applied.
    with pytest.raises(ValueError, match="num_experts"):
        build_cross_node_plan(task, num_nodes=4, local_world=8, rank=2)


def test_cross_plan_covers_only_routed_experts_on_full_task_numbering():
    from batchgen.weight_replica_group import derive_frame_plan
    from batchgen.weight_replica_transport import build_cross_node_plan

    # Layer 0 is dense (attention only); routed experts start at layer 1.
    task = {
        "attn": ["attn_0", "attn_1", "attn_2"],
        "routed_expert": [
            f"routed_expert_{l}_{e}" for l in (1, 2) for e in (4, 5)
        ],
    }
    member = build_cross_node_plan(
        task, num_nodes=4, local_world=8, rank=26, num_experts=16
    )
    # Only routed experts cross nodes; attention stays per-node.
    assert set(member.kinds) == set(task["routed_expert"])
    records = [r for f in member.broadcast.frames for r in f.records]
    assert {r.module_type for r in records} == {"routed_expert"}
    # Frame ids and record offsets follow the FULL task's plan, so they
    # match the engine's numbering (layer 0 has no routed frame).
    full = derive_frame_plan(task)
    assert [f.frame for f in member.broadcast.frames] == [
        full.position["routed_expert_1_4"][0],
        full.position["routed_expert_2_4"][0],
    ]
    routed_offset = sum(
        full.frame_size[t] for t in sorted(full.frame_size) if t < "routed_expert"
    )
    assert min(r.record_index for r in records) == routed_offset


def test_in_frame_positions_contiguous_and_global_blocks():
    from batchgen.weight_replica_group import derive_frame_plan
    from batchgen.weight_replica_transport import in_frame_positions

    contiguous = {"routed_expert": [f"routed_expert_{l}_{e}" for l in (0, 1) for e in range(4)]}
    pos = in_frame_positions(contiguous, derive_frame_plan(contiguous))
    # Contiguous 0..E-1 experts: position equals the expert id (GLM shape).
    assert all(pos[f"routed_expert_{l}_{e}"] == e for l in (0, 1) for e in range(4))

    block = {"routed_expert": [f"routed_expert_{l}_{e}" for l in (0, 1) for e in (112, 113, 114)]}
    pos = in_frame_positions(block, derive_frame_plan(block))
    # A slot's block of global ids maps to 0..block-1 in every frame.
    assert [pos[f"routed_expert_0_{e}"] for e in (112, 113, 114)] == [0, 1, 2]
    assert [pos[f"routed_expert_1_{e}"] for e in (112, 113, 114)] == [0, 1, 2]
