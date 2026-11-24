"""Phase layout: the rank groups a phase's attention and MoE form (contract C1, C2)."""
import pytest

from batchgen.parallel_declaration import (
    ModuleParallelism as M,
    ParallelDeclarationError,
    PhaseLayout,
    ServerLayout,
)

DP_STREAMED = M("dp", "streamed")
DP_RESIDENT = M("dp", "resident")


def tp(n, residency="resident"):
    return M("tp", residency, n)


def ep(n, residency="streamed"):
    return M("ep", residency, n)


def test_the_dp_worker_is_the_larger_of_the_two_groups():
    # POIS's example: attention groups of 2, MoE groups of 4, 16 ranks.
    layout = PhaseLayout("prefill", tp(2), ep(4), 16)
    assert layout.dp_worker_size == 4 and layout.num_dp_workers == 4
    assert layout.dp_workers == ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15))
    assert len(layout.attention_groups) == 8 and len(layout.moe_groups) == 4
    # One DP worker holds two attention groups and one MoE group.
    assert layout.attention_group_of(6) == (6, 7)
    assert layout.moe_group_of(6) == (4, 5, 6, 7)
    assert layout.dp_worker_of(6) == (4, 5, 6, 7)


def test_pure_data_parallel_is_one_rank_per_worker():
    layout = PhaseLayout("prefill", DP_STREAMED, DP_STREAMED, 8)
    assert layout.dp_worker_size == 1 and layout.num_dp_workers == 8
    assert layout.dp_worker_of(5) == layout.attention_group_of(5) == (5,)


@pytest.mark.parametrize("attention,moe,world", [
    (tp(8), ep(8), 16),          # today's Kimi streamed prefill with 8-way shards
    (DP_RESIDENT, ep(8, "resident"), 8),
    (tp(8), DP_STREAMED, 32),    # attention wider than the MoE
    (tp(2), ep(8), 16),
])
def test_groups_tile_the_world_and_nest(attention, moe, world):
    layout = PhaseLayout("decode", attention, moe, world)
    for groups in (layout.attention_groups, layout.moe_groups, layout.dp_workers):
        assert sorted(rank for group in groups for rank in group) == list(range(world))
    for rank in range(world):
        worker = set(layout.dp_worker_of(rank))
        # No group crosses a DP worker: sequences and collectives stay inside it.
        assert set(layout.attention_group_of(rank)) <= worker
        assert set(layout.moe_group_of(rank)) <= worker


@pytest.mark.parametrize("attention,moe,world,message", [
    (tp(2), ep(3), 12, "must divide one another"),
    (tp(4), ep(8), 12, "does not divide the world size"),
    (tp(8), ep(8), 0, "world_size must be a positive int"),
    (M("none", None), ep(8), 8, "attention cannot be 'none'"),
])
def test_an_impossible_layout_is_refused_by_name(attention, moe, world, message):
    with pytest.raises(ParallelDeclarationError, match=message):
        PhaseLayout("prefill", attention, moe, world)


def test_a_dense_model_has_no_moe_groups_to_cross():
    layout = PhaseLayout("decode", tp(4), M("none", None), 8)
    assert layout.dp_worker_size == 4
    assert layout.render() == "attention=tp4:resident moe=none"


def test_a_rank_outside_the_world_is_refused():
    layout = PhaseLayout("prefill", DP_STREAMED, DP_STREAMED, 8)
    for rank in (-1, 8, True, "0"):
        with pytest.raises(ParallelDeclarationError, match="outside the world"):
            layout.dp_worker_of(rank)


def test_each_phase_owns_its_own_layout_and_the_report_comes_from_it():
    server = ServerLayout(
        prefill=PhaseLayout("prefill", DP_STREAMED, DP_STREAMED, 16),
        decode=PhaseLayout("decode", tp(8), ep(8, "resident"), 16),
    )
    assert server.phase("prefill").dp_worker_size == 1
    assert server.phase("decode").dp_worker_size == 8
    assert server.declaration().rendered_slots() == {
        "prefill_attention": "dp:streamed",
        "prefill_moe": "dp:streamed",
        "decode_attention": "tp8:resident",
        "decode_moe": "ep8:resident",
    }


@pytest.mark.parametrize("prefill,decode,message", [
    (PhaseLayout("decode", DP_STREAMED, DP_STREAMED, 8),
     PhaseLayout("decode", DP_RESIDENT, DP_RESIDENT, 8), "must be the prefill PhaseLayout"),
    (PhaseLayout("prefill", DP_STREAMED, DP_STREAMED, 8),
     PhaseLayout("decode", DP_RESIDENT, DP_RESIDENT, 16), "same world"),
    (PhaseLayout("prefill", DP_STREAMED, M("none", None), 8),
     PhaseLayout("decode", DP_RESIDENT, DP_RESIDENT, 8), "both phases or in neither"),
])
def test_an_inconsistent_server_layout_is_refused(prefill, decode, message):
    with pytest.raises(ParallelDeclarationError, match=message):
        ServerLayout(prefill=prefill, decode=decode)
