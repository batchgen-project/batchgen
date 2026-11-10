"""Sizing table for the in-wave prefill prefix pool reserve.

The reserve is a pure table over (machine family, attention group size,
model family, world size); these tests pin every row so a resize is a
deliberate edit here, never a silent behavior change at serving time.
"""

import pytest

from batchgen.hardware.machine_family import MachineFamily
from batchgen.models.moonshotai.kimi_linear.machine_profile import (
    KL48_PREFIX_POOL_WORKSPACE_BYTES,
    UnsupportedK3TopologyError,
    k3_engine_profile,
)
from batchgen.models.moonshotai.kimi_linear.planner import (
    kimi_prefix_pool_workspace_bytes,
)

GIB = 1024 ** 3


def _reserve(family, *, is_k3, group=8, world=16):
    return kimi_prefix_pool_workspace_bytes(
        machine_family=family,
        attention_group_size=group,
        is_k3=is_k3,
        world_size=world,
    )


def test_h200_tp8_keeps_the_k3_sized_reserve_for_both_families():
    assert _reserve(MachineFamily.H200, is_k3=True, world=16) == 40 * GIB
    assert _reserve(MachineFamily.H200, is_k3=False) == 40 * GIB


def test_h20_tp8_sizes_each_family_with_its_own_row():
    assert _reserve(MachineFamily.H20, is_k3=False) == 24 * GIB
    assert _reserve(MachineFamily.H20, is_k3=True, world=32) == 24 * GIB


def test_k3_h20_reserve_is_sized_and_below_the_h200_reserve():
    # The H20 token cap shrinks K3's peak terms, so its reserve must be a
    # positive value strictly below the H200 one.
    h20 = k3_engine_profile(MachineFamily.H20, 32, 8).prefix_pool_workspace_bytes
    h200 = k3_engine_profile(MachineFamily.H200, 16, 8).prefix_pool_workspace_bytes
    assert 0 < h20 < h200


def test_every_family_has_a_48b_row_at_g8():
    assert set(KL48_PREFIX_POOL_WORKSPACE_BYTES) == {
        (family, 8) for family in MachineFamily
    }


def test_pure_dp_layouts_are_pool_off_without_a_machine_family():
    for is_k3 in (True, False):
        assert _reserve(None, is_k3=is_k3, group=1) == 0


def test_dp_attention_layouts_need_the_machine_family():
    for group in (2, 4, 16):
        for is_k3 in (True, False):
            with pytest.raises(ValueError, match="machine family"):
                _reserve(None, is_k3=is_k3, group=group)


def test_k3_g4_reserve_comes_from_its_profile_row():
    assert _reserve(MachineFamily.GB300, is_k3=True, group=4, world=8) == (
        k3_engine_profile(MachineFamily.GB300, 8, 4).prefix_pool_workspace_bytes
    )


def test_48b_without_a_recorded_reserve_runs_pool_off():
    assert _reserve(MachineFamily.GB300, is_k3=False, group=4, world=8) == 0


def test_tp8_reserve_needs_the_machine_family():
    with pytest.raises(ValueError, match="machine family"):
        _reserve(None, is_k3=False)


def test_k3_tp8_reserve_needs_a_profile_row():
    with pytest.raises(UnsupportedK3TopologyError):
        _reserve(MachineFamily.H20, is_k3=True, world=16)


def test_nonpositive_group_size_is_refused():
    with pytest.raises(ValueError):
        _reserve(MachineFamily.H200, is_k3=False, group=0)
