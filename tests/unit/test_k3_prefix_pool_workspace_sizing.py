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
    streamed_prefill_workspace_bytes,
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


def test_h200_tp8_sizes_k3_for_a_cap_sized_pass_and_keeps_the_48b_reserve():
    # K3: one 262,144-row pass at the measured bytes per row plus headroom,
    # the same on both H200 topologies (machine_profile
    # streamed_prefill_workspace_bytes). The 48B keeps its own table value.
    k3 = streamed_prefill_workspace_bytes(262_144)
    assert _reserve(MachineFamily.H200, is_k3=True, world=16) == k3
    assert _reserve(MachineFamily.H200, is_k3=True, world=32) == k3
    assert 65 * GIB < k3 < 66 * GIB
    assert _reserve(MachineFamily.H200, is_k3=False) == 40 * GIB


def test_h20_tp8_sizes_each_family_with_its_own_row():
    assert _reserve(MachineFamily.H20, is_k3=False) == 24 * GIB
    assert _reserve(MachineFamily.H20, is_k3=True, world=32) == 24 * GIB


def test_k3_h20_reserve_is_sized_and_below_the_h200_reserve():
    # The H20 token cap shrinks K3's peak terms, so its reserve must be a
    # positive value strictly below the H200 one.
    h20 = k3_engine_profile(MachineFamily.H20, 32).prefix_pool_workspace_bytes
    h200 = k3_engine_profile(MachineFamily.H200, 16).prefix_pool_workspace_bytes
    assert 0 < h20 < h200


def test_every_family_has_a_48b_row():
    assert set(KL48_PREFIX_POOL_WORKSPACE_BYTES) == set(MachineFamily)


def test_non_tp8_layouts_are_pool_off_without_a_machine_family():
    for group in (1, 2, 4, 16):
        for is_k3 in (True, False):
            assert _reserve(None, is_k3=is_k3, group=group) == 0


def test_tp8_reserve_needs_the_machine_family():
    with pytest.raises(ValueError, match="machine family"):
        _reserve(None, is_k3=False)


def test_k3_tp8_reserve_needs_a_profile_row():
    with pytest.raises(UnsupportedK3TopologyError):
        _reserve(MachineFamily.H20, is_k3=True, world=16)


def test_nonpositive_group_size_is_refused():
    with pytest.raises(ValueError):
        _reserve(MachineFamily.H200, is_k3=False, group=0)
