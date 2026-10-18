"""Sizing table for the in-wave prefill prefix pool reserve.

The function is a pure table over (GPU memory class, attention group size,
model family); these tests pin every row so a resize is a deliberate edit
here, never a silent behavior change at serving time.
"""

import pytest

from batchgen.models.moonshotai.kimi_linear.planner import (
    _GIB,
    _K3_H200_MIN_MEMORY_BYTES,
    _K3_H200_PREFIX_POOL_WORKSPACE_BYTES,
    _KL48_H20_MIN_MEMORY_BYTES,
    _KL48_H20_PREFIX_POOL_WORKSPACE_BYTES,
    k3_prefill_prefix_pool_workspace_bytes,
)

H200 = _K3_H200_MIN_MEMORY_BYTES
H20 = 97 * _GIB


def test_h200_class_tp8_keeps_the_k3_sized_reserve_for_both_families():
    for is_k3 in (True, False):
        assert k3_prefill_prefix_pool_workspace_bytes(
            gpu_total_memory_bytes=H200, attention_group_size=8, is_k3=is_k3
        ) == _K3_H200_PREFIX_POOL_WORKSPACE_BYTES


def test_h20_class_tp8_sizes_only_the_48b_family():
    assert k3_prefill_prefix_pool_workspace_bytes(
        gpu_total_memory_bytes=H20, attention_group_size=8, is_k3=False
    ) == _KL48_H20_PREFIX_POOL_WORKSPACE_BYTES
    # K3 is unsized on H20: the pool stays off and the admission-side named
    # refusal is the intended behavior.
    assert k3_prefill_prefix_pool_workspace_bytes(
        gpu_total_memory_bytes=H20, attention_group_size=8, is_k3=True
    ) == 0


def test_below_the_h20_threshold_everything_is_pool_off():
    assert _KL48_H20_MIN_MEMORY_BYTES < _K3_H200_MIN_MEMORY_BYTES
    for is_k3 in (True, False):
        assert k3_prefill_prefix_pool_workspace_bytes(
            gpu_total_memory_bytes=_KL48_H20_MIN_MEMORY_BYTES - 1,
            attention_group_size=8,
            is_k3=is_k3,
        ) == 0


def test_non_tp8_layouts_and_unknown_memory_are_pool_off():
    for group in (1, 2, 4, 16):
        assert k3_prefill_prefix_pool_workspace_bytes(
            gpu_total_memory_bytes=H200, attention_group_size=group, is_k3=False
        ) == 0
    assert k3_prefill_prefix_pool_workspace_bytes(
        gpu_total_memory_bytes=None, attention_group_size=8, is_k3=False
    ) == 0


def test_default_keeps_the_pre_change_k3_behavior():
    # Callers that do not pass is_k3 get the old table exactly.
    assert k3_prefill_prefix_pool_workspace_bytes(
        gpu_total_memory_bytes=H20, attention_group_size=8
    ) == 0
    assert k3_prefill_prefix_pool_workspace_bytes(
        gpu_total_memory_bytes=H200, attention_group_size=8
    ) == _K3_H200_PREFIX_POOL_WORKSPACE_BYTES


def test_nonpositive_group_size_is_refused():
    with pytest.raises(ValueError):
        k3_prefill_prefix_pool_workspace_bytes(
            gpu_total_memory_bytes=H200, attention_group_size=0
        )
