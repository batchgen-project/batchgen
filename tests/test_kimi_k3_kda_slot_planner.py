import pytest

from batchgen.models.moonshotai.kimi_linear.planner import (
    k3_kda_state_slots,
)


GIB = 1024 ** 3


def test_h20_tp8_keeps_validated_four_slots():
    assert k3_kda_state_slots(
        gpu_total_memory_bytes=96 * GIB,
        attention_group_size=8,
    ) == 4


def test_h200_tp8_uses_32_slots():
    assert k3_kda_state_slots(
        gpu_total_memory_bytes=140 * GIB,
        attention_group_size=8,
    ) == 32


def test_h200_non_tp8_fails_safe_to_four_slots():
    assert k3_kda_state_slots(
        gpu_total_memory_bytes=140 * GIB,
        attention_group_size=1,
    ) == 4


def test_unknown_memory_fails_safe_to_four_slots():
    assert k3_kda_state_slots(
        gpu_total_memory_bytes=None,
        attention_group_size=8,
    ) == 4


@pytest.mark.parametrize(
    ("memory_bytes", "group_size"),
    [(0, 8), (-1, 8), (96 * GIB, 0), (96 * GIB, -1)],
)
def test_invalid_capacity_inputs_fail_closed(memory_bytes, group_size):
    with pytest.raises(ValueError):
        k3_kda_state_slots(
            gpu_total_memory_bytes=memory_bytes,
            attention_group_size=group_size,
        )
