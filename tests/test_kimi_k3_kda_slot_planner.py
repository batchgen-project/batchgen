import importlib.util
import sys
import types
from pathlib import Path

import pytest

_config_path = Path(__file__).resolve().parents[1] / "batchgen" / "config" / "config.py"
_config_pkg = types.ModuleType("batchgen.config")
_config_pkg.__path__ = [str(_config_path.parent)]
sys.modules.setdefault("batchgen.config", _config_pkg)
_config_spec = importlib.util.spec_from_file_location("batchgen.config.config", _config_path)
_config_module = importlib.util.module_from_spec(_config_spec)
sys.modules.setdefault("batchgen.config.config", _config_module)
assert _config_spec.loader is not None
_config_spec.loader.exec_module(_config_module)
EngineConfig = _config_module.EngineConfig

from batchgen.models.moonshotai.kimi_linear.planner import (
    KimiLinearPlanner,
    k3_kda_state_slots,
    k3_prefill_collective_stripe_threshold_rows,
    k3_prefill_grouped_chunk_rows,
    k3_prefill_micro_batch_token_cap,
)


GIB = 1024 ** 3


def test_h20_tp8_keeps_four_user_slots():
    assert k3_kda_state_slots(
        gpu_total_memory_bytes=96 * GIB,
        attention_group_size=8,
    ) == 4


def test_h200_tp8_uses_256_user_slots():
    """H200's slot count IS the per-node decode admission cap, so it is sized
    for decode concurrency (256 x 53.57 MiB = 13.4 GiB/rank beside the 84 GiB
    resident expert shard), not for the old 32-slot KV-headroom guard."""
    assert k3_kda_state_slots(
        gpu_total_memory_bytes=140 * GIB,
        attention_group_size=8,
    ) == 256


def test_h20_tp8_keeps_single_long_prompt_prefill_cap():
    assert k3_prefill_micro_batch_token_cap(
        gpu_total_memory_bytes=96 * GIB,
        attention_group_size=8,
    ) == 16_384


def test_h200_tp8_batches_eight_exact_64k_prompts_per_model_pass():
    assert k3_prefill_micro_batch_token_cap(
        gpu_total_memory_bytes=140 * GIB,
        attention_group_size=8,
    ) == 8 * 65_536


def test_h20_tp8_keeps_striped_streamed_prefill_collectives():
    assert k3_prefill_collective_stripe_threshold_rows(
        gpu_total_memory_bytes=96 * GIB,
        attention_group_size=8,
    ) == 32_768


def test_h200_tp8_uses_one_wide_exact64_node_collective():
    assert k3_prefill_collective_stripe_threshold_rows(
        gpu_total_memory_bytes=140 * GIB,
        attention_group_size=8,
    ) == 8 * 65_536


def test_h20_tp8_keeps_2k_grouped_expert_chunks():
    assert k3_prefill_grouped_chunk_rows(
        gpu_total_memory_bytes=96 * GIB,
        attention_group_size=8,
    ) == 2_048


def test_h200_tp8_uses_32k_grouped_expert_chunks():
    assert k3_prefill_grouped_chunk_rows(
        gpu_total_memory_bytes=140 * GIB,
        attention_group_size=8,
    ) == 32_768


def test_h200_non_tp8_fails_safe_to_four_slots():
    assert k3_kda_state_slots(
        gpu_total_memory_bytes=140 * GIB,
        attention_group_size=1,
    ) == 4

    assert k3_prefill_micro_batch_token_cap(
        gpu_total_memory_bytes=140 * GIB,
        attention_group_size=1,
    ) == 16_384
    assert k3_prefill_collective_stripe_threshold_rows(
        gpu_total_memory_bytes=140 * GIB,
        attention_group_size=1,
    ) == 32_768
    assert k3_prefill_grouped_chunk_rows(
        gpu_total_memory_bytes=140 * GIB,
        attention_group_size=1,
    ) == 2_048


def test_unknown_memory_fails_safe_to_four_slots():
    assert k3_kda_state_slots(
        gpu_total_memory_bytes=None,
        attention_group_size=8,
    ) == 4
    assert k3_prefill_micro_batch_token_cap(
        gpu_total_memory_bytes=None,
        attention_group_size=8,
    ) == 16_384
    assert k3_prefill_collective_stripe_threshold_rows(
        gpu_total_memory_bytes=None,
        attention_group_size=8,
    ) == 32_768
    assert k3_prefill_grouped_chunk_rows(
        gpu_total_memory_bytes=None,
        attention_group_size=8,
    ) == 2_048


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
    with pytest.raises(ValueError):
        k3_prefill_grouped_chunk_rows(
            gpu_total_memory_bytes=memory_bytes,
            attention_group_size=group_size,
        )


def _plan_k3(gpu_memory_bytes):
    config = EngineConfig()
    config.Basic_Config.kv_dtype = "bfloat16"
    config.Basic_Config.set_max_prompt_length(8192)
    config.Basic_Config.max_decoding_length = 128
    config.Basic_Config.world_size = 16
    planner = KimiLinearPlanner(
        is_k3=True,
        stream_all_modules=False,
        attention_group_size=8,
        gpu_total_memory_bytes=gpu_memory_bytes,
    )
    return planner.generate_config(config)


def test_h20_plan_adds_graph_scratch_without_reducing_user_capacity():
    config = _plan_k3(96 * GIB)

    assert config.GPU_Buffer_Config.kda_state_slots == 5
    assert config.Module_Batching_Config.MoE_decoding_micro_batch_size == 4
    assert config.Basic_Config.decode_graph_buckets == [1, 2, 4]


def test_h200_plan_exposes_32_users_plus_separate_graph_scratch():
    config = _plan_k3(140 * GIB)

    assert config.GPU_Buffer_Config.kda_state_slots == 33
    assert config.Module_Batching_Config.MoE_decoding_micro_batch_size == 32
    assert config.Basic_Config.decode_graph_buckets == [
        1, 2, 4, 8, 16, 24, 32,
    ]
    assert (
        config.Module_Batching_Config.prefill_micro_batch_token_cap
        == 8 * 65_536
    )
    assert (
        config.Module_Batching_Config
        .k3_prefill_collective_stripe_threshold_rows
        == 8 * 65_536
    )
    assert (
        config.Module_Batching_Config.k3_prefill_grouped_chunk_rows
        == 32_768
    )
