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

from batchgen.hardware.machine_family import (  # noqa: E402
    DeviceFacts,
    UnknownMachineFamilyError,
)
from batchgen.models.moonshotai.kimi_linear.machine_profile import (  # noqa: E402
    UnsupportedK3TopologyError,
)
from batchgen.models.moonshotai.kimi_linear.planner import (  # noqa: E402
    KimiLinearPlanner,
)


GIB = 1024 ** 3
MIB = 1024 ** 2

# CUDA-reported facts of each machine family (observed totals).
FACTS = {
    "h20": DeviceFacts((9, 0), 78, 97_871 * MIB, "x86_64"),
    "h200": DeviceFacts((9, 0), 132, int(139.81 * GIB), "x86_64"),
    "b200": DeviceFacts((10, 0), 148, 182_640 * MIB, "x86_64"),
    "b300": DeviceFacts((10, 3), 148, int(267.68 * GIB), "x86_64"),
    "gb300": DeviceFacts((10, 3), 152, 283_270 * MIB, "aarch64"),
}


def _plan(
    family,
    *,
    world_size,
    group_size=8,
    is_k3=True,
    capturable=True,
    facts=None,
):
    config = EngineConfig()
    config.Basic_Config.kv_dtype = "bfloat16"
    config.Basic_Config.set_max_prompt_length(8192)
    config.Basic_Config.max_decoding_length = 128
    config.Basic_Config.world_size = world_size
    planner = KimiLinearPlanner(
        is_k3=is_k3,
        stream_all_modules=False,
        attention_group_size=group_size,
        device_facts=FACTS[family] if facts is None and family else facts,
        decode_graph_capturable=capturable,
    )
    return planner.generate_config(config, "kimi_k3")


def test_h20_plan_funds_128_slots_latent_only_with_graphs():
    config = _plan("h20", world_size=32)

    # 128 user slots + 1 graph scratch + the prefix-pool segment reserve,
    # which prefill_sequence_limits holds back so admission sees exactly 128.
    assert config.GPU_Buffer_Config.kda_prefix_segment_slots == 16
    assert config.GPU_Buffer_Config.kda_state_slots == 129 + 16
    assert config.Module_Batching_Config.MoE_decoding_micro_batch_size == 128
    assert config.Basic_Config.decode_graph_buckets == [
        1, 2, 4, 8, 16, 24, 32, 64, 96, 128,
    ]
    # the 128-row plan takes the whole-model decode graph like H200 (eager
    # decode at this width is launch-latency-bound)
    assert config.Basic_Config.decode_graph_mode == "graph"
    # H20 keeps the release-and-stream admission cycle (resident handoff off)
    assert config.Module_Batching_Config.k3_resident_prefill_token_cap == 0
    assert config.Module_Batching_Config.k3_resident_prefill_chunk_rows == 0
    assert config.Module_Batching_Config.prefill_micro_batch_token_cap == 16_384
    assert (
        config.Module_Batching_Config.k3_prefill_collective_stripe_threshold_rows
        == 32_768
    )
    assert config.Module_Batching_Config.k3_prefill_grouped_chunk_rows == 2_048
    assert (
        config.Module_Batching_Config.prefill_prefix_pool_workspace_bytes
        == 24 * GIB
    )
    # latent shards; vocab-parallel stays off until
    # batchgen-k3-vocab-parallel-h20-corruption is fixed
    assert config.Module_Batching_Config.k3_shard_latent_projections is True
    assert config.Module_Batching_Config.k3_shard_vocab_parallel is False


def test_h200_plan_exposes_256_users_plus_separate_graph_scratch():
    config = _plan("h200", world_size=16)

    # TP8-sharded latent projections (8.3 GiB) + vocab-parallel embed/lm_head (3.84 GiB) -> 256 KDA slots/node
    assert config.Module_Batching_Config.k3_shard_latent_projections is True
    assert config.Module_Batching_Config.k3_shard_vocab_parallel is True
    # 256 user slots + 1 graph scratch + the prefix-pool segment reserve, which
    # prefill_sequence_limits holds back so admission still sees exactly 256.
    assert config.GPU_Buffer_Config.kda_prefix_segment_slots == 16
    assert config.GPU_Buffer_Config.kda_state_slots == 257 + 16
    assert config.Module_Batching_Config.MoE_decoding_micro_batch_size == 256
    # the admission cap itself must be a bucket: a full batch with a lower top
    # bucket has no graph and silently runs eager for the whole run
    assert config.Basic_Config.decode_graph_buckets == [
        1, 2, 4, 8, 16, 24, 32, 64, 96, 128, 160, 192, 256,
    ]
    # whole-model CUDA-graph decode on H200 (SGLang: cuda-graph-backend-decode full)
    assert config.Basic_Config.decode_graph_mode == "graph"
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
    assert (
        config.Module_Batching_Config.k3_resident_prefill_token_cap == 16_384
    )
    assert config.Module_Batching_Config.k3_resident_prefill_chunk_rows == 2048
    assert (
        config.Module_Batching_Config.prefill_prefix_pool_workspace_bytes
        == 40 * GIB
    )


def test_b300_single_node_plan_takes_its_own_kv_solved_slots():
    config = _plan("b300", world_size=8)

    # 360 user slots (kv-solve on the measured single-node decode memory) +
    # 1 graph scratch + 16 prefix segments.
    assert config.GPU_Buffer_Config.kda_state_slots == 360 + 1 + 16
    assert config.Module_Batching_Config.MoE_decoding_micro_batch_size == 360
    assert config.Basic_Config.decode_graph_buckets == [
        1, 2, 4, 8, 16, 24, 32, 64, 96, 128, 160, 192, 256, 320, 360,
    ]
    assert config.Module_Batching_Config.k3_shard_latent_projections is True
    assert config.Module_Batching_Config.k3_shard_vocab_parallel is True
    # prefill keeps the values the machine ran under the retired tiers
    assert config.Module_Batching_Config.prefill_micro_batch_token_cap == 524_288
    assert config.Module_Batching_Config.k3_resident_prefill_token_cap == 16_384
    assert config.Module_Batching_Config.k3_resident_prefill_chunk_rows == 2048


def test_blackwell_decode_runs_eager_when_kernel_not_capturable():
    config = _plan("b300", world_size=8, capturable=False)
    assert config.Basic_Config.decode_graph_mode == "eager"


def test_gb300_two_tray_plan_carries_the_inherited_values():
    config = _plan("gb300", world_size=8)
    assert config.Module_Batching_Config.MoE_decoding_micro_batch_size == 256
    assert config.GPU_Buffer_Config.kda_state_slots == 257 + 16


@pytest.mark.parametrize(
    ("family", "world_size"),
    [
        ("h20", 16),   # an EP16 shard does not fit an H20 GPU
        ("h200", 8),   # the EP8 shard does not fit an H200 GPU
        ("h200", 32),  # never measured
        ("b300", 16),  # never measured
        ("gb300", 4),  # one tray cannot hold the routed expert set
        ("b200", 8),   # no K3 row
    ],
)
def test_k3_tp8_without_a_profile_row_is_refused_by_name(family, world_size):
    with pytest.raises(UnsupportedK3TopologyError, match=family):
        _plan(family, world_size=world_size)


def test_k3_tp8_on_an_unknown_machine_is_refused_by_name():
    unknown = DeviceFacts((9, 0), 132, 80 * GIB, "x86_64")
    with pytest.raises(UnknownMachineFamilyError):
        _plan(None, world_size=16, facts=unknown)


def test_tp8_plan_without_device_facts_is_refused():
    for is_k3 in (True, False):
        with pytest.raises(ValueError, match="machine family"):
            _plan(None, world_size=16, is_k3=is_k3)


@pytest.mark.parametrize("family", [None, "h20", "h200", "b300"])
def test_non_tp8_k3_plan_is_machine_independent(family):
    config = _plan(family, world_size=8, group_size=1)
    # 4 user slots + 1 graph scratch, no prefix pool, unsharded, eager
    assert config.GPU_Buffer_Config.kda_state_slots == 5
    assert config.GPU_Buffer_Config.kda_prefix_segment_slots == 0
    assert config.Module_Batching_Config.MoE_decoding_micro_batch_size == 4
    assert config.Module_Batching_Config.k3_shard_latent_projections is False
    assert config.Module_Batching_Config.k3_shard_vocab_parallel is False
    assert config.Basic_Config.decode_graph_mode == "eager"
    assert config.Module_Batching_Config.prefill_micro_batch_token_cap == 262_144
    assert config.Module_Batching_Config.k3_resident_prefill_token_cap == 0


def test_head_parallel_non_tp8_k3_keeps_the_single_prompt_cap():
    config = _plan(None, world_size=8, group_size=4)
    assert config.Module_Batching_Config.prefill_micro_batch_token_cap == 16_384
    assert config.GPU_Buffer_Config.kda_state_slots == 5


def test_nonpositive_group_size_is_refused():
    with pytest.raises(ValueError):
        _plan("h200", world_size=16, group_size=0)
