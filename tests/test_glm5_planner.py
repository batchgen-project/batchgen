import sys
import types
import importlib.util
from pathlib import Path

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

from batchgen.models.glm.glm5.planner import GLM5Planner


def _plan(world_size: int) -> EngineConfig:
    config = EngineConfig()
    config.Basic_Config.kv_dtype = "bfloat16"
    config.Basic_Config.set_max_prompt_length(8192)
    config.Basic_Config.max_decoding_length = 16
    config.Basic_Config.world_size = world_size
    return GLM5Planner("zai-org/GLM-5-FP8").generate_config(config)


def test_glm5_planner_keeps_full_persistent_single_node_h200_decode():
    config = _plan(world_size=8)

    assert config.Basic_Config.attn_mode == 3
    assert config.EP_Config.num_local_expert_per_layer == 32
    assert config.GPU_Buffer_Config.num_decoding_module_buffer["routed_expert"] == 0
    assert config.GPU_Buffer_Config.num_prefill_module_buffer["routed_expert"] == 512
    assert config.GPU_Buffer_Config.num_prefill_module_buffer["shared_expert"] == 2



def test_glm5_planner_keeps_full_persistent_two_node_decode():
    config = _plan(world_size=16)

    assert config.Basic_Config.attn_mode == 3
    assert config.EP_Config.num_local_expert_per_layer == 16
    assert config.GPU_Buffer_Config.num_decoding_module_buffer["routed_expert"] == 0
    assert config.GPU_Buffer_Config.num_prefill_module_buffer["routed_expert"] == 512
    assert config.GPU_Buffer_Config.num_prefill_module_buffer["shared_expert"] == 2
