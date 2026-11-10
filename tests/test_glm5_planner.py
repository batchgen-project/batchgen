import ast
import sys
import types
import importlib.util
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

from batchgen.models.glm.glm5.planner import GLM5Planner


def test_glm5_planner_consumes_mode_without_writing_it():
    source_path = _config_path.parents[1] / "models" / "glm" / "glm5" / "planner.py"
    tree = ast.parse(source_path.read_text())
    assert not [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.Attribute, ast.Name))
        and isinstance(node.ctx, ast.Store)
        and (node.attr if isinstance(node, ast.Attribute) else node.id) == "attn_mode"
    ]


def _plan(world_size: int, available_gpu_gib: float = 96 * 0.85) -> EngineConfig:
    config = EngineConfig()
    config.Basic_Config.attn_mode = 1
    config.Basic_Config.kv_dtype = "bfloat16"
    config.Basic_Config.set_max_prompt_length(8192)
    config.Basic_Config.max_decoding_length = 16
    config.Basic_Config.world_size = world_size
    planner = GLM5Planner("zai-org/GLM-5-FP8")
    planner._available_gpu_memory_gb = lambda: available_gpu_gib
    return planner.generate_config(config, "glm_moe_dsa")


def test_glm5_planner_rejects_single_node_with_h20_memory_budget():
    with pytest.raises(RuntimeError, match="requires all local MoE experts"):
        _plan(world_size=8)


def test_glm5_planner_keeps_full_persistent_single_node_h200_decode():
    config = _plan(world_size=8, available_gpu_gib=144 * 0.85)

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
