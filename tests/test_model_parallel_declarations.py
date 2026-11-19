"""Every registered model's parallel declaration, and the undeclared ratchet.

The declarations are computed from launch configuration at start-up, so each
test drives the real method on a bare instance (``__new__`` plus exactly the
attributes it reads). MiniMax-M2.5 and DeepSeek-V3's manager modules cannot
be imported on a CPU-only host (their import chain needs a real triton), so
their methods are lifted from the source files instead, the
``tests/worker_method_source.py`` way; the ratchet below checks every
registered manager's source either way.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.triton_cpu_stub import install_triton_cpu_stub

install_triton_cpu_stub()

ROOT = Path(__file__).resolve().parents[1]

# Every (source file, manager class) the registration seam
# (batchgen/get_parallel_strategy_manager.py) can return.
REGISTERED_MANAGERS = {
    "minimax-m2.5": (
        "batchgen/models/minimax/minimax_m25/Parallel_Strategy_Manager.py",
        "MiniMaxM25ParallelStrategyManager",
    ),
    "kimi-linear": (
        "batchgen/models/moonshotai/kimi_linear/Parallel_Strategy_Manager.py",
        "KimiLinearParallelStrategyManager",
    ),
    "kimi-k2.5": (
        "batchgen/models/moonshotai/kimi_k25/Parallel_Strategy_Manager.py",
        "KimiK25ParallelStrategyManager",
    ),
    "deepseek-v4-flash": (
        "batchgen/models/deepseek/deepseekv4_flash/Parallel_Strategy_Manager.py",
        "DeepSeekV4FlashParallelStrategyManager",
    ),
    "deepseek-v3": (
        "batchgen/models/deepseek/deepseekv3/Parallel_Strategy_Manager.py",
        "DeepseekV3ParallelStrategyManager",
    ),
    "gpt-oss-120b": (
        "batchgen/models/openai/gpt_oss_120b/Parallel_Strategy_Manager.py",
        "GptOssParallelStrategyManager",
    ),
    "glm-5": (
        "batchgen/models/glm/glm5/Parallel_Strategy_Manager.py",
        "GLM5ParallelStrategyManager",
    ),
}

# The ratchet: models whose manager has not declared yet, with the reason.
# Shrink-only — a model may move OUT of this set, never back in.
UNDECLARED_MODELS: dict = {}


def test_registration_seam_and_this_list_agree():
    source = (ROOT / "batchgen/get_parallel_strategy_manager.py").read_text()
    for name, (_, cls) in REGISTERED_MANAGERS.items():
        assert cls in source, f"{name}: {cls} not in the registration seam"
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert imported == {cls for _, cls in REGISTERED_MANAGERS.values()}, (
        "a manager was registered or removed without updating this ratchet"
    )


def test_every_registered_manager_is_declared():
    undeclared = {}
    for name, (path, cls_name) in REGISTERED_MANAGERS.items():
        tree = ast.parse((ROOT / path).read_text())
        cls = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == cls_name
        )
        if not any(
            isinstance(node, ast.FunctionDef) and node.name == "parallel_declaration"
            for node in cls.body
        ):
            undeclared[name] = "no parallel_declaration on the manager"
    assert undeclared == UNDECLARED_MODELS
    # Acceptance ratchet: the GLM-5 family is declared.
    assert "glm-5" not in undeclared


def _lift_methods(model: str, names, namespace):
    """Compile manager methods straight from source (CPU-unimportable modules)."""
    path, cls_name = REGISTERED_MANAGERS[model]
    tree = ast.parse((ROOT / path).read_text())
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == cls_name
    )
    out = {}
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            scope = dict(namespace)
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), path, "exec"),
                scope,
            )
            out[node.name] = scope[node.name]
    assert set(out) == set(names), f"{model}: methods missing from {cls_name}"
    return out


def _ep_config(offloading=False, ratio=0.0, num_local=None, **extra):
    return SimpleNamespace(
        EP_Config=SimpleNamespace(
            enable_offloading=offloading,
            offloading_ratio=ratio,
            num_local_expert_per_layer=num_local,
            **extra,
        ),
        Basic_Config=SimpleNamespace(),
    )


def rendered(declaration):
    return declaration.rendered_slots()


# -- GLM-5 (required by the acceptance ratchet) ------------------------------

def _glm5(world_size, engine_config):
    from batchgen.models.glm.glm5.Parallel_Strategy_Manager import (
        GLM5ParallelStrategyManager,
    )

    pm = GLM5ParallelStrategyManager.__new__(GLM5ParallelStrategyManager)
    pm.world_size = world_size
    pm.engine_config = engine_config
    return pm


def test_glm5_declares_resident_ep_decode():
    pm = _glm5(8, _ep_config())
    assert rendered(pm.parallel_declaration()) == {
        "prefill_attention": "dp:streamed",
        "prefill_moe": "dp:streamed",
        "decode_attention": "dp:resident",
        "decode_moe": "ep8:resident",
    }


def test_glm5_declares_partial_residency_under_offloading():
    pm = _glm5(8, _ep_config(offloading=True, ratio=0.5))
    assert rendered(pm.parallel_declaration())["decode_moe"] == "ep8:partial"
    # The declaration follows the same planner the configure assigns from.
    assert pm._plan_local_expert_split() == (True, 16)


# -- GPT-OSS-120B ------------------------------------------------------------

def _gptoss(world_size, engine_config, num_local_experts=128):
    from batchgen.models.openai.gpt_oss_120b.Parallel_Strategy_Manager import (
        GptOssParallelStrategyManager,
    )

    pm = GptOssParallelStrategyManager.__new__(GptOssParallelStrategyManager)
    pm.world_size = world_size
    pm.engine_config = engine_config
    pm.model_config = SimpleNamespace(num_local_experts=num_local_experts)
    return pm


def test_gptoss_single_gpu_keeps_everything_resident_but_streams_prefill_attn():
    assert rendered(_gptoss(1, _ep_config()).parallel_declaration()) == {
        "prefill_attention": "dp:streamed",
        "prefill_moe": "dp:resident",
        "decode_attention": "dp:resident",
        "decode_moe": "ep1:resident",
    }


def test_gptoss_offloading_streams_prefill_and_splits_decode():
    declared = rendered(
        _gptoss(4, _ep_config(offloading=True, ratio=0.5)).parallel_declaration()
    )
    assert declared["prefill_moe"] == "dp:streamed"
    assert declared["decode_moe"] == "ep4:partial"


# -- Kimi-Linear / K3 --------------------------------------------------------

def _kimi_linear(world_size, G, stream_all, prefill_mode, decode_mode):
    from batchgen.models.moonshotai.kimi_linear.Parallel_Strategy_Manager import (
        KimiLinearParallelStrategyManager,
    )

    pm = KimiLinearParallelStrategyManager.__new__(KimiLinearParallelStrategyManager)
    pm.world_size = world_size
    pm._attn_tp_size = G
    pm._stream_all_modules = stream_all
    pm._prefill_moe_mode = prefill_mode
    pm.engine_config = _ep_config(decode_moe_mode=decode_mode)
    return pm


def test_kimi_k3_declares_head_parallel_attention_and_resident_ep():
    pm = _kimi_linear(16, 8, False, "streamed_sp8", "resident_ep")
    assert rendered(pm.parallel_declaration()) == {
        "prefill_attention": "tp8:resident",
        "prefill_moe": "dp:streamed",
        "decode_attention": "tp8:resident",
        "decode_moe": "ep16:resident",
    }


def test_kimi_48b_default_is_resident_dp_attention():
    pm = _kimi_linear(8, 1, False, "streamed", "resident_ep")
    assert rendered(pm.parallel_declaration()) == {
        "prefill_attention": "dp:resident",
        "prefill_moe": "dp:streamed",
        "decode_attention": "dp:resident",
        "decode_moe": "ep8:resident",
    }


def test_kimi_streamed_prefill_and_unknown_modes():
    from batchgen.parallel_declaration import ParallelDeclarationError

    pm = _kimi_linear(8, 1, True, "streamed", "streamed")
    declared = rendered(pm.parallel_declaration())
    assert declared["prefill_attention"] == "dp:streamed"
    assert declared["decode_moe"] == "dp:streamed"
    pm = _kimi_linear(8, 1, False, "quantum", "resident_ep")
    with pytest.raises(ParallelDeclarationError, match="prefill MoE mode"):
        pm.parallel_declaration()


# -- Kimi-K2.5 ---------------------------------------------------------------

def _k25(world_size, engine_config):
    from batchgen.models.moonshotai.kimi_k25.Parallel_Strategy_Manager import (
        KimiK25ParallelStrategyManager,
    )

    pm = KimiK25ParallelStrategyManager.__new__(KimiK25ParallelStrategyManager)
    pm.world_size = world_size
    pm.engine_config = engine_config
    return pm


def test_k25_declares_resident_ep_decode_and_partial_under_offloading():
    assert rendered(_k25(8, _ep_config()).parallel_declaration()) == {
        "prefill_attention": "dp:streamed",
        "prefill_moe": "dp:streamed",
        "decode_attention": "dp:resident",
        "decode_moe": "ep8:resident",
    }
    pm = _k25(8, _ep_config(offloading=True, ratio=0.25))
    assert rendered(pm.parallel_declaration())["decode_moe"] == "ep8:partial"
    assert pm._plan_local_expert_split() == (True, 36)


# -- DeepSeek-V4-Flash -------------------------------------------------------

def test_deepseek_v4_flash_streams_every_module_in_both_phases():
    from batchgen.models.deepseek.deepseekv4_flash.Parallel_Strategy_Manager import (
        DeepSeekV4FlashParallelStrategyManager,
    )

    pm = DeepSeekV4FlashParallelStrategyManager.__new__(
        DeepSeekV4FlashParallelStrategyManager
    )
    pm.world_size = 4
    assert rendered(pm.parallel_declaration()) == {
        "prefill_attention": "dp:streamed",
        "prefill_moe": "dp:streamed",
        "decode_attention": "dp:streamed",
        "decode_moe": "ep4:streamed",
    }


# -- DeepSeek-V3 and MiniMax-M2.5 (source-lifted; modules need real triton) --

def _lifted_manager(model, namespace, world_size, engine_config, **attrs):
    methods = _lift_methods(
        model, ["parallel_declaration", "_plan_local_expert_split"], namespace
    )
    pm = SimpleNamespace(world_size=world_size, engine_config=engine_config, **attrs)
    pm._plan_local_expert_split = lambda: methods["_plan_local_expert_split"](pm)
    pm.parallel_declaration = lambda: methods["parallel_declaration"](pm)
    return pm


def test_deepseek_v3_declares_resident_ep_decode():
    pm = _lifted_manager(
        "deepseek-v3", {}, 8, _ep_config(), NUM_TOTAL_EXPERTS=256
    )
    assert rendered(pm.parallel_declaration()) == {
        "prefill_attention": "dp:streamed",
        "prefill_moe": "dp:streamed",
        "decode_attention": "dp:resident",
        "decode_moe": "ep8:resident",
    }
    pm = _lifted_manager(
        "deepseek-v3", {}, 8, _ep_config(offloading=True, ratio=0.5),
        NUM_TOTAL_EXPERTS=256,
    )
    assert rendered(pm.parallel_declaration())["decode_moe"] == "ep8:partial"


def test_minimax_declares_resident_ep_decode():
    namespace = {"NUM_TOTAL_EXPERTS": 256}
    pm = _lifted_manager("minimax-m2.5", namespace, 8, _ep_config())
    assert rendered(pm.parallel_declaration()) == {
        "prefill_attention": "dp:streamed",
        "prefill_moe": "dp:streamed",
        "decode_attention": "dp:resident",
        "decode_moe": "ep8:resident",
    }
    pm = _lifted_manager(
        "minimax-m2.5", namespace, 8,
        _ep_config(num_local=16),
    )
    assert rendered(pm.parallel_declaration())["decode_moe"] == "ep8:partial"
