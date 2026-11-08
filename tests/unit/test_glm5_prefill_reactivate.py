"""CPU coverage for the GLM-5 prefill in-place reactivation seam."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
PSM = ROOT / "batchgen" / "models" / "glm" / "glm5" / "Parallel_Strategy_Manager.py"


def _isolated_psm():
    tree = ast.parse(PSM.read_text())
    source_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "GLM5ParallelStrategyManager"
    )
    names = {
        "set_prefill_moe_mode",
        "_prepare_prefill_activation",
        "reactivate_prefill_in_place_applicable",
        "reactivate_prefill_in_place",
        "activate_prefill",
    }
    methods = [
        copy.deepcopy(node)
        for node in source_class.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(
        body=[
            ast.ClassDef(
                name="Psm",
                bases=[],
                keywords=[],
                body=methods,
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )

    calls = []

    class _Grouped:
        @classmethod
        def reset_prefill_grouped_state(cls):
            calls.append("reset")

        @classmethod
        def init_prefill_grouped_buffers(cls, config, device):
            calls.append(("init", config, device))

    namespace = {"Glm5MoE": _Grouped}
    exec(compile(ast.fix_missing_locations(module), str(PSM), "exec"), namespace)
    cls = namespace["Psm"]
    cls._calls = calls
    cls._sync = lambda: calls.append("sync")
    for name in names:
        member = getattr(cls, name)
        member.__globals__["_synchronize_prefill_preloads"] = cls._sync
    return cls, calls


def test_glm5_reactivate_prefill_keeps_live_model_and_rebuilds_grouped_state_once():
    cls, calls = _isolated_psm()
    psm = cls()
    live_model = object()
    weight_copy_task = {"attn": ["attn_0"]}
    config = SimpleNamespace(phase="prefill")
    psm.loaded_model_config = config
    psm.engine_config = SimpleNamespace(
        Basic_Config=SimpleNamespace(device_torch="cpu")
    )
    psm.model = live_model
    psm.weight_copy_task = weight_copy_task
    psm._prefill_moe_mode = "streamed"
    psm._check_late_deepgemm_compiles = lambda where: None

    result = psm.reactivate_prefill_in_place()

    assert result == (live_model, weight_copy_task)
    assert psm.model is live_model
    assert calls == ["reset", ("init", config, "cpu"), "sync"]

    psm.set_prefill_moe_mode("resident_ep")
    assert (
        psm.reactivate_prefill_in_place_applicable(prefill_moe_mode="resident_ep")
        is False
    )


def test_glm5_activate_prefill_uses_the_same_grouped_state_helper():
    cls, calls = _isolated_psm()
    psm = cls()
    config = SimpleNamespace(phase="decode")
    psm.loaded_model_config = config
    psm.engine_config = SimpleNamespace(
        Basic_Config=SimpleNamespace(device_torch="cpu")
    )
    psm.prefill_instance = (object(), {"shared_expert": ["shared_0"]})
    psm._check_late_deepgemm_compiles = lambda where: None

    result = psm.activate_prefill()

    assert result == psm.prefill_instance
    assert config.phase == "prefill"
    assert calls == ["reset", ("init", config, "cpu"), "sync"]
