"""CPU-only tests of the decode strategy owner and its consumer boundaries."""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from batchgen.config.decode_strategy import (
    DecodeStrategy,
    apply_normalized_attn_mode,
    normalize_attn_mode,
    strategy_for,
)


ROOT = Path(__file__).resolve().parents[2]
LOGGER = "batchgen.config.decode_strategy"


@pytest.mark.parametrize("raw_mode", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("forced_mode", [None, 0, 3])
def test_normalize_attn_mode(raw_mode, forced_mode):
    if raw_mode == 4:
        # Validation of the configured code runs BEFORE any forced mode applies.
        with pytest.raises(ValueError, match=r"4.*\{0, 1, 2, 3\}"):
            normalize_attn_mode(raw_mode, forced_mode)
    else:
        expected = forced_mode if forced_mode is not None else raw_mode
        result = normalize_attn_mode(raw_mode, forced_mode)
        assert type(result) is int
        assert result == expected


def test_normalize_rejects_invalid_forced_mode():
    with pytest.raises(ValueError, match=r"forced decode mode 4"):
        normalize_attn_mode(1, 4)


@pytest.mark.parametrize("code, member", [
    (0, DecodeStrategy.CPU_ATTN),
    (1, DecodeStrategy.LEGACY_MICROBATCH_GPU),
    (2, DecodeStrategy.HYBRID_CPU_GPU),
    (3, DecodeStrategy.DECOUPLED_GPU),
])
def test_strategy_for_round_trip(code, member):
    assert strategy_for(code) is member
    assert DecodeStrategy(member.value) is member


def test_strategy_for_rejects_unknown_code():
    with pytest.raises(ValueError):
        strategy_for(4)


class RecordingConfig:
    def __init__(self, raw_mode):
        self._mode = raw_mode
        self.writes = []

    @property
    def attn_mode(self):
        return self._mode

    @attn_mode.setter
    def attn_mode(self, value):
        self.writes.append((self._mode, value))
        self._mode = value


@pytest.mark.parametrize("raw_mode, model_type, forced_mode, expected", [
    (1, "glm_moe_dsa", 3, 3), (0, "GLM-5.2", 3, 3), (1, "deepseek_v3", None, 1), (3, None, None, 3),
])
def test_apply_writes_returns_and_logs_once(raw_mode, model_type, forced_mode, expected, caplog):
    config = RecordingConfig(raw_mode)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        result = apply_normalized_attn_mode(config, model_type, forced_mode)
    assert type(result) is int
    assert result == config.attn_mode == expected
    assert config.writes == [(raw_mode, expected)]
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.name == LOGGER
    assert record.levelno == logging.INFO
    assert record.msg == "decode strategy: raw=%d effective=%d model_type=%s"
    assert record.args == (raw_mode, expected, model_type)
    assert record.getMessage() == (
        f"decode strategy: raw={raw_mode} effective={expected} model_type={model_type}"
    )


def test_apply_is_idempotent_with_a_no_op_second_write(caplog):
    config = RecordingConfig(1)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        first = apply_normalized_attn_mode(config, "glm_moe_dsa", 3)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        second = apply_normalized_attn_mode(config, "glm_moe_dsa", 3)
    assert first == second == config.attn_mode == 3
    # The second assignment writes the existing value, with no state change.
    assert config.writes == [(1, 3), (3, 3)]
    assert len(caplog.records) == 1
    assert caplog.records[0].args == (3, 3, "glm_moe_dsa")


@pytest.mark.parametrize("model_type, forced_mode", [("glm_moe_dsa", 3), ("gpt_oss", None), (None, None)])
def test_invalid_apply_neither_writes_nor_logs(model_type, forced_mode, caplog):
    config = RecordingConfig(4)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        with pytest.raises(ValueError):
            apply_normalized_attn_mode(config, model_type, forced_mode)
    assert config.attn_mode == 4
    assert config.writes == []
    assert caplog.records == []


def test_planners_schedulers_and_initializers_never_write_attn_mode():
    violations = []
    for path in (ROOT / "batchgen").rglob("*.py"):
        if not any(term in path.as_posix().lower() for term in (
            "planner", "scheduler", "initializer", "set_basic_config",
        )):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.Attribute, ast.Name)):
                name = node.attr if isinstance(node, ast.Attribute) else node.id
                if name == "attn_mode" and isinstance(node.ctx, ast.Store):
                    violations.append((str(path.relative_to(ROOT)), node.lineno))
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "setattr" and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value == "attn_mode"):
                violations.append((str(path.relative_to(ROOT)), node.lineno))
    assert violations == []


def test_worker_reads_mode_only_once_at_decode_entry():
    tree = ast.parse((ROOT / "batchgen/batchgen_worker.py").read_text())
    methods = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }

    def reads(node):
        return [n for n in ast.walk(node) if isinstance(n, ast.Attribute)
                and n.attr == "attn_mode" and isinstance(n.ctx, ast.Load)]

    assert len(reads(tree)) == 1
    assert reads(tree) == reads(methods["decoding_continuous"])
    assert reads(methods["_decoding_legacy_modes"]) == []
    legacy = methods["_decoding_legacy_modes"]
    assert "strategy" in [arg.arg for arg in legacy.args.args]
    heads = [ast.unparse(n.test) for n in ast.walk(legacy) if isinstance(n, ast.If)]
    for name in ("CPU_ATTN", "LEGACY_MICROBATCH_GPU", "HYBRID_CPU_GPU"):
        assert f"strategy is DecodeStrategy.{name}" in heads


@pytest.mark.parametrize("raw_mode, model_type, forced_mode, expected", [
    (1, "glm_moe_dsa", 3, 3), (1, "deepseek_v3", None, 1), (3, "gpt_oss", None, 3),
])
def test_base_planning_consumers_see_normalized_mode(raw_mode, model_type, forced_mode, expected):
    # Execute the real entry with fake planning hooks, without importing models.
    path = ROOT / "batchgen/planner/base_planner.py"
    tree = ast.parse(path.read_text())
    entry = next(n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == "generate_config")
    namespace = {
        "EngineConfig": SimpleNamespace,
        "apply_normalized_attn_mode": apply_normalized_attn_mode,
    }
    exec(compile(ast.Module(body=[entry], type_ignores=[]), str(path), "exec"), namespace)
    config = SimpleNamespace(Basic_Config=RecordingConfig(raw_mode))
    config.Basic_Config.padding_length = 8
    config.Basic_Config.max_decoding_length = 4
    config.Basic_Config.world_size = 1
    observed = []
    planner = SimpleNamespace(**{
        name: lambda: observed.append(config.Basic_Config.attn_mode)
        for name in ("_set_default_configs", "_compute_batch_configs", "_adjust_config_for_model")
    })
    planner.FORCED_DECODE_MODE = forced_mode
    assert namespace["generate_config"](planner, config, model_type) is config
    assert observed == [expected, expected, expected]
    assert config.Basic_Config.writes == [(raw_mode, expected)]
