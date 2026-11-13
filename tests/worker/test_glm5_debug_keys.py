"""GLM-5 debug behaviors are reachable only through ``batchgen_debug`` keys.

The nine ``BATCHGEN_GLM5_*`` env-var entry points (dispatch trace, MoE graph
compare, graph-path log, whole-model graph compare / timing / compare-fail /
probe layers / compare atol / compare rtol) are retired: the server-level graph
mode is decided by the CLI at startup, and a batch-level key only selects
compare / trace behavior within it. Each behavior is pinned here through its
key, and each retired env var is pinned as not honored.

The worker methods are compiled from source (``load_worker_methods``) and the
models helper is lifted from its module source, so these run without the GPU
kernel stack.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

from tests.worker_method_source import load_worker_methods

ROOT = Path(__file__).resolve().parents[2]

RETIRED_ENV_VARS = (
    "BATCHGEN_GLM5_DISPATCH_TRACE",
    "BATCHGEN_GLM5_MOE_GRAPH_COMPARE",
    "BATCHGEN_GLM5_GRAPH_PATH_LOG",
    "BATCHGEN_GLM5_WHOLE_MODEL_GRAPH_COMPARE",
    "BATCHGEN_GLM5_WHOLE_MODEL_GRAPH_TIMING",
    "BATCHGEN_GLM5_WHOLE_MODEL_GRAPH_COMPARE_FAIL",
    "BATCHGEN_GLM5_WHOLE_MODEL_GRAPH_PROBE_LAYERS",
    "BATCHGEN_GLM5_WHOLE_MODEL_GRAPH_COMPARE_ATOL",
    "BATCHGEN_GLM5_WHOLE_MODEL_GRAPH_COMPARE_RTOL",
)

_METHODS = (
    "_debug_flag_enabled",
    "_glm5_dispatch_trace_enabled",
    "_glm5_graph_path_log_requested_for_current_batch",
    "_glm5_whole_model_graph_compare_requested_for_current_batch",
    "_glm5_whole_model_graph_timing_requested_for_current_batch",
    "_glm5_whole_model_graph_compare_fail_on_mismatch",
    "_glm5_whole_model_graph_debug_value",
    "_glm5_whole_model_graph_compare_tolerance",
    "_glm5_whole_model_graph_probe_layers",
    "_compare_glm5_whole_model_graph_logits",
)


class _Worker:
    """Carries the compiled worker methods and the batch debug state."""


def _worker(debug: Optional[dict], *, wrapper_debug=None):
    wrapper = SimpleNamespace(batchgen_debug=wrapper_debug)
    methods = load_worker_methods(
        _METHODS, {"AttnWrapperBase": wrapper, "Optional": Optional}
    )
    worker = _Worker()
    for name, fn in methods.items():
        setattr(worker, name, fn.__get__(worker))
    worker._batchgen_debug = debug
    return worker


@pytest.fixture
def all_retired_env_vars_set(monkeypatch):
    for name in RETIRED_ENV_VARS:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("BATCHGEN_GLM5_WHOLE_MODEL_GRAPH_PROBE_LAYERS", "all")
    monkeypatch.setenv("BATCHGEN_GLM5_WHOLE_MODEL_GRAPH_COMPARE_ATOL", "0.5")
    monkeypatch.setenv("BATCHGEN_GLM5_WHOLE_MODEL_GRAPH_COMPARE_RTOL", "0.5")


# ------------------------------------------------------------ boolean keys


@pytest.mark.parametrize(
    "method,key",
    [
        ("_glm5_graph_path_log_requested_for_current_batch", "glm5_graph_path_log"),
        ("_glm5_whole_model_graph_compare_requested_for_current_batch", "glm5_whole_model_graph_compare"),
        ("_glm5_whole_model_graph_timing_requested_for_current_batch", "glm5_whole_model_graph_timing"),
        ("_glm5_whole_model_graph_compare_fail_on_mismatch", "glm5_whole_model_graph_compare_fail"),
    ],
)
def test_each_flag_follows_its_key_and_ignores_the_retired_env(
    method, key, all_retired_env_vars_set
):
    assert getattr(_worker({key: True}), method)()
    assert not getattr(_worker({}), method)()
    # The wrapper-level batch debug is the same channel (decode installs it).
    assert getattr(_worker(None, wrapper_debug={key: "1"}), method)()


def test_timing_also_follows_the_compare_key():
    assert _worker({"glm5_whole_model_graph_compare": True})._glm5_whole_model_graph_timing_requested_for_current_batch()


def test_dispatch_trace_follows_its_key_and_ignores_the_retired_env(all_retired_env_vars_set):
    worker = _worker({})
    assert worker._glm5_dispatch_trace_enabled({"glm5_dispatch_trace": True})
    assert not worker._glm5_dispatch_trace_enabled({})
    assert not worker._glm5_dispatch_trace_enabled(None)


# ---------------------------------------------------------- probe layers


def test_probe_layers_follow_the_key_with_the_env_grammar(all_retired_env_vars_set):
    key = "glm5_whole_model_graph_probe_layers"
    assert _worker({})._glm5_whole_model_graph_probe_layers(4) == ()
    assert _worker({key: "all"})._glm5_whole_model_graph_probe_layers(4) == (0, 1, 2, 3)
    assert _worker({key: " ALL "})._glm5_whole_model_graph_probe_layers(3) == (0, 1, 2)
    assert _worker({key: "3, 20,77"})._glm5_whole_model_graph_probe_layers(80) == (3, 20, 77)
    assert _worker({key: ""})._glm5_whole_model_graph_probe_layers(4) == ()
    assert _worker({key: [1, 2]})._glm5_whole_model_graph_probe_layers(4) == (1, 2)


@pytest.mark.parametrize("bad", [True, 3, {"layers": 1}, [1, "2"]])
def test_probe_layers_refuse_other_shapes_by_name(bad):
    with pytest.raises(ValueError, match="glm5_whole_model_graph_probe_layers"):
        _worker({"glm5_whole_model_graph_probe_layers": bad})._glm5_whole_model_graph_probe_layers(4)


# ------------------------------------------------------- compare tolerances


def _compare_kwargs(monkeypatch, worker):
    seen = {}

    def fake_compare(**kwargs):
        seen.update(kwargs)
        return {"ok": True}

    monkeypatch.setitem(
        sys.modules,
        "batchgen.models.glm.glm5.whole_model_cuda_graph_segments",
        SimpleNamespace(compare_glm5_whole_model_graph_logits=fake_compare),
    )
    worker._compare_glm5_whole_model_graph_logits(eager_logits="e", graph_logits="g")
    return seen


def test_compare_tolerances_default_to_the_old_env_defaults(monkeypatch, all_retired_env_vars_set):
    seen = _compare_kwargs(monkeypatch, _worker({}))
    assert seen == {"eager_logits": "e", "graph_logits": "g", "atol": 1e-2, "rtol": 1e-2}


def test_compare_tolerances_follow_their_keys(monkeypatch):
    seen = _compare_kwargs(monkeypatch, _worker({
        "glm5_whole_model_graph_compare_atol": 0.25,
        "glm5_whole_model_graph_compare_rtol": "3e-2",
    }))
    assert seen["atol"] == 0.25 and seen["rtol"] == 3e-2


@pytest.mark.parametrize("bad", [True, "loose", [1e-2]])
def test_compare_tolerances_refuse_non_numbers_by_name(bad):
    with pytest.raises(ValueError, match="glm5_whole_model_graph_compare_atol"):
        _worker({"glm5_whole_model_graph_compare_atol": bad})._glm5_whole_model_graph_compare_tolerance(
            "glm5_whole_model_graph_compare_atol"
        )


# ------------------------------------------------------ MoE graph compare


def _lift_models_function(name: str, namespace: dict):
    path = ROOT / "batchgen" / "models" / "glm" / "glm5" / "model.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize(
    "debug,expected",
    [
        ({"glm5_moe_graph_compare": True}, True),
        ({"glm5_moe_graph_compare": True, "glm5_moe_mode": "eager"}, False),
        ({}, False),
    ],
)
def test_moe_graph_compare_follows_its_key_and_ignores_the_retired_env(
    debug, expected, all_retired_env_vars_set
):
    def debug_flag_enabled(value):
        return value is True or value in ("1", 1)

    mode = debug.get("glm5_moe_mode")
    active = _lift_models_function(
        "_glm5_moe_graph_compare_active",
        {
            "_glm5_moe_debug_mode": lambda: mode,
            "_glm5_moe_debug_dict": lambda: debug,
            "_debug_flag_enabled": debug_flag_enabled,
        },
    )
    assert active() is expected


# --------------------------------------------- no reader of the retired vars


def test_no_retired_env_var_is_read_under_batchgen():
    hits = []
    for path in sorted((ROOT / "batchgen").rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value in RETIRED_ENV_VARS:
                hits.append(f"{path.relative_to(ROOT)}:{node.lineno}:{node.value}")
    assert not hits, hits
