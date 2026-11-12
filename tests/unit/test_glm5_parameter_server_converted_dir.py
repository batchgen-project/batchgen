"""The GLM5 parameter server honors a caller-provided converted_ckpt_dir.

A pre-converted checkpoint can live outside the cache dir (for example on
node-local storage). The server must hand that directory to the converter as
the conversion's output directory instead of silently reconverting into
``<cache_dir>/converted_ckpt`` — the converter then validates and reuses it.
The resolution lives in the leaf module ``converted_ckpt`` so this contract is
testable without the engine runtime; the parameter server's use of it is
pinned by a source assertion because importing that module needs the native
runtime.
"""

import ast
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def recorded_calls(monkeypatch):
    calls = []

    class _FakeConverter:
        def convert_model_directory(self, input_dir, output_dir=None, **kwargs):
            calls.append({"input_dir": input_dir, "output_dir": output_dir})
            return output_dir if output_dir is not None else f"{input_dir}/converted_ckpt"

    converter_module = importlib.import_module(
        "batchgen.ckpt_converter.ckpt_converter"
    )
    monkeypatch.setattr(converter_module, "ckpt_converter", _FakeConverter)
    return calls


def test_provided_converted_dir_is_the_conversion_output(recorded_calls):
    from batchgen.models.glm.glm5.converted_ckpt import resolve_converted_ckpt_dir

    resolved = resolve_converted_ckpt_dir(
        "/cache/GLM-5.3", "/local/glm53_converted_ckpt"
    )
    assert resolved == "/local/glm53_converted_ckpt"
    assert recorded_calls == [
        {"input_dir": "/cache/GLM-5.3", "output_dir": "/local/glm53_converted_ckpt"}
    ]


def test_path_valued_converted_dir_is_normalized_to_str(recorded_calls):
    from pathlib import Path

    from batchgen.models.glm.glm5.converted_ckpt import resolve_converted_ckpt_dir

    resolved = resolve_converted_ckpt_dir(
        "/cache/GLM-5.3", Path("/local/glm53_converted_ckpt")
    )
    assert resolved == "/local/glm53_converted_ckpt"
    assert type(resolved) is str
    assert recorded_calls == [
        {"input_dir": "/cache/GLM-5.3", "output_dir": "/local/glm53_converted_ckpt"}
    ]


def test_absent_converted_dir_keeps_the_cache_default(recorded_calls):
    from batchgen.models.glm.glm5.converted_ckpt import resolve_converted_ckpt_dir

    resolved = resolve_converted_ckpt_dir("/cache/GLM-5.3", None)
    assert resolved == "/cache/GLM-5.3/converted_ckpt"
    assert recorded_calls == [{"input_dir": "/cache/GLM-5.3", "output_dir": None}]


def test_parameter_server_resolves_through_the_leaf_module():
    """Source-level pin: GLM5_Parameter_Server calls resolve_converted_ckpt_dir
    with (cache_dir, converted_ckpt_dir) and assigns the result back — the
    module itself needs the native runtime, so the wiring is pinned by AST."""
    source_path = (
        Path(__file__).resolve().parents[2]
        / "batchgen"
        / "models"
        / "glm"
        / "glm5"
        / "glm5_parameter_server.py"
    )
    tree = ast.parse(source_path.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "resolve_converted_ckpt_dir"
        ):
            target = node.targets[0]
            assert isinstance(target, ast.Attribute)
            assert target.attr == "converted_ckpt_dir"
            args = [ast.dump(a) for a in node.value.args]
            assert args == [
                ast.dump(
                    ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()),
                        attr="cache_dir",
                        ctx=ast.Load(),
                    )
                ),
                ast.dump(
                    ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()),
                        attr="converted_ckpt_dir",
                        ctx=ast.Load(),
                    )
                ),
            ]
            break
    else:
        pytest.fail(
            "GLM5_Parameter_Server no longer resolves converted_ckpt_dir through "
            "resolve_converted_ckpt_dir(self.cache_dir, self.converted_ckpt_dir)"
        )
