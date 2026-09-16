import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
PSM = ROOT / "batchgen/models/glm/glm5/Parallel_Strategy_Manager.py"
MODEL = ROOT / "batchgen/models/glm/glm5/model.py"
WRAPPERS = ROOT / "batchgen/models/glm/glm5/wrappers.py"


def _load_init_method(import_failures=None, model_import_failures=None):
    tree = ast.parse(PSM.read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "GLM5ParallelStrategyManager"
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_init_fused_kernels"
    )
    isolated_class = ast.ClassDef(
        name="Manager",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[isolated_class], type_ignores=[]))
    namespace = {
        "logging": logging,
        "_required_dsa_kernel_import_failures": lambda: import_failures or {},
        "_required_dsa_model_kernel_import_failures": (
            lambda: model_import_failures or {}
        ),
    }
    exec(compile(module, str(PSM), "exec"), namespace)
    return namespace["Manager"]._init_fused_kernels


def _wrapper(*, active=True, wp2=True, wp4=True, wp5=True, calls=None):
    calls = calls if calls is not None else []
    wrapper = SimpleNamespace(
        module=SimpleNamespace(indexer=object() if active else None),
        _indexer_cuda_weights=object() if wp2 else None,
        _fused_wqb_weights=object() if wp4 else None,
        _fp8_absorb_weights=object() if wp5 else None,
    )
    wrapper.initialize_fused_kernels = lambda: calls.append(wrapper)
    return wrapper


def _manager(wrappers, phase="prefill"):
    layers = [SimpleNamespace(self_attn=wrapper) for wrapper in wrappers]
    return SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(layers=layers)),
        loaded_model_config=SimpleNamespace(phase=phase),
        rank=0,
    )


def test_dense_mla_skips_required_dsa_kernel_check():
    calls = []
    wrapper = _wrapper(active=False, wp2=False, wp4=False, wp5=False, calls=calls)
    method = _load_init_method({"WP2": ImportError("missing")})

    method(_manager([wrapper]))

    assert calls == []


def test_missing_required_import_fails_before_initialization():
    calls = []
    wrapper = _wrapper(calls=calls)
    method = _load_init_method({"WP5 FP8 absorb": ImportError("missing symbol")})

    with pytest.raises(RuntimeError, match="WP5 FP8 absorb.*missing symbol"):
        method(_manager([wrapper]))

    assert calls == []


def test_missing_fused_rope_hadamard_fails_before_initialization():
    calls = []
    wrapper = _wrapper(calls=calls)
    method = _load_init_method(
        model_import_failures={
            "fused RoPE+Hadamard": ImportError("extension unavailable")
        }
    )

    with pytest.raises(RuntimeError, match="fused RoPE.Hadamard.*extension unavailable"):
        method(_manager([wrapper]))

    assert calls == []


def test_prefill_requires_wp2_wp4_only_on_active_indexer_layers():
    calls = []
    active = _wrapper(wp5=False, calls=calls)
    shared = _wrapper(active=False, wp2=False, wp4=False, wp5=False, calls=calls)

    _load_init_method()(_manager([active, shared], phase="prefill"))

    assert calls == [active]


def test_missing_wp2_and_wp4_fail_with_exact_layer_ids():
    wrappers = [
        _wrapper(wp2=False),
        _wrapper(active=False, wp2=False, wp4=False),
        _wrapper(wp4=False),
    ]

    with pytest.raises(RuntimeError) as exc_info:
        _load_init_method()(_manager(wrappers, phase="prefill"))

    message = str(exc_info.value)
    assert "WP2 fused indexer KV projection missing on layers [0]" in message
    assert "WP4 fused indexer scoring missing on layers [2]" in message


def test_decode_requires_wp5_on_every_active_indexer_layer():
    wrappers = [_wrapper(), _wrapper(wp5=False), _wrapper(active=False, wp5=False)]

    with pytest.raises(RuntimeError, match=r"WP5 FP8 absorb missing on layers \[1\]"):
        _load_init_method()(_manager(wrappers, phase="decode"))


def test_runtime_dsa_fallbacks_are_fail_closed():
    model_source = " ".join(MODEL.read_text().split())
    wrappers_source = " ".join(WRAPPERS.read_text().split())

    assert (
        '"GLM-5 DSA requires fused RoPE+Hadamard; separate-op fallback " '
        '"is disabled"'
    ) in model_source
    assert "GLM-5 DSA requires WP5 FP8" in wrappers_source
    assert "out_absorb; PyTorch/BF16 fallback is disabled" in wrappers_source
