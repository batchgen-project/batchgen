"""``prefill_pool_layout`` must answer in a phase window, model or no model.

The worker plans a pooled prefill wave at reservation time, which runs in the
decode phase: with the phase instances rebuilt rather than kept, the model has
been deep-freed and ``self.model`` is ``None``, so reading
``self.model.model.layers[...]`` there raised ``AttributeError: 'NoneType'
object has no attribute 'model'`` on every rank. The layout is a model
constant — the K/V widths are never sharded and the head count is sharded by
the fixed ``_attn_tp_size`` — so it is derived from the config and only
cross-checked against the module when one is resident.

The method is compiled straight out of the PSM's AST onto fakes (same approach
as the worker-seam tests), so nothing here imports the PSM, torch extensions
or CUDA.
"""

import ast
import functools
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest


PSM = (
    Path(__file__).resolve().parents[2] / "batchgen" / "models" / "moonshotai"
    / "kimi_linear" / "Parallel_Strategy_Manager.py"
)

# Layer 0 and 2 are KDA, the rest MLA: the layer_map must carry -1 for the KDA
# rows and dense indices for the MLA ones.
KDA_LAYERS = (0, 2)
NUM_LAYERS = 6


@functools.lru_cache(maxsize=1)
def _layout_ast():
    tree = ast.parse(PSM.read_text())
    psm = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "KimiLinearParallelStrategyManager"
    )
    return next(
        node for node in psm.body
        if isinstance(node, ast.FunctionDef) and node.name == "prefill_pool_layout"
    )


@functools.lru_cache(maxsize=1)
def _layout_method():
    module = ast.fix_missing_locations(
        ast.Module(body=[_layout_ast()], type_ignores=[])
    )
    namespace = {}
    exec(compile(module, str(PSM), "exec"), namespace)
    return namespace["prefill_pool_layout"]


def _config():
    cfg = SimpleNamespace(
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=32,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
    )
    cfg.is_mla_layer = lambda layer_idx: layer_idx not in KDA_LAYERS
    return cfg


def _attn(num_heads, *, k_nope=128, k_rope=64, v=128, wrapped=False):
    attn = SimpleNamespace(
        num_heads=num_heads, qk_nope_head_dim=k_nope,
        qk_rope_head_dim=k_rope, v_head_dim=v,
    )
    return SimpleNamespace(module=attn) if wrapped else attn


def _model(attn):
    layers = [
        SimpleNamespace(self_attn=attn) for _ in range(NUM_LAYERS)
    ]
    return SimpleNamespace(model=SimpleNamespace(layers=layers))


def _psm(*, attn_tp_size, model=None, cfg=None):
    psm = SimpleNamespace(
        loaded_model_config=cfg or _config(),
        _attn_tp_size=attn_tp_size,
        model=model,
    )
    psm.prefill_pool_layout = MethodType(_layout_method(), psm)
    return psm


EXPECTED_MAP = (-1, 0, -1, 1, 2, 3)


def test_layout_without_a_model_is_the_full_layout():
    layout = _psm(attn_tp_size=1).prefill_pool_layout()
    assert layout == {
        "num_layers": 4,
        "layer_map": EXPECTED_MAP,
        "num_kv_heads": 32,
        "k_head_dim": 192,
        "v_head_dim": 128,
    }


def test_layout_without_a_model_shards_the_heads_by_attn_tp_size():
    layout = _psm(attn_tp_size=8).prefill_pool_layout()
    assert layout["num_kv_heads"] == 4
    # Widths are per head and never sharded.
    assert (layout["k_head_dim"], layout["v_head_dim"]) == (192, 128)


def test_a_resident_model_returns_the_same_layout():
    without = _psm(attn_tp_size=8).prefill_pool_layout()
    with_model = _psm(
        attn_tp_size=8, model=_model(_attn(4))
    ).prefill_pool_layout()
    assert with_model == without


def test_a_resident_wrapped_module_is_unwrapped():
    layout = _psm(
        attn_tp_size=1, model=_model(_attn(32, wrapped=True))
    ).prefill_pool_layout()
    assert layout["num_kv_heads"] == 32


@pytest.mark.parametrize("attn,expected", [
    (_attn(16), "num_kv_heads module=16 config=32"),
    (_attn(32, k_nope=64), "k_head_dim module=128 config=192"),
    (_attn(32, v=256), "v_head_dim module=256 config=128"),
])
def test_a_module_that_disagrees_with_the_config_is_refused(attn, expected):
    psm = _psm(attn_tp_size=1, model=_model(attn))
    with pytest.raises(RuntimeError) as excinfo:
        psm.prefill_pool_layout()
    message = str(excinfo.value)
    assert "disagrees with the resident MLA module" in message
    assert "attn_tp_size=1" in message
    assert expected in message


def test_the_first_mla_layer_is_the_one_inspected():
    # Layer 0 is KDA here, so a KDA module must never be read: the fake only
    # exposes MLA attributes, and reading index 0 would pick the same object,
    # so assert the chosen index instead.
    body = ast.unparse(_layout_ast())
    assert "self.model.model.layers[mla_layers[0]].self_attn" in body


def test_the_config_derivation_runs_before_the_model_is_touched():
    body = ast.unparse(_layout_ast())
    derived = body.index("num_kv_heads': heads")
    guard = body.index("if self.model is None")
    touch = body.index("self.model.model.layers[mla_layers[0]]")
    assert derived < guard < touch
