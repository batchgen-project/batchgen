"""KDA state pools must survive a model rebuild, or refuse by name.

For a linear-attention model a sequence's recurrent + conv state IS its KV:
the prefill writes it, every decode step reads it. ``init_state_pools`` runs
inside the PSM's ``_build_model``, which runs AGAIN on a prefill->decode
switch that does not keep its phase instances
(``deep_free_model_memory`` then ``if self.model is None: self._build_model()``
in ``configure_decoding``). Building a second ``KDAStateGPUManager`` there
replaces the sequence->slot map with an empty one, and the first decode step
raises ``KeyError: Sequence <global_idx> has no KDA state item`` for every
sequence the prefill just computed.

``kda_state_pool_decision`` is pure, so the rebuild is reproduced here without
CUDA — and without importing ``batchgen.kv_cache``, which needs triton. The
call site is checked on the module's AST.
"""

import ast
import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

from batchgen.models.moonshotai.kimi_linear.wrappers import (
    kda_state_pool_decision,
)


WRAPPERS = (
    Path(__file__).resolve().parents[2] / "batchgen" / "models" / "moonshotai"
    / "kimi_linear" / "wrappers.py"
)


@dataclass(frozen=True)
class _Config:
    """Stands in for KDAStateGPUConfig: a frozen, comparable layout."""

    num_kda_layers: int = 20
    num_state_items: int = 272
    num_heads: int = 4
    head_dim: int = 128
    conv_dim: Optional[int] = 512
    conv_width: int = 4


CPU = "cpu"


def _decide(existing, live, wanted, *, existing_device=CPU, wanted_device=CPU):
    return kda_state_pool_decision(
        existing_config=existing,
        existing_device=None if existing is None else existing_device,
        live_sequences=live,
        wanted_config=wanted,
        wanted_device=wanted_device,
    )


def test_first_build_rebuilds():
    assert _decide(None, 0, _Config()) == "rebuild"


def test_identical_layout_is_reused_so_live_state_survives():
    # The failing shape: a 32-sequence prefill wrote state, then the phase
    # switch rebuilt the model with the very same layout.
    assert _decide(_Config(), 32, _Config()) == "reuse"


def test_identical_layout_with_nothing_live_is_still_reused():
    # Reuse keeps the pools' addresses fixed, which CUDA-graph capture needs.
    assert _decide(_Config(), 0, _Config()) == "reuse"


@pytest.mark.parametrize("field,value", [
    ("num_kda_layers", 21),
    ("num_state_items", 136),
    ("num_heads", 8),
    ("head_dim", 64),
    ("conv_dim", 1024),
    ("conv_width", 5),
])
def test_changed_layout_with_live_state_is_refused(field, value):
    wanted = _Config(**{field: value})
    with pytest.raises(RuntimeError) as excinfo:
        _decide(_Config(), 7, wanted)
    message = str(excinfo.value)
    assert "7 sequences still hold state" in message
    assert "cannot be recomputed" in message


def test_changed_device_with_live_state_is_refused():
    with pytest.raises(RuntimeError, match="3 sequences still hold state"):
        _decide(_Config(), 3, _Config(), existing_device="cpu",
                wanted_device="cuda:0")


def test_changed_layout_with_nothing_live_rebuilds():
    assert _decide(_Config(), 0, _Config(num_state_items=136)) == "rebuild"


def test_changed_device_with_nothing_live_rebuilds():
    assert _decide(_Config(), 0, _Config(), existing_device="cpu",
                   wanted_device="cuda:0") == "rebuild"


# ----------------------------------------------------------- call site --- #


@functools.lru_cache(maxsize=1)
def _init_state_pools_ast():
    tree = ast.parse(WRAPPERS.read_text())
    wrapper = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KimiLinearKDAWrapper"
    )
    return next(
        node for node in wrapper.body
        if isinstance(node, ast.FunctionDef) and node.name == "init_state_pools"
    )


def test_init_state_pools_routes_through_the_decision():
    body = ast.unparse(_init_state_pools_ast())
    assert "kda_state_pool_decision(" in body
    assert "live_sequences=live" in body


def test_a_new_manager_is_built_only_when_the_decision_says_rebuild():
    node = _init_state_pools_ast()
    constructions = [
        item for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Name)
        and item.func.id == "KDAStateGPUManager"
    ]
    assert len(constructions) == 1
    guarded = [
        item for item in ast.walk(node)
        if isinstance(item, ast.If)
        and "decision == 'reuse'" in ast.unparse(item.test)
        and "KDAStateGPUManager(" in ast.unparse(item.orelse)
        and "KDAStateGPUManager(" not in ast.unparse(item.body)
    ]
    assert len(guarded) == 1


def test_the_facade_and_the_views_are_rebuilt_on_both_paths():
    # The model is new, so its layers need fresh views; they come from the
    # same pools, so the addresses the graphs baked do not move.
    body = ast.unparse(_init_state_pools_ast())
    facade = body.index("cls.slot_manager = KDASlotManager(cls.state_manager)")
    pools = body.index("cls.layer_pools = {}")
    decision = body.index("decision = kda_state_pool_decision(")
    assert decision < facade < pools
