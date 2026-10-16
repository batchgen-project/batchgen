"""A rebuilt model must stage its own resident-EP decode shard.

``_resident_ep_built`` records that the resident expert shard is attached to
``layer.block_sparse_moe._resident_ep_moe``. The worker's main (non-persistent)
path destroys the model at every phase switch
(``deep_free_model_memory``) and the next ``configure_*`` rebuilds it, so the
flag was a claim about a model that no longer existed: the second
``configure_decoding`` skipped ``_init_resident_ep_decode()``, the new MoE
blocks carried no resident layer, and decode fell through to the streamed
expert wrapper whose decode weight-copy task is empty under resident EP —
``RuntimeError: GPU_Weight_Buffer get_weights()`` on every rank. The first
cycle worked because the shard was genuinely built there.

The PSM methods are compiled out of its AST onto fakes, so nothing here
imports the PSM, torch extensions or CUDA.
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


@functools.lru_cache(maxsize=1)
def _psm_class():
    tree = ast.parse(PSM.read_text())
    return next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "KimiLinearParallelStrategyManager"
    )


def _method_ast(name):
    return next(
        node for node in _psm_class().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


@functools.lru_cache(maxsize=8)
def _methods(names):
    module = ast.fix_missing_locations(
        ast.Module(body=[_method_ast(name) for name in names], type_ignores=[])
    )
    namespace = {}
    exec(compile(module, str(PSM), "exec"), namespace)
    return {name: namespace[name] for name in names}


class _Graph:
    def __init__(self):
        self.released = 0

    def release(self):
        self.released += 1


def _model(*, attached):
    layers = [
        SimpleNamespace(block_sparse_moe=SimpleNamespace(
            _resident_ep_moe=object() if attached else None
        ))
        for _ in range(3)
    ]
    return SimpleNamespace(model=SimpleNamespace(layers=layers))


def _psm(*, model=None, built=False, graph=None, buffer=None):
    psm = SimpleNamespace(
        model=model,
        _resident_ep_built=built,
        _streamed_sp8_buffer=buffer,
        _decode_graph=graph,
    )
    for name, function in _methods((
        "_reset_model_attached_state",
        "_assert_resident_ep_attached",
        "release_decode_graph",
        "_init_resident_ep_decode",
    )).items():
        setattr(psm, name, MethodType(function, psm))
    return psm


# ------------------------------------------------------- the reset itself --- #


def test_reset_drops_every_attachment_claim():
    graph = _Graph()
    psm = _psm(model=_model(attached=True), built=True, graph=graph,
               buffer=object())
    psm._reset_model_attached_state()
    assert psm._resident_ep_built is False
    assert psm._streamed_sp8_buffer is None
    # The captured graph swapped the OLD model's forwards, so it is released,
    # not merely dropped.
    assert (graph.released, psm._decode_graph) == (1, None)


def test_reset_is_a_noop_on_a_first_build():
    psm = _psm(model=_model(attached=False))
    psm._reset_model_attached_state()
    assert psm._resident_ep_built is False
    assert psm._decode_graph is None


# --------------------------------------------------- the checked claim --- #


def test_skipping_the_build_requires_the_shard_to_be_attached():
    psm = _psm(model=_model(attached=True), built=True)
    # Returns without touching _comm / core_engine: nothing to build.
    psm._init_resident_ep_decode()
    assert psm._resident_ep_built is True


def test_a_flag_that_survived_a_rebuild_is_refused_by_name():
    psm = _psm(model=_model(attached=False), built=True)
    with pytest.raises(RuntimeError) as excinfo:
        psm._init_resident_ep_decode()
    message = str(excinfo.value)
    assert "marked built but the live model's MoE blocks carry no resident" in message
    assert "survived a model rebuild" in message


def test_a_flag_without_a_model_is_refused_by_name():
    psm = _psm(model=None, built=True)
    with pytest.raises(RuntimeError, match="holds no model"):
        psm._init_resident_ep_decode()


def test_layers_without_a_moe_block_are_skipped():
    psm = _psm(model=None, built=True)
    psm.model = SimpleNamespace(model=SimpleNamespace(layers=[
        SimpleNamespace(block_sparse_moe=None),
        SimpleNamespace(block_sparse_moe=SimpleNamespace(
            _resident_ep_moe=object()
        )),
    ]))
    psm._assert_resident_ep_attached()


# ------------------------------------------- the two-cycle state machine --- #


def test_the_second_decode_of_the_rebuild_path_stages_the_shard_again():
    # Cycle 1: a model is built, the decode phase stages the shard.
    psm = _psm(model=_model(attached=False))
    psm._reset_model_attached_state()          # at _build_model
    assert psm._resident_ep_built is False     # => configure_decoding builds
    psm.model = _model(attached=True)
    psm._resident_ep_built = True
    psm._init_resident_ep_decode()             # second call in the same cycle
    assert psm._resident_ep_built is True      # idempotent, nothing rebuilt

    # Phase switch on the main path: the model is destroyed, then rebuilt.
    psm.model = None
    psm.model = _model(attached=False)
    psm._reset_model_attached_state()           # at _build_model
    # Cycle 2 must stage it again rather than trust the old claim.
    assert psm._resident_ep_built is False


# ------------------------------------------------------------ call site --- #


def test_build_model_resets_before_it_loads_anything():
    body = ast.unparse(_method_ast("_build_model"))
    assign = body.index("self.model = KimiLinearForCausalLM(cfg)")
    reset = body.index("self._reset_model_attached_state()")
    skeleton = body.index("self._load_model_skeleton()")
    pools = body.index("KimiLinearKDAWrapper.init_state_pools(")
    assert assign < reset < skeleton < pools


def test_the_resident_build_checks_the_claim_on_the_skip_path():
    body = ast.unparse(_method_ast("_init_resident_ep_decode"))
    guard = body.index("if self._resident_ep_built:")
    check = body.index("self._assert_resident_ep_attached()")
    build = body.index("_resident_ep_built = True")
    assert guard < check < build
