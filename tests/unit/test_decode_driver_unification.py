"""Step 3 of the three-step extraction: one decode driver, no forward patch.

The worker's decode driver (``decode_engine.DecoupledGpuDecode.run``) owns the
per-step forward dispatch for every family. The Kimi-Linear whole-model replay
goes through the driver's step API (``begin_decode_step`` /
``replay_whole_model_step`` / ``end_decode_step``); the former
``KimiLinearModel.forward`` patch that hid that dispatch inside the model is
retired, so the driver's dispatch is the only decode path.

These are the behaviour tests that replace the retired mechanism's pins:

* the patch mechanism's text/attributes no longer exist, and ``install()``
  leaves the model-level forward untouched (identity check on the callable);
* the step API reproduces the retired hook's decisions (validation, fallback,
  flag lifecycle, replay + lm_head pairing) — runtime oracles on the real
  methods;
* the driver spells the dispatch in the pre-change order: arm, forward
  (replay or the one eager call), disarm in a ``finally``, then one token
  selection — pinned structurally on the real ``run()`` source.

CPU-only: the policy object is driven through isolated instances; no CUDA.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "batchgen" / "worker" / "decode_engine.py"
SEGMENTS = (
    ROOT / "batchgen" / "models" / "moonshotai" / "kimi_linear"
    / "cuda_graph_segments.py"
)
WORKER = ROOT / "batchgen" / "batchgen_worker.py"
PSM = (
    ROOT / "batchgen" / "models" / "moonshotai" / "kimi_linear"
    / "Parallel_Strategy_Manager.py"
)

ENGINE_SOURCE = ENGINE.read_text()
SEGMENTS_SOURCE = SEGMENTS.read_text()


def _class_method(source: str, class_name: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    klass = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node for node in klass.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _compile_methods(source_path: Path, class_name: str, names, namespace):
    """Lift methods out of a class body and compile them against fakes."""
    source = source_path.read_text()
    out = {}
    for name in names:
        node = _class_method(source, class_name, name)
        module = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(module)
        ns = dict(namespace)
        exec(compile(module, str(source_path), "exec"), ns)
        out[name] = ns[name]
    return out


# --------------------------------------------------------------------------- #
#  (1) the retired patch mechanism no longer exists
# --------------------------------------------------------------------------- #


def test_the_model_forward_patch_mechanism_is_gone():
    assert "_make_model_forward" not in SEGMENTS_SOURCE
    assert "_orig_model_forward" not in SEGMENTS_SOURCE


def test_install_never_touches_the_model_level_forward():
    """``install`` patches layer forwards only; ``release`` restores only
    what install patched. Neither assigns ``inner.forward``."""
    for name in ("install", "release"):
        node = _class_method(SEGMENTS_SOURCE, "KimiLinearDecodeGraph", name)
        for stmt in ast.walk(node):
            if not isinstance(stmt, ast.Assign):
                continue
            for target in stmt.targets:
                assert ast.unparse(target) != "inner.forward", (
                    f"{name} must not patch the model-level forward; the "
                    "decode driver owns the step dispatch"
                )


def test_install_runtime_leaves_the_model_forward_callable_identical():
    """Runtime twin of the AST check: after install, ``model.model.forward``
    IS the original callable (identity), while the layer hook is installed."""

    class SlotManager:
        def __init__(self):
            self.owners = {}

        def alloc(self, seq_id):
            return self.owners.setdefault(seq_id, len(self.owners))

    original_model_forward = object()
    original_layer_forward = object()
    inner = SimpleNamespace(
        forward=original_model_forward,
        layers=[SimpleNamespace(forward=original_layer_forward)],
    )
    graph = SimpleNamespace(
        _installed=False,
        _scratch_slot=None,
        _uses_block_residual=False,
        model=SimpleNamespace(model=inner),
        _orig_new_block_residual=None,
        _orig_layer_forwards={},
        _make_layer_forward=lambda layer_idx, orig: ("layer", layer_idx, orig),
        rank=1,
        mode="eager",
        bucketing=SimpleNamespace(bucket_sizes=[1]),
        compare_every=64,
    )
    install = _compile_methods(
        SEGMENTS, "KimiLinearDecodeGraph", ["install"],
        {
            "KimiLinearKDAWrapper": SimpleNamespace(slot_manager=SlotManager()),
            "GRAPH_SCRATCH_SEQ_ID": -1_000_001,
            "logger": SimpleNamespace(info=lambda *a, **k: None),
        },
    )["install"].__get__(graph)
    install()
    assert inner.forward is original_model_forward
    assert inner.layers[0].forward == ("layer", 0, original_layer_forward)
    assert graph._orig_layer_forwards == {0: original_layer_forward}


# --------------------------------------------------------------------------- #
#  (2) the step API reproduces the retired hook's decisions (runtime oracle)
# --------------------------------------------------------------------------- #


def _policy(whole_active, bsz, calls):
    """An isolated policy instance whose ``_begin_step`` arms a known state."""
    methods = _compile_methods(
        SEGMENTS,
        "KimiLinearDecodeGraph",
        ["begin_decode_step", "replay_whole_model_step", "end_decode_step"],
        {"torch": torch},
    )
    policy = SimpleNamespace(
        _bsz=bsz,
        _fallbacks=[],
    )

    def _begin_step():
        calls.append("begin_step")
        policy._step_active = True
        policy._moe_step_active = False
        policy._whole_step_active = whole_active

    policy._begin_step = _begin_step
    policy._fallback = lambda reason: policy._fallbacks.append(reason)
    hidden = torch.zeros(bsz, 1, 4)

    def _run_whole_model(input_ids):
        calls.append(("replay", input_ids))
        return hidden

    policy._run_whole_model = _run_whole_model
    logits = torch.zeros(bsz, 1, 8)

    def lm_head(hidden_states):
        calls.append(("lm_head", hidden_states))
        return logits

    policy.model = SimpleNamespace(lm_head=lm_head)
    policy.begin_decode_step = methods["begin_decode_step"].__get__(policy)
    policy.replay_whole_model_step = (
        methods["replay_whole_model_step"].__get__(policy)
    )
    policy.end_decode_step = methods["end_decode_step"].__get__(policy)
    policy._expected_hidden = hidden
    policy._expected_logits = logits
    return policy


def test_begin_arms_the_whole_step_for_valid_decode_input():
    calls = []
    policy = _policy(True, bsz=2, calls=calls)
    assert policy.begin_decode_step(torch.zeros(2, 1, dtype=torch.long)) is True
    assert policy._whole_step_active and policy._step_active
    assert policy._fallbacks == []
    assert calls == ["begin_step"]


@pytest.mark.parametrize("bad_input", [
    None,                                    # not a tensor
    torch.zeros(2, dtype=torch.long),        # 1-D
    torch.zeros(2, 2, dtype=torch.long),     # seq dim != 1
    torch.zeros(3, 1, dtype=torch.long),     # rows != local bsz
])
def test_begin_falls_back_to_eager_on_invalid_input(bad_input):
    """The retired hook's validation: a non-[local_bsz, 1] input disarms the
    whole step AND the per-layer arm, and records the same fallback reason."""
    calls = []
    policy = _policy(True, bsz=2, calls=calls)
    assert policy.begin_decode_step(bad_input) is False
    assert not policy._whole_step_active
    assert not policy._step_active
    assert policy._fallbacks == [
        "whole-model graph requires decode input_ids [local_bsz, 1]"
    ]


def test_begin_keeps_the_per_layer_arm_when_whole_is_not_armed():
    """A per-layer step (compare mode, missing whole bucket) returns False to
    the driver — run the eager forward — while ``_step_active`` stays armed so
    the layer hooks replay, exactly as under the retired hook."""
    calls = []
    policy = _policy(False, bsz=2, calls=calls)
    assert policy.begin_decode_step(torch.zeros(2, 1, dtype=torch.long)) is False
    assert policy._step_active
    assert not policy._whole_step_active


def test_replay_pairs_the_graph_body_with_the_lm_head():
    """The retired hook returned hidden states for the CausalLM's ``lm_head``;
    the step API keeps that exact pair (identity on both tensors)."""
    calls = []
    policy = _policy(True, bsz=2, calls=calls)
    input_ids = torch.zeros(2, 1, dtype=torch.long)
    assert policy.begin_decode_step(input_ids) is True
    out = policy.replay_whole_model_step(input_ids)
    assert out is policy._expected_logits
    assert calls == [
        "begin_step",
        ("replay", input_ids),
        ("lm_head", policy._expected_hidden),
    ]
    assert calls[1][1] is input_ids
    assert calls[2][1] is policy._expected_hidden


def test_end_decode_step_is_the_retired_finally():
    calls = []
    policy = _policy(True, bsz=2, calls=calls)
    policy.begin_decode_step(torch.zeros(2, 1, dtype=torch.long))
    policy.end_decode_step()
    assert not policy._step_active
    assert not policy._moe_step_active
    assert not policy._whole_step_active


# --------------------------------------------------------------------------- #
#  (3) the driver's dispatch is the only path, in the pre-change order
# --------------------------------------------------------------------------- #


def _decode_run():
    return _class_method(ENGINE_SOURCE, "DecoupledGpuDecode", "run")


def _calls_of(node, attr):
    return [
        n for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == attr
    ]


def test_the_driver_owns_exactly_one_step_dispatch():
    run = _decode_run()
    assert len(_calls_of(run, "begin_decode_step")) == 1
    assert len(_calls_of(run, "replay_whole_model_step")) == 1
    assert len(_calls_of(run, "end_decode_step")) == 1
    assert len(_calls_of(run, "_glm5_decode_model_forward")) == 1
    assert len(_calls_of(run, "_read_decode_step_graph")) == 1


def test_the_dispatch_order_matches_the_retired_hook():
    """arm -> (replay | the one eager forward) -> disarm in a finally ->
    one token selection, all inside the step's weight pass."""
    run = _decode_run()
    begin = _calls_of(run, "begin_decode_step")[0]
    replay = _calls_of(run, "replay_whole_model_step")[0]
    eager = _calls_of(run, "_glm5_decode_model_forward")[0]
    end = _calls_of(run, "end_decode_step")[0]

    # The replay and the eager forward sit in one Try whose finally disarms.
    tries = [
        n for n in ast.walk(run)
        if isinstance(n, ast.Try)
        and any(c is end for f in n.finalbody for c in ast.walk(f))
    ]
    assert len(tries) == 1
    dispatch = tries[0]
    body_calls = [c for s in dispatch.body for c in ast.walk(s)]
    assert replay in body_calls and eager in body_calls

    # Order: arm strictly before the forward; disarm after both arms.
    assert begin.lineno < dispatch.lineno
    assert end.lineno > max(replay.lineno, eager.lineno)

    # Both arms feed the same single token selection after the dispatch.
    selects = [
        c for c in _calls_of(run, "_select_tokens")
        if c.lineno > end.lineno
    ]
    assert selects, "the dispatch must be followed by the step's selection"
    select = min(selects, key=lambda c: c.lineno)

    # The whole dispatch runs inside the step's weight pass.
    pass_begins = [
        c for c in _calls_of(run, "begin")
        if ast.unparse(c.func) == "self._weight_passes.begin"
    ]
    pass_ends = [
        c for c in _calls_of(run, "end")
        if ast.unparse(c.func) == "self._weight_passes.end"
    ]
    assert any(c.lineno < begin.lineno for c in pass_begins)
    assert any(c.lineno > select.lineno for c in pass_ends)


def test_the_worker_resolves_the_policy_through_the_declared_accessor():
    """``_read_decode_step_graph`` goes through the strategy manager's
    ``decode_step_graph()`` accessor and answers None without one."""
    import sys
    sys.path.insert(0, str(ROOT))
    try:
        from tests.worker_method_source import load_worker_methods
    finally:
        sys.path.pop(0)
    read = load_worker_methods(["_read_decode_step_graph"], {})[
        "_read_decode_step_graph"
    ]
    handle = object()
    with_policy = SimpleNamespace(
        parallel_manager=SimpleNamespace(decode_step_graph=lambda: handle)
    )
    without = SimpleNamespace(parallel_manager=SimpleNamespace())
    assert read(with_policy) is handle
    assert read(without) is None


def test_the_strategy_manager_accessor_returns_its_decode_graph():
    node = _class_method(
        PSM.read_text(), "KimiLinearParallelStrategyManager", "decode_step_graph"
    )
    returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
    assert [ast.unparse(r.value) for r in returns] == ["self._decode_graph"]
