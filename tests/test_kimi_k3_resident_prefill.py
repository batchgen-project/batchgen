import ast
import textwrap
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PSM = (
    ROOT
    / "batchgen"
    / "models"
    / "moonshotai"
    / "kimi_linear"
    / "Parallel_Strategy_Manager.py"
)


def _class_method_source(path, class_name, method_name):
    source = path.read_text()
    tree = ast.parse(source)
    cls = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    fn = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    lines = source.splitlines(keepends=True)
    return textwrap.dedent("".join(lines[fn.lineno - 1 : fn.end_lineno]))


def _psm_module_constant(name):
    """A literal module-level constant of the K3 manager, read from its AST."""
    tree = ast.parse(PSM.read_text())
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(target, "id", None) == name for target in node.targets)
    )
    return ast.literal_eval(node.value)


def _psm_stub(**attrs):
    namespace = {"PREFILL_MOE_MODES": _psm_module_constant("PREFILL_MOE_MODES")}
    for name in (
        "set_prefill_moe_mode",
        "default_prefill_moe_mode",
        "release_decode_graph",
        "prefill_uses_resident_ep",
    ):
        source = _class_method_source(
            PSM, "KimiLinearParallelStrategyManager", name
        )
        exec(source, namespace)

    class _PSM:
        pass

    for name in (
        "set_prefill_moe_mode",
        "default_prefill_moe_mode",
        "release_decode_graph",
        "prefill_uses_resident_ep",
    ):
        setattr(_PSM, name, namespace[name])
    obj = _PSM()
    obj._is_k3 = True
    obj._distributed_weight_sharded = True
    obj._resident_ep_built = False
    obj._attn_tp_size = 8
    obj._stream_all_modules = False
    obj._prefill_moe_mode = "streamed_sp8"
    obj._latent_tp_planned = False
    obj._decode_graph = None
    obj._stream_span = 8
    obj.engine_config = type("_Config", (), {})()
    obj.engine_config.Basic_Config = type("_Basic", (), {})()
    obj.engine_config.Basic_Config.prefill_moe_mode = None
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


def test_distributed_weights_accept_resident_ep_and_refuse_streamed():
    # A host stores its own ranks' EP experts, so a resident shard is always
    # local: resident_ep is accepted before the shard exists (the server
    # option builds it at startup). Only the legacy replicated path, which
    # requests every expert by name, is refused.
    psm = _psm_stub()
    psm.set_prefill_moe_mode("resident_ep")
    assert psm.prefill_uses_resident_ep()
    psm.set_prefill_moe_mode("streamed_sp8")
    assert psm._prefill_moe_mode == "streamed_sp8"
    with pytest.raises(ValueError, match="streamed_sp8' or 'resident_ep"):
        psm.set_prefill_moe_mode("streamed")


def test_streamed_sp8_needs_a_planned_stream_span():
    psm = _psm_stub(_distributed_weight_sharded=False, _stream_span=0)
    with pytest.raises(ValueError, match="k3_prefill_stream_span"):
        psm.set_prefill_moe_mode("streamed_sp8")


def test_release_decode_graph_drops_the_captured_graphs():
    class _Graph:
        released = False

        def release(self):
            self.released = True

    graph = _Graph()
    psm = _psm_stub(_decode_graph=graph)
    psm.release_decode_graph()
    assert graph.released and psm._decode_graph is None
    psm.release_decode_graph()


def test_latent_sharded_single_node_defaults_to_streamed_sp8_and_rejects_streamed():
    # Single node (replicated host weights) with TP-sharded LatentMoE
    # projections: the legacy ``streamed`` path would feed each rank's latent
    # column slice to full-width experts, so the default must be streamed_sp8.
    psm = _psm_stub(_distributed_weight_sharded=False, _latent_tp_planned=True)
    assert psm.default_prefill_moe_mode() == "streamed_sp8"
    with pytest.raises(ValueError, match="unsharded LatentMoE"):
        psm.set_prefill_moe_mode("streamed")
    psm.set_prefill_moe_mode("streamed_sp8")
    assert psm._prefill_moe_mode == "streamed_sp8"


def test_unsharded_single_node_keeps_streamed_default():
    psm = _psm_stub(_distributed_weight_sharded=False, _latent_tp_planned=False)
    assert psm.default_prefill_moe_mode() == "streamed"
    psm.set_prefill_moe_mode("streamed")
    assert psm._prefill_moe_mode == "streamed"


def test_distributed_weights_default_to_streamed_sp8():
    for latent in (False, True):
        psm = _psm_stub(_latent_tp_planned=latent)
        assert psm.default_prefill_moe_mode() == "streamed_sp8"


def _latent_plan_stub(*, is_k3, want, group_size):
    namespace = {}
    exec(
        _class_method_source(
            PSM, "KimiLinearParallelStrategyManager", "_plans_latent_tp_sharding"
        ),
        namespace,
    )
    obj = type("_PSM", (), {})()
    obj._is_k3 = is_k3
    obj._attn_tp_size = group_size
    obj.engine_config = type("_Config", (), {})()
    obj.engine_config.Module_Batching_Config = type("_Batching", (), {})()
    obj.engine_config.Module_Batching_Config.k3_shard_latent_projections = want
    return namespace["_plans_latent_tp_sharding"](obj)


def test_latent_sharding_plan_is_fixed_by_config_before_the_build():
    # The server prefill mode is chosen from this before any model exists, so
    # it must depend on the planner flag and topology only.
    assert _latent_plan_stub(is_k3=True, want=True, group_size=8)
    assert not _latent_plan_stub(is_k3=True, want=False, group_size=8)
    assert not _latent_plan_stub(is_k3=True, want=True, group_size=1)
    assert not _latent_plan_stub(is_k3=False, want=True, group_size=8)


def test_no_resident_prefill_handoff_remains_in_the_manager():
    source = PSM.read_text()
    assert "def resident_ep_prefill_available" not in source


def test_phase_instance_reuse_no_longer_declines_on_the_resident_cap():
    # The decline existed only for the removed post-decode resident handoff;
    # resident-EP decode now reuses its instances on every memory tier.
    source = _class_method_source(
        PSM, "KimiLinearParallelStrategyManager", "_instances_are_reused"
    )
    assert "resident_ep_prefill_token_cap" not in source
