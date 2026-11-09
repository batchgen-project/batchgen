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


def _psm_stub(**attrs):
    namespace = {}
    for name in (
        "set_prefill_moe_mode",
        "resident_ep_prefill_available",
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
        "resident_ep_prefill_available",
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
    obj._decode_graph = None
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


def test_distributed_weights_accept_resident_ep_only_after_the_shard_exists():
    psm = _psm_stub()
    with pytest.raises(ValueError):
        psm.set_prefill_moe_mode("resident_ep")
    psm.set_prefill_moe_mode("streamed_sp8")
    assert psm._prefill_moe_mode == "streamed_sp8"
    psm._resident_ep_built = True
    psm.set_prefill_moe_mode("resident_ep")
    assert psm.prefill_uses_resident_ep()
    with pytest.raises(ValueError):
        psm.set_prefill_moe_mode("streamed")


def test_resident_prefill_availability_needs_built_shard_and_tp_attention():
    assert not _psm_stub().resident_ep_prefill_available()
    assert _psm_stub(_resident_ep_built=True).resident_ep_prefill_available()
    assert not _psm_stub(
        _resident_ep_built=True, _attn_tp_size=1
    ).resident_ep_prefill_available()
    assert not _psm_stub(
        _resident_ep_built=True, _stream_all_modules=True
    ).resident_ep_prefill_available()
    assert not _psm_stub(
        _resident_ep_built=True, _is_k3=False
    ).resident_ep_prefill_available()


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
