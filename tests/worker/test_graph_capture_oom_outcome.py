"""CPU-only execution of the capture OOM block and the real worker delegator.

At the main baseline the enabled whole-model path also requires the graph,
so its OOM normally re-raises. Exercise the retained non-fatal handler by
supplying whole_graph_required=False to the extracted, unmodified block.
No GLM model source or CUDA capture implementation is imported here.
"""

import ast
import inspect
import logging
import textwrap
from types import SimpleNamespace

import pytest
import torch

from batchgen.worker.graph_runtime import (
    GRAPH_SETUP_UNCHANGED,
    GraphRuntime,
    GraphSetupOutcome,
)
from tests.worker_method_source import load_worker_methods


def _capture_oom_block(scope):
    """Compile probe installation through the actual try/except, unchanged."""
    setup = ast.parse(textwrap.dedent(inspect.getsource(GraphRuntime._setup_cuda_graphs))).body[0]
    glm_path = next(
        node for node in setup.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "glm5_whole_graph_enabled"
    )
    start = next(
        i for i, node in enumerate(glm_path.body)
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == "setup_criticality_probe"
        and isinstance(node.value, ast.Constant) and node.value.value is None
    )
    end = next(
        i for i, node in enumerate(glm_path.body)
        if isinstance(node, ast.Try)
        and any(ast.unparse(h.type) == "torch.OutOfMemoryError" for h in node.handlers)
    )
    setup.body = glm_path.body[start:end + 1]
    setup.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[setup], type_ignores=[]))
    exec(compile(module, inspect.getfile(GraphRuntime), "exec"), scope)
    return scope["_setup_cuda_graphs"]


@pytest.mark.parametrize("probe_requested", [False, True])
@pytest.mark.parametrize("recapture_armed", [False, True])
@pytest.mark.parametrize("failure_site", ["capture_inputs", "capture"])
def test_nonfatal_capture_oom_matches_original_worker_terminal_state(
    monkeypatch, probe_requested, recapture_armed, failure_site,
):
    # Expected values come from origin/main's GraphRuntime._setup_cuda_graphs,
    # lines 675-676: self._criticality_probe = None;
    # self._criticality_probe_recapture = False, before the optional re-raise.
    # Neither assignment is conditional on the request or incoming arm flag.
    expected_probe, expected_recapture = None, False
    installed_probe, prior_probe = object(), object()
    request = object() if probe_requested else None
    events = []

    def install(actual_request, **kwargs):
        assert actual_request is request
        events.append("install")
        return installed_probe

    def capture_inputs(**kwargs):
        events.append("inputs")
        if failure_site == "capture_inputs":
            raise torch.OutOfMemoryError("input allocation failed")
        return {}

    def capture(buckets):
        assert buckets == [4]
        events.append("capture")
        raise torch.OutOfMemoryError("capture failed")

    manager = SimpleNamespace(
        warmup_and_capture_buckets=capture,
        drop_bucket=lambda bucket: events.append(("drop", bucket)),
    )
    graph = GraphRuntime()
    graph.cuda_graph_adapter = SimpleNamespace(install_criticality_probe=install)
    block = _capture_oom_block({
        "Any": object, "Callable": object, "GraphSetupContext": object,
        "GraphSetupOutcome": GraphSetupOutcome, "torch": torch,
        "dist": SimpleNamespace(barrier=lambda: None), "logging": logging,
        "manager": manager, "whole_seg": SimpleNamespace(set_capture_inputs=lambda **kw: None),
        "bucketing": object(), "capture_buckets": [1, 4], "num_heads": 1,
        "whole_graph_required": False,
        "setup_decode_graph_ladder_logged": GRAPH_SETUP_UNCHANGED,
        "setup_criticality_probe_recapture": GRAPH_SETUP_UNCHANGED,
    })
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    outcomes = []

    def setup(ctx, gpu_manager, **helpers):
        result = block(ctx, gpu_manager, **helpers)
        outcomes.append(result)
        return result

    graph._setup_cuda_graphs = setup
    worker = SimpleNamespace(
        _graph=graph, model_name="glm5", args=None, torch_device="cpu",
        model_config=None, model=None, core_engine=None, rank=0, world_size=1,
        glm5_disable_decode_host_kv_staging=False,
        _criticality_probe_request=request, _criticality_probe=prior_probe,
        _criticality_probe_recapture=recapture_armed,
    )

    def unexpected_helper(*args, **kwargs):
        raise AssertionError("the OOM block called an unrelated helper")

    for name, parameter in inspect.signature(GraphRuntime._setup_cuda_graphs).parameters.items():
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            setattr(worker, name, unexpected_helper)
    worker._make_glm5_whole_model_capture_inputs = capture_inputs
    delegator = load_worker_methods(["_setup_cuda_graphs"], {})["_setup_cuda_graphs"]
    assert delegator(worker, manager) is None

    assert len(outcomes) == 1
    assert outcomes[0]._criticality_probe is expected_probe
    assert outcomes[0]._criticality_probe_recapture is expected_recapture
    assert worker._criticality_probe is expected_probe
    assert worker._criticality_probe_recapture is expected_recapture
    assert ("install" in events) is probe_requested
    assert ("capture" in events) is (failure_site == "capture")
    assert events[-2:] == [("drop", 4), "empty_cache"]
    assert graph.glm5_whole_model_graph_failed_buckets == {4}
    assert graph.cuda_graph_manager is None
    assert graph.whole_model_segment is None
