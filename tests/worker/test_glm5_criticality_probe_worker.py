"""Worker and adapter wiring of the GLM-5 decode criticality probe (CPU).

The worker stays model-neutral: it reaches the probe only through the model's
graph adapter (``ModelCudaGraphAdapter.criticality_probe_request`` /
``set_criticality_probe_request`` / ``install_criticality_probe``), and the
GLM-5 adapter owns the ``glm5_criticality_probe`` flag. Worker methods are
exec'd from ``batchgen_worker.py`` in isolation (no CUDA, core engine or
model), as tests/test_phase_instance_reuse.py does; the decode loop's
placement rules are source contracts over the same file.

Contract:

* decode configure keeps the resident decode graph when the requested probe
  equals the installed one (None included), and otherwise arms exactly one
  recapture, which the configured-capture refusal and the reuse lifecycle's
  capture-once guard both admit;
* the request is part of both capture signatures;
* the GLM-5 adapter parses the flag (cached for an unchanged request), refuses
  a request without the decode graph, and installs the probe into the
  whole-model segment for this round's request only; the base adapter has no
  probe;
* in the decode loop the delay-table update precedes both GLM-5 replays, the
  ring pair carries the probe tag, and every harvest feeds the probe.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from batchgen.cuda_graph.adapter import ModelCudaGraphAdapter
from batchgen.models.glm.glm5.criticality_probe import (
    DEBUG_KEY,
    CriticalityProbe,
    CriticalityProbeSpec,
)
from batchgen.models.glm.glm5.cuda_graph_adapter import Glm5CudaGraphAdapter

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "batchgen" / "batchgen_worker.py"
WORKER_TREE = ast.parse(WORKER.read_text())

_RAW = {
    "sites": ["after_router"],
    "schedule": [{"steps": 8, "delay_us": {}}, {"steps": 8, "delay_us": {"after_router": 10}}],
}


def _worker_class():
    return next(
        node for node in WORKER_TREE.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )


def _method_ast(name):
    return next(
        node for node in _worker_class().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _isolated(names, **globals_override):
    bodies = [copy.deepcopy(_method_ast(name)) for name in names]
    module = ast.Module(
        body=[ast.ClassDef(name="Worker", bases=[], keywords=[], body=bodies,
                           decorator_list=[])],
        type_ignores=[],
    )
    namespace = dict(globals_override)
    exec(compile(ast.fix_missing_locations(module), str(WORKER), "exec"), namespace)
    return namespace["Worker"]


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, *args, **kwargs):
        self.lines.append(args)

    warning = info


def _adapter():
    return Glm5CudaGraphAdapter(model_config=None, engine_config=None, world_size=2, rank=0)


def _probe(raw=_RAW):
    return CriticalityProbe(
        CriticalityProbeSpec.parse(raw), torch.device("cpu"), launch=lambda *a: None
    )


def _worker(*, adapter="glm5", installed=None, enable_cuda_graph=True):
    log = _Log()
    cls = _isolated(["_configure_criticality_probe"], Optional=Optional, logging=log)
    worker = cls()
    worker.rank = 0
    worker.args = SimpleNamespace(enable_cuda_graph=enable_cuda_graph, disable_cuda_graphs=False)
    worker._cuda_graph_adapter = _adapter() if adapter == "glm5" else adapter
    worker._criticality_probe = installed
    worker._criticality_probe_request = None
    worker._criticality_probe_recapture = False
    return worker, log


# --------------------------------------------------------------------------- #
# Decode configure decision
# --------------------------------------------------------------------------- #

def test_inactive_round_keeps_the_graph():
    worker, log = _worker()
    for debug in (None, {}, {"glm5_dispatch_trace": True}):
        worker._configure_criticality_probe(debug)
        assert worker._criticality_probe_request is None
        assert worker._criticality_probe_recapture is False
        assert worker._cuda_graph_adapter._round_criticality_probe_request is None
    assert log.lines == []


def test_first_probe_round_arms_one_recapture():
    worker, log = _worker()
    worker._configure_criticality_probe({DEBUG_KEY: _RAW})
    spec = CriticalityProbeSpec.parse(_RAW)
    assert worker._criticality_probe_request == spec.signature()
    assert worker._criticality_probe_recapture is True
    assert worker._cuda_graph_adapter._round_criticality_probe_request == spec.signature()
    assert len(log.lines) == 1


def test_same_request_keeps_the_instrumented_graph():
    worker, _ = _worker(installed=_probe())
    worker._configure_criticality_probe({DEBUG_KEY: dict(_RAW)})
    assert worker._criticality_probe_recapture is False


@pytest.mark.parametrize(
    "debug",
    [None, {DEBUG_KEY: dict(_RAW, skip_steps_after_change=2)}],
    ids=["plain-round-after-probe", "different-request"],
)
def test_a_changed_request_arms_a_recapture(debug):
    worker, _ = _worker(installed=_probe())
    worker._configure_criticality_probe(debug)
    assert worker._criticality_probe_recapture is True


def test_without_an_adapter_there_is_no_request():
    worker, _ = _worker(adapter=None)
    worker._configure_criticality_probe({DEBUG_KEY: _RAW})
    assert worker._criticality_probe_request is None
    assert worker._criticality_probe_recapture is False


def test_request_without_the_decode_graph_fails_loud():
    worker, _ = _worker(enable_cuda_graph=False)
    with pytest.raises(ValueError, match="--enable-cuda-graph"):
        worker._configure_criticality_probe({DEBUG_KEY: _RAW})
    worker, _ = _worker(enable_cuda_graph=False)
    worker._configure_criticality_probe({})


def test_malformed_request_fails_loud():
    worker, _ = _worker()
    with pytest.raises(ValueError, match="unknown site"):
        worker._configure_criticality_probe({DEBUG_KEY: dict(_RAW, sites=["x"])})


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #

class _Kv:
    def get_cuda_graph_page_table_storage(self):
        return SimpleNamespace(data_ptr=lambda: 1, shape=(2, 3), dtype="i32", device="cuda:0")


def test_worker_capture_signature_carries_the_request():
    cls = _isolated(["_glm5_whole_model_graph_capture_signature"], Optional=Optional)
    worker = cls()
    gpu = SimpleNamespace(primary=_Kv(), auxiliary=_Kv())
    worker._get_cuda_graph_gpu_manager = lambda: gpu
    worker.core_engine = SimpleNamespace(gpu_paged_kv_manager_aux=None)
    plain = worker._glm5_whole_model_graph_capture_signature()
    assert len(plain) == 3 and plain[-1] is None
    worker._criticality_probe_request = ("probe",)
    probed = worker._glm5_whole_model_graph_capture_signature()
    assert probed[:-1] == plain[:-1] and probed[-1] == ("probe",)


def test_adapter_capture_signature_carries_the_round_request():
    adapter = _adapter()
    gpu = SimpleNamespace(primary=_Kv(), auxiliary=_Kv())
    plain = adapter.capture_signature(bucket=4, gpu_kv_manager=gpu, max_seqlen=64)
    assert plain[-1] is None
    request = adapter.criticality_probe_request({DEBUG_KEY: _RAW}, graph_enabled=True)
    # Parsing alone (the per-step mid-round check) does not change the round.
    assert adapter.capture_signature(bucket=4, gpu_kv_manager=gpu, max_seqlen=64) == plain
    adapter.set_criticality_probe_request(request)
    probed = adapter.capture_signature(bucket=4, gpu_kv_manager=gpu, max_seqlen=64)
    assert probed[:-1] == plain[:-1] and probed[-1] == request


def test_adapter_parse_is_cached_for_an_unchanged_request(monkeypatch):
    adapter = _adapter()
    parses = []
    real = CriticalityProbeSpec.parse.__func__

    def counting(cls, raw):
        parses.append(raw)
        return real(cls, raw)

    monkeypatch.setattr(CriticalityProbeSpec, "parse", classmethod(counting))
    first = adapter.criticality_probe_request({DEBUG_KEY: _RAW}, graph_enabled=True)
    for _ in range(5):
        assert adapter.criticality_probe_request(
            {DEBUG_KEY: dict(_RAW)}, graph_enabled=True
        ) == first
    assert len(parses) == 1
    assert adapter.criticality_probe_request({"other": 1}, graph_enabled=True) is None


def test_adapter_installs_only_this_rounds_request():
    adapter = _adapter()
    installed = []
    segment = SimpleNamespace(set_criticality_probe=installed.append)
    request = adapter.criticality_probe_request({DEBUG_KEY: _RAW}, graph_enabled=True)
    with pytest.raises(RuntimeError, match="not this round's"):
        adapter.install_criticality_probe(
            request, whole_model_segment=segment, device=torch.device("cpu"))
    adapter.set_criticality_probe_request(request)
    # A mid-round parse of another request must not change what is installed.
    other = adapter.criticality_probe_request(
        {DEBUG_KEY: dict(_RAW, skip_steps_after_change=2)}, graph_enabled=True)
    assert other != request
    monkey_launch = []
    import batchgen_kernels.common.spin_delay as spin_module

    original = spin_module.preload_spin_delay_kernel
    spin_module.preload_spin_delay_kernel = lambda: monkey_launch.append("preload")
    try:
        probe = adapter.install_criticality_probe(
            request, whole_model_segment=segment, device=torch.device("cpu"))
    finally:
        spin_module.preload_spin_delay_kernel = original
    assert monkey_launch == ["preload"]
    assert installed == [probe]
    assert probe.signature() == request
    with pytest.raises(RuntimeError, match="not produced by this adapter"):
        adapter.set_criticality_probe_request(("forged",))


class _NoProbeAdapter(ModelCudaGraphAdapter):
    def is_supported(self, engine_config):
        return True

    def select_buckets(self, engine_config):
        return [1]

    def advertised_modes(self):
        return []

    def build_segments(self, **kwargs):
        raise NotImplementedError

    def capture_signature(self, **kwargs):
        return None

    def capture_inputs_for(self, **kwargs):
        return {}

    def eligibility(self, batch_state):
        raise NotImplementedError

    def prepare_replay_inputs(self, **kwargs):
        return {}

    def stage_post_graph_kv(self, **kwargs):
        return None

    def run_eager_reference(self, **kwargs):
        return {}


def test_base_adapter_has_no_probe():
    adapter = _NoProbeAdapter()
    assert adapter.criticality_probe_request({DEBUG_KEY: _RAW}, graph_enabled=True) is None
    adapter.set_criticality_probe_request(None)
    with pytest.raises(RuntimeError, match="implements no criticality probe"):
        adapter.set_criticality_probe_request(("probe",))
    with pytest.raises(RuntimeError, match="implements no criticality probe"):
        adapter.install_criticality_probe(
            ("probe",), whole_model_segment=None, device=torch.device("cpu"))


# --------------------------------------------------------------------------- #
# Source contracts
# --------------------------------------------------------------------------- #

def _calls(method, attr):
    return sorted(
        node.lineno for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == attr
    )


def test_configure_runs_before_the_graph_warmup_decision_and_can_trigger_it():
    generate = _method_ast("generate")
    configure = _calls(generate, "_configure_criticality_probe")
    warmup = _calls(generate, "_warmup_cuda_graphs")
    config = _calls(generate, "_bind_decode_attention_metadata_for_graph_config")
    assert len(configure) == 1 and warmup and config
    assert config[0] < configure[0] < warmup[0]
    trigger = next(
        node for node in ast.walk(generate)
        if isinstance(node, ast.If)
        and any(n.lineno == warmup[0] for n in ast.walk(node) if isinstance(n, ast.Call))
        and "generic_cuda_graph_warmup_needed" in ast.unparse(node.test)
    )
    assert "self._criticality_probe_recapture" in ast.unparse(trigger.test)


def test_capture_path_admits_only_the_armed_recapture():
    setup = _method_ast("_setup_cuda_graphs")
    guards = [
        node for node in ast.walk(setup)
        if isinstance(node, ast.If)
        and (
            "_decode_instance_is_reusable" in ast.unparse(node.test)
            or ast.unparse(node.test).startswith("capture_attempted")
        )
        and any(isinstance(n, (ast.Raise, ast.Return)) for n in node.body)
    ]
    assert len(guards) == 2
    for guard in guards:
        assert "_criticality_probe_recapture" in ast.unparse(guard.test)
    src = ast.unparse(setup)
    assert src.index("install_criticality_probe(") < src.index(
        "manager.warmup_and_capture_buckets"
    )
    assert src.index("self._decode_graph_captures = ") < src.rindex(
        "self._criticality_probe_recapture = False"
    )


def test_release_drops_the_installed_probe():
    for name in ("_release_glm5_whole_model_graph_state", "deep_free_model_memory"):
        assert "self._criticality_probe = None" in ast.unparse(_method_ast(name)), name


def test_decode_loop_updates_the_table_before_each_glm5_replay_and_tags_the_pair():
    loop = _method_ast("decoding_continuous")
    replays = [
        node for node in ast.walk(loop)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "self._cuda_graph_manager.replay"
    ]
    glm5_replays = [
        node for node in replays
        if ast.unparse(node.args[0]) in ("_wm_seg_name", "'glm5_whole_model'")
    ]
    assert len(glm5_replays) == 2
    before = _calls(loop, "before_replay")
    assert len(before) == 2
    for replay, update in zip(sorted(n.lineno for n in glm5_replays), before):
        assert update < replay
        assert not [r.lineno for r in replays if update < r.lineno < replay]
    ends = [
        node for node in ast.walk(loop)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "_fwd_ring.end"
    ]
    assert len(ends) == 1
    assert [ast.unparse(kw.value) for kw in ends[0].keywords if kw.arg == "tag"] == ["_probe_tag"]
    harvests = [
        node for node in ast.walk(loop)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "_fwd_ring.harvest"
    ]
    records = [
        node for node in ast.walk(loop)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "_probe.record"
    ]
    assert len(harvests) == 3 and len(records) == 3


def test_worker_never_names_the_model_or_imports_the_probe_module():
    src = WORKER.read_text()
    assert "criticality_probe import" not in src
    assert "glm5_criticality_probe" not in src
    for name in ("_configure_criticality_probe",):
        body = ast.unparse(_method_ast(name)).lower()
        assert "glm" not in body


def test_spin_kernel_is_part_of_every_dsa_preflight_contract():
    import batchgen.runtime_preflight as preflight

    ext = "batchgen_kernels.common._C_spin_delay"
    assert ext in preflight._GLM5_DSA_EXTENSIONS
    for model_type in ("glm_moe_dsa", "glm_moe_dsa_5_2", "glm_moe_dsa_5_3"):
        assert ext in preflight._CONTRACTS[model_type].extensions
