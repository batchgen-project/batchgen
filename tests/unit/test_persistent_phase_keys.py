"""Persistent and reused phase instances survive the prefill->decode switch.

Two defects made a held decode instance re-enter graph capture:

- The decode key read the prefill MoE mode from a manager attribute that the
  GLM-5 manager commits only on a rebuild, so after the persistent startup
  builder (or any resumed prefill) it was None while the live prefill key said
  "streamed". The key mismatch turned a resumable prefill exit into RETIRE.
- RETIRE's deep_free_model_memory dropped the decode graph state although the
  held decode instance owns it and replays it on its next activation.
"""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

from batchgen.phase_lifecycle import ExitMode, PhaseEnterContext, PhaseKey
from tests.worker_method_source import worker_shaped_method_node


ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "batchgen" / "batchgen_worker.py"


def _worker_class():
    tree = ast.parse(WORKER.read_text())
    source = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )
    names = {"_desired_phase_key", "_phase_exit_mode", "_transition_phase",
             "_prepare_phase_exit"}
    methods = [
        copy.deepcopy(node)
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(
        body=[ast.ClassDef(name="Worker", bases=[], keywords=[], body=methods,
                           decorator_list=[])],
        type_ignores=[],
    )
    namespace = {"ExitMode": ExitMode, "PhaseEnterContext": PhaseEnterContext,
                 "PhaseKey": PhaseKey}
    exec(compile(ast.fix_missing_locations(module), str(WORKER), "exec"),
         namespace)
    return namespace["Worker"]


class _GlmLikeManager:
    """Holds both instances; commits no prefill mode attribute (GLM-5)."""

    def __init__(self, events):
        self.events = events
        self._prefill_moe_mode = None

    def holds_instance(self, phase):
        return True

    def phase_exit(self, key, mode):
        self.events.append(("exit", key.phase, mode))

    def phase_enter(self, key, ctx):
        self.events.append(("enter", key.phase))
        return object(), {}


def _worker(events):
    worker = _worker_class()()
    worker._live_phase_key = None
    worker._prefill_moe_mode_in_effect = None
    worker._phase_profile_overrides = ()
    worker.parallel_manager = _GlmLikeManager(events)
    worker.args = SimpleNamespace(persistent_phase_instances=True)
    worker._streamed_sp8_h2d_installed = False
    worker._weight_passes = SimpleNamespace(close=lambda: None)
    worker.core_engine = SimpleNamespace(
        stop_h2d_worker=lambda: None,
        clear_kv_copy_queue=lambda: None,
        reset_decoding_buffer=lambda: None,
    )
    worker._destroy_gpu_paged_kv_cache = lambda **kw: events.append("destroy_kv")
    worker._persistent_phase_enabled = lambda: True
    worker._release_suspended_decode_kv_pool = lambda: None
    worker.deep_free_model_memory = lambda: events.append("deep_free")
    worker.init_nvshmem = lambda: None
    worker.comm = None
    return worker


def test_decode_key_carries_the_mode_of_the_prefill_entry():
    events = []
    worker = _worker(events)
    worker._transition_phase(PhaseKey("prefill", "streamed"))
    decode_key = worker._desired_phase_key("decode")
    assert decode_key == PhaseKey("decode", "streamed", ())


def test_held_prefill_suspends_on_the_switch_to_decode():
    events = []
    worker = _worker(events)
    worker._transition_phase(PhaseKey("prefill", "streamed"))
    worker._transition_phase(
        worker._desired_phase_key("decode"), comm="comm", max_num_seq=8
    )
    assert ("exit", "prefill", ExitMode.SUSPEND) in events
    assert "deep_free" not in events


def _function(source, name):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(name)


def test_deep_free_keeps_graph_state_of_a_held_decode_instance():
    source = WORKER.read_text()
    # S9: deep free and the release run in WorkerLifecycleEngine (the worker
    # delegates); read the bodies that run, in the worker's shape.
    deep_free = ast.unparse(worker_shaped_method_node("deep_free_model_memory"))
    assert "self._cuda_graph_manager = None" not in deep_free
    guard = "if not self.parallel_manager.holds_instance('decode'):"
    assert guard in deep_free
    assert deep_free.index(guard) < deep_free.index(
        "self._release_decode_graph_state()"
    )
    # S2 made GraphRuntime (self._graph) the owner of decode-graph state;
    # the pinned spellings follow the owned fields.
    release = ast.unparse(worker_shaped_method_node("_release_decode_graph_state"))
    assert "self._graph.cuda_graph_manager = None" in release
    assert "self._graph.cuda_graph_adapter.release_context()" in release
    assert "self._graph.decode_graph_captures = 0" in release


def test_persistent_builder_tells_the_manager_its_prefill_mode():
    builder = ast.unparse(
        _function(WORKER.read_text(), "_build_persistent_phase_instances")
    )
    set_mode = builder.index("pm.set_prefill_moe_mode(prefill_mode)")
    assert set_mode < builder.index("self._transition_phase(prefill_key)")
    assert "decode_key = self._desired_phase_key('decode')" in builder


def test_k3_startup_primes_the_mode_before_the_decode_build():
    """The K3 startup enters decode before any prefill; the worker's record
    of the intended prefill mode must be primed first, or the decode key
    carries None, the exit turns into RETIRE, and the resident shard stays
    mapped under the prefill build."""
    builder = ast.unparse(
        _function(WORKER.read_text(), "prepare_kimi_k3_startup")
    )
    prime = builder.index(
        "self._prefill_moe_mode_in_effect = startup_prefill_mode"
    )
    assert prime < builder.index("self._load_decode_model(")


def test_decode_first_startup_suspends_on_the_switch_to_prefill():
    events = []
    worker = _worker(events)
    worker._prefill_moe_mode_in_effect = "streamed"
    worker.init_nvshmem = lambda: events.append("init_nvshmem")

    worker._transition_phase(
        worker._desired_phase_key("decode"), comm="comm", max_num_seq=8
    )
    worker._transition_phase(PhaseKey("prefill", "streamed"))

    assert ("exit", "decode", ExitMode.SUSPEND) in events
    assert "deep_free" not in events
