"""Held phase instances across the prefill-decode switch, under two rulings.

- The prefill MoE mode is a server constant: every PhaseKey (prefill and
  decode alike) derives it from ``_server_prefill_moe_mode()``. There is no
  per-entry record of "the mode of the most recent prefill" and nothing to
  prime for a decode-first startup: whatever the entry order, both keys carry
  the same server mode, so a resumable exit never turns into RETIRE over a
  mode mismatch.
- A hold never outlives the model it references: a kept instance is resumed
  only under the key it was entered with, a RETIRE retires every kept
  instance before ``deep_free_model_memory``, and the deep free refuses to
  run under any remaining hold instead of silently preserving its graphs
  (the silent preserve is what pinned ~70 GiB across the switch and OOMed
  the K3.1 4-node H20 startup).
"""

import ast
import copy
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

from batchgen.phase_lifecycle import (
    PHASES,
    ExitMode,
    PhaseEnterContext,
    PhaseKey,
)


ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "batchgen" / "batchgen_worker.py"
KIMI_PSM = (
    ROOT
    / "batchgen"
    / "models"
    / "moonshotai"
    / "kimi_linear"
    / "Parallel_Strategy_Manager.py"
)

_WORKER_METHODS = {
    "_desired_phase_key",
    "_phase_exit_mode",
    "_transition_phase",
    "_prepare_phase_exit",
    "_server_prefill_moe_mode",
    "_require_server_prefill_moe_mode",
    "_retire_kept_phase_instances",
    "_require_resumable_instance",
}


def _class_methods(path, class_name, names):
    tree = ast.parse(path.read_text())
    source = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return [
        copy.deepcopy(node)
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]


def _exec_class(path, class_name, names):
    module = ast.Module(
        body=[
            ast.ClassDef(
                name="Extracted",
                bases=[],
                keywords=[],
                body=_class_methods(path, class_name, names),
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    namespace = {
        "ExitMode": ExitMode,
        "PhaseEnterContext": PhaseEnterContext,
        "PhaseKey": PhaseKey,
        "PHASES": PHASES,
        "logging": logging,
        "Optional": Optional,
    }
    exec(
        compile(ast.fix_missing_locations(module), str(path), "exec"),
        namespace,
    )
    return namespace["Extracted"]


def _worker_class():
    return _exec_class(WORKER, "BatchGenWorker", _WORKER_METHODS)


class _HoldingManager:
    """Keeps every entered instance until it is retired (K3/GLM-5 shape)."""

    def __init__(self, events):
        self.events = events
        self.held = {phase: False for phase in PHASES}

    def default_prefill_moe_mode(self):
        return "streamed"

    def set_prefill_moe_mode(self, mode):
        self.events.append(("set_mode", mode))

    def holds_instance(self, phase):
        return self.held[phase]

    def phase_exit(self, key, mode):
        self.events.append(("exit", key.phase, mode))
        if mode is ExitMode.RETIRE:
            self.held[key.phase] = False

    def phase_enter(self, key, ctx):
        self.events.append(("enter", key.phase))
        self.held[key.phase] = True
        return object(), {}


def _worker(events):
    worker = _worker_class()()
    worker.rank = 0
    worker._live_phase_key = None
    worker._kept_phase_keys = {}
    worker._server_prefill_moe_mode_resolved = None
    worker._phase_profile_overrides = ()
    worker.parallel_manager = _HoldingManager(events)
    worker.args = SimpleNamespace(
        persistent_phase_instances=True, prefill_moe_mode=None
    )
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


def test_decode_key_is_the_server_constant_before_any_prefill_entry():
    """Decode-first startup needs no primed per-entry record (ex-#652)."""
    events = []
    worker = _worker(events)
    assert worker._desired_phase_key("decode") == PhaseKey(
        "decode", "streamed", ()
    )


def test_held_prefill_suspends_on_the_switch_to_decode():
    events = []
    worker = _worker(events)
    worker._transition_phase(PhaseKey("prefill", "streamed"))
    worker._transition_phase(
        worker._desired_phase_key("decode"), comm="comm", max_num_seq=8
    )
    assert ("exit", "prefill", ExitMode.SUSPEND) in events
    assert "deep_free" not in events


def test_decode_first_startup_suspends_on_the_switch_to_prefill():
    events = []
    worker = _worker(events)
    worker._transition_phase(
        worker._desired_phase_key("decode"), comm="comm", max_num_seq=8
    )
    worker._transition_phase(PhaseKey("prefill", "streamed"))
    assert ("exit", "decode", ExitMode.SUSPEND) in events
    assert "deep_free" not in events


def test_a_key_with_another_mode_is_refused_by_name():
    events = []
    worker = _worker(events)
    with pytest.raises(RuntimeError, match="runs 'streamed' for its lifetime"):
        worker._transition_phase(PhaseKey("prefill", "resident_ep"))
    assert ("enter", "prefill") not in events


def test_retire_retires_every_kept_instance_before_the_deep_free():
    events = []
    worker = _worker(events)
    worker._transition_phase(PhaseKey("prefill", "streamed"))
    worker._transition_phase(
        worker._desired_phase_key("decode"), comm="comm", max_num_seq=8
    )
    # A profile-override change forces RETIRE of the live decode instance;
    # the kept prefill instance must go with it, before the deep free.
    worker._phase_profile_overrides = (("k3_prefill_profile", True),)
    worker._transition_phase(
        PhaseKey("prefill", "streamed", (("k3_prefill_profile", True),))
    )
    retire_decode = events.index(("exit", "decode", ExitMode.RETIRE))
    retire_prefill = events.index(("exit", "prefill", ExitMode.RETIRE))
    deep_free = events.index("deep_free")
    assert retire_decode < deep_free
    assert retire_prefill < deep_free
    # The decode hold did not survive the retirement; only the re-entered
    # live prefill instance is held, under its own (new) key.
    assert worker.parallel_manager.held["decode"] is False
    assert set(worker._kept_phase_keys) == {"prefill"}
    assert worker._kept_phase_keys["prefill"] == PhaseKey(
        "prefill", "streamed", (("k3_prefill_profile", True),)
    )


def _function(source, name):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(name)


def test_deep_free_refuses_a_held_instance_and_releases_graph_state():
    source = WORKER.read_text()
    deep_free = ast.unparse(_function(source, "deep_free_model_memory"))
    # The ex-#649 silent preserve is gone: no hold survives the deep free.
    assert "if not self.parallel_manager.holds_instance" not in deep_free
    assert "a hold never outlives the model" in deep_free
    assert "self._release_decode_graph_state()" in deep_free
    release = ast.unparse(_function(source, "_release_decode_graph_state"))
    assert "self._cuda_graph_manager = None" in release
    assert "self._cuda_graph_adapter.release_context()" in release
    assert "self._decode_graph_captures = 0" in release


def test_no_per_entry_prefill_mode_record_remains():
    """One path: the server constant, never a per-entry record (ex-#649)."""
    assert "_prefill_moe_mode_in_effect" not in WORKER.read_text()


def test_persistent_builder_uses_the_server_mode_and_one_key_path():
    builder = ast.unparse(
        _function(WORKER.read_text(), "_build_persistent_phase_instances")
    )
    assert "prefill_mode = self._server_prefill_moe_mode()" in builder
    set_mode = builder.index("pm.set_prefill_moe_mode(prefill_mode)")
    assert set_mode < builder.index("self._transition_phase(prefill_key)")
    assert "decode_key = self._desired_phase_key('decode')" in builder


def test_k3_startup_reads_the_server_mode_without_priming():
    builder = ast.unparse(
        _function(WORKER.read_text(), "prepare_kimi_k3_startup")
    )
    assert "startup_prefill_mode = self._server_prefill_moe_mode()" in builder
    assert "_prefill_moe_mode_in_effect" not in builder


# --------------------------------------------------------------------- #
#  Row-#36 regression: held instances + a new model skeleton never coexist
# --------------------------------------------------------------------- #


def _kimi_guard_class():
    return _exec_class(
        KIMI_PSM,
        "KimiLinearParallelStrategyManager",
        {"_require_no_held_instances_for_build"},
    )


def test_build_model_refuses_held_instances_of_the_previous_model():
    """The K3.1 4-node H20 startup OOM shape: a kept instance pins the
    previous model's device memory, so building the next skeleton over it
    must fail by name instead of OOMing minutes later."""
    psm = _kimi_guard_class()()
    psm.rank = 0
    psm.prefill_instance = (object(), {})
    psm.decode_instance = (object(), {})
    with pytest.raises(RuntimeError, match="prefill and decode"):
        psm._require_no_held_instances_for_build()
    psm.prefill_instance = None
    with pytest.raises(RuntimeError, match="decode instance"):
        psm._require_no_held_instances_for_build()
    psm.decode_instance = None
    psm._require_no_held_instances_for_build()


def test_build_model_runs_the_guard_before_the_skeleton():
    build = ast.unparse(_function(KIMI_PSM.read_text(), "_build_model"))
    guard = build.index("self._require_no_held_instances_for_build()")
    assert guard < build.index("self._load_model_skeleton()")
