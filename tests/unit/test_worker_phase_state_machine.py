"""AST-isolated CPU checks for the worker phase transition table."""

import ast
import copy
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

from batchgen.phase_lifecycle import ExitMode, PhaseEnterContext, PhaseKey
from batchgen.phase_lifecycle import verify_phase_contract


ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "batchgen" / "batchgen_worker.py"


def _isolated_worker_class():
    tree = ast.parse(WORKER.read_text())
    source_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )
    names = {
        "_desired_phase_key",
        "_server_prefill_moe_mode",
        "_prefill_moe_mode_for_debug",
        "_prefill_token_cap_for_mode",
        "_live_prefill_token_cap",
        "_planned_prefill_token_cap",
        "_phase_exit_mode",
        "_prepare_phase_exit",
        "_transition_phase",
        "_verify_manager_phase_contract",
    }
    methods = [
        copy.deepcopy(node)
        for node in source_class.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in methods} == names
    module = ast.Module(
        body=[
            ast.ClassDef(
                name="Worker",
                bases=[],
                keywords=[],
                body=methods,
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    namespace = {
        "AttnWrapperBase": SimpleNamespace(batchgen_debug=None),
        "Optional": Optional,
        "logging": logging,
        "ExitMode": ExitMode,
        "PhaseEnterContext": PhaseEnterContext,
        "PhaseKey": PhaseKey,
        "verify_phase_contract": verify_phase_contract,
    }
    exec(compile(ast.fix_missing_locations(module), str(WORKER), "exec"), namespace)
    return namespace["Worker"]


class _Manager:
    def __init__(self, events, held):
        self.events = events
        self.held = held

    def holds_instance(self, phase):
        return self.held.get(phase, False)

    def phase_exit(self, key, mode):
        self.events.append(("exit", key.phase, mode))

    def phase_enter(self, key, ctx):
        self.events.append(("enter", key.phase, ctx.comm, ctx.max_num_seq))
        return object(), {}


def _worker(manager, events):
    cls = _isolated_worker_class()
    worker = cls()
    worker._live_phase_key = None
    worker.parallel_manager = manager
    worker.args = SimpleNamespace(persistent_phase_instances=False)
    worker._streamed_sp8_h2d_installed = False
    worker._weight_passes = SimpleNamespace(
        close=lambda: events.append("weight_close")
    )
    worker.core_engine = SimpleNamespace(
        stop_h2d_worker=lambda: events.append("stop_h2d"),
        clear_kv_copy_queue=lambda: events.append("clear_kv"),
        reset_decoding_buffer=lambda: events.append("reset_decode"),
    )
    worker._destroy_gpu_paged_kv_cache = lambda **kwargs: events.append(
        ("destroy_kv", kwargs)
    )
    worker._persistent_phase_enabled = lambda: worker.args.persistent_phase_instances
    worker._release_suspended_decode_kv_pool = lambda: events.append(
        "release_kv_pool"
    )
    worker.deep_free_model_memory = lambda: events.append("deep_free")
    worker.init_nvshmem = lambda: events.append("init_nvshmem")
    worker.model = None
    worker.weight_copy_task = {}
    return worker


STREAMED_CAP = 524_288
RESIDENT_CAP = 16_384


class _ServerManager:
    """Records mode selections and phase entries; holds every instance."""

    def __init__(self, events, default_mode="streamed_sp8"):
        self.events = events
        self.default_mode = default_mode
        self.default_calls = 0
        self._prefill_moe_mode = default_mode
        self.set_modes = []

    def default_prefill_moe_mode(self):
        self.default_calls += 1
        return self.default_mode

    def set_prefill_moe_mode(self, mode):
        self.set_modes.append(mode)
        self._prefill_moe_mode = mode

    def holds_instance(self, phase):
        return True

    def phase_exit(self, key, mode):
        self.events.append(("exit", key.phase, key.prefill_moe_mode, mode))

    def phase_enter(self, key, ctx):
        self.events.append(("enter", key.phase, key.prefill_moe_mode))
        return object(), {}


def _server_worker(*, resident_cap=RESIDENT_CAP, default_mode="streamed_sp8"):
    events = []
    worker = _worker(_ServerManager(events, default_mode), events)
    worker.rank = 0
    worker._resolved_server_prefill_moe_mode = None
    worker._phase_request_uuids = ()
    worker.global_batch = SimpleNamespace(get_sequence=lambda uuid: None)
    worker._wave_debug = None
    worker._active_batchgen_debug_for_sequences = (
        lambda sequences: worker._wave_debug
    )
    worker._debug_flag_enabled = lambda value: bool(value)
    worker._weight_pass_end_delay = lambda value: 0.0
    worker._weight_passes = SimpleNamespace(
        end_delay_s=None, close=lambda: events.append("weight_close")
    )
    worker.engine_config = SimpleNamespace(
        Module_Batching_Config=SimpleNamespace(
            prefill_micro_batch_token_cap=STREAMED_CAP,
            k3_resident_prefill_token_cap=resident_cap,
        )
    )
    return worker, events


def _run_prefill_wave(worker, debug=None):
    """Plan, transition and read caps the way one admission wave does."""
    worker._wave_debug = debug
    planned_cap = worker._planned_prefill_token_cap([])
    key = worker._desired_phase_key("prefill")
    worker._transition_phase(key)
    return key, planned_cap, worker._live_prefill_token_cap()


def _run_decode(worker):
    worker._transition_phase(
        worker._desired_phase_key("decode"), comm="comm", max_num_seq=8
    )


def test_same_key_runs_no_transition_work():
    events = []
    key = PhaseKey("prefill", "streamed")
    manager = _Manager(events, {"prefill": True})
    worker = _worker(manager, events)
    worker._live_phase_key = key

    assert worker._transition_phase(key) == (False, False)
    assert events == []


def test_key_change_exits_before_entering():
    events = []
    manager = _Manager(events, {"prefill": True, "decode": True})
    worker = _worker(manager, events)
    worker._live_phase_key = PhaseKey("prefill", "streamed")

    changed, resumed = worker._transition_phase(
        PhaseKey("decode", "streamed"), comm="comm", max_num_seq=16
    )

    assert changed is True
    assert resumed is True
    assert events.index(("exit", "prefill", ExitMode.SUSPEND)) < events.index(
        ("enter", "decode", "comm", 16)
    )


def test_unheld_outgoing_instance_uses_retire_mechanics():
    events = []
    manager = _Manager(events, {"decode": False})
    worker = _worker(manager, events)
    worker._live_phase_key = PhaseKey("decode", "streamed")

    worker._transition_phase(PhaseKey("prefill", "streamed"))

    exit_index = events.index(("exit", "decode", ExitMode.RETIRE))
    assert events.index(("destroy_kv", {})) < exit_index
    assert exit_index < events.index("deep_free") < events.index(
        ("enter", "prefill", None, None)
    )
    assert "release_kv_pool" not in events


def test_suspended_decode_releases_pool_pages_before_manager_exit():
    events = []
    manager = _Manager(events, {"decode": True})
    worker = _worker(manager, events)
    worker._live_phase_key = PhaseKey("decode", "streamed")

    worker._transition_phase(PhaseKey("prefill", "streamed"))

    exit_index = events.index(("exit", "decode", ExitMode.SUSPEND))
    assert events.index("release_kv_pool") < exit_index
    assert "deep_free" not in events
    assert not [e for e in events if isinstance(e, tuple) and e[0] == "destroy_kv"]


def test_persistent_suspended_decode_keeps_kv_buffers():
    events = []
    manager = _Manager(events, {"decode": True})
    worker = _worker(manager, events)
    worker.args.persistent_phase_instances = True
    worker._live_phase_key = PhaseKey("decode", "streamed")

    worker._transition_phase(PhaseKey("prefill", "streamed"))

    exit_index = events.index(("exit", "decode", ExitMode.SUSPEND))
    assert events.index(("destroy_kv", {"keep_buffers": True})) < exit_index
    assert "release_kv_pool" not in events
    assert "deep_free" not in events


def test_prepare_phase_exit_orders_cleanup_before_manager_exit_and_resets_only_for_unstreamed_prefill_to_decode():
    events = []
    manager = _Manager(events, {"prefill": True, "decode": True})
    worker = _worker(manager, events)
    worker._live_phase_key = PhaseKey("prefill", "streamed")

    worker._transition_phase(PhaseKey("decode", "streamed"))

    exit_index = events.index(("exit", "prefill", ExitMode.SUSPEND))
    for event in ("weight_close", "stop_h2d", "clear_kv", "reset_decode"):
        assert events.index(event) < exit_index

    events = []
    manager = _Manager(events, {"prefill": True, "decode": True})
    worker = _worker(manager, events)
    worker._streamed_sp8_h2d_installed = True
    worker._live_phase_key = PhaseKey("prefill", "streamed")
    worker._transition_phase(PhaseKey("decode", "streamed"))
    assert "reset_decode" not in events

    events = []
    manager = _Manager(events, {"prefill": True, "decode": True})
    worker = _worker(manager, events)
    worker._live_phase_key = PhaseKey("decode", "streamed")
    worker._transition_phase(PhaseKey("prefill", "streamed"))
    assert "reset_decode" not in events


def test_every_prefill_wave_uses_the_server_mode_across_decode_phases():
    worker, events = _server_worker()
    # Startup resolves the server mode once and enters prefill in it.
    startup_mode = worker._server_prefill_moe_mode()
    worker._transition_phase(PhaseKey("prefill", startup_mode, ()))

    keys = []
    for wave in range(3):
        key, planned_cap, live_cap = _run_prefill_wave(worker)
        keys.append(key)
        assert planned_cap == live_cap == STREAMED_CAP
        if wave < 2:
            _run_decode(worker)

    assert {key.prefill_moe_mode for key in keys} == {"streamed_sp8"}
    assert worker.parallel_manager.set_modes == ["streamed_sp8"] * 3
    # Decided once for the process, never re-read per wave.
    assert worker.parallel_manager.default_calls == 1
    prefill_entries = [e for e in events if e[:2] == ("enter", "prefill")]
    assert {e[2] for e in prefill_entries} == {"streamed_sp8"}
    assert worker.engine_config.Module_Batching_Config.prefill_micro_batch_token_cap == (
        STREAMED_CAP
    )


def test_batch_override_applies_to_its_own_wave_only():
    worker, _ = _server_worker()
    worker._transition_phase(
        PhaseKey("prefill", worker._server_prefill_moe_mode(), ())
    )
    _run_decode(worker)

    key, planned_cap, live_cap = _run_prefill_wave(
        worker, {"k3_prefill_moe_mode": "resident_ep"}
    )
    assert key.prefill_moe_mode == "resident_ep"
    assert planned_cap == live_cap == RESIDENT_CAP
    _run_decode(worker)

    key, planned_cap, live_cap = _run_prefill_wave(worker)
    assert key.prefill_moe_mode == "streamed_sp8"
    assert planned_cap == live_cap == STREAMED_CAP
    assert worker.parallel_manager.set_modes == ["resident_ep", "streamed_sp8"]
    # The resident wave never rewrites the planner's shared streamed cap.
    batching = worker.engine_config.Module_Batching_Config
    assert batching.prefill_micro_batch_token_cap == STREAMED_CAP
    assert batching.k3_resident_prefill_token_cap == RESIDENT_CAP


def test_resident_override_without_a_planned_resident_cap_uses_the_streamed_cap():
    worker, _ = _server_worker(resident_cap=0)
    assert worker._prefill_token_cap_for_mode("resident_ep") == STREAMED_CAP
    for mode in ("streamed", "streamed_sp8", None):
        assert worker._prefill_token_cap_for_mode(mode) == STREAMED_CAP


def test_manager_without_a_choice_runs_streamed():
    worker, _ = _server_worker()
    worker.parallel_manager = SimpleNamespace()
    assert worker._server_prefill_moe_mode() == "streamed"
    assert worker._prefill_moe_mode_for_debug(None) == "streamed"
    assert worker._prefill_moe_mode_for_debug(
        {"k3_prefill_moe_mode": "streamed_sp8"}
    ) == "streamed_sp8"


def test_live_cap_refuses_a_read_outside_a_live_prefill_phase():
    worker, _ = _server_worker()
    with pytest.raises(RuntimeError, match="live prefill wave"):
        worker._live_prefill_token_cap()
    worker._live_phase_key = PhaseKey("decode", "streamed_sp8")
    with pytest.raises(RuntimeError, match="live prefill wave"):
        worker._live_prefill_token_cap()


def _worker_function_nodes():
    tree = ast.parse(WORKER.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )
    return [node for node in cls.body if isinstance(node, ast.FunctionDef)]


def test_every_cap_reader_goes_through_the_mode_accessor():
    readers = sorted(
        node.name
        for node in _worker_function_nodes()
        for sub in ast.walk(node)
        if isinstance(sub, ast.Attribute)
        and sub.attr in ("prefill_micro_batch_token_cap", "k3_resident_prefill_token_cap")
    )
    assert readers == ["_prefill_token_cap_for_mode"]
    # String getattr reads count too: only the accessor names the fields.
    named = sorted(
        node.name
        for node in _worker_function_nodes()
        for sub in ast.walk(node)
        if isinstance(sub, ast.Constant)
        and sub.value in ("prefill_micro_batch_token_cap", "k3_resident_prefill_token_cap")
    )
    assert named == ["_prefill_token_cap_for_mode"]


def test_no_resident_prefill_handoff_remains_in_the_worker():
    source = WORKER.read_text()
    for name in (
        "resident_handoff",
        "_phase_resident_handoff",
        "_phase_resident_token_cap",
        "resident_ep_prefill_available",
    ):
        assert name not in source


def test_startup_gate_raises_with_bad_manager():
    worker = _isolated_worker_class()()
    worker.parallel_manager = object()
    with pytest.raises(RuntimeError, match="phase lifecycle contract failed"):
        worker._verify_manager_phase_contract()
