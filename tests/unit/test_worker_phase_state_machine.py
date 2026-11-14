"""AST-isolated CPU checks for the worker phase transition table."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

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


class _DesiredPhaseManager:
    def __init__(self, available):
        self.available = available
        self._prefill_moe_mode = "streamed"
        self.set_modes = []

    def default_prefill_moe_mode(self):
        return "streamed"

    def resident_ep_prefill_available(self):
        return self.available

    def set_prefill_moe_mode(self, mode):
        self.set_modes.append(mode)
        self._prefill_moe_mode = mode


def _desired_phase_worker(*, cap, available, debug):
    worker = _isolated_worker_class()()
    worker.parallel_manager = _DesiredPhaseManager(available)
    worker._phase_request_uuids = ()
    worker.global_batch = SimpleNamespace(get_sequence=lambda uuid: None)
    worker._active_batchgen_debug_for_sequences = lambda sequences: debug
    worker._debug_flag_enabled = lambda value: bool(value)
    worker._weight_pass_end_delay = lambda value: 0.0
    worker._weight_passes = SimpleNamespace(end_delay_s=None)
    worker.engine_config = SimpleNamespace(
        Module_Batching_Config=SimpleNamespace(
            k3_resident_prefill_token_cap=cap
        )
    )
    return worker


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


def test_decode_entry_resumes_a_held_decode_instance_after_prefill_retires():
    # A manager that keeps no prefill instance retires prefill on every
    # switch; its kept decode instance is still the one phase_enter resumes.
    events = []
    manager = _Manager(events, {"prefill": False, "decode": True})
    worker = _worker(manager, events)
    worker._live_phase_key = PhaseKey("prefill", "streamed")

    changed, resumed = worker._transition_phase(
        PhaseKey("decode", "streamed"), comm="comm", max_num_seq=16
    )

    assert (changed, resumed) == (True, True)
    assert ("exit", "prefill", ExitMode.RETIRE) in events
    assert events.index("deep_free") < events.index(("enter", "decode", "comm", 16))


def test_decode_entry_is_not_a_resume_when_no_decode_instance_survives_the_exit():
    events = []

    class _RetiringManager(_Manager):
        def phase_exit(self, key, mode):
            super().phase_exit(key, mode)
            if mode is ExitMode.RETIRE and key.phase == "decode":
                self.held["decode"] = False

    manager = _RetiringManager(events, {"decode": True})
    worker = _worker(manager, events)
    worker._live_phase_key = PhaseKey("decode", "streamed")

    # A key change retires the live decode instance; the entry rebuilds.
    changed, resumed = worker._transition_phase(
        PhaseKey("decode", "resident_ep"), comm="comm", max_num_seq=16
    )
    assert (changed, resumed) == (True, False)
    assert ("exit", "decode", ExitMode.RETIRE) in events

    events.clear()
    manager = _Manager(events, {"prefill": True, "decode": False})
    worker = _worker(manager, events)
    worker._live_phase_key = PhaseKey("prefill", "streamed")
    assert worker._transition_phase(
        PhaseKey("decode", "streamed"), comm="comm", max_num_seq=16
    ) == (True, False)


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


def test_desired_phase_key_requires_positive_cap_and_available_resident_ep():
    worker = _desired_phase_worker(cap=8, available=True, debug={})
    assert worker._desired_phase_key("prefill").prefill_moe_mode == "resident_ep"
    assert worker.parallel_manager.set_modes == ["resident_ep"]

    worker = _desired_phase_worker(cap=8, available=False, debug={})
    assert worker._desired_phase_key("prefill").prefill_moe_mode == "streamed"


def test_desired_phase_key_explicit_mode_wins_and_zero_cap_disables_resident_ep():
    worker = _desired_phase_worker(
        cap=8, available=True, debug={"k3_prefill_moe_mode": "streamed_sp8"}
    )
    assert worker._desired_phase_key("prefill").prefill_moe_mode == "streamed_sp8"
    assert worker.parallel_manager.set_modes == ["streamed_sp8"]

    worker = _desired_phase_worker(cap=0, available=True, debug={})
    assert worker._desired_phase_key("prefill").prefill_moe_mode == "streamed"
    assert worker.parallel_manager.set_modes == ["streamed"]


def test_startup_gate_raises_with_bad_manager():
    worker = _isolated_worker_class()()
    worker.parallel_manager = object()
    with pytest.raises(RuntimeError, match="phase lifecycle contract failed"):
        worker._verify_manager_phase_contract()
