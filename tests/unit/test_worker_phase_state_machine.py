"""AST-isolated CPU checks for the worker phase transition table."""

import ast
import copy
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

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
        "_begin_prefill_wave",
        "_prefill_token_cap_for_mode",
        "_live_prefill_token_cap",
        "_wave_prefill_token_cap",
        "_placement_prefill_token_cap",
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
        "List": List,
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
RESIDENT_DEBUG = {"k3_prefill_moe_mode": "resident_ep"}


class _ServerManager:
    """Records mode selections and phase entries; holds every instance.

    It also offers the surface the removed post-decode resident handoff
    consulted (a built resident shard that reports itself available), so a
    reintroduced handoff would switch these waves to resident-EP and fail the
    server-mode assertions, not only the source-string check.
    """

    def __init__(self, events, default_mode="streamed_sp8"):
        self.events = events
        self.default_mode = default_mode
        self.default_calls = 0
        self._prefill_moe_mode = default_mode
        self._resident_ep_built = True
        self.set_modes = []

    def default_prefill_moe_mode(self):
        self.default_calls += 1
        return self.default_mode

    def set_prefill_moe_mode(self, mode):
        self.set_modes.append(mode)
        self._prefill_moe_mode = mode

    def resident_ep_prefill_available(self):
        return True

    def holds_instance(self, phase):
        return True

    def phase_exit(self, key, mode):
        self.events.append(("exit", key.phase, key.prefill_moe_mode, mode))

    def phase_enter(self, key, ctx):
        self.events.append(("enter", key.phase, key.prefill_moe_mode))
        return object(), {}


def _sequences(*debugs):
    """One sequence per debug dict (None = no per-request flags)."""
    return {
        f"u{index}": SimpleNamespace(uuid=f"u{index}", batchgen_debug=debug)
        for index, debug in enumerate(debugs)
    }


def _server_worker(*, resident_cap=RESIDENT_CAP, manager=None):
    events = []
    worker = _worker(manager or _ServerManager(events), events)
    if manager is not None:
        manager.events = events
    worker.rank = 0
    worker._server_prefill_moe_mode_resolved = None
    worker._prefill_wave_uuids = None
    worker._phase_request_uuids = ()
    worker._sequences = {}
    worker.global_batch = SimpleNamespace(
        get_sequence=lambda uuid: worker._sequences.get(uuid)
    )

    # The worker's own first-value-wins merge of per-request flags.
    def merge(sequences):
        merged = {}
        for seq in sequences:
            for key, value in (seq.batchgen_debug or {}).items():
                merged.setdefault(key, value)
        return merged or None

    worker._active_batchgen_debug_for_sequences = merge
    worker._debug_flag_enabled = lambda value: bool(value)
    worker._weight_pass_end_delay = lambda value: 0.0
    worker._weight_passes = SimpleNamespace(
        end_delay_s=None, close=lambda: events.append("weight_close")
    )
    worker.engine_config = SimpleNamespace(
        Module_Batching_Config=SimpleNamespace(
            prefill_micro_batch_token_cap=STREAMED_CAP,
            resident_ep_prefill_token_cap=resident_cap,
        )
    )
    return worker, events


def _run_prefill_wave(worker, *debugs):
    """Resolve, plan, transition and read caps the way one wave does.

    Returns the wave's key and the caps its readers see: placement (receipt
    and wave-time packing), the pre-transition pool plan and the live wave.
    """
    worker._sequences = _sequences(*(debugs or (None,)))
    uuids = list(worker._sequences)
    worker._begin_prefill_wave(uuids)
    placement_cap = worker._placement_prefill_token_cap()
    pool_cap = worker._wave_prefill_token_cap()
    worker._phase_request_uuids = tuple(uuids)
    key = worker._desired_phase_key("prefill")
    worker._transition_phase(key)
    return key, placement_cap, pool_cap, worker._live_prefill_token_cap()


def _run_decode(worker):
    worker._transition_phase(
        worker._desired_phase_key("decode"), comm="comm", max_num_seq=8
    )


def _start(worker):
    worker._transition_phase(
        PhaseKey("prefill", worker._server_prefill_moe_mode(), ())
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
    _start(worker)

    keys = []
    for wave in range(3):
        key, placement_cap, pool_cap, live_cap = _run_prefill_wave(worker)
        keys.append(key)
        assert placement_cap == pool_cap == live_cap == STREAMED_CAP
        if wave < 2:
            _run_decode(worker)

    assert {key.prefill_moe_mode for key in keys} == {"streamed_sp8"}
    assert worker.parallel_manager.set_modes == ["streamed_sp8"] * 3
    # Decided once for the process, never re-read per wave, even though the
    # manager offers a built, available resident shard after every decode.
    assert worker.parallel_manager.default_calls == 1
    prefill_entries = [e for e in events if e[:2] == ("enter", "prefill")]
    assert {e[2] for e in prefill_entries} == {"streamed_sp8"}
    assert worker.engine_config.Module_Batching_Config.prefill_micro_batch_token_cap == (
        STREAMED_CAP
    )


def test_batch_override_applies_to_its_own_wave_only():
    worker, _ = _server_worker()
    _start(worker)
    _run_decode(worker)

    key, placement_cap, pool_cap, live_cap = _run_prefill_wave(
        worker, RESIDENT_DEBUG
    )
    assert key.prefill_moe_mode == "resident_ep"
    assert pool_cap == live_cap == RESIDENT_CAP
    # Placement scores the same at receipt and at wave time: server mode.
    assert placement_cap == STREAMED_CAP
    _run_decode(worker)

    key, placement_cap, pool_cap, live_cap = _run_prefill_wave(worker)
    assert key.prefill_moe_mode == "streamed_sp8"
    assert placement_cap == pool_cap == live_cap == STREAMED_CAP
    assert worker.parallel_manager.set_modes == ["resident_ep", "streamed_sp8"]
    # The resident wave never rewrites the planner's shared streamed cap.
    batching = worker.engine_config.Module_Batching_Config
    assert batching.prefill_micro_batch_token_cap == STREAMED_CAP
    assert batching.resident_ep_prefill_token_cap == RESIDENT_CAP


def test_a_mixed_debug_wave_plans_and_executes_under_one_mode():
    # One request asks for resident-EP, one carries no flag. The wave mode is
    # resolved once from both; a group serving only the plain request plans
    # its pool with the wave's cap, not a mode re-derived from its subset.
    worker, _ = _server_worker()
    _start(worker)
    _run_decode(worker)
    key, _, pool_cap, live_cap = _run_prefill_wave(worker, None, RESIDENT_DEBUG)
    assert key.prefill_moe_mode == "resident_ep"
    assert pool_cap == live_cap == RESIDENT_CAP


def test_pool_plan_cap_is_the_wave_cap_for_every_group_subset():
    source = _method_source("_plan_prefill_pool_wave")
    assert "self._wave_prefill_token_cap()" in source
    assert "_active_batchgen_debug_for_sequences" not in source


def test_phase_key_refuses_a_wave_that_was_not_resolved():
    worker, _ = _server_worker()
    worker._sequences = _sequences(None)
    worker._phase_request_uuids = ("u0",)
    with pytest.raises(RuntimeError, match="did not resolve"):
        worker._desired_phase_key("prefill")
    with pytest.raises(RuntimeError, match="before _begin_prefill_wave"):
        worker._wave_prefill_token_cap()


def test_resident_override_without_a_planned_resident_cap_uses_the_streamed_cap():
    worker, _ = _server_worker(resident_cap=0)
    assert worker._prefill_token_cap_for_mode("resident_ep") == STREAMED_CAP
    for mode in ("streamed", "streamed_sp8", None):
        assert worker._prefill_token_cap_for_mode(mode) == STREAMED_CAP


class _NoModeManager(_ServerManager):
    """A manager without the prefill MoE mode capability (GLM, GPT-OSS)."""

    default_prefill_moe_mode = None
    set_prefill_moe_mode = None
    resident_ep_prefill_available = None

    def __getattribute__(self, name):
        if name in (
            "default_prefill_moe_mode",
            "set_prefill_moe_mode",
            "resident_ep_prefill_available",
        ):
            raise AttributeError(name)
        return super().__getattribute__(name)


def test_a_manager_without_prefill_moe_modes_runs_mode_none():
    worker, _ = _server_worker(manager=_NoModeManager([]))
    assert worker._server_prefill_moe_mode() is None
    _start(worker)
    key, placement_cap, pool_cap, live_cap = _run_prefill_wave(worker)
    assert key.prefill_moe_mode is None
    assert placement_cap == pool_cap == live_cap == STREAMED_CAP
    with pytest.raises(ValueError, match="has no prefill MoE modes"):
        worker._prefill_moe_mode_for_debug(RESIDENT_DEBUG)


def test_a_manager_declaring_half_the_mode_capability_is_refused():
    class _HalfManager:
        def set_prefill_moe_mode(self, mode):
            pass

    worker, _ = _server_worker()
    worker.parallel_manager = _HalfManager()
    with pytest.raises(RuntimeError, match="declares only one of"):
        worker._server_prefill_moe_mode()


def test_live_cap_refuses_a_read_outside_a_live_prefill_phase():
    worker, _ = _server_worker()
    with pytest.raises(RuntimeError, match="live prefill wave"):
        worker._live_prefill_token_cap()
    worker._live_phase_key = PhaseKey("decode", "streamed_sp8")
    with pytest.raises(RuntimeError, match="live prefill wave"):
        worker._live_prefill_token_cap()


def _class_function_nodes(path, class_name):
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return [node for node in cls.body if isinstance(node, ast.FunctionDef)]


def _method_source(name):
    node = next(
        node
        for node in _class_function_nodes(WORKER, "BatchGenWorker")
        if node.name == name
    )
    return ast.unparse(node)


CAP_FIELDS = ("prefill_micro_batch_token_cap", "resident_ep_prefill_token_cap")
KIMI_PSM = (
    ROOT / "batchgen" / "models" / "moonshotai" / "kimi_linear"
    / "Parallel_Strategy_Manager.py"
)
CONFIG = ROOT / "batchgen" / "config" / "config.py"


def _cap_field_readers(nodes):
    """Functions naming a cap field as an attribute or a getattr string."""
    return sorted(
        node.name
        for node in nodes
        for sub in ast.walk(node)
        if (isinstance(sub, ast.Attribute) and sub.attr in CAP_FIELDS)
        or (isinstance(sub, ast.Constant) and sub.value in CAP_FIELDS)
    )


def test_every_cap_reader_goes_through_the_mode_accessor():
    # The worker and the K3 manager never read the planner caps directly;
    # prefill_token_cap_for_mode (config.py) is the single reader.
    assert _cap_field_readers(_class_function_nodes(WORKER, "BatchGenWorker")) == []
    assert _cap_field_readers(
        _class_function_nodes(KIMI_PSM, "KimiLinearParallelStrategyManager")
    ) == []
    config_functions = [
        node for node in ast.parse(CONFIG.read_text()).body
        if isinstance(node, ast.FunctionDef)
    ]
    assert sorted(set(_cap_field_readers(config_functions))) == [
        "prefill_token_cap_for_mode",
    ]


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
