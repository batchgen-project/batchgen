"""AST-isolated CPU checks for the worker phase transition table."""

import ast
import copy
import gc
import logging
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import pytest

from batchgen.phase_lifecycle import PHASES, ExitMode, PhaseEnterContext, PhaseKey
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
        "_begin_prefill_wave",
        "_server_prefill_token_cap",
        "_require_server_prefill_moe_mode",
        "_phase_exit_mode",
        "_retire_kept_phase_instances",
        "_require_resumable_instance",
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
        "PHASES": PHASES,
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
        if mode is ExitMode.RETIRE:
            self.held[key.phase] = False

    def phase_enter(self, key, ctx):
        self.events.append(("enter", key.phase, ctx.comm, ctx.max_num_seq))
        return object(), {}


def _worker(manager, events):
    cls = _isolated_worker_class()
    worker = cls()
    worker.rank = 0
    worker._live_phase_key = None
    worker._kept_phase_keys = {}
    # The prefill MoE mode the server was started in (see _server_worker
    # for resolution from a manager).
    worker._server_prefill_moe_mode_resolved = ("streamed",)
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
    """Records mode selections and phase entries; keeps every instance it enters.

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
        self.kept = set()

    def default_prefill_moe_mode(self):
        self.default_calls += 1
        return self.default_mode

    def set_prefill_moe_mode(self, mode):
        self.set_modes.append(mode)
        self._prefill_moe_mode = mode

    def resident_ep_prefill_available(self):
        return True

    def holds_instance(self, phase):
        return phase in self.kept

    def phase_exit(self, key, mode):
        self.events.append(("exit", key.phase, key.prefill_moe_mode, mode))
        if mode is ExitMode.RETIRE:
            self.kept.discard(key.phase)

    def phase_enter(self, key, ctx):
        self.events.append(("enter", key.phase, key.prefill_moe_mode))
        self.kept.add(key.phase)
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
    """Resolve, plan and transition the way one wave does.

    Returns the wave's key and the token cap its readers (receipt placement,
    the pre-transition pool plan and the live prefill) all use.
    """
    worker._sequences = _sequences(*(debugs or (None,)))
    uuids = list(worker._sequences)
    worker._begin_prefill_wave(uuids)
    worker._phase_request_uuids = tuple(uuids)
    key = worker._desired_phase_key("prefill")
    worker._transition_phase(key)
    return key, worker._server_prefill_token_cap()


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
    worker._kept_phase_keys = {
        "prefill": PhaseKey("prefill", "streamed"),
        "decode": PhaseKey("decode", "streamed"),
    }

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


_BOTH_KEPT_STREAMED = {
    "prefill": PhaseKey("prefill", "streamed"),
    "decode": PhaseKey("decode", "streamed"),
}


def test_prepare_phase_exit_orders_cleanup_before_manager_exit_and_resets_only_for_unstreamed_prefill_to_decode():
    events = []
    manager = _Manager(events, {"prefill": True, "decode": True})
    worker = _worker(manager, events)
    worker._live_phase_key = PhaseKey("prefill", "streamed")
    worker._kept_phase_keys = _BOTH_KEPT_STREAMED.copy()

    worker._transition_phase(PhaseKey("decode", "streamed"))

    exit_index = events.index(("exit", "prefill", ExitMode.SUSPEND))
    for event in ("weight_close", "stop_h2d", "clear_kv", "reset_decode"):
        assert events.index(event) < exit_index

    events = []
    manager = _Manager(events, {"prefill": True, "decode": True})
    worker = _worker(manager, events)
    worker._streamed_sp8_h2d_installed = True
    worker._live_phase_key = PhaseKey("prefill", "streamed")
    worker._kept_phase_keys = _BOTH_KEPT_STREAMED.copy()
    worker._transition_phase(PhaseKey("decode", "streamed"))
    assert "reset_decode" not in events

    events = []
    manager = _Manager(events, {"prefill": True, "decode": True})
    worker = _worker(manager, events)
    worker._live_phase_key = PhaseKey("decode", "streamed")
    worker._kept_phase_keys = _BOTH_KEPT_STREAMED.copy()
    worker._transition_phase(PhaseKey("prefill", "streamed"))
    assert "reset_decode" not in events


def test_every_prefill_wave_uses_the_server_mode_across_decode_phases():
    worker, events = _server_worker()
    # Startup resolves the server mode once and enters prefill in it.
    _start(worker)

    keys = []
    for wave in range(3):
        key, cap = _run_prefill_wave(worker)
        keys.append(key)
        assert cap == STREAMED_CAP
        if wave < 2:
            _run_decode(worker)

    assert {key.prefill_moe_mode for key in keys} == {"streamed_sp8"}
    # The worker never re-selects the mode per wave; startup installs it.
    assert worker.parallel_manager.set_modes == []
    # Decided once for the process, never re-read per wave, even though the
    # manager offers a built, available resident shard after every decode.
    assert worker.parallel_manager.default_calls == 1
    entries = [e for e in events if e[0] == "enter"]
    assert {e[2] for e in entries} == {"streamed_sp8"}


def test_a_resident_ep_server_runs_every_wave_resident_from_the_first():
    worker, events = _server_worker(
        manager=_ServerManager([], default_mode="resident_ep")
    )
    _start(worker)
    for _ in range(2):
        key, cap = _run_prefill_wave(worker)
        assert key.prefill_moe_mode == "resident_ep"
        assert cap == RESIDENT_CAP
        _run_decode(worker)
    assert worker._desired_phase_key("decode").prefill_moe_mode == "resident_ep"
    assert {e[2] for e in events if e[0] == "enter"} == {"resident_ep"}


def test_a_retired_mode_debug_key_never_changes_the_wave_mode():
    # Admission refuses the key; even if one reached a worker it selects
    # nothing: the wave runs the server mode.
    worker, _ = _server_worker()
    _start(worker)
    _run_decode(worker)
    key, cap = _run_prefill_wave(worker, {"k3_prefill_moe_mode": "resident_ep"})
    assert key.prefill_moe_mode == "streamed_sp8"
    assert cap == STREAMED_CAP
    assert worker.parallel_manager.set_modes == []


def test_profile_overrides_are_the_only_key_field_that_varies_between_waves():
    worker, _ = _server_worker()
    _start(worker)
    plain, _ = _run_prefill_wave(worker)
    _run_decode(worker)
    profiled, _ = _run_prefill_wave(worker, {"k3_prefill_profile": True})
    assert plain.prefill_moe_mode == profiled.prefill_moe_mode == "streamed_sp8"
    assert plain.profile_overrides == ()
    assert profiled.profile_overrides == (("k3_prefill_profile", True),)


def test_phase_enter_refuses_a_key_with_another_mode_and_names_both():
    events = []
    manager = _Manager(events, {"prefill": True, "decode": True})
    worker = _worker(manager, events)
    live = PhaseKey("decode", "streamed")
    worker._live_phase_key = live
    worker._kept_phase_keys = {"decode": live}
    with pytest.raises(
        RuntimeError, match="'resident_ep', but this server runs 'streamed'"
    ):
        worker._transition_phase(PhaseKey("prefill", "resident_ep"))
    # Refused before the live phase is left: nothing exited or entered.
    assert events == []
    assert worker._live_phase_key == live
    with pytest.raises(RuntimeError, match="no wave changes it"):
        worker._transition_phase(PhaseKey("decode", None))


def test_pool_plan_cap_is_the_server_cap_for_every_group_subset():
    source = _method_source("_plan_prefill_pool_wave")
    assert "self._server_prefill_token_cap()" in source
    assert "_active_batchgen_debug_for_sequences" not in source


def test_phase_key_refuses_a_wave_that_was_not_resolved():
    worker, _ = _server_worker()
    worker._sequences = _sequences(None)
    worker._phase_request_uuids = ("u0",)
    with pytest.raises(RuntimeError, match="did not resolve"):
        worker._desired_phase_key("prefill")


def test_a_resident_ep_server_without_a_planned_resident_cap_uses_the_streamed_cap():
    worker, _ = _server_worker(
        resident_cap=0, manager=_ServerManager([], default_mode="resident_ep")
    )
    assert worker._server_prefill_token_cap() == STREAMED_CAP
    for mode in ("streamed", "streamed_sp8"):
        worker, _ = _server_worker(manager=_ServerManager([], default_mode=mode))
        assert worker._server_prefill_token_cap() == STREAMED_CAP


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
    key, cap = _run_prefill_wave(worker)
    assert key.prefill_moe_mode is None
    assert cap == STREAMED_CAP


def test_the_server_option_on_a_manager_without_modes_is_refused():
    worker, _ = _server_worker(manager=_NoModeManager([]))
    worker.args.prefill_moe_mode = "resident_ep"
    with pytest.raises(ValueError, match="has no prefill MoE modes"):
        worker._server_prefill_moe_mode()


def test_a_manager_that_does_not_honour_the_server_option_is_refused():
    worker, _ = _server_worker()
    worker.args.prefill_moe_mode = "resident_ep"
    with pytest.raises(RuntimeError, match="started with --prefill-moe-mode"):
        worker._server_prefill_moe_mode()


def test_a_manager_declaring_half_the_mode_capability_is_refused():
    class _HalfManager:
        def set_prefill_moe_mode(self, mode):
            pass

    worker, _ = _server_worker()
    worker.parallel_manager = _HalfManager()
    with pytest.raises(RuntimeError, match="declares only one of"):
        worker._server_prefill_moe_mode()


def test_no_per_batch_prefill_mode_selection_remains_in_the_worker():
    source = WORKER.read_text()
    for name in (
        "k3_prefill_moe_mode",
        "_prefill_moe_mode_for_debug",
        "_prefill_wave_moe_mode",
    ):
        assert name not in source


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


# --------------------------------------------------------------------------- #
#  Real Kimi-K3 phase hooks driven through the real worker transition table   #
# --------------------------------------------------------------------------- #

# Manager methods taken verbatim from the K3 Parallel_Strategy_Manager: every
# hook of the phase contract and the instance bookkeeping they consult.
K3_REAL_METHODS = {
    "phase_enter",
    "admit",
    "phase_exit",
    "holds_instance",
    "activate_prefill",
    "activate_decoding",
    "configure_prefill",
    "configure_decoding",
    "_require_built_model",
    "_publish_prefill_instance",
    "_publish_decode_instance",
    "_instances_are_reused",
    "decode_experts_resident",
    "release_decode_routed_experts",
    "_reset_model_attached_state",
    "release_decode_graph",
    "set_comm",
    "set_prefill_moe_mode",
    "_validated_prefill_moe_mode",
    "default_prefill_moe_mode",
    "prefill_uses_resident_ep",
    "prefill_uses_streamed_sp8",
    "_decode_moe_mode",
}


class _FakeModel:
    def __init__(self, serial):
        self.serial = serial


class _FakeStreamedSP8Buffer:
    """Stands in for StreamedSP8LayerBuffer: attached to one model's modules."""

    def __init__(self, model):
        self.model = model
        self.closed = False

    def close(self):
        # Closing detaches the buffer from the model's modules.
        self.closed = True
        self.model = None


def _k3_manager_class():
    tree = ast.parse(KIMI_PSM.read_text())
    source_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "KimiLinearParallelStrategyManager"
    )
    methods = [
        copy.deepcopy(node)
        for node in source_class.body
        if isinstance(node, ast.FunctionDef) and node.name in K3_REAL_METHODS
    ]
    assert {node.name for node in methods} == K3_REAL_METHODS
    module = ast.Module(
        body=[
            ast.ClassDef(
                name="RealK3Hooks",
                bases=[],
                keywords=[],
                body=methods,
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    namespace = {
        "AdmitContext": object,
        "ExitMode": ExitMode,
        "PhaseEnterContext": PhaseEnterContext,
        "PhaseKey": PhaseKey,
        "PREFILL_MOE_MODES": _k3_module_constant("PREFILL_MOE_MODES"),
        "logging": logging,
        "time": __import__("time"),
    }
    exec(compile(ast.fix_missing_locations(module), str(KIMI_PSM), "exec"), namespace)

    class K3Manager(namespace["RealK3Hooks"]):
        """Real K3 hooks over a model whose build and mode switches are fakes.

        The fakes keep the two properties the lifecycle depends on: one model
        object serves both phases, and the streamed-SP8 buffer is attached to
        that model and must be closed before the model goes.
        """

        def __init__(self, events):
            self.events = events
            self.rank = 0
            self.global_rank = 0
            self._is_k3 = True
            self._distributed_weight_sharded = False
            self._latent_tp_planned = True
            self._stream_all_modules = False
            self._attn_tp_size = 8
            self.engine_config = SimpleNamespace(
                EP_Config=SimpleNamespace(decode_moe_mode="resident_ep"),
                Basic_Config=SimpleNamespace(prefill_moe_mode=None),
            )
            self._comm = None
            self._resident_ep_built = False
            self._streamed_sp8_buffer = None
            self._prefill_moe_mode = self.default_prefill_moe_mode()
            self._decode_graph = None
            self.reuse_phase_instances = False
            self.prefill_instance = None
            self.decode_instance = None
            self._active_phase = None
            self.model = None
            self.weight_copy_task = {}
            self.builds = 0
            self.buffers = []
            self.fail_next_build = False

        def _build_model(self):
            if self.fail_next_build:
                self.fail_next_build = False
                raise RuntimeError("injected build failure")
            self.builds += 1
            self.model = _FakeModel(self.builds)
            self._reset_model_attached_state()
            self.events.append(("build", self.model.serial))

        def _enter_prefill(self):
            assert self.model is not None
            self._active_phase = "prefill"
            if self.prefill_uses_streamed_sp8():
                self._release_resident_ep_decode()
                if self._streamed_sp8_buffer is None:
                    self._streamed_sp8_buffer = _FakeStreamedSP8Buffer(self.model)
                    self.buffers.append(self._streamed_sp8_buffer)
            elif self.prefill_uses_resident_ep():
                self._release_streamed_sp8_prefill()
                self._resident_ep_built = True
            self.events.append(
                ("model_prefill", self.model.serial, self._prefill_moe_mode)
            )
            return {}

        def _enter_decoding(self):
            assert self.model is not None
            self._active_phase = "decode"
            self._release_streamed_sp8_prefill()
            self._resident_ep_built = True
            self.events.append(("model_decode", self.model.serial))
            return dict.fromkeys(
                ("release_streamed", "copy_task", "resident_ep", "decode_graph"),
                0.0,
            )

        def _release_streamed_sp8_prefill(self):
            if self._streamed_sp8_buffer is not None:
                self._streamed_sp8_buffer.close()
            self._streamed_sp8_buffer = None

        def _release_resident_ep_decode(self):
            if not self._resident_ep_built:
                return 0
            self._resident_ep_built = False
            return 0

        def _prewarm_triton_kernels(self):
            pass

        def _log_phase_activation(self, phase, start, stages):
            pass

    return K3Manager


def _k3_module_constant(name):
    tree = ast.parse(KIMI_PSM.read_text())
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(target, "id", None) == name for target in node.targets)
    )
    return ast.literal_eval(node.value)


class _DeadModels:
    """Freed models tracked by weak identity.

    Membership compares objects (``ref() is model``), never ``id()``: a
    collected fake's address can be reused by the next ``_FakeModel``
    allocation, and an id-keyed set then flags a live model as dead
    (seen ~1/5 runs). Weak references also leave the freed models
    collectable, which ``_assert_freed`` requires.
    """

    def __init__(self):
        self._refs = []

    def add(self, model):
        self._refs.append(weakref.ref(model))

    def __contains__(self, model):
        return any(ref() is model for ref in self._refs)


def _k3_worker():
    manager = _k3_manager_class()([])
    worker, events = _server_worker(manager=manager)
    dead = _DeadModels()

    def deep_free():
        events.append("deep_free")
        if worker.model is not None:
            dead.add(worker.model)
            worker.model = None

    worker.deep_free_model_memory = deep_free
    worker.dead_models = dead
    worker.comm = "comm"
    return worker, manager, events


def _k3_start(worker):
    """The K3 startup order: decode shard first, then the server prefill mode."""
    worker.parallel_manager.reuse_phase_instances = True
    worker._transition_phase(
        worker._desired_phase_key("decode"), comm="comm", max_num_seq=8
    )
    mode = worker._server_prefill_moe_mode()
    worker.parallel_manager.set_prefill_moe_mode(mode)
    worker._phase_profile_overrides = ()
    worker._transition_phase(PhaseKey("prefill", mode, ()))
    _assert_no_state_outlives_its_model(worker)


def _assert_no_state_outlives_its_model(worker):
    manager = worker.parallel_manager
    assert manager.model is not None
    assert manager.model not in worker.dead_models
    assert worker.model is manager.model
    for instance in (manager.prefill_instance, manager.decode_instance):
        if instance is not None:
            assert instance[0] is manager.model
    for buffer in manager.buffers:
        # A buffer is either closed or attached to the live model.
        assert buffer.closed or (
            buffer is manager._streamed_sp8_buffer
            and buffer.model is manager.model
        )


def _k3_wave(worker, *debugs):
    key = _run_prefill_wave(worker, *debugs)[0]
    _assert_no_state_outlives_its_model(worker)
    return key


def _k3_decode(worker):
    changed, resumed = worker._transition_phase(
        worker._desired_phase_key("decode"), comm="comm", max_num_seq=8
    )
    assert changed
    _assert_no_state_outlives_its_model(worker)
    return resumed


PROFILE_ON = {"k3_prefill_profile": True}


def _assert_freed(ref):
    gc.collect()
    assert ref() is None, "a retired model is still referenced"


def test_k3_profile_key_change_after_decode_cycles_retires_every_kept_instance():
    # The reported failure shape on the key field that still varies between waves:
    # after server-mode decode cycles, a profile_overrides change makes the
    # decode exit a RETIRE. Before the fix the K3 RETIRE dropped only the
    # decode tuple, the prefill instance kept from the earlier waves held the
    # old model through deep_free_model_memory, and phase_enter resumed it.
    worker, manager, events = _k3_worker()
    _k3_start(worker)
    for _ in range(4):
        assert _k3_wave(worker).prefill_moe_mode == "streamed_sp8"
        _k3_decode(worker)
    assert manager.builds == 1
    first_model = weakref.ref(manager.model)

    key = _k3_wave(worker, PROFILE_ON)

    assert key == PhaseKey(
        "prefill", "streamed_sp8", (("k3_prefill_profile", True),)
    )
    assert manager.builds == 2
    assert events.count("deep_free") == 1
    assert ("model_prefill", 2, "streamed_sp8") in events
    assert worker._kept_phase_keys == {"prefill": key}
    assert not manager.holds_instance("decode")
    _assert_freed(first_model)


def test_k3_same_mode_waves_resume_the_kept_instances():
    worker, manager, events = _k3_worker()
    _k3_start(worker)
    resumed = []
    for _ in range(3):
        _k3_wave(worker)
        resumed.append(_k3_decode(worker))

    assert manager.builds == 1
    assert resumed == [True, True, True]
    assert "deep_free" not in events
    assert all(
        event[1] == 1
        for event in events
        if isinstance(event, tuple) and event[0].startswith("model_")
    )


def test_k3_profile_key_change_on_the_first_wave_closes_the_startup_buffer():
    # The first wave after startup is a prefill->prefill key change: the
    # startup streamed_sp8 buffer is attached to the model being retired.
    worker, manager, events = _k3_worker()
    _k3_start(worker)
    startup_buffer = manager._streamed_sp8_buffer
    assert startup_buffer is not None and not startup_buffer.closed
    startup_model = weakref.ref(manager.model)

    _k3_wave(worker, PROFILE_ON)
    assert startup_buffer.closed
    assert manager.builds == 2
    _assert_freed(startup_model)

    # Back to no profile override: another key change, another build.
    _k3_wave(worker)
    assert manager.builds == 3
    _k3_decode(worker)
    _k3_wave(worker)
    assert manager.builds == 3


def test_k3_wave_after_a_failed_transition_enters_cleanly():
    worker, manager, events = _k3_worker()
    _k3_start(worker)
    _k3_wave(worker)
    _k3_decode(worker)

    manager.fail_next_build = True
    with pytest.raises(RuntimeError, match="injected build failure"):
        _run_prefill_wave(worker, PROFILE_ON)
    # The decode phase was left and retired before the build failed: nothing
    # is live and nothing is kept.
    assert worker._live_phase_key is None
    assert not manager.holds_instance("prefill")
    assert not manager.holds_instance("decode")
    exits_before = events.count("weight_close")

    assert _k3_wave(worker).prefill_moe_mode == "streamed_sp8"
    # Entering from "nothing live" runs no second exit of the retired phase.
    assert events.count("weight_close") == exits_before
    assert manager.builds == 2
    _k3_decode(worker)
    _k3_wave(worker)
    assert manager.builds == 2


# --------------------------------------------------------------------------- #
#  Kept-instance keys: every manager, not only K3                             #
# --------------------------------------------------------------------------- #


def test_retire_retires_every_kept_instance_under_its_own_key_before_deep_free():
    # A manager with one model per phase (GLM-5 shape): a key change retires
    # the live decode instance, and the prefill instance kept under the old
    # key goes with it instead of being resumed under the new one.
    events = []
    manager = _Manager(events, {"prefill": True, "decode": True})
    worker = _worker(manager, events)
    old_prefill = PhaseKey("prefill", "streamed", (("k3_prefill_profile", False),))
    live = PhaseKey("decode", "streamed", (("k3_prefill_profile", False),))
    worker._live_phase_key = live
    worker._kept_phase_keys = {"prefill": old_prefill, "decode": live}

    desired = PhaseKey("prefill", "streamed", (("k3_prefill_profile", True),))
    worker._transition_phase(desired)

    live_exit = events.index(("exit", "decode", ExitMode.RETIRE))
    kept_exit = events.index(("exit", "prefill", ExitMode.RETIRE))
    assert live_exit < kept_exit < events.index("deep_free") < events.index(
        ("enter", "prefill", None, None)
    )
    assert worker._live_phase_key == desired


def test_a_kept_incoming_instance_under_another_key_is_refused_by_name():
    # Unreachable while every key change RETIREs every kept instance; if a
    # manager or a later change ever left one, entry fails loudly instead of
    # resuming an instance built for another key.
    events = []
    manager = _Manager(events, {"prefill": True, "decode": True})
    worker = _worker(manager, events)
    live = PhaseKey("decode", "streamed")
    worker._live_phase_key = live
    worker._kept_phase_keys = {
        "prefill": PhaseKey("prefill", "streamed", (("k3_prefill_profile", True),)),
        "decode": live,
    }
    with pytest.raises(RuntimeError, match="resumed only under its own key"):
        worker._transition_phase(PhaseKey("prefill", "streamed"))
    assert not [e for e in events if isinstance(e, tuple) and e[0] == "enter"]


def test_entering_over_an_instance_the_worker_never_entered_fails_loudly():
    events = []
    manager = _Manager(events, {"prefill": True})
    worker = _worker(manager, events)
    with pytest.raises(RuntimeError, match="kept under None"):
        worker._transition_phase(PhaseKey("prefill", "streamed"))
    assert not [e for e in events if isinstance(e, tuple) and e[0] == "enter"]


def test_a_retire_that_keeps_its_instance_is_refused():
    class _StickyManager(_Manager):
        def phase_exit(self, key, mode):
            self.events.append(("exit", key.phase, mode))

    events = []
    manager = _StickyManager(events, {"prefill": True})
    worker = _worker(manager, events)
    live = PhaseKey("prefill", "streamed")
    worker._live_phase_key = live
    worker._kept_phase_keys = {"prefill": live}
    with pytest.raises(RuntimeError, match="still holds a prefill instance"):
        worker._transition_phase(PhaseKey("prefill", "streamed", (("k3_prefill_profile", True),)))
    assert "deep_free" not in events
