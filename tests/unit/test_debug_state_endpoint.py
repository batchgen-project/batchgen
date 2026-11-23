"""POST/GET /v1/admin/debug-state: the one way to change the server's debug state.

The real handlers are lifted out of ``create_app`` and run against a fake
worker manager, so these need neither a worker pool nor a GPU.
"""
import ast
import asyncio
import copy
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi import HTTPException, Request  # noqa: E402

from batchgen.config.model_registry import DEBUG_KEY_REGISTRY  # noqa: E402
from batchgen.server_debug_state import ADMIN_ENDPOINT, DebugState, DebugStateError  # noqa: E402

SERVER = Path(__file__).resolve().parents[2] / "batchgen" / "server"


def _handlers():
    tree = ast.parse((SERVER / "http_server.py").read_text())
    nodes = [
        copy.deepcopy(node) for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name in ("get_debug_state", "change_debug_state")
    ]
    assert len(nodes) == 2
    routes = {
        node.name: [ast.unparse(decorator) for decorator in node.decorator_list]
        for node in nodes
    }
    for node in nodes:
        node.decorator_list = []
    namespace = {
        "HTTPException": HTTPException,
        "Request": Request,
        "DebugStateError": DebugStateError,
        "asyncio": asyncio,
        "make_main_thread_dispatcher": lambda loop: ("dispatcher", loop),
    }
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "http_server.py", "exec"), namespace)
    return namespace["get_debug_state"], namespace["change_debug_state"], routes


GET, POST, ROUTES = _handlers()


class _Manager:
    """What the handlers use of the worker manager."""

    def __init__(self, *, restart_error=None):
        self.debug_state = DEBUG_KEY_REGISTRY.empty
        self.published = []
        self.restarts = []
        self.dispatcher = None
        self._restart_error = restart_error

    def publish_debug_state(self, state):
        self.debug_state = state
        self.published.append(state.to_wire())

    def set_main_thread_dispatcher(self, dispatcher):
        self.dispatcher = dispatcher

    def restart_workers(self, *, timeout_s):
        # The pool restarts under the state the handler already installed.
        self.restarts.append((timeout_s, self.debug_state.to_wire()))
        if self._restart_error is not None:
            raise self._restart_error
        return {"state": "serving", "generation": 4}


def _request(manager, body, *, node_rank=0):
    async def json():
        if isinstance(body, Exception):
            raise body
        return body

    state = SimpleNamespace(worker=manager, server_args=SimpleNamespace(node_rank=node_rank))
    return SimpleNamespace(app=SimpleNamespace(state=state), json=json)


def _post(manager, body, **kwargs):
    return asyncio.run(POST(_request(manager, body, **kwargs)))


def test_both_handlers_are_routed_at_the_admin_endpoint():
    assert ADMIN_ENDPOINT == "/v1/admin/debug-state"
    assert ROUTES == {
        "get_debug_state": ["app.get(DEBUG_STATE_ENDPOINT)"],
        "change_debug_state": ["app.post(DEBUG_STATE_ENDPOINT)"],
    }


def test_get_returns_the_current_generation_and_values():
    manager = _Manager()
    assert asyncio.run(GET(_request(manager, None))) == {"generation": 0, "values": {}}
    _post(manager, {"set": {"glm5_graph_path_log": "on"}})
    assert asyncio.run(GET(_request(manager, None))) == {
        "generation": 1, "values": {"glm5_graph_path_log": True},
    }


def test_a_phase_entry_key_is_published_to_the_running_pool_without_a_restart():
    manager = _Manager()
    answer = _post(manager, {"set": {"prefix_pool_pad_poison": True}})
    assert answer == {
        "generation": 1,
        "values": {"prefix_pool_pad_poison": True},
        "applied": "next_phase_entry",
    }
    assert manager.published == [{"generation": 1, "values": {"prefix_pool_pad_poison": True}}]
    assert manager.restarts == []

    answer = _post(manager, {"clear": ["prefix_pool_pad_poison"]})
    assert answer == {"generation": 2, "values": {}, "applied": "next_phase_entry"}
    assert manager.published[-1] == {"generation": 2, "values": {}}


@pytest.mark.parametrize("body,message", [
    ({"set": {"glm5_grpah_path_log": True}}, "unknown debug key 'glm5_grpah_path_log'"),
    ({"set": {"kimi_decode_graph_mode": "fast"}}, "debug key kimi_decode_graph_mode must be one of"),
    ({"set": {"prefix_rank_assignment_mode": "global_idx"}}, "--prefix-rank-assignment-mode"),
    ({"clear": ["glm5_moe_mode"]}, "glm5_moe_mode is retired"),
])
def test_an_unknown_retired_or_malformed_key_is_refused_by_name_and_changes_nothing(
    body, message
):
    manager = _Manager()
    with pytest.raises(HTTPException) as refused:
        _post(manager, body)
    assert refused.value.status_code == 400
    assert message in refused.value.detail
    assert manager.debug_state is DEBUG_KEY_REGISTRY.empty
    assert manager.published == [] and manager.restarts == []


@pytest.mark.parametrize("body", [None, [], "x", {"set": {}, "extra": 1}, ValueError("no json")])
def test_a_body_of_the_wrong_shape_is_refused(body):
    manager = _Manager()
    with pytest.raises(HTTPException) as refused:
        _post(manager, body)
    assert refused.value.status_code == 400
    assert manager.published == [] and manager.restarts == []


def test_a_rewarm_key_is_refused_until_the_request_asks_for_the_rewarm():
    manager = _Manager()
    with pytest.raises(HTTPException) as refused:
        _post(manager, {"set": {"kimi_decode_graph_mode": "eager", "triton_strict": True}})
    assert refused.value.status_code == 409
    assert refused.value.detail["rewarm_keys"] == ["kimi_decode_graph_mode"]
    assert "rewarm: true" in refused.value.detail["message"]
    # Refused as a whole: the phase-entry key of the same request is not applied.
    assert manager.debug_state is DEBUG_KEY_REGISTRY.empty
    assert manager.published == [] and manager.restarts == []


def test_a_rewarm_restarts_the_pool_under_the_new_state_and_publishes_nothing():
    manager = _Manager()
    answer = _post(manager, {
        "set": {"kimi_decode_graph_mode": "eager"}, "rewarm": True, "timeout": 30,
    })
    new = {"generation": 1, "values": {"kimi_decode_graph_mode": "eager"}}
    assert answer == {**new, "applied": "rewarm",
                      "restart": {"state": "serving", "generation": 4}}
    # The new pool's launch arguments carry the state; the old pool is not told.
    assert manager.restarts == [(30.0, new)]
    assert manager.published == []
    assert manager.dispatcher is not None
    assert manager.debug_state.to_wire() == new


def test_clearing_a_rewarm_key_is_a_rewarm_too():
    manager = _Manager()
    _post(manager, {"set": {"glm52_recv_count_hist": True}, "rewarm": True})
    with pytest.raises(HTTPException) as refused:
        _post(manager, {"clear": ["glm52_recv_count_hist"]})
    assert refused.value.status_code == 409
    answer = _post(manager, {"clear": ["glm52_recv_count_hist"], "rewarm": True})
    assert answer["applied"] == "rewarm" and answer["values"] == {}


def test_setting_a_rewarm_key_to_its_current_value_needs_no_rewarm():
    manager = _Manager()
    _post(manager, {"set": {"k3_prefill_profile": True}, "rewarm": True})
    answer = _post(manager, {"set": {"k3_prefill_profile": "true", "triton_strict": True}})
    assert answer["applied"] == "next_phase_entry"
    assert len(manager.restarts) == 1


def test_a_failed_rewarm_keeps_the_state_the_pool_was_started_with():
    manager = _Manager(restart_error=RuntimeError("workers did not come back"))
    with pytest.raises(RuntimeError, match="did not come back"):
        _post(manager, {"set": {"kimi_decode_graph_mode": "eager"}, "rewarm": True})
    assert manager.debug_state is DEBUG_KEY_REGISTRY.empty


def test_only_node_zero_owns_the_debug_state():
    manager = _Manager()
    with pytest.raises(HTTPException) as refused:
        _post(manager, {"set": {"triton_strict": True}}, node_rank=1)
    assert refused.value.status_code == 409
    assert refused.value.detail["coordinator_node"] == 0
    assert manager.published == []


# ------------------------------------------------- the manager's publish path

def _publish():
    tree = ast.parse((SERVER / "worker_manager.py").read_text())
    node = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "publish_debug_state")
    namespace = {"DebugState": DebugState}
    exec(compile(ast.fix_missing_locations(
        ast.Module(body=[copy.deepcopy(node)], type_ignores=[])
    ), "worker_manager.py", "exec"), namespace)
    return namespace["publish_debug_state"]


class _Queue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def test_publishing_installs_the_state_and_queues_it_on_the_admission_queue():
    state = DEBUG_KEY_REGISTRY.empty.changed(set={"triton_strict": True})
    manager = SimpleNamespace(
        _lock=threading.Lock(), debug_state=DEBUG_KEY_REGISTRY.empty, request_queue=_Queue()
    )
    _publish()(manager, state)
    assert manager.debug_state is state
    assert manager.request_queue.items == [
        {"type": "debug_state", "state": {"generation": 1, "values": {"triton_strict": True}}}
    ]


def test_publishing_without_worker_channels_only_installs_the_state():
    state = DEBUG_KEY_REGISTRY.empty.changed(set={"triton_strict": True})
    manager = SimpleNamespace(
        _lock=threading.Lock(), debug_state=DEBUG_KEY_REGISTRY.empty, request_queue=None
    )
    _publish()(manager, state)
    # The next pool start carries it in its init message.
    assert manager.debug_state is state


def test_every_pool_start_carries_the_debug_state_in_the_workers_launch_arguments():
    # Not in the init message: that arrives after the start-up warm-up, too
    # late for a key that is baked into captured graphs (seen on a real
    # server: a restarted pool warmed up under the empty state).
    tree = ast.parse((SERVER / "worker_manager.py").read_text())
    spawn = next(node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name == "_spawn_workers")
    keywords = [
        ast.unparse(keyword.value) for node in ast.walk(spawn)
        if isinstance(node, ast.Call) for keyword in node.keywords
        if keyword.arg == "debug_state"
    ]
    assert keywords == ["self.debug_state.to_wire()"]
    assert '"debug_state"' not in (SERVER / "batch_scheduler.py").read_text()
    assert '"debug_state":' not in (SERVER / "worker_manager.py").read_text()


def test_no_request_entry_carries_debug_switches_to_a_worker():
    assert '"batchgen_debug"' not in (SERVER / "batch_scheduler.py").read_text()
