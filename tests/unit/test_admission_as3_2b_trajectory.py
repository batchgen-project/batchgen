from types import SimpleNamespace

from tests.worker_method_source import load_worker_methods


_METHOD = load_worker_methods(["_trajectory_frontier_eviction_hook"], {})[
    "_trajectory_frontier_eviction_hook"
]


def test_worker_trajectory_frontier_eviction_hook_delegates_to_store():
    hook = object()
    store = SimpleNamespace(trajectory_frontier_eviction_hook=lambda: hook)
    worker = SimpleNamespace(_traj=store)

    assert _METHOD(worker) is hook
