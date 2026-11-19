"""The [SERVER_CONFIG] contract: value validation, the pinned line, the seams."""

import ast
import asyncio
import logging
import queue
from pathlib import Path
from types import SimpleNamespace

import pytest

from batchgen.worker import server_config as sc

ROOT = Path(__file__).resolve().parents[2]


def config(**overrides):
    values = dict(
        prefill_attn="dp:streamed",
        prefill_moe="dp:streamed",
        decode_attn="dp:resident",
        decode_moe="ep8:partial",
        cuda_graphs=True,
        graph_buckets=(1, 2, 4, 8),
        host_kv_gib=860.0,
        host_kv_pages=56320,
        device_kv_gib=52.5,
        device_kv_pages=3440,
        gpu_memory_frac=0.9,
        max_pool=3200,
    )
    values.update(overrides)
    return sc.ServerConfig(**values)


# -- the log line is pinned --------------------------------------------------

def test_log_line_format_is_pinned():
    assert config().log_line(1, 0) == (
        "[SERVER_CONFIG] generation=1 node=0 prefill_attn=dp:streamed "
        "prefill_moe=dp:streamed decode_attn=dp:resident decode_moe=ep8:partial "
        "cuda_graphs=on graph_buckets=1,2,4,8 "
        "host_kv_gib=860.000 host_kv_pages=56320 "
        "device_kv_gib=52.500 device_kv_pages=3440 "
        "gpu_memory_frac=0.9 max_pool=3200"
    )
    eager = config(cuda_graphs=False, graph_buckets=None, decode_moe="undeclared")
    assert eager.log_line(2, 3) == (
        "[SERVER_CONFIG] generation=2 node=3 prefill_attn=dp:streamed "
        "prefill_moe=dp:streamed decode_attn=dp:resident decode_moe=undeclared "
        "cuda_graphs=off graph_buckets=none "
        "host_kv_gib=860.000 host_kv_pages=56320 "
        "device_kv_gib=52.500 device_kv_pages=3440 "
        "gpu_memory_frac=0.9 max_pool=3200"
    )
    for line in (config().log_line(1, 0), eager.log_line(2, 3)):
        for pair in line.split()[1:]:
            assert "=" in pair  # key=value pairs, no spaces inside values


# -- value validation --------------------------------------------------------

def test_round_trip_and_refusals():
    payload = config().as_dict()
    assert payload["graph_buckets"] == [1, 2, 4, 8]  # queue-transportable
    assert sc.ServerConfig.from_dict(payload) == config()
    for bad in (
        None,
        [],
        {},
        {**payload, "extra": 1},
        {k: v for k, v in payload.items() if k != "max_pool"},
    ):
        with pytest.raises(sc.ServerConfigError):
            sc.ServerConfig.from_dict(bad)


@pytest.mark.parametrize("overrides", [
    dict(prefill_attn="fastest"),          # outside the vocabulary
    dict(decode_moe="ep:resident"),        # grouped scheme without its width
    dict(decode_attn="dp resident"),       # a space can never appear
    dict(cuda_graphs="on"),                # bool, not its rendering
    dict(graph_buckets=()),                # captured means at least one bucket
    dict(graph_buckets=(4, 2)),            # ladder is strictly increasing
    dict(graph_buckets=(1, 2), cuda_graphs=False),  # captured without graphs
    dict(host_kv_pages=-1),
    dict(device_kv_pages=2.5),
    dict(host_kv_gib=-0.1),
    dict(gpu_memory_frac=0.0),
    dict(gpu_memory_frac=1.5),
    dict(max_pool=True),
])
def test_invalid_values_are_refused(overrides):
    with pytest.raises(sc.ServerConfigError):
        config(**overrides)


def test_undeclared_slots_are_legal_but_unknown_tokens_are_not():
    assert config(
        prefill_attn="undeclared", prefill_moe="undeclared",
        decode_attn="undeclared", decode_moe="undeclared",
    ).log_line(1, 0).count("undeclared") == 4
    with pytest.raises(sc.ServerConfigError):
        config(prefill_moe="not-declared")


# -- publication: worker main loop -> scheduler -> WorkerManager -------------

def test_main_loop_publishes_the_snapshot_with_the_capacity():
    tree = ast.parse((ROOT / "batchgen/server_worker_main_loop.py").read_text())
    (message,) = [
        node for node in ast.walk(tree) if isinstance(node, ast.Dict)
        and any(isinstance(v, ast.Constant) and v.value == "deployment_capacity"
                for v in node.values)
    ]
    entries = {k.value: ast.unparse(v) for k, v in zip(message.keys, message.values)}
    assert entries["server_config"] == "worker.server_config"


def _transient():
    return [{"flavor": "eager", "seam": "startup-deployment-prepare",
             "padding_bsz": 1, "unavailable": "CUDA is unavailable"}]


NO_GIT_IDENTITY = {"source": "none", "revision": "unknown", "dirty": None,
                   "reason": "no_git_metadata"}


def test_scheduler_records_the_config_before_the_deployment_is_ready():
    # The scheduler module without the server's uvicorn import chain.
    from tests.test_batch_in_progress_failure import bs

    async def run():
        recorded = []
        worker = SimpleNamespace(response_queue=queue.Queue())
        scheduler = bs.BatchScheduler(SimpleNamespace(), worker, SimpleNamespace())

        def record(cohort, published):
            # Nothing of this generation can be dispatched yet.
            assert not scheduler._deployment_ready.is_set()
            recorded.append((cohort, published))

        worker.record_code_identity = lambda cohort, identity: None
        worker.record_server_config = record
        await scheduler.start()
        try:
            base = {"type": "deployment_capacity", "max_model_len": 129,
                    "startup_transient": _transient(),
                    "code_identity": NO_GIT_IDENTITY, "cohort": 0}
            # A publication without the config never makes the deployment ready.
            worker.response_queue.put(dict(base))
            await asyncio.sleep(0.2)
            assert not scheduler._deployment_ready.is_set() and recorded == []
            published = config().as_dict()
            worker.response_queue.put(dict(base, cohort=3,
                                           server_config=published))
            await asyncio.wait_for(scheduler.wait_for_deployment(), timeout=5)
            assert recorded == [(3, published)]
        finally:
            await scheduler.stop()
            worker.response_queue.put(None)

    asyncio.run(run())


def test_worker_manager_logs_one_line_per_generation_and_node(caplog):
    # The class without the server's uvicorn import chain (AST-lifted).
    from tests.test_worker_pool_restart import _load_worker_manager_class

    WorkerManager = _load_worker_manager_class()
    fake = SimpleNamespace(
        args=SimpleNamespace(node_rank=3),
        _generation_of_epoch=WorkerManager._generation_of_epoch,
    )
    with caplog.at_level(logging.INFO):
        WorkerManager.record_server_config(fake, 2, config().as_dict())
    lines = [record.getMessage() for record in caplog.records
             if record.getMessage().startswith("[SERVER_CONFIG]")]
    # The cohort of epoch 2 serves generation 3, on this node.
    assert lines == [config().log_line(3, 3)]
    with pytest.raises(sc.ServerConfigError):
        WorkerManager.record_server_config(fake, 2, {"prefill_attn": "dp:streamed"})


# -- the worker-side snapshot ------------------------------------------------

def _graph_state(**overrides):
    state = SimpleNamespace(
        whole_model_graph=False, cuda_graph_manager=None, whole_model_bucketing=None
    )
    for name, value in overrides.items():
        setattr(state, name, value)
    return state


def _snapshot_worker(**overrides):
    from tests.worker_method_source import load_worker_methods

    class _Book:
        def reservation_total_capacity(self, length):
            assert length == 1024
            return 3200

    worker = SimpleNamespace(
        parallel_manager=SimpleNamespace(),  # no declaration
        engine_config=SimpleNamespace(
            Basic_Config=SimpleNamespace(enable_cuda_graphs=False)
        ),
        _graph=_graph_state(),
        gpu_paged_kv_cache_manager=SimpleNamespace(
            get_stats=lambda: SimpleNamespace(num_total_pages=3440)
        ),
        gpu_kv_cache_size_gb=52.5,
        global_host_kv_cache_size_gb=860,
        _get_host_kv_total_pages=lambda: 56320,
        gpu_memory_frac=0.9,
        model_context_length=1024,
        _traj=SimpleNamespace(trajectory_book=_Book()),
        server_config=None,
    )
    for name, value in overrides.items():
        setattr(worker, name, value)
    snapshot = load_worker_methods(["_snapshot_server_config"], {})[
        "_snapshot_server_config"
    ]
    return worker, snapshot


def test_snapshot_reads_the_owners_and_reports_undeclared_models():
    worker, snapshot = _snapshot_worker()
    snapshot(worker)
    assert sc.ServerConfig.from_dict(worker.server_config) == config(
        prefill_attn="undeclared", prefill_moe="undeclared",
        decode_attn="undeclared", decode_moe="undeclared",
        cuda_graphs=False, graph_buckets=None,
        host_kv_gib=860.0, host_kv_pages=56320,
        device_kv_gib=52.5, device_kv_pages=3440,
        gpu_memory_frac=0.9, max_pool=3200,
    )


def test_snapshot_reports_the_declaration_and_the_captured_ladder():
    from batchgen.parallel_declaration import ModuleParallelism, ParallelDeclaration

    declared = ParallelDeclaration(
        prefill_attention=ModuleParallelism("dp", "streamed"),
        prefill_moe=ModuleParallelism("dp", "streamed"),
        decode_attention=ModuleParallelism("dp", "resident"),
        decode_moe=ModuleParallelism("ep", "partial", group_size=8),
    )
    worker, snapshot = _snapshot_worker(
        parallel_manager=SimpleNamespace(parallel_declaration=lambda: declared),
        engine_config=SimpleNamespace(
            Basic_Config=SimpleNamespace(enable_cuda_graphs=True)
        ),
        _graph=_graph_state(
            whole_model_graph=True,
            cuda_graph_manager=object(),
            whole_model_bucketing=SimpleNamespace(bucket_sizes=[1, 2, 4, 8]),
        ),
    )
    snapshot(worker)
    assert sc.ServerConfig.from_dict(worker.server_config) == config()


def test_snapshot_refuses_to_run_after_the_pool_is_retired():
    worker, snapshot = _snapshot_worker(gpu_paged_kv_cache_manager=None)
    with pytest.raises(RuntimeError, match="device-KV pool"):
        snapshot(worker)
    assert worker.server_config is None
