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
        num_layers=78,
        num_layers_full=78,
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
        "gpu_memory_frac=0.9 max_pool=3200 num_layers=78 num_layers_full=78"
    )
    eager = config(cuda_graphs=False, graph_buckets=None, decode_moe="undeclared")
    assert eager.log_line(2, 3) == (
        "[SERVER_CONFIG] generation=2 node=3 prefill_attn=dp:streamed "
        "prefill_moe=dp:streamed decode_attn=dp:resident decode_moe=undeclared "
        "cuda_graphs=off graph_buckets=none "
        "host_kv_gib=860.000 host_kv_pages=56320 "
        "device_kv_gib=52.500 device_kv_pages=3440 "
        "gpu_memory_frac=0.9 max_pool=3200 num_layers=78 num_layers_full=78"
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
    dict(num_layers=0),                    # a served model has layers
    dict(num_layers_full=0),
    dict(num_layers=2.5),
    dict(num_layers=True),
    dict(num_layers=79),                   # truncation only serves a prefix
])
def test_invalid_values_are_refused(overrides):
    with pytest.raises(sc.ServerConfigError):
        config(**overrides)


def test_truncated_layer_counts_are_legal_and_in_the_line():
    truncated = config(num_layers=12)
    assert truncated.log_line(1, 0).endswith("num_layers=12 num_layers_full=78")
    assert sc.ServerConfig.from_dict(truncated.as_dict()) == truncated


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
        world_size=8,
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
        model_config=SimpleNamespace(num_hidden_layers=78),
        loaded_model_config=SimpleNamespace(num_hidden_layers_full=78),
        server_config=None,
    )
    for name, value in overrides.items():
        setattr(worker, name, value)
    methods = load_worker_methods(
        ["_snapshot_server_config", "_resolve_server_layout"], {}
    )
    if "_server_layout" not in overrides:
        # What worker construction does right after it builds the manager.
        methods["_resolve_server_layout"](worker)
    return worker, methods["_snapshot_server_config"]


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
        parallel_manager=SimpleNamespace(
            parallel_declaration=lambda graph_state=None: ParallelDeclaration(
                prefill_attention=declared.prefill_attention,
                prefill_moe=declared.prefill_moe,
                decode_attention=declared.decode_attention,
                decode_moe=declared.decode_moe,
                **({} if graph_state is None else {"decode_graph": graph_state}),
            )
        ),
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


def test_snapshot_uses_the_manager_graph_declaration_over_worker_graph_fields():
    from batchgen.parallel_declaration import (
        DecodeGraphDeclaration,
        ModuleParallelism,
        ParallelDeclaration,
    )

    declared = ParallelDeclaration(
        prefill_attention=ModuleParallelism("dp", "streamed"),
        prefill_moe=ModuleParallelism("dp", "streamed"),
        decode_attention=ModuleParallelism("dp", "resident"),
        decode_moe=ModuleParallelism("ep", "resident", group_size=8),
        decode_graph=DecodeGraphDeclaration.off(),
    )
    worker, snapshot = _snapshot_worker(
        parallel_manager=SimpleNamespace(
            parallel_declaration=lambda graph_state=None: declared
        ),
        _graph=_graph_state(
            whole_model_graph=True,
            cuda_graph_manager=object(),
            whole_model_bucketing=SimpleNamespace(bucket_sizes=[1, 2, 4]),
        ),
    )
    snapshot(worker)
    assert worker.server_config["cuda_graphs"] is False
    assert worker.server_config["graph_buckets"] is None


def test_snapshot_reports_truncated_and_full_layer_counts():
    # Under --debug-num-layers the engine ModelConfig holds the served count
    # and the rich config carries the checkpoint's stamped full count.
    worker, snapshot = _snapshot_worker(
        model_config=SimpleNamespace(num_hidden_layers=12),
        loaded_model_config=SimpleNamespace(num_hidden_layers_full=78),
    )
    snapshot(worker)
    assert worker.server_config["num_layers"] == 12
    assert worker.server_config["num_layers_full"] == 78


def test_snapshot_defaults_full_count_for_models_without_the_stamp():
    # An initializer that predates apply_layer_truncation never stamps the
    # rich config; it serves all layers, so full == effective.
    worker, snapshot = _snapshot_worker(
        model_config=SimpleNamespace(num_hidden_layers=36),
        loaded_model_config=SimpleNamespace(),
    )
    snapshot(worker)
    assert worker.server_config["num_layers"] == 36
    assert worker.server_config["num_layers_full"] == 36


def test_snapshot_refuses_to_run_after_the_pool_is_retired():
    worker, snapshot = _snapshot_worker(gpu_paged_kv_cache_manager=None)
    with pytest.raises(RuntimeError, match="device-KV pool"):
        snapshot(worker)
    assert worker.server_config is None


def _layout_manager(prefill_attention, attn_tp_size=1):
    from batchgen.parallel_declaration import (
        ModuleParallelism,
        ParallelDeclaration,
        replicated_prefill_world_ep_decode,
    )

    return SimpleNamespace(
        attn_tp_size=attn_tp_size,
        SUPPORTED_LAYOUTS=replicated_prefill_world_ep_decode(),
        parallel_declaration=lambda graph_state=None: ParallelDeclaration(
            prefill_attention=prefill_attention,
            prefill_moe=ModuleParallelism("dp", "streamed"),
            decode_attention=ModuleParallelism("dp", "resident"),
            decode_moe=ModuleParallelism("ep", "resident", group_size=8),
            **({} if graph_state is None else {"decode_graph": graph_state}),
        ),
    )


def test_the_layout_is_resolved_at_construction_and_the_report_is_rendered_from_it():
    from batchgen.parallel_declaration import ModuleParallelism

    worker, snapshot = _snapshot_worker(
        parallel_manager=_layout_manager(ModuleParallelism("dp", "streamed"))
    )
    layout = worker._server_layout
    assert layout.decode.moe_groups == (tuple(range(8)),)
    snapshot(worker)
    assert worker._server_layout is layout
    reported = sc.ServerConfig.from_dict(worker.server_config)
    assert (reported.prefill_attn, reported.decode_moe) == (
        layout.prefill.attention.render(), layout.decode.moe.render()
    )


def test_an_unimplemented_layout_is_refused_at_construction_before_any_build():
    from batchgen.parallel_declaration import ModuleParallelism, UnsupportedLayoutError

    with pytest.raises(UnsupportedLayoutError, match="prefill attention=tp2:resident"):
        _snapshot_worker(
            parallel_manager=_layout_manager(
                ModuleParallelism("tp", "resident", group_size=2)
            )
        )


def test_the_snapshot_refuses_a_built_model_that_declares_another_layout():
    from batchgen.parallel_declaration import ModuleParallelism

    worker, snapshot = _snapshot_worker(
        parallel_manager=_layout_manager(ModuleParallelism("dp", "streamed"))
    )
    worker.parallel_manager = _layout_manager(ModuleParallelism("dp", "streamed"))
    worker.parallel_manager.parallel_declaration = (
        lambda graph_state=None, inner=worker.parallel_manager.parallel_declaration: (
            type(inner())(
                prefill_attention=inner().prefill_attention,
                prefill_moe=inner().prefill_moe,
                decode_attention=inner().decode_attention,
                decode_moe=ModuleParallelism("ep", "partial", group_size=8),
            )
        )
    )
    with pytest.raises(RuntimeError, match="differs from the layout resolved at construction"):
        snapshot(worker)
    assert worker.server_config is None


def test_a_manager_with_an_attention_group_must_declare_its_layout():
    from batchgen.parallel_declaration import ParallelDeclarationError

    with pytest.raises(ParallelDeclarationError, match="declares no layout"):
        _snapshot_worker(parallel_manager=SimpleNamespace(attn_tp_size=2))


def test_the_group_width_is_read_from_the_layout():
    from tests.worker_method_source import attention_group_layout, load_worker_methods

    width = load_worker_methods(["_decode_attn_tp_size"], {})["_decode_attn_tp_size"]
    assert width(SimpleNamespace(_server_layout=None)) == 1
    assert width(SimpleNamespace(_server_layout=attention_group_layout(4, 8))) == 4


def test_each_phase_reads_its_own_attention_group_width():
    from batchgen.parallel_declaration import ModuleParallelism, PhaseLayout, ServerLayout
    from tests.worker_method_source import load_worker_methods

    methods = load_worker_methods(["_decode_attn_tp_size", "_prefill_attn_tp_size"], {})
    none = ModuleParallelism("none", None)
    layout = ServerLayout(
        prefill=PhaseLayout("prefill", ModuleParallelism("tp", "resident", group_size=2), none, 8),
        decode=PhaseLayout("decode", ModuleParallelism("tp", "resident", group_size=4), none, 8),
    )
    worker = SimpleNamespace(_server_layout=layout)
    assert methods["_prefill_attn_tp_size"](worker) == 2
    assert methods["_decode_attn_tp_size"](worker) == 4
    assert methods["_prefill_attn_tp_size"](SimpleNamespace(_server_layout=None)) == 1
