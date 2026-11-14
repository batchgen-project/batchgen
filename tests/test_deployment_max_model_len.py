"""CPU checks of the deployment limit at the real worker and server seams."""

import ast
import asyncio
import gc
import json
import logging
import queue
import sys
import dataclasses
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from batchgen.config.max_model_len import (
    MaxModelLenCapacityError,
    SequenceLengthCapacity,
    resolve_max_model_len,
    verify_max_model_len,
)
from batchgen.worker import startup_transient
from batchgen.worker.kv_manager import KVCacheManager
from tests.test_kv_fixed_address_page_table import MANAGER
from tests.test_prompt_rejection_fails_one_request import (
    _chat_line, _items, _scheduler,
)
from tests.test_batch_in_progress_failure import bs
from tests.worker_method_source import load_worker_methods


ROOT = Path(__file__).resolve().parents[1]


def capacity(**kwargs):
    values = dict(
        resource="GPU KV", region="0", node=0, rank=0,
        page_size_tokens=64, bytes_per_page=1024, available_pages=4,
    )
    values.update(kwargs)
    return SequenceLengthCapacity(**values)


@pytest.mark.parametrize("requested,expected", [(None, 1024), (-1, 1024), (513, 513)])
def test_resolve_launch_length(requested, expected):
    assert resolve_max_model_len(requested, 1024) == expected


@pytest.mark.parametrize("requested", [-2, 0, 1, 1025])
def test_invalid_launch_length(requested):
    with pytest.raises(ValueError, match="max_model_len"):
        resolve_max_model_len(requested, 1024)


@pytest.mark.parametrize("resource", ["GPU KV", "host KV"])
def test_failure_names_the_limiting_node_region_and_exact_bytes(resource):
    short = capacity(resource=resource, region="auxiliary", node=1, rank=2, copies=2)
    with pytest.raises(MaxModelLenCapacityError) as info:
        verify_max_model_len(129, [capacity(available_pages=100), short])
    error = info.value
    assert (error.resource, error.region, error.node, error.rank) == (
        resource, "auxiliary", 1, 2)
    assert (error.required_pages, error.available_pages) == (6, 4)
    assert (error.required_bytes, error.available_bytes) == (6144, 4096)
    assert error.estimated_feasible_length == 128
    assert "max_model_len=129" in str(error)
    assert "node=1" in str(error) and "region=auxiliary" in str(error)


def test_auto_fit_takes_the_largest_length_that_fits_every_region():
    resources = [
        capacity(available_pages=100),
        capacity(resource="host KV", node=1, region="1", copies=2, available_pages=5),
        capacity(page_size_tokens=16, available_pages=9),
    ]
    assert verify_max_model_len(1024, resources, auto_fit=True) == 128
    assert verify_max_model_len(127, resources, auto_fit=True) == 127
    assert verify_max_model_len(128, resources) == 128
    with pytest.raises(MaxModelLenCapacityError):
        verify_max_model_len(129, resources)


def test_auto_fit_refuses_a_pool_that_cannot_hold_any_sequence():
    with pytest.raises(MaxModelLenCapacityError) as info:
        verify_max_model_len(1024, [capacity(available_pages=0)], auto_fit=True)
    assert info.value.estimated_feasible_length == 0


@pytest.mark.parametrize("page_size", [16, 64])
@pytest.mark.parametrize("prompt,decode", [(1, 1), (8192, 65536)])
def test_real_page_table_rows_use_length_not_batch_hints(page_size, prompt, decode):
    config = MANAGER.GPUPagedKVConfig(
        num_layers=1, num_pages=64, page_size_tokens=page_size,
        num_k_heads=1, k_head_dim=4, num_v_heads=0, v_head_dim=0,
        kv_dtype=torch.float32, max_model_len=129, cuda_graph_max_slots=2,
    )
    manager = MANAGER.GPUPagedKVCacheManager(config=config, device="cpu")
    manager._engine_config = SimpleNamespace(Basic_Config=SimpleNamespace(
        max_model_len=129, max_decoding_length=decode,
        get_max_prompt_length=lambda: prompt,
    ))
    manager.initialize()
    try:
        assert manager._resolve_page_table_max_pages_per_sequence() == -(-129 // page_size)
        # Force actual page-table storage creation on the CPU.
        manager.allocate_pages_for_sequences([0], [1])
        manager.rebuild_page_table([0])
        assert manager.get_cuda_graph_page_table().shape[1] == -(-129 // page_size)
    finally:
        manager.destroy()


def test_pool_input_matches_the_residency_formula():
    assert KVCacheManager.provisional_pool_gb(20, 8, 0.75) == 7
    assert KVCacheManager.provisional_pool_gb(20, 16, 0.75) == -1


def test_launch_limit_is_a_worker_argument():
    tree = ast.parse((ROOT / "batchgen/batchgen_worker.py").read_text())
    node = next(node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorkerArgs")
    scope = {"__name__": __name__, "dataclass": dataclasses.dataclass, "Optional": Optional}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "worker_args.py", "exec"), scope)
    cls = scope["BatchGenWorkerArgs"]
    required = {field.name: None for field in dataclasses.fields(cls)
                if field.default is dataclasses.MISSING}
    assert cls(**required, max_model_len=129).max_model_len == 129
    manager = ast.parse((ROOT / "batchgen/server/worker_manager.py").read_text())
    build = next(node for node in ast.walk(manager) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "BatchGenWorkerArgs")
    assert {kw.arg: ast.unparse(kw.value) for kw in build.keywords}["max_model_len"] == "self.args.max_model_len"


def test_glm_graph_and_prewarm_use_the_same_non_page_aligned_length(monkeypatch):
    from tests.unit.test_glm5_triton_prewarm import TP, _shapes

    monkeypatch.setattr(TP, "_aux_is_page_split", lambda *args: False)
    manager = SimpleNamespace(
        config=SimpleNamespace(max_model_len=129, page_size_tokens=64, kv_dtype=torch.bfloat16),
        _gpu_page_table_manager=SimpleNamespace(graph_max_pages_per_sequence=3, max_slots=2),
    )
    geometry, _, _ = TP.decode_geometry(
        SimpleNamespace(primary=manager, auxiliary=manager), _shapes(), [1, 2])
    score_capacity = load_worker_methods(["_glm5_dsa_graph_score_capacity_tokens"], {})[
        "_glm5_dsa_graph_score_capacity_tokens"]
    table = SimpleNamespace(shape=(2, 3))
    span = score_capacity(table, 64, table, 64, deployment_context_tokens=129)
    assert span == geometry.graph_max_seqlen == 129

    source = ast.parse((ROOT / "batchgen/models/glm/glm5/cuda_graph_segments.py").read_text())
    buffers = next(node for node in source.body
                   if isinstance(node, ast.FunctionDef) and node.name == "_make_scorer_buffers")
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[buffers], type_ignores=[]), "score_buffers.py", "exec"), scope)
    scores, *_ = scope["_make_scorer_buffers"](
        page_split=False, bucket_size=2, max_seqlen=span, num_index_heads=1,
        index_head_dim=4, max_aux_pages_per_seq=3, device=torch.device("cpu"))
    assert scores.shape == (2, 129)


def test_pool_is_verified_before_allocation_and_recreation_uses_cached_size():
    calls = []
    cuda = SimpleNamespace(
        synchronize=lambda device: None, empty_cache=lambda: None,
        mem_get_info=lambda rank: (12 * 1024 ** 3, 20 * 1024 ** 3),
    )

    def reduce_pool(tensor, op):
        assert op == "MIN"
        tensor.fill_(6)  # a peer has less space than this rank's 7 GiB

    init = load_worker_methods(["_init_gpu_kv_with_actual_size"], {
        "torch": SimpleNamespace(cuda=cuda, tensor=torch.tensor, float64=torch.float64),
        "dist": SimpleNamespace(all_reduce=reduce_pool, ReduceOp=SimpleNamespace(MIN="MIN")),
        "KVCacheManager": KVCacheManager, "logging": logging, "gc": gc,
        "GPU_KV_POOL_HEADROOM_GB": 2,
    })["_init_gpu_kv_with_actual_size"]
    worker = SimpleNamespace(
        gpu_paged_kv_cache_manager=None, _phase="decode", rank=0, local_rank=0,
        torch_device="cpu", _gpu_kv_cached_pool_gb=None, gpu_memory_frac=0.75,
        _verify_deployment_max_model_len=lambda size: calls.append(("verify", size)),
    )

    def allocate():
        calls.append(("allocate", worker.gpu_kv_cache_size_gb))
        worker.gpu_paged_kv_cache_manager = SimpleNamespace(
            is_initialized=True, get_stats=lambda: SimpleNamespace(num_total_pages=100))

    worker._initialize_gpu_kv_manager_fixed_size = allocate
    init(worker)
    assert calls == [("verify", 7), ("allocate", 6)]
    worker._gpu_kv_cached_pool_gb = worker.gpu_kv_cache_size_gb
    worker.gpu_paged_kv_cache_manager = None
    init(worker)
    assert calls[-1] == ("allocate", 6)
    assert len(calls) == 3

    # A failed verification never allocates the undersized pool.
    worker._gpu_kv_cached_pool_gb = None
    worker.gpu_paged_kv_cache_manager = None

    def refuse(size):
        raise MaxModelLenCapacityError(129, capacity(available_pages=1))

    worker._verify_deployment_max_model_len = refuse
    with pytest.raises(MaxModelLenCapacityError):
        init(worker)
    assert len(calls) == 3


@pytest.mark.parametrize("auto_fit", [False, True])
def test_worker_collects_remote_host_regions_before_accepting_length(monkeypatch, auto_fit):
    profile = SimpleNamespace(page_size=64, num_layers=2, bytes_per_page=lambda: 512)
    module = SimpleNamespace(
        _resolve_profile=lambda name: profile,
        _resolve_indexer_profile=lambda name: profile,
        is_dsa_model=lambda name: True,
    )
    monkeypatch.setitem(sys.modules, "batchgen.kv_cache.host_kv_mananger_config", module)
    gathered_local = []

    def gather(out, local):
        gathered_local.extend(local)
        out[0] = local
        out[1] = [capacity(
            resource="host KV", region="1", node=1, rank=2,
            copies=2, available_pages=5,
        )]

    method = load_worker_methods(["_verify_deployment_max_model_len"], {
        "SequenceLengthCapacity": SequenceLengthCapacity,
        "verify_max_model_len": verify_max_model_len,
        "QueryBook": SimpleNamespace(TOKEN_BYTES=4),
        "dist": SimpleNamespace(all_gather_object=gather),
    })["_verify_deployment_max_model_len"]
    worker = SimpleNamespace(
        huggingface_ckpt_name="test-model", rank=0, local_rank=0,
        local_world_size=2, world_size=2, model_context_length=1024,
        args=SimpleNamespace(max_model_len=-1 if auto_fit else 1024),
        engine_config=SimpleNamespace(Basic_Config=SimpleNamespace()),
        _host_views_by_group=lambda: {
            i: SimpleNamespace(get_stats=lambda: SimpleNamespace(num_total_pages=100))
            for i in (0, 1)
        },
        _decode_attn_tp_size=lambda: 2,
        _trajectory_book=SimpleNamespace(page_tokens=64, page_count=100),
    )
    if auto_fit:
        method(worker, 100 * 2048 / (1024 ** 3))
        assert worker.model_context_length == 128
        assert worker.engine_config.Basic_Config.max_model_len == 128
        # Once published, even auto-fit cannot silently lower L again.
        with pytest.raises(MaxModelLenCapacityError):
            method(worker, 2048 / (1024 ** 3))
    else:
        with pytest.raises(MaxModelLenCapacityError) as info:
            method(worker, 100 * 2048 / (1024 ** 3))
        assert (info.value.node, info.value.region) == (1, "1")
    assert [(r.region, r.available_pages) for r in gathered_local[:4:2]] == [
        ("0", 100), ("1", 100)]
    assert all(r.copies == 2 for r in gathered_local if r.resource == "host KV")


@pytest.mark.parametrize("persistent", [False, True])
def test_pre_readiness_sizing_freezes_the_pool_in_every_mode(persistent):
    events = []
    worker = SimpleNamespace(
        _startup_phase=lambda name: nullcontext(), _core_initialized=False,
        _gpu_kv_cached_pool_gb=None, _gpu_kv_sizing_provisional=True,
        model_context_length=129, comm=object(),
        Init=lambda *args: events.append("core"),
        _ensure_pynccl_communicator=lambda: events.append("comm"),
        _persistent_phase_enabled=lambda: persistent,
        _desired_phase_key=lambda phase: SimpleNamespace(prefill_moe_mode="streamed"),
        _max_decode_rank_bsz=lambda: 8,
        _load_decode_model=lambda *args, **kwargs: events.append("decode"),
        _transition_phase=lambda key: events.append("prefill"),
        set_phase=lambda phase: None,
        _install_prefill_weight_copy_pipeline=lambda flag: None,
        _probe_startup_transient=lambda **kwargs: events.append(
            ("probe", kwargs["seam"], kwargs["capture"])),
        # No retained decode instance / no requested graphs: no startup capture.
        _startup_decode_graph_capture=lambda key: None,
    )

    def size_pool():
        events.append("verify")
        worker.gpu_kv_cache_size_gb = 3
        worker._gpu_kv_cached_pool_gb = 3
        worker._gpu_kv_sizing_provisional = False

    worker._init_gpu_kv_with_actual_size = size_pool
    worker._build_persistent_phase_instances = size_pool
    prepare = load_worker_methods(["prepare_deployment"], {
        "logging": logging, "_startup_transient": startup_transient,
    })["prepare_deployment"]
    prepare(worker)
    assert worker._gpu_kv_cached_pool_gb == 3
    assert worker._gpu_kv_sizing_provisional is False
    assert events == (["core", "comm", "verify"] if persistent else [
        "core", "comm", "decode", "verify",
        ("probe", startup_transient.SEAM_DEPLOYMENT_PREPARE, None), "prefill"])


@pytest.mark.parametrize("capture", [False, True])
def test_persistent_builder_checks_the_provisional_pool_before_capture(capture):
    from batchgen.phase_lifecycle import PhaseKey

    events = []
    methods = load_worker_methods([
        "_build_persistent_phase_instances", "_capture_startup_decode_graphs",
        "_decode_graph_state_present",
    ], {
        "PhaseKey": PhaseKey, "logging": logging, "_startup_transient": startup_transient,
        "time": SimpleNamespace(perf_counter=lambda: 0),
    })
    build = methods["_build_persistent_phase_instances"]

    def captured_graphs():
        events.append("capture")
        worker._decode_graph_captures += 1
        worker._cuda_graph_manager = object()
    worker = SimpleNamespace(
        rank=0, comm=None, parallel_manager=SimpleNamespace(decode_instance=None),
        _transition_phase=lambda key, **kwargs: (events.append(key.phase) or True, False),
        set_phase=lambda phase: None, _install_prefill_weight_copy_pipeline=lambda _: None,
        _desired_phase_key=lambda phase: PhaseKey(phase, "streamed", ()),
        _max_decode_rank_bsz=lambda: 8,
        _check_decode_width_within_graph_buckets=lambda width: events.append(
            ("width_check", width)),
        _assert_phase_switch_clean=lambda *args, **kwargs: None,
        _install_decode_weight_copy_queue=lambda: None,
        _init_gpu_kv_with_actual_size=lambda: events.append("verify_pool"),
        _glm5_whole_model_graph_requested_for_current_batch=lambda: capture,
        _sync_decode_moe_rank_counts=lambda *args, **kwargs: None,
        _bind_decode_attention_metadata_for_graph_config=lambda _: None,
        _warmup_cuda_graphs=captured_graphs,
        _decode_graph_captures=0, _cuda_graph_manager=None,
    )
    worker._capture_startup_decode_graphs = (
        lambda: methods["_capture_startup_decode_graphs"](worker))
    worker._decode_graph_state_present = (
        lambda: methods["_decode_graph_state_present"](worker))

    def probe(*, seam, capture):
        assert seam == startup_transient.SEAM_PERSISTENT_BUILD
        events.append("probe")
        if capture is not None:
            capture()

    worker._probe_startup_transient = probe
    build(worker)
    assert events == ["prefill", ("width_check", 8), "decode", "verify_pool", "probe"] + (
        ["capture"] if capture else []) + ["prefill"]
    assert worker._decode_padding_bsz == 8

    def refuse():
        raise MaxModelLenCapacityError(129, capacity(available_pages=1))

    events.clear()
    worker._init_gpu_kv_with_actual_size = refuse
    with pytest.raises(MaxModelLenCapacityError):
        build(worker)
    assert events == ["prefill", ("width_check", 8), "decode"]


def test_prompt_at_or_above_limit_fails_alone_and_omission_gets_remaining_budget(tmp_path):
    lines = [_chat_line("short", "hi"), _chat_line("equal", "abcdef"), _chat_line("long", "abcdefg")]
    short = json.loads(lines[0])
    del short["body"]["max_tokens"]
    lines[0] = json.dumps(short)
    scheduler, updates, finalized = _scheduler(tmp_path, lines)
    scheduler.max_model_len = len("<rendered>abcdef")
    asyncio.run(scheduler._process_batch("length-batch"))
    assert [s for s, _ in updates] == [bs.BatchStatus.IN_PROGRESS]
    (entry,) = list(scheduler._intake_pool._normal)
    assert entry.raw_request["max_tokens"] == 4
    assert entry.raw_request["input_ids"].dtype.name == "int32"
    errors = _items(tmp_path, "length-batch")
    assert {item["custom_id"] for item in errors} == {"equal", "long"}
    assert all(item["error"]["status_code"] == 400 for item in errors)
    assert all("max_model_len=16" in item["error"]["message"] for item in errors)
    tracker = scheduler._scheduling_pool.get_batch_tracker("length-batch")
    assert tracker.completed_requests == 2 and not tracker.is_complete
    assert finalized == ["length-batch"]


def published_transient():
    """The struct the worker publishes when neither flavor is measurable."""
    return [
        startup_transient.StartupTransient.missing(
            flavor, startup_transient.SEAM_DEPLOYMENT_PREPARE, 8, "CUDA is unavailable",
        ).as_dict()
        for flavor in startup_transient.FLAVORS
    ]


def _models_app(scheduler):
    """Compile the real endpoint functions, avoiding the CUDA server imports."""
    pytest.importorskip("fastapi")
    from fastapi import FastAPI, HTTPException, Request
    from tests.unit.test_models_api import ModelObject, ListModelsResponse

    app = FastAPI()
    app.state.scheduler = scheduler
    app.state.model_object = ModelObject(id="test-model", created=1, max_context_length=99999)
    tree = ast.parse((ROOT / "batchgen/server/http_server.py").read_text())
    names = {"published_model_object", "list_models", "retrieve_model"}
    nodes = {node.name: node for node in ast.walk(tree)
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names}
    scope = dict(app=app, Request=Request, HTTPException=HTTPException,
                 ModelObject=ModelObject, ListModelsResponse=ListModelsResponse)
    for name in ("published_model_object", "list_models", "retrieve_model"):
        exec(compile(ast.Module(body=[nodes[name]], type_ignores=[]), "http_server.py", "exec"), scope)
    return app


def test_real_models_endpoints_read_the_published_length_after_restart(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    async def run():
        worker = SimpleNamespace(response_queue=queue.Queue())
        scheduler = bs.BatchScheduler(SimpleNamespace(), worker, SimpleNamespace())
        await scheduler.start()
        try:
            worker.response_queue.put({"type": "deployment_capacity", "max_model_len": 129,
                                        "startup_transient": published_transient()})
            await scheduler.wait_for_deployment()
            with TestClient(_models_app(scheduler)) as client:
                for path in ("/v1/models", "/v1/models/test-model"):
                    response = client.get(path)
                    assert response.status_code == 200
                    obj = response.json()
                    if "data" in obj:
                        obj = obj["data"][0]
                    assert obj["max_model_len"] == obj["max_context_length"] == 129
                    assert obj["startup_transient"] == published_transient()
                scheduler._clear_deployment()
                assert client.get("/v1/models").status_code == 503
                worker.response_queue.put({"type": "deployment_capacity", "max_model_len": 65,
                                        "startup_transient": published_transient()})
                await scheduler.wait_for_deployment()
                assert client.get("/v1/models/test-model").json()["max_model_len"] == 65
                assert client.get("/v1/models/missing").status_code == 404
        finally:
            await scheduler.stop()
            worker.response_queue.put(None)

    asyncio.run(run())


def test_worker_verifies_and_publishes_before_signalling_ready():
    tree = ast.parse((ROOT / "batchgen/server_worker_main_loop.py").read_text())
    prepare = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Attribute) and node.func.attr == "prepare_deployment"]
    ready = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "_signal_local_worker_manager_ready"]
    publish = [node for node in ast.walk(tree) if isinstance(node, ast.Constant)
               and node.value == "deployment_capacity"]
    assert len(prepare) == len(ready) == len(publish) == 1
    assert prepare[0].lineno < publish[0].lineno < ready[0].lineno
