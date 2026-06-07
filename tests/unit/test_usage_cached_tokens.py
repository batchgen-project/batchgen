import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(monkeypatch, module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _load_lightweight_usage_modules(monkeypatch):
    batchgen_pkg = types.ModuleType("batchgen")
    batchgen_pkg.__path__ = [str(REPO_ROOT / "batchgen")]
    server_pkg = types.ModuleType("batchgen.server")
    server_pkg.__path__ = [str(REPO_ROOT / "batchgen" / "server")]
    monkeypatch.setitem(sys.modules, "batchgen", batchgen_pkg)
    monkeypatch.setitem(sys.modules, "batchgen.server", server_pkg)

    io_struct = _load_module(
        monkeypatch,
        "batchgen.server.io_struct",
        REPO_ROOT / "batchgen" / "server" / "io_struct.py",
    )
    usage = _load_module(
        monkeypatch,
        "batchgen.server.usage",
        REPO_ROOT / "batchgen" / "server" / "usage.py",
    )
    return io_struct, usage


def _stub_batch_scheduler_deps(monkeypatch):
    dependencies = {
        "batchgen.server.intake_pool": {
            "IntakeEntry": object,
            "IntakePool": object,
            "Priority": object,
        },
        "batchgen.server.scheduling_pool": {"SchedulingPool": object},
        "batchgen.server.server_args": {"ServerArgs": object},
        "batchgen.server.storage": {"StorageManager": object},
        "batchgen.server.worker_manager": {"WorkerManager": object},
    }
    for module_name, attrs in dependencies.items():
        module = types.ModuleType(module_name)
        for attr_name, attr_value in attrs.items():
            setattr(module, attr_name, attr_value)
        monkeypatch.setitem(sys.modules, module_name, module)


def _load_batch_scheduler(monkeypatch):
    _load_lightweight_usage_modules(monkeypatch)
    _stub_batch_scheduler_deps(monkeypatch)
    return _load_module(
        monkeypatch,
        "batchgen.server.batch_scheduler",
        REPO_ROOT / "batchgen" / "server" / "batch_scheduler.py",
    )


def _model_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def test_usage_serializes_prompt_cached_tokens(monkeypatch):
    _, usage_module = _load_lightweight_usage_modules(monkeypatch)

    usage = usage_module.build_usage(
        prompt_tokens=128,
        completion_tokens=16,
        cached_tokens=64,
    )

    usage_dict = _model_dict(usage)
    assert usage_dict["prompt_tokens_details"] == {"cached_tokens": 64}
    assert usage_dict["total_tokens"] == 144


def test_usage_clamps_cached_tokens_to_prompt_tokens(monkeypatch):
    _, usage_module = _load_lightweight_usage_modules(monkeypatch)

    usage = usage_module.build_usage(
        prompt_tokens=32,
        completion_tokens=4,
        cached_tokens=128,
    )

    assert usage.prompt_tokens_details.cached_tokens == 32


def test_pool_completion_writes_cached_tokens(tmp_path, monkeypatch):
    batch_scheduler = _load_batch_scheduler(monkeypatch)
    scheduler = batch_scheduler.BatchScheduler.__new__(
        batch_scheduler.BatchScheduler
    )
    scheduler.server_args = SimpleNamespace(
        incremental_output_dir=str(tmp_path)
    )
    scheduler._pool_request_meta = {
        "batch_1": {
            "req_1": {
                "custom_id": "custom-1",
                "model": "openai/gpt-oss-120b",
                "url": "/v1/chat/completions",
            }
        }
    }

    scheduler._write_pool_completion(
        "batch_1",
        "req_1",
        {
            "text": "answer",
            "prompt_length": 128,
            "decoded_length": 8,
            "cached_tokens": 64,
            "finish_reason": "stop",
        },
    )

    output_path = tmp_path / "batch_1.jsonl"
    line = json.loads(output_path.read_text().strip())
    usage = line["response"]["body"]["usage"]

    assert usage["prompt_tokens"] == 128
    assert usage["completion_tokens"] == 8
    assert usage["total_tokens"] == 136
    assert usage["prompt_tokens_details"] == {"cached_tokens": 64}


def test_batch_output_metrics_summarize_cached_tokens(tmp_path, monkeypatch):
    batch_scheduler = _load_batch_scheduler(monkeypatch)
    output_path = tmp_path / "batch.jsonl"
    rows = [
        {
            "custom_id": "req-1",
            "response": {
                "status_code": 200,
                "body": {
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 10,
                        "total_tokens": 110,
                        "prompt_tokens_details": {"cached_tokens": 40},
                    }
                },
            },
            "error": None,
        },
        {
            "custom_id": "req-2",
            "response": {
                "status_code": 200,
                "body": {
                    "usage": {
                        "prompt_tokens": 50,
                        "completion_tokens": 5,
                        "total_tokens": 55,
                        "prompt_tokens_details": {"cached_tokens": 0},
                    }
                },
            },
            "error": None,
        },
        {"custom_id": "req-3", "response": None, "error": {"message": "bad"}},
    ]
    output_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    metrics = batch_scheduler._summarize_batch_output_file(output_path)

    assert metrics.rows == 3
    assert metrics.errors == 1
    assert metrics.prompt_tokens == 150
    assert metrics.completion_tokens == 15
    assert metrics.total_tokens == 165
    assert metrics.cached_tokens == 40
    assert metrics.requests_with_cache == 1
    assert metrics.cache_hit_rate == 40 / 150
