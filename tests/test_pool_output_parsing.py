"""Pool-mode output must honor the same parser flags as legacy batch mode.

It must also keep a per-request failure a FAILURE: a worker that fails one
request (over-context prompt, a reservation no Host pool can ever hold)
publishes an ``error`` on the completion, and the batch output line has to be
the Batch API error item, not a 200 with empty text.
"""

import ast
import copy
import json
import logging
import time
import uuid
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = ROOT / "batchgen" / "server" / "batch_scheduler.py"


WRITER_METHODS = (
    "_write_pool_completion", "_append_pool_result_item", "_chat_completion_body",
)


def _isolated_write_pool_completion():
    """Compile the writer without importing the GPU/JIT-heavy server stack."""
    tree = ast.parse(SCHEDULER.read_text(), filename=str(SCHEDULER))
    scheduler = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchScheduler"
    )
    methods = [
        copy.deepcopy(node)
        for node in scheduler.body
        if isinstance(node, ast.FunctionDef) and node.name in WRITER_METHODS
    ]
    assert {node.name for node in methods} == set(WRITER_METHODS)
    module = ast.Module(
        body=[ast.ClassDef(
            name="IsolatedBatchScheduler",
            bases=[],
            keywords=[],
            body=methods,
            decorator_list=[],
        )],
        type_ignores=[],
    )
    namespace = {
        "json": json,
        "logger": logging.getLogger(__name__),
        "time": time,
        "uuid": uuid,
    }
    exec(
        compile(ast.fix_missing_locations(module), str(SCHEDULER), "exec"),
        namespace,
    )
    return namespace["IsolatedBatchScheduler"]


def test_pool_completion_separates_k3_reasoning(tmp_path):
    BatchScheduler = _isolated_write_pool_completion()
    scheduler = BatchScheduler.__new__(BatchScheduler)
    scheduler.server_args = SimpleNamespace(
        incremental_output_dir=str(tmp_path),
    )
    scheduler._pool_request_meta = {
        "batch-1": {
            "request-1": {
                "custom_id": "request-1",
                "url": "/v1/chat/completions",
                "model": "moonshotai/Kimi-K3",
                "prompt_text": "question",
            }
        }
    }
    raw = (
        "private reasoning<|close|>think<|sep|>"
        "<|open|>response<|sep|>The answer is (D)."
        "<|close|>response<|sep|><|close|>message<|sep|><|end_of_msg|>"
    )
    scheduler._parse_output = lambda model, text: (
        "The answer is (D).",
        "private reasoning",
        None,
    )

    scheduler._write_pool_completion(
        "batch-1",
        "request-1",
        {
            "text": raw,
            "prompt_length": 10,
            "decoded_length": 20,
            "finish_reason": "stop",
        },
    )

    record = json.loads((tmp_path / "batch-1.jsonl").read_text())
    message = record["response"]["body"]["choices"][0]["message"]
    assert message["content"] == "The answer is (D)."
    assert message["reasoning_content"] == "private reasoning"
    assert "<|" not in message["content"]
    assert record["error"] is None


def _error_scheduler(tmp_path):
    BatchScheduler = _isolated_write_pool_completion()
    scheduler = BatchScheduler.__new__(BatchScheduler)
    scheduler.server_args = SimpleNamespace(
        incremental_output_dir=str(tmp_path),
    )
    scheduler._pool_request_meta = {
        "batch-1": {
            "request-1": {
                "custom_id": "custom-1",
                "url": "/v1/chat/completions",
                "model": "moonshotai/Kimi-K3",
            }
        }
    }
    return scheduler


def test_a_failed_request_is_written_as_a_batch_error_item(tmp_path):
    scheduler = _error_scheduler(tmp_path)

    scheduler._write_pool_completion(
        "batch-1",
        "request-1",
        {
            "text": "",
            "error": {
                "code": "host_kv_capacity_exceeded",
                "message": "Prompt needs 900 Host KV pages, more than the "
                           "whole node pool (512 pages)",
            },
        },
    )

    record = json.loads((tmp_path / "batch-1.jsonl").read_text())
    assert record["custom_id"] == "custom-1"
    # The Batch API shape the e2e rejection test asserts: no response at all.
    assert record["response"] is None
    assert record["error"]["code"] == "host_kv_capacity_exceeded"
    assert "512 pages" in record["error"]["message"]


def test_an_error_without_a_code_still_fails_the_request(tmp_path):
    scheduler = _error_scheduler(tmp_path)

    scheduler._write_pool_completion(
        "batch-1", "request-1", {"error": {"message": "boom"}},
    )

    record = json.loads((tmp_path / "batch-1.jsonl").read_text())
    assert record["response"] is None
    assert record["error"] == {"code": "worker_error", "message": "boom"}
