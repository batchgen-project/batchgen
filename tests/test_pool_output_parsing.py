"""Pool-mode output must honor the same parser flags as legacy batch mode."""

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


def _isolated_write_pool_completion():
    """Compile the writer without importing the GPU/JIT-heavy server stack."""
    tree = ast.parse(SCHEDULER.read_text(), filename=str(SCHEDULER))
    scheduler = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchScheduler"
    )
    method = copy.deepcopy(next(
        node
        for node in scheduler.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_write_pool_completion"
    ))
    module = ast.Module(
        body=[ast.ClassDef(
            name="IsolatedBatchScheduler",
            bases=[],
            keywords=[],
            body=[method],
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
