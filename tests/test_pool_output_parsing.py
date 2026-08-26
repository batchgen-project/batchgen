"""Pool-mode output must honor the same parser flags as legacy batch mode."""

import json
from types import SimpleNamespace

from batchgen.server.batch_scheduler import BatchScheduler


def test_pool_completion_separates_k3_reasoning(tmp_path):
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
