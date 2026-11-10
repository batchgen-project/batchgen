"""A request whose prompt cannot be constructed fails alone.

The chat template (or request validation) can refuse one conversation, e.g.
an unknown role or a malformed tool call. That ValueError used to propagate
out of _convert_requests_to_worker_inputs and fail the WHOLE batch, so one
such request among 48 returned zero completions. Now the request gets the
Batch API error item and the rest of the batch reaches the worker.
"""
import asyncio
import json
from types import SimpleNamespace

from tests.test_batch_in_progress_failure import bs

MODEL = "moonshotai/Kimi-K3"


def _encode(text):
    return [ord(ch) for ch in text]


class _Tokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        assert tokenize is True
        content = messages[-1]["content"]
        if "unfaithful" in content:
            raise ValueError(
                "Kimi-K3: this conversation cannot be rendered faithfully"
            )
        return _encode(f"<rendered>{content}")


def _chat_line(custom_id, content):
    return json.dumps({
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 8,
        },
    })


def _scheduler(tmp_path, lines):
    input_id = "file-input"
    (tmp_path / input_id).write_bytes("\n".join(lines).encode())
    updates = []
    scheduler = bs.BatchScheduler.__new__(bs.BatchScheduler)
    scheduler.server_args = SimpleNamespace(
        model=MODEL, incremental_output_dir=str(tmp_path / "incr"),
    )
    scheduler._scheduling_pool = bs.SchedulingPool(capacity=8)
    scheduler._intake_pool = bs.IntakePool(max_capacity=16)
    scheduler._pool_request_meta = {}
    scheduler._get_tokenizer = lambda model: _Tokenizer()
    scheduler.storage = SimpleNamespace(
        files_dir=tmp_path,
        output_dir=tmp_path,
        load_batch=lambda batch_id: SimpleNamespace(
            status=bs.BatchStatus.VALIDATING,
            input_file_id=input_id,
            max_decoding_length=None,
            max_context_length=None,
            batchgen_debug=None,
            temperature=None,
            top_p=None,
            top_k=None,
        ),
        load_metadata=lambda file_id: {"id": file_id},
        update_batch_status=lambda batch_id, status, **kw: updates.append(
            (status, kw.get("error"))),
    )
    finalized = []

    async def record_finalize(batch_id, requests):
        finalized.append(batch_id)

    scheduler._wait_and_finalize_batch = record_finalize
    return scheduler, updates, finalized


def _items(tmp_path, batch_id):
    path = tmp_path / "incr" / f"{batch_id}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_an_unrenderable_conversation_fails_only_its_own_request(tmp_path):
    scheduler, updates, finalized = _scheduler(tmp_path, [
        _chat_line("ok-1", "hello"),
        _chat_line("bad", "an unfaithful conversation"),
        _chat_line("ok-2", "world"),
    ])

    asyncio.run(scheduler._process_batch("batch-1"))

    assert [status for status, _ in updates] == [bs.BatchStatus.IN_PROGRESS]
    queued = [entry.raw_request["input_ids"].tolist()
              for entry in scheduler._intake_pool._normal]
    assert queued == [_encode("<rendered>hello"), _encode("<rendered>world")]

    (item,) = _items(tmp_path, "batch-1")
    assert item["custom_id"] == "bad"
    assert item["response"] is None
    assert item["error"]["code"] == "invalid_prompt"
    assert "cannot be rendered faithfully" in item["error"]["message"]
    assert "'bad'" in item["error"]["message"]

    tracker = scheduler._scheduling_pool.get_batch_tracker("batch-1")
    assert tracker.total_requests == 3
    assert tracker.completed_requests == 1
    assert not tracker.is_complete
    assert finalized == ["batch-1"]


def test_a_batch_of_only_unrenderable_requests_completes_with_errors(tmp_path):
    scheduler, updates, finalized = _scheduler(tmp_path, [
        _chat_line("bad-1", "unfaithful a"),
        _chat_line("bad-2", "unfaithful b"),
    ])

    asyncio.run(scheduler._process_batch("batch-2"))

    assert [status for status, _ in updates] == [bs.BatchStatus.IN_PROGRESS]
    assert scheduler._intake_pool.size() == 0
    assert sorted(i["custom_id"] for i in _items(tmp_path, "batch-2")) == [
        "bad-1", "bad-2"]
    assert scheduler._scheduling_pool.get_batch_tracker("batch-2").is_complete
    assert finalized == ["batch-2"]


def test_a_rejected_request_does_not_need_a_token_limit(tmp_path):
    """The missing-max_tokens check covers the requests that will run."""
    lines = [_chat_line("ok", "hello"), json.dumps({
        "custom_id": "bad-no-limit", "method": "POST", "url": "/v1/chat/completions",
        "body": {"model": MODEL, "messages": [{"role": "user", "content": "unfaithful"}]},
    })]
    scheduler, updates, _ = _scheduler(tmp_path, lines)

    asyncio.run(scheduler._process_batch("batch-3"))

    assert [status for status, _ in updates] == [bs.BatchStatus.IN_PROGRESS]
    assert [i["custom_id"] for i in _items(tmp_path, "batch-3")] == ["bad-no-limit"]
