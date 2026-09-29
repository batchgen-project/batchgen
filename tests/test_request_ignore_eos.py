"""Per-request ignore_eos: request model, pool intake entry, worker EOS marking."""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

from batchgen.server.batch_scheduler import BatchScheduler
from batchgen.server.io_struct import ChatCompletionRequest, CompletionRequest
from batchgen.worker.completion import CompletionContext, CompletionHandler

WORKER = Path(__file__).resolve().parents[1] / "batchgen" / "batchgen_worker.py"


def _chat(ignore_eos=None):
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    if ignore_eos is not None:
        body["ignore_eos"] = ignore_eos
    return ChatCompletionRequest(**body)


def test_request_models_accept_ignore_eos_and_default_to_false():
    assert _chat().ignore_eos is False
    assert _chat(True).ignore_eos is True
    assert CompletionRequest(model="m", prompt="hi").ignore_eos is False
    assert CompletionRequest(model="m", prompt="hi", ignore_eos=True).ignore_eos is True


def test_pool_intake_entry_carries_each_request_flag():
    submitted = []
    scheduler = object.__new__(BatchScheduler)
    scheduler._scheduling_pool = SimpleNamespace(register_batch=lambda **kw: None)
    scheduler._intake_pool = SimpleNamespace(
        submit_batch=lambda batch_id, entries, priority: submitted.extend(entries) or True,
        size=lambda: len(submitted),
        max_capacity=10,
    )
    scheduler.storage = SimpleNamespace()
    scheduler.server_args = SimpleNamespace(incremental_output_dir=None)
    scheduler._pool_request_meta = {}
    url = SimpleNamespace(value="/v1/chat/completions")
    requests = [
        SimpleNamespace(custom_id="a", url=url, body=_chat(True)),
        SimpleNamespace(custom_id="b", url=url, body=_chat()),
    ]

    asyncio.run(
        scheduler._process_batch_pool_mode(
            "batch-1",
            SimpleNamespace(batchgen_debug=None, max_context_length=None),
            requests,
            prompts=["p0", "p1"],
            per_request_max_tokens=[8, 8],
            sampling_params=[{}, {}],
            incremental_kwargs={},
        )
    )

    assert [e.raw_request["ignore_eos"] for e in submitted] == [True, False]


def test_eos_decision_respects_the_sequence_flag():
    ctx = CompletionContext(
        eos_token_ids=frozenset({2}), model_context_length=64, rank=0
    )
    assert CompletionHandler.should_stop_at_eos(ctx, 2) is True
    assert CompletionHandler.should_stop_at_eos(ctx, 2, True) is False
    assert CompletionHandler.should_stop_at_eos(ctx, 7, False) is False


def _worker_calls(name):
    tree = ast.parse(WORKER.read_text())
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
    ]


def test_every_worker_eos_marking_passes_the_sequence():
    calls = _worker_calls("_should_stop_at_eos")
    assert calls, "no EOS marking sites found"
    for call in calls:
        assert len(call.args) == 2 and isinstance(call.args[1], ast.Name), (
            f"line {call.lineno}: _should_stop_at_eos must receive the sequence"
        )


def test_admission_sets_the_sequence_flag():
    source = WORKER.read_text()
    assert 'seq.ignore_eos = bool(entry.get("ignore_eos", False))' in source
