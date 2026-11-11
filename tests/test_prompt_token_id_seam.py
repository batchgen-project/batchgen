"""Token ids are the only prompt representation crossing the worker seam.

The scheduler builds each prompt's ids (chat: ``apply_chat_template(
tokenize=True)``; completion: the encode call the worker used to make) and the
admission message carries them as ``input_ids``. Rendering a chat template to
text and re-encoding it is not the identity for a template defined on token
segments (Kimi-K3), so a text seam either serves ids the template never
defined or refuses valid conversations.

batch_scheduler is loaded by path with the real io_struct / IntakePool /
SchedulingPool and stubs for the engine-backed modules; the worker's admission
methods are compiled from the real source (tests/worker_method_source.py).
"""
import asyncio
import importlib.util
import json
import logging
import pickle
import queue
import random
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import pytest
import torch

from batchgen.query_book import QueryBook
from batchgen.sequence import SequenceBatch, SequenceEntry
from tests.worker.worker_fakes import attach_prefix_state, attach_trajectory_state
from tests.worker_method_source import load_worker_functions, load_worker_methods

SERVER = Path(__file__).resolve().parents[1] / "batchgen" / "server"


def _load_server_modules():
    names = ["batchgen.server"] + [
        f"batchgen.server.{m}" for m in (
            "io_struct", "intake_pool", "scheduling_pool", "server_args",
            "storage", "worker_manager", "batch_scheduler")]
    saved = {name: sys.modules.get(name) for name in names}
    try:
        package = types.ModuleType("batchgen.server")
        package.__path__ = [str(SERVER)]
        sys.modules["batchgen.server"] = package
        for module, attr in (("server_args", "ServerArgs"), ("storage", "StorageManager"),
                             ("worker_manager", "WorkerManager")):
            stub = types.ModuleType(f"batchgen.server.{module}")
            setattr(stub, attr, object)
            sys.modules[stub.__name__] = stub
        loaded = {}
        for module in ("io_struct", "intake_pool", "scheduling_pool", "batch_scheduler"):
            spec = importlib.util.spec_from_file_location(
                f"batchgen.server.{module}", SERVER / f"{module}.py")
            loaded[module] = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = loaded[module]
            spec.loader.exec_module(loaded[module])
        return loaded["io_struct"], loaded["batch_scheduler"]
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


io_struct, bs = _load_server_modules()


# --------------------------------------------------------------------------- #
#  A template defined on token segments                                        #
# --------------------------------------------------------------------------- #

def _encode(text):
    return [ord(ch) for ch in text]


class _SegmentTemplateTokenizer:
    """Chat structure is emitted as control ids, content as ordinary ids.

    Its rendered text re-encodes to entirely different ids, the property that
    makes a text seam unfaithful for segment-native templates.
    """

    OPEN, CLOSE, GENERATE = 900_001, 900_002, 900_003

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kwargs):
        if not tokenize:
            return "".join(
                f"<open>{m['role']}:{m['content']}<close>" for m in messages
            ) + "<generate>"
        ids = []
        for m in messages:
            ids += [self.OPEN] + _encode(f"{m['role']}:{m['content']}") + [self.CLOSE]
        return ids + [self.GENERATE]

    def __call__(self, texts, **kwargs):
        return {"input_ids": [_encode(text) for text in texts]}


MODEL = "moonshotai/Kimi-K3"
MESSAGES = [{"role": "user", "content": "hello"}]


def _batch_line(custom_id, body):
    return json.dumps({
        "custom_id": custom_id, "method": "POST",
        "url": "/v1/chat/completions" if "messages" in body else "/v1/completions",
        "body": {"model": MODEL, "max_tokens": 8, **body},
    })


def _scheduler(tmp_path, worker, tokenizer, lines):
    input_id = "file-input"
    (tmp_path / input_id).write_bytes("\n".join(lines).encode())
    updates = []
    storage = SimpleNamespace(
        files_dir=tmp_path,
        output_dir=tmp_path,
        updates=updates,
        load_batch=lambda batch_id: SimpleNamespace(
            status=bs.BatchStatus.VALIDATING, input_file_id=input_id,
            max_decoding_length=None, max_context_length=None,
            batchgen_debug=None, temperature=None, top_p=None, top_k=None),
        load_metadata=lambda file_id: {"id": file_id},
        update_batch_status=lambda batch_id, status, **kwargs: updates.append(
            (status, kwargs.get("error"))),
    )
    args = SimpleNamespace(
        model=MODEL, max_intake_capacity=1024,
        incremental_output_dir=str(tmp_path / "incr"),
        no_incremental_save=False, parse_thinking=False, parse_tool_call=False,
    )
    scheduler = bs.BatchScheduler(storage, worker, args)
    scheduler.max_model_len = 4096
    scheduler._deployment_ready.set()
    scheduler._get_tokenizer = lambda model: tokenizer
    scheduler._scheduling_pool.set_capacity(8)
    scheduler._pool_initialized = True

    async def no_finalizer(*args):
        pass

    scheduler._wait_and_finalize_batch = no_finalizer
    return scheduler


def _admission_message(tmp_path, lines, turn_body=None):
    """Run a batch (and optionally one turn) through the real drain."""
    tokenizer = _SegmentTemplateTokenizer()

    async def run():
        worker = SimpleNamespace(
            state="serving", request_queue=queue.Queue(),
            report_worker_fatal=lambda *args, **kwargs: None,
        )
        scheduler = _scheduler(tmp_path, worker, tokenizer, lines)
        scheduler._loop = asyncio.get_running_loop()
        await scheduler._process_batch("batch-1")
        if turn_body is not None:
            scheduler.submit_turn(
                "turn-1", io_struct.TurnRequest(**turn_body), time.time())
        drain = asyncio.ensure_future(scheduler._drain_intake_to_worker())
        msg = await asyncio.get_running_loop().run_in_executor(
            None, lambda: worker.request_queue.get(timeout=5))
        scheduler._stopped.set()
        await drain
        return msg

    return asyncio.run(run()), tokenizer


def test_admission_entries_carry_the_template_ids_and_no_text(tmp_path):
    turn = {
        "model": MODEL, "messages": [{"role": "user", "content": "turn"}],
        "max_tokens": 4,
        "session": {"traj_id": "t", "session_id": "s", "parent_session_id": None},
    }
    msg, tokenizer = _admission_message(tmp_path, [
        _batch_line("chat", {"messages": MESSAGES}),
        _batch_line("plain", {"prompt": "complete me"}),
    ], turn_body=turn)

    entries = {e["request_id"]: e for e in msg["entries"]}
    assert set(entries) == {"chat@batch-1", "plain@batch-1", "turn-1"}
    for entry in entries.values():
        assert "text" not in entry
        assert isinstance(entry["input_ids"], np.ndarray)
        assert entry["input_ids"].dtype == np.int32 and entry["input_ids"].ndim == 1

    chat_ids = entries["chat@batch-1"]["input_ids"].tolist()
    assert chat_ids == tokenizer.apply_chat_template(MESSAGES, tokenize=True)
    rendered = tokenizer.apply_chat_template(MESSAGES, tokenize=False)
    assert chat_ids != tokenizer([rendered])["input_ids"][0]
    turn_messages = [{"role": "user", "content": "turn"}]
    assert entries["turn-1"]["input_ids"].tolist() == tokenizer.apply_chat_template(
        turn_messages, tokenize=True)
    # A completion prompt is encoded exactly as the worker encoded text.
    assert entries["plain@batch-1"]["input_ids"].tolist() == _encode("complete me")


def test_the_worker_binds_the_ids_it_receives(tmp_path):
    """Scheduler -> pickled admission message -> worker trajectory book."""
    msg, tokenizer = _admission_message(
        tmp_path, [_batch_line("chat", {"messages": MESSAGES})])
    msg = pickle.loads(pickle.dumps(msg))  # the process queue and rank broadcast

    methods = load_worker_methods(
        ["_admit_sequences_from_message", "_bind_admitted_prompts"],
        {
            "List": List, "logging": logging, "time": time, "torch": torch,
            "SequenceEntry": SequenceEntry,
            "_admitted_prompt_ids": load_worker_functions(
                ["_admitted_prompt_ids"], {"np": np, "torch": torch},
            )["_admitted_prompt_ids"],
        },
    )

    class _Worker:
        rank = 0
        max_decoding_length = 8
        max_input_length = 1 << 20
        model_context_length = 64
        _response_queue = None
        _admit_sequences_from_message = methods["_admit_sequences_from_message"]
        _bind_admitted_prompts = methods["_bind_admitted_prompts"]

        def __init__(self):
            self.global_batch = SequenceBatch()
            self._group_uuids = set()
            attach_trajectory_state(
                self,
                trajectory_book=QueryBook(capacity_bytes=64 * 4, page_tokens=64),
            )
            attach_prefix_state(self)
            # Alias kept for the assertions below.
            self._trajectory_book = self._traj.trajectory_book

        def _publish_trajectory_pool_capacity(self):
            pass

        def _prefill_prefix_pool_enabled(self):
            return False

        def _build_local_query_book_for_admitted(self, uuids):
            pass

    worker = _Worker()
    worker._admit_sequences_from_message(msg)

    seq = worker.global_batch.get_sequence("chat@batch-1")
    expected = tokenizer.apply_chat_template(MESSAGES, tokenize=True)
    slot = worker._trajectory_book.slot_for(seq.uuid)
    assert worker._trajectory_book.tokens(slot).tolist() == expected
    assert seq.prompt_length == len(expected)


def test_a_non_flat_template_result_fails_loudly():
    scheduler = bs.BatchScheduler.__new__(bs.BatchScheduler)

    class _Nested(_SegmentTemplateTokenizer):
        def apply_chat_template(self, messages, tokenize=False, **kwargs):
            return [super().apply_chat_template(messages, tokenize=True)]

    ok, error, (request,) = bs.parse_batch_file(
        _batch_line("chat", {"messages": MESSAGES}).encode())
    assert ok, error
    with pytest.raises(TypeError, match="flat sequence of ints"):
        scheduler._construct_prompt(request, _Nested())


# --------------------------------------------------------------------------- #
#  Batch prompt construction runs off the event loop                          #
# --------------------------------------------------------------------------- #

class _RecordingTokenizer(_SegmentTemplateTokenizer):
    """Records the building thread and finishes out of order. "refuse" raises
    ValueError from the template (a per-request rejection); "break" returns
    a malformed result, a server defect that must fail the whole batch."""

    def __init__(self):
        self.threads = set()
        self._lock = threading.Lock()

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        with self._lock:
            self.threads.add(threading.get_ident())
        content = messages[-1]["content"]
        time.sleep(random.Random(content).random() * 0.002)
        if content.startswith("refuse"):
            raise ValueError("the template refuses this conversation")
        if content.startswith("break"):
            return {"input_ids": [1, 2]}
        return super().apply_chat_template(messages, tokenize=tokenize, **kwargs)


def _run_batch(tmp_path, contents):
    tokenizer = _RecordingTokenizer()
    lines = [_batch_line(f"c{i}", {"messages": [{"role": "user", "content": c}]})
             for i, c in enumerate(contents)]

    async def run():
        worker = SimpleNamespace(state="serving", request_queue=queue.Queue(),
                                 report_worker_fatal=lambda *a, **k: None)
        scheduler = _scheduler(tmp_path, worker, tokenizer, lines)
        scheduler._scheduling_pool.set_capacity(len(contents))
        loop_thread = threading.get_ident()
        await scheduler._process_batch("batch-1")
        return scheduler, loop_thread

    scheduler, loop_thread = asyncio.run(run())
    return scheduler, loop_thread, tokenizer


def _items(tmp_path):
    path = tmp_path / "incr" / "batch-1.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_batch_prompts_are_built_off_the_event_loop(tmp_path):
    scheduler, loop_thread, tokenizer = _run_batch(
        tmp_path, [f"prompt {i}" for i in range(64)])
    assert tokenizer.threads and loop_thread not in tokenizer.threads
    assert len(tokenizer.threads) <= bs._PROMPT_BUILD_THREADS
    assert scheduler._intake_pool.size() == 64


def test_parallel_construction_keeps_order_and_per_request_rejections(tmp_path):
    contents = [("refuse " if i % 7 == 3 else "prompt ") + str(i) for i in range(300)]
    scheduler, _, tokenizer = _run_batch(tmp_path, contents)

    assert [s for s, _ in scheduler.storage.updates] == [bs.BatchStatus.IN_PROGRESS]
    queued = list(scheduler._intake_pool._normal)
    kept = [i for i, c in enumerate(contents) if not c.startswith("refuse")]
    assert [e.request_id for e in queued] == [f"c{i}@batch-1" for i in kept]
    for entry, i in zip(queued, kept):
        messages = [{"role": "user", "content": contents[i]}]
        assert entry.raw_request["input_ids"].tolist() == (
            tokenizer.apply_chat_template(messages, tokenize=True))
    refused = sorted((item for item in _items(tmp_path)),
                     key=lambda item: int(item["custom_id"][1:]))
    assert [item["custom_id"] for item in refused] == [
        f"c{i}" for i, c in enumerate(contents) if c.startswith("refuse")]
    assert all(item["error"]["code"] == "invalid_prompt" for item in refused)
    assert all("refuses this conversation" in item["error"]["message"]
               for item in refused)


def test_a_non_value_error_during_construction_fails_the_batch(tmp_path):
    contents = [f"prompt {i}" for i in range(40)] + ["break 40"]
    scheduler, _, _ = _run_batch(tmp_path, contents)

    status, error = scheduler.storage.updates[-1]
    assert status == bs.BatchStatus.FAILED
    assert error.startswith("Batch processing failed: TypeError: prompt token ids")
    assert scheduler._intake_pool.size() == 0
    assert _items(tmp_path) == []


# --------------------------------------------------------------------------- #
#  Real tokenizers                                                            #
# --------------------------------------------------------------------------- #

def _real_scheduler():
    scheduler = bs.BatchScheduler.__new__(bs.BatchScheduler)
    scheduler._tokenizer = None
    scheduler._tokenizer_model = None
    return scheduler


def _load_or_skip(model):
    from batchgen.config.tokenizer_registry import load_tokenizer

    try:
        return load_tokenizer(model)
    except Exception as exc:  # assets not staged on this machine
        pytest.skip(f"{model} tokenizer unavailable: {type(exc).__name__}: {exc}")


def _request(model, body):
    line = json.dumps({
        "custom_id": "r", "method": "POST",
        "url": "/v1/chat/completions" if "messages" in body else "/v1/completions",
        "body": {"model": model, "max_tokens": 8, **body},
    })
    ok, error, (request,) = bs.parse_batch_file(line.encode())
    assert ok, error
    return request


def _worker_encode(tokenizer, text):
    """The encode call the worker applied to prompt text."""
    return list(tokenizer(
        [text], return_tensors=None, truncation=False, padding=False,
        return_attention_mask=False,
    )["input_ids"][0])


# A run longer than the Kimi tokenizers' 25k-character split, then ordinary
# prose, special-token spellings and CJK.
_LONG_RUN = "a" * 30_000 + " " + "b c " * 500
_CONVERSATIONS = {
    "multi_turn": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What does [EOS] mean? <|im_end|>"},
        {"role": "assistant", "content": "A stop marker."},
        {"role": "user", "content": "你好，世界 and\ttabs\n\n"},
    ],
    "long_run": [{"role": "user", "content": _LONG_RUN}],
}


@pytest.mark.parametrize("model", [
    "zai-org/GLM-5", "zai-org/GLM-5.1", "zai-org/GLM-5.2", "zai-org/GLM-5.3",
    "openai/gpt-oss-120b", "deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-V4-Flash",
    "MiniMaxAI/MiniMax-M2.5", "moonshotai/Kimi-K2.5",
    "moonshotai/Kimi-Linear-48B-A3B-Instruct",
])
@pytest.mark.parametrize("conversation", sorted(_CONVERSATIONS))
def test_text_template_ids_equal_the_old_worker_encode(model, conversation):
    """For templates defined on text, tokenize=True is encode(render(...))."""
    tokenizer = _load_or_skip(model)
    messages = _CONVERSATIONS[conversation]
    scheduler = _real_scheduler()

    ids = scheduler._construct_prompt(
        _request(model, {"messages": messages}), scheduler._get_tokenizer(model))

    template_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True)
    assert isinstance(template_ids, list)
    assert all(isinstance(token, int) for token in template_ids)
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    assert ids.tolist() == template_ids == _worker_encode(tokenizer, rendered)


@pytest.mark.parametrize("model", ["zai-org/GLM-5", "moonshotai/Kimi-K3"])
def test_completion_ids_equal_the_old_worker_encode(model):
    tokenizer = _load_or_skip(model)
    prompt = ["first line [EOS]", "second line " + "x" * 30_000]

    ids = _real_scheduler()._construct_prompt(
        _request(model, {"prompt": prompt}), tokenizer)

    assert ids.tolist() == _worker_encode(tokenizer, "\n".join(prompt))


def test_kimi_k3_serves_a_conversation_its_string_path_refuses():
    model = "moonshotai/Kimi-K3"
    tokenizer = _load_or_skip(model)
    messages = [{"role": "user", "content": _LONG_RUN}]
    with pytest.raises(ValueError, match="cannot be rendered faithfully"):
        tokenizer.apply_chat_template(messages, tokenize=False)

    ids = _real_scheduler()._construct_prompt(
        _request(model, {"messages": messages}), tokenizer)

    assert ids.tolist() == tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True)
