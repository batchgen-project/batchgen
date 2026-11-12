"""GPT-OSS harmony output parsing: channels, tool calls, and the tool-call stop.

The parser runs on the completion text the worker decodes with
``--detokenization-include-special-tokens`` (the harmony markers are special
tokens). Covered here:

* the marker spellings and ids, pinned to the checkpoint's tokenizer assets;
* the decode stop set, pinned to ``generation_config.json`` (``<|call|>`` ends
  a tool-calling turn);
* ``parse_thinking`` / ``parse_tool_calls`` alone and chained the way the
  scheduler chains them;
* malformed and truncated harmony degrading to raw content with a warning;
* the shared response seam (``BatchScheduler._chat_completion_body``) on both
  the batch path (``_write_pool_completion``) and the turn path
  (``build_turn_response``).
"""

import __future__
import ast
import copy
import importlib.util
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest
import tiktoken

from batchgen.config import tokenizer_assets as ta
from batchgen.models.openai.gpt_oss_120b import tokenizer as gpt_oss_tok
from batchgen.models.openai.gpt_oss_120b.tokenizer import (
    GPT_OSS_SPECIAL_TOKENS,
    GPT_OSS_STOP_TOKEN_IDS,
    GPTOssTokenizer,
    HarmonyMessage,
    parse_harmony,
)
from batchgen.worker.completion import CompletionContext, CompletionHandler

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "batchgen" / "models" / "openai" / "gpt_oss_120b"
SCHEDULER = ROOT / "batchgen" / "server" / "batch_scheduler.py"
IO_STRUCT = ROOT / "batchgen" / "server" / "io_struct.py"
LOGGER = gpt_oss_tok.logger.name

ANALYSIS = "<|channel|>analysis<|message|>User asks weather in Paris. Use tool.<|end|>"
TOOL_CALL = (
    "<|start|>assistant<|channel|>commentary to=functions.get_weather "
    '<|constrain|>json<|message|>{"location": "Paris", "unit": "c"}<|call|>'
)
FINAL = "<|start|>assistant<|channel|>final<|message|>It is 21 C in Paris.<|return|>"


def _parser() -> GPTOssTokenizer:
    """A tokenizer whose parse methods run without the staged encoding."""
    return GPTOssTokenizer.__new__(GPTOssTokenizer)


def _chain(text: str):
    """parse_thinking then parse_tool_calls, as BatchScheduler._parse_output."""
    tok = _parser()
    reasoning, content = tok.parse_thinking(text)
    calls, content = tok.parse_tool_calls(content)
    return reasoning, content, calls


# --------------------------------------------------------------------------- #
# Token spellings and the stop set come from the checkpoint assets
# --------------------------------------------------------------------------- #


def test_harmony_marker_ids_match_tokenizer_config():
    config = json.loads((MODEL_DIR / "tokenizer_config.json").read_text())
    decoder = {
        entry["content"]: int(token_id)
        for token_id, entry in config["added_tokens_decoder"].items()
        if entry.get("special")
    }
    for name in (
        "<|start|>", "<|channel|>", "<|constrain|>", "<|message|>",
        "<|end|>", "<|call|>", "<|return|>", "<|endoftext|>",
    ):
        assert GPT_OSS_SPECIAL_TOKENS[name] == decoder[name], name


def test_stop_ids_match_generation_config():
    generation = json.loads((MODEL_DIR / "generation_config.json").read_text())
    assert GPT_OSS_STOP_TOKEN_IDS == frozenset(generation["eos_token_id"])
    assert GPT_OSS_SPECIAL_TOKENS["<|call|>"] in GPT_OSS_STOP_TOKEN_IDS


def _byte_level_encoding() -> tiktoken.Encoding:
    """Byte-level ranks plus the real GPT-OSS special tokens."""
    return tiktoken.Encoding(
        name="gpt_oss_test",
        pat_str=ta.O200K_BASE.pat_str,
        mergeable_ranks={bytes([b]): b for b in range(256)},
        special_tokens=dict(GPT_OSS_SPECIAL_TOKENS),
    )


@pytest.fixture
def byte_tokenizer(monkeypatch) -> GPTOssTokenizer:
    monkeypatch.setattr(gpt_oss_tok, "_get_tiktoken_tokenizer", _byte_level_encoding)
    return GPTOssTokenizer()


def test_tokenizer_exposes_the_stop_set_to_the_worker(byte_tokenizer):
    assert byte_tokenizer.eos_token_ids == set(GPT_OSS_STOP_TOKEN_IDS)
    assert byte_tokenizer.eos_token_id == GPT_OSS_SPECIAL_TOKENS["<|return|>"]


def test_decode_halts_at_call_and_the_turn_parses_to_a_tool_call(byte_tokenizer):
    """Model emits a tool call, then (unstopped) a fabricated tool result.

    Walk the sampled ids through the worker's stop predicate exactly as decode
    does: the sequence ends at <|call|>, so the fabricated result is never
    generated. Detokenize what was kept (stop token included, as with
    --detokenization-include-special-tokens) and parse it.
    """
    fabricated = (
        "<|start|>functions.get_weather to=assistant<|channel|>commentary"
        '<|message|>{"temp": 99}<|end|>'
    )
    ids = byte_tokenizer.encode(ANALYSIS + TOOL_CALL + fabricated)
    ctx = CompletionContext(
        eos_token_ids=frozenset(byte_tokenizer.eos_token_ids),
        model_context_length=1 << 20,
        rank=0,
    )
    kept = []
    for token_id in ids:
        kept.append(token_id)
        if CompletionHandler.should_stop_at_eos(ctx, token_id):
            break
    assert kept[-1] == GPT_OSS_SPECIAL_TOKENS["<|call|>"]

    text = byte_tokenizer.decode(kept, skip_special_tokens=False)
    assert text == ANALYSIS + TOOL_CALL

    reasoning, content, calls = _chain(text)
    assert reasoning == "User asks weather in Paris. Use tool."
    assert content == ""
    assert len(calls) == 1
    assert calls[0]["type"] == "function"
    assert calls[0]["id"].startswith("call_")
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "location": "Paris", "unit": "c",
    }


def test_ignore_eos_does_not_stop_at_call(byte_tokenizer):
    ctx = CompletionContext(
        eos_token_ids=frozenset(byte_tokenizer.eos_token_ids),
        model_context_length=1 << 20,
        rank=0,
    )
    call_id = GPT_OSS_SPECIAL_TOKENS["<|call|>"]
    assert CompletionHandler.should_stop_at_eos(ctx, call_id)
    assert not CompletionHandler.should_stop_at_eos(ctx, call_id, seq_ignore_eos=True)


# --------------------------------------------------------------------------- #
# Pure-text turns
# --------------------------------------------------------------------------- #


def test_analysis_then_final():
    reasoning, content, calls = _chain(
        "<|channel|>analysis<|message|>Compute 6*7.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>The answer is 42.<|return|>"
    )
    assert reasoning == "Compute 6*7."
    assert content == "The answer is 42."
    assert calls is None


def test_final_only_has_no_reasoning():
    reasoning, content, calls = _chain(
        "<|channel|>final<|message|>Just the answer.<|return|>"
    )
    assert reasoning is None
    assert content == "Just the answer."
    assert calls is None


def test_commentary_preamble_is_visible_content():
    reasoning, content, calls = _chain(
        ANALYSIS
        + "<|start|>assistant<|channel|>commentary<|message|>Checking the weather now."
        "<|end|>" + TOOL_CALL
    )
    assert reasoning == "User asks weather in Paris. Use tool."
    assert content == "Checking the weather now."
    assert [c["function"]["name"] for c in calls] == ["get_weather"]


def test_truncated_analysis_is_partial_reasoning():
    """finish_reason=length mid-analysis: reasoning so far, no content."""
    reasoning, content, calls = _chain(
        "<|channel|>analysis<|message|>Let me think about"
    )
    assert reasoning == "Let me think about"
    assert content == ""
    assert calls is None


def test_parse_thinking_alone_keeps_the_tool_call_as_harmony():
    reasoning, content = _parser().parse_thinking(ANALYSIS + TOOL_CALL)
    assert reasoning == "User asks weather in Paris. Use tool."
    assert content == TOOL_CALL


def test_parse_tool_calls_alone_keeps_the_analysis_as_harmony():
    calls, content = _parser().parse_tool_calls(ANALYSIS + TOOL_CALL)
    assert [c["function"]["name"] for c in calls] == ["get_weather"]
    assert content == "<|start|>assistant" + ANALYSIS


def test_parse_tool_calls_without_a_call_returns_text_unchanged():
    text = ANALYSIS + FINAL
    assert _parser().parse_tool_calls(text) == (None, text)


# --------------------------------------------------------------------------- #
# Tool-calling turns
# --------------------------------------------------------------------------- #


def test_recipient_in_the_role_header_of_the_first_message():
    """The prompt ends with <|start|>assistant; ' to=...' is the header rest."""
    reasoning, content, calls = _chain(
        " to=functions.lookup<|channel|>commentary json<|message|>"
        '{"q": "x"}<|call|>'
    )
    assert reasoning is None
    assert content == ""
    assert calls[0]["function"]["name"] == "lookup"
    assert calls[0]["function"]["arguments"] == '{"q": "x"}'


def test_recipient_in_a_later_role_header():
    reasoning, content, calls = _chain(
        ANALYSIS
        + '<|start|>assistant to=functions.lookup<|channel|>commentary <|constrain|>json'
        '<|message|>{"q": 1}<|call|>'
    )
    assert reasoning == "User asks weather in Paris. Use tool."
    assert calls[0]["function"]["name"] == "lookup"


def test_multiple_tool_calls_in_one_completion():
    """Only reachable with the stop disabled (ignore_eos) or in replayed
    trajectories; each <|call|>-ended message is its own call, in order."""
    second = (
        "<|start|>assistant<|channel|>commentary to=functions.get_time "
        '<|constrain|>json<|message|>{"tz": "CET"}<|call|>'
    )
    reasoning, content, calls = _chain(ANALYSIS + TOOL_CALL + second)
    assert [c["function"]["name"] for c in calls] == ["get_weather", "get_time"]
    assert json.loads(calls[1]["function"]["arguments"]) == {"tz": "CET"}
    assert len({c["id"] for c in calls}) == 2
    assert content == ""


def test_arguments_are_returned_verbatim():
    raw_args = '{\n  "location": "Paris"\n}'
    _, _, calls = _chain(
        "<|channel|>commentary to=functions.get_weather <|constrain|>json"
        f"<|message|>{raw_args}<|call|>"
    )
    assert calls[0]["function"]["arguments"] == raw_args


# --------------------------------------------------------------------------- #
# Malformed / truncated harmony degrades to raw content with a warning
# --------------------------------------------------------------------------- #


def test_tool_call_truncated_before_call_degrades(caplog):
    truncated = (
        "<|start|>assistant<|channel|>commentary to=functions.get_weather "
        '<|constrain|>json<|message|>{"location": "Par'
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        reasoning, content, calls = _chain(ANALYSIS + truncated)
    assert reasoning == "User asks weather in Paris. Use tool."
    assert calls is None
    assert content == truncated
    assert "not <|call|>" in caplog.text


def test_tool_call_with_non_json_arguments_degrades(caplog):
    bad = (
        "<|start|>assistant<|channel|>commentary to=functions.f "
        "<|constrain|>json<|message|>{location: Paris}<|call|>"
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        reasoning, content, calls = _chain(ANALYSIS + bad)
    assert calls is None
    assert content == bad
    assert "arguments are not JSON" in caplog.text


def test_one_bad_call_does_not_drop_a_good_one(caplog):
    bad = (
        "<|start|>assistant<|channel|>commentary to=functions.f "
        "<|message|>not json<|call|>"
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _, content, calls = _chain(ANALYSIS + TOOL_CALL + bad)
    assert [c["function"]["name"] for c in calls] == ["get_weather"]
    assert content == bad


def test_fabricated_tool_result_message_degrades(caplog):
    """A non-assistant message in model output (the model inventing the tool
    result) is not parsed as content or as a call."""
    fabricated = (
        "<|start|>functions.get_weather to=assistant<|channel|>commentary"
        '<|message|>{"temp": 99}<|end|>'
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        reasoning, content, calls = _chain(ANALYSIS + fabricated)
    assert calls is None
    assert content == fabricated
    assert "malformed harmony message" in caplog.text


def test_unknown_channel_degrades(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        reasoning, content, calls = _chain(
            "<|channel|>weird<|message|>hello<|end|>"
        )
    assert reasoning is None
    assert content == "<|start|>assistant<|channel|>weird<|message|>hello<|end|>"
    assert "malformed harmony message" in caplog.text


def test_header_without_message_marker_degrades(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        reasoning, content, calls = _chain(
            ANALYSIS + "<|start|>assistant<|channel|>final"
        )
    assert reasoning == "User asks weather in Paris. Use tool."
    assert content == "<|start|>assistant<|channel|>final"
    assert calls is None


def test_text_without_harmony_markers_warns_once(caplog):
    """The stripped decode (no --detokenization-include-special-tokens)."""
    tok = _parser()
    stripped = "analysisUser asks weather.assistantfinalIt is 21 C."
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        first = tok.parse_thinking(stripped)
        second = tok.parse_thinking(stripped)
    assert first == (None, stripped)
    assert second == (None, stripped)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "--detokenization-include-special-tokens" in warnings[0].getMessage()
    assert tok.parse_tool_calls(stripped) == (None, stripped)


def test_empty_text_parses_quietly(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _chain("") == (None, "", None)
    assert caplog.records == []


def test_missing_end_between_messages_is_tolerated():
    reasoning, content, calls = _chain(
        "<|channel|>analysis<|message|>thinking"
        "<|channel|>final<|message|>answer<|return|>"
    )
    assert reasoning == "thinking"
    assert content == "answer"
    assert calls is None


@pytest.mark.parametrize("text", [
    "<|message|><|end|><|call|>",
    "<|start|>",
    "<|start|><|start|>assistant",
    "<|channel|>",
    "<|constrain|>json<|message|>{}<|call|>",
    "plain <|end|> text <|return|>",
    "<|start|>assistant to=<|channel|>commentary<|message|>{}<|call|>",
])
def test_garbage_never_raises(text):
    reasoning, content, calls = _chain(text)
    assert isinstance(content, str)


def test_serialize_round_trips_through_parse():
    for text in (TOOL_CALL, FINAL, ANALYSIS):
        (message,) = [i for i in parse_harmony(text) if isinstance(i, HarmonyMessage)]
        (again,) = [
            i for i in parse_harmony(message.serialize())
            if isinstance(i, HarmonyMessage)
        ]
        assert (again.kind, again.channel, again.recipient, again.body,
                again.terminator) == (
            message.kind, message.channel, message.recipient, message.body,
            message.terminator)


# --------------------------------------------------------------------------- #
# The shared response seam: batch path and turn path
# --------------------------------------------------------------------------- #


SEAM_METHODS = (
    "_parse_output", "_chat_completion_body", "_write_pool_completion",
    "_append_pool_result_item", "build_turn_response",
)


def _load_io_struct():
    """io_struct by path: importing batchgen.server pulls in uvicorn."""
    spec = importlib.util.spec_from_file_location("_gpt_oss_io_struct", IO_STRUCT)
    module = importlib.util.module_from_spec(spec)
    # pydantic resolves postponed annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _isolated_scheduler_class():
    """The response seam of BatchScheduler, compiled without the server stack."""
    tree = ast.parse(SCHEDULER.read_text(), filename=str(SCHEDULER))
    scheduler = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchScheduler"
    )
    methods = [
        copy.deepcopy(node) for node in scheduler.body
        if isinstance(node, ast.FunctionDef) and node.name in SEAM_METHODS
    ]
    assert {node.name for node in methods} == set(SEAM_METHODS)
    helpers = [
        copy.deepcopy(node) for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "turn_timing_summary"
    ]
    module = ast.Module(
        body=helpers + [ast.ClassDef(
            name="IsolatedBatchScheduler", bases=[], keywords=[],
            body=methods, decorator_list=[],
        )],
        type_ignores=[],
    )
    io_struct = _load_io_struct()

    class TurnError(Exception):
        pass

    namespace = {
        "Any": Any, "Dict": Dict, "List": List, "Optional": Optional,
        "json": json, "logger": logging.getLogger("test.gpt_oss.seam"),
        "time": time, "uuid": uuid,
        "ToolCall": io_struct.ToolCall,
        "ToolCallFunction": io_struct.ToolCallFunction,
        "TurnError": TurnError,
        "_TURN_REQUEST_ERROR_CODES": (),
    }
    # batch_scheduler.py uses postponed annotations (``_TurnWaiter`` etc.).
    code = compile(
        ast.fix_missing_locations(module), str(SCHEDULER), "exec",
        flags=__future__.annotations.compiler_flag, dont_inherit=True,
    )
    exec(code, namespace)
    return namespace["IsolatedBatchScheduler"]


MODEL = "openai/gpt-oss-120b"


def _scheduler(tmp_path, parse_thinking=True, parse_tool_call=True):
    cls = _isolated_scheduler_class()
    scheduler = cls.__new__(cls)
    scheduler.server_args = SimpleNamespace(
        incremental_output_dir=str(tmp_path),
        parse_thinking=parse_thinking,
        parse_tool_call=parse_tool_call,
    )
    tok = _parser()
    scheduler._get_tokenizer = lambda model: tok
    scheduler._pool_request_meta = {
        "batch-1": {
            "request-1": {
                "custom_id": "request-1",
                "url": "/v1/chat/completions",
                "model": MODEL,
            }
        }
    }
    return scheduler


def _worker_result(text):
    return {
        "text": text,
        "prompt_length": 50,
        "decoded_length": 30,
        "finish_reason": "stop",
        "cached_tokens": 0,
        "computed_tokens": 50,
    }


def _assert_tool_call_message(message):
    assert message["role"] == "assistant"
    assert message["content"] == ""
    assert message["reasoning_content"] == "User asks weather in Paris. Use tool."
    (call,) = message["tool_calls"]
    assert call["type"] == "function"
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {
        "location": "Paris", "unit": "c",
    }
    assert "<|" not in message["content"]


def test_batch_path_returns_tool_calls(tmp_path):
    scheduler = _scheduler(tmp_path)
    scheduler._write_pool_completion(
        "batch-1", "request-1", _worker_result(ANALYSIS + TOOL_CALL),
    )
    record = json.loads((tmp_path / "batch-1.jsonl").read_text())
    assert record["error"] is None
    _assert_tool_call_message(record["response"]["body"]["choices"][0]["message"])


def test_batch_path_returns_reasoning_and_final(tmp_path):
    scheduler = _scheduler(tmp_path)
    scheduler._write_pool_completion(
        "batch-1", "request-1",
        _worker_result(
            "<|channel|>analysis<|message|>Compute 6*7.<|end|>"
            "<|start|>assistant<|channel|>final<|message|>42<|return|>"
        ),
    )
    record = json.loads((tmp_path / "batch-1.jsonl").read_text())
    message = record["response"]["body"]["choices"][0]["message"]
    assert message == {
        "role": "assistant", "content": "42", "reasoning_content": "Compute 6*7.",
    }


def test_turn_path_returns_tool_calls(tmp_path):
    scheduler = _scheduler(tmp_path)
    waiter = SimpleNamespace(
        turn_id="turn-1",
        model=MODEL,
        session=SimpleNamespace(
            traj_id="traj-1", session_id="sess-1", parent_session_id=None,
        ),
        timing={"http_received_at": time.time()},
    )
    body = scheduler.build_turn_response(waiter, _worker_result(ANALYSIS + TOOL_CALL))
    _assert_tool_call_message(body["choices"][0]["message"])
    assert body["batchgen"]["turn_id"] == "turn-1"


def test_flags_off_leave_the_raw_harmony_text(tmp_path):
    scheduler = _scheduler(tmp_path, parse_thinking=False, parse_tool_call=False)
    body = scheduler._chat_completion_body(
        MODEL, _worker_result(ANALYSIS + TOOL_CALL), int(time.time()),
    )
    message = body["choices"][0]["message"]
    assert message == {"role": "assistant", "content": ANALYSIS + TOOL_CALL}
