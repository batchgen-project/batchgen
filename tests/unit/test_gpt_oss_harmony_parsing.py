"""GPT-OSS harmony output parsing: channels, tool calls, and the tool-call stop.

GPT-OSS chat output is parsed by default (the tokenizer declares
``parse_thinking_default`` / ``parse_tool_call_default``), and the parser runs
on the completion text the worker decodes by default: the harmony grammar
markers survive ``skip_special_tokens=True`` and the stop token is decoded
with the completion. Covered here:

* the marker spellings and ids, pinned to the checkpoint's tokenizer assets;
* the decode stop set, pinned to ``generation_config.json`` (``<|call|>`` ends
  a tool-calling turn);
* default decode keeping the grammar markers and stripping the rest;
* ``parse_thinking`` / ``parse_tool_calls`` alone and chained the way the
  scheduler chains them;
* malformed and truncated harmony degrading to raw content with a warning;
* the shared response seam (``BatchScheduler._chat_completion_body``) on both
  the batch path (``_write_pool_completion``) and the turn path
  (``build_turn_response``);
* the server flags resolving against the model default, and the whole default
  path (worker decode, then the response seam) from sampled ids.
"""

import __future__
import ast
import copy
import dataclasses
import importlib
import importlib.util
import json
import logging
import sys
import time
import types
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest
import tiktoken
import torch

from batchgen.config import tokenizer_assets as ta
from batchgen.models.openai.gpt_oss_120b import tokenizer as gpt_oss_tok
from batchgen.models.openai.gpt_oss_120b.tokenizer import (
    GPT_OSS_GRAMMAR_TOKEN_IDS,
    GPT_OSS_SPECIAL_TOKENS,
    GPT_OSS_STOP_TOKEN_IDS,
    GPTOssTokenizer,
    HarmonyMessage,
    parse_harmony,
)
from batchgen.worker.completion import CompletionContext, CompletionHandler
from tests.worker_method_source import load_worker_methods

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
    """Text with every marker stripped (what decode produced before the
    grammar markers became non-skipped)."""
    tok = _parser()
    stripped = "analysisUser asks weather.assistantfinalIt is 21 C."
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        first = tok.parse_thinking(stripped)
        second = tok.parse_thinking(stripped)
    assert first == (None, stripped)
    assert second == (None, stripped)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "no harmony message markers" in warnings[0].getMessage()
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
    "_parse_output", "_effective_output_parsing", "_chat_completion_body",
    "_write_pool_completion",
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


def _scheduler(tmp_path, parse_thinking=None, parse_tool_call=None, tok=None):
    """The seam with the server's flags; None (the default) = model default."""
    cls = _isolated_scheduler_class()
    scheduler = cls.__new__(cls)
    scheduler.server_args = SimpleNamespace(
        incremental_output_dir=str(tmp_path),
        parse_thinking=parse_thinking,
        parse_tool_call=parse_tool_call,
    )
    tok = _parser() if tok is None else tok
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


# --------------------------------------------------------------------------- #
# Default decode keeps the harmony grammar
# --------------------------------------------------------------------------- #


GRAMMAR_MARKERS = (
    "<|start|>", "<|channel|>", "<|constrain|>", "<|message|>",
    "<|end|>", "<|call|>", "<|return|>",
)


def test_grammar_token_ids_are_the_harmony_markers():
    assert GPT_OSS_GRAMMAR_TOKEN_IDS == frozenset(
        GPT_OSS_SPECIAL_TOKENS[m] for m in GRAMMAR_MARKERS
    )


def test_default_decode_keeps_grammar_markers(byte_tokenizer):
    text = ANALYSIS + TOOL_CALL
    ids = byte_tokenizer.encode(text)
    assert byte_tokenizer.decode(ids, skip_special_tokens=True) == text
    for marker in GRAMMAR_MARKERS:
        (marker_id,) = byte_tokenizer.encode(marker)
        assert byte_tokenizer.decode([marker_id], skip_special_tokens=True) == marker


@pytest.mark.parametrize("token_id", [
    GPT_OSS_SPECIAL_TOKENS["<|startoftext|>"],
    GPT_OSS_SPECIAL_TOKENS["<|endoftext|>"],
    GPT_OSS_SPECIAL_TOKENS["<|reserved_200000|>"],
    GPT_OSS_SPECIAL_TOKENS["<|reserved_200011|>"],
    200013, 200018, 201087,
])
def test_default_decode_strips_non_grammar_specials(byte_tokenizer, token_id):
    body = byte_tokenizer.encode("<|channel|>final<|message|>hi")
    assert byte_tokenizer.decode(body + [token_id], skip_special_tokens=True) == (
        "<|channel|>final<|message|>hi"
    )


def test_include_special_tokens_decode_is_unchanged(byte_tokenizer):
    ids = byte_tokenizer.encode(FINAL) + [GPT_OSS_SPECIAL_TOKENS["<|endoftext|>"]]
    assert byte_tokenizer.decode(ids, skip_special_tokens=False) == FINAL + "<|endoftext|>"


# --------------------------------------------------------------------------- #
# Output parsing flags: None = the model's default, explicit flags override
# --------------------------------------------------------------------------- #


def _non_harmony_tokenizer():
    from batchgen.config.tokenizer_registry import load_tokenizer
    return load_tokenizer("deepseek-ai/DeepSeek-R1")


def test_model_declared_defaults():
    from batchgen.config.base_tokenizer import BaseTokenizer
    assert (BaseTokenizer.parse_thinking_default,
            BaseTokenizer.parse_tool_call_default) == (False, False)
    assert (GPTOssTokenizer.parse_thinking_default,
            GPTOssTokenizer.parse_tool_call_default) == (True, True)
    tok = _non_harmony_tokenizer()
    assert (tok.parse_thinking_default, tok.parse_tool_call_default) == (False, False)


@pytest.mark.parametrize("flags, gpt_oss, other", [
    ((None, None), (True, True), (False, False)),
    ((True, True), (True, True), (True, True)),
    ((False, False), (False, False), (False, False)),
    ((False, None), (False, True), (False, False)),
    ((None, False), (True, False), (False, False)),
    ((True, None), (True, True), (True, False)),
    ((None, True), (True, True), (False, True)),
])
def test_flag_resolution(tmp_path, flags, gpt_oss, other):
    scheduler = _scheduler(tmp_path, *flags)
    assert scheduler._effective_output_parsing(_parser()) == gpt_oss
    assert scheduler._effective_output_parsing(_non_harmony_tokenizer()) == other


def test_non_harmony_model_default_leaves_content_raw(tmp_path):
    raw = "<think>Let me reason.</think>The answer is 42."
    tok = _non_harmony_tokenizer()
    body = _scheduler(tmp_path, tok=tok)._chat_completion_body(
        "deepseek-ai/DeepSeek-R1", _worker_result(raw), int(time.time()),
    )
    assert body["choices"][0]["message"] == {"role": "assistant", "content": raw}
    body = _scheduler(tmp_path, parse_thinking=True, tok=tok)._chat_completion_body(
        "deepseek-ai/DeepSeek-R1", _worker_result(raw), int(time.time()),
    )
    message = body["choices"][0]["message"]
    assert message["content"] == "The answer is 42."
    assert message["reasoning_content"] == "Let me reason."


# --------------------------------------------------------------------------- #
# The default server path end to end: sampled ids -> worker decode -> response
# --------------------------------------------------------------------------- #


_WORKER_DECODE = load_worker_methods(
    ["_decode_tokens_to_string"], {"torch": torch}
)["_decode_tokens_to_string"]


def _load_server_args_module():
    """server_args without the batchgen.server package init (uvicorn)."""
    previous = sys.modules.get("batchgen.server")
    package = types.ModuleType("batchgen.server")
    package.__path__ = [str(ROOT / "batchgen" / "server")]
    sys.modules["batchgen.server"] = package
    try:
        return importlib.import_module("batchgen.server.server_args")
    finally:
        if previous is None:
            sys.modules.pop("batchgen.server", None)
        else:
            sys.modules["batchgen.server"] = previous


def _default_server_args():
    """parse_thinking / parse_tool_call / detokenization defaults of ServerArgs."""
    ServerArgs = _load_server_args_module().ServerArgs
    fields = {f.name: f.default for f in dataclasses.fields(ServerArgs)}
    return (fields["parse_thinking"], fields["parse_tool_call"],
            fields["detokenization_include_special_tokens"])


def _served_message(tmp_path, tok, sampled_text, *, max_tokens=None,
                    parse_thinking=None, parse_tool_call=None,
                    include_special_tokens=False):
    """Run the sampled ids through the stop predicate, the real worker decode
    and the real response seam, with the given server flags."""
    ctx = CompletionContext(
        eos_token_ids=frozenset(tok.eos_token_ids),
        model_context_length=1 << 20, rank=0,
    )
    generated = []
    for token_id in tok.encode(sampled_text):
        generated.append(token_id)
        if CompletionHandler.should_stop_at_eos(ctx, token_id):
            break
        if max_tokens is not None and len(generated) == max_tokens:
            break
    # The trajectory buffer is padded past the generated tokens.
    buffer = torch.tensor(generated + [tok.pad_token_id] * 4, dtype=torch.int64)
    worker = SimpleNamespace(
        eos_token_ids=tok.eos_token_ids, pad_token_id=tok.pad_token_id,
        tokenizer=tok,
        detokenization_include_special_tokens=include_special_tokens,
    )
    text = _WORKER_DECODE(worker, buffer)
    scheduler = _scheduler(tmp_path, parse_thinking, parse_tool_call, tok=tok)
    body = scheduler._chat_completion_body(MODEL, _worker_result(text), int(time.time()))
    return text, body["choices"][0]["message"]


SIMPLE_TURN = (
    "<|channel|>analysis<|message|>The user asks: what is 2+3? So answer: 5.<|end|>"
    "<|start|>assistant<|channel|>final<|message|>5<|return|>"
)


def test_server_defaults_resolve_to_the_model():
    assert _default_server_args() == (None, None, False)


def test_default_server_returns_reasoning_and_content(tmp_path, byte_tokenizer):
    """The reported failure: with default flags gpt-oss content was
    'analysisThe user asks: ... So answer: 5.assistantfinal5'."""
    parse_thinking, parse_tool_call, include = _default_server_args()
    text, message = _served_message(
        tmp_path, byte_tokenizer, SIMPLE_TURN,
        parse_thinking=parse_thinking, parse_tool_call=parse_tool_call,
        include_special_tokens=include,
    )
    assert text == SIMPLE_TURN
    assert message == {
        "role": "assistant",
        "content": "5",
        "reasoning_content": "The user asks: what is 2+3? So answer: 5.",
    }


def test_default_server_truncated_analysis_is_partial_reasoning(tmp_path, byte_tokenizer):
    prefix = "<|channel|>analysis<|message|>The user asks"
    _, message = _served_message(
        tmp_path, byte_tokenizer, SIMPLE_TURN,
        max_tokens=len(byte_tokenizer.encode(prefix)),
    )
    assert message == {
        "role": "assistant", "content": "", "reasoning_content": "The user asks",
    }


def test_default_server_tool_call_ended_by_call(tmp_path, byte_tokenizer):
    fabricated = (
        "<|start|>functions.get_weather to=assistant<|channel|>commentary"
        '<|message|>{"temp": 99}<|end|>'
    )
    text, message = _served_message(
        tmp_path, byte_tokenizer, ANALYSIS + TOOL_CALL + fabricated,
    )
    assert text.endswith("<|call|>")
    _assert_tool_call_message(message)


def test_default_server_truncated_tool_call_stays_raw(tmp_path, byte_tokenizer, caplog):
    cut = len(byte_tokenizer.encode(ANALYSIS + TOOL_CALL)) - 3
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        text, message = _served_message(
            tmp_path, byte_tokenizer, ANALYSIS + TOOL_CALL, max_tokens=cut,
        )
    assert "tool_calls" not in message
    assert message["reasoning_content"] == "User asks weather in Paris. Use tool."
    assert message["content"] == text[len(ANALYSIS):]
    assert message["content"].startswith("<|start|>assistant<|channel|>commentary to=")
    assert "<|call|>" not in message["content"]
    assert "not <|call|>" in caplog.text


def test_no_parse_flags_return_recoverable_raw_harmony(tmp_path, byte_tokenizer):
    text, message = _served_message(
        tmp_path, byte_tokenizer, SIMPLE_TURN,
        parse_thinking=False, parse_tool_call=False,
    )
    assert message == {"role": "assistant", "content": SIMPLE_TURN}
    # Recoverable: the raw content parses to the same answer.
    assert _chain(message["content"])[:2] == (
        "The user asks: what is 2+3? So answer: 5.", "5",
    )


def test_no_parse_thinking_alone_still_extracts_tool_calls(tmp_path, byte_tokenizer):
    _, message = _served_message(
        tmp_path, byte_tokenizer, ANALYSIS + TOOL_CALL, parse_thinking=False,
    )
    assert "reasoning_content" not in message
    assert message["content"] == "<|start|>assistant" + ANALYSIS
    assert [c["function"]["name"] for c in message["tool_calls"]] == ["get_weather"]


def test_endoftext_stop_is_stripped(tmp_path, byte_tokenizer):
    turn = "<|channel|>final<|message|>done<|endoftext|>"
    text, message = _served_message(tmp_path, byte_tokenizer, turn)
    assert text == "<|channel|>final<|message|>done"
    assert message == {"role": "assistant", "content": "done"}


def test_include_special_tokens_flag_still_parses(tmp_path, byte_tokenizer):
    text, message = _served_message(
        tmp_path, byte_tokenizer, SIMPLE_TURN, include_special_tokens=True,
    )
    assert text == SIMPLE_TURN
    assert message["content"] == "5"


# --------------------------------------------------------------------------- #
# A completion that is nothing but grammar markers is an empty message
# (first-token stop: the decode slice's min-token floor keeps the marker)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bare", ["<|return|>", "<|call|>", "<|end|>",
                                  "<|return|>\n", " <|call|> "])
def test_bare_stop_marker_parses_as_empty_message(bare, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        reasoning, content, calls = _chain(bare)
    assert reasoning is None
    assert content == ""
    assert calls is None
    assert caplog.records == []


def test_bare_marker_with_payload_still_falls_back_raw():
    # Anything beyond markers keeps the existing raw fallback.
    reasoning, content = _parser().parse_thinking("<|return|>oops")
    assert reasoning is None
    assert content == "<|return|>oops"


def test_parse_tool_calls_alone_empties_a_bare_marker():
    # parse_tool_calls may run without parse_thinking (thinking parsing
    # off); a bare marker is an empty message there too. The --no-parse
    # contract is unaffected: with parsing off the server never calls
    # either method.
    tok = _parser()
    calls, content = tok.parse_tool_calls("<|return|>")
    assert calls is None and content == ""
