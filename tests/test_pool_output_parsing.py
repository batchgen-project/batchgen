"""Pool-mode output must honor the same parser flags as legacy batch mode.

It must also keep a per-request failure a FAILURE: a worker that fails one
request (over-context prompt, a reservation no Host pool can ever hold)
publishes an ``error`` on the completion, and the batch output line has to be
the Batch API error item, not a 200 with empty text.

A chat completion whose parsed message carries tool calls reports
``finish_reason="tool_calls"`` when the worker stopped on its own, for every
registered model family's real tool-call parser; truncation and repetition
keep the worker's reason.

A DeepSeek-V4 completion is split into reasoning, content and tool calls as
the vendored V4 parser reads it, in thinking and in chat mode.
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

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = ROOT / "batchgen" / "server" / "batch_scheduler.py"
IO_STRUCT = ROOT / "batchgen" / "server" / "io_struct.py"


WRITER_METHODS = (
    "_write_pool_completion", "_append_pool_result_item", "_chat_completion_body",
)
PARSER_METHODS = ("_parse_output", "_effective_output_parsing")


def _io_struct():
    """``io_struct`` by path: the server package init imports the GPU stack."""
    name = "_pool_output_parsing_io_struct"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, IO_STRUCT)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _isolated_write_pool_completion(extra_methods=()):
    """Compile the writer without importing the GPU/JIT-heavy server stack."""
    tree = ast.parse(SCHEDULER.read_text(), filename=str(SCHEDULER))
    scheduler = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchScheduler"
    )
    wanted = set(WRITER_METHODS) | set(extra_methods)
    methods = [
        copy.deepcopy(node)
        for node in scheduler.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in methods} == wanted
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
    if extra_methods:
        io_struct = _io_struct()
        namespace["ToolCall"] = io_struct.ToolCall
        namespace["ToolCallFunction"] = io_struct.ToolCallFunction
    # batch_scheduler.py defers annotation evaluation; so must this compile.
    exec(
        compile(
            ast.fix_missing_locations(module), str(SCHEDULER), "exec",
            flags=__future__.annotations.compiler_flag,
        ),
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


# ------------------------------------------------- finish_reason=tool_calls

# One complete call and the same call cut off by the token budget, in each
# registered model family's own tool-call syntax. Keyed by tokenizer type so a
# newly registered family without a case fails
# test_every_registered_tool_call_parser_has_a_case. ``attrs`` are the
# instance fields the parser reads; the tokenizers are built with __new__ so
# no vocabulary or chat template is loaded.
_GLM5_CASE = {
    "complete": (
        "<tool_call>get_weather<arg_key>city</arg_key>"
        "<arg_value>Paris</arg_value></tool_call>"
    ),
    "truncated": (
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Par"
    ),
}
_KIMI_SECTION_CASE = {
    "complete": (
        "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0"
        '<|tool_call_argument_begin|>{"city": "Paris"}<|tool_call_end|>'
        "<|tool_calls_section_end|>"
    ),
    "truncated": (
        "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0"
        '<|tool_call_argument_begin|>{"city": "Par'
    ),
}
TOOL_CALL_CASES = {
    "glm_moe_dsa": _GLM5_CASE,
    "glm_moe_dsa_5_1": _GLM5_CASE,
    "glm_moe_dsa_5_2": _GLM5_CASE,
    "glm_moe_dsa_5_3": _GLM5_CASE,
    "kimi_k25": _KIMI_SECTION_CASE,
    "kimi_linear": _KIMI_SECTION_CASE,
    "kimi_k3": {
        "complete": (
            '<|open|>tools<|sep|><|open|>call tool="get_weather" index="0"<|sep|>'
            '<|open|>argument key="city" type="string"<|sep|>Paris'
            "<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|>"
        ),
        "truncated": (
            '<|open|>tools<|sep|><|open|>call tool="get_weather" index="0"<|sep|>'
            '<|open|>argument key="city" type="string"<|sep|>Par'
        ),
    },
    "minimax_m25": {
        "complete": (
            '<minimax:tool_call><invoke name="get_weather">'
            '<parameter name="city">Paris</parameter></invoke></minimax:tool_call>'
        ),
        "truncated": (
            '<minimax:tool_call><invoke name="get_weather">'
            '<parameter name="city">Par'
        ),
    },
    "deepseek_v3": {
        "complete": (
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather\n"
            '{"city": "Paris"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
        ),
        "truncated": (
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather\n"
            '{"city": "Par'
        ),
    },
    "deepseek_v4": {
        "complete": (
            "\n\n<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="get_weather">\n'
            '<｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
        ),
        "truncated": (
            "\n\n<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="get_weather">\n'
            '<｜DSML｜parameter name="city" string="true">Par'
        ),
        "attrs": {"eos_token": "<｜end▁of▁sentence｜>"},
    },
    "gpt_oss": {
        "complete": (
            "<|start|>assistant<|channel|>commentary to=functions.get_weather "
            '<|constrain|>json<|message|>{"city": "Paris"}<|call|>'
        ),
        "truncated": (
            "<|start|>assistant<|channel|>commentary to=functions.get_weather "
            '<|constrain|>json<|message|>{"city": "Par'
        ),
    },
}


@pytest.fixture(scope="module")
def registered_tokenizers():
    from batchgen.config.tokenizer_registry import get_registered_tokenizers

    return get_registered_tokenizers()


def _parsing_scheduler(tmp_path, tokenizer, url="/v1/chat/completions"):
    """The real writer and ``_parse_output`` around one model's tokenizer."""
    BatchScheduler = _isolated_write_pool_completion(PARSER_METHODS)
    scheduler = BatchScheduler.__new__(BatchScheduler)
    scheduler.server_args = SimpleNamespace(
        incremental_output_dir=str(tmp_path),
        parse_thinking=False,
        parse_tool_call=True,
    )
    scheduler._get_tokenizer = lambda model: tokenizer
    scheduler._pool_request_meta = {
        "batch-1": {
            "request-1": {
                "custom_id": "custom-1",
                "url": url,
                "model": "model-under-test",
            }
        }
    }
    return scheduler


def _parser_only(cls, case):
    tokenizer = cls.__new__(cls)
    for name, value in case.get("attrs", {}).items():
        setattr(tokenizer, name, value)
    return tokenizer


def _choice(scheduler, tmp_path, text, finish_reason):
    scheduler._write_pool_completion(
        "batch-1",
        "request-1",
        {
            "text": text,
            "prompt_length": 10,
            "decoded_length": 20,
            "finish_reason": finish_reason,
        },
    )
    record = json.loads((tmp_path / "batch-1.jsonl").read_text())
    assert record["error"] is None
    return record["response"]["body"]["choices"][0]


def test_every_registered_tool_call_parser_has_a_case(registered_tokenizers):
    assert set(registered_tokenizers) == set(TOOL_CALL_CASES)


FAMILIES = sorted(TOOL_CALL_CASES)


@pytest.mark.parametrize("family", FAMILIES)
def test_every_parser_returns_calls_in_the_contract_shape(
    family, registered_tokenizers,
):
    """``BaseTokenizer.parse_tool_calls``: each call is
    ``{id, type, function: {name, arguments}}``; the server reads every key."""
    case = TOOL_CALL_CASES[family]
    tokenizer = _parser_only(registered_tokenizers[family], case)

    calls, _ = tokenizer.parse_tool_calls(case["complete"])

    assert calls, f"{family}: a well-formed call was not parsed"
    for call in calls:
        assert isinstance(call["id"], str) and call["id"], family
        assert call["type"] == "function", family
        assert call["function"]["name"] == "get_weather", family
        assert isinstance(call["function"]["arguments"], str), family
        assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}


def test_deepseek_v4_calls_in_one_response_get_distinct_ids(
    registered_tokenizers,
):
    case = TOOL_CALL_CASES["deepseek_v4"]
    tokenizer = _parser_only(registered_tokenizers["deepseek_v4"], case)
    text = (
        "\n\n<｜DSML｜tool_calls>\n"
        '<｜DSML｜invoke name="get_weather">\n'
        '<｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        '<｜DSML｜invoke name="get_weather">\n'
        '<｜DSML｜parameter name="city" string="true">Rome</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
    )

    calls, content = tokenizer.parse_tool_calls(text)

    assert [json.loads(c["function"]["arguments"]) for c in calls] == [
        {"city": "Paris"}, {"city": "Rome"},
    ]
    ids = [call["id"] for call in calls]
    assert all(ids) and len(set(ids)) == len(ids)
    assert content == ""


# Completion text the vendored DeepSeek-V4 parser
# (deepseekv4_flash/assets/encoding/encoding_dsv4.py) rejects, one per
# rejection reachable from a completion string, with the exception it raises
# in the mode the wrapper parses that text in: ``thinking`` when the text holds
# a ``</think>``, else ``chat``.
_DSML = "｜DSML｜"
_DSV4_INVOKE = f'<{_DSML}invoke name="get_weather">\n'
_DSV4_PARAM = (
    f'<{_DSML}parameter name="city" string="true">Paris</{_DSML}parameter>\n'
)
_DSV4_BLOCK = (
    f"\n\n<{_DSML}tool_calls>\n{_DSV4_INVOKE}{_DSV4_PARAM}"
    f"</{_DSML}invoke>\n</{_DSML}tool_calls>"
)
DEEPSEEK_V4_REJECTED = {
    # parse_tool_calls: block opener not followed by '>\n'
    "block_opener": (
        ValueError, "chat",
        _DSV4_BLOCK.replace("tool_calls>\n", "tool_calls> \n", 1),
    ),
    # parse_tool_calls: invoke without a name="..." attribute
    "tool_name": (
        ValueError, "chat", _DSV4_BLOCK.replace('name="get_weather"', 'nme="x"'),
    ),
    # parse_tool_calls: a JSON body instead of DSML parameters is not DSML
    "json_invoke_body": (
        ValueError, "chat",
        _DSV4_BLOCK.replace(_DSV4_PARAM, '{"city": "Paris"}\n'),
    ),
    # parse_tool_calls: parameter without its string="..." attribute
    "parameter": (ValueError, "chat", _DSV4_BLOCK.replace(' string="true"', "")),
    # parse_tool_calls: the same parameter twice
    "duplicate_parameter": (
        ValueError, "chat", _DSV4_BLOCK.replace(_DSV4_PARAM, _DSV4_PARAM * 2),
    ),
    # parse_tool_calls: parameter close not followed by '>\n'
    "parameter_separator": (
        ValueError, "chat",
        _DSV4_BLOCK.replace(f"parameter>\n</{_DSML}invoke", f"parameter> </{_DSML}invoke"),
    ),
    # 'Unexpected content after tool calls'
    "content_after_calls": (AssertionError, "chat", _DSV4_BLOCK + "\nDone."),
    "thinking_content_after_calls": (
        AssertionError, "thinking", "Warm.</think>" + _DSV4_BLOCK + "\nDone.",
    ),
    # 'Unexpected content at end': text after an end-of-sentence token
    "content_after_eos": (
        AssertionError, "chat", "It is<｜end▁of▁sentence｜> 21 C.",
    ),
    "calls_then_content_after_eos": (
        AssertionError, "chat", _DSV4_BLOCK + "<｜end▁of▁sentence｜>Done.",
    ),
    # 'Invalid thinking format: missing </think>': calls inside the reasoning
    "calls_inside_reasoning": (
        AssertionError, "thinking", "Warm." + _DSV4_BLOCK + "</think>Done.",
    ),
    # 'Unexpected special token ... in content'
    "think_open_in_content": (AssertionError, "chat", "<think>It is 21 C."),
    "think_open_in_reasoning": (
        AssertionError, "thinking", "<think>Warm.</think>It is 21 C.",
    ),
    "second_think_close": (
        AssertionError, "thinking", "Warm.</think>It is</think> 21 C.",
    ),
    "second_think_close_before_calls": (
        AssertionError, "thinking", "Warm.</think>Sunny.</think>" + _DSV4_BLOCK,
    ),
    "bos_in_content": (AssertionError, "chat", "<｜begin▁of▁sentence｜>It is 21 C."),
    "dsml_in_content": (AssertionError, "chat", _DSV4_BLOCK[1:]),
    "dsml_in_thinking_content": (
        AssertionError, "thinking", "Warm.</think>" + _DSV4_BLOCK[1:],
    ),
}


@pytest.mark.parametrize("case", sorted(DEEPSEEK_V4_REJECTED))
def test_deepseek_v4_rejected_text_is_returned_as_content(
    case, registered_tokenizers, tmp_path,
):
    """``BaseTokenizer.parse_tool_calls``: text that is not a well-formed
    tool-call message comes back as content with no calls; no parser
    rejection escapes ``parse_tool_calls`` or the server's ``_parse_output``."""
    from batchgen.models.deepseek.deepseekv4_flash.assets.encoding.encoding_dsv4 import (
        eos_token,
        parse_message_from_completion_text,
    )

    rejection, thinking_mode, text = DEEPSEEK_V4_REJECTED[case]
    tool_case = TOOL_CALL_CASES["deepseek_v4"]
    tokenizer = _parser_only(registered_tokenizers["deepseek_v4"], tool_case)
    with pytest.raises(rejection):
        parse_message_from_completion_text(
            text + eos_token, thinking_mode=thinking_mode,
        )

    assert tokenizer.parse_tool_calls(text) == (None, text)

    scheduler = _parsing_scheduler(tmp_path, tokenizer)
    choice = _choice(scheduler, tmp_path, text, "stop")
    assert choice["finish_reason"] == "stop"
    assert "tool_calls" not in choice["message"]
    assert choice["message"]["content"] == text

    choice = _dsv4_choice(tokenizer, tmp_path, text, "stop", parse_thinking=True)
    assert choice["finish_reason"] == "stop"
    assert "tool_calls" not in choice["message"]


# ------------------------------------------- DeepSeek-V4 thinking-mode output

# The V4 encoder ends a thinking-mode prompt with ``<think>`` and a chat-mode
# one with ``</think>`` (encoding_dsv4.render_message), so a thinking-mode
# completion is ``reasoning</think>`` + what a chat-mode completion holds
# (content, then an optional "\n\n"-opened DSML tool-call block).
_DSV4_EOS = "<｜end▁of▁sentence｜>"
_DSV4_REASONING = "The user is in Paris; look up the weather."
_DSV4_CALL = [{"city": "Paris"}]

# reasoning / content: the message with parse_thinking and parse_tool_call on.
# raw_content: the content with only parse_tool_call on (the scheduler
# default of these tests), where the reasoning is not split out.
# vendored_mode: the mode the vendored parser accepts the text in, as the
# ground truth for reasoning, content and calls (None: no mode accepts it).
DEEPSEEK_V4_OUTPUTS = {
    # (a) thinking mode, reasoning + answer
    "thinking_answer": dict(
        text=f"{_DSV4_REASONING}</think>It is 21 C.",
        reasoning=_DSV4_REASONING, content="It is 21 C.", calls=None,
        raw_content=f"{_DSV4_REASONING}</think>It is 21 C.",
        vendored_mode="thinking",
    ),
    # (b) thinking mode, reasoning + tool call, with and without content
    "thinking_tool_call": dict(
        text=f"{_DSV4_REASONING}</think>{_DSV4_BLOCK}",
        reasoning=_DSV4_REASONING, content="", calls=_DSV4_CALL,
        raw_content=f"{_DSV4_REASONING}</think>",
        vendored_mode="thinking",
    ),
    "thinking_answer_tool_call": dict(
        text=f"{_DSV4_REASONING}</think>Checking.{_DSV4_BLOCK}",
        reasoning=_DSV4_REASONING, content="Checking.", calls=_DSV4_CALL,
        raw_content=f"{_DSV4_REASONING}</think>Checking.",
        vendored_mode="thinking",
    ),
    # (c) thinking mode cut off before </think>: the text alone cannot tell
    # it from a chat answer (the prompt's <think> is not in it), so no
    # reasoning block is found and the text stays content.
    "thinking_truncated": dict(
        text="The user is in Par",
        reasoning=None, content="The user is in Par", calls=None,
        raw_content="The user is in Par",
        vendored_mode=None,
    ),
    # (d) chat mode
    "chat_answer": dict(
        text="It is 21 C.",
        reasoning=None, content="It is 21 C.", calls=None,
        raw_content="It is 21 C.",
        vendored_mode="chat",
    ),
    "chat_tool_call": dict(
        text=_DSV4_BLOCK,
        reasoning=None, content="", calls=_DSV4_CALL, raw_content="",
        vendored_mode="chat",
    ),
    # (e) a chat-mode completion with a stray </think>. The chat grammar
    # rejects it; it is byte-identical to a thinking-mode completion, the only
    # grammar that accepts it, and is parsed as one.
    "chat_stray_think_close": dict(
        text="Warm.</think>It is 21 C.",
        reasoning="Warm.", content="It is 21 C.", calls=None,
        raw_content="Warm.</think>It is 21 C.",
        vendored_mode="thinking",
    ),
}


def _dsv4_choice(tokenizer, tmp_path, text, finish_reason, parse_thinking):
    """One chat completion through the real writer and ``_parse_output``."""
    out_dir = tmp_path / f"parse_thinking_{parse_thinking}"
    out_dir.mkdir(exist_ok=True)
    scheduler = _parsing_scheduler(out_dir, tokenizer)
    scheduler.server_args.parse_thinking = parse_thinking
    return _choice(scheduler, out_dir, text, finish_reason)


def _arguments(calls):
    return [json.loads(call["function"]["arguments"]) for call in calls]


def test_deepseek_v4_prompt_opens_the_reasoning_block_only_in_thinking_mode(
    registered_tokenizers,
):
    tokenizer = _parser_only(
        registered_tokenizers["deepseek_v4"], TOOL_CALL_CASES["deepseek_v4"],
    )
    messages = [{"role": "user", "content": "Weather in Paris?"}]

    thinking = tokenizer.apply_chat_template(messages, enable_thinking=True)
    chat = tokenizer.apply_chat_template(messages, enable_thinking=False)

    assert thinking.endswith("<｜Assistant｜><think>")
    assert chat.endswith("<｜Assistant｜></think>")


@pytest.mark.parametrize("case", sorted(DEEPSEEK_V4_OUTPUTS))
def test_deepseek_v4_output_is_split_as_the_v4_grammar_reads_it(
    case, registered_tokenizers, tmp_path,
):
    """``BaseTokenizer.parse_thinking`` then ``parse_tool_calls`` (the
    scheduler's order) recover reasoning, content and calls of every V4
    completion shape, matching the vendored parser in the mode it accepts."""
    from batchgen.models.deepseek.deepseekv4_flash.assets.encoding.encoding_dsv4 import (
        parse_message_from_completion_text,
    )

    expected = DEEPSEEK_V4_OUTPUTS[case]
    text = expected["text"]
    tokenizer = _parser_only(
        registered_tokenizers["deepseek_v4"], TOOL_CALL_CASES["deepseek_v4"],
    )
    finish_reason = "length" if case == "thinking_truncated" else "stop"

    if expected["vendored_mode"] is not None:
        truth = parse_message_from_completion_text(
            text + _DSV4_EOS, thinking_mode=expected["vendored_mode"],
        )
        assert (truth["reasoning_content"] or None) == expected["reasoning"]
        assert truth["content"] == expected["content"]
        assert _arguments(truth["tool_calls"]) == (expected["calls"] or [])

    reasoning, visible = tokenizer.parse_thinking(text)
    assert reasoning == expected["reasoning"]
    calls, content = tokenizer.parse_tool_calls(visible)
    assert content == expected["content"]
    assert (calls and _arguments(calls)) == expected["calls"]

    calls, raw_content = tokenizer.parse_tool_calls(text)
    assert raw_content == expected["raw_content"]
    assert (calls and _arguments(calls)) == expected["calls"]

    tool_reason = "tool_calls" if expected["calls"] else finish_reason
    choice = _dsv4_choice(tokenizer, tmp_path, text, finish_reason, True)
    message = choice["message"]
    assert message["content"] == expected["content"]
    assert message.get("reasoning_content") == expected["reasoning"]
    assert _arguments(message.get("tool_calls", [])) == (expected["calls"] or [])
    assert choice["finish_reason"] == tool_reason

    choice = _dsv4_choice(tokenizer, tmp_path, text, finish_reason, False)
    message = choice["message"]
    assert message["content"] == expected["raw_content"]
    assert "reasoning_content" not in message
    assert _arguments(message.get("tool_calls", [])) == (expected["calls"] or [])
    assert choice["finish_reason"] == tool_reason


def test_deepseek_v4_json_non_string_parameter_is_parsed(registered_tokenizers):
    tokenizer = _parser_only(
        registered_tokenizers["deepseek_v4"], TOOL_CALL_CASES["deepseek_v4"],
    )
    days = f'<{_DSML}parameter name="days" string="false">3</{_DSML}parameter>\n'
    text = _DSV4_BLOCK.replace(_DSV4_PARAM, _DSV4_PARAM + days)

    calls, content = tokenizer.parse_tool_calls(text)

    assert _arguments(calls) == [{"city": "Paris", "days": 3}]
    assert content == ""


@pytest.mark.parametrize("reasoning", [None, _DSV4_REASONING])
def test_deepseek_v4_call_with_non_json_arguments_is_content(
    reasoning, registered_tokenizers, tmp_path,
):
    """A ``string="false"`` value must be JSON (the DSML tool prompt); the
    vendored parser pastes it into the arguments unchecked. Such a call is
    malformed and stays content, as every other rejected call does."""
    from batchgen.models.deepseek.deepseekv4_flash.assets.encoding.encoding_dsv4 import (
        parse_message_from_completion_text,
    )

    block = _DSV4_BLOCK.replace('string="true">Paris', 'string="false">Paris')
    text = block if reasoning is None else f"{reasoning}</think>{block}"
    tokenizer = _parser_only(
        registered_tokenizers["deepseek_v4"], TOOL_CALL_CASES["deepseek_v4"],
    )
    mode = "chat" if reasoning is None else "thinking"
    parsed = parse_message_from_completion_text(text + _DSV4_EOS, thinking_mode=mode)
    arguments = parsed["tool_calls"][0]["function"]["arguments"]
    assert arguments == '{"city": Paris}'
    with pytest.raises(ValueError):
        json.loads(arguments)

    assert tokenizer.parse_tool_calls(text) == (None, text)

    choice = _dsv4_choice(tokenizer, tmp_path, text, "stop", False)
    assert choice["finish_reason"] == "stop"
    assert "tool_calls" not in choice["message"]
    assert choice["message"]["content"] == text

    choice = _dsv4_choice(tokenizer, tmp_path, text, "stop", True)
    assert choice["finish_reason"] == "stop"
    assert "tool_calls" not in choice["message"]
    assert choice["message"]["content"] == block
    assert choice["message"].get("reasoning_content") == reasoning


@pytest.mark.parametrize("family", FAMILIES)
def test_a_stopped_generation_with_a_tool_call_reports_tool_calls(
    family, registered_tokenizers, tmp_path,
):
    case = TOOL_CALL_CASES[family]
    tokenizer = _parser_only(registered_tokenizers[family], case)
    scheduler = _parsing_scheduler(tmp_path, tokenizer)

    choice = _choice(scheduler, tmp_path, case["complete"], "stop")

    assert choice["finish_reason"] == "tool_calls"
    calls = choice["message"]["tool_calls"]
    assert [call["function"]["name"] for call in calls] == ["get_weather"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}
    assert calls[0]["id"]


@pytest.mark.parametrize("family", FAMILIES)
def test_a_truncated_tool_call_stays_content_and_reports_length(
    family, registered_tokenizers, tmp_path,
):
    case = TOOL_CALL_CASES[family]
    tokenizer = _parser_only(registered_tokenizers[family], case)
    scheduler = _parsing_scheduler(tmp_path, tokenizer)

    choice = _choice(scheduler, tmp_path, case["truncated"], "length")

    assert choice["finish_reason"] == "length"
    assert "tool_calls" not in choice["message"]
    assert "get_weather" in choice["message"]["content"]


@pytest.mark.parametrize("finish_reason", ["length", "repetition"])
@pytest.mark.parametrize("family", FAMILIES)
def test_a_parsed_call_keeps_a_truncation_or_repetition_reason(
    family, finish_reason, registered_tokenizers, tmp_path,
):
    """The budget ran out (or repetition was cut) right at a complete call."""
    case = TOOL_CALL_CASES[family]
    tokenizer = _parser_only(registered_tokenizers[family], case)
    scheduler = _parsing_scheduler(tmp_path, tokenizer)

    choice = _choice(scheduler, tmp_path, case["complete"], finish_reason)

    assert choice["finish_reason"] == finish_reason
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"


@pytest.mark.parametrize("family", FAMILIES)
def test_a_stopped_generation_without_a_tool_call_reports_stop(
    family, registered_tokenizers, tmp_path,
):
    case = TOOL_CALL_CASES[family]
    tokenizer = _parser_only(registered_tokenizers[family], case)
    scheduler = _parsing_scheduler(tmp_path, tokenizer)

    choice = _choice(scheduler, tmp_path, "It is 21 C in Paris.", "stop")

    assert choice["finish_reason"] == "stop"
    assert "tool_calls" not in choice["message"]
    assert choice["message"]["content"] == "It is 21 C in Paris."


def test_tool_call_parsing_off_keeps_stop(registered_tokenizers, tmp_path):
    case = TOOL_CALL_CASES["gpt_oss"]
    tokenizer = _parser_only(registered_tokenizers["gpt_oss"], case)
    scheduler = _parsing_scheduler(tmp_path, tokenizer)
    scheduler.server_args.parse_tool_call = False

    choice = _choice(scheduler, tmp_path, case["complete"], "stop")

    assert choice["finish_reason"] == "stop"
    assert "tool_calls" not in choice["message"]


def test_text_completions_report_the_worker_finish_reason(
    registered_tokenizers, tmp_path,
):
    """/v1/completions returns raw text; nothing is parsed into tool calls."""
    case = TOOL_CALL_CASES["gpt_oss"]
    tokenizer = _parser_only(registered_tokenizers["gpt_oss"], case)
    scheduler = _parsing_scheduler(tmp_path, tokenizer, url="/v1/completions")

    choice = _choice(scheduler, tmp_path, case["complete"], "stop")

    assert choice["finish_reason"] == "stop"
    assert choice["text"] == case["complete"]
