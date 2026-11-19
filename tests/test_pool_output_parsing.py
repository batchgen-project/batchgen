"""Pool-mode output must honor the same parser flags as legacy batch mode.

It must also keep a per-request failure a FAILURE: a worker that fails one
request (over-context prompt, a reservation no Host pool can ever hold)
publishes an ``error`` on the completion, and the batch output line has to be
the Batch API error item, not a 200 with empty text.

A chat completion whose parsed message carries tool calls reports
``finish_reason="tool_calls"`` when the worker stopped on its own, for every
registered model family's real tool-call parser; truncation and repetition
keep the worker's reason.
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
