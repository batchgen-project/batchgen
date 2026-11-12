"""GPT-OSS harmony prompt rendering: tools, tool-call history, tool results.

The expected texts below are the harmony reference renderer's output for the
same conversations and schemas (the format GPT-OSS reads its tools in). Covered:

* a conversation without tools renders byte-identically to the previous
  Jinja template (pinned by rendering that template here);
* ``tools`` render as the developer message's ``functions`` namespace, with
  JSON-schema parameters converted to the TypeScript-like declarations;
* schemas outside the supported subset degrade to ``any`` plus the raw JSON
  schema in a comment, never to a misleading type;
* assistant ``tool_calls`` history and ``tool`` results render as harmony
  call / result messages, and the rendered call re-parses through the
  output parser to the same call;
* the real tokenizer assets encode every harmony marker as one special token;
* the scheduler's prompt seam forwards the request's ``tools`` to the
  renderer.
"""

import __future__
import ast
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

import numpy as np
import pytest
import tiktoken

from batchgen.config import tokenizer_assets as ta
from batchgen.models.openai.gpt_oss_120b import tokenizer as gpt_oss_tok
from batchgen.models.openai.gpt_oss_120b.harmony_tools import (
    render_functions_namespace,
)
from batchgen.models.openai.gpt_oss_120b.tokenizer import (
    GPT_OSS_SPECIAL_TOKENS,
    GPTOssTokenizer,
    HarmonyMessage,
    parse_harmony,
    render_harmony_prompt,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "batchgen" / "models" / "openai" / "gpt_oss_120b"
SCHEDULER = ROOT / "batchgen" / "server" / "batch_scheduler.py"
IO_STRUCT = ROOT / "batchgen" / "server" / "io_struct.py"

SYSTEM_SUFFIX = (
    "\n\n# Valid channels: analysis, commentary, final. "
    "Channel must be included for every message."
)
TOOL_NOTE = "\nCalls to these tools must go to the commentary channel: 'functions'."

# The chat template GPT-OSS rendered with before tools were supported.
PREVIOUS_TEMPLATE = (
    "{% for message in messages %}{% if message.role == 'system' %}"
    "<|start|>system<|message|>{{ message.content }}\n\n# Valid channels: "
    "analysis, commentary, final. Channel must be included for every message."
    "<|end|>{% elif message.role == 'user' %}<|start|>user<|message|>"
    "{{ message.content }}<|end|>{% elif message.role == 'assistant' %}"
    "<|start|>assistant<|channel|>final<|message|>{{ message.content }}<|end|>"
    "{% endif %}{% endfor %}{% if add_generation_prompt %}<|start|>assistant"
    "{% endif %}"
)


def _tokenizer() -> GPTOssTokenizer:
    """A tokenizer whose string rendering runs without the staged encoding."""
    return GPTOssTokenizer.__new__(GPTOssTokenizer)


def _tool(name, description, parameters=None, *, with_parameters=True):
    function = {"name": name, "description": description}
    if with_parameters:
        function["parameters"] = parameters
    return {"type": "function", "function": function}


def _declaration(parameters, description=""):
    """The one declaration a single tool renders to, without the section."""
    text = render_functions_namespace([_tool("f", description, parameters)])
    head = "# Tools\n\n## functions\n\nnamespace functions {\n\n"
    tail = "\n\n} // namespace functions"
    assert text.startswith(head) and text.endswith(tail)
    return text[len(head):-len(tail)]


WEATHER_TOOL = _tool(
    "get_weather",
    "Get the weather.",
    {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["c", "f"]},
            "days": {"type": "integer", "description": "Forecast days", "default": 1},
            "precise": {"type": "boolean"},
            "ratio": {"type": "number"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "nums": {"type": "array", "items": {"type": "integer"}},
            "opt": {"type": ["string", "null"]},
            "nested": {"type": "object", "properties": {"a": {"type": "string"}}},
        },
        "required": ["location"],
    },
)

WEATHER_NAMESPACE = (
    "# Tools\n\n## functions\n\nnamespace functions {\n\n"
    "// Get the weather.\n"
    "type get_weather = (_: {\n"
    "// City name\n"
    "location: string,\n"
    'unit?: "c" | "f",\n'
    "// Forecast days\n"
    "days?: number, // default: 1\n"
    "precise?: boolean,\n"
    "ratio?: number,\n"
    "tags?: string[],\n"
    "nums?: number[],\n"
    "opt?: string | null,\n"
    "nested?: {\n"
    "    a?: string,\n"
    "    },\n"
    "}) => any;\n\n"
    "// Takes nothing.\n"
    "type no_args = () => any;\n\n"
    "// Empty object.\n"
    "type empty_obj = (_: {\n"
    "}) => any;\n\n"
    "} // namespace functions"
)

TOOLS = [
    WEATHER_TOOL,
    _tool("no_args", "Takes nothing.", with_parameters=False),
    _tool("empty_obj", "Empty object.", {"type": "object", "properties": {}}),
]


# --------------------------------------------------------------------------- #
# Conversations without tools render exactly as before
# --------------------------------------------------------------------------- #


NO_TOOL_CONVERSATIONS = [
    [{"role": "system", "content": "Reasoning: low"},
     {"role": "user", "content": "What is 6*7?"}],
    [{"role": "system", "content": "Reasoning: high\nYou are terse."},
     {"role": "user", "content": "hi"},
     {"role": "assistant", "content": "hello", "reasoning_content": "greet back"},
     {"role": "user", "content": "again\n\nwith <b>markup</b> and {{ braces }}"}],
    [{"role": "user", "content": "no system message"}],
    [{"role": "system", "content": ""},
     {"role": "user", "content": ""},
     {"role": "assistant", "content": ""}],
    [{"role": "system", "content": "Reasoning: medium"},
     {"role": "user", "content": "u1", "name": "alice"},
     {"role": "assistant", "content": "a1"},
     {"role": "system", "content": "late system"},
     {"role": "user", "content": "u2"}],
]


@pytest.mark.parametrize("messages", NO_TOOL_CONVERSATIONS)
@pytest.mark.parametrize("add_generation_prompt", [True, False])
def test_no_tools_render_matches_previous_template(messages, add_generation_prompt):
    jinja2 = pytest.importorskip("jinja2")
    previous = jinja2.Template(
        PREVIOUS_TEMPLATE, undefined=jinja2.StrictUndefined
    ).render(messages=messages, add_generation_prompt=add_generation_prompt)
    for tools in (None, []):
        assert render_harmony_prompt(
            messages, tools=tools, add_generation_prompt=add_generation_prompt
        ) == previous


def test_no_tools_golden():
    rendered = _tokenizer().apply_chat_template(
        [{"role": "system", "content": "Reasoning: low\nBe brief."},
         {"role": "user", "content": "hi"},
         {"role": "assistant", "content": "hello"},
         {"role": "user", "content": "bye"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,  # forwarded for every model; no effect here
    )
    assert rendered == (
        "<|start|>system<|message|>Reasoning: low\nBe brief." + SYSTEM_SUFFIX
        + "<|end|><|start|>user<|message|>hi<|end|>"
        "<|start|>assistant<|channel|>final<|message|>hello<|end|>"
        "<|start|>user<|message|>bye<|end|><|start|>assistant"
    )


# --------------------------------------------------------------------------- #
# Tools render as the developer message's functions namespace
# --------------------------------------------------------------------------- #


def test_tools_namespace_golden():
    assert render_functions_namespace(TOOLS) == WEATHER_NAMESPACE


def test_tools_render_into_system_and_developer_messages():
    rendered = render_harmony_prompt(
        [{"role": "system", "content": "Reasoning: low"},
         {"role": "user", "content": "Weather in Paris?"}],
        tools=TOOLS,
        add_generation_prompt=True,
    )
    assert rendered == (
        "<|start|>system<|message|>Reasoning: low" + SYSTEM_SUFFIX + TOOL_NOTE
        + "<|end|><|start|>developer<|message|>" + WEATHER_NAMESPACE
        + "<|end|><|start|>user<|message|>Weather in Paris?<|end|>"
        "<|start|>assistant"
    )


def test_tools_without_a_leading_system_message_open_the_prompt():
    rendered = render_harmony_prompt(
        [{"role": "user", "content": "q"}], tools=TOOLS[1:2],
    )
    assert rendered.startswith("<|start|>developer<|message|># Tools")
    assert TOOL_NOTE not in rendered
    assert rendered.endswith("<|end|><|start|>user<|message|>q<|end|>")


# Reference-renderer output for each schema (tool description empty).
SUPPORTED_SCHEMAS = {
    "empty_schema_is_any": ({}, "type f = (_: any) => any;"),
    "object_without_properties": (
        {"type": "object"}, "type f = (_: {\n}) => any;"),
    "top_level_scalar": ({"type": "string"}, "type f = (_: string) => any;"),
    "top_level_array": (
        {"type": "array", "items": {"type": "string"}},
        "type f = (_: string[]) => any;"),
    "top_level_description": (
        {"type": "object", "description": "Top level.", "title": "T",
         "properties": {"a": {"type": "string"}}},
        "type f = (_: // Top level.\n{\na?: string,\n}) => any;"),
    "property_order_and_required": (
        {"type": "object", "properties": {"z": {"type": "string"}, "a": {"type": "string"}},
         "required": ["a", "not_a_property"], "additionalProperties": False},
        "type f = (_: {\nz?: string,\na: string,\n}) => any;"),
    "validation_keywords_ignored": (
        {"type": "object", "properties": {
            "s": {"type": "string", "format": "date", "minLength": 1},
            "n": {"type": "integer", "minimum": 0, "maximum": 10}}},
        "type f = (_: {\ns?: string,\nn?: number,\n}) => any;"),
    "scalar_type_list": (
        {"type": "object", "properties": {
            "t": {"type": ["string", "number", "boolean"]},
            "u": {"type": ["integer", "null"]}}},
        "type f = (_: {\nt?: string | number | boolean,\nu?: number | null,\n}) => any;"),
    "enum_with_description_and_default": (
        {"type": "object", "properties": {"u": {
            "type": "string", "enum": ["c", "f"], "description": "Unit", "default": "c"}}},
        'type f = (_: {\n// Unit\nu?: "c" | "f", // default: c\n}) => any;'),
    "defaults": (
        {"type": "object", "properties": {
            "s": {"type": "string", "default": "hi"},
            "b": {"type": "boolean", "default": False},
            "n": {"type": "number", "default": 1.5},
            "i": {"type": "integer", "default": -3},
            "l": {"type": "array", "items": {"type": "string"}, "default": ["x", "y"]},
            "z": {"type": "string", "default": None},
            "q": {"type": "number", "default": "3"}}},
        "type f = (_: {\n"
        's?: string, // default: "hi"\n'
        "b?: boolean, // default: false\n"
        "n?: number, // default: 1.5\n"
        "i?: number, // default: -3\n"
        'l?: string[], // default: ["x","y"]\n'
        "z?: string, // default: null\n"
        'q?: number, // default: "3"\n'
        "}) => any;"),
    "float_and_object_defaults": (
        {"type": "object", "properties": {
            "a": {"type": "number", "default": 1e20},
            "b": {"type": "number", "default": 0.1},
            "c": {"type": "object", "default": {"k": [1, "é"]}}}},
        "type f = (_: {\na?: number, // default: 1e20\nb?: number, // default: 0.1\n"
        'c?: {\n    }, // default: {"k":[1,"é"]}\n}) => any;'),
    "required_with_default": (
        {"type": "object", "properties": {"a": {"type": "string", "default": "x"}},
         "required": ["a"]},
        'type f = (_: {\na: string, // default: "x"\n}) => any;'),
    "arrays": (
        {"type": "object", "properties": {
            "xs": {"type": "array"},
            "ys": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "zs": {"type": "array", "items": {"type": "string", "description": "an item"}},
            "es": {"type": "array", "items": {"type": "string", "enum": ["x"]},
                   "default": ["x"]}}},
        "type f = (_: {\nxs?: Array<any>,\nys?: number[][],\nzs?: string[],\n"
        'es?: "x"[], // default: ["x"]\n}) => any;'),
    "array_of_objects": (
        {"type": "object", "properties": {"xs": {
            "type": "array", "items": {"type": "object", "properties": {"k": {"type": "string"}}}}}},
        "type f = (_: {\nxs?: {\n    k?: string,\n    }[],\n}) => any;"),
    "array_of_described_objects": (
        {"type": "object", "properties": {"xs": {
            "type": "array", "description": "xs desc",
            "items": {"type": "object", "description": "item",
                      "properties": {"k": {"type": "string"}}}}}},
        "type f = (_: {\n// xs desc\nxs?:     // item\n{\n    k?: string,\n    }[],\n}) => any;"),
    "nested_objects": (
        {"type": "object", "properties": {"o": {"type": "object", "properties": {
            "p": {"type": "object", "properties": {"q": {"type": "string"}}}}}}},
        "type f = (_: {\no?: {\n    p?: {\n        q?: string,\n        },\n    },\n}) => any;"),
    "nested_object_with_descriptions": (
        {"type": "object", "properties": {"o": {
            "type": "object", "description": "outer",
            "properties": {"a": {"type": "string", "description": "inner a"},
                           "b": {"type": "integer"}},
            "required": ["a"]}}, "required": ["o"]},
        "type f = (_: {\n// outer\no:     // outer\n{\n    // inner a\n    a: string,\n"
        "    b?: number,\n    },\n}) => any;"),
    "described_object_two_levels_down": (
        {"type": "object", "properties": {"o": {"type": "object", "properties": {
            "p": {"type": "object", "description": "pp",
                  "properties": {"q": {"type": "string", "description": "qq"}}}}}}},
        "type f = (_: {\no?: {\n    // pp\n    p?:         // pp\n{\n        // qq\n"
        "        q?: string,\n        },\n    },\n}) => any;"),
    "nested_required_default_enum": (
        {"type": "object", "properties": {"o": {"type": "object", "properties": {
            "u": {"type": "string", "enum": ["a"], "default": "a"},
            "n": {"type": "integer", "default": 3}}, "required": ["u"]}}},
        'type f = (_: {\no?: {\n    u: "a", // default: a\n    n?: number, // default: 3\n'
        "    },\n}) => any;"),
    "described_object_with_default": (
        {"type": "object", "properties": {"o": {
            "type": "object", "description": "outer", "default": {},
            "properties": {"a": {"type": "string"}}}}},
        "type f = (_: {\n// outer\no?:     // outer\n{\n    a?: string,\n    }, // default: {}\n"
        "}) => any;"),
    "titles_and_examples": (
        {"type": "object", "properties": {
            "a": {"type": "string", "title": "A"},
            "b": {"type": "string", "title": "B", "description": "desc"},
            "c": {"type": "string", "examples": ["x", 1]},
            "d": {"type": "string", "description": "d", "examples": ["y"]},
            "e": {"type": "string", "examples": []}}},
        "type f = (_: {\n// A\n//\na?: string,\n// B\n//\n// desc\nb?: string,\n"
        '// Examples:\n// - "x"\nc?: string,\n// d\n// Examples:\n// - "y"\nd?: string,\n'
        "e?: string,\n}) => any;"),
    "titled_nested_object": (
        {"type": "object", "properties": {"o": {
            "type": "object", "title": "O", "description": "od",
            "properties": {"a": {"type": "string", "title": "A"}}}}},
        "type f = (_: {\n// O\n//\n// od\no?:     // od\n{\n    // A\n    //\n"
        "    a?: string,\n    },\n}) => any;"),
    "text_is_verbatim": (
        {"type": "object", "properties": {
            "a": {"type": "string", "description": "温度 °C"},
            "b": {"type": "string", "enum": ['he said "x"', "b"]},
            "c": {"type": "string", "description": ""}}},
        'type f = (_: {\n// 温度 °C\na?: string,\nb?: "he said "x"" | "b",\n'
        "// \nc?: string,\n}) => any;"),
}


@pytest.mark.parametrize("case", sorted(SUPPORTED_SCHEMAS))
def test_supported_schema_matches_reference_rendering(case):
    schema, expected = SUPPORTED_SCHEMAS[case]
    assert _declaration(schema) == expected


def test_function_descriptions_comment_every_line():
    text = render_functions_namespace([
        _tool("a", "", {"type": "object", "properties": {"x": {"type": "string"}}}),
        _tool("b", "Multi\nline desc.", with_parameters=False),
    ])
    assert text.endswith(
        "{\n\ntype a = (_: {\nx?: string,\n}) => any;\n\n"
        "// Multi\n// line desc.\ntype b = () => any;\n\n} // namespace functions"
    )


def test_multiline_property_description_stays_inside_comments():
    # Deliberate difference from the reference, which leaves "line2" as bare
    # text inside the type declaration.
    assert _declaration({"type": "object", "properties": {
        "s": {"type": "string", "description": "line1\nline2"}}}) == (
        "type f = (_: {\n// line1\n// line2\ns?: string,\n}) => any;"
    )


def _schema_comment(schema, indent=""):
    return f"{indent}// JSON schema: {json.dumps(schema, ensure_ascii=False)}\n"


EXOTIC_PROPERTY_SCHEMAS = {
    "anyOf": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    "oneOf": {"oneOf": [{"type": "string"}, {"type": "number"}]},
    "allOf": {"allOf": [{"type": "string"}]},
    "ref": {"$ref": "#/$defs/X"},
    "const": {"const": "x"},
    "enum_without_type": {"enum": ["a", "b"]},
    "integer_enum": {"type": "integer", "enum": [1, 2, 3]},
    "nullable_enum": {"type": ["string", "null"], "enum": ["a", None]},
    "empty_enum": {"type": "string", "enum": []},
    "null_only": {"type": "null"},
    "type_list_with_object": {"type": ["object", "null"],
                              "properties": {"a": {"type": "string"}}},
    "type_list_with_array": {"type": ["array", "null"], "items": {"type": "string"}},
    "tuple_items": {"type": "array", "items": [{"type": "string"}, {"type": "integer"}]},
    "array_of_exotic_items": {"type": "array", "items": {"anyOf": [{"type": "string"}]}},
    "map_object": {"type": "object", "additionalProperties": {"type": "string"}},
    "unknown_keyword": {"type": "string", "nullable": True},
    "unknown_type": {"type": "file"},
    "non_string_title": {"type": "string", "title": 3},
    "not_a_schema": True,
}


@pytest.mark.parametrize("case", sorted(EXOTIC_PROPERTY_SCHEMAS))
def test_exotic_property_degrades_to_any_with_its_schema(case):
    schema = EXOTIC_PROPERTY_SCHEMAS[case]
    declaration = _declaration({
        "type": "object",
        "properties": {"ok": {"type": "string"}, "x": schema},
        "required": ["x"],
    })
    assert declaration == (
        "type f = (_: {\nok?: string,\n" + _schema_comment(schema) + "x: any,\n}) => any;"
    )


def test_exotic_nested_property_degrades_only_itself():
    schema = {"oneOf": [{"type": "string"}, {"type": "number"}]}
    assert _declaration({"type": "object", "properties": {"o": {
        "type": "object", "properties": {"choice": dict(schema, description="d"),
                                         "k": {"type": "string"}}}}}) == (
        "type f = (_: {\no?: {\n    // d\n"
        + _schema_comment(dict(schema, description="d"), "    ")
        + "    choice?: any,\n    k?: string,\n    },\n}) => any;"
    )


def test_exotic_top_level_parameters_degrade_the_whole_argument():
    parameters = {
        "type": "object",
        "properties": {"x": {"$ref": "#/$defs/X"}},
        "$defs": {"X": {"type": "string"}},
    }
    assert _declaration(parameters, description="Uses refs.") == (
        "// Uses refs.\n" + _schema_comment(parameters) + "type f = (_: any) => any;"
    )


@pytest.mark.parametrize(
    "tools, message",
    [
        ({"type": "function"}, "tools must be a list"),
        ([{"type": "code_interpreter"}], "only {\"type\": \"function\""),
        ([{"type": "function", "function": "f"}], "'function' must be an object"),
        ([{"type": "function", "function": {"description": "no name"}}],
         "function name must be a non-empty string"),
        ([_tool("get weather", "")], "contains whitespace"),
        ([_tool("f", ""), _tool("f", "")], "duplicate function name 'f'"),
        ([_tool("f", "", ["not", "a", "schema"])], "parameters must be a JSON-schema object"),
        ([{"type": "function", "function": {"name": "f", "description": 3}}],
         "description must be a string"),
    ],
)
def test_malformed_tools_are_rejected(tools, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        render_harmony_prompt([{"role": "user", "content": "q"}], tools=tools)


# --------------------------------------------------------------------------- #
# Tool-call history and tool results
# --------------------------------------------------------------------------- #

CALL_ARGS = '{"location": "Paris", "unit": "c"}'
TWO_TURN = [
    {"role": "system", "content": "Reasoning: low"},
    {"role": "user", "content": "Weather in Paris?"},
    {"role": "assistant", "content": "",
     "reasoning_content": "User asks weather. Use tool.",
     "tool_calls": [{"id": "call_1", "type": "function",
                     "function": {"name": "get_weather", "arguments": CALL_ARGS}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": '{"temp_c": 21}'},
]

CALL_MESSAGE = (
    "<|start|>assistant to=functions.get_weather<|channel|>commentary "
    "<|constrain|>json<|message|>" + CALL_ARGS + "<|call|>"
)
RESULT_MESSAGE = (
    "<|start|>functions.get_weather to=assistant<|channel|>commentary"
    '<|message|>{"temp_c": 21}<|end|>'
)


def test_two_turn_tool_conversation_golden():
    rendered = render_harmony_prompt(TWO_TURN, tools=[WEATHER_TOOL], add_generation_prompt=True)
    assert rendered == (
        "<|start|>system<|message|>Reasoning: low" + SYSTEM_SUFFIX + TOOL_NOTE + "<|end|>"
        "<|start|>developer<|message|>" + render_functions_namespace([WEATHER_TOOL]) + "<|end|>"
        "<|start|>user<|message|>Weather in Paris?<|end|>"
        "<|start|>assistant<|channel|>analysis<|message|>User asks weather. Use tool.<|end|>"
        + CALL_MESSAGE + RESULT_MESSAGE + "<|start|>assistant"
    )


def test_preamble_and_parallel_calls_render_in_order():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "Checking both.", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "f", "arguments": "{}"}},
            {"id": "b", "type": "function", "function": {"name": "g", "arguments": '{"x":1}'}},
        ]},
        {"role": "tool", "tool_call_id": "a", "content": "r"},
        {"role": "tool", "tool_call_id": "b", "name": "g", "content": "r2"},
    ]
    assert render_harmony_prompt(messages) == (
        "<|start|>user<|message|>q<|end|>"
        "<|start|>assistant<|channel|>commentary<|message|>Checking both.<|end|>"
        "<|start|>assistant to=functions.f<|channel|>commentary <|constrain|>json"
        "<|message|>{}<|call|>"
        "<|start|>assistant to=functions.g<|channel|>commentary <|constrain|>json"
        '<|message|>{"x":1}<|call|>'
        "<|start|>functions.f to=assistant<|channel|>commentary<|message|>r<|end|>"
        "<|start|>functions.g to=assistant<|channel|>commentary<|message|>r2<|end|>"
    )


def test_reasoning_of_a_finished_turn_is_dropped():
    finished = TWO_TURN + [
        {"role": "assistant", "content": "It is 21 C.", "reasoning_content": "Answer."},
        {"role": "user", "content": "Thanks. And Rome?"},
    ]
    rendered = render_harmony_prompt(finished, tools=[WEATHER_TOOL], add_generation_prompt=True)
    assert "<|channel|>analysis" not in rendered
    assert rendered.endswith(
        CALL_MESSAGE + RESULT_MESSAGE
        + "<|start|>assistant<|channel|>final<|message|>It is 21 C.<|end|>"
        "<|start|>user<|message|>Thanks. And Rome?<|end|><|start|>assistant"
    )


def test_tool_message_can_name_its_function_without_an_id():
    rendered = render_harmony_prompt(
        [{"role": "tool", "name": "lookup", "content": "x"}]
    )
    assert rendered == (
        "<|start|>functions.lookup to=assistant<|channel|>commentary<|message|>x<|end|>"
    )


@pytest.mark.parametrize(
    "messages, message",
    [
        ([{"role": "tool", "tool_call_id": "nope", "content": "x"}],
         "tool_call_id 'nope' matches no earlier assistant tool call"),
        ([{"role": "tool", "content": "x"}], "needs tool_call_id (or name)"),
        (TWO_TURN[:3] + [{"role": "tool", "tool_call_id": "call_1", "name": "other",
                          "content": "x"}],
         "name 'other' disagrees with tool_call_id 'call_1'"),
        ([{"role": "assistant", "tool_calls": [
            {"id": "a", "type": "function",
             "function": {"name": "f", "arguments": {"x": 1}}}]}],
         "function.arguments must be the JSON-encoded string"),
        ([{"role": "assistant", "tool_calls": [
            {"id": "a", "type": "custom", "function": {"name": "f", "arguments": "{}"}}]}],
         "only function tool calls are supported"),
        ([{"role": "assistant", "tool_calls": [
            {"id": "a", "function": {"name": "f g", "arguments": "{}"}}]}],
         "contains whitespace"),
        ([{"role": "developer", "content": "x"}], "role 'developer' is not supported"),
        ([{"role": "user"}], "'content' must be a string"),
        ([{"role": "assistant"}], "'content' must be a string"),
        ([{"role": "tool", "name": "f"}], "'content' must be a string"),
    ],
)
def test_unrenderable_conversations_are_rejected(messages, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        render_harmony_prompt(messages)


# --------------------------------------------------------------------------- #
# Renders stay parseable by the output parser
# --------------------------------------------------------------------------- #


def _harmony_messages(text) -> List[HarmonyMessage]:
    items = parse_harmony(text)
    assert all(isinstance(i, HarmonyMessage) for i in items), items
    return items


def test_rendered_call_reparses_to_the_same_tool_call():
    (message,) = _harmony_messages(CALL_MESSAGE)
    call = GPTOssTokenizer._tool_call(message)
    assert call["function"] == {"name": "get_weather", "arguments": CALL_ARGS}
    (result,) = _harmony_messages(RESULT_MESSAGE)
    assert (result.role, result.recipient, result.channel, result.body) == (
        "functions.get_weather", "assistant", "commentary", '{"temp_c": 21}',
    )


def test_parsed_completion_round_trips_through_history():
    """Turn 1 completion -> parsed tool_calls -> turn 2 history -> same call."""
    tok = _tokenizer()
    completion = (
        "<|channel|>analysis<|message|>User asks weather. Use tool.<|end|>"
        "<|start|>assistant<|channel|>commentary to=functions.get_weather "
        "<|constrain|>json<|message|>" + CALL_ARGS + "<|call|>"
    )
    reasoning, content = tok.parse_thinking(completion)
    calls, content = tok.parse_tool_calls(content)
    assistant = {"role": "assistant", "content": content,
                 "reasoning_content": reasoning, "tool_calls": calls}
    turn2 = TWO_TURN[:2] + [
        assistant,
        {"role": "tool", "tool_call_id": calls[0]["id"], "content": '{"temp_c": 21}'},
    ]
    rendered = tok.apply_chat_template(
        turn2, tokenize=False, add_generation_prompt=True, tools=[WEATHER_TOOL]
    )
    assert rendered == render_harmony_prompt(
        TWO_TURN, tools=[WEATHER_TOOL], add_generation_prompt=True
    )
    # The history after the user message is the harmony the model produced,
    # in the reference's canonical header order, and parses back to it.
    history = rendered.split("Weather in Paris?<|end|>", 1)[1]
    analysis, call, result, open_turn = parse_harmony(history)
    assert (analysis.kind, analysis.body) == ("reasoning", reasoning)
    assert GPTOssTokenizer._tool_call(call)["function"] == calls[0]["function"]
    assert result.role == "functions.get_weather"
    assert open_turn.role == "assistant" and not open_turn.header_complete


# --------------------------------------------------------------------------- #
# Tokenizer round trip on the real asset files
# --------------------------------------------------------------------------- #


def _asset_tokenizer():
    tokenizers = pytest.importorskip("tokenizers")
    return tokenizers.Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))


def test_rendered_markers_are_single_special_tokens_in_the_tokenizer_assets():
    asset = _asset_tokenizer()
    rendered = render_harmony_prompt(TWO_TURN, tools=TOOLS, add_generation_prompt=True)
    ids = asset.encode(rendered, add_special_tokens=False).ids
    assert asset.decode(ids, skip_special_tokens=False) == rendered
    marker_ids = {i for i in ids if i in set(GPT_OSS_SPECIAL_TOKENS.values())}
    expected = {GPT_OSS_SPECIAL_TOKENS[m] for m in (
        "<|start|>", "<|message|>", "<|end|>", "<|channel|>", "<|constrain|>", "<|call|>")}
    assert marker_ids == expected
    for marker, token_id in GPT_OSS_SPECIAL_TOKENS.items():
        if marker in ("<|start|>", "<|message|>", "<|end|>", "<|channel|>",
                      "<|constrain|>", "<|call|>"):
            assert ids.count(token_id) == rendered.count(marker), marker


@pytest.mark.skipif(
    not ta.asset_path(ta.O200K_BASE).exists(),
    reason="o200k_base is not staged in this environment "
    "(python -m batchgen.config.tokenizer_assets stage)",
)
def test_serving_tokenizer_ids_match_the_tokenizer_assets():
    asset = _asset_tokenizer()
    tok = GPTOssTokenizer()
    ids = tok.apply_chat_template(
        TWO_TURN, tokenize=True, add_generation_prompt=True, tools=TOOLS
    )
    rendered = render_harmony_prompt(TWO_TURN, tools=TOOLS, add_generation_prompt=True)
    assert ids == asset.encode(rendered, add_special_tokens=False).ids
    assert tok.decode(ids) == rendered


# --------------------------------------------------------------------------- #
# The scheduler's prompt seam forwards the request's tools
# --------------------------------------------------------------------------- #

SEAM_METHODS = ("_construct_prompt", "_inject_reasoning_effort", "_chat_prompt_ids")
SEAM_FUNCTIONS = ("completion_prompt_to_text", "prompt_ids_array")


def _load_io_struct():
    """io_struct by path: importing batchgen.server pulls in uvicorn."""
    spec = importlib.util.spec_from_file_location("_gpt_oss_tmpl_io_struct", IO_STRUCT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _isolated_prompt_seam(io_struct):
    tree = ast.parse(SCHEDULER.read_text(), filename=str(SCHEDULER))
    scheduler = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchScheduler"
    )
    methods = [n for n in scheduler.body
               if isinstance(n, ast.FunctionDef) and n.name in SEAM_METHODS]
    assert {n.name for n in methods} == set(SEAM_METHODS)
    functions = [n for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name in SEAM_FUNCTIONS]
    assert {n.name for n in functions} == set(SEAM_FUNCTIONS)
    module = ast.Module(
        body=functions + [ast.ClassDef(
            name="IsolatedPromptSeam", bases=[], keywords=[],
            body=methods, decorator_list=[],
        )],
        type_ignores=[],
    )
    namespace = {
        "Any": Any, "List": List, "Optional": Optional, "np": np,
        "_INT32_MAX": np.iinfo(np.int32).max,
        "ChatCompletionRequest": io_struct.ChatCompletionRequest,
        "CompletionRequest": io_struct.CompletionRequest,
    }
    code = compile(
        ast.fix_missing_locations(module), str(SCHEDULER), "exec",
        flags=__future__.annotations.compiler_flag, dont_inherit=True,
    )
    exec(code, namespace)
    return namespace["IsolatedPromptSeam"]()


def _byte_level_encoding() -> tiktoken.Encoding:
    """Byte-level ranks plus the real GPT-OSS special tokens."""
    return tiktoken.Encoding(
        name="gpt_oss_template_test",
        pat_str=ta.O200K_BASE.pat_str,
        mergeable_ranks={bytes([b]): b for b in range(256)},
        special_tokens=dict(GPT_OSS_SPECIAL_TOKENS),
    )


def test_scheduler_prompt_seam_renders_request_tools(monkeypatch):
    monkeypatch.setattr(gpt_oss_tok, "_get_tiktoken_tokenizer", _byte_level_encoding)
    tok = GPTOssTokenizer()
    io_struct = _load_io_struct()
    seam = _isolated_prompt_seam(io_struct)
    body = io_struct.ChatCompletionRequest(
        model="openai/gpt-oss-120b",
        messages=TWO_TURN[1:],
        tools=[WEATHER_TOOL],
        enable_thinking=True,
    )
    ids = seam._construct_prompt(SimpleNamespace(body=body, custom_id="r1"), tok)
    assert ids.dtype == np.int32
    # The scheduler adds the default reasoning level as the system message.
    expected = render_harmony_prompt(
        [{"role": "system", "content": "Reasoning: low"}] + TWO_TURN[1:],
        tools=[WEATHER_TOOL],
        add_generation_prompt=True,
    )
    assert expected == render_harmony_prompt(
        TWO_TURN, tools=[WEATHER_TOOL], add_generation_prompt=True
    )
    assert tok.decode(ids.tolist()) == expected


def test_scheduler_prompt_seam_rejects_an_unanswerable_tool_message(monkeypatch):
    monkeypatch.setattr(gpt_oss_tok, "_get_tiktoken_tokenizer", _byte_level_encoding)
    tok = GPTOssTokenizer()
    io_struct = _load_io_struct()
    seam = _isolated_prompt_seam(io_struct)
    body = io_struct.ChatCompletionRequest(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": "q"},
                  {"role": "tool", "tool_call_id": "ghost", "content": "x"}],
    )
    with pytest.raises(ValueError, match="request 'r1': the chat template rejected"):
        seam._construct_prompt(SimpleNamespace(body=body, custom_id="r1"), tok)
