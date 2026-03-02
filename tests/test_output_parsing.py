"""Tests for --parse-thinking and --parse-tool-call output parsing.

Tests the parsing regex logic directly, and the IO struct serialization.
Run with: python tests/test_output_parsing.py

For full integration tests (instantiating tokenizer classes), run on
a remote machine with all dependencies installed.
"""

import json
import re
import uuid
from typing import Optional


# ---- Extracted parsing logic (mirrors tokenizer methods) ----

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_DS_TOOL_RE = re.compile(
    r"<｜tool▁call▁begin｜>(.*?)<｜tool▁call▁end｜>", re.DOTALL
)
_HARMONY_CHANNEL_RE = re.compile(
    r"<\|channel\|>(.*?)<\|message\|>(.*?)(?=<\|channel\|>|<\|return\|>|$)",
    re.DOTALL,
)


def parse_thinking_r1(text: str) -> tuple[Optional[str], str]:
    m = _THINK_RE.search(text)
    if not m:
        return None, text
    reasoning = m.group(1).strip()
    visible = _THINK_RE.sub("", text, count=1).strip()
    return reasoning, visible


def parse_tool_calls_deepseek(text: str) -> tuple[Optional[list], str]:
    matches = _DS_TOOL_RE.findall(text)
    if not matches:
        return None, text
    tool_calls = []
    for raw in matches:
        lines = raw.strip().split("\n", 1)
        name = lines[0].strip()
        arguments = lines[1].strip() if len(lines) > 1 else "{}"
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        })
    visible = _DS_TOOL_RE.sub("", text).strip()
    return tool_calls, visible


def parse_thinking_harmony(text: str) -> tuple[Optional[str], str]:
    channels = {}
    for m in _HARMONY_CHANNEL_RE.finditer(text):
        channel_name = m.group(1).strip()
        channel_content = m.group(2).strip()
        channels[channel_name] = channel_content
    if not channels:
        return None, text
    reasoning = channels.get("analysis")
    visible = channels.get("final", text)
    return reasoning, visible


def parse_tool_calls_gptoss(text: str) -> tuple[Optional[list], str]:
    if "<|call|>" not in text:
        return None, text
    parts = text.split("<|call|>")
    visible = parts[0].strip()
    tool_calls = []
    for raw_call in parts[1:]:
        raw_call = raw_call.strip()
        for tok in ("<|return|>", "<|end|>"):
            if raw_call.endswith(tok):
                raw_call = raw_call[: -len(tok)].strip()
        try:
            payload = json.loads(raw_call)
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": payload.get("name", ""),
                    "arguments": json.dumps(
                        payload.get("parameters", payload.get("arguments", {}))
                    ),
                },
            })
        except json.JSONDecodeError:
            pass
    return tool_calls or None, visible


# ---- Tests ----

def test_r1_parse_thinking():
    # Basic
    text = "<think>Let me reason step by step.\nStep 1: ...</think>The answer is 42."
    reasoning, visible = parse_thinking_r1(text)
    assert reasoning == "Let me reason step by step.\nStep 1: ...", f"Got: {reasoning!r}"
    assert visible == "The answer is 42.", f"Got: {visible!r}"

    # No thinking
    reasoning, visible = parse_thinking_r1("Just a plain answer.")
    assert reasoning is None
    assert visible == "Just a plain answer."

    # Empty thinking
    reasoning, visible = parse_thinking_r1("<think></think>Answer")
    assert reasoning == ""
    assert visible == "Answer"

    # Thinking with trailing whitespace
    text = "<think>thought</think>\n\nAnswer here"
    reasoning, visible = parse_thinking_r1(text)
    assert reasoning == "thought"
    assert visible == "Answer here"

    print("  PASS: R1-style parse_thinking")


def test_deepseek_parse_tool_calls():
    # Single tool call
    text = 'Some text<｜tool▁call▁begin｜>get_weather\n{"city": "London"}<｜tool▁call▁end｜>'
    calls, visible = parse_tool_calls_deepseek(text)
    assert calls is not None
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "London"}
    assert calls[0]["id"].startswith("call_")
    assert visible == "Some text"

    # Multiple tool calls
    text = (
        '<｜tool▁call▁begin｜>fn1\n{"a": 1}<｜tool▁call▁end｜>'
        '<｜tool▁call▁begin｜>fn2\n{"b": 2}<｜tool▁call▁end｜>'
    )
    calls, visible = parse_tool_calls_deepseek(text)
    assert len(calls) == 2
    assert calls[0]["function"]["name"] == "fn1"
    assert calls[1]["function"]["name"] == "fn2"

    # No tool calls
    calls, visible = parse_tool_calls_deepseek("plain text")
    assert calls is None
    assert visible == "plain text"

    # Tool call without arguments
    text = '<｜tool▁call▁begin｜>no_args<｜tool▁call▁end｜>'
    calls, visible = parse_tool_calls_deepseek(text)
    assert calls[0]["function"]["name"] == "no_args"
    assert calls[0]["function"]["arguments"] == "{}"

    print("  PASS: DeepSeek parse_tool_calls")


def test_harmony_parse_thinking():
    # Analysis + final
    text = "<|channel|>analysis<|message|>Let me think...<|channel|>final<|message|>The answer is 42."
    reasoning, visible = parse_thinking_harmony(text)
    assert reasoning == "Let me think..."
    assert visible == "The answer is 42."

    # No channels
    reasoning, visible = parse_thinking_harmony("Plain text")
    assert reasoning is None
    assert visible == "Plain text"

    # Only final (no analysis)
    text = "<|channel|>final<|message|>Just the answer."
    reasoning, visible = parse_thinking_harmony(text)
    assert reasoning is None
    assert visible == "Just the answer."

    # With <|return|> at end
    text = "<|channel|>analysis<|message|>thinking<|channel|>final<|message|>answer<|return|>"
    reasoning, visible = parse_thinking_harmony(text)
    assert reasoning == "thinking"
    assert visible == "answer"

    print("  PASS: Harmony parse_thinking")


def test_gptoss_parse_tool_calls():
    # Tool call with <|call|>
    text = 'Some response<|call|>{"name": "search", "parameters": {"query": "hello"}}'
    calls, visible = parse_tool_calls_gptoss(text)
    assert calls is not None
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "search"
    assert json.loads(calls[0]["function"]["arguments"]) == {"query": "hello"}
    assert visible == "Some response"

    # No tool calls
    calls, visible = parse_tool_calls_gptoss("plain text")
    assert calls is None

    # Multiple tool calls
    text = 'text<|call|>{"name": "a", "parameters": {}}<|call|>{"name": "b", "parameters": {}}'
    calls, visible = parse_tool_calls_gptoss(text)
    assert len(calls) == 2

    print("  PASS: GPT-OSS parse_tool_calls")


def test_combined_thinking_and_tool_calls():
    """Test thinking + tool calls in same output (DeepSeek style)."""
    text = (
        '<think>I need to check the weather.</think>'
        'Let me check.<｜tool▁call▁begin｜>get_weather\n{"city": "London"}<｜tool▁call▁end｜>'
    )
    reasoning, after_think = parse_thinking_r1(text)
    assert reasoning == "I need to check the weather."

    calls, visible = parse_tool_calls_deepseek(after_think)
    assert calls is not None
    assert calls[0]["function"]["name"] == "get_weather"
    assert visible == "Let me check."

    print("  PASS: Combined thinking + tool calls")


def test_glm5_parse_tool_calls():
    """Test GLM-5 tool call parsing: <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>"""
    _GLM5_TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    _GLM5_ARG_RE = re.compile(
        r"<arg_key>(.*?)</arg_key><arg_value>(.*?)</arg_value>", re.DOTALL
    )

    def parse_tool_calls_glm5(text):
        matches = _GLM5_TOOL_RE.findall(text)
        if not matches:
            return None, text
        tool_calls = []
        for raw in matches:
            raw = raw.strip()
            arg_start = raw.find("<arg_key>")
            if arg_start == -1:
                name = raw
                arguments = {}
            else:
                name = raw[:arg_start].strip()
                arguments = {}
                for am in _GLM5_ARG_RE.finditer(raw):
                    key = am.group(1).strip()
                    val = am.group(2).strip()
                    try:
                        arguments[key] = json.loads(val)
                    except (ValueError, json.JSONDecodeError):
                        arguments[key] = val
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            })
        visible = _GLM5_TOOL_RE.sub("", text).strip()
        return tool_calls, visible

    # Single tool call
    text = '<tool_call>get_weather<arg_key>city</arg_key><arg_value>"London"</arg_value></tool_call>'
    calls, visible = parse_tool_calls_glm5(text)
    assert calls is not None
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "London"}
    assert visible == ""

    # Multiple args
    text = '<tool_call>search<arg_key>query</arg_key><arg_value>"hello"</arg_value><arg_key>limit</arg_key><arg_value>10</arg_value></tool_call>'
    calls, visible = parse_tool_calls_glm5(text)
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"query": "hello", "limit": 10}

    # With surrounding text
    text = 'Let me check the weather.<tool_call>get_weather<arg_key>city</arg_key><arg_value>"London"</arg_value></tool_call>'
    calls, visible = parse_tool_calls_glm5(text)
    assert visible == "Let me check the weather."

    print("  PASS: GLM-5 parse_tool_calls")


def test_minimax_parse_tool_calls():
    """Test MiniMax tool call parsing: <minimax:tool_call><invoke name="...">...</invoke></minimax:tool_call>"""
    _MM_TOOL_RE = re.compile(r"<minimax:tool_call>(.*?)</minimax:tool_call>", re.DOTALL)
    _MM_INVOKE_RE = re.compile(r'<invoke\s+name="([^"]+)">(.*?)</invoke>', re.DOTALL)
    _MM_PARAM_RE = re.compile(r'<parameter\s+name="([^"]+)">(.*?)</parameter>', re.DOTALL)

    def parse_tool_calls_minimax(text):
        matches = _MM_TOOL_RE.findall(text)
        if not matches:
            return None, text
        tool_calls = []
        for raw in matches:
            for inv in _MM_INVOKE_RE.finditer(raw):
                name = inv.group(1)
                body = inv.group(2)
                arguments = {}
                for pm in _MM_PARAM_RE.finditer(body):
                    key = pm.group(1).strip()
                    val = pm.group(2).strip()
                    try:
                        arguments[key] = json.loads(val)
                    except (ValueError, json.JSONDecodeError):
                        arguments[key] = val
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                })
        visible = _MM_TOOL_RE.sub("", text).strip()
        return tool_calls or None, visible

    # Single tool call
    text = '<minimax:tool_call><invoke name="get_weather"><parameter name="city">London</parameter></invoke></minimax:tool_call>'
    calls, visible = parse_tool_calls_minimax(text)
    assert calls is not None
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "London"}

    # Multiple params
    text = '<minimax:tool_call><invoke name="search"><parameter name="query">hello</parameter><parameter name="limit">10</parameter></invoke></minimax:tool_call>'
    calls, visible = parse_tool_calls_minimax(text)
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["query"] == "hello"
    assert args["limit"] == 10

    # No tool calls
    calls, visible = parse_tool_calls_minimax("plain text")
    assert calls is None

    print("  PASS: MiniMax parse_tool_calls")


def test_io_struct_pydantic():
    """Test that the Pydantic models serialize correctly.

    Requires pydantic — skipped if not installed (run on remote machine).
    """
    try:
        import pydantic  # noqa: F401
    except ImportError:
        print("  SKIP: IO struct Pydantic serialization (pydantic not installed)")
        return

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import types
    if "batchgen" not in sys.modules:
        sys.modules["batchgen"] = types.ModuleType("batchgen")
        sys.modules["batchgen"].__path__ = [
            str(Path(__file__).parent.parent / "batchgen")
        ]
    for pkg in ["batchgen.server"]:
        if pkg not in sys.modules:
            mod = types.ModuleType(pkg)
            mod.__path__ = [str(Path(__file__).parent.parent / pkg.replace(".", "/"))]
            sys.modules[pkg] = mod

    from batchgen.server.io_struct import (
        ChatCompletionChoiceMessage,
        ToolCall,
        ToolCallFunction,
    )

    # With reasoning_content
    msg = ChatCompletionChoiceMessage(
        content="The answer is 42.",
        reasoning_content="Let me think...",
    )
    d = msg.dict()
    assert d["content"] == "The answer is 42."
    assert d["reasoning_content"] == "Let me think..."
    assert d["tool_calls"] is None

    # With tool_calls
    msg = ChatCompletionChoiceMessage(
        content=None,
        tool_calls=[
            ToolCall(
                id="call_abc123",
                function=ToolCallFunction(
                    name="get_weather",
                    arguments='{"city": "London"}',
                ),
            )
        ],
    )
    d = msg.dict()
    assert d["content"] is None
    assert len(d["tool_calls"]) == 1
    assert d["tool_calls"][0]["function"]["name"] == "get_weather"
    assert d["tool_calls"][0]["type"] == "function"

    # JSON round-trip
    j = json.loads(msg.json())
    assert j["tool_calls"][0]["id"] == "call_abc123"

    print("  PASS: IO struct Pydantic serialization")


if __name__ == "__main__":
    print("Running output parsing tests...")
    test_r1_parse_thinking()
    test_deepseek_parse_tool_calls()
    test_harmony_parse_thinking()
    test_gptoss_parse_tool_calls()
    test_combined_thinking_and_tool_calls()
    test_glm5_parse_tool_calls()
    test_minimax_parse_tool_calls()
    test_io_struct_pydantic()
    print("\nAll tests passed!")
