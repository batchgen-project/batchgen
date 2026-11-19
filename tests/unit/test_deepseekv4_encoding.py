"""
Test suite for DeepSeek-V4 Encoding.

Upstream ships this suite as ``assets/encoding/test_encoding_dsv4.py`` next to
``encoding_dsv4.py``; it lives here so the runtime package carries no test
modules. Both vendored copies (Flash, whose module the DeepSeek-V4 tokenizer
imports, and Pro) are checked against the upstream golden files that stay in
each ``assets/encoding/tests/`` directory.

Run: pytest tests/unit/test_deepseekv4_encoding.py
"""

import importlib
import json
import os

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_VARIANTS = ("deepseekv4_flash", "deepseekv4_pro")


@pytest.fixture(params=_VARIANTS)
def encoding(request):
    """(encode_messages, parse_message_from_completion_text, TESTS_DIR) per vendored copy."""
    mod = importlib.import_module(
        f"batchgen.models.deepseek.{request.param}.assets.encoding.encoding_dsv4"
    )
    tests_dir = os.path.join(
        _REPO_ROOT, "batchgen", "models", "deepseek", request.param,
        "assets", "encoding", "tests",
    )
    return mod.encode_messages, mod.parse_message_from_completion_text, tests_dir


def test_case_1(encoding):
    """Thinking mode with tool calls (multi-turn, tool results merged into user)."""
    encode_messages, parse_message_from_completion_text, TESTS_DIR = encoding
    with open(os.path.join(TESTS_DIR, "test_input_1.json")) as f:
        td = json.load(f)
        messages = td["messages"]
        messages[0]["tools"] = td["tools"]
    gold = open(os.path.join(TESTS_DIR, "test_output_1.txt")).read()
    prompt = encode_messages(messages, thinking_mode="thinking")
    assert prompt == gold

    # Parse: assistant turn with tool call
    marker = "<｜Assistant｜><think>"
    first_start = prompt.find(marker) + len(marker)
    first_end = prompt.find("<｜User｜>", first_start)
    parsed_tc = parse_message_from_completion_text(prompt[first_start:first_end], thinking_mode="thinking")
    assert parsed_tc["reasoning_content"] == "The user wants to know the weather in Beijing. I should use the get_weather tool."
    assert parsed_tc["content"] == ""
    assert len(parsed_tc["tool_calls"]) == 1
    assert parsed_tc["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(parsed_tc["tool_calls"][0]["function"]["arguments"]) == {"location": "Beijing", "unit": "celsius"}

    # Parse: final assistant turn with content
    last_start = prompt.rfind(marker) + len(marker)
    parsed_final = parse_message_from_completion_text(prompt[last_start:], thinking_mode="thinking")
    assert parsed_final["reasoning_content"] == "Got the weather data. Let me format a nice response."
    assert "22°C" in parsed_final["content"]
    assert parsed_final["tool_calls"] == []

    print("  [PASS] case 1: thinking with tools (encode + parse)")


def test_case_2(encoding):
    """Thinking mode without tools (drop_thinking removes earlier reasoning)."""
    encode_messages, parse_message_from_completion_text, TESTS_DIR = encoding
    messages = json.load(open(os.path.join(TESTS_DIR, "test_input_2.json")))
    gold = open(os.path.join(TESTS_DIR, "test_output_2.txt")).read()
    prompt = encode_messages(messages, thinking_mode="thinking")
    assert prompt == gold

    # Parse: last assistant turn
    marker = "<｜Assistant｜><think>"
    last_start = prompt.rfind(marker) + len(marker)
    parsed = parse_message_from_completion_text(prompt[last_start:], thinking_mode="thinking")
    assert parsed["reasoning_content"] == "The user asks about the capital of France. It is Paris."
    assert parsed["content"] == "The capital of France is Paris."
    assert parsed["tool_calls"] == []

    # Verify drop_thinking: first assistant's reasoning should be absent
    assert "The user said hello" not in prompt

    print("  [PASS] case 2: thinking without tools (encode + parse)")


def test_case_3(encoding):
    """Interleaved thinking + search (developer with tools, latest_reminder)."""
    encode_messages, _, TESTS_DIR = encoding
    messages = json.load(open(os.path.join(TESTS_DIR, "test_input_3.json")))
    gold = open(os.path.join(TESTS_DIR, "test_output_3.txt")).read()
    assert encode_messages(messages, thinking_mode="thinking") == gold
    print("  [PASS] case 3: interleaved thinking + search")


def test_case_4(encoding):
    """Quick instruction task with latest_reminder (chat mode, action task)."""
    encode_messages, _, TESTS_DIR = encoding
    messages = json.load(open(os.path.join(TESTS_DIR, "test_input_4.json")))
    gold = open(os.path.join(TESTS_DIR, "test_output_4.txt")).read()
    assert encode_messages(messages, thinking_mode="chat") == gold
    print("  [PASS] case 4: quick instruction task")

