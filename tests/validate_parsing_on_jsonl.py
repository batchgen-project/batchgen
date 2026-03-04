"""Validate output parsers against real model JSONL files.

Usage:
    python tests/validate_parsing_on_jsonl.py <input.jsonl> \
        --parser {deepseek,gptoss,kimi} \
        [--parse-thinking] [--parse-tool-call] \
        [--show N] [--output parsed.jsonl]

This reads a raw JSONL output file (from a BatchGen run without parsing),
applies the specified parser, and shows the results for manual inspection.

Examples:
    # Inspect thinking extraction on DeepSeek-R1 output
    python tests/validate_parsing_on_jsonl.py output.jsonl \
        --parser deepseek --parse-thinking --show 5

    # Full parsing + write parsed output
    python tests/validate_parsing_on_jsonl.py output.jsonl \
        --parser gptoss --parse-thinking --parse-tool-call \
        --output parsed.jsonl
"""

import argparse
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Optional


# ---- Parsers (standalone, no BatchGen imports needed) ----

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


def parse_thinking_harmony(text: str) -> tuple[Optional[str], str]:
    channels = {}
    for m in _HARMONY_CHANNEL_RE.finditer(text):
        channels[m.group(1).strip()] = m.group(2).strip()
    if not channels:
        return None, text
    return channels.get("analysis"), channels.get("final", text)


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
    return tool_calls, _DS_TOOL_RE.sub("", text).strip()


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
            tool_calls.append({"parse_error": raw_call[:200]})
    return tool_calls or None, visible


def parse_tool_calls_glm5(text: str) -> tuple[Optional[list], str]:
    _GLM5_TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    _GLM5_ARG_RE = re.compile(
        r"<arg_key>(.*?)</arg_key><arg_value>(.*?)</arg_value>", re.DOTALL
    )
    matches = _GLM5_TOOL_RE.findall(text)
    if not matches:
        return None, text
    tool_calls = []
    for raw in matches:
        raw = raw.strip()
        arg_start = raw.find("<arg_key>")
        name = raw[:arg_start].strip() if arg_start != -1 else raw
        arguments = {}
        for am in _GLM5_ARG_RE.finditer(raw):
            key, val = am.group(1).strip(), am.group(2).strip()
            try:
                arguments[key] = json.loads(val)
            except (ValueError, json.JSONDecodeError):
                arguments[key] = val
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        })
    return tool_calls, _GLM5_TOOL_RE.sub("", text).strip()


def parse_tool_calls_minimax(text: str) -> tuple[Optional[list], str]:
    _MM_TOOL_RE = re.compile(r"<minimax:tool_call>(.*?)</minimax:tool_call>", re.DOTALL)
    _MM_INVOKE_RE = re.compile(r'<invoke\s+name="([^"]+)">(.*?)</invoke>', re.DOTALL)
    _MM_PARAM_RE = re.compile(r'<parameter\s+name="([^"]+)">(.*?)</parameter>', re.DOTALL)
    matches = _MM_TOOL_RE.findall(text)
    if not matches:
        return None, text
    tool_calls = []
    for raw in matches:
        for inv in _MM_INVOKE_RE.finditer(raw):
            name = inv.group(1)
            arguments = {}
            for pm in _MM_PARAM_RE.finditer(inv.group(2)):
                key, val = pm.group(1).strip(), pm.group(2).strip()
                try:
                    arguments[key] = json.loads(val)
                except (ValueError, json.JSONDecodeError):
                    arguments[key] = val
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            })
    return tool_calls or None, _MM_TOOL_RE.sub("", text).strip()


PARSERS = {
    "deepseek": {
        "thinking": parse_thinking_r1,
        "tool_calls": parse_tool_calls_deepseek,
    },
    "gptoss": {
        "thinking": parse_thinking_harmony,
        "tool_calls": parse_tool_calls_gptoss,
    },
    "kimi": {
        "thinking": parse_thinking_r1,
        "tool_calls": None,
    },
    "glm5": {
        "thinking": parse_thinking_r1,
        "tool_calls": parse_tool_calls_glm5,
    },
    "minimax": {
        "thinking": parse_thinking_r1,
        "tool_calls": parse_tool_calls_minimax,
    },
}


def extract_content(item: dict) -> Optional[str]:
    """Extract content from a BatchGen JSONL result item."""
    try:
        resp = item.get("response", {})
        body = resp.get("body", {})
        choices = body.get("choices", [])
        if not choices:
            return None
        choice = choices[0]
        # Chat completion
        if "message" in choice:
            return choice["message"].get("content")
        # Text completion
        return choice.get("text")
    except (KeyError, IndexError, TypeError):
        return None


def main():
    ap = argparse.ArgumentParser(description="Validate parsers on real JSONL")
    ap.add_argument("input", type=Path, help="Input JSONL file")
    ap.add_argument(
        "--parser",
        choices=list(PARSERS.keys()),
        required=True,
        help="Which model's parser to use",
    )
    ap.add_argument("--parse-thinking", action="store_true")
    ap.add_argument("--parse-tool-call", action="store_true")
    ap.add_argument(
        "--show",
        type=int,
        default=5,
        help="Number of results to display in detail (default: 5)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write parsed results to this JSONL file",
    )
    args = ap.parse_args()

    if not args.parse_thinking and not args.parse_tool_call:
        print("ERROR: Specify at least one of --parse-thinking or --parse-tool-call")
        sys.exit(1)

    parser_funcs = PARSERS[args.parser]

    # Read input
    items = []
    with args.input.open() as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    print(f"Loaded {len(items)} items from {args.input}")
    print(f"Parser: {args.parser}")
    print(f"Flags: thinking={args.parse_thinking}, tool_call={args.parse_tool_call}")
    print("=" * 80)

    # Stats
    n_thinking_found = 0
    n_tool_calls_found = 0
    n_no_content = 0
    parsed_items = []

    for i, item in enumerate(items):
        content = extract_content(item)
        if content is None:
            n_no_content += 1
            parsed_items.append(item)
            continue

        reasoning_content = None
        tool_calls = None
        visible = content

        if args.parse_thinking and parser_funcs.get("thinking"):
            reasoning_content, visible = parser_funcs["thinking"](visible)
            if reasoning_content is not None:
                n_thinking_found += 1

        if args.parse_tool_call and parser_funcs.get("tool_calls"):
            tool_calls, visible = parser_funcs["tool_calls"](visible)
            if tool_calls is not None:
                n_tool_calls_found += 1

        # Show detailed output for first N items
        if i < args.show:
            custom_id = item.get("custom_id", f"item_{i}")
            print(f"\n--- [{i}] custom_id={custom_id} ---")
            print(f"RAW ({len(content)} chars): {content[:200]}{'...' if len(content) > 200 else ''}")
            if args.parse_thinking:
                if reasoning_content is not None:
                    print(f"REASONING ({len(reasoning_content)} chars): {reasoning_content[:200]}{'...' if len(reasoning_content) > 200 else ''}")
                else:
                    print("REASONING: (none)")
            if args.parse_tool_call:
                if tool_calls is not None:
                    print(f"TOOL_CALLS ({len(tool_calls)}): {json.dumps(tool_calls, indent=2)[:500]}")
                else:
                    print("TOOL_CALLS: (none)")
            print(f"CONTENT ({len(visible)} chars): {visible[:200]}{'...' if len(visible) > 200 else ''}")

        # Build parsed item for output
        if args.output:
            parsed = json.loads(json.dumps(item))  # deep copy
            try:
                msg = parsed["response"]["body"]["choices"][0]["message"]
                msg["content"] = visible
                if reasoning_content is not None:
                    msg["reasoning_content"] = reasoning_content
                if tool_calls is not None:
                    msg["tool_calls"] = tool_calls
                    msg["content"] = None
            except (KeyError, IndexError, TypeError):
                pass
            parsed_items.append(parsed)

    # Summary
    print("\n" + "=" * 80)
    print(f"SUMMARY:")
    print(f"  Total items:        {len(items)}")
    print(f"  No content:         {n_no_content}")
    if args.parse_thinking:
        print(f"  Thinking found:     {n_thinking_found}/{len(items) - n_no_content}")
    if args.parse_tool_call:
        print(f"  Tool calls found:   {n_tool_calls_found}/{len(items) - n_no_content}")

    # Data loss check
    if args.parse_thinking and n_thinking_found == 0:
        print("\n  WARNING: No thinking blocks found — parser may not match model output format!")
    if args.parse_tool_call and n_tool_calls_found == 0:
        print("\n  NOTE: No tool calls found (normal if prompts didn't request tool use)")

    # Write output
    if args.output and parsed_items:
        with args.output.open("w") as f:
            for item in parsed_items:
                f.write(json.dumps(item, default=str))
                f.write("\n")
        print(f"\nParsed output written to {args.output}")


if __name__ == "__main__":
    main()
