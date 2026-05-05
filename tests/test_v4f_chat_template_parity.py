"""Run the BatchGen-adapter chat-template logic against encoding_dsv4 + the 4 ground-truth fixtures.

We replicate the body of DeepSeekV4Tokenizer.apply_chat_template inline (it's small)
so the test doesn't drag in the entire batchgen package + JIT compile of core_engine.
The replica is byte-equal to the adapter as of afd8cda6:
  - normalises tools / response_format into the system message
  - maps enable_thinking → thinking_mode
  - calls encode_messages

Run on Gemini node-1 (no env wrappers needed): python3 /tmp/v4f/v4f_chat_template_parity.py
"""
from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path

ROOT = Path("/home/qspace/BatchGen-deepseek-v4-flash")
ENC = ROOT / "batchgen/models/deepseek/deepseekv4_flash/assets/encoding"
TESTS = ENC / "tests"

sys.path.insert(0, str(ENC))
from encoding_dsv4 import encode_messages  # noqa: E402

OK = "\033[32mOK\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def adapter_apply_chat_template(
    messages,
    tokenize: bool = False,
    add_generation_prompt: bool = True,
    **kwargs,
):
    """Replica of DeepSeekV4Tokenizer.apply_chat_template (worktree afd8cda6)."""
    rendered_messages = [dict(message) for message in messages]
    tools = kwargs.get("tools")
    if tools is not None and not any("tools" in m for m in rendered_messages):
        if rendered_messages and rendered_messages[0].get("role") == "system":
            rendered_messages[0]["tools"] = tools
        else:
            rendered_messages.insert(0, {"role": "system", "content": "", "tools": tools})

    response_format = kwargs.get("response_format")
    if response_format is not None and not any(
        "response_format" in m for m in rendered_messages
    ):
        if rendered_messages and rendered_messages[0].get("role") == "system":
            rendered_messages[0]["response_format"] = response_format
        else:
            rendered_messages.insert(
                0,
                {"role": "system", "content": "", "response_format": response_format},
            )

    enable_thinking = kwargs.get("enable_thinking", kwargs.get("thinking", False))
    thinking_mode = "thinking" if enable_thinking else "chat"
    preserve_thinking = kwargs.get("preserve_thinking", False)
    reasoning_effort = kwargs.get("reasoning_effort")
    return encode_messages(
        rendered_messages,
        thinking_mode=thinking_mode,
        drop_thinking=not preserve_thinking,
        reasoning_effort=reasoning_effort,
    )


def _show_diff(a: str, b: str, label_a: str, label_b: str, max_lines: int = 30) -> None:
    diff = list(
        difflib.unified_diff(
            a.splitlines(keepends=True),
            b.splitlines(keepends=True),
            fromfile=label_a,
            tofile=label_b,
            n=2,
        )
    )
    print("".join(diff[:max_lines]))
    if len(diff) > max_lines:
        print(f"... ({len(diff) - max_lines} more diff lines)")


def _normalise_input(inp):
    """Fixture inputs are either {messages: [...], tools: [...]} or a bare messages list."""
    if isinstance(inp, list):
        return inp, None
    return inp.get("messages", []), inp.get("tools")


def _embed_tools(messages, tools):
    """Mirror adapter logic: tools belong in messages[0]['tools']."""
    if tools is None or any("tools" in m for m in messages):
        return [dict(m) for m in messages]
    out = [dict(m) for m in messages]
    if out and out[0].get("role") == "system":
        out[0]["tools"] = tools
    else:
        out.insert(0, {"role": "system", "content": "", "tools": tools})
    return out


def main() -> int:
    failures = 0
    for n in (1, 2, 3, 4):
        inp = json.loads((TESTS / f"test_input_{n}.json").read_text())
        expected = (TESTS / f"test_output_{n}.txt").read_text()

        messages, tools = _normalise_input(inp)
        msgs_with_tools = _embed_tools(messages, tools)

        # Probe both modes / drop-thinking variants for which one matches the fixture.
        candidates = []
        for mode in ("thinking", "chat"):
            for drop in (True, False):
                candidates.append(
                    (
                        mode,
                        drop,
                        encode_messages(
                            msgs_with_tools, thinking_mode=mode, drop_thinking=drop
                        ),
                    )
                )
        chosen = next(((m, d, r) for (m, d, r) in candidates if r == expected), None)

        print(f"\n=== test {n} (has_tools={tools is not None}) ===")
        if chosen is None:
            print(f"  {FAIL} encode_messages reference does NOT match test_output_{n}.txt"
                  " under any (thinking-mode, drop_thinking) variant.")
            best = candidates[0][2]  # arbitrary candidate
            _show_diff(best, expected,
                       "encode_messages(thinking,drop=True)",
                       f"test_output_{n}.txt")
            failures += 1
            continue
        mode, drop, ref_rendered = chosen
        print(f"  ref matches: thinking_mode={mode!r}, drop_thinking={drop}")

        # Adapter check: BatchGen's apply_chat_template logic with the
        # corresponding kwargs.
        adapter_rendered = adapter_apply_chat_template(
            messages,
            enable_thinking=(mode == "thinking"),
            preserve_thinking=not drop,
            tools=tools,
        )
        if adapter_rendered == ref_rendered:
            print(f"  {OK} adapter byte-equal to encode_messages ({len(adapter_rendered)} chars)")
        else:
            print(f"  {FAIL} adapter diverges from encode_messages")
            print(f"  ref length: {len(ref_rendered)}   adapter length: {len(adapter_rendered)}")
            _show_diff(adapter_rendered, ref_rendered,
                       "BatchGen.apply_chat_template (replica)",
                       "encode_messages")
            failures += 1

    print()
    if failures:
        print(f"FAIL: {failures} divergences")
        return 2
    print("PASS: BatchGen adapter mirrors encoding_dsv4 byte-for-byte against all 4 ground-truth fixtures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
