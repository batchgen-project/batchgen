# Output Format Reference

BatchGen output follows the OpenAI Batch API format. Each line in the output JSONL file is a `BatchResultItem`.

## Base Structure

```json
{
  "id": "batch_req_<hex>",
  "custom_id": "<from_input_file>",
  "response": {
    "status_code": 200,
    "request_id": "req_<hex>",
    "body": { ... }
  },
  "error": null
}
```

## Chat Completion Response (`/v1/chat/completions`)

```json
{
  "id": "chatcmpl-<hex>",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "deepseek-ai/DeepSeek-R1",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The answer is 42.",
        "reasoning_content": null,
        "tool_calls": null
      },
      "logprobs": null,
      "finish_reason": null
    }
  ],
  "usage": {
    "prompt_tokens": 128,
    "completion_tokens": 64,
    "total_tokens": 192
  },
  "system_fingerprint": null
}
```

## Text Completion Response (`/v1/completions`)

```json
{
  "id": "cmpl-<hex>",
  "object": "text_completion",
  "created": 1234567890,
  "model": "deepseek-ai/DeepSeek-R1",
  "choices": [
    {
      "index": 0,
      "text": "The decoded output text...",
      "logprobs": null,
      "finish_reason": null
    }
  ],
  "usage": { ... }
}
```

Note: `--parse-thinking` and `--parse-tool-call` only apply to chat completion responses. Text completion responses always return raw decoded text.

---

## Structured Output Fields

By default, the raw decoded text (including any special tokens) is placed in `content`. The following server flags enable server-side parsing into structured fields.

### `--parse-thinking`

Extracts model reasoning/thinking into a separate `reasoning_content` field. The `content` field contains only the visible answer.

**Without `--parse-thinking`:**
```json
{
  "content": "<think>Let me reason step by step...</think>The answer is 42.",
  "reasoning_content": null
}
```

**With `--parse-thinking`:**
```json
{
  "content": "The answer is 42.",
  "reasoning_content": "Let me reason step by step..."
}
```

### `--parse-tool-call`

Extracts tool/function calls into an OpenAI-compatible `tool_calls` array.

**Without `--parse-tool-call`:**
```json
{
  "content": "Let me check.<｜tool▁call▁begin｜>get_weather\n{\"city\": \"London\"}<｜tool▁call▁end｜>",
  "tool_calls": null
}
```

**With `--parse-tool-call`:**
```json
{
  "content": "Let me check.",
  "tool_calls": [
    {
      "id": "call_abc123def456",
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": "{\"city\": \"London\"}"
      }
    }
  ]
}
```

---

## Supported Model Formats

Each model uses different special tokens for thinking and tool calls. BatchGen's tokenizer classes handle the model-specific parsing automatically.

| Model | Thinking Format | Tool Call Format |
|-------|----------------|-----------------|
| DeepSeek-R1 | `<think>...</think>` | `<｜tool▁call▁begin｜>name\n{args}<｜tool▁call▁end｜>` |
| GPT-OSS-120B | `<\|channel\|>analysis<\|message\|>...<\|end\|>` | `<\|channel\|>commentary to=functions.NAME <\|constrain\|>json<\|message\|>{args}<\|call\|>` |
| Kimi-K2.5 | `<think>...</think>` | Not yet supported |
| GLM-5 | `<think>...</think>` | `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>` |
| MiniMax-M2.5 | `<think>...</think>` | `<minimax:tool_call><invoke name="...">...</invoke></minimax:tool_call>` |

GPT-OSS-120B writes its output in the harmony format, whose message markers are special tokens. Run the server with `--detokenization-include-special-tokens` as well, so the decoded text keeps them; without it the markers are stripped (`analysis...assistantfinal...`) and the output is returned raw in `content`. Decode stops at `<|call|>`, so a tool-calling turn ends with exactly one call; `content` is empty unless the model wrote a `final` or `commentary` preamble message first. A malformed or truncated message (for example a tool call cut off before `<|call|>` by `max_tokens`) is returned raw in `content` and logged as a warning.

### GPT-OSS-120B tool conversations

Pass tools in the request's `tools` field (OpenAI function-calling format); do not write them into the system message. GPT-OSS renders the conversation in the harmony format:

- `tools` become a developer message declaring a `functions` namespace, and the system message names the `commentary` channel for tool calls.
- An assistant message with `tool_calls` renders as one harmony call per entry (`<|start|>assistant to=functions.NAME<|channel|>commentary <|constrain|>json<|message|>ARGS<|call|>`), after its `content` (as a `commentary` preamble) when that is non-empty. Its `reasoning_content` is kept while the tool turn is still open (no final assistant answer after it) and dropped once a final answer follows, as harmony prescribes.
- A `tool` message renders as the result of the function its `tool_call_id` refers to (`<|start|>functions.NAME to=assistant<|channel|>commentary<|message|>RESULT<|end|>`); without `tool_call_id` its `name` gives the function.

To continue after a tool call, send the assistant message exactly as returned (`content`, `reasoning_content`, `tool_calls`) followed by one `tool` message per call, and keep `tools` in the request.

Each tool's JSON-schema `parameters` is converted to a TypeScript-like declaration. Converted: `type` `string`, `number`, `integer`, `boolean`, `array` (with or without `items`), `object` (`properties`, `required` for non-optional fields, nested objects); a list of scalar types with `null` (`["string", "null"]`); string `enum`; `description`, `title` and `examples` as comments; `default` as a comment. Validation keywords that do not change the value type (`format`, `minimum`, `maxLength`, `pattern`, ...) and `additionalProperties: true|false` are ignored. Any other construct (`anyOf`/`oneOf`/`allOf`, `$ref`/`$defs`, `const`, non-string `enum`, tuple `items`, map-typed `additionalProperties`, unknown keywords) is not approximated: that property is declared `any` with its raw JSON schema in a comment, so the model still sees the whole contract.

A request the format cannot carry is rejected with a message naming the problem: a tool that is not `{"type": "function", ...}`, a missing, duplicate or whitespace-containing function name, `parameters` that is not an object, `function.arguments` that is not a string, or a `tool` message whose `tool_call_id` matches no earlier call.

---

## Parsing Results in Python

```python
import json

with open("results.jsonl", "r") as f:
    for line in f:
        result = json.loads(line)
        custom_id = result["custom_id"]
        msg = result["response"]["body"]["choices"][0]["message"]

        # Visible content
        print(f"[{custom_id}] {msg['content']}")

        # Reasoning (when --parse-thinking is enabled)
        if msg.get("reasoning_content"):
            print(f"  Reasoning: {msg['reasoning_content'][:100]}...")

        # Tool calls (when --parse-tool-call is enabled)
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc["function"]
                print(f"  Tool call: {fn['name']}({fn['arguments']})")
```

---

## See Also

- [Server Flags](server-flags.md#output-parsing) - Enable `--parse-thinking` and `--parse-tool-call`
- [Client API](client-api.md) - Submitting batches and downloading results
