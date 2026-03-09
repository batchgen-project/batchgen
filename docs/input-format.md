# Batch Input JSONL Format

Reference for the JSONL input file format used by BatchGen's `/v1/batches` API. Each line is a JSON object representing one request.

## Request Structure

```json
{
  "custom_id": "req-1",
  "method": "POST",
  "url": "/v1/chat/completions",
  "body": {
    "model": "deepseek-ai/DeepSeek-R1",
    "messages": [{"role": "user", "content": "What is AI?"}],
    "max_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50
  }
}
```

### Top-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `custom_id` | string | Yes | Unique identifier for matching results to requests |
| `method` | string | Yes | HTTP method — must be `"POST"` |
| `url` | string | Yes | Endpoint — `/v1/chat/completions` or `/v1/completions` |
| `body` | object | Yes | Request body (see below) |

### Body Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Model identifier (for OpenAI compatibility — BatchGen uses the server's loaded model) |
| `messages` | array | Yes* | Chat messages with `role` and `content` (for `/v1/chat/completions`) |
| `prompt` | string | Yes* | Text prompt (for `/v1/completions`) |
| `max_completion_tokens` | int | No | Maximum output tokens to generate per request (preferred, OpenAI-compatible) |
| `max_tokens` | int | No | Maximum output tokens to generate per request (legacy alias for `max_completion_tokens`) |
| `temperature` | float | No | Sampling temperature (see [Sampling Parameters](#sampling-parameters)) |
| `top_p` | float | No | Nucleus sampling threshold (see [Sampling Parameters](#sampling-parameters)) |
| `top_k` | int | No | Top-k filtering threshold (see [Sampling Parameters](#sampling-parameters)) |

*One of `messages` or `prompt` is required depending on the endpoint.

---

## Sampling Parameters

Each request in a batch can specify its own sampling parameters. This enables mixed sampling strategies within a single batch — for example, some requests using greedy decoding while others use creative sampling.

### Parameter Reference

| Parameter | Type | Default | Effect |
|-----------|------|---------|--------|
| `temperature` | float | None (greedy) | Controls randomness. 0.0 = deterministic (argmax). Higher values = more random. |
| `top_p` | float | None (disabled) | Nucleus sampling — only sample from tokens whose cumulative probability exceeds this threshold. 1.0 = all tokens eligible. |
| `top_k` | int | None (disabled) | Only sample from the top-k highest-probability tokens. 0 = disabled. 1 = argmax. |

### Default Behavior

When a sampling parameter is not specified in the request body:

| Condition | Behavior |
|-----------|----------|
| `temperature` absent | Greedy decoding (argmax) |
| `top_p` absent | No nucleus filtering (equivalent to top_p=1.0) |
| `top_k` absent | No top-k filtering (equivalent to top_k=0) |

### Output Length Priority

Output length can be set at three levels. Per-request values always take priority:

| Priority | Source | Field |
|----------|--------|-------|
| 1 (highest) | Per-request `max_completion_tokens` in JSONL body | `body.max_completion_tokens` |
| 2 | Per-request `max_tokens` in JSONL body (legacy) | `body.max_tokens` |
| 3 (lowest) | Batch-level `max_decoding_length` (fallback) | `create_batch(max_decoding_length=...)` |

When both `max_completion_tokens` and `max_tokens` are set on the same request, `max_completion_tokens` wins. When neither is set, the batch-level `max_decoding_length` is used as the fallback. **If none of these are set, the batch is rejected with an error.** Each sequence is checked independently — different sequences in the same batch can have different output limits.

### Sampling Override Priority

Sampling parameters can be set at two levels. Per-request values always take priority:

| Priority | Source | Example |
|----------|--------|---------|
| 1 (highest) | Per-request value in JSONL body | `"temperature": 0.7` in the request body |
| 2 | Batch-level default (`create_batch()` / `submit_batch()`) | `temperature=0.5` passed to the client API |
| 3 (lowest) | None | Greedy decoding / filtering disabled |

When batch-level sampling parameters are set, they serve as defaults for requests that omit those fields. A warning is logged:

> "Batch-level sampling params serve as defaults; per-request values in JSONL body take priority."

### Processing Pipeline

```
Per-request body → batch_scheduler extracts params (with batch-level fallback)
  → SequenceEntry stores params (immutable, set once at creation)
  → ActiveBatch builds [B] tensors (cached, rebuilt at page boundaries)
  → sample_tokens() applies vectorized sampling per sequence
```

All sampling operations are vectorized on GPU — no Python loops over sequences. The implementation splits greedy sequences (temperature ≤ 0) from sampling sequences, processes them separately, and merges results.

---

## Examples

### Greedy Decoding (Default)

No sampling parameters specified — all requests use argmax:

```jsonl
{"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "m", "messages": [{"role": "user", "content": "2+2=?"}], "max_completion_tokens": 10}}
```

### Per-Sequence Output Limits

Different output lengths per request — each sequence is checked independently:

```jsonl
{"custom_id": "short", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "m", "messages": [{"role": "user", "content": "What is 2+2?"}], "max_completion_tokens": 10}}
{"custom_id": "long", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "m", "messages": [{"role": "user", "content": "Write a detailed essay"}], "max_completion_tokens": 4096}}
{"custom_id": "legacy", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "m", "messages": [{"role": "user", "content": "Summarize this"}], "max_tokens": 100}}
```

### Mixed Sampling in One Batch

Different sampling strategies per request:

```jsonl
{"custom_id": "greedy", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "m", "messages": [{"role": "user", "content": "What is 2+2?"}], "max_completion_tokens": 10, "temperature": 0.0}}
{"custom_id": "creative", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "m", "messages": [{"role": "user", "content": "Write a poem"}], "max_completion_tokens": 200, "temperature": 1.0, "top_p": 0.9}}
{"custom_id": "focused", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "m", "messages": [{"role": "user", "content": "Summarize this"}], "max_completion_tokens": 100, "temperature": 0.3, "top_k": 10}}
```

### Text Completions

Both `max_completion_tokens` and `max_tokens` are supported for `/v1/completions` requests, with the same priority rules as chat completions.

```jsonl
{"custom_id": "tc-1", "method": "POST", "url": "/v1/completions", "body": {"model": "m", "prompt": "The meaning of life is", "max_completion_tokens": 100, "temperature": 0.7}}
{"custom_id": "tc-2", "method": "POST", "url": "/v1/completions", "body": {"model": "m", "prompt": "Once upon a time", "max_tokens": 200}}
```

---

## See Also

- [Client API Reference](client-api.md) — Python client for submitting batches
- [Output Format](output-format.md) — Result JSONL structure
- [Server Flags](server-flags.md) — Server configuration options
