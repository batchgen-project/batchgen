# BatchGen Client API Reference

Python client for interacting with the BatchGen server.

## BatchGenHttpClient

The recommended client for interacting with BatchGen's OpenAI-compatible HTTP API.

### Initialization

```python
from batchgen.batchgen_client import BatchGenHttpClient

client = BatchGenHttpClient(
    base_url="http://localhost:10900",
    timeout_s=None  # None = wait forever
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | string | required | Server URL (e.g., `http://localhost:10900`) |
| `timeout_s` | float | None | Request timeout in seconds (None = unlimited) |

---

## Batch API (Recommended)

The Batch API is designed for processing large numbers of requests asynchronously.

### submit_batch()

Submit a batch job and wait for completion. This is a convenience method that uploads the file, creates the batch, waits for completion, and downloads results.

```python
result = client.submit_batch(
    input_file_path="requests.jsonl",
    output_file_path="results.jsonl",
    endpoint="/v1/chat/completions",
    poll_interval=5.0,
    timeout=None,
    max_decoding_length=1024,
    temperature=0.7,
    top_p=0.9,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_file_path` | string | required | Path to input JSONL file |
| `output_file_path` | string | None | Path to save output JSONL (optional) |
| `endpoint` | string | `/v1/chat/completions` | Target endpoint (`/v1/chat/completions` or `/v1/completions`) |
| `poll_interval` | float | 5.0 | Seconds between status checks |
| `timeout` | float | None | Maximum seconds to wait (None = unlimited) |
| `max_decoding_length` | int | None | Batch-level fallback for requests without explicit `max_completion_tokens` or `max_tokens` (None = required per-request) |
| `temperature` | float | None | Sampling temperature (None = greedy decoding) |
| `top_p` | float | None | Nucleus sampling threshold (None = disabled) |
| `top_k` | int | None | Top-k filtering threshold (None = disabled) |

### Step-by-Step Batch API

For more control over the batch lifecycle:

```python
# Step 1: Upload input file
file_obj = client.upload_file("requests.jsonl", purpose="batch")

# Step 2: Create batch job
batch = client.create_batch(
    input_file_id=file_obj["id"],
    endpoint="/v1/chat/completions",
    max_decoding_length=512,
)

# Step 3: Wait for completion
batch = client.wait_for_batch(batch["id"])

# Step 4: Download results
if batch["output_file_id"]:
    content = client.download_file_content(batch["output_file_id"])
    with open("results.jsonl", "wb") as f:
        f.write(content)
```

### upload_file()

Upload a file to the server.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | string | required | Path to the file to upload |
| `purpose` | string | `batch` | File purpose (`batch` for input files) |

### create_batch()

Create a batch job.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_file_id` | string | required | ID of the uploaded input file |
| `endpoint` | string | `/v1/chat/completions` | Target endpoint |
| `completion_window` | string | `24h` | Time window for completion |
| `metadata` | dict | None | Optional metadata dictionary |
| `max_decoding_length` | int | None | Batch-level fallback for requests without explicit `max_completion_tokens` or `max_tokens` |
| `temperature` | float | None | Sampling temperature |
| `top_p` | float | None | Nucleus sampling threshold |
| `top_k` | int | None | Top-k filtering threshold |

### wait_for_batch()

Wait for a batch to complete.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_id` | string | required | ID of the batch |
| `poll_interval` | float | 5.0 | Seconds between status checks |
| `timeout` | float | None | Maximum seconds to wait (None = unlimited) |

### get_batch()

Get batch status.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_id` | string | required | ID of the batch |

### download_file_content()

Download file content.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_id` | string | required | ID of the file to download |

---

## Direct Inference API

For synchronous inference without the batch workflow.

### submit_inference()

Submit an inference request and get decoded string results.

```python
results = client.submit_inference(
    prompts=["What is AI?", "Explain machine learning."],
    max_input_len=None,
    max_output_len=128,
    ignore_eos=False,
    temperature=None,
    top_p=None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompts` | list[str] | required | List of prompt strings |
| `max_input_len` | int | None | Maximum input sequence length (None = dynamic from longest prompt) |
| `max_output_len` | int | 128 | Maximum output/decoding length |
| `ignore_eos` | bool | False | If True, ignore EOS tokens and decode to max_output_len (for benchmarking) |
| `temperature` | float | None | Sampling temperature (None = greedy decoding) |
| `top_p` | float | None | Nucleus sampling threshold (None = disabled) |

---

## Utility Methods

### health_check()

Check if the server is healthy.

```python
is_healthy = client.health_check()  # Returns True/False
```

### close()

Close the HTTP session.

```python
client.close()
```

---

## Input File Format

Batch input files use OpenAI-compatible JSONL format. Each line is a separate request.

### Chat Completions

```jsonl
{"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "deepseek-r1", "messages": [{"role": "user", "content": "What is AI?"}], "max_completion_tokens": 100}}
{"custom_id": "req-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "deepseek-r1", "messages": [{"role": "user", "content": "Explain ML."}], "max_completion_tokens": 200}}
```

### Text Completions

```jsonl
{"custom_id": "req-1", "method": "POST", "url": "/v1/completions", "body": {"model": "deepseek-r1", "prompt": "The meaning of life is", "max_tokens": 100}}
```

### Per-Request Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `custom_id` | string | Unique identifier for matching results |
| `model` | string | Model identifier (for compatibility, not used by BatchGen) |
| `messages` | array | Chat messages with `role` and `content` |
| `prompt` | string | Text prompt (for text completions) |
| `max_completion_tokens` | int | Maximum output tokens to generate (preferred, OpenAI-compatible) |
| `max_tokens` | int | Maximum output tokens to generate (legacy alias for `max_completion_tokens`) |
| `temperature` | float | Sampling temperature (0.0 = greedy) |
| `top_p` | float | Nucleus sampling threshold |
| `top_k` | int | Top-k filtering threshold |

**Note:** Per-request sampling parameters override batch-level defaults. See [Input Format](input-format.md) for the full override logic.

---

## Output File Format

Batch results are returned in JSONL format:

```jsonl
{"id": "batch_req_abc", "custom_id": "req-1", "response": {"status_code": 200, "body": {"id": "chatcmpl-xxx", "choices": [{"message": {"role": "assistant", "content": "AI is..."}}]}}}
```

### Parsing Results

```python
import json

with open("results.jsonl", "r") as f:
    for line in f:
        result = json.loads(line)
        custom_id = result["custom_id"]
        content = result["response"]["body"]["choices"][0]["message"]["content"]
        print(f"[{custom_id}] {content}")
```

See [Output Format](output-format.md) for the full response schema, including structured fields for thinking and tool calls.

---

## Test Script CLI Arguments

BatchGen provides test scripts that demonstrate how to use the client API. These scripts accept command-line arguments that map to client API parameters.

### Example: MMLU Pro Batch Test

```bash
python tests/e2e/r1_mmlu_pro_test/r1_mmlu_pro_batch_test.py \
    --hugging_face_checkpoint /path/to/DeepSeek-R1 \
    --max_decoding_length 10240 \
    --server_host localhost \
    --server_port 10900 \
    --temperature 0.7 \
    --top_p 0.9 \
    --top_k 50 \
    --max_prompts 100
```

### CLI Arguments Reference

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--hugging_face_checkpoint` | string | Yes | Path to model checkpoint (used for tokenizer) |
| `--max_decoding_length` | int | Conditional | Batch-level fallback max output tokens (required unless `--random_max_completion_tokens` is set) |
| `--random_max_completion_tokens` | flag | No | Generate random per-request `max_completion_tokens` for each request |
| `--min_completion_tokens` | int | No | Lower bound for random `max_completion_tokens` (default: 16) |
| `--max_completion_tokens` | int | No | Upper bound for random `max_completion_tokens` (default: `--max_decoding_length`) |
| `--server_host` | string | No | Server hostname (default: `localhost`) |
| `--server_port` | int | No | Server port (default: `10900`) |
| `--base_url` | string | No | Full server URL (alternative to host/port, e.g., `http://localhost:10900`) |
| `--temperature` | float | No | Sampling temperature (default: None = greedy decoding) |
| `--top_p` | float | No | Nucleus sampling threshold (default: None = disabled) |
| `--top_k` | int | No | Top-k filtering threshold (default: None = disabled) |
| `--random_sampling_params` | flag | No | Generate random per-request sampling params for each request |
| `--poll_interval` | float | No | Seconds between batch status checks (default: `5.0`) |
| `--timeout` | float | No | Maximum seconds to wait for batch completion (default: None = unlimited) |
| `--max_prompts` | int | No | Limit number of prompts to process (default: None = all) |

### Argument Mapping

The CLI arguments map to client API parameters as follows:

| CLI Argument | Client API Parameter | Request Body Field |
|--------------|---------------------|-------------------|
| `--max_decoding_length` | `max_decoding_length` | Batch-level fallback (not written to body) |
| `--random_max_completion_tokens` | - | `body.max_completion_tokens` (random per-request) |
| `--server_host` + `--server_port` | `base_url` | - |
| `--temperature` | `temperature` | `body.temperature` |
| `--top_p` | `top_p` | `body.top_p` |
| `--top_k` | `top_k` | `body.top_k` |

### Custom Test Scripts

When writing your own test scripts, use the `BatchGenHttpClient`:

```python
from batchgen.batchgen_client import BatchGenHttpClient

# Option 1: Use base_url directly
client = BatchGenHttpClient(base_url="http://localhost:10900")

# Option 2: Construct from host and port
host = "localhost"
port = 10900
client = BatchGenHttpClient(base_url=f"http://{host}:{port}")

# Submit batch with generation parameters
batch = client.submit_batch(
    input_file_path="requests.jsonl",
    output_file_path="results.jsonl",
    endpoint="/v1/chat/completions",
    max_decoding_length=10240,  # From --max_decoding_length
    temperature=0.7,            # From --temperature
    top_p=0.9,                  # From --top_p
)
```

---

## See Also

- [Deployment Guide](deploy-deepseek-r1-h20.md) - Server setup and job submission
- [Server Flags Reference](server-flags.md) - Server configuration options
