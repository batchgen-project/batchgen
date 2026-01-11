# Deploy DeepSeek-R1 on 2 H20 Nodes

This guide provides step-by-step instructions for deploying DeepSeek-R1 (or DeepSeek-V3) on 2 NVIDIA H20 GPU nodes (16 GPUs total) using BatchGen.

## Table of Contents

1. [Download HuggingFace Model Checkpoints](#1-download-huggingface-model-checkpoints)
2. [Convert Checkpoints to BatchGen Format](#2-convert-checkpoints-to-batchgen-format)
3. [Build Docker Container or Install BatchGen](#3-build-docker-container-or-install-batchgen)
4. [Prepare Request Files (JSONL)](#4-prepare-request-files-jsonl)
5. [Start BatchGen Server](#5-start-batchgen-server)
6. [Submit Jobs with Python APIs](#6-submit-jobs-with-python-apis)
7. [Download Results](#7-download-results)

---

## 1. Download HuggingFace Model Checkpoints

Download the model weights to a shared storage location accessible by all nodes.

```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Login to HuggingFace (required for gated models)
huggingface-cli login

# Download DeepSeek-R1 to shared storage
huggingface-cli download deepseek-ai/DeepSeek-R1 \
    --local-dir /shared/models/DeepSeek-R1 \
    --local-dir-use-symlinks False
```

### Verify Download

After download, verify the checkpoint files:

```bash
ls -la /shared/models/DeepSeek-R1/
# Should see files like:
# - model-00001-of-00055.safetensors
# - model-00002-of-00055.safetensors
# - ...
# - config.json
# - tokenizer.json
# - tokenizer_config.json
```

---

## 2. Convert Checkpoints to BatchGen Format

BatchGen uses a custom checkpoint format optimized for peak SSD read performance. Converting checkpoints before deployment eliminates cold-start conversion time.

### Why Convert?

- **Performance**: Contiguous tensor storage enables sequential SSD reads
- **Cold-start elimination**: Pre-converted checkpoints load instantly
- **Format**: Each `.safetensors` file becomes a `.bin` (data) + `.json` (metadata) pair

### Using the CLI Tool

```bash
# Convert all checkpoint files in the model directory
python -m batchgen.tools.convert_checkpoint \
    --input-dir /shared/models/DeepSeek-R1

# This creates /shared/models/DeepSeek-R1/converted_ckpt/ with:
# - model-00001-of-00055.bin
# - model-00001-of-00055.json
# - model-00002-of-00055.bin
# - model-00002-of-00055.json
# - ...
```

### CLI Options

```bash
python -m batchgen.tools.convert_checkpoint --help

Options:
  --input-dir, -i   Directory containing .safetensors/.pt files (required)
  --output-dir, -o  Output directory (default: <input-dir>/converted_ckpt)
  --force, -f       Force reconversion even if output exists
  --validate-only   Only validate existing converted files
  --verbose, -v     Enable debug logging
```

### Using Python API

```python
from batchgen import ckpt_converter

converter = ckpt_converter()

# Convert entire model directory
converted_path = converter.convert_model_directory(
    input_dir="/shared/models/DeepSeek-R1",
    output_dir=None,  # Default: <input_dir>/converted_ckpt
    force=False       # Skip if already converted
)

print(f"Converted checkpoints at: {converted_path}")
```

### Validate Conversion

```bash
# Validate without reconverting
python -m batchgen.tools.convert_checkpoint \
    --input-dir /shared/models/DeepSeek-R1 \
    --validate-only

# Expected output:
# Validation successful! All converted files are consistent with source checkpoints.
```

---

## 3. Build Docker Container

Build the BatchGen Docker image using the provided Dockerfile in the `docker/` directory.

```bash
# From the BatchGen project root
docker buildx build --progress=plain -f docker/Dockerfile -t batchgen:latest .
```

For more details, see [`docker/README.md`](../docker/README.md).

### Run Container

Run on each node with the appropriate `--node-rank`. The container provides an interactive shell where you can start the server.

```bash
# Node 0 (Master)
docker run -it \
    --cap-add=SYS_NICE \
    --cap-add=SYS_ADMIN \
    --runtime=nvidia \
    --gpus all \
    --network=host \
    -v /shared/models:/models:ro \
    -v /shared/storage:/storage \
    batchgen:latest

# Node 1
docker run -it \
    --cap-add=SYS_NICE \
    --cap-add=SYS_ADMIN \
    --runtime=nvidia \
    --gpus all \
    --network=host \
    -v /shared/models:/models:ro \
    -v /shared/storage:/storage \
    batchgen:latest
```

**Optional flags** (add if needed):
- `--privileged`: Full host access (use with caution)
- `--ipc=host`: Share host IPC namespace

Once inside the container, start the server using the commands in [Section 5](#5-start-batchgen-server).

---

## 4. Prepare Request Files (JSONL)

BatchGen supports OpenAI-compatible batch request format. Each line in the JSONL file is a separate request.

### Chat Completions Format (Recommended)

```jsonl
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "deepseek-r1", "messages": [{"role": "user", "content": "What is the capital of France?"}], "max_tokens": 100}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "deepseek-r1", "messages": [{"role": "user", "content": "Explain quantum computing in simple terms."}], "max_tokens": 500}}
{"custom_id": "request-3", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "deepseek-r1", "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Write a Python function to calculate fibonacci numbers."}], "max_tokens": 1000}}
```

### Text Completions Format

```jsonl
{"custom_id": "request-1", "method": "POST", "url": "/v1/completions", "body": {"model": "deepseek-r1", "prompt": "The meaning of life is", "max_tokens": 100}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/completions", "body": {"model": "deepseek-r1", "prompt": "def fibonacci(n):\n    ", "max_tokens": 200}}
```

### Request Body Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Model identifier (for compatibility, not used by BatchGen) |
| `messages` | array | Chat messages with `role` and `content` (chat completions) |
| `prompt` | string | Text prompt (text completions) |
| `max_tokens` | int | Maximum tokens to generate |
| `temperature` | float | Sampling temperature (0.0 = greedy, default) |
| `top_p` | float | Nucleus sampling threshold |

### Create Request File with Python

```python
import json

requests = [
    {
        "custom_id": f"req-{i}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "deepseek-r1",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512
        }
    }
    for i, prompt in enumerate([
        "What is machine learning?",
        "Explain neural networks.",
        "What is deep learning?",
        # ... more prompts
    ])
]

# Write JSONL file
with open("requests.jsonl", "w") as f:
    for req in requests:
        f.write(json.dumps(req) + "\n")
```

---

## 5. Start BatchGen Server

### Prerequisites: Mount Shared Memory

BatchGen uses `/dev/shm` for host KV cache. Before starting the server, ensure `/dev/shm` is mounted with sufficient size (should match your host memory):

```bash
# Check current size
df -h /dev/shm

# Mount with full host memory size (replace 1500G with your host memory)
sudo mount -o remount,size=1500G /dev/shm

# To make permanent, add to /etc/fstab:
# tmpfs /dev/shm tmpfs defaults,size=1500G 0 0
```

### 2 Nodes × 8 GPUs = 16 GPUs

#### Node 0 (Master)

```bash
python -m batchgen.launch_http_server \
    --model deepseek-ai/DeepSeek-R1 \
    --cache-dir /shared/models/DeepSeek-R1 \
    --listen-port 10900 \
    --world-size 16 \
    --nnodes 2 \
    --node-rank 0 \
    --dist-init-addr node0-ip:12355 \
    --storage-path /shared/storage \
    --host-kv-cache-size 256
```

#### Node 1

```bash
python -m batchgen.launch_http_server \
    --model deepseek-ai/DeepSeek-R1 \
    --cache-dir /shared/models/DeepSeek-R1 \
    --listen-port 10900 \
    --world-size 16 \
    --nnodes 2 \
    --node-rank 1 \
    --dist-init-addr node0-ip:12355 \
    --storage-path /shared/storage \
    --host-kv-cache-size 256
```

### Server Arguments Reference

Key arguments for multi-node deployment:

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | required | HuggingFace model name |
| `--cache-dir` | None | Path to pre-downloaded model files |
| `--world-size` | 1 | Total number of GPUs across all nodes |
| `--nnodes` | 1 | Number of nodes in the cluster |
| `--node-rank` | 0 | Rank of this node (0 = master) |
| `--dist-init-addr` | localhost:12355 | Address for distributed init (`master-ip:port`) |
| `--host-kv-cache-size` | None | Host KV cache size in GB (critical for throughput) |
| `--storage-path` | batchgen/storage/ | Directory for files and batches |
| `--save-result` | false | Save inference results to `{storage_path}/outputs/` |

For the complete list of all server flags, see **[Server Flags Reference](server-flags.md)**.

---

## 6. Submit Jobs with Python APIs

### Method A: Batch API (Recommended for Large Jobs)

The Batch API is best for processing large numbers of requests asynchronously.

```python
from batchgen.batchgen_client import BatchGenHttpClient

# Connect to server
client = BatchGenHttpClient(base_url="http://node0-ip:10900")

# Submit batch job (upload, create, wait, download in one call)
result = client.submit_batch(
    input_file_path="requests.jsonl",
    output_file_path="results.jsonl",
    endpoint="/v1/chat/completions",
    max_decoding_length=1024,  # Override per-request max_tokens
    temperature=0.7,           # Sampling temperature
)

print(f"Batch completed: {result['id']}")
print(f"Completed requests: {result['request_counts']['completed']}")
```

### Method B: Step-by-Step Batch API

For more control over the batch lifecycle:

```python
from batchgen.batchgen_client import BatchGenHttpClient
import time

client = BatchGenHttpClient(base_url="http://node0-ip:10900")

# Step 1: Upload input file
file_obj = client.upload_file("requests.jsonl", purpose="batch")
print(f"Uploaded file: {file_obj['id']}")

# Step 2: Create batch job
batch = client.create_batch(
    input_file_id=file_obj["id"],
    endpoint="/v1/chat/completions",
    max_decoding_length=512,
)
print(f"Created batch: {batch['id']}")

# Step 3: Wait for completion (no timeout by default)
batch = client.wait_for_batch(batch_id=batch["id"])
print(f"Batch status: {batch['status']}")

# Step 4: Download results
if batch["output_file_id"]:
    content = client.download_file_content(batch["output_file_id"])
    with open("results.jsonl", "wb") as f:
        f.write(content)
    print("Results saved to results.jsonl")
```

---

## 7. Download Results

### Batch Results Format

Batch results are returned in JSONL format, one response per line:

```jsonl
{"id": "batch_req_abc123", "custom_id": "request-1", "response": {"status_code": 200, "body": {"id": "chatcmpl-xxx", "choices": [{"message": {"role": "assistant", "content": "The capital of France is Paris."}}]}}}
{"id": "batch_req_def456", "custom_id": "request-2", "response": {"status_code": 200, "body": {"id": "chatcmpl-yyy", "choices": [{"message": {"role": "assistant", "content": "Quantum computing uses quantum mechanics..."}}]}}}
```

### Parse Results with Python

```python
import json

results = []
with open("results.jsonl", "r") as f:
    for line in f:
        result = json.loads(line)
        custom_id = result["custom_id"]
        response = result["response"]["body"]
        content = response["choices"][0]["message"]["content"]
        results.append({
            "id": custom_id,
            "response": content
        })

# Process results
for r in results:
    print(f"[{r['id']}] {r['response'][:100]}...")
```

### Download via API

```python
from batchgen.batchgen_client import BatchGenHttpClient

client = BatchGenHttpClient(base_url="http://node0-ip:10900")

# Get batch info
batch = client.get_batch("batch_abc123")

# Download output file
if batch["output_file_id"]:
    content = client.download_file_content(batch["output_file_id"])

    # Save to file
    with open("results.jsonl", "wb") as f:
        f.write(content)

    # Or parse directly
    import json
    for line in content.decode("utf-8").strip().split("\n"):
        result = json.loads(line)
        print(result)
```

### List All Batches

```python
from batchgen.batchgen_client import BatchGenHttpClient
import requests

client = BatchGenHttpClient(base_url="http://node0-ip:10900")

# List batches via HTTP
response = requests.get(f"http://node0-ip:10900/v1/batches")
batches = response.json()["data"]

for batch in batches:
    print(f"Batch: {batch['id']}, Status: {batch['status']}")
```

---

## Performance Tips

1. **Pre-convert checkpoints**: Always convert checkpoints before deployment to eliminate cold-start time.

2. **Host KV cache size**: The `--host-kv-cache-size` parameter is essential for performance. Set it to the maximum available:
   ```
   host_kv_cache_size = 0.9 × host_memory - model_size
   ```
   For example, with 1.5TB host memory and DeepSeek-R1 (~700GB model), use `--host-kv-cache-size 650`.

3. **Use hugeTLBFS**: Enable `--enable-hugetlbfs` for better memory performance (requires system configuration).

4. **Batch requests**: Group requests into larger batches for better throughput.

---

## See Also

- [Server Flags Reference](server-flags.md) - Complete list of all server flags
- [Client API Reference](client-api.md) - Python client usage and parameters
- [README](../README.md) - Installation and quick start

## Support

For issues and questions:
- GitHub Issues: https://github.com/EfficientMoE/BatchGen/issues
