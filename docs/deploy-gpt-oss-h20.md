# Deploy GPT-OSS-120B on H20 Node

This guide provides step-by-step instructions for deploying OpenAI's GPT-OSS-120B model on NVIDIA H20 GPUs using BatchGen.

## Table of Contents

1. [Download Model Checkpoints](#1-download-model-checkpoints)
2. [Convert Checkpoints to BatchGen Format](#2-convert-checkpoints-to-batchgen-format)
3. [Build Docker Container](#3-build-docker-container)
4. [Start BatchGen Server](#4-start-batchgen-server)
5. [Submit Jobs with Python APIs](#5-submit-jobs-with-python-apis)

---

## 1. Download Model Checkpoints

Download GPT-OSS-120B from HuggingFace. Only the `original/` directory is needed.

```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Login to HuggingFace (required for gated models)
huggingface-cli login

# Download only the original directory
huggingface-cli download openai/gpt-oss-120b \
    --include "original/*" \
    --local-dir /path/to/models/gpt-oss-120b \
    --local-dir-use-symlinks False
```

---

## 2. Convert Checkpoints to BatchGen Format

BatchGen uses a custom checkpoint format optimized for peak SSD read performance.

```bash
# Convert all checkpoint files
python -m batchgen.tools.convert_checkpoint \
    --input-dir /path/to/models/gpt-oss-120b/original

# This creates /path/to/models/gpt-oss-120b/original/converted_ckpt/
```

### Validate Conversion

```bash
python -m batchgen.tools.convert_checkpoint \
    --input-dir /path/to/models/gpt-oss-120b/original \
    --validate-only
```

---

## 3. Build Docker Container

```bash
# From BatchGen project root
docker buildx build --progress=plain -f docker/Dockerfile -t batchgen:latest .
```

### Run Container

```bash
docker run -it \
    --cap-add=SYS_ADMIN \
    --runtime=nvidia \
    --gpus all \
    --network=host \
    -v /path/to/models:/models:ro \
    -v /path/to/storage:/storage \
    batchgen:latest
```

Inside the container, remount `/dev/shm` with all available host memory:

```bash
# Recommended: mount all available host memory
mount -o remount,size=2048G /dev/shm
```

---

## 4. Start BatchGen Server

### 8-GPU Deployment (Recommended)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m batchgen.launch_http_server \
    --model openai/gpt-oss-120b \
    --cache-dir /models/gpt-oss-120b/original \
    --dist-init-addr "localhost:33001" \
    --nnodes 1 \
    --node-rank 0 \
    --world-size 8 \
    --kv-dtype "bf16" \
    --host-kv-cache-size 1600 \
    --enable-hugetlbfs \
    --gpu-memory-frac 0.96 \
    --storage-path /storage \
    --save-result
```

For other GPU configurations, adjust `CUDA_VISIBLE_DEVICES` and `--world-size`:
- 4 GPUs: `CUDA_VISIBLE_DEVICES=0,1,2,3` with `--world-size 4`
- 2 GPUs: `CUDA_VISIBLE_DEVICES=0,1` with `--world-size 2`
- 1 GPU: `CUDA_VISIBLE_DEVICES=0` with `--world-size 1`

### Server Arguments Reference

| Argument | Description |
|----------|-------------|
| `--model` | Model identifier (`openai/gpt-oss-120b`) |
| `--cache-dir` | Path to downloaded model checkpoints |
| `--world-size` | Number of GPUs |
| `--dist-init-addr` | Distributed init address (e.g., `localhost:33001`) |
| `--nnodes` | Number of nodes (always 1 for single-node) |
| `--node-rank` | Node rank (always 0 for single-node) |
| `--kv-dtype` | KV cache dtype (`bf16` recommended) |
| `--host-kv-cache-size` | Host KV cache in GB |
| `--enable-hugetlbfs` | Enable huge pages for better memory performance |
| `--gpu-memory-frac` | Fraction of GPU memory for KV cache (0.96 recommended) |
| `--storage-path` | Directory for batch files and results |
| `--save-result` | Save inference results to storage |

---

## 5. Submit Jobs with Python APIs

### Request Format with Reasoning Effort

GPT-OSS-120B supports the `reasoning_effort` parameter for extended thinking:

```jsonl
{"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-oss-120b", "messages": [{"role": "user", "content": "Solve: What is the integral of x^2 * e^x?"}], "max_completion_tokens": 4096, "reasoning_effort": "high"}}
{"custom_id": "req-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-oss-120b", "messages": [{"role": "user", "content": "What is AI?"}], "max_completion_tokens": 512}}
```

| Parameter | Values | Description |
|-----------|--------|-------------|
| `reasoning_effort` | `low`, `medium`, `high` | Controls reasoning depth. `high` enables extended thinking for complex problems. |

### Batch API

```python
from batchgen.batchgen_client import BatchGenHttpClient

# Connect to server
client = BatchGenHttpClient(base_url="http://localhost:10900")

# Submit batch job
result = client.submit_batch(
    input_file_path="requests.jsonl",
    output_file_path="results.jsonl",
    endpoint="/v1/chat/completions",
    max_decoding_length=4096,  # Fallback for requests without per-request max_completion_tokens
    temperature=0.7,
)

print(f"Batch completed: {result['id']}")
```

### Create Request File with Reasoning

```python
import json

requests = [
    {
        "custom_id": f"req-{i}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-oss-120b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
            "reasoning_effort": "high"
        }
    }
    for i, prompt in enumerate([
        "Prove that the square root of 2 is irrational.",
        "Write a Python function to solve the N-Queens problem.",
        "Explain the proof of Fermat's Last Theorem.",
    ])
]

with open("reasoning_requests.jsonl", "w") as f:
    for req in requests:
        f.write(json.dumps(req) + "\n")
```

---

## See Also

- [Server Flags Reference](server-flags.md)
- [Client API Reference](client-api.md)
- [DeepSeek-R1 Deployment Guide](deploy-deepseek-r1-h20.md)

## Support

For issues and questions:
- GitHub Issues: https://github.com/batchgen-project/batchgen/issues
