# Deploy GPT-OSS-120B on Single H20 GPU

This guide provides step-by-step instructions for deploying OpenAI's GPT-OSS-120B model on a single NVIDIA H20 GPU (96GB) using BatchGen.

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Download Model Checkpoints](#2-download-model-checkpoints)
3. [Convert Checkpoints to BatchGen Format](#3-convert-checkpoints-to-batchgen-format)
4. [Build Docker Container](#4-build-docker-container)
5. [Start BatchGen Server](#5-start-batchgen-server)
6. [Submit Jobs with Python APIs](#6-submit-jobs-with-python-apis)
7. [Verify MXFP4 Dequantization](#7-verify-mxfp4-dequantization)

---

## 1. Prerequisites

### Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA H20 (96GB HBM3) |
| Compute Capability | SM90a (9.0+) |
| Host Memory | 256GB+ recommended |
| Storage | 100GB+ SSD for checkpoints |

### Software Requirements

| Software | Version |
|----------|---------|
| CUDA | 12.1+ |
| Python | 3.10+ |
| PyTorch | 2.1+ |
| Triton | 2.1+ |

### Verify H20 GPU

```bash
# Check GPU model and compute capability
nvidia-smi --query-gpu=name,compute_cap --format=csv
# Expected: NVIDIA H20, 9.0

# Verify CUDA version
nvcc --version
# Expected: CUDA 12.1 or higher
```

---

## 2. Download Model Checkpoints

Download GPT-OSS-120B from HuggingFace. The model uses MXFP4 quantization (~55GB).

```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Login to HuggingFace (required for gated models)
huggingface-cli login

# Download GPT-OSS-120B to local storage
huggingface-cli download openai/gpt-oss-120b \
    --local-dir /path/to/models/gpt-oss-120b \
    --local-dir-use-symlinks False
```

### Verify Download

```bash
ls -la /path/to/models/gpt-oss-120b/
# Should see files like:
# - model-*.safetensors (MXFP4 quantized weights)
# - config.json
# - tokenizer.json
# - tokenizer_config.json
```

---

## 3. Convert Checkpoints to BatchGen Format

BatchGen uses a custom checkpoint format optimized for peak SSD read performance.

```bash
# Convert all checkpoint files
python -m batchgen.tools.convert_checkpoint \
    --input-dir /path/to/models/gpt-oss-120b

# This creates /path/to/models/gpt-oss-120b/converted_ckpt/
```

### Validate Conversion

```bash
python -m batchgen.tools.convert_checkpoint \
    --input-dir /path/to/models/gpt-oss-120b \
    --validate-only
```

---

## 4. Build Docker Container

```bash
# From BatchGen project root
docker buildx build --progress=plain -f docker/Dockerfile -t batchgen:latest .
```

### Run Container

```bash
docker run -it \
    --runtime=nvidia \
    --gpus all \
    --network=host \
    -v /path/to/models:/models:ro \
    -v /path/to/storage:/storage \
    batchgen:latest
```

---

## 5. Start BatchGen Server

### Mount Shared Memory

```bash
# Ensure /dev/shm has sufficient size
df -h /dev/shm

# Mount with appropriate size (adjust based on host memory)
sudo mount -o remount,size=200G /dev/shm
```

### Single H20 GPU Deployment

```bash
python -m batchgen.launch_http_server \
    --model openai/gpt-oss-120b \
    --cache-dir /models/gpt-oss-120b \
    --listen-port 10900 \
    --world-size 1 \
    --storage-path /storage \
    --host-kv-cache-size 128
```

### Server Arguments for GPT-OSS-120B

| Argument | Recommended Value | Description |
|----------|-------------------|-------------|
| `--model` | `openai/gpt-oss-120b` | Model identifier |
| `--cache-dir` | `/models/gpt-oss-120b` | Path to downloaded model |
| `--world-size` | `1` | Single GPU deployment |
| `--host-kv-cache-size` | `128` | Host KV cache in GB |
| `--listen-port` | `10900` | HTTP server port |

### Memory Budget (Single H20, 96GB)

| Component | Memory |
|-----------|--------|
| MXFP4 Expert Weights | ~55 GB |
| BF16 Attention Weights | ~3 GB |
| Embeddings + LM Head | ~2 GB |
| KV Cache (batch=16) | ~3 GB |
| Activations + Workspace | ~8 GB |
| **Total** | **~71 GB** |
| **Headroom** | **~25 GB** |

---

## 6. Submit Jobs with Python APIs

### Batch API (Recommended)

```python
from batchgen.batchgen_client import BatchGenHttpClient

# Connect to server
client = BatchGenHttpClient(base_url="http://localhost:10900")

# Submit batch job
result = client.submit_batch(
    input_file_path="requests.jsonl",
    output_file_path="results.jsonl",
    endpoint="/v1/chat/completions",
    max_decoding_length=1024,
    temperature=0.7,
)

print(f"Batch completed: {result['id']}")
```

### Direct Inference

```python
from batchgen.batchgen_client import BatchGenHttpClient

client = BatchGenHttpClient(base_url="http://localhost:10900")

results = client.submit_inference(
    prompts=["What is machine learning?", "Explain neural networks."],
    max_output_len=256,
    temperature=0.7,
)

for i, result in enumerate(results):
    print(f"Response {i}: {result}")
```

### Request Format

```jsonl
{"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-oss-120b", "messages": [{"role": "user", "content": "What is AI?"}], "max_tokens": 512}}
```

---

## 7. Verify MXFP4 Dequantization

Run the MXFP4 unit tests to verify dequantization correctness on your H20 GPU.

### Run Unit Tests

```bash
cd /path/to/BatchGen

# Run MXFP4 dequantization tests
python -m pytest batchgen/quantization/test_mxfp4.py -v

# Expected output:
# test_fp4_lookup_values PASSED
# test_unpack_nibbles PASSED
# test_scale_application PASSED
# test_negative_exponent PASSED
# test_negative_fp4_values PASSED
# test_multiple_scales PASSED
# test_2d_input PASSED
# test_triton_vs_reference PASSED
```

### Run Standalone Test

```bash
python batchgen/quantization/test_mxfp4.py
```

### Benchmark Dequantization Performance

```python
import torch
from batchgen.quantization.mxfp4 import mxfp4_dequantize

# Simulate GPT-OSS-120B expert weights
M, K = 2880, 2880  # hidden_size x intermediate_size
packed = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device="cuda")
scales = torch.randint(100, 154, (M, K // 32), dtype=torch.uint8, device="cuda")

# Warmup
for _ in range(3):
    _ = mxfp4_dequantize(packed, scales)

# Benchmark
torch.cuda.synchronize()
import time
start = time.perf_counter()
for _ in range(100):
    result = mxfp4_dequantize(packed, scales)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

print(f"MXFP4 dequant: {elapsed/100*1000:.3f} ms/iter")
print(f"Throughput: {M * K * 2 / (elapsed/100) / 1e9:.2f} GB/s")
```

---

## Model Architecture Summary

| Feature | Value |
|---------|-------|
| Parameters | 117B (5.1B active) |
| Layers | 36 |
| Hidden Size | 2,880 |
| Attention | GQA: 64 heads, 8 KV heads |
| Head Dim | 64 |
| Experts | 128 total, Top-4 routing |
| Expert FFN | SwiGLU, intermediate=2880 |
| Context | 131K tokens |
| Quantization | MXFP4 (~4.25 bits/param) |
| Attention Pattern | Alternating sliding (128) / full |
| RoPE | YaRN: theta=150000, factor=32 |

---

## Troubleshooting

### CUDA Out of Memory

If you encounter OOM errors:

1. Reduce `--host-kv-cache-size`
2. Reduce batch size in requests
3. Reduce `max_tokens` in requests

### MXFP4 Dequantization Errors

Ensure your GPU supports compute capability 9.0+:

```bash
python -c "import torch; print(torch.cuda.get_device_capability())"
# Expected: (9, 0) for H20
```

### Slow Inference

1. Verify SSD performance (sequential read > 3 GB/s recommended)
2. Ensure `/dev/shm` is properly mounted
3. Check host KV cache size is maximized

---

## See Also

- [Server Flags Reference](server-flags.md)
- [Client API Reference](client-api.md)
- [DeepSeek-R1 Deployment Guide](deploy-deepseek-r1-h20.md)

## Support

For issues and questions:
- GitHub Issues: https://github.com/EfficientMoE/BatchGen/issues
