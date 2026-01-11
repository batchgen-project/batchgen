<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/8e70c5d1-6c3a-4507-b3b2-821ebf127989">
    <img src="https://github.com/user-attachments/assets/5587e43e-a2ef-4dde-a84c-365c31f284f8" width=55%>
  </picture>
</p>

<div align="center">

[![GitHub Stars](https://img.shields.io/github/stars/EfficientMoE/BatchGen?style=social)](https://github.com/EfficientMoE/BatchGen/stargazers)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**High-Throughput Batch Inference**

[Documentation](docs/) | [Deployment Guide](docs/deploy-deepseek-r1-h20.md) | [Server Flags](docs/server-flags.md)

</div>

---

## News

- [2026/01] BatchGen v1.0 released with support for DeepSeek-R1/V3-671B.

---

## About

BatchGen is a high-throughput batch inference engine designed to minimize batch completion time (BCT) for large-scale batch workloads and MoE-based LLMs.

### Key Innovations

BatchGen introduces the sequence coroutine compute model, which treats each sequence's computation as an event-driven coroutine that can be paused, resumed, and reorganized. A static planner optimizes batch configurations through lightweight profiling, while a dynamic sequence scheduler yields, combines, and migrates sequence coroutines at runtime—enabling larger expert-level batches for sparse MoE models, mitigating long-tail stragglers, and maintaining high device utilization across GPU clusters.

### Application Scenarios

- Large-scale offline inference and data processing pipelines
- Synthetic data generation
- Model evaluation and benchmarking
- Test-time scaling (e.g., chain-of-thought, self-consistency)
- RL rollouts and post-training workflows


## Getting Started

### Hardware Requirements

**Host Memory**: Must be larger than the model size. Additional memory is used for the host KV cache, which stores KV states for sequences waiting to be processed. Larger host KV cache sizes generally result in better throughput.

For DeepSeek-R1-671B (~700GB model weights), we recommend at least 1TB host memory: ~700GB for model weights + 300GB for host KV cache.

### Installation

```bash
git clone https://github.com/EfficientMoE/BatchGen.git
cd BatchGen
./scripts/install_deps.sh
```

This script automatically installs:
- PyTorch with CUDA 12.8 support
- flash-attention 3 (Hopper optimized)
- FlashMLA
- DeepGEMM
- BatchGen

For manual installation, see the [Manual Installation Guide](docs/manual-installation.md).

### Quick Start

**1. Download Model Checkpoints**

```bash
huggingface-cli download deepseek-ai/DeepSeek-R1 \
    --local-dir /shared/models/DeepSeek-R1
```

**2. Start the Server**

```bash
python -m batchgen.launch_http_server \
    --model deepseek-ai/DeepSeek-R1 \
    --cache-dir /shared/models/DeepSeek-R1 \
    --host-kv-cache-size 256
```

**3. Submit Batch Jobs**

```python
from batchgen.batchgen_client import BatchGenHttpClient

client = BatchGenHttpClient(host="localhost", port=10900)

result = client.submit_batch(
    input_file_path="requests.jsonl",
    output_file_path="results.jsonl",
    endpoint="/v1/chat/completions",
)
```

For multi-node deployment, see the [Deployment Guide](docs/deploy-deepseek-r1-h20.md).

---

## Documentation

- **[Deployment Guide](docs/deploy-deepseek-r1-h20.md)** - Step-by-step guide for multi-node deployment
- **[Server Flags Reference](docs/server-flags.md)** - Complete list of all server configuration flags
- **[Manual Installation](docs/manual-installation.md)** - Step-by-step manual installation instructions

---

## Acknowledgements

We learned from the following projects when building BatchGen:

- [SGLang](https://github.com/sgl-project/sglang) - High-performance serving framework for LLMs
- [vLLM](https://github.com/vllm-project/vllm) - High-throughput and memory-efficient inference engine

---
