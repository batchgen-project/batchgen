<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/8e70c5d1-6c3a-4507-b3b2-821ebf127989">
    <img src="https://github.com/user-attachments/assets/5587e43e-a2ef-4dde-a84c-365c31f284f8" width=55%>
  </picture>
</p>

<div align="center">

[![GitHub Stars](https://img.shields.io/github/stars/EfficientMoE/BatchGen?style=social)](https://github.com/EfficientMoE/BatchGen/stargazers)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**High-throughput Offline Inference for MoE Models with Limited GPU Memory**

[Documentation](docs/) | [Deployment Guide](docs/deploy-deepseek-r1-h20.md) | [Server Flags](docs/server-flags.md)

</div>

---

## News

- [2025/01] BatchGen v1.0 released with support for DeepSeek-R1/V3-671B full precision inference.

---

## About

BatchGen is an efficient serving engine optimized specifically for **Mixture-of-Expert (MoE)** based large language models. It is designed for bulk **offline inference** tasks on **limited GPU resources**, enabling low-cost serving for latency-insensitive applications.

### Core Features

- **Module-Based Batching**: A fine-grained batching strategy ensures consistently high GPU utilization throughout every forward pass.

- **Efficient Data Swapping Engine**: Supports inference of large-scale models (e.g., DeepSeek-R1) on constrained hardware setups such as single NVIDIA A5000 or RTX 4090 GPUs, aggressively maximizing overlap between computation and memory transfers to achieve optimal efficiency.

- **Tailored Offloading and Parallel Strategy**: Different parallel strategies, model weights offloading and KV-Cache offloading are applied to different models and hardware settings.

### Application Scenarios

- MoE model evaluation
- Company deployed LLM workflow for raw data formation
- Latency-insensitive bulk inference tasks (e.g., large batch inference during off-peak hours)
- Deep-research applications that deliver high-quality results overnight

### Supported Models

| Model | Precision | Status |
|-------|-----------|--------|
| DeepSeek-R1-671B | Full (BF16/FP8) | Supported |
| DeepSeek-V3-671B | Full (BF16/FP8) | Supported |

### Supported Hardware

- **Hopper Architecture**: H100, H20
- **Ampere Architecture**: A100, A5000, RTX 4090

Recommended configurations for 8xH20, 8xA100, and 8xA5000 nodes are included in `./batchgen/configurations/`.

---

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
- [FlashMLA](https://github.com/deepseek-ai/FlashMLA) - Optimized Multi-head Latent Attention for DeepSeek models
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) - FP8 GEMM kernels for Hopper GPUs

---

## Citation

```bibtex
@misc{xu2025moegenhighthroughputmoeinference,
      title={BatchGen: High-Throughput MoE Inference on a Single GPU with Module-Based Batching},
      author={Tairan Xu and Leyang Xue and Zhan Lu and Adrian Jackson and Luo Mai},
      year={2025},
      eprint={2503.09716},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2503.09716},
}
```
