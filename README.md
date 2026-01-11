<p align="center">
  <img src="assets/BatchGen_Icon.png" width=55%>
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

### Deployment

For complete deployment instructions including model download, checkpoint conversion, server setup, and submitting jobs, see the **[Deployment Guide](docs/deploy-deepseek-r1-h20.md)**.

---

## Documentation

- **[Deployment Guide](docs/deploy-deepseek-r1-h20.md)** - Step-by-step guide for multi-node deployment
- **[Server Flags Reference](docs/server-flags.md)** - Complete list of all server configuration flags
- **[Client API Reference](docs/client-api.md)** - Python client usage and parameters
- **[Manual Installation](docs/manual-installation.md)** - Step-by-step manual installation instructions

---

## Acknowledgements

We learned from the following projects when building BatchGen:

- [SGLang](https://github.com/sgl-project/sglang) - High-performance serving framework for LLMs
- [vLLM](https://github.com/vllm-project/vllm) - High-throughput and memory-efficient inference engine

---
