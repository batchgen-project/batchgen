<p align="center">
  <img src="assets/BatchGen_Icon.png" width=55%>
</p>

<div align="center">

[![GitHub Stars](https://img.shields.io/github/stars/batchgen-project/batchgen?style=social)](https://github.com/batchgen-project/batchgen/stargazers)
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

## Performance

### Kimi-K2.5 (1T)

Batch completion time (BCT) in seconds for 16 input requests on H20 GPUs. Lower is better. SGLang uses its default settings, while SGLang-Opt is tuned for higher throughput. Speedup is measured for BatchGen over SGLang-Opt.

| Deployment | Workload | SGLang | SGLang-Opt | BatchGen | Speedup |
|:-----------|:---------|-------:|-----------:|---------:|--------:|
| 1×8 H20 | 64K-1 | 233.1 | 233.1 | **160.2** | **1.46×** |
| 1×8 H20 | 128K-1 | 637.5 | 605.6 | **455.9** | **1.33×** |
| 1×8 H20 | 255K-1 | OOM | OOM | **1497.8** | — |
| 2×8 H20 | 64K-1 | 106.2 | 106.2 | **86.7** | **1.22×** |
| 2×8 H20 | 128K-1 | 528.8 | 304.8 | **234.3** | **1.30×** |
| 2×8 H20 | 255K-1 | 1912.6 | 1202.9 | **748.4** | **1.61×** |
| 2×8 H20 | 64K-1K | 213.5 | 182.9 | **146.8** | **1.25×** |
| 2×8 H20 | 128K-1K | 559.3 | 456.0 | **303.6** | **1.50×** |
| 2×8 H20 | 255K-1K | 1829.5 | 1268.6 | **838.5** | **1.51×** |

BatchGen is the only system in this comparison that serves the 255K workload on a single node.


## Getting Started

### Hardware Requirements

**Host Memory**: Must be larger than the model size. Additional memory is used for the host KV cache, which stores KV states for sequences waiting to be processed. Larger host KV cache sizes generally result in better throughput.

For DeepSeek-R1-671B (~700GB model weights), we recommend at least 1TB host memory: ~700GB for model weights + 300GB for host KV cache.

### Installation

```bash
git clone https://github.com/batchgen-project/batchgen.git
cd batchgen
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
