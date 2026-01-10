<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/8e70c5d1-6c3a-4507-b3b2-821ebf127989">
    <img src="https://github.com/user-attachments/assets/5587e43e-a2ef-4dde-a84c-365c31f284f8" width=55%>
  </picture>
</p>


<div align="center">
 <h3> High-throughput Offline Inference for MoE Models with Limited GPU Memory</h3>
  <strong><a href="#Installation"> Installation</a> | <a href="#Documentation">Documentation</a></strong>
</div>


# About
BatchGen is an efficient serving engine optimized specifically for **Mixture-of-Expert(MoE)** based large language models. It is tailored for bulk **offline inference** tasks and **limited GPU resources**. It enables low cost serving for latency-insensitive applications.

**Core Features**

- **Module-Based Batching**: A fine-grained batching strategy ensures consistently high GPU utilization throughout every forward pass.
- **Efficient Data Swapping Engine**: Supports inference of large-scale models (e.g., DeepSeek-R1) on constrained hardware setups such as single NVIDIA A5000 or RTX 4090 GPUs, aggressively maximizing overlap between computation and memory transfers to achieve optimal efficiency.
- **Tailored Offloading and Parallel Strategy**: Different parallel strategies, model weights offloaidng and KV-Cache offloading are applied to different models and hardware settings.


# Application Scenarios
- MoE model evaluation.
- Company deployed LLM workflow for raw data formation.
- Latency-insensitive bulk inference tasks. Such as large batch inference launched in valley period.
- Deep-research applications. Deliver high-quality results overnight.



# Supported Models
- **DeepSeek-R1/V3-671B. FULL Precision.**

# Supported Hardware
Hopper and Ampere archtecture are supported.

Recommended configurations for 8xH20, 8xA100 and 8xA5000 node are included in ./batchgen/configurations/


## Installation

### Hardware Requirements

**Host Memory**: Must be larger than the model size. Additional memory is used for the host KV cache, which stores KV states for sequences waiting to be processed. Larger host KV cache sizes generally result in better throughput as more sequences can be batched together.

For example, to serve DeepSeek-R1-671B (~700GB model weights), we recommend at least 1TB host memory: ~700GB for model weights + 300GB for host KV cache.

### Quick Install (Recommended)

```bash
git clone git@github.com:EfficientMoE/BatchGen.git
cd BatchGen
./scripts/install_deps.sh
```

This script automatically installs:
- PyTorch with CUDA 12.8 support
- flash-attention 3 (Hopper optimized)
- FlashMLA
- DeepGEMM
- BatchGen

Alternatively, use make:
```bash
make install-all
```

For manual installation or more control, see the [Manual Installation Guide](docs/manual-installation.md).

## Documentation

- **[Deployment Guide](docs/deploy-deepseek-r1-h20.md)** - Step-by-step guide for multi-node deployment
- **[Server Flags Reference](docs/server-flags.md)** - Complete list of all server configuration flags
- **[Manual Installation](docs/manual-installation.md)** - Step-by-step manual installation instructions


## Citation
```
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
