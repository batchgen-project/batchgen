# BatchGen

<p align="center">
  <img src="assets/BatchGen_Icon.png" width="55%" alt="BatchGen">
</p>

<div align="center">

[![GitHub Stars](https://img.shields.io/github/stars/batchgen-project/batchgen?style=social)](https://github.com/batchgen-project/batchgen/stargazers)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**High-throughput batch inference for large MoE models**

[Support matrix](docs/support-matrix.md) · [Deployment guides](docs/) · [Batch API](docs/batch-api-guide.md) · [Contributing](CONTRIBUTING.md) · [Paper](https://www.usenix.org/conference/osdi26/presentation/xu-tairan)

</div>

---

## News

- **2026/09 — GLM-5.3-FP8 support.** Added the dedicated GLM-5.3 chat
  template, reasoning parser, request controls, and an 8×H200 deployment
  guide. H200 short correctness and structured `reasoning_content` checks pass;
  the 2,048-sequence MMLU-Pro qualification is still running.
- **2026/09 — GLM-5.2-FP8 on 8×H200.** Added a one-node deployment guide and
  source-qualified `8×1M` plus `32×256K` accumulated-token prefill. On the matched
  `8×1M` workload, BatchGen reached 20,648 prompt tok/s, **1.71× faster** than
  the best completed point in the recorded SGLang 0.5.18 tuning sweep.
- **2026/09 — Kimi-K3** (2.8T-parameter, 896-expert MXFP4, top-16) support —
  multi-node deployment on [2×8 H200](docs/deploy-kimi-k3-h200.md) or
  [4×8 H20](docs/deploy-kimi-k3-h20.md).
- **2026/01 — BatchGen v1.0** released with support for DeepSeek-R1/V3-671B.

## What is BatchGen?

BatchGen is a batch-inference engine for large language models, with a focus on sparse mixture-of-experts (MoE) models and long-context workloads. It minimizes **batch completion time (BCT)**—the time to finish a batch of requests—by coordinating sequence-level scheduling, expert-level batching, and host/device KV-cache movement across GPU clusters.

BatchGen is intended for both production users and researchers: the same batch API can drive offline generation, evaluation, synthetic-data pipelines, test-time scaling, and RL rollouts while exposing the scheduling and systems ideas needed for experimentation.

### Why BatchGen

- **Batch-first execution.** Optimize the completion time of a whole workload, not only single-request latency.
- **Long-context prefill.** Keep large prompt batches productive when model weights and KV state exceed a single GPU's capacity.
- **MoE-aware scheduling.** Reorganize sequence work to form larger expert-level batches and reduce long-tail stragglers.
- **Host KV cache.** Use CPU memory as an extension of the KV-cache hierarchy for workloads that cannot fit entirely on device.
- **Research-friendly controls.** The batch API and deployment scripts make it straightforward to reproduce experiments and compare scheduling policies.

## Selected results

The numbers below are workload-specific measurements, not a universal ranking. Comparisons use the same request set and report the measured wall-clock or throughput metric for that workload. “Faster” is computed as baseline divided by BatchGen.

### Recent long-context engineering snapshots

| Model and hardware | Workload | BatchGen | Reference | Advantage |
| --- | --- | ---: | ---: | ---: |
| Kimi-K3 (2.8T), 2×8 H200 | Exact 64K-token prefill | 116.7 s service wall | SGLang 0.5.18 · TP16/EP16: 166.3 s | **1.42× faster** |
| Kimi-K2.5 (1.04T), 16× H20 | 255K-token prefill, 16 sequences | 748.9 s | SGLang 0.5.9 · DP8/TP16: 1,202.9 s | **1.61× faster** |
| [GLM-5.2-FP8](https://huggingface.co/zai-org/GLM-5.2-FP8) (753B), 8× H200 | 8×1M accumulated-token prefill, pure DP8 | 20,648 prompt tok/s | SGLang 0.5.18 · DP8 attention / TP8 MoE, tuned CPU offload + chunk: 12,106 prompt tok/s | **1.71× faster** |

These snapshots come from the latest gated campaigns available to this repository. The GLM-5.2 row is the exact eight-prompt, one-request-per-rank workload; it covers prefill and the first sampled token rather than sustained 1M-context decode.

Measurement provenance: Kimi-K3 uses the `ae374617` campaign cohort; Kimi-K2.5 uses the 2026-07-10 long-context campaign; GLM-5.2 uses source assembly `954b2bc6` and a bracketed SGLang 0.5.18 chunk/offload sweep from 2026-09-23. The GLM source result still requires final installed-wheel replay. Re-run with the exact topology and baseline versions before using these figures for capacity planning.

The reference column reports the recorded configuration rather than claiming a global optimum. Future hardware-specific tuning campaigns should publish their search space and replace these rows only with directly comparable measurements.

## Supported models and hardware

The [support matrix](docs/support-matrix.md) separates a registered model path from a documented deployment and from a workload with repeatable performance evidence. In short, current deployment guides cover DeepSeek-R1, Kimi-K3, GPT-OSS-120B, GLM-5.1, GLM-5.2, and GLM-5.3; recent benchmark evidence also covers Kimi-K2.5. Experimental paths include MiniMax-M2.5, Kimi-Linear, and related MoE variants. H20 and H200 are the primary validated accelerators; exact model/hardware topology matters.

## Quick start

### Installation

If repository access is restricted, authenticate with GitHub first. The supported full installation is:

```bash
git clone https://github.com/batchgen-project/batchgen.git
cd batchgen
./scripts/install_deps.sh --all
```

`--all` installs the complete runtime: the matching PyTorch build plus FlashAttention, FlashMLA, DeepGEMM, the compiled `batchgen_kernels` package, and BatchGen. The current no-argument form is retained as an alias for the same full install. Use `--from-source` when you need to build the dependencies locally; component flags such as `--flash-attn` or `--batchgen` are available for targeted development installs.

For a manual or component-by-component setup, see [INSTALL.md](docs/INSTALL.md).

### Deploy a server

Choose the guide that matches your model and topology:

- [GLM-5.2-FP8 on H200](docs/deploy-glm-5.2-h200.md)
- [GLM-5.3-FP8 on H200](docs/deploy-glm-5.3-h200.md)
- [DeepSeek-R1 on H20](docs/deploy-deepseek-r1-h20.md)
- [Kimi-K3 on H200](docs/deploy-kimi-k3-h200.md)
- [Kimi-K3 on H20](docs/deploy-kimi-k3-h20.md)
- [GPT-OSS-120B on H20](docs/deploy-gpt-oss-h20.md)

### Submit a batch

After the server is healthy, submit requests through the batch API:

```python
from batchgen.batchgen_client import BatchGenHttpClient

client = BatchGenHttpClient("http://localhost:10900")
batch = client.submit_batch(
    input_file_path="input.jsonl",
    output_file_path="output.jsonl",
    max_decoding_length=256,
)
print(batch["status"])
```

See the [batch API guide](docs/batch-api-guide.md) for JSONL input, polling, retries, and result retrieval.

## Documentation

- [Support matrix](docs/support-matrix.md) — models, hardware, maturity, and evidence level
- [GLM-5.2-FP8 on H200](docs/deploy-glm-5.2-h200.md) — download, conversion, one-node launch, 1M-context qualification, and memory limits
- [GLM-5.3-FP8 on H200](docs/deploy-glm-5.3-h200.md) — local-SSD staging, one-node launch, reasoning controls, and qualification status
- [Installation](docs/INSTALL.md) — dependency and source-install details
- [Batch API guide](docs/batch-api-guide.md) — request format and lifecycle
- [Server flags](docs/server-flags.md) — runtime configuration
- [Deployment guides](docs/) — model-specific startup and troubleshooting

## Roadmap

- Stabilize partition and migration primitives.
- Develop more adaptive scheduling policies for resource utilization and workload balance.
- Expand documented model and hardware coverage.
- Support OpenAI-compatible tool calling.

## Citation

We ask that any work, product, or publication that uses BatchGen or code derived from it cite the BatchGen repository and/or the BatchGen paper. GitHub's "Cite this repository" button exports the same metadata from [CITATION.cff](CITATION.cff).

```bibtex
@inproceedings{xu2026batchgen,
  author    = {Tairan Xu and Leyang Xue and Zhan Lu and Jinfu Deng and Hongyang Xiao and Yinsicheng Jiang and Congjie He and Matej Sandor and Le Xu and Luo Mai},
  title     = {{BatchGen}: An Architecture for Scalable and Efficient Batch Inference},
  booktitle = {20th USENIX Symposium on Operating Systems Design and Implementation (OSDI 26)},
  year      = {2026},
  address   = {Seattle, WA},
  pages     = {1125--1141},
  publisher = {USENIX Association},
  month     = jul,
  url       = {https://www.usenix.org/conference/osdi26/presentation/xu-tairan}
}
```

Paper: [BatchGen: An Architecture for Scalable and Efficient Batch Inference](https://www.usenix.org/conference/osdi26/presentation/xu-tairan).

## Acknowledgements

BatchGen learns from and draws on the ecosystem work of [SGLang](https://github.com/sgl-project/sglang) and [vLLM](https://github.com/vllm-project/vllm), among other open-source projects. Parts of our FP8 blockwise grouped GEMM and GQA decode kernels are adapted from Tencent's [hpc-ops](https://github.com/Tencent/hpc-ops); third-party components and their licenses are listed in [NOTICE](NOTICE).

## Contributing

We welcome model integrations, kernels, scheduler improvements, evaluation tooling, documentation, and bug fixes. Start with [CONTRIBUTING.md](CONTRIBUTING.md), then check the [PR Merge Policy Contract](PR_MERGE_POLICY.md) before opening a pull request.

## License

BatchGen is released under the [Apache 2.0 license](LICENSE). Redistributions must retain the [NOTICE](NOTICE) file, which includes the attribution and citation request above.
