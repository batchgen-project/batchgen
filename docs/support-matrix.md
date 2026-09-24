# BatchGen support matrix

This page is the README-specific map of current model support. It is deliberately more conservative than a registry listing: a model can have a code path without having a validated multi-node deployment or repeatable performance data.

## Support levels

- **Documented deployment** — a model-specific guide describes the expected checkpoint format, topology, startup, and known constraints.
- **Benchmark evidence** — the repository contains a recent, reproducible measurement for the listed workload and hardware. This does not imply that BatchGen wins on every workload.
- **Experimental** — the path is available for development or evaluation, but coverage, accuracy, or performance evidence is incomplete.

## Matrix

| Model family | Approx. scale | Precision / path | Hardware or topology | Support level | Evidence and notes |
| --- | ---: | --- | --- | --- | --- |
| DeepSeek-R1 / V3 | 671B | FP8 | 2×8 H20 guide; H200 evaluation | Documented deployment · Benchmark evidence | Published OSDI'26 BCT results: 1.66× faster than SGLang-Opt on 16×H20 for 8K → 2K. |
| Kimi-K2.5 | 1.04T | INT4 / W4A16 | 16× H20 long-context campaign | Benchmark evidence | 255K prefill comparison uses SGLang 0.5.9 DP8/TP16: 1.61× faster in the recorded workload. |
| Kimi-K3 | 2.8T | MXFP4, 896 experts | 2×8 H200; 4×8 H20 guides | Documented deployment · Benchmark evidence | Exact-64K comparison uses SGLang 0.5.18 TP16/EP16: 1.42× faster in the recorded workload. |
| GLM-5 / GLM-5.1 | 754B (≈40B active) | FP8 | H20 support note | Documented deployment | Deployment and validation instructions are available; performance coverage is workload-dependent. |
| GLM-5.2 | 753B MoE | FP8 | 1×8 H200 guide and long-context prefill campaign | Documented deployment · Benchmark evidence | The pure-DP8 `8×1M` comparison uses the best completed SGLang 0.5.18 chunk/offload point: 1.71× faster in the recorded workload. Final installed-wheel replay remains required. |
| GPT-OSS-120B | 117B (≈5.1B active) | MXFP4 | H20 deployment guide | Documented deployment · Experimental | Functional path is documented; no comparable recent end-to-end advantage is claimed here. |
| MiniMax-M2.5 | 230B (≈10B active) | FP8 | H20 testbed | Experimental | Registered and tested; broader deployment and performance coverage is still being expanded. |
| Kimi-Linear-48B-A3B | 48B | BF16 | Development testbed | Experimental | Useful for integration testing; not a current headline performance target. |

### How to read this table

“Benchmark evidence” means that the exact model, precision, topology, and workload are recorded. It is not a promise of a general speedup over vLLM or SGLang. In particular, long-context prefill, decode-heavy batches, host-memory pressure, and request-length distributions can change the relative result substantially.

For a production rollout, start from the deployment guide, confirm the checkpoint and topology, then run a representative batch from your own workload. For a research comparison, record the BatchGen commit, baseline version, GPU topology, input/output lengths, batch size, and whether host KV cache or offloading is enabled.

## Related guides

- [DeepSeek-R1 on H20](deploy-deepseek-r1-h20.md)
- [Kimi-K3 on H200](deploy-kimi-k3-h200.md)
- [Kimi-K3 on H20](deploy-kimi-k3-h20.md)
- [GPT-OSS-120B on H20](deploy-gpt-oss-h20.md)
- [GLM-5.2-FP8 on H200](deploy-glm-5.2-h200.md)
- [GLM-5.1 support notes](support-glm-5.1.md)
