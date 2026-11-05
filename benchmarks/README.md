# Decode benchmark reproduction

This directory contains the exact scripts behind the end-to-end batch decode
numbers in the top-level README, so the measurements can be reproduced against
any OpenAI-compatible serving endpoint.

All comparisons follow the same contract:

- **Same request set.** Both engines consume the *same* workload JSONL
  (token-identical prompts, same sampling parameters, same arrival order seed).
- **Tuned baselines only.** A reference number is published only when the
  baseline engine went through a tuning sweep on the same hardware and
  workload; the best completed configuration is reported and documented below.
- **Batch wall clock.** The metric is the client-observed wall time from the
  first request issued to the last completion received (batch completion time),
  with all requests offered at once.

## Scripts

| File | Purpose |
| --- | --- |
| `decode_workload.py` | Builds a workload JSONL + manifest from public datasets (MMLU-Pro, LongBench, NarrativeQA, LiveCodeBench) for a named profile and tokenizer. |
| `workload_policy.py` | Loads and validates the profile registry; the builder and this README share it. |
| `configs/decode_workloads.json` | The profile registry: every parameter that affects the token stream (strata, sampling, completion caps, selection seeds) is fixed here, not on the command line. |
| `nqa_prep.py` | Downloads the parquet conversion of `deepmind/narrativeqa` (test split) and prepares `stories.jsonl` for the grouped long-prefix profile. |
| `nqa_prep_distinct.py` | Same download over the train+test pool, keeping every book long enough for a 64K-token truncation; prepares the distinct long-context profile. |
| `decode_batch_driver.py` | Issues the workload against a server (`--mode concurrent-chat` posts every row at once to `/v1/chat/completions`; `--mode batch-api` uses the BatchGen batch API) and reports the batch wall time. |
| `mmlu_pro_test.py` | Prompt templates and answer extraction used by the workload builder. |

The builder takes the profile name and row count; everything else that shapes
the tokens comes from the registry. `--mmlu-dir` is a required argument even
for profiles that do not sample MMLU-Pro. `--gpu-pool-pages` is the target
engine's per-rank GPU KV page count (printed at server start); profiles whose
rows cannot all be GPU-resident at once are built with
`--allow-oversubscribed` and say so in their manifest.

## Workload profiles

### `nqa64k-grouped-v1` — document-grouped long shared prefixes

64 distinct NarrativeQA books, each truncated to a 64K-token prefix, with 8
questions per book: 512 requests totalling ~32.8M prompt tokens. Groups share
their prefix; different groups are disjoint. EOS is honoured with a 4,096-token
completion cap, temperature 1.0, thinking enabled. This is the RAG/document-QA
shape: many questions over the same long context. The theoretical
cross-request reuse ceiling is 87% of prompt tokens.

```bash
python nqa_prep.py nqa_data/
python decode_workload.py build \
    --profile nqa64k-grouped-v1 --rows 512 \
    --model zai-org/GLM-5.3-FP8 \
    --batchgen-root /path/to/batchgen \
    --mmlu-dir /path/to/datasets/mmlu_pro \
    --narrativeqa-dir nqa_data/ \
    --gpu-pool-pages <per-rank pages> --allow-oversubscribed \
    --out-dir workloads/
python decode_batch_driver.py \
    --name nqa64k-run \
    --input workloads/nqa64k-grouped-v1_n512.jsonl \
    --workload-manifest workloads/nqa64k-grouped-v1_n512.manifest.json \
    --out-dir results/nqa64k-run \
    --base-url http://127.0.0.1:30000 \
    --mode concurrent-chat \
    --arrival-order-seed 20261013 \
    --poll-seconds 30 --timeout-seconds 14400
```

### `nqa64k-distinct-v1` — long contexts with no cross-request sharing

312 distinct NarrativeQA books (train+test pool), each truncated to a
64K-token prefix, one question per book. Cross-request prefix overlap is
~0.1% of prompt tokens — the controlled counterpart of the grouped profile:
same documents, same lengths, no reuse to exploit. EOS is honoured with a
4,096-token completion cap, temperature 1.0, thinking enabled.

The batch's KV footprint (~20M prompt tokens) is several times larger than
the serving node's device KV pool, so this profile specifically exercises
host-side KV offloading and prefill admission planning: an engine must either
extend KV capacity beyond HBM or pay recompute to rotate the batch.

```bash
python nqa_prep_distinct.py nqa_distinct_data/
mv nqa_distinct_data/stories_distinct.jsonl nqa_distinct_data/stories.jsonl
python decode_workload.py build \
    --profile nqa64k-distinct-v1 --rows 312 \
    --model zai-org/GLM-5.3-FP8 \
    --batchgen-root /path/to/batchgen \
    --mmlu-dir /path/to/datasets/mmlu_pro \
    --narrativeqa-dir nqa_distinct_data/ \
    --gpu-pool-pages <per-rank pages> --allow-oversubscribed \
    --out-dir workloads/
```

### `lcb-fixed4096-v1` — code-generation long decode

512 distinct LiveCodeBench (`code_generation_lite`) medium/hard
competitive-programming prompts at natural length, each with a forced
4,096-token completion (`ignore_eos`), so every configuration decodes exactly
the same token volume. This is the long-forced-decode shape: prefill is small,
and batch completion time is dominated by sustained decode throughput.

```bash
python decode_workload.py build \
    --profile lcb-fixed4096-v1 --rows 512 \
    --model moonshotai/Kimi-K3 \
    --batchgen-root /path/to/batchgen \
    --mmlu-dir /path/to/datasets/mmlu_pro \
    --livecodebench-dir /path/to/datasets/livecodebench \
    --gpu-pool-pages <per-rank pages> \
    --out-dir workloads/
```

### `k3-diverse-v1` — diverse mixed eval decode

512 distinct requests (384 MMLU-Pro, 128 LongBench) with stratified prompt
lengths, temperature 1.0, and a forced 1,024-token completion per request
(`ignore_eos`), so every configuration decodes exactly the same token volume.

Disclosure: the MMLU-Pro rows share few-shot template content — about 29% of
the batch's prompt characters recur across rows. The tuned baseline's radix
cache was measured with this sharing available and showed no measurable
wall-time benefit from it in the sweep (on/off within run-to-run noise), so
the sharing does not materially favour either engine on this profile.

```bash
python decode_workload.py build \
    --profile k3-diverse-v1 --rows 512 \
    --model moonshotai/Kimi-K3 \
    --batchgen-root /path/to/batchgen \
    --mmlu-dir /path/to/datasets/mmlu_pro \
    --longbench-dir /path/to/datasets/longbench \
    --gpu-pool-pages <per-rank pages> \
    --out-dir workloads/
```

## Published rows and the tuned baseline configurations

### GLM-5.3-FP8, 8×H200, `nqa64k-grouped-v1` (512 requests)

- **BatchGen** v1.1.0, single node, launched per
  [`docs/deploy-glm-5.3-h200.md`](../docs/deploy-glm-5.3-h200.md) with the host
  KV cache enabled; page-level prefix reuse stores each 64K group prefix once.
- **SGLang 0.5.21 baseline (tuned sweep, best arm reported).** All arms used
  `--tp-size 8 --dp-size 8 --enable-dp-attention --ep-size 8
  --moe-a2a-backend deepep --deepep-mode auto --max-running-requests 1024
  --cuda-graph-max-bs-decode 128` on the same node. Swept dimensions:
  - memory budget: `--mem-fraction-static` 0.90 (0.905/0.91/0.92 fail to boot)
    vs 0.85 with `--chunked-prefill-size`/`--max-prefill-tokens` 65536
    (chunked prefill is clamped to 8192 by the server under DP attention);
  - scheduling: default FCFS vs `--schedule-policy lpm`
    (longest-prefix-match);
  - routing: in-process round-robin dispatch vs the external
    `sglang-router` 0.3.2 gateway with `--policy cache_aware --dp-aware`,
    which routes each request to the data-parallel rank whose radix cache
    holds the longest matching prefix. **The router arm is the best arm** and
    is the published reference: it reaches a measured 85% prefix-cache hit
    rate on this workload, essentially the profile's 87% ceiling.
- **Router gateway settings for long-prefill batches.** A 512×64K batch
  offered at once sits in the server queue far longer than the router's
  defaults tolerate, and the router's liveness checks mistake a server
  saturated by its first heavy prefill wave for a dead one. The published
  router arm uses: `--request-timeout-secs 14400 --disable-circuit-breaker
  --disable-health-check`, retries left enabled (they only re-send requests
  that never reached a server), `ulimit -n 65535` in the launch shell, and
  `SGLANG_TIMEOUT_KEEP_ALIVE=14400` on the servers. In addition, a freshly
  started engine drops part of the first heavy prefill wave sent through the
  router regardless of these settings, so the protocol runs one *disjoint*
  warm batch (short, unrelated prompts) through the router first; the
  measured batch still sees a completely cold radix cache because its
  prompts share nothing with the warm batch.

### GLM-5.3-FP8, 8×H200, `nqa64k-distinct-v1` (312 requests)

- **BatchGen**: the grouped-row deployment unchanged. The published number is
  measured on v1.1.0 plus
  [#588](https://github.com/batchgen-project/batchgen/pull/588) (merged to
  `main` immediately after the release), which removed a host-KV eviction
  policy that forced full-prompt recomputes on exactly this no-sharing shape.
  The batch admits in a handful of large prefill waves and decodes to
  completion with host KV extending capacity beyond the device pool.
- **SGLang 0.5.21 baseline**: the grouped-tuned stack reused (TP8/DP8
  attention/EP8, mem-fraction 0.90, in-process dispatch). Prefix-aware
  routing is irrelevant at ~0.1% cross-request overlap, and no arm of the
  grouped sweep's other dimensions is specific to prefix reuse, so the
  grouped sweep's best non-router configuration carries over.

### Kimi-K3, 4×8 H20, `lcb-fixed4096-v1` (512 requests)

- **BatchGen** v1.1.0, 4 nodes, launched per
  [`docs/deploy-kimi-k3-h20.md`](../docs/deploy-kimi-k3-h20.md) with DeepEP
  enabled (dedicated environment described in the deployment guide).
- **SGLang 0.5.22 baseline (3-arm tuned sweep, best arm reported).** The
  mamba/KDA state budget splits across DP groups, so the sweep covered both
  parallelism families: the diverse-profile best configuration
  (`--enable-dp-attention --dp-size 4`, mamba cache 320, mem-fraction 0.92),
  a DP4 arm with mamba cache 480 at mem-fraction 0.93, and a TP32 arm with
  mamba cache 160. **Best: TP32 + mamba cache 160** — on a long forced
  decode, keeping one large shared state pool beats splitting it four ways.
  Chat form reported.

### Kimi-K3, 4×8 H20, `k3-diverse-v1` (512 requests)

- **BatchGen** v1.1.0, 4 nodes, launched per
  [`docs/deploy-kimi-k3-h20.md`](../docs/deploy-kimi-k3-h20.md) with DeepEP
  enabled (dedicated environment described in the deployment guide).
  Reported as the mean of two independent server lifetimes.
- **SGLang 0.5.22 baseline (tuned sweep, best configuration reported).**
  Swept TP32 against DP-attention family configurations; the mamba state
  budget splits across DP groups, so `--max-mamba-cache-size` was scaled per
  arm: TP32 + mamba 160; dp4 + mamba 160/320/480 with
  `--enable-dp-attention --dp-size 4` and `--mem-fraction-static` 0.92/0.93.
  Best: dp4 + mamba-cache 320 at mem-fraction 0.92, reported as the pooled
  mean of four runs. Chat form reported (a token-exact completions-format twin
  measured ~15% slower on the same server).
- **vLLM 0.32.0 baseline (tuned sweep, best configuration reported).** Swept
  TP32 arms (an alternative MoE backend failed at weight creation; the
  working arms used the Marlin MoE path with expert parallelism) at
  GPU memory utilization 0.88 and 0.93; best is 0.93, reported as the mean
  of two runs.
- **DeepEP note (both Kimi-K3 rows).** The SGLang 0.5.22 arms run the model's
  MoE layers without DeepEP because that release has no DeepEP path for this
  model's MXFP4 experts. The first release that adds one (0.5.23) cannot
  bring the model up on 96 GiB H20: both of its MXFP4 weight-loading paths
  exceed device memory during weight processing. There is therefore no
  completed DeepEP baseline arm on this hardware; the sweep's best completed
  configuration is reported per the contract above.

Rows whose baseline has not been through a sweep on the same hardware and
workload are not published here.
