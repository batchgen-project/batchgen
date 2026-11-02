# Deploy GLM-5.2-FP8 on 8 H200 GPUs

BatchGen serves `zai-org/GLM-5.2-FP8` on one node with **8 H200 GPUs**. The
qualified topology uses tensor-parallel model weights and pure data-parallel
prefill: each active rank owns a complete prompt. The model's context window is
1,048,576 tokens.

This guide covers checkpoint download, conversion, installation, host setup,
server launch, request submission, and the qualified long-context workload.
For the complete client API, see the [Batch API Guide](batch-api-guide.md).

## Table of Contents

1. [Download the Checkpoint](#1-download-the-checkpoint)
2. [Convert to BatchGen Format](#2-convert-to-batchgen-format)
3. [Install BatchGen](#3-install-batchgen)
4. [Prepare the Host](#4-prepare-the-host)
5. [Start the Server](#5-start-the-server)
6. [Submit Jobs](#6-submit-jobs)
7. [Qualified Performance and Limits](#7-qualified-performance-and-limits)
8. [Startup Kernel Prewarm](#8-startup-kernel-prewarm)

---

## 1. Download the Checkpoint

Download the complete checkpoint to storage local to the H200 node:

```bash
pip install -U huggingface_hub

huggingface-cli download zai-org/GLM-5.2-FP8 \
    --local-dir /models/GLM-5.2-FP8
```

Verify that the directory contains `config.json`, tokenizer files, and all
`model-*.safetensors` shards. Keep the launch identifier exactly
`zai-org/GLM-5.2-FP8`; BatchGen uses the registered identifier to select the
GLM-5.2 configuration, tokenizer, parallel strategy, and dual-KV profiles.

## 2. Convert to BatchGen Format

Convert the Hugging Face shards into BatchGen's contiguous checkpoint format:

```bash
python -m batchgen.tools.convert_checkpoint \
    --input-dir /models/GLM-5.2-FP8
```

The default output is `/models/GLM-5.2-FP8/converted_ckpt`. Validate an
existing conversion without rewriting it:

```bash
python -m batchgen.tools.convert_checkpoint \
    --input-dir /models/GLM-5.2-FP8 \
    --validate-only
```

## 3. Install BatchGen

Use the first BatchGen release that includes this guide and its GLM-5.2 runtime
dependencies; this deployment path is not present in v1.0.11. The server
rejects a `batchgen_kernels` package older than its minimum required version
(0.6.0 for v1.1.0). Since 0.6.0 the GLM-5 DSA indexer kernels and the FP8
WGMMA MoE kernels are AOT extensions inside `batchgen_kernels`, and the startup
preflight requires them, so an environment built for an earlier release must
rebuild `batchgen_kernels` (re-run `install_deps.sh --all` from the new
checkout).
A full source installation is:

```bash
git clone https://github.com/batchgen-project/batchgen.git
cd batchgen
./scripts/install_deps.sh --all
```

The full Hopper installation includes the matching PyTorch build,
FlashAttention, FlashMLA, SGL DeepGEMM with its TVM FFI dependency,
`batchgen_kernels`, and BatchGen. See [Installation](INSTALL.md) and
[Manual Installation](manual-installation.md) for wheel and component-specific
flows.

## 4. Prepare the Host

The qualified configuration requires one otherwise-idle 8×H200 host with about
2 TiB of RAM. It stages about 703.7 GiB of FP8 weights in shared memory and
uses a large Host KV allocation in addition to worker-private memory.

`--fast-init` uses shared-memory transparent huge pages. Check the setting and
enable it before launch when necessary:

```bash
cat /sys/kernel/mm/transparent_hugepage/shmem_enabled
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/shmem_enabled
```

Stop any earlier BatchGen service through its normal lifecycle before starting
a new one. Confirm that no prior process owns GPU memory and remove only
shared-memory objects whose ownership belongs to that stopped service.

The `860` GiB Host KV and pool-32 settings below are the exact qualification
point for the two accumulated-token workloads in §7. They leave no measured
host-memory margin on the qualified 2 TiB machine. Use a smaller Host KV cache
or pool for ordinary workloads, and size production hosts with additional
headroom.

## 5. Start the Server

Replace the checkpoint and storage paths and choose an unused distributed port:

```bash
numactl --interleave=all python -m batchgen.launch_http_server \
    --model zai-org/GLM-5.2-FP8 \
    --cache-dir /models/GLM-5.2-FP8 \
    --converted-ckpt-dir /models/GLM-5.2-FP8/converted_ckpt \
    --listen-port 10900 \
    --world-size 8 \
    --nnodes 1 \
    --node-rank 0 \
    --dist-init-addr 127.0.0.1:29500 \
    --host-kv-cache-size 860 \
    --kv-dtype bfloat16 \
    --max-pool-size 32 \
    --gpu-memory-frac 0.56 \
    --fast-init \
    --persistent-phase-instances \
    --enable-cuda-graph \
    --storage-path /shared/batchgen/glm52 \
    --startup-timeout 3600 \
    --watchdog-timeout 10800
```

The launch flags match the qualified source campaign:

- `--persistent-phase-instances` keeps the prefill and decode instances alive
  across phase changes.
- `--enable-cuda-graph` captures the supported decode buckets. Graph context
  capacity is derived from the model and its primary and auxiliary KV page
  tables; there is no CUDA-graph context-length flag.
- `--cuda-graph-max-bucket-size`, when changed from its default, controls
  decode batch size per rank rather than prompt length. A batch above the
  largest captured bucket uses eager execution.
- `--kv-dtype bfloat16` is the qualified dual-KV format for this model.
- `--max-pool-size` is now a deprecated compatibility value: it is still
  accepted but no longer limits how many sequences the server holds. That
  capacity is derived from `--input-ids-pool-size-gb` and the model context.
  The pool-32 qualification in §7 was recorded with an earlier build in which
  the flag still set the pool size.

Wait for readiness before submitting work:

```bash
curl --fail http://127.0.0.1:10900/health
```

### Startup Triton prewarm

GLM-5 runs several Triton kernels (FP8 activation casts, fused norms, RoPE and
KV writes, the DSA indexer transforms and the router GEMM). Triton compiles one
binary per specialisation, so BatchGen compiles every specialisation the
serving path can reach before it is needed, in three phases that each run once
per process:

- `prefill`, when the prefill instance is configured;
- `decode`, when the decode instance is configured;
- `geometry`, right before decode CUDA-graph setup, for the kernels keyed on
  the KV page-table widths. It runs again if those widths change.

With `--persistent-phase-instances`, as above, the `prefill` and `decode`
phases run while the worker builds both instances, before it admits the first
request; the `geometry` phase runs when the first batch enters decode. The
ranks of the node split each phase's compiles over Triton's shared cache
directory (`TRITON_CACHE_DIR`, by default `~/.triton/cache`), meet at a
barrier, and then every rank loads the full set. The first start on an empty
cache therefore takes longer; with a warm cache the phases only load binaries.
Keep the cache on storage that outlives the container to avoid recompiling.

Each phase logs its compile and load counts, and the end of the `geometry`
phase arms a late-compile guard:

```text
[GLM5_TRITON_WARMUP] prefill rank=R phase1 compiled=N loaded=M entries=E failed=F wall=Xs
[GLM5_TRITON_WARMUP] rank=R prefill entries=E total wall=Xs
[GLM5_TRITON_WARMUP] rank=R geometry primary_width=... aux_width=... page_sizes=.../... ...
[GLM5_TRITON_WARMUP] rank=R late-compile guard armed: later compiles log at WARNING; batchgen_debug.glm5_triton_strict makes them raise
```

The `decode` phase also logs which kernels the configuration can reach without
prewarming them; those lines are at WARNING when such a kernel is reachable.
Any Triton compilation after the guard is armed is logged at WARNING with the
kernel and its full specialisation key:

```text
[GLM5_TRITON_WARMUP] late compile kernel=<module.kernel> source=compile|load wall=Xs
    options=... constexprs={...} signature={...} attrs={...}
```

Decode batches larger than the top CUDA-graph bucket run eagerly, and some
eager-decode kernels are keyed on the active page-table width, which follows
the longest sequence; those keys are not prewarmed, so late-compile lines on
that path are expected. To make a late compile raise instead (acceptance runs
only, never in production), create the batch with
`"batchgen_debug": {"glm5_triton_strict": true}` (see
[Batch API](batch-api.md#post-v1batches)). The error is raised inside the
model forward, so it stops the worker pool, not only that batch, and it applies
to every sequence decoded in the same step.

## 6. Submit Jobs

GLM-5.2 uses the standard OpenAI-compatible batch API. First create an
`input.jsonl` file with one request per line:

```json
{"custom_id":"glm52-1","method":"POST","url":"/v1/chat/completions","body":{"model":"zai-org/GLM-5.2-FP8","messages":[{"role":"user","content":"Explain sparse attention."}],"max_completion_tokens":256,"temperature":0.0}}
```

Then upload the file, create the batch, wait for completion, and download its
output:

```python
from batchgen.batchgen_client import BatchGenHttpClient

client = BatchGenHttpClient("http://127.0.0.1:10900")
batch = client.submit_batch(
    input_file_path="input.jsonl",
    output_file_path="output.jsonl",
    max_context_length=1_048_576,
)
print(batch["status"])
```

See the [Batch API Guide](batch-api-guide.md) for polling, cancellation, curl,
and larger JSONL workloads.

## 7. Qualified Performance and Limits

Source assembly `954b2bc6` completed both required pure-DP8 prefill shapes on
one pool-32 service:

- **8×1M:** eight prompts of 1,048,575 tokens, one prompt per rank, followed by
  one generated token; 8,388,600 accumulated prompt tokens.
- **32×256K:** 32 prompts of 262,144 tokens, four prompts per rank; 8,388,608
  accumulated prompt tokens.

For the `8×1M` workload, the recorded server-side prefill result was:

| System | Configuration | Prefill wall time | Aggregate prompt throughput |
| --- | --- | ---: | ---: |
| BatchGen source RC | DP8, pool 32, 860 GiB Host KV | 406.27 s | 20,648 prompt tok/s |
| SGLang 0.5.18, best completed point | DP8 attention, TP8 MoE, NSA, CUDA graphs disabled, 72 GiB CPU offload/rank, 9,216-token/rank prefill chunk | 692.95 s | 12,106 prompt tok/s |

On this exact accumulated-token workload, BatchGen is **1.71× faster**. The
SGLang point is the best completed point in a 4,096 / 8,192 / 9,216 / 16,384
tokens-per-rank chunk sweep; the completed points on both sides of 9,216 were
slower.

These are source-candidate measurements on 8 H200 GPUs, not final installed-
wheel release evidence. They cover prefill and the first sampled token rather
than sustained decode from eight simultaneous 1M-token contexts. Throughput is
workload-specific: prompt length, admitted concurrency, output length, Host KV,
and offload settings can change the result.

During the same-service qualification, host `MemAvailable` reached 0 KiB
without an OOM or worker loss. Treat `--host-kv-cache-size 860` with pool 32 as
a zero-margin test configuration. The final release gate repeats both shapes
from the exact installed wheels and records host-memory margin again.

Large admissions cross a deterministic per-rank threshold and tokenize one
rank at a time to bound host-memory peaks; smaller jobs retain parallel
tokenization. If a worker exits, inspect the first Python traceback from the
failing rank. When no Python exception exists, check the host kernel log for an
OOM kill before attributing later distributed-store or supervisor errors.

## 8. Startup Kernel Prewarm

Every worker compiles the Triton and DeepGEMM kernels the GLM-5 serving code
can select before it needs them, in three phases: at the first prefill
configuration, at the first decode configuration, and right before decode
CUDA-graph capture. The eight ranks of the node split each phase's compiles,
meet at a barrier, and then load the full set. Each phase logs:

```text
[GLM5_TRITON_WARMUP] rank=R <phase> entries=E total wall=Xs DeepGEMM cache kernels A -> B
```

DeepGEMM JIT-compiles one kernel per configuration it selects for a GEMM, and
selects the configuration from the GEMM's row count with a cost model. The
prewarm reproduces that selection
(`batchgen_kernels/deepgemm_sm90_configs.py`, written for DeepGEMM
0.1.5.post3; any other installed version logs a WARNING and may compile late)
and launches one representative row count per configuration: the attention,
indexer, dense-MLP and expert projections over every row count up to the
larger of the prefill micro-batch token cap and the longest prompt; the decode
graph's masked projections and, with `--enable-deepep`, the routed experts'
masked GEMMs over every batch up to the decode page table's slot capacity; the
indexer's paged scorer for every 32-row batch alignment up to that capacity;
and the packed sparse prefill's indexer scorer. The kernels persist in
`DG_JIT_CACHE_DIR` (default `~/.deep_gemm`), so a warm cache only loads them.
After the graph phase each worker records the cache's kernel count and
compares it at every later phase switch. A kernel compiled in between is
reported as

```text
[GLM5_DEEPGEMM_WARMUP] rank=R DeepGEMM compiled N kernels after the prewarm (cache ...: A -> B, seen at ...)
```

and raises instead when the batch sets
`"batchgen_debug": {"glm5_triton_strict": true}` (acceptance runs only). The
cache is shared by the node's ranks, so the count covers all of them. An eager
decode batch larger than the slot capacity can still compile the paged
scorer's schedule kernel for its batch alignment.
