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
rejects a `batchgen_kernels` package older than its minimum required version.
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

Wait for readiness before submitting work:

```bash
curl --fail http://127.0.0.1:10900/health
```

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
