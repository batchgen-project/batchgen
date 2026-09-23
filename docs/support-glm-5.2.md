# GLM-5.2-FP8 on H200

BatchGen supports `zai-org/GLM-5.2-FP8` on one node with 8 H200 GPUs. The
model uses tensor-parallel weights and pure data-parallel prefill: each active
rank owns a complete prompt. The qualified context window is 1,048,576 tokens.

This path differs from GLM-5 in three places that affect deployment:

- GLM-5.2 has a 1M-token context window and an 8e6 RoPE theta.
- Its DSA indexer recomputes top-k indices on 21 layers and reuses them on 57
  shared layers.
- Very long prefill uses bounded query and dense-MLP workspaces. Large text
  admission is serialized across ranks to prevent tokenizer peaks from
  exhausting host memory.

## Requirements

- One Linux host with 8 H200 GPUs and enough host RAM for the FP8 checkpoint,
  Host KV cache, worker-private memory, and `/dev/shm`.
- A complete `zai-org/GLM-5.2-FP8` checkpoint plus its BatchGen
  `converted_ckpt/` directory.
- BatchGen and `batchgen_kernels` wheels from the same release. The server
  rejects an incompatible kernel package.
- NUMA interleaving for the server process.
- Transparent Huge Pages for shmem when `--fast-init` is enabled. The server
  checks this at startup and fails with the required setting if it is absent.

Before launching, make sure no earlier server still owns GPU memory or visible
shared-memory files. Host KV is allocated in shared memory, so a stale service
can consume most of the machine even after its HTTP endpoint disappears.

## Launch

The following is the qualified one-node topology. Replace the checkpoint and
storage paths, and choose an unused distributed port.

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
    --kv-dtype bf16 \
    --max-pool-size 32 \
    --gpu-memory-frac 0.56 \
    --fast-init \
    --persistent-phase-instances \
    --enable-cuda-graph \
    --storage-path /shared/batchgen/glm52
```

`--persistent-phase-instances` retains the prefill and decode instances across
phase changes. `--enable-cuda-graph` captures the supported decode buckets.
CUDA-graph context capacity is derived from the model and its primary and
auxiliary KV page tables; there is no context-length flag for graph capture.
`--cuda-graph-max-bucket-size` controls decode batch size per rank, not prompt
length. Decode batches above the largest captured bucket use the eager path.

The `860` GiB Host KV and pool-32 values above were qualified on a 2 TiB host.
They are one matched configuration, not universal defaults. Lower
`--host-kv-cache-size` or `--max-pool-size` when the host has less free memory,
and leave headroom for tokenization and worker-private allocations. A server
that cannot meet its memory plan must fail before accepting work.

Check readiness before submitting a batch:

```bash
curl --fail http://127.0.0.1:10900/health
```

See the [Batch API Guide](batch-api-guide.md) for file upload, batch creation,
and result retrieval. Set `max_context_length` to `1048576` for requests that
may reach the full GLM-5.2 window.

## Qualified long-prefill shapes

On the one-node DP8 topology, the release-candidate source stack completed both
of these accumulated-token loads:

- 32 prompts × 262,144 tokens: four prompts per rank, 1,048,576 prompt tokens
  per rank.
- 8 prompts × 1,048,575 tokens plus one generated token: one full-window
  sequence per rank.

An earlier source candidate completed the shapes in the order above. A newer
source candidate with metadata-sized weight shared memory and bounded grouped
MoE router/shared-expert workspaces completed them in the reverse order on one
pool-32 service. Both runs returned valid, nonempty rows and kept all eight
workers alive. Final release qualification repeats both shapes from the exact
installed wheels.

This qualification covers prefill and the first sampled token. It does not
claim sustained decode performance from eight simultaneous 1M-token contexts.
Throughput depends on prompt length and admitted concurrency; a fixed request
count is not a general performance profile.

## Memory limits

One same-service long-prefill qualification reached a minimum host
`MemAvailable` of about 1.02 GiB with `--host-kv-cache-size 860` and
`--max-pool-size 32`; a newer source qualification on another 2 TiB host
reached 0 KiB without an OOM or worker loss. Treat that combination as a
zero-margin qualification point. Run it only on a dedicated, otherwise idle
host with at least the same memory capacity. Prefer additional host-memory
headroom in production. Reducing Host KV or admitted concurrency also reduces
the maximum simultaneous full-window workload.

Large admissions cross a deterministic per-rank text threshold and tokenize
one rank at a time. Smaller jobs retain parallel tokenization. If a worker is
lost, inspect the first Python traceback on the failing rank. When none exists,
check the host kernel log for an OOM kill before attributing later distributed
store or supervisor errors.
