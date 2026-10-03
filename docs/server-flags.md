# BatchGen Server Flags Reference

Complete reference for all `batchgen.launch_http_server` command-line flags.

## Quick Start

```bash
python -m batchgen.launch_http_server \
    --model deepseek-ai/DeepSeek-R1 \
    --cache-dir /path/to/model \
    --host-kv-cache-size 256
```

---

## Required Arguments

| Flag | Type | Description |
|------|------|-------------|
| `--model` | string | HuggingFace model name (e.g., `deepseek-ai/DeepSeek-R1`) |

---

## Network Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--instance-id` | `default` | Logical instance name used in this run's mutable resource namespace. This does not enable shared-host operation. |
| `--runtime-mode` | `exclusive` | Host admission mode. `shared` is accepted only for the qualified GPT-OSS topology with a valid inherited lane lease. |
| `--lane-lease-manifest-fd` | None | Inherited lease-manifest descriptor supplied by the lane launcher; required for `shared`. |
| `--listen-ip` | `0.0.0.0` | IP address the server listens on |
| `--listen-port` | `10900` | Port the server listens on |

---

## Model & Checkpoint Paths

| Flag | Default | Description |
|------|---------|-------------|
| `--cache-dir` | None | Path to downloaded model weights. Use this for pre-downloaded checkpoints. |
| `--converted-ckpt-dir` | None | Path to pre-converted checkpoint directory (skips conversion step on startup). |

**Usage Notes:**
- Use `--cache-dir` when you've downloaded the model to a specific location
- Use `--converted-ckpt-dir` to point to a checkpoint already converted to BatchGen format, skipping the conversion step

---

## Distributed Configuration

For multi-node deployments. See [Deployment Guide](deploy-deepseek-r1-h20.md) for examples.

| Flag | Default | Description |
|------|---------|-------------|
| `--world-size` | `1` | Total number of GPUs across all nodes |
| `--nnodes` | `1` | Number of nodes in the cluster |
| `--node-rank` | `0` | Rank of this node (0-indexed, master node is 0) |
| `--dist-init-addr` | `localhost:12355` | Address for torch.distributed initialization (`host:port`) |
| `--pynccl-port-base` | `20003` | First port in this runtime's bounded PyNccl range |
| `--pynccl-port-span` | `100` | Number of ports reserved for PyNccl initialization and recovery |

`--world-size` must divide evenly by `--nnodes`. The derived local world size
is passed explicitly to every worker; PyNccl initialization and recovery never
scan outside `[pynccl-port-base, pynccl-port-base + pynccl-port-span)`.

### Shared-host admission

`--runtime-mode shared` is fail-closed and currently accepts only
`openai/gpt-oss-120b`, one node, and world sizes 1, 2, 4, or 8. It rejects
fast-init, hugetlbfs, DeepEP, EP offloading, CUDA graph capture, distributed
host weights, default writable paths, and missing launcher leases.

The lane launcher passes `--lane-lease-manifest-fd`. The inherited JSON
manifest names the instance, ordered physical GPU UUIDs, communication ports,
and exact storage, conversion, temporary, and compiler-cache paths. Its
`resources` map contains a live inherited lock descriptor for every GPU, port
allocation, and writable-path hash. BatchGen verifies the manifest against its
arguments and environment, marks all lease descriptors close-on-exec, and
holds them until owner-scoped resource teardown completes. Operators should
not construct this internal descriptor bundle by hand; the versioned lane
launcher owns its schema and admission checks.

**Example: 2 Nodes x 8 GPUs**

```bash
# Node 0 (Master)
python -m batchgen.launch_http_server \
    --model deepseek-ai/DeepSeek-R1 \
    --world-size 16 --nnodes 2 --node-rank 0 \
    --host-kv-cache-size 650 \
    --dist-init-addr master-ip:12355

# Node 1
python -m batchgen.launch_http_server \
    --model deepseek-ai/DeepSeek-R1 \
    --world-size 16 --nnodes 2 --node-rank 1 \
    --host-kv-cache-size 650 \
    --dist-init-addr master-ip:12355
```

---

## Memory Configuration

### Host Memory (KV Cache)

| Flag | Default | Description |
|------|---------|-------------|
| `--host-kv-cache-size` | Required | Host KV cache size in GB. Critical for throughput. |
| `--kv-dtype` | `bfloat16` | Data type for KV cache (`bfloat16`, `float16`, `float8_e4m3fn`). Values are not validated at parse time — typos are accepted silently. |

**Sizing guideline:**
```
host_kv_cache_size ≈ host_mem × 0.9 - model_size
```

For DeepSeek-R1 (~700GB model) on a 1.5TB memory node:
```bash
--host-kv-cache-size 650  # (1500 * 0.9) - 700 ≈ 650 GB
```

The host KV cache, like the model weights, lives in anonymous shared memory
(`memfd_create`) that is charged to host memory. It is not a file in `/dev/shm`,
so the size of the `/dev/shm` mount does not limit it and no remount is needed.

### GPU Memory

| Flag | Default | Description |
|------|---------|-------------|
| `--gpu-memory-frac` | `0.9` | Fraction of GPU memory for KV cache (0.0-1.0) |
| `--gpu-arch` | Auto | GPU architecture hint (`hopper`, `ampere`). Auto-detected if not specified. |

**GPU KV cache size formula:**
```
gpu_kv_cache = GPU_memory × gpu_memory_frac - model_instance_size
```

### Shared Memory

| Flag | Default | Description |
|------|---------|-------------|
| `--enable-hugetlbfs` | `false` | Back the model weights with pages from the host's reserved huge page pool. Requires root privileges (sudo). |
| `--fast-init` | `false` | Request Transparent Huge Pages (THP) for the shared weights and host KV cache, with memory compaction and prefault, for faster memory registration. Recommended. |

**Lifetime in every mode:** the model weights, tensor metadata, host KV cache,
QueryBook table and weight skeleton are anonymous shared memory created with
`memfd_create`. Workers attach to them through `/proc/<pid>/fd/<n>`; nothing is
created by name in `/dev/shm` or a hugetlbfs mount. The kernel frees each region
when the last process using it exits, however the processes stop, and workers
are killed when their server process dies. After an abnormal exit, the only
BatchGen objects that can remain are the per-run runtime directory under the
temp directory and, if the whole process group was killed at once, Python's
`sem.mp-*` semaphores. The next
start of the same instance removes a previous run's runtime directory once its
run lock proves every process of that run has exited, and refuses to start while
one is still alive. The temp directory must be local to each node.

**Default page size:** without `--fast-init` or `--enable-hugetlbfs`, BatchGen
does not advise the shared regions, so their page size follows the host's
`/sys/kernel/mm/transparent_hugepage/shmem_enabled` policy.

**`--fast-init` details:**

Requests THP (2MB pages instead of 4KB) for the weights and host KV cache, which
makes `cudaHostRegister` substantially faster. Before allocation it runs Linux
memory compaction (`drop_caches` + `compact_memory`) to defragment physical
memory, and it prefaults the weights and the host KV cache. `--fast-init` is an explicit flag because
it needs the host setup below; the deployment recipes include it.

**Requirements:**
```bash
# Enable THP for shared memory (required, set once per boot); the server refuses
# --fast-init unless shmem_enabled is "always" or "within_size"
echo always > /sys/kernel/mm/transparent_hugepage/shmem_enabled

# Root access (for memory compaction; runs inside docker as root)
```

**`--enable-hugetlbfs` details:** the server sets `vm.nr_hugepages` to cover the
model and allocates the weights with `memfd_create(MFD_HUGETLB)`; no hugetlbfs
mount is needed. If the pool cannot satisfy the request, it falls back to a
regular memfd with a warning. The `vm.nr_hugepages` reservation is host
configuration: BatchGen does not reset it at shutdown. Setting it, like the
`--fast-init` compaction, needs a writable `/proc/sys`; in a container that
usually means `--privileged`.

**Priority:** When both `--enable-hugetlbfs` and `--fast-init` are set, hugetlbfs takes priority for weights (explicit huge pages > THP). The host KV cache never uses hugetlbfs.

---

## Inference Configuration

### Dynamic Sequence Management

Controls how BatchGen schedules sequences on GPU.

| Flag | Default | Description |
|------|---------|-------------|
| `--initial-gpu-page-buffer` | `32` | Pages to reserve when first loading sequence to GPU. Each page = 64 tokens. |
| `--extension-gpu-page-buffer` | `4` | Pages to add at page boundaries during decode |
| `--decision-frequency-pages` | `2` | How often to make scheduling decisions (in pages). Must be <= `--extension-gpu-page-buffer`, otherwise the server fails at startup with a `ValueError`. |
| `--host-kv-watermark` | `70` | Percentage threshold for prioritizing prefill over decode |
| `--enable-decode-preemption` | `true` | Allow interrupting decode to prefill new sequences (always on) |

**GPU Page Buffer Design:**

When a sequence is first loaded to GPU, it reserves `initial_gpu_page_buffer` pages (default 32 pages = 2048 tokens) beyond its current context. This reduces the frequency of load/unload operations.

At page boundaries during decode, `extension_gpu_page_buffer` pages are added. The `decision_frequency_pages` controls how often scheduling decisions are made.

**Constraint:** `extension_gpu_page_buffer >= decision_frequency_pages` (to prevent overflow)

### Host KV Scheduling

Controls how host KV cache pages are allocated and reclaimed during inference. By default, each sequence reserves its full KV token budget at prefill time. With dynamic reservation, sequences start with a small chunk and grow incrementally, enabling host KV oversubscription.

| Flag | Default | Description |
|------|---------|-------------|
| `--host-kv-chunk-size` | `8192` | Initial host-KV reservation chunk in tokens. Each sequence initially reserves its prompt plus the effective chunk (subject to the GPU initial-page buffer), instead of its full per-request decode budget. The effective chunk is capped by the worker's first pool-init decode length and rounded to 64-token pages. This flag does not override an explicit per-request `max_completion_tokens`; a later request with a larger budget grows its host reservation as needed. |
| `--enable-host-kv-eviction` | _(ignored)_ | **[Deprecated]** Host KV eviction is now always enabled when chunked reservation is active. This flag is ignored. Evicted sequences are automatically re-prefilled (recomputed) when pages become available. |
| `--host-kv-eviction-watermark` | `10` | Trigger eviction when free pages drop below this percentage (0-100). |
| `--adaptive-chunk` | `true` | Enable EMA-based adaptive chunk sizing. Tracks completed sequence decode lengths and adjusts the chunk size to reduce waste. |
| `--no-adaptive-chunk` | - | Disable adaptive chunk sizing (use static `--host-kv-chunk-size`). |
| `--adaptive-chunk-min` | `1024` | Minimum adaptive chunk size in tokens. |
| `--adaptive-chunk-max` | `65536` | Maximum adaptive chunk size in tokens. |
| `--adaptive-chunk-ema-alpha` | `0.1` | EMA smoothing factor (0-1]. Lower values make adaptation slower but more stable. |
| `--adaptive-chunk-multiplier` | `1.5` | Safety multiplier applied to the EMA estimate. Values > 1.0 reduce eviction frequency at the cost of higher memory usage. |

**How chunk-based reservation works:**

1. The first pool drain sends one worker `init` message. Its `max_output_len` is the maximum per-request output length in that first drain. The worker uses that value only to cap host-KV chunk sizing; it is not a server-wide output limit.
2. At prefill, each sequence reserves pages for its prompt plus the effective chunk, capped by that sequence's full KV budget (`prompt + max_completion_tokens` or the applicable fallback).
3. During decode, a sequence approaching its current host-KV capacity requests another chunk, capped by its own KV budget. Successful growth is not eviction and is not re-entry.
4. Eviction occurs when the host-KV growth planner determines that the active set cannot safely satisfy the required admission or growth while preserving the configured free-page watermark and safety margin. This can be proactive; it does not require a page allocation to fail first. The scheduler then releases a sequence's KV; that sequence becomes `EVICTED` and later re-enters through prefill/recompute before decoding again.
5. If adaptive chunk is enabled, the chunk size is adjusted based on observed decode lengths (EMA).

The first-init rule matters when later admissions use a larger `max_completion_tokens`: the per-request decode budget remains authoritative, but the initial host-KV reservation may be smaller and grow during decode.

**Example: High oversubscription with eviction**

```bash
python -m batchgen.launch_http_server \
    --model deepseek-ai/DeepSeek-R1 \
    --host-kv-cache-size 256 \
    --host-kv-chunk-size 512 \
    --host-kv-eviction-watermark 10 \
    --adaptive-chunk
```

This allows serving more concurrent sequences than the host KV cache can hold at full decode length. Sequences that exhaust their chunk grow incrementally, and if memory runs out, the least-progressed sequences are automatically evicted and recomputed later. Eviction is always enabled — no flag needed.

### Prefill Optimization

| Flag | Default | Description |
|------|---------|-------------|
| `--enable-prepack` | `true` | Enable prepack optimization for efficient prefill batching (always on) |

Prepack optimization packs multiple sequences into a single batch for efficient prefill. This is always enabled.

### Expert Parallelism with Offloading

For single-node deployments where GPU memory is limited, enable partial expert offloading to run large MoE models with high throughput.

| Flag | Default | Description |
|------|---------|-------------|
| `--enable-ep-with-offloading` | `false` | Enable Expert Parallelism with partial expert offloading mode |
| `--ep-offloading-ratio` | `0.0` | Ratio of experts to offload (0.0-1.0). Higher values save GPU memory but reduce throughput |

**Example: Single Node with 8 H20 GPUs**

```bash
python -m batchgen.launch_http_server \
    --model deepseek-ai/DeepSeek-R1 \
    --cache-dir /shared/models/DeepSeek-R1 \
    --kv-dtype "bf16" \
    --world-size 8 \
    --host-kv-cache-size 128 \
    --enable-hugetlbfs \
    --gpu-memory-frac 0.96 \
    --enable-ep-with-offloading \
    --ep-offloading-ratio 0.3
```

**How offloading works:**
- DeepSeek-R1 has 256 experts per MoE layer
- With 8 GPUs, each GPU handles 32 experts (256 / 8)
- `--ep-offloading-ratio 0.3` keeps 70% of experts persistent on GPU (22 experts per GPU)
- The remaining 30% (10 experts) are loaded synchronously from host memory, overlapped with computation as much as possible

**Constraints:**
- Requires `--enable-ep-with-offloading` to use `--ep-offloading-ratio > 0`
- Offloading ratio must be between 0.0 and 1.0
- Not needed if GPU memory is sufficient (e.g., two-node H20 deployment)

### CUDA Graph Acceleration

CUDA graphs capture the GPU kernel launch sequence and replay it with minimal CPU overhead. Enabled by default for supported models during the decode phase.

| Flag | Default | Description |
|------|---------|-------------|
| `--enable-cuda-graph` | `false` | Explicitly enable CUDA graph capture for decode (alias: `--enable-cuda-graphs`). Mutually exclusive with `--disable-cuda-graphs`. |
| `--disable-cuda-graphs` | `false` | Disable CUDA graph capture for decode. Use if encountering compatibility issues. Mutually exclusive with `--enable-cuda-graph`. |
| `--cuda-graph-max-bucket-size` | `128` | Maximum batch size per rank for CUDA graph capture. Batches exceeding this fall back to eager execution. |
| `--cuda-graph-num-buckets` | `16` | Number of CUDA graph bucket sizes. More buckets = longer startup capture time but less padding waste. |
| `--persistent-phase-instances` | `false` | Keep supported prefill and decode model instances alive across phase switches. For GLM-5/5.1/5.2 this also retains decode graphs and swaps only the routed experts needed by the active phase. |

**How it works:**
- At startup, CUDA graphs are captured at multiple discrete batch sizes (buckets) from 1 to `--cuda-graph-max-bucket-size`
- During decode, the actual batch size is rounded up to the nearest bucket and the pre-captured graph is replayed
- If the batch size exceeds the max bucket on any rank, all ranks fall back to eager execution for that step
- Use `BATCHGEN_SEGMENTED_GRAPH=1` to switch from whole-model graph to per-segment graph mode

For GLM-5/5.1/5.2, graph context capacity is derived from the model and its
primary and auxiliary KV page tables. `--cuda-graph-max-bucket-size` limits
decode batch size per rank; it does not limit prompt or context length.

---

## Output Parsing

| Flag | Default | Description |
|------|---------|-------------|
| `--parse-thinking` | `false` | Extract thinking/reasoning blocks into `reasoning_content` field |
| `--parse-tool-call` | `false` | Extract tool call blocks into `tool_calls` array |

---

## Storage Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--storage-path` | `batchgen/storage/` | Directory for uploaded files, batches, and outputs |
| `--save-result` | `false` | Save direct inference results to `{storage_path}/outputs/` as JSONL |

The storage directory structure:
```
storage/
├── uploads/       # Uploaded batch input files
├── batches/       # Batch job metadata
├── outputs/       # Inference results (when --save-result is enabled)
└── incremental/   # Incremental JSONL results (crash-resilient, enabled by default)
```

### Incremental Result Saving

Completed sequences are written incrementally to a JSONL file on disk as they finish, with `fsync` after each write. This enables recovery from spot instance preemption or crashes — partial results are preserved on disk even if the server is killed mid-batch.

| Flag | Default | Description |
|------|---------|-------------|
| `--incremental-output-dir` | `{storage_path}/incremental/` | Directory for incremental JSONL output files. Each batch produces one file named `{batch_id}.jsonl`. |
| `--no-incremental-save` | `false` | Disable incremental saving entirely |

**Enabled by default.** No extra flags are needed. To customize the output directory:

```bash
python -m batchgen.launch_http_server \
    --model deepseek-ai/DeepSeek-R1 \
    --host-kv-cache-size 650 \
    --incremental-output-dir /data/incremental_results
```

To disable:

```bash
python -m batchgen.launch_http_server \
    --model deepseek-ai/DeepSeek-R1 \
    --host-kv-cache-size 650 \
    --no-incremental-save
```

Each line in the JSONL file is a complete `BatchResultItem` with `custom_id`, `response` (containing `status_code`, `body` with `choices` and `usage`). Lines are written in completion order (not input order).

---

## Watchdog Configuration

The watchdog monitors worker processes and reports health via the `/health` endpoint. See [Watchdog & Health Monitoring](watchdog.md) for full documentation.

| Flag | Default | Description |
|------|---------|-------------|
| `--watchdog-timeout` | Disabled | General per-step/micro-batch timeout in seconds. Recommended: 600 for production. |
| `--decode-step-timeout` | Disabled | Max seconds for a single decode iteration. Recommended: 300 for production. |
| `--startup-timeout` | Disabled | Max seconds from process launch to server ready. Recommended: 1800 for large models. |
| `--no-watchdog` | - | Disable watchdog (default behavior, kept for compatibility) |
| `--watchdog-heartbeat-interval` | None | Idle heartbeat interval in seconds when watchdog is enabled |
| `--watchdog-test-stuck-time` | `0.0` | Deliberately sleep during watchdog feed (testing only) |

**When to enable watchdog:**
- For production deployments: use `--watchdog-timeout 600 --decode-step-timeout 300 --startup-timeout 1800`
- Increase timeouts for very long sequences or slow hardware

---

## Additional Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--pre-dequantize-weights` | `false` | Pre-dequantize MoE routed expert MXFP4 weights to BF16 at load time (higher HBM usage, lower compute overhead). Other weights are unaffected. |
| `--enable-deepep` | `false` | Enable the DeepEP low-latency expert-parallel exchange for the decode graph (default off = NCCL all-gather + reduce-scatter). Requires the DeepEP build and **fails fast** at startup if it is unavailable (no silent NCCL fallback). Generic across EP models; on Kimi-K3 it also needs H200 TP8 and the K3 DeepEP build. |
| `--distributed-weight-config` | None | Path to a node-local distributed host-weight source config (JSON). When set, the server skips the replicated parameter server and workers map the compact per-node store it describes. |
| `--max-pool-size` | `10240` | Max QueryBook pool capacity for persistent request scheduling. Must be > 0; the server refuses to start otherwise (`0` used to select the removed non-pool mode). |
| `--max-intake-capacity` | `1000000` | Max total requests in the intake pool. Prevents OOM under high load. |
| `--detokenization-include-special-tokens` | `false` | Include special tokens in detokenized output (default: off, special tokens stripped). |

---

## Example: Two Nodes (16 GPUs)

```bash
# Node 0 (Master)
python -m batchgen.launch_http_server \
    --model deepseek-ai/DeepSeek-R1 \
    --cache-dir /shared/models/DeepSeek-R1 \
    --world-size 16 --nnodes 2 --node-rank 0 \
    --dist-init-addr 192.168.1.100:12355 \
    --host-kv-cache-size 650 \
    --storage-path /shared/storage

# Node 1
python -m batchgen.launch_http_server \
    --model deepseek-ai/DeepSeek-R1 \
    --cache-dir /shared/models/DeepSeek-R1 \
    --world-size 16 --nnodes 2 --node-rank 1 \
    --dist-init-addr 192.168.1.100:12355 \
    --host-kv-cache-size 650 \
    --storage-path /shared/storage
```

---

## See Also

- [Deployment Guide](deploy-deepseek-r1-h20.md) - Step-by-step multi-node deployment
- [Client API Reference](client-api.md) - Python client usage and parameters
- [README](../README.md) - Installation and quick start
