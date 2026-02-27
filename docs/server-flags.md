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
| `--listen-ip` | `0.0.0.0` | IP address the server listens on |
| `--listen-port` | `10900` | Port the server listens on |

---

## Model & Checkpoint Paths

| Flag | Default | Description |
|------|---------|-------------|
| `--cache-dir` | None | Path to downloaded model weights. Use this for pre-downloaded checkpoints. |

**Usage Notes:**
- Use `--cache-dir` when you've downloaded the model to a specific location

---

## Distributed Configuration

For multi-node deployments. See [Deployment Guide](deploy-deepseek-r1-h20.md) for examples.

| Flag | Default | Description |
|------|---------|-------------|
| `--world-size` | `1` | Total number of GPUs across all nodes |
| `--nnodes` | `1` | Number of nodes in the cluster |
| `--node-rank` | `0` | Rank of this node (0-indexed, master node is 0) |
| `--dist-init-addr` | `localhost:12355` | Address for torch.distributed initialization (`host:port`) |

**Example: 2 Nodes x 8 GPUs**

```bash
# Node 0 (Master)
python -m batchgen.launch_http_server \
    --model deepseek-ai/DeepSeek-R1 \
    --world-size 16 --nnodes 2 --node-rank 0 \
    --dist-init-addr master-ip:12355

# Node 1
python -m batchgen.launch_http_server \
    --model deepseek-ai/DeepSeek-R1 \
    --world-size 16 --nnodes 2 --node-rank 1 \
    --dist-init-addr master-ip:12355
```

---

## Memory Configuration

### Host Memory (KV Cache)

| Flag | Default | Description |
|------|---------|-------------|
| `--host-kv-cache-size` | Auto | Host KV cache size in GB. Critical for throughput. |
| `--kv-dtype` | `bfloat16` | Data type for KV cache (`bfloat16`, `float16`, `float8_e4m3fn`) |

**Auto-detection formula** (when `--host-kv-cache-size` is not specified):
```
host_kv_cache_size = min(host_mem × 0.9 - model_size, /dev/shm_free_space)
```

If `/dev/shm` free space is smaller than the calculated budget, a warning will be logged recommending to increase `/dev/shm` size.

For DeepSeek-R1 (~700GB model) on a 1.5TB memory node:
```bash
--host-kv-cache-size 650  # (1500 * 0.9) - 700 ≈ 650 GB
```

**Important: /dev/shm size requirement**

Host KV cache uses shared memory (`/dev/shm`). If the cache size exceeds available `/dev/shm` space, you must increase it first:

```bash
# Check current size
df -h /dev/shm

# Increase temporarily (replace 1500G with your host memory size)
sudo mount -o remount,size=1500G /dev/shm

# Or make permanent by adding to /etc/fstab:
# tmpfs /dev/shm tmpfs defaults,size=1500G 0 0
```

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
| `--enable-hugetlbfs` | `false` | Enable hugeTLBFS for shared memory. Requires root privileges (sudo). |

**Note:** When `--enable-hugetlbfs` is enabled, BatchGen will automatically configure huge pages. This requires running the server with root privileges (sudo).

---

## Inference Configuration

### Dynamic Sequence Management

Controls how BatchGen schedules sequences on GPU.

| Flag | Default | Description |
|------|---------|-------------|
| `--initial-gpu-page-buffer` | `32` | Pages to reserve when first loading sequence to GPU. Each page = 64 tokens. |
| `--extension-gpu-page-buffer` | `4` | Pages to add at page boundaries during decode |
| `--decision-frequency-pages` | `2` | How often to make scheduling decisions (in pages) |
| `--host-kv-watermark` | `70` | Percentage threshold for prioritizing prefill over decode |
| `--enable-decode-preemption` | `true` | Allow interrupting decode to prefill new sequences (always on) |

**GPU Page Buffer Design:**

When a sequence is first loaded to GPU, it reserves `initial_gpu_page_buffer` pages (default 32 pages = 2048 tokens) beyond its current context. This reduces the frequency of load/unload operations.

At page boundaries during decode, `extension_gpu_page_buffer` pages are added. The `decision_frequency_pages` controls how often scheduling decisions are made.

**Constraint:** `extension_gpu_page_buffer >= decision_frequency_pages` (to prevent overflow)

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
| `--disable-cuda-graphs` | `false` | Disable CUDA graph capture for decode. Use if encountering compatibility issues. |
| `--cuda-graph-max-bucket-size` | `128` | Maximum batch size per rank for CUDA graph capture. Batches exceeding this fall back to eager execution. |
| `--cuda-graph-num-buckets` | `16` | Number of CUDA graph bucket sizes. More buckets = longer startup capture time but less padding waste. |

**How it works:**
- At startup, CUDA graphs are captured at multiple discrete batch sizes (buckets) from 1 to `--cuda-graph-max-bucket-size`
- During decode, the actual batch size is rounded up to the nearest bucket and the pre-captured graph is replayed
- If the batch size exceeds the max bucket on any rank, all ranks fall back to eager execution for that step
- Use `BATCHGEN_SEGMENTED_GRAPH=1` to switch from whole-model graph to per-segment graph mode

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
├── uploads/      # Uploaded batch input files
├── batches/      # Batch job metadata
└── outputs/      # Inference results (when --save-result is enabled)
```

---

## Watchdog Configuration

The watchdog monitors worker processes and restarts them if they become unresponsive. It is fed after each prefill micro-batch and each decode step.

| Flag | Default | Description |
|------|---------|-------------|
| `--watchdog-timeout` | Disabled | Timeout in seconds per micro-batch/decode step. Recommended: 300 for production. |
| `--no-watchdog` | - | Disable watchdog (default behavior, kept for compatibility) |

**When to enable watchdog:**
- For production deployments: use `--watchdog-timeout 300` (5 minutes)
- Increase timeout for very long sequences or slow hardware

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
