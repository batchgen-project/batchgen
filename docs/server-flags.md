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
| `--hf-cache-dir` | None | HuggingFace cache directory (where `transformers` stores downloaded models) |
| `--pt-ckpt-dir` | None | Path to PyTorch checkpoint files (`.pt` format) |

**Usage Notes:**
- Use `--cache-dir` when you've downloaded the model to a specific location
- Use `--hf-cache-dir` to point to your HuggingFace cache (typically `~/.cache/huggingface`)
- Use `--pt-ckpt-dir` for custom PyTorch checkpoint directories

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

**Recommended `--host-kv-cache-size` calculation:**
```
host_kv_cache_size = 0.9 × available_host_memory - model_size
```

For DeepSeek-R1 (~700GB model) on a 1.5TB memory node:
```bash
--host-kv-cache-size 650  # (1500 * 0.9) - 700 ≈ 650 GB
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
| `--enable-hugetlbfs` | `false` | Enable hugeTLBFS for shared memory. Requires system configuration. |

**To enable hugeTLBFS:**
```bash
# Allocate huge pages (requires root)
echo 10000 | sudo tee /proc/sys/vm/nr_hugepages

# Mount hugetlbfs
sudo mkdir -p /mnt/hugepages
sudo mount -t hugetlbfs none /mnt/hugepages
```

---

## Inference Configuration

### Sequence Length Limits

| Flag | Default | Description |
|------|---------|-------------|
| `--max-input-len` | `1024` | Default maximum input sequence length (tokens) |
| `--max-output-len` | `128` | Default maximum output/generation length (tokens) |

These are defaults that can be overridden per-request via the API.

### Continuous Batching

Controls how BatchGen schedules sequences on GPU.

| Flag | Default | Description |
|------|---------|-------------|
| `--initial-gpu-page-buffer` | `32` | Pages to reserve when first loading sequence to GPU. Each page = 64 tokens. |
| `--extension-gpu-page-buffer` | `4` | Pages to add at page boundaries during decode |
| `--decision-frequency-pages` | `2` | How often to make scheduling decisions (in pages) |
| `--host-kv-watermark` | `70` | Percentage threshold for prioritizing prefill over decode |
| `--enable-decode-preemption` | `true` | Allow interrupting decode to prefill new sequences |
| `--no-decode-preemption` | - | Disable decode preemption |

**GPU Page Buffer Design:**

When a sequence is first loaded to GPU, it reserves `initial_gpu_page_buffer` pages (default 32 pages = 2048 tokens) beyond its current context. This reduces the frequency of load/unload operations.

At page boundaries during decode, `extension_gpu_page_buffer` pages are added. The `decision_frequency_pages` controls how often scheduling decisions are made.

**Constraint:** `extension_gpu_page_buffer >= decision_frequency_pages` (to prevent overflow)

### Prefill Optimization

| Flag | Default | Description |
|------|---------|-------------|
| `--enable-prepack` | `true` | Enable prepack optimization for efficient prefill batching |
| `--no-prepack` | - | Disable prepack optimization (not recommended) |

Prepack optimization packs multiple sequences into a single batch for efficient prefill. It's recommended to always keep this enabled.

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

The watchdog monitors worker processes and restarts them if they become unresponsive.

| Flag | Default | Description |
|------|---------|-------------|
| `--watchdog-timeout` | `180` | Seconds before declaring a worker stuck. Set to 0 to disable. |
| `--no-watchdog` | - | Disable watchdog (equivalent to `--watchdog-timeout 0`) |
| `--watchdog-heartbeat-interval` | None | Heartbeat interval when workers are idle (seconds) |
| `--watchdog-test-stuck-time` | `0` | Deliberately sleep during watchdog feed (testing only) |

**When to adjust watchdog timeout:**
- Increase for very long sequences or slow hardware
- Disable (`--no-watchdog`) during development/debugging

---

## Complete Examples

### Single Node (8 GPUs)

```bash
python -m batchgen.launch_http_server \
    --model deepseek-ai/DeepSeek-R1 \
    --cache-dir /models/DeepSeek-R1 \
    --host-kv-cache-size 400 \
    --max-input-len 8192 \
    --max-output-len 4096
```

### Two Nodes (16 GPUs)

```bash
# Node 0 (Master)
python -m batchgen.launch_http_server \
    --model deepseek-ai/DeepSeek-R1 \
    --cache-dir /shared/models/DeepSeek-R1 \
    --world-size 16 --nnodes 2 --node-rank 0 \
    --dist-init-addr 192.168.1.100:12355 \
    --host-kv-cache-size 650 \
    --storage-path /shared/storage \
    --max-input-len 16000 \
    --max-output-len 8000

# Node 1
python -m batchgen.launch_http_server \
    --model deepseek-ai/DeepSeek-R1 \
    --cache-dir /shared/models/DeepSeek-R1 \
    --world-size 16 --nnodes 2 --node-rank 1 \
    --dist-init-addr 192.168.1.100:12355 \
    --host-kv-cache-size 650 \
    --storage-path /shared/storage \
    --max-input-len 16000 \
    --max-output-len 8000
```

### Development Mode

```bash
python -m batchgen.launch_http_server \
    --model deepseek-ai/DeepSeek-R1 \
    --cache-dir /models/DeepSeek-R1 \
    --no-watchdog \
    --save-result \
    --max-input-len 512 \
    --max-output-len 128
```

### Maximum Performance

```bash
python -m batchgen.launch_http_server \
    --model deepseek-ai/DeepSeek-R1 \
    --cache-dir /models/DeepSeek-R1 \
    --host-kv-cache-size 650 \
    --gpu-memory-frac 0.95 \
    --enable-hugetlbfs \
    --initial-gpu-page-buffer 64 \
    --extension-gpu-page-buffer 8 \
    --max-input-len 16000 \
    --max-output-len 8000
```

---

## Environment Variables

Some settings can also be configured via environment variables (deprecated, prefer CLI flags):

| Environment Variable | CLI Flag |
|---------------------|----------|
| `BATCHGEN_ENABLE_PREPACK` | `--enable-prepack` / `--no-prepack` |
| `BATCHGEN_HOST_KV_WATERMARK` | `--host-kv-watermark` |
| `BATCHGEN_ENABLE_DECODE_PREEMPTION` | `--enable-decode-preemption` |
| `BATCHGEN_GPU_KV_CACHE_SIZE_GB` | `--gpu-memory-frac` |

---

## See Also

- [Deployment Guide](deploy-deepseek-r1-h20.md) - Step-by-step multi-node deployment
- [README](../README.md) - Installation and quick start
