# Deploy Kimi-K3 on 2 H200 Nodes

Kimi-K3 is a 2.8T-parameter Mixture-of-Experts model (93 layers, 896 MXFP4 experts,
top-16 routing). BatchGen serves it across **2 nodes × 8 H200 (16 GPUs)**. This guide
covers checkpoint download, conversion, install, and the multi-node launch. For request
submission and the client API, see the [Batch API Guide](batch-api-guide.md).

## Table of Contents

1. [Download the Checkpoint](#1-download-the-checkpoint)
2. [Convert to BatchGen Format](#2-convert-to-batchgen-format)
3. [Install BatchGen](#3-install-batchgen)
4. [Start the Server (2 × 8 H200)](#4-start-the-server-2--8-h200)
5. [Submit Jobs](#5-submit-jobs)
6. [Optional: DeepEP and Distributed Weights](#6-optional-deepep-and-distributed-weights)

---

## 1. Download the Checkpoint

```bash
pip install -U huggingface_hub

# Download to shared storage reachable from both nodes
huggingface-cli download moonshotai/Kimi-K3 --local-dir /shared/models/Kimi-K3
```

Verify the download contains `config.json`, the tokenizer files, and the
`model-*.safetensors` shards.

## 2. Convert to BatchGen Format

Kimi-K3's routed experts are MXFP4. The converter repacks them into the Marlin tile layout
the decode/prefill kernels expect. Run it once to a shared path:

```bash
python -m batchgen.tools.convert_checkpoint --input-dir /shared/models/Kimi-K3
# -> /shared/models/Kimi-K3/converted_ckpt/   (pass this to --converted-ckpt-dir)
```

Validate an existing conversion without redoing it:

```bash
python -m batchgen.tools.convert_checkpoint --input-dir /shared/models/Kimi-K3 --validate-only
```

## 3. Install BatchGen

Build the Docker image or install from source — see [`docker/README.md`](../docker/README.md)
and [Manual Installation](manual-installation.md). The default decode/prefill exchange is
**NCCL**, which needs no extra build. (An optional DeepEP low-latency exchange is covered in
§6.)

## 4. Start the Server (2 × 8 H200)

### Mount shared memory

BatchGen uses `/dev/shm` for the host KV cache. Mount it to your host memory size on **both**
nodes before launch:

```bash
df -h /dev/shm
sudo mount -o remount,size=1500G /dev/shm   # replace 1500G with your host memory
```

### NCCL environment (optional tuning)

```bash
export NCCL_BUFFSIZE=16777216   # 16 MB
```

### Node 0 (master)

```bash
python -m batchgen.launch_http_server \
    --model moonshotai/Kimi-K3 \
    --converted-ckpt-dir /shared/models/Kimi-K3/converted_ckpt \
    --listen-port 10900 \
    --world-size 16 --nnodes 2 --node-rank 0 \
    --dist-init-addr node0-ip:12355 \
    --kv-dtype bf16 \
    --host-kv-cache-size 512 \
    --gpu-memory-frac 0.9 \
    --parse-thinking \
    --storage-path /shared/storage
```

### Node 1

Identical, with `--node-rank 1` and the **same** `--dist-init-addr node0-ip:12355`:

```bash
python -m batchgen.launch_http_server \
    --model moonshotai/Kimi-K3 \
    --converted-ckpt-dir /shared/models/Kimi-K3/converted_ckpt \
    --listen-port 10900 \
    --world-size 16 --nnodes 2 --node-rank 1 \
    --dist-init-addr node0-ip:12355 \
    --kv-dtype bf16 \
    --host-kv-cache-size 512 \
    --gpu-memory-frac 0.9 \
    --parse-thinking \
    --storage-path /shared/storage
```

BatchGen's planner selects the parallelism from the checkpoint config; you only pass the
16-GPU / 2-node topology. `--parse-thinking` extracts K3 reasoning blocks into the
`reasoning_content` field. See [Server Flags](server-flags.md) for the full flag reference and
[Troubleshooting](troubleshooting.md) for startup issues.

## 5. Submit Jobs

Kimi-K3 uses the standard OpenAI-compatible batch API. Set `"model": "kimi-k3"` in each
request body and follow the [Batch API Guide](batch-api-guide.md) to upload a JSONL file,
create a batch, and download results.

## 6. Optional: DeepEP and Distributed Weights

Both are opt-in; the defaults above run without either.

- **`--enable-deepep`** — use the DeepEP low-latency expert-parallel exchange for the decode
  graph instead of the default NCCL all-gather + reduce-scatter. It requires a DeepEP build
  patched for K3 (top-16 routing; upstream DeepEP asserts `topk <= 11`). If the build is not
  importable the server **fails fast at startup** (no silent NCCL fallback) — omit the flag to
  stay on NCCL.
- **`--distributed-weight-config <node0.json | node1.json>`** — map a compact per-node
  host-weight store instead of the replicated parameter server. Useful to avoid
  re-materializing weights after a node reset. Pass the config file for the matching node
  rank.
