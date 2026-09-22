# Deploy Kimi-K3 on 4 H20 Nodes

Kimi-K3 is a 2.8T-parameter Mixture-of-Experts model (93 layers, 896 MXFP4 experts,
top-16 routing). On H20 (which has less HBM per GPU than H200) BatchGen serves it across
**4 nodes × 8 H20 (32 GPUs, expert-parallel)** using the compact per-node
distributed-weight store. This guide covers only the 4-node launch; for the checkpoint
download, conversion, and install steps — which are identical — follow §1–§3 of the
[2×8 H200 guide](deploy-kimi-k3-h200.md). For request submission, see the
[Batch API Guide](batch-api-guide.md).

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Prepare the Distributed-Weight Store](#2-prepare-the-distributed-weight-store)
3. [Start the Server (4 × 8 H20)](#3-start-the-server-4--8-h20)
4. [Submit Jobs](#4-submit-jobs)

---

## 1. Prerequisites

Complete §1–§3 of the [Kimi-K3 on 2×8 H200 guide](deploy-kimi-k3-h200.md): download the
checkpoint, convert it to the BatchGen (Marlin) format, and install BatchGen (source or
Docker) on **all four** nodes from shared storage. On H20, multi-node distributed weights
run over RDMA, so the same **IB-enabled UCX** prerequisite applies — see the UCX note in
§6 of that guide (the `libucx-cu12` wheel is CUDA/TCP-only; build one with the InfiniBand
transport and set `UCX_IB_GID_INDEX` / `NCCL_IB_GID_INDEX` for a routed RoCE fabric).

## 2. Prepare the Distributed-Weight Store

The replicated parameter server does not fit H20's smaller host/GPU budget, so the 4-node
path uses the **compact per-node distributed-weight store** (`--distributed-weight-config`).
Each node maps only its own shard. Generate one `node{rank}.json` per node (ranks 0–3) on
shared storage; each declares its `store_path`, `store_bytes`, the peer node IPs, and the
RDMA rail devices. Stage each node's store on a local **tmpfs** (`/dev/shm`, remounted large
enough) so the store mmap is transparent-hugepage-backed and NIC memory-key registration
does not read over the network — the same reasoning as the H200 guide's tmpfs note.

On **all four** nodes, before launch, lower the kernel memory watermark (fleet images that
ship `vm.watermark_scale_factor=1000` reserve a large high-watermark that starves the
weight load):

```bash
echo 10 | sudo tee /proc/sys/vm/watermark_scale_factor
```

## 3. Start the Server (4 × 8 H20)

Launch on **each** node with its own `--node-rank` (0–3) and the **same**
`--dist-init-addr` (node 0's address). `--world-size 32 --nnodes 4` describes the topology;
BatchGen's planner derives the parallelism from the checkpoint config.

```bash
# Recommended allocator setting (this build reads PYTORCH_CUDA_ALLOC_CONF):
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Point NCCL/UCX at your fabric's RDMA rails and, on a routed RoCE fabric, the RoCE v2 GID:
export NCCL_IB_GID_INDEX=3 UCX_IB_GID_INDEX=3

python -m batchgen.launch_http_server \
    --model moonshotai/Kimi-K3 \
    --converted-ckpt-dir /shared/models/Kimi-K3/converted_ckpt \
    --distributed-weight-config /shared/dwconfig/node${RANK}.json \
    --listen-port 10900 \
    --world-size 32 --nnodes 4 --node-rank ${RANK} \
    --dist-init-addr node0-ip:12355 \
    --kv-dtype bf16 \
    --host-kv-cache-size 512 \
    --gpu-memory-frac 0.80 \
    --parse-thinking \
    --storage-path /shared/storage
```

H20-specific notes (the rest matches the H200 guide):

- **`--model` must be the registered id `moonshotai/Kimi-K3`, not a local path** — same
  requirement and failure mode as the H200 guide (both the config registry and the host-KV
  profile key on the id).
- **`--gpu-memory-frac` is lower than on H200.** H20's smaller HBM leaves less room for the
  resident MXFP4 experts, so start conservatively (shown: `0.80`) and raise only if the
  resident-expert repack and KV cache still fit; if a worker dies at
  `build_resident_ep_mxfp4_layers … .to(device)` with a CUDA OOM, lower it further.
- **Distributed weights are the practical path on H20** (the replicated store does not fit);
  spreading the model across 4 nodes (EP over 32 GPUs) is what makes the 2.8T model fit.
- `--host-kv-cache-size` is required and positive; K3 sizes its host paged-KV view from it.

See [Server Flags](server-flags.md) and [Troubleshooting](troubleshooting.md) for details.

## 4. Submit Jobs

Identical to the H200 guide: set `"model": "kimi-k3"` in each request body and use the
[Batch API Guide](batch-api-guide.md) to upload a JSONL file, create a batch, and download
results.
