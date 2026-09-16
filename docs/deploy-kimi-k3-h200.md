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

Kimi-K3's KDA (Kimi Delta Attention) needs `fla-core>=0.5.0` (flash-linear-attention, the `fla` package). It is listed in `requirements.txt`, so the standard install pulls it automatically — no separate step.

## 4. Start the Server (2 × 8 H200)

### Mount shared memory

BatchGen stages the **full converted checkpoint** in `/dev/shm` on each node — all 8 workers on a
node mmap their shards from this single copy. K3's staged size is **~1454 GiB**, larger than the
default `/dev/shm` (50% of RAM). On **both** nodes, before launch, enable transparent-hugepage
shmem first (or per-worker page tables cost ~26 GB) and remount `/dev/shm` to the launcher's
printed requirement **plus ~64 GiB** of headroom:

```bash
# 1. transparent hugepages for shmem (saves ~26 GB of page tables across 8 workers)
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/shmem_enabled

# 2. remount /dev/shm for the staged checkpoint (2 TiB-RAM H200 shown: 1453.66 GiB + 64)
sudo mount -o remount,size=1517G /dev/shm
df -h /dev/shm                              # confirm the new size (should read ~1.5T)
```

If the launcher aborts with `Shared memory size is not enough ... Required: <N> GiB`, remount to
`<N> + 64` GiB. Do **not** set it to full host RAM — leave headroom for the workers.

### Lower the kernel memory watermark (2 TB-RAM nodes)

Some fleet images ship `vm.watermark_scale_factor=1000`, which reserves a **~412 GiB** high
watermark the kernel keeps free. On a 2 TB node that reservation plus the ~1454 GiB replicated
staging leaves almost no headroom for the 8 workers' pinned/HBM-staging buffers, and the node
OOM-kills (or the container restarts, silently reverting `/dev/shm`) partway through weight load.
On **both** nodes, before launch, lower it:

```bash
cat /proc/sys/vm/watermark_scale_factor         # 1000 on affected images
echo 10 | sudo tee /proc/sys/vm/watermark_scale_factor
```

If the replicated ~1454 GiB still does not fit with headroom, use the compact per-node
distributed-weight store instead (§6) — it stages only ~838 GiB/node.

On **shared / multi-tenant hosts** (e.g. Kubernetes nodes that co-schedule other pods), the
replicated ~1454 GiB store can trip a **pod eviction / container restart** from host-level memory
pressure *even when the container's own cgroup limit is unlimited* — the ~838 GiB/node
distributed-weight store (§6) is the reliable path there.

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
    --gpu-memory-frac 0.85 \
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
    --gpu-memory-frac 0.85 \
    --parse-thinking \
    --storage-path /shared/storage
```

> **HBM headroom for K3 (`--gpu-memory-frac` + allocator).** K3 materializes its routed experts as
> resident MXFP4 Marlin shards in HBM during `configure_decoding`. At `--gpu-memory-frac 0.9` the KV
> cache claims enough HBM that this expert repack OOMs on H200 (141 GiB) — the worker dies at
> `build_resident_ep_mxfp4_layers → _repack_projection → .to(device)` and takes the whole distributed
> job down. Use **`--gpu-memory-frac 0.85`** and export **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**
> (this build reads the `PYTORCH_CUDA_ALLOC_CONF` name, not `PYTORCH_ALLOC_CONF`) so fragmented reserved
> HBM is reusable, so the resident-expert repack fits within the per-GPU budget alongside the KV cache.

BatchGen's planner selects the parallelism from the checkpoint config; you only pass the
16-GPU / 2-node topology. `--parse-thinking` extracts K3 reasoning blocks into the
`reasoning_content` field. See [Server Flags](server-flags.md) for the full flag reference and
[Troubleshooting](troubleshooting.md) for startup issues.

> **Pass `--model` as the registered id `moonshotai/Kimi-K3`, not a local checkpoint path.** K3 *does*
> have a host-KV profile (`kimi_k3_mla`) and the model must resolve in **two** registries: the config
> registry (`_detect_model_type_from_identifier`, which pattern-matches the identifier string) and the
> host-KV profile registry. Only `moonshotai/Kimi-K3` satisfies both. If you pass a filesystem path,
> startup fails — either `Model '<path>' is not registered in BatchGen CONFIG_REGISTRY` or, later, a
> **fatal worker crash** `Unsupported model '<path>' for host KV cache` (this is *not* a benign
> warning; it kills every rank). To serve from a local checkpoint offline, symlink the checkpoint dir
> to the id and launch from the parent (`mkdir -p moonshotai && ln -s /local/Kimi-K3 moonshotai/Kimi-K3`,
> then `--model moonshotai/Kimi-K3` with that dir as the working directory) so `--model` is both a
> valid directory (config loads offline) and the registered id.
>
> `--host-kv-cache-size` is required and positive (e.g. `512`); K3 builds a host paged-KV view sized
> from it. `--parse-thinking` extracts K3 reasoning into `reasoning_content`.

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
  host-weight store instead of the replicated parameter server. Each node stages only its own
  ~838 GiB shard (routed experts + replicated bytes) rather than the full ~1454 GiB, so this is the
  practical path on **2 TB-RAM nodes** where the replicated store does not fit with headroom. Pass
  the config file for the matching node rank.

  > **Requires a UCX with the InfiniBand transport.** The daemon exchanges shards between nodes over
  > UCX RDMA (`rc_x`). The `libucx-cu12` pip wheel provisioned by `install_deps.sh` is a
  > **CUDA/TCP/SM-only** UCX build — it has **no `libuct_ib.so`**, so the daemon fails at startup
  > with `distributed weights daemon failed: owner bootstrap failed: ucp_ep_create: Destination is
  > unreachable`.
  >
  > Build an IB-enabled UCX and **replace the wheel's core libraries** — dropping only a
  > `libuct_ib.so` next to the wheel's `libuct.so` is **not** enough, because that `libuct.so` was
  > built with `UCT_MODULES="cuda cma"` and never dlopens the IB module:
  >
  > ```bash
  > ./configure --prefix=<ucx-ib> --with-verbs --with-cuda=$CUDA_HOME \
  >     --without-rdmacm --enable-mt --disable-logging && make -j && make install
  > # replace libucp / libuct / libucs / libucm (+ the ucx/ module dir) in the libucx wheel's lib/
  > ```
  >
  > Verify `rc_mlx5` is present: `ucx_info -d | grep -i rc_mlx5` (or `ls <libucx>/lib/ucx/ | grep
  > libuct_ib`). On a **routed RoCE v2** fabric — the two nodes' `bond0` on different /30 subnets —
  > also export `UCX_IB_GID_INDEX=3` and `NCCL_IB_GID_INDEX=3` (gid[3] is the RoCE v2 IPv4 GID on
  > every rail); without it the daemon hangs with `timed out waiting for distributed weights
  > network`. The replicated parameter server (the default in §4) does not use UCX and has no such
  > requirement.

  > **For repeatable fast bring-up, stage the store on tmpfs.** The daemon mmaps the store and
  > registers it with the NIC (`ibv_reg_mr` → `mlx5_core_create_mkey`). A store on a **shared /
  > network FS** stays on 4 KB pages — `--fast-init`'s hugepage path (`MADV_HUGEPAGE`) covers
  > anonymous / tmpfs (shmem) mappings but **not** network-FS file mmaps — so ~838 GiB / 4 KB ≈ 200M
  > page-table entries per memory region × (daemon + 8 workers) makes NIC key registration, not disk,
  > dominate startup. Copy each node's store to a large-enough **tmpfs** and point `store_path` at it:
  >
  > ```bash
  > sudo mount -o remount,size=2048G /dev/shm
  > cp /shared/.../node${RANK}_store.bin /dev/shm/node${RANK}_store.bin   # then set store_path to it
  > ```
  >
  > This is the same reason the single-node path (store on local/tmpfs) loads fast. On 2 TB-RAM nodes
  > the ~838 GiB tmpfs store still leaves ~1.1 TB for the workers.
