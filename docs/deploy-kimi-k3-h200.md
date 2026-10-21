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

### Choose the weight source (required on multiple nodes)

Multi-node Kimi-K3 **defaults to the compact per-node distributed-weight store** (§6): launch
without `--distributed-weight-config` and the server aborts at argument validation, naming the
flag. The replicated parameter server the commands below use is still available, but it is now an
explicit `--replicated-weights` opt-in — add that flag to both nodes, and only where ~1.5 TiB of
host memory per node is genuinely free (never on a shared / multi-tenant host). Single-node launches
are unaffected.

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
    --replicated-weights \
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
    --replicated-weights \
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

DeepEP is opt-in. The distributed-weight store is the **default** multi-node weight source: the
commands in §4 reach the replicated parameter server only because they pass `--replicated-weights`.

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

  Build each node's store with the store builder, one `--output-dir` per node, on persistent
  storage (a local disk or the shared FS). The builder output is the **source**: the server never
  maps it directly and never deletes it.

  ```bash
  python -m batchgen.models.moonshotai.kimi_linear.store_builder \
      --converted-ckpt /shared/models/Kimi-K3/converted_ckpt \
      --cache-dir /shared/models/Kimi-K3 \
      --output-dir /shared/k3-stores/node${RANK} \
      --num-nodes 2 --node-rank ${RANK} \
      --node-ips node0-ip,node1-ip \
      --store-residency run           # or: resident
  ```

  The builder checks the converted checkpoint against the full Kimi-K3 tensor map from
  `config.json` (nothing missing, nothing unexpected) and every source span against its `.bin`
  before writing anything, hashes every tensor while it copies, re-reads the store to verify it,
  and only then publishes `store.bin`, `metadata.tsv` and `node${RANK}.json`, followed by the
  completion manifest `store_manifest.json`. An interrupted build leaves no manifest; `--dry-run`
  writes nothing.

  At launch, argument validation refuses a store whose completion manifest is missing or
  incomplete, whose files no longer match it (a cleared tmpfs, a partial copy, metadata of another
  build), that was built from a different converted checkpoint or `config.json` than the launch's
  `--converted-ckpt-dir` (or `--cache-dir`/converted_ckpt) and `--cache-dir`, that belongs to
  another node rank, or whose sampled bytes no longer hash to the manifest. A distributed-store
  launch must therefore name its converted checkpoint. The refusal ends with the exact builder
  command that rebuilds the store.

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

  **Staged copy and its lifetime.** At start the server copies `store.bin` (with its metadata and
  manifest) into a directory it owns under `store_staging_dir` (default
  `/dev/shm/batchgen-weight-stores`, which must be a tmpfs), verifying every manifest segment as
  it copies, and the daemon and every worker map that copy. Do **not** copy the store into
  `/dev/shm` by hand and do **not** delete staged copies by hand. `--store-residency` (written
  into `node${RANK}.json`) chooses how long the copy lives:

  - `run` (default) — the copy is removed when the server stops, like every other run resource.
    A copy left by a killed server is reused by the next start of the same build or removed by
    the next start of another build.
  - `resident` — the copy stays in the tmpfs after the server stops, so the next start of the
    same build maps it without copying the source again (a fast cold restart). It still uses
    tmpfs memory between runs.

  A worker-pool restart never touches the staged copy. Each start removes idle copies of other
  builds whatever their residency, so switching builds never strands one; a copy another server
  is still using is never removed. A config written before these keys existed keeps mapping
  `store_path` directly. The staging tmpfs needs room for the whole store; if it is too small the
  server refuses before copying and names the shortfall
  (`sudo mount -o remount,size=<bytes> /dev/shm`).

  > **`--fast-init` requires a huge-page tmpfs.** The daemon and the workers advise the staged
  > mapping with `MADV_HUGEPAGE`, but a tmpfs backs files with transparent huge pages only when it
  > is mounted with `huge=within_size` or `huge=always`. Without that the store stays on 4 KB pages
  > and NIC key registration (`ibv_reg_mr`) dominates startup, so under `--fast-init` the server
  > refuses to start and prints the remount command for the staging mount:
  >
  > ```bash
  > sudo mount -o remount,huge=within_size /dev/shm
  > ```
  >
  > Without `--fast-init` any tmpfs is accepted.
