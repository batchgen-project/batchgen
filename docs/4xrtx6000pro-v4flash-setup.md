# 4× RTX PRO 6000 Blackwell (Server Edition) — DeepSeek‑V4‑Flash Setup & Tuning

Operator guide for serving **DeepSeek‑V4‑Flash** (batchgen) on a 4‑GPU RTX PRO 6000
Blackwell **Server Edition** box (`gala2`). Captures the verified hardware profile, the
known‑good launch configuration, the measured baseline, and the system/serving
hyperparameters — split into **VERIFIED** (validated working) and **SWEEP** (recommended to
tune in the follow‑up optimization).

---

## 1. Hardware overview

| Component | Detail |
|---|---|
| GPUs | 6× RTX PRO 6000 Blackwell, **96 GB** (97887 MiB) each, **sm_120** (cc 12.0) |
| GPU 0–3 | **Server Edition** — use these for the 4‑GPU V4 workload |
| GPU 4–5 | Max‑Q Workstation Edition — power‑limited, **different NUMA node**, avoid mixing |
| Driver / CUDA | 590.48.01 / CUDA 13.x |
| CPU | 2× AMD EPYC 9355 32‑Core (64C / 128T), **2 NUMA nodes** |
| RAM | ~1.5 TB |
| NUMA node0 | CPUs `0-31,64-95` → **GPUs 0–3** |
| NUMA node1 | CPUs `32-63,96-127` → GPUs 4–5 |

### Interconnect topology (`nvidia-smi topo -m`) — **NO NVLink**

```
        GPU0  GPU1  GPU2  GPU3  GPU4  GPU5   NUMA
GPU0     X    PIX   NODE  NODE  SYS   SYS     0
GPU1    PIX    X    NODE  NODE  SYS   SYS     0
GPU2    NODE  NODE   X    PIX   SYS   SYS     0
GPU3    NODE  NODE  PIX    X    SYS   SYS     0
GPU4    SYS   SYS   SYS   SYS    X    PIX     1
GPU5    SYS   SYS   SYS   SYS   PIX    X      1
```

- `PIX` = single PCIe bridge (fastest): **GPU0↔GPU1** and **GPU2↔GPU3** are fast pairs.
- `NODE` = across PCIe host bridges within NUMA0: **GPU0/1 ↔ GPU2/3** is slower than PIX.
- `SYS` = cross‑NUMA (UPI): GPUs 0–3 ↔ 4–5 is slowest — **do not span an EP/TP group to 4–5**.
- PCIe **gen5 x16** (idle GPUs report gen1 due to power down‑clock; scales to gen5 under load).

---

## 2. The fundamental constraint: PCIe‑only interconnect

PCIe5 x16 ≈ **~64 GB/s/dir** vs NVLink ≈ **~900 GB/s** — so cross‑GPU collectives on this box
are **~10–14× slower** than on an NVLink server. The measured decode bottleneck (~1.1 tok/s) is
**EP collectives over PCIe + per‑layer serialization**, *not* the MoE kernel (which is fast and
verified). Implications:

- Favor parallelism layouts that **minimize cross‑GPU traffic per token**.
- The two fast pairs (0‑1, 2‑3) communicate cheaply; cross‑pair (NODE) and cross‑NUMA (SYS) are
  the expensive hops — keep collective‑heavy groups within a pair where possible.
- Throughput, not memory, is the limiter at decode; memory matters most at load/prefill.

---

## 3. System / OS hyperparameters

> **[SWEEP]** = validate/tune in the optimization loop. **[VERIFIED]** = confirmed working.

| Knob | Setting | Status | Why |
|---|---|---|---|
| NUMA pinning | `numactl --cpunodebind=0 --membind=0` for the 4‑GPU server | **[SWEEP]** | GPUs 0–3 are NUMA0; pinning CPU+memory to node0 avoids cross‑NUMA host traffic for the param server / dataloader. |
| GPU selection | `CUDA_VISIBLE_DEVICES=0,1,2,3` | **[VERIFIED]** | Server‑edition, all NUMA0; never mix Max‑Q 4–5 (SYS). |
| Container shm | `--shm-size=400g` | **[VERIFIED]** | Param server needs ~320 GB host shm; default container shm → "Shared memory size is not enough". |
| Allocator | `PYTORCH_ALLOC_CONF=expandable_segments:True` | **[VERIFIED]** | Reduces fragmentation OOM (esp. rank0). (Older name `PYTORCH_CUDA_ALLOC_CONF` is deprecated.) |
| IPC | `--ipc=host` | **[VERIFIED]** | Shared‑memory IPC for the param server. |
| NCCL P2P level | `NCCL_P2P_LEVEL` (try `PIX` / `NODE`) | **[SWEEP]** | No NVLink → controls when P2P is used over PCIe; PIX pairs benefit, cross‑pair may not. A/B it. |
| NCCL channels | `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` | **[SWEEP]** | Tune channel count for PCIe bandwidth; too many can thrash. |
| NCCL buffer | `NCCL_BUFFSIZE` | **[SWEEP]** | Larger buffers can help PCIe collective efficiency. |
| NCCL algo/proto | `NCCL_ALGO` (Ring/Tree), `NCCL_PROTO` | **[SWEEP]** | Ring vs Tree behaves differently without NVLink; measure. |
| NCCL SHM | `NCCL_SHM_DISABLE` (default 0) | **[SWEEP]** | SHM transport between same‑node GPUs; usually keep enabled. |
| memlock | `ulimit -l unlimited` (or `--ulimit memlock=-1`) | **[SWEEP]** | For pinned host memory registration of the 320 GB store. |
| Hugepages / fast‑init | `--fast-init` (memfd + THP) | **[SWEEP]** | Faster/stabler 320 GB registration if THP + root available. |

---

## 4. batchgen V4‑Flash launch reference **[VERIFIED working]**

- **Image:** `batchgen:v4flash-blackwell-src` (ships / JIT‑builds `core_engine` for sm_120).
- **Checkpoint (MP4 FP8, sharded for world_size=4):** `/home/leyang/v4flash_converted_mp4`
  (`model{0-3}-mp4.json/.bin`). The generic HF `converted_ckpt` does **not** work — the EP
  loader needs `model{rank}-mp{world}` sharding.

```bash
docker run -d --name v4flash --gpus '"device=0,1,2,3"' --ipc=host --shm-size=400g --network=host \
  -v /mnt/raid0nvme0/leyang/batchgen:/workspace/batchgen \
  -v /mnt/raid0nvme0/public/huggingface:/root/.cache/huggingface \
  -v /home/leyang/v4flash_converted_mp4:/ckpt_mp4 \
  -e HF_HUB_OFFLINE=1 -e BATCHGEN_V4_RESIDENT_EXPERTS=1 -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -w /workspace/batchgen batchgen:v4flash-blackwell-src bash -c 'sleep infinity'

# inside container (SNAP = the HF snapshot dir):
python -m batchgen.launch_http_server --model deepseek-ai/DeepSeek-V4-Flash \
  --converted-ckpt-dir /ckpt_mp4 --cache-dir "$SNAP" \
  --kv-dtype fp8 --host-kv-cache-size 60 --gpu-memory-frac 0.3 --gpu-arch blackwell \
  --dist-init-addr localhost:12457 --world-size 4 --listen-port 12345 --watchdog-timeout 1200
```

**Flag rationale**

| Flag | Value | Why |
|---|---|---|
| `--world-size 4` | 4 | EP across the 4 Server GPUs. |
| `--gpu-memory-frac` | 0.3 | Caps GPU‑KV (~24.6 GB). Default grabbed ~81.6 GB → prefill OOM. **[SWEEP]** |
| `--host-kv-cache-size` | 60 | Host‑side KV (GB). **[SWEEP]** |
| `--kv-dtype` | fp8 | Halves KV footprint. **[SWEEP: bf16 vs fp8]** |
| `--gpu-arch` | blackwell | sm_120 codepaths. **[VERIFIED]** |
| `BATCHGEN_V4_RESIDENT_EXPERTS` | 1 | Owned experts resident; threaded to workers via worker‑args (env alone did not propagate to subprocesses). **[VERIFIED]** |
| `--initial-gpu-page-buffer` / `--extension-gpu-page-buffer` | — | GPU page‑buffer sizing. **[SWEEP]** |

**MoE path:** decode uses the fast **grouped mega3** kernel (resident, single‑copy bundle);
**prefill uses eager** MoE (the 256‑expert grouped bundle does not fit rank0/GPU0). Boot ~170–300 s;
the **first request JIT‑compiles kernels (~340 s — not representative)**, so always warm up before timing.

---

## 5. Tuning levers to sweep (for the optimization loop)

Metric: **decode tokens/sec** (and TTFT/prefill, accuracy as guardrails). Given the PCIe constraint:

| Lever | Range to try | Expected direction |
|---|---|---|
| `--gpu-memory-frac` | 0.25 → 0.6 | More GPU‑KV ⇒ larger concurrent batch ⇒ higher aggregate tok/s, until prefill/activation OOM. |
| `--host-kv-cache-size` | 40 → 120 | More host KV ⇒ more concurrent seqs; watch host RAM + reg time. |
| `--kv-dtype` | fp8 vs bf16 | fp8 = more KV/throughput; bf16 = accuracy headroom. |
| world‑size / layout | EP4 vs TP within pairs vs 2×(pair) | PCIe pairs (PIX) are cheap; cross‑pair (NODE) is the cost — layouts that keep collectives within a pair should win. |
| NCCL env | `P2P_LEVEL`, `ALGO`, `NCHANNELS`, `BUFFSIZE` | Tune the PCIe collective path (the decode bottleneck). |
| NUMA pinning | `numactl` node0 vs none | Pinning should reduce host‑side jitter. |
| concurrency / batch | sweep request concurrency | Decode is collective‑bound per token; batching amortizes the fixed per‑step collective cost ⇒ aggregate tok/s should rise with concurrency. |
| page buffers | initial/extension | Affects KV growth + fragmentation. |

The single highest‑leverage idea given the data: **batch/concurrency + KV sizing** (amortize the
PCIe per‑token collective over more sequences) and **NCCL‑over‑PCIe tuning**. The per‑token MoE
kernel is already fast; the win is in amortizing/reducing the collective + dispatch overhead.

---

## 6. Measured baseline (the number the sweep must beat)

From `benchmarks/grouped_moe_probes/E2E_CORRECTNESS.md` (real 284B V4‑Flash, 4× sm_120):

| Metric | Value | Notes |
|---|---|---|
| Decode throughput | **~1.1 tok/s** (worker) / ~0.695 tok/s e2e | Collective‑bound over PCIe; single‑stream. |
| Prefill TTFT | 16.7 s @57tok · 17.9 s @505tok · 20.0 s @2003tok | Eager prefill; grows slowly with length. |
| Accuracy | **MMLU‑Pro 71%** (100‑prompt sample) | Coherent + correct; ~6% extraction failures. |
| Boot | ~170–300 s | First request +~340 s JIT (warm up first). |

> Note: the baseline is **single‑stream**. Aggregate throughput under concurrency is the more
> meaningful serving metric and is the primary sweep target.

---

## 7. Known constraints & gotchas

- **Disk:** `/mnt/raid0nvme0` is shared and frequently near‑full (14 TB raid). Use `/` (~1.1 TB) or
  `/dev/shm` (756 GB) for scratch; do **not** write large files to `/mnt`.
- **Container shutdown wedge:** the server container can wedge on `docker rm -f` (D‑state process
  holding GPU/shm). Cleanup: `pkill -9 -f launch_http_server` inside; `docker rm -f`; if a GPU
  stays occupied, `kill -9` the `nvidia-smi --query-compute-apps=pid` PID; then `docker rm -f`.
- **/dev/shm leak:** wedged servers leak **320 GB** `shm_*` + `batchgen_host_kv_cache` segments
  (root‑owned). Clear via:
  `docker run --rm -v /dev/shm:/hostshm batchgen:v4flash-blackwell-src bash -c 'rm -f /hostshm/shm_* /hostshm/batchgen_host_kv_cache'`.
  Always verify `nvidia-smi` → 0 MiB and `df -h /dev/shm` after a run.
- **rank0/GPU0 is the memory hotspot** (holds embed + lm_head + extra) — it OOMs first; budget for it.
- **Max‑Q GPUs 4–5** are on NUMA1 (SYS) and power‑limited — keep them out of the V4 group.

## 8. Recommended prefill settings **[VERIFIED 2026-07-03]**

Best-performing prefill with the least tokens in batch, from the result-count-validated batch-scale
study (`benchmarks/grouped_moe_probes/autoresearch_v4/README.md`, "VERIFIED large-batch prefill"):

| setting | value | why |
|---|---|---|
| in-flight prefill batch | **~128 sequences x 8192 tokens (~1.05M tokens)** | best verified aggregate **~2.3-2.4K tok/s** (5.6x single-request); throughput still rising at this size but the server silently DROPS sequences beyond it (see below) |
| sequence length | 8192 (RoPE fix required; cache floors at `original_seq_len` 65536) | longer seqs raise tokens/expert; 8192 verified end-to-end |
| experts | `BATCHGEN_V4_RESIDENT_EXPERTS=1` (default) | streamed-vs-resident ties within 4% at this batch (compute-bound) — keep the simpler default |
| `--gpu-memory-frac` | 0.15-0.30 (indifferent for prefill) | ties within 4%; use 0.3+ if the same server also decodes (c20_f06) |
| sparse prefill | `BATCHGEN_V4_SPARSE_PREFILL=1` (default) | dense fallback OOMs at 8192 (eager softmax 17 GiB) |
| scaling economy | halving batch to 64x8192 keeps 62% of throughput | if 1M tokens in flight is too much for your workload |

**HARD LIMIT — silent-drop server bug:** a single `/v1/inference` request whose total tokens exceed
KV-page capacity (~1.05M tokens at frac 0.15) is not backpressured: admission raises mid-flight in
`allocate_pages_for_sequences`, all but ~2 sequences are dropped, and the response returns quickly
with NO error. Clients must cap per-request batches (<=128x8192 here) until fixed.

**Decode side-note from the same campaign:** resident experts + `request_concurrency=32` +
`--gpu-memory-frac 0.15` measured **decode 33.0 tok/s** (coherent output) — above the previous best
c20_f06 (22.3). Concurrency remains the dominant decode lever; validate accuracy before adopting.

**Open items:** (1) MMLU accuracy guard for the recommended config is UNRECORDED — first attempt
failed on missing `PYTHONPATH` (guard subprocess needs `PYTHONPATH=<repo>`), the rerun triggered a
**host-RAM runaway to 95%** during guard decode (no OOM logged; same silent host-leak family as the
post-OOM leak) and was hard-aborted at the guard threshold. (2) The silent-drop and host-leak bugs
deserve server-side fixes (backpressure + reset accounting).

### Fix status (2026-07-03 late)

**UPDATE 2026-07-04 (post-reboot verification runs):**
- **Prefill silent-drop fix VERIFIED WORKING**: the admission preflight required two follow-up fixes
  (unwrap `DualKVCacheCoordinator.primary`; trigger the collective GPU-KV reinit inside admission
  because reset destroys the coordinator and the stock reinit ran only *after* selection). With
  those, the 192x8192 repro shows `[PREFILL] GPU-KV backpressure: admitting 3/192 ... 27/189 ...`
  wave-cycling and **ZERO `Insufficient free pages` errors** (was 9). No sequences are dropped
  during prefill anymore.
- **Decode-side over-admission: FIXED and VERIFIED.** Root cause: decode selection
  (`_prepare_decode_batch`) estimated pages via `get_gpu_pages_for_two_page_buffer()` (working
  set only) while the actual allocation loads the FULL context KV from host — 192x8192 selected,
  `need 6336 worker pages` crash. Fix: the same V4 coordinator preflight as prefill
  (`_truncate_batch_to_gpu_kv_fit`, shared helper, full-context token estimate, MIN-allreduce).
  Verified: `[DECODE] GPU-KV backpressure: admitting 33/192 -> 33/159 -> ...` waves.
- **END-TO-END VERIFIED (2026-07-04 13:17): 192x8192 request returns `results=192 nonempty=192`**
  with ZERO allocation/inference failures; "Detokenization complete: 192 sequences"; host RAM
  bounded. Aggregate 1,008 tok/s for the full prefill+decode of 1.57M tokens at honest capacity
  batching (the earlier "faster" numbers were failure fast-paths). The suspected response-gather
  bug was a phantom — purely downstream fallout of the two admission bugs.
- **Host-RAM runaway did NOT reproduce after the box rebooted** (same config that hit 99.8%
  pre-reboot plateaued at 52%): the leak likely depends on accumulated pre-reboot host state.
  The smaps attribution watcher (`/tmp/autoresearch_v4/run_leakhunt.sh` pattern: 10s
  `smaps_rollup` sampling + 80% auto-kill) is the standing protocol if it recurs.

**Silent-drop fix IMPLEMENTED (verification blocked):** `batchgen_worker.py` —
`_prepare_prefill_batch` now calls `_truncate_prefill_to_gpu_kv_capacity`, which preflights the
admitted list against the V4 GPU-KV pools (`can_allocate_pages_for_sequences`, cumulative prefix)
and MIN-all-reduces the fit count so the SPMD batch stays rank-identical; overflow sequences stay
queued for the next wave (真 backpressure). Logs `[PREFILL] GPU-KV backpressure: admitting X/Y`.
Syntax-verified; behavioral verification (expect 192/192 results on the old 2/192 repro) is
BLOCKED by the host-RAM runaway below.

**Reset-leak fix IMPLEMENTED:** `_reset_for_new_batch` now waits (best-effort, `defer_errors`) and
clears `_pending_kv_append_tasks/_tensors`, which previously survived reset and pinned old-batch
tensors (post-OOM leak, scenario A).

**HOST-RAM RUNAWAY — still open, now better characterized (top-priority bug):**
- Struck 2 of the last 2 server runs (rec_guard2 at 95%, pf_b192f at **99.8%**), configs that were
  previously stable; onset can be as early as **the first warmup decode** ("Selected 1 sequences"
  was the last admission log before the climb).
- Growth ~1.5 GB/s sustained; consumed by the server container (host drained 99.8%→16% on
  `docker kill`). No CUDA OOM, no assert in logs; server actively logging decode timing tables.
- NOT explained by: KV data volume (~2 GB total for the workload), shm segment (fixed-size
  `ftruncate`), `_pending_kv_append_tensors` (bounded by MAX_PENDING_KV_TASKS=256 + cleared).
- Candidate mechanisms to chase (from code audit): DtoH/HtoD engine staging allocations
  (`torch::empty_like` in `tensor_on_demand_copy`), host-KV chunk growth loop
  (`grow_pages_for_sequences` retry), or an allocation loop in the decode-timing path
  (`BATCHGEN_DECODE_TIMING=1` is set by the harness in ALL recent runs — but was also set in
  stable runs).
- Reproduction: any recent harness run may trigger it; watch `free` from the driver and hard-kill
  at 85% (`docker kill <container>` reaps despite the "did not receive an exit event" error).

---

*Status: §1–2, §4 (launch), §6 (baseline), §8 (prefill recommendation), and the VERIFIED rows are
confirmed from the campaign. SWEEP rows are recommendations to validate in the follow‑up
optimization loop (Deliverable 2/3).*
