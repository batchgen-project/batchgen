# DeepSeek-V4-Flash Multi-GPU Decode Deadlock — Root Cause, Fix & Validation

This file covers TWO distinct MP8 decode deadlocks found and fixed in sequence:

1. **Decode-entry collective skip** (below) — a rank with empty local
   decode_uuids skipped a decode-entry all_reduce. Fixed in batchgen_worker.py
   (collective-safe sync helpers + break-validation). Validated markers-ON.
2. **PyNCCL per-layer EP Heisenbug** (next section) — surfaced AFTER fix #1, only
   with the debug tracer OFF. Worked around for the benchmark by defaulting
   BATCHGEN_V4_PYNCCL_COMM=0 (use torch.distributed). A proper PyNCCL repair is
   still open.

---

## Deadlock #2: PyNCCL per-layer EP collective Heisenbug (markers-off)

Status: **worked around** (PyNCCL off by default in the H20 runbook); proper
PyNCCL fix still **open**.

### Symptom
MP8, BATCHGEN_V4_GROUPED_MOE=0 (per-expert loop, experts resident, no host
offload — the intended best config when HBM is sufficient), markers OFF: decode
HANGS at the first decode forward. /proc wchan: 7 ranks state=R (GPU-spin) + 1
rank futex_wait_queue_me. Log frozen 22+ min at the decoding_continuous entry.

### Heisenbug
The EXACT same config + binary runs cleanly when BATCHGEN_DECODE_DEADLOCK_TRACE=1:
markers climb, layers advance L=0 -> L=42, tokens emit. The only difference is an
os.write(fd2) syscall bracketing each per-layer EP collective. That host-side
syscall (a CPU/GIL yield, NOT a CUDA sync) is what unblocks it.

### Root cause (Oracle-adjudicated)
Per-rank backend / collective-order divergence in the PyNCCL EP path. The
per-layer `_ep_all_gather` / `_ep_all_reduce` (model.py) dispatch to EITHER a
PyNcclCommunicator OR torch.distributed via `_use_pynccl()`, and PyNCCL launches
fire-and-forget on the current stream with no work.wait()/event ordering. When
ranks race ahead launching back-to-back collectives, one rank's host path lags
(futex sleep) while the other 7 GPUs spin in NCCL waiting for the missing 8th
participant — the classic 7R+1futex shape. The os.write yields just enough CPU to
let the lagging rank/progress path advance, masking the race. `_use_pynccl()`
already carries a code comment that per-rank divergence "-> collective backend
mismatch -> hang".

### Fix (this branch): default to torch.distributed
The H20 runbook (`v4_h20_rebuild_and_launch.sh`) previously HARDCODED
BATCHGEN_V4_PYNCCL_COMM=1, so PyNCCL could not be turned off via env. Now it is
`${BATCHGEN_V4_PYNCCL_COMM:-0}` — torch.distributed (one backend, one ordering
model, timing-independent) is the default; PyNCCL is opt-in.

Cost: torch.distributed all_gather_into_tensor adds ~8ms/call CPU launch+sync
(~340ms/token over 43 layers per the code comment) vs PyNCCL's stream-direct
submit. Acceptable for correctness / the MMLU-Pro benchmark.

### Proper fix (open, ranked by Oracle)
1. Force torch.distributed for this path (done — the default flip above).
2. Make `_use_pynccl()` globally uniform + fail-closed: broadcast/all-reduce a
   single eligibility bool per decode step; if ranks disagree, ALL fall back to
   torch.distributed.
3. Repair PyNCCL ordering: comm stream waits on producer stream; record a
   completion event; consumer stream waits on it (or work.wait() before reading
   gather/reduce outputs).
Do NOT ship the os.write/yield as a fix — it gives no collective-ordering
guarantee.

### Diagnostic note
The fd-2 marker tracer (BATCHGEN_DECODE_DEADLOCK_TRACE) MUST NOT be used as a
latency or pass/fail probe here: it perturbs the very timing of this bug. For a
real non-perturbing trace, log per-(rank,layer) ep-sequence/backend/shape into a
preallocated ring buffer and dump only on timeout.

---

## Deadlock #1: decode-entry collective skip

## Symptom

Full V4-Flash serving on H20 (MP4 and MP8). Model loads, prefill completes, then
the **decode phase hangs** on the first inference: GPUs pin at ~95% util, no
tokens emitted, request never returns.

## Root cause (confirmed by code + Oracle adjudication)

A rank with an **empty local `decode_uuids` skipped a decode-ENTRY collective**
that the other ranks executed, desyncing the group. The two offending helpers
each returned early *before* their `dist.all_reduce`:

- `_sync_decode_uuids_tensor` — `if not decode_uuids: return []` before the
  presence `all_reduce`.
- `_sync_completion_status_tensor` — `if not decode_uuids: return ...` (and a
  second `if not idx_to_uuid: return ...`) before the completion `all_reduce`.

An idle rank took the early return and raced ahead to `dist.barrier()` (futex
sleep) while the other ranks blocked forever inside the skipped `all_reduce` /
the subsequent per-layer MoE all-gather. Process-state inspection
(`/proc/<pid>/wchan`):

| Run | Rank split |
|-----|------------|
| MP4 | 5 ranks `state=R` (GPU-spinning in a collective) + 3 ranks `futex_wait_queue_me` |
| MP8 | 7 ranks `state=R` + 1 rank `futex_wait_queue_me` |

### Why the earlier "loop-skip" theory was wrong

The original handoff theorized that an idle rank never enters the
`while decode_uuids:` loop (worker.py) and so never runs the per-layer MoE
all-gather. Oracle refuted this: `decode_uuids` is rebuilt every iteration from
the **replicated** `global_batch` (worker.py ~7351/7354, sorted by `global_idx`),
so for a single prompt **all** ranks see it as non-empty and **all** enter the
loop. The loop-skip theory predicts a `1 spinning + N-1 sleeping` split; the
observed split is the inverse (`7+1` on MP8), which matches the decode-entry
collective-skip above, not loop-skip.

Isolation performed:
- `BATCHGEN_V4_PYNCCL_COMM=0` → same deadlock ⟹ NOT PyNccl-specific.
- `TORCH_NCCL_DESYNC_DEBUG=1 TORCH_NCCL_BLOCKING_WAIT=1` + 180s → watchdog never
  fired ⟹ NOT a watched torch.distributed NCCL collective.
- `RELOAD-TEST-V4 DEADLOCK-FIXED hot-reload` log line is a **red herring** — a
  per-call debug marker at the top of `decoding_continuous()`, not a reload.

## NOT caused by the sm-aware kernel work

The deadlock is in DP-decode entry-sync orchestration, independent of kernels.
sm-aware indexer FP8 / MoE gate / prefill (tilelang) all ran cleanly with zero
PTXASError / cvt.e2m1 / FlashMLA errors.

## The fix (this branch, native decode path)

Two collective-safety changes in `batchgen/batchgen_worker.py`, deemed
necessary AND sufficient by Oracle:

1. **Collective-safe sync helpers.** `_sync_decode_uuids_tensor` and
   `_sync_completion_status_tensor` no longer return before their collectives.
   Tensor size and the empty-decision are derived from an `all_reduce(MAX)` of a
   local max-index, so every rank runs the identical collective sequence even
   when its local `decode_uuids` is empty.
2. **Global break-validation.** Before the decode-entry `break`, an
   `all_reduce(MAX)` of a per-rank "has decode work" flag makes the break a
   collective decision; ranks leave the loop together, and a `RuntimeError`
   fires fast on any residual desync.

This is the native-path equivalent of the symmetric-participation intent — and
notably **smaller** than the "port tairan's dummy-token loop" approach the
original handoff proposed (that machinery — zero-row `new_tokens`, the symmetric
padded MoE collective from `3afd3ae9`, "do NOT skip forward on empty batch" —
already exists here; only the entry collectives needed to be made rank-safe).

### Diagnostic tracer

`BATCHGEN_DECODE_DEADLOCK_TRACE=1` (off by default) emits per-rank,
immediately-flushed fd-2 markers (`[DDL] pid=… rank=… …`) at the decode-entry
syncs and the per-layer MoE collectives (states/ids all-gather, all-reduce).
Markers go to fd 2 directly so they survive a hung/buffered logging pipeline.
Used to confirm rank lockstep; see `v4_h20_validate_decode_fix.sh`.

## Validation (H20, MP8, world_size=8)

Launched on 8× H20 with the tracer on. Result:

- **Decode progressed from iteration 0 → 128+**, advancing through all 43
  transformer layers, with **all 8 ranks in lockstep** (per-rank last marker
  identical or one async micro-step apart). Pre-fix, decode hung at iteration 0.
- Zero `cudaHostRegister` / NCCL / desync errors during the decode run.

This is conclusive for the deadlock. A clean generated-token string from the
HTTP smoke could not be captured because the first successful decode triggers a
cold torch-JIT compile of several per-shape decode kernels, which on this node
is pathologically slow and exceeded every curl timeout — a throughput artifact,
not a correctness issue (see JIT cache note below).

## Secondary findings (fixed alongside)

1. **Per-expert host sync (perf).** `_run_owned_experts`
   (`deepseekv4_flash/model.py`) called `counts[expert_idx].item()` once per
   owned expert — ~32 D2H syncs/layer → ~1.4k/token over 43 layers, the dominant
   decode-step cost. Replaced with a single `.tolist()` of the owned slice
   (numerically identical). Perf only; not a deadlock contributor.
2. **`cudaHostRegister failed: invalid argument` (infra).** Fresh launches
   crashed at Host-KV init because the container ran with `ulimit -l` = 64 KB;
   pinning the 100 GB host-KV region needs unlimited memlock. The known-good
   sibling container `luzhan-moegen-runtime-peel-m1a` runs with `memlock=-1`.
   Fixed in the runbook (`--ulimit memlock=-1 --ulimit stack=67108864`).
3. **Non-persistent torch JIT cache (infra).** `core_engine` + the runtime
   `load()` kernels JIT-compile into the container's
   `/root/.cache/torch_extensions` and are lost on teardown, recompiling
   (~15-20 min) every fresh container. Fixed by mounting a persistent host
   volume there (`TORCH_EXT_CACHE`) so later launches reuse the compiled `.so`;
   a `warmup` runbook command populates it once per image build. Verified: all
   four extensions cached to the host volume and survived container teardown.

## Files

- `batchgen/batchgen_worker.py` — collective-safe sync helpers, break-validation,
  `BATCHGEN_DECODE_DEADLOCK_TRACE` markers.
- `batchgen/models/deepseek/deepseekv4_flash/model.py` — MoE-collective markers;
  `_run_owned_experts` per-expert-sync batching (perf).
- `docker/v4_h20_rebuild_and_launch.sh` — persistent JIT cache, memlock fix,
  `warmup` / `cache-status` subcommands.
- `docker/v4_h20_validate_decode_fix.sh` — launch + decode-smoke + wchan/marker
  capture for reproducing and confirming the fix on H20.
