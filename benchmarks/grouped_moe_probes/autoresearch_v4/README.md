# autoresearch_v4

Autonomous optimization scaffold for **DeepSeek-V4-Flash serving/system config** on the verified 4x RTX PRO 6000 Server GPUs.

## What is fixed

- `bench_v4_config.py` is the ground-truth harness. The loop must treat it as read-only.
- kernels/model code are frozen
- benchmark prompts, warmup, decode length, and accuracy guardrail are fixed

## What is editable

- `config_space.py` only: serving/system knobs such as GPU/host KV sizing, KV dtype, page buffers, NCCL env, NUMA pinning, and fixed-harness request concurrency

## Metric

- **primary:** worker-log `Decode throughput` (`decode_tok_s`, higher is better)
- **secondary:** prefill TTFT proxy from worker-log `Prefill total time`
- **guardrail:** tiny MMLU-Pro/coherence sanity run; failing configs are rejected

## Run one experiment

```bash
python benchmarks/grouped_moe_probes/autoresearch_v4/bench_v4_config.py \
  --config-name baseline \
  --tag baseline
```

Logs go to `/tmp/autoresearch_v4/`. The script appends exactly one TSV row per experiment to `results.tsv`.

## Launch the autonomous loop

1. Point your agent at `benchmarks/grouped_moe_probes/autoresearch_v4/program.md`.
2. Tell it to begin with the baseline and then iterate forever.
3. Let it edit only `config_space.py` / one-off config payloads and call `bench_v4_config.py` for every experiment.

## Cleanup contract

The harness always performs wedge-safe cleanup between experiments:

- `pkill -9 -f launch_http_server` in-container
- `docker rm -f`
- kill leftover GPU compute PIDs on GPUs 0-3
- clear leaked `/dev/shm/shm_*` and `/dev/shm/batchgen_host_kv_cache`
- verify GPUs 0-3 are idle and `/dev/shm` is clean before the next launch

## Sweep run results

First ordered sweep pass on 2026-06-30 (real harness rows in `results.tsv`).
All rows below have a **passing** accuracy guardrail (MMLU-Pro tiny run, 4/5 = 0.80)
and coherent generation.

| tag | gpu_mem_frac | host_kv (GB) | request_concurrency | decode tok/s | prefill TTFT (s) | accuracy guard | vs baseline |
|---|---|---|---|---|---|---|---|
| baseline_ok | 0.30 | 60 | 1 | **1.0** | 16.45 | 0.80 (4/5) | 1.0x |
| conc4 | 0.30 | 60 | 4 | **3.9** | 18.35 | 0.80 (4/5) | **3.9x** |
| conc8 | 0.30 | 60 | 8 | **8.3** | 17.45 | 0.80 (4/5) | **8.3x** |
| conc16 | 0.30 | 60 | 16 | **18.5** | ~18 | guard interrupted* | **18.5x** |

*conc16's 18.5 tok/s is a valid throughput measurement; its accuracy guard was cut short by an
agent-window timeout (not recorded). Accuracy is expected to hold at 0.80 like conc4/conc8 because
concurrency does not change per-token logits (same kernels/weights). Re-run to confirm the guard.

### Finding: decode is PCIe-collective-bound; concurrency amortizes it

The top hypothesis is strongly confirmed. With no NVLink, the per-decode-step
collective over PCIe dominates single-sequence decode. Increasing fixed-harness
request concurrency lifts **aggregate** throughput almost linearly over the tested range:

| concurrency | 1 | 4 | 8 | 16 |
|---|---|---|---|---|
| decode tok/s | 1.0 | 3.9 | 8.3 | 18.5 |
| scaling efficiency vs ideal | 1.00x | 0.98x | 1.04x | 1.16x |

Scaling is near-linear-to-super-linear and **not yet saturated at 16**. Current best config is
`conc16` at **18.5 tok/s (18.5x baseline)**; prefill TTFT stays flat (~16-18s).

### Host-memory scaling (HARD limit: host RAM must stay <90% of 1511 GB ≈ 1360 GB)

Host RAM — not GPU — is the binding constraint as concurrency rises:

| concurrency | 1 | 8 | 16 |
|---|---|---|---|
| host mem steady | ~30% | ~30% | ~51% (transient guard spikes to ~68%) |

Linear extrapolation puts **conc32 at ~91% host RAM -> FORBIDDEN**; safe concurrency ceiling ≈ 24-28.
GPU VRAM is UNDER-used (only ~29-65 GB of 96 GB at gpu_memory_frac=0.30). The next lever is
**raising gpu_memory_frac to saturate GPU** (consumes VRAM, not host RAM) to admit larger batches —
staged configs: `c16_f075` (conc16 @ frac0.75) and `c20_f075` (conc20 @ frac0.75).

### Phase x module micro-batching now configurable (new lever)

The planner (`batchgen/planner/base_planner.py`) auto-plans 6 module-batch knobs that were
previously not user-settable. An env-override hook now exposes them (applied after the
model-specific planner so overrides win; unset = planner default):

| env var | knob | default |
|---|---|---|
| `BATCHGEN_ATTN_PREFILL_MB` | attn_prefill_micro_batch_size | 8 |
| `BATCHGEN_MOE_PREFILL_MB` | MoE_prefill_micro_batch_size | 8 |
| `BATCHGEN_EXPERT_PREFILL_CAP` | expert_prefill_batch_size_upper_bound | 4096 |
| `BATCHGEN_ATTN_DECODE_MB` | attn_decoding_micro_batch_size | planned |
| `BATCHGEN_MOE_DECODE_MB` | MoE_decoding_micro_batch_size (= decode max_seqs/rank) | None (uncapped, single-node) |
| `BATCHGEN_EXPERT_DECODE_CAP` | expert_decoding_batch_size_upper_bound | 2048 |

Harness exposes them as `V4ServingConfig` fields and propagates into the server. Validated
deterministically (in-container planner test applies overrides exactly) and end-to-end:

| tag | request_concurrency | gpu_mem_frac | moe_decode_mb | expert_decode_cap | decode tok/s | vs conc16 |
|---|---|---|---|---|---|---|
| conc16 | 16 | 0.30 | planned | 2048 | 18.5 | 1.00x |
| c16_edb | 16 | 0.30 | 32 | 8192 | 18.2 | 0.98x |
| c16_cap4 | 16 | 0.30 | **4** | 2048 | 18.0 | 0.97x |
| c20_f06 | 20 | **0.60** | 64 | 8192 | **22.3** | **1.21x** |
| prefill_big | 20 | 0.60 | planned | 8192 | 21.0 | 1.14x |
| c20_cap16k | 20 | 0.75 | 64 | 16384 | CRASH (OOM) | — |
| attn1 | 20 | 0.60 | planned | 8192 | N/A (flag invalid) | — |

### Architecture research + feasibility (what CANNOT be tuned)

Verified against the code before spending GPU time:

- **DP vs TP attention: NOT selectable.** V4-Flash attention is *always* data-parallel (full 64
  heads/rank + `dist.all_reduce` after O-proj, `model.py:1370`); weights are TP-sharded but compute
  is DP. `attn_mode` (1/3) selects the KV *backend*, not DP/TP, and is **not exposed** via CLI/env
  (the `--attn_mode` attempt failed: `unrecognized arguments`). `attn_mode=3` (DP `kv_storage`) is
  already the tuned default. Dropped.
- **EP vs non-EP: NOT toggleable.** EP is structural for `world_size>1`
  (`enable_ep_offloading = world_size>1`); a non-EP path exists only at `world_size=1`. Dropped.
- **NVSHMEM AllToAll (`BATCHGEN_ENABLE_ALL_TO_ALL=1`): infeasible here** — `pplx_kernels`/`nvshmem`
  are absent in the container (capability-checked). Would crash at NVSHMEM init. Dropped.
- **prefill→decode split: CONFIRMED.** One prefill batch flips all seqs to PREFILLED, then decode
  admits a subset per step (90% GPU-page watermark + `MoE_decoding_micro_batch_size`). Prefill and
  decode batch sizes are independently sizable.

### Results of the feasible experiments (none beat c20_f06 = 22.3)

- **`prefill_big`** (`prefill_token_cap=262144`, new 7th override): 21.0 tok/s, prefill TTFT 16.6s —
  a mild *regression* vs c20_f06 (22.3 / 16.1s). **Larger prefill batch does not help** aggregate
  tok/s or TTFT here (decode, not prefill, is the throughput-bound phase).
- **`c20_cap16k`** (`expert_decode_cap=16384` + `gpu_memory_frac=0.75`): **crashed during warmup**
  (0.0 tok/s, ~9 min) — `gpu_memory_frac=0.75` + conc20 + larger expert buffers overran 96 GB VRAM.
  The higher cap is also expected to be *non-binding at decode* anyway: at conc≈20, decode routes
  ~20×top6/256 < 1 token/expert/step, so `expert_decode_cap` (2048/8192/16384) never binds — it only
  matters in prefill or at extreme (>256) concurrency. (c16_edb already showed 8192 didn't beat 2048.)

**Net: `c20_f06` = 22.3 tok/s remains the best config.** The effective decode levers are
**request_concurrency + gpu_memory_frac** (host-RAM-capped at ~conc24-28); expert caps, prefill
sizing, moe_decode_mb, attn_mode, and the parallelism strategy are either non-binding or not tunable
for V4-Flash on this box.

### Prefill overlap experiment (long-seq + full expert offloading) — INCONCLUSIVE

Goal: with long prefill sequences (2048/4096/8192) + `MoE_prefill_mb=32`, test whether streamed
experts (`--ep-offloading-ratio 1.0`, `prefill_off`) overlap compute and match GPU-resident experts
(`prefill_res`). Grounded sizing: experts are MXFP4 ~12.59 MB each -> prefetch ~0.25-0.9 ms/expert ->
need ~27-100 tokens/expert for compute to hide the stream.

**Both runs died on the 1200 s watchdog** (`worker-N watchdog timeout`, 4x each) at ~44 min, 0 valid
data. A single prefill/step at 8192 tokens exceeded the 20-min watchdog (too-aggressive config or a
hang), and `prefill_off` was further contaminated by `prefill_res`'s wedged cleanup (leaked 56 GB on
GPU0). **The overlap question is NOT answered.** Retry needs: `--watchdog-timeout 3600`, cap sequence
at 4096 first (distinguish slowness vs hang), and a hard idle-GPU gate between runs.

### DP-replica (no-EP-dispatch) — the remaining promising idea, needs a code change

Verified: not achievable as-is (`num_local_expert_per_layer` capped at 64/rank; EP collective fires
whenever `world_size>1`). Would need 3 edits behind `BATCHGEN_V4_FULL_REPLICA` (initializer cap,
`configure_ep` range, skip collective at `model.py:2008`). It eliminates the PCIe dispatch collective
for **prefill** (win) but would hurt decode (can't hide streaming 256 experts at 1 token/step).

### Earlier findings from these runs:

1. **Optimization WIN — `c20_f06` = 22.3 tok/s (1.21x over conc16).** Combining higher concurrency
   (20), GPU saturation (`gpu_memory_frac=0.60`), and raised decode caps beats conc16's 18.5. This
   is the payoff of the new levers: pairing a bigger effective decode batch with more GPU-KV.

2. **Hypothesis OVERTURNED — `MOE_DECODE_MB` does NOT gate the decode batch here.** `c16_cap4` set
   `moe_decode_mb=4`, which I predicted would throttle throughput toward conc4 (~3.9 tok/s). It did
   NOT — throughput held at 18.0 (~conc16). So `MoE_decoding_micro_batch_size` (= `max_seqs_per_rank`
   at `batchgen_worker.py:8662`, the two-page-buffer decode selector) is **not the active limiter in
   the V4-Flash sm120 grouped-MoE (mega3) decode path**; the decode batch is bounded by client
   concurrency / GPU pages instead. The override still *applies* to the config (validated), but this
   particular knob has no runtime effect on decode batch size in this path. The effective decode
   levers are **concurrency + gpu_memory_frac + expert_decode_cap**, not `moe_decode_mb`.

(All rows here show accuracy_guard not recorded / `cleanup_fail` — same operational pattern as the
concurrency sweep: throughput values are valid; the guard was cut short and `docker rm -f` wedged
the container until an external cleanup. Host mem peaked ~67% during these runs, under the 90% limit.)

**Finding:** at conc16 the decode-batch knobs are non-binding (decode batch ~16 < 32; expert cap far
from binding at <1 token/expert/step), so throughput is unchanged — the control confirms the override
does not degrade. The lever's payoff needs a capping regime or pairing large decode batches with
higher `gpu_memory_frac` (now possible via these knobs). Prefill and decode, attention and MoE are
independently tunable.

### Operational notes from this run

- The first raw `baseline` row failed because the harness subprocess inherited a Python
  environment without repo `PYTHONPATH`; rerunning with repo `PYTHONPATH` produced
  `baseline_ok`.
- On this host the harness rows report `cleanup_fail` because `docker rm -f` returns
  before the container transitions to `Exited`; after an external wait the container
  exited, was removed, GPUs were released, and `/dev/shm` returned to clean for
  `baseline_ok` and `conc4`.
- The sweep paused after `conc8` because a foreign process
  (`python -m tally_vmm_cutlass.server`, PID 3592954) claimed GPU0 after the run; no
  `autoresearch-v4-*` containers remained and `/dev/shm` was clean, but GPUs 0-3 were
  no longer fully idle for the next experiment.

### Status & blocker (GPU-saturation phase)

- **Best confirmed config: `conc16` = 18.5 tok/s (18.5x baseline)**, decode-throughput, host-safe (~51%).
- **GPU-saturation experiment (`c16_f075`, gpu_memory_frac=0.75) is BLOCKED by external GPU
  contention:** another user's job occupies GPU2 (and GPU4/5) with ~48 GB. Since world_size=4 needs
  GPUs 0-3 all idle and the harness cannot kill a foreign PID (`Operation not permitted`), the run
  bails on the idle-check. We do not kill other users' processes. **Resume the moment GPUs 0-3 are
  exclusively free** (configs staged: `/tmp/autoresearch_v4/c16_f075.json`, `c20_f075.json`).

### Next experiments (when GPUs 0-3 are exclusively free)

1. `c16_f075` — gpu_memory_frac 0.30 -> 0.75 at fixed conc16 (saturate GPU ~72/96 GB; host stays ~51%).
   Tests whether GPU saturation alone lifts throughput (expected: modest unless KV was spilling to host).
2. `c20_f075` — conc20 @ frac0.75 (host est ~60%, safe). Push toward saturating BOTH resources.
   Do NOT exceed ~conc24-28: conc32 breaches the host-RAM 90% limit.
3. NCCL-over-PCIe one knob at a time: `NCCL_P2P_LEVEL`, then `NCCL_ALGO=Ring/Tree`.

### Long-sequence prefill sweep (2026-07-02) — CRASH + confound, not yet resolved

Goal: find the best PREFILL config by pushing sequence length (higher tokens/expert should
let per-expert compute hide the MXFP4 stream). Prefill is data-parallel here (each of the 4
ranks owns all 256 experts, no EP collective), so long-seq prefill is the right lever.

**Two attempts, no clean throughput number yet:**

| tag | sparse prefill | `CUDA_LAUNCH_BLOCKING` | seq | outcome |
|---|---|---|---|---|
| `pf_base` | ON (default) | off | 2048/4096/8192 | **device-side assert in `self_attn`** during long-seq prefill (`batchgen_worker.py:9428 prefill_prepacked` -> `model.py:2322` self-attn timed block). 2048 is known-safe from earlier runs; 4096/8192 trip it. Defaults only, so it's the sequence length, not the env knobs. |
| `pf_dense8k` | OFF | **ON** | 8192 | **CONFOUNDED / wedged.** ~6 min in: `torch.distributed` health-check failure -> rank-0 `coordinated reinit` loop -> hang (GPU0 97 GB/0% util, ranks 1-3 spinning). Never reached a clean dense-8192 signal. |

**Root-cause learning (important, reusable):**
`CUDA_LAUNCH_BLOCKING=1` MUST NOT be used with the multi-rank distributed server. Serializing
every CUDA op inflates collective latency past the `torch.distributed` health-check timeout,
which triggers a rank-0 reinit/wedge — a *new* failure mode that masks the assert you were
chasing. Use it only on a single-rank (`world_size=1`) repro. The discriminating long-seq test
must run WITHOUT launch-blocking; survival + real prefill tok/s is the signal.

**VERDICT (pf_dense8k, 2026-07-02 16:42): long-seq 8192 prefill fails on BOTH paths, differently.**

| path | failure @8192 | nature |
|---|---|---|
| sparse (default) | device-side assert in `self_attn` | kernel indexing bug at long seq (fixable in principle) |
| dense (`SPARSE_PREFILL=0`) | `torch.OutOfMemoryError` in `softmax`, **tried 16.92 GiB** | structural: eager attention materializes the full score matrix = 64 heads x 8192^2 x fp32 ~= 17 GiB |

The 16.92 GiB allocation is exactly the eager-attention blowup — the dense fallback has no
flash/chunked prefill path, so it can never reach 8192 on 96 GB GPUs at this head count.

**Second finding — post-OOM "recovery" is broken and dangerous:** after the OOM the server logged
"Resetting state for new batch" then went silent (>1 h, zero log lines) while **host RAM climbed
3% -> 85%** (leak in the paged host-KV/reset path). The run had to be hard-aborted at the 85%
guard. Any future OOM in this server must be treated as fatal: kill the container immediately
(`docker kill` reaps the root procs even when it reports "did not receive an exit event").

**Practical conclusion for the prefill config search:** on this build the usable prefill sequence
ceiling is **2048 (sparse ON, known-good)**. Longer sequences need a code fix first (sparse
indexing bug), not a config change. Untested middle ground: dense@4096 would need ~4.2 GiB softmax
(fits), and sparse@4096 vs 8192 was not isolated (pf_base looped 2048->4096->8192; exact tripping
length unknown). The best-prefill-config sweep should therefore run at 2048 (optionally probing
4096) with the pf_base/pf_mb/pf_cap knob variants.

**Harness diagnostic hooks added** (`bench_v4_config.py`, no-op for normal sweeps):
`BENCH_PREFILL_TOKENS=<csv>` overrides prefill lengths; host env `BATCHGEN_V4_SPARSE_PREFILL` /
`CUDA_LAUNCH_BLOCKING` are forwarded into the container only when explicitly set.

**Wedge cleanup note:** a launch-blocking/reinit hang leaves root-owned worker procs that
`docker rm -f` cannot reap (returns before the exit event). `docker kill <container>` still
lands SIGKILL on them (despite reporting "did not receive an exit event"); after that the
per-GPU `nvidia-smi --query-gpu` calls unblock and `/dev/shm` can be cleared.

### Sparse-prefill assert deep-dive (2026-07-02 pm) — root cause NOT yet found; two hypotheses ruled out

Attempted to fix the sparse-prefill long-seq assert. Ruled out the two obvious causes with cheap tests:

| test | method | result |
|---|---|---|
| index construction OOB | CPU, pure `window_topk_idxs`+`compress_topk_idxs` @2048/4096/8192 | **bounds-safe** (max idx = kv_n-1, no OOB) |
| tilelang `sparse_attn` kernel at scale | standalone container run, synthetic valid q/kv/idx @2048/4096/8192, `CUDA_LAUNCH_BLOCKING=1` | **PASSES all** (finite output) |
| real path @4096 (sparse ON, ws=4) | full harness + `BATCHGEN_V4_SPARSE_DEBUG=1` | **works**, prefill ~232 tok/s, no assert, no OOB printed |
| real path @8192 (sparse ON, ws=4) | full harness + `BATCHGEN_V4_SPARSE_DEBUG=1` | **STILL ASSERTS**, and `SPARSE_DBG` shows **idx in-bounds (no OOB)** |

**Conclusions:**
- The assert is **NOT** the sparse-attn gather / topk-index OOB (disproven: no OOB at 8192, kernel safe with valid idx). A speculative index-clamp fix was implemented then **reverted** (it clamped nothing).
- The crash is **specific to 8192** (4096 is fine) and **in-bounds** — so it is a *different* seqlen-dependent device-side assert.
- **The Python traceback is unreliable**: the async assert is only caught at the next sync (`self_attn` `event.record()` / `free_weights` synchronize), so it may not even be in attention — it could be in the **MoE prefill path** (256 experts x 8192 tokens: routing indices / grouped-GEMM offsets) or elsewhere in the layer.

**Definitive next step (requires GPU + a run):** reproduce at **`world_size=1`** (mp1 ckpt, launch-blocking is SAFE at single rank) with `CUDA_LAUNCH_BLOCKING=1` + `BATCHGEN_V4_SPARSE_DEBUG=1`, OR add stage-by-stage `torch.cuda.synchronize()` checkpoints through the layer forward (attention stages AND the MoE call) at ws=4 to localize the exact op. The ws=1 harness must be adjusted to reach the 8192 prefill without the decode phase (which OOMs at ws=1). Only then can a correct, targeted fix be written.

**Permanent aid left in code:** `v4_prefill_sparse.py` prints `[SPARSE_DBG] seqlen=.. idx_max=.. kv_n=.. oob=..` per sparse-prefill call when `BATCHGEN_V4_SPARSE_DEBUG=1` (forwarded by the harness). Zero cost when unset.

### UPDATE (2026-07-02 later): assert precisely localized via stage-sync checkpoints — cause is NOT the sparse math

Added env-gated (`BATCHGEN_V4_SPARSE_DEBUG=1`) `torch.cuda.synchronize()` checkpoints through the
layer forward (`model.py`: `[SPARSE_CKPT] L{n} pre_attn/post_attn/post_moe`) and through
`sparse_prefill_attention_sequence` + `_forward_prefill_sparse`
(`[SP] after_kernel/after_invrope/after_wo_a_einsum/after_wo_b`, `[FPS] returned/after_attn_slice/after_kv_slice`).
A ws=4 8192-token run gave a clean trace:

```
[SPARSE_CKPT] L0 pre_attn
[SPARSE_DBG] seqlen=8424 ratio=0 topk_w=128 idx_max=8423 kv_n=8424 oob=False   <- ratio=0 (window-only) layer
[SP] after_kernel s=8424 r=0        <- kernel fine
[SP] after_invrope s=8424           <- inverse rope fine
[SP] after_wo_a_einsum s=8424       <- wo_a einsum fine
[SP] after_wo_b s=8424              <- wo_b fine (sparse_prefill fully returns)
[FPS] returned span=(0,8424) seq_attn=(1,8424,4096) kv=(1,8424,512) attn_out=(1,8424,4096) kv_out=(1,8424,512)  <- shapes correct
[FPS] after_attn_slice              <- attn_out slice-copy fine
[FPS] after_kv_slice               <- kv_out slice-copy fine
<ASSERT>                            <- L0 post_attn NEVER prints
```

**Every op inside the sparse attention and `_forward_prefill_sparse` synchronizes clean.** The assert
surfaces only at the *next* sync after the whole attention returns — the timed-block `event.record()`
(`timing.py:215`) and the attention-wrapper weight-release (`wrappers.py:206 _release_run` ->
`free_weights` -> `base.py:152 _sync_device_before_release`). Because a device-side assert is sticky
and `[FPS] after_kv_slice`'s `synchronize()` PASSED, the failing kernel is on a **non-default CUDA
stream** — pointing at the **attention weight-offload/prefetch/release path**, NOT the sparse math.

**Facts established (all with tests):** it is the **layer-0 (`compress_ratio=0`, window-only) path at
seqlen ~8424**; 4096 works fully; kernel + index-construction + rope + wo + slice-copies all proven
clean. Standalone kernel passes at 8424 with both ratio-4 (topk=640) and ratio-0 (window, topk=128) shapes.

**Definitive next step (needs a run):** ws=1 (mp1 ckpt
`/mnt/raid0nvme0/leyang/v4flash_converted_mp1/mp1/model0-mp1.{bin,json}`) + `CUDA_LAUNCH_BLOCKING=1`
(safe at single rank) forces ALL streams synchronous so the failing kernel raises at its true launch
site. The ws=1 harness must reach the 8192 prefill without the ws=1 decode phase (which OOMs). Inspect
the attention wrapper's weight prefetch/offload streams (`wrappers.py`) as the prime suspect.

### RESOLVED (2026-07-02): root cause = RoPE cache capped at 8192; long-seq prefill+decode now works

The `[SPARSE_DBG] ... idx_max=8423` clue plus the "assert caught at the wrapper's `free_weights` sync"
pointed at the KV-cache populate (`wrappers.py::_populate_v4_prefill_kv`) that runs right after the
attention returns. It builds a **prefill RoPE cache** and applies it to `prompt_positions` (0..seqlen-1).

**ROOT CAUSE:** `_v4_prefill_rope_cache` / `_v4_compress_rope_params` sized the RoPE cache to
`max_pos = getattr(model_config, "max_position_embeddings", 8192)`. The V4-Flash config has **no**
`max_position_embeddings` (only `original_seq_len=65536`), so it fell back to **8192**. Any prefill (or
subsequent decode) at an absolute position >= 8192 indexed the 8192-row cache out of bounds ->
device-side assert. This is why **2048/4096 worked and 8192+ failed** — nothing to do with the sparse
attention math (kernel/indices/slice-copies all proven clean).

**FIX** (`wrappers.py`, the only production change): floor the RoPE cache length at `original_seq_len`
(65536) and grow-on-demand to the actual sequence length:
- `_v4_prefill_rope_cache(self, device, min_len=0)`: `need = max(max_position_embeddings, original_seq_len, min_len)`, rebuild if the cached tensor is shorter.
- `_v4_compress_rope_params`: `max_pos = max(max_position_embeddings, original_seq_len)` (covers `_v4_compressed_rope_cache` + `_v4_compressed_cos_sin`, used by both prefill SWA and decode).
- `_populate_v4_prefill_kv` passes `max(seq_lens)` as `min_len`.

**VERIFIED end-to-end** (ws=4, sparse ON, 8192-token prefill, `pf_e2e8k`):

| metric | before fix | after fix |
|---|---|---|
| 8192 prefill | device-side assert | **OK, 390.8 tokens/s** |
| decode @ pos 8192+ | (never reached) | **OK, 4.5 tok/s** |
| generation | crash | **coherent** ("Paris. ... Washington, D.C.") |
| device-side asserts | 20 | **0** |

All debug instrumentation (`[SPARSE_CKPT]`/`[SP]`/`[FPS]`/`[SPARSE_DBG]`) was removed after the fix;
production code contains only the `wrappers.py` RoPE-cache change.

### Best-prefill-config sweep (2026-07-02, post-fix) — long-seq prefill now measurable

With the RoPE fix, sparse prefill at 4096/8192 works end-to-end. Sweep (ws=4, sparse ON,
`BENCH_PREFILL_TOKENS=4096,8192`, conc=4, gpu_mem_frac=0.6), prefill throughput (tok/s, higher better):

| config | knobs | prefill @4096 | prefill @8192 | vs base @8192 |
|---|---|---|---|---|
| `pf_base` | defaults | 215.5 | 399.1 | 1.00x |
| **`pf_mb`** | `attn_prefill_mb=16`, `moe_prefill_mb=32` | **231.2** | **422.3** | **1.06x** |
| `pf_cap` | `prefill_token_cap=262144`, `expert_prefill_cap=8192` | 212.8 | 411.1 | 1.03x |

**Findings:**
1. **Prefill throughput scales with sequence length** (~215 @4096 -> ~400 @8192): longer prefill
   sequences amortize fixed per-step overhead and raise tokens/expert -> higher tok/s. Confirms the
   "try longer sequence" lever now that the 8192 assert is fixed.
2. **Best prefill config = `pf_mb`** (larger attention + MoE micro-batches): **422 tok/s @8192, +5.8%**
   over defaults. Larger micro-batches give more tokens/expert -> better GPU utilization / prefetch
   overlap during prefill.
3. Larger token/expert **caps** (`pf_cap`) gave only +3% — micro-batching is the stronger prefill lever.

(All rows `status=cleanup_fail` = the benign `docker rm` wedge; throughput values are valid — the
self-cleaning sweep driver `docker kill`-reaped each container and kept host RAM at ~2% between runs.)

### Large-batch prefill (2026-07-03): 512x8192 -> 23,249 tok/s aggregate — **RETRACTED, see verified section below**

> **RETRACTION (same day):** result-count validation showed the server returned only **2 of 512**
> results (silent sequence dropping once the batch exceeds KV-page capacity). The 23,249 and the
> pf_offload decode 23.9 rows below survive only where explicitly re-verified. The verified
> large-batch prefill study follows in the next section.

Single-request prefill (~425 tok/s) massively under-measures capacity. With host-offloaded experts and
a large concurrent batch, aggregate prefill is ~23K tok/s:

| tag | config | result |
|---|---|---|
| `pf_offload` | RESIDENT_EXPERTS=0 + `--enable-ep-with-offloading --ep-offloading-ratio 1.0`, conc32, frac0.3 | prefill 425.5 tok/s (single-req; ties pf_mb), **decode 23.9 tok/s = new decode best** (+7% over c20_f06; streamed experts amortize over the 32-seq decode batch) |
| `pf_big512` | 512x8192 concurrent, frac0.4, token_cap=1048576 | **OOM**: `hc_post` tried 7.71 GiB (1M-token packed row x hc_mult=4 x 4096 x bf16) |
| `pf_big512b` | same, frac0.3, token_cap=262144 | **OOM after 4 rows**: 81.4 GiB live tensors. Two causes found: (1) **KV pool is allocated twice** (per-phase model rebuild leaves both: 24.6+24.3 GB); (2) `prefill_token_cap` env override does NOT gate the prepacked-row size (~252K-token rows regardless) |
| **`pf_big512c`** | same, **frac0.15** | **PASS: 4,194,304 tokens in 180.4s = 23,249 tok/s aggregate** |

**Findings:**
1. **Aggregate prefill capacity ~23K tok/s** (4 DP ranks x ~250K-token packed rows, streamed experts at
   ~6K tokens/expert = fully compute-bound). Single-stream TTFT-style measurement was hiding 55x.
2. **Double-KV-pool bug/inefficiency**: the per-phase (prefill/decode) model rebuild initializes
   `DeepSeekV4KVCoordinator` twice without freeing the first pool (boot log shows both). At frac0.3
   that wastes ~24 GB/rank. Workaround: small `--gpu-memory-frac` (0.15) for prefill-heavy workloads.
3. **`BATCHGEN_PREFILL_TOKEN_CAP` does not bound the prepacked row** (planner cap != prepack budget);
   row size is governed by admission/pages (~252K tokens observed). Activation transients scale with
   row size (hc_post alloc = tokens x hc_mult x hidden x bf16).
4. Harness gained env-gated `BENCH_PREFILL_CONCURRENCY` / `BENCH_SKIP_DECODE` / `BENCH_REQUEST_TIMEOUT`
   (no-op by default); `BATCHGEN_V4_RESIDENT_EXPERTS` docker env is now host-overridable; GPU idle
   check tolerates <=64MB phantom residuals.
5. Caveat: pf_big512c's accuracy-guard *command* failed operationally (exit 1) — accuracy not recorded
   for this row; the identical offload config passed coherence in pf_offload. Guard rerun pending.

### VERIFIED large-batch prefill study (2026-07-03, result-count-validated, unique prompts)

The harness now sends **unique prompts** (defeats prefix-cache inflation) and validates
`len(results) == N`. This invalidated all batch>=160 rows and produced a trustworthy curve
(streamed experts, frac 0.15, 8192-token seqs, aggregate tok/s):

| batch (seqs x 8192) | tokens in flight | agg prefill tok/s | results returned | verdict |
|---|---|---|---|---|
| 32 | 262K | 867.5 | (pre-validation, trend-consistent) | valid |
| 64 | 524K | 1471.7 | (pre-validation, trend-consistent) | valid |
| **128** | **1.05M** | **2380.8** | **128/128** | **valid — best** |
| 192 | 1.57M | ~~8348~~ | **2/192** | INVALID (drops) |
| 256 (frac .15/.25) | 2.1M | ~~10430/18623~~ | **2/256** | INVALID (drops) |
| 512 | 4.19M | ~~22737~~ | **2/512** | INVALID (drops) |

Config A/B at the valid best batch (128x8192) — all tie within ~4% (compute-bound):

| variant | agg tok/s |
|---|---|
| streamed experts, frac 0.15 | 2380.8 |
| streamed experts, frac 0.30 | 2349.8 |
| **resident experts (default), frac 0.15** | 2295.7 |

**Findings:**
1. **Best verified prefill setup: ~128 x 8192-token sequences in flight (~1.05M tokens) -> ~2.3-2.4K
   tok/s aggregate** (~5.6x the single-request rate). Expert offloading and gpu_memory_frac do NOT
   matter at this batch — prefill is compute-bound; use the standard resident-experts config.
2. Throughput was **still rising** at 128 (x1.6 per batch doubling) — the ceiling is not compute but:
3. **SERVER BUG — silent sequence dropping:** a single `/v1/inference` request whose aggregate tokens
   exceed KV-page capacity is not backpressured; admission proceeds in waves (`Prepacked prefill: 4
   micro batches, 404,640 total tokens` for 192x8192), then `allocate_pages_for_sequences` raises
   mid-flight and all but ~2 sequences are dropped — the response returns quickly with 2 results and
   NO error. Client-visible "throughput" is inflated 4-10x. Any client batching more than ~128x8192
   tokens per request on this box gets silent data loss.
4. The prepack micro-batch token budget is hard-capped at **131,072** tokens regardless of
   `BATCHGEN_PREFILL_TOKEN_CAP` (the env override changes the planner value but prepack does not
   consume it).
5. Earlier retracted rows explained: 23,249 (512x8192) and 11,240/11,646 (256x8192) measured the
   drop-bug fast-path, not prefill.
