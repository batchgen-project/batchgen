# Phase 2.8 Handoff — Native decode + boundary port

This file is the single source of truth for the next session. It answers:
- Where did we leave off (commits, test state, validated benchmarks)?
- Why do we need Phase 2.8 (the L4 crash and its root cause)?
- Exactly what to port, in what order, with what invariants, to what gates?

---

## 1. Current state (2026-04-13 end-of-session snapshot)

### Branch & commits
Worktree: `/home/tairan/workspace/BatchGen-reextract` on branch `tairan/worker-reextract`. Editable install points at the same path in the wechat_87 and wechat_96 containers.

Most recent commits (top of branch → older):
```
b4ef7a03 fix(phase-2.7): per-rank decode cap falls back to attn when MoE is 0
91ad6469 test(phase-2.7): update prefill tests for ensure_prefill_setup split
c4b191a4 fix(phase-2.7): flush_and_reconfigure moves to ensure_prefill_setup (once per phase)
80940c63 fix(phase-2.7): scheduler page reservation matches allocator verbatim
229d1b49 fix(phase-2.5): capacity-aware decode prepare_batch
78ce3f29 fix(phase-2.5): make GPU/host KV backends resolve manager dynamically
c9f2dcd2 fix(phase-2.5): hard-fail on GPU KV exhaustion + page-table inconsistency
2f4c1c51 feat(F9-F10): collapse worker_reextract_entry.py + archive legacy paths
f1720df6 feat(F8): orchestrator becomes the only path (delete BATCHGEN_USE_REEXTRACT gate)
cd87ed0a feat(F4-F6): native prefill forward + decode setup + decoding_continuous via LegacyInfraBackend
b066b1f2 feat(F3): native prefill config via LegacyInfraBackend (delete prefill_config_delegate)
…earlier F1/F2 commits…
```

### Tests green
`pytest tests/unit/worker/` — **376 passed** on the `batchgen` conda env in the `tairan-batchgen` container at wechat_87.

### Benchmarks already validated on branch HEAD `b4ef7a03`
| Model / level | Accuracy | Tok/s | Notes |
|---|---|---|---|
| GPT-OSS-120B L1 (32/512) | 65.62 % | 128 | matches legacy |
| GPT-OSS-120B L2 (128/4096) | 83.59 % | 251 | matches legacy |
| GPT-OSS-120B L3 (2048/10240) | 68.21 % | 632 | `reasoning_effort="low"` baseline |
| Kimi-K2.5 L1 (32/512) | 18.75 % | ~105 | matches legacy (512 cap is too tight, expected) |
| Kimi-K2.5 L2 (128/4096) | 85.16 % | 311 | matches legacy |
| Kimi-K2.5 L3 (2048/10240) | 80.62 % | 2107 | on-par subset accuracy (87.5 % on L2-overlap subset) |

### Open L4 failure (the reason this phase exists)
Kimi L4 stress: `stress_client.sh` (full MMLU-Pro + ~5k LongBench, `--max-decode 262144`, per-request `max_completion_tokens ∈ [131072, 262144]`, random batches 1000–4000 every 5–20 min, `host_kv=768 GB`).

Crashed at ~1 h 5 min with:
```
[WATERMARK] Rank 0: Decode interrupted - putting 605 sequences ON_HOLD
ValueError: Invalid status transition from QUEUEING to ON_HOLD for sequence stress-b0-1708
  sequence.py:219  status_transition
  batchgen_worker.py:3919  _put_sequences_on_hold
  batchgen_worker.py:8007  (_decode_handle_boundary → watermark branch)
  legacy_adapter.py:315   decoding_continuous
  decode/scheduler.py:280 DecodeScheduler.run_continuous
```
Before the crash: `completed=965/2099` on batch 0; three more batches queued.

### Root cause (confirmed by log trace)
Two defects stack on the **legacy** `decoding_continuous` path that our adapter still wraps:

1. **Dual admission pollers.** `AdmissionCoordinator` (orchestrator-native) drains `self._admission_queue` as the outer loop's admission path. `_poll_admissions()` inside legacy `decoding_continuous` (`batchgen_worker.py:8018`) ALSO drains the same queue at every page boundary. Sequences admitted by the legacy poll don't go through `BatchFormation.tokenize / assign_ranks / build_query_book`; they land in `global_batch` as `QUEUEING`.
2. **Status-blind watermark eviction.** `_put_sequences_on_hold(decode_uuids)` at `batchgen_worker.py:3919` calls `global_batch.update_status(uuid, ON_HOLD)` on every uuid without checking current status. `_boundary_execute_decisions` inside `_page_boundary_fast` appends to `decode_uuids` via the async-load path; when those newly-added-but-still-QUEUEING uuids reach the watermark branch, the strict `status_transition` from `SequenceStatus.status_transition` (sequence.py:219 — introduced during Phase 1, invariant #9) correctly rejects the invalid jump and raises.

The ValueError is the Phase-1 invariant doing its job. The real bug is that legacy control flow (admission poll + watermark eviction) remains reachable through the adapter wrap. Phase 2.8 eliminates that.

### POIS's contract (from this session)
- "Do not take any short-cut. Plan a proper fix and do not call back any legacy code."
- The scheduler must compute page reservations **identically** to the allocator; no estimates, no drift. (Done in `80940c63` for prefill; the decode side already uses `get_gpu_pages_for_two_page_buffer` which the allocator also uses — keep it that way.)
- Legacy methods stay in `batchgen_worker.py` for **reference** only (analogous to the existing `_generate_legacy_bak`). No changes to legacy. No re-enabling any silent fallback (the seven Phase 2.5 raises stay as-is).

---

## 2. Phase 2.8 scope

**Goal:** `DecodeScheduler.run_continuous(uuids)` runs the full decode loop natively inside `batchgen/worker/`. No call to `LegacyInfraBackend.decoding_continuous`, no call to `_page_boundary_fast`, no call to `_put_sequences_on_hold`, no call to `_poll_admissions` from any path the orchestrator touches.

After Phase 2.8:
- `AdmissionCoordinator` is the **sole** admission path in the orchestrator's lifecycle.
- `BoundaryHandler.run(uuids)` is the sole boundary path; the skeleton at `batchgen/worker/boundary/` is fleshed out from the legacy phase helpers.
- Watermark eviction lives in `batchgen/worker/decode/eviction.py`. Its contract: only touches IN_DECODE sequences, raises on anything else. The Phase-1 state-machine invariant is enforced **by construction** in the native path — we never even try the invalid transition.
- Remove `decoding_continuous` from `LegacyInfraBackend` Protocol, `LegacyWorkerBackend`, `FakeLegacyBackend`. Rename legacy `BatchGenWorker.decoding_continuous` → `_decoding_continuous_legacy_bak` (reference-only, per the existing `_generate_legacy_bak` pattern).
- L4 stress can run for hours producing results; no `Invalid status transition`, no `GpuKvExhaustion`, no `AssertionError`.

### What is **in** scope
1. `batchgen/worker/boundary/` — the six phase helpers become native (see Stage 1).
2. `batchgen/worker/decode/` — the seven decode helpers + the continuous loop become native (Stage 2).
3. `batchgen/worker/decode/eviction.py` — new, native watermark eviction (Stage 3).
4. Adapter surgery — remove `decoding_continuous` and other control-flow legacy entry points (Stage 4).
5. Tests at each stage + a new integration test reproducing the L4 watermark-eviction crash signature (Stage 5).

### What is **out** of scope
- L1 / L2 / L3 re-validation — already green at HEAD; re-run only to confirm no regression.
- `reasoning_effort` for GPT-OSS — benchmark-config question, separate.
- Gloo `new_group` timeout (the Phase 2.6 item from an earlier plan). Native eviction uses `CollectiveBackend.barrier`, not gloo `new_group`; layer-2 is naturally sidestepped.
- Any edit to legacy methods. They are reference-only.
- Refactors beyond the port (no new features, no new abstractions, no new logging beyond what legacy had).

---

## 3. Stage-by-stage execution

### Conventions (keep these front of mind)
- Branch `tairan/worker-reextract`. No feature branches. No Co-Authored-By. `git commit -s`.
- Stage files by name (`git add <explicit list>`), never `git add -A`.
- Never use `--no-verify` or any hook-skip flag.
- `docker exec <container> pkill -9 python` for cleanup (never host-level).
- Servers run from `/tmp` on the remote to avoid `batchgen_kernels` path shadowing.
- Sync: `source /home/tairan/workspace/scripts/remote/env.sh && h20_n0_exec "cd /data2/tairan/workspace/BatchGen-reextract && git fetch origin tairan/worker-reextract && git reset --hard origin/tairan/worker-reextract && python -m pytest tests/unit/worker/ -x -q"` (+ analogous `h20_n1_exec` against `/data3/tairan/workspace/BatchGen-reextract` for Kimi 2-node runs).
- Each stage lands as its own commit so bisect stays sharp.
- Testing ladder after each stage before moving on: unit → L1 GPT-OSS → L1 Kimi → (L2 Kimi) → (L3 Kimi) → L4 Kimi stress.

### Stage 0 — Pre-flight (read before you code)
Read the skeleton you're filling in. These files already have docstrings, method signatures, and in some cases trace-replay tests. The port is **completing** them, not designing from scratch:
- `batchgen/worker/boundary/__init__.py` — `BoundaryHandler.run(uuids)` orchestration is already written and assumes the six phases exist. 173 LOC.
- `batchgen/worker/boundary/decisions.py` — sealed union `PageBoundaryDecision` already finalized. 171 LOC.
- `batchgen/worker/boundary/planner.py` — `BoundaryPlanner.plan(snapshot, gpu_free, host_free, has_pending)` signature exists; body is a stub. 243 LOC.
- `batchgen/worker/boundary/synchronizer.py` — `sync_metadata_in`, `broadcast_plan` defined. 68 LOC.
- `batchgen/worker/boundary/executor.py` — `BoundaryExecutor.apply(plan)` defined. 126 LOC.
- `batchgen/worker/boundary/guards.py` — `check_pre`, `check_post` defined. 174 LOC.
- `batchgen/worker/decode/scheduler.py` — `DecodeScheduler.run_continuous` currently delegates to `legacy.decoding_continuous`. 301 LOC.
- `batchgen/worker/decode/continuous_loop.py` — a 61-LOC CPU-fake `run_decode_interval`; this becomes the real loop's test target.

Legacy sources to port (all in `batchgen/batchgen_worker.py`):
| Helper | Line range | Approx LOC |
|---|---|---|
| `_page_boundary_fast` | 7336–7407 | 72 (orchestrator; the six phases are separate) |
| `_boundary_wait_pending` | 6637–6725 | 89 |
| `_boundary_gather_state` | 6726–6804 | 79 |
| `_boundary_merge_and_decide` | 6805–6911 | 107 |
| `_boundary_execute_decisions` | 6912–7131 | 220 |
| `_boundary_async_load` | 7132–7223 | 92 |
| `_boundary_finalize` | 7224–7335 | 112 |
| `_decode_bind_attn_wrapper` | 7838–7894 | 57 |
| `_decode_init_state` | 7896–7922 | 27 |
| `_decode_initial_moe_sync` | 7924–7940 | 17 |
| `_decode_handle_boundary` | 7941–8128 | 188 |
| `_decode_forward_step` | 8130–8376 | 247 |
| `_decode_update_sequences` | 8377–8448 | 72 |
| `_decode_cleanup` | 8449–8494 | 46 |
| `decoding_continuous` | 8495–8602 | 108 |
| `_put_sequences_on_hold` | 3868–3922 | 55 |
| `_poll_admissions` | 780–~870 | ~90 (skip — already handled by AdmissionCoordinator) |

Total to port: ~1,550 LOC of worker control flow, plus ~100 LOC of `_put_sequences_on_hold` whose status-blind body we replace with a strict native function.

Legacy invariants that MUST be preserved bit-for-bit:
- **Collective order.** Every `dist.all_gather_*`, `dist.broadcast`, `dist.barrier`, `dist.all_reduce` in the legacy helpers MUST happen in the same relative order in the native port; route them through `CollectiveBackend` (`batchgen/worker/protocols.py`). The `test_collective_ordering_fuzzer.py` property test catches regressions.
- **Page-table / slot ordering.** `slot_to_seq_id` on the GPU paged KV manager MUST match `AttnWrapperBase.cur_batch` at every decode forward entry. Legacy `_decode_bind_attn_wrapper` (line 7880) raises on mismatch — Phase 2.5 made that a hard `AssertionError` (`c9f2dcd2`). The port keeps the same check.
- **CUDA graph integrity.** `_decode_forward_step` chooses between graph launch and eager based on `self._cuda_graph_manager`; the graph buffer bindings must be set with the same `AttnWrapperBase` fields (`gpu_paged_kv_manager`, `cur_batch`, `scale`, `past_key_states`, `past_value_states`). Missed bindings surface as NaN logits in L1 GPT-OSS.
- **MoE EP (`PyNccl`) buffer sizing.** Legacy `_decode_initial_moe_sync` calls `parallel_manager.set_num_tokens_per_rank` and `set_rank_token_counts` based on `dist.all_gather_into_tensor` of per-rank batch sizes. Native port must do the same before the first forward in each decode phase (not per iteration).
- **State-machine transitions.** `SequenceEntry.status_transition` in `sequence.py:219` rejects invalid jumps. Every native mutation goes through `global_batch.update_status(...)`.
- **Hard-fail raises from Phase 2.5.** Seven raise sites already flip from "log + continue" to `raise`. Native code MUST NOT introduce any new `log + continue` error path. When in doubt, `raise`.

### Stage 1 — Native page boundary (`batchgen/worker/boundary/`)

**Goal:** `BoundaryHandler.run(uuids)` (already wired by the existing `__init__.py`) produces a `BoundaryPlan` and applies it, natively. No call into `_page_boundary_fast` or its six helpers.

#### 1a — `boundary/synchronizer.py`
Fill `BoundarySynchronizer.sync_metadata_in(uuids)` and `broadcast_plan(local_plan)`:
- Port lines **6726–6804** (`_boundary_gather_state`). The collective pattern: each rank builds a per-uuid payload dict; one `dist.all_gather_object(all_payloads, local_payload)` collects them. Route through `CollectiveBackend.all_gather_object`.
- Port the chunk-size extraction (`_get_effective_chunk_size`, already on adapter as `effective_chunk_size()`).
- `broadcast_plan`: `dist.broadcast_object_list([plan], src=0)` on rank 0; other ranks receive. Route via `CollectiveBackend.broadcast_object`.

Gotcha: the payload is a plain dict. Keep it small — legacy batches sequence metadata + completion flags + extension requests + free-page counts into one collective to avoid round-trips. Do the same.

#### 1b — `boundary/planner.py`
Fill `BoundaryPlanner.plan(snapshot, gpu_free, host_free, has_pending) -> BoundaryPlan`:
- Port lines **6805–6911** (`_boundary_merge_and_decide`). This is rank-0-only.
- Legacy returns a `decisions` list + global_seq_state + per_rank_free. Native packs into `BoundaryPlan` (sealed union already in `decisions.py`).
- Watermark trigger: same as legacy (`self._check_host_kv_watermark_trigger()` — exposed on adapter as `check_host_kv_watermark_trigger()`). When `has_pending and watermark_triggered`, the plan's break-to-prefill flag is set. The new native-boundary path surfaces this via `BoundaryPlan.watermark_break: bool`.
- Decisions: `Evict`, `OnHold`, `ExtendPages`, `ReleasePages`, `AsyncLoadHostToGpu` — all already defined in `decisions.py`. Port the logic that chooses among them.
- **Stage 1 watermark policy (the fix):** when producing `OnHold` decisions on the watermark path, include **only** uuids currently IN_DECODE. The planner has full state (per-uuid status in `snapshot`); filter there, not in the executor. QUEUEING and EVICTED never land in an `OnHold` decision.

#### 1c — `boundary/executor.py`
Fill `BoundaryExecutor.apply(plan)`:
- Port lines **6912–7131** (`_boundary_execute_decisions`). This is per-rank work.
- Use adapter primitives: `release_gpu_kv_pages(local_indices)`, `release_host_kv_pages_for_batch(uuids)`, `extend_gpu_kv_allocation(uuids)`. All already on `LegacyInfraBackend`.
- Canonical order: completions → evictions → on-hold → extensions → new loads. Legacy preserves this order; match it.
- Port lines **7132–7223** (`_boundary_async_load`) as a sub-step of executor (the new-loads path). This is where GPU page allocation for freshly-prefilled or reloaded sequences happens.
- Status mutations: `global_batch.update_status(uuid, new_status)`. The planner has ALREADY filtered to valid transitions, so this just lets the state machine catch bugs.

#### 1d — `boundary/guards.py`
Fill `check_pre(plan)` and `check_post()`:
- Port the inline asserts from `_page_boundary_fast` (lines 7370–7405).
- `check_pre`: every uuid in `plan` exists in `state.global_batch`; every `OnHold` target is IN_DECODE (sanity double-check on planner output).
- `check_post`: CTX invariant (every IN_DECODE uuid satisfies `current_context_length == original_prompt_length + decoded_length`); slot_to_seq_id matches cur_batch for the post-boundary batch.
- Raise `GuardViolation` on any failure. Do not `log + continue`.

#### 1e — `boundary/__init__.py`
No changes expected — the orchestration is already written. Smoke-test the Handler against fakes.

#### 1f — `boundary/finalize.py` (new) + wire into Handler
Port lines **7224–7335** (`_boundary_finalize`). This is the last step: page-table rebuild, MoE buffer update (`parallel_manager.set_num_tokens_per_rank` / `set_rank_token_counts`), final `dist.barrier()`, and `watermark_triggered = self._check_host_kv_watermark_trigger()` return.

Thread the watermark bool into `BoundaryHandler.run`'s return (a new field on `BoundaryPlan`, or a tuple return — both work; match what `DecodeScheduler.run_continuous` needs).

#### 1g — Adapter expansion (new methods)
Some legacy reads/writes need adapter passthrough. All **read-only or atomic** — no control flow in these:
- `set_num_tokens_per_rank(n: int)` and `set_rank_token_counts(t: torch.Tensor)` → `parallel_manager.*`.
- `rebuild_page_table(global_ids: list[int])` — already exposed as `rebuild_page_table_for_batch`; make sure the signature matches. Legacy call site passes global IDs directly; check the wiring.
- `has_cuda_graph_manager() -> bool` — if you need a cheap flag. Likely unnecessary if the forward step is driven by a separate adapter method.

Add to Protocol + `LegacyWorkerBackend` + `FakeLegacyBackend`. Keep the "explicit list, no `__getattr__`" discipline.

#### 1h — Tests
- `tests/unit/worker/boundary/` already contains `test_executor.py` and others. Extend or add:
  - `test_planner.py`: feed a synthetic snapshot with a mix of IN_DECODE + QUEUEING sequences; assert the plan's `OnHold` decisions contain ONLY the IN_DECODE ones.
  - `test_synchronizer.py`: fake collective; verify the single `all_gather_object` + single `broadcast_object` call counts.
  - `test_executor.py`: extend to cover the async-load path.
  - `test_guards.py`: CTX invariant, slot ordering, raises on any violation.
  - `test_boundary_handler.py`: end-to-end handler run on fakes.

#### 1i — Stage 1 gate
- `pytest tests/unit/worker/` stays ≥ 376 green (+ new boundary tests).
- `pytest tests/property/worker/test_page_table_order_fuzzer.py` green.
- Commit: `feat(phase-2.8.1): native page boundary (no more _page_boundary_fast callback)`.

### Stage 2 — Native decode helpers + loop (`batchgen/worker/decode/`)

**Goal:** `DecodeScheduler.run_continuous(uuids)` runs the full loop natively — forward, sample, update, boundary — with zero call to `legacy.decoding_continuous`. Each decode helper is a module.

#### 2a — `decode/bind.py`
Port lines **7838–7894** (`_decode_bind_attn_wrapper`). Wires `AttnWrapperBase` and `Attn_Wrapper` class-level singletons:
- `gpu_paged_kv_manager` (plus `_aux` for dual-manager models)
- `host_paged_kv_worker_view`
- `cur_batch` (global IDs translated from local indices)
- `scale`, `past_key_states`, `past_value_states` — threaded from the call site

The function returns `(gpu_manager, worker_view)` (same return shape as legacy) so the loop can reuse them.

New adapter passthroughs needed:
- `attn_wrapper_bind(gpu_manager, worker_view, cur_batch, scale, past_k, past_v)` — or a single `bind_decode_context(...)` method. Legacy `Attn_Wrapper` / `AttnWrapperBase` are class-level singletons; binding is `Class.field = value` — genuinely infrastructure, not control flow. OK to expose as one adapter call.

Keep the Phase 2.5 `AssertionError` on page-table-order mismatch.

#### 2b — `decode/init_state.py`
Port lines **7896–7922** (`_decode_init_state`). Initializes `_pending_kv_append_tasks`, `_pending_kv_append_tensors`, rebuilds `_sequences_with_gpu_kv` from the batch, seeds cumulative counters. Returns `(local_iteration=0, last_boundary=0, global_batch_size)`.

Native form: a `DecodeState` dataclass in `decode/state.py` (new) that holds:
```python
@dataclass
class DecodeState:
    local_iteration: int = 0
    last_boundary: int = 0
    global_batch_size: int = 0
    pending_async_task: object | None = None
    pending_load_uuids: list[UUID] = field(default_factory=list)
    pending_load_local: list[int] = field(default_factory=list)
    pending_load_global: list[int] = field(default_factory=list)
    decode_uuids: list[UUID] = field(default_factory=list)
    batch: list[int] = field(default_factory=list)
    new_tokens: torch.Tensor | None = None
    page_table_verified: bool = True
```

The loop mutates this object; helpers take it explicitly. No `self._`-carrying hidden state on the scheduler.

#### 2c — `decode/moe_sync.py`
Port lines **7924–7940** (`_decode_initial_moe_sync`). One `all_gather_into_tensor` (route through `CollectiveBackend.all_gather_into_tensor`) + two `parallel_manager.set_*` adapter calls.

#### 2d — `decode/forward_step.py`
Port lines **8130–8376** (`_decode_forward_step`). This is the biggest piece:
- CUDA graph launch branch (`self._cuda_graph_manager.run(...)`).
- Eager branch (`self.model(input_ids, ...)`).
- Token selection: `self._select_tokens(logits)`.
- Layer-by-layer KV write-back into host via `_append_decode_kv_to_host_async` (already usable via adapter `flush_deferred_kv_to_host` + `_pending_kv_append_tasks`).

This one must stay faithful: timing metadata, per-layer callbacks, graph invalidation on batch change. Port line by line, swap `self.<method>` → `adapter.<method>` where the method is infrastructure. The only control-flow decisions inside — which branch to take based on batch change or graph availability — become explicit branches in the native function.

Adapter methods needed (some already exist):
- `forward_decode_step(batch: list[int], new_tokens: torch.Tensor, page_table_verified: bool) -> torch.Tensor` returning new tokens post-sample. This is a **narrow** infrastructure wrapper — it calls the model, not the loop. It's OK for it to touch CUDA-graph internals because those are infrastructure. Put the minimum control flow (graph vs eager selection) on the adapter side.

Alternative: keep legacy `_decode_forward_step` as an adapter passthrough (`forward_decode_step`) — it's a single step, no admission polling, no status mutations, no watermark. Per POIS's rule this would be re-using legacy infrastructure, which is permitted; only control flow must move.

**Recommendation:** expose `adapter.forward_decode_step(batch, new_tokens, page_table_verified, local_iteration) -> torch.Tensor`. Keep the legacy implementation intact (it's infrastructure-heavy). That avoids porting the CUDA graph / layer-callback plumbing, which is high-risk for correctness. POIS's concern was control flow (admission, watermark) leaking into legacy — not the step-level forward pass.

#### 2e — `decode/update_sequences.py`
Port lines **8377–8448** (`_decode_update_sequences`). Writes tokens to `decoded_tokens` buffers, advances `decoded_length`, checks EOS via `adapter.should_stop_at_eos(token_id)`, triggers repetition detection via the existing server-side detector (already on legacy — leave it in place as infrastructure; native calls the adapter helper).

Reuse `batchgen/worker/completion.CompletionHandler` where it overlaps. The existing handler has `check_and_handle(uuids)` which gathers + reports completions; `update_sequences` is the per-step EOS/rep check before the gather.

#### 2f — `decode/handle_boundary.py`
Port lines **7941–8128** (`_decode_handle_boundary`) — **BUT**:
- Call `BoundaryHandler.run(state.decode_uuids)` from Stage 1 instead of `_page_boundary_fast`.
- Use native `put_on_hold` from Stage 3 instead of `_put_sequences_on_hold`.
- **DELETE** the `_poll_admissions` call (lines 8014–8037). Admission is owned by `AdmissionCoordinator` in the outer loop; the inner boundary never polls.
- Port the post-boundary page-table verification (lines 7978–7988): on any mismatch, `raise AssertionError` (per Phase 2.5).
- Port the "empty decode_uuids + pending async load" tail (lines 8085–8116).

Output: same `(decode_state, should_break, should_continue)` shape — updated via the `DecodeState` dataclass.

#### 2g — `decode/cleanup.py`
Port lines **8449–8494** (`_decode_cleanup`). Wait pending KV appends, unbind attn wrapper fields (`Attn_Wrapper.cur_batch = None`, etc.), disable decode watchdog, emit the final summary log line. Pure teardown.

#### 2h — `decode/continuous_loop.py` (rewrite)
Replace the current CPU-fake `run_decode_interval` with the real loop:

```python
def run_decode_interval(state, adapter, collectives, boundary_handler,
                        completion_handler, decision_frequency_pages,
                        decode_state: DecodeState) -> DecodeStepResult:
    # Top of loop: invariant checks
    while decode_state.decode_uuids:
        decode_state.local_iteration += 1
        adapter.feed_watchdog()
        adapter.feed_decode_watchdog()

        if decode_state.local_iteration - decode_state.last_boundary >= decision_frequency_pages * PAGE_SIZE:
            decode_state.last_boundary = decode_state.local_iteration
            handle_boundary(decode_state, adapter, collectives,
                            boundary_handler, completion_handler)
            if decode_state.should_break:
                break
            if decode_state.should_continue:
                continue

        # Forward step (infrastructure via adapter)
        decode_state.new_tokens = adapter.forward_decode_step(
            decode_state.batch, decode_state.new_tokens,
            decode_state.page_table_verified, decode_state.local_iteration,
        )
        adapter.flush_deferred_kv_to_host()

        new_tokens_cpu = decode_state.new_tokens.cpu()
        batch_seqs = [state.global_batch.get_sequence(adapter.local_to_uuid_map()[idx]) for idx in decode_state.batch]
        update_sequences(state, adapter, decode_state.batch, batch_seqs,
                         new_tokens_cpu, decode_state.local_iteration)

    return DecodeStepResult(
        tokens_produced=decode_state.local_iteration,
        uuids_decoded=tuple(decode_state.decode_uuids),
    )
```

(Schematic — real code will need to port a few per-iteration invariant assertions from legacy lines 8547–8597.)

#### 2i — `decode/scheduler.py` rewrite
`DecodeScheduler.run_continuous(uuids)` becomes:

```python
def run_continuous(self, uuids):
    # Bind attn wrapper (native), init state, initial MoE sync
    gpu_manager, worker_view = bind_attn_wrapper(self._state, self._legacy, list(uuids))
    decode_state = init_decode_state(self._state, self._legacy, list(uuids))
    decode_state.new_tokens = self._legacy.rebuild_input_tokens(decode_state.batch)
    initial_moe_sync(self._state, self._legacy, self._collectives, decode_state.batch)

    self._legacy.enable_decode_watchdog()
    try:
        result = run_decode_interval(
            self._state, self._legacy, self._collectives,
            self._boundary, self._completion,
            decision_frequency_pages=self._decision_frequency_pages,
            decode_state=decode_state,
        )
    finally:
        decode_cleanup(self._state, self._legacy, decode_state)

    return result
```

The old `self._legacy.decoding_continuous(list(uuids))` branch is gone.

#### 2j — Tests
`tests/unit/worker/decode/` (new directory):
- `test_bind.py` — AttnWrapperBase binding, order-mismatch raise
- `test_init_state.py` — DecodeState constructor + seeds
- `test_moe_sync.py` — fake collectives, verify the single all_gather
- `test_forward_step.py` — fake adapter, verify one step advances state
- `test_update_sequences.py` — EOS, length, repetition, full gather
- `test_handle_boundary.py` — calls BoundaryHandler (stubbed), routes break/continue correctly, asserts `_poll_admissions` is NOT called on the fake adapter
- `test_cleanup.py` — teardown fields are reset
- `test_continuous_loop.py` — full loop, multiple iterations, boundary every N pages

#### 2k — Stage 2 gate
- Unit tests: ≥ 376 + ~50 new cases.
- Property fuzzers green (`test_collective_ordering_fuzzer.py`, `test_page_table_order_fuzzer.py`, eviction-reentry fuzzer).
- Commit: `feat(phase-2.8.2): native decode loop + helpers (no more legacy decoding_continuous callback)`.

### Stage 3 — Native eviction

**Goal:** `batchgen/worker/decode/eviction.py` — native `put_on_hold(state, uuids, adapter, collectives)` with the strict IN_DECODE contract.

#### 3a — Implementation
```python
# batchgen/worker/decode/eviction.py
from __future__ import annotations

from batchgen.sequence import SequenceStatus
from batchgen.worker.protocols import UUID, CollectiveBackend, LegacyInfraBackend
from batchgen.worker.state import WorkerState


def put_on_hold(
    state: WorkerState,
    uuids: list[UUID],
    adapter: LegacyInfraBackend,
    collectives: CollectiveBackend,
) -> None:
    """Watermark-triggered eviction of IN_DECODE sequences to ON_HOLD.

    Contract (enforced by `raise` below, never by silent filter):
      - Every uuid MUST be IN_DECODE at call time.
      - GPU KV pages are released; host KV pages stay (so the sequence
        can be reloaded at the next prefill-to-decode transition).
      - The sequence stays in `global_batch`; only status transitions.
      - A final `CollectiveBackend.barrier()` keeps ranks in lockstep.

    The Phase-1 state-machine invariant (`status_transition`) catches
    any upstream bug that sends a non-IN_DECODE uuid here. Never
    silently filter; the raise is the contract.
    """
    if not uuids:
        return

    # 1. Strict precondition
    for uuid in uuids:
        seq = state.global_batch.get_sequence(uuid)
        if seq is None:
            raise KeyError(f"put_on_hold: uuid {uuid!r} not in global_batch")
        if seq.status != SequenceStatus.IN_DECODE:
            raise ValueError(
                f"put_on_hold: uuid {uuid!r} is {seq.status.name}, "
                f"expected IN_DECODE. Eviction only targets IN_DECODE "
                f"sequences; the upstream caller must filter before "
                f"calling put_on_hold."
            )

    # 2. Release GPU KV (rank-local)
    local_indices = adapter.get_local_indices_for_uuids(uuids)
    if local_indices:
        adapter.release_gpu_kv_pages(local_indices)

    # 3. Transition IN_DECODE → ON_HOLD
    for uuid in uuids:
        state.global_batch.update_status(uuid, SequenceStatus.ON_HOLD)

    # 4. Cross-rank sync
    collectives.barrier()
```

#### 3b — Wire-in
`decode/handle_boundary.py` (from Stage 2f) calls `put_on_hold(state, ONLY_IN_DECODE_subset, adapter, collectives)` when the boundary plan sets `watermark_break=True`. The "only IN_DECODE" filter lives in the **planner** (Stage 1b), so by the time we reach `put_on_hold` the list is already clean — the precondition is a belt-and-braces check for upstream bugs.

#### 3c — Tests
`tests/unit/worker/decode/test_eviction.py`:
- Happy path: all uuids IN_DECODE → GPU release + status transition + barrier.
- Raises on any uuid in PREFILLED / IN_PREFILL / QUEUEING / ON_HOLD / COMPLETED / EVICTED.
- Raises on unknown uuid.
- Empty list is a no-op.

#### 3d — Remove the old adapter entry
`LegacyInfraBackend.put_sequences_on_hold` and its backing in `LegacyWorkerBackend` and `FakeLegacyBackend` stay for now (Stage 4 removes them), BUT the native path never calls them.

#### 3e — Stage 3 gate
- Unit tests: ≥ 376 + ~6 new eviction cases.
- Commit: `feat(phase-2.8.3): native put_on_hold (strict IN_DECODE invariant)`.

### Stage 4 — Adapter surgery & legacy archival

**Goal:** remove the legacy callbacks from the Protocol, rename the legacy methods to `_legacy_bak`.

#### 4a — Protocol subtraction
Edit `batchgen/worker/protocols.py`:
- Remove `decoding_continuous` from `LegacyInfraBackend`.
- Remove `put_sequences_on_hold` (now only the native one is used).
- Remove `poll_admission_queue_nowait` if no native caller remains (double-check `AdmissionCoordinator` — it uses `admission_queue.get_nowait()` directly, so it doesn't need the adapter method).
- Remove `admit_sequences_from_message` (native AdmissionCoordinator handles it).

Update `LegacyWorkerBackend` (delete method bodies) and `FakeLegacyBackend` (delete method stubs) accordingly.

#### 4b — Legacy archival
In `batchgen/batchgen_worker.py`, rename with `_legacy_bak` suffix:
- `decoding_continuous` → `_decoding_continuous_legacy_bak`
- `_decode_bind_attn_wrapper` → `_decode_bind_attn_wrapper_legacy_bak`
- (all seven `_decode_*` helpers similarly)
- `_page_boundary_fast` → `_page_boundary_fast_legacy_bak`
- (all six `_boundary_*` helpers similarly)
- `_put_sequences_on_hold` → `_put_sequences_on_hold_legacy_bak`

Leave bodies intact (pattern: `_generate_legacy_bak` from Phase 2 F8). These are reference-only.

Search the codebase post-rename (`grep -rn "self\.\(_page_boundary_fast\|decoding_continuous\|_decode_bind_\|_decode_init_state\|_decode_initial_moe_sync\|_decode_handle_boundary\|_decode_forward_step\|_decode_update_sequences\|_decode_cleanup\|_put_sequences_on_hold\)\b" batchgen/`) — every remaining reference must be in `_legacy_bak`-suffixed code paths (unreachable) or in a `_legacy_bak` function's body.

#### 4c — `DecodeScheduler` cleanup
- Remove the `if self._legacy is not None: self._legacy.decoding_continuous(list(uuids))` early-return branch in `run_continuous` — gone.
- Remove `_decode_delegate` attribute fully (not just set to None).
- Remove any remaining `decoding_continuous` references.

#### 4d — Tests
- Remove the `FakeLegacyBackend.decoding_continuous` stub.
- Update any test that asserts adapter calls to `decoding_continuous` — they should now assert the native loop was called (e.g., `adapter.forward_decode_step` was recorded N times, or `BoundaryHandler.run` was called).

#### 4e — Grep gate
```bash
grep -rn "self\._legacy\.decoding_continuous\|_decode_delegate\|_put_sequences_on_hold(" batchgen/worker/
# expect: zero hits
grep -rn "decoding_continuous\|_decode_bind_\|_decode_init_state\|_decode_initial_moe_sync\|_decode_handle_boundary\|_decode_forward_step\|_decode_update_sequences\|_decode_cleanup\|_page_boundary_fast\|_boundary_wait_pending\|_boundary_gather_state\|_boundary_merge_and_decide\|_boundary_execute_decisions\|_boundary_async_load\|_boundary_finalize\|_put_sequences_on_hold" batchgen/worker/
# expect: zero hits (all native callers use the new modules)
```

#### 4f — Stage 4 gate
- Unit tests still ≥ 376 + new cases.
- Commit: `feat(phase-2.8.4): remove decoding_continuous from adapter + archive legacy helpers as _legacy_bak`.

### Stage 5 — Validation ladder + L4

Run after each earlier stage, and fully after Stage 4.

1. **CPU tests** — `h20_n0_exec "cd /data2/tairan/workspace/BatchGen-reextract && python -m pytest tests/unit/worker/ tests/property/worker/ -x -q"`. Must stay green.
2. **GPT-OSS L1** — `h20_n0_host "docker exec -d tairan-batchgen bash -c 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate batchgen && cd /tmp && bash /data2/tairan/workspace/batchgen-benchmark/models/gpt-oss/server.sh 128'"` (wait ~5 min for Uvicorn ready) → `bash /data2/tairan/workspace/batchgen-benchmark/models/gpt-oss/mmlu_client.sh 32 512`. Expect `Accuracy: 65.62 %`.
3. **Kimi L1** — 2-node start (`server.sh 1 768` on wechat_96 first, then `server.sh 0 768` on wechat_87). Wait for Uvicorn (~15 min for 768 GB host KV register). `mmlu_client.sh 32 512` → 18.75 %.
4. **Kimi L2** — `mmlu_client.sh 128 4096` → 85.16 %.
5. **Kimi L3** — `mmlu_client.sh 2048 10240` (~30 min). Expect 80.62 % ± noise.
6. **Kimi L4 stress** — `stress_client.sh` (already patched for `--max-decode 262144` + per-request `randint(131072, 262144)`). Expectations:
   - `stress_batch_0_results.jsonl` appears within ~1 h.
   - Log ratio: `grep -c "_prefill_flush_and_reconfigure" server_log` ≤ (number of completed prefill phases) × 8, not per-round. (Phase 2.7 metric; keep it.)
   - `grep -c "_legacy_bak" server_log` == 0 (no legacy call-through).
   - Zero `Invalid status transition`, zero `GpuKvExhaustion`, zero `AssertionError`, zero TCPStore timeout.
   - Multi-batch sustain ≥ 4 h without hang.

A new integration test reproduces the L4 crash signature in CPU-land:
- `tests/integration/worker/test_watermark_eviction.py`: admit mixed batch with some QUEUEING that haven't finished prefill; trigger watermark via fake host-KV backend; assert `put_on_hold` raises a clean `ValueError` when handed a QUEUEING uuid AND the planner correctly filters so the raise is NEVER reached in normal operation.

---

## 4. Critical file map (what you'll create / modify)

### New modules
- `batchgen/worker/decode/bind.py`
- `batchgen/worker/decode/init_state.py`
- `batchgen/worker/decode/state.py` (dataclass `DecodeState`)
- `batchgen/worker/decode/moe_sync.py`
- `batchgen/worker/decode/forward_step.py` (thin — delegates to `adapter.forward_decode_step`)
- `batchgen/worker/decode/update_sequences.py`
- `batchgen/worker/decode/handle_boundary.py`
- `batchgen/worker/decode/cleanup.py`
- `batchgen/worker/decode/eviction.py` (Stage 3)
- `batchgen/worker/boundary/finalize.py` (Stage 1f)
- `tests/unit/worker/decode/test_*.py` (one per module above)
- `tests/unit/worker/boundary/test_planner.py`, `test_synchronizer.py`, `test_guards.py`, `test_boundary_handler.py`
- `tests/integration/worker/test_watermark_eviction.py`

### Modified modules
- `batchgen/worker/decode/continuous_loop.py` — rewrite from CPU-fake to real native loop.
- `batchgen/worker/decode/scheduler.py` — `run_continuous` calls native helpers; remove adapter delegation branch; drop `_decode_delegate`.
- `batchgen/worker/boundary/planner.py`, `synchronizer.py`, `executor.py`, `guards.py` — fill in bodies.
- `batchgen/worker/backends/legacy_adapter.py` — add `bind_decode_context` / `forward_decode_step` / `set_num_tokens_per_rank` / `set_rank_token_counts` passthroughs; remove `decoding_continuous`, `put_sequences_on_hold`, stale admission wrappers in Stage 4.
- `batchgen/worker/protocols.py` — Protocol shrinks in Stage 4.
- `tests/unit/worker/fakes.py` — add new passthroughs, remove old ones.
- `batchgen/batchgen_worker.py` — rename (only renames, no body edits) with `_legacy_bak` suffix.

---

## 5. Known hazards (learned from earlier sessions)

- **Stale references cached at construction.** See Phase 2.5 `78ce3f29`: `TorchGpuKvBackend._m` cached `None`. Anywhere you hold a manager reference, resolve it **lazily** via getter.
- **Decode setup must re-run after prefill.** See Phase 2.5 `888ea9e7` and `41cdb807`: `_prefill_flush_and_reconfigure` frees the decode model + destroys GPU KV; `decode_setup_once` needs to re-build them. The existing `_decode_setup_done` flag on `LegacyWorkerBackend` handles this; keep it working.
- **Page reservation must match allocator exactly.** See Phase 2.7 `80940c63`: the scheduler used `ceil((prompt + max_decode) / PAGE_SIZE)` while the allocator used `max(prompt + chunk_size, prompt + INITIAL_GPU_PAGE_BUFFER*PAGE_SIZE)` capped at `kv_token_budget`. Always use `SequenceEntry.get_host_pages_for_initial_chunk(chunk_size)` / `get_gpu_pages_for_two_page_buffer()` — never re-derive.
- **Flush-and-reconfigure amortizes per phase, not per round.** Phase 2.7 `c4b191a4`. Carry this rule into the native port — each phase-level state transition (decode↔prefill) fires once; no per-round flush.
- **Zero-value caps mean "unlimited", not "cap at zero".** Phase 2.7 `b4ef7a03`. If `MoE_decoding_micro_batch_size == 0`, fall back to `attn_decoding_micro_batch_size`; if that's also 0, treat as uncapped.
- **`AttnWrapperBase` bindings are class-level singletons.** Every field you bind on entry must be unbound on exit (see `cleanup.py`). Missed unbinds → stale pointers the next phase trips over.
- **MoE EP (PyNccl) init is truly once-per-process.** The `_pynccl_initialized` flag on `LegacyWorkerBackend` is the guard. Native port must not re-trigger the init at every decode phase.
- **Kimi's `_gloo_migration_group` bug (Phase 2.6).** `dist.new_group(backend="gloo")` at `batchgen_worker.py:2241` times out because not all 16 ranks reach it together. Native path sidesteps because `put_on_hold` uses `CollectiveBackend.barrier`, not `new_group`. Do NOT accidentally reintroduce a legacy path that needs the gloo group.
- **Strict state machine (Phase 1 invariant #9).** `SequenceEntry.status_transition` rejects invalid jumps. Every status change goes through `global_batch.update_status(uuid, new_status)`. If the state machine raises, the upstream is wrong — do not soften the transition table.

---

## 6. Decision log for the implementer

Questions the port will surface. Record the answer in the plan file or a new ADR when you make the call:

1. **`forward_decode_step` on the adapter vs native.** I recommend keeping it on the adapter (legacy `_decode_forward_step` is infrastructure-heavy: CUDA graph, layer callbacks). POIS's directive is about control flow (admission, watermark, state transitions), not per-step forward passes. If POIS disagrees, port the forward step into `decode/forward_step.py` — budget +400 LOC and non-trivial CUDA-graph handling.
2. **`CompletionHandler` vs native `update_sequences`.** The existing `CompletionHandler.check_and_handle(uuids)` gathers completions across ranks and reports to the sink. `update_sequences` is the per-step write-back + EOS check. They overlap at the EOS detection call. Reuse `adapter.should_stop_at_eos(token_id)` in `update_sequences`; let `CompletionHandler` handle the final gather/report after the loop exits.
3. **Per-step admission polling.** Legacy polls admissions at every page boundary. `AdmissionCoordinator` polls only at the top of `generate_persistent`'s outer loop. That means a batch submitted mid-decode waits up to one full decode phase before being prefilled — acceptable latency penalty for eliminating the dual-poller bug. Alternative: have the native `handle_boundary` break out when `AdmissionCoordinator.has_pending()` is true, letting `generate_persistent` loop back to admission. Pick the first option (simpler) unless L4 shows unacceptable latency.
4. **Testing the CUDA-graph path in CPU.** Can't. Unit tests cover the control flow (state transitions, decisions, collective call order) with fakes. The CUDA-graph behavior is validated only by L1 GPT-OSS smoke. If L1 GPT-OSS fails, bisect which helper drift caused it.

---

## 7. Commit plan (chronological)

- `feat(phase-2.8.0): scaffold native decode/boundary module skeletons` (DecodeState dataclass, empty modules, imports) — optional, fine to merge into later commits.
- `feat(phase-2.8.1a-e): native page boundary (synchronizer + planner + executor + guards + finalize)` — can be one commit or five.
- `feat(phase-2.8.1f): BoundaryPlanner filters OnHold to IN_DECODE only` — if split from 1b.
- `feat(phase-2.8.2): native decode loop + helpers (no more legacy decoding_continuous callback)`.
- `feat(phase-2.8.3): native put_on_hold (strict IN_DECODE invariant)`.
- `feat(phase-2.8.4): remove decoding_continuous from adapter + archive legacy helpers as _legacy_bak`.
- `test(phase-2.8): integration test reproducing L4 watermark-eviction crash signature`.
- (Optional) `fix(phase-2.8): <anything surfaced during L1/L2/L3/L4 validation>` — per-fix commits.

---

## 8. Summary handoff

**Where to resume.** HEAD is `b4ef7a03`. All CPU tests green. Ladder validated up through Kimi L3 (80.62 %). L4 crashes at the first watermark eviction because legacy `decoding_continuous` still runs through the adapter and its status-blind `_put_sequences_on_hold` trips the Phase-1 strict state machine.

**What to do next.** Execute Stages 1 → 2 → 3 → 4 in order, running the validation ladder after each. The scope is real (~1,500 LOC moved + ~600 LOC new tests + ~100 LOC deleted) and the NCCL / CUDA-graph invariants demand careful porting. Follow the legacy line-by-line; keep the collective order the same; expose per-step forward as adapter (POIS's control-flow rule is about admission/watermark, not GPU primitives).

**Done criteria.** L4 stress runs clean for ≥ 4 h with `stress_batch_3+` completing, zero legacy callbacks in the server log (`grep -c _legacy_bak` on server log == 0), and every earlier benchmark stays at the HEAD baseline.

**If you get stuck.** Fall back to Stage 3 alone as a minimal shippable increment — it adds the native eviction module + strict invariant tests but does NOT yet replace the adapter wrap. L4 still crashes in that case, but the native invariant is in place for Stages 2 + 4 to wire in. Commit Stage 3 standalone is reviewable even without the others landing.
