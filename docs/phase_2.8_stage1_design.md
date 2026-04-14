# Phase 2.8 Stage 1 — Native Page Boundary — Design Addendum

**Status**: Design only. No code changes in the commit that lands this doc.
**Supersedes the Stage 1 sub-steps in**: `deep-bubbling-flamingo.md §3 Stage 1`.
**Why this addendum**: the Stage 1 section of the master plan assumed the
`batchgen/worker/boundary/` modules were "signatures + stubs". They are not —
they contain an M4-complete CPU-testable implementation whose decision schema
(sealed union: `ReleasePages | Evict | OnHold | ExtendPages | AsyncLoadHostToGpu`)
is incompatible with what the legacy `BoundaryDecisions` struct carries
(`host_growth_uuids`, `host_growth_pages`, `growth_feasible`, `host_evicted_uuids`,
`seqs_needing_extension`, `new_load_uuids`, `decode_uuids_final`). A faithful
port of `_page_boundary_fast` requires migrating that schema, not filling
empty stubs.

This addendum maps each legacy phase to the native module it becomes, fixes
signatures, lists the adapter methods the executor needs, describes the
decision-schema migration, enumerates test churn, and orders the sub-commits.

---

## 1. Current state vs target

### M4 skeleton (HEAD `b4ef7a03`, already in tree)

| Module | LOC | Status |
|---|---|---|
| `boundary/__init__.py` (`BoundaryHandler.run`) | 173 | Orchestration wired; 7-step loop. Calls sync → plan → broadcast → guards.pre → executor.apply → guards.post. |
| `boundary/decisions.py` (`BoundaryPlan`, sealed union) | 171 | Sealed union of 5 decision types. Missing: `HostGrow`, `HostEvict`, `AsyncLoadHostToGpu` (stubbed), watermark_break flag. |
| `boundary/synchronizer.py` | 68 | `sync_metadata_in` delegates to `SyncCoordinator.sync_metadata`. `broadcast_plan` via `CollectiveBackend.broadcast_object`. Missing: gather of free_pages + candidate_state + additional_pages_needed + host_growth fields. |
| `boundary/planner.py` | 243 | Rules 1–3: Release + OnHold(WATERMARK_TRIGGER) + per-seq Extend/OnHold(EXTENSION_FAILED). IN_DECODE-only OnHold filter is present (M4 `_onhold_all_in_decode:172`). Missing: host growth, host eviction, new-load selection, rank-aware extension arithmetic. |
| `boundary/executor.py` | 126 | Release/Evict(minimal)/OnHold/Extend wired through `KVCacheManager` + `HostKVRebalancer`. `_apply_async_load` raises `NotImplementedError`. Missing: host growth, host eviction, new-load async launch, worker_view interactions, seq-metadata scalar mutations. |
| `boundary/guards.py` | 174 | `check_pre` (live-seq ref) + `check_post` (CTX + index-map consistency). Missing: post-boundary page-table order verification. |
| `DecodeScheduler.run_continuous` | 301 | Production path delegates to `legacy.decoding_continuous`. BoundaryHandler runs only in CPU-test mode. |

### Legacy phases to replace (`batchgen/batchgen_worker.py`)

| Phase | Legacy helper | LOC | Maps to native module |
|---|---|---|---|
| 0 | `_boundary_wait_pending` | 6637–6724 | **NEW** `boundary/wait_pending.py` |
| 1 | `_boundary_gather_state` | 6726–6804 | `boundary/synchronizer.py` (extend) |
| 2–3 | `_boundary_merge_and_decide` | 6805–6911 | Split: metadata absorption → synchronizer; rank-0 decide → planner; broadcast → synchronizer |
| — | `_compute_boundary_decisions` | 6443–6629 | `boundary/planner.py` (rewrite) |
| 4 | `_boundary_execute_decisions` | 6912–7131 | `boundary/executor.py` (rewrite) |
| 4E | `_boundary_async_load` | 7132–7223 | `boundary/executor.py` async-load sub-step |
| 5 | `_boundary_finalize` | 7224–7335 | **NEW** `boundary/finalize.py` |
| — | `_put_sequences_on_hold` | 3868–3922 | **NEW** `decode/eviction.py` (Stage 3, not Stage 1) |
| — | `_page_boundary_fast` | 7336–7413 | `boundary/__init__.py` `BoundaryHandler.run` (extend return) |

Total legacy to port in Stage 1: ~1,100 LOC (Stage 3's `_put_sequences_on_hold` not counted here).

---

## 2. Decision-schema migration

### The problem

M4's sealed union models each decision as a distinct dataclass. Legacy's
`BoundaryDecisions` is a flat struct with 10 fields. These do not map 1:1:
- Legacy's `host_growth_uuids` + `host_growth_pages` + `growth_feasible` is
  one atomic unit (feasibility decides whether to apply any growth); the
  sealed union would split into N `HostGrow(uuid, pages)` decisions plus
  a separate feasibility flag.
- Legacy's `new_load_uuids` represents decisions that kick off an async
  operation returning a handle the caller must thread through the next
  boundary cycle; the sealed union's `AsyncLoadHostToGpu` elides the
  returned handle shape.
- Legacy's `decode_uuids_final` / `active_uuids` / `onhold_uuids` are
  outputs of the planner driving what `run_continuous` uses as the next
  iteration's batch; the sealed union currently has no output channel
  for "post-boundary active batch" — `BoundaryHandler.run` just returns
  the plan and the caller rebuilds the batch.

### The migration

Option A — **Extend the sealed union** (keep current M4 shape). Add three
new decision types and a plan-level flag:
- `HostGrow(uuids: tuple[UUID], pages: tuple[int], feasible: bool)`
- `HostEvict(uuids: tuple[UUID])`
- `NewLoadAsync(uuids: tuple[UUID], rank_pages: dict[int, int])` (return
  shape: the async handle lives on `BoundaryHandler.run`'s return tuple,
  not inside the decision — immutable plan stays immutable).
- `BoundaryPlan.watermark_break: bool` (surfaced from the watermark
  planner path; DecodeScheduler reads this to break-to-prefill).
- `BoundaryPlan.decode_uuids_final: tuple[UUID, ...]` (ordered output
  batch; saves callers from re-filtering).

Option B — **Replace with `BoundaryDecisions` struct** (mirror legacy).
Mechanically cleaner (one-to-one port), but loses type safety and requires
rewriting every existing planner/executor test. ~8 test files + ~15 tests.

**Recommendation: Option A.** Keeps the M4 tests mostly intact (they assert
decisions by isinstance type); new decision types are additive. Test churn
is writing tests for the new types, not rewriting existing ones.

### New sealed-union schema (Option A)

```python
# boundary/decisions.py — additions
@dataclass(frozen=True)
class HostGrow:
    """Rank-0 decision: grow host-KV allocation for N sequences.

    `feasible` is False when total growth > host_free - safety_margin;
    in that case the executor SKIPS the growth entirely (legacy behavior
    at line 6974). `feasible` is part of the decision, not the executor.
    """
    uuids: tuple[UUID, ...]
    pages: tuple[int, ...]       # parallel to uuids
    feasible: bool

@dataclass(frozen=True)
class HostEvict:
    """Rank-0 decision: evict N sequences from host KV to EVICTED.

    Executor: release host pages via worker_view, rebuild evicted_token_ids,
    transition IN_DECODE/ON_HOLD → EVICTED. Reentry math
    (`prompt_length = prompt_length + new_decoded_count`) is applied
    deterministically on every rank.
    """
    uuids: tuple[UUID, ...]

@dataclass(frozen=True)
class NewLoadAsync:
    """Rank-0 decision: async-load N host-resident sequences onto GPU.

    `rank_pages` is the per-rank page budget arithmetic the planner
    precomputed (no extra collective in executor).
    """
    uuids: tuple[UUID, ...]
    rank_pages: tuple[tuple[int, int], ...]  # (rank, pages_allocated)

PageBoundaryDecision = ReleasePages | Evict | OnHold | ExtendPages \
    | HostGrow | HostEvict | NewLoadAsync | AsyncLoadHostToGpu

@dataclass(frozen=True)
class BoundaryPlan:
    decisions: tuple[PageBoundaryDecision, ...] = field(default_factory=tuple)
    metadata_snapshot: dict[UUID, SeqMetadata] = field(default_factory=dict)
    decode_uuids_final: tuple[UUID, ...] = field(default_factory=tuple)
    watermark_break: bool = False
```

### Handler return shape

```python
@dataclass(frozen=True)
class BoundaryResult:
    plan: BoundaryPlan
    new_async_task: object | None        # the async handle to thread through
    new_load_uuids: tuple[UUID, ...]
    new_load_local: tuple[int, ...]
    new_load_global: tuple[int, ...]
    watermark_triggered: bool

def BoundaryHandler.run(self, decode_uuids, batch, gpu_manager,
                        pending_async_load_task, pending_load_uuids,
                        pending_load_local_indices, pending_load_global_ids
                       ) -> BoundaryResult: ...
```

The signature now matches legacy `_page_boundary_fast` input/output 1:1,
which is what lets DecodeScheduler swap the call out in Stage 2.

---

## 3. Module-by-module design

### 3.1 `boundary/wait_pending.py` (NEW)

**Replaces**: `_boundary_wait_pending` (6637–6724, 88 LOC).

**Public API**:
```python
def wait_pending(
    state: WorkerState,
    adapter: LegacyInfraBackend,
    collectives: CollectiveBackend,
    decode_uuids: list[UUID],
    batch: list[int],
    pending_async_task: object | None,
    pending_load_uuids: list[UUID],
    pending_load_local: list[int],
    pending_load_global: list[int],
) -> tuple[list[UUID], list[int]]:
    ...
```

**Responsibilities**:
1. `adapter.wait_pending_kv_append_tasks()` — drain deferred KV writes.
2. (debug-mode only) `collectives.all_gather_object(local_decode_set)`
   for desync detection, rank-0 authoritative repair. Not ported to
   production path; remains legacy-only. **Decision: skip porting**;
   the `BATCHGEN_CB_DEBUG` flag's desync repair is not an invariant of
   the native path (AdmissionCoordinator makes uuids rank-consistent).
3. If `pending_load_uuids` non-empty: `pending_async_task.wait()`,
   `torch.cuda.synchronize`, `collectives.barrier()`.
4. `adapter.finalize_async_load(pending_async_task, pending_load_uuids,
   pending_load_local, pending_load_global, decode_uuids, batch,
   gpu_manager)` → returns `(decode_uuids, batch)`.
5. `adapter.rebuild_page_table_for_batch(batch, gpu_manager)` +
   verify-and-fix (post-finalize slot-to-seq mismatch re-rebuild).

### 3.2 `boundary/synchronizer.py` (EXTEND)

**Replaces**: `_boundary_gather_state` (6726–6804, 79 LOC) +
metadata absorption from `_boundary_merge_and_decide` (6869–6888).

**Additions**:
```python
@dataclass(frozen=True)
class BoundaryPayload:
    free_pages: int
    seq_state: dict[UUID, SeqBoundaryState]
    candidate_state: dict[UUID, LoadCandidateState]

@dataclass(frozen=True)
class SeqBoundaryState:
    decoded_length: int
    current_context_length: int
    gpu_pages_allocated: int
    eos_reached: bool
    completed: bool
    additional_pages_needed: int
    assigned_rank: int
    needs_host_growth: bool
    host_growth_pages: int
    host_pages_allocated: int
    host_token_capacity: int
    prompt_length: int
    total_decoded_before_eviction: int

@dataclass(frozen=True)
class LoadCandidateState:
    pages_needed: int
    assigned_rank: int
    status: str                 # SequenceStatus name
    decoded_length: int

class BoundarySynchronizer:
    # existing
    def sync_metadata_in(self, uuids: list[UUID]) -> None: ...
    def broadcast_plan(self, plan: BoundaryPlan | None) -> BoundaryPlan: ...

    # NEW
    def gather_boundary_state(
        self, decode_uuids: list[UUID], adapter: LegacyInfraBackend,
    ) -> tuple[list[BoundaryPayload | None], int]:
        """Single `all_gather_object` of (free_pages, seq_state,
        candidate_state) from every rank. Returns all_payloads and the
        effective chunk_size. The metadata absorption step (update local
        SequenceEntry from non-owned ranks) is done inside
        `_absorb_cross_rank_metadata` after the gather."""

    def absorb_cross_rank_metadata(
        self, decode_uuids: list[UUID],
        all_payloads: list[BoundaryPayload | None],
    ) -> list[UUID]:
        """Walk all_payloads, update local SequenceEntry fields for
        non-owned sequences, handle `missing_uuids` orphan path
        (force-complete + remove from decode_uuids). Returns the
        possibly-pruned decode_uuids."""
```

**Collective discipline**: one `all_gather_object` in the whole Stage 1 path,
plus the existing `broadcast_plan`. Total: 2 collectives per boundary cycle,
matching legacy's "2–3 collectives" comment.

### 3.3 `boundary/planner.py` (REWRITE)

**Replaces**: `_compute_boundary_decisions` (6443–6629, 187 LOC).

**New signature**:
```python
def BoundaryPlanner.plan(
    self,
    snapshot: dict[UUID, SeqMetadata],
    *,
    global_seq_state: dict[UUID, SeqBoundaryState],
    global_candidate_info: dict[UUID, LoadCandidateState],
    per_rank_free: list[int],
    chunk_size: int,
    worker_view_stats: HostViewStats | None,     # host-KV free/total pages
    has_pending: bool,
    world_size: int,
    enable_host_kv_eviction: bool,
    host_kv_eviction_watermark: int,
) -> BoundaryPlan:
    ...
```

**Rules** (preserve legacy order exactly):
1. Completed sequence split (`completed_uuids` / `active_uuids` by
   `global_seq_state[uuid].completed`).
2. Host growth (`HostGrow(uuids, pages, feasible)`): sum `host_growth_pages`
   for active seqs; feasibility = `total_growth_needed <= host_free - 5%
   safety_margin`.
3. Host eviction (`HostEvict(uuids)`): when host-free < watermark, run
   `select_host_kv_eviction(candidates, target_free, SHORTEST_FIRST)`
   (existing helper in `continuous_batching.py`).
4. GPU extension + on-hold: rank-aware arithmetic
   (`total_additional_by_rank[r] <= per_rank_free[r]`). If any rank
   over-budget, sort that rank's seqs by `(priority, decoded_length,
   global_idx)` and emit `OnHold(EXTENSION_FAILED)` smallest-first until
   feasible. Remaining seqs → `ExtendPages`.
5. Watermark trigger (`OnHold(WATERMARK_TRIGGER)`): already implemented
   in M4 planner; keep the IN_DECODE-only filter. Surface
   `watermark_break=True` on the plan.
6. New-load selection (`NewLoadAsync`): from `global_candidate_info`,
   sort by `(-decoded_length, global_idx)`, fit into
   `per_rank_free[r] - actual_extension_by_rank[r]`.

**Critical**: the planner is still a pure function. Everything it reads
is on its arguments. No collective, no state mutation.

### 3.4 `boundary/executor.py` (REWRITE)

**Replaces**: `_boundary_execute_decisions` (6912–7131, 220 LOC) +
`_boundary_async_load` (7132–7223, 92 LOC).

**Executor now uses the adapter directly** for infrastructure, because
the existing `KVCacheManager` + `HostKVRebalancer` don't cover the legacy
surface (worker_view interactions, scalar seq metadata, evicted_token_ids
construction, async load launch).

**Canonical apply order** (unchanged from legacy):
1. ReleasePages (completed): `adapter.release_gpu_kv_pages(local_indices)`,
   `adapter.release_host_kv_pages_for_batch(uuids)`, zero out
   `seq.gpu_pages_allocated` / `host_pages_allocated` /
   `host_token_capacity`, `adapter.submit_completed(...)`, emit completion
   reports.
2. Active uuids update: `decode_uuids = decisions.active_uuids` (after
   Releases).
3. HostGrow: if feasible, loop and call
   `worker_view.grow_pages_for_sequences(host_grow_requests)`. Seq fields:
   `host_token_capacity += pages * PAGE_SIZE`, `host_pages_allocated += pages`.
4. HostEvict: per-uuid, local only: `release_gpu_kv_pages`,
   construct `evicted_token_ids = prompt_tokens + new_decoded`,
   `worker_view.release_sequence_pages` +
   `worker_view.unregister_sequences`. All ranks: reentry math
   (`prompt_length = prompt_length + new_decoded_count`,
   `current_context_length = new_reentry_len`, status EVICTED).
5. OnHold: per-uuid, local only: `gpu_manager.free_pages_for_sequences`.
   All ranks: `seq.gpu_pages_allocated = 0`, status ON_HOLD.
6. ExtendPages: per-uuid, local only: `adapter.extend_gpu_kv_allocation(
   my_remaining_ext)`. On failure: OOM → log warning + force those seqs
   to ON_HOLD.
7. NewLoadAsync: `_apply_new_load_async` sub-step. Rank-local: match uuid
   → local_idx for `assigned_rank == self.rank`, check actual free pages,
   filter down if over-budget, `gpu_manager.allocate_pages_for_sequences`,
   `gpu_manager.rebuild_page_table(new_load_global)`,
   `worker_view.async_load_layer_paged_kv_to_device(...)` → returns async
   handle. Pre-async state saved on `self._async_load_tensors` so the
   handle's `.wait()` can complete correctly next cycle.

**Return value of `apply`**: `(decode_uuids, batch, new_async_task,
new_load_uuids, new_load_local, new_load_global)`. Hand back to Handler.

### 3.5 `boundary/finalize.py` (NEW)

**Replaces**: `_boundary_finalize` (7224–7333, 110 LOC).

**Public API**:
```python
def finalize(
    state: WorkerState,
    adapter: LegacyInfraBackend,
    collectives: CollectiveBackend,
    decode_uuids: list[UUID], batch: list[int],
    gpu_manager: object,      # opaque; GPUPagedKVCacheManager
) -> bool:
    """
    Steps:
      1. adapter.rebuild_page_table_for_batch(batch, gpu_manager)
      2. Per-rank batch-size gather via all_gather_into_tensor.
      3. adapter.set_num_tokens_per_rank(max_batch_size)
         + adapter.set_rank_token_counts(tensor)
      4. collectives.barrier()
      5. Batch consistency verification (batch == expected_local?
         else repair + rebuild_page_table_for_batch).
      6. Final page-table shape verification against len(batch).
      7. Return adapter.check_host_kv_watermark_trigger().
    """
```

**Adapter methods required** (new):
- `set_num_tokens_per_rank(n: int)` — `self.parallel_manager.set_num_tokens_per_rank(n)`
- `set_rank_token_counts(counts: torch.Tensor)` — `self.parallel_manager.set_rank_token_counts(counts)`
- `rebuild_page_table_for_batch(batch: list[int], gpu_manager: object)` —
  already exists on some variant; verify signature.
- `check_host_kv_watermark_trigger() -> bool` — thin wrapper on
  `_check_host_kv_watermark_trigger`.

### 3.6 `boundary/guards.py` (EXTEND)

**Additions**:
- `check_post_page_table_order(batch, gpu_manager)`: raises
  `GuardViolation` if `slot_to_seq_id != local_indices_to_global_seq_ids(
  batch)`. Called inside `BoundaryHandler.run` after finalize.

### 3.7 `boundary/__init__.py` (REWIRE)

**`BoundaryHandler.run` new signature** (matching legacy
`_page_boundary_fast`):
```python
def run(
    self,
    decode_uuids: list[UUID],
    batch: list[int],
    gpu_manager: object,
    pending_async_task: object | None,
    pending_load_uuids: list[UUID],
    pending_load_local: list[int],
    pending_load_global: list[int],
) -> BoundaryResult:
    ...
```

**Flow**:
```
1. (decode_uuids, batch) = wait_pending(...)
   if empty: return BoundaryResult(empty_plan, None, ...)
2. sync_metadata_in(decode_uuids)
3. (all_payloads, chunk_size) = synchronizer.gather_boundary_state(
       decode_uuids, adapter)
4. decode_uuids = synchronizer.absorb_cross_rank_metadata(
       decode_uuids, all_payloads)
5. snapshot = _build_snapshot(decode_uuids)           # existing helper
6. if rank == 0:
       plan = planner.plan(snapshot, global_seq_state=..., ...)
   else:
       plan = None
   plan = synchronizer.broadcast_plan(plan)
7. guards.check_pre(plan)
8. (decode_uuids, batch, new_async_task, new_load_uuids,
    new_load_local, new_load_global) = executor.apply(plan)
   if empty decode_uuids: return BoundaryResult(plan, None, ...)
9. watermark_triggered = finalize(decode_uuids, batch, gpu_manager)
10. guards.check_post() + guards.check_post_page_table_order(
        batch, gpu_manager)
11. return BoundaryResult(plan, new_async_task, new_load_uuids,
        new_load_local, new_load_global, watermark_triggered)
```

### 3.8 `DecodeScheduler` wiring (Stage 2 foreshadow)

**Not in Stage 1's scope**, but noted here because the new `BoundaryResult`
shape is what Stage 2's native decode loop will consume. Stage 1 does NOT
change `DecodeScheduler.run_continuous`; it stays on the legacy
`decoding_continuous` delegation. The new `BoundaryHandler.run` is wired
up only through CPU unit tests until Stage 2.

---

## 4. Adapter surface additions

Add to `LegacyInfraBackend` Protocol + `LegacyWorkerBackend` (production
wrap) + `FakeLegacyBackend` (test fake). Discipline unchanged: explicit
list, no `__getattr__`.

| Method | Delegates to | Used by |
|---|---|---|
| `wait_pending_kv_append_tasks() -> int` | `self._w._wait_pending_kv_append_tasks()` | wait_pending |
| `finalize_async_load(task, uuids, local, global_, decode_uuids, batch, gpu_manager) -> tuple[list, list]` | `self._w._finalize_async_load_minimal(...)` | wait_pending |
| `rebuild_page_table_for_batch(batch, gpu_manager) -> None` | `self._w._rebuild_page_table_for_batch(batch, gpu_manager)` | wait_pending, finalize, executor |
| `release_gpu_kv_pages(local_indices) -> None` | `self._w._release_gpu_kv_pages(local_indices)` | executor |
| `release_host_kv_pages_for_batch(uuids) -> None` | `self._w._release_host_kv_pages_for_batch(uuids)` | executor |
| `extend_gpu_kv_allocation(uuids) -> bool` | `self._w._extend_gpu_kv_allocation(uuids)` | executor |
| `submit_completed_to_incremental_writer(uuids) -> None` | `self._w._submit_completed_to_incremental_writer(uuids)` | executor |
| `gather_completed_tokens(uuids) -> dict[UUID, str]` | `self._w._gather_completed_tokens(uuids)` | executor |
| `report_completion(uuid, gathered_text) -> None` | `self._w._report_completion(uuid, gathered_text=...)` | executor |
| `set_num_tokens_per_rank(n) -> None` | `self._w.parallel_manager.set_num_tokens_per_rank(n)` | finalize |
| `set_rank_token_counts(counts) -> None` | `self._w.parallel_manager.set_rank_token_counts(counts)` | finalize |
| `check_host_kv_watermark_trigger() -> bool` | `self._w._check_host_kv_watermark_trigger()` | finalize |
| `get_local_indices_for_uuids(uuids) -> list[int]` | `self._w._get_local_indices_for_uuids(uuids)` | executor (already exists? verify) |
| `local_indices_to_global_seq_ids(batch) -> list[int]` | `self._w._local_indices_to_global_seq_ids(batch)` | wait_pending, finalize |
| `host_paged_kv_worker_view()` | `getattr(self._w.core_engine, "host_paged_kv_worker_view", None)` | planner (rank-0 only, for host stats), executor |

**Collectives on the adapter**: NONE. Collectives go through
`CollectiveBackend`. Exception: legacy's `all_gather_into_tensor` in
finalize; we'll route that through `CollectiveBackend.all_gather_into_tensor`
which already exists (`sync.py`).

---

## 5. Test churn

### Existing tests that stay (planner rule 1-3 unchanged)
- `tests/unit/worker/boundary/test_executor.py` (existing) — extend with
  new decision types.
- `tests/unit/worker/boundary/test_planner_rules.py` (if exists) —
  keep for rule 1-3 coverage.

### Existing tests that get adjusted
- Any test constructing `BoundaryPlan(decisions=(...))` directly may need
  to add `decode_uuids_final=(...)` / `watermark_break=False`. ~5 tests.

### New tests required (~12 files, ~400 LOC)
- `tests/unit/worker/boundary/test_wait_pending.py` (Phase 0)
- `tests/unit/worker/boundary/test_synchronizer_gather.py` (new
  `gather_boundary_state` + `absorb_cross_rank_metadata`)
- `tests/unit/worker/boundary/test_planner_full.py` (all 6 legacy rules)
- `tests/unit/worker/boundary/test_executor_full.py` (all 8 decision
  types, async-load path, per-rank budgets, evicted_token_ids math)
- `tests/unit/worker/boundary/test_finalize.py`
- `tests/unit/worker/boundary/test_guards_post.py` (page-table order)
- `tests/unit/worker/boundary/test_handler_full.py` (end-to-end with
  fakes, 12-step flow)

### FakeLegacyBackend additions
~15 new method stubs (see §4 adapter table). Record call args for
assertion in tests.

---

## 6. Sub-commit ordering

Each bullet is an independent commit on `tairan/worker-reextract`.

1. **`feat(phase-2.8.1-0)`: Add this design addendum + update master plan pointer.**
   Just `docs/phase_2.8_stage1_design.md`. No code. (This commit.)

2. **`feat(phase-2.8.1-1a)`: Extend BoundarySynchronizer with
   `gather_boundary_state` + `absorb_cross_rank_metadata`.**
   - New: `BoundaryPayload`, `SeqBoundaryState`, `LoadCandidateState`.
   - Uses existing `CollectiveBackend.all_gather_object`.
   - Unit test: fake collective, mixed-rank payload, verify absorption +
     missing_uuids orphan path.
   - No changes to planner/executor yet.

3. **`feat(phase-2.8.1-1b)`: Extend BoundaryPlan with new decision types
   + watermark_break + decode_uuids_final.**
   - New: `HostGrow`, `HostEvict`, `NewLoadAsync` decisions.
   - Unit test: construct plan, verify `decisions_of(HostGrow)` etc.
   - Executor stubs for new types raise `NotImplementedError` — landed
     in a later commit.

4. **`feat(phase-2.8.1-1c)`: Add adapter passthroughs.**
   - Protocol additions + `LegacyWorkerBackend` bodies +
     `FakeLegacyBackend` stubs.
   - Unit test: fake records calls.
   - No executor / finalize use yet.

5. **`feat(phase-2.8.1-1d)`: Rewrite BoundaryPlanner with all 6 rules.**
   - Planner consumes the richer inputs and emits new decision types.
   - Unit test: all 6 rules covered.
   - Executor still raises `NotImplementedError` for new types.

6. **`feat(phase-2.8.1-1e)`: Rewrite BoundaryExecutor with adapter
   passthroughs + async-load sub-step.**
   - All 8 decision types handled.
   - Unit test: per-rank budgets, evicted_token_ids math, async-load
     returns handle.

7. **`feat(phase-2.8.1-1f)`: New boundary/wait_pending.py.**
   - Phase 0 port.
   - Unit test: pending task wait + finalize + rebuild.

8. **`feat(phase-2.8.1-1g)`: New boundary/finalize.py.**
   - Phase 5 port.
   - Unit test: finalize steps + watermark return.

9. **`feat(phase-2.8.1-1h)`: Extend guards.check_post with page-table
   order verification.**
   - Unit test: mismatch → GuardViolation.

10. **`feat(phase-2.8.1-1i)`: Rewire BoundaryHandler.run with full
    legacy-compatible signature + BoundaryResult return.**
    - End-to-end handler unit tests on fakes.
    - `pytest tests/unit/worker/ -x -q` stays green (≥376 + ~50 new).
    - Property fuzzers unchanged (no new collectives, only
      `all_gather_object` was previously implicit via `sync_metadata`).

### Stage 1 gate (after 1i lands)
- Unit: ≥ 376 + ~50 new = ~426.
- Property fuzzers green.
- Commit tagged `phase-2.8.1-stage1-complete`.
- **No production effect yet**: `DecodeScheduler.run_continuous` still
  delegates to legacy `decoding_continuous` (changed in Stage 2).
- L1 GPT-OSS / L2 / L3 unchanged from baseline — because prod path
  doesn't hit BoundaryHandler.

---

## 7. What this addendum does NOT cover

- Stage 2 (native decode loop + helpers) — see master plan §3 Stage 2.
- Stage 3 (native `put_on_hold`) — see master plan §3 Stage 3.
- Stage 4 (adapter surgery + legacy archival) — see master plan §3 Stage 4.
- Stage 5 (L1/L2/L3/L4 validation) — see master plan §3 Stage 5.
- The `AsyncLoadHostToGpu` legacy-sealed-union type — legacy implementation
  uses `NewLoadAsync` shape (ours). The old `AsyncLoadHostToGpu` type
  stays in the union unused for now; Stage 2 will delete or reuse it.

---

## 8. Risk register

| Risk | Mitigation |
|---|---|
| `BoundaryDecisions` in `continuous_batching.py` drifts between legacy and new | Both will coexist until Stage 4; legacy is unchanged reference; the new sealed union is the native. |
| `FakeLegacyBackend` stub bodies diverge from production | Adapter bodies are thin one-liner passthroughs (§4). Test assertions on call args catch drift. |
| Port misses a legacy mutation (e.g. `self._sequences_with_gpu_kv.discard`) | Full legacy text quoted inline in the port PR. Reviewer can diff line-by-line. `_sequences_with_gpu_kv` goes on the adapter too. |
| New collective order breaks `test_collective_ordering_fuzzer.py` | Collective count: 1 `all_gather_object` + 1 `broadcast_object` + 1 `all_gather_into_tensor` + 1 `barrier` = 4. Legacy has same 4 (finalize's `all_gather_into_tensor` and `barrier`, plus gather_state's `all_gather_object` and `broadcast_object_list`). Match in both order and count. |
| Planner rewrite breaks M4 unit tests | Option A (extend, not replace) — old rule-1-3 tests unchanged. |

---

## 9. Follow-through

After this addendum lands (commit 1):
- Next session implements 1a (`gather_boundary_state`) — budget ~200 LOC
  + tests. Self-contained commit.
- Sessions after that continue through the sub-commit ordering.
- Every commit is `pytest tests/unit/worker/ -x -q` green on the remote.

The master plan remains the primary reference for Stages 2–5; this
addendum only resets Stage 1's concrete work to match the actual tree.
