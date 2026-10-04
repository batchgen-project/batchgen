## 2026-10-04 — Oversized decode candidates livelocked at the soft page watermark

- **Symptom.** A long ON_HOLD trajectory whose GPU reservation exceeded the
  decode batching watermark was skipped on every selection round. The worker then
  emitted `Prepared batch: 0 sequences` indefinitely even though the request
  still fit in the physical GPU-KV pool.
- **Root cause.** `DecodeScheduler` treated the batching watermark as a hard
  capacity limit. Its predicate could never become true for a candidate over
  the watermark, so repeated selection could not make progress.
- **Fix.** `3738ad13` treats the watermark as a soft batching guard: a candidate
  that fits the physical pool may enter as a singleton when its capacity bucket
  is empty. Only a request larger than the physical pool raises
  `DecodeCapacityError`; a zero-page pool keeps the existing empty selection.
  The follow-up worker snapshot also accounts for pages held by IN_DECODE rows,
  so the singleton fallback cannot overcommit an active capacity bucket.
- **Validation.** The focused decode unit suite, `compileall`, and
  `git diff --check` passed.

## 2026-10-04 — Decode admission used a stale watermark instead of live capacity

- **Symptom.** Modern decode admission could strand a PREFILLED or ON_HOLD
  sequence even when its two-page GPU reservation fit the current allocator
  free pages. The selector also read one rank's total page count and resident
  metadata rather than a rank-wide live-free snapshot; the padded sequence cap
  did not include existing `IN_DECODE` rows.
- **Root cause.** The extracted selector retained the legacy batching watermark as a safety
  predicate. Allocation then checked free
  pages locally, so rank capacity drift could be discovered only after other
  ranks had begun mutation.
- **Fix.** `8171b654` — admission now consumes a collective snapshot whose TP
  group free capacity is the minimum across ranks and whose immutable total
  page count must match. The selector uses live free pages as the physical
  invariant, counts resident decode rows against `max_rank_bsz`, and the
  modern allocator performs an all-rank preflight before updating metadata or
  allocating pages.
- **Validation.** Pure decode tests cover admission past the old batching watermark,
  stale metadata versus live free pages, resident sequence caps, TP reduction,
  total-page mismatch, `py_compile`, `compileall`, and `git diff --check`.

## 2026-10-04 — Page-boundary loads could diverge after rank-local capacity checks

- **Symptom.** A page boundary could select a replicated TP load or extension from
  one rank's free-page view, then drop it locally on another rank. A rank entering
  the boundary with a different UUID set could also take the pending-load/empty
  return while peers entered the next collective.
- **Root cause.** Boundary planning did not carry the immutable total-page snapshot
  or the existing decode-row count, Phase E filtered selected loads independently on
  each rank, and legacy modes still used a local free-page loop. Rank 0 also had no
  broadcasted error path for planner validation failures.
- **Fix.** `b049a941`, `c25efc63`, and `a716b851` validate total/free capacity on every rank before rank-0
  planning, applies TP-tightest capacity and the persistent row cap to boundary
  selection, preflights every extension/load collectively before allocator mutation,
  and broadcasts rank-0 planner errors. The entry UUID guard runs before pending-load
  and empty handling. Legacy boundary admission delegates to the same decode selector
  and collective allocator preflight; failed extensions release the full allocation
  before moving sequences to `ON_HOLD`.
- **Validation.** Focused boundary and payload-validator tests cover one-rank TP
  extension/load shortfalls, total-page mismatch, row-cap admission, UUID
  alignment, and pre-broadcast validation order; 31 tests pass. `py_compile`,
  PR hygiene, and `git diff --check` pass.

## 2026-10-04 — Prefill admission could publish status before host allocation

- **Symptom.** Prefill selection used a host-KV snapshot, then marked rows
  `IN_PREFILL` and entered phase configuration before the shared host allocator
  accepted the reservation. A stale snapshot or a concurrent allocator could
  raise after partial registration, leaving status or per-sequence host-page
  metadata advanced without a committed allocation.
- **Root cause.** Host capacity was checked as a local selection hint rather
  than a collective transaction. The native batch allocator also acquired each
  sequence independently, so an over-capacity wave could consume earlier rows
  before failing on a later row.
- **Fix.** The prefill path now builds one shared reservation formula, gathers a
  fresh per-node free-page snapshot and owner demand across all ranks, and
  commits host-page metadata only after collective allocator success. Failed
  peers release pages and unregister tentative rows before the scheduler leaves
  candidates in `QUEUEING`/`EVICTED`. Native `AcquirePagesForSequences` now
  checks aggregate demand under the shared allocation lock and rolls back its
  metadata/free-stack mutation on unexpected failure.
- **Validation.** Focused lifecycle guards pass; changed Python files compile;
  the native integration regression is CUDA-gated and requires the Linux CUDA
  extension environment.
