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
