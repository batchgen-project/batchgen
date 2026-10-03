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
