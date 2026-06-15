# HANDOFF — batchgen ⇄ DeepSeek-V4-Flash character-exact A/B

## 🟦 SESSION 8 (2026-06-15) — QAT experiment: big drift reduction, residual gap remains

### Step 1 (parity) — QAT path is BIT-EXACT (confirmed)
`test_v4_linear_numerics_parity.py` IN CONTAINER:
- WITHOUT flag (default bf16-dequant): fp8 linear cos=0.9996 rel=2.67e-2; fp4 expert cos=0.9987
  rel=5.1e-2 (FAILS >0.999). This is the drift source.
- WITH `BATCHGEN_V4_QAT_LINEAR=1`: fp8 linear AND fp4 expert both **cos=1.000000 rel=0.0** vs
  official. `[V4_QAT_LINEAR] active` logged. So per-op QAT numerics are exact.

### Step 2 (A/B with QAT flags) — divergence DELAYED massively, not eliminated
Server flags: BATCHGEN_V4_QAT_LINEAR=1 + BATCHGEN_V4_GROUPED_MOE=1 + BATCHGEN_GLM5_LMHEAD_FP32=1
(+ PYNCCL + MLA_SM120_TRITON, KV=52GB, frac 0.62, pool 64). compare_ab.py vs golden.jsonl:
```
[EXACT] tiny-math   (1/1)
[DIFF]  identity     first char diff at 106  (was ~10 WITHOUT QAT) <-- QAT cut drift ~10x
[DIFF]  haiku        first char diff at 0    (golden 'A vast...' vs bg "The ocean's...")
[DIFF]  sys-math     first char diff at 51
exact match: 1/4
```
QAT moved the identity divergence from char ~10 to char 106 (first 105 chars now identical) =>
the dense-linear QAT gap was a REAL and major contributor. But residual divergence remains; some
numeric path still differs. haiku diverging at char 0 (different first token) suggests a
remaining gap that flips even the first decoded token for some prompts.

### Step 3 (NEXT) — localize the RESIDUAL with layer-by-layer DIVTRACE
QAT is active with no FAILED/SKIPPED, so the residual is NOT the dense linears already covered.
Candidates for the remaining gap (per Session 6 traps + this result):
- Attention internals: MLA q/kv rope, sparse indexer topk, attn_sink, fp8 KV quant of prompt KV.
- MoE router: hash-routing layers 0-2 tid2eid int64 vs official int32; topk gather/order.
- The GROUPED MoE kernel (v4_grouped_mxfp4_moe_forward_3d_ptrs) vs per-expert: Step-1 parity was
  on the PLACEHOLDER expert, NOT the grouped kernel — grouped path parity still unproven in-situ.
- lm_head fp32: verify GLM5_LMHEAD_FP32 actually matched official ParallelHead.float() (argmax
  tie-breaks).
Use BATCHGEN_V4_DIVTRACE=1 (+_PREFILL=1) dump vs official inference/dump_ref_acts.py for ONE
prompt (e.g. haiku, since it diverges at token 0 = easiest to localize). Diff per-layer h_in/
attn_out/h_after_attn/h_after_ffn cosine + final logits top-1/top-2 margin. First layer where
cosine drops <0.9999 OR router topk ids differ = the culprit. tools/analyze_divtrace.py +
v4flash_official/results/debug/compare_{traces,decode_traces,attn_internals}.py.

NOTE perf: QAT path is slower (more tilelang JIT first-call); A/B of 4 prompts took ~8min.

### RESIDUAL DRIFT ROOT CAUSE FOUND (code inspection, Oracle-guided) — grouped MoE decode kernel is NOT QAT-faithful
The grouped MoE decode kernel `v4_grouped_mxfp4_moe_forward_3d_ptrs` -> `grouped_mxfp4_gemm_3d`
(batchgen/moe/mxfp4_grouped_gemm.py) DEQUANTIZES the FP4 expert weights to BF16 and runs a BF16
GEMM against BF16 (NON-quantized) activations: `weight_bf16 = mxfp4_dequantize(...)`;
`acc += tl.dot(lhs_tile, val_bf16.T)` (mxfp4_grouped_gemm.py:241-246 unfused path, :418-422 triton
kernel). hidden_3d input is BF16 ([E,M_max,K] BF16, line 903/920). There is NO activation
quantization (no act_quant to fp8/fp4 of the input).

This is EXACTLY the non-QAT "dequant weights to bf16, F.linear" pattern that BATCHGEN_V4_QAT_LINEAR
replaced for the DENSE linears — but the grouped DECODE-expert kernel still uses it. So:
- dense/attention linears: QAT-fixed, bit-exact (QAT_LINEAR=1)
- prefill experts (per-expert loop, >512 tok): QAT-fixed, bit-exact
- DECODE experts (grouped kernel, <=512 tok): STILL bf16-dequant-weights => ~5e-2/GEMM error
  => the residual decode drift (identity char-106, haiku token-0).

Why this matches: grouped kernel is decode-only (<=BATCHGEN_V4_GROUPED_MOE_MAX_TOKENS=512); prefill
correctness is fine (per-expert), decode MoE has the gap. Confirmed WITHOUT the slow per-expert
server loop (which is ~5s/token, impractical: 11min produced no result).

### THE FIX (decision pending) — make grouped decode MoE QAT-faithful
Options:
  A. Make `grouped_mxfp4_gemm_3d` act-quantize activations (block-128 ue8m0 for w2 input, fp4 for
     w1/w3 like the official Expert) and do the GEMM in quantized space, matching
     `_qat_linear`/official Expert exactly. This is a real kernel change (triton mxfp4 grouped
     gemm currently dequant-to-bf16). HIGH effort.
  B. For char-exact runs, set BATCHGEN_V4_GROUPED_MOE_MAX_TOKENS=0 (or GROUPED_MOE=0) so decode
     also uses the bit-exact per-expert loop. CORRECTNESS-exact but ~28x slower decode
     (4893ms/token) — fine for char-exact VALIDATION, not for throughput.
  C. Accept the tradeoff: grouped MoE for throughput (72% MMLU, fast) vs per-expert for char-exact
     (slow). They are different operating points.

Cheapest validation of the diagnosis: run A/B (or even 1 token for haiku) with GROUPED_MOE=0 +
QAT_LINEAR=1 + LMHEAD_FP32=1 — if char-exact improves (haiku token-0 becomes correct), the grouped
kernel is confirmed as the residual. (Blocked only by per-expert speed; a unit test of
grouped_mxfp4_gemm_3d vs act-quant reference is the deterministic alternative.)


## 🟧 SESSION 7 (2026-06-15) — full-run crash chain (2 fixed, 1 pending Oracle)

Investigating why the full 12k MMLU SIGSEGV'd at ~105/12032. Found a CHAIN of crashes in the
watermark-driven on-hold/re-prefill cycle (only triggers on long multi-wave runs; the 100-prompt
run finished before any watermark fired). All are SEPARATE from the page-accounting fix.

### CRASH 1 (FIXED + E2E verified, committed 12e2e2a4)
`_put_sequences_on_hold` (worker.py:5752) + `_put_sequences_onhold` (2344) filtered GPU-tracked ids
via `mgr._sequences`, which DeepSeekV4KVCoordinator lacks (it fans out to 4 sub-pools) ->
AttributeError when host-KV watermark interrupts decode to evict -> SIGSEGV in GPU_KV_Buffer dtor.
FIX: added `coordinator.tracked_sequence_ids()` (authoritative via swa._sequences) + routed both
on-hold paths through it (V4-aware branch). Unit test `test_v4_tracked_sequence_ids_filters_unknown`
passes (5/5). E2E VERIFIED: 500-prompt run hit "[WATERMARK] putting 20 sequences ON_HOLD" on all 4
ranks at Decode 2 / iter 256 (the exact prior crash point) with NO AttributeError/SIGSEGV.

### CRASH 2 (FIXED + E2E verified, committed 3db33857)
Coordinator re-init on watermark re-prefill. configure_prefill deep-frees the coordinator
(destroyed-but-not-None); the prefill re-init gate only fired on `is None` -> re-prefill ran
against a destroyed coordinator -> "DeepSeekV4KVCoordinator is not initialized" -> SIGSEGV.
FIX (Oracle bg_7e97b68c): new `_maybe_reinit_v4_gpu_kv_for_prefill(prefill_uuids)` decides re-init
COLLECTIVELY — gated on the GLOBAL prefill_uuids (identical across ranks), predicate
`is None or not is_initialized`, with an all_gather consistency guard that raises rather than
deadlocks if ranks diverge (because _init_gpu_kv_with_actual_size runs a dist.broadcast +
early-returns on initialized ranks). Kept destroy-before-configure_prefill (skipping it re-OOMs).
E2E VERIFIED: 500-prompt run hit `PREFILL TRIGGER` + `Breaking for prefill` + a 2ND prefill config
(the exact prior crash path) and SURVIVED — 109+ completions, 25min, no "not initialized"/SIGSEGV.

### Commits this session: 12e2e2a4 (CRASH 1), 3db33857 (CRASH 2). Full-run crash chain resolved.
The watermark on-hold -> re-prefill -> resume-decode cycle now works for V4. Bounded AND
multi-wave runs survive. Remaining: the QAT char-exact experiment (Session 6 plan) is still
pending; a full 12k throughput run is slow (~hours) but no longer crashes.

### (superseded) CRASH 2 original diagnosis — coordinator not re-initialized on watermark re-prefill
After CRASH 1 fixed, the watermark "[DECODE] Breaking for prefill - 99 queued" loops back to PREFILL
phase. `configure_prefill` ALWAYS deep-frees + `_destroy_gpu_paged_kv_cache()` (worker.py:7849, "Bug
Fix 7.2": free 20-30GB GPU KV so prefill model loads without OOM). Coordinator is now
is_initialized=False but NOT None. The prefill re-init gate (worker.py:7237-7246) only fires when
`gpu_paged_kv_cache_manager is None` -> SKIPPED -> prefill_prepacked -> decoder_layer ->
coordinator.allocate_pages_for_sequences -> `_ensure_initialized()` raises "DeepSeekV4KVCoordinator
is not initialized" -> SIGSEGV. Crash traceback: worker.py:9348 decoder_layer ->
coordinator.py allocate_pages_for_sequences -> _ensure_initialized.

PROPOSED (pending Oracle): change gate from `is None` to `is None or not is_initialized`. OPEN
RISKS Oracle is checking: (a) `_init_gpu_kv_with_actual_size` issues dist.broadcast(src=0) — the
re-init condition must evaluate identically across all 4 ranks or the collective desyncs (ties to
the earlier deadlock fix); (b) re-initing 52GB KV before re-prefill may re-introduce the OOM that
Bug Fix 7.2 avoids (decode model + 34GB resident experts may still be loaded); (c) configure_prefill
deep-free releases ALL GPU KV pages INCLUDING in-flight IN_DECODE sequences' KV — does the
destroy/recreate cycle corrupt in-flight decode state for V4's GPU-resident-only KV? This is a
deeper architecture question: V4 needs coordinator-KV-before-prefill (resident, no host upload),
but the generic streaming path assumes prefill/decode models don't coexist and freely
destroys/recreates KV. The on-hold->re-prefill->resume-decode cycle may need V4-specific handling
to preserve resident KV of still-in-flight sequences.

### Net: full 12k still blocked by CRASH 2. Bounded runs (<=~100 prompts, no watermark) work fine
(verified 72% on 100). The watermark fires only when host-KV crosses 70% with queued seqs, i.e.
sustained multi-wave load.


## 🟦 SESSION 6 (2026-06-15) — CHAR-EXACT PLAN (Oracle-validated). Run AFTER 12k MMLU finishes.

### Root cause of multi-token greedy drift (PROVEN)
Model is QAT-trained: official `linear()` quantizes ACTIVATIONS to fp8 (block-128, ue8m0) before
every quantized GEMM. batchgen default `_linear_from_weight` dequantizes WEIGHTS to bf16 + F.linear
=> ~2.6e-2 rel/GEMM, compounds to hidden cos 0.98@L0 -> 0.92@L42 -> greedy argmax flips at token
2-12. Single token (tiny-math "4") matches; longer generations drift. Proven by
`tests/integration/test_v4_linear_numerics_parity.py`: `_qat_linear` (model.py:747, env
BATCHGEN_V4_QAT_LINEAR=1) is cos=1.0 rel=0.0 vs official; default path is not. Oracle confirmed
this magnitude alone explains the drift (no extra discrete bug needed unless a layer-LOCAL collapse
persists with QAT on).

### Updated insight: per-expert launch-storm blocker is GONE
Old note said BATCHGEN_V4_QAT_LINEAR=1 couldn't be enabled (per-expert tilelang launch storm wedged
the server). But the grouped MXFP4 MoE kernel `v4_grouped_mxfp4_moe_forward_3d_ptrs` (model.py:1789,
env BATCHGEN_V4_GROUPED_MOE=1) already does `act_quant` once per layer's batch — that's the MMLU
path. So experts already use QAT-faithful quant; only DENSE/ATTENTION linears + lm_head remain on the
non-QAT path. All 3 flags verified WIRED:
- BATCHGEN_V4_QAT_LINEAR -> _qat_linear in _linear_from_weight (model.py:816, graceful fallback)
- BATCHGEN_V4_GROUPED_MOE -> grouped expert kernel (already on for MMLU)
- BATCHGEN_GLM5_LMHEAD_FP32 -> force_fp32 lm_head (worker.py:8811/9025/9355; model.py:187/239)

### Oracle-validated experiment plan (cheap -> decisive; DO IN ORDER)
1. **Grouped-MoE parity in isolation FIRST.** "calls act_quant" is necessary NOT sufficient. For one
   layer/token-batch, compare official MoE output vs grouped kernel with identical hidden states,
   router logits/topk ids/weights, expert weights/scales, accumulation dtype. Verify block-128
   layout, UE8M0 scale rounding, expert packing, routing order, top-k norm, combine order, dist
   ownership/all-reduce. (Can be a unit script — minimal GPU.)
2. **One-prompt DIVTRACE with all 3 flags ON**, diff vs official per-layer dump:
   - batchgen: `BATCHGEN_V4_DIVTRACE=1 BATCHGEN_V4_DIVTRACE_PREFILL=1 BATCHGEN_V4_DIVTRACE_DUMP_PATH=<dir>`
     + BATCHGEN_V4_GROUPED_MOE=1 BATCHGEN_V4_QAT_LINEAR=1 BATCHGEN_GLM5_LMHEAD_FP32=1
     -> divtrace_rank{0-3}.pt
   - official: `v4flash_official/inference/dump_ref_acts.py` (torchrun --nproc-per-node 4, same prompt)
   - Confirmation = NO progressive cosine decay across layers + router/topk id agreement + prefill
     final logits top-1 AND top-2 margin agree. (Cosine alone hides logit-order issues — check
     top-1 id + top-2 margin too.)
3. **Teacher-forced multi-step**: feed official tokens 16 steps, compare logits/top-1 each step.
   Separates model-state parity from greedy-trajectory divergence.
4. **THEN** short greedy A/B vs golden.jsonl (`v4flash_official/results/ab_small/compare_ab.py`).

### TRAPS (Oracle)
- lm_head: must be PLAIN fp32 projection (BATCHGEN_GLM5_LMHEAD_FP32=1), do NOT route through
  QAT_LINEAR activation-quant — else double-quantize. (_linear_from_weight QAT gate requires
  scale!=None and bias is None; lm_head goes through vocab_parallel_lm_head, separate path — verify
  it's not also QAT'd.)
- If QAT dense STILL wedges: do NOT group dense GEMMs first. Fix JIT hygiene — pre-warm exact
  (M,N,K,dtype,block) shapes in EACH worker AFTER cuda init, serialize compile across ranks,
  persistent JIT cache + file lock, bucket token counts. Group dense only if profiling still shows
  launch overhead after warmup.

### Diagnostic infra map (file:line)
- DIVTRACE: model.py:91-100 (flags), dump fns 323-686, flush 348.
- Attn tensor dump: v4_flashmla_adapter.py:21-27 (BATCHGEN_V4_ATTN_TENSOR_DUMP=<dir>).
- analyze tool: tools/analyze_divtrace.py ; trace script: tools/v4_divtrace_blackwell.sh.
- parity tests: tests/integration/test_v4_linear_numerics_parity.py (cos>0.999),
  test_v4_prefill_sparse_parity.py (cos>0.999) — run IN CONTAINER (bare metal lacks cuda.h).
- official: v4flash_official/inference/{dump_ref_acts.py,gen_golden.py};
  results/ab_small/{golden.jsonl,compare_ab.py}; results/debug/compare_{traces,decode_traces,attn_internals}.py.

### Status: 12 commits landed (b done). 12k MMLU running (let it finish, then start step 1 above).


## 🟩 SESSION 5d (2026-06-15) — FIX IMPLEMENTED & VERIFIED: compression-aware page accounting

The 4-pool over-allocation bug (Session 5c) is FIXED. The 100-prompt MMLU run that previously
crashed within ~2 min now runs 14+ min with NO "Insufficient free pages" / NO SIGSEGV, sequences
reach EOS ("completed" in logs), and the server stays healthy. Same admission config that crashed
before (frac 0.62, --max-pool-size 1024, KV=52GB): now admits 100 seqs and decodes cleanly.

### Code changes (uncommitted, working tree)
1. `batchgen/kv_cache/deepseek_v4_kv_coordinator.py`:
   - `_POOL_COMPRESS_RATIO = {swa:1, c4:4, c128:128, indexer:4}` + `_pool_logical_tokens()`:
     converts raw context tokens to each pool's compressed token space (ceil, max(1) floor).
   - `allocate_pages_for_sequences()` now charges each pool `pool_logical_tokens`, not raw.
     (extend_pages_for_sequence delegates, so auto-fixed.)
   - NEW `can_allocate_pages_for_sequences()` (per-pool preflight, ALL pools must fit),
     `additional_pages_needed_by_pool()`, `free_worker_pages(page_size=64)` (min-over-pools
     binding free capacity in worker 64-tok pages). `get_stats()` UNCHANGED (still sums; metrics).
2. `batchgen/batchgen_worker.py`:
   - NEW `_gpu_kv_can_allocate(manager, {global_id: target_tokens})` and
     `_gpu_kv_free_worker_pages(manager)` — route V4 coordinator to the per-pool methods, fall
     back to legacy scalar for other managers.
   - Switched ALL GPU-KV admission/extension/onhold decision sites from
     `get_stats().num_free_pages` to the V4-aware helpers: `_extend_gpu_kv_allocation`,
     `_allocate_gpu_kv_two_page_buffer`, `_select_sequences_for_onhold`, `_get_gpu_kv_free_pages`,
     `_prepare_decode_batch_two_page_buffer`, the boundary all-gather sources + alloc guards in
     `_page_boundary_fast` and the `_try_load_new_sequences*` family. (Host-KV worker_view sites
     left as-is — different subsystem.)
3. `tests/kv_cache/test_v4_kv_coordinator.py`: +4 tests (c128=4 not 512 for raw 1024; per-pool
   preflight False when one pool empty but sum>0; preflight True when all fit; free_worker_pages
   = binding pool). Full suite: 8 passed, 1 skipped (FlashMLA ref file absent). Run IN CONTAINER
   (`docker exec bg-v4 ... pytest`) — bare metal lacks cuda.h to build core_engine.

### Status: crash FIXED + accuracy VERIFIED AT SCALE.
100-prompt run (previously crashed at ~2min) ran 25+ min healthy. 91/100 sequences completed
cleanly (9 slow reasoning stragglers still decoding to the 1024-token cap, no crash). Partial
accuracy on the 91 completed, scored with the harness's own `extract_prediction` (index-based
custom_id -> dataset["answer"][idx], <think>-aware): **72/91 correct = 79.1%, 0 extraction
failures.** Consistent with the earlier 20-prompt 70% and a plausible V4-Flash MMLU-Pro score.
The fix produces CORRECT results at scale, not just "no crash".

Partial-score snippet (run in container):
  python3 -c 'import json,sys; sys.path.insert(0,"/work");
  from tests.e2e.v4flash_mmlu_pro_test.v4flash_mmlu_pro_batch_test import extract_prediction;
  import pandas as pd; gt=pd.read_parquet("/work/tests/e2e/r1_mmlu_pro_test/mmlu_pro_test.parquet")["answer"].tolist();
  ... idx=int(custom_id.split("-")[1]); pred==gt[idx] ...'

100-prompt run COMPLETED via the official harness: **Total: 100  Correct: 72  Accuracy: 72.00%**
(4 extraction failures). End-to-end, no crash, ~30 min wall. Definitive proof the fix works.

Full 12k MMLU launched (user choice: max_dec=1024). ~30min/100 reasoning seqs -> many hours.
Output: .sisyphus/blackwell/mmlu_full_grouped.json. Monitor via the batch incremental file
under .sisyphus/mmlu_storage/incremental/.

### IMPORTANT: full 12k needs --max-pool-size 128 (NOT 1024)
With `--max-pool-size 1024` the 12k run admits a 1024-sequence prefill wave (648k tokens) and dies
with **CUDA OOM "Tried to allocate ~31.8 GiB"** during PREFILL activation/compute (NOT KV pages —
that's the page-accounting fix working; this is prefill forward activation memory). 1024 concurrent
prefill seqs need ~32GB activation on top of 34GB resident experts + 52GB KV -> only ~18GB free ->
OOM. FIX: `--max-pool-size 128`. The pool refills as sequences complete, so all 12,032 still
process; each wave admits ~128 seqs (4,244 pages), no OOM, no crash. VERIFIED: 12k run with pool 128
is healthy, processing 128-seq waves, GPU ~60-92%, no OOM/page/init errors.
(Aside: a separate pre-existing `DeepSeekV4KVCoordinator is not initialized` crash in
`_populate_v4_prefill_kv` (wrappers.py:661 -> coordinator.allocate_pages_for_sequences) was seen
once on a SECOND batch on the same server — coordinator lifecycle across batches. Not triggered by
the page-accounting fix; first-batch runs are fine. Flagged for later if multi-batch reuse needed.)

WORKING full-12k launch: same docker config as above but `--max-pool-size 128`, on a FRESH server
(first batch), storage cleared. Run = batch in .sisyphus/mmlu_storage/incremental/.

---

## 🟧 SESSION 5c (2026-06-14) — ROOT CAUSE of MMLU "page exhaustion" FOUND: 4-pool free-page accounting bug

### It is NOT a page leak. Pages ARE freed on EOS (verified). It is a SCHEDULER ACCOUNTING bug.
Two parallel explore agents confirmed the page-release path is correct in BOTH pool mode
(batchgen_worker.py:7348-7403) and legacy mode (10193-10248): completed seqs call
`_release_gpu_kv_pages` -> `manager.free_pages_for_sequences` -> all 4 V4 pools push pages back.
So nothing leaks. The crash has a different cause.

### THE BUG: V4 coordinator sums free pages across 4 heterogeneous pools; scheduler over-admits
`DeepSeekV4KVCoordinator` (batchgen/kv_cache/deepseek_v4_kv_coordinator.py) runs **4 independent
pools with DIFFERENT page sizes** (lines 61-63, 75-99), each with the SAME `num_pages` capacity:
- `swa`     page_size = 128 tokens/page
- `c4`      page_size = 64  tokens/page   (base_page_size 256 // 4)
- `c128`    page_size = **2** tokens/page (base_page_size 256 // 128)   <-- drains ~64x faster
- `indexer` page_size = 64  tokens/page

A sequence at N context tokens consumes from ALL FOUR pools, but wildly different counts. At
N=1024: swa=ceil(1024/128)=8, c4=16, indexer=16, but **c128=ceil(1024/2)=512 pages**. The c128
pool is the BINDING CONSTRAINT and empties ~64x faster than swa.

`get_stats()` (coordinator lines 269-284) returns the **SUM** of free pages across all 4 pools.
The worker's admission/extension/on-hold logic (batchgen_worker.py:2182, 2257, 2271, 2308) all
compare required pages against this SUMMED `num_free_pages`. The sum is dominated by the 3
slow-draining pools, so it looks healthy even when c128 is nearly empty. When c128 actually runs
dry, `allocate_pages_for_sequences` -> `_PageStack.pop()` raises the HARD
`RuntimeError: Insufficient free pages` (deepseek_v4_single_kv_pool.py:81-83), BYPASSING the
graceful ON_HOLD / extension-failure safety valve (which trusted the bogus summed count).

### Why this matches EVERY observation
- "55,916 total pages" in logs = the SUM (4 × ~13,979). Real binding capacity ≈ ONE pool's
  ~13,979 page-units, and for long decode the c128 pool is even tighter per-token.
- Crash "need 540, have 128" at only 20-100 seqs: the c128 pool hits zero while the SUM still
  looks huge.
- Independent of frac / max-pool-size / KV-GB: all of them scale the SUM, not the per-pool
  imbalance. 20 prompts (short) completed; 100 (more decode) exhausted c128.

### THE FIX (recommended; pick 1, prefer A)
**A. Make free-page accounting pool-aware (correct fix).** The scheduler must treat "free pages"
as the MIN headroom across pools relative to each pool's per-token page cost — not the SUM.
Options:
  - Add a coordinator method e.g. `max_additional_tokens()` / `free_pages_normalized()` that
    returns the BINDING constraint: for each pool, `free_pages_pool * pool.page_size_tokens` =
    free TOKENS that pool can still hold; the sequence-admissible budget = MIN over pools of
    free-tokens, converted back to the worker's PAGE_SIZE=64 unit. Use THAT everywhere the worker
    currently calls `get_stats().num_free_pages` for admission/extension/on-hold decisions
    (batchgen_worker.py:2182, 2257, 2271, 2308, and `_get_gpu_kv_free_pages` 4622).
  - OR change `get_stats().num_free_pages` for the V4 coordinator to report the MIN-normalized
    free capacity instead of the SUM (simplest, but get_stats is also used for display/metrics —
    check call sites first). Safer to add a NEW method and switch the admission/extension sites.
**B. Stopgap (no code change):** cap concurrency so the c128 pool never exhausts. c128 at 1024
tokens needs 512 pages/seq; with ~13,979 c128 pages, safe concurrent count ≈ 13979 / (max_tokens/2)
/ safety. For max_decoding_length=1024: ~27 seqs max, ~20 safe. THIS is exactly why --max_prompts
20 worked and 100 didn't. So: run full MMLU in **chunks of ~16-20 prompts** (legacy mode survives
the error gracefully) and aggregate. Slow but unblocks the number today.

### Verification before/after fix
Repro: KV=52GB, `--max_prompts 100 --max_decoding_length 1024` -> crashes. With fix, the scheduler
should ON_HOLD/evict instead of crashing, and the run should complete (slower) for any prompt
count. Add a debug log of per-pool `get_stats()` (swa/c4/c128/indexer free) right before the
admission check to SEE c128 hit zero first — that single log line proves the diagnosis.

### Key file:line map
- coordinator pools + sizes: deepseek_v4_kv_coordinator.py:61-63, 75-99
- SUM bug: deepseek_v4_kv_coordinator.py:269-284 (get_stats)
- hard raise: deepseek_v4_single_kv_pool.py:81-83 (_PageStack.pop)
- worker admission/extension/onhold reads: batchgen_worker.py:2182, 2257, 2271-2276, 2308, 4622
- correct (but bypassed) safety valve: _extend_gpu_kv_allocation 2246-2291 (returns False, no
  raise) + _put_sequences_onhold 2339-2374 + boundary extension-fail handler 10505-10548
- release path (CORRECT, not the bug): _release_gpu_kv_pages 3487-3517; coordinator
  free_pages_for_sequences 255-268; pool free_pages_for_sequences (single pool) 392-407

### Proven-good result this session (unchanged): --max_prompts 20 = 70% accuracy (14/20).

### ⭐ REFINED ROOT CAUSE (Oracle-verified, bg_6d317457) — allocator charges compressed pools in RAW token space
The SUM-accounting bug is real (scheduler over-admits then hard-raises), BUT the DEEPER bug is in
allocation, and it's why even ~20-100 seqs exhaust:

`DeepSeekV4KVCoordinator.allocate_pages_for_sequences(seq_ids, num_tokens)` (coordinator
lines 209-225) passes the SAME raw `num_tokens` to ALL 4 pools. But the compressed pools only
ever STORE compressed tokens:
- c4 / indexer store `c4_kv.shape[0]` rows with `c4_positions = arange(c4_kv.shape[0])`
  (≈ raw/4)  — v4_prefill_populate.py:70-89
- c128 stores `compressed.shape[0]` rows with `c128_positions = arange(compressed.shape[0])`
  (≈ raw/128) — v4_prefill_populate.py:97-120
- swa stores raw tokens (ratio 1).

So for a 1024-token prompt the c128 pool only USES positions 0..7 (8 compressed tokens →
ceil(8/2)=4 pages), but the allocator RESERVES ceil(1024/2)=512 c128 pages. **A 128× over-
allocation on c128 (and 4× on c4/indexer).** c128 is NOT inherently 64x heavier — the page_size=2
is intended (2 compressed tokens × 128 ratio = 256 raw tokens/page); the bug is feeding it raw
token counts. This drains c128 ~128x too fast → exhaustion at tiny concurrency.

### THE FIX (Oracle-recommended, supersedes earlier "MIN accounting" plan)
Make the coordinator translate raw context-token capacity into each pool's COMPRESSED token space
for BOTH allocation and preflight:
```
ratio = {"swa": 1, "c4": 4, "indexer": 4, "c128": 128}
logical_tokens = max(1, ceil(raw_tokens / ratio[pool]))
required_pages = ceil(logical_tokens / pool.page_size_tokens)
# => for raw T: swa=ceil(T/128), c4=ceil(T/256), indexer=ceil(T/256), c128=ceil(T/256)
```
Steps (Oracle action plan):
1. Coordinator: add per-pool required-page helper using the ratio map above. Change
   `allocate_pages_for_sequences` + `extend_pages_for_sequence` to charge each pool its
   COMPRESSED page count, not raw. Keep external contract "num_tokens = raw context tokens".
2. Add coordinator preflight: `can_allocate_pages_for_sequences(seq_ids, raw_tokens) -> bool`
   that checks PER-POOL: `all(missing[p] <= free_pages[p])`. (Per-pool, NOT a scalar MIN — rounding
   is per-seq per-pool.) Also `additional_pages_needed_by_pool(...)` for diagnostics, and
   `free_worker_pages(page_size_tokens=64)` as a conservative scalar for legacy greedy paths.
3. Call `can_allocate_pages_for_sequences` immediately BEFORE every real V4 alloc/extension so the
   scheduler returns ON_HOLD/skip gracefully instead of hitting the hard `_PageStack.pop()` raise.
4. Keep `get_stats()` summing for METRICS/display, but route all ALLOCATION DECISIONS through the
   new V4-aware helper. Patch ALL decision sites, not just 5: Oracle flagged
   batchgen_worker.py:2182, 2257, 2271, 2308, 4622, AND 3286(direct alloc), 5924, 6466, 8528,
   8586, 9968, 10575, 14580/14645, 14731/14820. Add a single worker helper that routes V4 managers
   to V4-aware capacity and use it everywhere.
5. Keep hard-raise in `_PageStack.pop()` as an invariant check (should never fire after fix).
6. Regression tests: raw 1024 → c128≈4 pages (NOT 512), c4/indexer≈4, swa=8; and a test where
   summed free is large but one pool is exhausted must preflight-False before allocating.
Effort: MEDIUM (1-2 days), touches scheduler admission + needs distributed regression coverage.

VERIFIED FACTS behind this (don't re-derive): c4/c128/indexer store compressed positions
(arange over compressed.shape[0]) — v4_prefill_populate.py:70-89 (c4), 97-120 (c128). Pool sizes
swa=128/c4=64/c128=2/indexer=64 tok/page — coordinator 61-99. allocate passes same raw num_tokens
to all 4 — coordinator 217-219. get_stats SUMs — coordinator 269-284. hard raise —
deepseek_v4_single_kv_pool.py:81-83.

---

---

## 🟥 SESSION 5b (2026-06-14) — FULL MMLU-PRO BLOCKED BY GPU-KV POOL-MODE PAGE EXHAUSTION

### Goal
Run full MMLU-Pro (12,032 prompts, max_decoding_length=1024) with prefill-offload (experts
streamed) + decode-no-offload (experts resident) = `BATCHGEN_V4_GROUPED_MOE=1`. World-size 4,
Blackwell sm120, docker `batchgen:v4-kernels-user`.

### Two prerequisite issues FIXED this session (so the run can even start)
1. **Storage PermissionError** — server runs as uid 1003 (leyang) but `batchgen/storage/{files,
   batches,files_meta}` are root-owned 755 → `POST /v1/files 500 PermissionError`. FIX: pass
   `--storage-path /work/.sisyphus/mmlu_storage` (a leyang-owned 777 dir). MUST clear stale
   `files/ files_meta/ batches/` between runs or you get `400 ... already has active batch`
   (file dedup by hash + persisted IN_PROGRESS batch from a crashed run).
2. Use `docker run --init` so killed workers get reaped (otherwise zombie blocks `docker rm`).

### THE BLOCKER (NOT a config problem — looks like a page leak)
Every full-run attempt crashes with `Error in pool mode on rank N: Insufficient free pages:
need ~5xx, have <few>` → SIGSEGV on all 4 ranks. Swept the entire memory config space:

| gpu-memory-frac | max-pool-size | KV cap (GB) | KV page-units | result |
|---|---|---|---|---|
| 0.4  | 10240 | auto  | ~8.4k  | page exhaustion (admitted 2081 seqs/wave) |
| 0.95 | 10240 | auto  | 86GB→  | CUDA OOM (no room for 34GB resident experts) |
| 0.52 | 1024  | 48    | 11,561 | page exhaustion |
| 0.62 | 192   | 52    | 13,979 | page exhaustion |
| 0.62 | 64    | 52    | 13,979 | page exhaustion |

**Decisive data point (pool=64):** prefill of 64 sequences used only **2,126 of 55,916 pages
(~4%)**, prefill COMPLETED, then DECODE crashed `need 540, have 128`. 64 sequences cannot
legitimately drain 55,916 pages → this is a **page leak / double-free / missing-release in the
GPU-KV pool-mode decode path**, not a sizing problem. Confirmed empirically: more KV / fewer
seqs does NOT help.

Why prior validated runs didn't hit it: they used `--max_prompts 40` (tiny, bounded) and
finished before the leak accumulated. The full 12k run sustains pool admit/refill long enough
to drain all pages.

### KEY NUMBERS for whoever debugs this
- Resident experts (grouped MoE) = **34.27 GiB/rank**, allocated at decode-config time, AFTER
  KV is sized. So KV cap must be ≤ ~52GB or experts OOM. Use `BATCHGEN_GPU_KV_CACHE_SIZE_GB=52`
  (env override; bypasses the `total*frac-used` sizing-before-experts trap). frac alone is a
  trap: at 0.95 KV grabbed 86GB pre-experts → OOM.
- At KV=52GB: 13,979 page-units = 55,916 pages (×4). 64 seqs prefill = 2,126 pages.
- Crash site: `Error in pool mode` in the worker decode loop; pages from
  `DeepSeekV4KVCoordinator` (GPU-KV). 43 layers, c4_layers=21, c128_layers=20.

### NEXT-DEBUG POINTERS (where to look)
- `batchgen/batchgen_worker.py:7174-7203` — V4 decode reads prompt KV from GPU coordinator pools
  ONLY ("no host->GPU upload path"); `_populate_v4_prefill_kv` resident path; line 7203 "Wait for
  all async KV offloads before decode". **Suspect: GPU pages allocated per decode step but not
  released on sequence EOS / completion in pool mode.**
- GPU-KV page release path: `batchgen/core/KV_Storage/` (host_paged_kv_manager.cpp,
  host_paged_kv_worker_view.h, GPU_KV_Buffer). Look for where decode-step page allocation frees
  on EOS vs accumulates.
- `model.py:1642` `enable_ep_offloading = world_size > 1` and `model.py:1710` grouped path gate.
- Compare pool mode (`--max-pool-size >0`, default 10240) vs legacy batch-FIFO
  (`--max-pool-size 0`): the leak may be pool-mode-specific (`Error in pool mode` string).
- Repro fast: `--max-pool-size 64`, KV=52, then watch `grep "free pages\|Insufficient" server.log`
  — pages monotonically drop during decode and never recover = confirms leak.

### Decode-KV-offload experiment — TRIED, DOES NOT WORK (result recorded)
Ran with `--host-kv-eviction-watermark 50 --host-kv-watermark 50` (aggressive host-KV eviction)
+ `--max-pool-size 128`, KV=52GB. SAME crash: `Error in pool mode: Insufficient free pages:
need 539, have 44` → SIGSEGV. CONCLUSION: host-KV eviction does NOT relieve the GPU-KV DECODE
page exhaustion. This matches worker.py:7175 ("V4 decode reads prompt KV from the GPU coordinator
pools ONLY, no host->GPU upload path") — the host-offload/eviction machinery does NOT manage V4
decode GPU pages, so it cannot free them. This further localizes the bug to the GPU-KV
coordinator's own decode-step page allocation/release (DeepSeekV4KVCoordinator + GPU_KV_Buffer),
independent of host KV.

### `--max-pool-size 0` (legacy batch-FIFO) — TRIED. Better but still leak-bound.
- Legacy mode admits the WHOLE batch wave (2081 seqs / 76,182 pages > 13,979 page-units) and
  hits the same page exhaustion — BUT the error is `Error during inference` (graceful), the
  server SURVIVES (no SIGSEGV, stays healthy). So legacy mode is strictly more robust than pool
  mode for this bug. The full-12k batch still fails to produce results because the wave is too big.
- Even `--max_prompts 100` (100 seqs = 3,321 pages prefill, 6% of capacity) FAILS during decode
  with page exhaustion — and there ARE `EOS` markers, so some seqs finish, but pages are not
  reclaimed → confirms a **page leak/non-release during decode**, NOT concurrency. 100 seqs
  cannot legitimately drain 55,916 pages.

### ✅✅ BOUNDED RUN WORKS — FIRST REAL ACCURACY NUMBER
`--max_prompts 20` (legacy mode, KV=52GB, grouped MoE) **COMPLETED end-to-end**:
```
Batch completed: completed
Total: 20  Correct: 14  Accuracy: 70.00%
```
This proves the full pipeline (deadlock fixes + grouped MoE + persistence fix) is FUNCTIONALLY
CORRECT and produces gradeable MMLU-Pro answers at a plausible accuracy. Took ~12 min for 20
reasoning-model prompts × up to 1024 tokens. GPUs 84-96% util during decode.

### PATH TO FULL 12k (recommended for next session)
The page leak caps how many prompts decode before exhaustion (~somewhere between 20 (works) and
100 (fails)). Two options:
1. **Chunk the eval**: run `--max_prompts` in slices of ~20-30, restarting the server between
   chunks (or if pages reclaim on batch completion, sequentially). Aggregate accuracy. Slow but
   gets the full number without fixing the leak. (Need to verify pages reclaim after a batch
   completes — the 20-run finished and server went to 0% util, so a follow-up chunk may work
   without restart. UNTESTED.)
2. **Fix the leak** (proper fix): GPU-KV decode page release. See pointers above
   (core/KV_Storage/, GPU_KV_Buffer, DeepSeekV4KVCoordinator). The smoking gun: pages allocated
   per decode step / per admitted seq are not freed on EOS or step completion in BOTH pool and
   legacy modes. host-KV eviction does NOT touch these (V4 decode = GPU-pools-only).

---

## 🟢🟢🟢 SESSION 5 RESULT (2026-06-14) — ENGINE RUNS END-TO-END; A/B = 1/4 EXACT

### What works now (verified live)
The multi-session prefill hang is FIXED and the engine produces real outputs. A/B vs golden
(`compare_ab.py`, world-size 4, grouped MoE, max_output_len 128):
```
[EXACT] tiny-math   golden_len=1   bg_len=1     ← CHARACTER-EXACT MATCH ✓
[DIFF]  identity     golden_len=566 bg_len=536   diverges at char 10
[DIFF]  haiku        golden_len=82  bg_len=79    diverges at char 2
[DIFF]  sys-math     golden_len=184 bg_len=201   diverges at char 51
exact match: 1/4
```
tiny-math (single greedy token "4") is byte-exact. Multi-token cases diverge early — this is the
QAT/activation-quant NUMERIC drift the prior sessions predicted, NOT a hang or crash.

### Three fixes landed this session (all uncommitted, in working tree)
1. **Prefill collective deadlock (batchgen_worker.py)** — vocab-parallel embedding/lm_head are
   TP collectives; 0-seq ranks must join them. Restored empty-rank participation + made it
   microbatch-count-aware (`all_reduce(MAX)` on local microbatch count). New helpers
   `_needs_vocab_parallel_prefill_participation()` / `_run_empty_vocab_parallel_prefill_collectives()`.
   ALSO: `_init_gpu_kv_with_actual_size()` does a `dist.broadcast` — gated it to also run for
   empty deepseek ranks (was the 2nd-order desync).
2. **Decode 28x speedup** — run with `BATCHGEN_V4_GROUPED_MOE=1` (resident experts 34.27GiB/rank;
   use `--gpu-memory-frac 0.4` so KV + resident experts fit). moe_expert_loop 4893→171 ms/tok.
3. **Grouped-MoE prefill persistence bug (Parallel_Strategy_Manager.py)** — `configure_decoding`
   mutates the SHARED `weight_copy_task` to mark owned experts persistent; that leaked into the
   next `configure_prefill` (which streams all 256 experts) → "expert weights are not loaded".
   FIX: snapshot pristine `_pristine_routed_expert_task` at init; `configure_prefill` resets the
   routed-expert task to pristine (always-streamed); `_mark_local_experts_persistent` rebuilds
   from pristine (idempotent). Also removed a stray duplicate `set_runtime_tensors` (rank-0-only,
   last-expert) dead code. Verified: 0 "expert weights not loaded" across repeated batches.

### EXACT working launch (copy-paste)
```bash
docker rm -f bg-v4; rm -f /dev/shm/batchgen_host_kv_cache /dev/shm/shm_*   # shm files are leyang-owned now, deletable w/o sudo
docker run -d --name bg-v4 --gpus '"device=0,1,2,3"' --ipc=host --shm-size=400g \
  -v /mnt/raid0nvme0/leyang/batchgen:/work \
  -v /mnt/raid0nvme0/public/huggingface:/mnt/raid0nvme0/public/huggingface \
  -v /mnt/raid0nvme0/leyang/v4flash_converted:/mnt/raid0nvme0/leyang/v4flash_converted \
  -v /mnt/raid0nvme0/leyang/v4flash_official:/mnt/raid0nvme0/leyang/v4flash_official \
  -e PYTHONPATH=/work:/work/tools -e BATCHGEN_KERNELS_DEV=1 -e HF_HUB_OFFLINE=1 \
  -e BATCHGEN_V4_GROUPED_MOE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e BATCHGEN_NCCL_TIMEOUT_SEC=86400 \
  -w /work batchgen:v4-kernels-user \
  python -m batchgen.launch_http_server --model deepseek-ai/DeepSeek-V4-Flash \
    --converted-ckpt-dir /mnt/raid0nvme0/leyang/v4flash_converted \
    --cache-dir /mnt/raid0nvme0/public/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136 \
    --kv-dtype fp8 --host-kv-cache-size 100 --gpu-arch blackwell --gpu-memory-frac 0.4 \
    --dist-init-addr localhost:12461 --world-size 4 --listen-port 10931 --watchdog-timeout 86400
# ~190s to ready. A/B (golden mounted inside container):
docker exec bg-v4 bash -c "cd /mnt/raid0nvme0/leyang/v4flash_official/results/ab_small && \
  python compare_ab.py --golden golden.jsonl --base-url http://127.0.0.1:10931 --max-output-len 128"
```
NOTE: first request JIT-compiles the slot kernel (~slow); subsequent are fast. py-spy works
inside the container with `docker exec --privileged -u 0 bg-v4 /root/moegen/.venv/bin/py-spy dump --pid <pid>`.

### NEXT PROBLEM: multi-token character-exact (numeric drift)
tiny-math matches; longer greedy generations diverge within a few chars. Root cause is the
documented QAT activation-quant / lm_head-fp32 / kernel-numeric path (see Session 3 notes +
`.sisyphus/V4-EXACT-MATCH-STATUS.md`). Next steps to chase exact match:
- Try `BATCHGEN_GLM5_LMHEAD_FP32=1` (lm_head fp32 cast) and compare first-token logits.
- Compare sm120 MLA/MoE kernel numerics vs a torch reference (`BATCHGEN_V4_MLA_TORCH=1` control).
- DIVTRACE A/B vs official dump_ref_acts.py per `.sisyphus/PREFILL-ATTN-ROOTCAUSE.md`.
- Verify greedy/temperature handling: compare_ab sends temperature=None — confirm that maps to
  argmax greedy in http_server, matching the golden's greedy decode.

### Commit note
The 3 fixes (worker deadlock, PSM persistence) are genuine bug fixes, uncommitted. Consider an
atomic commit once you decide grouped-MoE default. NOTE grouped MoE is still env-gated default 0;
these fixes make it CORRECT when enabled. The deadlock fix is needed regardless of grouped MoE.

---

## 🟩🟩🟩 SESSION 5 (2026-06-14) — TRUE ROOT CAUSE FOUND: vocab-parallel embedding collective deadlock

### TL;DR (this supersedes the Session 3/4 "wgmma/sm120" theory for the HANG)
The bs=1 prefill hang is a **distributed collective deadlock**, NOT the MXFP4/wgmma kernel issue.
Confirmed via **py-spy stack dumps of all 4 ranks** (py-spy works as root INSIDE the container):
- **Rank 0** (owns the 1 sequence): blocked in `vocab_parallel_embedding` (model.py:138
  `dist.all_gather`), called from `prefill_prepacked` (batchgen_worker.py:9174).
- **Ranks 1-3** (0 sequences): blocked at `batchgen_worker.py:7231` `dist.barrier()`.
- all_gather (rank0) vs barrier (ranks1-3) = permanent deadlock. GPUs 100%/~100W spin.

### Mechanism (exact)
- DeepSeek-V4 shards `embed_tokens.weight` AND `lm_head.weight` by VOCAB rows across all 4 TP
  ranks. So `vocab_parallel_embedding()` / `vocab_parallel_lm_head()` are **collectives**
  (all_gather row_counts -> all_gather ids -> all_reduce embeddings). EVERY rank must call them.
- Prefill scheduling is data-parallel: bs=1 -> only rank 0 gets the seq; ranks 1-3 get 0 seqs.
- **The smoking gun is an UNCOMMITTED change**: `batchgen_worker.py:8951-8952` `if not batch: return`
  at the TOP of `prefill_prepacked`. This makes 0-seq ranks early-return BEFORE the embedding
  all_gather at line 9174. Rank 0 still calls the collective -> deadlock.
- There IS a pre-existing partial mechanism `needs_empty_vocab_parallel_lm_head`
  (batchgen_worker.py:7140-7161) that lets 0-seq ranks ENTER the prefill body to join the lm_head
  gather - but the new `if not batch: return` short-circuits it before ANY collective runs. The
  two mechanisms now conflict.

### Why Jun-10 sanity worked but bs=1 hangs
Sanity used 4 prompts -> ~1 seq/rank -> all ranks entered prefill and called the collectives
together. bs=1 -> asymmetric (only rank0) -> deadlock. (Sanity also produced GIBBERISH output -
that is a SEPARATE numerics problem, not the hang.)

### Key facts established this session
- `BATCHGEN_V4_SPARSE_PREFILL=0` does NOT fix it (hang is upstream of attention, in embedding).
- Prebuilt MoE `.so` (`_C_expert_mxfp4_wgmma`, `_C_grouped_mxfp4_wgmma`) are **sm_90/sm_90a ONLY**
  (cuobjdump confirmed) - Session 3/4 "copy prebuilt .so" path (a) is REFUTED for Blackwell.
  `_C_v4_attn` has NO prebuilt .so anywhere -> always JITs (this is normal, not the hang).
- Host GPU = RTX PRO 6000 Blackwell, compute_cap 12.0 (sm_120).
- py-spy IS at `/root/moegen/.venv/bin/py-spy`; works with `docker exec --privileged -u 0`.

### THE FIX — IMPLEMENTED & VERIFIED (prefill deadlock GONE)
Two distinct rank-asymmetric collective desyncs were fixed (both in `batchgen_worker.py`):

1. **embedding/lm_head collective (prefill_prepacked).** Restored the empty-rank participation
   handler (the uncommitted edits had reverted it to `if not batch: return`) AND made it
   **microbatch-count-aware**: active ranks `all_reduce(MAX)` their local microbatch count;
   empty ranks read the same global count and call
   `_run_empty_vocab_parallel_prefill_collectives()` that many times. New helpers:
   `_needs_vocab_parallel_prefill_participation()` and
   `_run_empty_vocab_parallel_prefill_collectives()` (defined just above `prefill_prepacked`).
   Active loop now iterates `range(global_mb_count)` and runs the empty collective for
   `batch_idx >= local_mb_count`.

2. **`_init_gpu_kv_with_actual_size()` broadcast (the second-order desync).** That function does
   `dist.broadcast(size_tensor, src=0)` but was called ONLY by ranks WITH sequences (gate at
   worker.py ~7180 `if local_prefill_indices and ...`). Empty ranks skipped it -> NCCL matched
   rank0's broadcast against empty ranks' mb_count all_reduce -> hang. FIX: gate now
   `if (local_prefill_indices or needs_empty_vocab_parallel_lm_head) and ...` so empty ranks
   also enter and join the broadcast.

Oracle (bg_4f787c3d / ses_13a150f5effezSyPbU4AEUyjj7) confirmed: decoder layers do NOT contain
collectives that 0-seq ranks must join for V4-Flash prefill (MoE prefill world_size=1/no EP;
attention returns via sparse/prefill-DP path). So 0-seq ranks only need embedding + lm_head.

**VERIFIED**: bs=1 tiny-math request now PASSES prefill (`Prepacked Prefill: 100%|...| 1/1
[01:13]`) and entered the DECODE phase — all 4 ranks cycle through the layer
load_weights->forward->free streaming loop (confirmed advancing via repeated py-spy). The
22-hour-style prefill hang is GONE.

### NEW remaining issue (performance, NOT a deadlock)
Cold single-token DECODE is pathologically slow (>13 min and counting for 1 token): each of 43
layers streams MoE expert weights HtoD then frees them (`load_weights`/`free_weights` loop in
wrappers.py:814-824). Not hung (layers advance), but far too slow. This is the documented
per-expert MoE decode path. Next: investigate decode weight-streaming / per-expert MoE perf
(separate from the now-fixed correctness bug). The A/B char-exact comparison still pending a
returned token.

### Non-prepack prefill() (worker.py:8718) NOT fixed
Its committed empty handler calls the collectives ONCE (not microbatch-count-aware). Only matters
if `enable_prepack=False` (default True). Apply the same global-count pattern there if non-prepack
is ever used with vocab-sharded V4.

### Current live state
- docker container `bg-v4` (batchgen:v4-kernels-user) was RUNNING and HUNG on the bs=1 test;
  may still be up. Kill before relaunch: `docker rm -f bg-v4`.
- Launch config that reaches the hang (server starts fine, ~184s): see SESSION 3 FINAL docker
  run, image `batchgen:v4-kernels-user`, dist-init port 12456, listen 10931, world-size 4.
- /dev/shm: leaked regions need host `sudo rm -f /dev/shm/batchgen_host_kv_cache /dev/shm/shm_*`
  (container/privileged root CANNOT delete; only host sudo). Clear before each launch.

---

## 🟦 SESSION 4 HANDOFF (2026-06-14) — START HERE

### TL;DR
The remaining todo "Re-run comparison and verify character-level match" was **NOT completed**
this session. No inference was executed and no comparison result (MATCH/MISMATCH) was produced.
The session looped on the comparison step without ever clearing the real blocker. This section
records the **verified live state** so the next session starts from facts, not narrative.

### Verified live state (checked this session, not assumed)
- **No server running.** Port 10931 NOT listening. No `launch_http_server` process.
- **No docker container running** (`docker ps` empty).
- **Both images present locally:** `batchgen:v4-kernels` and `batchgen:v4-kernels-user`.
- **⚠️ GPU ANOMALY — investigate FIRST:** `nvidia-smi` shows **GPU 1,2,3 at 100% util / ~99W
  but 0 MiB memory used and NO owning process/container.** This is the documented spin-wait
  signature (100% util, low power) but with zero allocated memory and nothing visible holding
  the GPUs. Before any launch, determine what is pinning GPUs 1–3 (could be a leaked kernel from
  a prior killed container, or another user's job). GPU 0/4/5 are idle. Do NOT assume the GPUs
  are free.
- **Git:** branch `feature/deepseek-v4-kernel-integration`. **9 modified files + several
  untracked** are UNCOMMITTED working-tree changes (see `git status`); these contain in-progress
  V4 fixes and MUST be present for any repro. Key modified: `batchgen_worker.py`,
  `models/deepseek/deepseekv4_flash/{model.py,wrappers.py}`,
  `attention/dsa/v4_flashmla_adapter.py`, `ckpt_converter/metadata_loader.py`,
  `Parallel_Strategy_Manager.py`, `server_worker_main_loop.py`. New untracked:
  `models/deepseek/deepseekv4_flash/v4_prefill_sparse.py`,
  `tests/integration/test_v4_{linear_numerics,prefill_sparse}_parity.py`,
  `tools/v4_{acc_eval,divtrace,sanity}_blackwell.sh`.
- The `.sisyphus/*.md` files referenced lower in this doc were **not found** in the working tree
  this session (`.sisyphus/` glob returned nothing). Treat their quoted content below as
  historical memory only; re-derive status from code + a real run.

### THE REAL BLOCKER (unchanged from Session 3 FINAL — read that section below)
Prefill hangs because the MXFP4 MoE expert kernels use **`wgmma.wait_group`**, a Hopper
(sm_90a) instruction that **ptxas refuses to assemble for sm_120 (Blackwell)**. When AOT
import of the prebuilt `.so` is shadowed by the host source tree, JIT fallback fails → MoE
falls back to a per-expert loop that **wedges the multi-process prefill**. See
"SESSION 3 FINAL" below for the exact ptxas error and the two disambiguation paths (a)/(b).

### The single cheapest next experiment (do this, in order)
1. **Clear the stuck GPUs 1–3** and confirm they return to idle (~10–30W, 0% util) before
   anything else. If a hidden process owns them, find and kill it (or ask the user — may be
   another user's job; do not kill blindly).
2. **Run the docker server WITHOUT shadowing the prebuilt sm120 kernels.** Per Session 3 FINAL:
   either drop `BATCHGEN_KERNELS_DEV=1` / don't put `/work` first on PYTHONPATH for
   `batchgen_kernels`, OR `cp` the prebuilt `_C_*wgmma*.so` from the image venv
   (`/root/moegen/.venv/lib/python3.11/site-packages/batchgen_kernels/moe/*.so`) into
   `/work/batchgen_kernels/moe/` so AOT import succeeds from the host tree (host python +
   working prebuilt kernels, no JIT). Exact `docker run` is in "SESSION 3 FINAL" below.
3. **One-token smoke test** (run INSIDE the container; golden file isn't mounted) — tiny-math
   prompt, expect token `'4'` in seconds:
   ```bash
   docker exec bg-v4 python -c "import requests,time; \
   p='<｜begin▁of▁sentence｜><｜User｜>What is 2+2? Answer briefly.<｜Assistant｜></think>'; \
   t=time.time(); r=requests.post('http://127.0.0.1:10931/v1/inference', \
   json={'prompts':[p],'max_output_len':1,'temperature':0},timeout=400); \
   print(r.status_code,'%.1fs'%(time.time()-t), r.text[:300])"
   ```
4. **Only after a token returns**, run the A/B comparison vs golden
   (`/mnt/raid0nvme0/leyang/v4flash_official/results/ab_small/golden.jsonl`) and verify
   character-level match. That is what closes the open todo.

### Hard truths for the next agent
- Do NOT mark "Re-run comparison and verify character-level match" complete until a real
  `MATCH`/`MISMATCH` is observed from an actual run. "Timeout/NO_OUTPUT" is NOT completion.
- Do NOT retry the comparison script repeatedly — it cannot succeed while prefill hangs. Fix
  the kernel/runtime path first.
- Do NOT use bare-metal conda env — it lacks `tilelang` and hangs at prefill layer 0. Use the
  docker `batchgen:v4-kernels(-user)` runtime.
- Verify the request/response schema of `/v1/inference` against
  `batchgen/server/http_server.py` before trusting field names in any `/tmp/compare_*.py`.

### Open consult
- Oracle session `ses_13a8ebec1ffe90Q43bJIR3sUn8` (ws=1 mp4 OOB analysis; ws=4 hang is a genuine
  layer-0 prefill spin, not a masked OOB). Continue that session if re-consulting.

---

## Goal (unchanged)
Achieve **character-exact output parity** between the real batchgen DeepSeek-V4-Flash
inference server and the official DeepSeek golden outputs.
Start small: **bs=1, max_output_len=1, temperature=0 (greedy)**, then scale.
Must use the **real batchgen server** (NOT the mock on port 18031).

User constraints (verbatim):
- "use the output text as the golden standard, our batchgen engine should match each characters"
- "start from small scale first, use exact generation pipeline and hyperparameters"

---

## 🟩🟩 SESSION 3 FINAL — TRUE ROOT CAUSE OF THE PREFILL HANG (ptxas / wgmma on sm120)

Ran the proper **docker `batchgen:v4-kernels`** server (world-size 4, GPUs 0-3). Server
started healthy in-container (uses the original `/root/moegen/.venv`). Sent bs=1 1-token
request → **SAME HANG** at "Prepacked Prefill: 0%", GPUs 100%/~100W. So the hang is NOT a
bare-metal artifact. Then I read the container logs and found the smoking gun:

```
WARNING:batchgen_kernels:[DEV] AOT import failed for batchgen_kernels.moe._C_expert_mxfp4_wgmma, attempting JIT...
ptxas .../expert_mxfp4_wgmma.ptx, line 4489; error : Instruction 'wgmma.wait_group' not supported on .target 'sm_120'
ptxas fatal : Ptx assembly aborted due to errors
WARNING:root:Failed to load WGMMA fused MoE kernels: Error building extension '_C_expert_mxfp4_wgmma'
(same for _C_grouped_mxfp4_wgmma)
```

### ROOT CAUSE (definitive, hardware-level)
- The MoE expert kernels `batchgen_kernels/src/moe/expert_mxfp4_wgmma.cu` and
  `grouped_mxfp4_wgmma.cu` use **`wgmma.wait_group`** — a **Hopper (sm_90a) warpgroup-MMA**
  PTX instruction that **does NOT exist on Blackwell sm_120** (RTX PRO 6000). ptxas refuses
  to assemble it. The JIT fallback in `batchgen_kernels/__init__.py:70-83` naively rewrites
  `sm_90a`→`sm_120` but the WGMMA instruction itself is unsupported, so it can never build.
- When these MoE kernels fail to load, the V4-Flash MoE silently falls back to a
  **per-expert Python loop**, which is exactly the path documented to **wedge/hang the
  multi-process prefill** (V4-EXACT-MATCH-STATUS.md L50-53: "per-expert tilelang launches
  wedge the multi-process loop; server hangs at 100% GPU with no progress").

### WHY my docker run hit this (and how the image is "supposed" to work)
- The image ships PREBUILT kernels at
  `/root/moegen/.venv/lib/python3.11/site-packages/batchgen_kernels/moe/*.so`.
- BUT I ran with `-v /mnt/.../batchgen:/work -e PYTHONPATH=/work -e BATCHGEN_KERNELS_DEV=1`.
  That makes Python import `batchgen_kernels` from the **host source tree `/work/batchgen_kernels`**,
  which has NO compiled `_C_*wgmma*.so` (confirmed: `ls /work/batchgen_kernels/moe/_C_*wgmma*`
  → none). So AOT import fails → DEV mode triggers the broken sm120 JIT → ptxas fatal → MoE
  falls back to the hanging per-expert loop.
- TWO open possibilities for next session (MUST disambiguate):
  (a) The image's prebuilt `.so` ARE valid Blackwell sm120 kernels (built without WGMMA, via a
      different codegen) and the ONLY problem is my bind-mount/PYTHONPATH/DEV shadowing them.
      → FIX: run the image WITHOUT overlaying `batchgen_kernels` (don't put /work first on
      PYTHONPATH for that package, or don't set BATCHGEN_KERNELS_DEV, or bind-mount only the
      `batchgen/` subdir not the whole repo). Then the prebuilt sm120 MoE loads and prefill
      should proceed → tiny-math should return '4'.
  (b) The WGMMA MoE kernels are Hopper-only and there is NO working sm120 prebuilt MoE in the
      image either → then prefill MoE on Blackwell needs the Triton sm120 grouped path
      (`batchgen/moe/v4_slot_moe_sm120.py`, gated by env `BATCHGEN_V4_GROUPED_MOE=1`, but it's
      currently wired only for EP-decode `_run_owned_experts_grouped`, NOT prefill, and caps at
      512 tokens). Real work = route prefill MoE to the sm120 Triton path (or another
      non-WGMMA fp4 GEMM). This is genuine kernel-porting, the actual blocker.

### EXACT NEXT EXPERIMENT (cheapest, do first)
Re-run the docker server WITHOUT shadowing the prebuilt kernels:
```bash
docker run -d --name bg-v4 --gpus '"device=0,1,2,3"' --ipc=host --shm-size=400g \
  -v /mnt/raid0nvme0/leyang/batchgen:/work \
  -v /mnt/raid0nvme0/public/huggingface:/mnt/raid0nvme0/public/huggingface \
  -v /mnt/raid0nvme0/leyang/v4flash_converted:/mnt/raid0nvme0/leyang/v4flash_converted \
  -e HF_HUB_OFFLINE=1 -w /work batchgen:v4-kernels \
  python -m batchgen.launch_http_server --model deepseek-ai/DeepSeek-V4-Flash \
    --converted-ckpt-dir /mnt/raid0nvme0/leyang/v4flash_converted \
    --cache-dir /mnt/.../snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136 \
    --kv-dtype fp8 --host-kv-cache-size 100 --gpu-arch blackwell --gpu-memory-frac 0.65 \
    --dist-init-addr localhost:12455 --world-size 4 --listen-port 10931 --watchdog-timeout 86400
```
KEY CHANGES vs my run: **drop `BATCHGEN_KERNELS_DEV=1` and drop `PYTHONPATH=/work` for the
kernels** so `batchgen_kernels` resolves to the installed prebuilt sm120 package, not host
source. CAVEAT: this also stops `/work` python overrides — but the V4 *model* fixes (sparse
prefill etc.) may be INSIDE the image already (it was built from this branch). Check: does the
image's `/root/moegen/batchgen` differ from host `/work/batchgen`? If host has newer fixes you
need BOTH host `batchgen/` python AND installed prebuilt `batchgen_kernels`. Achieve that by:
mount repo at /work, set `PYTHONPATH=/work` BUT first `pip install -e /work/batchgen_kernels`
is wrong (rebuilds). Instead: `cp` the prebuilt `_C_*wgmma*.so` from the venv site-packages
into `/work/batchgen_kernels/moe/` so AOT import succeeds from the host tree. That gives host
python + working prebuilt kernels with no JIT.

Test command (run INSIDE container, golden file isn't mounted):
```bash
docker exec bg-v4 python -c "import requests,time; \
p='<｜begin▁of▁sentence｜><｜User｜>What is 2+2? Answer briefly.<｜Assistant｜></think>'; \
t=time.time(); r=requests.post('http://127.0.0.1:10931/v1/inference', \
json={'prompts':[p],'max_output_len':1,'temperature':0},timeout=400); \
print(r.status_code, '%.1fs'%(time.time()-t), r.text[:300])"
```
Expected if fixed: returns token '4' in seconds. Golden completion for tiny-math = '4'.

### NETWORKING NOTE
Container uses default bridge net; port 10931 is NOT published to host. Either add `-p
10931:10931` (or `--network host`) to curl from host, OR `docker exec` into the container to
hit 127.0.0.1:10931 (what I did).

---

## 🟥🟥 SESSION 3 — env discovery (superseded by the FINAL section above, kept for context)

**I was running in the WRONG ENVIRONMENT the entire time.** The whole bare-metal
conda-env effort (installing uvicorn/ninja/CUDA/multipart/tokenizers, the prefill "hang")
was misguided. Discovered late via `.sisyphus/V4-EXACT-MATCH-STATUS.md` and
`.sisyphus/PREFILL-ATTN-ROOTCAUSE.md`:

### The truth
- DeepSeek-V4-Flash prefill uses **tilelang sparse-attention kernels for sm120/Blackwell**
  (`BATCHGEN_V4_SPARSE_PREFILL=1`, default ON). **`tilelang` is NOT installed in the conda
  env** (`import tilelang` → ModuleNotFoundError). That is why prefill HANGS at layer 0 at
  100% GPU / ~100W (the tilelang kernel path can't run / JIT-wedges). My "hang" exactly
  matches the documented symptom in V4-EXACT-MATCH-STATUS.md lines 50-53.
- **The validated runtime is a DOCKER image: `batchgen:v4-kernels` / `batchgen:v4-kernels-user`**
  (tilelang 0.1.9 + tvm-ffi 0.1.5 + fht, built for sm120). These images EXIST locally
  (`docker images | grep v4-kernels`). Docker works WITHOUT sudo here (`docker ps` ok).
- **The model is ALREADY essentially working.** Per V4-EXACT-MATCH-STATUS.md line 34:
  the `tiny-math` prompt **already generates `'4'` + EOS = exact golden match** in the
  proper docker env. The REAL open problem is char-exact match over 128 tokens (drift from
  QAT activation-quant numerics), NOT "does the server run at all."

### CORRECT REPRO PATH (do this; from `.sisyphus/HANDOFF-blackwell-v4-mmlu.md` §TL;DR)
```bash
cd /mnt/raid0nvme0/leyang/batchgen
# clean any bare-metal leftovers first:
pkill -9 -f launch_http_server; rm -f /dev/shm/batchgen_host_kv_cache /dev/shm/shm_*
# start server INSIDE the v4-kernels docker (sm120 kernels live there):
docker run -d --name bg-v4 --gpus '"device=0,1,2,3"' --ipc=host --shm-size=400g \
  -v /mnt/raid0nvme0/leyang/batchgen:/work \
  -v /mnt/raid0nvme0/public/huggingface:/mnt/raid0nvme0/public/huggingface \
  -v /mnt/raid0nvme0/leyang/v4flash_converted:/mnt/raid0nvme0/leyang/v4flash_converted \
  -e PYTHONPATH=/work:/work/tools -e BATCHGEN_KERNELS_DEV=1 -e HF_HUB_OFFLINE=1 \
  -w /work batchgen:v4-kernels \
  python -m batchgen.launch_http_server --model deepseek-ai/DeepSeek-V4-Flash \
    --converted-ckpt-dir /mnt/raid0nvme0/leyang/v4flash_converted \
    --cache-dir /mnt/raid0nvme0/public/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136 \
    --kv-dtype fp8 --host-kv-cache-size 100 --gpu-arch blackwell --gpu-memory-frac 0.65 \
    --dist-init-addr localhost:12455 --world-size 4 --listen-port 10931 --watchdog-timeout 86400
# (verify exact flags/mounts against .sisyphus/HANDOFF-blackwell-v4-mmlu.md and
#  PREFILL-ATTN-ROOTCAUSE.md §Validation loop — there may be uncommitted working-tree
#  edits that must be present; check `git status --short`.)
```
Then A/B:
```bash
python /mnt/raid0nvme0/leyang/v4flash_official/results/ab_small/compare_ab.py \
  --golden /mnt/raid0nvme0/leyang/v4flash_official/results/ab_small/golden.jsonl \
  --base-url http://127.0.0.1:10931
```

### KEY .sisyphus DOCS (the real project memory — READ THESE, they supersede my notes)
- `.sisyphus/V4-EXACT-MATCH-STATUS.md` (2026-06-13) — current status: tiny-math matches;
  128-tok char-exact blocked on QAT linear (act fp8 quant). Path to exact match in §"Path".
- `.sisyphus/PREFILL-ATTN-ROOTCAUSE.md` (2026-06-12) — prefill attention bug history +
  validation loop (DIVTRACE A/B vs official dump_ref_acts.py).
- `.sisyphus/HANDOFF-blackwell-v4-mmlu.md` — exact docker run + Blackwell sm120 notes +
  3 uncommitted decode fixes that must be present.
- `.sisyphus/RESUME-v4-repro.md` — full history.

### THE REAL REMAINING WORK (not "make it run" — it runs in docker)
Per V4-EXACT-MATCH-STATUS.md §"Path to exact match":
1. Batch the QAT expert path per layer (act_quant once/layer, grouped fp4 GEMM over owned
   experts) — the per-expert tilelang loop is what wedges the server, so enabling
   `BATCHGEN_V4_QAT_LINEAR=1` naively re-creates the hang. Must batch + pre-warm tilelang JIT.
2. Pre-warm tilelang JIT cache for all (N,K) shapes at server start (before worker fork).
3. Verify hash-routing tid2eid int64-vs-int32 gather + lm_head fp32
   (`BATCHGEN_GLM5_LMHEAD_FP32=1`) vs official ParallelHead.float().

### Bottom line for the todo "Re-run comparison and verify character-level match"
- bs=1 single-token (tiny-math → '4') is ALREADY a known exact match in the docker env.
- To actually re-verify: run the docker server above + compare_ab.py. Do NOT keep trying
  bare-metal — it lacks tilelang and will hang forever at prefill layer 0.

---

## 🟢 SESSION 3 UPDATE (2026-06-14 ~09:30) — HANG ISOLATED TO LAYER-0 PREFILL COMPUTE

Decisive new experiments this session (server fully runs now; deps + shm + GPU all OK):

### Experiment 1 — world-size=1 (GPU0 only): CRASHES with a real CUDA assert
- `/dev/shm` had to be cleared first (leaked 320G+320G+101G regions from killed servers;
  `rm /dev/shm/shm_* /dev/shm/batchgen_host_kv_cache` — safe when no server running).
- ws=1 server started healthy. bs=1 request → **device-side assert**:
  ```
  /pytorch/aten/src/ATen/native/cuda/Indexing.cu:1587: indexSelectSmallIndex:
    Assertion `srcIndex < srcSelectDimSize` failed.   (many threads)
  CUDA error 710 at HtoD_Engine.cu:237 blocking_copy_: device-side assert triggered → SIGABRT
  ```
  Fires IMMEDIATELY at prefill start (first layer index_select).
- ROOT CAUSE of THIS crash: **the checkpoint is mp4 (4-way model-parallel sharded:
  `model{0,1,2,3}-mp4.bin`).** Running ws=1 loads only shard 0 → ~64 of 256 routed experts,
  but the per-layer routing table `layers.N.ffn.gate.tid2eid` (int64, shape [129280, 6] =
  vocab×experts_per_tok) still holds GLOBAL expert ids 0..255 → index_select into a local
  64-expert table with id≥64 → OOB. **=> ws=1 is INVALID for an mp4 ckpt. Do not pursue ws=1
  unless the engine supports merging mp shards (it doesn't appear to).** vocab_size=129280 and
  max prompt token id=128822, so this is NOT an embedding-vocab problem — it's expert-shard.

### Experiment 2 — world-size=4 + CUDA_LAUNCH_BLOCKING=1 (the INTENDED config): HANGS, NO assert
- Env: `CUDA_LAUNCH_BLOCKING=1 TORCH_SHOW_CPP_STACKTRACES=1 NCCL_DEBUG=WARN
  TORCH_NCCL_ASYNC_ERROR_HANDLING=1`. Port 10933.
- Server healthy. bs=1 request → **hangs at "Prepacked Prefill: 0%"** for 5+ min.
  `grep -c Assertion|Indexing.cu|CUDA error` in log = **0**. GPUs 0-3 100% util / ~100W
  (spin), workers in R state burning ~3 cores each.
- **KEY CONCLUSION: ws=4 does NOT reproduce the ws=1 OOB assert.** Even with
  CUDA_LAUNCH_BLOCKING=1 (which makes any bad kernel fail synchronously at its launch
  site), there is NO assert. So Oracle's "hang = NCCL-waiting-on-a-crashed-peer" theory is
  **REFUTED**. This is a GENUINE hang/spin in the layer-0 prefill compute, not a masked OOB.

### WHERE the hang is (code path, narrowed)
`batchgen/batchgen_worker.py` ~line 9082-9206, the `with torch.inference_mode():` prepacked
prefill loop. Sequence per micro-batch: `vocab_parallel_embedding` (9174) → reshape + V4
hyper-connection expand (9188) → **`for layer_idx, decoder_layer in enumerate(self.model.model.layers): decoder_layer(...)` (9195-9206)**. The tqdm bar never advances past 0%, so it
hangs INSIDE the first `decoder_layer()` call (layer 0): MLA attention or MoE expert
dispatch/gather, or a host→device weight-stream wait (HtoD_Engine) that never completes.
The V4-Flash decoder layer + MoE wrappers live in:
- `batchgen/models/deepseek/deepseekv4_flash/model.py` (tid2eid at L1467; vocab_parallel_embedding/lm_head)
- `batchgen/models/deepseek/deepseekv4_flash/wrappers.py`
- `batchgen/attention/mla/fa3_backend.py`

### DIAGNOSTIC CONSTRAINTS (important for next session)
- **No sudo** (password required). `/proc/sys/kernel/yama/ptrace_scope = 1` → **py-spy/gdb
  cannot attach** without sudo. `gdb`, `cuda-gdb`, `compute-sanitizer` NOT installed. `nsys`
  IS at /usr/local/bin/nsys. py-spy installed but needs sudo.
- => The realistic next diagnostic is **add Python-level logging inside the prefill layer
  loop** (print rank/layer + a `torch.cuda.synchronize()` before/after each decoder_layer and
  each sub-step) to find the exact op in layer 0 that never returns. Insert around
  batchgen_worker.py:9195-9206. Then rerun ws=4 and watch which log line is last.
- Alternative: get the user to (a) enable sudo / lower ptrace_scope so py-spy works, or
  (b) provide access to the ORIGINAL `/root/moegen/.venv` to test env-parity. The env theory
  is still open: we rebuilt core_engine via JIT against conda torch (/home/leyang/.local),
  NOT the original venv. A wrong-ABI core_engine could plausibly deadlock in the C++ HtoD/
  attention path. Testing in the original venv is the cleanest way to rule this in/out.

### Oracle consult (session_id ses_13a8ebec1ffe90Q43bJIR3sUn8) summary
Confirmed ws=1 mp4 explanation; said ws=1 does NOT prove ws=4 has same bug (correct — exp 2
refuted it). Prioritized plan: surface ws=4 failure loudly (done — it hangs, no assert), then
add bounds/sync logging around the failing op; only after locating it, test env-parity in the
original venv; don't clamp/mod expert ids. Since ws=4 shows NO assert, follow Oracle's branch
#7: investigate the TRUE hang (host stacks / per-layer sync logging), py-spy only to
distinguish "blocked in NCCL" vs "stuck in scheduler/compute".

### NEXT ACTIONS (priority order)
1. Add per-layer + per-substep logging with torch.cuda.synchronize() in the prefill loop
   (batchgen_worker.py ~9195). Rerun ws=4, see the last-printed line → exact hanging op.
2. If it's MoE: inspect expert dispatch/all-to-all in V4-Flash wrappers for a collective that
   deadlocks with a single 14-token sequence on rank 0 (ranks 1-3 have 0 tokens).
3. If it's HtoD weight streaming: inspect HtoD_Engine wait/copy for layer-0 expert weights.
4. In parallel, ask user about: sudo/ptrace for py-spy, AND the original /root/moegen/.venv
   working launch command (did bs=1 EVER work there? same world-size?).

---

## 🔴 BREAKING UPDATE (2026-06-14 09:03) — SERVER RUNS, BUT PREFILL HANGS

The full dependency chain is fixed and **the server now starts and serves**
(`/health` → healthy, all 4 workers entered main loop, "End-to-end server ready in 189.62s").
BUT the first real inference **hangs in prefill and never returns**.

### Exact symptom
- Request: `POST /v1/inference {"prompts":["<tiny-math prompt>"],"max_output_len":1}`
  (golden id `tiny-math`, prompt "What is 2+2?", golden completion `"4"`, single token)
- Server logs progress through model load → KV coordinator init → prepack, then:
  ```
  Prepacked prefill: 1 micro batches, 14 total tokens ...
  Prepacked Prefill:   0%|          | 0/1 [00:00<?, ?it/s]
  ```
  …and **stops there for 8+ minutes** (a 14-token prefill should be ~instant).
- All 4 worker procs (pgrep -f spawn_main): state **R (running)**, CPU ticks climbing
  (~3 cores each) → busy-spinning, not blocked.
- GPUs 0–3: **100% util but only ~100W** (cap 600W). Low power + full util =
  **spin-wait / deadlock**, NOT real matmul compute (which would draw 400–600W).
- No nvcc/cicc/ptxas/ninja running → it is NOT first-run kernel JIT autotuning.

### Interpretation
This is the SAME hang that produced the original 22-hour stuck batch (GPUs pinned,
no progress). It is a real **prefill-path deadlock** — most likely a cross-rank
synchronization issue (NCCL barrier / collective mismatch) or a host↔device polling
loop in the DeepSeek-V4-Flash prefill (candidates: QAT path, MoE dispatch, MLA
attention, or the c4/c128 KV coordinator). The 4 ranks are waiting on each other or
on a flag that never flips.

### ⭐ KEY NEW CLUE — asymmetric rank work distribution (TP collective deadlock)
From the server log right before the hang:
```
[BatchGenWorker-0] [PREFILL] Selected 1 sequences, per-node pages: [1]
[BatchGenWorker-0] Rank 0 BEFORE prefill (1 seqs)
[BatchGenWorker-1] Rank 1 BEFORE prefill (0 seqs)
[BatchGenWorker-2] Rank 2 BEFORE prefill (0 seqs)
[BatchGenWorker-3] Rank 3 BEFORE prefill (0 seqs)
```
**Rank 0 has the only sequence; ranks 1–3 have 0 seqs.** In a world-size-4
tensor/expert-parallel setup, every rank must participate in each collective
(all-reduce/all-gather for attention; all-to-all for MoE expert dispatch). The most
likely bug: with a single tiny sequence, ranks 1–3 take a code path that skips or
mismatches a collective that rank 0 still issues → permanent spin (100% util, ~100W,
R-state). This is the #1 hypothesis to verify.

GPU power signature confirms it: **100% util but only ~100W of 600W cap** = spinning on
a sync barrier, not doing real GEMM (which would pull 300–600W).

### FASTEST PATH TO GET A GOLDEN RESULT: try --world-size 1
If V4-Flash fits on one GPU (94GB; model load showed ~63GB on rank 0), relaunch with
`--world-size 1`. If it works at ws=1 but hangs at ws=4, that PROVES the distributed
collective deadlock and also unblocks the character-level comparison immediately.
Check docs/server-flags.md + model config for min-GPU support first.

### Worker PIDs (this run): main 4038952 → workers 4041657/58/59/60
### Next debugging actions (HIGH VALUE — do these first next session)
1. Get a Python stack at the hang point. py-spy is installed but needs sudo:
   `sudo env "PATH=$PATH" py-spy dump --pid 4041657` (and the other 3 ranks).
   This will name the exact function/kernel-launch line where each rank is spinning.
2. If sudo unavailable: add `faulthandler.dump_traceback_later()` or send SIGUSR1
   handler into the worker; or set `CUDA_LAUNCH_BLOCKING=1` + `NCCL_DEBUG=INFO`
   and relaunch to see if it's an NCCL collective that hangs.
3. Compare against how the ORIGINAL `/root/moegen/.venv` server ran — if it worked
   there, this is very likely env drift (NCCL / torch / kernel build mismatch in the
   conda env) rather than a model-logic bug. This is the single most important fork:
   **env-drift hang vs genuine code bug.**
4. Check `--gpu-arch blackwell` + sm_120 gencode path: confirm the prefill kernels
   were built/selected for Blackwell correctly (RTX PRO 6000 Blackwell).

### To clear the hang before retrying
`pkill -9 -f launch_http_server` then re-check `nvidia-smi` for freed GPUs 0–3.

---

## ⭐ CURRENT STATE (start here)

**The entire dependency/build blocker chain is now RESOLVED.** The server is NOT
currently running — it just needs to be (re)launched. No infra blockers remain.

- Port 10931: NOT listening
- No `launch_http_server` process alive
- All Python deps verified present: `uvicorn, fastapi, ninja, tokenizers, multipart (python-multipart), transformers` → all `True`
- `core_engine.so` already JIT-compiled successfully (cached; relaunch shows `ninja: no work to do`)

### The root cause that wasted the whole prior session
Every "server exits after startup" failure was just a **missing dependency**, surfaced
one at a time. Final foreground run exposed the last one:
```
ERROR:fastapi: Form data requires "python-multipart" to be installed.
```
That is now installed. **Run the server in FOREGROUND first** to confirm it actually
binds the port and stays alive — do NOT background-launch and then guess.

---

## ✅ Blockers cleared this session (in order)
1. Stuck batch job (`batch_2d04ee...`, hung ~22h) → killed worker PIDs + main server
2. Wrong interpreter: original server ran under `/root/moegen/.venv` (perm denied; sudo needs password) → switched to current `python` (anaconda) + pip-installed deps
3. `ModuleNotFoundError: uvicorn` → `pip install uvicorn fastapi`
4. `ninja not installed` (JIT build) → `pip install ninja`
5. `cuda.h / cuda_runtime_api.h not found` → **export CUDA env** (see below). core_engine then compiled fully (`[21/21] ... core_engine.so`)
6. `ModuleNotFoundError: tokenizers` → `pip install tokenizers`
7. `python-multipart` required by FastAPI → `pip install python-multipart transformers` (DONE)

---

## 🔧 Exact launch command (USE THIS)

CUDA env export is REQUIRED every shell (headers needed for any JIT rebuild):
```bash
export CUDA_HOME=/usr/local/cuda
export CPATH=$CUDA_HOME/include:$CPATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# FOREGROUND first to confirm it stays alive:
python -m batchgen.launch_http_server \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --converted-ckpt-dir /mnt/raid0nvme0/leyang/v4flash_converted \
  --cache-dir /mnt/raid0nvme0/public/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136 \
  --kv-dtype fp8 --host-kv-cache-size 100 --gpu-arch blackwell \
  --gpu-memory-frac 0.65 --dist-init-addr localhost:12455 \
  --world-size 4 --listen-port 10931 --watchdog-timeout 86400
```
Once it prints that it's serving / uvicorn running, re-run with `> /tmp/batchgen_server.log 2>&1 &`.

Health check: `curl -sS -m 5 http://127.0.0.1:10931/health` → expect `{"status":"healthy"}`

> NOTE: world-size 4 uses GPUs 0–3. Model load + 4 workers + KV init can take tens of
> seconds. Wait and confirm the process is ALIVE (`ps -ef | grep launch_http_server`)
> before declaring failure. Don't confuse "still loading" with "crashed".

---

## ▶️ NEXT STEPS (the only remaining task)

1. Launch server (above), confirm `/health` healthy AND process stays up.
2. Run the bs=1 / max_output_len=1 / temperature=0 comparison vs golden.
3. Verify **character-level** match. If mismatch → debug order: execution → logits → layers → components (QAT / MoE / attention / KV).
4. Expand to multi-token (e.g. 4 tokens) → watch for KV / RoPE divergence.

### Inference endpoint contract
`POST http://127.0.0.1:10931/v1/inference`
```json
{"prompts": ["<prompt text>"], "max_output_len": 1, "temperature": 0}
```
(Confirm exact request/response schema against
`batchgen/server/http_server.py` `/v1/inference` handler before trusting field names —
the prior comparison scripts in /tmp may use stale fields.)

### Golden data
`/mnt/raid0nvme0/leyang/v4flash_official/results/ab_small/golden.jsonl`

### Comparison script
`/tmp/compare_single_qat.py` — ⚠️ originally pointed at the MOCK server **port 18031**.
Must use the REAL server **port 10931**. Verify/rewrite before use.

---

## ⚠️ Parity caveat (important — discuss with user if mismatch)
The original working server ran under `/root/moegen/.venv`. We are now running under the
**anaconda python** with pip-installed deps + locally JIT-compiled core_engine. Kernels
should be equivalent (same source, same CUDA 13.x, sm_120/Blackwell gencode) but this is
NOT byte-identical to the original env. If a mismatch appears, first rule out env drift
(torch build / kernel differences) before concluding it's a model bug. If exact original
env is required, it needs root access to `/root/moegen/.venv` (sudo needs a password we
don't have).

---

## Key paths
- Repo root: `/mnt/raid0nvme0/leyang/batchgen/`
- HTTP server: `batchgen/server/http_server.py` (`/v1/inference`, `/v1/batches`, `/health`)
- Server log: `/tmp/batchgen_server.log`
- Converted ckpt: `/mnt/raid0nvme0/leyang/v4flash_converted`
- HF snapshot cache: `/mnt/raid0nvme0/public/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136`
- Golden: `/mnt/raid0nvme0/leyang/v4flash_official/results/ab_small/golden.jsonl`
- CUDA: `/usr/local/cuda` (also cuda-12.9, cuda-13.0, cuda-13.1 available)

## GPUs
GPU 0–3 = batchgen workers (world-size 4). GPU 4–5 idle. Confirm 0–3 are free of
stale processes before launch (`nvidia-smi`); kill leftovers if a prior run hung.
