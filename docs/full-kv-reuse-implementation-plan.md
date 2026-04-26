# Page-Level Prefix KV Reuse Implementation Plan

## Source

This plan follows the requirements in GitHub PR #138:

<https://github.com/EfficientMoE/BatchGen/pull/138>

The PR asks for an opt-in, staged, **page-level** prefix KV reuse implementation. The feature must not revive the older token-level prefix-cache approach directly, and it must not implement token-level radix granularity or partial-page sharing.

## Goal

Implement prefix KV reuse in three separately reviewable milestones:

1. Host KV-cache page reuse for host memory efficiency.
2. Prefix-aware prefill that computes and offloads only non-hit suffix pages/tokens.
3. Decode-batch GPU page materialization that loads each shared host page once per rank/batch and lets multiple sequence rows reference the same physical GPU page when safe.

The feature is disabled by default and must be explicitly enabled with:

```text
--enable-prefix-reuse
```

Disabled behavior must preserve the current request-pool and dynamic-host-KV behavior.

## Non-Goals

- Do not implement token-level radix prefix matching.
- Do not share partial pages.
- Do not split a physical KV page between shared and private ownership.
- Do not key the cache by raw prompt text.
- Do not silently fall back to full prefill for exact full-prefix hits.
- Do not enable unsupported DSA/MLA paths unless they are explicitly implemented or explicitly gated.

## Core Model

The reuse unit is a complete KV page.

Token IDs are used only to hash and validate full pages. A prompt can reuse prefix KV only up to the largest contiguous full-page prefix that matches the cache. If a prompt matches 2.5 pages, only the first 2 full pages are shared; the remaining tokens are private suffix work.

### Page-Level Chained Hash

Each full prompt page gets a chained hash key:

```text
PrefixPageKey = (
  model_or_cache_namespace,
  page_size,
  page_index,
  parent_page_hash,
  current_page_token_hash
)
```

The parent hash makes the key prefix-sensitive:

```text
page0_hash = hash(namespace, page_size, 0, ROOT, hash(tokens[0:page_size]))
page1_hash = hash(namespace, page_size, 1, page0_hash, hash(tokens[page_size:2*page_size]))
page2_hash = hash(namespace, page_size, 2, page1_hash, hash(tokens[2*page_size:3*page_size]))
```

Two pages with the same local token content must not be shared if their previous prefix differs. The parent hash prevents that invalid reuse.

### Logical Versus Physical Pages

Every sequence needs a logical page view:

```text
logical pages = shared prefix pages + private suffix/decode pages
```

Physical page ownership is different:

- Shared prefix pages are owned by cache entries and referenced by one or more sequences.
- Private suffix/decode pages are owned by a sequence.
- Decode writes must only target private pages.

The legacy combined page view must remain available for existing callers:

```text
Pages(sequence_id) -> shared_prefix_pages + private_pages
```

## Milestone 0: Feature Gate and Compatibility Shell

Milestone 0 adds the visible feature boundary before changing behavior.

### Tasks

1. Add `--enable-prefix-reuse`, default `False`.
2. Keep existing behavior unchanged when the flag is disabled.
3. Thread the flag through server args, worker args, host KV manager config, and GPU KV manager setup.
4. Add explicit capability checks for model/wrapper support.
5. Gate unsupported DSA/MLA paths with a clear error or a disabled-path fallback before cache matching is attempted.
6. Add logging that distinguishes feature disabled, unsupported model, no full-page hit, host-only hit, suffix-prefill hit, and GPU-sharing hit.

### Acceptance

- Running without `--enable-prefix-reuse` must use the old request-pool and dynamic-host-KV behavior.
- No prefix cache state is created or mutated when the feature is disabled.
- Unsupported paths do not silently enter partial prefix reuse.

## Milestone 1: Host KV-Cache Page Reuse

Milestone 1 only targets host KV page efficiency. Prefill may still compute the full prompt in this milestone, but host rows, ownership, refcounts, release, and stats must already be correct for shared pages.

### 1. Host Prefix Index

Add a page-level prefix index keyed by `PrefixPageKey`.

Suggested records:

```cpp
struct PrefixPageKey {
  uint64_t namespace_hash;
  int32_t page_size;
  int32_t page_index;
  uint64_t parent_page_hash;
  uint64_t page_token_hash;
};

struct PrefixPageEntry {
  PrefixPageKey key;
  uint64_t page_chain_hash;
  int32_t host_page_id;
  int32_t page_size;
  uint64_t token_validation_hash;
  uint32_t pin_count;
};
```

Implementation requirements:

- Lookup walks prompt tokens in full-page chunks only.
- Lookup stops at the first missing full page.
- The returned hit length is always `matched_full_pages * page_size`.
- Partial final prompt pages are never inserted as shared prefix entries.
- Cache entries pin host pages independently from sequence references.
- Hash namespace must distinguish model/cache settings that affect KV compatibility.

### 2. Host Page Table Extension

Extend the host page-table sequence record from a single flat page vector to a logical row that can represent shared and private pages.

Suggested shape:

```cpp
struct SequenceRecord {
  std::vector<int32_t> shared_prefix_pages;
  std::vector<int32_t> private_pages;
  int64_t shared_prefix_tokens;
  int64_t private_start_token;
  int64_t logical_context_tokens;
};
```

Implementation requirements:

- Preserve `Pages(sequence_id)` as a combined logical view.
- Add accessors for shared prefix pages and private pages.
- Add `shared_prefix_tokens` and private start position.
- Ensure all append/offload paths can compute whether a logical token offset maps to a shared page or a private page.
- Reject writes to shared prefix pages.

### 3. Host Page Refcounts and Prefix Pins

Host pages need refcounts that account for both sequence references and prefix-cache entry pins.

Required semantics:

- Attaching a shared prefix page to a sequence increments the sequence refcount.
- Committing a full private page into the prefix index increments the prefix-entry pin.
- Releasing a sequence decrements only sequence references.
- Evicting/removing a prefix entry decrements only prefix-entry pins.
- A physical host page can be recycled only when all sequence refs and prefix pins are gone.

### 4. Prefix-Aware Host Allocation and Binding

Add or update an API such as:

```text
allocate_pages_for_sequences_with_prefix(requests)
```

Request input should include:

- sequence id
- prompt token IDs
- logical prompt length
- cache namespace
- page size

Response output should include:

- shared prefix host pages
- private suffix host pages
- `shared_prefix_tokens`
- `private_start_token`
- logical page count
- physical pages newly allocated
- fallback or miss reason

Allocation flow:

1. Compute full-page chained hashes from the prompt.
2. Lookup contiguous shared prefix pages.
3. Attach matched shared pages to the sequence row.
4. Allocate private host pages only for suffix and future decode runway.
5. Roll back attached shared refs and newly allocated private pages if any later step fails.

### 5. Shared-Page-Safe Host Operations

Make these operations shared-page safe:

- `ReleaseSequence()`
- host unregister/release
- allocation rollback
- host KV reservation/growth
- host KV eviction/re-entry
- `AsyncOffloadLayerKVToHost()`
- `AsyncAppendDecodeKVToHost()`

Rules:

- Full-prompt prefill in Milestone 1 may compute all tokens, but offload must not overwrite shared prefix pages.
- If the implementation still copies full prompt KV, it must skip shared prefix pages and copy only private suffix pages.
- Decode append must always write to private pages.
- Decode append can commit newly completed private pages into the prefix index after the page becomes full and immutable.

### 6. Host Stats

Add stats that make host page savings visible:

- logical host pages
- physical host pages
- shared prefix pages
- private pages
- prefix lookup hits/misses
- shared pages attached
- private pages allocated
- host page refcount increments/decrements
- prefix-entry pin increments/decrements
- host pages saved
- allocation rollback count

### 7. Milestone 1 Tests

Required tests:

- page-level lookup hits only full pages.
- prompts with matching partial final pages do not share the partial page.
- different parent page hashes prevent invalid reuse.
- host page table returns the legacy combined `Pages(sequence_id)` view.
- shared and private page accessors return correct segments.
- release sequence does not free prefix-pinned pages.
- prefix entry eviction does not free sequence-referenced pages.
- allocation rollback restores refcounts and free lists.
- repeated full-page prefixes show fewer physical host pages than logical pages.

## Milestone 2: Prefix-Aware Prefill Compute and Offload Reduction

Milestone 2 starts only after Milestone 1 host rows are correct. It reduces prefill compute and D2H offload for prefix hits.

### 1. Suffix-Only Prefill Metadata

Build explicit prefill metadata per sequence:

```text
prefix_shared_tokens
suffix_input_ids
suffix_start_pos
suffix_length
full_logical_context_length
```

Rules:

- Miss request: `prefix_shared_tokens = 0`, suffix is the full prompt.
- Partial full-page hit: suffix starts at `prefix_shared_tokens`.
- Full hit: `suffix_length == 0` and must use an exact full-hit path or be explicitly rejected with a clear error.
- Prefix-hit and prefix-miss sequences can coexist in the same prefill batch.
- Position IDs and RoPE offsets must use absolute logical positions.

### 2. Prefill Planning Module

Keep planning modular and side-effect free.

Suggested Python dataclasses:

```python
@dataclass
class PrefixReuseSequencePlan:
    local_idx: int
    sequence_id: int
    prompt_length: int
    prefix_shared_tokens: int
    suffix_start_pos: int
    suffix_length: int
    full_logical_context_length: int
    is_full_hit: bool
    fallback_reason: str | None


@dataclass
class PrefixReusePrefillPlan:
    sequences: list[PrefixReuseSequencePlan]
    suffix_input_ids: list[torch.Tensor]
    suffix_position_ids: list[torch.Tensor]
    cache_seqlens: torch.Tensor
    total_prompt_tokens: int
    total_suffix_tokens: int
    saved_prefill_tokens: int
```

Public functions:

```text
build_prefix_reuse_prefill_plan(...)
split_prefix_reuse_plan_for_micro_batch(...)
validate_prefix_reuse_plan(...)
```

The planner must not allocate GPU pages, load host KV, mutate host page tables, or run model code.

### 3. GPT-OSS/GQA Suffix Prefill

GPT-OSS/GQA is the first target path.

Required behavior:

- Compute Q/K/V only for suffix tokens.
- Suffix Q attends over cached prefix K/V plus newly computed suffix K/V.
- Suffix position IDs use absolute positions starting at `suffix_start_pos`.
- The full logical context length is visible to attention and logits extraction.
- Logits must be produced for the correct last logical prompt token.

If the current FlashAttention path cannot consume paged prefix KV plus suffix K/V directly, use a temporary batch-local KV view for prefill only:

```text
temporary prefill KV view = gathered cached prefix KV + current suffix KV
```

This temporary view must not become the long-lived storage format.

### 4. Suffix-Only Host Offload

Offload only newly computed suffix K/V into private host pages.

The offload API must support separate source and destination offsets:

```text
source_token_start
destination_token_start
tokens_to_copy
```

For suffix-only prefill:

```text
source_token_start = 0
destination_token_start = prefix_shared_tokens
tokens_to_copy = suffix_length
```

The API must reject writes that map into shared prefix pages.

### 5. Exact Full-Hit Behavior

For `suffix_length == 0`, full-hit handling must be explicit.

Allowed first implementation choices:

- Implement a decode-like or cached-prefill path that produces the next-token logits without recomputing the full prompt.
- Or fail loudly with a clear unsupported full-hit error while the feature is enabled.

Not allowed:

- silently falling back to full prefill.
- implicitly recomputing the last token without documenting it as the exact full-hit behavior.

### 6. Milestone 2 Stats

Add stats for compute/offload savings:

- total prompt tokens
- suffix tokens computed
- prefix tokens skipped
- suffix KV tokens offloaded
- prefix KV tokens not offloaded
- full-hit exact path count
- full-hit guarded error count
- fallback/gated path count by reason

### 7. Milestone 2 Tests

Required tests:

- mixed hit/miss prefill batch.
- suffix-only input IDs are correct.
- absolute position IDs/RoPE offsets are correct.
- GPT-OSS/GQA suffix-prefill output matches full-prefill baseline within accepted tolerance.
- offload writes suffix KV to private pages at the correct destination offset.
- shared prefix pages are not overwritten.
- exact full-hit behavior is implemented or clearly rejected.
- unsupported wrapper paths fail loudly or are gated before partial reuse.

## Milestone 3: Decode-Batch GPU Page Materialization

Milestone 3 reduces GPU page pressure after decode batches are formed.

### 1. Decode-Batch Plan

Build a per-rank decode-batch plan from each sequence's logical host row:

```text
logical host row = shared prefix host pages + private suffix/decode host pages
```

The plan should identify:

- all logical pages needed by each sequence row.
- which host pages are shared across rows.
- which host pages are already materialized on GPU.
- which unique host pages must be loaded.
- which decode runway pages must remain private.

### 2. Deduplicated H2D Materialization

Within a rank/decode batch:

1. Deduplicate host pages.
2. Allocate one GPU physical page for each unique missing host page.
3. Load each unique host page once.
4. Point all sequence page-table rows that need that prefix page to the same GPU physical page.
5. Keep suffix/decode runway pages private.

### 3. GPU Sequence State

Extend GPU sequence state so it can represent logical rows whose pages may be shared.

Required capabilities:

- logical page table rows can reference shared physical GPU pages.
- private decode pages remain sequence-owned.
- page-table rebuild preserves shared physical page references.
- GPU page release is physical-refcount aware.

### 4. GPU Refcount Lifecycle

Make GPU release and transition logic refcount-safe for:

- `PREFILLED`
- `IN_DECODE`
- `ON_HOLD`
- `EVICTED`
- `COMPLETED`
- extension failure
- `IN_DECODE -> ON_HOLD`
- `ON_HOLD -> IN_DECODE`

Rules:

- Entering a decode batch increments refs for shared GPU prefix pages used by the sequence row.
- Leaving decode or moving on hold decrements only that sequence row's refs.
- A shared GPU page returns to the free list only when its physical refcount reaches zero.
- Decode writes never target shared prefix pages.

### 5. Milestone 3 Stats

Add GPU materialization stats:

- logical GPU pages
- physical GPU pages
- unique host pages loaded
- duplicate H2D loads skipped
- shared GPU prefix pages
- private GPU decode pages
- GPU shared page refcount increments/decrements
- GPU pages saved
- GPU materialization rollback count

### 6. Milestone 3 Tests

Required tests:

- two decode rows sharing the same prefix host pages load each unique host page once.
- GPU page-table rows point shared prefix pages to the same physical GPU page.
- decode runway pages are private.
- completion releases private and shared GPU refs correctly.
- `IN_DECODE -> ON_HOLD -> IN_DECODE` preserves correctness and refcounts.
- extension failure rolls back GPU refs and allocations.
- decode output matches the non-sharing baseline.

## End-to-End Implementation Order

1. Add `--enable-prefix-reuse` and disabled-by-default wiring.
2. Add page-level chained hash types and namespace hashing.
3. Add host prefix index with lookup and commit for full pages only.
4. Extend host page-table sequence records to shared prefix pages plus private pages.
5. Preserve the legacy combined `Pages(sequence_id)` view.
6. Add host page refcounts and prefix-entry pins.
7. Implement prefix-aware host allocation/binding with rollback.
8. Make release, unregister, offload, append, host growth, eviction, and re-entry shared-page safe.
9. Add Milestone 1 host stats and tests.
10. Add side-effect-free suffix prefill planner.
11. Add GPT-OSS/GQA suffix-only prefill path.
12. Add suffix attention over cached prefix KV plus suffix KV, using a temporary batch-local KV view if needed.
13. Add suffix-only host offload with separate source and destination offsets.
14. Define and implement or explicitly guard exact full-hit behavior.
15. Add Milestone 2 stats and correctness tests.
16. Add decode-batch host-page dedup planning.
17. Extend GPU page state to support shared physical prefix pages plus private decode pages.
18. Add GPU refcounts and shared-page-safe release transitions.
19. Add deduplicated H2D materialization and page-table rebuild support.
20. Add Milestone 3 stats and GPU lifecycle tests.
21. Run approved GPU validation with clean/verify before launching the server.

## Acceptance Checklist

- [ ] Feature is disabled by default behind `--enable-prefix-reuse`.
- [ ] Disabled-feature behavior preserves existing request-pool and dynamic-host-KV behavior.
- [ ] Prefix matching is page-level only.
- [ ] No token-level split.
- [ ] No partial-page sharing.
- [ ] No raw prompt text keying.
- [ ] Host KV supports shared prefix pages with correct page refcounts.
- [ ] Prefix entries pin host pages.
- [ ] Allocation rollback restores all host refs and free lists.
- [ ] Release behavior is shared-page safe.
- [ ] Host page-table rows represent `[shared prefix pages + private suffix pages]`.
- [ ] Legacy combined `Pages(sequence_id)` view is preserved.
- [ ] Host stats show fewer physical host pages than logical pages for repeated full-page prefixes.
- [ ] GPT-OSS/GQA prefix-hit prefill computes only suffix tokens.
- [ ] GPT-OSS/GQA prefix-hit prefill offloads only suffix tokens.
- [ ] Prefix-hit correctness matches full prefill within accepted tolerance.
- [ ] Exact full-hit behavior is implemented or explicitly guarded with a clear error.
- [ ] No silent full-prefill fallback for exact full hits.
- [ ] Decode-batch planning deduplicates shared host pages.
- [ ] Each unique needed host page is loaded to GPU once per rank/batch.
- [ ] GPU page-table rows can reference shared physical prefix pages and private decode pages safely.
- [ ] Decode writes never target shared prefix pages.
- [ ] Lifecycle transitions are refcount-safe for `PREFILLED`, `IN_DECODE`, `ON_HOLD`, `EVICTED`, and `COMPLETED`.
- [ ] Prefix-hit and prefix-miss sequences can coexist in the same prefill batch.
- [ ] Prefix-hit and prefix-miss sequences can coexist in the same decode batch.
- [ ] DSA/MLA unsupported paths fail loudly or are explicitly gated.
- [ ] Tests cover page-level lookup.
- [ ] Tests cover host refcounts.
- [ ] Tests cover GPU refcounts.
- [ ] Tests cover suffix-only prefill correctness.
- [ ] Tests cover decode-batch GPU sharing.
- [ ] Tests cover `IN_DECODE -> ON_HOLD -> IN_DECODE`.
- [ ] Tests cover host eviction/re-entry.
- [ ] Tests cover completion release.
- [ ] GPU validation passes on an approved GPU host with mandatory clean/verify before server launch.
