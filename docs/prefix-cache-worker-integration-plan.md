# Prefix Cache Worker Integration Plan

## Scope

This document reviews the current `feature/add-staged-page-level-prefix-reuse`
code path and defines the remaining work needed to make Host-side prefix cache
reuse active in `BatchGenWorker`.

The immediate target is GPT-OSS / GQA because its model path already has a
prefix-aware extend-prefill backend. The worker integration should still be
written through generic Host prefix-cache and forward-metadata abstractions so
MLA/SWA/auxiliary groups can reuse the same lifecycle later.

Prefix cache remains a Host-side shared-memory index over existing Host KV
pages. It does not allocate Host KV pages. KV managers allocate, write, load,
and release physical pages; the coordinator indexes resident page handles,
attaches them during lookup, protects them during Host-to-GPU loads, and returns
evicted handles to the caller.

## Current Code Review

### Already Wired

- Server flags exist:
  - `batchgen/server/server_args.py`
  - legacy `batchgen/batchgen_server.py`
  - flags: `--enable-prefix-cache`, `--prefix-cache-debug-stats`
- Server creates the C++ Host coordinator with `create_region=True`:
  - `batchgen/server/worker_manager.py::_initialize_prefix_cache_owner`
  - legacy `batchgen/batchgen_server.py::_initialize_prefix_cache_owner`
  - The new `server/worker_manager.py` path derives prefix-cache capacity from
    `host_kv_cache_size_per_rank`.
  - The legacy `batchgen_server.py` path currently derives owner capacity from
    the user-facing total Host KV size. This can mismatch the worker attach
    config and should be fixed or the legacy path should be disabled for prefix
    cache validation.
- Workers attach to the coordinator with `create_region=False`:
  - `BatchGenWorker._initialize_prefix_cache_worker`
- Runtime config is derived from Host KV profiles:
  - `batchgen/prefix_reuse/config.py`
  - GPT-OSS resolves to one required `FULL_KV` group.
  - MLA models resolve to `MLA_COMPRESSED_KV`.
  - DSA/indexer models can add an auxiliary required group.
- Lookup helper exists:
  - `batchgen/prefix_reuse/prefill.py::lookup_prefix_cache_for_prefill`
  - `BatchGenWorker._lookup_prefix_cache_for_prefill`
- Estimate-only helper exists:
  - `batchgen/prefix_reuse/prefill.py::estimate_prefix_cache_for_prefill`
  - `BatchGenWorker._estimate_prefix_cache_for_prefill`
- Prefix-reuse prefill planning exists:
  - `batchgen/prefill/prefix_reuse.py`
  - current full-hit planning recomputes the final prompt token. This remains
    acceptable for the first shared-page implementation as an idempotent
    overwrite of already-cached KV.
- First-class prefill metadata can express prefix reuse:
  - `batchgen/attention/forward_metadata.py`
  - `batchgen/prefill/attention_metadata_builder.py`
  - `q_seq_lens`, `kv_seq_lens`, and `append_seq_lens` are separate.
- Legacy wrapper compatibility exists:
  - `batchgen/attention/forward_metadata_context.py`
  - It mirrors metadata into `AttnWrapperBase` fields for current wrappers.
- GQA compute path exists:
  - `batchgen/attention/prefix_aware_backend.py::GqaPrefixAwareAttentionBackend`
  - `batchgen/attention/gqa/fa_extend.py::gqa_extend_fa`
  - GPT-OSS wrapper calls the prefix-aware backend in prepacked prefill.
- Host KV offload can append only newly computed suffix tokens:
  - `batchgen/kv_cache/prefill_offload.py`
  - It uses `async_offload_layer_kv_range_to_host` when prefix reuse is active.
- GPU materialization helper exists:
  - `batchgen/prefix_reuse/materialization.py`
  - It can convert attached Host prefix pages into GPU paged KV pages.
- Commit helper exists:
  - `batchgen/prefix_reuse/commit.py`
  - It can build aligned `commit_prefix_pages` requests from existing Host KV
    page tables.
- API usage has a `cached_tokens` field:
  - `batchgen/server/usage.py`
  - `SequenceEntry.prefix_shared_tokens` already exists.

### Current Blocker

`BatchGenWorker.prefill_prepacked()` still does not run the actual reuse path.

Current production flow:

```text
collect full prompt tensors
  -> _estimate_prefix_cache_for_prefill(...)
  -> prepack full prompt
  -> manually set Attn_Wrapper / AttnWrapperBase fields
  -> run model
  -> offload full prompt KV to Host
  -> select first decode token
```

The comment in `batchgen_worker.py` explicitly keeps prefix cache in
estimate-only mode until Host sequence KV completeness is solved. That is the
right safety guard: suffix-only prefill is incorrect unless the sequence's Host
KV state also contains the reused prefix pages before the sequence enters
decode or later Host-to-GPU reload.

### Missing End-to-End Pieces

- Replace estimate-only lookup with real lookup/attach in the prefix-enabled
  prefill admission path.
- Build suffix-only prepack inputs from lookup results.
- Build `ForwardBatchMetadata` for each micro-batch instead of manually setting
  parallel wrapper class variables.
- Materialize attached Host prefix pages into GPU paged KV for the current
  micro-batch.
- Make the sequence Host KV table logically complete:
  - shared prefix pages from the coordinator
  - private suffix/decode pages from the sequence allocation
- Ensure decode load/reload sees the complete logical KV, not just private
  suffix pages.
- Move lookup early enough to affect Host KV allocation, or explicitly accept a
  correctness-only first version that allocates full private Host capacity.
- Commit completed aligned prompt pages into the coordinator after prefill
  offload completes.
- Commit aligned prompt+decode pages at request completion before Host pages
  are released or recycled.
- Keep lookup attachments alive while any sequence Host page table references
  shared prefix pages; release them only when the sequence detaches those pages.
- Feed evicted prefix page handles back to the owning Host KV manager before
  relying on those pages as free capacity.
- Populate `cached_tokens` from the page-aligned tokens actually attached and
  reused by the worker, not from the raw lookup result.

## Required Invariants

### Prefix Lookup

For every sequence, the coordinator must return an attachable page-boundary
hit. Page alignment is a lookup/admission invariant, not a planner
responsibility. The worker should validate the invariant before attaching pages
and fail loudly if it is violated; it should not silently floor the hit length.

```text
raw_cached_tokens = coordinator.common_cached_tokens
assert raw_cached_tokens % page_size == 0
assert raw_cached_tokens <= prompt_length
shared_prefix_tokens = raw_cached_tokens
```

For normal partial hits:

```text
shared_prefix_tokens < prompt_length
query_tokens = prompt[shared_prefix_tokens : prompt_length]
position_ids = range(shared_prefix_tokens, prompt_length)
logical_kv_len = prompt_length
append_tokens = prompt_length - shared_prefix_tokens
usage.cached_tokens = shared_prefix_tokens
```

For raw full hits, attach the full prompt but still run the existing one-token
continuation step. In practice this only applies when the full prompt has been
published at the prefix-cache boundary; otherwise the lookup returns the
largest published page-aligned prefix and the request is a partial hit.

```text
raw_cached_tokens = prompt_length
shared_prefix_tokens = prompt_length
compute_cached_tokens = prompt_length - 1
query_tokens = [prompt[-1]]
position_ids = [prompt_length - 1]
logical_kv_len = prompt_length
append_tokens = 1
usage.cached_tokens = shared_prefix_tokens
```

This writes the final token KV back to the same logical page that already
contains it. That is intentionally treated as an idempotent overwrite: the
request has the same prompt tokens and the same prefix context, so the produced
KV is semantically the same cached KV. Do not introduce page rollback, overlay
pages, or a separate query-only full-hit path for the first implementation.

### Host KV Completeness

Before a sequence transitions from prefill to decode, the Host KV representation
for that sequence must cover the full logical prompt:

```text
[shared prefix pages] + [private suffix pages]
```

GPU materialization alone is not sufficient because GPU pages are transient.
Decode ON_HOLD reload, migration, host eviction/re-entry, and completion commit
all depend on Host KV being the source of truth.

### Host Allocation Timing

Current Host KV pages for prefill are allocated before `prefill_prepacked()`.
The allocation code reserves capacity from `seq.prompt_length + chunk_size`
before the worker currently performs the estimate-only prefix lookup.

That means real prefix reuse cannot simply be inserted inside the existing
`prefill_prepacked()` body if the goal is Host page sharing:

```text
current order:
  allocate private Host pages for full prompt
  -> run prefill_prepacked()
  -> estimate prefix cache
```

For correctness-only validation, this is acceptable if reused prefix pages are
copied into the already allocated private sequence pages. It does not save Host
memory, but it lets compute reuse be tested.

For the target shared-page design, lookup must move earlier:

```text
target order:
  collect prompt token ids
  -> prefix lookup
  -> reserve/attach shared prefix pages
  -> allocate private Host pages for the existing initial Host KV reserve,
     minus attached shared prefix tokens
  -> run suffix prefill
```

This also affects `SequenceEntry` metadata. Today `host_pages_allocated` and
`host_token_capacity` mean private sequence-owned capacity. With shared prefix
attachment, do not reinterpret those fields as logical capacity. Keep them as
private capacity and add explicit shared-prefix metadata:

Validation should compare logical capacity as:

```text
logical_host_tokens =
  shared_prefix_tokens + private_host_token_capacity
```

Do not silently reinterpret `host_pages_allocated`; it is already used for host
KV pressure planning and release ordering.

### Sequence Page Layout

The target design must treat a sequence's Host KV page table as a flat logical
address map. The page table should not own pages and should not need to know
whether a page is shared or private.

Prefix reuse is page-granular. A shared prefix attachment is valid only when it
is page-aligned:

```text
shared_prefix_tokens % page_size == 0
shared_prefix_pages == shared_prefix_tokens / page_size
```

If lookup returns a token hit that cannot be represented as full pages, that is
a coordinator/configuration bug. Do not clamp it in the planner, and do not
attach partial pages.

For a prefix-hit sequence:

```text
logical Host KV page table

  token range:   [0 ................................ prompt_length)
                 [cached prefix pages] [private suffix pages]

  ownership:     prefix coordinator     Host KV manager sequence allocation
  lifecycle:     resident/attached      released with the sequence
```

In this design, "prepare pages for a sequence" means two different operations:

- attach existing shared prefix pages returned by the coordinator lookup
- allocate new private pages for suffix and future decode growth

Shared prefix pages are not allocated again. They are inserted into the
sequence's flat logical Host page table and protected by coordinator
attachments while the sequence uses them. Private pages are allocated by the
Host KV manager and remain owned by that sequence.

Keep ownership out of `HostKVPageTable`:

```text
HostKVPageTable:
  sequence_id -> [shared prefix page handles..., private page handles...]
  no ownership, no shared/private flags

HostPrefixCacheCoordinator:
  owns shared-resident page references, attachment refs, eviction state

HostPagedKVBackend / allocator:
  owns sequence-private pages only
```

The worker should not implement this by manually concatenating Python lists of
page ids. Add a per-Host-KV-worker-view API that creates or updates that
manager's sequence logical page table in C++:

```python
host_worker_view.prepare_sequence_with_shared_prefix(
    sequence_id: int,
    shared_prefix_pages: Sequence[HostPageHandle],
    shared_prefix_tokens: int,
    private_token_capacity: int,
)
```

For multi-KV-manager models, the worker integration calls the same API on each
required worker view with that group's own pages. Cross-group hit consistency
is enforced by the coordinator/lookup result, not by overloading one page-table
API with group maps.

Equivalent split APIs are acceptable only if they are used as one transaction:

```python
host_worker_view.attach_shared_prefix_pages(...)
host_worker_view.allocate_private_pages_for_sequence(...)
```

The resulting Host worker view must expose logical page tables for later load
and commit paths:

```text
build_page_table(sequence_id)
  -> [shared page 0, shared page 1, ..., private page 0, private page 1, ...]
```

`build_page_table(...)` should not need a new public shape. It can continue to
return the flat logical page vector. The important requirement is that all
logical KV read/write paths use this table instead of asking the backend for
sequence-private pages only.

Suffix offload with `raw_start_position` must use the original logical token
position. Because shared prefix attachment is page-aligned, normal page-table
indexing is sufficient:

```text
page_index = raw_start_position / page_size
page_offset = raw_start_position % page_size
target_page = logical_pages[page_index]
```

For example, if `shared_prefix_tokens == 128` and `page_size == 64`, suffix
offload at `raw_start_position=128` resolves to `logical_pages[2]`, the first
private page. No special shared/private check is needed in the page table.

Raw full-hit is the exception to the "suffix writes private pages" intuition:
the one-token continuation has `raw_start_position=prompt_length - 1`, so it
resolves to the last shared prompt page and idempotently overwrites that KV. Do
not allocate a private overlay page for this case.

Completion and eviction release rules:

- sequence completion asks the backend to release sequence-private pages,
  releases coordinator attachments for shared prefix pages, and removes the
  flat logical page-table record
- shared resident prefix pages are only returned to the Host KV manager after
  prefix coordinator eviction
- decode commit collects the logical page table, so it can publish chains that
  contain both shared prefix pages and private decode pages

GPU materialization is separate from this Host logical layout. For prefill
compute, `materialize_single_group_lookup_results(...)` allocates temporary GPU
pages for the full logical KV, loads shared Host prefix pages into those GPU
pages, and lets the attention backend append suffix KV. Those GPU pages are
runtime scratch for attention and do not replace the Host sequence page table.

### Page Ownership

- KV managers allocate physical Host pages.
- Prefix coordinator stores resident references to already-written pages.
- A page can be resident in prefix cache with zero active lookup/load
  references.
- A resident page with zero active references is evictable, not free.
- Only explicit prefix eviction returns page handles to the owning Host KV
  manager.
- Sequence cleanup must release private pages but must not free shared resident
  prefix pages unless the coordinator evicts them.

## Recommended Worker Architecture

### New Worker-Side Integration Helper

Add a small module instead of expanding `batchgen_worker.py` further:

```text
batchgen/prefix_reuse/worker_integration.py
```

Responsibilities:

- Convert worker-local batch data into prompt token lists.
- Run coordinator lookup.
- Update `SequenceEntry.prefix_shared_tokens`.
- Keep sequence-level prefix attachment handles until Host shared pages are
  detached from the sequence logical page table.
- Build `PrefixReusePrefillPlan`.
- Build suffix-only prepack input lists.
- Slice prefix plans for micro-batches.
- Build `ForwardBatchMetadata` and `KVCacheMetadata`.
- Materialize required compute groups for a micro-batch.
- Release prefix attachments during sequence cleanup, not immediately after the
  first GPU materialization.
- Build prompt/decode commit requests.

Keep side effects explicit. The helper may mutate sequence usage fields and
call coordinator/KV manager APIs, but it should not own the model forward loop.

### Worker State To Add

Keep these fields inside `BatchGenWorker`:

```python
self.prefix_cache_runtime_config
self.prefix_cache_coordinator
self._active_prefix_sequence_attachments
```

Do not add user-configurable prefix metadata to worker args. Derived config
stays runtime-only.

### Main Worker Flow

Prefix lookup is part of prefill admission/configuration, not part of the
model-forward body. The target order is:

```text
_prepare_prefill_batch()
  -> collect prompt token ids for admitted requests
  -> lookup_and_attach prefix cache entries
  -> derive per-sequence private Host KV capacity
  -> register sequence Host KV tables
  -> attach shared prefix pages
  -> allocate private suffix/decode pages
  -> prefill_prepacked() runs suffix/continuation forward
```

This order is required because the private Host page allocation depends on the
attached shared token length returned by lookup. If lookup happens after
`allocate_pages_for_sequences`, the worker has already allocated pages for the
full prompt and cannot realize Host memory sharing.

After admission/configuration, `prefill_prepacked()` should branch once:

```text
if not enable_prefix_cache:
    run current full-prompt path unchanged
else:
    run prefix-aware prepacked path
```

Do not mix manual `Attn_Wrapper` assignment with metadata context in the same
prefix path. The prefix path should use `bind_forward_batch_metadata(...)`; the
disabled path can keep the current behavior until it is separately cleaned up.

## Detailed Prefill Plan

### Prefill Step 1: Admission Lookup And Plan

Input:

- admitted prefill request ids from `_prepare_prefill_batch()`
- collected full `input_ids_list`
- `prompt_lengths`

Steps:

1. Call `_lookup_prefix_cache_for_prefill(...)`.
2. Validate the raw hit is page-aligned and within the prompt length.
3. Store it as `SequenceEntry.prefix_shared_tokens`; this value is the
   validated page-aligned shared prefix length, not a planner-derived clamp.
4. Attach shared prefix pages and allocate only private suffix/decode pages.
5. Build `PrefixCachePrefillInputs` using
   `_build_prefix_reuse_prepack_inputs(...)`.
6. Use `plan.suffix_input_ids` and `plan.suffix_position_ids` as the query
   input source.
7. If every `prefix_shared_tokens == 0`, the path may either:
   - fall back to the current full-prompt path; or
   - keep the unified path with full suffix inputs.

Recommended first implementation: keep the unified path even on miss. It tests
one path and should produce identical metadata when `prefix_reuse_mode` is
false.

Placement:

- For copy fallback, lookup can initially live inside `prefill_prepacked()`
  because full private pages are still allocated before the copy.
- For shared-page attachment, this must be lifted into the prefill admission /
  Host allocation stage so allocation can reserve private suffix capacity only.
- The target implementation must not perform the first real lookup inside
  `prefill_prepacked()`.

### Prefill Step 2: Suffix Prepack

Build prepack metadata from suffix inputs:

```text
prepack_sequences(prefix_inputs.input_ids_list, prefix_inputs.attention_mask_list)
```

Use the plan's position ids, not `torch.arange(seq_len)`. Partial-hit suffix
positions start at `prefix_shared_tokens`; raw full-hit continuation starts
at `prompt_length - 1` even though `prefix_shared_tokens == prompt_length`.

For each sequence in packed order:

```text
query_len = plan.suffix_length
position_ids = plan.suffix_position_ids
global_sequence_id = plan.sequence_id
```

### Prefill Step 3: Micro-Batch Metadata

For each micro-batch:

1. Slice `PrefixReusePrefillPlan`.
2. Build spans with global sequence ids in the same order as suffix prepack.
3. Build `ForwardBatchMetadata`:

```python
ForwardBatchMetadata(
    phase="prefill",
    global_sequence_ids=[...],
    prefill=PrefillAttentionMetadata(
        cu_seqlens_q=...,
        cu_seqlens_k=...,
        q_seq_lens=suffix_query_lens,
        kv_seq_lens=prompt_lengths,
        append_seq_lens=append_lengths,
        position_ids=suffix_position_ids,
    ),
    kv_cache=KVCacheMetadata(
        gpu_paged_kv_manager=...,
        host_worker_view=...,
        aux_gpu_paged_kv_manager=...,
        aux_host_worker_view=...,
        prefill_prefix_materialization=...,
    ),
)
```

The prefix path should not manually set:

- `Attn_Wrapper.prepack_*`
- `AttnWrapperBase.prepack_*`
- `AttnWrapperBase.prefill_prefix_materialization`

Those should be derived by `bind_forward_batch_metadata(...)`.

### Prefill Step 4: GPU Prefix Materialization

For GPT-OSS / GQA first:

1. Create a temporary `GPUPagedKVCacheManager` or reuse the existing worker GPU
   manager if it can safely isolate prefill materialization from active decode
   pages.
2. Call `materialize_single_group_lookup_results(...)` for group `0`.
3. Wrap it in `PrefixMaterializationBundle.from_single(0, materialization)`.
4. Pass the bundle in `KVCacheMetadata.prefill_prefix_materialization`.
5. The attention backend will:
   - wait for the layer load through `wait_for_layer(layer_idx)`
   - append suffix KV into GPU paged KV
   - call `gqa_extend_fa(...)`.

Important implementation choice:

- If the same `gpu_paged_kv_cache_manager` is used for decode and prefix
  prefill, release materialization pages immediately after the micro-batch
  forward and rebuild the decode page table if needed.
- If a separate temporary prefill manager is used, keep ownership simpler:
  destroy/free it after the micro-batch. This is safer for the first worker
  integration.

### Prefill Step 5: Host KV Table Completeness

This must be implemented before enabling real suffix-only prefill.

Preferred API:

```python
host_worker_view.attach_shared_prefix_pages(
    sequence_id: int,
    pages: Sequence[HostPageHandle],
    prefix_tokens: int,
)
```

Semantics:

- The sequence Host KV table is updated to start with shared resident pages.
- `prefix_tokens` must be page-aligned.
- Partial-hit suffix offload writes into private pages through ordinary logical
  `raw_start_position` indexing.
- Raw full-hit one-token continuation may idempotently overwrite the final KV
  in a shared page.
- `HostKVPageTable` remains a flat logical page list and does not store
  shared/private flags.
- Releasing the sequence releases only backend-owned private pages, drops the
  coordinator attachment to shared pages, then removes the flat page-table
  entry.
- Shared prefix pages remain resident until prefix eviction.
- Sequence metadata tracks shared-prefix length separately from private Host
  capacity, but page-table entries do not need ownership metadata.

If this API is too invasive, use a temporary correctness fallback:

```text
copy shared prefix pages into the sequence's private Host KV allocation
```

The fallback is slower and loses Host memory sharing, but it proves the compute
path before flat logical page-table attachment lands.

Fallback requirements:

- Prefix pages must be copied before suffix offload or before the sequence
  enters decode.
- `host_pages_allocated` may remain the full private allocation.
- Commit can read one private Host page table.
- This fallback should be marked temporary because it does not exercise the
  resident shared-page lifecycle.

Do not ship the suffix-only path without either shared-page attachment or copy.
Otherwise decode reload will observe incomplete Host KV.

### Prefill Step 6: Forward Execution

Inside the micro-batch loop:

```python
with bind_forward_batch_metadata(forward_metadata):
    inputs_embeds = model.model.embed_tokens(batch_input_ids_flat)
    hidden_states = inputs_embeds.unsqueeze(0)
    for layer in model.model.layers:
        hidden_states = layer(...)[0]
```

After the forward:

- select logits from `batch_cu_seqlens[1:] - 1`
- write the first generated token exactly as the existing path does
- keep `seq.current_context_length = seq.original_prompt_length + seq.decoded_length`

### Prefill Step 7: Attachment Lifetime

Lookup attachments protect resident prefix nodes from eviction. With shared
Host page-table attachment, they must cover the whole period where the
sequence's logical Host page table references shared pages, not just the first
Host-to-GPU materialization.

Load order:

1. Lookup attaches node.
2. Materialization calls `begin_attachment_load(handle)`.
3. Host-to-GPU load task completes.
4. Materialization calls `end_attachment_load(handle)`.

The attachment itself remains active after step 4. Release it only when the
sequence detaches shared prefix pages:

```text
sequence complete / cancelled / migrated away
  -> stop using the logical page table entry
  -> release private Host pages
  -> release prefix-cache attachment handle
  -> remove sequence page-table entry
```

If the implementation uses the temporary copy fallback instead of shared page
attachment, the attachment may be released after the copied prefix pages and all
dependent GPU loads are complete, because the sequence no longer references
shared resident Host pages.

Use `try/finally` around prefill admission and forward errors. On failure before
the sequence page table owns the shared pages, release the lookup attachment
immediately. On failure after attachment, run the normal sequence cleanup path.

## Commit Plan

### Prompt Commit

Commit after prefill Host offload tasks are complete.

Steps:

1. Retire pending prefill offload tasks.
2. For each owner-local sequence, compute:

```text
commit_tokens =
  floor(prompt_length / publish_boundary_tokens) * publish_boundary_tokens
```

3. Collect pages for all required groups:
   - primary worker view
   - aux worker view if the runtime config has group `1`
4. Call `build_prefix_commit_request(...)`.
5. Call `request.commit(prefix_cache_coordinator)`.
6. On metadata capacity failure, evict unprotected prefix nodes and retry once.

Partial-hit commit must publish a semantically complete prefix chain. If
shared-attachment mode is used, the page list can include already-shared prefix
pages plus private suffix pages. If copy mode is used, the page list is simply
the sequence's private table.

If a sequence has no newly computed aligned prompt pages beyond
`prefix_shared_tokens` (for example a raw full hit), prompt commit should be a
no-op for that sequence. Recommitting an already resident chain is unnecessary.

### Decode Commit

Decode-generated tokens should enter prefix cache, but only after they are no
longer being mutated.

First implementation:

1. At completion, before `_release_host_kv_pages_for_batch(...)`, wait for
   pending decode Host KV append tasks.
2. Compute:

```text
total_tokens = prompt_length + decoded_length
commit_tokens =
  floor(total_tokens / publish_boundary_tokens) * publish_boundary_tokens
```

3. Commit only full aligned pages.
4. Skip the final partial page.
5. Then run sequence cleanup, which releases private pages and drops shared
   prefix attachments.

Do not commit decode pages at every step initially. Completion-time commit is
simpler and avoids publishing pages that are still being appended.

## Eviction Integration

Coordinator eviction returns page handles; it does not release physical Host
pages by itself.

Worker/server integration must add:

```text
evicted = coordinator.evict_until_free(...)
for group in evicted.evicted_group_pages:
    owning_host_worker_view.release_prefix_resident_pages(group.pages)
```

Required semantics:

- Evict whole prefix nodes, not individual groups.
- Do not evict active lookup/load attachments.
- Do not put resident pages into the normal Host KV free list until the
  coordinator has removed the node.
- Host KV allocation pressure and prefix metadata pressure should both be able
  to trigger eviction.

If the current Host KV manager lacks an API to free page handles that are not
attached to a live sequence, add one in C++ rather than faking it in Python.

## Distributed Behavior

Initial scope:

- Prefix cache is per node.
- Each node owns one coordinator shared-memory region.
- Workers on the same node attach to the same region.
- No cross-node prefix sharing.

For tensor/expert parallel:

- Every rank that writes a Host KV shard must commit its own pages.
- Lookup token decisions must be deterministic across ranks.
- Page handles are rank/node local. Do not broadcast raw page handles across
  nodes.
- Usage accounting can be owner-rank only, but compute materialization must
  happen on ranks that execute attention for that sequence.

## Concrete Implementation Order

### Step 1: Worker Prefix Path Skeleton

- Add `batchgen/prefix_reuse/worker_integration.py`.
- Move prompt-token extraction and prefix lookup helpers out of
  `batchgen_worker.py`.
- Add a prefix-enabled branch in `prefill_prepacked()`.
- Keep behavior unchanged when disabled.
- Add unit tests for helper-only code.
- Fix legacy server owner/worker config symmetry or explicitly block
  `--enable-prefix-cache` on the legacy path until it is fixed.

### Step 2: Metadata Binding

- Replace manual wrapper field writes in the prefix branch with
  `ForwardBatchMetadata` and `bind_forward_batch_metadata(...)`.
- Use `build_prefill_forward_metadata(...)`.
- Ensure no-prefix metadata still produces the same `AttnWrapperBase` fields in
  unit tests.

### Step 3: Suffix Prepack

- Prepack `PrefixCachePrefillInputs.input_ids_list`.
- Use `plan.suffix_position_ids` for flattened position ids.
- Cover miss, partial hit, full hit, mixed hit/miss in tests.

### Step 4: GPU Materialization For GQA

- Materialize group `0` lookup results for each micro-batch.
- Attach the resulting bundle to `KVCacheMetadata`.
- Use a temporary prefill GPU KV manager first unless reusing the decode manager
  can be proven safe.
- Free materialization GPU pages after the micro-batch.
- Add tests with fake materialization managers.

### Step 5: Host KV Completeness

- Implement either:
  - shared-prefix logical page-table attachment in Host KV worker view; or
  - a correctness-only copy fallback.
- Ensure `_load_host_kv_to_gpu(...)` can reload a prefixed sequence and see the
  full logical prompt.
- Add integration tests at C++/binding level for shared prefix + private suffix
  page tables.
- If logical attachment is implemented, update `SequenceEntry` metadata and
  validation so private Host capacity and shared logical prefix length are not
  conflated. Do not add shared/private ownership state to `HostKVPageTable`.

### Step 6: Enable Real Lookup Before Private Host Allocation

- Required for the target shared-page attachment design.
- Prefix lookup must run during prefill admission/configuration, before
  `register_sequences(...)` and `allocate_pages_for_sequences(...)`.
- Allocation should reuse the existing non-prefix Host KV reserve formula. Do
  not introduce a new prefix-specific runway knob:

```text
post_prefill_length = prompt_length + 1
gpu_initial_pages = ceil(post_prefill_length / page_size) + INITIAL_GPU_PAGE_BUFFER
gpu_initial_tokens = gpu_initial_pages * page_size

logical_initial_capacity =
  min(
    max(prompt_length + chunk_size, gpu_initial_tokens),
    kv_token_budget,
  )

private_initial_capacity =
  max(logical_initial_capacity - prefix_shared_tokens, append_tokens)

private_pages = ceil(private_initial_capacity / page_size)
```

This preserves the current `chunk_size` and `INITIAL_GPU_PAGE_BUFFER` behavior
while avoiding private allocation for attached shared prefix pages.

- The sequence Host KV view then attaches shared prefix pages and allocates
  private suffix pages.
- Replace `_estimate_prefix_cache_for_prefill(...)` with real
  `lookup_and_attach(...)` in the prefix admission branch.
- Keep estimate-only logs as an optional debug mode if useful.
- Store attachment handles on the sequence or worker state until sequence
  cleanup.
- Set `SequenceEntry.prefix_shared_tokens` from the validated page-aligned
  lookup result.
- For the copy fallback only, moving lookup before private allocation can be
  deferred because the fallback still allocates full private Host pages.

### Step 7: Prompt Commit

- Wait for prefill Host KV offload tasks.
- Collect aligned prompt pages.
- Commit required groups together.
- Evict/retry on coordinator metadata pressure.
- Add tests for aligned/unaligned prompt lengths and multi-group page lists.

### Step 8: Decode Commit At Completion

- Before completed sequence Host page release:
  - wait for pending decode append tasks
  - commit aligned prompt+decode pages
  - then run sequence cleanup, which releases private pages and drops shared
    prefix attachments
- Ensure completion reporting includes `cached_tokens`.
- Add tests for completion-time commit ordering.

### Step 9: Eviction Hook

- Add a Host KV manager API for releasing evicted resident page handles if
  missing.
- Wire coordinator eviction into Host KV allocation pressure and commit retry.
- Add tests that protected attachments are not evicted.

### Step 10: Remote Validation

Run in increasing scale:

1. GPT-OSS prefix disabled: sanity baseline.
2. GPT-OSS prefix enabled, empty cache: miss path.
3. GPT-OSS repeated page-aligned prompts: hit path with nonzero
   `cached_tokens`.
4. 20 MMLU Pro requests, short decode, output sanity.
5. 1000 MMLU Pro requests, larger decode.
6. Compare accuracy and output length with main/baseline.
7. Inspect coordinator stats, Host KV stats, `/dev/shm`, and GPU processes
   after shutdown.

## Tests To Add Or Update

- `tests/unit/test_prefix_worker_integration.py`
  - prompt extraction
  - lookup result to sequence usage
  - suffix prepack source selection
  - micro-batch slicing
  - attachment release on exception
  - attachment handle remains active after materialization when shared pages
    are part of the sequence logical page table
  - copy fallback may release lookup attachment after copy/load completion
- `tests/unit/test_prefill_attention_metadata_builder.py`
  - keep existing mixed hit/full hit cases
  - add assertion that raw full hit has one query token, full prompt KV length,
    and append length one
- `tests/unit/test_prefix_materialization.py`
  - temporary manager release behavior
  - bundle group lookup behavior
- `tests/unit/test_prefix_commit_helpers.py`
  - prompt commit alignment
  - decode completion commit alignment
  - multi-group required pages
- C++/binding tests:
  - `HostPrefixCacheCoordinator` lookup/commit/evict with GPT-OSS full-KV group
  - flat logical Host KV page table containing shared prefix pages followed by
    private suffix pages
  - eviction returns page handles and does not free active attachments

## Risks And Open Questions

- Host logical page-table attachment is the main correctness blocker. Without
  either attachment or a copy fallback, suffix-only prefill cannot safely enter
  decode.
- Reusing the global decode GPU KV manager for prefill materialization may
  disturb active decode page tables. Prefer a temporary prefill manager first.
- Prompt commit must wait for asynchronous prefill offload completion;
  otherwise the coordinator may publish pages before all layers are written.
- Completion-time decode commit must run before Host page release.
- Multi-group models require all required groups to hit together. Partial group
  hit must be a miss for reuse correctness.
- Auxiliary/indexer groups may be required for later decode correctness even if
  the current attention kernel only materializes group `0`.
- Raw full-hit semantics intentionally allow idempotent overwrite of the final
  prompt token KV in the shared page. If a future backend cannot tolerate this
  write pattern, handle it in that backend; do not make the generic worker path
  more complex upfront.

## Definition Of Done

Prefix cache is considered worker-integrated for GPT-OSS when all are true:

- `--enable-prefix-cache` triggers real lookup, not estimate-only.
- Repeated page-aligned prompts report nonzero `cached_tokens`.
- Prefix-hit prefill uses suffix/continuation inputs.
- Host KV remains complete for decode reload and ON_HOLD reload.
- Prompt pages are committed after prefill.
- Decode pages are committed at request completion.
- Shared resident pages are not freed until coordinator eviction.
- Prefix-disabled behavior is unchanged.
- Remote GPT-OSS sanity and MMLU Pro runs complete without output corruption.
