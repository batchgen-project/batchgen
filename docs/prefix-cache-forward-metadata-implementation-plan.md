# Prefix Cache Forward Metadata Implementation Plan

## Goal

Make prefix-cache metadata a first-class runtime concept in BatchGen.

The target design is:

- [ ] Prefix reuse metadata is constructed once in the worker, validated once, and passed explicitly.
- [ ] Attention wrappers and backends no longer infer batch state from `AttnWrapperBase` class variables.
- [ ] Prefill metadata natively represents `q_len != kv_len`, meaning suffix queries attend over full prefix-plus-suffix KV.
- [ ] GQA, MLA, and DSA consume the same forward-batch metadata shape.
- [ ] Model-specific code is limited to projection, output projection, and auxiliary-cache adapters.
- [ ] Missing required metadata raises a runtime exception instead of silently falling back.

## Current Problem

BatchGen already has partial metadata objects, but they are not the owner of runtime state.

- `PrefixReusePrefillPlan` is side-effect-free and useful, but only covers planning.
- `PrefixCachePrepackMetadata` validates wrapper state, but reconstructs metadata from class variables.
- `AttnWrapperBase` currently carries prepack metadata, decode metadata, GPU managers, host worker views, and DSA hints as class-level global state.
- `batchgen_worker.py` manually writes the same prepack and prefix fields into both `Attn_Wrapper` and `AttnWrapperBase`.

This makes prefix reuse behave like wrapper glue instead of a core forward-batch contract.

## Target Architecture

```text
BatchGenWorker
  -> builds ForwardBatchMetadata
  -> validates ForwardBatchMetadata
  -> binds metadata for this forward
  -> model layer forward
  -> attention wrapper
  -> prefix-aware attention backend
  -> GQA / MLA / DSA implementation
```

Compatibility should be preserved during migration:

```text
ForwardBatchMetadata
  -> temporary compatibility binder
  -> legacy AttnWrapperBase fields
```

The compatibility binder is temporary and must not remain the long-term owner of state.

## Milestone 1: Define First-Class Metadata Types

- [x] Add `batchgen/attention/forward_metadata.py`.
- [x] Define `PrefixReuseMetadata`.
- [x] Include `prefix_lens`.
- [x] Include `suffix_lens`.
- [x] Include `full_seq_lens`.
- [x] Include `saved_tokens`.
- [x] Include `is_full_hit`.
- [x] Include `global_sequence_ids`.
- [x] Define `PrefillAttentionMetadata`.
- [x] Include `cu_seqlens_q`.
- [x] Include `cu_seqlens_k`.
- [x] Include `max_seqlen_q`.
- [x] Include `max_seqlen_k`.
- [x] Include `q_seq_lens`.
- [x] Include `kv_seq_lens`.
- [x] Include `position_ids`.
- [x] Include optional `prefix_reuse`.
- [x] Define `DecodeAttentionMetadata`.
- [x] Include `cache_seqlens`.
- [x] Include `max_seqlen`.
- [x] Include `page_table`.
- [x] Include `slot_indices`.
- [x] Include optional `batch_slice`.
- [x] Define `KVCacheMetadata`.
- [x] Include `gpu_paged_kv_manager`.
- [x] Include `host_worker_view`.
- [x] Include `aux_gpu_paged_kv_manager`.
- [x] Include `aux_host_worker_view`.
- [x] Define `ForwardBatchMetadata`.
- [x] Include `phase`.
- [x] Include `global_sequence_ids`.
- [x] Include optional `prefill`.
- [x] Include optional `decode`.
- [x] Include optional `kv_cache`.
- [x] Add `validate()` methods to all metadata dataclasses.
- [x] Validate tensor dtype and device requirements.
- [x] Validate sequence counts.
- [x] Validate `cu_seqlens_q` length.
- [x] Validate `cu_seqlens_k` length.
- [x] Validate `q_seq_lens` against `cu_seqlens_q`.
- [x] Validate `kv_seq_lens` against `cu_seqlens_k`.
- [x] Validate prefix plus suffix equals full sequence length.
- [x] Validate full-hit suffix length is zero.
- [x] Add unit tests for no reuse.
- [x] Add unit tests for partial reuse.
- [x] Add unit tests for full reuse.
- [x] Add unit tests for miss.
- [x] Add unit tests for invalid shape, dtype, and length mismatches.

## Milestone 2: Add a Compatibility Binding Layer

- [x] Add `batchgen/attention/forward_metadata_context.py`.
- [x] Store current `ForwardBatchMetadata` in a `contextvars.ContextVar`.
- [x] Implement `bind_forward_batch_metadata(metadata)` as a context manager.
- [x] Implement `get_current_forward_batch_metadata(required: bool = False)`.
- [x] In the context manager, temporarily sync metadata into legacy `AttnWrapperBase` fields.
- [x] Sync prepack mode fields.
- [x] Sync prepack cu-seqlens.
- [x] Sync prepack max seqlen.
- [x] Sync prepack sequence count.
- [x] Sync prepack sequence lengths.
- [x] Sync prefix reuse mode.
- [x] Sync prefix shared token counts.
- [x] Sync full sequence lengths.
- [x] Sync position ids.
- [x] Sync current global sequence ids.
- [x] Sync decode cache seqlens where applicable.
- [x] Restore all previous legacy fields when the context exits.
- [x] Restore correctly if an exception is raised inside the context.
- [x] `required=True` must raise if metadata is missing.
- [x] Add unit tests for normal entry and exit.
- [x] Add unit tests for exception exit.
- [x] Add unit tests for nested contexts.
- [x] Add unit tests that legacy fields do not leak across batches.

## Milestone 3: Move Worker Metadata Construction Into a Builder

- [x] Add `batchgen/prefill/attention_metadata_builder.py`.
- [x] Implement `build_prefill_forward_metadata(...)`.
- [x] Inputs should include prepack metadata.
- [x] Inputs should include batch spans.
- [x] Inputs should include optional `PrefixReusePrefillPlan`.
- [x] Inputs should include flattened position ids.
- [x] Inputs should include target device.
- [x] Output should be `ForwardBatchMetadata`.
- [x] For no prefix reuse, generate `cu_seqlens_q == cu_seqlens_k`.
- [x] For prefix reuse, generate suffix `cu_seqlens_q`.
- [x] For prefix reuse, generate full-context `cu_seqlens_k`.
- [x] For prefix reuse, generate `PrefixReuseMetadata`.
- [x] Preserve current `global_sequence_ids` ordering.
- [x] Preserve current suffix-only position id semantics.
- [x] Replace worker-local prepack metadata plumbing with the builder.
- [x] Wrap model-layer execution in `bind_forward_batch_metadata(forward_meta)`.
- [x] Remove direct duplicate writes to `Attn_Wrapper` and `AttnWrapperBase` from worker code.
- [x] Keep compatibility writes only inside the context manager.
- [x] Add unit tests for builder output with no prefix reuse.
- [x] Add unit tests for builder output with partial prefix reuse.
- [x] Add unit tests for builder output with full hit.
- [x] Add unit tests for mixed hit and miss.
- [x] Run `py_compile` for the touched modules.

## Milestone 4: Make Wrappers Prefer Explicit Metadata

- [x] Change `AttnWrapperBase.prefix_cache_metadata()` to read current `ForwardBatchMetadata` first.
- [x] Keep class-variable fallback temporarily for compatibility.
- [x] Mark `PrefixCachePrepackMetadata.from_wrapper_cls()` as a legacy compatibility path.
- [x] Add a constructor from `PrefillAttentionMetadata` to `PrefixCachePrepackMetadata` if needed.
- [x] Update GQA prefix replay helpers to accept `PrefillAttentionMetadata`.
- [x] Update MLA prefix replay helpers to accept `PrefillAttentionMetadata`.
- [x] Update `PrefixAwarePrefillOffloader` to consume explicit metadata.
- [x] Use `global_sequence_ids` from metadata for host offload.
- [x] Use `q_seq_lens` from metadata for suffix spans.
- [x] Use `prefix_lens` from metadata for destination offsets.
- [x] Update GPT-OSS wrappers to prefer explicit metadata.
- [x] Update DeepSeek wrappers to prefer explicit metadata.
- [x] Update GLM wrappers to prefer explicit metadata.
- [x] Update Kimi wrappers to prefer explicit metadata.
- [x] Update MiniMax wrappers to prefer explicit metadata.
- [x] Add tests showing explicit metadata and legacy fallback produce identical spans.
- [x] Add tests showing incomplete fallback raises.

## Milestone 5: Introduce a Prefix-Aware Attention Backend Interface

- [x] Add `batchgen/attention/prefix_aware_backend.py`.
- [x] Define a `PrefixAwareAttentionBackend` protocol or base class.
- [x] Add `forward_prefill(query, key, value, metadata, kv_cache_metadata)`.
- [x] Add a GQA backend adapter.
- [x] The GQA adapter should reuse existing `gqa_prefill_fa` first.
- [x] The GQA adapter should support `cu_seqlens_q != cu_seqlens_k`.
- [x] Add an MLA backend adapter.
- [x] The MLA adapter should reuse existing prepacked MLA and prefix replay logic first.
- [x] The first version should not introduce new kernels.
- [x] The first version should not change numerical behavior.
- [x] Wrappers should select backend adapters instead of directly managing prefix cache details.
- [x] Add tests for no prefix, partial prefix, and full prefix using the same interface.
- [x] Add tests that missing required backend metadata raises.

## Milestone 6: Consolidate Model-Specific MLA Adapters

- [x] Keep model-specific differences in `batchgen/models/wrappers/prefix_mla_model_adapters.py`.
- [x] DeepSeek adapter should only handle absorbed query projection and output projection.
- [x] GLM adapter should only handle DSA auxiliary cache and GLM-specific projection details.
- [x] Kimi adapter should only handle Kimi MLA projection details.
- [x] MiniMax should remain on the shared GQA prefix backend instead of adding an MLA adapter.
- [x] Adapters should accept explicit `PrefillAttentionMetadata`.
- [x] Adapters should not read `AttnWrapperBase` class variables.
- [x] Remove duplicated prefix length handling from model wrappers.
- [x] Remove duplicated cu-seqlens handling from model wrappers.
- [x] Remove duplicated global sequence id handling from model wrappers.
- [x] Add smoke tests that all supported MLA wrappers enter through the shared adapter path.
- [x] Run `py_compile` for model wrapper modules.

## Milestone 7: Add True Extend-Mode KV Writes

- [x] Extend `GPUPagedKVCacheManager` with a multi-token suffix append API.
- [x] The API should support writing multiple suffix tokens per sequence.
- [x] The API should accept `global_sequence_ids`.
- [x] The API should accept `prefix_lens`.
- [x] The API should accept `suffix_lens`.
- [x] The API should accept explicit destination slots or page table metadata.
- [x] GQA prefill should write suffix K/V directly into GPU paged KV.
- [x] MLA prefill should write suffix compressed MLA KV directly into GPU paged KV.
- [x] Attention backend should attend via page table over full context.
- [x] Remove host-prefix KV concatenation from hot path where backend support exists.
- [x] Keep the true extend path behind an explicit experimental flag and leave replay as the default compatibility fallback until fully validated.
- [ ] Validate partial reuse exactness.
- [ ] Validate full reuse exactness.
- [ ] Validate miss exactness.
- [ ] Measure prefill wall time before and after.

Notes:

- True extend-mode attention is currently gated by `BATCHGEN_PREFIX_REUSE_GPU_EXTEND_ATTENTION=1`.
- GPU suffix append without switching attention is gated by `BATCHGEN_PREFIX_REUSE_GPU_EXTEND_WRITES=1`.
- Replay remains the default compatibility path until exactness and timing validation are complete.
- The first true extend-mode attention path supports single-sequence suffix micro-batches, matching the current prefix-reuse isolation policy.

## Milestone 8: Remove Legacy Global Metadata Ownership

- [ ] Delete or deprecate `AttnWrapperBase.prepack_prefix_reuse_mode`.
- [ ] Delete or deprecate `AttnWrapperBase.prepack_prefix_shared_tokens`.
- [ ] Delete or deprecate `AttnWrapperBase.prepack_full_seq_lengths`.
- [ ] Delete wrapper paths that reconstruct prefix metadata from class variables.
- [ ] Keep decode legacy fields only until decode metadata migration is complete.
- [ ] Add runtime warnings for any remaining legacy fallback.
- [ ] Document that new model integrations must use explicit metadata and adapters.
- [ ] Remove compatibility fallback after all supported models are migrated.

## Milestone 9: Validation Matrix

- [ ] Run `py_compile` on `batchgen/attention`.
- [ ] Run `py_compile` on `batchgen/prefill`.
- [ ] Run `py_compile` on `batchgen/models/wrappers`.
- [ ] Run unit tests for metadata validation.
- [ ] Run unit tests for metadata builder.
- [ ] Run unit tests for metadata context manager.
- [ ] Run unit tests for offloader span calculation.
- [ ] Run small E2E with no reuse.
- [ ] Run small E2E with partial reuse.
- [ ] Run small E2E with full reuse.
- [ ] Run small E2E with mixed hit, full hit, and miss.
- [ ] Verify prefix reuse enabled output exactly matches prefix reuse disabled output.
- [ ] Smoke test GPT-OSS.
- [ ] Smoke test DeepSeek.
- [ ] Smoke test GLM.
- [ ] Smoke test Kimi.
- [ ] Smoke test MiniMax.
- [ ] Track prefix hit rate.
- [ ] Track saved prefill tokens.
- [ ] Track microbatch count.
- [ ] Track prefill wall time.
- [ ] Run full MMLU Pro only after the metadata and backend tests are stable.

## Commit Strategy

- [ ] Use one commit per milestone.
- [ ] Keep Milestones 1 to 3 behavior-preserving.
- [ ] Use commit messages that explicitly say when a change is metadata plumbing only.
- [ ] Keep backend interface changes separate from KV manager extend-write changes.
- [ ] Keep kernel changes separate from metadata refactors.
- [ ] Do not introduce silent fallback.
- [ ] Raise runtime exceptions for missing required metadata.
- [ ] Run the relevant unit tests or `py_compile` before each commit.
- [ ] Stop and document risks before changing exactness behavior or kernel behavior.
