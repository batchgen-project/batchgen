"""Prefill scheduling — host-KV-capacity-bounded sequence selection.

Slice 6 of the worker decouple initiative (issue #175). Extracts the
pure *selection decision* from ``_prepare_prefill_batch``:

  - ``PrefillScheduler.select_prefill_batch`` — greedily admit candidate
    sequences (evicted first, by recompute priority; then queued, by
    arrival) into a prefill batch, bounded by each node's free host-KV
    pages.

Only the *decision* is ported. The cross-rank ``dist.all_gather`` that
collects per-node host-KV free pages, the candidate enumeration over
``global_batch``, and the rank-0 logging stay on the worker, which
builds the request and uses the returned uuid list.

Host KV is per-node: a sequence assigned to a rank on node N draws from
node N's host-KV capacity, so selection is a per-node bin-packing, not a
global one.

Design follows the per-slice frozen-snapshot pattern (no shared mutable
``WorkerState``): the worker passes exactly the fields this decision
consumes; the handler is pure and deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import logging
import os

import torch
from tqdm import tqdm

from batchgen.models.wrappers import AttnWrapperBase
from batchgen.utils import create_position_ids_from_attention_mask
from batchgen.prefill.prepack import (
    prepack_sequences,
    get_prepack_stats,
    build_prefill_micro_batches,
)
from batchgen.batch_order import (
    build_prefill_sequence_spans,
    prefill_sequence_spans_to_cu_seqlens,
    prefill_sequence_spans_to_global_seq_ids,
)

Attn_Wrapper = AttnWrapperBase


@dataclass(frozen=True)
class PrefillCandidate:
    """A sequence eligible for prefill admission (EVICTED or QUEUEING).

    ``is_evicted`` selects the priority group: evicted sequences are
    admitted first (ordered by most-decoded-first to minimise wasted
    recompute), then queueing sequences (ordered by arrival ``global_idx``).
    """

    uuid: str
    assigned_rank: int
    is_evicted: bool
    global_idx: int
    total_decoded_before_eviction: int
    prompt_length: int
    kv_token_budget: int
    page_size: int


@dataclass(frozen=True)
class PrefillSelectionRequest:
    """Frozen snapshot for ``select_prefill_batch``.

    ``per_node_host_free`` is indexed by node id (gathered across ranks).
    ``initial_gpu_page_buffer`` is ``INITIAL_GPU_PAGE_BUFFER`` — pages
    reserved on a sequence's first GPU load.
    """

    candidates: Tuple[PrefillCandidate, ...]
    per_node_host_free: Tuple[int, ...]
    chunk_size: int
    num_nodes: int
    gpus_per_node: int
    initial_gpu_page_buffer: int


class PrefillScheduler:
    """Prefill admission decision — pure, deterministic across ranks."""

    @staticmethod
    def select_prefill_batch(req: PrefillSelectionRequest) -> List[str]:
        """Select which candidate sequences to prefill, bounded by host KV.

        Priority order: EVICTED sequences first (most decoded → least
        wasted recompute, ``global_idx`` tie-break), then QUEUEING (by
        ``global_idx``). Each candidate is admitted iff its node still has
        room for its initial page reservation.

        Initial reservation (matches the worker's dynamic host-KV sizing):
        ``max(prompt_length + chunk_size, gpu_initial_tokens)`` capped at
        ``kv_token_budget``, rounded up to whole pages — where
        ``gpu_initial_tokens`` covers ``prompt_length + 1`` plus the GPU
        page buffer. No safety margin: selection and allocation use the
        same formula by design.

        Pure: reads only the candidate snapshots + per-node free pages.
        The NCCL gather and the ``global_batch`` enumeration stay on the
        worker. Deterministic across ranks (stable sort keys), so every
        rank produces the identical batch without communication.
        """
        evicted = [c for c in req.candidates if c.is_evicted]
        queueing = [c for c in req.candidates if not c.is_evicted]
        evicted.sort(key=lambda c: (-c.total_decoded_before_eviction, c.global_idx))
        queueing.sort(key=lambda c: c.global_idx)
        all_candidates = evicted + queueing
        if not all_candidates:
            return []

        per_node_effective_free = list(req.per_node_host_free)
        node_pages_used = [0] * req.num_nodes
        prefill_batch: List[str] = []

        for c in all_candidates:
            seq_node = c.assigned_rank // req.gpus_per_node
            post_prefill_length = c.prompt_length + 1
            gpu_initial_pages = (
                math.ceil(post_prefill_length / c.page_size)
                + req.initial_gpu_page_buffer
            )
            gpu_initial_tokens = gpu_initial_pages * c.page_size
            initial_capacity = max(c.prompt_length + req.chunk_size, gpu_initial_tokens)
            initial_capacity = min(initial_capacity, c.kv_token_budget)
            req_pages = math.ceil(initial_capacity / c.page_size)

            if node_pages_used[seq_node] + req_pages <= per_node_effective_free[seq_node]:
                prefill_batch.append(c.uuid)
                node_pages_used[seq_node] += req_pages

        return prefill_batch


# ============================================================================
# Prefill execution (moved verbatim from BatchGenWorker.prefill /
# .prefill_prepacked). These are worker-parameterized free functions: the
# real worker is passed in and remains the single source of truth. No shared
# WorkerState; coupling is unchanged, only the code location.
# ============================================================================

def run_prefill(worker, batch):
    """
    Handle the prefill for a batch.
    batch: list of local indices
    """
    # Bind AttnWrapperBase.host_paged_kv_worker_view_aux BEFORE the decoder
    # loop. Without this binding, GLM-5's prefill indexer-K offload at
    # wrappers.py:_offload_prepacked_indexer_kv silently early-returns
    # (host_paged_kv_worker_view_aux is None), so the aux cache is never
    # populated for prompt tokens and any later decode past 2048 tokens
    # reads unwritten aux pages.
    # Prefill offloads KV directly to host via host_paged_kv_worker_view_aux;
    # it does NOT use the GPU paged KV manager. Binding host_*_aux here
    # ensures `_offload_prepacked_indexer_kv` actually pushes indexer K to
    # the host aux cache instead of early-returning on a None view.
    AttnWrapperBase.host_paged_kv_worker_view = getattr(worker.core_engine, "host_paged_kv_worker_view", None)
    AttnWrapperBase.host_paged_kv_worker_view_aux = getattr(worker, "host_paged_kv_worker_view_aux", None)

    if "deepseek" in worker.model_config.model_type:
        worker.model.model._use_flash_attention_2 = False

    # Dynamic padding: find max length within THIS batch, not global max
    # This is critical for long-tailed distributions
    batch_seq_lengths = [
        worker.query_book[query_idx].encoded["input_ids"].shape[1]
        for query_idx in batch
    ]
    batch_max_len = max(batch_seq_lengths)

    # Pad each sequence to batch_max_len and construct attention masks on-the-fly
    padded_input_ids = []
    padded_attention_masks = []
    for query_idx in batch:
        seq_input_ids = worker.query_book[query_idx].encoded["input_ids"]
        uuid = worker._local_to_uuid_map[query_idx]
        seq = worker.global_batch.get_sequence(uuid)
        prompt_len = seq.prompt_length
        seq_len = seq_input_ids.shape[1]

        # Construct attention mask from prompt_length (1s for valid tokens, 0s for padding)
        seq_attention_mask = torch.zeros((1, seq_len), dtype=torch.int64)
        seq_attention_mask[0, :prompt_len] = 1

        if seq_len < batch_max_len:
            # Pad with zeros (left-aligned tokens, right-padded)
            pad_len = batch_max_len - seq_len
            seq_input_ids = torch.cat([
                seq_input_ids,
                torch.zeros((1, pad_len), dtype=seq_input_ids.dtype)
            ], dim=1)
            seq_attention_mask = torch.cat([
                seq_attention_mask,
                torch.zeros((1, pad_len), dtype=seq_attention_mask.dtype)
            ], dim=1)

        padded_input_ids.append(seq_input_ids)
        padded_attention_masks.append(seq_attention_mask)

    input_ids = torch.cat(padded_input_ids, dim=0)
    attention_masks = torch.cat(padded_attention_masks, dim=0)

    num_prefill_micro_batches = math.ceil(
        len(batch) / worker.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size
    )
    prefill_micro_batch_input_ids = torch.split(
        input_ids,
        worker.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size,
    )
    prefill_micro_batch_attention_masks = torch.split(
        attention_masks,
        worker.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size,
    )
    if worker.rank == 0:
        logging.info(f"Number of prefill micro batches: {num_prefill_micro_batches}")

    cur_batch_start = 0
    output_tokens = []

    for micro_batch_idx in tqdm(range(num_prefill_micro_batches), desc="Prefill Micro Batch"):
        # Feed watchdog during long prefill operations
        worker.feed_watchdog()

        with torch.inference_mode():
            Attn_Wrapper.attention_mask = prefill_micro_batch_attention_masks[micro_batch_idx]
            Attn_Wrapper.position_ids = create_position_ids_from_attention_mask(
                prefill_micro_batch_attention_masks[micro_batch_idx]
            )

            cur_batch_size = prefill_micro_batch_input_ids[micro_batch_idx].shape[0]
            cur_batch_local = batch[cur_batch_start : cur_batch_start + cur_batch_size]

            # Pass local indices - the C++ layer handles rank offset internally
            Attn_Wrapper.cur_batch = worker._local_indices_to_global_seq_ids(cur_batch_local)

            cur_batch_start += cur_batch_size
            assert len(cur_batch_local) == cur_batch_size

            outputs = worker.model(
                prefill_micro_batch_input_ids[micro_batch_idx].to(worker.torch_device),
                attention_mask=prefill_micro_batch_attention_masks[micro_batch_idx].to(worker.torch_device),
                use_cache=False,
            )
            cur_batch_sequences = [
                worker.global_batch.get_sequence(worker._local_to_uuid_map[local_idx])
                for local_idx in cur_batch_local
            ]
            new_tokens = worker._select_tokens(outputs.logits[:, -1, :], cur_batch_sequences)
            output_tokens.append(new_tokens)

    new_tokens = torch.cat(output_tokens, dim=0)

    # Update sequence state after prefill
    # For evicted re-entry: first new token goes at decoded_length offset (not 0)
    # For fresh sequences: decoded_length is 0, so offset is 0 (same as before)
    new_tokens_cpu = new_tokens.cpu()
    for i, local_idx in enumerate(batch):
        uuid = worker._local_to_uuid_map[local_idx]
        seq = worker.global_batch.get_sequence(uuid)
        # Write token at correct offset (handles both fresh and re-entered sequences)
        token_pos = seq.decoded_length  # 0 for fresh, prev_decoded for re-entry
        worker.query_book[local_idx].decoded_tokens[:, token_pos] = new_tokens_cpu[i]
        seq.decoded_length = token_pos + 1
        seq.current_context_length = seq.original_prompt_length + seq.decoded_length

        # MODIFIED: Check for EOS respecting ignore_eos flag
        if worker._should_stop_at_eos(new_tokens_cpu[i].item()):
            seq.eos_reached = True

    return new_tokens


def run_prefill_prepacked(worker, batch):
    """
    Handle prefill for a batch using prepack optimization.

    Prepack combines multiple shorter sequences into rows to minimize padding waste,
    which is especially beneficial for MLP/MoE layers.

    Args:
        batch: list of local indices
    """
    # Bind AttnWrapperBase.host_paged_kv_worker_view_aux BEFORE the decoder
    # loop. Without this binding, GLM-5's prefill indexer-K offload at
    # wrappers.py:_offload_prepacked_indexer_kv silently early-returns
    # (host_paged_kv_worker_view_aux is None), so the aux cache is never
    # populated for prompt tokens and any later decode past 2048 tokens
    # reads unwritten aux pages.
    # Prefill offloads KV directly to host via host_paged_kv_worker_view_aux;
    # it does NOT use the GPU paged KV manager. Binding host_*_aux here
    # ensures `_offload_prepacked_indexer_kv` actually pushes indexer K to
    # the host aux cache instead of early-returning on a None view.
    AttnWrapperBase.host_paged_kv_worker_view = getattr(worker.core_engine, "host_paged_kv_worker_view", None)
    AttnWrapperBase.host_paged_kv_worker_view_aux = getattr(worker, "host_paged_kv_worker_view_aux", None)

    if "deepseek" in worker.model_config.model_type:
        worker.model.model._use_flash_attention_2 = False

    # Collect input_ids and attention_masks as lists for prepacking
    input_ids_list = []
    attention_mask_list = []
    seq_lengths = []

    for query_idx in batch:
        uuid = worker._local_to_uuid_map[query_idx]
        seq = worker.global_batch.get_sequence(uuid)
        query_entry = worker.query_book[query_idx]
        encoded = query_entry.encoded["input_ids"]
        if encoded.data_ptr() != seq.input_ids.data_ptr():
            raise RuntimeError(
                f"Rank {worker.rank}: stale query_book input_ids binding for "
                f"local_idx={query_idx} uuid={uuid[:8]} "
                f"(query_book_ptr={encoded.data_ptr():#x}, seq_ptr={seq.input_ids.data_ptr():#x})"
            )
        if query_entry.decoded_tokens.data_ptr() != seq.decoded_tokens.data_ptr():
            raise RuntimeError(
                f"Rank {worker.rank}: stale query_book decoded_tokens binding for "
                f"local_idx={query_idx} uuid={uuid[:8]} "
                f"(query_book_ptr={query_entry.decoded_tokens.data_ptr():#x}, "
                f"seq_ptr={seq.decoded_tokens.data_ptr():#x})"
            )
        # NO truncation: every prompt is tokenized to its OWN length.
        # An earlier `[:, :worker.max_input_length]` slice silently dropped
        # the tail of long LongBench prompts when max_input_length was
        # carried over from a smaller earlier admit batch, causing the
        # model to "continue" mid-sentence instead of answering. Bind
        # everything to seq.prompt_length directly.
        L = seq.prompt_length
        assert encoded.size(-1) >= L, (
            f"encoded prompt length {encoded.size(-1)} < seq.prompt_length {L} "
            f"for query_idx={query_idx} uuid={uuid[:8]}"
        )
        input_ids = encoded[:, :L]
        seq_lengths.append(L)

        # Per-seq mask marks the L valid positions for the prepacker.
        # Causal attention is enforced by FA varlen + cu_seqlens.
        attention_mask = torch.zeros_like(input_ids, dtype=torch.int64)
        attention_mask[0, :L] = 1

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)

    # Prepack sequences
    # Row capacity is set by planner in config (None = no limit, use max sequence length)
    row_capacity = worker.engine_config.Module_Batching_Config.prepack_row_capacity
    prepack_meta = prepack_sequences(
        input_ids_list,
        attention_mask_list,
        row_capacity=row_capacity,
        device=worker.torch_device,
    )

    # Log prepack statistics
    if worker.rank == 0:
        stats = get_prepack_stats(prepack_meta)
        logging.info(
            f"Prepack stats: {stats['num_sequences']} seqs -> {stats['num_packed_rows']} rows, "
            f"padding saved: {stats['padding_saved']} tokens, "
            f"efficiency: {stats['packing_efficiency']:.2%}"
        )

    # Create flattened tensors for prepacked forward
    # Flatten packed_input_ids to [total_tokens]
    total_tokens = sum(prepack_meta.original_seq_lengths)

    # Extract only valid tokens (non-padding) in order
    packed_input_ids_flat = []
    packed_position_ids_flat = []

    for seq_idx in range(prepack_meta.num_original_sequences):
        row_idx, start_pos = prepack_meta.pack_assignment[seq_idx]
        seq_len = prepack_meta.original_seq_lengths[seq_idx]

        # Extract tokens for this sequence
        seq_input_ids = prepack_meta.packed_input_ids[row_idx, start_pos:start_pos + seq_len]
        packed_input_ids_flat.append(seq_input_ids)

        # Position IDs are 0, 1, 2, ... for each sequence
        packed_position_ids_flat.append(torch.arange(seq_len, device=worker.torch_device))

    packed_input_ids_flat = torch.cat(packed_input_ids_flat, dim=0)  # [total_tokens]
    packed_position_ids_flat = torch.cat(packed_position_ids_flat, dim=0)  # [total_tokens]

    # Split sequences into micro-batches based on TOKEN count (not sequence count)
    # This prevents OOM when sequences have varying lengths
    # Token cap is set by planner in config, worker reads from config (no hardcoded values)
    MAX_TOKENS_PER_MICRO_BATCH = worker.engine_config.Module_Batching_Config.prefill_micro_batch_token_cap
    num_sequences = prepack_meta.num_original_sequences
    seq_lengths_list = prepack_meta.original_seq_lengths

    # Create micro-batches bounded by token count, optionally also by sum(L^2)
    # so the per-microbatch attention work (which is O(L^2)) doesn't pile up
    # on one micro-batch when a single very long sequence is present.
    import os as _os_mb
    _USE_L2_MB = _os_mb.environ.get("BATCHGEN_L2_BALANCE", "1") == "1"
    micro_batches, l2_cap = build_prefill_micro_batches(
        seq_lengths_list,
        MAX_TOKENS_PER_MICRO_BATCH,
        l2_balance=_USE_L2_MB,
    )
    total_tokens_all = sum(seq_lengths_list)

    if worker.rank == 0:
        logging.info(
            f"Prepacked prefill: {len(micro_batches)} micro batches, "
            f"{total_tokens_all:,} total tokens, max {MAX_TOKENS_PER_MICRO_BATCH:,} tokens/batch"
            + (f", l2_cap={l2_cap:,}" if l2_cap > 0 else "")
        )

    output_tokens = []

    with torch.inference_mode():
        for batch_idx, (seq_start, seq_end) in tqdm(
            enumerate(micro_batches),
            total=len(micro_batches),
            desc="Prepacked Prefill",
            disable=(worker.rank != 0)  # Only show progress on rank 0
        ):
            # Feed watchdog during long prefill operations
            worker.feed_watchdog()

            # Get sequences for this micro-batch
            batch_seq_lengths = seq_lengths_list[seq_start:seq_end]
            batch_num_seqs = seq_end - seq_start

            # Extract tokens for this micro-batch
            batch_input_ids = []
            batch_position_ids = []
            token_offset = sum(seq_lengths_list[:seq_start])  # Offset into flat tensors

            for seq_idx in range(seq_start, seq_end):
                seq_len = seq_lengths_list[seq_idx]
                # Calculate where this sequence's tokens are in the flat tensor
                seq_token_start = sum(seq_lengths_list[:seq_idx])
                seq_token_end = seq_token_start + seq_len

                batch_input_ids.append(packed_input_ids_flat[seq_token_start:seq_token_end])
                batch_position_ids.append(packed_position_ids_flat[seq_token_start:seq_token_end])

            batch_input_ids_flat = torch.cat(batch_input_ids, dim=0)
            batch_position_ids_flat = torch.cat(batch_position_ids, dim=0)

            batch_local_indices = batch[seq_start:seq_end]
            local_to_global_seq_id_map = {}
            for local_idx in batch_local_indices:
                uuid = worker._local_to_uuid_map.get(local_idx)
                if uuid is None:
                    raise RuntimeError(
                        f"Rank {worker.rank}: missing UUID for prefill local_idx={local_idx}"
                    )
                seq = worker.global_batch.get_sequence(uuid)
                if seq is None:
                    raise RuntimeError(
                        f"Rank {worker.rank}: missing SequenceEntry for prefill uuid={uuid[:8]}"
                    )
                local_to_global_seq_id_map[local_idx] = seq.global_idx

            batch_spans = build_prefill_sequence_spans(
                batch_local_indices,
                batch_seq_lengths,
                worker._local_to_uuid_map,
                local_to_global_seq_id_map,
            )
            batch_cu_seqlens = torch.tensor(
                prefill_sequence_spans_to_cu_seqlens(batch_spans),
                dtype=torch.int32,
                device=worker.torch_device,
            )
            batch_max_seqlen = max(batch_seq_lengths)

            # Set up Attn_Wrapper for this micro-batch
            Attn_Wrapper.prepack_mode = True
            Attn_Wrapper.prepack_cu_seqlens = batch_cu_seqlens
            Attn_Wrapper.prepack_max_seqlen = batch_max_seqlen
            Attn_Wrapper.prepack_num_sequences = batch_num_seqs
            Attn_Wrapper.prepack_seq_lengths = batch_seq_lengths
            Attn_Wrapper.position_ids = batch_position_ids_flat
            Attn_Wrapper.cur_batch = prefill_sequence_spans_to_global_seq_ids(batch_spans)

            # CRITICAL: Also bind to AttnWrapperBase for models using new wrapper system (GPT-OSS)
            # Without this, GPT-OSS uses _forward_prefill instead of _forward_prefill_prepacked,
            # which does NOT offload KV to host, causing decode to read garbage.
            AttnWrapperBase.prepack_mode = True
            AttnWrapperBase.prepack_cu_seqlens = batch_cu_seqlens
            AttnWrapperBase.prepack_max_seqlen = batch_max_seqlen
            AttnWrapperBase.prepack_num_sequences = batch_num_seqs
            AttnWrapperBase.prepack_seq_lengths = batch_seq_lengths
            AttnWrapperBase.position_ids = batch_position_ids_flat
            AttnWrapperBase.cur_batch = Attn_Wrapper.cur_batch

            # Embed tokens
            inputs_embeds = worker.model.model.embed_tokens(batch_input_ids_flat.to(worker.torch_device))

            # Reshape to 3D: [1, batch_total_tokens, hidden_dim]
            hidden_states = inputs_embeds.unsqueeze(0)

            for layer_idx, decoder_layer in enumerate(worker.model.model.layers):
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=None,
                    position_ids=None,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                )
                hidden_states = layer_outputs[0]

            # Final norm
            hidden_states = worker.model.model.norm(hidden_states)

            # Extract last token hidden states for each sequence
            last_token_indices = batch_cu_seqlens[1:] - 1
            last_token_hidden = hidden_states[0, last_token_indices, :]

            # lm_head matmul: BF16 by default (matches HF / SGLang / vLLM).
            # Opt into FP32-cast via BATCHGEN_GLM5_LMHEAD_FP32=1 for debugging.
            if os.environ.get("BATCHGEN_GLM5_LMHEAD_FP32", "0") == "1":
                logits = torch.nn.functional.linear(
                    last_token_hidden.float(),
                    worker.model.lm_head.weight.float(),
                    worker.model.lm_head.bias.float() if hasattr(worker.model.lm_head, 'bias') and worker.model.lm_head.bias is not None else None
                )
            else:
                logits = torch.nn.functional.linear(
                    last_token_hidden,
                    worker.model.lm_head.weight,
                    worker.model.lm_head.bias if hasattr(worker.model.lm_head, 'bias') and worker.model.lm_head.bias is not None else None
                ).float()

            batch_sequences = [
                worker.global_batch.get_sequence(worker._local_to_uuid_map[local_idx])
                for local_idx in batch_local_indices
            ]
            batch_new_tokens = worker._select_tokens(logits, batch_sequences)
            if batch_new_tokens.shape[0] != batch_num_seqs:
                raise RuntimeError(
                    f"Rank {worker.rank}: prefill token selection shape mismatch, "
                    f"got {batch_new_tokens.shape[0]} rows for {batch_num_seqs} sequences"
                )
            output_tokens.append(batch_new_tokens)

    # Reset prepack mode
    Attn_Wrapper.prepack_mode = False
    Attn_Wrapper.prepack_cu_seqlens = None
    Attn_Wrapper.prepack_max_seqlen = None
    Attn_Wrapper.prepack_num_sequences = None
    Attn_Wrapper.prepack_seq_lengths = None

    # Also reset AttnWrapperBase for models using new wrapper system (GPT-OSS)
    AttnWrapperBase.prepack_mode = False
    AttnWrapperBase.prepack_cu_seqlens = None
    AttnWrapperBase.prepack_max_seqlen = None
    AttnWrapperBase.prepack_num_sequences = None
    AttnWrapperBase.prepack_seq_lengths = None

    # Log timing summary for GPT-OSS if timing was enabled
    worker._log_prefill_timing()

    new_tokens = torch.cat(output_tokens, dim=0)
    if new_tokens.shape[0] != len(batch):
        raise RuntimeError(
            f"Rank {worker.rank}: prefill writeback shape mismatch, "
            f"got {new_tokens.shape[0]} rows for {len(batch)} local sequences"
        )

    # Update sequence state after prefill
    # For evicted re-entry: first new token goes at decoded_length offset (not 0)
    new_tokens_cpu = new_tokens.cpu()
    for i, local_idx in enumerate(batch):
        uuid = worker._local_to_uuid_map[local_idx]
        seq = worker.global_batch.get_sequence(uuid)
        token_pos = seq.decoded_length  # 0 for fresh, prev_decoded for re-entry
        worker.query_book[local_idx].decoded_tokens[:, token_pos] = new_tokens_cpu[i]
        seq.decoded_length = token_pos + 1
        seq.current_context_length = seq.original_prompt_length + seq.decoded_length

        # Check for EOS respecting ignore_eos flag
        if worker._should_stop_at_eos(new_tokens_cpu[i].item()):
            seq.eos_reached = True

    return new_tokens

