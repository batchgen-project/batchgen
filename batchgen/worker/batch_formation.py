"""BatchFormation — stateless tokenize / assign_ranks / build_query_book primitives.

Each method takes an explicit iterable of UUIDs and operates on the sequences
currently in `state.global_batch`. There is no "initial" vs "admit" variant:
the orchestrator calls the same three methods at batch start and at every
pool-mode admission (plan redesign, section "Pool mode, redesigned").

Dependencies:
  - `state: WorkerState` — holds `global_batch` and rank identity.
  - `tokenizer: TokenizerBackend` — per-rank text → token-id encoder.
  - `collectives: CollectiveBackend` — cross-rank gather of tokenized slices.
  - `index: IndexManager` — local-index allocation for the query book.
  - `model_context_length: int` — overflow threshold for `reject_overflow`.

Convention: tokenize is parallelized across ranks — each rank encodes a
`range(rank, len(uuids), world_size)` slice then the results are merged via
`all_gather_object`. This avoids N-way tokenizer contention and serializing
long prompts through rank 0.
"""

from __future__ import annotations

from typing import Any

import torch

from batchgen.worker.indexing import IndexManager
from batchgen.worker.protocols import (
    UUID,
    CollectiveBackend,
    TokenizerBackend,
)
from batchgen.worker.state import WorkerState


class BatchFormation:
    """Pure primitives — no status transitions, no KV ops, no model calls."""

    def __init__(
        self,
        state: WorkerState,
        tokenizer: TokenizerBackend,
        collectives: CollectiveBackend,
        index: IndexManager,
        model_context_length: int,
    ) -> None:
        self._state = state
        self._tokenizer = tokenizer
        self._collectives = collectives
        self._index = index
        self._model_context_length = model_context_length

    # -- tokenize ----------------------------------------------------------

    def tokenize(self, uuids: list[UUID]) -> None:
        """Tokenize text for each UUID and write `input_ids` + `prompt_length`.

        Parallel across ranks: each rank encodes its
        `range(rank, len(uuids), world_size)` slice, then `all_gather_object`
        merges all local results so every rank ends with the full mapping
        written back onto `state.global_batch`.

        UUIDs whose sequence is missing or has `text is None` are skipped
        silently (handled elsewhere — tokenize is a primitive).
        """
        rank = self._state.rank
        world = self._state.world_size
        local_results: dict[str, list[int]] = {}
        for i in range(rank, len(uuids), world):
            uuid = uuids[i]
            seq = self._state.global_batch.get_sequence(uuid)
            if seq is None or seq.text is None:
                continue
            local_results[uuid] = self._tokenizer.encode(seq.text)

        gathered: list[Any] = [None] * world
        self._collectives.all_gather_object(gathered, local_results)

        for rank_result in gathered:
            if not rank_result:
                continue
            for uuid, ids in rank_result.items():
                seq = self._state.global_batch.get_sequence(uuid)
                if seq is None:
                    continue
                tensor_ids = torch.tensor(ids, dtype=torch.long)
                seq.input_ids = tensor_ids
                seq.prompt_length = len(ids)
                seq.current_context_length = len(ids)
                seq.original_prompt_length = len(ids)

    # -- assign_ranks ------------------------------------------------------

    def assign_ranks(self, uuids: list[UUID]) -> None:
        """Round-robin `assigned_rank` across `world_size`, deterministic across ranks.

        Ordering is taken from the sorted `uuids` list so every rank assigns
        identical ranks to identical UUIDs without any collective. UUIDs
        missing from `global_batch` are filtered OUT before round-robin so
        they do not silently consume a slot; the remaining UUIDs see a
        gap-free assignment.
        """
        world = self._state.world_size
        present = sorted(u for u in uuids if self._state.global_batch.get_sequence(u) is not None)
        for i, uuid in enumerate(present):
            self._state.global_batch.assign_rank(uuid, i % world)

    # -- build_query_book --------------------------------------------------

    def build_query_book(self, uuids: list[UUID]) -> list[int]:
        """Register every UUID assigned to this rank into `IndexManager`.

        Returns the list of allocated local indices in the same order as
        the input `uuids`. UUIDs that belong to a different rank, UUIDs
        already registered, and UUIDs missing from `global_batch` are
        skipped silently.
        """
        allocated: list[int] = []
        for uuid in uuids:
            seq = self._state.global_batch.get_sequence(uuid)
            if seq is None:
                continue
            if seq.assigned_rank != self._state.rank:
                continue
            if self._index.is_registered(uuid):
                continue
            local_idx = self._index.register(uuid)
            allocated.append(local_idx)
        return allocated

    # -- reject_overflow ---------------------------------------------------

    def reject_overflow(self, uuids: list[UUID]) -> set[UUID]:
        """Return UUIDs whose tokenized prompt >= `model_context_length`.

        Uses `>=` (not `>`) to match main's behavior at
        `batchgen_worker.py:951`: a prompt exactly equal to the model's
        context window leaves zero room for generation.

        Call AFTER `tokenize()` so `seq.prompt_length` reflects the actual
        token count, not the initial pre-tokenize placeholder.
        """
        rejected: set[UUID] = set()
        for uuid in uuids:
            seq = self._state.global_batch.get_sequence(uuid)
            if seq is None:
                continue
            if seq.prompt_length >= self._model_context_length:
                rejected.add(uuid)
        return rejected


__all__ = ["BatchFormation"]
