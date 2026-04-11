"""SyncCoordinator — all cross-rank worker state synchronization.

Every `torch.distributed` call in a worker handler funnels through this
module. Handlers never call `dist.*` directly (plan "Protocol DI in detail").

Exposed methods:
  - `sync_metadata(uuids)`: `all_gather_object` of `SeqSnapshot` per rank;
    fast-fails on CTX invariant violation at both sender pre-gather and
    receiver post-gather. Absorbs cross-rank updates for non-owned
    sequences (every rank holds a shadow of every sequence's metadata).
  - `sync_completion_status(uuids)`: tensor-based union via `all_reduce_max`;
    replaces main's only live completion-sync method (the other two main
    variants are dead code per M0 audit).
  - `sync_decode_uuids(candidates)`: tensor-based intersection via
    `all_reduce_min`; returns the subset of candidates every rank agrees
    exists in their local `global_batch`.
  - `gather_rank_token_counts(local_count)`: `all_gather_into_tensor` of a
    single int per rank, returns the per-rank list.

All methods are CPU-safe when given a `torch_device=torch.device("cpu")` on
the WorkerState; production uses `cuda:<device>`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from batchgen.worker.exceptions import CtxInvariantViolation
from batchgen.worker.protocols import UUID, CollectiveBackend
from batchgen.worker.state import WorkerState


@dataclass(frozen=True)
class SeqSnapshot:
    """Per-sequence metadata exchanged during `sync_metadata`.

    Only the fields that downstream handlers (boundary/decode/host
    rebalancer) read across ranks are included. Adding a field here is
    additive — receiving ranks update just the listed attributes on their
    shadow copies.
    """

    uuid: UUID
    prompt_length: int
    original_prompt_length: int
    decoded_length: int
    current_context_length: int
    gpu_pages_allocated: int
    host_pages_allocated: int
    eos_reached: bool

    @classmethod
    def from_seq(cls, seq) -> "SeqSnapshot":  # type: ignore[no-untyped-def]
        return cls(
            uuid=seq.uuid,
            prompt_length=seq.prompt_length,
            original_prompt_length=seq.original_prompt_length,
            decoded_length=seq.decoded_length,
            current_context_length=seq.current_context_length,
            gpu_pages_allocated=seq.gpu_pages_allocated,
            host_pages_allocated=seq.host_pages_allocated,
            eos_reached=seq.eos_reached,
        )


class SyncCoordinator:
    def __init__(
        self,
        state: WorkerState,
        collectives: CollectiveBackend,
    ) -> None:
        self._state = state
        self._collectives = collectives

    # ------------------------------------------------------------------
    # sync_metadata — gather + CTX fast-fail + cross-rank absorb
    # ------------------------------------------------------------------

    def sync_metadata(self, uuids: list[UUID]) -> None:
        """Gather per-rank `SeqSnapshot`s for `uuids`; fast-fail on CTX drift.

        Step 1 (sender side): build the local payload for sequences this
        rank OWNS (`seq.assigned_rank == state.rank`). Verify each
        snapshot's CTX invariant BEFORE issuing the collective so a
        broken rank raises without affecting peers.

        Step 2: `all_gather_object` to collect payloads from every rank.

        Step 3 (receiver side): for every received snapshot, re-check the
        CTX invariant (a peer could have sent us a broken value). On
        success, absorb the received values onto non-owned local
        sequences so every rank's shadow stays consistent.

        Raises:
            CtxInvariantViolation: sender or receiver CTX invariant failed.
                Sender-side raises emit zero collectives. Receiver-side
                raises happen after the `all_gather_object` returns.
        """
        rank = self._state.rank
        local: dict[UUID, SeqSnapshot] = {}
        for uuid in uuids:
            seq = self._state.global_batch.get_sequence(uuid)
            if seq is None:
                continue
            if seq.assigned_rank != rank:
                continue
            expected = seq.original_prompt_length + seq.decoded_length
            if seq.current_context_length != expected:
                raise CtxInvariantViolation(
                    uuid=uuid,
                    side="sender",
                    had=seq.current_context_length,
                    expected=expected,
                )
            local[uuid] = SeqSnapshot.from_seq(seq)

        gathered: list[dict[UUID, SeqSnapshot] | None] = [None] * self._state.world_size
        self._collectives.all_gather_object(gathered, local)  # type: ignore[arg-type]

        for rank_payload in gathered:
            if not rank_payload:
                continue
            for uuid, snap in rank_payload.items():
                expected = snap.original_prompt_length + snap.decoded_length
                if snap.current_context_length != expected:
                    raise CtxInvariantViolation(
                        uuid=uuid,
                        side="receiver",
                        had=snap.current_context_length,
                        expected=expected,
                    )
                seq = self._state.global_batch.get_sequence(uuid)
                if seq is None:
                    continue
                if seq.assigned_rank == rank:
                    continue  # owned locally — do not overwrite authoritative copy
                seq.prompt_length = snap.prompt_length
                seq.original_prompt_length = snap.original_prompt_length
                seq.decoded_length = snap.decoded_length
                seq.current_context_length = snap.current_context_length
                seq.gpu_pages_allocated = snap.gpu_pages_allocated
                seq.host_pages_allocated = snap.host_pages_allocated
                seq.eos_reached = snap.eos_reached

    # ------------------------------------------------------------------
    # sync_completion_status — tensor union of eos_reached across ranks
    # ------------------------------------------------------------------

    def sync_completion_status(self, uuids: list[UUID]) -> set[UUID]:
        """Return the set of UUIDs any rank reports as completed.

        Each rank writes ``1`` into slot ``i`` if ``uuids[i]`` is locally
        ``eos_reached`` (OR completed via any other local predicate the
        caller folded into ``eos_reached``). ``all_reduce_max`` gives the
        cross-rank union. Returns empty set on empty input without
        issuing any collective.
        """
        if not uuids:
            return set()
        n = len(uuids)
        tensor = torch.zeros(n, dtype=torch.int32, device=self._state.torch_device)
        for i, uuid in enumerate(uuids):
            seq = self._state.global_batch.get_sequence(uuid)
            if seq is not None and seq.eos_reached:
                tensor[i] = 1
        self._collectives.all_reduce_max(tensor)
        return {uuids[i] for i in range(n) if int(tensor[i].item()) == 1}

    # ------------------------------------------------------------------
    # sync_decode_uuids — tensor intersection of local presence
    # ------------------------------------------------------------------

    def sync_decode_uuids(self, candidates: list[UUID]) -> set[UUID]:
        """Return the subset of `candidates` every rank holds locally.

        Each rank writes ``1`` if the candidate is present in
        ``state.global_batch``, ``0`` otherwise. ``all_reduce_min`` selects
        the intersection — a value survives only when every rank saw it.
        """
        if not candidates:
            return set()
        n = len(candidates)
        tensor = torch.ones(n, dtype=torch.int32, device=self._state.torch_device)
        for i, uuid in enumerate(candidates):
            if self._state.global_batch.get_sequence(uuid) is None:
                tensor[i] = 0
        self._collectives.all_reduce_min(tensor)
        return {candidates[i] for i in range(n) if int(tensor[i].item()) == 1}

    # ------------------------------------------------------------------
    # gather_rank_token_counts — per-rank int via all_gather_into_tensor
    # ------------------------------------------------------------------

    def gather_rank_token_counts(self, local_count: int) -> list[int]:
        """Gather a single int per rank, return the per-rank list.

        Replaces main's `all_reduce(MAX)` call sites with
        `all_gather_into_tensor` — the MoE path needs the full per-rank
        distribution (not just the max) to compute batch-invariant gate
        and lm_head activations (plan "Inventory of main's behavior",
        NCCL tree / MoE padding mask changes).
        """
        world = self._state.world_size
        out = torch.zeros(world, dtype=torch.int64, device=self._state.torch_device)
        local = torch.tensor([local_count], dtype=torch.int64, device=self._state.torch_device)
        self._collectives.all_gather_into_tensor(out, local)
        return [int(out[i].item()) for i in range(world)]


__all__ = ["SeqSnapshot", "SyncCoordinator"]
