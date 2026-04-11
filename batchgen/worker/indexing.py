"""IndexManager — local-index allocation and UUID mapping.

Owns four fields on `WorkerState`:
  - `local_to_uuid_map: dict[int, str]`
  - `uuid_to_local_map: dict[str, int]`
  - `free_local_indices: set[int]`
  - `next_local_idx: int`

The local index is the slot number each sequence occupies in the worker's
preallocated `QueryBookBufferPool`. Pool mode recycles slots aggressively:
`register` pops from `free_local_indices` before growing `next_local_idx`, and
`unregister` returns slots to the free set. The lockstep invariant with
`QueryBookBufferPool._free_slots` is enforced by the orchestrator (see plan
Decision #7), not here.

IndexManager does NOT own status-based filtering — that's `SequenceBatch`'s
job via `get_sequences_by_status`. Callers that need "all UUIDs in state X"
go through `state.global_batch` directly.
"""

from __future__ import annotations

from batchgen.worker.state import WorkerState


class DuplicateSequenceError(Exception):
    """Raised when `register` is called with a UUID that is already mapped."""


class UnknownSequenceError(KeyError):
    """Raised when `unregister`/`local_for_uuid` is called with an unmapped UUID."""


class IndexManager:
    """Authoritative writer of the local-index state on `WorkerState`.

    Constructor takes `WorkerState` so every mutation lands on the shared
    container. The manager holds no shadow copies.
    """

    def __init__(self, state: WorkerState) -> None:
        self._state = state

    # -- allocation --------------------------------------------------------

    def register(self, uuid: str) -> int:
        """Allocate a local index for a new UUID.

        Pops from `free_local_indices` first; grows `next_local_idx` only when
        the free set is empty. Raises `DuplicateSequenceError` if the UUID is
        already registered.
        """
        if uuid in self._state.uuid_to_local_map:
            raise DuplicateSequenceError(
                f"IndexManager.register: uuid {uuid!r} already has local index "
                f"{self._state.uuid_to_local_map[uuid]}"
            )
        if self._state.free_local_indices:
            local_idx = min(self._state.free_local_indices)
            self._state.free_local_indices.remove(local_idx)
        else:
            local_idx = self._state.next_local_idx
            self._state.next_local_idx += 1
        self._state.local_to_uuid_map[local_idx] = uuid
        self._state.uuid_to_local_map[uuid] = local_idx
        return local_idx

    def unregister(self, uuid: str) -> None:
        """Release a UUID's local index back to the free set.

        Raises `UnknownSequenceError` if the UUID was never registered.
        """
        if uuid not in self._state.uuid_to_local_map:
            raise UnknownSequenceError(
                f"IndexManager.unregister: uuid {uuid!r} is not registered"
            )
        local_idx = self._state.uuid_to_local_map.pop(uuid)
        self._state.local_to_uuid_map.pop(local_idx, None)
        self._state.free_local_indices.add(local_idx)

    # -- lookups -----------------------------------------------------------

    def local_for_uuid(self, uuid: str) -> int:
        try:
            return self._state.uuid_to_local_map[uuid]
        except KeyError as exc:
            raise UnknownSequenceError(
                f"IndexManager.local_for_uuid: uuid {uuid!r} is not registered"
            ) from exc

    def uuid_for_local(self, local_idx: int) -> str:
        try:
            return self._state.local_to_uuid_map[local_idx]
        except KeyError as exc:
            raise UnknownSequenceError(
                f"IndexManager.uuid_for_local: local_idx {local_idx} is not registered"
            ) from exc

    def global_ids_for_local(self, local_indices: list[int]) -> list[int]:
        """Map a list of local indices to their `SequenceEntry.global_idx` values.

        Preserves order. Every local_idx must be registered and resolve to a
        sequence present in `state.global_batch`; unknown indices raise
        `UnknownSequenceError`.
        """
        out: list[int] = []
        for local_idx in local_indices:
            uuid = self.uuid_for_local(local_idx)
            seq = self._state.global_batch.get_sequence(uuid)
            if seq is None:
                raise UnknownSequenceError(
                    f"IndexManager.global_ids_for_local: uuid {uuid!r} for "
                    f"local_idx {local_idx} is not in state.global_batch"
                )
            out.append(seq.global_idx)
        return out

    # -- introspection -----------------------------------------------------

    def is_registered(self, uuid: str) -> bool:
        return uuid in self._state.uuid_to_local_map

    def live_count(self) -> int:
        """Number of UUIDs currently holding a local index."""
        return len(self._state.uuid_to_local_map)


__all__ = [
    "DuplicateSequenceError",
    "UnknownSequenceError",
    "IndexManager",
]
