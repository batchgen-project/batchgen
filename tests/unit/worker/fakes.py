"""Hand-rolled fakes for every batchgen.worker Protocol.

Fakes are plain Python classes that structurally satisfy the Protocols in
`batchgen.worker.protocols`. They are CPU-only, deterministic, and record
every call so tests can assert on call order and argument shapes.

Design rules:
  - Never import torch.distributed, CUDA, or any production backend.
  - Record the minimum information needed to write crisp assertions:
    method name, positional arg shapes/values, keyword args if non-trivial.
  - Accept injected responses for methods that return data the caller
    consumes (e.g. all_gather_object results, GPU page allocations).
  - Raise clearly when a required injection is missing — failing fakes
    should point at the test fixture, not the code under test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Collective backend
# ---------------------------------------------------------------------------


@dataclass
class CollectiveCall:
    """One recorded call on `FakeCollectiveBackend`."""

    name: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


class FakeCollectiveBackend:
    """Deterministic CPU-only stand-in for `torch.distributed` collectives.

    Parameters:
        rank: This rank's id.
        world_size: Total ranks in the fake world.
        all_gather_object_responses: Per-call injection for `all_gather_object`.
            Each element is the `obj_list` that will be returned for the
            corresponding call. Consumed in call order.
        all_reduce_max_deltas: Per-call injection for `all_reduce_max`.
            Each element is a tensor with the same shape as the input; the
            output is `max(self_value, injected)`.
        broadcast_object_responses: Per-call injection for `broadcast_object`.
            Each element is the `obj_list` returned to non-src ranks.
    """

    def __init__(
        self,
        rank: int = 0,
        world_size: int = 1,
        *,
        all_gather_object_responses: list[list[Any]] | None = None,
        all_reduce_max_deltas: list[torch.Tensor] | None = None,
        broadcast_object_responses: list[list[Any]] | None = None,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.calls: list[CollectiveCall] = []
        self._agather_obj_resp = list(all_gather_object_responses or [])
        self._arm_deltas = list(all_reduce_max_deltas or [])
        self._bcast_obj_resp = list(broadcast_object_responses or [])

    # -- record helper ------------------------------------------------------
    def _record(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append(CollectiveCall(name=name, args=args, kwargs=kwargs))

    # -- Protocol methods ---------------------------------------------------
    def all_reduce_max(self, tensor: torch.Tensor) -> None:
        self._record("all_reduce_max", tuple(tensor.shape))
        if self._arm_deltas:
            delta = self._arm_deltas.pop(0)
            tensor.copy_(torch.maximum(tensor, delta))

    def all_reduce_sum(self, tensor: torch.Tensor) -> None:
        self._record("all_reduce_sum", tuple(tensor.shape))

    def all_gather_tensor(
        self, tensor_list: list[torch.Tensor], tensor: torch.Tensor
    ) -> None:
        self._record(
            "all_gather_tensor",
            len(tensor_list),
            tuple(tensor.shape),
        )
        for i in range(len(tensor_list)):
            if i == self.rank:
                tensor_list[i].copy_(tensor)

    def all_gather_into_tensor(
        self, out: torch.Tensor, tensor: torch.Tensor
    ) -> None:
        self._record(
            "all_gather_into_tensor",
            tuple(out.shape),
            tuple(tensor.shape),
        )

    def all_gather_object(self, obj_list: list[Any], obj: Any) -> None:
        self._record("all_gather_object", len(obj_list))
        if not self._agather_obj_resp:
            # Default: every rank sees only self's contribution.
            for i in range(len(obj_list)):
                obj_list[i] = obj if i == self.rank else None
            return
        injected = self._agather_obj_resp.pop(0)
        if len(injected) != len(obj_list):
            raise AssertionError(
                f"FakeCollectiveBackend.all_gather_object: injected response "
                f"len={len(injected)} but obj_list len={len(obj_list)}"
            )
        for i, value in enumerate(injected):
            obj_list[i] = obj if i == self.rank else value

    def broadcast_tensor(self, tensor: torch.Tensor, src: int) -> None:
        self._record("broadcast_tensor", tuple(tensor.shape), src=src)

    def broadcast_object(self, obj_list: list[Any], src: int) -> None:
        self._record("broadcast_object", len(obj_list), src=src)
        if self.rank == src:
            return
        if not self._bcast_obj_resp:
            raise AssertionError(
                "FakeCollectiveBackend.broadcast_object: non-src rank has no "
                "injected response; provide broadcast_object_responses"
            )
        injected = self._bcast_obj_resp.pop(0)
        if len(injected) != len(obj_list):
            raise AssertionError(
                f"FakeCollectiveBackend.broadcast_object: injected response "
                f"len={len(injected)} but obj_list len={len(obj_list)}"
            )
        for i, value in enumerate(injected):
            obj_list[i] = value

    def barrier(self) -> None:
        self._record("barrier")

    # -- helpers for assertions --------------------------------------------
    def call_names(self) -> list[str]:
        return [c.name for c in self.calls]


# ---------------------------------------------------------------------------
# GPU / host KV backends
# ---------------------------------------------------------------------------


class FakeGpuKvBackend:
    """In-memory GPU paged KV accounting.

    Tracks `free_count` and a `uuid -> list[PageId]` allocation map. Pages are
    integer ids drawn sequentially from a monotonic counter. Release returns
    pages to the free pool without recycling ids (makes tests easier to read).
    """

    def __init__(self, free_pages: int = 64) -> None:
        self._free = free_pages
        self._next_page_id = 0
        self._allocs: dict[str, list[int]] = {}
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self._rebuilt_page_tables: list[list[str]] = []

    def _record(self, name: str, *args: Any) -> None:
        self.calls.append((name, args))

    def allocate_pages(self, uuid: str, n: int) -> list[int]:
        self._record("allocate_pages", uuid, n)
        if n > self._free:
            raise RuntimeError(
                f"FakeGpuKvBackend: allocate_pages({uuid}, {n}) exceeds "
                f"free={self._free}"
            )
        pages = [self._next_page_id + i for i in range(n)]
        self._next_page_id += n
        self._free -= n
        self._allocs.setdefault(uuid, []).extend(pages)
        return pages

    def release_pages(self, uuid: str) -> None:
        self._record("release_pages", uuid)
        pages = self._allocs.pop(uuid, [])
        self._free += len(pages)

    def extend_pages(self, uuid: str, n: int) -> list[int]:
        self._record("extend_pages", uuid, n)
        if uuid not in self._allocs:
            raise RuntimeError(
                f"FakeGpuKvBackend: extend_pages({uuid}) before allocate"
            )
        return self.allocate_pages(uuid, n)

    def append_kv(self, uuid: str, layer: int, kv: torch.Tensor) -> None:
        self._record("append_kv", uuid, layer, tuple(kv.shape))

    def free_pages(self) -> int:
        return self._free

    def rebuild_page_table(self, uuids: list[str]) -> None:
        self._record("rebuild_page_table", tuple(uuids))
        self._rebuilt_page_tables.append(list(uuids))

    # -- test helpers ------------------------------------------------------
    def allocated_pages(self, uuid: str) -> list[int]:
        return list(self._allocs.get(uuid, []))

    def live_uuids(self) -> set[str]:
        return set(self._allocs)


class FakeHostKvBackend:
    """In-memory host KV page store + async handle stub.

    `load_to_gpu_async` returns a monotonically-increasing integer handle.
    Tests can inspect `recent_handles` to assert which uuids were kicked off.
    """

    def __init__(self, free_pages: int = 256) -> None:
        self._free = free_pages
        self._next_page_id = 0
        self._allocs: dict[str, list[int]] = {}
        self._next_handle = 1
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.recent_handles: list[tuple[str, list[int], int]] = []

    def _record(self, name: str, *args: Any) -> None:
        self.calls.append((name, args))

    def allocate_pages(self, uuid: str, n: int) -> list[int]:
        self._record("allocate_pages", uuid, n)
        if n > self._free:
            raise RuntimeError(
                f"FakeHostKvBackend: allocate_pages({uuid}, {n}) exceeds "
                f"free={self._free}"
            )
        pages = [self._next_page_id + i for i in range(n)]
        self._next_page_id += n
        self._free -= n
        self._allocs.setdefault(uuid, []).extend(pages)
        return pages

    def release_pages(self, uuid: str) -> None:
        self._record("release_pages", uuid)
        pages = self._allocs.pop(uuid, [])
        self._free += len(pages)

    def load_to_gpu_async(self, uuid: str, page_ids: list[int]) -> int:
        self._record("load_to_gpu_async", uuid, tuple(page_ids))
        handle = self._next_handle
        self._next_handle += 1
        self.recent_handles.append((uuid, list(page_ids), handle))
        return handle

    def free_pages(self) -> int:
        return self._free


# ---------------------------------------------------------------------------
# Tokenizer / model executor
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Deterministic byte-level tokenizer for BatchFormation tests.

    `encode(text)` returns `[ord(c) for c in text]` truncated to `max_len`.
    `eos_token_ids` is a set (plural) per the `conventions.md` rule.
    """

    def __init__(
        self,
        eos_token_ids: set[int] | None = None,
        max_len: int | None = None,
    ) -> None:
        self.eos_token_ids: set[int] = set(eos_token_ids or {0})
        self._max_len = max_len
        self.encode_calls: list[str] = []
        self.decode_calls: list[list[int]] = []

    def encode(self, text: str) -> list[int]:
        self.encode_calls.append(text)
        ids = [ord(c) for c in text]
        if self._max_len is not None:
            ids = ids[: self._max_len]
        return ids

    def decode(self, ids: list[int]) -> str:
        """Inverse of encode for ascii-range ids; space-joined int form otherwise."""
        self.decode_calls.append(list(ids))
        try:
            return "".join(chr(i) for i in ids)
        except (ValueError, OverflowError):
            return " ".join(str(i) for i in ids)


class FakeModelExecutor:
    """Canned prefill/decode outputs for scheduler tests."""

    def __init__(
        self,
        prefill_output: Any = None,
        decode_output: Any = None,
    ) -> None:
        self._prefill_out = prefill_output
        self._decode_out = decode_output
        self.prefill_batches: list[Any] = []
        self.decode_batches: list[Any] = []

    def forward_prefill(self, batch: Any) -> Any:
        self.prefill_batches.append(batch)
        return self._prefill_out

    def forward_decode(self, batch: Any) -> Any:
        self.decode_batches.append(batch)
        return self._decode_out


# ---------------------------------------------------------------------------
# Lifespan logger / clock / response sink
# ---------------------------------------------------------------------------


@dataclass
class LifespanEvent:
    seq: Any
    event: Any
    detail: dict[str, Any]


class RecordingLifespanLogger:
    """Appends every `log()` call to `events` for test assertions."""

    def __init__(self) -> None:
        self.events: list[LifespanEvent] = []

    def log(self, seq: Any, event: Any, detail: dict[str, Any]) -> None:
        self.events.append(LifespanEvent(seq=seq, event=event, detail=dict(detail)))

    def events_for(self, seq: Any) -> list[LifespanEvent]:
        return [e for e in self.events if e.seq is seq]


class FakeClock:
    """Monotonic counter clock. Default increments by `step` per `now()` call."""

    def __init__(self, start: float = 0.0, step: float = 1.0) -> None:
        self._t = start
        self._step = step
        self.call_count = 0

    def now(self) -> float:
        value = self._t
        self._t += self._step
        self.call_count += 1
        return value


class FakeResponseSink:
    """Records every finished-sequence payload routed through it."""

    def __init__(self) -> None:
        self.reported: dict[str, dict[str, Any]] = {}
        self.call_order: list[str] = []

    def put(self, uuid: str, payload: dict[str, Any]) -> None:
        self.reported[uuid] = dict(payload)
        self.call_order.append(uuid)


__all__ = [
    "CollectiveCall",
    "FakeCollectiveBackend",
    "FakeGpuKvBackend",
    "FakeHostKvBackend",
    "FakeTokenizer",
    "FakeModelExecutor",
    "LifespanEvent",
    "RecordingLifespanLogger",
    "FakeClock",
    "FakeResponseSink",
]
