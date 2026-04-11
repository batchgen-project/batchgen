"""Dependency Protocols for the worker handler package.

Every external subsystem (torch.distributed collectives, GPU/host KV managers,
tokenizer, model forward pass, lifespan logging, wall clock, response sink) is
accessed through a Protocol defined here. Handlers depend only on these
interfaces, never on concrete implementations.

  - Production code wires `TorchCollectiveBackend`, `TorchGpuKvBackend`, etc.
    (thin wrappers that satisfy the Protocols).
  - Tests wire `FakeCollectiveBackend`, `FakeGpuKvBackend`, etc. from
    `tests/unit/worker/fakes.py` — pure Python, CPU-only, records every call.

Protocols are deliberately NOT `@runtime_checkable`. Static type checkers and
hand-rolled fakes are enough; runtime `isinstance` adds no value and hides
structural mismatches until first call.

Type placeholders (`UUID = str`, `PageId = int`, `AsyncHandle = Any`) keep the
module import-light and defer richer typing to the slice that needs it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import torch


# ---------------------------------------------------------------------------
# Type aliases used by the Protocols below. Kept simple so handlers do not
# transitively import torch or batchgen.sequence just to type-check.
# ---------------------------------------------------------------------------

UUID = str
PageId = int
AsyncHandle = Any


# ---------------------------------------------------------------------------
# Collective communication
# ---------------------------------------------------------------------------


class CollectiveBackend(Protocol):
    """Thin wrapper around `torch.distributed` primitives.

    Every cross-rank operation in the worker handlers goes through a method
    here. The production impl wraps a `ProcessGroup`; the fake records calls.
    """

    rank: int
    world_size: int

    def all_reduce_max(self, tensor: "torch.Tensor") -> None: ...

    def all_reduce_sum(self, tensor: "torch.Tensor") -> None: ...

    def all_gather_tensor(
        self, tensor_list: list["torch.Tensor"], tensor: "torch.Tensor"
    ) -> None: ...

    def all_gather_into_tensor(
        self, out: "torch.Tensor", tensor: "torch.Tensor"
    ) -> None: ...

    def all_gather_object(self, obj_list: list[Any], obj: Any) -> None: ...

    def broadcast_tensor(self, tensor: "torch.Tensor", src: int) -> None: ...

    def broadcast_object(self, obj_list: list[Any], src: int) -> None: ...

    def barrier(self) -> None: ...


# ---------------------------------------------------------------------------
# KV cache backends (GPU + host)
# ---------------------------------------------------------------------------


class GpuKvBackend(Protocol):
    """GPU paged KV cache primitive operations.

    Wraps the production `GpuKvManager` without exposing its CUDA-specific
    internals. Handlers call these methods; fakes record them.
    """

    def allocate_pages(self, uuid: UUID, n: int) -> list[PageId]: ...

    def release_pages(self, uuid: UUID) -> None: ...

    def extend_pages(self, uuid: UUID, n: int) -> list[PageId]: ...

    def append_kv(self, uuid: UUID, layer: int, kv: "torch.Tensor") -> None: ...

    def free_pages(self) -> int: ...

    def rebuild_page_table(self, uuids: list[UUID]) -> None: ...


class HostKvBackend(Protocol):
    """Host-side KV page store plus async host→GPU transfers."""

    def allocate_pages(self, uuid: UUID, n: int) -> list[PageId]: ...

    def release_pages(self, uuid: UUID) -> None: ...

    def load_to_gpu_async(self, uuid: UUID, page_ids: list[PageId]) -> AsyncHandle: ...

    def free_pages(self) -> int: ...


# ---------------------------------------------------------------------------
# Tokenizer / model executor
# ---------------------------------------------------------------------------


class TokenizerBackend(Protocol):
    """Minimal tokenizer surface used by BatchFormation.

    Honors the plural-EOS convention from `conventions.md`: `eos_token_ids` is
    a set so custom tokenizers can declare multiple stop tokens.
    """

    eos_token_ids: set[int]

    def encode(self, text: str) -> list[int]: ...


class ModelExecutorBackend(Protocol):
    """Model forward-pass surface used by PrefillScheduler and DecodeScheduler.

    Arg and return types are deliberately `Any` at this layer — the concrete
    batch / output dataclasses are introduced when prefill and decode slices
    land and the exact shapes can be pinned without churn.
    """

    def forward_prefill(self, batch: Any) -> Any: ...

    def forward_decode(self, batch: Any) -> Any: ...


# ---------------------------------------------------------------------------
# Lifespan logging, clock, response sink
# ---------------------------------------------------------------------------


class LifespanLoggerBackend(Protocol):
    """Wraps `batchgen.lifespan.SeqEvent` logging.

    Every state transition in a handler emits an event via this Protocol so
    traces of a run can be replayed deterministically. The fake just appends
    to a list; the production impl forwards to `batchgen/lifespan.py`.
    """

    def log(self, seq: Any, event: Any, detail: dict[str, Any]) -> None: ...


class ClockBackend(Protocol):
    """Monotonic clock surface. Injected so tests are time-independent."""

    def now(self) -> float: ...


class ResponseSinkBackend(Protocol):
    """Where `CompletionHandler.report` sends finished sequences.

    In production this wraps the response mp.Queue plus the incremental
    writer. In tests, it records finished sequences for assertion.
    """

    def put(self, uuid: UUID, payload: dict[str, Any]) -> None: ...


__all__ = [
    "UUID",
    "PageId",
    "AsyncHandle",
    "CollectiveBackend",
    "GpuKvBackend",
    "HostKvBackend",
    "TokenizerBackend",
    "ModelExecutorBackend",
    "LifespanLoggerBackend",
    "ClockBackend",
    "ResponseSinkBackend",
]
