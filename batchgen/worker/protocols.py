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

    def all_reduce_min(self, tensor: "torch.Tensor") -> None: ...

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
    """Minimal tokenizer surface used by BatchFormation and CompletionHandler.

    Honors the plural-EOS convention from `conventions.md`: `eos_token_ids` is
    a set so custom tokenizers can declare multiple stop tokens.
    """

    eos_token_ids: set[int]

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


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


# ---------------------------------------------------------------------------
# Legacy infrastructure surface (Phase 2 full refactor)
# ---------------------------------------------------------------------------


class LegacyInfraBackend(Protocol):
    """Infrastructure surface of the legacy `BatchGenWorker`.

    This Protocol is the boundary between the native `batchgen/worker/`
    handlers and the legacy `BatchGenWorker` instance. It exposes ONLY the
    CUDA/KV/parallel-manager primitives that handlers need — never the
    control-flow methods (`prefill`, `decoding_continuous`, `_page_boundary_fast`,
    `_config_*_for_batch`, `_poll_admissions`) which are being ported into the
    `worker/` package as part of Phase 2.

    The production adapter (`LegacyWorkerBackend` in
    `batchgen/worker/backends/legacy_adapter.py`) wraps a
    `BatchGenWorker` instance. The fake (`FakeLegacyBackend` in
    `tests/unit/worker/fakes.py`) records calls for CPU unit tests.

    This surface is DELIBERATELY explicit — no catch-all `__getattr__`.
    Any missing method surfaces as an `AttributeError` in the first L1 run
    after a phase, which is the signal to extend the surface (the plan's
    risk-mitigation strategy).
    """

    # --- rank / topology ---
    rank: int
    local_rank: int
    world_size: int

    # --- model lifecycle ---
    def configure_prefill_model(self) -> Any: ...
    """Delegates to `parallel_manager.configure_prefill()`.
    Returns (model, weight_copy_task).
    """

    def configure_decode_model(self, max_num_seq: int, comm: Any) -> Any: ...
    """Delegates to `parallel_manager.configure_decoding(...)`.
    Returns (model, weight_copy_task).
    """

    def deep_free_model_memory(self) -> None: ...
    def init_nvshmem(self) -> None: ...
    def set_phase(self, phase: str) -> None: ...
    def destroy_gpu_paged_kv_cache(self) -> None: ...

    # --- KV cache primitives ---
    def release_gpu_kv_pages(self, local_indices: list[int]) -> None: ...
    def release_host_kv_pages_for_batch(self, uuids: list[UUID]) -> None: ...
    def extend_gpu_kv_allocation(self, uuids: list[UUID]) -> bool: ...
    def allocate_gpu_kv_two_page_buffer(
        self, local_indices: list[int], load_from_host: bool
    ) -> bool: ...
    def flush_deferred_kv_to_host(self) -> None: ...
    def wait_pending_kv_append_tasks(self) -> int: ...
    def rebuild_page_table_for_batch(self, batch: list[int], gpu_manager: Any) -> None: ...
    def finalize_async_load_minimal(self, *args: Any, **kwargs: Any) -> Any: ...
    def check_host_kv_watermark_trigger(self) -> bool: ...
    def get_effective_chunk_size(self) -> int: ...
    def put_sequences_on_hold(self, uuids: list[UUID]) -> None: ...

    # --- index / UUID mapping (read-only access to legacy state) ---
    def local_indices_to_global_seq_ids(self, batch: list[int]) -> list[int]: ...
    def get_local_indices_for_uuids(self, uuids: list[UUID]) -> list[int]: ...
    def uuid_to_local_map(self) -> dict[UUID, int]: ...
    def local_to_uuid_map(self) -> dict[int, UUID]: ...
    def sequences_with_gpu_kv(self) -> set[UUID]: ...

    # --- sampling / IO ---
    def select_tokens(self, logits: "torch.Tensor") -> "torch.Tensor": ...
    def should_stop_at_eos(self, token_id: int) -> bool: ...
    def rebuild_input_tokens(self, batch: list[int]) -> "torch.Tensor": ...
    def decode_tokens_to_string(self, tokens: "torch.Tensor") -> str: ...
    def report_completion(self, uuid: UUID, gathered_text: str | None) -> None: ...
    def gather_completed_tokens(self, uuids: list[UUID]) -> dict[UUID, str]: ...
    def submit_completed_to_incremental_writer(self, uuids: list[UUID]) -> None: ...

    # --- admission / tokenization (message parsing + query_book build) ---
    def poll_admission_queue_nowait(self) -> Any: ...
    """Returns a raw message from the admission queue or raises queue.Empty.
    Only called on rank 0."""

    def admit_sequences_from_message(self, msg: dict) -> list[UUID]: ...
    """Legacy `_admit_sequences_from_message`: parses message, creates
    SequenceEntry objects, adds to global_batch, returns new uuids."""

    def tokenize_admitted_sequences(self, uuids: list[UUID]) -> None: ...
    """Legacy `_tokenize_admitted_sequences`: tokenizes the listed uuids,
    sets prompt_length, populates buffer pool slots, updates
    max_input_length + padding_length via
    `_update_config_after_tokenization`."""

    def assign_admitted_sequences_to_ranks(self, uuids: list[UUID]) -> None: ...
    """Legacy `_assign_admitted_sequences_to_ranks`: round-robin rank
    assignment continuing from existing batch state."""

    def build_local_query_book_for_admitted(self, uuids: list[UUID]) -> None: ...
    """Legacy `_build_local_query_book_for_admitted`: constructs
    `query_book[local_idx]` entries for this rank's admitted uuids."""

    def update_max_input_length(self, new_len: int) -> None: ...
    """Updates ``self.max_input_length`` on the worker and propagates
    to ``engine_config.Basic_Config.padding_length`` via
    ``_update_config_after_tokenization``. No-op if new_len <= current."""

    # --- sequence-batch helpers ---
    def is_sequence_completed(self, seq: Any) -> bool: ...
    def update_batch_status(self, uuids: list[UUID], status: Any) -> None: ...

    # --- lifecycle infrastructure ---
    def feed_watchdog(self) -> None: ...
    def enable_decode_watchdog(self) -> None: ...
    def disable_decode_watchdog(self) -> None: ...
    def feed_decode_watchdog(self) -> None: ...

    # --- prefill forward (F4: native PrefillScheduler.run) ---
    def prefill_forward(self, uuids: list[UUID]) -> Any: ...
    """Legacy ``BatchGenWorker.prefill`` — the standard multi-sequence
    padded prefill forward pass. Takes UUIDs (the adapter maps to the
    worker's local indices internally). Returns any value the worker
    returns (currently ignored by the scheduler)."""

    def prefill_forward_prepacked(self, uuids: list[UUID]) -> Any: ...
    """Legacy ``BatchGenWorker.prefill_prepacked`` — the prepacked
    variant required for GPT-OSS and other models whose plain
    ``prefill()`` path doesn't forward ``position_ids`` through,
    producing ``rope_cos`` shape mismatches for batch > 1."""

    def enable_prepack(self) -> bool: ...
    """True when the worker has the prepacked prefill path wired
    (``worker.enable_prepack`` + ``worker.prefill_prepacked``). The
    scheduler uses this to decide which forward function to call."""

    # --- prefill config (F3: native PrefillScheduler.config_for_batch) ---
    def prefill_flush_and_reconfigure(self) -> None: ...
    """Legacy `_prefill_flush_and_reconfigure`: flush pending KV append
    tasks, deep-free decode model memory, destroy GPU paged KV cache,
    configure the model for prefill (``parallel_manager.configure_prefill``),
    set phase='prefill', restart h2d worker with the prefill weight-copy
    queue. Called once per prefill round before the forward pass."""

    def prefill_prepare_reentry(self, uuids: list[UUID]) -> None: ...
    """Legacy `_prefill_prepare_reentry`: for every EVICTED uuid, rebuild
    scalar re-entry state (decoded_length, baseline, max_decode_length,
    eos flags) on all ranks, and on the owning rank rebuild input_ids +
    decoded_tokens buffer views + query_book entry. Fresh QUEUEING uuids
    are skipped."""

    def prefill_allocate_host_kv(self, uuids: list[UUID]) -> None: ...
    """Legacy `_prefill_allocate_host_kv`: register rank-owned uuids in
    ``_uuid_to_local_map`` / ``_local_to_uuid_map`` (reusing freed
    indices), then compute per-sequence initial host capacity and
    invoke ``host_paged_kv_worker_view.register_sequences`` +
    ``allocate_pages_for_sequences``."""

    # --- decode setup + continuous (F5/F6) ---
    def decode_setup_once(self, max_num_seq: int) -> None: ...
    """F5 native decode one-time setup (idempotent). Combines:
      1. `_generate_ensure_comms` — PyNccl init for MoE EP
      2. `_load_decode_model(max_num_seq, comm)` — decode model load
      3. `_init_gpu_kv_with_actual_size` — GPU KV manager init
    All ranks must call in lockstep; returns when all three steps
    completed at least once."""

    def decode_config_for_batch(self, uuids: list[UUID]) -> None: ...
    """F5 per-batch decode config (called before every decode cycle).
    Runs `_decode_config_repair_ctx_lengths(uuids)` +
    `_decode_config_allocate_gpu_kv(local_batch)` via the adapter.
    Skips the all_gather-based validate step (hybrid ranks reach this
    point at different times, causing deadlock)."""

    def decoding_continuous(self, uuids: list[UUID]) -> None: ...
    """F6 native decoding_continuous cycle. Builds the initial
    `new_tokens` tensor via `_rebuild_input_tokens`, then invokes
    legacy `BatchGenWorker.decoding_continuous(new_tokens, uuids,
    local_batch)` which handles the inner loop, page boundaries,
    sampling, completion detection, and state mutation."""

    # --- distributed init (PyNccl for MoE EP) ---
    def ensure_comms(self) -> None: ...
    """Legacy `_generate_ensure_comms`: verifies dist, coordinates PyNccl
    init so `self.comm` is set on all ranks. Idempotent."""

    def init_gpu_kv_with_actual_size(self) -> None: ...
    """Legacy `_init_gpu_kv_with_actual_size`: first-time GPU KV manager
    init after model is loaded (sizes based on actual free HBM).
    Idempotent — subsequent calls return immediately."""


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
    "LegacyInfraBackend",
]
