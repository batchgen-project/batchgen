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
        all_reduce_min_deltas: list[torch.Tensor] | None = None,
        all_gather_into_tensor_responses: list[torch.Tensor] | None = None,
        broadcast_object_responses: list[list[Any]] | None = None,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.calls: list[CollectiveCall] = []
        self._agather_obj_resp = list(all_gather_object_responses or [])
        self._arm_deltas = list(all_reduce_max_deltas or [])
        self._armin_deltas = list(all_reduce_min_deltas or [])
        self._agit_resp = list(all_gather_into_tensor_responses or [])
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

    def all_reduce_min(self, tensor: torch.Tensor) -> None:
        self._record("all_reduce_min", tuple(tensor.shape))
        if self._armin_deltas:
            delta = self._armin_deltas.pop(0)
            tensor.copy_(torch.minimum(tensor, delta))

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
        if self._agit_resp:
            full = self._agit_resp.pop(0)
            out.copy_(full)
        else:
            # Default: populate only the self-slice, leave other slices zero.
            stride = tensor.shape[0] if tensor.ndim > 0 else 1
            out[self.rank * stride : (self.rank + 1) * stride].copy_(tensor)

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
        """Grant `n` more pages to `uuid`. Matches main's semantic where
        extend is "allocate more for this uuid" — if the uuid had no
        prior allocation, this creates one. The KVCacheManager layer
        still treats extend as a post-allocate operation; the fake
        tracks either order."""
        self._record("extend_pages", uuid, n)
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


class FakeLegacyBackend:
    """Records every call routed through the LegacyInfraBackend surface.

    CPU-only, no torch/CUDA. Every method is a no-op (or returns a sensible
    default) and appends to `self.calls` so tests can assert on the exact
    sequence of infrastructure calls a handler made.

    Satisfies :class:`batchgen.worker.protocols.LegacyInfraBackend`.
    """

    def __init__(
        self,
        rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.calls: list[tuple[str, tuple, dict]] = []
        # Mutable state the fake owns, so tests can inspect and preset
        self._uuid_to_local: dict[str, int] = {}
        self._local_to_uuid: dict[int, str] = {}
        self._sequences_with_gpu_kv: set[str] = set()
        self._admission_messages: list[Any] = []

    def _record(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((name, args, kwargs))

    # --- model lifecycle ---
    def configure_prefill_model(self) -> Any:
        self._record("configure_prefill_model")
        return ("fake_prefill_model", None)

    def configure_decode_model(self, max_num_seq: int, comm: Any) -> Any:
        self._record("configure_decode_model", max_num_seq, comm)
        return ("fake_decode_model", None)

    def deep_free_model_memory(self) -> None:
        self._record("deep_free_model_memory")

    def init_nvshmem(self) -> None:
        self._record("init_nvshmem")

    def set_phase(self, phase: str) -> None:
        self._record("set_phase", phase)

    def destroy_gpu_paged_kv_cache(self) -> None:
        self._record("destroy_gpu_paged_kv_cache")

    # --- KV cache primitives ---
    def release_gpu_kv_pages(self, local_indices: list[int]) -> None:
        self._record("release_gpu_kv_pages", local_indices)

    def release_host_kv_pages_for_batch(self, uuids: list[str]) -> None:
        self._record("release_host_kv_pages_for_batch", uuids)

    def extend_gpu_kv_allocation(self, uuids: list[str]) -> bool:
        self._record("extend_gpu_kv_allocation", uuids)
        return True

    def allocate_gpu_kv_two_page_buffer(
        self, local_indices: list[int], load_from_host: bool
    ) -> bool:
        self._record(
            "allocate_gpu_kv_two_page_buffer",
            local_indices,
            load_from_host=load_from_host,
        )
        return True

    def flush_deferred_kv_to_host(self) -> None:
        self._record("flush_deferred_kv_to_host")

    def wait_pending_kv_append_tasks(self) -> int:
        self._record("wait_pending_kv_append_tasks")
        return 0

    def rebuild_page_table_for_batch(self, batch: list[int], gpu_manager: Any) -> None:
        self._record("rebuild_page_table_for_batch", batch)

    def finalize_async_load_minimal(self, *args: Any, **kwargs: Any) -> Any:
        self._record("finalize_async_load_minimal", *args, **kwargs)
        return ([], [])

    def check_host_kv_watermark_trigger(self) -> bool:
        self._record("check_host_kv_watermark_trigger")
        return False

    def get_effective_chunk_size(self) -> int:
        self._record("get_effective_chunk_size")
        return 4096

    def put_sequences_on_hold(self, uuids: list[str]) -> None:
        self._record("put_sequences_on_hold", uuids)

    # --- boundary Stage 1 passthroughs ---
    def set_num_tokens_per_rank(self, n: int) -> None:
        self._record("set_num_tokens_per_rank", n)

    def set_rank_token_counts(self, counts: Any) -> None:
        self._record("set_rank_token_counts", counts)

    def host_paged_kv_worker_view(self) -> Any:
        self._record("host_paged_kv_worker_view")
        return getattr(self, "_host_paged_kv_worker_view", None)

    def report_chunk_sizer_completion(self, decoded_length: int) -> None:
        self._record("report_chunk_sizer_completion", decoded_length)

    # --- decode Stage 2 passthroughs ---
    def bind_decode_context(
        self,
        *,
        batch: list[int],
        past_key_states: Any,
        past_value_states: Any,
        scale_dict: Any,
    ) -> tuple[Any, Any]:
        self._record(
            "bind_decode_context",
            batch=list(batch),
            past_key_states=past_key_states,
            past_value_states=past_value_states,
            scale_dict=scale_dict,
        )
        return (
            getattr(self, "_bind_gpu_manager", None),
            getattr(self, "_bind_worker_view", None),
        )

    def forward_decode_step(
        self,
        *,
        batch: list[int],
        new_tokens: Any,
        gpu_manager: Any,
        page_table_verified: bool,
        local_iteration: int,
    ) -> Any:
        self._record(
            "forward_decode_step",
            batch=list(batch),
            new_tokens=new_tokens,
            gpu_manager=gpu_manager,
            page_table_verified=page_table_verified,
            local_iteration=local_iteration,
        )
        # Default: echo the input tokens unchanged — tests that need
        # sampling side effects set `_forward_step_output` instead.
        out = getattr(self, "_forward_step_output", None)
        return out if out is not None else new_tokens

    def record_decoded_token(
        self, *, local_idx: int, decode_pos: int, token: Any
    ) -> None:
        self._record(
            "record_decoded_token",
            local_idx=local_idx, decode_pos=decode_pos, token=token,
        )

    def check_repeating_ngram_pattern(
        self, *, local_idx: int, decoded_length: int
    ) -> bool:
        self._record(
            "check_repeating_ngram_pattern",
            local_idx=local_idx, decoded_length=decoded_length,
        )
        return bool(getattr(self, "_ngram_pattern_result", False))

    # --- index / UUID mapping ---
    def local_indices_to_global_seq_ids(self, batch: list[int]) -> list[int]:
        self._record("local_indices_to_global_seq_ids", batch)
        return list(batch)

    def get_local_indices_for_uuids(self, uuids: list[str]) -> list[int]:
        self._record("get_local_indices_for_uuids", uuids)
        return [self._uuid_to_local[u] for u in uuids if u in self._uuid_to_local]

    def uuid_to_local_map(self) -> dict[str, int]:
        return self._uuid_to_local

    def local_to_uuid_map(self) -> dict[int, str]:
        return self._local_to_uuid

    def sequences_with_gpu_kv(self) -> set[str]:
        return self._sequences_with_gpu_kv

    # --- sampling / IO ---
    def select_tokens(self, logits: Any) -> Any:
        self._record("select_tokens")
        return None

    def should_stop_at_eos(self, token_id: int) -> bool:
        self._record("should_stop_at_eos", token_id)
        return False

    def rebuild_input_tokens(self, batch: list[int]) -> Any:
        self._record("rebuild_input_tokens", batch)
        return None

    def decode_tokens_to_string(self, tokens: Any) -> str:
        self._record("decode_tokens_to_string")
        return ""

    def report_completion(self, uuid: str, gathered_text: str | None) -> None:
        self._record("report_completion", uuid, gathered_text)

    def gather_completed_tokens(self, uuids: list[str]) -> dict[str, str]:
        self._record("gather_completed_tokens", uuids)
        return {u: "" for u in uuids}

    def submit_completed_to_incremental_writer(self, uuids: list[str]) -> None:
        self._record("submit_completed_to_incremental_writer", uuids)

    # --- admission / tokenization ---
    def poll_admission_queue_nowait(self) -> Any:
        self._record("poll_admission_queue_nowait")
        if not self._admission_messages:
            import queue as _queue
            raise _queue.Empty
        return self._admission_messages.pop(0)

    def admit_sequences_from_message(self, msg: dict) -> list[str]:
        self._record("admit_sequences_from_message", msg)
        return []

    def tokenize_admitted_sequences(self, uuids: list[str]) -> None:
        self._record("tokenize_admitted_sequences", uuids)

    def assign_admitted_sequences_to_ranks(self, uuids: list[str]) -> None:
        self._record("assign_admitted_sequences_to_ranks", uuids)

    def build_local_query_book_for_admitted(self, uuids: list[str]) -> None:
        self._record("build_local_query_book_for_admitted", uuids)

    def update_max_input_length(self, new_len: int) -> None:
        self._record("update_max_input_length", new_len)

    # --- sequence-batch helpers ---
    def is_sequence_completed(self, seq: Any) -> bool:
        self._record("is_sequence_completed")
        return bool(getattr(seq, "eos_reached", False))

    def update_batch_status(self, uuids: list[str], status: Any) -> None:
        self._record("update_batch_status", uuids, status)

    # --- lifecycle infrastructure ---
    def feed_watchdog(self) -> None:
        self._record("feed_watchdog")

    def enable_decode_watchdog(self) -> None:
        self._record("enable_decode_watchdog")

    def disable_decode_watchdog(self) -> None:
        self._record("disable_decode_watchdog")

    def feed_decode_watchdog(self) -> None:
        self._record("feed_decode_watchdog")

    # --- decode setup + continuous (F5/F6) ---
    def decode_setup_once(self, max_num_seq: int) -> None:
        self._record("decode_setup_once", max_num_seq)

    def decode_config_for_batch(self, uuids: list[str]) -> None:
        self._record("decode_config_for_batch", uuids)

    def decoding_continuous(self, uuids: list[str]) -> None:
        self._record("decoding_continuous", uuids)

    # --- prefill forward (F4) ---
    def prefill_forward(self, uuids: list[str]) -> Any:
        self._record("prefill_forward", uuids)
        return None

    def prefill_forward_prepacked(self, uuids: list[str]) -> Any:
        self._record("prefill_forward_prepacked", uuids)
        return None

    def enable_prepack(self) -> bool:
        return False

    # --- prefill sizing (Phase 2.7) ---
    def effective_chunk_size(self) -> int:
        self._record("effective_chunk_size")
        return getattr(self, "_chunk_size", 4096)

    def prefill_setup_done(self) -> bool:
        return getattr(self, "_prefill_setup_done", False)

    # --- prefill config (F3) ---
    def prefill_flush_and_reconfigure(self) -> None:
        self._record("prefill_flush_and_reconfigure")
        self._prefill_setup_done = True

    def prefill_prepare_reentry(self, uuids: list[str]) -> None:
        self._record("prefill_prepare_reentry", uuids)

    def prefill_allocate_host_kv(self, uuids: list[str]) -> None:
        self._record("prefill_allocate_host_kv", uuids)

    # --- distributed init ---
    def ensure_comms(self) -> None:
        self._record("ensure_comms")

    def init_gpu_kv_with_actual_size(self) -> None:
        self._record("init_gpu_kv_with_actual_size")


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
    "FakeLegacyBackend",
]
