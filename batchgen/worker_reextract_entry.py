"""Feature-flagged orchestrator entry for the legacy BatchGenWorker.

Provides two functions the monolithic worker calls from its
``generate()`` / ``generate_persistent()`` entry points:

  - :func:`should_use_reextract` reads ``BATCHGEN_USE_REEXTRACT`` from
    ``os.environ`` and returns ``True`` only when the flag is set to
    ``"1"``. Default is off — production runs are unchanged.

  - :func:`build_orchestrator` constructs a
    :class:`batchgen.worker.orchestrator.WorkerOrchestrator` by
    wrapping the legacy worker's fields in production backends and
    a :class:`batchgen.worker.config.WorkerConfig`.

The orchestrator built here shares the legacy worker's
``state.global_batch`` (by passing in the same ``WorkerState``
instance), which the orchestrator's handlers mutate in place. On
return the legacy worker observes the exact same sequence state it
would have produced on its own.

**This module is the only coupling point** between the monolithic
worker and the re-extracted package. The legacy worker imports this
one module from exactly two ``generate()`` / ``generate_persistent()``
entry sites; everything else (every handler, every backend adapter,
every test) stays strictly inside ``batchgen/worker/``.
"""

from __future__ import annotations

import os
from typing import Any

from batchgen.worker.backends.torch_collectives import TorchCollectiveBackend
from batchgen.worker.backends.torch_gpu_kv import TorchGpuKvBackend
from batchgen.worker.backends.torch_host_kv import TorchHostKvBackend
from batchgen.worker.backends.torch_lifespan import TorchLifespanLogger
from batchgen.worker.backends.torch_model_executor import TorchModelExecutorBackend
from batchgen.worker.backends.torch_response_sink import TorchResponseSink
from batchgen.worker.backends.torch_tokenizer import TorchTokenizerBackend
from batchgen.worker.config import WorkerConfig
from batchgen.worker.orchestrator import WorkerOrchestrator
from batchgen.worker.state import WorkerState


def should_use_reextract() -> bool:
    """True when ``BATCHGEN_USE_REEXTRACT=1`` is set in ``os.environ``.

    Default off: production runs behave identically to the untouched
    monolithic worker until POIS flips the flag for a smoke test.
    """
    return os.environ.get("BATCHGEN_USE_REEXTRACT", "0") == "1"


def _find(worker: Any, *names: str) -> Any:
    """Return the first attribute of `worker` whose name matches any of
    `names`, or ``None`` if none exist. Used to bridge field-name
    differences between versions of the legacy worker."""
    for name in names:
        if hasattr(worker, name):
            return getattr(worker, name)
    return None


def _derive_state(worker: Any) -> WorkerState:
    """Construct a fresh ``WorkerState`` from the legacy worker's fields.

    Rather than sharing a WorkerState instance across both code paths
    (risky — the legacy worker has no such field), we build a shadow
    that points at the same ``global_batch`` and ``index maps`` the
    legacy worker already owns. Mutations via orchestrator handlers
    land on the legacy worker's own fields because the mappings are
    passed by reference.
    """
    import torch

    device = getattr(worker, "device", 0)
    local_rank = getattr(worker, "local_rank", 0)
    rank = getattr(worker, "rank", getattr(worker, "global_rank", 0))
    world_size = getattr(worker, "world_size", 1)
    torch_device = getattr(worker, "torch_device", None) or torch.device(
        f"cuda:{device}"
    )

    state = WorkerState(
        rank=int(rank),
        local_rank=int(local_rank),
        world_size=int(world_size),
        device=int(device),
        torch_device=torch_device,
    )
    # Share the legacy worker's live containers by aliasing. The
    # WorkerState dataclass default_factory gave us fresh dicts/sets
    # but we immediately replace them with the legacy fields so both
    # paths observe the same mutation state.
    state.global_batch = worker.global_batch
    if hasattr(worker, "_local_to_uuid_map"):
        state.local_to_uuid_map = worker._local_to_uuid_map
    if hasattr(worker, "_uuid_to_local_map"):
        state.uuid_to_local_map = worker._uuid_to_local_map
    if hasattr(worker, "_free_local_indices"):
        state.free_local_indices = worker._free_local_indices
    if hasattr(worker, "_next_local_idx"):
        state.next_local_idx = int(worker._next_local_idx)
    return state


def _derive_config(worker: Any) -> WorkerConfig:
    """Construct a ``WorkerConfig`` from the legacy worker's attributes.

    Reads explicit fields the legacy worker exposes and falls back to
    :meth:`WorkerConfig.from_env` for env-driven knobs.
    """
    host_total = (
        _find(worker, "host_kv_total_pages", "_host_kv_total_pages")
        or _find(worker, "host_kv_num_pages")
        or 10000
    )
    model_context_length = _find(
        worker, "model_context_length", "_model_context_length"
    ) or 4096
    max_pool_size = _find(worker, "_max_pool_size", "max_pool_size") or 0
    return WorkerConfig.from_env(
        host_kv_total_pages=int(host_total),
        model_context_length=int(model_context_length),
        max_pool_size=int(max_pool_size),
    )


def build_orchestrator(worker: Any) -> WorkerOrchestrator:
    """Construct a :class:`WorkerOrchestrator` from a legacy
    ``BatchGenWorker`` instance.

    Every backend is a thin adapter around fields the legacy worker
    already initialized (process group, KV managers, tokenizer,
    response queue). Nothing here re-initializes CUDA, touches the
    NCCL store, or loads the model — it all reuses what's already
    there.
    """
    state = _derive_state(worker)
    config = _derive_config(worker)

    # -- collectives ---------------------------------------------------
    process_group = _find(
        worker, "process_group", "_process_group", "model_parallel_group"
    )
    collectives = TorchCollectiveBackend(
        process_group=process_group,
        rank=state.rank,
        world_size=state.world_size,
    )

    # -- GPU / host KV backends ---------------------------------------
    gpu_manager = _find(
        worker,
        "gpu_paged_kv_cache_manager",
        "gpu_kv_manager",
        "_gpu_paged_kv_cache_manager",
    )
    host_worker_view = _find(
        worker,
        "host_paged_kv_worker_view",
        "host_kv_view",
        "_host_paged_kv_worker_view",
    )
    gpu_kv = TorchGpuKvBackend(gpu_manager, state)
    host_kv = TorchHostKvBackend(
        host_worker_view, state, total_pages=config.host_kv_total_pages
    )

    # -- tokenizer -----------------------------------------------------
    tokenizer = TorchTokenizerBackend(worker.tokenizer)

    # -- model executor ------------------------------------------------
    def _uuids_to_local_indices(uuids: list[str]) -> list[int]:
        """Translate UUIDs to legacy worker local indices.

        The legacy ``prefill`` / ``decoding_continuous`` methods take
        ``batch: list[int]`` where the ints are local indices into
        ``query_book``. The orchestrator speaks UUIDs everywhere, so
        every call into the monolithic methods translates via the
        legacy worker's own ``_uuid_to_local_map``.
        """
        out: list[int] = []
        for uuid in uuids:
            local_idx = worker._uuid_to_local_map.get(uuid)
            if local_idx is not None:
                out.append(local_idx)
        return out

    def prefill_config_delegate(uuids: list[str]) -> None:
        """Delegate prefill configuration to legacy worker.

        Calls _config_prefill_for_batch which handles:
        - Flush pending KV + deep free decode model memory
        - Reconfigure model for prefill (configure_prefill)
        - Prepare evicted sequences for re-entry
        - Allocate host KV pages
        """
        config_fn = getattr(worker, "_config_prefill_for_batch", None)
        if config_fn is not None:
            config_fn(uuids)

    def prefill_fn(batch: dict[str, Any]) -> Any:
        uuids = batch.get("uuids", [])
        local_batch = _uuids_to_local_indices(uuids)
        if not local_batch:
            return None
        use_prepacked = batch.get("prepacked", False) and hasattr(
            worker, "prefill_prepacked"
        )
        if use_prepacked:
            return worker.prefill_prepacked(local_batch)
        if hasattr(worker, "prefill"):
            return worker.prefill(local_batch)
        return None

    def decode_fn(batch: dict[str, Any]) -> Any:
        # Unused in the hybrid path — DecodeScheduler bypasses
        # forward_decode when decode_delegate is set. Kept as a no-op
        # so any stray call does not raise.
        return None

    model = TorchModelExecutorBackend(prefill_fn=prefill_fn, decode_fn=decode_fn)

    # -- decode setup delegate (hybrid production path) ---------------
    _decode_model_loaded = [False]  # mutable closure state

    def decode_setup_delegate(uuids: list[str]) -> None:
        """Lazy-load decode model + init GPU KV + config decode batch.

        Called once before the first decode cycle. Subsequent calls only
        run _config_decoding_for_batch (model stays loaded).
        """
        if not _decode_model_loaded[0]:
            # Comms setup (PyNccl)
            ensure_comms = getattr(worker, "_generate_ensure_comms", None)
            if ensure_comms is not None:
                ensure_comms()

            # Load decode model
            max_num_seq = len(worker.global_batch) if worker.global_batch else 1
            load_fn = getattr(worker, "_load_decode_model", None)
            if load_fn is not None:
                load_fn(max_num_seq, getattr(worker, "comm", None))

            # Init GPU KV with actual size
            init_kv = getattr(worker, "_init_gpu_kv_with_actual_size", None)
            if init_kv is not None:
                init_kv()

            _decode_model_loaded[0] = True

        # Config decode batch (GPU KV allocation for these sequences)
        local_batch = _uuids_to_local_indices(uuids)
        config_fn = getattr(worker, "_config_decoding_for_batch", None)
        if config_fn is not None and local_batch:
            config_fn(uuids, local_batch)

    # -- decode delegate (hybrid production path) --------------------
    def decode_delegate(uuids: list[str]) -> None:
        """Run one full decode cycle via legacy
        ``BatchGenWorker.decoding_continuous``.

        The legacy method takes ``(new_tokens, decode_uuids, batch)``
        where ``batch`` is a list of local indices. We translate the
        orchestrator's uuid list on every call and let the legacy
        method handle the inner loop, page boundaries, sampling,
        completion detection, and state mutation on ``global_batch``.
        """
        local_batch = _uuids_to_local_indices(uuids)
        if not local_batch:
            return

        # Build initial new_tokens tensor via the legacy helper which
        # pulls the last token of each sequence from its decoded_tokens
        # buffer. ``_rebuild_input_tokens`` returns a (batch_size, 1)
        # tensor on the worker's torch_device.
        rebuild = getattr(worker, "_rebuild_input_tokens", None)
        if rebuild is None:
            # Fallback: empty zero tensor — legacy method will
            # recompute via its own _rebuild_input_tokens path after
            # the first page boundary.
            import torch

            new_tokens = torch.zeros(
                (len(local_batch), 1), dtype=torch.int64, device=worker.torch_device
            )
        else:
            new_tokens = rebuild(local_batch)

        # Call the legacy all-in-one decode loop. Its return value
        # (updated decode_uuids, batch) is currently ignored — the
        # orchestrator's outer run_batch loop re-reads
        # state.global_batch after each decode phase to determine
        # what remains.
        worker.decoding_continuous(new_tokens, list(uuids), list(local_batch))

    # -- lifespan + response sink -------------------------------------
    lifespan = TorchLifespanLogger(rank=state.rank)
    sink = TorchResponseSink(
        response_queue=_find(worker, "_response_queue", "response_queue"),
        incremental_writer=_find(worker, "_incremental_writer", "incremental_writer"),
    )

    # -- clock ---------------------------------------------------------
    class _WallClock:
        def now(self) -> float:
            import time

            return time.monotonic()

    # -- admission queue ----------------------------------------------
    admission_queue = _find(worker, "_admission_queue", "admission_queue")

    # -- admission delegate (hybrid production path) -----------------
    def admission_delegate() -> bool:
        """Delegate the entire admission cycle to legacy ``_poll_admissions``.

        Legacy ``_poll_admissions`` handles: polling the queue on rank 0,
        broadcasting the message to all ranks, converting message
        ``entries`` to ``SequenceEntry``, tokenizing via
        ``_tokenize_admitted_sequences``, assigning ranks, and
        **critically** building the legacy ``query_book`` dict that
        ``worker.prefill`` / ``worker.decoding_continuous`` consume.

        Returns the bool result from legacy, which the
        AdmissionCoordinator normalizes to an empty uuid list — the
        orchestrator's run_batch rediscovers admitted sequences by
        re-reading ``state.global_batch`` afterwards.
        """
        poll = getattr(worker, "_poll_admissions", None)
        if poll is None:
            return False
        return bool(poll())

    return WorkerOrchestrator(
        state,
        config,
        collectives=collectives,
        gpu_kv=gpu_kv,
        host_kv=host_kv,
        tokenizer=tokenizer,
        model=model,
        lifespan=lifespan,
        sink=sink,
        clock=_WallClock(),
        admission_queue=admission_queue,
        decode_delegate=decode_delegate,
        decode_setup_delegate=decode_setup_delegate,
        admission_delegate=admission_delegate,
        prefill_config_delegate=prefill_config_delegate,
    )


__all__ = ["should_use_reextract", "build_orchestrator"]
