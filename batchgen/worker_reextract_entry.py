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
    """DEPRECATED — Phase-2 full refactor retires the env-var gate.

    The orchestrator is the only code path; this function always
    returns True and is kept only so external callers that still
    import it don't break. It will be removed once no caller remains.
    """
    return True


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

    # -- model executor (F4: prefill forward is native via adapter) ----
    # TorchModelExecutorBackend stays only as a no-op shim for protocol
    # satisfaction — the scheduler never invokes it in production when
    # `legacy_infra` is set (the adapter's prefill_forward* methods are
    # called directly).
    def _noop(batch: dict[str, Any]) -> Any:
        return None

    model = TorchModelExecutorBackend(prefill_fn=_noop, decode_fn=_noop)

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

    # -- Phase 2 full-refactor infrastructure adapter ----------------
    # Wraps the BatchGenWorker instance, exposing the infrastructure surface
    # (CUDA/KV/parallel_manager) that native handlers ported from
    # batchgen_worker.py in Phases F2-F6 call. All control-flow now lives
    # in batchgen/worker/; the adapter is the ONLY coupling point.
    from batchgen.worker.backends.legacy_adapter import LegacyWorkerBackend
    legacy_infra = LegacyWorkerBackend(worker)

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
        legacy_infra=legacy_infra,
    )


__all__ = ["should_use_reextract", "build_orchestrator"]
