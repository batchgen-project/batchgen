"""WorkerConfig — every `BATCHGEN_*` knob, read once.

Plan Decision #3: handlers never touch ``os.environ``. The orchestrator
reads every knob exactly once at startup via :meth:`WorkerConfig.from_env`
and passes the resolved values into handler constructors as explicit
arguments. Tests construct :class:`WorkerConfig` directly with
deterministic values — they never rely on ambient environment state.

Every field is documented with:
  - the environment variable name (``BATCHGEN_*``) in main today
  - the production default
  - which handler reads it

Adding a new knob is additive: (1) add a field here, (2) add a line to
:meth:`from_env`, (3) update the handler constructor that consumes it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _bool_env(name: str, default: bool) -> bool:
    """Parse a ``BATCHGEN_*`` env var as a boolean.

    Truthy: ``"1"``, ``"true"``, ``"yes"``, ``"on"`` (case-insensitive).
    Falsy: ``"0"``, ``"false"``, ``"no"``, ``"off"``, empty string.
    Any other value falls back to `default` — the handler stays robust
    against typos rather than crashing at startup.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off", ""):
        return False
    return default


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class WorkerConfig:
    """Frozen configuration snapshot passed through handler constructors."""

    # -- boundary / decode pacing (main `SequenceEntry` + env defaults) ----
    decision_frequency_pages: int = 2
    """How many pages a decode sequence advances before a page boundary
    fires. Env: (none — currently a SequenceEntry constant in main).
    Handler: BoundaryPlanner, DecodeScheduler, KVCacheManager."""

    initial_gpu_page_buffer: int = 32
    """First GPU KV reservation per sequence (2048 tokens = 32 pages).
    Env: (constant in main). Handler: KVCacheManager, DecodeScheduler."""

    extension_gpu_page_buffer: int = 4
    """Additional GPU pages granted per boundary extension. Env:
    (constant in main). Handler: KVCacheManager, BoundaryPlanner."""

    # -- host KV watermarks ------------------------------------------------
    prefill_watermark_pct: int = 70
    """Free host-KV % above which the decode phase yields to prefill.
    Env: ``BATCHGEN_HOST_KV_WATERMARK``. Handler: KVCacheManager,
    BoundaryPlanner. Main default 70."""

    eviction_watermark_pct: int = 10
    """Free host-KV % below which eviction fires. Env: (no dedicated
    env var in main; exposed via BatchGenWorkerArgs). Handler:
    KVCacheManager. Main default 10."""

    enable_host_kv_eviction: bool = True
    """Host-KV eviction master switch. Env:
    ``BATCHGEN_ENABLE_HOST_KV_EVICTION``. Main default ``True`` (see
    batchgen_worker.py:379). Handler: BoundaryPlanner (Stage 1d
    Rule 3). Disabling it bypasses the HostEvict decision path."""

    host_kv_eviction_watermark_pct: int = 20
    """Free host-KV % below which Stage 1d plan_full emits HostEvict
    decisions. Env: ``BATCHGEN_HOST_KV_EVICTION_WATERMARK`` (same
    env var as ``eviction_watermark_pct`` for compatibility; the two
    knobs are kept separate so future code can disentangle them).
    Handler: BoundaryHandler (via BoundaryHandlerConfig)."""

    host_kv_total_pages: int = 10000
    """Total host KV page count used for watermark % math. Env:
    (derived from ``BATCHGEN_GPU_KV_CACHE_SIZE_GB`` + host KV args).
    Handler: KVCacheManager, BoundaryPlanner."""

    # -- feature flags (plan Decision #3 ablation knobs) -------------------
    rep_detection_enabled: bool = True
    """Enable N-gram repetition detection. Env: ``BATCHGEN_REP_DETECTION``.
    Handler: CompletionHandler."""

    preemption_enabled: bool = True
    """Enable watermark-trigger decode preemption. Env:
    ``BATCHGEN_ENABLE_DECODE_PREEMPTION``. Handler: HostKVRebalancer,
    DecodeScheduler."""

    ignore_eos: bool = False
    """Force decoding past EOS. Env: (per-request in main; static knob
    here). Handler: CompletionHandler."""

    # -- sizes -------------------------------------------------------------
    model_context_length: int = 4096
    """Model's max context window in tokens. Env: (derived from model
    config). Handler: BatchFormation (overflow rejection),
    CompletionHandler (context-limit completion)."""

    # -- pool mode ---------------------------------------------------------
    max_pool_size: int = 0
    """Pool-mode sequence capacity. 0 = legacy batch-FIFO mode; > 0
    enables `generate_persistent`. Env: (BatchGenWorkerArgs). Handler:
    AdmissionCoordinator, orchestrator."""

    # -- diagnostics (kept as env-gated knobs, default off) ----------------
    decode_assert: bool = False
    """``BATCHGEN_DECODE_ASSERT`` — extra runtime checks in the decode
    loop. Default off."""

    multi_batch_diag: bool = False
    """``BATCHGEN_MULTI_BATCH_DIAG`` — per-batch diagnostic logging."""

    decode_timing: bool = False
    """``BATCHGEN_DECODE_TIMING`` — per-section decode timing dump."""

    critical_diags: bool = False
    """``BATCHGEN_ENABLE_CRITICAL_DIAGS`` — host KV migration critical
    diagnostics."""

    cb_log: str = ""
    """``BATCHGEN_CB_LOG`` — boundary scheduling debug log label."""

    # -- raw env snapshot (for traceability only; never reshaped) ---------
    raw_env: dict[str, str] = field(default_factory=dict)
    """Frozen copy of the ``BATCHGEN_*`` subset of os.environ at
    from_env() time. Handlers never read this; it exists so traces can
    record exactly which env the orchestrator was started with."""

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        *,
        host_kv_total_pages: int,
        model_context_length: int,
        max_pool_size: int = 0,
    ) -> "WorkerConfig":
        """Read every ``BATCHGEN_*`` knob from `os.environ` exactly once.

        Fields that are not plumbed through env vars in main
        (``host_kv_total_pages``, ``model_context_length``, ``max_pool_size``)
        are passed in explicitly by the orchestrator because they come
        from :class:`BatchGenWorkerArgs` / model config, not env.
        """
        raw = {k: v for k, v in os.environ.items() if k.startswith("BATCHGEN_")}
        return cls(
            decision_frequency_pages=_int_env("BATCHGEN_DECISION_FREQUENCY_PAGES", 2),
            initial_gpu_page_buffer=_int_env("BATCHGEN_INITIAL_GPU_PAGE_BUFFER", 32),
            extension_gpu_page_buffer=_int_env("BATCHGEN_EXTENSION_GPU_PAGE_BUFFER", 4),
            prefill_watermark_pct=_int_env("BATCHGEN_HOST_KV_WATERMARK", 70),
            eviction_watermark_pct=_int_env("BATCHGEN_HOST_KV_EVICTION_WATERMARK", 10),
            enable_host_kv_eviction=_bool_env("BATCHGEN_ENABLE_HOST_KV_EVICTION", True),
            host_kv_eviction_watermark_pct=_int_env("BATCHGEN_HOST_KV_EVICTION_WATERMARK", 20),
            host_kv_total_pages=host_kv_total_pages,
            rep_detection_enabled=_bool_env("BATCHGEN_REP_DETECTION", True),
            preemption_enabled=_bool_env("BATCHGEN_ENABLE_DECODE_PREEMPTION", True),
            ignore_eos=False,  # per-request in main; static default here
            model_context_length=model_context_length,
            max_pool_size=max_pool_size,
            decode_assert=_bool_env("BATCHGEN_DECODE_ASSERT", False),
            multi_batch_diag=_bool_env("BATCHGEN_MULTI_BATCH_DIAG", False),
            decode_timing=_bool_env("BATCHGEN_DECODE_TIMING", False),
            critical_diags=_bool_env("BATCHGEN_ENABLE_CRITICAL_DIAGS", False),
            cb_log=os.environ.get("BATCHGEN_CB_LOG", ""),
            raw_env=raw,
        )


__all__ = ["WorkerConfig"]
