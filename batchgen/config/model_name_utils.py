"""Shared user-facing model-name helpers."""

from __future__ import annotations

KIMI_K25_BACKEND_MODEL_IDS = (
    "moonshotai/Kimi-K2.5",
    "moonshotai/Kimi-K2.6",
)

KIMI_K25_BACKEND_NAME_PATTERNS = (
    "moonshotai/kimi-k2.5",
    "moonshotai/kimi-k2.6",
    "kimi-k2.5",
    "kimi_k2.5",
    "kimi-k25",
    "kimi_k25",
    "kimi-k2.6",
    "kimi_k2.6",
    "kimi-k26",
    "kimi_k26",
)


def is_kimi_k25_backend_model(model_name: str | None) -> bool:
    normalized = (model_name or "").strip().lower()
    return any(pattern in normalized for pattern in KIMI_K25_BACKEND_NAME_PATTERNS)


# The served models whose workload-independent one-time work — model build,
# resident decode shards, streamed buffers, the weight schedule and the
# model's own runtime extensions — runs BEFORE the worker signals readiness
# (``BatchGenWorker.prepare_kimi_k3_startup``). Matched EXACTLY: the sibling
# Kimi-Linear / K2.5 checkpoints share the architecture but not this
# lifecycle, and an unlisted Kimi-K3.x must not inherit it by substring.
PRE_READINESS_STARTUP_MODEL_IDS = frozenset({
    "moonshotai/Kimi-K3",
    "moonshotai/Kimi-K3.1-Pro",
})


def runs_startup_before_readiness(model_name: str | None) -> bool:
    """Whether this exact served model id takes the pre-readiness startup."""
    return model_name in PRE_READINESS_STARTUP_MODEL_IDS
