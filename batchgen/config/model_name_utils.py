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


GLM5_BACKEND_NAME_PATTERNS = (
    "zai-org/glm-5",
    "zai-org/glm-5.1",
    "glm-5",
    "glm_5",
    "glm-5.1",
    "glm_5.1",
    "glm5",
    "glm51",
)


def is_glm5_backend_model(model_name: str | None) -> bool:
    normalized = (model_name or "").strip().lower()
    return any(pattern in normalized for pattern in GLM5_BACKEND_NAME_PATTERNS)
