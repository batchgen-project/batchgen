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


def _matches_model_name(model_name: str | None, patterns: tuple[str, ...]) -> bool:
    normalized = (model_name or "").strip().lower()
    return any(pattern in normalized for pattern in patterns)


def is_kimi_k25_backend_model(model_name: str | None) -> bool:
    return _matches_model_name(model_name, KIMI_K25_BACKEND_NAME_PATTERNS)


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
    return _matches_model_name(model_name, GLM5_BACKEND_NAME_PATTERNS)


DEEPSEEK_PREFIX_REUSE_NAME_PATTERNS = (
    "deepseek-ai/deepseek-r1",
    "deepseek/deepseek-r1",
    "deepseek-r1",
    "deepseek-ai/deepseek-v3",
    "deepseek/deepseek-v3",
    "deepseek-v3",
    "deepseek-ai/deepseek-v4-flash",
    "deepseek/deepseek-v4-flash",
    "deepseek-v4-flash",
    "deepseek-ai/deepseek-v4-pro",
    "deepseek/deepseek-v4-pro",
    "deepseek-v4-pro",
)


def is_deepseek_prefix_reuse_model(model_name: str | None) -> bool:
    return _matches_model_name(model_name, DEEPSEEK_PREFIX_REUSE_NAME_PATTERNS)


MINIMAX_M25_BACKEND_NAME_PATTERNS = (
    "minimaxai/minimax-m2.5",
    "minimax-m2.5",
    "minimax_m2.5",
    "minimax-m25",
    "minimax_m25",
)


def is_minimax_m25_backend_model(model_name: str | None) -> bool:
    return _matches_model_name(model_name, MINIMAX_M25_BACKEND_NAME_PATTERNS)


PREFIX_REUSE_GPT_OSS_PATTERNS = (
    "gpt-oss",
)


def is_prefix_reuse_supported_model(model_name: str | None) -> bool:
    """Return whether the model has a prefix-cache-aware prefill path."""
    return (
        _matches_model_name(model_name, PREFIX_REUSE_GPT_OSS_PATTERNS)
        or is_glm5_backend_model(model_name)
        or is_deepseek_prefix_reuse_model(model_name)
        or is_kimi_k25_backend_model(model_name)
        or is_minimax_m25_backend_model(model_name)
    )
