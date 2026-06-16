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


# DeepSeek-R1 shares the DeepseekV3 architecture (plain MLA + FP8 3D-blockwise MoE).
# Patterns deliberately match R1 and the v3 base but NOT deepseek-v4 (separate backend).
DEEPSEEK_R1_BACKEND_NAME_PATTERNS = (
    "deepseek-ai/deepseek-r1",
    "deepseek-r1",
    "deepseek_r1",
    "deepseek-ai/deepseek-v3",
    "deepseek-v3",
    "deepseek_v3",
)


def is_deepseek_r1_backend_model(model_name: str | None) -> bool:
    normalized = (model_name or "").strip().lower()
    return any(pattern in normalized for pattern in DEEPSEEK_R1_BACKEND_NAME_PATTERNS)
