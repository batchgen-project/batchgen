from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


# Ordered from more specific to less specific because matching uses substring checks.
MODEL_NAME_PATTERNS: Dict[str, str] = {
    "MiniMaxAI/MiniMax-M2.5": "minimax_m25",
    "MiniMax-M2.5": "minimax_m25",
    "moonshotai/Kimi-K2.5": "kimi_k25",
    "Kimi-K2.5": "kimi_k25",
    "THUDM/GLM-5": "glm5",
    "GLM-5-FP8": "glm5",
    "GLM-5": "glm5",
    "DeepSeek-R1": "deepseek_v3",
    "DeepSeek-V3": "deepseek_v3",
    "DeepSeek-V2-Lite": "deepseek_v2",
    "DeepSeek-V2": "deepseek_v2",
    "Mixtral-8x22B": "mixtral",
    "Mixtral-8x7B": "mixtral",
    "openai/gpt-oss-120b": "gpt_oss",
    "gpt-oss": "gpt_oss",
}

ARCH_PATTERNS: Dict[str, str] = {
    "DeepseekV3": "deepseek_v3",
    "DeepseekV2": "deepseek_v2",
    "Mixtral": "mixtral",
    "GptOss": "gpt_oss",
    "Qwen2Moe": "qwen2_moe",
    "MiniMaxM2": "minimax_m25",
    "KimiK25": "kimi_k25",
    "ChatGLM": "glm5",
    "GLM": "glm5",
}

MODEL_TYPE_ALIASES: Dict[str, str] = {
    "gpt_oss": "gpt_oss",
    "deepseek_v3": "deepseek_v3",
    "deepseek_v2": "deepseek_v2",
    "minimax_m25": "minimax_m25",
    "kimi_k25": "kimi_k25",
    "kimi_k2": "kimi_k25",
    "glm5": "glm5",
    "chatglm": "glm5",
}


def detect_model_type_from_identifier(model_identifier: str) -> Optional[str]:
    if not model_identifier:
        return None

    for pattern, model_type in MODEL_NAME_PATTERNS.items():
        if pattern in model_identifier:
            return model_type
    return None


def detect_model_type_from_config_dict(data: Dict[str, Any]) -> Optional[str]:
    model_type = data.get("model_type")
    if model_type in MODEL_TYPE_ALIASES:
        return MODEL_TYPE_ALIASES[model_type]

    for arch in data.get("architectures", []):
        for pattern, config_type in ARCH_PATTERNS.items():
            if pattern in arch:
                return config_type

    return None


def detect_model_type_from_directory(model_dir: str | Path) -> Optional[str]:
    config_path = Path(model_dir) / "config.json"
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
        detected = detect_model_type_from_config_dict(data)
        if detected is not None:
            return detected

    return detect_model_type_from_identifier(str(model_dir))
