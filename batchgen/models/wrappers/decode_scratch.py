"""Decode scratch-memory reservation registry."""

from __future__ import annotations

from typing import Any, Callable, Dict

DecodeScratchEstimator = Callable[..., float]


_ESTIMATORS: Dict[str, DecodeScratchEstimator] = {}


def register_decode_scratch_estimator(
    model_type: str,
    estimator: DecodeScratchEstimator,
) -> None:
    key = _normalize_model_type(model_type)
    if not callable(estimator):
        raise RuntimeError(f"Decode scratch estimator for {key!r} is not callable")
    _ESTIMATORS[key] = estimator


def register_no_decode_scratch_model(model_type: str) -> None:
    register_decode_scratch_estimator(model_type, _estimate_no_decode_scratch)


def estimate_decode_scratch_reserve_gb(
    *,
    model_config: Any,
    world_size: int,
    max_num_seq_per_rank: int,
) -> float:
    model_type = _normalize_model_type(getattr(model_config, "model_type", None))
    estimator = _ESTIMATORS.get(model_type)
    if estimator is None:
        raise RuntimeError(
            "Decode scratch reserve estimator is not registered for "
            f"model_type={model_type!r}"
        )

    reserve_gb = float(
        estimator(
            model_config=model_config,
            world_size=world_size,
            max_num_seq_per_rank=max_num_seq_per_rank,
        )
    )
    if reserve_gb < 0:
        raise RuntimeError(
            "Decode scratch reserve estimator returned a negative value: "
            f"model_type={model_type!r}, reserve_gb={reserve_gb}"
        )
    return reserve_gb


def _estimate_no_decode_scratch(
    *,
    model_config: Any,
    world_size: int,
    max_num_seq_per_rank: int,
) -> float:
    del model_config, world_size, max_num_seq_per_rank
    return 0.0


def _normalize_model_type(model_type: Any) -> str:
    if model_type is None:
        raise RuntimeError("Decode scratch reserve requires model_config.model_type")
    key = str(model_type).strip()
    if not key:
        raise RuntimeError("Decode scratch reserve requires non-empty model_type")
    return key


for _MODEL_TYPE in (
    "deepseek_v2",
    "deepseek_v3",
    "deepseek_v4",
    "glm_moe_dsa",
    "kimi_k25",
    "minimax_m25",
    "mixtral",
    "Qwen2",
):
    register_no_decode_scratch_model(_MODEL_TYPE)


def _register_builtin_estimators() -> None:
    from batchgen.models.openai.gpt_oss_120b.decode_scratch import (
        estimate_gpt_oss_decode_scratch_reserve_gb,
    )

    register_decode_scratch_estimator(
        "gpt_oss",
        estimate_gpt_oss_decode_scratch_reserve_gb,
    )


_register_builtin_estimators()
