"""GLM-5 CUDA graph routing policy helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping

GLM5_DSA_CUDA_GRAPH_ENV = "BATCHGEN_GLM5_DSA_CUDA_GRAPH"
GLM5_MOE_CUDA_GRAPH_ENV = "BATCHGEN_GLM5_MOE_CUDA_GRAPH"
GLM5_WHOLE_MODEL_CUDA_GRAPH_ENV = "BATCHGEN_GLM5_WHOLE_MODEL_CUDA_GRAPH"
GLM5_WHOLE_MODEL_GRAPH_COMPARE_ENV = "BATCHGEN_GLM5_WHOLE_MODEL_GRAPH_COMPARE"
GLM5_POWER_OF_TWO_BUCKETS_32 = [1, 2, 4, 8, 16, 32]


def _is_glm_model(model_name: str | None) -> bool:
    return "glm" in (model_name or "").lower()


def _is_glm5_fp8_graph_default_model(model_name: str | None) -> bool:
    normalized = (model_name or "").strip().lower()
    return any(
        pattern in normalized
        for pattern in (
            "zai-org/glm-5-fp8",
            "zai-org/glm-5.1-fp8",
            "glm-5-fp8",
            "glm_5_fp8",
            "glm-5.1-fp8",
            "glm_5.1_fp8",
            "glm5-fp8",
            "glm5_fp8",
            "glm51-fp8",
            "glm51_fp8",
        )
    )


def is_glm5_fp8_graph_default_model(model_name: str | None) -> bool:
    return _is_glm5_fp8_graph_default_model(model_name)


def glm5_dsa_cuda_graph_requested_for_model(
    model_name: str | None,
    *,
    enable_cuda_graph: bool = False,
    environ: Mapping[str, str] | None = None,
) -> bool:
    env = os.environ if environ is None else environ
    return (
        (
            env.get(GLM5_DSA_CUDA_GRAPH_ENV, "0") == "1"
            or (enable_cuda_graph and _is_glm5_fp8_graph_default_model(model_name))
        )
        and _is_glm_model(model_name)
    )


def glm5_moe_cuda_graph_requested_for_model(
    model_name: str | None,
    *,
    enable_cuda_graph: bool = False,
    environ: Mapping[str, str] | None = None,
) -> bool:
    env = os.environ if environ is None else environ
    return (
        (
            env.get(GLM5_MOE_CUDA_GRAPH_ENV, "0") == "1"
            or (enable_cuda_graph and _is_glm5_fp8_graph_default_model(model_name))
        )
        and _is_glm_model(model_name)
    )


def glm5_whole_model_cuda_graph_requested_for_model(
    model_name: str | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> bool:
    env = os.environ if environ is None else environ
    return env.get(GLM5_WHOLE_MODEL_CUDA_GRAPH_ENV, "0") == "1" and _is_glm_model(model_name)


def glm5_whole_model_cuda_graph_compare_requested_for_model(
    model_name: str | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> bool:
    env = os.environ if environ is None else environ
    return env.get(GLM5_WHOLE_MODEL_GRAPH_COMPARE_ENV, "0") == "1" and _is_glm_model(model_name)


def glm5_segmented_cuda_graph_requested_for_model(
    model_name: str | None,
    *,
    enable_cuda_graph: bool = False,
    environ: Mapping[str, str] | None = None,
) -> bool:
    return (
        glm5_dsa_cuda_graph_requested_for_model(
            model_name,
            enable_cuda_graph=enable_cuda_graph,
            environ=environ,
        )
        or glm5_moe_cuda_graph_requested_for_model(
            model_name,
            enable_cuda_graph=enable_cuda_graph,
            environ=environ,
        )
    )


def glm5_any_cuda_graph_requested_for_model(
    model_name: str | None,
    *,
    enable_cuda_graph: bool = False,
    environ: Mapping[str, str] | None = None,
) -> bool:
    return (
        glm5_whole_model_cuda_graph_requested_for_model(model_name, environ=environ)
        or glm5_whole_model_cuda_graph_compare_requested_for_model(model_name, environ=environ)
        or glm5_segmented_cuda_graph_requested_for_model(
            model_name,
            enable_cuda_graph=enable_cuda_graph,
            environ=environ,
        )
    )


def glm5_effective_decode_attn_mode(
    model_type: str | None,
    configured_attn_mode: int,
) -> int:
    """GLM-5 decode uses the modern continuous path even on single-node runs."""
    if _is_glm_model(model_type):
        return 3
    return configured_attn_mode


def glm5_cuda_graph_bucket_for_batch_size(
    batch_size: int,
    bucket_sizes: list[int] | tuple[int, ...] = GLM5_POWER_OF_TWO_BUCKETS_32,
) -> int | None:
    """Return the smallest configured GLM-5 graph bucket for batch_size.

    ``None`` means no graph bucket can represent the batch and the caller should
    use eager execution. A zero batch has no graph work to replay.
    """

    if batch_size <= 0:
        return None
    for bucket_size in bucket_sizes:
        if batch_size <= int(bucket_size):
            return int(bucket_size)
    return None


def glm5_moe_graph_bucket_capacity(
    *,
    max_rank_batch_size: int,
    world_size: int,
    bucket_sizes: list[int] | tuple[int, ...] = GLM5_POWER_OF_TWO_BUCKETS_32,
) -> tuple[int, int] | None:
    """Return ``(per_rank_bucket, effective_global_rows)`` for MoE graph replay.

    GLM-5 MoE graph buckets are per-rank max batch sizes. The captured global
    routing domain is ``world_size * per_rank_bucket``.
    """

    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    bucket = glm5_cuda_graph_bucket_for_batch_size(
        max_rank_batch_size,
        bucket_sizes,
    )
    if bucket is None:
        return None
    return bucket, int(world_size) * bucket


def should_warmup_cuda_graphs_before_decode(
    *,
    graph_manager_is_initialized: bool,
    global_batch_has_queueing: bool,
    model_name: str | None,
    enable_cuda_graph: bool = False,
    environ: Mapping[str, str] | None = None,
) -> bool:
    if graph_manager_is_initialized:
        return False
    if not global_batch_has_queueing:
        return True
    return glm5_any_cuda_graph_requested_for_model(
        model_name,
        enable_cuda_graph=enable_cuda_graph,
        environ=environ,
    )
