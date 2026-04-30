"""GLM-5 CUDA graph routing policy helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping

GLM5_DSA_CUDA_GRAPH_ENV = "BATCHGEN_GLM5_DSA_CUDA_GRAPH"
GLM5_MOE_CUDA_GRAPH_ENV = "BATCHGEN_GLM5_MOE_CUDA_GRAPH"
GLM5_WHOLE_MODEL_CUDA_GRAPH_ENV = "BATCHGEN_GLM5_WHOLE_MODEL_CUDA_GRAPH"


def _is_glm_model(model_name: str | None) -> bool:
    return "glm" in (model_name or "").lower()


def glm5_dsa_cuda_graph_requested_for_model(
    model_name: str | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> bool:
    env = os.environ if environ is None else environ
    return env.get(GLM5_DSA_CUDA_GRAPH_ENV, "0") == "1" and _is_glm_model(model_name)


def glm5_moe_cuda_graph_requested_for_model(
    model_name: str | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> bool:
    env = os.environ if environ is None else environ
    return env.get(GLM5_MOE_CUDA_GRAPH_ENV, "0") == "1" and _is_glm_model(model_name)


def glm5_whole_model_cuda_graph_requested_for_model(
    model_name: str | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> bool:
    env = os.environ if environ is None else environ
    return env.get(GLM5_WHOLE_MODEL_CUDA_GRAPH_ENV, "0") == "1" and _is_glm_model(model_name)


def glm5_segmented_cuda_graph_requested_for_model(
    model_name: str | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> bool:
    return (
        glm5_dsa_cuda_graph_requested_for_model(model_name, environ=environ)
        or glm5_moe_cuda_graph_requested_for_model(model_name, environ=environ)
    )


def glm5_any_cuda_graph_requested_for_model(
    model_name: str | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> bool:
    return (
        glm5_whole_model_cuda_graph_requested_for_model(model_name, environ=environ)
        or glm5_segmented_cuda_graph_requested_for_model(model_name, environ=environ)
    )


def should_warmup_cuda_graphs_before_decode(
    *,
    graph_manager_is_initialized: bool,
    global_batch_has_queueing: bool,
    model_name: str | None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    if graph_manager_is_initialized:
        return False
    if not global_batch_has_queueing:
        return True
    return glm5_any_cuda_graph_requested_for_model(model_name, environ=environ)
