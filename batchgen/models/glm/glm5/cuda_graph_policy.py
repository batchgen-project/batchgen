"""GLM-5 DSA CUDA graph routing policy helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping


def glm5_dsa_cuda_graph_requested_for_model(
    model_name: str | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> bool:
    env = os.environ if environ is None else environ
    return (
        env.get("BATCHGEN_GLM5_DSA_CUDA_GRAPH", "0") == "1"
        and "glm" in (model_name or "").lower()
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
    return glm5_dsa_cuda_graph_requested_for_model(model_name, environ=environ)
