"""GLM-5 CUDA graph routing policy helpers.

Phase C: layer / MoE / DSA / DSA-full / segmented graph modes are retired.
The whole-model graph (driven by ``--enable-cuda-graph`` on GLM-5-FP8) is
the only supported mode. Mode env vars (``BATCHGEN_GLM5_*_CUDA_GRAPH``,
``BATCHGEN_SEGMENTED_GRAPH``) are no longer read here; the predicates for
the retired modes return ``False`` unconditionally.

The server-level graph mode is decided only by the CLI at startup. The
eager-vs-graph compare is a per-batch choice inside that mode
(``batchgen_debug.glm5_whole_model_graph_compare``, read by the worker); it
no longer feeds these routing predicates.
"""

from __future__ import annotations

from collections.abc import Mapping

from batchgen.models.glm.glm5 import releases as glm5_releases

GLM5_POWER_OF_TWO_BUCKETS_32 = [1, 2, 4, 8, 16, 32]

# Every GLM-5 release's FP8 checkpoint name, in each spelling users pass.
_GLM5_FP8_GRAPH_DEFAULT_PATTERNS = glm5_releases.cuda_graph_fp8_default_patterns()


def _is_glm_model(model_name: str | None) -> bool:
    return "glm" in (model_name or "").lower()


def _is_glm5_fp8_graph_default_model(model_name: str | None) -> bool:
    normalized = (model_name or "").strip().lower()
    return any(pattern in normalized for pattern in _GLM5_FP8_GRAPH_DEFAULT_PATTERNS)


def is_glm5_fp8_graph_default_model(model_name: str | None) -> bool:
    return _is_glm5_fp8_graph_default_model(model_name)


def glm5_dsa_cuda_graph_requested_for_model(
    model_name: str | None,
    *,
    enable_cuda_graph: bool = False,
    environ: Mapping[str, str] | None = None,
) -> bool:
    # Phase C: DSA-only graph mode retired. Always False.
    return False


def glm5_dsa_full_cuda_graph_requested(
    *,
    environ: Mapping[str, str] | None = None,
) -> bool:
    # Phase C: DSA-full graph mode retired. Always False.
    return False


def glm5_moe_cuda_graph_requested_for_model(
    model_name: str | None,
    *,
    enable_cuda_graph: bool = False,
    environ: Mapping[str, str] | None = None,
) -> bool:
    # Phase C: MoE-only graph mode retired. Always False.
    return False


def glm5_whole_model_cuda_graph_requested_for_model(
    model_name: str | None,
    *,
    enable_cuda_graph: bool = False,
    environ: Mapping[str, str] | None = None,
) -> bool:
    # Phase C: whole-model graph activates iff `--enable-cuda-graph` is set
    # on a GLM-5-FP8 model. No env-var path; the legacy
    # `BATCHGEN_GLM5_WHOLE_MODEL_CUDA_GRAPH` and `BATCHGEN_SEGMENTED_GRAPH`
    # mode env vars are retired.
    if not _is_glm_model(model_name):
        return False
    return bool(enable_cuda_graph) and _is_glm5_fp8_graph_default_model(model_name)


def glm5_segmented_cuda_graph_requested_for_model(
    model_name: str | None,
    *,
    enable_cuda_graph: bool = False,
    environ: Mapping[str, str] | None = None,
) -> bool:
    # Phase C: segmented (DSA-only + MoE-only) graph mode retired.
    return False


def glm5_any_cuda_graph_requested_for_model(
    model_name: str | None,
    *,
    enable_cuda_graph: bool = False,
) -> bool:
    # Phase C: only the whole-model graph remains, and only the CLI decides
    # it (the per-batch compare selects behavior within that mode).
    return glm5_whole_model_cuda_graph_requested_for_model(
        model_name,
        enable_cuda_graph=enable_cuda_graph,
    )


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
