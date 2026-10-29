"""Per-model routing of the ``--enable-cuda-graph`` CLI flag.

``Basic_Config.enable_cuda_graphs`` defaults to off (``config.py``); the
worker turns it on from the CLI flag only for models with a decode-graph path
the flag controls:

* GLM-5 FP8 family: whole-model decode graph (routed by
  ``batchgen.models.glm.glm5.cuda_graph_policy``).
* Kimi K2.5 / K2.6: whole-model decode graph through the cuda-graph adapter.
* GPT-OSS-120B: whole-model decode graph
  (``batchgen.models.openai.gpt_oss_120b.cuda_graph_segments``).

For every other model ``--enable-cuda-graph`` is rejected with
:class:`UnsupportedCudaGraphFlagError` at argument validation, before any
worker starts. Ignoring it would let a launch look graph-enabled while it
decodes eagerly, which is how GPT-OSS ran eager unnoticed before this module.
Kimi-Linear / Kimi-K3 decode graphs are chosen by their planner
(``Basic_Config.decode_graph_mode``) and per batch by
``batchgen_debug.kimi_decode_graph_mode``, so the CLI flag is rejected for
them too, with an error that names the knob that does control them.
"""

from __future__ import annotations

from batchgen.config.model_name_utils import is_kimi_k25_backend_model
from batchgen.models.glm.glm5.cuda_graph_policy import is_glm5_fp8_graph_default_model

GPT_OSS_GRAPH_FAMILY = "gpt-oss-120b"
KIMI_K25_GRAPH_FAMILY = "kimi-k2.5"
GLM5_FP8_GRAPH_FAMILY = "glm-5-fp8"


class UnsupportedCudaGraphFlagError(ValueError):
    """``--enable-cuda-graph`` was passed for a model the flag cannot affect."""


def cuda_graph_flag_family(model_name: str | None) -> str | None:
    """Return the decode-graph family ``--enable-cuda-graph`` turns on, or None.

    The name tests match the model guards in ``_warmup_cuda_graphs`` and
    ``_setup_cuda_graphs`` (``batchgen_worker.py``), so a model accepted here
    is one the worker captures graphs for.
    """
    lowered = (model_name or "").lower()
    if "gpt-oss-120b" in lowered:
        return GPT_OSS_GRAPH_FAMILY
    if is_kimi_k25_backend_model(model_name):
        return KIMI_K25_GRAPH_FAMILY
    if "glm" in lowered and is_glm5_fp8_graph_default_model(model_name):
        return GLM5_FP8_GRAPH_FAMILY
    return None


def require_cuda_graph_flag_supported(
    model_name: str | None,
    *,
    enable_cuda_graph: bool,
) -> None:
    """Raise if ``--enable-cuda-graph`` is set for a model it cannot affect."""
    if not enable_cuda_graph or cuda_graph_flag_family(model_name) is not None:
        return
    lowered = (model_name or "").lower()
    if "kimi-linear" in lowered or "kimi-k3" in lowered:
        raise UnsupportedCudaGraphFlagError(
            f"--enable-cuda-graph does not apply to {model_name!r}: Kimi-Linear/K3 "
            "decode graphs are chosen by the planner (Basic_Config.decode_graph_mode) "
            "and per batch by batchgen_debug.kimi_decode_graph_mode. Drop the flag."
        )
    raise UnsupportedCudaGraphFlagError(
        f"--enable-cuda-graph is not supported for {model_name!r}: the model has "
        "no decode CUDA graph this flag can enable, so decode would run eager. "
        "Supported: GPT-OSS-120B, Kimi K2.5/K2.6, GLM-5 FP8 family. Drop the flag."
    )


def cli_enables_whole_model_graph(
    model_name: str | None,
    *,
    enable_cuda_graph: bool,
) -> bool:
    """True when ``--enable-cuda-graph`` turns on a K2.5 or GPT-OSS decode graph.

    GLM-5 is routed separately (``glm5_any_cuda_graph_requested_for_model``),
    so this returns False for it.
    """
    return bool(enable_cuda_graph) and cuda_graph_flag_family(model_name) in (
        GPT_OSS_GRAPH_FAMILY,
        KIMI_K25_GRAPH_FAMILY,
    )
