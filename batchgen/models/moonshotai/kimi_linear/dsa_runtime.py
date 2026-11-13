# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                         #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #
"""Kimi-K3.1 DSA: the native code its serving path loads, declared once.

Every DSA kernel this model reaches is imported lazily on first use. That
laziness is what lets a host import the model without the kernels built — and
it is also why a missing build is invisible until the first DSA layer runs.

Two startup seams close that gap, both keyed by model type:

* the core ``runtime_preflight`` contract for ``kimi_k31`` refuses a host that
  lacks these extensions or DeepGEMM before any worker exists;
* :func:`preload_dsa_runtime`, run by the parallel strategy manager's
  ``preload_runtime_extensions`` during the pre-readiness startup pass, imports
  every extension through ``importlib`` (never the dev-JIT loader, so nothing
  can compile at serve time) and checks every third-party symbol the DSA
  paths call. A server that reports ready has already resolved all of them.

``tests/test_kimi_k31_dsa_runtime.py`` holds these declarations to the code:
every extension a DSA path resolves is listed here, every one listed is in the
unconditional AOT build (``batchgen_kernels/setup.py``) and the dev-JIT
registry, and the core preflight contract carries them.

Reached on route B (page-split FP8 index keys):

====================================  ========================================
extension / module                    reached by
====================================  ========================================
``_C_fused_indexer_k_norm_rope_       the decode index-key writer (eager and
write_fp8``                           graph steps)
``_C_dsa_fast_topk``                  the decode paged scorer and the causal
                                      prefill scorer (``fast_topk_2048_out``)
``deep_gemm``                         ``fp8_paged_mqa_logits`` (+ metadata) on
                                      decode, ``fp8_mqa_logits`` on prefill
sparse attention modules              ``sparse_dispatch.site_symbols()`` for
                                      the device: FA3 ``qv`` on Hopper;
                                      flashinfer trtllm-gen sparse MLA
                                      (decode) and ``flash_mla`` sparse
                                      prefill on Blackwell
====================================  ========================================

Triton kernels on the same paths (the selection transform, the scorer
prologue, ``act_quant``) compile on first launch; the decode graph's warmup
iterations launch them before any capture. Shape-complete prewarm of the
DeepGEMM JIT kernels (one per aligned batch) is not done here.
"""

from __future__ import annotations

import importlib
from typing import Callable, Dict, Optional, Tuple

#: The fused LayerNorm + identity-RoPE + e4m3 quantize + page-split write of a
#: decode step's index keys (``indexer.indexer_k_norm_rope_write_fp8``).
FUSED_INDEX_K_WRITER_EXTENSION = (
    "batchgen_kernels.attention._C_fused_indexer_k_norm_rope_write_fp8"
)

#: ``fast_topk_2048_out`` (``dsa.fast_topk_cuda``), the top-k of both scorers.
FAST_TOPK_EXTENSION = "batchgen_kernels.attention._C_dsa_fast_topk"

#: Every AOT extension the Kimi-K3.1 DSA serving path loads, in addition to
#: everything the Kimi-K3 contract already requires.
K31_DSA_AOT_EXTENSIONS: Tuple[str, ...] = (
    FUSED_INDEX_K_WRITER_EXTENSION,
    FAST_TOPK_EXTENSION,
)

#: Third-party modules the DSA indexer calls, and the callables it needs. The
#: sparse-attention modules depend on the device and come from
#: ``batchgen.attention.mla.sparse_dispatch.site_symbols()``.
K31_DSA_SITE_SYMBOLS: Dict[str, Tuple[str, ...]] = {
    "deep_gemm": (
        "fp8_paged_mqa_logits",
        "get_paged_mqa_logits_metadata",
        "get_num_sms",
        "fp8_mqa_logits",
    ),
}


#: The indexer geometry the route-B kernels are instantiated for: the fused
#: writer and both scorers read a 128-wide index row, and the causal prefill
#: top-k is ``fast_topk_2048_out`` only. The paged scorer additionally needs
#: the head count to divide 128.
SERVED_INDEX_HEAD_DIM = 128
SERVED_INDEX_TOPK = 2048


class DsaRuntimeError(RuntimeError):
    """A native dependency of the Kimi-K3.1 DSA serving path is missing."""


def require_servable_indexer(dsa) -> None:
    """Refuse, before any weight is mapped, an indexer geometry the DSA
    serving kernels cannot run — instead of at the first long prompt.

    ``dsa`` is the parsed ``KimiDsaConfig`` (``KimiLinearConfig.dsa()``); its
    parser already refuses a rotated index, a non-LayerNorm key norm and a
    top-k reuse schedule.
    """
    if dsa is None:
        raise ValueError(
            "a Kimi-K3.1 (model_type='kimi_k31') config without dsa_config "
            "cannot be served: the indexer is what makes it K3.1"
        )
    problems = []
    if int(dsa.index_head_dim) != SERVED_INDEX_HEAD_DIM:
        problems.append(f"index_head_dim={dsa.index_head_dim} (kernels: "
                        f"{SERVED_INDEX_HEAD_DIM})")
    if int(dsa.index_topk) != SERVED_INDEX_TOPK:
        problems.append(f"index_topk={dsa.index_topk} (prefill top-k: "
                        f"{SERVED_INDEX_TOPK} only)")
    heads = int(dsa.index_num_heads)
    if heads <= 0 or SERVED_INDEX_HEAD_DIM % heads != 0:
        problems.append(f"index_num_heads={heads} (the paged scorer needs a "
                        f"divisor of {SERVED_INDEX_HEAD_DIM})")
    if problems:
        raise NotImplementedError(
            "Kimi-K3.1 DSA serving is instantiated for the released indexer "
            "geometry; this config has " + ", ".join(problems)
        )


def preload_dsa_runtime(
    import_module: Callable[[str], object] = importlib.import_module,
    attention_symbols: Optional[Dict[str, Tuple[str, ...]]] = None,
) -> Tuple[str, ...]:
    """Import every DSA extension and check every third-party symbol.

    AOT-only by construction: a plain import, so a host without the build
    raises here — before readiness — instead of reaching the kernels'
    dev-JIT fallback at the first DSA layer. A loaded extension stays in
    ``sys.modules``, so the kernels' own lazy loaders resolve to it later.

    ``attention_symbols`` defaults to the sparse-attention seam's modules for
    this device (``sparse_dispatch.site_symbols()``).

    Returns the names resolved, in order. Raises :class:`DsaRuntimeError`
    naming the first missing piece.
    """
    if attention_symbols is None:
        from batchgen.attention.mla.sparse_dispatch import site_symbols

        attention_symbols = site_symbols()
    resolved = []
    for name in K31_DSA_AOT_EXTENSIONS:
        try:
            import_module(name)
        except ImportError as exc:
            raise DsaRuntimeError(
                f"Kimi-K3.1 DSA extension {name} is not built: {exc}. Build "
                "batchgen_kernels ahead of time; the DSA path never compiles "
                "at serve time"
            ) from exc
        resolved.append(name)
    for module_name, symbols in {**K31_DSA_SITE_SYMBOLS, **attention_symbols}.items():
        try:
            module = import_module(module_name)
        except ImportError as exc:
            raise DsaRuntimeError(
                f"Kimi-K3.1 DSA needs {module_name}, which failed to import: {exc}"
            ) from exc
        missing = [s for s in symbols if not callable(getattr(module, s, None))]
        if missing:
            raise DsaRuntimeError(
                f"Kimi-K3.1 DSA needs {module_name}.{', '.join(missing)}, which "
                "the installed package does not provide"
            )
        resolved.append(module_name)
    return tuple(resolved)
