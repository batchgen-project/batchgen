# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                            #
# ---------------------------------------------------------------------------- #

"""GLM-5 specific wrappers for BatchGen execution.

Standalone wrappers for GLM-5 FP8 model — no cross-model imports.
- GLM5ExpertWrapper: Expert wrapper with FP8 dequantization
- GLM5AttnWrapper: Attention wrapper with FP8 dequant + DSA integration

Key differences from DeepSeek:
- kv_a_proj_with_mqa naming (same)
- DSA indexer integration in attention wrapper
- Different MLA dimensions (qk_nope=192, v_head=256)
- rope_interleave=True
"""

import logging
import os
from contextlib import nullcontext as _nullctx
from typing import Any, ClassVar, Dict, Optional, Set, Tuple

import torch
import torch.nn as nn

import torch.nn.functional as F

from batchgen.models.wrappers import ExpertWrapperBase, AttnWrapperBase
from batchgen.timing import init_decode_timer


def _quantize_indexer_k_pagesplit(
    k_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(e4m3 keys, fp32 per-token scales) for a page-split aux host write.

    The host region stores the layout DeepGEMM reads, so the quantization
    happens on the GPU before the copy: the host never sees bf16 indexer K,
    and the C++ copy of each page's two regions stays a flat memcpy.
    """

    from batchgen_kernels.attention.dsa.indexer_k_quant import (
        quantize_indexer_k_fp8,
    )

    k_fp8, scales = quantize_indexer_k_fp8(k_tensor)
    return k_fp8.contiguous(), scales.contiguous()

# BF16 strided-batched absorb + container dispatch (slice 2c-2). Triton-free,
# so this import is unguarded: decode depends on it.
from batchgen_kernels.attention.dsa.bf16_absorb import (
    BF16AbsorbWeights,
    absorb_out,
)

# Try importing FP8 absorb kernels (WP5)
try:
    from batchgen_kernels.attention.dsa.fp8_absorb import (
        FP8AbsorbWeights, fp8_q_absorb, fp8_out_absorb,
    )
    _HAS_FP8_ABSORB = True
    _FP8_ABSORB_IMPORT_ERROR = None
except Exception as _e:
    _HAS_FP8_ABSORB = False
    _FP8_ABSORB_IMPORT_ERROR = _e
    logging.debug(f"[WP5] FP8 absorb import failed: {_e}")

# Try importing fused indexer KV proj (WP2)
try:
    from batchgen_kernels.attention.dsa.fused_indexer_kv_proj_cuda import (
        build_module as _build_indexer_module,
        FP8IndexerWeightsCUDA,
    )
    _HAS_FUSED_INDEXER_KV = True
    _FUSED_INDEXER_KV_IMPORT_ERROR = None
except Exception as _e:
    _HAS_FUSED_INDEXER_KV = False
    _FUSED_INDEXER_KV_IMPORT_ERROR = _e
    logging.debug(f"[WP2] Fused indexer KV proj import failed: {_e}")

# Try importing fused scoring pipeline (WP4)
try:
    from batchgen_kernels.attention.dsa.fused_indexer_score import (
        FP8WqbWeightsCUDA,
        fused_score_pipeline,
    )
    _HAS_FUSED_SCORE = True
    _FUSED_SCORE_IMPORT_ERROR = None
except Exception as _e:
    _HAS_FUSED_SCORE = False
    _FUSED_SCORE_IMPORT_ERROR = _e
    logging.debug(f"[WP4] Fused scoring import failed: {_e}")

# Merged [q_b | wq_b] decode GEMM (kernel version management): the switch
# decides whether initialize_fused_kernels builds the merged operand; the
# construction itself is torch-only and fails loud on a misaligned member.
from batchgen.models.glm.glm5.q_b_wq_b_merge import build_q_b_wq_b_merged
from batchgen.models.glm.glm5.qb_wqb_merge_policy import glm5_qb_wqb_merge_enabled


def _required_dsa_kernel_import_failures():
    """Return unavailable production DSA kernels and their import errors."""
    failures = {}
    if not _HAS_FUSED_INDEXER_KV:
        failures["WP2 fused indexer KV projection"] = _FUSED_INDEXER_KV_IMPORT_ERROR
    if not _HAS_FUSED_SCORE:
        failures["WP4 fused indexer scoring"] = _FUSED_SCORE_IMPORT_ERROR
    if not _HAS_FP8_ABSORB:
        failures["WP5 FP8 absorb"] = _FP8_ABSORB_IMPORT_ERROR
    return failures

# Initialize GLM-5 decode timer (activated by BATCHGEN_DECODE_TIMING=1)
_GLM5_ATTN_CATEGORIES = [
    "act_quant", "q_proj", "kv_proj", "kv_write",
    "indexer_k", "indexer_score", "sparse_gather",
    "q_absorb", "sparse_attn", "o_proj",
]
_GLM5_MOE_CATEGORIES = [
    "allgather", "routing", "dispatch", "grouped_gemm",
    "expert_loop", "scatter_reduce", "allreduce",
]
_glm5_decode_timer = init_decode_timer(
    "GLM-5", _GLM5_ATTN_CATEGORIES + _GLM5_MOE_CATEGORIES
)

_GLM5_DSA_CUDA_GRAPH_ENV = "BATCHGEN_GLM5_DSA_CUDA_GRAPH"
_GLM5_DSA_FULL_CUDA_GRAPH_ENV = "BATCHGEN_GLM5_DSA_FULL_CUDA_GRAPH"
_GLM5_DSA_GRAPH_COMPARE_ENV = "BATCHGEN_GLM5_DSA_GRAPH_COMPARE"
_glm5_dsa_graph_eager_fallback_logged = False
_glm5_dsa_graph_compare_unavailable_logged = False


def _record_glm5_dsa_dispatch(
    path: str,
    *,
    layer_idx: int,
    bsz: int,
    reason: str,
) -> None:
    AttnWrapperBase.record_glm5_dispatch(
        kind="dsa",
        path=path,
        layer_idx=layer_idx,
        bsz=bsz,
        reason=reason,
    )


def _glm5_dsa_cuda_graph_required() -> bool:
    mode = _glm5_dsa_debug_mode()
    if mode == "eager":
        return False
    if mode == "graph":
        return True
    return (
        os.environ.get(_GLM5_DSA_CUDA_GRAPH_ENV, "0") == "1"
        or os.environ.get(_GLM5_DSA_FULL_CUDA_GRAPH_ENV, "0") == "1"
    )


def _debug_flag_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _glm5_dsa_debug_dict() -> dict:
    debug = getattr(AttnWrapperBase, "batchgen_debug", None) or {}
    return debug if isinstance(debug, dict) else {}


def _glm5_dsa_debug_mode() -> Optional[str]:
    value = _glm5_dsa_debug_dict().get("glm5_dsa_mode")
    if not isinstance(value, str):
        return None
    mode = value.strip().lower()
    return mode if mode in {"graph", "eager"} else None


def _glm5_dsa_graph_compare_active() -> bool:
    if _glm5_dsa_debug_mode() == "eager":
        return False
    debug = _glm5_dsa_debug_dict()
    return (
        _debug_flag_enabled(debug.get("glm5_dsa_graph_compare"))
        or os.environ.get(_GLM5_DSA_GRAPH_COMPARE_ENV, "0") == "1"
    )


def _glm5_dsa_graph_compare_layer_enabled(layer_idx: int) -> bool:
    if not _glm5_dsa_graph_compare_active():
        return False
    debug = _glm5_dsa_debug_dict()
    layers = debug.get("glm5_dsa_graph_compare_layers")
    if layers is None:
        layers = os.environ.get("BATCHGEN_GLM5_DSA_GRAPH_COMPARE_LAYERS", "0")
    if layers in ("all", "*"):
        return True
    if isinstance(layers, int):
        return layer_idx == layers
    if isinstance(layers, str):
        return str(layer_idx) in {
            item.strip() for item in layers.split(",") if item.strip()
        }
    if isinstance(layers, (list, tuple, set)):
        try:
            return layer_idx in {int(item) for item in layers}
        except (TypeError, ValueError):
            logging.warning(
                "Ignoring invalid glm5_dsa_graph_compare_layers=%r; defaulting to layer 0",
                layers,
            )
            return layer_idx == 0
    return layer_idx == 0


def _glm5_dsa_graph_compare_fail_on_mismatch() -> bool:
    debug = getattr(AttnWrapperBase, "batchgen_debug", None) or {}
    return _debug_flag_enabled(debug.get("glm5_dsa_graph_compare_fail_on_mismatch"))


def _glm5_dsa_cuda_graph_can_replay(
    cache_seqlens: torch.Tensor,
    max_seqlen: int,
    index_topk: int,
    *,
    captured_max_seqlen: Optional[int] = None,
) -> bool:
    """Return whether the fixed selected-KV graph contract can replay.

    The unified selector always writes a fixed selected-KV buffer
    ``[B, index_topk, 1, kv_dim]``. Runtime row lengths are carried by
    ``selected_lengths`` and must not route short rows to eager DSA.
    """

    if max_seqlen <= 0 or index_topk <= 0:
        return False
    if cache_seqlens.ndim != 1:
        return False
    return True


def _log_glm5_dsa_graph_eager_fallback_once(
    layer_idx: int,
    cache_seqlens: torch.Tensor,
    index_topk: int,
    reason: str = "rows are not all graph-safe",
) -> None:
    global _glm5_dsa_graph_eager_fallback_logged
    if _glm5_dsa_graph_eager_fallback_logged or layer_idx != 0:
        return
    _glm5_dsa_graph_eager_fallback_logged = True
    logging.warning(
        "GLM-5 DSA CUDA graph requested but %s; using eager DSA "
        "(index_topk=%s, min_cache_seqlen=%s).",
        reason,
        index_topk,
        int(cache_seqlens.min().item()),
    )


def _log_glm5_dsa_graph_compare_unavailable_once(
    layer_idx: int,
    reason: str,
) -> None:
    global _glm5_dsa_graph_compare_unavailable_logged
    if _glm5_dsa_graph_compare_unavailable_logged or layer_idx != 0:
        return
    _glm5_dsa_graph_compare_unavailable_logged = True
    logging.warning(
        "GLM-5 DSA graph/eager compare requested but graph side-channel is "
        "unavailable (%s); returning eager DSA output.",
        reason,
    )


def _glm5_dsa_gpu_page_table_tensor(gpu_paged_kv_manager) -> Optional[torch.Tensor]:
    get_storage = getattr(gpu_paged_kv_manager, "get_cuda_graph_page_table_storage", None)
    if get_storage is not None:
        try:
            return get_storage()
        except RuntimeError:
            return None
    get_graph_table = getattr(gpu_paged_kv_manager, "get_cuda_graph_page_table", None)
    if get_graph_table is None:
        return None
    try:
        return get_graph_table()
    except RuntimeError:
        return None


def _glm5_dsa_page_table_signature(
    page_table: Optional[torch.Tensor],
) -> Optional[Tuple[int, Tuple[int, ...], str, str]]:
    if page_table is None:
        return None
    return (
        int(page_table.data_ptr()),
        tuple(int(dim) for dim in page_table.shape),
        str(page_table.dtype),
        str(page_table.device),
    )


def _fail_if_glm5_dsa_cuda_graph_required_without_replay() -> None:
    if not _glm5_dsa_cuda_graph_required():
        return
    raise RuntimeError(
        f"{_GLM5_DSA_CUDA_GRAPH_ENV}=1 requested GLM-5 DSA CUDA graph replay, "
        "but no replayable graph is available for this decode step. Refusing to "
        "silently fall back to eager DSA decode."
    )


def glm5_fp8_dequantization(
    weight_data_fp8: torch.Tensor,
    weight_scale_inv_fp32: torch.Tensor,
    block_size=(128, 128),
) -> torch.Tensor:
    """Blockwise FP8 dequantization (standalone, no cross-model import)."""
    rows, cols = weight_data_fp8.shape
    block_rows, block_cols = block_size
    n_block_rows = rows // block_rows
    n_block_cols = cols // block_cols
    weight_4d = weight_data_fp8.reshape(
        n_block_rows, block_rows, n_block_cols, block_cols
    ).to(torch.float32)
    scale_4d = weight_scale_inv_fp32.unsqueeze(1).unsqueeze(-1)
    dequantized_4d = weight_4d * scale_4d
    return dequantized_4d.reshape(rows, cols).to(torch.bfloat16)


class GLM5ExpertWrapper(ExpertWrapperBase):
    """Expert wrapper for GLM-5 models.

    Supports both variants:
    - GLM-5 (BF16): standard nn.Linear forward
    - GLM-5-FP8: FP8 deepgemm forward with w8a16_gemm
    Controlled by `is_fp8` flag set during PSM configuration.
    """

    def __init__(
        self,
        module: nn.Module,
        layer_idx: int,
        expert_idx: int,
        core_engine,
        engine_config,
        model_config,
        persistent: bool = False,
        weight_dequant_scale: Optional[Dict[str, torch.Tensor]] = None,
        is_fp8: bool = False,
    ):
        super().__init__(
            module, layer_idx, expert_idx, core_engine, engine_config, model_config,
            persistent
        )
        self.weight_dequant_scale = weight_dequant_scale or {}
        self.is_fp8 = is_fp8
        self.cached_gate = None
        self.cached_up = None
        self.cached_down = None
        # Fused gate/up binding (persistent shared expert only, see
        # ``_bind_fused_gate_up``): one [2I, K] FP8 weight whose two row halves
        # ARE ``cached_gate`` / ``cached_up``, and its [2I/128, K/128] scale.
        self.cached_gate_up = None
        self.cached_gate_up_scale = None

    def dequantize_weights(
        self, weights_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if not self.is_fp8:
            return weights_dict
        result = {}
        for name, weight in weights_dict.items():
            scale_key = f"{name}_scale_inv"
            if scale_key in self.weight_dequant_scale:
                result[name] = glm5_fp8_dequantization(
                    weight, self.weight_dequant_scale[scale_key]
                )
            else:
                result[name] = weight
        return result

    def _register_fp8_weights(self):
        """Cache weight pointers for persistent experts.

        Supports both:
        - Placeholders with flat attrs (routed experts, set by _load_local_routed_experts)
        - Real Glm5Expert modules (shared experts, weights in gate_proj.weight.data)
        """
        if hasattr(self.module, 'fp8_gate'):
            # Placeholder with flat attrs
            self.cached_gate = self.module.fp8_gate
            self.cached_up = self.module.fp8_up
            self.cached_down = self.module.fp8_down
        else:
            # Real nn.Module (shared experts)
            self.cached_gate = self.module.gate_proj.weight.data
            self.cached_up = self.module.up_proj.weight.data
            self.cached_down = self.module.down_proj.weight.data
            # FP8 only: the fused chain is a DeepGEMM FP8 chain, and a BF16
            # checkpoint keeps the per-projection path it always ran.
            gate_up = getattr(self.module, "gate_up_weight", None)
            if gate_up is not None and self.is_fp8:
                self._bind_fused_gate_up(gate_up)

    def _unregister_fp8_weights(self):
        self.cached_gate = None
        self.cached_up = None
        self.cached_down = None
        self.cached_gate_up = None
        self.cached_gate_up_scale = None

    def _bind_fused_gate_up(self, gate_up: torch.Tensor) -> None:
        """Bind the fused [gate; up] weight the loader allocated.

        The loader stores gate and up as the two row halves of ONE [2I, K]
        allocation (``Glm5ParallelStrategyManager._load_shared_expert_module``),
        so the fused GEMM reads exactly the bytes the separate gate and up
        GEMMs read -- no second copy of either weight exists. Every assumption
        that makes ``[gate; up] @ x`` equal to the two separate products is
        checked here, once, rather than trusted at every forward:
          * the halves alias the fused rows (same storage, same order);
          * I is a whole number of 128-row scale blocks, so the stacked block
            scales ``[gate_scale; up_scale]`` are the fused weight's own
            block scales (no block straddles the gate/up boundary).
        The stacked scale is a new [2I/128, K/128] fp32 tensor (6 KiB per layer
        at GLM-5.2 shapes); the per-half scales stay bound for the unfused path.
        """
        if not self.is_fp8:
            raise RuntimeError(
                f"Layer {self.layer_idx}: fused gate/up binding requires FP8 weights")
        gate, up = self.cached_gate, self.cached_up
        inter, hidden = gate.shape
        if (gate_up.dim() != 2 or tuple(gate_up.shape) != (2 * inter, hidden)
                or tuple(up.shape) != (inter, hidden) or not gate_up.is_contiguous()):
            raise RuntimeError(
                f"Layer {self.layer_idx}: fused gate/up weight must be a contiguous "
                f"[2I, K] = {(2 * inter, hidden)} tensor over gate {tuple(gate.shape)} "
                f"and up {tuple(up.shape)}, got {tuple(gate_up.shape)}")
        if (gate.data_ptr() != gate_up.data_ptr()
                or up.data_ptr() != gate_up[inter:].data_ptr()):
            raise RuntimeError(
                f"Layer {self.layer_idx}: gate/up weights are not the row halves of "
                "the fused gate/up allocation; the fused GEMM would read different "
                "bytes than the per-projection GEMMs")
        block = 128
        if inter % block != 0 or hidden % block != 0:
            raise RuntimeError(
                f"Layer {self.layer_idx}: fused gate/up needs I={inter} and K={hidden} "
                f"to be multiples of the {block}-wide FP8 scale block")
        gate_scale = self.weight_dequant_scale.get("gate_proj.weight_scale_inv")
        up_scale = self.weight_dequant_scale.get("up_proj.weight_scale_inv")
        want = (inter // block, hidden // block)
        if (gate_scale is None or up_scale is None
                or tuple(gate_scale.shape) != want or tuple(up_scale.shape) != want
                or gate_scale.dtype != torch.float32 or up_scale.dtype != torch.float32):
            raise RuntimeError(
                f"Layer {self.layer_idx}: fused gate/up needs fp32 gate and up block "
                f"scales of shape {want}, got "
                f"{None if gate_scale is None else (tuple(gate_scale.shape), gate_scale.dtype)} / "
                f"{None if up_scale is None else (tuple(up_scale.shape), up_scale.dtype)}")
        if gate_scale.device != gate_up.device or up_scale.device != gate_up.device:
            raise RuntimeError(
                f"Layer {self.layer_idx}: fused gate/up weight and scales must share a device")
        self.cached_gate_up = gate_up
        self.cached_gate_up_scale = torch.cat([gate_scale, up_scale], dim=0).contiguous()

    def load_weights_pinned(self) -> Dict[str, torch.Tensor]:
        """Wait without the single-expert sliding-window eviction policy.

        Grouped prefill owns all 256 keys until its completion event; evicting
        an earlier key while acquiring a later expert invalidates that layer.
        """
        return self.core_engine.get_weights_pinned(self.module_key)

    def _forward_impl(
        self, hidden_states: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """FP8 forward using cached weight tensors directly (no nn.Module delegation).

        With the fused gate/up binding this is the four-launch chain of
        ``_forward_fused_gate_up``; otherwise three ``w8a16_gemm`` calls. ``out``
        (the down projection's destination) is only accepted by the fused chain,
        which is the one that can write into a caller's buffer without a copy.
        """
        if self.cached_gate_up is not None:
            return self._forward_fused_gate_up(hidden_states, out)
        if out is not None:
            raise ValueError(
                f"Layer {self.layer_idx}: out= requires the fused gate/up binding")
        return self._forward_unfused(hidden_states)

    def _forward_unfused(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Per-projection chain: gate, up and down each quantize their own
        input and run their own GEMM (routed single-expert / non-persistent
        experts, and the reference the fused chain is pinned against)."""
        from batchgen.attention.mla.fa3_backend import w8a16_gemm
        gate = w8a16_gemm(
            self.cached_gate, self.weight_dequant_scale.get('gate_proj.weight_scale_inv'), hidden_states
        )
        up = w8a16_gemm(
            self.cached_up, self.weight_dequant_scale.get('up_proj.weight_scale_inv'), hidden_states
        )
        intermediate = F.silu(gate) * up
        return w8a16_gemm(
            self.cached_down, self.weight_dequant_scale.get('down_proj.weight_scale_inv'), intermediate
        )

    def _forward_fused_gate_up(
        self, hidden_states: torch.Tensor, out: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Shared-expert chain in four launches, none of them a layout fix-up:

            quantize(x)            -> x_fp8, MN-major TMA-aligned x scales
            fp8 GEMM [gate; up]    -> gate_up [T, 2I] bf16      (one DeepGEMM)
            SiLU(gate) * up + quantize -> h_fp8, aligned h scales (one Triton)
            fp8 GEMM down          -> out [T, H] bf16           (one DeepGEMM)

        against the unfused chain's quantize/transpose/GEMM x2 for gate and up,
        a SiLU, a multiply, a quantize/transpose/GEMM for down and a staging
        copy. Operand values are the unfused chain's: the x quantization is the
        same kernel (the two unfused gate/up quantizations of x were identical);
        scales are only written in the layout DeepGEMM would otherwise
        transpose them into; the fused GEMM's gate and up columns are the two
        separate products (same K order and block scales); and the fused
        activation/quantize reproduces ``F.silu(g) * u`` in bf16 before the
        same quantization. Bit-identity is asserted on GPU by
        tests/test_glm5_shared_expert_fused_gate_up.py.
        """
        import deep_gemm
        from batchgen_kernels.triton.fp8_quantize import (
            per_token_blocked_quantize_bf16_to_fp8_1d,
            silu_mul_per_token_blocked_quantize_bf16_to_fp8_1d,
        )

        if hidden_states.dtype != torch.bfloat16:
            raise TypeError(
                f"Layer {self.layer_idx}: fused shared expert takes bf16 input, "
                f"got {hidden_states.dtype}")
        x = hidden_states.view(-1, hidden_states.size(-1))
        rows = x.shape[0]
        n_out = self.cached_down.shape[0]
        if out is None:
            out = torch.empty((rows, n_out), dtype=torch.bfloat16, device=x.device)
        elif (tuple(out.shape) != (rows, n_out) or out.dtype != torch.bfloat16
              or not out.is_contiguous() or out.device != x.device):
            raise ValueError(
                f"Layer {self.layer_idx}: shared-expert out must be a contiguous "
                f"bf16 [{rows}, {n_out}] tensor on {x.device}, got "
                f"{tuple(out.shape)}/{out.dtype}/contiguous={out.is_contiguous()}")
        if rows == 0:
            return out if hidden_states.dim() == 2 else out.view(
                *hidden_states.shape[:-1], n_out)

        x_fp8, x_scale = per_token_blocked_quantize_bf16_to_fp8_1d(
            x, scale_tma_aligned=True)
        gate_up = torch.empty(
            (rows, self.cached_gate_up.shape[0]), dtype=torch.bfloat16, device=x.device)
        deep_gemm.fp8_gemm_nt(
            (x_fp8, x_scale),
            (self.cached_gate_up, self.cached_gate_up_scale),
            gate_up,
        )
        h_fp8, h_scale = silu_mul_per_token_blocked_quantize_bf16_to_fp8_1d(
            gate_up, scale_tma_aligned=True)
        deep_gemm.fp8_gemm_nt(
            (h_fp8, h_scale),
            (self.cached_down, self.weight_dequant_scale.get("down_proj.weight_scale_inv")),
            out,
        )
        return out if hidden_states.dim() == 2 else out.view(
            *hidden_states.shape[:-1], n_out)

    def forward(
        self, hidden_states: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if not self.persistent:
            if out is not None:
                raise ValueError(
                    f"Layer {self.layer_idx}: out= requires a persistent expert")
            weights = self.load_weights(self.module_key)
            self.cached_gate = weights["gate_proj.weight"]
            self.cached_up = weights["up_proj.weight"]
            self.cached_down = weights["down_proj.weight"]

        result = self._forward_impl(hidden_states, out)

        if not self.persistent:
            torch.cuda.current_stream().synchronize()
            self.free_weights(self.module_key)
            self.cached_gate = self.cached_up = self.cached_down = None

        return result


class GLM5AttnWrapper(AttnWrapperBase):
    """Attention wrapper with FP8 dequant + DSA for GLM-5.

    Key differences from DeepSeek:
    - kv_a_proj_with_mqa (same naming)
    - DSA indexer integration: prefill populates auxiliary cache,
      decode uses indexer scoring for sparse attention
    - MLA dims: qk_nope=192, v_head=256, q_lora_rank=2048
    """

    # Phase C (audit §A finding #8): GLM-5-specific ClassVars previously
    # lived on `AttnWrapperBase` and leaked into every other model's class
    # scope. They are now owned by this GLM-5 subclass.
    #
    # Dispatch-trace instrumentation (per-step counts of GLM-5 dispatch
    # paths; toggled by `batchgen_debug.glm5_dispatch_trace`).
    glm5_dispatch_trace_enabled: ClassVar[bool] = False
    glm5_dispatch_trace_id: ClassVar[Optional[str]] = None
    glm5_dispatch_trace_context: ClassVar[Optional[Dict[str, Any]]] = None
    glm5_dispatch_counts: ClassVar[Dict[str, int]] = {}
    glm5_dispatch_seen: ClassVar[Set[Tuple[str, str, str, int]]] = set()
    # DSA per-step dispatch hint: count of rows with cache_seqlen <= index_topk.
    # Set once per decode step by the worker so per-layer _forward_decode_dsa
    # branches on it without doing a D2H .sum().item() 78 times per step.
    _dsa_short_count: ClassVar[Optional[int]] = None
    # DSA top-k index REUSE (GLM-5.2): carried top-k indices from the most recent
    # FULL layer, reused by subsequent shared layers. Reset per decode step by the
    # worker. For GLM-5 (all layers full) this is never read by a shared branch.
    _dsa_prev_topk_indices: ClassVar[Optional[torch.Tensor]] = None
    # GLM-5.2 packed sparse PREFILL carry. `_dsa_prefill_prev_topk_indices` holds
    # the most recent full layer's packed-global top-k buffer: shared layers
    # consume it, the next full layer overwrites it in place. The packed causal
    # ranges are derived once per micro-batch (layer 0) and reused by every full
    # layer. `_dsa_prefill_path_counts` is the per-layer route audit.
    _dsa_prefill_prev_topk_indices: ClassVar[Optional[torch.Tensor]] = None
    _dsa_prefill_causal_starts: ClassVar[Optional[torch.Tensor]] = None
    _dsa_prefill_causal_ends: ClassVar[Optional[torch.Tensor]] = None
    # Prefix-pool chunk addressing (sparse_prefill.Glm52PoolPrefill) or None.
    _dsa_prefill_pool: ClassVar[Optional[Any]] = None
    _dsa_prefill_path_counts: ClassVar[Optional[Dict[str, Dict[int, int]]]] = None
    # Whole-model CUDA graph can pad local rows to a global NCCL bucket. These
    # graph-owned overrides let GLM-5 DSA use explicit slot sentinels for padded
    # rows instead of deriving slot count from cur_batch.
    glm5_decode_primary_slot_indices: ClassVar[Optional[torch.Tensor]] = None
    glm5_decode_aux_slot_indices: ClassVar[Optional[torch.Tensor]] = None
    glm5_dsa_graph_forward_state: ClassVar[Optional[Dict[str, Any]]] = None
    glm5_dsa_flashmla_graph_metadata: ClassVar[Optional[Dict[str, Any]]] = None

    def __init__(
        self,
        module: nn.Module,
        layer_idx: int,
        core_engine,
        engine_config,
        model_config,
        persistent: bool = True,
        weight_dequant_scale: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__(
            module, layer_idx, core_engine, engine_config, model_config,
            persistent, weight_dequant_scale
        )
        # FP8 weight caching for GLM-5 MLA
        self.fp8_q_a_proj = None
        self.fp8_q_b_proj = None
        self.fp8_kv_a_proj = None
        self.fp8_kv_b_proj = None
        self.fp8_o_proj = None
        self._fp8_qkv_a_proj = None
        self._fp8_qkv_a_scale = None
        # WP5: FP8 absorb weights (pre-quantized once at init). The retired
        # BF16 BMM absorb cache (_cached_q_absorb / w_kc / w_vc) is gone:
        # every decode consumer uses these FP8 weights and fails loud when
        # they are missing.
        self._fp8_absorb_weights = None
        # WP2: Fused indexer KV proj (CUDA WGMMA)
        self._indexer_cuda_module = None
        self._indexer_cuda_weights = None
        # WP4: Fused scoring pipeline
        self._fused_wqb_weights = None
        self._fused_score_module = None
        # Merged [q_b | wq_b] decode GEMM operand (GLM5_QB_WQB_MERGE). Non-None
        # only on a registered persistent indexer layer with the switch on; the
        # WP4 container above stays built either way (prefill and the
        # separate version read it), so the old version stays selectable.
        self._fp8_q_b_wq_b_proj = None
        self._fp8_q_b_wq_b_scale = None
        # Fallback warning guards
        self._warned_fp8_absorb_fallback = False
        self._warned_indexer_kv_fallback = False
        self._dsa_cuda_graph_manager = None
        self._dsa_cuda_graph_segment_name = None
        self._dsa_cuda_graph_max_seqlen = 0
        self._dsa_cuda_graph_max_primary_pages = 0
        self._dsa_cuda_graph_max_aux_pages = 0
        self._dsa_cuda_graph_primary_page_table_signature = None
        self._dsa_cuda_graph_aux_page_table_signature = None
        self._dsa_cuda_graph_required = False
        self._dsa_cuda_graph_full = False

    def _register_fp8_weights(self):
        """Cache FP8 attention weights. GLM-5 uses kv_a_proj_with_mqa."""
        q_a_weight = self.module.q_a_proj.weight.data
        kv_a_weight = self.module.kv_a_proj_with_mqa.weight.data
        q_a_scale = self.weight_dequant_scale.get("q_a_proj.weight_scale_inv")
        kv_a_scale = self.weight_dequant_scale.get(
            "kv_a_proj_with_mqa.weight_scale_inv"
        )
        if q_a_scale is None or kv_a_scale is None:
            raise RuntimeError(
                f"[layer {self.layer_idx}] fused Q-A/KV-A requires both FP8 scales"
            )
        self._fp8_qkv_a_proj = torch.cat(
            (q_a_weight, kv_a_weight),
            dim=0,
        ).contiguous()
        self._fp8_qkv_a_scale = torch.cat(
            (q_a_scale, kv_a_scale),
            dim=0,
        ).contiguous()
        q_rows = q_a_weight.shape[0]
        q_scale_rows = q_a_scale.shape[0]
        self.module.q_a_proj.weight.data = self._fp8_qkv_a_proj[:q_rows]
        self.module.kv_a_proj_with_mqa.weight.data = self._fp8_qkv_a_proj[q_rows:]
        self.weight_dequant_scale["q_a_proj.weight_scale_inv"] = (
            self._fp8_qkv_a_scale[:q_scale_rows]
        )
        self.weight_dequant_scale["kv_a_proj_with_mqa.weight_scale_inv"] = (
            self._fp8_qkv_a_scale[q_scale_rows:]
        )
        self.fp8_q_a_proj = self.module.q_a_proj.weight.data
        self.fp8_q_b_proj = self.module.q_b_proj.weight.data
        self.fp8_kv_a_proj = self.module.kv_a_proj_with_mqa.weight.data
        self.fp8_kv_b_proj = self.module.kv_b_proj.weight.data
        self.fp8_o_proj = self.module.o_proj.weight.data

    def _unregister_fp8_weights(self):
        self.fp8_q_a_proj = None
        self.fp8_q_b_proj = None
        self.fp8_kv_a_proj = None
        self.fp8_kv_b_proj = None
        self.fp8_o_proj = None
        self._fp8_qkv_a_proj = None
        self._fp8_qkv_a_scale = None
        self._fp8_q_b_wq_b_proj = None
        self._fp8_q_b_wq_b_scale = None

    def initialize_decode_absorb(self):
        """Pre-compute absorbed projections from FP8 kv_b_proj weight.

        Dequantizes the static FP8 weight once and caches the BF16
        q_absorb and out_absorb matrices. Eliminates 78× per-step
        dequantization in _forward_decode_dsa().
        """
        from batchgen.attention.mla.flashmla_backend import deepseek_v3_dequantization
        attn = self.module
        weight_scale = self.weight_dequant_scale
        if weight_scale is None or "kv_b_proj.weight_scale_inv" not in weight_scale:
            return
        kv_b_proj = deepseek_v3_dequantization(
            attn.kv_b_proj.weight.data,
            weight_scale["kv_b_proj.weight_scale_inv"],
        ).view(attn.num_heads, -1, attn.kv_lora_rank)
        # Slice 2c-2: these BF16 matrices are now the ONES WE KEEP. They used
        # to be quantizer input, freed immediately; the decode absorb is now a
        # bf16 strided-batched bmm, which is both faster than the FP8 WGMMA
        # kernel it replaces and numerically exact against a bf16 reference
        # (the fp8 kernel is not), so it also removes an fp8 error source
        # from the attention critical path.
        #
        # HBM accounting, per rank: bf16 q_absorb [H,192,512] 12.58 MB +
        # out_absorb [H,256,512] 16.78 MB = 29.36 MB/layer kept, against the
        # 14.68 MB/layer of FP8 copies no longer built -- a NET
        # +14.68 MB/layer = +1.15 GB/GPU over 78 layers. (The comment this
        # replaces claimed ~4.6 GB; that figure also counted the dequantised
        # kv_b_proj intermediate, which is still a temporary.) The cost
        # competes with the KV cache and is the slice's one open trade: it is
        # settled by the paired cold M1 plus the HBM/KV-pool readings and must
        # be reported with them.
        cached_q_absorb = kv_b_proj[:, :attn.qk_nope_head_dim, :].contiguous()
        cached_out_absorb = kv_b_proj[:, attn.qk_nope_head_dim:, :].contiguous()

        # The container decides the decode kernel: BF16AbsorbWeights -> bmm,
        # FP8AbsorbWeights -> the Triton WGMMA path (kept in-tree, NOT
        # deleted). Building bf16 means the Triton kernel is no longer reached
        # from decode, so its build is no longer a hard requirement and the
        # _HAS_FP8_ABSORB gate that used to gate this block is gone.
        #
        # The attribute keeps its historical `_fp8_absorb_weights` name ON
        # PURPOSE: the core scaffolding reads it by that name
        # (batchgen_worker.py checks it for None and forwards it to the graph
        # segments), so renaming it would turn this model-class change into a
        # core-class one. The name no longer implies the fp8 container.
        try:
            self._fp8_absorb_weights = BF16AbsorbWeights(
                cached_q_absorb,    # [H, 192, 512] kept as [H, K, N]
                cached_out_absorb,  # [H, 256, 512] flipped to [H, 512, 256]
            )
        except Exception as e:
            raise RuntimeError(
                f"[layer {self.layer_idx}] BF16 absorb init failed: {e}"
            ) from e
        logging.debug(
            f"[layer {self.layer_idx}] BF16 absorb weights initialized "
            f"({self._fp8_absorb_weights.nbytes() / 1e6:.2f} MB)"
        )

        # WP2/WP4 init moved to initialize_fused_kernels() — must run after set_device

    def initialize_fused_kernels(self):
        """Initialize TMA-based CUDA kernels (WP2/WP4).

        Must be called AFTER torch.cuda.set_device(local_rank) — TMA descriptors
        contain physical GPU addresses and are not portable across devices —
        AND after _setup_fp8_scales has attached indexer.wk_scale / wq_b_scale.
        """
        attn = self.module
        if not hasattr(attn, "indexer") or attn.indexer is None:
            return
        indexer = attn.indexer

        # Load-bearing ordering contract: WP2/WP4 both need the FP8 dequant scales
        # attached by PSM._setup_fp8_scales. If this fires, somebody moved
        # _init_fused_kernels back in front of _setup_fp8_scales (see commit
        # d3b99222).
        assert hasattr(indexer, 'wk_scale'), (
            f"[layer {self.layer_idx}] initialize_fused_kernels called before "
            "_setup_fp8_scales attached indexer.wk_scale"
        )

        # WP2: Fused indexer KV proj (GEMM-only path, LayerNorm stays in PyTorch)
        if _HAS_FUSED_INDEXER_KV:
            try:
                self._indexer_cuda_module = _build_indexer_module()
                # Dequantize FP8 wk weight to BF16 (kernel re-quantizes internally for TMA)
                wk_bf16 = glm5_fp8_dequantization(
                    indexer.wk.weight.data, indexer.wk_scale,
                )
                self._indexer_cuda_weights = FP8IndexerWeightsCUDA(
                    wk_bf16, self._indexer_cuda_module,
                )
                logging.debug(
                    f"[layer {self.layer_idx}] WP2 fused indexer KV proj initialized"
                )
            except Exception as e:
                logging.warning(
                    f"[layer {self.layer_idx}] WP2 init failed: {e}"
                )
                self._indexer_cuda_module = None
                self._indexer_cuda_weights = None

        # WP4: Fused scoring pipeline (CUDA WGMMA wq_b + RoPE + Hadamard + scoring + topk)
        if _HAS_FUSED_SCORE and self._indexer_cuda_module is not None and hasattr(indexer, 'wq_b_scale'):
            try:
                wq_b_bf16 = glm5_fp8_dequantization(
                    indexer.wq_b.weight.data, indexer.wq_b_scale,
                )
                self._fused_wqb_weights = FP8WqbWeightsCUDA(
                    wq_b_bf16, self._indexer_cuda_module,
                )
                # Attach to indexer so score_and_select_paged can use it
                indexer._fused_score_weights = self._fused_wqb_weights
                indexer._fused_score_module = self._indexer_cuda_module
                indexer._warned_fused_score_fallback = False
                logging.debug(
                    f"[layer {self.layer_idx}] WP4 fused scoring pipeline initialized"
                )
            except Exception as e:
                logging.warning(
                    f"[layer {self.layer_idx}] WP4 init failed: {e}"
                )
                self._fused_wqb_weights = None

        # Merged [q_b | wq_b] decode GEMM. Built here, and only on layers that
        # reach this line -- the ones WITH an indexer -- because wq_b exists
        # only there and indexer.wq_b_scale is attached by _setup_fp8_scales
        # before this runs (the ordering contract asserted above). No
        # try/except: with the switch on the decode graph cannot run without
        # this operand, and a warn-and-None would be a silent fallback to the
        # other version's numerics.
        if glm5_qb_wqb_merge_enabled():
            self._register_q_b_wq_b_merged(indexer)

    def _register_q_b_wq_b_merged(self, indexer) -> None:
        """Concatenate ``[q_b_proj | indexer.wq_b]`` into one DeepGEMM operand.

        Both weights are the checkpoint's own e4m3 bytes with its 128x128
        block scales, so the merged GEMM's q_b slice reads exactly the bytes
        the separate q_b GEMM read, and the wq_b slice replaces the WP4
        container's dequantize -> re-quantize round trip.

        Aliasing: ``q_b_proj`` is a phase-managed weight, so it is re-pointed
        at its slice of the merged buffer (no second resident copy), with its
        scale entry re-pointed the same way -- the ``_register_fp8_weights``
        discipline. ``indexer.wq_b`` is NOT re-pointed: indexer parameters are
        bound to the shared device skeleton that the prefill and decode
        instances both alias, and rebinding one would trip
        ``_assert_skeleton_still_shared``. The merged buffer therefore holds a
        COPY of the wq_b rows (index_n_heads * index_head_dim x q_lora_rank
        e4m3 per indexer layer); the skeleton tensor stays the authority for
        every other reader.
        """
        if self.fp8_q_b_proj is None:
            # No registered resident q_b (a streaming prefill wrapper binds
            # its weights per forward): nothing to merge, and nothing on this
            # wrapper reads the merged operand. Every decode consumer fails
            # loud if it is absent where it is required.
            self._fp8_q_b_wq_b_proj = None
            self._fp8_q_b_wq_b_scale = None
            return
        wq_b_scale = getattr(indexer, "wq_b_scale", None)
        if wq_b_scale is None:
            raise RuntimeError(
                f"[layer {self.layer_idx}] GLM5_QB_WQB_MERGE needs "
                "indexer.wq_b_scale (the checkpoint's wq_b weight_scale_inv), "
                "but _setup_fp8_scales did not attach it"
            )
        q_b_scale = self.weight_dequant_scale.get("q_b_proj.weight_scale_inv")
        if q_b_scale is None:
            raise RuntimeError(
                f"[layer {self.layer_idx}] GLM5_QB_WQB_MERGE needs the "
                "q_b_proj.weight_scale_inv FP8 scale"
            )
        merged = build_q_b_wq_b_merged(
            self.module.q_b_proj.weight.data,
            q_b_scale,
            indexer.wq_b.weight.data,
            wq_b_scale,
        )
        self._fp8_q_b_wq_b_proj = merged.weight
        self._fp8_q_b_wq_b_scale = merged.scale
        self.module.q_b_proj.weight.data = merged.weight[: merged.q_b_rows]
        self.weight_dequant_scale["q_b_proj.weight_scale_inv"] = (
            merged.scale[: merged.q_b_scale_rows]
        )
        self.fp8_q_b_proj = self.module.q_b_proj.weight.data
        logging.debug(
            f"[layer {self.layer_idx}] merged [q_b | wq_b] operand "
            f"{tuple(merged.weight.shape)} / scale {tuple(merged.scale.shape)}"
        )

    def dequantize_weights(
        self, weights_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Return FP8 weights unchanged — deepgemm handles FP8 directly."""
        return weights_dict

    def _clear_nonpersistent_weight_bindings(self) -> None:
        """Drop buffer-backed parameters without draining the CUDA stream."""
        applied = getattr(self, "_applied_param_keys", None)
        for name, param in self.module.named_parameters():
            if applied is not None and name not in applied:
                continue
            param.data = torch.empty(0, device=param.data.device)
        self._applied_param_keys = None

    def forward(self, *args, **kwargs) -> torch.Tensor:
        if self.persistent or self.phase != "prefill":
            return super().forward(*args, **kwargs)

        rank = self.get_rank_safe()
        logging.debug(
            f"[Rank {rank} Layer {self.layer_idx}] "
            f"Attn forward. Phase: {self.phase}"
        )

        weights = self.load_weights(self.module_key)
        self.apply_weights(self.dequantize_weights(weights))
        hidden_states = kwargs.pop("hidden_states", None)
        result = self._forward_prefill(hidden_states, **kwargs)

        # The core records an event on this thread's active CUDA stream and
        # keeps the backing slot unavailable until all queued consumers finish.
        # Python bindings can therefore be cleared without a host stream drain.
        self._clear_nonpersistent_weight_bindings()
        self.core_engine.free_weights_buffer_async(self.module_key)

        logging.debug(
            f"[Rank {rank} Layer {self.layer_idx}] "
            f"Attn forward complete. Phase: {self.phase}"
        )
        return result

    def enable_dsa_cuda_graph(
        self,
        manager,
        segment_name: str,
        *,
        max_seqlen: int,
        max_primary_pages_per_seq: Optional[int] = None,
        max_aux_pages_per_seq: Optional[int] = None,
        primary_page_table: Optional[torch.Tensor] = None,
        aux_page_table: Optional[torch.Tensor] = None,
        graph_output_required: bool = False,
        full_segment: bool = False,
    ) -> None:
        self._dsa_cuda_graph_manager = manager
        self._dsa_cuda_graph_segment_name = segment_name
        self._dsa_cuda_graph_max_seqlen = max_seqlen
        self._dsa_cuda_graph_required = graph_output_required
        self._dsa_cuda_graph_full = full_segment
        self._dsa_cuda_graph_max_primary_pages = (
            primary_page_table.shape[1]
            if primary_page_table is not None
            else int(max_primary_pages_per_seq or 0)
        )
        self._dsa_cuda_graph_max_aux_pages = (
            aux_page_table.shape[1]
            if aux_page_table is not None
            else int(max_aux_pages_per_seq or 0)
        )
        self._dsa_cuda_graph_primary_page_table_signature = _glm5_dsa_page_table_signature(
            primary_page_table
        )
        self._dsa_cuda_graph_aux_page_table_signature = _glm5_dsa_page_table_signature(
            aux_page_table
        )

    def _dsa_cuda_graph_page_tables_match(
        self,
        gpu_paged_kv_manager,
        gpu_paged_kv_manager_aux,
    ) -> bool:
        primary_expected = getattr(
            self,
            "_dsa_cuda_graph_primary_page_table_signature",
            None,
        )
        aux_expected = getattr(
            self,
            "_dsa_cuda_graph_aux_page_table_signature",
            None,
        )
        if primary_expected is None or aux_expected is None:
            return False
        return (
            _glm5_dsa_page_table_signature(
                _glm5_dsa_gpu_page_table_tensor(gpu_paged_kv_manager)
            ) == primary_expected
            and _glm5_dsa_page_table_signature(
                _glm5_dsa_gpu_page_table_tensor(gpu_paged_kv_manager_aux)
            ) == aux_expected
        )

    @classmethod
    def _reset_glm52_prefill_path_counts(cls) -> None:
        GLM5AttnWrapper._dsa_prefill_path_counts = {
            "sparse": {},
            "indexer_compute": {},
            "indexer_reuse": {},
        }

    @classmethod
    def _record_glm52_prefill_path(cls, path: str, layer_idx: int) -> None:
        counts = GLM5AttnWrapper._dsa_prefill_path_counts
        if counts is None:
            raise RuntimeError(
                "GLM-5.2 sparse prefill path audit was not started before "
                f"recording layer {layer_idx} path {path!r}"
            )
        per_layer = counts.setdefault(path, {})
        per_layer[layer_idx] = per_layer.get(layer_idx, 0) + 1

    def _finish_glm52_prefill_path_counts(self) -> None:
        """Assert every layer took the sparse route exactly once.

        Full (indexer-carrying) layers must compute top-k once; shared layers
        must reuse it once. A repeated `indexer_compute` means the packed top-k
        was scored twice for one layer, which is silent duplicated work rather
        than a crash — so it is a hard failure here.
        """
        counts = GLM5AttnWrapper._dsa_prefill_path_counts
        GLM5AttnWrapper._dsa_prefill_path_counts = None
        if counts is None:
            raise RuntimeError(
                "GLM-5.2 sparse prefill path audit finished without being started"
            )

        from .dsa_schedule import dsa_layer_skips_topk

        config = self.module.config
        num_layers = config.num_hidden_layers
        shared_layers = {
            layer_idx
            for layer_idx in range(num_layers)
            if dsa_layer_skips_topk(config, layer_idx)
        }
        expected = {
            "sparse": set(range(num_layers)),
            "indexer_compute": set(range(num_layers)) - shared_layers,
            "indexer_reuse": shared_layers,
        }
        problems = []
        for path, expected_layers in expected.items():
            actual = counts.get(path, {})
            missing = sorted(expected_layers - set(actual))
            unexpected = sorted(set(actual) - expected_layers)
            repeated = sorted(idx for idx, n in actual.items() if n != 1)
            if missing or unexpected or repeated:
                problems.append(
                    f"{path}: missing={missing} unexpected={unexpected} "
                    f"repeated={repeated}"
                )
        if problems:
            raise RuntimeError(
                "GLM-5.2 sparse prefill path coverage mismatch: "
                + "; ".join(problems)
            )
        logging.info(
            "[GLM5.2_PREFILL_PATH] mode=sparse layers=%d full=%d shared=%d",
            num_layers,
            num_layers - len(shared_layers),
            len(shared_layers),
        )

    @classmethod
    def _reset_glm52_prefill_carry(cls) -> None:
        GLM5AttnWrapper._dsa_prefill_prev_topk_indices = None
        GLM5AttnWrapper._dsa_prefill_causal_starts = None
        GLM5AttnWrapper._dsa_prefill_causal_ends = None
        GLM5AttnWrapper._dsa_prefill_pool = None

    def _indexer_applies_hadamard(self) -> bool:
        """Whether this run's DSA indexer rotates q and k before fp8 quant.

        Prefill writes the index-K cache decode later scores, so it must use
        decode's rotation. The aux indexer host-KV view carries the same
        page-split fact the GPU aux cache does, and
        ``indexer_applies_hadamard`` turns that one fact into the predicate --
        False for GLM-5.2's page-split FP8 indexer (upstream v3.2 semantics,
        dropped on both q and k), True for a bf16 AoS indexer.
        """
        from batchgen_kernels.attention.dsa.indexer_k_quant import (
            indexer_applies_hadamard,
        )

        aux_view = AttnWrapperBase.host_paged_kv_worker_view_aux
        if aux_view is None:
            raise RuntimeError(
                "GLM-5 DSA prefill requires the auxiliary indexer host KV "
                "view to resolve the indexer Hadamard semantics"
            )
        return indexer_applies_hadamard(
            page_split=bool(getattr(aux_view, "is_page_split", False))
        )

    def _glm52_sparse_prefill_active(self) -> bool:
        from . import sparse_prefill

        config = getattr(self.module, "config", None)
        if AttnWrapperBase.prefill_pool_batch is not None:
            # Pool rows continue pooled prefixes: only the sparse path reads the
            # pool, and it equals dense attention for contexts <= index_topk.
            if getattr(config, "model_type", None) not in {
                "glm_moe_dsa_5_2", "glm_moe_dsa_5_3",
            }:
                raise RuntimeError("the prefill prefix pool needs GLM-5.2 sparse DSA prefill")
            return True
        return sparse_prefill.should_use_glm52_sparse_prefill(
            getattr(config, "model_type", None),
            int(self.prepack_max_seqlen),
            int(getattr(config, "index_topk", 0)),
        )

    def _forward_prefill_glm52_sparse(self, hidden_states_2d: torch.Tensor) -> Tuple:
        """Packed GLM-5.2 DSA prefill: sparse absorbed FlashMLA, never dense FA3.

        Full layers score the packed indexer top-k; shared layers carry the most
        recent full layer's selection. Primary KV is offloaded on every layer and
        indexer KV only on full layers, matching what decode later reads back.
        """
        from . import sparse_prefill

        config = self.module.config
        device = hidden_states_2d.device
        position_ids = self.position_ids.to(device)

        if self.layer_idx == 0:
            # One micro-batch owns one carry: rebuild here so a previous
            # micro-batch (or an aborted one) can never leak its ranges/top-k.
            self._reset_glm52_prefill_carry()
            self._reset_glm52_prefill_path_counts()
            pool_batch = AttnWrapperBase.prefill_pool_batch
            if pool_batch is not None:
                pool_ctx = sparse_prefill.build_glm52_pool_prefill(
                    page_table=pool_batch.page_table,
                    cache_seqlens=pool_batch.cache_seqlens,
                    cu_seqlens_q=pool_batch.cu_seqlens_q,
                    position_ids=position_ids,
                    slots=pool_batch.slots,
                    page_tokens=AttnWrapperBase.prefill_pool.page_tokens,
                    max_cache_seqlen=pool_batch.max_cache_seqlen,
                )
                GLM5AttnWrapper._dsa_prefill_pool = pool_ctx
                starts, ends = pool_ctx.causal_starts, pool_ctx.causal_ends
            else:
                starts, ends = sparse_prefill.build_packed_causal_ranges(
                    self.prepack_cu_seqlens.to(device),
                    position_ids,
                    hidden_states_2d.shape[0],
                    sequence_lengths=list(self.prepack_seq_lengths),
                )
            GLM5AttnWrapper._dsa_prefill_causal_starts = starts
            GLM5AttnWrapper._dsa_prefill_causal_ends = ends

        indexer = self.module.indexer
        carried = GLM5AttnWrapper._dsa_prefill_prev_topk_indices
        pool_ctx = GLM5AttnWrapper._dsa_prefill_pool
        latent_pool = indexer_pool = None
        if pool_ctx is not None:
            prefill_pool = AttnWrapperBase.prefill_pool
            latent_pool = prefill_pool.k[self.layer_idx]
            if indexer is not None:
                indexer_pool = prefill_pool.indexer[
                    prefill_pool.indexer_row[self.layer_idx]
                ]
        result = sparse_prefill.glm52_sparse_prefill_prepacked(
            attn=self.module,
            hidden_states=hidden_states_2d,
            position_ids=position_ids,
            max_seqlen=self.prepack_max_seqlen,
            weight_scale=self.weight_dequant_scale,
            indexer=indexer,
            indexer_applies_hadamard=self._indexer_applies_hadamard(),
            carried_topk_indices=None if indexer is not None else carried,
            reusable_topk_indices=carried if indexer is not None else None,
            causal_starts=GLM5AttnWrapper._dsa_prefill_causal_starts,
            causal_ends=GLM5AttnWrapper._dsa_prefill_causal_ends,
            pool=pool_ctx,
            latent_pool=latent_pool,
            indexer_pool=indexer_pool,
        )
        GLM5AttnWrapper._dsa_prefill_prev_topk_indices = result.topk_indices
        self._record_glm52_prefill_path("sparse", self.layer_idx)
        self._record_glm52_prefill_path(
            "indexer_compute" if indexer is not None else "indexer_reuse",
            self.layer_idx,
        )

        if indexer is not None and result.indexer_kv is None:
            raise RuntimeError(
                "GLM-5.2 full sparse-prefill layer returned no indexer KV; "
                "refusing primary-only host offload"
            )
        if pool_ctx is not None:
            self._offload_prefill_pool_rows(
                result.primary_kv,
                None if indexer is None else result.indexer_kv[0, :, 0, :],
            )
        else:
            if indexer is not None:
                self._offload_prepacked_indexer_kv(result.indexer_kv.squeeze(0))
            self._offload_prepacked_kv(result.primary_kv)

        if self.layer_idx == config.num_hidden_layers - 1:
            self._finish_glm52_prefill_path_counts()
            self._reset_glm52_prefill_carry()
        return (result.attn_output.unsqueeze(0), None, None)

    def _forward_prefill(self, hidden_states: torch.Tensor, **kwargs) -> Tuple:
        """Prefill forward with DSA auxiliary cache population.

        1. Standard MLA prefill via FA3 (full attention), or packed sparse
           absorbed FlashMLA for long GLM-5.2 prompts
        2. Compute indexer K and write to auxiliary cache
        """
        AttnWrapperBase.retire_pending_prefill_offloads_before_layer(
            self.layer_idx,
            device=hidden_states.device,
        )
        if self.prepack_mode:
            hidden_states_2d = hidden_states.squeeze(0)
            if self._glm52_sparse_prefill_active():
                return self._forward_prefill_glm52_sparse(hidden_states_2d)
            attn_output, offload_kv = self.module.prefill_attn_w8a16_prepacked(
                hidden_states_2d,
                self.position_ids.to(hidden_states_2d.device),
                self.prepack_cu_seqlens.to(hidden_states_2d.device),
                self.prepack_max_seqlen,
                self.prepack_num_sequences,
                self.weight_dequant_scale
            )

            # DSA: compute indexer K and offload to auxiliary host cache.
            # This path MUST run for every prompt token during prefill — otherwise
            # aux cache is unpopulated and any later decode past 2048 tokens reads
            # unwritten memory. `_offload_prepacked_indexer_kv` fails fast if
            # host_paged_kv_worker_view_aux is missing.
            if not hasattr(self.module, 'indexer'):
                raise RuntimeError(
                    "GLM-5 DSA prefill requires indexer KV; refusing primary-only host offload"
                )
            # Shared layers (GLM-5.2) have indexer is None: they reuse a full
            # layer's top-k at decode and never read an aux indexer cache, so
            # skip the indexer-K compute + offload entirely. GLM-5 layers always
            # have a real indexer so this block always runs there.
            if self.module.indexer is not None:
                indexer_kv = self.module.indexer.compute_indexer_kv(
                    hidden_states_2d.unsqueeze(0),
                    positions=self.position_ids.to(hidden_states_2d.device),
                    apply_hadamard=self._indexer_applies_hadamard(),
                )
                if indexer_kv is None:
                    raise RuntimeError(
                        "GLM-5 DSA prefill indexer returned no KV; refusing primary-only host offload"
                    )
                self._offload_prepacked_indexer_kv(indexer_kv.squeeze(0))

            self._offload_prepacked_kv(offload_kv)
            attn_output = attn_output.unsqueeze(0)
            return (attn_output, None, None)
        else:
            raise RuntimeError(
                "GLM-5 DSA prefill requires prepack_mode so primary and "
                "auxiliary/indexer KV are offloaded with identical sequence "
                "boundaries"
            )

    def _offload_prefill_pool_rows(
        self, primary_kv: torch.Tensor, indexer_kv: Optional[torch.Tensor]
    ) -> None:
        """Offload each prefix-pool row from its own activations, at its start.

        A pooled segment row goes to its segment's pseudo sequence at 0, a tail
        row to its prompt's pages at the tail start. Primary latent on every
        layer; indexer keys only on full layers (``indexer_kv`` given).
        """
        view = self.core_engine.host_paged_kv_worker_view
        primary_view = getattr(view, "primary", view)
        aux_view = AttnWrapperBase.host_paged_kv_worker_view_aux
        if indexer_kv is not None and aux_view is None:
            raise RuntimeError(
                "GLM-5 DSA auxiliary host KV worker view is required for "
                "indexer KV offload"
            )
        bounds = AttnWrapperBase.prefill_pool_batch.row_bounds
        starts = AttnWrapperBase.prefill_pool_row_starts
        sequence_ids = self.cur_batch
        if len(bounds) - 1 != len(sequence_ids) or len(starts) != len(sequence_ids):
            raise RuntimeError("GLM-5 prefix-pool row metadata size mismatch")
        targets = [(primary_view, primary_kv, False)]
        if indexer_kv is not None:
            # A page-split aux region stores e4m3 keys plus one fp32 scale per
            # token, so the host write quantizes here, at the offload
            # boundary: the pool's own rows stay bf16 for this layer's
            # prefill scoring.
            targets.append(
                (aux_view, indexer_kv, bool(getattr(aux_view, "is_page_split", False)))
            )
        for row, (start, end) in enumerate(zip(bounds, bounds[1:])):
            for worker_view, kv, quantize in targets:
                row_kv = kv[start:end].view(1, end - start, 1, kv.shape[-1]).contiguous()
                scale_tensor = None
                if quantize:
                    row_kv, row_scales = _quantize_indexer_k_pagesplit(row_kv)
                    scale_tensor = row_scales
                task = worker_view.async_offload_layer_kv_range_to_host(
                    layer_idx=self.layer_idx,
                    sequence_ids=[sequence_ids[row]],
                    k_tensor=row_kv,
                    v_tensor=None,
                    raw_start_positions=[starts[row]],
                    token_counts=[end - start],
                    scale_tensor=scale_tensor,
                )
                AttnWrapperBase.pin_prefill_offload_tensor(row_kv, self.layer_idx)
                if scale_tensor is not None:
                    AttnWrapperBase.pin_prefill_offload_tensor(
                        scale_tensor, self.layer_idx
                    )
                AttnWrapperBase.track_prefill_offload_task(task, self.layer_idx)

    def _offload_prepacked_indexer_kv(self, offload_kv: torch.Tensor):
        """Offload packed indexer KV with one asynchronous task per layer."""
        if AttnWrapperBase.host_paged_kv_worker_view_aux is None:
            raise RuntimeError(
                "GLM-5 DSA auxiliary host KV worker view is required for "
                "indexer KV offload"
            )
        sequence_lengths = list(self.prepack_seq_lengths)
        global_sequence_ids = self.cur_batch
        if len(sequence_lengths) != len(global_sequence_ids):
            raise RuntimeError("GLM-5 packed indexer KV metadata size mismatch")
        if sum(sequence_lengths) != offload_kv.shape[0]:
            raise RuntimeError("GLM-5 packed indexer KV token count mismatch")

        aux_view = AttnWrapperBase.host_paged_kv_worker_view_aux
        packed_kv = offload_kv.contiguous()
        scale_tensor = None
        if getattr(aux_view, "is_page_split", False):
            packed_kv, scale_tensor = _quantize_indexer_k_pagesplit(packed_kv)
        task = aux_view.async_offload_packed_layer_kv_to_host(
            layer_idx=self.layer_idx,
            sequence_ids=global_sequence_ids,
            k_tensor=packed_kv,
            v_tensor=None,
            sequence_lengths=sequence_lengths,
            scale_tensor=scale_tensor,
        )
        AttnWrapperBase.pin_prefill_offload_tensor(packed_kv, self.layer_idx)
        if scale_tensor is not None:
            AttnWrapperBase.pin_prefill_offload_tensor(scale_tensor, self.layer_idx)
        AttnWrapperBase.track_prefill_offload_task(task, self.layer_idx)

    def _offload_prepacked_kv(self, offload_kv: torch.Tensor):
        """Offload packed primary KV with one asynchronous task per layer."""
        sequence_lengths = list(self.prepack_seq_lengths)
        global_sequence_ids = self.cur_batch
        if len(sequence_lengths) != len(global_sequence_ids):
            raise RuntimeError("GLM-5 packed primary KV metadata size mismatch")
        if sum(sequence_lengths) != offload_kv.shape[0]:
            raise RuntimeError("GLM-5 packed primary KV token count mismatch")

        packed_kv = offload_kv.unsqueeze(1).contiguous()
        task = self.core_engine.host_paged_kv_worker_view.async_offload_packed_layer_kv_to_host(
            layer_idx=self.layer_idx,
            sequence_ids=global_sequence_ids,
            k_tensor=packed_kv,
            v_tensor=None,
            sequence_lengths=sequence_lengths,
        )
        AttnWrapperBase.pin_prefill_offload_tensor(packed_kv, self.layer_idx)
        AttnWrapperBase.track_prefill_offload_task(task, self.layer_idx)

    def _forward_decode(self, hidden_states: torch.Tensor, **kwargs) -> Tuple:
        """Decode forward with DSA sparse attention.

        For DSA models:
        1. Compute MLA compressed KV, write to primary cache
        2. Compute indexer K for new token, write to auxiliary cache
        3. Score all cached tokens, select top-K
        4. Gather MLA KV at top-K positions from primary cache
        5. Compute absorbed Q, run sparse FlashMLA
        6. out_absorb → o_proj

        Falls back to standard full-cache FlashMLA when DSA is not active.
        """
        past_key_states = AttnWrapperBase.past_key_states
        position_ids = AttnWrapperBase.position_ids
        cache_seqlens = AttnWrapperBase.cache_seqlens
        max_seqlen = AttnWrapperBase.max_seqlen
        gpu_paged_kv_manager = AttnWrapperBase.gpu_paged_kv_manager
        gpu_paged_kv_manager_aux = AttnWrapperBase.gpu_paged_kv_manager_aux

        if gpu_paged_kv_manager is not None:
            dsa_active = (
                gpu_paged_kv_manager_aux is not None
                and hasattr(self.module, 'indexer')
            )

            if dsa_active:
                attn_output = self._forward_decode_dsa(
                    hidden_states, position_ids, cache_seqlens, max_seqlen,
                    gpu_paged_kv_manager, gpu_paged_kv_manager_aux,
                )
                return (attn_output, None, None)

            # Dense-MLA path (use_dense_mla=True, indexer not constructed).
            # Uses the shared eager MLA page-KV backend already used by Kimi.
            attn_output, k_tensor = self.module.decoding_attn_mode_3_bf16(
                hidden_states,
                position_ids,
                cache_seqlens,
                max_seqlen,
                self.weight_dequant_scale,
                gpu_paged_kv_manager,
                self.layer_idx,
                kwargs.get("batch_slice"),
            )
            if AttnWrapperBase.kv_append_callback is not None:
                AttnWrapperBase.kv_append_callback(self.layer_idx, k_tensor, None)
            return (attn_output, None, None)
        else:
            # FP8 KV cache with tensor references
            attention_mask = AttnWrapperBase.attention_mask
            scale = AttnWrapperBase.scale
            layer_past_key = past_key_states[self.layer_idx] if past_key_states else None
            layer_scale = scale[self.layer_idx] if scale else None
            attn_output, updated_past_key, updated_scale = self.module.decoding_attn_mode_3_fp8(
                hidden_states,
                layer_past_key,
                None,
                attention_mask,
                position_ids,
                layer_scale,
                cache_seqlens,
                max_seqlen,
                self.weight_dequant_scale
            )
            return (attn_output, updated_past_key, updated_scale)

    def _forward_decode_dsa_graph(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_seqlen: int,
        gpu_paged_kv_manager,
        gpu_paged_kv_manager_aux,
    ) -> torch.Tensor:
        if self._dsa_cuda_graph_manager is None or self._dsa_cuda_graph_segment_name is None:
            raise RuntimeError(
                f"[layer {self.layer_idx}] GLM-5 DSA CUDA graph was requested, "
                "but this attention wrapper has no registered DSA CUDA graph segment"
            )
        index_topk = getattr(getattr(self.module, "indexer", None), "index_topk", 2048)
        if not _glm5_dsa_cuda_graph_can_replay(
            cache_seqlens,
            max_seqlen,
            index_topk,
            captured_max_seqlen=getattr(self, "_dsa_cuda_graph_max_seqlen", None),
        ):
            raise RuntimeError(
                f"[layer {self.layer_idx}] GLM-5 DSA CUDA graph replay is not "
                f"valid for max_seqlen={max_seqlen}, index_topk={index_topk}"
            )
        if not self._dsa_cuda_graph_page_tables_match(
            gpu_paged_kv_manager,
            gpu_paged_kv_manager_aux,
        ):
            raise RuntimeError(
                f"[layer {self.layer_idx}] GLM-5 DSA CUDA graph captured page-table "
                "storage no longer matches the active GPU page-table storage"
            )

        bsz = hidden_states.shape[0]
        flashmla_tile_scheduler_metadata, flashmla_num_splits = (
            self._dsa_cuda_graph_flashmla_metadata_inputs(bsz)
        )
        if getattr(self, "_dsa_cuda_graph_full", False):
            primary_slot_indices = getattr(
                AttnWrapperBase,
                "glm5_decode_primary_slot_indices",
                None,
            )
            aux_slot_indices = getattr(
                AttnWrapperBase,
                "glm5_decode_aux_slot_indices",
                None,
            )
            if primary_slot_indices is None or aux_slot_indices is None:
                raise RuntimeError(
                    f"[layer {self.layer_idx}] GLM-5 full DSA CUDA graph requires "
                    "per-forward primary/aux slot tensors to be prepared"
                )
            graph_outputs = self._dsa_cuda_graph_manager.replay(
                self._dsa_cuda_graph_segment_name,
                bsz,
                hidden_states=hidden_states,
                position_ids=position_ids,
                cache_seqlens=cache_seqlens.to(dtype=torch.int32, device=hidden_states.device),
                primary_slot_indices=primary_slot_indices[:bsz].to(
                    dtype=torch.int32,
                    device=hidden_states.device,
                ),
                aux_slot_indices=aux_slot_indices[:bsz].to(
                    dtype=torch.int32,
                    device=hidden_states.device,
                ),
                flashmla_tile_scheduler_metadata=flashmla_tile_scheduler_metadata,
                flashmla_num_splits=flashmla_num_splits,
            )
            if AttnWrapperBase.kv_append_callback is not None:
                AttnWrapperBase.kv_append_callback(
                    self.layer_idx,
                    graph_outputs["primary_k_tensor"],
                    None,
                )
            if AttnWrapperBase.kv_append_callback_aux is not None:
                # `indexer_k_scale` is present only when the aux cache is
                # page-split FP8; the host region requires the scales there and
                # rejects them for a bf16 AoS region.
                AttnWrapperBase.kv_append_callback_aux(
                    self.layer_idx,
                    graph_outputs["indexer_k_tensor"],
                    None,
                    graph_outputs.get("indexer_k_scale"),
                )
            return graph_outputs["attn_output"]

        from batchgen.attention.dsa.glm5_decode_selector import (
            build_glm5_dsa_graph_segment_inputs,
        )
        graph_inputs = build_glm5_dsa_graph_segment_inputs(
            self,
            hidden_states,
            position_ids,
            cache_seqlens,
            max_seqlen,
            gpu_paged_kv_manager,
            gpu_paged_kv_manager_aux,
        )
        if AttnWrapperBase.kv_append_callback is not None:
            AttnWrapperBase.kv_append_callback(
                self.layer_idx,
                graph_inputs.primary_k_tensor,
                None,
            )
        if AttnWrapperBase.kv_append_callback_aux is not None:
            AttnWrapperBase.kv_append_callback_aux(
                self.layer_idx,
                graph_inputs.indexer_k_tensor,
                None,
                graph_inputs.indexer_k_scales,
            )

        graph_outputs = self._dsa_cuda_graph_manager.replay(
            self._dsa_cuda_graph_segment_name,
            bsz,
            q_a=graph_inputs.q_a,
            q_nope=graph_inputs.q_nope,
            q_rope=graph_inputs.q_rope,
            head_gates=graph_inputs.head_gates,
            cache_seqlens=graph_inputs.cache_seqlens,
            positions_expanded=graph_inputs.positions_expanded,
            primary_slot_indices=graph_inputs.primary_slot_indices,
            aux_slot_indices=graph_inputs.aux_slot_indices,
            flashmla_tile_scheduler_metadata=flashmla_tile_scheduler_metadata,
            flashmla_num_splits=flashmla_num_splits,
        )

        return self._project_dsa_attn_heads(graph_outputs["attn_heads"])

    def _project_dsa_attn_heads(self, attn_heads: torch.Tensor) -> torch.Tensor:
        from batchgen.attention.mla.fa3_backend import act_quant
        from batchgen.gemm.w8a8_deepgemm import w8a8_deepgemm

        attn = self.module
        bsz = attn_heads.shape[0]
        attn_heads_flat = attn_heads.reshape(
            bsz,
            attn.num_heads * attn.v_head_dim,
        )
        attn_output_fp8, attn_output_scale = act_quant(attn_heads_flat)
        attn_output = w8a8_deepgemm(
            attn_output_fp8,
            attn_output_scale,
            attn.o_proj.weight,
            self.weight_dequant_scale["o_proj.weight_scale_inv"],
        )
        return attn_output.view(bsz, 1, -1)

    def _dsa_cuda_graph_forward_state_allows_replay(
        self,
        batch_size: int,
    ) -> tuple[bool, str]:
        if self._dsa_cuda_graph_manager is None:
            return False, "no graph manager"
        bucket_size = self._dsa_cuda_graph_manager.bucketing.get_padded_size(batch_size)
        state = getattr(GLM5AttnWrapper, "glm5_dsa_graph_forward_state", None)
        if not isinstance(state, dict):
            return False, "missing per-forward graph state"
        if state.get("path") != "graph":
            reason = state.get("reason", "unknown")
            return False, f"worker selected eager DSA ({reason})"
        if int(state.get("bucket", -1)) != int(bucket_size):
            return (
                False,
                f"worker graph bucket {state.get('bucket')} does not match replay bucket {bucket_size}",
            )
        if not bool(state.get("metadata_prepared", False)):
            return False, "per-forward FlashMLA metadata was not prepared"
        metadata = getattr(GLM5AttnWrapper, "glm5_dsa_flashmla_graph_metadata", None)
        if not isinstance(metadata, dict):
            return False, "missing per-forward FlashMLA metadata"
        if int(metadata.get("bucket_size", -1)) != int(bucket_size):
            return (
                False,
                f"FlashMLA metadata bucket {metadata.get('bucket_size')} does not match replay bucket {bucket_size}",
            )
        tile_scheduler_metadata = metadata.get("tile_scheduler_metadata")
        num_splits = metadata.get("num_splits")
        if not isinstance(tile_scheduler_metadata, torch.Tensor) or not isinstance(
            num_splits,
            torch.Tensor,
        ):
            return False, "FlashMLA metadata tensors are missing"
        return True, "captured"

    def _dsa_cuda_graph_flashmla_metadata_inputs(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._dsa_cuda_graph_manager is None:
            raise RuntimeError(
                f"[layer {self.layer_idx}] GLM-5 DSA CUDA graph metadata requested "
                "without a graph manager"
            )
        bucket_size = self._dsa_cuda_graph_manager.bucketing.get_padded_size(batch_size)
        metadata = getattr(GLM5AttnWrapper, "glm5_dsa_flashmla_graph_metadata", None)
        if not isinstance(metadata, dict):
            raise RuntimeError(
                f"[layer {self.layer_idx}] GLM-5 DSA CUDA graph replay requires "
                "per-forward FlashMLA metadata to be prepared before model forward"
            )
        if int(metadata.get("bucket_size", -1)) != int(bucket_size):
            raise RuntimeError(
                f"[layer {self.layer_idx}] GLM-5 DSA CUDA graph metadata bucket "
                f"{metadata.get('bucket_size')} does not match replay bucket {bucket_size}"
            )
        tile_scheduler_metadata = metadata.get("tile_scheduler_metadata")
        num_splits = metadata.get("num_splits")
        if not isinstance(tile_scheduler_metadata, torch.Tensor) or not isinstance(
            num_splits,
            torch.Tensor,
        ):
            raise RuntimeError(
                f"[layer {self.layer_idx}] GLM-5 DSA CUDA graph metadata must be "
                "tensor FlashMLA metadata buffers"
            )
        return tile_scheduler_metadata, num_splits

    @staticmethod
    def _compare_tensor_summary(
        name: str,
        graph_tensor: Optional[torch.Tensor],
        eager_tensor: Optional[torch.Tensor],
        *,
        exact: bool = False,
        atol: float = 5e-2,
        rtol: float = 5e-2,
    ) -> Tuple[bool, str]:
        if graph_tensor is None or eager_tensor is None:
            return False, f"{name}: skipped"
        if tuple(graph_tensor.shape) != tuple(eager_tensor.shape):
            return True, (
                f"{name}: shape_mismatch graph={tuple(graph_tensor.shape)} "
                f"eager={tuple(eager_tensor.shape)}"
            )
        if graph_tensor.numel() == 0:
            return False, f"{name}: empty"

        if exact:
            eager_same = eager_tensor.to(device=graph_tensor.device)
            mismatch = int((graph_tensor != eager_same).sum().item())
            if mismatch == 0:
                return False, f"{name}: exact"
            diff = (graph_tensor.float() - eager_same.float()).abs()
            max_abs = float(diff.max().item())
            return True, (
                f"{name}: mismatch={mismatch}/{graph_tensor.numel()} "
                f"max_abs={max_abs:.6g}"
            )

        if not graph_tensor.is_floating_point():
            graph_i = graph_tensor.to(dtype=torch.int64)
            eager_i = eager_tensor.to(dtype=torch.int64, device=graph_tensor.device)
            mismatch = int((graph_i != eager_i).sum().item())
            if mismatch == 0:
                return False, f"{name}: exact"
            max_abs = int((graph_i - eager_i).abs().max().item())
            return True, f"{name}: mismatch={mismatch}/{graph_i.numel()} max_abs={max_abs}"

        graph_f = graph_tensor.float()
        eager_f = eager_tensor.to(device=graph_tensor.device).float()
        diff = (graph_f - eager_f).abs()
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())
        close = bool(torch.allclose(graph_f, eager_f, atol=atol, rtol=rtol))
        return (
            not close,
            f"{name}: max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} "
            f"atol={atol:g} rtol={rtol:g}",
        )

    def _forward_decode_dsa_graph_compare(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_seqlen: int,
        gpu_paged_kv_manager,
        gpu_paged_kv_manager_aux,
        *,
        eager_output: torch.Tensor,
        eager_debug: Dict[str, Any],
    ) -> None:
        """Run graph replay as a side-channel and return/log eager-vs-graph diffs."""

        fail_on_mismatch = _glm5_dsa_graph_compare_fail_on_mismatch()
        try:
            bsz = hidden_states.shape[0]
            flashmla_tile_scheduler_metadata, flashmla_num_splits = (
                self._dsa_cuda_graph_flashmla_metadata_inputs(bsz)
            )
            if getattr(self, "_dsa_cuda_graph_full", False):
                if not fail_on_mismatch:
                    logging.warning(
                        "[GLM5_DSA_FULL_GRAPH_COMPARE] layer=%s skipped because "
                        "full-DSA side-channel replay writes active GPU KV cache; "
                        "set batchgen_debug.glm5_dsa_graph_compare_fail_on_mismatch=true "
                        "for fail-fast diagnostic runs.",
                        self.layer_idx,
                    )
                    return
                primary_slot_indices = getattr(
                    AttnWrapperBase,
                    "glm5_decode_primary_slot_indices",
                    None,
                )
                aux_slot_indices = getattr(
                    AttnWrapperBase,
                    "glm5_decode_aux_slot_indices",
                    None,
                )
                if primary_slot_indices is None or aux_slot_indices is None:
                    raise RuntimeError(
                        f"[layer {self.layer_idx}] GLM-5 full DSA graph compare "
                        "requires per-forward primary/aux slot tensors"
                    )
                graph_outputs = self._dsa_cuda_graph_manager.replay(
                    self._dsa_cuda_graph_segment_name,
                    bsz,
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    cache_seqlens=cache_seqlens.to(
                        dtype=torch.int32,
                        device=hidden_states.device,
                    ),
                    primary_slot_indices=primary_slot_indices[:bsz].to(
                        dtype=torch.int32,
                        device=hidden_states.device,
                    ),
                    aux_slot_indices=aux_slot_indices[:bsz].to(
                        dtype=torch.int32,
                        device=hidden_states.device,
                    ),
                    flashmla_tile_scheduler_metadata=flashmla_tile_scheduler_metadata,
                    flashmla_num_splits=flashmla_num_splits,
                )

                selector_inputs = eager_debug.get("selector_inputs")
                checks = [
                    self._compare_tensor_summary(
                        "final_o_proj",
                        graph_outputs.get("attn_output"),
                        eager_output,
                    ),
                ]
                if selector_inputs is not None:
                    checks.extend(
                        [
                            self._compare_tensor_summary(
                                "primary_k_tensor",
                                graph_outputs.get("primary_k_tensor"),
                                selector_inputs.primary_k_tensor,
                                exact=True,
                            ),
                            self._compare_tensor_summary(
                                "indexer_k_tensor",
                                graph_outputs.get("indexer_k_tensor"),
                                selector_inputs.indexer_k_tensor,
                                exact=True,
                            ),
                        ]
                    )
                failed = any(item[0] for item in checks)
                log_fn = logging.error if failed else logging.info
                log_fn(
                    "[GLM5_DSA_FULL_GRAPH_COMPARE] layer=%s status=%s bsz=%s "
                    "max_seqlen=%s min_cache=%s %s",
                    self.layer_idx,
                    "FAIL" if failed else "OK",
                    bsz,
                    max_seqlen,
                    int(cache_seqlens.min().item()),
                    "; ".join(message for _, message in checks),
                )
                if failed:
                    raise RuntimeError(
                        f"GLM-5 full DSA graph/eager compare failed on layer {self.layer_idx}"
                    )
                return

            from batchgen.attention.dsa.glm5_decode_selector import (
                build_glm5_dsa_graph_segment_inputs,
            )

            graph_inputs = build_glm5_dsa_graph_segment_inputs(
                self,
                hidden_states,
                position_ids,
                cache_seqlens,
                max_seqlen,
                gpu_paged_kv_manager,
                gpu_paged_kv_manager_aux,
                write_kv=False,
            )
            graph_outputs = self._dsa_cuda_graph_manager.replay(
                self._dsa_cuda_graph_segment_name,
                bsz,
                q_a=graph_inputs.q_a,
                q_nope=graph_inputs.q_nope,
                q_rope=graph_inputs.q_rope,
                head_gates=graph_inputs.head_gates,
                cache_seqlens=graph_inputs.cache_seqlens,
                positions_expanded=graph_inputs.positions_expanded,
                primary_slot_indices=graph_inputs.primary_slot_indices,
                aux_slot_indices=graph_inputs.aux_slot_indices,
                flashmla_tile_scheduler_metadata=flashmla_tile_scheduler_metadata,
                flashmla_num_splits=flashmla_num_splits,
            )
            graph_output = self._project_dsa_attn_heads(graph_outputs["attn_heads"])

            selector_inputs = eager_debug.get("selector_inputs")
            eager_attn_heads = eager_debug.get("attn_heads")
            checks = [
                self._compare_tensor_summary(
                    "final_o_proj", graph_output, eager_output,
                ),
                self._compare_tensor_summary(
                    "attn_heads", graph_outputs.get("attn_heads"), eager_attn_heads,
                ),
            ]
            if selector_inputs is not None:
                checks.extend(
                    [
                        self._compare_tensor_summary(
                            "q_nope",
                            graph_inputs.q_nope,
                            selector_inputs.q_nope,
                        ),
                        self._compare_tensor_summary(
                            "q_rope",
                            graph_inputs.q_rope,
                            selector_inputs.q_rope,
                        ),
                        self._compare_tensor_summary(
                            "primary_k_tensor",
                            graph_inputs.primary_k_tensor,
                            selector_inputs.primary_k_tensor,
                        ),
                        self._compare_tensor_summary(
                            "indexer_k_tensor",
                            graph_inputs.indexer_k_tensor,
                            selector_inputs.indexer_k_tensor,
                        ),
                        self._compare_tensor_summary(
                            "selected_lengths",
                            graph_outputs.get("selected_lengths"),
                            selector_inputs.selected_lengths,
                            exact=True,
                        ),
                        # The eager path now runs FA3 directly over the paged
                        # KV, so it no longer produces logical
                        # `selected_indices`, a padded `selected_mla_kv` slab
                        # or a packed FlashMLA `query_states` to compare the
                        # legacy standalone graph against. The four checks
                        # built on them (selected_indices, selected_mla_kv,
                        # absorbed_q, query_states) are dropped; final_o_proj
                        # / attn_heads / raw_attn_out still catch a mismatch.
                        self._compare_tensor_summary(
                            "raw_attn_out",
                            graph_outputs.get("raw_attn_out"),
                            eager_debug.get("raw_attn_out"),
                        ),
                    ]
                )

            failed = any(item[0] for item in checks)
            log_fn = logging.error if failed else logging.info
            log_fn(
                "[GLM5_DSA_GRAPH_COMPARE] layer=%s status=%s bsz=%s "
                "max_seqlen=%s min_cache=%s %s",
                self.layer_idx,
                "FAIL" if failed else "OK",
                bsz,
                max_seqlen,
                int(cache_seqlens.min().item()),
                "; ".join(message for _, message in checks),
            )
            if failed and fail_on_mismatch:
                raise RuntimeError(
                    f"GLM-5 DSA graph/eager compare failed on layer {self.layer_idx}"
                )
        except Exception:
            logging.exception(
                "[GLM5_DSA_GRAPH_COMPARE] layer=%s side-channel graph replay failed",
                self.layer_idx,
            )
            if fail_on_mismatch:
                raise

    def _forward_decode_dsa(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_seqlen: int,
        gpu_paged_kv_manager,
        gpu_paged_kv_manager_aux,
    ) -> torch.Tensor:
        """DSA sparse attention decode path.

        Computes MLA KV and writes to primary cache first, then runs indexer
        scoring on aux cache, gathers sparse MLA KV, and runs sparse FlashMLA.
        """
        attn = self.module
        bsz = hidden_states.shape[0]

        # Handle empty batch (some DP ranks have 0 sequences at late decode stages)
        if bsz == 0:
            return hidden_states.new_empty(0, 1, attn.hidden_size)

        debug_mode = _glm5_dsa_debug_mode()
        compare_active = (
            False if debug_mode == "eager"
            else _glm5_dsa_graph_compare_active()
        )
        compare_this_layer = _glm5_dsa_graph_compare_layer_enabled(self.layer_idx)
        graph_requested = (
            debug_mode == "graph"
            or (
                debug_mode != "eager"
                and (
                    _glm5_dsa_cuda_graph_required()
                    or getattr(self, "_dsa_cuda_graph_required", False)
                    or compare_active
                )
            )
        )
        compare_after_eager = False
        if graph_requested:
            index_topk = getattr(getattr(attn, "indexer", None), "index_topk", 2048)
            graph_can_replay = _glm5_dsa_cuda_graph_can_replay(
                cache_seqlens,
                max_seqlen,
                index_topk,
                captured_max_seqlen=getattr(self, "_dsa_cuda_graph_max_seqlen", None),
            )
            graph_has_bucket = (
                self._dsa_cuda_graph_manager is not None
                and self._dsa_cuda_graph_segment_name is not None
                and self._dsa_cuda_graph_manager.has_graph(
                    self._dsa_cuda_graph_segment_name,
                    bsz,
                )
            )
            graph_page_tables_match = graph_has_bucket and self._dsa_cuda_graph_page_tables_match(
                gpu_paged_kv_manager,
                gpu_paged_kv_manager_aux,
            )
            if graph_has_bucket:
                graph_forward_ready, graph_forward_reason = (
                    self._dsa_cuda_graph_forward_state_allows_replay(bsz)
                )
            else:
                graph_forward_ready = False
                graph_forward_reason = "batch size has no captured graph bucket"
            if (
                graph_can_replay
                and graph_has_bucket
                and graph_page_tables_match
                and graph_forward_ready
                and not compare_active
            ):
                _record_glm5_dsa_dispatch(
                    "graph",
                    layer_idx=self.layer_idx,
                    bsz=bsz,
                    reason="graph replay",
                )
                return self._forward_decode_dsa_graph(
                    hidden_states,
                    position_ids,
                    cache_seqlens,
                    max_seqlen,
                    gpu_paged_kv_manager,
                    gpu_paged_kv_manager_aux,
                )
            compare_after_eager = bool(
                compare_active
                and compare_this_layer
                and graph_can_replay
                and graph_has_bucket
                and graph_page_tables_match
                and graph_forward_ready
            )
            if not graph_can_replay:
                if index_topk <= 0:
                    reason = f"invalid index_topk={index_topk}"
                elif cache_seqlens.ndim != 1:
                    reason = f"cache_seqlens ndim {cache_seqlens.ndim} is not 1"
                else:
                    reason = "current decode metadata is not graph-compatible"
            elif not graph_has_bucket:
                reason = f"batch size {bsz} has no captured graph bucket"
            elif not graph_page_tables_match:
                reason = "captured page-table storage no longer matches active storage"
            elif not graph_forward_ready:
                reason = graph_forward_reason
            else:
                reason = "graph/eager compare mode is returning eager output"
            _log_glm5_dsa_graph_eager_fallback_once(
                self.layer_idx,
                cache_seqlens,
                index_topk,
                reason,
            )
            if compare_active and compare_this_layer and not compare_after_eager:
                _log_glm5_dsa_graph_compare_unavailable_once(
                    self.layer_idx,
                    reason,
                )
        else:
            if debug_mode == "eager":
                reason = "debug mode requested eager"
            else:
                reason = "graph not requested"

        _record_glm5_dsa_dispatch(
            "eager",
            layer_idx=self.layer_idx,
            bsz=bsz,
            reason=reason,
        )
        eager_result = self._forward_decode_dsa_eager(
            hidden_states,
            position_ids,
            cache_seqlens,
            max_seqlen,
            gpu_paged_kv_manager,
            gpu_paged_kv_manager_aux,
            return_debug=compare_after_eager,
        )
        if not compare_after_eager:
            return eager_result

        eager_output, eager_debug = eager_result
        self._forward_decode_dsa_graph_compare(
            hidden_states,
            position_ids,
            cache_seqlens,
            max_seqlen,
            gpu_paged_kv_manager,
            gpu_paged_kv_manager_aux,
            eager_output=eager_output,
            eager_debug=eager_debug,
        )
        return eager_output

    def _forward_decode_dsa_eager(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_seqlen: int,
        gpu_paged_kv_manager,
        gpu_paged_kv_manager_aux,
        *,
        return_debug: bool = False,
    ):
        attn = self.module
        bsz = hidden_states.shape[0]
        li = self.layer_idx

        from batchgen.attention.dsa.glm5_decode_selector import (
            build_glm5_dsa_flashmla_inputs,
        )
        from batchgen.attention.mla.fa3_backend import act_quant
        from batchgen.gemm.w8a8_deepgemm import w8a8_deepgemm
        from batchgen.timing import get_decode_timer

        weight_scale = self.weight_dequant_scale
        dt = get_decode_timer()
        # The builder runs sparse attention itself on FA3 over the resident
        # paged KV and owns the "sparse_attn" timer; there is no padded
        # gather and no FlashMLA call left on this path.
        selector_inputs = build_glm5_dsa_flashmla_inputs(
            self,
            hidden_states,
            position_ids,
            cache_seqlens,
            max_seqlen,
            gpu_paged_kv_manager,
            gpu_paged_kv_manager_aux,
            return_selected_indices=return_debug,
        )
        attn_out = selector_inputs.attn_out

        # --- Step 6: out_absorb → o_proj ---
        with (dt.timed("o_proj", li) if dt else _nullctx()):
            if self._fp8_absorb_weights is None:
                raise RuntimeError(
                    f"[layer {self.layer_idx}] GLM-5 DSA requires WP5 FP8 "
                    "out_absorb; PyTorch/BF16 fallback is disabled"
                )
            attn_heads = absorb_out(attn_out, self._fp8_absorb_weights)
            attn_output = attn_heads.reshape(bsz, attn.num_heads * attn.v_head_dim)
            attn_output_fp8, attn_output_scale = act_quant(attn_output)
            attn_output = w8a8_deepgemm(
                attn_output_fp8, attn_output_scale,
                attn.o_proj.weight, weight_scale["o_proj.weight_scale_inv"],
            )
            attn_output = attn_output.view(bsz, 1, -1)

        if return_debug:
            return attn_output, {
                "selector_inputs": selector_inputs,
                "raw_attn_out": attn_out,
                "attn_heads": attn_heads,
            }
        return attn_output
