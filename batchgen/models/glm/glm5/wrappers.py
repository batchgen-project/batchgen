# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
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
from batchgen.timing import init_decode_timer, init_prefill_timer

# Try importing FP8 absorb kernels (WP5)
try:
    from batchgen_kernels.attention.dsa.fp8_absorb import (
        FP8AbsorbWeights, fp8_q_absorb, fp8_out_absorb,
    )
    _HAS_FP8_ABSORB = True
except Exception as _e:
    _HAS_FP8_ABSORB = False
    logging.debug(f"[WP5] FP8 absorb import failed: {_e}")

# Try importing fused indexer KV proj (WP2)
try:
    from batchgen_kernels.attention.dsa.fused_indexer_kv_proj_cuda import (
        build_module as _build_indexer_module,
        FP8IndexerWeightsCUDA,
    )
    _HAS_FUSED_INDEXER_KV = True
except Exception as _e:
    _HAS_FUSED_INDEXER_KV = False
    logging.debug(f"[WP2] Fused indexer KV proj import failed: {_e}")

# Try importing fused scoring pipeline (WP4)
try:
    from batchgen_kernels.attention.dsa.fused_indexer_score import (
        FP8WqbWeightsCUDA,
        fused_score_pipeline,
    )
    _HAS_FUSED_SCORE = True
except Exception as _e:
    _HAS_FUSED_SCORE = False
    logging.debug(f"[WP4] Fused scoring import failed: {_e}")

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

_GLM5_PREFILL_CATEGORIES = [
    "scheduler_capacity_all_gather", "setup_prepack", "setup_flatten",
    "microbatch_input_concat", "microbatch_cu_seqlens", "embedding",
    "input_norm", "add_rmsnorm", "dense_mlp", "residual_add",
    "attn_q_a", "attn_q_norm", "attn_q_b", "attn_kv_a",
    "attn_kv_norm", "attn_rope", "attn_primary_kv_materialize",
    "attn_kv_b", "attn_qkv_materialize", "attn_fa3", "attn_o",
    "indexer_wk", "indexer_norm", "indexer_rope_hadamard",
    "indexer_kv_materialize", "primary_kv_materialize",
    "moe_pointer_table_h2d",
    "moe_router", "moe_dispatch", "moe_act_quant_s1", "moe_grouped_s1",
    "moe_act_quant_s3", "moe_grouped_s3", "moe_reduce", "moe_shared",
    "final_norm", "last_token_gather", "lm_head", "token_select",
]
_glm5_prefill_timer = init_prefill_timer("GLM-5", _GLM5_PREFILL_CATEGORIES)

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

    def _unregister_fp8_weights(self):
        self.cached_gate = None
        self.cached_up = None
        self.cached_down = None

    def load_weights_pinned(self) -> Dict[str, torch.Tensor]:
        """Wait without the single-expert sliding-window eviction policy.

        Grouped prefill owns all 256 keys until its completion event; evicting
        an earlier key while acquiring a later expert invalidates that layer.
        """
        return self.core_engine.get_weights_pinned(self.module_key)

    def _forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """FP8 forward using cached weight tensors directly (no nn.Module delegation)."""
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.persistent:
            weights = self.load_weights(self.module_key)
            self.cached_gate = weights["gate_proj.weight"]
            self.cached_up = weights["up_proj.weight"]
            self.cached_down = weights["down_proj.weight"]

        result = self._forward_impl(hidden_states)

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
    # Whole-model CUDA graph can pad local rows to a global NCCL bucket. These
    # graph-owned overrides let GLM-5 DSA use explicit slot sentinels for padded
    # rows instead of deriving slot count from cur_batch.
    glm5_decode_primary_slot_indices: ClassVar[Optional[torch.Tensor]] = None
    glm5_decode_aux_slot_indices: ClassVar[Optional[torch.Tensor]] = None
    glm5_dsa_graph_forward_state: ClassVar[Optional[Dict[str, Any]]] = None
    glm5_dsa_flashmla_graph_metadata: ClassVar[Optional[Dict[str, Any]]] = None
    # Runtime proof that every prefill token's primary MLA KV and every
    # applicable auxiliary/indexer KV were scheduled for host offload.
    glm5_prefill_kv_offload_audit: ClassVar[Optional[Dict[str, Dict[int, dict]]]] = None

    @classmethod
    def start_prefill_kv_offload_audit(cls) -> None:
        if cls.glm5_prefill_kv_offload_audit is not None:
            raise RuntimeError("GLM-5 prefill KV offload audit is already active")
        cls.glm5_prefill_kv_offload_audit = {"primary": {}, "aux": {}}

    @classmethod
    def record_prefill_kv_offload(
        cls,
        kind: str,
        layer_idx: int,
        *,
        sequences: int,
        tokens: int,
    ) -> None:
        audit = cls.glm5_prefill_kv_offload_audit
        if audit is None:
            return
        entry = audit[kind].setdefault(
            int(layer_idx),
            {"calls": 0, "sequences": 0, "tokens": 0},
        )
        entry["calls"] += 1
        entry["sequences"] += int(sequences)
        entry["tokens"] += int(tokens)

    @classmethod
    def finish_prefill_kv_offload_audit(cls) -> Dict[str, Dict[int, dict]]:
        audit = cls.glm5_prefill_kv_offload_audit
        cls.glm5_prefill_kv_offload_audit = None
        if audit is None:
            raise RuntimeError("GLM-5 prefill KV offload audit was not active")
        return audit

    @classmethod
    def abort_prefill_kv_offload_audit(cls) -> None:
        cls.glm5_prefill_kv_offload_audit = None

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
        self._fp8_folded_q_b_proj = None
        self._fp8_folded_q_b_scale = None
        self._folded_q_b_retained_rows = None
        # Cached absorbed projections (Fix 1: avoid 78× FP8 dequant per step).
        # These are BF16 WEIGHT copies, not workspaces. They are quantizer
        # INPUT only: once _fp8_absorb_weights is built they are freed by
        # initialize_decode_absorb, so decode must never read them.
        self._cached_q_absorb = None
        self._cached_out_absorb = None
        # SGLang-aligned BF16 BMM absorb weight, built by
        # initialize_decode_absorb ONLY when FP8 absorb is unavailable.
        # w_vc: [H, 512, 256] BF16 non-contig view for the bmm fallback.
        self.w_vc = None
        # WP5: FP8 absorb weights (pre-quantized once at init)
        self._fp8_absorb_weights = None
        # WP2: Fused indexer KV proj (CUDA WGMMA)
        self._indexer_cuda_module = None
        self._indexer_cuda_weights = None
        # WP4: Fused scoring pipeline
        self._fused_wqb_weights = None
        self._fused_score_module = None
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
        self._fp8_folded_q_b_proj = None
        self._fp8_folded_q_b_scale = None
        self._folded_q_b_retained_rows = None

    def _initialize_folded_q_b(self) -> None:
        """Fold the static q-nope absorb into Q-B for graph decode."""
        if self._fp8_folded_q_b_proj is not None:
            return
        q_b_scale = self.weight_dequant_scale.get("q_b_proj.weight_scale_inv")
        kv_b_scale = self.weight_dequant_scale.get("kv_b_proj.weight_scale_inv")
        if q_b_scale is None or kv_b_scale is None:
            raise RuntimeError(
                f"[layer {self.layer_idx}] folded Q-B requires Q-B and KV-B FP8 scales"
            )

        import deep_gemm

        attn = self.module
        block_size = 128
        q_b_weight = attn.q_b_proj.weight.data
        q_absorb = self._cached_q_absorb
        if q_absorb is None:
            kv_b_proj = glm5_fp8_dequantization(
                attn.kv_b_proj.weight.data,
                kv_b_scale,
            ).view(attn.num_heads, -1, attn.kv_lora_rank)
            q_absorb = kv_b_proj[:, :attn.qk_nope_head_dim].contiguous()
        q_b_bf16 = glm5_fp8_dequantization(q_b_weight, q_b_scale)
        q_b_heads_bf16 = q_b_bf16.view(
            attn.num_heads,
            attn.q_head_dim,
            attn.q_lora_rank,
        )
        q_b_heads_fp8 = q_b_weight.view(
            attn.num_heads,
            attn.q_head_dim,
            attn.q_lora_rank,
        )
        q_b_scale_heads = q_b_scale.view(
            attn.num_heads,
            attn.q_head_dim // block_size,
            attn.q_lora_rank // block_size,
        )
        folded_q_nope = torch.bmm(
            q_absorb.transpose(1, 2).float(),
            q_b_heads_bf16[:, : attn.qk_nope_head_dim].float(),
        )
        folded_q_nope_fp8 = []
        folded_q_nope_scale = []
        for head in range(attn.num_heads):
            weight, scale = deep_gemm.per_block_cast_to_fp8(
                folded_q_nope[head],
                use_ue8m0=False,
            )
            folded_q_nope_fp8.append(weight)
            folded_q_nope_scale.append(scale)

        retained_rows = block_size
        self._folded_q_b_retained_rows = retained_rows
        retained_weight = q_b_heads_fp8[:, -retained_rows:].flatten(0, 1)
        retained_scale = q_b_scale_heads[:, -1:].flatten(0, 1)
        self._fp8_folded_q_b_proj = torch.cat(
            (torch.cat(folded_q_nope_fp8, dim=0), retained_weight),
            dim=0,
        ).contiguous()
        self._fp8_folded_q_b_scale = torch.cat(
            (torch.cat(folded_q_nope_scale, dim=0), retained_scale),
            dim=0,
        ).contiguous()

        # The GLM-5.2 graph path is fail-closed, so the original Q-B/KV-B
        # tensors and q-absorb FP8 copy have no remaining decode consumer.
        # Releasing them offsets most of the wider folded projection.
        attn.q_b_proj.weight.data = torch.empty(
            0,
            dtype=q_b_weight.dtype,
            device=q_b_weight.device,
        )
        attn.kv_b_proj.weight.data = torch.empty(
            0,
            dtype=attn.kv_b_proj.weight.dtype,
            device=attn.kv_b_proj.weight.device,
        )
        self.weight_dequant_scale["q_b_proj.weight_scale_inv"] = torch.empty(
            0,
            dtype=q_b_scale.dtype,
            device=q_b_scale.device,
        )
        self.weight_dequant_scale["kv_b_proj.weight_scale_inv"] = torch.empty(
            0,
            dtype=kv_b_scale.dtype,
            device=kv_b_scale.device,
        )
        self.fp8_q_b_proj = None
        self.fp8_kv_b_proj = None
        self._fp8_absorb_weights.q_absorb_fp8 = torch.empty(
            0,
            dtype=self._fp8_absorb_weights.q_absorb_fp8.dtype,
            device=self._fp8_absorb_weights.q_absorb_fp8.device,
        )
        self._fp8_absorb_weights.q_absorb_scale = torch.empty(
            0,
            dtype=self._fp8_absorb_weights.q_absorb_scale.dtype,
            device=self._fp8_absorb_weights.q_absorb_scale.device,
        )
        logging.info(
            "[layer %s] initialized folded GLM-5.2 Q-B weight %s",
            self.layer_idx,
            tuple(self._fp8_folded_q_b_proj.shape),
        )

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
        self._cached_q_absorb = kv_b_proj[:, :attn.qk_nope_head_dim, :].contiguous()
        self._cached_out_absorb = kv_b_proj[:, attn.qk_nope_head_dim:, :].contiguous()

        # WP5: Pre-quantize absorb weights for FP8 WGMMA kernel
        if _HAS_FP8_ABSORB:
            try:
                self._fp8_absorb_weights = FP8AbsorbWeights(
                    self._cached_q_absorb,   # [H, 192, 512]
                    self._cached_out_absorb,  # [H, 256, 512]
                )
                logging.debug(
                    f"[layer {self.layer_idx}] FP8 absorb weights initialized"
                )
            except Exception as e:
                logging.warning(
                    f"[layer {self.layer_idx}] FP8 absorb init failed: {e}"
                )
                self._fp8_absorb_weights = None

        if self._fp8_absorb_weights is not None:
            # The FP8 kernel owns both absorb GEMMs on every live decode path
            # (glm5_decode_selector._build_query_states hard-fails without it),
            # so the BF16 originals were quantizer input only. Free them here —
            # BEFORE _init_gpu_kv_with_actual_size sizes the KV pool — instead
            # of holding [H,192,512] + [H,256,512] BF16 per layer for nothing.
            # H*448*512*2 B/layer freed; the BF16 bmm fallback (w_vc) is not
            # built at all on this path.
            self._cached_q_absorb = None
            self._cached_out_absorb = None
            self.w_vc = None
        else:
            # No FP8 absorb kernel: keep the SGLang-aligned BF16 BMM weight
            # (matches deepseek_weight_loader.py:572-578 layout exactly).
            #   self.w_vc — [H, kv_lora=512, v_head=256], BF16
            #     transposed so `bmm(attn_out_T, w_vc)` produces [H, B, 256].
            # Mirror SGLang exactly: .contiguous() first, then .transpose — the
            # final tensor is a non-contiguous view with physical [H, 256, 512]
            # and logical [H, 512, 256]. Adding a trailing .contiguous() would
            # re-lay it out and change which cuBLAS kernel bmm dispatches to.
            self.w_vc = self._cached_out_absorb.contiguous().transpose(1, 2)

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

    def dequantize_weights(
        self, weights_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Return FP8 weights unchanged — deepgemm handles FP8 directly."""
        return weights_dict

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

    def _forward_prefill(self, hidden_states: torch.Tensor, **kwargs) -> Tuple:
        """Prefill forward with DSA auxiliary cache population.

        1. Standard MLA prefill via FA3 (full attention)
        2. Compute indexer K and write to auxiliary cache
        """
        AttnWrapperBase.retire_pending_prefill_offloads_before_layer(
            self.layer_idx,
            device=hidden_states.device,
        )
        if self.prepack_mode:
            hidden_states_2d = hidden_states.squeeze(0)
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
                    max_seqlen=self.prepack_max_seqlen,
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

        from batchgen.timing import get_prefill_timer
        timer = get_prefill_timer()
        materialize_ctx = (
            timer.timed("indexer_kv_materialize", self.layer_idx)
            if timer is not None else _nullctx()
        )
        with materialize_ctx:
            packed_kv = offload_kv.contiguous()
        enqueue_ctx = (
            timer.host_timed("indexer_kv_offload_enqueue", self.layer_idx)
            if timer is not None else _nullctx()
        )
        with enqueue_ctx:
            task = AttnWrapperBase.host_paged_kv_worker_view_aux.async_offload_packed_layer_kv_to_host(
                layer_idx=self.layer_idx,
                sequence_ids=global_sequence_ids,
                k_tensor=packed_kv,
                v_tensor=None,
                sequence_lengths=sequence_lengths,
            )
        AttnWrapperBase.pin_prefill_offload_tensor(packed_kv, self.layer_idx)
        AttnWrapperBase.track_prefill_offload_task(task, self.layer_idx)
        self.record_prefill_kv_offload(
            "aux",
            self.layer_idx,
            sequences=len(sequence_lengths),
            tokens=sum(sequence_lengths),
        )

    def _offload_prepacked_kv(self, offload_kv: torch.Tensor):
        """Offload packed primary KV with one asynchronous task per layer."""
        sequence_lengths = list(self.prepack_seq_lengths)
        global_sequence_ids = self.cur_batch
        if len(sequence_lengths) != len(global_sequence_ids):
            raise RuntimeError("GLM-5 packed primary KV metadata size mismatch")
        if sum(sequence_lengths) != offload_kv.shape[0]:
            raise RuntimeError("GLM-5 packed primary KV token count mismatch")

        from batchgen.timing import get_prefill_timer
        timer = get_prefill_timer()
        materialize_ctx = (
            timer.timed("primary_kv_materialize", self.layer_idx)
            if timer is not None else _nullctx()
        )
        with materialize_ctx:
            packed_kv = offload_kv.unsqueeze(1).contiguous()
        enqueue_ctx = (
            timer.host_timed("primary_kv_offload_enqueue", self.layer_idx)
            if timer is not None else _nullctx()
        )
        with enqueue_ctx:
            task = self.core_engine.host_paged_kv_worker_view.async_offload_packed_layer_kv_to_host(
                layer_idx=self.layer_idx,
                sequence_ids=global_sequence_ids,
                k_tensor=packed_kv,
                v_tensor=None,
                sequence_lengths=sequence_lengths,
            )
        AttnWrapperBase.pin_prefill_offload_tensor(packed_kv, self.layer_idx)
        AttnWrapperBase.track_prefill_offload_task(task, self.layer_idx)
        self.record_prefill_kv_offload(
            "primary",
            self.layer_idx,
            sequences=len(sequence_lengths),
            tokens=sum(sequence_lengths),
        )

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
                AttnWrapperBase.kv_append_callback_aux(
                    self.layer_idx,
                    graph_outputs["indexer_k_tensor"],
                    None,
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

            attn = self.module
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
                        self._compare_tensor_summary(
                            "selected_indices",
                            graph_outputs.get("top_k_indices"),
                            selector_inputs.selected_indices,
                            exact=True,
                        ),
                        self._compare_tensor_summary(
                            "selected_mla_kv",
                            graph_outputs.get("selected_mla_kv"),
                            selector_inputs.selected_mla_kv,
                            atol=0.0,
                            rtol=0.0,
                        ),
                        self._compare_tensor_summary(
                            "absorbed_q",
                            graph_outputs.get("absorbed_q"),
                            selector_inputs.query_states[:, 0, :, : attn.kv_lora_rank],
                        ),
                        self._compare_tensor_summary(
                            "query_states",
                            graph_outputs.get("query_states"),
                            selector_inputs.query_states,
                            atol=0.0,
                            rtol=0.0,
                        ),
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
        from batchgen.attention.dsa.sparse_decode_mla import (
            run_prepared_sparse_flash_mla_decode,
        )
        from batchgen.attention.mla.fa3_backend import act_quant
        from batchgen.gemm.w8a8_deepgemm import w8a8_deepgemm
        from batchgen.timing import get_decode_timer

        weight_scale = self.weight_dequant_scale
        dt = get_decode_timer()
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
        with (dt.timed("sparse_attn", li) if dt else _nullctx()):
            attn_out = run_prepared_sparse_flash_mla_decode(selector_inputs.flashmla)

        # --- Step 6: out_absorb → o_proj ---
        with (dt.timed("o_proj", li) if dt else _nullctx()):
            # WP5: FP8 out_absorb kernel, or the SGLang-aligned BF16 BMM when
            # no FP8 absorb kernel was built for this layer.
            if self._fp8_absorb_weights is not None:
                attn_heads = fp8_out_absorb(attn_out, self._fp8_absorb_weights)
            elif self.w_vc is not None:
                # SGLang forward_mla.py:548 — bmm(attn_output.T, w_vc).
                attn_out_3d = attn_out.squeeze(1)  # [B, H, 512]
                attn_heads = torch.bmm(
                    attn_out_3d.transpose(0, 1), self.w_vc,
                ).transpose(0, 1).unsqueeze(1)  # [B, 1, H, v_head_dim]
            else:
                # No silent fallback: w_vc is only None when the FP8 weights
                # existed at init (and freed it) but are gone now — i.e. state
                # was torn down mid-flight, not a supported configuration.
                raise RuntimeError(
                    f"[layer {self.layer_idx}] GLM-5 out_absorb has neither FP8 "
                    "absorb weights nor the BF16 w_vc fallback: the BF16 copies "
                    "were freed by initialize_decode_absorb once FP8 absorb was "
                    "built, so _fp8_absorb_weights must not be cleared "
                    "afterwards. Re-run initialize_decode_absorb()."
                )
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
