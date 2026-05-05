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
from contextlib import nullcontext as _nullctx
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

import torch.nn.functional as F

from batchgen.models.glm.glm5.decode_utils import (
    build_flat_paged_gather_indices,
    build_batch_slot_indices,
    build_paged_gather_cache_key,
    build_clamped_dense_token_indices,
    reorder_block_table_to_batch_slots,
    validate_effective_block_table_pages,
    validate_token_indices_within_seqlens,
)
from batchgen.models.glm.glm5.prefix_reuse import (
    offload_glm5_prepacked_mla_kv,
    run_glm5_prefix_aware_prefill,
)
from batchgen.models.wrappers import ExpertWrapperBase, AttnWrapperBase
from batchgen.timing import init_decode_timer

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
    logging.warning(f"[WP5] FP8 absorb import failed: {_e!r}")

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
    logging.warning(f"[WP2] Fused indexer KV proj import failed: {_e!r}")

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
    logging.warning(f"[WP4] Fused scoring import failed: {_e!r}")

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


def _allow_dsa_fallback() -> bool:
    return False


def _raise_required_dsa_kernel(name: str, error: Optional[BaseException] = None) -> None:
    msg = (
        f"GLM-5 DSA requires {name}; silent fallback is disabled. "
        "Install the repaired batchgen_kernels package; production GLM-5 "
        "DSA does not support runtime fallback."
    )
    if error is not None:
        msg = f"{msg} Import/init error: {error}"
    raise RuntimeError(msg)


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
        # Cached absorbed projections (Fix 1: avoid 78× FP8 dequant per step)
        self._cached_q_absorb = None
        self._cached_out_absorb = None
        # SGLang-aligned BF16 BMM absorb weights (set by
        # initialize_decode_absorb). w_kc: [H, 192, 512] BF16 with
        # SGLang's stride trick; w_vc: [H, 512, 256] BF16 non-contig view.
        self.w_kc = None
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

    def _register_fp8_weights(self):
        """Cache FP8 attention weights. GLM-5 uses kv_a_proj_with_mqa."""
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

        # SGLang-aligned absorb weights for BF16 BMM (matches
        # deepseek_weight_loader.py:572-578 layout exactly).
        #
        #   self.w_kc — [H, qk_nope=192, kv_lora=512], BF16
        #     `.transpose(1,2).contiguous().transpose(1,2)` is SGLang's stride
        #     trick: physical memory laid out as [H, 512, 192] contiguous,
        #     strides swapped on dim 1/2. Math-identical to [H, 192, 512] but
        #     makes bmm/bmm_fp8 hit the same cuBLAS kernel SGLang triggers.
        #   self.w_vc — [H, kv_lora=512, v_head=256], BF16
        #     transposed so `bmm(attn_out_T, w_vc)` produces [H, B, 256].
        self.w_kc = self._cached_q_absorb.transpose(1, 2).contiguous().transpose(1, 2)
        # Mirror SGLang exactly: .contiguous() first, then .transpose — the
        # final tensor is a non-contiguous view with physical [H, 256, 512]
        # and logical [H, 512, 256]. Adding a trailing .contiguous() would
        # re-lay it out and change which cuBLAS kernel bmm dispatches to.
        self.w_vc = self._cached_out_absorb.contiguous().transpose(1, 2)

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

        # WP2/WP4 init moved to initialize_fused_kernels() — must run after set_device

    def initialize_fused_kernels(self):
        """Initialize TMA-based CUDA kernels (WP2/WP4).

        Must be called AFTER torch.cuda.set_device(local_rank) — TMA descriptors
        contain physical GPU addresses and are not portable across devices —
        AND after _setup_fp8_scales has attached indexer.wk_scale / wq_b_scale.
        """
        attn = self.module
        if not hasattr(attn, "indexer"):
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
        if not _HAS_FUSED_INDEXER_KV:
            if not _allow_dsa_fallback():
                _raise_required_dsa_kernel("WP2 fused indexer KV projection", _FUSED_INDEXER_KV_IMPORT_ERROR)
        else:
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
                if not _allow_dsa_fallback():
                    _raise_required_dsa_kernel("WP2 fused indexer KV projection", e)
                logging.warning(
                    f"[layer {self.layer_idx}] WP2 init failed: {e}"
                )
                self._indexer_cuda_module = None
                self._indexer_cuda_weights = None

        # WP4: Fused scoring pipeline (CUDA WGMMA wq_b + RoPE + Hadamard + scoring + topk)
        if not _HAS_FUSED_SCORE:
            if not _allow_dsa_fallback():
                _raise_required_dsa_kernel("WP4 fused indexer scoring", _FUSED_SCORE_IMPORT_ERROR)
        elif self._indexer_cuda_module is None:
            if not _allow_dsa_fallback():
                _raise_required_dsa_kernel("WP4 fused indexer scoring", RuntimeError("WP2 module is unavailable"))
        elif hasattr(indexer, 'wq_b_scale'):
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
                if not _allow_dsa_fallback():
                    _raise_required_dsa_kernel("WP4 fused indexer scoring", e)
                logging.warning(
                    f"[layer {self.layer_idx}] WP4 init failed: {e}"
                )
                self._fused_wqb_weights = None

    def dequantize_weights(
        self, weights_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Return FP8 weights unchanged — deepgemm handles FP8 directly."""
        return weights_dict

    def _forward_prefill(self, hidden_states: torch.Tensor, **kwargs) -> Tuple:
        """Prefill forward with DSA auxiliary cache population.

        1. Standard MLA prefill via FA3 (full attention)
        2. Compute indexer K and write to auxiliary cache
        """
        if self.prepack_mode:
            hidden_states_2d = hidden_states.squeeze(0)
            metadata = self.prefix_cache_metadata()
            position_ids = self.position_ids.to(hidden_states_2d.device)
            if metadata.prefix_reuse_mode:
                attn_output, offload_kv = run_glm5_prefix_aware_prefill(
                    wrapper=self,
                    hidden_states_2d=hidden_states_2d,
                    position_ids=position_ids,
                    metadata=metadata,
                )
            else:
                attn_output, offload_kv = self.module.prefill_attn_w8a16_prepacked(
                    hidden_states_2d,
                    position_ids,
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
            indexer_kv = self.module.indexer.compute_indexer_kv(
                hidden_states_2d.unsqueeze(0),
                positions=position_ids,
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
        """Offload indexer KV cache per-sequence to auxiliary host memory."""
        if AttnWrapperBase.host_paged_kv_worker_view_aux is None:
            raise RuntimeError(
                "GLM-5 DSA auxiliary host KV worker view is required for "
                "indexer KV offload"
            )
        offload_glm5_prepacked_mla_kv(
            key=offload_kv,
            worker_view=AttnWrapperBase.host_paged_kv_worker_view_aux,
            layer_idx=self.layer_idx,
            metadata=self.prefix_cache_metadata(),
        )

    def _offload_prepacked_kv(self, offload_kv: torch.Tensor):
        """Offload KV cache per-sequence to host memory."""
        worker_view = getattr(self.core_engine, "host_paged_kv_worker_view", None)
        if worker_view is None:
            raise RuntimeError("GLM-5 primary host KV worker view is required")
        offload_glm5_prepacked_mla_kv(
            key=offload_kv,
            worker_view=worker_view,
            layer_idx=self.layer_idx,
            metadata=self.prefix_cache_metadata(),
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
        from batchgen.attention.mla.fa3_backend import act_quant
        from batchgen.attention.mla.fused_rmsnorm_rope import (
            fused_rmsnorm_rope_with_q_native as _fused_rmsnorm_rope,
        )
        from batchgen.attention.mla.flashmla_backend import deepseek_v3_dequantization
        from batchgen.attention.dsa.sparse_gather import sparse_gather_from_paged_kv
        from batchgen.attention.dsa.sparse_decode_mla import sparse_flash_mla_decode
        from batchgen.gemm.w8a8_deepgemm import w8a8_deepgemm
        from batchgen.timing import get_decode_timer

        weight_scale = self.weight_dequant_scale
        attn = self.module
        indexer = attn.indexer
        bsz = hidden_states.shape[0]
        dt = get_decode_timer()
        li = self.layer_idx

        # Some DP ranks have 0 sequences at late decode stages. Empty rows are
        # valid only as a fully empty local batch; avoid reductions such as
        # cache_seqlens.max() below.
        if bsz == 0:
            if hidden_states.dim() != 3 or hidden_states.shape[1] != 1:
                raise RuntimeError(
                    f"GLM-5 DSA layer {li}: empty hidden_states must be [0,1,H], got {tuple(hidden_states.shape)}"
                )
            return hidden_states.new_empty(0, 1, attn.hidden_size)

        if hidden_states.dim() != 3 or hidden_states.shape[1] != 1:
            raise RuntimeError(
                f"GLM-5 DSA layer {li}: hidden_states must be [B,1,H], got {tuple(hidden_states.shape)}"
            )
        if hidden_states.dtype != torch.bfloat16:
            raise RuntimeError(
                f"GLM-5 DSA layer {li}: hidden_states dtype must be bf16, got {hidden_states.dtype}"
            )
        if position_ids is None or position_ids.dim() != 2 or position_ids.shape != (bsz, 1):
            raise RuntimeError(
                f"GLM-5 DSA layer {li}: position_ids must be [B,1], got "
                f"{None if position_ids is None else tuple(position_ids.shape)}"
            )
        if cache_seqlens is None or cache_seqlens.dim() != 1 or cache_seqlens.shape[0] != bsz:
            raise RuntimeError(
                f"GLM-5 DSA layer {li}: cache_seqlens must be [B], got "
                f"{None if cache_seqlens is None else tuple(cache_seqlens.shape)} for bsz={bsz}"
            )
        if not cache_seqlens.is_cuda or not position_ids.is_cuda:
            raise RuntimeError(
                f"GLM-5 DSA layer {li}: cache_seqlens and position_ids must be CUDA tensors"
            )
        if gpu_paged_kv_manager is None or gpu_paged_kv_manager_aux is None:
            raise RuntimeError(f"GLM-5 DSA layer {li}: primary and auxiliary GPU KV managers are required")
        if gpu_paged_kv_manager.config.page_size_tokens != gpu_paged_kv_manager_aux.config.page_size_tokens:
            raise RuntimeError(
                f"GLM-5 DSA layer {li}: primary/aux page-size mismatch: "
                f"primary={gpu_paged_kv_manager.config.page_size_tokens}, "
                f"aux={gpu_paged_kv_manager_aux.config.page_size_tokens}"
            )
        if max_seqlen != int(cache_seqlens.max().item()):
            raise RuntimeError(
                f"GLM-5 DSA layer {li}: max_seqlen={max_seqlen} must equal "
                f"cache_seqlens.max()={int(cache_seqlens.max().item())}"
            )
        if not torch.equal(position_ids.squeeze(-1).to(cache_seqlens.device), cache_seqlens - 1):
            raise RuntimeError(
                f"GLM-5 DSA layer {li}: position_ids must equal cache_seqlens - 1 for every row"
            )

        # --- Shared FP8 activation quantization ---
        with (dt.timed("act_quant", li) if dt else _nullctx()):
            hidden_flat = hidden_states.squeeze(1)  # [batch, hidden_size]
            hidden_fp8, hidden_scale = act_quant(hidden_flat)

        # --- Q path: q_a_proj → layernorm → q_b_proj → split → RoPE ---
        with (dt.timed("q_proj", li) if dt else _nullctx()):
            q_a = w8a8_deepgemm(
                hidden_fp8, hidden_scale,
                attn.q_a_proj.weight, weight_scale["q_a_proj.weight_scale_inv"],
            )
            q_a_normed = attn.q_a_layernorm(q_a)
            q_a_fp8, q_a_scale = act_quant(q_a_normed)
            q = w8a8_deepgemm(
                q_a_fp8, q_a_scale,
                attn.q_b_proj.weight, weight_scale["q_b_proj.weight_scale_inv"],
            )
            q = q.view(bsz, 1, attn.num_heads, attn.q_head_dim).transpose(1, 2)
            q_nope, q_pe = torch.split(
                q, [attn.qk_nope_head_dim, attn.qk_rope_head_dim], dim=-1
            )
            q_pe = q_pe.contiguous()

        # --- KV path: kv_a_proj → fused_rmsnorm_rope → compressed KV ---
        with (dt.timed("kv_proj", li) if dt else _nullctx()):
            new_compressed_kv = w8a8_deepgemm(
                hidden_fp8, hidden_scale,
                attn.kv_a_proj_with_mqa.weight,
                weight_scale["kv_a_proj_with_mqa.weight_scale_inv"],
            ).view(bsz, 1, -1)

            cos, sin = attn.rotary_emb(q_pe, seq_len=max_seqlen)
            # Explicitly pass the module's RMSNorm eps (config.rms_norm_eps=1e-5
            # for GLM-5). The kernel default is 1e-6; without this override the
            # decode-side new-KV gets normalized with a 10× tighter regularizer
            # than the prefill-cached KV, creating a slow drift that compounds
            # across 78 layers × decode steps and manifests as ngram loops /
            # off-topic generation after ~hundreds of tokens.
            offload_kv = _fused_rmsnorm_rope(
                new_compressed_kv, q_pe, cos, sin, position_ids,
                attn.kv_a_layernorm.weight,
                attn.kv_lora_rank, attn.qk_rope_head_dim,
                eps=attn.kv_a_layernorm.eps,
            )

        # Pre-compute seq_lengths_i32 once (shared by kv_write and indexer_k)
        new_token_pos = position_ids.squeeze(-1)  # [batch]
        manager_device = gpu_paged_kv_manager.device
        seq_lengths_i32 = new_token_pos.to(dtype=torch.int32, device=manager_device)
        current_batch = list(AttnWrapperBase.cur_batch) if AttnWrapperBase.cur_batch else []
        # Read per-step hoisted tensors populated by the worker (see
        # batchgen_worker.py near the _dsa_short_count block). Falls back to
        # the per-layer build only when the hoist wasn't populated — keeps
        # test paths + CUDA-graph capture paths correct. The fallback MUST
        # NOT run under graph capture (torch.tensor from a python list
        # performs HtoD + sync, which capture can't express).
        primary_slot_indices = AttnWrapperBase.primary_slot_indices
        if primary_slot_indices is None:
            raise RuntimeError(
                "GLM-5 DSA requires worker-hoisted primary_slot_indices; "
                "per-layer reconstruction is not a valid production fallback"
            )
        aux_device = gpu_paged_kv_manager_aux.device
        aux_slot_indices = AttnWrapperBase.aux_slot_indices
        if aux_slot_indices is None:
            raise RuntimeError(
                "GLM-5 DSA requires worker-hoisted aux_slot_indices; "
                "per-layer reconstruction is not a valid production fallback"
            )
        if primary_slot_indices.shape != (bsz,) or aux_slot_indices.shape != (bsz,):
            raise RuntimeError(
                f"GLM-5 DSA layer {li}: slot index shape mismatch: "
                f"primary={tuple(primary_slot_indices.shape)}, aux={tuple(aux_slot_indices.shape)}, bsz={bsz}"
            )
        if primary_slot_indices.dtype not in (torch.int32, torch.int64) or aux_slot_indices.dtype not in (torch.int32, torch.int64):
            raise RuntimeError(
                f"GLM-5 DSA layer {li}: slot indices must be int32/int64, "
                f"primary={primary_slot_indices.dtype}, aux={aux_slot_indices.dtype}"
            )
        if primary_slot_indices.device != manager_device or aux_slot_indices.device != aux_device:
            raise RuntimeError(
                f"GLM-5 DSA layer {li}: slot index device mismatch: "
                f"primary={primary_slot_indices.device}/{manager_device}, aux={aux_slot_indices.device}/{aux_device}"
            )

        # Index-OOB diagnostic (BATCHGEN_GLM5_VERIFY_INDICES=1). Logs the
        # bounds relevant to every OOB-able op on the DSA decode path and
        # asserts the pre-conditions. Fires only on layers 0..4 so we catch
        # the first-decode-step crash without spamming every layer. The
        # .item()s force a sync so the assertion traceback points at the
        # true violator instead of a downstream H2D copy.
        import os as _os_verify
        _VERIFY = _os_verify.environ.get("BATCHGEN_GLM5_VERIFY_INDICES", "0") == "1"
        if _VERIFY and self.layer_idx <= 4:
            _rk = AttnWrapperBase.get_rank_safe()
            # Slot indices must index into their respective page tables.
            _prim_pt = gpu_paged_kv_manager._gpu_page_table_manager.gpu_table
            _aux_pt = gpu_paged_kv_manager_aux._gpu_page_table_manager.gpu_table
            _prim_rows = 0 if _prim_pt is None else int(_prim_pt.shape[0])
            _aux_rows = 0 if _aux_pt is None else int(_aux_pt.shape[0])
            _prim_cols = 0 if _prim_pt is None else int(_prim_pt.shape[1])
            _aux_cols = 0 if _aux_pt is None else int(_aux_pt.shape[1])
            _prim_max = int(primary_slot_indices.max().item())
            _prim_min = int(primary_slot_indices.min().item())
            _aux_max = int(aux_slot_indices.max().item())
            _aux_min = int(aux_slot_indices.min().item())
            _cs_max = int(updated_seqlens.max().item()) if 'updated_seqlens' in dir() else int(cache_seqlens.max().item())
            _cs_min = int(cache_seqlens.min().item())
            _pos_max = int(new_token_pos.max().item())
            _pos_min = int(new_token_pos.min().item())
            _prim_pages = int(gpu_paged_kv_manager.config.num_pages)
            _aux_pages = int(gpu_paged_kv_manager_aux.config.num_pages)
            _prim_psz = int(gpu_paged_kv_manager.config.page_size_tokens)
            _aux_psz = int(gpu_paged_kv_manager_aux.config.page_size_tokens)
            logging.warning(
                f"[VERIFY-DSA rank={_rk} L{self.layer_idx} bsz={bsz}] "
                f"prim_slot=[{_prim_min},{_prim_max}] max_rows={_prim_rows} cols={_prim_cols} "
                f"num_pages={_prim_pages} page_sz={_prim_psz} | "
                f"aux_slot=[{_aux_min},{_aux_max}] max_rows={_aux_rows} cols={_aux_cols} "
                f"num_pages={_aux_pages} page_sz={_aux_psz} | "
                f"cache_seq=[{_cs_min},{_cs_max}] pos=[{_pos_min},{_pos_max}] "
                f"max_seqlen={max_seqlen}"
            )
            assert _prim_max < _prim_rows, (
                f"primary_slot_indices.max()={_prim_max} >= page_table.shape[0]={_prim_rows}"
            )
            assert _aux_max < _aux_rows, (
                f"aux_slot_indices.max()={_aux_max} >= aux page_table.shape[0]={_aux_rows}"
            )
            _expected_prim_pages_for_ctx = (max_seqlen + _prim_psz - 1) // _prim_psz
            _expected_aux_pages_for_ctx = (max_seqlen + _aux_psz - 1) // _aux_psz
            if _expected_prim_pages_for_ctx > _prim_cols:
                logging.warning(
                    f"[VERIFY-DSA rank={_rk} L{self.layer_idx}] "
                    f"max_seqlen={max_seqlen} needs {_expected_prim_pages_for_ctx} primary pages "
                    f"but page_table only has {_prim_cols} cols — gather will wrap"
                )
            if _expected_aux_pages_for_ctx > _aux_cols:
                logging.warning(
                    f"[VERIFY-DSA rank={_rk} L{self.layer_idx}] "
                    f"max_seqlen={max_seqlen} needs {_expected_aux_pages_for_ctx} aux pages "
                    f"but aux page_table only has {_aux_cols} cols — gather will wrap"
                )

        # --- Step 1: Write new MLA KV to primary cache ---
        with (dt.timed("kv_write", li) if dt else _nullctx()):
            k_tensor = offload_kv.view(bsz, 1, 1, offload_kv.size(-1))
            if k_tensor.device != manager_device:
                k_tensor = k_tensor.to(manager_device)
            gpu_paged_kv_manager.update_layer_decode_new_token(
                k_tensor=k_tensor,
                v_tensor=None,
                sequence_lengths=seq_lengths_i32,
                layer_idx=li,
                slot_indices=primary_slot_indices,
            )
            if AttnWrapperBase.kv_append_callback is not None:
                AttnWrapperBase.kv_append_callback(li, k_tensor, None)

        # --- Step 2: Write indexer K to auxiliary cache ---
        if gpu_paged_kv_manager_aux is not None:
            with (dt.timed("indexer_k", li) if dt else _nullctx()):
                # WP2: Fused CUDA WGMMA wk_proj (GEMM only) + PyTorch LayerNorm + RoPE + Hadamard
                if self._indexer_cuda_weights is not None:
                    from batchgen_kernels.attention.dsa.fused_indexer_kv_proj_cuda import cuda_wk_proj_gemm_only
                    # CUDA kernel: hidden_states → FP8 act_quant → WGMMA wk_proj
                    k_raw = cuda_wk_proj_gemm_only(
                        hidden_flat,  # [B, hidden_size] — already squeezed above
                        self._indexer_cuda_weights,
                        self._indexer_cuda_module,
                    )  # [B, index_head_dim=128]
                    # LayerNorm (model uses nn.LayerNorm with bias, not RMSNorm)
                    k_normed = indexer.k_norm(k_raw)
                    # Apply RoPE + Hadamard (reuse indexer's fused op)
                    k_normed_3d = k_normed.unsqueeze(1)  # [B, 1, 128]
                    indexer_kv = indexer._fused_rope_hadamard_or_fallback(
                        k_normed_3d, new_token_pos, max_seqlen=max_seqlen,
                    ).unsqueeze(2)  # [B, 1, 1, 128]
                else:
                    _raise_required_dsa_kernel(
                        "WP2 fused indexer KV projection",
                        RuntimeError(f"layer {self.layer_idx} was not initialized"),
                    )
                indexer_k_tensor = indexer_kv  # [batch, 1, 1, index_dim]
                seq_lengths_i32_aux = seq_lengths_i32 if aux_device == manager_device else new_token_pos.to(dtype=torch.int32, device=aux_device)
                gpu_paged_kv_manager_aux.update_layer_decode_new_token(
                    k_tensor=indexer_k_tensor,
                    v_tensor=None,
                    sequence_lengths=seq_lengths_i32_aux,
                    layer_idx=li,
                    slot_indices=aux_slot_indices,
                )
                # Offload indexer K to auxiliary host cache
                if AttnWrapperBase.kv_append_callback_aux is not None:
                    AttnWrapperBase.kv_append_callback_aux(li, indexer_k_tensor, None)

        # --- Step 3: Score all cached tokens (including new), select top-K ---
        with (dt.timed("indexer_score", li) if dt else _nullctx()):
            # cache_seqlens already includes the new token (pre-incremented in worker)
            updated_seqlens = cache_seqlens

            # Per-seq dispatch between the dense short-circuit and the indexer
            # scoring path. Previously dispatched at the batch level using
            # max(cache_seqlens), which dragged short rows (cache_seqlen <=
            # index_topk) through the indexer whenever any row in the batch
            # exceeded index_topk. Now:
            #   - all rows <= index_topk  : dense (topk = max_seqlen)
            #   - all rows >  index_topk  : indexer (topk = index_topk)
            #   - mixed                   : per-row dispatch, unified to
            #     [batch, index_topk] via row-mask assignment.
            _idx_topk = indexer.index_topk
            _short_mask = updated_seqlens <= _idx_topk  # [batch] bool
            _bsz = _short_mask.shape[0]
            # Dispatch hint is precomputed once per decode step by the worker
            # (batchgen_worker.py decode loop) so the per-layer .sum().item()
            # here — which would fire 78x per step — becomes a plain int read.
            _short_count = AttnWrapperBase._dsa_short_count
            if _short_count is None:
                raise RuntimeError(
                    "GLM-5 DSA requires worker-hoisted _dsa_short_count; "
                    "per-layer recomputation is not a valid production fallback"
                )
            if _short_count < 0 or _short_count > _bsz:
                raise RuntimeError(
                    f"GLM-5 DSA layer {li}: _dsa_short_count out of range: "
                    f"{_short_count} for bsz={_bsz}"
                )
            if _VERIFY and self.layer_idx <= 4:
                _actual_short_count = int(_short_mask.sum().item())
                if _actual_short_count != _short_count:
                    raise RuntimeError(
                        f"GLM-5 DSA layer {li}: _dsa_short_count mismatch: "
                        f"hoisted={_short_count}, actual={_actual_short_count}, bsz={_bsz}"
                    )
            _any_short = _short_count > 0
            _any_long = _short_count < _bsz
            if not _any_long:
                # All rows <= index_topk: dense short-circuit (topk = max_seqlen).
                top_k_indices = build_clamped_dense_token_indices(
                    updated_seqlens, max_seqlen, hidden_states.device,
                )
                validate_token_indices_within_seqlens(
                    top_k_indices,
                    updated_seqlens,
                    context=f"GLM-5 DSA L{li}/dense_topk",
                    effective_lengths=torch.clamp(updated_seqlens, max=top_k_indices.shape[1]),
                    debug_sync=_VERIFY and self.layer_idx <= 4,
                )
            elif not _any_short:
                # All rows > index_topk: full-batch indexer scoring (topk = index_topk).
                q_a_for_indexer = q_a_normed.unsqueeze(1)
                indexer_blocked_k, _, idx_block_table = \
                    gpu_paged_kv_manager_aux.get_layer_kv_with_page_table(li)
                idx_block_table = reorder_block_table_to_batch_slots(
                    idx_block_table, aux_slot_indices,
                )
                aux_page_size = gpu_paged_kv_manager_aux.config.page_size_tokens
                top_k_indices = indexer.score_and_select_paged(
                    q_a_for_indexer, hidden_states,
                    indexer_blocked_k, idx_block_table,
                    updated_seqlens, gpu_paged_kv_manager_aux, aux_page_size,
                    positions=new_token_pos,
                    max_seqlen=max_seqlen,
                )
            else:
                # Mixed batch: per-seq dispatch, unified to [batch, index_topk].
                # This branch uses a per-row dispatch with a host-sync on
                # `_long_cache_seqlens.max().item()` below — not capture-safe.
                # The scheduler must route mixed batches to the eager fallback
                # variant (see plan §5: Glm5BatchDescriptor dsa_variant="eager").
                assert not torch.cuda.is_current_stream_capturing(), (
                    "Mixed DSA batch (some rows <= index_topk, some >) is "
                    "not capture-safe; dispatch to eager fallback before "
                    "graph.replay()."
                )
                _device = hidden_states.device
                _long_mask = ~_short_mask
                top_k_indices = torch.empty(
                    _bsz, _idx_topk, dtype=torch.long, device=_device,
                )
                # Short rows: dense indices padded to index_topk via clamp.
                _short_indices = build_clamped_dense_token_indices(
                    updated_seqlens[_short_mask], _idx_topk, _device,
                )  # [num_short, index_topk]
                top_k_indices[_short_mask] = _short_indices
                # Long rows: indexer scoring on the subset.
                _long_cache_seqlens = updated_seqlens[_long_mask]
                _long_max_seqlen = int(_long_cache_seqlens.max().item())
                _q_a_long = q_a_normed[_long_mask].unsqueeze(1)
                _hidden_long = hidden_states[_long_mask]
                _new_token_pos_long = (
                    new_token_pos[_long_mask] if new_token_pos is not None else None
                )
                _long_mask_aux = _long_mask.to(aux_slot_indices.device)
                _aux_slot_indices_long = aux_slot_indices[_long_mask_aux]
                indexer_blocked_k, _, idx_block_table = \
                    gpu_paged_kv_manager_aux.get_layer_kv_with_page_table(li)
                _idx_block_table_long = reorder_block_table_to_batch_slots(
                    idx_block_table, _aux_slot_indices_long,
                )
                aux_page_size = gpu_paged_kv_manager_aux.config.page_size_tokens
                _long_top_k = indexer.score_and_select_paged(
                    _q_a_long, _hidden_long,
                    indexer_blocked_k, _idx_block_table_long,
                    _long_cache_seqlens, gpu_paged_kv_manager_aux, aux_page_size,
                    positions=_new_token_pos_long,
                    max_seqlen=_long_max_seqlen,
                )  # [num_long, min(index_topk, _long_max_seqlen)] = [num_long, index_topk]
                top_k_indices[_long_mask] = _long_top_k
                validate_token_indices_within_seqlens(
                    top_k_indices,
                    updated_seqlens,
                    context=f"GLM-5 DSA L{li}/mixed_topk",
                    effective_lengths=torch.clamp(updated_seqlens, max=top_k_indices.shape[1]),
                    debug_sync=_VERIFY and self.layer_idx <= 4,
                )

            # Sparse seqlens define which top-k columns are semantically consumed
            # by sparse_flash_mla_decode. Padded tail entries may be clamped, but
            # every effective entry must map to a valid cache token and page.
            sparse_seqlens = torch.clamp(updated_seqlens, max=top_k_indices.shape[1])
            validate_token_indices_within_seqlens(
                top_k_indices,
                updated_seqlens,
                context=f"GLM-5 DSA L{li}/effective_topk",
                effective_lengths=sparse_seqlens,
                debug_sync=_VERIFY and self.layer_idx <= 4,
            )

        # --- Step 4: Sparse gather MLA KV at top-K positions ---
        with (dt.timed("sparse_gather", li) if dt else _nullctx()):
            mla_blocked_k, _, mla_block_table = \
                gpu_paged_kv_manager.get_layer_kv_with_page_table(li)
            mla_block_table = reorder_block_table_to_batch_slots(
                mla_block_table, primary_slot_indices,
            )
            mla_page_size = gpu_paged_kv_manager.config.page_size_tokens

            validate_effective_block_table_pages(
                mla_block_table,
                top_k_indices,
                sparse_seqlens,
                mla_page_size,
                context=f"GLM-5 DSA L{li}/mla_sparse_gather",
                debug_sync=_VERIFY and self.layer_idx <= 4,
            )

            if _VERIFY and self.layer_idx <= 4:
                _rk = AttnWrapperBase.get_rank_safe()
                _tk_max = int(top_k_indices.max().item())
                _tk_min = int(top_k_indices.min().item())
                _tk_shape = tuple(top_k_indices.shape)
                _bt_shape = tuple(mla_block_table.shape)
                _bk_shape = tuple(mla_blocked_k.shape)  # [num_pages, page_sz, heads, dim]
                _num_pages_loaded = _bk_shape[0]
                _max_flat_idx = _num_pages_loaded * mla_page_size
                _expected_max_valid_tok = _bt_shape[1] * mla_page_size
                logging.warning(
                    f"[VERIFY-GATHER rank={_rk} L{self.layer_idx} bsz={bsz}] "
                    f"top_k=[{_tk_min},{_tk_max}] shape={_tk_shape} | "
                    f"mla_block_table.shape={_bt_shape} mla_blocked_k.shape={_bk_shape} "
                    f"page_size={mla_page_size} "
                    f"max_flat_idx_if_clean={_max_flat_idx} "
                    f"max_tok_pos_in_bt_cols={_expected_max_valid_tok} | "
                    f"which branch: "
                    f"{'dense-short-circuit' if not _any_long else ('full-indexer' if not _any_short else 'mixed')}"
                )
                if _tk_max >= _expected_max_valid_tok:
                    raise RuntimeError(
                        f"[VERIFY-GATHER rank={_rk} L{self.layer_idx}] top_k_indices "
                        f"contains position {_tk_max} >= block_table_cols*page_size="
                        f"{_expected_max_valid_tok}; refusing sparse gather"
                    )
            sparse_mla_kv = sparse_gather_from_paged_kv(
                mla_blocked_k, mla_block_table, top_k_indices, mla_page_size,
            )
            # sparse_mla_kv: [batch, topk, 1, 576]

        # --- Step 5: Absorbed Q → sparse FlashMLA ---
        with (dt.timed("q_absorb", li) if dt else _nullctx()):
            if self._cached_q_absorb is not None:
                q_absorb = self._cached_q_absorb
                out_absorb = self._cached_out_absorb
            else:
                kv_b_proj = deepseek_v3_dequantization(
                    attn.kv_b_proj.weight.data,
                    weight_scale["kv_b_proj.weight_scale_inv"],
                ).view(attn.num_heads, -1, attn.kv_lora_rank)
                q_absorb = kv_b_proj[:, :attn.qk_nope_head_dim, :].contiguous()
                out_absorb = kv_b_proj[:, attn.qk_nope_head_dim:, :].contiguous()
                # JIT SGLang-layout weights for the BMM fallback path.
                if self.w_kc is None:
                    self.w_kc = q_absorb.transpose(1, 2).contiguous().transpose(1, 2)
                    self.w_vc = out_absorb.contiguous().transpose(1, 2)

            qk_head_dim = attn.kv_lora_rank + attn.qk_rope_head_dim
            query_states = torch.empty(
                bsz, attn.num_heads, 1, qk_head_dim,
                dtype=sparse_mla_kv.dtype, device=sparse_mla_kv.device,
            )
            q_nope_squeezed = q_nope.squeeze(2)  # [B, H, qk_nope_head_dim]

            # WP5: FP8 absorb kernel or SGLang-aligned BF16 BMM fallback
            if self._fp8_absorb_weights is not None:
                absorbed_q = fp8_q_absorb(q_nope_squeezed, self._fp8_absorb_weights)
                query_states[:, :, :, :attn.kv_lora_rank] = absorbed_q.view(
                    bsz, attn.num_heads, 1, attn.kv_lora_rank
                )
            else:
                _raise_required_dsa_kernel(
                    "WP5 FP8 q_absorb",
                    RuntimeError(f"layer {self.layer_idx} was not initialized"),
                )

            query_states[:, :, :, attn.kv_lora_rank:] = q_pe
            query_states = query_states.view(bsz, 1, attn.num_heads, qk_head_dim)

        with (dt.timed("sparse_attn", li) if dt else _nullctx()):
            attn_out = sparse_flash_mla_decode(
                query_states, sparse_mla_kv, sparse_seqlens,
                attn.num_heads, attn.softmax_scale,
                head_dim_v=attn.kv_lora_rank,
                page_size=mla_page_size,
            )

        # --- Step 6: out_absorb → o_proj ---
        with (dt.timed("o_proj", li) if dt else _nullctx()):
            # WP5: FP8 out_absorb kernel or SGLang-aligned BF16 BMM fallback
            if self._fp8_absorb_weights is not None:
                attn_output = fp8_out_absorb(attn_out, self._fp8_absorb_weights)
            else:
                _raise_required_dsa_kernel(
                    "WP5 FP8 out_absorb",
                    RuntimeError(f"layer {self.layer_idx} was not initialized"),
                )
            attn_output = attn_output.reshape(bsz, attn.num_heads * attn.v_head_dim)
            attn_output_fp8, attn_output_scale = act_quant(attn_output)
            attn_output = w8a8_deepgemm(
                attn_output_fp8, attn_output_scale,
                attn.o_proj.weight, weight_scale["o_proj.weight_scale_inv"],
            )
            attn_output = attn_output.view(bsz, 1, -1)

        return attn_output
