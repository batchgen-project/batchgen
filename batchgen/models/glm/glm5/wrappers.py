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
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

import torch.nn.functional as F

from batchgen.models.wrappers import ExpertWrapperBase, AttnWrapperBase
from batchgen.timing import init_decode_timer

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

_GLM5_DSA_CUDA_GRAPH_ENV = "BATCHGEN_GLM5_DSA_CUDA_GRAPH"
_glm5_dsa_graph_eager_fallback_logged = False


def _glm5_dsa_cuda_graph_required() -> bool:
    return os.environ.get(_GLM5_DSA_CUDA_GRAPH_ENV, "0") == "1"


def _glm5_dsa_cuda_graph_can_replay(
    cache_seqlens: torch.Tensor,
    max_seqlen: int,
    index_topk: int,
    *,
    captured_max_seqlen: Optional[int] = None,
) -> bool:
    """The current graph segment is valid only for all-long DSA rows.

    `Glm5DsaAttnSegment` captures FlashMLA metadata for `index_topk` selected
    tokens. Short rows need per-step metadata for their actual selected length,
    so they must stay on the eager path until length-bucketed metadata is added.
    """

    if max_seqlen <= index_topk:
        return False
    if captured_max_seqlen is not None and max_seqlen > captured_max_seqlen:
        return False
    return bool((cache_seqlens > index_topk).all().item())


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


def _fail_if_glm5_dsa_cuda_graph_required_without_replay() -> None:
    if not _glm5_dsa_cuda_graph_required():
        return
    raise RuntimeError(
        f"{_GLM5_DSA_CUDA_GRAPH_ENV}=1 requested GLM-5 DSA CUDA graph replay, "
        "but production hidden-state-to-o_proj graph routing is not wired in this "
        "integration slice. Refusing to silently fall back to eager DSA decode."
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
        self._dsa_cuda_graph_manager = None
        self._dsa_cuda_graph_segment_name = None
        self._dsa_cuda_graph_max_seqlen = 0
        self._dsa_cuda_graph_max_primary_pages = 0
        self._dsa_cuda_graph_max_aux_pages = 0

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
        max_primary_pages_per_seq: int,
        max_aux_pages_per_seq: int,
    ) -> None:
        self._dsa_cuda_graph_manager = manager
        self._dsa_cuda_graph_segment_name = segment_name
        self._dsa_cuda_graph_max_seqlen = max_seqlen
        self._dsa_cuda_graph_max_primary_pages = max_primary_pages_per_seq
        self._dsa_cuda_graph_max_aux_pages = max_aux_pages_per_seq

    def _forward_prefill(self, hidden_states: torch.Tensor, **kwargs) -> Tuple:
        """Prefill forward with DSA auxiliary cache population.

        1. Standard MLA prefill via FA3 (full attention)
        2. Compute indexer K and write to auxiliary cache
        """
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
            indexer_kv = self.module.indexer.compute_indexer_kv(
                hidden_states_2d.unsqueeze(0),
                positions=self.position_ids.to(hidden_states_2d.device),
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
        cu_seqlens = self.prepack_cu_seqlens
        num_sequences = self.prepack_num_sequences
        global_sequence_ids = self.cur_batch

        # Lifespan management mirrored from decode-side `_pending_kv_append_*`
        # (worker.py:1898-1925). Drain compute stream via a CUDA event so the
        # FA3 prefill kernel that wrote `offload_kv` has fully retired before
        # the C++ async lambda's d2h memcpy reads the source memory; pin the
        # source tensor (and the parent `offload_kv`) in the class-level list
        # so PyTorch's caching allocator cannot re-hand the same physical
        # pages to a later layer's K/V tensor while the d2h is in flight.
        AttnWrapperBase.pending_prefill_offload_tensors.append(offload_kv)
        evt = torch.cuda.Event()
        evt.record(torch.cuda.current_stream())
        evt.synchronize()

        # Single D2H sync for all seq boundaries instead of 2N per-seq .item() calls.
        cu = cu_seqlens.tolist()
        for seq_idx in range(num_sequences):
            start_idx = cu[seq_idx]
            end_idx = cu[seq_idx + 1]
            seq_len = end_idx - start_idx
            # indexer_kv is already [T, H=1, D=128] after caller's .squeeze(0),
            # so only .unsqueeze(0) is needed to add the B dim; don't also
            # .unsqueeze(2) (that would make 5D — the primary-MLA path copy-paste
            # of this code was for a 2D [T, kv_lora+rope] input).
            seq_kv = offload_kv[start_idx:end_idx].unsqueeze(0)
            seq_global_id = [global_sequence_ids[seq_idx]]
            task = AttnWrapperBase.host_paged_kv_worker_view_aux.async_offload_layer_kv_to_host(
                layer_idx=self.layer_idx,
                sequence_ids=seq_global_id,
                k_tensor=seq_kv,
                v_tensor=None,
                sequence_lengths=[seq_len],
            )
            # Pin both the per-seq view AND the parent offload_kv (already
            # pinned outside the loop) so neither's storage is reclaimed.
            AttnWrapperBase.pending_prefill_offload_tensors.append(seq_kv)
            if task is not None:
                AttnWrapperBase.pending_prefill_offload_tasks.append(task)

    def _offload_prepacked_kv(self, offload_kv: torch.Tensor):
        """Offload KV cache per-sequence to host memory."""
        cu_seqlens = self.prepack_cu_seqlens
        num_sequences = self.prepack_num_sequences
        global_sequence_ids = self.cur_batch

        # See _offload_prepacked_indexer_kv for rationale.
        AttnWrapperBase.pending_prefill_offload_tensors.append(offload_kv)
        evt = torch.cuda.Event()
        evt.record(torch.cuda.current_stream())
        evt.synchronize()

        # Single D2H sync for all seq boundaries instead of 2N per-seq .item() calls.
        cu = cu_seqlens.tolist()
        for seq_idx in range(num_sequences):
            start_idx = cu[seq_idx]
            end_idx = cu[seq_idx + 1]
            seq_len = end_idx - start_idx
            seq_kv = offload_kv[start_idx:end_idx].unsqueeze(0).unsqueeze(2)
            seq_global_id = [global_sequence_ids[seq_idx]]
            task = self.core_engine.host_paged_kv_worker_view.async_offload_layer_kv_to_host(
                layer_idx=self.layer_idx,
                sequence_ids=seq_global_id,
                k_tensor=seq_kv,
                v_tensor=None,
                sequence_lengths=[seq_len],
            )
            AttnWrapperBase.pending_prefill_offload_tensors.append(seq_kv)
            if task is not None:
                AttnWrapperBase.pending_prefill_offload_tasks.append(task)

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
                f"[layer {self.layer_idx}] BATCHGEN_GLM5_DSA_CUDA_GRAPH=1 was requested, "
                "but this attention wrapper has no registered DSA CUDA graph segment"
            )
        if max_seqlen > self._dsa_cuda_graph_max_seqlen:
            raise RuntimeError(
                f"[layer {self.layer_idx}] GLM-5 DSA CUDA graph max_seqlen={max_seqlen} "
                f"exceeds captured cap {self._dsa_cuda_graph_max_seqlen}"
            )
        index_topk = getattr(getattr(self.module, "indexer", None), "index_topk", 2048)
        if not _glm5_dsa_cuda_graph_can_replay(cache_seqlens, max_seqlen, index_topk):
            raise RuntimeError(
                f"[layer {self.layer_idx}] GLM-5 DSA CUDA graph replay is only "
                f"valid when every row has cache_seqlens > index_topk={index_topk}; "
                "short or mixed rows require eager DSA FlashMLA metadata"
            )

        from batchgen.attention.dsa.glm5_decode_selector import (
            build_glm5_dsa_graph_segment_inputs,
        )
        from batchgen.attention.mla.fa3_backend import act_quant
        from batchgen.gemm.w8a8_deepgemm import w8a8_deepgemm

        attn = self.module
        bsz = hidden_states.shape[0]
        graph_inputs = build_glm5_dsa_graph_segment_inputs(
            self,
            hidden_states,
            position_ids,
            cache_seqlens,
            max_seqlen,
            gpu_paged_kv_manager,
            gpu_paged_kv_manager_aux,
            max_primary_pages_per_seq=self._dsa_cuda_graph_max_primary_pages,
            max_aux_pages_per_seq=self._dsa_cuda_graph_max_aux_pages,
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
            primary_page_table=graph_inputs.primary_page_table,
            aux_page_table=graph_inputs.aux_page_table,
        )

        attn_heads = graph_outputs["attn_heads"].reshape(
            bsz,
            attn.num_heads * attn.v_head_dim,
        )
        attn_output_fp8, attn_output_scale = act_quant(attn_heads)
        attn_output = w8a8_deepgemm(
            attn_output_fp8,
            attn_output_scale,
            attn.o_proj.weight,
            self.weight_dequant_scale["o_proj.weight_scale_inv"],
        )
        return attn_output.view(bsz, 1, -1)

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
        li = self.layer_idx

        # Handle empty batch (some DP ranks have 0 sequences at late decode stages)
        if bsz == 0:
            return hidden_states.new_empty(0, 1, attn.hidden_size)

        if _glm5_dsa_cuda_graph_required():
            index_topk = getattr(getattr(attn, "indexer", None), "index_topk", 2048)
            graph_can_replay = _glm5_dsa_cuda_graph_can_replay(
                cache_seqlens,
                max_seqlen,
                index_topk,
                captured_max_seqlen=self._dsa_cuda_graph_max_seqlen,
            )
            graph_has_bucket = (
                self._dsa_cuda_graph_manager is not None
                and self._dsa_cuda_graph_segment_name is not None
                and self._dsa_cuda_graph_manager.has_graph(
                    self._dsa_cuda_graph_segment_name,
                    bsz,
                )
            )
            if graph_can_replay and graph_has_bucket:
                return self._forward_decode_dsa_graph(
                    hidden_states,
                    position_ids,
                    cache_seqlens,
                    max_seqlen,
                    gpu_paged_kv_manager,
                    gpu_paged_kv_manager_aux,
                )
            if not graph_can_replay:
                if max_seqlen > self._dsa_cuda_graph_max_seqlen:
                    reason = (
                        f"max_seqlen={max_seqlen} exceeds captured cap "
                        f"{self._dsa_cuda_graph_max_seqlen}"
                    )
                else:
                    reason = "current decode rows are not all longer than index_topk"
            else:
                reason = f"batch size {bsz} has no captured graph bucket"
            _log_glm5_dsa_graph_eager_fallback_once(
                self.layer_idx,
                cache_seqlens,
                index_topk,
                reason,
            )

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
        )
        with (dt.timed("sparse_attn", li) if dt else _nullctx()):
            attn_out = run_prepared_sparse_flash_mla_decode(selector_inputs.flashmla)

        # --- Step 6: out_absorb → o_proj ---
        with (dt.timed("o_proj", li) if dt else _nullctx()):
            # WP5: FP8 out_absorb kernel or SGLang-aligned BF16 BMM fallback
            if self._fp8_absorb_weights is not None:
                attn_output = fp8_out_absorb(attn_out, self._fp8_absorb_weights)
            else:
                # SGLang forward_mla.py:548 — bmm(attn_output.T, w_vc).
                attn_out_3d = attn_out.squeeze(1)  # [B, H, 512]
                attn_output = torch.bmm(
                    attn_out_3d.transpose(0, 1), self.w_vc,
                ).transpose(0, 1).unsqueeze(1)  # [B, 1, H, v_head_dim]
            attn_output = attn_output.reshape(bsz, attn.num_heads * attn.v_head_dim)
            attn_output_fp8, attn_output_scale = act_quant(attn_output)
            attn_output = w8a8_deepgemm(
                attn_output_fp8, attn_output_scale,
                attn.o_proj.weight, weight_scale["o_proj.weight_scale_inv"],
            )
            attn_output = attn_output.view(bsz, 1, -1)

        return attn_output
