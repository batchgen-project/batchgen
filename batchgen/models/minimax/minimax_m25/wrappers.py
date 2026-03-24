# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
# ---------------------------------------------------------------------------- #

"""MiniMax-M2.5 wrappers for BatchGen execution.

Provides wrappers for MiniMax-M2.5 with FP8 quantization:
- MiniMaxM25ExpertWrapper: Expert wrapper with FP8 block-wise dequantization
- MiniMaxM25AttnWrapper: Attention wrapper with GQA + QK norm + partial RoPE

All weights (attention + experts) are FP8 e4m3fn with F32 block-wise scales.

Core engine tensor keys (from parameter_server.py):
  Expert: w1.weight, w1.weight_scale_inv  (gate_proj)
          w2.weight, w2.weight_scale_inv  (down_proj)
          w3.weight, w3.weight_scale_inv  (up_proj)
  Attn:   q_proj.weight, q_proj.weight_scale_inv,
          k_proj.weight, k_proj.weight_scale_inv,
          v_proj.weight, v_proj.weight_scale_inv,
          o_proj.weight, o_proj.weight_scale_inv,
          q_norm.weight, k_norm.weight
"""

import logging
import math
import os
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from batchgen.models.wrappers import ExpertWrapperBase, AttnWrapperBase
from batchgen.quantization.fp8e4m3 import deepseek_v3_dequantization
from .model import rotate_half


# ============================================================================
# Per-sub-op decode timing (BATCHGEN_DECODE_TIMING=1)
# ============================================================================

_DECODE_TIMING = os.environ.get("BATCHGEN_DECODE_TIMING", "0") == "1"
_DECODE_TIMING_INTERVAL = int(os.environ.get("BATCHGEN_DECODE_TIMING_INTERVAL", "50"))


class DecodeTimingStats:
    """Accumulates per-sub-op timing across decode steps. Logs every N steps."""
    _step = 0
    _accum = {}  # op_name -> total_us

    @classmethod
    def _sync_record(cls, op: str, t0: float):
        torch.cuda.synchronize()
        elapsed_us = (time.perf_counter() - t0) * 1e6
        cls._accum[op] = cls._accum.get(op, 0.0) + elapsed_us

    @classmethod
    def step_done(cls):
        cls._step += 1
        if cls._step % _DECODE_TIMING_INTERVAL != 0:
            return
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != 0:
            cls._accum.clear()
            return
        n = _DECODE_TIMING_INTERVAL
        lines = [f"\n=== MiniMax Decode Timing (avg over {n} steps, step {cls._step}) ==="]
        total = sum(cls._accum.values())
        for op, us in sorted(cls._accum.items(), key=lambda x: -x[1]):
            pct = 100 * us / max(total, 1)
            lines.append(f"  {op:20s}: {us/n:8.1f} us/step ({pct:5.1f}%)")
        lines.append(f"  {'TOTAL':20s}: {total/n:8.1f} us/step")
        logging.info("\n".join(lines))
        cls._accum.clear()

try:
    from batchgen.attention.mla.fa3_backend import w8a16_gemm
    _HAS_W8A16 = True
except ImportError:
    _HAS_W8A16 = False

# CUDA RoPE kernel (partial rotation with passthrough copy)
try:
    from batchgen_kernels.attention._C_fused_ops import rope_forward as _rope_forward_cuda
    _HAS_CUDA_ROPE = True
except ImportError:
    _HAS_CUDA_ROPE = False
    logging.warning("[RoPE] CUDA rope_forward not available — will use PyTorch fallback")


def _cuda_rope_forward(query, key, cos, sin, half_dim):
    """CUDA RoPE with partial rotation passthrough.

    Args:
        query: [B, S=1, num_q_heads, head_dim]
        key:   [B, S=1, num_kv_heads, head_dim]
        cos:   [B, S=1, head_dim] (padded to head_dim)
        sin:   [B, S=1, head_dim]
        half_dim: rotary_dim // 2

    Returns:
        (q_rotated, k_rotated) same shapes as input
    """
    return _rope_forward_cuda(query, key, cos, sin, half_dim, None)


def _fp8_linear(weight_fp8, scale, x):
    """FP8 weight × BF16 activation. Uses CUDA act_quant_3d(E=1) if available."""
    if _HAS_CUDA_ACT_QUANT:
        return _fp8_linear_cuda_quant(weight_fp8, scale, x)
    return w8a16_gemm(weight_fp8, scale, x)


# CUDA act_quant for attention path (avoid Triton overhead on small M)
try:
    from batchgen_kernels.moe._C_fp8_blockwise_ops import act_quant_3d as _cuda_act_quant_3d
    import deep_gemm
    _HAS_CUDA_ACT_QUANT = True
except ImportError:
    _HAS_CUDA_ACT_QUANT = False
    logging.warning("[Attn] CUDA act_quant_3d not available — using Triton act_quant in w8a16_gemm")

_warned_cuda_act_quant = False


def _fp8_linear_cuda_quant(weight_fp8, scale, x):
    """FP8 linear with CUDA act_quant_3d(E=1) replacing Triton act_quant."""
    global _warned_cuda_act_quant
    if not _warned_cuda_act_quant:
        logging.warning("[Attn] HOT PATH: _fp8_linear with CUDA act_quant_3d(E=1)")
        _warned_cuda_act_quant = True

    orig_shape = x.shape
    x_2d = x.view(-1, x.size(-1))  # [M, K]
    M, K = x_2d.shape
    N = weight_fp8.size(0)

    # CUDA act_quant_3d with E=1
    tokens_per_expert = torch.tensor([M], dtype=torch.int32, device=x.device)
    x_3d = x_2d.view(1, M, K)
    x_fp8_3d, x_scale_3d = _cuda_act_quant_3d(x_3d, tokens_per_expert)
    x_fp8 = x_fp8_3d.view(M, K)
    x_scale = x_scale_3d.view(M, -1)

    # DeepGEMM FP8×FP8→BF16
    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    deep_gemm.fp8_gemm_nt(
        (x_fp8.view(torch.float8_e4m3fn), x_scale),
        (weight_fp8, scale),
        out, disable_ue8m0_cast=True,
    )

    if len(orig_shape) == 3:
        out = out.view(orig_shape[0], orig_shape[1], N)
    return out


class MiniMaxM25ExpertWrapper(ExpertWrapperBase):
    """Expert wrapper with FP8 dequantization for MiniMax-M2.5.

    Wraps _ExpertPlaceholder (not a real nn.Module). All computation is done
    directly in this wrapper using w8a16_gemm (FP8 weight × BF16 activation).

    Core engine tensor keys: w1 (gate_proj), w2 (down_proj), w3 (up_proj).
    Activation: standard SwiGLU — silu(w1(x)) * w3(x) → w2(intermediate).
    """

    def __init__(self, module, layer_idx, expert_idx, core_engine, engine_config,
                 model_config, persistent=False, weight_dequant_scale=None):
        super().__init__(
            module, layer_idx, expert_idx, core_engine, engine_config, model_config,
            persistent
        )
        self.weight_dequant_scale = weight_dequant_scale or {}
        # Cached FP8 weight data (set by _register_fp8_weights for persistent experts)
        self.fp8_gate = None   # w1.weight
        self.fp8_up = None     # w3.weight
        self.fp8_down = None   # w2.weight

    def dequantize_weights(self, weights_dict):
        """Dequantize FP8 weights using block-wise scale factors."""
        result = {}
        for name, weight in weights_dict.items():
            scale_key = f"{name}_scale_inv"
            if scale_key in self.weight_dequant_scale:
                result[name] = deepseek_v3_dequantization(
                    weight, self.weight_dequant_scale[scale_key]
                )
            else:
                result[name] = weight
        return result

    def clear_weights(self):
        """No-op: _ExpertPlaceholder has no nn.Module parameters to clear."""
        pass

    def _register_fp8_weights(self, tensors):
        """Cache FP8 weights and scales from core_engine tensors.

        Called by PSM._load_local_routed_experts() after wrapping.
        tensors keys: w1.weight, w1.weight_scale_inv, w2.weight, etc.
        """
        self.fp8_gate = tensors["w1.weight"]
        self.fp8_up = tensors["w3.weight"]
        self.fp8_down = tensors["w2.weight"]
        self.weight_dequant_scale = {
            "w1.weight_scale_inv": tensors["w1.weight_scale_inv"],
            "w2.weight_scale_inv": tensors["w2.weight_scale_inv"],
            "w3.weight_scale_inv": tensors["w3.weight_scale_inv"],
        }

    def _unregister_fp8_weights(self):
        self.fp8_gate = None
        self.fp8_up = None
        self.fp8_down = None

    def forward(self, hidden_states):
        """Forward: FP8 weight × BF16 activation via w8a16_gemm.

        Non-persistent: load weights from core_engine on demand.
        Persistent: use cached FP8 weights.
        SwiGLU: silu(gate(x)) * up(x) → down(intermediate)
        """
        if not self.persistent:
            tensors = self.load_weights(self.module_key)
            fp8_gate = tensors["w1.weight"]
            fp8_up = tensors["w3.weight"]
            fp8_down = tensors["w2.weight"]
            gate_scale = tensors["w1.weight_scale_inv"]
            up_scale = tensors["w3.weight_scale_inv"]
            down_scale = tensors["w2.weight_scale_inv"]
        else:
            fp8_gate = self.fp8_gate
            fp8_up = self.fp8_up
            fp8_down = self.fp8_down
            gate_scale = self.weight_dequant_scale["w1.weight_scale_inv"]
            up_scale = self.weight_dequant_scale["w3.weight_scale_inv"]
            down_scale = self.weight_dequant_scale["w2.weight_scale_inv"]

        # DeepGEMM requires float32 scale factors
        gate_scale = gate_scale.float()
        up_scale = up_scale.float()
        down_scale = down_scale.float()

        # w8a16: FP8 weight × BF16 activation
        gate = w8a16_gemm(fp8_gate, gate_scale, hidden_states)
        up = w8a16_gemm(fp8_up, up_scale, hidden_states)
        intermediate = F.silu(gate) * up
        result = w8a16_gemm(fp8_down, down_scale, intermediate)

        if not self.persistent:
            torch.cuda.current_stream(
                self.engine_config.Basic_Config.device_torch
            ).synchronize()
            self.free_weights(self.module_key)
            self.clear_weights()

        return result


class MiniMaxM25AttnWrapper(AttnWrapperBase):
    """Attention wrapper for MiniMax-M2.5 GQA with QK norm + partial RoPE.

    M2.5 attention projections are FP8 (same as experts). QK norms are BF16.
    Uses w8a16_gemm for Q/K/V/O projections (act_quant → DeepGEMM).
    """

    def __init__(self, module, layer_idx, core_engine, engine_config,
                 model_config, persistent=True, weight_dequant_scale=None):
        super().__init__(
            module, layer_idx, core_engine, engine_config, model_config,
            persistent, weight_dequant_scale
        )
        # FP8 attention weights + scales (set by _register_fp8_weights)
        self.fp8_q = None
        self.fp8_k = None
        self.fp8_v = None
        self.fp8_o = None
        self.q_scale = None
        self.k_scale = None
        self.v_scale = None
        self.o_scale = None

    def _register_fp8_weights(self, tensors):
        """Cache FP8 attention weights and scales."""
        device = self.engine_config.Basic_Config.device_torch
        self.fp8_q = tensors["q_proj.weight"].to(device)
        self.fp8_k = tensors["k_proj.weight"].to(device)
        self.fp8_v = tensors["v_proj.weight"].to(device)
        self.fp8_o = tensors["o_proj.weight"].to(device)
        self.q_scale = tensors["q_proj.weight_scale_inv"].to(device)
        self.k_scale = tensors["k_proj.weight_scale_inv"].to(device)
        self.v_scale = tensors["v_proj.weight_scale_inv"].to(device)
        self.o_scale = tensors["o_proj.weight_scale_inv"].to(device)
        # QK norms (BF16) — keep on module
        self.module.q_norm.weight.data = tensors["q_norm.weight"].to(device)
        self.module.k_norm.weight.data = tensors["k_norm.weight"].to(device)

    def _unregister_fp8_weights(self):
        self.fp8_q = None
        self.fp8_k = None
        self.fp8_v = None
        self.fp8_o = None
        self.q_scale = None
        self.k_scale = None
        self.v_scale = None
        self.o_scale = None

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward with FP8-specific weight lifecycle.

        Overrides AttnWrapperBase.forward() to avoid double load/free.
        The base class loads weights via apply_weights (for BF16 nn.Module params),
        but FP8 attention loads weights directly in _forward_prefill/_forward_decode
        via _get_attn_weights(). Using both paths causes double load_weights +
        double free_weights on the same module_key, crashing the C++ buffer system.
        """
        hidden_states = kwargs.pop("hidden_states", None)
        if self.phase == "prefill":
            result = self._forward_prefill(hidden_states, **kwargs)
        else:
            result = self._forward_decode(hidden_states, **kwargs)
        return result

    def _get_attn_weights(self):
        """Get FP8 weights + scales. Persistent: cached. Non-persistent: load from core_engine."""
        if self.persistent:
            return (self.fp8_q, self.q_scale, self.fp8_k, self.k_scale,
                    self.fp8_v, self.v_scale, self.fp8_o, self.o_scale)
        else:
            tensors = self.load_weights(self.module_key)
            # Also load QK norms into self.module (not returned as FP8 weights)
            self.module.q_norm.weight.data = tensors["q_norm.weight"]
            self.module.k_norm.weight.data = tensors["k_norm.weight"]
            return (tensors["q_proj.weight"], tensors["q_proj.weight_scale_inv"],
                    tensors["k_proj.weight"], tensors["k_proj.weight_scale_inv"],
                    tensors["v_proj.weight"], tensors["v_proj.weight_scale_inv"],
                    tensors["o_proj.weight"], tensors["o_proj.weight_scale_inv"])

    def dequantize_weights(self, weights_dict):
        """Not used — we use w8a16_gemm directly."""
        return weights_dict

    def _forward_prefill(self, hidden_states, **kwargs):
        """Prefill forward: FP8 Q/K/V projection + QK norm + partial RoPE + FA varlen."""
        from batchgen.attention.gqa import gqa_prefill_fa
        from batchgen.attention.fused_kernels import cuda_rmsnorm

        fp8_q, q_scale, fp8_k, k_scale, fp8_v, v_scale, fp8_o, o_scale = self._get_attn_weights()

        # Always prepack mode in production prefill
        if hidden_states.dim() == 3:
            assert hidden_states.shape[0] == 1, "Prepack mode expects batch_size=1"
            hidden_states_2d = hidden_states.squeeze(0)
        else:
            hidden_states_2d = hidden_states

        total_tokens = hidden_states_2d.shape[0]
        cu_seqlens = self.prepack_cu_seqlens.to(hidden_states_2d.device)
        max_seqlen = self.prepack_max_seqlen
        position_ids = self.position_ids.to(hidden_states_2d.device)

        # Q/K/V projection via FP8 GEMM
        query = _fp8_linear(fp8_q, q_scale, hidden_states_2d)
        key = _fp8_linear(fp8_k, k_scale, hidden_states_2d)
        value = _fp8_linear(fp8_v, v_scale, hidden_states_2d)

        # QK Norm (on flat projected dims, before reshape)
        query = cuda_rmsnorm(query, self.module.q_norm.weight, self.module.q_norm.eps)
        key = cuda_rmsnorm(key, self.module.k_norm.weight, self.module.k_norm.eps)

        # Reshape to [total_tokens, num_heads, head_dim]
        num_heads = self.module.num_heads
        num_kv_heads = self.module.num_kv_heads
        head_dim = self.module.head_dim
        rotary_dim = self.module.rotary_dim

        query = query.view(total_tokens, num_heads, head_dim)
        key = key.view(total_tokens, num_kv_heads, head_dim)
        value = value.view(total_tokens, num_kv_heads, head_dim)

        # Partial RoPE (rotate first rotary_dim=64 dims, passthrough rest)
        cos, sin = self.module.rotary_emb(value, seq_len=max_seqlen)
        cos = cos[position_ids]  # [total_tokens, rotary_dim]
        sin = sin[position_ids]

        # Apply RoPE to rotary part only
        q_rot = query[..., :rotary_dim]
        q_pass = query[..., rotary_dim:]
        k_rot = key[..., :rotary_dim]
        k_pass = key[..., rotary_dim:]

        cos = cos.unsqueeze(1)  # [total_tokens, 1, rotary_dim]
        sin = sin.unsqueeze(1)

        query = torch.cat([
            q_rot * cos + rotate_half(q_rot) * sin,
            q_pass,
        ], dim=-1)
        key = torch.cat([
            k_rot * cos + rotate_half(k_rot) * sin,
            k_pass,
        ], dim=-1)

        # FlashAttention varlen GQA
        attn_output, lse = gqa_prefill_fa(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
        )

        # Output projection via FP8 GEMM
        attn_output = attn_output.view(total_tokens, num_heads * head_dim)
        attn_output = _fp8_linear(fp8_o, o_scale, attn_output)

        # Non-persistent: free weights
        if not self.persistent:
            torch.cuda.current_stream().synchronize()
            self.free_weights(self.module_key)

        # Offload KV cache to host
        torch.cuda.current_stream().synchronize()
        self._offload_prepacked_kv_gqa(key.view(total_tokens, num_kv_heads, head_dim),
                                        value)

        attn_output = attn_output.unsqueeze(0)
        return (attn_output, None, None)

    def _offload_prepacked_kv_gqa(self, k_cache, v_cache):
        """Offload GQA KV cache per-sequence to host memory."""
        cu_seqlens = self.prepack_cu_seqlens
        num_sequences = self.prepack_num_sequences
        global_sequence_ids = self.cur_batch

        for seq_idx in range(num_sequences):
            start_idx = cu_seqlens[seq_idx].item()
            end_idx = cu_seqlens[seq_idx + 1].item()
            seq_len = end_idx - start_idx

            seq_k = k_cache[start_idx:end_idx].unsqueeze(0)
            seq_v = v_cache[start_idx:end_idx].unsqueeze(0)
            seq_global_id = [global_sequence_ids[seq_idx]]

            self.core_engine.host_paged_kv_worker_view.async_offload_layer_kv_to_host(
                layer_idx=self.layer_idx,
                sequence_ids=seq_global_id,
                k_tensor=seq_k,
                v_tensor=seq_v,
                sequence_lengths=[seq_len],
            )

    def _forward_decode(self, hidden_states, **kwargs):
        """Decode forward: FP8 Q/K/V + QK norm + partial RoPE + paged KV attention.

        Uses batchgen WGMMA decode kernel on H20, FlashAttention fallback otherwise.
        Partial RoPE via CUDA rope_forward kernel (half_dim=rotary_dim//2).
        """
        from batchgen.attention.gqa import gqa_decode_fa
        from batchgen.attention.fused_kernels import cuda_rmsnorm

        fp8_q, q_scale, fp8_k, k_scale, fp8_v, v_scale, fp8_o, o_scale = self._get_attn_weights()

        batch_slice = kwargs.get("batch_slice", None)
        batch, seq_len, _ = hidden_states.shape
        assert seq_len == 1, "Decode expects single token"

        if batch == 0 or hidden_states.numel() == 0:
            return (hidden_states, None, None)

        gpu_kv_manager = AttnWrapperBase.gpu_paged_kv_manager
        cache_seqlens = AttnWrapperBase.cache_seqlens

        # Micro-batch slicing
        micro_cache_seqlens = cache_seqlens
        if cache_seqlens is not None and batch_slice is not None:
            start_idx, end_idx = batch_slice
            micro_cache_seqlens = cache_seqlens[start_idx:end_idx]

        current_token_position = micro_cache_seqlens - 1 if micro_cache_seqlens is not None else None

        num_heads = self.module.num_heads
        num_kv_heads = self.module.num_kv_heads
        head_dim = self.module.head_dim
        rotary_dim = self.module.rotary_dim

        # Q/K/V projection via FP8 GEMM
        if _DECODE_TIMING:
            torch.cuda.synchronize()
            _t0 = time.perf_counter()

        query = _fp8_linear(fp8_q, q_scale, hidden_states)
        key = _fp8_linear(fp8_k, k_scale, hidden_states)
        value = _fp8_linear(fp8_v, v_scale, hidden_states)

        # QK Norm
        query = cuda_rmsnorm(query, self.module.q_norm.weight, self.module.q_norm.eps)
        key = cuda_rmsnorm(key, self.module.k_norm.weight, self.module.k_norm.eps)

        if _DECODE_TIMING:
            DecodeTimingStats._sync_record("attn_qkv_proj", _t0)
            _t0 = time.perf_counter()

        # Reshape: [batch, 1, num_heads, head_dim]
        query = query.view(batch, seq_len, num_heads, head_dim)
        key = key.view(batch, seq_len, num_kv_heads, head_dim)
        value = value.view(batch, seq_len, num_kv_heads, head_dim)

        # Partial RoPE — CUDA kernel or PyTorch fallback
        max_seqlen = AttnWrapperBase.max_seqlen
        cos, sin = self.module.rotary_emb(value, seq_len=max_seqlen)

        if _HAS_CUDA_ROPE:
            if not getattr(self.__class__, '_warned_rope', False):
                logging.warning("[Attn] HOT PATH: CUDA rope_forward (partial rotation, passthrough copy)")
                self.__class__._warned_rope = True
            # cos/sin from rotary_emb: [seq_len, rotary_dim]
            # Index by position: [batch, rotary_dim]
            cos_pos = cos[current_token_position]  # [batch, rotary_dim]
            sin_pos = sin[current_token_position]  # [batch, rotary_dim]
            # Pad to head_dim for kernel (kernel reads cos[batch_seq_idx * head_dim + tid])
            cos_padded = torch.nn.functional.pad(cos_pos, (0, head_dim - rotary_dim))  # [batch, head_dim]
            sin_padded = torch.nn.functional.pad(sin_pos, (0, head_dim - rotary_dim))  # [batch, head_dim]
            # Reshape for kernel: [B, S=1, head_dim]
            cos_k = cos_padded.unsqueeze(1)  # [batch, 1, head_dim]
            sin_k = sin_padded.unsqueeze(1)  # [batch, 1, head_dim]
            half_dim = rotary_dim // 2  # 32 for MiniMax
            query, key = _cuda_rope_forward(
                query, key, cos_k, sin_k, half_dim)
        else:
            if not getattr(self.__class__, '_warned_rope', False):
                logging.warning("[Attn] HOT PATH: PyTorch rotate_half RoPE (fallback)")
                self.__class__._warned_rope = True
            cos = cos[current_token_position].unsqueeze(1).unsqueeze(2)
            sin = sin[current_token_position].unsqueeze(1).unsqueeze(2)

            q_rot = query[..., :rotary_dim]
            q_pass = query[..., rotary_dim:]
            k_rot = key[..., :rotary_dim]
            k_pass = key[..., rotary_dim:]

            query = torch.cat([
                q_rot * cos + rotate_half(q_rot) * sin,
                q_pass,
            ], dim=-1)
            key = torch.cat([
                k_rot * cos + rotate_half(k_rot) * sin,
                k_pass,
            ], dim=-1)

        if _DECODE_TIMING:
            DecodeTimingStats._sync_record("attn_rope", _t0)
            _t0 = time.perf_counter()

        # Write new K,V to paged GPU cache
        gpu_kv_manager.update_layer_decode_new_token(
            k_tensor=key,
            v_tensor=value,
            sequence_lengths=current_token_position,
            layer_idx=self.layer_idx,
            batch_slice=batch_slice,
        )

        # Retrieve paged KV cache
        k_cache_layer, v_cache_layer, page_table = gpu_kv_manager.get_layer_kv_with_page_table(
            self.layer_idx
        )
        if batch_slice is not None:
            start_idx, end_idx = batch_slice
            page_table = page_table[start_idx:end_idx]

        if _DECODE_TIMING:
            DecodeTimingStats._sync_record("attn_kv_update", _t0)
            _t0 = time.perf_counter()

        cache_seqlens_for_attn = micro_cache_seqlens

        # Decode attention — use WGMMA kernel if available, else FlashAttention
        _use_wgmma = os.environ.get("BATCHGEN_USE_WGMMA_DECODE", "0") == "1"
        if _use_wgmma:
            from batchgen.attention.gqa import batchgen_gqa_decode_bf16
            if not getattr(self.__class__, '_warned_attn', False):
                logging.warning("[Attn] HOT PATH: batchgen_gqa_decode_bf16 (WGMMA)")
                self.__class__._warned_attn = True
            attn_output, _ = batchgen_gqa_decode_bf16(
                q=query,
                k_cache=k_cache_layer,
                v_cache=v_cache_layer,
                cache_seqlens=cache_seqlens_for_attn,
                block_table=page_table,
            )
        else:
            if not getattr(self.__class__, '_warned_attn', False):
                logging.warning("[Attn] HOT PATH: gqa_decode_fa (FlashAttention)")
                self.__class__._warned_attn = True
            attn_output, _ = gqa_decode_fa(
                q=query,
                k_cache=k_cache_layer,
                v_cache=v_cache_layer,
                cache_seqlens=cache_seqlens_for_attn,
                block_table=page_table,
            )

        if _DECODE_TIMING:
            DecodeTimingStats._sync_record("attn_forward", _t0)
            _t0 = time.perf_counter()

        # Output projection via FP8 GEMM
        attn_output = attn_output.view(batch, 1, num_heads * head_dim)
        attn_output = _fp8_linear(fp8_o, o_scale, attn_output)

        if _DECODE_TIMING:
            DecodeTimingStats._sync_record("attn_output_proj", _t0)

        # Append KV to host
        kv_append_callback = getattr(AttnWrapperBase, 'kv_append_callback', None)
        if kv_append_callback is not None:
            kv_append_callback(self.layer_idx, key, value)

        return (attn_output, None, None)
