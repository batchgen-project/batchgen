# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
# ---------------------------------------------------------------------------- #

"""DeepSeek-V4-Flash model definition.

This file is intentionally self-contained.  It mirrors the V4 tensor names from
``assets/inference/model.py`` while exposing the BatchGen worker contract:
``ForCausalLM.model``, ``ForCausalLM.lm_head``, ``model.embed_tokens``,
``model.layers``, ``model.norm``, ``layer.self_attn`` and ``layer.mlp``.

The structure is DP-attention + EP-MoE oriented: attention modules hold full
head projections, and MoE layers expose global expert slots that the parallel
strategy manager assigns to per-rank expert-parallel ranges.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from batchgen_kernels.common.v4_hyper_connections import hc_post, hc_pre
from batchgen_kernels.moe.v4_hash_routing import hash_routing
from batchgen_kernels.moe.v4_sqrtsoftplus_topk import sqrtsoftplus_topk


_FP4_E2M1_TABLE_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


@dataclass
class _CausalLMOutput:
    """Minimal output container with ``.logits`` for BatchGen workers."""

    logits: torch.Tensor


def _cfg(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default)


def _linear_from_weight(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Linear helper for V4 slots.

    FP8 checkpoint tensors are block scaled.  This fallback dequantizes them in
    PyTorch for correctness-oriented bring-up; optimized wrappers/kernels replace
    this path in production.
    """

    raw_weight_shape = tuple(weight.shape)
    weight = _dequant_weight(weight, scale, x.dtype)
    if x.shape[-1] != weight.shape[-1]:
        scale_shape = None if scale is None else tuple(scale.shape)
        raise RuntimeError(
            "DeepSeek-V4 linear shape mismatch: "
            f"input={tuple(x.shape)}, weight={tuple(weight.shape)}, "
            f"raw_weight={raw_weight_shape}, scale={scale_shape}"
        )
    return F.linear(x, weight, bias)


def _dequant_weight(
    weight: torch.Tensor,
    scale: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> torch.Tensor:
    if _is_fp4_e2m1_weight(weight, scale):
        return _dequant_fp4_e2m1_weight(weight, scale, dtype)
    if scale is not None and scale.ndim == 2 and weight.ndim == 2:
        row_block = max(weight.shape[0] // scale.shape[0], 1)
        col_block = max(weight.shape[1] // scale.shape[1], 1)
        expanded_scale = (
            scale.to(torch.float32)
            .repeat_interleave(row_block, dim=0)
            .repeat_interleave(col_block, dim=1)
        )
        expanded_scale = expanded_scale[: weight.shape[0], : weight.shape[1]]
        return (weight.to(torch.float32) * expanded_scale).to(dtype)
    return weight.to(dtype)


def _is_fp4_e2m1_weight(
    weight: torch.Tensor,
    scale: Optional[torch.Tensor],
) -> bool:
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4_dtype is not None and weight.dtype == fp4_dtype:
        return True
    if weight.dtype in (torch.int8, torch.uint8):
        return True
    return (
        scale is not None
        and weight.ndim == 2
        and scale.ndim == 2
        and weight.shape[0] == scale.shape[0]
        and weight.shape[1] == scale.shape[1] * 16
    )


def _fp4_packed_bytes(weight: torch.Tensor) -> torch.Tensor:
    if weight.element_size() == 1:
        return weight.contiguous().view(torch.uint8)
    return weight.contiguous().to(torch.uint8)


def _dequant_fp4_e2m1_weight(
    weight: torch.Tensor,
    scale: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> torch.Tensor:
    if scale is None:
        raise RuntimeError(
            "DeepSeek-V4 FP4 weight is missing its E8M0 scale tensor."
        )
    packed = _fp4_packed_bytes(weight)
    table = torch.tensor(
        _FP4_E2M1_TABLE_VALUES,
        dtype=torch.float32,
        device=packed.device,
    )
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    unpacked_shape = packed.shape[:-1] + (packed.shape[-1] * 2,)
    unpacked = torch.empty(
        unpacked_shape, dtype=torch.float32, device=packed.device
    )
    unpacked[..., 0::2] = table[low.long()]
    unpacked[..., 1::2] = table[high.long()]

    expanded_scale = (
        scale.to(torch.float32)
        .unsqueeze(-1)
        .expand(*scale.shape, 32)
        .reshape(*scale.shape[:-1], scale.shape[-1] * 32)
    )
    expanded_scale = expanded_scale[..., : unpacked.shape[-1]]
    return (unpacked * expanded_scale).to(dtype)


class DeepSeekV4FlashLinearSlot(nn.Module):
    """Runtime-loaded linear slot.

    Attention and expert bundle tensors are owned by the BatchGen parameter
    server/wrappers, not by the skeleton state dict.  The slot records shape
    metadata and receives tensors at wrapper execution time.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight: Optional[torch.Tensor] = None
        self.scale: Optional[torch.Tensor] = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def set_runtime_tensors(
        self, tensors: Dict[str, torch.Tensor], prefix: str
    ) -> None:
        self.weight = tensors.get(f"{prefix}.weight")
        self.scale = tensors.get(f"{prefix}.scale")

    def clear_runtime_tensors(self) -> None:
        self.weight = None
        self.scale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is None:
            raise RuntimeError(
                f"DeepSeek-V4 linear slot ({self.out_features}, {self.in_features}) "
                "has no runtime weight loaded."
            )
        return _linear_from_weight(x, self.weight, self.scale, self.bias)


class DeepSeekV4FlashRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.square().mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (x * self.weight.float()).to(dtype)


class DeepSeekV4FlashCompressor(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        rope_head_dim: int,
        compress_ratio: int,
        eps: float,
        overlap: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = overlap
        coeff = 2 if overlap else 1
        self.ape = nn.Parameter(
            torch.empty(compress_ratio, coeff * head_dim, dtype=torch.float32)
        )
        self.wkv = DeepSeekV4FlashLinearSlot(hidden_size, coeff * head_dim)
        self.wgate = DeepSeekV4FlashLinearSlot(hidden_size, coeff * head_dim)
        self.norm = DeepSeekV4FlashRMSNorm(head_dim, eps)


class DeepSeekV4FlashIndexer(nn.Module):
    def __init__(self, config: Any, compress_ratio: int):
        super().__init__()
        hidden_size = int(
            _cfg(config, "hidden_size", _cfg(config, "dim", 4096))
        )
        q_lora_rank = int(_cfg(config, "q_lora_rank", 1024))
        head_dim = int(_cfg(config, "index_head_dim", 128))
        n_heads = int(_cfg(config, "index_n_heads", 64))
        rope_head_dim = int(
            _cfg(config, "qk_rope_head_dim", _cfg(config, "rope_head_dim", 64))
        )
        eps = float(
            _cfg(config, "rms_norm_eps", _cfg(config, "norm_eps", 1e-6))
        )

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.index_topk = int(_cfg(config, "index_topk", 512))
        self.wq_b = DeepSeekV4FlashLinearSlot(q_lora_rank, n_heads * head_dim)
        self.weights_proj = DeepSeekV4FlashLinearSlot(hidden_size, n_heads)
        self.compressor = DeepSeekV4FlashCompressor(
            hidden_size,
            head_dim,
            rope_head_dim,
            compress_ratio,
            eps,
            overlap=True,
        )


class DeepSeekV4FlashAttention(nn.Module):
    """DP attention surface for V4.

    All V4 projection tensors use their checkpoint names as attributes.  The
    optimized sparse/compressed attention implementation is attached by the V4
    attention wrapper; this module also carries a small PyTorch fallback for
    early smoke tests on short prompts.
    """

    def __init__(self, config: Any, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = int(
            _cfg(config, "hidden_size", _cfg(config, "dim", 4096))
        )
        self.n_heads = int(
            _cfg(config, "num_attention_heads", _cfg(config, "n_heads", 64))
        )
        self.head_dim = int(_cfg(config, "head_dim", 512))
        self.q_lora_rank = int(_cfg(config, "q_lora_rank", 1024))
        self.o_groups = int(_cfg(config, "o_groups", 8))
        self.world_size = int(
            _cfg(
                config,
                "world_size",
                dist.get_world_size() if dist.is_initialized() else 1,
            )
        )
        if self.n_heads % self.world_size != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by world_size "
                f"({self.world_size}) for tensor-parallel attention"
            )
        if self.o_groups % self.world_size != 0:
            raise ValueError(
                f"o_groups ({self.o_groups}) must be divisible by world_size "
                f"({self.world_size}) for tensor-parallel output projection"
            )
        self.n_local_heads = self.n_heads // self.world_size
        self.n_local_groups = self.o_groups // self.world_size
        self.o_lora_rank = int(_cfg(config, "o_lora_rank", 1024))
        self.eps = float(
            _cfg(config, "rms_norm_eps", _cfg(config, "norm_eps", 1e-6))
        )
        self.softmax_scale = self.head_dim**-0.5

        ratios = list(_cfg(config, "compress_ratios", []))
        self.compress_ratio = (
            int(ratios[layer_idx]) if layer_idx < len(ratios) else 0
        )

        self.runtime_phase = "prefill"
        self._prefill_full_tensors: Dict[str, torch.Tensor] = {}

        self.attn_sink = nn.Parameter(
            torch.empty(self.n_local_heads, dtype=torch.float32)
        )
        self.wq_a = DeepSeekV4FlashLinearSlot(
            self.hidden_size, self.q_lora_rank
        )
        self.q_norm = DeepSeekV4FlashRMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = DeepSeekV4FlashLinearSlot(
            self.q_lora_rank, self.n_heads * self.head_dim
        )
        self.wkv = DeepSeekV4FlashLinearSlot(self.hidden_size, self.head_dim)
        self.kv_norm = DeepSeekV4FlashRMSNorm(self.head_dim, self.eps)
        self.wo_a = DeepSeekV4FlashLinearSlot(
            self.n_heads * self.head_dim // self.o_groups,
            self.o_groups * self.o_lora_rank,
        )
        self.wo_b = DeepSeekV4FlashLinearSlot(
            self.o_groups * self.o_lora_rank, self.hidden_size
        )

        if self.compress_ratio:
            rope_head_dim = int(
                _cfg(
                    config,
                    "qk_rope_head_dim",
                    _cfg(config, "rope_head_dim", 64),
                )
            )
            self.compressor = DeepSeekV4FlashCompressor(
                self.hidden_size,
                self.head_dim,
                rope_head_dim,
                self.compress_ratio,
                self.eps,
                overlap=self.compress_ratio == 4,
            )
            self.indexer = (
                DeepSeekV4FlashIndexer(config, self.compress_ratio)
                if self.compress_ratio == 4
                else None
            )
        else:
            self.compressor = None
            self.indexer = None

    def set_runtime_tensors(self, tensors: Dict[str, torch.Tensor]) -> None:
        for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            getattr(self, name).set_runtime_tensors(tensors, name)
        if "attn_sink" in tensors:
            self.attn_sink.data = tensors["attn_sink"].to(self.attn_sink.device)
        if "q_norm.weight" in tensors:
            self.q_norm.weight.data = tensors["q_norm.weight"].to(
                self.q_norm.weight.device
            )
        if "kv_norm.weight" in tensors:
            self.kv_norm.weight.data = tensors["kv_norm.weight"].to(
                self.kv_norm.weight.device
            )
        self._set_compressor_runtime(self.compressor, tensors, "compressor")
        if self.indexer is not None:
            self.indexer.wq_b.set_runtime_tensors(tensors, "indexer.wq_b")
            self.indexer.weights_proj.set_runtime_tensors(
                tensors, "indexer.weights_proj"
            )
            self._set_compressor_runtime(
                self.indexer.compressor, tensors, "indexer.compressor"
            )

    @staticmethod
    def _set_compressor_runtime(comp, tensors, prefix: str) -> None:
        if comp is None:
            return
        ape_key = f"{prefix}.ape"
        norm_key = f"{prefix}.norm.weight"
        if ape_key in tensors:
            comp.ape.data = tensors[ape_key].to(comp.ape.device)
        if norm_key in tensors:
            comp.norm.weight.data = tensors[norm_key].to(
                comp.norm.weight.device
            )
        comp.wkv.set_runtime_tensors(tensors, f"{prefix}.wkv")
        comp.wgate.set_runtime_tensors(tensors, f"{prefix}.wgate")

    def set_prefill_full_tensors(
        self, tensors: Dict[str, torch.Tensor]
    ) -> None:
        self._prefill_full_tensors = tensors

    def clear_prefill_full_tensors(self) -> None:
        self._prefill_full_tensors = {}

    def _get_prefill_full_tensor(self, name: str) -> torch.Tensor:
        tensor = self._prefill_full_tensors.get(name)
        if tensor is None:
            raise RuntimeError(
                f"DeepSeek-V4 prefill requires full replicated tensor "
                f"'{name}' for layer {self.layer_idx}"
            )
        return tensor

    def clear_runtime_tensors(self) -> None:
        for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            getattr(self, name).clear_runtime_tensors()
        if self.compressor is not None:
            self.compressor.wkv.clear_runtime_tensors()
            self.compressor.wgate.clear_runtime_tensors()
        if self.indexer is not None:
            self.indexer.wq_b.clear_runtime_tensors()
            self.indexer.weights_proj.clear_runtime_tensors()
            self.indexer.compressor.wkv.clear_runtime_tensors()
            self.indexer.compressor.wgate.clear_runtime_tensors()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, ...]] = None,
        cache_seqlens: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, ...]]
    ]:
        del position_ids, use_cache
        bsz, q_len, _ = hidden_states.shape

        prefill_dp = self.runtime_phase == "prefill" and self.world_size > 1
        if prefill_dp:
            n_heads = self.n_heads
            n_groups = self.o_groups
        else:
            n_heads = self.n_local_heads
            n_groups = self.n_local_groups

        q_low = self.q_norm(self.wq_a(hidden_states))
        if prefill_dp:
            q = _linear_from_weight(
                q_low,
                self._get_prefill_full_tensor("wq_b.weight"),
                self._prefill_full_tensors.get("wq_b.scale"),
            )
        else:
            q = self.wq_b(q_low)
        q = q.view(bsz, q_len, n_heads, self.head_dim)
        q = q * torch.rsqrt(q.square().mean(dim=-1, keepdim=True) + self.eps)

        kv = self.kv_norm(self.wkv(hidden_states))
        kv_for_attn = kv
        if past_key_value is not None:
            kv_for_attn = self._normalize_past_kv(past_key_value)
            if q_len == 1 and cache_seqlens is not None:
                self._write_current_kv(kv_for_attn, kv, cache_seqlens)
        k = kv_for_attn.unsqueeze(2).expand(-1, -1, n_heads, -1)
        v = k
        attn_scores = torch.einsum("bshd,bthd->bhst", q, k) * self.softmax_scale
        attn_scores = self._apply_fallback_masks(
            attn_scores,
            attention_mask,
            cache_seqlens,
            q_len,
            kv_for_attn.size(1),
            past_key_value is not None,
        )
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(
            q.dtype
        )
        attn_output = torch.einsum("bhst,bthd->bshd", attn_weights, v)

        attn_output = attn_output.reshape(
            bsz,
            q_len,
            n_groups,
            n_heads // n_groups * self.head_dim,
        )
        if prefill_dp:
            wo_a_weight = _dequant_weight(
                self._get_prefill_full_tensor("wo_a.weight"),
                None,
                hidden_states.dtype,
            )
            wo_a = wo_a_weight.view(
                n_groups,
                self.o_lora_rank,
                n_heads // n_groups * self.head_dim,
            )
            attn_output = torch.einsum("bsgd,grd->bsgr", attn_output, wo_a)
            attn_output = _linear_from_weight(
                attn_output.flatten(2),
                self._get_prefill_full_tensor("wo_b.weight"),
                self._prefill_full_tensors.get("wo_b.scale"),
            )
            return attn_output, None, kv

        wo_a_weight = self.wo_a.weight
        if wo_a_weight is None:
            raise RuntimeError(
                "DeepSeek-V4 attention wo_a weight is not loaded."
            )
        wo_a_weight = _dequant_weight(
            wo_a_weight,
            self.wo_a.scale,
            hidden_states.dtype,
        )
        wo_a = wo_a_weight.view(
            n_groups,
            self.o_lora_rank,
            n_heads // n_groups * self.head_dim,
        )
        attn_output = torch.einsum("bsgd,grd->bsgr", attn_output, wo_a)
        attn_output = self.wo_b(attn_output.flatten(2))
        if self.world_size > 1 and dist.is_initialized():
            dist.all_reduce(attn_output)
        return attn_output, None, kv

    @staticmethod
    def _normalize_past_kv(past_key_value: torch.Tensor) -> torch.Tensor:
        if past_key_value.dim() == 4 and past_key_value.size(2) == 1:
            return past_key_value.squeeze(2)
        if past_key_value.dim() == 3:
            return past_key_value
        raise RuntimeError(
            "DeepSeek-V4 fallback attention expected past KV with shape "
            f"[B, T, D] or [B, T, 1, D], got {tuple(past_key_value.shape)}"
        )

    @staticmethod
    def _write_current_kv(
        past_kv: torch.Tensor,
        current_kv: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> None:
        positions = (cache_seqlens.to(current_kv.device).long() - 1).clamp_min(
            0
        )
        batch_idx = torch.arange(current_kv.size(0), device=current_kv.device)
        valid = positions < past_kv.size(1)
        if valid.any():
            past_kv[batch_idx[valid], positions[valid]] = current_kv[valid, 0]

    @staticmethod
    def _apply_fallback_masks(
        attn_scores: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        cache_seqlens: Optional[torch.Tensor],
        q_len: int,
        kv_len: int,
        using_past: bool,
    ) -> torch.Tensor:
        neg_inf = torch.finfo(attn_scores.dtype).min
        device = attn_scores.device
        if cache_seqlens is not None:
            valid_lens = cache_seqlens.to(device).long().clamp(max=kv_len)
            key_pos = torch.arange(kv_len, device=device).unsqueeze(0)
            mask = key_pos >= valid_lens.unsqueeze(1)
            return attn_scores.masked_fill(mask[:, None, None, :], neg_inf)

        if attention_mask is not None and attention_mask.dim() == 2:
            key_mask = attention_mask[:, -kv_len:].to(device) == 0
            attn_scores = attn_scores.masked_fill(
                key_mask[:, None, None, :], neg_inf
            )
        elif attention_mask is not None and attention_mask.dim() == 4:
            attn_scores = attn_scores + attention_mask.to(device)

        if not using_past and q_len > 1:
            causal = torch.triu(
                torch.ones(q_len, kv_len, dtype=torch.bool, device=device),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(
                causal[None, None, :, :], neg_inf
            )
        return attn_scores


class DeepSeekV4FlashGate(nn.Module):
    def __init__(self, config: Any, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = int(
            _cfg(config, "hidden_size", _cfg(config, "dim", 4096))
        )
        self.num_experts = int(
            _cfg(
                config,
                "n_routed_experts",
                _cfg(config, "num_local_experts", 256),
            )
        )
        self.topk = int(
            _cfg(
                config,
                "num_experts_per_tok",
                _cfg(config, "n_activated_experts", 6),
            )
        )
        self.score_func = str(
            _cfg(
                config,
                "scoring_func",
                _cfg(config, "score_func", "sqrtsoftplus"),
            )
        )
        self.route_scale = float(
            _cfg(
                config,
                "routed_scaling_factor",
                _cfg(config, "route_scale", 1.5),
            )
        )
        self.norm_topk_prob = bool(_cfg(config, "norm_topk_prob", True))
        self.is_hash_layer = layer_idx < int(
            _cfg(config, "num_hash_layers", _cfg(config, "n_hash_layers", 3))
        )

        self.weight = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size)
        )
        if self.is_hash_layer:
            vocab_size = int(_cfg(config, "vocab_size", 129280))
            self.tid2eid = nn.Parameter(
                torch.empty(vocab_size, self.topk, dtype=torch.long),
                requires_grad=False,
            )
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(
                torch.empty(self.num_experts, dtype=torch.float32)
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.is_hash_layer:
            return hash_routing(
                input_ids=input_ids,
                tid2eid=self.tid2eid,
                hidden_states=hidden_states,
                gate_weight=self.weight,
                topk=self.topk,
                route_scale=self.route_scale,
                score_func=self.score_func,
                norm_topk_prob=self.norm_topk_prob,
            )
        return sqrtsoftplus_topk(
            hidden_states=hidden_states,
            gate_weight=self.weight,
            bias=self.bias,
            topk=self.topk,
            route_scale=self.route_scale,
            norm_topk_prob=self.norm_topk_prob,
        )


class DeepSeekV4FlashExpertPlaceholder(nn.Module):
    """Lightweight expert slot replaced/configured by V4 expert wrappers."""

    def __init__(
        self, hidden_size: int, intermediate_size: int, swiglu_limit: float
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.swiglu_limit = swiglu_limit
        self.runtime_weights: Optional[Dict[str, torch.Tensor]] = None

    def set_runtime_tensors(self, tensors: Dict[str, torch.Tensor]) -> None:
        self.runtime_weights = tensors

    def clear_runtime_tensors(self) -> None:
        self.runtime_weights = None

    def _linear(self, x: torch.Tensor, name: str) -> torch.Tensor:
        if self.runtime_weights is None:
            raise RuntimeError("DeepSeek-V4 expert weights are not loaded.")
        return _linear_from_weight(
            x,
            self.runtime_weights[f"{name}.weight"],
            self.runtime_weights.get(f"{name}.scale"),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gate = self._linear(hidden_states, "w1")
        up = self._linear(hidden_states, "w3")
        if self.swiglu_limit > 0:
            gate = torch.clamp(gate.float(), max=self.swiglu_limit).to(
                gate.dtype
            )
            up = torch.clamp(
                up.float(), min=-self.swiglu_limit, max=self.swiglu_limit
            ).to(up.dtype)
        try:
            from batchgen_kernels.moe.silu_mul_quant import (
                fused_silu_mul_quant_cuda,
            )

            activated_fp8, _scales = fused_silu_mul_quant_cuda(
                gate.to(torch.bfloat16), up.to(torch.bfloat16)
            )
            activated = activated_fp8.float() * _scales.unsqueeze(-1)
        except (ImportError, RuntimeError):
            activated = F.silu(gate.float()) * up.float()
        if weights is not None:
            activated = activated * weights
        return self._linear(
            activated.to(
                weights.dtype if weights is not None else hidden_states.dtype
            ),
            "w2",
        )


class DeepSeekV4FlashMoE(nn.Module):
    """V4 EP-MoE surface with global expert slots."""

    def __init__(self, config: Any, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = int(
            _cfg(config, "hidden_size", _cfg(config, "dim", 4096))
        )
        self.intermediate_size = int(
            _cfg(
                config,
                "moe_intermediate_size",
                _cfg(config, "moe_inter_dim", 2048),
            )
        )
        self.total_experts = int(
            _cfg(
                config,
                "n_routed_experts",
                _cfg(config, "num_local_experts", 256),
            )
        )
        self.num_experts_per_tok = int(
            _cfg(
                config,
                "num_experts_per_tok",
                _cfg(config, "n_activated_experts", 6),
            )
        )
        self.swiglu_limit = float(_cfg(config, "swiglu_limit", 10.0))
        self.gate = DeepSeekV4FlashGate(config, layer_idx)
        self.experts = nn.ModuleList(
            [
                DeepSeekV4FlashExpertPlaceholder(
                    self.hidden_size, self.intermediate_size, self.swiglu_limit
                )
                for _ in range(self.total_experts)
            ]
        )
        self.shared_experts = DeepSeekV4FlashExpertPlaceholder(
            self.hidden_size, self.intermediate_size, 0.0
        )
        self.comm = None
        self.rank = 0
        self.world_size = 1
        self.routed_expert_start_idx = 0
        self.routed_expert_end_idx = self.total_experts
        self.experts_per_rank = self.total_experts
        self.enable_ep_offloading = False
        self.num_tokens_per_rank = None
        self.max_num_tokens_per_rank = None
        self.pad_token_id = int(_cfg(config, "pad_token_id", 0))

    def configure_ep(self, rank: int, world_size: int, comm=None) -> None:
        self.comm = comm
        self.rank = rank
        self.world_size = world_size
        self.experts_per_rank = math.ceil(self.total_experts / world_size)
        self.routed_expert_start_idx = min(
            rank * self.experts_per_rank, self.total_experts
        )
        self.routed_expert_end_idx = min(
            (rank + 1) * self.experts_per_rank, self.total_experts
        )
        self.enable_ep_offloading = world_size > 1

    def init_num_tokens(self, num_tokens_per_rank: int) -> None:
        self.num_tokens_per_rank = int(num_tokens_per_rank)
        self.max_num_tokens_per_rank = int(num_tokens_per_rank)

    def set_num_tokens_per_rank(self, num_tokens_per_rank: int) -> None:
        num_tokens_per_rank = int(num_tokens_per_rank)
        if (
            self.max_num_tokens_per_rank is None
            or num_tokens_per_rank > self.max_num_tokens_per_rank
        ):
            self.max_num_tokens_per_rank = num_tokens_per_rank
        self.num_tokens_per_rank = num_tokens_per_rank

    def _run_owned_experts(
        self,
        token_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        routed = torch.zeros_like(token_states, dtype=torch.float32)
        counts = torch.bincount(
            topk_indices.reshape(-1), minlength=self.total_experts
        )
        for expert_idx in range(
            self.routed_expert_start_idx, self.routed_expert_end_idx
        ):
            if counts[expert_idx].item() == 0:
                continue
            token_idx, topk_pos = torch.where(topk_indices == expert_idx)
            expert_out = self.experts[expert_idx](
                token_states[token_idx],
                topk_weights[token_idx, topk_pos].unsqueeze(-1),
            )
            routed[token_idx] += expert_out.float()
        return routed

    def _forward_local_routed(
        self, flat_states: torch.Tensor, flat_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        topk_weights, topk_indices = self.gate(flat_states, flat_ids)
        return self._run_owned_experts(flat_states, topk_weights, topk_indices)

    def _forward_ep_decode_routed(
        self, flat_states: torch.Tensor, flat_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if self.num_tokens_per_rank is None:
            raise RuntimeError(
                "DeepSeek-V4 MoE num_tokens_per_rank is not initialized; "
                "configure_decoding must call init_num_tokens before EP decode."
            )
        real_tokens = flat_states.shape[0]
        ntpr = int(self.num_tokens_per_rank)
        if real_tokens > ntpr:
            raise RuntimeError(
                f"DeepSeek-V4 MoE buffer overflow: real_tokens={real_tokens} > "
                f"num_tokens_per_rank={ntpr}"
            )

        padded = flat_states.new_zeros((ntpr, self.hidden_size))
        if real_tokens > 0:
            padded[:real_tokens] = flat_states
        global_states = flat_states.new_empty(
            (self.world_size * ntpr, self.hidden_size)
        )
        dist.all_gather_into_tensor(global_states, padded)

        global_ids = None
        if flat_ids is not None:
            padded_ids = torch.full(
                (ntpr,),
                self.pad_token_id,
                dtype=flat_ids.dtype,
                device=flat_ids.device,
            )
            if real_tokens > 0:
                padded_ids[:real_tokens] = flat_ids
            global_ids = torch.empty(
                (self.world_size * ntpr,),
                dtype=flat_ids.dtype,
                device=flat_ids.device,
            )
            dist.all_gather_into_tensor(global_ids, padded_ids)
        elif getattr(self.gate, "is_hash_layer", False):
            raise RuntimeError(
                "DeepSeek-V4 hash-routing MoE requires input_ids during EP decode."
            )

        topk_weights, topk_indices = self.gate(global_states, global_ids)
        global_routed = self._run_owned_experts(
            global_states, topk_weights, topk_indices
        )
        dist.all_reduce(global_routed, op=dist.ReduceOp.SUM)

        start = self.rank * ntpr
        return global_routed[start : start + real_tokens]

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        shape = hidden_states.shape
        flat_states = hidden_states.reshape(-1, self.hidden_size)
        flat_ids = input_ids.reshape(-1) if input_ids is not None else None

        if self.enable_ep_offloading and dist.is_initialized():
            routed = self._forward_ep_decode_routed(flat_states, flat_ids)
        else:
            routed = self._forward_local_routed(flat_states, flat_ids)

        shared = self.shared_experts(flat_states).float()
        return (routed + shared).to(hidden_states.dtype).view(shape)


class DeepSeekV4FlashDecoderLayer(nn.Module):
    def __init__(self, config: Any, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = int(
            _cfg(config, "hidden_size", _cfg(config, "dim", 4096))
        )
        self.hc_mult = int(_cfg(config, "hc_mult", 4))
        self.hc_eps = float(_cfg(config, "hc_eps", 1e-6))
        self.hc_sinkhorn_iters = int(_cfg(config, "hc_sinkhorn_iters", 20))
        self.rms_norm_eps = float(
            _cfg(config, "rms_norm_eps", _cfg(config, "norm_eps", 1e-6))
        )
        hc_dim = self.hc_mult * self.hidden_size
        mix_hc = (2 + self.hc_mult) * self.hc_mult

        self.self_attn = DeepSeekV4FlashAttention(config, layer_idx)
        self.attn = self.self_attn
        self.mlp = DeepSeekV4FlashMoE(config, layer_idx)
        self.ffn = self.mlp
        self.attn_norm = DeepSeekV4FlashRMSNorm(
            self.hidden_size, self.rms_norm_eps
        )
        self.ffn_norm = DeepSeekV4FlashRMSNorm(
            self.hidden_size, self.rms_norm_eps
        )
        self.input_layernorm = self.attn_norm
        self.post_attention_layernorm = self.ffn_norm

        self.hc_attn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32)
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32)
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32)
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32)
        )
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, ...]] = None,
        cache_seqlens: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, ...]]
    ]:
        del output_attentions, kwargs
        collapse_hc_state = hidden_states.dim() == 3
        if collapse_hc_state:
            hidden_states = (
                hidden_states.unsqueeze(2)
                .expand(-1, -1, self.hc_mult, -1)
                .contiguous()
            )

        residual = hidden_states
        attn_input, post, comb = hc_pre(
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
            self.rms_norm_eps,
        )
        attn_input = self.attn_norm(attn_input)
        attn_out, attn_weights, present = self.self_attn(
            attn_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            cache_seqlens=cache_seqlens,
            use_cache=use_cache,
        )
        hidden_states = hc_post(attn_out, residual, post, comb)

        residual = hidden_states
        mlp_input, post, comb = hc_pre(
            hidden_states,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
            self.rms_norm_eps,
        )
        mlp_input = self.ffn_norm(mlp_input)
        mlp_out = self.mlp(mlp_input, input_ids)
        hidden_states = hc_post(mlp_out, residual, post, comb)
        if collapse_hc_state:
            hidden_states = hidden_states.mean(dim=2)
        return hidden_states, attn_weights, present


class DeepSeekV4FlashModel(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.hidden_size = int(
            _cfg(config, "hidden_size", _cfg(config, "dim", 4096))
        )
        self.vocab_size = int(_cfg(config, "vocab_size", 129280))
        self.hc_mult = int(_cfg(config, "hc_mult", 4))
        self.hc_eps = float(_cfg(config, "hc_eps", 1e-6))
        self.rms_norm_eps = float(
            _cfg(config, "rms_norm_eps", _cfg(config, "norm_eps", 1e-6))
        )
        self.embed_tokens = nn.Embedding(
            self.vocab_size,
            self.hidden_size,
            int(_cfg(config, "pad_token_id", 1)),
        )
        self.embed = self.embed_tokens
        self.layers = nn.ModuleList(
            [
                DeepSeekV4FlashDecoderLayer(config, layer_idx)
                for layer_idx in range(
                    int(
                        _cfg(
                            config,
                            "num_hidden_layers",
                            _cfg(config, "n_layers", 43),
                        )
                    )
                )
            ]
        )
        self.norm = DeepSeekV4FlashRMSNorm(self.hidden_size, self.rms_norm_eps)
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, hc_dim, dtype=torch.float32)
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32)
        )
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    def _hc_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        flat = hidden_states.flatten(2).float()
        rsqrt = torch.rsqrt(
            flat.square().mean(-1, keepdim=True) + self.rms_norm_eps
        )
        mixes = F.linear(flat, self.hc_head_fn) * rsqrt
        pre = (
            torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base)
            + self.hc_eps
        )
        return torch.sum(pre.unsqueeze(-1) * flat.view(shape), dim=2).to(
            hidden_states.dtype
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...], ...]] = None,
        output_attentions: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        del return_dict, kwargs
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be provided")
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = (
            inputs_embeds.unsqueeze(2)
            .expand(-1, -1, self.hc_mult, -1)
            .contiguous()
        )
        presents = []
        for idx, layer in enumerate(self.layers):
            past_kv = (
                past_key_values[idx] if past_key_values is not None else None
            )
            hidden_states, _, present = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                input_ids=input_ids,
                past_key_value=past_kv,
                output_attentions=bool(output_attentions),
                use_cache=bool(use_cache),
            )
            if use_cache:
                presents.append(present)
        hidden_states = self.norm(self._hc_head(hidden_states))
        if use_cache:
            return hidden_states, tuple(presents)
        return (hidden_states,)


class DeepSeekV4FlashForCausalLM(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.model = DeepSeekV4FlashModel(config)
        self.vocab_size = int(_cfg(config, "vocab_size", 129280))
        hidden_size = int(
            _cfg(config, "hidden_size", _cfg(config, "dim", 4096))
        )
        self.lm_head = nn.Linear(hidden_size, self.vocab_size, bias=False)
        self.head = self.lm_head

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...], ...]] = None,
        output_attentions: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> _CausalLMOutput:
        del kwargs
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        return _CausalLMOutput(logits=logits)
