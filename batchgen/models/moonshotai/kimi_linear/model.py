# ---------------------------------------------------------------------------- #
#  BatchGen                                                                     #
#  Copyright (c) EfficientMoE team 2025                                         #
#                                                                               #
#  Licensed under the Apache License, Version 2.0 (the "License");              #
#  you may not use this file except in compliance with the License.            #
# ---------------------------------------------------------------------------- #
"""Kimi-Linear / Kimi-K3 (K3 family) model definition for BatchGen.

Clean, BatchGen-resident nn.Module port of the verified oracle
(k3dev/kref/modeling_kimi_linear.py + configuration_kimi_k3.py). The math is a
faithful, correctness-first reimplementation:

  * KDA (Kimi Delta Attention) linear/recurrent layers use the git-main `fla`
    kernels (`chunk_kda` for prefill, `fused_recurrent_kda` for single-token
    decode), exactly as the oracle does.
  * NoPE-MLA attention is eager (fp32 softmax), handling q_head_dim(192) vs
    v_head_dim(128) and an optional sigmoid output gate.
  * MoE routing is torch (sigmoid gate + e_score_correction_bias / noaux_tc
    top-k, renormalize, routed_scaling_factor, shared expert, optional LatentMoE
    down/up projection with optional norm).

Parameter (and submodule) names are IDENTICAL to the checkpoint so the real
Kimi-Linear-48B / Kimi-K3 weights load directly.

Covers both members of the family via a single `KimiLinearConfig`:
  * Kimi-Linear-48B-A3B (testbed): low-rank KDA gate, NoPE-MLA (no output gate),
    hidden-space MoE, SiLU, no attention residuals, q_lora_rank=None.
  * Kimi-K3: full-rank KDA gate, gated NoPE-MLA, LatentMoE + SiTU + AttnRes,
    q_lora_rank set, 2 shared experts.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.ops.kda import chunk_kda, fused_recurrent_kda

from .block_residual import BlockResidualBuffer, num_block_residual_columns
from .block_residual import apply_attn_res as _block_residual_apply_attn_res
from .config import KimiLinearConfig


# ============================================================================
#  Activations
# ============================================================================
class SituAndMul(nn.Module):
    """SiTU-and-multiply activation (Moonshot).

        situ_a = beta * tanh(gate / beta) * sigmoid(gate)
        out    = situ_a * up
    When ``linear_beta`` is set, ``up`` is first passed through
    ``linear_beta * tanh(up / linear_beta)``. Input is a concatenated
    ``[gate, up]`` on the last axis (like SwiGLU's gate/up split).
    """

    def __init__(self, beta: float = 1.0, linear_beta: Optional[float] = None):
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = x[..., :d].to(torch.float32)
        up = x[..., d:].to(torch.float32)
        situ_a = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (situ_a * up).to(x.dtype)


def _get_situ_activation_params(config: KimiLinearConfig) -> Tuple[float, Optional[float]]:
    beta = getattr(config, "activation_situ_beta", None)
    linear_beta = getattr(config, "activation_situ_linear_beta", None)
    return beta or 1.0, linear_beta


def build_activation(config: KimiLinearConfig) -> nn.Module | Callable:
    """Activation factory keyed on ``config.hidden_act`` ('silu' | 'situ').

    Returns a module/callable applied to a concatenated ``[gate, up]`` tensor
    for 'situ', or a plain elementwise activation for 'silu' (the SwiGLU
    ``act(gate) * up`` composition is done by the caller).
    """
    act = config.hidden_act
    if act == "situ":
        beta, linear_beta = _get_situ_activation_params(config)
        return SituAndMul(beta=beta, linear_beta=linear_beta)
    if act == "silu":
        return F.silu
    raise NotImplementedError(f"Unsupported hidden_act: {act}")


# ============================================================================
#  Norm
# ============================================================================
class KimiRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self._resident_prefill_token_tile = None

    def _norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        x = hidden_states.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        token_tile = self._resident_prefill_token_tile
        num_tokens = hidden_states.numel() // hidden_states.shape[-1]
        if token_tile is None or num_tokens <= int(token_tile):
            return self._norm(hidden_states)

        flat = hidden_states.reshape(num_tokens, hidden_states.shape[-1])
        output = None
        for start in range(0, num_tokens, int(token_tile)):
            end = min(start + int(token_tile), num_tokens)
            y = self._norm(flat[start:end])
            if output is None:
                output = y.new_empty((num_tokens, y.shape[-1]))
            output[start:end].copy_(y)
            del y
        return output.view_as(hidden_states)


# ============================================================================
#  Dense MLP / expert MLP
# ============================================================================
# Token-tile width for KimiMLP's chunked body.
#
# The FFN is elementwise in the token axis but very wide in the feature axis:
# at its peak SituAndMul holds FIVE full (tokens, intermediate) fp32 tensors at
# once — `gate`, `up`, `beta*tanh(gate/beta)`, `sigmoid(gate)` and their
# product — while KimiMLP's bf16 `[gate, up]` cat is still bound by the caller
# frame. That is 5*4 + 2*2 = 24 bytes per (token, intermediate) element.
# Unchunked at S=131,072 (batchgen_design/model_support/kimi_k3/
# PREFILL_MEMORY_AUDIT.md section 4):
#
#   layer 0 dense MLP, intermediate 33,792 : 24*S*I = 98.998 GiB
#   MoE shared expert, intermediate  6,144 : 24*S*I = 18.000 GiB
#
# Tiled, the same term is 24*TILE*I and no longer scales with S: 6.19 GiB and
# 1.13 GiB respectively at TILE=8192. 8192 rows is far past the knee of these
# GEMMs (K=7168 with N=2*intermediate already saturates the GPU on the N axis
# alone), so the tiling costs no throughput; doubling it would only double the
# term that is being removed.
_FFN_TOKEN_TILE = 8192


class KimiMLP(nn.Module):
    """Dense SwiGLU/SiTU MLP (gate_proj / up_proj / down_proj naming)."""

    def __init__(self, config: KimiLinearConfig, hidden_size: Optional[int] = None,
                 intermediate_size: Optional[int] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = build_activation(config)
        self._resident_prefill_token_tile = None

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        """The FFN body, verbatim. Applied to the whole input or to one token
        tile — same ops, same order, either way."""
        if self.config.hidden_act == "situ":
            gate_up = torch.cat([self.gate_proj(x), self.up_proj(x)], dim=-1)
            return self.down_proj(self.act_fn(gate_up))
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    def _reduce_tp_output(self, output: torch.Tensor) -> torch.Tensor:
        """Sum row-parallel shared-expert partials across its TP group."""
        if getattr(self, "_tp_size", 1) > 1:
            import torch.distributed as dist

            profiler = getattr(self, "_streamed_sp8_profiler", None)
            if (
                profiler is not None
                and not profiler._prefill_profile_enabled
            ):
                profiler = None
            span = (
                profiler.begin_profile_span() if profiler is not None else None
            )
            dist.all_reduce(output, group=self._tp_group)
            if profiler is not None:
                profiler.end_profile_span("shared_expert_reduce", span)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Token-tiled FFN — BIT-EXACT against the unchunked body.

        Every output row depends only on its own input row (the projections are
        bias-free row-wise GEMMs, everything between them is elementwise), and
        `_ffn` is the untouched body, so no sum is reassociated and no op order
        changes. Only the peak allocation changes.

        Tiles are EVEN, not `fixed width + ragged remainder`: a remainder of a
        handful of rows is a degenerate GEMM shape, and a degenerate M can move
        the BLAS onto a different (GEMV-like) reduction order. MEASURED on CPU
        fp32: a 1-row `F.linear` does NOT reproduce the corresponding row of the
        full GEMM, while every tile >= 7 rows does. Even tiles keep every tile
        at `_FFN_TOKEN_TILE / 2` rows or wider, so no such shape is ever
        emitted. Pinned by tests/test_kimi_linear_ffn_chunk.py.
        """
        num_tokens = x.numel() // x.shape[-1]
        token_tile = _FFN_TOKEN_TILE
        resident_tile = self._resident_prefill_token_tile
        if (
            resident_tile is not None
            and num_tokens > token_tile
            and num_tokens % int(resident_tile) == 0
        ):
            # The registered W2 shape is 16,384 rows, exactly 32 x 512.
            # Ragged 512-ish GEMMs can select a different cuBLAS reduction
            # order, so non-divisible shapes keep the validated 8,192-row
            # even tiler instead of forcing a 512-row remainder.
            token_tile = min(token_tile, int(resident_tile))
        if token_tile <= 0:
            raise ValueError("KimiMLP token tile must be positive")
        if num_tokens <= token_tile:
            # Decode and short prefill: the pre-chunking call, unchanged.
            return self._reduce_tp_output(self._ffn(x))

        flat = x.reshape(num_tokens, x.shape[-1])
        n_tiles = math.ceil(num_tokens / token_tile)
        out = None
        for i in range(n_tiles):
            start = (i * num_tokens) // n_tiles
            end = ((i + 1) * num_tokens) // n_tiles
            y = self._ffn(flat[start:end])
            if out is None:
                # Allocated from the first tile so the output dtype is the one
                # the body produces, never a guess off `x`.
                out = y.new_empty((num_tokens, y.shape[-1]))
            out[start:end] = y
            # `y` is rebound only on the NEXT iteration's assignment, i.e. after
            # `self._ffn(...)` for that tile has already peaked. Without this
            # `del`, one whole (tile, hidden) output tile is co-live with the
            # peak — MEASURED at +T*H*2 bytes = +0.109 GiB at K3 scale, which is
            # small but is carried into the prefill budget and is free to drop.
            del y
        return self._reduce_tp_output(
            out.view(*x.shape[:-1], out.shape[-1])
        )

    def forward_into(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Run the token-tiled FFN into caller-owned storage.

        ``out`` may alias ``x``. Each output row depends only on the matching
        input row, so overwriting a completed tile cannot affect a later tile.
        The TP reduction remains one full-shape collective after every local
        tile is written, matching :meth:`forward` rather than changing its
        reduction shape or order.
        """
        if (
            out.shape != x.shape
            or out.dtype != x.dtype
            or out.device != x.device
        ):
            raise ValueError("KimiMLP forward_into requires matching tensors")

        num_tokens = x.numel() // x.shape[-1]
        token_tile = _FFN_TOKEN_TILE
        resident_tile = self._resident_prefill_token_tile
        if (
            resident_tile is not None
            and num_tokens > token_tile
            and num_tokens % int(resident_tile) == 0
        ):
            token_tile = min(token_tile, int(resident_tile))
        if token_tile <= 0:
            raise ValueError("KimiMLP token tile must be positive")

        flat_x = x.reshape(num_tokens, x.shape[-1])
        flat_out = out.reshape(num_tokens, out.shape[-1])
        n_tiles = max(1, math.ceil(num_tokens / token_tile))
        for i in range(n_tiles):
            start = (i * num_tokens) // n_tiles
            end = ((i + 1) * num_tokens) // n_tiles
            y = self._ffn(flat_x[start:end])
            flat_out[start:end].copy_(y)
            del y
        return self._reduce_tp_output(out)


class KimiBlockSparseMLP(nn.Module):
    """Routed-expert MLP (w1=gate, w2=down, w3=up naming, per checkpoint)."""

    def __init__(self, config: KimiLinearConfig, hidden_size: Optional[int] = None,
                 intermediate_size: Optional[int] = None):
        super().__init__()
        self.config = config
        self.ffn_dim = config.intermediate_size if intermediate_size is None else intermediate_size
        self.hidden_dim = config.hidden_size if hidden_size is None else hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)   # gate
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)   # down
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)   # up
        self.act_fn = build_activation(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.hidden_act == "situ":
            gate_up = torch.cat([self.w1(hidden_states), self.w3(hidden_states)], dim=-1)
            current = self.act_fn(gate_up)
        else:
            current = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        return self.w2(current)


# ============================================================================
#  MoE gate + sparse block
# ============================================================================
class KimiMoEGate(nn.Module):
    """Sigmoid/softmax gate with e_score_correction_bias (noaux_tc top-k)."""

    def __init__(self, config: KimiLinearConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.moe_router_activation_func = config.scoring_func
        self.num_expert_group = getattr(config, "n_group", 1) or 1
        self.topk_group = getattr(config, "topk_group", 1) or 1

        self.moe_renormalize = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.num_experts, self.gating_dim)))
        self.e_score_correction_bias = nn.Parameter(torch.empty(self.num_experts))

    def router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """FP32 router logits for a flat ``[tokens, hidden]`` activation."""
        return F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )

    def select_experts(self, logits: torch.Tensor):
        """Top-k selection from FP32 router logits ``[tokens, num_experts]``."""
        num_tokens = logits.shape[0]
        if self.moe_router_activation_func == "sigmoid":
            scores = logits.sigmoid()
        elif self.moe_router_activation_func == "softmax":
            scores = logits.softmax(dim=1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.moe_router_activation_func}"
            )

        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        if self.num_expert_group > 1 and self.num_expert_group > self.topk_group:
            group_scores = (
                scores_for_choice.view(num_tokens, self.num_expert_group, -1)
                .topk(2, dim=-1)[0]
                .sum(dim=-1)
            )
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(num_tokens, self.num_expert_group,
                        self.num_experts // self.num_expert_group)
                .reshape(num_tokens, -1)
            )
            tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        else:
            tmp_scores = scores_for_choice

        _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
        topk_weight = scores.gather(1, topk_idx)

        if self.top_k > 1 and self.moe_renormalize:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight

    def forward(self, hidden_states: torch.Tensor):
        _, _, h = hidden_states.shape
        return self.select_experts(self.router_logits(hidden_states.view(-1, h)))


class KimiSparseMoeBlock(nn.Module):
    """Routed experts + shared expert, with optional LatentMoE down/up projection."""

    def __init__(self, config: KimiLinearConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_size
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.moe_renormalize = config.norm_topk_prob

        self.use_latent_moe = getattr(config, "routed_expert_hidden_size", None) is not None
        self.moe_hidden_size = (
            config.routed_expert_hidden_size if self.use_latent_moe else config.hidden_size
        )
        self.latent_moe_use_norm = getattr(config, "latent_moe_use_norm", False)

        self.ep_size = 1
        self.experts_per_rank = config.n_routed_experts
        self.ep_rank = 0
        # K3 ships routed experts MXFP4-packed and NO `.weight` for any of
        # them, so a BF16 KimiBlockSparseMLP here is not "unquantized" — it is
        # silently empty (k3/mxfp4_expert.py banner). Evaluated once, not per
        # expert: is_mxfp4_quantized() runs the full contract validation.
        from .k3.mxfp4_expert import is_mxfp4_quantized

        if is_mxfp4_quantized(config):
            from .k3.mxfp4_expert import K3MXFP4Expert

            def _build_expert():
                return K3MXFP4Expert(
                    config,
                    hidden_size=self.moe_hidden_size,
                    intermediate_size=config.moe_intermediate_size,
                )
        else:
            def _build_expert():
                return KimiBlockSparseMLP(
                    config,
                    hidden_size=self.moe_hidden_size,
                    intermediate_size=config.moe_intermediate_size,
                )

        self.experts = nn.ModuleList(
            [_build_expert() for _ in range(config.n_routed_experts)]
        )
        self.gate = KimiMoEGate(config)

        self.n_shared_experts = getattr(config, "n_shared_experts", None)
        if self.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * self.n_shared_experts
            self.shared_experts = KimiMLP(config=config, intermediate_size=intermediate_size)

        if self.use_latent_moe:
            self.routed_expert_down_proj = nn.Linear(
                config.hidden_size, self.moe_hidden_size, bias=False
            )
            self.routed_expert_up_proj = nn.Linear(
                self.moe_hidden_size, config.hidden_size, bias=False
            )
            if self.latent_moe_use_norm:
                self.routed_expert_norm = KimiRMSNorm(
                    self.moe_hidden_size, eps=config.rms_norm_eps
                )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        if self.use_latent_moe:
            hidden_states = self.routed_expert_down_proj(hidden_states)

        y = self.moe_infer(hidden_states, topk_idx, topk_weight)

        if self.use_latent_moe:
            if self.latent_moe_use_norm:
                y = self.routed_expert_norm(y)
            y = self.routed_expert_up_proj(y)

        y = y.view(*orig_shape)

        if self.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]

        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out


# Alias requested by the task spec.
KimiMoE = KimiSparseMoeBlock


# ============================================================================
#  MLA (NoPE) attention
# ============================================================================
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    return torch.repeat_interleave(hidden_states, dim=1, repeats=n_rep)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
):
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)

    scores = torch.einsum("bhqd,bhkd->bhqk", query, key) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : key.shape[-2]]

    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    probs = F.dropout(probs, p=dropout, training=module.training)
    out = torch.einsum("bhqk,bhkd->bhqd", probs, value).transpose(1, 2).contiguous()
    return out, probs


class KimiMLAAttention(nn.Module):
    """NoPE Multi-Latent Attention (adapted from DeepSeek-V3)."""

    def __init__(self, config: KimiLinearConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)

        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.use_nope = config.mla_use_nope
        self.scaling = self.q_head_dim ** (-0.5)

        assert self.use_nope, "KimiMLAAttention only supports NoPE (mla_use_nope=True)"

        if self.q_lora_rank is not None:
            self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
            self.q_a_layernorm = KimiRMSNorm(self.q_lora_rank)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )
        else:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.q_head_dim, bias=False
            )
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = KimiRMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)
        self.is_causal = True

        self.use_output_gate = getattr(config, "mla_use_output_gate", False)
        if self.use_output_gate:
            self.g_proj = nn.Linear(self.hidden_size, self.num_heads * self.v_head_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.q_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        if self.q_lora_rank is not None:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        else:
            q_states = self.q_proj(hidden_states)
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(
            q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(
            k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        # NoPE: the "rope" sub-dim is carried through unrotated.
        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attn_output, _ = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        if self.use_output_gate:
            g = self.g_proj(hidden_states).sigmoid()
            attn_output = attn_output * g
        return self.o_proj(attn_output)


# ============================================================================
#  KDA (Kimi Delta Attention) linear attention
# ============================================================================
class KimiLinearCache:
    """Minimal per-layer state container for KDA prefill -> decode carry-over.

    Holds three short-conv states (q, k, v) and the recurrent (delta) state per
    layer. For pure prefill parity, pass ``cache_params=None`` (no cache).
    """

    def __init__(self, num_layers: int):
        self.conv_states: List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = [
            None for _ in range(num_layers)
        ]
        self.recurrent_states: List[Optional[torch.Tensor]] = [None for _ in range(num_layers)]


class KimiKDAAttention(nn.Module):
    """Kimi Delta Attention (linear/recurrent) using fla kernels.

    prefill (q_len > 1 or no cache) -> chunk_kda
    decode  (cached and q_len == 1) -> fused_recurrent_kda
    """

    def __init__(self, config: KimiLinearConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.mode = "chunk"
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.conv_size = config.kda_conv_size
        self.head_dim = config.kda_head_dim
        self.num_heads = config.kda_num_heads
        self.head_k_dim = self.head_dim
        self.num_k_heads = self.num_heads

        projection_k_size = self.head_k_dim * self.num_k_heads
        projection_size = self.head_dim * self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, projection_k_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, projection_k_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, projection_size, bias=False)

        self.q_conv1d = ShortConvolution(
            hidden_size=projection_k_size, kernel_size=self.conv_size, activation="silu"
        )
        self.k_conv1d = ShortConvolution(
            hidden_size=projection_k_size, kernel_size=self.conv_size, activation="silu"
        )
        self.v_conv1d = ShortConvolution(
            hidden_size=projection_size, kernel_size=self.conv_size, activation="silu"
        )

        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16))
        )

        self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.dt_bias = nn.Parameter(torch.empty(projection_size, dtype=torch.float32))

        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        self.use_full_rank_gate = config.kda_use_full_rank_gate
        self.gate_lower_bound = config.kda_gate_lower_bound
        if self.use_full_rank_gate:
            self.g_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        else:
            self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
            self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.o_norm = FusedRMSNormGated(
            self.head_dim, eps=config.rms_norm_eps, activation="sigmoid"
        )
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_params: Optional[KimiLinearCache] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # This standalone module only supports the unpadded (no 2D mask) path,
        # which is what BatchGen prefill/decode uses (packed sequences via
        # cu_seqlens). Padding-mask unpad is handled by the surrounding runtime.
        if attention_mask is not None:
            raise NotImplementedError(
                "KimiKDAAttention expects a packed layout; pass cu_seqlens, not a 2D mask."
            )

        use_cache = cache_params is not None
        batch_size, q_len, _ = hidden_states.shape
        mode = "fused_recurrent" if use_cache and q_len == 1 else self.mode

        conv_state_q = conv_state_k = conv_state_v = None
        recurrent_state = None
        if cache_params is not None:
            if cache_params.conv_states[self.layer_idx] is not None:
                conv_state_q, conv_state_k, conv_state_v = cache_params.conv_states[self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]

        q_proj_states = self.q_proj(hidden_states)
        k_proj_states = self.k_proj(hidden_states)
        v_proj_states = self.v_proj(hidden_states)
        q, conv_state_q = self.q_conv1d(
            x=q_proj_states, cache=conv_state_q, output_final_state=use_cache, cu_seqlens=cu_seqlens
        )
        k, conv_state_k = self.k_conv1d(
            x=k_proj_states, cache=conv_state_k, output_final_state=use_cache, cu_seqlens=cu_seqlens
        )
        v, conv_state_v = self.v_conv1d(
            x=v_proj_states, cache=conv_state_v, output_final_state=use_cache, cu_seqlens=cu_seqlens
        )

        g = self.f_b_proj(self.f_a_proj(hidden_states))
        g = rearrange(g, "... (h d) -> ... h d", d=self.head_dim)
        beta = self.b_proj(hidden_states).float()

        q, k = map(lambda t: rearrange(t, "... (h d) -> ... h d", d=self.head_k_dim), (q, k))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)

        if mode == "chunk":
            o, recurrent_state = chunk_kda(
                q=q, k=k, v=v, g=g, beta=beta,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                safe_gate=self.gate_lower_bound is not None,
                lower_bound=self.gate_lower_bound,
                transpose_state_layout=True,
                cu_seqlens=cu_seqlens,
            )
        else:
            o, recurrent_state = fused_recurrent_kda(
                q=q, k=k, v=v, g=g, beta=beta,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                lower_bound=self.gate_lower_bound,
                transpose_state_layout=True,
                cu_seqlens=cu_seqlens,
            )

        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = recurrent_state
            cache_params.conv_states[self.layer_idx] = (conv_state_q, conv_state_k, conv_state_v)

        if self.use_full_rank_gate:
            g = self.g_proj(hidden_states)
        else:
            g = self.g_b_proj(self.g_a_proj(hidden_states))
        g = rearrange(g, "... (h d) -> ... h d", d=self.head_dim)
        o = self.o_norm(o, g)

        o = rearrange(o, "b t h d -> b t (h d)")
        return self.o_proj(o)


# ============================================================================
#  Attention-residual helper (K3 only)
# ============================================================================
# One mixer for both paths. The chunked (memory-lean) form agrees with the
# unchunked reference to < 1e-6 max_abs, NOT bitwise: every op is token-parallel
# so no reduction crosses a chunk, but a ragged final chunk (T not a multiple of
# chunk_size) is a differently-shaped tensor and ATen/cuBLAS pick a different
# batched-GEMM/reduction strategy for it. MEASURED on H20, H=512, fp32:
# torch.equal True at T in {13,1024,2048,8192}, False at T in {1025,4097} with
# max_abs 2.4e-7 (nb=3) / 2.6e-6 (nb=9). The gate is the 1e-6 tolerance in
# tests/test_kimi_k3_model.py::test_attn_res_lean_equiv, not bit equality.
# What the chunking buys is the fp32 transient: O(chunk * (nb+1) * H) instead of
# O(T * (nb+1) * H), which is what makes a packed serving prefill (T = the whole
# micro-batch) affordable. See block_residual.apply_attn_res.
_apply_attn_res = _block_residual_apply_attn_res


# ============================================================================
#  Decoder layer
# ============================================================================
class KimiDecoderLayer(nn.Module):
    def __init__(self, config: KimiLinearConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_idx = layer_idx

        if config.is_kda_layer(layer_idx):
            self.is_linear_attn = True
            self.self_attn = KimiKDAAttention(config=config, layer_idx=layer_idx)
        else:
            self.is_linear_attn = False
            self.self_attn = KimiMLAAttention(config=config, layer_idx=layer_idx)

        if (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % getattr(config, "moe_layer_freq", 1) == 0
        ):
            self.block_sparse_moe = KimiSparseMoeBlock(config)
        else:
            self.mlp = KimiMLP(config)

        self.input_layernorm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.use_attn_residuals = getattr(config, "attn_res_block_size", None) is not None
        if self.use_attn_residuals:
            self.attn_res_block_size = config.attn_res_block_size
            self.self_attention_res_norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.mlp_res_norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.self_attention_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
            self.mlp_res_proj = nn.Linear(config.hidden_size, 1, bias=False)

    def _run_attn(self, hidden_states, attention_mask, position_ids, past_key_values,
                  cu_seqlens, **kwargs):
        if self.is_linear_attn:
            return self.self_attn(
                hidden_states=hidden_states,
                attention_mask=None,
                cache_params=past_key_values,
                cu_seqlens=cu_seqlens,
                **kwargs,
            )
        return self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )

    def _run_ffn(self, hidden_states):
        if hasattr(self, "block_sparse_moe"):
            return self.block_sparse_moe(hidden_states)
        return self.mlp(hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        cu_seqlens: Optional[torch.Tensor] = None,
        block_residual: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if self.use_attn_residuals:
            return self._forward_attn_residual(
                hidden_states, attention_mask, position_ids, past_key_values,
                cu_seqlens, block_residual, **kwargs
            )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self._run_attn(
            hidden_states, attention_mask, position_ids, past_key_values, cu_seqlens, **kwargs
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self._run_ffn(hidden_states)
        hidden_states = residual + hidden_states
        # Return a 1-tuple so the BatchGen worker's prepack loop can index
        # layer_outputs[0]; the model's own forward unpacks [0] too.
        return (hidden_states,)

    def _forward_attn_residual(
        self, hidden_states, attention_mask, position_ids, past_key_values,
        cu_seqlens, block_residual, **kwargs
    ):
        batch_size, seq_len, hidden_size = hidden_states.shape
        prefix_sum = hidden_states

        if block_residual is not None and block_residual.shape[1] > 0:
            hidden_states = _apply_attn_res(
                prefix_sum.view(-1, hidden_size),
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
            ).view(batch_size, seq_len, hidden_size)

        if self.layer_idx % self.attn_res_block_size == 0:
            # Boundary: snapshot the PRE-mix prefix_sum, then RESET (assignment,
            # not add). Value-for-value the old
            #   torch.cat([block_residual, prefix_sum.unsqueeze(1)], dim=1)
            # but written into a column of the pass's preallocated buffer, so
            # the (S,nb,H) and (S,nb+1,H) tensors are never co-live — 12.25 GiB
            # of transient at the last K3 boundary. What comes back is the
            # NARROWED (S,nb+1,H) view, so shape[1] still counts boundaries.
            block_residual = BlockResidualBuffer.append(
                block_residual, prefix_sum.view(-1, hidden_size)
            )
            prefix_sum = None

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self._run_attn(
            hidden_states, attention_mask, position_ids, past_key_values, cu_seqlens, **kwargs
        )

        if prefix_sum is not None:
            prefix_sum = prefix_sum + hidden_states
        else:
            prefix_sum = hidden_states

        hidden_states = _apply_attn_res(
            prefix_sum.view(-1, hidden_size),
            block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
        ).view(batch_size, seq_len, hidden_size)

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self._run_ffn(hidden_states)

        if prefix_sum is None:
            prefix_sum = hidden_states
        else:
            # The FFN output is dead after this residual merge. Reuse the
            # surviving prefix buffer instead of allocating another full
            # (tokens, hidden) tensor; exact-64K K3 is 896 MiB per rank here.
            prefix_sum.add_(hidden_states)

        return prefix_sum, block_residual


# ============================================================================
#  Full model
# ============================================================================
def _build_causal_mask(seq_len: int, device, dtype) -> torch.Tensor:
    mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
    return mask.triu(1)


class KimiLinearModel(nn.Module):
    def __init__(self, config: KimiLinearConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [KimiDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.use_attn_residuals = getattr(config, "attn_res_block_size", None) is not None
        if self.use_attn_residuals:
            self.output_attn_res_norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.output_attn_res_proj = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        cu_seqlens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        batch_size, seq_len = hidden_states.shape[:2]

        # Eager causal mask for the MLA (full-attention) layers.
        causal_mask = None
        if seq_len > 1:
            causal_mask = _build_causal_mask(seq_len, hidden_states.device, hidden_states.dtype)

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)

        block_residual = None
        if self.use_attn_residuals:
            block_residual = self._new_block_residual(hidden_states)

        for decoder_layer in self.layers:
            layer_mask = None if decoder_layer.is_linear_attn else causal_mask
            if self.use_attn_residuals:
                hidden_states, block_residual = decoder_layer(
                    hidden_states,
                    attention_mask=layer_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    cu_seqlens=cu_seqlens,
                    block_residual=block_residual,
                    **kwargs,
                )
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=layer_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    cu_seqlens=cu_seqlens,
                    **kwargs,
                )[0]

        if self.use_attn_residuals:
            hidden_states = self._apply_output_attn_res(
                hidden_states.view(-1, self.config.hidden_size), block_residual
            ).view(batch_size, seq_len, self.config.hidden_size)
            # The pass is over. `block_residual` is a view of class-state
            # scratch, so unlike the local it replaced it does not die with
            # this frame — drop both, or the (S, 8, H) buffer stays pinned
            # (14.00 GiB at S=131,072). The carried path does the same through
            # BlockResidualCarrier.take().
            block_residual = None
            BlockResidualBuffer.reset()

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def _new_block_residual(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Seed one pass's ``block_residual`` scratch — a ZERO-column view of a
        preallocated ``(num_tokens, num_boundaries, hidden)`` buffer.

        Same contract as the ``new_zeros(num_tokens, 0, hidden)`` it replaces:
        intra-forward scratch, re-zeroed every forward, ``shape[1]`` counts the
        boundaries seen so far. What it adds is that the eight boundary appends
        no longer reallocate — see :class:`BlockResidualBuffer`.

        A method (like ``_apply_output_attn_res``) so the serving prefill loop
        in ``batchgen_worker.py`` can seed the buffer the same way without
        importing this module or knowing K3's boundary arithmetic.
        """
        hidden = self.config.hidden_size
        return BlockResidualBuffer.seed(
            hidden_states.numel() // hidden,
            hidden,
            num_block_residual_columns(self.config.num_hidden_layers,
                                       self.config.attn_res_block_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

    def _apply_output_attn_res(self, flat_hidden: torch.Tensor,
                               block_residual: torch.Tensor) -> torch.Tensor:
        """Final depth-mix over the block residuals, on a FLAT (N, hidden) view.

        Exists as a method so the serving prefill loop
        (``batchgen_worker.py``) can perform the same mix without importing
        this module's internals — the worker stays model-agnostic and there is
        exactly one implementation of the mix, so the two paths cannot drift.
        Callers apply ``self.norm`` AFTER this: mix-then-norm is the reference
        order and swapping it changes every output.
        """
        return _apply_attn_res(
            flat_hidden,
            block_residual,
            self.output_attn_res_proj,
            self.output_attn_res_norm,
        )


class KimiLinearForCausalLM(nn.Module):
    def __init__(self, config: KimiLinearConfig):
        super().__init__()
        self.config = config
        self.model = KimiLinearModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        cu_seqlens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cu_seqlens=cu_seqlens,
            **kwargs,
        )
        return self.lm_head(hidden_states)
