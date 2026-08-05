# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3                                                           #
#  copyright (c) EfficientMoE team 2025                                         #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""Kimi-K3 decoder — M2: PREFILL-ONLY, eager, BF16 (+ FP32 islands).

Faithful nn.Module port of the checkpoint's own reference
(``modeling_kimi_linear.py``, md5 4e3de36ab2a5de1232c05ce346a3426e; the K3
VLM wrapper ``modeling_kimi_k3.py`` adds no text-model override).  Line
references in comments are to that file ("ML:<line>").  The per-layer
shape/dtype contract is ``batchgen_design/model_support/kimi_k3/
ACTIVATION_FLOW.md``.

Architecture (93 layers at full size):
  * 69 KDA (Kimi Delta Attention, full-rank output gate) + 24 NoPE-MLA
    (q-LoRA + sigmoid output gate); config layer lists are 1-BASED.
  * Layer 0 is KDA + dense SiTU MLP; layers 1.. are LatentMoE (router on the
    7168 hidden, experts in a 3584 latent, 2-wide shared expert in hidden
    space) with sigmoid noaux_tc top-16.
  * Block Attention Residuals: prefix_sum / hidden_states DIVERGE — the
    depth-mix output feeds the norms and never the accumulator; at each
    block boundary prefix_sum RESETS (assignment, not add).  The depth mixer
    here is the memory-LEAN form (POIS decision 2, 2026-08-04): it never
    materializes the (T, nb+1, hidden) fp32 ``k`` tensor of the reference —
    that form is load-bearing for the unchunked single-rank prefill bound.

Deliberately shared nothing with ``kimi_linear/model.py`` at import time:
that module imports fla at module scope (kimi_linear/model.py:43-44) and is
therefore un-importable on CPU dev machines and in file-path-loaded tests.
The pure-torch components below are ports of the same oracle the kimi_linear
module ports; provenance comments mark each.

Hard-fail policy (POIS decision 1): decode-shaped calls, vision tokens,
padding masks, varlen, training, position_ids — every unsupported input
raises, naming the milestone that will implement it.  No silent fallbacks.

Dtype policy (ACTIVATION_FLOW §5.4 / checkpoint facts): ``A_log``,
``dt_bias``, ``{q,k,v}_conv1d.weight``, ``o_norm.weight`` and
``gate.e_score_correction_bias`` ship FP32 and STAY FP32 — see
``K3_FP32_PARAM_SUFFIXES`` and :func:`cast_model_to_inference_dtype`.
"""

from __future__ import annotations

import inspect
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import KimiK3Config

__all__ = [
    "KimiK3Config",
    "KimiK3ForCausalLM",
    "KimiK3Model",
    "KimiK3DecoderLayer",
    "KimiK3KDAAttention",
    "KimiK3MLAAttention",
    "KimiMoEGate",
    "KimiSparseMoeBlock",
    "KimiMLP",
    "KimiBlockSparseMLP",
    "KimiRMSNorm",
    "KimiGatedRMSNormSigmoid",
    "CausalConv1dSilu",
    "SituAndMul",
    "K3_FP32_PARAM_SUFFIXES",
    "cast_model_to_inference_dtype",
]


# --------------------------------------------------------------------------- #
#  Hard-fail messages (asserted by tests/test_kimi_k3_model.py)                #
# --------------------------------------------------------------------------- #
M3_DECODE_MSG = (
    "Kimi-K3 M2 is PREFILL-ONLY: KV-cache / recurrent-state decode arrives in "
    "M3. Reject the request at admission instead of calling the model with "
    "cache state."
)
M4_VARLEN_MSG = (
    "Kimi-K3 M2 supports only a dense equal-length batch (attention_mask=None, "
    "cu_seqlens=None). Packed/varlen prefill arrives in M4."
)
VISION_MSG = (
    "Kimi-K3 vision (MoonViT tower / mm_projector) is not implemented "
    "(post-M7): the prompt contains the media placeholder token. Reject "
    "multimodal requests at admission."
)
NOPE_POSITION_MSG = (
    "Kimi-K3 is NoPE — no module consumes position_ids (the reference has no "
    "rotary at all, ML:403). Pass position_ids=None; a non-None value would be "
    "silently ignored, which this build refuses to do."
)

#: Checkpoint tensors that ship FP32 and stay FP32 through inference
#: (kimi_linear/k3/tensor_map.py tensor_dtypes + ACTIVATION_FLOW §5.4-G).
K3_FP32_PARAM_SUFFIXES = (
    "q_conv1d.weight",
    "k_conv1d.weight",
    "v_conv1d.weight",
    "A_log",
    "dt_bias",
    "o_norm.weight",
    "gate.e_score_correction_bias",
)

#: MLA q_a/kv_a layernorms are constructed WITHOUT an eps argument in the
#: reference (ML:368/383) and therefore take KimiRMSNorm's default 1e-6 —
#: while every other norm reads config.rms_norm_eps (1e-5).  Module-level so
#: the mutation suite can flip it.
MLA_LORA_LAYERNORM_EPS = 1e-6


def cast_model_to_inference_dtype(model: nn.Module,
                                  dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    """Cast every parameter to ``dtype`` EXCEPT the FP32 set (which the
    checkpoint ships FP32 and the kernels consume FP32).  A blanket
    ``model.to(bf16)`` would silently downcast ``A_log``/``dt_bias`` — the KDA
    gate transform is an fp32 island and must see fp32 parameters."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith(K3_FP32_PARAM_SUFFIXES):
                if param.dtype != torch.float32:
                    raise RuntimeError(
                        "FP32-set parameter {} was constructed as {} — the "
                        "construction path is broken".format(name, param.dtype))
                continue
            param.data = param.data.to(dtype)
    return model


# --------------------------------------------------------------------------- #
#  Norms                                                                       #
# --------------------------------------------------------------------------- #
class KimiRMSNorm(nn.Module):
    """RMSNorm, fp32 interior; the gain multiplies AFTER the downcast (ML:232-236)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        x = hidden_states.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(dtype)


class KimiGatedRMSNormSigmoid(nn.Module):
    """Per-head gated RMSNorm — pure-torch equivalent of fla's
    ``FusedRMSNormGated(head_dim, eps, activation='sigmoid')`` (the oracle's
    ``o_norm``, ML:539-540).  Entire op fp32, including sigmoid(g); weight is
    applied to the normalized x BEFORE the gate (fla fused_norm_gate.py:92-104);
    store dtype = input dtype.  Weight ships FP32 in the K3 checkpoint."""

    def __init__(self, head_dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim, dtype=torch.float32))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        g32 = gate.float()
        y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        y = y * self.weight.float()
        y = y * torch.sigmoid(g32)
        return y.to(x.dtype)


# --------------------------------------------------------------------------- #
#  SiTU                                                                        #
# --------------------------------------------------------------------------- #
class SituAndMul(nn.Module):
    """SiTU-and-multiply (ML:64-82), an FP32 island.

        gate, up = split(x)              # concatenated [gate, up] input
        situ_a   = beta * tanh(gate/beta) * sigmoid(gate)     # sigmoid of RAW gate
        up       = linear_beta * tanh(up/linear_beta)
        out      = (situ_a * up).to(x.dtype)

    Both betas are REQUIRED: the reference's ``beta or 1.0`` idiom (ML:91)
    would turn a legitimate 0.0 into 1.0 and is refused here (flag F).
    """

    def __init__(self, beta: float, linear_beta: float):
        super().__init__()
        if beta is None or linear_beta is None:
            raise ValueError(
                "SituAndMul requires explicit beta and linear_beta (K3: 4.0 / "
                "25.0); refusing the `beta or 1.0` fallback of the reference")
        self.beta = float(beta)
        self.linear_beta = float(linear_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = x[..., :d].to(torch.float32)
        up = x[..., d:].to(torch.float32)
        situ_a = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (situ_a * up).to(x.dtype)


def _build_situ(config: KimiK3Config) -> SituAndMul:
    if config.hidden_act != "situ":
        raise NotImplementedError(
            "Kimi-K3 model only implements hidden_act='situ' (got {!r})".format(
                config.hidden_act))
    return SituAndMul(config.activation_situ_beta, config.activation_situ_linear_beta)


# --------------------------------------------------------------------------- #
#  MLPs                                                                        #
# --------------------------------------------------------------------------- #
class KimiMLP(nn.Module):
    """Dense SiTU MLP (gate_proj/up_proj/down_proj) — layer 0 and the shared
    experts (ML:274-301).  The activation input is the CONCATENATED
    [gate, up]."""

    def __init__(self, config: KimiK3Config, hidden_size: Optional[int] = None,
                 intermediate_size: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = _build_situ(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = torch.cat([self.gate_proj(x), self.up_proj(x)], dim=-1)
        return self.down_proj(self.act_fn(gate_up))


class KimiBlockSparseMLP(nn.Module):
    """Routed-expert MLP in the LATENT space (w1=gate, w2=down, w3=up,
    per checkpoint naming; ML:246-270)."""

    def __init__(self, config: KimiK3Config, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)   # gate
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)   # down
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)   # up
        self.act_fn = _build_situ(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = torch.cat([self.w1(hidden_states), self.w3(hidden_states)], dim=-1)
        return self.w2(self.act_fn(gate_up))


# --------------------------------------------------------------------------- #
#  MoE gate + sparse block                                                     #
# --------------------------------------------------------------------------- #
class KimiMoEGate(nn.Module):
    """Sigmoid noaux_tc router, all-FP32 (ML:703-759).

    Selection uses scores + e_score_correction_bias; the combine WEIGHTS are
    gathered from the PRE-bias scores (ML:750).  K3 pins num_expert_group=1,
    which makes the reference's group-limited branch dead code (`1 > 1` is
    False, ML:724-746) — plain top-k is the faithful implementation, and a
    config that would enable the branch is rejected at parse time."""

    def __init__(self, config: KimiK3Config):
        super().__init__()
        if int(config.num_expert_group or 1) != 1 or int(config.topk_group or 1) != 1:
            raise NotImplementedError(
                "KimiMoEGate implements only the K3 case num_expert_group=="
                "topk_group==1 (the reference's group branch is dead code there)")
        if config.moe_router_activation_func != "sigmoid":
            raise NotImplementedError(
                "KimiMoEGate implements only the sigmoid scoring function")
        self.top_k = int(config.num_experts_per_token)
        self.num_experts = int(config.num_experts)
        self.routed_scaling_factor = float(config.routed_scaling_factor)
        self.moe_renormalize = bool(config.moe_renormalize)
        self.gating_dim = int(config.hidden_size)
        self.weight = nn.Parameter(torch.empty((self.num_experts, self.gating_dim)))
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(self.num_experts, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None)
        scores = logits.sigmoid()

        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        _, topk_idx = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)
        topk_weight = scores.gather(1, topk_idx)          # PRE-bias scores (ML:750)

        if self.top_k > 1 and self.moe_renormalize:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight                      # topk_weight stays fp32


def _moe_combine(new_x: torch.Tensor, topk_shape, topk_weight: torch.Tensor
                 ) -> torch.Tensor:
    """FP32 top-k combine (ML:867-873).  Factored out as the mutation seam for
    ``combine_bf16``."""
    return (
        new_x.view(*topk_shape, -1)
        .type(topk_weight.dtype)          # fp32
        .mul_(topk_weight.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )


class KimiSparseMoeBlock(nn.Module):
    """LatentMoE (ML:768-874): router on the PRE-down 7168 hidden;
    routed_expert_down_proj applied ONCE per token before dispatch; experts in
    the latent space; routed_expert_norm + up_proj on the combined output;
    ONE shared-expert KimiMLP (num_shared_experts is a WIDTH multiplier) added
    in hidden space."""

    def __init__(self, config: KimiK3Config):
        super().__init__()
        if config.routed_expert_hidden_size is None or not config.latent_moe_use_norm:
            raise NotImplementedError(
                "KimiSparseMoeBlock implements only the K3 LatentMoE form "
                "(routed_expert_hidden_size set, latent_moe_use_norm=True)")
        self.num_experts = int(config.num_experts)
        self.top_k = int(config.num_experts_per_token)
        self.moe_hidden_size = int(config.routed_expert_hidden_size)

        self.experts = nn.ModuleList([
            KimiBlockSparseMLP(
                config,
                hidden_size=self.moe_hidden_size,
                intermediate_size=int(config.moe_intermediate_size),
            )
            for _ in range(self.num_experts)
        ])
        self.gate = KimiMoEGate(config)
        self.shared_experts = KimiMLP(
            config,
            intermediate_size=int(config.moe_intermediate_size) * int(config.num_shared_experts),
        )
        self.routed_expert_down_proj = nn.Linear(
            config.hidden_size, self.moe_hidden_size, bias=False)
        self.routed_expert_up_proj = nn.Linear(
            self.moe_hidden_size, config.hidden_size, bias=False)
        self.routed_expert_norm = KimiRMSNorm(self.moe_hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)   # router sees 7168 hidden
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        hidden_states = self.routed_expert_down_proj(hidden_states)   # once per token
        y = self.moe_infer(hidden_states, topk_idx, topk_weight)
        y = self.routed_expert_norm(y)
        y = self.routed_expert_up_proj(y)
        y = y.view(*orig_shape)
        return y + self.shared_experts(identity)

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        """Dense per-expert loop (ML:840-874).  The argsort need not be stable:
        the inverse scatter new_x[idxs] = outs cancels any permutation among
        equal expert ids, so the output is deterministic (flag I)."""
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
            expert_out = self.experts[i](sorted_tokens[start_idx:end_idx])
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        return _moe_combine(new_x, topk_ids.shape, topk_weight)


# --------------------------------------------------------------------------- #
#  MLA (NoPE, q-LoRA, gated) attention                                         #
# --------------------------------------------------------------------------- #
def _nope_join(q_pass, q_rot, k_pass, k_rot):
    """NoPE seam: the 64-wide 'rope' sub-dim is concatenated UNROTATED
    (ML:439-440 — no rotary is applied anywhere).  Factored out so the
    mutation suite can inject a rotation and prove the tests see it."""
    query_states = torch.cat((q_pass, q_rot), dim=-1)
    key_states = torch.cat((k_pass, k_rot), dim=-1)
    return query_states, key_states


def eager_attention_forward(query, key, value, attention_mask, scaling):
    """Eager attention (ML:311-332): bf16 scores + mask, FP32 softmax,
    downcast, bf16 PV.  K3 has num_key_value_heads == num_attention_heads so
    the reference's repeat_kv is a no-op (enforced at config parse) and is
    omitted here."""
    scores = torch.einsum("bhqd,bhkd->bhqk", query, key) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : key.shape[-2]]
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.einsum("bhqk,bhkd->bhqd", probs, value).transpose(1, 2).contiguous()
    return out


class KimiK3MLAAttention(nn.Module):
    """NoPE Multi-Latent Attention with q-LoRA and sigmoid output gate
    (ML:340-474)."""

    def __init__(self, config: KimiK3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scaling = self.q_head_dim ** (-0.5)

        if not config.mla_use_nope:
            raise NotImplementedError(
                "KimiK3MLAAttention only implements NoPE (mla_use_nope=True); "
                "the rotary path does not exist in the reference either (ML:403)")
        if self.q_lora_rank is None:
            raise NotImplementedError(
                "KimiK3MLAAttention requires q_lora_rank (K3: 1536); the direct "
                "q_proj variant belongs to the 48B, not K3")
        if not config.mla_use_output_gate:
            raise NotImplementedError(
                "KimiK3MLAAttention requires mla_use_output_gate=True (the "
                "checkpoint ships g_proj on every MLA layer)")

        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        # No eps argument in the reference (ML:368/383) -> default 1e-6, NOT
        # config.rms_norm_eps.
        self.q_a_layernorm = KimiRMSNorm(self.q_lora_rank, eps=MLA_LORA_LAYERNORM_EPS)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = KimiRMSNorm(self.kv_lora_rank, eps=MLA_LORA_LAYERNORM_EPS)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)
        self.g_proj = nn.Linear(self.hidden_size, self.num_heads * self.v_head_dim, bias=False)

    def _apply_output_gate(self, attn_output: torch.Tensor,
                           hidden_states: torch.Tensor) -> torch.Tensor:
        """Sigmoid output gate from the POST-input_layernorm hidden, in model
        dtype (bf16 — deliberately NOT an fp32 island, ML:470-472), PRE-o_proj."""
        return attn_output * self.g_proj(hidden_states).sigmoid()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
    ) -> torch.Tensor:
        if past_key_values is not None:
            raise NotImplementedError(M3_DECODE_MSG)
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.q_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(
            q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(
            k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # ONE shared rope-carrier head, broadcast across all heads (ML:435-437).
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states, key_states = _nope_join(q_pass, q_rot, k_pass, k_rot)

        attn_output = eager_attention_forward(
            query_states, key_states, value_states, attention_mask, self.scaling)
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self._apply_output_gate(attn_output, hidden_states)
        return self.o_proj(attn_output)


# --------------------------------------------------------------------------- #
#  KDA (Kimi Delta Attention)                                                  #
# --------------------------------------------------------------------------- #
class CausalConv1dSilu(nn.Module):
    """Causal depthwise conv + SiLU — pure-torch equivalent of fla's
    ``ShortConvolution(hidden, W, activation='silu')`` (fla short_conv.py:
    ``nn.Conv1d(D, D, W, groups=D, bias=False, padding=W-1)`` + silu).
    Weight [D, 1, W] ships FP32 in the K3 checkpoint and stays FP32; the op is
    computed fp32 and stored in the input dtype (the triton kernel's
    contract).  Prefill-only: no cache in, no cache out."""

    def __init__(self, hidden_size: int, kernel_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(
            torch.empty(hidden_size, 1, kernel_size, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> (B, T, D)
        seq_len = x.shape[1]
        y = F.conv1d(
            x.transpose(1, 2).float(),
            self.weight,
            bias=None,
            groups=self.hidden_size,
            padding=self.kernel_size - 1,
        )[..., :seq_len]
        return F.silu(y).transpose(1, 2).to(x.dtype)


#: The kwarg that makes the fla chunk kernel apply ``sigmoid(beta)`` itself.
#: Its PRESENCE is the capability probe below — see ``_import_chunk_kda``.
_BETA_SIGMOID_KWARG = "use_beta_sigmoid_in_kernel"


def _import_chunk_kda():
    """Lazy fla import — fla is triton/GPU-only and must not poison CPU
    importability of this module.  NEVER falls back."""
    try:
        from fla.ops.kda import chunk_kda  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "fla-core (>= 0.5.0) is required for the KDA prefill kernel "
            "(kda_backend='fla_chunk'). CPU parity tests must construct the "
            "model with kda_backend='reference' explicitly — there is no "
            "automatic fallback."
        ) from exc

    # fla < 0.5 accepts the oracle's call form and computes something ELSE:
    # `use_beta_sigmoid_in_kernel` lands in **kwargs and nothing in that chunk
    # path sigmoids beta, so unbounded logits enter the delta rule.  Finite,
    # plausible, wrong — the failure class this project bans.
    #
    # The probe is the SIGNATURE, not `fla.__version__`: the version string is
    # exactly what misled us here (the vendored torch reference in
    # kda_reference.py was labelled 0.4.2 while being 0.5.2 byte-for-byte).
    # A kernel that names this parameter is one that honours it.
    if _BETA_SIGMOID_KWARG not in inspect.signature(chunk_kda).parameters:
        raise RuntimeError(
            "fla's chunk_kda does not accept {!r}, so it does NOT apply "
            "sigmoid(beta) and would silently consume raw beta logits "
            "(fla < 0.5.0 behaves this way; the kwarg is swallowed by "
            "**kwargs rather than rejected). Install fla-core >= 0.5.0. "
            "Refusing to run rather than produce plausible wrong "
            "numerics.".format(_BETA_SIGMOID_KWARG))
    return chunk_kda


class KimiK3KDAAttention(nn.Module):
    """Kimi Delta Attention, prefill chunk form (ML:478-663), full-rank output
    gate (K3: use_full_rank_gate=True, so a single ``g_proj`` and NO
    g_a_proj/g_b_proj).

    ``kda_backend``:
      * ``"fla_chunk"`` (default) — fla's triton ``chunk_kda`` with the
        byte-identical flag set of the oracle call (ML:609-627).  The oracle
        was written against fla >= 0.5 semantics, where
        ``use_beta_sigmoid_in_kernel=True`` makes the kernel apply
        ``sigmoid(beta)``; ``_import_chunk_kda`` refuses any fla that does not
        name that parameter, because older ones consume beta RAW without
        complaining.
      * ``"reference"`` — the vendored pure-torch composition in
        ``kda_reference.py``.  Parity/testing only; never auto-selected.
    """

    #: l2-normalize q/k inside the kernel (oracle flag, ML:619).  Class-level
    #: so the mutation suite can flip it for BOTH backends at once.
    use_qk_l2norm = True

    def __init__(self, config: KimiK3Config, layer_idx: int,
                 kda_backend: str = "fla_chunk"):
        super().__init__()
        if kda_backend not in ("fla_chunk", "reference"):
            raise ValueError(
                "kda_backend must be 'fla_chunk' or 'reference', got {!r}".format(kda_backend))
        if not config.kda_use_full_rank_gate:
            raise NotImplementedError(
                "KimiK3KDAAttention implements only the full-rank gate (the K3 "
                "checkpoint ships one g_proj per KDA layer); the low-rank "
                "g_a/g_b pair belongs to the 48B")
        self.kda_backend = kda_backend
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.conv_size = config.kda_conv_size
        self.head_dim = config.kda_head_dim
        self.num_heads = config.kda_num_heads
        self.gate_lower_bound = config.kda_gate_lower_bound
        self.a_log_padded_len = int(config.a_log_padded_len)

        projection_size = self.head_dim * self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, projection_size, bias=False)

        self.q_conv1d = CausalConv1dSilu(projection_size, self.conv_size)
        self.k_conv1d = CausalConv1dSilu(projection_size, self.conv_size)
        self.v_conv1d = CausalConv1dSilu(projection_size, self.conv_size)

        # Checkpoint fact (tensor_map.py / ACTIVATION_FLOW D2): A_log ships
        # F32[128] — a [num_heads] vector zero-padded.  The reference's [96]
        # allocation (ML:520-521) cannot strict-load its own checkpoint.
        # Entries [:num_heads] are consumed; the pad is inert (proved by
        # test_a_log_pad_poison and GPU Part C).
        a_log = torch.zeros(self.a_log_padded_len, dtype=torch.float32)
        a_log[: self.num_heads] = torch.log(
            torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16))
        self.A_log = nn.Parameter(a_log)

        self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(projection_size, dtype=torch.float32))
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.g_proj = nn.Linear(self.hidden_size, projection_size, bias=False)

        self.o_norm = KimiGatedRMSNormSigmoid(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False)

    # --- small accessors: mutation seams AND the single place the padded
    # --- checkpoint layouts are interpreted ---
    def _a_log(self) -> torch.Tensor:
        """The live [:num_heads] slice of the padded F32[128] buffer."""
        return self.A_log[: self.num_heads]

    def _dt_bias(self) -> torch.Tensor:
        """Flat F32[num_heads*head_dim], viewed (H, K) row-major by consumers."""
        return self.dt_bias

    def _beta(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """RAW beta logits, fp32 at the call site (ML:603); sigmoid is applied
        inside the kernel / reference."""
        return self.b_proj(hidden_states).float()

    def _run_kda_core(self, q, k, v, g_raw, beta_raw):
        if self.kda_backend == "fla_chunk":
            chunk_kda = _import_chunk_kda()
            o, _ = chunk_kda(
                q=q, k=k, v=v, g=g_raw, beta=beta_raw,
                A_log=self._a_log(),
                dt_bias=self._dt_bias(),
                initial_state=None,
                output_final_state=True,          # oracle passes True (ML:618)
                use_qk_l2norm_in_kernel=self.use_qk_l2norm,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,  # load-bearing; guarded at import
                safe_gate=self.gate_lower_bound is not None,
                lower_bound=self.gate_lower_bound,
                transpose_state_layout=True,
                cu_seqlens=None,
            )
            return o
        from .kda_reference import kda_reference_prefill  # noqa: PLC0415
        return kda_reference_prefill(
            q, k, v, g_raw, beta_raw,
            A_log=self._a_log(),
            dt_bias=self._dt_bias(),
            lower_bound=self.gate_lower_bound,
            use_qk_l2norm=self.use_qk_l2norm,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_params=None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cache_params is not None:
            raise NotImplementedError(M3_DECODE_MSG)
        if attention_mask is not None or cu_seqlens is not None:
            raise NotImplementedError(M4_VARLEN_MSG)

        batch_size, seq_len, _ = hidden_states.shape
        h, k_dim = self.num_heads, self.head_dim

        q = self.q_conv1d(self.q_proj(hidden_states))
        k = self.k_conv1d(self.k_proj(hidden_states))
        v = self.v_conv1d(self.v_proj(hidden_states))

        g_raw = self.f_b_proj(self.f_a_proj(hidden_states))
        g_raw = g_raw.view(batch_size, seq_len, h, k_dim)
        beta_raw = self._beta(hidden_states)

        q = q.view(batch_size, seq_len, h, k_dim)
        k = k.view(batch_size, seq_len, h, k_dim)
        v = v.view(batch_size, seq_len, h, k_dim)

        o = self._run_kda_core(q, k, v, g_raw, beta_raw)

        g_out = self.g_proj(hidden_states).view(batch_size, seq_len, h, k_dim)
        o = self.o_norm(o, g_out)
        o = o.reshape(batch_size, seq_len, h * k_dim)
        return self.o_proj(o)


# --------------------------------------------------------------------------- #
#  Block Attention Residuals — memory-LEAN depth mixer                         #
# --------------------------------------------------------------------------- #
def _attn_res_score_weight(proj: nn.Linear, norm: KimiRMSNorm) -> torch.Tensor:
    """Rank-1 key-projection fold: norm.weight ⊙ proj.weight (ML:1084), fp32."""
    return norm.weight.float() * proj.weight.squeeze(0).float()


def _apply_attn_res_lean(prefix_sum: torch.Tensor,
                         block_residual: torch.Tensor,
                         proj: nn.Linear,
                         norm: KimiRMSNorm,
                         chunk_size: int = 1024) -> torch.Tensor:
    """Memory-lean Block-Attention-Residual depth mixer (POIS decision 2).

    Reference semantics (ML:1075-1088): per token, 1 query over nb+1 keys,
    fp32 throughout; ``scores[:, j] = ((v_j * rsqrt(mean(v_j^2)+eps)) . w)``
    with ``w = norm.weight ⊙ proj.weight`` (the rank-1 fold the reference
    itself does at ML:1084); the value matmul uses the UNNORMALIZED fp32 v.
    Every op is token-parallel, so the mixer is evaluated in token CHUNKS with
    the verbatim reference op order inside each chunk: nothing of shape
    (T, nb+1, hidden) is ever materialized in fp32 — peak fp32 scratch is
    O(chunk_size * (nb+1) * hidden), which is what moves the unchunked
    single-rank prefill upper bound.

    Chunking along tokens changes no reduction (variance, score-dot, softmax
    and the value matmul are all per-token), so the result is BIT-IDENTICAL to
    the reference; the further algebraic fold ``(v @ w) * rsqrt(var)`` was
    measured at max_abs 1.07e-6 > the 1e-6 gate at (T,nb,H)=(4096,8,1024) and
    is deliberately NOT used.  Equivalence is gated at max_abs_diff < 1e-6 in
    fp32 (tests/test_kimi_k3_model.py::test_attn_res_lean_equiv) and the
    no-materialization property at test_attn_res_lean_no_materialization.

    prefix_sum:     (num_tokens, hidden)
    block_residual: (num_tokens, num_blocks, hidden)   [num_blocks may be 0]
    """
    num_tokens, hidden = prefix_sum.shape
    eps = norm.variance_epsilon
    w = _attn_res_score_weight(proj, norm)
    out = torch.empty_like(prefix_sum)
    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        v = torch.cat(
            (block_residual[start:end], prefix_sum[start:end].unsqueeze(1)),
            dim=1).float()                                     # (c, nb+1, H) fp32
        k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
        scores = (k * w).sum(-1)                                 # (c, nb+1)
        probs = scores.softmax(-1).unsqueeze(1)                  # (c, 1, nb+1)
        out[start:end] = torch.matmul(probs, v).squeeze(1).to(out.dtype)
    return out


# --------------------------------------------------------------------------- #
#  Decoder layer                                                               #
# --------------------------------------------------------------------------- #
class KimiK3DecoderLayer(nn.Module):
    """K3 decoder layer — Block-Attention-Residual body ONLY (ML:973-1046).
    The classic pre-norm residual body (ML:936-971) is dead code for K3 and is
    deliberately not carried.

    Contract (ACTIVATION_FLOW §3.1): input ``hidden_states`` is the previous
    layer's prefix_sum; returns ``(prefix_sum, block_residual)``.  The mixed
    tensor feeds the norms only — the accumulator NEVER sees it; at a block
    boundary (layer_idx % block_size == 0) the PRE-mix prefix_sum is appended
    to block_residual and the accumulator resets (assignment, not add)."""

    def __init__(self, config: KimiK3Config, layer_idx: int,
                 kda_backend: str = "fla_chunk"):
        super().__init__()
        if config.attn_res_block_size is None:
            raise ValueError(
                "K3 requires Block Attention Residuals (attn_res_block_size); "
                "the classic residual body is not implemented")
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.attn_res_block_size = int(config.attn_res_block_size)

        self.is_linear_attn = config.is_kda_layer(layer_idx)
        if self.is_linear_attn:
            self.self_attn = KimiK3KDAAttention(config, layer_idx, kda_backend=kda_backend)
        else:
            self.self_attn = KimiK3MLAAttention(config, layer_idx)

        if layer_idx >= int(config.first_k_dense_replace):
            self.block_sparse_moe = KimiSparseMoeBlock(config)
        else:
            self.mlp = KimiMLP(config)

        self.input_layernorm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attention_res_norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp_res_norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attention_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.mlp_res_proj = nn.Linear(config.hidden_size, 1, bias=False)

    def _run_attn(self, hidden_states, attention_mask):
        # Both mixer kinds take the same call. KDA ignores the causal mask
        # (its recurrence is causal by construction) and the model already
        # passes None for KDA layers (ML:1194-1195), so there is nothing to
        # branch on here.
        return self.self_attn(hidden_states, attention_mask=attention_mask)

    def _run_ffn(self, hidden_states):
        if hasattr(self, "block_sparse_moe"):
            return self.block_sparse_moe(hidden_states)
        return self.mlp(hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        block_residual: torch.Tensor,
    ):
        batch_size, seq_len, hidden_size = hidden_states.shape
        prefix_sum = hidden_states                                     # ML:985

        if block_residual.shape[1] > 0:                                # ML:987
            hidden_states = _apply_attn_res_lean(
                prefix_sum.view(-1, hidden_size),
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
            ).view(batch_size, seq_len, hidden_size)

        if self.layer_idx % self.attn_res_block_size == 0:             # ML:995
            # Boundary: snapshot the PRE-mix prefix_sum, then RESET.
            block_residual = torch.cat(
                [block_residual, prefix_sum.view(-1, hidden_size).unsqueeze(1)], dim=1)
            prefix_sum = None                                          # ML:998

        hidden_states = self.input_layernorm(hidden_states)            # ML:1000
        hidden_states = self._run_attn(hidden_states, attention_mask)  # ML:1003

        if prefix_sum is not None:                                     # ML:1023-1026
            prefix_sum = prefix_sum + hidden_states
        else:
            prefix_sum = hidden_states

        hidden_states = _apply_attn_res_lean(                          # ML:1028-1033
            prefix_sum.view(-1, hidden_size),
            block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
        ).view(batch_size, seq_len, hidden_size)

        hidden_states = self.post_attention_layernorm(hidden_states)   # ML:1035
        hidden_states = self._run_ffn(hidden_states)                   # ML:1036

        prefix_sum = prefix_sum + hidden_states                        # ML:1041-1044
        return prefix_sum, block_residual


# --------------------------------------------------------------------------- #
#  Full model                                                                  #
# --------------------------------------------------------------------------- #
def _build_causal_mask(seq_len: int, device, dtype) -> torch.Tensor:
    """Additive causal bias for the eager MLA path.  -inf above the diagonal;
    under the fp32 softmax this is exactly equivalent to the finfo.min bias
    transformers' create_causal_mask builds (exp underflows to 0 either way)."""
    mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
    return mask.triu(1)


class KimiK3Model(nn.Module):
    def __init__(self, config: KimiK3Config, kda_backend: str = "fla_chunk"):
        super().__init__()
        config.validate()   # defensive; parse_k3_config already ran this
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                         config.pad_token_id)
        self.layers = nn.ModuleList([
            KimiK3DecoderLayer(config, layer_idx, kda_backend=kda_backend)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output_attn_res_norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output_attn_res_proj = nn.Linear(config.hidden_size, 1, bias=False)

    # --- guards -----------------------------------------------------------
    def _guard_prefill_only(self, past_key_values, attention_mask, position_ids):
        """Hard-fail perimeter (POIS decision 1).  Factored out as the mutation
        seam for ``hard_fail_removed_decode``."""
        if past_key_values is not None:
            raise NotImplementedError(M3_DECODE_MSG)
        if attention_mask is not None:
            raise NotImplementedError(M4_VARLEN_MSG)
        if position_ids is not None:
            raise ValueError(NOPE_POSITION_MSG)
        if self.training:
            raise RuntimeError(
                "KimiK3Model is an inference-only module; call .eval()")

    def _guard_no_vision_tokens(self, input_ids: torch.Tensor) -> None:
        if (input_ids == self.config.media_placeholder_token_id).any():
            raise NotImplementedError(VISION_MSG)

    def _initial_block_residual(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Intra-forward scratch, RE-ZEROED every forward (ML:1188-1192).
        Factored out as the mutation seam for ``block_residual_carried``."""
        batch, seq_len, hidden = inputs_embeds.shape
        return inputs_embeds.new_zeros(batch * seq_len, 0, hidden)

    def _finalize(self, hidden_states: torch.Tensor,
                  block_residual: torch.Tensor) -> torch.Tensor:
        """Output-stage depth mix (output_attn_res_* params, ML:1215-1233),
        THEN the final norm (ML:1219) — in that order."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states = _apply_attn_res_lean(
            hidden_states.view(-1, hidden_size),
            block_residual,
            self.output_attn_res_proj,
            self.output_attn_res_norm,
        ).view(batch_size, seq_len, hidden_size)
        return self.norm(hidden_states)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
    ) -> torch.Tensor:
        self._guard_prefill_only(past_key_values, attention_mask, position_ids)
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Pass exactly one of input_ids / inputs_embeds")
        if input_ids is not None:
            self._guard_no_vision_tokens(input_ids)
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        batch_size, seq_len = hidden_states.shape[:2]

        causal_mask = None
        if seq_len > 1:
            causal_mask = _build_causal_mask(
                seq_len, hidden_states.device, hidden_states.dtype)

        block_residual = self._initial_block_residual(hidden_states)

        for decoder_layer in self.layers:
            layer_mask = None if decoder_layer.is_linear_attn else causal_mask
            hidden_states, block_residual = decoder_layer(
                hidden_states, layer_mask, block_residual)

        return self._finalize(hidden_states, block_residual)


class KimiK3ForCausalLM(nn.Module):
    def __init__(self, config: KimiK3Config, kda_backend: str = "fla_chunk"):
        super().__init__()
        self.config = config
        self.model = KimiK3Model(config, kda_backend=kda_backend)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        # BF16 head, no fp32 cast (ML:1298-1301).
        return self.lm_head(hidden_states)

    def configure_decoding(self, *args, **kwargs):
        raise NotImplementedError(M3_DECODE_MSG)
