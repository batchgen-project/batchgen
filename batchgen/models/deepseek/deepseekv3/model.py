# ---------------------------------------------------------------------------- #
#  BatchGen                                                                     #
#  Copyright (c) EfficientMoE team 2025                                         #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
# ---------------------------------------------------------------------------- #

"""DeepSeek-R1 / DeepSeek-V3 native model definition for BatchGen.

Architecture (DeepSeek-V3 / R1):
    - 61 transformer layers (first 3 dense, layers 3-60 MoE)
    - MLA attention: 128 heads, q_lora_rank=1536, kv_lora_rank=512,
      qk_nope=128, qk_rope=64, v_head=128
    - 256 routed experts + 1 shared expert, top-8 routing
    - Group-limited "noaux_tc" routing (n_group=8, topk_group=4)
    - FP8 blockwise (e4m3, [128,128]) routed-expert weights, BF16 KV
    - Shared YaRN RoPE (single instance across all layers)

Design (mirrors moonshotai/kimi_k25/model.py structure + glm/glm5 FP8 MoE):
    - DeepseekV3ForCausalLM (outer): .model + .lm_head (worker-compatible)
    - DeepseekV3Model (inner): .embed_tokens, .layers, .norm
    - DeepseekV3Attention: structural; DeepSeekAttnWrapper handles forward
      (paged MLA decode via decoding_attn_mode_3_bf16 + decode absorb)
    - DeepseekV3MoE: EP decode via the shared 3D-strided dispatch +
      grouped_fp8_blockwise GEMM path (glm5/minimax/kimi pattern); prefill via
      per-expert wrapper loop
    - DeepSeekExpertWrapper (wrappers.py) handles FP8 expert weights; the MoE
      stacks the persistent experts' FP8 weights into [E,N,K] for grouped GEMM

The MoE decode hot path is identical in shape to Glm5MoE._forward_decode_3d;
the only DeepSeek-V3-specific difference is the group-limited gate (Kimi/GLM-5
use plain sigmoid top-k; DeepSeek-V3 masks to topk_group groups first).
"""

import logging
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from batchgen.layers.rotary_embedding import YarnRotaryEmbedding

# --- Shared 3D-strided MoE kernels (reused from batchgen/moe, glm5 pattern) ---
try:
    from batchgen.moe.grouped_fp8_blockwise_moe import (
        grouped_fp8_blockwise_s1_silu,
        grouped_fp8_blockwise_fused_s1,
        grouped_fp8_blockwise_s3,
    )
    _HAS_FP8_BLOCKWISE = True
except ImportError:
    _HAS_FP8_BLOCKWISE = False

try:
    from batchgen_kernels.moe._C_fp8_blockwise_ops import act_quant_3d
    _HAS_FP8_OPS = True
except ImportError:
    _HAS_FP8_OPS = False

try:
    from batchgen.moe.dispatch_scatter_3d import (
        dispatch_scatter_3d,
        reduce_weighted_scatter,
    )
    _HAS_DISPATCH_3D = True
except ImportError:
    _HAS_DISPATCH_3D = False

_MTP_BLOCK = 128            # align mtp to FP8 blockwise block size (TMA-friendly)
_DEFAULT_MTP = int(os.environ.get("BATCHGEN_R1_3D_MTP", "4096"))


@dataclass
class _CausalLMOutput:
    logits: torch.Tensor


# ============================================================================
# RMSNorm
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm (fused CUDA kernel with PyTorch fallback)."""

    _fused_fn = None

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    @staticmethod
    def _get_fused_fn():
        if RMSNorm._fused_fn is not None:
            return RMSNorm._fused_fn
        try:
            from batchgen_kernels.common.mgn import fused_rmsnorm
            RMSNorm._fused_fn = fused_rmsnorm
            return fused_rmsnorm
        except ImportError:
            return None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        fn = self._get_fused_fn()
        if fn is not None:
            return fn(hidden_states, self.weight, self.variance_epsilon)
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


def _yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


# ============================================================================
# MLA Attention (structural — forward handled by DeepSeekAttnWrapper)
# ============================================================================

class DeepseekV3Attention(nn.Module):
    """Multi-head Latent Attention for DeepSeek-V3 / R1.

    Structural definition only — the actual forward (paged FlashMLA decode,
    FA3 prefill, RoPE, KV cache, decode weight-absorption) is supplied by
    DeepSeekAttnWrapper + the methods monkey-patched onto this module by the PSM
    (mla_decoding_flashmla_attn_mode_3_bf16, mla_prefill_flashattention3_*, ...).

    rotary_emb is assigned by DeepseekV3Model so a single YarnRotaryEmbedding
    instance is shared across all layers.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads          # 128
        self.q_lora_rank = config.q_lora_rank                # 1536
        self.kv_lora_rank = config.kv_lora_rank              # 512
        self.qk_nope_head_dim = config.qk_nope_head_dim      # 128
        self.qk_rope_head_dim = config.qk_rope_head_dim      # 64
        self.v_head_dim = config.v_head_dim                  # 128
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim  # 192
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)

        # Q low-rank compression
        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
        )

        # KV low-rank compression (MQA-style)
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, self.hidden_size, bias=False
        )

        # Shared RoPE — assigned by DeepseekV3Model
        self.rotary_emb = None

        # MLA softmax scales (materialized / unmaterialized KV)
        self.qkv_materialized_softmax_scale = self.q_head_dim ** -0.5
        self.qkv_unmaterialized_softmax_scale = (
            self.kv_lora_rank + self.qk_rope_head_dim
        ) ** -0.5
        if config.rope_scaling is not None:
            mscale_all_dim = config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = _yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.qkv_materialized_softmax_scale *= mscale * mscale
                self.qkv_unmaterialized_softmax_scale *= mscale * mscale
        self.softmax_scale = self.qkv_materialized_softmax_scale

    def initialize(self):
        """Pre-compute absorbed kv_b projections for the decode phase.

        out_absorb / q_absorb let decode skip the explicit kv_b_proj projection
        per step (the MLA "weight absorption" trick used by Kimi/GLM-5).
        """
        if getattr(self.config, "phase", None) == "decode":
            kv_b_proj = self.kv_b_proj.weight.view(
                self.num_heads, -1, self.kv_lora_rank
            )
            self.q_absorb = kv_b_proj[:, : self.qk_nope_head_dim, :]
            self.out_absorb = kv_b_proj[:, self.qk_nope_head_dim :, :]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "DeepseekV3Attention.forward() is structural; use DeepSeekAttnWrapper."
        )


# ============================================================================
# Expert MLP (FP8 routed expert / BF16-or-FP8 shared expert)
# ============================================================================

class DeepseekV3Expert(nn.Module):
    """Single expert FFN with SiLU gating.

    Routed experts: FP8 weights, executed in decode via the stacked grouped
    blockwise GEMM (this module's gate/up/down weights are stacked into the MoE
    3D buffers). Prefill + shared-expert paths run via DeepSeekExpertWrapper,
    which calls deepgemm_forward() below.
    """

    def __init__(self, config, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    @torch.inference_mode()
    def deepgemm_forward(self, x: torch.Tensor, scale) -> torch.Tensor:
        """FP8 w8a16 expert forward (used by DeepSeekExpertWrapper).

        Mirrors DeepseekV3MLP.deepgemm_forward in modeling_deepseek_v3.py.
        """
        from batchgen.attention.mla.fa3_backend import w8a16_gemm
        up = w8a16_gemm(self.up_proj.weight.data, scale["up_proj.weight_scale_inv"], x)
        gate = w8a16_gemm(self.gate_proj.weight.data, scale["gate_proj.weight_scale_inv"], x)
        intermediate = self.act_fn(gate) * up
        return w8a16_gemm(
            self.down_proj.weight.data, scale["down_proj.weight_scale_inv"], intermediate
        )


class DenseMLP(nn.Module):
    """Dense FFN for the first `first_k_dense_replace` layers (non-MoE)."""

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    @torch.inference_mode()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )

    @torch.inference_mode()
    def deepgemm_forward(self, x: torch.Tensor, scale) -> torch.Tensor:
        from batchgen.attention.mla.fa3_backend import w8a16_gemm
        up = w8a16_gemm(self.up_proj.weight.data, scale["up_proj.weight_scale_inv"], x)
        gate = w8a16_gemm(self.gate_proj.weight.data, scale["gate_proj.weight_scale_inv"], x)
        intermediate = self.act_fn(gate) * up
        return w8a16_gemm(
            self.down_proj.weight.data, scale["down_proj.weight_scale_inv"], intermediate
        )


# ============================================================================
# MoE Gate (group-limited noaux_tc routing — DeepSeek-V3 specific)
# ============================================================================

class MoEGate(nn.Module):
    """DeepSeek-V3 / R1 router: sigmoid scoring + group-limited top-k.

    Unlike Kimi/GLM-5 (plain sigmoid + top-k, n_group=1), DeepSeek-V3 restricts
    routing to `topk_group` of `n_group` expert groups before the final top-k.
    Kept eager (fixed-shape per bucket → CUDA-graph capturable later).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size

        self.weight = nn.Parameter(torch.empty(self.n_routed_experts, self.gating_dim))
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(torch.empty(self.n_routed_experts))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @torch.inference_mode()
    def warmup(self):
        pass

    @torch.inference_mode()
    def decoding_forward(self, hidden_states: torch.Tensor):
        """Alias used by the PSM warmup path."""
        return self.forward(hidden_states)

    @torch.inference_mode()
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (topk_idx int32 [n, top_k], topk_weight fp32 [n, top_k])."""
        h = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, h)
        n = hidden_states.shape[0]
        if n == 0:
            return (
                torch.empty(0, self.top_k, dtype=torch.int32, device=hidden_states.device),
                torch.empty(0, self.top_k, dtype=torch.float32, device=hidden_states.device),
            )

        logits = F.linear(hidden_states.float(), self.weight.float(), None)
        if self.scoring_func != "sigmoid":
            raise NotImplementedError(f"unsupported scoring_func: {self.scoring_func}")
        scores = logits.sigmoid()

        if self.topk_method != "noaux_tc":
            raise NotImplementedError(f"unsupported topk_method: {self.topk_method}")

        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(n, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        )  # [n, n_group]
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(n, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(n, -1)
        )  # [n, e]
        tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
        topk_weight = scores.gather(1, topk_idx)

        if self.top_k > 1 and self.norm_topk_prob:
            denom = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denom
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx.to(torch.int32), topk_weight.float()


# ============================================================================
# Shared 3D-strided MoE buffer manager (one per model, all MoE layers share it)
# ============================================================================

class DeepseekV3MoEBufferManager:
    """Pre-allocated buffers for the 3D-strided FP8 MoE decode pipeline.

    Mirrors Glm5MoE3DBuffers (glm/glm5/model.py) / KimiK25MoEBufferManager.
    Each expert e owns rows [e*mtp, (e+1)*mtp) in the per-expert buffers;
    dispatch scatters tokens into those slots, the grouped blockwise GEMM runs
    in-place, and reduce gathers them back via topk_pos.
    """

    def __init__(
        self,
        E_local: int,
        max_global_bsz: int,
        H: int,
        N_inter: int,
        topk: int,
        num_tokens_per_rank: int,
        device: torch.device,
        max_tokens_padded: int = _DEFAULT_MTP,
    ):
        self.E_local = E_local
        self.H = H
        self.N_inter = N_inter
        self.topk = topk
        self.max_global_bsz = max_global_bsz
        self.num_tokens_per_rank = num_tokens_per_rank
        self.device = device
        self.max_tokens_padded = max_tokens_padded

        NK = max_global_bsz * topk
        buf_rows = E_local * max_tokens_padded

        self.all_tokens = torch.zeros(max_global_bsz, H, dtype=torch.bfloat16, device=device)
        self.padded = torch.zeros(num_tokens_per_rank, H, dtype=torch.bfloat16, device=device)
        self.expert_counts = torch.zeros(E_local, dtype=torch.int32, device=device)
        self.expert_counters = torch.zeros(E_local, dtype=torch.int32, device=device)
        self.topk_pos = torch.full((NK,), -1, dtype=torch.int32, device=device)
        self.dispatched_x = torch.zeros(buf_rows, H, dtype=torch.bfloat16, device=device)
        self.expert_out = torch.zeros(buf_rows, H, dtype=torch.bfloat16, device=device)
        self.result_buffer = torch.empty(max_global_bsz, H, dtype=torch.bfloat16, device=device)

        total_bytes = sum(
            t.nelement() * t.element_size()
            for t in (self.all_tokens, self.padded, self.expert_counts,
                      self.expert_counters, self.topk_pos, self.dispatched_x,
                      self.expert_out, self.result_buffer)
        )
        logging.info(
            f"[DeepseekV3MoEBufferManager] E_local={E_local}, mtp={max_tokens_padded}, "
            f"buf_rows={buf_rows}, H={H}, N_inter={N_inter}, "
            f"total={total_bytes / (1024**3):.2f} GiB"
        )

    def resize_if_needed(self, global_bsz: int):
        grew_comm = global_bsz > self.max_global_bsz
        grew_mtp = global_bsz > self.max_tokens_padded
        if not grew_comm and not grew_mtp:
            return
        if grew_comm:
            logging.info(
                f"[DeepseekV3MoEBufferManager] Resizing comm buffers: "
                f"{self.max_global_bsz} -> {global_bsz}")
            self.max_global_bsz = global_bsz
            NK = global_bsz * self.topk
            self.all_tokens = torch.zeros(global_bsz, self.H, dtype=torch.bfloat16, device=self.device)
            self.topk_pos = torch.full((NK,), -1, dtype=torch.int32, device=self.device)
            self.result_buffer = torch.empty(global_bsz, self.H, dtype=torch.bfloat16, device=self.device)
        if grew_mtp:
            new_mtp = ((global_bsz + _MTP_BLOCK - 1) // _MTP_BLOCK) * _MTP_BLOCK
            logging.info(
                f"[DeepseekV3MoEBufferManager] Resizing 3D buffers: "
                f"mtp {self.max_tokens_padded} -> {new_mtp}")
            self.max_tokens_padded = new_mtp
            buf_rows = self.E_local * new_mtp
            self.dispatched_x = torch.zeros(buf_rows, self.H, dtype=torch.bfloat16, device=self.device)
            self.expert_out = torch.zeros(buf_rows, self.H, dtype=torch.bfloat16, device=self.device)


def _moe_3d_blockwise_supported(
    experts_per_rank: int,
    num_persistent_local_experts: int,
    enable_ep_offloading: bool,
) -> bool:
    """3D FP8 blockwise MoE path. Supported for all-resident (every local routed
    expert on GPU, num_persistent==E) AND for mixed ep-offload (persistent subset
    resident, offloaded experts streamed per step into a shared buffer)."""
    return (
        int(num_persistent_local_experts) == int(experts_per_rank)
        or bool(enable_ep_offloading)
    )


# ============================================================================
# MoE layer (256 routed + 1 shared expert) — EP decode + prefill
# ============================================================================

class DeepseekV3MoE(nn.Module):
    """DeepSeek-V3 / R1 MoE layer.

    Decode path (EP): AllGather -> group gate -> 3D dispatch -> FP8 blockwise
    grouped GEMM -> weighted reduce -> AllReduce -> slice + shared expert.
    Mirrors Glm5MoE._forward_decode_3d; only the gate is group-limited.

    Class variables (shared across all MoE layers, set by PSM):
        _buf:               shared DeepseekV3MoEBufferManager
        _rank_token_counts: [world_size] real token count per rank (pad masking)
    """

    _buf: Optional["DeepseekV3MoEBufferManager"] = None
    _rank_token_counts: Optional[torch.Tensor] = None
    # Shared offloaded-expert weight staging buffers (n_offloaded rows), filled
    # per layer per step by streaming from the host ring. ONE set for the whole
    # model, reused across all MoE layers (allocated by the first layer's init).
    _offload_gate_w3d: Optional[torch.Tensor] = None
    _offload_up_w3d: Optional[torch.Tensor] = None
    _offload_down_w3d: Optional[torch.Tensor] = None
    _warned_hot_path = False
    _warned_gemm_3d = False
    _warned_weights_stacked = False
    _warned_partial_3d = False

    def __init__(self, config, layer_idx: int = -1, comm=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts            # 256
        self.top_k = config.num_experts_per_tok               # 8
        self.num_experts_per_tok = config.num_experts_per_tok
        self.moe_intermediate_size = config.moe_intermediate_size  # 2048
        self.comm = comm

        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank, self.world_size = 0, 1

        self.experts_per_rank = self.num_experts // self.world_size
        self.total_experts = self.world_size * self.experts_per_rank
        self.routed_expert_start_idx = self.rank * self.experts_per_rank
        self.routed_expert_end_idx = (self.rank + 1) * self.experts_per_rank

        self.device = torch.device("cuda", self.rank % torch.cuda.device_count())
        self.num_tokens_per_rank = None

        # Set by PSM. Default: all local experts resident, no offloading.
        self.enable_ep_offloading = False
        self.num_persistent_local_experts = self.experts_per_rank
        self.use_3d_moe = True
        self._fp8_blockwise_ready = False

        self.gate = MoEGate(config)

        # Routed experts: real modules only at local indices (EP); None elsewhere.
        if self.world_size > 1 and dist.is_initialized():
            self.experts = nn.ModuleList([
                DeepseekV3Expert(config, self.hidden_size, self.moe_intermediate_size)
                if self.routed_expert_start_idx <= i < self.routed_expert_end_idx else None
                for i in range(self.total_experts)
            ])
        else:
            self.experts = nn.ModuleList([
                DeepseekV3Expert(config, self.hidden_size, self.moe_intermediate_size)
                for _ in range(self.total_experts)
            ])

        n_shared = getattr(config, "n_shared_experts", 1)
        self.shared_experts = DeepseekV3Expert(
            config, self.hidden_size, self.moe_intermediate_size * n_shared
        )

    # ── token-count management (called by PSM) ──

    def init_num_tokens(self, num_tokens_per_rank: int):
        self.num_tokens_per_rank = num_tokens_per_rank
        global_num_tokens = num_tokens_per_rank * self.world_size
        if (
            self.use_3d_moe
            and _HAS_DISPATCH_3D
            and DeepseekV3MoE._buf is None
            and _moe_3d_blockwise_supported(
                self.experts_per_rank,
                self.num_persistent_local_experts,
                self.enable_ep_offloading,
            )
        ):
            # Size the per-expert 3D stride (mtp) to the actual global batch, NOT
            # the 4096 default: with 32 experts/rank on a single node the FP8
            # weights already use ~90 GB, so a 4096-mtp pool (dispatched_x +
            # expert_out ~3.8 GB) OOMs. Worst case one expert receives every
            # routed token, so mtp >= global_num_tokens; resize_if_needed grows
            # it later. Mirrors the Kimi single-node OOM-mtp fix.
            mtp = max(
                ((global_num_tokens + _MTP_BLOCK - 1) // _MTP_BLOCK) * _MTP_BLOCK,
                _MTP_BLOCK,
            )
            DeepseekV3MoE._buf = DeepseekV3MoEBufferManager(
                E_local=self.experts_per_rank,
                max_global_bsz=global_num_tokens,
                H=self.hidden_size,
                N_inter=self.moe_intermediate_size,
                topk=self.top_k,
                num_tokens_per_rank=num_tokens_per_rank,
                device=self.device,
                max_tokens_padded=mtp,
            )

    def set_num_tokens_per_rank(self, num_tokens_per_rank: int):
        if num_tokens_per_rank == self.num_tokens_per_rank:
            return
        self.num_tokens_per_rank = num_tokens_per_rank
        if self.use_3d_moe and DeepseekV3MoE._buf is not None:
            buf = DeepseekV3MoE._buf
            buf.resize_if_needed(num_tokens_per_rank * self.world_size)
            # send buffer MUST match num_tokens_per_rank exactly (all_gather sends
            # input.numel() per rank); see Glm5MoE.set_num_tokens_per_rank note.
            if buf.padded.shape[0] != num_tokens_per_rank:
                buf.padded = torch.zeros(
                    num_tokens_per_rank, buf.H, dtype=torch.bfloat16, device=buf.device
                )
                buf.num_tokens_per_rank = num_tokens_per_rank

    # ── FP8 weight stacking (called by PSM via init) ──

    def init(self, micro_batch_size=None):
        """Stack persistent experts' FP8 weights into [E,N,K] for grouped GEMM."""
        if not (self.use_3d_moe and _HAS_FP8_BLOCKWISE):
            return
        if not _moe_3d_blockwise_supported(
            self.experts_per_rank, self.num_persistent_local_experts, self.enable_ep_offloading
        ):
            self._fp8_blockwise_ready = False
            if not DeepseekV3MoE._warned_partial_3d:
                logging.warning(
                    "[DeepseekV3MoE] 3D FP8 MoE disabled: persistent=%s/%s "
                    "ep_offloading=%s; falling back to per-expert decode.",
                    self.num_persistent_local_experts, self.experts_per_rank,
                    self.enable_ep_offloading,
                )
                DeepseekV3MoE._warned_partial_3d = True
            return
        self._init_fp8_blockwise_weights()

    def _init_fp8_blockwise_weights(self):
        """Stack per-expert FP8 weights + per-block scales into 3D tensors.

        Reads FP8 weights from the DeepSeekExpertWrapper (wrapper.fp8_gate/up/down)
        and scales from wrapper.weight_dequant_scale. Uses the copy-and-rebind
        trick (mirrors Glm5MoE._init_fp8_blockwise_weights) so each per-expert
        original frees before the next copy — without it 2x the stacked weights
        (~150 GB/rank) would coexist and OOM the 96 GB H20.
        """
        E = self.experts_per_rank
        K = self.hidden_size                    # 7168
        N = self.moe_intermediate_size          # 2048
        scale_block = 128
        k_blocks = K // scale_block             # 56
        n_blocks = N // scale_block             # 16
        k_blocks_pad4 = (k_blocks + 3) // 4 * 4
        n_blocks_pad4 = (n_blocks + 3) // 4 * 4
        start = self.routed_expert_start_idx
        device = self.device

        n_persistent = self.num_persistent_local_experts
        wrappers = [self.experts[start + i] for i in range(E)]
        # Persistent experts [0:n_persistent] have resident FP8 weights and stack
        # into the per-layer w3d (memory win = n_persistent rows, NOT E). Offloaded
        # experts [n_persistent:E] stream per step; only their (tiny) scales are
        # resident. Require resident weights for the persistent experts only.
        for i in range(n_persistent):
            if getattr(wrappers[i], "fp8_gate", None) is None:
                raise RuntimeError(
                    f"[DeepseekV3MoE] layer {self.layer_idx} persistent expert "
                    f"{start + i}: FP8 weights not registered before stacking."
                )

        # The per-expert FP8 weight has TWO live references: wrapper.<attr> and
        # wrapper.module.<proj>.weight.data (same tensor). Rebind BOTH to the
        # stacked view so the original frees (refcount 0) before the next copy —
        # otherwise stacked + originals = 2x (~164 GB) and OOMs. (glm5 pattern.)
        proj_of = {"fp8_gate": "gate_proj", "fp8_up": "up_proj", "fp8_down": "down_proj"}

        def _stack_w(attr: str):
            proj = proj_of[attr]
            shape = getattr(wrappers[0], attr).shape
            w3d = torch.empty((n_persistent, *shape), dtype=getattr(wrappers[0], attr).dtype, device=device)
            for i in range(n_persistent):
                w = wrappers[i]
                w3d[i].copy_(getattr(w, attr))
                view = w3d[i]
                setattr(w, attr, view)
                getattr(w.module, proj).weight.data = view
            return w3d

        self.fp8_gate_w3d = _stack_w("fp8_gate")
        self.fp8_up_w3d = _stack_w("fp8_up")
        self.fp8_down_w3d = _stack_w("fp8_down")

        self.fp8_gate_ws3d = torch.zeros(E, n_blocks, k_blocks_pad4, dtype=torch.float32, device=device)
        self.fp8_up_ws3d = torch.zeros(E, n_blocks, k_blocks_pad4, dtype=torch.float32, device=device)
        self.fp8_down_ws3d = torch.zeros(E, k_blocks, n_blocks_pad4, dtype=torch.float32, device=device)
        for i, w in enumerate(wrappers):
            self.fp8_gate_ws3d[i, :, :k_blocks] = w.weight_dequant_scale["gate_proj.weight_scale_inv"]
            self.fp8_up_ws3d[i, :, :k_blocks] = w.weight_dequant_scale["up_proj.weight_scale_inv"]
            self.fp8_down_ws3d[i, :, :n_blocks] = w.weight_dequant_scale["down_proj.weight_scale_inv"]

        # Allocate the shared offloaded-expert weight staging buffer once (first
        # MoE layer to init under ep-offload). Holds n_offloaded experts' FP8
        # weights; reused across all layers each decode step (streamed in Pass 2).
        n_offloaded = E - n_persistent
        if n_offloaded > 0 and DeepseekV3MoE._offload_gate_w3d is None:
            DeepseekV3MoE._offload_gate_w3d = torch.empty(
                (n_offloaded, *self.fp8_gate_w3d.shape[1:]),
                dtype=self.fp8_gate_w3d.dtype, device=device)
            DeepseekV3MoE._offload_up_w3d = torch.empty(
                (n_offloaded, *self.fp8_up_w3d.shape[1:]),
                dtype=self.fp8_up_w3d.dtype, device=device)
            DeepseekV3MoE._offload_down_w3d = torch.empty(
                (n_offloaded, *self.fp8_down_w3d.shape[1:]),
                dtype=self.fp8_down_w3d.dtype, device=device)
            logging.info(
                f"[DeepseekV3MoE] offload staging buffer: {n_offloaded} experts, "
                f"gate={list(DeepseekV3MoE._offload_gate_w3d.shape)}")

        # Free the offloaded experts' GPU-resident module weights. Under Method A
        # the offloaded experts [n_persistent:E] stream from the host ring into the
        # shared staging buffer each decode step (stream_weights_into); their
        # per-expert module.weight copies are never read on the 3D path, so any
        # resident copy is dead memory competing with the decode KV pool. (Persistent
        # experts were rebound to w3d above; offloaded ones were never loaded/rebound,
        # so their construction/skeleton weights linger on GPU.)
        freed_mb = 0.0
        for i in range(n_persistent, E):
            w = wrappers[i]
            for proj in ("gate_proj", "up_proj", "down_proj"):
                p = getattr(w.module, proj).weight
                if p.data.is_cuda and p.data.numel() > 0:
                    freed_mb += p.data.numel() * p.data.element_size() / (1024 ** 2)
                    p.data = torch.empty(0, dtype=p.data.dtype, device=p.data.device)
        if freed_mb > 0 and not DeepseekV3MoE._warned_weights_stacked:
            logging.info(
                f"[DeepseekV3MoE] freed {freed_mb:.1f} MB offloaded-expert resident "
                f"weights / layer (~{freed_mb * self.config.num_hidden_layers / 1024:.1f} "
                f"GiB across all layers) -> reclaimed for KV pool")

        self._fp8_blockwise_ready = True
        if not DeepseekV3MoE._warned_weights_stacked:
            logging.info(
                f"[DeepseekV3MoE] FP8 blockwise 3D weights ready: "
                f"gate={list(self.fp8_gate_w3d.shape)}, down={list(self.fp8_down_w3d.shape)}, "
                f"gate_scale={list(self.fp8_gate_ws3d.shape)}"
            )
            DeepseekV3MoE._warned_weights_stacked = True

    # ── forward ──

    @torch.inference_mode()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if getattr(self.config, "phase", "decode") == "decode":
            if (self.use_3d_moe and self._fp8_blockwise_ready
                    and DeepseekV3MoE._buf is not None and _HAS_DISPATCH_3D):
                return self._forward_decode_3d(hidden_states)
            return self._forward_prefill(hidden_states)  # fallback (per-expert loop)
        return self._forward_prefill(hidden_states)

    @torch.inference_mode()
    def _forward_decode_3d(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """EP decode: AllGather -> group gate -> 3D dispatch -> FP8 GEMM ->
        weighted scatter -> AllReduce -> slice + shared expert."""
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        identity = hidden_states
        num_tokens, hidden_size = hidden_states.shape

        if num_tokens > self.num_tokens_per_rank:
            raise RuntimeError(
                f"MoE buffer overflow: num_tokens={num_tokens} > "
                f"num_tokens_per_rank={self.num_tokens_per_rank}"
            )

        ntp = self.num_tokens_per_rank
        num_global = ntp * self.world_size
        topk = self.top_k
        buf = DeepseekV3MoE._buf
        buf.resize_if_needed(num_global)

        if not DeepseekV3MoE._warned_hot_path:
            logging.warning("[DeepseekV3MoE] HOT PATH: 3D dispatch + FP8 blockwise GEMM")
            DeepseekV3MoE._warned_hot_path = True

        # 1) AllGather padded local tokens -> all_tokens
        all_tokens = buf.all_tokens[:num_global]
        padded = buf.padded
        padded.zero_()
        if num_tokens > 0:
            padded[:num_tokens] = hidden_states
        with self.comm.change_state(enable=True):
            self.comm.all_gather(
                all_tokens, padded, stream=torch.cuda.current_stream(self.device)
            )

        # 2) Group-limited gate on global tokens
        topk_idx, topk_weight = self.gate(all_tokens)

        # 2b) Mask rank-padding tokens so they don't inflate expert_counts.
        rank_counts = DeepseekV3MoE._rank_token_counts
        if rank_counts is not None:
            positions = torch.arange(num_global, device=self.device)
            rank_ids = positions // ntp
            local_pos = positions % ntp
            padding_mask = local_pos >= rank_counts[rank_ids]
            padding_mask_2d = padding_mask.unsqueeze(1).expand_as(topk_idx)
            topk_idx = torch.where(padding_mask_2d, torch.full_like(topk_idx, -1), topk_idx)
            topk_weight = torch.where(padding_mask_2d, torch.zeros_like(topk_weight), topk_weight)

        # 3) 3D dispatch scatter into strided buffer
        buf.dispatched_x.zero_()
        expert_counts, topk_pos = dispatch_scatter_3d(
            all_tokens, topk_idx.to(torch.int32),
            buf.dispatched_x,
            self.routed_expert_start_idx, self.experts_per_rank,
            buf.max_tokens_padded,
            buf.expert_counts, buf.expert_counters,
            buf.topk_pos[:num_global * topk],
        )

        # 4) FP8 blockwise grouped GEMM (in-place on buf.expert_out). Mixed
        #    ep-offload splits into a persistent pass + a streamed-offloaded pass.
        if self.enable_ep_offloading:
            self._mixed_expert_gemm(buf, expert_counts)
        else:
            self._fp8_blockwise_gemm_3d(buf, expert_counts)

        # 5) Weighted scatter reduce
        result_buf = buf.result_buffer[:num_global]
        result_buf.zero_()
        global_results = reduce_weighted_scatter(
            buf.expert_out, topk_pos, topk_weight,
            num_global, hidden_size, topk, output=result_buf,
        )

        # 6) AllReduce across EP ranks
        with self.comm.change_state(enable=True):
            self.comm.all_reduce(
                global_results, op=dist.ReduceOp.SUM,
                stream=torch.cuda.current_stream(self.device),
            )

        # 7) Slice local tokens + add shared expert
        if num_tokens == 0:
            return torch.empty(orig_shape, device=self.device, dtype=hidden_states.dtype)
        start = self.rank * ntp
        out = global_results[start:start + num_tokens].to(hidden_states.dtype)
        out = out + self.shared_expert_forward(identity)
        return out.view(*orig_shape)

    @torch.inference_mode()
    def _fp8_blockwise_gemm_3d(self, buf, expert_counts):
        """All-resident FP8 blockwise grouped GEMM over all E_local experts
        (mirrors Glm5MoE._fp8_blockwise_gemm_3d). Reads buf.dispatched_x ->
        buf.expert_out via the shared _grouped_gemm helper."""
        if not DeepseekV3MoE._warned_gemm_3d:
            logging.warning(
                f"[DeepseekV3MoE] HOT PATH: _fp8_blockwise_gemm_3d (act_quant_3d={_HAS_FP8_OPS})")
            DeepseekV3MoE._warned_gemm_3d = True
        self._grouped_gemm(
            self.fp8_gate_w3d, self.fp8_up_w3d, self.fp8_down_w3d,
            self.fp8_gate_ws3d, self.fp8_up_ws3d, self.fp8_down_ws3d,
            buf.dispatched_x, expert_counts,
            self.experts_per_rank, 0, buf.max_tokens_padded, buf.expert_out,
        )

    @torch.inference_mode()
    def _mixed_expert_gemm(self, buf, expert_counts):
        """Method A (two-pass) mixed ep-offload GEMM. Pass 1: grouped GEMM over
        the n_persistent resident experts. Pass 2: stream the n_offloaded experts'
        FP8 weights from the host ring into the shared staging buffer (in task
        order, EVERY expert incl 0-token per the weight-buffer copy contract),
        then one grouped GEMM over them. Both write the matching bucket range of
        buf.expert_out so the downstream reduce/scatter is unchanged."""
        if not DeepseekV3MoE._warned_gemm_3d:
            logging.warning("[DeepseekV3MoE] HOT PATH: mixed ep-offload GEMM "
                            "(persistent grouped + streamed-offloaded grouped)")
            DeepseekV3MoE._warned_gemm_3d = True

        n_persistent = self.num_persistent_local_experts
        E = self.experts_per_rank
        mtp = buf.max_tokens_padded

        # Pass 1: persistent experts [0:n_persistent]
        if n_persistent > 0:
            self._grouped_gemm(
                self.fp8_gate_w3d, self.fp8_up_w3d, self.fp8_down_w3d,
                self.fp8_gate_ws3d[:n_persistent], self.fp8_up_ws3d[:n_persistent],
                self.fp8_down_ws3d[:n_persistent],
                buf.dispatched_x, expert_counts,
                n_persistent, 0, mtp, buf.expert_out,
            )

        # Pass 2: stream offloaded experts [n_persistent:E] into the staging
        # buffer (task order = ascending expert idx), then grouped GEMM.
        n_offloaded = E - n_persistent
        if n_offloaded > 0:
            start = self.routed_expert_start_idx
            for j in range(n_offloaded):
                wrapper = self.experts[start + n_persistent + j]
                wrapper.stream_weights_into(
                    DeepseekV3MoE._offload_gate_w3d[j],
                    DeepseekV3MoE._offload_up_w3d[j],
                    DeepseekV3MoE._offload_down_w3d[j],
                )
            self._grouped_gemm(
                DeepseekV3MoE._offload_gate_w3d, DeepseekV3MoE._offload_up_w3d,
                DeepseekV3MoE._offload_down_w3d,
                self.fp8_gate_ws3d[n_persistent:E], self.fp8_up_ws3d[n_persistent:E],
                self.fp8_down_ws3d[n_persistent:E],
                buf.dispatched_x, expert_counts,
                n_offloaded, n_persistent, mtp, buf.expert_out,
            )

    @torch.inference_mode()
    def _grouped_gemm(self, gate_w, up_w, down_w, gate_s, up_s, down_s,
                      dispatched_x, expert_counts, n_exp, expert_offset, mtp, out):
        """Grouped FP8 blockwise GEMM over `n_exp` experts.

        Reads dispatched_x buckets [expert_offset : expert_offset+n_exp] (each
        mtp-strided), weights gate_w/up_w/down_w[n_exp] + scales gate_s/up_s/down_s
        [n_exp] (all 0-based over the n_exp experts), writes the same bucket range
        of `out`. Resident path: expert_offset=0, n_exp=E_local. Mixed (offload)
        path calls it twice: persistent [0:n_persistent] then offloaded
        [n_persistent:E_local] with the streamed offloaded weight buffer."""
        K = self.hidden_size
        N = self.moe_intermediate_size
        E = n_exp
        lo = expert_offset * mtp
        hi = (expert_offset + E) * mtp
        x = dispatched_x[lo:hi]
        cu_seqlens = torch.arange(
            0, (E + 1) * mtp, mtp, dtype=torch.int32, device=x.device)
        seqlens = expert_counts[expert_offset:expert_offset + E]
        avg = max(mtp // max(E, 1), 1)

        # input activation quant
        if _HAS_FP8_OPS:
            x_3d = x.view(E, mtp, K)
            x_quant_3d, x_scale_3d = act_quant_3d(x_3d, seqlens)
            x_quant = x_quant_3d.view(E * mtp, K)
            x_scale_t = x_scale_3d.view(E * mtp, -1).t().contiguous()
        else:
            from batchgen.attention.mla.fa3_backend import act_quant as _act_quant
            x_quant, x_scale = _act_quant(x)
            x_scale_t = x_scale.t().contiguous()

        # S1: gate + up + SiLU -> BF16 intermediate
        if _HAS_FP8_OPS:
            s1 = grouped_fp8_blockwise_fused_s1(
                x_quant.view(torch.float8_e4m3fn), x_scale_t,
                gate_w.view(torch.float8_e4m3fn),
                up_w.view(torch.float8_e4m3fn),
                gate_s, up_s,
                seqlens, cu_seqlens, avg,
            )
            inter_quant_3d, inter_scale_3d = act_quant_3d(s1.view(E, mtp, N), seqlens)
            inter_quant = inter_quant_3d.view(E * mtp, N)
            inter_scale_t = inter_scale_3d.view(E * mtp, -1).t().contiguous()
        else:
            from batchgen.attention.mla.fa3_backend import act_quant as _act_quant
            intermediate = grouped_fp8_blockwise_s1_silu(
                x_quant.view(torch.float8_e4m3fn), x_scale_t,
                gate_w.view(torch.float8_e4m3fn),
                up_w.view(torch.float8_e4m3fn),
                gate_s, up_s,
                seqlens, cu_seqlens, avg,
            )
            inter_quant, inter_scale = _act_quant(intermediate)
            inter_scale_t = inter_scale.t().contiguous()

        # S3: down projection
        result = grouped_fp8_blockwise_s3(
            inter_quant.view(torch.float8_e4m3fn), inter_scale_t,
            down_w.view(torch.float8_e4m3fn),
            down_s,
            seqlens, cu_seqlens, avg,
        )
        out[lo:hi].copy_(result[:E * mtp])

    def shared_expert_forward(self, identity: torch.Tensor) -> torch.Tensor:
        return self.shared_experts(identity)

    @torch.inference_mode()
    def _forward_prefill(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Prefill / fallback: per-expert wrapper loop (local experts) + shared."""
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        identity = hidden_states
        num_tokens, hidden_size = hidden_states.shape
        device = hidden_states.device
        topk = self.top_k

        topk_idx, topk_weight = self.gate(identity)

        flat_expert_idx = topk_idx.reshape(-1).long()
        token_indices = torch.arange(num_tokens, device=device).repeat_interleave(topk)
        topk_positions = torch.arange(topk, device=device).repeat(num_tokens)

        results = torch.zeros(num_tokens, hidden_size, device=device, dtype=torch.float32)
        for expert_idx in range(self.routed_expert_start_idx, self.routed_expert_end_idx):
            expert = self.experts[expert_idx]
            if expert is None:
                continue
            mask = flat_expert_idx == expert_idx
            if not mask.any():
                continue
            tok = token_indices[mask]
            pos = topk_positions[mask]
            expert_out = expert(hidden_states[tok])
            w = topk_weight[tok, pos]
            results.index_add_(0, tok, expert_out.float() * w.unsqueeze(-1))

        results = results.to(hidden_states.dtype)
        results = results + self.shared_expert_forward(identity)
        return results.view(*orig_shape)


# ============================================================================
# Decoder layer
# ============================================================================

class DeepseekV3DecoderLayer(nn.Module):
    """Pre-norm transformer layer. Layers < first_k_dense_replace are dense."""

    _fused_add_rmsnorm_fn = None

    def __init__(self, config, layer_idx: int, comm=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = DeepseekV3Attention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if layer_idx < config.first_k_dense_replace:
            self.mlp = DenseMLP(config)
        else:
            self.mlp = DeepseekV3MoE(config, layer_idx=layer_idx, comm=comm)

    @staticmethod
    def _get_fused_add_rmsnorm_fn():
        if DeepseekV3DecoderLayer._fused_add_rmsnorm_fn is not None:
            return DeepseekV3DecoderLayer._fused_add_rmsnorm_fn
        try:
            from batchgen.other_kernels.cuda_rmsnorm import cuda_add_rmsnorm
            DeepseekV3DecoderLayer._fused_add_rmsnorm_fn = cuda_add_rmsnorm
            return cuda_add_rmsnorm
        except Exception:
            return None

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        # Drained DP rank: 0 local decode tokens (this rank's sequences all
        # finished while other ranks keep decoding). The fused RMSNorm / MLA
        # kernels raise "CUDA kernel error" on a 0-row input, so skip all local
        # compute. MoE layers MUST still run self.mlp so their EP all_gather /
        # all_reduce stay in lockstep with the other ranks; dense layers have no
        # collective and return unchanged.
        if hidden_states.shape[0] == 0:
            if isinstance(self.mlp, DeepseekV3MoE):
                hidden_states = self.mlp(hidden_states)
            return (hidden_states, None, None)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(hidden_states=hidden_states)
        hidden_states = attn_out[0] if isinstance(attn_out, tuple) else attn_out

        fused_add_norm = self._get_fused_add_rmsnorm_fn()
        if fused_add_norm is not None:
            hidden_states, residual = fused_add_norm(
                residual, hidden_states,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.variance_epsilon,
            )
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return (hidden_states, None, None)


# ============================================================================
# Inner model + outer ForCausalLM
# ============================================================================

class DeepseekV3Model(nn.Module):
    """Inner transformer (embed_tokens, layers, norm). Shares one YaRN RoPE."""

    def __init__(self, config, comm=None):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        rope_scaling = config.rope_scaling or {}
        self._shared_rotary_emb = YarnRotaryEmbedding(
            dim=config.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_factor=rope_scaling.get("factor", 1.0),
            original_max_position_embeddings=rope_scaling.get("original_max_position_embeddings", 4096),
            beta_fast=rope_scaling.get("beta_fast", 32.0),
            beta_slow=rope_scaling.get("beta_slow", 1.0),
            mscale=rope_scaling.get("mscale"),
            mscale_all_dim=rope_scaling.get("mscale_all_dim"),
        )

        self.layers = nn.ModuleList([
            DeepseekV3DecoderLayer(config, layer_idx=i, comm=comm)
            for i in range(config.num_hidden_layers)
        ])
        for layer in self.layers:
            layer.self_attn.rotary_emb = self._shared_rotary_emb

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Must provide either input_ids or inputs_embeds")
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        for layer_idx, layer in enumerate(self.layers):
            layer_output = layer(hidden_states, layer_idx=layer_idx)
            hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output
        # Guard the final RMSNorm against a drained DP rank (0-row fused kernel
        # crashes); see DeepseekV3DecoderLayer.forward.
        if hidden_states.shape[0] == 0:
            return hidden_states
        return self.norm(hidden_states)


class DeepseekV3ForCausalLM(nn.Module):
    """DeepSeek-V3 / R1 with LM head. Exposes .model and .lm_head for the worker."""

    def __init__(self, config, comm=None):
        super().__init__()
        self.config = config
        self.model = DeepseekV3Model(config, comm=comm)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )
        logits = self.lm_head(hidden_states).float()
        return _CausalLMOutput(logits=logits)

    def eval(self):
        return super().eval()
