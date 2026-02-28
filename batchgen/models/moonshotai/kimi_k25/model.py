# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) EfficientMoE team 2025                                         #
#                                                                               #
#  Licensed under the Apache License, Version 2.0 (the "License");              #
#  you may not use this file except in compliance with the License.             #
#                                                                               #
#  You may obtain a copy of the License at                                      #
#                                                                               #
#                  http://www.apache.org/licenses/LICENSE-2.0                   #
#                                                                               #
#  Unless required by applicable law or agreed to in writing, software          #
#  distributed under the License is distributed on an "AS IS" BASIS,            #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.     #
#  See the License for the specific language governing permissions and          #
#  limitations under the License.                                               #
# ---------------------------------------------------------------------------- #

"""Kimi K2.5 model definition following BatchGen design pattern.

Architecture:
    - 61 transformer layers (1 dense + 60 MoE)
    - MLA attention with 64 heads, kv_lora_rank=512
    - 384 routed experts + 1 shared expert per MoE layer
    - INT4 W4A16 quantization (routed experts only)
    - Shared YaRN RoPE (single instance across all layers)
    - RMSNorm (eps=1e-6)

Design:
    - KimiK25ForCausalLM (outer): .model + .lm_head (worker-compatible)
    - KimiK25Model (inner): .embed_tokens, .layers, .norm
    - Wrappers handle optimized forward (INT4 dequant, FlashAttention, KV cache)
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple

from dataclasses import dataclass
from batchgen.layers.rotary_embedding import YarnRotaryEmbedding
from batchgen.moe.routing import gate_sigmoid_topk_cuda
from batchgen.moe.fused_int4_wgmma_grouped import (
    _load_int4_grouped_module,
    create_tma_descriptor,
)
from batchgen.moe.int4_single_expert_wgmma import single_expert_int4_forward
from batchgen.moe.dispatch_scatter_3d import (
    dispatch_scatter_3d,
    reduce_weighted_scatter,
)


@dataclass
class _CausalLMOutput:
    """Minimal output container with .logits attribute for worker compatibility."""
    logits: torch.Tensor


# ============================================================================
# MoE Buffer Manager (shared across all MoE layers)
# ============================================================================

_BLOCK_M = 64  # TMA constraint: global tensor M >= BLOCK_M
_DEFAULT_MTP = 4096  # Default max_tokens_padded (stride per expert in 3D buffer)


class KimiK25MoEBufferManager:
    """Pre-allocated buffers for K2.5 MoE decode pipeline (3D strided layout).

    One instance per model, shared across all 60 MoE layers via class variable.

    Buffer layout: [E_local * max_tokens_padded, dim] (3D strided).
    Each expert e owns rows [e * mtp, (e+1) * mtp) in the activation buffer.
    Dispatch scatter writes tokens directly into fixed-stride slots.
    WGMMA kernel reads/writes inplace using expert_idx * mtp addressing.
    Reduce reads from strided positions via topk_pos.

    Single shared TMA descriptor covers the full [E*mtp, dim] buffer.
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

        # Communication buffers only (debug: no 3D GEMM buffers)
        self.all_tokens = torch.zeros(max_global_bsz, H, dtype=torch.bfloat16, device=device)
        self.padded = torch.zeros(num_tokens_per_rank, H, dtype=torch.bfloat16, device=device)

        logging.info(
            f"[MoEBufferManager] Comm-only mode: max_global_bsz={max_global_bsz}, "
            f"num_tokens_per_rank={num_tokens_per_rank}, H={H}"
        )

    def _init_tma_descriptors(self):
        """Create TMA descriptors for dispatched_x and intermediate buffers."""
        try:
            self.tma_dispatched = create_tma_descriptor(self.dispatched_x, _BLOCK_M, 64)
            self.tma_intermediate = create_tma_descriptor(self.intermediate, _BLOCK_M, 64)
        except Exception as e:
            logging.warning(f"[MoEBufferManager] TMA descriptor creation failed: {e}")
            self.tma_dispatched = None
            self.tma_intermediate = None

    def resize_if_needed(self, global_bsz: int):
        """Resize comm buffers if global_bsz exceeds capacity."""
        if global_bsz <= self.max_global_bsz:
            return
        logging.info(f"[MoEBufferManager] Resizing: {self.max_global_bsz} → {global_bsz}")
        self.max_global_bsz = global_bsz
        self.all_tokens = torch.zeros(global_bsz, self.H, dtype=torch.bfloat16, device=self.device)

    def _total_bytes(self):
        total = 0
        for attr in ['all_tokens', 'padded']:
            t = getattr(self, attr)
            total += t.nelement() * t.element_size()
        return total


# ============================================================================
# RMSNorm
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


# ============================================================================
# MLA Attention (Multi-head Latent Attention)
# ============================================================================

class KimiK25Attention(nn.Module):
    """MLA attention for K2.5.

    Structural definition — actual forward (FlashAttention, RoPE, KV cache)
    is handled by KimiK25AttnWrapper.

    The rotary_emb attribute is assigned externally by KimiK25Model to share
    a single YarnRotaryEmbedding instance across all layers.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads  # 64
        self.q_lora_rank = config.q_lora_rank  # 1536
        self.kv_lora_rank = config.kv_lora_rank  # 512
        self.qk_nope_head_dim = config.qk_nope_head_dim  # 128
        self.qk_rope_head_dim = config.qk_rope_head_dim  # 64
        self.v_head_dim = config.v_head_dim  # 128
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim  # 192
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        # Q projection with low-rank compression
        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
        )

        # KV projection with MQA-style compression
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        # Output projection
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, self.hidden_size, bias=False
        )

        # RoPE — assigned by KimiK25Model (shared across layers)
        self.rotary_emb = None

        # Softmax scales for MLA (materialized and unmaterialized KV)
        self.qkv_materialized_softmax_scale = self.q_head_dim ** -0.5
        self.qkv_unmaterialized_softmax_scale = (self.kv_lora_rank + self.qk_rope_head_dim) ** -0.5
        if config.rope_scaling is not None:
            mscale_all_dim = config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = _yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.qkv_materialized_softmax_scale *= mscale * mscale
                self.qkv_unmaterialized_softmax_scale *= mscale * mscale
        self.softmax_scale = self.qkv_materialized_softmax_scale

    def initialize(self):
        """Pre-compute absorbed projections for decode phase."""
        if getattr(self.config, 'phase', None) == "decode":
            kv_b_proj = self.kv_b_proj.weight.view(
                self.num_heads, -1, self.kv_lora_rank
            )
            self.q_absorb = kv_b_proj[:, : self.qk_nope_head_dim, :]
            self.out_absorb = kv_b_proj[:, self.qk_nope_head_dim :, :]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "KimiK25Attention.forward() is structural. "
            "Use KimiK25AttnWrapper for actual attention computation."
        )


def _yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


# ============================================================================
# Expert MLP
# ============================================================================

class KimiK25Expert(nn.Module):
    """Single expert FFN with SiLU gating.

    Used for both routed experts (INT4 W4A16, weights managed by wrappers)
    and shared experts (BF16, weights loaded directly).
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ============================================================================
# Dense MLP (Layer 0)
# ============================================================================

class DenseMLP(nn.Module):
    """Dense FFN for K2.5 layer 0 (non-MoE).

    Uses larger intermediate_size (18432) than MoE experts (2048).
    """

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


# ============================================================================
# MoE Gate (Router)
# ============================================================================

class MoEGate(nn.Module):
    """MoE router with sigmoid scoring and top-k selection.

    K2.5 specifics:
        - Sigmoid scoring (not softmax)
        - n_group=1, topk_group=1 (no group-based selection)
        - routed_scaling_factor=2.5
        - e_score_correction_bias for noaux_tc routing
    """

    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok  # 8
        self.n_routed_experts = config.n_routed_experts  # 384
        self.routed_scaling_factor = config.routed_scaling_factor  # 2.5
        self.scoring_func = config.scoring_func  # "sigmoid"
        self.topk_method = config.topk_method  # "noaux_tc"
        self.n_group = config.n_group  # 1
        self.topk_group = config.topk_group  # 1
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(
            torch.empty(self.n_routed_experts, config.hidden_size)
        )
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(
                torch.empty(self.n_routed_experts)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @torch.inference_mode()
    def warmup(self):
        pass

    def forward(self, hidden_states: torch.Tensor):
        """Standard routing forward.

        Args:
            hidden_states: [batch, seq, hidden_size]

        Returns:
            topk_idx: [total_tokens, top_k] — selected expert indices
            topk_weight: [total_tokens, top_k] — normalized + scaled weights
        """
        if hidden_states.dim() == 2:
            num_tokens, h = hidden_states.shape
        else:
            bsz, seq_len, h = hidden_states.shape
            num_tokens = bsz * seq_len
        hidden_states = hidden_states.view(-1, h)

        # Early return for zero tokens (can happen on some ranks during decode)
        if num_tokens == 0:
            return (
                torch.empty(0, self.top_k, dtype=torch.long, device=hidden_states.device),
                torch.empty(0, self.top_k, dtype=hidden_states.dtype, device=hidden_states.device),
            )

        logits = F.linear(hidden_states, self.weight, None).float()
        topk_idx, topk_weight = gate_sigmoid_topk_cuda(
            logits, self.e_score_correction_bias.float(),
            k=self.top_k, routed_scaling_factor=self.routed_scaling_factor,
        )
        return topk_idx, topk_weight

    @torch.inference_mode()
    def decoding_forward(self, hidden_states):
        """Decode-optimized routing (CUDA kernel)."""
        if hidden_states.dim() == 2:
            num_tokens, h = hidden_states.shape
        else:
            bsz, seq_len, h = hidden_states.shape
            num_tokens = bsz * seq_len
        hidden_states = hidden_states.view(-1, h)

        logits = F.linear(hidden_states, self.weight, None).float()
        topk_idx, topk_weight = gate_sigmoid_topk_cuda(
            logits, self.e_score_correction_bias.float(),
            k=self.top_k, routed_scaling_factor=self.routed_scaling_factor,
        )
        return topk_idx, topk_weight


# ============================================================================
# MoE Layer (384 Routed + 1 Shared Expert)
# ============================================================================

class KimiK25MoE(nn.Module):
    """MoE layer with 384 routed + 1 shared expert.

    Supports two execution modes:
    - EP mode (decode): AllGather → gate on global tokens → local expert loop → AllReduce
    - Local mode (prefill or single-GPU): gate on local tokens → all-expert loop

    Class variable:
        _buf: KimiK25MoEBufferManager — shared across all MoE layers, set by PSM.
    """

    _buf: Optional['KimiK25MoEBufferManager'] = None

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts  # 384
        self.top_k = config.num_experts_per_tok  # 8
        self.moe_intermediate_size = config.moe_intermediate_size  # 2048
        self.num_experts_per_tok = config.num_experts_per_tok

        # EP metadata
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        self.experts_per_rank = self.num_experts // self.world_size
        self.routed_expert_start_idx = self.rank * self.experts_per_rank
        self.routed_expert_end_idx = (self.rank + 1) * self.experts_per_rank

        # Set by PSM after model creation
        self.comm = None
        self.device = None
        self.num_tokens_per_rank = None

        # Persistent vs non-persistent expert tracking (set by PSM)
        # Default: all local experts are persistent (no offloading)
        all_local = list(range(self.routed_expert_start_idx, self.routed_expert_end_idx))
        self.persistent_expert_ids = all_local  # global IDs, GPU-resident
        self.nonpersistent_expert_ids = []      # global IDs, host-offloaded
        self.num_persistent_local_experts = self.experts_per_rank

        # Router
        self.gate = MoEGate(config)

        # Routed experts — EP mode creates None placeholders for non-local experts
        ep_size = getattr(config, 'ep_size', 1)
        if ep_size > 1 and dist.is_initialized():
            self.experts = nn.ModuleList([
                KimiK25Expert(self.hidden_size, self.moe_intermediate_size)
                if self.routed_expert_start_idx <= i < self.routed_expert_end_idx else None
                for i in range(self.num_experts)
            ])
        else:
            self.experts = nn.ModuleList([
                KimiK25Expert(self.hidden_size, self.moe_intermediate_size)
                for _ in range(self.num_experts)
            ])

        # Shared expert (BF16, always active)
        n_shared = getattr(config, 'n_shared_experts', 1)
        self.shared_experts = KimiK25Expert(
            self.hidden_size,
            self.moe_intermediate_size * n_shared,
        )

    def init_num_tokens(self, num_tokens_per_rank):
        """Initialize num_tokens_per_rank (called by PSM during decode setup)."""
        self.num_tokens_per_rank = num_tokens_per_rank

    def set_num_tokens_per_rank(self, num_tokens_per_rank: int):
        """Update num_tokens_per_rank dynamically."""
        self.num_tokens_per_rank = num_tokens_per_rank
        buf = self.__class__._buf
        if buf is not None and buf.padded.shape[0] != num_tokens_per_rank:
            buf.padded = torch.zeros(
                num_tokens_per_rank, buf.H,
                dtype=torch.bfloat16, device=buf.device,
            )
            buf.num_tokens_per_rank = num_tokens_per_rank

    @torch.inference_mode()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """MoE forward — routes to decode or prefill path."""
        if self.config.phase == "decode":
            return self._forward_decode(hidden_states)
        else:
            return self._forward_prefill(hidden_states)

    def init_grouped_wgmma(self):
        """Setup weight pointer arrays for grouped WGMMA kernels.

        Called by PSM after expert wrapping + INT4 weights moved to GPU.
        Only sets up pointers for persistent (GPU-resident) experts.
        """
        if self.num_persistent_local_experts == 0:
            self._use_grouped_wgmma = False
            return

        try:
            from batchgen.moe.grouped_int4_wgmma import setup_expert_weight_pointers
            self._wgmma_weight_ptrs = setup_expert_weight_pointers(
                self.experts, self.num_persistent_local_experts,
                self.routed_expert_start_idx, self.device,
            )
            self._use_grouped_wgmma = True
        except Exception as e:
            logging.warning(f"[MoE] Grouped WGMMA init failed, using loop fallback: {e}")
            self._use_grouped_wgmma = False

    @torch.inference_mode()
    def _forward_decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """EP decode: AllGather → CUDA gate → CUDA dispatch → grouped WGMMA → CUDA reduce → AllReduce."""
        from batchgen.moe.routing import dispatch_count_gather_cuda, reduce_weighted_scatter_cuda

        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        identity = hidden_states
        num_tokens, hidden_size = hidden_states.shape
        device = self.device or hidden_states.device
        K = self.top_k

        # 1) AllGather: collect tokens from all ranks
        buf = self.__class__._buf
        num_global = self.world_size * self.num_tokens_per_rank
        if buf is not None:
            buf.resize_if_needed(num_global)
            all_tokens = buf.all_tokens[:num_global]
            all_tokens.zero_()
            padded = buf.padded
            padded.zero_()
        else:
            all_tokens = torch.zeros(num_global, hidden_size, device=device, dtype=torch.bfloat16)
            padded = torch.zeros(self.num_tokens_per_rank, hidden_size, device=device, dtype=hidden_states.dtype)
        if num_tokens > 0:
            padded[:num_tokens] = hidden_states

        with self.comm.change_state(enable=True):
            self.comm.all_gather(
                all_tokens, padded,
                stream=torch.cuda.default_stream(device),
            )

        # 2) CUDA gate on global tokens
        global_x = all_tokens
        num_global = global_x.shape[0]
        topk_idx, topk_weight = self.gate(global_x.view(num_global, 1, hidden_size))

        # 3) CUDA dispatch
        dispatched_x, expert_counts, expert_offsets, topk_pos = dispatch_count_gather_cuda(
            global_x, topk_idx.to(torch.int32),
            self.routed_expert_start_idx, self.experts_per_rank,
        )

        # 4) Expert execution
        n_persistent = self.num_persistent_local_experts
        all_persistent = (n_persistent == self.experts_per_rank)

        if all_persistent and getattr(self, '_use_grouped_wgmma', False):
            from batchgen.moe.grouped_int4_wgmma import grouped_int4_moe_forward
            expert_output = grouped_int4_moe_forward(
                dispatched_x, expert_offsets,
                self._wgmma_weight_ptrs,
                self.moe_intermediate_size, hidden_size,
            )
        else:
            # Mixed path: grouped WGMMA for persistent + loop for non-persistent
            offsets_cpu = expert_offsets.tolist()
            actual = offsets_cpu[-1]
            if actual == 0:
                global_results = torch.zeros(num_global, hidden_size, device=device, dtype=torch.bfloat16)
                with self.comm.change_state(enable=True):
                    self.comm.all_reduce(
                        global_results, op=dist.ReduceOp.SUM,
                        stream=torch.cuda.default_stream(device),
                    )
                start = self.rank * self.num_tokens_per_rank
                out = global_results[start:start + num_tokens]
                out = out + self.shared_experts(identity)
                return out.view(*orig_shape)

            expert_output = dispatched_x.new_empty(actual, hidden_size)

            if getattr(self, '_use_grouped_wgmma', False) and n_persistent > 0:
                from batchgen.moe.grouped_int4_wgmma import grouped_int4_moe_forward
                persistent_end = offsets_cpu[n_persistent]
                if persistent_end > 0:
                    wgmma_offsets = expert_offsets[:n_persistent + 1]
                    wgmma_out = grouped_int4_moe_forward(
                        dispatched_x[:persistent_end], wgmma_offsets,
                        self._wgmma_weight_ptrs,
                        self.moe_intermediate_size, hidden_size,
                    )
                    expert_output[:persistent_end] = wgmma_out[:persistent_end]

            for local_e in range(n_persistent, self.experts_per_rank):
                start_off = offsets_cpu[local_e]
                end_off = offsets_cpu[local_e + 1]
                if start_off == end_off:
                    continue
                global_e = self.routed_expert_start_idx + local_e
                expert_output[start_off:end_off] = self.experts[global_e](dispatched_x[start_off:end_off])

        # 5) CUDA reduce
        global_results = reduce_weighted_scatter_cuda(
            expert_output, topk_pos, topk_weight,
            num_global, hidden_size, K,
        )

        # 6) AllReduce: sum results from all ranks
        with self.comm.change_state(enable=True):
            self.comm.all_reduce(
                global_results, op=dist.ReduceOp.SUM,
                stream=torch.cuda.default_stream(device),
            )

        # 7) Extract local results + add shared expert
        start = self.rank * self.num_tokens_per_rank
        end = start + num_tokens
        out = global_results[start:end]
        out = out + self.shared_experts(identity)
        return out.view(*orig_shape)

    @torch.inference_mode()
    def _forward_prefill(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Prefill: CUDA gate → per-expert wrapper loop (all experts local, no EP)."""
        identity = hidden_states
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        num_tokens, hidden_size = hidden_states.shape
        device = hidden_states.device
        K = self.top_k

        topk_idx, topk_weight = self.gate(identity)

        flat_expert_idx = topk_idx.view(-1)
        token_indices = torch.arange(num_tokens, device=device).repeat_interleave(K)
        topk_positions = torch.arange(K, device=device).repeat(num_tokens)

        results = torch.zeros(num_tokens, hidden_size, device=device, dtype=torch.float32)

        for expert_idx, expert in enumerate(self.experts):
            if expert is None:
                continue
            mask = flat_expert_idx == expert_idx
            if not mask.any():
                continue
            expert_token_idx = token_indices[mask]
            expert_topk_pos = topk_positions[mask]
            tokens_for_expert = hidden_states[expert_token_idx]
            expert_output = expert(tokens_for_expert)
            expert_weights = topk_weight[expert_token_idx, expert_topk_pos]
            weighted_output = expert_output.float() * expert_weights.unsqueeze(-1)
            results.index_add_(0, expert_token_idx, weighted_output)

        results = results.to(hidden_states.dtype)
        results = results + self.shared_experts(identity.view(-1, hidden_size))
        return results.view(*orig_shape)


# ============================================================================
# Decoder Layer
# ============================================================================

class KimiK25DecoderLayer(nn.Module):
    """Single K2.5 transformer layer with pre-norm architecture.

    Layer 0: dense MLP. Layers 1-60: MoE with 384 routed + 1 shared expert.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = KimiK25Attention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if layer_idx < config.first_k_dense_replace:
            self.mlp = DenseMLP(config)
        else:
            self.mlp = KimiK25MoE(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        # Pre-norm attention + residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(hidden_states=hidden_states)
        hidden_states = attn_out[0] if isinstance(attn_out, tuple) else attn_out
        hidden_states = residual + hidden_states

        # Pre-norm MoE/FFN + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states, None, None)


# ============================================================================
# Inner Model
# ============================================================================

class KimiK25Model(nn.Module):
    """Kimi K2.5 transformer model (inner, no lm_head).

    Contains embed_tokens, layers, norm. A single shared YarnRotaryEmbedding
    instance is created and assigned to all attention layers to avoid
    duplicated cos/sin caches on GPU (~2.4 GiB savings).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Shared RoPE — single instance for all 61 attention layers
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
            KimiK25DecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])

        # Assign shared RoPE to all attention layers
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
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Must provide either input_ids or inputs_embeds")
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        for layer in self.layers:
            layer_output = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output

        hidden_states = self.norm(hidden_states)
        return hidden_states


# ============================================================================
# Outer Wrapper (worker-compatible)
# ============================================================================

class KimiK25ForCausalLM(nn.Module):
    """Kimi K2.5 model with language modeling head.

    Provides .model and .lm_head attributes expected by batchgen_worker.py:
        - self.model.layers[i]  (via KimiK25Model)
        - self.lm_head          (output projection)
    """

    def __init__(self, config, comm=None):
        super().__init__()
        self.config = config
        self.model = KimiK25Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
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
        logits = self.lm_head(hidden_states)
        return _CausalLMOutput(logits=logits)

    def eval(self):
        return super().eval()
