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

"""DeepSeek-R1 model definition following BatchGen design pattern.

Architecture:
    - 61 transformer layers (first 3 dense, layers 3-60 MoE)
    - MLA attention with 128 heads, q_lora_rank=1536, kv_lora_rank=512
    - 256 routed experts + 1 shared expert per MoE layer, top-8
    - n_group=8, topk_group=4 grouped routing
    - FP8 e4m3 blockwise quantization (CuTe persistent kernels)
    - Hidden size 7168, moe_intermediate_size 2048, dense intermediate 18432
    - Shared YaRN RoPE (theta=10000, factor=40)
    - RMSNorm (eps=1e-6)

Design:
    - DeepSeekR1ForCausalLM (outer): .model + .lm_head (worker-compatible)
    - DeepSeekR1Model (inner): .embed_tokens, .layers, .norm
    - MoE uses MiniMax's FP8 blockwise CuTe kernels (not INT4 WGMMA/Marlin)
    - Grouped routing (n_group=8, topk_group=4) — different from Kimi's n_group=1
"""

import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, List
from dataclasses import dataclass

from batchgen.layers.rotary_embedding import YarnRotaryEmbedding
from batchgen.moe.dispatch_scatter_3d import dispatch_scatter_3d, reduce_weighted_scatter

# FP8 blockwise kernels (from MiniMax pattern)
try:
    from batchgen_kernels.moe._C_fp8_blockwise_ops import act_quant_3d
    _HAS_FP8_OPS = True
except ImportError:
    _HAS_FP8_OPS = False

try:
    from batchgen.moe.grouped_fp8_blockwise_moe import (
        grouped_fp8_blockwise_fused_s1,
        grouped_fp8_blockwise_s3,
    )
    _HAS_FP8_GEMM = True
except ImportError:
    _HAS_FP8_GEMM = False


logger = logging.getLogger(__name__)


# ============================================================================
# Minimal output container
# ============================================================================

@dataclass
class _CausalLMOutput:
    """Minimal output container with .logits attribute for worker compatibility."""
    logits: torch.Tensor


# ============================================================================
# Per-Layer Sub-Op Timer (BATCHGEN_R1_TIMING=1 to enable)
# ============================================================================

_R1_TIMING_ENABLED = os.environ.get("BATCHGEN_R1_TIMING", "0") == "1"
_R1_TIMING_LOG_INTERVAL = int(os.environ.get("BATCHGEN_R1_TIMING_INTERVAL", "10"))
_R1_TIMING_SYNC_EVERY = int(os.environ.get("BATCHGEN_R1_TIMING_SYNC_EVERY", "50"))

# MoE sub-op names (order matters for table output)
_MOE_OPS = [
    "allgather", "gate", "dispatch", "fp8_s1", "fp8_s3",
    "reduce", "allreduce", "shared_expert", "extract_combine",
]
# Layer-level op names
_LAYER_OPS = ["input_ln", "mla", "post_ln_res", "mlp", "res_add"]


class R1DecodeTimer:
    """CUDA event-based per-layer sub-op timer for R1 decode.

    Enabled by BATCHGEN_R1_TIMING=1. Accumulates timings across steps,
    logs averages every BATCHGEN_R1_TIMING_INTERVAL steps (default 10).

    Events are recorded without sync during forward. A single sync happens
    at step_done() to read all event pairs at once (avoids per-layer sync).
    """

    def __init__(self, num_layers: int = 61, device: str = "cuda"):
        self.num_layers = num_layers
        self.device = device
        self.step_count = 0

        # Per-layer, per-op accumulators (us)
        self.layer_accum = {
            op: [0.0] * num_layers for op in _LAYER_OPS
        }
        self.moe_accum = {
            op: [0.0] * num_layers for op in _MOE_OPS
        }
        self.samples = 0

        # Deferred event pairs: list of (layer_idx, events, op_names, "layer"|"moe")
        self._pending_events = []

    def make_events(self, n: int):
        """Create n+1 CUDA events (n intervals between n+1 markers)."""
        return [torch.cuda.Event(enable_timing=True) for _ in range(n + 1)]

    def defer_layer_times(self, layer_idx: int, events, op_names):
        """Store events for deferred processing (no sync needed)."""
        self._pending_events.append((layer_idx, events, op_names, "layer"))

    def defer_moe_times(self, layer_idx: int, events, op_names):
        """Store events for deferred processing (no sync needed)."""
        self._pending_events.append((layer_idx, events, op_names, "moe"))

    def step_done(self):
        """Called after one full forward pass (all 61 layers).

        Only sync every _R1_TIMING_SYNC_EVERY steps to avoid serializing
        the GPU pipeline. On non-sync steps, discard pending events.
        """
        self.step_count += 1

        if self.step_count % _R1_TIMING_SYNC_EVERY != 0:
            # Non-sync step: discard events to avoid memory buildup
            self._pending_events.clear()
            return

        # Sync step: read all deferred events
        torch.cuda.synchronize()
        for layer_idx, events, op_names, kind in self._pending_events:
            accum = self.layer_accum if kind == "layer" else self.moe_accum
            for i, op in enumerate(op_names):
                us = events[i].elapsed_time(events[i + 1]) * 1000.0
                accum[op][layer_idx] += us
        self._pending_events.clear()

        self.samples += 1
        if self.samples >= _R1_TIMING_LOG_INTERVAL:
            self._log_and_reset()

    def _log_and_reset(self):
        n = self.samples
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != 0:
            self.samples = 0
            for op in _LAYER_OPS:
                self.layer_accum[op] = [0.0] * self.num_layers
            for op in _MOE_OPS:
                self.moe_accum[op] = [0.0] * self.num_layers
            return

        header_layer = "layer | " + " | ".join(f"{op:>10}" for op in _LAYER_OPS) + " | total_us"
        header_moe = "layer | " + " | ".join(f"{op:>14}" for op in _MOE_OPS) + " | moe_total"
        sep_layer = "-" * len(header_layer)
        sep_moe = "-" * len(header_moe)

        lines = [
            f"\n[R1 Timing] avg over {n} steps (step {self.step_count - n + 1}-{self.step_count}):",
            "",
            "=== Layer-level breakdown (us) ===",
            header_layer,
            sep_layer,
        ]
        for l in range(self.num_layers):
            vals = [self.layer_accum[op][l] / n for op in _LAYER_OPS]
            total = sum(vals)
            row = f"  {l:3d} | " + " | ".join(f"{v:10.1f}" for v in vals) + f" | {total:8.1f}"
            lines.append(row)

        # Averages for layer 0 vs 1-60
        lines.append(sep_layer)
        avg0 = {op: self.layer_accum[op][0] / n for op in _LAYER_OPS}
        avg_rest = {op: sum(self.layer_accum[op][1:]) / (n * 60) for op in _LAYER_OPS}
        row0 = "  L0  | " + " | ".join(f"{avg0[op]:10.1f}" for op in _LAYER_OPS) + f" | {sum(avg0.values()):8.1f}"
        row_r = " L1-60| " + " | ".join(f"{avg_rest[op]:10.1f}" for op in _LAYER_OPS) + f" | {sum(avg_rest.values()):8.1f}"
        lines.extend([row0, row_r])

        lines.extend([
            "",
            "=== MoE sub-op breakdown (us, layers 3-60 only) ===",
            header_moe,
            sep_moe,
        ])
        for l in range(3, self.num_layers):
            vals = [self.moe_accum[op][l] / n for op in _MOE_OPS]
            total = sum(vals)
            row = f"  {l:3d} | " + " | ".join(f"{v:14.1f}" for v in vals) + f" | {total:9.1f}"
            lines.append(row)

        # MoE averages
        lines.append(sep_moe)
        avg_moe = {op: sum(self.moe_accum[op][3:]) / (n * 58) for op in _MOE_OPS}
        row_m = " avg  | " + " | ".join(f"{avg_moe[op]:14.1f}" for op in _MOE_OPS) + f" | {sum(avg_moe.values()):9.1f}"
        lines.append(row_m)

        logging.info("\n".join(lines))

        # Reset
        self.samples = 0
        for op in _LAYER_OPS:
            self.layer_accum[op] = [0.0] * self.num_layers
        for op in _MOE_OPS:
            self.moe_accum[op] = [0.0] * self.num_layers


# Global timer instance (lazy init)
_r1_timer: Optional[R1DecodeTimer] = None


def _get_r1_timer(num_layers: int = 61) -> Optional[R1DecodeTimer]:
    global _r1_timer
    if not _R1_TIMING_ENABLED:
        return None
    if _r1_timer is None:
        _r1_timer = R1DecodeTimer(num_layers=num_layers)
        logging.info(f"[R1 Timing] Enabled: {num_layers} layers, log every {_R1_TIMING_LOG_INTERVAL} steps")
    return _r1_timer


# ============================================================================
# MoE Buffer Manager (shared across all MoE layers)
# ============================================================================

_DEFAULT_MTP = 64  # fixed max_tokens_padded for decode batch sizes


class DeepSeekR1MoEBufferManager:
    """Pre-allocated buffers for R1 MoE decode pipeline (3D strided layout).

    One instance per model, shared across all 58 MoE layers via class variable.
    Follows K2.5 pattern: dispatch_scatter_3d writes tokens directly into
    fixed-stride slots, GEMM operates inplace, reduce reads via topk_pos.

    Buffer layout: [E_local * max_tokens_padded, dim] (3D strided).
    Each expert e owns rows [e * mtp, (e+1) * mtp) in the activation buffer.
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

        # Communication buffers
        self.all_tokens = torch.zeros(max_global_bsz, H, dtype=torch.bfloat16, device=device)
        self.padded = torch.zeros(num_tokens_per_rank, H, dtype=torch.bfloat16, device=device)

        # Routing metadata
        self.expert_counts = torch.zeros(E_local, dtype=torch.int32, device=device)
        self.expert_counters = torch.zeros(E_local, dtype=torch.int32, device=device)
        self.topk_pos = torch.full((NK,), -1, dtype=torch.int32, device=device)

        # 3D strided GEMM buffers
        self.dispatched_x = torch.zeros(buf_rows, H, dtype=torch.bfloat16, device=device)
        self.expert_out = torch.zeros(buf_rows, H, dtype=torch.bfloat16, device=device)

        # Result buffer
        self.result_buffer = torch.empty(max_global_bsz, H, dtype=torch.bfloat16, device=device)

        logging.info(
            f"[MoEBufferManager] 3D strided: E_local={E_local}, mtp={max_tokens_padded}, "
            f"buf_rows={buf_rows}, H={H}, N_inter={N_inter}, "
            f"total={self._total_bytes() / (1024**3):.2f} GiB"
        )

    def resize_if_needed(self, global_bsz: int):
        """Resize communication/routing buffers if global_bsz exceeds capacity."""
        if global_bsz <= self.max_global_bsz:
            return
        logging.info(f"[MoEBufferManager] Resizing: {self.max_global_bsz} -> {global_bsz}")
        self.max_global_bsz = global_bsz
        NK = global_bsz * self.topk
        self.all_tokens = torch.zeros(global_bsz, self.H, dtype=torch.bfloat16, device=self.device)
        self.topk_pos = torch.full((NK,), -1, dtype=torch.int32, device=self.device)
        self.result_buffer = torch.empty(global_bsz, self.H, dtype=torch.bfloat16, device=self.device)

    def _total_bytes(self):
        total = 0
        for attr in ['all_tokens', 'padded', 'expert_counts', 'expert_counters',
                      'topk_pos', 'dispatched_x', 'expert_out', 'result_buffer']:
            t = getattr(self, attr)
            total += t.nelement() * t.element_size()
        return total


# ============================================================================
# RMSNorm
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (fused CUDA kernel)."""

    _fused_fn = None  # cached kernel function

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    @staticmethod
    def _get_fused_fn():
        if RMSNorm._fused_fn is not None:
            return RMSNorm._fused_fn
        try:
            from batchgen.other_kernels.cuda_rmsnorm import cuda_rmsnorm
            RMSNorm._fused_fn = cuda_rmsnorm
            return cuda_rmsnorm
        except Exception:
            pass
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
        # Fallback: pure PyTorch
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


# ============================================================================
# MLA Attention (Multi-head Latent Attention)
# ============================================================================

class DeepSeekR1Attention(nn.Module):
    """MLA attention for DeepSeek-R1.

    Structural definition — actual forward (FlashAttention, RoPE, KV cache)
    is handled by DeepSeekR1AttnWrapper.

    The rotary_emb attribute is assigned externally by DeepSeekR1Model to share
    a single YarnRotaryEmbedding instance across all layers.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads  # 128
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

        # RoPE — assigned by DeepSeekR1Model (shared across layers)
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
            "DeepSeekR1Attention.forward() is structural. "
            "Use DeepSeekR1AttnWrapper for actual attention computation."
        )


def _yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


# ============================================================================
# Expert MLP
# ============================================================================

class DeepSeekR1Expert(nn.Module):
    """Single expert FFN with SiLU gating.

    Used for both routed experts (FP8 e4m3, weights managed by wrappers)
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
# Dense MLP (Layers 0-2)
# ============================================================================

class DenseMLP(nn.Module):
    """Dense FFN for R1 layers 0-2 (non-MoE).

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
# MoE Gate (Grouped Routing — n_group=8, topk_group=4)
# ============================================================================

class DeepSeekR1MoEGate(nn.Module):
    """MoE router with sigmoid scoring and grouped top-k selection.

    R1 specifics:
        - Sigmoid scoring (not softmax)
        - n_group=8, topk_group=4 (grouped routing)
        - routed_scaling_factor from config
        - e_score_correction_bias for noaux_tc routing

    Different from Kimi's MoEGate which uses n_group=1, topk_group=1.
    The grouped routing first selects top groups, then selects top experts
    within those groups, following DeepSeek-V3 routing logic.
    """

    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok  # 8
        self.n_routed_experts = config.n_routed_experts  # 256
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func  # "sigmoid"
        self.topk_method = config.topk_method  # "noaux_tc"
        self.n_group = config.n_group  # 8
        self.topk_group = config.topk_group  # 4
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

    @torch.inference_mode()
    def forward(self, hidden_states: torch.Tensor):
        """Grouped routing forward.

        Args:
            hidden_states: [batch, seq, hidden_size] or [total_tokens, hidden_size]

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

        logits = F.linear(hidden_states.float(), self.weight.float(), None)
        scores = logits.sigmoid()

        # Group-based routing (n_group=8, topk_group=4)
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        group_scores = scores_for_choice.view(
            num_tokens, self.n_group, -1
        ).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(
            group_scores, k=self.topk_group, dim=-1, sorted=False
        )[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(
            num_tokens, self.n_group, self.n_routed_experts // self.n_group
        ).reshape(num_tokens, -1)
        tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)

        topk_weight = scores.gather(1, topk_idx)
        denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
        topk_weight = topk_weight / denominator * self.routed_scaling_factor

        return topk_idx, topk_weight

    @torch.inference_mode()
    def decoding_forward(self, hidden_states):
        """Decode-optimized routing (same as forward for grouped routing)."""
        return self.forward(hidden_states)


# ============================================================================
# MoE Layer (256 Routed + 1 Shared Expert, FP8 Blockwise)
# ============================================================================

class DeepSeekR1MoE(nn.Module):
    """MoE layer with 256 routed + 1 shared expert.

    Supports two execution modes:
    - EP mode (decode): AllGather -> gate on global tokens -> local expert loop -> AllReduce
    - Local mode (prefill or single-GPU): gate on local tokens -> all-expert loop

    Uses FP8 blockwise CuTe kernels for expert compute (not INT4 WGMMA/Marlin).
    Grouped routing with n_group=8, topk_group=4.

    Class variable:
        _buf: DeepSeekR1MoEBufferManager — shared across all MoE layers, set by PSM.
    """

    _buf: Optional['DeepSeekR1MoEBufferManager'] = None
    _shared_expert_stream = None

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts  # 256
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

        # Router (grouped routing)
        self.gate = DeepSeekR1MoEGate(config)

        # Routed experts — EP mode creates None placeholders for non-local experts
        ep_size = getattr(config, 'ep_size', 1)
        if ep_size > 1 and dist.is_initialized():
            self.experts = nn.ModuleList([
                DeepSeekR1Expert(self.hidden_size, self.moe_intermediate_size)
                if self.routed_expert_start_idx <= i < self.routed_expert_end_idx else None
                for i in range(self.num_experts)
            ])
        else:
            self.experts = nn.ModuleList([
                DeepSeekR1Expert(self.hidden_size, self.moe_intermediate_size)
                for _ in range(self.num_experts)
            ])

        # Shared expert (BF16, always active)
        n_shared = getattr(config, 'n_shared_experts', 1)
        self.shared_experts = DeepSeekR1Expert(
            self.hidden_size,
            self.moe_intermediate_size * n_shared,
        )

        # FP8 blockwise state
        self._fp8_blockwise_ready = False

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

    def init_fp8_blockwise_weights(self):
        """Stack per-expert FP8 weights into 3D tensors for grouped GEMM.

        Creates [E_local, out_dim, in_dim] contiguous fp8 weight tensors
        and [E_local, out_dim/128, (in_dim/128+3)//4*4] padded scale tensors.
        Called once by PSM after expert wrappers are configured and weights loaded.
        """
        E = self.experts_per_rank
        K = self.hidden_size  # 7168
        N = self.moe_intermediate_size  # 2048
        scale_block = 128

        k_blocks = K // scale_block
        n_blocks = N // scale_block
        k_blocks_pad4 = (k_blocks + 3) // 4 * 4
        n_blocks_pad4 = (n_blocks + 3) // 4 * 4

        gate_ws, up_ws, down_ws = [], [], []
        gate_wss, up_wss, down_wss = [], [], []

        for local_e in range(E):
            global_e = self.routed_expert_start_idx + local_e
            wrapper = self.experts[global_e]
            module = wrapper.module if hasattr(wrapper, 'module') else wrapper
            scale = wrapper.weight_dequant_scale if hasattr(wrapper, 'weight_dequant_scale') else {}

            gate_ws.append(module.gate_proj.weight.data)
            up_ws.append(module.up_proj.weight.data)
            down_ws.append(module.down_proj.weight.data)
            gate_wss.append(scale.get('gate_proj.weight_scale_inv',
                                      scale.get('w1.weight_scale_inv',
                                                 torch.ones(n_blocks, k_blocks, device=self.device))))
            up_wss.append(scale.get('up_proj.weight_scale_inv',
                                    scale.get('w3.weight_scale_inv',
                                              torch.ones(n_blocks, k_blocks, device=self.device))))
            down_wss.append(scale.get('down_proj.weight_scale_inv',
                                      scale.get('w2.weight_scale_inv',
                                                 torch.ones(k_blocks, n_blocks, device=self.device))))

        # Stack weights: [E, N, K] for gate/up, [E, K, N] for down
        self.fp8_gate_w3d = torch.stack(gate_ws).contiguous()
        self.fp8_up_w3d = torch.stack(up_ws).contiguous()
        self.fp8_down_w3d = torch.stack(down_ws).contiguous()

        # Gate weight scales: [E, N/128, (K/128+3)//4*4]
        self.fp8_gate_ws3d = torch.zeros(
            E, n_blocks, k_blocks_pad4,
            dtype=torch.float32, device=self.device)
        for i, s in enumerate(gate_wss):
            self.fp8_gate_ws3d[i, :, :k_blocks] = s

        # Up weight scales: [E, N/128, (K/128+3)//4*4]
        self.fp8_up_ws3d = torch.zeros(
            E, n_blocks, k_blocks_pad4,
            dtype=torch.float32, device=self.device)
        for i, s in enumerate(up_wss):
            self.fp8_up_ws3d[i, :, :k_blocks] = s

        # Down weight scales: [E, K/128, (N/128+3)//4*4]
        self.fp8_down_ws3d = torch.zeros(
            E, k_blocks, n_blocks_pad4,
            dtype=torch.float32, device=self.device)
        for i, s in enumerate(down_wss):
            self.fp8_down_ws3d[i, :, :n_blocks] = s

        self._fp8_blockwise_ready = True

        logging.info(
            f"[MoE] FP8 blockwise weights stacked: "
            f"gate={list(self.fp8_gate_w3d.shape)}, "
            f"down={list(self.fp8_down_w3d.shape)}, "
            f"gate_scale={list(self.fp8_gate_ws3d.shape)}"
        )

    def _fp8_blockwise_gemm_3d(self, buf, expert_counts):
        """FP8 blockwise grouped GEMM on 3D strided buffer (no scatter/gather).

        Reads from buf.dispatched_x, writes to buf.expert_out.
        Uses CuTe persistent kernel with uniform cu_seqlens stride.
        P3a: CUDA act_quant_3d replaces Triton act_quant
        P3b: CUDA fused_silu_quant_3d replaces SiLU + act_quant
        """
        if not getattr(self.__class__, '_warned_gemm_3d', False):
            logging.warning(
                f"[MoE] HOT PATH: _fp8_blockwise_gemm_3d "
                f"(act_quant_3d={_HAS_FP8_OPS}, fused_silu_quant={_HAS_FP8_OPS})")
            self.__class__._warned_gemm_3d = True
        E = self.experts_per_rank
        K = self.hidden_size
        N = self.moe_intermediate_size
        mtp = buf.max_tokens_padded

        cu_seqlens = torch.arange(
            0, (E + 1) * mtp, mtp, dtype=torch.int32, device=buf.dispatched_x.device)

        seqlens = expert_counts[:E]
        # Avoid .item() GPU->CPU sync in hot path — use fixed estimate for TileM hint.
        # For decode, M per expert is small (1-8). Kernel auto-selects TileM regardless.
        avg = max(mtp // max(E, 1), 1)

        # P3a: Quantize input — CUDA act_quant_3d or Triton fallback
        if _HAS_FP8_OPS:
            x_3d = buf.dispatched_x[:E * mtp].view(E, mtp, K)
            x_quant_3d, x_scale_3d = act_quant_3d(x_3d, seqlens)
            x_quant = x_quant_3d.view(E * mtp, K)
            x_scale_t = x_scale_3d.view(E * mtp, -1).t().contiguous()
        else:
            from batchgen.attention.mla.fa3_backend import act_quant
            x_quant, x_scale = act_quant(buf.dispatched_x[:E * mtp])
            x_scale_t = x_scale.t().contiguous()

        # S1: gate + up + SiLU -> FP8 quantize for S3 down projection
        # Fused S1: single kernel for gate+up+SiLU
        if _HAS_FP8_OPS:
            s1_result = grouped_fp8_blockwise_fused_s1(
                x_quant.view(torch.float8_e4m3fn), x_scale_t,
                self.fp8_gate_w3d.view(torch.float8_e4m3fn),
                self.fp8_up_w3d.view(torch.float8_e4m3fn),
                self.fp8_gate_ws3d, self.fp8_up_ws3d,
                seqlens, cu_seqlens, avg,
            )
            inter_quant_3d, inter_scale_3d = act_quant_3d(
                s1_result.view(E, mtp, N), seqlens)
            inter_quant = inter_quant_3d.view(E * mtp, N)
            inter_scale_t = inter_scale_3d.view(E * mtp, -1).t().contiguous()
        else:
            from batchgen.moe.grouped_fp8_blockwise_moe import grouped_fp8_blockwise_s1_silu
            from batchgen.attention.mla.fa3_backend import act_quant
            intermediate = grouped_fp8_blockwise_s1_silu(
                x_quant.view(torch.float8_e4m3fn), x_scale_t,
                self.fp8_gate_w3d.view(torch.float8_e4m3fn),
                self.fp8_up_w3d.view(torch.float8_e4m3fn),
                self.fp8_gate_ws3d, self.fp8_up_ws3d,
                seqlens, cu_seqlens, avg,
            )
            inter_quant, inter_scale = act_quant(intermediate)
            inter_scale_t = inter_scale.t().contiguous()

        # S3: down projection -> writes to expert_out buffer
        result = grouped_fp8_blockwise_s3(
            inter_quant.view(torch.float8_e4m3fn), inter_scale_t,
            self.fp8_down_w3d.view(torch.float8_e4m3fn),
            self.fp8_down_ws3d,
            seqlens, cu_seqlens, avg,
        )

        # Copy result to expert_out buffer for reduce
        buf.expert_out[:E * mtp].copy_(result[:E * mtp])

    @torch.inference_mode()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """MoE forward — routes to decode or prefill path."""
        if self.config.phase == "decode":
            return self._forward_decode(hidden_states)
        else:
            return self._forward_prefill(hidden_states)

    @torch.inference_mode()
    def _forward_decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """EP decode: AllGather -> grouped gate -> 3D dispatch -> FP8 GEMM -> reduce -> AllReduce."""
        timer = _get_r1_timer()
        buf = self.__class__._buf
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        identity = hidden_states
        num_tokens, H = hidden_states.shape
        device = self.device or hidden_states.device
        topk = self.top_k
        N = self.moe_intermediate_size  # 2048
        K = H                           # 7168
        num_global = self.world_size * self.num_tokens_per_rank

        # Resize buffers if needed
        if buf is not None:
            buf.resize_if_needed(num_global)

        # 1) AllGather into reserved buffer
        if buf is not None:
            all_tokens = buf.all_tokens[:num_global]
            padded = buf.padded
            padded.zero_()
        else:
            all_tokens = torch.zeros(num_global, H, device=device, dtype=torch.bfloat16)
            padded = torch.zeros(self.num_tokens_per_rank, H, device=device, dtype=hidden_states.dtype)

        if num_tokens > 0:
            padded[:num_tokens] = hidden_states

        # 1b) Launch shared expert on dedicated stream (overlaps with AllGather + entire MoE pipeline)
        compute_stream = torch.cuda.current_stream(device)
        if not hasattr(self.__class__, '_shared_expert_stream') or self.__class__._shared_expert_stream is None:
            self.__class__._shared_expert_stream = torch.cuda.Stream(device)
        shared_stream = self.__class__._shared_expert_stream

        # Shared expert on its own stream (depends on identity which is ready on compute_stream)
        shared_stream.wait_stream(compute_stream)
        with torch.cuda.stream(shared_stream):
            shared_out = self.shared_experts(identity)

        # --- MoE sub-op timing (9 ops, 10 events on compute stream) ---
        if timer is not None:
            ev = timer.make_events(len(_MOE_OPS))
            ev[0].record()  # before allgather

        # AllGather on compute stream (normal path)
        with self.comm.change_state(enable=True):
            self.comm.all_gather(
                all_tokens, padded,
                stream=torch.cuda.default_stream(device),
            )

        if timer is not None:
            ev[1].record()  # after allgather, before gate

        # 2) Grouped gate: sigmoid + group selection + top-k + normalize + scale
        topk_idx, topk_weight = self.gate(all_tokens.view(num_global, 1, H))

        if timer is not None:
            ev[2].record()  # after gate, before dispatch

        # 3) 3D dispatch scatter into strided buffer
        if buf is not None:
            expert_counts, topk_pos = dispatch_scatter_3d(
                all_tokens, topk_idx.to(torch.int32),
                buf.dispatched_x,
                self.routed_expert_start_idx, self.experts_per_rank,
                buf.max_tokens_padded,
                buf.expert_counts, buf.expert_counters,
                buf.topk_pos[:num_global * topk],
            )
        else:
            mtp = _DEFAULT_MTP
            E = self.experts_per_rank
            dispatched_x = torch.zeros(E * mtp, H, dtype=torch.bfloat16, device=device)
            expert_counts = torch.zeros(E, dtype=torch.int32, device=device)
            expert_counters = torch.zeros(E, dtype=torch.int32, device=device)
            topk_pos = torch.full((num_global * topk,), -1, dtype=torch.int32, device=device)
            expert_counts, topk_pos = dispatch_scatter_3d(
                all_tokens, topk_idx.to(torch.int32),
                dispatched_x,
                self.routed_expert_start_idx, E, mtp,
                expert_counts, expert_counters, topk_pos,
            )

        if timer is not None:
            ev[3].record()  # after dispatch, before fp8_s1

        # 4) Expert compute: FP8 blockwise GEMM on 3D strided buffers
        if buf is not None and self._fp8_blockwise_ready:
            self._fp8_blockwise_gemm_3d(buf, expert_counts)

        if timer is not None:
            ev[4].record()  # after fp8_s1 (includes s1+s3), before fp8_s3 marker
            # Note: fp8_s1 and fp8_s3 are measured as a single interval since
            # _fp8_blockwise_gemm_3d runs both stages. The s3 marker is recorded
            # immediately to maintain event count consistency.

        if timer is not None:
            ev[5].record()  # after fp8_s3, before reduce

        # 5) Reduce: weighted scatter from 3D strided buffer to flat [G, H]
        if buf is not None:
            result_buf = buf.result_buffer[:num_global]
            global_results = reduce_weighted_scatter(
                buf.expert_out, topk_pos, topk_weight,
                num_global, H, topk, output=result_buf,
            )
        else:
            global_results = reduce_weighted_scatter(
                dispatched_x, topk_pos, topk_weight,
                num_global, H, topk,
            )

        if timer is not None:
            ev[6].record()  # after reduce, before allreduce

        # 6) AllReduce
        with self.comm.change_state(enable=True):
            self.comm.all_reduce(
                global_results, op=dist.ReduceOp.SUM,
                stream=torch.cuda.default_stream(device),
            )

        if timer is not None:
            ev[7].record()  # after allreduce, before shared_expert sync

        # 7) Sync shared expert stream and combine
        compute_stream.wait_stream(shared_stream)

        if timer is not None:
            ev[8].record()  # after shared_expert sync, before extract_combine

        start = self.rank * self.num_tokens_per_rank
        end = start + num_tokens
        out = global_results[start:end] + shared_out

        if timer is not None:
            ev[9].record()  # after extract_combine
            timer.defer_moe_times(self._layer_idx, ev, _MOE_OPS)

        return out.view(*orig_shape)

    @torch.inference_mode()
    def _forward_prefill(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Prefill: grouped gate -> per-expert wrapper loop (all experts local, no EP)."""
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

class DeepSeekR1DecoderLayer(nn.Module):
    """Single R1 transformer layer with pre-norm architecture.

    Layers 0-2: dense MLP. Layers 3-60: MoE with 256 routed + 1 shared expert.
    """

    _fused_add_rmsnorm_fn = None  # cached CUDA fused add+rmsnorm kernel

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = DeepSeekR1Attention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if layer_idx < config.first_k_dense_replace:
            self.mlp = DenseMLP(config)
        else:
            self.mlp = DeepSeekR1MoE(config)
            self.mlp._layer_idx = layer_idx

    @staticmethod
    def _get_fused_add_rmsnorm_fn():
        if DeepSeekR1DecoderLayer._fused_add_rmsnorm_fn is not None:
            return DeepSeekR1DecoderLayer._fused_add_rmsnorm_fn
        try:
            from batchgen.other_kernels.cuda_rmsnorm import cuda_add_rmsnorm
            DeepSeekR1DecoderLayer._fused_add_rmsnorm_fn = cuda_add_rmsnorm
            return cuda_add_rmsnorm
        except Exception:
            return None

    def enable_cuda_graph(self, manager, attn_name: str, max_pages_per_seq: int = 0):
        """Enable per-layer CUDA graph for MLA attention.

        MoE stays eager (preserves async shared expert overlap).
        Each rank uses local batch_size for bucket selection (DP-attention, no NCCL).
        """
        self.cuda_graph_manager = manager
        self._attn_segment_name = attn_name
        self._graph_max_pages = max_pages_per_seq

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
        from batchgen.models.wrappers.attention import AttnWrapperBase

        timer = _get_r1_timer()
        layer_idx = kwargs.get("layer_idx", self.layer_idx)
        batch_size = hidden_states.shape[0]
        fused_add_norm = self._get_fused_add_rmsnorm_fn()

        if timer is not None:
            ev = timer.make_events(len(_LAYER_OPS))
            ev[0].record()

        # ---- MLA Attention: CUDA graph or eager ----
        use_graph = (hasattr(self, 'cuda_graph_manager')
                     and self.cuda_graph_manager is not None
                     and getattr(self, '_attn_segment_name', None) is not None
                     and batch_size > 0)
        attn_done = False

        if use_graph:
            try:
                gpu_kv_manager = AttnWrapperBase.gpu_paged_kv_manager
                _, _, page_table = gpu_kv_manager.get_layer_kv_with_page_table(self.layer_idx)
                slot_indices = gpu_kv_manager._gpu_page_table_manager._slot_index_tensor
                cache_seqlens = AttnWrapperBase.cache_seqlens

                pt = page_table[:batch_size]
                if self._graph_max_pages > 0 and pt.shape[1] < self._graph_max_pages:
                    pt = torch.nn.functional.pad(
                        pt, (0, self._graph_max_pages - pt.shape[1]), value=0
                    )
                elif self._graph_max_pages > 0 and pt.shape[1] > self._graph_max_pages:
                    pt = pt[:, :self._graph_max_pages]

                out = self.cuda_graph_manager.replay(
                    self._attn_segment_name,
                    batch_size,
                    hidden_states=hidden_states,
                    cache_seqlens=cache_seqlens[:batch_size],
                    page_table=pt,
                    slot_indices=slot_indices[:batch_size],
                )

                hidden_states = out["normed"]
                residual = out["residual"]

                kv_cb = getattr(AttnWrapperBase, 'kv_append_callback', None)
                if kv_cb is not None:
                    kv_cb(self.layer_idx, out["k_tensor"][:batch_size].clone(), None)

                attn_done = True
            except (ValueError, RuntimeError) as e:
                logging.warning(f"Layer {self.layer_idx} graph replay failed: {e}, falling back to eager")
                attn_done = False

        if timer is not None:
            ev[1].record()  # input_ln (includes graph setup overhead if graph path)

        if not attn_done:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            if timer is not None:
                ev[1].record()  # re-record after actual input_ln

            attn_out = self.self_attn(hidden_states=hidden_states)
            hidden_states = attn_out[0] if isinstance(attn_out, tuple) else attn_out

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

        if timer is not None:
            ev[2].record()  # after mla

        if timer is not None:
            ev[3].record()  # post_ln_res (fused with mla in eager, part of graph in graph path)

        # ---- MoE/FFN: always eager (preserves async shared expert overlap) ----
        hidden_states = self.mlp(hidden_states)

        if timer is not None:
            ev[4].record()  # after mlp

        # residual += mlp_out
        hidden_states = residual + hidden_states

        if timer is not None:
            ev[5].record()  # after res_add
            timer.defer_layer_times(layer_idx, ev, _LAYER_OPS)

        return (hidden_states, None, None)


# ============================================================================
# Inner Model
# ============================================================================

class DeepSeekR1Model(nn.Module):
    """DeepSeek-R1 transformer model (inner, no lm_head).

    Contains embed_tokens, layers, norm. A single shared YarnRotaryEmbedding
    instance is created and assigned to all attention layers to avoid
    duplicated cos/sin caches on GPU.
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
            DeepSeekR1DecoderLayer(config, layer_idx=i)
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

        for layer_idx, layer in enumerate(self.layers):
            layer_output = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                layer_idx=layer_idx,
            )
            hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output

        hidden_states = self.norm(hidden_states)

        # Signal step done to timer
        timer = _get_r1_timer()
        if timer is not None:
            timer.step_done()

        return hidden_states


# ============================================================================
# Outer Wrapper (worker-compatible)
# ============================================================================

class DeepSeekR1ForCausalLM(nn.Module):
    """DeepSeek-R1 model with language modeling head.

    Provides .model and .lm_head attributes expected by batchgen_worker.py:
        - self.model.layers[i]  (via DeepSeekR1Model)
        - self.lm_head          (output projection)
    """

    def __init__(self, config, comm=None):
        super().__init__()
        self.config = config
        self.model = DeepSeekR1Model(config)
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
