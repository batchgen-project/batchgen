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
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple

from dataclasses import dataclass
from batchgen.layers.rotary_embedding import YarnRotaryEmbedding
from batchgen.moe.routing import gate_sigmoid_topk_cuda
from batchgen.batch_invariant_matmul import matmul_persistent as _bi_matmul
_HAS_BATCH_INVARIANT = os.environ.get("BATCHGEN_BATCH_INVARIANT", "0") == "1"
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
# Per-Layer Sub-Op Timer (BATCHGEN_K25_TIMING=1 to enable)
# ============================================================================

_K25_TIMING_ENABLED = os.environ.get("BATCHGEN_K25_TIMING", "0") == "1"
_K25_TIMING_LOG_INTERVAL = int(os.environ.get("BATCHGEN_K25_TIMING_INTERVAL", "10"))
_K25_TIMING_SYNC_EVERY = int(os.environ.get("BATCHGEN_K25_TIMING_SYNC_EVERY", "50"))

# MoE sub-op names (order matters for table output)
_MOE_OPS = [
    "allgather", "gate", "dispatch", "wgmma_s1", "wgmma_s2",
    "reduce", "allreduce", "shared_expert", "extract_combine",
]
# Layer-level op names
_LAYER_OPS = ["input_ln", "mla", "post_ln_res", "mlp", "res_add"]


class K25DecodeTimer:
    """CUDA event-based per-layer sub-op timer for K2.5 decode.

    Enabled by BATCHGEN_K25_TIMING=1. Accumulates timings across steps,
    logs averages every BATCHGEN_K25_TIMING_INTERVAL steps (default 10).

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

        Only sync every _K25_TIMING_SYNC_EVERY steps to avoid serializing
        the GPU pipeline. On non-sync steps, discard pending events.
        """
        self.step_count += 1

        if self.step_count % _K25_TIMING_SYNC_EVERY != 0:
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
        if self.samples >= _K25_TIMING_LOG_INTERVAL:
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
            f"\n[K25 Timing] avg over {n} steps (step {self.step_count - n + 1}-{self.step_count}):",
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
            "=== MoE sub-op breakdown (us, layers 1-60 only) ===",
            header_moe,
            sep_moe,
        ])
        for l in range(1, self.num_layers):
            vals = [self.moe_accum[op][l] / n for op in _MOE_OPS]
            total = sum(vals)
            row = f"  {l:3d} | " + " | ".join(f"{v:14.1f}" for v in vals) + f" | {total:9.1f}"
            lines.append(row)

        # MoE averages
        lines.append(sep_moe)
        avg_moe = {op: sum(self.moe_accum[op][1:]) / (n * 60) for op in _MOE_OPS}
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
_k25_timer: Optional[K25DecodeTimer] = None


def _get_k25_timer(num_layers: int = 61) -> Optional[K25DecodeTimer]:
    global _k25_timer
    if not _K25_TIMING_ENABLED:
        return None
    if _k25_timer is None:
        _k25_timer = K25DecodeTimer(num_layers=num_layers)
        logging.info(f"[K25 Timing] Enabled: {num_layers} layers, log every {_K25_TIMING_LOG_INTERVAL} steps")
    return _k25_timer


# ============================================================================
# MoE Buffer Manager (shared across all MoE layers)
# ============================================================================

_BLOCK_M = 64  # TMA constraint: global tensor M >= BLOCK_M
_DEFAULT_MTP = 4096  # Default max_tokens_padded (stride per expert in 3D buffer)


def round_moe_buffer_tokens(num_tokens: int) -> int:
    """Round MoE 3D-stride capacity to the WGMMA/TMA tile requirement."""
    if num_tokens <= 0:
        return _BLOCK_M
    return max(_BLOCK_M, ((num_tokens + _BLOCK_M - 1) // _BLOCK_M) * _BLOCK_M)


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
        self.max_tokens_padded = round_moe_buffer_tokens(max_tokens_padded)

        NK = max_global_bsz * topk
        buf_rows = E_local * self.max_tokens_padded  # 3D strided: E * mtp

        # Communication buffers
        self.all_tokens = torch.zeros(max_global_bsz, H, dtype=torch.bfloat16, device=device)
        self.padded = torch.zeros(num_tokens_per_rank, H, dtype=torch.bfloat16, device=device)

        # Routing metadata
        self.expert_counts = torch.zeros(E_local, dtype=torch.int32, device=device)
        self.expert_counters = torch.zeros(E_local, dtype=torch.int32, device=device)
        self.topk_pos = torch.full((NK,), -1, dtype=torch.int32, device=device)

        # Reserved GEMM buffers (3D strided: [E_local * mtp, dim])
        self.dispatched_x = torch.zeros(buf_rows, H, dtype=torch.bfloat16, device=device)
        self.intermediate = torch.zeros(buf_rows, N_inter, dtype=torch.bfloat16, device=device)
        self.expert_out = torch.zeros(buf_rows, H, dtype=torch.bfloat16, device=device)

        # Result buffer
        self.result_buffer = torch.empty(max_global_bsz, H, dtype=torch.bfloat16, device=device)

        # Empty bias (reusable)
        self.empty_bias = torch.empty(0, dtype=torch.int64, device=device)

        # Cached TMA descriptors (single shared TMA on full [E*mtp, dim] buffer)
        self.tma_dispatched = None
        self.tma_intermediate = None
        self._init_tma_descriptors()

        logging.debug(
            f"[MoEBufferManager] 3D strided layout: E_local={E_local}, mtp={self.max_tokens_padded}, "
            f"buf_rows={buf_rows}, H={H}, N_inter={N_inter}, "
            f"total={self._total_bytes() / (1024**3):.2f} GiB"
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
        """Resize communication/routing buffers if global_bsz exceeds capacity."""
        if global_bsz <= self.max_global_bsz:
            return

        logging.info(f"[MoEBufferManager] Resizing comm buffers: {self.max_global_bsz} → {global_bsz}")
        self.max_global_bsz = global_bsz
        NK = global_bsz * self.topk
        device = self.device

        self.all_tokens = torch.zeros(global_bsz, self.H, dtype=torch.bfloat16, device=device)
        self.topk_pos = torch.full((NK,), -1, dtype=torch.int32, device=device)
        self.result_buffer = torch.empty(global_bsz, self.H, dtype=torch.bfloat16, device=device)

        # Resize 3D buffers only if needed
        if global_bsz > self.max_tokens_padded:
            new_mtp = round_moe_buffer_tokens(global_bsz)
            logging.info(f"[MoEBufferManager] Resizing 3D buffers: mtp {self.max_tokens_padded} → {new_mtp}")
            self.max_tokens_padded = new_mtp
            buf_rows = self.E_local * new_mtp
            self.dispatched_x = torch.zeros(buf_rows, self.H, dtype=torch.bfloat16, device=device)
            self.intermediate = torch.zeros(buf_rows, self.N_inter, dtype=torch.bfloat16, device=device)
            self.expert_out = torch.zeros(buf_rows, self.H, dtype=torch.bfloat16, device=device)
            self._init_tma_descriptors()

    def _total_bytes(self):
        total = 0
        for attr in ['all_tokens', 'padded', 'expert_counts', 'expert_counters',
                      'topk_pos', 'dispatched_x', 'intermediate', 'expert_out',
                      'result_buffer']:
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

        if _HAS_BATCH_INVARIANT:
            logits = _bi_matmul(hidden_states, self.weight.t().contiguous()).float()
        else:
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

        if _HAS_BATCH_INVARIANT:
            logits = _bi_matmul(hidden_states, self.weight.t().contiguous()).float()
        else:
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
    _rank_token_counts: Optional[torch.Tensor] = None  # [world_size] per-rank token counts for padding masking

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
        """Setup WGMMA module and weight pointer arrays for inplace 3D strided kernels.

        Called by PSM after expert wrapping + INT4 weights moved to GPU.
        Builds pointer arrays for gate/up/down weights and scales.
        """
        if self.num_persistent_local_experts == 0:
            self._use_grouped_wgmma = False
            return

        try:
            E = self.num_persistent_local_experts
            device = self.device
            weights = {}

            for prefix in ('gate', 'up', 'down'):
                w_ptrs = torch.empty(E, dtype=torch.int64, device=device)
                s_ptrs = torch.empty(E, dtype=torch.int64, device=device)

                for local_e in range(E):
                    global_e = self.routed_expert_start_idx + local_e
                    wrapper = self.experts[global_e]
                    module = wrapper.module if hasattr(wrapper, 'module') else wrapper
                    w_packed = getattr(module, f'int4_{prefix}_packed')
                    w_scale = getattr(module, f'int4_{prefix}_scale')

                    w_ptrs[local_e] = w_packed.data_ptr()
                    s_ptrs[local_e] = w_scale.data_ptr()

                weights[f'_ptr_{prefix}'] = w_ptrs
                weights[f'_ptr_{prefix}_scale'] = s_ptrs

            self.__class__._dtype_logged = True

            self._moe_weights = weights
            self._wgmma_mod = _load_int4_grouped_module()
            self._use_grouped_wgmma = True

            # Marlin W4A16 3-stage decode: weight pointers only (mtp-dependent buffers deferred)
            self._use_marlin_decode = False
            self._marlin_mtp = 0
            # Marlin decode is default for K2.5
            try:
                N = self.moe_intermediate_size
                K = self.config.hidden_size

                # Collect weight pointers for all 3 projections
                gate_w_ptrs = torch.empty(E, dtype=torch.int64, device=device)
                gate_s_ptrs = torch.empty(E, dtype=torch.int64, device=device)
                up_w_ptrs = torch.empty(E, dtype=torch.int64, device=device)
                up_s_ptrs = torch.empty(E, dtype=torch.int64, device=device)
                down_w_ptrs = torch.empty(E, dtype=torch.int64, device=device)
                down_s_ptrs = torch.empty(E, dtype=torch.int64, device=device)

                for local_e in range(E):
                    global_e = self.routed_expert_start_idx + local_e
                    wrapper = self.experts[global_e]
                    module = wrapper.module if hasattr(wrapper, 'module') else wrapper
                    gate_w_ptrs[local_e] = module.marlin_gate_qw.data_ptr()
                    gate_s_ptrs[local_e] = module.marlin_gate_scale.data_ptr()
                    up_w_ptrs[local_e] = module.marlin_up_qw.data_ptr()
                    up_s_ptrs[local_e] = module.marlin_up_scale.data_ptr()
                    down_w_ptrs[local_e] = module.marlin_down_qw.data_ptr()
                    down_s_ptrs[local_e] = module.marlin_down_scale.data_ptr()

                mw = {}
                # S1 fused: separate gate/up [E] each
                mw['gate_B_ptrs'] = gate_w_ptrs
                mw['gate_scales_ptrs'] = gate_s_ptrs
                mw['up_B_ptrs'] = up_w_ptrs
                mw['up_scales_ptrs'] = up_s_ptrs
                # S3: down [E]
                mw['s3_B_ptrs'] = down_w_ptrs
                mw['s3_scales_ptrs'] = down_s_ptrs
                mw['N'] = N
                mw['K'] = K

                self._marlin_weights = mw
                self._use_marlin_decode = True
            except AttributeError as e:
                logging.warning(f"[MoE] Marlin weights not found: {e}")
                self._use_marlin_decode = False
        except Exception as e:
            logging.warning(f"[MoE] Grouped WGMMA init failed, using loop fallback: {e}")
            self._use_grouped_wgmma = False

    # Class-level shared Marlin buffers (like _buf, allocated once for all layers)
    _marlin_gate_buf = None
    _marlin_up_buf = None

    def _init_marlin_buffers(self, mtp: int):
        """Init Marlin 3-stage decode buffers.

        gate_buf and up_buf are class-level (shared across all 60 MoE layers).
        Called once during model init, before KV cache sizing.
        Per-instance: expert_starts, C_ptrs, workspaces.
        """
        mw = self._marlin_weights
        E = self.num_persistent_local_experts
        N = mw['N']  # 2048
        K = mw['K']  # 7168
        device = self.device

        # Expert starts: mtp stride for 3D buffer access
        mw['expert_starts'] = torch.arange(E, dtype=torch.int32, device=device) * mtp

        # S1 fused C_ptrs [E]: output into buf.intermediate at mtp stride
        buf = self.__class__._buf
        bpr_N = N * 2  # bytes per row (BF16)
        mw['s1_fused_C_ptrs'] = torch.tensor(
            [buf.intermediate.data_ptr() + e * mtp * bpr_N for e in range(E)],
            dtype=torch.int64, device=device)

        # S1 workspace (E matrices for fused kernel)
        n_tiles_s1 = N // 256  # 8
        mw['s1_workspace'] = torch.zeros(
            E * (n_tiles_s1 + 17), dtype=torch.int32, device=device)

        # S3 workspace
        n_tiles_s3 = K // 256  # 28
        mw['s3_workspace'] = torch.zeros(
            E * (n_tiles_s3 + 17), dtype=torch.int32, device=device)

        # S3 C_ptrs: pre-computed, recomputed only if expert_out moves
        mw['_s3_expert_out_ptr'] = None
        mw['s3_C_ptrs'] = None

        # Cache Marlin module to avoid per-step import + function call
        from batchgen.moe.marlin_grouped_moe import _load_module
        self._marlin_mod = _load_module()

        self._marlin_mtp = mtp
        if self.rank == 0:
            logging.info(f"[MoE] Marlin 3-stage fused buffers initialized "
                         f"(E={E}, mtp={mtp})")

    def _compact_after_allgather(self, all_tokens, num_tokens, num_global, H, device):
        """Strip padding from AllGathered buffer so all GEMMs see only real tokens.

        Returns (compact_tokens, total_real, my_offset).
        Adds ~2s/128 iters overhead from CPU-GPU sync + Python copy loop.
        Enable with BATCHGEN_MOE_COMPACT=1.
        """
        _local_cnt = torch.tensor([num_tokens], dtype=torch.int32, device=device)
        _all_cnts = torch.zeros(self.world_size, dtype=torch.int32, device=device)
        with self.comm.change_state(enable=True):
            self.comm.all_gather(
                _all_cnts, _local_cnt,
                stream=torch.cuda.default_stream(device),
            )
        _counts_cpu = _all_cnts.tolist()
        total_real = sum(_counts_cpu)

        ntp = self.num_tokens_per_rank
        if total_real < num_global and total_real > 0:
            compact = torch.empty(total_real, H, device=device, dtype=all_tokens.dtype)
            dst = 0
            for r in range(self.world_size):
                c = _counts_cpu[r]
                if c > 0:
                    compact[dst:dst + c] = all_tokens[r * ntp : r * ntp + c]
                    dst += c
            my_offset = sum(_counts_cpu[:self.rank])
            return compact, total_real, my_offset
        else:
            t = total_real if total_real > 0 else num_global
            return all_tokens[:t], t, self.rank * ntp

    @torch.inference_mode()
    def _forward_decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """EP decode: AllGather → CUDA gate → 3D dispatch → inplace WGMMA → reduce → AllReduce."""
        timer = _get_k25_timer()
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

        total_real = num_global
        _my_offset = self.rank * self.num_tokens_per_rank

        if timer is not None:
            ev[1].record()  # after allgather, before gate

        # 2) CUDA gate: sigmoid + top-k + normalize + scale
        topk_idx, topk_weight = self.gate(all_tokens.view(num_global, 1, H))

        # Mask padding tokens to prevent them from inflating expert_counts.
        # rank_token_counts[r] = real token count for rank r (set by worker via PSM).
        # Padding at positions [r*ntp + count_r, (r+1)*ntp) for each rank r.
        # Setting topk_idx=-1 makes dispatch skip them (existing local_expert<0 guard).
        rank_counts = self.__class__._rank_token_counts
        if rank_counts is not None:
            ntp = self.num_tokens_per_rank
            positions = torch.arange(num_global, device=device)
            rank_ids = positions // ntp
            local_pos = positions % ntp
            max_valid = rank_counts[rank_ids]
            padding_mask = local_pos >= max_valid
            if padding_mask.any():
                topk_idx[padding_mask] = -1
                topk_weight[padding_mask] = 0.0

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
                buf.topk_pos[:total_real * topk],
            )
        else:
            mtp = _DEFAULT_MTP
            E = self.experts_per_rank
            dispatched_x = torch.zeros(E * mtp, H, dtype=torch.bfloat16, device=device)
            expert_counts = torch.zeros(E, dtype=torch.int32, device=device)
            expert_counters = torch.zeros(E, dtype=torch.int32, device=device)
            topk_pos = torch.full((total_real * topk,), -1, dtype=torch.int32, device=device)
            expert_counts, topk_pos = dispatch_scatter_3d(
                all_tokens, topk_idx.to(torch.int32),
                dispatched_x,
                self.routed_expert_start_idx, E, mtp,
                expert_counts, expert_counters, topk_pos,
            )

        if timer is not None:
            ev[3].record()  # after dispatch, before wgmma_s1

        # 4) Expert compute: grouped INT4 WGMMA inplace on 3D strided buffers
        mtp = buf.max_tokens_padded if buf is not None else 0

        if getattr(self, '_use_marlin_decode', False) and buf is not None:
            # === DECODE: Marlin 3-Stage (weight-format-driven, no runtime dispatch) ===
            # Marlin weights are loaded at init time. Kernel selection follows weight format.
            if self._marlin_mtp != mtp:
                self._init_marlin_buffers(mtp)

            # Pigeonhole bound for CTA M-tiling grid size
            max_possible_m = min(total_real, mtp)
            max_marlin_m_tiles = (max_possible_m + 15) // 16

            mod_m = self._marlin_mod
            mw = self._marlin_weights
            E_local = buf.E_local

            # S3 C_ptrs: recompute only if expert_out buffer moved
            if mw['_s3_expert_out_ptr'] != buf.expert_out.data_ptr():
                bpr_K = mw['K'] * 2
                mw['s3_C_ptrs'] = torch.tensor(
                    [buf.expert_out.data_ptr() + e * mtp * bpr_K for e in range(E_local)],
                    dtype=torch.int64, device=device)
                mw['_s3_expert_out_ptr'] = buf.expert_out.data_ptr()

            # Stage 1: fused gate+up+SiLU (single kernel, E matrices)
            n_tiles_s1 = N // 256
            mod_m.grouped_marlin_gemm_m16_s1(
                buf.dispatched_x,
                mw['gate_B_ptrs'], mw['up_B_ptrs'], mw['s1_fused_C_ptrs'],
                mw['gate_scales_ptrs'], mw['up_scales_ptrs'],
                mw['expert_starts'], expert_counts,
                E_local, N, K, mw['s1_workspace'], n_tiles_s1, max_marlin_m_tiles)

            if timer is not None:
                ev[4].record()  # after fused S1, before S3

            # Stage 3: down GEMM (E matrices)
            n_tiles_s3 = K // 256
            mod_m.grouped_marlin_gemm_m16(
                buf.intermediate, mw['s3_B_ptrs'], mw['s3_C_ptrs'],
                mw['s3_scales_ptrs'], mw['expert_starts'], expert_counts,
                E_local, K, N, mw['s3_workspace'],
                E_local, n_tiles_s3, max_marlin_m_tiles)

        elif getattr(self, '_use_grouped_wgmma', False) \
                and buf is not None and buf.tma_dispatched is not None:
            # === WGMMA 2-Stage (used when Marlin disabled or prefill mode) ===
            mod = self._wgmma_mod
            w = self._moe_weights
            avg_per_expert = (total_real * topk + buf.E_local - 1) // buf.E_local
            max_m_tiles = (min(avg_per_expert * 2, mtp) + _BLOCK_M - 1) // _BLOCK_M
            max_m_tiles = max(max_m_tiles, 1)

            mod.grouped_int4_moe_stage1_inplace(
                buf.dispatched_x, buf.intermediate, buf.tma_dispatched,
                expert_counts,
                w["_ptr_gate"], w["_ptr_gate_scale"],
                w["_ptr_up"], w["_ptr_up_scale"],
                buf.empty_bias, buf.empty_bias,
                N, K // 2, K // 32, max_m_tiles, mtp,
            )
            if timer is not None:
                ev[4].record()  # after stage1, before stage2
            mod.grouped_int4_moe_stage2_inplace(
                buf.intermediate, buf.expert_out, buf.tma_intermediate,
                expert_counts,
                w["_ptr_down"], w["_ptr_down_scale"],
                buf.empty_bias,
                K, N // 2, N // 32, max_m_tiles, mtp,
            )

        if timer is not None:
            ev[5].record()  # after wgmma_s2, before reduce

        # 5) Reduce: weighted scatter from 3D strided buffer to flat [total_real, H]
        if buf is not None:
            result_buf = buf.result_buffer[:total_real]
            global_results = reduce_weighted_scatter(
                buf.expert_out, topk_pos, topk_weight,
                total_real, H, topk, output=result_buf,
            )
        else:
            global_results = reduce_weighted_scatter(
                dispatched_x, topk_pos, topk_weight,
                total_real, H, topk,
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

        out = global_results[_my_offset:_my_offset + num_tokens] + shared_out

        if timer is not None:
            ev[9].record()  # after extract_combine
            timer.defer_moe_times(self._layer_idx, ev, _MOE_OPS)

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

    _fused_add_rmsnorm_fn = None  # cached CUDA fused add+rmsnorm kernel

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
            self.mlp._layer_idx = layer_idx

    @staticmethod
    def _get_fused_add_rmsnorm_fn():
        if KimiK25DecoderLayer._fused_add_rmsnorm_fn is not None:
            return KimiK25DecoderLayer._fused_add_rmsnorm_fn
        try:
            from batchgen.other_kernels.cuda_rmsnorm import cuda_add_rmsnorm
            KimiK25DecoderLayer._fused_add_rmsnorm_fn = cuda_add_rmsnorm
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

        timer = _get_k25_timer()
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
        timer = _get_k25_timer()
        if timer is not None:
            timer.step_done()

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
        if _HAS_BATCH_INVARIANT and hidden_states.numel() > 0:
            hs_shape = hidden_states.shape
            hs_2d = hidden_states.view(-1, hs_shape[-1])
            M = hs_2d.shape[0]
            if not hasattr(self, '_lm_head_weight_t') or self._lm_head_weight_t is None:
                self._lm_head_weight_t = self.lm_head.weight.t().contiguous()
            _CHUNK = 4096
            if M <= _CHUNK:
                logits = _bi_matmul(hs_2d, self._lm_head_weight_t).float()
            else:
                logits = torch.empty(M, self._lm_head_weight_t.shape[1],
                                     device=hs_2d.device, dtype=torch.float32)
                for start in range(0, M, _CHUNK):
                    end = min(start + _CHUNK, M)
                    logits[start:end] = _bi_matmul(
                        hs_2d[start:end], self._lm_head_weight_t).float()
            logits = logits.view(*hs_shape[:-1], -1)
        else:
            logits = self.lm_head(hidden_states).float()
        return _CausalLMOutput(logits=logits)

    def eval(self):
        return super().eval()
