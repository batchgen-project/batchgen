# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #

"""Kimi-Linear serving compute functions (injected into model modules by the
ParallelStrategyManager via types.MethodType).

Contents:
  - NoPE-MLA prefill / paged decode (adapted from batchgen.attention.mla
    fa3/flashmla backends with all RoPE calls removed; q_lora_rank=None
    direct q_proj path; optional output gate).
  - KDA prefill / decode using the validated kernels:
      prefill: causal_conv1d_fwd (CUDA, varlen + pooled state write)
               + fla chunk_kda (varlen)
      decode:  causal_conv1d_update (CUDA, pooled) + fla
               fused_recurrent_kda_fwd (pooled, ssm_state_indices,
               inplace_final_state)
  - MoE forward: PREFILL streams every expert per rank (pure DP, no
    collectives), in the hidden space on the 48B and in the 3584 LATENT space
    on K3 (routed_expert_down_proj / norm / up_proj — LatentMoE).
    DECODE (decode_moe_mode="resident_ep", M4 P0.3) uses the
    resident EP-8 shard: all-gather tokens -> local experts via
    fused_moe_bf16 (masked routing) -> all-reduce partial sums
    (batchgen.moe.fused_moe_bf16_resident seam).

KDA state pools are owned by the KDAStateGPUManager (M5.1 unification); the
KDALayerState objects consumed here hold fixed-address VIEWS of the manager's
tensors (see wrappers.py). This module only implements the math.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .wrappers import KimiLinearExpertWrapper


logger = logging.getLogger(__name__)
_KDA_FUSED_DECODE_IMPORT_FAILED = False

# ============================================================================
#  NoPE-MLA
# ============================================================================


def _wait_streamed_sp8_cross_launch(module):
    """Host-enqueue the pending cross-node weight calls before a TP collective.

    Installed on the underlying attention module by the PSM for streamed-SP8
    prefill only, and removed again on release, so decode and every other
    prefill mode see no attribute and pay nothing. The next layer's
    cross-node broadcast gate opened at the end of the previous layer's MoE;
    under ``NCCL_LAUNCH_ORDER_IMPLICIT=1`` this all-reduce is the next TP8
    launch, so every rank must observe the same cross-then-TP8 host order
    here or the implicit ordering serializes them against each other.
    """
    order_wait = getattr(module, "_streamed_sp8_order_wait", None)
    if order_wait is not None:
        order_wait()


def _begin_streamed_sp8_profile(module):
    profiler = getattr(module, "_streamed_sp8_profiler", None)
    if (
        profiler is not None
        and not profiler._prefill_profile_enabled
    ):
        profiler = None
    span = profiler.begin_profile_span() if profiler is not None else None
    return profiler, span


def _end_streamed_sp8_profile(profiler, name, start):
    if profiler is not None:
        profiler.end_profile_span(name, start)


def _reduce_mla_tp_output(module, output):
    """Sum the row-parallel MLA output projection across the TP group."""
    if getattr(module, "attn_tp_size", 1) > 1:
        import torch.distributed as dist

        _wait_streamed_sp8_cross_launch(module)
        profiler, span = _begin_streamed_sp8_profile(module)
        # Preserve the established BF16 all-reduce reduction order.  A direct
        # reduce-scatter changes that order and can amplify across K3's 93
        # layers; streamed prefill retains the local row slice only after the
        # same full reduction used by the resident/reference path.
        dist.all_reduce(output, group=module.attn_tp_group)
        if getattr(module, "_streamed_sp8_output_row_shard", False):
            from .moe_tp_reshard import scatter_rows

            output = scatter_rows(
                output, module.attn_tp_size, module.attn_tp_rank
            ).clone()
        _end_streamed_sp8_profile(profiler, "attention_reduce", span)
    return output


def mla_prefill_nope(self, hidden_states, attention_mask, position_ids):
    """NoPE MLA prefill (sdpa fallback path; FA3 varlen upgrade is P1).

    Args:
        hidden_states: (bsz, seq_len, hidden)
        attention_mask: (bsz, seq_len) 1/0 padding mask
        position_ids: unused (NoPE)

    Returns:
        attn_output: (bsz, seq_len, hidden)
        offload_kv: (bsz, seq_len, kv_lora_rank + qk_rope_head_dim)
    """
    bsz, seq_len, _ = hidden_states.shape

    if self.q_lora_rank is not None:
        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    else:
        q_states = self.q_proj(hidden_states)
    q_states = q_states.view(bsz, seq_len, self.num_heads, self.q_head_dim)

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    kv, k_pe = torch.split(
        compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )
    normed_kv = self.kv_a_layernorm(kv)
    offload_kv = torch.cat([normed_kv, k_pe], dim=-1)

    kv_full = self.kv_b_proj(normed_kv)
    kv_full = kv_full.view(bsz, seq_len, self.num_heads, -1)
    k_nope, value_states = torch.split(
        kv_full, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
    )

    query_states = q_states  # (bsz, seq, H, q_head_dim) — NoPE, no rotation
    key_states = torch.cat(
        [k_nope, k_pe.view(bsz, seq_len, 1, self.qk_rope_head_dim)
                  .expand(-1, -1, self.num_heads, -1)],
        dim=-1,
    )

    # causal + padding mask -> (bsz, 1, seq, seq) additive
    q = query_states.transpose(1, 2)
    k = key_states.transpose(1, 2)
    v = value_states.transpose(1, 2)
    causal = torch.full(
        (seq_len, seq_len), torch.finfo(q.dtype).min, device=q.device
    ).triu(1)
    attn_mask = causal.unsqueeze(0).unsqueeze(0)
    if attention_mask is not None:
        pad = (1.0 - attention_mask[:, None, None, :].to(q.dtype)) * torch.finfo(
            q.dtype
        ).min
        attn_mask = attn_mask + pad
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, scale=self.scaling
    )
    out = out.transpose(1, 2).reshape(bsz, seq_len, -1).contiguous()
    if self.use_output_gate:
        out = out * self.g_proj(hidden_states).sigmoid()
    return _reduce_mla_tp_output(self, self.o_proj(out)), offload_kv


def mla_prefill_nope_prepacked(
    self, hidden_states_2d, position_ids, cu_seqlens, max_seqlen, num_sequences
):
    """NoPE MLA prefill for PREPACKED (varlen) sequences.

    Adapted from batchgen.attention.mla.fa3_backend.mla_prefill_flashattention3_prepacked
    with: (1) no rotary embedding (NoPE), (2) direct q_proj (q_lora_rank=None),
    (3) optional output gate.

    Args:
        hidden_states_2d: (total_tokens, hidden)
        position_ids: (total_tokens,) — unused (NoPE)
        cu_seqlens: (num_sequences + 1,) int32
        max_seqlen: int
        num_sequences: int

    Returns:
        attn_output: (total_tokens, hidden)
        offload_kv: (total_tokens, kv_lora_rank + qk_rope_head_dim)
    """
    # FA3 (flash_attn_interface); FA2 `flash_attn` is not installed on the
    # H20 node. Same kwargs; the tuple return is normalized below.
    from flash_attn_interface import flash_attn_varlen_func

    total_tokens = hidden_states_2d.shape[0]

    if self.q_lora_rank is not None:
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states_2d)))
    else:
        q = self.q_proj(hidden_states_2d)
    q = q.view(total_tokens, self.num_heads, self.q_head_dim)
    q_nope, q_pe = torch.split(
        q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    )

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states_2d)
    compressed_kv, k_pe = torch.split(
        compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )
    normed_kv = self.kv_a_layernorm(compressed_kv)
    offload_kv = torch.cat([normed_kv, k_pe], dim=-1)  # (total, 576)

    kv = self.kv_b_proj(normed_kv)
    kv = kv.view(total_tokens, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
    k_nope, value_states = torch.split(
        kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
    )

    query_states = q.new_empty(total_tokens, self.num_heads, self.q_head_dim)
    query_states[:, :, : self.qk_nope_head_dim] = q_nope
    query_states[:, :, self.qk_nope_head_dim :] = q_pe

    key_states = q.new_empty(total_tokens, self.num_heads, self.q_head_dim)
    k_pe_h = k_pe.view(total_tokens, 1, self.qk_rope_head_dim)
    key_states[:, :, : self.qk_nope_head_dim] = k_nope
    key_states[:, :, self.qk_nope_head_dim :] = k_pe_h  # broadcast over heads

    query_states = query_states.contiguous()
    key_states = key_states.contiguous()
    value_states = value_states.contiguous()

    attn_output = flash_attn_varlen_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=self.scaling,
        causal=True,
    )
    if isinstance(attn_output, tuple):
        attn_output = attn_output[0]

    attn_output = attn_output.reshape(total_tokens, self.num_heads * self.v_head_dim).contiguous()
    if self.use_output_gate:
        attn_output = attn_output * self.g_proj(hidden_states_2d).sigmoid()
    attn_output = self.o_proj(attn_output)
    attn_output = _reduce_mla_tp_output(self, attn_output)
    return attn_output, offload_kv


def mla_decoding_nope_with_pagekv(
    self,
    hidden_states: torch.Tensor,
    q_position_ids: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen: int,
    weight_scale: Optional[dict] = None,
    gpu_paged_kv_manager=None,
    layer_idx: int = 0,
    batch_slice: Optional[tuple] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """NoPE MLA decode over paged KV (FlashMLA), adapted from
    mla_decoding_flashmla_attn_mode_3_pure_bf16_with_pagekv with RoPE removed.

    Returns:
        attn_output: (bsz, 1, hidden)
        k_tensor: (bsz, 1, 1, 576) new KV (already written into pages)
    """
    # K3's pure-BF16 NoPE path only consumes the two FlashMLA entry points.
    # Importing the legacy BatchGen backend here also imports fa3_backend and
    # DeepGEMM, neither of which participates in this forward.  The K3 server
    # has already imported, validated, and kernel-warmed flash_mla before HTTP
    # readiness, so this cached import performs no first-admission setup.
    from flash_mla import (
        flash_mla_with_kvcache,
        get_mla_metadata,
    )

    if gpu_paged_kv_manager is None:
        raise ValueError("gpu_paged_kv_manager must be provided")
    bsz, q_len, _ = hidden_states.size()
    if bsz == 0:
        return hidden_states, torch.empty(
            0, 1, 1, self.kv_lora_rank + self.qk_rope_head_dim,
            dtype=hidden_states.dtype, device=hidden_states.device,
        )
    if q_len != 1:
        raise ValueError("decode path only supports q_len=1")

    hidden_states = hidden_states.squeeze(1)  # (bsz, hidden)

    # ---- projections (q_lora_rank=None direct path) ----
    if self.q_lora_rank is not None:
        q = F.linear(hidden_states, self.q_a_proj.weight)
        q = self.q_a_layernorm(q)
        q = F.linear(q, self.q_b_proj.weight)
    else:
        q = F.linear(hidden_states, self.q_proj.weight)
    new_compressed_kv = F.linear(
        hidden_states, self.kv_a_proj_with_mqa.weight
    ).view(bsz, 1, -1)

    q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(
        q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    )

    # ---- KV cache update (NoPE: no rotation on k_pe) ----
    kv, k_pe = torch.split(
        new_compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )
    normed_kv = self.kv_a_layernorm(kv)
    offload_kv = torch.cat([normed_kv, k_pe], dim=-1)

    manager_device = gpu_paged_kv_manager.device
    k_tensor = offload_kv.view(bsz, 1, 1, offload_kv.size(-1)).to(manager_device)
    sequence_lengths = q_position_ids.squeeze(-1).to(
        dtype=torch.int32, device=manager_device
    )
    gpu_paged_kv_manager.update_layer_decode_new_token(
        k_tensor=k_tensor,
        v_tensor=None,
        sequence_lengths=sequence_lengths,
        layer_idx=layer_idx,
        batch_slice=batch_slice,
    )
    blocked_k, _, block_table = gpu_paged_kv_manager.get_layer_kv_with_page_table(
        layer_idx=layer_idx
    )
    if batch_slice is not None:
        start_idx, end_idx = batch_slice
        block_table = block_table[start_idx:end_idx]

    # ---- absorbed query ----
    kv_b_proj = self.kv_b_proj.weight.data.view(self.num_heads, -1, self.kv_lora_rank)
    q_absorb = kv_b_proj[:, : self.qk_nope_head_dim, :]
    out_absorb = kv_b_proj[:, self.qk_nope_head_dim :, :]

    qk_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
    query_states = torch.empty(
        bsz, self.num_heads, 1, qk_head_dim,
        dtype=blocked_k.dtype, device=blocked_k.device,
    )
    q_nope = q_nope.squeeze(2)
    query_states[:, :, :, : self.kv_lora_rank] = torch.einsum(
        "bhd,hdc->bhc", q_nope, q_absorb
    ).view(bsz, self.num_heads, 1, self.kv_lora_rank)
    query_states[:, :, :, self.kv_lora_rank :] = q_pe
    query_states = query_states.view(bsz, 1, self.num_heads, qk_head_dim)

    tile_scheduler_metadata, num_splits = get_mla_metadata(
        cache_seqlens, self.num_heads, 1
    )
    attn_out, _ = flash_mla_with_kvcache(
        query_states,
        blocked_k,
        block_table,
        cache_seqlens,
        self.kv_lora_rank,
        tile_scheduler_metadata,
        num_splits,
        self.scaling,
        True,
    )

    attn_output = torch.einsum("bqhc,hdc->bhqd", attn_out, out_absorb)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, self.num_heads * self.v_head_dim)
    if self.use_output_gate:
        gate = F.linear(hidden_states, self.g_proj.weight).sigmoid()
        attn_output = attn_output * gate
    attn_output = F.linear(attn_output, self.o_proj.weight)
    attn_output = _reduce_mla_tp_output(self, attn_output)
    return attn_output.view(bsz, 1, -1), k_tensor


# ============================================================================
#  KDA
# ============================================================================


_KDA_DECODE_FUSED_WEIGHT = "_kda_decode_fused_weight"
_KDA_DECODE_FUSED_SIZES = "_kda_decode_fused_sizes"


def fuse_kda_decode_projections(self) -> bool:
    """Fuse K3's independent decode projections into one GEMM weight.

    K3's KDA layer has four large projections (Q/K/V/full-rank output gate),
    one small per-head beta projection, and the first projection of the
    low-rank forget gate.  They all consume the same decode hidden row.  The
    second forget-gate projection remains a dependent GEMM, so the fused path
    reduces the projection front from seven GEMMs to two.

    This is deliberately decode-only: prefill keeps the original projection
    calls so its established numerical path is unchanged.  The original
    ``Linear.weight`` parameters are rebound to views of the single buffer,
    avoiding a second copy of the weights while retaining the prefill API.
    Returns ``False`` for non-K3/low-rank variants or mixed/ineligible dtypes.
    """
    if getattr(self, _KDA_DECODE_FUSED_WEIGHT, None) is not None:
        return True
    if not bool(getattr(self, "use_full_rank_gate", False)):
        return False

    names = ("q_proj", "k_proj", "v_proj", "g_proj", "b_proj", "f_a_proj")
    weights = []
    for name in names:
        projection = getattr(self, name, None)
        weight = getattr(projection, "weight", None)
        if weight is None or weight.ndim != 2 or weight.is_meta:
            return False
        if getattr(projection, "bias", None) is not None:
            return False
        weights.append(weight)

    hidden_size = weights[0].shape[1]
    dtype = weights[0].dtype
    device = weights[0].device
    if any(
        weight.shape[1] != hidden_size
        or weight.dtype != dtype
        or weight.device != device
        for weight in weights
    ):
        return False

    # The order is part of the serving contract consumed by _kda_project.
    sizes = tuple(int(weight.shape[0]) for weight in weights)
    with torch.no_grad():
        fused = torch.cat([weight.detach() for weight in weights], dim=0)
    self.register_buffer(_KDA_DECODE_FUSED_WEIGHT, fused, persistent=False)
    setattr(self, _KDA_DECODE_FUSED_SIZES, sizes)

    start = 0
    for name, size in zip(names, sizes):
        projection = getattr(self, name)
        projection.weight = torch.nn.Parameter(
            fused[start : start + size], requires_grad=False
        )
        start += size
    return True


def _kda_project(self, hidden_states_2d, *, decode=False,
                 return_mixed_qkv=False):
    """Run all KDA projections on packed (total_tokens, hidden) input.

    Returns q, k, v (total, proj), f gate (total, H, K), beta raw (total, H),
    and the output gate z (total, H, K).
    """
    num_heads, head_dim = self.num_heads, self.head_dim
    fused = getattr(self, _KDA_DECODE_FUSED_WEIGHT, None) if decode else None
    if fused is not None:
        # The six rows below are all independent linear maps of the same
        # activation.  Keeping f_a in this GEMM is safe because only f_b is
        # dependent; the fused output is still split into the exact native
        # layouts consumed by the recurrent kernel.
        fused_states = F.linear(hidden_states_2d, fused)
        q, k, v, z, beta, f_a = torch.split(
            fused_states, getattr(self, _KDA_DECODE_FUSED_SIZES), dim=-1
        )
        f = F.linear(f_a, self.f_b_proj.weight).view(-1, num_heads, head_dim)
        z = z.view(-1, num_heads, head_dim)
        mixed_qkv = fused_states[:, : 3 * num_heads * head_dim]
        if return_mixed_qkv:
            return q, k, v, f, beta, z, mixed_qkv
        return q, k, v, f, beta, z

    q = self.q_proj(hidden_states_2d)
    k = self.k_proj(hidden_states_2d)
    v = self.v_proj(hidden_states_2d)
    # No activation between the low-rank stacks (matches model.py:605 and
    # fla's nn.Sequential(Linear, Linear) reference; silu here drifted the
    # gate by up to 6.6e-2 vs the validated oracle — P-6(c) finding).
    f = self.f_b_proj(self.f_a_proj(hidden_states_2d))
    f = f.view(-1, num_heads, head_dim)
    beta = self.b_proj(hidden_states_2d)
    if self.use_full_rank_gate:
        z = self.g_proj(hidden_states_2d).view(-1, num_heads, head_dim)
    else:
        z = self.g_b_proj(self.g_a_proj(hidden_states_2d))
        z = z.view(-1, num_heads, head_dim)
    if return_mixed_qkv:
        return q, k, v, f, beta, z, None
    return q, k, v, f, beta, z


def _kda_fused_conv_args(conv):
    """Expose a K3 ShortConvolution in the fused kernel's native layout.

    K3 stores the depthwise weights as FP32 ``[channels, 1, 4]``.  Squeezing
    the singleton axis produces a stride-preserving ``[channels, 4]`` view;
    no per-layer transpose or dtype conversion is allowed on the decode path.
    """
    weight = getattr(conv, "weight", None)
    if weight is None or weight.dtype is not torch.float32:
        return None
    if weight.ndim == 3:
        if weight.shape[1] != 1:
            return None
        weight = weight[:, 0, :]
    if weight.ndim != 2 or weight.shape[1] != 4:
        return None
    bias = getattr(conv, "bias", None)
    if bias is not None and (
        bias.dtype is not torch.float32 or bias.ndim != 1
        or bias.shape[0] != weight.shape[0]
    ):
        return None
    return weight, bias


def _kda_o_norm_eps(attention):
    """Return the output-norm epsilon across the supported FLA APIs.

    The local K3 model shim calls this field ``variance_epsilon``, while the
    FLA ``FusedRMSNormGated`` used by the serving runtime exposes the same
    value as ``eps``.  Keep the model config as a final fallback so the fused
    kernel receives the checkpoint's configured epsilon instead of failing at
    graph capture.
    """
    eps = getattr(attention.o_norm, "variance_epsilon", None)
    if eps is None:
        eps = getattr(attention.o_norm, "eps", None)
    if eps is None:
        eps = getattr(getattr(attention, "config", None), "rms_norm_eps", None)
    if eps is None:
        raise AttributeError(
            "KDA output norm exposes neither variance_epsilon nor eps, and "
            "the attention config has no rms_norm_eps"
        )
    return eps


def _kda_fused_decode(self, mixed_qkv, f, beta, z, kda_state, slot_ids):
    """Try the AOT K3 end-to-end KDA decode kernel.

    ``None`` means the established FLA chain must be used.  The fallback is
    intentionally retained for non-K3 variants and for installations whose
    wheel predates this optional kernel; production K3 startup still records
    the fallback so it cannot be mistaken for the optimized path.
    """
    global _KDA_FUSED_DECODE_IMPORT_FAILED
    if mixed_qkv is None or not bool(getattr(self, "use_full_rank_gate", False)):
        return None
    try:
        from batchgen_kernels.attention.kda_fused_decode import (
            covered,
            kda_fused_decode,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        if not _KDA_FUSED_DECODE_IMPORT_FAILED:
            logger.warning(
                "K3 fused KDA decode extension unavailable; using FLA chain: %s",
                exc,
            )
            _KDA_FUSED_DECODE_IMPORT_FAILED = True
        return None

    q_args = _kda_fused_conv_args(self.q_conv1d)
    k_args = _kda_fused_conv_args(self.k_conv1d)
    v_args = _kda_fused_conv_args(self.v_conv1d)
    if q_args is None or k_args is None or v_args is None:
        return None
    q_weight, q_bias = q_args
    k_weight, k_bias = k_args
    v_weight, v_bias = v_args
    f_flat = f.reshape(f.shape[0], -1)
    z_flat = z.reshape(z.shape[0], -1)
    if not covered(
        mixed_qkv, f_flat, beta,
        kda_state.conv_q, kda_state.conv_k, kda_state.conv_v,
        q_weight, k_weight, v_weight, q_bias, k_bias, v_bias,
        self.A_log, self.dt_bias, z_flat, self.o_norm.weight,
        kda_state.recurrent_pool, slot_ids,
    ):
        return None
    onorm_eps = _kda_o_norm_eps(self)
    try:
        return kda_fused_decode(
            mixed_qkv, f_flat, beta,
            kda_state.conv_q, kda_state.conv_k, kda_state.conv_v,
            q_weight, k_weight, v_weight, q_bias, k_bias, v_bias,
            self.A_log, self.dt_bias, z_flat, self.o_norm.weight,
            kda_state.recurrent_pool, slot_ids,
            scale=self.head_dim ** -0.5,
            onorm_eps=onorm_eps,
            lower_bound=self.gate_lower_bound,
        )
    except ImportError as exc:
        # AOT import can succeed while the extension's transitive CUDA image
        # is absent from an old wheel. Do not repeatedly probe every KDA layer.
        if not _KDA_FUSED_DECODE_IMPORT_FAILED:
            logger.warning(
                "K3 fused KDA decode failed to load; using FLA chain: %s", exc
            )
            _KDA_FUSED_DECODE_IMPORT_FAILED = True
        return None


def _conv_weights(conv, dtype):
    """Return (weight, bias) for a ShortConvolution cast to the activation dtype.

    Kimi-K3 ships q/k/v_conv1d.weight as fp32 while the projections are bf16;
    the causal_conv1d CUDA kernel requires weight dtype == input dtype. `.to()`
    is a no-op when they already match (e.g. Kimi-Linear 48B).
    """
    bias = getattr(conv, "bias", None)
    return conv.weight.to(dtype), (None if bias is None else bias.to(dtype))


def _linear_no_bias_into(linear, x, out):
    """Evaluate a 2-D bias-free ``nn.Linear`` into caller-owned storage."""
    if linear.bias is not None:
        raise ValueError("linear output reuse requires bias=False")
    if x.ndim != 2 or out.ndim != 2:
        raise ValueError("linear output reuse requires 2-D tensors")
    if out.shape != (x.shape[0], linear.weight.shape[0]):
        raise ValueError("linear output reuse received an incompatible output")
    return torch.mm(x, linear.weight.t(), out=out)


# fla's chunk_kda internal chunk size (its `chunk_size` kwarg default). Every
# segment cut must be a multiple of this measured from the start of the
# sequence it falls in, so that a segment's per-sequence chunk grid is exactly
# the restriction of the unsegmented grid. See _kda_segment_plan.
_KDA_CHUNK_SIZE = 64

# Token budget for one KDA prefill segment. chunk_kda's own scratch is 12
# (T, num_heads * head_dim) bf16-equivalents live at once (w/u/kg, q_l2/k_l2,
# Aqk+Akk, v_new, its output, plus the fp32 gate cumsum and h at 2 each); on K3
# (96 x 128 = 12288) that is 294,912 B per token — 36.0 GiB at S=131,072
# against 4.5 GiB at 16,384. Must be a multiple of _KDA_CHUNK_SIZE.
KDA_PREFILL_SEGMENT_TOKENS = 16384


def _kda_segment_plan(cu_list, segment_tokens):
    """Plan a segmented sweep over a packed (varlen) token range.

    Args:
        cu_list: cu_seqlens as a Python list of ints, ``cu_list[-1] == total``.
        segment_tokens: target segment size; must be a multiple of
            ``_KDA_CHUNK_SIZE``.

    Returns:
        List of ``(start, end, seq_lo, seq_hi, bounds)``: ``[start, end)`` is
        the packed token range of the segment, sequences ``seq_lo:seq_hi``
        overlap it, and ``bounds`` is the segment-relative cu_seqlens for those
        sequences (``bounds[0] == 0``, ``bounds[-1] == end - start``).

    Every cut is either a sequence boundary or ``k * _KDA_CHUNK_SIZE`` tokens
    from the start of the sequence it falls in.  That is what makes the sweep
    EXACT: each sequence's chunk grid inside a segment is the restriction of
    the grid the unsegmented call would have used, so chunk-local work
    (gate cumsum, WY transform, intra-chunk attention) is unchanged and only
    the inter-chunk recurrent state crosses the boundary — in fp32, through the
    state pool.
    """
    if segment_tokens <= 0 or segment_tokens % _KDA_CHUNK_SIZE:
        raise ValueError(
            "segment_tokens ({}) must be a positive multiple of the chunk_kda "
            "chunk size ({})".format(segment_tokens, _KDA_CHUNK_SIZE)
        )
    # Reject zero-length sequences ANYWHERE, not just at the ends. A segment is
    # identified by its token range, so a sequence with no tokens is invisible
    # to the sweep and its state slot would silently never be written. Checking
    # the input is the only check that catches an INTERIOR one; the coverage
    # assertion below cannot, because the segments around it still tile the
    # token axis perfectly.
    if len(cu_list) < 2 or cu_list[0] != 0 or any(
            b <= a for a, b in zip(cu_list, cu_list[1:])):
        raise ValueError(
            "cu_seqlens must start at 0 and be strictly increasing; got {}. A "
            "zero-length sequence would be dropped from the segmented sweep "
            "(and the conv1d kernel cannot handle one either)".format(cu_list)
        )
    total = cu_list[-1]
    plan = []
    start, cut_seq, lo = 0, 0, 0
    while start < total:
        end = start + segment_tokens
        if end >= total:
            end = total
        else:
            # Snap the cut back to a chunk boundary of the sequence it lands
            # in. `start` is itself such a boundary and segment_tokens is a
            # multiple of _KDA_CHUNK_SIZE, so end > start always holds.
            while cu_list[cut_seq + 1] <= end:
                cut_seq += 1
            seq_start = cu_list[cut_seq]
            n_chunks = (end - seq_start) // _KDA_CHUNK_SIZE
            end = seq_start + n_chunks * _KDA_CHUNK_SIZE
        while cu_list[lo + 1] <= start:
            lo += 1
        hi = lo
        while cu_list[hi + 1] < end:
            hi += 1
        bounds = [min(max(c, start), end) - start for c in cu_list[lo:hi + 2]]
        plan.append((start, end, lo, hi + 1, bounds))
        start = end
    # Every sequence must be visited by at least one segment, or its recurrent
    # state slot is never written. This guards the planner loop itself (the
    # precondition above guards its input) — a sequence dropped here would
    # produce a plausible-looking output and a stale state.
    covered = set()
    for _, _, lo, hi, _ in plan:
        covered.update(range(lo, hi))
    if covered != set(range(len(cu_list) - 1)):
        raise ValueError(
            "segment plan misses sequences {} of {}; cu_seqlens={} "
            "segment_tokens={}".format(
                sorted(set(range(len(cu_list) - 1)) - covered),
                len(cu_list) - 1, cu_list, segment_tokens)
        )
    return plan


# One-entry cache of the segment plan for the current prefill microbatch.
# The worker builds ``prepack_cu_seqlens`` on the device once per microbatch
# and every KDA layer receives that same tensor object, so keying on identity
# is exact.  Without it each of the 69 KDA layers re-ran ``cu_seqlens.tolist()``
# (a device sync) and built one small device tensor per segment from host
# memory (a pageable copy that drains the stream): 128 stalls per layer at
# exact 64K, measured as ~100 ms/layer of idle inside the chunk sweep.
_KDA_SEGMENT_PLAN_CACHE = {}


def _kda_cached_segment_plan(cu_seqlens, segment_tokens):
    """Return ``[(start, end, seq_lo, seq_hi, bounds_tensor)]`` for this batch."""
    entry = _KDA_SEGMENT_PLAN_CACHE.get("entry")
    if (
        entry is not None
        and entry["cu_seqlens"] is cu_seqlens
        and entry["segment_tokens"] == segment_tokens
    ):
        return entry["plan"]
    plan = [
        (
            start, end, lo, hi,
            torch.tensor(bounds, dtype=torch.long, device=cu_seqlens.device),
        )
        for start, end, lo, hi, bounds in _kda_segment_plan(
            cu_seqlens.tolist(), segment_tokens
        )
    ]
    _KDA_SEGMENT_PLAN_CACHE["entry"] = {
        "cu_seqlens": cu_seqlens,
        "segment_tokens": segment_tokens,
        "plan": plan,
    }
    return plan


def _kda_chunk_segments(chunk_kda_fn, q, k, v, f, beta, cu_seqlens, slot_ids,
                        recurrent_pool, kernel_kwargs, segment_tokens):
    """Run chunk_kda over a packed range, in token segments, and write the
    per-sequence final recurrent states back into ``recurrent_pool``.

    ``chunk_kda_fn`` is passed in rather than imported so the segment driver
    can be exercised against a CPU reference kernel
    (tests/test_kimi_k3_kda_segmented.py) — fla is CUDA-only.

    Segmenting is a memory optimisation only.  Per segment, chunk_kda is fed
    the same q/k/v/g/beta rows, cut on the same chunk grid, and the recurrent
    state is handed over in fp32 (the pool's dtype, and the dtype chunk_kda
    both requires for ``initial_state`` and produces for ``final_state``), so
    the arithmetic is identical to the unsegmented call.
    """
    slots = slot_ids.long()
    total = q.shape[1]
    if segment_tokens is None or total <= segment_tokens:
        o, recurrent_out = chunk_kda_fn(
            q=q, k=k, v=v, g=f, beta=beta,
            initial_state=recurrent_pool.index_select(0, slots),
            cu_seqlens=cu_seqlens.to(torch.long),
            **kernel_kwargs,
        )
        recurrent_pool.index_copy_(0, slots, recurrent_out)
        return o

    o = torch.empty(q.shape[0], total, v.shape[2], v.shape[3],
                    dtype=v.dtype, device=v.device)
    for start, end, lo, hi, bounds in _kda_cached_segment_plan(
            cu_seqlens, segment_tokens):
        seg_slots = slots[lo:hi]
        o_seg, recurrent_out = chunk_kda_fn(
            q=q[:, start:end], k=k[:, start:end], v=v[:, start:end],
            g=f[:, start:end], beta=beta[:, start:end],
            initial_state=recurrent_pool.index_select(0, seg_slots),
            cu_seqlens=bounds,
            **kernel_kwargs,
        )
        # The sequence that straddles `end` gets a PARTIAL state here; the
        # segment that continues it reads that state back and overwrites it.
        recurrent_pool.index_copy_(0, seg_slots, recurrent_out)
        o[:, start:end] = o_seg
    return o


_KDA_CONV_FALLBACK_WARNED = False


def _kda_prefill_conv(x, weight, bias, conv_pool, cu_seqlens, slot_ids,
                      has_initial_state):
    """Causal conv with state, in ``x``'s storage; token-major Triton on CUDA.

    The Triton path is bit-identical to ``causal_conv1d_fwd`` and about an
    order of magnitude faster at exact 64K; any unsupported shape takes the
    CUDA kernel and says so once.
    """
    global _KDA_CONV_FALLBACK_WARNED
    if x.is_cuda:
        from .kda_conv_triton import (
            kda_causal_conv1d_triton,
            supports_kda_conv_triton,
        )

        if supports_kda_conv_triton(x, weight, conv_pool):
            return kda_causal_conv1d_triton(
                x, weight, bias, conv_pool, cu_seqlens, slot_ids,
                has_initial_state,
            )
        if not _KDA_CONV_FALLBACK_WARNED:
            _KDA_CONV_FALLBACK_WARNED = True
            logging.warning(
                "Kimi-K3 KDA prefill conv shape %s/%s is outside the token-major "
                "Triton contract; using the channel-major CUDA kernel",
                tuple(x.shape), tuple(weight.shape),
            )
    from batchgen_kernels.conv1d import causal_conv1d_fwd

    return causal_conv1d_fwd(
        x, weight, bias=bias,
        conv_states=conv_pool, query_start_loc=cu_seqlens,
        cache_indices=slot_ids, has_initial_state=has_initial_state,
        overwrite_x=True,
    )


def kda_prefill_serving(self, hidden_states_2d, cu_seqlens, slot_ids,
                        has_initial_state, kda_state,
                        segment_tokens=KDA_PREFILL_SEGMENT_TOKENS):
    """KDA prefill for PREPACKED (varlen) sequences.

    Args:
        hidden_states_2d: (total_tokens, hidden) — densely packed varlen tokens.
        cu_seqlens: (num_sequences + 1,) int32 cumulative lengths.
        slot_ids: (num_sequences,) int32 KDA state-pool slot per sequence.
        has_initial_state: (num_sequences,) bool or None.
        kda_state: KDALayerState (conv/recurrent pools; mutated in place).
        segment_tokens: token budget for one chunk_kda call; None or a value
            >= total_tokens reproduces the single-call path exactly.

    Returns:
        (total_tokens, hidden) attention output.

    The projections, the convs and the output norm/projection run over the
    WHOLE packed range; only chunk_kda is segmented.  That split is deliberate:
      - Re-running an nn.Linear at a smaller M changes the cuBLAS kernel choice
        and therefore the result by ~3e-7, which this model amplifies without
        bound (see tests/gpu/test_kimi_k3_kda_fla_parity.py, test_F docstrings).
        So no GEMM shape may change.
      - overwrite_x makes the conv write back into the projection's own buffer,
        so segmenting the conv would save nothing (its input IS its output)
        while it would have to carry the causal width-1 = 3 token context
        across every cut. Not worth the known trap for zero bytes.
      - chunk_kda's own scratch is 12x the token stream against the 6x that has
        to stay resident, so segmenting it alone takes a KDA layer from
        51.0 GiB to 18.0 + 4.5 GiB at S=131,072 / T=16,384.
    """
    from einops import rearrange
    from fla.ops.kda import chunk_kda

    from batchgen_kernels.conv1d import causal_conv1d_fwd

    total = hidden_states_2d.shape[0]
    num_heads, head_dim = self.num_heads, self.head_dim

    profiler, span = _begin_streamed_sp8_profile(self)
    q, k, v, f, beta, z = _kda_project(self, hidden_states_2d)
    _end_streamed_sp8_profile(profiler, "kda_project", span)
    span = profiler.begin_profile_span() if profiler is not None else None

    # conv (silu) with final-state write into the pools at slot_ids.
    # overwrite_x=True: the conv result is transposed back into the projection's
    # own (total, dim) buffer, which is dead after the call. q/k/v stay
    # token-major CONTIGUOUS, so fla's @input_guard .contiguous() is a no-op
    # instead of allocating a copy of each.
    #
    # SIZE OF THAT SAVING, with the segmented sweep below in place: fla copies
    # whatever slice it is handed, and it is handed one SEGMENT, so the win is
    # 3 x (segment_tokens, 12288) bf16 = 1.125 GiB/layer at T=16,384 — not the
    # 3 x (S, 12288) = 9.00 GiB/layer that PREFILL_MEMORY_AUDIT.md fix 2 quotes
    # for the unsegmented sweep it was measured against. Still worth having:
    # nothing else makes the segment slices contiguous, and it is free.
    qw, qb = _conv_weights(self.q_conv1d, q.dtype)
    q = _kda_prefill_conv(
        q, qw, qb, kda_state.conv_q, cu_seqlens, slot_ids, has_initial_state
    )
    kw, kb = _conv_weights(self.k_conv1d, k.dtype)
    k = _kda_prefill_conv(
        k, kw, kb, kda_state.conv_k, cu_seqlens, slot_ids, has_initial_state
    )
    vw, vb = _conv_weights(self.v_conv1d, v.dtype)
    v = _kda_prefill_conv(
        v, vw, vb, kda_state.conv_v, cu_seqlens, slot_ids, has_initial_state
    )

    _end_streamed_sp8_profile(profiler, "kda_conv", span)
    span = profiler.begin_profile_span() if profiler is not None else None

    q = rearrange(q, "l (h d) -> 1 l h d", h=num_heads)
    k = rearrange(k, "l (h d) -> 1 l h d", h=num_heads)
    v = rearrange(v, "l (h d) -> 1 l h d", h=num_heads)
    f = f.unsqueeze(0)        # (1, total, H, K)
    beta = beta.unsqueeze(0)  # (1, total, H)

    o = _kda_chunk_segments(
        chunk_kda, q, k, v, f, beta, cu_seqlens, slot_ids,
        kda_state.recurrent_pool,
        dict(
            A_log=self.A_log, dt_bias=self.dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            lower_bound=self.gate_lower_bound,
            output_final_state=True,
        ),
        segment_tokens,
    )
    _end_streamed_sp8_profile(profiler, "kda_chunk", span)
    span = profiler.begin_profile_span() if profiler is not None else None

    o = self.o_norm(o.reshape(total, num_heads, head_dim), z)
    # All projections from ``hidden_states_2d`` have completed, so its storage
    # is dead until this attention result is returned. Reuse that exact
    # (total, hidden) allocation for the row-parallel output projection. At
    # exact 64K K3 this avoids a late 896 MiB allocation while preserving the
    # same bias-free GEMM (verified bit-identical at the production shape).
    o = _linear_no_bias_into(
        self.o_proj,
        o.reshape(total, num_heads * head_dim),
        hidden_states_2d,
    )
    _end_streamed_sp8_profile(profiler, "kda_output", span)
    # M2a head-parallel KDA: sum the row-parallel o_proj shards. Streamed-SP8
    # retains only this rank's token rows; every other phase all-reduces.
    return _reduce_mla_tp_output(self, o)


def kda_decode_serving(self, hidden_states, kda_state, *, cu_seqlens=None):
    """KDA single-token decode over pooled state.

    Args:
        hidden_states: (bsz, 1, hidden)
        kda_state: layer state (cur_decode_slots must be set by the wrapper).

    Returns:
        (bsz, 1, hidden) attention output.
    """
    bsz = hidden_states.shape[0]
    device = hidden_states.device
    num_heads, head_dim = self.num_heads, self.head_dim

    hidden_2d = hidden_states.squeeze(1)
    q, k, v, f, beta, z, mixed_qkv = _kda_project(
        self, hidden_2d, decode=True, return_mixed_qkv=True
    )

    slot_ids = kda_state.cur_decode_slots  # (bsz,) int32

    fused_output = _kda_fused_decode(
        self, mixed_qkv, f, beta, z, kda_state, slot_ids
    )
    if fused_output is not None:
        o = self.o_proj(fused_output)
        o = _reduce_mla_tp_output(self, o)
        return o.unsqueeze(1)

    # Established correctness fallback for Kimi-Linear-48B, old wheels, and
    # any K3 tensor layout outside the AOT kernel's covered contract.
    from fla.ops.kda.fused_recurrent import fused_recurrent_kda_fwd

    from batchgen_kernels.conv1d import causal_conv1d_update

    qw, qb = _conv_weights(self.q_conv1d, q.dtype)
    q = causal_conv1d_update(
        q, kda_state.conv_q, qw, bias=qb,
        conv_state_indices=slot_ids,
    )
    kw, kb = _conv_weights(self.k_conv1d, k.dtype)
    k = causal_conv1d_update(
        k, kda_state.conv_k, kw, bias=kb,
        conv_state_indices=slot_ids,
    )
    vw, vb = _conv_weights(self.v_conv1d, v.dtype)
    v = causal_conv1d_update(
        v, kda_state.conv_v, vw, bias=vb,
        conv_state_indices=slot_ids,
    )

    q = q.view(1, bsz, num_heads, head_dim)
    k = k.view(1, bsz, num_heads, head_dim)
    v = v.view(1, bsz, num_heads, head_dim)
    f = f.view(1, bsz, num_heads, head_dim)
    beta = beta.view(1, bsz, num_heads)
    if cu_seqlens is None:
        cu = torch.arange(bsz + 1, dtype=torch.long, device=device)
    else:
        cu = cu_seqlens

    o = torch.empty(1, bsz, num_heads, head_dim, dtype=v.dtype, device=device)
    fused_recurrent_kda_fwd(
        q=q, k=k, v=v, g=f, beta=beta,
        A_log=self.A_log, dt_bias=self.dt_bias,
        initial_state=kda_state.recurrent_pool,
        output_final_state=True,
        inplace_final_state=True,
        cu_seqlens=cu,
        ssm_state_indices=slot_ids,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=self.gate_lower_bound,
        out=o,
    )

    o = self.o_norm(o.reshape(bsz, num_heads, head_dim), z)
    o = self.o_proj(o.reshape(bsz, num_heads * head_dim))
    o = _reduce_mla_tp_output(self, o)
    return o.unsqueeze(1)


# ============================================================================
#  MoE (EP)
# ============================================================================


def _grouped_moe_bf16(x, w13, w2, weights, ids):
    """Reference grouped BF16 MoE over the local EP shard (Phase 0; correct,
    O(num_local_experts) loop — replaced by a fused kernel in Phase 1).

    Args:
        x: (T, H) tokens.
        w13: (E_local, 2I, H) stacked gate+up weights.
        w2: (E_local, H, I) down weights.
        weights: (T, top_k) routing weights (already scaled), zeroed for
            non-local experts.
        ids: (T, top_k) local expert index (clamped to [0, E_local)).

    Returns:
        (T, H) summed local-expert contribution.
    """
    T, H = x.shape
    E = w13.shape[0]
    out = torch.zeros_like(x)
    for e in range(E):
        mask = ids == e                     # (T, top_k)
        tok_sel = mask.any(dim=1)
        if not bool(tok_sel.any()):
            continue
        tok_idx = tok_sel.nonzero(as_tuple=False).squeeze(-1)
        xe = x.index_select(0, tok_idx)     # (n, H)
        h = xe @ w13[e].t()                 # (n, 2I)
        gate, up = h.chunk(2, dim=-1)
        h = F.silu(gate) * up
        ye = h @ w2[e].t()                  # (n, H)
        w = (weights * mask).sum(dim=1).index_select(0, tok_idx).unsqueeze(-1)
        out.index_add_(0, tok_idx, ye * w.to(ye.dtype))
    return out


def _require_k3_latent_moe(self):
    """A ``kimi_k3`` config MUST reach the LatentMoE path — or die here.

    ``KimiSparseMoeBlock.use_latent_moe`` is derived from
    ``config.routed_expert_hidden_size``, which K3 declares under
    ``text_config`` (3584). A config that says ``model_type == 'kimi_k3'`` but
    arrives WITHOUT that key — a truncated dict, a hand-built config, a
    ``routed_expert_hidden_size`` that got filtered out by
    ``from_hf_dict``'s unknown-key filter — builds hidden-space experts and
    runs the 7168 hidden straight into 3584-wide latent weights. Same class of
    silent-default bug as the 896 -> 256 expert undercount
    (``config.require_num_routed_experts``), so it fails loudly instead.

    SCOPE, precisely: the only trigger is ``model_type == 'kimi_k3'``. A config
    that never went through ``KimiLinearConfig.from_hf_dict`` at all is NOT
    caught here — ``from_hf_dict`` is what stamps ``'kimi_k3'`` (config.py
    :169-170); without it the field holds the dataclass default
    ``'kimi_linear'`` and this guard returns silently. That case is covered
    elsewhere, loudly: ``Parallel_Strategy_Manager._is_k3`` keys off the same
    attribute, so a K3 with a stripped ``model_type`` fails the skeleton lookup
    first with all 1026 skeleton params missing.
    """
    cfg = getattr(self, "config", None)
    if getattr(cfg, "model_type", None) != "kimi_k3":
        return                      # 48B (and every other model): no LatentMoE
    if not getattr(self, "use_latent_moe", False):
        raise RuntimeError(
            "Kimi-K3 MoE reached the hidden-space (non-latent) branch: "
            "config.routed_expert_hidden_size is None. K3's routed experts "
            "live in a LATENT space (w1/w3: 3584->3072, w2: 3072->3584) fed "
            "by routed_expert_down_proj (7168->3584); running them on the "
            "7168 hidden is wrong math. Load the real config.json so "
            "text_config.routed_expert_hidden_size=3584 is parsed."
        )
    if not getattr(self, "latent_moe_use_norm", False):
        raise RuntimeError(
            "Kimi-K3 MoE has latent_moe_use_norm=False, but the checkpoint "
            "ships block_sparse_moe.routed_expert_norm on every MoE layer "
            "(k3/tensor_map.py:326-330). Skipping that norm silently changes "
            "the routed_expert_up_proj input scale."
        )


def _merge_resident_prefill_shared(routed, shared):
    """Add the shared path without allocating a second full hidden output."""
    routed.add_(shared)
    return routed


def moe_forward_resident_ep_decode(self, hidden_states, resident):
    """KimiSparseMoeBlock DECODE forward — resident EP-8 + fused BF16 MoE.

    The routed-expert math (pad -> all_gather -> router on global tokens ->
    fused_moe_bf16 on the local shard -> all_reduce -> local slice) lives in
    the ResidentEPMoELayer seam; this function only adds the DP-local shared
    expert. Empty ranks (0 rows) MUST still reach resident.forward — the
    collectives run on every rank, every decode step (worker :9746 no-skip
    invariant); only the returned local slice is empty.

    HIDDEN-SPACE ONLY: the resident shard stacks w1/w2/w3 and routes in the
    hidden space, so a LatentMoE config has no representation here.
    """
    if getattr(self, "use_latent_moe", False):
        # M3.1a (A13): MXFP4 LatentMoE resident decode. The resident layer runs
        # the full latent expert path (routed_expert_down_proj once/token ->
        # grouped MXFP4 S1(SiTU)+S3 on the resident shard -> fp32 top-k combine
        # -> routed_expert_norm -> routed_expert_up_proj); THIS seam only adds
        # the DP-local shared expert, exactly like the BF16 branch below. A
        # latent config MUST have built ResidentEPMXFP4MoELayer (PSM
        # is_mxfp4_quantized branch); the BF16 stacked shard has no latent seam
        # and is refused here rather than silently run in the wrong space.
        from batchgen.moe.fused_moe_mxfp4_resident import ResidentEPMXFP4MoELayer

        if not isinstance(resident, ResidentEPMXFP4MoELayer):
            raise RuntimeError(
                "LatentMoE decode reached moe_forward_resident_ep_decode with a "
                f"non-MXFP4 resident ({type(resident).__name__}): the BF16 "
                "stacked shard (fused_moe_bf16_resident) has no "
                "routed_expert_down/norm/up latent seam. A K3 latent config must "
                "build ResidentEPMXFP4MoELayer via the PSM is_mxfp4_quantized "
                "branch."
            )
        # M3.1b (A13/A16): the EP all_gather/all_reduce now live INSIDE
        # resident.forward. Under TP-G decode the G ranks of a group hold the
        # SAME rows (replicated attention) but the resident layer is a DP
        # contract, so — exactly like the BF16 branch below — scatter the
        # group's rows into G distinct DP slices, route each through the
        # UNCHANGED latent forward, then gather back to the full group batch.
        identity = hidden_states
        orig_shape = hidden_states.shape
        x = hidden_states.reshape(-1, self.hidden_dim)
        G = int(getattr(self, "attn_tp_size", 1))
        if G > 1:
            from .moe_tp_reshard import all_gather_rows, scatter_rows

            B_grp = x.shape[0]
            x_local = scatter_rows(x, G, self.attn_tp_rank)
            routed_local = resident.forward(x_local, self.gate)
            routed = all_gather_rows(
                routed_local, B_grp, G, self.attn_tp_rank, self.attn_tp_group
            )
        else:
            routed = resident.forward(x, self.gate)
        out = routed.reshape(orig_shape)
        if getattr(self, "shared_experts", None) is not None:
            shared = self.shared_experts(identity)
            if getattr(self, "_resident_ep_prefill_enabled", False):
                out = _merge_resident_prefill_shared(out, shared)
            else:
                out = out + shared
        return out

    identity = hidden_states
    orig_shape = hidden_states.shape
    x = hidden_states.reshape(-1, self.hidden_dim)

    # M2b: under DP-(world/G) x TP-G decode the G ranks of a group hold the SAME
    # rows (replicated attention), but ResidentEPMoELayer is a DP-32 contract.
    # Scatter the group's rows into G distinct slices (DP-32 restored), route
    # each slice through the UNCHANGED resident forward, then gather back to the
    # full group batch (attention downstream needs all rows on every rank). An
    # empty rank (B_grp<G) still runs resident.forward + the gather (lockstep).
    G = int(getattr(self, "attn_tp_size", 1))
    if G > 1:
        from .moe_tp_reshard import all_gather_rows, scatter_rows

        B_grp = x.shape[0]
        x_local = scatter_rows(x, G, self.attn_tp_rank)
        routed_local = resident.forward(x_local, self.gate)
        routed = all_gather_rows(
            routed_local, B_grp, G, self.attn_tp_rank, self.attn_tp_group
        )
    else:
        routed = resident.forward(x, self.gate)

    out = routed.reshape(orig_shape)
    if getattr(self, "shared_experts", None) is not None:
        out = out + self.shared_experts(identity)
    return out


def moe_forward_serving(self, hidden_states):
    """KimiSparseMoeBlock serving forward.

    DECODE with a materialized resident shard (decode_moe_mode="resident_ep",
    injected by the PSM at configure_decoding) dispatches to
    moe_forward_resident_ep_decode — no streaming, one all_gather +
    all_reduce per layer. Everything below serves PREFILL (and the
    decode_moe_mode="streamed" fallback): pure DP with streamed experts.

    Each rank independently processes its own tokens; the routed experts are
    host-offloaded and streamed to GPU per layer by the copy engine. The MoE
    drives EVERY expert wrapper in ascending index order (== weight-copy-task
    order) so the producer never stalls: experts with no routed tokens still
    run a load+free (via a 0-row forward). No cross-rank collectives.

    LatentMoE (K3, ``config.routed_expert_hidden_size``): the routed experts
    do NOT live in the hidden space. Op order, from the eager reference
    (kimi_k3/model.py::KimiSparseMoeBlock, bit-exact to the HF oracle):

        router(identity)                    # PRE-down 7168 hidden, fp32
        x = routed_expert_down_proj(x)      # 7168 -> 3584, ONCE per token,
                                            #   BEFORE dispatch
        y = sum_k w_k * expert_k(x)         # experts in the 3584 latent,
                                            #   combine in FP32
        y = routed_expert_norm(y)           # ONCE, post-combine, pre-up
        y = routed_expert_up_proj(y)        # 3584 -> 7168
        out = y + shared_experts(identity)  # shared expert in HIDDEN space

    Two ways to get the seam wrong, and they are NOT the same kind of wrong —
    stated separately because conflating them mis-sold the evidence once
    already:

      * norm per expert instead of once post-combine, or after up_proj instead
        of before: a DIFFERENT FUNCTION. MEASURED err_ratio 1.486e-01 and
        8.180e-01 against the eager reference. Numeric parity catches these.
      * down-proj per (token, expert) instead of once per token: the SAME
        function, 16x wasted FLOPs. ``routed_expert_down_proj`` is a bias-free
        ``nn.Linear``, so applying it to a duplicated row is idempotent —
        MEASURED err_ratio 0.000e+00 (bf16) / 2.259e-07 (fp32) with the
        duplication in place. Numeric parity CANNOT see this; it is pinned
        instead by the call-count hook test
        (``tests/gpu/test_kimi_linear_latent_moe_serving.py``, which asserts
        one down-proj call with T rows, not T*top_k).

    Requires (set by the PSM):
        self.experts   — ModuleList of KimiLinearExpertWrapper (streamed)
        self.gate      — router; returns (topk_idx, topk_weight[, ...])
        self.shared_experts — resident BF16 shared expert (or None)
        LatentMoE only: self.routed_expert_{down_proj,norm,up_proj}
    """
    _require_k3_latent_moe(self)

    resident = getattr(self, "_resident_ep_moe", None)
    resident_prefill = bool(
        getattr(self, "_resident_ep_prefill_enabled", False)
    )
    if resident is not None and (
        KimiLinearExpertWrapper.phase == "decode" or resident_prefill
    ):
        return moe_forward_resident_ep_decode(self, hidden_states, resident)

    streamed_sp8 = getattr(self, "_streamed_sp8_moe", None)
    if (
        streamed_sp8 is not None
        and getattr(self, "_streamed_sp8_prefill_enabled", False)
        and KimiLinearExpertWrapper.phase == "prefill"
    ):
        profiler = type(streamed_sp8)
        profile = profiler._prefill_profile_enabled
        moe_span = profiler.begin_profile_span() if profile else None
        orig_shape = hidden_states.shape
        x = hidden_states.reshape(-1, self.hidden_dim)
        G = int(getattr(self, "attn_tp_size", 1))
        if G <= 1:
            raise RuntimeError(
                "streamed-SP8 prefill reached MoE without TP row sharding"
            )
        from .moe_tp_reshard import (
            all_gather_rows_add_,
            all_gather_rows,
            balanced_row_split,
            scatter_rows,
        )

        global_rows = getattr(self, "_streamed_sp8_global_rows", None)
        input_sharded = bool(
            getattr(self, "_streamed_sp8_sharded_carry", False)
            and global_rows is not None
        )
        num_rows = int(global_rows if input_sharded else x.shape[0])
        splits = balanced_row_split(num_rows, G)
        local_start, local_end = splits[self.attn_tp_rank]
        if input_sharded:
            if x.shape[0] != local_end - local_start:
                raise ValueError(
                    "streamed-SP8 MoE input has the wrong local row count"
                )
            x_local = x
        else:
            x_local = scatter_rows(x, G, self.attn_tp_rank)
        # The layer is expert-parallel inside the node, so it needs the node's
        # pre-split row count to pad this slice to the shared ntp stride its
        # node-local latent gather and reduce-scatter are laid out on.
        routed_local = streamed_sp8.forward(x_local, self.gate, num_rows)
        if getattr(self, "shared_experts", None) is not None:
            shared_span = profiler.begin_profile_span() if profile else None
            if input_sharded:
                # Shared-expert weights are row-parallel over their
                # intermediate dimension, so every TP rank still needs every
                # input row. Reassemble those rows, run the unchanged local
                # weight shard, then preserve the established BF16 all-reduce
                # before retaining this rank's row slice. Calling
                # ``forward_into`` on distinct local rows would all-reduce
                # unrelated tokens and is mathematically wrong. A direct
                # reduce-scatter was rejected by the world-8 parity gate: its
                # different NCCL reduction order exceeded the max-error budget.
                gather_span = (
                    profiler.begin_profile_span() if profile else None
                )
                shared_input = all_gather_rows(
                    x,
                    num_rows,
                    G,
                    self.attn_tp_rank,
                    self.attn_tp_group,
                )
                if profile:
                    profiler.end_profile_span(
                        "shared_input_gather", gather_span
                    )
                shared_partial = self.shared_experts._ffn_into(
                    shared_input, shared_input
                )
                reduce_span = (
                    profiler.begin_profile_span() if profile else None
                )
                import torch.distributed as dist

                dist.all_reduce(
                    shared_partial, group=self.attn_tp_group
                )
                shared_output = scatter_rows(
                    shared_partial, G, self.attn_tp_rank
                ).clone()
                del shared_partial
                if profile:
                    profiler.end_profile_span(
                        "shared_expert_reduce", reduce_span
                    )
            else:
                shared_output = self.shared_experts.forward_into(x, x)
            if profile:
                profiler.end_profile_span("shared_expert", shared_span)
        else:
            shared_output = x.zero_()
        if input_sharded:
            shared_output.add_(routed_local)
            out = shared_output.reshape(orig_shape)
            if profile:
                profiler.end_profile_span("moe_serving_total", moe_span)
            streamed_sp8.buffer.allow_cross_launch()
            return out

        routed_gather_span = (
            profiler.begin_profile_span() if profile else None
        )
        all_gather_rows_add_(
            x,
            routed_local,
            num_rows,
            G,
            self.attn_tp_rank,
            self.attn_tp_group,
        )
        if profile:
            profiler.end_profile_span(
                "routed_output_gather", routed_gather_span
            )
        out = x.reshape(orig_shape)
        # Every TP8 collective of this layer -- the node-local latent/routing
        # gathers, the FP32 reduce-scatter, the routed-output all-gather above
        # and the shared expert's row-parallel all-reduce -- has now been
        # issued. Only here may the parked prefetch thread launch the next
        # layer's cross-node broadcasts, so under NCCL_LAUNCH_ORDER_IMPLICIT=1
        # their payloads overlap the next layer's attention compute instead of
        # pushing the peers' wait into one of the collectives above (E1).
        if profile:
            profiler.end_profile_span("moe_serving_total", moe_span)
        streamed_sp8.buffer.allow_cross_launch()
        return out

    identity = hidden_states
    orig_shape = hidden_states.shape
    x = hidden_states.reshape(-1, self.hidden_dim)
    num_tokens = x.shape[0]
    device = x.device
    use_latent = bool(getattr(self, "use_latent_moe", False))

    # Routing reads the PRE-down-proj hidden (`identity`), never the latent.
    if num_tokens == 0:
        # Empty DP rank: KimiMoEGate dies on scores.view(0, -1). Build empty
        # routing instead so the drive loop below still load+frees every
        # expert in lockstep with the non-empty ranks (first-smoke finding).
        topk_idx = x.new_empty((0, self.gate.top_k), dtype=torch.long)
        topk_weight = x.new_empty((0, self.gate.top_k))
    else:
        gate_out = self.gate(identity)
        topk_idx, topk_weight = gate_out[0], gate_out[1]
    K = topk_idx.shape[-1]

    if use_latent:
        # ONCE per token, before dispatch — a token routed to 16 experts is
        # projected once, not 16 times.
        x = self.routed_expert_down_proj(x)
    expert_dim = x.shape[-1]

    flat_expert_idx = topk_idx.reshape(-1)
    token_indices = torch.arange(num_tokens, device=device).repeat_interleave(K)

    results = torch.zeros(num_tokens, expert_dim, device=device,
                          dtype=torch.float32)

    for expert_idx, expert in enumerate(self.experts):
        if expert is None:
            continue
        mask = flat_expert_idx == expert_idx
        sel = mask.nonzero(as_tuple=False).squeeze(-1)
        expert_token_idx = token_indices[sel]
        # Drive the wrapper even for 0 tokens (streamed load+free, no stall).
        tokens_for_expert = x.index_select(0, expert_token_idx)
        expert_output = expert(tokens_for_expert)
        if sel.numel() == 0:
            continue
        w = topk_weight.reshape(-1)[sel].unsqueeze(-1)
        results.index_add_(0, expert_token_idx,
                           expert_output.float() * w.float())

    # FP32 combine -> model dtype, exactly where the reference downcasts
    # (kimi_k3/model.py::_moe_combine's trailing `.type(new_x.dtype)`).
    y = results.to(identity.dtype)
    if use_latent:
        if self.latent_moe_use_norm:
            y = self.routed_expert_norm(y)
        y = self.routed_expert_up_proj(y)

    out = y.reshape(orig_shape)
    if getattr(self, "shared_experts", None) is not None:
        out = out + self.shared_experts(identity)
    return out
