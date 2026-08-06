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

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .wrappers import KimiLinearExpertWrapper

# ============================================================================
#  NoPE-MLA
# ============================================================================


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
    return self.o_proj(out), offload_kv


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
    # FA3 (flash_attn_interface); FA2 `flash_attn` is not assumed to be
    # installed. Same kwargs; the tuple return is normalized below.
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
    from batchgen.attention.mla.flashmla_backend import (
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
    return attn_output.view(bsz, 1, -1), k_tensor


# ============================================================================
#  KDA
# ============================================================================


def _kda_project(self, hidden_states_2d):
    """Run all KDA projections on packed (total_tokens, hidden) input.

    Returns q, k, v (total, proj), f gate (total, H, K), beta raw (total, H),
    and the output gate z (total, H, K).
    """
    num_heads, head_dim = self.num_heads, self.head_dim
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
    return q, k, v, f, beta, z


def _conv_weights(conv, dtype):
    """Return (weight, bias) for a ShortConvolution cast to the activation dtype.

    Kimi-K3 ships q/k/v_conv1d.weight as fp32 while the projections are bf16;
    the causal_conv1d CUDA kernel requires weight dtype == input dtype. `.to()`
    is a no-op when they already match (e.g. Kimi-Linear 48B).
    """
    bias = getattr(conv, "bias", None)
    return conv.weight.to(dtype), (None if bias is None else bias.to(dtype))


def kda_prefill_serving(self, hidden_states_2d, cu_seqlens, slot_ids,
                        has_initial_state, kda_state):
    """KDA prefill for PREPACKED (varlen) sequences.

    Args:
        hidden_states_2d: (total_tokens, hidden) — densely packed varlen tokens.
        cu_seqlens: (num_sequences + 1,) int32 cumulative lengths.
        slot_ids: (num_sequences,) int32 KDA state-pool slot per sequence.
        has_initial_state: (num_sequences,) bool or None.
        kda_state: KDALayerState (conv/recurrent pools; mutated in place).

    Returns:
        (total_tokens, hidden) attention output.
    """
    from einops import rearrange
    from fla.ops.kda import chunk_kda

    from batchgen_kernels.conv1d import causal_conv1d_fwd

    total = hidden_states_2d.shape[0]
    num_heads, head_dim = self.num_heads, self.head_dim

    q, k, v, f, beta, z = _kda_project(self, hidden_states_2d)

    # conv (silu) with final-state write into the pools at slot_ids
    qw, qb = _conv_weights(self.q_conv1d, q.dtype)
    q = causal_conv1d_fwd(
        q, qw, bias=qb,
        conv_states=kda_state.conv_q, query_start_loc=cu_seqlens,
        cache_indices=slot_ids, has_initial_state=has_initial_state,
    )
    kw, kb = _conv_weights(self.k_conv1d, k.dtype)
    k = causal_conv1d_fwd(
        k, kw, bias=kb,
        conv_states=kda_state.conv_k, query_start_loc=cu_seqlens,
        cache_indices=slot_ids, has_initial_state=has_initial_state,
    )
    vw, vb = _conv_weights(self.v_conv1d, v.dtype)
    v = causal_conv1d_fwd(
        v, vw, bias=vb,
        conv_states=kda_state.conv_v, query_start_loc=cu_seqlens,
        cache_indices=slot_ids, has_initial_state=has_initial_state,
    )

    q = rearrange(q, "l (h d) -> 1 l h d", h=num_heads)
    k = rearrange(k, "l (h d) -> 1 l h d", h=num_heads)
    v = rearrange(v, "l (h d) -> 1 l h d", h=num_heads)
    f = f.unsqueeze(0)        # (1, total, H, K)
    beta = beta.unsqueeze(0)  # (1, total, H)

    recurrent_in = kda_state.recurrent_pool.index_select(0, slot_ids.long())
    o, recurrent_out = chunk_kda(
        q=q, k=k, v=v, g=f, beta=beta,
        A_log=self.A_log, dt_bias=self.dt_bias,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=self.gate_lower_bound,
        initial_state=recurrent_in,
        output_final_state=True,
        cu_seqlens=cu_seqlens.to(torch.long),
    )
    kda_state.recurrent_pool.index_copy_(0, slot_ids.long(), recurrent_out)

    o = self.o_norm(o.reshape(total, num_heads, head_dim), z)
    o = self.o_proj(o.reshape(total, num_heads * head_dim))
    return o


def kda_decode_serving(self, hidden_states, kda_state):
    """KDA single-token decode over pooled state.

    Args:
        hidden_states: (bsz, 1, hidden)
        kda_state: layer state (cur_decode_slots must be set by the wrapper).

    Returns:
        (bsz, 1, hidden) attention output.
    """
    from fla.ops.kda.fused_recurrent import fused_recurrent_kda_fwd

    from batchgen_kernels.conv1d import causal_conv1d_update

    bsz = hidden_states.shape[0]
    device = hidden_states.device
    num_heads, head_dim = self.num_heads, self.head_dim

    hidden_2d = hidden_states.squeeze(1)
    q, k, v, f, beta, z = _kda_project(self, hidden_2d)

    slot_ids = kda_state.cur_decode_slots  # (bsz,) int32

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
    cu = torch.arange(bsz + 1, dtype=torch.long, device=device)

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
        raise NotImplementedError(
            "resident-EP decode does not implement LatentMoE: the stacked "
            "shard (batchgen.moe.fused_moe_bf16_resident) routes and runs the "
            "experts in the hidden space, with no routed_expert_down_proj / "
            "routed_expert_norm / routed_expert_up_proj seam. Run K3 decode "
            "with decode_moe_mode='streamed' (moe_forward_serving's streamed "
            "path implements the latent form) until the resident shard grows "
            "one."
        )
    identity = hidden_states
    orig_shape = hidden_states.shape
    x = hidden_states.reshape(-1, self.hidden_dim)

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
    if resident is not None and KimiLinearExpertWrapper.phase == "decode":
        return moe_forward_resident_ep_decode(self, hidden_states, resident)

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
