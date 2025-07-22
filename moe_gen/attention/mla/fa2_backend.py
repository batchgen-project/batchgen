import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func

from .padding import _upad_input, pad_input
from .rotary_embedding import apply_rotary_pos_emb, rotate_half


@torch.inference_mode()
def mla_prefill_flashattention2(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
):
    bsz, seq_len, _ = hidden_states.shape
    query_states = self.q_b_proj(
        self.q_a_layernorm(self.q_a_proj(hidden_states))
    )
    query_states = query_states.view(
        bsz, seq_len, self.num_heads, self.q_head_dim
    ).transpose(1, 2)
    q_nope, q_pe = torch.split(
        query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    )
    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    compressed_kv_ref, k_pe = torch.split(
        compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )
    k_pe = k_pe.view(bsz, seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)
    kv = (
        self.kv_b_proj(self.kv_a_layernorm(compressed_kv_ref))
        .view(
            bsz,
            seq_len,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        .transpose(1, 2)
    )
    k_nope, value_states = torch.split(
        kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
    )
    cos, sin = self.rotary_emb(value_states, seq_len=seq_len)
    q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

    query_states = k_pe.new_empty(bsz, self.num_heads, seq_len, self.q_head_dim)
    query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
    query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

    key_states = k_pe.new_empty(bsz, self.num_heads, seq_len, self.q_head_dim)
    key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
    key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

    # Concat value states with zero to have same head dim as key states
    value_states = torch.cat(
        [
            value_states,
            torch.zeros(
                (
                    bsz,
                    self.num_heads,
                    seq_len,
                    self.q_head_dim - self.v_head_dim,
                ),
                dtype=value_states.dtype,
                device=value_states.device,
            ),
        ],
        dim=-1,
    )

    query_states = query_states.transpose(1, 2).contiguous()
    key_states = key_states.transpose(1, 2).contiguous()
    value_states = value_states.transpose(1, 2).contiguous()

    (
        query_states,
        key_states,
        value_states,
        indices_q,
        cu_seq_lens,
        max_seq_lens,
    ) = _upad_input(
        query_states, key_states, value_states, attention_mask, seq_len
    )
    cu_seqlens_q, cu_seqlens_k = cu_seq_lens
    max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
    attn_output_unpad = flash_attn_varlen_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_in_batch_q,
        max_seqlen_k=max_seqlen_in_batch_k,
        softmax_scale=self.softmax_scale,
        causal=True,
    )
    attn_output = pad_input(attn_output_unpad, indices_q, bsz, seq_len).view(
        bsz, seq_len, self.num_heads, self.q_head_dim
    )
    # revert to head_dim_k
    attn_output = attn_output[:, :, :, : self.v_head_dim]
    attn_output = attn_output.reshape(
        bsz, seq_len, self.num_heads * self.v_head_dim
    ).contiguous()

    attn_output = self.o_proj(attn_output)

    return attn_output, compressed_kv


# @torch.inference_mode()
# def mla_chunked_prefill_flashattention2(
#     self,
#     hidden_states: torch.Tensor,
#     attention_mask: torch.Tensor,
#     position_ids: torch.Tensor,
#     chunk_size: int = 1024,  # Added chunk_size parameter with default 1024
# ):
#     bsz, seq_len, _ = hidden_states.shape

#     # Process q projection
#     query_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
#     query_states = query_states.view(bsz, seq_len, self.num_heads, self.q_head_dim).transpose(1, 2)
#     q_nope, q_pe = torch.split(
#         query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
#     )

#     # Process kv projection
#     compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
#     compressed_kv_ref, k_pe = torch.split(
#         compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
#     )

#     k_pe = k_pe.view(bsz, seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)
#     kv = (
#         self.kv_b_proj(self.kv_a_layernorm(compressed_kv_ref))
#         .view(
#             bsz, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
#         )
#         .transpose(1, 2)
#     )
#     k_nope, value_states = torch.split(
#         kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
#     )

#     # Apply rotary position embeddings
#     cos, sin = self.rotary_emb(value_states, seq_len=seq_len)
#     q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

#     # Combine nope and pe components for query and key states
#     query_states = k_pe.new_empty(bsz, self.num_heads, seq_len, self.q_head_dim)
#     query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
#     query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

#     key_states = k_pe.new_empty(
#         bsz, self.num_heads, seq_len, self.q_head_dim
#     )
#     key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
#     key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

#     # Pad value states to match key head dimension as required by FlashAttention
#     value_states = torch.cat([
#         value_states,
#         torch.zeros(
#             (bsz, self.num_heads, seq_len, self.q_head_dim - self.v_head_dim),
#             dtype=value_states.dtype,
#             device=value_states.device
#         )
#     ], dim=-1)

#     # Transpose to match FlashAttention expected format
#     query_states = query_states.transpose(1, 2).contiguous()  # [bsz, seq_len, num_heads, head_dim]
#     key_states = key_states.transpose(1, 2).contiguous()
#     value_states = value_states.transpose(1, 2).contiguous()

#     # Initialize output tensor to store chunked results
#     attn_output = torch.zeros(
#         bsz, seq_len, self.num_heads, self.v_head_dim,
#         dtype=query_states.dtype,
#         device=query_states.device
#     )

#     # Process in chunks if chunk_size is specified and smaller than sequence length
#     if chunk_size is None or seq_len <= chunk_size:
#         # Process entire sequence at once with FlashAttention
#         (
#             query_states_unpad,
#             key_states_unpad,
#             value_states_unpad,
#             indices_q,
#             cu_seq_lens,
#             max_seq_lens,
#         ) = _upad_input(
#             query_states, key_states, value_states, attention_mask, seq_len
#         )

#         cu_seqlens_q, cu_seqlens_k = cu_seq_lens
#         max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

#         attn_output_unpad = flash_attn_varlen_func(
#             query_states_unpad,
#             key_states_unpad,
#             value_states_unpad,
#             cu_seqlens_q=cu_seqlens_q,
#             cu_seqlens_k=cu_seqlens_k,
#             max_seqlen_q=max_seqlen_in_batch_q,
#             max_seqlen_k=max_seqlen_in_batch_k,
#             softmax_scale=self.softmax_scale,
#             causal=True
#         )

#         attn_output = pad_input(
#             attn_output_unpad, indices_q, bsz, seq_len
#         ).view(
#             bsz, seq_len, self.num_heads, self.q_head_dim
#         )

#     else:
#         # Process in chunks
#         num_chunks = (seq_len + chunk_size - 1) // chunk_size  # Ceiling division

#         for chunk_idx in range(num_chunks):
#             chunk_start = chunk_idx * chunk_size
#             chunk_end = min(chunk_start + chunk_size, seq_len)
#             chunk_length = chunk_end - chunk_start

#             # Extract query chunk
#             query_chunk = query_states[:, chunk_start:chunk_end, :, :]

#             # Create attention mask for this chunk if provided
#             # For FlashAttention, we need to handle masking differently
#             chunk_attention_mask = attention_mask
#             if attention_mask is not None:
#                 chunk_attention_mask = attention_mask[:, :, chunk_start:chunk_end, :]

#             # Process this chunk with FlashAttention
#             (
#                 query_chunk_unpad,
#                 key_states_unpad,
#                 value_states_unpad,
#                 indices_q,
#                 cu_seq_lens,
#                 max_seq_lens,
#             ) = _upad_input(
#                 query_chunk, key_states, value_states, chunk_attention_mask, chunk_length
#             )

#             cu_seqlens_q, cu_seqlens_k = cu_seq_lens
#             max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

#             chunk_output_unpad = flash_attn_varlen_func(
#                 query_chunk_unpad,
#                 key_states_unpad,
#                 value_states_unpad,
#                 cu_seqlens_q=cu_seqlens_q,
#                 cu_seqlens_k=cu_seqlens_k,
#                 max_seqlen_q=max_seqlen_in_batch_q,
#                 max_seqlen_k=max_seqlen_in_batch_k,
#                 softmax_scale=self.softmax_scale,
#                 causal=True
#             )

#             # Pad the chunk output back and reshape
#             chunk_output = pad_input(
#                 chunk_output_unpad, indices_q, bsz, chunk_length
#             ).view(
#                 bsz, chunk_length, self.num_heads, self.q_head_dim
#             )

#             # Store the output for this chunk
#             attn_output[:, chunk_start:chunk_end, :, :] = chunk_output

#     # Extract only the needed value head dimensions
#     attn_output = attn_output[:, :, :, :self.v_head_dim]

#     # Reshape to final output shape
#     attn_output = attn_output.reshape(
#         bsz, seq_len, self.num_heads * self.v_head_dim
#     ).contiguous()

#     # Final projection
#     attn_output = self.o_proj(attn_output)

#     return attn_output, compressed_kv


@torch.inference_mode()
def mla_chunked_prefill_flashattention2(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    chunk_size: int = 1024,
):
    bsz, seq_len, _ = hidden_states.shape

    # Process q projection
    query_states = self.q_b_proj(
        self.q_a_layernorm(self.q_a_proj(hidden_states))
    )
    query_states = query_states.view(
        bsz, seq_len, self.num_heads, self.q_head_dim
    ).transpose(1, 2)
    q_nope, q_pe = torch.split(
        query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    )

    # Process kv projection
    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    compressed_kv_ref, k_pe = torch.split(
        compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )

    k_pe = k_pe.view(bsz, seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)
    kv = (
        self.kv_b_proj(self.kv_a_layernorm(compressed_kv_ref))
        .view(
            bsz,
            seq_len,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        .transpose(1, 2)
    )
    k_nope, value_states = torch.split(
        kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
    )

    # Apply rotary position embeddings
    cos, sin = self.rotary_emb(value_states, seq_len=seq_len)
    q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

    # Combine nope and pe components for query and key states
    query_states = k_pe.new_empty(bsz, self.num_heads, seq_len, self.q_head_dim)
    query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
    query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

    key_states = k_pe.new_empty(bsz, self.num_heads, seq_len, self.q_head_dim)
    key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
    key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

    # Pad value states to match key head dimension as required by FlashAttention
    value_states = torch.cat(
        [
            value_states,
            torch.zeros(
                (
                    bsz,
                    self.num_heads,
                    seq_len,
                    self.q_head_dim - self.v_head_dim,
                ),
                dtype=value_states.dtype,
                device=value_states.device,
            ),
        ],
        dim=-1,
    )

    # Transpose to match FlashAttention expected format
    query_states = query_states.transpose(
        1, 2
    ).contiguous()  # [bsz, seq_len, num_heads, head_dim]
    key_states = key_states.transpose(1, 2).contiguous()
    value_states = value_states.transpose(1, 2).contiguous()

    # Initialize output tensor to store chunked results
    attn_output = torch.zeros(
        bsz,
        seq_len,
        self.num_heads,
        self.v_head_dim,
        dtype=query_states.dtype,
        device=query_states.device,
    )

    # Process in chunks if chunk_size is specified and smaller than sequence length
    if chunk_size is None or seq_len <= chunk_size:
        # Process entire sequence at once with FlashAttention
        (
            query_states_unpad,
            key_states_unpad,
            value_states_unpad,
            indices_q,
            cu_seq_lens,
            max_seq_lens,
        ) = _upad_input(
            query_states, key_states, value_states, attention_mask, seq_len
        )

        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        attn_output_unpad = flash_attn_varlen_func(
            query_states_unpad,
            key_states_unpad,
            value_states_unpad,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            softmax_scale=self.softmax_scale,
            causal=True,
        )

        attn_output = pad_input(
            attn_output_unpad, indices_q, bsz, seq_len
        ).view(bsz, seq_len, self.num_heads, self.q_head_dim)

    else:
        # Process in chunks
        num_chunks = (
            seq_len + chunk_size - 1
        ) // chunk_size  # Ceiling division

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk_length = chunk_end - chunk_start

            # Extract query chunk
            query_chunk = query_states[:, chunk_start:chunk_end, :, :]

            # Critical change: For causal attention, each query chunk can only attend to
            # keys up to the current chunk end position
            key_limit = chunk_end  # This ensures causality is preserved
            key_chunk = key_states[:, :key_limit, :, :]
            value_chunk = value_states[:, :key_limit, :, :]

            # Create attention mask for this chunk
            # For chunked attention with causality, the mask needs to be properly adjusted
            chunk_attention_mask = None
            if attention_mask is not None:
                # Extract relevant portion of attention mask for current query chunk
                # and all previous + current keys
                chunk_attention_mask = attention_mask[
                    :, :, chunk_start:chunk_end, :key_limit
                ]

            # Process this chunk with FlashAttention
            (
                query_chunk_unpad,
                key_chunk_unpad,
                value_chunk_unpad,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = _upad_input(
                query_chunk,
                key_chunk,
                value_chunk,
                chunk_attention_mask,
                chunk_length,
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            chunk_output_unpad = flash_attn_varlen_func(
                query_chunk_unpad,
                key_chunk_unpad,
                value_chunk_unpad,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                softmax_scale=self.softmax_scale,
                causal=True,
            )

            # Pad the chunk output back and reshape
            chunk_output = pad_input(
                chunk_output_unpad, indices_q, bsz, chunk_length
            ).view(bsz, chunk_length, self.num_heads, self.q_head_dim)

            # Store the output for this chunk
            attn_output[:, chunk_start:chunk_end, :, :] = chunk_output

    # Extract only the needed value head dimensions
    attn_output = attn_output[:, :, :, : self.v_head_dim]

    # Reshape to final output shape
    attn_output = attn_output.reshape(
        bsz, seq_len, self.num_heads * self.v_head_dim
    ).contiguous()

    # Final projection
    attn_output = self.o_proj(attn_output)

    return attn_output, compressed_kv
