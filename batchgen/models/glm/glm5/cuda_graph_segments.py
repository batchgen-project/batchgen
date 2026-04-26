"""CUDA Graph capturable segments for GLM-5 decode.

Segments (Phase 2a of the whole-mode CUDA graph plan — per-layer DSA):
  Glm5DsaAttnSegment:
    Input layernorm → DSA attention (indexer + sparse gather + sparse FlashMLA
    + O_proj) → post-attn residual-add + RMSNorm. Captured per layer with two
    variants (dsa_short, dsa_long). Mixed DSA batches fall back to eager.

Design mirrors K25AttnSegment (moonshotai/kimi_k25/cuda_graph_segments.py):
  - `page_table` / `slot_indices` passed as static-address input tensors;
    never read from the live `gpu_paged_kv_manager.gpu_table` (which is
    dynamically reallocated when page counts change).
  - Weights / RoPE / layer-norm parameters held as static GPU tensors via
    closures captured at segment init.
  - KV writes use `run_paged_kv_token_update_fused` directly, bypassing
    `GPUPagedKVCacheManager.update_layer_decode_new_token` (not capture-safe).

SCOPE OF THIS FILE IN PHASE 2a-i: class scaffolding + I/O contract only.
`forward()` raises NotImplementedError pending the capture-safe refactor of
`GLM5AttnWrapper._forward_decode_dsa` (tracked as Phase 2a-ii). The forward
body below is annotated with step-by-step references to `wrappers.py` lines
so the follow-up implementation is a straight port — no re-planning needed.
"""
from __future__ import annotations

import logging
from typing import Dict, Literal

import torch

from batchgen.cuda_graph.graph_manager import TensorSpec
from batchgen.models.wrappers.attention import AttnWrapperBase

logger = logging.getLogger(__name__)

DsaVariant = Literal["short", "long"]


class Glm5DsaAttnSegment:
    """GLM-5 DSA attention block as a single CUDA-graph-capturable segment.

    One instance per (layer_idx, variant) pair. Register 2 × num_layers
    segments total (`layer_{i}_dsa_short` and `layer_{i}_dsa_long`).

    The caller's dispatcher (outside the graph, plan §5) reads
    `AttnWrapperBase._dsa_short_count` on the host and selects the variant
    before calling `manager.replay(...)`.

    Inputs:
      hidden_states         [B, 1, hidden_size]   bf16
      cache_seqlens         [B]                    int32 (post-increment)
      position_ids          [B, 1]                 int64
      primary_page_table    [B, max_pages]         int32 — primary MLA KV pages
      aux_page_table        [B, max_aux_pages]     int32 — DSA indexer KV pages
      primary_slot_indices  [B]                    int32
      aux_slot_indices      [B]                    int32

    Outputs:
      normed                [B, 1, hidden_size]    bf16 — MoE input (post-attn RMSNorm)
      residual              [B, 1, hidden_size]    bf16 — residual carry for post-MoE sum
      k_tensor              [B, 1, 1, kv_dim]      bf16 — for kv_append_callback
      indexer_k_tensor      [B, 1, 1, index_dim]   bf16 — for kv_append_callback_aux
    """

    def __init__(
        self,
        decoder_layer,                  # Glm5DecoderLayer
        attn_wrapper,                   # GLM5AttnWrapper
        layer_idx: int,
        variant: DsaVariant,
        max_seq_len: int,               # for static RoPE cache selection
        max_pages_per_seq: int,
        max_aux_pages_per_seq: int,
        page_size_tokens: int,
        aux_page_size_tokens: int,
    ) -> None:
        assert variant in ("short", "long"), f"unknown variant {variant!r}"
        self.decoder_layer = decoder_layer
        self.attn_wrapper = attn_wrapper
        self.attn_mod = attn_wrapper.module
        self.indexer = self.attn_mod.indexer
        self.layer_idx = layer_idx
        self.variant = variant
        self.max_seq_len = max_seq_len
        self.max_pages_per_seq = max_pages_per_seq
        self.max_aux_pages_per_seq = max_aux_pages_per_seq
        self.page_size_tokens = page_size_tokens
        self.aux_page_size_tokens = aux_page_size_tokens

        # Dimensions ---------------------------------------------------------
        attn = self.attn_mod
        self.hidden_size = attn.hidden_size
        self.num_heads = attn.num_heads                # 64 for GLM-5
        self.q_lora_rank = attn.q_lora_rank            # 2048
        self.kv_lora_rank = attn.kv_lora_rank          # 512
        self.qk_nope_head_dim = attn.qk_nope_head_dim  # 192
        self.qk_rope_head_dim = attn.qk_rope_head_dim  # 64
        self.v_head_dim = attn.v_head_dim              # 256
        self.q_head_dim = attn.q_head_dim              # 256
        self.kv_dim = self.kv_lora_rank + self.qk_rope_head_dim  # 576
        self.index_dim = self.indexer.index_head_dim   # 128
        self.index_topk = self.indexer.index_topk      # 2048

        # Pre-attn / post-attn layer norms (static weights) ------------------
        self.input_ln_weight = decoder_layer.input_layernorm.weight
        self.input_ln_eps = decoder_layer.input_layernorm.variance_epsilon
        self.post_ln_weight = decoder_layer.post_attention_layernorm.weight
        self.post_ln_eps = decoder_layer.post_attention_layernorm.variance_epsilon

    # -------------------------------------------------------------------- specs

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "hidden_states": TensorSpec(
                ("batch_size", 1, self.hidden_size), torch.bfloat16
            ),
            "cache_seqlens": TensorSpec(
                ("batch_size",), torch.int32, fill_value=1
            ),
            "position_ids": TensorSpec(
                ("batch_size", 1), torch.int64, fill_value=0
            ),
            "primary_page_table": TensorSpec(
                ("batch_size", self.max_pages_per_seq), torch.int32, fill_value=0
            ),
            "aux_page_table": TensorSpec(
                ("batch_size", self.max_aux_pages_per_seq), torch.int32, fill_value=0
            ),
            "primary_slot_indices": TensorSpec(
                ("batch_size",), torch.int32, fill_value=0
            ),
            "aux_slot_indices": TensorSpec(
                ("batch_size",), torch.int32, fill_value=0
            ),
        }

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "normed": TensorSpec(
                ("batch_size", 1, self.hidden_size), torch.bfloat16
            ),
            "residual": TensorSpec(
                ("batch_size", 1, self.hidden_size), torch.bfloat16
            ),
            "k_tensor": TensorSpec(
                ("batch_size", 1, 1, self.kv_dim), torch.bfloat16
            ),
            "indexer_k_tensor": TensorSpec(
                ("batch_size", 1, 1, self.index_dim), torch.bfloat16
            ),
        }

    # ----------------------------------------------------------------- forward

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_seqlens: torch.Tensor,
        position_ids: torch.Tensor,
        primary_page_table: torch.Tensor,
        aux_page_table: torch.Tensor,
        primary_slot_indices: torch.Tensor,
        aux_slot_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Capture-safe DSA decode for one layer.

        Phase 2a-i status: NOT YET IMPLEMENTED — raises NotImplementedError.
        The step-by-step implementation plan below maps to the existing eager
        path at `batchgen/models/glm/glm5/wrappers.py::_forward_decode_dsa`.

        Implementation steps (matches wrappers.py _forward_decode_dsa):

        1. Pre-attn RMSNorm and residual capture.
           - residual = hidden_states
           - normed   = Glm5RMSNorm(input_ln_weight, input_ln_eps)(hidden_states)

        2. Shared FP8 act quant of normed (wrappers.py:617-619).
           - hidden_flat            = normed.squeeze(1)
           - hidden_fp8, hidden_sc  = act_quant(hidden_flat)

        3. Q projection chain (wrappers.py:622-637).
           - q_a  = w8a8_deepgemm(hidden_fp8, hidden_sc, q_a_proj.weight, ws["q_a"])
           - q_an = q_a_layernorm(q_a)
           - q_af, q_asc = act_quant(q_an)
           - q    = w8a8_deepgemm(q_af, q_asc, q_b_proj.weight, ws["q_b"])
           - q    = reshape + transpose + split(q_nope, q_pe)

        4. KV projection + fused RMSNorm-RoPE (wrappers.py:640-659).
           - new_compressed_kv = w8a8_deepgemm(... kv_a_proj_with_mqa ...)
           - cos, sin = rotary_emb(q_pe, seq_len=self.max_seq_len)  # STATIC
           - offload_kv = fused_rmsnorm_rope_with_q_native(
                 new_compressed_kv, q_pe, cos, sin, position_ids,
                 kv_a_layernorm.weight, kv_lora_rank, qk_rope_head_dim,
                 eps=kv_a_layernorm.eps,
             )
           - k_tensor = offload_kv.view(B, 1, 1, kv_dim)        # graph output

        5. PRIMARY KV write via run_paged_kv_token_update_fused
           (replaces update_layer_decode_new_token from wrappers.py:744-750).
           Refactor required in Phase 2a-ii: update_layer_decode_new_token
           reads the live gpu_paged_kv_manager.gpu_table, which reallocates —
           not graph-safe. Use the static primary_page_table input instead.
               blocked_k, _, _ = gpu_paged_kv_manager.get_layer_kv_with_page_table(li)
               run_paged_kv_token_update_fused(
                   k_cache=blocked_k,
                   k_tokens=k_tensor.view(B, -1),
                   page_table=primary_page_table,
                   slot_indices=primary_slot_indices,
                   token_indices=position_ids.squeeze(-1).to(torch.int32),
                   page_size_tokens=self.page_size_tokens,
               )

        6. AUX indexer K computation + write (wrappers.py:754-794).
           - indexer_kv = _fused_rope_hadamard(
                 indexer.k_norm(cuda_wk_proj_gemm_only(hidden_flat, ...)),
                 position_ids, max_seqlen=self.max_seq_len,
             )
           - indexer_k_tensor = indexer_kv  # [B, 1, 1, index_dim]
           - (Same static-page-table treatment as step 5, using
              aux_page_table / aux_slot_indices / aux_page_size_tokens.)

        7. VARIANT BRANCH — top-k index selection:
           if self.variant == "short":
             top_k_indices = build_clamped_dense_token_indices(
                 cache_seqlens, self.max_seq_len, hidden_states.device,
             )
             # NOTE: max_seqlen arg uses self.max_seq_len (static) instead of
             # per-step max(cache_seqlens). build_clamped_dense_token_indices
             # internally clamps to the actual cache length per row, so this
             # is semantically equivalent.
           else:  # "long"
             indexer_blocked_k, _, idx_block_table = \
                 gpu_paged_kv_manager_aux.get_layer_kv_with_page_table(li)
             idx_block_table = reorder_block_table_to_batch_slots(
                 idx_block_table, aux_slot_indices,
             )
             top_k_indices = indexer.score_and_select_paged(
                 q_a_normed.unsqueeze(1), hidden_states,
                 indexer_blocked_k, idx_block_table,
                 cache_seqlens, gpu_paged_kv_manager_aux, self.aux_page_size_tokens,
                 positions=position_ids.squeeze(-1),
                 max_seqlen=self.max_seq_len,  # STATIC
             )
           # "mixed" variant is captured as EAGER FALLBACK by the scheduler
           # (see wrappers.py:851 assertion); not represented here.

        8. Sparse gather MLA KV (wrappers.py:897-943).
           - mla_blocked_k, _, mla_block_table = primary KV + page table
           - mla_block_table = reorder_block_table_to_batch_slots(
                 mla_block_table, primary_slot_indices,
             )
           - sparse_mla_kv = sparse_gather_from_paged_kv(
                 mla_blocked_k, mla_block_table, top_k_indices,
                 self.page_size_tokens,
             )

        9. Absorb Q + sparse FlashMLA (wrappers.py:946-1005).
           - Use cached attn_wrapper._cached_q_absorb / _cached_out_absorb
             or the BF16 BMM fallback (wrappers.py:971-993). FP8 absorb
             path (self._fp8_absorb_weights) is default-disabled per
             project memory — use BMM in initial implementation.
           - query_states = concat(q_nope_absorbed, q_pe) via empty+index assign
           - sparse_seqlens = clamp(cache_seqlens, max=top_k_indices.shape[1])
           - attn_out = sparse_flash_mla_decode(
                 query_states, sparse_mla_kv, sparse_seqlens,
                 num_heads, softmax_scale, head_dim_v=kv_lora_rank,
                 page_size=self.page_size_tokens,
             )

        10. O_proj with out-absorb (wrappers.py:1008-1024).
            - attn_output = bmm(attn_out.T, self.w_vc).T  # BMM fallback
            - attn_output = reshape to [B, num_heads * v_head_dim]
            - attn_output_fp8, attn_output_scale = act_quant(attn_output)
            - attn_output = w8a8_deepgemm(... o_proj ...)
            - attn_output = attn_output.view(B, 1, -1)

        11. Post-attn fused add + RMSNorm (model.py:1919-1924).
            - normed_out, residual_out = cuda_add_rmsnorm(
                  residual, attn_output,
                  self.post_ln_weight, self.post_ln_eps,
              )

        Return: {"normed": normed_out, "residual": residual_out,
                 "k_tensor": k_tensor, "indexer_k_tensor": indexer_k_tensor}
        """
        raise NotImplementedError(
            f"Glm5DsaAttnSegment.forward (variant={self.variant!r}) is Phase "
            "2a scaffolding — implementation pending the capture-safe refactor "
            "of _forward_decode_dsa (Phase 2a-ii). See this method's docstring "
            "for the step-by-step porting plan."
        )


# ============================================================================
# Phase 2b: per-MoE-layer CUDA graph segment
# ============================================================================


class Glm5MoeSegment:
    """GLM-5 MoE layer as a CUDA-graph-capturable segment (bucket-agnostic).

    One instance per MoE layer. The CUDAGraphManager captures it once per
    registered bucket; `setup_static_buffers(bucket_size)` (called by the
    manager before each capture) aligns `moe.num_tokens_per_rank` with the
    bucket so `_forward_decode_3d` reads consistent state.

    At world_size=16, Phase 2b targets buckets matching
    `ntp in {1, 2, 4, 8, 16, 32}` (= `mtp in {16, 32, 64, 128, 256, 512}` when
    fed to `BATCHGEN_GLM5_MTP_BUCKETS`). Above bucket_ntp=32 the scheduler
    falls back to eager.

    Input / output
    --------------
    Inputs:
      hidden_states  [bucket_ntp, 1, hidden_size] bf16 — post-attn-norm MoE
                      input. Scheduler pads rows `actual_ntp..bucket_ntp`
                      with zeros (zeros propagate as zeros through gate +
                      FP8 GEMMs and scatter back as zero contributions).
    Outputs:
      moe_output     [bucket_ntp, 1, hidden_size] bf16

    Scheduler contract at replay time
    ---------------------------------
    Before `graph.replay(f"layer_{i}_moe", bucket_ntp, hidden_states=H)`:

    (1) `H.shape[0] == bucket_ntp` — pad actual input with zeros.
    (2) `moe.num_tokens_per_rank == bucket_ntp` on THIS layer's MoE
        instance — the segment's `setup_static_buffers` does this at
        capture time; the scheduler must re-apply before each replay
        (typically in the decode loop's pre-replay hook).

    Padding correctness note
    ------------------------
    `Glm5MoE._rank_token_counts` is not touched by this segment. Zero-
    padded input rows produce zero output after gate + FP8 GEMM + scatter,
    so slicing `[:actual_ntp]` post-replay recovers the correct result
    without needing the mask. Compute for padding rows is O(fill_ratio)
    wasted — acceptable at small buckets.

    Why delegation, not inlining
    ----------------------------
    `_forward_decode_3d` is already capture-safe: zero host syncs in body
    and callees, static-address bucket buffer via `_select_3d_buf`, PyNccl
    collectives via `change_state(enable=True)`. Inlining would duplicate
    ~100 lines of logic for no correctness benefit; delegation inherits
    future fixes to the eager path.
    """

    def __init__(
        self,
        decoder_layer,          # Glm5DecoderLayer
        moe_module,             # Glm5MoE instance (decoder_layer.mlp)
        layer_idx: int,
        hidden_size: int,
    ) -> None:
        self.decoder_layer = decoder_layer
        self.moe = moe_module
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size

    def setup_static_buffers(self, bucket_size: int) -> None:
        """Called by CUDAGraphManager before capture at this bucket size.

        Aligns `self.moe.num_tokens_per_rank` with the bucket so
        `_forward_decode_3d` computes `num_global = bucket * world_size`
        and the bucket selector picks the matching pre-allocated buffer.

        Bypasses `Glm5MoE.set_num_tokens_per_rank()` to avoid its side
        effects (`token_idx` / `topk_pos` realloc, `resize_if_needed` on
        the singleton) — those tensors are only used by the non-3D path
        (`_triton_compute` / `expert_compute_mixed`), never by the 3D
        decode path this segment captures.

        Also bakes "all rows valid" into the static padding-mask buffer for
        this bucket. The captured masking ops then record as effective
        no-ops at warmup; per-step replay overrides via the static buffer's
        contents (in-place .copy_() in PSM.set_rank_token_counts).
        """
        from .model import Glm5MoE
        self.moe.num_tokens_per_rank = bucket_size
        rt_counts = Glm5MoE._rank_token_counts
        if rt_counts is not None:
            rt_counts.fill_(bucket_size)

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "hidden_states": TensorSpec(
                ("batch_size", 1, self.hidden_size), torch.bfloat16
            ),
        }

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "moe_output": TensorSpec(
                ("batch_size", 1, self.hidden_size), torch.bfloat16
            ),
        }

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Capture-safe MoE decode for one layer. Delegates to
        `_forward_decode_3d`; see class docstring for the scheduler contract.
        """
        # Capture-time invariant: moe.num_tokens_per_rank must match the
        # input's bucket size. setup_static_buffers enforces this at capture
        # time; replay relies on the scheduler to re-apply per step.
        if torch.cuda.is_current_stream_capturing():
            assert self.moe.num_tokens_per_rank == hidden_states.shape[0], (
                f"Glm5MoeSegment(layer={self.layer_idx}): "
                f"moe.num_tokens_per_rank={self.moe.num_tokens_per_rank} "
                f"!= hidden_states.shape[0]={hidden_states.shape[0]} "
                "under graph capture. setup_static_buffers() must run "
                "before forward() for each bucket."
            )
        moe_output = self.moe._forward_decode_3d(hidden_states)
        return {"moe_output": moe_output}


# ============================================================================
# Phase 2b-ii: decode-time replay proxy
# ============================================================================


class Glm5MoeReplayProxy(torch.nn.Module):
    """nn.Module wrapper that replaces a `Glm5MoE` child after graph capture.

    `Glm5DecoderLayer.forward` calls `self.mlp(hidden_states)` — rewrapping
    `decoder_layer.mlp` with this proxy routes the call through the captured
    graph instead of eager `Glm5MoE.forward`, cutting per-layer kernel-launch
    overhead on top of the 2a-alpha bucket-buffer HBM savings.

    Must inherit `nn.Module` because `decoder_layer.mlp = proxy` goes through
    `nn.Module.__setattr__` which rejects anything non-Module for attributes
    registered as child modules (torch/nn/modules/module.py:2017).

    Per-step forward flow:

      1. bucket = bucketing.get_padded_size(actual_ntp)   # pure Python
      2. moe.num_tokens_per_rank = bucket                  # align for replay
      3. graph_manager.replay(
             f"layer_{layer_idx}_moe", actual_ntp,
             hidden_states=input,                          # manager zeros tail
         )
      4. return out["moe_output"][:actual_ntp]             # unpad

    Oversize / zero-batch fall back to eager `self.moe(hidden_states)`.

    `__getattr__` falls through to the wrapped Glm5MoE for any attribute the
    proxy doesn't own — preserves compatibility with callers that read
    `mlp.num_tokens_per_rank`, `mlp.comm`, etc.
    """

    def __init__(self, moe_module, layer_idx: int, graph_manager, bucketing):
        super().__init__()
        # Registered as a child nn.Module so .to(device), .eval() etc. still
        # propagate — even though the proxy's forward doesn't call moe's
        # eager forward on the happy path, keeping it as a child keeps the
        # parent decoder_layer's Module tree consistent.
        self.moe = moe_module
        self.layer_idx = layer_idx
        # Bypass nn.Module's __setattr__ for non-Module / non-Tensor attrs —
        # these are plain Python refs (not part of the Module tree) so
        # storing in __dict__ via object.__setattr__ avoids accidental
        # registration and also dodges nn.Module's __getattr__ fallback.
        object.__setattr__(self, "graph_manager", graph_manager)
        object.__setattr__(self, "bucketing", bucketing)
        object.__setattr__(self, "_segment_name", f"layer_{layer_idx}_moe")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        actual_ntp = hidden_states.shape[0]

        # Bucket selection MUST be globally consistent across all ranks.
        # Drive it from `self.moe.num_tokens_per_rank`, which the worker
        # propagates via `parallel_manager.set_num_tokens_per_rank(_max_bs)`
        # using a per-step all_gather of local batch sizes (see
        # batchgen_worker.py:9018-9028). Every rank then reads the same
        # value, so all 16 ranks pick the same bucket and replay the same
        # captured graph (matching NCCL collective payloads = no deadlock).
        #
        # Earlier bug: this code keyed on `actual_ntp = hidden_states.shape[0]`
        # which is per-rank and varies as sequences complete mid-window.
        # That caused different ranks to replay different buckets → NCCL
        # all_gather byte-count mismatch → cluster-wide hang.
        #
        # Zero-batch: pass smallest bucket, slice [:0] post-replay. Captured
        # graph handles zero real input correctly (replay zeros the tail).
        #
        # Oversize: synced_ntp > largest captured bucket. Fall back to eager
        # `_forward_decode_3d` — all 16 ranks see the same oversize and fall
        # back together. Eager path uses the singleton `_3d_buf`
        # (resize_if_needed permitted there), independent of bucket buffers.
        # Bucket selection is driven ONLY by synced_ntp (cross-rank-identical).
        # Local actual_ntp can be 0 on some ranks while peers have work; if
        # we keyed bucket on actual_ntp here, those ranks would replay a
        # different captured graph than peers → NCCL byte-count mismatch.
        # Even with actual_ntp=0, the captured graph still fires its NCCL
        # collectives with the bucket-sized payload (the static input is
        # zero-filled by graph_manager.replay's input-padding path).
        synced_ntp = int(self.moe.num_tokens_per_rank or 0)
        if synced_ntp == 0:
            bucket = min(self.bucketing.bucket_sizes)
        else:
            try:
                bucket = self.bucketing.get_padded_size(synced_ntp)
            except ValueError:
                # Oversize: log once and route to eager.
                if not getattr(Glm5MoeReplayProxy, "_warned_oversize", False):
                    logging.warning(
                        f"Glm5MoeReplayProxy(layer={self.layer_idx}): "
                        f"synced_ntp={synced_ntp} exceeds max bucket "
                        f"{max(self.bucketing.bucket_sizes)}; falling back "
                        "to eager _forward_decode_3d. Increase "
                        "BATCHGEN_GLM5_MTP_BUCKETS to capture this size."
                    )
                    Glm5MoeReplayProxy._warned_oversize = True
                return self.moe._forward_decode_3d(hidden_states)

        # Pass `bucket` (not actual_ntp) as graph_manager.replay's batch_size
        # so all 16 ranks select the SAME captured graph for this step. The
        # replay function re-derives `bucket = get_padded_size(batch_size)`
        # internally; if we pass per-rank actual_ntp, ranks with smaller
        # batches pick a smaller bucket → different captured NCCL kernels →
        # mismatched payloads → cluster-wide hang.
        # (Root cause of the 2026-04-25 stress hang.)
        replay_bs = bucket

        # Dispatch. graph_manager.replay handles bucket padding: copies
        # input[:actual_ntp] into static[:actual_ntp] and zeros the tail.
        out = self.graph_manager.replay(
            self._segment_name,
            replay_bs,
            hidden_states=hidden_states,
        )
        # Slice back to actual batch.
        return out["moe_output"][:actual_ntp]

    def __getattr__(self, name):
        # nn.Module's __getattr__ walks _parameters/_buffers/_modules.
        # If not found there, fall back to the wrapped moe so callers
        # that read moe.hidden_size / moe.config etc. still work.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.moe, name)
