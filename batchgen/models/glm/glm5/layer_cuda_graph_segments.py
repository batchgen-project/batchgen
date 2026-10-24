"""CUDA-graph capturable GLM-5 full decoder-layer segments."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from batchgen.cuda_graph.graph_manager import TensorSpec


def make_glm5_layer_graph_segment_name(layer_idx: int) -> str:
    return f"glm5_layer_{int(layer_idx)}_full_layer"


class Glm5DecoderLayerGraphSegment:
    """Graph-capturable GLM-5 decoder layer.

    The segment composes the graph-safe full-DSA and full-MoE implementations
    directly. It must not call a nested CUDA graph replay from inside capture.
    """

    def __init__(
        self,
        *,
        layer,
        dsa_segment,
        moe_segment=None,
        device: torch.device,
        world_size: int,
        capture_local_bsz: Optional[int] = None,
        capture_rank_token_counts: Optional[torch.Tensor] = None,
    ) -> None:
        self.layer = layer
        self.dsa_segment = dsa_segment
        self.moe_segment = moe_segment
        self.device = device
        self.world_size = int(world_size)
        self.layer_idx = int(getattr(layer, "layer_idx", -1))
        self.hidden_size = int(getattr(layer, "hidden_size"))
        self.max_seqlen = int(getattr(dsa_segment, "max_seqlen"))
        self.index_topk = int(getattr(dsa_segment, "index_topk"))
        self.primary_kv_dim = int(dsa_segment.primary_blocked_k.shape[3])
        # Reuse-topk segments (GLM-5.2 skip layers) have no aux/indexer KV;
        # aux_kv_dim=0 marks the layer as producing no indexer_k_tensor.
        # A page-split FP8 aux cache carries BYTES per token in its trailing
        # dim (data + fp32 scale), so the dim and dtype of the tensor the layer
        # hands out come from the DSA segment, which resolved the aux format
        # when it was built. A segment that declares neither is AoS, and the
        # cache's trailing dim is then the element count.
        _aux_blocked_k = getattr(dsa_segment, "aux_blocked_k", None)
        if _aux_blocked_k is None:
            self.aux_kv_dim = 0
            self.aux_kv_dtype = torch.bfloat16
            self.aux_is_page_split = False
        else:
            self.aux_is_page_split = bool(
                getattr(dsa_segment, "aux_is_page_split", False)
            )
            self.aux_kv_dim = int(
                getattr(dsa_segment, "aux_host_kv_dim", _aux_blocked_k.shape[3])
            )
            self.aux_kv_dtype = getattr(
                dsa_segment, "aux_host_kv_dtype", torch.bfloat16
            )
        self.capture_local_bsz: Optional[int] = None
        self.capture_rank_token_counts: Optional[torch.Tensor] = None
        self.set_capture_context(
            local_bsz=capture_local_bsz,
            rank_token_counts=capture_rank_token_counts,
        )

    def set_capture_context(
        self,
        *,
        local_bsz: Optional[int],
        rank_token_counts: Optional[torch.Tensor],
    ) -> None:
        self.capture_local_bsz = None if local_bsz is None else max(0, int(local_bsz))
        if rank_token_counts is None:
            self.capture_rank_token_counts = None
            return
        if rank_token_counts.numel() != self.world_size:
            raise ValueError(
                f"rank_token_counts must have {self.world_size} elements, "
                f"got {rank_token_counts.numel()}"
            )
        self.capture_rank_token_counts = (
            rank_token_counts.detach()
            .to(device=self.device, dtype=torch.int64)
            .clone()
        )

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "hidden_states": TensorSpec(
                ("batch_size", 1, self.hidden_size),
                torch.bfloat16,
            ),
            "position_ids": TensorSpec(
                ("batch_size", 1),
                torch.int64,
                fill_value=float(self.max_seqlen - 1),
            ),
            "cache_seqlens": TensorSpec(
                ("batch_size",),
                torch.int32,
                fill_value=float(self.max_seqlen),
            ),
            "primary_slot_indices": TensorSpec(("batch_size",), torch.int32, fill_value=-1),
            "aux_slot_indices": TensorSpec(("batch_size",), torch.int32, fill_value=-1),
            "num_valid_tokens": TensorSpec((1,), torch.int32, fill_value=float(bucket_size)),
            "rank_token_counts": TensorSpec((self.world_size,), torch.int64, fill_value=0),
        }

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        specs = {
            "hidden_states": TensorSpec(
                ("batch_size", 1, self.hidden_size),
                torch.bfloat16,
            ),
            "primary_k_tensor": TensorSpec(
                ("batch_size", 1, 1, self.primary_kv_dim),
                torch.bfloat16,
            ),
        }
        if self.aux_kv_dim > 0:
            specs["indexer_k_tensor"] = TensorSpec(
                ("batch_size", 1, 1, self.aux_kv_dim),
                self.aux_kv_dtype,
            )
            if self.aux_is_page_split:
                # The host region takes the quantized keys with their fp32
                # per-token scales, so the layer exposes both.
                specs["indexer_k_scale"] = TensorSpec(
                    ("batch_size",), torch.float32,
                )
        return specs

    def setup_static_buffers(self, bucket_size: int) -> None:
        self.dsa_segment.setup_static_buffers(bucket_size)
        if self.moe_segment is not None:
            self.moe_segment.setup_static_buffers(bucket_size)

    def initialize_static_inputs(
        self,
        static_inputs: Dict[str, torch.Tensor],
        bucket_size: int,
    ) -> None:
        self.dsa_segment.initialize_static_inputs(static_inputs, bucket_size)
        # Empty ranks still have to capture/replay the MoE collectives, but they
        # must not run DSA kernels against a dummy slot 0 page-table row.
        if self.capture_local_bsz is not None and self.capture_local_bsz <= 0:
            static_inputs["num_valid_tokens"].zero_()
        if self.capture_rank_token_counts is not None:
            static_inputs["rank_token_counts"].copy_(
                self.capture_rank_token_counts,
                non_blocking=True,
            )
        else:
            static_inputs["rank_token_counts"].fill_(1)

    def row_validity_prologue_key(self, bucket_size: int) -> tuple:
        """Delegate: identity of the DSA buffer set the prologue writes."""
        return self.dsa_segment.row_validity_prologue_key(bucket_size)

    def run_row_validity_prologue(
        self,
        bucket_size: int,
        *,
        cache_seqlens: torch.Tensor,
        primary_slot_indices: torch.Tensor,
        aux_slot_indices: torch.Tensor,
        position_ids: torch.Tensor,
        num_valid_tokens: Optional[torch.Tensor] = None,
    ) -> None:
        """Delegate: run the hoisted step-level row-validity prologue once."""
        self.dsa_segment.run_row_validity_prologue(
            bucket_size,
            cache_seqlens=cache_seqlens,
            primary_slot_indices=primary_slot_indices,
            aux_slot_indices=aux_slot_indices,
            position_ids=position_ids,
            num_valid_tokens=num_valid_tokens,
        )

    def paged_mqa_prep_key(self, bucket_size: int) -> Optional[tuple]:
        """Delegate: identity of the step-level paged-MQA inputs, or None."""
        return self.dsa_segment.paged_mqa_prep_key(bucket_size)

    def run_paged_mqa_prep(
        self,
        bucket_size: int,
        *,
        num_valid_tokens: Optional[torch.Tensor] = None,
    ) -> None:
        """Delegate: run the hoisted step-level paged-MQA preparation once."""
        self.dsa_segment.run_paged_mqa_prep(
            bucket_size,
            num_valid_tokens=num_valid_tokens,
        )

    @property
    def selects_top_k(self) -> bool:
        """Whether this layer runs the indexer and writes the shared top-k."""
        return bool(self.dsa_segment.SELECTS_TOP_K)

    @property
    def WRITES_AUX_HOST_KV(self) -> bool:
        """Delegate: does the inner attention segment produce aux host-KV rows?

        Reflected rather than redeclared: the answer is a property of the
        attention segment this layer composes (True for a full-DSA indexer
        layer, False for a reuse-topk skip layer), and the whole-model driver
        reads it off the LAYER segment to decide whether to hand over aux
        staging slices. A hardcoded value here would silently disagree with
        the inner segment on one of the two layer kinds.
        """
        return bool(getattr(self.dsa_segment, "WRITES_AUX_HOST_KV", False))

    def set_host_kv_staging(
        self,
        *,
        key: Optional[torch.Tensor],
        aux_key: Optional[torch.Tensor] = None,
        aux_scale: Optional[torch.Tensor] = None,
    ) -> None:
        """Delegate: bind the inner segment's host-append outputs to staging.

        The whole-model segment composes LAYER segments, so it installs decode
        host-KV staging here, but the buffers belong to the attention segment's
        producing kernels. ``release_static_buffers`` deliberately does not
        undo this: the whole-model release loop calls this method with
        ``key=None`` itself, so uninstalling here too would be a second,
        redundant clear.
        """

        install = getattr(self.dsa_segment, "set_host_kv_staging", None)
        if install is None:
            raise RuntimeError(
                "GLM-5 whole-model decode host-KV staging binds each layer's "
                "host-append outputs to its staging slice, so layer "
                f"{self.layer_idx}'s attention segment "
                f"({type(self.dsa_segment).__name__}) must expose "
                "set_host_kv_staging"
            )
        install(key=key, aux_key=aux_key, aux_scale=aux_scale)

    def selected_positions_key(self, bucket_size: int) -> Optional[tuple]:
        """Delegate: identity of the selected-position transform's tensors.

        ``None`` for a layer that selects its own top-k, whose transform is
        never shareable with another layer's.
        """
        key_fn = getattr(self.dsa_segment, "selected_positions_key", None)
        if key_fn is None:
            return None
        return key_fn(bucket_size)

    def release_static_buffers(self, bucket_size: int) -> None:
        release = getattr(self.dsa_segment, "release_static_buffers", None)
        if release is not None:
            release(bucket_size)
        if self.moe_segment is not None:
            release = getattr(self.moe_segment, "release_static_buffers", None)
            if release is not None:
                release(bucket_size)

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cache_seqlens: torch.Tensor,
        primary_slot_indices: torch.Tensor,
        aux_slot_indices: torch.Tensor,
        num_valid_tokens: torch.Tensor,
        rank_token_counts: torch.Tensor,
        row_validity_ready: bool = False,
        paged_mqa_prep_ready: bool = False,
        selected_positions_ready: bool = False,
        residual: Optional[torch.Tensor] = None,
        defer_residual_add: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run one decoder layer.

        ``residual`` / ``defer_residual_add`` let a driver chain the layers in
        the fused-add form. With ``defer_residual_add`` the layer does NOT
        materialize ``residual + mlp_output``; it returns the two halves and the
        NEXT layer's ``input_layernorm`` becomes the fused add+RMSNorm that was
        going to run anyway, so the standalone add disappears from the graph.
        ``residual`` is the previous layer's carried residual and is ``None``
        only for the first layer, whose input comes straight from the
        embedding. The driver owes the final add after the last layer.

        Mixing the two conventions mid-chain would drop a residual, so
        ``residual`` without ``defer_residual_add`` is a hard error rather than
        a silent fallback.
        """
        from batchgen.attention.fused_kernels import cuda_add_rmsnorm

        if residual is not None and not defer_residual_add:
            raise ValueError(
                "Glm5DecoderLayerGraphSegment.forward: residual was carried in "
                "but defer_residual_add is False, which would materialize this "
                "layer's add while the previous layer's add is still pending; "
                "a driver must use one convention for the whole layer chain"
            )

        batch_size = int(hidden_states.shape[0])

        if residual is None:
            # First layer of the chain: its input IS the residual stream.
            residual = hidden_states
            attn_input = self.layer.input_layernorm(hidden_states)
        else:
            # The previous layer's deferred `residual + mlp_output` happens
            # HERE, inside the input-norm launch. `hidden_states` is that
            # layer's mlp_output.
            attn_input, residual = cuda_add_rmsnorm(
                residual,
                hidden_states,
                self.layer.input_layernorm.weight,
                self.layer.input_layernorm.eps,
            )

        dsa_out = self.dsa_segment.forward(
            hidden_states=attn_input,
            position_ids=position_ids,
            cache_seqlens=cache_seqlens,
            primary_slot_indices=primary_slot_indices,
            aux_slot_indices=aux_slot_indices,
            num_valid_tokens=num_valid_tokens,
            row_validity_ready=row_validity_ready,
            # Both say "the driver already recorded this step-level work":
            # the shared paged-MQA preparation, and (for a layer that reuses
            # another's top-k) the selected-page-table transform the first
            # layer of this reuse run already ran.
            paged_mqa_prep_ready=paged_mqa_prep_ready,
            selected_positions_ready=selected_positions_ready,
            # This layer absorbs the segment's `attn_output` row mask into the
            # add below, so the segment must not spend a launch on it.
            attn_output_mask_deferred=True,
        )
        hidden_states, residual = cuda_add_rmsnorm(
            residual,
            dsa_out["attn_output"],
            self.layer.post_attention_layernorm.weight,
            self.layer.post_attention_layernorm.eps,
            row_valid_mask=self.dsa_segment.attn_output_row_mask(batch_size),
        )

        if self.moe_segment is None:
            mlp_output = self.layer.mlp(hidden_states)
        else:
            flat_hidden = hidden_states.view(batch_size, self.hidden_size)
            moe_output = self.moe_segment.forward(
                padded=flat_hidden,
                rank_token_counts=rank_token_counts,
            )["moe_output"]
            mlp_output = moe_output.view(batch_size, 1, self.hidden_size)

        if defer_residual_add:
            # `mlp_output` may be the MoE pool's shared static buffer, which
            # the NEXT layer's MoE block overwrites. That is safe only because
            # the next layer's add+norm READS it to produce the input its own
            # MoE block consumes, so the overwrite is downstream of the read by
            # a true data dependency, not by launch-order luck.
            out = {
                "hidden_states": mlp_output,
                "residual": residual,
                "primary_k_tensor": dsa_out["primary_k_tensor"],
            }
        else:
            out = {
                "hidden_states": residual + mlp_output,
                "primary_k_tensor": dsa_out["primary_k_tensor"],
            }
        if "indexer_k_tensor" in dsa_out:
            out["indexer_k_tensor"] = dsa_out["indexer_k_tensor"]
        if "indexer_k_scale" in dsa_out:
            out["indexer_k_scale"] = dsa_out["indexer_k_scale"]
        return out
