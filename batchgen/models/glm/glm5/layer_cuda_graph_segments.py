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
        self.aux_kv_dim = int(dsa_segment.aux_blocked_k.shape[3])
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

    def _flashmla_tensor_metadata_specs(
        self,
        bucket_size: int,
    ) -> tuple[tuple[int, ...], torch.dtype, tuple[int, ...], torch.dtype]:
        return self.dsa_segment._flashmla_tensor_metadata_specs(bucket_size)

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        tile_shape, tile_dtype, num_splits_shape, num_splits_dtype = (
            self._flashmla_tensor_metadata_specs(bucket_size)
        )
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
            "flashmla_tile_scheduler_metadata": TensorSpec(tile_shape, tile_dtype),
            "flashmla_num_splits": TensorSpec(num_splits_shape, num_splits_dtype),
        }

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "hidden_states": TensorSpec(
                ("batch_size", 1, self.hidden_size),
                torch.bfloat16,
            ),
            "primary_k_tensor": TensorSpec(
                ("batch_size", 1, 1, self.primary_kv_dim),
                torch.bfloat16,
            ),
            "indexer_k_tensor": TensorSpec(
                ("batch_size", 1, 1, self.aux_kv_dim),
                torch.bfloat16,
            ),
        }

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

    def release_static_buffers(self, bucket_size: int) -> None:
        release = getattr(self.dsa_segment, "release_static_buffers", None)
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
        flashmla_tile_scheduler_metadata: torch.Tensor,
        flashmla_num_splits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        from batchgen.attention.fused_kernels import cuda_add_rmsnorm

        residual = hidden_states
        attn_input = self.layer.input_layernorm(hidden_states)
        dsa_out = self.dsa_segment.forward(
            hidden_states=attn_input,
            position_ids=position_ids,
            cache_seqlens=cache_seqlens,
            primary_slot_indices=primary_slot_indices,
            aux_slot_indices=aux_slot_indices,
            num_valid_tokens=num_valid_tokens,
            flashmla_tile_scheduler_metadata=flashmla_tile_scheduler_metadata,
            flashmla_num_splits=flashmla_num_splits,
        )
        hidden_states, residual = cuda_add_rmsnorm(
            residual,
            dsa_out["attn_output"],
            self.layer.post_attention_layernorm.weight,
            self.layer.post_attention_layernorm.eps,
        )

        if self.moe_segment is None:
            mlp_output = self.layer.mlp(hidden_states)
        else:
            batch_size = int(hidden_states.shape[0])
            flat_hidden = hidden_states.view(batch_size, self.hidden_size)
            moe_output = self.moe_segment.forward(
                padded=flat_hidden,
                rank_token_counts=rank_token_counts,
            )["moe_output"]
            mlp_output = moe_output.view(batch_size, 1, self.hidden_size)

        return {
            "hidden_states": residual + mlp_output,
            "primary_k_tensor": dsa_out["primary_k_tensor"],
            "indexer_k_tensor": dsa_out["indexer_k_tensor"],
        }
