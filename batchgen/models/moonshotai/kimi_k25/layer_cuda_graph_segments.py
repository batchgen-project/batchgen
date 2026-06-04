"""CUDA-graph capturable Kimi-K2.5 full decoder-layer segment.

Mirrors ``glm5/layer_cuda_graph_segments.py`` but is simpler: the K2.5
``K25AttnSegment`` already performs BOTH the input RMSNorm and the post-attn
RMSNorm internally (returning ``normed`` / ``residual`` / ``k_tensor``), so the
layer glue is just ``residual + mlp(normed)``. K2.5 is plain MLA (no DSA
indexer) → primary KV only, and attention derives position from
``cache_seqlens`` internally, so there is no position_ids / aux-slot / FlashMLA
metadata input to thread.

Layer 0 is a dense MLP (``DenseMLP``); layers 1..N-1 are MoE. A dense layer is
built with ``moe_segment=None`` and the glue runs the dense MLP inline.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from batchgen.cuda_graph.graph_manager import TensorSpec


def make_k25_layer_graph_segment_name(layer_idx: int) -> str:
    return f"k25_layer_{int(layer_idx)}_full_layer"


class K25DecoderLayerGraphSegment:
    """Graph-capturable K2.5 decoder layer (attn + MoE/dense)."""

    def __init__(
        self,
        *,
        layer,
        attn_segment,
        moe_segment=None,
        device: torch.device,
        world_size: int,
    ) -> None:
        self.layer = layer
        self.attn_segment = attn_segment
        self.moe_segment = moe_segment
        self.device = device
        self.world_size = int(world_size)
        self.layer_idx = int(getattr(layer, "layer_idx", -1))
        self.hidden_size = int(attn_segment.hidden_size)
        self.max_pages_per_seq = int(attn_segment.max_pages_per_seq)
        self.primary_kv_dim = int(attn_segment.kv_dim)
        # Dense (layer 0) runs the model's MLP module directly inside the glue.
        self._dense_mlp = layer.mlp if moe_segment is None else None

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        specs = {
            "hidden_states": TensorSpec(
                ("batch_size", 1, self.hidden_size), torch.bfloat16
            ),
            "cache_seqlens": TensorSpec(("batch_size",), torch.int32, fill_value=1),
            "page_table": TensorSpec(
                ("batch_size", self.max_pages_per_seq), torch.int32, fill_value=0
            ),
            "slot_indices": TensorSpec(("batch_size",), torch.int32, fill_value=0),
        }
        if self.moe_segment is not None:
            specs["rank_token_counts"] = TensorSpec(
                (self.world_size,), torch.int64, fill_value=0
            )
        return specs

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "hidden_states": TensorSpec(
                ("batch_size", 1, self.hidden_size), torch.bfloat16
            ),
            "primary_k_tensor": TensorSpec(
                ("batch_size", 1, 1, self.primary_kv_dim), torch.bfloat16
            ),
        }

    def setup_static_buffers(self, bucket_size: int) -> None:
        setup = getattr(self.attn_segment, "setup_static_buffers", None)
        if setup is not None:
            setup(bucket_size)
        if self.moe_segment is not None:
            setup = getattr(self.moe_segment, "setup_static_buffers", None)
            if setup is not None:
                setup(bucket_size)

    def release_static_buffers(self, bucket_size: int) -> None:
        release = getattr(self.attn_segment, "release_static_buffers", None)
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
        cache_seqlens: torch.Tensor,
        page_table: torch.Tensor,
        slot_indices: torch.Tensor,
        rank_token_counts: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # K25AttnSegment does input_ln -> MLA -> post_attn_ln, returning the
        # MoE input (normed), the post-attn residual, and the KV offload tensor.
        attn_out = self.attn_segment.forward(
            hidden_states=hidden_states,
            cache_seqlens=cache_seqlens,
            page_table=page_table,
            slot_indices=slot_indices,
        )
        normed = attn_out["normed"]
        residual = attn_out["residual"]
        k_tensor = attn_out["k_tensor"]

        if self.moe_segment is None:
            mlp_output = self._dense_mlp(normed)
        else:
            batch_size = int(normed.shape[0])
            moe_output = self.moe_segment.forward(
                padded=normed.view(batch_size, self.hidden_size),
                rank_token_counts=rank_token_counts,
            )["moe_output"]
            mlp_output = moe_output.view(batch_size, 1, self.hidden_size)

        return {
            "hidden_states": residual + mlp_output,
            "primary_k_tensor": k_tensor,
        }
