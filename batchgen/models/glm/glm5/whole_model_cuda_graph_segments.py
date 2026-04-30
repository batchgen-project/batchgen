"""GLM-5 decode CUDA graph segment.

This first whole-model milestone captures the existing GLM-5 decode forward as
one CUDA graph for stable, exact-bucket decode batches. It deliberately keeps
host KV offload outside the graph by redirecting per-layer decode KV callbacks
to static GPU buffers during capture; the worker clones and appends those
buffers after replay.

The segment is intentionally strict: it is an opt-in sanity/performance path,
not a silent fallback. General padded-bucket support still needs the planned
explicit slot/page-table inputs to replace the current wrapper-side Python batch
bookkeeping.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Dict

import torch

from batchgen.cuda_graph.graph_manager import TensorSpec
from batchgen.models.wrappers import AttnWrapperBase


class Glm5WholeModelSegment:
    """Graph-capturable GLM-5 decode forward for exact local buckets."""

    def __init__(
        self,
        *,
        model,
        device: torch.device,
        world_size: int,
        max_pages_per_seq: int,
        max_aux_pages_per_seq: int,
        vocab_size: int,
        hidden_size: int,
        max_bucket_size: int,
        max_seqlen: int,
        include_embedding: bool = True,
        include_lm_head: bool = True,
    ) -> None:
        if not include_embedding:
            raise NotImplementedError(
                "GLM-5 whole-model graph currently captures input_ids -> embedding; "
                "hidden-state input will be added with the padded-slot hardening pass"
            )
        if not include_lm_head:
            raise NotImplementedError(
                "GLM-5 whole-model graph currently returns logits; hidden-state "
                "output will be added with the padded-slot hardening pass"
            )
        if max_pages_per_seq <= 0:
            raise ValueError("max_pages_per_seq must be positive")
        if max_aux_pages_per_seq <= 0:
            raise ValueError("max_aux_pages_per_seq must be positive")
        if max_bucket_size <= 0:
            raise ValueError("max_bucket_size must be positive")
        if max_seqlen <= 0:
            raise ValueError("max_seqlen must be positive")
        if world_size <= 0:
            raise ValueError("world_size must be positive")

        self.model = model
        self.device = device
        self.world_size = int(world_size)
        self.max_pages_per_seq = int(max_pages_per_seq)
        self.max_aux_pages_per_seq = int(max_aux_pages_per_seq)
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.max_bucket_size = int(max_bucket_size)
        self.max_seqlen = int(max_seqlen)
        self.include_embedding = bool(include_embedding)
        self.include_lm_head = bool(include_lm_head)

        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None:
            raise ValueError("Glm5WholeModelSegment requires model.model.layers")
        self.num_layers = len(layers)
        if self.num_layers <= 0:
            raise ValueError("Glm5WholeModelSegment requires at least one decoder layer")

        attn0 = layers[0].self_attn.module
        indexer0 = getattr(attn0, "indexer", None)
        if indexer0 is None:
            raise ValueError("GLM-5 whole-model graph requires DSA indexer modules")
        self.primary_kv_dim = int(attn0.kv_lora_rank + attn0.qk_rope_head_dim)
        self.aux_kv_dim = int(indexer0.index_head_dim)

        self._kv_buffers: list[dict[str, torch.Tensor | None]] | None = None
        self._aux_kv_buffers: list[dict[str, torch.Tensor | None]] | None = None
        self.primary_kv_offload_buffers: list[dict[str, torch.Tensor | None]] | None = None
        self.aux_kv_offload_buffers: list[dict[str, torch.Tensor | None]] | None = None
        self._no_v_cache = True
        self._capture_inputs: dict[str, torch.Tensor] | None = None

    def set_capture_inputs(self, **inputs: torch.Tensor) -> None:
        required = set(self.get_static_input_specs(self.max_bucket_size))
        missing = required - set(inputs)
        if missing:
            raise ValueError(f"missing GLM-5 whole-model capture inputs: {sorted(missing)}")
        self._capture_inputs = dict(inputs)

    def initialize_static_inputs(
        self,
        static_inputs: Mapping[str, torch.Tensor],
        bucket_size: int,
    ) -> None:
        if self._capture_inputs is None:
            raise RuntimeError(
                "GLM-5 whole-model graph capture requires runtime inputs so warmup "
                "does not mutate real KV cache with fill-value positions"
            )
        for name, source in self._capture_inputs.items():
            target = static_inputs[name]
            if source.shape[0] != target.shape[0]:
                raise ValueError(
                    f"GLM-5 whole-model capture input {name} shape {tuple(source.shape)} "
                    f"does not match static shape {tuple(target.shape)} for bucket {bucket_size}"
                )
            target.copy_(source.to(device=target.device, dtype=target.dtype), non_blocking=True)

    def setup_static_buffers(self, bucket_size: int) -> None:
        if bucket_size > self.max_bucket_size:
            raise ValueError(
                f"bucket_size {bucket_size} exceeds max_bucket_size {self.max_bucket_size}"
            )
        if self._kv_buffers is not None and self._aux_kv_buffers is not None:
            return

        alloc_size = self.max_bucket_size
        self._kv_buffers = []
        self._aux_kv_buffers = []
        for _ in range(self.num_layers):
            self._kv_buffers.append(
                {
                    "key": torch.zeros(
                        alloc_size,
                        1,
                        1,
                        self.primary_kv_dim,
                        dtype=torch.bfloat16,
                        device=self.device,
                    ),
                    "value": None,
                }
            )
            self._aux_kv_buffers.append(
                {
                    "key": torch.zeros(
                        alloc_size,
                        1,
                        1,
                        self.aux_kv_dim,
                        dtype=torch.bfloat16,
                        device=self.device,
                    ),
                    "value": None,
                }
            )
        self.primary_kv_offload_buffers = self._kv_buffers
        self.aux_kv_offload_buffers = self._aux_kv_buffers

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "input_ids": TensorSpec(("batch_size", 1), torch.int64, fill_value=0),
            "cache_seqlens": TensorSpec(
                ("batch_size",), torch.int32, fill_value=float(self.max_seqlen)
            ),
            "position_ids": TensorSpec(("batch_size", 1), torch.int64, fill_value=0),
            "primary_page_table": TensorSpec(
                ("batch_size", self.max_pages_per_seq), torch.int32, fill_value=0
            ),
            "aux_page_table": TensorSpec(
                ("batch_size", self.max_aux_pages_per_seq), torch.int32, fill_value=0
            ),
            "primary_slot_indices": TensorSpec(("batch_size",), torch.int32, fill_value=-1),
            "aux_slot_indices": TensorSpec(("batch_size",), torch.int32, fill_value=-1),
            "rank_token_counts": TensorSpec((self.world_size,), torch.int64, fill_value=0),
            "num_valid_tokens": TensorSpec((1,), torch.int32, fill_value=0),
        }

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "logits": TensorSpec(("batch_size", self.vocab_size), torch.bfloat16),
        }

    def _copy_primary_kv(self, layer_idx: int, k_tensor: torch.Tensor, _v_tensor=None) -> None:
        if self._kv_buffers is None:
            raise RuntimeError("GLM-5 whole-model graph KV buffers are not initialized")
        k_tensor = self._normalize_k_tensor(k_tensor, self.primary_kv_dim)
        self._kv_buffers[int(layer_idx)]["key"][: k_tensor.shape[0]].copy_(k_tensor)

    def _copy_aux_kv(self, layer_idx: int, k_tensor: torch.Tensor, _v_tensor=None) -> None:
        if self._aux_kv_buffers is None:
            raise RuntimeError("GLM-5 whole-model graph aux KV buffers are not initialized")
        k_tensor = self._normalize_k_tensor(k_tensor, self.aux_kv_dim)
        self._aux_kv_buffers[int(layer_idx)]["key"][: k_tensor.shape[0]].copy_(k_tensor)

    @staticmethod
    def _normalize_k_tensor(k_tensor: torch.Tensor, expected_dim: int) -> torch.Tensor:
        if k_tensor.dim() == 3:
            k_tensor = k_tensor.unsqueeze(2)
        if k_tensor.dim() != 4:
            raise RuntimeError(
                f"GLM-5 whole-model graph expected 3-D/4-D KV tensor, got {tuple(k_tensor.shape)}"
            )
        if k_tensor.shape[-1] != expected_dim:
            raise RuntimeError(
                f"GLM-5 whole-model graph KV dim mismatch: got {k_tensor.shape[-1]}, "
                f"expected {expected_dim}"
            )
        return k_tensor

    def _set_moe_bucket_state(self, bucket_size: int, rank_token_counts: torch.Tensor) -> None:
        from batchgen.models.glm.glm5.model import Glm5MoE

        Glm5MoE._rank_token_counts = rank_token_counts
        for layer in self.model.model.layers:
            mlp = getattr(layer, "mlp", None)
            if isinstance(mlp, Glm5MoE):
                mlp.num_tokens_per_rank = int(bucket_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_seqlens: torch.Tensor,
        position_ids: torch.Tensor,
        primary_page_table: torch.Tensor,
        aux_page_table: torch.Tensor,
        primary_slot_indices: torch.Tensor,
        aux_slot_indices: torch.Tensor,
        rank_token_counts: torch.Tensor,
        num_valid_tokens: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        del primary_page_table, aux_page_table, primary_slot_indices, aux_slot_indices
        del num_valid_tokens

        bucket_size = int(input_ids.shape[0])
        self._set_moe_bucket_state(bucket_size, rank_token_counts)

        old_cache_seqlens = AttnWrapperBase.cache_seqlens
        old_position_ids = AttnWrapperBase.position_ids
        old_max_seqlen = AttnWrapperBase.max_seqlen
        old_kv_cb = AttnWrapperBase.kv_append_callback
        old_aux_cb = AttnWrapperBase.kv_append_callback_aux
        try:
            AttnWrapperBase.cache_seqlens = cache_seqlens
            AttnWrapperBase.position_ids = position_ids
            AttnWrapperBase.max_seqlen = self.max_seqlen
            AttnWrapperBase.kv_append_callback = self._copy_primary_kv
            AttnWrapperBase.kv_append_callback_aux = self._copy_aux_kv
            outputs = self.model(
                input_ids,
                position_ids=position_ids,
                use_cache=False,
            )
        finally:
            AttnWrapperBase.cache_seqlens = old_cache_seqlens
            AttnWrapperBase.position_ids = old_position_ids
            AttnWrapperBase.max_seqlen = old_max_seqlen
            AttnWrapperBase.kv_append_callback = old_kv_cb
            AttnWrapperBase.kv_append_callback_aux = old_aux_cb

        logits = outputs.logits[:, -1, :]
        return {"logits": logits}


def make_glm5_whole_model_graph_segment_name() -> str:
    return "glm5_whole_model"
