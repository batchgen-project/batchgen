"""GLM-5 decode CUDA graph segment.

This first whole-model milestone captures the existing GLM-5 decode forward as
one CUDA graph for stable, globally bucketed decode batches. It deliberately keeps
host KV offload outside the graph by redirecting per-layer decode KV callbacks
to static GPU buffers during capture; the worker clones and appends those
buffers after replay.

The segment is intentionally strict: it is an opt-in sanity/performance path,
not a silent fallback. Padded rows use explicit -1 slot sentinels so all ranks
can participate in NCCL graph capture/replay even when a rank has no local rows.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Dict, Iterable

import torch

from batchgen.cuda_graph.graph_manager import TensorSpec
from batchgen.models.wrappers import AttnWrapperBase


class Glm5WholeModelSegment:
    """Graph-capturable GLM-5 decode forward for global padded buckets."""

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
        compare_probe_layers: Iterable[int] | None = None,
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
        probes = sorted({int(layer_idx) for layer_idx in (compare_probe_layers or [])})
        invalid_probes = [layer_idx for layer_idx in probes if layer_idx < 0 or layer_idx >= self.num_layers]
        if invalid_probes:
            raise ValueError(
                f"GLM-5 whole-model probe layers out of range: {invalid_probes}; "
                f"num_layers={self.num_layers}"
            )
        self.compare_probe_layers = tuple(probes)
        self._compare_probe_layer_set = set(self.compare_probe_layers)

        attn0 = layers[0].self_attn.module
        indexer0 = getattr(attn0, "indexer", None)
        if indexer0 is None:
            raise ValueError("GLM-5 whole-model graph requires DSA indexer modules")
        self.primary_kv_dim = int(attn0.kv_lora_rank + attn0.qk_rope_head_dim)
        self.aux_kv_dim = int(indexer0.index_head_dim)
        self.index_topk = int(indexer0.index_topk)

        self._kv_buffers: list[dict[str, torch.Tensor | None]] | None = None
        self._aux_kv_buffers: list[dict[str, torch.Tensor | None]] | None = None
        self._kv_key_buffer: torch.Tensor | None = None
        self._aux_kv_key_buffer: torch.Tensor | None = None
        self.primary_kv_offload_buffers: list[dict[str, torch.Tensor | None]] | None = None
        self.aux_kv_offload_buffers: list[dict[str, torch.Tensor | None]] | None = None
        self._no_v_cache = True
        self._capture_inputs: dict[str, torch.Tensor] | None = None
        self._capture_dsa_short_count: int | None = None

    def set_capture_inputs(self, **inputs: torch.Tensor) -> None:
        required = set(self.get_static_input_specs(self.max_bucket_size))
        missing = required - set(inputs)
        if missing:
            raise ValueError(f"missing GLM-5 whole-model capture inputs: {sorted(missing)}")
        unknown = set(inputs) - required
        if unknown:
            raise ValueError(f"unknown GLM-5 whole-model capture inputs: {sorted(unknown)}")
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
        input_specs = self.get_static_input_specs(bucket_size)
        for name, target in static_inputs.items():
            spec = input_specs.get(name)
            if spec is not None:
                target.fill_(spec.fill_value)

        for name, source in self._capture_inputs.items():
            target = static_inputs[name]
            if name == "rank_token_counts":
                if tuple(source.shape) != tuple(target.shape):
                    raise ValueError(
                        f"GLM-5 whole-model capture input {name} shape {tuple(source.shape)} "
                        f"does not match static shape {tuple(target.shape)} for bucket {bucket_size}"
                    )
                target.copy_(source.to(device=target.device, dtype=target.dtype), non_blocking=True)
                continue
            if source.shape[0] > target.shape[0]:
                raise ValueError(
                    f"GLM-5 whole-model capture input {name} batch dim {source.shape[0]} "
                    f"exceeds static shape {tuple(target.shape)} for bucket {bucket_size}"
                )
            if source.shape[1:] != target.shape[1:]:
                raise ValueError(
                    f"GLM-5 whole-model capture input {name} trailing shape "
                    f"{tuple(source.shape[1:])} does not match static shape "
                    f"{tuple(target.shape[1:])} for bucket {bucket_size}"
                )
            if source.shape[0] > 0:
                target[: source.shape[0]].copy_(
                    source.to(device=target.device, dtype=target.dtype),
                    non_blocking=True,
                )
        cache_seqlens = static_inputs.get("cache_seqlens")
        primary_slot_indices = static_inputs.get("primary_slot_indices")
        if cache_seqlens is not None and primary_slot_indices is not None:
            valid_rows = primary_slot_indices >= 0
            self._capture_dsa_short_count = int(
                ((cache_seqlens <= self.index_topk) & valid_rows).sum().item()
            )

    def setup_static_buffers(self, bucket_size: int) -> None:
        if bucket_size > self.max_bucket_size:
            raise ValueError(
                f"bucket_size {bucket_size} exceeds max_bucket_size {self.max_bucket_size}"
            )
        if self._kv_buffers is not None and self._aux_kv_buffers is not None:
            return

        alloc_size = self.max_bucket_size
        self._kv_key_buffer = torch.zeros(
            self.num_layers,
            alloc_size,
            1,
            1,
            self.primary_kv_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        self._aux_kv_key_buffer = torch.zeros(
            self.num_layers,
            alloc_size,
            1,
            1,
            self.aux_kv_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        self._kv_buffers = [
            {"key": self._kv_key_buffer[layer_idx], "value": None}
            for layer_idx in range(self.num_layers)
        ]
        self._aux_kv_buffers = [
            {"key": self._aux_kv_key_buffer[layer_idx], "value": None}
            for layer_idx in range(self.num_layers)
        ]
        self.primary_kv_offload_buffers = self._kv_buffers
        self.aux_kv_offload_buffers = self._aux_kv_buffers

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "input_ids": TensorSpec(("batch_size", 1), torch.int64, fill_value=0),
            "cache_seqlens": TensorSpec(
                ("batch_size",), torch.int32, fill_value=float(self.max_seqlen)
            ),
            "position_ids": TensorSpec(("batch_size", 1), torch.int64, fill_value=0),
            "primary_slot_indices": TensorSpec(("batch_size",), torch.int32, fill_value=-1),
            "aux_slot_indices": TensorSpec(("batch_size",), torch.int32, fill_value=-1),
            "rank_token_counts": TensorSpec((self.world_size,), torch.int64, fill_value=0),
        }

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        specs = {
            "hidden_states": TensorSpec(("batch_size", self.hidden_size), torch.bfloat16),
            "logits": TensorSpec(("batch_size", self.vocab_size), torch.bfloat16),
        }
        for layer_idx in self.compare_probe_layers:
            specs[self._probe_output_name(layer_idx)] = TensorSpec(
                ("batch_size", self.hidden_size), torch.bfloat16
            )
        return specs

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
                mlp.set_num_tokens_per_rank(int(bucket_size))

    @staticmethod
    def _probe_output_name(layer_idx: int) -> str:
        return f"probe_layer_{int(layer_idx):03d}_hidden"

    def run_model_with_probes(
        self,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden_states = self.model.model.embed_tokens(input_ids)
        outputs: dict[str, torch.Tensor] = {}
        for layer_idx, layer in enumerate(self.model.model.layers):
            hidden_states, _, _ = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                use_cache=False,
            )
            if layer_idx in self._compare_probe_layer_set:
                outputs[self._probe_output_name(layer_idx)] = hidden_states[:, -1, :]

        hidden_states = self.model.model.norm(hidden_states)
        logits = self.model.lm_head(hidden_states)
        outputs["hidden_states"] = hidden_states[:, -1, :]
        outputs["logits"] = logits[:, -1, :]
        return outputs

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_seqlens: torch.Tensor,
        position_ids: torch.Tensor,
        primary_slot_indices: torch.Tensor,
        aux_slot_indices: torch.Tensor,
        rank_token_counts: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        bucket_size = int(input_ids.shape[0])
        self._set_moe_bucket_state(bucket_size, rank_token_counts)

        old_cache_seqlens = AttnWrapperBase.cache_seqlens
        old_position_ids = AttnWrapperBase.position_ids
        old_max_seqlen = AttnWrapperBase.max_seqlen
        old_kv_cb = AttnWrapperBase.kv_append_callback
        old_aux_cb = AttnWrapperBase.kv_append_callback_aux
        old_dsa_short_count = AttnWrapperBase._dsa_short_count
        old_primary_slots = AttnWrapperBase.glm5_decode_primary_slot_indices
        old_aux_slots = AttnWrapperBase.glm5_decode_aux_slot_indices
        try:
            AttnWrapperBase.cache_seqlens = cache_seqlens
            AttnWrapperBase.position_ids = position_ids
            AttnWrapperBase.max_seqlen = self.max_seqlen
            AttnWrapperBase.kv_append_callback = self._copy_primary_kv
            AttnWrapperBase.kv_append_callback_aux = self._copy_aux_kv
            AttnWrapperBase._dsa_short_count = self._capture_dsa_short_count
            AttnWrapperBase.glm5_decode_primary_slot_indices = primary_slot_indices
            AttnWrapperBase.glm5_decode_aux_slot_indices = aux_slot_indices
            outputs = self.run_model_with_probes(
                input_ids=input_ids,
                position_ids=position_ids,
            )
        finally:
            AttnWrapperBase.cache_seqlens = old_cache_seqlens
            AttnWrapperBase.position_ids = old_position_ids
            AttnWrapperBase.max_seqlen = old_max_seqlen
            AttnWrapperBase.kv_append_callback = old_kv_cb
            AttnWrapperBase.kv_append_callback_aux = old_aux_cb
            AttnWrapperBase._dsa_short_count = old_dsa_short_count
            AttnWrapperBase.glm5_decode_primary_slot_indices = old_primary_slots
            AttnWrapperBase.glm5_decode_aux_slot_indices = old_aux_slots

        return outputs


def make_glm5_whole_model_graph_segment_name() -> str:
    return "glm5_whole_model"


def compare_glm5_whole_model_graph_logits(
    *,
    eager_logits: torch.Tensor,
    graph_logits: torch.Tensor,
    eager_hidden_states: torch.Tensor | None = None,
    graph_hidden_states: torch.Tensor | None = None,
    eager_probe_hidden_states: Mapping[str, torch.Tensor] | None = None,
    graph_probe_hidden_states: Mapping[str, torch.Tensor] | None = None,
    eager_tokens: torch.Tensor | None = None,
    graph_tokens: torch.Tensor | None = None,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> dict[str, object]:
    """Compare eager and graph logits without changing decode control flow."""
    if eager_logits.shape != graph_logits.shape:
        return {
            "ok": False,
            "shape_match": False,
            "eager_shape": tuple(int(dim) for dim in eager_logits.shape),
            "graph_shape": tuple(int(dim) for dim in graph_logits.shape),
            "max_abs": float("inf"),
            "mean_abs": float("inf"),
            "argmax_mismatch": -1,
            "token_mismatch": -1,
        }

    eager_f = eager_logits.detach().to(torch.float32)
    graph_f = graph_logits.detach().to(torch.float32)
    diff = (eager_f - graph_f).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    logits_ok = bool(torch.allclose(eager_f, graph_f, atol=atol, rtol=rtol))
    argmax_mismatch = int(
        (torch.argmax(eager_f, dim=-1) != torch.argmax(graph_f, dim=-1)).sum().item()
    )

    token_mismatch = 0
    token_shape_match = True
    if eager_tokens is not None and graph_tokens is not None:
        token_shape_match = eager_tokens.shape == graph_tokens.shape
        if token_shape_match:
            token_mismatch = int((eager_tokens != graph_tokens).sum().item())
        else:
            token_mismatch = -1

    hidden_ok = True
    hidden_shape_match = True
    hidden_max_abs = 0.0
    hidden_mean_abs = 0.0
    if eager_hidden_states is not None or graph_hidden_states is not None:
        if eager_hidden_states is None or graph_hidden_states is None:
            hidden_ok = False
            hidden_shape_match = False
            hidden_max_abs = float("inf")
            hidden_mean_abs = float("inf")
        else:
            hidden_shape_match = eager_hidden_states.shape == graph_hidden_states.shape
            if hidden_shape_match:
                eager_h = eager_hidden_states.detach().to(torch.float32)
                graph_h = graph_hidden_states.detach().to(torch.float32)
                hidden_diff = (eager_h - graph_h).abs()
                hidden_max_abs = float(hidden_diff.max().item()) if hidden_diff.numel() else 0.0
                hidden_mean_abs = float(hidden_diff.mean().item()) if hidden_diff.numel() else 0.0
                hidden_ok = bool(torch.allclose(eager_h, graph_h, atol=atol, rtol=rtol))
            else:
                hidden_ok = False
                hidden_max_abs = float("inf")
                hidden_mean_abs = float("inf")

    probe_first_mismatch = ""
    probe_max_abs = 0.0
    probe_mean_abs = 0.0
    probe_shape_match = True
    probe_ok = True
    if eager_probe_hidden_states is not None or graph_probe_hidden_states is not None:
        eager_probe_hidden_states = eager_probe_hidden_states or {}
        graph_probe_hidden_states = graph_probe_hidden_states or {}
        if set(eager_probe_hidden_states) != set(graph_probe_hidden_states):
            probe_ok = False
            probe_shape_match = False
            probe_first_mismatch = "probe_key_set"
            probe_max_abs = float("inf")
            probe_mean_abs = float("inf")
        else:
            for name in sorted(eager_probe_hidden_states):
                eager_probe = eager_probe_hidden_states[name]
                graph_probe = graph_probe_hidden_states[name]
                if eager_probe.shape != graph_probe.shape:
                    probe_ok = False
                    probe_shape_match = False
                    probe_first_mismatch = name
                    probe_max_abs = float("inf")
                    probe_mean_abs = float("inf")
                    break
                eager_p = eager_probe.detach().to(torch.float32)
                graph_p = graph_probe.detach().to(torch.float32)
                probe_diff = (eager_p - graph_p).abs()
                cur_max = float(probe_diff.max().item()) if probe_diff.numel() else 0.0
                cur_mean = float(probe_diff.mean().item()) if probe_diff.numel() else 0.0
                probe_max_abs = max(probe_max_abs, cur_max)
                probe_mean_abs = max(probe_mean_abs, cur_mean)
                cur_ok = bool(torch.allclose(eager_p, graph_p, atol=atol, rtol=rtol))
                if not cur_ok and not probe_first_mismatch:
                    probe_first_mismatch = name
                    probe_ok = False

    return {
        "ok": bool(
            logits_ok
            and hidden_ok
            and probe_ok
            and argmax_mismatch == 0
            and token_shape_match
            and token_mismatch == 0
        ),
        "shape_match": True,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "hidden_shape_match": hidden_shape_match,
        "hidden_max_abs": hidden_max_abs,
        "hidden_mean_abs": hidden_mean_abs,
        "probe_shape_match": probe_shape_match,
        "probe_first_mismatch": probe_first_mismatch,
        "probe_max_abs": probe_max_abs,
        "probe_mean_abs": probe_mean_abs,
        "argmax_mismatch": argmax_mismatch,
        "token_mismatch": token_mismatch,
        "token_shape_match": token_shape_match,
    }
