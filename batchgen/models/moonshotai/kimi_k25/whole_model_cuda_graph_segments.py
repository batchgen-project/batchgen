"""Kimi-K2.5 whole-model decode CUDA graph segment.

Captures the entire decode forward (embed -> 61 layers -> norm -> lm_head) as
one CUDA graph for globally bucketed decode batches. Mirrors
``glm5/whole_model_cuda_graph_segments.py`` but is **primary-KV only** (K2.5 is
plain MLA, no DSA indexer) and threads ``cache_seqlens`` / ``page_table`` /
``slot_indices`` explicitly through the layer segments, so it needs no
``AttnWrapperBase`` ClassVar bind/restore.

KV handling, two destinations (matching the working per-layer K2.5 path):
  * GPU paged KV cache: each ``K25AttnSegment`` writes in-graph to its own
    ``_k_cache[layer_idx]`` via the static ``page_table`` + ``slot_indices``
    (the page table is shared across layers — confirmed in
    ``gpu_paged_kv_manager.get_layer_kv_with_page_table``).
  * Host KV offload: each layer's ``primary_k_tensor`` is copied in-graph into a
    contiguous ``[num_layers, max_bucket, 1, 1, kv_dim]`` staging buffer; the
    worker clones once post-replay and dispatches ``kv_append_callback`` per
    layer (contract §C constraint #2).

Capture uses real runtime inputs (``set_capture_inputs``) so warmup does not
write fill-value positions into the real paged KV cache.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Dict, Iterable

import torch

from batchgen.cuda_graph.graph_manager import TensorSpec


class K25WholeModelSegment:
    """Graph-capturable K2.5 decode forward for global padded buckets."""

    def __init__(
        self,
        *,
        model,
        device: torch.device,
        world_size: int,
        max_pages_per_seq: int,
        vocab_size: int,
        hidden_size: int,
        max_bucket_size: int,
        layer_segments: Iterable[object],
        compare_probe_layers: Iterable[int] | None = None,
    ) -> None:
        if max_pages_per_seq <= 0:
            raise ValueError("max_pages_per_seq must be positive")
        if max_bucket_size <= 0:
            raise ValueError("max_bucket_size must be positive")
        if world_size <= 0:
            raise ValueError("world_size must be positive")

        self.model = model
        self.device = device
        self.world_size = int(world_size)
        self.max_pages_per_seq = int(max_pages_per_seq)
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.max_bucket_size = int(max_bucket_size)

        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None:
            raise ValueError("K25WholeModelSegment requires model.model.layers")
        self.num_layers = len(layers)
        if self.num_layers <= 0:
            raise ValueError("K25WholeModelSegment requires at least one decoder layer")

        self.layer_segments = list(layer_segments)
        if len(self.layer_segments) != self.num_layers:
            raise ValueError(
                "K25WholeModelSegment layer_segments length must match "
                f"num_layers={self.num_layers}, got {len(self.layer_segments)}"
            )
        self.primary_kv_dim = int(self.layer_segments[0].primary_kv_dim)

        probes = sorted({int(i) for i in (compare_probe_layers or [])})
        invalid = [i for i in probes if i < 0 or i >= self.num_layers]
        if invalid:
            raise ValueError(
                f"K2.5 whole-model probe layers out of range: {invalid}; "
                f"num_layers={self.num_layers}"
            )
        self.compare_probe_layers = tuple(probes)
        self._compare_probe_layer_set = set(self.compare_probe_layers)

        self._kv_buffers: list[dict[str, torch.Tensor | None]] | None = None
        self._kv_key_buffer: torch.Tensor | None = None
        self.primary_kv_offload_buffers: list[dict[str, torch.Tensor | None]] | None = None
        self._no_v_cache = True
        self._capture_inputs: dict[str, torch.Tensor] | None = None

    # ---- capture-time runtime inputs -------------------------------------

    def set_capture_inputs(self, **inputs: torch.Tensor) -> None:
        required = set(self.get_static_input_specs(self.max_bucket_size))
        missing = required - set(inputs)
        if missing:
            raise ValueError(f"missing K2.5 whole-model capture inputs: {sorted(missing)}")
        unknown = set(inputs) - required
        if unknown:
            raise ValueError(f"unknown K2.5 whole-model capture inputs: {sorted(unknown)}")
        self._capture_inputs = dict(inputs)

    def initialize_static_inputs(
        self,
        static_inputs: Mapping[str, torch.Tensor],
        bucket_size: int,
    ) -> None:
        # Always start from fill values (matches the shipped per-layer
        # K25AttnSegment warmup: cache_seqlens=1, page_table=0, slot_indices=0).
        input_specs = self.get_static_input_specs(bucket_size)
        for name, target in static_inputs.items():
            spec = input_specs.get(name)
            if spec is not None:
                target.fill_(spec.fill_value)

        # Optional: bind real runtime inputs if the worker provided them via
        # set_capture_inputs (lets capture run over the live batch instead of
        # fill-value padding). Not required — capture happens at startup before
        # any real decode, so fill-value slot-0 writes are overwritten on the
        # first real step, exactly as in the per-layer path.
        if self._capture_inputs is None:
            return
        for name, source in self._capture_inputs.items():
            target = static_inputs[name]
            if name == "rank_token_counts":
                if tuple(source.shape) != tuple(target.shape):
                    raise ValueError(
                        f"K2.5 whole-model capture input {name} shape "
                        f"{tuple(source.shape)} != static {tuple(target.shape)}"
                    )
                target.copy_(source.to(device=target.device, dtype=target.dtype), non_blocking=True)
                continue
            if source.shape[0] > target.shape[0]:
                raise ValueError(
                    f"K2.5 whole-model capture input {name} batch dim {source.shape[0]} "
                    f"exceeds static {tuple(target.shape)} for bucket {bucket_size}"
                )
            if source.shape[1:] != target.shape[1:]:
                raise ValueError(
                    f"K2.5 whole-model capture input {name} trailing shape "
                    f"{tuple(source.shape[1:])} != static {tuple(target.shape[1:])}"
                )
            if source.shape[0] > 0:
                target[: source.shape[0]].copy_(
                    source.to(device=target.device, dtype=target.dtype),
                    non_blocking=True,
                )

    # ---- static buffers --------------------------------------------------

    def setup_static_buffers(self, bucket_size: int) -> None:
        if bucket_size > self.max_bucket_size:
            raise ValueError(
                f"bucket_size {bucket_size} exceeds max_bucket_size {self.max_bucket_size}"
            )
        if self._kv_buffers is not None:
            return
        self._kv_key_buffer = torch.zeros(
            self.num_layers,
            self.max_bucket_size,
            1,
            1,
            self.primary_kv_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        self._kv_buffers = [
            {"key": self._kv_key_buffer[i], "value": None}
            for i in range(self.num_layers)
        ]
        self.primary_kv_offload_buffers = self._kv_buffers
        for layer_segment in self.layer_segments:
            setup = getattr(layer_segment, "setup_static_buffers", None)
            if setup is not None:
                setup(bucket_size)

    def release_static_buffers(self, bucket_size: int) -> None:
        for layer_segment in self.layer_segments:
            release = getattr(layer_segment, "release_static_buffers", None)
            if release is not None:
                release(bucket_size)
        self._kv_buffers = None
        self._kv_key_buffer = None
        self.primary_kv_offload_buffers = None
        self._capture_inputs = None

    # ---- specs -----------------------------------------------------------

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "input_ids": TensorSpec(("batch_size", 1), torch.int64, fill_value=0),
            "cache_seqlens": TensorSpec(("batch_size",), torch.int32, fill_value=1),
            "page_table": TensorSpec(
                ("batch_size", self.max_pages_per_seq), torch.int32, fill_value=0
            ),
            # -1 sentinel: padded bucket rows skip the in-graph KV write
            # (run_paged_kv_token_update_fused skips slot<0), so padding never
            # corrupts real sequences' KV. Matches the GLM-5 whole-model path.
            "slot_indices": TensorSpec(("batch_size",), torch.int32, fill_value=-1),
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

    # ---- KV staging ------------------------------------------------------

    def _copy_primary_kv(self, layer_idx: int, k_tensor: torch.Tensor) -> None:
        if self._kv_buffers is None:
            raise RuntimeError("K2.5 whole-model graph KV buffers are not initialized")
        if k_tensor.dim() == 3:
            k_tensor = k_tensor.unsqueeze(2)
        if k_tensor.dim() != 4 or k_tensor.shape[-1] != self.primary_kv_dim:
            raise RuntimeError(
                f"K2.5 whole-model KV tensor mismatch: got {tuple(k_tensor.shape)}, "
                f"expected (*, 1, 1, {self.primary_kv_dim})"
            )
        self._kv_buffers[int(layer_idx)]["key"][: k_tensor.shape[0]].copy_(k_tensor)

    @staticmethod
    def _probe_output_name(layer_idx: int) -> str:
        return f"probe_layer_{int(layer_idx):03d}_hidden"

    # ---- forward ---------------------------------------------------------

    def run_model_with_probes(
        self,
        *,
        input_ids: torch.Tensor,
        cache_seqlens: torch.Tensor,
        page_table: torch.Tensor,
        slot_indices: torch.Tensor,
        rank_token_counts: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden_states = self.model.model.embed_tokens(input_ids)
        outputs: dict[str, torch.Tensor] = {}
        for layer_idx, layer_segment in enumerate(self.layer_segments):
            graph_out = layer_segment.forward(
                hidden_states=hidden_states,
                cache_seqlens=cache_seqlens,
                page_table=page_table,
                slot_indices=slot_indices,
                rank_token_counts=rank_token_counts,
            )
            hidden_states = graph_out["hidden_states"]
            self._copy_primary_kv(layer_idx, graph_out["primary_k_tensor"])
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
        page_table: torch.Tensor,
        slot_indices: torch.Tensor,
        rank_token_counts: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        return self.run_model_with_probes(
            input_ids=input_ids,
            cache_seqlens=cache_seqlens,
            page_table=page_table,
            slot_indices=slot_indices,
            rank_token_counts=rank_token_counts,
        )


def make_k25_whole_model_graph_segment_name() -> str:
    return "k25_whole_model"
