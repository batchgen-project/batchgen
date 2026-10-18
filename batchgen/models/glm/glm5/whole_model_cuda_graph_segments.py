"""GLM-5 decode CUDA graph segment.

This production whole-model path captures the existing GLM-5 decode forward as
one CUDA graph for stable, globally bucketed decode batches. It deliberately keeps
host KV offload outside the graph by redirecting per-layer decode KV callbacks
to static GPU buffers during capture; the worker stages and appends those
buffers after replay.

The segment is intentionally strict: when selected by the server CUDA graph
policy it must either replay a valid whole-model graph or surface a clear
runtime error. Padded rows use explicit -1 slot sentinels so all ranks can
participate in NCCL graph capture/replay even when a rank has no local rows.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Dict, Iterable

import torch

from batchgen.cuda_graph.graph_manager import TensorSpec
from batchgen.models.wrappers import AttnWrapperBase
# Phase C: GLM-5-specific ClassVars (_dsa_short_count, glm5_decode_*_slot_indices)
# moved from AttnWrapperBase to GLM5AttnWrapper per audit §A finding #8.
from batchgen.models.glm.glm5.wrappers import GLM5AttnWrapper


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
        layer_segments: Iterable[object] | None = None,
        aux_kv_staging_dtype: torch.dtype = torch.bfloat16,
        primary_kv_dim: int | None = None,
        primary_kv_staging_dtype: torch.dtype | None = None,
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
        self.layer_segments = list(layer_segments or [])
        if self.layer_segments and len(self.layer_segments) != self.num_layers:
            raise ValueError(
                "GLM-5 whole-model graph layer_segments length must match "
                f"num_layers={self.num_layers}, got {len(self.layer_segments)}"
            )

        attn0 = layers[0].self_attn.module
        indexer0 = getattr(attn0, "indexer", None)
        if indexer0 is None:
            raise ValueError("GLM-5 whole-model graph requires DSA indexer modules")
        self.aux_kv_dim = int(indexer0.index_head_dim)
        self.index_topk = int(indexer0.index_topk)
        # The PRIMARY staging record. `kv_lora_rank + qk_rope_head_dim` is the
        # model's latent width (576); it is the record width only for a bf16
        # AoS cache. GLM-5.2's packed FP8 cache stores 656 BYTES per token, so
        # the staging buffer's dim AND dtype have to come from the same place
        # the graph segment's record comes from -- the ALLOCATED cache -- or
        # the buffer stages something that is neither the producer's bytes nor
        # the host region's layout. See `_resolve_primary_record` below.
        (
            self.primary_kv_dim,
            self.primary_kv_staging_dtype,
        ) = self._resolve_primary_record(
            model_source_dim=int(attn0.kv_lora_rank + attn0.qk_rope_head_dim),
            layer_segments=self.layer_segments,
            explicit_dim=primary_kv_dim,
            explicit_dtype=primary_kv_staging_dtype,
        )
        self.primary_kv_is_packed = self.primary_kv_staging_dtype is torch.uint8
        # The aux KV stream need not share the primary's element type: GLM-5.2
        # stages its indexer keys as e4m3 bytes (one fp32 scale per token
        # alongside), everything else stages bf16. Staging the FP8 stream in
        # bf16 would requantize it on the clone, so the dtype is explicit and
        # anything else is rejected here.
        if aux_kv_staging_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            raise ValueError(
                "aux_kv_staging_dtype must be torch.bfloat16 or "
                f"torch.float8_e4m3fn, got {aux_kv_staging_dtype}"
            )
        self.aux_kv_staging_dtype = aux_kv_staging_dtype
        self.aux_kv_is_page_split = aux_kv_staging_dtype is torch.float8_e4m3fn

        self._kv_buffers: list[dict[str, torch.Tensor | None]] | None = None
        self._aux_kv_buffers: list[dict[str, torch.Tensor | None]] | None = None
        self._kv_key_buffer: torch.Tensor | None = None
        self._aux_kv_key_buffer: torch.Tensor | None = None
        # Page-split aux only: one fp32 scale per (layer, row), staged next to
        # the quantized keys because the host region needs both or neither.
        self._aux_kv_scale_buffer: torch.Tensor | None = None
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
        if self._kv_buffers is None or self._aux_kv_buffers is None:
            alloc_size = self.max_bucket_size
            self._kv_key_buffer = torch.zeros(
                self.num_layers,
                alloc_size,
                1,
                1,
                self.primary_kv_dim,
                # Resolved from the allocated primary cache, not from the
                # model's latent width: bf16 x 576 elements, or uint8 x 656
                # bytes for FlashMLA's packed FP8 record. A hardcoded bf16
                # here would make `_copy_primary_kv`'s `.copy_()` requantize
                # a packed record per element.
                dtype=self.primary_kv_staging_dtype,
                device=self.device,
            )
            self._aux_kv_key_buffer = torch.zeros(
                self.num_layers,
                alloc_size,
                1,
                1,
                self.aux_kv_dim,
                dtype=self.aux_kv_staging_dtype,
                device=self.device,
            )
            self._aux_kv_scale_buffer = (
                torch.zeros(
                    self.num_layers,
                    alloc_size,
                    dtype=torch.float32,
                    device=self.device,
                )
                if self.aux_kv_is_page_split
                else None
            )
            self._kv_buffers = [
                {"key": self._kv_key_buffer[layer_idx], "value": None}
                for layer_idx in range(self.num_layers)
            ]
            self._aux_kv_buffers = [
                {
                    "key": self._aux_kv_key_buffer[layer_idx],
                    "value": None,
                    "scale": (
                        None
                        if self._aux_kv_scale_buffer is None
                        else self._aux_kv_scale_buffer[layer_idx]
                    ),
                }
                for layer_idx in range(self.num_layers)
            ]
            self.primary_kv_offload_buffers = self._kv_buffers
            self.aux_kv_offload_buffers = self._aux_kv_buffers
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
        self._aux_kv_buffers = None
        self._kv_key_buffer = None
        self._aux_kv_key_buffer = None
        self._aux_kv_scale_buffer = None
        self.primary_kv_offload_buffers = None
        self.aux_kv_offload_buffers = None
        self._capture_inputs = None
        self._capture_dsa_short_count = None

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        specs = {
            "input_ids": TensorSpec(("batch_size", 1), torch.int64, fill_value=0),
            "cache_seqlens": TensorSpec(
                ("batch_size",), torch.int32, fill_value=float(self.max_seqlen)
            ),
            "position_ids": TensorSpec(("batch_size", 1), torch.int64, fill_value=0),
            "primary_slot_indices": TensorSpec(("batch_size",), torch.int32, fill_value=-1),
            "aux_slot_indices": TensorSpec(("batch_size",), torch.int32, fill_value=-1),
            "rank_token_counts": TensorSpec((self.world_size,), torch.int64, fill_value=0),
        }
        if self.layer_segments:
            specs["num_valid_tokens"] = TensorSpec(
                (1,), torch.int32, fill_value=float(bucket_size)
            )
        return specs

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
        """Stage one layer's primary MLA record for host offload.

        Bound as ``AttnWrapperBase.kv_append_callback`` from three places --
        the in-graph layer loop in `run_model_with_probes`, the eager
        save/restore window in `forward`, and
        `batchgen/models/glm/glm5/cuda_graph_adapter.py:572` -- and all three
        hand over the same tensor: the DSA segment's ``primary_k_tensor``
        static output, whose spec is declared from the cache's record geometry
        at `batchgen/models/glm/glm5/cuda_graph_segments.py:1311-1312`.

        ``copy_()`` CONVERTS dtypes. A bf16 record copied into a uint8 staging
        buffer would be requantized per element and the staged page would be
        neither the producer's bytes nor the layout FlashMLA reads, so the
        dtype is asserted first -- the same guard
        `batchgen_kernels/triton/kv_cache.py:12-28`
        (``_require_matching_cache_dtype``) applies to the paged-KV update for
        exactly this reason.
        """

        if self._kv_buffers is None:
            raise RuntimeError("GLM-5 whole-model graph KV buffers are not initialized")
        k_tensor = self._normalize_k_tensor(
            k_tensor, self.primary_kv_dim, self.primary_kv_staging_dtype
        )
        self._kv_buffers[int(layer_idx)]["key"][: k_tensor.shape[0]].copy_(k_tensor)

    def _copy_aux_kv(
        self,
        layer_idx: int,
        k_tensor: torch.Tensor,
        _v_tensor=None,
        scale_tensor: torch.Tensor | None = None,
    ) -> None:
        """Stage one layer's indexer keys (and their scales) for host offload.

        Bound as ``kv_append_callback_aux`` for the capture/replay window, so
        it takes the same optional trailing scales the real callback takes: a
        page-split FP8 aux stream must carry them, and an AoS stream must not.
        """

        if self._aux_kv_buffers is None:
            raise RuntimeError("GLM-5 whole-model graph aux KV buffers are not initialized")
        k_tensor = self._normalize_k_tensor(
            k_tensor, self.aux_kv_dim, self.aux_kv_staging_dtype
        )
        self._aux_kv_buffers[int(layer_idx)]["key"][: k_tensor.shape[0]].copy_(k_tensor)
        scale_buffer = self._aux_kv_buffers[int(layer_idx)].get("scale")
        if (scale_tensor is None) != (scale_buffer is None):
            raise RuntimeError(
                "GLM-5 whole-model graph aux KV staging expects per-token "
                "scales exactly when the aux stream is page-split FP8 "
                f"(got scales={scale_tensor is not None}, "
                f"page_split={scale_buffer is not None})"
            )
        if scale_tensor is not None:
            scale_flat = scale_tensor.reshape(-1)
            scale_buffer[: scale_flat.shape[0]].copy_(scale_flat)

    @staticmethod
    def _normalize_k_tensor(
        k_tensor: torch.Tensor,
        expected_dim: int,
        expected_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Shape- and dtype-check one staged KV stream's incoming tensor.

        ``expected_dtype`` is a parameter rather than a constant because the
        primary and aux streams legitimately differ: GLM-5.2 stages packed
        uint8 primary records beside e4m3 indexer keys, GLM-5 stages bf16 for
        both. The dtype check is what turns a format mismatch into a named
        error instead of a silent per-element conversion inside the caller's
        ``copy_()``.
        """

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
        if k_tensor.dtype != expected_dtype:
            raise RuntimeError(
                "GLM-5 whole-model graph KV staging dtype mismatch: got "
                f"{k_tensor.dtype}, expected {expected_dtype}. The KV record "
                "is produced once, at the writer, and every hop after it is a "
                "byte copy; copying across dtypes here would requantize the "
                "record per element."
            )
        return k_tensor

    @staticmethod
    def _resolve_primary_record(
        *,
        model_source_dim: int,
        layer_segments: list,
        explicit_dim: int | None,
        explicit_dtype: torch.dtype | None,
    ) -> tuple[int, torch.dtype]:
        """``(record_dim, record_dtype)`` of the PRIMARY staging buffer.

        Resolution order, strongest authority first:

        1. The allocated primary cache, reached through the layer segments.
           A layer segment declares the record of the ``primary_k_tensor`` it
           will actually hand to ``_copy_primary_kv`` as
           ``primary_kv_dim`` / ``primary_kv_dtype``
           (`batchgen/models/glm/glm5/layer_cuda_graph_segments.py:51-63`,
           used for its output spec at :141-144), and its DSA segment
           independently publishes ``primary_record_dim`` /
           ``primary_record_dtype`` read off ``primary_blocked_k``
           (`batchgen/models/glm/glm5/cuda_graph_segments.py:1018-1019`).
           Both are read here and must agree -- they are two derivations of
           one cache (``shape[3]`` and ``shape[-1]`` of the same 4-D
           ``[pages, page_size, 1, record_dim]`` tensor), so a disagreement
           means one of them has drifted and is named rather than picked
           between. All layers must agree too: one cache, one record.
        2. An explicit ``primary_kv_dim`` / ``primary_kv_staging_dtype``, which
           a caller that holds the GPU KV manager but builds no layer segments
           can pass. When BOTH sources are present they must match; the
           disagreement is named rather than silently resolved in favour of
           one.
        3. The model's own latent width in bf16 -- the pre-migration behaviour,
           and correct for every bf16 AoS cache (GLM-5 / 5.1 / 5.3). It is NOT
           a silent fallback for a packed cache: ``_copy_primary_kv``'s dtype
           guard rejects the uint8 record this buffer could not hold, naming
           the mismatch before any byte is converted.
        """

        derived: tuple[int, torch.dtype] | None = None
        for index, layer_segment in enumerate(layer_segments):
            candidates = []
            for holder, dim_attr, dtype_attr in (
                (layer_segment, "primary_kv_dim", "primary_kv_dtype"),
                (
                    getattr(layer_segment, "dsa_segment", None),
                    "primary_record_dim",
                    "primary_record_dtype",
                ),
            ):
                if holder is None:
                    continue
                record_dim = getattr(holder, dim_attr, None)
                record_dtype = getattr(holder, dtype_attr, None)
                if record_dim is None or record_dtype is None:
                    continue
                candidates.append((int(record_dim), record_dtype))
            if len(candidates) == 2 and candidates[0] != candidates[1]:
                raise ValueError(
                    f"GLM-5 whole-model graph layer {index} declares primary "
                    f"KV record {candidates[0]} but its DSA segment declares "
                    f"{candidates[1]}; both derive from one allocated cache"
                )
            if not candidates:
                continue
            candidate = candidates[0]
            if derived is None:
                derived = candidate
            elif derived != candidate:
                raise ValueError(
                    "GLM-5 whole-model graph primary KV record disagrees "
                    f"between layer segments: layer 0 declares {derived}, "
                    f"layer {index} declares {candidate}; one primary cache "
                    "has one record format"
                )

        explicit: tuple[int, torch.dtype] | None = None
        if (explicit_dim is None) != (explicit_dtype is None):
            raise ValueError(
                "GLM-5 whole-model graph primary KV record must be given as a "
                "dim AND a dtype or neither (got dim="
                f"{explicit_dim}, dtype={explicit_dtype}); a dim without its "
                "dtype is how a packed record ends up staged in bf16"
            )
        if explicit_dim is not None:
            explicit = (int(explicit_dim), explicit_dtype)

        if derived is not None and explicit is not None and derived != explicit:
            raise ValueError(
                "GLM-5 whole-model graph primary KV record mismatch: the "
                f"allocated cache declares {derived} but the caller passed "
                f"{explicit}"
            )
        resolved = derived or explicit
        if resolved is None:
            resolved = (int(model_source_dim), torch.bfloat16)

        record_dim, record_dtype = resolved
        if record_dim <= 0:
            raise ValueError(
                f"GLM-5 whole-model graph primary KV record dim must be > 0, "
                f"got {record_dim}"
            )
        if record_dtype not in (torch.bfloat16, torch.uint8):
            raise ValueError(
                "GLM-5 whole-model graph primary KV record must be bf16 AoS "
                "or a uint8 packed byte carrier, got "
                f"{record_dtype}"
            )
        if record_dtype is torch.uint8:
            # Mirrors the cache-side discriminator
            # (`batchgen_kernels/attention/mla_kv_packed.cache_is_packed_fp8`):
            # a byte carrier must be FlashMLA's exact 656-byte record, and the
            # latent it encodes must be the 576-element one.
            from batchgen_kernels.attention.mla_kv_packed import (
                SOURCE_DIM,
                TOKEN_BYTES,
            )

            if int(model_source_dim) != SOURCE_DIM:
                raise ValueError(
                    "the packed FP8 MLA record encodes a "
                    f"{SOURCE_DIM}-element latent, but this model's latent is "
                    f"{model_source_dim} elements"
                )
            if record_dim != TOKEN_BYTES:
                raise ValueError(
                    "a uint8 primary KV record must be FlashMLA's packed FP8 "
                    f"record of {TOKEN_BYTES} bytes per token, got "
                    f"{record_dim}"
                )
        elif record_dim != int(model_source_dim):
            raise ValueError(
                "a bf16 AoS primary KV record carries the model's "
                f"{model_source_dim}-element latent, got {record_dim}"
            )
        return record_dim, record_dtype

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
        cache_seqlens: torch.Tensor | None = None,
        primary_slot_indices: torch.Tensor | None = None,
        aux_slot_indices: torch.Tensor | None = None,
        num_valid_tokens: torch.Tensor | None = None,
        rank_token_counts: torch.Tensor | None = None,
        use_layer_segments: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden_states = self.model.model.embed_tokens(input_ids)
        outputs: dict[str, torch.Tensor] = {}
        if self.layer_segments and use_layer_segments is not False:
            required = {
                "cache_seqlens": cache_seqlens,
                "primary_slot_indices": primary_slot_indices,
                "aux_slot_indices": aux_slot_indices,
                "num_valid_tokens": num_valid_tokens,
                "rank_token_counts": rank_token_counts,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise RuntimeError(
                    "GLM-5 whole-model layer-composed graph missing inputs: "
                    f"{missing}"
                )
            for layer_idx, layer_segment in enumerate(self.layer_segments):
                graph_out = layer_segment.forward(
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    cache_seqlens=cache_seqlens,
                    primary_slot_indices=primary_slot_indices,
                    aux_slot_indices=aux_slot_indices,
                    num_valid_tokens=num_valid_tokens,
                    rank_token_counts=rank_token_counts,
                )
                hidden_states = graph_out["hidden_states"]
                self._copy_primary_kv(layer_idx, graph_out["primary_k_tensor"], None)
                # GLM-5.2 skip layers have no indexer KV (reuse-topk segments
                # return no indexer_k_tensor).
                _indexer_k = graph_out.get("indexer_k_tensor")
                if _indexer_k is not None:
                    self._copy_aux_kv(
                        layer_idx,
                        _indexer_k,
                        None,
                        graph_out.get("indexer_k_scale"),
                    )
                if layer_idx in self._compare_probe_layer_set:
                    outputs[self._probe_output_name(layer_idx)] = hidden_states[:, -1, :]
        else:
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
        num_valid_tokens: torch.Tensor | None = None,
    ) -> Mapping[str, torch.Tensor]:
        bucket_size = int(input_ids.shape[0])
        self._set_moe_bucket_state(bucket_size, rank_token_counts)

        old_cache_seqlens = AttnWrapperBase.cache_seqlens
        old_position_ids = AttnWrapperBase.position_ids
        old_max_seqlen = AttnWrapperBase.max_seqlen
        old_kv_cb = AttnWrapperBase.kv_append_callback
        old_aux_cb = AttnWrapperBase.kv_append_callback_aux
        old_dsa_short_count = GLM5AttnWrapper._dsa_short_count
        old_primary_slots = GLM5AttnWrapper.glm5_decode_primary_slot_indices
        old_aux_slots = GLM5AttnWrapper.glm5_decode_aux_slot_indices
        try:
            AttnWrapperBase.cache_seqlens = cache_seqlens
            AttnWrapperBase.position_ids = position_ids
            AttnWrapperBase.max_seqlen = self.max_seqlen
            AttnWrapperBase.kv_append_callback = self._copy_primary_kv
            AttnWrapperBase.kv_append_callback_aux = self._copy_aux_kv
            GLM5AttnWrapper._dsa_short_count = self._capture_dsa_short_count
            GLM5AttnWrapper.glm5_decode_primary_slot_indices = primary_slot_indices
            GLM5AttnWrapper.glm5_decode_aux_slot_indices = aux_slot_indices
            outputs = self.run_model_with_probes(
                input_ids=input_ids,
                position_ids=position_ids,
                cache_seqlens=cache_seqlens,
                primary_slot_indices=primary_slot_indices,
                aux_slot_indices=aux_slot_indices,
                num_valid_tokens=num_valid_tokens,
                rank_token_counts=rank_token_counts,
            )
        finally:
            AttnWrapperBase.cache_seqlens = old_cache_seqlens
            AttnWrapperBase.position_ids = old_position_ids
            AttnWrapperBase.max_seqlen = old_max_seqlen
            AttnWrapperBase.kv_append_callback = old_kv_cb
            AttnWrapperBase.kv_append_callback_aux = old_aux_cb
            GLM5AttnWrapper._dsa_short_count = old_dsa_short_count
            GLM5AttnWrapper.glm5_decode_primary_slot_indices = old_primary_slots
            GLM5AttnWrapper.glm5_decode_aux_slot_indices = old_aux_slots

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
