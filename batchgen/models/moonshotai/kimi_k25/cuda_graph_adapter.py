"""Kimi-K2.5 implementation of ``ModelCudaGraphAdapter``.

Mirrors ``glm5/cuda_graph_adapter.py`` but is simpler: K2.5 is plain MLA (no DSA
indexer / no auxiliary KV manager / no FlashMLA-metadata threading), and the
attention segment derives position from ``cache_seqlens`` internally, so replay
inputs are just ``input_ids / cache_seqlens / page_table / slot_indices /
rank_token_counts`` and no ``AttnWrapperBase`` ClassVar binding is needed at
replay (the page table is shared across layers; each layer writes its own
``_k_cache[layer_idx]`` in-graph).

``build_segments`` is self-contained: it constructs the full K2.5 stack
(per-layer ``K25AttnSegment`` + ``K25MoEGraphSegment`` → ``K25DecoderLayerGraphSegment``
→ ``K25WholeModelSegment``). No ``attach_existing_segment`` Phase-B crutch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Hashable, Iterable, List, Optional, Tuple

import torch

from batchgen.cuda_graph.adapter import (
    BatchState,
    DebugOpts,
    GraphDecision,
    GraphMode,
    ModelCudaGraphAdapter,
    SegmentBundle,
)
from batchgen.cuda_graph.buckets import generate_bucket_sizes
from batchgen.cuda_graph.composition import CaptureContext
from batchgen.cuda_graph.flags import DecodeGraphFlags
from batchgen.cuda_graph.graph_manager import (
    BatchSizeBucketing,
    CapturableSegment,
    CUDAGraphManager,
)
from batchgen.models.wrappers.attention import AttnWrapperBase

logger = logging.getLogger(__name__)

SEGMENT_NAME_WHOLE_MODEL = "k25_whole_model"


@dataclass
class _K25AdapterContext:
    model: torch.nn.Module
    bucketing: BatchSizeBucketing
    gpu_kv_manager: Any
    world_size: int
    rank: int
    device: torch.device
    max_seqlen_cap: int
    whole_model_segment: Optional[CapturableSegment]
    bundle: SegmentBundle


class KimiK25CudaGraphAdapter(ModelCudaGraphAdapter):
    """K2.5 adapter — whole-model decode graph."""

    DEFAULT_NUM_BUCKETS = 9

    def __init__(
        self,
        *,
        model_config: Any,
        engine_config: Any,
        world_size: int,
        rank: int,
    ):
        self.model_config = model_config
        self.engine_config = engine_config
        self.world_size = int(world_size)
        self.rank = int(rank)
        self._ctx: Optional[_K25AdapterContext] = None
        self._captured_signatures: Dict[Tuple[str, int], Hashable] = {}
        self._failed_buckets: set = set()
        self._capture_attempted: bool = False

    # ---- Boot-time ------------------------------------------------------

    def is_supported(self, engine_config: Any) -> bool:
        return True

    def select_buckets(self, engine_config: Any) -> List[int]:
        max_bucket = int(getattr(engine_config, "max_batch_size", None) or 32)
        num_buckets = int(
            getattr(engine_config, "cuda_graph_num_buckets", None)
            or self.DEFAULT_NUM_BUCKETS
        )
        return generate_bucket_sizes(max_bucket, num_buckets)

    def advertised_modes(self) -> List[GraphMode]:
        return [GraphMode.WHOLE_MODEL]

    def build_segments(
        self,
        *,
        model: torch.nn.Module,
        bucketing: BatchSizeBucketing,
        gpu_kv_manager: Any,
        world_size: int,
        rank: int,
        device: torch.device,
        max_seqlen_cap: int,
    ) -> SegmentBundle:
        from batchgen.models.moonshotai.kimi_k25.cuda_graph_segments import K25AttnSegment
        from batchgen.models.moonshotai.kimi_k25.layer_cuda_graph_segments import (
            K25DecoderLayerGraphSegment,
        )
        from batchgen.models.moonshotai.kimi_k25.moe_cuda_graph_segments import (
            K25MoEGraphBufferPool,
            K25MoEGraphSegment,
        )
        from batchgen.models.moonshotai.kimi_k25.whole_model_cuda_graph_segments import (
            K25WholeModelSegment,
        )

        primary_manager = getattr(gpu_kv_manager, "primary", gpu_kv_manager)
        page_size_tokens = int(primary_manager.config.page_size_tokens)
        max_rope_len = int(getattr(model.config, "max_position_embeddings", 131072))
        max_seq_len = min(int(max_seqlen_cap), max_rope_len) if max_seqlen_cap else max_rope_len
        max_pages_per_seq = (max_seq_len + page_size_tokens - 1) // page_size_tokens
        vocab_size = int(getattr(model, "vocab_size", None) or model.config.vocab_size)
        hidden_size = int(model.config.hidden_size)
        bucket_sizes = list(bucketing.bucket_sizes)
        max_bucket_size = max(bucket_sizes)

        layers = model.model.layers

        # Shared MoE graph pool for all MoE layers on this rank. Derive expert
        # geometry from the first MoE layer (layer >= first_k_dense_replace).
        moe_pool: Optional[K25MoEGraphBufferPool] = None
        for layer in layers:
            mlp = layer.mlp
            if hasattr(mlp, "experts_per_rank") and getattr(mlp, "comm", None) is not None:
                moe_pool = K25MoEGraphBufferPool(
                    world_size=world_size,
                    hidden_size=int(mlp.hidden_size),
                    num_experts=int(mlp.num_experts),
                    num_experts_per_tok=int(mlp.top_k),
                    num_local_experts=int(mlp.experts_per_rank),
                    intermediate_size=int(mlp.moe_intermediate_size),
                    device=device,
                    bucket_sizes=bucket_sizes,
                )
                break

        layer_segments: List[CapturableSegment] = []
        for layer_idx, layer in enumerate(layers):
            attn_seg = K25AttnSegment(
                layer, layer.self_attn, layer_idx,
                max_seq_len=max_seq_len,
                max_pages_per_seq=max_pages_per_seq,
                page_size_tokens=page_size_tokens,
            )
            mlp = layer.mlp
            moe_seg = None
            if hasattr(mlp, "experts_per_rank") and getattr(mlp, "comm", None) is not None:
                if moe_pool is None:
                    raise RuntimeError("K2.5 adapter: MoE pool not initialized")
                moe_seg = K25MoEGraphSegment(
                    mlp, moe_pool, mlp.comm,
                    world_size=world_size, rank=rank, device=device,
                )
            layer_segments.append(
                K25DecoderLayerGraphSegment(
                    layer=layer,
                    attn_segment=attn_seg,
                    moe_segment=moe_seg,
                    device=device,
                    world_size=world_size,
                )
            )

        # Probe layers (BATCHGEN_DECODE_GRAPH_PROBE_LAYERS) make the graph emit
        # per-layer hidden states so the eager-vs-graph compare can localize drift.
        compare_probe_layers = DecodeGraphFlags.from_env().probe_layers
        whole_segment = K25WholeModelSegment(
            model=model,
            device=device,
            world_size=world_size,
            max_pages_per_seq=max_pages_per_seq,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_bucket_size=max_bucket_size,
            layer_segments=layer_segments,
            compare_probe_layers=compare_probe_layers,
        )
        bundle = SegmentBundle(whole_model=whole_segment)

        self._ctx = _K25AdapterContext(
            model=model,
            bucketing=bucketing,
            gpu_kv_manager=gpu_kv_manager,
            world_size=world_size,
            rank=rank,
            device=device,
            max_seqlen_cap=int(max_seqlen_cap),
            whole_model_segment=whole_segment,
            bundle=bundle,
        )
        return bundle

    # ---- Capture-time ---------------------------------------------------

    def capture_signature(
        self,
        *,
        bucket: int,
        gpu_kv_manager: Any,
        max_seqlen: int,
    ) -> Hashable:
        if gpu_kv_manager is None:
            return None
        primary_manager = getattr(gpu_kv_manager, "primary", gpu_kv_manager)

        def _table_sig(manager):
            get_storage = getattr(manager, "get_cuda_graph_page_table_storage", None)
            get_graph_table = getattr(manager, "get_cuda_graph_page_table", None)
            try:
                if get_storage is not None:
                    table = get_storage()
                else:
                    table = get_graph_table() if get_graph_table is not None else None
            except RuntimeError:
                return None
            if table is None:
                return None
            return (
                int(table.data_ptr()),
                tuple(int(d) for d in table.shape),
                str(table.dtype),
                str(table.device),
            )

        return (
            SEGMENT_NAME_WHOLE_MODEL,
            int(bucket),
            int(max_seqlen),
            int(self.world_size),
            _table_sig(primary_manager),
        )

    def capture_inputs_for(
        self,
        *,
        bucket: int,
        segment_name: str,
        batch_state: BatchState,
    ) -> Dict[str, torch.Tensor]:
        """Empty-batch (batch dim 0) runtime inputs for warmup+capture.

        Batch dim 0 means no real rows are bound, so warmup leaves the static
        buffers at fill values (cache_seqlens=1, page_table=0, slot_indices=0) —
        matching the shipped per-layer K25AttnSegment warmup. ``rank_token_counts``
        is world-size shaped so the in-graph NCCL collectives are well-formed.
        """
        self._require_ctx()
        device = self._ctx.device
        max_pages = int(self._ctx.whole_model_segment.max_pages_per_seq)
        return {
            "input_ids": torch.empty((0, 1), dtype=torch.int64, device=device),
            "cache_seqlens": torch.empty((0,), dtype=torch.int32, device=device),
            "page_table": torch.empty((0, max_pages), dtype=torch.int32, device=device),
            "slot_indices": torch.empty((0,), dtype=torch.int32, device=device),
            "rank_token_counts": torch.zeros(
                (self.world_size,), dtype=torch.int64, device=device,
            ),
        }

    # ---- Step-time ------------------------------------------------------

    def eligibility(self, batch_state: BatchState) -> GraphDecision:
        if self._ctx is None:
            return GraphDecision(mode=GraphMode.EAGER, bucket=None, reason="adapter_not_built")

        max_rank_bsz = int(batch_state.max_rank_bsz)
        if max_rank_bsz <= 0:
            return GraphDecision(mode=GraphMode.EAGER, bucket=None, reason="empty_global_batch")

        try:
            bucket = self._ctx.bucketing.get_padded_size(max_rank_bsz)
        except ValueError:
            return GraphDecision(mode=GraphMode.EAGER, bucket=None, reason="over_bucket")

        if bucket in self._failed_buckets:
            return GraphDecision(mode=GraphMode.EAGER, bucket=bucket, reason="failed_bucket")

        current_max_seqlen = int(getattr(AttnWrapperBase, "max_seqlen", 0) or 0)
        captured_max_seqlen = int(self._ctx.max_seqlen_cap)
        if (
            current_max_seqlen > 0
            and captured_max_seqlen > 0
            and current_max_seqlen > captured_max_seqlen
        ):
            return GraphDecision(
                mode=GraphMode.EAGER, bucket=bucket, reason="max_seqlen_exceeds_capture",
            )

        sig = self.capture_signature(
            bucket=bucket,
            gpu_kv_manager=self._ctx.gpu_kv_manager,
            max_seqlen=captured_max_seqlen,
        )
        captured_sig = self._captured_signatures.get((SEGMENT_NAME_WHOLE_MODEL, bucket))
        if captured_sig is None:
            return GraphDecision(mode=GraphMode.EAGER, bucket=bucket, reason="bucket_not_captured")
        if captured_sig != sig:
            return GraphDecision(
                mode=GraphMode.EAGER, bucket=bucket, reason="page_table_storage_changed",
            )

        return GraphDecision(mode=GraphMode.WHOLE_MODEL, bucket=bucket, reason="captured")

    def prepare_replay_inputs(
        self,
        *,
        decision: GraphDecision,
        batch_state: BatchState,
        segment_name: str,
    ) -> Dict[str, torch.Tensor]:
        self._require_ctx()
        device = self._ctx.device
        gpu_manager = batch_state.gpu_kv_manager or self._ctx.gpu_kv_manager
        primary_manager = getattr(gpu_manager, "primary", gpu_manager)

        local_bsz = int(batch_state.local_bsz)
        active_sequence_ids = list(batch_state.cur_batch_sequence_ids or ())

        ensure_graph_table = getattr(primary_manager, "ensure_cuda_graph_page_table", None)
        if ensure_graph_table is not None:
            ensure_graph_table(active_sequence_ids)
        page_table = primary_manager.get_cuda_graph_page_table()
        slot_indices = primary_manager._gpu_page_table_manager._slot_index_tensor
        if slot_indices is None:
            slot_indices = torch.arange(local_bsz, dtype=torch.int32, device=device)

        cache_seqlens = batch_state.cache_seqlens
        if cache_seqlens is None:
            raise RuntimeError(
                "KimiK25CudaGraphAdapter.prepare_replay_inputs requires cache_seqlens"
            )
        rank_token_counts = batch_state.rank_token_counts
        if rank_token_counts is None:
            raise RuntimeError(
                "KimiK25CudaGraphAdapter.prepare_replay_inputs requires rank_token_counts"
            )
        input_ids = batch_state.input_ids
        if input_ids is None:
            raise RuntimeError(
                "KimiK25CudaGraphAdapter.prepare_replay_inputs requires input_ids"
            )

        return {
            "input_ids": input_ids[:local_bsz],
            "cache_seqlens": cache_seqlens[:local_bsz].to(dtype=torch.int32, device=device),
            "page_table": page_table[:local_bsz].to(dtype=torch.int32, device=device),
            "slot_indices": slot_indices[:local_bsz].to(dtype=torch.int32, device=device),
            "rank_token_counts": rank_token_counts,
        }

    def stage_post_graph_kv(
        self,
        *,
        decision: GraphDecision,
        batch_state: BatchState,
        graph_outputs: Dict[str, torch.Tensor],
    ) -> None:
        self._require_ctx()
        seg = self._ctx.whole_model_segment
        if seg is None:
            return
        bsz = int(batch_state.local_bsz)
        if bsz <= 0:
            return
        primary_cb = AttnWrapperBase.kv_append_callback
        primary_buf = getattr(seg, "_kv_key_buffer", None)
        if primary_cb is not None and primary_buf is not None:
            primary_stage = primary_buf[:, :bsz].clone()
            for layer_idx in range(primary_stage.shape[0]):
                primary_cb(layer_idx, primary_stage[layer_idx], None)

    # ---- Compare / debug ------------------------------------------------

    def run_eager_reference(
        self,
        *,
        segment_name: str,
        batch_state: BatchState,
        captured_inputs: Dict[str, torch.Tensor],
        probe_layers: Iterable[int] = (),
    ) -> Dict[str, torch.Tensor]:
        self._require_ctx()
        seg = self._ctx.whole_model_segment
        if seg is None:
            return {}
        # K25AttnSegment.forward reads AttnWrapperBase.gpu_paged_kv_manager
        # (cuda_graph_segments.py). Bind it so the eager reference does not rely
        # on global state being set. (The eager ref re-writes the same KV values
        # to the same paged-cache slots as the graph replay — idempotent for
        # compare; no host kv_append_callback is installed.)
        with CaptureContext(
            AttnWrapperBase,
            {"gpu_paged_kv_manager": self._ctx.gpu_kv_manager},
        ):
            outputs = seg.run_model_with_probes(
                input_ids=captured_inputs["input_ids"],
                cache_seqlens=captured_inputs["cache_seqlens"],
                page_table=captured_inputs["page_table"],
                slot_indices=captured_inputs["slot_indices"],
                rank_token_counts=captured_inputs["rank_token_counts"],
            )
        out: Dict[str, torch.Tensor] = dict(outputs)
        for layer_idx in list(probe_layers):
            probe_key_seg = seg._probe_output_name(int(layer_idx))
            if probe_key_seg in out:
                out[f"hidden_states_layer_{int(layer_idx)}"] = out.pop(probe_key_seg)
        return out

    def debug_options(self, batch_state: BatchState) -> DebugOpts:
        return DecodeGraphFlags.from_env().to_debug_opts()

    def release_all(self, *, manager: Optional[CUDAGraphManager]) -> None:
        super().release_all(manager=manager)
        self._ctx = None
        self._captured_signatures.clear()
        self._failed_buckets.clear()
        self._capture_attempted = False

    def release_context(self) -> None:
        self._ctx = None
        self._captured_signatures.clear()
        self._capture_attempted = False

    # ---- Worker hooks: record capture / failure -------------------------

    def record_capture(self, *, segment_name: str, bucket: int, signature: Hashable) -> None:
        self._captured_signatures[(segment_name, int(bucket))] = signature
        self._capture_attempted = True

    def record_capture_failure(self, *, bucket: int) -> None:
        self._failed_buckets.add(int(bucket))
        self._capture_attempted = True

    # ---- Private --------------------------------------------------------

    def _require_ctx(self) -> None:
        if self._ctx is None:
            raise RuntimeError(
                "KimiK25CudaGraphAdapter has no build context; call build_segments() first"
            )


__all__ = ["KimiK25CudaGraphAdapter", "SEGMENT_NAME_WHOLE_MODEL"]
