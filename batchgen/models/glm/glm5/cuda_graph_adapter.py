"""GLM-5 reference implementation of `ModelCudaGraphAdapter`.

Production-equivalent: lifts logic from the worker's `_glm5_*` methods so the
adapter is the single source of truth for capture signature, eligibility,
replay-input preparation, and post-graph KV staging. The legacy worker
methods stay for Phase B's dual-path safety net (gated by
`BATCHGEN_DECODE_GRAPH_ADAPTER_DUAL=1`) and are deleted in Phase C.

Logic moved from `batchgen_worker.py`:
  - `_glm5_whole_model_graph_capture_signature` (9823) -> `capture_signature`
  - `_glm5_whole_graph_path_state` (9948) -> `eligibility`
  - `_prepare_glm5_layer_graph_inputs` (10253) -> `prepare_replay_inputs`
  - `_make_glm5_whole_model_capture_inputs` (10533) -> `capture_inputs_for`
  - post-replay KV-callback block (12246-12295) -> `stage_post_graph_kv`
  - eager-compare path (12162-12241) wraps `run_eager_reference`
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


SEGMENT_NAME_WHOLE_MODEL = "glm5_whole_model"


@dataclass
class _Glm5AdapterContext:
    """Stashed state from `build_segments` so step-time methods have it
    without the worker passing it in repeatedly.
    """
    model: torch.nn.Module
    bucketing: BatchSizeBucketing
    gpu_kv_manager: Any
    world_size: int
    rank: int
    device: torch.device
    max_seqlen_cap: int
    whole_model_segment: Optional[CapturableSegment]
    bundle: SegmentBundle
    num_heads: int
    index_topk: int


class Glm5CudaGraphAdapter(ModelCudaGraphAdapter):
    """GLM-5 adapter — full-model graph (today's release path).

    Constructor consumes fields the `GLM5Initializer` already owns.
    """

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
        self._ctx: Optional[_Glm5AdapterContext] = None
        # Worker informs the adapter when a bucket is successfully captured;
        # eligibility() consults this to decide if a fallback is needed.
        self._captured_signatures: Dict[Tuple[str, int], Hashable] = {}
        self._failed_buckets: set = set()
        self._capture_attempted: bool = False
        self._fallback_warned: set = set()
        self._state_change_logged: bool = False

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

    def attach_existing_segment(
        self,
        *,
        model: torch.nn.Module,
        whole_model_segment: CapturableSegment,
        bucketing: BatchSizeBucketing,
        gpu_kv_manager: Any,
        device: torch.device,
        max_seqlen_cap: int,
    ) -> None:
        """Adopt a `Glm5WholeModelSegment` already built and captured by the
        worker. Populates `_ctx` so `eligibility()` / `prepare_replay_inputs()`
        / `stage_post_graph_kv()` can run.

        Used by `_setup_cuda_graphs` in `batchgen_worker.py` instead of
        `build_segments(...)` so the adapter shares the captured segment with
        the legacy code path during the Phase B dual-mode period. When Phase
        C deletes the legacy capture code, `build_segments` becomes the only
        entry point and this method is removed.
        """
        attn0 = model.model.layers[0].self_attn.module
        indexer0 = getattr(attn0, "indexer", None)
        if indexer0 is None:
            raise RuntimeError(
                "Glm5CudaGraphAdapter.attach_existing_segment: DSA indexer missing"
            )
        bundle = SegmentBundle(whole_model=whole_model_segment)
        self._ctx = _Glm5AdapterContext(
            model=model,
            bucketing=bucketing,
            gpu_kv_manager=gpu_kv_manager,
            world_size=self.world_size,
            rank=self.rank,
            device=device,
            max_seqlen_cap=int(max_seqlen_cap),
            whole_model_segment=whole_model_segment,
            bundle=bundle,
            num_heads=int(attn0.num_heads),
            index_topk=int(indexer0.index_topk),
        )

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
        from batchgen.models.glm.glm5.whole_model_cuda_graph_segments import (
            Glm5WholeModelSegment,
        )

        attn0 = model.model.layers[0].self_attn.module
        indexer0 = getattr(attn0, "indexer", None)
        if indexer0 is None:
            raise RuntimeError(
                "Glm5CudaGraphAdapter requires DSA indexer; this model is not GLM-5-DSA"
            )

        primary_manager = getattr(gpu_kv_manager, "primary", gpu_kv_manager)
        aux_manager = getattr(gpu_kv_manager, "auxiliary", None)
        if aux_manager is None:
            raise RuntimeError("Glm5CudaGraphAdapter requires auxiliary GPU KV manager")

        max_pages_per_seq = int(primary_manager.max_pages_per_sequence)
        max_aux_pages_per_seq = int(aux_manager.max_pages_per_sequence)
        vocab_size = int(model.config.vocab_size)
        hidden_size = int(model.config.hidden_size)
        num_heads = int(attn0.num_heads)
        index_topk = int(indexer0.index_topk)

        whole_segment = Glm5WholeModelSegment(
            model=model,
            device=device,
            world_size=world_size,
            max_pages_per_seq=max_pages_per_seq,
            max_aux_pages_per_seq=max_aux_pages_per_seq,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_bucket_size=max(bucketing.bucket_sizes),
            max_seqlen=max_seqlen_cap,
        )
        bundle = SegmentBundle(whole_model=whole_segment)

        self._ctx = _Glm5AdapterContext(
            model=model,
            bucketing=bucketing,
            gpu_kv_manager=gpu_kv_manager,
            world_size=world_size,
            rank=rank,
            device=device,
            max_seqlen_cap=max_seqlen_cap,
            whole_model_segment=whole_segment,
            bundle=bundle,
            num_heads=num_heads,
            index_topk=index_topk,
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
        """Page-table-storage fingerprint plus world_size/bucket/max_seqlen.

        Lifted from `_glm5_whole_model_graph_capture_signature` (worker:9823)
        with the additional invariants required to invalidate the graph on
        any of: world_size change, bucket-set change, max_seqlen change, or
        head-count change (audit §A finding #5).
        """
        if gpu_kv_manager is None:
            return None
        primary_manager = getattr(gpu_kv_manager, "primary", gpu_kv_manager)
        aux_manager = getattr(gpu_kv_manager, "auxiliary", None)
        if aux_manager is None:
            return None

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
                tuple(int(dim) for dim in table.shape),
                str(table.dtype),
                str(table.device),
            )

        num_heads = int(self._ctx.num_heads) if self._ctx else 0
        return (
            SEGMENT_NAME_WHOLE_MODEL,
            int(bucket),
            int(max_seqlen),
            int(self.world_size),
            int(num_heads),
            _table_sig(primary_manager),
            _table_sig(aux_manager),
        )

    def capture_inputs_for(
        self,
        *,
        bucket: int,
        segment_name: str,
        batch_state: BatchState,
    ) -> Dict[str, torch.Tensor]:
        """Empty-but-shape-correct runtime inputs for warmup+capture.

        Mirrors `_make_glm5_whole_model_capture_inputs` (worker:10533): the
        captured graph is forward-only and warmup MUST NOT mutate the real
        KV cache, so the bound inputs have batch dim 0 (capture-time stub)
        except for `rank_token_counts` which is world-size-shaped.
        """
        self._require_ctx()
        from batchgen.attention.dsa.sparse_decode_mla import (
            prepare_sparse_flash_mla_decode_tensor_metadata,
        )

        device = self._ctx.device
        num_heads = self._ctx.num_heads
        selected_lengths = torch.ones((int(bucket),), dtype=torch.int32, device=device)
        tile_scheduler_metadata, num_splits = prepare_sparse_flash_mla_decode_tensor_metadata(
            selected_lengths, int(num_heads),
        )
        return {
            "input_ids": torch.empty((0, 1), dtype=torch.int64, device=device),
            "cache_seqlens": torch.empty((0,), dtype=torch.int32, device=device),
            "position_ids": torch.empty((0, 1), dtype=torch.int64, device=device),
            "primary_slot_indices": torch.empty((0,), dtype=torch.int32, device=device),
            "aux_slot_indices": torch.empty((0,), dtype=torch.int32, device=device),
            "rank_token_counts": torch.zeros(
                (self.world_size,), dtype=torch.int64, device=device,
            ),
            "num_valid_tokens": torch.zeros((1,), dtype=torch.int32, device=device),
            "flashmla_tile_scheduler_metadata": tile_scheduler_metadata,
            "flashmla_num_splits": num_splits,
        }

    # ---- Step-time ------------------------------------------------------

    def eligibility(self, batch_state: BatchState) -> GraphDecision:
        """Single state machine replacing the 4 `_glm5_*_path_state` methods.

        Lifted from `_glm5_whole_graph_path_state` (worker:9948).
        """
        if self._ctx is None:
            return GraphDecision(mode=GraphMode.EAGER, bucket=None, reason="adapter_not_built")

        max_rank_bsz = int(batch_state.max_rank_bsz)
        if max_rank_bsz <= 0:
            return GraphDecision(mode=GraphMode.EAGER, bucket=None, reason="empty_global_batch")

        bucketing = self._ctx.bucketing
        try:
            bucket = bucketing.get_padded_size(max_rank_bsz)
        except ValueError:
            return GraphDecision(mode=GraphMode.EAGER, bucket=None, reason="over_bucket")

        if bucket in self._failed_buckets:
            return GraphDecision(mode=GraphMode.EAGER, bucket=bucket, reason="failed_bucket")

        # max_seqlen check: AttnWrapperBase.max_seqlen reflects current decode
        # max_seqlen; segment.max_seqlen is what was captured.
        current_max_seqlen = int(getattr(AttnWrapperBase, "max_seqlen", 0) or 0)
        captured_max_seqlen = int(self._ctx.max_seqlen_cap)
        if (
            current_max_seqlen > 0
            and captured_max_seqlen > 0
            and current_max_seqlen > captured_max_seqlen
        ):
            return GraphDecision(
                mode=GraphMode.EAGER, bucket=bucket,
                reason="max_seqlen_exceeds_capture",
            )

        # Capture-signature check.
        sig = self.capture_signature(
            bucket=bucket,
            gpu_kv_manager=self._ctx.gpu_kv_manager,
            max_seqlen=captured_max_seqlen,
        )
        captured_sig = self._captured_signatures.get((SEGMENT_NAME_WHOLE_MODEL, bucket))
        if captured_sig is None:
            return GraphDecision(
                mode=GraphMode.EAGER, bucket=bucket, reason="bucket_not_captured",
            )
        if captured_sig != sig:
            return GraphDecision(
                mode=GraphMode.EAGER, bucket=bucket,
                reason="page_table_storage_changed",
            )

        return GraphDecision(mode=GraphMode.WHOLE_MODEL, bucket=bucket, reason="captured")

    def prepare_replay_inputs(
        self,
        *,
        decision: GraphDecision,
        batch_state: BatchState,
        segment_name: str,
    ) -> Dict[str, torch.Tensor]:
        """Build the kwargs dict for `manager.replay(SEGMENT_NAME_WHOLE_MODEL, ...)`.

        Lifted from `_prepare_glm5_layer_graph_inputs` (worker:10253) +
        the inline replay-arg construction at worker:12104-12124. Includes
        `input_ids` and `rank_token_counts` so the worker passes exactly
        this dict to `manager.replay`.
        """
        self._require_ctx()
        from batchgen.attention.dsa.sparse_decode_mla import (
            prepare_sparse_flash_mla_decode_tensor_metadata,
        )

        device = self._ctx.device
        gpu_manager = batch_state.gpu_kv_manager or self._ctx.gpu_kv_manager
        primary_manager = getattr(gpu_manager, "primary", gpu_manager)
        aux_manager = getattr(gpu_manager, "auxiliary", None)
        if aux_manager is None:
            raise RuntimeError(
                "Glm5CudaGraphAdapter.prepare_replay_inputs requires auxiliary GPU KV manager"
            )

        bucket = int(decision.bucket or 0)
        local_bsz = int(batch_state.local_bsz)
        active_sequence_ids = list(batch_state.cur_batch_sequence_ids or ())

        def _graph_slots(manager):
            ensure_graph_table = getattr(manager, "ensure_cuda_graph_page_table", None)
            if ensure_graph_table is not None:
                ensure_graph_table(active_sequence_ids)
            slot_indices = manager._gpu_page_table_manager._slot_index_tensor
            if slot_indices is None:
                slot_indices = torch.arange(
                    local_bsz, dtype=torch.int32, device=device,
                )
            return slot_indices[:local_bsz].to(dtype=torch.int32, device=device)

        cache_seqlens = batch_state.cache_seqlens
        position_ids = batch_state.position_ids
        if cache_seqlens is None or position_ids is None:
            raise RuntimeError(
                "Glm5CudaGraphAdapter.prepare_replay_inputs requires cache_seqlens and position_ids"
            )
        if local_bsz > 0:
            cache_seqlens_i32 = cache_seqlens[:local_bsz].to(dtype=torch.int32, device=device)
            position_ids_i64 = position_ids[:local_bsz].to(dtype=torch.int64, device=device)
        else:
            cache_seqlens_i32 = torch.empty((0,), dtype=torch.int32, device=device)
            position_ids_i64 = torch.empty((0, 1), dtype=torch.int64, device=device)

        index_topk = int(self._ctx.index_topk)
        num_heads = int(self._ctx.num_heads)
        graph_max_seqlen = int(
            self._ctx.max_seqlen_cap
            or getattr(AttnWrapperBase, "max_seqlen", 0)
            or index_topk
        )
        selected_lengths = torch.empty((bucket,), dtype=torch.int32, device=device)
        if local_bsz > 0:
            selected_lengths[:local_bsz].copy_(
                torch.clamp(cache_seqlens_i32, max=index_topk),
                non_blocking=True,
            )
        if local_bsz < bucket:
            selected_lengths[local_bsz:].fill_(min(graph_max_seqlen, index_topk))

        tile_scheduler_metadata, num_splits = prepare_sparse_flash_mla_decode_tensor_metadata(
            selected_lengths, num_heads,
        )
        num_valid_tokens = torch.empty((1,), dtype=torch.int32, device=device)
        num_valid_tokens.fill_(local_bsz)

        rank_token_counts = batch_state.rank_token_counts
        if rank_token_counts is None:
            raise RuntimeError(
                "Glm5CudaGraphAdapter.prepare_replay_inputs requires rank_token_counts "
                "(globally-synced count per rank)"
            )
        input_ids = batch_state.input_ids
        if input_ids is None:
            raise RuntimeError(
                "Glm5CudaGraphAdapter.prepare_replay_inputs requires input_ids "
                "(next-token tensor for this step)"
            )

        return {
            "input_ids": input_ids[:local_bsz],
            "cache_seqlens": cache_seqlens_i32,
            "position_ids": position_ids_i64,
            "primary_slot_indices": _graph_slots(primary_manager),
            "aux_slot_indices": _graph_slots(aux_manager),
            "rank_token_counts": rank_token_counts,
            "num_valid_tokens": num_valid_tokens,
            "flashmla_tile_scheduler_metadata": tile_scheduler_metadata,
            "flashmla_num_splits": num_splits,
        }

    def stage_post_graph_kv(
        self,
        *,
        decision: GraphDecision,
        batch_state: BatchState,
        graph_outputs: Dict[str, torch.Tensor],
    ) -> None:
        """Single-clone contiguous KV staging.

        Lifted from worker:12246-12295 with audit §A finding #6 applied:
        primary/aux contiguous clones only — no per-layer fallback branch.
        """
        self._require_ctx()
        seg = self._ctx.whole_model_segment
        if seg is None:
            return
        bsz = int(batch_state.local_bsz)
        if bsz <= 0:
            return

        primary_cb = AttnWrapperBase.kv_append_callback
        aux_cb = AttnWrapperBase.kv_append_callback_aux

        primary_buf = getattr(seg, "_kv_key_buffer", None)
        if primary_cb is not None and primary_buf is not None:
            primary_stage = primary_buf[:, :bsz].clone()
            num_layers = primary_stage.shape[0]
            for layer_idx in range(num_layers):
                primary_cb(layer_idx, primary_stage[layer_idx], None)

        aux_buf = getattr(seg, "_aux_kv_key_buffer", None)
        if aux_cb is not None and aux_buf is not None:
            aux_stage = aux_buf[:, :bsz].clone()
            num_layers = aux_stage.shape[0]
            for layer_idx in range(num_layers):
                aux_cb(layer_idx, aux_stage[layer_idx], None)

    # ---- Compare / debug ------------------------------------------------

    def run_eager_reference(
        self,
        *,
        segment_name: str,
        batch_state: BatchState,
        captured_inputs: Dict[str, torch.Tensor],
        probe_layers: Iterable[int] = (),
    ) -> Dict[str, torch.Tensor]:
        """Eager-path forward via `Glm5WholeModelSegment.run_model_with_probes`.

        Binds the exact ClassVars the graph forward expects via CaptureContext
        (audit §A findings #8/#9). Does NOT install kv_append_callback — eager
        path does not stage KV; the worker's post-replay path stages from the
        graph buffers only.
        """
        self._require_ctx()
        seg = self._ctx.whole_model_segment
        if seg is None:
            return {}

        ctx_attrs: Dict[str, Any] = {
            "cache_seqlens": captured_inputs.get("cache_seqlens"),
            "position_ids": captured_inputs.get("position_ids"),
            "max_seqlen": int(self._ctx.max_seqlen_cap),
            "kv_append_callback": None,
            "kv_append_callback_aux": None,
        }
        with CaptureContext(AttnWrapperBase, ctx_attrs):
            outputs = seg.run_model_with_probes(
                input_ids=captured_inputs["input_ids"],
                position_ids=captured_inputs["position_ids"],
                cache_seqlens=captured_inputs.get("cache_seqlens"),
                primary_slot_indices=captured_inputs.get("primary_slot_indices"),
                aux_slot_indices=captured_inputs.get("aux_slot_indices"),
                num_valid_tokens=captured_inputs.get("num_valid_tokens"),
                rank_token_counts=captured_inputs.get("rank_token_counts"),
                flashmla_tile_scheduler_metadata=captured_inputs.get(
                    "flashmla_tile_scheduler_metadata"
                ),
                flashmla_num_splits=captured_inputs.get("flashmla_num_splits"),
                use_layer_segments=False,
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
        self._fallback_warned.clear()
        self._state_change_logged = False

    # ---- Capture-context attrs (worker wraps replay with CaptureContext) ---

    def capture_context_attrs(
        self,
        *,
        replay_inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """ClassVars to bind on `AttnWrapperBase` for a graph replay window.

        Replaces the inline save/restore at
        `whole_model_cuda_graph_segments.py:394-430` (audit §A finding #9).
        """
        self._require_ctx()
        seg = self._ctx.whole_model_segment
        return {
            "cache_seqlens": replay_inputs.get("cache_seqlens"),
            "position_ids": replay_inputs.get("position_ids"),
            "max_seqlen": int(self._ctx.max_seqlen_cap),
            "kv_append_callback": getattr(seg, "_copy_primary_kv", None) if seg is not None else None,
            "kv_append_callback_aux": getattr(seg, "_copy_aux_kv", None) if seg is not None else None,
            "_dsa_short_count": getattr(seg, "_capture_dsa_short_count", None) if seg is not None else None,
            "glm5_decode_primary_slot_indices": replay_inputs.get("primary_slot_indices"),
            "glm5_decode_aux_slot_indices": replay_inputs.get("aux_slot_indices"),
        }

    # ---- Worker hooks (Phase B): record capture / failure -----

    def record_capture(self, *, segment_name: str, bucket: int, signature: Hashable) -> None:
        """Worker calls after a successful warmup+capture so `eligibility`
        knows the bucket is ready and pins the signature snapshot.
        """
        self._captured_signatures[(segment_name, int(bucket))] = signature
        self._capture_attempted = True

    def record_capture_failure(self, *, bucket: int) -> None:
        self._failed_buckets.add(int(bucket))
        self._capture_attempted = True

    # ---- Private helpers ------------------------------------------------

    def _require_ctx(self) -> None:
        if self._ctx is None:
            raise RuntimeError(
                "Glm5CudaGraphAdapter has no build context; call build_segments() first"
            )


__all__ = ["Glm5CudaGraphAdapter", "SEGMENT_NAME_WHOLE_MODEL"]
