"""GLM-5 reference implementation of `ModelCudaGraphAdapter`.

This is the Phase-A SHELL: it implements every ABC method by delegating to the
existing GLM-5 segment classes (`Glm5WholeModelSegment`,
`Glm5DecoderLayerGraphSegment`, `Glm5FullDsaAttnSegment`, `Glm5MoEGraphSegment`)
and uses `CaptureContext` from `batchgen.cuda_graph.composition` instead of
the inline ClassVar monkey-patch at
`whole_model_cuda_graph_segments.py:394-430` (audit §A finding #9).

Phase A leaves the worker's `_glm5_*` private methods in place; nothing in
`batchgen_worker.py` calls this adapter yet. Phase B introduces a dual-path
gate that lets the worker exercise this adapter while keeping the legacy
path as a safety net. Phase C deletes the legacy worker methods after the
parity gate is green.

Production-detailed implementations of `capture_inputs_for`,
`prepare_replay_inputs`, and `stage_post_graph_kv` are LIFTED from the
worker in Phase B (concretely: `_make_glm5_whole_model_capture_inputs`
worker:10533, `_prepare_glm5_layer_graph_inputs` worker:10253, the
post-graph KV-callback block worker:12246-12295). For Phase A we provide
working implementations sufficient for unit tests at small dimensions; the
production lift is a behavior-preserving move.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Tuple

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


@dataclass
class _Glm5AdapterContext:
    """Stashed state from `build_segments` so step-time methods don't need
    the worker to pass it in repeatedly."""
    model: torch.nn.Module
    bucketing: BatchSizeBucketing
    gpu_kv_manager: Any
    world_size: int
    rank: int
    device: torch.device
    max_seqlen_cap: int
    whole_model_segment: Optional[CapturableSegment]
    bundle: SegmentBundle


class Glm5CudaGraphAdapter(ModelCudaGraphAdapter):
    """GLM-5 adapter — full-model graph (today's release path).

    Constructor consumes the same fields the `GLM5Initializer` already owns:
    `model_config`, `engine_config`, `world_size`, `rank`.
    """

    SEGMENT_NAME_WHOLE_MODEL = "glm5/whole_model"
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
        self._captured_signatures: Dict[Tuple[str, int], Hashable] = {}
        self._fallback_warned: set = set()

    # ---- Boot-time ------------------------------------------------------

    def is_supported(self, engine_config: Any) -> bool:
        return True

    def select_buckets(self, engine_config: Any) -> List[int]:
        max_bucket = int(getattr(engine_config, "max_batch_size", 32) or 32)
        num_buckets = int(getattr(engine_config, "cuda_graph_num_buckets", self.DEFAULT_NUM_BUCKETS) or self.DEFAULT_NUM_BUCKETS)
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
        """Construct the Glm5WholeModelSegment via the existing constructor.

        Phase A delegates to the production segment class verbatim. Phase B/C
        cleanup focuses on the worker; the segment class itself stays the
        same (it has no audit findings against it that the contract needs to
        address before lift).
        """
        from batchgen.models.glm.glm5.whole_model_cuda_graph_segments import (
            Glm5WholeModelSegment,
        )

        attn0 = model.model.layers[0].self_attn.module
        indexer0 = getattr(attn0, "indexer", None)
        if indexer0 is None:
            raise RuntimeError(
                "Glm5CudaGraphAdapter requires DSA indexer; this model is not GLM-5-DSA"
            )
        max_pages_per_seq = int(gpu_kv_manager.max_pages_per_sequence)
        max_aux_pages_per_seq = int(
            getattr(gpu_kv_manager, "aux_max_pages_per_sequence", max_pages_per_seq)
        )
        vocab_size = int(model.config.vocab_size)
        hidden_size = int(model.config.hidden_size)

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
        """Fingerprint capture-time invariants (audit §A finding #5: include
        world_size, bucket, max_seqlen, num_heads — not just page-table storage).
        """
        primary_storage = self._page_table_storage_id(gpu_kv_manager, "primary")
        aux_storage = self._page_table_storage_id(gpu_kv_manager, "aux")
        num_heads = int(self._ctx.model.model.layers[0].self_attn.module.num_heads) if self._ctx else 0
        return (
            "glm5/whole_model",
            int(bucket),
            int(max_seqlen),
            int(self.world_size),
            int(num_heads),
            primary_storage,
            aux_storage,
        )

    def capture_inputs_for(
        self,
        *,
        bucket: int,
        segment_name: str,
        batch_state: BatchState,
    ) -> Dict[str, torch.Tensor]:
        """Runtime inputs to bind into static buffers before warmup+capture.

        Production logic lives in worker `_make_glm5_whole_model_capture_inputs`
        (worker:10533). In Phase B that body lifts here verbatim. For the
        Phase A shell, we build a minimal valid input dict from `batch_state`;
        unit tests at small dims exercise this path.
        """
        self._require_ctx()
        seg = self._ctx.whole_model_segment
        specs = seg.get_static_input_specs(bucket)
        out: Dict[str, torch.Tensor] = {}
        for name, spec in specs.items():
            shape = spec.resolve_shape(bucket)
            out[name] = torch.full(shape, spec.fill_value, dtype=spec.dtype, device=self._ctx.device)
        # Overlay the runtime tensors we have on hand.
        if batch_state.cache_seqlens is not None and "cache_seqlens" in out:
            n = min(batch_state.cache_seqlens.shape[0], out["cache_seqlens"].shape[0])
            out["cache_seqlens"][:n].copy_(batch_state.cache_seqlens[:n].to(out["cache_seqlens"].dtype))
        if batch_state.position_ids is not None and "position_ids" in out:
            n = min(batch_state.position_ids.shape[0], out["position_ids"].shape[0])
            out["position_ids"][:n].copy_(batch_state.position_ids[:n].to(out["position_ids"].dtype))
        if batch_state.rank_token_counts is not None and "rank_token_counts" in out:
            out["rank_token_counts"].copy_(
                batch_state.rank_token_counts.to(out["rank_token_counts"].dtype)
            )
        return out

    # ---- Step-time ------------------------------------------------------

    def eligibility(self, batch_state: BatchState) -> GraphDecision:
        """Single state machine replacing the 4 `_glm5_*_path_state` methods.

        Walks advertised modes; returns the first whose preconditions hold.
        """
        if self._ctx is None:
            return GraphDecision(mode=GraphMode.EAGER, bucket=None, reason="adapter_not_built")
        bucketing = self._ctx.bucketing
        if batch_state.max_rank_bsz > bucketing.bucket_sizes[-1]:
            return GraphDecision(
                mode=GraphMode.EAGER, bucket=None,
                reason=f"batch_size={batch_state.max_rank_bsz} exceeds max_bucket={bucketing.bucket_sizes[-1]}",
            )
        bucket = bucketing.get_padded_size(batch_state.max_rank_bsz)
        preferred = self.advertised_modes()[0]
        for mode in self.advertised_modes():
            ok, reason = self._mode_preconditions(mode, bucket, batch_state)
            if ok:
                if mode is not preferred:
                    key = (preferred, mode, reason)
                    if key not in self._fallback_warned:
                        logger.warning(
                            "glm5 adapter fallback %s -> %s: %s",
                            preferred.value, mode.value, reason,
                        )
                        self._fallback_warned.add(key)
                return GraphDecision(mode=mode, bucket=bucket, reason=reason or "ok")
        return GraphDecision(mode=GraphMode.EAGER, bucket=None, reason="no_mode_ready")

    def prepare_replay_inputs(
        self,
        *,
        decision: GraphDecision,
        batch_state: BatchState,
        segment_name: str,
    ) -> Dict[str, torch.Tensor]:
        """Build kwargs for `manager.replay`. Phase B lifts the production
        body from `_prepare_glm5_layer_graph_inputs` (worker:10253). The
        shell reuses `capture_inputs_for` since the input-shape contract is
        identical between capture and replay for GLM-5.
        """
        return self.capture_inputs_for(
            bucket=decision.bucket or 0,
            segment_name=segment_name,
            batch_state=batch_state,
        )

    def stage_post_graph_kv(
        self,
        *,
        decision: GraphDecision,
        batch_state: BatchState,
        graph_outputs: Dict[str, torch.Tensor],
    ) -> None:
        """Single-clone contiguous KV staging (audit §A finding #6 — no
        per-layer fallback branch).

        Phase B lifts the production loop from worker:12246-12295; the shell
        defers to whatever `kv_append_callback` is registered today.
        """
        self._require_ctx()
        seg = self._ctx.whole_model_segment
        bsz = int(batch_state.local_bsz)
        kv_primary_buf = getattr(seg, "_kv_key_buffer", None)
        kv_aux_buf = getattr(seg, "_aux_kv_key_buffer", None)
        primary_cb = AttnWrapperBase.kv_append_callback
        aux_cb = AttnWrapperBase.kv_append_callback_aux
        if kv_primary_buf is not None and primary_cb is not None:
            primary_stage = kv_primary_buf[:, :bsz].clone()
            for layer_idx in range(primary_stage.shape[0]):
                primary_cb(layer_idx, primary_stage[layer_idx], None)
        if kv_aux_buf is not None and aux_cb is not None:
            aux_stage = kv_aux_buf[:, :bsz].clone()
            for layer_idx in range(aux_stage.shape[0]):
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
        """Eager-path forward using `Glm5WholeModelSegment.run_model_with_probes`.

        Contract guarantees enforced (§B run_eager_reference):
          - Same `captured_inputs` consumed verbatim.
          - KV state not mutated (we DO NOT install kv_append_callback for
            the eager run; the segment writes to its own staging buffers
            which we discard).
          - Output keys match the graph segment's outputs.
        """
        self._require_ctx()
        seg = self._ctx.whole_model_segment
        # Bind cache_seqlens / position_ids / max_seqlen via CaptureContext so
        # we restore the worker's previous values on exit. NO kv_append_callback.
        ctx_attrs = {
            "cache_seqlens": captured_inputs.get("cache_seqlens"),
            "position_ids": captured_inputs.get("position_ids"),
            "max_seqlen": int(self._ctx.max_seqlen_cap),
            "kv_append_callback": None,
            "kv_append_callback_aux": None,
        }
        probe_list = list(probe_layers)
        with CaptureContext(AttnWrapperBase, ctx_attrs):
            outputs = seg.run_model_with_probes(
                input_ids=captured_inputs["input_ids"],
                position_ids=captured_inputs["position_ids"],
                cache_seqlens=captured_inputs.get("cache_seqlens"),
                primary_slot_indices=captured_inputs.get("primary_slot_indices"),
                aux_slot_indices=captured_inputs.get("aux_slot_indices"),
                num_valid_tokens=captured_inputs.get("num_valid_tokens"),
                rank_token_counts=captured_inputs.get("rank_token_counts"),
                flashmla_tile_scheduler_metadata=captured_inputs.get("flashmla_tile_scheduler_metadata"),
                flashmla_num_splits=captured_inputs.get("flashmla_num_splits"),
                use_layer_segments=False,
            )
        out: Dict[str, torch.Tensor] = dict(outputs)
        for layer_idx in probe_list:
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
        self._fallback_warned.clear()

    # ---- Private helpers ------------------------------------------------

    def _require_ctx(self) -> None:
        if self._ctx is None:
            raise RuntimeError(
                "Glm5CudaGraphAdapter has no build context; call build_segments() first"
            )

    def _mode_preconditions(
        self, mode: GraphMode, bucket: int, batch_state: BatchState,
    ) -> Tuple[bool, str]:
        """Single source of truth replacing _glm5_{dsa,layer,moe,whole}_current_bucket_missing.

        For Phase A the adapter only advertises WHOLE_MODEL; preconditions
        check that the captured bucket exists and the capture signature
        matches the current page-table storage.
        """
        if mode is GraphMode.WHOLE_MODEL:
            if self._ctx.whole_model_segment is None:
                return False, "whole_model_segment_missing"
            sig = self.capture_signature(
                bucket=bucket,
                gpu_kv_manager=self._ctx.gpu_kv_manager,
                max_seqlen=self._ctx.max_seqlen_cap,
            )
            captured_sig = self._captured_signatures.get((self.SEGMENT_NAME_WHOLE_MODEL, bucket))
            if captured_sig is None:
                return False, f"bucket={bucket}_not_captured"
            if captured_sig != sig:
                return False, f"capture_signature_mismatch_bucket={bucket}"
            return True, ""
        return False, f"mode={mode.value}_not_advertised"

    @staticmethod
    def _page_table_storage_id(gpu_kv_manager: Any, kind: str) -> int:
        """Stable ID for a page-table backing storage; None-safe for tests."""
        if gpu_kv_manager is None:
            return 0
        get = getattr(gpu_kv_manager, "get_cuda_graph_page_table_storage", None)
        if not callable(get):
            return 0
        try:
            storage = get(kind) if get.__code__.co_argcount >= 2 else get()
        except TypeError:
            storage = get()
        if storage is None:
            return 0
        return int(storage.data_ptr())

    def record_capture(self, *, segment_name: str, bucket: int, signature: Hashable) -> None:
        """Worker calls this after a successful capture so eligibility can
        consult `_captured_signatures`. Phase B wires this from
        `_setup_decode_graphs` (the post-capture acknowledgement path).
        """
        self._captured_signatures[(segment_name, int(bucket))] = signature


__all__ = ["Glm5CudaGraphAdapter"]
