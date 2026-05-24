"""Model-side CUDA-graph contract.

Each model that supports full-model / per-layer / segmented CUDA-graph decode
ships a `ModelCudaGraphAdapter` implementation. The worker holds at most one
adapter (returned by the model's `Initializer.get_cuda_graph_adapter`) and
routes every graph-related decision through it.

The contract is intentionally minimal: utilities under
`batchgen.cuda_graph.composition` provide reusable building blocks so
adapters compose existing `CapturableSegment` implementations instead of
reimplementing graph wiring.

CUDA-graph activation in production is gated solely by the server-side
`--enable-cuda-graph` flag. Mode selection within an enabled adapter is
automatic: the adapter advertises the modes it has completed integration for
(`advertised_modes`), and the worker selects the highest-preferred mode
whose buckets are captured and whose runtime preconditions are met for the
current batch. Eager is always the implicit fallback.

The compare/timing/probe facilities (`DebugOpts`, `compare_decode_outputs`,
`StepTimingRecorder`) are formal infrastructure for bringing up new models:
land eager first, then port modes one-by-one with eager-vs-graph parity
proven at each step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Hashable, Iterable, List, Optional, Tuple

import torch

from batchgen.cuda_graph.graph_manager import (
    BatchSizeBucketing,
    CapturableSegment,
    CUDAGraphManager,
)


# ---------------------------------------------------------------------------
# Graph mode enum
# ---------------------------------------------------------------------------

class GraphMode(str, Enum):
    """Decode-time graph granularity for a single step.

    The order EAGER < SEGMENTED < LAYER < WHOLE_MODEL is the natural growth
    path described in the contract: a model may advertise any subset, and
    the worker prefers the highest mode whose preconditions hold.
    """
    EAGER = "eager"
    SEGMENTED = "segmented"
    LAYER = "layer"
    WHOLE_MODEL = "whole_model"


# ---------------------------------------------------------------------------
# Per-step state passed worker -> adapter
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BatchState:
    """Snapshot of the per-step decode state the adapter needs.

    Constructed once per decode step by the worker. Adapters MUST NOT reach
    into worker attributes outside what BatchState exposes: this is the only
    coupling point between the worker and the adapter for runtime decisions.
    """
    local_bsz: int
    max_rank_bsz: int
    rank_token_counts: Optional[torch.Tensor]
    cache_seqlens: Optional[torch.Tensor]
    position_ids: Optional[torch.Tensor]
    max_seqlen: int
    cur_batch_sequence_ids: Tuple[int, ...]
    gpu_kv_manager: Any
    decode_iter: int


# ---------------------------------------------------------------------------
# Mode-selection result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GraphDecision:
    """Result of `adapter.eligibility(batch_state)`.

    `reason` is a debug breadcrumb, not a control-flow input — the worker
    branches on `mode` alone. `fail_on_eager` is set when the developer has
    requested `BATCHGEN_DECODE_GRAPH_COMPARE_FAIL=1` and the adapter detects
    a mode unexpectedly fell back to EAGER.
    """
    mode: GraphMode
    bucket: Optional[int]
    reason: str = ""
    fail_on_eager: bool = False


# ---------------------------------------------------------------------------
# Developer/maintainer debug options
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DebugOpts:
    """Developer-facing observability switches.

    Parsed by `batchgen.cuda_graph.flags.DecodeGraphFlags`. MUST NOT influence
    mode selection or sampled tokens — these are observability-only by
    contract (see `cuda_graph_contract.md` §E).
    """
    compare_against_eager: bool = False
    fail_on_mismatch: bool = False
    log_path_breadcrumbs: bool = False
    timing: bool = False
    compare_atol: float = 1e-2
    compare_rtol: float = 1e-2
    probe_layers: Tuple[int, ...] = ()


# ---------------------------------------------------------------------------
# Segment bundle returned by build_segments
# ---------------------------------------------------------------------------

@dataclass
class SegmentBundle:
    """Named `CapturableSegment` instances an adapter has built.

    A model returns the subset of modes it supports. Names are namespaced by
    adapter (e.g. "glm5/whole_model", "glm5/layer_0", "glm5/attn_0"). The
    worker registers exactly these names with `CUDAGraphManager`.

    `shared_resources` holds objects that outlive a single segment (e.g.
    `Glm5MoEGraphBufferPool`) so `release_all` can drop them after teardown.
    """
    attn: Dict[str, CapturableSegment] = field(default_factory=dict)
    moe: Dict[str, CapturableSegment] = field(default_factory=dict)
    layer: Dict[str, CapturableSegment] = field(default_factory=dict)
    whole_model: Optional[CapturableSegment] = None
    shared_resources: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Adapter ABC
# ---------------------------------------------------------------------------

class ModelCudaGraphAdapter(ABC):
    """Contract every model implements to plug into the worker's graph machinery.

    Implementations live in `batchgen/models/<family>/<model>/cuda_graph_adapter.py`.
    See `batchgen_design/cuda_graph/cuda_graph_contract.md` for the full design
    and `batchgen_design/cuda_graph/MODULE.md` for the new-model bring-up
    checklist.
    """

    # ---- Boot-time ------------------------------------------------------

    @abstractmethod
    def is_supported(self, engine_config: Any) -> bool:
        """Return True if this adapter can run any non-eager mode on `engine_config`.

        Must be cheap. Called once at worker boot to decide whether to
        construct segments at all.
        """

    @abstractmethod
    def select_buckets(self, engine_config: Any) -> List[int]:
        """Bucket sizes to capture for this adapter.

        Typically derived from `engine_config.max_batch_size` via
        `batchgen.cuda_graph.buckets.generate_bucket_sizes`.
        """

    @abstractmethod
    def advertised_modes(self) -> List[GraphMode]:
        """Modes this adapter has completed integration for.

        Ordered from highest preference (WHOLE_MODEL) to lowest fallback. The
        list MUST contain at least one non-EAGER mode for the adapter to be
        useful; an adapter under development MAY advertise only LAYER or
        SEGMENTED. EAGER is the implicit terminal fallback and is not
        included in this list.

        The worker walks this list in order; the first mode whose
        `eligibility` returns a non-EAGER decision wins. Fallbacks log a
        one-shot warning per `(preferred_mode, fallback_mode, reason)`.
        """

    @abstractmethod
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
        """Construct every `CapturableSegment` the adapter advertises.

        Called once per server lifetime (re-called on hot reload). The
        returned bundle's segments are registered with `CUDAGraphManager` by
        the worker.
        """

    # ---- Capture-time ---------------------------------------------------

    @abstractmethod
    def capture_signature(
        self,
        *,
        bucket: int,
        gpu_kv_manager: Any,
        max_seqlen: int,
    ) -> Hashable:
        """Hashable fingerprint of everything material to a captured graph.

        Must include at least: page-table storage IDs, page sizes, bucket
        size, dtype, max_seqlen, world_size, num_heads. A change in the
        signature invalidates the captured graph; the worker re-captures or
        falls back to eager.
        """

    @abstractmethod
    def capture_inputs_for(
        self,
        *,
        bucket: int,
        segment_name: str,
        batch_state: BatchState,
    ) -> Dict[str, torch.Tensor]:
        """Runtime inputs to bind into static buffers before warmup+capture.

        Returned dict keys MUST match the segment's
        `get_static_input_specs(bucket)` keys exactly. Tensors are copied
        into the segment's static buffers by the manager.
        """

    # ---- Step-time ------------------------------------------------------

    @abstractmethod
    def eligibility(self, batch_state: BatchState) -> GraphDecision:
        """Pick the best mode for this step.

        Walks `advertised_modes()` in order; returns the first whose bucket
        is captured AND whose runtime preconditions are met (page-table
        storage matches `capture_signature`, `max_rank_bsz` fits within the
        max bucket, etc.).

        Returns `GraphDecision(mode=EAGER, ...)` when no captured graph
        applies. The `reason` field MUST document the rejection cause when
        the decision is EAGER or when a fallback occurred.
        """

    @abstractmethod
    def prepare_replay_inputs(
        self,
        *,
        decision: GraphDecision,
        batch_state: BatchState,
        segment_name: str,
    ) -> Dict[str, torch.Tensor]:
        """Build the kwargs dict for `manager.replay(segment_name, ...)`.

        Returned dict keys MUST match the segment's
        `get_static_input_specs(decision.bucket)` keys exactly.
        """

    @abstractmethod
    def stage_post_graph_kv(
        self,
        *,
        decision: GraphDecision,
        batch_state: BatchState,
        graph_outputs: Dict[str, torch.Tensor],
    ) -> None:
        """Stage KV writes from the captured graph into the host KV cache.

        Implementations MUST use the segment's contiguous primary/aux KV
        staging tensors (one clone per kv stream, not one per layer — see
        `cuda_graph_contract.md` §A finding #6).

        Implementations MUST invoke `AttnWrapperBase.kv_append_callback` and
        `kv_append_callback_aux` with the cloned tensors so the existing
        async D2H offload path remains intact.
        """

    # ---- Compare / debug ------------------------------------------------

    @abstractmethod
    def run_eager_reference(
        self,
        *,
        segment_name: str,
        batch_state: BatchState,
        captured_inputs: Dict[str, torch.Tensor],
        probe_layers: Iterable[int] = (),
    ) -> Dict[str, torch.Tensor]:
        """Run the same computation as the captured graph on the eager path.

        Used by `batchgen.cuda_graph.compare.compare_decode_outputs` and by
        the bring-up workflow ("land eager first, then port modes with
        compare green at each step").

        Contract:

        * MUST consume the exact `captured_inputs` dict the graph replay
          used (no test-side rewrite of inputs).
        * MUST NOT mutate KV state: implementations run against cloned KV
          views and MUST NOT call `kv_append_callback`.
        * MUST emit the same output dict keys and dtypes as the graph
          segment's `forward`.
        * When `probe_layers` is non-empty, additionally returns
          `hidden_states_layer_<i>` for each probed layer index `i`.

        Every mode in `advertised_modes()` MUST have a working eager
        reference; this is the bring-up gate's load-bearing requirement.
        """

    def debug_options(self, batch_state: BatchState) -> DebugOpts:
        """Return developer/maintainer debug options for this step.

        Default returns `DebugOpts()` (all observability off — production
        path is unchanged). Adapters override to read
        `batchgen.cuda_graph.flags.DecodeGraphFlags` plus any per-batch
        overrides from `AttnWrapperBase.batchgen_debug` whose keys match
        the `decode_graph_*` namespace.

        Adapters MAY add adapter-specific knobs but MUST NOT influence mode
        selection or sampled tokens through this method (see contract §E
        guarantee #1).
        """
        return DebugOpts()

    # ---- Release --------------------------------------------------------

    def release_all(self, *, manager: Optional[CUDAGraphManager]) -> None:
        """Drop every captured bucket and release adapter-owned buffers.

        Default implementation iterates `manager.bucketing.bucket_sizes` and
        calls `manager.drop_bucket(bucket)` — adapters override only to
        release shared resources outside any single segment's scope.
        """
        if manager is None:
            return
        bucketing = getattr(manager, "bucketing", None)
        if bucketing is None:
            return
        for bucket in list(bucketing.bucket_sizes):
            manager.drop_bucket(int(bucket))


# ---------------------------------------------------------------------------
# Initializer protocol (duck-typed; no inheritance required)
# ---------------------------------------------------------------------------

from typing import Protocol, runtime_checkable


@runtime_checkable
class HasCudaGraphAdapter(Protocol):
    """Protocol every model `*Initializer` may implement to expose an adapter.

    Existing initializers (`GLM5Initializer`, `DeepseekV3Initializer`,
    `MiniMaxM25Initializer`, etc.) do not share a base class today. Rather
    than forcing inheritance, the worker uses duck-typing:

        adapter = getattr(initializer, "get_cuda_graph_adapter", lambda: None)()

    Initializers that don't implement the method return `None` by default;
    the worker takes the eager path.
    """

    def get_cuda_graph_adapter(self) -> Optional[ModelCudaGraphAdapter]:
        ...


__all__ = [
    "GraphMode",
    "BatchState",
    "GraphDecision",
    "DebugOpts",
    "SegmentBundle",
    "ModelCudaGraphAdapter",
    "HasCudaGraphAdapter",
]
