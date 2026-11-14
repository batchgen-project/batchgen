"""Decode CUDA-graph state of one ``BatchGenWorker`` rank.

``GraphRuntime`` exclusively owns the 29 pieces of coupled graph state that
used to be loose worker attributes: the manager/adapter handles, the
whole-model capture flags (model-agnostic and GLM-5), the capture-once
counter, the span-warn rate limiter, and the Phase-C residue fields that are
still initialized and reset by the worker's teardown.

It also owns every ``batchgen.cuda_graph`` import the worker used to make:
the manager/bucketing constructors, the adapter ABC used by the startup type
backstop, the per-step ``BatchState`` snapshot and the model-agnostic
eager-vs-graph compare. The one exception is the decode bucket ladder
(``batchgen.cuda_graph.buckets``: ``decode_graph_bucket_sizes``,
``check_decode_width_within_buckets``, ``DecodeWidthExceedsGraphBucketError``),
which the worker imports directly since the rebase onto main 1c372209;
routing it through this class is a follow-up, not part of the rebase.

What this class is NOT, in this slice: a capture/replay engine. The capture,
eligibility, and replay logic stays in the worker and reads this object's
fields; only the state (and the imports) moved. The deeper move of that logic
behind ``ModelCudaGraphAdapter`` is a later slice -- it is not behavior-neutral
(GPT-OSS ships no adapter, admission reads graph state, and the worker's seam
is wider than the ABC).

Each ``batchgen.cuda_graph`` import below sits inside the method that needs
it, exactly where the worker made it: the imports are late-bound on purpose
(a test monkeypatches ``batchgen.cuda_graph.CUDAGraphManager`` between calls),
and module-level imports here would silently break that binding.

Imports: stdlib + ``torch``-level ``batchgen.cuda_graph`` (function-local).
"""

from __future__ import annotations

import logging


class GraphRuntime:
    """The worker's decode-graph state machine, as owned state.

    Three asymmetries of today's lifecycle are reproduced here exactly and
    are PINNED, NOT ENDORSED -- fixing any of them would change behavior and
    belongs in its own slice:

    * ``k25_whole_model_graph`` and ``whole_model_segment_name`` are written
      only by the K2.5 capture and are reset by NEITHER the release helper nor
      the worker's deep free. A deep free followed by a capture of a different
      mode reads a stale True / stale name.
    * the GLM-5 capture-OOM handler nulls the whole manager but adds only the
      failing bucket to ``glm5_whole_model_graph_failed_buckets``, so the
      buckets already captured in that episode are dropped with the manager
      yet stay eligible for the next warmup.
    * ``decode_graph_captures`` deliberately survives the worker's deep free
      while every other field resets: it is the capture-once guard.
    """

    def __init__(self) -> None:
        # --- core manager / adapter state -------------------------------
        self.cuda_graph_manager = None
        # Phase C: cuda-graph adapter (cached from
        # initializer.get_cuda_graph_adapter()) becomes the default and only
        # decode-graph path. When an adapter is present it owns eligibility /
        # replay-inputs / KV staging; the legacy `glm5_*` whole-model code
        # path is unreachable for models that ship an adapter. Phase B's
        # BATCHGEN_DECODE_GRAPH_ADAPTER_DUAL env gate is retired -- adapters
        # are now the production path, not a migration toggle.
        self.cuda_graph_adapter = None
        self.cuda_graph_adapter_dual = True
        self.glm5_moe_cuda_graph_manager = None
        self.glm5_layer_cuda_graph_manager = None

        # --- whole-model graph state (model-agnostic names) -------------
        self.whole_model_graph = False
        # These four were never set in the worker's __init__: every read
        # before the first capture went through a getattr default. The field
        # initializers below ARE those defaults, so a direct read is exactly
        # what the getattr returned.
        self.whole_model_bucketing = None
        self.whole_model_segment = None
        # Absent => "glm5_whole_model" at the single read site, which spells
        # that default itself (`... or "glm5_whole_model"`). This field must
        # stay None here: a getattr-with-default against an initialized field
        # would hand the read a None segment name and break GLM-5 replay.
        self.whole_model_segment_name = None
        self.k25_whole_model_graph = False
        # Whole-model decode graph capture episodes completed by this process.
        # On the reuse lifecycle it must stay at 1: the switch invariants and
        # the capture-once guard in _setup_cuda_graphs read it.
        self.decode_graph_captures = 0
        # Monotone stamp of the last "step exceeds the captured span" WARN.
        self.decode_graph_span_warn_monotonic = None

        # --- GLM-5 whole-model state ------------------------------------
        self.glm5_whole_model_graph = False
        self.glm5_whole_model_graph_failed_buckets = set()
        self.glm5_whole_model_graph_capture_attempted_for_batch = False
        self.glm5_whole_model_graph_state_change_after_capture_logged = False
        self.glm5_whole_model_graph_signature = None
        self.glm5_whole_model_graph_unavailable_reason = None
        # Late-init in the worker too (getattr defaults False / 0 / None).
        self.glm5_whole_model_all_short = False
        self.glm5_whole_model_index_topk = 0
        self.glm5_whole_model_capture_input_ids = None

        # --- Phase-C retired-mode residue -------------------------------
        # Initialized and reset by the worker's teardown, with no live reader
        # beyond the phase-switch asserts. Moved as-is: deleting them is a
        # separate dead-code slice.
        self.glm5_layer_graph_failed_buckets = set()
        self.glm5_layer_graph_capture_attempted_for_batch = False
        self.glm5_layer_graph_signature = None
        self.glm5_layer_graph_max_seqlen = None
        self.glm5_moe_graph_failed_buckets = set()
        self.glm5_dsa_graph_capture_attempted_for_batch = False
        self.glm5_moe_graph_capture_attempted_for_batch = False
        self.glm5_dsa_graph_page_table_change_after_capture_logged = False

    # ------------------------------------------------------------------ #
    # cuda_graph-layer constructors
    # ------------------------------------------------------------------ #

    def new_bucketing(self, bucket_sizes):
        """``BatchSizeBucketing`` over ``bucket_sizes``."""
        from batchgen.cuda_graph import BatchSizeBucketing

        return BatchSizeBucketing(bucket_sizes)

    def new_manager(self, bucketing, *, device):
        """A fresh ``CUDAGraphManager`` for ``bucketing`` on ``device``."""
        from batchgen.cuda_graph import CUDAGraphManager

        return CUDAGraphManager(bucketing, device=device)

    # ------------------------------------------------------------------ #
    # adapter handle
    # ------------------------------------------------------------------ #

    def adopt_adapter(self, initializer, *, model_name) -> None:
        """Fetch and type-check the model's cuda-graph adapter, once.

        Pulls the adapter from the initializer (None if the model does not
        implement one) and stores it. ``model_name`` names the model in the
        rejection message; the worker passes its own.
        """
        self.cuda_graph_adapter = getattr(
            initializer, "get_cuda_graph_adapter", lambda: None
        )()
        if self.cuda_graph_adapter is not None:
            # Every model's graph behavior must come through the
            # ModelCudaGraphAdapter ABC. An object that merely happens to carry
            # the right method names would pass every call site here and then
            # fail mid-decode on the first hook it does not implement, so the
            # type is asserted once, at startup, on the fetch itself.
            from batchgen.cuda_graph.adapter import ModelCudaGraphAdapter

            if not isinstance(self.cuda_graph_adapter, ModelCudaGraphAdapter):
                adapter_type = type(self.cuda_graph_adapter)
                raise RuntimeError(
                    "get_cuda_graph_adapter() for model "
                    f"{model_name!r} returned "
                    f"{adapter_type.__module__}.{adapter_type.__qualname__}, "
                    "which is not a ModelCudaGraphAdapter. Every model's "
                    "graph behavior must come through this ABC: subclass "
                    "batchgen.cuda_graph.adapter.ModelCudaGraphAdapter "
                    "instead of duck-typing it."
                )
            logging.info(
                "Cuda-graph adapter discovered: %s (advertises %s)",
                type(self.cuda_graph_adapter).__name__,
                [m.value for m in self.cuda_graph_adapter.advertised_modes()],
            )

    # ------------------------------------------------------------------ #
    # per-step seam objects
    # ------------------------------------------------------------------ #

    def make_batch_state(self, **fields):
        """The per-step ``BatchState`` snapshot the adapter seam consumes."""
        from batchgen.cuda_graph.adapter import BatchState

        return BatchState(**fields)

    def compare_decode_outputs(self, **kwargs):
        """Model-agnostic eager-vs-graph compare (observability only)."""
        from batchgen.cuda_graph.compare import compare_decode_outputs

        return compare_decode_outputs(**kwargs)

    # ------------------------------------------------------------------ #
    # teardown
    # ------------------------------------------------------------------ #

    def drop_refs_for_shutdown(self) -> None:
        """Drop the graph refs that retain model / segment / KV memory.

        The order is load-bearing and is the ordered-shutdown's own: the
        adapter's ``_ctx`` holds model, segment and KV-manager references, so
        it goes first; then the managers and the captured segment. Python's
        attribute destruction order is not an ownership contract, which is why
        the shutdown path spells it out.
        """
        self.cuda_graph_adapter = None
        self.cuda_graph_manager = None
        self.glm5_moe_cuda_graph_manager = None
        self.glm5_layer_cuda_graph_manager = None
        self.whole_model_segment = None
