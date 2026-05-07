"""Model-agnostic CUDA Graph capture and replay for LLM decode phase.

This module provides infrastructure for capturing and replaying CUDA graphs
during the decode phase of LLM inference. It is designed to be model-agnostic:
models register "capturable segments" (e.g., attention blocks) and the manager
handles bucketing, warmup, capture, and replay.

Usage:
    bucketing = BatchSizeBucketing([1, 2, 4, 8, 16, 32])
    manager = CUDAGraphManager(bucketing, device=torch.device("cuda"))

    # Model registers its capturable segments
    manager.register_segment("layer_0_attn", attn_segment)

    # Pre-capture all graphs at startup
    manager.warmup_and_capture_all()

    # At runtime: replay with actual inputs
    outputs = manager.replay("layer_0_attn", batch_size=5, hidden_states=x, ...)
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tensor specification for static buffer allocation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TensorSpec:
    """Describes shape and dtype of a static tensor for graph capture.

    Shape dimensions can reference 'batch_size' as a placeholder that gets
    resolved to the actual bucket size at capture time.
    """
    shape: Tuple[Any, ...]  # e.g. ('batch_size', 1, 128, 576)
    dtype: torch.dtype
    fill_value: float = 0.0

    def resolve_shape(self, batch_size: int) -> Tuple[int, ...]:
        return tuple(
            batch_size if (s == "batch_size") else int(s)
            for s in self.shape
        )


# ---------------------------------------------------------------------------
# Protocol for capturable segments
# ---------------------------------------------------------------------------

@runtime_checkable
class CapturableSegment(Protocol):
    """Interface that model-specific code implements for CUDA graph capture.

    A capturable segment is a contiguous block of GPU computation with
    fixed tensor shapes (given a batch size). Examples: attention block,
    dense MLP, embedding + LM head.
    """

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        """Return specs for all input tensors at the given bucket size."""
        ...

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        """Return specs for all output tensors at the given bucket size."""
        ...

    def forward(self, **inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Execute the segment. During capture, this is recorded into the graph."""
        ...


# ---------------------------------------------------------------------------
# Batch size bucketing
# ---------------------------------------------------------------------------

class BatchSizeBucketing:
    """Manages discrete batch sizes for CUDA graph capture.

    Given a set of bucket sizes, provides O(1) lookup for the smallest bucket
    that fits a given batch size, and computes required padding.
    """

    DEFAULT_BUCKET_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]

    def __init__(self, bucket_sizes: Optional[List[int]] = None):
        self.bucket_sizes: List[int] = sorted(bucket_sizes or self.DEFAULT_BUCKET_SIZES)
        if not self.bucket_sizes:
            raise ValueError("At least one bucket size is required")
        self._max_bucket = self.bucket_sizes[-1]
        # Build O(1) lookup table: for any BS in [1, max_bucket], store the
        # smallest bucket that fits.
        self._lookup = [0] * (self._max_bucket + 1)
        bucket_idx = 0
        for bs in range(1, self._max_bucket + 1):
            while bucket_idx < len(self.bucket_sizes) - 1 and self.bucket_sizes[bucket_idx] < bs:
                bucket_idx += 1
            self._lookup[bs] = self.bucket_sizes[bucket_idx]

    def get_padded_size(self, batch_size: int) -> int:
        """Return the smallest bucket size >= batch_size.

        Raises ValueError if batch_size exceeds the largest bucket.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if batch_size > self._max_bucket:
            raise ValueError(
                f"batch_size {batch_size} exceeds max bucket {self._max_bucket}. "
                f"Either increase bucket sizes or fall back to eager execution."
            )
        return self._lookup[batch_size]

    def get_padding_count(self, batch_size: int) -> int:
        """Return number of dummy tokens needed to pad to the nearest bucket."""
        return self.get_padded_size(batch_size) - batch_size

    def __repr__(self) -> str:
        return f"BatchSizeBucketing(sizes={self.bucket_sizes})"


# ---------------------------------------------------------------------------
# Captured graph storage
# ---------------------------------------------------------------------------

@dataclass
class CapturedGraph:
    """Stores a captured CUDA graph along with its static I/O buffers."""
    bucket_size: int
    graph: torch.cuda.CUDAGraph
    static_inputs: Dict[str, torch.Tensor]
    static_outputs: Dict[str, torch.Tensor]
    input_fill_values: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CUDA Graph Manager
# ---------------------------------------------------------------------------

class CUDAGraphManager:
    """Model-agnostic CUDA graph capture and replay engine.

    Manages multiple named capturable segments, each captured at multiple
    bucket sizes. All graphs share a single memory pool to minimize HBM usage.

    Thread safety: NOT thread-safe. Designed for single-thread decode loops.
    """

    WARMUP_ITERATIONS = 2

    def __init__(
        self,
        bucketing: BatchSizeBucketing,
        device: Optional[torch.device] = None,
    ):
        self.bucketing = bucketing
        self.device = device or torch.device("cuda")

        # Shared memory pool across all graphs to minimize HBM usage.
        self._pool = torch.cuda.graph_pool_handle()

        # segment_name → {bucket_size → CapturedGraph}
        self._graphs: Dict[str, Dict[int, CapturedGraph]] = {}
        self._segments: Dict[str, CapturableSegment] = {}

        self._total_capture_time_ms: float = 0.0
        self._is_captured: bool = False
        self._memory_diag_enabled = os.environ.get(
            "BATCHGEN_CUDA_GRAPH_MEMORY_DIAG", ""
        ).lower() in {"1", "true", "yes", "on"}
        self._capture_memory_stats: List[Dict[str, Any]] = []

    @property
    def is_captured(self) -> bool:
        return self._is_captured

    # -- Registration -------------------------------------------------------

    def register_segment(self, name: str, segment: CapturableSegment) -> None:
        """Register a capturable segment. Must be called before capture."""
        if self._is_captured:
            raise RuntimeError("Cannot register segments after capture. Create a new manager.")
        if name in self._segments:
            raise ValueError(f"Segment '{name}' already registered")
        self._segments[name] = segment
        self._graphs[name] = {}

    # -- Capture ------------------------------------------------------------

    def warmup_and_capture_all(self) -> None:
        """Pre-capture all registered segments at all bucket sizes.

        This should be called once during config_decode(), before the first
        decode step. Blocks until all graphs are captured.
        """
        self.warmup_and_capture_buckets(self.bucketing.bucket_sizes)

    def warmup_and_capture_buckets(self, bucket_sizes: List[int]) -> None:
        """Capture all registered segments for the requested bucket sizes.

        This is used by memory-heavy model-specific segments that should be
        captured lazily only for buckets observed in live decode traffic.
        """
        if not self._segments:
            return

        unknown = [b for b in bucket_sizes if b not in self.bucketing.bucket_sizes]
        if unknown:
            raise ValueError(
                f"Cannot capture unknown CUDA graph buckets {unknown}; "
                f"available buckets are {self.bucketing.bucket_sizes}"
            )

        total_start = time.perf_counter()
        capture_sizes = list(dict.fromkeys(bucket_sizes))
        num_buckets = len(capture_sizes)

        for i, bucket_size in enumerate(capture_sizes):
            for seg_name, segment in self._segments.items():
                if bucket_size in self._graphs.get(seg_name, {}):
                    continue
                self._capture_one(seg_name, segment, bucket_size)
            done = i + 1
            bar = "█" * done + "░" * (num_buckets - done)
            logger.info(
                f"CUDA graph capture [{bar}] {done}/{num_buckets} "
                f"BS={bucket_size}"
            )

        elapsed_ms = (time.perf_counter() - total_start) * 1000
        self._total_capture_time_ms += elapsed_ms
        self._is_captured = any(self._graphs[name] for name in self._segments)

    def _capture_one(
        self, name: str, segment: CapturableSegment, bucket_size: int
    ) -> None:
        """Warmup and capture a single graph for one segment at one bucket size."""
        memory_record = self._new_capture_memory_record(name, bucket_size)
        phase_start = self._capture_memory_snapshot()
        overall_start = phase_start

        # 1. Allocate static input buffers
        input_specs = segment.get_static_input_specs(bucket_size)
        static_inputs = {}
        fill_values = {}
        for key, spec in input_specs.items():
            shape = spec.resolve_shape(bucket_size)
            t = torch.full(shape, spec.fill_value, dtype=spec.dtype, device=self.device)
            static_inputs[key] = t
            fill_values[key] = spec.fill_value
        phase_start = self._record_capture_memory_phase(
            memory_record, "static_input_allocation", phase_start
        )

        # 1b. Pre-allocate internal buffers if segment supports it
        # (e.g. MoESegment needs TMA descriptors and intermediate buffers
        #  created before graph capture since these are CPU-side operations)
        if hasattr(segment, 'setup_static_buffers'):
            segment.setup_static_buffers(bucket_size)
        if hasattr(segment, 'initialize_static_inputs'):
            segment.initialize_static_inputs(static_inputs, bucket_size)
        phase_start = self._record_capture_memory_phase(
            memory_record, "segment_setup", phase_start
        )

        # 2. Warmup on current stream
        for _ in range(self.WARMUP_ITERATIONS):
            with torch.inference_mode():
                segment.forward(**static_inputs)
        phase_start = self._record_capture_memory_phase(
            memory_record, "warmup", phase_start
        )

        # 3. Capture on current stream with shared pool
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            with torch.inference_mode():
                static_outputs = segment.forward(**static_inputs)
        phase_start = self._record_capture_memory_phase(
            memory_record, "graph_capture", phase_start
        )

        # Normalize outputs to dict
        if not isinstance(static_outputs, dict):
            static_outputs = {"output": static_outputs}

        self._graphs[name][bucket_size] = CapturedGraph(
            bucket_size=bucket_size,
            graph=graph,
            static_inputs=static_inputs,
            static_outputs=static_outputs,
            input_fill_values=fill_values,
        )
        self._finish_capture_memory_record(memory_record, overall_start, phase_start)


    # -- Replay -------------------------------------------------------------

    def replay(
        self,
        name: str,
        batch_size: int,
        **inputs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Replay a captured graph with new inputs.

        Args:
            name: Segment name (must have been registered and captured).
            batch_size: Actual batch size (will be padded to nearest bucket).
            **inputs: Tensor inputs matching the captured graph's input specs.

        Returns:
            Dict of output tensors, sliced to actual batch_size (unpadded).
        """
        bucket_size = self.bucketing.get_padded_size(batch_size)
        captured = self._graphs[name].get(bucket_size)
        if captured is None:
            raise RuntimeError(
                f"No graph captured for segment '{name}' at bucket {bucket_size}. "
                f"Available: {list(self._graphs[name].keys())}"
            )

        # Copy inputs to static buffers
        for key, tensor in inputs.items():
            static_tensor = captured.static_inputs.get(key)
            if static_tensor is None:
                raise KeyError(
                    f"Input '{key}' not found in captured graph '{name}'. "
                    f"Available: {list(captured.static_inputs.keys())}"
                )
            if tensor.shape[0] < static_tensor.shape[0]:
                fill_val = captured.input_fill_values.get(key, 0.0)
                padding_slice = static_tensor[tensor.shape[0]:]
                if fill_val == 0.0:
                    padding_slice.zero_()
                else:
                    padding_slice.fill_(fill_val)
                static_tensor[:tensor.shape[0]].copy_(tensor, non_blocking=True)
            elif tensor.shape[0] == static_tensor.shape[0]:
                static_tensor.copy_(tensor, non_blocking=True)
            else:
                raise ValueError(
                    f"Input '{key}' batch dim {tensor.shape[0]} exceeds "
                    f"static buffer {static_tensor.shape[0]} for bucket {bucket_size}"
                )

        # Replay on current stream — no cross-stream sync needed
        captured.graph.replay()

        # Return unpadded outputs
        result = {}
        for key, static_out in captured.static_outputs.items():
            if static_out.shape[0] == bucket_size and batch_size < bucket_size:
                result[key] = static_out[:batch_size]
            else:
                result[key] = static_out
        return result

    # -- Introspection ------------------------------------------------------

    def get_capture_stats(self) -> Dict[str, Any]:
        """Return capture statistics for logging/monitoring."""
        stats = {
            "total_capture_time_ms": self._total_capture_time_ms,
            "num_segments": len(self._segments),
            "bucket_sizes": self.bucketing.bucket_sizes,
            "graphs_per_segment": {
                name: list(graphs.keys())
                for name, graphs in self._graphs.items()
            },
            "capture_memory_diag_enabled": getattr(
                self, "_memory_diag_enabled", False
            ),
            "capture_memory_stats": list(getattr(
                self, "_capture_memory_stats", []
            )),
        }
        return stats

    # -- Capture memory diagnostics ----------------------------------------

    def _new_capture_memory_record(
        self, name: str, bucket_size: int
    ) -> Optional[Dict[str, Any]]:
        if not getattr(self, "_memory_diag_enabled", False):
            return None
        return {"segment": name, "bucket_size": bucket_size, "phases": []}

    def _capture_memory_snapshot(self) -> Optional[Dict[str, int]]:
        if not getattr(self, "_memory_diag_enabled", False):
            return None
        if not torch.cuda.is_available() or self.device.type != "cuda":
            return None

        torch.cuda.synchronize(self.device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        return {
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "used_bytes": int(total_bytes - free_bytes),
            "allocated_bytes": int(torch.cuda.memory_allocated(self.device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
        }

    @staticmethod
    def _memory_delta(
        before: Dict[str, int],
        after: Dict[str, int],
    ) -> Dict[str, int]:
        return {
            "free_delta_bytes": after["free_bytes"] - before["free_bytes"],
            "used_delta_bytes": after["used_bytes"] - before["used_bytes"],
            "allocated_delta_bytes": (
                after["allocated_bytes"] - before["allocated_bytes"]
            ),
            "reserved_delta_bytes": (
                after["reserved_bytes"] - before["reserved_bytes"]
            ),
        }

    def _record_capture_memory_phase(
        self,
        record: Optional[Dict[str, Any]],
        phase: str,
        before: Optional[Dict[str, int]],
    ) -> Optional[Dict[str, int]]:
        if record is None or before is None:
            return None
        after = self._capture_memory_snapshot()
        if after is None:
            return None

        delta = self._memory_delta(before, after)
        record["phases"].append(
            {
                "phase": phase,
                "before": before,
                "after": after,
                "delta": delta,
            }
        )
        logger.info(
            "[CUDA_GRAPH_MEMORY] segment=%s bucket=%s phase=%s "
            "used_delta=%.2fMiB allocated_delta=%.2fMiB "
            "reserved_delta=%.2fMiB free_delta=%.2fMiB",
            record["segment"],
            record["bucket_size"],
            phase,
            delta["used_delta_bytes"] / (1024 ** 2),
            delta["allocated_delta_bytes"] / (1024 ** 2),
            delta["reserved_delta_bytes"] / (1024 ** 2),
            delta["free_delta_bytes"] / (1024 ** 2),
        )
        return after

    def _finish_capture_memory_record(
        self,
        record: Optional[Dict[str, Any]],
        overall_start: Optional[Dict[str, int]],
        overall_end: Optional[Dict[str, int]],
    ) -> None:
        if record is None or overall_start is None or overall_end is None:
            return
        record["total_delta"] = self._memory_delta(overall_start, overall_end)
        self._capture_memory_stats.append(record)

    def has_graph(self, name: str, batch_size: int) -> bool:
        """Check if a graph exists for the given segment and batch size."""
        try:
            bucket = self.bucketing.get_padded_size(batch_size)
        except ValueError:
            return False
        return bucket in self._graphs.get(name, {})

    def has_bucket_for_all_segments(self, batch_size: int) -> bool:
        """Check whether every registered segment has the bucket for batch_size."""
        bucket = self.bucketing.get_padded_size(batch_size)
        return all(bucket in self._graphs.get(name, {}) for name in self._segments)

    def drop_bucket(self, bucket_size: int) -> None:
        """Release captured graphs and segment-owned static buffers for a bucket."""
        for graphs in self._graphs.values():
            graphs.pop(bucket_size, None)
        for segment in self._segments.values():
            release = getattr(segment, "release_static_buffers", None)
            if release is not None:
                release(bucket_size)
        self._is_captured = any(self._graphs[name] for name in self._segments)

    def __repr__(self) -> str:
        seg_info = ", ".join(
            f"{name}({len(graphs)} buckets)"
            for name, graphs in self._graphs.items()
        )
        return f"CUDAGraphManager(segments=[{seg_info}], captured={self._is_captured})"
