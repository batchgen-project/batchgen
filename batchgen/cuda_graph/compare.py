"""Eager-vs-graph compare facility (formal integration safety-net infra).

`compare_decode_outputs` runs the adapter's `run_eager_reference` against the
same inputs the captured graph consumed and produces a `CompareReport`. This
is the bring-up gate for new models and new modes: an adapter is considered
to have "landed mode X" only when this facility reports `passed=True` across
the unit-test battery defined in
`batchgen_design/cuda_graph/cuda_graph_contract.md` §G.

The facility is observability-only by contract (§E guarantee #1): enabling
`BATCHGEN_DECODE_GRAPH_COMPARE=1` MUST NOT influence mode selection or which
tokens get sampled. The eager re-run produces a diff source; the graph
output is what production uses.

Generalized from the GLM-5-specific helper at
`batchgen/models/glm/glm5/whole_model_cuda_graph_segments.py:439`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, TYPE_CHECKING, Tuple

import torch

if TYPE_CHECKING:
    from batchgen.cuda_graph.adapter import (
        BatchState, GraphDecision, GraphMode, ModelCudaGraphAdapter,
    )

logger = logging.getLogger(__name__)


class CompareFailure(RuntimeError):
    """Raised when `fail_on_mismatch=True` and the report does not pass."""


@dataclass
class CompareReport:
    """Result of one `compare_decode_outputs` call.

    `per_key` holds the per-output-key (max_abs, max_rel). `probe_results`
    holds the per-probe-layer (max_abs, max_rel) for keys matching
    `hidden_states_layer_<i>`.
    """
    segment_name: str
    mode: "GraphMode"
    bucket: int
    max_abs: float
    max_rel: float
    per_key: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    probe_results: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    passed: bool = True
    mismatched_keys: List[str] = field(default_factory=list)
    missing_eager_keys: List[str] = field(default_factory=list)
    missing_graph_keys: List[str] = field(default_factory=list)


def _diff_pair(eager: torch.Tensor, graph: torch.Tensor) -> Tuple[float, float, bool]:
    """Return (max_abs, max_rel, allclose) for two tensors of matching shape."""
    if eager.shape != graph.shape:
        return float("inf"), float("inf"), False
    eager_f = eager.detach().to(torch.float32)
    graph_f = graph.detach().to(torch.float32)
    if eager_f.numel() == 0:
        return 0.0, 0.0, True
    diff = (eager_f - graph_f).abs()
    max_abs = float(diff.max().item())
    denom = graph_f.abs().clamp(min=1e-12)
    max_rel = float((diff / denom).max().item())
    return max_abs, max_rel, True


def compare_decode_outputs(
    *,
    adapter: "ModelCudaGraphAdapter",
    decision: "GraphDecision",
    batch_state: "BatchState",
    segment_name: str,
    captured_inputs: Dict[str, torch.Tensor],
    graph_outputs: Dict[str, torch.Tensor],
    probe_layers: Iterable[int] = (),
    atol: float = 1e-2,
    rtol: float = 1e-2,
    fail_on_mismatch: bool = False,
) -> CompareReport:
    """Run adapter's eager reference, diff against graph outputs, return report.

    Contract guarantees enforced here:
      * Eager runs against `captured_inputs` byte-identical to the graph
        replay (no test-side rewrite).
      * `adapter.run_eager_reference` MUST not mutate KV state (contract §B
        eager-reference rules); we do not enforce that here, but a violation
        produces drift visible in the report.
      * Output key sets are compared; missing keys are reported, not silently
        dropped.
      * Probe-layer keys follow the `hidden_states_layer_<i>` convention.

    Raises `CompareFailure` only when `fail_on_mismatch=True` and the report
    does not pass. Otherwise returns the report unconditionally; callers
    decide what to do with it.
    """
    probe_list = list(probe_layers)
    eager_outputs = adapter.run_eager_reference(
        segment_name=segment_name,
        batch_state=batch_state,
        captured_inputs=captured_inputs,
        probe_layers=probe_list,
    )

    bucket = decision.bucket if decision.bucket is not None else 0
    report = CompareReport(
        segment_name=segment_name,
        mode=decision.mode,
        bucket=bucket,
        max_abs=0.0,
        max_rel=0.0,
    )

    eager_keys = set(eager_outputs.keys())
    graph_keys = set(graph_outputs.keys())
    report.missing_eager_keys = sorted(graph_keys - eager_keys)
    report.missing_graph_keys = sorted(eager_keys - graph_keys)
    if report.missing_eager_keys or report.missing_graph_keys:
        report.passed = False

    overall_max_abs = 0.0
    overall_max_rel = 0.0
    for key in sorted(eager_keys & graph_keys):
        max_abs, max_rel, shape_ok = _diff_pair(eager_outputs[key], graph_outputs[key])
        report.per_key[key] = (max_abs, max_rel)
        overall_max_abs = max(overall_max_abs, max_abs)
        overall_max_rel = max(overall_max_rel, max_rel)
        if (not shape_ok) or (max_abs > atol and max_rel > rtol):
            report.mismatched_keys.append(key)
            report.passed = False

    # Pull per-probe-layer rows from the per_key map.
    probe_prefix = "hidden_states_layer_"
    for key, (max_abs, max_rel) in report.per_key.items():
        if not key.startswith(probe_prefix):
            continue
        try:
            layer_idx = int(key[len(probe_prefix):])
        except ValueError:
            continue
        report.probe_results[layer_idx] = (max_abs, max_rel)

    report.max_abs = overall_max_abs
    report.max_rel = overall_max_rel

    if not report.passed and fail_on_mismatch:
        raise CompareFailure(
            f"compare_decode_outputs failed: segment={segment_name} "
            f"mode={decision.mode.value} bucket={bucket} "
            f"max_abs={report.max_abs:.3e} max_rel={report.max_rel:.3e} "
            f"mismatched={report.mismatched_keys} "
            f"missing_eager={report.missing_eager_keys} "
            f"missing_graph={report.missing_graph_keys}"
        )

    return report


__all__ = ["CompareReport", "CompareFailure", "compare_decode_outputs"]
