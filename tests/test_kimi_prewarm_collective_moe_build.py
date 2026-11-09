# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-Linear                                                       #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""K3 decode prewarm must run the EP-world MoE build on EVERY rank (CPU).

``_ensure_moe_built`` is an EP-world collective: it sets up the shared
exchange pool and runs the DeepEP self-check across all ranks in lockstep.
``prewarm_decode_graphs`` is called on every rank at decode configure, but
everything after it reads ``AttnWrapperBase.cur_batch`` is shaped by the
rank's LOCAL batch. If the collective build sits behind a batch-shaped early
return (the historical ``if bsz == 0: return False``), a first decode batch
that leaves any attention group empty strands the rowed ranks inside the
self-check dispatch waiting for peers that already returned; the dispatch
kernel times out and kills the cohort.

The test pins the ordering structurally: within ``prewarm_decode_graphs``,
the ``_ensure_moe_built`` call must appear before the first read of
``cur_batch`` (and hence before every batch-shaped return). Non-vacuity: the
method must actually contain both the build call and a ``cur_batch`` read.

Run: python -m pytest tests/test_kimi_prewarm_collective_moe_build.py -x -q
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "batchgen" / "models" / "moonshotai" / "kimi_linear" / "cuda_graph_segments.py"


def _prewarm_method() -> ast.FunctionDef:
    tree = ast.parse(SOURCE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if (
                    isinstance(item, ast.FunctionDef)
                    and item.name == "prewarm_decode_graphs"
                ):
                    return item
    raise AssertionError("prewarm_decode_graphs not found")


def test_moe_build_precedes_batch_shaped_logic():
    method = _prewarm_method()

    build_lines = []
    cur_batch_lines = []
    for node in ast.walk(method):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_ensure_moe_built"
        ):
            build_lines.append(node.lineno)
        if isinstance(node, ast.Attribute) and node.attr == "cur_batch":
            cur_batch_lines.append(node.lineno)
    # ast.walk is not in source order; compare the earliest occurrences.
    build_line = min(build_lines) if build_lines else None
    cur_batch_line = min(cur_batch_lines) if cur_batch_lines else None

    # Non-vacuity: both anchors must exist in the method.
    assert build_line is not None, "prewarm_decode_graphs no longer builds MoE graphs"
    assert cur_batch_line is not None, "prewarm_decode_graphs no longer reads cur_batch"

    assert build_line < cur_batch_line, (
        "the EP-world collective _ensure_moe_built must run before any "
        "batch-shaped logic in prewarm_decode_graphs; behind a local-batch "
        "early return it deadlocks the cohort when a decode batch leaves an "
        "attention group empty"
    )
