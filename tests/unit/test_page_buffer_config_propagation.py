"""Reconfigured page-buffer sizes reach every consumer at use time.

``batchgen.sequence`` keeps INITIAL_GPU_PAGE_BUFFER, EXTENSION_GPU_PAGE_BUFFER
and DECISION_FREQUENCY_PAGES as module globals that ``configure_page_buffers``
rebinds from the worker args when ``BatchGenWorker`` is constructed. A
``from batchgen.sequence import INITIAL_GPU_PAGE_BUFFER`` executed at import
time binds the import-time default instead, so with a non-default
``--initial-gpu-page-buffer`` the worker's prefill host reservation and the
sequence's own GPU-load formula would disagree.

The worker methods are compiled straight out of ``batchgen_worker.py``'s AST
onto fakes, with the module-level ``batchgen.sequence`` imports of the real
worker source executed first, exactly as module import precedes worker
construction. Nothing here needs CUDA.
"""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import pytest

import batchgen.sequence as sequence_module
from batchgen.sequence import SequenceEntry, configure_page_buffers
from batchgen.worker.prefill import (
    PrefillCandidate,
    PrefillSelectionRequest,
    compute_prefill_host_reservation,
)


REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "batchgen"
WORKER = PACKAGE / "batchgen_worker.py"
PAGE_BUFFER_NAMES = frozenset({
    "INITIAL_GPU_PAGE_BUFFER",
    "EXTENSION_GPU_PAGE_BUFFER",
    "DECISION_FREQUENCY_PAGES",
})
DEFAULTS = dict(
    initial_gpu_page_buffer=32,
    extension_gpu_page_buffer=4,
    decision_frequency_pages=2,
)
CONFIGURED_INITIAL = 7  # any value != the default 32
CHUNK_TOKENS = 64


@pytest.fixture(autouse=True)
def _default_page_buffers():
    """Start from the import-time defaults and leave them that way."""
    configure_page_buffers(**DEFAULTS)
    yield
    configure_page_buffers(**DEFAULTS)


def _imported_module(node: ast.ImportFrom, path: Path) -> str:
    if node.level == 0:
        return node.module or ""
    package = list(path.relative_to(REPO).parent.parts)
    package = package[: len(package) - (node.level - 1)]
    return ".".join(package + ([node.module] if node.module else []))


def test_no_module_imports_the_page_buffer_globals_by_value():
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if _imported_module(node, path) != "batchgen.sequence":
                continue
            for alias in node.names:
                if alias.name in PAGE_BUFFER_NAMES:
                    offenders.append(
                        f"{path.relative_to(REPO)}:{node.lineno} {alias.name}"
                    )
    assert not offenders, (
        "page-buffer sizes are rebound by configure_page_buffers(); importing "
        "them by value freezes the import-time default: " + ", ".join(offenders)
    )


def _worker_sequence_imports() -> dict:
    """Run the worker module's own top-level ``batchgen.sequence`` imports."""
    tree = ast.parse(WORKER.read_text(), filename=str(WORKER))
    nodes = [
        copy.deepcopy(node) for node in tree.body
        if (isinstance(node, ast.ImportFrom)
            and _imported_module(node, WORKER) == "batchgen.sequence")
        or (isinstance(node, ast.Import)
            and any(a.name == "batchgen.sequence" for a in node.names))
    ]
    assert nodes, "batchgen_worker.py no longer imports batchgen.sequence"
    namespace: dict = {}
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(WORKER), "exec"), namespace)
    return namespace


def _worker_method(name: str, namespace: dict):
    tree = ast.parse(WORKER.read_text(), filename=str(WORKER))
    worker = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )
    method = next(
        node for node in worker.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[copy.deepcopy(method)], type_ignores=[])
    )
    namespace.update(Dict=Dict, List=List, Optional=Optional, Tuple=Tuple)
    exec(compile(module, str(WORKER), "exec"), namespace)
    return namespace[name]


def _sequence() -> SequenceEntry:
    seq = SequenceEntry("a", global_idx=0, prompt_length=100,
                        max_decode_length=8192)
    seq.assigned_rank = 0
    return seq


def test_prefill_host_reservation_uses_the_configured_initial_buffer():
    namespace = _worker_sequence_imports()  # module import precedes __init__
    configure_page_buffers(
        initial_gpu_page_buffer=CONFIGURED_INITIAL,
        extension_gpu_page_buffer=4,
        decision_frequency_pages=2,
    )
    seen = []

    def recording_reservation(**kwargs):
        seen.append(kwargs["initial_gpu_page_buffer"])
        return compute_prefill_host_reservation(**kwargs)

    namespace["compute_prefill_host_reservation"] = recording_reservation
    plan = _worker_method("_prefill_host_reservation_plan", namespace)
    seq = _sequence()
    worker = SimpleNamespace(
        rank=0,
        enable_prefix_cache=False,
        _prefix_sequence_states={},
        _get_effective_chunk_size=lambda: CHUNK_TOKENS,
        _owns_host_kv=lambda s: True,
        global_batch=SimpleNamespace(get_sequence=lambda uuid: seq),
    )

    reservations, owner_requests = plan(worker, ["a"])

    assert seen == [CONFIGURED_INITIAL]
    pages, tokens = reservations["a"]
    # The host reservation must equal the sequence's own initial-chunk formula,
    # which reads the reconfigured global ...
    assert pages == seq.get_host_pages_for_initial_chunk(CHUNK_TOKENS)
    # ... and cover the first GPU load after prefill (prompt + 1 token).
    seq.current_context_length = seq.prompt_length + 1
    assert pages >= seq.get_gpu_pages_for_initial_load()
    assert owner_requests == [("a", tokens)]


def test_prefill_selection_request_uses_the_configured_initial_buffer():
    namespace = _worker_sequence_imports()
    configure_page_buffers(
        initial_gpu_page_buffer=CONFIGURED_INITIAL,
        extension_gpu_page_buffer=4,
        decision_frequency_pages=2,
    )
    namespace.update(
        PrefillCandidate=PrefillCandidate,
        PrefillSelectionRequest=PrefillSelectionRequest,
    )
    make_request = _worker_method("_make_prefill_selection_request", namespace)
    seq = _sequence()
    worker = SimpleNamespace(
        local_world_size=8,
        _decode_attn_tp_size=lambda: 1,
        global_batch=SimpleNamespace(get_sequence=lambda uuid: seq),
    )

    request = make_request(worker, ["a"], [1 << 20], num_nodes=1,
                           chunk_size=CHUNK_TOKENS)

    assert request.initial_gpu_page_buffer == CONFIGURED_INITIAL
    assert sequence_module.INITIAL_GPU_PAGE_BUFFER == CONFIGURED_INITIAL
