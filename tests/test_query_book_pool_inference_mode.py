"""A QueryBook pool grown inside the decode loop's inference_mode must stay
writable by admissions that run outside inference_mode."""

import ast
import logging
import types
from pathlib import Path
from typing import Optional

import torch


WORKER = Path(__file__).resolve().parents[1] / "batchgen" / "batchgen_worker.py"
WORKER_TREE = ast.parse(WORKER.read_text(), filename=str(WORKER))


def _load():
    """Load the real pool class and grow method without the inference engine."""
    classes = [
        node
        for node in WORKER_TREE.body
        if isinstance(node, ast.ClassDef)
        and node.name in ("QueryBookPoolCapacityError", "QueryBookBufferPool")
    ]
    worker_class = next(
        node
        for node in WORKER_TREE.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )
    method = next(
        node
        for node in worker_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_ensure_buffer_pool"
    )
    namespace = {
        "Optional": Optional,
        "logging": logging,
        "torch": torch,
        "dist": types.SimpleNamespace(barrier=lambda: None),
        "allocate_node_shared_int64": lambda name, rows, width, is_creator, barrier: (
            torch.zeros((rows, width), dtype=torch.long),
            None,
        ),
    }
    module = ast.Module(body=classes + [method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(WORKER), "exec"), namespace)
    return namespace["_ensure_buffer_pool"]


ENSURE_BUFFER_POOL = _load()


def _worker():
    worker = types.SimpleNamespace(
        _buffer_pool=None,
        _buffer_pool_generation=0,
        rank=0,
        local_world_size=1,
        _query_book_shm_prefix="test",
        pad_token_id=7,
        _rebind_buffer_pool_views=lambda: None,
        _retire_buffer_pool=lambda old, is_creator: None,
    )
    worker.ensure = types.MethodType(ENSURE_BUFFER_POOL, worker)
    return worker


def test_pool_grown_in_inference_mode_accepts_slot_reuse_outside_it():
    worker = _worker()
    worker.ensure(required_rows=4, required_input_width=8,
                  required_decode_width=16, reason="initial")
    pool = worker._buffer_pool
    first = pool.allocate_slot()
    pool.allocate_slot()

    # Mid-decode admission: the decode loop runs under inference_mode.
    with torch.inference_mode():
        worker.ensure(required_rows=4, required_input_width=12,
                      required_decode_width=16, reason="mid-decode growth")
    grown = worker._buffer_pool
    assert grown is not pool
    assert not grown.decoded_tokens_buffer.is_inference()
    assert not grown.input_ids_buffer.is_inference()

    # Admission after the decode interval runs outside inference_mode and
    # reuses a freed slot, which clears it in place.
    grown.decoded_tokens_buffer[first, :] = 3
    grown.free_slot(first)
    assert grown.allocate_slot() == first
    assert torch.all(grown.decoded_tokens_buffer[first] == 7)
