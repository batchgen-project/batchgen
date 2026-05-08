"""Tests for preserving batch-level create parameters."""

import importlib.util
import sys
from pathlib import Path


def _load_io_struct():
    path = Path(__file__).resolve().parents[2] / "batchgen" / "server" / "io_struct.py"
    spec = importlib.util.spec_from_file_location("batchgen.server.io_struct", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["batchgen.server.io_struct"] = module
    spec.loader.exec_module(module)
    return module


def test_create_batch_request_preserves_debug_and_sampling_fields():
    io_struct = _load_io_struct()
    request = io_struct.CreateBatchRequest(
        input_file_id="file-test",
        max_decoding_length=128,
        max_context_length=4096,
        temperature=0.0,
        top_p=1.0,
        top_k=5,
        batchgen_debug={
            "glm5_dsa_mode": "eager",
            "glm5_moe_mode": "graph",
        },
    )

    batch = io_struct.build_batch_object_from_create_request(
        batch_id="batch-test",
        body=request,
        created_at=100,
        expires_at=200,
    )

    assert batch.max_decoding_length == 128
    assert batch.max_context_length == 4096
    assert batch.temperature == 0.0
    assert batch.top_p == 1.0
    assert batch.top_k == 5
    assert batch.batchgen_debug == {
        "glm5_dsa_mode": "eager",
        "glm5_moe_mode": "graph",
    }
