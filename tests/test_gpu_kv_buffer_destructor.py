"""~GPU_KV_Buffer must free exactly the buffers its constructor allocated.

`GPU_Buffer_Config::num_kv_buffer` is never parsed from Python, so it holds an
indeterminate value. Using it as the destructor's loop bound indexed
`k_buffers_` out of range and segfaulted the ordered worker shutdown.
"""

from __future__ import annotations

import re
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "core" / "GPU_KV_Buffer" / "GPU_KV_Buffer.cpp"


def _destructor_body() -> str:
    text = SOURCE.read_text()
    start = text.index("GPU_KV_Buffer::~GPU_KV_Buffer()")
    return text[start:text.index("\n};", start)]


def test_destructor_bounds_its_loop_by_the_allocated_buffers():
    body = _destructor_body()
    assert re.search(r"buffer_idx\s*<\s*this->k_buffers_\.size\(\)", body)
    assert "num_kv_buffer" not in body
