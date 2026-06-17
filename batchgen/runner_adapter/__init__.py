"""M1 runner adapter — decode-only SGLang ModelRunner driven by BatchGen.

This package hosts the thin glue that lets BatchGen's decode phase run through a
per-rank, decode-only SGLang ``ModelRunner`` that ADOPTS BatchGen's
already-initialized process group and reads KV through the
``BatchGenNSAKVAdapter`` (``batchgen/attention/dsa/sglang_kv_bridge.py``).

Public surface (Slice 1):
  * ``build_sglang_decode_runner`` — construct the ModelRunner + inject adapter.
  * ``build_decode_forward_batch`` — build a DECODE ``ForwardBatch`` per step.
"""

from __future__ import annotations

from batchgen.runner_adapter.sglang_decode_runner import (
    build_decode_forward_batch,
    build_sglang_decode_runner,
)

__all__ = [
    "build_sglang_decode_runner",
    "build_decode_forward_batch",
]
