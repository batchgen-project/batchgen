"""Standardized per-sub-op decode timing for all BatchGen MoE models.

Usage:
    1. Enable via env var: BATCHGEN_DECODE_TIMING=1
    2. Control interval: BATCHGEN_DECODE_TIMING_INTERVAL=50 (default)
    3. Instrument model forward:

        from batchgen.profiling.decode_timing import DecodeTimingStats, decode_timing_enabled

        if decode_timing_enabled():
            torch.cuda.synchronize()
            _t0 = time.perf_counter()

        # ... do work ...

        if decode_timing_enabled():
            DecodeTimingStats.record("moe_gemm", _t0)

    4. Call DecodeTimingStats.step_done() at end of each decode step.

Standard instrumentation points (all models should report these):
    MoE: moe_allgather, moe_gate, moe_dispatch, moe_gemm, moe_reduce, moe_allreduce
    Attn: attn_qkv_proj, attn_rope, attn_kv_update, attn_forward, attn_output_proj
"""

import logging
import os
import time

import torch

_DECODE_TIMING = os.environ.get("BATCHGEN_DECODE_TIMING", "0") == "1"
_DECODE_TIMING_INTERVAL = int(os.environ.get("BATCHGEN_DECODE_TIMING_INTERVAL", "50"))


def decode_timing_enabled() -> bool:
    """Check if decode timing is active."""
    return _DECODE_TIMING


class DecodeTimingStats:
    """Accumulates per-sub-op timing across decode steps. Logs every N steps.

    All timing uses cuda.synchronize() + perf_counter for accurate per-op measurement.
    Note: synchronization serializes the pipeline — timing overhead is real.
    Only enable for profiling, not production runs.
    """
    _step = 0
    _accum = {}  # op_name -> total_us
    _model_name = "Model"  # Set by model wrapper for log readability

    @classmethod
    def set_model_name(cls, name: str):
        cls._model_name = name

    @classmethod
    def record(cls, op: str, t0: float):
        """Record elapsed time since t0 for the named op. Syncs GPU first."""
        torch.cuda.synchronize()
        elapsed_us = (time.perf_counter() - t0) * 1e6
        cls._accum[op] = cls._accum.get(op, 0.0) + elapsed_us

    # Legacy alias
    _sync_record = record

    @classmethod
    def step_done(cls):
        """Call at end of each decode step. Logs and resets every N steps."""
        cls._step += 1
        if cls._step % _DECODE_TIMING_INTERVAL != 0:
            return
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != 0:
            cls._accum.clear()
            return
        n = _DECODE_TIMING_INTERVAL
        lines = [f"\n=== {cls._model_name} Decode Timing (avg over {n} steps, step {cls._step}) ==="]
        total = sum(cls._accum.values())
        for op, us in sorted(cls._accum.items(), key=lambda x: -x[1]):
            pct = 100 * us / max(total, 1)
            lines.append(f"  {op:20s}: {us/n:8.1f} us/step ({pct:5.1f}%)")
        lines.append(f"  {'TOTAL':20s}: {total/n:8.1f} us/step")
        logging.info("\n".join(lines))
        cls._accum.clear()

    @classmethod
    def reset(cls):
        """Reset all accumulated timing data."""
        cls._step = 0
        cls._accum.clear()
