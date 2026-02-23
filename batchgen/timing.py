"""Reusable timing infrastructure for BatchGen decode/prefill ablation.

Enable via environment variable: BATCHGEN_DECODE_TIMING=1

Usage:
    # Create a timer with model-specific categories
    timer = DecodeTimingStats("GLM-5", [
        "q_proj", "kv_proj", "kv_write", "attn_forward", "o_proj",
        "allgather", "routing", "dispatch", "grouped_gemm", "allreduce",
    ])

    # Instrument code with context manager
    with timer.timed("q_proj", layer_idx):
        q = w8a8_deepgemm(...)

    # After iteration, log and reset
    timer.log_summary()
    timer.reset()
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Dict, List, Optional

import torch


class TimingStats:
    """Per-iteration timing accumulator with named categories.

    Each category accumulates total ms and per-layer ms.
    The timed() context manager handles cuda sync + perf_counter.
    """

    def __init__(self, model_name: str, phase: str, categories: List[str]):
        self.model_name = model_name
        self.phase = phase  # "decode" or "prefill"
        self._categories = list(categories)
        self.enabled: bool = False
        # category -> total ms
        self._totals: Dict[str, float] = {}
        # category -> {layer_idx -> ms}
        self._per_layer: Dict[str, Dict[int, float]] = {}
        # category -> call count
        self._counts: Dict[str, int] = {}
        self.reset()

    def reset(self):
        self._totals = {c: 0.0 for c in self._categories}
        self._per_layer = {c: {} for c in self._categories}
        self._counts = {c: 0 for c in self._categories}

    def enable(self):
        self.enabled = True
        self.reset()

    def disable(self):
        self.enabled = False

    def add_category(self, name: str):
        """Register a new category (e.g., model-specific DSA categories)."""
        if name not in self._totals:
            self._categories.append(name)
            self._totals[name] = 0.0
            self._per_layer[name] = {}
            self._counts[name] = 0

    def record(self, category: str, layer_idx: int, time_ms: float):
        if not self.enabled:
            return
        if category not in self._totals:
            self.add_category(category)
        self._totals[category] += time_ms
        per_layer = self._per_layer[category]
        per_layer[layer_idx] = per_layer.get(layer_idx, 0.0) + time_ms
        self._counts[category] += 1

    @contextmanager
    def timed(self, category: str, layer_idx: int = 0):
        """Context manager: sync → measure → sync → record."""
        if not self.enabled:
            yield
            return
        torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.record(category, layer_idx, elapsed_ms)

    def log_summary(self):
        if not self.enabled:
            return
        total_all = sum(self._totals.values())
        if total_all <= 0:
            return

        logging.info("=" * 70)
        logging.info(f"{self.model_name} {self.phase.capitalize()} Timing Summary")
        logging.info("=" * 70)

        # Category breakdown
        for cat in self._categories:
            ms = self._totals[cat]
            if ms <= 0:
                continue
            pct = ms / total_all * 100
            calls = self._counts[cat]
            logging.info(f"  {cat:<20s} {ms:10.2f} ms  ({pct:5.1f}%)  [{calls} calls]")

        logging.info(f"  {'TOTAL':<20s} {total_all:10.2f} ms")

        # Per-layer table (first 3 + last layer)
        all_layers = set()
        for per_layer in self._per_layer.values():
            all_layers.update(per_layer.keys())
        if all_layers:
            sorted_layers = sorted(all_layers)
            show_layers = sorted_layers[:3]
            if sorted_layers[-1] not in show_layers:
                show_layers.append(sorted_layers[-1])

            logging.info("-" * 70)
            # Header
            cats_with_data = [c for c in self._categories if self._totals[c] > 0]
            hdr = f"{'Layer':>6s}"
            for cat in cats_with_data:
                hdr += f"  {cat[:10]:>10s}"
            logging.info(hdr)

            for li in show_layers:
                row = f"{li:6d}"
                for cat in cats_with_data:
                    ms = self._per_layer[cat].get(li, 0.0)
                    row += f"  {ms:10.2f}"
                logging.info(row)

        logging.info("=" * 70)


# ---------------------------------------------------------------------------
# Convenience: global decode timer (one per process)
# ---------------------------------------------------------------------------

_decode_timer: Optional[TimingStats] = None


def get_decode_timer() -> Optional[TimingStats]:
    """Return the global decode timer, or None if not enabled."""
    return _decode_timer


def init_decode_timer(model_name: str, categories: List[str]) -> TimingStats:
    """Initialize and return the global decode timer."""
    global _decode_timer
    _decode_timer = TimingStats(model_name, "decode", categories)
    if os.environ.get("BATCHGEN_DECODE_TIMING", "0") == "1":
        _decode_timer.enable()
        logging.info(f"DecodeTimingStats enabled for {model_name}")
    return _decode_timer


_prefill_timer: Optional[TimingStats] = None


def get_prefill_timer() -> Optional[TimingStats]:
    return _prefill_timer


def init_prefill_timer(model_name: str, categories: List[str]) -> TimingStats:
    global _prefill_timer
    _prefill_timer = TimingStats(model_name, "prefill", categories)
    if os.environ.get("BATCHGEN_PREFILL_TIMING", "0") == "1":
        _prefill_timer.enable()
        logging.info(f"PrefillTimingStats enabled for {model_name}")
    return _prefill_timer
