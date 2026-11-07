"""GLM-5 decode criticality probe: injected delays inside the captured graph.

The criticality of an op is ``d(step time) / d(op time)`` in ``[0, 1]``: 1 for
an op on the critical path of the decode step, 0 for one fully hidden behind
other work (a side-stream op overlapped by the main stream, or a main-stream op
the step waits past anyway). It is measured, not modelled: a spin kernel is
captured right after the op, its duration is swept across replays of ONE
captured graph, and the slope of the mean step time against the delay injected
per step is the criticality of that position.

Activation is batch-level only, through ``batchgen_debug``::

    "batchgen_debug": {"glm5_criticality_probe": {
        "sites": ["after_router"], "layers": "all",
        "schedule": [{"steps": 128, "delay_us": {}},
                     {"steps": 128, "delay_us": {"after_router": 10}}],
        "skip_steps_after_change": 8}}

``sites`` fixes which spin nodes exist in the captured graph; ``schedule``
drives only the host-side delay table (a site missing from a level is 0) and
cycles when exhausted. Without the key nothing in this module runs: the
segments' hooks are ``is not None`` checks, the spin extension is never
imported, and the captured graph is exactly the uninstrumented one.

Host-sync contract on the steady-state decode path: ``before_replay`` enqueues
at most one async host->device copy, from a pinned row that is written once at
construction and never again, on the stream the replay is issued on. Stream
order makes the copy land before the replay reads the table, and a pinned row
that is never rewritten cannot be raced by the host.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple

import torch


DEBUG_KEY = "glm5_criticality_probe"

# Fixed site indices: the index is baked into each captured spin launch, so the
# order of this tuple is part of the graph's identity and must only be appended
# to. Each name says what the spin node follows.
SITES: Tuple[str, ...] = (
    # layer input_layernorm (with its fused FP8 cast), or layer 0's act_quant
    "after_input_norm_quant",
    # q_a RMSNorm and its FP8 cast
    "after_q_a_norm_quant",
    # indexer scoring + top-k (indexer layers of a scoring graph only)
    "after_indexer",
    # end of the indexer K path, ON the indexer-K side stream
    "after_indexer_k_path",
    # the attention segment's o_proj output, before the post-attention norm
    "after_attention",
    # post_attention_layernorm (with its fused FP8 cast)
    "after_post_attn_norm_quant",
    # the router's top-k gate (never between the router GEMM and the gate)
    "after_router",
    # DeepEP dispatch (send + receive)
    "after_dispatch",
    # routed expert GEMM chain (R1, activation, R2)
    "after_expert_gemm",
    # DeepEP combine (send + receive), before the shared-expert join
    "after_combine",
    # shared expert, ON the shared-expert side stream (inside the fork)
    "after_shared_expert",
)
SITE_INDEX: Dict[str, int] = {name: index for index, name in enumerate(SITES)}

# Sites inside the MoE block. Only the DeepEP decode path has them at defined
# positions; the NCCL MoE path refuses any probe.
MOE_SITES = frozenset({
    "after_router",
    "after_dispatch",
    "after_expert_gemm",
    "after_combine",
    "after_shared_expert",
})

# Largest delay a level may request per site per mark. A 78-layer sweep at the
# cap already adds ~156 ms per step; anything larger is a typo, not a probe.
MAX_DELAY_US = 2000

_SPEC_KEYS = frozenset({"sites", "layers", "schedule", "skip_steps_after_change"})
_LEVEL_KEYS = frozenset({"steps", "delay_us"})


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class CriticalityProbeSpec:
    """A validated ``glm5_criticality_probe`` request.

    ``sites`` is in ``SITES`` order; ``layers`` is ``None`` for every layer;
    each schedule level is ``(steps, delays_ns)`` with ``delays_ns`` aligned to
    ``SITES`` (0 for sites the level does not name).
    """

    sites: Tuple[str, ...]
    layers: Optional[Tuple[int, ...]]
    schedule: Tuple[Tuple[int, Tuple[int, ...]], ...]
    skip_steps_after_change: int

    @classmethod
    def from_batchgen_debug(cls, debug: Any) -> Optional["CriticalityProbeSpec"]:
        """The probe a batch's merged ``batchgen_debug`` requests, or ``None``."""
        if not isinstance(debug, Mapping):
            return None
        raw = debug.get(DEBUG_KEY)
        if raw is None:
            return None
        return cls.parse(raw)

    @classmethod
    def parse(cls, raw: Any) -> "CriticalityProbeSpec":
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"{DEBUG_KEY} must be an object, got {type(raw).__name__}")
        unknown = sorted(set(raw) - _SPEC_KEYS)
        if unknown:
            raise ValueError(
                f"{DEBUG_KEY}: unknown keys {unknown}; expected a subset of "
                f"{sorted(_SPEC_KEYS)}")

        raw_sites = raw.get("sites")
        if not isinstance(raw_sites, (list, tuple)) or not raw_sites:
            raise ValueError(f"{DEBUG_KEY}.sites must be a non-empty list of site names")
        for site in raw_sites:
            if site not in SITE_INDEX:
                raise ValueError(
                    f"{DEBUG_KEY}.sites: unknown site {site!r}; known sites are "
                    f"{list(SITES)}")
        if len(set(raw_sites)) != len(raw_sites):
            raise ValueError(f"{DEBUG_KEY}.sites lists a site twice: {list(raw_sites)}")
        sites = tuple(sorted(raw_sites, key=SITE_INDEX.__getitem__))

        raw_layers = raw.get("layers", "all")
        if raw_layers == "all":
            layers = None
        else:
            if not isinstance(raw_layers, (list, tuple)) or not raw_layers:
                raise ValueError(
                    f"{DEBUG_KEY}.layers must be \"all\" or a non-empty list of "
                    f"layer indices, got {raw_layers!r}")
            for layer in raw_layers:
                if not _is_int(layer) or layer < 0:
                    raise ValueError(
                        f"{DEBUG_KEY}.layers: {layer!r} is not a non-negative "
                        "integer layer index")
            if len(set(raw_layers)) != len(raw_layers):
                raise ValueError(
                    f"{DEBUG_KEY}.layers lists a layer twice: {list(raw_layers)}")
            layers = tuple(sorted(raw_layers))

        skip = raw.get("skip_steps_after_change", 0)
        if not _is_int(skip) or skip < 0:
            raise ValueError(
                f"{DEBUG_KEY}.skip_steps_after_change must be a non-negative "
                f"integer, got {skip!r}")

        raw_schedule = raw.get("schedule")
        if not isinstance(raw_schedule, (list, tuple)) or not raw_schedule:
            raise ValueError(f"{DEBUG_KEY}.schedule must be a non-empty list of levels")
        schedule = []
        for level_idx, level in enumerate(raw_schedule):
            where = f"{DEBUG_KEY}.schedule[{level_idx}]"
            if not isinstance(level, Mapping):
                raise ValueError(f"{where} must be an object, got {type(level).__name__}")
            unknown = sorted(set(level) - _LEVEL_KEYS)
            if unknown:
                raise ValueError(
                    f"{where}: unknown keys {unknown}; expected {sorted(_LEVEL_KEYS)}")
            steps = level.get("steps")
            if not _is_int(steps) or steps <= 0:
                raise ValueError(f"{where}.steps must be a positive integer, got {steps!r}")
            if skip >= steps:
                raise ValueError(
                    f"{where}.steps={steps} leaves no measured step after "
                    f"skip_steps_after_change={skip}")
            delay_us = level.get("delay_us", {})
            if not isinstance(delay_us, Mapping):
                raise ValueError(f"{where}.delay_us must be an object of site -> microseconds")
            delays_ns = [0] * len(SITES)
            for site, value in delay_us.items():
                if site not in SITE_INDEX:
                    raise ValueError(f"{where}.delay_us: unknown site {site!r}")
                if site not in sites:
                    raise ValueError(
                        f"{where}.delay_us names {site!r}, which is not in "
                        f"{DEBUG_KEY}.sites; the captured graph has no node there")
                if not _is_int(value):
                    raise ValueError(
                        f"{where}.delay_us[{site!r}] must be an integer number "
                        f"of microseconds, got {value!r}")
                if value < 0 or value > MAX_DELAY_US:
                    raise ValueError(
                        f"{where}.delay_us[{site!r}]={value} is outside "
                        f"[0, {MAX_DELAY_US}] microseconds")
                delays_ns[SITE_INDEX[site]] = int(value) * 1000
            schedule.append((int(steps), tuple(delays_ns)))

        return cls(
            sites=sites,
            layers=layers,
            schedule=tuple(schedule),
            skip_steps_after_change=int(skip),
        )

    @property
    def num_levels(self) -> int:
        return len(self.schedule)

    @property
    def period_steps(self) -> int:
        return sum(steps for steps, _ in self.schedule)

    def level_at(self, step: int) -> Tuple[int, int]:
        """``(level, steps into that level)`` for 1-based decode ``step``.

        The schedule cycles: step ``period + 1`` is level 0's first step again,
        and its offset restarts at 0 so the post-change skip applies again.
        """
        if not _is_int(step) or step < 1:
            raise ValueError(f"decode step must be a positive integer, got {step!r}")
        position = (step - 1) % self.period_steps
        bounds = self._level_bounds()
        level = bisect.bisect_right(bounds, position)
        start = bounds[level - 1] if level > 0 else 0
        return level, position - start

    def _level_bounds(self) -> Tuple[int, ...]:
        bounds = []
        total = 0
        for steps, _ in self.schedule:
            total += steps
            bounds.append(total)
        return tuple(bounds)

    def signature(self) -> Hashable:
        """Hashable identity of the request; part of the graph capture signature."""
        return (DEBUG_KEY, self.sites, self.layers, self.schedule, self.skip_steps_after_change)


class _LevelStats:
    __slots__ = ("count", "sum_ms", "sumsq_ms", "min_ms", "max_ms", "skipped")

    def __init__(self) -> None:
        self.count = 0
        self.sum_ms = 0.0
        self.sumsq_ms = 0.0
        self.min_ms = math.inf
        self.max_ms = -math.inf
        self.skipped = 0

    def add(self, ms: float) -> None:
        self.count += 1
        self.sum_ms += ms
        self.sumsq_ms += ms * ms
        if ms < self.min_ms:
            self.min_ms = ms
        if ms > self.max_ms:
            self.max_ms = ms


class CriticalityProbe:
    """One captured graph's probe: the delay table, the marks, the samples.

    ``mark`` is host code run only while the segments' forward runs, i.e. at
    warmup and capture; at replay only the captured spin nodes execute. The
    per-forward mark count of each site is the slope denominator ``L_site``
    (spin nodes per step): it is recorded per whole-model forward and must be
    identical for every forward of the graph, or the injected delay per step
    would be ill-defined.
    """

    def __init__(
        self,
        spec: CriticalityProbeSpec,
        device: torch.device,
        *,
        launch: Optional[Callable[[torch.Tensor, int], None]] = None,
    ) -> None:
        if launch is None:
            from batchgen_kernels.common.spin_delay import (
                preload_spin_delay_kernel,
                spin_delay,
            )

            # Resolve the extension HERE, before warmup and capture: a missing
            # build then fails at probe construction with a named ImportError
            # instead of inside the capture.
            preload_spin_delay_kernel()
            launch = spin_delay
        self.spec = spec
        self.device = device
        self._launch = launch
        self._enabled = frozenset(SITE_INDEX[site] for site in spec.sites)
        self._layers = None if spec.layers is None else frozenset(spec.layers)
        # Graph-baked: the spin nodes capture this tensor's pointer. It lives
        # as long as the probe, which the worker holds for the graph's life.
        self.delay_dev = torch.zeros(len(SITES), dtype=torch.int64, device=device)
        table = torch.tensor(
            [list(delays) for _, delays in spec.schedule], dtype=torch.int64
        )
        # Written once, here, and never again: the async copies in
        # ``before_replay`` may still be reading a row when the host returns.
        self.levels_host = table.pin_memory() if device.type == "cuda" else table
        self._forward_counts: Optional[List[int]] = None
        self.layer_marks: Optional[Tuple[int, ...]] = None
        self._installed_level: Optional[int] = None
        self._stats = [_LevelStats() for _ in range(spec.num_levels)]

    def signature(self) -> Hashable:
        return self.spec.signature()

    # -- capture time --------------------------------------------------------

    def validate_num_layers(self, num_layers: int) -> None:
        if self.spec.layers is None:
            return
        out_of_range = [layer for layer in self.spec.layers if layer >= num_layers]
        if out_of_range:
            raise ValueError(
                f"{DEBUG_KEY}.layers {out_of_range} are out of range for a "
                f"{num_layers}-layer model")

    def begin_forward(self) -> None:
        self._forward_counts = [0] * len(SITES)

    def end_forward(self) -> None:
        counts = self._forward_counts
        self._forward_counts = None
        if counts is None:
            raise RuntimeError("CriticalityProbe.end_forward without begin_forward")
        counts = tuple(counts)
        if self.layer_marks is None:
            self.layer_marks = counts
        elif counts != self.layer_marks:
            raise RuntimeError(
                "GLM-5 criticality probe: two forwards of one graph marked "
                f"different spin-node counts ({dict(zip(SITES, self.layer_marks))} "
                f"vs {dict(zip(SITES, counts))}); the injected delay per step "
                "would be ill-defined")

    def mark(self, site: str, layer_idx: int) -> None:
        """Launch the site's spin node on the current stream (host, capture time)."""
        index = SITE_INDEX.get(site)
        if index is None:
            raise ValueError(f"GLM-5 criticality probe: unknown site {site!r}")
        if index not in self._enabled:
            return
        if self._layers is not None and int(layer_idx) not in self._layers:
            return
        self._launch(self.delay_dev, index)
        if self._forward_counts is not None:
            self._forward_counts[index] += 1

    # -- replay time -----------------------------------------------------------

    def begin_round(self) -> None:
        """Reset the samples; the next ``before_replay`` reinstalls its level."""
        self._installed_level = None
        self._stats = [_LevelStats() for _ in range(self.spec.num_levels)]

    def before_replay(self, step: int) -> Tuple[int, int]:
        """Install ``step``'s delay level; returns the sample tag for the step.

        Enqueues the copy only when the level changes. Call on the stream the
        replay is issued on, after the previous replay was issued.
        """
        level, offset = self.spec.level_at(step)
        if level != self._installed_level:
            self.delay_dev.copy_(self.levels_host[level], non_blocking=True)
            self._installed_level = level
        return level, offset

    def record(self, tag: Optional[Tuple[int, int]], elapsed_ms: float) -> None:
        """Add one harvested step time; ``tag`` comes from ``before_replay``."""
        if tag is None:
            return
        level, offset = tag
        stats = self._stats[level]
        if offset < self.spec.skip_steps_after_change:
            stats.skipped += 1
            return
        stats.add(float(elapsed_ms))

    # -- report ----------------------------------------------------------------

    def injected_ms_per_step(self, level: int, site: str) -> float:
        """``L_site * d_level(site)`` in ms; 0 before any forward was counted."""
        if self.layer_marks is None:
            return 0.0
        index = SITE_INDEX[site]
        return self.layer_marks[index] * self.spec.schedule[level][1][index] / 1.0e6

    def site_slope(self, site: str) -> Dict[str, Any]:
        """Least-squares slope of step time vs injected delay for ``site``.

        Uses the levels whose other sites are all at 0, so a multi-site
        schedule cannot attribute one site's delay to another. ``slope`` is
        ``None`` when it is undefined: no spin node was captured for the site
        (``L_site = 0``, e.g. a dense layer set or an ``all_short`` graph's
        indexer), fewer than two distinct injected delays, or under three
        samples.
        """
        index = SITE_INDEX[site]
        marks = 0 if self.layer_marks is None else self.layer_marks[index]
        result: Dict[str, Any] = {
            "site": site, "marks_per_step": marks, "slope": None,
            "stderr": None, "intercept_ms": None, "samples": 0,
            "max_residual_us": None, "reason": "",
        }
        if marks == 0:
            result["reason"] = "L_site=0: no spin node of this site in the graph"
            return result
        points = []
        for level, (steps, delays) in enumerate(self.spec.schedule):
            if any(delays[i] != 0 for i in range(len(SITES)) if i != index):
                continue
            stats = self._stats[level]
            if stats.count == 0:
                continue
            points.append((self.injected_ms_per_step(level, site), stats))
        n = sum(stats.count for _, stats in points)
        result["samples"] = n
        if len({x for x, _ in points}) < 2 or n < 3:
            result["reason"] = "fewer than two distinct injected delays or three samples"
            return result
        sum_x = sum(x * stats.count for x, stats in points)
        sum_y = sum(stats.sum_ms for _, stats in points)
        mean_x = sum_x / n
        mean_y = sum_y / n
        sxx = sum(stats.count * x * x for x, stats in points) - n * mean_x * mean_x
        sxy = sum(x * stats.sum_ms for x, stats in points) - n * mean_x * mean_y
        syy = sum(stats.sumsq_ms for _, stats in points) - n * mean_y * mean_y
        slope = sxy / sxx
        intercept = mean_y - slope * mean_x
        ssr = max(syy - slope * sxy, 0.0)
        result["slope"] = slope
        result["intercept_ms"] = intercept
        result["stderr"] = math.sqrt(ssr / (n - 2) / sxx)
        # Linearity: how far each level's mean sits off the fitted line.
        result["max_residual_us"] = max(
            abs(stats.sum_ms / stats.count - (intercept + slope * x)) * 1000.0
            for x, stats in points
        )
        return result

    def report(self) -> str:
        """Multi-line end-of-round summary: per-level stats, per-site slopes."""
        layers = "all" if self.spec.layers is None else ",".join(map(str, self.spec.layers))
        if self.layer_marks is None:
            marks_text = "none (no forward was captured)"
        else:
            marks_text = ",".join(
                f"{site}={self.layer_marks[SITE_INDEX[site]]}" for site in self.spec.sites
            )
        lines = [
            f"[CRITICALITY_PROBE] sites={','.join(self.spec.sites)} layers={layers} "
            f"marks_per_step={marks_text} skip_steps_after_change="
            f"{self.spec.skip_steps_after_change}"
        ]
        for level, (steps, delays) in enumerate(self.spec.schedule):
            stats = self._stats[level]
            delay_text = ",".join(
                f"{site}={delays[SITE_INDEX[site]] // 1000}us" for site in self.spec.sites
                if delays[SITE_INDEX[site]]
            ) or "none"
            injected_text = ",".join(
                f"{site}={self.injected_ms_per_step(level, site) * 1000.0:.1f}us"
                for site in self.spec.sites
            )
            if stats.count:
                mean = stats.sum_ms / stats.count
                var = max(stats.sumsq_ms / stats.count - mean * mean, 0.0)
                timing = (
                    f"n={stats.count} mean_ms={mean:.4f} std_ms={math.sqrt(var):.4f} "
                    f"min_ms={stats.min_ms:.4f} max_ms={stats.max_ms:.4f}"
                )
            else:
                timing = "n=0"
            lines.append(
                f"[CRITICALITY_PROBE]   level={level} steps={steps} delay={delay_text} "
                f"injected_per_step={injected_text} skipped={stats.skipped} {timing}"
            )
        for site in self.spec.sites:
            fit = self.site_slope(site)
            if fit["slope"] is None:
                lines.append(
                    f"[CRITICALITY_PROBE]   site={site} L_site={fit['marks_per_step']} "
                    f"slope=undefined ({fit['reason']}) samples={fit['samples']}"
                )
            else:
                lines.append(
                    f"[CRITICALITY_PROBE]   site={site} L_site={fit['marks_per_step']} "
                    f"slope={fit['slope']:.4f} stderr={fit['stderr']:.4f} "
                    f"intercept_ms={fit['intercept_ms']:.4f} "
                    f"max_level_residual_us={fit['max_residual_us']:.2f} "
                    f"samples={fit['samples']}"
                )
        return "\n".join(lines)
