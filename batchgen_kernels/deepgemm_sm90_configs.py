# ---------------------------------------------------------------------------- #
#  BatchGen                                                                     #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #

"""DeepGEMM's SM90 FP8 GEMM config selection, ported for JIT prewarming.

DeepGEMM JIT-compiles one kernel per set of template arguments and picks
those arguments on the host from the GEMM shape with a cycle cost model
(``get_best_config<SM90ArchSpec>``). A served call whose shape selects a
config no earlier call produced compiles inside the request. This module
reproduces that selection for the two kernels BatchGen's FP8 GEMMs reach on
Hopper, so a prewarm can enumerate the configs a shape range selects and
launch one representative of each:

  * ``fp8_gemm_nt`` (GemmType Normal, 1D2D, FP8 A/B K-major, BF16 output):
    M is not a template argument, and the config depends on M only through
    ``q = ceil(M / 16)``: the ``M <= 16`` / ``M <= 32`` block-M thresholds
    and every ``ceil(M / block_m)`` are functions of ``q``;
  * ``fp8_m_grouped_gemm_nt_masked`` (GemmType MGroupedMasked): N and K are
    compiled in, block M is 64 or 128 and the cost model reads
    ``expected_m``, so only ``ceil(expected_m / 64)`` matters.

The arithmetic is the C++ one, kept exactly: integer divisions, the float32
wave efficiency, truncation to int64, first-wins on equal cycles and the
candidate order cluster (1, 1), (1, 2), (2, 1) x block M x block N. The port
covers DeepGEMM ``SUPPORTED_DEEPGEMM_VERSION`` only; a different installed
version may choose differently, which ``installed_version`` lets a caller
detect.

Model-neutral and importable without torch, Triton or CUDA (numpy only).
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


SUPPORTED_DEEPGEMM_VERSION = "0.1.5.post3"

# Distribution names DeepGEMM is published under, in lookup order.
DEEPGEMM_DISTRIBUTIONS = ("sgl-deep-gemm", "deep-gemm")

# ``GemmType`` values as DeepGEMM numbers them.
GEMM_NORMAL = 0
GEMM_MASKED = 2

DEFAULT_NUM_SMS = 132
SMEM_CAPACITY = 232448
MAX_STAGES = 16
BLOCK_K = 128          # 128 bytes of FP8 per K block
WGMMA_M = 64
# FP8 A/B, BF16 C/D.
_AB_BYTES = 1
_CD_BYTES = 2
# 1D2D kernels: block N runs 16, 32, ... 192.
_BLOCK_N_CANDIDATES = tuple(range(16, 193, 16))
_CLUSTERS = ((1, 1), (1, 2), (2, 1))
_NEVER = np.iinfo(np.int64).max


def installed_version() -> Optional[str]:
    """The installed DeepGEMM distribution's version, or None."""
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover - Python < 3.8
        return None
    for name in DEEPGEMM_DISTRIBUTIONS:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _align(a: int, b: int) -> int:
    return _ceil_div(a, b) * b


def _swizzle_mode(nbytes: int) -> int:
    for mode in (128, 64, 32, 16):
        if nbytes % mode == 0:
            return mode
    raise ValueError(f"no swizzle mode divides {nbytes} bytes")


@dataclass(frozen=True)
class Sm90Fp8Config:
    """The layout, pipeline and launch values DeepGEMM selects for a shape."""

    block_m: int
    block_n: int
    cluster_m: int
    cluster_n: int
    num_stages: int
    swizzle_cd: int
    math_threads: int

    def kernel_key(self) -> tuple:
        """The template arguments that vary between shapes of one site.

        The others are fixed for a site: block K, the A/B swizzles, the TMA
        thread count, the group count, the SM count, the GEMM type, the
        output dtype and the epilogue (and, for the masked kernel, N and K).
        """
        return (self.block_m, self.block_n, self.swizzle_cd, self.num_stages,
                self.math_threads, self.cluster_m * self.cluster_n,
                self.cluster_n > 1)


def _num_stages(block_m: int, block_n: int, k: int) -> int:
    smem_cd = _align(block_m * block_n * _CD_BYTES, 1024)
    smem_barriers = MAX_STAGES * 8 * 2
    per_stage = (block_m * BLOCK_K * _AB_BYTES + block_n * BLOCK_K * _AB_BYTES
                 + _align(block_m * 4, 128))
    uniform_sfb = 1 if BLOCK_K % block_n == 0 else 2
    extra_sfb = _align(_ceil_div(k, BLOCK_K) * 4 * uniform_sfb, 8)
    extra = smem_cd + smem_barriers + extra_sfb
    return min((SMEM_CAPACITY - extra) // per_stage, MAX_STAGES)


def _block_ms(m: int, masked: bool) -> List[int]:
    if masked:
        return [64, 128]
    block_ms = [64, 128]
    if m <= 16:
        block_ms.append(16)
    if m <= 32:
        block_ms.append(32)
    block_ms.append(256)  # BF16 output
    return block_ms


def _candidates(m: int, n: int, k: int, masked: bool,
                num_sms: int) -> Tuple[Tuple[int, int, int, int], ...]:
    """``(block_m, block_n, cluster_m, cluster_n)`` in DeepGEMM's order."""
    # The list depends on m only through the 16- and 32-row thresholds.
    m_class = 0 if masked else (16 if m <= 16 else 32 if m <= 32 else 48)
    return _candidates_cached(m_class, n, k, masked, num_sms)


@lru_cache(maxsize=None)
def _candidates_cached(m: int, n: int, k: int, masked: bool,
                       num_sms: int) -> Tuple[Tuple[int, int, int, int], ...]:
    if num_sms % 2:
        raise ValueError(f"DeepGEMM requires an even SM count, got {num_sms}")
    out = []
    for cluster_m, cluster_n in _CLUSTERS:
        if num_sms % (cluster_m * cluster_n):
            continue
        for block_m in _block_ms(m, masked):
            for block_n in _BLOCK_N_CANDIDATES:
                # 1D2D unroll requirement.
                if block_n > BLOCK_K and (
                        block_n % (block_n - BLOCK_K) != 0
                        and BLOCK_K % (block_n - BLOCK_K) != 0):
                    continue
                # Multicast legality of the masked layout.
                if masked and _ceil_div(n, block_n) % (cluster_m * cluster_n):
                    continue
                if block_m > 128 and block_n > 128:
                    continue
                # A/B swizzles are 128 B (block K of FP8): always legal.
                stages = _num_stages(block_m, block_n, k)
                if stages < 3 or (block_m * block_n < 128 * 192 and stages < 4):
                    continue
                out.append((block_m, block_n, cluster_m, cluster_n))
    if not out:
        raise ValueError(f"no DeepGEMM SM90 layout fits n={n} k={k}")
    return tuple(out)


def _cycles(expected_m, n: int, k: int, groups: int, block_m: int,
            block_n: int, cluster_m: int, cluster_n: int,
            num_sms: int) -> np.ndarray:
    """``get_layout_info``'s ``num_cycles`` for an int64 array of expected M."""
    em = np.asarray(expected_m, dtype=np.int64)
    num_blocks = ((em + block_m - 1) // block_m) * _ceil_div(n, block_n) * groups
    num_waves = (num_blocks + num_sms - 1) // num_sms
    l2_bandwidth = int(min(64.0 * num_sms, 8e6 / 1.3e3))
    l1_bandwidth = 128 * num_sms
    bytes_l2_ab = k * (block_m // cluster_n + block_n // cluster_m) * _AB_BYTES
    bytes_l1_ab = k * (block_m + block_n) * _AB_BYTES
    bytes_l1_tc = (k * (max(WGMMA_M, block_m) + block_n) * _AB_BYTES
                   + block_m * block_n * _CD_BYTES)
    bytes_cd = block_m * block_n * _CD_BYTES
    l2_cycles = (bytes_l2_ab + bytes_cd) * num_blocks // l2_bandwidth
    l1_cycles = (bytes_l1_ab + bytes_l1_tc + bytes_cd) * num_blocks // l1_bandwidth
    efficiency = (num_blocks.astype(np.float32)
                  / (num_waves * num_sms).astype(np.float32))
    cycles = (np.maximum(l1_cycles, l2_cycles).astype(np.float32)
              / efficiency).astype(np.int64)
    if cluster_m * cluster_n > 1:
        cycles = np.where(num_waves <= 1, _NEVER, cycles)
    return cycles


def _cycles_scalar(expected_m: int, n: int, k: int, groups: int,
                   block_m: int, block_n: int, cluster_m: int, cluster_n: int,
                   num_sms: int) -> int:
    """``_cycles`` for one expected M, in scalar arithmetic: an independent
    statement of the same cost model, which the tests hold the vectorised
    one against."""
    num_blocks = (_ceil_div(expected_m, block_m) * _ceil_div(n, block_n)
                  * groups)
    num_waves = _ceil_div(num_blocks, num_sms)
    l2_bandwidth = int(min(64.0 * num_sms, 8e6 / 1.3e3))
    l1_bandwidth = 128 * num_sms
    bytes_l2_ab = k * (block_m // cluster_n + block_n // cluster_m) * _AB_BYTES
    bytes_l1_ab = k * (block_m + block_n) * _AB_BYTES
    bytes_l1_tc = (k * (max(WGMMA_M, block_m) + block_n) * _AB_BYTES
                   + block_m * block_n * _CD_BYTES)
    bytes_cd = block_m * block_n * _CD_BYTES
    l2_cycles = (bytes_l2_ab + bytes_cd) * num_blocks // l2_bandwidth
    l1_cycles = (bytes_l1_ab + bytes_l1_tc + bytes_cd) * num_blocks // l1_bandwidth
    efficiency = np.float32(num_blocks) / np.float32(num_waves * num_sms)
    cycles = int(np.float32(max(l1_cycles, l2_cycles)) / efficiency)
    if cluster_m * cluster_n > 1 and num_waves <= 1:
        return int(_NEVER)
    return cycles


def _config(layout: Tuple[int, int, int, int], k: int) -> Sm90Fp8Config:
    block_m, block_n, cluster_m, cluster_n = layout
    return Sm90Fp8Config(
        block_m=block_m, block_n=block_n, cluster_m=cluster_m,
        cluster_n=cluster_n, num_stages=_num_stages(block_m, block_n, k),
        swizzle_cd=_swizzle_mode(block_n * _CD_BYTES),
        math_threads=128 if block_m <= 64 else 256,
    )


def _select(expected_ms, candidates, n: int, k: int, groups: int,
            num_sms: int) -> np.ndarray:
    """Index of the winning candidate per expected M (first wins ties)."""
    table = np.stack([_cycles(expected_ms, n, k, groups, *c, num_sms)
                      for c in candidates])
    return np.argmin(table, axis=0)


def best_config(m: int, n: int, k: int, *, groups: int = 1,
                masked: bool = False, expected_m: Optional[int] = None,
                num_sms: int = DEFAULT_NUM_SMS) -> Sm90Fp8Config:
    """The config DeepGEMM selects for one call.

    Normal GEMMs model ``m`` itself; the masked kernel models
    ``expected_m`` (``m`` when not given), as its host wrapper does.
    """
    m, n, k, groups = int(m), int(n), int(k), int(groups)
    if min(m, n, k, groups) < 1:
        raise ValueError(f"DeepGEMM shape must be positive: m={m} n={n} "
                         f"k={k} groups={groups}")
    em = int(expected_m) if masked and expected_m else m
    best, best_cycles = None, None
    for layout in _candidates(m, n, k, masked, num_sms):
        cycles = _cycles_scalar(em, n, k, groups, *layout, num_sms)
        if best is None or cycles < best_cycles:
            best, best_cycles = layout, cycles
    return _config(best, k)


def normal_configs(n: int, k: int, m_max: int,
                   num_sms: int = DEFAULT_NUM_SMS) -> Dict[int, Sm90Fp8Config]:
    """``q -> config`` of the Normal kernel for ``q = 1 .. ceil(m_max / 16)``,
    where ``q = ceil(m / 16)`` stands for every ``m`` in ``(16q - 16, 16q]``."""
    n, k, m_max = int(n), int(k), int(m_max)
    if m_max < 1:
        raise ValueError(f"m_max must be positive, got {m_max}")
    q_max = _ceil_div(m_max, 16)
    configs: Dict[int, Sm90Fp8Config] = {}
    # q = 1 and q = 2 add the 16- and 32-row blocks to the candidates.
    for q in range(1, min(q_max, 2) + 1):
        configs[q] = best_config(16 * q, n, k, num_sms=num_sms)
    if q_max >= 3:
        candidates = _candidates(48, n, k, False, num_sms)
        qs = np.arange(3, q_max + 1, dtype=np.int64)
        winners = _select(16 * qs, candidates, n, k, 1, num_sms)
        layouts = {index: _config(candidates[index], k)
                   for index in np.unique(winners).tolist()}
        for q, index in zip(qs.tolist(), winners.tolist()):
            configs[q] = layouts[index]
    return configs


def normal_representative_ms(n: int, k: int, m_max: int,
                             num_sms: int = DEFAULT_NUM_SMS) -> List[int]:
    """The smallest ``m`` in ``[1, m_max]`` of each distinct Normal kernel
    key: one call per returned ``m`` compiles every key the range selects."""
    chosen: Dict[tuple, int] = {}
    for q, config in sorted(normal_configs(n, k, m_max, num_sms).items()):
        chosen.setdefault(config.kernel_key(), 16 * (q - 1) + 1)
    return sorted(chosen.values())


def masked_representative_expected_ms(
        n: int, k: int, groups: int, expected_ms: Iterable[int],
        num_sms: int = DEFAULT_NUM_SMS) -> List[int]:
    """The smallest ``expected_m`` of ``expected_ms`` per distinct masked
    kernel key."""
    values = sorted({int(v) for v in expected_ms})
    if not values:
        return []
    if values[0] < 1:
        raise ValueError(f"expected_m must be positive, got {values[0]}")
    n, k, groups = int(n), int(k), int(groups)
    candidates = _candidates(64, n, k, True, num_sms)
    winners = _select(values, candidates, n, k, groups, num_sms)
    chosen: Dict[tuple, int] = {}
    for value, index in zip(values, winners.tolist()):
        chosen.setdefault(_config(candidates[index], k).kernel_key(), value)
    return sorted(chosen.values())


def keys_for_normal_ms(n: int, k: int, ms: Sequence[int],
                       num_sms: int = DEFAULT_NUM_SMS) -> set:
    """The Normal kernel keys a set of row counts selects."""
    return {best_config(m, n, k, num_sms=num_sms).kernel_key() for m in ms}
