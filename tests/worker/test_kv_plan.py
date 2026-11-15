"""``kv_plan``: the pure host/GPU KV capacity arithmetic of the worker.

Every function here replaced inline worker code. The parity tests run the
OLD inline code -- copied verbatim from ``batchgen_worker.py`` at
``ad4a0942`` into this file as frozen reference functions (no runtime git
dependency) -- side by side with the new definition, over hand-picked edge
cases and seeded random inputs. A difference anywhere is a behaviour change.

The verdict feeds multi-rank MIN consensus, so it must be bit-identical on
identical gathered rows: the parity tests compare the whole tuple
``(ok, rows_usable, rows)`` including dict insertion order.
"""

from __future__ import annotations

import ast
import itertools
import math
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import pytest

from batchgen.worker import kv_plan
from batchgen.worker.kv_plan import (
    NodeCapacityReport,
    NodeCapacityVerdict,
    credit_reclaimable_pages,
    effective_chunk_size,
    host_capacity_due,
    prefill_owner_demand_pages,
    provisional_pool_gb,
    verdict_node_capacity,
)

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "batchgen" / "batchgen_worker.py"
PAGE = 64


# --------------------------------------------------------------------------- #
#  Gathered-row fakes: the worker reads every field through ``.item()``.
# --------------------------------------------------------------------------- #


class _Scalar:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


def _gathered(rows):
    return [[_Scalar(v) for v in row] for row in rows]


def _reports(rows):
    # The worker wrappers build exactly this from the gathered tensors.
    return [
        NodeCapacityReport(
            node_id=int(r[0]), need=int(r[1]), free=int(r[2]),
            total=int(r[3]), snapshot_error=int(r[4]),
        )
        for r in rows
    ]


# --------------------------------------------------------------------------- #
#  Frozen references (verbatim old code)
# --------------------------------------------------------------------------- #


def _old_prefill_verdict(gathered):
    # copied verbatim from batchgen_worker.py:6911-6940 @ ad4a0942
    node_needs: Dict[int, int] = {}
    node_capacity: Dict[int, Tuple[int, int]] = {}
    ok = True
    # A rejection that is merely "short of pages" is retryable by reclaim; a
    # broken snapshot is not, and the two must not be confused.
    rows_usable = True
    for item in gathered:
        reported_node = int(item[0].item())
        node_needs[reported_node] = node_needs.get(reported_node, 0) + int(
            item[1].item()
        )
        if int(item[4].item()) != 0:
            ok = False
            rows_usable = False
        if int(item[2].item()) >= 0:
            capacity = (int(item[2].item()), int(item[3].item()))
            previous = node_capacity.get(reported_node)
            if previous is not None and previous != capacity:
                ok = False
                rows_usable = False
            node_capacity[reported_node] = capacity
    for node, needed in node_needs.items():
        free_total = node_capacity.get(node)
        if free_total is None:
            ok = False
            rows_usable = False
            continue
        free_pages, total_pages = free_total
        if needed < 0 or needed > free_pages or needed > total_pages:
            ok = False
    # The returned rows when the reduced rows_usable is 1; the dict body is
    # copied verbatim from batchgen_worker.py:6949-6950 @ ad4a0942. Identical
    # gathered rows on every rank make the MIN reduction equal the local value.
    if not rows_usable:
        return ok, rows_usable, None
    return ok, rows_usable, {
        node: (needed, node_capacity[node][0])
        for node, needed in node_needs.items()
    }


def _old_growth_verdict(gathered):
    # copied verbatim from batchgen_worker.py:7446-7469 @ ad4a0942
    node_needs: Dict[int, int] = {}
    node_capacity: Dict[int, Tuple[int, int]] = {}
    ok = True
    for item in gathered:
        reported_node = int(item[0].item())
        node_needs[reported_node] = node_needs.get(reported_node, 0) + int(
            item[1].item()
        )
        if int(item[4].item()) != 0:
            ok = False
        if int(item[2].item()) >= 0:
            capacity = (int(item[2].item()), int(item[3].item()))
            previous = node_capacity.get(reported_node)
            if previous is not None and previous != capacity:
                ok = False
            node_capacity[reported_node] = capacity
    for node, needed in node_needs.items():
        capacity = node_capacity.get(node)
        if capacity is None:
            ok = False
            continue
        free_pages, total_pages = capacity
        if needed < 0 or needed > free_pages or needed > total_pages:
            ok = False
    return ok


def _old_owner_demand(self, owner_requests):
    # copied verbatim from batchgen_worker.py:6859-6871 @ ad4a0942
    local_need = sum(
        math.ceil(int(tokens) / self.PAGE_SIZE)
        for _, tokens in owner_requests
    )
    # Members' demands exclude their shared chains, but the pooled
    # segments themselves allocate one Host copy (mirrored per group, so
    # one count). Leaving them out over-admits the wave and the
    # allocation fails right after the preflight passed. Only the wave's
    # host-KV LEADER allocates segments, so only a rank with owner demand
    # reports them — at G>1 the followers reporting the same segments
    # would multiply the node demand by G and falsely reject the wave.
    if owner_requests:
        local_need += self._prefill_pool_segment_host_pages()
    return local_need


def _old_credit(stats, coordinator):
    # copied verbatim from batchgen_worker.py:5705-5723 @ ad4a0942
    releasable = max(0, int(coordinator.count_releasable_pages(0)))
    if releasable == 0:
        return stats
    total = int(stats["num_total_pages"])
    used = int(stats["num_used_pages"])
    # A retained page is charged to `used`, so crediting it moves the same
    # count. Clamped so a stale read can never make `used` negative or
    # free% exceed 100 -- a watermark fed an impossible snapshot would
    # preempt decode on every step.
    credited = min(releasable, used)
    free = int(stats["num_free_pages"]) + credited
    updated = dict(stats)
    updated["num_used_pages"] = used - credited
    updated["num_free_pages"] = free
    updated["num_reclaimable_prefix_pages"] = credited
    updated["free_percent"] = (
        int((free / total) * 100) if total > 0 else 100
    )
    return updated


class _Entry:
    PAGE_SIZE = PAGE


def _old_chunk(self, SequenceEntry=_Entry):
    # copied verbatim from batchgen_worker.py @ main post-#648 (the
    # max_decoding_length cap was removed there; chunks are sized by each
    # sequence's own decode budget at admission instead)
    if self.adaptive_chunk_sizer is not None:
        chunk = self.adaptive_chunk_sizer.get_chunk_size()
    else:
        chunk = self.host_kv_chunk_size
    # A boundary is the only place where host pages may be allocated.  Keep
    # one complete decision interval of runway so the next boundary is
    # reached while the metadata invariant still holds.  This removes the
    # unsafe post-sample emergency growth path.
    chunk = max(chunk, self.DECISION_INTERVAL)
    # Round up to page boundary
    chunk = math.ceil(chunk / SequenceEntry.PAGE_SIZE) * SequenceEntry.PAGE_SIZE
    return chunk


class _Log:
    def __init__(self):
        self.lines = []

    def warning(self, msg):
        self.lines.append(msg)


def _old_site1(self, total_mem_gb, used_mem_gb, logging):
    # copied verbatim from batchgen_worker.py:1772-1781 @ ad4a0942
    gpu_kv_cache_gb = total_mem_gb * self.gpu_memory_frac - used_mem_gb

    # Ensure positive value
    if gpu_kv_cache_gb <= 0:
        logging.warning(
            f"[GPU-KV] Calculated size is non-positive ({gpu_kv_cache_gb:.2f} GB). "
            f"Total: {total_mem_gb:.2f} GB × frac: {self.gpu_memory_frac} - used: {used_mem_gb:.2f} GB. "
            f"Using minimum 1 GB."
        )
        gpu_kv_cache_gb = 1.0
    return gpu_kv_cache_gb


def _old_site2(self, total_mem_gb, used_mem_gb):
    # copied verbatim from batchgen_worker.py:11854-11865 @ ad4a0942
    new_gpu_kv_cache_size = total_mem_gb * self.gpu_memory_frac - used_mem_gb
    if new_gpu_kv_cache_size > 0:
        self.gpu_kv_cache_size_gb = new_gpu_kv_cache_size
    else:
        # Fail-loud (contract invariant 3): a non-positive remainder is a
        # configuration error, never a silently accepted minimum.
        raise RuntimeError(
            f"[GPU-KV] Calculated pool size non-positive "
            f"({new_gpu_kv_cache_size:.2f} GB): total {total_mem_gb:.2f} GB "
            f"x frac {self.gpu_memory_frac} - used {used_mem_gb:.2f} GB. "
            "Lower resident memory or raise gpu_memory_frac."
        )
    return self.gpu_kv_cache_size_gb


def _old_growth_pages(seq, required_tokens):
    # copied verbatim from batchgen_worker.py:4473-4476 @ ad4a0942
    # (the append target is returned instead of appended)
    if required_tokens > int(seq.host_token_capacity):
        growth_pages = math.ceil(
            (required_tokens - int(seq.host_token_capacity)) / seq.PAGE_SIZE
        )
        return growth_pages
    return None


def _old_due(seq):
    # copied verbatim from batchgen_worker.py:6767-6768 @ ad4a0942
    if int(seq.host_token_capacity) <= int(seq.current_context_length):
        return True
    return False


# --------------------------------------------------------------------------- #
#  (a) verdict_node_capacity -- both call shapes
# --------------------------------------------------------------------------- #


def _new_prefill(rows):
    v = verdict_node_capacity(_reports(rows), want_rows=True)
    return v.ok, v.rows_usable, v.rows


def _new_growth(rows):
    v = verdict_node_capacity(_reports(rows), want_rows=False)
    assert v.rows_usable is None and v.rows is None
    return v.ok


EDGE_CASES = {
    # 2 nodes x 2 ranks, leader (local_rank 0) reports free/total.
    "fits": [(0, 3, 10, 20, 0), (0, 2, -1, -1, 0), (1, 5, 5, 8, 0), (1, 0, -1, -1, 0)],
    "short_of_pages": [(0, 11, 10, 20, 0), (0, 0, -1, -1, 0)],
    "need_exceeds_total": [(0, 9, 10, 8, 0)],
    "exact_fit": [(0, 10, 10, 10, 0)],
    "zero_need": [(0, 0, 0, 0, 0)],
    "negative_need": [(0, -1, 10, 20, 0)],
    "negative_sum_from_two_ranks": [(0, 3, 10, 20, 0), (0, -4, -1, -1, 0)],
    "snapshot_error": [(0, 1, 10, 20, 0), (0, 1, -1, -1, 1)],
    "snapshot_error_nonunit": [(0, 1, 10, 20, 7)],
    "snapshot_error_negative": [(0, 1, 10, 20, -1)],
    "missing_node_capacity": [(0, 1, 10, 20, 0), (1, 1, -1, -1, 0)],
    "same_node_mismatch_free": [(0, 1, 10, 20, 0), (0, 1, 9, 20, 0)],
    "same_node_mismatch_total": [(0, 1, 10, 20, 0), (0, 1, 10, 21, 0)],
    "same_node_agreement": [(0, 1, 10, 20, 0), (0, 1, 10, 20, 0)],
    "mismatch_then_short": [(0, 30, 10, 20, 0), (0, 0, 9, 20, 0)],
    "free_zero_reported": [(0, 0, 0, 5, 0)],
    "negative_node_id": [(-1, 2, 5, 5, 0)],
    "unordered_nodes": [(3, 1, -1, -1, 0), (1, 1, 4, 4, 0), (3, 1, 9, 9, 0)],
    "empty": [],
}


@pytest.mark.parametrize("name", sorted(EDGE_CASES))
def test_prefill_shape_matches_the_old_loop_on_edge_cases(name):
    rows = EDGE_CASES[name]
    old = _old_prefill_verdict(_gathered(rows))
    new = _new_prefill(rows)
    assert new == old
    if new[2] is not None:
        assert list(new[2]) == list(old[2])  # insertion order


@pytest.mark.parametrize("name", sorted(EDGE_CASES))
def test_growth_shape_matches_the_old_loop_on_edge_cases(name):
    rows = EDGE_CASES[name]
    assert _new_growth(rows) == _old_growth_verdict(_gathered(rows))


def _random_rows(rng):
    world = rng.choice([1, 2, 3, 4, 8, 16, 32])
    local_world = rng.choice([1, 2, 4, 8])
    nodes_free = {}
    rows = []
    for rank in range(world):
        node = rank // local_world
        if rng.random() < 0.05:
            node = rng.randint(-2, 6)  # corrupt id
        need = rng.randint(-3, 40) if rng.random() < 0.1 else rng.randint(0, 40)
        leader = rank % local_world == 0
        if leader or rng.random() < 0.1:
            base = nodes_free.setdefault(node, (rng.randint(0, 60), rng.randint(0, 80)))
            if rng.random() < 0.1:
                base = (base[0] + rng.choice([-1, 1]), base[1])
            if rng.random() < 0.05:
                base = (base[0], base[1] + 1)
            free, total = base
            if rng.random() < 0.05:
                free = -1  # a leader that reported nothing
        else:
            free, total = -1, -1
        err = 0
        if rng.random() < 0.05:
            err = rng.choice([1, 2, -1])
        rows.append((node, need, free, total, err))
    if rng.random() < 0.3:
        rng.shuffle(rows)
    return rows


def test_randomized_rows_match_both_old_loops():
    rng = random.Random(20261108)
    outcomes = {"prefill_ok": 0, "prefill_unusable": 0, "growth_ok": 0}
    for _ in range(5000):
        rows = _random_rows(rng)
        old_p = _old_prefill_verdict(_gathered(rows))
        new_p = _new_prefill(rows)
        assert new_p == old_p, rows
        if new_p[2] is not None:
            assert list(new_p[2]) == list(old_p[2]), rows
        old_g = _old_growth_verdict(_gathered(rows))
        assert _new_growth(rows) == old_g, rows
        outcomes["prefill_ok"] += old_p[0]
        outcomes["prefill_unusable"] += not old_p[1]
        outcomes["growth_ok"] += old_g
    # The generator must exercise every verdict branch, not only one.
    assert 0 < outcomes["prefill_ok"] < 5000
    assert 0 < outcomes["prefill_unusable"] < 5000
    assert 0 < outcomes["growth_ok"] < 5000


def test_growth_ok_always_equals_prefill_ok():
    # The two call shapes differ ONLY in whether rows are returned; the
    # same-node mismatch check is part of both.
    rng = random.Random(7)
    for _ in range(2000):
        rows = _random_rows(rng)
        v_p = verdict_node_capacity(_reports(rows), want_rows=True)
        v_g = verdict_node_capacity(_reports(rows), want_rows=False)
        assert v_p.ok == v_g.ok


def test_mismatch_check_is_unconditional_for_the_growth_shape():
    rows = [(0, 1, 10, 20, 0), (0, 1, 9, 20, 0)]
    assert verdict_node_capacity(_reports(rows), want_rows=False).ok is False


def test_rows_report_need_and_free_per_node():
    rows = [(0, 3, 10, 20, 0), (0, 2, -1, -1, 0), (1, 5, 6, 8, 0)]
    v = verdict_node_capacity(_reports(rows), want_rows=True)
    assert v == NodeCapacityVerdict(ok=True, rows_usable=True, rows={0: (5, 10), 1: (5, 6)})


def test_short_of_pages_is_rejected_but_rows_stay_usable():
    v = verdict_node_capacity(_reports([(0, 11, 10, 20, 0)]), want_rows=True)
    assert v.ok is False and v.rows_usable is True and v.rows == {0: (11, 10)}


def test_reports_are_frozen():
    r = NodeCapacityReport(node_id=0, need=1, free=2, total=3, snapshot_error=0)
    with pytest.raises(Exception):
        r.need = 5  # type: ignore[misc]


# --------------------------------------------------------------------------- #
#  owner demand
# --------------------------------------------------------------------------- #


def _owner_self(segment_pages):
    calls = []

    def _segments():
        calls.append(1)
        return segment_pages

    return SimpleNamespace(PAGE_SIZE=PAGE, _prefill_pool_segment_host_pages=_segments), calls


def _new_owner_demand(self, owner_requests):
    # The worker call, verbatim in shape.
    return prefill_owner_demand_pages(
        [int(tokens) for _, tokens in owner_requests],
        page_size=self.PAGE_SIZE,
        pool_segment_pages=(
            self._prefill_pool_segment_host_pages() if owner_requests else 0
        ),
    )


@pytest.mark.parametrize("segment_pages", [0, 1, 7])
@pytest.mark.parametrize(
    "tokens",
    [[], [0], [1], [63], [64], [65], [128, 1, 64], [10**6 + 3], [2**53 + 1]],
)
def test_owner_demand_matches_the_old_inline_sum(tokens, segment_pages):
    reqs = [(f"u{i}", t) for i, t in enumerate(tokens)]
    old_self, old_calls = _owner_self(segment_pages)
    new_self, new_calls = _owner_self(segment_pages)
    assert _new_owner_demand(new_self, reqs) == _old_owner_demand(old_self, reqs)
    # Segments are charged only by a rank with owner demand (the wave leader).
    assert len(new_calls) == len(old_calls) == (1 if reqs else 0)


def test_owner_demand_random_parity():
    rng = random.Random(3)
    for _ in range(2000):
        reqs = [(str(i), rng.randint(0, 5000)) for i in range(rng.randint(0, 6))]
        seg = rng.randint(0, 9)
        a, _ = _owner_self(seg)
        b, _ = _owner_self(seg)
        assert _new_owner_demand(b, reqs) == _old_owner_demand(a, reqs)


# --------------------------------------------------------------------------- #
#  (b) credit_reclaimable_pages
# --------------------------------------------------------------------------- #


def _stats(free, used, total, **extra):
    s = {"num_free_pages": free, "num_used_pages": used, "num_total_pages": total}
    s.update(extra)
    return s


def _coord(n):
    return SimpleNamespace(count_releasable_pages=lambda node: n)


@pytest.mark.parametrize(
    "stats,releasable",
    [
        (_stats(10, 90, 100), 0),
        (_stats(10, 90, 100), -5),       # stale negative read clamps to 0
        (_stats(10, 90, 100), 30),
        (_stats(10, 90, 100), 90),
        (_stats(10, 90, 100), 500),      # releasable > used clamps to used
        (_stats(0, 0, 0), 4),            # total=0 -> free_percent 100
        (_stats(3, 0, 3), 4),            # nothing used -> credited 0
        (_stats(1, 2, 3, free_percent=33, node_id=5), 1),
    ],
)
def test_credit_matches_the_old_body(stats, releasable):
    old = _old_credit(dict(stats), _coord(releasable))
    new = credit_reclaimable_pages(dict(stats), releasable)
    assert new == old
    assert list(new) == list(old)


@pytest.mark.parametrize("releasable", [0, -1])
def test_credit_identity_when_nothing_is_releasable(releasable):
    stats = _stats(10, 90, 100)
    assert credit_reclaimable_pages(stats, releasable) is stats


def test_credit_never_mutates_its_input():
    stats = _stats(10, 90, 100)
    before = dict(stats)
    out = credit_reclaimable_pages(stats, 40)
    assert stats == before and out is not stats
    assert out["num_used_pages"] == 50 and out["num_free_pages"] == 50
    assert out["num_reclaimable_prefix_pages"] == 40 and out["free_percent"] == 50


def test_credit_random_parity():
    rng = random.Random(11)
    for _ in range(3000):
        total = rng.randint(0, 300)
        used = rng.randint(0, total)
        stats = _stats(total - used, used, total)
        rel = rng.randint(-5, 400)
        assert credit_reclaimable_pages(dict(stats), rel) == _old_credit(dict(stats), _coord(rel))


# --------------------------------------------------------------------------- #
#  (c) provisional_pool_gb: floor vs fail-loud
# --------------------------------------------------------------------------- #


def _new_site1(self, total_mem_gb, used_mem_gb, logging):
    # The worker's _calculate_gpu_kv_cache_size rank-0 branch, in shape.
    calculated_gb = provisional_pool_gb(
        total_mem_gb, self.gpu_memory_frac, used_mem_gb, floor_gb=None
    )
    gpu_kv_cache_gb = provisional_pool_gb(
        total_mem_gb, self.gpu_memory_frac, used_mem_gb, floor_gb=1.0
    )
    if calculated_gb <= 0:
        logging.warning(
            f"[GPU-KV] Calculated size is non-positive ({calculated_gb:.2f} GB). "
            f"Total: {total_mem_gb:.2f} GB × frac: {self.gpu_memory_frac} - used: {used_mem_gb:.2f} GB. "
            f"Using minimum 1 GB."
        )
    return gpu_kv_cache_gb


def _new_site2(self, total_mem_gb, used_mem_gb):
    new_gpu_kv_cache_size = provisional_pool_gb(
        total_mem_gb, self.gpu_memory_frac, used_mem_gb, floor_gb=None
    )
    if new_gpu_kv_cache_size > 0:
        self.gpu_kv_cache_size_gb = new_gpu_kv_cache_size
    else:
        raise RuntimeError(
            f"[GPU-KV] Calculated pool size non-positive "
            f"({new_gpu_kv_cache_size:.2f} GB): total {total_mem_gb:.2f} GB "
            f"x frac {self.gpu_memory_frac} - used {used_mem_gb:.2f} GB. "
            "Lower resident memory or raise gpu_memory_frac."
        )
    return self.gpu_kv_cache_size_gb


SIZING = [
    (95.0, 0.9, 20.0),
    (95.0, 0.9, 85.5),    # exactly zero -> non-positive
    (95.0, 0.9, 90.0),    # negative
    (80.0, 0.93, 74.4),
    (139.8, 0.88, 60.123),
    (0.0, 0.9, 0.0),
    (95.0, float("nan"), 1.0),
    (95.0, 0.9, -3.0),
]


@pytest.mark.parametrize("total,frac,used", SIZING)
def test_floor_site_matches_the_old_warn_and_clamp(total, frac, used):
    self_ = SimpleNamespace(gpu_memory_frac=frac)
    old_log, new_log = _Log(), _Log()
    old = _old_site1(self_, total, used, old_log)
    new = _new_site1(self_, total, used, new_log)
    assert (math.isnan(old) and math.isnan(new)) or new == old
    assert new_log.lines == old_log.lines


@pytest.mark.parametrize("total,frac,used", SIZING)
def test_fail_loud_site_matches_the_old_raise(total, frac, used):
    def run(fn):
        s = SimpleNamespace(gpu_memory_frac=frac, gpu_kv_cache_size_gb=None)
        try:
            return ("ok", fn(s, total, used))
        except RuntimeError as exc:
            return ("raise", str(exc))
    old, new = run(_old_site2), run(_new_site2)
    assert old == new


def test_floor_is_only_substituted_for_a_non_positive_value():
    assert provisional_pool_gb(100.0, 0.9, 10.0, floor_gb=1.0) == 100.0 * 0.9 - 10.0
    assert provisional_pool_gb(100.0, 0.9, 90.0, floor_gb=1.0) == 1.0
    assert provisional_pool_gb(100.0, 0.9, 90.0, floor_gb=None) == 100.0 * 0.9 - 90.0
    # NaN is not <= 0: it passes through, as the old inline clamp let it.
    assert math.isnan(provisional_pool_gb(1.0, float("nan"), 0.0, floor_gb=1.0))


def test_floor_gb_is_keyword_only_and_required():
    with pytest.raises(TypeError):
        provisional_pool_gb(1.0, 0.5, 0.0)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        provisional_pool_gb(1.0, 0.5, 0.0, 1.0)  # type: ignore[misc]


# --------------------------------------------------------------------------- #
#  (e) effective_chunk_size
# --------------------------------------------------------------------------- #


def _chunk_self(chunk, interval, adaptive=None):
    return SimpleNamespace(
        adaptive_chunk_sizer=adaptive, host_kv_chunk_size=chunk,
        DECISION_INTERVAL=interval,
    )


def _new_chunk(self):
    # The worker delegate, in shape.
    if self.adaptive_chunk_sizer is not None:
        chunk = self.adaptive_chunk_sizer.get_chunk_size()
    else:
        chunk = self.host_kv_chunk_size
    return effective_chunk_size(
        chunk,
        decision_interval=self.DECISION_INTERVAL,
        page_size=_Entry.PAGE_SIZE,
    )


@pytest.mark.parametrize(
    "chunk,interval",
    list(itertools.product(
        [0, 1, 63, 64, 65, 127, 128, 4096, 5000],
        [0, 1, 64, 128, 130],
    )),
)
def test_chunk_matches_the_old_method(chunk, interval):
    s = _chunk_self(chunk, interval)
    assert _new_chunk(s) == _old_chunk(s)


def test_chunk_uses_the_adaptive_sizer_when_present():
    sizer = SimpleNamespace(get_chunk_size=lambda: 300)
    s = _chunk_self(64, 128, adaptive=sizer)
    assert _new_chunk(s) == _old_chunk(s) == 320


def test_chunk_rounds_up_and_keeps_one_decision_interval():
    assert effective_chunk_size(64, decision_interval=128, page_size=64) == 128
    assert effective_chunk_size(65, decision_interval=0, page_size=64) == 128
    assert effective_chunk_size(4096, decision_interval=1, page_size=64) == 4096


# --------------------------------------------------------------------------- #
#  host_capacity_due
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cap,ctx", list(itertools.product([0, 1, 63, 64, 65], [0, 1, 64, 65])))
def test_capacity_due_matches_the_old_comparison(cap, ctx):
    seq = SimpleNamespace(host_token_capacity=cap, current_context_length=ctx)
    assert host_capacity_due(int(cap), int(ctx)) == _old_due(seq)


# --------------------------------------------------------------------------- #
#  (d) append growth: SequenceEntry.get_host_growth_pages owns the term
# --------------------------------------------------------------------------- #


def _owner_growth_pages(seq, chunk):
    # Run the REAL owner (sequence.py) on a duck-typed entry.
    from batchgen.sequence import SequenceEntry

    return SequenceEntry.get_host_growth_pages(seq, chunk)


def _guards_pass(write_pos, capacity, budget):
    # The worker's preceding raises in _ensure_host_kv_append_capacity:
    # write_pos >= 0, capacity > 0, required <= budget.
    required = write_pos + 1
    return write_pos >= 0 and capacity > 0 and required <= budget


def test_growth_pages_owner_equals_the_inline_derivation_where_reachable():
    from batchgen.sequence import SequenceEntry

    page = SequenceEntry.PAGE_SIZE
    checked = 0
    for capacity in [1, 2, 63, 64, 65, 127, 128, 129, 640]:
        for budget in [1, 64, 65, 128, 129, 640, 641, 4096]:
            for write_pos in range(0, 700):
                if not _guards_pass(write_pos, capacity, budget):
                    continue
                seq = SimpleNamespace(
                    host_token_capacity=capacity, kv_token_budget=budget,
                    PAGE_SIZE=page,
                )
                required = write_pos + 1
                old = _old_growth_pages(seq, required)
                if old is None:
                    continue  # no growth branch
                new = _owner_growth_pages(seq, required - int(seq.host_token_capacity))
                assert new == old, (capacity, budget, write_pos)
                assert type(new) is type(old) is int
                checked += 1
    assert checked > 1000


def test_growth_pages_random_parity_where_reachable():
    from batchgen.sequence import SequenceEntry

    rng = random.Random(5)
    checked = 0
    for _ in range(20000):
        capacity = rng.randint(1, 10**7)
        budget = rng.randint(capacity, capacity + 10**6)
        write_pos = rng.randint(capacity - 1, budget - 1)
        seq = SimpleNamespace(
            host_token_capacity=capacity, kv_token_budget=budget,
            PAGE_SIZE=SequenceEntry.PAGE_SIZE,
        )
        old = _old_growth_pages(seq, write_pos + 1)
        if old is None:
            continue
        assert _owner_growth_pages(seq, write_pos + 1 - capacity) == old
        checked += 1
    assert checked > 10000


def test_growth_pages_owner_caps_only_outside_the_reachable_domain():
    # Documented: the owner caps at the remaining budget. At the append site
    # a required_tokens above the budget raises BEFORE the growth branch, so
    # the cap can never engage there. This pins that the two differ only on
    # inputs the guards reject.
    seq = SimpleNamespace(host_token_capacity=64, kv_token_budget=100, PAGE_SIZE=64)
    required = 200  # > budget: the worker raises "exceed token budget" first
    assert not _guards_pass(required - 1, 64, 100)
    assert _old_growth_pages(seq, required) == 3
    assert _owner_growth_pages(seq, required - 64) == 1


def test_append_site_calls_the_owner_and_no_inline_ceil():
    # S10c: the append-capacity check runs in KVCapacityEngine (worker shape).
    from tests.worker_method_source import worker_shaped_method_node

    fn = worker_shaped_method_node("_ensure_host_kv_append_capacity")
    calls = [ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)]
    assert calls.count("seq.get_host_growth_pages") == 1
    assert "math.ceil" not in calls


# --------------------------------------------------------------------------- #
#  Leaf invariant
# --------------------------------------------------------------------------- #


def test_kv_plan_imports_stdlib_only():
    tree = ast.parse(Path(kv_plan.__file__).read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0])
    mods.discard("__future__")
    assert mods <= set(sys.stdlib_module_names), mods
    assert "torch" not in mods and "batchgen" not in mods


# --------------------------------------------------------------------------- #
#  Worker wrappers: real method bodies compiled onto fakes (multi-rank)
# --------------------------------------------------------------------------- #


def _compile_worker_method(name, namespace):
    # S10b: the host-capacity collectives run in KVCapacityEngine; compile
    # the engine body in the worker's shape.
    from tests.worker_method_source import delegated_method_names, worker_shaped_method_node

    if name in delegated_method_names():
        module = ast.fix_missing_locations(
            ast.Module(body=[worker_shaped_method_node(name)], type_ignores=[])
        )
        exec(compile(module, str(WORKER), "exec"), namespace)
        return namespace[name]
    tree = ast.parse(WORKER.read_text())
    cls = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "BatchGenWorker"
    )
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    exec(compile(module, str(WORKER), "exec"), namespace)
    return namespace[name]


class _Dist:
    """all_gather fills scripted peer rows; all_reduce MINs scripted peers."""

    class ReduceOp:
        MIN = "min"

    def __init__(self, torch, rank, peer_rows, peer_decisions):
        self.torch = torch
        self.rank = rank
        self.peer_rows = peer_rows
        self.peer_decisions = peer_decisions

    def is_initialized(self):
        return True

    def all_gather(self, out, tensor):
        rows = list(self.peer_rows)
        rows.insert(self.rank, tensor.tolist())
        for slot, row in zip(out, rows):
            slot.copy_(self.torch.tensor(row, dtype=tensor.dtype))

    def all_reduce(self, tensor, op):
        assert op == self.ReduceOp.MIN
        for peer in self.peer_decisions:
            tensor.copy_(self.torch.minimum(tensor, self.torch.tensor(peer, dtype=tensor.dtype)))


def _wrapper_worker(torch, dist, *, rank, world, local_world, free, total, fail=False):
    from tests.worker.worker_fakes import attach_identity

    def get_stats():
        if fail:
            raise RuntimeError("stats down")
        return SimpleNamespace(num_free_pages=free, num_total_pages=total)

    worker = SimpleNamespace(
        PAGE_SIZE=PAGE, torch_device="cpu",
        host_paged_kv_worker_view=SimpleNamespace(get_stats=get_stats),
        _prefill_pool_segment_host_pages=lambda: 0,
    )
    attach_identity(worker, rank=rank, world_size=world, local_world_size=local_world)
    ns = {
        "math": math, "logging": SimpleNamespace(exception=lambda *a, **k: None),
        "torch": torch, "dist": dist, "Dict": Dict, "Tuple": Tuple,
        "List": List, "Optional": Optional,
        "NodeCapacityReport": NodeCapacityReport,
        "prefill_owner_demand_pages": prefill_owner_demand_pages,
        "verdict_node_capacity": verdict_node_capacity,
    }
    for name in ("_collective_prefill_host_capacity_ok", "_collective_host_kv_growth_capacity_ok"):
        setattr(worker, name, types.MethodType(_compile_worker_method(name, dict(ns)), worker))
    return worker


@pytest.mark.parametrize(
    "peer_rows,peer_decisions,expected",
    [
        # rank 0 (node leader, needs 2 pages) + rank 1 (needs 3): fits.
        ([[0, 3, -1, -1, 0]], [[1, 1]], (True, {0: (5, 10)})),
        # short of pages: rejected, rows still usable for reclaim.
        ([[0, 9, -1, -1, 0]], [[0, 1]], (False, {0: (11, 10)})),
        # a peer's stats failure: rows unusable.
        ([[0, 1, -1, -1, 1]], [[0, 0]], (False, None)),
        # same-node disagreement: rows unusable.
        ([[0, 1, 9, 20, 0]], [[0, 0]], (False, None)),
        # local verdict fine but a peer reduced rows_usable to 0.
        ([[0, 1, -1, -1, 0]], [[1, 0]], (True, None)),
    ],
)
def test_prefill_wrapper_shapes_the_reduced_decision(peer_rows, peer_decisions, expected):
    torch = pytest.importorskip("torch")
    dist = _Dist(torch, 0, peer_rows, peer_decisions)
    worker = _wrapper_worker(torch, dist, rank=0, world=2, local_world=2, free=10, total=20)
    assert worker._collective_prefill_host_capacity_ok([("a", 2 * PAGE)]) == expected


@pytest.mark.parametrize(
    "peer_rows,peer_decisions,expected",
    [
        ([[0, 3, -1, -1, 0]], [[1]], True),
        ([[0, 9, -1, -1, 0]], [[0]], False),
        ([[0, 1, 9, 20, 0]], [[0]], False),   # mismatch rejects growth too
        ([[0, 1, -1, -1, 0]], [[0]], False),  # a peer vetoes
    ],
)
def test_growth_wrapper_reduces_a_single_ok(peer_rows, peer_decisions, expected):
    torch = pytest.importorskip("torch")
    dist = _Dist(torch, 0, peer_rows, peer_decisions)
    worker = _wrapper_worker(torch, dist, rank=0, world=2, local_world=2, free=10, total=20)
    assert worker._collective_host_kv_growth_capacity_ok([(1, 2)]) is expected


def test_single_rank_wrappers_keep_their_early_returns():
    torch = pytest.importorskip("torch")
    dist = _Dist(torch, 0, [], [])
    ok_worker = _wrapper_worker(torch, dist, rank=0, world=1, local_world=1, free=10, total=20)
    assert ok_worker._collective_prefill_host_capacity_ok([("a", 2 * PAGE)]) == (True, {0: (2, 10)})
    assert ok_worker._collective_host_kv_growth_capacity_ok([(1, 11)]) is False
    bad = _wrapper_worker(torch, dist, rank=0, world=1, local_world=1, free=10, total=20, fail=True)
    assert bad._collective_prefill_host_capacity_ok([("a", 1)]) == (False, None)
    assert bad._collective_host_kv_growth_capacity_ok([(1, 1)]) is False
