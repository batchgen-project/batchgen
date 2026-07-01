"""Unit tests for prefill DP-rank load balancing (_assign_admitted_sequences_to_ranks).

Kimi-K2.5 prefill is embarrassingly-parallel DP: no cross-rank collective inside the
forward, only a single trailing barrier, so prefill wall = max_rank(T_rank). The legacy
balancer minimized per-rank Sum(L^2) — a correct proxy for block-diagonal varlen attention
but blind to the LINEAR MoE/token term and the per-seq term — and was node-blind with a
lowest-index tie-break, so it piled the FFD remainder onto node0.

These CPU-only tests (no torch/GPU) mirror the balancer's assignment logic exactly and
assert on the per-rank makespan spread under a TRUE multi-term cost oracle, before (legacy
Sum(L^2)) and after (multi-term cost + node-aware tie-break). They keep the constants in
sync with batchgen_worker.py::_assign_admitted_sequences_to_ranks — cannot import that
module (it JIT-loads a CUDA op at import).
"""
import math

NUM_GPUS_PER_NODE, TOKEN_CAP, WS = 8, 262144, 16
# True-cost ORACLE (what real wall-time looks like): attention L^2 + MoE tokens + per-seq
# fixed + per-micro-batch. Illustrative physically-motivated coeffs (attention:MoE ~ 80:20
# for a lone 64k seq).
A, B_ORACLE, G_ORACLE, D_ORACLE = 1.0, 16000.0, 1e8, 5e8
# Balancer's cost(L) coeffs — must match batchgen_worker BATCHGEN_L2_BETA / _GAMMA defaults.
BETA, GAMMA = 16000.0, 1e8


def true_cost(lengths):
    """Oracle per-rank wall-time proxy including the terms the legacy metric omits."""
    if not lengths:
        return 0.0
    return (A * sum(L * L for L in lengths)
            + B_ORACLE * sum(lengths)
            + G_ORACLE * len(lengths)
            + D_ORACLE * max(1, math.ceil(sum(lengths) / TOKEN_CAP)))


def _per_rank_lengths(assign):
    per = [[] for _ in range(WS)]
    for r, L in assign:
        per[r].append(L)
    return per


def makespan_spread(assign):
    costs = [true_cost(x) for x in _per_rank_lengths(assign)]
    lo = min(c for c in costs if c > 0) if any(costs) else 0
    hi = max(costs)
    return (hi / lo if lo > 0 else float("inf")), costs


def node_counts(assign):
    c = [0] * WS
    for r, _ in assign:
        c[r] += 1
    return sum(c[:NUM_GPUS_PER_NODE]), sum(c[NUM_GPUS_PER_NODE:]), c


def assign_legacy(lengths):
    """Legacy: pure Sum(L^2), flat-16 argmin, lowest-index tie-break."""
    load = [0.0] * WS
    out = []
    for L in sorted(lengths, reverse=True):
        r = min(range(WS), key=lambda r: load[r])
        out.append((r, L))
        load[r] += float(L) * float(L)
    return out


def _seq_cost(L):
    L = float(L)
    return L * L + BETA * L + GAMMA


def assign_fixed(lengths):
    """Fixed: multi-term cost(L) LPT + node-aware tie-break (mirrors the worker)."""
    cost = [0.0] * WS
    out = []
    npn = NUM_GPUS_PER_NODE

    def node_load(nd):
        return sum(cost[nd * npn:nd * npn + npn])

    for L in sorted(lengths, reverse=True):
        r = min(range(WS), key=lambda r: (cost[r], node_load(r // npn), r))
        out.append((r, L))
        cost[r] += _seq_cost(L)
    return out


def test_uniform_documents_remainder_bias():
    """100 x 64k: legacy degenerates to count-balance and piles the remainder on node0."""
    L = [64000] * 100
    legacy, fixed = assign_legacy(L), assign_fixed(L)

    n0l, n1l, counts = node_counts(legacy)
    assert sorted(counts, reverse=True) == [7, 7, 7, 7] + [6] * 12, counts  # count-balance degeneracy
    assert (n0l, n1l) == (52, 48)  # legacy: FFD remainder all on node0's low ranks

    n0f, n1f, _ = node_counts(fixed)
    assert (n0f, n1f) == (50, 50)  # FIXED: node-aware tie-break spreads remainder across nodes

    # The residual ~1/16 tail for perfectly-uniform lengths is irreducible without chunked
    # prefill (a larger change); the fix removes the NODE imbalance, not this tail.
    s_legacy, _ = makespan_spread(legacy)
    s_fixed, _ = makespan_spread(fixed)
    assert 1.10 < s_legacy < 1.20
    assert s_fixed <= s_legacy + 0.02


def test_skewed_reproduces_node0_first():
    """Equal-Sum(L^2) trap: legacy says 'perfectly balanced' but real makespan is ~3x off,
    with node1 the straggler (== POIS's 'node0 done, node1 still working')."""
    L = [64000] * 8 + [8000] * 512  # 8 long (one per node0 rank) + 512 short
    legacy, fixed = assign_legacy(L), assign_fixed(L)

    # Legacy Sum(L^2) is blind: each rank gets one 64k OR 64 x 8k, both Sum(L^2)=4.096e9.
    l2 = [0.0] * WS
    for r, x in legacy:
        l2[r] += x * x
    assert max(l2) / min(l2) < 1.01  # metric declares near-perfect

    s_legacy, costs = makespan_spread(legacy)
    assert s_legacy > 2.5  # real makespan ~3x off
    # node1 (short-seq ranks) is the straggler -> node0 finishes FIRST
    assert max(costs[NUM_GPUS_PER_NODE:]) > 2.0 * max(costs[:NUM_GPUS_PER_NODE])

    s_fixed, _ = makespan_spread(fixed)
    assert s_fixed < 1.20  # FIXED: mixing long+short across ranks equalizes real time


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
