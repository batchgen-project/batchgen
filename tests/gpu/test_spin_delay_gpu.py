# ---------------------------------------------------------------------------- #
#  BatchGen — GLM-5 decode criticality probe                                    #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""REMOTE-ONLY GPU gate: the run-time-duration spin delay kernel.

Run on a GPU machine (single device is enough), never on the dev box:

    pytest -o log_cli=true -o log_cli_level=INFO tests/gpu/test_spin_delay_gpu.py

The measured numbers are logged (``log_cli``) as well as asserted, because
they are the probe's calibration: the delay-0 cost of one spin node is the
baseline shift the probe reports, and the delay error bounds how finely a
criticality slope can be resolved.

  A  Delay accuracy, standalone and inside a captured graph, at 10 / 30 /
     100 us: the median elapsed time is at least the request and exceeds it
     by at most the globaltimer granularity plus a small launch allowance.
  B  Zero-delay cost: one spin node at delay 0 inside a graph costs a bounded
     number of microseconds (median over a 100-node chain).
  C  The delay of ONE captured graph changes between replays when the table
     is updated by an async copy from a pinned row on the replay stream, with
     no host synchronization between the copy and the replay -- the probe's
     steady-state contract.
  D  Argument validation fails loud (wrong dtype, out-of-range site).
"""

import logging
import statistics

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="spin_delay is a CUDA kernel"
)

logger = logging.getLogger(__name__)

SITE = 3
TABLE_SIZE = 11
REPEATS = 50
# Allowance above the request: globaltimer granularity (up to ~1 us on some
# parts), one __nanosleep quantum, and the launch / event overhead. Generous on
# purpose: the probe fits a slope, so a constant offset cancels; what must not
# happen is a delay that is SHORT or that scales wrongly with the request.
SLACK_US = 6.0


def _spin():
    from batchgen_kernels.common.spin_delay import preload_spin_delay_kernel, spin_delay

    preload_spin_delay_kernel()
    return spin_delay


def _table(delay_us: int) -> torch.Tensor:
    table = torch.zeros(TABLE_SIZE, dtype=torch.int64, device="cuda")
    table[SITE] = int(delay_us) * 1000
    return table


def _time_us(fn, repeats=REPEATS):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(repeats):
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(samples)


@pytest.mark.parametrize("delay_us", [10, 30, 100])
def test_standalone_delay_accuracy(delay_us):
    spin = _spin()
    table = _table(delay_us)
    zero = _table(0)
    spin(zero, SITE)
    torch.cuda.synchronize()
    base = _time_us(lambda: spin(zero, SITE))
    got = _time_us(lambda: spin(table, SITE))
    logger.info(
        "spin_delay standalone: request=%d us median=%.2f us delay0_median=%.2f us",
        delay_us, got, base,
    )
    assert got >= delay_us, f"spin returned early: {got:.2f} us < {delay_us} us"
    assert got <= delay_us + SLACK_US, (
        f"spin overshoots: {got:.2f} us for {delay_us} us (delay-0 {base:.2f} us)"
    )


def _graph_chain(spin, table, nodes):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(nodes):
            spin(table, SITE)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(nodes):
            spin(table, SITE)
    torch.cuda.synchronize()
    return graph


@pytest.mark.parametrize("delay_us", [10, 30, 100])
def test_in_graph_delay_accuracy(delay_us):
    spin = _spin()
    nodes = 8
    table = _table(0)
    graph = _graph_chain(spin, table, nodes)
    zero_us = _time_us(graph.replay)
    table[SITE] = delay_us * 1000
    torch.cuda.synchronize()
    got_us = _time_us(graph.replay)
    per_node = (got_us - zero_us) / nodes
    logger.info(
        "spin_delay in graph: request=%d us per-node added=%.2f us "
        "(chain %d nodes: delay0 %.2f us, delayed %.2f us)",
        delay_us, per_node, nodes, zero_us, got_us,
    )
    assert per_node >= delay_us - 0.5, f"per-node delay {per_node:.2f} us < {delay_us} us"
    assert per_node <= delay_us + SLACK_US, f"per-node delay {per_node:.2f} us"


def test_zero_delay_node_cost_is_bounded():
    spin = _spin()
    table = _table(0)
    nodes = 100
    chain = _graph_chain(spin, table, nodes)
    single = _graph_chain(spin, table, 1)
    chain_us = _time_us(chain.replay)
    single_us = _time_us(single.replay)
    per_node = (chain_us - single_us) / (nodes - 1)
    logger.info(
        "spin_delay delay-0 cost in graph: %.3f us per node "
        "(100-node chain %.2f us, 1-node %.2f us)",
        per_node, chain_us, single_us,
    )
    # The plan budgets ~1-3 us per main-chain node; 5 us is the failure line.
    assert per_node < 5.0, f"delay-0 spin node costs {per_node:.2f} us"


def test_table_update_between_replays_of_one_graph():
    """Alternate levels across replays of ONE graph, copies never synced.

    Each replay is preceded by a non_blocking copy of a pinned row into the
    table, on the same stream, exactly as ``CriticalityProbe.before_replay``
    issues it. The host never waits between a copy and its replay; only the
    timing events are read at the end.
    """
    spin = _spin()
    nodes = 4
    table = _table(0)
    graph = _graph_chain(spin, table, nodes)
    rows_us = [0, 30, 0, 100, 10, 0]
    host = torch.zeros(len(rows_us), TABLE_SIZE, dtype=torch.int64)
    for i, us in enumerate(rows_us):
        host[i, SITE] = us * 1000
    host = host.pin_memory()

    rounds = 20
    events = []
    for _ in range(rounds):
        for i in range(len(rows_us)):
            table.copy_(host[i], non_blocking=True)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            events.append((i, start, end))
    torch.cuda.synchronize()
    per_row = {i: [] for i in range(len(rows_us))}
    for i, start, end in events:
        per_row[i].append(start.elapsed_time(end) * 1000.0)
    medians = {i: statistics.median(v) for i, v in per_row.items()}
    zero = statistics.median(
        [medians[i] for i, us in enumerate(rows_us) if us == 0]
    )
    for i, us in enumerate(rows_us):
        added = (medians[i] - zero) / nodes
        logger.info(
            "spin_delay table update: row=%d request=%d us per-node added=%.2f us",
            i, us, added,
        )
        if us == 0:
            assert abs(added) < 1.0, f"row {i}: delay-0 replay moved by {added:.2f} us/node"
        else:
            assert us - 0.5 <= added <= us + SLACK_US, (
                f"row {i}: requested {us} us, replay added {added:.2f} us per node; "
                "the copy before the replay did not take effect in stream order"
            )


def test_argument_validation_fails_loud():
    spin = _spin()
    with pytest.raises(RuntimeError, match="int64"):
        spin(torch.zeros(TABLE_SIZE, dtype=torch.int32, device="cuda"), SITE)
    with pytest.raises(RuntimeError, match="outside"):
        spin(_table(0), TABLE_SIZE)
    with pytest.raises(RuntimeError, match="CUDA"):
        spin(torch.zeros(TABLE_SIZE, dtype=torch.int64), SITE)
