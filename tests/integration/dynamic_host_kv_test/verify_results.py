"""Verify dynamic host KV test results.

Parses server logs for errors/desyncs, compares greedy outputs between
baseline and dynamic runs, and extracts host KV utilization metrics.

Usage:
    # Compare dynamic vs baseline
    python verify_results.py \
        --baseline results/baseline_scenarioA.jsonl \
        --dynamic results/dynamic_scenarioA.jsonl \
        --server-log logs/server_dynamic_growth.log

    # Single-run verification (no comparison)
    python verify_results.py \
        --dynamic results/eviction_scenarioE.jsonl \
        --server-log logs/server_eviction.log
"""

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ============ Result Parsing ============

def load_results(path: str) -> Dict[str, str]:
    """Load JSONL results into {custom_id: content} map."""
    results = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            cid = item.get("custom_id", "")
            response = item.get("response", {})
            body = response.get("body", {})
            choices = body.get("choices", [])
            content = ""
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")
            results[cid] = content
    return results


def load_input_ids(path: str) -> List[str]:
    """Load input JSONL custom_ids."""
    ids = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                ids.append(item.get("custom_id", ""))
    return ids


# ============ Log Parsing ============

def parse_log_errors(log_path: str) -> Dict[str, List[str]]:
    """Parse server log for errors, CUDA errors, desyncs, page errors."""
    categories = {
        "errors": [],
        "cuda_errors": [],
        "desyncs": [],
        "page_errors": [],
        "tracebacks": [],
    }

    traceback_mode = False
    traceback_lines = []

    with open(log_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            # Traceback detection
            if "Traceback" in line:
                traceback_mode = True
                traceback_lines = [f"L{line_num}: {line.rstrip()}"]
                continue
            if traceback_mode:
                traceback_lines.append(f"L{line_num}: {line.rstrip()}")
                if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                    categories["tracebacks"].append("\n".join(traceback_lines))
                    traceback_mode = False
                    traceback_lines = []
                continue

            # ERROR/CRITICAL level
            if " ERROR " in line or " CRITICAL " in line:
                categories["errors"].append(f"L{line_num}: {line.rstrip()}")

            # CUDA errors
            if "IllegalMemoryAccess" in line or "CUDA error" in line or "cuda error" in line:
                categories["cuda_errors"].append(f"L{line_num}: {line.rstrip()}")

            # Desync
            if "DESYNC" in line or "desync" in line:
                categories["desyncs"].append(f"L{line_num}: {line.rstrip()}")

            # Page errors
            if "Insufficient free pages" in line:
                categories["page_errors"].append(f"L{line_num}: {line.rstrip()}")

    return categories


def parse_host_kv_summaries(log_path: str) -> List[Dict]:
    """Parse [HOST_KV_SUMMARY] lines from log."""
    summaries = []
    pattern = re.compile(
        r"\[HOST_KV_SUMMARY\]\[Iter (\d+)\] "
        r"host_pages: total=(\d+) free=(\d+) used=(\d+) \(([0-9.]+)%\) "
        r"chunk_size=(\d+)"
    )
    with open(log_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                summaries.append({
                    "iter": int(m.group(1)),
                    "total": int(m.group(2)),
                    "free": int(m.group(3)),
                    "used": int(m.group(4)),
                    "pct": float(m.group(5)),
                    "chunk_size": int(m.group(6)),
                })
    return summaries


def parse_growth_events(log_path: str) -> Tuple[int, Dict[str, int]]:
    """Parse [HOST_KV_GROWTH] and [HOST_KV_GROWTH_DETAIL] lines."""
    total_events = 0
    per_seq_count = defaultdict(int)

    with open(log_path, "r") as f:
        for line in f:
            if "[HOST_KV_GROWTH_DETAIL]" in line:
                m = re.search(r"seq=(\w+)", line)
                if m:
                    per_seq_count[m.group(1)] += 1
                    total_events += 1
            elif "[HOST_KV_GROWTH] Grew" in line:
                m = re.search(r"Grew (\d+) sequences", line)
                if m:
                    # Aggregate growth event (when detail not available)
                    if total_events == 0:
                        total_events += int(m.group(1))

    return total_events, dict(per_seq_count)


def parse_eviction_events(log_path: str) -> Tuple[int, int, Dict[str, int]]:
    """Parse [HOST_KV_EVICT] and [HOST_KV_EVICT_DETAIL] lines."""
    total_evicted = 0
    total_freed_pages = 0
    per_seq_count = defaultdict(int)

    with open(log_path, "r") as f:
        for line in f:
            if "[HOST_KV_EVICT_DETAIL]" in line:
                m = re.search(r"seq=(\w+)", line)
                if m:
                    per_seq_count[m.group(1)] += 1
            if "[HOST_KV_EVICT] Evicted" in line:
                m = re.search(r"Evicted (\d+) sequences.*freed ~(\d+) host pages", line)
                if m:
                    total_evicted += int(m.group(1))
                    total_freed_pages += int(m.group(2))

    return total_evicted, total_freed_pages, dict(per_seq_count)


def parse_adaptive_chunk(log_path: str) -> List[Dict]:
    """Parse [ADAPTIVE_CHUNK] lines."""
    transitions = []
    pattern = re.compile(r"\[ADAPTIVE_CHUNK\] (\d+) -> (\d+) ema=([0-9.]+) completed=(\d+)")
    with open(log_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                transitions.append({
                    "old": int(m.group(1)),
                    "new": int(m.group(2)),
                    "ema": float(m.group(3)),
                    "completed": int(m.group(4)),
                })
    return transitions


def parse_completions(log_path: str) -> List[Dict]:
    """Parse [COMPLETION] lines."""
    completions = []
    pattern = re.compile(
        r"\[COMPLETION\] seq=(\w+) decoded=(\d+) prompt=(\d+) "
        r"was_evicted=(\w+) host_pages=(\d+)"
    )
    with open(log_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                completions.append({
                    "seq": m.group(1),
                    "decoded": int(m.group(2)),
                    "prompt": int(m.group(3)),
                    "was_evicted": m.group(4) == "True",
                    "host_pages": int(m.group(5)),
                })
    return completions


def parse_boundary_timing(log_path: str) -> List[float]:
    """Parse boundary total_ms from timing logs."""
    timings = []
    pattern = re.compile(r"total=([0-9.]+)ms")
    with open(log_path, "r") as f:
        for line in f:
            if "wait_kv=" in line and "gather=" in line:  # Boundary timing line
                m = pattern.search(line)
                if m:
                    timings.append(float(m.group(1)))
    return timings


# ============ Verification ============

def verify_completeness(input_path: str, results: Dict[str, str]) -> Tuple[int, int, List[str]]:
    """Verify all input IDs have results."""
    input_ids = load_input_ids(input_path)
    missing = [cid for cid in input_ids if cid not in results]
    return len(input_ids), len(results), missing


def verify_greedy_match(baseline: Dict[str, str], dynamic: Dict[str, str]) -> Tuple[int, int, List[Dict]]:
    """Compare greedy outputs between baseline and dynamic."""
    common_ids = set(baseline.keys()) & set(dynamic.keys())
    matches = 0
    mismatches = []

    for cid in sorted(common_ids):
        b_content = baseline[cid]
        d_content = dynamic[cid]
        if b_content == d_content:
            matches += 1
        else:
            # Find first difference
            diff_pos = 0
            for i, (bc, dc) in enumerate(zip(b_content, d_content)):
                if bc != dc:
                    diff_pos = i
                    break
            else:
                diff_pos = min(len(b_content), len(d_content))

            mismatches.append({
                "custom_id": cid,
                "diff_pos": diff_pos,
                "baseline_len": len(b_content),
                "dynamic_len": len(d_content),
                "baseline_snippet": b_content[max(0, diff_pos - 20):diff_pos + 20],
                "dynamic_snippet": d_content[max(0, diff_pos - 20):diff_pos + 20],
            })

    return matches, len(common_ids), mismatches


# ============ Report ============

def generate_report(
    input_path: Optional[str],
    dynamic_results: Dict[str, str],
    baseline_results: Optional[Dict[str, str]],
    log_errors: Dict[str, List[str]],
    summaries: List[Dict],
    growth_total: int,
    growth_per_seq: Dict[str, int],
    eviction_total: int,
    eviction_freed: int,
    eviction_per_seq: Dict[str, int],
    adaptive_transitions: List[Dict],
    completions: List[Dict],
    boundary_timings: List[float],
    completeness: Tuple[int, int, List[str]],
    greedy_match: Optional[Tuple[int, int, List[Dict]]],
):
    """Print the full verification report."""
    n_input, n_results, missing = completeness

    print(f"\n{'='*60}")
    print(f"Dynamic Host KV Test Report")
    print(f"{'='*60}")

    # Completeness
    pct = n_results / n_input * 100 if n_input > 0 else 0
    status = "PASS" if n_results == n_input else "FAIL"
    print(f"\nCompleteness:  {n_results}/{n_input} ({pct:.1f}%) [{status}]")
    if missing:
        print(f"  Missing: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    # Greedy match
    if greedy_match:
        matches, total, mismatches = greedy_match
        pct = matches / total * 100 if total > 0 else 0
        status = "PASS" if matches == total else "FAIL"
        print(f"Greedy Match:  {matches}/{total} ({pct:.1f}%) [{status}]")
        if mismatches:
            print(f"  Mismatches:")
            for mm in mismatches[:5]:
                print(f"    {mm['custom_id']}: diff at char {mm['diff_pos']} "
                      f"(baseline={mm['baseline_len']}, dynamic={mm['dynamic_len']})")
            if len(mismatches) > 5:
                print(f"    ... and {len(mismatches) - 5} more")

    # Errors
    n_errors = len(log_errors.get("errors", []))
    n_cuda = len(log_errors.get("cuda_errors", []))
    n_desyncs = len(log_errors.get("desyncs", []))
    n_page = len(log_errors.get("page_errors", []))
    n_tb = len(log_errors.get("tracebacks", []))

    print(f"\nErrors:        {n_errors} {'[PASS]' if n_errors == 0 else '[FAIL]'}")
    print(f"CUDA Errors:   {n_cuda} {'[PASS]' if n_cuda == 0 else '[FAIL]'}")
    print(f"Desyncs:       {n_desyncs} {'[PASS]' if n_desyncs == 0 else '[FAIL]'}")
    print(f"Page Errors:   {n_page} {'[PASS]' if n_page == 0 else '[FAIL]'}")
    print(f"Tracebacks:    {n_tb} {'[PASS]' if n_tb == 0 else '[FAIL]'}")

    if n_errors > 0:
        print(f"\n  First 5 errors:")
        for err in log_errors["errors"][:5]:
            print(f"    {err[:120]}")
    if n_cuda > 0:
        print(f"\n  CUDA errors:")
        for err in log_errors["cuda_errors"][:3]:
            print(f"    {err[:120]}")
    if n_desyncs > 0:
        print(f"\n  Desyncs:")
        for err in log_errors["desyncs"][:3]:
            print(f"    {err[:120]}")

    # Host KV utilization
    if summaries:
        peak_pct = max(s["pct"] for s in summaries)
        avg_pct = sum(s["pct"] for s in summaries) / len(summaries)
        chunk_vals = [s["chunk_size"] for s in summaries]
        print(f"\nHost KV Utilization:")
        print(f"  Peak: {peak_pct:.1f}%  Avg: {avg_pct:.1f}%")
        print(f"  Chunk size range: {min(chunk_vals)} - {max(chunk_vals)}")

    # Growth
    print(f"\nGrowth Events: {growth_total} total", end="")
    if growth_per_seq:
        avg_growth = sum(growth_per_seq.values()) / len(growth_per_seq)
        print(f" (avg {avg_growth:.1f}/seq across {len(growth_per_seq)} seqs)")
    else:
        print()

    # Eviction
    print(f"Eviction Events: {eviction_total} total ({eviction_freed} pages freed)")
    if eviction_per_seq:
        print(f"  Evicted sequences: {len(eviction_per_seq)}")

    # Adaptive chunk
    if adaptive_transitions:
        first = adaptive_transitions[0]
        last = adaptive_transitions[-1]
        print(f"Adaptive Chunk: {first['old']} -> {last['new']} ({len(adaptive_transitions)} transitions)")
    else:
        print(f"Adaptive Chunk: no transitions")

    # Completion stats
    if completions:
        decoded_lens = [c["decoded"] for c in completions]
        n_evicted = sum(1 for c in completions if c["was_evicted"])
        print(f"\nCompletion Stats:")
        print(f"  Total: {len(completions)}")
        print(f"  Mean decode: {sum(decoded_lens)/len(decoded_lens):,.0f} tokens")
        print(f"  Min/Max decode: {min(decoded_lens):,}/{max(decoded_lens):,}")
        print(f"  Evicted: {n_evicted}/{len(completions)} ({n_evicted/len(completions)*100:.1f}%)")

    # Boundary timing
    if boundary_timings:
        sorted_t = sorted(boundary_timings)
        p50 = sorted_t[len(sorted_t) // 2]
        p99 = sorted_t[int(len(sorted_t) * 0.99)]
        print(f"\nBoundary Timing:")
        print(f"  Mean: {sum(boundary_timings)/len(boundary_timings):.1f}ms  "
              f"P50: {p50:.1f}ms  P99: {p99:.1f}ms  Max: {max(boundary_timings):.1f}ms")

    # Overall verdict
    all_pass = (
        n_results == n_input
        and n_cuda == 0
        and n_desyncs == 0
        and n_page == 0
        and n_tb == 0
        and (greedy_match is None or greedy_match[0] == greedy_match[1])
    )
    print(f"\n{'='*60}")
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print(f"{'='*60}")

    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Verify dynamic host KV test results")
    parser.add_argument("--dynamic", type=str, required=True, help="Dynamic run results JSONL")
    parser.add_argument("--baseline", type=str, default=None, help="Baseline run results JSONL (for comparison)")
    parser.add_argument("--input", type=str, default=None, help="Input JSONL (for completeness check)")
    parser.add_argument("--server-log", type=str, default=None, help="Server log file to parse")
    args = parser.parse_args()

    # Load results
    logger.info(f"Loading dynamic results: {args.dynamic}")
    dynamic_results = load_results(args.dynamic)
    logger.info(f"  {len(dynamic_results)} results loaded")

    baseline_results = None
    if args.baseline:
        logger.info(f"Loading baseline results: {args.baseline}")
        baseline_results = load_results(args.baseline)
        logger.info(f"  {len(baseline_results)} results loaded")

    # Completeness check
    if args.input:
        completeness = verify_completeness(args.input, dynamic_results)
    else:
        completeness = (len(dynamic_results), len(dynamic_results), [])

    # Greedy match
    greedy_match = None
    if baseline_results:
        greedy_match = verify_greedy_match(baseline_results, dynamic_results)

    # Log parsing
    log_errors = {"errors": [], "cuda_errors": [], "desyncs": [], "page_errors": [], "tracebacks": []}
    summaries = []
    growth_total, growth_per_seq = 0, {}
    eviction_total, eviction_freed, eviction_per_seq = 0, 0, {}
    adaptive_transitions = []
    completions = []
    boundary_timings = []

    if args.server_log and Path(args.server_log).exists():
        logger.info(f"Parsing server log: {args.server_log}")
        log_errors = parse_log_errors(args.server_log)
        summaries = parse_host_kv_summaries(args.server_log)
        growth_total, growth_per_seq = parse_growth_events(args.server_log)
        eviction_total, eviction_freed, eviction_per_seq = parse_eviction_events(args.server_log)
        adaptive_transitions = parse_adaptive_chunk(args.server_log)
        completions = parse_completions(args.server_log)
        boundary_timings = parse_boundary_timing(args.server_log)

    # Generate report
    all_pass = generate_report(
        input_path=args.input,
        dynamic_results=dynamic_results,
        baseline_results=baseline_results,
        log_errors=log_errors,
        summaries=summaries,
        growth_total=growth_total,
        growth_per_seq=growth_per_seq,
        eviction_total=eviction_total,
        eviction_freed=eviction_freed,
        eviction_per_seq=eviction_per_seq,
        adaptive_transitions=adaptive_transitions,
        completions=completions,
        boundary_timings=boundary_timings,
        completeness=completeness,
        greedy_match=greedy_match,
    )

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
