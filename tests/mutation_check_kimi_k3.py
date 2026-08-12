#!/usr/bin/env python3
# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3                                                           #
#  copyright (c) EfficientMoE team 2025                                         #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""Mutation discipline runner for the Kimi-K3 M2 CPU suite.

For every mutation in ``tests/kimi_k3_harness.py::MUTATIONS`` this spawns one
pytest subprocess with ``BATCHGEN_K3_MUTATION=<name>`` restricted (by default)
to the tests the mutation MUST turn red, and asserts a NON-ZERO exit.  It
finishes with one clean full run (no mutation) asserting green.  A detector
that stays green on a known-broken variant is a retired detector — this
project retired two of those in one week; this runner is the guard.

Usage:
    python tests/mutation_check_kimi_k3.py               # targeted (fast)
    python tests/mutation_check_kimi_k3.py --full        # full suite per mutation
    python tests/mutation_check_kimi_k3.py --mutations a,b,c
    python tests/mutation_check_kimi_k3.py --list

Deliberately NOT pytest-collected (no test_ prefix): one runner process
orchestrating many pytest subprocesses.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
SUITE = TESTS_DIR / "test_kimi_k3_model.py"

sys.path.insert(0, str(TESTS_DIR))
import kimi_k3_harness as H  # noqa: E402


#: `N failed` in pytest's terse summary — the only evidence that a detector
#: actually FIRED, as opposed to pytest exiting nonzero for another reason.
_FAILED_RE = re.compile(r"(\d+) failed")


def run_one(name: str, full: bool) -> tuple[bool, float, str]:
    """RED iff at least one test ran AND at least one test failed.

    A bare `returncode != 0` is not evidence of detection: pytest exits 5 when
    `-k` collects nothing and 2 on a collection error, so a single renamed
    test would silently retire its detector while still reporting RED — the
    way two detectors were lost on this project before.
    """
    mut = H.MUTATIONS[name]
    cmd = [sys.executable, "-m", "pytest", str(SUITE), "-x", "-q",
           "--no-header", "-p", "no:cacheprovider"]
    if not full:
        cmd += ["-k", mut.must_red]
    env = dict(os.environ)
    env[H.MUTATION_ENV] = name
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          cwd=str(TESTS_DIR.parent))
    dt = time.time() - t0
    out = proc.stdout + proc.stderr
    tail = out.strip().splitlines()[-3:]
    if proc.returncode == 5:
        return False, dt, ("NO TESTS COLLECTED for -k {!r} — the detector no "
                           "longer matches any test name".format(mut.must_red))
    failed = _FAILED_RE.search(out)
    red = bool(failed) and int(failed.group(1)) >= 1
    if proc.returncode != 0 and not red:
        return False, dt, ("pytest exited {} without a reported failure "
                           "(collection error?): {}".format(
                               proc.returncode, " | ".join(tail)))
    return red, dt, " | ".join(tail)


def run_clean() -> tuple[bool, float, str]:
    cmd = [sys.executable, "-m", "pytest", str(SUITE), "-q",
           "--no-header", "-p", "no:cacheprovider"]
    env = dict(os.environ)
    env.pop(H.MUTATION_ENV, None)
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          cwd=str(TESTS_DIR.parent))
    dt = time.time() - t0
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
    return proc.returncode == 0, dt, " | ".join(tail)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="run the FULL suite per mutation (slow)")
    parser.add_argument("--mutations", type=str, default="",
                        help="comma-separated subset (default: all)")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--skip-clean", action="store_true",
                        help="skip the final clean-run green check")
    args = parser.parse_args()

    names = sorted(H.MUTATIONS)
    if args.list:
        for n in names:
            m = H.MUTATIONS[n]
            print("{:32s} must_red: {}".format(n, m.must_red))
        return 0
    if args.mutations:
        requested = [n.strip() for n in args.mutations.split(",") if n.strip()]
        unknown = sorted(set(requested) - set(names))
        if unknown:
            print("Unknown mutation(s): {}".format(unknown))
            return 2
        names = requested

    survivors = []
    for i, name in enumerate(names, 1):
        red, dt, tail = run_one(name, args.full)
        status = "RED (ok)" if red else "GREEN -- SURVIVOR!"
        print("[{:2d}/{:2d}] {:32s} {:18s} {:6.1f}s".format(
            i, len(names), name, status, dt))
        if not red:
            survivors.append(name)
            print("        tail: {}".format(tail))

    if not args.skip_clean:
        ok, dt, tail = run_clean()
        print("[clean] unmutated full suite: {} ({:.1f}s)".format(
            "GREEN (ok)" if ok else "RED -- suite broken!", dt))
        if not ok:
            print("        tail: {}".format(tail))
            return 1

    if survivors:
        print("\n{} SURVIVING mutation(s) — the suite cannot see these breaks:".format(
            len(survivors)))
        for name in survivors:
            print("  - {} (must_red: {})".format(name, H.MUTATIONS[name].must_red))
        return 1
    print("\nAll {} mutations red; clean run green.".format(len(names)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
