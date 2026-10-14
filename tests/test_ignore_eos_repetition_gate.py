"""ignore_eos disables the per-request repetition breaker.

ignore_eos declares the content irrelevant and the length contract binding
(fixed-shape benchmark semantics), so the repetition breaker — which exists to
reclaim capacity from degenerate tokens the caller did not want — must not end
such a sequence. Requests without ignore_eos keep the breaker unchanged.
"""

import types

import pytest
import torch

pytest.importorskip("triton")  # batchgen_worker imports triton at module scope

from batchgen.batchgen_worker import (  # noqa: E402
    _check_repeating_pattern,
    _repetition_check_enabled,
)


def test_gate_disables_check_for_ignore_eos_sequences():
    assert _repetition_check_enabled(types.SimpleNamespace(ignore_eos=False))
    assert not _repetition_check_enabled(types.SimpleNamespace(ignore_eos=True))


def test_detector_itself_still_fires_on_a_repeating_tail():
    # 2-token pattern repeated 40 times at the tail (> the 32 required).
    tokens = torch.tensor([7] * 50 + [11, 13] * 40, dtype=torch.long)
    assert _check_repeating_pattern(tokens, tokens.numel())


def test_detector_stays_quiet_on_non_repeating_tail():
    torch.manual_seed(0)
    tokens = torch.randint(0, 50000, (256,), dtype=torch.long)
    assert not _check_repeating_pattern(tokens, tokens.numel())


def test_both_worker_sites_are_gated():
    """Source-level contract: every REP_DETECTION site consults the gate.

    The two detection sites live inline in large worker methods; this pins
    that each conditional carrying `_rep_detected` also carries the per-seq
    gate, so a future third site copied from an old revision fails here.
    """
    import inspect
    import batchgen.batchgen_worker as worker_mod

    src = inspect.getsource(worker_mod)
    srclines = src.splitlines()

    def _indent(line):
        return len(line) - len(line.lstrip())

    def _is_gated(idx):
        line = srclines[idx]
        # Gate on the same line or on the continuation line of a split
        # conditional.
        if "_repetition_check_enabled" in line:
            return True
        if idx + 1 < len(srclines) and "_repetition_check_enabled" in srclines[idx + 1]:
            return True
        # Otherwise an enclosing conditional at shallower indentation must
        # carry the gate (a nested check inside an already-gated block).
        ind = _indent(line)
        for j in range(idx - 1, max(idx - 60, -1), -1):
            prev = srclines[j]
            if not prev.strip():
                continue
            pind = _indent(prev)
            if pind < ind and prev.lstrip().startswith(("if ", "if(")):
                if "_repetition_check_enabled" in prev or (
                    j + 1 < len(srclines)
                    and _indent(srclines[j + 1]) > pind
                    and "_repetition_check_enabled" in srclines[j + 1]
                    and srclines[j].rstrip().endswith(("and", "(",))
                ):
                    return True
                ind = pind  # keep walking up the enclosing blocks
        return False

    sites = [
        idx for idx, line in enumerate(srclines)
        if "not seq._rep_detected" in line and line.lstrip().startswith("if ")
    ]
    assert sites, "expected repetition-check conditionals in batchgen_worker"
    ungated = [srclines[i].strip() for i in sites if not _is_gated(i)]
    assert not ungated, f"repetition-check sites missing the ignore_eos gate: {ungated}"
