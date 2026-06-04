"""Source-level regression tests for Phase 4 of the Blackwell-support effort.

The three MLA strategy managers that select an attention backend on
``gpu_arch`` (glm5, deepseekv3, kimi_k25) must route Blackwell (sm_100)
through the same FlashAttention-3 / FlashMLA path as Hopper. We assert this
at the source level rather than by instantiating the managers, because
constructing a ``Parallel_Strategy_Manager`` requires a full model load
(weights, NCCL, native engine) that is far too heavy for a unit test — the
same reason ``test_detect_gpu_arch_blackwell`` loads its target module in
isolation.

The check guards against a regression where the Blackwell case is dropped
and the branch reverts to a bare ``gpu_arch == "hopper"`` selection (which
would raise ``Unsupported GPU arch`` on a B200).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Strategy managers whose ``_config_attn_module`` selects the MLA attention
# backend based on ``gpu_arch``.
_ARCH_BRANCHING_MANAGERS = [
    "batchgen/models/glm/glm5/Parallel_Strategy_Manager.py",
    "batchgen/models/deepseek/deepseekv3/Parallel_Strategy_Manager.py",
    "batchgen/models/moonshotai/kimi_k25/Parallel_Strategy_Manager.py",
]

# A bare ``gpu_arch == "hopper"`` *selection* branch (not a membership test)
# is exactly the regression we want to prevent.
_BARE_HOPPER_RE = re.compile(r'gpu_arch\s*==\s*["\']hopper["\']')
_HOPPER_BLACKWELL_RE = re.compile(
    r'gpu_arch\s+in\s*\(\s*["\']hopper["\']\s*,\s*["\']blackwell["\']\s*\)'
)


@pytest.mark.parametrize("rel_path", _ARCH_BRANCHING_MANAGERS)
def test_blackwell_routed_with_hopper(rel_path: str) -> None:
    source = (_REPO_ROOT / rel_path).read_text()

    assert _HOPPER_BLACKWELL_RE.search(source), (
        f"{rel_path}: expected a `gpu_arch in (\"hopper\", \"blackwell\")` "
        "branch so Blackwell uses the same FA3/FlashMLA backend as Hopper."
    )
    assert not _BARE_HOPPER_RE.search(source), (
        f"{rel_path}: found a bare `gpu_arch == \"hopper\"` branch; Blackwell "
        "would fall through to the unsupported-arch error. Use "
        '`gpu_arch in ("hopper", "blackwell")`.'
    )


def test_no_bare_hopper_branch_in_any_model() -> None:
    """No model should select a backend on a bare ``gpu_arch == "hopper"``."""
    offenders = []
    for path in (_REPO_ROOT / "batchgen" / "models").rglob("*.py"):
        if _BARE_HOPPER_RE.search(path.read_text()):
            offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert not offenders, (
        "Bare `gpu_arch == \"hopper\"` branches remain (Blackwell would hit "
        f"the unsupported-arch path): {offenders}"
    )
