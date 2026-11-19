"""The shipped package trees carry no ``test_*.py`` modules.

``PR_MERGE_POLICY.md`` §1.2 keeps tests in the repository's ``tests/`` tree,
and the PR hygiene check blocks a NEW ``test_*.py`` under a production tree.
That check scans only the files a PR adds, so test modules that pre-date it
stayed inside the runtime packages and went into the wheel. They now live
under ``tests/`` (or were removed). This pin covers what the per-PR check
cannot: any ``test_*.py`` present under a production tree, whatever its age.
The trees are the hygiene check's ``PROD_RE`` set; ``core``, ``external`` and
``op_builder`` are copied into ``batchgen/`` at build time, so ``core`` and
``op_builder`` are walked as sources. The check walks the filesystem and does
not import batchgen.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PRODUCTION_TREES = ("batchgen", "batchgen_kernels", "core", "server", "op_builder")

# The relocated suites: their coverage moved, it was not dropped.
RELOCATED_SUITES = (
    "tests/unit/test_gqa_sink_correction.py",
    "tests/unit/test_mxfp4_dequant_reference.py",
    "tests/unit/test_deepseekv4_encoding.py",
    "tests/gpu/test_moe_routing_kernels.py",
    "tests/gpu/test_marlin_int4_gs32.py",
)


def test_no_test_modules_under_production_trees():
    present = sorted(
        str(path.relative_to(ROOT))
        for tree in PRODUCTION_TREES
        if (ROOT / tree).is_dir()
        for path in (ROOT / tree).rglob("test_*.py")
    )
    assert not present, present


def test_production_trees_exist():
    # A renamed tree would make the walk above pass vacuously.
    missing = [tree for tree in PRODUCTION_TREES if not (ROOT / tree).is_dir()]
    assert not missing, missing


def test_relocated_suites_live_under_tests():
    missing = [rel for rel in RELOCATED_SUITES if not (ROOT / rel).is_file()]
    assert not missing, missing
