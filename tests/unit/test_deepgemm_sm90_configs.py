# ---------------------------------------------------------------------------- #
#  BatchGen                                                                     #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""DeepGEMM's SM90 FP8 config selection, ported, on CPU.

Under test: ``batchgen_kernels/deepgemm_sm90_configs.py``.

  * the port reproduces every config of ``tests/data/deepgemm_sm90_golden.json``
    (the boundaries DeepGEMM printed for the GLM-5.2 FP8 GEMM shapes);
  * a Normal config depends on m only through ``ceil(m / 16)``;
  * the representative row counts reach every kernel key of their range,
    checked by brute force against ``best_config``, and at the GLM-5.2 shapes
    the key set does not grow past the prefill micro-batch cap;
  * the masked representatives reach every key of their expected-m set;
  * the module imports without torch.

Run: python -m pytest tests/unit/test_deepgemm_sm90_configs.py -q
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_MODULE = _ROOT / "batchgen_kernels" / "deepgemm_sm90_configs.py"
_GOLDEN = _ROOT / "tests" / "data" / "deepgemm_sm90_golden.json"


def _load(name="_deepgemm_sm90_configs_under_test"):
    spec = importlib.util.spec_from_file_location(name, _MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DG = _load()

# GLM-5.2's Normal (N, K) sites (``deepgemm_prewarm.dense_sites``).
GLM52_NORMAL = (
    (2048, 6144), (576, 6144), (16384, 2048), (20480, 2048), (28672, 512),
    (6144, 16384), (128, 6144), (4096, 2048), (12288, 6144), (6144, 12288),
    (4096, 6144), (6144, 2048),
)
PREFILL_CAP = 131_072  # ModuleBatchingConfig.prefill_micro_batch_token_cap


def _config_tuple(config):
    return (config.block_m, config.block_n, config.cluster_m,
            config.cluster_n, config.num_stages, config.swizzle_cd,
            config.math_threads)


# --------------------------------------------------------------------------- #
#  Golden data
# --------------------------------------------------------------------------- #
def test_port_reproduces_every_golden_config():
    data = json.loads(_GOLDEN.read_text())
    columns = data["columns"]
    assert columns == [
        "gemm_type", "m", "n", "k", "groups", "expected_m", "block_m",
        "block_n", "cluster_m", "cluster_n", "num_stages", "swizzle_cd",
        "math_threads",
    ]
    rows = data["rows"]
    assert len(rows) == 1543
    assert {row[0] for row in rows} == {DG.GEMM_NORMAL, DG.GEMM_MASKED}
    mismatches = []
    for row in rows:
        gemm_type, m, n, k, groups, expected_m = row[:6]
        config = DG.best_config(
            m, n, k, groups=groups, masked=gemm_type == DG.GEMM_MASKED,
            expected_m=expected_m or None,
        )
        if _config_tuple(config) != tuple(row[6:]):
            mismatches.append((row, _config_tuple(config)))
    assert mismatches == []


# --------------------------------------------------------------------------- #
#  Normal kernel
# --------------------------------------------------------------------------- #
def test_normal_config_is_a_function_of_ceil_m_over_16():
    for n, k in ((128, 6144), (2624, 6144)):
        for q in range(1, 400):
            configs = {DG.best_config(m, n, k)
                       for m in range(16 * q - 15, 16 * q + 1)}
            assert len(configs) == 1, (n, k, q)


@pytest.mark.parametrize("n,k", [(128, 6144), (6144, 16384)])
def test_representatives_reach_every_key_of_the_range(n, k):
    """``best_config`` is the scalar statement of the cost model and the
    representatives come from the vectorised one: the two must agree."""
    m_max = 8192
    reps = DG.normal_representative_ms(n, k, m_max)
    first = {}
    for m in range(1, m_max + 1):
        first.setdefault(DG.best_config(m, n, k).kernel_key(), m)
    # One per key, each the smallest m that selects it.
    assert reps == sorted(first.values())
    vectorised = DG.normal_configs(n, k, m_max)
    assert all(vectorised[-(-m // 16)] == DG.best_config(m, n, k)
               for m in range(1, m_max + 1, 7))


def test_representatives_for_a_range_inside_one_q_class():
    assert DG.normal_representative_ms(128, 6144, 1) == [1]
    assert DG.normal_representative_ms(128, 6144, 16) == [1]
    reps = DG.normal_representative_ms(128, 6144, 17)
    assert reps == [1, 17]
    with pytest.raises(ValueError):
        DG.normal_representative_ms(128, 6144, 0)


def test_glm52_key_set_stops_growing_below_the_prefill_cap():
    """A prompt longer than the cap is its own micro-batch, so prefill rows
    can exceed it; at these shapes that selects no further key."""
    for n, k in GLM52_NORMAL:
        capped = DG.normal_representative_ms(n, k, PREFILL_CAP)
        assert DG.normal_representative_ms(n, k, 8 * PREFILL_CAP) == capped
        assert max(capped) < PREFILL_CAP // 2


def test_sm_count_changes_the_selection():
    """The kernel and its choice depend on the SM count: the prewarm must
    model the device it runs on."""
    full = DG.normal_representative_ms(2624, 6144, PREFILL_CAP)
    fewer = DG.normal_representative_ms(2624, 6144, PREFILL_CAP, num_sms=78)
    assert full != fewer
    with pytest.raises(ValueError, match="even SM count"):
        DG.best_config(1, 128, 6144, num_sms=77)


# --------------------------------------------------------------------------- #
#  Masked kernel
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n,k,groups", [(2624, 6144, 1), (6144, 16384, 1),
                                        (4096, 6144, 32), (6144, 2048, 32)])
def test_masked_representatives_reach_every_key(n, k, groups):
    expected = range(1, 2049)
    reps = DG.masked_representative_expected_ms(n, k, groups, expected)
    reached = {DG.best_config(64, n, k, groups=groups, masked=True,
                              expected_m=e).kernel_key() for e in expected}
    witnessed = {DG.best_config(64, n, k, groups=groups, masked=True,
                                expected_m=e).kernel_key() for e in reps}
    assert witnessed == reached and len(reps) == len(reached)


def test_masked_config_depends_on_ceil_expected_m_over_64():
    for e in range(1, 1025):
        lo = DG.best_config(64, 2624, 6144, masked=True,
                            expected_m=(e - 1) // 64 * 64 + 1)
        assert DG.best_config(64, 2624, 6144, masked=True,
                              expected_m=e) == lo


def test_masked_candidates_respect_multicast_legality():
    for e in (1, 65, 1000, 2048):
        config = DG.best_config(64, 2624, 6144, masked=True, expected_m=e)
        cluster = config.cluster_m * config.cluster_n
        assert -(-2624 // config.block_n) % cluster == 0
        assert config.block_m in (64, 128)


def test_masked_representatives_of_an_empty_set():
    assert DG.masked_representative_expected_ms(2624, 6144, 1, []) == []
    with pytest.raises(ValueError):
        DG.masked_representative_expected_ms(2624, 6144, 1, [0, 1])


# --------------------------------------------------------------------------- #
#  Version and import hygiene
# --------------------------------------------------------------------------- #
def test_installed_version_reads_the_distribution(monkeypatch):
    from importlib import metadata

    found = {"deep-gemm": "9.9.9"}

    def version(name):
        if name in found:
            return found[name]
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", version)
    assert DG.installed_version() == "9.9.9"
    found["sgl-deep-gemm"] = DG.SUPPORTED_DEEPGEMM_VERSION
    assert DG.installed_version() == DG.SUPPORTED_DEEPGEMM_VERSION
    found.clear()
    assert DG.installed_version() is None


def test_imports_without_torch(monkeypatch):
    for name in ("torch", "triton", "deep_gemm"):
        monkeypatch.setitem(sys.modules, name, None)
    module = _load("_deepgemm_sm90_configs_no_torch")
    assert module.best_config(1, 128, 6144).block_m == 16
