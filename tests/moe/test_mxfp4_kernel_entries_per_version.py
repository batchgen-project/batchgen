"""Each MXFP4 wrapper requires only its own kernel version's entries.

The sm103a lane compiles the sm_90a-only v3 (wgmma) kernels out, so a build
with v1/v2 alone must serve the v1 paths and refuse v3 with a named error.
"""

from types import SimpleNamespace

import pytest

import batchgen.moe.marlin_grouped_moe as mgm


def _module_with(*versions):
    names = [n for v in versions for n in mgm._MXFP4_KERNEL_ENTRIES_BY_VERSION[v]]
    return SimpleNamespace(**{n: object() for n in names})


def test_v1_v2_build_serves_v1_and_v2(monkeypatch):
    monkeypatch.setattr(mgm, "_module", _module_with(1, 2))
    mgm._require_mxfp4_kernels(1)
    mgm._require_mxfp4_kernels(2)
    assert mgm.is_marlin_mxfp4_available(1)
    assert not mgm.is_marlin_mxfp4_available(3)


def test_v3_without_entries_names_the_lane(monkeypatch):
    monkeypatch.setattr(mgm, "_module", _module_with(1, 2))
    with pytest.raises(RuntimeError, match="sm90a build lane"):
        mgm._require_mxfp4_kernels(3)


def test_missing_v1_entries_still_hard_fail(monkeypatch):
    monkeypatch.setattr(mgm, "_module", _module_with(2, 3))
    with pytest.raises(RuntimeError, match="K3 refuses to run"):
        mgm._require_mxfp4_kernels(1)


def test_unknown_version_is_refused(monkeypatch):
    monkeypatch.setattr(mgm, "_module", _module_with(1, 2, 3))
    with pytest.raises(ValueError, match="must be one of"):
        mgm._require_mxfp4_kernels(4)


def test_every_wrapper_requires_its_own_version():
    import inspect

    source = inspect.getsource(mgm)
    expected = {
        "marlin_grouped_stage1_fused_mxfp4_situ": 1,
        "marlin_grouped_m16_mxfp4": 1,
        "marlin_grouped_stage1_fused_mxfp4_situ_v2": 2,
        "marlin_grouped_m16_mxfp4_v2": 2,
        "compute_mxfp4_v3_expert_order": 3,
        "marlin_grouped_stage1_fused_mxfp4_situ_v3": 3,
        "marlin_grouped_m16_mxfp4_v3": 3,
        "single_expert_marlin_mxfp4_decode": 1,
    }
    for name, version in expected.items():
        body = inspect.getsource(getattr(mgm, name))
        assert f"_require_mxfp4_kernels({version})" in body, name
    assert "_require_mxfp4_kernels()" not in source
