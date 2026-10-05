"""Resolution rules for the GLM-5 3D-MoE padded row capacity (mtp)."""

import pytest

from batchgen.models.glm.glm5.model import _GLM5_3D_MTP_ENV, resolve_glm5_3d_mtp


def test_unset_env_resolves_to_block_rounded_requirement(monkeypatch):
    monkeypatch.delenv(_GLM5_3D_MTP_ENV, raising=False)
    # 8 ranks x 128 max bucket -> exactly 1024 (already block-aligned)
    assert resolve_glm5_3d_mtp(8 * 128) == 1024
    # non-aligned requirement rounds up to the 128 block
    assert resolve_glm5_3d_mtp(1025) == 1152
    # degenerate inputs never resolve below one block
    assert resolve_glm5_3d_mtp(0) == 128
    assert resolve_glm5_3d_mtp(1) == 128


def test_env_acts_as_floor_not_cap(monkeypatch):
    monkeypatch.setenv(_GLM5_3D_MTP_ENV, "4096")
    # legacy default reproducible: env above requirement wins
    assert resolve_glm5_3d_mtp(1024) == 4096
    # an undersized env can never shrink below the safety requirement
    monkeypatch.setenv(_GLM5_3D_MTP_ENV, "256")
    assert resolve_glm5_3d_mtp(1024) == 1024


def test_env_invalid_value_raises(monkeypatch):
    monkeypatch.setenv(_GLM5_3D_MTP_ENV, "not-a-number")
    with pytest.raises(ValueError):
        resolve_glm5_3d_mtp(1024)
