"""The BF16 BMM absorb fallback is retired: every GLM-5 decode consumer uses
the WP5 FP8 absorb weights, a failed init fails loud at absorb init (not
mid-decode), and the eager selector no longer dequantizes kv_b_proj per
step."""
import inspect
import types

import pytest
import torch


def test_selector_has_no_per_step_dequant_fallback():
    selector = pytest.importorskip("batchgen.attention.dsa.glm5_decode_selector")
    src = inspect.getsource(selector)
    assert "deepseek_v3_dequantization" not in src
    assert "_cached_q_absorb" not in src
    assert "w_kc" not in src


def test_initialize_decode_absorb_fails_loud_without_fp8_kernel(monkeypatch):
    fb = pytest.importorskip("batchgen.attention.mla.flashmla_backend")
    wrappers = pytest.importorskip("batchgen.models.glm.glm5.wrappers")

    num_heads, nope, kv_rank = 2, 3, 4
    monkeypatch.setattr(
        fb, "deepseek_v3_dequantization",
        lambda w, s: torch.zeros(
            num_heads, nope + kv_rank, kv_rank, dtype=torch.bfloat16
        ),
    )
    monkeypatch.setattr(wrappers, "_HAS_FP8_ABSORB", False)
    fake = types.SimpleNamespace(
        layer_idx=0,
        module=types.SimpleNamespace(
            kv_b_proj=types.SimpleNamespace(
                weight=types.SimpleNamespace(data=torch.zeros(1))
            ),
            num_heads=num_heads,
            qk_nope_head_dim=nope,
            kv_lora_rank=kv_rank,
        ),
        weight_dequant_scale={"kv_b_proj.weight_scale_inv": torch.ones(1)},
    )
    with pytest.raises(RuntimeError, match="FP8 absorb"):
        wrappers.GLM5AttnWrapper.initialize_decode_absorb(fake)
