from __future__ import annotations

import torch

from batchgen.attention.mla.prefix_absorb import (
    absorb_mla_attention_output,
    build_absorbed_mla_query_states,
    build_full_hit_absorbed_mla_query_states,
    prefix_rotary_seq_len,
    project_absorbed_mla_output,
    project_absorbed_mla_output_w8a16,
)


def test_build_absorbed_mla_query_states_matches_manual_einsum():
    q_nope = torch.arange(12, dtype=torch.float32).view(2, 2, 3)
    q_pe = torch.arange(8, dtype=torch.float32).view(2, 2, 2)
    q_absorb = torch.arange(24, dtype=torch.float32).view(2, 3, 4)

    actual = build_absorbed_mla_query_states(
        q_nope=q_nope,
        q_pe=q_pe,
        q_absorb=q_absorb,
        dtype=torch.float32,
    )

    expected = torch.empty(1, 2, 2, 6)
    expected[0, :, :, :4] = torch.einsum("thd,hdc->thc", q_nope, q_absorb)
    expected[0, :, :, 4:] = q_pe
    assert torch.equal(actual, expected.contiguous())


def test_build_full_hit_absorbed_mla_query_states_uses_full_hit_layout():
    q_nope = torch.arange(12, dtype=torch.float32).view(2, 2, 3)
    q_pe = torch.arange(8, dtype=torch.float32).view(2, 2, 2)
    q_absorb = torch.arange(24, dtype=torch.float32).view(2, 3, 4)

    actual = build_full_hit_absorbed_mla_query_states(
        q_nope=q_nope,
        q_pe=q_pe,
        q_absorb=q_absorb,
        dtype=torch.float32,
    )

    suffix_layout = build_absorbed_mla_query_states(
        q_nope=q_nope,
        q_pe=q_pe,
        q_absorb=q_absorb,
        dtype=torch.float32,
    )
    expected = suffix_layout.view(2, 1, 2, 6).contiguous()
    assert torch.equal(actual, expected)


def test_project_absorbed_mla_output_uses_common_absorb_layout():
    attn_out = torch.arange(16, dtype=torch.float32).view(1, 2, 2, 4)
    out_absorb = torch.arange(24, dtype=torch.float32).view(2, 3, 4)
    projection = torch.nn.Linear(6, 5, bias=False)

    absorbed = absorb_mla_attention_output(
        attn_out=attn_out,
        out_absorb=out_absorb,
        v_head_dim=3,
    )
    actual = project_absorbed_mla_output(
        attn_out=attn_out,
        out_absorb=out_absorb,
        v_head_dim=3,
        output_projection=projection,
    )

    expected_absorbed = torch.einsum(
        "bqhc,hdc->bqhd",
        attn_out,
        out_absorb,
    ).reshape(2, 6)
    assert torch.equal(absorbed, expected_absorbed)
    assert torch.equal(actual, projection(absorbed))


def test_project_absorbed_mla_output_w8a16_delegates_to_gemm():
    attn_out = torch.arange(16, dtype=torch.float32).view(1, 2, 2, 4)
    out_absorb = torch.arange(24, dtype=torch.float32).view(2, 3, 4)
    weight = torch.randn(5, 6)
    scale = torch.ones(5)
    calls = {}
    expected_result = torch.randn(2, 5)

    def fake_gemm(w, s, x):
        calls["weight"] = w
        calls["scale"] = s
        calls["input"] = x
        return expected_result

    actual = project_absorbed_mla_output_w8a16(
        attn_out=attn_out,
        out_absorb=out_absorb,
        v_head_dim=3,
        o_proj_weight=weight,
        o_proj_scale=scale,
        gemm=fake_gemm,
    )

    assert actual is expected_result
    assert calls["weight"] is weight
    assert calls["scale"] is scale
    assert torch.equal(
        calls["input"],
        absorb_mla_attention_output(
            attn_out=attn_out,
            out_absorb=out_absorb,
            v_head_dim=3,
        ),
    )


def test_prefix_rotary_seq_len_covers_prefix_and_position_ids():
    position_ids = torch.tensor([3, 7, 8], dtype=torch.long)

    assert prefix_rotary_seq_len(5, position_ids) == 9
    assert prefix_rotary_seq_len(16, position_ids) == 16
