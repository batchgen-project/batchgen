"""Focused correctness gate for the AOT K3 KDA decode fusion.

The serving pool uses FLA's default recurrent-state layout ``[K, V]``.  The
test deliberately keeps the convolution weights to ``[0, 0, 0, 1]`` so the
reference can construct the post-convolution input without another extension;
the fused kernel still performs the real state shift and recurrence.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F


if os.environ.get("K3_GPU_STAGE") == "1" and not torch.cuda.is_available():
    raise RuntimeError(
        "K3_GPU_STAGE=1 but CUDA is unavailable — this staged run must not "
        "silently skip."
    )

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="staged for an H200 GPU"
)


@pytest.mark.parametrize("heads", [3, 6, 12])
def test_fused_kda_matches_fla_and_preserves_padding(heads: int):
    from batchgen_kernels.attention.kda_fused_decode import kda_fused_decode
    from fla.ops.kda.fused_recurrent import fused_recurrent_kda_fwd

    torch.manual_seed(20260829 + heads)
    device = torch.device("cuda")
    head_dim = 128
    segment = heads * head_dim
    batch = 3
    slots = torch.tensor([0, -1, 2], dtype=torch.int32, device=device)

    mixed_qkv = torch.randn(
        batch, 3 * segment, dtype=torch.bfloat16, device=device
    )
    forget_gate = torch.randn(batch, segment, dtype=torch.bfloat16, device=device)
    beta = torch.randn(batch, heads, dtype=torch.bfloat16, device=device)
    onorm_gate = torch.randn(batch, segment, dtype=torch.bfloat16, device=device)
    a_log = torch.randn(heads, dtype=torch.float32, device=device)
    dt_bias = torch.randn(segment, dtype=torch.float32, device=device)
    onorm_weight = torch.randn(head_dim, dtype=torch.float32, device=device)

    # The last tap is exactly one and the history starts at zero.  The fused
    # convolution therefore consumes silu(mixed_qkv) while still exercising
    # the in-place three-tap shift for valid slots.
    conv_weights = []
    conv_biases = []
    conv_states = []
    for _ in range(3):
        weight = torch.zeros(segment, 4, dtype=torch.float32, device=device)
        weight[:, 3] = 1.0
        conv_weights.append(weight)
        conv_biases.append(torch.zeros(segment, dtype=torch.float32, device=device))
        conv_states.append(
            torch.zeros(3, segment, 3, dtype=torch.bfloat16, device=device)
        )

    state = torch.randn(
        3, heads, head_dim, head_dim, dtype=torch.float32, device=device
    )
    state_before = state.clone()
    conv_before = [pool.clone() for pool in conv_states]
    fused = kda_fused_decode(
        mixed_qkv,
        forget_gate,
        beta,
        conv_states[0],
        conv_states[1],
        conv_states[2],
        conv_weights[0],
        conv_weights[1],
        conv_weights[2],
        conv_biases[0],
        conv_biases[1],
        conv_biases[2],
        a_log,
        dt_bias,
        onorm_gate,
        onorm_weight,
        state,
        slots,
        scale=head_dim ** -0.5,
        onorm_eps=1e-5,
        lower_bound=-5.0,
    )

    valid = torch.tensor([0, 2], dtype=torch.long, device=device)
    # The fused kernel evaluates the convolution in fp32 after reading the
    # BF16 input.  Feeding the same fp32 SiLU values to FLA isolates the
    # recurrent layout/recurrence rather than a separate conv implementation.
    qkv = [
        mixed_qkv[:, i * segment : (i + 1) * segment].float()
        for i in range(3)
    ]
    conv_output = [F.silu(x[valid]) for x in qkv]
    q, k, v = [x.view(2, 1, heads, head_dim) for x in conv_output]
    state_ref = state_before[valid].clone()
    recurrent, _ = fused_recurrent_kda_fwd(
        q=q,
        k=k,
        v=v,
        g=forget_gate[valid].float().view(2, 1, heads, head_dim),
        beta=beta[valid].float().view(2, 1, heads),
        A_log=a_log,
        dt_bias=dt_bias,
        initial_state=state_ref,
        output_final_state=True,
        inplace_final_state=True,
        state_v_first=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=-5.0,
    )
    recurrent = recurrent[:, 0]
    ref = recurrent * torch.rsqrt(
        recurrent.square().mean(dim=-1, keepdim=True) + 1e-5
    )
    ref = ref * onorm_weight.view(1, 1, head_dim)
    ref = ref * torch.sigmoid(onorm_gate[valid].float().view(2, heads, head_dim))
    ref = ref.reshape(2, segment).to(torch.bfloat16)

    actual = fused[valid]
    delta = (actual.float() - ref.float()).abs()
    rel_l2 = torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(
        ref.float()
    ).clamp_min(1e-30)
    cosine = F.cosine_similarity(
        actual.float().reshape(1, -1), ref.float().reshape(1, -1)
    ).item()
    assert delta.max().item() <= 3e-2
    assert rel_l2.item() <= 1e-2
    assert cosine >= 0.999

    # Native pool is [slots, heads, K, V].  A valid update must match FLA's
    # default state convention, and the padding row must remain untouched.
    state_delta = (state[valid] - state_ref).abs()
    assert state_delta.max().item() <= 2e-2
    assert (
        torch.linalg.vector_norm(state_delta)
        / torch.linalg.vector_norm(state_ref).clamp_min(1e-30)
    ).item() <= 2e-3
    assert torch.equal(state[1], state_before[1])
    assert torch.equal(fused[1], torch.zeros_like(fused[1]))

    for pool, before, raw in zip(conv_states, conv_before, qkv):
        expected = before.clone()
        expected[valid, :, 0] = before[valid, :, 1]
        expected[valid, :, 1] = before[valid, :, 2]
        expected[valid, :, 2] = raw[valid].to(torch.bfloat16)
        assert torch.equal(pool, expected)
        assert torch.equal(pool[1], before[1])
