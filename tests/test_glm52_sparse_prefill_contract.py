import logging
import sys
import types
from contextlib import nullcontext

import pytest
import torch

import batchgen.models.glm.glm5.sparse_prefill as sparse_prefill
from batchgen.models.glm.glm5.sparse_prefill import (
    _compute_indexer_kv_from_quantized_hidden,
    _fused_rope_hadamard_q,
    _score_chunk_rows,
    build_packed_causal_ranges,
    offset_packed_topk_indices_,
    should_use_glm52_sparse_prefill,
    validate_carried_topk,
)


def test_build_packed_causal_ranges():
    cu = torch.tensor([0, 3, 5], dtype=torch.int32)
    positions = torch.tensor([0, 1, 2, 0, 1], dtype=torch.int64)

    starts, ends = build_packed_causal_ranges(cu, positions, total_tokens=5)

    assert starts.tolist() == [0, 0, 0, 3, 3]
    assert ends.tolist() == [1, 2, 3, 4, 5]


def test_build_packed_causal_ranges_rejects_bad_metadata():
    cu = torch.tensor([0, 3], dtype=torch.int64)
    positions = torch.tensor([0, 1, 2], dtype=torch.int64)
    with pytest.raises(TypeError, match="int32"):
        build_packed_causal_ranges(cu, positions, total_tokens=3)


def test_build_packed_causal_ranges_rejects_bad_offsets_and_positions():
    positions = torch.tensor([0, 1, 0], dtype=torch.int64)
    with pytest.raises(ValueError, match="complete packed token tensor"):
        build_packed_causal_ranges(
            torch.tensor([1, 3, 4], dtype=torch.int32),
            positions,
            total_tokens=3,
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        build_packed_causal_ranges(
            torch.tensor([0, 2, 2, 3], dtype=torch.int32),
            positions,
            total_tokens=3,
        )
    with pytest.raises(ValueError, match="restart at zero"):
        build_packed_causal_ranges(
            torch.tensor([0, 2, 3], dtype=torch.int32),
            torch.tensor([0, 1, 1], dtype=torch.int64),
            total_tokens=3,
        )


def test_sparse_prefill_route_is_glm52_long_context_only():
    assert should_use_glm52_sparse_prefill(
        "glm_moe_dsa_5_2",
        2049,
        2048,
    )
    assert not should_use_glm52_sparse_prefill(
        "glm_moe_dsa_5_2",
        2048,
        2048,
    )
    assert not should_use_glm52_sparse_prefill(
        "glm_moe_dsa",
        4096,
        2048,
    )


def test_score_chunk_rows_bounds_logits_to_two_gib():
    assert _score_chunk_rows(65_536) == 8_192
    assert _score_chunk_rows(131_072) == 4_096
    assert _score_chunk_rows(262_144) == 2_048
    assert _score_chunk_rows(262_145) == 2_044


def test_rope_hadamard_q_uses_corrected_kernel_and_repeats_positions(monkeypatch):
    calls = {}
    module = types.ModuleType("batchgen.other_kernels.hadamard_transform")

    def fused_rope_hadamard(x, cos, sin, positions, scale):
        calls.update(
            x=x,
            cos=cos,
            sin=sin,
            positions=positions,
            scale=scale,
        )
        return x

    module.fused_rope_hadamard = fused_rope_hadamard
    monkeypatch.setitem(
        sys.modules,
        "batchgen.other_kernels.hadamard_transform",
        module,
    )

    q = torch.arange(2 * 32 * 128, dtype=torch.float32).to(
        torch.bfloat16
    ).reshape(2, 32, 128)
    cos = torch.ones(16, 64, dtype=torch.bfloat16)
    sin = torch.zeros(16, 64, dtype=torch.bfloat16)
    positions = torch.tensor([3, 9], dtype=torch.int64)

    output = _fused_rope_hadamard_q(q, cos, sin, positions)

    assert output.shape == q.shape
    assert torch.equal(output, q)
    assert calls["x"].shape == (64, 128)
    assert calls["cos"].dtype == torch.float32
    assert calls["sin"].dtype == torch.float32
    assert calls["positions"].tolist() == [3] * 32 + [9] * 32
    assert calls["scale"] == 128**-0.5


def test_compute_indexer_kv_uses_corrected_kernel_contract(monkeypatch):
    calls = {}
    module = types.ModuleType("batchgen.other_kernels.hadamard_transform")

    def fused_rope_hadamard(x, cos, sin, positions, scale):
        calls.update(
            x=x,
            cos=cos,
            sin=sin,
            positions=positions,
            scale=scale,
        )
        return x

    module.fused_rope_hadamard = fused_rope_hadamard
    monkeypatch.setitem(
        sys.modules,
        "batchgen.other_kernels.hadamard_transform",
        module,
    )
    projected = torch.arange(3 * 128, dtype=torch.float32).reshape(3, 128)
    monkeypatch.setattr(
        sparse_prefill,
        "_fp8_linear_from_quantized",
        lambda *_args: projected,
    )
    records = []
    indexer = types.SimpleNamespace(
        wk=types.SimpleNamespace(weight=types.SimpleNamespace(data=torch.empty(128, 4))),
        wk_scale=torch.ones(1),
        k_norm=lambda value: value,
        rotary_emb=lambda _value, _max_seqlen: (
            torch.ones(16, 64, dtype=torch.bfloat16),
            torch.zeros(16, 64, dtype=torch.bfloat16),
        ),
        index_head_dim=128,
        layer_idx=6,
        record_prefill_rope_hadamard_path=lambda path, layer: records.append(
            (path, layer)
        ),
    )
    positions = torch.tensor([[0, 1, 2]], dtype=torch.int64)

    output = _compute_indexer_kv_from_quantized_hidden(
        indexer=indexer,
        hidden_fp8=torch.empty(3, 4),
        hidden_scale=torch.empty(3, 1),
        position_ids=positions,
        max_seqlen=16,
        timed=lambda _name: nullcontext(),
    )

    assert output.shape == (1, 3, 1, 128)
    assert calls["x"].shape == (3, 128)
    assert calls["x"].dtype == torch.bfloat16
    assert calls["cos"].dtype == torch.float32
    assert calls["sin"].dtype == torch.float32
    assert calls["positions"].tolist() == [0, 1, 2]
    assert calls["scale"] == 128**-0.5
    assert records == [("fused", 6)]


def test_indexer_score_weights_match_deepgemm_contract(monkeypatch):
    from batchgen.models.glm.glm5.sparse_prefill import select_packed_glm52_topk

    deep_gemm = types.ModuleType("deep_gemm")
    deep_gemm.fp8_mqa_logits = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "deep_gemm", deep_gemm)

    q_after_projection = torch.arange(2 * 2 * 4, dtype=torch.float32).reshape(
        2, 2, 4
    )
    q_scale = torch.tensor(
        [[[0.25], [0.5]], [[0.75], [1.0]]],
        dtype=torch.float32,
    )
    k_scale = torch.tensor([[1.25], [1.5]], dtype=torch.float32)
    quant_calls = []

    def fake_act_quant(value):
        quant_calls.append(value)
        if value.shape == (2, 4):
            return value, k_scale
        if value.shape == (2, 2, 4):
            return value, q_scale
        raise AssertionError(f"unexpected quantization shape {tuple(value.shape)}")

    fa3_backend = types.ModuleType("batchgen.attention.mla.fa3_backend")
    fa3_backend.act_quant = fake_act_quant
    fa3_backend.w8a16_gemm = lambda *_args, **_kwargs: None
    monkeypatch.setitem(
        sys.modules,
        "batchgen.attention.mla.fa3_backend",
        fa3_backend,
    )
    monkeypatch.setattr(
        sparse_prefill,
        "_fp8_linear_from_quantized",
        lambda *_args: q_after_projection.reshape(2, 8),
    )
    monkeypatch.setattr(
        sparse_prefill,
        "_fused_rope_hadamard_q",
        lambda q, _cos, _sin, _positions: q,
    )
    monkeypatch.setattr(
        sparse_prefill.torch,
        "mm",
        lambda left, right, out_dtype=None: left @ right,
    )

    captured = {}

    def fake_score_chunk(**kwargs):
        captured.update(kwargs)
        kwargs["output"].fill_(-1)
        return kwargs["output"]

    monkeypatch.setattr(
        sparse_prefill,
        "_score_packed_indexer_topk_chunk",
        fake_score_chunk,
    )
    hidden_states = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=torch.float32,
    )
    gate_weight = torch.tensor(
        [[1.0, 0.0, -1.0], [0.5, 1.0, 0.0]],
        dtype=torch.float32,
    )
    indexer = types.SimpleNamespace(
        index_topk=2048,
        index_n_heads=2,
        index_head_dim=4,
        rope_head_dim=2,
        softmax_scale=0.25,
        wq_b=types.SimpleNamespace(
            weight=types.SimpleNamespace(data=torch.empty(8, 1))
        ),
        wq_b_scale=torch.ones(1),
        weights_proj=types.SimpleNamespace(
            weight=types.SimpleNamespace(data=gate_weight)
        ),
        rotary_emb=lambda _value, _max_seqlen: (
            torch.ones(4, 2),
            torch.zeros(4, 2),
        ),
    )
    indexer_kv = torch.arange(2 * 4, dtype=torch.float32).reshape(1, 2, 1, 4)

    output = select_packed_glm52_topk(
        indexer=indexer,
        hidden_states=hidden_states,
        q_a_fp8=torch.empty(2, 1),
        q_a_scale=torch.empty(2, 1),
        indexer_kv=indexer_kv,
        position_ids=torch.tensor([0, 1], dtype=torch.int64),
        causal_starts=torch.tensor([0, 0], dtype=torch.int32),
        causal_ends=torch.tensor([1, 2], dtype=torch.int32),
        max_seqlen=2,
    )

    expected_gates = hidden_states @ gate_weight.t()
    expected_weights = (
        expected_gates.unsqueeze(-1)
        * 2**-0.5
        * q_scale
        * indexer.softmax_scale
    ).squeeze(-1)
    assert output.shape == (2, 2048)
    assert quant_calls[0].shape == (2, 4)
    assert quant_calls[1].shape == (2, 2, 4)
    assert torch.equal(captured["q_fp8"], q_after_projection)
    assert torch.equal(captured["k_fp8_with_scale"][1], k_scale.squeeze(-1))
    torch.testing.assert_close(captured["score_weights"], expected_weights)
    assert captured["max_seqlen"] == 2


def test_glm52_shared_layer_requires_carried_topk():
    with pytest.raises(RuntimeError, match="no carried top-k"):
        validate_carried_topk(None, total_tokens=3, index_topk=2048)


def test_glm52_shared_layer_validates_carried_topk_shape_and_dtype():
    with pytest.raises(RuntimeError, match="shape mismatch"):
        validate_carried_topk(
            torch.empty(2, 2048, dtype=torch.int32),
            total_tokens=3,
            index_topk=2048,
        )
    with pytest.raises(RuntimeError, match="must be int32"):
        validate_carried_topk(
            torch.empty(3, 2048, dtype=torch.int64),
            total_tokens=3,
            index_topk=2048,
        )


def test_glm52_sparse_prefill_path_audit_exact_schedule(caplog):
    from batchgen.models.glm.glm5.wrappers import GLM5AttnWrapper

    config = type("Config", (), {
        "model_type": "glm_moe_dsa_5_2",
        "num_hidden_layers": 78,
        "index_topk_freq": 4,
        "index_skip_topk_offset": 3,
    })()
    wrapper = object.__new__(GLM5AttnWrapper)
    wrapper.module = type("Module", (), {"config": config})()
    caplog.set_level(logging.INFO)
    GLM5AttnWrapper._reset_glm52_prefill_path_counts()
    for layer_idx in range(78):
        GLM5AttnWrapper._record_glm52_prefill_path("sparse", layer_idx)
        if layer_idx in [0, 1, 2, *range(6, 75, 4)]:
            GLM5AttnWrapper._record_glm52_prefill_path(
                "indexer_compute",
                layer_idx,
            )
        else:
            GLM5AttnWrapper._record_glm52_prefill_path(
                "indexer_reuse",
                layer_idx,
            )

    wrapper._finish_glm52_prefill_path_counts()

    assert "mode=sparse" in caplog.text
    assert GLM5AttnWrapper._dsa_prefill_path_counts is None


def test_glm52_sparse_prefill_path_audit_rejects_duplicate_compute():
    from batchgen.models.glm.glm5.wrappers import GLM5AttnWrapper

    config = type("Config", (), {
        "model_type": "glm_moe_dsa_5_2",
        "num_hidden_layers": 2,
        "index_topk_pattern": ["F", "S"],
    })()
    wrapper = object.__new__(GLM5AttnWrapper)
    wrapper.module = type("Module", (), {"config": config})()
    GLM5AttnWrapper._reset_glm52_prefill_path_counts()
    GLM5AttnWrapper._record_glm52_prefill_path("sparse", 0)
    GLM5AttnWrapper._record_glm52_prefill_path("sparse", 1)
    GLM5AttnWrapper._record_glm52_prefill_path("indexer_compute", 0)
    GLM5AttnWrapper._record_glm52_prefill_path("indexer_compute", 0)
    GLM5AttnWrapper._record_glm52_prefill_path("indexer_reuse", 1)

    with pytest.raises(RuntimeError, match="coverage mismatch"):
        wrapper._finish_glm52_prefill_path_counts()


def test_glm52_prefill_reuses_one_topk_buffer_across_shared_layers(monkeypatch):
    from batchgen.models.glm.glm5.sparse_prefill import Glm52SparsePrefillResult
    from batchgen.models.glm.glm5.wrappers import GLM5AttnWrapper
    from batchgen.models.wrappers import AttnWrapperBase

    config = types.SimpleNamespace(
        model_type="glm_moe_dsa_5_2",
        num_hidden_layers=5,
        index_topk=2048,
        index_topk_pattern=["F", "S", "S", "F", "S"],
    )
    topk = torch.full((3, 2048), -1, dtype=torch.int32)
    calls = []
    offloads = []

    def fake_sparse_prefill(**kwargs):
        carried = kwargs["carried_topk_indices"]
        reusable = kwargs["reusable_topk_indices"]
        calls.append(
            (
                kwargs["attn"].layer_idx,
                kwargs["indexer"] is not None,
                carried is topk,
                reusable is topk,
                None if carried is None else carried[0, 0].item(),
            )
        )
        output = reusable
        if output is None:
            output = topk
        if kwargs["indexer"] is not None:
            output.fill_(kwargs["attn"].layer_idx)
        return Glm52SparsePrefillResult(
            attn_output=torch.zeros(3, 4),
            primary_kv=torch.zeros(3, 576),
            indexer_kv=(
                torch.zeros(1, 3, 1, 128)
                if kwargs["indexer"] is not None
                else None
            ),
            topk_indices=output,
        )

    monkeypatch.setattr(
        sparse_prefill,
        "glm52_sparse_prefill_prepacked",
        fake_sparse_prefill,
    )
    monkeypatch.setattr(
        AttnWrapperBase,
        "retire_pending_prefill_offloads_before_layer",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        GLM5AttnWrapper,
        "_offload_prepacked_indexer_kv",
        lambda self, _kv: offloads.append(("aux", self.layer_idx)),
    )
    monkeypatch.setattr(
        GLM5AttnWrapper,
        "_offload_prepacked_kv",
        lambda self, _kv: offloads.append(("primary", self.layer_idx)),
    )

    wrappers = []
    for layer_idx, is_full in enumerate((True, False, False, True, False)):
        wrapper = object.__new__(GLM5AttnWrapper)
        wrapper.layer_idx = layer_idx
        wrapper.prepack_mode = True
        wrapper.position_ids = torch.arange(3, dtype=torch.int64)
        wrapper.prepack_cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)
        wrapper.prepack_max_seqlen = 2049
        wrapper.prepack_num_sequences = 1
        wrapper.prepack_seq_lengths = [3]
        wrapper.cur_batch = ["sequence-0"]
        wrapper.weight_dequant_scale = {}
        wrapper.module = types.SimpleNamespace(
            layer_idx=layer_idx,
            config=config,
            indexer=object() if is_full else None,
            next_skip_topk=layer_idx in (0, 3),
        )
        wrappers.append(wrapper)

    GLM5AttnWrapper._dsa_prefill_prev_topk_indices = None
    GLM5AttnWrapper._dsa_prefill_causal_starts = None
    GLM5AttnWrapper._dsa_prefill_causal_ends = None
    GLM5AttnWrapper._dsa_prefill_path_counts = None
    for wrapper in wrappers:
        wrapper._forward_prefill(torch.ones(1, 3, 4))

    assert calls == [
        (0, True, False, False, None),
        (1, False, True, False, 0),
        (2, False, True, False, 0),
        (3, True, False, True, None),
        (4, False, True, False, 3),
    ]
    assert offloads == [
        ("aux", 0),
        ("primary", 0),
        ("primary", 1),
        ("primary", 2),
        ("aux", 3),
        ("primary", 3),
        ("primary", 4),
    ]
    assert GLM5AttnWrapper._dsa_prefill_prev_topk_indices is None
    assert GLM5AttnWrapper._dsa_prefill_causal_starts is None
    assert GLM5AttnWrapper._dsa_prefill_causal_ends is None
    assert GLM5AttnWrapper._dsa_prefill_path_counts is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_offset_topk_indices_preserves_invalid_tail():
    indices = torch.tensor(
        [[0, 2, -1, -1], [1, 3, 4, -1]],
        dtype=torch.int32,
        device="cuda",
    )
    starts = torch.tensor([10, 20], dtype=torch.int32, device="cuda")

    offset_packed_topk_indices_(indices, starts)

    assert indices.cpu().tolist() == [
        [10, 12, -1, -1],
        [21, 23, 24, -1],
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_glm52_sparse_prefill_runtime_smoke():
    device_index = torch.cuda.current_device()
    sparse_prefill._VALIDATED_RUNTIME_DEVICES.discard(device_index)

    sparse_prefill.validate_glm52_sparse_prefill_runtime()

    assert device_index in sparse_prefill._VALIDATED_RUNTIME_DEVICES


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_packed_indexer_scoring_is_causal_and_sequence_local_at_2049():
    from batchgen.attention.mla.fa3_backend import act_quant
    from batchgen.models.glm.glm5.sparse_prefill import (
        _score_packed_indexer_topk_chunk,
    )

    device = "cuda"
    seq_len = 2049
    total_tokens = seq_len * 2
    q = torch.ones(2, 32, 128, dtype=torch.bfloat16, device=device)
    k = torch.ones(total_tokens, 128, dtype=torch.bfloat16, device=device)
    k[0] = 0
    k[seq_len] = 0
    q_fp8, q_scale = act_quant(q)
    k_fp8, k_scale = act_quant(k)
    weights = (
        torch.ones(2, 32, device=device, dtype=torch.float32)
        * q_scale.squeeze(-1)
        * (32**-0.5)
        * (128**-0.5)
    ).contiguous()
    starts = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    ends = torch.tensor(
        [seq_len, total_tokens],
        dtype=torch.int32,
        device=device,
    )
    lengths = ends - starts
    output = torch.empty(2, 2048, dtype=torch.int32, device=device)

    _score_packed_indexer_topk_chunk(
        q_fp8=q_fp8,
        k_fp8_with_scale=(k_fp8, k_scale.squeeze(-1).contiguous()),
        score_weights=weights,
        causal_starts=starts,
        causal_ends=ends,
        causal_lengths=lengths,
        output=output,
        max_seqlen=seq_len,
    )

    output = output.cpu()
    assert torch.equal(
        output[0].sort().values,
        torch.arange(1, seq_len, dtype=torch.int32),
    )
    assert torch.equal(
        output[1].sort().values,
        torch.arange(seq_len + 1, total_tokens, dtype=torch.int32),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pinned_flashmla_sparse_prefill_matches_reference():
    from flash_mla import flash_mla_sparse_fwd

    torch.manual_seed(0)
    device = "cuda"
    s_q, s_kv, heads, dim, topk = 4, 2050, 64, 576, 2048
    q = (torch.randn(s_q, heads, dim, device=device) * 0.1).to(torch.bfloat16)
    kv = (torch.randn(s_kv, 1, dim, device=device) * 0.1).to(torch.bfloat16)
    indices = torch.full(
        (s_q, 1, topk),
        -1,
        dtype=torch.int32,
        device=device,
    )
    for row in range(s_q):
        indices[row, 0, : row + 1] = torch.arange(
            row + 1,
            dtype=torch.int32,
            device=device,
        )

    scale = dim**-0.5
    output, _, _ = flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale=scale,
        d_v=512,
    )

    kv2 = kv[:, 0].float()
    safe = indices[:, 0].clamp_min(0).long()
    selected = kv2[safe]
    invalid = indices[:, 0] < 0
    scores = torch.einsum("qhd,qkd->qhk", q.float(), selected) * scale
    scores.masked_fill_(invalid.unsqueeze(1), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    reference = torch.einsum("qhk,qkd->qhd", probs, selected[:, :, :512])

    torch.testing.assert_close(
        output.float(),
        reference,
        atol=8e-4,
        rtol=2.01 / 128,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sparse_absorbed_layer_matches_dense_attention(monkeypatch):
    from flash_mla import flash_mla_sparse_fwd

    from batchgen.attention.mla import fa3_backend, flashmla_backend
    from batchgen.models.glm.glm5.sparse_prefill import (
        glm52_sparse_prefill_prepacked,
    )

    torch.manual_seed(1)
    device = "cuda"
    tokens = 16
    num_heads = 64
    qk_nope_dim = 8
    rope_dim = 64
    kv_lora_rank = 512
    v_head_dim = 4
    q_lora_rank = 16
    hidden_size = num_heads * v_head_dim

    q_a = torch.zeros(tokens, q_lora_rank, dtype=torch.bfloat16, device=device)
    q = (
        torch.randn(
            tokens,
            num_heads,
            qk_nope_dim + rope_dim,
            dtype=torch.float32,
            device=device,
        )
        * 0.02
    ).to(torch.bfloat16)
    latent_kv = (
        torch.randn(tokens, kv_lora_rank, dtype=torch.float32, device=device)
        * 0.02
    ).to(torch.bfloat16)
    k_pe = (
        torch.randn(tokens, rope_dim, dtype=torch.float32, device=device)
        * 0.02
    ).to(torch.bfloat16)
    q_absorb = (
        torch.randn(
            num_heads,
            qk_nope_dim,
            kv_lora_rank,
            dtype=torch.float32,
            device=device,
        )
        * 0.01
    ).to(torch.bfloat16)
    out_absorb = (
        torch.randn(
            num_heads,
            v_head_dim,
            kv_lora_rank,
            dtype=torch.float32,
            device=device,
        )
        * 0.01
    ).to(torch.bfloat16)
    kv_b = torch.cat([q_absorb, out_absorb], dim=1).reshape(
        num_heads * (qk_nope_dim + v_head_dim),
        kv_lora_rank,
    )

    q_a_weight = torch.empty(q_lora_rank, 1, device=device)
    kv_a_weight = torch.empty(kv_lora_rank + rope_dim, 1, device=device)
    q_b_weight = torch.empty(
        num_heads * (qk_nope_dim + rope_dim),
        1,
        device=device,
    )
    o_weight = torch.empty(hidden_size, 1, device=device)

    def fake_fp8_linear(weight, _weight_scale, _x_fp8, _x_scale):
        if weight is q_a_weight:
            return q_a
        if weight is kv_a_weight:
            return torch.cat([latent_kv, k_pe], dim=-1)
        if weight is q_b_weight:
            return q.reshape(tokens, -1)
        raise AssertionError(f"unexpected projection shape {tuple(weight.shape)}")

    monkeypatch.setattr(
        sparse_prefill,
        "_fp8_linear_from_quantized",
        fake_fp8_linear,
    )
    monkeypatch.setattr(
        flashmla_backend,
        "deepseek_v3_dequantization",
        lambda _weight, _scale: kv_b,
    )
    real_w8a16_gemm = fa3_backend.w8a16_gemm
    monkeypatch.setattr(
        fa3_backend,
        "w8a16_gemm",
        lambda weight, _scale, activation: (
            activation if weight is o_weight else None
        ),
    )

    class IdentityRotary:
        def __call__(self, _x, seq_len):
            return (
                torch.ones(seq_len, rope_dim, dtype=torch.float32, device=device),
                torch.zeros(seq_len, rope_dim, dtype=torch.float32, device=device),
            )

    attn = types.SimpleNamespace(
        layer_idx=3,
        q_a_proj=types.SimpleNamespace(weight=types.SimpleNamespace(data=q_a_weight)),
        q_a_layernorm=lambda value: value,
        kv_a_proj_with_mqa=types.SimpleNamespace(
            weight=types.SimpleNamespace(data=kv_a_weight)
        ),
        kv_a_layernorm=lambda value: value,
        q_b_proj=types.SimpleNamespace(weight=types.SimpleNamespace(data=q_b_weight)),
        kv_b_proj=types.SimpleNamespace(weight=types.SimpleNamespace(data=object())),
        o_proj=types.SimpleNamespace(weight=types.SimpleNamespace(data=o_weight)),
        rotary_emb=IdentityRotary(),
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=rope_dim,
        qk_nope_head_dim=qk_nope_dim,
        q_head_dim=qk_nope_dim + rope_dim,
        v_head_dim=v_head_dim,
        num_heads=num_heads,
        hidden_size=hidden_size,
        softmax_scale=256**-0.5,
        config=types.SimpleNamespace(index_topk=2048),
    )
    indices = torch.full(
        (tokens, 2048),
        -1,
        dtype=torch.int32,
        device=device,
    )
    for row in range(tokens):
        indices[row, : row + 1] = torch.arange(
            row + 1,
            dtype=torch.int32,
            device=device,
        )
    scales = {
        "q_a_proj.weight_scale_inv": torch.ones(1, device=device),
        "kv_a_proj_with_mqa.weight_scale_inv": torch.ones(1, device=device),
        "q_b_proj.weight_scale_inv": torch.ones(1, device=device),
        "kv_b_proj.weight_scale_inv": torch.ones(1, device=device),
        "o_proj.weight_scale_inv": torch.ones(1, device=device),
    }
    hidden_states = torch.zeros(
        tokens,
        128,
        dtype=torch.bfloat16,
        device=device,
    )
    position_ids = torch.arange(tokens, dtype=torch.int64, device=device)

    result = glm52_sparse_prefill_prepacked(
        attn=attn,
        hidden_states=hidden_states,
        position_ids=position_ids,
        max_seqlen=tokens,
        weight_scale=scales,
        indexer=None,
        carried_topk_indices=indices,
        reusable_topk_indices=None,
        causal_starts=None,
        causal_ends=None,
    )

    q_nope, q_rope = torch.split(q.float(), [qk_nope_dim, rope_dim], dim=-1)
    k_nope = torch.einsum("tc,hnc->thn", latent_kv.float(), q_absorb.float())
    values = torch.einsum("tc,hvc->thv", latent_kv.float(), out_absorb.float())
    keys = torch.cat(
        [k_nope, k_pe.float().unsqueeze(1).expand(-1, num_heads, -1)],
        dim=-1,
    )
    queries = torch.cat([q_nope, q_rope], dim=-1)
    scores = torch.einsum("thd,khd->thk", queries, keys) * attn.softmax_scale
    scores.masked_fill_(
        torch.triu(
            torch.ones(tokens, tokens, dtype=torch.bool, device=device),
            diagonal=1,
        ).unsqueeze(1),
        float("-inf"),
    )
    probabilities = torch.softmax(scores, dim=-1)
    reference = torch.einsum("thk,khd->thd", probabilities, values).reshape(
        tokens,
        hidden_size,
    )

    assert result.topk_indices is indices
    assert result.indexer_kv is None
    assert fa3_backend.w8a16_gemm is not real_w8a16_gemm
    torch.testing.assert_close(
        result.attn_output.float(),
        reference,
        atol=8e-4,
        rtol=3.01 / 128,
    )
