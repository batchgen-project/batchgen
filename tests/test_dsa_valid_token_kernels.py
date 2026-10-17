import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _assert_bf16_wgmma_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    diff = (actual.float() - expected.float()).abs()
    tol = 1e-5 + 1.6e-2 * expected.float().abs()
    n_fail = (diff > tol).sum().item()
    n_total = diff.numel()
    assert n_fail == 0 or n_fail / n_total < 1e-4, (
        f"max_abs={diff.max().item()} failures={n_fail}/{n_total}"
    )


@pytest.mark.parametrize("rows", [1, 4])
def test_act_quant_valid_tokens_zeroes_padding_rows(rows: int):
    from batchgen.attention.mla.fa3_backend import act_quant

    valid_rows = min(rows, 2)
    x = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
    x[valid_rows:].fill_(123)
    num_valid = torch.tensor([valid_rows], device="cuda", dtype=torch.int32)

    y, scale = act_quant(
        x.contiguous(),
        num_valid_tokens=num_valid,
        scale_tma_aligned=True,
    )
    torch.cuda.synchronize()

    aligned_m = ((rows + 3) // 4) * 4
    assert scale.stride() == (1, aligned_m)
    assert torch.count_nonzero(y[valid_rows:].float()).item() == 0
    assert torch.all(scale[valid_rows:] == 1e-12)


def test_head_gates_valid_tokens_zeroes_padding_rows():
    from batchgen_kernels.attention.dsa.head_gates import head_gates_out

    hidden = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
    hidden[2:].fill_(99)
    weight = torch.randn(3, 8, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(4, 3, device="cuda", dtype=torch.float32)
    num_valid = torch.tensor([2], device="cuda", dtype=torch.int32)
    scale = 0.125

    head_gates_out(hidden, weight, out, scale=scale, num_valid_tokens=num_valid)
    torch.cuda.synchronize()

    ref = torch.nn.functional.linear(hidden[:2].float(), weight.float()) * scale
    assert torch.allclose(out[:2], ref, atol=1e-3, rtol=1e-3)
    assert torch.count_nonzero(out[2:]).item() == 0


def test_query_pack_valid_tokens_zeroes_padding_rows():
    from batchgen_kernels.attention.dsa.query_pack import pack_flashmla_query_out

    absorbed = torch.randn(4, 2, 512, device="cuda", dtype=torch.bfloat16)
    rope = torch.randn(4, 2, 64, device="cuda", dtype=torch.bfloat16)
    absorbed[2:].fill_(77)
    rope[2:].fill_(88)
    out = torch.empty(4, 1, 2, 576, device="cuda", dtype=torch.bfloat16)
    num_valid = torch.tensor([2], device="cuda", dtype=torch.int32)

    pack_flashmla_query_out(absorbed, rope, out, num_valid_tokens=num_valid)
    torch.cuda.synchronize()

    assert torch.equal(out[:2, 0, :, :512], absorbed[:2])
    assert torch.equal(out[:2, 0, :, 512:], rope[:2])
    assert torch.count_nonzero(out[2:].float()).item() == 0


@pytest.mark.parametrize(
    ("batch", "valid_rows", "heads"),
    [
        (4, 0, 2),
        (4, 2, 2),
        (32, 0, 64),
    ],
)
def test_fp8_absorb_valid_tokens_zeroes_padding_rows(batch: int, valid_rows: int, heads: int):
    from batchgen_kernels.attention.dsa.fp8_absorb import (
        FP8AbsorbWeights,
        fp8_out_absorb_out,
        fp8_q_absorb_out,
    )

    torch.manual_seed(1234)
    num_valid = torch.tensor([valid_rows], device="cuda", dtype=torch.int32)
    q_absorb = (torch.randn(heads, 192, 512, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    out_absorb = (torch.randn(heads, 256, 512, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    weights = FP8AbsorbWeights(q_absorb, out_absorb)

    q_nope = (torch.randn(batch, heads, 192, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    q_nope[valid_rows:].fill_(77)
    q_full = torch.empty(batch, heads, 512, device="cuda", dtype=torch.bfloat16)
    q_valid = torch.full_like(q_full, 9)
    fp8_q_absorb_out(q_nope, weights, q_full)
    fp8_q_absorb_out(q_nope, weights, q_valid, num_valid_tokens=num_valid)

    attn = (torch.randn(batch, 1, heads, 512, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    attn[valid_rows:].fill_(88)
    out_full = torch.empty(batch, 1, heads, 256, device="cuda", dtype=torch.bfloat16)
    out_valid = torch.full_like(out_full, 11)
    fp8_out_absorb_out(attn, weights, out_full)
    fp8_out_absorb_out(attn, weights, out_valid, num_valid_tokens=num_valid)
    torch.cuda.synchronize()

    _assert_bf16_wgmma_close(q_valid[:valid_rows], q_full[:valid_rows])
    _assert_bf16_wgmma_close(out_valid[:valid_rows], out_full[:valid_rows])
    assert torch.count_nonzero(q_valid[valid_rows:].float()).item() == 0
    assert torch.count_nonzero(out_valid[valid_rows:].float()).item() == 0


def test_fp8_out_absorb_zero_valid_cuda_graph_capture():
    from batchgen_kernels.attention.dsa.fp8_absorb import (
        FP8AbsorbWeights,
        fp8_out_absorb_out,
    )

    torch.manual_seed(5678)
    batch = 32
    heads = 64
    out_absorb = (torch.randn(heads, 256, 512, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    q_absorb = (torch.randn(heads, 192, 512, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    weights = FP8AbsorbWeights(q_absorb, out_absorb)
    num_valid = torch.zeros(1, device="cuda", dtype=torch.int32)
    attn = (torch.randn(batch, 1, heads, 512, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    out = torch.full((batch, 1, heads, 256), 7.0, device="cuda", dtype=torch.bfloat16)

    fp8_out_absorb_out(attn, weights, out, num_valid_tokens=num_valid)
    torch.cuda.synchronize()
    assert torch.count_nonzero(out.float()).item() == 0

    out.fill_(7)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fp8_out_absorb_out(attn, weights, out, num_valid_tokens=num_valid)
    graph.replay()
    torch.cuda.synchronize()

    assert torch.count_nonzero(out.float()).item() == 0


def test_paged_kv_update_valid_tokens_skips_invalid_slots():
    from batchgen_kernels.triton.kv_cache import run_paged_kv_token_update_fused

    cache = torch.zeros(2, 4, 1, 3, device="cuda", dtype=torch.bfloat16)
    tokens = torch.arange(12, device="cuda", dtype=torch.float32).view(4, 3).to(torch.bfloat16)
    page_table = torch.tensor([[0], [1]], device="cuda", dtype=torch.int32)
    slots = torch.tensor([0, 1, 12345, 67890], device="cuda", dtype=torch.int32)
    token_indices = torch.tensor([0, 1, 999, 999], device="cuda", dtype=torch.int32)
    num_valid = torch.tensor([2], device="cuda", dtype=torch.int32)

    run_paged_kv_token_update_fused(
        k_cache=cache,
        k_tokens=tokens,
        page_table=page_table,
        slot_indices=slots,
        token_indices=token_indices,
        page_size_tokens=4,
        num_valid_tokens=num_valid,
    )
    torch.cuda.synchronize()

    assert torch.equal(cache[0, 0, 0], tokens[0])
    assert torch.equal(cache[1, 1, 0], tokens[1])
    assert torch.count_nonzero(cache[0, 1:]).item() == 0
    assert torch.count_nonzero(cache[1, :1]).item() == 0
    assert torch.count_nonzero(cache[1, 2:]).item() == 0


def test_score_topk_valid_tokens_skips_padding_rows():
    from batchgen_kernels.attention.dsa.fused_indexer_score import (
        fused_paged_score_and_topk_with_slots_out,
    )

    q = torch.randn(4, 1, 4, device="cuda", dtype=torch.bfloat16)
    aux_k = torch.randn(2, 4, 1, 4, device="cuda", dtype=torch.bfloat16)
    page_table = torch.tensor([[0], [1]], device="cuda", dtype=torch.int32)
    slots = torch.tensor([0, 1, 999, 999], device="cuda", dtype=torch.int32)
    gates = torch.ones(4, 1, device="cuda", dtype=torch.float32)
    cache_seqlens = torch.tensor([2, 3, 0, 0], device="cuda", dtype=torch.int32)
    agg = torch.empty(4, 4, device="cuda", dtype=torch.float32)
    topk = torch.empty(4, 2, device="cuda", dtype=torch.int32)
    num_valid = torch.tensor([2], device="cuda", dtype=torch.int32)

    fused_paged_score_and_topk_with_slots_out(
        q,
        aux_k,
        page_table,
        slots,
        gates,
        cache_seqlens,
        agg,
        topk,
        topk=2,
        page_size=4,
        max_seqlen=4,
        num_valid_tokens=num_valid,
    )
    torch.cuda.synchronize()

    assert torch.all(topk[2:] == -1)
    assert torch.all(torch.isneginf(agg[2:]))


def test_fast_topk_2048_eager_path_omits_num_valid_tokens(monkeypatch):
    from batchgen_kernels.attention.dsa import fast_topk_cuda

    score = torch.randn(4, 4096, device="cuda", dtype=torch.float32)
    lengths = torch.tensor([1, 2, 3, 4], device="cuda", dtype=torch.int32)
    captured = {}

    class FakeModule:
        def fast_topk_2048_out(self, *args):
            captured["args"] = args

    monkeypatch.setattr(fast_topk_cuda, "_get_module", lambda: FakeModule())

    indices = fast_topk_cuda.fast_topk_2048(score, lengths)
    torch.cuda.synchronize()

    assert indices.shape == (4, 2048)
    assert len(captured["args"]) == 4
    assert captured["args"][3] is None


def test_selector_valid_tokens_skips_padding_rows():
    from batchgen_kernels.attention.dsa.fused_unified_selector import (
        fused_select_mla_kv_bf16_out,
    )

    blocked_kv = torch.arange(2 * 4 * 1 * 4, device="cuda", dtype=torch.float32).view(2, 4, 1, 4).to(torch.bfloat16)
    page_table = torch.tensor([[0], [1]], device="cuda", dtype=torch.int32)
    cache_seqlens = torch.tensor([2, 3, 0, 0], device="cuda", dtype=torch.int32)
    topk = torch.zeros(4, 4, device="cuda", dtype=torch.int32)
    slots = torch.tensor([0, 1, 999, 999], device="cuda", dtype=torch.int32)
    selected = torch.empty(4, 4, 1, 4, device="cuda", dtype=torch.bfloat16)
    lengths = torch.empty(4, device="cuda", dtype=torch.int32)
    row_modes = torch.empty(4, device="cuda", dtype=torch.int32)
    num_valid = torch.tensor([2], device="cuda", dtype=torch.int32)

    fused_select_mla_kv_bf16_out(
        blocked_kv,
        page_table,
        cache_seqlens,
        topk,
        4,
        selected,
        lengths,
        None,
        row_modes,
        return_indices=False,
        primary_slot_indices=slots,
        num_valid_tokens=num_valid,
    )
    torch.cuda.synchronize()

    assert torch.equal(lengths, torch.tensor([2, 3, 0, 0], device="cuda", dtype=torch.int32))
    assert torch.equal(row_modes[2:], torch.tensor([2, 2], device="cuda", dtype=torch.int32))
    assert torch.count_nonzero(selected[2:].float()).item() == 0


@pytest.mark.parametrize("batch_size,valid_m", [(64, 17), (128, 64)])
def test_indexer_wgmma_valid_m_variant_zeroes_padding_rows(batch_size: int, valid_m: int):
    from batchgen_kernels.attention.dsa.fused_indexer_kv_proj_cuda import (
        FP8IndexerWeightsCUDA,
        build_module,
    )

    torch.manual_seed(batch_size + valid_m)
    module = build_module()
    device = torch.device("cuda")
    K = 128
    N = 32
    padded_batch = max(batch_size, 64)
    valid = torch.tensor([valid_m], device=device, dtype=torch.int32)

    x_fp8 = (torch.randn(padded_batch, K, device=device) * 0.1).to(torch.float8_e4m3fn)
    x_scale = torch.rand(batch_size, device=device, dtype=torch.float32) * 0.01 + 1e-4
    weights = FP8IndexerWeightsCUDA(
        (torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.1).contiguous(),
        module,
    )
    a_tma_desc = module.create_tma_desc(x_fp8, padded_batch, K, 64, 128)
    full_out = torch.empty(batch_size, N, device=device, dtype=torch.bfloat16)
    valid_out = torch.full(
        (batch_size, N),
        7.0,
        device=device,
        dtype=torch.bfloat16,
    )

    module.indexer_kv_proj_gemm_only_out(
        a_tma_desc,
        weights.tma_desc,
        weights.w_scale,
        x_scale,
        full_out,
        batch_size,
        N,
        K,
    )
    module.indexer_kv_proj_gemm_only_valid_m_out(
        a_tma_desc,
        weights.tma_desc,
        weights.w_scale,
        x_scale,
        valid_out,
        valid,
        batch_size,
        N,
        K,
    )
    torch.cuda.synchronize()

    _assert_bf16_wgmma_close(valid_out[:valid_m], full_out[:valid_m])
    assert torch.count_nonzero(valid_out[valid_m:].float()).item() == 0


def test_score_topk_skips_dense_rows_and_keeps_long_row_result():
    from batchgen_kernels.attention.dsa.fused_indexer_score import (
        fused_paged_score_and_topk_with_slots_out,
    )

    torch.manual_seed(20260820)
    batch_size = 3
    num_heads = 1
    head_dim = 4
    page_size = 64
    max_seqlen = 2112
    topk_count = 2048
    q = torch.randn(
        batch_size,
        num_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    aux_k = torch.randn(
        batch_size * (max_seqlen // page_size),
        page_size,
        1,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    page_table = torch.arange(
        batch_size * (max_seqlen // page_size),
        device="cuda",
        dtype=torch.int32,
    ).view(batch_size, max_seqlen // page_size)
    slots = torch.arange(batch_size, device="cuda", dtype=torch.int32)
    gates = torch.randn(
        batch_size,
        num_heads,
        device="cuda",
        dtype=torch.float32,
    )
    cache_seqlens = torch.tensor(
        [17, topk_count, topk_count + 1],
        device="cuda",
        dtype=torch.int32,
    )
    agg = torch.empty(
        batch_size,
        max_seqlen,
        device="cuda",
        dtype=torch.float32,
    )
    topk = torch.empty(
        batch_size,
        topk_count,
        device="cuda",
        dtype=torch.int32,
    )

    fused_paged_score_and_topk_with_slots_out(
        q,
        aux_k,
        page_table,
        slots,
        gates,
        cache_seqlens,
        agg,
        topk,
        topk=topk_count,
        page_size=page_size,
        max_seqlen=max_seqlen,
    )
    torch.cuda.synchronize()

    assert torch.equal(
        topk[0, :17],
        torch.arange(17, device="cuda", dtype=torch.int32),
    )
    assert torch.all(topk[0, 17:] == -1)
    assert torch.equal(
        topk[1],
        torch.arange(topk_count, device="cuda", dtype=torch.int32),
    )
    assert torch.all(torch.isneginf(agg[:2]))

    long_k = aux_k[
        2 * (max_seqlen // page_size) : 3 * (max_seqlen // page_size)
    ].reshape(max_seqlen, head_dim)[: topk_count + 1].float()
    reference_scores = (
        q[2].float().unsqueeze(1) * long_k.unsqueeze(0)
    ).sum(dim=-1)
    reference_scores = (
        reference_scores * gates[2].view(num_heads, 1)
    ).sum(dim=0)
    reference_topk = torch.topk(
        reference_scores,
        topk_count,
    ).indices.sort().values
    assert torch.equal(topk[2].sort().values, reference_topk.to(torch.int32))


def test_fused_rmsnorm_rope_native_accepts_strided_kv_input():
    from batchgen_kernels.triton.fused_rmsnorm_rope import (
        fused_rmsnorm_rope_with_q_native,
    )

    torch.manual_seed(20260819)
    batch_size = 3
    num_heads = 2
    kv_lora_rank = 8
    rope_dim = 4
    total_dim = kv_lora_rank + rope_dim
    base = torch.randn(
        batch_size,
        total_dim + 5,
        device="cuda",
        dtype=torch.bfloat16,
    )
    strided_kv = base[:, :total_dim].view(batch_size, 1, total_dim)
    assert not strided_kv.is_contiguous()
    contiguous_kv = strided_kv.contiguous()
    q = torch.randn(
        batch_size,
        num_heads,
        1,
        rope_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q_strided = q.clone()
    q_contiguous = q.clone()
    cos = torch.randn(16, rope_dim, device="cuda", dtype=torch.float32)
    sin = torch.randn_like(cos)
    positions = torch.tensor([[1], [4], [7]], device="cuda", dtype=torch.int64)
    weight = torch.randn(kv_lora_rank, device="cuda", dtype=torch.bfloat16)

    strided_out = fused_rmsnorm_rope_with_q_native(
        strided_kv,
        q_strided,
        cos,
        sin,
        positions,
        weight,
        kv_lora_rank,
        rope_dim,
    )
    contiguous_out = fused_rmsnorm_rope_with_q_native(
        contiguous_kv,
        q_contiguous,
        cos,
        sin,
        positions,
        weight,
        kv_lora_rank,
        rope_dim,
    )
    torch.cuda.synchronize()

    assert torch.equal(strided_out, contiguous_out)
    assert torch.equal(q_strided, q_contiguous)


def test_triton_rmsnorm_matches_cuda_on_strided_q_a_view():
    from batchgen.attention.fused_kernels import cuda_rmsnorm
    from batchgen_kernels.triton.rmsnorm import fused_rmsnorm

    torch.manual_seed(20260820)
    batch_size = 7
    q_rank = 2048
    fused_width = q_rank + 576
    fused = torch.randn(
        batch_size,
        fused_width,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q_a = fused[:, :q_rank]
    assert not q_a.is_contiguous()
    weight = torch.randn(q_rank, device="cuda", dtype=torch.bfloat16)
    reference = cuda_rmsnorm(q_a, weight, 1e-5)
    candidate = torch.empty_like(reference)
    fused_rmsnorm(q_a, weight, 1e-5, out=candidate)
    torch.cuda.synchronize()

    diff = (reference.float() - candidate.float()).abs()
    assert float(diff.max()) <= 0.03125
    assert torch.nn.functional.cosine_similarity(
        reference.float().reshape(-1),
        candidate.float().reshape(-1),
        dim=0,
    ) >= 0.9999


# ---------------------------------------------------------------------------
# Containment of NON-FINITE padding rows.
#
# The GLM-5 decode segments used to mask every masked-DeepGEMM output with a
# full-buffer `mul_(valid_rows_bf16)`, so an invalid row's activations were
# always exactly zero and could never be NaN or Inf. Those per-layer masks are
# gone: the only mask left is on the `attn_output` that leaves the segment. An
# invalid row now carries whatever the previous replay left behind, and a
# row-local op on stale garbage can produce a non-finite value (rmsnorm of a
# huge row, for instance). These tests pin the three gates the removal leans
# on, with the padding rows deliberately poisoned.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("poison", [float("nan"), float("inf"), -float("inf"), 3.0e38])
def test_act_quant_contains_non_finite_padding_rows(poison: float):
    """The keystone: no bf16 garbage reaches a GEMM, finite or not.

    `act_quant` writes the padding rows unconditionally (zeros + eps scale)
    rather than quantizing them, so a non-finite stale row cannot become a
    non-finite FP8 row, and the live rows must be bit-identical to the same
    call with clean padding.
    """
    from batchgen.attention.mla.fa3_backend import act_quant

    rows, valid_rows, cols = 8, 3, 256
    torch.manual_seed(0)
    live = torch.randn(valid_rows, cols, device="cuda", dtype=torch.bfloat16)

    clean = torch.zeros(rows, cols, device="cuda", dtype=torch.bfloat16)
    clean[:valid_rows] = live
    poisoned = clean.clone()
    poisoned[valid_rows:].fill_(poison)
    num_valid = torch.tensor([valid_rows], device="cuda", dtype=torch.int32)

    y_clean, s_clean = act_quant(
        clean, num_valid_tokens=num_valid, scale_tma_aligned=True
    )
    y_poisoned, s_poisoned = act_quant(
        poisoned, num_valid_tokens=num_valid, scale_tma_aligned=True
    )
    torch.cuda.synchronize()

    # Live rows: bitwise, not approximate. Removing the upstream masks must not
    # move a single bit of a real token's activations.
    assert torch.equal(
        y_poisoned[:valid_rows].view(torch.uint8), y_clean[:valid_rows].view(torch.uint8)
    )
    assert torch.equal(s_poisoned[:valid_rows], s_clean[:valid_rows])
    # Padding rows: exactly zero, and finite, whatever went in.
    padded = y_poisoned[valid_rows:].float()
    assert torch.count_nonzero(padded).item() == 0
    assert torch.isfinite(padded).all()
    assert torch.all(s_poisoned[valid_rows:] == 1e-12)


def test_selected_positions_zeroes_lengths_for_invalid_rows_with_poisoned_topk():
    """FA3 reads nothing for an invalid row, whatever the top-k buffer holds.

    `top_k_indices` is produced by a row-gated scorer, so an invalid row keeps
    whatever a previous replay wrote. The transform must still store
    `selected_lengths = 0` for it -- that, not the content of the index buffer,
    is what keeps a stale page out of the attention read.
    """
    from batchgen_kernels.attention.dsa.selected_page_table import (
        transform_selected_positions_out,
    )

    batch, index_topk, slots, max_pages = 4, 8, 2, 4
    page_table = torch.arange(slots * max_pages, device="cuda", dtype=torch.int32).view(
        slots, max_pages
    )
    cache_seqlens = torch.tensor([6, 5, 9, 7], device="cuda", dtype=torch.int32)
    # Rows 2-3 are padding; poison their selections with out-of-range positions.
    topk = torch.zeros(batch, index_topk, device="cuda", dtype=torch.int32)
    topk[2:] = 123456
    slot_indices = torch.tensor([0, 1, 0, 1], device="cuda", dtype=torch.int32)
    physical = torch.full((batch, index_topk), 7, device="cuda", dtype=torch.int32)
    lengths = torch.full((batch,), 7, device="cuda", dtype=torch.int32)
    num_valid = torch.tensor([2], device="cuda", dtype=torch.int32)

    transform_selected_positions_out(
        page_table,
        cache_seqlens,
        topk,
        physical,
        lengths,
        page_size=2,
        primary_slot_indices=slot_indices,
        num_valid_tokens=num_valid,
    )
    torch.cuda.synchronize()

    assert torch.equal(lengths[2:], torch.zeros(2, device="cuda", dtype=torch.int32))
    assert torch.all(physical[2:] < 0), "invalid rows must select no physical page"
    # ...and the live rows are untouched by the poison.
    assert torch.all(lengths[:2] > 0)


def test_masked_select_not_multiply_cleans_the_offload_staging_write():
    """Why the primary_k_tensor copy became a select.

    The staging tensor leaves the segment for host KV offload, so an invalid
    row's bytes would escape. bf16 `NaN * 0` is `NaN`, so the old
    multiply-after-the-fact cannot clean a non-finite stale row; a select can.
    """
    batch, kv_dim, valid_rows = 4, 16, 2
    offload = torch.randn(batch, 1, 1, kv_dim, device="cuda", dtype=torch.bfloat16)
    offload[valid_rows:] = float("nan")
    valid_mask = torch.tensor(
        [True] * valid_rows + [False] * (batch - valid_rows), device="cuda"
    )
    zero = torch.zeros(1, device="cuda", dtype=torch.bfloat16)
    staged = torch.full_like(offload, 7.0)

    torch.where(valid_mask.view(batch, 1, 1, 1), offload, zero, out=staged)
    torch.cuda.synchronize()

    assert torch.equal(staged[:valid_rows], offload[:valid_rows])
    assert torch.count_nonzero(staged[valid_rows:].float()).item() == 0
    assert torch.isfinite(staged.float()).all()
    # MANDATORY control: the multiply the select replaces does NOT clean it.
    multiplied = offload * valid_mask.view(batch, 1, 1, 1).to(torch.bfloat16)
    assert torch.isnan(multiplied[valid_rows:].float()).any()
