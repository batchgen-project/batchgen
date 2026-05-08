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


def test_act_quant_valid_tokens_zeroes_padding_rows():
    from batchgen.attention.mla.fa3_backend import act_quant

    x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    x[2:].fill_(123)
    num_valid = torch.tensor([2], device="cuda", dtype=torch.int32)

    y, scale = act_quant(
        x.contiguous(),
        num_valid_tokens=num_valid,
        scale_tma_aligned=True,
    )
    torch.cuda.synchronize()

    assert scale.stride(0) == 1
    assert torch.count_nonzero(y[2:].float()).item() == 0
    assert torch.all(scale[2:] == 1e-12)


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
