"""Fused masked row gathers vs the index_select formulation (CUDA only)."""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows,src_rows,hidden", [(8, 8, 7168), (32, 40, 7168), (256, 300, 6144), (5, 9, 1000)])
def test_row_gathers_match_index_select(rows, src_rows, hidden):
    from batchgen.models.moonshotai.kimi_linear.row_gather_triton import (
        add_gathered_rows_masked, gather_rows_masked, row_gather_triton_available)
    if not row_gather_triton_available():
        pytest.skip("triton unavailable")
    torch.manual_seed(0)
    src = torch.randn(src_rows, hidden, device="cuda", dtype=torch.bfloat16)
    idx = torch.randint(0, src_rows, (rows,), device="cuda", dtype=torch.int64)
    valid = (torch.rand(rows, device="cuda") > 0.3)
    valid_i32 = valid.to(torch.int32)
    ref = src.index_select(0, idx) * valid.to(src.dtype).unsqueeze(-1)
    got = gather_rows_masked(src, idx, valid_i32)
    assert torch.equal(got.float().abs(), ref.float().abs())      # signed zeros may differ
    residual = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    ref2 = residual + ref
    got2 = add_gathered_rows_masked(residual, src, idx, valid_i32)
    assert torch.equal(got2, ref2)
