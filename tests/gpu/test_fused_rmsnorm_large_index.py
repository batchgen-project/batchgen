import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_rmsnorm_offsets_beyond_int32():
    from batchgen.attention.fused_kernels import cuda_add_rmsnorm, cuda_rmsnorm

    hidden = 6144
    crossing_row = (1 << 31) // hidden + 1
    rows = crossing_row + 1
    if torch.cuda.mem_get_info()[0] < 20 * (1 << 30):
        pytest.skip("requires 20 GiB free HBM for the large-index boundary")

    x = torch.ones((rows, hidden), device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(hidden, device="cuda", dtype=torch.bfloat16)
    normalized = cuda_rmsnorm(x, weight)
    torch.cuda.synchronize()
    for row in (0, crossing_row):
        torch.testing.assert_close(normalized[row, :4], torch.ones_like(normalized[row, :4]))

    del normalized
    residual = torch.ones_like(x)
    normalized, updated_residual = cuda_add_rmsnorm(residual, x, weight)
    torch.cuda.synchronize()
    for row in (0, crossing_row):
        torch.testing.assert_close(normalized[row, :4], torch.ones_like(normalized[row, :4]))
        torch.testing.assert_close(
            updated_residual[row, :4],
            torch.full_like(updated_residual[row, :4], 2),
        )
