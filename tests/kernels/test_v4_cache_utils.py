# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
# ---------------------------------------------------------------------------- #

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _make_cache(
    num_blocks: int, block_size: int, device: str = "cuda"
) -> torch.Tensor:
    from batchgen_kernels.triton.v4_cache_utils import SCALE_DIM, TOKEN_DATA_SIZE

    return torch.zeros(
        (num_blocks, block_size * (TOKEN_DATA_SIZE + SCALE_DIM)),
        dtype=torch.uint8,
        device=device,
    )


def _slot_mapping_from_block_table(
    block_table: torch.Tensor,
    token_positions: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    logical_block = torch.div(
        token_positions, block_size, rounding_mode="floor"
    )
    block_offset = token_positions.remainder(block_size)
    physical_block = block_table[
        token_to_req_indices.long(),
        logical_block.long(),
    ]
    return physical_block.long() * block_size + block_offset.long()


def _reference_encode_scale(absmax: float) -> int:
    if absmax == 0.0:
        return 0
    return max(0, int(math.ceil(math.log2(absmax / 448.0)) + 127))


def _reference_quantize_and_insert(
    k_bf16: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    from batchgen_kernels.triton.v4_cache_utils import (
        FP8_MAX,
        NOPE_DIM,
        QUANT_BLOCK_SIZE,
        ROPE_DIM,
        SCALE_DIM,
        TOKEN_DATA_SIZE,
    )

    out = cache.clone()
    for token_idx in range(slot_mapping.numel()):
        slot = int(slot_mapping[token_idx].item())
        if slot < 0:
            continue
        block_idx = slot // block_size
        pos_in_block = slot % block_size
        data_base = pos_in_block * TOKEN_DATA_SIZE
        scale_base = block_size * TOKEN_DATA_SIZE + pos_in_block * SCALE_DIM
        token = k_bf16[token_idx].float()
        fp8_part = token[:NOPE_DIM].view(-1, QUANT_BLOCK_SIZE)
        rope_part = k_bf16[
            token_idx, NOPE_DIM : NOPE_DIM + ROPE_DIM
        ].contiguous()
        for qblock_idx in range(NOPE_DIM // QUANT_BLOCK_SIZE):
            block = fp8_part[qblock_idx]
            absmax = float(block.abs().amax().item())
            if absmax == 0.0:
                scale = 1.0
                encoded = 0
            else:
                exponent = math.ceil(math.log2(absmax / FP8_MAX))
                scale = 2.0**exponent
                encoded = int(exponent + 127)
            q = torch.clamp(block / scale, -FP8_MAX, FP8_MAX).to(
                torch.float8_e4m3fn
            )
            start = data_base + qblock_idx * QUANT_BLOCK_SIZE
            out[block_idx, start : start + QUANT_BLOCK_SIZE] = q.view(
                torch.uint8
            )
            out[block_idx, scale_base + qblock_idx] = encoded
        out[block_idx, scale_base + (NOPE_DIM // QUANT_BLOCK_SIZE)] = 0
        rope_bytes = rope_part.view(torch.uint8)
        rope_start = data_base + NOPE_DIM
        out[block_idx, rope_start : rope_start + rope_bytes.numel()] = (
            rope_bytes
        )
    return out


def test_quant_roundtrip():
    from batchgen_kernels.triton.v4_cache_utils import (
        dequantize_and_gather_k,
        quantize_and_insert_k,
    )

    for T in (1, 32, 128):
        torch.manual_seed(T)
        block_size = 64
        num_blocks = max(1, (T + block_size - 1) // block_size)
        cache = _make_cache(num_blocks, block_size)
        k = torch.randn(T, 512, dtype=torch.bfloat16, device="cuda")
        slot_mapping = torch.arange(T, device="cuda", dtype=torch.int64)
        quantize_and_insert_k(
            k, cache, slot_mapping=slot_mapping, block_size=block_size
        )
        restored = dequantize_and_gather_k(
            cache, slot_mapping, block_size=block_size
        )
        torch.testing.assert_close(
            restored.float(), k.float(), atol=0.05, rtol=0.05
        )


def test_ue8m0_scale_encoding():
    from batchgen_kernels.triton.v4_cache_utils import (
        QUANT_BLOCK_SIZE,
        TOKEN_DATA_SIZE,
        quantize_and_insert_k,
    )

    block_size = 1
    cache = _make_cache(1, block_size)
    k = torch.zeros(1, 512, dtype=torch.bfloat16, device="cuda")
    absmax_values = [448.0, 512.0, 224.0, 896.0, 112.0, 56.0, 28.0]
    for i, absmax in enumerate(absmax_values):
        start = i * QUANT_BLOCK_SIZE
        k[0, start] = absmax
    quantize_and_insert_k(
        k,
        cache,
        slot_mapping=torch.zeros(1, device="cuda", dtype=torch.int64),
        block_size=block_size,
    )
    scales = cache[0, TOKEN_DATA_SIZE : TOKEN_DATA_SIZE + 8].cpu().tolist()
    expected = [_reference_encode_scale(v) for v in absmax_values] + [0]
    assert scales == expected


def test_paged_insert_block_table():
    from batchgen_kernels.triton.v4_cache_utils import (
        dequantize_and_gather_k,
        quantize_and_insert_k,
    )

    T = 32
    block_size = 8
    block_table = torch.tensor([[2, 0, 3, 1]], device="cuda", dtype=torch.int32)
    token_positions = torch.arange(T, device="cuda", dtype=torch.int64)
    token_to_req_indices = torch.zeros(T, device="cuda", dtype=torch.int32)
    slot_mapping = _slot_mapping_from_block_table(
        block_table, token_positions, token_to_req_indices, block_size
    )
    cache = _make_cache(4, block_size)
    torch.manual_seed(11)
    k = torch.randn(T, 512, dtype=torch.bfloat16, device="cuda")
    quantize_and_insert_k(
        k,
        cache,
        block_table=block_table,
        token_positions=token_positions,
        token_to_req_indices=token_to_req_indices,
        block_size=block_size,
    )
    probe = torch.tensor([16, 0, 24, 8], device="cuda", dtype=torch.int64)
    gathered = dequantize_and_gather_k(cache, probe, block_size=block_size)
    expected = k[torch.tensor([0, 8, 16, 24], device="cuda")]
    torch.testing.assert_close(
        gathered.float(), expected.float(), atol=0.05, rtol=0.05
    )
    assert torch.equal(slot_mapping[::8].cpu(), probe.cpu())


def test_gather_specific_tokens():
    from batchgen_kernels.triton.v4_cache_utils import (
        dequantize_and_gather_k,
        quantize_and_insert_k,
    )

    T = 32
    block_size = 8
    block_table = torch.tensor([[1, 3, 0, 2]], device="cuda", dtype=torch.int32)
    token_positions = torch.arange(T, device="cuda", dtype=torch.int64)
    token_to_req_indices = torch.zeros(T, device="cuda", dtype=torch.int32)
    slot_mapping = _slot_mapping_from_block_table(
        block_table, token_positions, token_to_req_indices, block_size
    )
    cache = _make_cache(4, block_size)
    torch.manual_seed(17)
    k = torch.randn(T, 512, dtype=torch.bfloat16, device="cuda")
    quantize_and_insert_k(
        k, cache, slot_mapping=slot_mapping, block_size=block_size
    )
    token_ids = torch.tensor(
        [0, 3, 7, 8, 14, 17, 23, 31], device="cuda", dtype=torch.int64
    )
    gathered = dequantize_and_gather_k(
        cache, slot_mapping[token_ids], block_size=block_size
    )
    torch.testing.assert_close(
        gathered.float(), k[token_ids].float(), atol=0.05, rtol=0.05
    )


def test_global_topk_index_mapping():
    from batchgen_kernels.triton.v4_cache_utils import (
        compute_global_topk_indices_and_lens,
    )

    topk_indices = torch.stack(
        (
            torch.arange(512, device="cuda", dtype=torch.int32),
            torch.arange(512, device="cuda", dtype=torch.int32),
        )
    )
    token_to_req_indices = torch.tensor(
        [0, 0], device="cuda", dtype=torch.int32
    )
    block_table = torch.tensor([[3, 1, 0, 2]], device="cuda", dtype=torch.int32)
    is_valid_token = torch.tensor([True, False], device="cuda")
    global_topk, lens = compute_global_topk_indices_and_lens(
        topk_indices,
        token_to_req_indices,
        block_table,
        128,
        is_valid_token,
    )
    expected = torch.cat(
        (
            torch.arange(384, 512, device="cuda", dtype=torch.int32),
            torch.arange(128, 256, device="cuda", dtype=torch.int32),
            torch.arange(0, 128, device="cuda", dtype=torch.int32),
            torch.arange(256, 384, device="cuda", dtype=torch.int32),
        )
    )
    assert torch.equal(global_topk[0], expected)
    assert lens.tolist() == [512, 0]


def test_combine_topk_swa_pad128():
    from batchgen_kernels.triton.v4_cache_utils import combine_topk_swa_indices

    topk = torch.arange(512, device="cuda", dtype=torch.int32).view(1, 512)
    swa = torch.arange(512, 640, device="cuda", dtype=torch.int32).view(1, 128)
    combined, lens = combine_topk_swa_indices(topk, swa)
    valid = combined[0, : lens[0]].cpu()
    assert combined.shape == (1, 640)
    assert int(lens[0].item()) == 640
    assert len(valid.unique()) == 640


def test_empty_topk():
    from batchgen_kernels.triton.v4_cache_utils import combine_topk_swa_indices

    topk = torch.empty(2, 0, device="cuda", dtype=torch.int32)
    swa = torch.tensor(
        [[0, 1, 2], [5, -1, -1]], device="cuda", dtype=torch.int32
    )
    combined, lens = combine_topk_swa_indices(topk, swa)
    assert lens.tolist() == [3, 1]
    assert torch.equal(
        combined[0, :3].cpu(), torch.tensor([0, 1, 2], dtype=torch.int32)
    )
    assert int(combined[1, 0].item()) == 5


def test_swa_exceeds_sequence():
    from batchgen_kernels.triton.v4_cache_utils import combine_topk_swa_indices

    topk = torch.empty(1, 0, device="cuda", dtype=torch.int32)
    swa = torch.full((1, 128), -1, device="cuda", dtype=torch.int32)
    swa[0, :64] = torch.arange(64, device="cuda", dtype=torch.int32)
    combined, lens = combine_topk_swa_indices(topk, swa)
    assert int(lens[0].item()) == 64
    assert torch.equal(
        combined[0, :64].cpu(), torch.arange(64, dtype=torch.int32)
    )


def test_topk_swa_overlap():
    from batchgen_kernels.triton.v4_cache_utils import combine_topk_swa_indices

    topk = torch.arange(512, device="cuda", dtype=torch.int32).view(1, 512)
    swa = torch.arange(480, 608, device="cuda", dtype=torch.int32).view(1, 128)
    combined, lens = combine_topk_swa_indices(topk, swa)
    valid = combined[0, : lens[0]]
    assert int(lens[0].item()) == 608
    assert int((valid == 480).sum().item()) == 1
    assert int((valid == 511).sum().item()) == 1
    assert int((valid == 607).sum().item()) == 1


def test_benchmark():
    from batchgen_kernels.triton.v4_cache_utils import quantize_and_insert_k
    from tests.kernels.conftest import _bench

    block_size = 64
    for T in (1, 32, 128, 1024):
        torch.manual_seed(T + 100)
        num_blocks = max(1, (T + block_size - 1) // block_size)
        k = torch.randn(T, 512, dtype=torch.bfloat16, device="cuda")
        slot_mapping = torch.arange(T, device="cuda", dtype=torch.int64)
        cache_triton = _make_cache(num_blocks, block_size)
        cache_ref = _make_cache(num_blocks, block_size)

        def triton_impl():
            quantize_and_insert_k(
                k,
                cache_triton,
                slot_mapping=slot_mapping,
                block_size=block_size,
            )

        def torch_impl():
            _reference_quantize_and_insert(
                k, cache_ref, slot_mapping, block_size
            )

        triton_ms = _bench(triton_impl, warmup=1, iters=5)
        torch_ms = _bench(torch_impl, warmup=1, iters=5)
        print(
            f"\ncache_utils T={T} triton={triton_ms:.3f} ms torch={torch_ms:.3f} ms"
        )
        assert triton_ms > 0
        assert torch_ms > 0
