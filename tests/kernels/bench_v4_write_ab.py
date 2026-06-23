import torch

from batchgen.kv_cache.deepseek_v4_single_kv_pool import DeepSeekV4SingleKVPool
from batchgen_kernels.triton.v4_fused_compress_quant import (
    fused_kv_compress_norm_rope_insert_sparse_attn,
)
from tests.kernels.conftest import _bench

HEAD = 512
ROPE = 64
RATIO = 4
BLOCK = 64


def _cos_sin(max_pos):
    inv = 1.0 / (
        10000.0
        ** (torch.arange(0, ROPE, 2, device="cuda", dtype=torch.float32) / ROPE)
    )
    ang = torch.outer(
        torch.arange(max_pos, device="cuda", dtype=torch.float32), inv
    )
    return torch.cat((ang.cos(), ang.sin()), dim=-1)


def _eager_emit(kv_state, score_state, weight, cos_sin, chunk_pos):
    pooled = (kv_state.float() * torch.softmax(score_state.float(), dim=1)).sum(
        dim=1
    )
    var = pooled.square().mean(dim=-1, keepdim=True)
    pooled = pooled * torch.rsqrt(var + 1e-6) * weight.float()
    half = ROPE // 2
    cache = cos_sin.index_select(0, chunk_pos)
    cos = cache[:, :half]
    sin = cache[:, half:]
    rope = pooled[:, -ROPE:].view(-1, half, 2)
    e, o = rope[..., 0], rope[..., 1]
    pooled[:, -ROPE:] = torch.stack(
        (e * cos - o * sin, e * sin + o * cos), -1
    ).flatten(1)
    return pooled.to(torch.bfloat16)


def run(B):
    torch.manual_seed(0)
    weight = torch.randn(HEAD, device="cuda", dtype=torch.float32)
    cos_sin = _cos_sin(8192)

    kv_state = torch.randn(B, RATIO, HEAD, device="cuda", dtype=torch.float32)
    score_state = torch.randn(
        B, RATIO, HEAD, device="cuda", dtype=torch.float32
    )
    chunk_pos = (torch.arange(B, device="cuda", dtype=torch.int64) % 64) * RATIO

    pool = DeepSeekV4SingleKVPool(
        num_layers=1, num_pages=B + 8, page_size_tokens=BLOCK, device="cuda"
    )
    pool.initialize()
    token_slots = torch.arange(B, device="cuda", dtype=torch.int64)

    def eager_path():
        emitted = _eager_emit(kv_state, score_state, weight, cos_sin, chunk_pos)
        pool.store_kv(
            layer_idx=0, token_slots=token_slots, kv_processed=emitted
        )

    state_cache = torch.randn(
        B + 8, BLOCK, HEAD * 4, device="cuda", dtype=torch.float32
    )
    positions = (
        torch.arange(B, device="cuda", dtype=torch.int64) % 64
    ) * RATIO + (RATIO - 1)
    t2r = torch.zeros(B, device="cuda", dtype=torch.int32)
    slot_mapping = torch.arange(B, device="cuda", dtype=torch.int64)
    kv_slot_mapping = torch.arange(B, device="cuda", dtype=torch.int64)
    block_table = (
        torch.arange(B + 8, device="cuda", dtype=torch.int32)
        .view(1, -1)
        .repeat(B, 1)
    )
    k_cache = torch.zeros(
        B + 8, BLOCK * 576 + BLOCK * 8, device="cuda", dtype=torch.uint8
    )

    def fused_path():
        fused_kv_compress_norm_rope_insert_sparse_attn(
            state_cache,
            t2r,
            positions,
            slot_mapping,
            block_table,
            weight,
            cos_sin,
            k_cache,
            kv_slot_mapping,
            block_size=BLOCK,
            kv_cache_block_size=BLOCK,
            compress_ratio=RATIO,
            overlap=0,
        )

    e_ms = _bench(eager_path, warmup=10, iters=50)
    f_ms = _bench(fused_path, warmup=10, iters=50)
    return e_ms, f_ms


print(f"{'B':>6} {'eager_ms':>10} {'fused_ms':>10} {'speedup':>9}")
for B in (1, 8, 32, 128, 256, 512):
    e, f = run(B)
    print(f"{B:>6} {e:>10.4f} {f:>10.4f} {e / f:>8.2f}x")
