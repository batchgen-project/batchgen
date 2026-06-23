from __future__ import annotations

import os
import statistics

os.environ.setdefault("BATCHGEN_DECODE_TIMING", "1")

import torch

from batchgen.attention.dsa.v4_flashmla_adapter import (
    DeepSeekV4FlashMLADecodeAdapter,
    build_v4_decode_attn_metadata,
)
from batchgen.attention.v4_backend import (
    DeepseekV4AttnBackend,
    build_layer_configs_from_compress_ratios,
)
from batchgen.kv_cache.deepseek_v4_kv_coordinator import DeepSeekV4KVCoordinator
from batchgen.timing import get_decode_timer, init_decode_timer
from batchgen_kernels.attention.v4_compressor import DeepSeekV4Compressor


def _make_rope_cache(max_pos, rope_dim=64, base=10000.0):
    device = torch.device("cuda")
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32)
            / rope_dim
        )
    )
    pos = torch.arange(max_pos, device=device, dtype=torch.float32)
    ang = torch.outer(pos, inv_freq)
    return torch.cat((ang.cos(), ang.sin()), dim=-1)


def trace(seq_len=512, warmup=64):
    device = torch.device("cuda")
    num_heads = 64
    head_dim = 512
    layer_idx = 2
    compress_ratios = [0, 4, 128]
    sequence_ids = [31337]
    softmax_scale = head_dim**-0.5

    torch.manual_seed(0)
    hidden_states = (
        torch.randn(seq_len, head_dim, dtype=torch.float32, device=device)
        .div_(10)
        .clamp_(-1, 1)
    )
    kv_tokens = (
        torch.randn(seq_len, head_dim, dtype=torch.bfloat16, device=device)
        .div_(10)
        .clamp_(-1, 1)
    )
    q_tokens = torch.randn(
        seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    q_tokens = (
        q_tokens
        * torch.rsqrt(q_tokens.square().mean(dim=-1, keepdim=True) + 1e-6)
    ).clamp_(-1, 1)
    rope_cache = _make_rope_cache(seq_len + 4)
    attn_sink = torch.zeros(num_heads, dtype=torch.float32, device=device)

    compressor = DeepSeekV4Compressor(
        head_dim, head_dim, 64, 128, 1e-6, overlap=False
    ).to(device)

    coordinator = DeepSeekV4KVCoordinator(
        compress_ratios=compress_ratios,
        num_pages=max(256, seq_len + 16),
        device=device,
        base_page_size=max(256, seq_len),
    )
    coordinator.initialize()

    init_decode_timer(
        "v4-trace", ["attn_q_rope", "attn_kv_store", "attn_kv_fetch"]
    )
    dt = get_decode_timer()

    try:
        coordinator.allocate_pages_for_sequences(sequence_ids, [seq_len])
        page_tables = coordinator.rebuild_page_table(sequence_ids)
        layer_config = build_layer_configs_from_compress_ratios(
            compress_ratios=compress_ratios,
            n_heads=num_heads,
            head_dim=head_dim,
            rope_head_dim=64,
        )[layer_idx]
        backend = DeepseekV4AttnBackend(
            layer_configs=[layer_config],
            page_size=coordinator.swa.page_size_tokens,
            flashmla_backend=DeepSeekV4FlashMLADecodeAdapter(coordinator),
        )

        step_ms = []
        for step in range(seq_len):
            metadata = build_v4_decode_attn_metadata(
                coordinator=coordinator,
                sequence_ids=sequence_ids,
                cache_seqlens=torch.tensor(
                    [step + 1], dtype=torch.int32, device=device
                ),
                positions=torch.tensor(
                    [step], dtype=torch.int32, device=device
                ),
                page_tables=page_tables,
                rope_cache=rope_cache,
            )
            backend.init_metadata(metadata)

            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            backend.forward(
                layer_config=layer_config,
                q=q_tokens[step : step + 1],
                kv=kv_tokens[step : step + 1],
                attn_sink=attn_sink,
                softmax_scale=softmax_scale,
                compressor=compressor,
                compress_hidden_states=hidden_states[step : step + 1],
            )
            e.record()
            torch.cuda.synchronize()
            if dt:
                dt.step_done()
            if step >= warmup:
                step_ms.append(s.elapsed_time(e))

        return step_ms, dt
    finally:
        coordinator.destroy()


def main():
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"device={name} sm{cap[0]}{cap[1]}")

    step_ms, dt = trace()
    step_ms.sort()
    n = len(step_ms)
    mean = statistics.mean(step_ms)
    p50 = step_ms[n // 2]
    p90 = step_ms[min(n - 1, int(n * 0.90))]
    p99 = step_ms[min(n - 1, int(n * 0.99))]
    print(
        f"full decode step (c4+c128 layer, B=1): "
        f"mean={mean:.4f} p50={p50:.4f} p90={p90:.4f} p99={p99:.4f} ms "
        f"(n={n} steps)"
    )

    if dt:
        stats = dt._aggregate_by_op()
        print(
            f"\n{'op':<22s} {'count':>6s} {'mean_us':>9s} {'p50_us':>9s} {'p99_us':>9s} {'total_ms':>9s} {'pct':>6s}"
        )
        for op, s in sorted(stats.items(), key=lambda kv: -kv[1]["total_ms"]):
            print(
                f"{op:<22s} {int(s['count']):>6d} {s['mean_ms']*1000:>9.1f} "
                f"{s['p50_ms']*1000:>9.1f} {s['p99_ms']*1000:>9.1f} "
                f"{s['total_ms']:>9.2f} {s['pct']:>5.1f}%"
            )

    from batchgen.attention.dsa.v4_mla_sm120_triton import (
        _tiled_sparse_decode_kernel,
    )

    cache = getattr(_tiled_sparse_decode_kernel, "cache", None)
    if isinstance(cache, dict):
        print(f"\nautotune distinct keys (topk_rounded buckets): {len(cache)}")

    if dt:
        main = [
            (ev.step_idx, ev.elapsed_ms * 1000)
            for ev in dt._records
            if ev.op_name == "attn_sm120_main" and ev.elapsed_ms >= 0
        ]
        main.sort(key=lambda x: x[0])
        slow = [(s, us) for s, us in main if us > 500]
        print(f"\nattn_sm120_main: {len(main)} steps, {len(slow)} steps >500us")
        print("first 12 steps (step:us):")
        print("  " + " ".join(f"{s}:{us:.0f}" for s, us in main[:12]))
        print("slow steps (step:us), first 20:")
        print("  " + " ".join(f"{s}:{us:.0f}" for s, us in slow[:20]))
        if slow:
            slow_steps = [s for s, _ in slow]
            deltas = [
                slow_steps[i + 1] - slow_steps[i]
                for i in range(len(slow_steps) - 1)
            ]
            print(f"slow-step gaps (stride between slow steps): {deltas[:20]}")
        for op, s in sorted(stats.items(), key=lambda kv: -kv[1]["total_ms"]):
            print(
                f"{op:<22s} {int(s['count']):>6d} {s['mean_ms']*1000:>9.1f} "
                f"{s['total_ms']:>9.2f} {s['pct']:>5.1f}%"
            )


if __name__ == "__main__":
    main()
