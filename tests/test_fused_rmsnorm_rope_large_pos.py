"""Targeted correctness + OOB test for
`batchgen_kernels.triton.fused_rmsnorm_rope.fused_rmsnorm_rope_with_q_native`.

The kernel is the last Triton op on GLM-5's MLA DSA decode path before the
first H2D sync in the decoder layer, so any illegal-memory-access bug it has
surfaces at the NEXT `torch.tensor(..., device=cuda)` call. L4 (with
LongBench prompts) is the first workload that feeds it position_ids up to
~100_000+ — L2 capped at ~5000.

Tests:
    1. Correctness vs a pure PyTorch reference at L2-scale position_ids.
    2. Correctness vs ref at L4-scale position_ids (ranging up to 200000).
    3. Stress: per-batch pos_id values spanning small..huge in one call
       (replicates the L4 first-decode case where different rows have very
       different cache_seqlens).
    4. Boundary: pos_id = max_seq_len_cached - 1 (valid) vs
       pos_id = max_seq_len_cached (one past end, should fail cleanly).
    5. Dtype probes: position_ids as int32 vs int64.

Fail modes: the kernel may OOB-read cos/sin if pos_id is computed via
truncating int32 arithmetic inside Triton but the tensor is int64; it may
also silently return garbage if cos/sin cache wasn't extended to cover
max(position_ids).

Run on H20 (SM90a).
"""
import argparse
import sys

import torch


def _build_cos_sin_cache(max_seq: int, head_dim: int, device):
    """Replicates Glm5RotaryEmbedding._set_cos_sin_cache exactly (FP32)."""
    base = 1_000_000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(max_seq, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)   # [max_seq, head_dim]
    return emb.cos(), emb.sin()


def _apply_rope_native_interleaved_ref(q_pe_bf16, k_pe_bf16, cos, sin, pos_ids):
    """PyTorch reference: native-interleaved RoPE matching kernel semantics.

    q_pe_bf16: [bsz, num_heads, 1, head_dim] bf16
    k_pe_bf16: [bsz, 1, head_dim] bf16  (the k-pe slice from new_compressed_kv)
    cos, sin:  [max_seq, head_dim] fp32
    pos_ids:   [bsz, 1] int
    """
    bsz, num_heads, _, head_dim = q_pe_bf16.shape
    half = head_dim // 2
    # Gather per-seq cos/sin — use first half only (kernel semantics: cached is
    # cat((freqs, freqs)), so first & second halves are equal; kernel loads
    # only from the first half as cos_i / sin_i of length half_dim).
    cs = cos[pos_ids[:, 0]]        # [bsz, head_dim]
    sn = sin[pos_ids[:, 0]]
    cs_i = cs[:, :half]             # [bsz, half]
    sn_i = sn[:, :half]

    # Rotate k (native interleaved: stored at (2i, 2i+1))
    k_even = k_pe_bf16[:, :, 0::2].to(torch.float32)  # [bsz, 1, half]
    k_odd  = k_pe_bf16[:, :, 1::2].to(torch.float32)
    cs_e = cs_i.unsqueeze(1)        # [bsz, 1, half]
    sn_e = sn_i.unsqueeze(1)
    k_rot_even = k_even * cs_e - k_odd * sn_e
    k_rot_odd  = k_even * sn_e + k_odd * cs_e
    k_out = torch.empty_like(k_pe_bf16)
    k_out[:, :, 0::2] = k_rot_even.to(k_pe_bf16.dtype)
    k_out[:, :, 1::2] = k_rot_odd.to(k_pe_bf16.dtype)

    # Rotate q (all heads use the same cos/sin for that pos)
    q_even = q_pe_bf16[:, :, :, 0::2].to(torch.float32)  # [bsz, num_heads, 1, half]
    q_odd  = q_pe_bf16[:, :, :, 1::2].to(torch.float32)
    cs_q = cs_i.unsqueeze(1).unsqueeze(1)   # [bsz, 1, 1, half]
    sn_q = sn_i.unsqueeze(1).unsqueeze(1)
    q_rot_even = q_even * cs_q - q_odd * sn_q
    q_rot_odd  = q_even * sn_q + q_odd * cs_q
    q_out = torch.empty_like(q_pe_bf16)
    q_out[:, :, :, 0::2] = q_rot_even.to(q_pe_bf16.dtype)
    q_out[:, :, :, 1::2] = q_rot_odd.to(q_pe_bf16.dtype)
    return q_out, k_out


def _rmsnorm_ref(x_bf16, weight_bf16, eps: float):
    x_f = x_bf16.to(torch.float32)
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    normed = x_f * torch.rsqrt(var + eps)
    return (normed * weight_bf16.to(torch.float32).unsqueeze(0).unsqueeze(0)).to(x_bf16.dtype)


def run_case(bsz: int, pos_ids_list, cache_seq_len: int, dtype_pos=torch.int64,
             num_heads: int = 128, kv_lora_rank: int = 512, qk_rope_head_dim: int = 64,
             rms_eps: float = 1e-5, tag: str = "") -> dict:
    from batchgen_kernels.triton.fused_rmsnorm_rope import fused_rmsnorm_rope_with_q_native
    device = torch.device("cuda")
    total_dim = kv_lora_rank + qk_rope_head_dim
    # Inputs
    torch.manual_seed(0)
    new_compressed_kv = torch.randn(bsz, 1, total_dim, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(bsz, num_heads, 1, qk_rope_head_dim, dtype=torch.bfloat16, device=device)
    # Keep a pristine copy for the reference (kernel mutates q_pe in place).
    q_pe_ref_in = q_pe.clone()
    k_pe_ref_in = new_compressed_kv[:, :, kv_lora_rank:].clone()   # [bsz, 1, rope_dim]
    norm_weight = torch.randn(kv_lora_rank, dtype=torch.bfloat16, device=device)

    # Position ids
    pos = torch.tensor(pos_ids_list, dtype=dtype_pos, device=device).view(bsz, 1)

    # cos/sin cache: build covering all pos_ids
    cos, sin = _build_cos_sin_cache(cache_seq_len, qk_rope_head_dim, device)

    # Kernel call — this is what the production code does.
    try:
        out_kernel = fused_rmsnorm_rope_with_q_native(
            new_compressed_kv, q_pe, cos, sin, pos,
            norm_weight, kv_lora_rank, qk_rope_head_dim, eps=rms_eps,
        )
        # Force a sync to catch async CUDA errors here instead of silently
        # corrupting downstream.
        torch.cuda.synchronize(device)
        kernel_ok = True
        kernel_err = None
    except Exception as e:
        out_kernel = None
        kernel_ok = False
        kernel_err = f"{type(e).__name__}: {e}"

    # Reference
    q_rotated_ref, k_rotated_ref = _apply_rope_native_interleaved_ref(
        q_pe_ref_in, k_pe_ref_in, cos, sin, pos
    )
    kv_ref = new_compressed_kv.clone()
    # Replace lora slice with RMSNormed
    kv_ref[:, :, :kv_lora_rank] = _rmsnorm_ref(
        kv_ref[:, :, :kv_lora_rank], norm_weight, rms_eps
    )
    kv_ref[:, :, kv_lora_rank:] = k_rotated_ref

    # Diff vs reference (only if kernel succeeded).
    result = {
        "tag": tag, "bsz": bsz, "max_pos": int(max(pos_ids_list)),
        "cache_seq_len": cache_seq_len, "pos_dtype": str(dtype_pos),
        "kernel_ok": kernel_ok, "kernel_err": kernel_err,
    }
    if kernel_ok:
        # KV output
        kv_diff = (out_kernel.to(torch.float32) - kv_ref.to(torch.float32)).abs()
        result["kv_max_abs_diff"] = float(kv_diff.max().item())
        result["kv_mean_abs_diff"] = float(kv_diff.mean().item())
        # Q output (mutated in place)
        q_diff = (q_pe.to(torch.float32) - q_rotated_ref.to(torch.float32)).abs()
        result["q_max_abs_diff"] = float(q_diff.max().item())
        result["q_mean_abs_diff"] = float(q_diff.mean().item())
        result["kv_has_nan"] = bool(torch.isnan(out_kernel).any().item())
        result["q_has_nan"] = bool(torch.isnan(q_pe).any().item())
    return result


def _print_row(r: dict):
    if r["kernel_ok"]:
        print(f"  {r['tag']:<30}  bsz={r['bsz']:>4}  max_pos={r['max_pos']:>7}  "
              f"cache={r['cache_seq_len']:>7}  kv_max_d={r['kv_max_abs_diff']:.3e}  "
              f"q_max_d={r['q_max_abs_diff']:.3e}  "
              f"{'NaN!' if (r['kv_has_nan'] or r['q_has_nan']) else 'ok'}")
    else:
        print(f"  {r['tag']:<30}  bsz={r['bsz']:>4}  max_pos={r['max_pos']:>7}  "
              f"cache={r['cache_seq_len']:>7}  KERNEL-FAIL: {r['kernel_err']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("CUDA not available"); return 1
    device = torch.device("cuda")
    print(f"# Device: {torch.cuda.get_device_name(device)} (cap {torch.cuda.get_device_capability(device)})")

    fail = 0
    print("\n=== 1. L2-scale (short positions) — sanity ===")
    # Small positions, matches every L2 decode step we've shipped.
    for tag, bsz, maxpos in [
        ("L2-tiny",       8,    100),
        ("L2-decode-128", 128,  2000),
        ("L2-decode-256", 256,  4095),
    ]:
        cache = maxpos + 1
        pos = [i * maxpos // max(bsz - 1, 1) for i in range(bsz)]
        r = run_case(bsz, pos, cache, tag=tag)
        _print_row(r)
        if not r["kernel_ok"] or r.get("kv_max_abs_diff", 0) > 1e-1 or r.get("q_max_abs_diff", 0) > 1e-1:
            fail += 1

    print("\n=== 2. L4-scale (LongBench-sized positions) ===")
    # Positions typical of LongBench cache_seqlens = prompt_length on first decode step.
    for tag, bsz, maxpos in [
        ("LongBench-10k",  128, 10_000),
        ("LongBench-50k",  128, 50_000),
        ("LongBench-100k", 128, 100_000),
        ("LongBench-150k", 128, 150_000),
        ("LongBench-200k", 128, 200_000),
        ("rank2-size",     113, 180_000),   # match L4 rank-2 dimensions
    ]:
        cache = maxpos + 1
        pos = [i * maxpos // max(bsz - 1, 1) for i in range(bsz)]
        r = run_case(bsz, pos, cache, tag=tag)
        _print_row(r)
        if not r["kernel_ok"] or r.get("kv_max_abs_diff", 0) > 1.0 or r.get("q_max_abs_diff", 0) > 1.0:
            fail += 1

    print("\n=== 3. Mixed-length positions in one call (L4 reality) ===")
    # L4 first-decode: some seqs are MMLU (tiny cache_seqlen ~1000), some are
    # LongBench (huge cache_seqlen ~100k). Kernel must handle both in one batch.
    import random
    random.seed(0)
    mixed_pos = [
        random.choice([random.randint(100, 5000), random.randint(50_000, 150_000)])
        for _ in range(113)
    ]
    maxpos = max(mixed_pos)
    r = run_case(113, mixed_pos, maxpos + 1, tag="mixed-L4")
    _print_row(r)
    if not r["kernel_ok"] or r.get("kv_max_abs_diff", 0) > 1.0 or r.get("q_max_abs_diff", 0) > 1.0:
        fail += 1

    print("\n=== 4. pos_id dtype probe (int32 vs int64) ===")
    # If kernel assumes one but user passes other, can get wrong pos_id via
    # truncation. GLM-5 production passes int64 (worker.py:8647).
    for dt, label in [(torch.int64, "int64-prod"), (torch.int32, "int32-legacy")]:
        # Large pos to make dtype truncation visible (if any).
        maxpos = 150_000
        bsz = 64
        pos = [i * maxpos // (bsz - 1) for i in range(bsz)]
        r = run_case(bsz, pos, maxpos + 1, dtype_pos=dt, tag=f"dtype-{label}")
        _print_row(r)
        if not r["kernel_ok"]:
            fail += 1

    print("\n=== 5. Cache-bound edge: pos_id == cache_len - 1 ===")
    # Exact last valid position — kernel should load cos[last] correctly.
    for cache_len in [1024, 8192, 65536, 131072, 200000]:
        bsz = 8
        pos = [cache_len - 1] * bsz
        r = run_case(bsz, pos, cache_len, tag=f"cache-last-{cache_len}")
        _print_row(r)
        if not r["kernel_ok"] or r.get("kv_max_abs_diff", 0) > 1.0:
            fail += 1

    print("\n=== 6. Stress at prefill-scale (larger bsz) ===")
    # Prefill microbatches on LongBench can have bsz up to thousands; decode
    # is smaller but let's exercise larger bsz to rule out bsz-dep issues.
    for tag, bsz, maxpos in [
        ("prefill-bsz-512",  512,  80_000),
        ("prefill-bsz-1024", 1024, 80_000),
        ("prefill-bsz-2048", 2048, 80_000),
    ]:
        pos = [i * maxpos // (bsz - 1) for i in range(bsz)]
        r = run_case(bsz, pos, maxpos + 1, tag=tag)
        _print_row(r)
        if not r["kernel_ok"]:
            fail += 1

    print()
    print(f"# {'ALL GREEN' if fail == 0 else f'{fail} FAIL'}")
    return fail


if __name__ == "__main__":
    sys.exit(main())
