"""Single-GPU gates for the K3 prefill dequant-once routed path.

1. ``dequant_marlin_bf16`` reproduces the raw-layout dequant bit-exactly from
   the Marlin-order tensors the production shard holds.
2. ``K3PrefillDequantOnce.expert_path`` matches the compact
   ``_expert_path`` (M16 Marlin) contract and numerics: same owned rows,
   same combine within BF16 tolerance, on random routing with real dims.
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

FP4_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
N, K = 3072, 3584  # K3 routed expert: moe_intermediate, latent


def _dequant_raw(packed, scale):
    n = packed.shape[0]
    lut = torch.tensor(FP4_LUT, device=packed.device)
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(n, -1).long()
    exp = scale.to(torch.int32) - 127
    scl = torch.ldexp(torch.ones_like(exp, dtype=torch.float32), exp)
    return (lut[codes] * scl.repeat_interleave(32, dim=1)).to(torch.bfloat16)


def _random_expert(n_out, k_in, gen, device):
    packed = torch.randint(0, 256, (n_out, k_in // 2), dtype=torch.uint8, device=device, generator=gen)
    scale = torch.randint(118, 132, (n_out, k_in // 32), dtype=torch.uint8, device=device, generator=gen)
    return packed, scale


def test_dequant_from_marlin_order_is_bit_exact():
    from batchgen.models.moonshotai.kimi_linear.k3.mxfp4_expert import repack_mxfp4_to_marlin_device
    from batchgen.moe.k3_prefill_dequant_once import dequant_marlin_bf16
    dev = torch.device("cuda", 0)
    gen = torch.Generator(device=dev).manual_seed(11)
    for n_out, k_in in ((N, K), (K, N)):  # w1/w3 and w2 shapes
        packed, scale = _random_expert(n_out, k_in, gen, dev)
        qw, s = repack_mxfp4_to_marlin_device(packed, scale, k_in, n_out, scale_bf16=False)
        out = torch.empty(n_out, k_in, dtype=torch.bfloat16, device=dev)
        dequant_marlin_bf16(qw, s, out)
        ref = _dequant_raw(packed, scale)
        assert torch.equal(out, ref), f"{(out != ref).sum().item()} mismatches at {n_out}x{k_in}"


def test_expert_path_matches_marlin_compact_path():
    from batchgen.moe.fused_moe_mxfp4_resident import (
        ResidentEPMXFP4MoELayer, build_layer_shard, compact_dispatch_route_stats_by_chunk,
    )
    from batchgen.moe.k3_prefill_dequant_once import K3PrefillDequantOnce
    dev = torch.device("cuda", 0)
    gen = torch.Generator(device=dev).manual_seed(5)
    E, E_global, rows, top_k = 6, 24, 1500, 16
    raw = [{"w1": _random_expert(N, K, gen, dev), "w3": _random_expert(N, K, gen, dev),
            "w2": _random_expert(K, N, gen, dev)} for _ in range(E)]
    shard = build_layer_shard(raw, dev)
    helper = ResidentEPMXFP4MoELayer(0, shard, None, None, None, world_size=1, expert_start=8)
    helper.compact_dispatch = True
    x = torch.randn(rows, K, device=dev, dtype=torch.bfloat16, generator=gen) * 0.5
    topk = torch.randint(0, E_global, (rows, top_k), dtype=torch.int32, device=dev, generator=gen)
    weight = torch.rand(rows, top_k, device=dev, dtype=torch.float32, generator=gen)
    packed_max_rows, packed_capacity = compact_dispatch_route_stats_by_chunk(topk, 8, E, rows)[0]

    ref_out, ref_pos = helper._expert_path(x, topk, rows, packed_capacity=packed_capacity, packed_max_rows=packed_max_rows)
    dq = K3PrefillDequantOnce(shard, dev)
    out, pos = dq.expert_path(x, topk, rows, 8, packed_capacity=packed_capacity)

    owned = ref_pos >= 0
    assert torch.equal(owned, pos >= 0), "ownership masks differ"
    a = ref_out[ref_pos[owned].long()].float()
    b = out[pos[owned].long()].float()
    tol = 1e-5 + 1.6e-2 * a.abs()
    frac = ((a - b).abs() > tol).float().mean().item()
    assert frac < 1e-3, f"{frac:.2e} of owned outputs outside BF16 tolerance"
    ca = helper._combine_fp32(ref_out, ref_pos, weight, rows, K, top_k)
    cb = helper._combine_fp32(out, pos, weight, rows, K, top_k)
    ctol = 1e-4 + 1.6e-2 * ca.abs()
    cfrac = ((ca - cb).abs() > ctol).float().mean().item()
    assert cfrac < 1e-3, f"combine: {cfrac:.2e} outside tolerance"

    # streamed-SP8 shard shape (StreamedSP8LayerBuffer._make_shard): stacked
    # Marlin-order views instead of per-expert dicts -> identical staging
    from types import SimpleNamespace
    stacked = SimpleNamespace(
        num_local=E, N=N, K_latent=K,
        marlin_packed={p: torch.stack([t[p][0] for t in shard._tensors]) for p in ("w1", "w3", "w2")},
        marlin_scales={p: torch.stack([t[p][1] for t in shard._tensors]) for p in ("w1", "w3", "w2")},
    )
    dq2 = K3PrefillDequantOnce(stacked, dev)
    assert torch.equal(dq2.w_gu, dq.w_gu) and torch.equal(dq2.w_d, dq.w_d), "stacked-shard staging differs"
