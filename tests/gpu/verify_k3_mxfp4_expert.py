#!/usr/bin/env python3
"""Verification for the K3 MXFP4 routed-expert seam (kimi_linear/k3/mxfp4_expert).

STAGED — not part of the CPU workflow. Two independent checks:

  --meta  (no GPU)  Meta-device build of KimiLinearForCausalLM at the REAL K3
                    config, then name-set arithmetic between what each routed
                    expert declares (`named_parameters()`) and what the
                    parameter server serves for its module_key
                    (`build_k3_state_dict_name_map`). Reports the count of
                    unserved expert params; MUST be 0, in BOTH directions
                    (a served tensor with no parameter to land on is the same
                    silent-empty bug seen from the other side).
                    Depth/expert count are shrinkable so the build is seconds.

  --gpu   (GPU 1)   Single-expert numerical check at the real K3 shapes:
                      1. repack_mxfp4_to_marlin_device (device) must be
                         BIT-IDENTICAL to repack_mxfp4_to_marlin_gs32 (the
                         CPU function the frozen oracle gates), so the device
                         twin inherits that validation rather than asserting
                         its own correctness;
                      2. the wrapper's end-to-end expert output vs an
                         oracle-dequant + fp32 matmul + eager SiTU reference,
                         under the project gate from
                         tests/moe/gpu_parity_mxfp4_marlin.py
                         (tol = 1e-5 + 1.6e-2*|ref|, PASS iff finite and
                         fail_frac < 1e-4 and max_rel < 1.6e-2 on |ref| >
                         0.1*rms);
                      3. the hard-fail negatives: a dropped tensor, a wrong
                         dtype and a wrong shape must each RAISE.

Run (h20-instance-1, GPU 1 — GPU 0 belongs to the model workstream):
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate batchgen
    cd $WS/batchgen
    PYTHONPATH=$WS/fla-src python tests/gpu/verify_k3_mxfp4_expert.py --meta
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$WS/fla-src \
        python tests/gpu/verify_k3_mxfp4_expert.py --gpu
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
CONFIG_JSON = (ROOT / "batchgen" / "models" / "moonshotai" / "kimi_k3"
               / "assets" / "config.json")

K3_LATENT, K3_FFN = 3584, 3072      # routed_expert_hidden_size, moe_intermediate
SCALE_LO, SCALE_HI = 112, 122       # observed K3 E8M0 range (frozen verdict)

_results = []


def report(name, ok, detail=""):
    _results.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def gate(out, ref, name):
    """Project numerical gate (verbatim from tests/moe/gpu_parity_mxfp4_marlin.py)."""
    out = out.float()
    ref = ref.float()
    finite = bool(torch.isfinite(out).all())
    err = (out - ref).abs()
    tol = 1e-5 + 1.6e-2 * ref.abs()
    fail_frac = float((err > tol).float().mean())
    rms = float(ref.pow(2).mean().sqrt())
    mask = ref.abs() > 0.1 * rms
    max_rel = float((err[mask] / ref.abs()[mask]).max()) if mask.any() else 0.0
    passed = finite and fail_frac < 1e-4 and max_rel < 1.6e-2
    print(f"    {name}: {'gate-PASS' if passed else 'gate-FAIL'} "
          f"fail_frac={fail_frac:.3e} max_rel={max_rel:.3e} finite={finite}")
    return passed


def k3_config(num_layers=None, num_experts=None):
    from batchgen.models.moonshotai.kimi_linear.config import KimiLinearConfig

    cfg = KimiLinearConfig.from_json(str(CONFIG_JSON))
    if num_layers is not None:
        lac = dict(cfg.linear_attn_config)
        lac["kda_layers"] = [i for i in lac["kda_layers"] if i <= num_layers]
        lac["full_attn_layers"] = [i for i in lac["full_attn_layers"]
                                   if i <= num_layers]
        cfg.linear_attn_config = lac
        cfg.num_hidden_layers = int(num_layers)
    if num_experts is not None:
        cfg.n_routed_experts = int(num_experts)
    return cfg


# --------------------------------------------------------------------------- #
#  --meta : name-set arithmetic on a meta build                                #
# --------------------------------------------------------------------------- #

def check_meta(num_layers, num_experts):
    from batchgen.models.moonshotai.kimi_linear.k3.tensor_map import (
        build_k3_state_dict_name_map,
    )
    from batchgen.models.moonshotai.kimi_linear.model import KimiLinearForCausalLM

    cfg = k3_config(num_layers, num_experts)
    name_map, _ = build_k3_state_dict_name_map(cfg)

    served = {}
    for ckpt_name, entry in name_map.items():
        served.setdefault(entry["module_key"], set()).add(entry["tensor_key"])

    with torch.device("meta"):
        model = KimiLinearForCausalLM(cfg)

    n_experts_seen = 0
    unserved = 0          # declared by the module, never served -> empty(0)
    unlanded = 0          # served by the engine, no parameter -> dropped
    example = None
    for layer_idx in range(cfg.num_hidden_layers):
        moe = getattr(model.model.layers[layer_idx], "block_sparse_moe", None)
        if moe is None or moe.experts is None:
            continue
        for expert_idx in range(len(moe.experts)):
            module_key = f"routed_expert_{layer_idx}_{expert_idx}"
            have = served.get(module_key, set())
            want = {n for n, _ in moe.experts[expert_idx].named_parameters()}
            n_experts_seen += 1
            unserved += len(want - have)
            unlanded += len(have - want)
            if example is None:
                example = (module_key, sorted(want))

    print(f"    experts inspected: {n_experts_seen} "
          f"({cfg.num_hidden_layers} layers x {cfg.n_routed_experts})")
    print(f"    expert param names: {example[0]} -> {example[1]}")
    print(f"    unserved expert params (module declares, engine never sends): "
          f"{unserved}")
    print(f"    unlanded served tensors (engine sends, no parameter to hold): "
          f"{unlanded}")
    report("meta-build: unserved expert params == 0", unserved == 0,
           f"unserved={unserved}")
    report("meta-build: unlanded served tensors == 0", unlanded == 0,
           f"unlanded={unlanded}")


# --------------------------------------------------------------------------- #
#  --gpu : repack bit-identity, numerical parity, hard-fail negatives          #
# --------------------------------------------------------------------------- #

def rand_expert_weight(K, N, seed):
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(0, 256, (N, K // 2), generator=g,
                           dtype=torch.int16).to(torch.uint8)
    scale = torch.randint(SCALE_LO, SCALE_HI + 1, (N, K // 32), generator=g,
                          dtype=torch.int16).to(torch.uint8)
    return packed, scale


def check_repack_bit_identity(dev):
    from batchgen.moe.marlin_weight_prep import repack_mxfp4_to_marlin_gs32
    from batchgen.models.moonshotai.kimi_linear.k3.mxfp4_expert import (
        repack_mxfp4_to_marlin_device,
    )

    ok = True
    for tag, (K, N) in (("w1/w3 3584x3072", (K3_LATENT, K3_FFN)),
                        ("w2   3072x3584", (K3_FFN, K3_LATENT))):
        packed, scale = rand_expert_weight(K, N, seed=hash(tag) % 2**31)
        cpu_qw, cpu_s = repack_mxfp4_to_marlin_gs32(packed, scale, K, N,
                                                    emit_scale="bf16")
        dev_qw, dev_s = repack_mxfp4_to_marlin_device(
            packed.to(dev), scale.to(dev), K, N)
        same_qw = bool((dev_qw.cpu() == cpu_qw).all())
        same_s = bool((dev_s.cpu().view(torch.int16)
                       == cpu_s.view(torch.int16)).all())
        print(f"    {tag}: qw {tuple(dev_qw.shape)} {dev_qw.dtype} identical="
              f"{same_qw}; scale {tuple(dev_s.shape)} {dev_s.dtype} identical="
              f"{same_s}")
        ok = ok and same_qw and same_s
    report("device repack is BIT-IDENTICAL to the oracle-gated CPU repack", ok)


def situ_ref_fp32(gate_x, up_x):
    """Eager SiTU (kimi_k3/model.py::SituAndMul; beta=4, linear_beta=25)."""
    gate_x = gate_x.float()
    up_x = up_x.float()
    a = 4.0 * torch.tanh(gate_x / 4.0) * torch.sigmoid(gate_x)
    return a * (25.0 * torch.tanh(up_x / 25.0))


def build_wrapped_expert(dev, weights_cpu):
    """A K3MXFP4Expert behind its real wrapper, fed by a stub core engine.

    Exercises the production path end to end: load_weights -> dequantize_weights
    (slot validation) -> apply_weights (assert nothing dropped) ->
    micro_batch_forward -> free_weights.
    """
    from types import SimpleNamespace

    from batchgen.models.moonshotai.kimi_linear.k3.mxfp4_expert import (
        K3MXFP4Expert,
        KimiK3MXFP4ExpertWrapper,
    )

    class StubEngine:
        def __init__(self, served):
            self.served = served
            self.freed = []

        def get_weights(self, module_key, phase):
            return self.served

        def free_weights_buffer(self, module_key):
            self.freed.append(module_key)

    cfg = SimpleNamespace(activation_situ_beta=4.0,
                          activation_situ_linear_beta=25.0)
    module = K3MXFP4Expert(cfg, hidden_size=K3_LATENT,
                           intermediate_size=K3_FFN).to(dev)
    served = {k: v.to(dev).contiguous() for k, v in weights_cpu.items()}
    engine = StubEngine(served)
    engine_config = SimpleNamespace(
        Module_Batching_Config=SimpleNamespace(
            expert_prefill_batch_size_upper_bound=4096,
            expert_decoding_batch_size_upper_bound=2048,
        )
    )
    wrapper = KimiK3MXFP4ExpertWrapper(module, 4, 0, engine, engine_config,
                                       None, persistent=False)
    return wrapper, engine, served


def check_numerics(dev, num_tokens=64):
    from batchgen.moe.mxfp4_oracle_vector import mxfp4_dequantize_oracle

    torch.manual_seed(20260806)
    raw = {}
    for name, (K, N) in (("w1", (K3_LATENT, K3_FFN)),
                         ("w3", (K3_LATENT, K3_FFN)),
                         ("w2", (K3_FFN, K3_LATENT))):
        raw[name] = rand_expert_weight(K, N, seed=hash(name) % 2**31)

    weights_cpu = {}
    for name, (packed, scale) in raw.items():
        weights_cpu[f"{name}.weight_packed"] = packed
        weights_cpu[f"{name}.weight_scale"] = scale

    wrapper, engine, _ = build_wrapped_expert(dev, weights_cpu)
    x = (torch.randn(num_tokens, K3_LATENT, device=dev) * 0.5).to(torch.bfloat16)

    out = wrapper(x)

    # Reference: exact oracle dequant -> fp32 matmul -> eager SiTU -> fp32 down.
    ref_w = {n: mxfp4_dequantize_oracle(p, s).to(dev).float()
             for n, (p, s) in raw.items()}
    xf = x.float()
    gate_x = xf @ ref_w["w1"].t()
    up_x = xf @ ref_w["w3"].t()
    ref = situ_ref_fp32(gate_x, up_x) @ ref_w["w2"].t()

    print(f"    tokens={num_tokens} out={tuple(out.shape)} {out.dtype} "
          f"ref rms={float(ref.pow(2).mean().sqrt()):.4f}")
    report(f"expert output vs oracle-dequant reference (t={num_tokens})",
           gate(out, ref, "expert"))
    report("weight buffer released exactly once per forward",
           engine.freed == ["routed_expert_4_0"], f"freed={engine.freed}")

    # 0-token expert: the lockstep drive must still load+free, and must not
    # launch an M=0 grid.
    engine.freed.clear()
    empty_out = wrapper(torch.empty(0, K3_LATENT, device=dev,
                                    dtype=torch.bfloat16))
    report("0-token expert returns [0, latent] and still frees its slot",
           tuple(empty_out.shape) == (0, K3_LATENT)
           and engine.freed == ["routed_expert_4_0"],
           f"shape={tuple(empty_out.shape)} freed={engine.freed}")


def check_negatives(dev):
    from batchgen.models.moonshotai.kimi_linear.k3.mxfp4_layout import (
        K3QuantContractError,
    )

    raw = {}
    for name, (K, N) in (("w1", (K3_LATENT, K3_FFN)),
                         ("w3", (K3_LATENT, K3_FFN)),
                         ("w2", (K3_FFN, K3_LATENT))):
        raw[name] = rand_expert_weight(K, N, seed=hash(name) % 2**31)
    base = {}
    for name, (packed, scale) in raw.items():
        base[f"{name}.weight_packed"] = packed
        base[f"{name}.weight_scale"] = scale

    x = torch.zeros(8, K3_LATENT, device=dev, dtype=torch.bfloat16)

    mutations = {
        "dropped w3.weight_scale": lambda d: d.pop("w3.weight_scale"),
        "w2.weight_packed as bf16": lambda d: d.__setitem__(
            "w2.weight_packed", d["w2.weight_packed"].to(torch.bfloat16)),
        "w1.weight_packed one row short": lambda d: d.__setitem__(
            "w1.weight_packed", d["w1.weight_packed"][:-1].contiguous()),
        "extra unexpected tensor": lambda d: d.__setitem__(
            "w1.weight_zero_point", d["w1.weight_scale"].clone()),
    }
    caught = 0
    for tag, mutate in mutations.items():
        served = dict(base)
        mutate(served)
        wrapper, _, _ = build_wrapped_expert(dev, served)
        try:
            wrapper(x)
            print(f"    {tag}: NO RAISE  <-- silent")
        except K3QuantContractError as exc:
            caught += 1
            print(f"    {tag}: raised  ({str(exc).split(chr(10))[0][:88]})")
        except Exception as exc:                                 # noqa: BLE001
            print(f"    {tag}: raised {type(exc).__name__} (not the contract "
                  f"error): {str(exc)[:88]}")
    report("every slot mutation hard-fails", caught == len(mutations),
           f"{caught}/{len(mutations)}")

    # persistent=True must be refused outright.
    from types import SimpleNamespace

    from batchgen.models.moonshotai.kimi_linear.k3.mxfp4_expert import (
        K3MXFP4Expert,
        KimiK3MXFP4ExpertWrapper,
    )
    cfg = SimpleNamespace(activation_situ_beta=4.0,
                          activation_situ_linear_beta=25.0)
    try:
        KimiK3MXFP4ExpertWrapper(
            K3MXFP4Expert(cfg, K3_LATENT, K3_FFN), 0, 0, None, None, None,
            persistent=True)
        report("persistent=True refused", False, "no raise")
    except K3QuantContractError:
        report("persistent=True refused", True)

    # A config whose SiTU betas disagree with the compiled kernel constants.
    try:
        K3MXFP4Expert(SimpleNamespace(activation_situ_beta=2.0,
                                      activation_situ_linear_beta=25.0),
                      K3_LATENT, K3_FFN)
        report("SiTU beta mismatch refused", False, "no raise")
    except K3QuantContractError:
        report("SiTU beta mismatch refused", True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--layers", type=int, default=None,
                    help="shrink depth for the meta build (default: real 93)")
    ap.add_argument("--experts", type=int, default=None,
                    help="shrink experts/layer for the meta build "
                         "(default: real 896)")
    ap.add_argument("--tokens", type=int, default=64)
    args = ap.parse_args()
    if not (args.meta or args.gpu):
        ap.error("pass --meta and/or --gpu")

    if args.meta:
        print("== meta build: expert name-set arithmetic ==")
        check_meta(args.layers, args.experts)
    if args.gpu:
        if not torch.cuda.is_available():
            sys.exit("--gpu needs CUDA")
        dev = "cuda"
        print(f"== GPU checks on {torch.cuda.get_device_name(0)} ==")
        check_repack_bit_identity(dev)
        check_numerics(dev, args.tokens)
        check_negatives(dev)

    failed = [n for n, ok in _results if not ok]
    print(f"\n{len(_results) - len(failed)}/{len(_results)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
