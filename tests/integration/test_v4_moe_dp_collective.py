import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch


def _make_config(hidden_size, n_experts, topk, num_hash_layers=0):
    return SimpleNamespace(
        hidden_size=hidden_size,
        moe_intermediate_size=16,
        n_routed_experts=n_experts,
        num_experts_per_tok=topk,
        num_hash_layers=num_hash_layers,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        swiglu_limit=0.0,
        vocab_size=128,
        pad_token_id=0,
    )


def _install_weights(
    moe, hidden_size, n_experts, device, owned_only, tid2eid=None
):
    inter = moe.intermediate_size
    with torch.no_grad():
        gw = torch.zeros(
            n_experts, hidden_size, device=device, dtype=torch.bfloat16
        )
        for e in range(n_experts):
            gw[e, e] = 10.0
        moe.gate.weight.copy_(gw)
        if moe.gate.bias is not None:
            moe.gate.bias.zero_()
        if getattr(moe.gate, "is_hash_layer", False):
            assert tid2eid is not None
            table = torch.zeros_like(moe.gate.tid2eid)
            for token_id, experts in tid2eid.items():
                table[token_id] = torch.tensor(experts, device=table.device)
            moe.gate.tid2eid.copy_(table)

    lo, hi = moe.routed_expert_start_idx, moe.routed_expert_end_idx
    for e in range(n_experts):
        if owned_only and not (lo <= e < hi):
            continue
        w1 = torch.zeros(
            inter, hidden_size, device=device, dtype=torch.bfloat16
        )
        w3 = torch.zeros(
            inter, hidden_size, device=device, dtype=torch.bfloat16
        )
        w2 = torch.zeros(
            hidden_size, inter, device=device, dtype=torch.bfloat16
        )
        w1[0, :] = 1.0
        w3[0, :] = 1.0
        w2[e, 0] = float(e + 1)
        moe.experts[e].set_runtime_tensors(
            {"w1.weight": w1, "w3.weight": w3, "w2.weight": w2}
        )

    zero = lambda r, c: torch.zeros(r, c, device=device, dtype=torch.bfloat16)
    moe.shared_experts.set_runtime_tensors(
        {
            "w1.weight": zero(inter, hidden_size),
            "w3.weight": zero(inter, hidden_size),
            "w2.weight": zero(hidden_size, inter),
        }
    )


def _tokens(specs, hidden_size, device):
    x = torch.zeros(
        len(specs), 1, hidden_size, device=device, dtype=torch.bfloat16
    )
    ids = torch.tensor(
        [[tok_id] for _, tok_id in specs], device=device, dtype=torch.long
    )
    for i, (dim, _) in enumerate(specs):
        x[i, 0, dim] = 1.0
    return x, ids


def _scenario(name):
    if name == "swap":
        return dict(
            n_experts=4,
            topk=1,
            num_hash_layers=0,
            tid2eid=None,
            per_rank={0: [(2, 2)], 1: [(0, 0)]},
        )
    if name == "uneven":
        return dict(
            n_experts=4,
            topk=1,
            num_hash_layers=0,
            tid2eid=None,
            per_rank={0: [(0, 0), (1, 1)], 1: [(2, 2)]},
        )
    if name == "hash_empty_topk2":
        return dict(
            n_experts=4,
            topk=2,
            num_hash_layers=1,
            tid2eid={5: [0, 2], 6: [1, 3]},
            per_rank={0: [], 1: [(0, 5), (2, 6)]},
        )
    raise ValueError(name)


def _oracle_outputs(spec, hidden_size, device):
    from batchgen.models.deepseek.deepseekv4_flash.model import (
        DeepSeekV4FlashMoE,
    )

    cfg = _make_config(
        hidden_size, spec["n_experts"], spec["topk"], spec["num_hash_layers"]
    )
    moe = DeepSeekV4FlashMoE(cfg, layer_idx=0).to(device).to(torch.bfloat16)
    moe.configure_ep(0, 1)
    _install_weights(
        moe,
        hidden_size,
        spec["n_experts"],
        device,
        owned_only=False,
        tid2eid=spec["tid2eid"],
    )
    global_specs = []
    for r in sorted(spec["per_rank"]):
        global_specs.extend(spec["per_rank"][r])
    x, ids = _tokens(global_specs, hidden_size, device)
    with torch.inference_mode():
        out = moe(x, ids)
    return out.reshape(len(global_specs), hidden_size)


def _run_worker(scenario):
    import torch.distributed as dist
    from batchgen.models.deepseek.deepseekv4_flash.model import (
        DeepSeekV4FlashMoE,
    )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")

    hidden_size = 8
    if scenario == "resize":
        _run_resize_worker(rank, world_size, device, hidden_size)
        return

    spec = _scenario(scenario)
    cfg = _make_config(
        hidden_size, spec["n_experts"], spec["topk"], spec["num_hash_layers"]
    )
    max_tokens = max(1, max(len(v) for v in spec["per_rank"].values()))

    moe = DeepSeekV4FlashMoE(cfg, layer_idx=0).to(device).to(torch.bfloat16)
    moe.configure_ep(rank, world_size)
    moe.init_num_tokens(max_tokens)
    moe.set_num_tokens_per_rank(max_tokens)
    _install_weights(
        moe,
        hidden_size,
        spec["n_experts"],
        device,
        owned_only=True,
        tid2eid=spec["tid2eid"],
    )

    specs = spec["per_rank"][rank]
    x, ids = _tokens(specs, hidden_size, device)
    with torch.inference_mode():
        out = moe(x, ids).reshape(len(specs), hidden_size)

    oracle = _oracle_outputs(spec, hidden_size, device)
    offset = sum(len(spec["per_rank"][r]) for r in range(rank))
    expected = oracle[offset : offset + len(specs)]

    ok = torch.allclose(out.float(), expected.float(), atol=0.05, rtol=0.05)
    nonzero = len(specs) == 0 or float(out.float().abs().sum()) > 1e-3
    dist.barrier()
    if not (ok and nonzero):
        print(
            f"RANK{rank} MISMATCH ok={ok} nonzero={nonzero} "
            f"out={out.float().tolist()} expected={expected.float().tolist()}",
            flush=True,
        )
        dist.destroy_process_group()
        sys.exit(2)
    print(f"RANK{rank} OK", flush=True)
    dist.destroy_process_group()


def _run_resize_worker(rank, world_size, device, hidden_size):
    from batchgen.models.deepseek.deepseekv4_flash.model import (
        DeepSeekV4FlashMoE,
    )
    import torch.distributed as dist

    n_experts, topk = 4, 1
    cfg = _make_config(hidden_size, n_experts, topk, num_hash_layers=0)
    moe = DeepSeekV4FlashMoE(cfg, layer_idx=0).to(device).to(torch.bfloat16)
    moe.configure_ep(rank, world_size)
    moe.init_num_tokens(2)
    _install_weights(moe, hidden_size, n_experts, device, owned_only=True)

    steps = [
        {0: [(0, 0), (1, 1)], 1: [(2, 2)]},
        {0: [(0, 0)], 1: [(3, 3)]},
        {0: [(0, 0)], 1: [(2, 2), (3, 3)]},
        {0: [(0, 0)], 1: []},
    ]

    for step_idx, per_rank in enumerate(steps):
        max_tokens = max(1, max(len(v) for v in per_rank.values()))
        moe.set_num_tokens_per_rank(max_tokens)
        spec = dict(
            n_experts=n_experts,
            topk=topk,
            num_hash_layers=0,
            tid2eid=None,
            per_rank=per_rank,
        )
        specs = per_rank[rank]
        x, ids = _tokens(specs, hidden_size, device)
        with torch.inference_mode():
            out = moe(x, ids).reshape(len(specs), hidden_size)
        oracle = _oracle_outputs(spec, hidden_size, device)
        offset = sum(len(per_rank[r]) for r in range(rank))
        expected = oracle[offset : offset + len(specs)]
        ok = torch.allclose(out.float(), expected.float(), atol=0.05, rtol=0.05)
        if not ok:
            print(
                f"RANK{rank} STEP{step_idx} MISMATCH "
                f"out={out.float().tolist()} expected={expected.float().tolist()}",
                flush=True,
            )
            dist.destroy_process_group()
            sys.exit(2)
    print(f"RANK{rank} OK", flush=True)
    dist.destroy_process_group()


def _launch(scenario, timeout=120, port="29555"):
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        f"--master_port={port}",
        __file__,
        scenario,
    ]
    return subprocess.run(
        cmd,
        env=dict(os.environ),
        timeout=timeout,
        capture_output=True,
        text=True,
    )


requires_2gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires 2 GPUs",
)


@requires_2gpu
def test_dp_moe_routes_token_to_remote_rank_expert():
    result = _launch("swap", port="29555")
    assert result.returncode == 0, (
        f"swap failed (rc={result.returncode})\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr[-2000:]}"
    )


@requires_2gpu
def test_dp_moe_handles_uneven_token_counts_across_ranks():
    try:
        result = _launch("uneven", timeout=90, port="29556")
    except subprocess.TimeoutExpired:
        pytest.fail("uneven scenario hung (collective shape mismatch deadlock)")
    assert result.returncode == 0, (
        f"uneven failed (rc={result.returncode})\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr[-2000:]}"
    )


@requires_2gpu
def test_dp_moe_hash_routing_topk2_with_empty_rank():
    try:
        result = _launch("hash_empty_topk2", timeout=90, port="29557")
    except subprocess.TimeoutExpired:
        pytest.fail("hash_empty_topk2 hung (empty-rank collective deadlock)")
    assert result.returncode == 0, (
        f"hash_empty_topk2 failed (rc={result.returncode})\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr[-2000:]}"
    )


@requires_2gpu
def test_dp_moe_dynamic_num_tokens_per_rank_resize():
    try:
        result = _launch("resize", timeout=120, port="29558")
    except subprocess.TimeoutExpired:
        pytest.fail("resize scenario hung")
    assert result.returncode == 0, (
        f"resize failed (rc={result.returncode})\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr[-2000:]}"
    )


def _build_moe_for_guard():
    from batchgen.models.deepseek.deepseekv4_flash.model import (
        DeepSeekV4FlashMoE,
    )

    cfg = _make_config(hidden_size=8, n_experts=4, topk=1, num_hash_layers=0)
    moe = DeepSeekV4FlashMoE(cfg, layer_idx=0)
    moe.configure_ep(0, 2)
    return moe


def test_ep_decode_raises_when_num_tokens_not_initialized():
    moe = _build_moe_for_guard()
    flat = torch.zeros(1, 8)
    with pytest.raises(RuntimeError, match="num_tokens_per_rank is not"):
        moe._forward_ep_decode_routed(flat, None)


def test_ep_decode_raises_on_buffer_overflow():
    moe = _build_moe_for_guard()
    moe.init_num_tokens(1)
    flat = torch.zeros(2, 8)
    with pytest.raises(RuntimeError, match="buffer overflow"):
        moe._forward_ep_decode_routed(flat, None)


if __name__ == "__main__":
    _run_worker(sys.argv[1])
