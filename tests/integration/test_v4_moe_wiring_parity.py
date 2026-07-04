"""Module-level DeepSeek-V4-Flash MoE wiring parity tests.

This exercises the real ``DeepSeekV4FlashMoE`` staging/dispatch path:

router logits -> top-k routing -> ``configure_ep`` owned-range selection ->
``_stage_owned_expert_weights`` / ``setup_v4_expert_weight_pointers`` ->
mega3 or ragged kernel -> routed-output combine.

The config values mirror ``batchgen/models/deepseek/deepseekv4_flash/config.py``.
This test uses a lightweight namespace instead of importing that config module
directly because the package-level config registry triggers an unrelated
``core_engine`` JIT build during pytest collection in this environment.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from batchgen.models.deepseek.deepseekv4_flash.model import DeepSeekV4FlashMoE
from batchgen.models.deepseek.deepseekv4_flash.Parallel_Strategy_Manager import (
    DeepSeekV4FlashParallelStrategyManager,
)
from batchgen.moe.fp4_utils import dequant_fp4_e2m1_weight
from batchgen.server.worker_env import apply_worker_env_overrides
from benchmarks.grouped_moe_probes.common import compute_gate

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _sm120_or_newer() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= 12


@pytest.fixture(scope="module")
def v4_flash_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=4096,
        n_routed_experts=256,
        num_local_experts=256,
        num_experts_per_tok=6,
        moe_intermediate_size=2048,
        swiglu_limit=10.0,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        norm_topk_prob=True,
        num_hash_layers=3,
        vocab_size=129280,
        pad_token_id=1,
    )


@pytest.fixture(scope="module")
def expert_weights(
    v4_flash_config: SimpleNamespace,
) -> list[dict[str, torch.Tensor]]:
    torch.manual_seed(20260629)
    return _make_expert_weights(v4_flash_config, device=torch.device("cuda"))


def _max_rel_diff(ref: torch.Tensor, out: torch.Tensor) -> float:
    ref_f = ref.float()
    out_f = out.float()
    return float(((out_f - ref_f).abs() / ref_f.abs().clamp_min(1e-6)).max().item())


def _max_abs_diff(ref: torch.Tensor, out: torch.Tensor) -> float:
    return float((out.float() - ref.float()).abs().max().item())


_POSITIVE_FP4_LEVELS = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def _nearest_positive_level(values: torch.Tensor) -> torch.Tensor:
    levels = _POSITIVE_FP4_LEVELS.to(device=values.device)
    return (values.unsqueeze(-1) - levels).abs().argmin(dim=-1).to(torch.uint8)


def _make_mxfp4_linear(
    out_features: int,
    in_features: int,
    *,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert in_features % 32 == 0
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    weight = torch.rand(
        out_features,
        in_features,
        generator=gen,
        dtype=torch.float32,
    )
    weight = (weight * 0.05).to(device=device, dtype=torch.bfloat16)
    blocks = weight.float().reshape(out_features, in_features // 32, 32)
    raw_scale = torch.clamp(blocks.abs().amax(dim=-1) / 6.0, min=2.0**-20)
    log2_scale = torch.round(torch.log2(raw_scale))
    scale = torch.pow(
        torch.full_like(log2_scale, 2.0, dtype=torch.float32), log2_scale
    )
    normalized = blocks / scale.unsqueeze(-1)
    signs = normalized < 0
    magnitudes = normalized.abs().reshape(out_features, in_features)
    pos_codes = _nearest_positive_level(magnitudes)
    codes = pos_codes | (signs.reshape(out_features, in_features).to(torch.uint8) << 3)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is not None:
        scale = scale.to(e8m0_dtype)
    return packed.view(torch.float4_e2m1fn_x2).contiguous(), scale.contiguous()


def _make_expert_weights(
    cfg: SimpleNamespace,
    *,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    weights: list[dict[str, torch.Tensor]] = []
    for _ in range(cfg.n_routed_experts):
        expert_idx = len(weights)
        w1, s1 = _make_mxfp4_linear(
            cfg.moe_intermediate_size,
            cfg.hidden_size,
            device=device,
            seed=10_000 + expert_idx * 10 + 1,
        )
        w3, s3 = _make_mxfp4_linear(
            cfg.moe_intermediate_size,
            cfg.hidden_size,
            device=device,
            seed=10_000 + expert_idx * 10 + 2,
        )
        w2, s2 = _make_mxfp4_linear(
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            device=device,
            seed=10_000 + expert_idx * 10 + 3,
        )
        weights.append(
            {
                "w1.weight": w1,
                "w1.scale": s1,
                "w3.weight": w3,
                "w3.scale": s3,
                "w2.weight": w2,
                "w2.scale": s2,
            }
        )
    return weights


def _clone_runtime_weights(
    runtime_weights: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {name: tensor.clone() for name, tensor in runtime_weights.items()}


def _build_moe(
    cfg: SimpleNamespace,
    expert_weights: list[dict[str, torch.Tensor]],
) -> DeepSeekV4FlashMoE:
    moe = DeepSeekV4FlashMoE(cfg, layer_idx=0).cuda().eval()
    for expert, runtime_weights in zip(moe.experts, expert_weights):
        expert.set_runtime_tensors(runtime_weights)
    return moe


def _owned_runtime_bytes(moe: DeepSeekV4FlashMoE) -> int:
    return moe._owned_expert_runtime_bytes()


def _grouped_bundle_bytes(moe: DeepSeekV4FlashMoE) -> int:
    return moe._grouped_staged_bundle_bytes()


class _FakeCoreEngine:
    def __init__(self, tensors_by_key: dict[str, dict[str, torch.Tensor]]):
        self._tensors_by_key = tensors_by_key

    def get_tensor(self, key: str) -> dict[str, torch.Tensor]:
        return self._tensors_by_key[key]


class _FakeStreamingExpertWrapper(nn.Module):
    def __init__(self, module: nn.Module, weights: dict[str, torch.Tensor]):
        super().__init__()
        self.module = module
        self._weights = weights
        self.module_key = "fake_streaming_expert"

    def load_weights(self, key: str) -> dict[str, torch.Tensor]:
        assert key == self.module_key
        if self._weights is None:
            raise RuntimeError("streaming source weights were released")
        return {name: tensor.clone() for name, tensor in self._weights.items()}

    def forward(self, *args, **kwargs):
        tensors = self.load_weights(self.module_key)
        self.module.set_runtime_tensors(tensors)
        try:
            return self.module(*args, **kwargs)
        finally:
            self.module.clear_runtime_tensors()


class _ZeroSharedExperts(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(hidden_states)


def _configure_ep(
    moe: DeepSeekV4FlashMoE,
    *,
    prefill: bool,
    global_rank: int,
    world_size: int,
) -> None:
    if prefill:
        moe.configure_ep(rank=0, world_size=1, comm=None)
    else:
        moe.configure_ep(rank=global_rank, world_size=world_size, comm=None)


def _router_topk_from_logits(
    router_logits: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.softmax(router_logits.float(), dim=-1)
    weights, indices = torch.topk(probs, k=topk, dim=-1)
    return weights.to(torch.float32).contiguous(), indices.to(torch.int64).contiguous()


def _make_router_logits(
    *,
    num_tokens: int,
    num_experts: int,
    topk: int,
    device: torch.device,
    owned_start: int,
    owned_count: int,
    partial: bool,
    zero_owned: bool = False,
) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(num_tokens * 10_007 + owned_start * 97 + int(partial) * 13)
    logits = torch.randn(
        num_tokens, num_experts, generator=gen, dtype=torch.float32
    ).to(device=device)
    for token_idx in range(num_tokens):
        if partial:
            owned = [
                owned_start + ((token_idx * 7 + step * 11) % owned_count)
                for step in range(topk // 2)
            ]
            remote = []
            cursor = token_idx * 13 + 5
            while len(remote) < topk:
                expert_idx = cursor % num_experts
                cursor += 17
                if owned_start <= expert_idx < owned_start + owned_count:
                    continue
                remote.append(expert_idx)
            if zero_owned:
                logits[token_idx, owned_start : owned_start + owned_count] -= 8.0
                logits[token_idx, torch.tensor(remote[:topk], device=device)] += 6.0
            else:
                logits[token_idx, torch.tensor(owned, device=device)] += 3.0
                logits[token_idx, torch.tensor(remote[: topk - len(owned)], device=device)] += 1.0
    return logits


def _eager_reference(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    expert_weights: list[dict[str, torch.Tensor]],
    *,
    owned_start: int,
    owned_count: int,
    swiglu_limit: float,
) -> torch.Tensor:
    num_tokens, hidden = hidden_states.shape
    hidden_states = hidden_states.to(torch.bfloat16)
    out = torch.zeros((num_tokens, hidden), device=hidden_states.device, dtype=torch.float32)
    owned_end = owned_start + owned_count
    for expert_idx in range(owned_start, owned_end):
        token_ids, topk_pos = torch.where(topk_indices == expert_idx)
        if token_ids.numel() == 0:
            continue
        weights = topk_weights[token_ids, topk_pos].float().unsqueeze(-1)
        runtime = expert_weights[expert_idx]
        gate_w = dequant_fp4_e2m1_weight(
            runtime["w1.weight"], runtime["w1.scale"], torch.bfloat16
        )
        up_w = dequant_fp4_e2m1_weight(
            runtime["w3.weight"], runtime["w3.scale"], torch.bfloat16
        )
        down_w = dequant_fp4_e2m1_weight(
            runtime["w2.weight"], runtime["w2.scale"], torch.bfloat16
        )
        local_hidden = hidden_states.index_select(0, token_ids)
        gate = F.linear(local_hidden, gate_w).float()
        up = F.linear(local_hidden, up_w).float()
        if swiglu_limit > 0:
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
        activated = F.silu(gate) * up
        expert_out = F.linear(activated.to(torch.bfloat16), down_w).float()
        out.index_add_(0, token_ids, expert_out * weights)
    return out


@contextmanager
def _ragged_env(enabled: bool):
    prev = os.environ.get("BATCHGEN_V4_RAGGED_FALLBACK")
    if enabled:
        os.environ["BATCHGEN_V4_RAGGED_FALLBACK"] = "1"
    else:
        os.environ.pop("BATCHGEN_V4_RAGGED_FALLBACK", None)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("BATCHGEN_V4_RAGGED_FALLBACK", None)
        else:
            os.environ["BATCHGEN_V4_RAGGED_FALLBACK"] = prev


def _run_through_real_moe_path(
    moe: DeepSeekV4FlashMoE,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    ragged_fallback: bool,
) -> torch.Tensor:
    with _ragged_env(ragged_fallback), torch.inference_mode():
        assert moe._stage_owned_expert_weights()
        assert moe._grouped_staged is not None
        return moe._run_owned_experts_grouped(
            hidden_states, topk_weights, topk_indices
        )


@pytest.mark.skipif(not _sm120_or_newer(), reason="sm120+ required")
@pytest.mark.parametrize(
    ("name", "prefill", "global_rank", "world_size", "num_tokens", "partial"),
    [
        ("all_owned", True, 0, 1, 12, False),
        ("ep_partial", False, 1, 4, 9, True),
    ],
)
def test_v4_flash_moe_module_wiring_parity(
    v4_flash_config: SimpleNamespace,
    expert_weights: list[dict[str, torch.Tensor]],
    name: str,
    prefill: bool,
    global_rank: int,
    world_size: int,
    num_tokens: int,
    partial: bool,
):
    device = torch.device("cuda")
    moe = _build_moe(v4_flash_config, expert_weights)
    _configure_ep(
        moe,
        prefill=prefill,
        global_rank=global_rank,
        world_size=world_size,
    )

    hidden_states = torch.rand(
        num_tokens,
        v4_flash_config.hidden_size,
        dtype=torch.float32,
        device=device,
    ).to(torch.bfloat16) * 0.1
    router_logits = _make_router_logits(
        num_tokens=num_tokens,
        num_experts=v4_flash_config.n_routed_experts,
        topk=v4_flash_config.num_experts_per_tok,
        device=device,
        owned_start=moe.routed_expert_start_idx,
        owned_count=moe.routed_expert_end_idx - moe.routed_expert_start_idx,
        partial=partial,
    )
    topk_weights, topk_indices = _router_topk_from_logits(
        router_logits,
        v4_flash_config.num_experts_per_tok,
    )

    eager = _eager_reference(
        hidden_states,
        topk_weights,
        topk_indices,
        expert_weights,
        owned_start=moe.routed_expert_start_idx,
        owned_count=moe.routed_expert_end_idx - moe.routed_expert_start_idx,
        swiglu_limit=v4_flash_config.swiglu_limit,
    )
    mega3 = _run_through_real_moe_path(
        moe,
        hidden_states,
        topk_weights,
        topk_indices,
        ragged_fallback=False,
    )
    ragged = _run_through_real_moe_path(
        moe,
        hidden_states,
        topk_weights,
        topk_indices,
        ragged_fallback=True,
    )

    eager_gate = compute_gate(eager, mega3)
    ragged_gate = compute_gate(ragged, mega3)
    mega3_vs_eager = _max_rel_diff(eager, mega3)
    mega3_vs_ragged = _max_rel_diff(ragged, mega3)

    print(
        f"[{name}] mega3_vs_eager max_rel_diff={mega3_vs_eager:.6f} "
        f"max_abs_diff={_max_abs_diff(eager, mega3):.6f}"
    )
    print(
        f"[{name}] mega3_vs_ragged max_rel_diff={mega3_vs_ragged:.6f} "
        f"max_abs_diff={_max_abs_diff(ragged, mega3):.6f}"
    )

    assert torch.isfinite(eager).all(), f"{name}: eager produced non-finite values"
    assert torch.isfinite(mega3).all(), f"{name}: mega3 produced non-finite values"
    assert torch.isfinite(ragged).all(), f"{name}: ragged produced non-finite values"
    assert mega3_vs_eager < 0.05, (
        f"{name}: mega3 vs eager wiring diff {mega3_vs_eager:.6f} >= 0.05; "
        f"gate={eager_gate}"
    )
    assert mega3_vs_ragged < 0.02, (
        f"{name}: mega3 vs ragged wiring diff {mega3_vs_ragged:.6f} >= 0.02; "
        f"gate={ragged_gate}"
    )


@pytest.mark.skipif(not _sm120_or_newer(), reason="sm120+ required")
def test_v4_flash_moe_zero_owned_routes_do_not_crash(
    v4_flash_config: SimpleNamespace,
    expert_weights: list[dict[str, torch.Tensor]],
):
    torch.manual_seed(20260630)
    device = torch.device("cuda")
    moe = _build_moe(v4_flash_config, expert_weights)
    _configure_ep(
        moe,
        prefill=False,
        global_rank=1,
        world_size=4,
    )

    hidden_states = torch.rand(
        7,
        v4_flash_config.hidden_size,
        dtype=torch.float32,
        device=device,
    ).to(torch.bfloat16) * 0.1
    router_logits = _make_router_logits(
        num_tokens=hidden_states.shape[0],
        num_experts=v4_flash_config.n_routed_experts,
        topk=v4_flash_config.num_experts_per_tok,
        device=device,
        owned_start=moe.routed_expert_start_idx,
        owned_count=moe.routed_expert_end_idx - moe.routed_expert_start_idx,
        partial=True,
        zero_owned=True,
    )
    topk_weights, topk_indices = _router_topk_from_logits(
        router_logits,
        v4_flash_config.num_experts_per_tok,
    )

    eager = _eager_reference(
        hidden_states,
        topk_weights,
        topk_indices,
        expert_weights,
        owned_start=moe.routed_expert_start_idx,
        owned_count=moe.routed_expert_end_idx - moe.routed_expert_start_idx,
        swiglu_limit=v4_flash_config.swiglu_limit,
    )
    mega3 = _run_through_real_moe_path(
        moe,
        hidden_states,
        topk_weights,
        topk_indices,
        ragged_fallback=False,
    )
    ragged = _run_through_real_moe_path(
        moe,
        hidden_states,
        topk_weights,
        topk_indices,
        ragged_fallback=True,
    )

    assert torch.isfinite(eager).all()
    assert torch.isfinite(mega3).all()
    assert torch.isfinite(ragged).all()
    assert torch.count_nonzero(eager) == 0
    assert torch.equal(mega3, torch.zeros_like(mega3))
    assert torch.equal(ragged, torch.zeros_like(ragged))


@pytest.mark.skipif(not _sm120_or_newer(), reason="sm120+ required")
def test_v4_flash_prefill_all_owned_streams_eager_without_grouped_bundle(
    monkeypatch: pytest.MonkeyPatch,
    v4_flash_config: SimpleNamespace,
    expert_weights: list[dict[str, torch.Tensor]],
):
    torch.manual_seed(20260702)
    device = torch.device("cuda")
    moe = DeepSeekV4FlashMoE(v4_flash_config, layer_idx=0).cuda().eval()
    moe.shared_experts = _ZeroSharedExperts().cuda()
    for expert_idx, weights in enumerate(expert_weights):
        moe.experts[expert_idx] = _FakeStreamingExpertWrapper(
            moe.experts[expert_idx], weights
        )
    _configure_ep(moe, prefill=True, global_rank=0, world_size=1)

    hidden_states = torch.rand(
        11,
        v4_flash_config.hidden_size,
        dtype=torch.float32,
        device=device,
    ).to(torch.bfloat16) * 0.1
    input_ids = torch.arange(hidden_states.shape[0], device=device, dtype=torch.int64)
    router_logits = _make_router_logits(
        num_tokens=hidden_states.shape[0],
        num_experts=v4_flash_config.n_routed_experts,
        topk=v4_flash_config.num_experts_per_tok,
        device=device,
        owned_start=0,
        owned_count=v4_flash_config.n_routed_experts,
        partial=False,
    )
    topk_weights, topk_indices = _router_topk_from_logits(
        router_logits,
        v4_flash_config.num_experts_per_tok,
    )

    eager = _eager_reference(
        hidden_states,
        topk_weights,
        topk_indices,
        expert_weights,
        owned_start=0,
        owned_count=v4_flash_config.n_routed_experts,
        swiglu_limit=v4_flash_config.swiglu_limit,
    )

    monkeypatch.setattr(
        moe.gate,
        "forward",
        lambda _hidden_states, _input_ids=None: (topk_weights, topk_indices),
    )

    def _forbid_grouped_stage(*args, **kwargs):
        raise AssertionError("prefill all-owned path must not build grouped bundle")

    monkeypatch.setattr(moe, "_stage_owned_expert_weights", _forbid_grouped_stage)

    out = moe(hidden_states, input_ids)

    assert _grouped_bundle_bytes(moe) == 0
    assert moe._grouped_staged is None
    assert _max_rel_diff(eager, out) < 0.05
    for expert in moe.experts:
        module = getattr(expert, "module", expert)
        assert getattr(module, "runtime_weights", None) is None


@pytest.mark.skipif(not _sm120_or_newer(), reason="sm120+ required")
def test_v4_flash_resident_mode_prestages_grouped_bundle_at_load(
    monkeypatch: pytest.MonkeyPatch,
    v4_flash_config: SimpleNamespace,
    expert_weights: list[dict[str, torch.Tensor]],
):
    moe = DeepSeekV4FlashMoE(v4_flash_config, layer_idx=0).cuda().eval()
    for expert_idx, weights in enumerate(expert_weights):
        moe.experts[expert_idx] = _FakeStreamingExpertWrapper(
            moe.experts[expert_idx], weights
        )
    moe.configure_ep(rank=1, world_size=4, comm=None)
    owned_start = moe.routed_expert_start_idx
    owned_end = moe.routed_expert_end_idx
    tensors_by_key = {
        f"routed_expert_0_{expert_idx}": expert_weights[expert_idx]
        for expert_idx in range(owned_start, owned_end)
    }

    manager = object.__new__(DeepSeekV4FlashParallelStrategyManager)
    manager.model = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(mlp=moe)])
    )
    manager.core_engine = _FakeCoreEngine(tensors_by_key)
    manager.engine_config = SimpleNamespace(
        Basic_Config=SimpleNamespace(device_torch=torch.device("cuda"))
    )
    manager.rank = 0

    monkeypatch.delenv("BATCHGEN_V4_RESIDENT_EXPERTS", raising=False)
    apply_worker_env_overrides(SimpleNamespace(v4_resident_experts=True))
    assert os.environ["BATCHGEN_V4_RESIDENT_EXPERTS"] == "1"
    assert manager._resident_experts_enabled()
    assert moe._grouped_staged is None

    manager._load_local_routed_experts()

    assert moe._grouped_staged is not None
    assert moe._grouped_staged["global_expert_count"] == v4_flash_config.n_routed_experts
    assert _owned_runtime_bytes(moe) == 0
    assert _grouped_bundle_bytes(moe) > 0
    for expert_idx in range(owned_start, owned_end):
        module = getattr(moe.experts[expert_idx], "module", moe.experts[expert_idx])
        assert getattr(module, "runtime_weights", None) is None


@pytest.mark.skipif(not _sm120_or_newer(), reason="sm120+ required")
def test_v4_flash_grouped_staging_releases_original_runtime_tensors_single_copy(
    v4_flash_config: SimpleNamespace,
    expert_weights: list[dict[str, torch.Tensor]],
):
    device = torch.device("cuda")
    moe = DeepSeekV4FlashMoE(v4_flash_config, layer_idx=0).cuda().eval()
    _configure_ep(moe, prefill=False, global_rank=1, world_size=4)
    owned_start = moe.routed_expert_start_idx
    owned_end = moe.routed_expert_end_idx

    torch.cuda.synchronize(device)
    baseline_alloc = torch.cuda.memory_allocated(device)

    expected_original_bytes = 0
    for expert_idx in range(owned_start, owned_end):
        cloned = _clone_runtime_weights(expert_weights[expert_idx])
        expected_original_bytes += sum(
            tensor.numel() * tensor.element_size()
            for tensor in cloned.values()
            if tensor.is_cuda
        )
        moe.experts[expert_idx].set_runtime_tensors(cloned)

    torch.cuda.synchronize(device)
    after_originals = torch.cuda.memory_allocated(device)
    assert _owned_runtime_bytes(moe) == expected_original_bytes
    assert after_originals - baseline_alloc == expected_original_bytes

    assert moe._stage_owned_expert_weights(release_runtime_tensors=True)

    torch.cuda.synchronize(device)
    after_bundle = torch.cuda.memory_allocated(device)
    bundle_bytes = _grouped_bundle_bytes(moe)
    overhead_bytes = after_bundle - baseline_alloc - bundle_bytes

    print(
        "[single-copy] owned_original_bytes="
        f"{expected_original_bytes} bundle_bytes={bundle_bytes} "
        f"post_stage_delta={after_bundle - baseline_alloc} "
        f"overhead_bytes={overhead_bytes}"
    )

    assert _owned_runtime_bytes(moe) == 0
    for expert_idx in range(owned_start, owned_end):
        module = getattr(moe.experts[expert_idx], "module", moe.experts[expert_idx])
        assert getattr(module, "runtime_weights", None) is None
    assert bundle_bytes > 0
    assert 0 <= overhead_bytes <= 16 * 1024 * 1024
    assert after_bundle - baseline_alloc <= bundle_bytes + 16 * 1024 * 1024
    assert after_bundle - baseline_alloc < expected_original_bytes + bundle_bytes


@pytest.mark.skipif(not _sm120_or_newer(), reason="sm120+ required")
def test_v4_flash_streaming_mode_keeps_lazy_bundle_build_and_cached_outputs(
    monkeypatch: pytest.MonkeyPatch,
    v4_flash_config: SimpleNamespace,
    expert_weights: list[dict[str, torch.Tensor]],
):
    torch.manual_seed(20260701)
    device = torch.device("cuda")
    moe = DeepSeekV4FlashMoE(v4_flash_config, layer_idx=0).cuda().eval()
    for expert_idx, weights in enumerate(expert_weights):
        moe.experts[expert_idx] = _FakeStreamingExpertWrapper(
            moe.experts[expert_idx], weights
        )
    _configure_ep(moe, prefill=False, global_rank=1, world_size=4)

    hidden_states = torch.rand(
        9,
        v4_flash_config.hidden_size,
        dtype=torch.float32,
        device=device,
    ).to(torch.bfloat16) * 0.1
    router_logits = _make_router_logits(
        num_tokens=hidden_states.shape[0],
        num_experts=v4_flash_config.n_routed_experts,
        topk=v4_flash_config.num_experts_per_tok,
        device=device,
        owned_start=moe.routed_expert_start_idx,
        owned_count=moe.routed_expert_end_idx - moe.routed_expert_start_idx,
        partial=True,
    )
    topk_weights, topk_indices = _router_topk_from_logits(
        router_logits,
        v4_flash_config.num_experts_per_tok,
    )

    monkeypatch.setenv("BATCHGEN_V4_RESIDENT_EXPERTS", "0")
    assert moe._grouped_staged is None

    first = _run_through_real_moe_path(
        moe,
        hidden_states,
        topk_weights,
        topk_indices,
        ragged_fallback=False,
    )
    assert moe._grouped_staged is not None

    for wrapper in moe.experts:
        wrapper._weights = None

    second = _run_through_real_moe_path(
        moe,
        hidden_states,
        topk_weights,
        topk_indices,
        ragged_fallback=False,
    )

    assert torch.allclose(first, second)
