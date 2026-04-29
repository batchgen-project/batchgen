"""DeepSeek-V4 Flash reference sanity tests.

These CPU tests load the vendored ``assets/inference/model.py`` with a small
kernel stub, then compare BatchGen's V4 fallback math against the reference
module piece by piece. They intentionally use tiny dimensions and world_size=1;
distributed EP behavior is validated separately on H20.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from batchgen.models.deepseek.deepseekv4_flash import model as bg
from batchgen.models.deepseek.deepseekv4_flash.wrappers import (
    DeepSeekV4FlashAttnWrapper,
)
from batchgen.models.wrappers.attention import AttnWrapperBase


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_MODEL = (
    ROOT / "batchgen/models/deepseek/deepseekv4_flash/assets/inference/model.py"
)


def _kernel_hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    pre = torch.sigmoid(mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(
        mixes[..., hc_mult : 2 * hc_mult] * hc_scale[1]
        + hc_base[hc_mult : 2 * hc_mult]
    )
    comb = mixes[..., 2 * hc_mult :].view(*mixes.shape[:-1], hc_mult, hc_mult)
    comb_base = hc_base[2 * hc_mult :].view(hc_mult, hc_mult)
    comb = torch.softmax(comb * hc_scale[2] + comb_base, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def _kernel_sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    bsz, seqlen, n_heads, head_dim = q.shape
    out = torch.zeros_like(q)
    for b in range(bsz):
        for s in range(seqlen):
            valid = topk_idxs[b, s][topk_idxs[b, s] >= 0].long()
            if valid.numel() == 0:
                continue
            keys = kv[b, valid].float()
            scores = torch.einsum("hd,kd->hk", q[b, s].float(), keys) * softmax_scale
            sink = attn_sink.float().to(scores.device).view(n_heads, 1)
            scores_max = torch.maximum(scores.max(dim=-1, keepdim=True).values, sink)
            exp_scores = torch.exp(scores - scores_max)
            denom = exp_scores.sum(dim=-1, keepdim=True) + torch.exp(sink - scores_max)
            probs = (exp_scores / denom).to(q.dtype)
            out[b, s] = torch.einsum("hk,kd->hd", probs, kv[b, valid])
    return out


def _install_reference_kernel_stub() -> None:
    kernel = types.ModuleType("kernel")
    kernel.act_quant = lambda x, *args, **kwargs: (x, torch.ones((), device=x.device))
    kernel.fp4_act_quant = lambda x, *args, **kwargs: x
    kernel.fp8_gemm = lambda x, scale, weight, weight_scale, scale_dtype=None: F.linear(
        x, weight.to(dtype=x.dtype)
    )
    kernel.fp4_gemm = lambda x, scale, weight, weight_scale, scale_dtype=None: F.linear(
        x, weight.to(dtype=x.dtype)
    )
    kernel.sparse_attn = _kernel_sparse_attn
    kernel.hc_split_sinkhorn = _kernel_hc_split_sinkhorn
    sys.modules["kernel"] = kernel


@pytest.fixture(scope="module")
def ref():
    _install_reference_kernel_stub()
    spec = importlib.util.spec_from_file_location("deepseek_v4_flash_reference", REFERENCE_MODEL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.default_dtype = torch.float32
    module.scale_fmt = None
    module.scale_dtype = torch.float32
    return module


def _tiny_args(ref, *, compress_ratio: int = 0, n_hash_layers: int = 0):
    return ref.ModelArgs(
        max_batch_size=4,
        max_seq_len=32,
        dtype="bf16",
        scale_fmt=None,
        expert_dtype=None,
        scale_dtype="fp32",
        vocab_size=32,
        dim=8,
        moe_inter_dim=6,
        n_layers=1,
        n_hash_layers=n_hash_layers,
        n_mtp_layers=0,
        n_heads=2,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        score_func="sqrtsoftplus",
        route_scale=1.5,
        swiglu_limit=10.0,
        q_lora_rank=6,
        head_dim=4,
        rope_head_dim=2,
        norm_eps=1e-6,
        o_groups=1,
        o_lora_rank=5,
        window_size=16,
        compress_ratios=(compress_ratio,),
        compress_rope_theta=40000.0,
        original_seq_len=0,
        rope_theta=10000.0,
        rope_factor=16,
        beta_fast=32,
        beta_slow=1,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=4,
        hc_mult=3,
        hc_sinkhorn_iters=5,
        hc_eps=1e-6,
    )


def _tiny_config(args) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=args.dim,
        dim=args.dim,
        vocab_size=args.vocab_size,
        num_hidden_layers=args.n_layers,
        n_layers=args.n_layers,
        num_attention_heads=args.n_heads,
        n_heads=args.n_heads,
        head_dim=args.head_dim,
        qk_rope_head_dim=args.rope_head_dim,
        rope_head_dim=args.rope_head_dim,
        q_lora_rank=args.q_lora_rank,
        o_groups=args.o_groups,
        o_lora_rank=args.o_lora_rank,
        compress_ratios=list(args.compress_ratios),
        compress_rope_theta=args.compress_rope_theta,
        rope_theta=args.rope_theta,
        rope_scaling={
            "factor": args.rope_factor,
            "beta_fast": args.beta_fast,
            "beta_slow": args.beta_slow,
            "original_max_position_embeddings": args.original_seq_len,
        },
        rms_norm_eps=args.norm_eps,
        norm_eps=args.norm_eps,
        n_routed_experts=args.n_routed_experts,
        num_local_experts=args.n_routed_experts,
        num_experts_per_tok=args.n_activated_experts,
        n_activated_experts=args.n_activated_experts,
        num_hash_layers=args.n_hash_layers,
        n_hash_layers=args.n_hash_layers,
        routed_scaling_factor=args.route_scale,
        route_scale=args.route_scale,
        norm_topk_prob=True,
        scoring_func=args.score_func,
        score_func=args.score_func,
        moe_intermediate_size=args.moe_inter_dim,
        moe_inter_dim=args.moe_inter_dim,
        swiglu_limit=args.swiglu_limit,
        hc_mult=args.hc_mult,
        hc_sinkhorn_iters=args.hc_sinkhorn_iters,
        hc_eps=args.hc_eps,
        pad_token_id=1,
    )


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, *, atol=1e-5, rtol=1e-5):
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=atol,
        rtol=rtol,
        check_dtype=False,
    )


def _fill_float_param(param: torch.Tensor, scale: float = 0.2) -> None:
    param.copy_((torch.randn_like(param.float()) * scale).to(dtype=param.dtype))


def _copy_ref_expert_to_bg(ref_expert, bg_expert) -> None:
    bg_expert.set_runtime_tensors(
        {
            "w1.weight": ref_expert.w1.weight.detach().clone(),
            "w2.weight": ref_expert.w2.weight.detach().clone(),
            "w3.weight": ref_expert.w3.weight.detach().clone(),
        }
    )


def _force_reference_attention_fp32(ref_attn) -> None:
    for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
        layer = getattr(ref_attn, name)
        layer.weight.data = layer.weight.data.float()


def _init_reference_attention(ref_attn) -> None:
    _force_reference_attention_fp32(ref_attn)
    with torch.no_grad():
        for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            _fill_float_param(getattr(ref_attn, name).weight)
        _fill_float_param(ref_attn.attn_sink)
        _fill_float_param(ref_attn.q_norm.weight)
        _fill_float_param(ref_attn.kv_norm.weight)


def _copy_ref_attention_to_bg(ref_attn, bg_attn) -> None:
    bg_attn.set_runtime_tensors(
        {
            "attn_sink": ref_attn.attn_sink.detach().clone(),
            "q_norm.weight": ref_attn.q_norm.weight.detach().clone(),
            "kv_norm.weight": ref_attn.kv_norm.weight.detach().clone(),
            "wq_a.weight": ref_attn.wq_a.weight.detach().clone(),
            "wq_b.weight": ref_attn.wq_b.weight.detach().clone(),
            "wkv.weight": ref_attn.wkv.weight.detach().clone(),
            "wo_a.weight": ref_attn.wo_a.weight.detach().clone(),
            "wo_b.weight": ref_attn.wo_b.weight.detach().clone(),
        }
    )


class _RecordingHostView:
    def __init__(self):
        self.offloads = []

    def async_offload_layer_kv_to_host(self, **kwargs):
        self.offloads.append(kwargs)
        return None


class _CoreStub:
    def __init__(self):
        self.host_paged_kv_worker_view = _RecordingHostView()


class _FakePagedKVManager:
    def __init__(self, prefill_kv: torch.Tensor, page_size: int, max_len: int):
        bsz, prefill_len, head_dim = prefill_kv.shape
        self.config = SimpleNamespace(page_size_tokens=page_size)
        self.device = prefill_kv.device
        num_pages = (max_len + page_size - 1) // page_size
        total_pages = bsz * num_pages
        self.k_cache = prefill_kv.new_zeros(total_pages, page_size, 1, head_dim)
        self.page_table = torch.arange(total_pages, dtype=torch.int32).view(
            bsz, num_pages
        )
        for batch_idx in range(bsz):
            for token_idx in range(prefill_len):
                page = int(self.page_table[batch_idx, token_idx // page_size].item())
                offset = token_idx % page_size
                self.k_cache[page, offset, 0] = prefill_kv[batch_idx, token_idx]
        self.updated = []

    def get_layer_kv_with_page_table(self, layer_idx: int):
        del layer_idx
        return self.k_cache, None, self.page_table

    def update_layer_decode_new_token(
        self,
        k_tensor: torch.Tensor,
        v_tensor,
        sequence_lengths: torch.Tensor,
        layer_idx: int,
        batch_slice=None,
        slot_indices=None,
    ) -> None:
        del v_tensor, layer_idx, batch_slice, slot_indices
        for batch_idx, pos in enumerate(sequence_lengths.cpu().tolist()):
            page = int(
                self.page_table[
                    batch_idx,
                    pos // self.config.page_size_tokens,
                ].item()
            )
            offset = pos % self.config.page_size_tokens
            self.k_cache[page, offset, 0] = k_tensor[batch_idx, 0, 0]
        self.updated.append(k_tensor.detach().clone())


def _reset_attn_wrapper_state() -> None:
    AttnWrapperBase.phase = "prefill"
    AttnWrapperBase.prepack_mode = False
    AttnWrapperBase.prepack_cu_seqlens = None
    AttnWrapperBase.prepack_max_seqlen = None
    AttnWrapperBase.prepack_num_sequences = None
    AttnWrapperBase.prepack_seq_lengths = None
    AttnWrapperBase.position_ids = None
    AttnWrapperBase.cache_seqlens = None
    AttnWrapperBase.max_seqlen = None
    AttnWrapperBase.cur_batch = None
    AttnWrapperBase.past_key_states = None
    AttnWrapperBase.gpu_paged_kv_manager = None
    AttnWrapperBase.kv_append_callback = None
    AttnWrapperBase.pending_prefill_offload_tasks.clear()
    AttnWrapperBase.pending_prefill_offload_tensors.clear()


def test_rope_matches_reference(ref):
    torch.manual_seed(0)
    positions = torch.arange(3, 8)
    x_ref = torch.randn(1, positions.numel(), 2, 4)
    x_bg = x_ref.clone()

    freqs_ref = ref.precompute_freqs_cis(2, 16, 0, 10000.0, 16, 32, 1)[positions]
    ref.apply_rotary_emb(x_ref[..., -2:], freqs_ref)

    freqs_bg = bg._build_rope_freqs_cis(positions, 2, 10000.0, 16, 32, 1, 0)
    bg._apply_rotary_emb_inplace(x_bg[..., -2:], freqs_bg)

    _assert_close(x_bg, x_ref)


def test_hc_pre_post_matches_reference(ref):
    torch.manual_seed(1)
    args = _tiny_args(ref)
    cfg = _tiny_config(args)
    ref_block = ref.Block(0, args)
    bg_layer = bg.DeepSeekV4FlashDecoderLayer(cfg, 0)

    with torch.no_grad():
        for param in (
            ref_block.hc_attn_fn,
            ref_block.hc_attn_base,
            ref_block.hc_attn_scale,
        ):
            _fill_float_param(param)
        bg_layer.hc_attn_fn.copy_(ref_block.hc_attn_fn)
        bg_layer.hc_attn_base.copy_(ref_block.hc_attn_base)
        bg_layer.hc_attn_scale.copy_(ref_block.hc_attn_scale)

    x = torch.randn(2, 3, args.hc_mult, args.dim)
    ref_pre, ref_post, ref_comb = ref_block.hc_pre(
        x, ref_block.hc_attn_fn, ref_block.hc_attn_scale, ref_block.hc_attn_base
    )
    bg_pre, bg_post, bg_comb = bg_layer._hc_pre(
        x, bg_layer.hc_attn_fn, bg_layer.hc_attn_scale, bg_layer.hc_attn_base
    )
    _assert_close(bg_pre, ref_pre)
    _assert_close(bg_post, ref_post)
    _assert_close(bg_comb, ref_comb)

    y = torch.randn(2, 3, args.dim)
    _assert_close(
        bg_layer._hc_post(y, x, bg_post, bg_comb),
        ref_block.hc_post(y, x, ref_post, ref_comb),
    )


@pytest.mark.parametrize("n_hash_layers", [0, 1])
def test_gate_matches_reference(ref, n_hash_layers):
    torch.manual_seed(2 + n_hash_layers)
    args = _tiny_args(ref, n_hash_layers=n_hash_layers)
    cfg = _tiny_config(args)
    ref_gate = ref.Gate(0, args)
    bg_gate = bg.DeepSeekV4FlashGate(cfg, 0)

    with torch.no_grad():
        _fill_float_param(ref_gate.weight)
        bg_gate.weight.copy_(ref_gate.weight)
        if n_hash_layers:
            ids = torch.tensor(
                [[0, 1], [2, 3], [1, 0], [3, 2], *([[0, 2]] * (args.vocab_size - 4))],
                dtype=torch.long,
            )
            ref_gate.tid2eid.copy_(ids.to(dtype=ref_gate.tid2eid.dtype))
            bg_gate.tid2eid.copy_(ids)
        else:
            _fill_float_param(ref_gate.bias)
            bg_gate.bias.copy_(ref_gate.bias)

    x = torch.randn(4, args.dim)
    input_ids = torch.tensor([0, 1, 2, 3])
    ref_weights, ref_indices = ref_gate(x, input_ids)
    bg_weights, bg_indices = bg_gate(x, input_ids)

    assert torch.equal(bg_indices.cpu(), ref_indices.long().cpu())
    _assert_close(bg_weights, ref_weights)


def test_expert_matches_reference(ref):
    torch.manual_seed(4)
    args = _tiny_args(ref)
    ref_expert = ref.Expert(args.dim, args.moe_inter_dim, dtype=torch.float32, swiglu_limit=args.swiglu_limit)
    bg_expert = bg.DeepSeekV4FlashExpertPlaceholder(args.dim, args.moe_inter_dim, args.swiglu_limit)
    with torch.no_grad():
        for layer in (ref_expert.w1, ref_expert.w2, ref_expert.w3):
            _fill_float_param(layer.weight)
    _copy_ref_expert_to_bg(ref_expert, bg_expert)

    x = torch.randn(5, args.dim)
    weights = torch.rand(5, 1)
    _assert_close(bg_expert(x), ref_expert(x))
    _assert_close(bg_expert(x, weights), ref_expert(x, weights))


def test_moe_single_rank_matches_reference(ref):
    torch.manual_seed(5)
    args = _tiny_args(ref)
    cfg = _tiny_config(args)
    ref_moe = ref.MoE(0, args)
    bg_moe = bg.DeepSeekV4FlashMoE(cfg, 0)

    with torch.no_grad():
        _fill_float_param(ref_moe.gate.weight)
        _fill_float_param(ref_moe.gate.bias)
        bg_moe.gate.weight.copy_(ref_moe.gate.weight)
        bg_moe.gate.bias.copy_(ref_moe.gate.bias)
        for expert in list(ref_moe.experts) + [ref_moe.shared_experts]:
            for layer in (expert.w1, expert.w2, expert.w3):
                _fill_float_param(layer.weight)
    for idx in range(args.n_routed_experts):
        _copy_ref_expert_to_bg(ref_moe.experts[idx], bg_moe.experts[idx])
    _copy_ref_expert_to_bg(ref_moe.shared_experts, bg_moe.shared_experts)

    x = torch.randn(2, 3, args.dim)
    input_ids = torch.randint(0, args.vocab_size, (2, 3))
    _assert_close(bg_moe(x, input_ids), ref_moe(x, input_ids), atol=1e-4, rtol=1e-4)


def test_attention_cr0_prefill_matches_reference(ref):
    torch.manual_seed(6)
    args = _tiny_args(ref, compress_ratio=0)
    cfg = _tiny_config(args)
    ref_attn = ref.Attention(0, args)
    _init_reference_attention(ref_attn)
    bg_attn = bg.DeepSeekV4FlashAttention(cfg, 0)
    _copy_ref_attention_to_bg(ref_attn, bg_attn)

    x = torch.randn(2, 5, args.dim)
    position_ids = torch.arange(x.size(1)).unsqueeze(0).expand(x.size(0), -1)
    ref_out = ref_attn(x, start_pos=0)
    bg_out, _, _ = bg_attn(x, position_ids=position_ids)

    _assert_close(bg_out, ref_out, atol=1e-4, rtol=1e-4)


def test_attention_cr0_decode_matches_reference(ref):
    torch.manual_seed(7)
    args = _tiny_args(ref, compress_ratio=0)
    cfg = _tiny_config(args)
    ref_attn = ref.Attention(0, args)
    _init_reference_attention(ref_attn)
    bg_attn = bg.DeepSeekV4FlashAttention(cfg, 0)
    _copy_ref_attention_to_bg(ref_attn, bg_attn)

    bsz = 2
    prefill_len = 5
    x_prefill = torch.randn(bsz, prefill_len, args.dim)
    x_decode = torch.randn(bsz, 1, args.dim)

    ref_attn(x_prefill, start_pos=0)
    ref_decode = ref_attn(x_decode, start_pos=prefill_len)

    prefill_positions = torch.arange(prefill_len).unsqueeze(0).expand(bsz, -1)
    _, _, bg_prefill_kv = bg_attn(x_prefill, position_ids=prefill_positions)
    bg_past = x_prefill.new_zeros(bsz, args.window_size, args.head_dim)
    bg_past[:, :prefill_len] = bg_prefill_kv
    bg_decode, _, _ = bg_attn(
        x_decode,
        position_ids=torch.full((bsz, 1), prefill_len, dtype=torch.long),
        past_key_value=bg_past,
        cache_seqlens=torch.full((bsz,), prefill_len + 1, dtype=torch.long),
    )

    _assert_close(bg_decode, ref_decode, atol=1e-4, rtol=1e-4)


def test_attention_prepacked_prefill_is_sequence_local(ref):
    torch.manual_seed(8)
    args = _tiny_args(ref, compress_ratio=0)
    cfg = _tiny_config(args)
    ref_attn = ref.Attention(0, args)
    _init_reference_attention(ref_attn)
    bg_attn = bg.DeepSeekV4FlashAttention(cfg, 0)
    _copy_ref_attention_to_bg(ref_attn, bg_attn)
    core = _CoreStub()
    wrapper = DeepSeekV4FlashAttnWrapper(
        bg_attn,
        layer_idx=0,
        core_engine=core,
        engine_config=SimpleNamespace(),
        model_config=SimpleNamespace(),
        persistent=True,
    )

    seq_a = torch.randn(1, 3, args.dim)
    seq_b = torch.randn(1, 5, args.dim)
    packed = torch.cat([seq_a, seq_b], dim=1)

    _reset_attn_wrapper_state()
    AttnWrapperBase.phase = "prefill"
    AttnWrapperBase.prepack_mode = True
    AttnWrapperBase.prepack_cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)
    AttnWrapperBase.prepack_max_seqlen = 5
    AttnWrapperBase.prepack_num_sequences = 2
    AttnWrapperBase.position_ids = torch.cat(
        [torch.arange(3), torch.arange(5)]
    ).to(torch.long)
    AttnWrapperBase.cur_batch = [11, 12]
    try:
        wrapped_out, _, _ = wrapper(packed)
    finally:
        _reset_attn_wrapper_state()

    expected_a = ref_attn(seq_a, start_pos=0)
    expected_b = ref_attn(seq_b, start_pos=0)
    expected = torch.cat([expected_a, expected_b], dim=1)

    _assert_close(wrapped_out, expected, atol=1e-4, rtol=1e-4)
    assert len(core.host_paged_kv_worker_view.offloads) == 2
    assert core.host_paged_kv_worker_view.offloads[0]["sequence_lengths"] == [3]
    assert core.host_paged_kv_worker_view.offloads[1]["sequence_lengths"] == [5]


def test_attention_decode_reads_and_updates_paged_kv(ref):
    torch.manual_seed(9)
    args = _tiny_args(ref, compress_ratio=0)
    cfg = _tiny_config(args)
    ref_attn = ref.Attention(0, args)
    _init_reference_attention(ref_attn)
    bg_attn = bg.DeepSeekV4FlashAttention(cfg, 0)
    _copy_ref_attention_to_bg(ref_attn, bg_attn)
    wrapper = DeepSeekV4FlashAttnWrapper(
        bg_attn,
        layer_idx=0,
        core_engine=_CoreStub(),
        engine_config=SimpleNamespace(),
        model_config=SimpleNamespace(),
        persistent=True,
    )

    bsz = 2
    prefill_len = 5
    x_prefill = torch.randn(bsz, prefill_len, args.dim)
    x_decode = torch.randn(bsz, 1, args.dim)
    ref_attn(x_prefill, start_pos=0)
    ref_decode = ref_attn(x_decode, start_pos=prefill_len)

    prefill_positions = torch.arange(prefill_len).unsqueeze(0).expand(bsz, -1)
    _, _, bg_prefill_kv = bg_attn(x_prefill, position_ids=prefill_positions)
    manager = _FakePagedKVManager(
        bg_prefill_kv,
        page_size=4,
        max_len=prefill_len + 1,
    )
    appended = []

    _reset_attn_wrapper_state()
    AttnWrapperBase.phase = "decode"
    AttnWrapperBase.gpu_paged_kv_manager = manager
    AttnWrapperBase.cache_seqlens = torch.full(
        (bsz,),
        prefill_len + 1,
        dtype=torch.int32,
    )
    AttnWrapperBase.position_ids = torch.full((bsz, 1), prefill_len, dtype=torch.long)
    AttnWrapperBase.kv_append_callback = (
        lambda layer_idx, k_tensor, v_tensor=None: appended.append(
            (layer_idx, k_tensor.detach().clone(), v_tensor)
        )
    )
    try:
        wrapped_decode, _, _ = wrapper(x_decode)
    finally:
        _reset_attn_wrapper_state()

    _assert_close(wrapped_decode, ref_decode, atol=1e-4, rtol=1e-4)
    assert len(manager.updated) == 1
    assert len(appended) == 1
    assert appended[0][1].shape == (bsz, 1, 1, args.head_dim)
