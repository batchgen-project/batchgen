"""Parity: batchgen V4 sparse prefill attention vs official reference.

Builds the official Attention (assets/inference/model.py) and the batchgen
DeepSeekV4FlashAttention with IDENTICAL random weights, runs one prefill
sequence through both, and requires cosine > 0.999 on the attention output.
Requires CUDA + tilelang + fast_hadamard_transform (batchgen:v4-kernels-user).

Run:
  python -m pytest tests/integration/test_v4_prefill_sparse_parity.py -q
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ASSETS = (
    Path(__file__).resolve().parents[2]
    / "batchgen/models/deepseek/deepseekv4_flash/assets/inference"
)
sys.path.insert(0, str(ASSETS))

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="requires CUDA")

SEQLEN = 300
DIM = 512
N_HEADS = 8
HEAD_DIM = 256
ROPE_DIM = 64
O_GROUPS = 4
O_LORA = 128
Q_LORA = 128
WINDOW = 128
INDEX_HEADS = 8
INDEX_HEAD_DIM = 128
INDEX_TOPK = 32


def _official_args(compress_ratio: int):
    from model import ModelArgs

    return ModelArgs(
        max_batch_size=1,
        max_seq_len=2048,
        dtype="bf16",
        scale_fmt="ue8m0",
        scale_dtype="fp8",
        vocab_size=1024,
        dim=DIM,
        n_layers=1,
        n_heads=N_HEADS,
        q_lora_rank=Q_LORA,
        head_dim=HEAD_DIM,
        rope_head_dim=ROPE_DIM,
        o_groups=O_GROUPS,
        o_lora_rank=O_LORA,
        window_size=WINDOW,
        compress_ratios=(compress_ratio,),
        compress_rope_theta=160000.0,
        original_seq_len=65536,
        rope_theta=10000.0,
        rope_factor=16,
        beta_fast=32,
        beta_slow=1,
        index_n_heads=INDEX_HEADS,
        index_head_dim=INDEX_HEAD_DIM,
        index_topk=INDEX_TOPK,
    )


def _bg_config(compress_ratio: int):
    return SimpleNamespace(
        hidden_size=DIM,
        num_attention_heads=N_HEADS,
        head_dim=HEAD_DIM,
        q_lora_rank=Q_LORA,
        o_groups=O_GROUPS,
        o_lora_rank=O_LORA,
        rms_norm_eps=1e-6,
        compress_ratios=[compress_ratio],
        window_size=WINDOW,
        qk_rope_head_dim=ROPE_DIM,
        world_size=1,
        index_n_heads=INDEX_HEADS,
        index_head_dim=INDEX_HEAD_DIM,
        index_topk=INDEX_TOPK,
        compress_rope_theta=160000.0,
        original_seq_len=65536,
        rope_theta=10000.0,
        rope_factor=16,
        beta_fast=32,
        beta_slow=1,
    )


def _build_pair(compress_ratio: int):
    import model as official_model
    from batchgen.models.deepseek.deepseekv4_flash.model import (
        DeepSeekV4FlashAttention,
    )

    torch.manual_seed(7)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    official_model.default_dtype = torch.bfloat16
    official_model.scale_fmt = "ue8m0"
    official_model.scale_dtype = torch.float8_e8m0fnu

    ref = official_model.Attention(0, _official_args(compress_ratio)).cuda()
    for p in ref.parameters():
        if p.dtype.is_floating_point:
            torch.nn.init.normal_(p, std=0.02)
    if ref.compress_ratio:
        for comp in filter(
            None,
            [
                ref.compressor,
                getattr(ref.indexer, "compressor", None)
                if ref.indexer is not None
                else None,
            ],
        ):
            torch.nn.init.normal_(comp.ape, std=0.02)

    torch.set_default_device("cpu")
    bg = DeepSeekV4FlashAttention(_bg_config(compress_ratio), 0).cuda()
    bg.runtime_phase = "prefill"

    tensors = {
        "wq_a.weight": ref.wq_a.weight.data,
        "wq_b.weight": ref.wq_b.weight.data,
        "wkv.weight": ref.wkv.weight.data,
        "wo_a.weight": ref.wo_a.weight.data,
        "wo_b.weight": ref.wo_b.weight.data,
        "attn_sink": ref.attn_sink.data,
        "q_norm.weight": ref.q_norm.weight.data,
        "kv_norm.weight": ref.kv_norm.weight.data,
    }
    if compress_ratio:
        tensors.update(
            {
                "compressor.ape": ref.compressor.ape.data,
                "compressor.norm.weight": ref.compressor.norm.weight.data,
                "compressor.wkv.weight": ref.compressor.wkv.weight.data,
                "compressor.wgate.weight": ref.compressor.wgate.weight.data,
            }
        )
    if compress_ratio == 4:
        tensors.update(
            {
                "indexer.wq_b.weight": ref.indexer.wq_b.weight.data,
                "indexer.weights_proj.weight": (
                    ref.indexer.weights_proj.weight.data
                ),
                "indexer.compressor.ape": ref.indexer.compressor.ape.data,
                "indexer.compressor.norm.weight": (
                    ref.indexer.compressor.norm.weight.data
                ),
                "indexer.compressor.wkv.weight": (
                    ref.indexer.compressor.wkv.weight.data
                ),
                "indexer.compressor.wgate.weight": (
                    ref.indexer.compressor.wgate.weight.data
                ),
            }
        )
    bg.set_runtime_tensors(tensors)
    return ref, bg


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0)
    ).item()


@pytest.mark.parametrize("compress_ratio", [0, 128, 4])
def test_prefill_sparse_parity(compress_ratio):
    os.environ["BATCHGEN_V4_SPARSE_PREFILL"] = "1"
    ref, bg = _build_pair(compress_ratio)
    torch.manual_seed(11)
    x = torch.randn(1, SEQLEN, DIM, dtype=torch.bfloat16, device="cuda") * 0.5

    torch.set_default_device("cuda")
    with torch.inference_mode():
        ref_out = ref(x.clone(), start_pos=0)
        bg_out, _, bg_kv = bg._forward_prefill_sparse(x.clone())
    torch.set_default_device("cpu")

    cos_full = _cos(ref_out, bg_out)
    cos_last = _cos(ref_out[0, -1], bg_out[0, -1])
    print(
        f"ratio={compress_ratio} cos_full={cos_full:.6f} "
        f"cos_last={cos_last:.6f}"
    )
    assert cos_full > 0.999, f"full-seq cosine too low: {cos_full}"
    assert cos_last > 0.999, f"last-token cosine too low: {cos_last}"
