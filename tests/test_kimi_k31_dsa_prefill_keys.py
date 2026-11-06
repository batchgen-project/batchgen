# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3.1                                                        #
#  Copyright (c) 2025-2026 BatchGen Team                                       #
#  Licensed under the Apache License, Version 2.0                              #
# ---------------------------------------------------------------------------- #
"""Slice 1 of the Kimi-K3.1 DSA serving path: index-path couplings + prefill keys.

Route B: page-split FP8 index keys, scored by the ReLU-gated DeepGEMM paged
scorer. On this route the index path applies no transform (identity RoPE, no
Hadamard), and every producer of an index vector must agree on that.

CPU-provable (asserted below):
  * the couplings as DATA, and that all three producers (prefill K, decode K,
    Q) refuse together if the shared rotation predicate ever changes;
  * the decode writer's rotation arguments satisfy EVERY precondition of the
    fused kernel (``plan_norm_rope_write``, the kernel's own CPU statement);
  * prefill writes the SAME pre-quantize keys the fused decode writer would
    (``indexer_k_norm_rope_reference``, the kernel's own CPU oracle), and
    quantizes them with the shipped page-split recipe;
  * the packed Host write: e4m3 keys + fp32 scales, shapes, pin/track/wait,
    and every refusal;
  * the serving model owns the SAME indexer class the oracle tests, under the
    checkpoint's names, replicated under attention TP;
  * every not-yet-sparse DSA path refuses instead of running dense.

GPU-deferred (validated on-node):
  * the compiled fused writer equals ``indexer_k_norm_rope_reference`` +
    ``quantize_indexer_k_fp8`` within reduction-order rounding;
  * the aux worker view lands the page-split bytes + scales at the right
    physical rows of the ``kimi_k31_indexer`` region.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# kimi_linear/model.py imports fla at module scope; the K3 harness ships a CPU
# shim for exactly this.
_shim = _load_by_path(
    "_k31_fla_cpu_shim", ROOT / "tests" / "kimi_k3_oracle_assets" / "fla_cpu_shim.py")
_shim.install(force=True)

from batchgen.models.moonshotai.kimi_k3.dsa_indexer import KimiK3Indexer  # noqa: E402
from batchgen.models.moonshotai.kimi_linear import (  # noqa: E402
    dsa_index_rotation as rot,
    dsa_prefill,
)
from batchgen.models.moonshotai.kimi_linear.config import KimiLinearConfig  # noqa: E402
from batchgen.models.moonshotai.kimi_linear.tp_weight_sharding import (  # noqa: E402
    shard_mla_tensor,
)
from batchgen.models.wrappers import AttnWrapperBase  # noqa: E402

#: The shipped page-split helpers, by file path: the
#: ``batchgen_kernels.attention.dsa`` package ``__init__`` imports triton.
QUANT = _load_by_path(
    "_k31_indexer_k_quant",
    ROOT / "batchgen_kernels" / "attention" / "dsa" / "indexer_k_quant.py",
)


@pytest.fixture(autouse=True)
def _shipped_kernel_helpers(monkeypatch):
    """Route the module's seams to the REAL shipped functions."""
    monkeypatch.setattr(
        rot, "_hadamard_predicate", lambda: QUANT.indexer_applies_hadamard)
    monkeypatch.setattr(
        dsa_prefill, "_quantize_indexer_k_fp8", lambda: QUANT.quantize_indexer_k_fp8)


# --------------------------------------------------------------------------- #
#  Couplings as data                                                          #
# --------------------------------------------------------------------------- #

def test_identity_tables():
    cos, sin = rot.identity_rope_tables("cpu")
    assert cos.shape == sin.shape == (1, rot.INDEX_ROPE_TABLE_DIM) == (1, 64)
    assert cos.dtype is sin.dtype is torch.float32
    assert torch.equal(cos, torch.ones(1, 64))
    assert torch.equal(sin, torch.zeros(1, 64))
    assert cos.is_contiguous() and sin.is_contiguous()
    assert cos.stride(0) == sin.stride(0) == 64


def test_identity_tables_are_cached_per_device():
    first = rot.identity_rope_tables("cpu")
    second = rot.identity_rope_tables(torch.device("cpu"))
    assert first[0] is second[0] and first[1] is second[1]


def test_the_page_split_route_drops_the_hadamard(monkeypatch):
    seen = []

    def predicate(*, page_split):
        seen.append(page_split)
        return QUANT.indexer_applies_hadamard(page_split=page_split)

    monkeypatch.setattr(rot, "_hadamard_predicate", lambda: predicate)
    assert rot.index_applies_hadamard() is False
    assert seen == [True]
    assert rot.K31_INDEX_CACHE_PAGE_SPLIT is True


def test_decode_writer_rotation_args():
    args = rot.decode_writer_rotation_args(5, "cpu")
    cos, sin = rot.identity_rope_tables("cpu")
    assert args["cos_cache"] is cos and args["sin_cache"] is sin
    assert args["positions"].dtype is torch.int64
    assert torch.equal(args["positions"], torch.zeros(5, dtype=torch.int64))
    assert args["apply_hadamard"] is False


@pytest.mark.parametrize("producer", ["q", "prefill_k", "decode_k"])
def test_all_three_producers_refuse_together_if_the_predicate_rotates(monkeypatch, producer):
    """If page-split records ever became rotated, the Q producer and the
    prefill writer (which skip the rotation) would score a different space
    from the decode writer (which honours the flag). All three stop."""
    monkeypatch.setattr(rot, "_hadamard_predicate",
                        lambda: (lambda *, page_split: True))
    call = {
        "q": lambda: rot.index_q_rows(torch.randn(2, 4, 128)),
        "prefill_k": lambda: rot.prefill_index_k_rows(torch.randn(3, 128)),
        "decode_k": lambda: rot.decode_writer_rotation_args(3, "cpu"),
    }[producer]
    with pytest.raises(RuntimeError, match="different spaces"):
        call()


def test_q_rows_pass_through_unchanged():
    q = torch.randn(3, 4, 128)
    assert rot.index_q_rows(q) is q


@pytest.mark.parametrize("bad", [64, 256])
def test_only_the_128_wide_row_is_accepted(bad):
    with pytest.raises(NotImplementedError, match="128-wide"):
        rot.check_index_head_dim(bad)
    with pytest.raises(NotImplementedError):
        rot.index_q_rows(torch.randn(2, 4, bad))


def test_the_reference_transform_is_the_identity_on_this_route():
    rows = torch.randn(17, 128, generator=torch.Generator().manual_seed(1))
    assert torch.equal(rot.rotate_index_rows_reference(rows), rows.float())


# --------------------------------------------------------------------------- #
#  Decode writer launch preconditions (the kernel's own CPU statement)        #
# --------------------------------------------------------------------------- #

def _page_split_cache(num_pages=4, page_size=64):
    """A page-split aux cache exactly as the KV manager allocates it:
    ``[num_pages, page_size, 1, 128 + 4]`` with a 1-byte element type."""
    return torch.zeros(num_pages, page_size, 1, 128 + 4, dtype=torch.uint8)


def test_decode_writer_args_satisfy_every_kernel_precondition():
    n = 6
    k_raw = torch.randn(n, 128).to(torch.bfloat16).contiguous()
    norm = torch.nn.LayerNorm(128).to(torch.bfloat16)
    args = rot.decode_writer_rotation_args(n, "cpu")
    plan = QUANT.plan_norm_rope_write(
        _page_split_cache(), k_raw, norm.weight.data, norm.bias.data,
        args["cos_cache"], args["sin_cache"], args["positions"], 64,
    )
    assert plan.num_tokens == n
    assert plan.data_dim == 128
    assert plan.rope_dim == 64


def test_the_kernel_would_reject_a_wrong_table_dtype():
    """Negative control: the validator really checks what we rely on."""
    k_raw = torch.randn(2, 128).to(torch.bfloat16)
    norm = torch.nn.LayerNorm(128).to(torch.bfloat16)
    cos, sin = rot.identity_rope_tables("cpu")
    with pytest.raises(ValueError, match="float32"):
        QUANT.plan_norm_rope_write(
            _page_split_cache(), k_raw, norm.weight.data, norm.bias.data,
            cos.to(torch.bfloat16), sin, torch.zeros(2, dtype=torch.int64), 64,
        )


# --------------------------------------------------------------------------- #
#  The serving model owns the shared indexer                                  #
# --------------------------------------------------------------------------- #

def _k31_serving_config() -> KimiLinearConfig:
    """A small K3.1 serving config: 2 layers, layer 1 full-attention + DSA."""
    text = {
        "vocab_size": 256, "hidden_size": 64, "intermediate_size": 128,
        "num_hidden_layers": 2, "rms_norm_eps": 1e-5,
        "num_attention_heads": 4, "num_key_value_heads": 4,
        "q_lora_rank": 32, "kv_lora_rank": 48,
        "qk_nope_head_dim": 16, "qk_rope_head_dim": 8, "v_head_dim": 16,
        "mla_use_nope": True, "mla_use_output_gate": True,
        "linear_attn_config": {
            "kda_layers": [1], "full_attn_layers": [2], "num_heads": 4,
            "head_dim": 16, "short_conv_kernel_size": 4,
            "use_full_rank_gate": True, "gate_lower_bound": -5.0,
        },
        "dsa_config": {
            "dsa_layers": [2], "index_topk": 8, "index_head_dim": 128,
            "index_num_heads": 4, "index_rope_dim": 0,
            "index_k_norm_type": "layernorm", "index_topk_freq": 1,
            "index_skip_topk_offset": 1,
        },
    }
    return KimiLinearConfig.from_hf_dict(
        {"model_type": "kimi_k31", "text_config": text})


def _serving_mla(config, layer_idx=1):
    from batchgen.models.moonshotai.kimi_linear.model import KimiMLAAttention

    return KimiMLAAttention(config, layer_idx).eval()


def test_serving_mla_on_a_dsa_layer_owns_the_shared_indexer_class():
    attn = _serving_mla(_k31_serving_config())
    assert isinstance(attn.indexer, KimiK3Indexer)


def test_serving_indexer_names_equal_the_weight_map_names():
    from batchgen.models.moonshotai.kimi_linear.k3.tensor_map import (
        K3_MLA_INDEXER_TENSOR_NAMES,
    )

    attn = _serving_mla(_k31_serving_config())
    names = {n for n, _ in attn.named_parameters() if n.startswith("indexer.")}
    assert names == set(K3_MLA_INDEXER_TENSOR_NAMES)


def test_serving_indexer_exposes_what_the_worker_duck_types():
    attn = _serving_mla(_k31_serving_config())
    assert attn.indexer.index_head_dim == 128
    assert attn.indexer.index_topk == 8


def test_dense_k3_layers_have_no_indexer():
    k3 = _k31_serving_config()
    k3.model_type = "kimi_k3"
    k3.dsa_config = None
    attn = _serving_mla(k3)
    assert not hasattr(attn, "indexer")
    assert not any(n.startswith("indexer.") for n, _ in attn.named_parameters())


def test_serving_indexer_refuses_a_non_128_index_head_dim():
    cfg = _k31_serving_config()
    cfg.dsa_config = dict(cfg.dsa_config, index_head_dim=64)
    with pytest.raises(NotImplementedError, match="128-wide"):
        _serving_mla(cfg)


def test_the_dense_eager_forward_refuses_a_dsa_layer():
    attn = _serving_mla(_k31_serving_config())
    with pytest.raises(NotImplementedError, match="dense"):
        attn(torch.randn(1, 3, 64))


@pytest.mark.parametrize("name", [
    "indexer.wq_b.weight", "indexer.wk.weight", "indexer.weights_proj.weight",
    "indexer.k_norm.weight", "indexer.k_norm.bias",
])
def test_indexer_is_replicated_under_attention_tp(name):
    tensor = torch.randn(32, 16)
    for rank in range(4):
        assert torch.equal(shard_mla_tensor(tensor, name, 4, rank), tensor)


def test_mla_head_sharding_is_unchanged():
    tensor = torch.arange(32 * 4, dtype=torch.float32).reshape(32, 4)
    assert shard_mla_tensor(tensor, "q_b_proj.weight", 4, 1).shape == (8, 4)
    assert shard_mla_tensor(tensor, "o_proj.weight", 2, 0).shape == (32, 2)
    with pytest.raises(ValueError, match="unsupported MLA tensor"):
        shard_mla_tensor(tensor, "mystery.weight", 4, 0)


# --------------------------------------------------------------------------- #
#  Prefill index keys                                                         #
# --------------------------------------------------------------------------- #

def _indexer():
    attn = _serving_mla(_k31_serving_config())
    torch.manual_seed(5)
    with torch.no_grad():
        for p in attn.indexer.parameters():
            p.normal_(0.0, 0.3)
    return attn.indexer.to(torch.bfloat16)


def test_prefill_keys_are_the_bf16_layernorm_keys():
    indexer = _indexer()
    hidden = torch.randn(11, 64).to(torch.bfloat16)
    keys = dsa_prefill.prefill_index_keys(indexer, hidden)
    assert keys.shape == (11, 128) and keys.dtype is torch.bfloat16
    with torch.no_grad():
        assert torch.equal(keys, indexer.k_norm(indexer.wk(hidden)))


def test_prefill_writes_what_the_fused_decode_writer_would_write():
    """Cross-path consistency: decode reads prefill-written keys next to the
    keys it writes itself, so both must be the same pre-quantize values. The
    fused writer's own CPU oracle, fed the decode rotation arguments, must
    reproduce prefill's keys (to bf16 rounding of the LayerNorm reduction)."""
    indexer = _indexer()
    hidden = torch.randn(9, 64).to(torch.bfloat16)
    keys = dsa_prefill.prefill_index_keys(indexer, hidden)
    with torch.no_grad():
        k_raw = indexer.wk(hidden).contiguous()
    args = rot.decode_writer_rotation_args(9, "cpu")
    decode_view = QUANT.indexer_k_norm_rope_reference(
        k_raw, indexer.k_norm.weight.data, indexer.k_norm.bias.data,
        indexer.k_norm.eps, args["cos_cache"], args["sin_cache"],
        args["positions"], args["apply_hadamard"],
    )
    assert decode_view.dtype is torch.bfloat16
    diff = (decode_view.float() - keys.float()).abs()
    ulp = keys.float().abs().clamp(min=1e-3) * 2.0 ** -7
    assert bool((diff <= ulp).all()), float(diff.max())


def test_prefill_keys_refuse_an_unpacked_hidden():
    with pytest.raises(ValueError, match="packed"):
        dsa_prefill.prefill_index_keys(_indexer(), torch.randn(1, 4, 64))


def test_quantization_is_the_shipped_page_split_recipe():
    keys = torch.randn(7, 128).to(torch.bfloat16)
    k_fp8, scales = dsa_prefill.quantize_index_keys(keys)
    assert k_fp8.shape == (7, 1, 128) and k_fp8.dtype is torch.float8_e4m3fn
    assert scales.shape == (7, 1) and scales.dtype is torch.float32
    ref_fp8, ref_scales = QUANT.quantize_indexer_k_fp8(keys.unsqueeze(1))
    assert torch.equal(k_fp8.view(torch.uint8), ref_fp8.view(torch.uint8))
    assert torch.equal(scales, ref_scales)
    # Dequantized keys are the keys, to e4m3 precision.
    dequant = k_fp8.float() * scales.unsqueeze(-1)
    assert torch.allclose(dequant[:, 0, :], keys.float(), rtol=0.07, atol=1e-3)


class _Task:
    def __init__(self):
        self.waited = 0

    def wait(self):
        self.waited += 1


class _AuxView:
    is_page_split = True

    def __init__(self):
        self.calls = []
        self.task = _Task()

    def async_offload_packed_layer_kv_to_host(self, **kwargs):
        self.calls.append(kwargs)
        return self.task


def _keys(n):
    return dsa_prefill.quantize_index_keys(torch.randn(n, 128).to(torch.bfloat16))


def _offload(view, keys_fp8, scales, ids, lengths, layer_idx=3):
    pinned, tracked = [], []
    dsa_prefill.offload_packed_index_keys(
        aux_view=view, layer_idx=layer_idx, keys_fp8=keys_fp8, scales=scales,
        sequence_ids=ids, sequence_lengths=lengths,
        pin=lambda t, l: pinned.append((t, l)),
        track=lambda t, l: tracked.append((t, l)),
    )
    return pinned, tracked


def test_packed_offload_carries_the_scales():
    view = _AuxView()
    keys_fp8, scales = _keys(7)
    pinned, tracked = _offload(view, keys_fp8, scales, [11, 12], [3, 4])
    call = view.calls[0]
    assert call["layer_idx"] == 3
    assert call["sequence_ids"] == [11, 12]
    assert call["k_tensor"] is keys_fp8
    assert call["scale_tensor"] is scales
    assert call["v_tensor"] is None
    assert call["sequence_lengths"] == [3, 4]
    assert [t for t, _ in pinned] == [keys_fp8, scales]
    assert tracked == [(view.task, 3)]
    assert view.task.waited == 1


def test_offload_refuses_a_missing_aux_view():
    with pytest.raises(RuntimeError, match="no auxiliary Host KV"):
        _offload(None, *_keys(2), [1], [2])


def test_offload_refuses_an_aos_aux_region():
    view = _AuxView()
    view.is_page_split = False
    with pytest.raises(RuntimeError, match="not page-split"):
        _offload(view, *_keys(2), [1], [2])


def test_offload_refuses_wrong_dtypes_and_shapes():
    keys_fp8, scales = _keys(3)
    with pytest.raises(TypeError, match="e4m3"):
        _offload(_AuxView(), keys_fp8.float(), scales, [1], [3])
    with pytest.raises(TypeError, match="fp32"):
        _offload(_AuxView(), keys_fp8, scales.double(), [1], [3])
    with pytest.raises(ValueError, match="one per key"):
        _offload(_AuxView(), keys_fp8, scales[:2], [1], [3])


def test_offload_refuses_mismatched_metadata():
    keys_fp8, scales = _keys(2)
    with pytest.raises(RuntimeError, match="sequence ids but"):
        _offload(_AuxView(), keys_fp8, scales, [1, 2], [2])
    with pytest.raises(RuntimeError, match="sum to"):
        _offload(_AuxView(), keys_fp8, scales, [1, 2], [1, 2])


def test_offload_of_no_tokens_issues_nothing():
    view = _AuxView()
    keys_fp8, scales = _keys(0)
    pinned, tracked = _offload(view, keys_fp8, scales, [], [])
    assert view.calls == [] and pinned == [] and tracked == []


# --------------------------------------------------------------------------- #
#  Wrapper integration                                                        #
# --------------------------------------------------------------------------- #

def _bind_wrapper_statics(monkeypatch, view, *, ids, lengths, num_sequences):
    pinned, tracked = [], []
    monkeypatch.setattr(AttnWrapperBase, "host_paged_kv_worker_view_aux", view, raising=False)
    monkeypatch.setattr(AttnWrapperBase, "cur_batch", ids, raising=False)
    monkeypatch.setattr(AttnWrapperBase, "prepack_seq_lengths", lengths, raising=False)
    monkeypatch.setattr(AttnWrapperBase, "prepack_num_sequences", num_sequences, raising=False)
    monkeypatch.setattr(AttnWrapperBase, "pin_prefill_offload_tensor",
                        staticmethod(lambda t, l: pinned.append(l)), raising=False)
    monkeypatch.setattr(AttnWrapperBase, "track_prefill_offload_task",
                        staticmethod(lambda t, l: tracked.append(l)), raising=False)
    return pinned, tracked


def test_wrapper_writes_page_split_keys_with_the_latent_offloads_metadata(monkeypatch):
    from batchgen.models.moonshotai.kimi_linear.wrappers import KimiLinearAttnWrapper

    view = _AuxView()
    pinned, tracked = _bind_wrapper_statics(
        monkeypatch, view, ids=[21, 22], lengths=[2, 3], num_sequences=2)
    keys = dsa_prefill.prefill_index_keys(_indexer(), torch.randn(5, 64).to(torch.bfloat16))
    KimiLinearAttnWrapper._write_prefill_index_keys(
        SimpleNamespace(layer_idx=7), keys, torch.tensor([0, 2, 5]))
    call = view.calls[0]
    assert call["layer_idx"] == 7
    assert call["sequence_ids"] == [21, 22]
    assert call["sequence_lengths"] == [2, 3]
    assert call["k_tensor"].shape == (5, 1, 128)
    assert call["k_tensor"].dtype is torch.float8_e4m3fn
    assert call["scale_tensor"].shape == (5, 1)
    assert pinned == [7, 7] and tracked == [7]


def test_wrapper_falls_back_to_cu_seqlens_for_lengths(monkeypatch):
    from batchgen.models.moonshotai.kimi_linear.wrappers import KimiLinearAttnWrapper

    view = _AuxView()
    _bind_wrapper_statics(monkeypatch, view, ids=[1, 2], lengths=None, num_sequences=2)
    keys = dsa_prefill.prefill_index_keys(_indexer(), torch.randn(4, 64).to(torch.bfloat16))
    KimiLinearAttnWrapper._write_prefill_index_keys(
        SimpleNamespace(layer_idx=3), keys, torch.tensor([0, 1, 4]))
    assert view.calls[0]["sequence_lengths"] == [1, 3]


def test_wrapper_refuses_a_prefix_pool_wave_before_any_work(monkeypatch):
    from batchgen.models.moonshotai.kimi_linear.wrappers import KimiLinearAttnWrapper

    monkeypatch.setattr(AttnWrapperBase, "prefill_pool_batch", object(), raising=False)
    monkeypatch.setattr(AttnWrapperBase, "prepack_mode", True, raising=False)
    monkeypatch.setattr(AttnWrapperBase, "retire_pending_prefill_offloads_before_layer",
                        staticmethod(lambda *a, **k: None), raising=False)
    touched = []
    module = SimpleNamespace(
        indexer=object(),
        mla_prefill_nope_prepacked=lambda *a, **k: touched.append("attn"),
    )
    wrapper = SimpleNamespace(
        module=module, layer_idx=3,
        _offload_prepacked_kv=lambda *a, **k: touched.append("offload"),
    )
    with pytest.raises(NotImplementedError, match="prefix-pool"):
        KimiLinearAttnWrapper._forward_prefill(wrapper, torch.randn(1, 4, 64))
    assert touched == []
