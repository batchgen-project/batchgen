from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from batchgen.attention.v4_backend import (
    C4_TOPK,
    SWA_WINDOW,
    DSV4AttnMetadata,
    DSV4LayerConfig,
    DeepseekV4AttnBackend,
    V4AttnPath,
    build_layer_configs_from_compress_ratios,
)


def _meta(**overrides) -> DSV4AttnMetadata:
    base = dict(
        page_size=64,
        page_table=torch.zeros(1, 1, dtype=torch.int32),
        raw_out_loc=torch.zeros(1, dtype=torch.int32),
        seq_lens_casual=torch.tensor([128], dtype=torch.int32),
        positions_casual=torch.tensor([0], dtype=torch.int32),
        swa_page_indices=torch.zeros(1, 1, dtype=torch.int32),
        swa_topk_lengths=torch.zeros(1, dtype=torch.int32),
    )
    base.update(overrides)
    return DSV4AttnMetadata(**base)


def test_path_selection_from_compress_ratio():
    assert V4AttnPath.from_compress_ratio(0) is V4AttnPath.DENSE_MLA
    assert V4AttnPath.from_compress_ratio(4) is V4AttnPath.C4_SPARSE
    assert V4AttnPath.from_compress_ratio(128) is V4AttnPath.C128_COMPRESS


def test_path_selection_rejects_unsupported_ratio():
    with pytest.raises(ValueError, match="unsupported compress_ratio"):
        V4AttnPath.from_compress_ratio(7)


def test_build_layer_configs():
    cfgs = build_layer_configs_from_compress_ratios(
        compress_ratios=[0, 4, 128, 0, 4],
        n_heads=64,
        head_dim=512,
        rope_head_dim=64,
    )
    assert len(cfgs) == 5
    assert [c.compress_ratio for c in cfgs] == [0, 4, 128, 0, 4]
    assert [c.layer_idx for c in cfgs] == [0, 1, 2, 3, 4]
    assert cfgs[0].path is V4AttnPath.DENSE_MLA
    assert cfgs[1].path is V4AttnPath.C4_SPARSE
    assert cfgs[2].path is V4AttnPath.C128_COMPRESS


def test_metadata_must_be_initialized_before_access():
    backend = DeepseekV4AttnBackend(layer_configs=[])
    with pytest.raises(RuntimeError, match="before init_metadata"):
        _ = backend.metadata


def test_metadata_init_and_clear():
    backend = DeepseekV4AttnBackend(layer_configs=[])
    m = _meta()
    backend.init_metadata(m)
    assert backend.metadata is m
    backend.clear_metadata()
    with pytest.raises(RuntimeError):
        _ = backend.metadata


def test_dense_mla_requires_flashmla_backend():
    cfg = DSV4LayerConfig(
        layer_idx=0,
        compress_ratio=0,
        n_heads=64,
        head_dim=512,
        rope_head_dim=64,
    )
    backend = DeepseekV4AttnBackend(layer_configs=[cfg], flashmla_backend=None)
    backend.init_metadata(_meta())
    with pytest.raises(NotImplementedError, match="flashmla_backend"):
        backend.forward(cfg, q=torch.empty(0), kv=torch.empty(0))


def test_dense_mla_dispatches_to_flashmla():
    cfg = DSV4LayerConfig(
        layer_idx=3,
        compress_ratio=0,
        n_heads=64,
        head_dim=512,
        rope_head_dim=64,
    )
    flashmla = MagicMock(return_value=torch.tensor([42.0]))
    backend = DeepseekV4AttnBackend(
        layer_configs=[cfg], flashmla_backend=flashmla
    )
    meta = _meta()
    backend.init_metadata(meta)

    q = torch.empty(1)
    kv = torch.empty(1)
    out = backend.forward(cfg, q=q, kv=kv, attn_sink=None)

    assert torch.equal(out, torch.tensor([42.0]))
    flashmla.assert_called_once()
    kwargs = flashmla.call_args.kwargs
    assert kwargs["layer_idx"] == 3
    assert kwargs["metadata"] is meta
    assert kwargs["q"] is q
    assert kwargs["kv"] is kv


def test_c4_sparse_requires_c4_metadata_fields():
    cfg = DSV4LayerConfig(
        layer_idx=0,
        compress_ratio=4,
        n_heads=64,
        head_dim=512,
        rope_head_dim=64,
    )
    backend = DeepseekV4AttnBackend(
        layer_configs=[cfg], flashmla_backend=MagicMock()
    )
    backend.init_metadata(_meta(c4_out_loc=None))
    with pytest.raises(RuntimeError, match="c4_out_loc"):
        backend.forward(
            cfg,
            q=torch.empty(0),
            kv=torch.empty(0),
            head_gates=torch.empty(0),
        )


def test_c4_sparse_requires_head_gates():
    cfg = DSV4LayerConfig(
        layer_idx=0,
        compress_ratio=4,
        n_heads=64,
        head_dim=512,
        rope_head_dim=64,
    )
    backend = DeepseekV4AttnBackend(
        layer_configs=[cfg], flashmla_backend=MagicMock()
    )
    backend.init_metadata(
        _meta(
            c4_out_loc=torch.zeros(1, dtype=torch.int32),
            c4_topk_lengths_clamp1=torch.zeros(1, dtype=torch.int32),
        )
    )
    with pytest.raises(ValueError, match="head_gates"):
        backend.forward(cfg, q=torch.empty(0), kv=torch.empty(0))


def test_c128_compress_requires_c128_metadata_fields():
    cfg = DSV4LayerConfig(
        layer_idx=0,
        compress_ratio=128,
        n_heads=64,
        head_dim=512,
        rope_head_dim=64,
    )
    backend = DeepseekV4AttnBackend(
        layer_configs=[cfg], flashmla_backend=MagicMock()
    )
    backend.init_metadata(_meta(c128_page_indices=None))
    with pytest.raises(RuntimeError, match="c128"):
        backend.forward(cfg, q=torch.empty(0), kv=torch.empty(0))


def test_layer_config_path_property():
    cfg_dense = DSV4LayerConfig(
        layer_idx=0,
        compress_ratio=0,
        n_heads=64,
        head_dim=512,
        rope_head_dim=64,
    )
    cfg_c4 = DSV4LayerConfig(
        layer_idx=1,
        compress_ratio=4,
        n_heads=64,
        head_dim=512,
        rope_head_dim=64,
    )
    cfg_c128 = DSV4LayerConfig(
        layer_idx=2,
        compress_ratio=128,
        n_heads=64,
        head_dim=512,
        rope_head_dim=64,
    )
    assert cfg_dense.path is V4AttnPath.DENSE_MLA
    assert cfg_c4.path is V4AttnPath.C4_SPARSE
    assert cfg_c128.path is V4AttnPath.C128_COMPRESS


def test_swa_window_default():
    cfg = DSV4LayerConfig(
        layer_idx=0,
        compress_ratio=0,
        n_heads=64,
        head_dim=512,
        rope_head_dim=64,
    )
    assert cfg.swa_window == SWA_WINDOW


def test_c4_topk_default_in_metadata():
    m = _meta()
    assert m.c4_sparse_topk == C4_TOPK
