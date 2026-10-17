"""The aux indexer-K writer dispatches on the CACHE's format, not the model.

One decode path serves GLM-5/5.1/5.3 (bf16 AoS indexer cache) and GLM-5.2
(page-split FP8). ``write_aux_indexer_k`` is the only GPU writer of that
cache, so these tests pin which launch each format gets, and which host-append
tensors come back: an AoS cache must get the plain paged token update and NO
scales, a page-split cache must get the fused quantize+write and BOTH the
quantized keys and their fp32 per-token scales.

No GPU: the two launches are stubbed, which is exactly what the dispatch
contract is about. The byte-level behaviour of each launch is covered by
``test_indexer_k_page_split_write_contract.py``.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

# The writer imports the Triton paged-update launcher at module scope (it is
# the AoS path's kernel), so a host without Triton cannot import it at all.
pytest.importorskip("triton")

from batchgen.attention.dsa import aux_indexer_k_writer  # noqa: E402

DATA_DIM = 128
PAGE_SIZE = 64


def _aos_cache() -> torch.Tensor:
    return torch.zeros(4, PAGE_SIZE, 1, DATA_DIM, dtype=torch.bfloat16)


def _page_split_cache() -> torch.Tensor:
    return torch.zeros(4, PAGE_SIZE, 1, DATA_DIM + 4, dtype=torch.float8_e4m3fn)


def _keys(batch: int = 2) -> torch.Tensor:
    return torch.zeros(batch, 1, 1, DATA_DIM, dtype=torch.bfloat16)


def _common(cache: torch.Tensor) -> dict:
    return {
        "aux_blocked_k": cache,
        "indexer_k": _keys(),
        "page_table": torch.zeros(3, 2, dtype=torch.int32),
        "slot_indices": torch.zeros(2, dtype=torch.int32),
        "token_indices": torch.zeros(2, dtype=torch.int32),
        "page_size_tokens": PAGE_SIZE,
    }


def test_aos_cache_takes_the_paged_token_update_and_no_scales(monkeypatch):
    seen = {}

    def fake_update(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(
        aux_indexer_k_writer, "run_paged_kv_token_update_fused", fake_update
    )
    kwargs = _common(_aos_cache())
    num_valid = torch.tensor([1], dtype=torch.int32)
    k_host, scales = aux_indexer_k_writer.write_aux_indexer_k(
        **kwargs, num_valid_tokens=num_valid
    )

    assert k_host is kwargs["indexer_k"]
    assert scales is None
    assert seen["k_cache"] is kwargs["aux_blocked_k"]
    assert seen["k_tokens"].shape == (2, DATA_DIM)
    assert seen["page_table"] is kwargs["page_table"]
    assert seen["slot_indices"] is kwargs["slot_indices"]
    assert seen["token_indices"] is kwargs["token_indices"]
    assert seen["page_size_tokens"] == PAGE_SIZE
    assert seen["num_valid_tokens"] is num_valid


def test_aos_cache_rejects_host_fp8_buffers(monkeypatch):
    monkeypatch.setattr(
        aux_indexer_k_writer,
        "run_paged_kv_token_update_fused",
        lambda **kwargs: pytest.fail("must not launch"),
    )
    with pytest.raises(ValueError, match="rejects scales"):
        aux_indexer_k_writer.write_aux_indexer_k(
            **_common(_aos_cache()),
            scales_out=torch.zeros(2, dtype=torch.float32),
        )


def _install_fused_stub(monkeypatch, recorder):
    module_name = (
        "batchgen_kernels.attention.dsa.indexer.indexer_k_write_fp8"
    )
    stub = types.ModuleType(module_name)

    def fused(**kwargs):
        recorder.update(kwargs)
        return "k_fp8", "scales"

    stub.fused_indexer_k_write_fp8 = fused
    monkeypatch.setitem(sys.modules, module_name, stub)


def test_page_split_cache_takes_the_fused_write_and_returns_both(monkeypatch):
    seen: dict = {}
    _install_fused_stub(monkeypatch, seen)
    monkeypatch.setattr(
        aux_indexer_k_writer,
        "run_paged_kv_token_update_fused",
        lambda **kwargs: pytest.fail("a page-split cache must not take the AoS launch"),
    )

    kwargs = _common(_page_split_cache())
    k_fp8_out = torch.zeros(2, 1, 1, DATA_DIM, dtype=torch.float8_e4m3fn)
    scales_out = torch.zeros(2, dtype=torch.float32)
    k_host, scales = aux_indexer_k_writer.write_aux_indexer_k(
        **kwargs, k_fp8_out=k_fp8_out, scales_out=scales_out
    )

    assert (k_host, scales) == ("k_fp8", "scales")
    assert seen["aux_cache"] is kwargs["aux_blocked_k"]
    assert seen["k_src"] is kwargs["indexer_k"]
    assert seen["k_fp8_out"] is k_fp8_out
    assert seen["scales_out"] is scales_out
    assert seen["page_size_tokens"] == PAGE_SIZE


def test_a_mis_sized_aux_cache_is_named(monkeypatch):
    monkeypatch.setattr(
        aux_indexer_k_writer,
        "run_paged_kv_token_update_fused",
        lambda **kwargs: pytest.fail("must not launch"),
    )
    wrong = torch.zeros(4, PAGE_SIZE, 1, DATA_DIM, dtype=torch.float8_e4m3fn)
    kwargs = _common(wrong)
    with pytest.raises(ValueError, match="must be page-split"):
        aux_indexer_k_writer.write_aux_indexer_k(**kwargs)
