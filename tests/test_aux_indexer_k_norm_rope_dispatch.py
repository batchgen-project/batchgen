"""``write_aux_indexer_k_norm_rope`` is page-split only, and forwards verbatim.

The fused LayerNorm+RoPE+Hadamard+quantize+write launch replaces three
launches of the GLM-5.2 K tail, but it can only serve a page-split FP8 indexer
cache: a bf16 AoS cache stores unquantized records, so there is nothing for
the fp8 epilogue to write. These tests pin that the format gate fails loud
instead of quietly reverting, and that every argument the captured graph hands
over reaches the kernel wrapper unchanged -- the arguments ARE the contract
(wrong gamma, a stale rope table or a dropped ``num_valid_tokens`` would each
corrupt a page without raising).

No GPU: the launch is stubbed, which is what a dispatch contract is about.
The kernel's own byte behaviour lives in
``test_indexer_k_norm_rope_write_contract.py``.
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
ROPE_DIM = 64
PAGE_SIZE = 64
MAX_SEQ = 128
BATCH = 2

_WRAPPER_MODULE = (
    "batchgen_kernels.attention.dsa.indexer.indexer_k_norm_rope_write_fp8"
)


def _page_split_cache() -> torch.Tensor:
    return torch.zeros(4, PAGE_SIZE, 1, DATA_DIM + 4, dtype=torch.float8_e4m3fn)


def _aos_cache() -> torch.Tensor:
    return torch.zeros(4, PAGE_SIZE, 1, DATA_DIM, dtype=torch.bfloat16)


def _kwargs(cache: torch.Tensor) -> dict:
    return {
        "aux_blocked_k": cache,
        "k_raw": torch.zeros(BATCH, DATA_DIM, dtype=torch.bfloat16),
        "ln_weight": torch.ones(DATA_DIM, dtype=torch.bfloat16),
        "ln_bias": torch.zeros(DATA_DIM, dtype=torch.bfloat16),
        "ln_eps": 1e-5,
        "cos_cache": torch.ones(MAX_SEQ, ROPE_DIM, dtype=torch.float32),
        "sin_cache": torch.zeros(MAX_SEQ, ROPE_DIM, dtype=torch.float32),
        "positions": torch.zeros(BATCH, dtype=torch.int64),
        "page_table": torch.zeros(3, 2, dtype=torch.int32),
        "slot_indices": torch.zeros(BATCH, dtype=torch.int32),
        "token_indices": torch.zeros(BATCH, dtype=torch.int32),
        "page_size_tokens": PAGE_SIZE,
        "apply_hadamard": True,
        "k_fp8_out": torch.zeros(
            BATCH, 1, 1, DATA_DIM, dtype=torch.float8_e4m3fn
        ),
        "scales_out": torch.zeros(BATCH, dtype=torch.float32),
    }


def _install_stub(monkeypatch, recorder):
    stub = types.ModuleType(_WRAPPER_MODULE)

    def fused(**kwargs):
        recorder.update(kwargs)
        return "k_fp8", "scales"

    stub.fused_indexer_k_norm_rope_write_fp8 = fused
    monkeypatch.setitem(sys.modules, _WRAPPER_MODULE, stub)


def test_page_split_cache_takes_the_fused_launch_and_returns_both(monkeypatch):
    seen: dict = {}
    _install_stub(monkeypatch, seen)
    monkeypatch.setattr(
        aux_indexer_k_writer,
        "run_paged_kv_token_update_fused",
        lambda **kwargs: pytest.fail("the fused path must not take an AoS launch"),
    )

    kwargs = _kwargs(_page_split_cache())
    num_valid = torch.tensor([1], dtype=torch.int32)
    k_host, scales = aux_indexer_k_writer.write_aux_indexer_k_norm_rope(
        **kwargs, num_valid_tokens=num_valid
    )

    assert (k_host, scales) == ("k_fp8", "scales")
    assert seen["aux_cache"] is kwargs["aux_blocked_k"]
    assert seen["k_raw"] is kwargs["k_raw"]
    assert seen["ln_weight"] is kwargs["ln_weight"]
    assert seen["ln_bias"] is kwargs["ln_bias"]
    assert seen["cos_cache"] is kwargs["cos_cache"]
    assert seen["sin_cache"] is kwargs["sin_cache"]
    assert seen["positions"] is kwargs["positions"]
    assert seen["page_table"] is kwargs["page_table"]
    assert seen["slot_indices"] is kwargs["slot_indices"]
    assert seen["token_indices"] is kwargs["token_indices"]
    assert seen["k_fp8_out"] is kwargs["k_fp8_out"]
    assert seen["scales_out"] is kwargs["scales_out"]
    assert seen["num_valid_tokens"] is num_valid
    assert seen["page_size_tokens"] == PAGE_SIZE
    assert seen["ln_eps"] == pytest.approx(1e-5)
    assert seen["apply_hadamard"] is True


def test_the_rotation_flag_is_forwarded_not_defaulted(monkeypatch):
    """The Hadamard-free instantiation is upstream's; it must be explicit."""

    seen: dict = {}
    _install_stub(monkeypatch, seen)
    kwargs = _kwargs(_page_split_cache())
    kwargs["apply_hadamard"] = False
    aux_indexer_k_writer.write_aux_indexer_k_norm_rope(**kwargs)
    assert seen["apply_hadamard"] is False


def test_an_aos_cache_is_refused_by_name(monkeypatch):
    _install_stub(monkeypatch, {})
    monkeypatch.setattr(
        aux_indexer_k_writer,
        "run_paged_kv_token_update_fused",
        lambda **kwargs: pytest.fail("must not launch"),
    )
    with pytest.raises(ValueError, match="needs a page-split FP8 aux indexer"):
        aux_indexer_k_writer.write_aux_indexer_k_norm_rope(
            **_kwargs(_aos_cache())
        )


def test_a_mis_sized_page_split_cache_is_refused_by_name(monkeypatch):
    _install_stub(monkeypatch, {})
    wrong = torch.zeros(4, PAGE_SIZE, 1, DATA_DIM, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="needs a page-split FP8 aux indexer"):
        aux_indexer_k_writer.write_aux_indexer_k_norm_rope(**_kwargs(wrong))


def test_the_fused_writer_is_exported():
    assert "write_aux_indexer_k_norm_rope" in aux_indexer_k_writer.__all__
