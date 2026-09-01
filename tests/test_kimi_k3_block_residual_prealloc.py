# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3                                                           #
#  copyright (c) EfficientMoE team 2025                                         #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""Preallocated ``block_residual`` is BIT-EXACT with the ``torch.cat`` form.

What changed and why
--------------------
``KimiDecoderLayer._forward_attn_residual`` used to grow ``block_residual`` with
``torch.cat([block_residual, snapshot], dim=1)`` at every block boundary.  A cat
allocates the ``(S, nb+1, H)`` result while the ``(S, nb, H)`` input is still
live, so K3's last boundary (layer 84) held 12.25 + 14.00 GiB at once at
S=131,072 — ``batchgen_design/model_support/kimi_k3/PREFILL_MEMORY_AUDIT.md``
§4/§7 fix 3.  The append now writes column ``nb`` of a buffer that already
exists and hands back the narrowed ``buf[:, :nb+1]`` view.

Why this test exists
--------------------
This is a memory optimisation, so it is only allowed to be free.  K3 turns a
1e-7 seam into an O(1) logit difference through its discontinuous top-16 router
(``tests/gpu/test_kimi_k3_kda_fla_parity.py::test_E_kernel_seam_amplification``),
so the bar is ``torch.equal`` and not a tolerance.  Three drives of the SAME
93-layer / 8-boundary stack must agree bit for bit:

  ``M2``        the eager ground truth, ``kimi_k3/model.py``'s own untouched
                ``KimiK3DecoderLayer.forward`` + ``_apply_attn_res_lean``, which
                still uses ``torch.cat`` and is not on the change's diff at all;
  ``CAT``       the real, patched ``kimi_linear`` layer body with the buffer NOT
                seeded, so ``BlockResidualBuffer.append`` takes its ``torch.cat``
                branch — i.e. exactly the pre-fix behaviour, through the
                post-fix code;
  ``PREALLOC``  the real, patched layer body with the buffer seeded.

Both the per-layer ``(prefix_sum, block_residual)`` trace and the output stage
(depth mix, then the final norm) are compared.  ``test_prealloc_is_engaged``
proves ``PREALLOC`` is not passing trivially by falling back to the cat.

Runs on CPU with torch alone: ``fla`` (imported at ``kimi_linear/model.py``
module scope) and the ``batchgen`` package (whose ``__init__`` JIT-builds a CUDA
op) are stubbed if and only if the real ones cannot be imported.  Nothing under
test touches either.
"""

from __future__ import annotations

import importlib
import sys
import types
import weakref
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

_MOONSHOT = Path(__file__).resolve().parents[1] / "batchgen" / "models" / "moonshotai"

# 93 layers at block size 12 -> boundaries at 0, 12, ..., 84: nb reaches 8, the
# real K3 shape of the problem. T is deliberately ragged over apply_attn_res's
# 1024-token chunking so the ragged final chunk is exercised identically by all
# three drives.
_LAYERS = 93
_BLOCK = 12
_TOKENS = 2053
_HIDDEN = 32


# --------------------------------------------------------------------------- #
#  Import plumbing                                                             #
# --------------------------------------------------------------------------- #
def _load_as_package(alias: str, directory: Path, module: str):
    """Import ``directory/module.py`` as ``alias.module`` without running the
    real package's ``__init__``.  Relative imports inside it resolve against
    ``directory``."""
    pkg = types.ModuleType(alias)
    pkg.__path__ = [str(directory)]
    sys.modules[alias] = pkg
    return importlib.import_module("{}.{}".format(alias, module))


def _stub_fla() -> None:
    for name in ("fla", "fla.modules", "fla.ops", "fla.ops.kda"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["fla.modules"].FusedRMSNormGated = type("FusedRMSNormGated", (), {})
    sys.modules["fla.modules"].ShortConvolution = type("ShortConvolution", (), {})
    sys.modules["fla.ops.kda"].chunk_kda = lambda *a, **k: None
    sys.modules["fla.ops.kda"].fused_recurrent_kda = lambda *a, **k: None


def _stub_batchgen_config() -> None:
    """Only the two symbols ``kimi_linear/config.py`` imports."""
    bg = types.ModuleType("batchgen")
    bg.__path__ = []
    cfg_pkg = types.ModuleType("batchgen.config")
    cfg_pkg.__path__ = []
    model_config = types.ModuleType("batchgen.config.model_config")

    @dataclass
    class BaseModelConfig:  # noqa: D401 - stand-in
        pass

    model_config.BaseModelConfig = BaseModelConfig
    registry = types.ModuleType("batchgen.config.model_registry")
    registry.register_config = lambda *a, **k: (lambda cls: cls)
    sys.modules.update({
        "batchgen": bg,
        "batchgen.config": cfg_pkg,
        "batchgen.config.model_config": model_config,
        "batchgen.config.model_registry": registry,
    })


def _import_modules():
    try:
        import fla  # noqa: F401
    except Exception:
        _stub_fla()
    try:
        from batchgen.models.moonshotai.kimi_k3 import model as m2
        from batchgen.models.moonshotai.kimi_linear import model as kl
        return kl, m2
    except Exception:
        pass
    # The installed batchgen package is not importable here (its __init__ chain
    # JIT-builds a CUDA op). Drop whatever half-import it left behind and stand
    # in for the two symbols kimi_linear/config.py actually needs, then load
    # both model files directly. kimi_k3/model.py needs no stub at all.
    for name in [n for n in list(sys.modules)
                 if n == "batchgen" or n.startswith("batchgen.")]:
        del sys.modules[name]
    _stub_batchgen_config()
    kl = _load_as_package("_k3prealloc_kl", _MOONSHOT / "kimi_linear", "model")
    m2 = _load_as_package("_k3prealloc_m2", _MOONSHOT / "kimi_k3", "model")
    return kl, m2


try:
    KL, M2 = _import_modules()
except Exception as exc:  # pragma: no cover - environment problem, not a failure
    pytest.skip("cannot import the K3 model modules: {}".format(exc),
                allow_module_level=True)

BlockResidualBuffer = KL.BlockResidualBuffer
num_block_residual_columns = KL.num_block_residual_columns
# The carrier lives next to the buffer; reach it through whichever alias the
# import dance above resolved to.
BlockResidualCarrier = sys.modules[
    BlockResidualBuffer.__module__].BlockResidualCarrier
apply_attn_res = sys.modules[BlockResidualBuffer.__module__].apply_attn_res


# --------------------------------------------------------------------------- #
#  One stack, shared by all three drives                                       #
# --------------------------------------------------------------------------- #
class _StubLayer:
    """Everything ``_forward_attn_residual`` / ``KimiK3DecoderLayer.forward``
    read, and nothing else.

    Attention and FFN are one fixed matmul each, seeded by layer index and
    SHARED between the drives, so they cancel exactly: any nonzero difference is
    residual bookkeeping and nothing else.
    """

    def __init__(self, layer_idx, params, w_attn, w_ffn):
        self.layer_idx = layer_idx
        self.attn_res_block_size = _BLOCK
        self.input_layernorm = params["input_layernorm"]
        self.post_attention_layernorm = params["post_attention_layernorm"]
        self.self_attention_res_norm = params["self_attention_res_norm"]
        self.self_attention_res_proj = params["self_attention_res_proj"]
        self.mlp_res_norm = params["mlp_res_norm"]
        self.mlp_res_proj = params["mlp_res_proj"]
        self._w_attn = w_attn
        self._w_ffn = w_ffn

    def _run_attn(self, hidden_states, *args, **kwargs):
        return hidden_states @ self._w_attn

    def _run_ffn(self, hidden_states):
        return hidden_states @ self._w_ffn


class _StubModel:
    """The output stage, for ``KimiLinearModel._apply_output_attn_res``."""

    def __init__(self, norm, proj):
        self.output_attn_res_norm = norm
        self.output_attn_res_proj = proj


def _norm_and_proj(gen, dtype):
    norm = KL.KimiRMSNorm(_HIDDEN)
    proj = torch.nn.Linear(_HIDDEN, 1, bias=False)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(_HIDDEN, generator=gen))
        proj.weight.copy_(torch.randn(1, _HIDDEN, generator=gen))
    return norm.to(dtype), proj.to(dtype)


def _build_stack(dtype):
    """(layers, out_model, final_norm, x). The module instances are shared by
    every drive — the two files' ``KimiRMSNorm`` are the same implementation."""
    gen = torch.Generator().manual_seed(20260808)
    layers = []
    for layer_idx in range(_LAYERS):
        params = {}
        for key in ("input_layernorm", "post_attention_layernorm",
                    "self_attention_res_norm", "mlp_res_norm"):
            params[key] = _norm_and_proj(gen, dtype)[0]
        for key in ("self_attention_res_proj", "mlp_res_proj"):
            params[key] = _norm_and_proj(gen, dtype)[1]
        scale = _HIDDEN ** -0.5
        w_attn = (torch.randn(_HIDDEN, _HIDDEN, generator=gen) * scale).to(dtype)
        w_ffn = (torch.randn(_HIDDEN, _HIDDEN, generator=gen) * scale).to(dtype)
        layers.append(_StubLayer(layer_idx, params, w_attn, w_ffn))
    out_norm, out_proj = _norm_and_proj(gen, dtype)
    final_norm = KL.KimiRMSNorm(_HIDDEN).to(dtype)
    with torch.no_grad():
        final_norm.weight.copy_(torch.randn(_HIDDEN, generator=gen).to(dtype))
    x = (torch.randn(1, _TOKENS, _HIDDEN, generator=gen) * 0.5).to(dtype)
    return layers, _StubModel(out_norm, out_proj), final_norm, x


def _expected_num_blocks(layer_idx):
    """Boundaries appended once layer ``layer_idx`` has returned."""
    return layer_idx // _BLOCK + 1


# --------------------------------------------------------------------------- #
#  The three drives                                                            #
# --------------------------------------------------------------------------- #
def _drive_m2(layers, out_model, final_norm, x):
    """``kimi_k3/model.py``'s own body — the untouched cat-based ground truth."""
    block_residual = torch.zeros(_TOKENS, 0, _HIDDEN, dtype=x.dtype)
    hidden_states, trace = x, []
    for layer in layers:
        hidden_states, block_residual = M2.KimiK3DecoderLayer.forward(
            layer, hidden_states, None, block_residual, None)
        trace.append((hidden_states, block_residual))
    mixed = M2._apply_attn_res_lean(
        hidden_states.view(-1, _HIDDEN), block_residual,
        out_model.output_attn_res_proj, out_model.output_attn_res_norm)
    return trace, mixed, final_norm(mixed)


def _drive_kimi_linear(layers, out_model, final_norm, x, *, prealloc):
    """The real, patched ``kimi_linear`` body.

    ``prealloc=False`` hands the layer a block_residual the buffer never issued,
    so ``BlockResidualBuffer.append`` takes its ``torch.cat`` branch — the
    pre-fix behaviour, executed by the post-fix code.
    """
    BlockResidualBuffer.reset()
    if prealloc:
        block_residual = BlockResidualBuffer.seed(
            _TOKENS, _HIDDEN, num_block_residual_columns(_LAYERS, _BLOCK),
            dtype=x.dtype, device=x.device)
    else:
        block_residual = torch.zeros(_TOKENS, 0, _HIDDEN, dtype=x.dtype)

    hidden_states, trace = x, []
    for layer in layers:
        hidden_states, block_residual = KL.KimiDecoderLayer._forward_attn_residual(
            layer, hidden_states, None, None, None, None, block_residual)
        # The production stack owns each previous layer output and may reuse
        # it as the next layer's residual destination. Snapshot values here;
        # this trace is test instrumentation, not a production lifetime.
        trace.append((hidden_states.clone(), block_residual))
    mixed = KL.KimiLinearModel._apply_output_attn_res(
        out_model, hidden_states.view(-1, _HIDDEN), block_residual)
    return trace, mixed, final_norm(mixed)


# --------------------------------------------------------------------------- #
#  T1 — bit-exactness across the full 93-layer / 8-boundary drive              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16],
                         ids=["fp32", "bf16"])
def test_prealloc_bit_exact_over_full_stack(dtype):
    layers, out_model, final_norm, x = _build_stack(dtype)

    m2_trace, m2_mixed, m2_out = _drive_m2(layers, out_model, final_norm, x)
    cat_trace, cat_mixed, cat_out = _drive_kimi_linear(
        layers, out_model, final_norm, x, prealloc=False)
    pre_trace, pre_mixed, pre_out = _drive_kimi_linear(
        layers, out_model, final_norm, x, prealloc=True)

    print("\n[block-residual prealloc] dtype={} T={} H={} layers={} block={}"
          .format(dtype, _TOKENS, _HIDDEN, _LAYERS, _BLOCK))
    print("  boundaries = {}  (num_block_residual_columns -> {})".format(
        [i for i in range(_LAYERS) if i % _BLOCK == 0],
        num_block_residual_columns(_LAYERS, _BLOCK)))

    for layer_idx in range(_LAYERS):
        pre_ps, pre_br = pre_trace[layer_idx]
        cat_ps, cat_br = cat_trace[layer_idx]
        m2_ps, m2_br = m2_trace[layer_idx]
        nb = _expected_num_blocks(layer_idx)

        assert pre_br.shape == (_TOKENS, nb, _HIDDEN), (
            "layer {}: block_residual.shape[1] must still COUNT BOUNDARIES, got "
            "{} not {} — a narrowed view is what keeps the shape[1] > 0 gate and "
            "apply_attn_res honest".format(layer_idx, tuple(pre_br.shape),
                                           (_TOKENS, nb, _HIDDEN)))
        assert cat_br.shape == pre_br.shape == m2_br.shape

        assert torch.equal(pre_ps, cat_ps), (
            "layer {} prefix_sum differs from the cat form".format(layer_idx))
        assert torch.equal(pre_br, cat_br), (
            "layer {} block_residual differs from the cat form".format(layer_idx))
        assert torch.equal(pre_ps, m2_ps), (
            "layer {} prefix_sum differs from the M2 ground truth".format(layer_idx))
        assert torch.equal(pre_br, m2_br), (
            "layer {} block_residual differs from the M2 ground truth".format(layer_idx))

    assert torch.equal(pre_mixed, cat_mixed), "output depth mix differs from the cat form"
    assert torch.equal(pre_mixed, m2_mixed), "output depth mix differs from M2"
    assert torch.equal(pre_out, cat_out), "final (mix + norm) differs from the cat form"
    assert torch.equal(pre_out, m2_out), "final (mix + norm) differs from M2"
    print("  93/93 layers, output mix and final norm: torch.equal on all three drives")
    BlockResidualBuffer.reset()


# --------------------------------------------------------------------------- #
#  T2 — the optimisation is actually engaged (T1 must not pass by fallback)    #
# --------------------------------------------------------------------------- #
def test_prealloc_is_engaged():
    layers, out_model, final_norm, x = _build_stack(torch.float32)

    cat_trace, _, _ = _drive_kimi_linear(layers, out_model, final_norm, x,
                                         prealloc=False)
    pre_trace, _, _ = _drive_kimi_linear(layers, out_model, final_norm, x,
                                         prealloc=True)

    num_columns = num_block_residual_columns(_LAYERS, _BLOCK)
    assert num_columns == 8

    pointers, storages = set(), set()
    for layer_idx in range(_LAYERS):
        _, pre_br = pre_trace[layer_idx]
        _, cat_br = cat_trace[layer_idx]
        pointers.add(pre_br.data_ptr())
        storages.add(pre_br.untyped_storage().nbytes())
        # The cat form owns exactly its own bytes; the preallocated view sits in
        # a buffer sized for every boundary the stack will ever cross.
        assert cat_br.is_contiguous()
        assert cat_br.untyped_storage().nbytes() == cat_br.numel() * cat_br.element_size()
        assert pre_br.untyped_storage().nbytes() == (
            _TOKENS * num_columns * _HIDDEN * pre_br.element_size())

    assert pointers == {pre_trace[0][1].data_ptr()}, (
        "the preallocated block_residual moved: it was reallocated somewhere, "
        "which is the doubling this change removes")
    assert storages == {_TOKENS * num_columns * _HIDDEN * 4}
    print("\n[engaged] one {}-column buffer, one data_ptr across all 8 boundaries"
          .format(num_columns))
    BlockResidualBuffer.reset()


# --------------------------------------------------------------------------- #
#  T3 — intra-forward scratch: a second pass reproduces the first              #
# --------------------------------------------------------------------------- #
def test_second_pass_is_identical():
    """No pass may see another pass's scratch.

    ``seed`` resets the boundary count to zero and only ever exposes columns
    this pass has written, which is what makes this hold; it *also* allocates a
    freshly zeroed buffer, but that is server hygiene (a recycled buffer's
    untouched columns would hold another request's activations if a future bug
    ever exposed them), not something this assertion can observe.
    """
    layers, out_model, final_norm, x = _build_stack(torch.float32)
    _, _, first = _drive_kimi_linear(layers, out_model, final_norm, x, prealloc=True)
    _, _, second = _drive_kimi_linear(layers, out_model, final_norm, x, prealloc=True)
    assert torch.equal(first, second), (
        "block_residual leaked across forwards: it is intra-forward scratch")
    BlockResidualBuffer.reset()


# --------------------------------------------------------------------------- #
#  T4 — boundary arithmetic and the overflow hard-fail                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("num_layers,block_size,expected", [
    (93, 12, 8),     # K3: boundaries 0,12,...,84
    (25, 3, 9),      # SYN-25 test config: 0,3,...,24
    (12, 12, 1),
    (13, 12, 2),
    (1, 12, 1),
])
def test_num_block_residual_columns(num_layers, block_size, expected):
    got = num_block_residual_columns(num_layers, block_size)
    assert got == expected
    assert got == len([i for i in range(num_layers) if i % block_size == 0])


def test_append_beyond_the_buffer_hard_fails():
    """Under-sizing the buffer must raise, never silently drop a block."""
    BlockResidualBuffer.reset()
    block_residual = BlockResidualBuffer.seed(4, 3, 1, dtype=torch.float32,
                                              device=torch.device("cpu"))
    column = torch.randn(4, 3)
    block_residual = BlockResidualBuffer.append(block_residual, column)
    assert block_residual.shape == (4, 1, 3)
    with pytest.raises(RuntimeError, match="overflow"):
        BlockResidualBuffer.append(block_residual, column)
    BlockResidualBuffer.reset()


# --------------------------------------------------------------------------- #
#  T5 — the buffer is class state, so its RELEASE has to be wired              #
# --------------------------------------------------------------------------- #
#  The tensor this change replaced was a plain local in the worker's prefill
#  frame: it died when the frame returned. The buffer does not. Nothing here is
#  about numerics — it is about 14.00 GiB (S=131,072 / H=7168 / bf16) staying
#  pinned across configure_decoding() and the resident-EP build if the release
#  is ever unwired. Weakrefs, because "was it freed" is the actual question.
def test_buffer_reset_releases_the_allocation():
    BlockResidualBuffer.reset()
    view = BlockResidualBuffer.seed(4, 3, 8, dtype=torch.float32,
                                    device=torch.device("cpu"))
    buf_ref = weakref.ref(BlockResidualBuffer._buf)
    del view
    assert buf_ref() is not None, "class state should still hold the buffer"
    BlockResidualBuffer.reset()
    assert buf_ref() is None, "BlockResidualBuffer.reset() did not free the buffer"


def test_carrier_reset_releases_the_buffer():
    """``BlockResidualCarrier.reset()`` is what the PSM calls on every phase
    switch (configure_prefill / configure_decoding) and after an aborted pass.
    Dropping only the carrier's view would leave the whole buffer alive."""
    BlockResidualCarrier.configure(_LAYERS, _BLOCK)
    block_residual = BlockResidualCarrier.borrow(0, torch.zeros(1, 5, 3))
    buf_ref = weakref.ref(BlockResidualBuffer._buf)
    BlockResidualCarrier.stash(0, block_residual)
    del block_residual
    assert buf_ref() is not None
    BlockResidualCarrier.reset()
    assert buf_ref() is None, (
        "BlockResidualCarrier.reset() freed its view but left the "
        "(num_tokens, num_columns, hidden) buffer pinned as class state")


def test_take_releases_the_buffer_with_the_output_mix():
    """The production carried path: the norm pre-hook calls ``take()``, mixes,
    and returns. After ``take()`` no class state may reference the buffer — the
    consumer's own view is the ONLY thing keeping it alive, so it dies with the
    hook's frame instead of outliving the whole prefill."""
    BlockResidualCarrier.configure(2, 1)          # boundaries at layers 0 and 1
    hidden_states = torch.zeros(1, 5, 3)
    column = torch.randn(5, 3)

    block_residual = BlockResidualCarrier.borrow(0, hidden_states)
    block_residual = BlockResidualBuffer.append(block_residual, column)
    BlockResidualCarrier.stash(0, block_residual)
    block_residual = BlockResidualCarrier.borrow(1, hidden_states)
    block_residual = BlockResidualBuffer.append(block_residual, column)
    BlockResidualCarrier.stash(1, block_residual)
    del block_residual

    buf_ref = weakref.ref(BlockResidualBuffer._buf)
    taken = BlockResidualCarrier.take()
    assert taken.shape == (5, 2, 3)
    assert BlockResidualBuffer._buf is None and BlockResidualBuffer._view is None
    assert buf_ref() is not None, "the consumer's view must still pin the bytes"
    del taken
    assert buf_ref() is None, (
        "the buffer outlived the pass: take() left it referenced by class state")


def test_carried_drive_releases_the_buffer_end_to_end():
    """The production PREFILL path, end to end, on CPU.

    ``tests/test_kimi_linear_block_residual_serving.py`` covers this wiring
    against the M2 ground truth but cannot run without a GPU (its import chain
    JIT-builds a CUDA op), so the release property is pinned here instead:
    drive the real ``decoder_layer_forward_block_residual`` over all 93 layers
    the way the worker's prepack loop does — no ``block_residual`` argument,
    only ``layer_outputs[0]`` kept — then fire the real ``model.norm`` pre-hook
    and check nothing is left holding the buffer.
    """
    layers, out_model, final_norm, x = _build_stack(torch.float32)
    BR = sys.modules[BlockResidualBuffer.__module__]
    # The stub layer supplies _run_attn/_run_ffn and the norms; the body itself
    # is the real one, exactly as the PSM installs it.
    for layer in layers:
        layer._forward_attn_residual = types.MethodType(
            KL.KimiDecoderLayer._forward_attn_residual, layer)

    BlockResidualCarrier.configure(_LAYERS, _BLOCK)
    hidden_states = x
    for layer in layers:
        hidden_states = BR.decoder_layer_forward_block_residual(
            layer, hidden_states)[0]

    buf_ref = weakref.ref(BlockResidualBuffer._buf)
    assert BlockResidualCarrier.peek() is not None

    pre_hook = BR.make_output_block_residual_pre_hook(out_model)
    mixed = pre_hook(final_norm, (hidden_states,))[0]

    assert mixed.shape == hidden_states.shape
    assert BlockResidualCarrier.peek() is None
    assert BlockResidualBuffer._buf is None
    assert buf_ref() is None, (
        "the (num_tokens, 8, hidden) buffer survived the prefill pass — at "
        "S=131,072 that is 14.00 GiB pinned across configure_decoding()")

    # And a second pass still reproduces the first: releasing is not a leak of
    # state into the next forward.
    BlockResidualCarrier.configure(_LAYERS, _BLOCK)
    hidden_states = x
    for layer in layers:
        hidden_states = BR.decoder_layer_forward_block_residual(
            layer, hidden_states)[0]
    again = BR.make_output_block_residual_pre_hook(out_model)(
        final_norm, (hidden_states,))[0]
    assert torch.equal(mixed, again)
    BlockResidualCarrier.reset()


def test_consumer_cat_erases_the_stride_difference():
    """The one device-dependent risk, pinned as a device-INDEPENDENT invariant.

    Everything downstream of the append happens inside ``apply_attn_res``, whose
    first act is ``torch.cat((block_residual[start:end], prefix.unsqueeze(1)),
    dim=1).float()``. ``torch.cat`` always writes a fresh CONTIGUOUS tensor, so
    the narrowed view's larger row stride cannot survive into the reductions
    below it: ``v`` comes out byte-identical AND identically laid out either
    way. Same bytes + same strides + same shape = same kernel choice = same
    bits, which is why the CPU ``torch.equal`` above extrapolates to CUDA,
    where kernel selection is layout-sensitive.
    """
    tokens, num_blocks, num_columns, hidden = 61, 5, 8, 16
    gen = torch.Generator().manual_seed(7)
    values = torch.randn(tokens, num_blocks, hidden, generator=gen)
    prefix = torch.randn(tokens, hidden, generator=gen)

    contiguous = values.clone()
    buf = torch.zeros(tokens, num_columns, hidden)
    buf[:, :num_blocks] = values
    view = buf[:, :num_blocks]

    assert contiguous.is_contiguous() and not view.is_contiguous()
    assert torch.equal(contiguous, view)

    for start, end in ((0, 32), (32, tokens)):     # incl. a ragged tail chunk
        from_contiguous = torch.cat(
            (contiguous[start:end], prefix[start:end].unsqueeze(1)), dim=1).float()
        from_view = torch.cat(
            (view[start:end], prefix[start:end].unsqueeze(1)), dim=1).float()
        assert torch.equal(from_contiguous, from_view)
        assert from_contiguous.stride() == from_view.stride()
        assert from_contiguous.shape == from_view.shape
        assert from_view.is_contiguous()


def test_score_multiply_in_place_is_bit_exact():
    """The W2 scratch fix may reuse ``k`` only if scoring stays bit-exact."""
    tokens, num_blocks, hidden = 61, 5, 16
    gen = torch.Generator().manual_seed(260821)
    prefix = torch.randn(tokens, hidden, generator=gen, dtype=torch.bfloat16)
    block = torch.randn(
        tokens, num_blocks, hidden, generator=gen, dtype=torch.bfloat16
    )
    proj = torch.nn.Linear(hidden, 1, bias=False).float()
    norm = types.SimpleNamespace(
        weight=torch.randn(hidden, generator=gen),
        variance_epsilon=1e-6,
    )

    expected = torch.empty_like(prefix)
    w = norm.weight.float() * proj.weight.squeeze(0).float()
    for start in range(0, tokens, 32):
        end = min(start + 32, tokens)
        v = torch.cat(
            (block[start:end], prefix[start:end].unsqueeze(1)), dim=1
        ).float()
        k = v * torch.rsqrt(
            v.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon
        )
        scores = (k * w).sum(-1)
        probs = scores.softmax(-1).unsqueeze(1)
        expected[start:end] = torch.matmul(probs, v).squeeze(1).to(
            expected.dtype
        )

    actual = apply_attn_res(
        prefix, block, proj, norm, chunk_size=32
    )
    assert torch.equal(actual, expected)


def test_streamed_sp8_depth_mix_row_shard_is_bit_exact(monkeypatch):
    """TP row sharding preserves the token-independent depth mixer exactly."""
    tokens, num_blocks, hidden = 16, 5, 16
    group_size, group_rank = 4, 2
    gen = torch.Generator().manual_seed(260902)
    prefix = torch.randn(tokens, hidden, generator=gen, dtype=torch.bfloat16)
    block = torch.randn(
        tokens, num_blocks, hidden, generator=gen, dtype=torch.bfloat16
    )
    proj = torch.nn.Linear(hidden, 1, bias=False).float()
    norm = types.SimpleNamespace(
        weight=torch.randn(hidden, generator=gen),
        variance_epsilon=1e-6,
    )
    reference = apply_attn_res(prefix, block, proj, norm, chunk_size=4)
    rows_per_rank = tokens // group_size
    trace = []

    def fake_all_gather(gathered, send, group):
        assert group == "fake-group"
        start = group_rank * rows_per_rank
        end = start + rows_per_rank
        assert trace == ["order_wait"]
        assert torch.equal(send, reference[start:end])
        gathered.copy_(reference)

    monkeypatch.setattr(
        torch.distributed,
        "all_gather_into_tensor",
        fake_all_gather,
    )
    norm._streamed_sp8_row_group = (
        group_size,
        group_rank,
        "fake-group",
    )
    norm._streamed_sp8_order_wait = lambda: trace.append("order_wait")

    actual = apply_attn_res(prefix, block, proj, norm, chunk_size=4)

    assert torch.equal(actual, reference)
    assert trace == ["order_wait"]


def test_resident_prefill_bounds_depth_mixer_chunk_exactly():
    """Resident prefill reuses its 512-row memory tile in the depth mixer."""
    tokens, num_blocks, hidden = 61, 5, 16
    gen = torch.Generator().manual_seed(260821)
    prefix = torch.randn(tokens, hidden, generator=gen, dtype=torch.bfloat16)
    block = torch.randn(
        tokens, num_blocks, hidden, generator=gen, dtype=torch.bfloat16
    )
    proj = torch.nn.Linear(hidden, 1, bias=False).float()
    norm = types.SimpleNamespace(
        weight=torch.randn(hidden, generator=gen),
        variance_epsilon=1e-6,
        _resident_prefill_token_tile=16,
    )

    expected = apply_attn_res(
        prefix, block, proj, norm, chunk_size=16
    )
    actual = apply_attn_res(
        prefix, block, proj, norm, chunk_size=1024
    )
    assert torch.equal(actual, expected)


def test_append_falls_back_to_cat_for_a_foreign_tensor():
    """A caller that never seeded keeps the exact previous behaviour."""
    BlockResidualBuffer.reset()
    foreign = torch.zeros(4, 0, 3)
    column = torch.randn(4, 3)
    got = BlockResidualBuffer.append(foreign, column)
    assert torch.equal(got, torch.cat([foreign, column.unsqueeze(1)], dim=1))
    assert got.is_contiguous()
