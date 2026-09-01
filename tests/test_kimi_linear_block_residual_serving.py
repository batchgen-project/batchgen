# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3                                                           #
#  copyright (c) EfficientMoE team 2025                                         #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""Serving-path Block Attention Residuals vs the M2 eager ground truth.

What is under test: the wiring in
``batchgen/models/moonshotai/kimi_linear/block_residual.py`` plus the PSM call
site ``KimiLinearParallelStrategyManager._config_block_residual`` — i.e. that
the SERVING caller (the worker's prepack-prefill loop, which drives the decoder
layers itself, passes no ``block_residual`` and keeps only
``layer_outputs[0]``) reproduces the block_residual / prefix_sum evolution of
the M2 eager model ``batchgen/models/moonshotai/kimi_k3/model.py``.

Isolation: both stacks get the SAME ``_run_attn`` / ``_run_ffn`` — one fixed
matmul per layer, seeded by layer index and identical across the two stacks —
so attention, MoE and their kernels cancel exactly and any nonzero error is a
residual-plumbing difference and nothing else. That is deliberate: the whole
end-to-end parity of the K3 serving stack is a different question (and a
different track); this test answers only "does the depth-residual bookkeeping
survive the serving caller".

Everything else is real: real ``KimiK3DecoderLayer`` and real
``KimiDecoderLayer`` objects built from the shrunk K3-SYN-25 config (25 layers,
``attn_res_block_size=3`` -> boundaries at layers 0,3,6,...,24, so the drive
crosses eight full block boundaries), real ``_forward_attn_residual`` body,
real ``block_residual.py``, real PSM injection, and the worker's layer-drive
loop transcribed verbatim.

Both stacks are constructed on the META device (as the PSM does) and only the
parameters the residual path touches are materialized; the stubbed attention /
FFN submodules are never called, so their meta parameters are never read.
"""

from __future__ import annotations

import copy
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kimi_k3_harness as H  # noqa: E402

pytest.importorskip(
    "fla", reason="kimi_linear/model.py imports fla at module scope")

from batchgen.models.moonshotai.kimi_linear.Parallel_Strategy_Manager import (  # noqa: E402
    KimiLinearParallelStrategyManager as PSM,
)
from batchgen.models.moonshotai.kimi_linear.block_residual import (  # noqa: E402
    BlockResidualCarrier,
)
from batchgen.models.moonshotai.kimi_linear.config import (  # noqa: E402
    KimiLinearConfig,
)
from batchgen.models.moonshotai.kimi_linear.model import (  # noqa: E402
    KimiLinearModel,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters the Block-Attention-Residual path reads. Names are identical in
# both stacks (both mirror the checkpoint), which is what lets one seeded value
# per name feed both.
_LAYER_PARAMS = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attention_res_norm.weight",
    "self_attention_res_proj.weight",
    "mlp_res_norm.weight",
    "mlp_res_proj.weight",
)
_MODEL_PARAMS = (
    "norm.weight",
    "output_attn_res_norm.weight",
    "output_attn_res_proj.weight",
)


# --------------------------------------------------------------------------- #
#  Fixture plumbing                                                            #
# --------------------------------------------------------------------------- #
def _shared_param_names(num_layers: int):
    names = []
    for i in range(num_layers):
        names.extend("layers.{}.{}".format(i, p) for p in _LAYER_PARAMS)
    names.extend(_MODEL_PARAMS)
    return names


def _materialize(root, names, dtype):
    """Swap the named meta Parameters for seeded concrete ones (PSM pattern:
    ``p.data =`` cannot materialize a meta parameter)."""
    for name in names:
        shape = tuple(root.get_parameter(name).shape)
        value = H.seeded_master(name, shape).to(device=DEVICE, dtype=dtype)
        *parent, leaf = name.split(".")
        module = root.get_submodule(".".join(parent)) if parent else root
        module._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)


def _install_stubs(layers, hidden, dtype):
    """Replace both stacks' attention and FFN with the same fixed matmul.

    Assigned as plain instance attributes (not MethodType) so the existing
    ``self._run_attn(hidden_states, ...)`` call sites reach them unbound; the
    two stacks' differing extra arguments are swallowed by ``*a, **k``.
    """
    for i, layer in enumerate(layers):
        w_attn = H.seeded_master("stub.attn.{}".format(i), (hidden, hidden))
        w_ffn = H.seeded_master("stub.ffn.{}".format(i), (hidden, hidden))
        w_attn = w_attn.to(device=DEVICE, dtype=dtype)
        w_ffn = w_ffn.to(device=DEVICE, dtype=dtype)
        layer._run_attn = lambda hs, *a, _w=w_attn, **k: hs @ _w
        layer._run_ffn = lambda hs, *a, _w=w_ffn, **k: hs @ _w


def _build_pair(dtype):
    """(m2_model, kl_model, kl_config) — same config, same residual weights,
    same stubbed attention/FFN, kl wired by the PSM's own injector."""
    cfg_dict = H.syn25_config_dict()

    ours = H.load_our_modules()
    m2_cfg = H.build_our_config(cfg_dict)
    with torch.device("meta"):
        m2 = ours.model.KimiK3Model(m2_cfg)

    kl_cfg = KimiLinearConfig.from_hf_dict(cfg_dict)
    with torch.device("meta"):
        kl = KimiLinearModel(kl_cfg)

    assert kl_cfg.num_hidden_layers == m2_cfg.num_hidden_layers
    assert kl_cfg.attn_res_block_size == m2_cfg.attn_res_block_size
    names = _shared_param_names(kl_cfg.num_hidden_layers)
    _materialize(m2, names, dtype)
    _materialize(kl, names, dtype)
    _install_stubs(m2.layers, kl_cfg.hidden_size, dtype)
    _install_stubs(kl.layers, kl_cfg.hidden_size, dtype)

    # The real PSM injector, called unbound on a stand-in: it reads only
    # loaded_model_config / model / rank.
    PSM._config_block_residual(
        types.SimpleNamespace(loaded_model_config=kl_cfg,
                              model=types.SimpleNamespace(model=kl),
                              rank=1)
    )
    return m2, kl, kl_cfg


def _inputs(kl_cfg, dtype, num_tokens=13, tag="brs"):
    """(1, T, H) packed-prefill-shaped activations."""
    x = H.seeded_master("{}.inputs".format(tag), (1, num_tokens, kl_cfg.hidden_size))
    return x.to(device=DEVICE, dtype=dtype)


# --------------------------------------------------------------------------- #
#  Drivers                                                                     #
# --------------------------------------------------------------------------- #
def _drive_m2(m2, hidden_states):
    """M2's own forward body, minus the embedding (kimi_k3/model.py:1075-1082)."""
    block_residual = m2._initial_block_residual(hidden_states)
    trace = []
    for layer in m2.layers:
        hidden_states, block_residual = layer(
            hidden_states, None, block_residual, None)
        trace.append((hidden_states, block_residual))
    return m2._finalize(hidden_states, block_residual), trace


def _drive_serving(kl, hidden_states):
    """The worker's prepack-prefill loop, verbatim
    (batchgen/batchgen_worker.py:7027-7038): no block_residual is passed in,
    only ``layer_outputs[0]`` is kept, and ``model.norm`` is called directly."""
    trace = []
    for decoder_layer in kl.layers:
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )
        hidden_states = layer_outputs[0]
        # The next layer is allowed to reuse this owned output as its residual
        # destination. Preserve the per-layer value for this diagnostic trace.
        trace.append((hidden_states.clone(), BlockResidualCarrier.peek()))
    return kl.norm(hidden_states), trace


def _err_ratio(actual, ref):
    """RMS(actual - ref) / RMS(ref) — the harness's err_ratio metric."""
    a = actual.float()
    r = ref.float()
    assert torch.isfinite(a).all(), "non-finite values in the serving output"
    return ((a - r).pow(2).mean().sqrt() / (r.pow(2).mean().sqrt() + 1e-8)).item()


# --------------------------------------------------------------------------- #
#  T1 — per-layer parity across block boundaries                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16],
                         ids=["fp32", "bf16"])
def test_serving_stack_matches_m2_block_residual_evolution(dtype):
    m2, kl, cfg = _build_pair(dtype)
    x = _inputs(cfg, dtype)

    BlockResidualCarrier.reset()
    m2_out, m2_trace = _drive_m2(m2, x)
    kl_out, kl_trace = _drive_serving(kl, x)

    block = cfg.attn_res_block_size
    print("\n[block-residual] device={} dtype={} T={} H={} layers={} "
          "block_size={}".format(DEVICE.type, dtype, x.shape[1],
                                 cfg.hidden_size, cfg.num_hidden_layers, block))
    print("  layer  bnd  nb  prefix_sum err_ratio  block_residual err_ratio")
    worst = 0.0
    for i in range(cfg.num_hidden_layers):
        kl_ps, kl_br = kl_trace[i]
        m2_ps, m2_br = m2_trace[i]
        assert kl_ps.shape == m2_ps.shape, "layer {} prefix_sum shape".format(i)
        assert kl_br.shape == m2_br.shape, (
            "layer {} block_residual shape {} vs {} — the boundary append/reset "
            "did not happen where M2 puts it".format(i, tuple(kl_br.shape),
                                                     tuple(m2_br.shape)))
        ps_err = _err_ratio(kl_ps, m2_ps)
        br_err = _err_ratio(kl_br, m2_br)
        worst = max(worst, ps_err, br_err)
        print("  {:5d}  {:3s}  {:2d}  {:17.3e}  {:21.3e}".format(
            i, "yes" if i % block == 0 else "-", kl_br.shape[1], ps_err, br_err))

    out_err = _err_ratio(kl_out, m2_out)
    print("  final (output mix + norm) err_ratio = {:.3e}".format(out_err))
    print("  worst per-layer err_ratio            = {:.3e}".format(worst))

    # Both stacks run the identical op sequence in the identical dtype, so the
    # bar is numerical identity, not a tolerance: any real difference here is a
    # different function, not rounding.
    assert worst < 1e-6, "per-layer worst err_ratio {:.3e}".format(worst)
    assert out_err < 1e-6, "final err_ratio {:.3e}".format(out_err)


# --------------------------------------------------------------------------- #
#  T2 — block_residual is intra-forward scratch                                #
# --------------------------------------------------------------------------- #
def test_serving_block_residual_does_not_leak_across_passes():
    _, kl, cfg = _build_pair(torch.float32)
    x = _inputs(cfg, torch.float32)

    BlockResidualCarrier.reset()
    first, _ = _drive_serving(kl, x)
    assert BlockResidualCarrier.peek() is None, (
        "the output-stage hook did not consume the carrier; the next pass "
        "would mix against stale scratch")
    second, _ = _drive_serving(kl, x)
    assert torch.equal(first, second), (
        "a second serving pass differs: block_residual leaked across forwards")


# --------------------------------------------------------------------------- #
#  T3 — the explicit-threading path and the carried path agree, and the        #
#       output mix runs exactly once in each                                   #
# --------------------------------------------------------------------------- #
def test_explicit_and_carried_paths_agree():
    _, kl, cfg = _build_pair(torch.float32)
    x = _inputs(cfg, torch.float32)

    BlockResidualCarrier.reset()
    carried, _ = _drive_serving(kl, x)
    assert BlockResidualCarrier.peek() is None

    # KimiLinearModel.forward threads block_residual itself and applies the
    # output mix itself — the norm pre-hook must then be a no-op, or the mix
    # would be applied twice.
    explicit = kl(inputs_embeds=x)
    assert BlockResidualCarrier.peek() is None
    assert torch.equal(carried, explicit), (
        "worker-driven and model-driven paths disagree: the output depth mix "
        "is applied a different number of times on one of them")


# --------------------------------------------------------------------------- #
#  T4 — the carrier refuses a partial / out-of-order drive                     #
# --------------------------------------------------------------------------- #
def test_carrier_rejects_out_of_order_drive():
    _, kl, cfg = _build_pair(torch.float32)
    x = _inputs(cfg, torch.float32)

    BlockResidualCarrier.reset()
    with pytest.raises(RuntimeError, match="out of order"):
        kl.layers[1](x, attention_mask=None)


def test_output_hook_rejects_a_truncated_stack():
    _, kl, cfg = _build_pair(torch.float32)
    x = _inputs(cfg, torch.float32)

    BlockResidualCarrier.reset()
    hidden_states = x
    for decoder_layer in kl.layers[:-1]:          # stop one layer short
        hidden_states = decoder_layer(hidden_states, attention_mask=None)[0]
    with pytest.raises(RuntimeError, match="stale"):
        kl.norm(hidden_states)
    BlockResidualCarrier.reset()


# --------------------------------------------------------------------------- #
#  T5 — the decode CUDA-graph adapter is installed for block_residual          #
# --------------------------------------------------------------------------- #
def test_decode_graph_mode_installed_for_attn_res(monkeypatch):
    """K3 must install the graph adapter rather than retain the old refusal.

    The fake keeps this wiring test CPU-only; the dedicated GPU segment-capture
    test proves boundary writes and depth-mix parity under replay.
    """
    _, kl, cfg = _build_pair(torch.float32)
    installed = []

    class FakeDecodeGraph:
        def __init__(self, model, model_config, **kwargs):
            self.model = model
            self.model_config = model_config
            self.kwargs = kwargs
            self.modes = [kwargs["mode"]]

        def install(self):
            installed.append(self)

        def set_mode(self, mode):
            self.modes.append(mode)

    from batchgen.models.moonshotai.kimi_linear import cuda_graph_segments
    monkeypatch.setattr(
        cuda_graph_segments, "KimiLinearDecodeGraph", FakeDecodeGraph
    )

    stub = types.SimpleNamespace(
        loaded_model_config=cfg,
        model=types.SimpleNamespace(model=kl),
        rank=1,
        _decode_graph=None,
        _decode_graph_mode=lambda: "graph",
        engine_config=types.SimpleNamespace(
            Basic_Config=types.SimpleNamespace(
                device_torch=DEVICE,
                decode_graph_buckets=[1, 2, 4],
                decode_graph_compare_every=7,
            )
        ),
    )
    PSM._init_decode_graph(stub)
    assert installed == [stub._decode_graph]
    assert stub._decode_graph.model is stub.model
    assert stub._decode_graph.model_config is cfg
    assert stub._decode_graph.kwargs["mode"] == "graph"

    stub._decode_graph_mode = lambda: "eager"
    PSM._init_decode_graph(stub)
    assert stub._decode_graph.modes == ["graph", "eager"]


# --------------------------------------------------------------------------- #
#  T6 — the 48B (no attn residuals) is untouched                               #
# --------------------------------------------------------------------------- #
def test_no_wiring_without_attn_res_block_size():
    _, kl, cfg = _build_pair(torch.float32)
    for layer in kl.layers:
        layer.__dict__.pop("forward", None)       # undo _build_pair's injection

    cfg_48b = copy.copy(cfg)
    cfg_48b.attn_res_block_size = None
    PSM._config_block_residual(
        types.SimpleNamespace(loaded_model_config=cfg_48b,
                              model=types.SimpleNamespace(model=kl),
                              rank=1)
    )
    for i, layer in enumerate(kl.layers):
        assert "forward" not in layer.__dict__, (
            "layer {} forward was replaced for a config with no Block "
            "Attention Residuals".format(i))
