"""Step 4 of the three-step extraction: one prefill model loop.

The prepacked driver (``PrefillEngine.prefill_prepacked``) and the pooled
driver (``PrefillEngine._prefill_prepacked_pool``) used to spell the model
loop (embed -> layers -> depth-mix -> norm -> tail gather -> lm_head) twice.
Both now drive the one ``PrefillEngine._prefill_model_pass``; the drivers
keep batch assembly, worker->model state binding, token selection, trace
emission and metrics.

These are the behaviour tests that replace the retired duplicate's pins:

* structurally, the pass is the ONLY prefill code that drives decoder
  layers, and each driver calls it exactly once with its own tracer, flag
  value and (prepacked only) layer probe;
* runtime oracles on the real method, compiled in isolation and driven with
  CPU fakes: op order, the depth-mix/release contract, the tail gather with
  and without a sharded manager, the gather-side trace record, the lm_head
  variants, and the segments-only (no tails) arm;
* the one spelling change the unification made -- the no-gather branch norms
  the gathered tail rows instead of slicing the normed full hidden -- is
  covered by a bitwise RMSNorm row-independence oracle on the real
  ``KimiRMSNorm``.

CPU-only: no CUDA, no batchgen import.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "batchgen" / "worker" / "prefill_engine.py"
ENGINE_SOURCE = ENGINE.read_text()
ENGINE_TREE = ast.parse(ENGINE_SOURCE)
KIMI_MODEL = (
    ROOT / "batchgen" / "models" / "moonshotai" / "kimi_linear" / "model.py"
)

sys.path.insert(0, str(ROOT))
try:
    from tests.worker_method_source import (
        load_prefill_engine_methods,
        load_worker_methods,
    )
finally:
    sys.path.pop(0)


def _engine_functions():
    klass = next(
        node for node in ENGINE_TREE.body
        if isinstance(node, ast.ClassDef) and node.name == "PrefillEngine"
    )
    return {
        node.name: node for node in klass.body
        if isinstance(node, ast.FunctionDef)
    }


def _calls_of(node, attr):
    """Calls of ``attr``, whichever route the caller reaches it by.

    D7.5 converted ``_prefill_model_pass`` and the pooled driver: a converted
    body receives the operations it calls as keyword-only collaborators, so the
    callee is a bare ``Name`` there and an ``Attribute`` in a still
    worker-shaped body. Both are the same call.
    """
    return [
        n for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and (
            (isinstance(n.func, ast.Attribute) and n.func.attr == attr)
            or (isinstance(n.func, ast.Name) and n.func.id == attr)
        )
    ]


# --------------------------------------------------------------------------- #
#  (1) the pass is the only prefill model loop; each driver drives it once
# --------------------------------------------------------------------------- #


def test_the_model_pass_is_the_only_decoder_layer_driver():
    drivers = {
        name: [
            n for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and ast.unparse(n.func) == "decoder_layer"
        ]
        for name, fn in _engine_functions().items()
    }
    with_layers = sorted(name for name, calls in drivers.items() if calls)
    assert with_layers == ["_prefill_model_pass"]
    # Both body variants (plain, block-residual) in the one function.
    assert len(drivers["_prefill_model_pass"]) == 2


@pytest.mark.parametrize("driver", ["prefill_prepacked", "_prefill_prepacked_pool"])
def test_each_driver_calls_the_pass_exactly_once_and_keeps_no_epilogue(driver):
    fn = _engine_functions()[driver]
    assert len(_calls_of(fn, "_prefill_model_pass")) == 1
    # No second copy of the epilogue survives in the driver.
    assert not _calls_of(fn, "embed_tokens")
    assert not _calls_of(fn, "norm")
    assert not _calls_of(fn, "_apply_output_attn_res")
    assert not [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Attribute) and n.attr == "lm_head"
    ]
    # Selection stays with the driver, consuming the pass's logits.
    assert _calls_of(fn, "_select_tokens")


def test_the_drivers_hand_the_pass_their_own_seams():
    functions = _engine_functions()
    prepacked = _calls_of(functions["prefill_prepacked"], "_prefill_model_pass")[0]
    pool = _calls_of(functions["_prefill_prepacked_pool"], "_prefill_model_pass")[0]
    prepacked_kwargs = {kw.arg: ast.unparse(kw.value) for kw in prepacked.keywords}
    pool_kwargs = {kw.arg: ast.unparse(kw.value) for kw in pool.keywords}

    # The unpooled driver's debug read and env-gated memory probe go in as
    # values; the pooled driver has never read the lm_head FP32 key
    # (behaviour-preserving: kept and reported, not resolved here) and has
    # no layer probe.
    assert prepacked_kwargs == {
        "lmhead_fp32": "_lmhead_fp32",
        "trace": "unpooled_trace",
        "layer_probe": "_memprof_probe",
        # D7.5: the pass's one collaborator, which it no longer reaches
        # through a worker. The pooled driver does not pass it -- its own
        # delegator resolves it.
        "_record_unsharded_epilogue_gather": (
            "self._record_unsharded_epilogue_gather"
        ),
    }
    assert pool_kwargs == {
        "lmhead_fp32": "False",
        "trace": "epilogue_trace",
    }
    # The pooled chunk's tail arm: segments-only chunks pass None. Since D7.5
    # the pass takes a declared context, so no driver hands it a worker: the
    # CONVERTED pooled driver receives the pass itself as a keyword-only
    # collaborator and passes the chunk ids and the tail arm, nothing else.
    assert isinstance(pool.func, ast.Name)
    assert [ast.unparse(arg) for arg in pool.args] == [
        "rows.input_ids",
        "rows.tail_rows if rows.tail_wave_indices else None",
    ]
    # The still worker-shaped unpooled driver (D7.6) keeps the direct engine
    # call and builds the pass's context at the call site, out of the two
    # attributes it already reads -- routing it through the worker's delegator
    # instead would make the engine file name a worker attribute it has never
    # named, which the boundary census counts as a rise.
    assert ast.unparse(prepacked.func) == "_engine._prefill_model_pass"
    assert ast.unparse(prepacked.args[0]) == (
        "PrefillModelPassContext(model=self.model, "
        "parallel_manager=self.parallel_manager)"
    )


# --------------------------------------------------------------------------- #
#  (2) runtime oracles on the real pass (compiled in isolation, CPU fakes)
# --------------------------------------------------------------------------- #


def _load_pass():
    return load_prefill_engine_methods(
        ["_prefill_model_pass"], {"torch": torch}
    )["_prefill_model_pass"]


class _Layer:
    """A decoder layer fake that records its calls and shifts the hidden."""

    def __init__(self, delta, events, name, attn_res=False):
        self.delta = delta
        self.events = events
        self.name = name
        self.attn_res = attn_res

    def __call__(self, hidden_states, **kwargs):
        self.events.append((self.name, dict(kwargs)))
        out = hidden_states + self.delta
        if self.attn_res:
            return out, kwargs["block_residual"]
        return (out,)


def _fake_worker(
    *,
    events,
    num_tokens=6,
    hidden=8,
    vocab=16,
    attn_res=False,
    gather=None,
):
    torch.manual_seed(0)
    embedding = nn.Embedding(vocab, hidden)
    lm_weight = torch.randn(vocab, hidden)

    inner = SimpleNamespace(
        embed_tokens=lambda ids: (
            events.append(("embed", ids)),
            embedding(ids),
        )[1],
        layers=[
            _Layer(1.0, events, "layer0", attn_res),
            _Layer(2.0, events, "layer1", attn_res),
        ],
        norm=lambda x: (events.append(("norm", x.shape)), x * 0.5)[1],
    )
    if attn_res:
        inner.use_attn_residuals = True
        residual = object()
        inner._new_block_residual = lambda hidden_states: (
            events.append(("seed", hidden_states.shape)),
            residual,
        )[1]
        inner._apply_output_attn_res = lambda flat, block_residual: (
            events.append(("mix", block_residual)),
            flat * 2.0,
        )[1]
        inner._release_block_residual = lambda: events.append(("release",))

    manager = SimpleNamespace()
    if gather is not None:
        manager.gather_prefill_last_token_hidden = gather

    worker = SimpleNamespace(
        model=SimpleNamespace(model=inner, lm_head=SimpleNamespace(weight=lm_weight)),
        parallel_manager=manager,
        _record_unsharded_epilogue_gather=lambda trace, selected, *, num_global_rows: (
            events.append(("unsharded_record", trace, selected, num_global_rows))
        ),
    )
    worker._embedding = embedding
    worker._lm_weight = lm_weight
    worker._ids = torch.arange(num_tokens, dtype=torch.int64) % vocab
    return worker


def test_the_pass_reproduces_the_pre_change_plain_epilogue():
    """embed -> layers -> norm(tails) -> bf16-style lm_head .float(),
    numerically equal to the hand-written reference of the retired loops."""
    events = []
    worker = _fake_worker(events=events)
    tail_rows = torch.tensor([2, 5])
    logits = _load_pass()(worker, worker._ids, tail_rows, lmhead_fp32=False)

    hidden = worker._embedding(worker._ids).unsqueeze(0) + 3.0  # both layers
    expected = torch.nn.functional.linear(
        (hidden[:, tail_rows, :] * 0.5)[0], worker._lm_weight, None
    ).float()
    assert torch.equal(logits, expected)
    assert [event[0] for event in events] == ["embed", "layer0", "layer1", "norm"]
    # Every layer sees the retired loops' exact drive contract.
    for name, kwargs in [event for event in events if event[0].startswith("layer")]:
        assert kwargs == {
            "attention_mask": None,
            "position_ids": None,
            "past_key_value": None,
            "output_attentions": False,
            "use_cache": False,
        }
    # The norm ran on the gathered tail rows only (step 4's one respelling;
    # bitwise equality is the RMSNorm oracle below).
    assert events[-1] == ("norm", torch.Size([1, 2, 8]))


def test_the_pass_keeps_the_depth_mix_then_norm_order_and_releases_once():
    events = []
    worker = _fake_worker(events=events, attn_res=True)
    tail_rows = torch.tensor([0, 4])
    logits = _load_pass()(worker, worker._ids, tail_rows, lmhead_fp32=False)
    assert logits is not None

    names = [event[0] for event in events]
    assert names == [
        "embed", "seed", "layer0", "layer1", "mix", "release", "norm",
    ]
    # The mix consumed the block residual the layers carried: the fake
    # layers hand the seeded sentinel through unchanged, so the mix must
    # have received that exact object.
    mix = next(event for event in events if event[0] == "mix")
    layer_kwargs = [event[1] for event in events if event[0] == "layer0"]
    assert layer_kwargs[0]["block_residual"] is mix[1]


def test_a_segments_only_chunk_runs_every_layer_and_returns_none():
    """The pooled wave's no-tails arm: KV offloads are the product, so the
    layers run, the residual buffer is still released, and there is no
    norm / gather / lm_head."""
    events = []
    worker = _fake_worker(events=events, attn_res=True)
    out = _load_pass()(worker, worker._ids, None, lmhead_fp32=False)

    assert out is None
    names = [event[0] for event in events]
    assert names == ["embed", "seed", "layer0", "layer1", "release"]


def test_the_sharded_manager_gets_the_normed_full_hidden_and_the_trace():
    gather_calls = []

    def gather(hidden_states, tail_rows, num_global_rows, **kwargs):
        gather_calls.append((hidden_states, tail_rows, num_global_rows, kwargs))
        return hidden_states[0, tail_rows, :]

    events = []
    worker = _fake_worker(events=events, gather=gather)
    tail_rows = torch.tensor([1, 3])
    trace = object()
    _load_pass()(worker, worker._ids, tail_rows, lmhead_fp32=False, trace=trace)

    (hidden_states, rows, num_global_rows, kwargs), = gather_calls
    # The gather sees the WHOLE normed hidden (it owns the shard logic) ...
    assert ("norm", torch.Size([1, 6, 8])) in events
    assert hidden_states.shape == (1, 6, 8)
    assert rows is tail_rows and num_global_rows == 6
    # ... and the armed tracer goes INTO it; no hand-written record outside.
    assert kwargs == {"trace": trace}
    assert not [event for event in events if event[0] == "unsharded_record"]

    gather_calls.clear()
    _load_pass()(worker, worker._ids, tail_rows, lmhead_fp32=False, trace=None)
    assert gather_calls[0][3] == {}


def test_the_unsharded_record_goes_through_the_asserted_helper_only():
    events = []
    worker = _fake_worker(events=events)
    tail_rows = torch.tensor([2, 5])
    trace = object()
    logits = _load_pass()(
        worker, worker._ids, tail_rows, lmhead_fp32=False, trace=trace
    )
    assert logits is not None
    records = [event for event in events if event[0] == "unsharded_record"]
    assert len(records) == 1
    _, seen_trace, selected, num_global_rows = records[0]
    assert seen_trace is trace
    assert selected.shape == (2, 8)
    assert num_global_rows == 6

    events.clear()
    _load_pass()(worker, worker._ids, tail_rows, lmhead_fp32=False, trace=None)
    assert not [event for event in events if event[0] == "unsharded_record"]


def test_the_fp32_debug_arm_matches_the_retired_unpooled_branch():
    events = []
    worker = _fake_worker(events=events)
    tail_rows = torch.tensor([0])
    fp32 = _load_pass()(worker, worker._ids, tail_rows, lmhead_fp32=True)
    default = _load_pass()(worker, worker._ids, tail_rows, lmhead_fp32=False)

    hidden = worker._embedding(worker._ids).unsqueeze(0) + 3.0
    last = (hidden[:, tail_rows, :] * 0.5)[0]
    assert torch.equal(
        fp32,
        torch.nn.functional.linear(last.float(), worker._lm_weight.float(), None),
    )
    assert torch.equal(
        default,
        torch.nn.functional.linear(last, worker._lm_weight, None).float(),
    )


def test_the_vocab_parallel_head_is_driven_through_its_forward():
    events = []
    worker = _fake_worker(events=events)
    head_calls = []
    gathered = torch.randn(2, 16, dtype=torch.float64)
    worker.model.lm_head = SimpleNamespace(
        tp_size=8,
        forward=lambda last: (head_calls.append(last), gathered)[1],
    )
    logits = _load_pass()(
        worker, worker._ids, torch.tensor([2, 5]), lmhead_fp32=False
    )
    assert len(head_calls) == 1
    assert torch.equal(logits, gathered.float())


def test_the_layer_probe_observes_every_layer_in_order():
    events = []
    worker = _fake_worker(events=events, attn_res=True)
    seen = []
    _load_pass()(
        worker, worker._ids, torch.tensor([1]), lmhead_fp32=False,
        layer_probe=lambda layer_idx, decoder_layer, block_residual: seen.append(
            (layer_idx, decoder_layer.name)
        ),
    )
    assert seen == [(0, "layer0"), (1, "layer1")]


# --------------------------------------------------------------------------- #
#  (3) the one respelling has a bitwise oracle: RMSNorm is row-independent
# --------------------------------------------------------------------------- #


def _lift_kimi_rmsnorm():
    tree = ast.parse(KIMI_MODEL.read_text())
    node = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "KimiRMSNorm"
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch, "nn": nn}
    exec(compile(module, str(KIMI_MODEL), "exec"), namespace)  # noqa: S102
    return namespace["KimiRMSNorm"]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_norming_the_gathered_rows_equals_gathering_the_normed_rows(dtype):
    """The retired prepacked no-gather branch normed the FULL hidden and
    sliced; the one pass norms the gathered tail rows. RMSNorm reduces per
    row, so the two spellings are bitwise identical -- held here on the real
    ``KimiRMSNorm`` (eager arm, the one a CPU/non-fused input takes)."""
    norm = _lift_kimi_rmsnorm()(hidden_size=32).to(dtype)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(32))
    torch.manual_seed(7)
    full = torch.randn(1, 64, 32).to(dtype)
    rows = torch.tensor([0, 3, 17, 63])
    with torch.no_grad():
        assert torch.equal(norm(full)[:, rows, :], norm(full[:, rows, :]))


# --------------------------------------------------------------------------- #
#  (4) the worker's delegator hands the pass to the engine unchanged
# --------------------------------------------------------------------------- #


def test_the_worker_delegator_builds_the_passs_context_and_collaborator():
    """D7.5: the delegator resolves the pass's two values and its one callable.

    The pass no longer receives the worker, so the claim this replaces ("the
    delegator forwards the worker unchanged") is no longer true of anything.
    What it forwards now is the declared context -- read LIVE, because the
    model is bound and dropped with the phase -- plus the unsharded-gather
    record, which keeps its own read set and therefore stays a callable.
    """
    import sys as _sys
    from types import ModuleType

    seen = {}

    class Engine:
        def _prefill_model_pass(
            self, ctx, input_ids, tail_rows, *, lmhead_fp32,
            trace=None, layer_probe=None,
            _record_unsharded_epilogue_gather,
        ):
            seen.update(
                ctx=ctx, input_ids=input_ids, tail_rows=tail_rows,
                lmhead_fp32=lmhead_fp32, trace=trace, layer_probe=layer_probe,
                record=_record_unsharded_epilogue_gather,
            )
            return "logits"

    class PrefillModelPassContext:
        def __init__(self, *, model, parallel_manager):
            self.model, self.parallel_manager = model, parallel_manager

    module = ModuleType("batchgen.worker.prefill_engine")
    module.PrefillModelPassContext = PrefillModelPassContext
    saved = _sys.modules.get("batchgen.worker.prefill_engine")
    _sys.modules["batchgen.worker.prefill_engine"] = module
    try:
        delegator = load_worker_methods(["_prefill_model_pass"], {})[
            "_prefill_model_pass"
        ]
        model, manager = object(), object()
        record = lambda *a, **k: None  # noqa: E731
        worker = SimpleNamespace(
            _prefill_engine=Engine(), model=model, parallel_manager=manager,
            _record_unsharded_epilogue_gather=record,
        )
        ids, rows, trace, probe = object(), object(), object(), object()
        assert delegator(
            worker, ids, rows, lmhead_fp32=True, trace=trace,
            layer_probe=probe,
        ) == "logits"
    finally:
        if saved is None:
            del _sys.modules["batchgen.worker.prefill_engine"]
        else:
            _sys.modules["batchgen.worker.prefill_engine"] = saved

    ctx = seen.pop("ctx")
    assert isinstance(ctx, PrefillModelPassContext)
    assert ctx.model is model and ctx.parallel_manager is manager
    assert seen == {
        "input_ids": ids, "tail_rows": rows, "lmhead_fp32": True,
        "trace": trace, "layer_probe": probe, "record": record,
    }
