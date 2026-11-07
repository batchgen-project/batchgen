"""CPU contract tests for the GLM-5 decode criticality probe.

The probe inserts spin nodes into the captured whole-model decode graph at
fixed sites and sweeps their duration across replays of that one graph
(``batchgen/models/glm/glm5/criticality_probe.py``). What is checkable on a
CPU box:

* the ``batchgen_debug.glm5_criticality_probe`` request is parsed strictly
  (unknown site/key, non-integer, negative or over-cap delay, empty schedule
  all fail loud) and the schedule maps steps to levels with cycling;
* the probe launches exactly one spin node per enabled site and selected
  layer, counts them per forward, installs a level with ONE async copy per
  level change, drops the post-change steps, and fits the slope;
* the INACTIVE path: with no probe installed every instrumented forward issues
  exactly the calls it issued before the probe existed (a recorded runtime
  snapshot for the decoder layer, a source-order call snapshot for every
  instrumented method), and with a probe the spin call appears once per
  matching layer at its site;
* placement rules the GPU cannot be trusted to reveal: side-stream sites sit
  inside their ``fork()``, the router site follows the gate.

The DeepEP MoE runtime ordering is pinned in tests/test_glm52_deepep_ll_graph.py
and the full-DSA runtime ordering (GPU) in
tests/test_glm5_full_dsa_graph_segment.py.
"""

from __future__ import annotations

import ast
import hashlib
import math
import pathlib
import subprocess
import sys
import types

import pytest
import torch

import batchgen.attention.fused_kernels as fused_kernels
import batchgen_kernels.attention.add_rmsnorm_fp8_quant as add_norm_fp8
from batchgen.models.glm.glm5.criticality_probe import (
    DEBUG_KEY,
    MAX_DELAY_US,
    MOE_SITES,
    SITE_INDEX,
    SITES,
    CriticalityProbe,
    CriticalityProbeSpec,
)
from batchgen.models.glm.glm5.layer_cuda_graph_segments import (
    Glm5DecoderLayerGraphSegment,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_GLM5 = _REPO_ROOT / "batchgen/models/glm/glm5"


def _spec(**overrides):
    raw = {
        "sites": ["after_router", "after_combine"],
        "layers": "all",
        "schedule": [
            {"steps": 4, "delay_us": {}},
            {"steps": 4, "delay_us": {"after_router": 10}},
            {"steps": 4, "delay_us": {"after_router": 30}},
        ],
        "skip_steps_after_change": 1,
    }
    raw.update(overrides)
    return raw


# --------------------------------------------------------------------------- #
# Spec parsing
# --------------------------------------------------------------------------- #

def test_site_table_is_fixed():
    # The index of each site is baked into captured spin nodes; appending is
    # the only legal change.
    assert SITES == (
        "after_input_norm_quant", "after_q_a_norm_quant", "after_indexer",
        "after_indexer_k_path", "after_attention", "after_post_attn_norm_quant",
        "after_router", "after_dispatch", "after_expert_gemm", "after_combine",
        "after_shared_expert",
    )
    assert MOE_SITES <= set(SITES)
    assert MAX_DELAY_US == 2000


def test_valid_spec_parses_into_sites_order_and_ns():
    spec = CriticalityProbeSpec.parse(_spec(sites=["after_combine", "after_router"]))
    assert spec.sites == ("after_router", "after_combine")
    assert spec.layers is None
    assert spec.skip_steps_after_change == 1
    assert spec.num_levels == 3 and spec.period_steps == 12
    assert spec.schedule[1][1][SITE_INDEX["after_router"]] == 10_000
    assert spec.schedule[1][1][SITE_INDEX["after_combine"]] == 0
    assert all(len(delays) == len(SITES) for _, delays in spec.schedule)


def test_from_batchgen_debug_is_none_without_the_key():
    assert CriticalityProbeSpec.from_batchgen_debug(None) is None
    assert CriticalityProbeSpec.from_batchgen_debug({}) is None
    assert CriticalityProbeSpec.from_batchgen_debug({"glm5_dispatch_trace": True}) is None
    assert CriticalityProbeSpec.from_batchgen_debug({DEBUG_KEY: None}) is None
    spec = CriticalityProbeSpec.from_batchgen_debug({DEBUG_KEY: _spec()})
    assert spec == CriticalityProbeSpec.parse(_spec())


@pytest.mark.parametrize(
    "raw, match",
    [
        ("after_router", "must be an object"),
        (_spec(sites=["after_routr"]), "unknown site 'after_routr'"),
        (_spec(sites=[]), "non-empty list"),
        (_spec(sites=["after_router", "after_router"]), "twice"),
        (_spec(bogus=1), "unknown keys"),
        (_spec(layers=[]), "layers"),
        (_spec(layers=[-1]), "non-negative"),
        (_spec(layers=[True]), "non-negative"),
        (_spec(layers=[3, 3]), "twice"),
        (_spec(schedule=[]), "non-empty list of levels"),
        (_spec(schedule=None), "non-empty list of levels"),
        (_spec(schedule=[{"steps": 0, "delay_us": {}}]), "positive integer"),
        (_spec(schedule=[{"steps": 4, "delay_us": {}, "x": 1}]), "unknown keys"),
        (_spec(schedule=[{"steps": 4, "delay_us": {"after_router": 1.5}}]), "integer number"),
        (_spec(schedule=[{"steps": 4, "delay_us": {"after_router": 10.0}}]), "integer number"),
        (_spec(schedule=[{"steps": 4, "delay_us": {"after_router": True}}]), "integer number"),
        (_spec(schedule=[{"steps": 4, "delay_us": {"after_router": "10"}}]), "integer number"),
        (_spec(schedule=[{"steps": 4, "delay_us": {"after_router": -1}}]), "outside"),
        (_spec(schedule=[{"steps": 4, "delay_us": {"after_router": MAX_DELAY_US + 1}}]), "outside"),
        (_spec(schedule=[{"steps": 4, "delay_us": {"after_dispatch": 10}}]), "not in"),
        (_spec(schedule=[{"steps": 4, "delay_us": {"nope": 10}}]), "unknown site"),
        (_spec(skip_steps_after_change=-1), "non-negative"),
        (_spec(skip_steps_after_change=4), "leaves no measured step"),
    ],
)
def test_invalid_spec_fails_loud(raw, match):
    with pytest.raises(ValueError, match=match):
        CriticalityProbeSpec.parse(raw)


def test_cap_is_inclusive():
    spec = CriticalityProbeSpec.parse(
        _spec(schedule=[{"steps": 4, "delay_us": {"after_router": MAX_DELAY_US}}])
    )
    assert spec.schedule[0][1][SITE_INDEX["after_router"]] == MAX_DELAY_US * 1000


def test_level_lookup_and_cycling():
    spec = CriticalityProbeSpec.parse(_spec())
    got = [spec.level_at(step) for step in range(1, 26)]
    one_period = [(level, offset) for level in range(3) for offset in range(4)]
    assert got[:12] == one_period
    assert got[12:24] == one_period
    assert got[24] == (0, 0)
    with pytest.raises(ValueError):
        spec.level_at(0)


def test_signature_equality():
    a = CriticalityProbeSpec.parse(_spec())
    b = CriticalityProbeSpec.parse(_spec(sites=["after_combine", "after_router"]))
    assert a.signature() == b.signature() and hash(a.signature()) == hash(b.signature())
    c = CriticalityProbeSpec.parse(_spec(skip_steps_after_change=2))
    d = CriticalityProbeSpec.parse(_spec(layers=[0, 1]))
    e = CriticalityProbeSpec.parse(_spec(sites=["after_router"], schedule=[{"steps": 4}]))
    assert len({a.signature(), c.signature(), d.signature(), e.signature()}) == 4


# --------------------------------------------------------------------------- #
# Probe (CPU device, recording launcher)
# --------------------------------------------------------------------------- #

def _probe(raw=None, log=None):
    log = [] if log is None else log
    spec = CriticalityProbeSpec.parse(raw or _spec())

    def launch(table, site):
        log.append(("spin", SITES[site]))

    return CriticalityProbe(spec, torch.device("cpu"), launch=launch), log


def test_mark_filters_sites_and_layers_and_counts_per_forward():
    probe, log = _probe(_spec(layers=[1, 3]))
    probe.begin_forward()
    for layer in range(5):
        probe.mark("after_router", layer)
        probe.mark("after_dispatch", layer)   # not an enabled site
        probe.mark("after_combine", layer)
    probe.end_forward()
    assert log == [
        ("spin", "after_router"), ("spin", "after_combine"),
        ("spin", "after_router"), ("spin", "after_combine"),
    ]
    marks = dict(zip(SITES, probe.layer_marks))
    assert marks["after_router"] == 2 and marks["after_combine"] == 2
    assert marks["after_dispatch"] == 0
    with pytest.raises(ValueError, match="unknown site"):
        probe.mark("after_nothing", 0)


def test_forwards_must_mark_identically():
    probe, _ = _probe()
    probe.begin_forward()
    probe.mark("after_router", 0)
    probe.end_forward()
    probe.begin_forward()
    probe.mark("after_router", 0)
    probe.end_forward()
    probe.begin_forward()
    probe.mark("after_router", 0)
    probe.mark("after_router", 1)
    with pytest.raises(RuntimeError, match="different spin-node counts"):
        probe.end_forward()


def test_layers_must_exist():
    probe, _ = _probe(_spec(layers=[0, 77]))
    probe.validate_num_layers(78)
    with pytest.raises(ValueError, match="out of range"):
        probe.validate_num_layers(77)


class _CopyRecorder:
    """Stands in for the device delay table: records every copy_ call."""

    def __init__(self):
        self.copies = []

    def copy_(self, src, non_blocking=False):
        self.copies.append((src.clone(), non_blocking))
        return self


def test_before_replay_copies_once_per_level_change():
    probe, _ = _probe()
    rows = probe.levels_host.clone()
    recorder = _CopyRecorder()
    probe.delay_dev = recorder
    tags = [probe.before_replay(step) for step in range(1, 14)]
    assert tags[:5] == [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)]
    assert tags[12] == (0, 0)
    # Levels 0, 1, 2 and level 0 again after the wrap: four copies, each the
    # level's pinned row, each asynchronous.
    assert len(recorder.copies) == 4
    for (src, non_blocking), level in zip(recorder.copies, (0, 1, 2, 0)):
        assert torch.equal(src, rows[level]) and non_blocking is True
    # A new round reinstalls its first level even if the table holds it.
    probe.begin_round()
    probe.before_replay(1)
    assert len(recorder.copies) == 5
    # The host rows are written once and never again.
    assert torch.equal(probe.levels_host, rows)


def test_before_replay_installs_the_level_values():
    probe, _ = _probe()
    probe.before_replay(5)
    assert probe.delay_dev[SITE_INDEX["after_router"]].item() == 10_000
    probe.before_replay(9)
    assert probe.delay_dev[SITE_INDEX["after_router"]].item() == 30_000
    assert probe.delay_dev.sum().item() == 30_000


def _fill(probe, fn, periods=3):
    for step in range(1, probe.spec.period_steps * periods + 1):
        tag = probe.before_replay(step)
        level = tag[0]
        probe.record(tag, fn(step, level))


def test_record_skips_post_change_steps_and_untagged():
    probe, _ = _probe()
    probe.record(None, 99.0)
    _fill(probe, lambda step, level: 1.0 + level)
    stats = probe._stats
    # 3 periods x 4 steps per level, the first step of each occurrence dropped.
    assert [s.count for s in stats] == [9, 9, 9]
    assert [s.skipped for s in stats] == [3, 3, 3]
    assert stats[2].min_ms == stats[2].max_ms == 3.0


def test_slope_recovers_a_known_criticality():
    probe, _ = _probe()
    probe.begin_forward()
    for layer in range(75):
        probe.mark("after_router", layer)
    probe.end_forward()
    # Injected per step: 75 x 0 / 10 / 30 us. Step time = 20 ms + 0.6 x that,
    # plus a symmetric +-0.01 ms wobble so the residual is non-zero.
    def step_ms(step, level):
        injected_ms = 75 * (0, 10, 30)[level] / 1000.0
        return 20.0 + 0.6 * injected_ms + (0.01 if step % 2 else -0.01)

    _fill(probe, step_ms, periods=4)
    fit = probe.site_slope("after_router")
    assert fit["marks_per_step"] == 75
    assert fit["slope"] == pytest.approx(0.6, abs=1e-3)
    assert fit["intercept_ms"] == pytest.approx(20.0, abs=1e-2)
    assert fit["stderr"] is not None and fit["stderr"] < 0.01
    assert fit["max_residual_us"] < 5.0
    report = probe.report()
    assert "site=after_router L_site=75 slope=0.6" in report
    assert "level=2" in report and "injected_per_step=after_router=2250.0us" in report


def test_zero_marks_report_instead_of_dividing():
    probe, _ = _probe()
    _fill(probe, lambda step, level: 5.0)
    assert "marks_per_step=none (no forward was captured)" in probe.report()
    probe.begin_forward()
    probe.end_forward()
    fit = probe.site_slope("after_router")
    assert fit["slope"] is None and "L_site=0" in fit["reason"]
    report = probe.report()
    assert "site=after_router L_site=0 slope=undefined" in report
    assert "site=after_combine L_site=0 slope=undefined" in report


def test_a_site_without_its_own_sweep_has_no_slope():
    probe, _ = _probe()
    probe.begin_forward()
    probe.mark("after_combine", 0)
    probe.end_forward()
    _fill(probe, lambda step, level: 5.0)
    # after_combine is 0 at every level: one distinct injected delay.
    assert probe.site_slope("after_combine")["slope"] is None


def test_probe_module_does_not_import_the_spin_extension():
    code = (
        "import sys\n"
        "import batchgen.models.glm.glm5.criticality_probe as m\n"
        "spec = m.CriticalityProbeSpec.from_batchgen_debug({})\n"
        "assert spec is None\n"
        "bad = [n for n in sys.modules if 'spin_delay' in n]\n"
        "assert not bad, bad\n"
    )
    subprocess.run(
        [sys.executable, "-c", code], cwd=_REPO_ROOT, check=True,
        env={"PYTHONPATH": str(_REPO_ROOT), "PATH": "/usr/bin:/bin"},
    )


def test_only_the_probe_constructor_imports_the_spin_extension():
    hits = []
    for path in sorted((_REPO_ROOT / "batchgen").rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and "spin_delay" in node.module:
                hits.append((path.relative_to(_REPO_ROOT).as_posix(), node.lineno))
    assert len(hits) == 1 and hits[0][0] == "batchgen/models/glm/glm5/criticality_probe.py"
    init = _method(_GLM5 / "criticality_probe.py", "CriticalityProbe", "__init__")
    assert any(
        isinstance(n, ast.ImportFrom) and "spin_delay" in (n.module or "")
        for n in ast.walk(init)
    )


# --------------------------------------------------------------------------- #
# Inactive path: the pre-change call sequences
# --------------------------------------------------------------------------- #

def _method(path, class_name, method_name):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"{class_name}.{method_name} not found in {path}")


def _probe_call(node):
    return ast.unparse(node.func).startswith("self.criticality_probe.")


def _source_order_calls(fn):
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
    calls.sort(key=lambda n: (n.lineno, n.col_offset))
    return calls


# Captured from main 74e8a3f0, BEFORE the probe hooks existed: the number of
# calls in each instrumented method, in source order, and the sha256 of their
# callee expressions joined by newlines. With the probe's own calls removed the
# current methods must reproduce them exactly, i.e. the probe adds calls and
# nothing else.
_PRE_PROBE_CALL_SNAPSHOT = {
    ("cuda_graph_segments.py", "Glm5FullDsaAttnSegment", "forward"):
        (60, "33b3e845aeb6b44fe110c98044381da8a54c9a6016c16e8d2e926d741f47b7f8"),
    ("reuse_topk_segment.py", "Glm5ReuseTopkAttnSegment", "forward"):
        (35, "a10f74043a783277001b673b9554e28230fa302976a4bb8be1163c69eb836bb1"),
    ("layer_cuda_graph_segments.py", "Glm5DecoderLayerGraphSegment", "forward"):
        (18, "f8687608a3ab9fdb0d77149e422f1b0f20c25c4512fd29693274dcc6bd5473c7"),
    ("moe_cuda_graph_segments.py", "Glm5MoEGraphSegment", "forward"):
        (18, "af3676815db3a625036c84b10c12657859235c27bc3e986f6c341352162df7c5"),
    ("moe_cuda_graph_segments.py", "Glm5MoEGraphSegment", "_forward_deepep"):
        (24, "dc71b3a5d36cca3589eedda08d8fcc47dd2084b885ac216c8e0d63bd18db3cd8"),
    ("whole_model_cuda_graph_segments.py", "Glm5WholeModelSegment", "forward"):
        (3, "9ab482594f79ed5495d60d83cc9d641a3dc3f87af63f337cb42e1303469d68a2"),
    ("whole_model_cuda_graph_segments.py", "Glm5WholeModelSegment", "run_model_with_probes"):
        (15, "dfa472e1a41f3eacf5da638d7c31d13103ef1f5321259f7caa2a3de7b81990a0"),
}


@pytest.mark.parametrize("key", sorted(_PRE_PROBE_CALL_SNAPSHOT))
def test_inactive_call_sequence_matches_the_pre_probe_snapshot(key):
    filename, class_name, method_name = key
    fn = _method(_GLM5 / filename, class_name, method_name)
    calls = [ast.unparse(n.func) for n in _source_order_calls(fn) if not _probe_call(n)]
    count, digest = _PRE_PROBE_CALL_SNAPSHOT[key]
    assert len(calls) == count
    assert hashlib.sha256("\n".join(calls).encode()).hexdigest() == digest


def _probe_guards(fn):
    """Every ``if self.criticality_probe is not None`` in ``fn``."""
    return [
        n for n in ast.walk(fn)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "self.criticality_probe is not None"
    ]


@pytest.mark.parametrize("key", sorted(_PRE_PROBE_CALL_SNAPSHOT))
def test_every_probe_call_is_a_lone_none_guarded_statement(key):
    filename, class_name, method_name = key
    fn = _method(_GLM5 / filename, class_name, method_name)
    probe_calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and _probe_call(n)]
    guards = _probe_guards(fn)
    assert len(guards) == len(probe_calls)
    for guard in guards:
        assert not guard.orelse
        assert len(guard.body) == 1
        stmt = guard.body[0]
        assert isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
        assert _probe_call(stmt.value)


_SITE_OWNERS = {
    "after_input_norm_quant": [("layer_cuda_graph_segments.py", "Glm5DecoderLayerGraphSegment")],
    "after_attention": [("layer_cuda_graph_segments.py", "Glm5DecoderLayerGraphSegment")],
    "after_post_attn_norm_quant": [("layer_cuda_graph_segments.py", "Glm5DecoderLayerGraphSegment")],
    "after_q_a_norm_quant": [
        ("cuda_graph_segments.py", "Glm5FullDsaAttnSegment"),
        ("reuse_topk_segment.py", "Glm5ReuseTopkAttnSegment"),
    ],
    "after_indexer": [("cuda_graph_segments.py", "Glm5FullDsaAttnSegment")],
    "after_indexer_k_path": [("cuda_graph_segments.py", "Glm5FullDsaAttnSegment")],
    "after_router": [("moe_cuda_graph_segments.py", "Glm5MoEGraphSegment")],
    "after_dispatch": [("moe_cuda_graph_segments.py", "Glm5MoEGraphSegment")],
    "after_expert_gemm": [("moe_cuda_graph_segments.py", "Glm5MoEGraphSegment")],
    "after_combine": [("moe_cuda_graph_segments.py", "Glm5MoEGraphSegment")],
    "after_shared_expert": [("moe_cuda_graph_segments.py", "Glm5MoEGraphSegment")],
}


def _marks_in(fn):
    out = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "self.criticality_probe.mark":
            site = node.args[0].value
            out.setdefault(site, []).append(node)
    return out


def test_every_site_is_marked_exactly_once_per_owner():
    assert set(_SITE_OWNERS) == set(SITES)
    seen = {}
    for filename, class_name, method_name in _PRE_PROBE_CALL_SNAPSHOT:
        for site, nodes in _marks_in(_method(_GLM5 / filename, class_name, method_name)).items():
            seen.setdefault(site, []).append((filename, class_name, len(nodes)))
    for site, owners in _SITE_OWNERS.items():
        assert sorted(seen.get(site, [])) == sorted((f, c, 1) for f, c in owners), site


def _with_block(fn, context_src):
    blocks = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.With)
        and any(ast.unparse(item.context_expr) == context_src for item in n.items)
    ]
    assert len(blocks) == 1, context_src
    return blocks[0]


def test_side_stream_sites_are_issued_inside_their_fork():
    dsa = _method(_GLM5 / "cuda_graph_segments.py", "Glm5FullDsaAttnSegment", "forward")
    k_fork = _with_block(dsa, "k_path.fork()")
    inside = {id(n) for n in ast.walk(k_fork)}
    marks = _marks_in(dsa)
    assert id(marks["after_indexer_k_path"][0]) in inside
    for site in ("after_q_a_norm_quant", "after_indexer"):
        assert id(marks[site][0]) not in inside, site
    # Last statement of the fork body: after the K path's last launch.
    last = k_fork.body[-1]
    assert isinstance(last, ast.If) and marks["after_indexer_k_path"][0] in ast.walk(last)

    moe = _method(_GLM5 / "moe_cuda_graph_segments.py", "Glm5MoEGraphSegment", "_forward_deepep")
    s_fork = _with_block(moe, "self._shared_overlap.fork()")
    inside = {id(n) for n in ast.walk(s_fork)}
    marks = _marks_in(moe)
    assert id(marks["after_shared_expert"][0]) in inside
    for site in ("after_router", "after_dispatch", "after_expert_gemm", "after_combine"):
        assert id(marks[site][0]) not in inside, site
    last = s_fork.body[-1]
    assert isinstance(last, ast.If) and marks["after_shared_expert"][0] in ast.walk(last)


def test_moe_sites_follow_their_ops_and_never_split_the_router_gemm_and_gate():
    moe = _method(_GLM5 / "moe_cuda_graph_segments.py", "Glm5MoEGraphSegment", "_forward_deepep")
    order = [
        ("mark:" + n.args[0].value) if ast.unparse(n.func) == "self.criticality_probe.mark"
        else ast.unparse(n.func)
        for n in _source_order_calls(moe)
    ]
    want = [
        "gate_sigmoid_topk_warp_cuda", "mark:after_router",
        "exchange.dispatch_fp8", "mark:after_dispatch",
        "run_deepgemm_masked_moe", "mark:after_expert_gemm",
        "exchange.combine", "mark:after_combine",
        "self._shared_overlap.join",
    ]
    picked = [name for name in order if name in want]
    assert picked == want
    # Nothing between the router GEMM and the gate except the forked shared
    # expert (which lives on the side stream).
    gemm_at = max(i for i, n in enumerate(order) if "router_forward" in n or n == "split_k_router")
    gate_at = order.index("gate_sigmoid_topk_warp_cuda")
    assert not [n for n in order[gemm_at:gate_at] if n == "mark:after_router"]


def test_dsa_sites_follow_their_ops():
    dsa = _method(_GLM5 / "cuda_graph_segments.py", "Glm5FullDsaAttnSegment", "forward")
    order = [
        ("mark:" + n.args[0].value) if ast.unparse(n.func) == "self.criticality_probe.mark"
        else ast.unparse(n.func)
        for n in _source_order_calls(dsa)
    ]
    assert order.index("mark:after_indexer_k_path") > order.index("head_gates_out")
    assert order.index("mark:after_q_a_norm_quant") > order.index("fused_rmsnorm_fp8_quant")
    assert order.index("mark:after_q_a_norm_quant") < order.index("w8a8_deepgemm", order.index("fused_rmsnorm_fp8_quant"))
    assert order.index("self._score_and_topk") < order.index("mark:after_indexer") < order.index("absorb_q_out")


def test_nccl_moe_refuses_the_probe():
    from batchgen.models.glm.glm5.moe_cuda_graph_segments import Glm5MoEGraphSegment

    segment = Glm5MoEGraphSegment.__new__(Glm5MoEGraphSegment)
    segment.deepep_ll = False
    segment.moe = types.SimpleNamespace(layer_idx=4)
    assert segment.criticality_probe is None
    segment.set_criticality_probe(None)
    probe, _ = _probe()
    with pytest.raises(RuntimeError, match="only the DeepEP MoE decode path"):
        segment.set_criticality_probe(probe)
    assert segment.criticality_probe is None
    segment.deepep_ll = True
    segment.set_criticality_probe(probe)
    assert segment.criticality_probe is probe


# --------------------------------------------------------------------------- #
# Decoder layer segment at run time (CPU, stubbed kernels)
# --------------------------------------------------------------------------- #

_H = 8


def _build_layer(monkeypatch, calls, *, fuse_attn, moe, fuse_moe):
    class Norm:
        def __init__(self, name):
            self.name = name
            self.weight = torch.ones(_H, dtype=torch.bfloat16)
            self.eps = 1e-5

        def __call__(self, x):
            calls.append(self.name)
            return x

    class Layer:
        layer_idx = 3
        hidden_size = _H
        input_layernorm = Norm("input_layernorm")
        post_attention_layernorm = Norm("post_attention_layernorm")

        def mlp(self, x):
            calls.append("dense_mlp")
            return x

    class Dsa:
        max_seqlen = 16
        index_topk = 8
        primary_blocked_k = torch.empty(1, 1, 1, 6)
        aux_blocked_k = torch.empty(1, 1, 1, 3)
        consumes_fused_hidden_fp8 = fuse_attn
        criticality_probe = None

        def set_criticality_probe(self, probe):
            self.criticality_probe = probe

        def forward(self, **kw):
            calls.append("dsa.forward")
            b = kw["hidden_states"].shape[0]
            return {
                "attn_output": torch.zeros(b, 1, _H, dtype=torch.bfloat16),
                "primary_k_tensor": torch.zeros(b, 1, 1, 6),
                "indexer_k_tensor": torch.zeros(b, 1, 1, 3),
            }

        def attn_output_row_mask(self, b):
            return None

    class Moe:
        consumes_fused_padded_fp8 = fuse_moe
        criticality_probe = None

        def set_criticality_probe(self, probe):
            self.criticality_probe = probe

        def forward(self, **kw):
            calls.append("moe.forward")
            return {"moe_output": kw["padded"]}

    def fake_add_rmsnorm(residual, hidden, weight, eps, row_valid_mask=None):
        calls.append("cuda_add_rmsnorm")
        return hidden, residual

    def fake_add_rmsnorm_fp8(residual, hidden, weight, eps, **kw):
        calls.append("cuda_add_rmsnorm_fp8_quant")
        return hidden, residual, hidden, hidden

    def fake_act_quant(x, **kw):
        calls.append("act_quant")
        return x, x

    fa3 = types.ModuleType("batchgen.attention.mla.fa3_backend")
    fa3.act_quant = fake_act_quant
    monkeypatch.setitem(sys.modules, "batchgen.attention.mla.fa3_backend", fa3)
    monkeypatch.setattr(fused_kernels, "cuda_add_rmsnorm", fake_add_rmsnorm)
    monkeypatch.setattr(add_norm_fp8, "cuda_add_rmsnorm_fp8_quant", fake_add_rmsnorm_fp8)
    monkeypatch.setattr(add_norm_fp8, "preload_add_rmsnorm_fp8_quant_kernel", lambda: None)
    return Glm5DecoderLayerGraphSegment(
        layer=Layer(), dsa_segment=Dsa(), moe_segment=Moe() if moe else None,
        device=torch.device("cpu"), world_size=2,
    )


def _run_layer(segment, *, first):
    b = 2
    hidden = torch.zeros(b, 1, _H, dtype=torch.bfloat16)
    segment.forward(
        hidden_states=hidden,
        position_ids=torch.zeros(b, 1, dtype=torch.int64),
        cache_seqlens=torch.zeros(b, dtype=torch.int32),
        primary_slot_indices=torch.zeros(b, dtype=torch.int32),
        aux_slot_indices=torch.zeros(b, dtype=torch.int32),
        num_valid_tokens=torch.tensor([b], dtype=torch.int32),
        rank_token_counts=torch.tensor([b, b]),
        residual=None if first else hidden.clone(),
        defer_residual_add=True,
    )


# Recorded from main 74e8a3f0 with the same stubs, BEFORE the probe hooks
# existed. Key: (first layer of the chain, qkv_a norm-quant fused, MoE layer,
# shared-expert norm-quant fused).
_PRE_PROBE_LAYER_CALLS = {
    (True, False, False, False): ["input_layernorm", "dsa.forward", "cuda_add_rmsnorm", "dense_mlp"],
    (True, False, True, False): ["input_layernorm", "dsa.forward", "cuda_add_rmsnorm", "moe.forward"],
    (True, False, True, True): ["input_layernorm", "dsa.forward", "cuda_add_rmsnorm_fp8_quant", "moe.forward"],
    (True, True, False, False): ["input_layernorm", "act_quant", "dsa.forward", "cuda_add_rmsnorm", "dense_mlp"],
    (True, True, True, False): ["input_layernorm", "act_quant", "dsa.forward", "cuda_add_rmsnorm", "moe.forward"],
    (True, True, True, True): ["input_layernorm", "act_quant", "dsa.forward", "cuda_add_rmsnorm_fp8_quant", "moe.forward"],
    (False, False, False, False): ["cuda_add_rmsnorm", "dsa.forward", "cuda_add_rmsnorm", "dense_mlp"],
    (False, False, True, False): ["cuda_add_rmsnorm", "dsa.forward", "cuda_add_rmsnorm", "moe.forward"],
    (False, False, True, True): ["cuda_add_rmsnorm", "dsa.forward", "cuda_add_rmsnorm_fp8_quant", "moe.forward"],
    (False, True, False, False): ["cuda_add_rmsnorm_fp8_quant", "dsa.forward", "cuda_add_rmsnorm", "dense_mlp"],
    (False, True, True, False): ["cuda_add_rmsnorm_fp8_quant", "dsa.forward", "cuda_add_rmsnorm", "moe.forward"],
    (False, True, True, True): ["cuda_add_rmsnorm_fp8_quant", "dsa.forward", "cuda_add_rmsnorm_fp8_quant", "moe.forward"],
}


@pytest.mark.parametrize("key", sorted(_PRE_PROBE_LAYER_CALLS))
def test_layer_without_probe_issues_the_pre_probe_calls(monkeypatch, key):
    first, fuse_attn, moe, fuse_moe = key
    calls = []
    segment = _build_layer(monkeypatch, calls, fuse_attn=fuse_attn, moe=moe, fuse_moe=fuse_moe)
    assert segment.criticality_probe is None
    _run_layer(segment, first=first)
    assert calls == _PRE_PROBE_LAYER_CALLS[key]


_LAYER_SITES = ("after_input_norm_quant", "after_attention", "after_post_attn_norm_quant")


@pytest.mark.parametrize("key", sorted(_PRE_PROBE_LAYER_CALLS))
def test_layer_with_probe_inserts_one_spin_per_site(monkeypatch, key):
    first, fuse_attn, moe, fuse_moe = key
    calls = []
    segment = _build_layer(monkeypatch, calls, fuse_attn=fuse_attn, moe=moe, fuse_moe=fuse_moe)
    probe, _ = _probe(
        {"sites": list(_LAYER_SITES), "schedule": [{"steps": 1}]}, log=calls
    )
    segment.set_criticality_probe(probe)
    assert segment.dsa_segment.criticality_probe is probe
    if moe:
        assert segment.moe_segment.criticality_probe is probe
    _run_layer(segment, first=first)
    want = list(_PRE_PROBE_LAYER_CALLS[key])
    # After the input norm (and its FP8 cast), after the attention segment,
    # after the post-attention norm: exactly one spin each, in that order.
    want.insert(want.index("dsa.forward"), ("spin", "after_input_norm_quant"))
    want.insert(want.index("dsa.forward") + 1, ("spin", "after_attention"))
    want.insert(want.index(("spin", "after_attention")) + 2, ("spin", "after_post_attn_norm_quant"))
    assert calls == want


def test_layer_filter_skips_unselected_layers(monkeypatch):
    calls = []
    segment = _build_layer(monkeypatch, calls, fuse_attn=True, moe=True, fuse_moe=True)
    probe, _ = _probe(
        {"sites": ["after_attention"], "layers": [2, 4], "schedule": [{"steps": 1}]}, log=calls
    )
    segment.set_criticality_probe(probe)   # layer 3 is not selected
    _run_layer(segment, first=False)
    assert calls == _PRE_PROBE_LAYER_CALLS[(False, True, True, True)]


def test_clearing_the_probe_restores_the_plain_calls(monkeypatch):
    calls = []
    segment = _build_layer(monkeypatch, calls, fuse_attn=False, moe=True, fuse_moe=False)
    probe, _ = _probe({"sites": list(_LAYER_SITES), "schedule": [{"steps": 1}]}, log=calls)
    segment.set_criticality_probe(probe)
    segment.set_criticality_probe(None)
    assert segment.dsa_segment.criticality_probe is None
    assert segment.moe_segment.criticality_probe is None
    _run_layer(segment, first=False)
    assert calls == _PRE_PROBE_LAYER_CALLS[(False, False, True, False)]


def test_probe_counts_marks_across_a_forward(monkeypatch):
    calls = []
    segment = _build_layer(monkeypatch, calls, fuse_attn=False, moe=False, fuse_moe=False)
    probe, _ = _probe({"sites": ["after_attention"], "schedule": [{"steps": 1}]}, log=calls)
    segment.set_criticality_probe(probe)
    probe.begin_forward()
    for _ in range(3):
        _run_layer(segment, first=False)
    probe.end_forward()
    assert probe.layer_marks[SITE_INDEX["after_attention"]] == 3
    assert math.isclose(probe.injected_ms_per_step(0, "after_attention"), 0.0)


# --------------------------------------------------------------------------- #
# Spin kernel registration: AOT in both build paths, checked at startup
# --------------------------------------------------------------------------- #

_SPIN_EXT = "batchgen_kernels.common._C_spin_delay"
_SPIN_SRC = "src/common/spin_delay.cu"


def test_spin_kernel_is_registered_in_both_build_paths_without_fast_math():
    setup = (_REPO_ROOT / "batchgen_kernels/setup.py").read_text()
    registry = (_REPO_ROOT / "batchgen_kernels/_jit_registry.py").read_text()
    assert (_REPO_ROOT / "batchgen_kernels" / _SPIN_SRC).is_file()
    for text in (setup, registry):
        assert _SPIN_EXT in text and _SPIN_SRC in text
        tail = text.split(_SPIN_EXT, 1)[1]
        cut = tail.find("batchgen_kernels.")
        entry = tail if cut < 0 else tail[:cut]
        assert "--use_fast_math" not in entry


def test_kernel_version_was_bumped_for_the_new_extension():
    namespace = {"__file__": str(_REPO_ROOT / "batchgen_kernels/_version.py")}
    exec((_REPO_ROOT / "batchgen_kernels/_version.py").read_text(), namespace)
    assert namespace["version_info"] >= (0, 6, 1)
    assert namespace["__version__"] == ".".join(map(str, namespace["version_info"]))
