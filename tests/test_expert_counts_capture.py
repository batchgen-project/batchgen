"""CPU gate for the decode tokens-per-expert capture.

Covers the four properties the diagnostic has to hold on real hardware, none of
which needs a GPU:

* flag plumbing — the engine's batch-level hook arms, counts steps, flushes and
  rejects a malformed flag value;
* accumulation arithmetic — the matrix is the exact per-(layer, local expert)
  sum over the decode steps of the armed window;
* zero cost when off — nothing is allocated, the graph epoch stays 0, and the
  expert path's accumulate call is guarded by the capture argument the DECODE
  call sites pass and the prefill chunk loop does not;
* summary formatting — the digest line and the JSON payload are exact functions
  of the matrix.

The engine methods are executed through ``_isolated_method``, which compiles the
two hooks out of the worker's AST: it exercises the real code without importing
the worker module (whose import pulls in the serving stack).
"""

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from batchgen.moe import expert_counts_capture as ecc

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "batchgen" / "batchgen_worker.py"
RESIDENT = ROOT / "batchgen" / "moe" / "fused_moe_mxfp4_resident.py"
GRAPH_SEGMENT = (
    ROOT / "batchgen" / "models" / "moonshotai" / "kimi_linear"
    / "moe_cuda_graph_segments.py"
)
CAPTURE_MODULE = ROOT / "batchgen" / "moe" / "expert_counts_capture.py"


# --------------------------------------------------------------------------- #
#  helpers                                                                     #
# --------------------------------------------------------------------------- #


def _function(path, class_name, function_name):
    tree = ast.parse(path.read_text())
    if class_name is None:
        scope = tree.body
    else:
        scope = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ).body
    return next(
        node
        for node in scope
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )


def _isolated_method(path, class_name, function_name, globals_=None):
    method = copy.deepcopy(_function(path, class_name, function_name))
    module = ast.Module(
        body=[
            ast.ClassDef(
                name="Isolated",
                bases=[],
                keywords=[],
                body=[method],
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    namespace = dict(globals_ or {})
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return getattr(namespace["Isolated"], function_name)


def _worker_hooks():
    """The two engine hooks, bound to a stub ``self``."""
    attn_base = SimpleNamespace(batchgen_debug=None)
    step = _isolated_method(
        WORKER, "BatchGenWorker", "_expert_counts_capture_step",
        {"AttnWrapperBase": attn_base},
    )
    flush = _isolated_method(
        WORKER, "BatchGenWorker", "_expert_counts_capture_flush",
        {"AttnWrapperBase": attn_base},
    )
    worker = SimpleNamespace(
        rank=3,
        _batchgen_debug=None,
        _expert_counts_capture_armed=False,
    )
    worker._expert_counts_capture_step = lambda debug: step(worker, debug)
    worker._expert_counts_capture_flush = lambda reason: flush(worker, reason)
    return worker, attn_base


@pytest.fixture(autouse=True)
def _clean_capture_state():
    ecc._reset_for_tests()
    yield
    ecc._reset_for_tests()


def _register(num_layers=4, num_experts=3, expert_start=28):
    layer_ids = [i + 1 for i in range(num_layers)]
    ecc.register_geometry(layer_ids, num_experts, expert_start, "cpu")
    return layer_ids


def _counts(values):
    return torch.tensor(values, dtype=torch.int32)


# --------------------------------------------------------------------------- #
#  flag plumbing                                                              #
# --------------------------------------------------------------------------- #


def test_engine_hook_arms_counts_steps_and_flushes():
    _register(num_layers=2, num_experts=3)
    worker, _ = _worker_hooks()

    worker._expert_counts_capture_step({"expert_counts_capture": True})
    assert worker._expert_counts_capture_armed is True
    capture = ecc.active()
    assert capture is not None
    assert capture.steps == 1

    # Re-arming on a later step must not discard what is accumulated.
    capture.accumulate(0, _counts([1, 2, 3]))
    worker._expert_counts_capture_step({"expert_counts_capture": True})
    assert ecc.active() is capture
    assert capture.steps == 2
    assert capture.readout()[0] == [1, 2, 3]

    worker._expert_counts_capture_flush("decode_end")
    assert worker._expert_counts_capture_armed is False
    assert ecc.active() is None


def test_engine_hook_is_inert_while_the_flag_is_absent():
    _register()
    worker, _ = _worker_hooks()

    worker._expert_counts_capture_step(None)
    worker._expert_counts_capture_step({})
    worker._expert_counts_capture_step({"some_other_flag": 1})

    assert worker._expert_counts_capture_armed is False
    assert ecc.active() is None
    assert ecc.is_armed() is False
    # Nothing allocated: the accumulator is built by the first arm only.
    assert ecc._CAPTURE is None


def test_engine_hook_flushes_when_a_later_batch_drops_the_flag():
    _register(num_layers=2, num_experts=2)
    worker, _ = _worker_hooks()
    worker._expert_counts_capture_step({"expert_counts_capture": True})
    ecc.active().accumulate(1, _counts([4, 5]))

    worker._expert_counts_capture_step({"expert_counts_capture": False})

    assert worker._expert_counts_capture_armed is False
    assert ecc.active() is None


def test_engine_hook_rejects_a_non_boolean_flag():
    _register()
    worker, _ = _worker_hooks()
    for bad in ("true", 1, [], {"steps": 4}):
        with pytest.raises(ValueError, match="expert_counts_capture must be a boolean"):
            worker._expert_counts_capture_step({"expert_counts_capture": bad})
    assert ecc.active() is None


def test_engine_flush_rejects_a_non_string_artifact_dir():
    _register()
    worker, attn_base = _worker_hooks()
    worker._expert_counts_capture_step({"expert_counts_capture": True})
    attn_base.batchgen_debug = {
        "expert_counts_capture": True,
        "expert_counts_capture_dir": 17,
    }
    with pytest.raises(ValueError, match="expert_counts_capture_dir must be"):
        worker._expert_counts_capture_flush("decode_end")


def test_engine_flush_writes_one_json_artifact_per_rank(tmp_path):
    _register(num_layers=2, num_experts=2, expert_start=56)
    worker, attn_base = _worker_hooks()
    worker._expert_counts_capture_step({"expert_counts_capture": True})
    ecc.active().accumulate(0, _counts([7, 1]))
    ecc.active().accumulate(1, _counts([2, 2]))
    attn_base.batchgen_debug = {
        "expert_counts_capture": True,
        "expert_counts_capture_dir": str(tmp_path),
    }

    worker._expert_counts_capture_flush("decode_end")

    written = tmp_path / "expert_counts_rank3.json"
    assert written.exists()
    out = json.loads(written.read_text())
    assert out["rank"] == 3
    assert out["reason"] == "decode_end"
    assert out["counts"] == [[7, 1], [2, 2]]
    assert out["layer_ids"] == [1, 2]
    assert out["global_expert_start"] == 56


def test_engine_hooks_are_wired_into_the_decode_loop():
    source = WORKER.read_text()
    tree = ast.parse(source)
    calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    # Armed at the decode-config boundary and again per decode step, flushed
    # once at the decode boundary.
    assert calls.count("_expert_counts_capture_step") == 2
    assert calls.count("_expert_counts_capture_flush") == 2  # loop + one call site

    # The arm must precede the CUDA-graph prewarm: the accumulating add is
    # recorded INTO the decode graphs.
    lines = source.split("\n")
    arm_lines = [
        i for i, line in enumerate(lines)
        if "_expert_counts_capture_step(" in line and "def " not in line
    ]
    prewarm_lines = [
        i for i, line in enumerate(lines) if "prewarm_kimi_graphs(" in line
    ]
    assert arm_lines and prewarm_lines
    assert min(arm_lines) < min(prewarm_lines)

    # The flush must run while the batch's debug dict is still bound.
    flush_line = next(
        i for i, line in enumerate(lines)
        if '_expert_counts_capture_flush("decode_end")' in line
    )
    clear_line = next(
        i for i, line in enumerate(lines)
        if line.strip() == "AttnWrapperBase.batchgen_debug = None"
    )
    assert flush_line < clear_line


def test_flag_plumbing_adds_no_environment_guard():
    """Hygiene: a batch-level flag, not an env var, and no stray print.

    Scoped to the code this diagnostic adds -- the whole-file print count of
    the engine module is pre-existing and not this change's to assert on.
    """
    subtrees = [ast.parse(CAPTURE_MODULE.read_text())]
    for name in ("_expert_counts_capture_step", "_expert_counts_capture_flush"):
        subtrees.append(_function(WORKER, "BatchGenWorker", name))
    subtrees.append(
        _function(RESIDENT, "ResidentEPMXFP4MoELayer", "_expert_path")
    )
    subtrees.append(
        _function(GRAPH_SEGMENT, "K3MoEGraphSegment", "_capture_expert_counts")
    )
    for tree in subtrees:
        prints = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ]
        assert not prints
        names = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        assert "environ" not in names
        assert "getenv" not in names
    capture_source = CAPTURE_MODULE.read_text()
    assert "environ" not in capture_source
    assert "getenv" not in capture_source


# --------------------------------------------------------------------------- #
#  accumulation arithmetic                                                     #
# --------------------------------------------------------------------------- #


def test_accumulation_sums_every_layer_over_every_step():
    _register(num_layers=3, num_experts=4)
    capture = ecc.arm()
    assert capture.readout() == [[0] * 4] * 3

    steps = [
        [[1, 0, 2, 0], [0, 3, 0, 0], [4, 4, 4, 4]],
        [[2, 2, 0, 1], [1, 0, 0, 5], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [2, 2, 2, 2], [1, 1, 1, 1]],
    ]
    for step in steps:
        for slot, row in enumerate(step):
            capture.accumulate(slot, _counts(row))
        capture.mark_step()

    assert capture.steps == 3
    assert capture.readout() == [
        [3, 2, 2, 1],
        [3, 5, 2, 7],
        [5, 5, 5, 5],
    ]


def test_accumulator_is_int64_and_survives_int32_overflow_of_the_summands():
    """int32 counts, int64 accumulator: a long window cannot wrap."""
    _register(num_layers=1, num_experts=2)
    capture = ecc.arm()
    big = _counts([2 ** 30, 2 ** 30])
    for _ in range(8):
        capture.accumulate(0, big)
    assert capture.readout() == [[8 * 2 ** 30, 8 * 2 ** 30]]
    assert capture.readout()[0][0] > 2 ** 31


def test_accumulation_targets_a_stable_device_address():
    """The add must land in the persistent buffer, not a fresh tensor.

    This is the property that makes the add recordable in a CUDA graph: replay
    re-executes it against the address captured at capture time.
    """
    _register(num_layers=2, num_experts=2)
    capture = ecc.arm()
    before = capture._accum.data_ptr()
    rows = [row.data_ptr() for row in capture._rows]
    capture.accumulate(0, _counts([1, 1]))
    capture.accumulate(1, _counts([2, 2]))
    assert capture._accum.data_ptr() == before
    assert [row.data_ptr() for row in capture._rows] == rows

    # Re-arming reuses the same accumulator and address, so graphs captured in
    # an earlier armed window stay valid.
    ecc.disarm()
    again = ecc.arm()
    assert again is capture
    assert again._accum.data_ptr() == before
    assert again.readout() == [[0, 0], [0, 0]]
    assert again.steps == 0


def test_accumulation_accepts_a_flat_or_shaped_counts_tensor():
    _register(num_layers=1, num_experts=3)
    capture = ecc.arm()
    capture.accumulate(0, _counts([1, 2, 3]))
    capture.accumulate(0, _counts([[1, 2, 3]]))
    assert capture.readout() == [[2, 4, 6]]


def test_arm_without_registered_geometry_fails_loudly():
    with pytest.raises(ecc.ExpertCountsCaptureError, match="no MoE geometry"):
        ecc.arm()


def test_accumulate_rejects_an_unknown_slot_or_a_wrong_width():
    _register(num_layers=2, num_experts=3)
    capture = ecc.arm()
    with pytest.raises(ecc.ExpertCountsCaptureError, match="slot"):
        capture.accumulate(2, _counts([1, 1, 1]))
    with pytest.raises(ecc.ExpertCountsCaptureError, match="slot"):
        capture.accumulate(-1, _counts([1, 1, 1]))
    with pytest.raises(ecc.ExpertCountsCaptureError, match="expected 3 local"):
        capture.accumulate(0, _counts([1, 1]))


def test_degenerate_geometry_is_refused():
    with pytest.raises(ecc.ExpertCountsCaptureError, match="at least one MoE layer"):
        ecc.register_geometry([], 4, 0, "cpu")
    with pytest.raises(ecc.ExpertCountsCaptureError, match="positive local expert"):
        ecc.register_geometry([0, 1], 0, 0, "cpu")


# --------------------------------------------------------------------------- #
#  zero cost when off                                                          #
# --------------------------------------------------------------------------- #


def test_nothing_is_allocated_until_the_first_arm():
    _register()
    assert ecc.active() is None
    assert ecc.is_armed() is False
    assert ecc.graph_epoch() == 0
    assert ecc._CAPTURE is None


def test_graph_epoch_is_zero_off_and_stable_across_rearms():
    _register(num_layers=2, num_experts=2)
    assert ecc.graph_epoch() == 0

    ecc.arm()
    armed = ecc.graph_epoch()
    assert armed != 0

    ecc.disarm()
    assert ecc.graph_epoch() == 0

    ecc.arm()
    assert ecc.graph_epoch() == armed


def test_graph_epoch_changes_when_the_geometry_changes():
    _register(num_layers=2, num_experts=2)
    ecc.arm()
    first = ecc.graph_epoch()
    ecc.disarm()

    ecc.register_geometry([1, 2, 3], 2, 0, "cpu")
    assert ecc._CAPTURE is None
    ecc.arm()
    assert ecc.graph_epoch() != first


def test_registering_the_same_geometry_twice_keeps_the_accumulator():
    layer_ids = _register(num_layers=2, num_experts=2)
    capture = ecc.arm()
    capture.accumulate(0, _counts([5, 5]))
    ecc.register_geometry(layer_ids, 2, 28, "cpu")
    assert ecc._CAPTURE is capture
    assert ecc.active() is capture
    assert capture.readout()[0] == [5, 5]


def test_expert_path_accumulate_is_guarded_by_the_capture_argument():
    """Off path: the expert path must not touch the capture at all."""
    function = _function(RESIDENT, "ResidentEPMXFP4MoELayer", "_expert_path")
    accumulate_calls = [
        node for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "accumulate"
    ]
    assert len(accumulate_calls) == 1

    guards = [
        ast.unparse(node.test)
        for node in ast.walk(function)
        if isinstance(node, ast.If)
        and any(
            call is accumulate_calls[0]
            for call in ast.walk(ast.Module(body=node.body, type_ignores=[]))
            if isinstance(call, ast.Call)
        )
    ]
    assert guards == [
        "counts_capture is not None and self.counts_slot is not None"
    ]

    # The capture argument exists and defaults to off.
    names = [arg.arg for arg in function.args.args]
    assert names[-1] == "counts_capture"
    assert function.args.defaults[-1].value is None


def test_only_the_decode_call_sites_pass_the_capture():
    """A prefill chunk must never land in the decode matrix."""
    tree = ast.parse(RESIDENT.read_text())
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_expert_path"
    ]
    assert len(calls) == 3
    with_capture = [
        call for call in calls
        if any(kw.arg == "counts_capture" for kw in call.keywords)
    ]
    # The two decode entries (world=1 and the EP steady-state path) capture;
    # the chunked prefill call, which passes dispatch_capacity, does not.
    assert len(with_capture) == 2
    for call in calls:
        has_capacity = any(
            kw.arg == "dispatch_capacity" for kw in call.keywords
        )
        has_capture = any(
            kw.arg == "counts_capture" for kw in call.keywords
        )
        assert has_capacity != has_capture


def test_graph_segment_accumulates_on_both_exchanges():
    """Whichever exchange the planner picks, the counts are the same quantity."""
    tree = ast.parse(GRAPH_SEGMENT.read_text())
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_capture_expert_counts"
    ]
    assert len(calls) == 2
    helper = _function(GRAPH_SEGMENT, "K3MoEGraphSegment", "_capture_expert_counts")
    # Graph-safe: no host read of the counts tensor on the hot path.
    forbidden = {"item", "tolist", "cpu", "numpy", "synchronize"}
    attrs = {
        node.func.attr
        for node in ast.walk(helper)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not (attrs & forbidden)


def test_accumulate_performs_no_host_read():
    helper = _function(CAPTURE_MODULE, "ExpertCountsCapture", "accumulate")
    attrs = {
        node.func.attr
        for node in ast.walk(helper)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not (attrs & {"item", "tolist", "cpu", "numpy", "synchronize", "zeros"})
    assert "add_" in attrs


# --------------------------------------------------------------------------- #
#  summary formatting                                                          #
# --------------------------------------------------------------------------- #


def test_layer_stats_is_exact():
    assert ecc.layer_stats([4, 1, 1]) == (4, 2.0, 1, 6)
    assert ecc.layer_stats([0, 0]) == (0, 0.0, 0, 0)
    assert ecc.layer_stats([7]) == (7, 7.0, 7, 7)
    with pytest.raises(ecc.ExpertCountsCaptureError, match="empty layer row"):
        ecc.layer_stats([])


def test_summary_line_reports_the_imbalance_and_the_total():
    geometry = ecc.ExpertCountsGeometry([5, 9], 4, 28, "cpu")
    matrix = [
        [10, 10, 10, 10],   # perfectly balanced: max/mean = 1.0
        [40, 0, 0, 0],      # fully skewed: max/mean = 4.0
    ]
    line = ecc.summary_line(7, 64, matrix, geometry)

    assert line.startswith("[EXPERT_COUNTS] rank=7 steps=64 ")
    assert "layers=2" in line
    assert "local_experts=4" in line
    assert "global_expert_ids=28..31" in line
    assert "total_rows=80" in line
    # Reported against the MODEL layer ids, not the matrix rows.
    assert "worst=4.000@layer9" in line
    assert "best=1.000@layer5" in line
    assert "mean=2.500" in line
    assert "\n" not in line


def test_summary_line_survives_an_all_zero_layer():
    matrix = [[0, 0], [1, 3]]
    line = ecc.summary_line(0, 1, matrix)
    assert "total_rows=4" in line
    assert "worst=1.500@layer1" in line


def test_summary_line_refuses_an_empty_matrix():
    with pytest.raises(ecc.ExpertCountsCaptureError, match="no layers"):
        ecc.summary_line(0, 1, [])


def test_payload_is_json_serializable_and_carries_the_full_matrix():
    geometry = ecc.ExpertCountsGeometry([1, 2], 3, 84, "cpu")
    matrix = [[6, 3, 3], [1, 1, 1]]
    out = ecc.payload(2, 5, matrix, geometry)

    assert out["rank"] == 2
    assert out["steps"] == 5
    assert out["num_layers"] == 2
    assert out["num_local_experts"] == 3
    assert out["layer_ids"] == [1, 2]
    assert out["global_expert_start"] == 84
    assert out["counts"] == matrix
    assert out["per_layer"][0] == {
        "layer": 1,
        "max": 6,
        "mean": 4.0,
        "min": 3,
        "total": 12,
        "imbalance_max_over_mean": 1.5,
    }
    assert out["per_layer"][1]["imbalance_max_over_mean"] == 1.0
    assert json.loads(json.dumps(out)) == out
