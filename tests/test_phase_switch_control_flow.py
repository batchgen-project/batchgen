"""Worker-side phase-switch control flow for the default (rebuild) path.

The default (rebuild) GLM-5.2 phase switch dominates each batch's wall time.
Four of its causes live in ``batchgen_worker.py`` and are pinned here:

* the prefill routed-expert weight ring stayed mapped through the whole decode
  configure and was released only afterwards, so the decode expert upload and
  grouped-GEMM init ran at the tightest free-HBM point of the batch;
* ``deep_free_model_memory`` was a measured no-op on decode -> prefill
  (allocated bytes unchanged) because the parallel strategy manager kept its
  own reference to the decode model;
* the decode graph lifetime was chosen per batch (``all_short``), so a batch
  that outgrew ``index_topk`` forced a recapture or an eager fallback;
* nothing failed loud when a switch left the other phase's GPU state behind.

The tests exec single worker methods in isolation (the idiom in
``test_glm5_persistent_phase.py``) so no CUDA, no core engine and no model are
needed.
"""

import ast
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "batchgen" / "batchgen_worker.py"
WORKER_SRC = WORKER.read_text()


def _isolated_worker_method(name, **globals_override):
    tree = ast.parse(WORKER_SRC)
    worker = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )
    method = copy.deepcopy(
        next(
            node
            for node in worker.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
    )
    module = ast.Module(
        body=[
            ast.ClassDef(
                name="Worker",
                bases=[],
                keywords=[],
                body=[method],
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(WORKER), "exec"), namespace)
    func = getattr(namespace["Worker"], name)
    func.__globals__.update(globals_override)
    return func


# ---------------------------------------------------------------------------
# Ring release before the decode configure
# ---------------------------------------------------------------------------


def _worker_method_ast(name):
    tree = ast.parse(WORKER_SRC)
    worker = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )
    return next(
        node
        for node in worker.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _call_order(method, names):
    """Attribute-call names of interest, in source order."""
    hits = []
    for node in ast.walk(method):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in names
        ):
            hits.append((node.lineno, node.func.attr))
    return [name for _, name in sorted(hits)]


def test_decode_configure_releases_the_ring_before_building_the_model():
    """The prefill ring is released BEFORE the decode model is built.

    The ring used to stay mapped through the whole decode configure (released
    only by the trailing reset_decoding_buffer), so the fix is purely an
    ordering one and the regression guard is that order.
    """
    method = _worker_method_ast("_load_decode_model")
    names = _call_order(
        method,
        {
            "stop_h2d_worker",
            "clear_kv_copy_queue",
            "clear_weight_copy_queue",
            "reset_decoding_buffer",
            "configure_decoding",
            "_assert_phase_switch_clean",
            "_install_decode_weight_copy_queue",
        },
    )

    configure = names.index("configure_decoding")
    for released in (
        "stop_h2d_worker",
        "clear_kv_copy_queue",
        "clear_weight_copy_queue",
        "reset_decoding_buffer",
    ):
        assert names.index(released) < configure, released
    # The method has two exits (activate a reused instance, or build one), so
    # only the release-before-build ordering is asserted here; the activation
    # path is covered in tests/test_phase_instance_reuse.py. Both exits install
    # the decode weight-copy queue and assert their invariant.
    assert names.count("_install_decode_weight_copy_queue") == 2
    assert names.count("_assert_phase_switch_clean") == 2


def test_ring_release_still_preserves_an_installed_streamed_sp8_pipeline():
    """Reordering must not widen what the reset touches.

    Host-RDMA keeps its daemon cursor and its prefill-shaped slots, so the
    decode-layout reset stays behind the streamed-SP8 guard. (The K3 contract
    test ``test_resident_decode_preserves_streamed_sp8_weight_pipeline`` pins
    the same property; this one states it from the phase-switch side.)
    """
    method = _worker_method_ast("_load_decode_model")
    guarded = {"clear_weight_copy_queue", "reset_decoding_buffer"}
    seen = set()

    for node in ast.walk(method):
        if not isinstance(node, ast.If):
            continue
        if "_streamed_sp8_h2d_installed" not in ast.unparse(node.test):
            continue
        for call in _call_order(
            ast.Module(body=node.body, type_ignores=[]), guarded
        ):
            seen.add(call)

    assert seen == guarded

    # And they appear nowhere else in the method.
    all_calls = _call_order(method, guarded)
    assert sorted(all_calls) == sorted(guarded)


def test_ring_release_marks_the_switch_for_the_invariant():
    method = _worker_method_ast("_load_decode_model")
    assigned = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Attribute)
            and t.attr == "_prefill_ring_released_for_switch"
            for t in node.targets
        )
    ]
    assert len(assigned) == 1
    assert ast.unparse(assigned[0].value) == "True"


# ---------------------------------------------------------------------------
# Fail-loud switch invariants
# ---------------------------------------------------------------------------


def _rebuild_path_worker(**attrs):
    """A worker on the rebuild path: no reusable decode instance.

    The lifecycle predicate is stubbed False so these cases exercise the
    original invariant; the reuse inversion is covered in
    tests/test_phase_instance_reuse.py.
    """
    attrs.setdefault("_decode_instance_is_reusable", lambda: False)
    return SimpleNamespace(**attrs)


def test_prefill_to_decode_invariant_raises_when_the_ring_was_not_released():
    check = _isolated_worker_method("_assert_phase_switch_clean")
    worker = _rebuild_path_worker(rank=3, _prefill_ring_released_for_switch=False)

    with pytest.raises(RuntimeError, match="weight ring"):
        check(worker, "prefill_to_decode")


def test_prefill_to_decode_invariant_consumes_the_flag():
    check = _isolated_worker_method("_assert_phase_switch_clean")
    worker = _rebuild_path_worker(rank=0, _prefill_ring_released_for_switch=True)

    check(worker, "prefill_to_decode")

    # One release per switch: the next switch must release it again.
    assert worker._prefill_ring_released_for_switch is False
    with pytest.raises(RuntimeError, match="weight ring"):
        check(worker, "prefill_to_decode")


def test_decode_to_prefill_invariant_raises_on_a_planted_leftover():
    check = _isolated_worker_method("_assert_phase_switch_clean")
    clean = dict(
        rank=5,
        _cuda_graph_manager=None,
        _glm5_moe_cuda_graph_manager=None,
        _glm5_layer_cuda_graph_manager=None,
        _whole_model_segment=None,
    )

    check(_rebuild_path_worker(**clean), "decode_to_prefill")

    for name in (
        "_cuda_graph_manager",
        "_glm5_moe_cuda_graph_manager",
        "_glm5_layer_cuda_graph_manager",
        "_whole_model_segment",
    ):
        planted = dict(clean)
        planted[name] = object()
        with pytest.raises(RuntimeError, match=name):
            check(_rebuild_path_worker(**planted), "decode_to_prefill")


def test_unknown_switch_direction_is_a_programming_error():
    check = _isolated_worker_method("_assert_phase_switch_clean")
    with pytest.raises(ValueError, match="unknown phase-switch direction"):
        check(_rebuild_path_worker(rank=0), "sideways")


# ---------------------------------------------------------------------------
# deep_free_model_memory actually frees
# ---------------------------------------------------------------------------


def test_deep_free_drops_the_parallel_manager_model_reference():
    deep_free = _isolated_worker_method(
        "deep_free_model_memory",
        time=SimpleNamespace(perf_counter=lambda: 0.0),
        logging=SimpleNamespace(info=lambda *a, **k: None),
        gc=SimpleNamespace(collect=lambda: None),
        sys=SimpleNamespace(modules={}),
        torch=SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: False,
                synchronize=lambda *a: None,
                empty_cache=lambda: None,
            )
        ),
    )
    model = object()
    manager = SimpleNamespace(model=model, persistent_phase_instances=False)
    worker = SimpleNamespace(
        rank=0,
        model=model,
        torch_device="cpu",
        parallel_manager=manager,
        _cuda_graph_adapter=None,
        _glm5_layer_graph_failed_buckets=set(),
    )

    deep_free(worker)

    assert worker.model is None
    # The reference that used to keep the whole decode model alive past this
    # call, making the call a no-op.
    assert manager.model is None


def test_deep_free_leaves_persistent_instances_alone():
    deep_free = _isolated_worker_method(
        "deep_free_model_memory",
        time=SimpleNamespace(perf_counter=lambda: 0.0),
        logging=SimpleNamespace(info=lambda *a, **k: None),
        gc=SimpleNamespace(collect=lambda: None),
        sys=SimpleNamespace(modules={}),
        torch=SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: False,
                synchronize=lambda *a: None,
                empty_cache=lambda: None,
            )
        ),
    )
    model = object()
    manager = SimpleNamespace(model=model, persistent_phase_instances=True)
    worker = SimpleNamespace(
        rank=0,
        model=model,
        torch_device="cpu",
        parallel_manager=manager,
        _cuda_graph_adapter=None,
        _glm5_layer_graph_failed_buckets=set(),
    )

    deep_free(worker)

    assert manager.model is model


# ---------------------------------------------------------------------------
# One general decode graph lifetime
# ---------------------------------------------------------------------------


def test_decode_graph_lifetime_is_pinned_general():
    """The per-batch all_short selection is gone from the decision point."""
    assert "all_short = False" in WORKER_SRC
    assert (
        "all_short = bool(index_topk_cfg) and required_seqlen <= index_topk_cfg"
        not in WORKER_SRC
    )


def test_all_short_capture_code_is_not_deleted():
    """POIS: the all-short path stays in the tree, it is just never selected."""
    selector = (
        ROOT / "batchgen" / "attention" / "dsa" / "glm5_decode_selector.py"
    ).read_text()
    assert "_run_all_short_fa3" in selector
    assert "_glm5_whole_model_all_short" in WORKER_SRC
