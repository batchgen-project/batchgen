"""The default phase-switch lifecycle reuses its instances instead of rebuilding.

Before this, every prefill -> decode boundary rebuilt the decode model, reloaded
its weights, re-initialised its fused kernels and descriptors, re-allocated the
KV pools and recaptured every decode CUDA-graph bucket; the reverse boundary
threw all of it away again. The lifecycle here builds each phase once per
process and switches it in:

* the first decode configure builds the decode instance, sizes the KV pool and
  captures the graphs — strictly from the decode phase, after the prefill ring
  and the prefill model have been released, so the measurement cannot charge
  prefill-only residency against a pool that is then kept for the process;
* decode -> prefill releases only the routed experts, at reserved addresses, and
  keeps the KV region and the captured graph state mapped;
* prefill -> decode maps the experts back and activates, with no model
  construction, no kernel re-init and no capture.

The switch invariants are inverted for this path on purpose, and that inversion
is pinned here: on the reuse lifecycle a decode -> prefill switch that tore down
the graph state or the KV region is a bug, where on the rebuild path leaving
them behind was the bug.

Worker methods are exec'd in isolation, so no CUDA, core engine or model is
needed.
"""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "batchgen" / "batchgen_worker.py"
WORKER_SRC = WORKER.read_text()


def _worker_class():
    tree = ast.parse(WORKER_SRC)
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )


def _method_ast(name):
    return next(
        node
        for node in _worker_class().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _isolated(names, **globals_override):
    """A bare class carrying the named worker methods."""
    bodies = [copy.deepcopy(_method_ast(name)) for name in names]
    module = ast.Module(
        body=[
            ast.ClassDef(
                name="Worker",
                bases=[],
                keywords=[],
                body=bodies,
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(WORKER), "exec"), namespace)
    cls = namespace["Worker"]
    for name in names:
        member = getattr(cls, name)
        target = member.fget if isinstance(member, property) else member
        target.__globals__.update(globals_override)
    return cls


def _call_names(method, wanted):
    hits = []
    for node in ast.walk(method):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in wanted
        ):
            hits.append((node.lineno, node.func.attr))
    return [name for _, name in sorted(hits)]


# ---------------------------------------------------------------------------
# Requesting reuse, and when it is available
# ---------------------------------------------------------------------------


def _lifecycle(**attrs):
    cls = _isolated(
        [
            "_request_phase_instance_reuse",
            "_decode_instance_is_reusable",
            "_decode_graph_state_present",
        ]
    )
    worker = cls()
    for key, value in attrs.items():
        setattr(worker, key, value)
    return worker


def test_reuse_is_requested_from_a_manager_that_supports_it():
    pm = SimpleNamespace(
        reuse_phase_instances=False,
        activate_decoding=lambda: None,
        decode_instance=None,
    )
    worker = _lifecycle(parallel_manager=pm)

    assert worker._request_phase_instance_reuse() is True
    assert pm.reuse_phase_instances is True
    # Requested but not yet built: nothing to switch in.
    assert worker._decode_instance_is_reusable() is False


def test_a_manager_without_the_capability_is_left_alone():
    """Other model families keep the rebuild path untouched."""
    pm = SimpleNamespace()
    worker = _lifecycle(parallel_manager=pm)

    assert worker._request_phase_instance_reuse() is False
    assert not hasattr(pm, "reuse_phase_instances")
    assert worker._decode_instance_is_reusable() is False


def test_a_manager_that_declined_never_becomes_reusable():
    """A streaming rank: reuse asked for, but no instance was kept.

    `decode_instance is None` is how the worker learns to rebuild, so expert
    offloading keeps working exactly as before instead of failing.
    """
    pm = SimpleNamespace(
        reuse_phase_instances=True,
        activate_decoding=lambda: None,
        decode_instance=None,
    )
    worker = _lifecycle(parallel_manager=pm)

    assert worker._decode_instance_is_reusable() is False


def test_a_built_instance_is_reusable():
    pm = SimpleNamespace(
        reuse_phase_instances=True,
        activate_decoding=lambda: None,
        decode_instance=(object(), {}),
    )
    assert _lifecycle(parallel_manager=pm)._decode_instance_is_reusable() is True


# ---------------------------------------------------------------------------
# Decode -> prefill releases only the experts
# ---------------------------------------------------------------------------


def _release_worker(*, experts_resident=True, graph=True):
    cls = _isolated(
        [
            "_release_reused_decode_instance",
            "_decode_graph_state_present",
        ]
    )
    worker = cls()
    calls = []
    pm = SimpleNamespace(
        decode_experts_resident=experts_resident,
        release_decode_routed_experts=lambda: calls.append("release_experts"),
    )
    worker.parallel_manager = pm
    worker._destroy_gpu_paged_kv_cache = lambda **kw: calls.append(
        ("destroy_kv", tuple(sorted(kw.items())))
    )
    worker._cuda_graph_manager = object() if graph else None
    worker._glm5_moe_cuda_graph_manager = None
    worker._glm5_layer_cuda_graph_manager = None
    worker._whole_model_segment = object() if graph else None
    worker._reused_decode_graph_present = False
    return worker, calls


def test_release_drops_sequences_through_the_pools_and_unmaps_experts():
    worker, calls = _release_worker()

    worker._release_reused_decode_instance()

    assert calls == [
        ("destroy_kv", (("keep_buffers", True),)),
        "release_experts",
    ]
    # Recorded so the invariant can tell "torn down" from "never captured".
    assert worker._reused_decode_graph_present is True


def test_release_is_idempotent_when_the_experts_are_already_unmapped():
    worker, calls = _release_worker(experts_resident=False)

    worker._release_reused_decode_instance()

    assert calls == [("destroy_kv", (("keep_buffers", True),))]


def test_release_records_absent_graph_state_as_absent():
    worker, _ = _release_worker(graph=False)

    worker._release_reused_decode_instance()

    assert worker._reused_decode_graph_present is False


# ---------------------------------------------------------------------------
# The switch invariants, inverted for the reuse path
# ---------------------------------------------------------------------------


def _invariant_worker(
    *,
    reusable=True,
    graph_present=True,
    graph_was_present=True,
    kv_initialized=True,
    experts_resident=False,
    ring_released=True,
):
    cls = _isolated(
        [
            "_assert_phase_switch_clean",
            "_assert_kv_region_mapped",
            "_decode_instance_is_reusable",
            "_decode_graph_state_present",
        ]
    )
    worker = cls()
    worker.rank = 2
    worker.parallel_manager = SimpleNamespace(
        reuse_phase_instances=reusable,
        decode_instance=(object(), {}) if reusable else None,
        decode_experts_resident=experts_resident,
    )
    worker._cuda_graph_manager = object() if graph_present else None
    worker._glm5_moe_cuda_graph_manager = None
    worker._glm5_layer_cuda_graph_manager = None
    worker._whole_model_segment = None
    worker._reused_decode_graph_present = graph_was_present
    worker.gpu_paged_kv_cache_manager = (
        SimpleNamespace(is_initialized=True) if kv_initialized else None
    )
    worker._prefill_ring_released_for_switch = ring_released
    return worker


def test_reuse_decode_to_prefill_accepts_surviving_graphs_and_kv():
    _invariant_worker()._assert_phase_switch_clean("decode_to_prefill", reused=True)


def test_reuse_decode_to_prefill_raises_when_graph_state_was_torn_down():
    worker = _invariant_worker(graph_present=False, graph_was_present=True)

    with pytest.raises(RuntimeError, match="tore down the captured decode graph"):
        worker._assert_phase_switch_clean("decode_to_prefill", reused=True)


def test_reuse_decode_to_prefill_tolerates_a_lifetime_with_graphs_off():
    """Nothing to preserve when nothing was captured."""
    worker = _invariant_worker(graph_present=False, graph_was_present=False)
    worker._assert_phase_switch_clean("decode_to_prefill", reused=True)


def test_reuse_decode_to_prefill_raises_when_the_kv_region_went_away():
    worker = _invariant_worker(kv_initialized=False)

    with pytest.raises(RuntimeError, match="left the GPU KV region"):
        worker._assert_phase_switch_clean("decode_to_prefill", reused=True)


def test_reuse_decode_to_prefill_raises_when_the_experts_stayed_mapped():
    worker = _invariant_worker(experts_resident=True)

    with pytest.raises(RuntimeError, match="left the decode routed experts"):
        worker._assert_phase_switch_clean("decode_to_prefill", reused=True)


def test_reuse_prefill_to_decode_requires_mapped_experts_and_kv():
    _invariant_worker(experts_resident=True)._assert_phase_switch_clean(
        "prefill_to_decode", reused=True
    )

    with pytest.raises(RuntimeError, match="routed experts are not"):
        _invariant_worker(experts_resident=False)._assert_phase_switch_clean(
            "prefill_to_decode", reused=True
        )

    with pytest.raises(RuntimeError, match="left the GPU KV region"):
        _invariant_worker(
            experts_resident=True, kv_initialized=False
        )._assert_phase_switch_clean("prefill_to_decode", reused=True)


def test_rebuild_path_keeps_the_original_invariant():
    """With no reusable instance the old checks still apply, unchanged."""
    worker = _invariant_worker(reusable=False, graph_present=True)

    with pytest.raises(RuntimeError, match="_cuda_graph_manager"):
        worker._assert_phase_switch_clean("decode_to_prefill", reused=False)

    clean = _invariant_worker(reusable=False, graph_present=False)
    clean._assert_phase_switch_clean("decode_to_prefill", reused=False)


def test_prefill_to_decode_still_requires_the_ring_release():
    worker = _invariant_worker(ring_released=False, experts_resident=True)

    with pytest.raises(RuntimeError, match="weight ring"):
        worker._assert_phase_switch_clean("prefill_to_decode", reused=True)


# ---------------------------------------------------------------------------
# Sizing point
# ---------------------------------------------------------------------------


def test_kv_sizing_refuses_to_run_outside_the_decode_phase():
    """The pool is sized from what is free and then kept for the process.

    Sizing it while the prefill phase is resident is what an earlier
    startup-time attempt did, and it permanently charged that phase's bytes
    against decode capacity.
    """
    size = _isolated(["_init_gpu_kv_with_actual_size"])
    worker = size()
    worker.rank = 0
    worker.gpu_paged_kv_cache_manager = None
    worker._phase = "prefill"

    with pytest.raises(RuntimeError, match="must run from the decode phase"):
        worker._init_gpu_kv_with_actual_size()


def test_kv_sizing_is_skipped_once_the_manager_exists():
    size = _isolated(["_init_gpu_kv_with_actual_size"])
    worker = size()
    worker.gpu_paged_kv_cache_manager = SimpleNamespace(is_initialized=True)
    worker._phase = "prefill"

    # Returns before the phase guard: a live pool is never re-sized, which is
    # what keeps one size for the whole process.
    worker._init_gpu_kv_with_actual_size()


def test_set_phase_records_the_phase_for_the_sizing_guard():
    src = ast.unparse(_method_ast("set_phase"))
    assert "self._phase = phase" in src


# ---------------------------------------------------------------------------
# Control flow: no rebuild, no capture on switch 2+
# ---------------------------------------------------------------------------


def test_decode_activation_skips_the_whole_build_chain():
    method = _method_ast("_load_decode_model")
    reuse_branch = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and "_decode_instance_is_reusable" in ast.unparse(node.test)
    )
    src = ast.unparse(reuse_branch.body)

    assert "activate_decoding()" in src
    assert "_assert_phase_switch_clean('prefill_to_decode', reused=True)" in src
    assert "_install_decode_weight_copy_queue()" in src
    # Everything the rebuild path does must be absent from this branch.
    for forbidden in (
        "configure_decoding",
        "deep_free_model_memory",
        "init_nvshmem",
        "_warmup_cuda_graphs",
        "_init_gpu_kv_with_actual_size",
    ):
        assert forbidden not in src, forbidden
    # And it returns, so none of the build chain below can run.
    assert any(isinstance(node, ast.Return) for node in reuse_branch.body)


def test_the_first_build_sizes_for_the_largest_admissible_batch():
    """A reused instance's padded MoE buffers are never rebuilt."""
    src = ast.unparse(_method_ast("_load_decode_model"))
    assert "reuse_requested = self._request_phase_instance_reuse()" in src
    assert "self._max_decode_rank_bsz() if reuse_requested else max_num_seq" in src
    assert "padding_bsz=padding_bsz" in src
    assert "self._decode_padding_bsz = padding_bsz" in src


def test_decode_to_prefill_keeps_the_kv_pools_on_the_reuse_path():
    method = _method_ast("_config_prefill_for_batch")
    destroys = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_destroy_gpu_paged_kv_cache"
    ]
    assert len(destroys) == 1
    guard = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and destroys[0] in list(ast.walk(node))
        and "not reuse" in ast.unparse(node.test)
    )
    assert ast.unparse(guard.test) == "not persistent and (not reuse)"

    src = ast.unparse(method)
    assert "self._release_reused_decode_instance()" in src
    assert "reuse = not persistent and self._decode_instance_is_reusable()" in src


def test_prefill_activation_covers_both_reuse_paths():
    src = ast.unparse(_method_ast("_config_prefill_for_batch"))
    assert "(persistent or reuse) and self.parallel_manager.prefill_instance is not None" in src


def test_persistent_flag_path_is_untouched():
    """--persistent-phase-instances keeps its own branch everywhere."""
    decode = ast.unparse(_method_ast("_load_decode_model"))
    assert "if self._persistent_phase_enabled():" in decode
    assert "self._activate_persistent_decode_instance(comm)" in decode
    prefill = ast.unparse(_method_ast("_config_prefill_for_batch"))
    assert "self._release_persistent_decode_instance()" in prefill
    # The default lifecycle never consults the flag's gate for its own decision.
    assert "reuse = not persistent" in prefill


def test_decode_weight_copy_queue_install_is_shared():
    method = _method_ast("_load_decode_model")
    assert _call_names(method, {"_install_decode_weight_copy_queue"}) == [
        "_install_decode_weight_copy_queue",
        "_install_decode_weight_copy_queue",
    ]
    install = ast.unparse(_method_ast("_install_decode_weight_copy_queue"))
    assert "set_weight_copy_queue" in install
    assert "start_h2d_worker" in install
    # The streamed-SP8 conflict check moved with it, not away.
    assert "_streamed_sp8_h2d_installed" in install


def test_the_reuse_path_never_nulls_the_graph_manager():
    """This is the mechanism that forced a recapture on every batch.

    `should_warmup_cuda_graphs_before_decode` returns False as soon as the
    graph manager exists, and `deep_free_model_memory()` sets it to None. On
    the rebuild path that call runs at every decode -> prefill switch, so the
    next decode always re-warms. Neither reuse branch may reach it.
    """
    prefill = _method_ast("_config_prefill_for_batch")
    reuse_branch = None
    for node in ast.walk(prefill):
        if not isinstance(node, ast.If):
            continue
        for clause in [node] + [
            n for n in node.orelse if isinstance(n, ast.If)
        ]:
            if ast.unparse(clause.test).strip() == "reuse":
                reuse_branch = clause
    assert reuse_branch is not None, "no `elif reuse:` branch found"
    src = ast.unparse(reuse_branch.body)
    assert "self._release_reused_decode_instance()" in src
    assert "deep_free_model_memory" not in src

    decode = _method_ast("_load_decode_model")
    decode_reuse = next(
        node
        for node in ast.walk(decode)
        if isinstance(node, ast.If)
        and "_decode_instance_is_reusable" in ast.unparse(node.test)
    )
    assert "deep_free_model_memory" not in ast.unparse(decode_reuse.body)

    # And the rebuild path still frees, so nothing regressed for a rank that
    # cannot reuse its instances.
    assert "self.deep_free_model_memory()" in ast.unparse(decode)
    assert "self.deep_free_model_memory()" in ast.unparse(prefill)


def test_the_building_switch_does_not_require_a_mapped_kv_region():
    """Regression: the first prefill -> decode switch BUILDS the instance.

    The KV pool is created after `_load_decode_model` returns, by
    `_init_gpu_kv_with_actual_size()`. An earlier version of the invariant
    re-derived "am I on the reuse path?" from `_decode_instance_is_reusable()`,
    which flips to True the moment `configure_decoding` caches the instance —
    i.e. during the very switch that builds it, before the pool exists. Every
    rank then died on the first decode with "left the GPU KV region unmapped".
    The mode is now passed explicitly by each call site.
    """
    worker = _invariant_worker(
        reusable=True,             # the instance has just been cached
        kv_initialized=False,      # ...but the pool does not exist yet
        experts_resident=True,
        ring_released=True,
    )
    worker.gpu_paged_kv_cache_manager = None

    # The building switch asserts only the ring release.
    worker._assert_phase_switch_clean("prefill_to_decode", reused=False)

    # The activating switch still demands a mapped region.
    worker._prefill_ring_released_for_switch = True
    with pytest.raises(RuntimeError, match="left the GPU KV region"):
        worker._assert_phase_switch_clean("prefill_to_decode", reused=True)


def test_every_invariant_call_site_passes_the_mode_explicitly():
    """The mode must never be re-derived inside the invariant."""
    body = ast.unparse(_method_ast("_assert_phase_switch_clean"))
    assert "_decode_instance_is_reusable" not in body
    for name in ("_load_decode_model", "_config_prefill_for_batch"):
        for node in ast.walk(_method_ast(name)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_assert_phase_switch_clean"
            ):
                kw = {k.arg for k in node.keywords}
                assert "reused" in kw, f"{name}: call without reused="
