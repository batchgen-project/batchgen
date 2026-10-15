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
* decode -> prefill returns the physical memory of BOTH large decode classes —
  the routed-expert stacks and the KV pool — at reserved addresses, and keeps
  the captured graph state;
* prefill -> decode maps the experts and the KV pool back under the same
  addresses and activates, with no model construction, no sizing, no kernel
  re-init and no capture;
* the end of every decode interval leaves a reused instance alone: the
  rebuild path's unregister + deep-free there is what nulled the graph
  manager and the fused attention pointers in the first measured lifetime.

The graph-state invariant is inverted for this path on purpose, and that
inversion is pinned here: on the reuse lifecycle a decode -> prefill switch
that tore the captured graph state down is a bug, where on the rebuild path
leaving it behind was the bug. The memory invariants are not inverted — both
large classes still go back to the prefill phase, which is what keeps a long
prefill inside the budget of a smaller HBM class.

Worker methods are exec'd in isolation, so no CUDA, core engine or model is
needed.
"""

import ast
import contextlib
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

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
    # Seeded, not only patched afterwards: annotations evaluate at def time.
    namespace = dict(globals_override)
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


def _release_worker(*, experts_resident=True, graph=True, fixed_pool=True):
    cls = _isolated(
        [
            "_release_reused_decode_instance",
            "_assert_kv_pool_addresses_are_fixed",
            "_decode_graph_state_present",
        ]
    )
    worker = cls()
    worker.rank = 4
    calls = []
    pm = SimpleNamespace(
        decode_experts_resident=experts_resident,
        release_decode_routed_experts=lambda: calls.append("release_experts"),
    )
    worker.parallel_manager = pm
    worker.gpu_paged_kv_cache_manager = SimpleNamespace(
        uses_fixed_address_pool=fixed_pool
    )
    worker._destroy_gpu_paged_kv_cache = lambda **kw: calls.append(
        ("destroy_kv", tuple(sorted(kw.items())))
    )
    worker._cuda_graph_manager = object() if graph else None
    worker._glm5_moe_cuda_graph_manager = None
    worker._glm5_layer_cuda_graph_manager = None
    worker._whole_model_segment = object() if graph else None
    return worker, calls


def test_release_returns_the_kv_physical_pages_and_unmaps_the_experts():
    """Both large classes go back, at addresses that stay reserved.

    The first revision of this lifecycle kept the KV pools MAPPED across
    prefill (`keep_buffers=True`), which only fits where the device has the
    headroom to spare: a smaller HBM class cannot run a long prefill beside a
    resident KV pool. The release is a real `destroy()`, which on a
    reserved-range pool unmaps the physical pages and keeps every address.
    """
    worker, calls = _release_worker()

    worker._release_reused_decode_instance()

    assert calls == [("destroy_kv", ()), "release_experts"]


def test_release_is_idempotent_when_the_experts_are_already_unmapped():
    worker, calls = _release_worker(experts_resident=False)

    worker._release_reused_decode_instance()

    assert calls == [("destroy_kv", ())]


def test_release_refuses_a_pool_whose_addresses_can_move():
    """The captured graphs bake the pool pointers.

    Releasing an allocator-backed pool would hand those addresses to the next
    allocation, so an unsupported pool fails the switch instead of silently
    taking the old destroy/recreate path.
    """
    worker, calls = _release_worker(fixed_pool=False)

    with pytest.raises(RuntimeError, match="not held at fixed addresses"):
        worker._release_reused_decode_instance()

    assert calls == []


def test_release_tolerates_a_lifetime_that_never_sized_a_pool():
    worker, calls = _release_worker()
    worker.gpu_paged_kv_cache_manager = None

    worker._release_reused_decode_instance()

    assert calls == [("destroy_kv", ()), "release_experts"]


def test_release_leaves_the_captured_graph_state_in_place():
    worker, _ = _release_worker(graph=True)
    manager = worker._cuda_graph_manager
    segment = worker._whole_model_segment

    worker._release_reused_decode_instance()

    assert worker._cuda_graph_manager is manager
    assert worker._whole_model_segment is segment


# ---------------------------------------------------------------------------
# The switch invariants, inverted for the reuse path
# ---------------------------------------------------------------------------


def _attn(*, persistent=True, registered=True, glm5=True):
    wrapper = SimpleNamespace(persistent=persistent)
    if glm5:
        wrapper._fp8_qkv_a_proj = object() if registered else None
        wrapper._fp8_qkv_a_scale = object() if registered else None
    return wrapper


def _decode_model(*, registered=True, layers=4):
    return SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(self_attn=_attn(registered=registered))
                for _ in range(layers)
            ]
        )
    )


def _invariant_worker(
    *,
    reusable=True,
    graph_present=True,
    graph_was_present=True,
    kv="mapped",
    experts_resident=False,
    ring_released=True,
    attn_registered=True,
):
    """``kv`` is the pool's state at the switch: mapped, released or absent.

    "released" is a pool whose physical pages went back to the driver while its
    reserved addresses stayed — what decode -> prefill now leaves behind.
    "absent" is a lifetime that has not sized a pool yet.
    """
    cls = _isolated(
        [
            "_assert_phase_switch_clean",
            "_assert_kv_region_mapped",
            "_assert_kv_region_released",
            "_assert_decode_graph_state_kept",
            "_assert_decode_attention_weights_registered",
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
    worker._decode_graph_captures = 1 if graph_was_present else 0
    worker.model = _decode_model(registered=attn_registered)
    if kv == "absent":
        worker.gpu_paged_kv_cache_manager = None
    else:
        mapped = kv == "mapped"
        worker.gpu_paged_kv_cache_manager = SimpleNamespace(
            # initialize() maps the pool and destroy() releases it, so the two
            # move together on a reserved-range pool.
            is_initialized=mapped,
            pool_is_mapped=mapped,
            uses_fixed_address_pool=True,
        )
    worker._prefill_ring_released_for_switch = ring_released
    return worker


def test_reuse_decode_to_prefill_accepts_surviving_graphs_and_a_released_pool():
    _invariant_worker(kv="released")._assert_phase_switch_clean(
        "decode_to_prefill", reused=True
    )


def test_reuse_decode_to_prefill_raises_when_graph_state_was_torn_down():
    worker = _invariant_worker(
        kv="released", graph_present=False, graph_was_present=True
    )

    with pytest.raises(RuntimeError, match="tore down the captured decode graph"):
        worker._assert_phase_switch_clean("decode_to_prefill", reused=True)


def test_reuse_decode_to_prefill_tolerates_a_lifetime_with_graphs_off():
    """Nothing to preserve when nothing was captured."""
    worker = _invariant_worker(
        kv="released", graph_present=False, graph_was_present=False
    )
    worker._assert_phase_switch_clean("decode_to_prefill", reused=True)


def test_reuse_decode_to_prefill_raises_when_the_kv_pool_stayed_mapped():
    """The inversion POIS ruled on: prefill gets the KV pages back.

    Keeping the whole KV pool mapped through prefill is a bet on one HBM
    class. The pool's addresses stay reserved, so the remap on the way back is
    a map, not an allocation — there is no reason to hold the pages.
    """
    worker = _invariant_worker(kv="mapped")

    with pytest.raises(RuntimeError, match="left the GPU KV pool mapped"):
        worker._assert_phase_switch_clean("decode_to_prefill", reused=True)


def test_reuse_decode_to_prefill_tolerates_a_lifetime_without_a_pool_yet():
    worker = _invariant_worker(kv="absent")

    worker._assert_phase_switch_clean("decode_to_prefill", reused=True)


def test_reuse_decode_to_prefill_raises_when_the_experts_stayed_mapped():
    worker = _invariant_worker(kv="released", experts_resident=True)

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

    # The activation must have remapped the pool before this point.
    with pytest.raises(RuntimeError, match="left the GPU KV region"):
        _invariant_worker(
            experts_resident=True, kv="absent"
        )._assert_phase_switch_clean("prefill_to_decode", reused=True)

    with pytest.raises(RuntimeError, match="left the GPU KV region"):
        _invariant_worker(
            experts_resident=True, kv="released"
        )._assert_phase_switch_clean("prefill_to_decode", reused=True)


def test_prefill_to_decode_rejects_an_initialized_but_unmapped_pool():
    """A pool that reports initialized while its pages are gone is a bug.

    Belt and braces around the reserved-range lifecycle: the mapped check does
    not trust the initialized flag alone, because only the first is what a
    graph replay dereferences.
    """
    worker = _invariant_worker(experts_resident=True)
    worker.gpu_paged_kv_cache_manager = SimpleNamespace(
        is_initialized=True, pool_is_mapped=False, uses_fixed_address_pool=True
    )

    with pytest.raises(RuntimeError, match="physical pages unmapped"):
        worker._assert_phase_switch_clean("prefill_to_decode", reused=True)


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


def test_decode_to_prefill_releases_the_kv_pool_on_the_reuse_path():
    """The reuse path hands the pool's pages back through its own release.

    `_config_prefill_for_batch` still carries exactly one
    `_destroy_gpu_paged_kv_cache` call, guarded for the rebuild path only: the
    reuse path must not reach that one, because the destroy/recreate it
    performs is what moves the pool's addresses. Its own release
    (`_release_reused_decode_instance`) is the reserved-range one.
    """
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

    # And the release itself asks for a full destroy, not a buffer-keeping one.
    release = ast.unparse(_method_ast("_release_reused_decode_instance"))
    assert "self._destroy_gpu_paged_kv_cache()" in release
    assert "keep_buffers" not in release


def test_the_persistent_path_still_keeps_its_pools_mapped():
    """Untouched by this change: the flag's own lifecycle is unchanged."""
    release = ast.unparse(_method_ast("_release_persistent_decode_instance"))
    assert "self._destroy_gpu_paged_kv_cache(keep_buffers=True)" in release


def test_decode_activation_remaps_the_pool_before_anything_reads_it():
    """Order inside the activation: map, then assert, then switch the phase.

    Decode admission reads live capacity through the pool and the captured
    graphs dereference its baked pointers, so the remap cannot wait for
    `_init_gpu_kv_with_actual_size()` (which the caller runs afterwards, and
    which finds a live pool and returns).
    """
    method = _method_ast("_load_decode_model")
    reuse_branch = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and "_decode_instance_is_reusable" in ast.unparse(node.test)
    )
    order = _call_names(
        ast.Module(body=reuse_branch.body, type_ignores=[]),
        {
            "activate_decoding",
            "_remap_reused_kv_pool",
            "_assert_phase_switch_clean",
            "set_phase",
        },
    )
    assert order == [
        "activate_decoding",
        "_remap_reused_kv_pool",
        "_assert_phase_switch_clean",
        "set_phase",
    ]

    # generate() reads capacity only after the model step that remaps.
    generate = _method_ast("generate")
    names = _call_names(
        generate, {"_load_decode_model", "_init_gpu_kv_with_actual_size", "_prepare_decode_batch"}
    )
    assert names.index("_load_decode_model") < names.index(
        "_init_gpu_kv_with_actual_size"
    ) < names.index("_prepare_decode_batch")


def _remap_worker(*, pool="released", manager=True, fixed=True):
    cls = _isolated(
        [
            "_remap_reused_kv_pool",
            "_assert_kv_pool_addresses_are_fixed",
        ],
        time=SimpleNamespace(perf_counter=lambda: 0.0),
        logging=SimpleNamespace(info=lambda *a, **k: None),
    )
    worker = cls()
    worker.rank = 0
    calls = []

    class _Pool:
        """A reserved-range pool: initialize() maps, destroy() releases."""

        uses_fixed_address_pool = fixed

        def __init__(self, mapped):
            self.pool_is_mapped = mapped

        def initialize(self):
            calls.append("initialize")
            self.pool_is_mapped = True

        @staticmethod
        def get_stats():
            return SimpleNamespace(num_total_pages=512)

    worker.gpu_paged_kv_cache_manager = (
        _Pool(pool == "mapped") if manager else None
    )
    return worker, calls


def test_remap_maps_a_released_pool_back():
    worker, calls = _remap_worker(pool="released")

    worker._remap_reused_kv_pool()

    assert calls == ["initialize"]


def test_remap_is_a_no_op_on_an_already_mapped_pool():
    worker, calls = _remap_worker(pool="mapped")

    worker._remap_reused_kv_pool()

    assert calls == []


def test_remap_refuses_a_pool_whose_addresses_can_move():
    worker, calls = _remap_worker(fixed=False)

    with pytest.raises(RuntimeError, match="not held at fixed addresses"):
        worker._remap_reused_kv_pool()

    assert calls == []


def test_remap_fails_loud_without_a_pool_to_remap():
    """A reused instance's own build sized the pool, so one must exist."""
    worker, _ = _remap_worker(manager=False)

    with pytest.raises(RuntimeError, match="no GPU KV manager"):
        worker._remap_reused_kv_pool()


def test_sizing_remaps_a_reserved_pool_instead_of_building_a_second_one():
    """One pool, one page count, for the whole process.

    A rank that rebuilds its decode instance reaches the sizing path with its
    reserved range still held. Creating a second pool there would reserve a
    second virtual range AND apply the cached (smaller) finalized size, so
    `num_total_pages` would differ between that rank and a reusing rank — which
    the cross-rank capacity snapshot rejects.
    """
    method = _method_ast("_init_gpu_kv_with_actual_size")
    order = _call_names(
        method,
        {"_remap_reused_kv_pool", "_initialize_gpu_kv_manager_fixed_size"},
    )
    assert order[0] == "_remap_reused_kv_pool"

    guard = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and "uses_fixed_address_pool" in ast.unparse(node.test)
    )
    assert "_remap_reused_kv_pool" in ast.unparse(guard.body)
    assert any(isinstance(node, ast.Return) for node in guard.body)
    # Still after the fail-loud phase guard: nothing about the pool is touched
    # from the prefill phase.
    phase_guard = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If) and "_phase" in ast.unparse(node.test)
    )
    assert phase_guard.lineno < guard.lineno


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
        kv="absent",               # ...but the pool does not exist yet
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


# ---------------------------------------------------------------------------
# Capture once: the decode residents survive the prefill phase
# ---------------------------------------------------------------------------
#
# The first measured lifetime on the reuse path activated correctly and then
# died in graph capture with "Layer 0: GLM-5 whole-model graph requires fused
# Q-A/KV-A weights". Neither switch was at fault: the end-of-interval cleanup
# in generate() still ran the rebuild path's _unregister_fp8_weights() +
# deep_free_model_memory() on the reused instance. That nulled the graph
# manager (so the next decode re-entered capture) and every attention wrapper's
# fused Q-A/KV-A pointers (which capture checks). The decode->prefill invariant
# missed it because it compared against a graph-presence snapshot taken at
# release time, i.e. after the damage.


def test_reuse_prefill_to_decode_raises_when_graph_state_was_torn_down():
    worker = _invariant_worker(
        experts_resident=True, graph_present=False, graph_was_present=True
    )

    with pytest.raises(RuntimeError, match="tore down the captured decode graph"):
        worker._assert_phase_switch_clean("prefill_to_decode", reused=True)


def test_reuse_prefill_to_decode_raises_when_attention_weights_unregistered():
    worker = _invariant_worker(experts_resident=True, attn_registered=False)

    with pytest.raises(RuntimeError, match="fused Q-A/KV-A attention weights"):
        worker._assert_phase_switch_clean("prefill_to_decode", reused=True)


def test_attention_check_reads_only_resident_fused_wrappers():
    """Streamed wrappers and other wrapper types carry no registered pointers."""
    worker = _invariant_worker(experts_resident=True)
    worker.model.model.layers.append(
        SimpleNamespace(self_attn=_attn(persistent=False, registered=False))
    )
    worker.model.model.layers.append(
        SimpleNamespace(self_attn=_attn(glm5=False))
    )

    worker._assert_phase_switch_clean("prefill_to_decode", reused=True)


def test_the_building_switch_does_not_check_attention_registration():
    """reused=False: the build itself registers them."""
    worker = _invariant_worker(
        experts_resident=True, attn_registered=False, graph_was_present=False
    )
    worker._assert_phase_switch_clean("prefill_to_decode", reused=False)


def _interval_worker(*, persistent=False, reusable=True):
    cls = _isolated(["_finish_decode_interval", "_decode_instance_is_reusable"])
    worker = cls()
    calls = []
    worker.parallel_manager = SimpleNamespace(
        reuse_phase_instances=reusable,
        decode_instance=(object(), {}) if reusable else None,
    )
    worker._persistent_phase_enabled = lambda: persistent
    worker._unregister_fp8_weights = lambda: calls.append("unregister")
    worker.deep_free_model_memory = lambda: calls.append("deep_free")
    return worker, calls


def test_interval_cleanup_leaves_a_reused_instance_alone():
    worker, calls = _interval_worker(reusable=True)
    worker._finish_decode_interval()
    assert calls == []


def test_interval_cleanup_leaves_a_persistent_instance_alone():
    worker, calls = _interval_worker(persistent=True, reusable=False)
    worker._finish_decode_interval()
    assert calls == []


def test_interval_cleanup_still_frees_on_the_rebuild_path():
    worker, calls = _interval_worker(reusable=False)
    worker._finish_decode_interval()
    assert calls == ["unregister", "deep_free"]


def test_generate_routes_decode_cleanup_through_the_gated_helper():
    method = _method_ast("generate")
    names = _call_names(
        method,
        {"_finish_decode_interval", "deep_free_model_memory", "_unregister_fp8_weights"},
    )
    assert "_finish_decode_interval" in names
    assert "deep_free_model_memory" not in names
    # The remaining direct unregister is the prefill model's, after prefill.
    assert names.count("_unregister_fp8_weights") == 1
    assert names.index("_unregister_fp8_weights") < names.index("_finish_decode_interval")


def _active_worker(*, reusable=True, resident=True, active=True):
    cls = _isolated(["_decode_instance_is_active", "_decode_instance_is_reusable"])
    worker = cls()
    worker.rank = 1
    model = object()
    worker.parallel_manager = SimpleNamespace(
        reuse_phase_instances=reusable,
        decode_instance=(model, {}) if reusable else None,
        decode_experts_resident=resident,
    )
    worker.model = model if active else object()
    return worker


def test_decode_to_decode_interval_is_recognised():
    assert _active_worker()._decode_instance_is_active() is True
    # After decode -> prefill the experts are unmapped: activation is due.
    assert _active_worker(resident=False)._decode_instance_is_active() is False
    # No reusable instance: the rebuild path decides.
    assert _active_worker(reusable=False)._decode_instance_is_active() is False


def test_mapped_experts_on_an_inactive_instance_fail_loud():
    with pytest.raises(RuntimeError, match="not the active model"):
        _active_worker(active=False)._decode_instance_is_active()


def test_decode_to_decode_interval_returns_before_any_switch_work():
    method = _method_ast("_load_decode_model")
    early = next(
        node
        for node in method.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "self._decode_instance_is_active()"
    )
    assert [type(n) for n in early.body] == [ast.Return]
    first_switch_work = min(
        node.lineno
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"stop_h2d_worker", "activate_decoding", "configure_decoding"}
    )
    assert early.lineno < first_switch_work


def test_capture_guard_precedes_any_capture_state():
    method = _method_ast("_setup_cuda_graphs")
    guard = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and "_decode_graph_captures" in ast.unparse(node.test)
        and "_decode_instance_is_reusable" in ast.unparse(node.test)
    )
    assert any(isinstance(n, ast.Raise) for n in guard.body)
    later = [
        node.lineno
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "Glm5WholeModelSegment")
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr
                in {"_release_glm5_whole_model_graph_state", "ensure_cuda_graph_page_table"}
            )
        )
    ]
    assert later and guard.lineno < min(later)


def test_capture_episodes_are_counted_after_the_signature_is_recorded():
    src = ast.unparse(_method_ast("_setup_cuda_graphs"))
    sig = src.index(
        "self._glm5_whole_model_graph_signature = "
        "self._glm5_whole_model_graph_capture_signature()"
    )
    count = src.index("self._decode_graph_captures = ")
    assert sig < count


def test_required_seqlen_is_log_only():
    """The per-batch span never reaches the capture signature or a decision.

    Verified for Phase C: `required_seqlen` is read in exactly one place, the
    rank-0 log line. The graph span is `graph_max_seqlen`, derived from KV
    capacity, and the signature is the page-table storage identity.
    """
    method = _method_ast("_setup_cuda_graphs")
    loads = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Name)
        and node.id == "required_seqlen"
        and isinstance(node.ctx, ast.Load)
    ]
    assert len(loads) == 1
    log_calls = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "info"
    ]
    assert any(loads[0] in list(ast.walk(call)) for call in log_calls)
    sig = ast.unparse(_method_ast("_glm5_whole_model_graph_capture_signature"))
    assert "required_seqlen" not in sig and "max_seqlen" not in sig


class _FakeStorage:
    def __init__(self, ptr, shape):
        self._ptr = ptr
        self.shape = shape
        self.dtype = "torch.int32"
        self.device = "cuda:0"

    def data_ptr(self):
        return self._ptr


class _FakeKv:
    """Fixed-address graph page table: storage never moves (PR #458)."""

    def __init__(self, ptr):
        self._storage = _FakeStorage(ptr, (128, 4096))
        self.resets = 0

    def get_cuda_graph_page_table_storage(self):
        return self._storage

    def reset_allocations(self):
        self.resets += 1


class _FakeBucketing:
    bucket_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

    def get_padded_size(self, n):
        return next(b for b in self.bucket_sizes if b >= n)


class _FakeManager:
    bucketing = _FakeBucketing()

    def has_bucket_for_all_segments(self, n):
        return True


def _load_policy():
    spec = importlib.util.spec_from_file_location(
        "_cuda_graph_policy",
        ROOT / "batchgen" / "models" / "glm" / "glm5" / "cuda_graph_policy.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_batches_with_different_spans_never_invalidate_the_capture():
    """Two batches, different spans and sizes, one captured signature.

    Per-batch variation rides the dynamic graph inputs and the max-sized
    tables; the capture is keyed to KV capacity and page-table identity only.
    Between the batches the reuse lifecycle drops every sequence through the
    pools (decode -> prefill) and switches the instance back in.
    """
    attn = SimpleNamespace(max_seqlen=0)
    cls = _isolated(
        [
            "_glm5_whole_model_graph_current_bucket_missing",
            "_glm5_whole_graph_path_state",
            "_glm5_whole_model_graph_capture_signature",
        ],
        AttnWrapperBase=attn,
        Optional=Optional,
        logging=SimpleNamespace(info=lambda *a, **k: None),
    )
    worker = cls()
    worker.rank = 0
    worker.model_name = "glm-5.2-fp8"
    primary, aux = _FakeKv(0x7F0000000000), _FakeKv(0x7F1000000000)
    gpu = SimpleNamespace(primary=primary, auxiliary=aux)
    worker.core_engine = SimpleNamespace(gpu_paged_kv_manager_aux=aux)
    worker._get_cuda_graph_gpu_manager = lambda: gpu
    worker._glm5_whole_model_graph_requested_for_current_batch = lambda: True
    worker._glm5_whole_model_graph_unavailable_reason = None
    worker._glm5_whole_model_graph_failed_buckets = set()
    worker._glm5_whole_model_all_short = False
    worker._glm5_whole_model_index_topk = 2048
    # One capture, at the building switch, spanning the KV capacity.
    worker._cuda_graph_manager = _FakeManager()
    worker._glm5_whole_model_graph = True
    worker._whole_model_segment = SimpleNamespace(max_seqlen=200_000)
    worker._glm5_whole_model_graph_capture_attempted_for_batch = True
    worker._glm5_whole_model_graph_signature = (
        worker._glm5_whole_model_graph_capture_signature()
    )
    policy = _load_policy()

    invalidations = 0
    for max_bsz, span in ((37, 3_648), (128, 131_072), (5, 913)):
        primary.reset_allocations()  # decode -> prefill on the reuse path
        aux.reset_allocations()
        worker._current_decode_max_rank_batch_size = max_bsz
        attn.max_seqlen = span
        warm = policy.should_warmup_cuda_graphs_before_decode(
            graph_manager_is_initialized=worker._cuda_graph_manager is not None,
            global_batch_has_queueing=False,
            model_name=worker.model_name,
        ) or worker._glm5_whole_model_graph_current_bucket_missing()
        path, _, reason = worker._glm5_whole_graph_path_state(max_bsz)
        invalidations += int(warm)
        assert (path, reason) == ("graph", "captured"), (max_bsz, span, reason)

    assert invalidations == 0


# ---------------------------------------------------------------------------
# #424: the capacity snapshot holds the pool mapped
# ---------------------------------------------------------------------------


def _capacity_worker(world_size=1):
    class _Dist:
        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def all_gather(out, local):
            for item in out:
                item.copy_(local)

    cls = _isolated(
        ["_snapshot_decode_capacity"],
        torch=__import__("torch"),
        dist=_Dist,
        DecodeCapacitySnapshot=tuple,
        reduce_decode_capacity_snapshot=lambda totals, free, **kw: (
            totals,
            free,
            kw,
        ),
    )
    worker = cls()
    worker.rank = 0
    worker.world_size = world_size
    worker.torch_device = "cpu"
    worker._decode_attn_tp_size = lambda: 1
    return worker


class _SnapshotPool:
    """Records whether the stats read happened inside the guarded section."""

    is_initialized = True

    def __init__(self):
        self.depth = 0
        self.read_depths = []

    @contextlib.contextmanager
    def capacity_snapshot(self):
        self.depth += 1
        try:
            yield self
        finally:
            self.depth -= 1

    def get_stats(self):
        self.read_depths.append(self.depth)
        return SimpleNamespace(num_total_pages=512, num_free_pages=480)


def test_the_decode_capacity_snapshot_holds_the_pool_mapped():
    """The whole read, local stats and all_gather, runs inside the section.

    PR #424 made decode admission read live all-rank capacity. With the pool
    released for prefill at every switch, a release racing this read would
    produce a plan for pages that no longer exist; inside the section the
    release fails loud instead.
    """
    worker = _capacity_worker(world_size=2)
    pool = _SnapshotPool()
    worker.gpu_paged_kv_cache_manager = pool

    worker._snapshot_decode_capacity()

    assert pool.read_depths == [1]
    assert pool.depth == 0

    method = _method_ast("_snapshot_decode_capacity")
    guarded = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.With)
        and "capacity_snapshot" in ast.unparse(node.items[0].context_expr)
    )
    inner = _call_names(
        ast.Module(body=guarded.body, type_ignores=[]),
        {"get_stats", "all_gather"},
    )
    assert inner == ["get_stats", "all_gather"]


def test_the_capacity_snapshot_section_is_left_on_failure():
    worker = _capacity_worker()
    pool = _SnapshotPool()

    def boom():
        raise RuntimeError("allocator stats failed")

    pool.get_stats = boom
    worker.gpu_paged_kv_cache_manager = pool

    with pytest.raises(RuntimeError, match="allocator stats failed"):
        worker._snapshot_decode_capacity()

    assert pool.depth == 0


# ---------------------------------------------------------------------------
# First-batch reuse: the FIRST prefill instance is the one that is kept
# ---------------------------------------------------------------------------
#
# On the first lifetime of this lifecycle the first decode -> prefill switch
# cost an order of magnitude more than every later one, because the startup
# prefill configure ran BEFORE the first decode build requested reuse.
# The strategy manager therefore did not keep that prefill model, and the next
# prefill built it again — inside the first measured batch.


def test_reuse_is_requested_before_the_first_prefill_configure():
    method = _method_ast("_config_prefill_for_batch")
    order = _call_names(
        method,
        {
            "_persistent_phase_enabled",
            "_request_phase_instance_reuse",
            "_decode_instance_is_reusable",
            "configure_prefill",
            "activate_prefill",
        },
    )
    request = order.index("_request_phase_instance_reuse")
    assert request < order.index("configure_prefill")
    assert request < order.index("activate_prefill")
    # The persistent flag owns its own instances; it must not be asked.
    assert order.index("_persistent_phase_enabled") < request

    guard = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and "_request_phase_instance_reuse" in ast.unparse(node.body)
    )
    assert ast.unparse(guard.test) == "not persistent"


def test_requesting_reuse_early_does_not_change_the_switch_decision():
    """`reuse` still means "a decode instance is waiting", not "requested"."""
    pm = SimpleNamespace(
        reuse_phase_instances=False,
        activate_decoding=lambda: None,
        decode_instance=None,
    )
    worker = _lifecycle(parallel_manager=pm)

    assert worker._request_phase_instance_reuse() is True
    assert worker._decode_instance_is_reusable() is False


# ---------------------------------------------------------------------------
# The pools are built as reserved-range pools in the first place
# ---------------------------------------------------------------------------


def test_every_phase_switch_pool_is_built_at_fixed_addresses():
    """Both DSA pools and the single-pool path opt in.

    Without this the mechanism is dormant: `fixed_address_pool` defaults to
    False, so a `destroy()` would free the pools and the next `initialize()`
    would cudaMalloc them somewhere else, under the pointers capture baked.
    """
    method = _method_ast("_initialize_gpu_kv_manager_fixed_size")
    constructions = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "GPUPagedKVCacheManager"
    ]
    assert len(constructions) == 3  # DSA primary + aux, and the single pool
    src = ast.unparse(method)
    for call in constructions:
        passed = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
        assert "fixed_address_pool" in passed, ast.unparse(call)
        # Either the predicate directly or the local it was bound to.
        value = passed["fixed_address_pool"]
        assert (
            "_kv_fixed_address_pool_available" in value
            or f"{value} = self._kv_fixed_address_pool_available()" in src
        ), value


def test_the_fixed_address_pool_needs_an_indexed_cuda_device():
    cls = _isolated(
        ["_kv_fixed_address_pool_available"],
        torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
    )
    worker = cls()
    worker.local_rank = 0
    assert worker._kv_fixed_address_pool_available() is True
    worker.local_rank = None
    assert worker._kv_fixed_address_pool_available() is False

    cpu = _isolated(
        ["_kv_fixed_address_pool_available"],
        torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )()
    cpu.local_rank = 0
    assert cpu._kv_fixed_address_pool_available() is False
