"""Source guard: which worker attributes an engine file still writes.

Completion criterion P1 says every piece of mutable state has one owner, with
a source-level test that fails when a module other than the owner writes it.
Every other state item on that list already has its test; this is the row that
had none -- the worker's own attributes, written from the engine files under
``batchgen/worker/``.

An engine is not the owner of a worker attribute. The intended direction is
that an engine operation receives what it needs as declared inputs and returns
what it produces, so the worker applies its own writes. Until every engine
reaches that shape, this list records the writes that remain, and **it may only
shrink**: an entry appearing here that is not listed is a regression, and the
fix is to return the value and let the worker apply it, not to extend the list.

What counts as a write to the worker, in an engine file: a store or ``del`` of
an attribute whose receiver is a parameter literally named ``worker``, or
``self`` inside an operation that opens by rebinding ``self = worker`` (the
spelling the pre-seam engines use so a moved body reads unchanged). Those two
receivers are what an engine can reach the worker through.

NOT caught, and deliberately so: mutation of an object the worker merely holds
(``worker.pool.add(...)`` mutates the pool, which has its own owner and its own
guard), writes through a differently named alias, and ``setattr`` with a
computed name. None of those exists in the tree today; if one appears, the
check to extend is this file, with the reason.

The list below was measured on ``8ef48633``: 68 attribute names across 9 files.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINES = ROOT / "batchgen" / "worker"

# file -> the attribute names that file still writes on the worker. Only shrinks.
ALLOWED = {
    "batchgen/worker/decode_engine.py": (
        "_cohort_shutdown_observed",
        "_cumulative_boundary_ms",
        "_cumulative_decode_boundaries",
        "_cumulative_decode_iterations",
        "_cumulative_forward_ms",
        "_decode_tokens_unread",
        "_pending_kv_append_tasks",
        "_pending_kv_append_tensors",
        "_step_counters",
    ),
    "batchgen/worker/graph_runtime.py": (
        "_criticality_probe",
        "_criticality_probe_recapture",
        "_decode_graph_ladder_logged",
    ),
    "batchgen/worker/kv_capacity_engine.py": (
        "_aux_layers_with_slots_cache",
        "_deferred_kv_batch",
        "_deferred_kv_entries",
        "_deferred_kv_entries_aux",
        "_deferred_kv_worker_view",
        "_deferred_kv_worker_view_aux",
        "_deployment_length_verified",
        "_gpu_kv_cached_pool_gb",
        "_gpu_kv_pool_created",
        "_gpu_kv_sizing_provisional",
        "_host_kv_page_stats",
        "_logged_migration_skip_tp",
        "_migrated_sequences",
        "_pending_kv_append_tensors",
        "_pending_migrated_query_book",
        "_watermark_check_counter",
        "gpu_kv_cache_size_gb",
        "gpu_paged_kv_cache_manager",
        "model_context_length",
    ),
    "batchgen/worker/prefill.py": (
        "_prefill_wave_ordinal",
    ),
    "batchgen/worker/prefill_engine.py": (
        "_phase_request_uuids",
        "_prefill_admission_live",
        "_prefill_admission_selection",
        "_prefill_admission_state",
        "_prefill_admission_totals",
        "_prefill_pool_base",
        "_prefill_pool_layout_cache",
        "_prefill_pool_page_budget",
        "_prefill_pool_wave",
        "_prefill_prefix_kv_budget_bytes",
        "_prefill_reservation_outcome",
        "_prefix_rank_cost_profiles",
    ),
    "batchgen/worker/rank_sync_engine.py": (
        "_cohort_shutdown_probe",
        "_gloo_migration_group",
        "_nccl_group",
        "_nccl_port",
        "_nvshmem_initialized_this_run",
        "_sync_coordinator",
        "comm",
    ),
    "batchgen/worker/replica_transport_engine.py": (
        "_replica_plan_installed",
        "weight_copy_task",
    ),
    "batchgen/worker/step_observability_engine.py": (
        "_nsys_decode_profile_forward_count",
        "_nsys_decode_profile_started",
        "_nsys_decode_profile_stopped",
        "_phase_end",
    ),
    "batchgen/worker/worker_lifecycle_engine.py": (
        "_core_initialized",
        "_criticality_probe",
        "_decode_watchdog",
        "_phase_mem_switch_index",
        "_startup_phase_reporter",
        "_watchdog",
        "max_decoding_length",
        "max_input_length",
        "model",
        "model_batch_book",
        "rank_ctx",
    ),
}


def _worker_receivers(fn: ast.AST) -> set:
    """The names that denote the worker inside this operation."""
    names = set()
    args = [a.arg for a in fn.args.posonlyargs + fn.args.args]
    if "worker" in args:
        names.add("worker")
    for statement in fn.body[:3]:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == "self"
            and isinstance(statement.value, ast.Name)
            and statement.value.id == "worker"
        ):
            names.add("self")
    return names


def _writes(path: Path) -> dict:
    tree = ast.parse(path.read_text(), filename=str(path))
    found = {}
    for fn in [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]:
        receivers = _worker_receivers(fn)
        if not receivers:
            continue
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.ctx, (ast.Store, ast.Del))
                and isinstance(node.value, ast.Name)
                and node.value.id in receivers
            ):
                found.setdefault(node.attr, []).append((fn.name, node.lineno))
    return found


def _measured() -> dict:
    out = {}
    for path in sorted(ENGINES.rglob("*.py")):
        writes = _writes(path)
        if writes:
            out[str(path.relative_to(ROOT))] = writes
    return out


def test_no_engine_file_writes_a_worker_attribute_that_is_not_listed():
    measured = _measured()
    new = []
    for relpath, writes in sorted(measured.items()):
        allowed = set(ALLOWED.get(relpath, ()))
        for attr in sorted(writes):
            if attr not in allowed:
                sites = ", ".join(
                    f"{name}:{line}" for name, line in writes[attr]
                )
                new.append(f"{relpath} -> {attr} ({sites})")
    assert not new, (
        "an engine file writes a worker attribute that is not on the "
        "only-shrinks list; return the value and let the worker apply it:\n  "
        + "\n  ".join(new)
    )


def test_the_list_names_no_file_that_no_longer_exists():
    missing = [
        relpath for relpath in ALLOWED if not (ROOT / relpath).exists()
    ]
    assert not missing, (
        "the list names files that are gone; delete their entries: "
        + ", ".join(sorted(missing))
    )


def test_the_list_is_not_padded_with_writes_that_are_already_gone():
    """A stale entry hides a regression, so the list must stay exact.

    This is the half that makes the guard a ratchet rather than a ceiling: when
    a conversion removes a write, its entry has to go, otherwise the same write
    could come back unnoticed.
    """
    measured = _measured()
    stale = []
    for relpath, attrs in sorted(ALLOWED.items()):
        found = set(measured.get(relpath, {}))
        for attr in sorted(set(attrs) - found):
            stale.append(f"{relpath} -> {attr}")
    assert not stale, (
        "these listed writes no longer exist; remove them from the list so a "
        "regression cannot hide behind them:\n  " + "\n  ".join(stale)
    )
