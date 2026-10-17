"""Every phase-switch stage reports who still holds device memory.

The rebuild path tears a phase instance down and stages the next one, and
"each cycle gives back what it took" had no owner and no observable: a slice
retained per switch surfaced only as an OOM several cycles later, at whichever
allocation is largest (the KV pool remap's ``cuMemCreate``). Torch's counters
cannot see most of it — the KV pool is CUDA-VMM, captured graphs own private
pools, weight buffers are C++ — so the ledger reports the DRIVER's used/total
beside them and names the holders that survive a model.

Structural test: the stages are wired, the line carries the fields a per-cycle
subtraction needs, and the probe stays cheap (rank 0, no synchronize, no
``.item()``).
"""

import ast
import functools
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest


WORKER = Path(__file__).resolve().parents[2] / "batchgen" / "batchgen_worker.py"


@functools.lru_cache(maxsize=1)
def _worker_class():
    tree = ast.parse(WORKER.read_text())
    return next(node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker")


@functools.lru_cache(maxsize=8)
def _method(name):
    return next(node for node in _worker_class().body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def _ledger_labels(method_name):
    body = _method(method_name)
    return [
        node.args[0].value
        for node in ast.walk(body)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_log_phase_memory_ledger"
        and node.args and isinstance(node.args[0], ast.Constant)
    ]


def test_the_ledger_method_exists_and_is_rank_zero_only():
    body = ast.unparse(_method("_log_phase_memory_ledger"))
    assert "if self.rank != 0 or not torch.cuda.is_available():" in body
    assert "return" in body


@pytest.mark.parametrize("field", [
    "driver_used", "driver_total", "torch_alloc", "torch_reserved",
    "resident_ep", "sp8_buffer", "decode_graph", "kv_pool_mapped",
    "kda_slots", "kda_live", "switch=", "stage=",
])
def test_the_line_carries_every_field_a_subtraction_needs(field):
    assert field in ast.unparse(_method("_log_phase_memory_ledger"))


def _code_without_docstring(name):
    node = _method(name)
    statements = [
        item for item in node.body
        if not (isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant)
                and isinstance(item.value.value, str))
    ]
    return "\n".join(ast.unparse(item) for item in statements)


def test_the_probe_stays_cheap():
    body = _code_without_docstring("_log_phase_memory_ledger")
    # A synchronize or a device-to-host read per switch stage would make the
    # observable change what it measures.
    assert "synchronize" not in body
    assert ".item()" not in body
    assert "mem_get_info" in body


def test_deep_free_brackets_itself_and_counts_the_switch():
    labels = _ledger_labels("deep_free_model_memory")
    assert labels == ["deep_free_enter", "deep_free_done"]
    body = ast.unparse(_method("deep_free_model_memory"))
    # The counter advances once per switch, so a cycle is one subtraction.
    assert "self._phase_mem_switch_index = getattr(self, '_phase_mem_switch_index', 0) + 1" in body
    enter = body.index("'deep_free_enter'")
    delete = body.index("del self.model")
    cache = body.rindex("torch.cuda.empty_cache()")
    done = body.index("'deep_free_done'")
    # The exit reading is taken after the allocator has been asked to release,
    # so what it still shows is genuinely retained.
    assert enter < delete < cache < done


def test_both_configure_steps_report_after_staging():
    assert "after_configure_decoding" in _ledger_labels("_load_decode_model")
    prefill = [
        label
        for name in ("_load_prefill_model", "_config_prefill_for_batch",
                     "generate", "_load_model_for_prefill")
        if any(node.name == name for node in _worker_class().body
               if isinstance(node, ast.FunctionDef))
        for label in _ledger_labels(name)
    ]
    assert "after_configure_prefill" in prefill


def test_the_kda_pools_are_reported_so_the_reuse_stays_observable():
    # The pools are deliberately kept across a rebuild, so their slot count
    # must stay flat; a growing one would mean the reuse decision stopped
    # matching and every cycle allocated a new pool.
    body = ast.unparse(_method("_log_phase_memory_ledger"))
    assert "num_state_items" in body
    assert "sequence_state_items" in body


# ---------------------------------------------------------------------------
# Behavioral: the diagnostic must be unable to fail the switch it observes.
# ---------------------------------------------------------------------------

class _Recorder:
    """Stands in for ``logging``; keeps the formatted ledger lines."""

    def __init__(self):
        self.lines = []

    def info(self, fmt, *args):
        self.lines.append(fmt % args)


def _fake_torch():
    """Only the four driver/allocator queries the ledger makes."""
    gib = 1024 ** 3
    return SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=lambda device: (10 * gib, 80 * gib),
        memory_allocated=lambda device: 20 * gib,
        memory_reserved=lambda device: 30 * gib,
    ))


@functools.lru_cache(maxsize=1)
def _ledger_callable():
    module = ast.fix_missing_locations(
        ast.Module(body=[_method("_log_phase_memory_ledger")], type_ignores=[])
    )
    namespace = {"torch": _fake_torch(), "logging": _Recorder()}
    exec(compile(module, str(WORKER), "exec"), namespace)
    return namespace["_log_phase_memory_ledger"], namespace


def _run_ledger(label="after_configure_decoding"):
    function, namespace = _ledger_callable()
    recorder = _Recorder()
    namespace["logging"] = recorder
    worker = SimpleNamespace(
        rank=0,
        local_rank=0,
        model=None,
        _phase_mem_switch_index=2,
        parallel_manager=SimpleNamespace(
            model=None,
            _resident_ep_built=False,
            _streamed_sp8_buffer=None,
            _decode_graph=None,
        ),
        gpu_paged_kv_cache_manager=SimpleNamespace(pool_is_mapped=True),
    )
    MethodType(function, worker)(label)
    assert len(recorder.lines) == 1
    return recorder.lines[0]


class _HalfInitializedManager:
    """A manager mid-construction: the attribute is not there yet."""

    @property
    def config(self):
        raise AttributeError(
            "'KDAStateGPUManager' object has no attribute '_config'"
        )

    sequence_state_items = {}


class _WrongTypeManager:
    """A manager whose slot map is not sized yet (``len(None)``)."""

    config = SimpleNamespace(num_state_items=256)
    sequence_state_items = None


@pytest.mark.parametrize("manager,wanted", [
    (_HalfInitializedManager(), "kda_slots=-1"),
    (_WrongTypeManager(), "kda_live=-1"),
])
def test_a_half_initialized_manager_cannot_fail_the_phase_switch(
    monkeypatch, manager, wanted
):
    """Observability must not take down the switch it observes.

    The block was guarded by ``except ImportError`` alone, but its body reads
    ``manager.config.num_state_items`` and ``manager.sequence_state_items``:
    a manager caught mid-construction raises AttributeError/TypeError, and
    that exception propagated out of the ledger and through the phase switch
    — a diagnostic taking down the thing it was added to observe. The
    sentinel reports "unknown" instead.
    """
    from batchgen.models.moonshotai.kimi_linear.wrappers import (
        KimiLinearKDAWrapper,
    )

    monkeypatch.setattr(KimiLinearKDAWrapper, "state_manager", manager)
    line = _run_ledger()
    assert wanted in line
    # The rest of the line — the holders a per-cycle subtraction needs — is
    # still reported; the failure is confined to the KDA fields.
    assert "driver_used=70.00" in line
    assert "kv_pool_mapped=True" in line


def test_a_healthy_manager_still_reports_the_real_slot_counts(monkeypatch):
    from batchgen.models.moonshotai.kimi_linear.wrappers import (
        KimiLinearKDAWrapper,
    )

    monkeypatch.setattr(KimiLinearKDAWrapper, "state_manager", SimpleNamespace(
        config=SimpleNamespace(num_state_items=320),
        sequence_state_items={7: 0, 8: 1, 9: 2},
    ))
    line = _run_ledger()
    assert "kda_slots=320" in line
    assert "kda_live=3" in line


def test_no_manager_at_all_reports_the_sentinel(monkeypatch):
    from batchgen.models.moonshotai.kimi_linear.wrappers import (
        KimiLinearKDAWrapper,
    )

    monkeypatch.setattr(KimiLinearKDAWrapper, "state_manager", None)
    line = _run_ledger()
    assert "kda_slots=-1 kda_live=-1" in line


def test_the_kda_block_is_not_narrowed_to_import_errors():
    body = ast.unparse(_method("_log_phase_memory_ledger"))
    assert "except Exception:" in body
    assert "except ImportError:" not in body
