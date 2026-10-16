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
