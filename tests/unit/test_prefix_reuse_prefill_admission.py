import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]


def _install_torch_stub(monkeypatch):
    torch_stub = types.ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    batchgen_stub = types.ModuleType("batchgen")
    batchgen_stub.__path__ = [str(REPO_ROOT / "batchgen")]
    monkeypatch.setitem(sys.modules, "batchgen", batchgen_stub)


def _admission_module(monkeypatch):
    _install_torch_stub(monkeypatch)
    return importlib.import_module("batchgen.prefix_reuse.prefill_admission")


class _WorkerView:
    def estimate_pages_for_sequences_with_prefix(self, requests):
        sequence_id, _tokens, _capacity, _namespace = requests[0]
        return [{
            "sequence_id": sequence_id,
            "physical_pages_allocated": 2,
            "shared_prefix_pages": [10, 11],
        }]


def _seq(assigned_rank=0):
    return SimpleNamespace(
        global_idx=42,
        assigned_rank=assigned_rank,
        input_ids=object(),
    )


def test_prefix_admission_estimate_uses_worker_view_hit(monkeypatch):
    mod = _admission_module(monkeypatch)

    estimate = mod.estimate_prefix_allocation_for_admission(
        seq=_seq(assigned_rank=0),
        capacity_tokens=256,
        page_size=64,
        prefix_runtime_enabled=True,
        current_rank=0,
        worker_view=_WorkerView(),
        namespace_hash=123,
        prompt_tokens=lambda _seq: [1, 2, 3],
    )

    assert estimate.private_pages == 2
    assert estimate.shared_pages == [10, 11]


def test_prefix_admission_estimate_falls_back_to_logical_pages(monkeypatch):
    mod = _admission_module(monkeypatch)

    estimate = mod.estimate_prefix_allocation_for_admission(
        seq=_seq(assigned_rank=1),
        capacity_tokens=130,
        page_size=64,
        prefix_runtime_enabled=True,
        current_rank=0,
        worker_view=_WorkerView(),
        namespace_hash=123,
        prompt_tokens=lambda _seq: [1, 2, 3],
    )

    assert estimate.private_pages == 3
    assert estimate.shared_pages == []
