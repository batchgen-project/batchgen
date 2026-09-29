"""The residue the non-pool path left behind is gone.

Pool mode never created an incremental writer, never read the end-of-generate()
result gather, and never set the global sampling / ignore_eos fallbacks — so
the writer submit was a stray all_reduce on every completion batch and the
fallbacks were dead branches. Everything here is source-level or one isolated
worker method: no GPU, no worker process, no server stack.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from batchgen.worker.completion import CompletionContext, CompletionHandler


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "batchgen" / "batchgen_worker.py"
CLIENT = ROOT / "batchgen" / "batchgen_client.py"
PACKAGE_INIT = ROOT / "batchgen" / "__init__.py"

DELETED = (
    ROOT / "batchgen" / "server" / "incremental_writer.py",
    ROOT / "batchgen" / "worker" / "batch_formation.py",
    ROOT / "batchgen" / "batchgen_server.py",
    ROOT / "batchgen" / "batchgen_server_dev.py",
)


def _tree(path):
    return ast.parse(path.read_text(), filename=str(path))


def _method(path, class_name, name):
    klass = next(
        node
        for node in _tree(path).body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in klass.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _isolated_method(path, class_name, name, globals_=None):
    """Compile one method on its own, free of the module's import stack."""
    method = copy.deepcopy(_method(path, class_name, name))
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
    return getattr(namespace["Isolated"], name)


def _worker_calls(path, name):
    return [
        node
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
    ]


# ------------------------------------------------------------- deleted code


def test_worker_carries_no_incremental_writer_residue():
    source = WORKER.read_text()
    assert "_submit_completed_to_incremental_writer" not in source
    assert "_incremental_writer" not in source


@pytest.mark.parametrize("path", DELETED, ids=lambda p: p.name)
def test_module_is_deleted(path):
    assert not path.exists()


def test_client_no_longer_defines_the_tcp_client():
    classes = {
        node.name for node in _tree(CLIENT).body if isinstance(node, ast.ClassDef)
    }
    assert "BatchGenClient" not in classes
    assert "BatchGenHttpClient" in classes


def test_package_no_longer_exports_the_tcp_client():
    assert "BatchGenClient" not in PACKAGE_INIT.read_text()


# --------------------------------------------------------- what must remain


def test_every_gather_completed_tokens_call_site_survives():
    """The writer submit went; the text gather beside it is load-bearing."""
    assert len(_worker_calls(WORKER, "_gather_completed_tokens")) == 4


# ------------------------------------------------------ generate() tail gone


def test_generate_neither_gathers_results_nor_returns_any():
    generate = _method(WORKER, "BatchGenWorker", "generate")
    assert "all_gather_object" not in ast.unparse(generate)
    for node in ast.walk(generate):
        if isinstance(node, ast.Return):
            assert node.value is None, (
                f"line {node.lineno}: generate() must not return a value"
            )


# ------------------------------------------ _select_tokens greedy fallthrough


def _select_tokens():
    """Bind `_select_tokens` (and the helper it calls) to a bare fake self."""
    namespace = {"torch": torch, "logging": logging, "Optional": Optional}
    select = _isolated_method(WORKER, "BatchGenWorker", "_select_tokens", namespace)
    all_greedy = _isolated_method(
        WORKER, "BatchGenWorker", "_decode_batch_all_greedy", namespace
    )
    worker = SimpleNamespace(rank=0, _logged_sampling=False)
    worker._decode_batch_all_greedy = all_greedy.__get__(worker)
    worker._build_sampling_tensors = lambda seqs: pytest.fail(
        "a params-free batch must not build sampling tensors"
    )
    return select.__get__(worker)


LOGITS = torch.tensor([[0.1, 0.9, 0.3], [2.0, 0.0, 1.0]])
ARGMAX = torch.tensor([[1], [0]])


def test_select_tokens_is_greedy_without_a_batch():
    assert torch.equal(_select_tokens()(LOGITS, None), ARGMAX)


def test_select_tokens_is_greedy_when_no_sequence_carries_params():
    batch = [SimpleNamespace(sampling_params=None) for _ in range(2)]
    assert torch.equal(_select_tokens()(LOGITS, batch), ARGMAX)


def test_select_tokens_is_greedy_for_all_none_param_dicts():
    params = {"temperature": None, "top_p": None, "top_k": None}
    batch = [SimpleNamespace(sampling_params=dict(params)) for _ in range(2)]
    assert torch.equal(_select_tokens()(LOGITS, batch), ARGMAX)


def test_select_tokens_still_samples_a_request_with_temperature(monkeypatch):
    import batchgen.sampling as sampling

    calls = []
    monkeypatch.setattr(
        sampling, "sample_tokens",
        lambda logits, **kw: calls.append(kw) or torch.tensor([[2], [2]]),
    )
    select = _select_tokens()
    tensors = (torch.tensor([0.7, 0.0]), torch.tensor([1.0, 1.0]), torch.tensor([0, 0]))
    select.__self__._build_sampling_tensors = lambda seqs: tensors
    batch = [SimpleNamespace(sampling_params={"temperature": 0.7}),
             SimpleNamespace(sampling_params=None)]
    assert torch.equal(select(LOGITS, batch), torch.tensor([[2], [2]]))
    assert len(calls) == 1 and calls[0]["temperature"] is tensors[0]


@pytest.mark.parametrize("attr", [
    "_ignore_eos", "_temperature", "_top_p", "_logged_greedy",
    "_per_sequence_sampling_params", "_batch_completed", "_prefill_completed_results",
])
def test_worker_has_no_global_fallback_state(attr):
    assert f"self.{attr}" not in WORKER.read_text()


# ------------------------------------------------- no global ignore_eos left


def test_completion_context_has_no_global_ignore_eos():
    fields = {f.name for f in dataclasses.fields(CompletionContext)}
    assert "ignore_eos" not in fields


def test_only_the_sequence_flag_can_suppress_eos():
    ctx = CompletionContext(
        eos_token_ids=frozenset({2}), model_context_length=64, rank=0
    )
    assert CompletionHandler.should_stop_at_eos(ctx, 2, True) is False
    assert CompletionHandler.should_stop_at_eos(ctx, 2) is True
