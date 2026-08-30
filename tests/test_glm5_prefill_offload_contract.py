import logging
from types import SimpleNamespace

import pytest
import torch

from batchgen.models.glm.glm5.Parallel_Strategy_Manager import (
    GLM5ParallelStrategyManager,
)
from batchgen.models.glm.glm5.wrappers import GLM5AttnWrapper
from batchgen.models.wrappers.attention import AttnWrapperBase
from batchgen.batchgen_worker import BatchGenWorker


def _fake_prefill_manager(num_layers=5):
    manager = object.__new__(GLM5ParallelStrategyManager)
    manager.rank = 0
    manager.model_config = SimpleNamespace(num_hidden_layers=num_layers)
    layers = []
    for layer_idx in range(num_layers):
        layer = SimpleNamespace(
            self_attn=SimpleNamespace(persistent=False),
            mlp=SimpleNamespace(),
        )
        if layer_idx >= manager.FIRST_K_DENSE:
            layer.mlp.shared_experts = SimpleNamespace(persistent=False)
            layer.mlp.experts = [
                SimpleNamespace(persistent=False)
                for _ in range(manager.NUM_TOTAL_EXPERTS)
            ]
        layers.append(layer)
    manager.model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    manager.weight_copy_task = {
        "attn": [f"attn_{layer_idx}" for layer_idx in range(num_layers)],
        "shared_expert": [
            f"shared_expert_{layer_idx}"
            for layer_idx in range(manager.FIRST_K_DENSE, num_layers)
        ],
        "routed_expert": [
            f"routed_expert_{layer_idx}_{expert_idx}"
            for layer_idx in range(manager.FIRST_K_DENSE, num_layers)
            for expert_idx in range(manager.NUM_TOTAL_EXPERTS)
        ],
    }
    return manager


def test_glm5_prefill_weight_offload_contract_accepts_all_streamed(caplog):
    manager = _fake_prefill_manager()
    caplog.set_level(logging.INFO)

    manager._validate_prefill_weight_offload_contract()

    assert "PREFILL_WEIGHT_OFFLOAD_CONTRACT" in caplog.text
    assert '"routed_experts_streamed":512' in caplog.text


def test_glm5_prefill_weight_offload_contract_rejects_persistent_expert():
    manager = _fake_prefill_manager()
    manager.model.model.layers[3].mlp.experts[7].persistent = True

    with pytest.raises(RuntimeError, match="persistent wrappers"):
        manager._validate_prefill_weight_offload_contract()


def test_glm5_prefill_kv_offload_audit_aggregates_layers():
    GLM5AttnWrapper.start_prefill_kv_offload_audit()
    GLM5AttnWrapper.record_prefill_kv_offload(
        "primary", 0, sequences=1, tokens=262_144
    )
    GLM5AttnWrapper.record_prefill_kv_offload(
        "primary", 0, sequences=1, tokens=65_536
    )
    GLM5AttnWrapper.record_prefill_kv_offload(
        "aux", 0, sequences=2, tokens=327_680
    )

    audit = GLM5AttnWrapper.finish_prefill_kv_offload_audit()

    assert audit["primary"][0] == {
        "calls": 2,
        "sequences": 2,
        "tokens": 327_680,
    }
    assert audit["aux"][0] == {
        "calls": 1,
        "sequences": 2,
        "tokens": 327_680,
    }
    assert GLM5AttnWrapper.glm5_prefill_kv_offload_audit is None


def test_prefill_retirement_audit_records_layer_and_task_count(monkeypatch):
    class DummyTask:
        def __init__(self):
            self.waited = False

        def wait(self):
            self.waited = True

    tasks = [DummyTask(), DummyTask()]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        AttnWrapperBase,
        "pending_prefill_offload_tasks",
        list(tasks),
    )
    monkeypatch.setattr(
        AttnWrapperBase,
        "pending_prefill_offload_tensors",
        [torch.zeros(1)],
    )
    monkeypatch.setattr(AttnWrapperBase, "pending_prefill_offload_layer_idx", 17)
    AttnWrapperBase.start_prefill_offload_retirement_audit()

    retired = AttnWrapperBase.retire_pending_prefill_offloads(
        device=torch.device("cpu"),
        reason="test",
    )
    audit = AttnWrapperBase.finish_prefill_offload_retirement_audit()

    assert retired == 2
    assert all(task.waited for task in tasks)
    assert audit == [{"layer_idx": 17, "tasks": 2}]


def test_worker_accepts_complete_per_layer_kv_offload_audit(monkeypatch, caplog):
    worker = object.__new__(BatchGenWorker)
    worker.rank = 3
    worker.model = SimpleNamespace(model=SimpleNamespace(layers=[
        SimpleNamespace(self_attn=SimpleNamespace(
            module=SimpleNamespace(indexer=object())
        )),
        SimpleNamespace(self_attn=SimpleNamespace(
            module=SimpleNamespace(indexer=None)
        )),
    ]))
    worker._local_to_uuid_map = {9: "sequence-9"}
    worker.global_batch = SimpleNamespace(
        get_sequence=lambda uuid: SimpleNamespace(prompt_length=262_144)
    )
    monkeypatch.setattr(GLM5AttnWrapper, "glm5_prefill_kv_offload_audit", {
        "primary": {
            0: {"calls": 1, "sequences": 1, "tokens": 262_144},
            1: {"calls": 1, "sequences": 1, "tokens": 262_144},
        },
        "aux": {
            0: {"calls": 1, "sequences": 1, "tokens": 262_144},
        },
    })
    monkeypatch.setattr(AttnWrapperBase, "prefill_offload_retirements", [
        {"layer_idx": 0, "tasks": 2},
        {"layer_idx": 1, "tasks": 1},
    ])
    monkeypatch.setattr(AttnWrapperBase, "pending_prefill_offload_tasks", [])
    monkeypatch.setattr(AttnWrapperBase, "pending_prefill_offload_tensors", [])
    monkeypatch.setattr(AttnWrapperBase, "pending_prefill_offload_layer_idx", None)
    caplog.set_level(logging.INFO)

    worker._finish_glm5_prefill_offload_audit([9])

    assert "PREFILL_KV_OFFLOAD_CONTRACT" in caplog.text
    assert '"sequence_lengths":[262144]' in caplog.text
    assert '"layers_without_auxiliary_by_design":[1]' in caplog.text
