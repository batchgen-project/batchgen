from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from batchgen.models.glm.glm5.Parallel_Strategy_Manager import (
    GLM5ParallelStrategyManager,
)


def _manager():
    manager = object.__new__(GLM5ParallelStrategyManager)
    manager.world_size = 8
    manager.global_rank = 0
    manager.rank = 0
    manager.model_config = SimpleNamespace(
        num_hidden_layers=78,
        num_local_experts=256,
    )
    manager.loaded_model_config = SimpleNamespace(model_type="glm_moe_dsa_5_2")
    manager.engine_config = SimpleNamespace(
        EP_Config=SimpleNamespace(
            enable_offloading=False,
            num_local_expert_per_layer=32,
        )
    )
    return manager


def _identity_document():
    identity = list(range(256))
    return {
        "schema": "batchgen.glm5_expert_placement",
        "version": 1,
        "model": "glm-5.2",
        "first_layer": 3,
        "last_layer": 77,
        "world_size": 8,
        "experts_per_rank": 32,
        "num_experts": 256,
        "physical_to_original": [identity.copy() for _ in range(75)],
    }


def test_glm52_identity_placement_validates_without_changing_source_indices():
    manager = _manager()
    placement = manager._validate_expert_placement_document(_identity_document())

    assert len(placement) == 75
    assert manager._expert_source_index(None, 3, 17) == 17
    assert manager._expert_source_index(placement, 3, 17) == 17


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc.update(version=2),
        lambda doc: doc.update(extra=True),
        lambda doc: doc["physical_to_original"].pop(),
        lambda doc: doc["physical_to_original"][0].pop(),
        lambda doc: doc["physical_to_original"][0].__setitem__(0, 0.5),
        lambda doc: doc["physical_to_original"][0].__setitem__(0, 1),
    ],
)
def test_glm52_placement_rejects_malformed_or_partial_mapping(mutate):
    document = _identity_document()
    mutate(document)

    with pytest.raises(ValueError):
        _manager()._validate_expert_placement_document(document)


def test_glm52_arbitrary_placement_permutes_gate_rows_and_bias_consistently():
    manager = _manager()
    document = _identity_document()
    document["physical_to_original"][0] = list(reversed(range(256)))
    placement = manager._validate_expert_placement_document(document)

    layers = []
    for _ in range(78):
        gate = SimpleNamespace(
            weight=torch.nn.Parameter(torch.arange(256).view(256, 1).float()),
            e_score_correction_bias=torch.nn.Parameter(torch.arange(256).float()),
        )
        layers.append(SimpleNamespace(mlp=SimpleNamespace(gate=gate)))
    manager.model = SimpleNamespace(model=SimpleNamespace(layers=layers))

    manager._apply_expert_placement_to_gates(placement)

    expected = torch.arange(255, -1, -1).float()
    assert torch.equal(layers[3].mlp.gate.weight[:, 0], expected)
    assert torch.equal(layers[3].mlp.gate.e_score_correction_bias, expected)
    assert manager._expert_source_index(placement, 3, 0) == 255
    assert manager._expert_source_index(placement, 3, 255) == 0


def test_glm52_placement_rejects_runtime_identity_mismatch():
    manager = _manager()
    manager.loaded_model_config = SimpleNamespace(model_type="glm_moe_dsa")

    with pytest.raises(ValueError, match="model_type"):
        manager._validate_expert_placement_document(deepcopy(_identity_document()))


def test_glm52_placement_rejects_partial_residency():
    manager = _manager()
    manager.engine_config.EP_Config.num_local_expert_per_layer = 16

    with pytest.raises(ValueError, match="all 32 physical experts"):
        manager._validate_expert_placement_document(_identity_document())
