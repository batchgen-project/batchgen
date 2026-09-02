from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from batchgen.models.glm.glm5 import model as glm5_model
from batchgen.models.glm.glm5.Parallel_Strategy_Manager import (
    _balanced_prefill_ep_eligible,
)
from batchgen.models.glm.glm5.model import Glm5MoE


class _Core:
    def __init__(self, log):
        self.log = log

    def free_weights_buffer(self, key):
        self.log.append(("free", key))

    def free_weights_buffers_async(self, keys):
        self.log.append(("free_many_async", tuple(keys)))

    def free_weights_buffer_async(self, key):
        self.log.append(("free_async", key))


class _Expert:
    def __init__(self, key, core, log):
        self.module_key = key
        self.core_engine = core
        self.log = log
        self.persistent = False
        self.is_fp8 = True
        self.weight_dequant_scale = {
            "gate_proj.weight_scale_inv": torch.ones(1, 1),
            "up_proj.weight_scale_inv": torch.ones(1, 1),
            "down_proj.weight_scale_inv": torch.ones(1, 1),
        }
        self.weights = {
            "gate_proj.weight": torch.ones(1, 1),
            "up_proj.weight": torch.ones(1, 1),
            "down_proj.weight": torch.ones(1, 1),
        }

    def load_weights_pinned(self):
        self.log.append(("load", self.module_key))
        return self.weights


class _Comm:
    def __init__(self, rank, log):
        self.rank = rank
        self.log = log

    def change_state(self, enable):
        assert enable is True
        return nullcontext()

    def all_gather(self, output, local, stream=None):
        self.log.append(("all_gather", local.dtype, local.clone()))
        rows = local.shape[0]
        if local.dtype == torch.bfloat16:
            remote = torch.arange(1, rows + 1, dtype=local.dtype).unsqueeze(1)
        elif local.dtype == torch.float32:
            remote = (
                torch.arange(1, rows + 1, dtype=local.dtype).unsqueeze(1) / 10
            )
        elif local.dtype == torch.int32:
            remote = torch.arange(rows, dtype=local.dtype).unsqueeze(1)
        else:
            raise AssertionError(f"unexpected all-gather dtype: {local.dtype}")
        output.copy_(torch.cat((remote, local), dim=0))

    def reduce_scatter(self, output, global_result, stream=None):
        self.log.append(("reduce_scatter", global_result.clone()))
        rank_chunk = global_result.view(2, *output.shape)[self.rank]
        output.copy_(rank_chunk + 100)


@pytest.fixture(autouse=True)
def _reset_prefill_state():
    Glm5MoE._prefill_buf = None
    Glm5MoE._prefill_ptrs_pinned = None
    Glm5MoE._prefill_ptrs_dev = None
    Glm5MoE._prefill_ring_pending = None
    Glm5MoE._prefill_shared_pending = None
    Glm5MoE._prefill_retire_executor = None
    Glm5MoE._prefill_retire_future = None
    Glm5MoE._prefill_grouped_logged = True
    yield
    Glm5MoE._prefill_buf = None
    Glm5MoE._prefill_ptrs_pinned = None
    Glm5MoE._prefill_ptrs_dev = None
    Glm5MoE._prefill_ring_pending = None
    Glm5MoE._prefill_shared_pending = None
    Glm5MoE._prefill_retire_executor = None
    Glm5MoE._prefill_retire_future = None


@pytest.mark.parametrize(
    ("rank_prompt_lengths", "world_size", "token_cap", "expected"),
    [
        ([[2048] * 42] * 8, 8, 131072, True),
        ([[65536]] * 8, 8, 131072, True),
        ([[131072]] * 8, 8, 131072, True),
        ([[262144]] * 8, 8, 131072, True),
        ([[2048], [], [2048], [2048]], 4, 131072, False),
        ([[2048, 1024], [1024, 2048]], 2, 131072, False),
        ([[70000, 70000]] * 8, 8, 131072, False),
        ([[2048]], 1, 131072, False),
    ],
)
def test_balanced_prefill_ep_eligibility(
    rank_prompt_lengths, world_size, token_cap, expected
):
    assert (
        _balanced_prefill_ep_eligible(
            rank_prompt_lengths,
            world_size,
            token_cap,
        )
        is expected
    )


def test_ep_prepare_loads_only_owned_expert_shard():
    log = []
    core = _Core(log)
    experts = [_Expert(f"expert_{idx}", core, log) for idx in range(4)]
    shared = _Expert("shared", core, log)

    moe = object.__new__(Glm5MoE)
    torch.nn.Module.__init__(moe)
    moe.config = SimpleNamespace(phase="prefill")
    moe.layer_idx = 3
    moe._prefill_grouped_enabled = True
    moe._prefill_ep_enabled = True
    moe._prefill_prepared_keys = None
    moe._prefill_weight_prototypes = None
    moe._prefill_shared_key = None
    moe.routed_expert_start_idx = 2
    moe.routed_expert_end_idx = 4
    moe.experts = experts
    moe.shared_experts = shared

    Glm5MoE._prefill_buf = SimpleNamespace(num_experts=2)
    Glm5MoE._prefill_ptrs_pinned = torch.empty(78, 6, 2, dtype=torch.int64)
    Glm5MoE._prefill_ptrs_dev = torch.empty(6, 2, dtype=torch.int64)

    moe._prefill_prepare_weights()

    assert log == [
        ("load", "expert_2"),
        ("load", "expert_3"),
        ("load", "shared"),
    ]
    assert moe._prefill_prepared_keys == ["expert_2", "expert_3"]
    assert moe._prefill_shared_key == "shared"
    assert Glm5MoE._prefill_ptrs_dev[0, 0].item() == experts[2].weights[
        "gate_proj.weight"
    ].data_ptr()
    assert Glm5MoE._prefill_ptrs_dev[0, 1].item() == experts[3].weights[
        "gate_proj.weight"
    ].data_ptr()


def test_ep_forward_routes_locally_then_gathers_routes_and_hidden(monkeypatch):
    log = []
    routed_core = _Core(log)
    shared_core = _Core(log)
    comm = _Comm(rank=1, log=log)

    moe = object.__new__(Glm5MoE)
    torch.nn.Module.__init__(moe)
    moe.layer_idx = 3
    moe.num_experts_per_tok = 1
    moe.world_size = 2
    moe.device = torch.device("cpu")
    moe.comm = comm
    moe._prefill_ep_enabled = True
    moe.routed_expert_start_idx = 2
    moe.routed_expert_end_idx = 4
    moe.experts = [SimpleNamespace(core_engine=routed_core) for _ in range(4)]

    def gate(local_hidden):
        log.append(("gate", local_hidden.clone()))
        return (
            local_hidden.float() / 10,
            torch.full(
                (local_hidden.shape[0], 1),
                2,
                dtype=torch.int64,
            ),
        )

    moe.gate = gate
    moe.shared_experts = SimpleNamespace(
        core_engine=shared_core,
        cached_gate=torch.ones(1),
        cached_up=torch.ones(1),
        cached_down=torch.ones(1),
        _forward_impl=lambda hidden: torch.ones_like(hidden),
    )
    moe._prefill_prepared_keys = ["expert_2", "expert_3"]
    moe._prefill_weight_prototypes = tuple(torch.ones(1) for _ in range(6))
    moe._prefill_shared_key = "shared"

    Glm5MoE._prefill_ptrs_dev = torch.zeros(6, 2, dtype=torch.int64)
    Glm5MoE._prefill_buf = SimpleNamespace(
        token_window=4,
        local_token_window=2,
        ep_size=2,
        num_experts=2,
        all_tokens=torch.empty(4, 1, dtype=torch.bfloat16),
        all_topk_weights=torch.empty(4, 1, dtype=torch.float32),
        all_topk_indices=torch.empty(4, 1, dtype=torch.int32),
        global_result=torch.empty(4, 1, dtype=torch.bfloat16),
    )

    def grouped_window(hidden, weights, indices, output, timer):
        log.append(
            (
                "grouped_window",
                hidden.clone(),
                weights.clone(),
                indices.clone(),
            )
        )
        output.copy_(hidden * 10)

    moe._prefill_grouped_routed_window = grouped_window
    monkeypatch.setattr(glm5_model.torch.cuda, "current_stream", lambda device: object())

    output = moe._forward_prefill_grouped(
        torch.tensor([[3.0], [4.0], [5.0]], dtype=torch.bfloat16)
    )

    torch.testing.assert_close(
        output,
        torch.tensor([[131.0], [141.0], [151.0]], dtype=torch.bfloat16),
    )
    gate_calls = [entry for entry in log if entry[0] == "gate"]
    assert len(gate_calls) == 1
    torch.testing.assert_close(
        gate_calls[0][1],
        torch.tensor([[3.0], [4.0], [5.0]], dtype=torch.bfloat16),
    )
    grouped = next(entry for entry in log if entry[0] == "grouped_window")
    torch.testing.assert_close(
        grouped[1],
        torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        grouped[2],
        torch.tensor([[0.1], [0.2], [0.3], [0.4]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        grouped[3],
        torch.tensor([[0], [1], [2], [2]], dtype=torch.int32),
    )
    grouped_windows = [entry for entry in log if entry[0] == "grouped_window"]
    assert len(grouped_windows) == 2
    torch.testing.assert_close(
        grouped_windows[1][1],
        torch.tensor([[1.0], [5.0]], dtype=torch.bfloat16),
    )
    assert [entry[0] for entry in log].count("all_gather") == 6
    assert [entry[0] for entry in log].count("reduce_scatter") == 2
    assert ("free_many_async", ("expert_2", "expert_3")) in log
    assert ("free_async", "shared") in log
