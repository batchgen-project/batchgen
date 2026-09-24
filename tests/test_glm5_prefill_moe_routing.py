import torch

from batchgen.models.glm.glm5.model import Glm5MoE


class _StaticGate(torch.nn.Module):
    def __init__(self, weights: torch.Tensor, indices: torch.Tensor):
        super().__init__()
        self.weights = weights
        self.indices = indices

    def forward(self, hidden_states: torch.Tensor):
        assert hidden_states.shape[0] == self.indices.shape[0]
        return self.weights, self.indices


class _ScaleExpert(torch.nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, hidden_states: torch.Tensor):
        return hidden_states * self.scale


def test_glm5_prefill_reuses_one_token_index_per_expert(monkeypatch):
    monkeypatch.delenv("BATCHGEN_GLM5_MOE_FP32_ACCUM", raising=False)

    hidden_states = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]
    )
    topk_indices = torch.tensor(
        [[0, 2], [1, 0], [2, 1], [0, 1]], dtype=torch.int64
    )
    topk_weights = torch.tensor(
        [[0.25, 0.75], [0.60, 0.40], [0.55, 0.45], [0.80, 0.20]]
    )

    moe = object.__new__(Glm5MoE)
    torch.nn.Module.__init__(moe)
    moe.total_experts = 3
    moe.gate = _StaticGate(topk_weights, topk_indices)
    moe.experts = torch.nn.ModuleList(
        [_ScaleExpert(1.0), _ScaleExpert(2.0), _ScaleExpert(3.0)]
    )
    moe.shared_experts = _ScaleExpert(0.5)

    actual = moe._forward_prefill(hidden_states)

    expected = hidden_states.clone() * 0.5
    flat_expected = expected.view(-1, expected.shape[-1])
    flat_hidden = hidden_states.view(-1, hidden_states.shape[-1])
    scales = torch.tensor([1.0, 2.0, 3.0])
    for token in range(topk_indices.shape[0]):
        for slot in range(topk_indices.shape[1]):
            expert = topk_indices[token, slot]
            flat_expected[token] += (
                flat_hidden[token] * scales[expert] * topk_weights[token, slot]
            )

    torch.testing.assert_close(actual, expected)
