from contextlib import contextmanager
from types import SimpleNamespace

import torch

from batchgen.models.glm.glm5.model import Glm5DecoderLayer, Glm5MLP
from batchgen.models.glm.glm5.wrappers import _GLM5_PREFILL_CATEGORIES


class _Timer:
    def __init__(self):
        self.ops = []
        self.enabled = True

    @contextmanager
    def timed(self, name, layer_idx):
        self.ops.append((name, layer_idx))
        yield


class _Attention(torch.nn.Module):
    def __init__(self, phase):
        super().__init__()
        self.module = SimpleNamespace(config=SimpleNamespace(phase=phase))

    def forward(self, *, hidden_states, **_kwargs):
        return hidden_states, None, None


class _DenseMLP(Glm5MLP):
    def __init__(self):
        torch.nn.Module.__init__(self)

    def forward(self, hidden_states):
        return hidden_states


def _make_dense_layer(phase):
    layer = object.__new__(Glm5DecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.layer_idx = 2
    layer.input_layernorm = torch.nn.Identity()
    layer.self_attn = _Attention(phase)
    layer.post_attention_layernorm = SimpleNamespace(
        weight=torch.ones(1),
        eps=1e-6,
    )
    layer.mlp = _DenseMLP()
    return layer


def test_prefill_layer_uses_prefill_timer_for_uncovered_layer_ops(monkeypatch):
    prefill_timer = _Timer()
    decode_timer = _Timer()
    monkeypatch.setattr("batchgen.timing.get_prefill_timer", lambda: prefill_timer)
    monkeypatch.setattr("batchgen.timing.get_decode_timer", lambda: decode_timer)
    monkeypatch.setattr(
        "batchgen.attention.fused_kernels.cuda_add_rmsnorm",
        lambda residual, hidden, _weight, _eps: (hidden, residual),
    )

    _make_dense_layer("prefill")(torch.ones(1, 1, 1))

    assert prefill_timer.ops == [
        ("input_norm", 2),
        ("add_rmsnorm", 2),
        ("dense_mlp", 2),
        ("residual_add", 2),
    ]
    assert decode_timer.ops == []


def test_prefill_category_inventory_covers_worker_and_layer_tail():
    required = {
        "scheduler_capacity_all_gather",
        "setup_prepack",
        "setup_flatten",
        "microbatch_input_concat",
        "microbatch_cu_seqlens",
        "embedding",
        "input_norm",
        "add_rmsnorm",
        "dense_mlp",
        "residual_add",
        "moe_pointer_table_h2d",
        "final_norm",
        "last_token_gather",
        "lm_head",
        "token_select",
    }

    assert required <= set(_GLM5_PREFILL_CATEGORIES)
