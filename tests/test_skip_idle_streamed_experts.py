"""Routing-informed release of idle streamed experts' weight slots.

A per-expert MoE loop declares the experts that got no tokens with
``skip_idle_streamed_experts`` right after routing, then runs only the experts
with tokens. CPU-only: the core engine is a fake that records every
``skip_weight_modules`` / ``get_weights`` / ``free_weights_buffer`` call.
"""

import ast
import sys
import types
from pathlib import Path

import pytest
import torch

from batchgen.models.wrappers.expert import (
    ExpertWrapperBase,
    skip_idle_streamed_experts,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "batchgen" / "models"


class _RecordingEngine:
    def __init__(self):
        self.events = []

    def skip_weight_modules(self, module_names):
        self.events.append(("skip", list(module_names)))
        return {"freed_present": 0, "release_on_landing": 0,
                "pending": len(module_names)}

    def get_weights(self, module_key):
        self.events.append(("get", module_key))
        return {}

    def free_weights_buffer(self, module_key):
        self.events.append(("free", module_key))


class _Scale(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, hidden_states):
        return hidden_states * self.scale


class _StreamedExpert(ExpertWrapperBase):
    """The base wrapper's load/free lifecycle without the CUDA stream sync."""

    def __init__(self, layer_idx, expert_idx, engine, scale=1.0,
                 persistent=False):
        super().__init__(_Scale(scale), layer_idx, expert_idx, engine, None,
                         None, persistent=persistent)

    def forward(self, hidden_states):
        if hidden_states.shape[0] == 0:
            raise AssertionError(f"{self.module_key} ran with 0 tokens")
        if not self.persistent:
            self.load_weights(self.module_key)
        out = self.module(hidden_states)
        if not self.persistent:
            self.core_engine.free_weights_buffer(self.module_key)
        return out


@pytest.fixture(autouse=True)
def _not_capturing(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing",
                        lambda: False)


def _ran(*module_keys):
    """The events of running each expert once, in order."""
    events = []
    for key in module_keys:
        events += [("get", key), ("free", key)]
    return events


def _assert_leased_and_freed_once(events, active, idle):
    """Every expert with tokens is leased once and freed once, after the one
    declaration of the idle ones; an idle expert is never leased."""
    skips = [event for event in events if event[0] == "skip"]
    assert skips == ([("skip", list(idle))] if idle else [])
    if idle:
        assert events[0] == skips[0]
    for key in active:
        assert events.count(("get", key)) == 1, key
        assert events.count(("free", key)) == 1, key
        assert events.index(("get", key)) < events.index(("free", key)), key
    for key in idle:
        assert ("get", key) not in events, key
        assert ("free", key) not in events, key


# --------------------------------------------------------------------------- #
#  The helper                                                                  #
# --------------------------------------------------------------------------- #
def test_idle_streamed_experts_are_declared_once_in_id_order():
    engine = _RecordingEngine()
    experts = [_StreamedExpert(5, e, engine) for e in range(4)]

    declared = skip_idle_streamed_experts(experts, range(4), [0, 3, 0, 0])

    keys = ["routed_expert_5_0", "routed_expert_5_2", "routed_expert_5_3"]
    assert declared == keys
    assert engine.events == [("skip", keys)]


def test_persistent_none_and_placeholder_experts_are_never_declared():
    engine = _RecordingEngine()
    experts = [
        None,
        types.SimpleNamespace(module_key="routed_expert_5_1"),
        _StreamedExpert(5, 2, engine, persistent=True),
        _StreamedExpert(5, 3, engine),
    ]

    declared = skip_idle_streamed_experts(experts, range(4), [0, 0, 0, 0])

    assert declared == ["routed_expert_5_3"]
    assert engine.events == [("skip", ["routed_expert_5_3"])]


def test_no_engine_call_when_nothing_streamed_is_idle():
    engine = _RecordingEngine()
    experts = [_StreamedExpert(5, 0, engine),
               _StreamedExpert(5, 1, engine, persistent=True)]

    assert skip_idle_streamed_experts(experts, range(2), [4, 1]) == []
    assert skip_idle_streamed_experts(experts, [1], [0]) == []
    assert engine.events == []


def test_counts_are_parallel_to_a_subset_of_ids():
    engine = _RecordingEngine()
    experts = [_StreamedExpert(2, e, engine) for e in range(6)]

    declared = skip_idle_streamed_experts(experts, [3, 4, 5], [0, 7, 0])

    assert declared == ["routed_expert_2_3", "routed_expert_2_5"]


def test_boolean_activity_counts_as_token_counts():
    engine = _RecordingEngine()
    experts = [_StreamedExpert(2, e, engine) for e in range(3)]

    declared = skip_idle_streamed_experts(experts, range(3),
                                          [True, False, True])

    assert declared == ["routed_expert_2_1"]


def test_count_length_mismatch_raises():
    engine = _RecordingEngine()
    experts = [_StreamedExpert(2, e, engine) for e in range(3)]

    with pytest.raises(ValueError, match="token_counts"):
        skip_idle_streamed_experts(experts, range(3), [0, 0])
    assert engine.events == []


def test_declaring_during_graph_capture_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing",
                        lambda: True)
    engine = _RecordingEngine()
    experts = [_StreamedExpert(2, e, engine) for e in range(2)]

    with pytest.raises(RuntimeError, match="capture"):
        skip_idle_streamed_experts(experts, range(2), [0, 1])
    assert engine.events == []
    # Nothing to declare: nothing to refuse.
    assert skip_idle_streamed_experts(experts, range(2), [1, 1]) == []


# --------------------------------------------------------------------------- #
#  Kimi-Linear / K3 serving MoE                                                #
# --------------------------------------------------------------------------- #
class _StaticGate:
    def __init__(self, topk_idx, topk_weight):
        self.topk_idx = topk_idx
        self.topk_weight = topk_weight
        self.top_k = topk_idx.shape[-1]

    def __call__(self, hidden_states):
        assert hidden_states.shape[-2] == self.topk_idx.shape[0]
        return self.topk_idx, self.topk_weight


def _kimi_linear_block(engine, topk_idx, topk_weight):
    scales = [1.0, 0.0, 3.0, 4.0, 5.0]
    experts = [
        _StreamedExpert(9, 0, engine, scales[0]),
        None,
        _StreamedExpert(9, 2, engine, scales[2]),
        _StreamedExpert(9, 3, engine, scales[3], persistent=True),
        _StreamedExpert(9, 4, engine, scales[4]),
    ]
    block = types.SimpleNamespace(
        config=None,
        hidden_dim=2,
        gate=_StaticGate(topk_idx, topk_weight),
        experts=experts,
        shared_experts=None,
        use_latent_moe=False,
    )
    return block, scales


def test_kimi_linear_serving_declares_idle_experts_and_runs_only_the_rest():
    from batchgen.models.moonshotai.kimi_linear.serving_modules import (
        moe_forward_serving,
    )

    hidden_states = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    topk_idx = torch.tensor([[4, 0], [3, 4], [0, 3]])
    topk_weight = torch.tensor([[0.25, 0.75], [0.6, 0.4], [0.5, 0.5]])
    engine = _RecordingEngine()
    block, scales = _kimi_linear_block(engine, topk_idx, topk_weight)

    out = moe_forward_serving(block, hidden_states)

    assert engine.events == (
        [("skip", ["routed_expert_9_2"])]
        + _ran("routed_expert_9_0", "routed_expert_9_4")
    )
    _assert_leased_and_freed_once(
        engine.events, ["routed_expert_9_0", "routed_expert_9_4"],
        ["routed_expert_9_2"],
    )
    expected = torch.zeros_like(hidden_states)
    for token in range(topk_idx.shape[0]):
        for slot in range(topk_idx.shape[1]):
            expected[0, token] += (hidden_states[0, token]
                                   * scales[topk_idx[token, slot]]
                                   * topk_weight[token, slot])
    torch.testing.assert_close(out, expected)


def test_kimi_linear_empty_rank_declares_every_streamed_expert_idle():
    from batchgen.models.moonshotai.kimi_linear.serving_modules import (
        moe_forward_serving,
    )

    engine = _RecordingEngine()
    block, _ = _kimi_linear_block(
        engine,
        torch.empty(0, 2, dtype=torch.long),
        torch.empty(0, 2),
    )

    out = moe_forward_serving(block, torch.empty(1, 0, 2))

    assert out.shape == (1, 0, 2)
    assert engine.events == [("skip", ["routed_expert_9_0",
                                       "routed_expert_9_2",
                                       "routed_expert_9_4"])]


def test_k3_mxfp4_wrapper_refuses_a_zero_token_forward_before_loading():
    from batchgen.models.moonshotai.kimi_linear.k3.mxfp4_expert import (
        KimiK3MXFP4ExpertWrapper,
    )

    engine = _RecordingEngine()
    wrapper = KimiK3MXFP4ExpertWrapper(_Scale(1.0), 4, 0, engine, None, None,
                                       persistent=False)

    with pytest.raises(ValueError, match="0-token"):
        wrapper(torch.empty(0, 8))
    assert engine.events == []


# --------------------------------------------------------------------------- #
#  GLM-5                                                                       #
# --------------------------------------------------------------------------- #
class _GlmGate(torch.nn.Module):
    def __init__(self, weights, indices):
        super().__init__()
        self.weights = weights
        self.indices = indices

    def forward(self, hidden_states):
        return self.weights, self.indices


def test_glm5_prefill_declares_idle_experts_before_loading(monkeypatch):
    from batchgen.models.glm.glm5.model import Glm5MoE

    monkeypatch.delenv("BATCHGEN_GLM5_MOE_FP32_ACCUM", raising=False)
    hidden_states = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    topk_indices = torch.tensor([[0, 2], [2, 0]])
    topk_weights = torch.tensor([[0.25, 0.75], [0.6, 0.4]])
    engine = _RecordingEngine()

    moe = object.__new__(Glm5MoE)
    torch.nn.Module.__init__(moe)
    moe.total_experts = 3
    moe.gate = _GlmGate(topk_weights, topk_indices)
    moe.experts = torch.nn.ModuleList(
        [_StreamedExpert(7, e, engine, float(e + 1)) for e in range(3)]
    )
    moe.shared_experts = _Scale(0.5)

    actual = moe._forward_prefill(hidden_states)

    assert engine.events == (
        [("skip", ["routed_expert_7_1"])]
        + _ran("routed_expert_7_0", "routed_expert_7_2")
    )
    _assert_leased_and_freed_once(
        engine.events, ["routed_expert_7_0", "routed_expert_7_2"],
        ["routed_expert_7_1"],
    )
    expected = hidden_states * 0.5
    for token in range(2):
        for slot in range(2):
            expert = int(topk_indices[token, slot])
            expected[0, token] += (hidden_states[0, token] * (expert + 1)
                                   * topk_weights[token, slot])
    torch.testing.assert_close(actual, expected)


def _glm5_mixed_moe(monkeypatch, engine, local_counts):
    """Rank 1 of 2, three local experts, all offloaded (no persistent ones)."""
    from batchgen.models.glm.glm5 import model as glm5_model

    offsets = [0]
    for count in local_counts:
        offsets.append(offsets[-1] + count)
    input_x = torch.arange(offsets[-1] * 2, dtype=torch.float32).view(-1, 2)

    def dispatch(global_x, topk_idx, token_idx, topk_pos, start, end):
        return (input_x, None, None, None, torch.tensor(local_counts),
                torch.tensor(offsets))

    mgn_kernel = types.ModuleType("mgn_kernel")
    mgn_kernel.fused_moe_token_dispatch = dispatch
    monkeypatch.setitem(sys.modules, "mgn_kernel", mgn_kernel)
    reduced = {}

    def scatter(res, *args):
        reduced["res"] = res
        return res

    monkeypatch.setattr(glm5_model, "scatter_weight_reduce_optimized", scatter)

    moe = object.__new__(glm5_model.Glm5MoE)
    torch.nn.Module.__init__(moe)
    moe.num_persistent_local_experts = 0
    moe.experts_per_rank = 3
    moe.routed_expert_start_idx = 3
    moe.routed_expert_end_idx = 6
    moe.token_idx = None
    moe.topk_pos = None
    moe.device = torch.device("cpu")
    moe.num_tokens_per_rank = 2
    moe.world_size = 2
    moe.num_experts_per_tok = 2
    moe.experts = [None] * 3 + [
        _StreamedExpert(11, e, engine, float(e)) for e in range(3, 6)
    ]
    return moe, input_x, reduced


def test_glm5_mixed_declares_idle_offloaded_experts_before_loading(
        monkeypatch):
    engine = _RecordingEngine()
    moe, input_x, reduced = _glm5_mixed_moe(monkeypatch, engine, [2, 0, 1])

    moe.expert_compute_mixed(torch.zeros(4, 2), None, None)

    assert engine.events == (
        [("skip", ["routed_expert_11_4"])]
        + _ran("routed_expert_11_3", "routed_expert_11_5")
    )
    _assert_leased_and_freed_once(
        engine.events, ["routed_expert_11_3", "routed_expert_11_5"],
        ["routed_expert_11_4"],
    )
    expected = torch.cat([input_x[:2] * 3.0, input_x[2:] * 5.0])
    torch.testing.assert_close(reduced["res"].float(), expected)


def test_glm5_mixed_declares_every_offloaded_expert_when_none_has_tokens(
        monkeypatch):
    engine = _RecordingEngine()
    moe, _, _ = _glm5_mixed_moe(monkeypatch, engine, [0, 0, 0])

    out = moe.expert_compute_mixed(torch.zeros(4, 2), None, None)

    assert out.shape == (4, 2)
    assert engine.events == [("skip", ["routed_expert_11_3",
                                       "routed_expert_11_4",
                                       "routed_expert_11_5"])]


# --------------------------------------------------------------------------- #
#  GPT-OSS EP-offload decode                                                   #
# --------------------------------------------------------------------------- #
def _gpt_oss_decode_moe(engine, scales):
    """Experts 0-2 persistent (grouped kernel, not run here), 3-5 offloaded."""
    from batchgen.models.openai.gpt_oss_120b.model import GptOssMoEDecode

    moe = object.__new__(GptOssMoEDecode)
    torch.nn.Module.__init__(moe)
    moe.total_experts = 6
    moe.non_persistent_expert_indices = [3, 4, 5]
    moe.experts = torch.nn.ModuleList(
        [None] * 3
        + [_StreamedExpert(13, e, engine, scales[e]) for e in range(3, 6)]
    )
    return moe


def test_gpt_oss_decode_streams_offloaded_experts_through_their_wrappers():
    scales = [0.0, 0.0, 0.0, 3.0, 4.0, 5.0]
    hidden_flat = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    topk_indices = torch.tensor([[3, 0], [5, 1], [0, 3], [2, 1]])
    topk_weights = torch.tensor([[0.25, 0.75], [0.6, 0.4], [0.5, 0.5],
                                 [0.3, 0.7]])
    output = torch.zeros_like(hidden_flat)
    engine = _RecordingEngine()
    moe = _gpt_oss_decode_moe(engine, scales)

    moe._single_expert_forward(hidden_flat, topk_indices, topk_weights, output)

    assert engine.events == (
        [("skip", ["routed_expert_13_4"])]
        + _ran("routed_expert_13_3", "routed_expert_13_5")
    )
    _assert_leased_and_freed_once(
        engine.events, ["routed_expert_13_3", "routed_expert_13_5"],
        ["routed_expert_13_4"],
    )
    expected = torch.zeros_like(hidden_flat)
    for token in range(topk_indices.shape[0]):
        for slot in range(topk_indices.shape[1]):
            expert = int(topk_indices[token, slot])
            if expert in moe.non_persistent_expert_indices:
                expected[token] += (hidden_flat[token] * scales[expert]
                                    * topk_weights[token, slot])
    torch.testing.assert_close(output, expected)


def test_gpt_oss_decode_declares_every_offloaded_expert_when_none_has_tokens():
    engine = _RecordingEngine()
    moe = _gpt_oss_decode_moe(engine, [1.0] * 6)
    hidden_flat = torch.ones(2, 2)
    output = torch.zeros_like(hidden_flat)

    moe._single_expert_forward(
        hidden_flat, torch.tensor([[0, 1], [2, 0]]), torch.full((2, 2), 0.5),
        output,
    )

    keys = ["routed_expert_13_3", "routed_expert_13_4", "routed_expert_13_5"]
    assert engine.events == [("skip", keys)]
    _assert_leased_and_freed_once(engine.events, [], keys)
    assert torch.count_nonzero(output) == 0


def test_gpt_oss_ep_offload_wraps_every_host_expert_as_streamed():
    """A host-resident expert has no GPU tensors: it must be a non-persistent
    wrapper, and the decode MoE refuses anything else for it."""
    from batchgen.models.openai.gpt_oss_120b.Parallel_Strategy_Manager import (
        GptOssParallelStrategyManager,
    )
    from batchgen.models.openai.gpt_oss_120b.wrappers import GptOssExpertWrapper

    resident = {
        f"mxfp4_{proj}_{kind}": torch.zeros(1)
        for proj in ("gate", "up", "down") for kind in ("packed", "scales")
    }
    experts = [types.SimpleNamespace() for _ in range(4)]
    experts[2] = types.SimpleNamespace(**resident)
    psm = object.__new__(GptOssParallelStrategyManager)
    psm.world_size = 2
    psm.rank = 1
    psm.routed_expert_gpu_start_idx = 2
    psm.routed_expert_gpu_end_idx = 3
    psm.routed_expert_host_end_idx = 4
    psm.core_engine = _RecordingEngine()
    psm.engine_config = None
    psm.model_config = types.SimpleNamespace(num_hidden_layers=1)
    psm.weight_copy_task = {"routed_expert": ["routed_expert_0_3"]}
    psm.model = types.SimpleNamespace(model=types.SimpleNamespace(layers=[
        types.SimpleNamespace(mlp=types.SimpleNamespace(experts=experts)),
    ]))

    psm._config_expert_module()

    assert isinstance(experts[2], GptOssExpertWrapper) and experts[2].persistent
    assert isinstance(experts[3], GptOssExpertWrapper)
    assert not experts[3].persistent
    assert experts[3].module_key == "routed_expert_0_3"
    # Another rank's experts stay unconfigured.
    assert not isinstance(experts[0], GptOssExpertWrapper)


def test_gpt_oss_decode_moe_has_no_direct_tensors_for_offloaded_experts():
    function = _function("openai/gpt_oss_120b/model.py", "GptOssMoEDecode",
                         "_single_expert_forward")
    read = {node.attr for node in ast.walk(function)
            if isinstance(node, ast.Attribute)}
    assert not read & {"gate_weights", "gate_scales", "up_weights",
                       "up_scales", "down_weights", "down_scales"}


# --------------------------------------------------------------------------- #
#  Every migrated loop declares before it runs an expert                      #
# --------------------------------------------------------------------------- #
MIGRATED_LOOPS = [
    ("moonshotai/kimi_linear/serving_modules.py", None, "moe_forward_serving"),
    ("moonshotai/kimi_k25/model.py", "KimiK25MoE", "_forward_decode"),
    ("moonshotai/kimi_k25/model.py", "KimiK25MoE", "_forward_prefill"),
    ("glm/glm5/model.py", "Glm5MoE", "expert_compute_mixed"),
    ("glm/glm5/model.py", "Glm5MoE", "_forward_prefill"),
    ("minimax/minimax_m25/model.py", "MiniMaxM25MoE",
     "moe_infer_loop_with_offloading"),
    ("minimax/minimax_m25/model.py", "MiniMaxM25MoE", "_forward_prefill"),
    ("deepseek/deepseekv3/modeling_deepseek_v3.py", "DeepseekV3MoE_Prefill",
     "moe_infer"),
    ("deepseek/deepseekv3/modeling_deepseek_v3.py",
     "DeepseekV3MoE_Decoding_FP8", "moe_infer_loop_with_offloading"),
    ("deepseek/deepseekv2/modeling_deepseek_v2.py", "DeepseekV2MoE",
     "moe_infer"),
    ("deepseek/deepseekv4_flash/model.py", "DeepSeekV4FlashMoE", "forward"),
    ("openai/gpt_oss_120b/model.py", "GptOssMoE", "forward"),
    ("openai/gpt_oss_120b/model.py", "GptOssMoEDecode",
     "_single_expert_forward"),
]


def _function(relative_path, class_name, name):
    tree = ast.parse((MODELS / relative_path).read_text())
    scope = tree.body
    if class_name is not None:
        scope = next(node for node in tree.body
                     if isinstance(node, ast.ClassDef)
                     and node.name == class_name).body
    return next(node for node in scope
                if isinstance(node, ast.FunctionDef) and node.name == name)


def _calls_expert(call):
    func = call.func
    if isinstance(func, ast.Name):
        return func.id == "expert"
    if isinstance(func, ast.Subscript):
        return (isinstance(func.value, ast.Attribute)
                and func.value.attr == "experts")
    return (isinstance(func, ast.Attribute)
            and func.attr == "forward_decode_offloaded")


def _expert_loops(function):
    return [
        node for node in ast.walk(function)
        if isinstance(node, ast.For)
        and any(isinstance(call, ast.Call) and _calls_expert(call)
                for call in ast.walk(node))
    ]


@pytest.mark.parametrize("relative_path,class_name,name", MIGRATED_LOOPS)
def test_migrated_loop_declares_idle_experts_before_its_expert_loop(
        relative_path, class_name, name):
    function = _function(relative_path, class_name, name)
    declares = [
        node for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "skip_idle_streamed_experts"
    ]
    loops = _expert_loops(function)

    assert len(declares) == 1, f"{relative_path}::{name}"
    assert loops, f"{relative_path}::{name} has no per-expert loop"
    assert declares[0].lineno < min(loop.lineno for loop in loops)
    # Every expert loop passes over its 0-token experts instead of running them.
    for loop in loops:
        assert any(isinstance(node, ast.Continue) for node in ast.walk(loop))


def test_k25_offloaded_decode_has_no_zero_token_load_free_branch():
    function = _function("moonshotai/kimi_k25/wrappers.py",
                         "KimiK25ExpertWrapper", "forward_decode_offloaded")
    body = [node for node in function.body
            if not (isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Constant))]
    calls = [node.func.attr for node in ast.walk(function)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)]

    # The first statement refuses a 0-token call before anything is loaded.
    guard = body[0]
    assert isinstance(guard, ast.If)
    assert any(isinstance(node, ast.Raise) for node in guard.body)
    assert not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("load_weights", "free_weights")
        for node in ast.walk(guard)
    )
    assert calls.count("load_weights") == 1
    assert calls.count("free_weights") == 1
