from pathlib import Path
from types import SimpleNamespace

import torch

from batchgen.models.wrappers.attention import AttnWrapperBase


_REPO_ROOT = Path(__file__).resolve().parents[1]


class _AsyncCore:
    def __init__(self, weights):
        self.weights = weights
        self.calls = []

    def get_weights(self, module_key, phase):
        self.calls.append(("get", module_key, phase))
        return self.weights

    def free_weights_buffer_async(self, module_key):
        self.calls.append(("free_async", module_key))


class _Attention(AttnWrapperBase):
    phase = "prefill"

    def _forward_prefill(self, hidden_states, **kwargs):
        return self.module(hidden_states)

    def _forward_decode(self, hidden_states, **kwargs):
        raise AssertionError("decode path must not run")


def test_prefill_attention_uses_async_release_without_host_stream_sync(monkeypatch):
    module = torch.nn.Linear(2, 2, bias=False)
    streamed_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    core = _AsyncCore({"weight": streamed_weight})
    wrapper = _Attention(
        module,
        layer_idx=7,
        core_engine=core,
        engine_config=SimpleNamespace(),
        model_config=SimpleNamespace(),
        persistent=False,
    )

    def _unexpected_sync(*args, **kwargs):
        raise AssertionError("async release must not request a host stream sync")

    monkeypatch.setattr(torch.cuda, "current_stream", _unexpected_sync)

    output = wrapper(hidden_states=torch.tensor([[5.0, 6.0]]))

    torch.testing.assert_close(output, torch.tensor([[17.0, 39.0]]))
    assert core.calls == [
        ("get", "attn_7", "prefill"),
        ("free_async", "attn_7"),
    ]
    assert module.weight.numel() == 0
    assert wrapper._applied_param_keys is None


def test_clear_weight_bindings_preserves_skeleton_parameters():
    module = torch.nn.Linear(2, 2, bias=True)
    original_bias = module.bias.detach().clone()
    wrapper = _Attention(
        module,
        layer_idx=0,
        core_engine=SimpleNamespace(),
        engine_config=SimpleNamespace(),
        model_config=SimpleNamespace(),
        persistent=True,
    )

    wrapper.apply_weights({"weight": torch.ones_like(module.weight)})
    wrapper.clear_weight_bindings()

    assert module.weight.numel() == 0
    torch.testing.assert_close(module.bias, original_bias)


def test_h2d_weight_worker_publishes_event_ordered_without_stream_sync():
    source = (
        _REPO_ROOT / "core" / "HtoD_Engine" / "HtoD_Engine.cu"
    ).read_text()
    worker = source[source.index("void HtoD_Engine::HtoD_Worker()") :]
    module_loop = worker[
        worker.index("for (auto& [tensor_name, host_tensor_storage] : src)") :
    ]
    copy_pos = module_loop.index("CUDA_CHECK(cudaMemcpyAsync(")
    loop_end_pos = module_loop.index("\n                }\n", copy_pos)
    publish_pos = module_loop.index(
        "this->gpu_weight_buffer_.weights_copy_enqueued("
    )

    assert copy_pos < loop_end_pos < publish_pos
    assert "cudaStreamSynchronize" not in module_loop[loop_end_pos:publish_pos]
    assert "this->blocking_copy_(" not in module_loop[:publish_pos]

    weight_buffer_source = (
        _REPO_ROOT
        / "core"
        / "GPU_Weight_Buffer"
        / "GPU_Weight_Buffer.cpp"
    ).read_text()
    publish = weight_buffer_source[
        weight_buffer_source.index(
            "void GPU_Weight_Buffer::weights_copy_enqueued("
        ) :
    ]
    event_pos = publish.index("CUDA_CHECK(cudaEventRecord(ready_event, copy_stream));")
    map_pos = publish.index("this->module_in_buffers_[module_name] =")
    notify_pos = publish.index("this->cv_.notify_all();")

    assert event_pos < map_pos < notify_pos


def test_weight_wait_timeout_allows_long_prefill_but_keeps_decode_watchdog():
    source = (
        _REPO_ROOT / "core" / "GPU_Weight_Buffer" / "GPU_Weight_Buffer.cpp"
    ).read_text()
    get_weights = source[
        source.index("module_weight_tensor_map GPU_Weight_Buffer::get_weights(") :
        source.index("void GPU_Weight_Buffer::weights_copy_enqueued(")
    ]

    assert 'phase == "prefill" ? std::chrono::seconds(120)' in get_weights
    assert ': std::chrono::seconds(2)' in get_weights
