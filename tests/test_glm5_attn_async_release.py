from types import MethodType

import torch

from batchgen.models.glm.glm5.wrappers import GLM5AttnWrapper


class _Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.streamed = torch.nn.Parameter(torch.ones(1))
        self.skeleton = torch.nn.Parameter(torch.full((1,), 7.0))


class _Core:
    def __init__(self, module, log):
        self.module = module
        self.log = log

    def get_weights(self, module_key, phase):
        self.log.append(("load", module_key, phase))
        return {"streamed": torch.full((1,), 3.0)}

    def free_weights_buffer(self, module_key):
        self.log.append(("free", module_key))

    def free_weights_buffer_async(self, module_key):
        self.log.append(
            (
                "free_async",
                module_key,
                self.module.streamed.numel(),
                self.module.skeleton.item(),
            )
        )


def _make_wrapper(phase):
    log = []
    module = _Attention()
    wrapper = object.__new__(GLM5AttnWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.module = module
    wrapper.layer_idx = 4
    wrapper.module_key = "attn_4"
    wrapper.persistent = False
    wrapper.phase = phase
    wrapper.core_engine = _Core(module, log)

    def _prefill(self, hidden_states, **kwargs):
        log.append(("prefill", hidden_states.item(), kwargs["marker"]))
        return "prefill-result"

    def _decode(self, hidden_states, **kwargs):
        log.append(("decode", hidden_states.item(), kwargs["marker"]))
        return "decode-result"

    wrapper._forward_prefill = MethodType(_prefill, wrapper)
    wrapper._forward_decode = MethodType(_decode, wrapper)
    return wrapper, log


def test_nonpersistent_prefill_releases_without_host_stream_sync(monkeypatch):
    wrapper, log = _make_wrapper("prefill")

    def _unexpected_current_stream(*args, **kwargs):
        raise AssertionError("prefill release must not query or synchronize in Python")

    monkeypatch.setattr(torch.cuda, "current_stream", _unexpected_current_stream)
    result = wrapper(hidden_states=torch.tensor(2.0), marker="ok")

    assert result == "prefill-result"
    assert log == [
        ("load", "attn_4", "prefill"),
        ("prefill", 2.0, "ok"),
        ("free_async", "attn_4", 0, 7.0),
    ]
    assert wrapper.module.streamed.numel() == 0
    torch.testing.assert_close(wrapper.module.skeleton, torch.full((1,), 7.0))


def test_nonpersistent_decode_keeps_synchronous_release(monkeypatch):
    wrapper, log = _make_wrapper("decode")

    class _Stream:
        def synchronize(self):
            log.append(("sync",))

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: _Stream())
    result = wrapper(hidden_states=torch.tensor(5.0), marker="ok")

    assert result == "decode-result"
    assert ("free", "attn_4") in log
    assert not any(entry[0] == "free_async" for entry in log)
    assert log.index(("sync",)) < log.index(("free", "attn_4"))
