"""DeepEP low-latency exchange: the send / receive split (``split_phases``).

``DeepEPLowLatencyExchange`` can run each LL dispatch / combine as DeepEP's
single kernel (``return_recv_hook=False``) or as a send launch followed by the
receive launch the returned hook issues (``return_recv_hook=True``). These
tests fake ``deep_ep.Buffer`` so the wrapper contract is checkable without the
extension; the real-extension equivalence (bit-identical receive buffers,
counts and combine outputs, eager and graph replay) is a node gate in
``tests/test_glm52_deepep_ll_gpu.py``.

Contract under test:
* both modes return the delivered values -- the split wrapper issues the hook
  before returning, exactly once, and never asks for ``async_finish``;
* single mode never asks DeepEP for a hook;
* a split request whose build hands back no callable hook is a named error,
  never a silent single-kernel fallback.
"""
import pytest
import torch

import batchgen.moe.deepep_ll as deepep_ll


class _PhaseBuffer:
    """Stand-in ``deep_ep.Buffer`` that models the two LL launch modes.

    ``return_recv_hook=False``: one call delivers the received values.
    ``return_recv_hook=True``: the call only "sends" -- its outputs hold a
    sentinel -- and the returned hook delivers the values. A wrapper that
    returns before issuing the hook therefore hands back the sentinel, and one
    that issues it twice is counted.
    """

    SENTINEL = -7.0

    def __init__(self, *, hook_missing=False):
        self.calls = []
        self.hooks_run = 0
        self.hook_missing = hook_missing

    def _deliver(self, dst, value):
        dst.fill_(value)

    def _maybe_split(self, kind, return_recv_hook, async_finish, dsts, value):
        self.calls.append((kind, bool(return_recv_hook), bool(async_finish)))
        if not return_recv_hook:
            for d in dsts:
                self._deliver(d, value)
            return None
        for d in dsts:
            d.fill_(self.SENTINEL)
        if self.hook_missing:
            return None

        def hook():
            self.hooks_run += 1
            for d in dsts:
                self._deliver(d, value)
        return hook

    def low_latency_dispatch(self, x, topk_idx, mtpr, num_experts, *,
                             use_fp8=False, round_scale=False, use_ue8m0=False,
                             async_finish=False, return_recv_hook=False):
        recv = torch.empty(2, 4, x.shape[1], dtype=torch.float32)
        scales = torch.empty(2, 4, x.shape[1] // 128, dtype=torch.float32)
        count = torch.empty(2, dtype=torch.float32)
        hook = self._maybe_split("dispatch", return_recv_hook, async_finish,
                                 (recv, scales, count), 3.0)
        payload = (recv, scales) if use_fp8 else recv
        return payload, count, ("h",), None, hook

    def low_latency_combine(self, x, topk_idx, topk_weights, handle, *,
                            async_finish=False, return_recv_hook=False,
                            out=None):
        hook = self._maybe_split("combine", return_recv_hook, async_finish,
                                 (out,), 5.0)
        return out, None, hook


def _phase_exchange(buffer):
    ex = deepep_ll.DeepEPLowLatencyExchange.__new__(
        deepep_ll.DeepEPLowLatencyExchange)
    ex.buffer = buffer
    ex.max_tokens_per_rank = 4
    ex.num_experts = 8
    return ex


@pytest.mark.parametrize("split", [False, True])
def test_the_exchange_split_delivers_the_single_kernel_values(split):
    buf = _PhaseBuffer()
    ex = _phase_exchange(buf)
    x = torch.zeros(4, 256, dtype=torch.bfloat16)
    idx = torch.zeros(4, 2, dtype=torch.int64)
    recv_x, recv_scales, count, handle = ex.dispatch_fp8(
        x, idx, split_phases=split)
    for t in (recv_x, recv_scales, count):
        assert torch.all(t == 3.0)
    plain = ex.dispatch(x, idx, split_phases=split)[0]
    assert torch.all(plain == 3.0)
    out = torch.empty(4, 256, dtype=torch.float32)
    combined = ex.combine(torch.zeros(2, 4, 256), idx, torch.ones(4, 2),
                          handle, out=out, split_phases=split)
    assert combined is out and torch.all(out == 5.0)
    # never async: the outputs are on the caller's stream when the call returns
    assert buf.calls == [("dispatch", split, False), ("dispatch", split, False),
                         ("combine", split, False)]
    # the hook is issued exactly once per split call, never in single mode
    assert buf.hooks_run == (3 if split else 0)


def test_the_exchange_split_without_a_hook_is_a_named_error():
    ex = _phase_exchange(_PhaseBuffer(hook_missing=True))
    x = torch.zeros(4, 256, dtype=torch.bfloat16)
    idx = torch.zeros(4, 2, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="no callable receive hook"):
        ex.dispatch_fp8(x, idx, split_phases=True)
    with pytest.raises(RuntimeError, match="no callable receive hook"):
        ex.combine(torch.zeros(2, 4, 256), idx, torch.ones(4, 2), ("h",),
                   out=torch.empty(4, 256), split_phases=True)
