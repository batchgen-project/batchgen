"""Regression: K3's prefill MLA KV offload must be tracked, not fire-and-forget.

`async_offload_layer_kv_to_host` returns a KVAsyncTask backed by a std::async
thread that issues cudaMemcpyAsync on a d2h stream. The Kimi-Linear wrapper
dropped that future, so the worker's end-of-prefill
`retire_pending_prefill_offloads` had nothing to wait on and decode's
host->GPU load raced the offload: pinned host pages that the memcpy had not
written yet (pinned memory is not zeroed) were loaded into physical MLA-KV
pages, L7 attention produced NaN for half the batch, and the 512-request
decode contract scored MMLU 2.93% vs SGLang 86.91%.

Bisected to the admission-cap change at 97f20a74 (16 -> 32 rows/rank), which
only EXPOSED the race; instrumented runs T5/T6/T7 localised it to
non-finite host-loaded KV on physical MLA layer 1, rank-specific, varying
run to run.

The module imports fla, so the real method source is exec'd in isolation
against a recording stand-in for AttnWrapperBase (GLM-5 registers the same
way; this pins K3 to that contract).
"""
import ast
from pathlib import Path

WRAPPERS = (Path(__file__).resolve().parents[1] / "batchgen" / "models"
            / "moonshotai" / "kimi_linear" / "wrappers.py")


def _method_source(class_name, method_name):
    tree = ast.parse(WRAPPERS.read_text())
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == class_name)
    fn = next(n for n in cls.body
              if isinstance(n, ast.FunctionDef) and n.name == method_name)
    return fn


class _RecordingAWB:
    cur_batch = None
    prepack_num_sequences = None
    prepack_seq_lengths = None
    pinned = []
    tracked = []

    @classmethod
    def pin_prefill_offload_tensor(cls, tensor, layer_idx):
        cls.pinned.append((tensor, layer_idx))

    @classmethod
    def track_prefill_offload_task(cls, task, layer_idx):
        cls.tracked.append((task, layer_idx))


def test_offload_prepacked_kv_tracks_pins_and_completes_every_task():
    import pytest
    torch = pytest.importorskip("torch")

    fn = _method_source("KimiLinearAttnWrapper", "_offload_prepacked_kv")
    ns = {"AttnWrapperBase": _RecordingAWB, "torch": torch}
    exec(compile(ast.Module([fn], []), str(WRAPPERS), "exec"), ns)
    offload = ns["_offload_prepacked_kv"]

    class _Task:
        def __init__(self, layer_idx, seq_ids, log):
            self.key = ("task", layer_idx, seq_ids)
            self.log = log
            self.waited = False

        def wait(self):
            self.waited = True
            self.log.append(("wait", self.key))

        def __eq__(self, other):
            return self.key == other

        __hash__ = None

    class _View:
        calls = []
        log = []

        def async_offload_layer_kv_to_host(self, **kw):
            self.calls.append(kw)
            self.log.append(("issue", ("task", kw["layer_idx"], tuple(kw["sequence_ids"]))))
            return _Task(kw["layer_idx"], tuple(kw["sequence_ids"]), self.log)

    class _Engine:
        host_paged_kv_worker_view = _View()

    class _Self:
        layer_idx = 7
        core_engine = _Engine()

    _RecordingAWB.cur_batch = [30, 32, 34]
    _RecordingAWB.prepack_num_sequences = 3
    _RecordingAWB.prepack_seq_lengths = [2, 3, 1]
    _RecordingAWB.pinned.clear()
    _RecordingAWB.tracked.clear()

    offload_kv = torch.arange(6 * 4, dtype=torch.float32).view(6, 4)
    cu_seqlens = torch.tensor([0, 2, 5, 6])
    offload(_Self(), offload_kv, cu_seqlens)

    assert len(_Engine.host_paged_kv_worker_view.calls) == 3
    # every issued task is registered for the end-of-prefill retire
    assert [t for t, _ in _RecordingAWB.tracked] == [
        ("task", 7, (30,)), ("task", 7, (32,)), ("task", 7, (34,))]
    assert all(li == 7 for _, li in _RecordingAWB.tracked)
    # every source view is pinned so the allocator cannot reuse its storage
    assert len(_RecordingAWB.pinned) == 3
    assert all(li == 7 for _, li in _RecordingAWB.pinned)
    assert all(t.data_ptr() >= offload_kv.data_ptr()
               for t, _ in _RecordingAWB.pinned)
    # every task is completed before the next one is issued: the async
    # window is what corrupted physical-layer-1 host KV on the busiest ranks
    assert all(t.waited for t, _ in _RecordingAWB.tracked)
    log = _Engine.host_paged_kv_worker_view.log
    assert [k for k, _ in log] == ["issue", "wait"] * 3, log


def test_forward_prefill_retires_previous_layer_offloads_first():
    """The first statement of `_forward_prefill` must retire the prior MLA
    layer's offloads, mirroring GLM-5: the d2h memcpy must land before this
    layer's K/V allocation can be handed the same storage."""
    fn = _method_source("KimiLinearAttnWrapper", "_forward_prefill")
    first = fn.body[0]
    assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
    call = first.value
    assert (isinstance(call.func, ast.Attribute)
            and call.func.attr == "retire_pending_prefill_offloads_before_layer"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "AttnWrapperBase"), ast.dump(first)
