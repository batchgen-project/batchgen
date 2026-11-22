from types import SimpleNamespace

from tests.worker_method_source import load_worker_methods


_METHOD = load_worker_methods(["_finalize_async_load_minimal"], {})[
    "_finalize_async_load_minimal"
]


class _Worker(SimpleNamespace):
    def _decode_phase_context(self):
        return "context"

    def _bump_decode_row_epoch(self):
        self.bumps += 1


def test_finalize_async_load_copies_tensor_clear_and_stamps_row_epoch():
    class Engine:
        def _finalize_async_load_minimal(self, ctx, *args, **kwargs):
            assert ctx == "context"
            assert kwargs["query_book"] == {1: "entry"}
            assert kwargs["async_load_tensors_ref"] == ["tensors"]
            kwargs["async_load_tensors_ref"][0] = None
            return ["a"], [7]

    worker = _Worker(
        _kv_capacity_engine=Engine(),
        bumps=0,
        _async_load_tensors="tensors",
        query_book={1: "entry"},
    )

    assert _METHOD(worker, "task", [], [], [], [], [], object()) == (["a"], [7])
    assert worker.bumps == 1
    assert worker._async_load_tensors is None


def test_finalize_async_load_failure_before_clear_keeps_tensors():
    class Engine:
        def _finalize_async_load_minimal(self, ctx, *args, **kwargs):
            raise RuntimeError("invalid pending")

    tensors = object()
    worker = _Worker(
        _kv_capacity_engine=Engine(), bumps=0, _async_load_tensors=tensors, query_book={}
    )
    try:
        _METHOD(worker, "task", [], [], [], [], [], object())
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected finalize failure")
    assert worker._async_load_tensors is tensors
    assert worker.bumps == 0


def test_finalize_async_load_failure_after_clear_keeps_the_clear():
    class Engine:
        def _finalize_async_load_minimal(self, ctx, *args, **kwargs):
            kwargs["async_load_tensors_ref"][0] = None
            raise ValueError("sort failed")

    worker = _Worker(
        _kv_capacity_engine=Engine(), bumps=0, _async_load_tensors=object(), query_book={}
    )
    try:
        _METHOD(worker, "task", [], [], [], [], [], object())
    except ValueError:
        pass
    else:
        raise AssertionError("expected finalize failure")
    assert worker._async_load_tensors is None
    assert worker.bumps == 0


def test_finalize_async_load_without_tensor_attribute_creates_none():
    class Engine:
        def _finalize_async_load_minimal(self, ctx, *args, **kwargs):
            assert kwargs["async_load_tensors_ref"] is None
            return [], []

    worker = _Worker(_kv_capacity_engine=Engine(), bumps=0, query_book={})
    _METHOD(worker, "task", [], [], [], [], [], object())
    assert not hasattr(worker, "_async_load_tensors")
