"""DeepEP low-latency availability gate: shape/feature-aware, fail-loud.

The LL kernels' limits are compile-time (``kNumMaxTopK``, ``SWITCH_HIDDEN``)
and invisible to Python, so the gate reasons about (a) what the import
actually provides and (b) which build is in hand. These tests fake the
``deep_ep`` import so the contract is checkable without the extension.

Contract under test:
* a missing / unloadable ``deep_ep`` is a reason, never a crash;
* an API that predates the kwargs we pass is a reason, not a late assert;
* the upstream build accepts GLM-5.2 shapes (hidden 6144, top-8) and refuses
  K3 shapes (hidden 3584, top-16); the K3-marked build accepts both;
* a caller that names no shape still gets K3's old behavior (patched build
  required), so an upstream wheel can never silently flip a K3 node;
* the FP8 dispatch preconditions from csrc/deep_ep.cpp are enforced here.
"""
import os
import sys
import types

import pytest

import batchgen.moe.deepep_ll as deepep_ll


GLM52 = dict(hidden=6144, topk=8, num_experts=256, num_ranks=8,
             max_tokens_per_rank=64)
K3 = dict(hidden=3584, topk=16, num_experts=256, num_ranks=16,
          max_tokens_per_rank=32)


def _make_buffer_cls(*, dispatch_kwargs=None, combine_kwargs=None,
                     methods=None):
    dispatch_kwargs = dispatch_kwargs if dispatch_kwargs is not None else [
        "use_fp8", "round_scale", "use_ue8m0", "async_finish",
        "return_recv_hook"]
    combine_kwargs = combine_kwargs if combine_kwargs is not None else [
        "async_finish", "return_recv_hook", "out"]

    ns = {}
    src_dispatch = (
        "def low_latency_dispatch(self, x, topk_idx, mtpr, num_experts, "
        + ", ".join(f"{k}=None" for k in dispatch_kwargs) + "): pass"
    )
    src_combine = (
        "def low_latency_combine(self, x, topk_idx, topk_weights, handle, "
        + ", ".join(f"{k}=None" for k in combine_kwargs) + "): pass"
    )
    exec(src_dispatch, ns)
    exec(src_combine, ns)
    attrs = {
        "low_latency_dispatch": ns["low_latency_dispatch"],
        "low_latency_combine": ns["low_latency_combine"],
        "clean_low_latency_buffer": lambda self, *a: None,
        "get_low_latency_rdma_size_hint": staticmethod(lambda *a: 0),
    }
    for name in methods or ():
        attrs.pop(name, None)
    return type("Buffer", (), attrs)


@pytest.fixture
def fake_deep_ep(monkeypatch, tmp_path):
    """Install a fake ``deep_ep`` package; returns an installer callable."""

    def install(*, k3_marker=False, buffer_cls=None, topk_idx_t=int,
                sm90=True, drop=()):
        pkg_dir = tmp_path / ("k3" if k3_marker else "upstream")
        pkg_dir.mkdir(exist_ok=True)
        (pkg_dir / "__init__.py").write_text("")
        if k3_marker:
            (pkg_dir / deepep_ll.K3_BUILD_MARKER).write_text("k3")
        module = types.ModuleType("deep_ep")
        module.__file__ = os.path.join(str(pkg_dir), "__init__.py")
        module.Buffer = buffer_cls if buffer_cls is not None else _make_buffer_cls()
        module.topk_idx_t = topk_idx_t
        for name in drop:
            delattr(module, name)
        cpp = types.ModuleType("deep_ep_cpp")
        if sm90 is not None:
            cpp.is_sm90_compiled = lambda: sm90
        monkeypatch.setitem(sys.modules, "deep_ep", module)
        monkeypatch.setitem(sys.modules, "deep_ep_cpp", cpp)
        return module

    return install


def test_missing_module_is_a_reason_not_a_crash(monkeypatch):
    real_import = __import__

    def no_deep_ep(name, *a, **kw):
        if name == "deep_ep":
            raise ModuleNotFoundError("No module named 'deep_ep'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr("builtins.__import__", no_deep_ep)
    reason = deepep_ll.deepep_unavailable_reason(**GLM52)
    assert reason is not None and "not importable" in reason
    assert deepep_ll.deepep_available(**GLM52) is False
    assert deepep_ll.deepep_capability() is None


def test_upstream_build_runs_glm52_shapes(fake_deep_ep):
    fake_deep_ep(k3_marker=False)
    capability = deepep_ll.deepep_capability()
    assert capability.source == "upstream"
    assert capability.max_topk == deepep_ll.UPSTREAM_MAX_TOPK == 11
    assert 6144 in capability.hidden_sizes and 3584 not in capability.hidden_sizes
    assert deepep_ll.deepep_unavailable_reason(**GLM52) is None
    assert deepep_ll.deepep_unavailable_reason(use_fp8=True, **GLM52) is None


def test_upstream_build_refuses_k3_shapes(fake_deep_ep):
    fake_deep_ep(k3_marker=False)
    reason = deepep_ll.deepep_unavailable_reason(**K3)
    assert reason is not None
    # top-16 trips the upstream kNumMaxTopK limit before hidden is examined
    assert "top-k <= 11" in reason
    reason_hidden = deepep_ll.deepep_unavailable_reason(hidden=3584, topk=8)
    assert reason_hidden is not None and "hidden 3584" in reason_hidden


def test_k3_marked_build_is_a_superset(fake_deep_ep):
    fake_deep_ep(k3_marker=True)
    capability = deepep_ll.deepep_capability()
    assert capability.source == "k3-patched"
    assert capability.max_topk == 16
    assert {3584, 6144} <= capability.hidden_sizes
    assert deepep_ll.deepep_unavailable_reason(**K3) is None
    assert deepep_ll.deepep_unavailable_reason(**GLM52) is None


def test_undeclared_shape_keeps_the_k3_marker_contract(fake_deep_ep):
    """K3's planner calls deepep_available() with no shapes. An upstream wheel
    imports fine and only asserts inside the first dispatch, so an undeclared
    shape must still require the patched build."""
    fake_deep_ep(k3_marker=False)
    assert deepep_ll.deepep_available() is False
    reason = deepep_ll.deepep_unavailable_reason()
    assert deepep_ll.K3_BUILD_MARKER in reason
    fake_deep_ep(k3_marker=True)
    assert deepep_ll.deepep_available() is True


def test_old_api_without_our_kwargs_is_refused(fake_deep_ep):
    fake_deep_ep(k3_marker=True, buffer_cls=_make_buffer_cls(
        dispatch_kwargs=["async_finish", "return_recv_hook"]))
    reason = deepep_ll.deepep_unavailable_reason(**GLM52)
    assert reason is not None and "use_fp8" in reason

    fake_deep_ep(k3_marker=True, buffer_cls=_make_buffer_cls(
        combine_kwargs=["async_finish", "return_recv_hook"]))
    reason = deepep_ll.deepep_unavailable_reason(**GLM52)
    assert reason is not None and "'out'" in reason


def test_missing_buffer_method_is_refused(fake_deep_ep):
    fake_deep_ep(k3_marker=True,
                 buffer_cls=_make_buffer_cls(methods=["low_latency_combine"]))
    reason = deepep_ll.deepep_unavailable_reason(**GLM52)
    assert reason == "deep_ep.Buffer has no low_latency_combine"


def test_missing_topk_idx_dtype_is_refused(fake_deep_ep):
    fake_deep_ep(k3_marker=True, topk_idx_t=None)
    reason = deepep_ll.deepep_unavailable_reason(**GLM52)
    assert reason is not None and "topk_idx_t" in reason


def test_non_sm90_build_is_refused(fake_deep_ep):
    fake_deep_ep(k3_marker=True, sm90=False)
    reason = deepep_ll.deepep_unavailable_reason(**GLM52)
    assert reason is not None and "SM90" in reason
    # a build that does not report the flag at all must not be refused for it
    fake_deep_ep(k3_marker=True, sm90=None)
    assert deepep_ll.deepep_unavailable_reason(**GLM52) is None


def test_fp8_preconditions(fake_deep_ep):
    """csrc/deep_ep.cpp low_latency_dispatch asserts hidden % 512 == 0 and
    (num_ranks * tokens/rank) % 4 == 0 for the FP8 path."""
    fake_deep_ep(k3_marker=True)
    capability = deepep_ll.deepep_capability()
    # Every hidden size either build supports is already a multiple of 512, so
    # the hidden guard is defensive only. Pin that invariant: if a future build
    # adds a non-multiple, the FP8 fold must refuse it rather than abort inside
    # the dispatch.
    assert all(h % deepep_ll.FP8_HIDDEN_MULTIPLE == 0
               for h in capability.hidden_sizes)
    # (num_ranks * tokens/rank) % 4 is reachable: it depends on the decode
    # bucket, not on the build.
    reason = deepep_ll.deepep_unavailable_reason(
        hidden=6144, topk=8, num_ranks=3, max_tokens_per_rank=1, use_fp8=True)
    assert reason is not None and "% 4" in reason
    # the same shape without FP8 is fine
    assert deepep_ll.deepep_unavailable_reason(
        hidden=6144, topk=8, num_ranks=3, max_tokens_per_rank=1) is None
    # a build whose dispatch has no use_fp8 cannot fold the cast
    fake_deep_ep(k3_marker=True, buffer_cls=_make_buffer_cls(
        dispatch_kwargs=["round_scale", "use_ue8m0", "async_finish",
                         "return_recv_hook"]))
    reason = deepep_ll.deepep_unavailable_reason(**GLM52)
    assert reason is not None and "use_fp8" in reason


def test_expert_sharding_and_bucket_sanity(fake_deep_ep):
    fake_deep_ep(k3_marker=True)
    reason = deepep_ll.deepep_unavailable_reason(
        hidden=6144, topk=8, num_experts=250, num_ranks=8)
    assert reason is not None and "divisible" in reason
    reason = deepep_ll.deepep_unavailable_reason(
        hidden=6144, topk=8, max_tokens_per_rank=0)
    assert reason is not None and "max_tokens_per_rank" in reason


def test_require_deepep_raises_with_the_reason(fake_deep_ep):
    fake_deep_ep(k3_marker=False)
    with pytest.raises(RuntimeError, match="top-k <= 11"):
        deepep_ll.require_deepep(**K3)
    assert deepep_ll.require_deepep(**GLM52).source == "upstream"
