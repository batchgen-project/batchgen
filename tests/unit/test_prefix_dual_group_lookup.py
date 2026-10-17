"""A DSA prefix hit must be resident in every KV group, with one span each."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


_MODULE_PATH = (
    Path(__file__).parents[2] / "batchgen" / "prefix_reuse" / "prefill.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "test_prefix_cache_dual_prefill", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

PrefixCacheSequenceState = _MODULE.PrefixCacheSequenceState
lookup_prefix_cache_for_prefill = _MODULE.lookup_prefix_cache_for_prefill

PAGE = 4


def _span(group_id, raw_end_token, num_pages, first_page=0):
    return SimpleNamespace(
        group_id=group_id,
        raw_end_token=raw_end_token,
        pages=[
            SimpleNamespace(page_id=first_page + index)
            for index in range(num_pages)
        ],
    )


def _result(cached_tokens, handle, spans):
    return SimpleNamespace(
        common_cached_tokens=cached_tokens,
        attachment_handle=handle,
        materialization_spans=spans,
    )


class _Coordinator:
    def __init__(self, result):
        self.result = result
        self.released = []

    def lookup_and_attach(self, namespace_digest, token_ids):
        return self.result

    def release_attachment(self, handle):
        self.released.append(int(handle))


def _lookup(result, num_groups):
    coordinator = _Coordinator(result)
    lookup = lookup_prefix_cache_for_prefill(
        coordinator=coordinator,
        namespace_digest=(1, 2, 3, 4),
        prompt_token_ids=[list(range(10))],
        page_size_tokens=PAGE,
        num_groups=num_groups,
    )
    return lookup, coordinator


def test_dsa_lookup_accepts_one_aligned_span_per_group():
    lookup, coordinator = _lookup(
        _result(8, 21, [_span(0, 8, 2, 100), _span(1, 8, 2, 500)]),
        num_groups=2,
    )

    assert lookup.attached_tokens == (8,)
    assert lookup.compute_cached_tokens == (8,)
    assert coordinator.released == []


@pytest.mark.parametrize(
    "spans, match",
    [
        ([_span(0, 8, 2, 100)], "one span per KV group"),
        ([_span(0, 8, 2, 100), _span(2, 8, 2, 500)], "one span per KV group"),
        (
            [_span(0, 8, 2, 100), _span(0, 8, 2, 500)],
            "one span per KV group",
        ),
        (
            [_span(0, 8, 2, 100), _span(1, 4, 2, 500)],
            "does not cover the attached prefix",
        ),
        (
            [_span(0, 8, 2, 100), _span(1, 8, 1, 500)],
            "page count mismatch",
        ),
    ],
)
def test_dsa_lookup_rejects_a_group_out_of_lockstep(spans, match):
    with pytest.raises(RuntimeError, match=match):
        _lookup(_result(8, 21, spans), num_groups=2)


def test_single_group_lookup_is_unchanged_and_rejects_a_second_span():
    lookup, _ = _lookup(_result(8, 21, [_span(0, 8, 2, 100)]), num_groups=1)
    assert lookup.attached_tokens == (8,)

    with pytest.raises(RuntimeError, match="one span per KV group"):
        _lookup(
            _result(8, 21, [_span(0, 8, 2, 100), _span(1, 8, 2, 500)]),
            num_groups=1,
        )


def test_pool_state_keeps_planned_chain_pages_per_group():
    state = PrefixCacheSequenceState(
        lookup_result=None,
        attached_tokens=8,
        compute_cached_tokens=8,
        planned_shared_page_ids={0: (100, 101), 1: (500, 501)},
    )

    assert state.shared_page_ids == (100, 101)  # group 0 stays the default view
    assert state.shared_page_ids_for_group(1) == (500, 501)
    assert state.attachment_handle == 0


def test_state_without_attachment_shares_nothing_in_any_group():
    state = PrefixCacheSequenceState(
        lookup_result=None, attached_tokens=0, compute_cached_tokens=0
    )

    assert state.shared_page_ids == ()
    assert state.shared_page_ids_for_group(1) == ()


def test_lookup_hit_state_reads_each_group_span():
    state = PrefixCacheSequenceState(
        lookup_result=_result(8, 21, [_span(0, 8, 2, 100), _span(1, 8, 2, 500)]),
        attached_tokens=8,
        compute_cached_tokens=8,
    )

    assert state.shared_page_ids == (100, 101)
    assert state.shared_page_ids_for_group(1) == (500, 501)
    with pytest.raises(RuntimeError, match="exactly one span for KV group 2"):
        state.shared_page_ids_for_group(2)
