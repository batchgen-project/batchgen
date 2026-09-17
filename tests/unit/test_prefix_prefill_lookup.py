import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


_MODULE_PATH = (
    Path(__file__).parents[2] / "batchgen" / "prefix_reuse" / "prefill.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "test_prefix_cache_prefill", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

lookup_prefix_cache_for_prefill = _MODULE.lookup_prefix_cache_for_prefill
release_prefix_lookup_attachments = _MODULE.release_prefix_lookup_attachments


def _result(cached_tokens, handle, page_size=4):
    spans = []
    if cached_tokens:
        spans = [
            SimpleNamespace(
                group_id=0,
                raw_end_token=cached_tokens,
                pages=[SimpleNamespace(page_id=index) for index in range(
                    cached_tokens // page_size
                )],
            )
        ]
    return SimpleNamespace(
        common_cached_tokens=cached_tokens,
        attachment_handle=handle,
        materialization_spans=spans,
    )


class _Coordinator:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []
        self.released = []

    def lookup_and_attach(self, namespace_digest, token_ids):
        self.calls.append((list(namespace_digest), list(token_ids)))
        return self.results[len(self.calls) - 1]

    def release_attachment(self, handle):
        self.released.append(int(handle))


def test_lookup_separates_host_attachment_from_full_hit_compute_boundary():
    coordinator = _Coordinator([_result(4, 11), _result(8, 12)])

    lookup = lookup_prefix_cache_for_prefill(
        coordinator=coordinator,
        namespace_digest=(1, 2, 3, 4),
        prompt_token_ids=[list(range(6)), list(range(8))],
        page_size_tokens=4,
    )

    assert lookup.attached_tokens == (4, 8)
    assert lookup.compute_cached_tokens == (4, 7)
    assert lookup.has_hit is True
    release_prefix_lookup_attachments(coordinator=coordinator, lookup=lookup)
    assert coordinator.released == [11, 12]


def test_lookup_miss_has_no_attachment():
    coordinator = _Coordinator([_result(0, 0)])

    lookup = lookup_prefix_cache_for_prefill(
        coordinator=coordinator,
        namespace_digest=(1, 2, 3, 4),
        prompt_token_ids=[[10, 11]],
        page_size_tokens=4,
    )

    assert lookup.attached_tokens == (0,)
    assert lookup.compute_cached_tokens == (0,)
    assert lookup.has_hit is False


@pytest.mark.parametrize(
    "bad_result, match",
    [
        (_result(3, 1), "non-page-aligned"),
        (_result(0, 1), "handle does not match"),
        (_result(4, 0), "handle does not match"),
    ],
)
def test_invalid_lookup_result_fails_loud_and_releases_handle(
    bad_result, match
):
    coordinator = _Coordinator([bad_result])

    with pytest.raises(RuntimeError, match=match):
        lookup_prefix_cache_for_prefill(
            coordinator=coordinator,
            namespace_digest=(1, 2, 3, 4),
            prompt_token_ids=[[10, 11, 12, 13]],
            page_size_tokens=4,
        )

    expected = [int(bad_result.attachment_handle)] if bad_result.attachment_handle else []
    assert coordinator.released == expected


def test_later_lookup_failure_releases_all_prior_attachments():
    coordinator = _Coordinator([_result(4, 11), _result(3, 12)])

    with pytest.raises(RuntimeError, match="non-page-aligned"):
        lookup_prefix_cache_for_prefill(
            coordinator=coordinator,
            namespace_digest=(1, 2, 3, 4),
            prompt_token_ids=[list(range(6)), list(range(5))],
            page_size_tokens=4,
        )

    assert coordinator.released == [11, 12]
