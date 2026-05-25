from __future__ import annotations

from types import SimpleNamespace

import torch

from batchgen.prefix_reuse.prefill import (
    build_prefix_cache_prefill_inputs,
    lookup_prefix_cache_for_prefill,
    release_prefix_cache_lookup_attachments,
)


class _Coordinator:
    def __init__(self, cached_tokens: list[int], handles: list[int]):
        self.cached_tokens = list(cached_tokens)
        self.handles = list(handles)
        self.lookup_calls = []
        self.release_calls = []

    def lookup_and_attach(self, namespace_digest, token_ids):
        index = len(self.lookup_calls)
        self.lookup_calls.append((list(namespace_digest), list(token_ids)))
        return SimpleNamespace(
            common_cached_tokens=self.cached_tokens[index],
            attachment_handle=self.handles[index],
        )

    def release_attachment(self, handle):
        self.release_calls.append(int(handle))


def test_lookup_prefix_cache_for_prefill_preserves_request_order():
    coordinator = _Coordinator(cached_tokens=[4, 0, 8], handles=[11, 0, 12])

    lookup = lookup_prefix_cache_for_prefill(
        coordinator=coordinator,
        namespace_digest=(1, 2, 3, 4),
        prompt_token_ids=[
            [10, 11, 12, 13, 14],
            [20, 21],
            [30, 31, 32, 33, 34, 35, 36, 37],
        ],
    )

    assert lookup.prefix_shared_tokens == (4, 0, 8)
    assert lookup.has_hit is True
    assert coordinator.lookup_calls == [
        ([1, 2, 3, 4], [10, 11, 12, 13, 14]),
        ([1, 2, 3, 4], [20, 21]),
        ([1, 2, 3, 4], [30, 31, 32, 33, 34, 35, 36, 37]),
    ]


def test_build_prefix_cache_prefill_inputs_uses_suffix_only_tokens():
    coordinator = _Coordinator(cached_tokens=[3, 0, 5], handles=[11, 0, 12])
    lookup = lookup_prefix_cache_for_prefill(
        coordinator=coordinator,
        namespace_digest=(1, 2, 3, 4),
        prompt_token_ids=[
            [10, 11, 12, 13, 14],
            [20, 21],
            [30, 31, 32, 33, 34],
        ],
    )

    inputs = build_prefix_cache_prefill_inputs(
        local_indices=[7, 8, 9],
        sequence_ids=[100, 101, 102],
        input_ids=[
            torch.tensor([[10, 11, 12, 13, 14]]),
            torch.tensor([[20, 21]]),
            torch.tensor([[30, 31, 32, 33, 34]]),
        ],
        prompt_lengths=[5, 2, 5],
        lookup=lookup,
    )

    assert [item.tolist() for item in inputs.plan.suffix_input_ids] == [
        [13, 14],
        [20, 21],
        [34],
    ]
    assert [item.tolist() for item in inputs.plan.suffix_position_ids] == [
        [3, 4],
        [0, 1],
        [4],
    ]
    assert [item.tolist() for item in inputs.input_ids_list] == [
        [[13, 14]],
        [[20, 21]],
        [[34]],
    ]
    assert [item.tolist() for item in inputs.attention_mask_list] == [
        [[1, 1]],
        [[1, 1]],
        [[1]],
    ]


def test_release_prefix_cache_lookup_attachments_deduplicates_handles():
    coordinator = _Coordinator(cached_tokens=[4, 4, 0], handles=[11, 11, 0])
    lookup = lookup_prefix_cache_for_prefill(
        coordinator=coordinator,
        namespace_digest=(1, 2, 3, 4),
        prompt_token_ids=[
            [10, 11, 12, 13],
            [10, 11, 12, 13],
            [20, 21],
        ],
    )

    release_prefix_cache_lookup_attachments(
        coordinator=coordinator,
        lookup=lookup,
    )

    assert coordinator.release_calls == [11]
