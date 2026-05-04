"""Context managers for exact full-prefix-hit prefill."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Iterator, List

import torch


@contextmanager
def full_hit_attention_state(
    *,
    wrapper_classes: Iterable[type],
    cu_seqlens: torch.Tensor,
    position_ids: torch.Tensor,
    global_sequence_ids: List[int],
    prompt_lengths: List[int],
) -> Iterator[None]:
    """Temporarily configure attention wrappers for full-hit prefix replay."""
    wrapper_classes = tuple(wrapper_classes)
    for wrapper_cls in wrapper_classes:
        wrapper_cls.prepack_mode = True
        wrapper_cls.prepack_cu_seqlens = cu_seqlens
        wrapper_cls.prepack_max_seqlen = 1
        wrapper_cls.prepack_num_sequences = len(global_sequence_ids)
        wrapper_cls.prepack_seq_lengths = [1] * len(global_sequence_ids)
        wrapper_cls.position_ids = position_ids
        wrapper_cls.cur_batch = global_sequence_ids
        wrapper_cls.prepack_prefix_reuse_mode = False
        wrapper_cls.prepack_prefix_shared_tokens = prompt_lengths
        wrapper_cls.prepack_full_seq_lengths = prompt_lengths
        wrapper_cls.prepack_full_hit_mode = True
    try:
        yield
    finally:
        for wrapper_cls in wrapper_classes:
            wrapper_cls.prepack_mode = False
            wrapper_cls.prepack_cu_seqlens = None
            wrapper_cls.prepack_max_seqlen = None
            wrapper_cls.prepack_num_sequences = None
            wrapper_cls.prepack_seq_lengths = None
            wrapper_cls.prepack_prefix_reuse_mode = False
            wrapper_cls.prepack_prefix_shared_tokens = None
            wrapper_cls.prepack_full_seq_lengths = None
            wrapper_cls.prepack_full_hit_mode = False
