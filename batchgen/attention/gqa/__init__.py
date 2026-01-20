"""GQA (Grouped Query Attention) with attention sinks support.

This module provides GQA implementations using flash-attention with optional
attention sink correction. It supports both prefill and decode modes.

Key components:
- gqa_prefill_fa: Prefill using flash_attn_varlen_func (unpadded sequences)
- gqa_decode_fa: Decode using flash_attn_with_kvcache (paged KV cache)
- apply_sink_correction: Post-correction for attention sinks
- attention_ref: Reference implementation for testing

Attention sinks are learned per-head values that modify the softmax
normalization, effectively "stealing" attention mass from all keys.
This is implemented as a post-correction: output *= sigmoid(lse - sinks)
"""

from .fa_prefill import gqa_prefill_fa
from .fa_decode import gqa_decode_fa, gqa_decode_fa_contiguous
from .sink_correction import apply_sink_correction
from .reference import attention_ref, attention_ref_no_sinks
from .gqa_mode3 import gqa_decoding_mode_3_bf16

__all__ = [
    'gqa_prefill_fa',
    'gqa_decode_fa',
    'gqa_decode_fa_contiguous',
    'apply_sink_correction',
    'attention_ref',
    'attention_ref_no_sinks',
    'gqa_decoding_mode_3_bf16',
]
