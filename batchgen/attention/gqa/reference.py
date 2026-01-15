"""Reference GQA implementation with attention sinks.

This is a direct adaptation of OpenAI's attention_ref() from gpt-oss for testing.
Used as ground truth for verifying flash-attention + sink correction approach.

Original source: gpt-oss/gpt_oss/triton/attention.py
"""

import torch
from typing import Optional


def attention_ref(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: torch.Tensor,
    sm_scale: float = 0.125,
    sliding_window: Optional[int] = None,
    start_q: int = 0,
) -> torch.Tensor:
    """Reference GQA implementation with attention sinks.

    This is the ground truth implementation for testing. It computes attention
    with the exact same math as OpenAI's implementation.

    Args:
        query: Query tensor [batch, num_queries, num_kv_heads, num_kv_groups, head_dim]
        key: Key tensor [batch, num_keys, num_kv_heads, head_dim]
        value: Value tensor [batch, num_keys, num_kv_heads, head_dim]
        sinks: Per-head sink values [num_heads] where num_heads = num_kv_heads * num_kv_groups
        sm_scale: Softmax scale factor (typically 1/sqrt(head_dim))
        sliding_window: If set, use sliding window attention with this bandwidth
        start_q: Position offset for the first query (for decode with KV cache)

    Returns:
        Output tensor [batch, num_queries, num_heads * head_dim]
    """
    batch_size, num_queries, num_kv_heads, num_kv_groups, head_dim = query.shape
    _, num_keys, _, _ = key.shape

    # Reshape sinks for broadcasting: [1, num_kv_heads, num_kv_groups, 1, 1]
    sinks = sinks.view(1, num_kv_heads, num_kv_groups, 1, 1).float()

    # Expand K/V for grouped attention: add group dimension
    key = key.unsqueeze(3)  # [batch, num_keys, num_kv_heads, 1, head_dim]
    value = value.unsqueeze(3)  # [batch, num_keys, num_kv_heads, 1, head_dim]

    # Build causal mask
    pos_keys = torch.arange(num_keys, device=query.device)
    pos_queries = torch.arange(num_queries, device=query.device) + start_q
    mask = pos_keys[None, :] > pos_queries[:, None]  # [num_queries, num_keys]
    mask = mask.float().masked_fill(mask, float("-inf"))

    # Apply sliding window mask if specified
    if sliding_window:
        too_old = pos_keys[None, :] < (pos_queries[:, None] - sliding_window + 1)
        mask.masked_fill_(too_old, float("-inf"))

    # Compute attention logits: [batch, num_kv_heads, num_kv_groups, num_queries, num_keys]
    logits = torch.einsum("bqhmd,bkhmd->bhmqk", query.float(), key.float()) * sm_scale
    logits = logits + mask[None, None, None, :, :]

    # Numerically stable softmax with sinks
    logits_max = torch.max(logits, dim=-1, keepdim=True).values
    logits_or_sinks_max = torch.maximum(sinks, logits_max)
    sinks_exp = torch.exp(sinks - logits_or_sinks_max)
    unnormalized_scores = torch.exp(logits - logits_or_sinks_max)
    normalizer = unnormalized_scores.sum(dim=-1, keepdim=True) + sinks_exp
    scores = unnormalized_scores / normalizer

    # Compute output: [batch, num_queries, num_kv_heads, num_kv_groups, head_dim]
    output = torch.einsum("bhmqk,bkhmd->bqhmd", scores, value.float())

    # Reshape to [batch, num_queries, num_heads * head_dim]
    num_heads = num_kv_heads * num_kv_groups
    output = output.reshape(batch_size, num_queries, num_heads * head_dim)

    return output.to(query.dtype)


def attention_ref_no_sinks(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sm_scale: float = 0.125,
    sliding_window: Optional[int] = None,
    start_q: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference GQA implementation without sinks, returning LSE for testing.

    This version returns the log-sum-exp values so we can verify the sink
    correction math separately.

    Args:
        query: Query tensor [batch, num_queries, num_kv_heads, num_kv_groups, head_dim]
        key: Key tensor [batch, num_keys, num_kv_heads, head_dim]
        value: Value tensor [batch, num_keys, num_kv_heads, head_dim]
        sm_scale: Softmax scale factor
        sliding_window: Sliding window bandwidth
        start_q: Position offset for queries

    Returns:
        Tuple of:
            - Output tensor [batch, num_queries, num_heads * head_dim]
            - LSE tensor [batch, num_heads, num_queries]
    """
    batch_size, num_queries, num_kv_heads, num_kv_groups, head_dim = query.shape
    _, num_keys, _, _ = key.shape

    # Expand K/V for grouped attention
    key = key.unsqueeze(3)
    value = value.unsqueeze(3)

    # Build causal mask
    pos_keys = torch.arange(num_keys, device=query.device)
    pos_queries = torch.arange(num_queries, device=query.device) + start_q
    mask = pos_keys[None, :] > pos_queries[:, None]
    mask = mask.float().masked_fill(mask, float("-inf"))

    if sliding_window:
        too_old = pos_keys[None, :] < (pos_queries[:, None] - sliding_window + 1)
        mask.masked_fill_(too_old, float("-inf"))

    # Compute attention logits
    logits = torch.einsum("bqhmd,bkhmd->bhmqk", query.float(), key.float()) * sm_scale
    logits = logits + mask[None, None, None, :, :]

    # Standard softmax (no sinks)
    lse = torch.logsumexp(logits, dim=-1)  # [batch, num_kv_heads, num_kv_groups, num_queries]
    scores = torch.softmax(logits, dim=-1)

    # Compute output
    output = torch.einsum("bhmqk,bkhmd->bqhmd", scores, value.float())

    # Reshape output and LSE
    num_heads = num_kv_heads * num_kv_groups
    output = output.reshape(batch_size, num_queries, num_heads * head_dim)
    lse = lse.reshape(batch_size, num_heads, num_queries)

    return output.to(query.dtype), lse
