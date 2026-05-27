"""BatchGenWorker-facing helpers for publishing Host KV pages."""

from __future__ import annotations

from typing import Mapping

from batchgen.prefix_reuse.commit import (
    aligned_prefix_tokens,
    build_committable_prefix_token_ids,
    build_prefix_commit_request,
    collect_required_group_pages_for_commit,
)
from batchgen.prefix_reuse.config import PrefixCacheRuntimeConfig
from batchgen.sequence import SequenceEntry


def sequence_token_ids_for_prefix_commit(
    seq: SequenceEntry,
    *,
    include_new_decode_tokens: bool,
    max_tokens: int,
) -> list[int]:
    """Return token ids matching the logical Host KV prefix for a sequence."""

    prompt_token_count = int(seq.prompt_length)
    prompt_tensor = seq.input_ids.reshape(-1)
    prompt_token_ids = [
        int(token_id)
        for token_id in prompt_tensor[:prompt_token_count].tolist()
    ]

    decoded_token_ids: list[int] = []
    decoded_start = 0
    if include_new_decode_tokens:
        decoded_start = int(seq.reentry_decoded_baseline)
        decoded_length = int(seq.decoded_length)
        decoded_tensor = seq.decoded_tokens
        if decoded_tensor is not None and decoded_length > 0:
            decoded_token_ids = [
                int(token_id)
                for token_id in decoded_tensor.reshape(-1)[
                    :decoded_length
                ].tolist()
            ]

    return build_committable_prefix_token_ids(
        prompt_token_ids=prompt_token_ids,
        decoded_token_ids=decoded_token_ids,
        decoded_start=decoded_start,
        max_tokens=max_tokens,
    )


def build_sequence_prefix_commit_request(
    *,
    core_engine_module: object,
    runtime_config: PrefixCacheRuntimeConfig,
    worker_views_by_group: Mapping[int, object],
    seq: SequenceEntry,
    include_new_decode_tokens: bool,
) -> tuple[object, int] | None:
    """Build a prefix-cache commit request for one worker-owned sequence."""

    decoded_start = int(seq.reentry_decoded_baseline)
    decoded_length = int(seq.decoded_length)
    new_decode_tokens = (
        max(0, decoded_length - decoded_start)
        if include_new_decode_tokens
        else 0
    )
    total_tokens = int(seq.prompt_length) + new_decode_tokens
    commit_tokens = aligned_prefix_tokens(
        total_tokens,
        int(runtime_config.publish_boundary_tokens),
    )
    if commit_tokens <= 0:
        return None

    shared_tokens = int(seq.prefix_shared_tokens)
    if commit_tokens <= shared_tokens:
        return None

    token_ids = sequence_token_ids_for_prefix_commit(
        seq,
        include_new_decode_tokens=include_new_decode_tokens,
        max_tokens=commit_tokens,
    )
    if len(token_ids) < commit_tokens:
        raise RuntimeError(
            "prefix cache commit has fewer token ids than committed tokens: "
            f"got {len(token_ids)}, expected {commit_tokens}"
        )

    pages_by_group = collect_required_group_pages_for_commit(
        worker_views_by_group=worker_views_by_group,
        sequence_id=int(seq.global_idx),
        commit_tokens=commit_tokens,
        group_specs=runtime_config.group_specs,
    )
    request = build_prefix_commit_request(
        core_engine_module=core_engine_module,
        namespace_digest=runtime_config.namespace_digest,
        token_ids=token_ids,
        publish_boundary_tokens=int(runtime_config.publish_boundary_tokens),
        pages_by_group=pages_by_group,
    )
    if request is None:
        return None
    return request, commit_tokens
