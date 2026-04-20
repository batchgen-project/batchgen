from dataclasses import dataclass
from typing import Dict, MutableMapping, Optional, Set, Tuple

import torch


@dataclass
class QueryBookEntry:
    text: Optional[str] = None
    encoded: Optional[Dict[str, torch.Tensor]] = None
    decoded_tokens: Optional[torch.Tensor] = None
    kv_token_budget: Optional[int] = None


def make_query_book_entry(sequence) -> QueryBookEntry:
    return QueryBookEntry(
        text=sequence.text,
        encoded={"input_ids": sequence.input_ids},
        decoded_tokens=sequence.decoded_tokens,
        kv_token_budget=sequence.kv_token_budget,
    )


def bind_local_sequence_to_query_book(
    uuid: str,
    sequence,
    *,
    query_book: MutableMapping[int, QueryBookEntry],
    local_to_uuid_map: MutableMapping[int, str],
    uuid_to_local_map: MutableMapping[str, int],
    free_local_indices: Set[int],
    next_local_idx: int,
    local_idx: Optional[int] = None,
) -> Tuple[int, int]:
    if sequence is None:
        raise KeyError(f"Sequence with UUID {uuid} not found in global_batch")

    if local_idx is None:
        if free_local_indices:
            local_idx = free_local_indices.pop()
        else:
            local_idx = next_local_idx
            next_local_idx += 1

    local_to_uuid_map[local_idx] = uuid
    uuid_to_local_map[uuid] = local_idx
    query_book[local_idx] = make_query_book_entry(sequence)
    return local_idx, next_local_idx


def release_local_query_slot(
    uuid: str,
    *,
    uuid_to_local_map: MutableMapping[str, int],
    local_to_uuid_map: MutableMapping[int, str],
    query_book: MutableMapping[int, QueryBookEntry],
    free_local_indices: Set[int],
) -> Optional[int]:
    local_idx = uuid_to_local_map.pop(uuid, None)
    if local_idx is None:
        return None

    local_to_uuid_map.pop(local_idx, None)
    query_book.pop(local_idx, None)
    free_local_indices.add(local_idx)
    return local_idx
