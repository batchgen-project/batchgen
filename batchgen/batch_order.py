from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class PrefillSequenceSpan:
    row_index: int
    local_idx: int
    uuid: str
    global_seq_id: int
    seq_len: int
    start: int
    end: int


def local_indices_to_uuid_order(
    local_indices: Sequence[int],
    local_to_uuid_map: Mapping[int, str],
) -> List[Optional[str]]:
    """Resolve batch-local indices to UUID order for diagnostics/order checks."""
    return [local_to_uuid_map.get(local_idx) for local_idx in local_indices]


def batch_matches_expected_uuid_order(
    local_indices: Sequence[int],
    expected_uuids: Sequence[str],
    uuid_to_local_map: Mapping[str, int],
) -> bool:
    """Return True only when local indices exactly match expected UUID order."""
    if len(local_indices) != len(expected_uuids):
        return False

    expected_local: List[int] = []
    for uuid in expected_uuids:
        local_idx = uuid_to_local_map.get(uuid)
        if local_idx is None:
            return False
        expected_local.append(local_idx)

    return list(local_indices) == expected_local


def build_prefill_sequence_spans(
    local_indices: Sequence[int],
    seq_lengths: Sequence[int],
    local_to_uuid_map: Mapping[int, str],
    local_to_global_seq_id_map: Mapping[int, int],
) -> List[PrefillSequenceSpan]:
    """Build per-sequence spans for a flattened prefill micro-batch."""
    if len(local_indices) != len(seq_lengths):
        raise ValueError(
            "local_indices and seq_lengths must have the same length, "
            f"got {len(local_indices)} and {len(seq_lengths)}"
        )

    spans: List[PrefillSequenceSpan] = []
    cursor = 0

    for row_index, (local_idx, seq_len) in enumerate(zip(local_indices, seq_lengths)):
        uuid = local_to_uuid_map.get(local_idx)
        if uuid is None:
            raise KeyError(f"Missing UUID for local_idx={local_idx}")

        global_seq_id = local_to_global_seq_id_map.get(local_idx)
        if global_seq_id is None:
            raise KeyError(
                f"Missing global sequence id for local_idx={local_idx} uuid={uuid}"
            )

        end = cursor + int(seq_len)
        spans.append(
            PrefillSequenceSpan(
                row_index=row_index,
                local_idx=local_idx,
                uuid=uuid,
                global_seq_id=global_seq_id,
                seq_len=int(seq_len),
                start=cursor,
                end=end,
            )
        )
        cursor = end

    return spans


def prefill_sequence_spans_to_cu_seqlens(
    spans: Sequence[PrefillSequenceSpan],
) -> List[int]:
    """Convert per-sequence spans to FA3-style cumulative sequence lengths."""
    return [0] + [span.end for span in spans]


def prefill_sequence_spans_to_global_seq_ids(
    spans: Sequence[PrefillSequenceSpan],
) -> List[int]:
    """Project per-sequence spans to the global sequence id order."""
    return [span.global_seq_id for span in spans]
