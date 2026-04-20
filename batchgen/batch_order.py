from typing import List, Mapping, Optional, Sequence


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
