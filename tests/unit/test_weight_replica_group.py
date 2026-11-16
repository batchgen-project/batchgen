import pytest

from batchgen.weight_replica_group import (
    RankRecord,
    compute_replica_groups,
    ordered_replica_group,
)


def _record(rank, *, fingerprint="same", source="local_memfd", row=None):
    return RankRecord(
        node_identity="node-a",
        device_index=rank,
        peer_access_row=[True, True] if row is None else row,
        list_fingerprint=fingerprint,
        weight_source_kind=source,
        rank=rank,
    )


def test_same_node_same_fingerprint_full_access_forms_ordered_group():
    records = [_record(0), _record(1)]

    assert compute_replica_groups(records) == {0: (0, 1), 1: (0, 1)}
    assert ordered_replica_group(records, 1) == (0, 1)


def test_fingerprint_and_node_split_groups():
    records = [
        _record(0),
        _record(1, fingerprint="other"),
        RankRecord("node-b", 2, [True, True, True], "same", "local_memfd", 2),
    ]

    assert compute_replica_groups(records) == {
        0: (0,),
        1: (1,),
        2: (2,),
    }


def test_partial_peer_access_fails_with_matrix():
    records = [_record(0, row=[True, True]), _record(1, row=[False, True])]

    with pytest.raises(ValueError, match=r"partial peer access.*matrix") as error:
        compute_replica_groups(records)
    assert "0: 1 1" in str(error.value)
    assert "1: 0 1" in str(error.value)


def test_daemon_backed_source_declares_every_reader_owner():
    records = [
        _record(0, source="daemon", row=[True, False]),
        _record(1, source="daemon", row=[False, True]),
    ]

    assert compute_replica_groups(records) == {0: (0,), 1: (1,)}


def test_mapping_records_infer_rank_from_mapping_key():
    records = {
        4: {
            "node_identity": "node-a",
            "device_index": 0,
            "peer_access_row": {4: True},
            "list_fingerprint": "fp",
            "weight_source_kind": "local_memfd",
        }
    }

    assert compute_replica_groups(records) == {4: (4,)}

