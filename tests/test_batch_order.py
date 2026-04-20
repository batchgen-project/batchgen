from batchgen.batch_order import (
    batch_matches_expected_uuid_order,
    local_indices_to_uuid_order,
)


def test_batch_matches_expected_uuid_order_requires_exact_order():
    uuid_to_local_map = {
        "seq-a": 4,
        "seq-b": 1,
    }

    assert batch_matches_expected_uuid_order(
        [4, 1],
        ["seq-a", "seq-b"],
        uuid_to_local_map,
    )
    assert not batch_matches_expected_uuid_order(
        [1, 4],
        ["seq-a", "seq-b"],
        uuid_to_local_map,
    )


def test_local_indices_to_uuid_order_preserves_missing_entries():
    local_to_uuid_map = {
        4: "seq-a",
        1: "seq-b",
    }

    assert local_indices_to_uuid_order([1, 4, 9], local_to_uuid_map) == [
        "seq-b",
        "seq-a",
        None,
    ]
