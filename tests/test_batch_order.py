from batchgen.batch_order import (
    batch_matches_expected_uuid_order,
    build_prefill_sequence_spans,
    local_indices_to_uuid_order,
    prefill_sequence_spans_to_cu_seqlens,
    prefill_sequence_spans_to_global_seq_ids,
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


def test_build_prefill_sequence_spans_preserves_exact_order():
    spans = build_prefill_sequence_spans(
        [4, 1],
        [1138, 1439],
        {
            4: "seq-a",
            1: "seq-b",
        },
        {
            4: 20,
            1: 3,
        },
    )

    assert [
        (span.local_idx, span.uuid, span.global_seq_id, span.seq_len, span.start, span.end)
        for span in spans
    ] == [
        (4, "seq-a", 20, 1138, 0, 1138),
        (1, "seq-b", 3, 1439, 1138, 2577),
    ]
    assert prefill_sequence_spans_to_cu_seqlens(spans) == [0, 1138, 2577]
    assert prefill_sequence_spans_to_global_seq_ids(spans) == [20, 3]


def test_build_prefill_sequence_spans_requires_length_alignment():
    try:
        build_prefill_sequence_spans(
            [4, 1],
            [1138],
            {4: "seq-a", 1: "seq-b"},
            {4: 20, 1: 3},
        )
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("Expected ValueError for misaligned local_indices/seq_lengths")


def test_build_prefill_sequence_spans_requires_local_identity_mappings():
    try:
        build_prefill_sequence_spans(
            [4, 1],
            [1138, 1439],
            {4: "seq-a"},
            {4: 20, 1: 3},
        )
    except KeyError as exc:
        assert "local_idx=1" in str(exc)
    else:
        raise AssertionError("Expected KeyError for missing local UUID mapping")

    try:
        build_prefill_sequence_spans(
            [4, 1],
            [1138, 1439],
            {4: "seq-a", 1: "seq-b"},
            {4: 20},
        )
    except KeyError as exc:
        assert "local_idx=1" in str(exc)
    else:
        raise AssertionError("Expected KeyError for missing local global-seq mapping")
