"""Structural guards for the worker's host-KV prefill transaction.

These tests do not construct a GPU worker.  They pin the lifecycle boundary:
the fresh host-capacity transaction and allocator consensus must complete
before IN_PREFILL or model/re-entry configuration can mutate worker state.
"""

from __future__ import annotations

from pathlib import Path


_WORKER = Path(__file__).parents[2] / "batchgen" / "batchgen_worker.py"


def _source() -> str:
    return _WORKER.read_text()


def test_prefill_reservation_precedes_status_and_phase_config():
    source = _source()
    loop = source[source.index("prefill_uuids = self._prepare_prefill_batch()") :]
    reservation = loop.index(
        "if prefill_uuids and not self._reserve_prefill_host_kv(prefill_uuids):"
    )
    status = loop.index(
        "self._update_batch_status(prefill_uuids, SequenceStatus.IN_PREFILL)"
    )
    config = loop.index("self._config_prefill_for_batch(prefill_uuids)")
    assert reservation < status < config


def test_config_prefill_cannot_allocate_host_pages_after_status_transition():
    source = _source()
    config = source[source.index("def _config_prefill_for_batch") :]
    config = config[: config.index("def _load_decode_model")]
    assert "allocate_pages_for_sequences" not in config
    assert "register_sequences" not in config


def test_prefill_transaction_has_collective_snapshot_and_allocator_consensus():
    source = _source()
    transaction = source[source.index("def _reserve_prefill_host_kv") :]
    transaction = transaction[: transaction.index("def _prepare_prefill_batch")]
    assert "_collective_prefill_host_capacity_ok" in transaction
    assert "dist.all_reduce(allocation, op=dist.ReduceOp.MIN)" in transaction
    assert "release_sequence_pages" in transaction
    assert "unregister_sequences" in transaction
