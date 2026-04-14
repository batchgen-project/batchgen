"""Smoke tests for the LegacyInfraBackend Protocol + FakeLegacyBackend.

Phase F1: these tests verify that (a) the Protocol is importable and
declares the expected surface, and (b) the FakeLegacyBackend records
calls correctly — the minimum needed before F2–F10 start wiring the
adapter into the native handlers.
"""

from __future__ import annotations

import pytest

from batchgen.worker.protocols import LegacyInfraBackend
from tests.unit.worker.fakes import FakeLegacyBackend


EXPECTED_METHODS = {
    # model lifecycle
    "configure_prefill_model",
    "configure_decode_model",
    "deep_free_model_memory",
    "init_nvshmem",
    "set_phase",
    "destroy_gpu_paged_kv_cache",
    # KV primitives
    "release_gpu_kv_pages",
    "release_host_kv_pages_for_batch",
    "extend_gpu_kv_allocation",
    "allocate_gpu_kv_two_page_buffer",
    "flush_deferred_kv_to_host",
    "wait_pending_kv_append_tasks",
    "rebuild_page_table_for_batch",
    "finalize_async_load_minimal",
    "check_host_kv_watermark_trigger",
    "get_effective_chunk_size",
    "put_sequences_on_hold",
    # index / UUID mapping
    "local_indices_to_global_seq_ids",
    "get_local_indices_for_uuids",
    "uuid_to_local_map",
    "local_to_uuid_map",
    "sequences_with_gpu_kv",
    # sampling / IO
    "select_tokens",
    "should_stop_at_eos",
    "rebuild_input_tokens",
    "decode_tokens_to_string",
    "report_completion",
    "gather_completed_tokens",
    "submit_completed_to_incremental_writer",
    # admission / tokenization
    "poll_admission_queue_nowait",
    "admit_sequences_from_message",
    "tokenize_admitted_sequences",
    "assign_admitted_sequences_to_ranks",
    "build_local_query_book_for_admitted",
    "update_max_input_length",
    # sequence-batch
    "is_sequence_completed",
    "update_batch_status",
    # watchdog
    "feed_watchdog",
    "enable_decode_watchdog",
    "disable_decode_watchdog",
    "feed_decode_watchdog",
    # dist init
    "ensure_comms",
    "init_gpu_kv_with_actual_size",
    # boundary Stage 1 passthroughs
    "set_num_tokens_per_rank",
    "set_rank_token_counts",
    "host_paged_kv_worker_view",
    "report_chunk_sizer_completion",
    # decode Stage 2 passthroughs
    "bind_decode_context",
    "forward_decode_step",
    "record_decoded_token",
    "check_repeating_ngram_pattern",
    "unbind_decode_context",
    "wait_async_load_task",
    "reset_pending_kv_append_tasks",
}


class TestLegacyInfraBackendProtocol:
    def test_protocol_declares_expected_methods(self) -> None:
        # Every expected method must be present as an annotation on the Protocol
        missing: list[str] = []
        for name in EXPECTED_METHODS:
            if not hasattr(LegacyInfraBackend, name):
                missing.append(name)
        assert not missing, f"Protocol missing methods: {sorted(missing)}"

    def test_protocol_has_rank_attrs(self) -> None:
        hints = LegacyInfraBackend.__annotations__
        assert "rank" in hints
        assert "local_rank" in hints
        assert "world_size" in hints


class TestFakeLegacyBackend:
    def test_fake_satisfies_protocol_surface(self) -> None:
        fake = FakeLegacyBackend()
        # Every Protocol method must exist on the fake (structural check)
        missing = [name for name in EXPECTED_METHODS if not hasattr(fake, name)]
        assert not missing, f"Fake missing methods: {sorted(missing)}"

    def test_fake_records_calls_in_order(self) -> None:
        fake = FakeLegacyBackend(rank=2, local_rank=2, world_size=4)
        fake.deep_free_model_memory()
        fake.set_phase("prefill")
        fake.release_gpu_kv_pages([1, 2, 3])
        names = [call[0] for call in fake.calls]
        assert names == [
            "deep_free_model_memory",
            "set_phase",
            "release_gpu_kv_pages",
        ]
        assert fake.calls[1][1] == ("prefill",)
        assert fake.calls[2][1] == ([1, 2, 3],)

    def test_fake_rank_attrs(self) -> None:
        fake = FakeLegacyBackend(rank=3, local_rank=1, world_size=8)
        assert fake.rank == 3
        assert fake.local_rank == 1
        assert fake.world_size == 8

    def test_fake_poll_admission_queue_raises_empty_when_no_messages(self) -> None:
        import queue as _queue
        fake = FakeLegacyBackend()
        with pytest.raises(_queue.Empty):
            fake.poll_admission_queue_nowait()

    def test_fake_poll_admission_queue_returns_preset_messages(self) -> None:
        fake = FakeLegacyBackend()
        fake._admission_messages = [{"type": "admit", "entries": []}]
        msg = fake.poll_admission_queue_nowait()
        assert msg == {"type": "admit", "entries": []}

    def test_fake_index_maps_are_mutable(self) -> None:
        fake = FakeLegacyBackend()
        fake._uuid_to_local["u1"] = 7
        assert fake.get_local_indices_for_uuids(["u1", "unknown"]) == [7]

    def test_fake_configure_prefill_model_returns_tuple(self) -> None:
        fake = FakeLegacyBackend()
        model, task = fake.configure_prefill_model()
        assert model == "fake_prefill_model"
        assert task is None
