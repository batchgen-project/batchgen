"""Import + surface tests for batchgen.worker.protocols.

These tests do not exercise behavior (Protocols have none) — they verify the
module loads on a CPU-only env and every expected Protocol name is exported.
Later slices add behavior tests for the fake implementations.
"""

from __future__ import annotations

from typing import Protocol

from batchgen.worker import protocols


EXPECTED_PROTOCOLS = {
    "CollectiveBackend",
    "GpuKvBackend",
    "HostKvBackend",
    "TokenizerBackend",
    "ModelExecutorBackend",
    "LifespanLoggerBackend",
    "ClockBackend",
    "ResponseSinkBackend",
    "LegacyInfraBackend",
}

EXPECTED_ALIASES = {"UUID", "PageId", "AsyncHandle"}


class TestModuleSurface:
    def test_all_protocols_exported(self) -> None:
        missing = EXPECTED_PROTOCOLS - set(protocols.__all__)
        assert not missing, f"protocols.__all__ missing: {sorted(missing)}"

    def test_all_aliases_exported(self) -> None:
        missing = EXPECTED_ALIASES - set(protocols.__all__)
        assert not missing, f"protocols.__all__ missing: {sorted(missing)}"

    def test_no_unexpected_exports(self) -> None:
        unexpected = set(protocols.__all__) - (EXPECTED_PROTOCOLS | EXPECTED_ALIASES)
        assert not unexpected, f"unexpected exports: {sorted(unexpected)}"


class TestProtocolShape:
    def test_every_protocol_is_a_protocol_subclass(self) -> None:
        for name in EXPECTED_PROTOCOLS:
            cls = getattr(protocols, name)
            assert issubclass(cls, Protocol), f"{name} is not a Protocol"

    def test_collective_backend_has_rank_attrs(self) -> None:
        hints = protocols.CollectiveBackend.__annotations__
        assert "rank" in hints
        assert "world_size" in hints

    def test_collective_backend_methods(self) -> None:
        expected = {
            "all_reduce_max",
            "all_reduce_sum",
            "all_gather_tensor",
            "all_gather_into_tensor",
            "all_gather_object",
            "broadcast_tensor",
            "broadcast_object",
            "barrier",
        }
        actual = {
            name
            for name in dir(protocols.CollectiveBackend)
            if not name.startswith("_") and callable(getattr(protocols.CollectiveBackend, name, None))
        }
        assert expected <= actual, f"CollectiveBackend missing: {expected - actual}"

    def test_gpu_kv_backend_methods(self) -> None:
        expected = {
            "allocate_pages",
            "release_pages",
            "extend_pages",
            "append_kv",
            "free_pages",
            "rebuild_page_table",
        }
        actual = {
            name
            for name in dir(protocols.GpuKvBackend)
            if not name.startswith("_") and callable(getattr(protocols.GpuKvBackend, name, None))
        }
        assert expected <= actual, f"GpuKvBackend missing: {expected - actual}"

    def test_host_kv_backend_methods(self) -> None:
        expected = {
            "allocate_pages",
            "release_pages",
            "load_to_gpu_async",
            "free_pages",
        }
        actual = {
            name
            for name in dir(protocols.HostKvBackend)
            if not name.startswith("_") and callable(getattr(protocols.HostKvBackend, name, None))
        }
        assert expected <= actual, f"HostKvBackend missing: {expected - actual}"


class TestTypeAliases:
    def test_uuid_is_str(self) -> None:
        assert protocols.UUID is str

    def test_page_id_is_int(self) -> None:
        assert protocols.PageId is int
