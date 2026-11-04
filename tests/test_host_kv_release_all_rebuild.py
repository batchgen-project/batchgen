"""ReleaseAllSequences heals the host-KV allocator after a killed writer.

A writer SIGKILLed between the free-stack commit and the sequence-metadata
update (or, on release, between unlinking pages and pushing them back) leaves
pages that are neither free nor reachable from any sequence chain. The restart
reconcile calls ReleaseAllSequences once every worker has exited, and that
call rebuilds the free state from page ownership. Invariant under test: after
ReleaseAllSequences the allocator state is identical to a fresh region plus
the prefix-resident set.

The native tests need the compiled extension on Linux. They reproduce each
kill window by writing the region's metadata directly through the memfd, the
way a writer that died mid-update would have left it. The metadata layout is
located by the fresh-region pattern rather than assumed, and the tests fail
loudly if it cannot be found.
"""

from __future__ import annotations

import mmap
import os
import struct
import sys
import uuid
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "core" / "KV_Storage" / "host_paged_kv_backend.cpp"

NUM_PAGES = 16
PAGE_TOKENS = 4
SEQUENCE_CAPACITY = 8
EMPTY = -(2**63)
TOMBSTONE = EMPTY + 1
RESIDENT = EMPTY + 2
FREE_STACK_TOP_OFFSET = 84
ACTIVE_SEQUENCES_OFFSET = 88
# sequences_inserted, sequences_released, acquire_failures (uint64 each),
# pinned by static_asserts in the backend.
LIFETIME_COUNTERS_OFFSET = 96
SEQUENCE_ENTRY = struct.Struct("<qIii4x")


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def test_rebuild_runs_under_both_locks_at_the_end_of_release_all():
    source = BACKEND.read_text()
    body = source.split(
        "void HostPagedKVBackend::SharedState::ReleaseAllSequences()", 1
    )[1].split("\n}\n", 1)[0]
    release_loop = body.index("ReleaseSequence(sequence_id)")
    allocation = body.index("ScopedMutexLock allocation_lock(&header->allocation_mutex)")
    sequence = body.index("ScopedMutexLock sequence_lock(&header->sequence_mutex)")
    rebuild = body.index("RebuildFreeStateLocked();")
    assert release_loop < allocation < sequence < rebuild
    rebuild_body = source.split(
        "void HostPagedKVBackend::SharedState::RebuildFreeStateLocked()", 1
    )[1].split("\n}\n", 1)[0]
    assert "kPrefixResidentSequenceId" in rebuild_body
    assert "memset" not in rebuild_body
    assert (
        "after ReleaseAllSequences the allocator state is identical to a\n"
        "// fresh region plus the prefix-resident set" in source
    )


def _core_engine():
    try:
        from batchgen.native_loader import core_engine
    except Exception as exc:
        pytest.skip(f"compiled extension unavailable: {exc}")
    return core_engine


class _Region:
    """Read/write view of one manager's allocator metadata."""

    def __init__(self, manager) -> None:
        fd = os.open(f"/proc/self/fd/{manager.memfd_fd()}", os.O_RDWR)
        try:
            self._mm = mmap.mmap(fd, 2 * mmap.PAGESIZE, mmap.MAP_SHARED)
        finally:
            os.close(fd)
        self.free_stack = self._locate_free_stack()
        self.owners = _align(self.free_stack + 4 * NUM_PAGES, 8)
        self.links = self.owners + 8 * NUM_PAGES
        self.table = _align(self.links + 4 * NUM_PAGES, 8)
        if (self.top, self.active) != (NUM_PAGES, 0):
            raise AssertionError(
                "free_stack_top/active_sequences not at their SharedHeader "
                "offsets in a fresh region; this test must follow the layout"
            )

    def _locate_free_stack(self) -> int:
        fresh = struct.pack(f"<{NUM_PAGES}i", *range(NUM_PAGES - 1, -1, -1))
        # The free-stack array is 8-byte aligned and sits after the counters
        # AND the two pthread mutexes, so its offset is not derivable from
        # ACTIVE_SEQUENCES_OFFSET alone (measured: 176 on glibc x86_64, which
        # is ≡ 0 mod 8 — an 8-stride scan starting at 92 ≡ 4 mod 8 never saw
        # it).  Scan every 4-byte offset instead; the content checks below
        # make a false match implausible.
        for offset in range(ACTIVE_SEQUENCES_OFFSET + 4, 2048, 4):
            if self._mm[offset:offset + 4 * NUM_PAGES] != fresh:
                continue
            owners = _align(offset + 4 * NUM_PAGES, 8)
            links = owners + 8 * NUM_PAGES
            if (
                struct.unpack_from(f"<{NUM_PAGES}q", self._mm, owners)
                == (EMPTY,) * NUM_PAGES
                and struct.unpack_from(f"<{NUM_PAGES}i", self._mm, links)
                == (-1,) * NUM_PAGES
            ):
                return offset
        raise AssertionError(
            "host-KV metadata layout not found in a fresh region; the "
            "SharedHeader layout changed and this test must follow it"
        )

    def u32(self, offset: int) -> int:
        return struct.unpack_from("<I", self._mm, offset)[0]

    def set_u32(self, offset: int, value: int) -> None:
        struct.pack_into("<I", self._mm, offset, value)

    @property
    def top(self) -> int:
        return self.u32(FREE_STACK_TOP_OFFSET)

    @top.setter
    def top(self, value: int) -> None:
        self.set_u32(FREE_STACK_TOP_OFFSET, value)

    @property
    def active(self) -> int:
        return self.u32(ACTIVE_SEQUENCES_OFFSET)

    @active.setter
    def active(self, value: int) -> None:
        self.set_u32(ACTIVE_SEQUENCES_OFFSET, value)

    def stack(self) -> list[int]:
        return list(struct.unpack_from(f"<{self.top}i", self._mm, self.free_stack))

    def owner(self, page: int) -> int:
        return struct.unpack_from("<q", self._mm, self.owners + 8 * page)[0]

    def set_owner(self, page: int, value: int) -> None:
        struct.pack_into("<q", self._mm, self.owners + 8 * page, value)

    def link(self, page: int) -> int:
        return struct.unpack_from("<i", self._mm, self.links + 4 * page)[0]

    def set_link(self, page: int, value: int) -> None:
        struct.pack_into("<i", self._mm, self.links + 4 * page, value)

    def allocator_state(self) -> tuple:
        return (
            self.top,
            self.active,
            bytes(self._mm[self.free_stack:self.free_stack + 4 * NUM_PAGES]),
            bytes(self._mm[self.owners:self.owners + 8 * NUM_PAGES]),
            bytes(self._mm[self.links:self.links + 4 * NUM_PAGES]),
            # Field-wise: struct padding is not allocator state.
            tuple(
                SEQUENCE_ENTRY.unpack_from(
                    self._mm, self.table + SEQUENCE_ENTRY.size * i
                )
                for i in range(SEQUENCE_CAPACITY)
            ),
        )

    def close(self) -> None:
        self._mm.close()


def _manager(core_engine):
    config = core_engine.HostPagedKVConfig()
    config.shm_name = f"batchgen_host_kv_{uuid.uuid4().hex}"
    config.num_layers = 1
    config.num_pages = NUM_PAGES
    config.page_size_tokens = PAGE_TOKENS
    config.num_k_heads = 1
    config.k_head_dim = 8
    config.num_v_heads = 0
    config.k_element_size_bytes = 2
    config.sequence_table_capacity = SEQUENCE_CAPACITY
    config.alignment_bytes = 64
    manager = core_engine.MLAHostPagedKVManager(config)
    manager.initialize(True)
    return manager


native = pytest.mark.skipif(
    sys.platform != "linux", reason="native host KV check requires Linux"
)


@pytest.fixture
def regions():
    core_engine = _core_engine()
    made = []

    def make():
        manager = _manager(core_engine)
        region = _Region(manager)
        made.append((manager, region))
        return manager, region

    yield make
    for _, region in made:
        region.close()


def _fresh_state(regions) -> tuple:
    _, fresh = regions()
    return fresh.allocator_state()


@native
def test_release_all_restores_the_fresh_allocator_state(regions):
    expected = _fresh_state(regions)
    manager, region = regions()
    manager.allocate_pages(1, 3 * PAGE_TOKENS)
    manager.allocate_pages(2, 2 * PAGE_TOKENS)
    manager.free_sequence(2)
    manager.allocate_pages(3, 1)

    manager.release_all_sequences()

    assert region.allocator_state() == expected
    stats = manager.get_stats()
    assert stats.num_free_pages == NUM_PAGES
    assert stats.num_active_sequences == 0


@native
def test_pages_stranded_inside_an_allocation_are_recovered(regions):
    """Writer killed after committing free_stack_top, before any metadata."""
    expected = _fresh_state(regions)
    manager, region = regions()
    manager.allocate_pages(1, 2 * PAGE_TOKENS)
    stranded = region.stack()[-3:]
    region.top = region.top - 3
    for page in stranded[:2]:
        # It died after writing the first owners of the batch.
        region.set_owner(page, 99)

    manager.release_all_sequences()

    assert region.allocator_state() == expected
    assert manager.get_stats().num_free_pages == NUM_PAGES


@native
def test_pages_stranded_inside_a_release_are_recovered(regions):
    """Release killed mid-way: the chain is cut after its first page."""
    expected = _fresh_state(regions)
    manager, region = regions()
    pages = list(manager.allocate_pages(1, 3 * PAGE_TOKENS))
    assert len(pages) == 3
    region.set_link(pages[0], -1)
    region.set_owner(pages[1], EMPTY)

    manager.release_all_sequences()

    assert region.allocator_state() == expected
    assert manager.get_stats().num_free_pages == NUM_PAGES


@native
def test_prefix_resident_pages_survive_the_rebuild(regions):
    manager, region = regions()
    taken = region.stack()[-2:]
    resident, stranded = taken[-1], taken[0]
    region.top = region.top - 2
    region.set_owner(resident, RESIDENT)
    region.set_link(resident, -1)
    region.set_owner(stranded, 7)
    manager.allocate_pages(1, 2 * PAGE_TOKENS)

    manager.release_all_sequences()

    stack = region.stack()
    assert region.top == NUM_PAGES - 1
    assert resident not in stack
    assert stranded in stack
    assert sorted(stack) == sorted(set(range(NUM_PAGES)) - {resident})
    assert region.owner(resident) == RESIDENT
    assert region.link(resident) == -1
    assert manager.get_stats().num_free_pages == NUM_PAGES - 1


@native
def test_a_wrapped_active_sequence_counter_is_healed(regions):
    """A writer killed between inserting a sequence and counting it lets the
    release's fetch_sub wrap the counter; the next call must still work."""
    manager, region = regions()
    manager.allocate_pages(1, PAGE_TOKENS)
    region.active = 0

    manager.release_all_sequences()
    assert region.active == 0

    region.active = 0xFFFFFFFF
    manager.release_all_sequences()
    assert region.active == 0
    assert manager.get_stats().num_free_pages == NUM_PAGES


@native
def test_audit_reports_stranded_pages_until_release_all_heals_them(regions):
    """The page identity is the leak detector: pages taken off the free stack
    and attached to nothing show up as a positive delta."""
    manager, region = regions()
    manager.allocate_pages(1, 2 * PAGE_TOKENS)
    assert manager.get_audited_stats().audit.page_identity_delta == 0

    region.top = region.top - 3  # writer killed after popping three pages
    audit = manager.get_audited_stats().audit
    assert audit.page_identity_delta == 3
    assert audit.live_sequence_pages == 2

    manager.release_all_sequences()
    audit = manager.get_audited_stats().audit
    assert audit.page_identity_delta == 0
    assert audit.live_entries == 0
    assert audit.empty_entries == SEQUENCE_CAPACITY


@native
def test_prefix_resident_pages_are_part_of_the_page_identity(regions):
    manager, region = regions()
    manager.allocate_pages(1, 2 * PAGE_TOKENS)
    resident = region.stack()[-1]
    region.top = region.top - 1
    region.set_owner(resident, RESIDENT)
    region.set_link(resident, -1)

    stats = manager.get_audited_stats()
    assert stats.num_used_pages == 3
    assert stats.audit.live_sequence_pages == 2
    assert stats.audit.prefix_resident_pages == 1
    assert stats.audit.page_identity_delta == 0


@native
def test_lifetime_counters_live_in_the_shared_header(regions):
    """They describe the region, not one process: they are read back through an
    independent mapping and survive the release-all rebuild."""
    manager, region = regions()
    manager.allocate_pages(1, PAGE_TOKENS)
    manager.allocate_pages(2, PAGE_TOKENS)
    manager.free_sequence(2)
    with pytest.raises(RuntimeError, match="Insufficient free pages"):
        manager.allocate_pages(3, (NUM_PAGES + 1) * PAGE_TOKENS)

    stats = manager.get_stats()
    counters = (stats.sequences_inserted, stats.sequences_released,
                stats.acquire_failures)
    assert counters == (2, 1, 1)
    assert struct.unpack_from("<3Q", region._mm, LIFETIME_COUNTERS_OFFSET) == counters

    manager.release_all_sequences()
    stats = manager.get_stats()
    assert (stats.sequences_inserted, stats.sequences_released,
            stats.acquire_failures) == (2, 2, 1)
    assert struct.unpack_from("<3Q", region._mm, LIFETIME_COUNTERS_OFFSET) == (2, 2, 1)
