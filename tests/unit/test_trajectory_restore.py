"""Contract tests for the trajectory lookup / attach / restore path (M3 2).

Pure Python, no GPU. The objects under test are REAL: the frontier registry and
record, the Slice 1b committer, the snapshot layout and profile, the arena
allocator, the arena reader, the snapshot reader/writer pair, and the live
``KDAStateGPUManager`` running on CPU. Only two things are stand-ins:

* ``_FakeHostView`` -- a byte-accurate model of one KV group's Host page
  region, so an arena really is written into pages and really is read back out
  of them. That is what makes the round-trip assertion meaningful;
* ``torch.device("cpu")`` for the state pool, so ``copy_`` into a strided
  ``[:, slot]`` view is the same code path with the H2D hop removed.

What these tests are defending:
* the binary hit predicate as the ADMISSION sees it -- continuation hits, a
  re-rendered prompt below the frontier misses in full, an unrelated prompt
  misses;
* the attachment really pins: the registry refuses to evict a referenced
  record by name, and accepts the eviction once the attachment is consumed;
* byte fidelity -- the restored KDA slot equals the committed slot chunk for
  chunk, including the case where the blob straddles a KV group boundary;
* every fallback releases its attachment and names its reason, so the live
  count returns to zero and the sequence proceeds as a full recompute;
* a failure between the arena read and the slot bind leaks neither a slot nor
  an attachment;
* the whole path is behind the SAME batch-level flag as the commit side.
"""

import ast
import pathlib
import sys
import types

import pytest
import torch


def _import_kda_state_pool():
    """The real state pool, stubbing Triton only on a CPU-only box.

    ``batchgen.kv_cache`` pulls in a Triton kernel module on import; the stub
    convention is ``tests/unit/test_kda_layer_map_isolation.py``'s, so CI keeps
    importing the real Triton when it is present.
    """

    try:
        from batchgen.kv_cache.kda_state_gpu_manager import (
            KDAStateGPUConfig,
            KDAStateGPUManager,
        )

        return KDAStateGPUConfig, KDAStateGPUManager
    except ModuleNotFoundError as exc:
        if exc.name != "triton":
            raise
    triton_stub = types.ModuleType("triton")
    triton_stub.jit = lambda fn: fn
    triton_stub.autotune = lambda *args, **kwargs: (lambda fn: fn)
    triton_stub.Config = lambda *args, **kwargs: None
    triton_stub.cdiv = lambda a, b: -(-a // b)
    language_stub = types.ModuleType("triton.language")
    language_stub.constexpr = object
    triton_stub.language = language_stub
    extra_stub = types.ModuleType("triton.language.extra")
    extra_stub.libdevice = types.ModuleType("triton.language.extra.libdevice")
    language_stub.extra = extra_stub
    sys.modules["triton"] = triton_stub
    sys.modules["triton.language"] = language_stub
    sys.modules["triton.language.extra"] = extra_stub
    sys.modules["triton.language.extra.libdevice"] = extra_stub.libdevice
    from batchgen.kv_cache.kda_state_gpu_manager import (
        KDAStateGPUConfig,
        KDAStateGPUManager,
    )

    return KDAStateGPUConfig, KDAStateGPUManager


KDAStateGPUConfig, KDAStateGPUManager = _import_kda_state_pool()

from batchgen.prefix_reuse.trajectory_commit import (  # noqa: E402
    DEBUG_FLAG as COMMIT_DEBUG_FLAG,
    SNAPSHOT_CHUNK_NAMES,
    HostKVSnapshotArenaAllocator,
    KDAStateSnapshotReader,
    SnapshotArenaCorrupt,
    SnapshotLayout,
    TrajectoryFrontierCommitter,
    TrajectorySnapshotShapeMismatch,
    profile_from_kda_state_config,
    trajectory_commit_enabled,
)
from batchgen.prefix_reuse.trajectory_frontier import (  # noqa: E402
    FrontierRefcountError,
)
from batchgen.prefix_reuse.trajectory_restore import (  # noqa: E402
    DEBUG_FLAG,
    DECLINE_ARENA_MISSING,
    DECLINE_ARENA_READ_FAILED,
    DECLINE_GROUP_MISMATCH,
    DECLINE_HEAD_SHARDED_GROUP,
    DECLINE_NOT_PLANNED,
    DECLINE_REASONS,
    DECLINE_SLOTS_EXHAUSTED,
    DECLINE_SLOT_WRITE_FAILED,
    DECLINE_SUFFIX_OFFSET,
    FAIL_NO_RESTORE_SOURCE,
    HostKVSnapshotArenaReader,
    KDAStateSlotsExhausted,
    KDAStateSnapshotWriter,
    TrajectoryAttachmentInvalid,
    TrajectoryPlannedRestoreFailed,
    TrajectoryRestoreError,
    TrajectoryRestorePlanDrift,
    TrajectoryRestorer,
    trajectory_reuse_enabled,
)


PAGE_TOKENS = 4
# Per-group Host page bytes. ``host_page_bytes_all_layers`` is the SUM across
# groups (prefix_reuse/config.py), so a one-group run uses this and a two-group
# run uses twice it -- which is how the blob comes to straddle a group.
GROUP_PAGE_BYTES = 256
NAMESPACE = (11, 22, 33, 44)
NUM_STATE_SLOTS = 3

# The frontier the tests commit: 8 tokens = two 4-token pages.
BASE_PROMPT = (101, 102, 103, 104, 105, 106, 107, 108)
FRONTIER_PAGES = {0: (31, 32)}


# ---------------------------------------------------------------------------
# A byte-accurate Host KV group
# ---------------------------------------------------------------------------


class _FakeHostView:
    """One KV group's Host page region, with real bytes in real pages.

    Models exactly the surface the arena allocator and the arena reader use.
    Page ownership is tracked the way the C++ view tracks it: a page is either
    free, owned by a sequence chain, or prefix-RESIDENT. ``attach_shared_prefix
    _pages`` never transfers ownership, and ``release_sequence_pages`` returns
    only chain-owned pages -- which is what makes the read row harmless.
    """

    def __init__(self, *, num_pages: int, page_bytes: int = GROUP_PAGE_BYTES):
        self.page_bytes = int(page_bytes)
        self._pages = {
            page_id: bytearray(self.page_bytes) for page_id in range(num_pages)
        }
        self._free = list(range(num_pages))
        self._chains: dict[int, list[int]] = {}
        self._attached: dict[int, list[int]] = {}
        # The rank-LOCAL page-table row, which the C++ view keeps separately
        # from the node-shared allocator chain. ``set_sequence_pages`` writes
        # only this, which is what lets a non-owner rank resolve pages it did
        # not allocate (and what lets a shard write address one window).
        self._rows: dict[int, list[int]] = {}
        self._resident: set[int] = set()
        self.calls: list[tuple] = []

    # -- registration ------------------------------------------------------

    def register_sequences(self, sequence_ids):
        for sequence_id in sequence_ids:
            self.calls.append(("register", int(sequence_id)))
            self._chains.setdefault(int(sequence_id), [])

    def unregister_sequences(self, sequence_ids):
        for sequence_id in sequence_ids:
            self.calls.append(("unregister", int(sequence_id)))
            self._chains.pop(int(sequence_id), None)
            self._attached.pop(int(sequence_id), None)
            self._rows.pop(int(sequence_id), None)

    def set_sequence_pages(self, sequence_id, page_ids):
        """Overwrite the LOCAL row only; the allocator chain is untouched."""

        self.calls.append(
            ("set_row", int(sequence_id), tuple(int(p) for p in page_ids))
        )
        self._rows[int(sequence_id)] = [int(page_id) for page_id in page_ids]

    # -- allocation --------------------------------------------------------

    def allocate_pages_for_sequences(self, requests):
        plans = [
            (int(sequence_id), int(tokens) // PAGE_TOKENS)
            for sequence_id, tokens in requests
        ]
        if sum(pages for _, pages in plans) > len(self._free):
            raise RuntimeError(
                "Insufficient free pages available for prefill batch"
            )
        for sequence_id, pages in plans:
            self.calls.append(("allocate", sequence_id, pages))
            taken = [self._free.pop(0) for _ in range(pages)]
            self._chains[sequence_id].extend(taken)

    def build_page_table(self, sequence_ids):
        return [self._table(int(sequence_id)) for sequence_id in sequence_ids]

    def _table(self, sequence_id: int) -> list[int]:
        # A local row set with ``set_sequence_pages`` wins: that is exactly the
        # override the C++ view gives it.
        if sequence_id in self._rows:
            return list(self._rows[sequence_id])
        if sequence_id in self._attached:
            return list(self._attached[sequence_id])
        if sequence_id not in self._chains:
            raise KeyError(f"sequence {sequence_id} is not registered")
        return list(self._chains[sequence_id])

    # -- byte access -------------------------------------------------------

    def read_sequence_kv_to_cpu(self, sequence_id):
        pages = self._table(int(sequence_id))
        self.calls.append(("read", int(sequence_id), tuple(pages)))
        blob = bytearray()
        for page_id in pages:
            blob += self._pages[page_id]
        window = torch.frombuffer(blob, dtype=torch.uint8).clone()
        # The real view returns [layers, pages, page_tokens, heads, dim]; only
        # the byte count and the dtype matter to the arena code, which treats
        # the window as a flat uint8 buffer.
        return (window, None)

    def write_sequence_kv_from_cpu(self, sequence_id, k_tensor, v_tensor=None):
        pages = self._table(int(sequence_id))
        self.calls.append(("write", int(sequence_id), tuple(pages)))
        data = bytes(
            k_tensor.reshape(-1).view(torch.uint8).contiguous().numpy()
        )
        if len(data) != len(pages) * self.page_bytes:
            raise AssertionError(
                f"write of {len(data)} B into {len(pages)} page(s)"
            )
        for index, page_id in enumerate(pages):
            start = index * self.page_bytes
            self._pages[page_id][:] = data[start : start + self.page_bytes]

    # -- ownership ---------------------------------------------------------

    def retain_sequence_pages(self, sequence_id, page_ids):
        chain = self._chains[int(sequence_id)]
        retained = []
        for page_id in page_ids:
            page_id = int(page_id)
            if page_id not in chain:
                raise AssertionError(f"page {page_id} is not in the chain")
            chain.remove(page_id)
            self._resident.add(page_id)
            retained.append(page_id)
        self.calls.append(("retain", int(sequence_id), tuple(retained)))
        return retained

    def attach_shared_prefix_pages(self, sequence_id, page_ids):
        sequence_id = int(sequence_id)
        if self._chains.get(sequence_id):
            raise AssertionError(
                "attach_shared_prefix_pages requires an empty page table"
            )
        self.calls.append(("attach", sequence_id, tuple(int(p) for p in page_ids)))
        self._attached[sequence_id] = [int(page_id) for page_id in page_ids]

    def release_sequence_pages(self, sequence_ids):
        for sequence_id in sequence_ids:
            sequence_id = int(sequence_id)
            self.calls.append(("release_chain", sequence_id))
            for page_id in self._chains.get(sequence_id, []):
                self._free.append(page_id)
            self._chains[sequence_id] = []

    def release_resident_pages(self, page_ids):
        self.calls.append(("release_resident", tuple(int(p) for p in page_ids)))
        for page_id in page_ids:
            page_id = int(page_id)
            if page_id not in self._resident:
                raise AssertionError(f"page {page_id} is not resident")
            self._resident.discard(page_id)
            self._free.append(page_id)

    # -- assertions --------------------------------------------------------

    @property
    def resident_pages(self) -> set[int]:
        return set(self._resident)

    @property
    def free_pages(self) -> int:
        return len(self._free)

    def page_bytes_of(self, page_id: int) -> bytes:
        return bytes(self._pages[int(page_id)])


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _state_manager(*, num_state_items: int = NUM_STATE_SLOTS):
    manager = KDAStateGPUManager(
        config=KDAStateGPUConfig(
            num_kda_layers=2,
            num_state_items=int(num_state_items),
            num_heads=2,
            head_dim=4,
            conv_dim=8,
            conv_width=4,
        ),
        device="cpu",
    )
    manager.initialize()
    return manager


def _seed_slot(manager, sequence_id: int, *, seed: int) -> int:
    """Fill one sequence's slot with distinctive, non-zero state."""

    slot = int(manager.allocate_state_item(int(sequence_id)))
    generator = torch.Generator().manual_seed(int(seed))
    recurrent = manager.get_recurrent_tensors()
    recurrent[:, slot] = torch.rand(
        recurrent[:, slot].shape, generator=generator, dtype=torch.float32
    )
    for index, conv in enumerate(manager.get_conv_tensors()):
        conv[:, slot] = torch.rand(
            conv[:, slot].shape, generator=generator, dtype=torch.float32
        ).to(conv.dtype) + float(index + 1)
    return slot


def _slot_chunks(manager, slot: int) -> tuple[bytes, ...]:
    """The four slot slices as raw bytes, in ``SNAPSHOT_CHUNK_NAMES`` order."""

    tensors = (manager.get_recurrent_tensors()[:, slot],) + tuple(
        conv[:, slot] for conv in manager.get_conv_tensors()
    )
    return tuple(
        bytes(
            tensor.contiguous().reshape(-1).view(torch.uint8).numpy()
        )
        for tensor in tensors
    )


class _Harness:
    """Committer + restorer over real objects and byte-accurate Host views."""

    def __init__(self, *, groups: int = 1, num_pages: int = 32, slots=None):
        self.manager = _state_manager(
            num_state_items=NUM_STATE_SLOTS if slots is None else int(slots)
        )
        self.profile = profile_from_kda_state_config(self.manager.config)
        self.layout = SnapshotLayout(profile=self.profile)
        self.views = {
            group_id: _FakeHostView(num_pages=num_pages)
            for group_id in range(int(groups))
        }
        self.host_page_bytes_all_layers = int(groups) * GROUP_PAGE_BYTES
        self._sid = 0
        self.committer = TrajectoryFrontierCommitter(
            rank=0,
            page_tokens=PAGE_TOKENS,
            host_page_bytes_all_layers=self.host_page_bytes_all_layers,
            arena_allocator=HostKVSnapshotArenaAllocator(
                host_views_provider=lambda: self.views,
                page_tokens=PAGE_TOKENS,
                host_page_bytes_all_layers=self.host_page_bytes_all_layers,
                next_pseudo_sid=self._next_pseudo_sid,
            ),
            snapshot_reader=KDAStateSnapshotReader(
                state_manager_provider=lambda: self.manager,
                profile=self.profile,
            ),
        )
        self.arena_reader = HostKVSnapshotArenaReader(
            host_views_provider=lambda: self.views,
            next_pseudo_sid=self._next_pseudo_sid,
        )
        self.state_writer = KDAStateSnapshotWriter(
            state_manager_provider=lambda: self.manager,
            profile=self.profile,
        )

    def _next_pseudo_sid(self) -> int:
        self._sid += 1
        return -self._sid

    def restorer(self, **kwargs) -> TrajectoryRestorer:
        kwargs.setdefault("arena_reader", self.arena_reader)
        kwargs.setdefault("state_writer", self.state_writer)
        return TrajectoryRestorer(
            rank=0, committer=self.committer, **kwargs
        )

    def frontier_pages(self) -> dict[int, tuple[int, ...]]:
        return {
            group_id: tuple(100 + 10 * group_id + i for i in range(2))
            for group_id in self.views
        }

    def commit(
        self,
        *,
        tokens=BASE_PROMPT,
        sequence_id=7,
        group_id=0,
        prompt_token_count=None,
    ):
        # The prefill ran exactly the committed boundary (Slice 1c's only
        # storable case), unless a test says otherwise.
        return self.committer.commit(
            namespace_digest=NAMESPACE,
            frontier_token_ids=tokens,
            page_ids_by_group=self.frontier_pages(),
            owning_group_id=int(group_id),
            sequence_id=int(sequence_id),
            prompt_token_count=(
                len(tokens) if prompt_token_count is None else prompt_token_count
            ),
        )


def _commit_seeded_frontier(harness, *, sequence_id=7, seed=1234):
    """Seed a slot, commit its frontier, and return (slot, chunks, outcome)."""

    slot = _seed_slot(harness.manager, sequence_id, seed=seed)
    expected = _slot_chunks(harness.manager, slot)
    outcome = harness.commit(sequence_id=sequence_id)
    assert outcome is not None
    return slot, expected, outcome


# ---------------------------------------------------------------------------
# The flag: one gate for the whole trajectory path
# ---------------------------------------------------------------------------


def test_slice_two_shares_the_commit_flag_exactly():
    assert DEBUG_FLAG == COMMIT_DEBUG_FLAG == "prefix_trajectory_commit"
    assert trajectory_reuse_enabled is trajectory_commit_enabled


def test_flag_off_means_zero_lookups():
    assert trajectory_reuse_enabled(None) is False
    assert trajectory_reuse_enabled({}) is False
    assert trajectory_reuse_enabled({DEBUG_FLAG: "off"}) is False
    harness = _Harness()
    restorer = harness.restorer()
    counters = restorer.counters()
    assert counters["trajectory_lookups"] == 0
    assert counters["trajectory_hits"] == 0
    assert counters["trajectory_restores"] == 0
    assert counters["trajectory_live_attachments"] == 0


def test_the_worker_lookup_is_guarded_by_that_one_flag():
    """Structural: the admission lookup has exactly one, gated, call site.

    A second ungated call site would make "flag off = byte-identical" false,
    and that is not visible from any unit of the mechanism itself.
    """

    source = (
        pathlib.Path(__file__).resolve().parents[2]
        / "batchgen"
        / "batchgen_worker.py"
    ).read_text()
    tree = ast.parse(source)

    def calls(node) -> list[str]:
        return [
            child.func.attr
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
        ]

    total = calls(tree).count("_lookup_trajectory_frontiers")
    assert total == 1, f"expected one lookup call site, found {total}"
    gated = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if "_trajectory_commit_requested" not in calls(node.test):
            continue
        for statement in node.body:
            gated += calls(statement).count("_lookup_trajectory_frontiers")
    assert gated == 1, "the admission lookup is not behind the batch flag"


# ---------------------------------------------------------------------------
# Lookup: the binary hit predicate as admission sees it
# ---------------------------------------------------------------------------


def test_a_continuation_prompt_hits_the_committed_frontier():
    harness = _Harness()
    _commit_seeded_frontier(harness)
    restorer = harness.restorer()

    attachment = restorer.lookup(
        namespace_digest=NAMESPACE,
        prompt_token_ids=BASE_PROMPT + (201, 202, 203),
        sequence_id=9,
    )

    assert attachment is not None
    assert attachment.boundary_tokens == len(BASE_PROMPT)
    assert attachment.lcp_tokens == len(BASE_PROMPT)
    assert attachment.owning_group_id == 0
    assert restorer.live_attachments == 1
    assert restorer.counters()["trajectory_hits"] == 1


def test_an_exact_frontier_prompt_hits():
    harness = _Harness()
    _commit_seeded_frontier(harness)
    restorer = harness.restorer()
    attachment = restorer.lookup(
        namespace_digest=NAMESPACE,
        prompt_token_ids=BASE_PROMPT,
        sequence_id=9,
    )
    assert attachment is not None
    assert attachment.lcp_tokens == len(BASE_PROMPT)


def test_a_re_rendered_prompt_below_the_frontier_is_a_full_miss():
    """Partial LCP is a FULL miss -- the pages below it are never half-used."""

    harness = _Harness()
    _commit_seeded_frontier(harness)
    restorer = harness.restorer()

    # Diverges at token 5: six tokens agree, the frontier boundary is eight.
    re_rendered = BASE_PROMPT[:5] + (900, 901, 902, 903)
    attachment = restorer.lookup(
        namespace_digest=NAMESPACE,
        prompt_token_ids=re_rendered,
        sequence_id=9,
    )

    assert attachment is None
    assert restorer.live_attachments == 0
    counters = restorer.counters()
    assert counters["trajectory_lookups"] == 1
    assert counters["trajectory_hits"] == 0


def test_a_prompt_shorter_than_the_frontier_is_a_miss():
    harness = _Harness()
    _commit_seeded_frontier(harness)
    restorer = harness.restorer()
    attachment = restorer.lookup(
        namespace_digest=NAMESPACE,
        prompt_token_ids=BASE_PROMPT[:4],
        sequence_id=9,
    )
    assert attachment is None
    assert restorer.live_attachments == 0


def test_an_unrelated_prompt_is_a_miss():
    harness = _Harness()
    _commit_seeded_frontier(harness)
    restorer = harness.restorer()
    attachment = restorer.lookup(
        namespace_digest=NAMESPACE,
        prompt_token_ids=(7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, 7009),
        sequence_id=9,
    )
    assert attachment is None
    assert restorer.counters()["trajectory_hits"] == 0


def test_a_lookup_against_an_empty_registry_is_a_miss():
    harness = _Harness()
    restorer = harness.restorer()
    assert (
        restorer.lookup(
            namespace_digest=NAMESPACE,
            prompt_token_ids=BASE_PROMPT,
            sequence_id=9,
        )
        is None
    )
    assert restorer.live_attachments == 0


def test_a_different_namespace_digest_never_hits():
    harness = _Harness()
    _commit_seeded_frontier(harness)
    restorer = harness.restorer()
    assert (
        restorer.lookup(
            namespace_digest=(1, 2, 3, 4),
            prompt_token_ids=BASE_PROMPT + (5,),
            sequence_id=9,
        )
        is None
    )


def test_a_miss_logs_the_longest_agreement(caplog):
    harness = _Harness()
    _commit_seeded_frontier(harness)
    restorer = harness.restorer()
    with caplog.at_level("INFO"):
        restorer.lookup(
            namespace_digest=NAMESPACE,
            prompt_token_ids=BASE_PROMPT[:5] + (900, 901, 902),
            sequence_id=9,
        )
    assert any(
        "TRAJECTORY_RESTORE" in record.getMessage()
        and "MISS" in record.getMessage()
        for record in caplog.records
    )


# ---------------------------------------------------------------------------
# Attach: the reference is what pins the record
# ---------------------------------------------------------------------------


def test_attach_takes_a_reference_and_eviction_is_refused_while_attached():
    harness = _Harness()
    _commit_seeded_frontier(harness)
    restorer = harness.restorer()
    record = harness.committer.registry.get(
        harness.committer.registry.session_keys[0]
    )
    assert record.refcount == 1

    attachment = restorer.lookup(
        namespace_digest=NAMESPACE,
        prompt_token_ids=BASE_PROMPT + (5,),
        sequence_id=9,
    )
    assert attachment is not None
    assert record.refcount == 2

    # Slice 1a semantics: the registry REFUSES a referenced record by name.
    with pytest.raises(FrontierRefcountError) as excinfo:
        harness.committer.on_prefix_pages_released(
            {0: list(harness.frontier_pages()[0])}
        )
    assert attachment.session_key in str(excinfo.value)
    assert "attached" in str(excinfo.value)
    assert harness.committer.registry.get(attachment.session_key) is not None


def test_eviction_succeeds_once_the_attachment_is_consumed():
    harness = _Harness()
    _commit_seeded_frontier(harness)
    restorer = harness.restorer()
    attachment = restorer.lookup(
        namespace_digest=NAMESPACE,
        prompt_token_ids=BASE_PROMPT + (5,),
        sequence_id=9,
    )
    session_key = attachment.session_key
    restorer.release(attachment)
    assert restorer.live_attachments == 0

    outcomes = harness.committer.on_prefix_pages_released(
        {0: list(harness.frontier_pages()[0])}
    )
    assert [outcome.session_key for outcome in outcomes] == [session_key]
    assert harness.committer.registry.get(session_key) is None
    # The arena's Host pages went back with the record.
    assert harness.views[0].resident_pages == set()


def test_an_attachment_cannot_be_consumed_twice():
    harness = _Harness()
    _commit_seeded_frontier(harness)
    restorer = harness.restorer()
    attachment = restorer.lookup(
        namespace_digest=NAMESPACE,
        prompt_token_ids=BASE_PROMPT + (5,),
        sequence_id=9,
    )
    restorer.release(attachment)
    with pytest.raises(TrajectoryAttachmentInvalid):
        restorer.release(attachment)
    assert restorer.live_attachments == 0


def test_a_consumed_attachment_cannot_be_restored():
    harness = _Harness()
    _commit_seeded_frontier(harness)
    restorer = harness.restorer()
    attachment = restorer.lookup(
        namespace_digest=NAMESPACE,
        prompt_token_ids=BASE_PROMPT + (5,),
        sequence_id=9,
    )
    restorer.release(attachment)
    with pytest.raises(TrajectoryAttachmentInvalid):
        restorer.restore(
            attachment,
            serving_group_id=0,
            compute_cached_tokens=len(BASE_PROMPT),
        )


# ---------------------------------------------------------------------------
# Restore: the bytes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("groups", [1, 2])
def test_restore_writes_the_committed_bytes_back_chunk_for_chunk(groups):
    """The round trip, through real Host pages.

    With two groups the blob is larger than group 0's whole page window, so it
    straddles the group boundary -- the arena-layout contract the commit path's
    ``_write_blob`` docstring states, exercised in both shapes.
    """

    harness = _Harness(groups=groups)
    if groups == 2:
        assert harness.layout.total_bytes > 2 * GROUP_PAGE_BYTES, (
            "the two-group case must straddle a group boundary to be the "
            "test it claims to be"
        )
    _slot, expected, outcome = _commit_seeded_frontier(harness, sequence_id=7)
    restorer = harness.restorer()

    attachment = restorer.lookup(
        namespace_digest=NAMESPACE,
        prompt_token_ids=BASE_PROMPT + (201, 202),
        sequence_id=9,
    )
    assert attachment is not None
    result = restorer.restore(
        attachment,
        serving_group_id=0,
        compute_cached_tokens=len(BASE_PROMPT),
    )

    assert result is not None
    assert result.session_key == outcome.session_key
    assert result.boundary_tokens == len(BASE_PROMPT)
    assert result.restored_bytes == harness.layout.total_bytes
    assert result.allocated_slot is True

    restored = _slot_chunks(harness.manager, result.state_slot)
    assert len(restored) == len(SNAPSHOT_CHUNK_NAMES)
    for name, got, want in zip(SNAPSHOT_CHUNK_NAMES, restored, expected):
        assert got == want, f"chunk {name} came back different"

    # The restore consumed its attachment: nothing is pinned afterwards.
    assert restorer.live_attachments == 0
    assert restorer.counters()["trajectory_restores"] == 1


def test_restore_binds_the_sequence_to_the_slot_it_wrote():
    harness = _Harness()
    _commit_seeded_frontier(harness, sequence_id=7)
    restorer = harness.restorer()
    attachment = restorer.lookup(
        namespace_digest=NAMESPACE,
        prompt_token_ids=BASE_PROMPT + (5,),
        sequence_id=9,
    )
    result = restorer.restore(
        attachment, serving_group_id=0, compute_cached_tokens=len(BASE_PROMPT)
    )
    assert harness.manager.has_sequence_state_item(9) is True
    assert harness.manager.get_sequence_state_item(9) == result.state_slot
    assert result.state_slot != harness.manager.get_sequence_state_item(7)


def test_restore_into_a_slot_the_sequence_already_held_keeps_it():
    """A row that already owns a slot is restored IN PLACE, not re-allocated."""

    harness = _Harness()
    _slot, expected, _ = _commit_seeded_frontier(harness, sequence_id=7)
    existing = _seed_slot(harness.manager, 9, seed=99)
    restorer = harness.restorer()
    attachment = restorer.lookup(
        namespace_digest=NAMESPACE,
        prompt_token_ids=BASE_PROMPT + (5,),
        sequence_id=9,
    )
    result = restorer.restore(
        attachment, serving_group_id=0, compute_cached_tokens=len(BASE_PROMPT)
    )
    assert result.state_slot == existing
    assert result.allocated_slot is False
    assert _slot_chunks(harness.manager, existing) == expected


def test_restore_does_not_advance_the_frontier():
    harness = _Harness()
    _commit_seeded_frontier(harness)
    restorer = harness.restorer()
    before = harness.committer.registry.session_keys
    record = harness.committer.registry.get(before[0])
    boundary = record.frontier_boundary_tokens
    attachment = restorer.lookup(
        namespace_digest=NAMESPACE,
        prompt_token_ids=BASE_PROMPT + (201, 202, 203),
        sequence_id=9,
    )
    restorer.restore(
        attachment, serving_group_id=0, compute_cached_tokens=len(BASE_PROMPT)
    )
    assert harness.committer.registry.session_keys == before
    assert (
        harness.committer.registry.get(before[0]).frontier_boundary_tokens
        == boundary
    )
    assert record.refcount == 1


def test_the_arena_reader_leaves_the_arena_pages_resident():
    harness = _Harness()
    _commit_seeded_frontier(harness)
    arena_pages = harness.views[0].resident_pages
    restorer = harness.restorer()
    attachment = restorer.lookup(
        namespace_digest=NAMESPACE,
        prompt_token_ids=BASE_PROMPT + (5,),
        sequence_id=9,
    )
    free_before = harness.views[0].free_pages
    restorer.restore(
        attachment, serving_group_id=0, compute_cached_tokens=len(BASE_PROMPT)
    )
    assert harness.views[0].resident_pages == arena_pages
    assert harness.views[0].free_pages == free_before
    # The read row went away; nothing in it released a page.
    assert ("release_chain", -2) not in harness.views[0].calls
    assert ("unregister", -2) in harness.views[0].calls


def test_the_arena_reader_reads_under_a_negative_pseudo_sid():
    harness = _Harness()
    _commit_seeded_frontier(harness)
    restorer = harness.restorer()
    attachment = restorer.lookup(
        namespace_digest=NAMESPACE,
        prompt_token_ids=BASE_PROMPT + (5,),
        sequence_id=9,
    )
    restorer.restore(
        attachment, serving_group_id=0, compute_cached_tokens=len(BASE_PROMPT)
    )
    reads = [call for call in harness.views[0].calls if call[0] == "read"]
    assert reads, "the arena was never read"
    assert all(call[1] < 0 for call in reads)


def test_the_arena_reader_refuses_a_non_negative_pseudo_sid():
    harness = _Harness()
    _commit_seeded_frontier(harness)
    reader = HostKVSnapshotArenaReader(
        host_views_provider=lambda: harness.views,
        next_pseudo_sid=lambda: 5,
    )
    record = harness.committer.registry.get(
        harness.committer.registry.session_keys[0]
    )
    arena = harness.committer.arena_for(record.snapshot.storage_handle)
    with pytest.raises(SnapshotArenaCorrupt) as excinfo:
        reader.read(arena, layout=harness.layout)
    assert "NEGATIVE" in str(excinfo.value)


def test_the_arena_reader_refuses_a_short_arena():
    harness = _Harness()
    _commit_seeded_frontier(harness)
    record = harness.committer.registry.get(
        harness.committer.registry.session_keys[0]
    )
    arena = harness.committer.arena_for(record.snapshot.storage_handle)
    # One page short of the blob: the arena no longer matches its profile.
    from dataclasses import replace

    short = replace(
        arena,
        page_ids_by_group={0: tuple(arena.page_ids_by_group[0][:1])},
        # Internally CONSISTENT but too small for the blob: the arena's own
        # geometry check now refuses a page list that disagrees with its page
        # count, so shortness has to be expressed as a smaller arena.
        pages_per_group=1,
    )
    with pytest.raises(SnapshotArenaCorrupt) as excinfo:
        harness.arena_reader.read(short, layout=harness.layout)
    assert "no longer matches its profile" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Refusals: every one releases its attachment and names its reason. A row
# whose compute starts at the boundary is PLANNED -- the wave is built around
# that start -- so every refusal of it is raised, never declined.
# ---------------------------------------------------------------------------


def _hit(harness, restorer, *, sequence_id=9):
    attachment = restorer.lookup(
        namespace_digest=NAMESPACE,
        prompt_token_ids=BASE_PROMPT + (201, 202),
        sequence_id=sequence_id,
    )
    assert attachment is not None
    return attachment


def test_slot_exhaustion_of_a_planned_row_fails_loud_and_leaks_nothing(caplog):
    """No wait, and no silent recompute: the planned row cannot run stateless.

    The plan set the row's compute start to the boundary, so declining here
    used to run its suffix on a KDA state that does not hold the prefix.
    """

    # Two slots: one holds the committing sequence's state, one is taken by a
    # live row, so the restore finds the free list empty.
    harness = _Harness(slots=2)
    _commit_seeded_frontier(harness, sequence_id=7)
    harness.manager.allocate_state_item(8)
    assert harness.manager.get_stats().num_free_state_items == 0
    restorer = harness.restorer()
    attachment = _hit(harness, restorer)
    record = attachment.record

    with caplog.at_level("WARNING"):
        with pytest.raises(TrajectoryPlannedRestoreFailed) as excinfo:
            restorer.restore(
                attachment,
                serving_group_id=0,
                compute_cached_tokens=len(BASE_PROMPT),
            )

    error = excinfo.value
    assert error.reason == DECLINE_SLOTS_EXHAUSTED
    assert error.sequence_id == 9
    assert error.session_key == attachment.session_key
    assert error.boundary_tokens == len(BASE_PROMPT)
    assert error.state_slot is None
    assert error.rank == 0
    assert "Insufficient free KDA state items" in error.detail
    # Consumed before the raise, so nothing leaks.
    assert restorer.live_attachments == 0
    assert record.refcount == 1
    assert attachment.consumed is True
    assert harness.manager.has_sequence_state_item(9) is False
    counters = restorer.counters()
    assert counters[f"trajectory_decline_{DECLINE_SLOTS_EXHAUSTED}"] == 1
    assert counters["trajectory_restores"] == 0
    message = "\n".join(record.getMessage() for record in caplog.records)
    assert "TRAJECTORY_RESTORE" in message
    assert "PLANNED_RESTORE_FAILED" in message
    assert f"reason={DECLINE_SLOTS_EXHAUSTED}" in message
    assert "full recompute" not in message


def test_the_slot_writer_names_exhaustion_and_hides_nothing_else():
    harness = _Harness(slots=1)
    harness.manager.allocate_state_item(1)
    with pytest.raises(KDAStateSlotsExhausted) as excinfo:
        harness.state_writer.allocate_slot(2)
    assert "no free KDA state slot" in str(excinfo.value)
    assert "Insufficient free KDA state items" in str(excinfo.value)


def test_a_non_capacity_slot_failure_is_not_downgraded_to_exhaustion():
    class _Manager:
        def allocate_state_item(self, sequence_id):
            raise RuntimeError("GPU page table is not initialized")

    writer = KDAStateSnapshotWriter(
        state_manager_provider=lambda: _Manager(), profile=None
    )
    with pytest.raises(RuntimeError) as excinfo:
        writer.allocate_slot(3)
    assert not isinstance(excinfo.value, KDAStateSlotsExhausted)
    assert "GPU page table is not initialized" in str(excinfo.value)


def test_a_failure_between_the_arena_read_and_the_bind_leaks_nothing():
    """The slot this restore took is handed back; the attachment is released.

    And the planned row fails loud, naming the slot it held.
    """

    harness = _Harness()
    _commit_seeded_frontier(harness, sequence_id=7)
    restorer = harness.restorer()
    attachment = _hit(harness, restorer)
    record = attachment.record
    free_before = harness.manager.get_stats().num_free_state_items

    def _boom(sequence_id, chunks):
        raise RuntimeError("slot write exploded")

    harness.state_writer.write_slot = _boom

    with pytest.raises(TrajectoryPlannedRestoreFailed) as excinfo:
        restorer.restore(
            attachment, serving_group_id=0, compute_cached_tokens=len(BASE_PROMPT)
        )

    error = excinfo.value
    assert error.reason == DECLINE_SLOT_WRITE_FAILED
    assert "slot write exploded" in error.detail
    # The slot this restore bound, and then handed back.
    assert error.state_slot is not None
    # No leaked slot: the free list is back where it was and the sequence is
    # not bound to a half-written slot.
    assert harness.manager.get_stats().num_free_state_items == free_before
    assert harness.manager.has_sequence_state_item(9) is False
    # No leaked attachment: the record is evictable again.
    assert restorer.live_attachments == 0
    assert record.refcount == 1
    assert (
        restorer.counters()[f"trajectory_decline_{DECLINE_SLOT_WRITE_FAILED}"]
        == 1
    )


def test_a_write_failure_keeps_a_slot_the_sequence_already_owned():
    harness = _Harness()
    _commit_seeded_frontier(harness, sequence_id=7)
    existing = _seed_slot(harness.manager, 9, seed=99)
    restorer = harness.restorer()
    attachment = _hit(harness, restorer)

    def _boom(sequence_id, chunks):
        raise RuntimeError("slot write exploded")

    harness.state_writer.write_slot = _boom
    with pytest.raises(TrajectoryPlannedRestoreFailed) as excinfo:
        restorer.restore(
            attachment,
            serving_group_id=0,
            compute_cached_tokens=len(BASE_PROMPT),
        )
    assert excinfo.value.reason == DECLINE_SLOT_WRITE_FAILED
    assert excinfo.value.state_slot == existing
    # The row's own slot is NOT handed back by a restore that did not take it.
    assert harness.manager.has_sequence_state_item(9) is True
    assert harness.manager.get_sequence_state_item(9) == existing
    assert restorer.live_attachments == 0


def test_an_arena_read_failure_of_a_planned_row_fails_loud_and_takes_no_slot():
    harness = _Harness()
    _commit_seeded_frontier(harness, sequence_id=7)
    restorer = harness.restorer()
    attachment = _hit(harness, restorer)
    free_before = harness.manager.get_stats().num_free_state_items

    def _boom(arena, *, layout, shard_index=0):
        raise SnapshotArenaCorrupt("arena read exploded")

    harness.arena_reader.read = _boom

    with pytest.raises(TrajectoryPlannedRestoreFailed) as excinfo:
        restorer.restore(
            attachment,
            serving_group_id=0,
            compute_cached_tokens=len(BASE_PROMPT),
        )
    assert excinfo.value.reason == DECLINE_ARENA_READ_FAILED
    assert "arena read exploded" in excinfo.value.detail
    assert excinfo.value.state_slot is None
    assert harness.manager.get_stats().num_free_state_items == free_before
    assert harness.manager.has_sequence_state_item(9) is False
    assert restorer.live_attachments == 0
    assert (
        restorer.counters()[f"trajectory_decline_{DECLINE_ARENA_READ_FAILED}"]
        == 1
    )


def test_a_missing_arena_of_a_planned_row_fails_loud_by_name():
    harness = _Harness()
    _commit_seeded_frontier(harness, sequence_id=7)
    restorer = harness.restorer()
    attachment = _hit(harness, restorer)
    handle = attachment.storage_handle
    # The handle->arena table is the committer's; emptying it models the one
    # state that must never be read through: a record whose arena is gone.
    harness.committer._arenas_by_handle.pop(handle)

    with pytest.raises(TrajectoryPlannedRestoreFailed) as excinfo:
        restorer.restore(
            attachment,
            serving_group_id=0,
            compute_cached_tokens=len(BASE_PROMPT),
        )
    assert excinfo.value.reason == DECLINE_ARENA_MISSING
    assert f"storage handle {handle}" in excinfo.value.detail
    assert harness.manager.has_sequence_state_item(9) is False
    assert restorer.live_attachments == 0
    assert (
        restorer.counters()[f"trajectory_decline_{DECLINE_ARENA_MISSING}"] == 1
    )


def test_a_suffix_that_does_not_start_at_the_frontier_is_planner_drift():
    """The offset identity, now an invariant the wave planner makes true.

    Only a planned row reaches the restore, and the plan set its start to
    the boundary; any other start is counted under its name and RAISED --
    never restored into, and never silently recomputed either.
    """

    harness = _Harness()
    _commit_seeded_frontier(harness, sequence_id=7)
    restorer = harness.restorer()
    for cached in (0, len(BASE_PROMPT) - 1, len(BASE_PROMPT) + 1):
        attachment = _hit(harness, restorer)
        with pytest.raises(TrajectoryRestorePlanDrift, match="disagree"):
            restorer.restore(
                attachment,
                serving_group_id=0,
                compute_cached_tokens=cached,
            )
        assert attachment.consumed is True
        assert restorer.live_attachments == 0
        assert harness.manager.has_sequence_state_item(9) is False
    assert (
        restorer.counters()[f"trajectory_decline_{DECLINE_SUFFIX_OFFSET}"] == 3
    )


def test_a_serving_group_other_than_the_owning_group_is_refused(caplog):
    """Planned off-group: the plan drops such a row, so reaching here raises."""

    harness = _Harness()
    _commit_seeded_frontier(harness, sequence_id=7)
    restorer = harness.restorer()
    attachment = _hit(harness, restorer)
    with pytest.raises(TrajectoryPlannedRestoreFailed) as excinfo:
        restorer.restore(
            attachment,
            serving_group_id=1,
            compute_cached_tokens=len(BASE_PROMPT),
        )
    assert excinfo.value.reason == DECLINE_GROUP_MISMATCH
    assert restorer.live_attachments == 0
    assert (
        restorer.counters()[f"trajectory_decline_{DECLINE_GROUP_MISMATCH}"] == 1
    )
    assert "planned_offgroup=true" in caplog.text


def test_a_shard_count_that_is_not_this_group_is_refused_by_name(caplog):
    """A record whose shard count is not this rank's group size is refused.

    Slice 1c made a G>1 record restorable -- every rank's head shard is in the
    arena -- so the refusal is no longer "G>1 at all". What stays refused is a
    MISMATCH: a one-shard record read by a rank of a 4-rank group would take
    the whole arena as its own heads, and a 4-shard record read at G=1 would
    take a quarter of the state as all of it. Both bind wrong state that no
    downstream assertion can see, so both are refused rather than approximated.
    """

    harness = _Harness()
    _commit_seeded_frontier(harness, sequence_id=7)
    restorer = harness.restorer(attention_group_size=4)
    attachment = _hit(harness, restorer)
    with caplog.at_level("WARNING"):
        with pytest.raises(TrajectoryPlannedRestoreFailed) as excinfo:
            restorer.restore(
                attachment,
                serving_group_id=0,
                compute_cached_tokens=len(BASE_PROMPT),
            )
    assert excinfo.value.reason == DECLINE_HEAD_SHARDED_GROUP
    assert restorer.live_attachments == 0
    assert harness.manager.has_sequence_state_item(9) is False
    assert (
        restorer.counters()[
            f"trajectory_decline_{DECLINE_HEAD_SHARDED_GROUP}"
        ]
        == 1
    )
    assert "binds 1 head shard(s) but this rank serves a 4-rank group" in (
        "\n".join(record.getMessage() for record in caplog.records)
    )


def test_an_unknown_decline_reason_is_refused():
    harness = _Harness()
    _commit_seeded_frontier(harness)
    restorer = harness.restorer()
    attachment = _hit(harness, restorer)
    with pytest.raises(TrajectoryAttachmentInvalid):
        restorer.decline(attachment, reason="because")
    # The refusal did not consume the attachment, so it is still releasable.
    assert restorer.live_attachments == 1
    restorer.release(attachment)
    assert restorer.live_attachments == 0


# ---------------------------------------------------------------------------
# The drain counter
# ---------------------------------------------------------------------------


def test_every_outcome_returns_the_live_count_to_zero():
    harness = _Harness()
    _commit_seeded_frontier(harness, sequence_id=7)
    restorer = harness.restorer()

    # A restore, an unplanned decline, a planned failure and a bare release,
    # in one batch.
    restored = _hit(harness, restorer, sequence_id=9)
    assert (
        restorer.restore(
            restored,
            serving_group_id=0,
            compute_cached_tokens=len(BASE_PROMPT),
        )
        is not None
    )
    declined = _hit(harness, restorer, sequence_id=10)
    assert (
        restorer.restore(declined, serving_group_id=7, compute_cached_tokens=0)
        is None
    )
    failed = _hit(harness, restorer, sequence_id=12)
    with pytest.raises(TrajectoryPlannedRestoreFailed):
        restorer.restore(
            failed, serving_group_id=7, compute_cached_tokens=len(BASE_PROMPT)
        )
    dropped = _hit(harness, restorer, sequence_id=11)
    restorer.release(dropped)

    counters = restorer.counters()
    assert counters["trajectory_live_attachments"] == 0
    assert counters["trajectory_hits"] == 4
    assert counters["trajectory_restores"] == 1
    assert counters["trajectory_declines"] == 2
    assert set(DECLINE_REASONS).issubset(
        {key.removeprefix("trajectory_decline_") for key in counters}
    )


def test_the_counter_surface_is_flat_ints():
    harness = _Harness()
    restorer = harness.restorer()
    counters = restorer.counters()
    assert counters
    assert all(isinstance(value, int) for value in counters.values())
    assert all(key.startswith("trajectory_") for key in counters)


# ---------------------------------------------------------------------------
# Writer geometry
# ---------------------------------------------------------------------------


def test_the_writer_refuses_a_chunk_of_the_wrong_size():
    harness = _Harness()
    harness.manager.allocate_state_item(5)
    chunks = list(
        harness.arena_reader.read(
            harness.committer.arena_for(
                harness.commit(sequence_id=5).record.snapshot.storage_handle
            ),
            layout=harness.layout,
        )
    )
    chunks[1] = chunks[1][:-1]
    with pytest.raises(TrajectorySnapshotShapeMismatch) as excinfo:
        harness.state_writer.write_slot(5, chunks)
    assert "conv_q" in str(excinfo.value)


def test_the_writer_refuses_the_wrong_chunk_count():
    harness = _Harness()
    harness.manager.allocate_state_item(5)
    with pytest.raises(TrajectorySnapshotShapeMismatch) as excinfo:
        harness.state_writer.write_slot(5, ())
    assert ", ".join(SNAPSHOT_CHUNK_NAMES) in str(excinfo.value)


def test_the_writer_refuses_a_sequence_with_no_slot():
    from batchgen.prefix_reuse.trajectory_commit import (
        TrajectoryStateUnavailable,
    )

    harness = _Harness()
    with pytest.raises(TrajectoryStateUnavailable) as excinfo:
        harness.state_writer.write_slot(404, (None, None, None, None))
    assert "allocate_slot must run first" in str(excinfo.value)


def test_restore_round_trips_under_inference_mode():
    """The real call site runs inside ``torch.inference_mode``.

    The slot pools are built outside it, so the in-place slot write is legal
    there -- but only if the restore never produces a tensor the pools then
    keep. Pinned here because the guard is invisible from the mechanism.
    """

    harness = _Harness()
    _slot, expected, _ = _commit_seeded_frontier(harness, sequence_id=7)
    restorer = harness.restorer()
    attachment = _hit(harness, restorer)

    with torch.inference_mode():
        result = restorer.restore(
            attachment,
            serving_group_id=0,
            compute_cached_tokens=len(BASE_PROMPT),
        )

    assert result is not None
    assert _slot_chunks(harness.manager, result.state_slot) == expected


def test_a_misaligned_chunk_offset_still_round_trips():
    """A conv dtype wider than the recurrent one mis-aligns later chunks.

    ``Tensor.view(dtype)`` refuses a storage offset that is not divisible by
    the element-size ratio, so the writer copies exactly those chunks. The
    bytes must be identical either way.
    """

    manager = KDAStateGPUManager(
        config=KDAStateGPUConfig(
            num_kda_layers=1,
            num_state_items=2,
            num_heads=1,
            head_dim=1,
            conv_dim=1,
            conv_width=4,
            recurrent_dtype=torch.float16,
            conv_dtype=torch.float32,
        ),
        device="cpu",
    )
    manager.initialize()
    profile = profile_from_kda_state_config(manager.config)
    layout = SnapshotLayout(profile=profile)
    # recurrent is fp16 (2 B) and conv is fp32 (4 B): the first conv chunk
    # starts at recurrent_bytes, which is not a multiple of 4 here.
    assert layout.chunk_offsets[1] % 4 != 0

    source = _seed_slot(manager, 1, seed=7)
    expected = _slot_chunks(manager, source)
    reader = KDAStateSnapshotReader(
        state_manager_provider=lambda: manager, profile=profile
    )
    writer = KDAStateSnapshotWriter(
        state_manager_provider=lambda: manager, profile=profile
    )
    blob = torch.cat([chunk for chunk in reader.read_slot(1)])
    chunks = [
        blob[offset : offset + size]
        for offset, size in zip(layout.chunk_offsets, layout.chunk_bytes)
    ]

    writer.allocate_slot(2)
    slot, written = writer.write_slot(2, chunks)
    assert written == layout.total_bytes
    assert _slot_chunks(manager, slot) == expected


# ---------------------------------------------------------------------------
# Slice 1c: G>1 -- every rank snapshots and restores its OWN head shard
# ---------------------------------------------------------------------------


class _GroupHarness:
    """G ranks of one serve group over ONE node-shared Host KV region.

    The views are shared because the real region is: that is what lets a
    follower write and read its own window without allocating anything. The
    KDA state managers are per rank, because the state is head-sharded -- each
    rank holds different bytes, which is exactly what the windows must keep
    apart.
    """

    def __init__(self, *, shard_count: int = 4, groups: int = 1, num_pages: int = 64):
        self.shard_count = int(shard_count)
        self.managers = [
            _state_manager() for _ in range(self.shard_count)
        ]
        self.profile = profile_from_kda_state_config(self.managers[0].config)
        self.layout = SnapshotLayout(profile=self.profile)
        self.views = {
            group_id: _FakeHostView(num_pages=num_pages)
            for group_id in range(int(groups))
        }
        self.host_page_bytes_all_layers = int(groups) * GROUP_PAGE_BYTES
        self._sid = 0
        self.allocator = HostKVSnapshotArenaAllocator(
            host_views_provider=lambda: self.views,
            page_tokens=PAGE_TOKENS,
            host_page_bytes_all_layers=self.host_page_bytes_all_layers,
            next_pseudo_sid=self._next_pseudo_sid,
        )
        self.committers = [
            TrajectoryFrontierCommitter(
                rank=index,
                page_tokens=PAGE_TOKENS,
                host_page_bytes_all_layers=self.host_page_bytes_all_layers,
                arena_allocator=self.allocator,
                snapshot_reader=KDAStateSnapshotReader(
                    state_manager_provider=(
                        lambda manager=self.managers[index]: manager
                    ),
                    profile=self.profile,
                ),
                shard_count=self.shard_count,
                shard_index=index,
                owns_registry=(index == 0),
            )
            for index in range(self.shard_count)
        ]

    def _next_pseudo_sid(self) -> int:
        self._sid += 1
        return -self._sid

    @property
    def leader(self):
        return self.committers[0]

    def frontier_pages(self):
        return {
            group_id: tuple(100 + 10 * group_id + i for i in range(2))
            for group_id in self.views
        }

    def restorer(self, shard_index: int, **kwargs) -> TrajectoryRestorer:
        kwargs.setdefault(
            "arena_reader",
            HostKVSnapshotArenaReader(
                host_views_provider=lambda: self.views,
                next_pseudo_sid=self._next_pseudo_sid,
            ),
        )
        kwargs.setdefault(
            "state_writer",
            KDAStateSnapshotWriter(
                state_manager_provider=(
                    lambda manager=self.managers[int(shard_index)]: manager
                ),
                profile=self.profile,
            ),
        )
        kwargs.setdefault("attention_group_size", self.shard_count)
        return TrajectoryRestorer(
            rank=int(shard_index),
            committer=self.leader,
            shard_index=int(shard_index),
            **kwargs,
        )

    def seed(self, sequence_id: int = 7):
        """Distinct state per rank, so a crossed window is unmissable."""

        slots, expected = [], []
        for index, manager in enumerate(self.managers):
            slot = _seed_slot(manager, sequence_id, seed=1000 + 7 * index)
            slots.append(slot)
            expected.append(_slot_chunks(manager, slot))
        return slots, expected

    def commit_pass(self, *, tokens=BASE_PROMPT, sequence_id=7):
        """The real pass: reserve in config, all G ranks write, then finalize."""

        arena = self.leader.reserve_arena(
            sequence_id=sequence_id,
            prefix_page_count=2,
            namespace_digest=NAMESPACE,
            frontier_token_ids=tokens,
        )
        assert arena is not None
        for follower in self.committers[1:]:
            follower.adopt_arena(sequence_id=sequence_id, arena=arena)
        assert (
            self.leader.commit(
                namespace_digest=NAMESPACE,
                frontier_token_ids=tokens,
                page_ids_by_group=self.frontier_pages(),
                owning_group_id=0,
                sequence_id=sequence_id,
                prompt_token_count=len(tokens),
            )
            is not None
        )
        for follower in self.committers[1:]:
            assert (
                follower.write_shard(
                    sequence_id=sequence_id,
                    frontier_token_ids=tokens,
                    prompt_token_count=len(tokens),
                )
                is not None
            )
        finalized = self.leader.finalize()
        for follower in self.committers[1:]:
            follower.finalize()
        assert len(finalized.installed) == 1
        return arena, finalized.installed[0]


def _group_hit(harness, restorer, *, sequence_id=9):
    attachment = restorer.lookup(
        namespace_digest=NAMESPACE,
        prompt_token_ids=BASE_PROMPT + (201, 202),
        sequence_id=sequence_id,
    )
    assert attachment is not None
    return attachment


@pytest.mark.parametrize("shard_count", [2, 4])
def test_a_matching_shard_count_is_restored_instead_of_refused(shard_count):
    """The Slice 2 blanket G>1 refusal is gone for a complete snapshot."""

    harness = _GroupHarness(shard_count=shard_count)
    harness.seed()
    harness.commit_pass()
    restorer = harness.restorer(0)
    outcome = restorer.restore(
        _group_hit(harness, restorer),
        serving_group_id=0,
        compute_cached_tokens=len(BASE_PROMPT),
    )

    assert outcome is not None
    assert restorer.counters()[
        f"trajectory_decline_{DECLINE_HEAD_SHARDED_GROUP}"
    ] == 0
    assert restorer.counters()["trajectory_restores"] == 1
    assert restorer.live_attachments == 0


@pytest.mark.parametrize("shard_index", [0, 1, 2, 3])
def test_each_rank_restores_exactly_its_own_window(shard_index):
    """Rank r's slot must come back with rank r's bytes, not a neighbour's.

    Every rank's source state is seeded differently, so reading the wrong
    window would restore bytes that are valid-looking but belong to another
    rank's heads -- the exact failure no downstream assertion can see.
    """

    harness = _GroupHarness(shard_count=4)
    _slots, expected = harness.seed()
    harness.commit_pass()

    restorer = harness.restorer(shard_index)
    outcome = restorer.restore(
        _group_hit(harness, restorer),
        serving_group_id=0,
        compute_cached_tokens=len(BASE_PROMPT),
    )

    assert outcome is not None
    manager = harness.managers[shard_index]
    restored = _slot_chunks(manager, outcome.state_slot)
    assert restored == expected[shard_index]
    # And it is NOT any other rank's shard.
    for other in range(4):
        if other != shard_index:
            assert restored != expected[other]


def test_a_shard_count_that_is_not_this_ranks_group_is_refused_at_g_gt_1():
    """A 4-shard record read by a rank that thinks G==2 is refused by name."""

    harness = _GroupHarness(shard_count=4)
    harness.seed()
    harness.commit_pass()
    restorer = harness.restorer(0, attention_group_size=2)
    attachment = _group_hit(harness, restorer)
    with pytest.raises(TrajectoryPlannedRestoreFailed) as excinfo:
        restorer.restore(
            attachment,
            serving_group_id=0,
            compute_cached_tokens=len(BASE_PROMPT),
        )
    assert excinfo.value.reason == DECLINE_HEAD_SHARDED_GROUP
    assert "4 head shard(s)" in excinfo.value.detail
    assert restorer.counters()[
        f"trajectory_decline_{DECLINE_HEAD_SHARDED_GROUP}"
    ] == 1
    assert restorer.live_attachments == 0


def test_a_restorer_refuses_a_shard_index_outside_its_group():
    harness = _GroupHarness(shard_count=2)
    with pytest.raises(TrajectoryRestoreError) as excinfo:
        harness.restorer(1, attention_group_size=1)
    assert "outside a 1-rank owning group" in str(excinfo.value)
    with pytest.raises(TrajectoryRestoreError) as excinfo:
        harness.restorer(0, attention_group_size=0)
    assert "attention_group_size must be positive" in str(excinfo.value)


@pytest.mark.parametrize("shard_index", [1, 2, 3])
def test_a_follower_restores_its_shard_from_the_mirrored_decision(shard_index):
    """A rank with NO registry restores from the leader's mirrored record.

    The follower keeps no record of its own precisely so a leader-side eviction
    cannot leave it restoring freed pages; the decision and the arena both come
    from the leader, so hit/miss cannot differ across the group.
    """

    harness = _GroupHarness(shard_count=4)
    _slots, expected = harness.seed()
    arena, installed = harness.commit_pass()

    restorer = harness.restorer(shard_index)
    outcome = restorer.restore_mirrored_shard(
        sequence_id=9,
        arena=arena,
        layout=harness.layout,
        session_key=installed.session_key,
        boundary_tokens=len(BASE_PROMPT),
        lcp_tokens=len(BASE_PROMPT),
        owning_group_id=0,
        serving_group_id=0,
        compute_cached_tokens=len(BASE_PROMPT),
    )

    assert outcome is not None
    assert outcome.session_key == installed.session_key
    restored = _slot_chunks(harness.managers[shard_index], outcome.state_slot)
    assert restored == expected[shard_index]
    counters = restorer.counters()
    assert counters["trajectory_restores"] == 1
    assert counters["trajectory_declines"] == 0
    # No attachment was ever taken, so none can leak.
    assert restorer.live_attachments == 0


def test_the_mirrored_path_names_the_same_reasons_as_the_attachment_path(caplog):
    harness = _GroupHarness(shard_count=4)
    harness.seed()
    arena, installed = harness.commit_pass()
    restorer = harness.restorer(1)

    # Wrong serving group, on a row that is not planned at the boundary:
    # declined, exactly as on the attachment path.
    assert (
        restorer.restore_mirrored_shard(
            sequence_id=9,
            arena=arena,
            layout=harness.layout,
            session_key=installed.session_key,
            boundary_tokens=len(BASE_PROMPT),
            lcp_tokens=len(BASE_PROMPT),
            owning_group_id=0,
            serving_group_id=3,
            compute_cached_tokens=0,
        )
        is None
    )
    # Suffix that does not start exactly at the boundary: planner drift,
    # counted under the same name and raised, as on the attachment path.
    with pytest.raises(TrajectoryRestorePlanDrift, match="disagree"):
        restorer.restore_mirrored_shard(
            sequence_id=9,
            arena=arena,
            layout=harness.layout,
            session_key=installed.session_key,
            boundary_tokens=len(BASE_PROMPT),
            lcp_tokens=len(BASE_PROMPT),
            owning_group_id=0,
            serving_group_id=0,
            compute_cached_tokens=len(BASE_PROMPT) - 1,
        )
    counters = restorer.counters()
    assert counters[f"trajectory_decline_{DECLINE_GROUP_MISMATCH}"] == 1
    planned = [r.message for r in caplog.records if "reason=group_mismatch" in r.message]
    assert len(planned) == 1 and "planned_offgroup=true" in planned[0]
    assert counters[f"trajectory_decline_{DECLINE_SUFFIX_OFFSET}"] == 1
    assert counters["trajectory_restores"] == 0
    assert restorer.live_attachments == 0


def test_the_mirrored_path_refuses_an_unknown_reason():
    harness = _GroupHarness(shard_count=2)
    restorer = harness.restorer(1)
    with pytest.raises(TrajectoryAttachmentInvalid):
        restorer._decline_mirrored(
            sequence_id=9,
            session_key="s",
            boundary_tokens=8,
            reason="not_a_reason",
            detail="",
        )


def test_the_g_wide_arena_holds_every_shard_and_only_those_pages():
    harness = _GroupHarness(shard_count=4)
    harness.seed()
    arena, _installed = harness.commit_pass()

    one_shard = harness.layout.shard_page_count(
        host_page_bytes_all_layers=harness.host_page_bytes_all_layers,
        prefix_page_count=2,
    )
    assert arena.pages_per_group == 4 * one_shard
    # Retained exactly once, by the leader, and every arena page is resident.
    view = harness.views[0]
    assert set(arena.page_ids_by_group[0]) <= view.resident_pages
    # Every window carries real bytes: no shard was left zeroed.
    for shard_index in range(4):
        window = arena.shard_window(shard_index)
        blob = b"".join(
            view.page_bytes_of(page) for page in window[0]
        )
        assert blob.strip(b"\x00"), f"shard {shard_index} window is all zeros"


# ---------------------------------------------------------------------------
# The plan decides; the restore checks the hand-off
# ---------------------------------------------------------------------------
#
# The worker's two restore seams, compiled out of ``batchgen_worker.py`` onto
# a fake that holds only what they read, driving the REAL restorer over the
# real committed frontier above. Scaled to this harness: a one-page pool
# chain stands for the 64-token system page and the two-page frontier for the
# 5120-token boundary of the original failure.


def _worker_seam(harness, restorer, *, compute_cached, planned):
    from types import SimpleNamespace

    from batchgen.prefix_reuse.prefill import PrefixCacheSequenceState

    names = (
        "_settle_trajectory_restore_plan",
        "_restore_trajectory_frontiers",
        "_get_trajectory_restorer",
    )
    worker_source = (
        pathlib.Path(__file__).resolve().parents[2]
        / "batchgen"
        / "batchgen_worker.py"
    )
    nodes = [
        node
        for node in ast.walk(ast.parse(worker_source.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert sorted(node.name for node in nodes) == sorted(names)
    namespace = {}
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(worker_source), "exec"),
        namespace,
    )
    from tests.worker.worker_fakes import attach_prefix_state, attach_trajectory_state
    seq = SimpleNamespace(uuid="u9", global_idx=9, decode_dp_group=0)
    worker = SimpleNamespace(
        rank=0,
        global_batch=SimpleNamespace(get_sequence={"u9": seq}.get),
        _local_to_uuid_map={0: "u9"},
        _host_views_by_group=lambda: {},
        _next_pool_segment_sid=lambda: -1,
    )
    attach_trajectory_state(
        worker,
        trajectory_frontier_committer=harness.committer,
        trajectory_restorer=restorer,
        trajectory_mirrored_restores={},
        trajectory_attachments={9: _hit(harness, restorer)},
    )
    attach_prefix_state(worker, prefix_sequence_states={
        9: PrefixCacheSequenceState(
            lookup_result=None,
            attached_tokens=compute_cached,
            compute_cached_tokens=compute_cached,
            trajectory_restore_boundary=planned,
        )
    })
    # Aliases kept for the assertions below.
    worker._trajectory_attachments = worker._traj.trajectory_attachments
    worker._prefix_sequence_states = worker._prefix.prefix_sequence_states
    for name in names:
        setattr(worker, name, types.MethodType(namespace[name], worker))
    return worker, seq


def test_restore_applies_when_the_plan_raised_the_start_to_the_boundary(caplog):
    harness = _Harness()
    _slot, expected, _ = _commit_seeded_frontier(harness, sequence_id=7)
    restorer = harness.restorer()
    boundary = len(BASE_PROMPT)
    worker, seq = _worker_seam(
        harness, restorer, compute_cached=boundary, planned=boundary
    )
    attachment = worker._trajectory_attachments[9]

    # Plan time: the row is planned, so nothing is declined.
    worker._settle_trajectory_restore_plan([seq], {}, "unused")
    assert worker._trajectory_attachments == {9: attachment}
    # Restore time: the plan's decision is applied at exactly the boundary.
    with caplog.at_level("INFO"):
        worker._restore_trajectory_frontiers([0])

    counters = restorer.counters()
    assert counters["trajectory_restores"] == 1
    assert counters["trajectory_declines"] == 0
    assert restorer.live_attachments == 0
    slot = harness.manager.get_sequence_state_item(9)
    assert _slot_chunks(harness.manager, slot) == expected
    assert f"suffix prefill resumes at token {boundary}" in caplog.text


def test_the_pre_fix_ladder_row_is_declined_at_plan_time_and_never_restored(caplog):
    """Pool chain one page, boundary two: the case that used to reach restore.

    It used to keep its attachment, start its prefill at the pool chain, and
    have the RESTORE decline it on ``suffix_offset_mismatch``. The plan now
    declines it by name when it drops the row, and the restore never sees it.
    """

    harness = _Harness()
    _commit_seeded_frontier(harness, sequence_id=7)
    restorer = harness.restorer()
    worker, seq = _worker_seam(
        harness, restorer, compute_cached=PAGE_TOKENS, planned=0
    )
    attachment = worker._trajectory_attachments[9]
    record = attachment.record
    restore_calls = []
    real_restore = restorer.restore

    def _counting_restore(*args, **kwargs):
        restore_calls.append(args)
        return real_restore(*args, **kwargs)

    restorer.restore = _counting_restore

    with caplog.at_level("WARNING"):
        worker._settle_trajectory_restore_plan(
            [seq], {0: "the serve group could not confirm the mirrored frontier set"},
            "unused",
        )
    assert worker._trajectory_attachments == {}
    assert attachment.consumed is True
    assert restorer.live_attachments == 0
    assert record.refcount == 1  # the frontier is evictable again
    worker._restore_trajectory_frontiers([0])

    assert restore_calls == []
    counters = restorer.counters()
    assert counters[f"trajectory_decline_{DECLINE_NOT_PLANNED}"] == 1
    assert counters[f"trajectory_decline_{DECLINE_SUFFIX_OFFSET}"] == 0
    assert counters["trajectory_restores"] == 0
    assert harness.manager.has_sequence_state_item(9) is False
    assert (
        f"reason={DECLINE_NOT_PLANNED}" in caplog.text
        and "at plan time: the serve group could not confirm" in caplog.text
    )


def test_a_prompt_diverging_below_the_boundary_never_becomes_an_attachment():
    """Token identity's first half: the lookup, not the restore, decides it."""

    harness = _Harness()
    _commit_seeded_frontier(harness)
    restorer = harness.restorer()
    diverged = BASE_PROMPT[:-1] + (999, 201, 202)  # differs at token B-1
    assert (
        restorer.lookup(
            namespace_digest=NAMESPACE, prompt_token_ids=diverged, sequence_id=9
        )
        is None
    )
    assert restorer.live_attachments == 0


def test_an_attachment_short_of_its_boundary_is_refused_without_a_slot_write():
    """Token identity's second half: an attachment no lookup could produce."""

    harness = _Harness()
    _commit_seeded_frontier(harness, sequence_id=7)
    restorer = harness.restorer()
    attachment = _hit(harness, restorer)
    record = attachment.record
    attachment._lcp_tokens = len(BASE_PROMPT) - 1  # fabricated
    free_before = harness.manager.get_stats().num_free_state_items

    with pytest.raises(TrajectoryAttachmentInvalid, match="fewer than its"):
        restorer.restore(
            attachment,
            serving_group_id=0,
            compute_cached_tokens=len(BASE_PROMPT),
        )
    assert attachment.consumed is True
    assert restorer.live_attachments == 0
    assert record.refcount == 1
    assert harness.manager.has_sequence_state_item(9) is False
    assert harness.manager.get_stats().num_free_state_items == free_before
    assert restorer.counters()["trajectory_restores"] == 0


def test_a_mirrored_restore_short_of_its_boundary_is_refused():
    harness = _GroupHarness(shard_count=2)
    harness.seed()
    arena, installed = harness.commit_pass()
    restorer = harness.restorer(1)
    with pytest.raises(TrajectoryAttachmentInvalid, match="fewer than its"):
        restorer.restore_mirrored_shard(
            sequence_id=9,
            arena=arena,
            layout=harness.layout,
            session_key=installed.session_key,
            boundary_tokens=len(BASE_PROMPT),
            lcp_tokens=len(BASE_PROMPT) - 1,
            owning_group_id=0,
            serving_group_id=0,
            compute_cached_tokens=len(BASE_PROMPT),
        )
    assert restorer.counters()["trajectory_restores"] == 0
    assert harness.managers[1].has_sequence_state_item(9) is False


# ---------------------------------------------------------------------------
# A PLANNED row's restore never declines silently
# ---------------------------------------------------------------------------
#
# The wave plan sets a restoring row's compute start to exactly the snapshot
# boundary and builds the wave around it (its chunk items begin there, its
# prefix is a materialized segment no row computes), so a refusal at apply
# time cannot fall back to a recompute: before this fix every refusal below
# returned ``None`` and the row's suffix ran on a state without its prefix.
# Each one is now counted under its own name AND raised, with the attachment
# consumed first. A row whose compute does NOT start at the boundary does not
# depend on the snapshot, and keeps its decline.

# The core's failure points, by the reason each reports and whether a slot
# was bound when it failed.
CORE_FAILURES = {
    "arena_read": (DECLINE_ARENA_READ_FAILED, False),
    "slots_exhausted": (DECLINE_SLOTS_EXHAUSTED, False),
    "has_slot": (DECLINE_SLOT_WRITE_FAILED, False),
    "bind": (DECLINE_SLOT_WRITE_FAILED, False),
    "write": (DECLINE_SLOT_WRITE_FAILED, True),
}


def _inject_core_failure(name, restorer, manager):
    """Make the restore core of ``restorer`` fail at point ``name``."""

    def _raise(message):
        def _boom(*args, **kwargs):
            raise RuntimeError(message)

        return _boom

    if name == "arena_read":
        def _corrupt(arena, *, layout, shard_index=0):
            raise SnapshotArenaCorrupt("arena read exploded")

        restorer._arena_reader.read = _corrupt
    elif name == "slots_exhausted":
        filler = 1000
        while manager.get_stats().num_free_state_items:
            manager.allocate_state_item(filler)
            filler += 1
    elif name == "has_slot":
        restorer._state_writer.has_slot = _raise("slot lookup exploded")
    elif name == "bind":
        restorer._state_writer.allocate_slot = _raise(
            "GPU page table is not initialized"
        )
    elif name == "write":
        restorer._state_writer.write_slot = _raise("slot write exploded")
    else:  # pragma: no cover - a typo in the table
        raise AssertionError(name)


@pytest.mark.parametrize("name", sorted(CORE_FAILURES))
def test_every_core_refusal_of_a_planned_row_raises_with_its_fields(name, caplog):
    harness = _Harness()
    _commit_seeded_frontier(harness, sequence_id=7)
    restorer = harness.restorer()
    attachment = _hit(harness, restorer)
    record = attachment.record
    _inject_core_failure(name, restorer, harness.manager)
    free_before = harness.manager.get_stats().num_free_state_items
    reason, bound = CORE_FAILURES[name]

    with caplog.at_level("WARNING"):
        with pytest.raises(TrajectoryPlannedRestoreFailed) as excinfo:
            restorer.restore(
                attachment,
                serving_group_id=0,
                compute_cached_tokens=len(BASE_PROMPT),
            )

    error = excinfo.value
    assert isinstance(error, TrajectoryRestoreError)
    assert error.reason == reason
    assert error.rank == 0
    assert error.sequence_id == 9
    assert error.session_key == attachment.session_key
    assert error.boundary_tokens == len(BASE_PROMPT)
    assert (error.state_slot is not None) is bound
    assert reason in str(error) and attachment.session_key in str(error)
    # Counted once under its own name, and nothing restored.
    counters = restorer.counters()
    assert counters[f"trajectory_decline_{reason}"] == 1
    assert counters["trajectory_declines"] == 1
    assert counters["trajectory_restores"] == 0
    # Consumed before the raise: no leaked reference, no leaked slot.
    assert attachment.consumed is True
    assert restorer.live_attachments == 0
    assert record.refcount == 1
    assert harness.manager.has_sequence_state_item(9) is False
    assert harness.manager.get_stats().num_free_state_items == free_before
    text = caplog.text
    assert f"PLANNED_RESTORE_FAILED reason={reason}" in text
    assert "full recompute" not in text


@pytest.mark.parametrize("refusal", ["head_sharded_group", "group_mismatch"])
def test_a_shard_or_group_refusal_raises_only_for_a_planned_row(refusal, caplog):
    """Same refusal, two rows: the one starting at the boundary raises.

    The row starting at 0 does not depend on the snapshot -- it recomputes
    its whole prompt -- so its decline stays a decline.
    """

    harness = _Harness()
    _commit_seeded_frontier(harness, sequence_id=7)
    group_size = 2 if refusal == "head_sharded_group" else 1
    serving = 1 if refusal == "group_mismatch" else 0
    restorer = harness.restorer(attention_group_size=group_size)

    unplanned = _hit(harness, restorer, sequence_id=10)
    with caplog.at_level("WARNING"):
        assert (
            restorer.restore(
                unplanned, serving_group_id=serving, compute_cached_tokens=0
            )
            is None
        )
    assert unplanned.consumed is True
    assert "full recompute" in caplog.text
    caplog.clear()

    planned = _hit(harness, restorer, sequence_id=9)
    with caplog.at_level("WARNING"):
        with pytest.raises(TrajectoryPlannedRestoreFailed) as excinfo:
            restorer.restore(
                planned,
                serving_group_id=serving,
                compute_cached_tokens=len(BASE_PROMPT),
            )
    assert excinfo.value.reason == refusal
    assert excinfo.value.sequence_id == 9
    assert excinfo.value.state_slot is None
    assert planned.consumed is True
    assert f"PLANNED_RESTORE_FAILED reason={refusal}" in caplog.text
    if refusal == "group_mismatch":
        # The off-group attribution rides the failure line too.
        assert "planned_offgroup=true" in caplog.text
    counters = restorer.counters()
    assert counters[f"trajectory_decline_{refusal}"] == 2
    assert counters["trajectory_restores"] == 0
    assert restorer.live_attachments == 0
    assert harness.manager.has_sequence_state_item(9) is False


@pytest.mark.parametrize(
    "reason",
    [
        DECLINE_ARENA_MISSING,
        DECLINE_ARENA_READ_FAILED,
        DECLINE_SLOTS_EXHAUSTED,
        DECLINE_SLOT_WRITE_FAILED,
        DECLINE_NOT_PLANNED,
    ],
)
def test_the_plan_time_decline_api_still_declines_every_reason(reason, caplog):
    """An attachment the plan does not restore is released, never raised.

    That is the path every unplanned row leaves through (``not_planned`` at
    plan time); the decline API itself keeps accepting every named reason.
    """

    harness = _Harness()
    _commit_seeded_frontier(harness, sequence_id=7)
    restorer = harness.restorer()
    attachment = _hit(harness, restorer)
    with caplog.at_level("WARNING"):
        restorer.decline(attachment, reason=reason, detail="at plan time: x")
    assert attachment.consumed is True
    assert restorer.live_attachments == 0
    assert restorer.counters()[f"trajectory_decline_{reason}"] == 1
    assert f"FALLBACK reason={reason}" in caplog.text
    assert "PLANNED_RESTORE_FAILED" not in caplog.text


def _mirrored(restorer, harness, arena, installed, **overrides):
    kwargs = dict(
        sequence_id=9,
        arena=arena,
        layout=harness.layout,
        session_key=installed.session_key,
        boundary_tokens=len(BASE_PROMPT),
        lcp_tokens=len(BASE_PROMPT),
        owning_group_id=0,
        serving_group_id=0,
        compute_cached_tokens=len(BASE_PROMPT),
    )
    kwargs.update(overrides)
    return restorer.restore_mirrored_shard(**kwargs)


@pytest.mark.parametrize("name", sorted(CORE_FAILURES))
def test_every_core_refusal_of_a_planned_mirrored_row_raises_the_same(name):
    """The follower path: same reasons, same counters, same raise."""

    harness = _GroupHarness(shard_count=2)
    harness.seed()
    arena, installed = harness.commit_pass()
    restorer = harness.restorer(1)
    manager = harness.managers[1]
    _inject_core_failure(name, restorer, manager)
    free_before = manager.get_stats().num_free_state_items
    reason, bound = CORE_FAILURES[name]

    with pytest.raises(TrajectoryPlannedRestoreFailed) as excinfo:
        _mirrored(restorer, harness, arena, installed)

    error = excinfo.value
    assert error.reason == reason
    assert error.rank == 1
    assert error.sequence_id == 9
    assert error.session_key == installed.session_key
    assert error.boundary_tokens == len(BASE_PROMPT)
    assert (error.state_slot is not None) is bound
    counters = restorer.counters()
    assert counters[f"trajectory_decline_{reason}"] == 1
    assert counters["trajectory_restores"] == 0
    assert restorer.live_attachments == 0
    assert manager.has_sequence_state_item(9) is False
    assert manager.get_stats().num_free_state_items == free_before


@pytest.mark.parametrize("refusal", ["head_sharded_group", "group_mismatch"])
def test_a_mirrored_shard_or_group_refusal_raises_only_for_a_planned_row(refusal):
    harness = _GroupHarness(shard_count=2)
    harness.seed()
    arena, installed = harness.commit_pass()
    group_size = 4 if refusal == "head_sharded_group" else 2
    serving = 1 if refusal == "group_mismatch" else 0
    restorer = harness.restorer(1, attention_group_size=group_size)

    assert (
        _mirrored(
            restorer, harness, arena, installed,
            serving_group_id=serving, compute_cached_tokens=0,
        )
        is None
    )
    with pytest.raises(TrajectoryPlannedRestoreFailed) as excinfo:
        _mirrored(restorer, harness, arena, installed, serving_group_id=serving)
    assert excinfo.value.reason == refusal
    assert restorer.counters()[f"trajectory_decline_{refusal}"] == 2
    assert harness.managers[1].has_sequence_state_item(9) is False


# -- the worker seam ----------------------------------------------------------


def test_the_seam_raises_a_planned_rows_capacity_refusal(caplog):
    """Through the real seam: the error reaches the prefill path, not None."""

    harness = _Harness(slots=2)
    _commit_seeded_frontier(harness, sequence_id=7)
    harness.manager.allocate_state_item(8)
    assert harness.manager.get_stats().num_free_state_items == 0
    restorer = harness.restorer()
    boundary = len(BASE_PROMPT)
    worker, seq = _worker_seam(
        harness, restorer, compute_cached=boundary, planned=boundary
    )
    attachment = worker._trajectory_attachments[9]
    worker._settle_trajectory_restore_plan([seq], {}, "unused")
    assert worker._trajectory_attachments == {9: attachment}

    with pytest.raises(TrajectoryPlannedRestoreFailed) as excinfo:
        worker._restore_trajectory_frontiers([0])

    assert excinfo.value.reason == DECLINE_SLOTS_EXHAUSTED
    assert excinfo.value.sequence_id == 9
    assert worker._trajectory_attachments == {}
    assert attachment.consumed is True
    assert restorer.live_attachments == 0
    assert harness.manager.has_sequence_state_item(9) is False


def test_the_seam_raises_a_planned_row_with_no_restore_source():
    """A planned row whose attachment is gone would run with no state."""

    harness = _Harness()
    _commit_seeded_frontier(harness, sequence_id=7)
    restorer = harness.restorer()
    boundary = len(BASE_PROMPT)
    worker, _seq = _worker_seam(
        harness, restorer, compute_cached=boundary, planned=boundary
    )
    restorer.release(worker._trajectory_attachments.pop(9))

    with pytest.raises(TrajectoryPlannedRestoreFailed) as excinfo:
        worker._restore_trajectory_frontiers([0])

    error = excinfo.value
    assert error.reason == FAIL_NO_RESTORE_SOURCE
    assert error.sequence_id == 9
    assert error.boundary_tokens == boundary
    assert restorer.counters()["trajectory_restores"] == 0
    assert harness.manager.has_sequence_state_item(9) is False


def test_the_seam_leaves_an_unplanned_row_without_an_attachment_alone():
    """No plan, no attachment: nothing to restore and nothing to raise."""

    harness = _Harness()
    _commit_seeded_frontier(harness, sequence_id=7)
    restorer = harness.restorer()
    worker, _seq = _worker_seam(
        harness, restorer, compute_cached=PAGE_TOKENS, planned=0
    )
    restorer.release(worker._trajectory_attachments.pop(9))
    worker._restore_trajectory_frontiers([0])
    assert restorer.counters()["trajectory_restores"] == 0
    assert restorer.counters()["trajectory_declines"] == 0
