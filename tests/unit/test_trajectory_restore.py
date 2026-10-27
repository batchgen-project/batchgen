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
    DECLINE_REASONS,
    DECLINE_SLOTS_EXHAUSTED,
    DECLINE_SLOT_WRITE_FAILED,
    DECLINE_SUFFIX_OFFSET,
    HostKVSnapshotArenaReader,
    KDAStateSlotsExhausted,
    KDAStateSnapshotWriter,
    TrajectoryAttachmentInvalid,
    TrajectoryRestoreError,
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
# Fallbacks: every one releases its attachment and names its reason
# ---------------------------------------------------------------------------


def _hit(harness, restorer, *, sequence_id=9):
    attachment = restorer.lookup(
        namespace_digest=NAMESPACE,
        prompt_token_ids=BASE_PROMPT + (201, 202),
        sequence_id=sequence_id,
    )
    assert attachment is not None
    return attachment


def test_slot_exhaustion_falls_back_by_name_and_releases_the_attachment(caplog):
    """The fail-safe: no wait, no raise, and the row recomputes in full."""

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
        result = restorer.restore(
            attachment,
            serving_group_id=0,
            compute_cached_tokens=len(BASE_PROMPT),
        )

    assert result is None
    assert restorer.live_attachments == 0
    assert record.refcount == 1
    assert attachment.consumed is True
    # The sequence proceeds as a recompute: it holds NO restored slot.
    assert harness.manager.has_sequence_state_item(9) is False
    counters = restorer.counters()
    assert counters[f"trajectory_decline_{DECLINE_SLOTS_EXHAUSTED}"] == 1
    assert counters["trajectory_restores"] == 0
    message = "\n".join(record.getMessage() for record in caplog.records)
    assert "TRAJECTORY_RESTORE" in message
    assert f"reason={DECLINE_SLOTS_EXHAUSTED}" in message
    assert "full recompute" in message


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
    """The slot this restore took is handed back; the attachment is released."""

    harness = _Harness()
    _commit_seeded_frontier(harness, sequence_id=7)
    restorer = harness.restorer()
    attachment = _hit(harness, restorer)
    record = attachment.record
    free_before = harness.manager.get_stats().num_free_state_items

    def _boom(sequence_id, chunks):
        raise RuntimeError("slot write exploded")

    harness.state_writer.write_slot = _boom

    result = restorer.restore(
        attachment, serving_group_id=0, compute_cached_tokens=len(BASE_PROMPT)
    )

    assert result is None
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
    assert (
        restorer.restore(
            attachment,
            serving_group_id=0,
            compute_cached_tokens=len(BASE_PROMPT),
        )
        is None
    )
    # The row's own slot is NOT handed back by a restore that did not take it.
    assert harness.manager.has_sequence_state_item(9) is True
    assert harness.manager.get_sequence_state_item(9) == existing
    assert restorer.live_attachments == 0


def test_an_arena_read_failure_falls_back_and_takes_no_slot():
    harness = _Harness()
    _commit_seeded_frontier(harness, sequence_id=7)
    restorer = harness.restorer()
    attachment = _hit(harness, restorer)
    free_before = harness.manager.get_stats().num_free_state_items

    def _boom(arena, *, layout):
        raise SnapshotArenaCorrupt("arena read exploded")

    harness.arena_reader.read = _boom

    assert (
        restorer.restore(
            attachment,
            serving_group_id=0,
            compute_cached_tokens=len(BASE_PROMPT),
        )
        is None
    )
    assert harness.manager.get_stats().num_free_state_items == free_before
    assert harness.manager.has_sequence_state_item(9) is False
    assert restorer.live_attachments == 0
    assert (
        restorer.counters()[f"trajectory_decline_{DECLINE_ARENA_READ_FAILED}"]
        == 1
    )


def test_a_missing_arena_falls_back_by_name():
    harness = _Harness()
    _commit_seeded_frontier(harness, sequence_id=7)
    restorer = harness.restorer()
    attachment = _hit(harness, restorer)
    handle = attachment.storage_handle
    # The handle->arena table is the committer's; emptying it models the one
    # state that must never be read through: a record whose arena is gone.
    harness.committer._arenas_by_handle.pop(handle)

    assert (
        restorer.restore(
            attachment,
            serving_group_id=0,
            compute_cached_tokens=len(BASE_PROMPT),
        )
        is None
    )
    assert restorer.live_attachments == 0
    assert (
        restorer.counters()[f"trajectory_decline_{DECLINE_ARENA_MISSING}"] == 1
    )


def test_a_suffix_that_does_not_start_at_the_frontier_is_refused():
    """The offset identity: a full-prompt recompute must not be restored into."""

    harness = _Harness()
    _commit_seeded_frontier(harness, sequence_id=7)
    restorer = harness.restorer()
    for cached in (0, len(BASE_PROMPT) - 1, len(BASE_PROMPT) + 1):
        attachment = _hit(harness, restorer)
        assert (
            restorer.restore(
                attachment,
                serving_group_id=0,
                compute_cached_tokens=cached,
            )
            is None
        )
        assert restorer.live_attachments == 0
        assert harness.manager.has_sequence_state_item(9) is False
    assert (
        restorer.counters()[f"trajectory_decline_{DECLINE_SUFFIX_OFFSET}"] == 3
    )


def test_a_serving_group_other_than_the_owning_group_is_refused():
    harness = _Harness()
    _commit_seeded_frontier(harness, sequence_id=7)
    restorer = harness.restorer()
    attachment = _hit(harness, restorer)
    assert (
        restorer.restore(
            attachment,
            serving_group_id=1,
            compute_cached_tokens=len(BASE_PROMPT),
        )
        is None
    )
    assert restorer.live_attachments == 0
    assert (
        restorer.counters()[f"trajectory_decline_{DECLINE_GROUP_MISMATCH}"] == 1
    )


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
        assert (
            restorer.restore(
                attachment,
                serving_group_id=0,
                compute_cached_tokens=len(BASE_PROMPT),
            )
            is None
        )
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

    # A restore, a decline and a bare release, in one batch.
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
        restorer.restore(
            declined, serving_group_id=7, compute_cached_tokens=len(BASE_PROMPT)
        )
        is None
    )
    dropped = _hit(harness, restorer, sequence_id=11)
    restorer.release(dropped)

    counters = restorer.counters()
    assert counters["trajectory_live_attachments"] == 0
    assert counters["trajectory_hits"] == 3
    assert counters["trajectory_restores"] == 1
    assert counters["trajectory_declines"] == 1
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
    assert (
        restorer.restore(
            attachment,
            serving_group_id=0,
            compute_cached_tokens=len(BASE_PROMPT),
        )
        is None
    )
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


def test_the_mirrored_path_names_the_same_reasons_as_the_attachment_path():
    harness = _GroupHarness(shard_count=4)
    harness.seed()
    arena, installed = harness.commit_pass()
    restorer = harness.restorer(1)

    # Wrong serving group.
    assert (
        restorer.restore_mirrored_shard(
            sequence_id=9,
            arena=arena,
            layout=harness.layout,
            session_key=installed.session_key,
            boundary_tokens=len(BASE_PROMPT),
            owning_group_id=0,
            serving_group_id=3,
            compute_cached_tokens=len(BASE_PROMPT),
        )
        is None
    )
    # Suffix that does not start exactly at the boundary.
    assert (
        restorer.restore_mirrored_shard(
            sequence_id=9,
            arena=arena,
            layout=harness.layout,
            session_key=installed.session_key,
            boundary_tokens=len(BASE_PROMPT),
            owning_group_id=0,
            serving_group_id=0,
            compute_cached_tokens=len(BASE_PROMPT) - 1,
        )
        is None
    )
    counters = restorer.counters()
    assert counters[f"trajectory_decline_{DECLINE_GROUP_MISMATCH}"] == 1
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
