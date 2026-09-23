"""Host side of the in-wave prefix pool against fake Host KV and coordinator."""

from types import SimpleNamespace

import pytest
import torch

from batchgen.prefix_reuse.executor import PoolChunkExecutor
from batchgen.prefix_reuse.pool import PrefillPrefixPool
from batchgen.prefix_reuse.wave_plan import plan_wave_prefix_sharing
from batchgen.prefix_reuse.wave_runtime import (
    PoolWave,
    allocate_segment_host_pages,
    attach_pool_members,
    commit_pool_segments,
)

B = 4
DIGEST = (1, 2, 3, 4)


def blk(n):
    return [n * 10 + k for k in range(B)]


A, A1, A2, BB = blk(1) + blk(2), blk(3), blk(4), blk(5) + blk(6)
TREE = [
    A + A1 + blk(70) + [900],
    A + A1 + blk(71) + [901],
    A + A2 + blk(72) + [902],
    A + A2 + blk(73) + [903],
    BB + blk(74) + [904],
    BB + blk(75) + [905],
    blk(76) + [906],
]


class FakeWorkerView:
    def __init__(self):
        self.next_page = 100
        self.owner = {}  # page -> sequence id or "resident"
        self.tables = {}

    def register_sequences(self, ids):
        for sid in ids:
            assert sid not in self.tables
            self.tables[sid] = []

    def allocate_pages_for_sequences(self, requests):
        for sid, tokens in requests:
            for _ in range(-(-tokens // B)):
                self.owner[self.next_page] = sid
                self.tables[sid].append(self.next_page)
                self.next_page += 1

    def build_page_table(self, ids):
        return [list(self.tables[sid]) for sid in ids]

    def retain_sequence_pages(self, sid, pages):
        for page in pages:
            assert self.owner[page] == sid, (page, self.owner[page], sid)
            self.owner[page] = "resident"
        return list(pages)


class FakeCoordinator:
    """Block-chain prefix index; handles are counted to catch leaks."""

    def __init__(self):
        self.nodes = {}  # token prefix tuple -> page
        self.open = set()
        self.next_handle = 1
        self.commits = []

    def _handle(self):
        handle = self.next_handle
        self.next_handle += 1
        self.open.add(handle)
        return handle

    def commit_prefix_page_ids(self, digest, token_ids, commit_tokens, groups, protect):
        (group, pages), = groups
        inserted = []
        for block in range(commit_tokens // B):
            key = tuple(token_ids[:(block + 1) * B])
            if key not in self.nodes:
                self.nodes[key] = pages[block]
                inserted.append(pages[block])
        self.commits.append(tuple(token_ids[:commit_tokens]))
        return SimpleNamespace(
            inserted_nodes=len(inserted),
            inserted_group_pages=[SimpleNamespace(group_id=group, pages=inserted)] if inserted else [],
            active_attachment_handle=self._handle() if inserted else 0,
        )

    def lookup_and_attach(self, digest, tokens):
        matched = 0
        while tuple(tokens[:(matched + 1) * B]) in self.nodes and (matched + 1) * B <= len(tokens):
            matched += 1
        return SimpleNamespace(common_cached_tokens=matched * B, attachment_handle=self._handle() if matched else 0)

    def release_attachment(self, handle):
        self.open.remove(handle)


def make_wave():
    plan = plan_wave_prefix_sharing(TREE, block_tokens=B, pool_pages=100, chunk_tokens=1000)
    pool = PrefillPrefixPool(num_layers=1, num_pages=200, page_tokens=B, num_kv_heads=1,
                             head_dim=2, dtype=torch.float32, device="cpu")
    wave = PoolWave(plan=plan, prompts=[list(p) for p in TREE], uuids=[f"u{i}" for i in range(len(TREE))],
                    global_ids=list(range(len(TREE))), pool=pool, executor=PoolChunkExecutor(plan, TREE, pool))
    return wave


def sids():
    counter = iter(range(1, 1000))
    return lambda: -next(counter)


def commit_and_attach(wave, view, coord):
    handles = commit_pool_segments(wave, coordinator=coord, worker_view=view, namespace_digest=DIGEST,
                                   publish_boundary_tokens=B, max_scan_nodes=100)
    members = attach_pool_members(wave, coordinator=coord, namespace_digest=DIGEST)
    for handle in handles:
        coord.release_attachment(handle)
    return members


def test_segments_get_exact_pages_and_members_their_chain():
    wave, view = make_wave(), FakeWorkerView()
    allocated = allocate_segment_host_pages(wave, view, sids())
    assert allocated == sum(s.tokens for s in wave.plan.segments) // B
    a, a1 = wave.segment_host_pages[0], wave.segment_host_pages[1]
    assert wave.chains[0] == (0, 1)
    assert wave.chain_host_pages(0) == a + a1
    assert wave.segment_prefix_pages(1) == a + a1
    assert wave.chains[6] == ()


def test_commit_parents_first_retains_segment_pages_and_pins_members():
    wave, view, coord = make_wave(), FakeWorkerView(), FakeCoordinator()
    allocate_segment_host_pages(wave, view, sids())
    members = commit_and_attach(wave, view, coord)
    # parents first: a segment's parent segment was committed before it
    order = {commit: i for i, commit in enumerate(coord.commits)}
    for seg in wave.plan.segments:
        if seg.parent_id is not None:
            parent = wave.plan.segments[seg.parent_id]
            prompt = TREE[seg.members[0]]
            assert order[tuple(prompt[:parent.token_end])] < order[tuple(prompt[:seg.token_end])]
    assert all(view.owner[p] == "resident" for pages in wave.segment_host_pages.values() for p in pages)
    assert sorted(members) == [0, 1, 2, 3, 4, 5]  # every prompt with a chain
    assert coord.open == set(members.values())  # segment handles released, no leak


def test_segment_already_published_by_another_rank_stays_with_pseudo_id():
    wave, view, coord = make_wave(), FakeWorkerView(), FakeCoordinator()
    for block in range(len(A) // B):  # another rank published A with its own pages
        coord.nodes[tuple(A[:(block + 1) * B])] = 9000 + block
    allocate_segment_host_pages(wave, view, sids())
    commit_and_attach(wave, view, coord)
    a_sid = wave.segment_sids[0]
    assert all(view.owner[p] == a_sid for p in wave.segment_host_pages[0])  # not retained
    assert all(view.owner[p] == "resident" for p in wave.segment_host_pages[1])  # A1 inserted


def test_missing_chain_raises_without_leaking_handles():
    wave, view, coord = make_wave(), FakeWorkerView(), FakeCoordinator()
    allocate_segment_host_pages(wave, view, sids())
    handles = commit_pool_segments(wave, coordinator=coord, worker_view=view, namespace_digest=DIGEST,
                                   publish_boundary_tokens=B, max_scan_nodes=100)
    del coord.nodes[tuple(A + A1)]  # the A1 node vanished before the members attach
    with pytest.raises(RuntimeError, match="resident"):
        attach_pool_members(wave, coordinator=coord, namespace_digest=DIGEST)
    for handle in handles:
        coord.release_attachment(handle)
    assert coord.open == set()
