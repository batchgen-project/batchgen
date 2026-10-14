"""A DSA pool wave keeps both Host-KV regions on the same pages and refcounts."""

from types import SimpleNamespace

import torch

from batchgen.prefix_reuse.commit import release_evicted_prefix_pages
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
GROUPS = (0, 1)  # 0 MLA latent (primary region), 1 indexer keys (aux region)


def blk(n):
    return [n * 10 + k for k in range(B)]


A, A1, A2 = blk(1) + blk(2), blk(3), blk(4)
TREE = [
    A + A1 + blk(70) + [900],
    A + A1 + blk(71) + [901],
    A + A2 + blk(72) + [902],
    A + A2 + blk(73) + [903],
    blk(74) + [904],
]


class FakeHostView:
    """One Host-KV region: a free list plus the ownership the prefix cache needs."""

    def __init__(self, first_page, num_pages=256):
        self.free = list(range(first_page, first_page + num_pages))
        self.tables = {}  # sequence id -> its page table
        self.owner = {}  # page -> owning sequence id
        self.resident = set()  # pages the prefix cache owns

    def free_pages(self):
        return len(self.free)

    def register_sequences(self, ids):
        for sid in ids:
            assert sid not in self.tables
            self.tables[sid] = []

    def allocate_pages_for_sequences(self, requests):
        for sid, tokens in requests:
            for _ in range(-(-int(tokens) // B)):
                page = self.free.pop(0)
                self.owner[page] = sid
                self.tables[sid].append(page)

    def build_page_table(self, ids):
        return [list(self.tables[sid]) for sid in ids]

    def attach_shared_prefix_pages(self, sid, pages):
        self.tables[sid] = list(pages) + self.tables[sid]

    def retain_sequence_pages(self, sid, pages):
        for page in pages:
            assert self.owner.pop(page) == sid, page
            self.resident.add(page)
        return list(pages)

    def release_sequence_pages(self, ids):
        for sid in ids:
            for page in self.tables.pop(sid, []):
                if self.owner.get(page) == sid:
                    del self.owner[page]
                    self.free.append(page)

    def release_resident_pages(self, pages):
        for page in pages:
            self.resident.remove(page)
            self.free.append(page)


class FakeCoordinator:
    """Block-chain prefix index storing one page per group; handles are counted."""

    def __init__(self):
        self.nodes = {}  # token prefix tuple -> {group id: page}
        self.open = set()
        self.next_handle = 1

    def _handle(self):
        handle = self.next_handle
        self.next_handle += 1
        self.open.add(handle)
        return handle

    def commit_prefix_page_ids(self, digest, token_ids, commit_tokens, groups, protect):
        pages_by_group = dict(groups)
        assert sorted(pages_by_group) == list(GROUPS)
        inserted = []
        for block in range(commit_tokens // B):
            key = tuple(token_ids[:(block + 1) * B])
            if key not in self.nodes:
                self.nodes[key] = {
                    group: pages[block] for group, pages in pages_by_group.items()
                }
                inserted.append(block)
        return SimpleNamespace(
            inserted_nodes=len(inserted),
            inserted_group_pages=[
                SimpleNamespace(
                    group_id=group, pages=[pages[block] for block in inserted]
                )
                for group, pages in sorted(pages_by_group.items())
            ] if inserted else [],
            active_attachment_handle=self._handle() if inserted else 0,
        )

    def lookup_and_attach(self, digest, tokens):
        matched = 0
        while tuple(tokens[:(matched + 1) * B]) in self.nodes and (matched + 1) * B <= len(tokens):
            matched += 1
        return SimpleNamespace(
            common_cached_tokens=matched * B,
            attachment_handle=self._handle() if matched else 0,
        )

    def release_attachment(self, handle):
        self.open.remove(handle)

    def resident_pages_by_group(self):
        pages = {group: [] for group in GROUPS}
        for group_pages in self.nodes.values():
            for group, page in group_pages.items():
                pages[group].append(page)
        return pages


def make_wave():
    plan = plan_wave_prefix_sharing(TREE, block_tokens=B, pool_pages=100, chunk_tokens=1000)
    pool = PrefillPrefixPool(num_layers=1, num_pages=200, page_tokens=B, num_kv_heads=1,
                             head_dim=2, dtype=torch.float32, device="cpu")
    return PoolWave(plan=plan, prompts=[list(p) for p in TREE],
                    uuids=[f"u{i}" for i in range(len(TREE))],
                    global_ids=list(range(len(TREE))),
                    page_tokens=pool.page_tokens, total_pages=pool.num_pages,
                    pool=pool, executor=PoolChunkExecutor(plan, TREE, pool))


def sids():
    counter = iter(range(1, 1000))
    return lambda: -next(counter)


def views():
    # Distinct page id ranges: the two regions allocate independently.
    return {0: FakeHostView(1000), 1: FakeHostView(5000)}


def test_segment_pages_are_mirrored_in_every_group():
    wave, host_views = make_wave(), views()
    initial = {group: view.free_pages() for group, view in host_views.items()}

    allocated = allocate_segment_host_pages(wave, host_views, sids())

    assert allocated == sum(seg.tokens for seg in wave.plan.segments) // B
    for group, view in host_views.items():
        assert initial[group] - view.free_pages() == allocated
        assert sorted(view.tables) == sorted(wave.segment_sids.values())
    for seg in wave.plan.segments:
        pages = wave.segment_host_pages[seg.segment_id]
        assert sorted(pages) == list(GROUPS)
        assert len(pages[0]) == len(pages[1]) == seg.tokens // B
        assert not set(pages[0]) & set(pages[1])
    for index in range(len(TREE)):
        chain = wave.chain_host_pages(index)
        if wave.chains[index]:
            assert len(chain[0]) == len(chain[1]) == wave.chain_tokens(index) // B
        else:
            assert chain == {}


def test_two_group_pool_wave_returns_every_page_to_both_regions():
    wave, host_views, coord = make_wave(), views(), FakeCoordinator()
    initial = {group: view.free_pages() for group, view in host_views.items()}
    allocated = allocate_segment_host_pages(wave, host_views, sids())

    segment_handles = commit_pool_segments(
        wave,
        coordinator=coord,
        worker_views_by_group=host_views,
        namespace_digest=DIGEST,
        publish_boundary_tokens=B,
        max_scan_nodes=100,
    )
    members = attach_pool_members(wave, coordinator=coord, namespace_digest=DIGEST)
    for handle in segment_handles:
        coord.release_attachment(handle)

    # Both regions retained exactly the pages the coordinator published.
    resident = coord.resident_pages_by_group()
    for group, view in host_views.items():
        assert view.resident == set(resident[group])
    assert sorted(members) == [0, 1, 2, 3]  # every prompt with a chain

    # Members leave, then the last reader of each pseudo sequence releases it.
    for handle in members.values():
        coord.release_attachment(handle)
    for sid in wave.segment_sids.values():
        for view in host_views.values():
            view.release_sequence_pages([sid])
    for group, view in host_views.items():
        # Still held by the prefix cache, in both regions, not leaked.
        assert initial[group] - view.free_pages() == allocated

    release_evicted_prefix_pages(
        eviction_result=SimpleNamespace(
            evicted_group_pages=[
                SimpleNamespace(group_id=group, pages=pages)
                for group, pages in sorted(resident.items())
            ]
        ),
        worker_views_by_group=host_views,
    )

    assert {group: view.free_pages() for group, view in host_views.items()} == initial
    assert all(not view.owner and not view.resident for view in host_views.values())
    assert coord.open == set()
