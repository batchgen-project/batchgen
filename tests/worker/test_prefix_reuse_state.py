import hashlib
import random
import sys
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import pytest

from batchgen.worker.prefix_reuse_state import (
    PrefixReuseState,
    PrefixSequenceStateMissingError,
    release_pool_segments,
)
from tests.worker.worker_fakes import attach_prefix_state, attach_trajectory_state


# copied verbatim from batchgen_worker.py:8358-8396 @ b0f5540e
def _old_prefix_page_digests(self, seq) -> Tuple[int, ...]:
	"""Cumulative page digests of this sequence's prompt, shareable pages.

	Digest ``k`` identifies the first ``k+1`` pages of the prompt: two
	sequences hold the same digest at ``k`` exactly when those pages carry the
	same tokens, hence the same KV, hence one physical copy in the pool. Using
	the digest itself as the ledger key therefore makes "same key" mean "same
	physical pages", which is what the charge requires.

	Deterministic across ranks: the digest is a function of the prompt
	tokens, which every rank reads from the node-shared trajectory store.
	"""
	cached = self._prefix_page_digest_cache.get(seq.global_idx)
	if cached is not None:
		return cached
	page_size = int(seq.PAGE_SIZE)
	# The final page is never shareable, so a fully shared prompt still
	# pays for the page holding the token it produces logits for.
	shareable = (int(seq.prompt_length) - 1) // page_size
	if shareable <= 0:
		self._prefix_page_digest_cache[seq.global_idx] = ()
		return ()
	tokens = self._trajectory_tokens(seq, length=seq.prompt_length)
	token_array = np.ascontiguousarray(
		np.asarray(tokens, dtype=np.int64)
	)
	digests = []
	running = b"\x00" * 16
	for page in range(shareable):
		block = token_array[page * page_size:(page + 1) * page_size]
		running = hashlib.blake2b(
			running + block.tobytes(), digest_size=16
		).digest()
		digests.append(
			int.from_bytes(running[:8], "big", signed=False)
		)
	chain = tuple(digests)
	self._prefix_page_digest_cache[seq.global_idx] = chain
	return chain


# copied verbatim from batchgen_worker.py:8410-8463 @ b0f5540e
def _old_admission_shared_prefix_segments(
	self, all_candidates: List[str]
) -> Dict[str, Tuple[Tuple[int, int], ...]]:
	"""Per candidate, the shared run it reads with an ADJACENT candidate.

	``_stamp_prefix_prefill_order`` sorts the admission by prompt token
	ids, which on a prefix tree is a depth-first walk: candidates sharing a
	prefix are contiguous. A candidate's shared run is therefore the
	longer of its page-granular common prefix with its two neighbours in
	admission order, and its key is the cumulative digest at the end of
	that run -- equal for exactly the candidates that share it.

	Conservative by construction: crediting only the deepest adjacent run
	under-counts a wave whose families themselves share an outer prefix
	(the outer run is charged once per family rather than once per wave),
	and under-counting can only admit fewer rows, never over-admit.
	"""
	if not self._shared_prefix_credit_enabled():
		return {}
	self._prune_prefix_page_digest_cache()
	ordered = sorted(
		all_candidates,
		key=lambda uuid: (
			self._admission_order_key(
				self.global_batch.get_sequence(uuid)
			),
			self.global_batch.get_sequence(uuid).global_idx,
		),
	)
	digests = [
		self._prefix_page_digests(self.global_batch.get_sequence(uuid))
		for uuid in ordered
	]

	def shared_pages(left: int, right: int) -> int:
		if left < 0 or right >= len(digests):
			return 0
		first, second = digests[left], digests[right]
		limit = min(len(first), len(second))
		index = 0
		while index < limit and first[index] == second[index]:
			index += 1
		return index

	segments: Dict[str, Tuple[Tuple[int, int], ...]] = {}
	for position, uuid in enumerate(ordered):
		run = max(
			shared_pages(position - 1, position),
			shared_pages(position, position + 1),
		)
		if run <= 0:
			continue
		segments[uuid] = ((digests[position][run - 1], run),)
	return segments


class _OldAdmissionHarness:
	def __init__(self, sequences):
		self.global_batch = SimpleNamespace(
			sequences=sequences, get_sequence=sequences.__getitem__
		)
		self._prefix_page_digest_cache = {}

	def _trajectory_tokens(self, seq, *, length):
		return np.asarray(seq.tokens[:length], dtype=np.int64)

	def _shared_prefix_credit_enabled(self):
		return True

	def _prune_prefix_page_digest_cache(self):
		if not self._prefix_page_digest_cache:
			return
		live = {seq.global_idx for seq in self.global_batch.sequences.values()}
		for global_idx in list(self._prefix_page_digest_cache):
			if global_idx not in live:
				del self._prefix_page_digest_cache[global_idx]

	def _admission_order_key(self, seq):
		order = getattr(seq, "prefill_order", None)
		return int(seq.global_idx if order is None else order)

	def _prefix_page_digests(self, seq):
		return _old_prefix_page_digests(self, seq)


def test_digest_walk_and_prune_are_deterministic():
    store = PrefixReuseState(rank=0, world_size=2)
    seqs = {
        "a": SimpleNamespace(uuid="a", global_idx=1, prompt_length=128, PAGE_SIZE=64, prefill_order=0),
        "b": SimpleNamespace(uuid="b", global_idx=2, prompt_length=128, PAGE_SIZE=64, prefill_order=1),
    }
    tokens = {"a": list(range(128)), "b": list(range(128))}
    assert store.admission_shared_prefix_segments(
        ["a", "b"],
        sequence_for=seqs.__getitem__,
        tokens_for=lambda seq: tokens[seq.uuid],
        credit_enabled=True,
    )

    # The test above uses the same algorithm with a minimal sequence object;
    # exercise the per-sequence digest directly for the frozen fixture.
    first = store.prefix_page_digests(
        global_id=1,
        prompt_length=128,
        page_size=64,
        tokens_for=lambda: tokens["a"],
    )
    second = store.prefix_page_digests(
        global_id=2,
        prompt_length=128,
        page_size=64,
        tokens_for=lambda: tokens["b"],
    )
    assert first == second and len(first) == 1
    store.prune_prefix_page_digest_cache([2])
    assert 1 not in store.prefix_page_digest_cache


def test_credit_latch_and_sid_counter():
    store = PrefixReuseState(rank=1, world_size=4)
    first = store.next_pool_segment_sid()
    second = store.next_pool_segment_sid()
    assert first < 0 and second < 0 and first != second
    store.shared_prefix_credit_applied = True
    assert store.suspend_shared_prefix_credit("test", rank=1)
    assert store.shared_prefix_credit_suspended
    assert not store.shared_prefix_credit_enabled(
        shared_prefix_admission=True,
        enable_prefix_cache=True,
        pool_enabled=True,
        prefill_pool_page_budget=1,
    )


def test_pool_refcounts_release_last_member():
    store = PrefixReuseState()
    store.pool_member_sids[9] = (-1,)
    store.pool_sid_members[-1] = 1

    class View:
        def __init__(self):
            self.released = []

        def release_sequence_pages(self, ids):
            self.released.append(("release", ids))

        def unregister_sequences(self, ids):
            self.released.append(("unregister", ids))

    view = View()
    freed = list(store.release_pool_segment_refs(9))
    assert freed == [-1]
    release_pool_segments({0: view}, freed)
    assert view.released == [("release", [-1]), ("unregister", [-1])]
    assert store.pool_sid_members == {}


# D-D/A6: frozen digest and admission walks match at boundary and random lengths.
def test_digest_walk_matches_frozen_fixture_at_boundary_lengths():
	page_size = 4
	lengths = [0, 1, page_size - 1, page_size, page_size + 1, 8, 12]
	for seed in range(7):
		rng = random.Random(seed)
		sequences = {}
		for index, length in enumerate(lengths):
			if index % 2 == 1:
				tokens = list(range(length))
			else:
				tokens = [rng.randrange(0, 1000) for _ in range(length)]
			sequences[f"u{index}"] = SimpleNamespace(
				uuid=f"u{index}",
				global_idx=index,
				prompt_length=length,
				PAGE_SIZE=page_size,
				prefill_order=index,
				tokens=tokens,
			)
		old = _OldAdmissionHarness(sequences)
		new = PrefixReuseState()
		for uuid, seq in sequences.items():
			assert _old_prefix_page_digests(old, seq) == new.prefix_page_digests(
				global_id=seq.global_idx,
				prompt_length=seq.prompt_length,
				page_size=seq.PAGE_SIZE,
				tokens_for=lambda seq=seq: seq.tokens,
			)
		assert _old_admission_shared_prefix_segments(old, list(sequences)) == new.admission_shared_prefix_segments(
			list(sequences),
			sequence_for=sequences.__getitem__,
			tokens_for=lambda seq: seq.tokens,
			credit_enabled=True,
		)


# D-E: the sid seed is eager and every generated sid is monotonic and unique.
def test_sid_counter_monotonicity_and_collision_guard_seed(monkeypatch):
	seed_ns = 1234567890123
	# Patch through the module object the class was defined in: a dotted string
	# re-resolves ``batchgen`` and breaks when another test left a mock there.
	owner_module = sys.modules[PrefixReuseState.__module__]
	monkeypatch.setattr(owner_module.time, "time_ns", lambda: seed_ns)
	store = PrefixReuseState(rank=2, world_size=8)
	assert store.pool_sid_counter == seed_ns // 1000
	first, second, third = (
		store.next_pool_segment_sid(),
		store.next_pool_segment_sid(),
		store.next_pool_segment_sid(),
	)
	assert len({first, second, third}) == 3
	assert first > second > third


# D-D: credit suspension and enablement transitions keep their latches distinct.
def test_credit_latch_transitions():
	store = PrefixReuseState()
	assert not store.suspend_shared_prefix_credit("without-credit", rank=0)
	store.shared_prefix_credit_applied = True
	assert store.suspend_shared_prefix_credit("credited-wave", rank=0)
	assert not store.shared_prefix_credit_applied
	assert store.shared_prefix_credit_suspended
	assert not store.shared_prefix_credit_enabled(
		shared_prefix_admission=True,
		enable_prefix_cache=True,
		pool_enabled=True,
		prefill_pool_page_budget=1,
	)
	store.shared_prefix_credit_suspended = False
	assert store.shared_prefix_credit_enabled(
		shared_prefix_admission=True,
		enable_prefix_cache=True,
		pool_enabled=True,
		prefill_pool_page_budget=1,
	)


# P3: resume is the pair of suspend — it clears only the suspension latch and
# leaves the per-wave applied latch to note_wave_credit.
def test_resume_shared_prefix_credit_clears_only_the_suspension_latch():
	store = PrefixReuseState()
	store.note_wave_credit(True)
	assert store.suspend_shared_prefix_credit("credited-wave", rank=0)
	assert store.shared_prefix_credit_suspended
	store.resume_shared_prefix_credit()
	assert not store.shared_prefix_credit_suspended
	assert not store.shared_prefix_credit_applied
	assert store.shared_prefix_credit_enabled(
		shared_prefix_admission=True,
		enable_prefix_cache=True,
		pool_enabled=True,
		prefill_pool_page_budget=1,
	)
	# Idempotent: resuming an unsuspended latch stays clear.
	store.resume_shared_prefix_credit()
	assert not store.shared_prefix_credit_suspended


# P3: note_wave_credit is a bare per-round setter of the applied latch; the
# value is collective-sourced, so each round's note overwrites the last.
def test_note_wave_credit_records_each_rounds_value():
	store = PrefixReuseState()
	assert not store.shared_prefix_credit_applied
	store.note_wave_credit(True)
	assert store.shared_prefix_credit_applied is True
	store.note_wave_credit(False)
	assert store.shared_prefix_credit_applied is False
	assert not store.shared_prefix_credit_suspended


# P2: the prefix_drain metric latch is edge-triggered — marked once per drain,
# re-armed by admission.
def test_drain_metric_latch_mark_and_rearm():
	store = PrefixReuseState()
	assert store.prefix_drain_metric_emitted is False
	store.mark_drain_metric_emitted()
	assert store.prefix_drain_metric_emitted is True
	# Idempotent until re-armed.
	store.mark_drain_metric_emitted()
	assert store.prefix_drain_metric_emitted is True
	store.rearm_drain_metric()
	assert store.prefix_drain_metric_emitted is False
	store.rearm_drain_metric()
	assert store.prefix_drain_metric_emitted is False


# D-E/A6: add and release paths keep pool sid refcounts until the last member.
def test_refcount_add_release_paths():
	store = PrefixReuseState()
	store.add_pool_member_refs(1, [-3, -4])
	store.add_pool_member_refs(2, [-3])
	assert store.pool_member_sids == {1: (-3, -4), 2: (-3,)}
	assert store.pool_sid_members == {-3: 2, -4: 1}

	class View:
		def __init__(self):
			self.calls = []

		def release_sequence_pages(self, ids):
			self.calls.append(("release", list(ids)))

		def unregister_sequences(self, ids):
			self.calls.append(("unregister", list(ids)))

	view = View()
	freed = list(store.release_pool_segment_refs(1))
	assert freed == [-4]
	release_pool_segments({0: view}, freed)
	assert store.pool_sid_members == {-3: 1}
	assert view.calls == [("release", [-4]), ("unregister", [-4])]
	freed = list(store.release_pool_segment_refs(2))
	assert freed == [-3]
	release_pool_segments({0: view}, freed)
	assert store.pool_sid_members == {}
	assert view.calls[-2:] == [("release", [-3]), ("unregister", [-3])]


# A member whose segments still have other members, or that has none, frees
# nothing, so its caller builds no Host views.
def test_release_returns_nothing_while_a_segment_has_members():
	store = PrefixReuseState()
	store.add_pool_member_refs(1, [-3])
	store.add_pool_member_refs(2, [-3])
	assert list(store.release_pool_segment_refs(1)) == []
	assert store.pool_sid_members == {-3: 1}
	assert list(store.release_pool_segment_refs(77)) == []
	assert store.pool_sid_members == {-3: 1}
	assert list(store.release_pool_segment_refs(2)) == [-3]


# Freed segments come out in the member's chain order, one at a time, and
# nothing is counted down until the iterator is consumed.
def test_release_yields_freed_segments_in_chain_order_when_consumed():
	store = PrefixReuseState()
	store.add_pool_member_refs(1, [-5, -6, -7])
	store.add_pool_member_refs(2, [-6])
	pending = store.release_pool_segment_refs(1)
	assert store.pool_member_sids == {1: (-5, -6, -7), 2: (-6,)}
	assert next(pending) == -5
	assert store.pool_sid_members == {-6: 2, -7: 1}
	assert list(pending) == [-7]
	assert store.pool_sid_members == {-6: 1}
	assert store.pool_member_sids == {2: (-6,)}


# A chain that names a segment without a count fails on that segment only:
# the segment freed before it has already been handed out for release.
def test_release_hands_out_a_freed_segment_before_a_later_one_fails():
	store = PrefixReuseState()
	store.pool_member_sids[9] = (-1, -2)
	store.pool_sid_members[-1] = 1
	released = []
	with pytest.raises(KeyError):
		for sid in store.release_pool_segment_refs(9):
			released.append(sid)
	assert released == [-1]
	assert store.pool_sid_members == {}


# One segment at a time; for each segment every view releases, then unregisters.
def test_release_pool_segments_order_across_views():
	calls = []

	class View:
		def __init__(self, name):
			self.name = name

		def release_sequence_pages(self, ids):
			calls.append((self.name, "release", list(ids)))

		def unregister_sequences(self, ids):
			calls.append((self.name, "unregister", list(ids)))

	release_pool_segments({0: View(0), 1: View(1)}, [-1, -2])
	assert calls == [
		(0, "release", [-1]), (0, "unregister", [-1]),
		(1, "release", [-1]), (1, "unregister", [-1]),
		(0, "release", [-2]), (0, "unregister", [-2]),
		(1, "release", [-2]), (1, "unregister", [-2]),
	]
	release_pool_segments({0: View(0)}, [])
	assert len(calls) == 8


def test_attach_helpers_raise_on_unknown_override():
	worker = SimpleNamespace()
	with pytest.raises(AttributeError):
		attach_trajectory_state(worker, unknown_trajectory_field=1)
	with pytest.raises(AttributeError):
		attach_prefix_state(worker, unknown_prefix_field=1)


def test_install_state_is_a_bare_setter_and_overwrite_is_legal():
    store = PrefixReuseState()
    first, second = object(), object()
    store.install_state(7, first)
    assert store.prefix_sequence_states == {7: first}
    # Overwrite without validation is the contract: the pool
    # lookup-replacement path releases the old handles at its call site
    # before overwriting, and the G>1 mirror installs wire values a rank
    # cannot validate locally.
    store.install_state(7, second)
    assert store.prefix_sequence_states == {7: second}


def test_append_commit_handle_appends_and_missing_state_raises_named_error():
    store = PrefixReuseState()
    store.install_state(
        3,
        store.new_sequence_state(
            lookup_result=None, attached_tokens=0, compute_cached_tokens=0
        ),
    )
    store.append_commit_handle(3, 11)
    store.append_commit_handle(3, 12)
    assert store.prefix_sequence_states[3].commit_attachment_handles == (11, 12)
    with pytest.raises(PrefixSequenceStateMissingError):
        store.append_commit_handle(4, 13)
    assert 4 not in store.prefix_sequence_states


def test_materialization_begin_end_and_lazy_drain():
    store = PrefixReuseState()
    first, second, third = object(), object(), object()
    store.begin_materialization(first)
    store.begin_materialization(second)
    store.end_materialization(second)
    assert store.active_prefix_materializations == [first]
    # Fail-loud: ending something never begun breaks the pairing.
    with pytest.raises(ValueError):
        store.end_materialization(third)
    store.begin_materialization(third)
    drain = store.drain_materializations()
    # Lazy: creating the iterator pops nothing.
    assert store.active_prefix_materializations == [first, third]
    assert next(drain) is third
    # Each item leaves the list BEFORE it is yielded.
    assert store.active_prefix_materializations == [first]
    assert list(drain) == [first]
    assert store.active_prefix_materializations == []
    assert list(store.drain_materializations()) == []


def test_drop_wave_refs_yields_each_sid_once_and_keeps_other_waves():
    store = PrefixReuseState()
    store.add_pool_member_refs(1, (-5, -6))
    store.add_pool_member_refs(2, (-6,))
    store.add_pool_member_refs(9, (-7,))  # another wave's member
    drop = store.drop_wave_refs([1, 2], [-5, -6, -6])
    # Lazy iterator: nothing is popped until it is consumed.
    assert store.pool_member_sids == {1: (-5, -6), 2: (-6,), 9: (-7,)}
    yielded = list(drop)
    # Each sid exactly once, duplicates collapsed, and the drop is
    # unconditional: -6 still had a second member (gid 2) in this same wave.
    assert sorted(yielded) == [-6, -5]
    assert store.pool_member_sids == {9: (-7,)}
    assert store.pool_sid_members == {-7: 1}
