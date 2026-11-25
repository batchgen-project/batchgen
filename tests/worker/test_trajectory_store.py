import ast
import logging
import random
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Sequence

import pytest
import torch

from batchgen.prefix_reuse.trajectory_commit import (
    SnapshotArena,
    TrajectoryStateUnavailable,
)
from batchgen.query_book import QueryBook
from batchgen.worker.prefix_reuse_state import PrefixReuseState
from batchgen.worker.trajectory_store import TrajectoryStore


from tests.worker_method_source import load_worker_shaped_methods


def _worker_method_from_source(name):
	"""Load one current worker method without importing CUDA-only modules."""
	path = Path(__file__).parents[2] / "batchgen" / "batchgen_worker.py"
	tree = ast.parse(path.read_text())
	worker_class = next(
		node for node in tree.body
		if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
	)
	node = next(
		node for node in worker_class.body
		if isinstance(node, ast.FunctionDef) and node.name == name
	)
	namespace = {"Sequence": Sequence}
	exec(
		compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"),
		namespace,
	)
	return namespace[name]


# copied verbatim from batchgen_worker.py:14590-14623 @ b0f5540e
def _old_fold_trajectory_session_groups(self, gathered) -> None:
	"""Fold every leader's session summary into this rank's map.

	Runs on EVERY rank, in gathered rank order, over the same list on
	every rank -- that identity is what lets the pin below be part of a
	replicated assignment. Nothing is built when no leader published a
	summary, so a flag-on pass before the first commit allocates nothing.
	"""
	summaries = [
		entry.get("frontier_sessions")
		for entry in gathered
		if isinstance(entry, dict) and entry.get("frontier_sessions")
	]
	if not summaries and self._trajectory_session_groups is None:
		return
	from batchgen.prefix_reuse.trajectory_frontier import (
		TrajectorySessionGroupIndex,
	)

	index = self._trajectory_session_groups
	if index is None:
		index = TrajectorySessionGroupIndex()
		self._trajectory_session_groups = index
	changed = index.fold(summaries)
	if changed:
		logging.info(
			"[TRAJECTORY_FRONTIER] Rank %s folded %s session-group "
			"change(s) from %s leader summary/summaries: %s session(s) "
			"now pinnable",
			self.rank,
			changed,
			len(summaries),
			len(index),
		)


def test_token_book_accessors_and_append_batch():
    class Book:
        def __init__(self):
            self.tokens = {"u": [1]}
            self._slots = {"u": 0}
            self.calls = []

        def has_sequence(self, uuid):
            return uuid in self._slots

        def slot_for(self, uuid):
            return self._slots[uuid]

        def append_token(self, slot, token):
            self.tokens["u"].append(token)
            return len(self.tokens["u"])

        def append_tokens(self, slots, tokens):
            self.calls.append((list(slots), list(tokens)))

    store = TrajectoryStore()
    store.trajectory_book = Book()
    assert store.has_sequence("u")
    assert store.slot_for("u") == 0
    assert store.append_token(0, 7) == 2
    store.append_tokens([0], [8])
    assert store.trajectory_book.calls == [([0], [8])]


def test_group_geometry_and_commit_tokens():
    store = TrajectoryStore()
    assert store.group_geometry(decode_attn_tp_size=4, rank=6) == (4, 2, False)
    runtime = SimpleNamespace(publish_boundary_tokens=64)
    assert store.commit_tokens(runtime, SimpleNamespace(prompt_length=128)) == 128
    assert store.commit_tokens(runtime, SimpleNamespace(prompt_length=129)) == 0


def test_kda_release_without_optional_model_is_noop(monkeypatch):
    store = TrajectoryStore()
    store.free_kda_sequences([1, 2])


def _state_owner(*, built=True):
    """A model-declared recurrent-state owner, pools built or not."""
    freed = []
    manager = SimpleNamespace(
        config=SimpleNamespace(num_state_items=4),
        sequence_state_items={7: 0},
    )
    return SimpleNamespace(
        state_manager=manager if built else None,
        slot_manager=object() if built else None,
        free_sequences=freed.extend,
        freed=freed,
    )


def test_model_without_recurrent_state_reads_as_absent():
    # The provider returning None is a model that declares no owner.
    for store in (TrajectoryStore(), TrajectoryStore(recurrent_state_owner=lambda: None)):
        assert store.kda_state_manager() is None
        assert store.kda_stats() == (-1, -1)
        store.free_kda_sequences([1, 2])
        with pytest.raises(TrajectoryStateUnavailable, match="this deployment has none"):
            store.build_committer(
                rank=0,
                runtime=SimpleNamespace(),
                group_geometry=(1, 0, True),
                unit_count=1,
                host_views_provider=lambda: {},
                next_pseudo_sid=lambda: -1,
            )


def test_declared_owner_serves_manager_stats_and_release():
    owner = _state_owner()
    store = TrajectoryStore(recurrent_state_owner=lambda: owner)
    assert store.kda_state_manager() is owner.state_manager
    assert store.kda_stats() == (4, 1)
    store.free_kda_sequences([])
    store.free_kda_sequences(iter([3, 5]))
    assert owner.freed == [3, 5]


def test_owner_before_its_pools_exist_is_a_noop():
    owner = _state_owner(built=False)
    store = TrajectoryStore(recurrent_state_owner=lambda: owner)
    assert store.kda_state_manager() is None
    assert store.kda_stats() == (-1, -1)
    store.free_kda_sequences([3])
    assert owner.freed == []


def test_owner_is_resolved_on_every_call_not_cached():
    # The store is built before the worker has a PSM, so the first answer
    # (None) must not stick once the model has declared its owner.
    declared = []
    store = TrajectoryStore(
        recurrent_state_owner=lambda: declared[0] if declared else None
    )
    assert store.kda_state_manager() is None
    declared.append(_state_owner())
    assert store.kda_state_manager() is declared[0].state_manager


# The seam: the worker names no model to reach the state; it asks the PSM.
def test_worker_resolves_the_owner_through_the_psm_declaration():
    resolve = _worker_method_from_source("_recurrent_state_owner")
    owner = object()
    assert resolve(SimpleNamespace()) is None
    assert resolve(SimpleNamespace(parallel_manager=None)) is None
    assert resolve(SimpleNamespace(parallel_manager=SimpleNamespace())) is None
    declaring = SimpleNamespace(recurrent_state_owner=lambda: owner)
    assert resolve(SimpleNamespace(parallel_manager=declaring)) is owner


def test_scaffolding_never_imports_the_model_state_wrapper():
    root = Path(__file__).parents[2] / "batchgen"
    worker = (root / "batchgen_worker.py").read_text()
    assert "TrajectoryStore(recurrent_state_owner=self._recurrent_state_owner)" in worker
    for path in [root / "batchgen_worker.py", *sorted((root / "worker").glob("*.py"))]:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                continue
            for module in modules:
                assert not module.endswith("kimi_linear.wrappers"), (path.name, node.lineno)


def test_kimi_linear_psm_declares_the_kda_wrapper_as_owner():
    # Read from source: importing the PSM needs the GPU kernel stack.
    path = (
        Path(__file__).parents[2] / "batchgen" / "models" / "moonshotai"
        / "kimi_linear" / "Parallel_Strategy_Manager.py"
    )
    psm = next(
        node for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef)
        and node.name == "KimiLinearParallelStrategyManager"
    )
    declares = next(
        node for node in psm.body
        if isinstance(node, ast.FunctionDef) and node.name == "recurrent_state_owner"
    )
    assert ast.unparse(declares.body[-1]) == "return KimiLinearKDAWrapper"


def test_frontier_prefix_is_available_without_eager_builder():
    store = TrajectoryStore()
    assert isinstance(store.FRONTIER_PREFIX, str)
    assert store.trajectory_frontier_committer is None


def test_fold_is_lazy_and_deterministic_for_all_ranks():
    stores = [TrajectoryStore(), TrajectoryStore()]
    gathered = [
        {"frontier_sessions": {"group_id": 0, "sessions": {"session": [0, 64, "digest"]}}},
        {"frontier_sessions": {}},
    ]
    for store in stores:
        store.fold_session_groups(gathered, rank=0)
    assert stores[0].trajectory_session_groups is not None
    assert stores[1].trajectory_session_groups is not None
    assert stores[0].trajectory_session_groups.entries == stores[1].trajectory_session_groups.entries
    empty = TrajectoryStore()
    empty.fold_session_groups([{}])
    assert empty.trajectory_session_groups is None


# A6: QueryBook slot, append, metadata, decoded-token, and append-batch parity.
def test_query_book_accessors_match_direct_query_book_fixture():
	def make_book():
		book = QueryBook(capacity_bytes=4 * 8 * 4, page_tokens=8)
		slot = book.bind("u", max_tokens=24)
		book.write_prompt(slot, torch.tensor([1, 2, 3], dtype=torch.int32))
		return book, slot

	direct, direct_slot = make_book()
	store_book, store_slot = make_book()
	store = TrajectoryStore()
	store.trajectory_book = store_book

	direct.append_token(direct_slot, 10)
	store.append_token(store_slot, 10)
	direct.append_tokens([direct_slot], [11])
	store.append_tokens([store_slot], [11])
	assert store.slot_for("u") == direct_slot
	assert store.metadata(store_slot) == direct.metadata(direct_slot)
	assert torch.equal(
		store.generated_tokens(store_slot), direct.generated_tokens(direct_slot)
	)
	out_direct = torch.empty(2, dtype=torch.int32)
	out_store = torch.empty(2, dtype=torch.int32)
	direct.copy_span_to(direct_slot, out_direct, start=1, length=2)
	store.copy_span_to(store_slot, out_store, start=1, length=2)
	assert torch.equal(out_store, out_direct)


# A6 item 4: restore_payload serializes an arena that arena_from_payload rebuilds.
def test_restore_payload_round_trips_through_arena_from_payload():
	store = TrajectoryStore()
	arena = SnapshotArena(
		storage_handle=7,
		pseudo_sid=-91,
		page_ids_by_group={0: (10, 11), 1: (20, 21)},
		pages_per_group=2,
		capacity_bytes=256,
		payload_bytes=128,
		unit_count=2,
	)
	store.trajectory_frontier_committer = SimpleNamespace(
		arena_for=lambda handle: arena
	)
	store.trajectory_attachments = {
		5: SimpleNamespace(
			storage_handle=7,
			session_key="session",
			boundary_tokens=64,
			lcp_tokens=64,
			owning_group_id=3,
		)
	}
	payload = store.restore_payload(
		group_size=2, is_leader=True, planned_boundary_for=lambda gid: 64
	)
	rebuilt = store.arena_from_payload(payload[5])
	assert rebuilt.pseudo_sid == arena.pseudo_sid
	assert rebuilt.page_ids_by_group == arena.page_ids_by_group
	assert rebuilt.pages_per_group == arena.pages_per_group
	assert rebuilt.capacity_bytes == arena.capacity_bytes
	assert rebuilt.unit_count == arena.unit_count


# A6 item 5: session_summary keeps leader/registry gating and wire fields.
def test_session_summary_preserves_leader_registry_payload():
	class Registry:
		def session_group_summary(self):
			return {"s": (3, 128, "idx")}

	store = TrajectoryStore()
	store.trajectory_frontier_committer = SimpleNamespace(
		owns_registry=True, registry=Registry()
	)
	assert store.session_summary(group_size=4, is_leader=True, rank=6) == {
		"group_id": 1,
		"sessions": {"s": [3, 128, "idx"]},
	}
	assert store.session_summary(group_size=4, is_leader=False, rank=6) == {}


# A6 item 6: adopt_mirrored_arenas reconstructs and passes each arena onward.
def test_adopt_mirrored_arenas_rebuilds_each_payload():
	class Committer:
		def __init__(self):
			self.adopted = []

		def adopt_arena(self, *, sequence_id, arena):
			self.adopted.append((sequence_id, arena))

	committer = Committer()
	store = TrajectoryStore()
	entry = {
		"pseudo_sid": -7,
		"pages": {0: [1, 2]},
		"pages_per_group": 2,
		"capacity_bytes": 128,
		"unit_count": 1,
	}
	store.adopt_mirrored_arenas({9: entry}, committer=committer)
	assert len(committer.adopted) == 1
	sequence_id, arena = committer.adopted[0]
	assert sequence_id == 9
	assert arena.page_ids_by_group == {0: (1, 2)}


# A6 item 7: restore_fields combines restorer counters with placement telemetry.
def test_restore_fields_combines_restorer_and_session_pins():
	store = TrajectoryStore()
	store.trajectory_restorer = SimpleNamespace(counters=lambda: {"restores": 4})
	store.trajectory_session_groups = SimpleNamespace(
		honored_on_owner=3, placed_offgroup=1
	)
	assert store.restore_fields() == {
		"restores": 4,
		"trajectory_group_honored_on_owner": 3,
		"trajectory_group_placed_offgroup": 1,
	}


# A6 item 8: commit_requested delegates the strict batch debug flag parser.
def test_commit_requested_accepts_only_the_trajectory_commit_flag():
	store = TrajectoryStore()
	assert not store.commit_requested(None)
	assert store.commit_requested({"prefix_trajectory_commit": "on"})
	assert not store.commit_requested({"prefix_trajectory_commit": "off"})
	with pytest.raises(ValueError):
		store.commit_requested({"prefix_trajectory_commit": "typo"})


# A5: release_attachments removes each attachment once and is idempotent.
def test_release_attachments_is_idempotent():
	released = []
	store = TrajectoryStore()
	store.trajectory_restorer = SimpleNamespace(
		release=lambda attachment: released.append(attachment)
	)
	attachment = object()
	store.trajectory_attachments = {4: attachment}
	store.release_attachments([4], rank=2)
	state_after_first = dict(store.trajectory_attachments)
	store.release_attachments([4], rank=2)
	assert state_after_first == store.trajectory_attachments == {}
	assert released == [attachment]


# A5: rollback_reservations must not repeat a reservation side effect.
def test_rollback_reservations_store_state_is_idempotent_committer_is_not():
	# Pins the b0f5540e semantics of _rollback_trajectory_reservations
	# (batchgen_worker.py:14883-14892): the two store-local dicts are popped
	# with a default, so a repeat call changes nothing there, but every call
	# forwards rollback_reservation to the committer. Idempotence of that
	# forwarded call is the committer's contract, never the worker's.
	calls = []
	store = TrajectoryStore()
	store.trajectory_frontier_committer = SimpleNamespace(
		rollback_reservation=lambda **kwargs: calls.append(kwargs)
	)
	store.trajectory_arena_payload = {4: "arena"}
	store.trajectory_mirrored_restores = {4: "mirror"}
	store.rollback_reservations([4], rank=2)
	assert store.trajectory_arena_payload == {}
	assert store.trajectory_mirrored_restores == {}
	assert calls == [{"sequence_id": 4}]
	store.rollback_reservations([4], rank=2)
	assert store.trajectory_arena_payload == {}
	assert store.trajectory_mirrored_restores == {}
	assert calls == [{"sequence_id": 4}, {"sequence_id": 4}]


# A5: the real worker release seam preserves all five externally visible calls.
def test_release_seam_call_order():
	events = []
	worker = SimpleNamespace()
	worker.enable_prefix_cache = True
	worker.rank = 2
	worker._traj = SimpleNamespace(
		release_attachments=lambda ids, *, rank: events.append(("trajectory", list(ids), rank)),
		rollback_reservations=lambda ids, *, rank: events.append(("rollback", list(ids), rank)),
	)
	worker._release_pool_segment_refs = lambda gid: events.append("pool")

	class Prefix(PrefixReuseState):
		def pop_state(self, global_id):
			events.append("prefix-pop")
			return super().pop_state(global_id)

	worker._prefix = Prefix()
	worker._prefix.prefix_sequence_states[7] = SimpleNamespace(
		commit_attachment_handles=("commit",), attachment_handle="active"
	)
	worker.prefix_cache_coordinator = SimpleNamespace(
		release_attachment=lambda handle: events.append(("coordinator", handle))
	)
	# The body lives in the KV capacity engine and takes the decode context;
	# it is compiled here in the worker's shape.
	release = load_worker_shaped_methods(
		["_release_prefix_cache_attachments"], {"Sequence": Sequence}
	)["_release_prefix_cache_attachments"]
	release(worker, [7])
	assert events == [
		("trajectory", [7], 2),
		("rollback", [7], 2),
		"pool",
		"prefix-pop",
		("coordinator", "commit"),
		("coordinator", "active"),
	]


# A6: the restorer builder refuses to construct anything before a committer.
def test_build_restorer_refuses_without_committer():
	store = TrajectoryStore()
	assert store.build_restorer(
		rank=1,
		host_views_provider=lambda: {},
		next_pseudo_sid=lambda: -1,
	) is None


# A6: follower construction preserves the old TrajectoryStateUnavailable branch.
def test_follower_trajectory_state_unavailable_raises(monkeypatch):
	from batchgen.models.moonshotai.kimi_linear.wrappers import KimiLinearKDAWrapper

	monkeypatch.setattr(KimiLinearKDAWrapper, "state_manager", None, raising=False)
	# Constructed the way the worker constructs it: the owner is injected.
	store = TrajectoryStore(recurrent_state_owner=lambda: KimiLinearKDAWrapper)
	store.trajectory_mirrored_restores = {9: {}}
	worker = SimpleNamespace()
	worker._traj = store
	worker.rank = 1
	worker.prefix_cache_runtime_config = SimpleNamespace()
	worker._trajectory_group_geometry = lambda: (2, 1, False)
	worker._trajectory_record_unit_count = lambda: 2
	worker._host_views_by_group = lambda: {}
	worker._next_pool_segment_sid = lambda: -1
	worker._get_trajectory_frontier_committer = MethodType(
		_worker_method_from_source("_get_trajectory_frontier_committer"), worker
	)
	with pytest.raises(
		TrajectoryStateUnavailable,
		match="prefix_trajectory_commit needs a live KDA state pool; this deployment has none",
	):
		_worker_method_from_source("_restore_mirrored_trajectory_units")(worker, [])


# A3: the migrated re-entry writer keeps the old QueryBook turn ledger.
def test_reentry_writer_begin_reentry_turn_ledger_parity():
	def seeded_book():
		book = QueryBook(capacity_bytes=4 * 8 * 4, page_tokens=8)
		slot = book.bind("u", max_tokens=24)
		book.write_prompt(slot, torch.tensor([1, 2, 3], dtype=torch.int32))
		book.append_tokens([slot], [10])
		book.append_tokens([slot], [11])
		return book, slot

	old_book, old_slot = seeded_book()
	new_book, new_slot = seeded_book()
	old_book.begin_reentry_turn(old_slot, prompt_length=5)
	store = TrajectoryStore()
	store.trajectory_book = new_book
	store.trajectory_book.begin_reentry_turn(new_slot, prompt_length=5)
	assert new_book.turns(new_slot) == old_book.turns(old_slot)
	assert new_book.metadata(new_slot) == old_book.metadata(old_slot)


def _random_gathered(seed):
	rng = random.Random(seed)
	gathered = []
	for rank in range(6):
		group_id = rank // 2
		sessions = {}
		for session_index in range(rng.randrange(0, 4)):
			key = f"s{session_index}"
			sessions[key] = [group_id, 64 + rng.randrange(4) * 64, f"idx-{key}"]
		gathered.append({"frontier_sessions": {"group_id": group_id, "sessions": sessions}})
	return gathered


# A4: the frozen fold body agrees with the migrated store on randomized lists.
def test_fold_session_groups_matches_frozen_fixture():
	for seed in range(8):
		gathered = _random_gathered(seed)
		old = SimpleNamespace(_trajectory_session_groups=None, rank=0)
		_old_fold_trajectory_session_groups(old, gathered)
		new = TrajectoryStore()
		new.fold_session_groups(gathered, rank=0)
		assert (
			None if old._trajectory_session_groups is None else old._trajectory_session_groups.entries
		) == (
			None if new.trajectory_session_groups is None else new.trajectory_session_groups.entries
		)


# A4: every simulated rank folds the same list into the same ordered groups.
def test_fold_is_deterministic_across_simulated_ranks():
	gathered = _random_gathered(31)
	stores = [TrajectoryStore() for _ in range(5)]
	for rank, store in enumerate(stores):
		store.fold_session_groups(gathered, rank=rank)
	entries = [store.trajectory_session_groups.entries for store in stores]
	assert all(value == entries[0] for value in entries)

