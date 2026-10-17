"""M3 Slice 3a: the session -> owning-group pin at admission.

A frontier registry is group-local, so a continuation that lands on a serve
group other than the one holding its session is a clean MISS. These tests pin
the two halves of the fix:

1. the WORLD map -- every group's leader publishes a compact session summary on
   the host-KV mirror's existing world-wide object gather, and EVERY rank folds
   every leader's summary into the same rank-local map;
2. the PIN -- group assignment stamps a flag-on continuation onto its session's
   owning group before the balancer sees it, and leaves everything else to
   default assignment.

Pure Python, no GPU and no torch: the frontier module is loaded by path and the
worker methods are compiled out of the module's AST onto a namespace of fakes,
exactly as the other prefix-reuse unit tests do.
"""

import ast
import copy
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER = _REPO_ROOT / "batchgen" / "batchgen_worker.py"


def _load(name, relative_path):
    spec = importlib.util.spec_from_file_location(
        name, _REPO_ROOT / relative_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_FRONTIER = _load(
    "test_session_pin_frontier_module",
    Path("batchgen") / "prefix_reuse" / "trajectory_frontier.py",
)
_DP_GROUP = _load(
    "test_session_pin_decode_dp_group", Path("batchgen") / "decode_dp_group.py"
)

KDASnapshotDescriptor = _FRONTIER.KDASnapshotDescriptor
KDASnapshotProfile = _FRONTIER.KDASnapshotProfile
SessionGroupMapError = _FRONTIER.SessionGroupMapError
TrajectoryFrontierRegistry = _FRONTIER.TrajectoryFrontierRegistry
TrajectorySessionGroupIndex = _FRONTIER.TrajectorySessionGroupIndex
frontier_index_key = _FRONTIER.frontier_index_key

PAGE_TOKENS = 4
HOST_PAGE_BYTES = 2048
NAMESPACE = (11, 22, 33, 44)


# --------------------------------------------------------------------------- #
#  Real registries: the summaries folded below are produced, not hand-written
# --------------------------------------------------------------------------- #


def _profile():
    return KDASnapshotProfile(
        num_kda_layers=3,
        num_heads_per_rank=2,
        head_dim=8,
        conv_dim_per_rank=16,
        conv_width=4,
        recurrent_element_bytes=4,
        conv_element_bytes=2,
    )


def _registry():
    return TrajectoryFrontierRegistry(
        page_tokens=PAGE_TOKENS,
        host_page_bytes_all_layers=HOST_PAGE_BYTES,
        release_record=lambda record: None,
    )


def _begin(registry, *, tokens, group_id, handle=7):
    boundary = len(tokens)
    return registry.begin(
        namespace_digest=NAMESPACE,
        frontier_token_ids=tokens,
        page_ids_by_group={
            0: [500 + index for index in range(boundary // PAGE_TOKENS)]
        },
        snapshot=KDASnapshotDescriptor(
            storage_handle=handle,
            owning_group_id=group_id,
            profile=_profile(),
            boundary_tokens=boundary,
            shard_count=2,
        ),
    )


def _advance(registry, record, *, tokens, group_id, handle):
    return registry.advance(
        session_key=record.session_key,
        namespace_digest=NAMESPACE,
        frontier_token_ids=tokens,
        page_ids_by_group={
            0: [700 + index for index in range(len(tokens) // PAGE_TOKENS)]
        },
        snapshot=KDASnapshotDescriptor(
            storage_handle=handle,
            owning_group_id=group_id,
            profile=_profile(),
            boundary_tokens=len(tokens),
            shard_count=2,
        ),
    )


def _summary(registry, *, group_id):
    """The wire form a leader publishes: plain ints, strs and lists."""
    return {
        "group_id": int(group_id),
        "sessions": {
            str(session_key): [int(owning), int(boundary), str(index_key)]
            for session_key, (owning, boundary, index_key) in registry
            .session_group_summary()
            .items()
        },
    }


G0_TOKENS = [10, 11, 12, 13, 14, 15, 16, 17]
G1_TOKENS = [90, 91, 92, 93, 94, 95, 96, 97]
G0_CONTINUATION = G0_TOKENS + [500, 501, 502]
G1_CONTINUATION = G1_TOKENS + [600, 601]
UNRELATED_PROMPT = [1, 2, 3, 4, 5, 6, 7, 8, 9]


# --------------------------------------------------------------------------- #
#  The map: folding, replacement, eviction, resolution
# --------------------------------------------------------------------------- #


def test_summary_index_key_matches_the_registry_derivation():
    registry = _registry()
    record = _begin(registry, tokens=G0_TOKENS, group_id=0)
    summary = registry.session_group_summary()
    owning, boundary, index_key = summary[record.session_key]
    assert (owning, boundary) == (0, len(G0_TOKENS))
    assert index_key == frontier_index_key(
        namespace_digest=NAMESPACE, frontier_token_ids=G0_TOKENS
    )


def test_fold_two_leader_summaries_resolves_both_groups():
    left, right = _registry(), _registry()
    left_record = _begin(left, tokens=G0_TOKENS, group_id=0)
    right_record = _begin(right, tokens=G1_TOKENS, group_id=1, handle=9)

    index = TrajectorySessionGroupIndex()
    changed = index.fold(
        [None, _summary(left, group_id=0), None, _summary(right, group_id=1)]
    )

    assert changed == 2
    assert len(index) == 2
    left_hit = index.resolve(
        namespace_digest=NAMESPACE, prompt_token_ids=G0_CONTINUATION
    )
    right_hit = index.resolve(
        namespace_digest=NAMESPACE, prompt_token_ids=G1_CONTINUATION
    )
    assert left_hit.owning_group_id == 0
    assert left_hit.session_key == left_record.session_key
    assert right_hit.owning_group_id == 1
    assert right_hit.session_key == right_record.session_key


def test_fold_order_does_not_change_the_map():
    left, right = _registry(), _registry()
    _begin(left, tokens=G0_TOKENS, group_id=0)
    _begin(right, tokens=G1_TOKENS, group_id=1, handle=9)
    forward, backward = (
        TrajectorySessionGroupIndex(),
        TrajectorySessionGroupIndex(),
    )

    forward.fold([_summary(left, group_id=0), _summary(right, group_id=1)])
    backward.fold([_summary(right, group_id=1), _summary(left, group_id=0)])

    assert forward.entries == backward.entries


def test_replacement_on_advance_retires_the_old_boundary():
    registry = _registry()
    record = _begin(registry, tokens=G0_TOKENS, group_id=0)
    index = TrajectorySessionGroupIndex()
    index.fold([_summary(registry, group_id=0)])

    advanced_tokens = G0_TOKENS + [500, 501, 502, 503]
    _advance(registry, record, tokens=advanced_tokens, group_id=0, handle=8)
    changed = index.fold([_summary(registry, group_id=0)])

    assert changed == 1
    assert len(index) == 1
    entry = index.get(record.session_key)
    assert entry.boundary_tokens == len(advanced_tokens)
    # The advanced frontier resolves; the retired one is gone from the index,
    # so a prompt that only reaches the OLD boundary no longer pins.
    assert (
        index.resolve(
            namespace_digest=NAMESPACE,
            prompt_token_ids=advanced_tokens + [999],
        ).session_key
        == record.session_key
    )
    assert (
        index.resolve(
            namespace_digest=NAMESPACE, prompt_token_ids=G0_CONTINUATION
        )
        is None
    )


def test_a_leaders_summary_is_authoritative_for_its_own_group():
    left, right = _registry(), _registry()
    left_record = _begin(left, tokens=G0_TOKENS, group_id=0)
    _begin(right, tokens=G1_TOKENS, group_id=1, handle=9)
    index = TrajectorySessionGroupIndex()
    index.fold([_summary(left, group_id=0), _summary(right, group_id=1)])

    # Group 0 evicts its only session and republishes.
    left.evict(left_record.session_key)
    changed = index.fold([_summary(left, group_id=0)])

    assert changed == 1
    assert len(index) == 1
    assert index.get(left_record.session_key) is None
    # Group 1 never republished, so its entry stays.
    assert index.resolve(
        namespace_digest=NAMESPACE, prompt_token_ids=G1_CONTINUATION
    ).owning_group_id == 1


def test_a_leader_cannot_publish_another_groups_session():
    registry = _registry()
    _begin(registry, tokens=G0_TOKENS, group_id=0)
    index = TrajectorySessionGroupIndex()

    with pytest.raises(SessionGroupMapError, match="registry of its own group"):
        index.fold([_summary(registry, group_id=1)])


def test_a_malformed_summary_is_refused_by_name():
    index = TrajectorySessionGroupIndex()
    with pytest.raises(SessionGroupMapError, match="group_id"):
        index.fold([{"sessions": {}}])
    with pytest.raises(SessionGroupMapError, match="must publish"):
        index.fold([{"group_id": 0, "sessions": {"abc": [0, 8]}}])


def test_the_same_session_on_two_groups_resolves_deterministically():
    """One baseline committed on both groups: newest boundary wins."""
    left, right = _registry(), _registry()
    left_record = _begin(left, tokens=G0_TOKENS, group_id=0)
    right_record = _begin(right, tokens=G0_TOKENS, group_id=1, handle=9)
    assert left_record.session_key == right_record.session_key
    _advance(
        right,
        right_record,
        tokens=G0_TOKENS + [500, 501, 502, 503],
        group_id=1,
        handle=11,
    )

    forward, backward = (
        TrajectorySessionGroupIndex(),
        TrajectorySessionGroupIndex(),
    )
    forward.fold([_summary(left, group_id=0), _summary(right, group_id=1)])
    backward.fold([_summary(right, group_id=1), _summary(left, group_id=0)])

    assert forward.entries == backward.entries
    assert forward.get(left_record.session_key).owning_group_id == 1


def test_a_non_matching_prompt_and_a_short_prompt_do_not_resolve():
    registry = _registry()
    _begin(registry, tokens=G0_TOKENS, group_id=0)
    index = TrajectorySessionGroupIndex()
    index.fold([_summary(registry, group_id=0)])

    assert (
        index.resolve(
            namespace_digest=NAMESPACE, prompt_token_ids=UNRELATED_PROMPT
        )
        is None
    )
    # Shorter than the committed boundary: there is nothing to continue.
    assert (
        index.resolve(
            namespace_digest=NAMESPACE, prompt_token_ids=G0_TOKENS[:4]
        )
        is None
    )
    # A different namespace is a different serving configuration.
    assert (
        index.resolve(
            namespace_digest=(1, 2, 3, 4), prompt_token_ids=G0_CONTINUATION
        )
        is None
    )


def test_an_empty_map_is_falsy_and_resolves_nothing():
    index = TrajectorySessionGroupIndex()
    assert not index
    assert index.fold([None, None]) == 0
    assert (
        index.resolve(
            namespace_digest=NAMESPACE, prompt_token_ids=G0_CONTINUATION
        )
        is None
    )
    assert index.pins == 0


# --------------------------------------------------------------------------- #
#  Worker wiring
# --------------------------------------------------------------------------- #


def _worker_class():
    tree = ast.parse(WORKER.read_text())
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )


def _worker_method_node(name):
    return next(
        node
        for node in _worker_class().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _worker_methods(names, globals_=None):
    wanted = [copy.deepcopy(_worker_method_node(name)) for name in names]
    module = ast.fix_missing_locations(ast.Module(body=wanted, type_ignores=[]))
    namespace = {
        "List": List,
        "Optional": Optional,
        "logging": _SilentLogging(),
        **(globals_ or {}),
    }
    exec(compile(module, str(WORKER), "exec"), namespace)
    return {name: namespace[name] for name in names}


class _SilentLogging:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


class _FakeView:
    """A rank's LOCAL page table over one shared host-KV region."""

    def __init__(self, pages):
        self.table = dict(pages)

    def build_page_table(self, global_ids):
        return [list(self.table.get(global_id, ())) for global_id in global_ids]

    def set_sequence_pages(self, global_id, page_ids):
        self.table[global_id] = list(page_ids)


class _FakeDist:
    """Ranks stepped sequentially in one process share the gather slots."""

    def __init__(self, world_size):
        self.slots = [None] * world_size
        self.rank = 0

    def all_gather_object(self, out, payload):
        self.slots[self.rank] = payload
        for index, value in enumerate(self.slots):
            out[index] = value


class _FakeTokens:
    def __init__(self, token_ids):
        self._token_ids = list(token_ids)

    def tolist(self):
        return list(self._token_ids)


@pytest.fixture
def stub_modules(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "batchgen.decode_dp_group",
        SimpleNamespace(
            host_kv_owner_rank=lambda group, size: int(group) * int(size),
            assign_decode_dp_groups=_DP_GROUP.assign_decode_dp_groups,
            num_decode_dp_groups=_DP_GROUP.num_decode_dp_groups,
            rank_in_decode_group=_DP_GROUP.rank_in_decode_group,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "batchgen.prefix_reuse.prefill",
        SimpleNamespace(PrefixCacheSequenceState=SimpleNamespace),
    )
    monkeypatch.setitem(
        sys.modules,
        "batchgen.prefix_reuse.trajectory_commit",
        SimpleNamespace(
            trajectory_commit_enabled=lambda debug: bool(debug)
            and debug.get("prefix_trajectory_commit") == "on"
        ),
    )
    frontier_package = types.ModuleType("batchgen.prefix_reuse")
    monkeypatch.setitem(sys.modules, "batchgen.prefix_reuse", frontier_package)
    monkeypatch.setitem(
        sys.modules, "batchgen.prefix_reuse.trajectory_frontier", _FRONTIER
    )
    yield


def _mirror_rank(
    *,
    rank,
    world_size,
    attn_tp_size,
    seqs,
    fake_dist,
    registry=None,
):
    """A fake worker holding only what the mirror + fold path reads."""
    methods = _worker_methods(
        [
            "_mirror_group_host_kv",
            "_owns_host_kv",
            "_decode_attn_tp_size",
            "_trajectory_group_geometry",
            "_trajectory_session_summary",
            "_fold_trajectory_session_groups",
            "_finalize_trajectory_frontiers",
        ],
        {"dist": fake_dist},
    )
    worker = SimpleNamespace(
        rank=rank,
        world_size=world_size,
        local_world_size=world_size,
        parallel_manager=SimpleNamespace(attn_tp_size=attn_tp_size),
        global_batch=SimpleNamespace(get_sequence=seqs.get),
        _prefix_sequence_states={},
        _prefill_pool_wave=None,
        _trajectory_session_groups=None,
        _trajectory_arena_payload={},
        _trajectory_mirrored_restores={},
        _trajectory_frontier_committer=(
            None
            if registry is None
            else SimpleNamespace(
                owns_registry=True,
                registry=registry,
                # The pre-publish finalize: nothing pending in these fakes.
                finalize=lambda: SimpleNamespace(
                    installed=(),
                    rolled_back_sequence_ids=(),
                    deduped_sequence_ids=(),
                    failed_sequence_ids=(),
                ),
            )
        ),
        _host_views_by_group=lambda: {0: _FakeView({})},
        _reserve_trajectory_arenas=lambda uuids: {},
        _trajectory_restore_payload=lambda: {},
        _adopt_mirrored_trajectory_arenas=lambda arenas: None,
    )
    for name, function in methods.items():
        setattr(worker, name, types.MethodType(function, worker))
    return worker


def test_every_rank_folds_every_leaders_summary_on_the_mirror(stub_modules):
    """Two groups, G=2: the gather is world-wide, so each rank sees both."""
    world_size, G = 4, 2
    left, right = _registry(), _registry()
    left_record = _begin(left, tokens=G0_TOKENS, group_id=0)
    right_record = _begin(right, tokens=G1_TOKENS, group_id=1, handle=9)
    seqs = {
        "g0": SimpleNamespace(uuid="g0", global_idx=1, decode_dp_group=0),
        "g1": SimpleNamespace(uuid="g1", global_idx=2, decode_dp_group=1),
    }
    fake_dist = _FakeDist(world_size)
    registries = {0: left, 2: right}
    workers = {}
    for rank in range(world_size):
        workers[rank] = _mirror_rank(
            rank=rank,
            world_size=world_size,
            attn_tp_size=G,
            seqs=seqs,
            fake_dist=fake_dist,
            registry=registries.get(rank),
        )
    # The single-process fake hands a rank only the payloads already
    # contributed, so the pass is replayed once; a real all_gather_object
    # delivers every rank's payload inside one call.
    for _round in range(2):
        for rank in range(world_size):
            fake_dist.rank = rank
            workers[rank]._mirror_group_host_kv(
                ["g0"] if rank // G == 0 else ["g1"], {}, trajectory=True
            )

    # The leaders published; both leaders AND both followers folded both.
    assert [entry is not None for entry in fake_dist.slots] == [
        True,
        False,
        True,
        False,
    ]
    for rank in range(world_size):
        index = workers[rank]._trajectory_session_groups
        assert len(index) == 2, rank
        assert index.get(left_record.session_key).owning_group_id == 0
        assert index.get(right_record.session_key).owning_group_id == 1
        assert (
            index.resolve(
                namespace_digest=NAMESPACE, prompt_token_ids=G1_CONTINUATION
            ).owning_group_id
            == 1
        )
    # World-identical: every rank folded the same list in the same order.
    assert len({tuple(w._trajectory_session_groups.entries) for w in workers.values()}) == 1


def test_an_idle_leader_still_publishes_its_sessions_on_a_flag_on_pass(
    stub_modules,
):
    """The summary is registry state, not pass state (the m3gate blind spot).

    A flag-on pass that lands entirely on one serve group must still carry the
    OTHER group's committed sessions: the idle leader has no host KV to mirror
    (payload would be None), but its registry speaks for sessions the next
    pass's pins and pool plan need. The idle leader publishes a summary-only
    payload; a follower with no sequences tolerates it (no host-KV fields).
    """
    world_size, G = 4, 2
    left, right = _registry(), _registry()
    left_record = _begin(left, tokens=G0_TOKENS, group_id=0)
    right_record = _begin(right, tokens=G1_TOKENS, group_id=1, handle=9)
    # The pass's only sequence serves on group 1; group 0 is idle.
    seqs = {"g1": SimpleNamespace(uuid="g1", global_idx=2, decode_dp_group=1)}
    fake_dist = _FakeDist(world_size)
    registries = {0: left, 2: right}
    workers = {
        rank: _mirror_rank(
            rank=rank,
            world_size=world_size,
            attn_tp_size=G,
            seqs=seqs,
            fake_dist=fake_dist,
            registry=registries.get(rank),
        )
        for rank in range(world_size)
    }
    for _round in range(2):
        for rank in range(world_size):
            fake_dist.rank = rank
            workers[rank]._mirror_group_host_kv(
                [] if rank // G == 0 else ["g1"], {}, trajectory=True
            )

    # The idle leader published a summary-only payload: sessions, no host KV.
    assert fake_dist.slots[0] == {
        "frontier_sessions": workers[0]._trajectory_session_summary()
    }
    assert "global_ids" not in fake_dist.slots[0]
    # Every rank -- the idle group's follower included -- resolves BOTH groups.
    for rank in range(world_size):
        index = workers[rank]._trajectory_session_groups
        assert len(index) == 2, rank
        assert index.get(left_record.session_key).owning_group_id == 0
        assert index.get(right_record.session_key).owning_group_id == 1


def test_a_flag_off_mirror_publishes_no_summary_and_builds_no_map(stub_modules):
    world_size, G = 4, 2
    registry = _registry()
    _begin(registry, tokens=G0_TOKENS, group_id=0)
    seqs = {"g0": SimpleNamespace(uuid="g0", global_idx=1, decode_dp_group=0)}
    fake_dist = _FakeDist(world_size)
    leader = _mirror_rank(
        rank=0,
        world_size=world_size,
        attn_tp_size=G,
        seqs=seqs,
        fake_dist=fake_dist,
        registry=registry,
    )
    fake_dist.rank = 0
    leader._mirror_group_host_kv(["g0"], {})

    assert fake_dist.slots[0]["frontier_sessions"] == {}
    assert leader._trajectory_session_groups is None


def test_the_fold_runs_before_the_leaders_early_return():
    """Source-level: a leader must fold too, or its own ranks disagree."""
    mirror = _worker_method_node("_mirror_group_host_kv")
    fold_line = next(
        node.lineno
        for node in ast.walk(mirror)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_fold_trajectory_session_groups"
    )
    leader_return_line = next(
        node.lineno
        for node in ast.walk(mirror)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "leader"
            for target in node.targets
        )
    )
    assert fold_line < leader_return_line


def _assign_worker(
    *,
    world_size,
    attn_tp_size,
    seqs,
    index,
    tokens_by_uuid,
    debug=None,
):
    methods = _worker_methods(
        [
            "_assign_decode_dp_groups",
            "_pin_trajectory_session_groups",
            "_trajectory_commit_requested",
            "_active_batchgen_debug_for_sequences",
            "_decode_attn_tp_size",
            "_trajectory_restore_fields",
        ]
    )
    worker = SimpleNamespace(
        rank=0,
        world_size=world_size,
        parallel_manager=SimpleNamespace(attn_tp_size=attn_tp_size),
        # The real GlobalBatch is iterable AND resolves by uuid; a
        # SimpleNamespace cannot route dunder lookups, so the batch is its own
        # small fake.
        global_batch=_FakeBatch(seqs),
        enable_prefix_cache=True,
        prefix_cache_runtime_config=SimpleNamespace(
            namespace_digest=NAMESPACE
        ),
        _batchgen_debug=debug,
        # The real worker latches this when an admitted sequence carries
        # per-request debug flags; these sequences do, so the merge path the
        # pin depends on stays reachable.
        _any_sequence_batchgen_debug=True,
        # These fakes exercise the balancer path, not main's sharing-aware
        # packer: no in-wave prefix pool is configured for them.
        _prefill_prefix_pool_enabled=lambda: False,
        _trajectory_book=SimpleNamespace(
            has_sequence=lambda uuid: uuid in tokens_by_uuid
        ),
        _trajectory_session_groups=index,
        _trajectory_restorer=None,
        _trajectory_tokens=lambda seq, length=None: _FakeTokens(
            tokens_by_uuid[seq.uuid][: length if length else None]
        ),
    )
    for name, function in methods.items():
        setattr(worker, name, types.MethodType(function, worker))
    return worker


class _FakeBatch:
    def __init__(self, seqs):
        self._seqs = seqs

    def get_sequence(self, uuid):
        return self._seqs.get(uuid)

    def __iter__(self):
        return iter(list(self._seqs.values()))


def _sequence(uuid, *, prompt_length, debug="on"):
    return SimpleNamespace(
        uuid=uuid,
        global_idx=int(uuid[-1]),
        prompt_length=prompt_length,
        decode_dp_group=None,
        # Real Sequence field read by the balancer path: a first admission
        # has no eviction history.
        evicted_from_dp_group=None,
        batchgen_debug=(
            {"prefix_trajectory_commit": debug} if debug is not None else None
        ),
    )


def _folded_index():
    left, right = _registry(), _registry()
    _begin(left, tokens=G0_TOKENS, group_id=0)
    _begin(right, tokens=G1_TOKENS, group_id=1, handle=9)
    index = TrajectorySessionGroupIndex()
    index.fold([_summary(left, group_id=0), _summary(right, group_id=1)])
    return index


def test_the_pin_stamps_the_owning_group_and_leaves_others_to_the_balancer(
    stub_modules,
):
    index = _folded_index()
    # A continuation of the group-1 session plus two prompts that continue
    # nothing. Lengths are equal, so without the pin the balancer would place
    # the first candidate on group 0.
    seqs = {
        "seq1": _sequence("seq1", prompt_length=len(G1_CONTINUATION)),
        "seq2": _sequence("seq2", prompt_length=len(UNRELATED_PROMPT)),
        "seq3": _sequence("seq3", prompt_length=len(UNRELATED_PROMPT)),
    }
    tokens = {
        "seq1": G1_CONTINUATION,
        "seq2": UNRELATED_PROMPT,
        "seq3": UNRELATED_PROMPT,
    }
    worker = _assign_worker(
        world_size=4,
        attn_tp_size=2,
        seqs=seqs,
        index=index,
        tokens_by_uuid=tokens,
    )

    worker._assign_decode_dp_groups(["seq1", "seq2", "seq3"])

    assert seqs["seq1"].decode_dp_group == 1
    assert index.pins == 1
    # Without the pin the balancer would have put the longest candidate on
    # group 0 -- the empty-world placement -- which is the parity flip the pin
    # exists to stop.
    assert _DP_GROUP.assign_decode_dp_groups(
        [len(G1_CONTINUATION)], 2, prior_load=[0.0, 0.0]
    ) == [0]
    # The pinned sequence's L^2 was charged to group 1 before the balancer
    # ran, so the two unpinned prompts go to the now-lighter group 0 instead
    # of being placed as if group 1 were empty.
    assert [seqs["seq2"].decode_dp_group, seqs["seq3"].decode_dp_group] == [
        0,
        0,
    ]
    assert worker._trajectory_restore_fields() == {"trajectory_group_pins": 1}


def test_the_pin_is_idempotent_across_repeated_assignment(stub_modules):
    index = _folded_index()
    seqs = {"seq1": _sequence("seq1", prompt_length=len(G0_CONTINUATION))}
    worker = _assign_worker(
        world_size=4,
        attn_tp_size=2,
        seqs=seqs,
        index=index,
        tokens_by_uuid={"seq1": G0_CONTINUATION},
    )

    worker._assign_decode_dp_groups(["seq1"])
    worker._assign_decode_dp_groups(["seq1"])

    assert seqs["seq1"].decode_dp_group == 0
    # The second pass skipped the already-grouped sequence before resolving.
    assert index.pins == 1


def test_a_flag_off_batch_is_assigned_exactly_as_before(stub_modules):
    index = _folded_index()
    flag_off = {
        "seq1": _sequence(
            "seq1", prompt_length=len(G1_CONTINUATION), debug="off"
        ),
        "seq2": _sequence("seq2", prompt_length=4096, debug="off"),
    }
    tokens = {"seq1": G1_CONTINUATION, "seq2": UNRELATED_PROMPT}
    worker = _assign_worker(
        world_size=4,
        attn_tp_size=2,
        seqs=flag_off,
        index=index,
        tokens_by_uuid=tokens,
    )

    worker._assign_decode_dp_groups(["seq1", "seq2"])

    expected = _DP_GROUP.assign_decode_dp_groups(
        [len(G1_CONTINUATION), 4096], 2, prior_load=[0.0, 0.0]
    )
    assert [
        flag_off["seq1"].decode_dp_group,
        flag_off["seq2"].decode_dp_group,
    ] == expected
    assert index.pins == 0


def test_no_map_means_no_pin_and_no_drain_field(stub_modules):
    seqs = {"seq1": _sequence("seq1", prompt_length=len(G0_CONTINUATION))}
    worker = _assign_worker(
        world_size=4,
        attn_tp_size=2,
        seqs=seqs,
        index=None,
        tokens_by_uuid={"seq1": G0_CONTINUATION},
    )

    assert worker._pin_trajectory_session_groups(["seq1"], num_groups=2) == {}
    worker._assign_decode_dp_groups(["seq1"])
    assert seqs["seq1"].decode_dp_group == 0  # pure balancer, empty world
    assert worker._trajectory_restore_fields() == {}


def test_a_session_owned_by_a_group_outside_the_world_is_refused(stub_modules):
    registry = _registry()
    _begin(registry, tokens=G0_TOKENS, group_id=3)
    index = TrajectorySessionGroupIndex()
    index.fold([_summary(registry, group_id=3)])
    seqs = {"seq1": _sequence("seq1", prompt_length=len(G0_CONTINUATION))}
    worker = _assign_worker(
        world_size=4,
        attn_tp_size=2,
        seqs=seqs,
        index=index,
        tokens_by_uuid={"seq1": G0_CONTINUATION},
    )

    with pytest.raises(RuntimeError, match="does not exist in a 2-group world"):
        worker._assign_decode_dp_groups(["seq1"])


def test_a_sequence_without_a_trajectory_reservation_is_not_pinned(
    stub_modules,
):
    index = _folded_index()
    seqs = {"seq1": _sequence("seq1", prompt_length=len(G0_CONTINUATION))}
    worker = _assign_worker(
        world_size=4,
        attn_tp_size=2,
        seqs=seqs,
        index=index,
        tokens_by_uuid={},
    )

    assert worker._pin_trajectory_session_groups(["seq1"], num_groups=2) == {}
    assert index.pins == 0
