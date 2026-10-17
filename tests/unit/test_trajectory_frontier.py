"""Contract tests for the Host-side trajectory frontier bookkeeping.

Pure Python, no GPU. The module is loaded by path, as the other prefix-reuse
unit tests do, so the contract is checked without importing the engine package.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).parents[2]
_MODULE_PATH = (
    _REPO_ROOT / "batchgen" / "prefix_reuse" / "trajectory_frontier.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "test_trajectory_frontier_module", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

FrontierCharge = _MODULE.FrontierCharge
FrontierRecordInvalid = _MODULE.FrontierRecordInvalid
FrontierRegistryInvariantError = _MODULE.FrontierRegistryInvariantError
FrontierRefcountError = _MODULE.FrontierRefcountError
FrontierRegistryError = _MODULE.FrontierRegistryError
FrontierReleaseFailed = _MODULE.FrontierReleaseFailed
KDASnapshotDescriptor = _MODULE.KDASnapshotDescriptor
KDASnapshotProfile = _MODULE.KDASnapshotProfile
KDASnapshotProfileInvalid = _MODULE.KDASnapshotProfileInvalid
TrajectoryFrontierRecord = _MODULE.TrajectoryFrontierRecord
TrajectoryFrontierRegistry = _MODULE.TrajectoryFrontierRegistry
build_kda_snapshot_profile = _MODULE.build_kda_snapshot_profile
dtype_element_bytes = _MODULE.dtype_element_bytes
frontier_index_key = _MODULE.frontier_index_key
is_frontier_hit = _MODULE.is_frontier_hit
longest_common_prefix_length = _MODULE.longest_common_prefix_length
session_key_for_baseline = _MODULE.session_key_for_baseline
snapshot_bytes = _MODULE.snapshot_bytes


PAGE_TOKENS = 4
# One MLA latent page across all layers; the real value comes from
# PrefixCacheRuntimeConfig.host_page_bytes_all_layers.
HOST_PAGE_BYTES = 2048
NAMESPACE = (11, 22, 33, 44)


class _Releaser:
    """Records whole-record releases so atomicity can be asserted."""

    def __init__(self, *, fail_times=0):
        self.calls = []
        self._fail_times = int(fail_times)

    def __call__(self, record):
        if self._fail_times > 0:
            self._fail_times -= 1
            raise RuntimeError("host page release rejected")
        self.calls.append(
            (
                record.session_key,
                {
                    group: tuple(pages)
                    for group, pages in record.page_ids_by_group.items()
                },
                record.snapshot.storage_handle,
            )
        )


def _profile(**overrides):
    base = dict(
        num_kda_layers=3,
        num_heads_per_rank=2,
        head_dim=8,
        conv_dim_per_rank=16,
        conv_width=4,
        recurrent_element_bytes=4,
        conv_element_bytes=2,
    )
    base.update(overrides)
    return KDASnapshotProfile(**base)


def _snapshot(
    *, boundary_tokens, handle=7, group_id=0, profile=None, shard_count=1
):
    return KDASnapshotDescriptor(
        storage_handle=handle,
        owning_group_id=group_id,
        profile=profile if profile is not None else _profile(),
        boundary_tokens=boundary_tokens,
        shard_count=shard_count,
    )


def _pages(count, *, groups=(0,), base=100):
    return {
        group: [base + group * 1000 + index for index in range(count)]
        for group in groups
    }


def _registry(*, releaser=None, page_tokens=PAGE_TOKENS):
    return TrajectoryFrontierRegistry(
        page_tokens=page_tokens,
        host_page_bytes_all_layers=HOST_PAGE_BYTES,
        release_record=releaser if releaser is not None else _Releaser(),
    )


def _begin(
    registry, *, tokens, groups=(0,), handle=7, group_id=0, shard_count=1
):
    boundary = len(tokens)
    return registry.begin(
        namespace_digest=NAMESPACE,
        frontier_token_ids=tokens,
        page_ids_by_group=_pages(boundary // registry.page_tokens, groups=groups),
        snapshot=_snapshot(
            boundary_tokens=boundary,
            handle=handle,
            group_id=group_id,
            shard_count=shard_count,
        ),
    )


# ---------------------------------------------------------------------------
# snapshot_bytes: derived from config, never a copied constant
# ---------------------------------------------------------------------------


def test_snapshot_bytes_matches_the_pool_geometry_computed_in_the_test():
    layers, heads, head_dim, conv_dim, width = 3, 2, 8, 16, 4
    recurrent_elem, conv_elem = 4, 2
    profile = _profile(
        num_kda_layers=layers,
        num_heads_per_rank=heads,
        head_dim=head_dim,
        conv_dim_per_rank=conv_dim,
        conv_width=width,
        recurrent_element_bytes=recurrent_elem,
        conv_element_bytes=conv_elem,
    )
    # Recurrent pool view (layers, 1 slot, heads, head_dim, head_dim) plus the
    # three conv pools (layers, 1 slot, conv_dim, width - 1).
    expected_recurrent = layers * heads * head_dim * head_dim * recurrent_elem
    expected_conv = 3 * layers * conv_dim * (width - 1) * conv_elem
    assert profile.recurrent_bytes == expected_recurrent
    assert profile.conv_bytes == expected_conv
    assert snapshot_bytes(profile) == expected_recurrent + expected_conv


def test_snapshot_bytes_tracks_every_geometry_axis():
    base = snapshot_bytes(_profile())
    assert snapshot_bytes(_profile(num_kda_layers=6)) == 2 * base
    # Doubling the head shard doubles the recurrent term and the conv term,
    # because conv_dim follows the shard in the real pool.
    doubled = _profile(num_heads_per_rank=4, conv_dim_per_rank=32)
    assert snapshot_bytes(doubled) == 2 * base
    quadrupled_recurrent = _profile(head_dim=16, conv_dim_per_rank=32)
    assert (
        quadrupled_recurrent.recurrent_bytes == 4 * _profile().recurrent_bytes
    )


def test_snapshot_profile_refuses_impossible_geometry():
    with pytest.raises(KDASnapshotProfileInvalid):
        _profile(num_kda_layers=0)
    with pytest.raises(KDASnapshotProfileInvalid):
        _profile(conv_width=1)
    with pytest.raises(KDASnapshotProfileInvalid):
        _profile(recurrent_element_bytes=0)


def test_unknown_state_dtype_is_refused_by_name():
    assert dtype_element_bytes("float32") == 4
    assert dtype_element_bytes("torch.bfloat16") == 2
    with pytest.raises(KDASnapshotProfileInvalid, match="unsupported KDA state dtype"):
        dtype_element_bytes("float24")


class _FakeModelConfig:
    def __init__(self, *, num_hidden_layers, kda_layers, heads, head_dim, conv):
        self.num_hidden_layers = num_hidden_layers
        self._kda_layers = set(kda_layers)
        self.kda_num_heads = heads
        self.kda_head_dim = head_dim
        self.kda_conv_size = conv

    def is_kda_layer(self, layer_idx):
        # The real configs list KDA layers 1-indexed.
        return (layer_idx + 1) in self._kda_layers


def test_profile_from_model_config_shards_heads_by_attention_group():
    config = _FakeModelConfig(
        num_hidden_layers=6, kda_layers=[1, 2, 4, 5], heads=8, head_dim=8, conv=4
    )
    profile = build_kda_snapshot_profile(
        model_config=config,
        attention_group_size=4,
        recurrent_dtype="float32",
        conv_dtype="bfloat16",
    )
    assert profile.num_kda_layers == 4
    assert profile.num_heads_per_rank == 2
    assert profile.conv_dim_per_rank == 2 * 8
    expected = 4 * 2 * 8 * 8 * 4 + 3 * 4 * (2 * 8) * 3 * 2
    assert snapshot_bytes(profile) == expected


def test_profile_refuses_a_head_count_the_group_cannot_shard():
    config = _FakeModelConfig(
        num_hidden_layers=4, kda_layers=[1, 2], heads=6, head_dim=8, conv=4
    )
    with pytest.raises(KDASnapshotProfileInvalid, match="not divisible"):
        build_kda_snapshot_profile(
            model_config=config,
            attention_group_size=4,
            recurrent_dtype="float32",
            conv_dtype="bfloat16",
        )


def test_profile_refuses_a_model_with_no_kda_layers():
    config = _FakeModelConfig(
        num_hidden_layers=4, kda_layers=[], heads=8, head_dim=8, conv=4
    )
    with pytest.raises(KDASnapshotProfileInvalid, match="at least one KDA layer"):
        build_kda_snapshot_profile(
            model_config=config,
            attention_group_size=1,
            recurrent_dtype="float32",
            conv_dtype="bfloat16",
        )


def test_snapshot_size_scales_with_the_served_model_geometry():
    """The helper reads a real served config, so no constant is copied in.

    Driven by the in-repo Kimi-K3 asset config: the snapshot cost is a pure
    function of its KDA layer list, head count, head dim and conv width, and
    of the attention group size that shards the heads.
    """

    asset = (
        _REPO_ROOT
        / "batchgen"
        / "models"
        / "moonshotai"
        / "kimi_k3"
        / "assets"
        / "config.json"
    )
    text_config = json.loads(asset.read_text())["text_config"]
    linear = text_config["linear_attn_config"]
    config = _FakeModelConfig(
        num_hidden_layers=text_config["num_hidden_layers"],
        kda_layers=linear["kda_layers"],
        heads=linear["num_heads"],
        head_dim=linear["head_dim"],
        conv=linear["short_conv_kernel_size"],
    )
    layers = sum(
        1
        for layer in range(text_config["num_hidden_layers"])
        if (layer + 1) in set(linear["kda_layers"])
    )
    for group_size in (1, 2, 8):
        profile = build_kda_snapshot_profile(
            model_config=config,
            attention_group_size=group_size,
            recurrent_dtype="float32",
            conv_dtype="bfloat16",
        )
        heads_per_rank = linear["num_heads"] // group_size
        head_dim = linear["head_dim"]
        width = linear["short_conv_kernel_size"]
        expected = (
            layers * heads_per_rank * head_dim * head_dim * 4
            + 3 * layers * (heads_per_rank * head_dim) * (width - 1) * 2
        )
        assert snapshot_bytes(profile) == expected
        # One snapshot per trajectory: the per-node cost is the per-rank cost
        # times the group, and it is independent of how the heads are split.
        assert snapshot_bytes(profile) * group_size == (
            layers * linear["num_heads"] * head_dim * head_dim * 4
            + 3 * layers * (linear["num_heads"] * head_dim) * (width - 1) * 2
        )


_STATE_POOL_SOURCE = (
    _REPO_ROOT / "batchgen" / "kv_cache" / "kda_state_gpu_manager.py"
).read_text()


def test_state_pool_declares_the_dtypes_the_sizing_assumes():
    """Pin the sizing to the state pool's declared dtypes.

    ``kda_state_gpu_manager`` cannot be imported without triton, so the
    coupling is pinned against its source: changing a pool dtype without
    changing the snapshot sizing fails here.
    """

    recurrent = re.search(
        r"^\s*recurrent_dtype: torch\.dtype = torch\.(\w+)\s*$",
        _STATE_POOL_SOURCE,
        re.MULTILINE,
    )
    conv = re.search(
        r"^\s*conv_dtype: torch\.dtype = torch\.(\w+)\s*$",
        _STATE_POOL_SOURCE,
        re.MULTILINE,
    )
    assert recurrent is not None and conv is not None
    assert dtype_element_bytes(recurrent.group(1)) == 4
    assert dtype_element_bytes(conv.group(1)) == 2
    profile = build_kda_snapshot_profile(
        model_config=_FakeModelConfig(
            num_hidden_layers=2, kda_layers=[1, 2], heads=4, head_dim=8, conv=4
        ),
        attention_group_size=1,
        recurrent_dtype=recurrent.group(1),
        conv_dtype=conv.group(1),
    )
    assert profile.recurrent_element_bytes == 4
    assert profile.conv_element_bytes == 2


def test_state_pool_declares_three_conv_pools_per_layer():
    """The snapshot copies q, k and v conv state, so the count is pinned."""

    declared = {
        name
        for name in ("_conv_q", "_conv_k", "_conv_v")
        if re.search(rf"^\s*self\.{name} = torch\.zeros\($", _STATE_POOL_SOURCE, re.M)
    }
    assert declared == {"_conv_q", "_conv_k", "_conv_v"}
    assert _profile().num_conv_pools == len(declared)


def test_state_dtype_widths_match_torch_itemsize():
    """The width table agrees with torch, where torch is installed."""

    torch = pytest.importorskip("torch")
    for name in ("float32", "bfloat16", "float16", "float64", "int32", "uint8"):
        dtype = getattr(torch, name)
        assert dtype_element_bytes(name) == torch.empty(0, dtype=dtype).element_size()


# ---------------------------------------------------------------------------
# Record construction: one snapshot, both halves, aligned boundary
# ---------------------------------------------------------------------------


def _record(**overrides):
    kwargs = dict(
        session_key="session",
        namespace_digest=NAMESPACE,
        frontier_token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        page_ids_by_group={0: [10, 11]},
        snapshot=_snapshot(boundary_tokens=8),
        page_tokens=PAGE_TOKENS,
        host_page_bytes_all_layers=HOST_PAGE_BYTES,
        release_record=_Releaser(),
    )
    kwargs.update(overrides)
    return TrajectoryFrontierRecord(**kwargs)


def test_record_holds_both_halves():
    record = _record()
    assert record.frontier_boundary_tokens == 8
    assert record.page_count == 2
    assert record.snapshot.storage_handle == 7
    assert record.refcount == 1
    assert record.is_released is False


def test_record_without_a_snapshot_is_unrepresentable():
    with pytest.raises(FrontierRecordInvalid, match="one KDA snapshot descriptor"):
        _record(snapshot=None)


def test_record_without_pages_is_unrepresentable():
    with pytest.raises(FrontierRecordInvalid, match="pages for every KV group"):
        _record(page_ids_by_group={})
    with pytest.raises(FrontierRecordInvalid, match="no committed pages"):
        _record(page_ids_by_group={0: []})


def test_record_refuses_an_unaligned_frontier_boundary():
    with pytest.raises(FrontierRecordInvalid, match="not aligned"):
        _record(
            frontier_token_ids=[1, 2, 3, 4, 5],
            snapshot=_snapshot(boundary_tokens=5),
        )


def test_record_refuses_a_snapshot_off_the_frontier_boundary():
    with pytest.raises(FrontierRecordInvalid, match="not restorable"):
        _record(snapshot=_snapshot(boundary_tokens=4))


def test_record_refuses_a_page_count_that_does_not_cover_the_frontier():
    with pytest.raises(FrontierRecordInvalid, match="expected 2"):
        _record(page_ids_by_group={0: [10, 11, 12]})


def test_record_refuses_duplicate_pages():
    with pytest.raises(FrontierRecordInvalid, match="repeats a page id"):
        _record(page_ids_by_group={0: [10, 10]})


def test_snapshot_descriptor_refuses_a_zero_handle():
    with pytest.raises(FrontierRecordInvalid, match="non-zero storage handle"):
        _snapshot(boundary_tokens=8, handle=0)


# ---------------------------------------------------------------------------
# Refcount and atomic release
# ---------------------------------------------------------------------------


def test_refcount_to_zero_releases_both_halves_in_one_call():
    releaser = _Releaser()
    record = _record(release_record=releaser)
    record.acquire()
    assert record.refcount == 2
    assert record.release() is None
    assert releaser.calls == []
    outcome = record.release()
    assert outcome is not None
    assert record.refcount == 0 and record.is_released is True
    # One releaser call carrying BOTH halves: pages and snapshot leave together.
    assert len(releaser.calls) == 1
    session_key, pages, handle = releaser.calls[0]
    assert session_key == "session"
    assert pages == {0: (10, 11)}
    assert handle == 7
    assert outcome.released_page_ids_by_group == {0: (10, 11)}
    assert outcome.released_snapshot_handle == 7


def test_double_release_is_refused():
    record = _record()
    record.release()
    with pytest.raises(FrontierRefcountError, match="already\n?.*released"):
        record.release()


def test_acquire_after_release_is_refused():
    record = _record()
    record.release()
    with pytest.raises(FrontierRefcountError, match="cannot be referenced again"):
        record.acquire()


def test_a_failed_physical_release_stays_retryable():
    releaser = _Releaser(fail_times=1)
    record = _record(release_record=releaser)
    with pytest.raises(RuntimeError, match="host page release rejected"):
        record.release()
    # Neither half left, so the record is still whole and still owns one ref.
    assert record.refcount == 1
    assert record.is_released is False
    outcome = record.release()
    assert outcome is not None
    assert len(releaser.calls) == 1


# ---------------------------------------------------------------------------
# Session key derivation and the binary predicate
# ---------------------------------------------------------------------------


def test_same_chain_yields_the_same_session_key():
    tokens = [5, 6, 7, 8]
    first = session_key_for_baseline(
        namespace_digest=NAMESPACE, baseline_token_ids=tokens
    )
    second = session_key_for_baseline(
        namespace_digest=NAMESPACE, baseline_token_ids=list(tokens)
    )
    assert first == second


def test_session_key_separates_chains_and_namespaces():
    tokens = [5, 6, 7, 8]
    base = session_key_for_baseline(
        namespace_digest=NAMESPACE, baseline_token_ids=tokens
    )
    other_chain = session_key_for_baseline(
        namespace_digest=NAMESPACE, baseline_token_ids=[5, 6, 7, 9]
    )
    other_namespace = session_key_for_baseline(
        namespace_digest=(99, 22, 33, 44), baseline_token_ids=tokens
    )
    # A prefix must not collide with the longer chain it is a prefix of.
    shorter = session_key_for_baseline(
        namespace_digest=NAMESPACE, baseline_token_ids=[5, 6]
    )
    assert len({base, other_chain, other_namespace, shorter}) == 4


def test_session_key_is_stable_across_frontier_advance():
    registry = _registry()
    baseline = [1, 2, 3, 4]
    record = _begin(registry, tokens=baseline)
    advanced = registry.advance(
        session_key=record.session_key,
        namespace_digest=NAMESPACE,
        frontier_token_ids=baseline + [5, 6, 7, 8],
        page_ids_by_group=_pages(2),
        snapshot=_snapshot(boundary_tokens=8, handle=8),
    )
    assert advanced.record.session_key == record.session_key


def test_frontier_index_key_depends_on_the_exact_frontier_tokens():
    same = frontier_index_key(
        namespace_digest=NAMESPACE, frontier_token_ids=[1, 2, 3, 4]
    )
    assert same == frontier_index_key(
        namespace_digest=NAMESPACE, frontier_token_ids=(1, 2, 3, 4)
    )
    assert same != frontier_index_key(
        namespace_digest=NAMESPACE, frontier_token_ids=[1, 2, 3, 4, 5, 6, 7, 8]
    )
    # The index key is domain-separated from the session key.
    assert same != session_key_for_baseline(
        namespace_digest=NAMESPACE, baseline_token_ids=[1, 2, 3, 4]
    )


def test_longest_common_prefix_length_cases():
    assert longest_common_prefix_length([1, 2, 3], [1, 2, 3]) == 3
    assert longest_common_prefix_length([1, 2, 3], [1, 2, 3, 4, 5]) == 3
    assert longest_common_prefix_length([1, 2, 9], [1, 2, 3]) == 2
    assert longest_common_prefix_length([9, 2, 3], [1, 2, 3]) == 0
    assert longest_common_prefix_length([], [1, 2]) == 0


def test_hit_predicate_is_binary_on_the_frontier():
    assert is_frontier_hit(lcp_tokens=8, frontier_boundary_tokens=8) is True
    assert is_frontier_hit(lcp_tokens=20, frontier_boundary_tokens=8) is True
    # One token short of the frontier is a FULL miss, not a partial hit.
    assert is_frontier_hit(lcp_tokens=7, frontier_boundary_tokens=8) is False
    assert is_frontier_hit(lcp_tokens=0, frontier_boundary_tokens=8) is False


def test_hit_predicate_refuses_a_nonsensical_boundary():
    with pytest.raises(FrontierRecordInvalid):
        is_frontier_hit(lcp_tokens=4, frontier_boundary_tokens=0)
    with pytest.raises(FrontierRecordInvalid):
        is_frontier_hit(lcp_tokens=-1, frontier_boundary_tokens=4)


# ---------------------------------------------------------------------------
# Resolution against the committed frontiers
# ---------------------------------------------------------------------------


def test_continuation_turn_resolves_to_the_committed_session():
    registry = _registry()
    frontier = [1, 2, 3, 4, 5, 6, 7, 8]
    record = _begin(registry, tokens=frontier)
    resolution = registry.resolve(
        namespace_digest=NAMESPACE,
        prompt_token_ids=frontier + [31, 32, 33],
    )
    assert resolution.hit is True
    assert resolution.record is record
    assert resolution.session_key == record.session_key
    assert resolution.lcp_tokens == 8
    assert resolution.frontier_boundary_tokens == 8


def test_exact_frontier_length_prompt_is_a_hit():
    registry = _registry()
    frontier = [1, 2, 3, 4]
    _begin(registry, tokens=frontier)
    resolution = registry.resolve(
        namespace_digest=NAMESPACE, prompt_token_ids=frontier
    )
    assert resolution.hit is True
    assert resolution.lcp_tokens == 4


def test_re_rendered_prompt_below_the_frontier_is_a_full_miss():
    registry = _registry()
    frontier = [1, 2, 3, 4, 5, 6, 7, 8]
    _begin(registry, tokens=frontier)
    # The re-render diverges inside the last committed page, so the LCP is
    # below the frontier boundary even though most tokens agree.
    resolution = registry.resolve(
        namespace_digest=NAMESPACE,
        prompt_token_ids=[1, 2, 3, 4, 5, 99, 7, 8, 31, 32],
    )
    assert resolution.hit is False
    assert resolution.record is None
    assert resolution.session_key is None
    # No partial credit is offered for the five agreeing tokens.
    assert resolution.frontier_boundary_tokens == 0


def test_prompt_shorter_than_the_frontier_is_a_miss():
    registry = _registry()
    _begin(registry, tokens=[1, 2, 3, 4, 5, 6, 7, 8])
    resolution = registry.resolve(
        namespace_digest=NAMESPACE, prompt_token_ids=[1, 2, 3, 4]
    )
    assert resolution.hit is False


def test_unrelated_chain_is_a_miss():
    registry = _registry()
    _begin(registry, tokens=[1, 2, 3, 4])
    resolution = registry.resolve(
        namespace_digest=NAMESPACE, prompt_token_ids=[71, 72, 73, 74, 75]
    )
    assert resolution.hit is False
    assert resolution.lcp_tokens == 0


def test_a_different_namespace_never_resolves():
    registry = _registry()
    frontier = [1, 2, 3, 4]
    _begin(registry, tokens=frontier)
    resolution = registry.resolve(
        namespace_digest=(99, 22, 33, 44), prompt_token_ids=frontier + [5]
    )
    assert resolution.hit is False


def test_longest_matching_frontier_wins():
    registry = _registry()
    short = _begin(registry, tokens=[1, 2, 3, 4], handle=7)
    long_tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    long_record = registry.begin(
        namespace_digest=NAMESPACE,
        frontier_token_ids=long_tokens,
        page_ids_by_group=_pages(2, base=500),
        snapshot=_snapshot(boundary_tokens=8, handle=8),
    )
    resolution = registry.resolve(
        namespace_digest=NAMESPACE, prompt_token_ids=long_tokens + [9]
    )
    assert resolution.record is long_record
    assert resolution.record is not short
    registry.check_invariants()


def test_resolution_refuses_an_empty_prompt():
    registry = _registry()
    with pytest.raises(FrontierRegistryError, match="non-empty prompt"):
        registry.resolve(namespace_digest=NAMESPACE, prompt_token_ids=[])


# ---------------------------------------------------------------------------
# Registry lifecycle: advance, replace ordering, eviction
# ---------------------------------------------------------------------------


def test_advance_replaces_the_record_and_releases_the_old_one():
    releaser = _Releaser()
    registry = _registry(releaser=releaser)
    baseline = [1, 2, 3, 4]
    first = _begin(registry, tokens=baseline)
    advanced = registry.advance(
        session_key=first.session_key,
        namespace_digest=NAMESPACE,
        frontier_token_ids=baseline + [5, 6, 7, 8],
        page_ids_by_group=_pages(2, base=500),
        snapshot=_snapshot(boundary_tokens=8, handle=8),
    )
    assert len(registry) == 1
    assert registry.get(first.session_key) is advanced.record
    assert advanced.record.frontier_boundary_tokens == 8
    # Exactly one snapshot survives: the old record left whole.
    assert advanced.previous_release is not None
    assert advanced.previous_release.released_snapshot_handle == 7
    assert len(releaser.calls) == 1
    assert first.is_released is True
    registry.check_invariants()
    # The old frontier no longer resolves; the new one does.
    assert (
        registry.resolve(
            namespace_digest=NAMESPACE, prompt_token_ids=baseline + [99]
        ).hit
        is False
    )
    assert (
        registry.resolve(
            namespace_digest=NAMESPACE,
            prompt_token_ids=baseline + [5, 6, 7, 8, 99],
        ).record
        is advanced.record
    )


def test_advance_never_accumulates_snapshots():
    registry = _registry()
    tokens = [1, 2, 3, 4]
    record = _begin(registry, tokens=tokens)
    handles = {record.snapshot.storage_handle}
    for step in range(1, 4):
        tokens = tokens + [10 + step, 20 + step, 30 + step, 40 + step]
        advanced = registry.advance(
            session_key=record.session_key,
            namespace_digest=NAMESPACE,
            frontier_token_ids=tokens,
            page_ids_by_group=_pages(len(tokens) // PAGE_TOKENS, base=1000 * step),
            snapshot=_snapshot(boundary_tokens=len(tokens), handle=100 + step),
        )
        handles.add(advanced.record.snapshot.storage_handle)
        assert len(registry) == 1
        # The registry holds exactly one snapshot at any time.
        assert registry.get(record.session_key).snapshot.storage_handle == 100 + step
        registry.check_invariants()
    assert len(handles) == 4


def test_failure_before_install_leaves_the_old_record_intact():
    releaser = _Releaser()
    registry = _registry(releaser=releaser)
    baseline = [1, 2, 3, 4]
    first = _begin(registry, tokens=baseline)
    # A snapshot taken off the new frontier boundary: the new record cannot be
    # built, so the replace must not have started.
    with pytest.raises(FrontierRecordInvalid):
        registry.advance(
            session_key=first.session_key,
            namespace_digest=NAMESPACE,
            frontier_token_ids=baseline + [5, 6, 7, 8],
            page_ids_by_group=_pages(2, base=500),
            snapshot=_snapshot(boundary_tokens=4, handle=8),
        )
    assert registry.get(first.session_key) is first
    assert first.is_released is False
    assert first.refcount == 1
    assert releaser.calls == []
    registry.check_invariants()
    assert (
        registry.resolve(
            namespace_digest=NAMESPACE, prompt_token_ids=baseline + [9]
        ).record
        is first
    )


def test_failure_releasing_the_old_record_keeps_the_registry_consistent():
    """The simulated crash between referencing the new and releasing the old."""

    releaser = _Releaser(fail_times=1)
    registry = _registry(releaser=releaser)
    baseline = [1, 2, 3, 4]
    first = _begin(registry, tokens=baseline)
    advanced_tokens = baseline + [5, 6, 7, 8]
    with pytest.raises(FrontierReleaseFailed) as excinfo:
        registry.advance(
            session_key=first.session_key,
            namespace_digest=NAMESPACE,
            frontier_token_ids=advanced_tokens,
            page_ids_by_group=_pages(2, base=500),
            snapshot=_snapshot(boundary_tokens=8, handle=8),
        )
    # The new record was fully referenced before the old one was touched.
    installed = registry.get(first.session_key)
    assert installed is not first
    assert installed.frontier_boundary_tokens == 8
    assert installed.snapshot.storage_handle == 8
    # The old record is intact: neither half left.
    assert excinfo.value.record is first
    assert first.is_released is False
    assert first.refcount == 1
    assert releaser.calls == []
    registry.check_invariants()
    assert (
        registry.resolve(
            namespace_digest=NAMESPACE, prompt_token_ids=advanced_tokens + [9]
        ).record
        is installed
    )
    # And the deferred release can still be completed, whole.
    outcome = excinfo.value.record.release()
    assert outcome.released_snapshot_handle == 7
    assert len(releaser.calls) == 1


def test_advance_refuses_an_unknown_session():
    registry = _registry()
    with pytest.raises(FrontierRegistryError, match="no committed frontier"):
        registry.advance(
            session_key="absent",
            namespace_digest=NAMESPACE,
            frontier_token_ids=[1, 2, 3, 4],
            page_ids_by_group=_pages(1),
            snapshot=_snapshot(boundary_tokens=4),
        )


def test_advance_refuses_a_backwards_frontier():
    registry = _registry()
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    record = _begin(registry, tokens=tokens)
    with pytest.raises(FrontierRegistryError, match="advance backwards"):
        registry.advance(
            session_key=record.session_key,
            namespace_digest=NAMESPACE,
            frontier_token_ids=tokens[:4],
            page_ids_by_group=_pages(1, base=500),
            snapshot=_snapshot(boundary_tokens=4, handle=8),
        )
    assert registry.get(record.session_key) is record


def test_advance_refuses_moving_a_session_to_another_group():
    registry = _registry()
    tokens = [1, 2, 3, 4]
    record = _begin(registry, tokens=tokens, group_id=1)
    with pytest.raises(FrontierRegistryError, match="pinned to KV group 1"):
        registry.advance(
            session_key=record.session_key,
            namespace_digest=NAMESPACE,
            frontier_token_ids=tokens + [5, 6, 7, 8],
            page_ids_by_group=_pages(2, base=500),
            snapshot=_snapshot(boundary_tokens=8, handle=8, group_id=2),
        )
    assert registry.get(record.session_key) is record
    assert record.is_released is False


def test_begin_refuses_a_session_that_already_has_a_frontier():
    registry = _registry()
    tokens = [1, 2, 3, 4]
    _begin(registry, tokens=tokens)
    with pytest.raises(FrontierRegistryError, match="already has a committed frontier"):
        _begin(registry, tokens=tokens)


def test_eviction_removes_pages_and_snapshot_together():
    releaser = _Releaser()
    registry = _registry(releaser=releaser)
    record = _begin(registry, tokens=[1, 2, 3, 4, 5, 6, 7, 8])
    outcome = registry.evict(record.session_key)
    assert len(registry) == 0
    assert registry.get(record.session_key) is None
    assert outcome.released_page_ids_by_group == {0: (100, 101)}
    assert outcome.released_snapshot_handle == 7
    assert len(releaser.calls) == 1
    assert record.is_released is True
    registry.check_invariants()
    # An evicted frontier is a miss by construction, not an MLA-only remnant.
    assert (
        registry.resolve(
            namespace_digest=NAMESPACE,
            prompt_token_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        ).hit
        is False
    )


def test_eviction_is_refused_while_the_record_is_attached():
    releaser = _Releaser()
    registry = _registry(releaser=releaser)
    record = _begin(registry, tokens=[1, 2, 3, 4])
    record.acquire()
    with pytest.raises(FrontierRefcountError, match="while it is attached"):
        registry.evict(record.session_key)
    assert registry.get(record.session_key) is record
    assert releaser.calls == []
    registry.check_invariants()


def test_eviction_refuses_an_unknown_session():
    registry = _registry()
    with pytest.raises(FrontierRegistryError, match="no committed frontier"):
        registry.evict("absent")


def test_multi_group_frontier_releases_every_group():
    releaser = _Releaser()
    registry = _registry(releaser=releaser)
    record = _begin(registry, tokens=[1, 2, 3, 4, 5, 6, 7, 8], groups=(0, 1))
    assert record.page_count == 4
    outcome = registry.evict(record.session_key)
    assert set(outcome.released_page_ids_by_group) == {0, 1}
    assert len(releaser.calls) == 1


# ---------------------------------------------------------------------------
# Accounting seam
# ---------------------------------------------------------------------------


def test_charge_is_pages_plus_snapshot():
    registry = _registry()
    record = _begin(registry, tokens=[1, 2, 3, 4, 5, 6, 7, 8])
    charge = record.charge()
    assert charge.page_count == 2
    assert charge.snapshot_bytes == snapshot_bytes(_profile())
    assert charge.page_bytes == 2 * HOST_PAGE_BYTES
    assert charge.total_bytes == charge.page_bytes + charge.snapshot_bytes
    # The #454 owner charge counts whole Host pages, so the snapshot is
    # converted into that unit, rounded up, and added to the retained pages.
    expected_snapshot_pages = -(-charge.snapshot_bytes // HOST_PAGE_BYTES)
    assert charge.snapshot_host_pages == expected_snapshot_pages
    assert charge.owned_host_pages == 2 + expected_snapshot_pages


def test_snapshot_page_charge_rounds_up_so_the_planner_cannot_overpromise():
    registry = _registry()
    # A snapshot that is one byte over a whole page must cost two pages.
    profile = _profile(
        num_kda_layers=1,
        num_heads_per_rank=1,
        head_dim=16,
        conv_dim_per_rank=4,
        conv_width=2,
        recurrent_element_bytes=4,
        conv_element_bytes=2,
    )
    assert snapshot_bytes(profile) == 16 * 16 * 4 + 3 * 4 * 1 * 2
    record = registry.begin(
        namespace_digest=NAMESPACE,
        frontier_token_ids=[1, 2, 3, 4],
        page_ids_by_group=_pages(1),
        snapshot=_snapshot(boundary_tokens=4, profile=profile),
    )
    charge = record.charge()
    assert charge.snapshot_bytes == 1048
    assert charge.snapshot_host_pages == 1
    bigger = _profile(
        num_kda_layers=2,
        num_heads_per_rank=1,
        head_dim=32,
        conv_dim_per_rank=4,
        conv_width=2,
        recurrent_element_bytes=4,
        conv_element_bytes=2,
    )
    # 2 * 32 * 32 * 4 + 3 * 2 * 4 * 1 * 2 = 8240 bytes over 2048-byte pages.
    assert snapshot_bytes(bigger) == 8240
    assert -(-8240 // HOST_PAGE_BYTES) == 5


def test_total_charge_sums_every_resident_frontier():
    registry = _registry()
    _begin(registry, tokens=[1, 2, 3, 4, 5, 6, 7, 8])
    registry.begin(
        namespace_digest=NAMESPACE,
        frontier_token_ids=[71, 72, 73, 74],
        page_ids_by_group=_pages(1, base=900),
        snapshot=_snapshot(boundary_tokens=4, handle=9),
    )
    total = registry.total_charge()
    assert total.page_count == 3
    assert total.snapshot_bytes == 2 * snapshot_bytes(_profile())
    assert total.total_bytes == 3 * HOST_PAGE_BYTES + total.snapshot_bytes


def test_release_outcome_carries_the_charge_that_left():
    registry = _registry()
    record = _begin(registry, tokens=[1, 2, 3, 4, 5, 6, 7, 8])
    expected = record.charge()
    outcome = registry.evict(record.session_key)
    assert outcome.charge.owned_host_pages == expected.owned_host_pages
    assert outcome.charge.total_bytes == expected.total_bytes


def test_registry_refuses_a_non_positive_page_byte_size():
    with pytest.raises(FrontierRegistryError, match="host_page_bytes_all_layers"):
        TrajectoryFrontierRegistry(
            page_tokens=PAGE_TOKENS,
            host_page_bytes_all_layers=0,
            release_record=_Releaser(),
        )


def test_registry_refuses_a_non_positive_page_size():
    with pytest.raises(FrontierRegistryError, match="page_tokens"):
        TrajectoryFrontierRegistry(
            page_tokens=0,
            host_page_bytes_all_layers=HOST_PAGE_BYTES,
            release_record=_Releaser(),
        )


# ---------------------------------------------------------------------------
# Slice 1c: the snapshot binds G head shards, and the charge counts all of them
# ---------------------------------------------------------------------------


def test_a_snapshot_defaults_to_one_shard():
    """Pure DP is unchanged: one rank, one shard, the pre-Slice-1c numbers."""

    snapshot = _snapshot(boundary_tokens=8)
    assert snapshot.shard_count == 1
    assert snapshot.snapshot_total_bytes == snapshot.snapshot_bytes


def test_a_snapshot_binds_every_shard_of_its_owning_group():
    snapshot = _snapshot(boundary_tokens=8, shard_count=4)
    assert snapshot.shard_count == 4
    # ``profile`` stays ONE rank's geometry -- what a single rank reads and
    # writes -- while the descriptor says how many of those the record binds.
    assert snapshot.snapshot_bytes == snapshot_bytes(snapshot.profile)
    assert snapshot.snapshot_total_bytes == 4 * snapshot.snapshot_bytes


def test_a_snapshot_needs_at_least_one_shard():
    for bad in (0, -1):
        with pytest.raises(FrontierRecordInvalid) as excinfo:
            _snapshot(boundary_tokens=8, shard_count=bad)
        assert "binds at least one head shard" in str(excinfo.value)


def test_the_charge_rounds_each_shard_up_independently():
    """A rank writes through its own page window, so its shard owns whole pages.

    Rounding the summed bytes once would give a smaller arena whose windows
    straddle a page boundary -- two ranks writing one page.
    """

    charge = FrontierCharge(
        page_count=2,
        host_page_bytes_all_layers=HOST_PAGE_BYTES,
        snapshot_bytes=HOST_PAGE_BYTES + 1,
        shard_count=4,
    )
    assert charge.snapshot_host_pages == 2
    assert charge.arena_host_pages == 8
    # The "round once" figure would have been 5 pages: strictly too small.
    assert -(-(4 * (HOST_PAGE_BYTES + 1)) // HOST_PAGE_BYTES) == 5
    assert charge.owned_host_pages == 2 + 8
    assert charge.snapshot_total_bytes == 4 * (HOST_PAGE_BYTES + 1)
    assert charge.total_bytes == charge.page_bytes + charge.snapshot_total_bytes


def test_a_one_shard_charge_is_the_pre_slice_1c_charge():
    charge = FrontierCharge(
        page_count=3,
        host_page_bytes_all_layers=HOST_PAGE_BYTES,
        snapshot_bytes=HOST_PAGE_BYTES + 1,
    )
    assert charge.shard_count == 1
    assert charge.arena_host_pages == charge.snapshot_host_pages == 2
    assert charge.owned_host_pages == 3 + 2
    assert charge.total_bytes == charge.page_bytes + charge.snapshot_bytes


def test_a_charge_needs_at_least_one_shard():
    with pytest.raises(KDASnapshotProfileInvalid) as excinfo:
        FrontierCharge(
            page_count=1,
            host_page_bytes_all_layers=HOST_PAGE_BYTES,
            snapshot_bytes=16,
            shard_count=0,
        )
    assert "at least one snapshot shard" in str(excinfo.value)


def test_a_record_reports_its_snapshot_shard_count():
    registry = _registry()
    record = _begin(registry, tokens=range(8), shard_count=4)
    assert record.shard_count == 4
    assert record.charge().shard_count == 4
    assert record.charge().arena_host_pages == (
        4 * record.charge().snapshot_host_pages
    )


def test_advance_refuses_a_shard_count_change():
    """A session's head split cannot move: its arena windows are that split."""

    registry = _registry()
    record = _begin(registry, tokens=range(8), shard_count=4)
    with pytest.raises(FrontierRegistryError) as excinfo:
        registry.advance(
            session_key=record.session_key,
            namespace_digest=NAMESPACE,
            frontier_token_ids=range(12),
            page_ids_by_group=_pages(3),
            snapshot=_snapshot(boundary_tokens=12, handle=8, shard_count=2),
        )
    assert "4-shard snapshot" in str(excinfo.value)
    assert "2-shard" in str(excinfo.value)
    # Refused BEFORE anything moved: the old frontier is untouched.
    assert len(registry) == 1
    assert registry.require(record.session_key) is record
    assert record.frontier_boundary_tokens == 8


def test_advance_keeps_the_shard_count():
    registry = _registry()
    record = _begin(registry, tokens=range(8), shard_count=4)
    advance = registry.advance(
        session_key=record.session_key,
        namespace_digest=NAMESPACE,
        frontier_token_ids=range(12),
        page_ids_by_group=_pages(3),
        snapshot=_snapshot(boundary_tokens=12, handle=8, shard_count=4),
    )
    assert advance.record.shard_count == 4
    registry.check_invariants()


def test_total_charge_refuses_records_that_disagree_on_shard_count():
    """A mix means one record outlived a reconfiguration: its arena windows no
    longer match the group that would restore it. Reported, not averaged."""

    registry = _registry()
    _begin(registry, tokens=range(8), shard_count=4)
    _begin(registry, tokens=range(100, 108), handle=9, shard_count=2)
    with pytest.raises(FrontierRegistryInvariantError) as excinfo:
        registry.total_charge()
    assert "disagree on their snapshot shard count" in str(excinfo.value)
    assert "[2, 4]" in str(excinfo.value)


def test_total_charge_of_matching_records_counts_every_shard():
    registry = _registry()
    first = _begin(registry, tokens=range(8), shard_count=4)
    second = _begin(registry, tokens=range(100, 108), handle=9, shard_count=4)
    charge = registry.total_charge()
    assert charge.shard_count == 4
    assert charge.page_count == first.page_count + second.page_count
    assert charge.snapshot_bytes == (
        first.snapshot.snapshot_bytes + second.snapshot.snapshot_bytes
    )
    assert charge.arena_host_pages == 4 * charge.snapshot_host_pages


def test_an_empty_registry_still_reports_a_usable_charge():
    charge = _registry().total_charge()
    assert charge.shard_count == 1
    assert charge.page_count == 0
    assert charge.arena_host_pages == 0
