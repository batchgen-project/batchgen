"""Host-side trajectory frontier bookkeeping for cross-wave prefix reuse.

One trajectory is one session. A session owns exactly ONE frontier record:
the committed MLA prefix pages up to a page-aligned frontier boundary, plus
ONE KDA snapshot object taken at that same boundary. A hit requires both, so
the two halves share a refcount and leave together -- a pages-without-snapshot
or snapshot-without-pages record cannot be constructed.

This module is pure Host-coordinator bookkeeping: no torch, no device work, no
worker imports. It owns the record lifetime, the session-key derivation, the
binary frontier predicate and the charge report; it does NOT perform the D2H
copy, the page retention, or the admission wiring.

The page-aligned boundary itself is produced by
``batchgen.prefix_reuse.commit.aligned_prefix_tokens``; this module only
validates that what it is handed is aligned, so alignment keeps one owner.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence


# ---------------------------------------------------------------------------
# Errors -- every refusal is named; nothing degrades silently.
# ---------------------------------------------------------------------------


class TrajectoryFrontierError(RuntimeError):
    """Base class for every trajectory-frontier refusal."""


class FrontierRecordInvalid(TrajectoryFrontierError):
    """A record was asked for in a state the mechanism does not allow."""


class FrontierRefcountError(TrajectoryFrontierError):
    """A reference was taken or dropped outside the record's lifetime."""


class FrontierReleaseFailed(TrajectoryFrontierError):
    """The physical release of a record's pages and snapshot failed.

    Carries the record so the caller can retry the release; the registry is
    already consistent when this is raised.
    """

    def __init__(self, message: str, *, record: "TrajectoryFrontierRecord"):
        super().__init__(message)
        self.record = record


class FrontierRegistryError(TrajectoryFrontierError):
    """A registry operation named a session that cannot serve it."""


class FrontierRegistryInvariantError(TrajectoryFrontierError):
    """The registry's always-true relations do not hold."""


class KDASnapshotProfileInvalid(TrajectoryFrontierError):
    """The KDA snapshot geometry cannot describe a real state slot."""


class SessionGroupMapError(TrajectoryFrontierError):
    """A session-group summary does not describe a real owning group."""


# ---------------------------------------------------------------------------
# KDA snapshot sizing -- derived from model + parallel config, never a constant
# ---------------------------------------------------------------------------

# ``KDAStateGPUManager`` allocates three short-causal-conv pools per KDA layer
# (``_conv_q``, ``_conv_k``, ``_conv_v``), each of per-layer view shape
# ``(num_state_items, conv_dim, conv_width - 1)``. That count is structural, not
# a size: a snapshot of one slot must copy all three.
_KDA_CONV_POOLS_PER_LAYER = 3

# A snapshot holds ONE state slot per KDA layer: the fla kernel updates a
# sequence's recurrent and conv state in place, so a sequence owns exactly one
# slot for its whole lifetime (``KDAStateGPUConfig`` docstring).
_KDA_STATE_SLOTS_PER_TRAJECTORY = 1

_DTYPE_ELEMENT_BYTES = {
    "float64": 8,
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "int32": 4,
    "uint8": 1,
}


def dtype_element_bytes(dtype_name: str) -> int:
    """Element width of a KDA state dtype, by name.

    An unsupported dtype is refused rather than guessed: a wrong width would
    silently mis-charge the Host pool.
    """

    key = str(dtype_name).strip().lower()
    key = key.rsplit(".", 1)[-1]
    if key not in _DTYPE_ELEMENT_BYTES:
        raise KDASnapshotProfileInvalid(
            f"unsupported KDA state dtype {dtype_name!r}; known dtypes are "
            + ", ".join(sorted(_DTYPE_ELEMENT_BYTES))
        )
    return _DTYPE_ELEMENT_BYTES[key]


@dataclass(frozen=True)
class KDASnapshotProfile:
    """Geometry of ONE trajectory's KDA state as ONE rank holds it.

    Mirrors ``KDAStateGPUConfig``: the recurrent pool is
    ``(layers, slots, heads, head_dim, head_dim)`` and each of the three conv
    pools is ``(layers, slots, conv_dim, conv_width - 1)``. Heads and
    ``conv_dim`` are the per-rank shard, because the KDA state is head-sharded
    exactly as the owning group holds it.
    """

    num_kda_layers: int
    num_heads_per_rank: int
    head_dim: int
    conv_dim_per_rank: int
    conv_width: int
    recurrent_element_bytes: int
    conv_element_bytes: int
    num_conv_pools: int = _KDA_CONV_POOLS_PER_LAYER
    num_state_slots: int = _KDA_STATE_SLOTS_PER_TRAJECTORY

    def __post_init__(self) -> None:
        positive = {
            "num_kda_layers": self.num_kda_layers,
            "num_heads_per_rank": self.num_heads_per_rank,
            "head_dim": self.head_dim,
            "conv_dim_per_rank": self.conv_dim_per_rank,
            "recurrent_element_bytes": self.recurrent_element_bytes,
            "conv_element_bytes": self.conv_element_bytes,
            "num_conv_pools": self.num_conv_pools,
            "num_state_slots": self.num_state_slots,
        }
        for name, value in positive.items():
            if int(value) <= 0:
                raise KDASnapshotProfileInvalid(
                    f"KDA snapshot profile needs a positive {name}, got {value}"
                )
        if int(self.conv_width) < 2:
            raise KDASnapshotProfileInvalid(
                "KDA conv pools store conv_width - 1 entries per slot, so "
                f"conv_width must be at least 2, got {self.conv_width}"
            )

    @property
    def recurrent_bytes(self) -> int:
        """Bytes of the recurrent state across all KDA layers on one rank."""

        return (
            int(self.num_kda_layers)
            * int(self.num_state_slots)
            * int(self.num_heads_per_rank)
            * int(self.head_dim)
            * int(self.head_dim)
            * int(self.recurrent_element_bytes)
        )

    @property
    def conv_bytes(self) -> int:
        """Bytes of the q/k/v short-conv states across all KDA layers."""

        return (
            int(self.num_conv_pools)
            * int(self.num_kda_layers)
            * int(self.num_state_slots)
            * int(self.conv_dim_per_rank)
            * (int(self.conv_width) - 1)
            * int(self.conv_element_bytes)
        )

    @property
    def total_bytes(self) -> int:
        return self.recurrent_bytes + self.conv_bytes


def snapshot_bytes(profile: KDASnapshotProfile) -> int:
    """Bytes one KDA snapshot costs on one rank.

    Derived from layers x per-rank head shard x (recurrent + conv) state dims
    x dtype width. There is no workload constant here: change the model or the
    attention group size and this follows.
    """

    if not isinstance(profile, KDASnapshotProfile):
        raise KDASnapshotProfileInvalid(
            "snapshot_bytes needs a KDASnapshotProfile, got "
            f"{type(profile).__name__}"
        )
    return profile.total_bytes


def build_kda_snapshot_profile(
    *,
    model_config: object,
    attention_group_size: int,
    recurrent_dtype: str,
    conv_dtype: str,
) -> KDASnapshotProfile:
    """Derive the per-rank snapshot geometry from the served model's config.

    Reads the same attributes the parallel strategy manager reads when it
    builds the real state pool: the KDA layer predicate over
    ``num_hidden_layers``, ``kda_num_heads``, ``kda_head_dim`` and
    ``kda_conv_size``. Heads are sharded by ``attention_group_size`` and the
    conv dimension follows the shard, as the pool's ``proj_size`` does.
    """

    group_size = int(attention_group_size)
    if group_size <= 0:
        raise KDASnapshotProfileInvalid(
            f"attention_group_size must be positive, got {attention_group_size}"
        )
    try:
        num_hidden_layers = int(model_config.num_hidden_layers)
        is_kda_layer = model_config.is_kda_layer
        num_heads = int(model_config.kda_num_heads)
        head_dim = int(model_config.kda_head_dim)
        conv_width = int(model_config.kda_conv_size)
    except AttributeError as exc:
        raise KDASnapshotProfileInvalid(
            "model config does not expose the KDA state geometry required for "
            f"a trajectory snapshot: {exc}"
        ) from exc
    if num_hidden_layers <= 0:
        raise KDASnapshotProfileInvalid(
            f"model config has no layers, got num_hidden_layers={num_hidden_layers}"
        )
    num_kda_layers = sum(
        1 for layer in range(num_hidden_layers) if is_kda_layer(layer)
    )
    if num_kda_layers <= 0:
        raise KDASnapshotProfileInvalid(
            "a trajectory snapshot needs at least one KDA layer; this model "
            "has none, so it has no KDA state to snapshot"
        )
    if num_heads % group_size:
        raise KDASnapshotProfileInvalid(
            f"kda_num_heads={num_heads} is not divisible by "
            f"attention_group_size={group_size}, so the KDA state is not "
            "head-shardable across the owning group"
        )
    heads_per_rank = num_heads // group_size
    return KDASnapshotProfile(
        num_kda_layers=num_kda_layers,
        num_heads_per_rank=heads_per_rank,
        head_dim=head_dim,
        conv_dim_per_rank=heads_per_rank * head_dim,
        conv_width=conv_width,
        recurrent_element_bytes=dtype_element_bytes(recurrent_dtype),
        conv_element_bytes=dtype_element_bytes(conv_dtype),
    )


@dataclass(frozen=True)
class KDASnapshotDescriptor:
    """The ONE KDA snapshot object a frontier record owns.

    ``storage_handle`` is the Host-side identity of the snapshot buffer; this
    slice only carries it, the commit path fills it. ``owning_group_id`` keeps
    the group affinity the head sharding implies: a snapshot is only restorable
    into the group whose layout produced it.

    ``shard_count`` is how many per-rank head shards the snapshot binds -- G,
    the attention group size at commit time. It is part of the record's
    identity, not a hint: a record taken at one G describes a head split that
    a different G does not have, so a restore under a different G is refused by
    name rather than reinterpreted. ``profile`` stays the geometry of ONE
    shard, which is what a single rank reads and writes.
    """

    storage_handle: int
    owning_group_id: int
    profile: KDASnapshotProfile
    boundary_tokens: int
    shard_count: int = 1

    def __post_init__(self) -> None:
        if int(self.shard_count) <= 0:
            raise FrontierRecordInvalid(
                "a KDA snapshot binds at least one head shard, got "
                f"shard_count={self.shard_count}"
            )
        if int(self.storage_handle) <= 0:
            raise FrontierRecordInvalid(
                "a KDA snapshot descriptor needs a non-zero storage handle; "
                f"got {self.storage_handle}"
            )
        if int(self.owning_group_id) < 0:
            raise FrontierRecordInvalid(
                f"owning_group_id must not be negative, got {self.owning_group_id}"
            )
        if int(self.boundary_tokens) <= 0:
            raise FrontierRecordInvalid(
                "a snapshot is taken at a positive frontier boundary, got "
                f"{self.boundary_tokens}"
            )
        if not isinstance(self.profile, KDASnapshotProfile):
            raise FrontierRecordInvalid(
                "a KDA snapshot descriptor needs a KDASnapshotProfile"
            )

    @property
    def snapshot_bytes(self) -> int:
        """Bytes of ONE rank's shard of this snapshot."""

        return snapshot_bytes(self.profile)

    @property
    def snapshot_total_bytes(self) -> int:
        """Bytes of every shard this snapshot binds."""

        return int(self.shard_count) * snapshot_bytes(self.profile)


# ---------------------------------------------------------------------------
# Session key derivation and the binary frontier predicate
# ---------------------------------------------------------------------------

_SESSION_KEY_DIGEST_BYTES = 16
_BASELINE_DOMAIN = b"batchgen.trajectory.baseline"
_FRONTIER_DOMAIN = b"batchgen.trajectory.frontier"


def _prefix_digest(
    *,
    domain: bytes,
    namespace_digest: Sequence[int],
    token_ids: Sequence[int],
) -> str:
    hasher = hashlib.blake2b(digest_size=_SESSION_KEY_DIGEST_BYTES)
    hasher.update(domain)
    hasher.update(b"\0")
    for value in namespace_digest:
        hasher.update(int(value).to_bytes(8, "little", signed=False))
    hasher.update(b"\0")
    hasher.update(len(token_ids).to_bytes(8, "little"))
    for token_id in token_ids:
        hasher.update(int(token_id).to_bytes(8, "little", signed=True))
    return hasher.hexdigest()


def session_key_for_baseline(
    *, namespace_digest: Sequence[int], baseline_token_ids: Sequence[int]
) -> str:
    """The stable session handle of the trajectory born at this baseline.

    Derived from the request chain alone -- the namespace digest of the serving
    configuration plus the trajectory's first committed page-aligned prefix --
    so no new batch field is needed. The key is fixed at session birth and does
    NOT move when the frontier advances; identity is the trajectory, not its
    current length.
    """

    if not baseline_token_ids:
        raise FrontierRegistryError(
            "a session baseline needs at least one committed token"
        )
    return _prefix_digest(
        domain=_BASELINE_DOMAIN,
        namespace_digest=namespace_digest,
        token_ids=baseline_token_ids,
    )


def frontier_index_key(
    *, namespace_digest: Sequence[int], frontier_token_ids: Sequence[int]
) -> str:
    """Lookup key of a committed frontier's exact token content.

    Equal keys mean the frontier tokens are equal, so probing this index with
    a candidate prompt's page-aligned prefix finds the sessions whose frontier
    that prompt could continue. The predicate is still decided on the tokens
    themselves -- this index only narrows the candidates.
    """

    if not frontier_token_ids:
        raise FrontierRegistryError("a frontier needs at least one token")
    return _prefix_digest(
        domain=_FRONTIER_DOMAIN,
        namespace_digest=namespace_digest,
        token_ids=frontier_token_ids,
    )


def longest_common_prefix_length(
    left: Sequence[int], right: Sequence[int]
) -> int:
    """Token count the two chains agree on from the start."""

    count = 0
    for left_token, right_token in zip(left, right):
        if int(left_token) != int(right_token):
            break
        count += 1
    return count


def is_frontier_hit(*, lcp_tokens: int, frontier_boundary_tokens: int) -> bool:
    """The binary reuse predicate: the frontier is a prefix of the new prompt.

    A partial LCP is a FULL miss, not a partial hit. The KDA state exists only
    at the frontier boundary, which lies beyond the divergence point of a
    re-rendered prompt, and without state at the LCP the MLA pages below it are
    equally unusable. There is deliberately no "partial" outcome here.
    """

    boundary = int(frontier_boundary_tokens)
    if boundary <= 0:
        raise FrontierRecordInvalid(
            f"frontier_boundary_tokens must be positive, got {boundary}"
        )
    if int(lcp_tokens) < 0:
        raise FrontierRecordInvalid(
            f"lcp_tokens must not be negative, got {lcp_tokens}"
        )
    return int(lcp_tokens) >= boundary


# ---------------------------------------------------------------------------
# Accounting seam
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrontierCharge:
    """What one frontier record costs the Host KV pool.

    ``owned_host_pages`` is the unit the #454 owner charge uses: whole Host
    pages, as ``SequenceEntry.host_owned_pages`` counts them and as the
    boundary growth/eviction planner credits them. The snapshot is charged in
    that same unit, rounded UP to whole pages, so the planner can never
    double-promise the bytes the snapshot is actually holding.

    ``total_bytes`` is the real-bytes view the budget arithmetic uses. It is
    reported alongside, never instead: a byte figure charged into a page
    counter would silently under-credit the pool.

    ``shard_count`` is G, the attention group size the snapshot was taken at.
    ``snapshot_bytes`` is ONE rank's shard, so the node-wide cost of the
    snapshot is ``shard_count`` shards: at G>1 every rank of the owning group
    holds its own head shard and all G of them have to be stored for the
    snapshot to describe the whole state. Each shard is rounded UP to whole
    Host pages INDEPENDENTLY, because a rank writes its shard through its own
    page-table window and a window is a whole number of pages -- summing bytes
    first and rounding once would hand two ranks a shared page.
    """

    page_count: int
    host_page_bytes_all_layers: int
    snapshot_bytes: int
    shard_count: int = 1

    def __post_init__(self) -> None:
        if int(self.shard_count) <= 0:
            raise KDASnapshotProfileInvalid(
                "a frontier charge needs at least one snapshot shard, got "
                f"{self.shard_count}"
            )

    @property
    def page_bytes(self) -> int:
        return int(self.page_count) * int(self.host_page_bytes_all_layers)

    @property
    def snapshot_host_pages(self) -> int:
        """Whole Host pages ONE rank's shard occupies, rounded up."""

        page_bytes = int(self.host_page_bytes_all_layers)
        return -(-int(self.snapshot_bytes) // page_bytes)

    @property
    def arena_host_pages(self) -> int:
        """Whole Host pages the WHOLE snapshot occupies: G x one shard."""

        return int(self.shard_count) * self.snapshot_host_pages

    @property
    def owned_host_pages(self) -> int:
        """Pages charged to the owner: retained prefix pages plus the snapshot."""

        return int(self.page_count) + self.arena_host_pages

    @property
    def snapshot_total_bytes(self) -> int:
        """Snapshot payload across every shard of the owning group."""

        return int(self.shard_count) * int(self.snapshot_bytes)

    @property
    def total_bytes(self) -> int:
        return self.page_bytes + self.snapshot_total_bytes


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrontierReleaseOutcome:
    """Exactly what left the pool when a record's refcount reached zero."""

    session_key: str
    released_page_ids_by_group: Mapping[int, tuple[int, ...]]
    released_snapshot_handle: int
    charge: FrontierCharge


class TrajectoryFrontierRecord:
    """A session's committed frontier: MLA prefix pages PLUS one KDA snapshot.

    Both halves share this refcount. There is no way to build a record with
    pages and no snapshot, or a snapshot and no pages, and no way to release
    one half: ``release`` hands the whole record to the releaser, so an
    MLA-only remnant is unrepresentable rather than merely discouraged.
    """

    __slots__ = (
        "_session_key",
        "_namespace_digest",
        "_frontier_token_ids",
        "_page_ids_by_group",
        "_snapshot",
        "_host_page_bytes_all_layers",
        "_release_record",
        "_refcount",
        "_released",
    )

    def __init__(
        self,
        *,
        session_key: str,
        namespace_digest: Sequence[int],
        frontier_token_ids: Sequence[int],
        page_ids_by_group: Mapping[int, Sequence[int]],
        snapshot: KDASnapshotDescriptor,
        page_tokens: int,
        host_page_bytes_all_layers: int,
        release_record: Callable[["TrajectoryFrontierRecord"], None],
    ) -> None:
        if not session_key:
            raise FrontierRecordInvalid("a frontier record needs a session key")
        if not isinstance(snapshot, KDASnapshotDescriptor):
            raise FrontierRecordInvalid(
                "a frontier record needs its one KDA snapshot descriptor; a "
                "record with pages and no snapshot is not a reusable frontier"
            )
        tokens = tuple(int(token_id) for token_id in frontier_token_ids)
        if not tokens:
            raise FrontierRecordInvalid(
                "a frontier record needs a non-empty committed token prefix"
            )
        boundary = int(page_tokens)
        if boundary <= 0:
            raise FrontierRecordInvalid(
                f"page_tokens must be positive, got {page_tokens}"
            )
        if len(tokens) % boundary:
            raise FrontierRecordInvalid(
                f"frontier boundary {len(tokens)} is not aligned to "
                f"page_tokens={boundary}; align it with aligned_prefix_tokens "
                "before committing a frontier"
            )
        if int(snapshot.boundary_tokens) != len(tokens):
            raise FrontierRecordInvalid(
                "the KDA snapshot was taken at boundary "
                f"{snapshot.boundary_tokens} but the committed frontier is "
                f"{len(tokens)} tokens; a snapshot off the frontier boundary "
                "is not restorable"
            )
        if int(host_page_bytes_all_layers) <= 0:
            raise FrontierRecordInvalid(
                "host_page_bytes_all_layers must be positive, got "
                f"{host_page_bytes_all_layers}"
            )
        if not page_ids_by_group:
            raise FrontierRecordInvalid(
                "a frontier record needs committed pages for every KV group; a "
                "snapshot with no pages is not a reusable frontier"
            )
        pages: dict[int, tuple[int, ...]] = {}
        expected_pages = len(tokens) // boundary
        for group_id, group_pages in sorted(
            (int(key), value) for key, value in page_ids_by_group.items()
        ):
            page_ids = tuple(int(page) for page in group_pages)
            if not page_ids:
                raise FrontierRecordInvalid(
                    f"frontier KV group {group_id} has no committed pages"
                )
            if len(page_ids) != len(set(page_ids)):
                raise FrontierRecordInvalid(
                    f"frontier KV group {group_id} repeats a page id"
                )
            if len(page_ids) != expected_pages:
                raise FrontierRecordInvalid(
                    f"frontier KV group {group_id} has {len(page_ids)} pages "
                    f"for {len(tokens)} tokens, expected {expected_pages}"
                )
            pages[group_id] = page_ids
        self._session_key = str(session_key)
        self._namespace_digest = tuple(int(value) for value in namespace_digest)
        self._frontier_token_ids = tokens
        self._page_ids_by_group = pages
        self._snapshot = snapshot
        self._host_page_bytes_all_layers = int(host_page_bytes_all_layers)
        self._release_record = release_record
        # The creator holds the first reference; the registry owns it.
        self._refcount = 1
        self._released = False

    # -- identity ----------------------------------------------------------

    @property
    def session_key(self) -> str:
        return self._session_key

    @property
    def namespace_digest(self) -> tuple[int, ...]:
        return self._namespace_digest

    @property
    def frontier_token_ids(self) -> tuple[int, ...]:
        return self._frontier_token_ids

    @property
    def frontier_boundary_tokens(self) -> int:
        return len(self._frontier_token_ids)

    @property
    def page_ids_by_group(self) -> Mapping[int, tuple[int, ...]]:
        return dict(self._page_ids_by_group)

    @property
    def page_count(self) -> int:
        return sum(len(pages) for pages in self._page_ids_by_group.values())

    @property
    def snapshot(self) -> KDASnapshotDescriptor:
        return self._snapshot

    @property
    def owning_group_id(self) -> int:
        return int(self._snapshot.owning_group_id)

    @property
    def shard_count(self) -> int:
        """Per-rank head shards this record's snapshot binds (G at commit)."""

        return int(self._snapshot.shard_count)

    @property
    def refcount(self) -> int:
        return self._refcount

    @property
    def is_released(self) -> bool:
        return self._released

    # -- accounting --------------------------------------------------------

    def charge(self) -> FrontierCharge:
        """Pages and snapshot this record charges to its owner.

        The seam Slice 1b charges inside the retention transaction: the
        snapshot is accounted exactly like retained pages, in the #454 unit.
        """

        return FrontierCharge(
            page_count=self.page_count,
            host_page_bytes_all_layers=self._host_page_bytes_all_layers,
            snapshot_bytes=self._snapshot.snapshot_bytes,
            shard_count=int(self._snapshot.shard_count),
        )

    # -- lifetime ----------------------------------------------------------

    def acquire(self) -> "TrajectoryFrontierRecord":
        """Take a reference. A released record can never be revived."""

        if self._released:
            raise FrontierRefcountError(
                f"frontier record for session {self._session_key} is already "
                "released and cannot be referenced again"
            )
        self._refcount += 1
        return self

    def release(self) -> FrontierReleaseOutcome | None:
        """Drop a reference; at zero, pages and snapshot leave together.

        Returns the release outcome only on the drop that reaches zero. A drop
        below zero is refused by name rather than ignored. A failing physical
        release leaves the refcount at one, so the release stays retryable
        instead of leaving the halves stranded.
        """

        if self._released:
            raise FrontierRefcountError(
                f"frontier record for session {self._session_key} was already "
                "released; a second release would double-free its pages"
            )
        if self._refcount <= 0:
            raise FrontierRefcountError(
                f"frontier record for session {self._session_key} has no "
                f"reference to drop (refcount={self._refcount})"
            )
        if self._refcount > 1:
            self._refcount -= 1
            return None
        charge = self.charge()
        # The releaser receives the WHOLE record, so there is no call shape
        # that frees the pages without the snapshot or the other way round.
        # Only a completed release retires the record.
        self._release_record(self)
        self._refcount = 0
        self._released = True
        return FrontierReleaseOutcome(
            session_key=self._session_key,
            released_page_ids_by_group=dict(self._page_ids_by_group),
            released_snapshot_handle=int(self._snapshot.storage_handle),
            charge=charge,
        )


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrontierResolution:
    """What the committed frontiers say about a candidate prompt.

    A miss carries no session key: the trajectory's baseline is fixed at commit
    time from the tokens actually committed, so admission must not invent one.
    """

    hit: bool
    record: TrajectoryFrontierRecord | None
    lcp_tokens: int

    @property
    def session_key(self) -> str | None:
        return None if self.record is None else self.record.session_key

    @property
    def frontier_boundary_tokens(self) -> int:
        return 0 if self.record is None else self.record.frontier_boundary_tokens


@dataclass(frozen=True)
class FrontierAdvance:
    """The outcome of installing a new frontier over a session's old one."""

    record: TrajectoryFrontierRecord
    previous_release: FrontierReleaseOutcome | None


class TrajectoryFrontierRegistry:
    """The Host coordinator's per-session frontier table.

    Exactly one record per session key, indexed by its frontier tokens so a
    continuation prompt finds it without a new batch field. Advancing a session
    installs the new record BEFORE releasing the old one, so a failure anywhere
    in between leaves the old frontier intact and the table consistent.
    """

    def __init__(
        self,
        *,
        page_tokens: int,
        host_page_bytes_all_layers: int,
        release_record: Callable[[TrajectoryFrontierRecord], None],
    ) -> None:
        if int(page_tokens) <= 0:
            raise FrontierRegistryError(
                f"page_tokens must be positive, got {page_tokens}"
            )
        if int(host_page_bytes_all_layers) <= 0:
            raise FrontierRegistryError(
                "host_page_bytes_all_layers must be positive, got "
                f"{host_page_bytes_all_layers}"
            )
        self._page_tokens = int(page_tokens)
        self._host_page_bytes_all_layers = int(host_page_bytes_all_layers)
        self._release_record = release_record
        self._records: dict[str, TrajectoryFrontierRecord] = {}
        self._by_frontier: dict[str, str] = {}

    # -- read ---------------------------------------------------------------

    @property
    def page_tokens(self) -> int:
        return self._page_tokens

    @property
    def session_keys(self) -> tuple[str, ...]:
        return tuple(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def get(self, session_key: str) -> TrajectoryFrontierRecord | None:
        return self._records.get(str(session_key))

    def require(self, session_key: str) -> TrajectoryFrontierRecord:
        record = self._records.get(str(session_key))
        if record is None:
            raise FrontierRegistryError(
                f"no committed frontier for session {session_key}"
            )
        return record

    def total_charge(self) -> FrontierCharge:
        """Aggregate charge of every resident frontier record.

        Every resident record must share one ``shard_count``: G is fixed for a
        rank's whole life, so a mix means a record survived a reconfiguration
        and its arena no longer matches the group that would restore it. That
        is reported as a broken invariant rather than averaged away -- the
        aggregate page figure would otherwise be silently wrong.
        """

        shard_counts = {
            int(record.shard_count) for record in self._records.values()
        }
        if len(shard_counts) > 1:
            raise FrontierRegistryInvariantError(
                "resident frontier records disagree on their snapshot shard "
                f"count ({sorted(shard_counts)}); one of them was committed "
                "under a different attention group size"
            )
        return FrontierCharge(
            page_count=sum(
                record.page_count for record in self._records.values()
            ),
            host_page_bytes_all_layers=self._host_page_bytes_all_layers,
            snapshot_bytes=sum(
                record.snapshot.snapshot_bytes
                for record in self._records.values()
            ),
            shard_count=shard_counts.pop() if shard_counts else 1,
        )

    def session_group_summary(self) -> dict[str, tuple[int, int, str]]:
        """Compact ``{session -> (owning group, boundary, index key)}`` view.

        What ONE serve group's registry publishes so the rest of the world can
        pin a continuation onto it. The index key is derived here with
        ``frontier_index_key`` over the record's own committed tokens, which is
        the SAME derivation ``resolve`` probes with, so a prompt that would hit
        this record maps to this entry and nothing re-implements the hashing.

        Tokens are deliberately NOT exported: the key is a digest of them, so
        the summary stays O(1) per session.
        """

        summary: dict[str, tuple[int, int, str]] = {}
        for session_key, record in self._records.items():
            summary[str(session_key)] = (
                int(record.owning_group_id),
                int(record.frontier_boundary_tokens),
                frontier_index_key(
                    namespace_digest=record.namespace_digest,
                    frontier_token_ids=record.frontier_token_ids,
                ),
            )
        return summary

    def resolve(
        self,
        *,
        namespace_digest: Sequence[int],
        prompt_token_ids: Sequence[int],
    ) -> FrontierResolution:
        """Find the committed frontier this prompt continues, if any.

        Candidates come from the frontier index -- only lengths that are
        actually committed are probed -- and the binary predicate then decides
        on the tokens themselves, so a digest collision cannot manufacture a
        hit. The longest matching frontier wins: it is the most specific
        continuation and its state is furthest along the trajectory.
        """

        prompt = tuple(int(token_id) for token_id in prompt_token_ids)
        if not prompt:
            raise FrontierRegistryError(
                "frontier resolution requires a non-empty prompt"
            )
        best_lcp = 0
        for boundary in self._candidate_boundaries(len(prompt)):
            index_key = frontier_index_key(
                namespace_digest=namespace_digest,
                frontier_token_ids=prompt[:boundary],
            )
            session_key = self._by_frontier.get(index_key)
            if session_key is None:
                continue
            record = self._records[session_key]
            lcp = longest_common_prefix_length(prompt, record.frontier_token_ids)
            best_lcp = max(best_lcp, lcp)
            if is_frontier_hit(
                lcp_tokens=lcp,
                frontier_boundary_tokens=record.frontier_boundary_tokens,
            ):
                return FrontierResolution(hit=True, record=record, lcp_tokens=lcp)
        return FrontierResolution(hit=False, record=None, lcp_tokens=best_lcp)

    def _candidate_boundaries(self, prompt_length: int) -> Iterable[int]:
        boundaries = {
            record.frontier_boundary_tokens for record in self._records.values()
        }
        return sorted(
            (
                boundary
                for boundary in boundaries
                if 0 < boundary <= int(prompt_length)
            ),
            reverse=True,
        )

    # -- write --------------------------------------------------------------

    def begin(
        self,
        *,
        namespace_digest: Sequence[int],
        frontier_token_ids: Sequence[int],
        page_ids_by_group: Mapping[int, Sequence[int]],
        snapshot: KDASnapshotDescriptor,
    ) -> TrajectoryFrontierRecord:
        """Register the first frontier of a new trajectory.

        The session key is derived from this baseline and stays fixed for the
        trajectory's whole life.
        """

        session_key = session_key_for_baseline(
            namespace_digest=namespace_digest,
            baseline_token_ids=frontier_token_ids,
        )
        if session_key in self._records:
            raise FrontierRegistryError(
                f"session {session_key} already has a committed frontier; "
                "advance it instead of beginning it again"
            )
        record = self._build_record(
            session_key=session_key,
            namespace_digest=namespace_digest,
            frontier_token_ids=frontier_token_ids,
            page_ids_by_group=page_ids_by_group,
            snapshot=snapshot,
        )
        self._install(record)
        return record

    def advance(
        self,
        *,
        session_key: str,
        namespace_digest: Sequence[int],
        frontier_token_ids: Sequence[int],
        page_ids_by_group: Mapping[int, Sequence[int]],
        snapshot: KDASnapshotDescriptor,
    ) -> FrontierAdvance:
        """Replace a session's frontier record, new one referenced first.

        Ordering is the crash-safety contract. The new record is built and
        installed -- so a lookup already finds it -- and only then is the old
        record's reference dropped. Any failure while building or installing
        leaves the old record installed, referenced and unreleased.
        """

        previous = self.require(session_key)
        if previous.owning_group_id != int(snapshot.owning_group_id):
            raise FrontierRegistryError(
                f"session {session_key} is pinned to KV group "
                f"{previous.owning_group_id}; its snapshot cannot move to group "
                f"{snapshot.owning_group_id}"
            )
        if previous.shard_count != int(snapshot.shard_count):
            raise FrontierRegistryError(
                f"session {session_key} holds a {previous.shard_count}-shard "
                "snapshot; advancing it to a "
                f"{int(snapshot.shard_count)}-shard one would change the head "
                "split the record describes"
            )
        if int(len(frontier_token_ids)) < previous.frontier_boundary_tokens:
            raise FrontierRegistryError(
                f"session {session_key} cannot advance backwards: new frontier "
                f"{len(frontier_token_ids)} tokens, committed "
                f"{previous.frontier_boundary_tokens}"
            )
        # Step 1: build and reference the NEW record. Raises here leave the old
        # record installed and untouched.
        record = self._build_record(
            session_key=previous.session_key,
            namespace_digest=namespace_digest,
            frontier_token_ids=frontier_token_ids,
            page_ids_by_group=page_ids_by_group,
            snapshot=snapshot,
        )
        self._install(record, replacing=previous)
        # Step 2: the new frontier is live and reachable; drop the old
        # reference. A failing physical release is reported with the record so
        # it can be retried -- the table is already consistent.
        try:
            released = previous.release()
        except Exception as exc:
            raise FrontierReleaseFailed(
                f"session {session_key} advanced to a {record.frontier_boundary_tokens}"
                "-token frontier but releasing the previous record failed: "
                f"{exc}",
                record=previous,
            ) from exc
        return FrontierAdvance(record=record, previous_release=released)

    def evict(self, session_key: str) -> FrontierReleaseOutcome:
        """Drop the registry's reference to a session's frontier.

        The pages and the snapshot leave together or not at all. A record still
        referenced by a live attachment is refused rather than half-evicted.
        """

        record = self.require(session_key)
        if record.refcount != 1:
            raise FrontierRefcountError(
                f"session {session_key} has {record.refcount} references; its "
                "frontier cannot be evicted while it is attached"
            )
        self._uninstall(record)
        try:
            released = record.release()
        except Exception as exc:
            raise FrontierReleaseFailed(
                f"releasing the frontier of session {session_key} failed: {exc}",
                record=record,
            ) from exc
        if released is None:
            raise FrontierRefcountError(
                f"evicting session {session_key} did not reach refcount zero"
            )
        return released

    # -- invariants ---------------------------------------------------------

    def check_invariants(self) -> None:
        """Assert the always-true relations of the table.

        Cheap enough to call from tests and from the commit transaction.
        """

        if len(self._records) != len(self._by_frontier):
            raise FrontierRegistryInvariantError(
                f"{len(self._records)} frontier records against "
                f"{len(self._by_frontier)} index entries"
            )
        for session_key, record in self._records.items():
            if record.session_key != session_key:
                raise FrontierRegistryInvariantError(
                    f"record filed under {session_key} reports session "
                    f"{record.session_key}"
                )
            if record.is_released:
                raise FrontierRegistryInvariantError(
                    f"session {session_key} holds a released frontier record"
                )
            if record.refcount < 1:
                raise FrontierRegistryInvariantError(
                    f"session {session_key} holds a frontier record with "
                    f"refcount {record.refcount}"
                )
            if not record.page_ids_by_group:
                raise FrontierRegistryInvariantError(
                    f"session {session_key} holds a frontier with no pages"
                )
            index_key = frontier_index_key(
                namespace_digest=record.namespace_digest,
                frontier_token_ids=record.frontier_token_ids,
            )
            if self._by_frontier.get(index_key) != session_key:
                raise FrontierRegistryInvariantError(
                    f"session {session_key} is not reachable through its "
                    "frontier index key"
                )

    # -- internals ----------------------------------------------------------

    def _build_record(
        self,
        *,
        session_key: str,
        namespace_digest: Sequence[int],
        frontier_token_ids: Sequence[int],
        page_ids_by_group: Mapping[int, Sequence[int]],
        snapshot: KDASnapshotDescriptor,
    ) -> TrajectoryFrontierRecord:
        return TrajectoryFrontierRecord(
            session_key=session_key,
            namespace_digest=namespace_digest,
            frontier_token_ids=frontier_token_ids,
            page_ids_by_group=page_ids_by_group,
            snapshot=snapshot,
            page_tokens=self._page_tokens,
            host_page_bytes_all_layers=self._host_page_bytes_all_layers,
            release_record=self._release_record,
        )

    def _install(
        self,
        record: TrajectoryFrontierRecord,
        *,
        replacing: TrajectoryFrontierRecord | None = None,
    ) -> None:
        index_key = frontier_index_key(
            namespace_digest=record.namespace_digest,
            frontier_token_ids=record.frontier_token_ids,
        )
        owner = self._by_frontier.get(index_key)
        if owner is not None and owner != record.session_key:
            raise FrontierRegistryError(
                f"frontier of session {record.session_key} is already committed "
                f"by session {owner}"
            )
        if replacing is not None:
            old_index_key = frontier_index_key(
                namespace_digest=replacing.namespace_digest,
                frontier_token_ids=replacing.frontier_token_ids,
            )
            if old_index_key != index_key:
                self._by_frontier.pop(old_index_key, None)
        self._by_frontier[index_key] = record.session_key
        self._records[record.session_key] = record

    def _uninstall(self, record: TrajectoryFrontierRecord) -> None:
        index_key = frontier_index_key(
            namespace_digest=record.namespace_digest,
            frontier_token_ids=record.frontier_token_ids,
        )
        self._by_frontier.pop(index_key, None)
        self._records.pop(record.session_key, None)


# ---------------------------------------------------------------------------
# Session -> owning serve group map (the admission pin)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionGroupEntry:
    """One session's owning serve group, as the whole world knows it."""

    session_key: str
    owning_group_id: int
    boundary_tokens: int
    frontier_index_key: str

    def __post_init__(self) -> None:
        if not str(self.session_key):
            raise SessionGroupMapError(
                "a session-group entry needs a session key"
            )
        if not str(self.frontier_index_key):
            raise SessionGroupMapError(
                f"session {self.session_key} published no frontier index key, "
                "so no continuation prompt could ever find it"
            )
        if int(self.owning_group_id) < 0:
            raise SessionGroupMapError(
                f"session {self.session_key} names a negative owning group "
                f"{self.owning_group_id}"
            )
        if int(self.boundary_tokens) <= 0:
            raise SessionGroupMapError(
                f"session {self.session_key} names a non-positive frontier "
                f"boundary {self.boundary_tokens}"
            )


class TrajectorySessionGroupIndex:
    """Rank-local copy of the WORLD's session -> owning-group map.

    A frontier registry is group-local: the leader of a serve group holds the
    ONE registry for that group, so a continuation that lands on a different
    group is a clean MISS no matter how exactly its tokens match. Default group
    assignment knows nothing about sessions, so consecutive batches place a
    session's continuations wherever the balancer likes -- in the dual-node
    run that was the opposite group every time. This map is the admission-side
    input that fixes it: the owning group of every committed session, known to
    EVERY rank.

    Why every rank: group assignment runs identically on all ranks and its
    output must agree, so the map it consults has to be world-identical. Every
    rank folds the SAME gathered leader summaries, and both the replacement
    rule and the resolution order below are total and order-independent, so
    two ranks that folded the same summaries resolve the same prompt the same
    way.

    TIMING -- what this map can and cannot see. Leaders publish their summary
    on the host-KV mirror, which runs at the END of a pass's prefill config,
    while the pass's own frontier records are installed later, at the
    post-barrier finalize. So a session committed by pass N first appears in
    the summary published by pass N+1's config, and only an admission after
    that sees it. A continuation admitted in the SAME pass could not hit
    anyway (its record does not exist until finalize), so nothing is built
    for that case.

    This map decides SCHEDULING only. A resolution is a digest match, not the
    token-level predicate: the lookup on the pinned group still decides hit or
    miss on the tokens themselves, so a digest collision costs a placement,
    never a wrong reuse.
    """

    __slots__ = ("_by_session", "_index", "_boundaries", "_pins")

    def __init__(self) -> None:
        self._by_session: dict[str, SessionGroupEntry] = {}
        self._index: dict[str, SessionGroupEntry] | None = None
        self._boundaries: tuple[int, ...] | None = None
        self._pins = 0

    # -- read ---------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._by_session)

    @property
    def pins(self) -> int:
        """Sequences this rank stamped onto a session's owning group."""

        return self._pins

    @property
    def entries(self) -> tuple[SessionGroupEntry, ...]:
        """Every known session, in session-key order (rank-independent)."""

        return tuple(
            self._by_session[key] for key in sorted(self._by_session)
        )

    def get(self, session_key: str) -> SessionGroupEntry | None:
        return self._by_session.get(str(session_key))

    def record_pin(self) -> None:
        """Count one sequence placed by this map, for the drain row."""

        self._pins += 1

    # -- write --------------------------------------------------------------

    def fold(self, summaries: Iterable[Mapping | None]) -> int:
        """Fold every leader's summary; returns how many entries moved.

        ``summaries`` is the gathered list in rank order, with a ``None`` or
        empty slot for every rank that published nothing.
        """

        changed = 0
        for summary in summaries:
            if not summary:
                continue
            try:
                group_id = summary["group_id"]
                sessions = summary["sessions"]
            except (KeyError, TypeError) as exc:
                raise SessionGroupMapError(
                    "a session-group summary needs 'group_id' and 'sessions': "
                    f"{exc}"
                ) from exc
            changed += self.fold_group_summary(
                group_id=group_id, sessions=sessions or {}
            )
        return changed

    def fold_group_summary(
        self,
        *,
        group_id: int,
        sessions: Mapping[str, Sequence],
    ) -> int:
        """Install ONE serve group's published sessions.

        The group's leader holds that group's only registry, so its summary is
        the WHOLE truth about it: a session this group no longer reports has
        been evicted and leaves the map here. An entry claiming a different
        owning group than the publisher is refused by name -- a leader cannot
        speak for another group's registry.

        A session key can legitimately exist on two groups: the key is derived
        from the trajectory's baseline tokens, so after an eviction the same
        baseline can be committed again on another group. The conflict is
        settled by (newest boundary, then lowest group id), which is a total
        order, so the outcome does not depend on fold order.
        """

        group = int(group_id)
        if group < 0:
            raise SessionGroupMapError(
                f"a session-group summary names a negative serve group {group}"
            )
        incoming: dict[str, SessionGroupEntry] = {}
        for session_key, fields in sessions.items():
            try:
                owning_group_id, boundary_tokens, index_key = fields
            except (TypeError, ValueError) as exc:
                raise SessionGroupMapError(
                    f"session {session_key} of group {group} must publish "
                    "(owning_group_id, boundary_tokens, frontier_index_key), "
                    f"got {fields!r}"
                ) from exc
            if int(owning_group_id) != group:
                raise SessionGroupMapError(
                    f"serve group {group} published session {session_key} as "
                    f"owned by group {int(owning_group_id)}; a leader holds "
                    "the registry of its own group only"
                )
            incoming[str(session_key)] = SessionGroupEntry(
                session_key=str(session_key),
                owning_group_id=group,
                boundary_tokens=int(boundary_tokens),
                frontier_index_key=str(index_key),
            )
        changed = 0
        for session_key, entry in list(self._by_session.items()):
            if entry.owning_group_id != group or session_key in incoming:
                continue
            del self._by_session[session_key]
            changed += 1
        for session_key, entry in incoming.items():
            existing = self._by_session.get(session_key)
            if existing is not None and existing.owning_group_id != group:
                keep_existing = (
                    existing.boundary_tokens > entry.boundary_tokens
                    or (
                        existing.boundary_tokens == entry.boundary_tokens
                        and existing.owning_group_id < group
                    )
                )
                if keep_existing:
                    continue
            if existing == entry:
                continue
            self._by_session[session_key] = entry
            changed += 1
        if changed:
            self._index = None
            self._boundaries = None
        return changed

    # -- the pin ------------------------------------------------------------

    def resolve(
        self,
        *,
        namespace_digest: Sequence[int],
        prompt_token_ids: Sequence[int],
    ) -> SessionGroupEntry | None:
        """The owning group of the session this prompt continues, if known.

        Probes the committed boundaries longest-first with
        ``frontier_index_key`` over the prompt's own prefix -- the SAME
        derivation the registry index is built from -- so a prompt that the
        owning group's registry would resolve is placed on that group. ``None``
        means no committed session is known for this prompt, and the caller
        leaves default assignment alone.
        """

        prompt = tuple(int(token_id) for token_id in prompt_token_ids)
        if not self._by_session or not prompt:
            return None
        index = self._frontier_index()
        for boundary in self._candidate_boundaries(len(prompt)):
            entry = index.get(
                frontier_index_key(
                    namespace_digest=namespace_digest,
                    frontier_token_ids=prompt[:boundary],
                )
            )
            if entry is not None:
                return entry
        return None

    # -- internals ----------------------------------------------------------

    def _frontier_index(self) -> dict[str, SessionGroupEntry]:
        """Index key -> entry, resolved by a total order.

        Two different sessions can share an index key: the same prompt served
        on two groups commits one session per group, and both are equally
        valid pins. The winner is (longest boundary, lowest group id, lowest
        session key), so every rank picks the same one regardless of the order
        it folded the summaries in.
        """

        if self._index is None:
            index: dict[str, SessionGroupEntry] = {}
            for entry in sorted(
                self._by_session.values(),
                key=lambda item: (
                    -item.boundary_tokens,
                    item.owning_group_id,
                    item.session_key,
                ),
            ):
                index.setdefault(entry.frontier_index_key, entry)
            self._index = index
        return self._index

    def _candidate_boundaries(self, prompt_length: int) -> tuple[int, ...]:
        if self._boundaries is None:
            self._boundaries = tuple(
                sorted(
                    {
                        entry.boundary_tokens
                        for entry in self._by_session.values()
                    },
                    reverse=True,
                )
            )
        return tuple(
            boundary
            for boundary in self._boundaries
            if 0 < boundary <= int(prompt_length)
        )
