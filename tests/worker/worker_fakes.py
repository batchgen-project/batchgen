"""Shared helper for attaching real worker identity state to a test fake.

Many worker tests build a ``SimpleNamespace`` (or a small hand-written class)
carrying only the fields the method under test reads, then bind the method's
source onto it. Rank identity and the local slot binding now live in
``RankContext`` and ``SequenceTable``, so a fake needs real instances of
those two rather than loose attributes.

``attach_identity`` is the single way tests install them. It is deliberately
explicit: no ``__getattr__`` magic, no defaults that hide a missing rank.
Later worker-decouple slices extend this module as more state moves into
owned components.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Union

from batchgen.worker.graph_runtime import GraphRuntime
from batchgen.worker.rank_context import RankContext
from batchgen.worker.sequence_table import SequenceTable

Bindings = Union[Mapping[str, int], Iterable[str], None]


def make_sequence_table(bound: Bindings = None) -> SequenceTable:
    """A ``SequenceTable`` pre-bound with ``bound``.

    ``bound`` is either a ``{uuid: local_idx}`` mapping, an iterable of uuids
    (bound to 0, 1, 2, ... in order), or ``None`` for an empty table.
    """
    table = SequenceTable()
    if bound is None:
        return table
    if isinstance(bound, Mapping):
        for uuid, local_idx in bound.items():
            table.bind(uuid, local_idx)
    else:
        for local_idx, uuid in enumerate(bound):
            table.bind(uuid, local_idx)
    return table


def make_rank_context(
    *,
    rank: int,
    world_size: int,
    local_world_size: int,
    local_rank: Optional[int] = None,
    decode_group_size: int = 1,
) -> RankContext:
    """A ``RankContext`` with ``local_rank`` defaulting to ``rank %
    local_world_size`` (what a real launch computes)."""
    if local_rank is None:
        local_rank = rank % local_world_size
    return RankContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        local_world_size=local_world_size,
        decode_group_size=decode_group_size,
    )


def attach_identity(
    fake,
    *,
    rank: int,
    world_size: int,
    local_world_size: int,
    local_rank: Optional[int] = None,
    decode_group_size: int = 1,
    bound: Bindings = None,
    table: Optional[SequenceTable] = None,
):
    """Set ``rank_ctx`` and ``seq_table`` on ``fake``, and return ``fake``.

    Also sets ``rank``, ``world_size``, ``local_rank`` and
    ``local_world_size`` on the fake, because worker code reads both the
    context and those plain fields. Pass ``table`` to share one table between
    several fakes (e.g. ranks of one serve group that must observe the same
    binding); otherwise one is built from ``bound``.
    """
    rank_ctx = make_rank_context(
        rank=rank,
        world_size=world_size,
        local_world_size=local_world_size,
        local_rank=local_rank,
        decode_group_size=decode_group_size,
    )
    if table is None:
        table = make_sequence_table(bound)
    elif bound is not None:
        raise TypeError("pass either `bound` or `table`, not both")

    fake.rank_ctx = rank_ctx
    fake.seq_table = table
    fake.rank = rank_ctx.rank
    fake.world_size = rank_ctx.world_size
    fake.local_rank = rank_ctx.local_rank
    fake.local_world_size = rank_ctx.local_world_size
    # A real BatchGenWorker exposes these as read-only properties over
    # `seq_table`, so on a real instance there is nothing to attach (and
    # assigning them would raise). A duck-typed fake is not an instance of
    # it, so the same live views are attached here: worker code that reads
    # them keeps working, and a stray write raises instead of silently
    # desyncing the table.
    for name, view in (
        ("_local_to_uuid_map", table.local_to_uuid),
        ("_uuid_to_local_map", table.uuid_to_local),
        ("_free_local_indices", table.free_local_indices),
    ):
        if not isinstance(getattr(type(fake), name, None), property):
            setattr(fake, name, view)
    return fake


def make_graph_state(**overrides) -> GraphRuntime:
    """A real ``GraphRuntime`` with ``overrides`` applied to its fields.

    Field names are the component's own (the worker attribute minus its
    leading underscore, e.g. ``cuda_graph_manager``). An unknown name is a
    typo in the test, not a new field, so it raises instead of planting a
    silent attribute the worker will never read.
    """
    graph = GraphRuntime()
    for name, value in overrides.items():
        if not hasattr(graph, name):
            raise AttributeError(f"GraphRuntime has no field {name!r}")
        setattr(graph, name, value)
    return graph


def attach_graph_state(fake, **overrides):
    """Set ``_graph`` on ``fake`` to a real ``GraphRuntime``, return ``fake``.

    The decode CUDA-graph state used to be ~29 loose worker attributes that
    every fake re-declared one by one; worker code now reads it through
    ``self._graph``. Installing the real component (not a namespace) is what
    gives a fake the same "absent means falsy" defaults the old getattr sites
    relied on.
    """
    fake._graph = make_graph_state(**overrides)
    return fake
