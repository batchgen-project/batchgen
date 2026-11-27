"""Prefill wave planning and lifecycle behind a worker seam.

The engine is intentionally stateless.  An operation that has not been
converted yet receives the live worker as its collaborator, then rebinds
``self`` to that worker so the moved statements keep their original state
access and call spellings.

D7.1 converted the small pure readers and predicates: each takes a frozen
plain-value context and explicit keyword-only collaborator callables, and none
of them rebinds ``self``.  Their contexts are deliberately narrow, one per read
set, because several of these reads were ``getattr`` defaults on the worker and
a wider context would turn "absent" into an ``AttributeError`` at the
delegator.  Where two readers' read sets genuinely coincide they share one
context and the caller reaches its sibling with it directly:
:class:`PrefillPoolParallelContext` carries the one PSM handle the pool's
placement and geometry readers touch, and :class:`PoolProbeContext` carries the
rank and the adopted debug state the pooled probes read.  No converted
operation writes a worker attribute, so this stage declares no outcome; the
probe read's two writes land on the wave object its caller passed, in place.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Protocol, Tuple

import torch
import torch.distributed as dist
from tqdm import tqdm

from batchgen.prefill.batch_order import (
    build_prefill_sequence_spans,
    prefill_sequence_spans_to_cu_seqlens,
    prefill_sequence_spans_to_global_seq_ids,
)
from batchgen.prefill.prepack import (
    build_prefill_micro_batches,
    get_prepack_stats,
    prepack_sequences,
)
from batchgen.lifespan import SeqEvent
from batchgen.native_loader import core_engine
from batchgen.phase_lifecycle import AdmitContext
from batchgen.query_book import (
    QueryBookPoolCapacityError,
    refresh_query_book_entry,
)
from batchgen.sequence import SequenceStatus
from batchgen.worker.decode_engine import BATCHGEN_CB_DEBUG
from batchgen.worker.prefill import (
    PREFILL_CAP_OVERSIZED_REQUESTS_REJECTED,
    PREFILL_CAP_SCHEDULING_POOL_EMPTY,
    PrefillAdmissionStalled,
    PrefillAdmissionStateDivergence,
    PrefillScheduler,
    PrefillWaveCapRecord,
    prefill_reclaim_target_pages,
    record_prefill_binding_cap,
)


def _reentry_history_length(
    prompt_length: int, original_prompt_length: int, original_max_decode_length: int,
) -> int:
    """Tokens a re-entering sequence already decoded, capped by its own budget.

    The reconstructed prompt carries the whole history, so the history length is
    prompt_length - original_prompt_length. Cap it by the request's absolute
    decode budget, which also bounds its decoded_tokens row (admission grows the
    pool to every request's max_decode_length). The worker-wide
    max_decoding_length must not be used: in pool mode it is the budget of the
    first batch the server saw and can be smaller than a later request's history.
    """
    return max(0, min(prompt_length - original_prompt_length, original_max_decode_length))


# Server debug-state keys of the pooled-prefill DIAGNOSTIC probes.
# Spelled out here, instead of imported from the four probe modules, so that
# deciding "is any probe armed on this prefill" is a membership test on a dict
# the engine already holds: a production prefill must not import a probe module
# or build a merged debug dict to be told that nothing is armed.
# ``tests/unit/test_pool_probe_debug_seam.py`` pins these against the modules'
# own constants, so the duplication cannot drift.
_EPILOGUE_TRACE_DEBUG_FLAG = "prefix_pool_epilogue_trace"
_EPILOGUE_TRACE_PROBE_TOKEN_FLAG = "prefix_pool_epilogue_probe_token"
_POOL_OFFLOAD_CHECK_DEBUG_FLAG = "prefix_pool_offload_check"
_POOL_KERNEL_DUMP_DEBUG_FLAG = "prefix_pool_dump_kernel_args"
_POOL_KERNEL_DUMP_DIR_FLAG = "prefix_pool_dump_dir"
_POOL_PAD_POISON_DEBUG_FLAG = "prefix_pool_pad_poison"
_POOL_PROBE_DEBUG_KEYS = (
    _EPILOGUE_TRACE_DEBUG_FLAG,
    _EPILOGUE_TRACE_PROBE_TOKEN_FLAG,
    _POOL_OFFLOAD_CHECK_DEBUG_FLAG,
    _POOL_KERNEL_DUMP_DEBUG_FLAG,
    _POOL_KERNEL_DUMP_DIR_FLAG,
    _POOL_PAD_POISON_DEBUG_FLAG,
)


# D7.1: the frozen value contexts of this engine's pure readers and predicates.
# One context per read set, named after the worker attributes it snapshots, and
# collaborators are never fields -- they arrive as keyword-only callables.


@dataclass(frozen=True)
class PrefillPoolParallelContext:
    """The ONE PSM handle the pool's placement and geometry readers touch.

    Shared by four readers because their read set is exactly this object: the
    group-safety declaration, the pool's process group, the PSM's pool layout
    override and the page dims that resolve through that override. The page
    dims therefore call the layout reader with THIS context instead of going
    back out through a worker delegator that would add nothing.
    """

    parallel_manager: object


@dataclass(frozen=True)
class PrefillPrefixPoolEnabledContext:
    """What deciding "is the in-wave prefix pool on" reads.

    Deliberately narrower than a prefill phase context: ``_prefix_integrity``
    is a diagnostic latch the pre-move body read with a ``getattr`` default, so
    the delegator keeps that default and a worker that never set it still
    answers False instead of raising.
    """

    parallel_manager: object
    engine_config: object
    enable_prefix_cache: bool
    prefix_runtime: object
    # ``False`` is the pre-move ``getattr`` default: the integrity check is off
    # unless a launch armed it.
    _prefix_integrity: bool


@dataclass(frozen=True)
class PrefillPoolPageBytesContext:
    """The page-byte arithmetic's two deployment reads.

    The layout and the page dims it also needs are sibling operations with
    their own contexts, so they arrive as collaborator callables -- the same
    worker delegators the pre-move body called.
    """

    prefix_runtime: object
    huggingface_ckpt_name: str


@dataclass(frozen=True)
class PrefillPoolStateSlotsContext:
    """The trajectory store the free recurrent-state slot count reads."""

    traj: object


@dataclass(frozen=True)
class PoolSegmentSidContext:
    """The prefix-reuse state that owns the pooled-segment sequence ids."""

    prefix: object


@dataclass(frozen=True)
class PrefillPoolWaveContext:
    """The planned wave, or ``None`` when no wave is in flight.

    ``None`` is the pre-move read's own "nothing planned" case, which the
    segment host-page count answers with zero.
    """

    _prefill_pool_wave: object


@dataclass(frozen=True)
class PrefillModelPassCountContext:
    """What counting this prefill's model forwards reads.

    The micro-batch planner is a sibling with a different read set (and its own
    delegator-resolved cap), so it arrives as a collaborator callable.
    """

    _prefill_pool_wave: object
    enable_prepack: bool
    engine_config: object
    pool: object
    local_to_uuid: object
    prefix: object


@dataclass(frozen=True)
class PoolProbeContext:
    """The rank and the debug state the pooled probes were armed through.

    Shared by the probe read and the pad-poison predicate, whose read sets
    coincide on exactly these two, so the predicate calls the read with THIS
    context. The three probe recorders read the same two.
    """

    rank: int
    _phase_prefill_debug: object


class PrefillModelStateBinder(Protocol):
    """Model-state seam the prefill forwards drive (S5b; the worker implements it).

    Every model-wrapper class-state statement of the moved prefill bodies lives
    behind one of these private worker methods, one method per contiguous run
    of the original body, so this module never names a model wrapper.  The
    moved bodies reach them through the rebound ``self`` (= the worker).
    """

    def _bind_prefill_host_views(self) -> None: ...
    def _bind_prefill_pool_segments(self, wave) -> None: ...
    def _bind_prefill_pool_layer_state(self, wave, pool_layout) -> None: ...
    def _install_prefill_pool_rows(self, rows, host_ids, host_starts) -> None: ...
    def _drop_prefill_pool_offload_recorder(self) -> None: ...
    def _unbind_prefill_pool_segments(self) -> None: ...
    def _bind_prefill_step_metadata(self, attention_mask) -> None: ...
    def _bind_prefill_step_batch(self, cur_batch_local) -> None: ...

    def _install_prefill_prepack_metadata(
        self,
        batch_cu_seqlens,
        batch_max_seqlen,
        batch_num_seqs,
        batch_seq_lengths,
        batch_position_ids_flat,
        batch_spans,
        prefix_materialization,
        batch_full_seq_lengths,
        batch_compute_cached,
        batch_attached_tokens,
    ) -> None: ...

    def _retire_prefill_pending_offloads(self, device, reason): ...
    def _clear_prefill_prefix_materialization(self) -> None: ...
    def _reset_prefill_prepack_metadata(self) -> None: ...
    def _verify_prefill_pool_offload_bytes(self) -> None: ...
    def _materialized_pool_layers(self): ...


class PrefillEngine:
    """Stateless prefill engine; the worker owns all mutable state.

    S5a moved the wave planning/epilogue bodies here; S5b moved the prefill
    forwards (``prefill``, ``prefill_prepacked``, ``_prefill_prepacked_pool``,
    ``_write_prefill_tokens``), whose model state goes through
    :class:`PrefillModelStateBinder`.
    """

    def _plan_prefill_micro_batches(
        self, worker,
        seq_lengths: List[int],
        full_seq_lengths: List[int],
        compute_cached_tokens: List[int],
        *,
        token_cap: int,
        attn_group_size: int,
    ) -> Tuple[List[Tuple[int, int]], int, bool]:
        """Build the one authoritative prepacked-forward plan for local rows.

        ``seq_lengths`` are the computed (suffix) lengths the forward packs;
        ``full_seq_lengths`` the prompt lengths. A micro-batch holding a
        prefix hit materializes temporary GPU KV for every member's full
        length, so its pages are capped by the prefill-phase prefix KV budget
        (``_measure_prefill_prefix_kv_budget``) and the micro-batch splits
        until the transient fits.
        """
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        # token_cap: the server mode's one prefill cap, resolved by the
        # worker's delegator and handed over explicitly.
        use_l2 = os.environ.get("BATCHGEN_L2_BALANCE", "1") == "1"
        single_sequence_only = (
            token_cap > 0
            and max(seq_lengths) > token_cap
            and hasattr(self.parallel_manager, "prefill_uses_streamed_sp8")
            and self.parallel_manager.prefill_uses_streamed_sp8()
        )
        prefix_hits = [int(tokens) > 0 for tokens in compute_cached_tokens]
        kv_pages = [0] * len(prefix_hits)
        kv_page_cap = 0
        if any(prefix_hits):
            from batchgen.prefix_reuse.materialization import (
                gpu_kv_page_bytes,
                temporary_kv_pages,
            )

            budget = self._prefill_prefix_kv_budget_bytes
            if budget is None:
                raise RuntimeError(
                    "prefix-hit prefill has no prefill-phase prefix KV budget: "
                    "the model planner sets no prefill_prefix_pool_workspace_bytes"
                    ", so the temporary full-length KV cannot be bounded"
                )
            if attn_group_size != 1:
                raise RuntimeError(
                    "prefix-hit prefill splits micro-batches by a rank-local "
                    "budget, which a serve group of attn_tp_size>1 cannot share"
                )
            page_config = self._prefix_materialization_kv_config([1])
            kv_pages = [
                temporary_kv_pages(length, page_config.page_size_tokens)
                for length in full_seq_lengths
            ]
            kv_page_cap = budget // gpu_kv_page_bytes(page_config)
        micro_batches, l2_cap = build_prefill_micro_batches(
            seq_lengths,
            token_cap,
            l2_balance=use_l2,
            single_sequence_only=single_sequence_only,
            prefix_hits=prefix_hits,
            kv_pages=kv_pages,
            kv_page_cap=kv_page_cap,
        )
        return micro_batches, l2_cap, single_sequence_only

    def _prefill_model_pass_count(
        self, ctx: PrefillModelPassCountContext, batch: List[int], *,
        _plan_prefill_micro_batches: Callable,
    ) -> int:
        """Return the number of complete model forwards local prefill will run."""
        if not batch:
            return 0
        if ctx._prefill_pool_wave is not None:
            # A pool wave runs one forward per planned chunk, not one per
            # micro-batch of whole prompts: a chunk packs pooled segments and
            # tails. The plan is identical on every rank of the serve group, so
            # the pass counts the transport padding aligns against agree.
            return len(ctx._prefill_pool_wave.plan.chunks)
        if not ctx.enable_prepack:
            micro_batch_size = (
                ctx.engine_config.Module_Batching_Config
                .MoE_prefill_micro_batch_size
            )
            return math.ceil(len(batch) / micro_batch_size)

        # The same inputs prefill_prepacked plans with: computed (suffix)
        # lengths, prompt lengths and the attached prefix of each row.
        full_seq_lengths = []
        compute_cached_tokens = []
        for local_idx in batch:
            seq = ctx.pool.get_sequence(ctx.local_to_uuid[local_idx])
            prefix_state = ctx.prefix.prefix_sequence_states.get(seq.global_idx)
            full_seq_lengths.append(seq.prompt_length)
            compute_cached_tokens.append(
                prefix_state.compute_cached_tokens
                if prefix_state is not None
                else 0
            )
        micro_batches, _, _ = _plan_prefill_micro_batches(
            [
                full - cached
                for full, cached in zip(full_seq_lengths, compute_cached_tokens)
            ],
            full_seq_lengths,
            compute_cached_tokens,
        )
        return len(micro_batches)

    def _prefill_prefix_pool_enabled(
        self, ctx: PrefillPrefixPoolEnabledContext, *, attn_group_size: int,
        _pool_prefill_group_safe: Callable[[], bool],
    ) -> bool:
        """In-wave prefix pool: prefix caching on, planner workspace set, pure DP prefill."""
        workspace = ctx.engine_config.Module_Batching_Config.prefill_prefix_pool_workspace_bytes
        if not ctx.enable_prefix_cache:
            return False
        from batchgen.prefix_reuse.config import PrefixKVGroupSemantic
        if any(
            spec.semantic is not PrefixKVGroupSemantic.FULL_KV
            for spec in ctx.prefix_runtime.group_specs
        ) and (workspace <= 0 or ctx._prefix_integrity):
            # Only the pool path serves MLA-latent prefix reuse (GLM DSA, Kimi-K3);
            # the cross-wave materialization path handles one full-K/V group.
            raise RuntimeError(
                "MLA-latent prefix reuse needs the prefill prefix pool, but this "
                "model's planner sets no pool workspace or the integrity check is on"
            )
        if workspace <= 0 or ctx._prefix_integrity:
            return False
        if (
            hasattr(ctx.parallel_manager, "prefill_uses_resident_ep")
            and ctx.parallel_manager.prefill_uses_resident_ep()
        ) or (
            attn_group_size != 1
            and not _pool_prefill_group_safe()
        ):
            raise RuntimeError(
                "the prefill prefix pool runs per DP rank; this model's prefill "
                "joins cross-rank collectives"
            )
        return True

    def _pool_prefill_group_safe(self, ctx: PrefillPoolParallelContext) -> bool:
        """Whether the PSM plans ONE identical pool wave per serve group (G>1).

        At G>1 a sequence is replicated onto all G ranks of its serve group and
        every one of them prefills it, so the pool is safe only when the whole
        group runs the same chunks in the same order — the per-layer collectives
        then still pair up. The PSM declares that; the resident-EP refusal above
        stands either way, since those passes join EP-world collectives.
        """
        declares = getattr(
            ctx.parallel_manager, "supports_group_pool_prefill", None
        )
        return declares is not None and bool(declares())

    def _prefill_pool_group(
        self, ctx: PrefillPoolParallelContext, *, attn_group_size: int,
    ):
        """The process group a pool wave must be identical over; None at G==1."""
        if attn_group_size <= 1:
            return None
        group = getattr(ctx.parallel_manager, "attn_tp_group", None)
        if group is None:
            raise RuntimeError(
                "the prefill prefix pool at attn_tp_size>1 needs the attention "
                "TP process group to plan one wave per serve group"
            )
        return group

    def _prefill_pool_free_state_slots(
        self, ctx: PrefillPoolStateSlotsContext,
    ) -> int:
        """Free recurrent-state slots, or -1 when the model keeps no such state.

        A pooled segment runs as a row of its own under a pseudo sequence id, so
        it holds one slot for the whole wave on top of the wave's sequences.
        """
        manager = ctx.traj.kda_state_manager()
        if manager is None:
            return -1
        return int(manager.get_stats().num_free_state_items)

    def _next_pool_segment_sid(self, ctx: PoolSegmentSidContext) -> int:
        """Negative Host sequence ids, rank-strided so no two ranks collide."""
        return ctx.prefix.next_pool_segment_sid()

    def _prefix_rank_packing_mode(self, seqs, *, mode: str) -> bool:
        """True when this batch runs the predicted-time wave packer.

        The packer is the default assignment for a pool-served wave: it is the
        arm that gives a prefix family one group's pool and balances the groups
        by predicted prefill seconds, so leaving it off by default would make
        every ordinary wave pay for a straggler the scheduler can see coming.
        A server asks for the arrival-order assignment back with the launch
        parameter ``--prefix-rank-assignment-mode global_idx``, which is what
        the A/B arms use; it holds for every wave of the server.

        It takes NO context: the launch value its delegator resolves is all it
        reads, so there is no worker state to snapshot and an empty context
        would claim otherwise.
        """
        if mode not in ("packing_v2", "global_idx"):
            raise ValueError(
                "--prefix-rank-assignment-mode must be "
                f"'packing_v2' or 'global_idx', got {mode!r}"
            )
        return mode == "packing_v2"

    def _prefix_rank_cost_profile_disabled(self, seqs, *, value: str) -> bool:
        """The launch parameter ``--prefix-rank-cost-profile off``.

        Scoring the packer's targets by predicted seconds and balancing their
        computed tokens are the two arms of the packing comparison. The arm is
        a property of the server: every rank of every group reads the same
        launch value, so the packing stays replicated.

        It takes NO context, for the same reason as the packing-mode predicate
        above: the delegator's launch value is the whole of its input.
        """
        if value not in ("on", "off"):
            raise ValueError(
                f"--prefix-rank-cost-profile must be 'on' or 'off', got {value!r}"
            )
        return value == "off"

    def _cached_prefix_rank_cost_profile(self, worker, disabled: bool = False, *, attn_group_size: int):
        """This deployment's prefill executor cost profile, looked up once.

        Model, device properties and parallel layout are deployment constants,
        so every packer call site shares one lookup. ``None`` is a valid result:
        no profile matches, and the packer then balances computed tokens. A
        profile scores one serve group, so it is keyed by the group size too.

        Every rank must resolve the same profile, or ranks pack differently.
        Device 0 stands for the node, not this rank's own device, and
        cross-node GPU homogeneity is a deployment precondition that the launch
        identity gate verifies; no collective is run here. Each rank logs what
        it resolved, so a divergence is visible in the logs.

        The cache is keyed by ``disabled``, not a single attribute: the A/B
        switch is batch level, so one process serves both arms, and a lookup
        cached under whichever arm arrived first would pin every later batch to
        it. ``disabled`` resolves to ``None`` — the same value as "no profile
        matches", which is exactly the fallback arm the packer already has.
        """
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        if not hasattr(self, "_prefix_rank_cost_profiles"):
            self._prefix_rank_cost_profiles = {}
        key = bool(disabled)
        if key not in self._prefix_rank_cost_profiles:
            from batchgen.prefix_reuse.executor_cost import find_executor_cost_profile

            G = attn_group_size
            group_size = G if G > 1 else 1
            if key:
                profile = None
            else:
                props = torch.cuda.get_device_properties(0)
                profile = find_executor_cost_profile(
                    self.model_name or "",
                    (props.major, props.minor),
                    props.total_memory,
                    self.world_size,
                    group_size,
                )
            self._prefix_rank_cost_profiles[key] = profile
            logging.info(
                "[PREFIX_PACK] rank %d executor cost profile: model=%s "
                "group_size=%d disabled=%d -> %s",
                self.rank,
                (self.model_name or "").lower().rsplit("/", 1)[-1],
                group_size,
                int(key),
                "none" if profile is None
                else f"{profile.model}/{profile.machine}/g{profile.group_size}",
            )
        return self._prefix_rank_cost_profiles[key]

    def _materialized_prefix_hits(self, worker, seqs, *, page_tokens: int, group):
        """``(hits, confirmed, pages, drops)`` for this wave's trajectory re-entries.

        M3 Slice 3, the plan input. EVERY rank of a serve group must derive
        the SAME hit set BEFORE planning, or two ranks pack different chunks
        of the per-layer collectives that pair them up. So the set is read
        from the MIRRORED session index -- replicated read-only data every
        rank folded from the same gather -- and never from the frontier
        registry, which only the group's host-KV leader holds. At G==1 a rank
        is its own group, and its own live attachments are that same data.

        ``confirmed`` is this rank's vote on whether the mirrored set is still
        true. The index distributes on the prefill-config mirror, so it lags a
        commit by one flag-on pass (Slice 3a addendum): a session that has
        advanced or been evicted since would otherwise be materialized from
        pages nobody holds any more. The leader compares the mirrored boundary
        against the record it is holding a reference on; a rank with no
        opinion votes yes and the group reduces with MIN, so one leader's
        "no" takes the whole group back to recomputing.

        ``pages`` maps a hit's wave index to the frontier's resident Host
        pages per KV group -- known only to the leader, which mirrors them to
        the group inside this same pass.

        ``drops`` maps the wave index of every row this pass found a reason
        NOT to plan to that reason, so the plan can decline the row's restore
        by name. Only a rank holding the row's live attachment acts on it.

        The ``attached_tokens < boundary`` veto stays: the frontier record's
        reference pins only its snapshot and registry entry, while the MLA
        pages ``[0, boundary)`` the load reads are kept resident by the row's
        own prefix-cache attachment alone (an eviction under a referenced
        record releases the pages first and only then refuses the record).
        """
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        from batchgen.prefix_reuse.wave_plan import MaterializedPrefix

        if not self._trajectory_commit_requested(seqs):
            return [], True, {}, {}
        digest = self.prefix_cache_runtime_config.namespace_digest
        index = self._traj.trajectory_session_groups
        hits: List[MaterializedPrefix] = []
        pages: Dict[int, Dict[int, List[int]]] = {}
        drops: Dict[int, str] = {}
        confirmed = True
        unknown = 0
        for position, seq in enumerate(seqs):
            if seq is None:
                continue
            attachment = self._traj.trajectory_attachments.get(int(seq.global_idx))
            if group is None:
                # Own group: no peer to agree with, so the live record is the
                # only truth there is.
                if attachment is None:
                    unknown += 1
                    continue
                boundary = int(attachment.boundary_tokens)
                entry_group = int(attachment.owning_group_id)
            else:
                entry = None if index is None else index.resolve(
                    namespace_digest=digest,
                    prompt_token_ids=self._trajectory_tokens(
                        seq, length=seq.prompt_length
                    ).tolist(),
                )
                if entry is None:
                    unknown += 1
                    drops[position] = (
                        "the mirrored session index does not place this session"
                    )
                    continue
                boundary = int(entry.boundary_tokens)
                entry_group = int(entry.owning_group_id)
            reason = self._unmaterializable_frontier(
                seq, boundary=boundary, owning_group_id=entry_group,
                page_tokens=page_tokens,
            )
            if reason is not None:
                logging.info(
                    "[PREFIX_POOL] Rank %s sequence %s continues a frontier at "
                    "%s token(s) the pool cannot materialize (%s); it is "
                    "prefilled in full",
                    self.rank, int(seq.global_idx), boundary, reason,
                )
                drops[position] = f"the pool cannot materialize it ({reason})"
                continue
            hits.append(MaterializedPrefix(index=position, tokens=boundary))
            if attachment is None:
                if self._owns_host_kv(seq):
                    # This rank holds the registry the index came from and has
                    # no live record for it: the mirror is stale.
                    logging.warning(
                        "[PREFIX_POOL] Rank %s holds no live frontier for "
                        "sequence %s, which the mirrored session index places "
                        "at boundary %s; the group recomputes this wave",
                        self.rank, int(seq.global_idx), boundary,
                    )
                    confirmed = False
                continue
            if int(attachment.boundary_tokens) != boundary:
                logging.warning(
                    "[PREFIX_POOL] Rank %s holds sequence %s's frontier at "
                    "boundary %s but the mirrored index says %s; the group "
                    "recomputes this wave",
                    self.rank, int(seq.global_idx),
                    int(attachment.boundary_tokens), boundary,
                )
                drops[position] = (
                    f"the live frontier is at boundary "
                    f"{int(attachment.boundary_tokens)} but the mirrored index "
                    f"says {boundary}"
                )
                confirmed = False
                continue
            state = self._prefix.prefix_sequence_states.get(int(seq.global_idx))
            attached = 0 if state is None else int(state.attached_tokens)
            if attached < boundary:
                # The resident pages the load reads are kept resident by this
                # sequence's own prefix-cache attachment, so that attachment
                # must already cover the frontier. It normally covers more (the
                # lookup attaches the longest resident prefix); less means the
                # cache does not in fact hold what the index advertised.
                logging.warning(
                    "[PREFIX_POOL] Rank %s: sequence %s holds %s attached "
                    "prefix token(s), fewer than the %s-token frontier the "
                    "index advertised; the group recomputes",
                    self.rank, int(seq.global_idx), attached, boundary,
                )
                drops[position] = (
                    f"its prefix-cache attachment covers {attached} token(s), "
                    f"short of the {boundary}-token frontier it must keep "
                    "resident"
                )
                confirmed = False
                continue
            record_pages = {
                int(group_id): [int(page) for page in page_ids]
                for group_id, page_ids in
                attachment.record.page_ids_by_group.items()
            }
            if not record_pages:
                logging.warning(
                    "[PREFIX_POOL] Rank %s: sequence %s's frontier record "
                    "names no resident Host page; the group recomputes",
                    self.rank, int(seq.global_idx),
                )
                drops[position] = "its frontier record names no resident Host page"
                confirmed = False
                continue
            pages[position] = record_pages
        if unknown:
            # A fresh (turn-1) prompt legitimately continues nothing, so this
            # is one aggregate line, not a per-sequence warning. It exists so
            # a wave whose EVERY continuation misses -- the mirrored index
            # lagging its leader's registry -- is visible in the log instead
            # of silently planning an unpooled wave.
            logging.info(
                "[PREFIX_POOL] Rank %s: %s of %s flag-on sequence(s) continue "
                "no session known to %s; they are prefilled in full",
                self.rank, unknown, len(seqs),
                "this rank's live attachments" if group is None
                else "the mirrored session index",
            )
        return hits, confirmed, pages, drops

    def _unmaterializable_frontier(
        self, worker, seq, *, boundary: int, owning_group_id: int, page_tokens: int
    ) -> Optional[str]:
        """Why this frontier cannot become a pool segment, or None.

        Each reason is a property of the wave, identical on every rank that
        derived the same hit, so skipping on it keeps the plans identical.
        """
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker

        if boundary <= 0 or boundary % int(page_tokens):
            return "boundary_off_the_page_grid"
        if boundary > int(seq.prompt_length) - 1:
            # The last prompt token must stay private: it produces the logit.
            return "no_private_token_left"
        if int(seq.decode_dp_group or 0) != int(owning_group_id):
            # The snapshot is head-sharded for the owning group, so a
            # sequence serving elsewhere could not restore its state.
            return "serving_group_is_not_the_owning_group"
        return None

    def _plan_prefill_pool_wave(self, worker, uuids, prefix_states, *, token_cap):
        """Plan this rank's in-wave sharing. CPU-side: the GPU pool is built in
        the prefill phase (`_construct_prefill_pool`), after configure_prefill
        settles the device layout.

        A prompt re-entering after eviction is planned like any other: its
        reconstructed tokens are its prompt. A pooled member's cross-wave
        lookup hit is released and its chain recomputed in-wave; when this
        returns None the wave keeps its lookup hits (the caller releases them
        for a model whose reuse only the pool can serve).

        The page budget is the prefill-phase free memory measured after the
        previous configure_prefill — a deployment constant, so the first wave
        after startup runs unpooled and seeds the measurement.

        At G>1 every rank of the serve group runs this over the same sequences
        and must come out with the same plan: the page and state-slot budgets
        are reduced to the group MIN and the plan digest is compared inside the
        group, so every early return and every chunk is group-identical.

        The plan is also the trajectory RESTORE decision: a row whose frontier
        it materializes carries ``trajectory_restore_boundary`` and starts its
        compute exactly there; every other live trajectory attachment of the
        wave is declined here, at plan time, with the reason it was dropped
        (``_settle_trajectory_restore_plan``) -- on every return path.
        """
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        from batchgen.kv_cache.host_kv_mananger_config import build_gpu_kv_config
        from batchgen.prefix_reuse.levels import order_chunk_items_by_level
        from batchgen.prefix_reuse.pool import scratch_pages_bound
        from batchgen.prefix_reuse.wave_plan import (
            plan_wave_prefix_sharing,
            validate_wave_prefix_plan,
        )
        from batchgen.prefix_reuse.wave_runtime import PoolWave

        seqs = [self.global_batch.get_sequence(uuid) for uuid in uuids]
        total_pages = self._prefill_pool_page_budget
        if total_pages is None:
            # Budget seeding is part of configure_prefill, which every rank of
            # a wave runs in the same step, so an unmeasured budget is
            # group-uniform. Declining before the group lookup also keeps the
            # first wave off attn_tp_group, which the PSM publishes only once
            # prefill has configured.
            logging.info(
                "[PREFIX_POOL] Rank %s first wave keeps the existing path: "
                "prefill-phase pool budget not yet measured",
                self.rank,
            )
            self._settle_trajectory_restore_plan(
                seqs, {}, "the prefill-phase pool budget is not yet measured"
            )
            return None
        group = self._prefill_pool_group()
        prompts = [
            self._trajectory_tokens(seq, length=seq.prompt_length).tolist()
            for seq in seqs
        ]
        page_tokens = int(self.prefix_cache_runtime_config.group_specs[0].raw_page_tokens)
        geometry = build_gpu_kv_config(self.huggingface_ckpt_name, [page_tokens])
        with_v, indexer_layers, indexer_dim = self._prefill_pool_layout(geometry)
        # Planned before this wave's phase transition (the live phase may still
        # be decode) and on this group's subset of the wave: the cap is the
        # server's, the one every wave and every group runs under, resolved by
        # the worker's delegator and handed over explicitly.
        chunk_tokens = token_cap
        state_slots = self._prefill_pool_free_state_slots()
        # M3 Slice 3: the wave's trajectory re-entry hits, derived from
        # replicated data so every rank of the group has the same set, plus
        # this rank's vote on whether that set is still live.
        hits, hits_confirmed, hit_pages, drops = self._materialized_prefix_hits(
            seqs, page_tokens=page_tokens, group=group
        )
        if group is not None:
            # Reached by every rank of the group (uuids is the group's sequence
            # set): the tightest rank's budget is the group's, so the identical
            # prompts then give every rank the identical plan. The
            # materialization vote rides the SAME reduce -- it decides the
            # plan, so it has to be settled before the plan is built, and a
            # second collective here would be a second place to diverge.
            budget = torch.tensor(
                [total_pages, state_slots, 1 if hits_confirmed else 0],
                dtype=torch.int64,
                device=self.torch_device,
            )
            dist.all_reduce(budget, op=dist.ReduceOp.MIN, group=group)
            total_pages = int(budget[0].item())
            state_slots = int(budget[1].item())
            hits_confirmed = bool(int(budget[2].item()))
        if hits and not hits_confirmed:
            logging.warning(
                "[PREFIX_POOL] Rank %s drops %d materialized prefix(es): the "
                "serve group could not confirm the mirrored frontier set; the "
                "wave recomputes them",
                self.rank, len(hits),
            )
            for hit in hits:
                drops.setdefault(
                    int(hit.index),
                    "the serve group could not confirm the mirrored frontier set",
                )
            hits = []
        if self._prefill_pool_psm_layout() is not None:
            # The pool holds EVERY token of a chunk, tails as well as segments,
            # so a chunk wider than the pool can never be planned; leave half the
            # pages for the pooled segments those tails read.
            pool_chunk_tokens = (
                max(0, (total_pages - len(prompts)) // 2) * page_tokens
            )
            if pool_chunk_tokens < chunk_tokens:
                logging.info(
                    "[PREFIX_POOL] Rank %s caps the pool chunk at %d tokens "
                    "(cap %d): %d pool pages hold every chunk token",
                    self.rank, pool_chunk_tokens, chunk_tokens, total_pages,
                )
                chunk_tokens = pool_chunk_tokens
        scratch = scratch_pages_bound(
            chunk_tokens, max(len(prompt) for prompt in prompts), len(prompts), page_tokens
        )
        if total_pages <= scratch:
            logging.info(
                "[PREFIX_POOL] Rank %s wave keeps the existing path: pool %d pages <= scratch %d",
                self.rank, total_pages, scratch,
            )
            self._settle_trajectory_restore_plan(
                seqs, drops,
                f"the wave is not pooled: pool {total_pages} pages <= scratch "
                f"{scratch}",
            )
            return None
        # A model that declares group-safe pool prefill (K3) writes a chunk's KV
        # before attending over it AND runs its recurrent layers level by level,
        # so a whole prefix tree finishes in one forward. Read from the PSM like
        # the pool layout above; GPT-OSS and GLM declare neither.
        declares = getattr(self.parallel_manager, "supports_group_pool_prefill", None)
        level_wise = declares is not None and bool(declares())
        plan_kwargs = dict(
            block_tokens=page_tokens,
            pool_pages=total_pages - scratch,
            chunk_tokens=chunk_tokens,
            # Every GLM prefill forward streams all experts, so a chunk is
            # expensive; its attention writes the chunk's KV before reading it.
            same_chunk_ancestors=level_wise or not with_v,
            # DSA pool attention gathers every row's prefix indexer keys (bf16
            # plus an FP8 copy, ~384 B each); keep that inside the workspace.
            chunk_keys=None if with_v else 8 * 1024 * 1024,
        )
        if hits and not plan_kwargs["same_chunk_ancestors"]:
            # A model whose rows cannot read a segment resident in their own
            # chunk cannot run a load either; named here rather than refused
            # inside the planner on every wave.
            logging.info(
                "[PREFIX_POOL] Rank %s drops %d materialized prefix(es): this "
                "model's pool does not read a segment resident in its own "
                "chunk",
                self.rank, len(hits),
            )
            for hit in hits:
                drops.setdefault(
                    int(hit.index),
                    "this model's pool does not read a segment resident in its "
                    "own chunk",
                )
            hits = []
        # Every computed segment holds a recurrent-state slot for the wave on
        # top of the wave's own sequences, so the planner pools only as many
        # (the highest-saving ones) as the group-MIN slot budget leaves spare,
        # instead of declining the whole wave when the full tree does not fit.
        if state_slots >= 0:
            plan_kwargs["max_computed_segments"] = max(0, state_slots - len(prompts))
        # The planner decides per row which hits it materializes (arbitration
        # against the in-wave pool, then partial fit) and names every one it
        # declines. Deterministic in the group's own inputs (the same prompts,
        # the same hit set, the same group-MIN budget), so every rank of the
        # group keeps and declines the same rows.
        plan = plan_wave_prefix_sharing(prompts, materialized=hits, **plan_kwargs)
        if plan.declined_materialized:
            logging.warning(
                "[PREFIX_POOL] Rank %s plans %d of %d materialized prefix(es); "
                "declined by the planner: %s",
                self.rank, len(hits) - len(plan.declined_materialized),
                len(hits),
                ", ".join(
                    f"sequence {int(seqs[index].global_idx)} ({reason})"
                    for index, reason in plan.declined_materialized
                ),
            )
            for index, reason in plan.declined_materialized:
                drops.setdefault(int(index), f"the wave planner declined it ({reason})")
        if level_wise:
            # Rows grouped by level, so a recurrent layer runs one packed call
            # per level, each after the level that left its parents' state.
            plan = order_chunk_items_by_level(plan)
        validate_wave_prefix_plan(plan, prompts)
        self._check_pool_plan_identical(plan, group)
        # Nothing shared, or no room for the state: a pooled segment is a prefill
        # row of its own, so on a model with per-sequence recurrent state it
        # holds a slot of its own until the wave ends. The slot budget is the
        # group MIN, so this decline is group-uniform like every other.
        # A materialized segment is no row, so it holds no recurrent state
        # slot; only the segments a row computes do.
        state_slots_needed = (
            len(plan.segments) - len(plan.materialized_segments) + len(prompts)
        )
        if not plan.segments or 0 <= state_slots < state_slots_needed:
            logging.info(
                "[PREFIX_POOL] Rank %s wave keeps the existing path: %d segments "
                "shared, %d state slots needed, %d free",
                self.rank, len(plan.segments), state_slots_needed, state_slots,
            )
            self._settle_trajectory_restore_plan(
                seqs, drops,
                f"the wave is not pooled: {len(plan.segments)} segment(s), "
                f"{state_slots_needed} state slot(s) needed, {state_slots} free",
            )
            return None
        wave = PoolWave(
            plan=plan,
            prompts=prompts,
            uuids=list(uuids),
            global_ids=[seq.global_idx for seq in seqs],
            page_tokens=page_tokens,
            total_pages=total_pages,
            scratch_pages=scratch,
        )
        for segment_id in sorted(wave.materialized):
            members = plan.segments[segment_id].members
            source = next(
                (hit_pages[index] for index in members if index in hit_pages),
                None,
            )
            if source is None:
                # A follower holds no record: the leader mirrors these pages
                # inside this same pass, before the forward reads them.
                continue
            wave.bind_materialized_host_pages(segment_id, source)
        # A materialized member's chain IS its frontier, so its compute start
        # is the snapshot boundary: that row, and only that row, restores.
        restore_boundaries = {
            index: wave.chain_tokens(index)
            for segment_id in wave.materialized
            for index in plan.segments[segment_id].members
        }
        self._replace_prefix_lookup_hits(
            seqs, prefix_states,
            [wave.chain_tokens(index) for index in range(len(seqs))],
            retain=frozenset(restore_boundaries),
            restore_boundaries=restore_boundaries,
        )
        self._settle_trajectory_restore_plan(
            seqs, drops, "the wave plan does not materialize its frontier"
        )
        logging.info(
            "[PREFIX_POOL] Rank %s wave plan: %d prompts, %d segments, %d chunks, "
            "threshold %s, compute %d of %d prompt tokens, %d pool pages "
            "(%d for segments, %d scratch)",
            self.rank, len(prompts), len(plan.segments), len(plan.chunks), plan.threshold,
            plan.computed_tokens, plan.prompt_tokens, total_pages,
            total_pages - scratch, scratch,
        )
        return wave

    def _prefill_pool_psm_layout(self, ctx: PrefillPoolParallelContext):
        """The PSM's pool-geometry override, or None to use the Host geometry.

        A model whose pool holds the EXPANDED heads (K3: 192-wide K and
        128-wide V per LOCAL head against one 576-wide latent) takes its pool
        geometry from the PSM; the Host geometry is the pool's only while
        prefill attention reads exactly what the Host stores.
        """
        layout = getattr(ctx.parallel_manager, "prefill_pool_layout", None)
        return layout() if layout is not None else None

    def _prefill_pool_page_dims(
        self, ctx: PrefillPoolParallelContext, geometry,
    ) -> tuple:
        """(layers, heads, k_head_dim, v_head_dim) of one pool page row.

        The PSM override is a sibling with the SAME context, so it is called
        with this one rather than through the worker delegator, which added
        nothing but the engine lookup.
        """
        layout = self._prefill_pool_psm_layout(ctx)
        if layout:
            return (
                int(layout["num_layers"]),
                int(layout["num_kv_heads"]),
                int(layout["k_head_dim"]),
                int(layout["v_head_dim"]),
            )
        return (
            geometry.num_layers,
            geometry.num_k_heads,
            geometry.k_head_dim,
            geometry.k_head_dim,
        )

    def _prefill_pool_page_bytes(
        self, ctx: PrefillPoolPageBytesContext, *,
        _prefill_pool_layout: Callable,
        _prefill_pool_page_dims: Callable,
    ) -> int:
        from batchgen.kv_cache.host_kv_mananger_config import build_gpu_kv_config

        page_tokens = int(ctx.prefix_runtime.group_specs[0].raw_page_tokens)
        geometry = build_gpu_kv_config(ctx.huggingface_ckpt_name, [page_tokens])
        element = torch.empty((), dtype=geometry.kv_dtype).element_size()
        with_v, indexer_layers, indexer_dim = _prefill_pool_layout(geometry)
        layers, heads, k_head_dim, v_head_dim = _prefill_pool_page_dims(geometry)
        return (
            layers * page_tokens * heads
            * (k_head_dim + (v_head_dim if with_v else 0))
            + len(indexer_layers) * page_tokens * indexer_dim
        ) * element

    def _measure_prefill_prefix_kv_budget(self, worker) -> None:
        """Measure the prefill-phase prefix KV budget. Runs after every
        configure_prefill while prefix caching is on; the device layout at
        that point is a deployment constant, so the measurement is stable.

        The planner's ``prefill_prefix_pool_workspace_bytes`` is the forward
        workspace of one prefill micro-batch; the free memory beside it is the
        prefix KV a prefill forward may hold, either the in-wave pool's pages
        or the temporary full-length KV of a cross-wave prefix hit. A model
        whose planner sets no workspace has no such budget (None)."""
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        from batchgen.prefix_reuse.pool import pool_pages_from_budget

        pool_enabled = self._prefill_prefix_pool_enabled()
        torch.cuda.synchronize(self.torch_device)
        torch.cuda.empty_cache()
        free_bytes, _ = torch.cuda.mem_get_info(self.local_rank)
        batching = self.engine_config.Module_Batching_Config
        workspace = int(batching.prefill_prefix_pool_workspace_bytes)
        kv_budget = max(0, free_bytes - workspace) if workspace > 0 else None
        if kv_budget != self._prefill_prefix_kv_budget_bytes:
            logging.info(
                "[PREFIX_KV] Rank %s prefill-phase prefix KV budget: %s B "
                "(%d B free, %d B workspace, was %s)",
                self.rank, kv_budget, free_bytes, workspace,
                self._prefill_prefix_kv_budget_bytes,
            )
        self._prefill_prefix_kv_budget_bytes = kv_budget
        if not pool_enabled:
            return
        budget = pool_pages_from_budget(
            free_bytes, workspace, self._prefill_pool_page_bytes(),
        )
        previous = self._prefill_pool_page_budget
        self._prefill_pool_page_budget = budget
        if previous is None or previous != budget:
            logging.info(
                "[PREFIX_POOL] Rank %s prefill-phase pool budget: %d pages "
                "(%d B free, %d B workspace, was %s)",
                self.rank, budget, free_bytes,
                int(batching.prefill_prefix_pool_workspace_bytes), previous,
            )

    def _construct_prefill_pool(self, worker, wave) -> None:
        """Build the planned wave's GPU pool and executor in the prefill phase."""
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        from batchgen.prefix_reuse.executor import PoolChunkExecutor
        from batchgen.prefix_reuse.pool import PrefillPrefixPool

        num_pages = wave.total_pages
        budget = self._prefill_pool_page_budget
        group = self._prefill_pool_group()
        if group is not None:
            # One pool size per serve group: a rank-local jitter shrink (or a
            # rank-local failure) would leave the group's ranks in different
            # chunk shapes mid-collective. The tightest rank decides.
            reduced = torch.tensor(
                [-1 if budget is None else int(budget)],
                dtype=torch.int64,
                device=self.torch_device,
            )
            dist.all_reduce(reduced, op=dist.ReduceOp.MIN, group=group)
            budget = None if int(reduced.item()) < 0 else int(reduced.item())
        if budget is not None and budget < num_pages:
            # Free-memory jitter of a few pages between measurements is not
            # layout drift. The pool only ever needs the planned segment peak
            # plus the executor's exact transient bound; shrink to the fresh
            # budget when that still fits, fail loud when it cannot.
            required = wave.plan.peak_pool_pages + wave.scratch_pages
            if budget < required:
                raise RuntimeError(
                    f"Rank {self.rank}: prefill pool budget shrank to {budget} "
                    f"pages below the required {required} (peak "
                    f"{wave.plan.peak_pool_pages} + scratch {wave.scratch_pages})"
                )
            logging.info(
                "[PREFIX_POOL] Rank %s pool shrinks to the fresh budget: "
                "%d pages (planned %d, required %d)",
                self.rank, budget, num_pages, required,
            )
            num_pages = budget
        from batchgen.kv_cache.host_kv_mananger_config import build_gpu_kv_config

        geometry_page_bytes = self._prefill_pool_page_bytes()
        geometry = build_gpu_kv_config(self.huggingface_ckpt_name, [wave.page_tokens])
        with_v, indexer_layers, indexer_dim = self._prefill_pool_layout(geometry)
        pool_layers, pool_heads, k_head_dim, v_head_dim = (
            self._prefill_pool_page_dims(geometry)
        )
        pool = PrefillPrefixPool(
            num_layers=pool_layers,
            num_pages=num_pages,
            page_tokens=wave.page_tokens,
            num_kv_heads=pool_heads,
            head_dim=k_head_dim,
            dtype=geometry.kv_dtype,
            device=self.torch_device,
            with_v=with_v,
            v_head_dim=v_head_dim,
            indexer_layers=indexer_layers,
            indexer_dim=indexer_dim,
        )
        if pool.page_bytes != geometry_page_bytes:
            raise RuntimeError(
                f"prefix pool page is {pool.page_bytes} B, budget assumed "
                f"{geometry_page_bytes} B"
            )
        wave.pool = pool
        pool.kernel_dump = self._new_pool_kernel_dump_recorder(wave)
        wave.executor = PoolChunkExecutor(
            wave.plan, wave.prompts, pool,
            pad_poison=self._pool_pad_poison_enabled(wave),
        )
        free_after, device_total = torch.cuda.mem_get_info(self.local_rank)
        reserved = torch.cuda.memory_reserved(self.local_rank)
        self._prefill_pool_base = (
            torch.cuda.memory_allocated(self.local_rank),
            reserved,
            device_total - free_after - reserved,
        )
        torch.cuda.reset_peak_memory_stats(self.local_rank)
        logging.info(
            "[PREFIX_POOL] Rank %s pool up: %d pages (%d B each, %d segments)",
            self.rank, pool.num_pages, pool.page_bytes, len(wave.plan.segments),
        )

    def _check_pool_plan_identical(self, worker, plan, group, *, attn_group_size: int) -> None:
        """Raise unless every rank of the serve group planned the same wave.

        Deterministic planning over replicated prompts and a group-MIN budget
        gives one plan per group. A divergence would send two ranks through
        different chunks of the per-layer collectives that pair them up, so it
        is named here instead of deadlocking or corrupting a row later. No-op at
        G==1, where the rank's plan is its own.
        """
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        if group is None:
            return
        hasher = hashlib.blake2b(digest_size=7)
        for chunk in plan.chunks:
            for item in chunk.items:
                hasher.update(
                    f"{item.kind}:{item.ref}:{item.token_start}:"
                    f"{item.token_end}:{item.prefix_segments};".encode()
                )
            hasher.update(f"|{chunk.release_after}|".encode())
        digest = torch.tensor(
            [int.from_bytes(hasher.digest(), "little")],
            dtype=torch.int64,
            device=self.torch_device,
        )
        gathered = [
            torch.zeros_like(digest) for _ in range(attn_group_size)
        ]
        dist.all_gather(gathered, digest, group=group)
        if len({int(item.item()) for item in gathered}) != 1:
            raise RuntimeError(
                f"Rank {self.rank}: the prefill prefix pool plan diverged inside "
                f"the serve group ({len(plan.chunks)} chunks, "
                f"{len(plan.segments)} segments, digest {int(digest.item())})"
            )

    def _prefill_pool_layout(self, worker, geometry) -> tuple:
        """(with_v, indexer_layers, indexer_dim) of this model's prefill pool.

        A DSA model (dual Host KV) pools the indexer keys of its full (top-k
        computing) layers next to its attention rows; a non-DSA model pools
        K/V and no index keys.

        Whether the pool carries V is a property of the POOL ROW, not of DSA:
        the row's owner is the PSM's ``prefill_pool_layout`` when it declares
        one (a model whose pool holds the expanded per-head K AND V, as
        ``_prefill_pool_page_dims`` already reads for the widths), otherwise
        the Host geometry, whose DSA form is a single latent with no V side.
        Deciding it from "is the coordinator dual" instead would drop the V
        rows of any DSA model that pools expanded heads while
        ``_prefill_pool_page_dims`` kept declaring ``v_head_dim``, and the
        two would disagree silently -- ``PrefillPrefixPool`` simply leaves
        ``self.v`` as None.
        """
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        if not self._prefill_host_view_is_dual():
            return True, (), 0
        psm_layout = self._prefill_pool_psm_layout()
        with_v = bool(psm_layout and int(psm_layout["v_head_dim"]) > 0)
        if self.model is None:
            # Wave planning runs at reservation time (decode phase), where the
            # model object is not resident. The layout is a model constant and
            # the budget gate guarantees a prefill measured it before any wave
            # plans, so the cached copy is always available here.
            cached = getattr(self, "_prefill_pool_layout_cache", None)
            if cached is None:
                raise RuntimeError(
                    "DSA prefill pool: layout requested before the first "
                    "prefill measured it"
                )
            return cached
        # The layers' own indexers decide what prefill scores and offloads: a
        # shared layer has none and reuses the previous full layer's top-k.
        indexers = {}
        for layer_idx, layer in enumerate(self.model.model.layers):
            attn = getattr(layer.self_attn, "module", layer.self_attn)
            if getattr(attn, "indexer", None) is not None:
                indexers[layer_idx] = attn.indexer
        # Compare against the primary cache's LOGICAL depth, not its row count:
        # a profile that stores its attention layers densely carries a
        # logical->physical map (Kimi-K3's 93 engine layers -> 24 MLA rows), and
        # `num_layers` is then the physical row count, which no model's layer
        # stack is ever equal to. The check is "does this cache describe this
        # model's depth", so it must read whichever of the two the profile
        # declares -- a table lookup, not a per-model branch.
        logical_map = getattr(geometry, "logical_to_physical_layer", None)
        kv_depth = len(logical_map) if logical_map else geometry.num_layers
        if not indexers or len(self.model.model.layers) != kv_depth:
            raise RuntimeError(
                "DSA prefill pool: indexer layers do not match the KV geometry "
                f"({len(indexers)} indexer layers, {len(self.model.model.layers)} "
                f"engine layers, KV logical depth {kv_depth})"
            )
        dims = {int(indexer.index_head_dim) for indexer in indexers.values()}
        if len(dims) != 1:
            raise RuntimeError(f"DSA prefill pool: mixed indexer head dims {sorted(dims)}")
        layout = (with_v, tuple(indexers), dims.pop())
        self._prefill_pool_layout_cache = layout
        return layout

    def _replace_prefix_lookup_hits(
        self, worker, seqs, prefix_states, chain_tokens, retain=frozenset(),
        restore_boundaries=None,
    ) -> None:
        """Release each cross-wave lookup hit; a prompt starts at its chain end.

        A serve-group follower plans the same wave as its leader but holds no
        lookup of its own — the leader owns every coordinator attachment and
        mirrors the state afterwards — so a missing entry is its normal case and
        leaves nothing to release.

        ``retain`` names the wave indices whose attachment must SURVIVE this
        call (M3 Slice 3: a materialized prefix is read out of the prefix
        cache's own resident pages for the whole wave, and that attachment is
        what keeps them resident; a computed segment needs no such protection
        because the wave publishes it afresh). The handle moves into the
        sequence's retained-handle tuple, so the one release lane every
        sequence leaves through frees it exactly once.

        ``restore_boundaries`` maps the wave indices whose trajectory frontier
        the plan materialized to that boundary: the row's restore decision,
        carried on its state as ``trajectory_restore_boundary``. Every other
        row gets 0, i.e. no restore.
        """
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        from batchgen.prefix_reuse.prefill import PrefixCacheSequenceState

        restores = restore_boundaries or {}
        for index, (seq, end) in enumerate(zip(seqs, chain_tokens)):
            old = prefix_states.get(seq.global_idx)
            handle = 0 if old is None else int(old.attachment_handle)
            if handle and index not in retain:
                self.prefix_cache_coordinator.release_attachment(handle)
                handle = 0
            restore = int(restores.get(index, 0))
            if restore and restore != int(end):
                raise RuntimeError(
                    f"Rank {self.rank}: sequence {int(seq.global_idx)} is "
                    f"planned to restore at token {restore} but its compute "
                    f"starts at {int(end)}"
                )
            state = PrefixCacheSequenceState(
                lookup_result=None,
                attached_tokens=end,
                compute_cached_tokens=end,
                commit_attachment_handles=(handle,) if handle else (),
                trajectory_restore_boundary=restore,
            )
            prefix_states[seq.global_idx] = state
            self._prefix.install_state(seq.global_idx, state)

    def _settle_trajectory_restore_plan(self, worker, seqs, drops, default_reason) -> None:
        """Decline, at plan time, every live attachment the plan does not restore.

        Runs once at the end of every wave plan, on every return path, over
        the wave's sequences in wave order. An attachment whose row carries a
        ``trajectory_restore_boundary`` is the plan's to restore and stays; any
        other is consumed HERE with ``not_planned`` and the reason the plan
        dropped it (``drops``: wave index -> reason; ``default_reason``
        otherwise), so it never reaches the restore seam and its reference is
        released exactly once. Only the rank holding attachments (the
        host-KV leader) has anything to do; a follower's map is empty.
        """
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        if not self._traj.trajectory_attachments:
            return
        from batchgen.prefix_reuse.trajectory_restore import DECLINE_NOT_PLANNED

        restorer = self._traj.trajectory_restorer
        for index, seq in enumerate(seqs):
            if seq is None:
                continue
            global_id = int(seq.global_idx)
            attachment = self._traj.trajectory_attachments.get(global_id)
            if attachment is None:
                continue
            state = self._prefix.prefix_sequence_states.get(global_id)
            planned = 0 if state is None else int(state.trajectory_restore_boundary)
            if planned:
                if planned != int(attachment.boundary_tokens):
                    raise RuntimeError(
                        f"Rank {self.rank}: sequence {global_id} is planned to "
                        f"restore at token {planned} but its live frontier is "
                        f"at {int(attachment.boundary_tokens)}"
                    )
                continue
            if restorer is None:
                raise RuntimeError(
                    f"Rank {self.rank}: sequence {global_id} holds a live "
                    "trajectory attachment but this rank has no restorer"
                )
            del self._traj.trajectory_attachments[global_id]
            restorer.decline(
                attachment,
                reason=DECLINE_NOT_PLANNED,
                detail=f"at plan time: {drops.get(index, default_reason)}",
            )

    def _prefill_pool_segment_host_pages(
        self, ctx: PrefillPoolWaveContext,
    ) -> int:
        """Host pages this wave allocates for its segments.

        A materialized segment is excluded: its pages are the frontier's,
        already allocated and retained by the wave that computed it.
        """
        wave = ctx._prefill_pool_wave
        if wave is None:
            return 0
        return sum(
            seg.tokens for seg in wave.computed_segments()
        ) // wave.page_tokens

    def _allocate_prefill_pool_host_pages(self, worker, host_views, prefix_states) -> None:
        """One Host copy per pooled segment; members attach their chain of them."""
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        from batchgen.prefix_reuse.wave_runtime import allocate_segment_host_pages

        wave = self._prefill_pool_wave
        allocated = allocate_segment_host_pages(
            wave, host_views, self._next_pool_segment_sid
        )
        if allocated != self._prefill_pool_segment_host_pages():
            raise RuntimeError(
                f"pool segments got {allocated} Host pages, planned "
                f"{self._prefill_pool_segment_host_pages()}"
            )
        for index, global_id in enumerate(wave.global_ids):
            if not wave.chains[index]:
                continue
            state = replace(
                prefix_states[global_id],
                planned_shared_page_ids={
                    group_id: tuple(pages)
                    for group_id, pages in wave.chain_host_pages(index).items()
                },
            )
            prefix_states[global_id] = state
            # ``install_state`` of the replaced value, not a dedicated
            # replace-setter: the call site needs the replaced state anyway
            # for its wave-local ``prefix_states`` map, so the owner would
            # otherwise duplicate the ``replace`` the caller already did.
            self._prefix.install_state(global_id, state)
            # A segment's Host pages live until its last member releases. A
            # materialized segment has no pseudo sequence to refcount: its
            # pages are the prefix cache's, kept by this sequence's own
            # attachment and released with it.
            sids = tuple(
                wave.segment_sids[seg]
                for seg in wave.chains[index]
                if seg not in wave.materialized
            )
            self._prefix.add_pool_member_refs(global_id, sids)

    def _materialized_segment_loader(self, worker, wave):
        """Staging for this wave's loads, or None when it materializes none.

        One loader per prefix-cache KV group, keyed by group id. Group 0 is the
        primary region's MLA latent, which the model expands. A dual-region
        (DSA) model adds group 1, the page-split FP8 indexer keys: those pages
        ARE the index keys, so they are read back and dequantized rather than
        reconstructed, and they are cut at the same page boundaries as the
        latent so the two slab streams pair up one for one.
        """
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        from batchgen.kv_cache.host_kv_mananger_config import (
            build_gpu_kv_config,
            build_gpu_kv_config_aux,
        )
        from batchgen.prefix_reuse.pool_materialize import (
            DEFAULT_SLAB_BYTES,
            MaterializedSegmentLoader,
            PageSplitIndexKeyLoader,
        )

        if not wave.materialized:
            return None
        views = self._host_views_by_group()
        if sorted(views) not in ([0], [0, 1]):
            raise RuntimeError(
                "materializing a prefix needs the primary Host KV region "
                "(group 0) alone or beside the DSA indexer region (group 1), "
                f"this model has groups {sorted(views)}"
            )
        geometry = build_gpu_kv_config(
            self.huggingface_ckpt_name, [int(wave.page_tokens)]
        )
        primary = MaterializedSegmentLoader(
            worker_view=views[0],
            num_layers=int(geometry.num_layers),
            page_tokens=int(wave.page_tokens),
            latent_dim=int(geometry.num_k_heads) * int(geometry.k_head_dim),
            dtype=geometry.kv_dtype,
            device=self.torch_device,
            max_slab_bytes=DEFAULT_SLAB_BYTES,
        )
        loaders = {0: primary}
        if 1 in views:
            aux_geometry = build_gpu_kv_config_aux(
                self.huggingface_ckpt_name, [int(wave.page_tokens)]
            )
            if aux_geometry is None:
                raise RuntimeError(
                    "this model has a second Host KV region but no DSA "
                    "indexer profile, so the auxiliary pages cannot be read "
                    "back as index keys"
                )
            if int(aux_geometry.num_layers) != int(geometry.num_layers):
                raise RuntimeError(
                    "the DSA indexer region keeps "
                    f"{int(aux_geometry.num_layers)} row(s) against the "
                    f"latent region's {int(geometry.num_layers)}; a prefix "
                    "load addresses both by the same row"
                )
            loaders[1] = PageSplitIndexKeyLoader(
                worker_view=views[1],
                num_layers=int(aux_geometry.num_layers),
                page_tokens=int(wave.page_tokens),
                carrier_bytes=self._prefix_pool_index_key_carrier(aux_geometry),
                slab_pages=primary.slab_pages,
                dtype=geometry.kv_dtype,
                device=self.torch_device,
            )
        logging.info(
            "[PREFIX_POOL] Rank %s stages %d materialized segment(s) through "
            "%d B of device staging (%d Host page(s) per slab, groups %s)",
            self.rank, len(wave.materialized),
            sum(loader.staging_bytes for loader in loaders.values()),
            primary.slab_pages, sorted(loaders),
        )
        return loaders

    def _prefix_pool_index_key_carrier(self, worker, aux_geometry) -> int:
        """Bytes per token in the auxiliary region, scale bytes included.

        A page-split FP8 region declares ``k_head_dim`` in BYTES per token (the
        e4m3 data bytes plus one fp32 scale); the loader turns that into the
        key width through the layout's own statement. An AoS indexer region
        declares ELEMENTS and carries no scale, so reading it as page-split
        bytes would silently reinterpret every key: refused here.
        """
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        if aux_geometry.kv_dtype is not torch.float8_e4m3fn:
            raise RuntimeError(
                "loading a cached prefix's index keys needs the page-split "
                f"FP8 indexer region, got {aux_geometry.kv_dtype}"
            )
        return int(aux_geometry.k_head_dim)

    def _materialized_pool_index_key_layers(self, worker, pool):
        """The attention layers that materialize INDEX KEYS, in pool-row order.

        The same assertion the latent walk makes, against the pool's indexer
        rows: the n-th index-key-materializing layer must be indexer row n,
        because the auxiliary Host region keeps one dense row per DSA layer in
        engine order and the load addresses it by that one index. A model whose
        two orders disagree would load every layer's prefix keys into the wrong
        row and the suffix would select plausible keys from the wrong depth.
        """
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        rows = getattr(pool, "indexer_row", None) if pool is not None else None
        if not rows:
            raise RuntimeError(
                "a materialized DSA prefix needs the pool's indexer rows"
            )
        layers = []
        for engine_layer, decoder_layer in enumerate(self.model.model.layers):
            attn = decoder_layer.self_attn
            if not hasattr(attn, "materialize_pool_segment_index_keys"):
                continue
            row = rows.get(engine_layer)
            if row is None:
                raise RuntimeError(
                    f"engine layer {engine_layer} materializes index keys but "
                    "has no prefix-pool indexer row"
                )
            if row != len(layers):
                raise RuntimeError(
                    f"engine layer {engine_layer} is pool indexer row {row} "
                    f"but the {len(layers)}th index-key materializing layer; "
                    "the pool and Host orders must agree for a prefix load"
                )
            layers.append(attn)
        if len(layers) != len(rows):
            raise RuntimeError(
                f"{len(layers)} layer(s) materialize index keys against "
                f"{len(rows)} pool indexer row(s)"
            )
        return layers

    def _load_materialized_segments(self, worker, wave, rows, loaders) -> int:
        """Fill this chunk's loaded segments BEFORE its forward reads them.

        Per slab: one H2D per KV group of the committed prefix across every
        layer, waited on, then each layer writes its own slab into the pool
        slots that layer's row would have written. The primary region holds the
        MLA latent, which the model expands through its own projection; a DSA
        model's auxiliary region holds the index keys themselves, which are
        dequantized from their page-split FP8 bytes and written as keys. Both
        regions carry the SAME segment pages in the same order, so the slabs
        pair up and one window of pool slots receives both. Returns the number
        of tokens loaded, for the wave's metrics.
        """
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        if not rows.materialized:
            return 0
        if not loaders:
            raise RuntimeError(
                f"Rank {self.rank}: this chunk loads "
                f"{len(rows.materialized)} materialized segment(s) with no "
                "staging; the wave was built without a loader"
            )
        layers = self._materialized_pool_layers()
        key_layers = (
            self._materialized_pool_index_key_layers(wave.pool)
            if 1 in loaders else ()
        )
        loaded = 0
        for stage in rows.materialized:
            pages_by_group = wave.segment_host_pages.get(stage.segment_id)
            if not pages_by_group:
                raise RuntimeError(
                    f"Rank {self.rank}: materialized segment "
                    f"{stage.segment_id} names no Host page to load from"
                )
            if sorted(pages_by_group) != sorted(loaders):
                raise RuntimeError(
                    f"Rank {self.rank}: materialized segment "
                    f"{stage.segment_id} names Host pages in groups "
                    f"{sorted(pages_by_group)}, this model loads groups "
                    f"{sorted(loaders)}"
                )
            pages = pages_by_group[0]
            streams = [
                loaders[group].slabs(
                    host_page_ids=pages_by_group[group],
                    tokens=int(stage.tokens),
                )
                for group in sorted(loaders)
            ]
            for slabs in zip(*streams):
                latent_slab = slabs[0]
                window = stage.slots[
                    latent_slab.token_start:
                    latent_slab.token_start + latent_slab.tokens
                ]
                for physical, attn in enumerate(layers):
                    attn.materialize_pool_segment(
                        latent=latent_slab.latent[physical], slots=window
                    )
                for key_slab in slabs[1:]:
                    if (
                        key_slab.token_start != latent_slab.token_start
                        or key_slab.tokens != latent_slab.tokens
                    ):
                        raise RuntimeError(
                            f"Rank {self.rank}: materialized segment "
                            f"{stage.segment_id} loaded index keys for "
                            f"[{key_slab.token_start}, {key_slab.tokens}) "
                            f"against a latent slab of "
                            f"[{latent_slab.token_start}, "
                            f"{latent_slab.tokens})"
                        )
                    for physical, attn in enumerate(key_layers):
                        attn.materialize_pool_segment_index_keys(
                            index_keys=key_slab.keys[physical], slots=window
                        )
                loaded += int(latent_slab.tokens)
            logging.info(
                "[PREFIX_POOL] Rank %s loaded materialized segment %s: %d "
                "token(s) over %d Host page(s) per region into %d pool "
                "page(s), groups %s",
                self.rank, stage.segment_id, int(stage.tokens), len(pages),
                len(stage.pages), sorted(loaders),
            )
        return loaded

    def _pool_probe_debug(self, ctx: PoolProbeContext, wave):
        """The probe keys this wave armed, or None. Read ONCE per wave.

        The four pooled probes (epilogue trace, offload byte check, kernel dump,
        pad poison) are armed through the server's debug state, which the wave
        adopted at its start, so an operator arms a probe against a server
        that is already up.

        Off is the production path and has to cost nothing measurable: an
        empty debug state costs one attribute read, the answer is cached on
        the wave, and the result dict is not created until a probe key is
        actually found. Each probe's own setup then imports its module only
        after this returned non-None.

        The two cached answers are written onto the WAVE the caller passed, in
        place: they are the wave's own fields, not worker attributes, so this
        stage needs no outcome for them.
        """

        if wave.probe_debug_read:
            return wave.probe_debug
        armed = None
        server_debug = ctx._phase_prefill_debug
        if server_debug:
            for key in _POOL_PROBE_DEBUG_KEYS:
                value = server_debug.get(key)
                if value is not None:
                    if armed is None:
                        armed = {}
                    armed[key] = value
        wave.probe_debug = armed
        wave.probe_debug_read = True
        return armed

    def _new_pool_epilogue_tracer(self, worker, wave):
        """Server-level ``debug_state.prefix_pool_epilogue_trace``.

        Same seam as ``prefix_pool_offload_check`` and the kernel dump: the
        key is read from the debug state the wave adopted. Traces the only stretch of the
        pooled path the local oracles and the GPU replays never reached - the
        tail-row gather, the streamed-SP8 shard reassembly and the row ->
        wave index -> uuid -> sequence pairing that carries a hidden state to
        sampling. Returns None when the batch did not ask.

        The flag check comes BEFORE the import: on a production wave
        ``_pool_probe_debug`` has already answered None and this method touches
        nothing else. A malformed flag value is a diagnostics failure, not a
        serving failure, so it leaves a ``probe_error`` line and no tracer.
        """
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        debug = self._pool_probe_debug(wave)
        if debug is None:
            return None
        from batchgen.prefix_reuse.epilogue_trace import tracer_from_debug
        from batchgen.prefix_reuse.probe_guard import run_probe

        tracer = run_probe(
            "epilogue_trace.setup",
            lambda: tracer_from_debug(debug, rank=self.rank, path="pool"),
            rank=self.rank,
        )
        if tracer is None:
            return None
        logging.info(
            "[EPILOGUE_TRACE] rank=%s armed path=pool: members=%s segments=%s "
            "chunks=%s probe_token=%s",
            self.rank,
            len(wave.uuids),
            len(wave.plan.segments),
            len(wave.plan.chunks),
            tracer.probe_token,
        )
        return tracer

    def _record_unsharded_epilogue_gather(
        self, worker, tracer, selected, *, num_global_rows
    ) -> None:
        """Hand-write the gather record for a path with NO sharded gather.

        Both prefill paths have a branch where the parallel manager exposes no
        ``gather_prefill_last_token_hidden`` at all, so this rank holds every
        row and the gather-side facts are trivially ``tp=0/1, owned=all``. That
        is the ONLY record a caller may write itself, and
        ``record_unsharded_gather`` asserts the premise against the manager
        rather than taking the branch's word for it - the alternative is a
        trace that reports a broken multi-rank gather as healthy, which is the
        one failure an acceptance probe must never have.

        Where a gather DOES exist the tracer goes into it (``trace=``) and the
        record is written from inside, with the real shard and owned mask.
        """
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker

        from batchgen.prefix_reuse.epilogue_trace import record_unsharded_gather
        from batchgen.prefix_reuse.probe_guard import run_probe

        run_probe(
            "epilogue_trace.gather_unsharded",
            lambda: record_unsharded_gather(
                tracer,
                selected,
                num_global_rows=num_global_rows,
                manager=self.parallel_manager,
            ),
            rank=self.rank,
        )

    def _new_pool_kernel_dump_recorder(self, worker, wave):
        """Server-level ``debug_state.prefix_pool_dump_kernel_args``.

        Same seam as ``prefix_pool_offload_check``: the key is read from the
        debug state the wave adopted. The value is the mode, ``"sync"`` or
        ``"nosync"`` - with ``sync`` the probe synchronizes before reading the
        pool back, so a corruption that only survives ``nosync`` is a
        write->attend ordering race. ``prefix_pool_dump_dir`` names the parent
        directory (default /dev/shm). Returns None when the batch did not ask.

        The flag check comes BEFORE the import, and the whole setup - mode
        parse, directory resolution, the recorder's own ``makedirs`` - is
        guarded: an undumpable directory must cost the dump, not the wave.
        """
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        debug = self._pool_probe_debug(wave)
        if debug is None:
            return None
        from batchgen.prefix_reuse.kernel_dump import (
            DEBUG_FLAG as _POOL_DUMP_FLAG,
            DUMP_DIR_FLAG as _POOL_DUMP_DIR_FLAG,
            PoolKernelDumpRecorder,
            dump_directory,
            parse_dump_mode,
        )
        from batchgen.prefix_reuse.probe_guard import run_probe

        mode = run_probe(
            "kernel_dump.setup",
            lambda: parse_dump_mode(debug.get(_POOL_DUMP_FLAG)),
            rank=self.rank,
        )
        if mode is None:
            return None
        directory = run_probe(
            "kernel_dump.setup",
            lambda: dump_directory(debug.get(_POOL_DUMP_DIR_FLAG), self.rank),
            rank=self.rank,
        )
        if directory is None:
            return None
        logging.info(
            "[PREFIX_POOL_KERNEL_DUMP] rank=%s mode=%s dir=%s members=%s "
            "segments=%s page_tokens=%s",
            self.rank,
            mode,
            directory,
            len(wave.uuids),
            len(wave.plan.segments),
            wave.page_tokens,
        )
        return run_probe(
            "kernel_dump.setup",
            lambda: PoolKernelDumpRecorder(
                rank=self.rank,
                mode=mode,
                directory=directory,
                page_tokens=wave.page_tokens,
                global_ids=list(wave.global_ids),
            ),
            rank=self.rank,
        )

    def _pool_pad_poison_enabled(
        self, ctx: PoolProbeContext, wave, *,
        _debug_flag_enabled: Callable[[object], bool],
    ) -> bool:
        """Server-level ``debug_state.prefix_pool_pad_poison``.

        Read through the same seam as ``prefix_pool_offload_check``: the key
        is part of the debug state the wave adopted, so an operator arms the
        probe against a server that is already up. False when it is not set, which
        leaves the packed page table byte-identical to the default.

        The flag check comes BEFORE the import: on a production wave
        ``_pool_probe_debug`` has already answered None.

        The probe read is a sibling with the SAME context, so it is called
        with this one instead of through the worker delegator.
        """
        debug = self._pool_probe_debug(ctx, wave)
        if debug is None:
            return False
        from batchgen.prefix_reuse.executor import PAD_POISON_DEBUG_FLAG

        if not _debug_flag_enabled(debug.get(PAD_POISON_DEBUG_FLAG)):
            return False
        logging.info(
            "[PREFIX_POOL_PAD_POISON] rank=%s padding each page-table row "
            "with its own last page: members=%s segments=%s",
            ctx.rank,
            len(wave.uuids),
            len(wave.plan.segments),
        )
        return True

    def _new_pool_offload_check_recorder(self, worker, wave):
        """Server-level ``debug_state.prefix_pool_offload_check``.

        The key is part of the debug state the wave adopted, so an operator
        arms the check against a server that is already up and no env var
        exists for it. Returns None
        when the batch did not ask, which is what keeps the check free.

        The flag check comes BEFORE the import: on a production wave
        ``_pool_probe_debug`` has already answered None.
        """
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        debug = self._pool_probe_debug(wave)
        if debug is None:
            return None
        from batchgen.prefix_reuse.offload_check import (
            DEBUG_FLAG as _POOL_OFFLOAD_CHECK_FLAG,
            PoolOffloadCheckRecorder,
        )

        if not self._debug_flag_enabled(debug.get(_POOL_OFFLOAD_CHECK_FLAG)):
            return None
        logging.info(
            "[PREFIX_POOL_OFFLOAD_CHECK] rank=%s arming byte-equality check: "
            "members=%s segments=%s page_tokens=%s",
            self.rank,
            len(wave.uuids),
            len(wave.plan.segments),
            wave.page_tokens,
        )
        from batchgen.prefix_reuse.probe_guard import run_probe

        return run_probe(
            "offload_check.setup",
            lambda: PoolOffloadCheckRecorder(
                rank=self.rank, page_tokens=wave.page_tokens
            ),
            rank=self.rank,
        )

    def _finish_prefill_pool_wave(self, worker) -> None:
        """After every D2H retired: publish the segments, pin members to them.

        The Host pages and the prefix-cache nodes are the serve group's leader's,
        so only the leader commits and attaches. A follower computed rows into
        the leader's pages and has nothing to publish: it drops the page-table
        record it mirrored for the pseudo sequences and keeps the mirrored prefix
        state. The leader's member refcount is what keeps those pages alive.
        """
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        wave = self._prefill_pool_wave
        if wave is None:
            return
        self._prefill_pool_wave = None
        if not all(
            self._owns_host_kv(self.global_batch.get_sequence(uuid))
            for uuid in wave.uuids
        ):
            for view in self._host_views_by_group().values():
                view.unregister_sequences(list(wave.segment_sids.values()))
            return
        from batchgen.prefix_reuse.wave_runtime import (
            attach_pool_members,
            commit_pool_segments,
        )

        runtime = self.prefix_cache_runtime_config
        try:
            segment_handles = commit_pool_segments(
                wave,
                coordinator=self.prefix_cache_coordinator,
                worker_views_by_group=self._host_views_by_group(),
                namespace_digest=runtime.namespace_digest,
                publish_boundary_tokens=runtime.publish_boundary_tokens,
                max_scan_nodes=runtime.max_nodes,
            )
            try:
                member_handles = attach_pool_members(
                    wave,
                    coordinator=self.prefix_cache_coordinator,
                    namespace_digest=runtime.namespace_digest,
                )
            finally:
                for handle in segment_handles:
                    self.prefix_cache_coordinator.release_attachment(handle)
            for index, handle in member_handles.items():
                global_id = wave.global_ids[index]
                self._prefix.append_commit_handle(global_id, handle)
        finally:
            # Success or failure, the wave object is done. Segment Host pages
            # stay tracked in _pool_sid_members/_pool_member_sids, so after a
            # failed commit/attach the per-member release lane still frees and
            # unregisters every pseudo-sid.
            self._prefill_pool_wave = None


    def prefill(self, worker, batch: list[int]):
        """
        Handle the prefill for a batch.
        batch: list of local indices
        """
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        # Bind the attention wrappers' host_paged_kv_worker_view_aux BEFORE the decoder
        # loop. Without this binding, GLM-5's prefill indexer-K offload at
        # wrappers.py:_offload_prepacked_indexer_kv silently early-returns
        # (host_paged_kv_worker_view_aux is None), so the aux cache is never
        # populated for prompt tokens and any later decode past 2048 tokens
        # reads unwritten aux pages.
        # Prefill offloads KV directly to host via host_paged_kv_worker_view_aux;
        # it does NOT use the GPU paged KV manager. Binding host_*_aux here
        # ensures `_offload_prepacked_indexer_kv` actually pushes indexer K to
        # the host aux cache instead of early-returning on a None view.
        self._bind_prefill_host_views()

        if self._prefill_deepseek_family_name() in self.model_config.model_type:
            self.model.model._use_flash_attention_2 = False

        # Dynamic padding: find max length within THIS batch, not global max
        # This is critical for long-tailed distributions
        batch_seq_lengths = []
        for query_idx in batch:
            uuid = self._local_to_uuid_map[query_idx]
            seq = self.global_batch.get_sequence(uuid)
            batch_seq_lengths.append(int(seq.prompt_length))
        batch_max_len = max(batch_seq_lengths)

        # Pad each sequence to batch_max_len and construct attention masks on-the-fly
        padded_input_ids = []
        padded_attention_masks = []
        for query_idx in batch:
            uuid = self._local_to_uuid_map[query_idx]
            seq = self.global_batch.get_sequence(uuid)
            prompt_len = seq.prompt_length
            seq_input_ids = self._trajectory_tokens(
                seq, length=prompt_len, dtype=torch.int64
            ).view(1, -1)
            seq_len = seq_input_ids.shape[1]

            # Construct attention mask from prompt_length (1s for valid tokens, 0s for padding)
            seq_attention_mask = torch.zeros((1, seq_len), dtype=torch.int64)
            seq_attention_mask[0, :prompt_len] = 1

            if seq_len < batch_max_len:
                # Pad with zeros (left-aligned tokens, right-padded)
                pad_len = batch_max_len - seq_len
                seq_input_ids = torch.cat([
                    seq_input_ids,
                    torch.zeros((1, pad_len), dtype=seq_input_ids.dtype)
                ], dim=1)
                seq_attention_mask = torch.cat([
                    seq_attention_mask,
                    torch.zeros((1, pad_len), dtype=seq_attention_mask.dtype)
                ], dim=1)

            padded_input_ids.append(seq_input_ids)
            padded_attention_masks.append(seq_attention_mask)

        input_ids = torch.cat(padded_input_ids, dim=0)
        attention_masks = torch.cat(padded_attention_masks, dim=0)

        num_prefill_micro_batches = math.ceil(
            len(batch) / self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size
        )
        prefill_micro_batch_input_ids = torch.split(
            input_ids,
            self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size,
        )
        prefill_micro_batch_attention_masks = torch.split(
            attention_masks,
            self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size,
        )
        if self.rank == 0:
            logging.info(f"Number of prefill micro batches: {num_prefill_micro_batches}")

        cur_batch_start = 0
        output_tokens = []

        with self._weight_passes.scope("prefill"):
            for micro_batch_idx in tqdm(range(num_prefill_micro_batches), desc="Prefill Micro Batch"):
                # Feed watchdog during long prefill operations
                self.feed_watchdog()
                self._weight_passes.begin(
                    carry=micro_batch_idx + 1 < num_prefill_micro_batches
                )

                with torch.inference_mode():
                    self._bind_prefill_step_metadata(
                        prefill_micro_batch_attention_masks[micro_batch_idx]
                    )

                    cur_batch_size = prefill_micro_batch_input_ids[micro_batch_idx].shape[0]
                    cur_batch_local = batch[cur_batch_start : cur_batch_start + cur_batch_size]

                    # Pass local indices - the C++ layer handles rank offset internally
                    self._bind_prefill_step_batch(cur_batch_local)

                    cur_batch_start += cur_batch_size
                    assert len(cur_batch_local) == cur_batch_size

                    outputs = self.model(
                        prefill_micro_batch_input_ids[micro_batch_idx].to(self.torch_device, dtype=torch.int64),
                        attention_mask=prefill_micro_batch_attention_masks[micro_batch_idx].to(self.torch_device),
                        use_cache=False,
                    )
                    cur_batch_sequences = [
                        self.global_batch.get_sequence(self._local_to_uuid_map[local_idx])
                        for local_idx in cur_batch_local
                    ]
                    new_tokens = self._select_tokens(outputs.logits[:, -1, :], cur_batch_sequences)
                    self._weight_passes.end()
                    output_tokens.append(new_tokens)

        new_tokens = torch.cat(output_tokens, dim=0)

        # Update sequence state after prefill
        # For evicted re-entry: first new token goes at decoded_length offset (not 0)
        # For fresh sequences: decoded_length is 0, so offset is 0 (same as before)
        new_tokens_cpu = new_tokens.cpu()
        self._append_trajectory_batch(batch, new_tokens_cpu)
        for i, local_idx in enumerate(batch):
            uuid = self._local_to_uuid_map[local_idx]
            seq = self.global_batch.get_sequence(uuid)
            seq.advance_one_token()

            # MODIFIED: Check for EOS respecting ignore_eos flag
            if self._should_stop_at_eos(new_tokens_cpu[i].item(), seq):
                seq.eos_reached = True

        return new_tokens

    def _prefill_model_pass(
        self,
        worker,
        input_ids: torch.Tensor,
        tail_rows,
        *,
        lmhead_fp32: bool,
        trace=None,
        layer_probe=None,
    ):
        """One prefill model pass: embed -> layers -> depth-mix -> norm ->
        tail gather -> lm_head.

        The ONE model loop both prefill drivers run (step 4 of the three-step
        extraction): ``prefill_prepacked`` drives it once per prepacked
        micro-batch and ``_prefill_prepacked_pool`` once per pooled chunk.
        The drivers keep their own batch assembly, worker->model state
        binding (``_install_prefill_prepack_metadata`` /
        ``_install_prefill_pool_rows``), token selection, trace emission and
        metrics; this pass owns everything between the bound state and the
        logits, and nothing else drives decoder layers for prefill.

        Args:
            input_ids: flat ``[total_tokens]`` int64 token ids, already on
                the device the model runs on.
            tail_rows: index tensor of the rows whose logits the driver
                decodes (one per sequence that ends in this pass), or
                ``None`` for a pooled chunk that computes shared segments
                only -- the layers still run (their KV offloads are the
                product) and the pass returns ``None``.
            lmhead_fp32: the driver's one read of the lm_head FP32 debug
                key. The pooled driver has never read that key and passes
                ``False``; this step is behaviour-preserving, so the
                asymmetry is kept at the call sites, not resolved here.
            trace: epilogue tracer, or ``None``. The gather-side record is
                written from inside the real gather (``trace=``) or by the
                asserted unsharded helper; the drivers keep chunk begin and
                row emission.
            layer_probe: optional observation callback
                ``(layer_idx, decoder_layer, block_residual) -> None`` run
                after every layer; the prepacked driver's env-gated memory
                profile. Observation only -- it must not touch the tensors.
        """
        # The moved statements are kept VERBATIM from the two former inline
        # loops where they agreed: source pins depend on the ``self.``
        # spellings. Rebinding self to the worker preserves those spellings;
        # this engine instance remains stateless.
        self = worker

        # The unsqueeze is a VIEW, so binding only ``hidden_states`` keeps
        # the embedding storage alive exactly as long as layer 0 needs it; a
        # second local pinning that storage for the whole stack cost 1.75 GiB
        # dead across layers 1-92 at S=131,072 / H=7168 / bf16
        # (batchgen_design/model_support/kimi_k3/PREFILL_MEMORY_AUDIT.md
        # section 7, fix 4).
        hidden_states = self.model.model.embed_tokens(input_ids).unsqueeze(0)

        # Block Attention Residuals (Kimi-K3): the depth-mix REPLACES the
        # classic residual body, so the per-layer state has to be carried
        # here. Without it every layer sees a zero-width residual and the
        # model runs happily while computing something that is not K3 --
        # wrong text, no error. Mirrors KimiLinearModel.forward
        # (kimi_linear/model.py:880-910), which is the eager reference.
        use_attn_res = getattr(self.model.model, "use_attn_residuals", False)
        block_residual = None
        if use_attn_res:
            # Zero-column view of a buffer preallocated for ALL the
            # stack's block boundaries, so the per-boundary `cat` never
            # holds the (S,nb,H) and (S,nb+1,H) tensors at once (12.25
            # GiB at K3's last boundary; PREFILL_MEMORY_AUDIT.md fix 3).
            block_residual = self.model.model._new_block_residual(hidden_states)

        for layer_idx, decoder_layer in enumerate(self.model.model.layers):
            if use_attn_res:
                hidden_states, block_residual = decoder_layer(
                    hidden_states,
                    attention_mask=None,
                    position_ids=None,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                    block_residual=block_residual,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=None,
                    position_ids=None,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                )
                hidden_states = layer_outputs[0]
            if layer_probe is not None:
                layer_probe(layer_idx, decoder_layer, block_residual)

        # Output depth-mix, then norm -- that ORDER is load-bearing
        # (kimi_linear/model.py:904-913). A pooled chunk with no tail rows
        # skips the mix: nothing downstream consumes the mixed hidden state.
        if use_attn_res and tail_rows is not None:
            hidden_dim = hidden_states.shape[2]
            batch_sz, seq_sz = hidden_states.shape[:2]
            hidden_states = self.model.model._apply_output_attn_res(
                hidden_states.view(-1, hidden_dim), block_residual
            ).view(batch_sz, seq_sz, hidden_dim)
        if use_attn_res:
            # The pass is over for the residual: the buffer behind the view
            # is model class state and outlives the local, so it is released
            # here rather than left pinned until the next seed
            # (PREFILL_MEMORY_AUDIT.md fix 3).
            block_residual = None
            self.model.model._release_block_residual()

        if tail_rows is None:
            return None

        gather_last_token_hidden = getattr(
            self.parallel_manager,
            "gather_prefill_last_token_hidden",
            None,
        )
        if gather_last_token_hidden is None:
            # Final norm over the tail rows only: RMSNorm is per-row, so
            # norming the gathered rows equals gathering the normed rows.
            last_token_hidden = self.model.model.norm(
                hidden_states[:, tail_rows, :]
            )[0]
            if trace is not None:
                # Genuinely single-rank: this manager has no gather, so
                # the rank holds every row. Asserted inside, not
                # assumed - see _record_unsharded_epilogue_gather.
                self._record_unsharded_epilogue_gather(
                    trace,
                    last_token_hidden,
                    num_global_rows=int(input_ids.numel()),
                )
        else:
            # The tracer goes INTO the real gather: under streamed-SP8 this
            # rank holds only a row shard, and the shard bounds, the owned
            # mask and the norms either side of the all-reduce are visible
            # only in there. Writing a tp=0/1 record out here instead would
            # report a broken multi-rank gather as healthy.
            last_token_hidden = gather_last_token_hidden(
                self.model.model.norm(hidden_states),
                tail_rows,
                int(input_ids.numel()),
                **(
                    {"trace": trace}
                    if trace is not None
                    else {}
                ),
            )

        # lm_head matmul: BF16 by default (matches HF / SGLang / vLLM).
        # The FP32 cast rides the driver's debug-state read (a debug-state
        # key; no env guard).
        if getattr(self.model.lm_head, "tp_size", 1) > 1:
            # vocab-parallel head (K3 TP8): the module all_gathers the
            # full logits; the raw shard weight must not be used here.
            # .forward skips the decode-only pre-hook (``[:, -1, :]``
            # expects [bsz, seq, H]; this input is the gathered [n, H]).
            logits = self.model.lm_head.forward(last_token_hidden).float()
        elif lmhead_fp32:
            logits = torch.nn.functional.linear(
                last_token_hidden.float(),
                self.model.lm_head.weight.float(),
                self.model.lm_head.bias.float() if hasattr(self.model.lm_head, 'bias') and self.model.lm_head.bias is not None else None
            )
        else:
            logits = torch.nn.functional.linear(
                last_token_hidden,
                self.model.lm_head.weight,
                self.model.lm_head.bias if hasattr(self.model.lm_head, 'bias') and self.model.lm_head.bias is not None else None
            ).float()
        return logits

    def prefill_prepacked(self, worker, batch: list[int], *, token_cap: int):
        """
        Handle prefill for a batch using prepack optimization.

        Prepack combines multiple shorter sequences into rows to minimize padding waste,
        which is especially beneficial for MLP/MoE layers.

        Args:
            batch: list of local indices
        """
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        _engine = self
        self = worker
        # Bind the attention wrappers' host_paged_kv_worker_view_aux BEFORE the decoder
        # loop. Without this binding, GLM-5's prefill indexer-K offload at
        # wrappers.py:_offload_prepacked_indexer_kv silently early-returns
        # (host_paged_kv_worker_view_aux is None), so the aux cache is never
        # populated for prompt tokens and any later decode past 2048 tokens
        # reads unwritten aux pages.
        # Prefill offloads KV directly to host via host_paged_kv_worker_view_aux;
        # it does NOT use the GPU paged KV manager. Binding host_*_aux here
        # ensures `_offload_prepacked_indexer_kv` actually pushes indexer K to
        # the host aux cache instead of early-returning on a None view.
        self._bind_prefill_host_views()

        if self._prefill_pool_wave is not None:
            with self._weight_passes.scope("prefill"):
                return self._prefill_prepacked_pool(batch)

        if self._prefill_deepseek_family_name() in self.model_config.model_type:
            self.model.model._use_flash_attention_2 = False

        # Collect input_ids and attention_masks as lists for prepacking
        input_ids_list = []
        attention_mask_list = []
        seq_lengths = []
        full_seq_lengths = []
        compute_cached_tokens = []
        attached_tokens = []
        prefix_lookup_results = []

        for query_idx in batch:
            uuid = self._local_to_uuid_map[query_idx]
            seq = self.global_batch.get_sequence(uuid)
            # NO truncation: every prompt is tokenized to its OWN length.
            # An earlier `[:, :self.max_input_length]` slice silently dropped
            # the tail of long LongBench prompts when max_input_length was
            # carried over from a smaller earlier admit batch, causing the
            # model to "continue" mid-sentence instead of answering. Bind
            # everything to seq.prompt_length directly.
            L = seq.prompt_length
            prefix_state = self._prefix.prefix_sequence_states.get(seq.global_idx)
            compute_cached = (
                prefix_state.compute_cached_tokens
                if prefix_state is not None
                else 0
            )
            attached = (
                prefix_state.attached_tokens
                if prefix_state is not None
                else 0
            )
            query_len = L - compute_cached
            if query_len <= 0:
                raise RuntimeError(
                    "prefix prefill must compute at least one prompt token"
                )
            # Suffix-only prefill: materialize just the uncached span of the
            # trajectory; the cached prefix is attached as resident paged KV.
            input_ids = self._trajectory_tokens(
                seq, start=compute_cached, length=query_len, dtype=torch.int64
            ).view(1, -1)
            seq_lengths.append(query_len)
            full_seq_lengths.append(L)
            compute_cached_tokens.append(compute_cached)
            attached_tokens.append(attached)
            prefix_lookup_results.append(
                prefix_state.lookup_result if prefix_state is not None else None
            )

            # Per-seq mask marks the L valid positions for the prepacker.
            # Causal attention is enforced by FA varlen + cu_seqlens.
            attention_mask = torch.zeros_like(input_ids, dtype=torch.int64)
            attention_mask[0, :query_len] = 1

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)

        # Prepack sequences
        # Row capacity is set by planner in config (None = no limit, use max sequence length)
        row_capacity = self.engine_config.Module_Batching_Config.prepack_row_capacity
        prepack_meta = prepack_sequences(
            input_ids_list,
            attention_mask_list,
            row_capacity=row_capacity,
            device=self.torch_device,
        )

        # Log prepack statistics
        if self.rank == 0:
            stats = get_prepack_stats(prepack_meta)
            logging.info(
                f"Prepack stats: {stats['num_sequences']} seqs -> {stats['num_packed_rows']} rows, "
                f"padding saved: {stats['padding_saved']} tokens, "
                f"efficiency: {stats['packing_efficiency']:.2%}"
            )

        # Create flattened tensors for prepacked forward
        # Flatten packed_input_ids to [total_tokens]
        total_tokens = sum(prepack_meta.original_seq_lengths)

        # Extract only valid tokens (non-padding) in order
        packed_input_ids_flat = []
        packed_position_ids_flat = []

        for seq_idx in range(prepack_meta.num_original_sequences):
            row_idx, start_pos = prepack_meta.pack_assignment[seq_idx]
            seq_len = prepack_meta.original_seq_lengths[seq_idx]

            # Extract tokens for this sequence
            seq_input_ids = prepack_meta.packed_input_ids[row_idx, start_pos:start_pos + seq_len]
            packed_input_ids_flat.append(seq_input_ids)

            # Prefix-hit suffixes retain their original absolute positions.
            position_start = compute_cached_tokens[seq_idx]
            packed_position_ids_flat.append(torch.arange(
                position_start,
                position_start + seq_len,
                device=self.torch_device,
            ))

        packed_input_ids_flat = torch.cat(packed_input_ids_flat, dim=0)  # [total_tokens]
        packed_position_ids_flat = torch.cat(packed_position_ids_flat, dim=0)  # [total_tokens]

        # Split sequences into micro-batches based on TOKEN count (not sequence count)
        # This prevents OOM when sequences have varying lengths
        # The planner sets the cap per prefill mode; the server's mode picks
        # it, and the worker's delegator hands it over explicitly.
        MAX_TOKENS_PER_MICRO_BATCH = token_cap
        num_sequences = prepack_meta.num_original_sequences
        seq_lengths_list = prepack_meta.original_seq_lengths

        # Create micro-batches bounded by token count, optionally also by sum(L^2)
        # so the per-microbatch attention work (which is O(L^2)) doesn't pile up
        # on one micro-batch when a single very long sequence is present.
        # The planner's token cap is a bound between sequence boundaries.  A
        # single K3 prompt may itself be much larger than that cap, so the
        # ordinary greedy planner cannot split it and would still co-reside two
        # 262K-token sequences in one decoder pass.  That pass allocates the
        # block-attention-residual scratch for *all* rows and exceeded H20 HBM
        # before the first streamed-SP8 expert layer.  K3's KDA/MLA state is
        # persistent per sequence, so keeping each long sequence in its own
        # prepack pass preserves the state contract; it is not a token-axis
        # split pretending to be a resumable model forward.
        micro_batches, l2_cap, _use_single_sequence_mb = (
            self._plan_prefill_micro_batches(
                seq_lengths_list, full_seq_lengths, compute_cached_tokens
            )
        )
        total_tokens_all = sum(seq_lengths_list)
        if (
            hasattr(self.parallel_manager, "prefill_uses_resident_ep")
            and self.parallel_manager.prefill_uses_resident_ep()
        ):
            # Pass counts were aligned across ALL ranks before this call (a
            # rank with no local rows never enters here, so no collective may
            # live in this function); ranks short of the global count pad with
            # collective-only passes afterwards.
            global_passes = getattr(self, "_prefill_global_passes", None)
            if global_passes is not None and len(micro_batches) > global_passes:
                raise RuntimeError(
                    "resident-EP prefill planned more micro-batches than the "
                    f"aligned global count: {len(micro_batches)} > {global_passes}"
                )

        if self.rank == 0:
            logging.info(
                f"Prepacked prefill: {len(micro_batches)} micro batches, "
                f"{total_tokens_all:,} total tokens, max {MAX_TOKENS_PER_MICRO_BATCH:,} tokens/batch"
                + (", single-sequence long-prompt guard" if _use_single_sequence_mb else "")
                + (f", l2_cap={l2_cap:,}" if l2_cap > 0 else "")
            )

        output_tokens = []
        prefill_sequences = [
            self.global_batch.get_sequence(self._local_to_uuid_map[local_idx])
            for local_idx in batch
        ]
        prefill_debug = self._phase_prefill_debug
        _k3_profile_enabled = self._debug_flag_enabled(
            prefill_debug.get(self._k3_prefill_profile_flag_name())
        )
        _k3_profile_logits = []
        # Server-level ``debug_state.glm5_lmhead_fp32``: read once here,
        # like ``k3_prefill_profile`` above, so the per-micro-batch lm_head
        # epilogue below costs one local bool test.
        _lmhead_fp32 = self._debug_flag_enabled(
            prefill_debug.get(self._glm5_lmhead_fp32_flag_name())
        )
        # Batch-level ``prefix_pool_epilogue_trace`` on the UNPOOLED path: the
        # same keys as the pooled wave, so one armed batch traces whichever
        # prefill runs (the first batch on a fresh server, or reuse OFF).
        # None in production: ``prefill_debug`` is read once above for
        # ``k3_prefill_profile``, and this one membership test on it is the
        # whole cost of the trace being off - no tracer, no import.
        unpooled_trace = None
        if _EPILOGUE_TRACE_DEBUG_FLAG in prefill_debug:
            from batchgen.prefix_reuse.epilogue_trace import tracer_from_debug
            from batchgen.prefix_reuse.probe_guard import run_probe

            unpooled_trace = run_probe(
                "epilogue_trace.setup",
                lambda: tracer_from_debug(
                    prefill_debug, rank=self.rank, path="unpooled"
                ),
                rank=self.rank,
            )
        if unpooled_trace is not None:
            logging.info(
                "[EPILOGUE_TRACE] rank=%s armed path=unpooled: sequences=%s "
                "micro_batches=%s probe_token=%s",
                self.rank, len(prefill_sequences), len(micro_batches),
                unpooled_trace.probe_token,
            )

        # Pure forward wall time: started here so configure_prefill (already
        # reported separately as `Config completed`) is NEVER folded in.
        _prefill_forward_t0 = time.perf_counter()
        with torch.inference_mode(), self._weight_passes.scope("prefill"):
            for batch_idx, (seq_start, seq_end) in tqdm(
                enumerate(micro_batches),
                total=len(micro_batches),
                desc="Prepacked Prefill",
                disable=(self.rank != 0)  # Only show progress on rank 0
            ):
                # Feed watchdog during long prefill operations
                self.feed_watchdog()
                self._weight_passes.begin(carry=batch_idx + 1 < len(micro_batches))

                # Get sequences for this micro-batch
                batch_seq_lengths = seq_lengths_list[seq_start:seq_end]
                batch_full_seq_lengths = full_seq_lengths[seq_start:seq_end]
                batch_compute_cached = compute_cached_tokens[seq_start:seq_end]
                batch_attached_tokens = attached_tokens[seq_start:seq_end]
                batch_lookup_results = prefix_lookup_results[seq_start:seq_end]
                batch_num_seqs = seq_end - seq_start

                # Extract tokens for this micro-batch
                batch_input_ids = []
                batch_position_ids = []
                token_offset = sum(seq_lengths_list[:seq_start])  # Offset into flat tensors

                for seq_idx in range(seq_start, seq_end):
                    seq_len = seq_lengths_list[seq_idx]
                    # Calculate where this sequence's tokens are in the flat tensor
                    seq_token_start = sum(seq_lengths_list[:seq_idx])
                    seq_token_end = seq_token_start + seq_len

                    batch_input_ids.append(packed_input_ids_flat[seq_token_start:seq_token_end])
                    batch_position_ids.append(packed_position_ids_flat[seq_token_start:seq_token_end])

                batch_input_ids_flat = torch.cat(batch_input_ids, dim=0)
                batch_position_ids_flat = torch.cat(batch_position_ids, dim=0)
                if (
                    hasattr(self.parallel_manager, "prefill_uses_resident_ep")
                    and self.parallel_manager.prefill_uses_resident_ep()
                ):
                    self._sync_prefill_moe_rank_counts(
                        int(batch_input_ids_flat.numel()),
                        reason=f"prefill_microbatch_{batch_idx}",
                    )

                batch_local_indices = batch[seq_start:seq_end]
                local_to_global_seq_id_map = {}
                for local_idx in batch_local_indices:
                    uuid = self._local_to_uuid_map.get(local_idx)
                    if uuid is None:
                        raise RuntimeError(
                            f"Rank {self.rank}: missing UUID for prefill local_idx={local_idx}"
                        )
                    seq = self.global_batch.get_sequence(uuid)
                    if seq is None:
                        raise RuntimeError(
                            f"Rank {self.rank}: missing SequenceEntry for prefill uuid={uuid[:8]}"
                        )
                    local_to_global_seq_id_map[local_idx] = seq.global_idx

                batch_spans = build_prefill_sequence_spans(
                    batch_local_indices,
                    batch_seq_lengths,
                    self._local_to_uuid_map,
                    local_to_global_seq_id_map,
                )
                # Pinned source + non_blocking: a pageable H2D here blocks the
                # host until the copy lands, i.e. once per micro-batch.
                batch_cu_seqlens = torch.tensor(
                    prefill_sequence_spans_to_cu_seqlens(batch_spans),
                    dtype=torch.int32,
                ).pin_memory().to(self.torch_device, non_blocking=True)
                batch_max_seqlen = max(batch_seq_lengths)
                prefix_materialization = None
                if any(tokens > 0 for tokens in batch_compute_cached):
                    if any(result is None for result in batch_lookup_results):
                        raise RuntimeError(
                            "prefix-hit microbatch is missing lookup results"
                        )
                    from batchgen.prefix_reuse.materialization import (
                        materialize_gpt_oss_prefixes,
                    )

                    host_page_tokens = int(
                        self.prefix_cache_runtime_config.group_specs[0]
                        .raw_page_tokens
                    )
                    # Sized by the same function the micro-batch planner
                    # capped against the prefill-phase prefix KV budget.
                    gpu_config = self._prefix_materialization_kv_config(
                        batch_full_seq_lengths
                    )
                    temporary_manager = self._new_prefill_temporary_gpu_kv_manager(gpu_config)
                    temporary_manager.initialize()
                    prefix_materialization = materialize_gpt_oss_prefixes(
                        gpu_manager=temporary_manager,
                        host_worker_view=(
                            self.core_engine.host_paged_kv_worker_view
                        ),
                        coordinator=self.prefix_cache_coordinator,
                        lookup_results=batch_lookup_results,
                        sequence_ids=(
                            prefill_sequence_spans_to_global_seq_ids(batch_spans)
                        ),
                        prompt_lengths=batch_full_seq_lengths,
                        compute_cached_tokens=batch_compute_cached,
                        raw_page_tokens=host_page_tokens,
                        collect_metrics=self.prefix_cache_debug_stats,
                        host_page_bytes_all_layers=(
                            self.prefix_cache_runtime_config
                            .host_page_bytes_all_layers
                        ),
                    )
                    self._prefix.begin_materialization(prefix_materialization)

                # Set up the attention wrappers for this micro-batch.
                # These class attrs are the per-step worker->model contract read
                # by attention wrappers; see semantics in
                #   batchgen-context/architecture/PSM_WORKER_CONTRACT.md (§2)
                self._install_prefill_prepack_metadata(
                    batch_cu_seqlens,
                    batch_max_seqlen,
                    batch_num_seqs,
                    batch_seq_lengths,
                    batch_position_ids_flat,
                    batch_spans,
                    prefix_materialization,
                    batch_full_seq_lengths,
                    batch_compute_cached,
                    batch_attached_tokens,
                )

                # DEBUG (env-gated): per-layer HBM watermarks. Cheap (allocator
                # bookkeeping is CPU-side, no sync) and independent of the
                # allocator-history trace, so it still localises the peak layer
                # if the trace ring buffer wraps.
                _memprof_layers = os.environ.get("BATCHGEN_MEM_PROFILE", "0") == "1"
                # The allocator-history ring buffer only holds the LAST max_entries
                # events, and one K3 prefill emits ~2e7 of them (896-expert
                # moe_infer loop x 92 layers), so a whole-forward trace only ever
                # retains the tail. The HBM peak is set in the FIRST layer, so
                # BATCHGEN_MEM_PROFILE_STOP_LAYER=N dumps and stops recording right
                # after layer N: a short, wrap-free trace of the window that
                # actually contains the peak.
                _memprof_stop_layer = int(os.environ.get("BATCHGEN_MEM_PROFILE_STOP_LAYER", "-1"))
                _memprof_probe = None
                if _memprof_layers:
                    def _memprof_probe(layer_idx, decoder_layer, block_residual,
                                       _stop_layer=_memprof_stop_layer):
                        logging.info(
                            f"[MEMPROF-L] rank={self.rank} layer={layer_idx} "
                            f"attn={type(getattr(decoder_layer, 'self_attn', decoder_layer)).__name__} "
                            f"alloc={torch.cuda.memory_allocated()/2**30:.3f}GiB "
                            f"cum_peak={torch.cuda.max_memory_allocated()/2**30:.3f}GiB "
                            f"reserved={torch.cuda.memory_reserved()/2**30:.3f}GiB "
                            f"bres={tuple(block_residual.shape) if block_residual is not None else None}"
                        )
                        if layer_idx == _stop_layer:
                            _p = os.path.join(
                                os.environ.get("BATCHGEN_MEM_PROFILE_DIR", "/tmp"),
                                f"memprof_rank{self.rank}_thru_layer{layer_idx}_{int(time.time())}.pickle",
                            )
                            torch.cuda.memory._dump_snapshot(_p)
                            torch.cuda.memory._record_memory_history(enabled=None)
                            logging.info(
                                f"[MEMPROF] Rank {self.rank}: stop-layer snapshot -> {_p} "
                                f"(peak_alloc so far={torch.cuda.max_memory_allocated()/2**30:.3f}GiB)"
                            )

                # Last-token row of each sequence in this micro-batch.
                last_token_indices = batch_cu_seqlens[1:] - 1
                if unpooled_trace is not None:
                    # Opens the chunk's gather-side record; the record itself
                    # is written inside the model pass (from the real gather,
                    # or by the asserted unsharded helper).
                    unpooled_trace.begin_chunk(batch_idx)
                # The ONE model loop (embed -> layers -> depth-mix -> norm ->
                # tail gather -> lm_head), shared with the pooled driver.
                logits = _engine._prefill_model_pass(
                    self,
                    batch_input_ids_flat.to(self.torch_device, dtype=torch.int64),
                    last_token_indices,
                    lmhead_fp32=_lmhead_fp32,
                    trace=unpooled_trace,
                    layer_probe=_memprof_probe,
                )
                if _k3_profile_enabled:
                    _k3_profile_logits.append(logits.detach())

                batch_sequences = [
                    self.global_batch.get_sequence(self._local_to_uuid_map[local_idx])
                    for local_idx in batch_local_indices
                ]
                batch_new_tokens = self._select_tokens(logits, batch_sequences)
                if batch_new_tokens.shape[0] != batch_num_seqs:
                    raise RuntimeError(
                        f"Rank {self.rank}: prefill token selection shape mismatch, "
                        f"got {batch_new_tokens.shape[0]} rows for {batch_num_seqs} sequences"
                    )
                if unpooled_trace is not None:
                    # One line per sequence, from the SAME logits the token
                    # choice consumed. The gather-side facts were recorded by
                    # the gather itself (or, where no gather exists, by the
                    # asserted single-rank helper), so nothing is hand-written
                    # here: the chunk was opened before the gather so those
                    # records survive to this line.
                    unpooled_trace.emit(
                        table_rows=list(range(batch_num_seqs)),
                        tail_rows=last_token_indices.tolist(),
                        wave_indices=list(range(batch_num_seqs)),
                        global_ids=[int(seq.global_idx) for seq in batch_sequences],
                        uuids=[str(seq.uuid) for seq in batch_sequences],
                        tokens=[int(token) for token in batch_new_tokens],
                        logits=logits,
                    )
                output_tokens.append(batch_new_tokens)
                self._weight_passes.end()

                if prefix_materialization is not None:
                    self._retire_prefill_pending_offloads(
                        device=self.torch_device,
                        reason="before releasing prefix materialization",
                    )
                    self._close_prefix_materialization(
                        prefix_materialization,
                        empty_cuda_cache=True,
                    )
                    self._prefix.end_materialization(prefix_materialization)
                    self._clear_prefill_prefix_materialization()


        # Keep the historical host-enqueue boundary for backwards-compatible
        # diagnostics, then establish two stronger completion boundaries. A host
        # perf-counter sample alone does not prove that a non-default CUDA stream
        # has finished, and the last layer's KV D2H task is intentionally not on
        # the compute stream.
        _prefill_forward_host_s = time.perf_counter() - _prefill_forward_t0
        torch.cuda.current_stream(self.torch_device).synchronize()
        _prefill_forward_gpu_complete_s = time.perf_counter() - _prefill_forward_t0
        # The FIRST generated tokens, straight out of prefill: a cross-check
        # of the response path (max_tokens=1 requests complete right after
        # prefill). Read once per wave after the completion boundary above,
        # never per micro-batch, so it adds no host-device sync. Rank 0 only;
        # ids only (the worker has no tokenizer -- decode them client-side).
        if self.rank == 0 and output_tokens:
            _ids = torch.cat([t.reshape(-1) for t in output_tokens]).tolist()
            logging.info(
                "[PREFILL] first sampled token ids: %s ... last: %s (n=%d)",
                _ids[:16], _ids[-8:], len(_ids))
        _num_final_kv_tasks = self._retire_prefill_pending_offloads(
            device=self.torch_device,
            reason="prefill metrics KV-ready boundary",
        )
        _prefill_kv_ready_s = time.perf_counter() - _prefill_forward_t0
        if _k3_profile_enabled:
            _k3_profile_topk = []
            for profile_logits in _k3_profile_logits:
                top_values, top_indices = torch.topk(
                    profile_logits,
                    k=min(8, profile_logits.shape[-1]),
                    dim=-1,
                )
                _k3_profile_topk.append({
                    "ids": top_indices.cpu().tolist(),
                    "values": top_values.cpu().tolist(),
                })
            self._emit_k3_prefill_profile_log(_k3_profile_topk)

        # Structured prefill record, one JSON line per rank that actually ran a
        # prefill (batchgen-benchmark docs/prefill_metrics_proposal.md). The
        # report tool prefers this over scraping the tqdm bar, which is
        # presentation, not an API.
        #
        # Deliberately NOT gated on rank 0, departing from the proposal's
        # "rank 0 only". prefill_prepacked runs only under
        # `if local_prefill_indices:`, and the tqdm bar above is
        # disable=(self.rank != 0) -- so when rank 0 owns none of the batch
        # there is neither a bar nor a rank-0 line anywhere in the log, and the
        # run reports `Prefill: 0.0s` with nothing to scrape. That is exactly
        # what the 131,069-token run produced when rank 2 owned the sequence.
        # Every participating rank emits its own tagged line; the wall time for
        # the batch is the MAX of each timing field over the emitting ranks.
        # `prefill_s` retains its historical host-forward meaning; schema-aware
        # reporters use `prefill_kv_ready_s` as the canonical capability boundary.
        # The legacy/full-attention path does not build a WavePrefixPlan, but it
        # still has the same workload geometry. Keep the stress metric contract
        # comparable: every prompt is recomputed, one item row is emitted per
        # sequence, and each FA3 micro-batch is one chunk. Causal pairs are the
        # exact full-attention work count for each prompt.
        legacy_full_attention_pairs = sum(
            (int(seq_len) * (int(seq_len) + 1)) // 2
            for seq_len in seq_lengths_list
        )
        logging.info("[METRICS] %s", json.dumps({
            "component": "prefill_full_attention",
            "phase": "prefill",
            "prefill_metrics_contract": "forward-host+gpu-complete+kv-ready-v1",
            "prefill_s": _prefill_forward_host_s,
            "prefill_forward_host_s": _prefill_forward_host_s,
            "prefill_forward_gpu_complete_s": _prefill_forward_gpu_complete_s,
            "prefill_kv_ready_s": _prefill_kv_ready_s,
            "final_kv_offload_tasks_retired": _num_final_kv_tasks,
            "sequences": num_sequences,
            "prompt_tokens": total_tokens_all,
            "computed_tokens": total_tokens_all,
            "full_attention_pairs": legacy_full_attention_pairs,
            "item_rows": num_sequences,
            "chunks": len(micro_batches),
            "saved_tokens": 0,
            "recomputed_tokens": total_tokens_all,
            "tokens_total": total_tokens_all,
            "seq_len_min": min(seq_lengths_list) if seq_lengths_list else 0,
            "seq_len_max": max(seq_lengths_list) if seq_lengths_list else 0,
            "micro_batches": len(micro_batches),
            "max_tokens_per_micro_batch": MAX_TOKENS_PER_MICRO_BATCH,
            "world_size": self.world_size,
            "rank": self.rank,
            # Guarded: the proposal's unguarded output_tokens[0] is an IndexError
            # on a rank that ran zero micro-batches.
            "first_sampled_token_ids": (
                output_tokens[0].reshape(-1).tolist()[:16] if output_tokens else []
            ),
        }, separators=(",", ":")))

        # Reset prepack mode
        self._reset_prefill_prepack_metadata()

        # Log timing summary for GPT-OSS if timing was enabled
        self._log_prefill_timing()

        new_tokens = torch.cat(output_tokens, dim=0)
        return self._write_prefill_tokens(batch, new_tokens)

    def _prefill_prepacked_pool(self, worker, batch: list[int]) -> torch.Tensor:
        """Prefill a planned wave: each pooled segment once, then every tail."""
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        _engine = self
        self = worker
        from batchgen.prefix_reuse.wave_plan import prefill_plan_work

        wave = self._prefill_pool_wave
        if wave.pool is None:
            self._construct_prefill_pool(wave)
        index_of = {uuid: index for index, uuid in enumerate(wave.uuids)}
        order = [index_of[self._local_to_uuid_map[local_idx]] for local_idx in batch]
        if sorted(order) != list(range(len(wave.uuids))):
            raise RuntimeError(
                f"Rank {self.rank}: prefill batch does not match the planned wave"
            )
        workspace = int(
            self.engine_config.Module_Batching_Config.prefill_prefix_pool_workspace_bytes
        )
        # The model seams prefill_prepacked also goes through (K3 block
        # residual + depth mix, streamed-SP8 row-shard gather, vocab-parallel
        # lm_head) live in the shared ``_prefill_model_pass``, duck-typed so
        # they collapse to the plain path.
        # Batch-level ``prefix_pool_epilogue_trace``: None in production.
        epilogue_trace = self._new_pool_epilogue_tracer(wave)
        if epilogue_trace is not None:
            # The wave boundary is the earliest point this seam can stamp: the
            # plan and the Host attachment happen in the reservation
            # transaction, before this method is entered.
            epilogue_trace.phase(
                "wave_begin",
                # Rows, so loaded segments do not count: they run none.
                rows=len(wave.computed_segments()) + len(wave.global_ids),
                tokens=wave.plan.computed_tokens,
                chunks=len(wave.plan.chunks),
            )
        tokens_by_index = {}
        executed = 0
        workspace_peak = 0
        reserved_peak = 0
        base_alloc, base_reserved, base_other = self._prefill_pool_base
        # As in prefill_prepacked: layers offload through these views, and a DSA
        # model writes its indexer keys through the aux one.
        self._bind_prefill_pool_segments(wave)
        # The pool row of an engine layer comes from the model's own pool
        # layout, never from a KV cache's layer mapping: a cache that keeps
        # every layer resolves to itself and would index past the pool.
        pool_layout = getattr(self.parallel_manager, "prefill_pool_layout", None)
        self._bind_prefill_pool_layer_state(wave, pool_layout)
        # Staging for the wave's loads; None unless it materializes something.
        materialized_loader = self._materialized_segment_loader(wave)
        materialized_tokens = 0
        offload_recorder_handed_off = False
        try:
            start = time.perf_counter()
            for chunk_index, chunk in enumerate(
                tqdm(wave.plan.chunks, desc="Pool Prefill", disable=(self.rank != 0))
            ):
                self.feed_watchdog()
                self._weight_passes.begin(
                    carry=chunk_index + 1 < len(wave.plan.chunks)
                )
                rows = wave.executor.build(chunk)
                if epilogue_trace is not None:
                    epilogue_trace.chunk_begin(
                        epilogue_trace.chunk + 1,
                        rows=len(rows.items),
                        tokens=int(rows.input_ids.numel()),
                    )
                host_ids, host_starts = wave.row_host_targets(rows)
                self._install_prefill_pool_rows(rows, host_ids, host_starts)
                # Before the forward, never during it: every layer's pool row
                # holds the committed prefix by the time any row attends over
                # it, and a failure here has not yet touched the forward.
                materialized_tokens += self._load_materialized_segments(
                    wave, rows, materialized_loader
                )
                # The ONE model loop (embed -> layers -> depth-mix -> norm ->
                # tail gather -> lm_head), shared with the prepacked driver.
                # A chunk with no tail rows computes pooled segments only:
                # the pass runs every layer (the KV offloads are the product)
                # and returns None. The pooled wave has never read the
                # unpooled driver's lm_head FP32 debug key; this step is
                # behaviour-preserving, so it keeps passing False.
                logits = _engine._prefill_model_pass(
                    self,
                    rows.input_ids,
                    rows.tail_rows if rows.tail_wave_indices else None,
                    lmhead_fp32=False,
                    trace=epilogue_trace,
                )
                if rows.tail_wave_indices:
                    selected = self._select_tokens(
                        logits,
                        [self.global_batch.get_sequence(wave.uuids[i]) for i in rows.tail_wave_indices],
                    )
                    for row, index in enumerate(rows.tail_wave_indices):
                        tokens_by_index[index] = selected[row]
                    if epilogue_trace is not None:
                        # The pairing side of the trace: the row -> wave index
                        # -> uuid -> global id chain, beside the gather-side
                        # facts the recorder already holds.
                        epilogue_trace.emit(
                            table_rows=[
                                row for row, item in enumerate(rows.items)
                                if item.kind == "tail"
                            ],
                            tail_rows=rows.tail_rows.tolist(),
                            wave_indices=list(rows.tail_wave_indices),
                            global_ids=list(wave.global_ids),
                            uuids=list(wave.uuids),
                            tokens=[int(token) for token in selected],
                            # The SAME tensor _select_tokens consumed: full
                            # vocabulary in both branches (the vocab-parallel
                            # head all-gathers before returning).
                            logits=logits,
                        )
                # The chunk's hidden state and the K3 block-residual buffer
                # were dropped/released inside the model pass before it
                # returned (PREFILL_MEMORY_AUDIT.md fix 3).
                self._weight_passes.end()
                executed += int(rows.input_ids.numel())
                wave.executor.finish(chunk)
                if epilogue_trace is not None:
                    epilogue_trace.chunk_end(
                        rows=len(rows.items),
                        tokens=int(rows.input_ids.numel()),
                    )
                # Allocated bytes are what a chunk needs. The allocator's reserve
                # grows into whatever the pool left free and drops cached blocks
                # only when a cudaMalloc fails, so it is reported, not checked.
                free_now, device_total = torch.cuda.mem_get_info(self.local_rank)
                reserved_now = torch.cuda.memory_reserved(self.local_rank)
                reserved_peak = max(
                    reserved_peak,
                    torch.cuda.max_memory_reserved(self.local_rank) - base_reserved
                    + (device_total - free_now - reserved_now) - base_other,
                )
                peak = torch.cuda.max_memory_allocated(self.local_rank) - base_alloc
                workspace_peak = max(workspace_peak, peak)
                if peak > workspace:
                    raise RuntimeError(
                        f"Rank {self.rank}: prefill workspace peaked at {peak} B, above the "
                        f"{workspace} B kept beside the prefix pool"
                    )
            prefill_s = time.perf_counter() - start
            if epilogue_trace is not None:
                # The last stamp a rank emits before it leaves the pooled
                # prefill: comparing wave_end across the two nodes is what
                # shows whether one group entered decode while the other was
                # still inside a pooled chunk.
                epilogue_trace.phase(
                    "wave_end",
                    elapsed_ms=prefill_s * 1e3,
                    tokens=executed,
                    chunks=len(wave.plan.chunks),
                )
            if executed != wave.plan.computed_tokens:
                raise RuntimeError(
                    f"Rank {self.rank}: pool prefill ran {executed} tokens, planned "
                    f"{wave.plan.computed_tokens}"
                )
            wave.executor.close()
            # The wave reached its end, so the offload recorder is handed to
            # _verify_prefill_pool_offload_bytes - which runs after the D2H has
            # retired, outside this method - and only it may clear the recorder.
            offload_recorder_handed_off = True
        finally:
            if wave.pool.free_pages != wave.pool.num_pages:
                logging.error("[PREFIX_POOL] Rank %s: pool released with pages still held", self.rank)
            # A segment's recurrent state is keyed by its pseudo sequence id and
            # has to live across the chunks that continue it, so it is released
            # here — the wave is over and no later batch carries those ids.
            self._release_kda_state_slots(list(wave.segment_sids.values()))
            if materialized_loader is not None:
                for group_loader in materialized_loader.values():
                    group_loader.release()
            # Probe recorder teardown, unconditional: a wave that died anywhere
            # above must not leave its records attached to a class attribute or
            # to a pool the next wave inherits. The dump recorder is finished
            # either way - it writes per chunk, nothing consumes it later - and
            # the offload recorder is dropped UNLESS this wave completed and
            # handed it to the post-D2H verification step.
            wave.pool.kernel_dump = None
            if not offload_recorder_handed_off:
                self._drop_prefill_pool_offload_recorder()
            wave.pool.k = wave.pool.v = wave.pool.indexer = None
            self._unbind_prefill_pool_segments()
            torch.cuda.empty_cache()
        work = prefill_plan_work(wave.plan)
        new_tokens = torch.stack([tokens_by_index[index] for index in order])
        # Ledger-contract fields the prepacked path already carries
        # (prefill_metrics_proposal.md): without seq_len_* a single-S run
        # cannot bind its FLOPs model, and without the sampled ids a pooled
        # wave has no determinism witness, so pooled waves were falling out
        # of prefill_report entirely. The wave is complete and the barrier
        # above has drained the compute stream, so the tolist() is a
        # wave-boundary scalar transfer, same as the prepacked emission.
        wave_seq_lens = [
            int(self.global_batch.get_sequence(uuid).prompt_length)
            for uuid in wave.uuids
        ]
        logging.info("[METRICS] " + json.dumps({
            "component": "prefix_pool",
            "phase": "prefill",
            "rank": self.rank,
            "sequences": len(order),
            "prompt_tokens": wave.plan.prompt_tokens,
            "computed_tokens": executed,
            "full_attention_pairs": work.full_attention_pairs,
            "item_rows": work.item_rows,
            "saved_tokens": wave.plan.prompt_tokens - executed,
            "materialized_tokens": materialized_tokens,
            "materialized_segments": len(wave.materialized),
            "segments": len(wave.plan.segments),
            "chunks": len(wave.plan.chunks),
            "threshold": wave.plan.threshold,
            "pool_pages": wave.pool.num_pages,
            "peak_pool_pages": wave.plan.peak_pool_pages,
            "workspace_peak_bytes": workspace_peak,
            "reserved_peak_bytes": reserved_peak,
            "prefill_s": prefill_s,
            "seq_len_min": min(wave_seq_lens) if wave_seq_lens else 0,
            "seq_len_max": max(wave_seq_lens) if wave_seq_lens else 0,
            "world_size": self.world_size,
            "first_sampled_token_ids": (
                new_tokens.reshape(-1).tolist()[:16]
            ),
        }, separators=(",", ":")))
        return self._write_prefill_tokens(batch, new_tokens)

    def _write_prefill_tokens(self, worker, batch: list[int], new_tokens: torch.Tensor) -> torch.Tensor:
        """Store each sequence's first generated token (batch order)."""
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        if new_tokens.shape[0] != len(batch):
            raise RuntimeError(
                f"Rank {self.rank}: prefill writeback shape mismatch, "
                f"got {new_tokens.shape[0]} rows for {len(batch)} local sequences"
            )

        # Update sequence state after prefill
        # For evicted re-entry: first new token goes at decoded_length offset (not 0)
        new_tokens_cpu = new_tokens.cpu()
        for i, local_idx in enumerate(batch):
            uuid = self._local_to_uuid_map[local_idx]
            seq = self.global_batch.get_sequence(uuid)
            self._append_trajectory_token(seq, int(new_tokens_cpu[i].item()))
            # The unified book owns the append position. Keep the scalar
            # completion counter in lockstep for fresh prefill and KV re-entry.
            seq.advance_one_token()

            # Check for EOS respecting ignore_eos flag
            if self._should_stop_at_eos(new_tokens_cpu[i].item(), seq):
                seq.eos_reached = True

        return new_tokens

    def _prepare_prefill_batch(self, worker) -> List[str]:
        """
        Select sequences for prefill based on HOST KV cache capacity.

        Key constraint: Host KV cache is PER NODE.
        - Each node has its own host KV capacity
        - Sequences assigned to ranks on node N use node N's host KV
        - Must check per-node capacity, not global

        With dynamic host KV reservation, sequences only need prompt + chunk_size
        pages initially (not the full kv_token_budget). This allows more sequences
        to be prefilled concurrently.

        EVICTED sequences get weighted priority (more decoded = higher priority)
        and re-enter through the prefill path.
        """
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        # Collect candidates: evicted sequences first (weighted priority), then new
        evicted_uuids = []
        if self.enable_host_kv_eviction:
            evicted_uuids = self.global_batch.get_sequences_by_status(SequenceStatus.EVICTED)
            # Weighted priority: more decoded tokens = higher priority (less wasted work)
            evicted_uuids.sort(key=lambda u: (
                -self.global_batch.get_sequence(u).total_decoded_before_eviction,
                self.global_batch.get_sequence(u).global_idx
            ))

        queueing_uuids = self.global_batch.get_sequences_by_status(SequenceStatus.QUEUEING)
        queueing_uuids.sort(key=lambda uuid: self.global_batch.get_sequence(uuid).global_idx)

        all_candidates = evicted_uuids + queueing_uuids
        self._prefill_admission_state = None
        self._prefill_admission_selection = None
        self._prefill_admission_totals = None
        self._prefill_admission_live = False
        # Reset here, not at the reservation: _prepare_prefill_batch always runs
        # before _reserve_prefill_host_kv in the scheduler loop, so an iteration
        # that selects nothing reports "not attempted" rather than carrying the
        # previous iteration's verdict into the guard.
        self._prefill_reservation_outcome = None
        if not all_candidates:
            # Clause C7: nothing offered -- the scheduling pool gave this wave
            # no QUEUEING or EVICTED candidate. Recorded so an idle formation
            # is distinguishable from a capacity-bound one in the summary.
            record_prefill_binding_cap(
                self,
                PrefillWaveCapRecord(
                    cap=PREFILL_CAP_SCHEDULING_POOL_EMPTY,
                    candidates_offered=0,
                    candidates_admitted=0,
                    admitted_prompt_tokens=0,
                ),
            )
            return []

        num_nodes = self._get_num_nodes()
        chunk_size = self._get_effective_chunk_size()
        sequence_limits = self._prefill_sequence_limits()
        node_sequence_free = sequence_limits.get("max_sequences_per_node")
        rank_sequence_free = sequence_limits.get("max_sequences_per_rank")
        if node_sequence_free is not None and rank_sequence_free is not None:
            raise ValueError(
                "prefill sequence capacity must be scoped to rank or node, not both"
            )
        if node_sequence_free is not None:
            limit_scope = 2
            local_sequence_free = int(node_sequence_free)
        elif rank_sequence_free is not None:
            limit_scope = 1
            local_sequence_free = int(rank_sequence_free)
        else:
            limit_scope = 0
            local_sequence_free = -1

        # Step 1: Get this node's host KV free pages
        local_host_free = self._get_host_kv_free_pages()

        # This iteration's live set, reported into the gather below so the
        # reclaim branch and the scheduler loop's no-progress guard both decide
        # on the OR across ranks. Read here because nothing between this point
        # and the guard changes a sequence status -- selection, admission
        # reclaim and the Host reservation do not -- while a rank-LOCAL read
        # lets ranks raise on different iterations, i.e. diverge on the very
        # failure path the guard exists to make clean.
        local_live = (
            self.global_batch.has_prefilled()
            or self.global_batch.has_in_decode()
            or self.global_batch.has_on_hold()
        )
        # Admission reclaim and the guard both branch on the candidate snapshot
        # each rank builds from its own global_batch. Carry a digest of the
        # fields that snapshot is derived from, so a divergence is named here
        # instead of sending one rank into a reclaim collective the others
        # never enter (same pattern as _check_pool_plan_identical).
        candidate_digest = self._prefill_candidate_digest(all_candidates)
        # Prefix-cache pressure moves while a prefix load or an attachment
        # release is in flight -- an attachment release in particular makes
        # pages reclaimable WITHOUT changing the free count -- which is what
        # separates "waiting on something" from "stuck" for the guard. Reading
        # it scans the coordinator's whole shared node table under its lock
        # (max_nodes tracks pool pages), so it is taken only when this node has
        # nothing live, the sole case where the guard can fire: a busy worker
        # pays nothing. Node-leader only, like the pool stats, and every rank of
        # the node uses the leader's gathered value, so gating on the leader's
        # own live set keeps the component rank-uniform.
        prefix_pressure = (
            self._prefill_prefix_pressure()
            if self.local_rank == 0 and not local_live
            else (-1, -1, -1)
        )
        # Retained prefix pages this node can reclaim right now (resident, no
        # attachment, no pending load). They join the admission budget below: a
        # retained prefix either gets attached at config time, dropping the
        # wave's demand and the evictable supply by the same page count, or
        # stays evictable for the reserve-time reclaim to release. Either way
        # demand <= free + reclaimable holds, so full-charge stays. Node-leader
        # only (one shared region per node); one O(max_nodes) scan under the
        # coordinator lock, same order as the pressure scan above, and unlike it
        # taken on every iteration -- which is why it mutates nothing.
        report_releasable = (
            self.prefix_cache_coordinator.count_releasable_pages(0)
            if (
                self.local_rank == 0
                and self.enable_prefix_cache
                and self.prefix_cache_coordinator is not None
            )
            else -1
        )

        # Step 2: Gather host KV plus persistent sequence capacity. Host KV is
        # one shared region per node, so only the node leader reports it --
        # free AND total pages, because the total is what makes "this request
        # can never fit the pool" decidable identically on every rank. KDA
        # state is per GPU; every rank reports its free count and the scheduler
        # uses the node minimum. This keeps every rank's selection identical
        # even if a prior lifecycle bug left one TP group asymmetric.
        report_node = self.rank_ctx.node_of(self.rank)
        report_host_free = local_host_free if self.local_rank == 0 else -1
        report_host_total = (
            self._get_host_kv_total_pages() if self.local_rank == 0 else -1
        )
        # One row per rank; -1 marks "not reported by this rank":
        #   [0] node id of this rank
        #   [1] Host KV free pages (node leader only)
        #   [2] sequence-slot limit scope: 0 none, 1 per rank, 2 per node
        #   [3] free sequence slots in that scope (-1 when unlimited)
        #   [4] Host KV total pages (node leader only): decides "can never
        #       fit" for the per-request rejection and the stall verdict
        #   [5] live bit: this rank has a prefilled/decoding/on-hold sequence;
        #       OR-reduced for the reclaim stall check and the progress guard
        #   [6] candidate digest (_prefill_candidate_digest); must be equal on
        #       every rank or PrefillAdmissionStateDivergence is raised
        #   [7..9] prefix-cache pressure of the node (resident nodes, active
        #       attachments, pending loads; node leader with nothing live
        #       only), so the guard tells an in-flight wait from a stall
        #   [10] reclaimable retained prefix pages of the node (node leader
        #       with the prefix cache only): joins the host admission budget
        free_tensor = torch.tensor(
            [
                report_node, report_host_free, limit_scope, local_sequence_free,
                report_host_total, int(local_live), candidate_digest,
                prefix_pressure[0], prefix_pressure[1], prefix_pressure[2],
                report_releasable,
            ],
            dtype=torch.int64,
            device=self.torch_device,
        )
        gathered = [torch.zeros_like(free_tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, free_tensor)

        gathered_digests = {int(item[6].item()) for item in gathered}
        if len(gathered_digests) != 1:
            raise PrefillAdmissionStateDivergence(
                f"prefill admission candidates diverged across ranks: rank "
                f"{self.rank} digest {candidate_digest} over "
                f"{len(all_candidates)} candidates; gathered "
                f"{sorted(gathered_digests)}"
            )
        # Rank-uniform liveness. OR, not AND: a false "live" only postpones the
        # guard, a false "not live" fails a worker that still had work to do.
        self._prefill_admission_live = any(
            int(item[5].item()) != 0 for item in gathered
        )

        # Extract per-node host KV free/total pages and prefix pressure
        reports_by_node = {}
        totals_by_node = {}
        pressure_by_node = {}
        reclaimable_by_node = {}
        for item in gathered:
            node_id = int(item[0].item())
            host_free = int(item[1].item())
            if node_id >= 0 and host_free >= 0:
                reports_by_node[node_id] = host_free
                totals_by_node[node_id] = int(item[4].item())
                pressure_by_node[node_id] = (
                    int(item[7].item()), int(item[8].item()), int(item[9].item())
                )
                # -1 is "this leader runs no prefix cache", i.e. no reclaimable
                # supply -- never a negative budget contribution.
                reclaimable_by_node[node_id] = max(0, int(item[10].item()))
        per_node_host_free = []
        per_node_host_total = []
        per_node_prefix_pressure = []
        per_node_host_reclaimable = []
        for node in range(num_nodes):
            per_node_host_free.append(reports_by_node.get(node, 0))
            per_node_host_reclaimable.append(reclaimable_by_node.get(node, 0))
            # -1 means "no leader report": an unknown total must not be read as
            # a zero-page pool that can never fit anything.
            per_node_host_total.append(totals_by_node.get(node, -1))
            per_node_prefix_pressure.append(
                pressure_by_node.get(node, (-1, -1, -1))
            )
        # Pool sizes for the stall report. Gathered, so a multi-node stall can
        # name "can never fit" too, where the old rank-local read could not.
        self._prefill_admission_totals = tuple(per_node_host_total)

        gathered_scopes = {int(item[2].item()) for item in gathered}
        if len(gathered_scopes) != 1:
            raise RuntimeError(
                f"prefill sequence-capacity scope diverged across ranks: "
                f"{sorted(gathered_scopes)}"
            )
        per_rank_sequence_free = None
        per_node_sequence_free = None
        gathered_scope = next(iter(gathered_scopes))
        if gathered_scope == 1:
            per_rank_sequence_free = [int(item[3].item()) for item in gathered]
        elif gathered_scope == 2:
            per_node_sequence_free = []
            for node in range(num_nodes):
                values = [
                    int(item[3].item())
                    for item in gathered
                    if int(item[0].item()) == node
                ]
                if not values:
                    raise RuntimeError(
                        f"missing persistent sequence-capacity report for node {node}"
                    )
                per_node_sequence_free.append(min(values))
        elif gathered_scope != 0:
            raise RuntimeError(
                f"unknown prefill sequence-capacity scope {gathered_scope}"
            )

        if self.rank == 0:
            logging.info(f"Per-node host KV free pages: {per_node_host_free} (chunk_size={chunk_size})")
            if per_node_sequence_free is not None:
                logging.info(
                    f"[PREFILL] Persistent sequence slots free by node: "
                    f"{per_node_sequence_free}"
                )

        # Under G>1 the selection request maps each candidate to a node via its
        # decode_dp_group, so every candidate must already carry one. Newly
        # queued sequences are grouped later (at the decode transition), so a
        # wave mixing queued with evicted candidates would raise "sequence ...
        # has no decode_dp_group before TP prefill selection". Stamp them here:
        # _assign_decode_dp_groups is documented to run at admission/prefill
        # config, is idempotent (re-entrant sequences keep the group their
        # head-sharded KDA state lives on), and is deterministic across ranks.
        self._assign_decode_dp_groups(all_candidates)
        # The map decides this wave's node mapping (below), its bindings and
        # its KDA state, so verify agreement BEFORE any of them is derived:
        # _prepare_prefill_batch is already collective, and this is the first
        # all-ranks point after the stamp.
        self._check_decode_dp_group_plan_identical(all_candidates)

        # Step 3: Select sequences considering per-node host KV capacity.
        # The NCCL gather above and the logging below stay here; the greedy
        # per-node admission is delegated to PrefillScheduler. Kimi-Linear
        # additionally supplies a persistent KDA-state limit: unlike the token
        # cap used later by prepack, a KDA slot remains occupied until the
        # sequence completes or is evicted.
        selection = self._make_prefill_selection_request(
            all_candidates,
            per_node_host_free,
            num_nodes,
            chunk_size,
            per_rank_sequence_free=per_rank_sequence_free,
            per_node_sequence_free=per_node_sequence_free,
            per_node_host_reclaimable=per_node_host_reclaimable,
        )
        # One reservation per candidate, reused by the rejection below, the
        # reclaim target and the guard state. Independent of free pages, so it
        # survives the post-reclaim `replace(selection, ...)`.
        required_pages = {
            candidate.uuid: PrefillScheduler.required_host_pages(
                candidate, selection
            )
            for candidate in selection.candidates
        }
        # A request whose initial Host reservation exceeds its node's WHOLE pool
        # can never be admitted -- not by this wave, not by any later one, no
        # matter how much is reclaimed. Fail that REQUEST, not the worker: the
        # stall exceptions below leave _prepare_prefill_batch for
        # generate_persistent(), which has no handler for them, so they reach
        # the worker fatal boundary and take every rank down for one oversized
        # prompt. Rank-uniform: per_node_host_total is gathered and the
        # candidate snapshot was digest-checked above, so every rank rejects the
        # same uuids with no further collective.
        rejected_uuids = self._reject_unsatisfiable_prefill_requests(
            selection, per_node_host_total, required_pages
        )
        if rejected_uuids:
            # The candidate set changed under the snapshot. Rebuild it on the
            # next iteration instead of selecting against removed sequences; the
            # cleared state also resets the no-progress streak, which is correct
            # -- this iteration did make progress.
            self._prefill_admission_state = None
            self._prefill_admission_selection = None
            # Clause C7: this wave selected nothing because oversized QUEUEING
            # requests were failed on the per-request channel, not because a
            # capacity refused the next candidate. Recorded under its own name
            # so the summary never counts it as a capacity verdict. The
            # ``available`` field carries the first rejected request's whole
            # node pool -- the total the rejection compared against.
            first_rejected = next(
                candidate
                for candidate in selection.candidates
                if candidate.uuid == rejected_uuids[0]
            )
            record_prefill_binding_cap(
                self,
                PrefillWaveCapRecord(
                    cap=PREFILL_CAP_OVERSIZED_REQUESTS_REJECTED,
                    candidates_offered=len(selection.candidates),
                    candidates_admitted=0,
                    admitted_prompt_tokens=0,
                    first_blocked_uuid=first_rejected.uuid,
                    first_blocked_node=first_rejected.node_id,
                    first_blocked_needed=required_pages[first_rejected.uuid],
                    first_blocked_available=int(
                        per_node_host_total[first_rejected.node_id]
                    ),
                ),
            )
            return []
        wave_selection = PrefillScheduler.select_prefill_wave(selection)
        prefill_batch = list(wave_selection.admitted_uuids)
        # The in-wave prefix pool releases every cross-wave lookup hit and
        # recomputes it (_plan_prefill_pool_wave), so a hit saves no Host pages
        # there: discounting it over-admits, and the allocation fails once the
        # retained prefixes are all pinned by live sequences. Those retained
        # pages are reclaimable instead.
        pool_mode = self._prefill_prefix_pool_enabled()
        if (
            self.enable_prefix_cache and selection.candidates and not pool_mode
            and per_rank_sequence_free is None
            and per_node_sequence_free is None
        ):
            # A cached prefix already occupies Host pages, so charging a hit its
            # full reservation lets one full-cost prompt fill the wave while
            # cheap hits wait (one sequence per wave under tight Host KV).
            # Estimate on each node leader and select with the discount whenever
            # any candidate has a hit; the actual lookup and attach still happen
            # in _config_prefill_for_batch. EVICTED prompts are reconstructed
            # there, so retain their existing full-page path.
            from batchgen.prefix_reuse.prefill import (
                estimate_prefix_cached_pages_for_prefill,
            )
            estimated_values = [0] * (len(selection.candidates) + 1)
            if self.local_rank == 0:
                try:
                    for index, candidate in enumerate(selection.candidates):
                        if candidate.node_id != report_node or candidate.is_evicted:
                            continue
                        seq = self.global_batch.get_sequence(candidate.uuid)
                        estimated_values[index] = estimate_prefix_cached_pages_for_prefill(
                            coordinator=self.prefix_cache_coordinator,
                            namespace_digest=(
                                self.prefix_cache_runtime_config.namespace_digest
                            ),
                            prompt_token_ids=self._trajectory_tokens(
                                seq, length=seq.prompt_length
                            ).tolist(),
                            page_size_tokens=(
                                self.prefix_cache_runtime_config.group_specs[0]
                                .raw_page_tokens
                            ),
                        )
                except Exception:
                    logging.exception("[PREFIX_CACHE] admission estimate failed")
                    estimated_values[-1] = 1
            estimated = torch.tensor(
                estimated_values, dtype=torch.int64,
                device=self.torch_device,
            )
            dist.all_reduce(estimated, op=dist.ReduceOp.MAX)
            if int(estimated[-1].item()):
                raise RuntimeError("prefix cache admission estimate failed on a worker")
            cached_pages = estimated[:-1].tolist()
            if any(cached_pages):
                estimated_selection = replace(
                    selection,
                    candidates=tuple(
                        replace(candidate, cached_prefix_pages=int(cached_pages[index]))
                        for index, candidate in enumerate(selection.candidates)
                    ),
                )
                wave_selection = PrefillScheduler.select_prefill_wave(
                    estimated_selection
                )
                prefill_batch = list(wave_selection.admitted_uuids)
        # Admission reclaim targets what the persistent sequence-slot limits
        # (Kimi-K3/Kimi-Linear KDA state) would admit with a fully reclaimed
        # Host pool. Without slot limits that is every candidate, i.e. the
        # previous trigger. A slot-limited model used to be excluded outright,
        # so retained prefixes filled its Host pool and every later wave
        # selected zero sequences forever with nothing ever evicted.
        slot_admissible = PrefillScheduler.slot_admissible_candidates(selection)
        # Split the slot-admissible set into what this node's pool can hold at all
        # and what it can never hold, from the GATHERED totals rather than a
        # rank-local stats read, so every rank enters the branch below -- and
        # raises its stall -- on one shared verdict. A node whose leader reported
        # no total (-1) counts as "unknown", never as a zero-page pool.
        def _fits(candidate) -> bool:
            total = per_node_host_total[candidate.node_id]
            return total < 0 or required_pages[candidate.uuid] <= total

        # The reclaim has to buy room for the next candidate the pool CANNOT
        # take yet, so exclude the ones this wave already admitted: in pool mode
        # the trigger fires while part of the set is in, and targeting the head
        # of the whole set would evict for a sequence already admitted.
        admitted_uuids = set(prefill_batch)
        fitting_requirements = [
            required_pages[candidate.uuid] for candidate in slot_admissible
            if candidate.node_id == report_node
            and candidate.uuid not in admitted_uuids
            and _fits(candidate)
        ]
        any_node_can_fit = any(_fits(c) for c in slot_admissible)
        reclaim_outcome = (0, 0)
        if (
            self.enable_prefix_cache
            and slot_admissible
            and (
                len(prefill_batch) < len(slot_admissible) if pool_mode
                else not prefill_batch
            )
        ):
            if not any_node_can_fit:
                # Every slot-admissible candidate needs more than its node's whole
                # pool. Oversized QUEUEING requests were already failed one by one
                # above, so what is left is an EVICTED sequence that no longer fits
                # a pool it did fit once: an engine invariant violation, not a bad
                # request. Raised before this branch's first collective, on a
                # verdict every rank computed from the same gathered totals.
                self._prefill_admission_selection = selection
                raise PrefillAdmissionStalled(
                    "no prefix-cache request fits the total Host KV pool: "
                    + self._describe_prefill_admission_blockers()
                )
            # No candidate fits even after a non-mutating prefix estimate.
            # Reclaim against the original full-page requirement so a later lookup
            # miss cannot make the allocation unsafe, and only as far as the head
            # of the priority order needs (prefill_reclaim_target_pages):
            # reclaiming for the whole admissible set evicts retained prefixes
            # this wave will not use, and every one of them is a hot prefix a
            # later prompt has to recompute.
            from batchgen.prefix_reuse.eviction import (
                EvictedPageRecorder,
                reclaim_prefix_pages_for_host_admission,
                retire_node_frontiers,
            )
            failure = 0
            updated_local_free = -1
            # A reclaim moves pages from reclaimable to free. Re-report BOTH, or
            # the refreshed budget below counts the released pages twice.
            updated_local_releasable = -1
            local_frontier_hook = self._trajectory_frontier_eviction_hook()
            reclaimed_pages = EvictedPageRecorder(local_frontier_hook)
            if self.local_rank == 0:
                try:
                    worker_view = self.core_engine.host_paged_kv_worker_view
                    stats = worker_view.get_stats()
                    updated_local_free = int(stats.num_free_pages)
                    required, page_target = prefill_reclaim_target_pages(
                        fitting_requirements=fitting_requirements,
                        free_pages=updated_local_free,
                    )
                    if required:
                        released = reclaim_prefix_pages_for_host_admission(
                            core_engine_module=core_engine,
                            coordinator=self.prefix_cache_coordinator,
                            worker_views_by_group=self._host_views_by_group(),
                            group_id=0,
                            page_target=page_target,
                            max_scan_nodes=self.prefix_cache_runtime_config.max_nodes,
                            frontier_hook=reclaimed_pages,
                        )
                        updated_local_free = self._get_host_kv_free_pages()
                        updated_local_releasable = (
                            self.prefix_cache_coordinator.count_releasable_pages(0)
                        )
                        logging.info(
                            "[PREFIX_CACHE] admission reclaimed %s Host pages on node %s "
                            "(free=%s, target=%s)",
                            released, report_node, updated_local_free, required,
                        )
                        if self.prefix_cache_debug_stats:
                            self._emit_prefix_cache_metric(
                                "prefix_reclaim",
                                reason="host_admission",
                                node=int(report_node),
                                page_target=int(page_target),
                                released_pages=int(released),
                                **self._prefix_host_resource_fields(),
                            )
                except Exception:
                    logging.exception("[PREFIX_CACHE] admission eviction failed")
                    failure = 1
            failed = torch.tensor(
                [failure, reclaimed_pages.page_count],
                dtype=torch.int64, device=self.torch_device,
            )
            dist.all_reduce(failed, op=dist.ReduceOp.MAX)
            if int(failed[0].item()) != 0:
                raise RuntimeError("prefix cache admission eviction failed on a worker")
            reclaimed_page_count = int(failed[1].item())
            reclaim_outcome = (1, reclaimed_page_count)
            # The leader already retired its own frontiers inside the release.
            # Any other rank of the node may own frontier records standing on
            # the reclaimed pages (G=1, or several serve groups per node), so
            # every rank runs its own hook on its node's reclaimed map.
            frontier_failure = 0
            if reclaimed_page_count != 0:
                gathered_reclaims = [None] * self.world_size
                dist.all_gather_object(
                    gathered_reclaims,
                    (
                        report_node,
                        reclaimed_pages.pages_by_group
                        if self.local_rank == 0 else None,
                    ),
                )
                if self.local_rank != 0:
                    try:
                        retire_node_frontiers(
                            gathered=gathered_reclaims,
                            node_id=report_node,
                            frontier_hook=local_frontier_hook,
                        )
                    except Exception:
                        logging.exception(
                            "[PREFIX_CACHE] frontier retirement after admission "
                            "reclaim failed"
                        )
                        frontier_failure = 1
            updated = torch.tensor(
                [
                    report_node, updated_local_free, frontier_failure,
                    updated_local_releasable,
                ],
                dtype=torch.int64, device=self.torch_device,
            )
            gathered = [torch.zeros_like(updated) for _ in range(self.world_size)]
            dist.all_gather(gathered, updated)
            if any(int(item[2].item()) != 0 for item in gathered):
                raise RuntimeError(
                    "trajectory frontier retirement after admission reclaim "
                    "failed on a worker"
                )
            for item in gathered:
                node, free = int(item[0].item()), int(item[1].item())
                if free >= 0:
                    per_node_host_free[node] = free
                releasable = int(item[3].item())
                if releasable >= 0:
                    per_node_host_reclaimable[node] = releasable
            selection = replace(
                selection,
                per_node_host_free=tuple(per_node_host_free),
                per_node_host_reclaimable=tuple(per_node_host_reclaimable),
            )
            wave_selection = PrefillScheduler.select_prefill_wave(selection)
            prefill_batch = list(wave_selection.admitted_uuids)
            if not prefill_batch and not self._prefill_admission_live:
                # Nothing is live on ANY rank, so nothing will decode, complete, or
                # free a page or a slot: an empty wave here would repeat forever.
                self._prefill_admission_selection = selection
                raise PrefillAdmissionStalled(
                    "prefix cache prefill admission made no progress after "
                    "eviction with no live sequence: "
                    + self._describe_prefill_admission_blockers()
                )

        # Rank-uniform admission state for the scheduler loop's no-progress guard
        # (PrefillProgressGuard). Every component is gathered, reduced, or derived
        # from the digest-checked candidate snapshot, and the set is wide enough to
        # separate a recoverable wait from a true stall: free AND total pages, the
        # slot vectors, the candidate IDENTITIES with their required pages (a count
        # alone cannot see one candidate swapped for another), the per-node
        # prefix-cache pressure counters that move while a load or an attachment
        # release is in flight, this wave's reclaim outcome, and the reduced live
        # bit. The scheduler loop folds the Host reservation outcome in on top
        # (_prefill_admission_guard_state).
        self._prefill_admission_state = (
            tuple(per_node_host_free),
            tuple(per_node_host_total),
            None if per_rank_sequence_free is None else tuple(per_rank_sequence_free),
            None if per_node_sequence_free is None else tuple(per_node_sequence_free),
            tuple(
                (
                    candidate.uuid,
                    candidate.node_id,
                    candidate.is_evicted,
                    required_pages[candidate.uuid],
                )
                for candidate in selection.candidates
            ),
            tuple(per_node_prefix_pressure),
            reclaim_outcome,
            self._prefill_admission_live,
        )
        self._prefill_admission_selection = selection

        if self.rank == 0:
            n_evicted = sum(
                1 for u in prefill_batch
                if self.global_batch.get_sequence(u).status == SequenceStatus.EVICTED
            )
            logging.info(
                f"[PREFILL] Selected {len(prefill_batch)} sequences "
                f"({n_evicted} recompute from eviction)"
            )

        # Clause C7: record THIS wave's binding cap -- the record of the final
        # selection the wave actually uses (initial, prefix-estimate, or
        # post-reclaim), produced by that selection's own exit path.
        record_prefill_binding_cap(self, wave_selection.cap_record)

        return prefill_batch

    def _config_prefill_for_batch(
        self, worker, prefill_uuids: List[str], *, token_cap: int,
        attn_group_size: int,
    ) -> None:
        """Configure prefill phase for a batch of sequences."""
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        start_time = time.perf_counter()
        self._phase_request_uuids = tuple(prefill_uuids)
        desired_key = self._desired_phase_key("prefill")
        prefill_sequences = [
            self.global_batch.get_sequence(uuid)
            for uuid in prefill_uuids
            if self.global_batch.get_sequence(uuid) is not None
        ]
        prefill_debug = self._phase_prefill_debug
        k3_prefill_profile = self._phase_k3_prefill_profile
        persistent = self._persistent_phase_enabled()
        if not persistent:
            self._request_phase_instance_reuse()
        if self.rank == 0:
            logging.info(
                f"[PREFILL] Configuring prefill phase for {len(prefill_uuids)} sequences"
            )

        in_decode = self.global_batch.get_sequences_by_status(SequenceStatus.IN_DECODE)
        on_hold = self.global_batch.get_sequences_by_status(SequenceStatus.ON_HOLD)
        prefilling = self.global_batch.get_sequences_by_status(SequenceStatus.PREFILLED)
        if (in_decode or on_hold) and BATCHGEN_CB_DEBUG:
            logging.debug(
                f"Rank {self.rank}: _config_prefill_for_batch called while "
                f"{len(in_decode)} IN_DECODE, {len(on_hold)} ON_HOLD, {len(prefilling)} PREFILLED sequences exist. "
                "This is a decode-to-prefill transition."
            )
            for uuid in (in_decode + on_hold)[:5]:
                seq = self.global_batch.get_sequence(uuid)
                logging.debug(
                    f"Rank {self.rank}: Affected seq {seq.uuid[:8]}: "
                    f"status={seq.status.name}, decoded_len={seq.decoded_length}, "
                    f"ctx_len={seq.current_context_length}, gpu_pages={seq.gpu_pages_allocated}, "
                    f"had_initial={seq.had_initial_gpu_reservation}"
                )

        if hasattr(self, "_pending_kv_append_tasks") and self._pending_kv_append_tasks:
            logging.info(
                f"Rank {self.rank}: Flushing {len(self._pending_kv_append_tasks)} pending KV append tasks before prefill config"
            )
            self._wait_pending_kv_append_tasks()
            torch.cuda.synchronize(self.torch_device)

        self._assign_decode_dp_groups(prefill_uuids)
        if self._live_phase_key != desired_key:
            self._log_prefill_entry_hbm("BEFORE")
        phase_changed, resumed = self._transition_phase(desired_key)
        if phase_changed:
            self._log_prefill_entry_hbm("AFTER")
        if phase_changed and not resumed:
            self._log_phase_memory_ledger("after_configure_prefill")
        if phase_changed:
            self.set_phase("prefill")
            if self.enable_prefix_cache:
                self._measure_prefill_prefix_kv_budget()
            self._install_prefill_weight_copy_pipeline(k3_prefill_profile)
        else:
            # phase_enter readies the instance for the first wave of a phase;
            # admit prepares every later wave of the same live phase.
            self.parallel_manager.admit(
                desired_key, AdmitContext(uuids=list(prefill_uuids))
            )

        if phase_changed and not persistent and k3_prefill_profile and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.local_rank)

        if (
            not persistent
            and hasattr(self.parallel_manager, "prefill_uses_resident_ep")
            and self.parallel_manager.prefill_uses_resident_ep()
        ):
            local_lengths = [
                int(seq.prompt_length)
                for seq in prefill_sequences
                if self._owns_local_sequence(seq)
            ]
            use_l2 = os.environ.get("BATCHGEN_L2_BALANCE", "1") == "1"
            predicted_batches, _ = build_prefill_micro_batches(
                local_lengths,
                token_cap,
                l2_balance=use_l2,
            )
            local_max_tokens = max(
                (
                    sum(local_lengths[start:end])
                    for start, end in predicted_batches
                ),
                default=0,
            )
            moe_ntp = self._sync_prefill_moe_rank_counts(
                local_max_tokens,
                reason="prefill_output_preallocate",
            )
            self.parallel_manager.prepare_resident_ep_prefill_output(
                self.world_size * moe_ntp
            )

        # STEP 3: Prepare evicted sequences for re-entry (before host KV allocation)
        #
        # The phase transition and manager admission are complete above. The
        # remaining body prepares metadata and binds this wave's local rows.

        #
        # Split into two loops:
        #   (a) All-ranks scalar metadata update (runs on every rank using
        #       fields already synchronized via Phase 4.C of the eviction
        #       boundary and via _sync_sequence_metadata).
        #   (b) Validate the retained trajectory and clear the legacy marker.
        #
        # The previous single-loop version ran both steps gated on
        # evicted_token_ids — which is an owner-only tensor — so non-owning
        # ranks silently skipped the scalar updates and held stale values
        # for decoded_length / reentry_decoded_baseline / max_decode_length
        # until the next _sync_sequence_metadata call.

        # (a) All-ranks scalar metadata update for re-entering sequences.
        for uuid in prefill_uuids:
            seq = self.global_batch.get_sequence(uuid)
            # total_decoded_before_eviction > 0 identifies sequences that have
            # been evicted at least once and are now re-entering. This field is
            # synced across ranks, unlike evicted_token_ids which is owner-only.
            if seq.total_decoded_before_eviction == 0:
                continue

            # seq.prompt_length and seq.current_context_length were already
            # updated to the new reconstructed length by Phase 4.C of the
            # eviction boundary and synced to all ranks.

            # Baseline = accumulated historical output length carried forward
            # into decoded_tokens. With the Phase 4.C cascade fix, this is
            # exactly (prompt_length - original_prompt_length) = sum of new
            # decoded counts across all past cycles.
            n_old = _reentry_history_length(
                seq.prompt_length, seq.original_prompt_length,
                seq.original_max_decode_length,
            )
            # Fail loud where the corrupted state is produced: the rebase
            # below rebuilds the context identity from n_old, so a history
            # length that does not match the reconstructed prompt would be
            # silently absorbed here and only surface later in
            # validate_metadata, far from the cause.
            if n_old != seq.prompt_length - seq.original_prompt_length:
                raise ValueError(
                    f"[REENTRY_HISTORY] sequence {uuid}: reconstructed "
                    f"history length {n_old} != prompt_length "
                    f"({seq.prompt_length}) - original_prompt_length "
                    f"({seq.original_prompt_length}) = "
                    f"{seq.prompt_length - seq.original_prompt_length}"
                )
            seq.rebase_decoded_length(n_old)
            seq.reentry_decoded_baseline = n_old

            # decoded_length is cumulative across eviction/re-entry cycles, so
            # max_decode_length must remain the absolute per-request completion
            # cap. Compute remaining budget as
            # original_max_decode_length - decoded_length at call sites instead
            # of storing a relative value here.
            seq.max_decode_length = seq.original_max_decode_length

            # Reset completion flags — the sequence may have hit EOS in its
            # previous decode cycle before being evicted. Without this reset,
            # _sync_completion_status_tensor falsely detects the re-entering
            # sequence as completed (stale eos_reached=True), calls
            # _report_completion (popping local_map), but the PREFILLED→COMPLETED
            # status transition fails (invalid), leaving a zombie: PREFILLED
            # status with no local_map entry, invisible to the boundary load
            # mechanism (which iterates local_map), stuck for the entire decode
            # cycle until _prepare_decode_batch picks it up → CRITICAL error.
            seq.eos_reached = False
            if hasattr(seq, '_rep_detected'):
                seq._rep_detected = False

        # (b) The trajectory is already the source of truth. Host-KV eviction
        # releases only KV pages; it must not reconstruct or copy a second token
        # buffer. Validate the retained trajectory and clear the legacy handoff
        # marker so old callers cannot accidentally reintroduce dense storage.
        for uuid in prefill_uuids:
            seq = self.global_batch.get_sequence(uuid)
            if seq.total_decoded_before_eviction == 0:
                continue
            prev_decoded = seq.total_decoded_before_eviction
            seq.log_event(SeqEvent.REENTRY_START, self.rank,
                f"new_prompt_len={seq.prompt_length}, prev_decoded={prev_decoded}")
            if self._traj.trajectory_book is None or not self._traj.trajectory_book.has_sequence(uuid):
                raise QueryBookPoolCapacityError(
                    f"Rank {self.rank}: re-entry of {uuid[:8]} has no trajectory slot"
                )
            trajectory_slot = self._trajectory_slot(seq)
            trajectory_length = self._traj.trajectory_book.metadata(trajectory_slot).token_length
            if trajectory_length < seq.prompt_length:
                raise RuntimeError(
                    f"Rank {self.rank}: trajectory for {uuid[:8]} is shorter than "
                    f"re-entry prompt ({trajectory_length} < {seq.prompt_length})"
                )
            # Record the replayed prompt boundary and make the local QueryBook
            # metadata agree with the scalar effective prompt length. No token bytes
            # are copied: the trajectory already contains this prefix.
            self._traj.trajectory_book.begin_reentry_turn(
                trajectory_slot, prompt_length=seq.prompt_length
            )
            # Clear the legacy owner-only tensor marker. No token copy occurs.
            seq.evicted_token_ids = None

            # Recreate query_book entry for this rank's evicted sequences (Q4).
            # G>1 binds the row on all G ranks of its serve group, and each of
            # them decodes from its own query_book, so every holder must rebuild
            # the entry — not just assigned_rank. The group id an evicted re-entry
            # lost is already restored by the _assign_decode_dp_groups call at the
            # top of this method. G==1: the same single rank as before.
            if self._owns_local_sequence(seq) and uuid in self._uuid_to_local_map:
                refresh_query_book_entry(
                    uuid, seq,
                    query_book=self.query_book,
                    local_of=self._uuid_to_local_map.get,
                )

            logging.info(
                f"Rank {self.rank}: Prepared EVICTED seq {uuid[:8]} for re-entry: "
                f"new_prompt={seq.prompt_length}, prev_decoded={prev_decoded}, "
                f"remaining_decode={seq.max_decode_length}, kv_budget={seq.kv_token_budget}"
            )

        # STEP 3.5 (Option 1, CORE): (re)assign the serve-group before binding.
        # Idempotent for fresh admits (already grouped at admission); RE-groups
        # evicted re-entries (whose decode_dp_group was cleared on eviction) so
        # the group predicate below binds them on all G ranks. No-op for G==1.
        self._assign_decode_dp_groups(prefill_uuids)

        # STEP 4: Bind local rows for sequences this rank serves.  Host-KV pages
        # were reserved and committed before IN_PREFILL was published, so this
        # phase must not perform another allocator mutation.
        my_prefill_uuids = []
        for uuid in prefill_uuids:
            seq = self.global_batch.get_sequence(uuid)
            if self._owns_local_sequence(seq):
                my_prefill_uuids.append(uuid)
                # Add to local maps if not already present (for new sequences)
                if uuid not in self._uuid_to_local_map:
                    new_local_idx = self._bind_local_sequence_to_query_book(uuid)
                    logging.debug(
                        f"Rank {self.rank}: Added new sequence {uuid[:8]}... to local maps "
                        f"(local_idx={new_local_idx})"
                    )
        if self.rank == 0:
            logging.debug(
                "[PREFILL] Host KV reservation already committed before phase config"
            )
        # The rows bound above are the rows this wave's prefill computes, and
        # the prefill forward is what creates each row's KDA state item. Catch
        # a binding that disagrees with the group map here, before the forward,
        # so the error names the seam instead of the decode step that misses
        # the state item.
        self._check_local_binding_matches_group(prefill_uuids, "prefill config")

        # Reached by EVERY rank — including ranks that serve no sequence this
        # wave — so the gather inside stays aligned. No-op for G==1; for G>1 it
        # hands the leader's page tables, prefix metadata and pool segment
        # pages to the rest of its serve group.
        #
        # M3 Slice 1c rides this same gather when the trajectory flag is on and
        # the group is head-sharded: the leader's G-wide snapshot arenas and
        # its frontier hits travel with the page tables, so no new collective
        # is introduced. The predicate is broadcast-identical (the flag is on
        # the batch's own sequences and the batch is replicated), so every rank
        # reaches the gather with the same expectation.
        self._traj.trajectory_arena_payload = {}
        self._traj.trajectory_mirrored_restores = {}
        mirror_trajectory = (
            attn_group_size > 1
            and self.enable_prefix_cache
            and self._trajectory_commit_requested(
                [
                    self.global_batch.get_sequence(uuid)
                    for uuid in prefill_uuids
                ]
            )
        )
        self._mirror_group_host_kv(
            my_prefill_uuids,
            self._prefix.prefix_sequence_states,
            trajectory=mirror_trajectory,
        )

        if self.rank == 0:
            logging.info(f"[PREFILL] Config completed: {(time.perf_counter() - start_time)*1000:.1f}ms")
