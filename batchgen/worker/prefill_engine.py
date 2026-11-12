"""Prefill wave planning and lifecycle behind a worker seam.

The engine is intentionally stateless.  Each operation receives the live worker
as its collaborator, then rebinds ``self`` to that worker so the moved statements
keep their original state access and call spellings.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from batchgen.kv_cache.dual_host_kv_coordinator import DualHostKVCoordinator
from batchgen.prefill.prepack import build_prefill_micro_batches


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


class PrefillEngine:
    """Stateless S5a wave planner; the worker owns all mutable state."""

    def _plan_prefill_micro_batches(
        self, worker,
        seq_lengths: List[int],
        full_seq_lengths: List[int],
        compute_cached_tokens: List[int],
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
        token_cap = (
            self.engine_config.Module_Batching_Config
            .prefill_micro_batch_token_cap
        )
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
            if self._decode_attn_tp_size() != 1:
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

    def _prefill_model_pass_count(self, worker, batch: List[int]) -> int:
        """Return the number of complete model forwards local prefill will run."""
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        if not batch:
            return 0
        if self._prefill_pool_wave is not None:
            # A pool wave runs one forward per planned chunk, not one per
            # micro-batch of whole prompts: a chunk packs pooled segments and
            # tails. The plan is identical on every rank of the serve group, so
            # the pass counts the transport padding aligns against agree.
            return len(self._prefill_pool_wave.plan.chunks)
        if not self.enable_prepack:
            micro_batch_size = (
                self.engine_config.Module_Batching_Config
                .MoE_prefill_micro_batch_size
            )
            return math.ceil(len(batch) / micro_batch_size)

        # The same inputs prefill_prepacked plans with: computed (suffix)
        # lengths, prompt lengths and the attached prefix of each row.
        full_seq_lengths = []
        compute_cached_tokens = []
        for local_idx in batch:
            seq = self.global_batch.get_sequence(self._local_to_uuid_map[local_idx])
            prefix_state = self._prefix.prefix_sequence_states.get(seq.global_idx)
            full_seq_lengths.append(seq.prompt_length)
            compute_cached_tokens.append(
                prefix_state.compute_cached_tokens
                if prefix_state is not None
                else 0
            )
        micro_batches, _, _ = self._plan_prefill_micro_batches(
            [
                full - cached
                for full, cached in zip(full_seq_lengths, compute_cached_tokens)
            ],
            full_seq_lengths,
            compute_cached_tokens,
        )
        return len(micro_batches)

    def _prefill_prefix_pool_enabled(self, worker) -> bool:
        """In-wave prefix pool: prefix caching on, planner workspace set, pure DP prefill."""
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        workspace = self.engine_config.Module_Batching_Config.prefill_prefix_pool_workspace_bytes
        if not self.enable_prefix_cache:
            return False
        from batchgen.prefix_reuse.config import PrefixKVGroupSemantic
        if any(
            spec.semantic is not PrefixKVGroupSemantic.FULL_KV
            for spec in self.prefix_cache_runtime_config.group_specs
        ) and (workspace <= 0 or getattr(self, "_prefix_integrity", False)):
            # Only the pool path serves MLA-latent prefix reuse (GLM DSA, Kimi-K3);
            # the cross-wave materialization path handles one full-K/V group.
            raise RuntimeError(
                "MLA-latent prefix reuse needs the prefill prefix pool, but this "
                "model's planner sets no pool workspace or the integrity check is on"
            )
        if workspace <= 0 or getattr(self, "_prefix_integrity", False):
            return False
        if (
            hasattr(self.parallel_manager, "prefill_uses_resident_ep")
            and self.parallel_manager.prefill_uses_resident_ep()
        ) or (
            self._decode_attn_tp_size() != 1
            and not self._pool_prefill_group_safe()
        ):
            raise RuntimeError(
                "the prefill prefix pool runs per DP rank; this model's prefill "
                "joins cross-rank collectives"
            )
        return True

    def _pool_prefill_group_safe(self, worker) -> bool:
        """Whether the PSM plans ONE identical pool wave per serve group (G>1).

        At G>1 a sequence is replicated onto all G ranks of its serve group and
        every one of them prefills it, so the pool is safe only when the whole
        group runs the same chunks in the same order — the per-layer collectives
        then still pair up. The PSM declares that; the resident-EP refusal above
        stands either way, since those passes join EP-world collectives.
        """
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        declares = getattr(
            self.parallel_manager, "supports_group_pool_prefill", None
        )
        return declares is not None and bool(declares())

    def _prefill_pool_group(self, worker):
        """The process group a pool wave must be identical over; None at G==1."""
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        if self._decode_attn_tp_size() <= 1:
            return None
        group = getattr(self.parallel_manager, "attn_tp_group", None)
        if group is None:
            raise RuntimeError(
                "the prefill prefix pool at attn_tp_size>1 needs the attention "
                "TP process group to plan one wave per serve group"
            )
        return group

    def _prefill_pool_free_state_slots(self, worker) -> int:
        """Free recurrent-state slots, or -1 when the model keeps no such state.

        A pooled segment runs as a row of its own under a pseudo sequence id, so
        it holds one slot for the whole wave on top of the wave's sequences.
        """
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        manager = self._traj.kda_state_manager()
        if manager is None:
            return -1
        return int(manager.get_stats().num_free_state_items)

    def _next_pool_segment_sid(self, worker) -> int:
        """Negative Host sequence ids, rank-strided so no two ranks collide."""
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        return self._prefix.next_pool_segment_sid()

    def _prefix_rank_packing_mode(self, worker, seqs) -> bool:
        """True when this batch asked for the predicted-time wave packer."""
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        active_debug = getattr(self, "_active_batchgen_debug_for_sequences", None)
        debug = (active_debug(seqs) if active_debug is not None else None) or {}
        return debug.get("prefix_rank_assignment_mode") == "packing_v2"

    def _prefix_rank_cost_profile_disabled(self, worker, seqs) -> bool:
        """``batchgen_debug.prefix_rank_cost_profile: "off"`` on this batch.

        Scoring the packer's targets by predicted seconds and balancing their
        computed tokens are the two arms of the packing comparison, and the
        difference between them is a wall time: measuring it across two servers
        would confound it with cold-start, allocator and JIT state, so the arm
        has to be switchable inside one process. Batch level and nothing else:
        the key rides the same admitted-batch broadcast as
        ``prefix_rank_assignment_mode``, so every rank of every group resolves
        the same value and the packing stays replicated. Set it on EVERY
        request of a corpus — ``_active_batchgen_debug_for_sequences`` merges
        per-sequence dicts first-non-None-wins, exactly as the mode key does.

        Only the two spellings are accepted; a mistyped value is refused rather
        than read as "on", because a silently ignored arm switch would make the
        two arms the same run.
        """
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        active = getattr(self, "_active_batchgen_debug_for_sequences", None)
        debug = (active(seqs) if active is not None else None) or {}
        value = debug.get("prefix_rank_cost_profile")
        if value is None:
            return False
        if isinstance(value, str):
            text = value.strip().lower()
            if text == "off":
                return True
            if text == "on":
                return False
        raise ValueError(
            "batchgen_debug.prefix_rank_cost_profile must be 'on' or 'off', "
            f"got {value!r}"
        )

    def _cached_prefix_rank_cost_profile(self, worker, disabled: bool = False):
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

            G = self._decode_attn_tp_size()
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

    def _plan_prefill_pool_wave(self, worker, uuids, prefix_states):
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
        batching = self.engine_config.Module_Batching_Config
        chunk_tokens = int(batching.prefill_micro_batch_token_cap)
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

    def _prefill_pool_psm_layout(self, worker):
        """The PSM's pool-geometry override, or None to use the Host geometry.

        A model whose pool holds the EXPANDED heads (K3: 192-wide K and
        128-wide V per LOCAL head against one 576-wide latent) takes its pool
        geometry from the PSM; the Host geometry is the pool's only while
        prefill attention reads exactly what the Host stores.
        """
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        layout = getattr(self.parallel_manager, "prefill_pool_layout", None)
        return layout() if layout is not None else None

    def _prefill_pool_page_dims(self, worker, geometry) -> tuple:
        """(layers, heads, k_head_dim, v_head_dim) of one pool page row."""
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        layout = self._prefill_pool_psm_layout()
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

    def _prefill_pool_page_bytes(self, worker) -> int:
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        from batchgen.kv_cache.host_kv_mananger_config import build_gpu_kv_config

        page_tokens = int(self.prefix_cache_runtime_config.group_specs[0].raw_page_tokens)
        geometry = build_gpu_kv_config(self.huggingface_ckpt_name, [page_tokens])
        element = torch.empty((), dtype=geometry.kv_dtype).element_size()
        with_v, indexer_layers, indexer_dim = self._prefill_pool_layout(geometry)
        layers, heads, k_head_dim, v_head_dim = self._prefill_pool_page_dims(geometry)
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

    def _check_pool_plan_identical(self, worker, plan, group) -> None:
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
            torch.zeros_like(digest) for _ in range(self._decode_attn_tp_size())
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
        if not isinstance(self.host_paged_kv_worker_view, DualHostKVCoordinator):
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
            self._prefix.prefix_sequence_states[seq.global_idx] = state

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

    def _prefill_pool_segment_host_pages(self, worker) -> int:
        """Host pages this wave allocates for its segments.

        A materialized segment is excluded: its pages are the frontier's,
        already allocated and retained by the wave that computed it.
        """
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        wave = self._prefill_pool_wave
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
            self._prefix.prefix_sequence_states[global_id] = state
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

    def _pool_probe_debug(self, worker, wave):
        """The probe keys this wave armed, or None. Read ONCE per wave.

        The four pooled probes (epilogue trace, offload byte check, kernel dump,
        pad poison) are armed through the same batch-level seam as
        ``prefix_rank_assignment_mode``: the flag rides on the wave's own
        sequences, so a batch arms a probe against a server that is already up.

        Off is the production path and has to cost nothing measurable, so this
        is the ONLY place that walks the wave's sequences, the answer is cached
        on the wave, and the walk itself allocates nothing: a member whose
        ``batchgen_debug`` is absent costs one attribute read, and the result
        dict is not created until a probe key is actually found. Each probe's
        own setup then imports its module only after this returned non-None.
        """
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker

        if wave.probe_debug_read:
            return wave.probe_debug
        armed = None
        server_debug = self._batchgen_debug
        if server_debug:
            # Server-level ``--batchgen-debug`` wins outright, exactly as
            # _active_batchgen_debug_for_sequences has it.
            for key in _POOL_PROBE_DEBUG_KEYS:
                value = server_debug.get(key)
                if value is not None:
                    if armed is None:
                        armed = {}
                    armed[key] = value
        else:
            for uuid in wave.uuids:
                seq_debug = getattr(
                    self.global_batch.get_sequence(uuid), "batchgen_debug", None
                )
                if not isinstance(seq_debug, dict):
                    continue
                for key in _POOL_PROBE_DEBUG_KEYS:
                    value = seq_debug.get(key)
                    # First member that names a key wins, as the merge in
                    # _active_batchgen_debug_for_sequences does.
                    if value is None or (armed is not None and key in armed):
                        continue
                    if armed is None:
                        armed = {}
                    armed[key] = value
        wave.probe_debug = armed
        wave.probe_debug_read = True
        return armed

    def _new_pool_epilogue_tracer(self, worker, wave):
        """Batch-level ``batchgen_debug.prefix_pool_epilogue_trace``.

        Same seam as ``prefix_pool_offload_check`` and the kernel dump: the
        flag rides on the wave's own sequences. Traces the only stretch of the
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
        """Batch-level ``batchgen_debug.prefix_pool_dump_kernel_args``.

        Same seam as ``prefix_pool_offload_check``: the flag rides on the
        wave's own sequences. The value is the mode, ``"sync"`` or
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

    def _pool_pad_poison_enabled(self, worker, wave) -> bool:
        """Batch-level ``batchgen_debug.prefix_pool_pad_poison``.

        Read through the same seam as ``prefix_pool_offload_check``: the flag
        rides on the wave's own sequences, so a batch arms the probe against a
        server that is already up. False when the batch did not ask, which
        leaves the packed page table byte-identical to the default.

        The flag check comes BEFORE the import: on a production wave
        ``_pool_probe_debug`` has already answered None.
        """
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        debug = self._pool_probe_debug(wave)
        if debug is None:
            return False
        from batchgen.prefix_reuse.executor import PAD_POISON_DEBUG_FLAG

        if not self._debug_flag_enabled(debug.get(PAD_POISON_DEBUG_FLAG)):
            return False
        logging.info(
            "[PREFIX_POOL_PAD_POISON] rank=%s padding each page-table row "
            "with its own last page: members=%s segments=%s",
            self.rank,
            len(wave.uuids),
            len(wave.plan.segments),
        )
        return True

    def _new_pool_offload_check_recorder(self, worker, wave):
        """Batch-level ``batchgen_debug.prefix_pool_offload_check``.

        Read through the same seam as ``prefix_rank_assignment_mode``: the flag
        rides on the wave's own sequences, so a batch arms the check against a
        server that is already up and no env var exists for it. Returns None
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
                state = self._prefix.prefix_sequence_states[global_id]
                self._prefix.prefix_sequence_states[global_id] = replace(
                    state, commit_attachment_handles=(*state.commit_attachment_handles, handle)
                )
        finally:
            # Success or failure, the wave object is done. Segment Host pages
            # stay tracked in _pool_sid_members/_pool_member_sids, so after a
            # failed commit/attach the per-member release lane still frees and
            # unregisters every pseudo-sid.
            self._prefill_pool_wave = None

    def _release_pool_segment_refs(self, worker, global_id: int) -> None:
        """Release pool pseudo-sids through the PrefixReuseState owner."""
        # The moved body is kept VERBATIM from batchgen_worker.py: source pins
        # depend on its original self. spellings. Rebinding self to the worker
        # preserves those spellings; this engine instance remains stateless.
        self = worker
        self._prefix.release_pool_segment_refs(
            int(global_id), host_views_provider=self._host_views_by_group
        )

