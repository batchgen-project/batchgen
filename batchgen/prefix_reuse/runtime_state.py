"""Runtime state and helpers for page-level prefix KV reuse."""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, MutableMapping, Optional, Set, Tuple

import torch

from batchgen.prefill.prefix_reuse import (
    PrefixReusePrefillPlan,
    build_prefix_reuse_prefill_plan,
    validate_prefix_reuse_plan,
)
from batchgen.prefix_cache_utils import clear_rank_cache_if_prefix_evicted
from batchgen.sequence import SequenceBatch, SequenceEntry

PrefixRankKeyCacheEntry = Tuple[int, int, int, int]


@dataclass
class PrefixReuseRuntimeState:
    namespace_hash: int
    allocations_by_global_id: Dict[int, dict] = field(default_factory=dict)
    prompt_rank_cache: Dict[int, int] = field(default_factory=dict)
    prompt_rank_key_cache: Dict[int, PrefixRankKeyCacheEntry] = (
        field(default_factory=dict)
    )
    rank_cache_epoch: int = 0
    prefill_stats: MutableMapping[str, int] = field(
        default_factory=lambda: {
            "total_prompt_tokens": 0,
            "total_suffix_tokens": 0,
            "prefix_tokens_skipped": 0,
            "full_hit_guarded_errors": 0,
            "full_hit_exact_paths": 0,
            "full_hit_tokens_computed": 0,
            "fallback_full_prefill_tokens": 0,
        }
    )


class PrefixReuseRuntime:
    """Owns worker-local prefix reuse caches and worker-facing helpers."""

    def __init__(
        self,
        *,
        enabled: bool,
        model_name: str,
        kv_dtype: str,
        page_size: int,
        rank: int,
        world_size: int,
        torch_device: torch.device,
    ) -> None:
        self.enabled = bool(enabled)
        self.model_name = model_name
        self.kv_dtype = kv_dtype
        self.page_size = int(page_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.torch_device = torch_device
        self.state = PrefixReuseRuntimeState(
            namespace_hash=self.build_namespace_hash(
                model_name=model_name,
                kv_dtype=kv_dtype,
                page_size=page_size,
            )
        )

    @staticmethod
    def build_namespace_hash(
        *,
        model_name: str,
        kv_dtype: str,
        page_size: int,
    ) -> int:
        material = (
            f"model={model_name}|kv_dtype={kv_dtype}|"
            f"page_size={int(page_size)}"
        ).encode("utf-8")
        return int.from_bytes(
            hashlib.blake2b(material, digest_size=8).digest(),
            "little",
        )

    def prompt_tokens(self, seq: SequenceEntry) -> List[int]:
        if seq.input_ids is None:
            raise ValueError(f"Sequence {seq.uuid} has no input_ids for prefix reuse")
        prompt = seq.input_ids[0, : seq.prompt_length].detach().cpu()
        return [int(token) for token in prompt.tolist()]

    def prompt_rank_key(self, seq: SequenceEntry) -> Optional[int]:
        """Hash full prefix-cache pages for rank-affinity scheduling."""
        if not self.enabled or seq.input_ids is None:
            return None
        prompt_len = int(getattr(seq, "prompt_length", 0) or 0)
        page_tokens = (prompt_len // self.page_size) * self.page_size
        if page_tokens <= 0:
            return None

        cache_entry = self.state.prompt_rank_key_cache.get(seq.global_idx)
        if cache_entry is not None:
            cached_page_tokens, cached_page_size, cached_namespace, cached_key = (
                cache_entry
            )
            if (
                cached_page_tokens == page_tokens
                and cached_page_size == self.page_size
                and cached_namespace == self.state.namespace_hash
            ):
                return cached_key

        prompt = seq.input_ids[0, :page_tokens].detach().cpu().tolist()
        hasher = hashlib.blake2b(digest_size=16)
        hasher.update(int(self.state.namespace_hash).to_bytes(8, "little"))
        hasher.update(int(self.page_size).to_bytes(4, "little"))
        hasher.update(int(page_tokens // self.page_size).to_bytes(4, "little"))
        for token in prompt:
            hasher.update(int(token).to_bytes(8, "little", signed=True))
        key = int.from_bytes(hasher.digest(), "little")
        self.state.prompt_rank_key_cache[seq.global_idx] = (
            page_tokens,
            self.page_size,
            self.state.namespace_hash,
            key,
        )
        return key

    def maybe_clear_rank_cache_after_eviction(self, worker_view: object) -> None:
        self.state.rank_cache_epoch = clear_rank_cache_if_prefix_evicted(
            enable_prefix_reuse=self.enabled,
            worker_view=worker_view,
            prompt_rank_cache=self.state.prompt_rank_cache,
            current_epoch=self.state.rank_cache_epoch,
            rank=self.rank,
            logger=logging.getLogger(__name__),
        )

    def cached_rank_for_sequence(
        self,
        seq: SequenceEntry,
        *,
        existing_sequences: Iterable[SequenceEntry],
        pending_uuids: Set[str],
        rank_hint_index: Optional[Dict[int, int]] = None,
    ) -> Optional[int]:
        """Return the rank that already owns a compatible prefix cache entry."""
        key = self.prompt_rank_key(seq)
        if key is None:
            return None

        cached_rank = self.state.prompt_rank_cache.get(key)
        if self._valid_rank(cached_rank):
            return int(cached_rank)
        if rank_hint_index is not None:
            cached_rank = rank_hint_index.get(key)
            if self._valid_rank(cached_rank):
                self.state.prompt_rank_cache[key] = int(cached_rank)
                return int(cached_rank)

        for existing in existing_sequences:
            if (
                existing.uuid == seq.uuid
                or existing.uuid in pending_uuids
                or existing.assigned_rank is None
            ):
                continue
            try:
                if self.prompt_rank_key(existing) == key:
                    rank = int(existing.assigned_rank)
                    self.state.prompt_rank_cache[key] = rank
                    return rank
            except Exception:
                continue
        return None

    def build_rank_hint_index(
        self,
        existing_sequences: Iterable[SequenceEntry],
        *,
        pending_uuids: Set[str],
    ) -> Dict[int, int]:
        """Build a per-admission prefix-key -> rank hint index."""
        rank_hint_index: Dict[int, int] = {}
        for key, rank in self.state.prompt_rank_cache.items():
            if self._valid_rank(rank):
                rank_hint_index[key] = int(rank)

        for existing in existing_sequences:
            if existing.uuid in pending_uuids or existing.assigned_rank is None:
                continue
            key = self.prompt_rank_key(existing)
            if key is None or key in rank_hint_index:
                continue
            rank_hint_index[key] = int(existing.assigned_rank)
        return rank_hint_index

    def commit_pages(
        self,
        *,
        prefill_uuids: List[str],
        global_batch: SequenceBatch,
        worker_view: object,
    ) -> None:
        if not self.enabled or self.exact_full_prefill_fallback_enabled():
            return
        if worker_view is None:
            return

        inserted_pages = 0
        committed_sequences = 0
        for uuid in prefill_uuids:
            seq = global_batch.get_sequence(uuid)
            if seq is None:
                continue
            key = self.prompt_rank_key(seq)
            if key is not None and seq.assigned_rank is not None:
                self.state.prompt_rank_cache[key] = int(seq.assigned_rank)
            if seq.assigned_rank != self.rank:
                continue
            inserted_pages += worker_view.commit_sequence_prefix_pages(
                seq.global_idx,
                self.prompt_tokens(seq),
                self.state.namespace_hash,
            )
            committed_sequences += 1

        if committed_sequences:
            stats = worker_view.get_prefix_cache_stats()
            logging.info(
                "Rank %s prefix reuse commit: sequences=%d inserted_pages=%d "
                "entries=%d saved_pages=%d lookup_hits=%d lookup_misses=%d "
                "shared_pages_attached=%d",
                self.rank,
                committed_sequences,
                inserted_pages,
                stats.entries,
                stats.host_pages_saved,
                stats.lookup_hits,
                stats.lookup_misses,
                stats.shared_pages_attached,
            )

    def record_allocations(
        self,
        *,
        allocations: Iterable[dict],
        prefill_uuids: List[str],
        global_batch: SequenceBatch,
    ) -> None:
        for allocation in allocations:
            sequence_id = int(allocation["sequence_id"])
            self.state.allocations_by_global_id[sequence_id] = dict(allocation)
            for uuid in prefill_uuids:
                seq = global_batch.get_sequence(uuid)
                if seq is not None and seq.global_idx == sequence_id:
                    seq.prefix_shared_tokens = int(
                        allocation.get("shared_prefix_tokens", 0)
                    )
                    break

    def clear_transient_allocation_state(self) -> None:
        self.state.allocations_by_global_id.clear()
        self.state.prompt_rank_key_cache.clear()

    def shared_tokens_for_sequence(
        self,
        seq: SequenceEntry,
        *,
        worker_view: object,
    ) -> int:
        if not self.enabled or self.exact_full_prefill_fallback_enabled():
            return 0
        cached_value = int(getattr(seq, "prefix_shared_tokens", 0) or 0)
        if cached_value > 0:
            return cached_value
        allocation = self.state.allocations_by_global_id.get(seq.global_idx)
        if allocation is not None:
            return int(allocation.get("shared_prefix_tokens", 0))
        if worker_view is None:
            return 0
        try:
            return int(worker_view.shared_prefix_tokens(seq.global_idx))
        except Exception:
            return 0

    def exact_full_prefill_fallback_enabled(self) -> bool:
        """Force full private prefill compute instead of prefix-reuse replay."""
        explicit = os.environ.get("BATCHGEN_PREFIX_REUSE_EXACT_FULL_PREFILL_FALLBACK")
        if explicit is not None:
            return explicit == "1"
        if os.environ.get("BATCHGEN_PREFIX_REUSE_ALLOW_UNSAFE_SUFFIX_COMPUTE", "0") == "1":
            return False
        if not torch.cuda.is_available():
            return False
        try:
            major, _minor = torch.cuda.get_device_capability(self.torch_device)
        except Exception:
            major, _minor = torch.cuda.get_device_capability()
        return major >= 12

    def runtime_enabled(self) -> bool:
        return bool(self.enabled and not self.exact_full_prefill_fallback_enabled())

    def sequence_uses_reused_prefix(
        self,
        seq: SequenceEntry,
        *,
        worker_view: object,
    ) -> bool:
        return bool(
            self.runtime_enabled()
            and self.shared_tokens_for_sequence(seq, worker_view=worker_view) > 0
        )

    @staticmethod
    def decode_rank_blocked(
        rank_counts: List[int],
        rank_has_reused_prefix: List[bool],
        assigned_rank: int,
        uses_reused_prefix: bool,
    ) -> bool:
        del rank_counts, rank_has_reused_prefix, assigned_rank, uses_reused_prefix
        return False

    def build_prefill_plan_for_batch(
        self,
        *,
        batch: List[int],
        local_to_uuid_map: Dict[int, str],
        global_batch: SequenceBatch,
        worker_view: object,
        compute_mode: str,
        allow_full_hits: bool = False,
        record_stats: bool = True,
    ) -> Optional[PrefixReusePrefillPlan]:
        """Build prefix prefill metadata and guard unsupported full-hit cases."""
        if not self.enabled or not batch:
            return None

        local_indices: List[int] = []
        sequence_ids: List[int] = []
        input_ids: List[torch.Tensor] = []
        prompt_lengths: List[int] = []
        shared_tokens: List[int] = []

        for local_idx in batch:
            uuid = local_to_uuid_map[local_idx]
            seq = global_batch.get_sequence(uuid)
            local_indices.append(local_idx)
            sequence_ids.append(seq.global_idx)
            input_ids.append(seq.input_ids)
            prompt_lengths.append(seq.prompt_length)
            shared_tokens.append(
                self.shared_tokens_for_sequence(seq, worker_view=worker_view)
            )

        plan = build_prefix_reuse_prefill_plan(
            local_indices=local_indices,
            sequence_ids=sequence_ids,
            input_ids=input_ids,
            prompt_lengths=prompt_lengths,
            prefix_shared_tokens=shared_tokens,
            device=torch.device("cpu"),
        )
        try:
            validate_prefix_reuse_plan(plan, allow_full_hits=allow_full_hits)
        except RuntimeError:
            self.state.prefill_stats["full_hit_guarded_errors"] += 1
            raise
        if plan.saved_prefill_tokens <= 0:
            return None

        if record_stats:
            self.state.prefill_stats["total_prompt_tokens"] += plan.total_prompt_tokens
            self.state.prefill_stats["total_suffix_tokens"] += plan.total_suffix_tokens
            if compute_mode == "suffix_compute":
                self.state.prefill_stats["prefix_tokens_skipped"] += (
                    plan.saved_prefill_tokens
                )
            else:
                self.state.prefill_stats["fallback_full_prefill_tokens"] += (
                    plan.total_prompt_tokens
                )

        if plan.saved_prefill_tokens > 0:
            logging.info(
                "Rank %s prefix reuse prefill plan: prompt_tokens=%d "
                "suffix_tokens=%d prefix_tokens=%d mode=%s",
                self.rank,
                plan.total_prompt_tokens,
                plan.total_suffix_tokens,
                plan.saved_prefill_tokens,
                compute_mode,
            )
        return plan

    def _valid_rank(self, rank: Optional[int]) -> bool:
        return rank is not None and 0 <= int(rank) < self.world_size
