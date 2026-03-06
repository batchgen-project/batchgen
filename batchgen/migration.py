"""KV Cache migration utilities for multi-node load balancing.

This module provides functionality to migrate KV cache data between nodes
to balance host memory utilization across a distributed inference cluster.

Classes:
    KVMigrationHelper: Manages KV cache migration operations
"""

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import torch
import torch.distributed as dist

from batchgen.sequence import SequenceStatus

logger = logging.getLogger(__name__)

# Constants for migration
NUM_GPUS_PER_NODE = 8  # TODO: Make configurable
HOST_KV_WATERMARK_PERCENT = 70  # Trigger prefill when >70% free


@dataclass
class MigrationOp:
    """A single KV migration operation."""
    uuid: str
    from_rank: int
    to_rank: int
    pages: int
    host_pages: int = 0  # Actual host pages allocated (may differ from pages during chunked growth)


@dataclass
class HostKVStats:
    """Host KV cache utilization statistics."""
    rank: int
    node_id: int
    num_free_pages: int
    num_total_pages: int
    num_used_pages: int
    free_percent: int
    num_in_decode: int = 0
    num_onhold: int = 0
    num_prefilled: int = 0
    num_valid_sequences: int = 0


class KVMigrationHelper:
    """Helper class for managing KV cache migrations between nodes.

    This class encapsulates all KV migration logic, including:
    - Planning migrations to rebalance host KV across nodes
    - Executing migrations in parallel using Gloo backend
    - Managing Gloo process groups for CPU tensor transfers
    - Tracking migration state and updating sequence ownership

    The helper requires a reference to the BatchGenWorker to access:
    - global_batch: Sequence metadata
    - host_paged_kv_worker_view: Host KV cache access
    - gpu_paged_kv_cache_manager: GPU KV cache access
    - query_book: Query data
    """

    def __init__(
        self,
        worker: Any,  # BatchGenWorker instance
        enable_decode_preemption: bool = True,
        debug: bool = False,
    ):
        """Initialize the migration helper.

        Args:
            worker: BatchGenWorker instance
            enable_decode_preemption: Whether to enable decode preemption (interrupt decode for prefill)
            debug: Enable verbose debug logging
        """
        self.worker = worker
        self.enable_decode_preemption = enable_decode_preemption
        self.debug = debug

        # Gloo process group for CPU tensor migrations
        self._gloo_migration_group: Optional[Any] = None

        # Round-robin counter for destination rank selection
        self._dest_rank_counter: Dict[int, int] = {}

        # Pending migration data
        self._pending_migrated_query_book: Dict[str, Dict] = {}
        self._migrated_sequences: Set[str] = set()

    @property
    def rank(self) -> int:
        return self.worker.rank

    @property
    def world_size(self) -> int:
        return self.worker.world_size

    @property
    def local_rank(self) -> int:
        return self.worker.local_rank

    @property
    def PAGE_SIZE(self) -> int:
        return self.worker.PAGE_SIZE

    def get_or_create_gloo_group(self):
        """Get or create a Gloo process group for CPU tensor migrations.

        Gloo backend supports CPU tensors and can use RDMA if available.
        This is more memory efficient than NCCL (which requires GPU staging).

        Returns:
            The Gloo process group for CPU tensor operations.
        """
        if self._gloo_migration_group is None:
            logger.debug(f"Rank {self.rank}: Creating Gloo process group for CPU migrations")
            self._gloo_migration_group = dist.new_group(
                ranks=list(range(self.world_size)),
                backend="gloo"
            )
            logger.debug(f"Rank {self.rank}: Gloo process group created")
        return self._gloo_migration_group

    def destroy_gloo_group(self):
        """Destroy the Gloo process group after migrations are done."""
        if self._gloo_migration_group is not None:
            logger.debug(f"Rank {self.rank}: Destroying Gloo process group")
            dist.destroy_process_group(self._gloo_migration_group)
            self._gloo_migration_group = None

    def get_host_kv_utilization(self) -> HostKVStats:
        """Get host KV stats counting sequences with KV in host memory.

        Valid sequences = PREFILLED, ON_HOLD, and IN_DECODE (all have KV in host).
        Host KV is shared per-node, so we count sequences from ALL ranks on this node.

        Returns:
            HostKVStats with utilization information
        """
        stats = self.worker.host_paged_kv_worker_view.get_stats()

        # Count pages used by sequences with KV in host on THIS NODE
        node_id = self.rank // NUM_GPUS_PER_NODE
        node_rank_start = node_id * NUM_GPUS_PER_NODE
        node_rank_end = min(node_rank_start + NUM_GPUS_PER_NODE, self.world_size)

        valid_statuses = {SequenceStatus.PREFILLED, SequenceStatus.ON_HOLD, SequenceStatus.IN_DECODE}

        # Count sequences per status
        status_counts = {status: [] for status in valid_statuses}
        for rank_on_node in range(node_rank_start, node_rank_end):
            for status in valid_statuses:
                seqs = self.worker.global_batch.get_sequences_for_rank_with_status(rank_on_node, status)
                status_counts[status].extend(seqs)

        valid_sequences = []
        for seqs in status_counts.values():
            valid_sequences.extend(seqs)

        # Calculate pages used
        used_pages = 0
        for uuid in valid_sequences:
            seq = self.worker.global_batch.get_sequence(uuid)
            pages_needed = math.ceil(seq.kv_token_budget / self.PAGE_SIZE)
            used_pages += pages_needed

        free_pages = stats.num_total_pages - used_pages
        free_percent = int((free_pages / stats.num_total_pages) * 100) if stats.num_total_pages > 0 else 100

        return HostKVStats(
            rank=self.rank,
            node_id=node_id,
            num_free_pages=free_pages,
            num_total_pages=stats.num_total_pages,
            num_used_pages=used_pages,
            free_percent=free_percent,
            num_in_decode=len(status_counts[SequenceStatus.IN_DECODE]),
            num_onhold=len(status_counts[SequenceStatus.ON_HOLD]),
            num_prefilled=len(status_counts[SequenceStatus.PREFILLED]),
            num_valid_sequences=len(valid_sequences),
        )

    def check_watermark_trigger(self) -> bool:
        """Check if any node exceeds host KV free page watermark.

        Watermark = 70% FREE (underutilized).
        Only checks if this rank is local_rank 0 (one check per node).

        Returns:
            True if should interrupt decode and switch to prefill
        """
        if not self.enable_decode_preemption:
            return False

        # Only local_rank 0 reports (one per node)
        if self.local_rank == 0:
            local_stats = self.get_host_kv_utilization()
            local_stats_dict = {
                'rank': local_stats.rank,
                'node_id': local_stats.node_id,
                'num_free_pages': local_stats.num_free_pages,
                'num_total_pages': local_stats.num_total_pages,
                'num_used_pages': local_stats.num_used_pages,
                'free_percent': local_stats.free_percent,
            }
        else:
            local_stats_dict = None

        # Gather stats from all local_rank 0 representatives
        all_stats = [None] * self.world_size
        dist.all_gather_object(all_stats, local_stats_dict)

        # Filter to only node representatives
        node_stats = [s for s in all_stats if s is not None]

        if not node_stats:
            return False

        # Check if any node above watermark (too much free space)
        max_free_percent = max(s['free_percent'] for s in node_stats)
        above_watermark = max_free_percent > HOST_KV_WATERMARK_PERCENT

        # Check if queued sequences available
        has_queued = self.worker.global_batch.has_queueing()

        should_trigger = above_watermark and has_queued

        if self.rank == 0 and should_trigger:
            logger.info(
                f"[Host KV Cache] PREFILL TRIGGER: max_node_free={max_free_percent}% > {HOST_KV_WATERMARK_PERCENT}%, "
                f"queued_sequences={len(self.worker.global_batch.get_sequences_by_status(SequenceStatus.QUEUEING))}"
            )

        return should_trigger

    def plan_migrations(self) -> List[MigrationOp]:
        """Plan sequence migrations to rebalance host KV across nodes.

        Returns:
            List of MigrationOp objects describing planned migrations
        """
        # Gather host KV stats from all local_rank 0
        if self.local_rank == 0:
            local_stats = self.get_host_kv_utilization()
            local_stats_dict = {
                'node_id': local_stats.node_id,
                'num_used_pages': local_stats.num_used_pages,
            }
        else:
            local_stats_dict = None

        all_stats = [None] * self.world_size
        dist.all_gather_object(all_stats, local_stats_dict)
        node_stats = {s['node_id']: s for s in all_stats if s is not None}

        if len(node_stats) <= 1:
            if self.rank == 0:
                logger.info("MIGRATION: Single node detected, skipping rebalancing")
            return []

        # Calculate target pages per node
        total_used = sum(s['num_used_pages'] for s in node_stats.values())
        num_nodes = len(node_stats)
        target_per_node = total_used // num_nodes

        if self.rank == 0:
            logger.info(
                f"MIGRATION: Planning rebalance: {total_used} total pages across {num_nodes} nodes, "
                f"target {target_per_node} pages/node"
            )

        # Identify overloaded and underutilized nodes
        overloaded = [(nid, s) for nid, s in node_stats.items() if s['num_used_pages'] > target_per_node]
        underutilized = [(nid, s) for nid, s in node_stats.items() if s['num_used_pages'] < target_per_node]

        if not overloaded or not underutilized:
            if self.rank == 0:
                logger.info("MIGRATION: Already balanced, no migrations needed")
            return []

        overloaded.sort(key=lambda x: x[1]['num_used_pages'], reverse=True)
        underutilized.sort(key=lambda x: x[1]['num_used_pages'])

        # Greedy migration planning
        migrations = []
        used_by_node = {nid: s['num_used_pages'] for nid, s in node_stats.items()}
        migrated_uuids: Set[str] = set()
        self._dest_rank_counter = {}

        for src_node_id, _ in overloaded:
            while used_by_node[src_node_id] > target_per_node and underutilized:
                # Find sequences to migrate from src_node
                src_rank_base = src_node_id * NUM_GPUS_PER_NODE
                candidate_sequences = []
                for gpu_offset in range(NUM_GPUS_PER_NODE):
                    src_rank = src_rank_base + gpu_offset
                    if src_rank >= self.world_size:
                        break
                    for status in [SequenceStatus.PREFILLED, SequenceStatus.ON_HOLD]:
                        for uuid in self.worker.global_batch.get_sequences_for_rank_with_status(src_rank, status):
                            if uuid not in migrated_uuids:
                                candidate_sequences.append(uuid)

                if not candidate_sequences:
                    break

                # Sort deterministically
                candidate_sequences.sort(
                    key=lambda u: self.worker.global_batch.get_sequence(u).global_idx
                )

                # Pick smallest sequence
                uuid = min(candidate_sequences, key=lambda u: (
                    self.worker.global_batch.get_sequence(u).kv_token_budget,
                    self.worker.global_batch.get_sequence(u).global_idx
                ))
                seq = self.worker.global_batch.get_sequence(uuid)
                pages_needed = math.ceil(seq.kv_token_budget / self.PAGE_SIZE)

                # Find dest node with most free space
                dest_node_id = min(underutilized, key=lambda x: (used_by_node[x[0]], x[0]))[0]

                # Distribute across ranks on dest node
                if dest_node_id not in self._dest_rank_counter:
                    self._dest_rank_counter[dest_node_id] = 0

                dest_rank_offset = self._dest_rank_counter[dest_node_id] % NUM_GPUS_PER_NODE
                dest_rank = dest_node_id * NUM_GPUS_PER_NODE + dest_rank_offset
                if dest_rank >= self.world_size:
                    dest_rank = dest_node_id * NUM_GPUS_PER_NODE
                self._dest_rank_counter[dest_node_id] += 1

                migrations.append(MigrationOp(
                    uuid=uuid,
                    from_rank=seq.assigned_rank,
                    to_rank=dest_rank,
                    pages=pages_needed
                ))

                migrated_uuids.add(uuid)
                used_by_node[src_node_id] -= pages_needed
                used_by_node[dest_node_id] += pages_needed

                if used_by_node[dest_node_id] >= target_per_node:
                    underutilized = [(nid, s) for nid, s in underutilized if nid != dest_node_id]

        if self.rank == 0 and migrations:
            logger.info(f"MIGRATION: Planned {len(migrations)} sequence migrations")

        return migrations

    def group_migrations_for_parallel_execution(
        self, migrations: List[MigrationOp]
    ) -> List[List[MigrationOp]]:
        """Group migrations into rounds that can execute in parallel.

        Migrations in the same round must not share any source or destination ranks.

        Args:
            migrations: List of MigrationOp objects

        Returns:
            List of rounds, where each round is a list of migrations
        """
        rounds = []
        remaining = list(migrations)

        while remaining:
            round_migrations = []
            used_ranks: Set[int] = set()

            for mig in remaining[:]:
                if mig.from_rank not in used_ranks and mig.to_rank not in used_ranks:
                    round_migrations.append(mig)
                    used_ranks.add(mig.from_rank)
                    used_ranks.add(mig.to_rank)
                    remaining.remove(mig)

            rounds.append(round_migrations)

        return rounds

    def execute_migrations_parallel(self, migrations: List[MigrationOp]) -> None:
        """Execute multiple KV migrations in parallel.

        Groups migrations by independent rank pairs and executes them concurrently.

        Args:
            migrations: List of MigrationOp objects
        """
        if not migrations:
            return

        # Create Gloo group (collective operation)
        self.get_or_create_gloo_group()
        dist.barrier()

        # Group into parallel rounds
        rounds = self.group_migrations_for_parallel_execution(migrations)

        if self.rank == 0:
            logger.info(f"MIGRATION: Executing {len(migrations)} migrations in {len(rounds)} parallel rounds")

        for round_idx, round_migrations in enumerate(rounds):
            if self.rank == 0:
                logger.info(f"MIGRATION: Round {round_idx+1}/{len(rounds)}: {len(round_migrations)} parallel migrations")

            # Find if this rank participates
            my_migration = None
            for mig in round_migrations:
                if self.rank == mig.from_rank or self.rank == mig.to_rank:
                    my_migration = mig
                    break

            if my_migration is not None:
                self._execute_single_migration(my_migration)

            dist.barrier()

        if self.rank == 0:
            logger.info(f"MIGRATION: All {len(rounds)} parallel rounds completed")

    def _execute_single_migration(self, mig: MigrationOp) -> None:
        """Execute a single KV migration.

        This method handles both send (from_rank) and receive (to_rank) sides.

        Args:
            mig: MigrationOp describing the migration
        """
        seq = self.worker.global_batch.get_sequence(mig.uuid)
        if seq is None:
            logger.error(f"Rank {self.rank}: Cannot migrate {mig.uuid[:8]}... - sequence not found")
            return

        # Get KV tensor shape from GPU KV manager config
        gpu_kv_config = self.worker.gpu_paged_kv_cache_manager.config
        num_layers = self.worker.model_config.num_hidden_layers
        num_k_heads = gpu_kv_config.num_k_heads
        k_head_dim = gpu_kv_config.k_head_dim
        kv_dtype = gpu_kv_config.kv_dtype
        page_size = gpu_kv_config.page_size_tokens

        global_idx = seq.global_idx
        pages_needed = mig.pages

        k_shape = (num_layers, pages_needed, page_size, num_k_heads, k_head_dim)
        gloo_group = self.get_or_create_gloo_group()

        if self.rank == mig.from_rank:
            self._send_kv(mig, seq, k_shape, gloo_group, global_idx, pages_needed, page_size)
        elif self.rank == mig.to_rank:
            self._recv_kv(mig, seq, k_shape, kv_dtype, gloo_group, global_idx, pages_needed, page_size, num_layers, num_k_heads, k_head_dim)

    def _send_kv(self, mig: MigrationOp, seq, k_shape, gloo_group, global_idx, pages_needed, page_size):
        """Send KV cache data to destination rank."""
        t0 = time.perf_counter()
        if self.debug:
            logger.debug(f"MIGRATION: Rank {self.rank}: Send {mig.uuid[:8]}... → rank {mig.to_rank}")

        manager = self.worker.gpu_paged_kv_cache_manager
        worker_view = self.worker.host_paged_kv_worker_view

        if not manager.is_initialized:
            manager.initialize()

        tokens_needed = pages_needed * page_size
        manager.allocate_pages_for_sequences([global_idx], [tokens_needed])
        manager.rebuild_page_table([global_idx])

        # Load host KV → GPU
        sequence_tensor = torch.tensor([global_idx], dtype=torch.int64, device="cpu")
        k_ptrs, v_ptrs = manager.get_padded_3d_page_pointers()
        active_page_counts = manager.export_active_sequence_page_counts()

        load_task = worker_view.async_load_layer_paged_kv_to_device(
            sequence_ids=sequence_tensor,
            active_page_counts=active_page_counts,
            k_device_ptrs=k_ptrs,
            v_device_ptrs=v_ptrs,
        )
        load_task.wait()
        torch.cuda.synchronize(self.worker.torch_device)

        # Extract to contiguous tensor
        k_gpu = manager.copy_kv_to_tensor(global_idx)

        # Move to CPU for Gloo transfer
        k_cpu = k_gpu.cpu().contiguous()

        # Send via Gloo
        dist.send(tensor=k_cpu, dst=mig.to_rank, group=gloo_group)

        # Free GPU and host pages
        manager.free_pages_for_sequences([global_idx])
        worker_view.release_sequence_pages([global_idx])

        # Send query_book data
        local_idx = self.worker._uuid_to_local_map.get(mig.uuid)
        if local_idx is not None and local_idx in self.worker.query_book:
            qb = self.worker.query_book[local_idx]
            dist.send(tensor=qb.encoded["input_ids"].cpu().contiguous(), dst=mig.to_rank, group=gloo_group)
            dist.send(tensor=qb.encoded["attention_mask"].cpu().contiguous(), dst=mig.to_rank, group=gloo_group)
            dist.send(tensor=qb.decoded_tokens.cpu().contiguous(), dst=mig.to_rank, group=gloo_group)

        if self.debug:
            logger.debug(f"MIGRATION: Rank {self.rank}: Sent {mig.uuid[:8]}... in {(time.perf_counter()-t0)*1000:.1f}ms")

    def _recv_kv(self, mig: MigrationOp, seq, k_shape, kv_dtype, gloo_group, global_idx, pages_needed, page_size, num_layers, num_k_heads, k_head_dim):
        """Receive KV cache data from source rank."""
        t0 = time.perf_counter()
        if self.debug:
            logger.debug(f"MIGRATION: Rank {self.rank}: Recv {mig.uuid[:8]}... ← rank {mig.from_rank}")

        # Allocate CPU buffer
        k_cpu = torch.empty(k_shape, dtype=kv_dtype, device="cpu", pin_memory=True)

        # Receive via Gloo
        dist.recv(tensor=k_cpu, src=mig.from_rank, group=gloo_group)

        # Register and allocate host KV pages
        worker_view = self.worker.host_paged_kv_worker_view
        tokens_needed = pages_needed * page_size
        worker_view.register_sequences([global_idx])
        worker_view.allocate_pages_for_sequences([(global_idx, tokens_needed)])

        # Move to GPU for offload
        k_gpu = k_cpu.to(self.worker.device, non_blocking=True)
        torch.cuda.synchronize(self.worker.torch_device)

        # Offload layer-by-layer to host
        seq_len = pages_needed * page_size
        sequence_ids_list = [global_idx]
        sequence_lengths = [seq_len]

        for layer_idx in range(num_layers):
            layer_k = k_gpu[layer_idx]
            layer_k_flat = layer_k.reshape(seq_len, num_k_heads, k_head_dim)
            layer_k_batch = layer_k_flat.unsqueeze(0)

            worker_view.async_offload_layer_kv_to_host(
                layer_idx=layer_idx,
                sequence_ids=sequence_ids_list,
                k_tensor=layer_k_batch,
                v_tensor=None,
                sequence_lengths=sequence_lengths,
            )

        torch.cuda.synchronize(self.worker.torch_device)

        # Receive query_book data
        input_ids_recv = torch.empty(seq.input_ids.shape, dtype=seq.input_ids.dtype, device="cpu")
        attention_mask_recv = torch.empty(seq.attention_mask.shape, dtype=seq.attention_mask.dtype, device="cpu")
        decoded_tokens_recv = torch.empty(seq.decoded_tokens.shape, dtype=seq.decoded_tokens.dtype, device="cpu")

        dist.recv(tensor=input_ids_recv, src=mig.from_rank, group=gloo_group)
        dist.recv(tensor=attention_mask_recv, src=mig.from_rank, group=gloo_group)
        dist.recv(tensor=decoded_tokens_recv, src=mig.from_rank, group=gloo_group)

        # Store pending data
        self._pending_migrated_query_book[mig.uuid] = {
            'text': seq.text,
            'input_ids': input_ids_recv,
            'attention_mask': attention_mask_recv,
            'decoded_tokens': decoded_tokens_recv,
            'kv_token_budget': seq.kv_token_budget,
        }
        self._migrated_sequences.add(mig.uuid)

        if self.debug:
            logger.debug(f"MIGRATION: Rank {self.rank}: Recvd {mig.uuid[:8]}... in {(time.perf_counter()-t0)*1000:.1f}ms")

    def get_pending_query_book(self, uuid: str) -> Optional[Dict]:
        """Get pending query book data for a migrated sequence.

        Args:
            uuid: Sequence UUID

        Returns:
            Dict with query book data or None if not found
        """
        return self._pending_migrated_query_book.pop(uuid, None)

    def is_migrated_sequence(self, uuid: str) -> bool:
        """Check if a sequence was migrated to this rank.

        Args:
            uuid: Sequence UUID

        Returns:
            True if this sequence was migrated to this rank
        """
        return uuid in self._migrated_sequences
