"""Host-KV rebalancing — parallel-execution round grouping.

Slice 7 of the worker decouple initiative (issue #175). Extracts the
pure *scheduling decision* from ``_group_migrations_for_parallel_execution``:

  - ``HostKVRebalancer.group_migrations_for_parallel_execution`` — pack a
    flat migration list into rounds that can run concurrently without any
    rank participating in two transfers at once, and without two source
    ranks on the same node contending for that node's GPU-KV staging
    pages.

The migration *planner* (which sequences move where) was extracted in
Slice 5.6 (``KVCacheManager.plan_kv_migration``). The actual NCCL
transfers (``_execute_kv_migrations_parallel`` /
``_execute_single_kv_migration``) and the ``_rebalance_host_kv``
orchestration are irreducible side effects and stay on the worker; this
module owns only the conflict-free round-grouping decision.

Design follows the per-slice frozen-snapshot pattern: the inputs are the
already-planned ``MigrationOp`` values (themselves a value dataclass) plus
the node topology; the handler is pure and deterministic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Sequence

if TYPE_CHECKING:
    from batchgen.migration import MigrationOp


class HostKVRebalancer:
    """Host-KV migration scheduling — pure, deterministic."""

    @staticmethod
    def group_migrations_for_parallel_execution(
        migrations: Sequence["MigrationOp"], num_gpus_per_node: int
    ) -> List[List["MigrationOp"]]:
        """Group migrations into rounds that can execute in parallel.

        Within a round, no rank may appear as a source or destination of
        more than one migration (a rank can't send/recv simultaneously),
        and no source *node* may be reused — migration stages through GPU
        KV (host→GPU→extract→CPU→send), and source ranks on the same node
        share GPU-KV pages, so parallel migrations off one node would
        exhaust the staging buffer.

        Greedy, order-preserving: each round scans the remaining
        migrations in order and admits the first non-conflicting ones.
        Pure — no NCCL, no worker state.
        """
        rounds: List[List["MigrationOp"]] = []
        remaining = list(migrations)

        while remaining:
            round_migrations: List["MigrationOp"] = []
            used_ranks: set = set()
            used_src_nodes: set = set()

            for mig in remaining[:]:  # iterate over a copy; mutate `remaining`
                from_rank = mig.from_rank
                to_rank = mig.to_rank
                src_node = from_rank // num_gpus_per_node

                if (
                    from_rank not in used_ranks
                    and to_rank not in used_ranks
                    and src_node not in used_src_nodes
                ):
                    round_migrations.append(mig)
                    used_ranks.add(from_rank)
                    used_ranks.add(to_rank)
                    used_src_nodes.add(src_node)
                    remaining.remove(mig)

            rounds.append(round_migrations)

        return rounds
