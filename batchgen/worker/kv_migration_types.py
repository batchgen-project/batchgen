"""KV Cache migration data structures for multi-node load balancing.

Classes:
    MigrationOp: One planned KV migration (one sequence, one source/target rank)
    HostKVStats: Host KV cache utilization statistics for one rank
"""

from dataclasses import dataclass


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
