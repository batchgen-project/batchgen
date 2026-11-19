"""Server config: the runtime configuration a worker cohort serves with.

Monitoring needs the server to say, from its own logs, what it is running:
the attention / MoE parallelism per phase (declared by the model's
parallel-strategy manager, see ``batchgen.parallel_declaration``), the
CUDA-graph configuration with the owning model's bucket ladder, the host- and
device-KV pool sizes, the GPU memory fraction and the derived scheduling
pool capacity. All of it is owned by worker-side objects, so — exactly like
the code identity — local rank 0 of every node snapshots it at the end of
its start-up, it travels with the deployment-capacity publication, and the
server logs it once per worker generation and node (``ServerConfig.log_line``)
before the generation serves its first request. A multi-node deployment
therefore prints one line per node, each under its own ``node=`` value.

Every value is reported as its owner holds it, never re-derived:

``prefill_attn`` / ``prefill_moe`` / ``decode_attn`` / ``decode_moe``
    The model's ``ParallelDeclaration`` slots, rendered; ``undeclared`` for
    a manager without the contract (never a guessed default).
``cuda_graphs`` / ``graph_buckets``
    The owning model's decode-graph declaration and its installed or captured
    bucket ladder; ``none`` when no decode graph is declared.
``host_kv_gib`` / ``host_kv_pages``
    The node's configured host-KV budget and its shared region's total page
    count (per region; a dual-region DSA pool keeps equal page counts).
``device_kv_gib`` / ``device_kv_pages``
    The measured device-KV pool of this node's rank 0 device, and its paged
    manager's total page count (per region, as above).
``gpu_memory_frac``
    The launch argument the pool was measured under.
``max_pool``
    The scheduling capacity the trajectory pool derived: the same number the
    worker publishes as ``total_capacity``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from batchgen.parallel_declaration import UNDECLARED

LOG_TAG = "[SERVER_CONFIG]"

# A rendered parallelism slot: a ParallelDeclaration token, or UNDECLARED.
_SLOT_TOKEN = re.compile(
    r"none|dp:(?:resident|streamed|partial)|(?:tp|ep)\d+:(?:resident|streamed|partial)"
)

# The line's key order is the dataclass field order below; parsed by tooling,
# so every field is always present and no value contains a space.
_FIELDS = (
    "prefill_attn",
    "prefill_moe",
    "decode_attn",
    "decode_moe",
    "cuda_graphs",
    "graph_buckets",
    "host_kv_gib",
    "host_kv_pages",
    "device_kv_gib",
    "device_kv_pages",
    "gpu_memory_frac",
    "max_pool",
)


class ServerConfigError(ValueError):
    """A server-config publication outside this module's contract."""


def _check_slot(name: str, value: Any) -> None:
    if value == UNDECLARED:
        return
    if not isinstance(value, str) or not _SLOT_TOKEN.fullmatch(value):
        raise ServerConfigError(
            f"server config {name} {value!r} is neither {UNDECLARED!r} nor a "
            "rendered ModuleParallelism token"
        )


def _check_count(name: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ServerConfigError(
            f"server config {name} must be a non-negative int, got {value!r}"
        )


def _check_gib(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ServerConfigError(
            f"server config {name} must be a non-negative number, got {value!r}"
        )


@dataclass(frozen=True)
class ServerConfig:
    prefill_attn: str
    prefill_moe: str
    decode_attn: str
    decode_moe: str
    cuda_graphs: bool
    # The owning decode-graph ladder; None when no graph is declared.
    graph_buckets: Optional[Tuple[int, ...]]
    host_kv_gib: float
    host_kv_pages: int
    device_kv_gib: float
    device_kv_pages: int
    gpu_memory_frac: float
    max_pool: int

    def __post_init__(self) -> None:
        for name in ("prefill_attn", "prefill_moe", "decode_attn", "decode_moe"):
            _check_slot(name, getattr(self, name))
        if not isinstance(self.cuda_graphs, bool):
            raise ServerConfigError(
                f"server config cuda_graphs must be a bool, got {self.cuda_graphs!r}"
            )
        if self.graph_buckets is not None:
            buckets = self.graph_buckets
            if (
                not isinstance(buckets, tuple)
                or not buckets
                or any(
                    not isinstance(b, int) or isinstance(b, bool) or b <= 0
                    for b in buckets
                )
                or list(buckets) != sorted(set(buckets))
            ):
                raise ServerConfigError(
                    "server config graph_buckets must be a strictly increasing "
                    f"non-empty tuple of positive ints, got {buckets!r}"
                )
            if not self.cuda_graphs:
                raise ServerConfigError(
                    "server config cannot carry captured graph_buckets with "
                    "cuda_graphs off"
                )
        for name in ("host_kv_pages", "device_kv_pages", "max_pool"):
            _check_count(name, getattr(self, name))
        for name in ("host_kv_gib", "device_kv_gib"):
            _check_gib(name, getattr(self, name))
        frac = self.gpu_memory_frac
        if isinstance(frac, bool) or not isinstance(frac, (int, float)) or not 0 < frac <= 1:
            raise ServerConfigError(
                f"server config gpu_memory_frac must be in (0, 1], got {frac!r}"
            )

    def as_dict(self) -> Dict[str, Any]:
        payload = {name: getattr(self, name) for name in _FIELDS}
        if self.graph_buckets is not None:
            payload["graph_buckets"] = list(self.graph_buckets)
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> "ServerConfig":
        if not isinstance(payload, dict) or set(payload) != set(_FIELDS):
            raise ServerConfigError(
                f"server config must be an object with exactly {list(_FIELDS)}, "
                f"got {payload!r:.300}"
            )
        values = dict(payload)
        buckets = values["graph_buckets"]
        if buckets is not None:
            if not isinstance(buckets, (list, tuple)):
                raise ServerConfigError(
                    f"server config graph_buckets must be a list, got {buckets!r}"
                )
            values["graph_buckets"] = tuple(buckets)
        return cls(**values)

    def log_line(self, generation: int, node: int) -> str:
        """The one server-log line of this config; parsed by tooling, so
        every field is always present and no value contains a space."""
        buckets = (
            "none"
            if self.graph_buckets is None
            else ",".join(str(b) for b in self.graph_buckets)
        )
        return (
            f"{LOG_TAG} generation={int(generation)} node={int(node)} "
            f"prefill_attn={self.prefill_attn} prefill_moe={self.prefill_moe} "
            f"decode_attn={self.decode_attn} decode_moe={self.decode_moe} "
            f"cuda_graphs={'on' if self.cuda_graphs else 'off'} "
            f"graph_buckets={buckets} "
            f"host_kv_gib={self.host_kv_gib:.3f} host_kv_pages={self.host_kv_pages} "
            f"device_kv_gib={self.device_kv_gib:.3f} "
            f"device_kv_pages={self.device_kv_pages} "
            f"gpu_memory_frac={self.gpu_memory_frac:g} max_pool={self.max_pool}"
        )
