"""Kimi-K3 compact distributed host-weight store helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


#: How a worker obtains this layer's 112-expert ingress shard.
#:
#: ``host_rdma`` (default, backward compatible) — every one of the 32 workers
#: pulls its own shard from its node's compact host store.
#: ``hierarchical_gdr`` — only eight source ranks pull from the host store and
#: replicate their shard to the other three nodes over dedicated cross-node
#: NCCL groups.
WEIGHT_TRANSPORTS = ("host_rdma", "hierarchical_gdr")


def load_distributed_weight_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = json.loads(config_path.read_text())
    required = {
        "node_rank",
        "node_ips",
        "workers",
        "store_path",
        "metadata_path",
        "daemon_socket",
        "summary_path",
        "store_bytes",
        "replicated_bytes",
        "module_bytes",
        "worker_sharded",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(
            f"distributed weight config {config_path} misses {missing}"
        )
    if len(config["node_ips"]) != 4:
        raise ValueError("distributed weight config requires four node IPs")
    if int(config["node_rank"]) not in range(4):
        raise ValueError("distributed weight node_rank must be in [0, 4)")
    if config["worker_sharded"] is not True:
        raise ValueError(
            "K3 distributed host weights require worker_sharded=true; "
            "each TP8 worker must acquire only its 112-expert ingress shard"
        )
    if int(config["workers"]) != 8:
        raise ValueError(
            "K3 distributed host weights require workers=8 per node"
        )
    # Optional; an existing config without the key keeps the validated
    # host-RDMA transport. Normalized into the returned dict so every caller
    # can read `config["transport"]` unconditionally.
    transport = config.get("transport", "host_rdma")
    if transport not in WEIGHT_TRANSPORTS:
        raise ValueError(
            f"distributed weight transport must be one of {WEIGHT_TRANSPORTS}; "
            f"got {config.get('transport')!r}"
        )
    config["transport"] = transport
    store = Path(config["store_path"])
    if store.stat().st_size != int(config["store_bytes"]):
        raise ValueError("distributed compact-store byte size mismatch")
    metadata = Path(config["metadata_path"])
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    return config
