"""Kimi-K3 compact distributed host-weight store helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


#: How a worker obtains this layer's 112-expert ingress shard.
#:
#: ``host_rdma`` (default, backward compatible) — every worker pulls its own
#: shard from its node's compact host store.
#: ``hierarchical_gdr`` — only eight source ranks pull from the host store and
#: replicate their shard to the remaining nodes over dedicated cross-node
#: NCCL groups.
WEIGHT_TRANSPORTS = ("host_rdma", "hierarchical_gdr")

#: Supported node counts. Each node runs eight TP8 workers, so these are the
#: world16 (2 nodes) and world32 (4 nodes) topologies.
DISTRIBUTED_NODE_COUNTS = (2, 4)

#: Eight UCX rails used by the original H20 deployment. Configs may override
#: these for fleets whose HCA numbering differs (for example H200 uses 0..7).
DEFAULT_RAIL_DEVICES = tuple(f"mlx5_bond_{index}:1" for index in range(1, 9))


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
    num_nodes = len(config["node_ips"])
    if num_nodes not in DISTRIBUTED_NODE_COUNTS:
        raise ValueError(
            "distributed weight config requires exactly "
            f"{' or '.join(str(n) for n in DISTRIBUTED_NODE_COUNTS)} node "
            f"IPs; got {num_nodes}"
        )
    if "num_nodes" in config and int(config["num_nodes"]) != num_nodes:
        raise ValueError(
            f"distributed weight num_nodes {config['num_nodes']} disagrees "
            f"with {num_nodes} node IPs"
        )
    # Normalized so every caller can read `config["num_nodes"]` instead of
    # re-deriving the topology from the IP list.
    config["num_nodes"] = num_nodes
    if int(config["node_rank"]) not in range(num_nodes):
        raise ValueError(
            f"distributed weight node_rank must be in [0, {num_nodes})"
        )
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
    rail_devices = config.get("rail_devices", list(DEFAULT_RAIL_DEVICES))
    if (
        not isinstance(rail_devices, list)
        or len(rail_devices) != len(DEFAULT_RAIL_DEVICES)
        or any(
            not isinstance(device, str) or not device
            for device in rail_devices
        )
    ):
        raise ValueError(
            "distributed weight rail_devices must contain exactly eight "
            "non-empty strings"
        )
    config["rail_devices"] = rail_devices
    store = Path(config["store_path"])
    if store.stat().st_size != int(config["store_bytes"]):
        raise ValueError("distributed compact-store byte size mismatch")
    metadata = Path(config["metadata_path"])
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    return config
