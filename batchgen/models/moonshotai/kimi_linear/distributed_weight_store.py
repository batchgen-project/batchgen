"""Kimi-K3 compact distributed host-weight store helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
    store = Path(config["store_path"])
    if store.stat().st_size != int(config["store_bytes"]):
        raise ValueError("distributed compact-store byte size mismatch")
    metadata = Path(config["metadata_path"])
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    return config
