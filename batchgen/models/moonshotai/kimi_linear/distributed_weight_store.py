"""Kimi-K3 compact distributed host-weight store helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List


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
    """Validate a node config AND the published store it describes.

    Every consumer (the server's daemon start, every worker's initializer)
    goes through here, so a store whose build never reached its completion
    manifest -- an interrupted build, a partial copy, a hand-made file -- is
    refused at this boundary rather than mapped with whatever bytes it holds.
    The manifest format and its checks live in ``store_builder`` (the one
    writer of the format); the import is local because that module imports
    this one.
    """
    config = read_distributed_weight_config(path)
    from .store_builder import verify_store_manifest

    verify_store_manifest(config, Path(path))
    return config


def read_distributed_weight_config(path: str | Path) -> dict[str, Any]:
    """Parse and schema-check a node config without judging the store bytes.

    Only :func:`load_distributed_weight_config` may hand a config to the
    engine; this half exists so the manifest verifier and the rebuild-command
    formatter can read a config whose store is known to be unusable.
    """
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
    # The store and metadata files themselves are judged by the completion
    # manifest check in load_distributed_weight_config, which also names the
    # rebuild command when they are missing, truncated, or foreign.
    return config


class DistributedMetadataError(ValueError):
    """A ``metadata.tsv`` row the engine's parser would reject."""


@dataclass(frozen=True)
class DistributedMetadataRow:
    """One ``T`` row of ``metadata.tsv``, as ``Weights_Storage.cpp`` reads it."""

    index: int
    module_key: str
    tensor_key: str
    owner: int
    compact_offset: int
    byte_size: int
    dtype: str
    shape: List[int]


def _routed_expert_index(module_key: str) -> int:
    """``parse_routed_expert`` (``Weights_Storage.cpp:53-58``): -1 if none."""
    prefix = "routed_expert_"
    if not module_key.startswith(prefix):
        return -1
    layer_text, separator, expert_text = module_key[len(prefix):].partition("_")
    if not separator or not layer_text.isdigit() or not expert_text.isdigit():
        return -1
    return int(expert_text)


def read_distributed_metadata(
    metadata_path: str | Path,
    *,
    store_bytes: int,
    num_nodes: int,
    n_experts: int,
    module_bytes: int,
) -> List[DistributedMetadataRow]:
    """Parse ``metadata.tsv`` under the engine's row contract.

    Mirrors ``Weights_Storage::InitDistributed`` (``Weights_Storage.cpp``,
    the ``std::getline`` loop and the remote-module pass after it): blank and
    ``H`` rows are skipped; every other row is nine tab-separated fields
    tagged ``T``; a tensor must end inside the store; ``owner`` is -1 or a
    node of this topology; an owned module must be the routed expert that
    ``expert // (n_experts / num_nodes)`` assigns to that owner; and every
    module's tensors must span exactly ``module_bytes`` from the module's
    lowest offset. The engine hardcodes ``n_experts`` (``kKimiK3Experts``);
    it is a parameter here so the builder's mini checkpoints can be read.
    """
    path = Path(metadata_path)
    if n_experts <= 0 or n_experts % num_nodes:
        raise DistributedMetadataError(
            f"{n_experts} experts do not divide across {num_nodes} nodes"
        )
    experts_per_owner = n_experts // num_nodes
    rows: List[DistributedMetadataRow] = []
    try:
        handle = path.open("r")
    except OSError as error:
        raise DistributedMetadataError(
            f"cannot open distributed weight metadata {path}: {error}"
        ) from error
    with handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            fields = line.split("\t") if line else []
            if not fields or fields[0] == "H":
                continue
            where = f"{path}:{line_number}"
            if len(fields) != 9 or fields[0] != "T":
                raise DistributedMetadataError(
                    f"{where}: invalid distributed weight metadata row"
                )
            try:
                index = int(fields[1])
                owner = int(fields[4])
                offset = int(fields[5])
                byte_size = int(fields[6])
                shape = json.loads(fields[8])
            except ValueError as error:
                raise DistributedMetadataError(
                    f"{where}: unparseable field: {error}"
                ) from error
            module_key, tensor_key, dtype = fields[2], fields[3], fields[7]
            if not isinstance(shape, list) or not all(
                isinstance(dim, int) and not isinstance(dim, bool)
                for dim in shape
            ):
                raise DistributedMetadataError(
                    f"{where}: shape {fields[8]} is not a list of integers"
                )
            if offset < 0 or byte_size < 0 or offset + byte_size > store_bytes:
                raise DistributedMetadataError(
                    f"{where}: distributed tensor exceeds compact store: "
                    f"{module_key}/{tensor_key}"
                )
            if owner < -1 or owner >= num_nodes:
                raise DistributedMetadataError(
                    f"{where}: distributed weight owner {owner} is outside "
                    f"[-1, {num_nodes}): {module_key}/{tensor_key}"
                )
            if owner >= 0:
                expert = _routed_expert_index(module_key)
                if expert < 0:
                    raise DistributedMetadataError(
                        f"{where}: distributed owned module is not a routed "
                        f"expert: {module_key}"
                    )
                if expert >= n_experts or owner != expert // experts_per_owner:
                    raise DistributedMetadataError(
                        f"{where}: distributed routed-expert owner mismatch: "
                        f"{module_key} owner={owner}"
                    )
            rows.append(
                DistributedMetadataRow(
                    index=index,
                    module_key=module_key,
                    tensor_key=tensor_key,
                    owner=owner,
                    compact_offset=offset,
                    byte_size=byte_size,
                    dtype=dtype,
                    shape=shape,
                )
            )

    module_spans: dict[str, tuple[int, int]] = {}
    for row in rows:
        if row.owner < 0:
            continue
        low, high = module_spans.get(row.module_key, (row.compact_offset, 0))
        module_spans[row.module_key] = (
            min(low, row.compact_offset),
            max(high, row.compact_offset + row.byte_size),
        )
    for module_key, (low, high) in module_spans.items():
        if high - low != module_bytes:
            raise DistributedMetadataError(
                f"{path}: remote module byte contract mismatch: {module_key} "
                f"spans {high - low} B, expected {module_bytes} B"
            )
    return rows
