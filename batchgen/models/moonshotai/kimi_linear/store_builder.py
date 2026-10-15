# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                         #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #

"""Build the Kimi-K3 compact distributed host-weight store.

One node of a K3 distributed deployment holds a single ``store.bin`` that
carries EVERY replicated tensor plus only the routed experts it owns, and a
``metadata.tsv`` that describes every tensor of every node.  This module turns
the marlin-converted checkpoint (per-shard ``*.json`` + ``*.bin``, the schema
``Parameter_Server.cpp:65-84`` parses) into those two files plus the per-node
config JSON the engine and the daemon read.

Layout
------
``store.bin`` is::

    [ replicated prefix ][ layer 1 local experts ][ layer 2 ... ] ... [ layer L ... ]

with no padding anywhere.  A routed-expert module begins at

    replicated_bytes
      + ((layer - 1) * experts_per_owner + expert % experts_per_owner)
        * module_bytes

which is exactly ``distributed_weight_daemon.cpp:1156-1164`` (``RemoteOffset``)
and is also the formula the engine's remote-module bookkeeping assumes
(``Weights_Storage.cpp:487-504``).  ``experts_per_owner = n_experts /
num_nodes`` and a node owns ``[rank * experts_per_owner, (rank + 1) *
experts_per_owner)`` (``Weights_Storage.cpp:418-423``).

Naming
------
The engine keys host weights by ``(module_key, tensor_key)``
(``Weights_Storage.cpp:443-446``, looked up at ``:758-759`` and ``:803-810``),
so the rows must carry the MAPPED names the replicated path produces, not the
raw checkpoint ``state_dict`` keys:

* a tensor in :func:`build_k3_state_dict_name_map` keeps that map's
  ``module_key``/``tensor_key`` (``tensor_map.py:382-438``) -- the same
  substitution ``Parameter_Server.cpp:359-368`` performs;
* every other tensor (skeleton and the explicitly ignored vision roots) goes
  into the single module ``__skeleton__`` under its RAW checkpoint name,
  because the worker rebuilds the skeleton state dict with
  ``get_tensor("__skeleton__")`` (``batchgen_worker.py:990-998``) and the
  replicated path keys ``skeleton_state_dict_`` by the checkpoint name
  (``Parameter_Server.cpp:398``, ``tensor_map.py:441-446``).

``summary.json`` is NOT written here: the daemon writes it at runtime
(``distributed_weight_daemon.cpp:1337-1378``) and only ever reads its path.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .distributed_weight_store import (
    DISTRIBUTED_NODE_COUNTS,
    WEIGHT_TRANSPORTS,
)
from .k3.mxfp4_layout import routed_expert_tensor_names
from .k3.tensor_map import (
    K3_CKPT_PREFIX,
    K3_KDA_TENSOR_NAMES,
    K3_MLA_TENSOR_NAMES,
    K3_SHARED_EXPERT_TENSOR_NAMES,
)

logger = logging.getLogger(__name__)


#: Released K3 routed-expert topology. The daemon hardcodes both
#: (``distributed_weight_daemon.cpp:51-52``) and so does the engine
#: (``Weights_Storage.cpp:45``), so a store built for any other shape would be
#: read with the wrong offsets. Overridable only through
#: :func:`build_store_plan`'s ``expected_topology``, never from the CLI.
K3_LAYERS = 92
K3_EXPERTS = 896

#: Eight TP8 workers per node (``distributed_weight_daemon.cpp:53``); the
#: loader rejects anything else (``distributed_weight_store.py:72-75``).
K3_WORKERS_PER_NODE = 8

#: The one module key that is not a routed expert and not a mapped ring module.
#: ``parse_routed_expert`` returns -1 for it (``Weights_Storage.cpp:53-58``), so
#: it passes the owner and pin guards as a replicated module.
SKELETON_MODULE_KEY = "__skeleton__"

#: ``owner`` column value for a tensor every node holds
#: (``Weights_Storage.cpp:404-410``).
REPLICATED_OWNER = -1

#: ``distributed_weight_daemon.cpp:338``.
DEFAULT_BASE_PORT = 26000

#: H200 HCA numbering. ``distributed_weight_store.DEFAULT_RAIL_DEVICES`` is
#: ``mlx5_bond_1:1``..``mlx5_bond_8:1``, which is the original H20 fleet's
#: numbering; the loader's own docstring records that H200 uses 0..7
#: (``distributed_weight_store.py:24-26``). The builder writes the list
#: explicitly so the loader default never applies.
H200_RAIL_DEVICES = tuple(f"mlx5_bond_{index}:1" for index in range(8))

#: Streamed copy granularity. A node store is orders of magnitude larger than
#: host RAM allows to buffer, so the copy reuses one fixed window and nothing
#: is ever assembled in memory.
DEFAULT_READ_BUFFER_BYTES = 64 * 1024 * 1024

#: First line of ``metadata.tsv``. The reader skips any row whose first field
#: is ``H`` (``Weights_Storage.cpp:379-381``).
METADATA_HEADER = "H"

#: Exactly the dtypes ``get_tensor`` resolves without falling back
#: (``Weights_Storage.cpp:820-857``). The fp8 fallback there is a
#: backward-compatibility shim for unknown dtypes and must never be reached
#: through a store this builder wrote, so an unknown dtype hard-fails.
DTYPE_ITEMSIZE = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "float8_e4m3fn": 1,
    "uint8": 1,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
}

_LAYER_PREFIX = K3_CKPT_PREFIX + "model.layers."
_SELF_ATTN_INFIX = "self_attn."
_SHARED_EXPERT_INFIX = "block_sparse_moe.shared_experts."
_ROUTED_EXPERT_INFIX = "block_sparse_moe.experts."

#: ``self_attn.*`` suffix sets that identify a layer kind. The map builder picks
#: the ring from ``cfg.is_kda_layer`` (``tensor_map.py:403-409``); the builder
#: has no config, so it requires the layer's suffix set to equal one of these
#: EXACTLY and fails on anything else rather than guessing a ring.
_ATTENTION_RINGS = (
    ("attn", frozenset(K3_MLA_TENSOR_NAMES)),
    ("kda_attn", frozenset(K3_KDA_TENSOR_NAMES)),
)


class K3StoreBuilderError(RuntimeError):
    """A converted checkpoint or a request the builder refuses to interpret."""


@dataclass(frozen=True)
class ShardMetadata:
    """One converted-checkpoint ``*.json`` sidecar."""

    json_name: str
    file_name: str
    total_byte_size: int
    #: ``ckpt_name -> (dtype, shape, offset, byte_size)``
    state_dict: Dict[str, Tuple[str, List[int], int, int]]


@dataclass
class TensorRecord:
    """One metadata row: where a checkpoint tensor lands in the store."""

    ckpt_name: str
    module_key: str
    tensor_key: str
    owner: int
    shard_file: str
    source_offset: int
    byte_size: int
    dtype: str
    shape: List[int]
    compact_offset: int = -1


@dataclass
class StorePlan:
    """The complete, node-independent layout plus this node's identity."""

    converted_ckpt: Path
    num_nodes: int
    node_rank: int
    n_layers: int
    n_experts: int
    experts_per_owner: int
    module_bytes: int
    replicated_bytes: int
    store_bytes: int
    #: Metadata emission order: replicated rows in prefix order, then every
    #: node's experts by (layer, expert, tensor_key).
    records: List[TensorRecord] = field(default_factory=list)
    replicated_records: List[TensorRecord] = field(default_factory=list)
    expert_records: List[TensorRecord] = field(default_factory=list)

    @property
    def local_records(self) -> List[TensorRecord]:
        """The rows whose bytes this node's ``store.bin`` must contain."""
        return [
            record
            for record in self.records
            if record.owner in (REPLICATED_OWNER, self.node_rank)
        ]


# --------------------------------------------------------------------------- #
#  1. Read the converted checkpoint index                                      #
# --------------------------------------------------------------------------- #


def read_shard_metadata(converted_ckpt: Path) -> List[ShardMetadata]:
    """Parse every ``*.json`` sidecar in ``converted_ckpt``.

    Schema and the ``file_name`` -> ``.bin`` resolution are
    ``Parameter_Server.cpp:65-84`` and ``:132``.
    """
    if not converted_ckpt.is_dir():
        raise K3StoreBuilderError(
            f"converted checkpoint directory does not exist: {converted_ckpt}"
        )
    json_paths = sorted(converted_ckpt.glob("*.json"))
    if not json_paths:
        raise K3StoreBuilderError(
            f"no converted-checkpoint *.json sidecars in {converted_ckpt}"
        )

    shards: List[ShardMetadata] = []
    for json_path in json_paths:
        raw = json.loads(json_path.read_text())
        missing = sorted(
            {"file_name", "state_dict", "total_byte_size"} - raw.keys()
        )
        if missing:
            raise K3StoreBuilderError(
                f"{json_path.name} is not a converted-checkpoint sidecar: "
                f"missing {missing}"
            )
        state_dict: Dict[str, Tuple[str, List[int], int, int]] = {}
        for ckpt_name, info in raw["state_dict"].items():
            info_missing = sorted(
                {"dtype", "shape", "offset", "byte_size"} - info.keys()
            )
            if info_missing:
                raise K3StoreBuilderError(
                    f"{json_path.name}: tensor {ckpt_name} misses "
                    f"{info_missing}"
                )
            dtype = str(info["dtype"])
            shape = [int(dim) for dim in info["shape"]]
            byte_size = int(info["byte_size"])
            if dtype not in DTYPE_ITEMSIZE:
                raise K3StoreBuilderError(
                    f"{json_path.name}: tensor {ckpt_name} has dtype "
                    f"{dtype!r}, which the engine cannot resolve without "
                    "falling back to fp8 (Weights_Storage.cpp:851-856)"
                )
            declared = DTYPE_ITEMSIZE[dtype]
            for dim in shape:
                declared *= dim
            if declared != byte_size:
                raise K3StoreBuilderError(
                    f"{json_path.name}: tensor {ckpt_name} declares "
                    f"byte_size={byte_size} but {dtype}{shape} is "
                    f"{declared} B"
                )
            state_dict[ckpt_name] = (
                dtype,
                shape,
                int(info["offset"]),
                byte_size,
            )
        shard = ShardMetadata(
            json_name=json_path.name,
            file_name=str(raw["file_name"]),
            total_byte_size=int(raw["total_byte_size"]),
            state_dict=state_dict,
        )
        if not (converted_ckpt / shard.file_name).is_file():
            raise K3StoreBuilderError(
                f"{json_path.name} names {shard.file_name}, which is not a "
                f"file in {converted_ckpt}"
            )
        shards.append(shard)

    seen: Dict[str, str] = {}
    for shard in shards:
        for ckpt_name in shard.state_dict:
            if ckpt_name in seen:
                raise K3StoreBuilderError(
                    f"tensor {ckpt_name} appears in both {seen[ckpt_name]} "
                    f"and {shard.json_name}"
                )
            seen[ckpt_name] = shard.json_name
    return shards


# --------------------------------------------------------------------------- #
#  2. Classify checkpoint names into (module_key, tensor_key)                  #
# --------------------------------------------------------------------------- #


def _split_layer_suffix(ckpt_name: str) -> Tuple[int, str] | None:
    """``language_model.model.layers.7.self_attn.o_proj.weight`` -> (7, ...)."""
    if not ckpt_name.startswith(_LAYER_PREFIX):
        return None
    remainder = ckpt_name[len(_LAYER_PREFIX) :]
    layer_text, separator, suffix = remainder.partition(".")
    if not separator or not layer_text.isdigit():
        return None
    return int(layer_text), suffix


def classify_tensor(
    ckpt_name: str,
) -> Tuple[str, str, int | None, int | None]:
    """Return ``(module_kind, tensor_key, layer, expert)``.

    ``module_kind`` is a routed/shared module key, the literal
    ``__skeleton__``, or ``"attn?"`` for a ``self_attn`` tensor whose ring is
    only decidable once the layer's whole suffix set is known.
    """
    split = _split_layer_suffix(ckpt_name)
    if split is None:
        return SKELETON_MODULE_KEY, ckpt_name, None, None
    layer, suffix = split

    if suffix.startswith(_SELF_ATTN_INFIX):
        return "attn?", suffix[len(_SELF_ATTN_INFIX) :], layer, None

    if suffix.startswith(_SHARED_EXPERT_INFIX):
        return (
            f"shared_expert_{layer}",
            suffix[len(_SHARED_EXPERT_INFIX) :],
            layer,
            None,
        )

    if suffix.startswith(_ROUTED_EXPERT_INFIX):
        remainder = suffix[len(_ROUTED_EXPERT_INFIX) :]
        expert_text, separator, tensor_key = remainder.partition(".")
        if not separator or not expert_text.isdigit() or not tensor_key:
            raise K3StoreBuilderError(
                f"cannot read an expert index out of {ckpt_name!r}"
            )
        expert = int(expert_text)
        return f"routed_expert_{layer}_{expert}", tensor_key, layer, expert

    return SKELETON_MODULE_KEY, ckpt_name, None, None


def _resolve_attention_rings(
    attention_suffixes: Dict[int, set],
) -> Dict[int, str]:
    """Per-layer ``attn`` vs ``kda_attn``, by exact suffix-set match."""
    rings: Dict[int, str] = {}
    for layer in sorted(attention_suffixes):
        present = frozenset(attention_suffixes[layer])
        for ring, declared in _ATTENTION_RINGS:
            if present == declared:
                rings[layer] = f"{ring}_{layer}"
                break
        else:
            raise K3StoreBuilderError(
                f"layer {layer} has self_attn tensors "
                f"{sorted(present)}, which match neither K3_MLA_TENSOR_NAMES "
                f"nor K3_KDA_TENSOR_NAMES; the ring cannot be chosen without "
                "a model config"
            )
    return rings


def _shared_expert_tensor_keys() -> frozenset:
    return frozenset(K3_SHARED_EXPERT_TENSOR_NAMES)


# --------------------------------------------------------------------------- #
#  3. Lay the store out                                                       #
# --------------------------------------------------------------------------- #


def build_store_plan(
    converted_ckpt: Path,
    num_nodes: int,
    node_rank: int,
    shards: Sequence[ShardMetadata] | None = None,
    expected_topology: Tuple[int, int] = (K3_LAYERS, K3_EXPERTS),
) -> StorePlan:
    """Compute the full layout. Reads only the ``*.json`` sidecars.

    ``expected_topology`` is ``(n_layers, n_experts)``; it exists so the unit
    tests can lay out a mini checkpoint. The CLI never passes it, so a real
    build is always held to the engine's hardcoded 92 x 896.
    """
    if num_nodes not in DISTRIBUTED_NODE_COUNTS:
        raise K3StoreBuilderError(
            f"num_nodes must be one of {DISTRIBUTED_NODE_COUNTS}; "
            f"got {num_nodes}"
        )
    if node_rank not in range(num_nodes):
        raise K3StoreBuilderError(
            f"node_rank must be in [0, {num_nodes}); got {node_rank}"
        )
    if shards is None:
        shards = read_shard_metadata(converted_ckpt)

    replicated: List[TensorRecord] = []
    expert_modules: Dict[Tuple[int, int], List[TensorRecord]] = defaultdict(
        list
    )
    attention: Dict[int, List[TensorRecord]] = defaultdict(list)
    attention_suffixes: Dict[int, set] = defaultdict(set)
    shared_expert_suffixes: Dict[int, set] = defaultdict(set)

    for shard in shards:
        for ckpt_name, (dtype, shape, offset, byte_size) in (
            shard.state_dict.items()
        ):
            module_kind, tensor_key, layer, expert = classify_tensor(
                ckpt_name
            )
            record = TensorRecord(
                ckpt_name=ckpt_name,
                module_key=module_kind,
                tensor_key=tensor_key,
                owner=REPLICATED_OWNER,
                shard_file=shard.file_name,
                source_offset=offset,
                byte_size=byte_size,
                dtype=dtype,
                shape=shape,
            )
            if module_kind == "attn?":
                attention[layer].append(record)
                attention_suffixes[layer].add(tensor_key)
            elif expert is not None:
                expert_modules[(layer, expert)].append(record)
            else:
                if module_kind != SKELETON_MODULE_KEY:
                    shared_expert_suffixes[layer].add(tensor_key)
                replicated.append(record)

    rings = _resolve_attention_rings(attention_suffixes)
    for layer, records in attention.items():
        for record in records:
            record.module_key = rings[layer]
        replicated.extend(records)

    declared_shared = _shared_expert_tensor_keys()
    for layer in sorted(shared_expert_suffixes):
        present = frozenset(shared_expert_suffixes[layer])
        if present != declared_shared:
            raise K3StoreBuilderError(
                f"shared_expert_{layer} has tensors {sorted(present)}, "
                f"expected {sorted(declared_shared)}"
            )

    n_layers, n_experts = _validate_expert_topology(
        expert_modules, expected_topology
    )
    if n_experts % num_nodes:
        raise K3StoreBuilderError(
            f"{n_experts} experts do not divide across {num_nodes} nodes"
        )
    experts_per_owner = n_experts // num_nodes
    module_bytes = _uniform_module_bytes(expert_modules)

    # Replicated prefix: deterministic order = (source file, source offset),
    # which preserves the sequential read locality of the source shards.
    replicated.sort(key=lambda record: (record.shard_file, record.source_offset))
    cursor = 0
    for record in replicated:
        record.compact_offset = cursor
        cursor += record.byte_size
    replicated_bytes = cursor

    store_bytes = (
        replicated_bytes + n_layers * experts_per_owner * module_bytes
    )

    expert_records: List[TensorRecord] = []
    for layer, expert in sorted(expert_modules):
        owner = expert // experts_per_owner
        base = replicated_bytes + (
            (layer - 1) * experts_per_owner + expert % experts_per_owner
        ) * module_bytes
        relative = 0
        # Within a module the six tensors are contiguous in sorted tensor_key
        # order. Their sizes sum to module_bytes, so the engine's
        # min-relative-offset 0 / max-relative-end == module_bytes contract
        # holds (Weights_Storage.cpp:487-504).
        module = sorted(
            expert_modules[(layer, expert)],
            key=lambda record: record.tensor_key,
        )
        for record in module:
            record.owner = owner
            record.compact_offset = base + relative
            relative += record.byte_size
        if relative != module_bytes:
            raise K3StoreBuilderError(
                f"routed_expert_{layer}_{expert} packs {relative} B, not "
                f"the uniform {module_bytes} B"
            )
        expert_records.extend(module)

    plan = StorePlan(
        converted_ckpt=converted_ckpt,
        num_nodes=num_nodes,
        node_rank=node_rank,
        n_layers=n_layers,
        n_experts=n_experts,
        experts_per_owner=experts_per_owner,
        module_bytes=module_bytes,
        replicated_bytes=replicated_bytes,
        store_bytes=store_bytes,
        records=replicated + expert_records,
        replicated_records=replicated,
        expert_records=expert_records,
    )
    _validate_plan_bounds(plan, shards)
    return plan


def _validate_expert_topology(
    expert_modules: Dict[Tuple[int, int], List[TensorRecord]],
    expected_topology: Tuple[int, int],
) -> Tuple[int, int]:
    if not expert_modules:
        raise K3StoreBuilderError(
            "the converted checkpoint contains no routed-expert tensors"
        )
    layers = sorted({layer for layer, _ in expert_modules})
    experts = sorted({expert for _, expert in expert_modules})
    # The daemon indexes layers from 1 (``RemoteOffset`` subtracts 1 and
    # ``parse_module_key`` rejects layer < 1), so the MoE layers must be the
    # contiguous run 1..n_layers.
    if layers != list(range(1, len(layers) + 1)):
        raise K3StoreBuilderError(
            f"MoE layers must be the contiguous run 1..N; got "
            f"{layers[0]}..{layers[-1]} with {len(layers)} distinct layers"
        )
    if experts != list(range(len(experts))):
        raise K3StoreBuilderError(
            f"expert indices must be the contiguous run 0..N-1; got "
            f"{experts[0]}..{experts[-1]} with {len(experts)} distinct experts"
        )
    n_layers, n_experts = len(layers), len(experts)
    if (n_layers, n_experts) != tuple(expected_topology):
        raise K3StoreBuilderError(
            f"checkpoint has {n_layers} MoE layers x {n_experts} experts, "
            f"but the store must be laid out for "
            f"{expected_topology[0]} x {expected_topology[1]}"
        )
    missing = [
        (layer, expert)
        for layer in layers
        for expert in experts
        if (layer, expert) not in expert_modules
    ]
    if missing:
        raise K3StoreBuilderError(
            f"{len(missing)} routed-expert modules are absent, first "
            f"routed_expert_{missing[0][0]}_{missing[0][1]}"
        )
    return n_layers, n_experts


def _uniform_module_bytes(
    expert_modules: Dict[Tuple[int, int], List[TensorRecord]],
) -> int:
    declared = sorted(routed_expert_tensor_names())
    sizes = set()
    for (layer, expert), records in expert_modules.items():
        keys = sorted(record.tensor_key for record in records)
        if keys != declared:
            raise K3StoreBuilderError(
                f"routed_expert_{layer}_{expert} carries {keys}, expected "
                f"the six names {declared}"
            )
        sizes.add(sum(record.byte_size for record in records))
    if len(sizes) != 1:
        raise K3StoreBuilderError(
            "routed-expert modules are not byte-uniform, which the daemon's "
            "single module_bytes stride requires; saw sizes "
            f"{sorted(sizes)}"
        )
    return sizes.pop()


def _validate_plan_bounds(
    plan: StorePlan, shards: Sequence[ShardMetadata]
) -> None:
    for record in plan.records:
        if record.compact_offset < 0:
            raise K3StoreBuilderError(
                f"{record.module_key}/{record.tensor_key} was never placed"
            )
        if record.compact_offset + record.byte_size > plan.store_bytes:
            raise K3StoreBuilderError(
                f"{record.module_key}/{record.tensor_key} ends past the "
                f"{plan.store_bytes} B store"
            )
    # Exact accounting: the replicated prefix plus EVERY node's experts is the
    # whole checkpoint, so the three numbers cannot drift apart silently.
    checkpoint_bytes = sum(
        byte_size
        for shard in shards
        for (_, _, _, byte_size) in shard.state_dict.values()
    )
    accounted = (
        plan.replicated_bytes
        + plan.n_layers * plan.n_experts * plan.module_bytes
    )
    if accounted != checkpoint_bytes:
        raise K3StoreBuilderError(
            f"replicated {plan.replicated_bytes} B + "
            f"{plan.n_layers}x{plan.n_experts}x{plan.module_bytes} B does not "
            f"account for the checkpoint's {checkpoint_bytes} B"
        )


# --------------------------------------------------------------------------- #
#  4. Emit metadata.tsv and node<R>.json                                      #
# --------------------------------------------------------------------------- #


def metadata_rows(plan: StorePlan) -> Iterable[str]:
    """``metadata.tsv`` lines, newline-free.

    One row per tensor of EVERY node: the engine needs the remote modules'
    rows to build ``remote_module_weights_``
    (``Weights_Storage.cpp:456-470``), and a remote module's offsets are the
    OWNER's, so the file is identical on every node.
    """
    yield METADATA_HEADER
    for index, record in enumerate(plan.records):
        yield "\t".join(
            (
                "T",
                str(index),
                record.module_key,
                record.tensor_key,
                str(record.owner),
                str(record.compact_offset),
                str(record.byte_size),
                record.dtype,
                json.dumps(record.shape, separators=(",", ":")),
            )
        )


def write_metadata(plan: StorePlan, metadata_path: Path) -> int:
    rows = list(metadata_rows(plan))
    metadata_path.write_text("\n".join(rows) + "\n")
    return len(rows) - 1


def node_config(
    plan: StorePlan,
    store_path: Path,
    metadata_path: Path,
    node_ips: Sequence[str],
    daemon_socket: Path,
    summary_path: Path,
    base_port: int,
    transport: str,
    rail_devices: Sequence[str],
) -> Dict[str, object]:
    """The per-node config JSON. Keys are the loader's required set plus the
    optional ones the daemon reads (``distributed_weight_store.py:32-44``,
    ``distributed_weight_daemon.cpp:324-344``)."""
    return {
        "node_rank": plan.node_rank,
        "num_nodes": plan.num_nodes,
        "node_ips": list(node_ips),
        "workers": K3_WORKERS_PER_NODE,
        "worker_sharded": True,
        "store_path": str(store_path),
        "metadata_path": str(metadata_path),
        "daemon_socket": str(daemon_socket),
        "summary_path": str(summary_path),
        "store_bytes": plan.store_bytes,
        "replicated_bytes": plan.replicated_bytes,
        "module_bytes": plan.module_bytes,
        "base_port": base_port,
        "transport": transport,
        "rail_devices": list(rail_devices),
    }


# --------------------------------------------------------------------------- #
#  5. Stream the bytes                                                        #
# --------------------------------------------------------------------------- #


def write_store(
    plan: StorePlan,
    store_path: Path,
    read_buffer_bytes: int = DEFAULT_READ_BUFFER_BYTES,
) -> int:
    """Copy this node's spans into ``store.bin``; return the bytes written.

    Each source ``.bin`` is opened once and read forward, so the copy is
    sequential on the source side and O(1) in memory.
    """
    if read_buffer_bytes <= 0:
        raise K3StoreBuilderError("read buffer must be a positive size")
    by_shard: Dict[str, List[TensorRecord]] = defaultdict(list)
    for record in plan.local_records:
        by_shard[record.shard_file].append(record)

    buffer = bytearray(read_buffer_bytes)
    view = memoryview(buffer)
    written = 0
    descriptor = os.open(
        store_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644
    )
    try:
        os.ftruncate(descriptor, plan.store_bytes)
        for shard_file in sorted(by_shard):
            source_path = plan.converted_ckpt / shard_file
            with open(source_path, "rb", buffering=0) as source:
                records = sorted(
                    by_shard[shard_file],
                    key=lambda record: record.source_offset,
                )
                for record in records:
                    written += _copy_span(source, descriptor, record, view)
            logger.info(
                "copied %s (%d tensors, %.2f GiB cumulative)",
                shard_file,
                len(by_shard[shard_file]),
                written / float(1 << 30),
            )
    finally:
        os.close(descriptor)
    return written


def _copy_span(source, descriptor: int, record: TensorRecord, view) -> int:
    source.seek(record.source_offset)
    remaining = record.byte_size
    destination = record.compact_offset
    while remaining:
        want = min(remaining, len(view))
        got = source.readinto(view[:want])
        if not got:
            raise K3StoreBuilderError(
                f"{record.shard_file} ended {remaining} B early while "
                f"reading {record.ckpt_name}"
            )
        put = os.pwrite(descriptor, view[:got], destination)
        if put != got:
            raise K3StoreBuilderError(
                f"short write of {record.ckpt_name}: {put} of {got} B"
            )
        destination += got
        remaining -= got
    return record.byte_size


def self_check(plan: StorePlan, store_path: Path, written: int) -> None:
    """Post-build gate. Raises :class:`K3StoreBuilderError` on any failure."""
    size = store_path.stat().st_size
    if size != plan.store_bytes:
        raise K3StoreBuilderError(
            f"{store_path} is {size} B, expected {plan.store_bytes} B"
        )
    # Every local byte written exactly once: the replicated prefix plus this
    # node's contiguous expert region is the whole file, with no padding.
    if written != plan.store_bytes:
        raise K3StoreBuilderError(
            f"wrote {written} B into a {plan.store_bytes} B store; the local "
            "spans do not tile it exactly"
        )
    local_experts = {
        (record.module_key)
        for record in plan.expert_records
        if record.owner == plan.node_rank
    }
    expected = plan.n_layers * plan.experts_per_owner
    if len(local_experts) != expected:
        raise K3StoreBuilderError(
            f"node {plan.node_rank} holds {len(local_experts)} expert "
            f"modules, expected {expected}"
        )


# --------------------------------------------------------------------------- #
#  6. CLI                                                                      #
# --------------------------------------------------------------------------- #


def _parse_csv(value: str) -> List[str]:
    items = [item.strip() for item in value.split(",")]
    if not all(items):
        raise argparse.ArgumentTypeError(
            f"{value!r} contains an empty comma-separated entry"
        )
    return items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m batchgen.models.moonshotai.kimi_linear.store_builder",
        description=(
            "Build one node's Kimi-K3 compact distributed host-weight store."
        ),
    )
    parser.add_argument("--converted-ckpt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--num-nodes", type=int, required=True, choices=DISTRIBUTED_NODE_COUNTS
    )
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument("--node-ips", type=_parse_csv, required=True)
    parser.add_argument("--daemon-socket", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--base-port", type=int, default=DEFAULT_BASE_PORT)
    parser.add_argument(
        "--transport", choices=WEIGHT_TRANSPORTS, default=WEIGHT_TRANSPORTS[0]
    )
    parser.add_argument(
        "--rail-devices", type=_parse_csv, default=list(H200_RAIL_DEVICES)
    )
    parser.add_argument(
        "--read-buffer-bytes", type=int, default=DEFAULT_READ_BUFFER_BYTES
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Write metadata.tsv and node<R>.json only, and report the sizes "
            "the real build would produce."
        ),
    )
    return parser


def build(
    argv: Sequence[str] | None = None,
    expected_topology: Tuple[int, int] = (K3_LAYERS, K3_EXPERTS),
) -> StorePlan:
    """Run one build. Returns the plan; raises on any validation failure.

    ``expected_topology`` is forwarded to :func:`build_store_plan` so the unit
    tests can drive this whole path with a mini checkpoint. :func:`main` never
    passes it, so the CLI always enforces the engine's 92 x 896.
    """
    args = build_parser().parse_args(argv)

    if len(args.node_ips) != args.num_nodes:
        raise K3StoreBuilderError(
            f"--node-ips lists {len(args.node_ips)} addresses for "
            f"--num-nodes {args.num_nodes}"
        )
    if len(args.rail_devices) != len(H200_RAIL_DEVICES):
        raise K3StoreBuilderError(
            f"--rail-devices needs exactly {len(H200_RAIL_DEVICES)} entries; "
            f"got {len(args.rail_devices)}"
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    store_path = output_dir / "store.bin"
    metadata_path = output_dir / "metadata.tsv"
    config_path = output_dir / f"node{args.node_rank}.json"
    daemon_socket = args.daemon_socket or output_dir / "daemon.sock"
    summary_path = args.summary_path or output_dir / "summary.json"

    plan = build_store_plan(
        converted_ckpt=args.converted_ckpt,
        num_nodes=args.num_nodes,
        node_rank=args.node_rank,
        expected_topology=expected_topology,
    )

    rows = write_metadata(plan, metadata_path)
    config_path.write_text(
        json.dumps(
            node_config(
                plan,
                store_path=store_path,
                metadata_path=metadata_path,
                node_ips=args.node_ips,
                daemon_socket=daemon_socket,
                summary_path=summary_path,
                base_port=args.base_port,
                transport=args.transport,
                rail_devices=args.rail_devices,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    logger.info(
        "node %d/%d: %d tensors (%d replicated, %d expert rows), "
        "experts_per_owner=%d",
        plan.node_rank,
        plan.num_nodes,
        len(plan.records),
        len(plan.replicated_records),
        len(plan.expert_records),
        plan.experts_per_owner,
    )
    logger.info(
        "module_bytes=%d replicated_bytes=%d store_bytes=%d (%.2f GiB)",
        plan.module_bytes,
        plan.replicated_bytes,
        plan.store_bytes,
        plan.store_bytes / float(1 << 30),
    )
    logger.info("wrote %s (%d rows) and %s", metadata_path, rows, config_path)

    if args.dry_run:
        logger.info(
            "--dry-run: store.bin not written, so %s cannot be validated by "
            "load_distributed_weight_config until the real build runs",
            config_path,
        )
        return plan

    written = write_store(plan, store_path, args.read_buffer_bytes)
    self_check(plan, store_path, written)
    logger.info("self-check passed: %s is %d B", store_path, plan.store_bytes)
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    try:
        build(argv)
    except K3StoreBuilderError as error:
        logger.error("store build failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
