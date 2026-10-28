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

The map is built from the model's own ``config.json`` (``--cache-dir``), so
which ring a layer belongs to is the config's ``is_kda_layer`` answer, never
a guess from the tensor names that happen to be present.

Completeness
------------
Before anything is written, the checkpoint is held to the full expected name
set in both directions: every name in :func:`build_k3_state_dict_name_map`
and :func:`k3_skeleton_declaration` must be present (every attention layer,
every shared and routed expert, every skeleton tensor), and every present name
must be one of those or under a :data:`K3_IGNORE_RULES` root. A missing module
would otherwise produce a smaller, self-consistent store that the engine only
discovers when a key lookup fails mid-load.

Integrity
---------
Every source span is checked against its ``.bin``'s real length: offsets are
non-negative, spans end inside the file, no two spans of one file overlap, and
the spans sum to both ``total_byte_size`` and the file length. The copy pass
hashes every tensor (BLAKE2b-128) as it streams; a verification pass then
re-reads ``store.bin`` and requires every tensor's bytes to hash to what was
read from the source, so a corrupted or concurrently modified store never
publishes.

Publication
-----------
A build writes into a fresh ``.k3-store-build-*`` directory under
``--output-dir``. Only after the copy, the self-check and the verification
pass succeed are ``store.bin``, ``metadata.tsv`` and ``node<R>.json`` renamed
into ``--output-dir``, followed by ``store_manifest.json`` -- the completion
manifest and the commit point. A previous build's manifest is removed before
its files are replaced, so at every instant the output directory either has a
manifest describing exactly the files beside it or has none.
:func:`~.distributed_weight_store.load_distributed_weight_config` refuses a
config whose manifest is missing or disagrees with the store, so an
interrupted build is never mapped. ``--dry-run`` writes nothing and refuses an
output directory that holds a store without a matching manifest.

The node config names no summary file: the daemon reports its state through
the server log and status, not through a file beside the store.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import stat
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from .distributed_weight_store import (
    DEFAULT_STORE_STAGING_DIR,
    DISTRIBUTED_NODE_COUNTS,
    STORE_RESIDENCIES,
    WEIGHT_TRANSPORTS,
    DistributedMetadataError,
    read_distributed_metadata,
    read_distributed_weight_config,
)
from .k3.mxfp4_layout import routed_expert_tensor_names
from .k3.tensor_map import (
    K3_IGNORE_RULES,
    build_k3_state_dict_name_map,
    k3_skeleton_declaration,
    load_k3_config,
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

#: Threads for the verification re-read. ``hashlib`` releases the GIL while
#: hashing large buffers, so segments hash in parallel.
DEFAULT_HASH_WORKERS = 8

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

# --------------------------------------------------------------------------- #
#  Published-store format (the single definition; the loader and the server   #
#  preflight import these names)                                              #
# --------------------------------------------------------------------------- #

BUILDER_MODULE = "batchgen.models.moonshotai.kimi_linear.store_builder"
#: Bumped whenever the bytes or rows a given checkpoint produces change.
STORE_BUILDER_VERSION = 1
MANIFEST_FORMAT = "batchgen.kimi_k3.distributed_store_manifest"
MANIFEST_FORMAT_VERSION = 1
STORE_FILE_NAME = "store.bin"
METADATA_FILE_NAME = "metadata.tsv"
MANIFEST_FILE_NAME = "store_manifest.json"
#: Written last into a server-staged copy (``batchgen.server.weight_store_staging``)
#: once every segment of the copy hashed to the manifest.
STAGED_COMPLETE_FILE_NAME = "COMPLETE"
STAGING_PREFIX = ".k3-store-build-"
LOCK_FILE_NAME = ".k3-store-build.lock"
HASH_ALGORITHM = "blake2b-128"
_DIGEST_BYTES = 16
#: The manifest hashes ``store.bin`` in segments of whole tensors of at most
#: this many bytes (one tensor larger than this is its own segment), so a
#: launch can re-hash a sample of the store instead of all of it.
SEGMENT_TARGET_BYTES = 1 << 30
#: Segments the launch preflight re-hashes: the first, the middle, the last.
PREFLIGHT_SPOT_CHECK_SEGMENTS = 3

_ROUTED_EXPERT_KEY = re.compile(r"routed_expert_(\d+)_(\d+)")
_NODE_CONFIG_NAME = re.compile(r"node(\d+)\.json")


class K3StoreBuilderError(RuntimeError):
    """A converted checkpoint or a request the builder refuses to interpret."""


class K3StoreManifestError(ValueError):
    """A published store that must not be mapped.

    The message always ends with the exact builder command that rebuilds it.
    A ``ValueError`` so every launch path that already reports argument and
    config errors reports this one the same way.
    """


def _digest() -> "hashlib._Hash":
    return hashlib.blake2b(digest_size=_DIGEST_BYTES)


def _rollup(digests: Iterable[bytes]) -> str:
    """One digest over an ordered list of tensor digests."""
    hasher = _digest()
    for digest in digests:
        hasher.update(digest)
    return hasher.hexdigest()


@dataclass(frozen=True)
class ShardMetadata:
    """One converted-checkpoint ``*.json`` sidecar, validated against its ``.bin``."""

    json_name: str
    file_name: str
    total_byte_size: int
    #: ``stat().st_size`` of the ``.bin`` at read time.
    file_bytes: int
    #: BLAKE2b-128 of the sidecar's raw bytes.
    sidecar_digest: str
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
    #: Index of this tensor's ``metadata.tsv`` row.
    row: int = -1


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
    shards: List[ShardMetadata] = field(default_factory=list)
    #: Tensors under a :data:`K3_IGNORE_RULES` root, carried in the skeleton.
    ignored_tensors: int = 0

    @property
    def local_records(self) -> List[TensorRecord]:
        """The rows whose bytes this node's ``store.bin`` must contain."""
        return [
            record
            for record in self.records
            if record.owner in (REPLICATED_OWNER, self.node_rank)
        ]


# --------------------------------------------------------------------------- #
#  1. Read and validate the converted checkpoint index                         #
# --------------------------------------------------------------------------- #


def _sidecar_int(value: Any, where: str, field_name: str) -> int:
    # bool is an int subclass; a JSON ``true`` is never a size or an offset.
    if isinstance(value, bool) or not isinstance(value, int):
        raise K3StoreBuilderError(
            f"{where}: {field_name} must be an integer, got {value!r}"
        )
    return value


def _read_sidecar(json_path: Path) -> Tuple[bytes, Dict[str, Any]]:
    try:
        raw_bytes = json_path.read_bytes()
    except OSError as error:
        raise K3StoreBuilderError(
            f"cannot read sidecar {json_path}: {error}"
        ) from error
    try:
        raw = json.loads(raw_bytes)
    except ValueError as error:
        raise K3StoreBuilderError(
            f"{json_path} is not valid JSON: {error}"
        ) from error
    if not isinstance(raw, dict):
        raise K3StoreBuilderError(f"{json_path} is not a JSON object")
    return raw_bytes, raw


def _parse_tensor(
    json_path: Path, ckpt_name: str, info: Any, file_bytes: int, bin_path: Path
) -> Tuple[str, List[int], int, int]:
    where = f"{json_path}: tensor {ckpt_name}"
    if not isinstance(info, dict):
        raise K3StoreBuilderError(f"{where} is not a JSON object")
    info_missing = sorted({"dtype", "shape", "offset", "byte_size"} - info.keys())
    if info_missing:
        raise K3StoreBuilderError(f"{where} misses {info_missing}")
    dtype = info["dtype"]
    if not isinstance(dtype, str) or dtype not in DTYPE_ITEMSIZE:
        raise K3StoreBuilderError(
            f"{where} has dtype {dtype!r}, which the engine cannot resolve "
            "without falling back to fp8 (Weights_Storage.cpp:851-856)"
        )
    raw_shape = info["shape"]
    if not isinstance(raw_shape, list):
        raise K3StoreBuilderError(f"{where} has a non-list shape {raw_shape!r}")
    shape = [_sidecar_int(dim, where, "a shape dimension") for dim in raw_shape]
    if any(dim < 0 for dim in shape):
        raise K3StoreBuilderError(f"{where} has a negative dimension in {shape}")
    offset = _sidecar_int(info["offset"], where, "offset")
    byte_size = _sidecar_int(info["byte_size"], where, "byte_size")
    if offset < 0:
        raise K3StoreBuilderError(f"{where} has negative offset {offset}")
    if byte_size < 0:
        raise K3StoreBuilderError(f"{where} has negative byte_size {byte_size}")
    declared = DTYPE_ITEMSIZE[dtype]
    for dim in shape:
        declared *= dim
    if declared != byte_size:
        raise K3StoreBuilderError(
            f"{where} declares byte_size={byte_size} but {dtype}{shape} is "
            f"{declared} B"
        )
    if offset + byte_size > file_bytes:
        raise K3StoreBuilderError(
            f"{where} spans [{offset}, {offset + byte_size}) but {bin_path} "
            f"is only {file_bytes} B"
        )
    return dtype, shape, offset, byte_size


def _check_no_overlap(
    json_path: Path, state_dict: Mapping[str, Tuple[str, List[int], int, int]]
) -> None:
    spans = sorted(
        (offset, offset + byte_size, name)
        for name, (_, _, offset, byte_size) in state_dict.items()
        if byte_size > 0
    )
    previous_end, previous_name = 0, None
    for begin, end, name in spans:
        if begin < previous_end:
            raise K3StoreBuilderError(
                f"{json_path}: tensor {name} at [{begin}, {end}) overlaps "
                f"tensor {previous_name}, which ends at {previous_end}"
            )
        previous_end, previous_name = end, name


def read_shard_metadata(converted_ckpt: Path) -> List[ShardMetadata]:
    """Parse and validate every ``*.json`` sidecar in ``converted_ckpt``.

    Schema and the ``file_name`` -> ``.bin`` resolution are
    ``Parameter_Server.cpp:65-84`` and ``:132``. Every span is checked against
    the ``.bin``'s real length; the spans must not overlap and must sum to
    both ``total_byte_size`` and that length (the converter writes the
    tensors back to back, ``shm_reservation.py`` holds the same identity).
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
        raw_bytes, raw = _read_sidecar(json_path)
        missing = sorted({"file_name", "state_dict", "total_byte_size"} - raw.keys())
        if missing:
            raise K3StoreBuilderError(
                f"{json_path.name} is not a converted-checkpoint sidecar: "
                f"missing {missing}"
            )
        file_name = raw["file_name"]
        if (
            not isinstance(file_name, str)
            or not file_name
            or Path(file_name).name != file_name
        ):
            raise K3StoreBuilderError(
                f"{json_path}: file_name {file_name!r} must name a file "
                f"directly inside {converted_ckpt}"
            )
        total_byte_size = _sidecar_int(
            raw["total_byte_size"], str(json_path), "total_byte_size"
        )
        if not isinstance(raw["state_dict"], dict):
            raise K3StoreBuilderError(f"{json_path}: state_dict is not an object")
        bin_path = converted_ckpt / file_name
        try:
            bin_stat = bin_path.stat()
        except FileNotFoundError:
            bin_stat = None
        except OSError as error:
            raise K3StoreBuilderError(f"cannot stat {bin_path}: {error}") from error
        if bin_stat is None or not stat.S_ISREG(bin_stat.st_mode):
            raise K3StoreBuilderError(
                f"{json_path.name} names {file_name}, which is not a file in "
                f"{converted_ckpt}"
            )
        file_bytes = bin_stat.st_size

        state_dict = {
            ckpt_name: _parse_tensor(json_path, ckpt_name, info, file_bytes, bin_path)
            for ckpt_name, info in raw["state_dict"].items()
        }
        _check_no_overlap(json_path, state_dict)
        tensor_total = sum(entry[3] for entry in state_dict.values())
        if tensor_total != total_byte_size:
            raise K3StoreBuilderError(
                f"{json_path}: tensor byte sizes sum to {tensor_total} B but "
                f"total_byte_size is {total_byte_size}"
            )
        if file_bytes != total_byte_size:
            raise K3StoreBuilderError(
                f"{bin_path} is {file_bytes} B but {json_path.name} declares "
                f"total_byte_size {total_byte_size}"
            )
        shards.append(
            ShardMetadata(
                json_name=json_path.name,
                file_name=file_name,
                total_byte_size=total_byte_size,
                file_bytes=file_bytes,
                sidecar_digest=_digest_bytes(raw_bytes),
                state_dict=state_dict,
            )
        )

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


def _digest_bytes(data: bytes) -> str:
    hasher = _digest()
    hasher.update(data)
    return hasher.hexdigest()


def checkpoint_fingerprint(shards: Sequence[ShardMetadata]) -> str:
    """Content identity of a converted checkpoint.

    Covers every sidecar's bytes (names, dtypes, shapes, offsets) and every
    ``.bin``'s length, so it is independent of the mount path and cheap
    enough to recompute at every launch; it does not re-hash weight bytes.
    """
    hasher = _digest()
    for shard in sorted(shards, key=lambda item: item.json_name):
        hasher.update(
            f"{shard.json_name}\0{shard.sidecar_digest}\0{shard.file_name}"
            f"\0{shard.file_bytes}\n".encode()
        )
    return hasher.hexdigest()


def checkpoint_identity(converted_ckpt: Path) -> Dict[str, Any]:
    """Validate ``converted_ckpt`` and return its identity record."""
    shards = read_shard_metadata(converted_ckpt)
    return {
        "converted_ckpt": os.path.abspath(converted_ckpt),
        "fingerprint": checkpoint_fingerprint(shards),
        "sidecars": len(shards),
        "tensors": sum(len(shard.state_dict) for shard in shards),
    }


# --------------------------------------------------------------------------- #
#  2. The expected tensor-name map, held in both directions                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExpectedTensorMap:
    """Every name a K3 checkpoint must and may contain."""

    #: ``ckpt_name -> (module_key, tensor_key)`` for the four module rings.
    mapped: Dict[str, Tuple[str, str]]
    #: Resident tensors, keyed by checkpoint name.
    skeleton: frozenset
    #: Roots that may be present and are carried in the skeleton untouched.
    ignored_prefixes: Tuple[str, ...]


def expected_tensor_map(cfg) -> ExpectedTensorMap:
    """The name map the engine loads with, from the model's own config."""
    try:
        name_map, _ = build_k3_state_dict_name_map(cfg)
        skeleton = frozenset(k3_skeleton_declaration(cfg))
    except (ValueError, KeyError, TypeError) as error:
        raise K3StoreBuilderError(
            f"model config is not a loadable Kimi-K3 config: {error}"
        ) from error
    mapped = {
        name: (entry["module_key"], entry["tensor_key"])
        for name, entry in name_map.items()
    }
    both = sorted(skeleton & mapped.keys())
    if both:
        raise K3StoreBuilderError(
            f"{len(both)} names are both module tensors and skeleton tensors, "
            f"first {both[0]}"
        )
    return ExpectedTensorMap(
        mapped=mapped,
        skeleton=skeleton,
        ignored_prefixes=tuple(rule.prefix for rule in K3_IGNORE_RULES),
    )


def _describe_name_mismatch(
    missing: Sequence[str],
    unexpected: Sequence[str],
    expected: ExpectedTensorMap,
    limit: int = 8,
) -> str:
    parts = []
    if missing:
        modules = Counter(
            expected.mapped[name][0] if name in expected.mapped else SKELETON_MODULE_KEY
            for name in missing
        )
        shown = ", ".join(
            f"{key} ({count} missing)" for key, count in sorted(modules.items())[:limit]
        )
        parts.append(
            f"{len(missing)} expected tensors are absent from {len(modules)} "
            f"modules: {shown}; first absent names {list(missing[:limit])}"
        )
    if unexpected:
        parts.append(
            f"{len(unexpected)} tensors are in neither the Kimi-K3 tensor map, "
            f"the skeleton declaration, nor an ignored root: "
            f"{list(unexpected[:limit])}"
        )
    return "converted checkpoint does not match the Kimi-K3 tensor map: " + "; ".join(
        parts
    )


def _classify(
    shards: Sequence[ShardMetadata], expected: ExpectedTensorMap
) -> Tuple[List[TensorRecord], int]:
    records: List[TensorRecord] = []
    present = set()
    unexpected: List[str] = []
    ignored = 0
    for shard in shards:
        for ckpt_name, (dtype, shape, offset, byte_size) in shard.state_dict.items():
            present.add(ckpt_name)
            mapped = expected.mapped.get(ckpt_name)
            if mapped is not None:
                module_key, tensor_key = mapped
            elif ckpt_name in expected.skeleton:
                module_key, tensor_key = SKELETON_MODULE_KEY, ckpt_name
            elif ckpt_name.startswith(expected.ignored_prefixes):
                module_key, tensor_key = SKELETON_MODULE_KEY, ckpt_name
                ignored += 1
            else:
                unexpected.append(ckpt_name)
                continue
            records.append(
                TensorRecord(
                    ckpt_name=ckpt_name,
                    module_key=module_key,
                    tensor_key=tensor_key,
                    owner=REPLICATED_OWNER,
                    shard_file=shard.file_name,
                    source_offset=offset,
                    byte_size=byte_size,
                    dtype=dtype,
                    shape=shape,
                )
            )
    missing = sorted((set(expected.mapped) | expected.skeleton) - present)
    if missing or unexpected:
        raise K3StoreBuilderError(
            _describe_name_mismatch(missing, sorted(unexpected), expected)
        )
    return records, ignored


# --------------------------------------------------------------------------- #
#  3. Lay the store out                                                       #
# --------------------------------------------------------------------------- #


def build_store_plan(
    converted_ckpt: Path,
    num_nodes: int,
    node_rank: int,
    cfg,
    shards: Sequence[ShardMetadata] | None = None,
    expected_topology: Tuple[int, int] = (K3_LAYERS, K3_EXPERTS),
) -> StorePlan:
    """Compute the full layout. Reads only the ``*.json`` sidecars.

    ``cfg`` is the K3 model config whose tensor map the checkpoint must
    match exactly. ``expected_topology`` is ``(n_moe_layers, n_experts)``; it
    exists so the unit tests can lay out a mini checkpoint. The CLI never
    passes it, so a real build is always held to the engine's hardcoded
    92 x 896.
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
    expected = expected_tensor_map(cfg)
    if shards is None:
        shards = read_shard_metadata(converted_ckpt)

    classified, ignored = _classify(shards, expected)
    replicated: List[TensorRecord] = []
    expert_modules: Dict[Tuple[int, int], List[TensorRecord]] = defaultdict(list)
    for record in classified:
        match = _ROUTED_EXPERT_KEY.fullmatch(record.module_key)
        if match:
            expert_modules[(int(match.group(1)), int(match.group(2)))].append(
                record
            )
        else:
            replicated.append(record)

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

    store_bytes = replicated_bytes + n_layers * experts_per_owner * module_bytes

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

    records = replicated + expert_records
    for row, record in enumerate(records):
        record.row = row
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
        records=records,
        replicated_records=replicated,
        expert_records=expert_records,
        shards=list(shards),
        ignored_tensors=ignored,
    )
    _validate_plan_bounds(plan, shards)
    _validate_local_tiling(plan)
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


def _validate_local_tiling(plan: StorePlan) -> None:
    """This node's spans cover ``[0, store_bytes)`` exactly once."""
    cursor = 0
    for record in sorted(plan.local_records, key=lambda item: item.compact_offset):
        if record.compact_offset != cursor:
            raise K3StoreBuilderError(
                f"local spans do not tile the store: {record.module_key}/"
                f"{record.tensor_key} starts at {record.compact_offset}, "
                f"expected {cursor}"
            )
        cursor += record.byte_size
    if cursor != plan.store_bytes:
        raise K3StoreBuilderError(
            f"local spans cover {cursor} B of a {plan.store_bytes} B store"
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
    for record in plan.records:
        yield "\t".join(
            (
                "T",
                str(record.row),
                record.module_key,
                record.tensor_key,
                str(record.owner),
                str(record.compact_offset),
                str(record.byte_size),
                record.dtype,
                json.dumps(record.shape, separators=(",", ":")),
            )
        )


def metadata_bytes(plan: StorePlan) -> bytes:
    return ("\n".join(metadata_rows(plan)) + "\n").encode()


def node_config(
    plan: StorePlan,
    store_path: Path,
    metadata_path: Path,
    manifest_path: Path,
    node_ips: Sequence[str],
    daemon_socket: Path,
    base_port: int,
    transport: str,
    rail_devices: Sequence[str],
    store_residency: str = STORE_RESIDENCIES[0],
    store_staging_dir: str = DEFAULT_STORE_STAGING_DIR,
) -> Dict[str, object]:
    """The per-node config JSON. Keys are the loader's required set plus the
    optional ones the daemon reads (``distributed_weight_store.py:32-44``,
    ``distributed_weight_daemon.cpp:324-344``) plus ``manifest_path``,
    ``store_residency`` and ``store_staging_dir``, which only the Python side
    reads (the server stages ``store_path`` into the staging directory and
    hands the daemon and workers the staged copy)."""
    return {
        "node_rank": plan.node_rank,
        "num_nodes": plan.num_nodes,
        "node_ips": list(node_ips),
        "workers": K3_WORKERS_PER_NODE,
        "worker_sharded": True,
        "store_path": str(store_path),
        "metadata_path": str(metadata_path),
        "manifest_path": str(manifest_path),
        "daemon_socket": str(daemon_socket),
        "store_bytes": plan.store_bytes,
        "replicated_bytes": plan.replicated_bytes,
        "module_bytes": plan.module_bytes,
        "base_port": base_port,
        "transport": transport,
        "rail_devices": list(rail_devices),
        "store_residency": store_residency,
        "store_staging_dir": store_staging_dir,
    }


# --------------------------------------------------------------------------- #
#  5. Stream the bytes, hashing every tensor                                  #
# --------------------------------------------------------------------------- #


@dataclass
class CopyResult:
    written: int
    #: ``row -> BLAKE2b-128`` of the bytes read from the source.
    record_digests: Dict[int, bytes]
    #: Per-source-shard manifest entries.
    shards: List[Dict[str, Any]]


def write_store(
    plan: StorePlan,
    store_path: Path,
    read_buffer_bytes: int = DEFAULT_READ_BUFFER_BYTES,
) -> CopyResult:
    """Copy this node's spans into a NEW ``store_path``, hashing each tensor.

    Each source ``.bin`` is opened once and read forward, so the copy is
    sequential on the source side and O(1) in memory.
    """
    if read_buffer_bytes <= 0:
        raise K3StoreBuilderError("read buffer must be a positive size")
    by_shard: Dict[str, List[TensorRecord]] = defaultdict(list)
    for record in plan.local_records:
        by_shard[record.shard_file].append(record)
    sidecars = {shard.file_name: shard for shard in plan.shards}

    buffer = bytearray(read_buffer_bytes)
    view = memoryview(buffer)
    written = 0
    digests: Dict[int, bytes] = {}
    shard_entries: List[Dict[str, Any]] = []
    try:
        descriptor = os.open(
            store_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644
        )
    except OSError as error:
        raise K3StoreBuilderError(f"cannot create {store_path}: {error}") from error
    try:
        try:
            os.ftruncate(descriptor, plan.store_bytes)
        except OSError as error:
            raise K3StoreBuilderError(
                f"cannot size {store_path} to {plan.store_bytes} B: {error}"
            ) from error
        for shard_file in sorted(by_shard):
            source_path = plan.converted_ckpt / shard_file
            records = sorted(
                by_shard[shard_file], key=lambda record: record.source_offset
            )
            try:
                source = open(source_path, "rb", buffering=0)
            except OSError as error:
                raise K3StoreBuilderError(
                    f"cannot open {source_path}: {error}"
                ) from error
            copied = 0
            with source:
                for record in records:
                    digests[record.row] = _copy_span(
                        source, descriptor, record, view, source_path, store_path
                    )
                    copied += record.byte_size
            written += copied
            sidecar = sidecars.get(shard_file)
            shard_entries.append(
                {
                    "sidecar": sidecar.json_name if sidecar else None,
                    "file": shard_file,
                    "file_bytes": sidecar.file_bytes if sidecar else None,
                    "sidecar_digest": sidecar.sidecar_digest if sidecar else None,
                    "tensors_copied": len(records),
                    "bytes_copied": copied,
                    "copied_digest": _rollup(digests[record.row] for record in records),
                }
            )
            logger.info(
                "copied %s (%d tensors, %.2f GiB cumulative)",
                shard_file,
                len(records),
                written / float(1 << 30),
            )
        try:
            os.fsync(descriptor)
        except OSError as error:
            raise K3StoreBuilderError(f"cannot fsync {store_path}: {error}") from error
    finally:
        os.close(descriptor)
    return CopyResult(written=written, record_digests=digests, shards=shard_entries)


def _copy_span(
    source, descriptor: int, record: TensorRecord, view, source_path: Path,
    store_path: Path,
) -> bytes:
    hasher = _digest()
    remaining = record.byte_size
    destination = record.compact_offset
    try:
        source.seek(record.source_offset)
        while remaining:
            want = min(remaining, len(view))
            got = source.readinto(view[:want])
            if not got:
                raise K3StoreBuilderError(
                    f"{source_path} ended {remaining} B early while reading "
                    f"{record.ckpt_name}"
                )
            chunk = view[:got]
            hasher.update(chunk)
            put = os.pwrite(descriptor, chunk, destination)
            if put != got:
                raise K3StoreBuilderError(
                    f"short write of {record.ckpt_name} into {store_path}: "
                    f"{put} of {got} B"
                )
            destination += got
            remaining -= got
    except OSError as error:
        raise K3StoreBuilderError(
            f"I/O error copying {record.ckpt_name} from {source_path} into "
            f"{store_path}: {error}"
        ) from error
    return hasher.digest()


def self_check(plan: StorePlan, store_path: Path, written: int) -> None:
    """Post-copy size gate. Raises :class:`K3StoreBuilderError` on any failure."""
    try:
        size = store_path.stat().st_size
    except OSError as error:
        raise K3StoreBuilderError(f"cannot stat {store_path}: {error}") from error
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
        record.module_key
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
#  6. Re-read and hash the store in segments                                  #
# --------------------------------------------------------------------------- #

#: ``(row, compact_offset, byte_size)`` of one local tensor.
Span = Tuple[int, int, int]


def segment_spans(
    spans: Sequence[Span], target_bytes: Optional[int] = None
) -> List[List[Span]]:
    """Group offset-ordered spans into runs of at most ``target_bytes``
    (default :data:`SEGMENT_TARGET_BYTES`)."""
    if target_bytes is None:
        target_bytes = SEGMENT_TARGET_BYTES
    segments: List[List[Span]] = []
    current: List[Span] = []
    current_bytes = 0
    for span in spans:
        if current and current_bytes + span[2] > target_bytes:
            segments.append(current)
            current, current_bytes = [], 0
        current.append(span)
        current_bytes += span[2]
    if current:
        segments.append(current)
    return segments


def _read_span_digests(
    descriptor: int,
    spans: Sequence[Span],
    read_buffer_bytes: int,
    sink: Optional[int] = None,
) -> List[bytes]:
    """Per-tensor digests of ``spans`` read from ``descriptor``.

    With ``sink``, every chunk is also written to that descriptor at the same
    offset, so a copy is hashed from exactly the bytes it wrote.
    """
    out = []
    for row, offset, byte_size in spans:
        hasher = _digest()
        position, end = offset, offset + byte_size
        while position < end:
            chunk = os.pread(
                descriptor, min(end - position, read_buffer_bytes), position
            )
            if not chunk:
                raise EOFError(
                    f"store ends at {position} B inside metadata row {row}"
                )
            hasher.update(chunk)
            if sink is not None:
                view = memoryview(chunk)
                written = 0
                while written < len(chunk):
                    written += os.pwrite(sink, view[written:], position + written)
            position += len(chunk)
        out.append(hasher.digest())
    return out


def hash_store_segment(
    descriptor: int,
    spans: Sequence[Span],
    read_buffer_bytes: int = DEFAULT_READ_BUFFER_BYTES,
    sink: Optional[int] = None,
) -> str:
    """One manifest segment's digest: the rollup of its tensors' digests.

    ``spans`` are the segment's spans from :func:`manifest_segment_spans`.
    With ``sink``, the segment is also copied into that descriptor at the
    same offsets. Raises ``OSError`` or ``EOFError``.
    """
    return _rollup(_read_span_digests(descriptor, spans, read_buffer_bytes, sink))


def verify_store(
    plan: StorePlan,
    store_path: Path,
    copy: CopyResult,
    read_buffer_bytes: int = DEFAULT_READ_BUFFER_BYTES,
    hash_workers: int = DEFAULT_HASH_WORKERS,
) -> List[Dict[str, Any]]:
    """Re-read ``store_path`` and require every tensor's source bytes.

    Returns the manifest's segment entries: each segment's digest is the
    rollup of its tensors' digests in store order.
    """
    spans = [
        (record.row, record.compact_offset, record.byte_size)
        for record in sorted(plan.local_records, key=lambda item: item.compact_offset)
    ]
    segments = segment_spans(spans)
    try:
        descriptor = os.open(store_path, os.O_RDONLY)
    except OSError as error:
        raise K3StoreBuilderError(f"cannot reopen {store_path}: {error}") from error

    def check(segment: List[Span]) -> str:
        digests = _read_span_digests(descriptor, segment, read_buffer_bytes)
        for (row, offset, byte_size), digest in zip(segment, digests):
            if copy.record_digests.get(row) != digest:
                record = plan.records[row]
                raise K3StoreBuilderError(
                    f"{store_path} bytes of {record.ckpt_name} at [{offset}, "
                    f"{offset + byte_size}) do not hash to the bytes copied "
                    f"from {record.shard_file}: the store was modified or "
                    "corrupted between copy and verification"
                )
        return _rollup(digests)

    try:
        with ThreadPoolExecutor(max_workers=hash_workers) as pool:
            rollups = list(pool.map(check, segments))
    except (OSError, EOFError) as error:
        raise K3StoreBuilderError(
            f"cannot re-read {store_path} for verification: {error}"
        ) from error
    finally:
        os.close(descriptor)
    return [
        {
            "offset": segment[0][1],
            "bytes": sum(span[2] for span in segment),
            "tensors": len(segment),
            "digest": digest,
        }
        for segment, digest in zip(segments, rollups)
    ]


# --------------------------------------------------------------------------- #
#  7. Completion manifest: write, read, verify                                #
# --------------------------------------------------------------------------- #


def build_manifest(
    plan: StorePlan,
    *,
    segments: Sequence[Mapping[str, Any]],
    copy: CopyResult,
    fingerprint: str,
    model_config: Mapping[str, str],
    metadata: bytes,
    build_command: str,
) -> Dict[str, Any]:
    return {
        "format": MANIFEST_FORMAT,
        "format_version": MANIFEST_FORMAT_VERSION,
        "builder": {"module": BUILDER_MODULE, "version": STORE_BUILDER_VERSION},
        "complete": True,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hash_algorithm": HASH_ALGORITHM,
        "node_rank": plan.node_rank,
        "num_nodes": plan.num_nodes,
        "n_layers": plan.n_layers,
        "n_experts": plan.n_experts,
        "experts_per_owner": plan.experts_per_owner,
        "store": {
            "file": STORE_FILE_NAME,
            "bytes": plan.store_bytes,
            "replicated_bytes": plan.replicated_bytes,
            "module_bytes": plan.module_bytes,
            "local_tensors": len(plan.local_records),
            "segment_target_bytes": SEGMENT_TARGET_BYTES,
            "segments": list(segments),
        },
        "metadata": {
            "file": METADATA_FILE_NAME,
            "bytes": len(metadata),
            "rows": len(plan.records),
            "digest": _digest_bytes(metadata),
        },
        "source": {
            "converted_ckpt": os.path.abspath(plan.converted_ckpt),
            "fingerprint": fingerprint,
            "sidecars": len(plan.shards),
            "ignored_tensors": plan.ignored_tensors,
            "shards": copy.shards,
            "model_config": dict(model_config),
        },
        "build_command": build_command,
    }


def _builder_argv(
    *,
    converted_ckpt: Any,
    cache_dir: Any,
    output_dir: Any,
    num_nodes: Any,
    node_rank: Any,
    node_ips: Any,
    config: Mapping[str, Any],
) -> List[str]:
    argv = [
        "python",
        "-m",
        BUILDER_MODULE,
        "--converted-ckpt",
        str(converted_ckpt),
        "--cache-dir",
        str(cache_dir),
        "--output-dir",
        str(output_dir),
        "--num-nodes",
        str(num_nodes),
        "--node-rank",
        str(node_rank),
        "--node-ips",
        node_ips if isinstance(node_ips, str) else ",".join(map(str, node_ips)),
    ]
    for key, flag in (
        ("daemon_socket", "--daemon-socket"),
        ("base_port", "--base-port"),
        ("transport", "--transport"),
        ("store_residency", "--store-residency"),
        ("store_staging_dir", "--store-staging-dir"),
    ):
        if key in config:
            argv += [flag, str(config[key])]
    if isinstance(config.get("rail_devices"), list):
        argv += ["--rail-devices", ",".join(map(str, config["rail_devices"]))]
    return argv


def rebuild_command(
    config_path: Path,
    config: Optional[Mapping[str, Any]] = None,
    *,
    converted_ckpt: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    manifest: Optional[Mapping[str, Any]] = None,
    node_rank: Optional[int] = None,
    num_nodes: Optional[int] = None,
) -> str:
    """The builder command that republishes the store ``config_path`` names.

    Node arguments come from the config (or, if it is unreadable, from the
    launch); the checkpoint comes from the launch, else from the manifest's
    record of the last build; anything unknowable is an explicit
    ``<placeholder>``.
    """
    config = dict(config or {})
    source = manifest.get("source", {}) if isinstance(manifest, Mapping) else {}
    if not isinstance(source, Mapping):
        source = {}
    model_config = source.get("model_config")
    recorded_cache_dir = (
        os.path.dirname(model_config["path"])
        if isinstance(model_config, Mapping) and model_config.get("path")
        else None
    )
    argv = _builder_argv(
        converted_ckpt=(
            os.path.abspath(converted_ckpt)
            if converted_ckpt is not None
            else source.get("converted_ckpt") or "<converted-ckpt-dir>"
        ),
        cache_dir=(
            os.path.abspath(cache_dir)
            if cache_dir is not None
            else recorded_cache_dir or "<hf-checkpoint-dir>"
        ),
        # A server-written effective config names the builder's config it was
        # staged from; the store is rebuilt there, not in the run directory.
        output_dir=os.path.dirname(
            os.path.abspath(config.get("staged_from") or config_path)
        ),
        num_nodes=config.get("num_nodes", num_nodes if num_nodes is not None else "<num-nodes>"),
        node_rank=config.get("node_rank", node_rank if node_rank is not None else "<node-rank>"),
        node_ips=config.get("node_ips") or "<node-ips>",
        config=config,
    )
    return shlex.join(argv)


def verify_store_manifest(
    config: Mapping[str, Any],
    config_path: Path,
    *,
    converted_ckpt: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Cheap load-time gate: the store is the complete output of one build.

    Requires the completion manifest the config names, the manifest's node
    identity and byte counts to equal the config's, the store file to exist
    at exactly the manifest's size, and ``metadata.tsv`` to hash to the
    manifest's digest. Raises :class:`K3StoreManifestError`; never reads the
    store's bytes (see :func:`spot_check_store` for that).
    """

    def refuse(reason: str, manifest: Optional[Mapping[str, Any]] = None):
        raise K3StoreManifestError(
            f"distributed weight store of {config_path} is not usable: "
            f"{reason}. Rebuild it with: "
            + rebuild_command(
                config_path,
                config,
                converted_ckpt=converted_ckpt,
                cache_dir=cache_dir,
                manifest=manifest,
            )
        )

    manifest_value = config.get("manifest_path")
    if not manifest_value:
        refuse(
            "the config names no manifest_path, so the store was not "
            "published by the store builder"
        )
    manifest_path = Path(str(manifest_value))
    try:
        raw = manifest_path.read_text()
    except FileNotFoundError:
        refuse(
            f"there is no completion manifest at {manifest_path}: the build "
            "that wrote this config never finished publishing (interrupted, "
            "failed, or still running)"
        )
    except OSError as error:
        refuse(f"cannot read {manifest_path}: {error}")
    try:
        manifest = json.loads(raw)
    except ValueError as error:
        refuse(f"{manifest_path} is not valid JSON: {error}")
    if not isinstance(manifest, dict):
        refuse(f"{manifest_path} is not a JSON object")
    if (
        manifest.get("format") != MANIFEST_FORMAT
        or manifest.get("format_version") != MANIFEST_FORMAT_VERSION
    ):
        refuse(
            f"{manifest_path} is format {manifest.get('format')!r} version "
            f"{manifest.get('format_version')!r}, expected {MANIFEST_FORMAT!r} "
            f"version {MANIFEST_FORMAT_VERSION}",
            manifest,
        )
    if manifest.get("complete") is not True:
        refuse(f"{manifest_path} does not mark the build complete", manifest)
    try:
        store_entry = manifest["store"]
        metadata_entry = manifest["metadata"]
        pairs = (
            ("node_rank", manifest["node_rank"], config["node_rank"]),
            ("num_nodes", manifest["num_nodes"], config["num_nodes"]),
            ("store_bytes", store_entry["bytes"], config["store_bytes"]),
            (
                "replicated_bytes",
                store_entry["replicated_bytes"],
                config["replicated_bytes"],
            ),
            ("module_bytes", store_entry["module_bytes"], config["module_bytes"]),
        )
        published_metadata = (int(metadata_entry["bytes"]), str(metadata_entry["digest"]))
        for name, published, configured in pairs:
            if int(published) != int(configured):
                refuse(
                    f"the manifest records {name}={published} but the config "
                    f"says {configured}: they belong to different builds",
                    manifest,
                )
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, K3StoreManifestError):
            raise
        refuse(f"{manifest_path} is malformed ({error!r})", manifest)

    store_path = Path(str(config["store_path"]))
    try:
        size = store_path.stat().st_size
    except OSError as error:
        refuse(f"the store {store_path} is not readable: {error}", manifest)
    if size != int(store_entry["bytes"]):
        refuse(
            f"the store {store_path} is {size} B but the manifest records "
            f"{store_entry['bytes']} B (a partial or foreign file)",
            manifest,
        )
    metadata_path = Path(str(config["metadata_path"]))
    try:
        metadata = metadata_path.read_bytes()
    except OSError as error:
        refuse(f"cannot read {metadata_path}: {error}", manifest)
    if (len(metadata), _digest_bytes(metadata)) != published_metadata:
        refuse(
            f"{metadata_path} does not hash to the manifest's metadata "
            "digest: it belongs to a different build",
            manifest,
        )
    return manifest


def spot_check_store(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    samples: int = PREFLIGHT_SPOT_CHECK_SEGMENTS,
    read_buffer_bytes: int = DEFAULT_READ_BUFFER_BYTES,
) -> List[int]:
    """Re-hash a sample of the store's segments against the manifest.

    Reads ``metadata.tsv`` under the engine's row contract
    (:func:`~.distributed_weight_store.read_distributed_metadata`), requires
    this node's rows to tile the store, and re-hashes the first, middle and
    last segments. Returns the checked segment indices. Raises
    :class:`K3StoreBuilderError` or
    :class:`~.distributed_weight_store.DistributedMetadataError`.
    """
    segments = manifest["store"]["segments"]
    spans_by_segment = manifest_segment_spans(config, manifest)
    chosen = sorted({0, len(segments) // 2, len(segments) - 1})[: max(samples, 0)]
    store_path = str(config["store_path"])
    try:
        descriptor = os.open(store_path, os.O_RDONLY)
    except OSError as error:
        raise K3StoreBuilderError(f"cannot open {store_path}: {error}") from error
    try:
        for index in chosen:
            segment = segments[index]
            begin, length = int(segment["offset"]), int(segment["bytes"])
            try:
                digest = hash_store_segment(
                    descriptor, spans_by_segment[index], read_buffer_bytes
                )
            except (OSError, EOFError) as error:
                raise K3StoreBuilderError(
                    f"cannot read segment {index} of {store_path}: {error}"
                ) from error
            if digest != segment["digest"]:
                raise K3StoreBuilderError(
                    f"segment {index} of {store_path} ([{begin}, "
                    f"{begin + length})) does not hash to the manifest: the "
                    "store bytes changed after the build published them"
                )
    finally:
        os.close(descriptor)
    return chosen


def manifest_segment_spans(
    config: Mapping[str, Any], manifest: Mapping[str, Any]
) -> List[List[Span]]:
    """This node's tensor spans grouped into the manifest's store segments.

    Reads ``metadata.tsv`` under the engine's row contract, requires this
    node's rows to tile the store exactly, and requires every manifest
    segment to start on a row and cover a whole number of rows. A segment's
    manifest digest is the rollup of its spans' tensor digests in this order
    (:func:`verify_store`), so these spans are what any re-hash of the store
    -- :func:`spot_check_store`, or a staging copy -- must hash. Raises
    :class:`K3StoreBuilderError` or
    :class:`~.distributed_weight_store.DistributedMetadataError`.
    """
    node_rank = int(config["node_rank"])
    store_bytes = int(config["store_bytes"])
    rows = read_distributed_metadata(
        config["metadata_path"],
        store_bytes=store_bytes,
        num_nodes=int(config["num_nodes"]),
        n_experts=int(manifest["n_experts"]),
        module_bytes=int(config["module_bytes"]),
    )
    local = sorted(
        (
            (row.index, row.compact_offset, row.byte_size)
            for row in rows
            if row.owner in (REPLICATED_OWNER, node_rank)
        ),
        key=lambda span: span[1],
    )
    cursor = 0
    for row, offset, byte_size in local:
        if offset != cursor:
            raise K3StoreBuilderError(
                f"node {node_rank}'s metadata rows do not tile the store: row "
                f"{row} starts at {offset}, expected {cursor}"
            )
        cursor += byte_size
    if cursor != store_bytes:
        raise K3StoreBuilderError(
            f"node {node_rank}'s metadata rows cover {cursor} B of a "
            f"{store_bytes} B store"
        )
    segments = manifest["store"]["segments"]
    if not segments:
        raise K3StoreBuilderError("the manifest records no store segments")
    starts = [span[1] for span in local]
    out: List[List[Span]] = []
    for index, segment in enumerate(segments):
        begin, length = int(segment["offset"]), int(segment["bytes"])
        first = _bisect_left(starts, begin)
        spans: List[Span] = []
        covered = 0
        for span in local[first:]:
            if covered >= length:
                break
            spans.append(span)
            covered += span[2]
        if (
            not spans
            or spans[0][1] != begin
            or covered != length
            or len(spans) != int(segment["tensors"])
        ):
            raise K3StoreBuilderError(
                f"manifest segment {index} [{begin}, {begin + length}) does "
                "not align with this node's metadata rows"
            )
        out.append(spans)
    return out


def _bisect_left(values: Sequence[int], target: int) -> int:
    low, high = 0, len(values)
    while low < high:
        middle = (low + high) // 2
        if values[middle] < target:
            low = middle + 1
        else:
            high = middle
    return low


def model_config_identity(cache_dir: Path) -> Dict[str, str]:
    config_json = Path(os.path.abspath(cache_dir)) / "config.json"
    try:
        raw = config_json.read_bytes()
    except OSError as error:
        raise K3StoreBuilderError(
            f"cannot read the Kimi-K3 model config {config_json}: {error}"
        ) from error
    return {"path": str(config_json), "digest": _digest_bytes(raw)}


def verify_published_store(
    config_path: Path,
    *,
    converted_ckpt: Optional[Path],
    cache_dir: Optional[Path] = None,
    node_rank: Optional[int] = None,
    num_nodes: Optional[int] = None,
    spot_check_segments: int = PREFLIGHT_SPOT_CHECK_SEGMENTS,
    spot_check_store_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Launch preflight for a distributed store.

    Everything :func:`verify_store_manifest` checks, plus: the store was
    built from the converted checkpoint this launch names (content
    fingerprint), from the ``config.json`` under the launch's ``cache_dir``
    when it names one, and a sample of its segments still hashes to the
    manifest. Every refusal is a :class:`K3StoreManifestError` ending in the
    exact rebuild command.

    ``spot_check_store_path`` samples another copy of the same build instead
    of ``store_path``: the complete staged copy the server is about to map,
    so a resident restart reads nothing from the source.
    """
    config_path = Path(config_path)

    def command(config=None, manifest=None) -> str:
        return rebuild_command(
            config_path,
            config,
            converted_ckpt=converted_ckpt,
            cache_dir=cache_dir,
            manifest=manifest,
            node_rank=node_rank,
            num_nodes=num_nodes,
        )

    try:
        config = read_distributed_weight_config(config_path)
    except (OSError, ValueError) as error:
        raise K3StoreManifestError(
            f"distributed weight config {config_path} is unusable: {error}. "
            f"Rebuild it with: {command()}"
        ) from error
    manifest = verify_store_manifest(
        config, config_path, converted_ckpt=converted_ckpt, cache_dir=cache_dir
    )

    def refuse(reason: str, cause: Optional[BaseException] = None):
        raise K3StoreManifestError(
            f"distributed weight store of {config_path} is not usable: "
            f"{reason}. Rebuild it with: {command(config, manifest)}"
        ) from cause

    if converted_ckpt is None:
        refuse(
            "the launch names no converted checkpoint (--converted-ckpt-dir "
            "or --cache-dir), so the store's checkpoint identity cannot be "
            "verified"
        )
    try:
        identity = checkpoint_identity(Path(converted_ckpt))
    except K3StoreBuilderError as error:
        refuse(
            f"the launch's converted checkpoint {converted_ckpt} cannot be "
            f"identified: {error}",
            error,
        )
    source = manifest.get("source") or {}
    if identity["fingerprint"] != source.get("fingerprint"):
        refuse(
            f"the store was built from {source.get('converted_ckpt')} "
            f"(fingerprint {source.get('fingerprint')}) but this launch names "
            f"{identity['converted_ckpt']} (fingerprint "
            f"{identity['fingerprint']}): the store is stale"
        )
    if cache_dir is not None:
        try:
            launch_config = model_config_identity(Path(cache_dir))
        except K3StoreBuilderError as error:
            refuse(str(error), error)
        recorded = (source.get("model_config") or {}).get("digest")
        if launch_config["digest"] != recorded:
            refuse(
                f"{launch_config['path']} is not the config.json the store was "
                f"built from ({(source.get('model_config') or {}).get('path')})"
            )
    sampled = config
    if spot_check_store_path is not None:
        sampled = {**config, "store_path": str(spot_check_store_path)}
    try:
        checked = spot_check_store(sampled, manifest, samples=spot_check_segments)
    except (K3StoreBuilderError, DistributedMetadataError) as error:
        if spot_check_store_path is not None:
            # A corrupt staged copy is not fixed by rebuilding the source.
            raise K3StoreManifestError(
                f"the staged copy {spot_check_store_path} of {config_path} "
                f"fails its spot check: {error}. Stop every server mapping it "
                f"and remove {Path(spot_check_store_path).parent}; the next "
                "start stages it again from the source"
            ) from error
        refuse(str(error), error)
    return {"config": config, "manifest": manifest, "spot_checked_segments": checked}


def staged_store_id(node_rank: int, manifest_bytes: bytes) -> str:
    return f"node{int(node_rank)}-{hashlib.sha256(manifest_bytes).hexdigest()[:16]}"


def complete_staged_store(config_path: Path | str) -> Optional[Path]:
    """The ``store.bin`` of this config's ``COMPLETE`` staged copy, if one
    exists; None for a legacy config, an unreadable manifest, or no copy.

    Lets the launch preflight sample the copy the server will map instead of
    re-reading the source on a resident restart. It takes no lock: staging
    re-checks the copy under ``use.lock`` before mapping it. An unreadable
    config is left to the preflight, which names the rebuild command.
    """
    try:
        config = read_distributed_weight_config(config_path)
    except (OSError, ValueError):
        return None
    if config["store_residency"] is None:
        return None
    try:
        manifest_bytes = Path(str(config.get("manifest_path"))).read_bytes()
    except OSError:
        return None
    directory = Path(config["store_staging_dir"]) / staged_store_id(
        config["node_rank"], manifest_bytes
    )
    if not (directory / STAGED_COMPLETE_FILE_NAME).is_file():
        return None
    return directory / STORE_FILE_NAME


# --------------------------------------------------------------------------- #
#  8. Atomic publication                                                      #
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def _build_lock(output_dir: Path) -> Iterator[None]:
    """One build per output directory; released when the process dies."""
    lock_path = output_dir / LOCK_FILE_NAME
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise K3StoreBuilderError(
                f"another store build holds {lock_path}; wait for it to finish"
            ) from error
        yield
    finally:
        os.close(descriptor)


def _sweep_staging_residue(output_dir: Path) -> None:
    """Remove staging directories of builds that died (lock is held)."""
    for path in sorted(output_dir.glob(STAGING_PREFIX + "*")):
        if path.is_dir() and not path.is_symlink():
            logger.warning("removing residue of an interrupted build: %s", path)
            shutil.rmtree(path)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_file(path: Path, data: bytes) -> None:
    with open(path, "xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _publish(staging: Path, output_dir: Path, config_name: str) -> None:
    """Move a verified build into place; the manifest rename commits it."""
    final_manifest = output_dir / MANIFEST_FILE_NAME
    # Retract the previous commit point BEFORE replacing anything it
    # describes, so a crash part-way through leaves no manifest at all.
    if final_manifest.exists() or final_manifest.is_symlink():
        final_manifest.unlink()
        _fsync_directory(output_dir)
    for name in (STORE_FILE_NAME, METADATA_FILE_NAME, config_name):
        os.replace(staging / name, output_dir / name)
    _fsync_directory(output_dir)
    os.replace(staging / MANIFEST_FILE_NAME, final_manifest)
    _fsync_directory(output_dir)


def _refuse_foreign_node_configs(output_dir: Path, node_rank: int) -> None:
    if not output_dir.is_dir():
        return
    for path in sorted(output_dir.glob("node*.json")):
        match = _NODE_CONFIG_NAME.fullmatch(path.name)
        if match and int(match.group(1)) != node_rank:
            raise K3StoreBuilderError(
                f"{output_dir} already holds {path.name}, the config of a "
                f"different node rank, which this node{node_rank} build would "
                "invalidate; build each node into its own --output-dir"
            )


def _check_dry_run_target(
    output_dir: Path,
    plan: StorePlan,
    fingerprint: str,
    metadata: bytes,
    config_name: str,
) -> None:
    present = [
        name
        for name in (STORE_FILE_NAME, METADATA_FILE_NAME, config_name, MANIFEST_FILE_NAME)
        if (output_dir / name).exists()
    ]
    if not present:
        logger.info("--dry-run: %s holds no published store", output_dir)
        return
    manifest_path = output_dir / MANIFEST_FILE_NAME
    if not manifest_path.is_file():
        raise K3StoreBuilderError(
            f"--dry-run refuses: {output_dir} holds {present} but no "
            f"{MANIFEST_FILE_NAME}, so they are what an unfinished build left "
            "behind and nothing may adopt them; run the build without "
            "--dry-run to replace them atomically"
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as error:
        raise K3StoreBuilderError(
            f"--dry-run refuses: cannot read {manifest_path}: {error}"
        ) from error
    try:
        observed = {
            "format": (manifest.get("format"), MANIFEST_FORMAT),
            "format_version": (manifest.get("format_version"), MANIFEST_FORMAT_VERSION),
            "complete": (manifest.get("complete"), True),
            "node_rank": (manifest.get("node_rank"), plan.node_rank),
            "num_nodes": (manifest.get("num_nodes"), plan.num_nodes),
            "store bytes": (manifest["store"]["bytes"], plan.store_bytes),
            "checkpoint fingerprint": (manifest["source"]["fingerprint"], fingerprint),
            "metadata digest": (manifest["metadata"]["digest"], _digest_bytes(metadata)),
        }
    except (KeyError, TypeError, AttributeError) as error:
        raise K3StoreBuilderError(
            f"--dry-run refuses: {manifest_path} is malformed ({error!r})"
        ) from error
    mismatches = [
        f"{name} {published!r} != {planned!r}"
        for name, (published, planned) in observed.items()
        if published != planned
    ]
    if mismatches:
        raise K3StoreBuilderError(
            f"--dry-run refuses: the store published in {output_dir} does not "
            f"match this plan ({'; '.join(mismatches)}); run the build without "
            "--dry-run to replace it atomically"
        )
    logger.info("--dry-run: the store published in %s matches this plan", output_dir)


# --------------------------------------------------------------------------- #
#  9. CLI                                                                      #
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
        prog=f"python -m {BUILDER_MODULE}",
        description=(
            "Build one node's Kimi-K3 compact distributed host-weight store."
        ),
    )
    parser.add_argument("--converted-ckpt", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help=(
            "The Kimi-K3 checkpoint directory holding config.json (the "
            "server's --cache-dir); its tensor map is the completeness "
            "contract the converted checkpoint must meet."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--num-nodes", type=int, required=True, choices=DISTRIBUTED_NODE_COUNTS
    )
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument("--node-ips", type=_parse_csv, required=True)
    parser.add_argument("--daemon-socket", type=Path, default=None)
    parser.add_argument("--base-port", type=int, default=DEFAULT_BASE_PORT)
    parser.add_argument(
        "--transport", choices=WEIGHT_TRANSPORTS, default=WEIGHT_TRANSPORTS[0]
    )
    parser.add_argument(
        "--rail-devices", type=_parse_csv, default=list(H200_RAIL_DEVICES)
    )
    parser.add_argument(
        "--store-residency",
        choices=STORE_RESIDENCIES,
        default=STORE_RESIDENCIES[0],
        help=(
            "Lifetime of the server's staged tmpfs copy of store.bin: 'run' "
            "removes it when the server stops; 'resident' keeps it so the "
            "next start maps it without copying this output again. The "
            "output written here is the source either way and is never "
            "deleted by the server."
        ),
    )
    parser.add_argument(
        "--store-staging-dir",
        default=DEFAULT_STORE_STAGING_DIR,
        help=(
            "Absolute tmpfs directory the server stages the store into "
            f"(default {DEFAULT_STORE_STAGING_DIR})."
        ),
    )
    parser.add_argument(
        "--read-buffer-bytes", type=int, default=DEFAULT_READ_BUFFER_BYTES
    )
    parser.add_argument(
        "--hash-workers", type=int, default=DEFAULT_HASH_WORKERS
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate the checkpoint and report the sizes the real build "
            "would produce. Writes nothing, and refuses an output directory "
            "that holds a store without a matching completion manifest."
        ),
    )
    return parser


def _load_model_config(cache_dir: Path):
    identity = model_config_identity(cache_dir)
    try:
        cfg = load_k3_config(str(cache_dir))
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise K3StoreBuilderError(
            f"{identity['path']} is not a loadable Kimi-K3 config: {error}"
        ) from error
    return cfg, identity


def build(
    argv: Sequence[str] | None = None,
    expected_topology: Tuple[int, int] = (K3_LAYERS, K3_EXPERTS),
) -> StorePlan:
    """Run one build. Returns the plan; raises on any validation failure.

    ``expected_topology`` is forwarded to :func:`build_store_plan` so the unit
    tests can drive this whole path with a mini checkpoint. :func:`main` never
    passes it, so the CLI always enforces the engine's 92 x 896.

    Nothing under ``--output-dir`` changes until the checkpoint, the plan and
    every argument have been validated.
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
    if args.read_buffer_bytes <= 0:
        raise K3StoreBuilderError("--read-buffer-bytes must be positive")
    if args.hash_workers <= 0:
        raise K3StoreBuilderError("--hash-workers must be positive")
    if not os.path.isabs(args.store_staging_dir):
        raise K3StoreBuilderError(
            f"--store-staging-dir must be an absolute path; got "
            f"{args.store_staging_dir!r}"
        )

    output_dir = Path(os.path.abspath(args.output_dir))
    converted_ckpt = Path(os.path.abspath(args.converted_ckpt))
    cache_dir = Path(os.path.abspath(args.cache_dir))
    cfg, model_config = _load_model_config(cache_dir)
    shards = read_shard_metadata(converted_ckpt)
    plan = build_store_plan(
        converted_ckpt=converted_ckpt,
        num_nodes=args.num_nodes,
        node_rank=args.node_rank,
        cfg=cfg,
        shards=shards,
        expected_topology=expected_topology,
    )
    fingerprint = checkpoint_fingerprint(shards)

    config_name = f"node{args.node_rank}.json"
    config = node_config(
        plan,
        store_path=output_dir / STORE_FILE_NAME,
        metadata_path=output_dir / METADATA_FILE_NAME,
        manifest_path=output_dir / MANIFEST_FILE_NAME,
        node_ips=args.node_ips,
        daemon_socket=(
            Path(os.path.abspath(args.daemon_socket))
            if args.daemon_socket
            else output_dir / "daemon.sock"
        ),
        base_port=args.base_port,
        transport=args.transport,
        rail_devices=args.rail_devices,
        store_residency=args.store_residency,
        store_staging_dir=args.store_staging_dir,
    )
    metadata = metadata_bytes(plan)
    command = shlex.join(
        _builder_argv(
            converted_ckpt=converted_ckpt,
            cache_dir=cache_dir,
            output_dir=output_dir,
            num_nodes=args.num_nodes,
            node_rank=args.node_rank,
            node_ips=args.node_ips,
            config=config,
        )
    )

    logger.info(
        "node %d/%d: %d tensors (%d replicated, %d expert rows, %d ignored "
        "in the skeleton), experts_per_owner=%d",
        plan.node_rank,
        plan.num_nodes,
        len(plan.records),
        len(plan.replicated_records),
        len(plan.expert_records),
        plan.ignored_tensors,
        plan.experts_per_owner,
    )
    logger.info(
        "module_bytes=%d replicated_bytes=%d store_bytes=%d (%.2f GiB)",
        plan.module_bytes,
        plan.replicated_bytes,
        plan.store_bytes,
        plan.store_bytes / float(1 << 30),
    )
    _refuse_foreign_node_configs(output_dir, args.node_rank)

    if args.dry_run:
        _check_dry_run_target(output_dir, plan, fingerprint, metadata, config_name)
        logger.info("--dry-run: nothing written; the build command is: %s", command)
        return plan

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with _build_lock(output_dir):
            _sweep_staging_residue(output_dir)
            staging = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=output_dir))
            try:
                store_path = staging / STORE_FILE_NAME
                copy = write_store(plan, store_path, args.read_buffer_bytes)
                self_check(plan, store_path, copy.written)
                segments = verify_store(
                    plan, store_path, copy, args.read_buffer_bytes, args.hash_workers
                )
                manifest = build_manifest(
                    plan,
                    segments=segments,
                    copy=copy,
                    fingerprint=fingerprint,
                    model_config=model_config,
                    metadata=metadata,
                    build_command=command,
                )
                _write_new_file(staging / METADATA_FILE_NAME, metadata)
                _write_new_file(
                    staging / config_name,
                    (json.dumps(config, indent=2, sort_keys=True) + "\n").encode(),
                )
                _write_new_file(
                    staging / MANIFEST_FILE_NAME,
                    (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
                )
                _publish(staging, output_dir, config_name)
            finally:
                # After a successful publish only the empty directory is left;
                # after a failure, everything not yet published goes with it.
                shutil.rmtree(staging, ignore_errors=True)
    except OSError as error:
        raise K3StoreBuilderError(
            f"filesystem error while building into {output_dir}: {error}"
        ) from error

    logger.info(
        "published %s (%d B, %d segments verified) with %s and %s",
        output_dir / STORE_FILE_NAME,
        plan.store_bytes,
        len(segments),
        output_dir / config_name,
        output_dir / MANIFEST_FILE_NAME,
    )
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
