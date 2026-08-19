"""Kimi-K3 compact distributed host-weight store helpers."""

from __future__ import annotations

import gc
import json
import mmap
import os
from pathlib import Path
from typing import Any

import torch


_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
}


def load_distributed_weight_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = json.loads(config_path.read_text())
    required = {
        "node_rank",
        "node_ips",
        "store_path",
        "metadata_path",
        "daemon_socket",
        "summary_path",
        "store_bytes",
        "replicated_bytes",
        "module_bytes",
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
    store = Path(config["store_path"])
    if store.stat().st_size != int(config["store_bytes"]):
        raise ValueError("distributed compact-store byte size mismatch")
    metadata = Path(config["metadata_path"])
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    return config


def save_compact_skeleton_state_dict(
    config_path: str | Path,
    output_path: str | Path,
) -> tuple[int, int]:
    """Serialize skeleton tensors directly from the compact node store."""
    config = load_distributed_weight_config(config_path)
    store_path = Path(config["store_path"])
    metadata_path = Path(config["metadata_path"])
    store_bytes = int(config["store_bytes"])

    fd = os.open(store_path, os.O_RDONLY)
    mapping = mmap.mmap(fd, store_bytes, access=mmap.ACCESS_READ)
    state_dict: dict[str, torch.Tensor] = {}
    try:
        with metadata_path.open() as handle:
            for raw in handle:
                fields = raw.rstrip("\n").split("\t")
                if fields[0] == "H":
                    continue
                if len(fields) != 9 or fields[0] != "T":
                    raise ValueError("invalid distributed metadata row")
                (
                    _tag,
                    checkpoint_name,
                    module_key,
                    _tensor_key,
                    _owner,
                    offset,
                    byte_size,
                    dtype_name,
                    shape_json,
                ) = fields
                if module_key != "__skeleton__":
                    continue
                dtype = _DTYPES.get(dtype_name)
                if dtype is None:
                    raise ValueError(
                        f"unsupported compact skeleton dtype {dtype_name!r}"
                    )
                offset_i = int(offset)
                byte_size_i = int(byte_size)
                shape = tuple(json.loads(shape_json))
                view = memoryview(mapping)[
                    offset_i : offset_i + byte_size_i
                ]
                tensor = torch.frombuffer(view, dtype=dtype)
                if tensor.numel() * tensor.element_size() != byte_size_i:
                    raise ValueError(
                        f"{checkpoint_name}: shape/dtype byte mismatch"
                    )
                state_dict[checkpoint_name] = tensor.reshape(shape)

        torch.save(state_dict, output_path)
        output_bytes = Path(output_path).stat().st_size
        return len(state_dict), output_bytes
    finally:
        state_dict.clear()
        if "tensor" in locals():
            del tensor
        if "view" in locals():
            del view
        gc.collect()
        mapping.close()
        os.close(fd)
