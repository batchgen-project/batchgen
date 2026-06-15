from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


TensorMeta = Dict[str, object]
MetadataMap = Dict[str, TensorMeta]


def load_checkpoint_metadata(
    converted_ckpt_dir: str | Path,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> MetadataMap:
    converted_ckpt_dir = Path(converted_ckpt_dir)
    if not converted_ckpt_dir.is_dir():
        raise FileNotFoundError(
            f"converted_ckpt_dir not found or not a directory: {converted_ckpt_dir}"
        )

    json_files = _select_metadata_files(converted_ckpt_dir, rank, world_size)
    if not json_files:
        raise FileNotFoundError(
            f"No metadata JSON files found in {converted_ckpt_dir} "
            f"(rank={rank}, world_size={world_size})"
        )

    metadata: MetadataMap = {}
    for json_path in json_files:
        with open(json_path) as fh:
            payload = json.load(fh)
        shard = payload.get("state_dict", payload)
        if not isinstance(shard, dict):
            raise ValueError(
                f"Unexpected metadata layout in {json_path}: state_dict is not a dict"
            )
        for tensor_name, meta in shard.items():
            if (
                not isinstance(meta, dict)
                or "dtype" not in meta
                or "shape" not in meta
            ):
                logging.warning(
                    "metadata_loader: skipping malformed entry %s in %s",
                    tensor_name,
                    json_path,
                )
                continue
            metadata[str(tensor_name)] = dict(meta)

    return metadata


def _select_metadata_files(
    converted_ckpt_dir: Path,
    rank: Optional[int],
    world_size: Optional[int],
) -> List[Path]:
    if rank is None:
        return sorted(converted_ckpt_dir.glob("*.json"))

    expected_basenames: List[str] = [f"model{rank}"]
    if world_size is not None:
        expected_basenames.append(f"model{rank}-mp{world_size}")

    candidates: List[Path] = []
    for basename in expected_basenames:
        match = converted_ckpt_dir / f"{basename}.json"
        if match.is_file():
            candidates.append(match)

    if candidates:
        return candidates

    return sorted(converted_ckpt_dir.glob(f"model{rank}*.json"))


def load_rank_shard_tensors(
    converted_ckpt_dir: str | Path,
    rank: int,
    world_size: Optional[int],
    tensor_names: Iterable[str],
):
    import torch

    converted_ckpt_dir = Path(converted_ckpt_dir)
    json_files = _select_metadata_files(converted_ckpt_dir, rank, world_size)
    if not json_files:
        raise FileNotFoundError(
            f"No metadata JSON files found in {converted_ckpt_dir} "
            f"(rank={rank}, world_size={world_size})"
        )

    wanted = set(tensor_names)
    result: Dict[str, "torch.Tensor"] = {}
    for json_path in json_files:
        with open(json_path) as fh:
            payload = json.load(fh)
        shard = payload.get("state_dict", payload)
        bin_path = json_path.with_suffix(".bin")
        if not bin_path.is_file():
            raise FileNotFoundError(f"Shard binary not found: {bin_path}")
        for name in list(wanted):
            meta = shard.get(name)
            if meta is None:
                continue
            byte_size = int(meta["byte_size"])
            offset = int(meta["offset"])
            with open(bin_path, "rb") as bf:
                bf.seek(offset)
                raw = bf.read(byte_size)
            torch_dtype = resolve_torch_dtype(str(meta["dtype"]))
            tensor = (
                torch.frombuffer(bytearray(raw), dtype=torch.uint8)
                .view(torch_dtype)
                .reshape(tuple(meta["shape"]))
            )
            result[name] = tensor
            wanted.discard(name)

    if wanted:
        raise KeyError(
            f"Tensors {sorted(wanted)} not found in rank {rank} shard "
            f"under {converted_ckpt_dir}"
        )
    return result


def build_module_metadata(
    tensor_metadata: MetadataMap,
    state_dict_name_map: Dict[str, Dict[str, str]],
) -> Dict[str, Dict[str, TensorMeta]]:
    module_meta: Dict[str, Dict[str, TensorMeta]] = {}
    for ckpt_name, routing in state_dict_name_map.items():
        module_type = routing.get("module_type") or _module_key_to_type(
            routing["module_key"]
        )
        tensor_key = routing["tensor_key"]
        if module_type is None:
            continue
        meta = tensor_metadata.get(ckpt_name)
        if meta is None:
            continue
        per_type = module_meta.setdefault(module_type, {})
        existing = per_type.get(tensor_key)
        if existing is not None and (
            existing.get("shape") != meta.get("shape")
            or existing.get("dtype") != meta.get("dtype")
            or existing.get("byte_size") != meta.get("byte_size")
        ):
            raise ValueError(
                f"Inconsistent metadata for ({module_type}, {tensor_key}): "
                f"existing={existing} new={meta} from ckpt_name={ckpt_name}"
            )
        per_type[tensor_key] = meta
    return module_meta


def _module_key_to_type(module_key: str) -> Optional[str]:
    if module_key.startswith("attn_"):
        return "attn"
    if module_key.startswith("routed_expert_"):
        return "routed_expert"
    if module_key.startswith("shared_expert_"):
        return "shared_expert"
    return None


_DTYPE_STR_TO_TORCH: Dict[str, str] = {
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float8_e4m3fn": "float8_e4m3fn",
    "float8_e8m0fnu": "float8_e8m0fnu",
    "float4_e2m1fn_x2": "float4_e2m1fn_x2",
    "int8": "int8",
    "uint8": "uint8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "float64": "float64",
}


def resolve_torch_dtype(dtype_str: str):
    import torch

    normalised = _DTYPE_STR_TO_TORCH.get(dtype_str)
    if normalised is None:
        raise ValueError(
            f"Unsupported tensor dtype '{dtype_str}' (extend metadata_loader._DTYPE_STR_TO_TORCH)"
        )
    torch_dtype = getattr(torch, normalised, None)
    if torch_dtype is None:
        raise RuntimeError(
            f"Installed torch lacks '{normalised}' dtype; upgrade torch or remove the requirement"
        )
    return torch_dtype


def diff_shapes(
    expected: Dict[str, Dict[str, TensorMeta]],
    declared: Dict[str, Dict[str, Iterable[int]]],
) -> List[Tuple[str, str, str]]:
    diffs: List[Tuple[str, str, str]] = []
    for module_type, tensors in expected.items():
        declared_for_type = declared.get(module_type, {})
        for tensor_key, meta in tensors.items():
            actual_shape = list(meta["shape"])
            declared_shape = declared_for_type.get(tensor_key)
            if declared_shape is None:
                diffs.append(
                    (
                        module_type,
                        tensor_key,
                        f"missing in declared, actual={actual_shape}",
                    )
                )
            elif list(declared_shape) != actual_shape:
                diffs.append(
                    (
                        module_type,
                        tensor_key,
                        f"declared={list(declared_shape)} actual={actual_shape}",
                    )
                )
        for tensor_key in declared_for_type:
            if tensor_key not in tensors:
                diffs.append(
                    (
                        module_type,
                        tensor_key,
                        "declared but absent in ckpt metadata",
                    )
                )
    return diffs
