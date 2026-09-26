# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                            #
# ---------------------------------------------------------------------------- #

"""Size the GLM-5 weight shared memory from the converted checkpoint metadata.

Mirrors the layout that ``Parameter_Server::_load_cus_format_file_to_host_mem``
(core/Parameter_Server/Parameter_Server.cpp) writes: every directory entry whose
name contains ``.json`` is a shard metadata file, each shard occupies
``total_byte_size`` bytes, and the running offset is rounded up to 4 KiB after
each shard. Validating here keeps a malformed checkpoint from being discovered
only after the C++ side has already allocated hundreds of GiB of SHM.

Stdlib only — this module must stay importable without torch.
"""

import json
import os

_SHARD_ALIGNMENT = 4096
_TOTAL_ALIGNMENT = 2 * 1024 * 1024


def _round_up(value, alignment):
    return ((value + alignment - 1) // alignment) * alignment


def _checked_int(value, path, field):
    # bool is an int subclass; a JSON `true` is never a valid size.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}: '{field}' must be an integer, got {value!r}")
    return value


def _shard_byte_size(metadata_path):
    """Validate one shard metadata file and return its 4 KiB-aligned footprint."""
    try:
        with open(metadata_path, "r") as metadata_file:
            metadata = json.load(metadata_file)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Failed to parse checkpoint metadata: {metadata_path} - {exc}"
        ) from exc

    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_path}: metadata must be a JSON object")
    for field in ("file_name", "total_byte_size", "state_dict"):
        if field not in metadata:
            raise ValueError(f"{metadata_path}: missing '{field}'")

    file_name = metadata["file_name"]
    if not isinstance(file_name, str) or not file_name:
        raise ValueError(f"{metadata_path}: 'file_name' must be a non-empty string")

    total_byte_size = _checked_int(
        metadata["total_byte_size"], metadata_path, "total_byte_size"
    )
    if total_byte_size <= 0:
        raise ValueError(
            f"{metadata_path}: 'total_byte_size' must be positive, got {total_byte_size}"
        )

    state_dict = metadata["state_dict"]
    if not isinstance(state_dict, dict):
        raise ValueError(f"{metadata_path}: 'state_dict' must be a JSON object")

    tensor_total = 0
    for tensor_name, tensor_info in state_dict.items():
        if not isinstance(tensor_info, dict) or "byte_size" not in tensor_info:
            raise ValueError(
                f"{metadata_path}: tensor '{tensor_name}' has no 'byte_size'"
            )
        byte_size = _checked_int(
            tensor_info["byte_size"], metadata_path, f"state_dict.{tensor_name}.byte_size"
        )
        if byte_size < 0:
            raise ValueError(
                f"{metadata_path}: tensor '{tensor_name}' has negative 'byte_size' {byte_size}"
            )
        tensor_total += byte_size

    if tensor_total != total_byte_size:
        raise ValueError(
            f"{metadata_path}: tensor byte sizes sum to {tensor_total}, "
            f"but 'total_byte_size' is {total_byte_size}"
        )

    bin_path = os.path.join(os.path.dirname(metadata_path), file_name)
    try:
        bin_byte_size = os.path.getsize(bin_path)
    except OSError as exc:
        raise ValueError(
            f"{metadata_path}: cannot read weight file {bin_path} - {exc}"
        ) from exc
    if bin_byte_size != total_byte_size:
        raise ValueError(
            f"{metadata_path}: weight file {bin_path} is {bin_byte_size} bytes, "
            f"expected {total_byte_size}"
        )

    return _round_up(total_byte_size, _SHARD_ALIGNMENT)


def converted_checkpoint_byte_size(converted_ckpt_dir):
    """Return the SHM bytes the C++ parameter server needs for this checkpoint.

    Raises ValueError naming the offending path if the directory is unreadable,
    holds no shard metadata, or any shard is inconsistent with its .bin file.
    """
    try:
        entries = sorted(os.listdir(converted_ckpt_dir))
    except OSError as exc:
        raise ValueError(
            f"Failed to open converted checkpoint directory: {converted_ckpt_dir} - {exc}"
        ) from exc

    total_byte_size = 0
    shard_count = 0
    for entry in entries:
        if ".json" not in entry:
            continue
        total_byte_size += _shard_byte_size(os.path.join(converted_ckpt_dir, entry))
        shard_count += 1

    if shard_count == 0:
        raise ValueError(
            f"No checkpoint metadata (.json) found in {converted_ckpt_dir}"
        )

    return _round_up(total_byte_size, _TOTAL_ALIGNMENT)
