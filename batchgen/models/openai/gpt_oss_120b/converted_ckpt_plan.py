# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                            #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
#                                                                               #
#  you may obtain a copy of the license at                                      #
#                                                                               #
#                  http://www.apache.org/licenses/license-2.0                   #
#                                                                               #
#  unless required by applicable law or agreed to in writing, software          #
#  distributed under the license is distributed on an "as is" basis,            #
#  without warranties or conditions of any kind, either express or implied.     #
#  see the license for the specific language governing permissions and          #
#  limitations under the license.                                               #
# ---------------------------------------------------------------------------- #

"""Checks on the GPT-OSS converted checkpoint directory, run before any read.

The C++ parameter server (core/Parameter_Server/Parameter_Server.cpp) reads
every entry whose name contains ".json" in the directory, and the .bin file
each one names, into one fixed weight buffer, back to back, each file starting
on a 4 KiB boundary. GPT-OSS loads only the set its own parameter server writes
on first launch (layer_<i>.bin/.json, non_layer.bin/.json, marker
.gpt_oss_converted). Shards written by ``python -m batchgen.tools.convert_checkpoint``
(one per source .safetensors, raw OpenAI tensor names) are never consumed by
GPT-OSS; next to the server set they double the load and overrun the buffer.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import List

WEIGHT_FILE_ALIGNMENT = 4096
_LAYER_JSON = re.compile(r"layer_(\d+)\.json")
_NON_LAYER_JSON = "non_layer.json"


class ConvertedCkptPlanError(RuntimeError):
    """Base class: the converted checkpoint directory cannot be loaded as planned."""


class MixedCheckpointFormatError(ConvertedCkptPlanError):
    """The directory holds loader-visible files outside the GPT-OSS server set."""


class CheckpointBufferOverflowError(ConvertedCkptPlanError):
    """The planned load does not fit the weight buffer."""


class CheckpointMetadataError(ConvertedCkptPlanError):
    """A metadata .json is unreadable or describes tensors outside its file."""


@dataclass(frozen=True)
class PlannedFile:
    json_name: str
    bin_name: str
    buffer_offset: int
    byte_size: int


def _loader_visible_names(converted_ckpt_dir: str) -> List[str]:
    # Same filter and order as the C++ loader: readdir order, name contains ".json".
    return [name for name in os.listdir(converted_ckpt_dir) if ".json" in name]


def _is_server_format(name: str, num_layers: int) -> bool:
    if name == _NON_LAYER_JSON:
        return True
    match = _LAYER_JSON.fullmatch(name)
    return match is not None and int(match.group(1)) < num_layers


def check_converted_ckpt_format(converted_ckpt_dir: str, num_layers: int) -> None:
    """Reject a directory holding loader-visible files outside the GPT-OSS server set.

    Runs before the server converts into the directory, so a directory holding
    only converter-tool shards is rejected too: converting next to them would
    produce the mixed directory.
    """
    if not os.path.isdir(converted_ckpt_dir):
        return
    foreign = sorted(
        name
        for name in _loader_visible_names(converted_ckpt_dir)
        if not _is_server_format(name, num_layers)
    )
    if not foreign:
        return
    companions = sorted(
        name[: -len(".json")] + ".bin"
        for name in foreign
        if name.endswith(".json")
        and os.path.exists(os.path.join(converted_ckpt_dir, name[: -len(".json")] + ".bin"))
    )
    raise MixedCheckpointFormatError(
        f"MixedCheckpointFormatError: {converted_ckpt_dir} holds {len(foreign)} metadata "
        f"file(s) that GPT-OSS does not load: {foreign}"
        + (f" (with data files {companions})" if companions else "")
        + ". GPT-OSS loads only the set its server writes from --cache-dir on first "
        f"launch: layer_<i>.bin/.json for i in 0..{num_layers - 1}, non_layer.bin/.json and "
        ".gpt_oss_converted. The listed files are other-format shards, such as the "
        "output of `python -m batchgen.tools.convert_checkpoint`; the C++ loader would "
        "read them into the same weight buffer as the server set. Remove those files "
        "(move them out of this directory); keep the layer_*/non_layer set."
    )


def plan_converted_ckpt_load(
    converted_ckpt_dir: str, buffer_bytes: int, num_layers: int
) -> List[PlannedFile]:
    """Plan the C++ load of ``converted_ckpt_dir`` into a ``buffer_bytes`` buffer.

    Mirrors the C++ loader's order and offsets. Raises before any weight byte is
    read: MixedCheckpointFormatError for a foreign file, CheckpointMetadataError
    for an unreadable .json or a tensor outside its file, and
    CheckpointBufferOverflowError when a file would end past the buffer.
    """
    check_converted_ckpt_format(converted_ckpt_dir, num_layers)
    if not os.path.isdir(converted_ckpt_dir):
        raise CheckpointMetadataError(
            f"CheckpointMetadataError: converted checkpoint directory {converted_ckpt_dir} "
            "does not exist after conversion; the C++ loader cannot open it"
        )

    entries = []
    for json_name in _loader_visible_names(converted_ckpt_dir):
        json_path = os.path.join(converted_ckpt_dir, json_name)
        try:
            with open(json_path, "r") as f:
                meta = json.load(f)
            bin_name = str(meta["file_name"])
            total = int(meta["total_byte_size"])
            tensors = [
                (name, int(info["offset"]), int(info["byte_size"]))
                for name, info in meta["state_dict"].items()
            ]
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            raise CheckpointMetadataError(
                f"CheckpointMetadataError: cannot read load metadata from {json_path}: {e!r}"
            ) from e
        if total < 0:
            raise CheckpointMetadataError(
                f"CheckpointMetadataError: {json_path} declares a negative "
                f"total_byte_size {total}"
            )
        for tensor_name, offset, size in tensors:
            if offset < 0 or size < 0 or offset + size > total:
                raise CheckpointMetadataError(
                    f"CheckpointMetadataError: tensor {tensor_name} in {json_path} spans "
                    f"[{offset}, {offset + size}) outside the file's total_byte_size {total}"
                )
        entries.append((json_name, bin_name, total))

    plan = []
    offset = 0
    for json_name, bin_name, total in entries:
        plan.append(PlannedFile(json_name, bin_name, offset, total))
        offset = -(-(offset + total) // WEIGHT_FILE_ALIGNMENT) * WEIGHT_FILE_ALIGNMENT
    planned_total = plan[-1].buffer_offset + plan[-1].byte_size if plan else 0

    for item in plan:
        end = item.buffer_offset + item.byte_size
        if end > buffer_bytes:
            raise CheckpointBufferOverflowError(
                f"CheckpointBufferOverflowError: planned load of {planned_total} bytes over "
                f"{len(plan)} files in {converted_ckpt_dir} exceeds the {buffer_bytes}-byte "
                f"weight buffer; first file past the bound: {item.json_name} "
                f"({item.bin_name}, {item.byte_size} bytes at buffer offset "
                f"{item.buffer_offset}, ending at {end})"
            )
    return plan
