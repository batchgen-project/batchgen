"""CPU-only tests for the GLM-5 converted-checkpoint SHM sizing helper.

Imports only the helper module (stdlib-backed) — no torch, no core_engine.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).parents[1]
    / "batchgen/models/glm/glm5/shm_reservation.py"
)
_SPEC = importlib.util.spec_from_file_location("glm5_shm_reservation", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
converted_checkpoint_byte_size = _MODULE.converted_checkpoint_byte_size


def _write_shard(directory, stem, tensor_byte_sizes, metadata_overrides=None,
                 bin_byte_size=None, write_bin=True):
    """Write one shard (.json + .bin) and return the metadata path."""
    offset = 0
    state_dict = {}
    for idx, byte_size in enumerate(tensor_byte_sizes):
        state_dict[f"tensor_{idx}"] = {
            "dtype": "bfloat16",
            "shape": [byte_size],
            "offset": offset,
            "byte_size": byte_size,
        }
        offset += byte_size

    metadata = {
        "file_name": f"{stem}.bin",
        "state_dict": state_dict,
        "total_byte_size": sum(tensor_byte_sizes),
    }
    if metadata_overrides:
        metadata.update(metadata_overrides)

    metadata_path = os.path.join(directory, f"{stem}.json")
    with open(metadata_path, "w") as handle:
        json.dump(metadata, handle)

    if write_bin:
        written = sum(tensor_byte_sizes) if bin_byte_size is None else bin_byte_size
        with open(os.path.join(directory, f"{stem}.bin"), "wb") as handle:
            handle.write(b"\0" * written)

    return metadata_path


def test_sums_shards_with_4kib_and_2mib_alignment(tmp_path):
    directory = str(tmp_path)
    # 4097 -> 8192, 4096 -> 4096; total 12288 -> one 2 MiB page.
    _write_shard(directory, "shard_0", [4096, 1])
    _write_shard(directory, "shard_1", [4096])

    assert converted_checkpoint_byte_size(directory) == 2 * 1024 * 1024


def test_large_total_rounds_up_to_next_2mib(tmp_path):
    directory = str(tmp_path)
    # 3 MiB + 4 KiB of payload -> 4 MiB reservation.
    _write_shard(directory, "shard_0", [3 * 1024 * 1024])
    _write_shard(directory, "shard_1", [4096])

    assert converted_checkpoint_byte_size(directory) == 4 * 1024 * 1024


def test_ignores_non_json_entries(tmp_path):
    directory = str(tmp_path)
    _write_shard(directory, "shard_0", [4096])
    with open(os.path.join(directory, "config.txt"), "w") as handle:
        handle.write("not metadata")

    assert converted_checkpoint_byte_size(directory) == 2 * 1024 * 1024


def test_missing_directory(tmp_path):
    missing = str(tmp_path / "absent")
    with pytest.raises(ValueError, match="absent"):
        converted_checkpoint_byte_size(missing)


def test_no_metadata_shards(tmp_path):
    with pytest.raises(ValueError, match="No checkpoint metadata"):
        converted_checkpoint_byte_size(str(tmp_path))


def test_malformed_json(tmp_path):
    directory = str(tmp_path)
    bad_path = os.path.join(directory, "shard_0.json")
    with open(bad_path, "w") as handle:
        handle.write("{not json")

    with pytest.raises(ValueError, match="shard_0.json"):
        converted_checkpoint_byte_size(directory)


def test_metadata_root_must_be_object(tmp_path):
    directory = str(tmp_path)
    metadata_path = os.path.join(directory, "shard_0.json")
    with open(metadata_path, "w") as handle:
        json.dump([], handle)

    with pytest.raises(ValueError, match="metadata must be a JSON object"):
        converted_checkpoint_byte_size(directory)


@pytest.mark.parametrize("missing_field", ["file_name", "total_byte_size", "state_dict"])
def test_missing_required_field(tmp_path, missing_field):
    directory = str(tmp_path)
    metadata_path = _write_shard(directory, "shard_0", [4096])
    with open(metadata_path) as handle:
        metadata = json.load(handle)
    del metadata[missing_field]
    with open(metadata_path, "w") as handle:
        json.dump(metadata, handle)

    with pytest.raises(ValueError, match=f"missing '{missing_field}'"):
        converted_checkpoint_byte_size(directory)


@pytest.mark.parametrize("bad_total", [True, 0, -4096, "4096", 4096.0])
def test_bad_total_byte_size(tmp_path, bad_total):
    directory = str(tmp_path)
    _write_shard(directory, "shard_0", [4096],
                 metadata_overrides={"total_byte_size": bad_total})

    with pytest.raises(ValueError, match="total_byte_size"):
        converted_checkpoint_byte_size(directory)


def test_bool_tensor_byte_size_rejected(tmp_path):
    directory = str(tmp_path)
    metadata_path = _write_shard(directory, "shard_0", [4096])
    with open(metadata_path) as handle:
        metadata = json.load(handle)
    metadata["state_dict"]["tensor_0"]["byte_size"] = True
    with open(metadata_path, "w") as handle:
        json.dump(metadata, handle)

    with pytest.raises(ValueError, match="byte_size"):
        converted_checkpoint_byte_size(directory)


def test_tensor_metadata_must_be_object(tmp_path):
    directory = str(tmp_path)
    metadata_path = _write_shard(directory, "shard_0", [4096])
    with open(metadata_path) as handle:
        metadata = json.load(handle)
    metadata["state_dict"]["tensor_0"] = 4096
    with open(metadata_path, "w") as handle:
        json.dump(metadata, handle)

    with pytest.raises(ValueError, match="tensor 'tensor_0' has no 'byte_size'"):
        converted_checkpoint_byte_size(directory)


def test_negative_tensor_byte_size_rejected(tmp_path):
    directory = str(tmp_path)
    metadata_path = _write_shard(directory, "shard_0", [4096])
    with open(metadata_path) as handle:
        metadata = json.load(handle)
    metadata["state_dict"]["tensor_0"]["byte_size"] = -1
    with open(metadata_path, "w") as handle:
        json.dump(metadata, handle)

    with pytest.raises(ValueError, match="negative 'byte_size' -1"):
        converted_checkpoint_byte_size(directory)


def test_state_dict_sum_mismatch(tmp_path):
    directory = str(tmp_path)
    _write_shard(directory, "shard_0", [4096],
                 metadata_overrides={"total_byte_size": 8192},
                 bin_byte_size=8192)

    with pytest.raises(ValueError, match="sum to 4096"):
        converted_checkpoint_byte_size(directory)


def test_missing_bin_file(tmp_path):
    directory = str(tmp_path)
    _write_shard(directory, "shard_0", [4096], write_bin=False)

    with pytest.raises(ValueError, match="shard_0.bin"):
        converted_checkpoint_byte_size(directory)


def test_wrong_size_bin_file(tmp_path):
    directory = str(tmp_path)
    _write_shard(directory, "shard_0", [4096], bin_byte_size=1024)

    with pytest.raises(ValueError, match="1024 bytes, expected 4096"):
        converted_checkpoint_byte_size(directory)


def test_stray_json_named_junk_is_rejected(tmp_path):
    directory = str(tmp_path)
    _write_shard(directory, "shard_0", [4096])
    with open(os.path.join(directory, "notes.json.bak"), "w") as handle:
        handle.write("scratch notes")

    with pytest.raises(ValueError, match="notes.json.bak"):
        converted_checkpoint_byte_size(directory)


def test_directory_named_like_metadata_is_rejected(tmp_path):
    directory = str(tmp_path)
    _write_shard(directory, "shard_0", [4096])
    os.mkdir(os.path.join(directory, "stale.json"))

    with pytest.raises(ValueError, match="stale.json"):
        converted_checkpoint_byte_size(directory)
