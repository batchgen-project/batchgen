"""ckpt_converter: parallel shard conversion, partitions, buffered writes.

CPU-only: conversion without Marlin repack plus the pure scheduling logic
(partition slicing, per-device locks, round-robin device choice). The GPU
repack itself is covered by tests/moe/test_mxfp4_marlin_repack.py and the
frozen oracle it gates.
"""

import json
import os
import sys
import threading

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from batchgen.ckpt_converter.ckpt_converter import (  # noqa: E402
    WRITE_BUFFER_BYTES,
    ckpt_converter,
)


def _write_shards(src, count=5, tensors_per_shard=7, seed=3):
    torch.manual_seed(seed)
    src.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        shard = {
            f"layer.{index}.t{t}": torch.randn(13, 17 + t)
            for t in range(tensors_per_shard)
        }
        torch.save(shard, src / f"model-{index:05d}-of-{count:05d}.pt")
    return sorted(path.name for path in src.iterdir())


def _dir_bytes(directory):
    out = {}
    for path in sorted(directory.iterdir()):
        out[path.name] = path.read_bytes()
    return out


def test_parallel_and_partitioned_outputs_equal_serial(tmp_path):
    src = tmp_path / "src"
    _write_shards(src)
    serial = tmp_path / "serial"
    ckpt_converter().convert_model_directory(
        str(src), str(serial), max_workers=1
    )

    parallel = tmp_path / "parallel"
    ckpt_converter().convert_model_directory(
        str(src), str(parallel), max_workers=4
    )
    assert _dir_bytes(parallel) == _dir_bytes(serial)

    parts = tmp_path / "parts"
    for index in (2, 0, 1):  # out of order, as concurrent nodes would be
        ckpt_converter().convert_model_directory(
            str(src), str(parts), partition_index=index, partition_count=3
        )
    assert _dir_bytes(parts) == _dir_bytes(serial)
    converter = ckpt_converter()
    ok, error = converter.validate_converted_directory(str(src), str(parts))
    assert ok, error


def test_partitions_split_the_sorted_list_without_overlap(tmp_path):
    src = tmp_path / "src"
    names = _write_shards(src, count=7)
    converter = ckpt_converter()
    full = [os.path.basename(p) for p in converter._get_checkpoint_files(str(src))]
    assert full == names
    pieces = [full[index::3] for index in range(3)]
    assert sorted(sum(pieces, [])) == sorted(full)
    assert all(
        not set(a) & set(b)
        for i, a in enumerate(pieces)
        for b in pieces[i + 1:]
    )


def test_partition_arguments_are_validated(tmp_path):
    src = tmp_path / "src"
    _write_shards(src, count=2)
    converter = ckpt_converter()
    for index, count in ((1, 1), (-1, 2), (2, 2), (0, 0)):
        with pytest.raises(ValueError, match="partition_index/partition_count"):
            converter.convert_model_directory(
                str(src), str(tmp_path / "out"),
                partition_index=index, partition_count=count,
            )


def test_partition_mode_skips_the_existing_output_shortcut(tmp_path):
    src = tmp_path / "src"
    _write_shards(src, count=4)
    out = tmp_path / "out"
    converter = ckpt_converter()
    # Partition 0 runs first and leaves a PARTIAL directory; partition 1
    # must convert its own files rather than refuse on the partial state
    # (force=False) or mistake it for a finished conversion.
    converter.convert_model_directory(
        str(src), str(out), partition_index=0, partition_count=2
    )
    assert len(list(out.glob("*.bin"))) == 2
    converter.convert_model_directory(
        str(src), str(out), partition_index=1, partition_count=2
    )
    assert len(list(out.glob("*.bin"))) == 4


def test_metadata_offsets_match_the_bin_layout(tmp_path):
    src = tmp_path / "src"
    _write_shards(src, count=1, tensors_per_shard=3)
    out = tmp_path / "out"
    ckpt_converter().convert_model_directory(str(src), str(out))
    meta = json.loads(next(out.glob("*.json")).read_text())
    blob = next(out.glob("*.bin")).read_bytes()
    assert meta["total_byte_size"] == len(blob)
    source = torch.load(next((tmp_path / "src").iterdir()), weights_only=True)
    for name, entry in meta["state_dict"].items():
        tensor = source[name].contiguous()
        start = entry["offset"]
        raw = blob[start:start + entry["byte_size"]]
        assert raw == tensor.numpy().tobytes()


def test_repack_devices_round_robin_and_lock_per_device(monkeypatch):
    converter = ckpt_converter()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    picks = [converter._next_repack_device() for _ in range(7)]
    assert picks == [
        "cuda:0", "cuda:1", "cuda:2", "cuda:0", "cuda:1", "cuda:2", "cuda:0",
    ]
    lock_a = converter._repack_lock("cuda:0")
    lock_b = converter._repack_lock("cuda:1")
    assert lock_a is converter._repack_lock("cuda:0")
    assert lock_a is not lock_b
    assert isinstance(lock_a, type(threading.Lock()))


def test_no_gpu_means_no_device_and_cpu_skip(monkeypatch):
    converter = ckpt_converter()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert converter._next_repack_device() is None
    ckpt = {"x": torch.ones(2)}
    assert converter._apply_marlin_repack(dict(ckpt), None)["x"].eq(1).all()


def test_write_buffer_constant_is_large():
    # The point of the buffer is to turn thousands of per-tensor writes
    # into few large sequential ones on a network filesystem.
    assert WRITE_BUFFER_BYTES >= 8 * 1024 * 1024
