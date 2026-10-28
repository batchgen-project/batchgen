"""CPU-only: the GPT-OSS converted checkpoint directory is checked before any read.

Regression for the 2026-10-28 GPT-OSS-120B launch on 4x H20 that died in
gpt_oss_parameter_server.py with "Failed to read aligned portion from file":
converted_ckpt/ held both the batchgen.tools.convert_checkpoint shards and the
server's own layer_*/non_layer set (65,248,815,744 bytes each), and the C++
loader read both into one 75,161,927,680-byte buffer without a bound.

The plan reads only the .json metadata, so the fake directories below declare
sizes in JSON and carry small or no .bin files.
"""

import builtins
import json
import sys
import types

import pytest

from batchgen.models.openai.gpt_oss_120b.converted_ckpt_plan import (
    WEIGHT_FILE_ALIGNMENT,
    CheckpointBufferOverflowError,
    CheckpointMetadataError,
    MixedCheckpointFormatError,
    check_converted_ckpt_format,
    plan_converted_ckpt_load,
)

NUM_LAYERS = 36
GPT_OSS_BUFFER = 70 * 1024 * 1024 * 1024  # gpt_oss_parameter_server.Init
LAYER_BYTES = 1_748_118_784  # measured layer_<i>.bin size
NON_LAYER_BYTES = 2_316_539_520  # measured non_layer.bin size


def _write_pair(directory, stem, total_byte_size, tensors=None, bin_bytes=b"\0" * 16):
    if tensors is None:
        tensors = {f"{stem}.w": {"dtype": "uint8", "shape": [total_byte_size],
                                 "offset": 0, "byte_size": total_byte_size}}
    meta = {"file_name": f"{stem}.bin", "state_dict": tensors,
            "total_byte_size": total_byte_size}
    (directory / f"{stem}.json").write_text(json.dumps(meta))
    if bin_bytes is not None:
        (directory / f"{stem}.bin").write_bytes(bin_bytes)


def _server_set(directory, layer_bytes=LAYER_BYTES, non_layer_bytes=NON_LAYER_BYTES):
    for i in range(NUM_LAYERS):
        _write_pair(directory, f"layer_{i}", layer_bytes)
    _write_pair(directory, "non_layer", non_layer_bytes)
    (directory / ".gpt_oss_converted").write_text("converted")


def _tool_shards(directory):
    for i in range(1, 8):
        _write_pair(directory, f"model--0000{i}-of-00007", 10_488_712_704)


@pytest.fixture
def no_bin_reads(monkeypatch):
    """Fail the test if anything opens a .bin weight file."""
    real_open = builtins.open

    def guarded(file, *args, **kwargs):
        if str(file).endswith(".bin"):
            raise AssertionError(f"weight file opened during planning: {file}")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded)


def test_mixed_tool_and_server_formats_are_rejected(tmp_path, no_bin_reads):
    _server_set(tmp_path)
    _tool_shards(tmp_path)

    with pytest.raises(MixedCheckpointFormatError) as excinfo:
        plan_converted_ckpt_load(str(tmp_path), GPT_OSS_BUFFER, NUM_LAYERS)

    message = str(excinfo.value)
    assert "model--00001-of-00007.json" in message
    assert "model--00007-of-00007.bin" in message
    assert "convert_checkpoint" in message
    assert "Remove those files" in message
    assert "'layer_0.json'" not in message


def test_tool_shards_alone_are_rejected_before_conversion(tmp_path):
    # The server would convert next to them and produce the mixed directory.
    _tool_shards(tmp_path)
    with pytest.raises(MixedCheckpointFormatError):
        check_converted_ckpt_format(str(tmp_path), NUM_LAYERS)


def test_missing_directory_passes_the_format_check(tmp_path):
    check_converted_ckpt_format(str(tmp_path / "not_yet_converted"), NUM_LAYERS)


def test_layer_index_past_the_model_is_foreign(tmp_path):
    _write_pair(tmp_path, f"layer_{NUM_LAYERS}", 4096)
    with pytest.raises(MixedCheckpointFormatError, match=f"layer_{NUM_LAYERS}.json"):
        check_converted_ckpt_format(str(tmp_path), NUM_LAYERS)


def test_over_budget_plan_is_rejected_before_any_read(tmp_path, no_bin_reads):
    _write_pair(tmp_path, "layer_0", 5000, bin_bytes=None)
    _write_pair(tmp_path, "layer_1", 5000, bin_bytes=None)
    buffer_bytes = 3 * WEIGHT_FILE_ALIGNMENT  # 12288: second file ends at 8192 + 5000

    with pytest.raises(CheckpointBufferOverflowError) as excinfo:
        plan_converted_ckpt_load(str(tmp_path), buffer_bytes, NUM_LAYERS)

    message = str(excinfo.value)
    assert "12288-byte weight buffer" in message
    assert "planned load of 13192 bytes over 2 files" in message
    assert "5000 bytes at buffer offset 8192, ending at 13192" in message
    assert ("layer_0.json" in message) != ("layer_1.json" in message)


def test_over_budget_server_set_at_the_real_buffer_size(tmp_path, no_bin_reads):
    # 36 x 1,748,118,784 + 20 GiB non_layer > 70 GiB.
    _server_set(tmp_path, non_layer_bytes=20 * 1024**3)
    with pytest.raises(CheckpointBufferOverflowError, match="75161927680-byte weight buffer"):
        plan_converted_ckpt_load(str(tmp_path), GPT_OSS_BUFFER, NUM_LAYERS)


def test_valid_server_set_passes_the_plan(tmp_path, no_bin_reads):
    _server_set(tmp_path)
    plan = plan_converted_ckpt_load(str(tmp_path), GPT_OSS_BUFFER, NUM_LAYERS)

    assert len(plan) == NUM_LAYERS + 1
    assert sorted(p.json_name for p in plan) == sorted(
        [f"layer_{i}.json" for i in range(NUM_LAYERS)] + ["non_layer.json"]
    )
    assert plan[0].buffer_offset == 0
    for prev, cur in zip(plan, plan[1:]):
        assert cur.buffer_offset % WEIGHT_FILE_ALIGNMENT == 0
        assert prev.buffer_offset + prev.byte_size <= cur.buffer_offset
        assert cur.buffer_offset - (prev.buffer_offset + prev.byte_size) < WEIGHT_FILE_ALIGNMENT
    assert plan[-1].buffer_offset + plan[-1].byte_size <= GPT_OSS_BUFFER


def test_exact_fit_passes_and_one_byte_less_fails(tmp_path):
    _write_pair(tmp_path, "layer_0", WEIGHT_FILE_ALIGNMENT)
    _write_pair(tmp_path, "non_layer", WEIGHT_FILE_ALIGNMENT)
    plan = plan_converted_ckpt_load(str(tmp_path), 2 * WEIGHT_FILE_ALIGNMENT, NUM_LAYERS)
    assert sorted(p.buffer_offset for p in plan) == [0, WEIGHT_FILE_ALIGNMENT]
    with pytest.raises(CheckpointBufferOverflowError):
        plan_converted_ckpt_load(str(tmp_path), 2 * WEIGHT_FILE_ALIGNMENT - 1, NUM_LAYERS)


def test_tensor_outside_its_file_is_rejected(tmp_path):
    tensors = {"w": {"dtype": "uint8", "shape": [64], "offset": 64, "byte_size": 64}}
    _write_pair(tmp_path, "layer_0", 100, tensors=tensors)
    with pytest.raises(CheckpointMetadataError, match=r"spans \[64, 128\)"):
        plan_converted_ckpt_load(str(tmp_path), GPT_OSS_BUFFER, NUM_LAYERS)


def test_unreadable_metadata_is_named(tmp_path):
    (tmp_path / "layer_0.json").write_text("{not json")
    with pytest.raises(CheckpointMetadataError, match="layer_0.json"):
        plan_converted_ckpt_load(str(tmp_path), GPT_OSS_BUFFER, NUM_LAYERS)


# --- wiring: GptOss_Parameter_Server.Init runs the checks before the C++ loader ---


class _FakeCppParameterServer:
    init_calls = []

    def __init__(self, enable_hugetlbfs, enable_thp=False):
        pass

    def Init(self, *args):
        _FakeCppParameterServer.init_calls.append(args)
        raise AssertionError("C++ Parameter_Server.Init reached")


@pytest.fixture
def gpt_oss_ps(monkeypatch):
    fake_core = types.ModuleType("batchgen.core_engine")
    fake_core.Parameter_Server = _FakeCppParameterServer
    monkeypatch.setitem(sys.modules, "batchgen.core_engine", fake_core)
    module_name = "batchgen.models.openai.gpt_oss_120b.gpt_oss_parameter_server"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    import importlib

    module = importlib.import_module(module_name)
    monkeypatch.setattr(module.torch.cuda, "mem_get_info", lambda: (1, 1))
    _FakeCppParameterServer.init_calls = []
    return module


def _bare_ps(module, converted_ckpt_dir, cache_dir):
    ps = object.__new__(module.GptOss_Parameter_Server)
    ps.converted_ckpt_dir = converted_ckpt_dir
    ps.cache_dir = cache_dir
    ps.num_layers = NUM_LAYERS
    ps.enable_hugetlbfs = False
    ps.enable_thp = False
    ps.state_dict_name_map = {}
    ps._parse_state_dict = lambda: None
    return ps


def test_init_rejects_mixed_directory_before_conversion(tmp_path, gpt_oss_ps):
    converted = tmp_path / "converted_ckpt"
    converted.mkdir()
    _tool_shards(converted)
    before = sorted(p.name for p in converted.iterdir())

    ps = _bare_ps(gpt_oss_ps, converted, tmp_path / "original")
    ps._parse_state_dict = lambda: pytest.fail("format check must run first")
    with pytest.raises(MixedCheckpointFormatError):
        ps.Init()

    assert sorted(p.name for p in converted.iterdir()) == before
    assert _FakeCppParameterServer.init_calls == []


def test_init_rejects_over_budget_directory_before_cpp_init(tmp_path, gpt_oss_ps):
    converted = tmp_path / "converted_ckpt"
    converted.mkdir()
    _server_set(converted, non_layer_bytes=20 * 1024**3)

    ps = _bare_ps(gpt_oss_ps, converted, tmp_path / "original")
    with pytest.raises(CheckpointBufferOverflowError):
        ps.Init()

    assert _FakeCppParameterServer.init_calls == []
