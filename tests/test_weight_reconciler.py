"""Unit tests for the model-agnostic weight reconciler. CPU only, no weights.

Every check has a fixture that must PASS and a deliberately-broken input that
must FAIL — a checker that cannot fail is decoration.
"""

import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "batchgen" / "models" / "weight_reconciler.py"
    spec = importlib.util.spec_from_file_location(
        "_batchgen_weight_reconciler", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


wr = _load()

BF16 = torch.bfloat16
U8 = torch.uint8

MODULE_SHAPES = {
    "attn": {"q.weight": [8, 4], "o.weight": [4, 8]},
    "expert": {"w1.weight_packed": [8, 2], "w1.weight_scale": [8, 1]},
}
WEIGHT_DTYPES = {"attn": BF16, "expert": U8}


def _name_map():
    mapping = {}
    for layer in (0, 1):
        for key in ("q.weight", "o.weight"):
            mapping["m.layers.{}.attn.{}".format(layer, key)] = {
                "module_key": "attn_{}".format(layer), "tensor_key": key}
        for key in ("w1.weight_packed", "w1.weight_scale"):
            mapping["m.layers.{}.experts.0.{}".format(layer, key)] = {
                "module_key": "expert_{}_0".format(layer), "tensor_key": key}
    return mapping


def _skeleton():
    return {"m.norm.weight": ([4], BF16)}


def _spec(**overrides):
    kwargs = dict(
        name_map=_name_map(),
        module_shapes={k: dict(v) for k, v in MODULE_SHAPES.items()},
        weight_dtypes=dict(WEIGHT_DTYPES),
        tensor_dtypes={},
        skeleton=_skeleton(),
        ignore_rules=(wr.IgnoreRule("vision.", "out of scope for this milestone"),),
    )
    kwargs.update(overrides)
    return wr.ReconcileSpec(**kwargs)


def _declared_bytes(spec):
    total = 0
    for entry in spec.name_map.values():
        module_type = entry["module_key"].rsplit("_", 1)[0]
        if module_type not in spec.module_shapes:
            module_type = module_type.rsplit("_", 1)[0]
        shape = spec.module_shapes[module_type][entry["tensor_key"]]
        dtype = spec.tensor_dtypes.get(module_type, {}).get(
            entry["tensor_key"], spec.weight_dtypes[module_type])
        total += wr.nbytes(shape, dtype)
    for shape, dtype in spec.skeleton.values():
        total += wr.nbytes(shape, dtype)
    return total


def _index(spec, extra_names=(), ignored_bytes=16):
    names = set(spec.name_map) | set(spec.skeleton) | set(extra_names)
    names.add("vision.tower.weight")
    return wr.CheckpointIndex(
        tensor_names=names,
        total_bytes=_declared_bytes(spec) + ignored_bytes,
        num_shards=1,
    )


def _fixture():
    spec = _spec(ignored_count=1, ignored_bytes=16)
    return _index(spec), spec


# --------------------------------------------------------------------------- #
#  The fixture itself must pass                                                #
# --------------------------------------------------------------------------- #

def test_clean_fixture_passes():
    checkpoint, spec = _fixture()
    report = wr.reconcile(checkpoint, spec)
    assert report.ok, report.render()
    assert report.n_mapped == 8
    assert report.n_skeleton == 1
    assert report.n_ignored == 1
    assert report.declared_bytes == report.checkpoint_bytes
    report.raise_for_status()


# --------------------------------------------------------------------------- #
#  unaccounted: checkpoint tensor with no destination                          #
# --------------------------------------------------------------------------- #

def test_unmapped_checkpoint_tensor_is_reported():
    checkpoint, spec = _fixture()
    checkpoint.tensor_names.add("m.layers.0.attn.g.weight")
    report = wr.reconcile(checkpoint, spec)
    assert report.counts["unaccounted"] == 1
    assert not report.ok
    with pytest.raises(wr.WeightReconcileError):
        report.raise_for_status()


def test_ignore_rule_accounts_for_a_tensor_that_is_not_loaded():
    checkpoint, spec = _fixture()
    checkpoint.tensor_names.add("vision.tower.bias")
    report = wr.reconcile(checkpoint, wr.ReconcileSpec(
        name_map=spec.name_map, module_shapes=spec.module_shapes,
        weight_dtypes=spec.weight_dtypes, skeleton=spec.skeleton,
        ignore_rules=spec.ignore_rules, ignored_count=2, ignored_bytes=16))
    assert "unaccounted" not in report.counts
    assert report.n_ignored == 2


def test_ignore_rules_must_carry_a_reason():
    with pytest.raises(TypeError):
        wr.IgnoreRule("vision.")            # reason is required


# --------------------------------------------------------------------------- #
#  dangling: a declaration naming a tensor the checkpoint lacks                 #
# --------------------------------------------------------------------------- #

def test_name_map_entry_for_a_missing_tensor_is_reported():
    checkpoint, spec = _fixture()
    spec.name_map["m.layers.0.attn.typo.weight"] = {
        "module_key": "attn_0", "tensor_key": "q.weight"}
    report = wr.reconcile(checkpoint, spec)
    assert report.counts["dangling"] == 1


def test_skeleton_entry_for_a_missing_tensor_is_reported():
    checkpoint, spec = _fixture()
    spec.skeleton["m.absent.weight"] = ([4], BF16)
    report = wr.reconcile(checkpoint, spec)
    assert report.counts["dangling"] == 1


# --------------------------------------------------------------------------- #
#  slot_never_written: the GLM-5 class                                         #
# --------------------------------------------------------------------------- #

def test_slot_allocated_but_never_written_is_reported():
    """module_shapes declares a tensor no checkpoint tensor is mapped to.

    Silent in production: GPU_Weight_Buffer allocates it with torch::zeros and
    nothing ever overwrites it (or, on slot reuse, the previous module's bytes
    survive).  This is the shape of the GLM-5 ones_()-init RMSNorm defect.
    """
    checkpoint, spec = _fixture()
    spec.module_shapes["attn"]["k_norm.weight"] = [4]
    report = wr.reconcile(checkpoint, spec)
    assert report.counts["slot_never_written"] == 1
    finding = next(f for f in report.findings if f.bucket == "slot_never_written")
    assert finding.key == "attn.k_norm.weight"
    assert "2 module(s)" in finding.detail
    # and it is invisible to a checkpoint-only census
    assert "unaccounted" not in report.counts


def test_source_without_a_slot_is_reported():
    """A mapped tensor module_shapes has no slot for is silently dropped."""
    checkpoint, spec = _fixture()
    spec.name_map["m.layers.0.attn.g.weight"] = {
        "module_key": "attn_0", "tensor_key": "g.weight"}
    checkpoint.tensor_names.add("m.layers.0.attn.g.weight")
    report = wr.reconcile(checkpoint, spec)
    assert report.counts["source_without_slot"] == 1
    # the byte total cannot be computed, and that is reported, not passed over
    assert report.skipped and "byte-total" in report.skipped[0]
    assert not report.ok


def test_module_type_nothing_maps_to_is_reported():
    checkpoint, spec = _fixture()
    spec.module_shapes["shared_expert"] = {"w.weight": [4, 4]}
    spec.weight_dtypes["shared_expert"] = BF16
    report = wr.reconcile(checkpoint, spec)
    assert report.counts["module_type_unused"] == 1


def test_unknown_module_key_is_reported():
    checkpoint, spec = _fixture()
    spec.name_map["m.layers.0.attn.q.weight"] = {
        "module_key": "mystery_0", "tensor_key": "q.weight"}
    report = wr.reconcile(checkpoint, spec)
    assert report.counts["unknown_module_type"] == 1


# --------------------------------------------------------------------------- #
#  double declaration                                                          #
# --------------------------------------------------------------------------- #

def test_tensor_claimed_by_both_name_map_and_skeleton_is_reported():
    checkpoint, spec = _fixture()
    spec.skeleton["m.layers.0.attn.q.weight"] = ([8, 4], BF16)
    report = wr.reconcile(checkpoint, spec)
    assert report.counts["double_declared"] == 1


# --------------------------------------------------------------------------- #
#  quantized packed/scale couples                                              #
# --------------------------------------------------------------------------- #

def test_packed_without_a_scale_in_the_checkpoint_is_reported():
    checkpoint, spec = _fixture()
    checkpoint.tensor_names.remove("m.layers.0.experts.0.w1.weight_scale")
    del spec.name_map["m.layers.0.experts.0.w1.weight_scale"]
    report = wr.reconcile(checkpoint, spec)
    assert report.counts["quant_pair"] == 1
    assert report.counts["slot_never_written"] == 1


def test_half_mapped_couple_is_reported():
    """The packed tensor is streamed and its scale is left behind."""
    checkpoint, spec = _fixture()
    del spec.name_map["m.layers.0.experts.0.w1.weight_scale"]
    report = wr.reconcile(checkpoint, spec)
    assert report.counts["quant_pair"] == 1
    assert report.counts["unaccounted"] == 1
    assert report.counts["slot_never_written"] == 1


# --------------------------------------------------------------------------- #
#  bytes                                                                       #
# --------------------------------------------------------------------------- #

def test_wrong_declared_shape_breaks_the_byte_total():
    checkpoint, spec = _fixture()
    spec.module_shapes["attn"]["q.weight"] = [8, 8]
    report = wr.reconcile(checkpoint, spec)
    assert report.counts["byte_total"] == 1


def test_wrong_declared_dtype_breaks_the_byte_total():
    checkpoint, spec = _fixture()
    spec.tensor_dtypes = {"expert": {"w1.weight_scale": BF16}}
    report = wr.reconcile(checkpoint, spec)
    assert report.counts["byte_total"] == 1


def test_unknown_ignored_size_is_a_skip_not_a_pass():
    spec = _spec(ignored_count=1)          # ignored_bytes deliberately absent
    report = wr.reconcile(_index(spec), spec)
    assert report.skipped
    assert not report.ok
    with pytest.raises(wr.WeightReconcileError):
        report.raise_for_status()
    report.raise_for_status(allow_skipped=True)


def test_changed_ignored_set_size_is_reported():
    checkpoint, spec = _fixture()
    checkpoint.tensor_names.add("vision.tower.bias")
    report = wr.reconcile(checkpoint, spec)
    assert report.counts["ignored_count"] == 1


# --------------------------------------------------------------------------- #
#  bounded output                                                              #
# --------------------------------------------------------------------------- #

def test_findings_are_bounded_but_counts_are_exact():
    checkpoint, spec = _fixture()
    for i in range(500):
        checkpoint.tensor_names.add("m.extra.{}".format(i))
    report = wr.reconcile(checkpoint, spec, max_findings=10)
    assert report.counts["unaccounted"] == 500
    assert len([f for f in report.findings if f.bucket == "unaccounted"]) == 10
    assert "suppressed" in report.render()


# --------------------------------------------------------------------------- #
#  index / shard-header readers                                                #
# --------------------------------------------------------------------------- #

def test_read_index(tmp_path):
    index = {
        "metadata": {"total_size": 4096},
        "weight_map": {"a": "s0.safetensors", "b": "s0.safetensors",
                       "c": "s1.safetensors"},
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    got = wr.read_index(str(tmp_path / "model.safetensors.index.json"))
    assert got.tensor_names == {"a", "b", "c"}
    assert got.total_bytes == 4096
    assert got.num_shards == 2


def _write_safetensors(path, tensors):
    header = {}
    offset = 0
    blobs = []
    for name, tensor in tensors.items():
        raw = tensor.contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()
        header[name] = {
            "dtype": {torch.uint8: "U8", torch.float32: "F32",
                      torch.bfloat16: "BF16"}[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
        blobs.append(raw)
    blob = json.dumps(header).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        for raw in blobs:
            handle.write(raw)


def test_shard_headers_catch_a_shape_error_the_byte_total_would_miss(tmp_path):
    """Two declaration errors that cancel still fail the per-tensor check."""
    _write_safetensors(tmp_path / "model-00001.safetensors", {
        "m.layers.0.attn.q.weight": torch.zeros(8, 4, dtype=BF16),
        "m.layers.0.attn.o.weight": torch.zeros(4, 8, dtype=BF16),
    })
    checkpoint = wr.read_safetensors_headers(str(tmp_path))
    assert checkpoint.tensors["m.layers.0.attn.q.weight"].dtype is BF16
    assert checkpoint.total_bytes == 2 * 32 * 2

    spec = wr.ReconcileSpec(
        name_map={
            "m.layers.0.attn.q.weight": {"module_key": "attn_0",
                                         "tensor_key": "q.weight"},
            "m.layers.0.attn.o.weight": {"module_key": "attn_0",
                                         "tensor_key": "o.weight"},
        },
        # transposed shapes: same bytes, wrong geometry
        module_shapes={"attn": {"q.weight": [4, 8], "o.weight": [8, 4]}},
        weight_dtypes={"attn": BF16},
    )
    report = wr.reconcile(checkpoint, spec)
    assert "byte_total" not in report.counts       # the totals agree
    assert report.counts["tensor_mismatch"] == 2   # the tensors do not
