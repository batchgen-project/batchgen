"""The primary MLA record's dim/dtype contract at the staging and host seams.

The packed FP8 primary record (656 BYTES per token) is produced exactly once,
at the writer, and every hop after it is a BYTE COPY. Two hops used to be able
to break that silently and are pinned here:

* ``Glm5WholeModelSegment`` stages each layer's record with ``copy_()``, which
  CONVERTS dtypes -- a bf16 record into a uint8 staging buffer would be
  requantized per element. Its buffer's dim AND dtype must therefore come from
  the allocated cache, not from the model's latent width.
* ``batchgen/cuda_graph/composition.allocate_kv_staging`` allocates the generic
  equivalent, where a broadcast dtype would reinterpret a byte-counted dim as
  an element-counted one.

Everything exercised here is dtype/geometry bookkeeping, so it runs on the dev
machine without Triton or a GPU: the resolver and the guards are plain
staticmethods and module functions. The one path that cannot run on CPU (the
Triton packed quantize) is asserted to be *reached*, which is what proves the
packed branch is not silently falling back to bf16.
"""

import pytest
import torch

from batchgen.cuda_graph.composition import (
    allocate_kv_staging,
    resolve_kv_staging_dtypes,
)
from batchgen.models.glm.glm5.whole_model_cuda_graph_segments import (
    Glm5WholeModelSegment,
)
from batchgen.models.glm.glm5.wrappers import (
    _prepare_primary_host_record,
    _primary_host_record_geometry,
    _primary_host_record_is_packed,
)
from batchgen_kernels.attention.mla_kv_packed import SOURCE_DIM, TOKEN_BYTES


CPU = torch.device("cpu")


# --- composition: a byte-carrier dim and an element dim are not one dtype ---


def test_resolve_kv_staging_dtypes_keeps_the_uniform_broadcast():
    dims = {"primary": SOURCE_DIM, "aux": 128}
    assert resolve_kv_staging_dtypes(dims, torch.bfloat16) == {
        "primary": torch.bfloat16,
        "aux": torch.bfloat16,
    }


def test_resolve_kv_staging_dtypes_refuses_a_broadcast_byte_carrier():
    # `primary` would be 656 BYTES and `aux` 128 ELEMENTS; one dtype cannot
    # describe both, and broadcasting uint8 would reinterpret the aux dim.
    dims = {"primary": TOKEN_BYTES, "aux": 128}
    with pytest.raises(ValueError, match="opaque byte carrier"):
        resolve_kv_staging_dtypes(dims, torch.uint8)


def test_resolve_kv_staging_dtypes_allows_a_single_byte_carrier_stream():
    # Unambiguous: there is no second stream whose dim could be misread.
    assert resolve_kv_staging_dtypes({"primary": TOKEN_BYTES}, torch.uint8) == {
        "primary": torch.uint8
    }


def test_allocate_kv_staging_allocates_a_packed_primary_stream():
    out = allocate_kv_staging(
        num_layers=3,
        max_bucket=4,
        kv_staging_dim={"primary": TOKEN_BYTES, "aux": 128},
        dtype={"primary": torch.uint8, "aux": torch.float8_e4m3fn},
        device=CPU,
    )
    assert out["primary"].shape == (3, 4, 1, 1, TOKEN_BYTES)
    assert out["primary"].dtype is torch.uint8
    assert out["aux"].dtype is torch.float8_e4m3fn


# --- the whole-model primary staging record ---------------------------------


class _FakeDsaSegment:
    def __init__(self, record_dim, record_dtype):
        self.primary_record_dim = record_dim
        self.primary_record_dtype = record_dtype


class _FakeLayerSegment:
    """A layer segment as `layer_cuda_graph_segments.py:51-63` builds one.

    It publishes the record of the ``primary_k_tensor`` it will hand over
    (``primary_kv_dim`` / ``primary_kv_dtype``) AND carries the DSA segment
    that published the same record off the allocated cache. The resolver reads
    both and must find them equal.
    """

    def __init__(self, record_dim, record_dtype, dsa_record=None):
        self.dsa_segment = _FakeDsaSegment(*(dsa_record or (record_dim, record_dtype)))
        self.primary_kv_dim = record_dim
        self.primary_kv_dtype = record_dtype


def _resolve(**kwargs):
    base = dict(
        model_source_dim=SOURCE_DIM,
        layer_segments=[],
        explicit_dim=None,
        explicit_dtype=None,
    )
    base.update(kwargs)
    return Glm5WholeModelSegment._resolve_primary_record(**base)


def test_primary_record_defaults_to_the_models_bf16_latent():
    # Pre-migration behaviour, and correct for every bf16 AoS cache.
    assert _resolve() == (SOURCE_DIM, torch.bfloat16)


def test_primary_record_comes_from_the_allocated_cache():
    segments = [_FakeLayerSegment(TOKEN_BYTES, torch.uint8) for _ in range(4)]
    assert _resolve(layer_segments=segments) == (TOKEN_BYTES, torch.uint8)


def test_primary_record_from_a_bf16_cache_matches_the_model_latent():
    segments = [_FakeLayerSegment(SOURCE_DIM, torch.bfloat16)]
    assert _resolve(layer_segments=segments) == (SOURCE_DIM, torch.bfloat16)


def test_primary_record_rejects_a_layer_that_contradicts_its_dsa_segment():
    # Two derivations of one cache: the layer's declared output-spec record and
    # the DSA segment's record read off `primary_blocked_k`. A drift between
    # them is named instead of one being preferred.
    segments = [
        _FakeLayerSegment(
            TOKEN_BYTES, torch.uint8, dsa_record=(SOURCE_DIM, torch.bfloat16)
        )
    ]
    with pytest.raises(ValueError, match="its DSA segment declares"):
        _resolve(layer_segments=segments)


def test_primary_record_rejects_layer_segments_that_disagree():
    segments = [
        _FakeLayerSegment(TOKEN_BYTES, torch.uint8),
        _FakeLayerSegment(SOURCE_DIM, torch.bfloat16),
    ]
    with pytest.raises(ValueError, match="disagrees"):
        _resolve(layer_segments=segments)


def test_primary_record_accepts_an_explicit_record_that_agrees():
    segments = [_FakeLayerSegment(TOKEN_BYTES, torch.uint8)]
    assert _resolve(
        layer_segments=segments,
        explicit_dim=TOKEN_BYTES,
        explicit_dtype=torch.uint8,
    ) == (TOKEN_BYTES, torch.uint8)


def test_primary_record_names_a_cache_vs_caller_mismatch():
    segments = [_FakeLayerSegment(TOKEN_BYTES, torch.uint8)]
    with pytest.raises(ValueError, match="primary KV record mismatch"):
        _resolve(
            layer_segments=segments,
            explicit_dim=SOURCE_DIM,
            explicit_dtype=torch.bfloat16,
        )


def test_primary_record_refuses_a_dim_without_its_dtype():
    with pytest.raises(ValueError, match="dim AND a dtype"):
        _resolve(explicit_dim=TOKEN_BYTES)
    with pytest.raises(ValueError, match="dim AND a dtype"):
        _resolve(explicit_dtype=torch.uint8)


def test_primary_record_refuses_a_byte_carrier_of_the_wrong_width():
    with pytest.raises(ValueError, match="packed FP8"):
        _resolve(explicit_dim=SOURCE_DIM, explicit_dtype=torch.uint8)


def test_primary_record_refuses_a_bf16_record_of_the_wrong_width():
    with pytest.raises(ValueError, match="bf16 AoS primary KV record"):
        _resolve(explicit_dim=TOKEN_BYTES, explicit_dtype=torch.bfloat16)


def test_primary_record_refuses_a_third_carrier():
    with pytest.raises(ValueError, match="bf16 AoS or a uint8 packed"):
        _resolve(explicit_dim=SOURCE_DIM, explicit_dtype=torch.float8_e4m3fn)


# --- the staging dtype guard ------------------------------------------------


def test_normalize_k_tensor_accepts_the_declared_record():
    packed = torch.zeros((4, 1, TOKEN_BYTES), dtype=torch.uint8)
    out = Glm5WholeModelSegment._normalize_k_tensor(
        packed, TOKEN_BYTES, torch.uint8
    )
    assert out.shape == (4, 1, 1, TOKEN_BYTES)
    assert out.dtype is torch.uint8


def test_normalize_k_tensor_rejects_a_bf16_record_for_a_packed_buffer():
    # THE hazard: `copy_()` would requantize these 576 bf16 values into 656
    # uint8 lanes element by element.
    bf16_record = torch.zeros((4, 1, 1, TOKEN_BYTES), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="staging dtype mismatch"):
        Glm5WholeModelSegment._normalize_k_tensor(
            bf16_record, TOKEN_BYTES, torch.uint8
        )


def test_normalize_k_tensor_rejects_a_packed_record_for_a_bf16_buffer():
    packed = torch.zeros((4, 1, 1, SOURCE_DIM), dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="staging dtype mismatch"):
        Glm5WholeModelSegment._normalize_k_tensor(
            packed, SOURCE_DIM, torch.bfloat16
        )


def test_normalize_k_tensor_still_rejects_a_dim_mismatch():
    bf16_record = torch.zeros((4, 1, 1, 512), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="KV dim mismatch"):
        Glm5WholeModelSegment._normalize_k_tensor(
            bf16_record, SOURCE_DIM, torch.bfloat16
        )


# --- the prefill host-offload record ---------------------------------------


class _FakeWorkerView:
    """A host worker view that publishes its region's record geometry."""

    def __init__(self, k_head_dim=None, k_element_size_bytes=None):
        if k_head_dim is not None:
            self.k_head_dim = k_head_dim
        if k_element_size_bytes is not None:
            self.k_element_size_bytes = k_element_size_bytes


def test_host_record_geometry_is_none_when_the_region_publishes_nothing():
    assert _primary_host_record_geometry(_FakeWorkerView()) is None
    # A half-published geometry is as unusable as none, and must not be
    # completed with a default.
    assert _primary_host_record_geometry(_FakeWorkerView(k_head_dim=656)) is None


def test_host_record_geometry_reads_both_fields():
    view = _FakeWorkerView(k_head_dim=TOKEN_BYTES, k_element_size_bytes=1)
    assert _primary_host_record_geometry(view) == (TOKEN_BYTES, 1)


def test_host_record_discriminates_the_two_supported_layouts():
    assert _primary_host_record_is_packed(TOKEN_BYTES, 1, SOURCE_DIM) is True
    assert _primary_host_record_is_packed(SOURCE_DIM, 2, SOURCE_DIM) is False


@pytest.mark.parametrize(
    "record_dim,element_bytes",
    [
        (SOURCE_DIM, 1),  # byte carrier of the wrong width
        (640, 1),  # byte carrier of the wrong width
        (SOURCE_DIM, 4),  # fp32 carrier: neither layout
        (512, 2),  # bf16 of the wrong width
    ],
)
def test_host_record_names_a_missized_region(record_dim, element_bytes):
    with pytest.raises(RuntimeError, match="primary host KV region"):
        _primary_host_record_is_packed(record_dim, element_bytes, SOURCE_DIM)


def test_prepare_host_record_is_the_identity_for_a_bf16_region():
    kv = torch.zeros((8, SOURCE_DIM), dtype=torch.bfloat16)
    view = _FakeWorkerView(k_head_dim=SOURCE_DIM, k_element_size_bytes=2)
    out = _prepare_primary_host_record(
        kv, primary_view=view, source_dim=SOURCE_DIM
    )
    # The bf16 record IS the latent: the same storage, so the host enqueue
    # stays the byte copy it has always been.
    assert out is kv


def test_prepare_host_record_keeps_bf16_when_the_region_publishes_nothing():
    kv = torch.zeros((8, SOURCE_DIM), dtype=torch.bfloat16)
    out = _prepare_primary_host_record(
        kv, primary_view=_FakeWorkerView(), source_dim=SOURCE_DIM
    )
    assert out is kv


def test_prepare_host_record_rejects_a_non_bf16_source():
    kv = torch.zeros((8, SOURCE_DIM), dtype=torch.float32)
    with pytest.raises(RuntimeError, match="bf16 compressed"):
        _prepare_primary_host_record(
            kv, primary_view=_FakeWorkerView(), source_dim=SOURCE_DIM
        )


def test_prepare_host_record_rejects_a_source_of_the_wrong_width():
    kv = torch.zeros((8, 512), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match=f"tokens, {SOURCE_DIM}"):
        _prepare_primary_host_record(
            kv, primary_view=_FakeWorkerView(), source_dim=SOURCE_DIM
        )


def test_prepare_host_record_reaches_the_packed_quantizer(monkeypatch):
    """A packed region must NOT be served the bf16 latent.

    Proven POSITIVELY, by substituting the quantizer module and recording the
    operands the dispatch hands it. The earlier form of this test inferred the
    branch from an ImportError on a CPU-only machine, which another test in
    the same session can defeat by placing a fake ``triton`` in
    ``sys.modules`` -- it then passed standalone and failed in a full run. The
    byte-level contract of the quantizer itself is pinned separately by
    ``test_glm52_packed_mla_kv_record.py`` (torch reference, CPU) and
    ``test_glm52_packed_mla_kv_kernel.py`` (Triton, node).
    """

    import sys
    import types

    calls = []

    def _fake_quantize(kv, out=None):
        calls.append((kv.shape, kv.dtype, out.shape, out.dtype))
        return out

    fake = types.ModuleType("batchgen_kernels.triton.mla_kv_packed")
    fake.quantize_mla_kv_packed = _fake_quantize
    monkeypatch.setitem(
        sys.modules, "batchgen_kernels.triton.mla_kv_packed", fake
    )

    kv = torch.zeros((8, SOURCE_DIM), dtype=torch.bfloat16)
    view = _FakeWorkerView(k_head_dim=TOKEN_BYTES, k_element_size_bytes=1)
    record = _prepare_primary_host_record(
        kv, primary_view=view, source_dim=SOURCE_DIM
    )

    # The dispatch went to the quantizer, exactly once, with the latent in and
    # a [tokens, 656] uint8 record out...
    assert len(calls) == 1
    src_shape, src_dtype, out_shape, out_dtype = calls[0]
    assert src_shape == (8, SOURCE_DIM)
    assert src_dtype is torch.bfloat16
    assert out_shape == (8, TOKEN_BYTES)
    assert out_dtype is torch.uint8
    # ...and the record returned is that destination, not the latent.
    assert record.shape == (8, TOKEN_BYTES)
    assert record.dtype is torch.uint8


def test_prepare_host_record_does_not_quantize_for_a_bf16_region(monkeypatch):
    """The other half of the dispatch: a bf16 region must bypass the kernel."""

    import sys
    import types

    calls = []
    fake = types.ModuleType("batchgen_kernels.triton.mla_kv_packed")
    fake.quantize_mla_kv_packed = lambda kv, out=None: calls.append(1)
    monkeypatch.setitem(
        sys.modules, "batchgen_kernels.triton.mla_kv_packed", fake
    )

    kv = torch.zeros((8, SOURCE_DIM), dtype=torch.bfloat16)
    view = _FakeWorkerView(k_head_dim=SOURCE_DIM, k_element_size_bytes=2)
    record = _prepare_primary_host_record(
        kv, primary_view=view, source_dim=SOURCE_DIM
    )

    assert calls == []
    # The record IS the latent, by identity: GLM-5/5.1/5.3 must be unchanged.
    assert record is kv
