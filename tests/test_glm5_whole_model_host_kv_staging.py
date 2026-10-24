"""CPU pins for the GLM-5 whole-model decode host-KV staging retarget.

The whole-model graph used to copy every layer's fresh decode KV into its
staging buffers after the layer ran: 78 primary copies plus 21 aux pairs, 120
captured kernels of pure launch overhead. The producing kernels already write
those rows to a static address -- the primary row through the fused norm+rope
launch's ``out=`` target, a page-split layer's aux row through the fused
writer's ``k_fp8_out``/``scales_out`` -- so the staging buffers are now bound AS
those kernels' output targets and no staging copy survives.

These tests pin the BINDING (which buffer each producing kernel writes into),
not the kernels themselves, so they need no GPU.
"""

from types import SimpleNamespace

import pytest
import torch

from batchgen.models.glm.glm5 import cuda_graph_segments as dsa
from batchgen.models.glm.glm5 import reuse_topk_segment as reuse
from batchgen.models.glm.glm5.whole_model_cuda_graph_segments import (
    Glm5WholeModelSegment,
)

KV_LORA_RANK = 24
QK_ROPE_HEAD_DIM = 8
PRIMARY_KV_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
AUX_KV_DIM = 16
HIDDEN_SIZE = 32
NUM_LAYERS = 4
# Layers 0 and 2 run the indexer and therefore own aux host-KV rows; 1 and 3
# are skip layers, exactly the GLM-5.2 shape in miniature.
AUX_LAYERS = (0, 2)
MAX_BUCKET = 8


class _FakeLayerSegment:
    """Records what the whole-model segment binds, and in which order."""

    def __init__(self, *, writes_aux: bool, log: list):
        self.WRITES_AUX_HOST_KV = writes_aux
        self._log = log
        self.staging = None
        self.staged_buckets = []

    def set_host_kv_staging(self, *, key, aux_key=None, aux_scale=None):
        self._log.append(("install", id(self)))
        self.staging = (
            None if key is None else {"key": key, "aux_key": aux_key, "aux_scale": aux_scale}
        )

    def setup_static_buffers(self, bucket_size):
        self._log.append(("setup", id(self)))
        self.staged_buckets.append(bucket_size)

    def release_static_buffers(self, bucket_size):
        pass

    # -- run_model_with_probes plumbing ---------------------------------------
    def row_validity_prologue_key(self, bucket_size):
        return ("fake", id(self))

    def run_row_validity_prologue(self, bucket_size, **_kw):
        return None

    def paged_mqa_prep_key(self, bucket_size):
        return None

    def run_paged_mqa_prep(self, bucket_size, **_kw):
        return None

    selects_top_k = True

    def selected_positions_key(self, bucket_size):
        return None

    def forward(self, *, hidden_states, residual=None, **_kw):
        bsz = hidden_states.shape[0]
        out = {
            "hidden_states": hidden_states,
            "residual": torch.zeros_like(hidden_states) if residual is None else residual,
            "primary_k_tensor": torch.zeros(bsz, 1, 1, PRIMARY_KV_DIM),
        }
        if self.WRITES_AUX_HOST_KV:
            out["indexer_k_tensor"] = torch.zeros(
                bsz, 1, 1, AUX_KV_DIM, dtype=torch.float8_e4m3fn
            )
            out["indexer_k_scale"] = torch.zeros(bsz)
        return out


def _fake_model():
    attn = SimpleNamespace(
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        num_heads=4,
        hidden_size=HIDDEN_SIZE,
        indexer=SimpleNamespace(index_head_dim=AUX_KV_DIM, index_topk=64),
    )
    layers = [SimpleNamespace(self_attn=SimpleNamespace(module=attn)) for _ in range(NUM_LAYERS)]
    return SimpleNamespace(
        model=SimpleNamespace(
            layers=layers,
            embed_tokens=lambda ids: torch.zeros(ids.shape[0], 1, HIDDEN_SIZE),
            norm=lambda h: h,
        ),
        lm_head=lambda h: torch.zeros(h.shape[0], 1, 7),
    )


def _make_segment(*, stage_kv_for_offload=True, layer_segments=None):
    return Glm5WholeModelSegment(
        model=_fake_model(),
        device=torch.device("cpu"),
        world_size=1,
        max_pages_per_seq=4,
        max_aux_pages_per_seq=4,
        vocab_size=7,
        hidden_size=HIDDEN_SIZE,
        max_bucket_size=MAX_BUCKET,
        max_seqlen=128,
        layer_segments=layer_segments,
        aux_kv_staging_dtype=torch.float8_e4m3fn,
        stage_kv_for_offload=stage_kv_for_offload,
    )


def _fake_layer_segments(log):
    return [
        _FakeLayerSegment(writes_aux=(idx in AUX_LAYERS), log=log)
        for idx in range(NUM_LAYERS)
    ]


# -- (a) no per-layer staging copies survive ---------------------------------


def test_layer_loop_performs_no_per_layer_staging_copy():
    """The composed layer loop must not copy KV anywhere.

    The rows are already in the staging buffers because the layer segments
    wrote them there, so a copy would be a self-copy. Both staging helpers are
    replaced with raisers: reaching either one is the regression.
    """

    log: list = []
    segments = _fake_layer_segments(log)
    seg = _make_segment(layer_segments=segments)
    seg.setup_static_buffers(MAX_BUCKET)

    def _boom(*_a, **_k):
        raise AssertionError("whole-model layer loop performed a staging copy")

    seg._copy_primary_kv = _boom
    seg._copy_aux_kv = _boom

    outputs = seg.run_model_with_probes(
        input_ids=torch.zeros(MAX_BUCKET, 1, dtype=torch.int64),
        position_ids=torch.zeros(MAX_BUCKET, 1, dtype=torch.int64),
        cache_seqlens=torch.ones(MAX_BUCKET, dtype=torch.int32),
        primary_slot_indices=torch.zeros(MAX_BUCKET, dtype=torch.int32),
        aux_slot_indices=torch.zeros(MAX_BUCKET, dtype=torch.int32),
        num_valid_tokens=torch.tensor([MAX_BUCKET], dtype=torch.int32),
        rank_token_counts=torch.tensor([MAX_BUCKET], dtype=torch.int64),
    )
    assert "logits" in outputs


# -- (b) the staging buffers ARE the layer segments' output targets ----------


def test_setup_binds_each_layer_its_own_staging_rows():
    log: list = []
    segments = _fake_layer_segments(log)
    seg = _make_segment(layer_segments=segments)
    seg.setup_static_buffers(MAX_BUCKET)

    for layer_idx, layer_segment in enumerate(segments):
        staging = layer_segment.staging
        assert staging is not None, f"layer {layer_idx} got no staging"
        # Exactly this layer's row of the primary staging buffer.
        assert staging["key"] is seg._kv_buffers[layer_idx]["key"]
        assert staging["key"].data_ptr() == seg._kv_key_buffer[layer_idx].data_ptr()
        if layer_idx in AUX_LAYERS:
            assert staging["aux_key"] is seg._aux_kv_buffers[layer_idx]["key"]
            # Page-split aux: keys and their fp32 scales stage together.
            assert staging["aux_scale"] is seg._aux_kv_buffers[layer_idx]["scale"]
            assert staging["aux_scale"].dtype is torch.float32
        else:
            # A skip layer never writes the aux cache, so it must not be
            # handed aux rows it would leave stale.
            assert staging["aux_key"] is None
            assert staging["aux_scale"] is None


def test_staging_is_installed_before_layer_segments_build_statics():
    """Ordering is load-bearing.

    A layer segment binds its host-append output buffers inside its own
    static setup, so an install that landed afterwards would leave it on
    private buffers and the staged rows permanently stale.
    """

    log: list = []
    segments = _fake_layer_segments(log)
    seg = _make_segment(layer_segments=segments)
    seg.setup_static_buffers(MAX_BUCKET)

    first_setup = next(i for i, (kind, _) in enumerate(log) if kind == "setup")
    installs = [i for i, (kind, _) in enumerate(log) if kind == "install"]
    assert installs, "no staging install happened"
    assert max(installs) < first_setup


def test_release_drops_the_staging_references():
    log: list = []
    segments = _fake_layer_segments(log)
    seg = _make_segment(layer_segments=segments)
    seg.setup_static_buffers(MAX_BUCKET)
    seg.release_static_buffers(MAX_BUCKET)
    assert seg._kv_key_buffer is None
    for layer_segment in segments:
        assert layer_segment.staging is None


# -- (c) switch off == today's arm-B semantics -------------------------------


def test_staging_switch_off_allocates_nothing_and_binds_nothing():
    log: list = []
    segments = _fake_layer_segments(log)
    seg = _make_segment(stage_kv_for_offload=False, layer_segments=segments)
    seg.setup_static_buffers(MAX_BUCKET)

    assert seg._kv_buffers is None
    assert seg._aux_kv_buffers is None
    assert seg._kv_key_buffer is None
    assert seg._aux_kv_key_buffer is None
    assert seg._aux_kv_scale_buffer is None
    assert seg.primary_kv_offload_buffers is None
    assert seg.aux_kv_offload_buffers is None
    # No retarget: every layer segment keeps its private buffers.
    for layer_segment in segments:
        assert layer_segment.staging is None
    # The layer segments are still set up for the bucket.
    for layer_segment in segments:
        assert layer_segment.staged_buckets == [MAX_BUCKET]


def test_missing_setter_fails_loud_rather_than_staging_nothing():
    log: list = []
    segments = _fake_layer_segments(log)
    # A segment that cannot take staging slices would silently keep private
    # buffers and leave its staged rows stale, so it must not be tolerated.
    segments[1] = SimpleNamespace(
        WRITES_AUX_HOST_KV=False,
        setup_static_buffers=lambda bucket_size: None,
        release_static_buffers=lambda bucket_size: None,
    )
    seg = _make_segment(layer_segments=segments)
    with pytest.raises(RuntimeError, match="set_host_kv_staging"):
        seg.setup_static_buffers(MAX_BUCKET)


# -- the DSA segments' own output-target selection ---------------------------


def _bare_full_segment(*, page_split=True):
    """A Glm5FullDsaAttnSegment with only the fields the two methods read.

    The real constructor dereferences indexer weights and FlashAttention-3, so
    the buffer-selection logic is exercised directly instead.
    """

    seg = object.__new__(dsa.Glm5FullDsaAttnSegment)
    seg.attn = SimpleNamespace(
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        hidden_size=HIDDEN_SIZE,
    )
    seg.primary_blocked_k = torch.zeros(1, 1, 1, PRIMARY_KV_DIM)
    seg.aux_host_kv_dim = AUX_KV_DIM
    seg.aux_host_kv_dtype = torch.float8_e4m3fn if page_split else torch.bfloat16
    seg.aux_is_page_split = page_split
    seg._outputs = {}
    seg._host_kv_staging = None
    return seg


def _bare_reuse_segment():
    seg = object.__new__(reuse.Glm5ReuseTopkAttnSegment)
    seg.attn = SimpleNamespace(
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        hidden_size=HIDDEN_SIZE,
    )
    seg.primary_blocked_k = torch.zeros(1, 1, 1, PRIMARY_KV_DIM)
    seg._outputs = {}
    seg._host_kv_staging = None
    return seg


def _staging_rows(layer_idx=2, num_layers=NUM_LAYERS, alloc=MAX_BUCKET):
    key = torch.zeros(num_layers, alloc, 1, 1, PRIMARY_KV_DIM, dtype=torch.bfloat16)
    aux_key = torch.zeros(num_layers, alloc, 1, 1, AUX_KV_DIM, dtype=torch.float8_e4m3fn)
    aux_scale = torch.zeros(num_layers, alloc, dtype=torch.float32)
    return key, aux_key, aux_scale, layer_idx


def test_full_segment_without_staging_allocates_private_buffers():
    seg = _bare_full_segment()
    seg._setup_static_output_buffers(MAX_BUCKET)
    outputs = seg._outputs[MAX_BUCKET]
    assert outputs.primary_k_tensor.shape == (MAX_BUCKET, 1, 1, PRIMARY_KV_DIM)
    assert outputs.indexer_k_tensor.shape == (MAX_BUCKET, 1, 1, AUX_KV_DIM)
    assert outputs.indexer_k_scale.shape == (MAX_BUCKET,)


def test_full_segment_with_staging_writes_into_the_staging_rows():
    key, aux_key, aux_scale, li = _staging_rows()
    seg = _bare_full_segment()
    seg.set_host_kv_staging(
        key=key[li], aux_key=aux_key[li], aux_scale=aux_scale[li]
    )
    seg._setup_static_output_buffers(MAX_BUCKET)
    outputs = seg._outputs[MAX_BUCKET]

    # Same storage, and at this layer's offset -- not merely the same shape.
    assert outputs.primary_k_tensor.untyped_storage().data_ptr() == (
        key.untyped_storage().data_ptr()
    )
    assert outputs.primary_k_tensor.data_ptr() == key[li].data_ptr()
    assert outputs.indexer_k_tensor.data_ptr() == aux_key[li].data_ptr()
    assert outputs.indexer_k_scale.data_ptr() == aux_scale[li].data_ptr()

    # A write through the output buffer lands in the staging buffer: that is
    # the whole mechanism, so it is asserted rather than inferred.
    outputs.primary_k_tensor.fill_(1.0)
    assert torch.all(key[li] == 1.0)
    assert torch.all(key[(li + 1) % NUM_LAYERS] == 0.0)


def test_staged_slices_satisfy_the_fused_writer_buffer_contract():
    """The native writers check numel and contiguity against N rows.

    ``fused_indexer_k_write_fp8`` views ``k_fp8_out`` as ``[N, data_dim]`` and
    both FP8 writers assert ``numel == N * data_dim`` / ``numel == N`` with
    contiguous storage, where N is this step's row count -- so a bucket smaller
    than the allocation must bind the prefix, never the whole buffer.
    """

    key, aux_key, aux_scale, li = _staging_rows()
    bucket = MAX_BUCKET // 2
    seg = _bare_full_segment()
    seg.set_host_kv_staging(key=key[li], aux_key=aux_key[li], aux_scale=aux_scale[li])
    seg._setup_static_output_buffers(bucket)
    outputs = seg._outputs[bucket]

    assert outputs.indexer_k_tensor.is_contiguous()
    assert outputs.indexer_k_tensor.numel() == bucket * AUX_KV_DIM
    assert outputs.indexer_k_scale.is_contiguous()
    assert outputs.indexer_k_scale.numel() == bucket
    assert outputs.primary_k_tensor.is_contiguous()
    # The fused norm+rope launch takes this `out=` view.
    assert outputs.primary_k_tensor.view(bucket, 1, -1).shape == (
        bucket,
        1,
        PRIMARY_KV_DIM,
    )


def test_smaller_bucket_binds_a_prefix_of_the_same_staging_rows():
    key, aux_key, aux_scale, li = _staging_rows()
    seg = _bare_full_segment()
    seg.set_host_kv_staging(key=key[li], aux_key=aux_key[li], aux_scale=aux_scale[li])
    seg._setup_static_output_buffers(MAX_BUCKET)
    seg._setup_static_output_buffers(MAX_BUCKET // 2)
    big = seg._outputs[MAX_BUCKET]
    small = seg._outputs[MAX_BUCKET // 2]
    assert small.primary_k_tensor.data_ptr() == big.primary_k_tensor.data_ptr()
    assert small.primary_k_tensor.shape[0] == MAX_BUCKET // 2
    # `attn_output` is this segment's own value, not a host-append source, so
    # it keeps the largest-bucket base-plus-view footprint.
    assert small.attn_output.data_ptr() == big.attn_output.data_ptr()


def test_reuse_segment_stages_primary_only():
    key, _aux_key, _aux_scale, li = _staging_rows()
    seg = _bare_reuse_segment()
    assert seg.WRITES_AUX_HOST_KV is False
    seg.set_host_kv_staging(key=key[li])
    seg._setup_static_output_buffers(MAX_BUCKET)
    outputs = seg._outputs[MAX_BUCKET]
    assert outputs.primary_k_tensor.data_ptr() == key[li].data_ptr()
    # Skip layers never write aux; the placeholder must stay a placeholder.
    assert outputs.indexer_k_tensor.numel() == 1
    assert outputs.indexer_k_scale is None


def test_unstaged_smaller_bucket_still_views_the_largest_base():
    """Arm B must keep today's static footprint.

    With staging off, a smaller bucket is a leading-dim view of the largest
    bucket's base, not its own allocation. Allocating instead would silently
    multiply the per-layer static footprint on the diagnostic arm.
    """

    seg = _bare_full_segment()
    seg._setup_static_output_buffers(MAX_BUCKET)
    seg._setup_static_output_buffers(MAX_BUCKET // 2)
    big = seg._outputs[MAX_BUCKET]
    small = seg._outputs[MAX_BUCKET // 2]
    assert small.primary_k_tensor.data_ptr() == big.primary_k_tensor.data_ptr()
    assert small.indexer_k_tensor.data_ptr() == big.indexer_k_tensor.data_ptr()
    assert small.indexer_k_scale.data_ptr() == big.indexer_k_scale.data_ptr()
    assert small.attn_output.data_ptr() == big.attn_output.data_ptr()

    reuse_seg = _bare_reuse_segment()
    reuse_seg._setup_static_output_buffers(MAX_BUCKET)
    reuse_seg._setup_static_output_buffers(MAX_BUCKET // 2)
    assert reuse_seg._outputs[MAX_BUCKET // 2].primary_k_tensor.data_ptr() == (
        reuse_seg._outputs[MAX_BUCKET].primary_k_tensor.data_ptr()
    )


def test_reuse_segment_without_staging_allocates_private_buffer():
    seg = _bare_reuse_segment()
    seg._setup_static_output_buffers(MAX_BUCKET)
    outputs = seg._outputs[MAX_BUCKET]
    assert outputs.primary_k_tensor.shape == (MAX_BUCKET, 1, 1, PRIMARY_KV_DIM)
    assert outputs.primary_k_tensor.untyped_storage().data_ptr() != 0


def test_clearing_staging_restores_private_buffers():
    key, aux_key, aux_scale, li = _staging_rows()
    seg = _bare_full_segment()
    seg.set_host_kv_staging(key=key[li], aux_key=aux_key[li], aux_scale=aux_scale[li])
    seg._setup_static_output_buffers(MAX_BUCKET)
    seg.set_host_kv_staging(key=None)
    seg._setup_static_output_buffers(MAX_BUCKET)
    outputs = seg._outputs[MAX_BUCKET]
    assert outputs.primary_k_tensor.data_ptr() != key[li].data_ptr()


def test_reinstalling_the_same_staging_keeps_the_buffer_set():
    """A repeated install must not clear ``_outputs``.

    Clearing would drop the largest-bucket base that every smaller bucket's
    ``attn_output`` is a view of, quietly multiplying the static footprint.
    """

    key, aux_key, aux_scale, li = _staging_rows()
    seg = _bare_full_segment()
    seg.set_host_kv_staging(key=key[li], aux_key=aux_key[li], aux_scale=aux_scale[li])
    seg._setup_static_output_buffers(MAX_BUCKET)
    before = seg._outputs[MAX_BUCKET]
    seg.set_host_kv_staging(key=key[li], aux_key=aux_key[li], aux_scale=aux_scale[li])
    assert seg._outputs[MAX_BUCKET] is before


# -- the aux scale-pairing contract stays fail-loud --------------------------


def test_page_split_staging_requires_scale_rows():
    key, aux_key, _aux_scale, li = _staging_rows()
    seg = _bare_full_segment(page_split=True)
    with pytest.raises(ValueError, match="page-split"):
        seg.set_host_kv_staging(key=key[li], aux_key=aux_key[li], aux_scale=None)


def test_aos_staging_rejects_scale_rows():
    key, _aux_key, aux_scale, li = _staging_rows()
    aos_key = torch.zeros(NUM_LAYERS, MAX_BUCKET, 1, 1, AUX_KV_DIM, dtype=torch.bfloat16)
    seg = _bare_full_segment(page_split=False)
    with pytest.raises(ValueError, match="page-split"):
        seg.set_host_kv_staging(
            key=key[li], aux_key=aos_key[li], aux_scale=aux_scale[li]
        )


def test_indexer_layer_requires_an_aux_slice():
    key, _aux_key, _aux_scale, li = _staging_rows()
    seg = _bare_full_segment()
    with pytest.raises(ValueError, match="aux key slice"):
        seg.set_host_kv_staging(key=key[li])


def test_skip_layer_rejects_aux_slices():
    key, aux_key, aux_scale, li = _staging_rows()
    seg = _bare_reuse_segment()
    with pytest.raises(ValueError, match="does not"):
        seg.set_host_kv_staging(
            key=key[li], aux_key=aux_key[li], aux_scale=aux_scale[li]
        )


def test_mismatched_primary_staging_dim_fails_loud():
    seg = _bare_full_segment()
    wrong = torch.zeros(MAX_BUCKET, 1, 1, PRIMARY_KV_DIM + 1, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="trailing elements"):
        seg.set_host_kv_staging(key=wrong)


def test_aux_slices_without_a_primary_slice_fail_loud():
    _key, aux_key, aux_scale, li = _staging_rows()
    seg = _bare_full_segment()
    with pytest.raises(ValueError, match="as a whole"):
        seg.set_host_kv_staging(key=None, aux_key=aux_key[li], aux_scale=aux_scale[li])
