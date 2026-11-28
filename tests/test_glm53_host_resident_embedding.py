"""Host-resident ``embed_tokens`` for GLM-5.3.

Two kinds of check:

* behavioural, on ``HostResidentEmbedding`` -- the gather must be bit-exact
  with the replicated ``nn.Embedding`` it replaces, and the table must stay
  out of everything that would put it back on the device;
* source-contract, on the three wiring sites -- ``share_skeleton_on_device``
  must not upload the table, the swap must happen before it, and the decode
  graph must take pre-gathered rows rather than ids.

The wiring sites import triton (GPU-only), so they are asserted against the
module source instead of being imported. That is deliberate: the invariants are
about structure ("this key is subtracted from the upload set", "this call comes
first"), and they must hold on a machine that cannot import the model.
"""
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from batchgen.models.glm.glm5.host_embedding import HostResidentEmbedding

try:
    # Pulls in triton and batchgen_kernels, which a CPU-only box lacks.
    # Resolved once, at collection: a per-test importorskip leaves the failed
    # module in sys.modules, and later tests then run against that partial
    # object instead of skipping.
    from batchgen.models.glm.glm5 import (
        whole_model_cuda_graph_segments as _segments,
    )
except ImportError:
    _segments = None

requires_model_package = pytest.mark.skipif(
    _segments is None,
    reason="GLM-5 model package unavailable (needs triton + batchgen_kernels)",
)

GLM5 = Path(__file__).resolve().parents[1] / "batchgen" / "models" / "glm" / "glm5"
VOCAB, HIDDEN = 154880, 6144


def _source(name: str) -> str:
    return (GLM5 / name).read_text()


def _function_source(text: str, name: str) -> str:
    """The body of ``def <name>`` up to the next same-indent ``def``."""
    marker = f"    def {name}("
    start = text.index(marker)
    rest = text[start + len(marker):]
    end = rest.find("\n    def ")
    return rest if end == -1 else rest[:end]


# --------------------------------------------------------------------------- #
#  Gather correctness
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(5, 1), (3, 4), (7,)])
def test_gather_is_bit_exact_with_nn_embedding(dtype, shape):
    torch.manual_seed(0)
    vocab, hidden = 97, 11
    table = torch.randn(vocab, hidden).to(dtype)
    ids = torch.randint(0, vocab, shape)

    module = HostResidentEmbedding(table, torch.device("cpu"))
    got = module(ids)
    want = torch.nn.functional.embedding(ids, table)

    assert got.shape == tuple(shape) + (hidden,)
    # Bit-exact, not allclose: a row gather must reproduce the row.
    assert torch.equal(got, want)


def test_padding_idx_row_is_still_returned_verbatim():
    """``padding_idx`` affects gradients only, so forward must not zero it."""
    torch.manual_seed(0)
    table = torch.randn(16, 4).to(torch.bfloat16)
    ids = torch.tensor([[3], [0], [3]])

    plain = torch.nn.Embedding.from_pretrained(table, padding_idx=3, freeze=True)
    module = HostResidentEmbedding(table, torch.device("cpu"), padding_idx=3)

    assert module.padding_idx == 3
    assert torch.equal(module(ids), plain(ids))


def test_empty_batch_gathers_zero_rows():
    table = torch.randn(16, 4).to(torch.bfloat16)
    module = HostResidentEmbedding(table, torch.device("cpu"))
    out = module(torch.empty((0, 1), dtype=torch.int64))
    assert out.shape == (0, 1, 4)


def test_decode_sized_staging_buffer_is_allocated_once_and_reused():
    table = torch.randn(32, 6).to(torch.bfloat16)
    module = HostResidentEmbedding(table, torch.device("cpu"), max_pinned_rows=8)

    small = module.gather_host_rows(torch.tensor([1, 2]))
    assert small.shape == (2, 6)
    first_ptr = module._staging.data_ptr()
    # Sized at the cap up front, so no row count up to it reallocates.
    assert module._staging.shape[0] == 8
    rows = module.gather_host_rows(torch.arange(8))
    assert module._staging.data_ptr() == first_ptr
    assert torch.equal(rows, table[:8])


def test_prefill_sized_gather_does_not_grow_the_pinned_buffer():
    """One row per prompt token is hundreds of MiB at 6144 x 2 B; a pinned
    allocation that large would cost more than the copy it accelerates."""
    table = torch.randn(64, 6).to(torch.bfloat16)
    module = HostResidentEmbedding(table, torch.device("cpu"), max_pinned_rows=8)

    ids = torch.arange(40)
    out = module(ids)
    assert torch.equal(out, torch.nn.functional.embedding(ids, table))
    # The big gather went to a throwaway buffer, not the pinned one.
    assert module._staging is None or module._staging.shape[0] == 8


def test_out_of_range_ids_fail_loud():
    table = torch.randn(8, 3).to(torch.bfloat16)
    module = HostResidentEmbedding(table, torch.device("cpu"))
    with pytest.raises(IndexError):
        module(torch.tensor([[8]]))
    with pytest.raises(IndexError):
        module(torch.tensor([[-1]]))


def test_device_ids_are_accepted_without_a_host_table_copy():
    table = torch.randn(8, 3).to(torch.bfloat16)
    module = HostResidentEmbedding(table, torch.device("cpu"))
    before = module.host_weight.data_ptr()
    module(torch.tensor([[2], [5]]))
    assert module.host_weight.data_ptr() == before


# --------------------------------------------------------------------------- #
#  The table must stay off the device
# --------------------------------------------------------------------------- #

def test_table_is_not_a_parameter_or_buffer():
    """``share_skeleton_on_device`` derives its upload set from
    ``named_parameters()``; a Parameter here would put the table back on the
    device, and a registered buffer would let ``.to(device)`` move it."""
    table = torch.randn(8, 3).to(torch.bfloat16)
    module = HostResidentEmbedding(table, torch.device("cpu"))

    assert list(module.named_parameters()) == []
    assert list(module.named_buffers()) == []
    assert module.state_dict() == {}
    assert module.weight.data_ptr() == table.data_ptr()


def test_module_to_does_not_move_the_table():
    table = torch.randn(8, 3).to(torch.bfloat16)
    module = HostResidentEmbedding(table, torch.device("cpu"))
    before = module.host_weight.data_ptr()
    module.to(torch.device("cpu"))
    module.float()
    assert module.host_weight.data_ptr() == before
    assert module.host_weight.dtype == torch.bfloat16


def test_module_does_not_pin_the_table():
    """Pinning a shared mapping privatizes its pages: one 1.77 GiB host copy
    would become one per rank. Only the small staging buffer may be pinned."""
    table = torch.randn(8, 3).to(torch.bfloat16)
    assert not table.is_pinned()
    module = HostResidentEmbedding(table, torch.device("cpu"))
    module(torch.tensor([[1]]))
    assert not module.host_weight.is_pinned()


def test_a_device_table_is_rejected():
    table = torch.randn(8, 3)
    module = HostResidentEmbedding(table, torch.device("cpu"))
    assert module.host_weight.device.type == "cpu"
    with pytest.raises(ValueError, match="2-D"):
        HostResidentEmbedding(torch.randn(8), torch.device("cpu"))


def test_reported_host_bytes_match_the_glm53_table():
    table = torch.empty(0, HIDDEN, dtype=torch.bfloat16)
    module = HostResidentEmbedding(table, torch.device("cpu"))
    module.num_embeddings = VOCAB
    assert VOCAB * HIDDEN * 2 == 1903165440
    assert round(1903165440 / (1024 ** 3), 4) == 1.7725


# --------------------------------------------------------------------------- #
#  Source contract: share_skeleton_on_device must not upload the table
# --------------------------------------------------------------------------- #

def test_share_skeleton_on_device_subtracts_the_host_resident_keys():
    text = _source("Parallel_Strategy_Manager.py")
    body = _function_source(text, "share_skeleton_on_device")
    assert "self._host_resident_skeleton_keys()" in body
    assert "upload -= host_resident" in body
    # The subtraction must come after the upload set is built, or it is a no-op.
    assert body.index("upload = self._skeleton_bind_keys()") < body.index(
        "upload -= host_resident"
    )


def test_host_resident_keys_cover_embed_tokens_and_not_lm_head():
    text = _source("Parallel_Strategy_Manager.py")
    body = _function_source(text, "_host_resident_skeleton_keys")
    assert "self.EMBED_TOKENS_SKELETON_KEY" in body
    assert "lm_head" not in body.split('"""')[2]
    assert 'EMBED_TOKENS_SKELETON_KEY = "model.embed_tokens.weight"' in text


def test_default_is_on_for_glm53_and_uses_no_env_guard():
    """Default ON for GLM-5.3, decided from the model type. The switch is a
    plain attribute, so a CLI flag can set it before the first configure; it is
    never an env guard."""
    text = _source("Parallel_Strategy_Manager.py")
    assert "self.host_resident_embed_tokens = (" in text
    assert '== "glm_moe_dsa_5_3"' in text
    # Scoped to the code this change added -- the file has unrelated
    # pre-existing env reads.
    added = "".join(
        _function_source(text, name)
        for name in (
            "_host_resident_skeleton_keys",
            "_config_host_resident_embed_tokens",
        )
    )
    for forbidden in ("BATCHGEN_", "os.environ", "os.getenv"):
        assert forbidden not in added, forbidden
    assert "BATCHGEN_" not in _source("host_embedding.py")


def test_the_swap_runs_before_the_skeleton_is_shared_or_bound():
    """Ordering is the whole mechanism: after ``share_skeleton_on_device`` the
    table is already on the device, and after ``.to(device)`` a Parameter would
    have been moved."""
    text = _source("Parallel_Strategy_Manager.py")
    for phase in ("configure_prefill", "configure_decoding"):
        body = _function_source(text, phase)
        assert "self._config_host_resident_embed_tokens()" in body, phase
        swap = body.index("self._config_host_resident_embed_tokens()")
        for later in ("self._load_model_skeleton()", "self.model.to("):
            if later in body:
                assert swap < body.index(later), (phase, later)
    prefill = _function_source(text, "configure_prefill")
    assert prefill.index("self._config_host_resident_embed_tokens()") < prefill.index(
        "self.share_skeleton_on_device()"
    )


def test_the_swap_fails_loud_if_the_table_is_already_on_the_device():
    body = _function_source(
        _source("Parallel_Strategy_Manager.py"), "_config_host_resident_embed_tokens"
    )
    assert 'if host_weight.device.type != "cpu":' in body
    assert "raise RuntimeError" in body


# --------------------------------------------------------------------------- #
#  Source contract: the decode graph takes rows, not ids
# --------------------------------------------------------------------------- #

def test_graph_replaces_input_ids_with_inputs_embeds_when_host_resident():
    body = _function_source(
        _source("whole_model_cuda_graph_segments.py"), "get_static_input_specs"
    )
    assert "if self.embed_is_host_resident:" in body
    # Dropping `input_ids` is what turns a stale-rows replay into a KeyError.
    assert 'specs.pop("input_ids")' in body
    assert 'specs["inputs_embeds"] = TensorSpec(' in body
    assert '("batch_size", 1, self.hidden_size), torch.bfloat16' in body


def test_graph_forward_requires_exactly_one_embedding_entry_point():
    body = _function_source(
        _source("whole_model_cuda_graph_segments.py"), "forward"
    )
    assert "if (input_ids is None) == (inputs_embeds is None):" in body
    assert "raise RuntimeError" in body


def test_capture_derives_rows_from_the_real_capture_ids():
    """Warmup runs the real model over live slots, so zero-filled embeds would
    write KV derived from zeros into real pages."""
    body = _function_source(
        _source("whole_model_cuda_graph_segments.py"), "initialize_static_inputs"
    )
    assert 'self._capture_inputs.get("input_ids")' in body
    assert "self.model.model.embed_tokens(ids)" in body


def test_adapter_sends_rows_and_drops_ids():
    text = _source("cuda_graph_adapter.py")
    body = _function_source(text, "prepare_replay_inputs")
    assert "self._host_resident_embed()" in body
    assert 'replay_inputs.pop("input_ids")' in body
    assert 'replay_inputs["inputs_embeds"] = host_embed(' in body
    helper = _function_source(text, "_host_resident_embed")
    assert 'getattr(embed, "host_resident", False)' in helper


def test_adapter_gathers_the_full_bucket_so_padded_rows_keep_embed_zero():
    """The device-resident build padded `input_ids` with id 0, so its padded
    hidden states were embed[0]. `manager.replay` zero-fills short inputs, so a
    local_bsz-sized embeds tensor would change what padding carries."""
    body = _function_source(
        _source("cuda_graph_adapter.py"), "prepare_replay_inputs"
    )
    assert "self._ctx.bucketing.get_padded_size(int(batch_state.max_rank_bsz))" in body
    assert "padded_ids = torch.zeros((bucket, 1), dtype=torch.int64, device=device)" in body
    assert "host_embed(padded_ids)" in body


def test_capture_stub_keeps_batch_dim_zero():
    """Capture must not mutate real KV, so every capture input stays 0-row --
    including the embeds buffer."""
    body = _function_source(_source("cuda_graph_adapter.py"), "capture_inputs_for")
    assert 'empty_inputs.pop("input_ids")' in body
    assert "(0, 1, host_embed.embedding_dim)" in body


# --------------------------------------------------------------------------- #
#  Behavioural segment contract (needs the model package: triton/kernels)
# --------------------------------------------------------------------------- #

def _bare_segment(embed):
    class _Inner:
        pass

    class _Model:
        pass

    inner, model = _Inner(), _Model()
    inner.embed_tokens = embed
    model.model = inner

    seg = object.__new__(_segments.Glm5WholeModelSegment)
    seg.model = model
    seg.world_size = 8
    seg.hidden_size = HIDDEN
    seg.max_seqlen = 4096
    seg.max_bucket_size = 128
    seg.layer_segments = []
    seg._capture_inputs = None
    seg._capture_only_inputs = frozenset()
    return seg


@requires_model_package
def test_segment_spec_swaps_ids_for_rows_only_when_host_resident():
    table = torch.randn(8, HIDDEN).to(torch.bfloat16)
    host = HostResidentEmbedding(table, torch.device("cpu"))

    seg = _bare_segment(host)
    assert seg.embed_is_host_resident is True
    specs = seg.get_static_input_specs(128)
    assert "input_ids" not in specs
    assert specs["inputs_embeds"].shape == ("batch_size", 1, HIDDEN)
    assert specs["inputs_embeds"].dtype == torch.bfloat16

    device_seg = _bare_segment(torch.nn.Embedding(8, HIDDEN))
    assert device_seg.embed_is_host_resident is False
    device_specs = device_seg.get_static_input_specs(128)
    assert "input_ids" in device_specs
    assert "inputs_embeds" not in device_specs


@requires_model_package
def test_segment_forward_rejects_both_and_neither():
    seg = _bare_segment(
        HostResidentEmbedding(
            torch.randn(8, HIDDEN).to(torch.bfloat16), torch.device("cpu")
        )
    )
    common = dict(
        cache_seqlens=torch.zeros(2, dtype=torch.int32),
        position_ids=torch.zeros((2, 1), dtype=torch.int64),
        primary_slot_indices=torch.zeros(2, dtype=torch.int32),
        aux_slot_indices=torch.zeros(2, dtype=torch.int32),
        rank_token_counts=torch.zeros(8, dtype=torch.int64),
    )
    with pytest.raises(RuntimeError, match="exactly one"):
        seg.forward(**common)
    with pytest.raises(RuntimeError, match="exactly one"):
        seg.forward(
            input_ids=torch.zeros((2, 1), dtype=torch.int64),
            inputs_embeds=torch.zeros((2, 1, HIDDEN), dtype=torch.bfloat16),
            **common,
        )


@requires_model_package
def test_set_capture_inputs_takes_ids_as_capture_only():
    seg = _bare_segment(
        HostResidentEmbedding(
            torch.randn(8, HIDDEN).to(torch.bfloat16), torch.device("cpu")
        )
    )
    seg.set_capture_inputs(
        input_ids=torch.zeros((0, 1), dtype=torch.int64),
        cache_seqlens=torch.zeros(0, dtype=torch.int32),
        position_ids=torch.zeros((0, 1), dtype=torch.int64),
        primary_slot_indices=torch.zeros(0, dtype=torch.int32),
        aux_slot_indices=torch.zeros(0, dtype=torch.int32),
        rank_token_counts=torch.zeros(8, dtype=torch.int64),
    )
    # Accepted, and marked as having no static buffer.
    assert seg._capture_only_inputs == frozenset({"input_ids"})
    with pytest.raises(ValueError, match="unknown"):
        seg.set_capture_inputs(
            input_ids=torch.zeros((0, 1), dtype=torch.int64),
            cache_seqlens=torch.zeros(0, dtype=torch.int32),
            position_ids=torch.zeros((0, 1), dtype=torch.int64),
            primary_slot_indices=torch.zeros(0, dtype=torch.int32),
            aux_slot_indices=torch.zeros(0, dtype=torch.int32),
            rank_token_counts=torch.zeros(8, dtype=torch.int64),
            nonsense=torch.zeros(1),
        )


def test_lm_head_is_untouched():
    """Only ``embed_tokens`` moves; ``lm_head`` is a full [V, H] matmul every
    step and must stay device-resident."""
    text = _source("Parallel_Strategy_Manager.py")
    body = _function_source(text, "_config_host_resident_embed_tokens")
    assert "lm_head" not in body
    assert "def _lm_head_forward_pre_hook" in text
    assert "self.model.lm_head.register_forward_pre_hook" in text
