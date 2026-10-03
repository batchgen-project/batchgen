from pathlib import Path

import pytest

from batchgen.sequence import SequenceEntry, SequenceStatus


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "batchgen/batchgen_worker.py"


def test_prefill_to_decode_metadata_sync_brackets_status_transition():
    source = WORKER.read_text()
    block_start = source.index("\t\t\t\tif not decode_uuids:\n\t\t\t\t\tbreak")
    block_end = source.index("\t\t\t\t# CUDA Graph Warmup", block_start)
    block = source[block_start:block_end]

    pre_config_sync = block.index("self._sync_sequence_metadata(decode_uuids)")
    config_decode = block.index("self._config_decoding_for_batch")
    mark_in_decode = block.index(
        "self._update_batch_status(decode_uuids, SequenceStatus.IN_DECODE)"
    )
    post_status_sync = block.index(
        "self._sync_sequence_metadata(decode_uuids)",
        mark_in_decode,
    )

    assert pre_config_sync < config_decode < mark_in_decode < post_status_sync


def test_initial_host_kv_capacity_is_page_rounded_before_metadata_validation():
    source = WORKER.read_text()
    assert (
        "seq.host_pages_allocated = math.ceil(initial_capacity / seq.PAGE_SIZE)\n"
        "\t\t\t\tseq.host_token_capacity = seq.host_pages_allocated * seq.PAGE_SIZE"
    ) in source

    seq = SequenceEntry("seq", global_idx=24, prompt_length=6087, max_decode_length=4096)
    seq.status = SequenceStatus.PREFILLED
    seq.assigned_rank = 1
    seq.host_pages_allocated = 96
    seq.host_token_capacity = 6087

    with pytest.raises(RuntimeError, match="host_token_capacity=6087"):
        seq.validate_metadata("unit")

    seq.host_token_capacity = seq.host_pages_allocated * seq.PAGE_SIZE
    seq.validate_metadata("unit")


def test_terminal_decode_boundary_allows_only_the_final_capacity_gap():
    """EOS/length completion is released before another forward is issued."""
    seq = SequenceEntry("seq", global_idx=25, prompt_length=100, max_decode_length=4000)
    seq.status = SequenceStatus.IN_DECODE
    seq.assigned_rank = 1
    seq.decoded_length = 3357
    seq.current_context_length = 3457
    seq.host_pages_allocated = 54
    seq.host_token_capacity = 3456
    seq.gpu_pages_allocated = 54
    seq.eos_reached = True

    with pytest.raises(RuntimeError, match="host_token_capacity=3456"):
        seq.validate_metadata("unit")
    seq.validate_metadata("unit", allow_terminal_capacity_gap=True)

    seq.eos_reached = False
    with pytest.raises(RuntimeError, match="host_token_capacity=3456"):
        seq.validate_metadata("unit", allow_terminal_capacity_gap=True)


def test_unified_trajectory_eviction_does_not_require_legacy_token_tensor():
    seq = SequenceEntry("seq", global_idx=24, prompt_length=100, max_decode_length=900)
    seq.status = SequenceStatus.EVICTED
    seq.assigned_rank = 1
    seq.prompt_length = 110
    seq.original_prompt_length = 100
    seq.decoded_length = 10
    seq.current_context_length = 110
    seq.total_decoded_before_eviction = 10
    seq.reentry_decoded_baseline = 0
    seq._buffer_slot = 3

    seq.validate_metadata("unit")


def test_synchronous_host_to_gpu_load_uses_dual_dsa_path():
    source = WORKER.read_text()
    start = source.index("\tdef _load_host_kv_to_gpu(")
    end = source.index("\n\tdef _release_gpu_kv_pages", start)
    body = source[start:end]

    dual_branch = body.index("if isinstance(manager, DualKVCacheCoordinator):")
    dual_prepare = body.index(
        "pointers = self._prepare_dual_kv_load_pointers(manager, global_sequence_ids)"
    )
    dual_launch = body.index("load_task = self._launch_dual_host_kv_load(pointers)")
    primary_only_call = body.index("k_ptrs, v_ptrs = manager.get_padded_3d_page_pointers()")

    assert dual_branch < dual_prepare < dual_launch < primary_only_call
    assert "async_load_layer_paged_kv_to_device_dual" not in body
    assert "host_paged_kv_worker_view_aux" not in body
