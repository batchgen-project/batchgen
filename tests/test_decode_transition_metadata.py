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
        "\t\t\t\tseq.host_token_capacity = seq.host_pages_allocated * seq.PAGE_SIZE\n"
        "\t\t\t\tsequence_tokens.append(seq.host_token_capacity)"
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
