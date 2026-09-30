"""Persistent QueryBook buffers must remain writable across inference phases."""

import gc
import os

import pytest
import torch

from batchgen.batchgen_worker import QueryBookBufferPool, allocate_node_shared_int64


def test_recycled_slot_after_pool_creation_in_inference_mode():
    with torch.inference_mode():
        pool = QueryBookBufferPool(2, 8, 8, pad_token_id=-1)
        slot = pool.allocate_slot()
        pool.decoded_tokens_buffer[slot, :] = 7
        pool.input_ids_buffer[slot, :] = 9
        pool.free_slot(slot)

    assert pool.allocate_slot() == slot
    assert torch.all(pool.decoded_tokens_buffer[slot, :] == -1)
    assert torch.all(pool.input_ids_buffer[slot, :] == 0)
    assert not pool.decoded_tokens_buffer.is_inference()
    assert not pool.input_ids_buffer.is_inference()


@pytest.mark.skipif(
    not hasattr(os, "memfd_create"), reason="memfd_create requires Linux"
)
def test_shared_input_buffer_created_in_inference_mode_is_later_writable():
    with torch.inference_mode():
        input_ids, memfd = allocate_node_shared_int64(
            "batchgen_querybook_test",
            rows=2,
            width=8,
            is_creator=True,
            exchange=lambda pid, fd: (pid, fd),
            barrier=lambda: None,
        )
    try:
        input_ids[0, 0] = 7
        assert input_ids[0, 0].item() == 7
        assert not input_ids.is_inference()
    finally:
        del input_ids
        gc.collect()
        memfd.release_fd()
