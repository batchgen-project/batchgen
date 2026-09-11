"""The host-KV copy streams must be ordered behind the caller's stream.

The worker runs the model on PyTorch's default stream, whose cudaStream_t
handle is 0. The view's copy streams come from the PyTorch pool and are
cudaStreamNonBlocking, so nothing orders them implicitly. The old
WaitForProducerStream returned early for a null handle, so:

  * a d2h offload issued right after the kernel that fills its source could
    copy the buffer's previous contents (Kimi-K3: the first sequence of every
    MLA layer's prefill offload; garbage after a re-configure -> NaN);
  * an h2d page load could land before the K-cache memset queued ahead of it
    and be zeroed.

Both tests make the producer deliberately slow so the race is deterministic.
"""

import ctypes
import errno
import random
import string

import pytest
import torch

_libc = ctypes.CDLL("libc.so.6", use_errno=True)

PAGE_TOKENS = 64
NUM_LAYERS = 2
NUM_PAGES = 64
NUM_K_HEADS = 1
K_HEAD_DIM = 8
K_ELEMENT_SIZE_BYTES = 2
ALIGNMENT_BYTES = 64

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="host-KV stream tests require CUDA"
)


@pytest.fixture(scope="module")
def bg():
    from batchgen.models.engine_loader import core_engine as bg_module

    return bg_module


def _random_shm_name() -> str:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
    return f"/batchgen_stream_order_{suffix}"


def _shm_unlink(name: str) -> None:
    res = _libc.shm_unlink(name.encode("utf-8"))
    if res != 0 and ctypes.get_errno() != errno.ENOENT:
        raise OSError(ctypes.get_errno(), f"shm_unlink({name}) failed")


def _make_config(bg, shm_name: str):
    cfg = bg.HostPagedKVConfig()
    cfg.shm_name = shm_name
    cfg.num_layers = NUM_LAYERS
    cfg.num_pages = NUM_PAGES
    cfg.page_size_tokens = PAGE_TOKENS
    cfg.num_k_heads = NUM_K_HEADS
    cfg.k_head_dim = K_HEAD_DIM
    cfg.num_v_heads = 0
    cfg.v_head_dim = 0
    cfg.k_element_size_bytes = K_ELEMENT_SIZE_BYTES
    cfg.v_element_size_bytes = 0
    cfg.sequence_table_capacity = 64
    cfg.alignment_bytes = ALIGNMENT_BYTES
    cfg.logger_name = "HostKVStreamOrderTest"
    return cfg


def _keep_default_stream_busy(device, iterations=48):
    """Queue ~hundreds of ms of work on the current (default) stream."""
    m = torch.randn(8192, 8192, dtype=torch.float16, device=device)
    for _ in range(iterations):
        m = m @ m
    return m  # keep the chain alive; never synchronised here


def _close(view, sequence_ids):
    if view is None:
        return
    try:
        view.release_sequence_pages(sequence_ids)
    except Exception:
        pass
    try:
        view.shutdown()
    except Exception:
        pass


def test_offload_copies_the_kernel_output_not_the_buffers_previous_contents(bg):
    shm_name = _random_shm_name()
    view = None
    sequence_ids = [7]
    tokens = PAGE_TOKENS * 2
    try:
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        view = bg.MLAHostPagedKVWorkerView(_make_config(bg, shm_name))
        view.initialize(0, True)
        view.register_sequences(sequence_ids)
        view.allocate_pages_for_sequences([(sequence_ids[0], tokens)])

        src = torch.full(
            (1, tokens, NUM_K_HEADS, K_HEAD_DIM), -1.0,
            dtype=torch.bfloat16, device=device,
        )
        torch.cuda.synchronize()
        busy = _keep_default_stream_busy(device)
        # The fill is queued behind the busy chain on the default stream. The
        # offload is issued immediately after; its d2h copy must not run until
        # the fill has executed.
        src.fill_(7.0)
        task = view.async_offload_layer_kv_to_host(
            0, sequence_ids, src, None, [tokens]
        )
        task.wait()
        k_cpu, _ = view.read_sequence_kv_to_cpu(sequence_ids[0])
        torch.cuda.synchronize()
        del busy
        got = k_cpu[0].reshape(-1, NUM_K_HEADS * K_HEAD_DIM)[:tokens].float()
        assert torch.equal(got, torch.full_like(got, 7.0)), (
            "offload copied stale source contents: "
            f"{(got != 7.0).sum().item()} of {got.numel()} elements differ"
        )
        view.release_sequence_pages(sequence_ids)
        view.shutdown()
        view = None
    finally:
        _close(view, sequence_ids)
        _shm_unlink(shm_name)


def test_page_load_lands_after_work_already_queued_on_the_callers_stream(bg):
    shm_name = _random_shm_name()
    view = None
    sequence_ids = [9]
    n_pages = 2
    tokens = PAGE_TOKENS * n_pages
    try:
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        view = bg.MLAHostPagedKVWorkerView(_make_config(bg, shm_name))
        view.initialize(0, True)
        view.register_sequences(sequence_ids)
        view.allocate_pages_for_sequences([(sequence_ids[0], tokens)])

        host_k = torch.full(
            (NUM_LAYERS, n_pages, PAGE_TOKENS, NUM_K_HEADS, K_HEAD_DIM), 5.0,
            dtype=torch.bfloat16,
        )
        view.write_sequence_kv_from_cpu(sequence_ids[0], host_k, None)

        dst = torch.full(
            (NUM_LAYERS, n_pages, PAGE_TOKENS, NUM_K_HEADS, K_HEAD_DIM), 3.0,
            dtype=torch.bfloat16, device=device,
        )
        k_ptrs = torch.tensor(
            [[[dst[layer, page].data_ptr() for page in range(n_pages)]]
             for layer in range(NUM_LAYERS)],
            dtype=torch.int64,
        )  # [num_layers, num_sequences, max_pages]
        torch.cuda.synchronize()
        busy = _keep_default_stream_busy(device)
        # A memset queued on the default stream before the load is issued
        # (the GPU K cache is torch.zeros'ed right before the first load of
        # every decode phase). The loaded pages must survive it.
        dst.zero_()
        task = view.async_load_layer_paged_kv_to_device(
            torch.tensor(sequence_ids, dtype=torch.int64),
            torch.tensor([n_pages], dtype=torch.int64),
            k_ptrs,
            None,
        )
        task.wait()
        torch.cuda.synchronize()
        del busy
        got = dst.float()
        assert torch.equal(got, torch.full_like(got, 5.0)), (
            "page load was overtaken by the memset queued ahead of it: "
            f"{(got != 5.0).sum().item()} of {got.numel()} elements differ"
        )
        view.release_sequence_pages(sequence_ids)
        view.shutdown()
        view = None
    finally:
        _close(view, sequence_ids)
        _shm_unlink(shm_name)
