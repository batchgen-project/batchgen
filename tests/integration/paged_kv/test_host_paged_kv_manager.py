import ctypes
import gc
import math
import multiprocessing as mp
import os
import random
import string
import sys
import time

import pytest
import torch
from tqdm import tqdm

if sys.platform != "linux":
    pytest.skip(
        "host paged KV backend is Linux-only", allow_module_level=True
    )

try:
    from batchgen.models.engine_loader import core_engine as bg
except ImportError:  # compiled extension not built in this environment
    pytest.skip("core_engine extension is unavailable", allow_module_level=True)


def _random_shm_name() -> str:
    """A diagnostic label only: the region itself is an anonymous memfd."""
    suffix = "".join(
        random.choices(string.ascii_lowercase + string.digits, k=10)
    )
    return f"/batchgen_kv_{suffix}"


def _attach_identity(manager) -> tuple:
    """(creator pid, memfd fd) another process needs to map the same region."""
    fd = manager.memfd_fd()
    assert fd >= 0
    return os.getpid(), fd


def _make_deepseek_r1_config(shm_name: str) -> bg.HostPagedKVConfig:  # type: ignore
    cfg = bg.HostPagedKVConfig()
    cfg.shm_name = shm_name
    cfg.num_layers = 61
    cfg.num_pages = 10000
    cfg.page_size_tokens = 64
    cfg.num_k_heads = 1
    cfg.k_head_dim = 512 + 64
    cfg.num_v_heads = 0
    cfg.v_head_dim = 0
    cfg.k_element_size_bytes = 2
    cfg.v_element_size_bytes = 0
    cfg.sequence_table_capacity = 10240
    cfg.alignment_bytes = 64
    return cfg


def _make_tiny_mla_config(shm_name: str) -> bg.HostPagedKVConfig:  # type: ignore
    cfg = bg.HostPagedKVConfig()
    cfg.shm_name = shm_name
    cfg.num_layers = 1
    cfg.num_pages = 4
    cfg.page_size_tokens = 4
    cfg.num_k_heads = 1
    cfg.k_head_dim = 8
    cfg.num_v_heads = 0
    cfg.v_head_dim = 0
    cfg.k_element_size_bytes = 2
    cfg.v_element_size_bytes = 0
    cfg.sequence_table_capacity = 8
    cfg.alignment_bytes = 64
    return cfg


def test_creator_leaves_no_named_shm_object():
    shm_name = _random_shm_name()
    manager = bg.MLAHostPagedKVManager(_make_tiny_mla_config(shm_name))
    manager.initialize(True)
    _, fd = _attach_identity(manager)

    assert "memfd:batchgen_kv" in os.readlink(f"/proc/self/fd/{fd}")
    assert not os.path.exists(f"/dev/shm/{shm_name.lstrip('/')}")

    del manager
    gc.collect()

    # The kernel owns the lifetime: nothing survives the last mapping.
    with pytest.raises(OSError):
        os.fstat(fd)


def test_same_label_creators_get_independent_regions():
    """The label no longer names anything, so two creators cannot collide."""
    shm_name = _random_shm_name()
    first = bg.MLAHostPagedKVManager(_make_tiny_mla_config(shm_name))
    second = bg.MLAHostPagedKVManager(_make_tiny_mla_config(shm_name))

    try:
        first.initialize(True)
        second.initialize(True)

        assert first.memfd_fd() != second.memfd_fd()
        first.allocate_pages(1, 4)

        assert first.get_stats().num_free_pages == 3
        assert second.get_stats().num_free_pages == 4
    finally:
        del second
        del first


def test_distinct_regions_coexist():
    first = bg.MLAHostPagedKVManager(_make_tiny_mla_config(_random_shm_name()))
    second = bg.MLAHostPagedKVManager(_make_tiny_mla_config(_random_shm_name()))

    try:
        first.initialize(True)
        second.initialize(True)

        first_stats = first.get_stats()
        second_stats = second.get_stats()
        assert first_stats.num_total_pages == 4
        assert second_stats.num_total_pages == 4
        assert first_stats.num_free_pages == 4
        assert second_stats.num_free_pages == 4
    finally:
        del second
        del first


def _attach_and_report_stats(creator_pid, memfd_fd, result_queue):
    cfg = _make_tiny_mla_config(_random_shm_name())
    cfg.memfd_creator_pid = creator_pid
    cfg.memfd_fd = memfd_fd
    attached = bg.MLAHostPagedKVManager(cfg)
    attached.initialize(False)
    stats = attached.get_stats()
    result_queue.put((stats.num_total_pages, stats.num_free_pages))


def test_second_process_attaches_through_proc():
    manager = bg.MLAHostPagedKVManager(_make_tiny_mla_config(_random_shm_name()))
    try:
        manager.initialize(True)
        creator_pid, memfd_fd = _attach_identity(manager)
        manager.allocate_pages(7, 4)

        result_queue = mp.Queue()
        child = mp.Process(
            target=_attach_and_report_stats,
            args=(creator_pid, memfd_fd, result_queue),
        )
        child.start()
        total_pages, free_pages = result_queue.get(timeout=60)
        child.join()

        assert child.exitcode == 0
        # The attacher sees the creator's allocation, not a fresh region.
        assert (total_pages, free_pages) == (4, 3)
    finally:
        del manager


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA devices")
def test_prefill_batch_allocation_is_atomic_on_capacity_failure():
    """An over-capacity wave must not consume an earlier sequence's pages."""
    worker = bg.MLAHostPagedKVWorkerView(_make_tiny_mla_config(_random_shm_name()))
    try:
        worker.initialize(0, True)
        worker.register_sequences([101, 102])
        with pytest.raises(RuntimeError, match="Insufficient free pages.*prefill batch"):
            worker.allocate_pages_for_sequences([(101, 12), (102, 12)])

        stats = worker.get_stats()
        assert stats.num_free_pages == stats.num_total_pages == 4

        # The failed two-row transaction left both sequence IDs reusable.
        allocations = worker.allocate_pages_for_sequences([(101, 12)])
        assert [len(pages) for pages in allocations] == [3]
    finally:
        try:
            worker.release_sequence_pages([101])
            worker.unregister_sequences([101, 102])
        finally:
            del worker


def _make_attach_config(creator_pid, memfd_fd) -> bg.HostPagedKVConfig:  # type: ignore
    """Worker-side config: same layout, plus the creator's memfd identity."""
    cfg = _make_deepseek_r1_config(_random_shm_name())
    cfg.memfd_creator_pid = creator_pid
    cfg.memfd_fd = memfd_fd
    return cfg


# 每个 worker 进程里做的事：attach memfd + 分配自己的 sequences + 打印自己的 page table
def _worker_proc_alloc(creator_pid, memfd_fd, device_index, requests):
    cfg = _make_attach_config(creator_pid, memfd_fd)
    worker = bg.MLAHostPagedKVWorkerView(cfg)
    worker.initialize(device_index, False)  # 只附着 creator 的 memfd

    seq_ids = [sid for (sid, _) in requests]

    if requests:
        worker.register_sequences(seq_ids)
        allocations = worker.allocate_pages_for_sequences(requests)
        print(f"[worker {device_index}] requests len: {len(requests)}")
        print(
            f"[worker {device_index}] allocations (pages) len: {[len(pages) for pages in allocations]}"
        )
    else:
        print(f"[worker {device_index}] no requests, only attached shm")

    stats = worker.get_stats()
    print(f"[worker {device_index}] stats: {stats}")

    # worker 视角下的 page table
    page_table = worker.build_page_table(seq_ids)
    print(f"[worker {device_index}] page_table: {page_table}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA devices")
def test_parallel_worker_allocate_sequences():
    shm_name = _random_shm_name()
    cfg = _make_deepseek_r1_config(shm_name)
    manager = bg.MLAHostPagedKVManager(cfg)

    # 准备一批 sequence + 各自 token 数
    sequence_ids = [i for i in range(1, 2000)]
    lens = [
        cfg.page_size_tokens * ((i % 5) + 1) + (i % 64)
        for i in range(1, len(sequence_ids) + 1)
    ]
    requests = [(sid, length) for sid, length in zip(sequence_ids, lens)]

    num_workers = 8

    # 简单 round-robin 把 requests 分给不同 worker
    worker_requests = [[] for _ in range(num_workers)]
    for i, req in enumerate(requests):
        worker_requests[i % num_workers].append(req)

    try:
        # 1) 主进程创建并初始化共享内存区域（但不在主进程 allocate）
        manager.initialize(True)
        creator_pid, memfd_fd = _attach_identity(manager)

        # 2) 启动多个 worker 进程，并发地在各自进程内 allocate pages
        procs = []
        for i in range(num_workers):
            p = mp.Process(
                target=_worker_proc_alloc,
                args=(creator_pid, memfd_fd, i, worker_requests[i]),
            )
            p.start()
            procs.append(p)

        # 3) 主进程这边等待所有 worker 完成
        for p in procs:
            p.join()
            assert p.exitcode == 0

        # 4) 在主进程用 manager 视角检查最终分配情况
        stats = manager.get_stats()
        print(f"[manager] final stats: {stats}")

        # manager 视角下的总 page table
        manager_page_table = manager.build_page_table(sequence_ids)
        print(f"[manager] final page_table: {manager_page_table}")

        # 校验：统计所有 sequence 的 page 数，应等于 num_used_pages
        total_pages_from_table = sum(len(pages) for pages in manager_page_table)
        assert total_pages_from_table == stats.num_used_pages
        assert stats.num_active_sequences == len(sequence_ids)

        # 每个 sequence 的 page 数应与预期相符
        for sid, pages in zip(sequence_ids, manager_page_table):
            req_length = dict(requests)[sid]
            expected_page_count = math.ceil(req_length / cfg.page_size_tokens)
            assert len(pages) == expected_page_count

    finally:
        # 5) 清理：释放所有 sequence，memfd 随最后一个映射一起消失
        try:
            manager.free_sequences(sequence_ids)
            stats_after_free = manager.get_stats()
            print(f"[manager] stats_after_free: {stats_after_free}")
            assert stats_after_free.num_used_pages == 0
            assert stats_after_free.num_active_sequences == 0
        finally:
            del manager


def _worker_proc_copy_prefill(creator_pid, memfd_fd, device_index, requests):
    cfg = _make_attach_config(creator_pid, memfd_fd)
    worker = bg.MLAHostPagedKVWorkerView(cfg)
    worker.initialize(device_index, False)  # 只附着 creator 的 memfd

    seq_ids = [sid for (sid, _) in requests]

    if requests:
        worker.register_sequences(seq_ids)
        allocations = worker.allocate_pages_for_sequences(requests)
        print(f"[worker {device_index}] requests len: {len(requests)}")
        print(
            f"[worker {device_index}] allocations (pages) len: {[len(pages) for pages in allocations]}"
        )

    # initialize 一个 [requests_num, 200 * 64, cfg.k_head_num, cfg.k_head_dim] 的 tensor，模拟从 device 端 prefill KV
    requests_num = len(requests)
    sequence_len = 200 * 64
    # init 一个全是device_index的tensor
    device_tensor = torch.full(
        (requests_num, sequence_len, cfg.num_k_heads, cfg.k_head_dim),
        fill_value=float(device_index + 1),
        dtype=torch.bfloat16,
        device=f"cuda:{device_index}",
    )

    torch.cuda.synchronize(device_index)

    start = time.time()
    task = worker.async_offload_layer_kv_to_host(
        layer_idx=13,
        sequence_ids=seq_ids,
        k_tensor=device_tensor,
        v_tensor=None,
        sequence_lengths=[length for (_, length) in requests],
    )

    task.wait()
    end = time.time()
    print(
        f"[worker {device_index}] async D2H prefill done in {end - start:.7f} seconds"
    )


def _worker_proc_copy_decode(creator_pid, memfd_fd, device_index, requests):
    cfg = _make_attach_config(creator_pid, memfd_fd)
    worker = bg.MLAHostPagedKVWorkerView(cfg)
    worker.initialize(device_index, False)

    if not requests:
        return

    seq_ids = [sid for (sid, _, _) in requests]
    capacities = [
        (sid, capacity_tokens) for (sid, capacity_tokens, _) in requests
    ]
    sequence_lengths = [start_token for (_, _, start_token) in requests]

    worker.register_sequences(seq_ids)
    worker.allocate_pages_for_sequences(capacities)

    device_tensor = torch.empty(
        (len(seq_ids), 1, cfg.num_k_heads, cfg.k_head_dim),
        dtype=torch.bfloat16,
        device=f"cuda:{device_index}",
    )
    for idx, sequence_id in enumerate(seq_ids):
        device_tensor[idx].fill_(float(sequence_id + 1))

    torch.cuda.synchronize(device_index)

    start = time.time()

    task = worker.async_append_decode_kv_to_host(
        layer_idx=13,
        sequence_ids=seq_ids,
        k_tensor=device_tensor,
        v_tensor=None,
        sequence_lengths=sequence_lengths,
    )

    task.wait()

    end = time.time()
    print(
        f"[worker {device_index}] async decode D2H done in {end - start:.7f} seconds"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA devices")
def test_kv_copy_prefill_d2h():
    PAGE_NUM_PER_SEQ = 200
    NUM_WORKERS = 8

    shm_name = _random_shm_name()
    cfg = _make_deepseek_r1_config(shm_name)
    manager = bg.MLAHostPagedKVManager(cfg)

    # 准备一批 sequence + 各自 token 数
    sequence_ids = [i for i in range(0, NUM_WORKERS * 6)]
    lens = [cfg.page_size_tokens * PAGE_NUM_PER_SEQ for _ in sequence_ids]
    requests = [(sid, length) for sid, length in zip(sequence_ids, lens)]
    num_workers = NUM_WORKERS

    # 简单 round-robin 把 requests 分给不同 worker
    worker_requests = [[] for _ in range(num_workers)]
    for i, req in enumerate(requests):
        worker_requests[req[0] % num_workers].append(req)

    try:
        # 1) 主进程创建并初始化共享内存区域（但不在主进程 allocate）
        manager.initialize(True)
        creator_pid, memfd_fd = _attach_identity(manager)

        # 2) 启动多个 worker 进程，并发地在各自进程内 allocate pages
        procs = []
        for i in range(num_workers):
            p = mp.Process(
                target=_worker_proc_copy_prefill,
                args=(creator_pid, memfd_fd, i, worker_requests[i]),
            )
            p.start()
            procs.append(p)

        # 3) 主进程这边等待所有 worker 完成
        for p in procs:
            p.join()
            assert p.exitcode == 0

        # 4) 在主进程用 manager 视角检查最终分配情况
        stats = manager.get_stats()
        print(f"[manager] final stats: {stats}")

        # manager 视角下的 pointer
        for sid in tqdm(sequence_ids, desc="Verifying sequence data"):
            pointers = manager.get_sequence_layer_page_pointers(sid, 13)
            k_ptrs, v_ptrs = pointers
            sid_worker = sid % num_workers
            assert len(k_ptrs) == PAGE_NUM_PER_SEQ
            for i, ptr in enumerate(k_ptrs):
                # 通过指针读回 host 端的数据，检查内容是否正确
                array_type = ctypes.c_uint16 * (
                    64 * cfg.num_k_heads * cfg.k_head_dim
                )
                host_array = array_type.from_address(ptr)

                # zero-copy 转成 Torch tensor
                buf = torch.frombuffer(host_array, dtype=torch.uint16)
                bf16 = buf.view(torch.bfloat16)

                expected = float(sid_worker + 1)

                mask = bf16 != expected

                if torch.any(mask):
                    # mismatch 全部 index
                    bad_indices = torch.nonzero(mask, as_tuple=False).squeeze(
                        -1
                    )

                    # mismatch 对应的值
                    bad_values = bf16[mask]

                    # 只打印前 10 个，避免太长
                    N = min(10, bad_indices.numel())
                    msg_lines = []
                    for j in range(N):
                        idx = bad_indices[j].item()
                        val = bad_values[j].item()
                        msg_lines.append(
                            f"[idx={idx}] value={val} expected={expected}"
                        )

                    msg = "\n".join(msg_lines)
                    raise AssertionError(
                        f"Sequence {sid} mismatch (first {N} mismatches):\n{msg}\n"
                        f"ptr={hex(ptr)}\n"
                    )
        print("All sequences verified successfully.")

    finally:
        # 5) 清理：释放所有 sequence，memfd 随最后一个映射一起消失
        try:
            manager.free_sequences(sequence_ids)
            stats_after_free = manager.get_stats()
            print(f"[manager] stats_after_free: {stats_after_free}")
            assert stats_after_free.num_used_pages == 0
            assert stats_after_free.num_active_sequences == 0
        finally:
            del manager


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA devices")
def test_kv_copy_decode_d2h():
    PAGE_NUM_PER_SEQ = 10
    NUM_WORKERS = 8
    SEQ_NUM_PER_WORKER = 10
    shm_name = _random_shm_name()
    cfg = _make_deepseek_r1_config(shm_name)
    manager = bg.MLAHostPagedKVManager(cfg)

    sequence_ids = [i for i in range(0, NUM_WORKERS * SEQ_NUM_PER_WORKER)]
    capacity_tokens = cfg.page_size_tokens * PAGE_NUM_PER_SEQ
    max_start_token = capacity_tokens - 1
    decode_positions = {
        sid: (sid * 7) % max_start_token for sid in sequence_ids
    }
    print(f"decode_positions: {decode_positions}")

    worker_requests = [[] for _ in range(NUM_WORKERS)]
    for sid in sequence_ids:
        worker_idx = sid % NUM_WORKERS
        worker_requests[worker_idx].append(
            (sid, capacity_tokens, decode_positions[sid])
        )

    try:
        manager.initialize(True)
        creator_pid, memfd_fd = _attach_identity(manager)

        procs = []
        for worker_idx in range(NUM_WORKERS):
            p = mp.Process(
                target=_worker_proc_copy_decode,
                args=(
                    creator_pid,
                    memfd_fd,
                    worker_idx,
                    worker_requests[worker_idx],
                ),
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
            assert p.exitcode == 0

        token_elems = cfg.num_k_heads * cfg.k_head_dim
        token_bytes = token_elems * 2  # bfloat16

        for sid in sequence_ids:
            start_token = decode_positions[sid]
            page_idx = start_token // cfg.page_size_tokens
            page_offset = start_token % cfg.page_size_tokens
            k_ptrs, _ = manager.get_sequence_layer_page_pointers(sid, 13)
            assert len(k_ptrs) >= page_idx + 1

            page_ptr = k_ptrs[page_idx] + page_offset * token_bytes
            array_type = ctypes.c_uint16 * token_elems
            token_array = array_type.from_address(page_ptr)
            bf16_view = torch.frombuffer(token_array, dtype=torch.uint16).view(
                torch.bfloat16
            )
            # print(f"sid={sid}, start_token={start_token}, page_idx={page_idx}, page_offset={page_offset}, ptr={hex(page_ptr)}, bf16_view[0]={bf16_view[0].item()}")
            expected = torch.full_like(bf16_view, float(sid + 1))
            if not torch.equal(bf16_view, expected):
                raise AssertionError(
                    f"Decode token mismatch for sequence {sid}: "
                    f"expected {expected[0].item()}, got {bf16_view[0].item()}"
                )
        print("All decode tokens verified successfully.")

    finally:
        try:
            manager.free_sequences(sequence_ids)
        finally:
            del manager


if __name__ == "__main__":
    # test_mla_manager_disables_v_cache()
    mp.set_start_method("spawn", force=True)
    # test_batch_allocate_and_free_sequences()
    # test_parallel_worker_allocate_sequences()
    test_kv_copy_prefill_d2h()
    # test_kv_copy_decode_d2h()
