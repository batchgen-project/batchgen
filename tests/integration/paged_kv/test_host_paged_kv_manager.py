import ctypes
import errno
import math
import multiprocessing as mp
import os
import random
import string
import time
from concurrent.futures import ProcessPoolExecutor

import torch
from tqdm import tqdm

from batchgen.models.engine_loader import core_engine as bg

_libc = ctypes.CDLL("libc.so.6", use_errno=True)


def _random_shm_name() -> str:
    suffix = "".join(
        random.choices(string.ascii_lowercase + string.digits, k=10)
    )
    return f"/batchgen_kv_{suffix}"


def _shm_unlink(name: str) -> None:
    if not name:
        return
    res = _libc.shm_unlink(name.encode("utf-8"))
    if res != 0:
        err = ctypes.get_errno()
        if err != errno.ENOENT:
            raise OSError(err, f"shm_unlink({name}) failed")


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


# 每个 worker 进程里做的事：attach shm + 分配自己的 sequences + 打印自己的 page table
def _worker_proc_alloc(shm_name, device_index, requests):
    # 每个进程里重新构造 cfg，shm_name 必须一致
    cfg = _make_deepseek_r1_config(shm_name)
    worker = bg.MLAHostPagedKVWorkerView(cfg)
    worker.initialize(device_index, False)  # 只附着已有的 shared memory

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

        # 2) 启动多个 worker 进程，并发地在各自进程内 allocate pages
        procs = []
        for i in range(num_workers):
            p = mp.Process(
                target=_worker_proc_alloc,
                args=(shm_name, i, worker_requests[i]),
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
        # 5) 清理：释放所有 sequence，删除 shm
        try:
            manager.free_sequences(sequence_ids)
            stats_after_free = manager.get_stats()
            print(f"[manager] stats_after_free: {stats_after_free}")
            assert stats_after_free.num_used_pages == 0
            assert stats_after_free.num_active_sequences == 0
        finally:
            del manager
            _shm_unlink(shm_name)


def test_worker_view_attaches_shared_prefix_pages_without_owning_them():
    shm_name = _random_shm_name()
    cfg = _make_deepseek_r1_config(shm_name)
    cfg.num_pages = 32
    worker = bg.MLAHostPagedKVWorkerView(cfg)

    try:
        worker.initialize(0, True)
        source_seq = 101
        target_seq = 202
        worker.register_sequences([source_seq, target_seq])

        shared_pages = worker.allocate_pages_for_sequences(
            [(source_seq, cfg.page_size_tokens * 2)]
        )[0]
        worker.attach_shared_prefix_pages(target_seq, shared_pages)
        private_pages = worker.allocate_pages_for_sequences(
            [(target_seq, cfg.page_size_tokens)]
        )[0]

        assert worker.build_page_table([target_seq]) == [
            shared_pages + private_pages
        ]

        before_release = worker.get_stats()
        worker.release_sequence_pages([target_seq])
        after_release = worker.get_stats()

        assert (
            after_release.num_used_pages
            == before_release.num_used_pages - len(private_pages)
        )
        assert worker.build_page_table([source_seq]) == [shared_pages]
    finally:
        for sequence_id in (202, 101):
            try:
                worker.release_sequence_pages([sequence_id])
            except Exception:
                pass
        try:
            worker.shutdown()
        except Exception:
            pass
        del worker
        _shm_unlink(shm_name)


def test_worker_view_retains_prefix_resident_pages_until_eviction_release():
    shm_name = _random_shm_name()
    cfg = _make_deepseek_r1_config(shm_name)
    cfg.num_pages = 16
    worker = bg.MLAHostPagedKVWorkerView(cfg)

    try:
        worker.initialize(0, True)
        sequence_id = 303
        worker.register_sequences([sequence_id])

        pages = worker.allocate_pages_for_sequences(
            [(sequence_id, cfg.page_size_tokens * 3)]
        )[0]
        retained = worker.retain_sequence_prefix_pages(sequence_id, 2)

        assert retained == pages[:2]
        assert worker.build_page_table([sequence_id]) == [pages]

        before_release = worker.get_stats()
        worker.release_sequence_pages([sequence_id])
        after_sequence_release = worker.get_stats()

        assert (
            after_sequence_release.num_used_pages
            == before_release.num_used_pages - 1
        )
        assert after_sequence_release.num_active_sequences == 0

        worker.release_resident_pages(retained)
        after_eviction_release = worker.get_stats()

        assert after_eviction_release.num_used_pages == 0

        grow_sequence_id = 404
        worker.register_sequences([grow_sequence_id])
        prefix_pages = worker.allocate_pages_for_sequences(
            [(grow_sequence_id, cfg.page_size_tokens * 2)]
        )[0]
        retained_prefix = worker.retain_sequence_prefix_pages(
            grow_sequence_id, 2
        )
        grown_pages = worker.grow_sequence_pages(grow_sequence_id, 1)

        assert retained_prefix == prefix_pages
        assert worker.build_page_table([grow_sequence_id]) == [
            prefix_pages + grown_pages
        ]

        before_grow_release = worker.get_stats()
        worker.release_sequence_pages([grow_sequence_id])
        after_grow_sequence_release = worker.get_stats()

        assert (
            after_grow_sequence_release.num_used_pages
            == before_grow_release.num_used_pages - len(grown_pages)
        )
        worker.release_resident_pages(retained_prefix)
        assert worker.get_stats().num_used_pages == 0

        range_sequence_id = 505
        worker.register_sequences([range_sequence_id])
        range_pages = worker.allocate_pages_for_sequences(
            [(range_sequence_id, cfg.page_size_tokens * 4)]
        )[0]
        retained_range = worker.retain_sequence_page_range(
            range_sequence_id, 2, 2
        )

        assert retained_range == range_pages[2:4]
        assert worker.build_page_table([range_sequence_id]) == [range_pages]

        before_range_release = worker.get_stats()
        worker.release_sequence_pages([range_sequence_id])
        after_range_sequence_release = worker.get_stats()

        assert (
            after_range_sequence_release.num_used_pages
            == before_range_release.num_used_pages - 2
        )
        worker.release_resident_pages(retained_range)
        assert worker.get_stats().num_used_pages == 0
    finally:
        try:
            worker.shutdown()
        except Exception:
            pass
        del worker
        _shm_unlink(shm_name)


def _worker_proc_copy_prefill(shm_name, device_index, requests):
    # 每个进程里重新构造 cfg，shm_name 必须一致
    cfg = _make_deepseek_r1_config(shm_name)
    worker = bg.MLAHostPagedKVWorkerView(cfg)
    worker.initialize(device_index, False)  # 只附着已有的 shared memory

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


def _worker_proc_copy_decode(shm_name, device_index, requests):
    cfg = _make_deepseek_r1_config(shm_name)
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

        # 2) 启动多个 worker 进程，并发地在各自进程内 allocate pages
        procs = []
        for i in range(num_workers):
            p = mp.Process(
                target=_worker_proc_copy_prefill,
                args=(shm_name, i, worker_requests[i]),
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
        # 5) 清理：释放所有 sequence，删除 shm
        try:
            manager.free_sequences(sequence_ids)
            stats_after_free = manager.get_stats()
            print(f"[manager] stats_after_free: {stats_after_free}")
            assert stats_after_free.num_used_pages == 0
            assert stats_after_free.num_active_sequences == 0
        finally:
            del manager
            _shm_unlink(shm_name)


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

        procs = []
        for worker_idx in range(NUM_WORKERS):
            p = mp.Process(
                target=_worker_proc_copy_decode,
                args=(shm_name, worker_idx, worker_requests[worker_idx]),
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
            _shm_unlink(shm_name)


if __name__ == "__main__":
    # test_mla_manager_disables_v_cache()
    mp.set_start_method("spawn", force=True)
    # test_batch_allocate_and_free_sequences()
    # test_parallel_worker_allocate_sequences()
    test_kv_copy_prefill_d2h()
    # test_kv_copy_decode_d2h()
