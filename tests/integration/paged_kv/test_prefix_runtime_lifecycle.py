import ctypes
import os
import uuid

from batchgen.models.engine_loader import core_engine as bg
from batchgen.prefix_reuse.commit import (
    build_prefix_commit_request,
    collect_group_pages_for_commit,
    release_evicted_prefix_pages,
    retain_inserted_prefix_pages,
)
from batchgen.prefix_reuse.config import (
    build_prefix_cache_runtime_config,
    create_host_prefix_cache_coordinator,
    unlink_prefix_cache_shared_memory,
)
from batchgen.prefix_reuse.prefill import lookup_prefix_cache_for_prefill


def _shm_unlink(name: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.shm_unlink(name.encode()) != 0 and ctypes.get_errno() != 2:
        raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))


def _host_config(shm_name: str):
    config = bg.HostPagedKVConfig()
    config.shm_name = shm_name
    config.num_layers = 1
    config.num_pages = 16
    config.page_size_tokens = 4
    config.num_k_heads = 1
    config.k_head_dim = 8
    config.num_v_heads = 1
    config.v_head_dim = 8
    config.k_element_size_bytes = 2
    config.v_element_size_bytes = 2
    config.sequence_table_capacity = 16
    return config


def test_python_runtime_commit_attach_and_evict_round_trip():
    host_shm = f"/prefix_runtime_host_{uuid.uuid4().hex}"
    host_config = _host_config(host_shm)
    runtime = build_prefix_cache_runtime_config(
        model_name="openai/gpt-oss-120b",
        kv_dtype="bfloat16",
        host_kv_config=host_config,
    )
    unlink_prefix_cache_shared_memory(runtime)
    host = bg.DefaultHostPagedKVWorkerView(host_config)
    owner = None
    worker = None
    try:
        host.initialize(0, True)
        owner = create_host_prefix_cache_coordinator(
            core_engine_module=bg,
            runtime_config=runtime,
            create_region=True,
        )
        worker = create_host_prefix_cache_coordinator(
            core_engine_module=bg,
            runtime_config=runtime,
            create_region=False,
        )

        source_sequence = 101
        tokens = list(range(12))
        host.register_sequences([source_sequence])
        host.allocate_pages_for_sequences([(source_sequence, len(tokens))])
        pages = collect_group_pages_for_commit(
            worker_views_by_group={0: host},
            sequence_id=source_sequence,
            commit_tokens=len(tokens),
            raw_page_tokens_by_group={0: 4},
        )
        request = build_prefix_commit_request(
            namespace_digest=runtime.namespace_digest,
            token_ids=tokens,
            publish_boundary_tokens=runtime.publish_boundary_tokens,
            pages_by_group=pages,
        )
        assert request is not None
        result = request.commit(worker)
        retained = retain_inserted_prefix_pages(
            commit_result=result,
            request=request,
            worker_views_by_group={0: host},
            sequence_id=source_sequence,
        )
        assert retained == {0: pages[0]}

        host.release_sequence_pages([source_sequence])
        assert host.get_stats().num_used_pages == 3

        lookup = lookup_prefix_cache_for_prefill(
            coordinator=worker,
            namespace_digest=runtime.namespace_digest,
            prompt_token_ids=[tokens],
            page_size_tokens=4,
        )
        target_sequence = 202
        host.register_sequences([target_sequence])
        shared_pages = [
            int(page.page_id)
            for page in lookup.lookup_results[0].materialization_spans[0].pages
        ]
        host.attach_shared_prefix_pages(target_sequence, shared_pages)
        private_pages = host.allocate_pages_for_sequences(
            [(target_sequence, 4)]
        )[0]
        assert host.build_page_table([target_sequence]) == [
            shared_pages + private_pages
        ]

        host.release_sequence_pages([target_sequence])
        worker.release_attachment(
            lookup.lookup_results[0].attachment_handle
        )
        assert host.get_stats().num_used_pages == 3

        eviction = owner.clear_unprotected()
        released = release_evicted_prefix_pages(
            eviction_result=eviction,
            worker_views_by_group={0: host},
        )
        assert released == {0: 3}
        assert host.get_stats().num_used_pages == 0
    finally:
        worker = None
        owner = None
        try:
            host.shutdown()
        except Exception:
            pass
        del host
        unlink_prefix_cache_shared_memory(runtime)
        _shm_unlink(host_shm)
