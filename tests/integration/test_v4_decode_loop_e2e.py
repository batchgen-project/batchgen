"""Thin end-to-end harness for the V4 worker decode wiring.

Verifies the cheap, high-value integration milestones WITHOUT a full model
launch (per the harness-first decision): the worker KV-init branch builds and
binds the real DeepSeekV4KVCoordinator, and the decode backend can be injected
into real DeepSeekV4FlashAttnWrapper instances. The real v4flash_mp4_fp8 launch
on H20 remains the ground truth for the full decode loop.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from batchgen.batchgen_worker import BatchGenWorker
from batchgen.kv_cache.deepseek_v4_kv_coordinator import DeepSeekV4KVCoordinator

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for V4 decode wiring"
)


def _make_worker_shell(compress_ratios, num_hidden_layers):
    worker = object.__new__(BatchGenWorker)
    worker.rank = 0
    worker.global_rank = 0
    worker.local_rank = 0
    worker.world_size = 1
    worker.huggingface_ckpt_name = "deepseek-v4-flash"
    worker.gpu_kv_cache_size_gb = 2.0
    worker.gpu_paged_kv_cache_manager = None
    worker.model_config = SimpleNamespace(
        num_hidden_layers=num_hidden_layers,
        compress_ratios=list(compress_ratios),
        num_attention_heads=64,
        head_dim=512,
        qk_rope_head_dim=64,
    )
    worker.loaded_model_config = None
    worker.core_engine = SimpleNamespace(
        gpu_paged_kv_manager=None, gpu_paged_kv_manager_aux=None
    )
    return worker


def test_kv_init_branch_builds_and_binds_v4_coordinator():
    worker = _make_worker_shell([0, 4, 128], 3)
    manager = worker._initialize_gpu_kv_manager_fixed_size()
    try:
        assert isinstance(manager, DeepSeekV4KVCoordinator)
        assert manager.compress_ratios == [0, 4, 128]
        assert worker.gpu_paged_kv_cache_manager is manager
        assert worker.core_engine.gpu_paged_kv_manager is manager
        assert worker.core_engine.gpu_paged_kv_manager_aux is None
        assert worker._is_deepseek_v4_kv_manager(manager) is True
    finally:
        manager.destroy(empty_cuda_cache=True)


def test_v4_coordinator_duck_methods_present():
    worker = _make_worker_shell([0, 4, 128], 3)
    manager = worker._initialize_gpu_kv_manager_fixed_size()
    try:
        manager.allocate_pages_for_sequences([7], [256])
        tables = manager.rebuild_page_table([7])
        assert set(tables.keys()) == {"swa", "c4", "c128", "indexer"}
        manager.clear_page_table()
        added = manager.extend_pages_for_sequence(7, 512)
        assert added >= 0
        manager.free_pages_for_sequences([7])
    finally:
        manager.destroy(empty_cuda_cache=True)


def test_install_decode_backend_injects_into_wrappers():
    from batchgen.models.deepseek.deepseekv4_flash.wrappers import (
        DeepSeekV4FlashAttnWrapper,
    )

    worker = _make_worker_shell([0, 4, 128], 3)
    manager = worker._initialize_gpu_kv_manager_fixed_size()
    try:
        layers = []
        for layer_idx in range(3):
            wrapper = object.__new__(DeepSeekV4FlashAttnWrapper)
            wrapper.layer_idx = layer_idx
            wrapper._v4_backend = None
            wrapper._layer_config = None
            layers.append(SimpleNamespace(self_attn=wrapper))
        worker.model = SimpleNamespace(model=SimpleNamespace(layers=layers))

        worker._install_deepseek_v4_decode_backend()

        backend = worker._deepseek_v4_decode_backend
        for layer_idx, layer in enumerate(layers):
            wrapper = layer.self_attn
            assert wrapper._v4_backend is backend
            assert wrapper._layer_config is backend.layer_configs[layer_idx]
            assert (
                wrapper._layer_config.compress_ratio == [0, 4, 128][layer_idx]
            )
    finally:
        manager.destroy(empty_cuda_cache=True)


def test_decode_metadata_hook_initializes_backend_metadata():
    from batchgen.models.deepseek.deepseekv4_flash.wrappers import (
        DeepSeekV4FlashAttnWrapper,
    )
    from batchgen.models.wrappers import AttnWrapperBase

    worker = _make_worker_shell([0, 4, 128], 3)
    worker.torch_device = torch.device("cuda:0")
    worker.model_context_length = 4096
    worker.model_config.rope_theta = 10000.0
    worker.model_config.max_position_embeddings = 8192
    manager = worker._initialize_gpu_kv_manager_fixed_size()
    try:
        layers = []
        for layer_idx in range(3):
            wrapper = object.__new__(DeepSeekV4FlashAttnWrapper)
            wrapper.layer_idx = layer_idx
            wrapper._v4_backend = None
            wrapper._layer_config = None
            layers.append(SimpleNamespace(self_attn=wrapper))
        worker.model = SimpleNamespace(model=SimpleNamespace(layers=layers))
        worker._install_deepseek_v4_decode_backend()
        backend = worker._deepseek_v4_decode_backend

        seq_id = 5
        manager.allocate_pages_for_sequences([seq_id], [256])
        prev_cur = AttnWrapperBase.cur_batch
        prev_seq = AttnWrapperBase.cache_seqlens
        prev_pos = AttnWrapperBase.position_ids
        try:
            AttnWrapperBase.cur_batch = [seq_id]
            AttnWrapperBase.cache_seqlens = torch.tensor(
                [200], dtype=torch.int32, device="cuda"
            )
            AttnWrapperBase.position_ids = torch.tensor(
                [199], dtype=torch.int32, device="cuda"
            )
            worker._prepare_deepseek_v4_decode_metadata_for_forward(manager)
            meta = backend.metadata
            assert meta is not None
            assert int(meta.seq_lens_casual[0]) == 200
        finally:
            AttnWrapperBase.cur_batch = prev_cur
            AttnWrapperBase.cache_seqlens = prev_seq
            AttnWrapperBase.position_ids = prev_pos
    finally:
        manager.destroy(empty_cuda_cache=True)


def test_decode_metadata_hook_clears_on_empty_batch():
    from batchgen.models.deepseek.deepseekv4_flash.wrappers import (
        DeepSeekV4FlashAttnWrapper,
    )
    from batchgen.models.wrappers import AttnWrapperBase

    worker = _make_worker_shell([0, 4, 128], 3)
    worker.torch_device = torch.device("cuda:0")
    worker.model_context_length = 4096
    manager = worker._initialize_gpu_kv_manager_fixed_size()
    try:
        layers = [
            SimpleNamespace(
                self_attn=object.__new__(DeepSeekV4FlashAttnWrapper)
            )
        ]
        for layer in layers:
            layer.self_attn.layer_idx = 0
            layer.self_attn._v4_backend = None
            layer.self_attn._layer_config = None
        worker.model = SimpleNamespace(model=SimpleNamespace(layers=layers))
        worker._install_deepseek_v4_decode_backend()
        prev_cur = AttnWrapperBase.cur_batch
        try:
            AttnWrapperBase.cur_batch = []
            worker._prepare_deepseek_v4_decode_metadata_for_forward(manager)
            assert worker._deepseek_v4_decode_backend._metadata is None
        finally:
            AttnWrapperBase.cur_batch = prev_cur
    finally:
        manager.destroy(empty_cuda_cache=True)


def test_v4_prefill_populate_writes_swa_for_dense_layer():
    from batchgen.models.deepseek.deepseekv4_flash.wrappers import (
        DeepSeekV4FlashAttnWrapper,
    )
    from batchgen.models.wrappers import AttnWrapperBase
    from batchgen.attention.v4_backend import DSV4LayerConfig

    worker = _make_worker_shell([0, 4, 128], 3)
    worker.torch_device = torch.device("cuda:0")
    manager = worker._initialize_gpu_kv_manager_fixed_size()
    try:
        wrapper = object.__new__(DeepSeekV4FlashAttnWrapper)
        wrapper.layer_idx = 0
        wrapper.core_engine = worker.core_engine
        wrapper.model_config = worker.model_config
        wrapper.model_config.max_position_embeddings = 4096
        wrapper.model_config.rope_theta = 10000.0
        wrapper.model_config.qk_rope_head_dim = 64
        wrapper._layer_config = DSV4LayerConfig(
            layer_idx=0,
            compress_ratio=0,
            n_heads=64,
            head_dim=512,
            rope_head_dim=64,
        )

        seq_len = 200
        prefill_kv = torch.randn(
            1, seq_len, 512, device="cuda", dtype=torch.bfloat16
        )
        attn_mask = torch.ones(1, seq_len, device="cuda")
        prev_cur = AttnWrapperBase.cur_batch
        prev_mask = AttnWrapperBase.attention_mask
        try:
            AttnWrapperBase.cur_batch = [3]
            AttnWrapperBase.attention_mask = attn_mask
            assert wrapper._is_v4_resident_prefill() is True
            wrapper._populate_v4_prefill_kv(prefill_kv, attn_mask)
            stored = manager.swa.debug_read_kv(
                layer_idx=0,
                token_slots=manager.swa.sequence_token_slots(
                    3, torch.arange(seq_len, device="cuda", dtype=torch.long)
                ),
            )
            assert stored.shape[0] == seq_len
            assert torch.isfinite(stored.float()).all()
        finally:
            AttnWrapperBase.cur_batch = prev_cur
            AttnWrapperBase.attention_mask = prev_mask
    finally:
        manager.destroy(empty_cuda_cache=True)


def test_runtime_kernel_compressor_bridges_weights_and_fail_fast():
    from batchgen.models.deepseek.deepseekv4_flash.wrappers import (
        DeepSeekV4FlashAttnWrapper,
    )
    from batchgen.models.deepseek.deepseekv4_flash.model import (
        DeepSeekV4FlashCompressor,
    )

    wrapper = object.__new__(DeepSeekV4FlashAttnWrapper)
    wrapper.layer_idx = 1

    src = DeepSeekV4FlashCompressor(512, 512, 64, 4, 1e-6, overlap=True).cuda()

    import pytest as _pytest

    with _pytest.raises(RuntimeError):
        wrapper._runtime_kernel_compressor(src, rotate=True)

    tensors = {
        "ape": torch.randn_like(src.ape),
        "norm.weight": torch.randn_like(src.norm.weight),
        "wkv.weight": torch.randn(1024, 512, device="cuda"),
        "wgate.weight": torch.randn(1024, 512, device="cuda"),
    }
    src.ape.data = tensors["ape"]
    src.norm.weight.data = tensors["norm.weight"]
    src.wkv.set_runtime_tensors(tensors, "wkv")
    src.wgate.set_runtime_tensors(tensors, "wgate")

    comp = wrapper._runtime_kernel_compressor(src, rotate=True)
    assert comp.rotate is True
    assert comp.overlap is True
    assert torch.equal(comp.wkv_weight.data, tensors["wkv.weight"])
    assert torch.equal(comp.ape.data, tensors["ape"])
    comp2 = wrapper._runtime_kernel_compressor(src, rotate=True)
    assert comp2 is comp


def test_v4_c4_indexer_inputs_reads_pool_and_shapes():
    from batchgen.models.deepseek.deepseekv4_flash.wrappers import (
        DeepSeekV4FlashAttnWrapper,
    )
    from batchgen.models.deepseek.deepseekv4_flash.model import (
        DeepSeekV4FlashIndexer,
    )
    from batchgen.attention.v4_backend import (
        DeepseekV4AttnBackend,
        build_layer_configs_from_compress_ratios,
    )
    from batchgen.attention.dsa.v4_flashmla_adapter import (
        DeepSeekV4FlashMLADecodeAdapter,
        build_v4_decode_attn_metadata,
    )

    if torch.cuda.get_device_capability()[0] >= 12:
        pytest.importorskip(
            "tilelang", reason="sm120 indexer fp4_act_quant needs tilelang"
        )

    cfg = SimpleNamespace(
        hidden_size=512,
        q_lora_rank=128,
        index_head_dim=128,
        index_n_heads=64,
        index_topk=512,
        qk_rope_head_dim=64,
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        compress_rope_theta=160000.0,
        rope_scaling={},
        num_attention_heads=64,
        head_dim=512,
    )
    worker = _make_worker_shell([0, 4, 128], 3)
    worker.torch_device = torch.device("cuda:0")
    manager = worker._initialize_gpu_kv_manager_fixed_size()
    try:
        layer_idx = 1
        route = manager.get_layer_routing(layer_idx)
        seq_id, seq_len = 9, 256
        clen = seq_len // 4
        manager.allocate_pages_for_sequences([seq_id], [seq_len])
        cpos = torch.arange(clen, device="cuda", dtype=torch.long)
        idx_k = torch.randn(
            clen, cfg.index_head_dim, device="cuda", dtype=torch.bfloat16
        ).div_(10)
        slots = manager.indexer.sequence_token_slots(seq_id, cpos)
        manager.indexer.store_indexer(
            layer_idx=route.indexer_layer_idx, token_slots=slots, index_k=idx_k
        )

        layer_configs = build_layer_configs_from_compress_ratios(
            [0, 4, 128], n_heads=64, head_dim=512, rope_head_dim=64
        )
        backend = DeepseekV4AttnBackend(
            layer_configs=layer_configs,
            page_size=manager.swa.page_size_tokens,
            flashmla_backend=DeepSeekV4FlashMLADecodeAdapter(manager),
        )
        metadata = build_v4_decode_attn_metadata(
            coordinator=manager,
            sequence_ids=[seq_id],
            cache_seqlens=torch.tensor(
                [seq_len], dtype=torch.int32, device="cuda"
            ),
            positions=torch.tensor(
                [seq_len - 1], dtype=torch.int32, device="cuda"
            ),
        )
        backend.init_metadata(metadata)

        indexer = DeepSeekV4FlashIndexer(cfg, 4).cuda()
        tensors = {
            "wq_b.weight": torch.randn(64 * 128, 128, device="cuda"),
            "weights_proj.weight": torch.randn(64, 512, device="cuda"),
        }
        indexer.wq_b.set_runtime_tensors(tensors, "wq_b")
        indexer.weights_proj.set_runtime_tensors(tensors, "weights_proj")

        wrapper = object.__new__(DeepSeekV4FlashAttnWrapper)
        wrapper.layer_idx = layer_idx
        wrapper.model_config = cfg
        wrapper._v4_backend = backend
        wrapper.module = SimpleNamespace(indexer=indexer, world_size=1)

        q_low = torch.randn(1, 128, device="cuda", dtype=torch.bfloat16)
        hidden = torch.randn(1, 512, device="cuda", dtype=torch.bfloat16)
        index_q, index_k, head_gates = wrapper._v4_c4_indexer_inputs(
            q_low, hidden
        )

        assert index_q.shape == (1, 64, 128)
        assert index_k.shape == (1, clen, 128)
        assert head_gates.shape == (1, 64)
        assert torch.allclose(
            index_k[0], idx_k.to(torch.bfloat16), atol=0.05, rtol=0
        )
        assert torch.isfinite(index_q).all()
    finally:
        manager.destroy(empty_cuda_cache=True)


def test_v4_c4_prefill_populates_indexer_and_c4_pools():
    from batchgen.models.deepseek.deepseekv4_flash.wrappers import (
        DeepSeekV4FlashAttnWrapper,
    )
    from batchgen.models.deepseek.deepseekv4_flash.model import (
        DeepSeekV4FlashCompressor,
        DeepSeekV4FlashIndexer,
    )
    from batchgen.models.wrappers import AttnWrapperBase
    from batchgen.attention.v4_backend import DSV4LayerConfig

    if torch.cuda.get_device_capability()[0] >= 12:
        pytest.importorskip(
            "tilelang", reason="sm120 indexer fp4_act_quant needs tilelang"
        )

    cfg = SimpleNamespace(
        hidden_size=512,
        q_lora_rank=128,
        index_head_dim=128,
        index_n_heads=64,
        index_topk=512,
        qk_rope_head_dim=64,
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        compress_rope_theta=160000.0,
        rope_scaling={},
        rope_theta=10000.0,
        num_attention_heads=64,
        head_dim=512,
    )
    worker = _make_worker_shell([0, 4, 128], 3)
    worker.torch_device = torch.device("cuda:0")
    manager = worker._initialize_gpu_kv_manager_fixed_size()
    try:
        layer_idx = 1
        route = manager.get_layer_routing(layer_idx)

        main_comp = DeepSeekV4FlashCompressor(
            512, 512, 64, 4, 1e-6, overlap=True
        ).cuda()
        indexer = DeepSeekV4FlashIndexer(cfg, 4).cuda()
        for comp in (main_comp, indexer.compressor):
            out_dim = (2 if comp.overlap else 1) * comp.head_dim
            comp.ape.data = torch.randn_like(comp.ape)
            comp.norm.weight.data = torch.randn_like(comp.norm.weight)
            t = {
                "wkv.weight": torch.randn(out_dim, 512, device="cuda"),
                "wgate.weight": torch.randn(out_dim, 512, device="cuda"),
            }
            comp.wkv.set_runtime_tensors(t, "wkv")
            comp.wgate.set_runtime_tensors(t, "wgate")
        module = SimpleNamespace(
            compressor=main_comp, indexer=indexer, world_size=1
        )

        wrapper = object.__new__(DeepSeekV4FlashAttnWrapper)
        wrapper.layer_idx = layer_idx
        wrapper.model_config = cfg
        wrapper.core_engine = worker.core_engine
        wrapper.module = module
        wrapper._layer_config = DSV4LayerConfig(
            layer_idx=layer_idx,
            compress_ratio=4,
            n_heads=64,
            head_dim=512,
            rope_head_dim=64,
        )

        seq_len = 256
        seq_id = 4
        prefill_kv = torch.randn(
            1, seq_len, 512, device="cuda", dtype=torch.bfloat16
        )
        hidden = torch.randn(
            1, seq_len, 512, device="cuda", dtype=torch.bfloat16
        )
        attn_mask = torch.ones(1, seq_len, device="cuda")
        prev_cur, prev_mask = (
            AttnWrapperBase.cur_batch,
            AttnWrapperBase.attention_mask,
        )
        try:
            AttnWrapperBase.cur_batch = [seq_id]
            AttnWrapperBase.attention_mask = attn_mask
            wrapper._populate_v4_prefill_kv(prefill_kv, attn_mask, hidden)

            clen = seq_len // 4
            cpos = torch.arange(clen, device="cuda", dtype=torch.long)
            idx_slots = manager.indexer.sequence_token_slots(seq_id, cpos)
            idx_k = manager.indexer.debug_read_indexer(
                layer_idx=route.indexer_layer_idx, token_slots=idx_slots
            )
            assert idx_k.shape == (clen, 128)
            assert torch.isfinite(idx_k.float()).all()
            assert idx_k.abs().sum() > 0
        finally:
            AttnWrapperBase.cur_batch = prev_cur
            AttnWrapperBase.attention_mask = prev_mask
    finally:
        manager.destroy(empty_cuda_cache=True)


def test_v4_c128_decode_emission_stores_compressed_token():
    from batchgen.models.deepseek.deepseekv4_flash.model import (
        DeepSeekV4FlashCompressor,
    )
    from batchgen.attention.dsa.v4_flashmla_adapter import (
        DeepSeekV4FlashMLADecodeAdapter,
        build_v4_compress_cos_sin_cache,
    )

    worker = _make_worker_shell([0, 4, 128], 3)
    worker.torch_device = torch.device("cuda:0")
    manager = worker._initialize_gpu_kv_manager_fixed_size()
    try:
        layer_idx = 2
        route = manager.get_layer_routing(layer_idx)
        seq_id = 6
        manager.allocate_pages_for_sequences([seq_id], [256])

        comp = DeepSeekV4FlashCompressor(
            512, 512, 64, 128, 1e-6, overlap=False
        ).cuda()
        out_dim = comp.head_dim
        t = {
            "wkv.weight": torch.randn(out_dim, 512, device="cuda"),
            "wgate.weight": torch.randn(out_dim, 512, device="cuda"),
        }
        comp.wkv.set_runtime_tensors(t, "wkv")
        comp.wgate.set_runtime_tensors(t, "wgate")
        comp.ape.data = torch.randn_like(comp.ape)
        comp.norm.weight.data = torch.randn_like(comp.norm.weight)

        from batchgen_kernels.attention.v4_compressor import (
            DeepSeekV4Compressor,
        )

        kernel_comp = DeepSeekV4Compressor(
            512, 512, 64, 128, 1e-6, overlap=False, rotate=False
        ).cuda()
        kernel_comp.ape.data = comp.ape.data
        kernel_comp.norm.weight.data = comp.norm.weight.data
        kernel_comp.wkv_weight.data = comp.wkv.weight
        kernel_comp.wgate_weight.data = comp.wgate.weight

        adapter = DeepSeekV4FlashMLADecodeAdapter(manager)
        cos_sin = build_v4_compress_cos_sin_cache(
            max_pos=512, theta=160000.0, rope_head_dim=64, device="cuda"
        )

        metadata = SimpleNamespace(
            c128_out_loc=torch.tensor([0], dtype=torch.int32, device="cuda"),
        )
        adapter._maybe_store_c128_emission(
            route=route,
            sequence_ids=[seq_id],
            positions=torch.tensor([127], dtype=torch.int64, device="cuda"),
            metadata=metadata,
            rope_cache=cos_sin,
            compress_hidden_states=torch.randn(
                1, 512, device="cuda", dtype=torch.float32
            ),
            compressor=kernel_comp,
        )

        stored = manager.c128.debug_read_kv(
            layer_idx=route.c128_layer_idx,
            token_slots=manager.c128.sequence_token_slots(
                seq_id, torch.tensor([0], device="cuda", dtype=torch.long)
            ),
        )
        assert stored.shape == (1, 512)
        assert torch.isfinite(stored.float()).all()
        assert stored.abs().sum() > 0
    finally:
        manager.destroy(empty_cuda_cache=True)


def test_v4_c128_remainder_seeding_fills_state():
    from batchgen.models.deepseek.deepseekv4_flash.model import (
        DeepSeekV4FlashCompressor,
    )
    from batchgen.attention.dsa.v4_flashmla_adapter import (
        DeepSeekV4FlashMLADecodeAdapter,
    )
    from batchgen_kernels.attention.v4_compressor import DeepSeekV4Compressor

    worker = _make_worker_shell([0, 4, 128], 3)
    worker.torch_device = torch.device("cuda:0")
    manager = worker._initialize_gpu_kv_manager_fixed_size()
    try:
        route = manager.get_layer_routing(2)
        seq_id = 8
        comp = DeepSeekV4Compressor(
            512, 512, 64, 128, 1e-6, overlap=False, rotate=False
        ).cuda()
        comp.ape.data = torch.randn_like(comp.ape)
        comp.norm.weight.data = torch.randn_like(comp.norm.weight)

        adapter = DeepSeekV4FlashMLADecodeAdapter(manager)
        remainder = 70
        cutoff = 128
        rem_hidden = torch.randn(
            remainder, 512, device="cuda", dtype=torch.float32
        )
        rem_pos = torch.arange(
            cutoff, cutoff + remainder, device="cuda", dtype=torch.int64
        )
        adapter.seed_c128_decode_state(
            c128_layer_idx=route.c128_layer_idx,
            sequence_id=seq_id,
            compressor=comp,
            remainder_hidden=rem_hidden,
            remainder_positions=rem_pos,
        )
        kv_state, score_state = adapter._c128_decode_state[
            (route.c128_layer_idx, seq_id)
        ]
        for slot in range(remainder):
            assert kv_state[slot].abs().sum() > 0
        for slot in range(remainder, 128):
            assert kv_state[slot].abs().sum() == 0
    finally:
        manager.destroy(empty_cuda_cache=True)


def test_install_decode_backend_noop_for_non_v4_manager():
    worker = _make_worker_shell([0, 4, 128], 3)
    worker.gpu_paged_kv_cache_manager = object()
    worker.model = SimpleNamespace(model=SimpleNamespace(layers=[]))
    worker._install_deepseek_v4_decode_backend()
    assert not hasattr(worker, "_deepseek_v4_decode_backend")


def test_backend_injection_into_real_wrappers():
    from batchgen.attention.v4_backend import (
        DeepseekV4AttnBackend,
        build_layer_configs_from_compress_ratios,
    )
    from batchgen.attention.dsa.v4_flashmla_adapter import (
        DeepSeekV4FlashMLADecodeAdapter,
    )

    worker = _make_worker_shell([0, 4, 128], 3)
    manager = worker._initialize_gpu_kv_manager_fixed_size()
    try:
        layer_configs = build_layer_configs_from_compress_ratios(
            [0, 4, 128],
            n_heads=64,
            head_dim=512,
            rope_head_dim=64,
        )
        backend = DeepseekV4AttnBackend(
            layer_configs=layer_configs,
            page_size=manager.swa.page_size_tokens,
            flashmla_backend=DeepSeekV4FlashMLADecodeAdapter(manager),
        )
        assert len(backend.layer_configs) == 3
        assert backend.layer_configs[0].compress_ratio == 0
        assert backend.layer_configs[1].compress_ratio == 4
        assert backend.layer_configs[2].compress_ratio == 128
        assert backend._flashmla is not None
    finally:
        manager.destroy(empty_cuda_cache=True)
