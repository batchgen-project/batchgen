import logging
from types import SimpleNamespace

import pytest
import torch

from batchgen.models.glm.glm5.Parallel_Strategy_Manager import (
    GLM5ParallelStrategyManager,
)
from batchgen.models.glm.glm5.glm5_parameter_server import GLM5_Parameter_Server
from batchgen.models.glm.glm5.config import dsa_layer_skips_topk
from batchgen.models.glm.glm5.model import Glm5Indexer
from batchgen.models.glm.glm5.wrappers import GLM5AttnWrapper
from batchgen.models.wrappers.attention import AttnWrapperBase
from batchgen.batchgen_worker import BatchGenWorker


def _fake_prefill_manager(num_layers=5, first_k_dense=3, num_experts=4):
    manager = object.__new__(GLM5ParallelStrategyManager)
    manager.rank = 0
    manager.is_fp8_experts = False
    manager.model_config = SimpleNamespace(num_hidden_layers=num_layers)
    manager.loaded_model_config = SimpleNamespace(
        num_hidden_layers=num_layers,
        first_k_dense_replace=first_k_dense,
        n_routed_experts=num_experts,
        hidden_size=6144,
        moe_intermediate_size=2048,
        quantization_config={"weight_block_size": [128, 128]},
    )
    layers = []
    for layer_idx in range(num_layers):
        layer = SimpleNamespace(
            self_attn=SimpleNamespace(
                persistent=False,
                module_key=f"attn_{layer_idx}",
            ),
            mlp=SimpleNamespace(),
        )
        if layer_idx >= first_k_dense:
            layer.mlp.shared_experts = SimpleNamespace(
                persistent=False,
                module_key=f"shared_expert_{layer_idx}",
            )
            layer.mlp.experts = [
                SimpleNamespace(
                    persistent=False,
                    module_key=f"routed_expert_{layer_idx}_{expert_idx}",
                )
                for expert_idx in range(num_experts)
            ]
        layers.append(layer)
    manager.model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    manager.weight_copy_task = {
        "attn": [f"attn_{layer_idx}" for layer_idx in range(num_layers)],
        "shared_expert": [
            f"shared_expert_{layer_idx}"
            for layer_idx in range(first_k_dense, num_layers)
        ],
        "routed_expert": [
            f"routed_expert_{layer_idx}_{expert_idx}"
            for layer_idx in range(first_k_dense, num_layers)
            for expert_idx in range(num_experts)
        ],
    }
    return manager


def test_glm5_prefill_manager_derives_fp8_from_checkpoint_config():
    manager = GLM5ParallelStrategyManager(
        loaded_model_config=SimpleNamespace(
            quantization="fp8",
            quantization_config={"quant_method": "fp8"},
        ),
        engine_config=None,
        model_config=None,
        core_engine=None,
        skeleton_state_dict={},
        local_rank=0,
        global_rank=0,
        world_size=8,
    )

    assert manager.is_fp8_experts is True
    assert manager._observed_fp8_expert_scales is False


def test_glm5_prefill_weight_offload_contract_accepts_all_streamed(caplog):
    manager = _fake_prefill_manager()
    caplog.set_level(logging.INFO)

    manager._validate_prefill_weight_offload_contract()

    assert "PREFILL_WEIGHT_OFFLOAD_CONTRACT" in caplog.text
    assert '"routed_experts_streamed":8' in caplog.text
    assert (
        '"expected_counts":{"attn":5,"shared_expert":2,"routed_expert":8}'
        in caplog.text
    )
    assert (
        '"actual_wrapper_counts":{"attn":5,"shared_expert":2,'
        '"routed_expert":8}' in caplog.text
    )


def test_glm5_prefill_weight_offload_contract_rejects_persistent_expert():
    manager = _fake_prefill_manager()
    manager.model.model.layers[3].mlp.experts[2].persistent = True

    with pytest.raises(RuntimeError, match="persistent wrappers"):
        manager._validate_prefill_weight_offload_contract()


def test_glm5_prefill_weight_offload_contract_rejects_wrong_wrapper_count():
    manager = _fake_prefill_manager()
    manager.model.model.layers[4].mlp.experts.pop()

    with pytest.raises(RuntimeError, match="wrapper cardinality mismatch"):
        manager._validate_prefill_weight_offload_contract()


def test_glm5_prefill_weight_offload_contract_rejects_wrong_wrapper_identity():
    manager = _fake_prefill_manager()
    manager.model.model.layers[3].mlp.experts[2].module_key = "routed_expert_3_99"

    with pytest.raises(RuntimeError, match="wrapper identity mismatch"):
        manager._validate_prefill_weight_offload_contract()


def _set_fake_expert_scales(manager, device):
    scale_shapes = {
        "gate_proj.weight_scale_inv": (16, 48),
        "up_proj.weight_scale_inv": (16, 48),
        "down_proj.weight_scale_inv": (48, 16),
    }
    for layer in manager.model.model.layers[3:]:
        for wrapper in [layer.mlp.shared_experts, *layer.mlp.experts]:
            wrapper.weight_dequant_scale = {
                key: torch.ones(shape, dtype=torch.float32, device=device)
                for key, shape in scale_shapes.items()
            }
    manager._observed_fp8_expert_scales = True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA residency")
def test_glm5_prefill_weight_offload_contract_accepts_resident_fp8_scales(caplog):
    manager = _fake_prefill_manager()
    manager.is_fp8_experts = True
    device = torch.device("cuda", torch.cuda.current_device())
    manager.engine_config = SimpleNamespace(
        Basic_Config=SimpleNamespace(device_torch=device)
    )
    _set_fake_expert_scales(manager, device)
    caplog.set_level(logging.INFO)

    manager._validate_prefill_weight_offload_contract()

    assert '"format":"fp8_block_scale_inv"' in caplog.text
    assert '"expected_tensors":30' in caplog.text
    assert '"actual_tensors":30' in caplog.text
    assert '"resident_on_device":true' in caplog.text
    assert '"all_shapes_validated":true' in caplog.text


def test_glm5_prefill_weight_offload_contract_rejects_missing_fp8_scales():
    manager = _fake_prefill_manager()
    manager.is_fp8_experts = True
    manager.engine_config = SimpleNamespace(
        Basic_Config=SimpleNamespace(device_torch="cuda")
    )

    with pytest.raises(RuntimeError, match="resident scale metadata mismatch"):
        manager._validate_prefill_weight_offload_contract()


def test_glm5_prefill_weight_offload_contract_rejects_scales_for_non_fp8():
    manager = _fake_prefill_manager()
    manager._observed_fp8_expert_scales = True

    with pytest.raises(RuntimeError, match="declares non-FP8 experts"):
        manager._validate_prefill_weight_offload_contract()


def test_glm5_prefill_weight_offload_contract_rejects_nonresident_fp8_scales():
    manager = _fake_prefill_manager()
    manager.is_fp8_experts = True
    manager.engine_config = SimpleNamespace(
        Basic_Config=SimpleNamespace(device_torch="cuda")
    )
    _set_fake_expert_scales(manager, torch.device("cpu"))

    with pytest.raises(RuntimeError, match="resident FP32 CUDA tensor"):
        manager._validate_prefill_weight_offload_contract()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA residency")
def test_glm5_prefill_weight_offload_contract_rejects_wrong_fp8_scale_shape():
    manager = _fake_prefill_manager()
    manager.is_fp8_experts = True
    device = torch.device("cuda", torch.cuda.current_device())
    manager.engine_config = SimpleNamespace(
        Basic_Config=SimpleNamespace(device_torch=device)
    )
    _set_fake_expert_scales(manager, device)
    manager.model.model.layers[3].mlp.experts[0].weight_dequant_scale[
        "gate_proj.weight_scale_inv"
    ] = torch.ones((1, 1), dtype=torch.float32, device=device)

    with pytest.raises(RuntimeError, match="expected block shape"):
        manager._validate_prefill_weight_offload_contract()


def _fake_parameter_server_inventory():
    server = object.__new__(GLM5_Parameter_Server)
    server.model_config = SimpleNamespace(
        num_hidden_layers=4,
        first_k_dense_replace=3,
        n_routed_experts=2,
    )
    server.weight_copy_task = server._expected_prefill_host_modules()
    server.state_dict_name_map = {}
    expected_tensor_keys = {}

    def add(module_key, tensor_keys):
        expected_tensor_keys[module_key] = list(tensor_keys)
        for tensor_key in tensor_keys:
            server.state_dict_name_map[f"{module_key}.{tensor_key}"] = {
                "module_key": module_key,
                "tensor_key": tensor_key,
            }

    for layer_idx in range(4):
        add(f"attn_{layer_idx}", ["q.weight"])
        if layer_idx == 3:
            add(
                f"shared_expert_{layer_idx}",
                ["gate_proj.weight", "up_proj.weight", "down_proj.weight"],
            )
            for expert_idx in range(2):
                add(
                    f"routed_expert_{layer_idx}_{expert_idx}",
                    ["gate_proj.weight", "up_proj.weight", "down_proj.weight"],
                )
    return server, expected_tensor_keys


def test_parameter_server_inventory_accepts_exact_modules_and_tensors():
    server, expected_tensor_keys = _fake_parameter_server_inventory()

    inventory = server._validate_prefill_host_weight_inventory(
        expected_tensor_keys
    )

    assert inventory["expected_module_counts"] == {
        "attn": 4,
        "routed_expert": 2,
        "shared_expert": 1,
    }
    assert inventory["actual_module_counts"] == inventory["expected_module_counts"]
    assert inventory["mapped_tensor_counts"] == {
        "attn": 4,
        "routed_expert": 6,
        "shared_expert": 3,
    }


def test_parameter_server_inventory_rejects_missing_module():
    server, expected_tensor_keys = _fake_parameter_server_inventory()
    server.weight_copy_task["routed_expert"].pop()

    with pytest.raises(RuntimeError, match="routed_expert inventory mismatch"):
        server._validate_prefill_host_weight_inventory(expected_tensor_keys)


def test_parameter_server_inventory_rejects_extra_mapped_module():
    server, expected_tensor_keys = _fake_parameter_server_inventory()
    server.state_dict_name_map["unexpected.weight"] = {
        "module_key": "unexpected",
        "tensor_key": "weight",
    }

    with pytest.raises(RuntimeError, match="ordered mapped-module inventory mismatch"):
        server._validate_prefill_host_weight_inventory(expected_tensor_keys)


def test_parameter_server_inventory_rejects_missing_expert_tensor():
    server, expected_tensor_keys = _fake_parameter_server_inventory()
    del server.state_dict_name_map[
        "shared_expert_3.down_proj.weight"
    ]

    with pytest.raises(RuntimeError, match="mapped tensors mismatch"):
        server._validate_prefill_host_weight_inventory(expected_tensor_keys)


def _attach_loaded_parameter_server(server, expected_tensor_keys):
    server._prefill_host_weight_inventory = (
        server._validate_prefill_host_weight_inventory(expected_tensor_keys)
    )
    loaded = {}
    offset = 0
    for module_key, tensor_keys in expected_tensor_keys.items():
        loaded[module_key] = {}
        for tensor_key in tensor_keys:
            loaded[module_key][tensor_key] = SimpleNamespace(
                offset=offset,
                byte_size=1,
                tensor_shape=[1],
                dtype="uint8",
            )
            offset += 1
    server.parameter_server = SimpleNamespace(
        module_weights_shm=lambda: loaded
    )
    return loaded


def test_parameter_server_loaded_inventory_accepts_exact_host_metadata():
    server, expected_tensor_keys = _fake_parameter_server_inventory()
    _attach_loaded_parameter_server(server, expected_tensor_keys)

    inventory = server._validate_loaded_prefill_host_weight_inventory(1_000)

    assert inventory["actual_module_counts"] == {
        "attn": 4,
        "routed_expert": 2,
        "shared_expert": 1,
    }
    assert inventory["mapped_tensor_counts"] == {
        "attn": 4,
        "routed_expert": 6,
        "shared_expert": 3,
    }


def test_parameter_server_loaded_inventory_rejects_missing_host_tensor():
    server, expected_tensor_keys = _fake_parameter_server_inventory()
    loaded = _attach_loaded_parameter_server(server, expected_tensor_keys)
    del loaded["routed_expert_3_1"]["down_proj.weight"]

    with pytest.raises(RuntimeError, match="loaded host tensors mismatch"):
        server._validate_loaded_prefill_host_weight_inventory(1_000)


def test_parameter_server_loaded_inventory_rejects_out_of_range_metadata():
    server, expected_tensor_keys = _fake_parameter_server_inventory()
    loaded = _attach_loaded_parameter_server(server, expected_tensor_keys)
    loaded["attn_0"]["q.weight"].offset = 1_000

    with pytest.raises(RuntimeError, match="range exceeds backing"):
        server._validate_loaded_prefill_host_weight_inventory(1_000)


def test_parameter_server_loaded_inventory_rejects_overlapping_metadata():
    server, expected_tensor_keys = _fake_parameter_server_inventory()
    loaded = _attach_loaded_parameter_server(server, expected_tensor_keys)
    loaded["attn_1"]["q.weight"].offset = 0

    with pytest.raises(RuntimeError, match="ranges overlap"):
        server._validate_loaded_prefill_host_weight_inventory(1_000)


def test_parameter_server_loaded_inventory_rejects_missing_metadata_fields():
    server, expected_tensor_keys = _fake_parameter_server_inventory()
    loaded = _attach_loaded_parameter_server(server, expected_tensor_keys)
    del loaded["attn_0"]["q.weight"].dtype

    with pytest.raises(RuntimeError, match="metadata is malformed"):
        server._validate_loaded_prefill_host_weight_inventory(1_000)


def test_parameter_server_loaded_inventory_rejects_invalid_shape():
    server, expected_tensor_keys = _fake_parameter_server_inventory()
    loaded = _attach_loaded_parameter_server(server, expected_tensor_keys)
    loaded["attn_0"]["q.weight"].tensor_shape = [0]

    with pytest.raises(RuntimeError, match="invalid range/shape"):
        server._validate_loaded_prefill_host_weight_inventory(1_000)


def test_parameter_server_loaded_inventory_rejects_dtype_size_mismatch():
    server, expected_tensor_keys = _fake_parameter_server_inventory()
    loaded = _attach_loaded_parameter_server(server, expected_tensor_keys)
    loaded["attn_0"]["q.weight"].dtype = "float32"

    with pytest.raises(RuntimeError, match="invalid dtype/size"):
        server._validate_loaded_prefill_host_weight_inventory(1_000)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Parameter_Server requires CUDA")
def test_core_parameter_server_returns_bound_tensor_metadata(tmp_path):
    import gc
    import json
    import uuid

    from batchgen.models.engine_loader import core_engine

    Parameter_Server = core_engine.Parameter_Server

    (tmp_path / "tiny.bin").write_bytes(b"\0\0\0\0")
    (tmp_path / "tiny.json").write_text(json.dumps({
        "file_name": "tiny.bin",
        "total_byte_size": 4,
        "state_dict": {
            "tiny.weight": {
                "dtype": "float32",
                "shape": [1],
                "offset": 0,
                "byte_size": 4,
            },
        },
    }))
    suffix = uuid.uuid4().hex
    weights_name = f"/batchgen_tensor_meta_weights_{suffix}"
    metadata_name = f"/batchgen_tensor_meta_metadata_{suffix}"
    parameter_server = Parameter_Server(False, True)
    parameter_server.Init(
        weights_name,
        metadata_name,
        4096,
        str(tmp_path),
        {
            "tiny.weight": {
                "module_key": "tiny_module",
                "tensor_key": "weight",
            },
        },
    )

    metadata = parameter_server.module_weights_shm()["tiny_module"]["weight"]

    assert type(metadata).__name__ == "TensorMeta"
    assert metadata.offset == 0
    assert list(metadata.tensor_shape) == [1]
    assert metadata.byte_size == 4
    assert metadata.dtype == "float32"
    del metadata
    del parameter_server
    gc.collect()


def _fake_backing_server(*, enable_memfd, enable_hugetlbfs=False):
    server = object.__new__(GLM5_Parameter_Server)
    server.enable_memfd = enable_memfd
    server.enable_hugetlbfs = enable_hugetlbfs
    server.shm_name = "/weights-test"
    server.tensor_meta_shm_name = "/metadata-test"
    return server


def test_parameter_server_backing_rejects_missing_memfd():
    server = _fake_backing_server(enable_memfd=True)
    server.parameter_server = SimpleNamespace(weights_memfd_fd=lambda: -1)

    with pytest.raises(RuntimeError, match="no live file descriptor"):
        server._prefill_host_backing_inventory(1_000)


def test_parameter_server_backing_rejects_invalid_memfd_target(monkeypatch):
    server = _fake_backing_server(enable_memfd=True)
    server.parameter_server = SimpleNamespace(weights_memfd_fd=lambda: 7)
    monkeypatch.setattr("os.readlink", lambda path: "/tmp/not-a-memfd")
    monkeypatch.setattr("os.fstat", lambda fd: SimpleNamespace(st_size=1_000))

    with pytest.raises(RuntimeError, match="memfd weight backing is invalid"):
        server._prefill_host_backing_inventory(1_000)


def test_parameter_server_backing_accepts_hugetlbfs_before_memfd(monkeypatch):
    server = _fake_backing_server(enable_memfd=True, enable_hugetlbfs=True)
    server.parameter_server = SimpleNamespace(weights_memfd_fd=lambda: -1)
    monkeypatch.setattr(
        "os.path.exists",
        lambda path: path == "/dev/hugepages/weights-test",
    )
    monkeypatch.setattr(
        "os.path.getsize",
        lambda path: 2_000
        if path == "/dev/hugepages/weights-test"
        else 1_000,
    )

    objects = server._prefill_host_backing_inventory(1_000)

    assert objects[0] == {
        "kind": "weights",
        "backend": "hugetlbfs",
        "name": "/weights-test",
        "path": "/dev/hugepages/weights-test",
        "size_bytes": 2_000,
    }


def test_parameter_server_backing_rejects_undersized_hugetlbfs(monkeypatch):
    server = _fake_backing_server(enable_memfd=True, enable_hugetlbfs=True)
    monkeypatch.setattr("os.path.exists", lambda path: True)
    monkeypatch.setattr("os.path.getsize", lambda path: 999)

    with pytest.raises(RuntimeError, match="hugetlbfs weight backing is undersized"):
        server._prefill_host_backing_inventory(1_000)


def test_parameter_server_backing_rejects_undersized_posix_shm(monkeypatch):
    server = _fake_backing_server(enable_memfd=False)
    monkeypatch.setattr("os.path.getsize", lambda path: 999)

    with pytest.raises(RuntimeError, match="POSIX-SHM weight backing is undersized"):
        server._prefill_host_backing_inventory(1_000)


def test_parameter_server_backing_rejects_empty_metadata_shm(monkeypatch):
    server = _fake_backing_server(enable_memfd=False)
    monkeypatch.setattr(
        "os.path.getsize",
        lambda path: 1_000 if path.endswith("weights-test") else 0,
    )

    with pytest.raises(RuntimeError, match="tensor-metadata shared memory is empty"):
        server._prefill_host_backing_inventory(1_000)


def test_glm5_prefill_kv_offload_audit_aggregates_layers():
    GLM5AttnWrapper.start_prefill_kv_offload_audit()
    GLM5AttnWrapper.record_prefill_kv_offload(
        "primary", 0, sequences=1, tokens=262_144
    )
    GLM5AttnWrapper.record_prefill_kv_offload(
        "primary", 0, sequences=1, tokens=65_536
    )
    GLM5AttnWrapper.record_prefill_kv_offload(
        "aux", 0, sequences=2, tokens=327_680
    )

    audit = GLM5AttnWrapper.finish_prefill_kv_offload_audit()

    assert audit["primary"][0] == {
        "calls": 2,
        "sequences": 2,
        "tokens": 327_680,
    }
    assert audit["aux"][0] == {
        "calls": 1,
        "sequences": 2,
        "tokens": 327_680,
    }
    assert GLM5AttnWrapper.glm5_prefill_kv_offload_audit is None


def test_prefill_retirement_audit_records_layer_and_task_count(monkeypatch):
    class DummyTask:
        def __init__(self):
            self.waited = False

        def wait(self):
            self.waited = True

    tasks = [DummyTask(), DummyTask()]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(AttnWrapperBase, "pending_prefill_offload_tasks", [])
    monkeypatch.setattr(AttnWrapperBase, "pending_prefill_offload_tensors", [])
    monkeypatch.setattr(AttnWrapperBase, "pending_prefill_offload_layer_idx", None)
    monkeypatch.setattr(AttnWrapperBase, "prefill_offload_retirements", None)
    AttnWrapperBase.start_prefill_offload_retirement_audit()
    AttnWrapperBase.pending_prefill_offload_tasks.extend(tasks)
    AttnWrapperBase.pending_prefill_offload_tensors.append(torch.zeros(1))
    AttnWrapperBase.pending_prefill_offload_layer_idx = 17

    retired = AttnWrapperBase.retire_pending_prefill_offloads(
        device=torch.device("cpu"),
        reason="test",
    )
    audit = AttnWrapperBase.finish_prefill_offload_retirement_audit()

    assert retired == 2
    assert all(task.waited for task in tasks)
    assert audit == [{"layer_idx": 17, "tasks": 2}]


def _fake_glm52_worker_audit(
    monkeypatch,
    *,
    retirements=None,
    pending_tasks=None,
    pending_tensors=None,
    pending_layer=None,
):
    config = SimpleNamespace(
        model_type="glm_moe_dsa_5_2",
        num_hidden_layers=2,
        index_topk_pattern=["F", "S"],
    )
    worker = object.__new__(BatchGenWorker)
    worker.rank = 3
    worker._glm52_prefill_fallback_audit_active = True
    worker.model = SimpleNamespace(model=SimpleNamespace(layers=[
        SimpleNamespace(self_attn=SimpleNamespace(
            module=SimpleNamespace(config=config, indexer=object())
        )),
        SimpleNamespace(self_attn=SimpleNamespace(
            module=SimpleNamespace(config=config, indexer=None)
        )),
    ]))
    worker._local_to_uuid_map = {9: "sequence-9"}
    worker.global_batch = SimpleNamespace(
        get_sequence=lambda uuid: SimpleNamespace(
            uuid=uuid,
            prompt_length=262_144,
            assigned_rank=3,
        )
    )
    monkeypatch.setattr(GLM5AttnWrapper, "glm5_prefill_kv_offload_audit", {
        "primary": {
            0: {"calls": 1, "sequences": 1, "tokens": 262_144},
            1: {"calls": 1, "sequences": 1, "tokens": 262_144},
        },
        "aux": {
            0: {"calls": 1, "sequences": 1, "tokens": 262_144},
        },
    })
    if retirements is None:
        retirements = [
            {"layer_idx": 0, "tasks": 2},
            {"layer_idx": 1, "tasks": 1},
        ]
    monkeypatch.setattr(AttnWrapperBase, "prefill_offload_retirements", retirements)
    monkeypatch.setattr(
        AttnWrapperBase, "pending_prefill_offload_tasks", pending_tasks or []
    )
    monkeypatch.setattr(
        AttnWrapperBase, "pending_prefill_offload_tensors", pending_tensors or []
    )
    monkeypatch.setattr(
        AttnWrapperBase, "pending_prefill_offload_layer_idx", pending_layer
    )
    monkeypatch.setattr(Glm5Indexer, "prefill_rope_hadamard_audit", {
        "fused": {0: 1},
        "fallback": {},
    })
    return worker


def test_worker_accepts_complete_per_layer_kv_offload_audit(monkeypatch, caplog):
    worker = _fake_glm52_worker_audit(monkeypatch)
    caplog.set_level(logging.INFO)

    worker._finish_glm5_prefill_offload_audit([9])

    assert "PREFILL_KV_OFFLOAD_CONTRACT" in caplog.text
    assert '"sequence_uuids":["sequence-9"]' in caplog.text
    assert '"sequence_lengths":[262144]' in caplog.text
    assert '"layers_without_auxiliary_by_design":[1]' in caplog.text
    assert '"retirement_tasks":3' in caplog.text
    assert '"retirement_tasks_by_layer":{"0":2,"1":1}' in caplog.text
    assert '"fallback_count":0' in caplog.text
    assert '"fused_indexer_layers":[0]' in caplog.text


def test_worker_rejects_prefill_indexer_fallback(monkeypatch):
    worker = _fake_glm52_worker_audit(monkeypatch)
    Glm5Indexer.record_prefill_rope_hadamard_path("fallback", 0)

    with pytest.raises(RuntimeError, match="unfused RoPE/Hadamard fallback"):
        worker._finish_glm5_prefill_offload_audit([9])


def test_worker_rejects_missing_fused_indexer_layer(monkeypatch):
    worker = _fake_glm52_worker_audit(monkeypatch)
    Glm5Indexer.prefill_rope_hadamard_audit["fused"].clear()

    with pytest.raises(RuntimeError, match="fused indexer layer coverage mismatch"):
        worker._finish_glm5_prefill_offload_audit([9])


def test_worker_rejects_retirement_task_mismatch(monkeypatch):
    worker = _fake_glm52_worker_audit(monkeypatch, retirements=[
        {"layer_idx": 0, "tasks": 1},
        {"layer_idx": 1, "tasks": 1},
    ])

    with pytest.raises(RuntimeError, match="per-layer KV retirement mismatch"):
        worker._finish_glm5_prefill_offload_audit([9])


def test_worker_rejects_pending_references(monkeypatch):
    worker = _fake_glm52_worker_audit(
        monkeypatch,
        pending_tasks=[object()],
        pending_tensors=[object()],
        pending_layer=1,
    )

    with pytest.raises(RuntimeError, match="references remain after retirement"):
        worker._finish_glm5_prefill_offload_audit([9])


def test_worker_rejects_sequence_assigned_to_another_rank(monkeypatch):
    worker = _fake_glm52_worker_audit(monkeypatch)
    worker.global_batch = SimpleNamespace(
        get_sequence=lambda uuid: SimpleNamespace(
            uuid=uuid,
            prompt_length=262_144,
            assigned_rank=7,
        )
    )

    with pytest.raises(RuntimeError, match="sequence/rank assignment mismatch"):
        worker._finish_glm5_prefill_offload_audit([9])


def test_worker_rejects_prefill_sequence_uuid_mismatch(monkeypatch):
    worker = _fake_glm52_worker_audit(monkeypatch)
    worker.global_batch = SimpleNamespace(
        get_sequence=lambda uuid: SimpleNamespace(
            uuid="different-sequence",
            prompt_length=262_144,
            assigned_rank=3,
        )
    )

    with pytest.raises(RuntimeError, match="local/sequence UUID mismatch"):
        worker._finish_glm5_prefill_offload_audit([9])


def test_worker_accepts_exact_glm52_78_primary_21_auxiliary_schedule(
    monkeypatch, caplog
):
    config = SimpleNamespace(
        model_type="glm_moe_dsa_5_2",
        num_hidden_layers=78,
        index_topk_freq=4,
        index_skip_topk_offset=3,
    )
    expected_aux = [
        layer_idx for layer_idx in range(78)
        if not dsa_layer_skips_topk(config, layer_idx)
    ]
    assert expected_aux == [0, 1, 2, *range(6, 75, 4)]
    assert len(expected_aux) == 21

    worker = object.__new__(BatchGenWorker)
    worker.rank = 3
    worker._glm52_prefill_fallback_audit_active = True
    worker.model = SimpleNamespace(model=SimpleNamespace(layers=[
        SimpleNamespace(self_attn=SimpleNamespace(module=SimpleNamespace(
            config=config,
            indexer=object() if layer_idx in expected_aux else None,
        )))
        for layer_idx in range(78)
    ]))
    worker._local_to_uuid_map = {9: "sequence-9"}
    worker.global_batch = SimpleNamespace(
        get_sequence=lambda uuid: SimpleNamespace(
            uuid=uuid,
            prompt_length=262_144,
            assigned_rank=3,
        )
    )
    monkeypatch.setattr(GLM5AttnWrapper, "glm5_prefill_kv_offload_audit", {
        "primary": {
            layer_idx: {"calls": 1, "sequences": 1, "tokens": 262_144}
            for layer_idx in range(78)
        },
        "aux": {
            layer_idx: {"calls": 1, "sequences": 1, "tokens": 262_144}
            for layer_idx in expected_aux
        },
    })
    monkeypatch.setattr(AttnWrapperBase, "prefill_offload_retirements", [
        {
            "layer_idx": layer_idx,
            "tasks": 2 if layer_idx in expected_aux else 1,
        }
        for layer_idx in range(78)
    ])
    monkeypatch.setattr(AttnWrapperBase, "pending_prefill_offload_tasks", [])
    monkeypatch.setattr(AttnWrapperBase, "pending_prefill_offload_tensors", [])
    monkeypatch.setattr(AttnWrapperBase, "pending_prefill_offload_layer_idx", None)
    monkeypatch.setattr(Glm5Indexer, "prefill_rope_hadamard_audit", {
        "fused": {layer_idx: 1 for layer_idx in expected_aux},
        "fallback": {},
    })
    caplog.set_level(logging.INFO)

    worker._finish_glm5_prefill_offload_audit([9])

    assert '"primary_layers":[' in caplog.text
    assert '"auxiliary_layers":[0,1,2,6' in caplog.text
    assert '"retirement_events":78' in caplog.text
    assert '"retirement_tasks":99' in caplog.text
    assert '"fallback_count":0' in caplog.text


def test_worker_prefill_audit_abort_drains_and_allows_retry(monkeypatch):
    class DummyTask:
        def __init__(self, fail=False):
            self.fail = fail
            self.waited = False

        def wait(self):
            self.waited = True
            if self.fail:
                raise RuntimeError("task failure")

    tasks = [DummyTask(fail=True), DummyTask()]
    original_tasks = list(tasks)
    worker = object.__new__(BatchGenWorker)
    worker.torch_device = torch.device("cpu")
    worker._glm52_prefill_fallback_audit_active = True
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(AttnWrapperBase, "pending_prefill_offload_tasks", tasks)
    monkeypatch.setattr(
        AttnWrapperBase, "pending_prefill_offload_tensors", [torch.zeros(1)]
    )
    monkeypatch.setattr(AttnWrapperBase, "pending_prefill_offload_layer_idx", 4)
    monkeypatch.setattr(AttnWrapperBase, "prefill_offload_retirements", [])
    monkeypatch.setattr(GLM5AttnWrapper, "glm5_prefill_kv_offload_audit", {
        "primary": {}, "aux": {},
    })
    monkeypatch.setattr(Glm5Indexer, "prefill_rope_hadamard_audit", {
        "fused": {}, "fallback": {},
    })

    with pytest.raises(RuntimeError, match="cleanup failures"):
        worker._abort_glm5_prefill_offload_audit()

    assert all(task.waited for task in original_tasks)
    assert AttnWrapperBase.pending_prefill_offload_tasks == []
    assert AttnWrapperBase.pending_prefill_offload_tensors == []
    assert AttnWrapperBase.pending_prefill_offload_layer_idx is None
    assert AttnWrapperBase.prefill_offload_retirements is None
    assert GLM5AttnWrapper.glm5_prefill_kv_offload_audit is None
    assert Glm5Indexer.prefill_rope_hadamard_audit is None

    GLM5AttnWrapper.start_prefill_kv_offload_audit()
    AttnWrapperBase.start_prefill_offload_retirement_audit()
    Glm5Indexer.start_prefill_rope_hadamard_audit()
    worker._glm52_prefill_fallback_audit_active = True
    worker._abort_glm5_prefill_offload_audit()

    assert GLM5AttnWrapper.glm5_prefill_kv_offload_audit is None
    assert AttnWrapperBase.prefill_offload_retirements is None
    assert Glm5Indexer.prefill_rope_hadamard_audit is None


def test_worker_prefill_handler_preserves_original_error_and_allows_retry(
    monkeypatch,
):
    class OriginalPrefillError(RuntimeError):
        pass

    worker = object.__new__(BatchGenWorker)
    worker.enable_prepack = False
    worker.rank = 0
    worker.torch_device = torch.device("cpu")
    starts = []
    finishes = []
    aborts = []

    monkeypatch.setattr(
        worker,
        "_start_glm5_prefill_offload_audit",
        lambda: starts.append(True) or True,
    )
    monkeypatch.setattr(
        AttnWrapperBase,
        "retire_pending_prefill_offloads",
        lambda **kwargs: 0,
    )
    monkeypatch.setattr(
        worker,
        "_finish_glm5_prefill_offload_audit",
        lambda indices: finishes.append(list(indices)),
    )

    def failed_prefill(indices):
        raise OriginalPrefillError("original prefill failure")

    def failed_cleanup():
        aborts.append(True)
        raise RuntimeError("cleanup failure")

    worker.prefill = failed_prefill
    worker._abort_glm5_prefill_offload_audit = failed_cleanup

    with pytest.raises(OriginalPrefillError, match="original prefill failure"):
        worker._run_prefill_with_offload_audit([9])

    worker.prefill = lambda indices: None
    worker._abort_glm5_prefill_offload_audit = lambda: aborts.append(True)
    elapsed = worker._run_prefill_with_offload_audit([9])

    assert elapsed >= 0
    assert len(starts) == 2
    assert aborts == [True]
    assert finishes == [[9]]
