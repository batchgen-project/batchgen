"""CPU source guards and binding checks for the rank-sync ownership seam."""

import ast
import copy
import dataclasses
import gc
import logging
import os
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "batchgen/worker/rank_sync_engine.py"
WORKER = ROOT / "batchgen/batchgen_worker.py"


def engine_method(name):
    cls = next(node for node in ast.parse(ENGINE.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == "RankSyncEngine")
    return copy.deepcopy(next(node for node in cls.body
                              if isinstance(node, ast.FunctionDef) and node.name == name))


def load_rank_sync(namespace):
    """Lift the real definitions, retaining the caller's fake device and clock.

    Like the replica-transport source tests, this compiles the definitions
    without importing the native dependency graph. The supplied namespace is
    also the methods' globals, so fixture overrides keep their original reach.
    """
    tree = ast.parse(ENGINE.read_text())
    nodes = [node for node in tree.body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             or (isinstance(node, ast.ImportFrom) and node.module == "__future__")
             or (isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name)
                         and target.id.startswith("_COHORT_QUIESCE_")
                         for target in node.targets))]
    for name, value in dict(__name__=__name__, dataclass=dataclasses.dataclass, gc=gc,
                            logging=logging, os=os, socket=socket, time=time).items():
        namespace.setdefault(name, value)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ENGINE), "exec"), namespace)
    return namespace


def _binding(namespace):
    cls = next(node for node in ast.parse(WORKER.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker")
    constructor = next(node for node in cls.body
                       if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    builds = [node for node in ast.walk(constructor)
              if isinstance(node, ast.Assign)
              and any(ast.unparse(target) == "self._rank_sync_engine"
                      for target in node.targets)]
    assert len(builds) == 1
    init = ast.parse("def __init__(self):\n    self.pynccl_port_base = 21000").body[0]
    init.body.append(copy.deepcopy(builds[0]))
    names = {"_cohort_shutdown_probe", "set_cohort_shutdown_probe",
             "_cohort_shutdown_requested", "_ensure_pynccl_communicator",
             "_check_and_reinit_pynccl"}
    members = [copy.deepcopy(node) for node in cls.body
               if isinstance(node, ast.FunctionDef) and node.name in names]
    cls.body = [init, *members]
    cls.bases = []
    cls.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 str(WORKER), "exec"), namespace)
    return namespace["BatchGenWorker"]


def test_engine_never_receives_or_retains_a_worker():
    tree = ast.parse(ENGINE.read_text())
    owned = {"_nccl_group", "_nccl_port", "comm", "_sync_coordinator",
             "_gloo_migration_group", "_nvshmem_initialized_this_run",
             "_cohort_shutdown_probe_value"}
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "RankSyncEngine")
    methods = {node.name for node in cls.body if isinstance(node, ast.FunctionDef)}
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            assert "worker" not in node.arg.lower(), (node.lineno, node.arg)
        if isinstance(node, ast.Name):
            name = node.id
            assert name == "WORKER_SELF_EXIT_CODE" or "worker" not in name.lower(), (
                node.lineno, name
            )
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert "BatchGenWorker" not in node.value
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            assert "batchgen_worker" not in ast.unparse(node)
            assert "BatchGenWorker" not in ast.unparse(node)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "self":
                assert node.attr in owned | methods, (node.lineno, node.attr)
        if isinstance(node, ast.Assign):
            assert not any(isinstance(target, ast.Name) and target.id == "self"
                           for target in node.targets), node.lineno
        assert not isinstance(node, (ast.Yield, ast.YieldFrom, ast.GeneratorExp))
        if isinstance(node, ast.Call):
            assert ast.unparse(node.func) != "os._exit"


def test_delegator_builds_once_reuses_the_engine_and_binds_comm_outcomes():
    namespace = load_rank_sync({"dist": SimpleNamespace(is_initialized=lambda: False)})
    real_engine = namespace["RankSyncEngine"]
    built = []

    def build(**kwargs):
        instance = real_engine(**kwargs)
        built.append(instance)
        return instance

    namespace["RankSyncEngine"] = build
    worker_type = _binding(namespace)
    worker = worker_type()
    worker.rank = worker.local_rank = 0
    worker.world_size = 1
    worker.torch_device = "cuda:0"
    worker.pynccl_port_span = 4
    engine = worker._rank_sync_engine
    comm = SimpleNamespace(available=False)
    engine.comm = comm
    for _ in range(2):
        worker._ensure_pynccl_communicator()
        assert worker.comm is comm
        assert worker._check_and_reinit_pynccl() is True
        assert worker._rank_sync_engine is engine
    assert built == [engine]
    assert engine._nccl_port == 21000


def test_probe_property_covers_uninitialised_fixtures_and_keeps_identity():
    namespace = load_rank_sync({})
    worker_type = _binding(namespace)
    worker = object.__new__(worker_type)
    assert worker._cohort_shutdown_probe is None
    probe = lambda: True
    worker._cohort_shutdown_probe = probe
    engine = worker._rank_sync_engine
    assert worker._cohort_shutdown_probe is probe
    assert engine._cohort_shutdown_probe_value is probe
    assert "_cohort_shutdown_probe" not in vars(worker)
    assert worker._cohort_shutdown_requested() is True
    worker.set_cohort_shutdown_probe(None)
    assert worker._cohort_shutdown_probe is None
    assert worker._rank_sync_engine is engine
    assert worker._cohort_shutdown_requested() is False


def test_contexts_and_outcomes_are_frozen_and_carry_no_callables():
    namespace = load_rank_sync({})
    for name in ("QuiesceContext", "PortRangeContext", "CommunicatorContext",
                 "PrefillMoeContext", "CommunicatorOutcome", "QuiesceOutcome",
                 "SelfExitOutcome"):
        cls = namespace[name]
        fields = dataclasses.fields(cls)
        value = cls(**{field.name: None for field in fields})
        assert not any(callable(getattr(value, field.name)) for field in fields)
        assert not any("Callable" in str(field.type) for field in fields)
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(value, fields[0].name, object())


def test_recoverable_comm_cleanup_keeps_owned_state_and_returns_the_binding():
    import torch

    def fail_destroy():
        raise RuntimeError("destroy failed")

    def fail_health(*args, **kwargs):
        raise RuntimeError("health failed")

    namespace = load_rank_sync({
        "torch": SimpleNamespace(
            ones=fail_health,
            tensor=lambda value, **kwargs: torch.tensor(value), int32=torch.int32,
            float32=torch.float32,
        ),
        "dist": SimpleNamespace(is_initialized=lambda: False,
                                broadcast=lambda tensor, src: tensor.fill_(-1)),
    })
    engine = namespace["RankSyncEngine"](nccl_port=21000)
    engine.comm = SimpleNamespace(available=True, destroy=fail_destroy)
    engine._nccl_group = object()
    context = namespace["CommunicatorContext"](1, 2, 1, "cuda:1", 21000, 4)
    outcome = engine._check_and_reinit_pynccl(context)
    assert outcome.healthy is False
    assert outcome.comm is engine.comm is None
    assert engine._nccl_group is None
    assert engine._nccl_port == -1


def test_fatal_comm_initialisation_keeps_partial_owned_state(monkeypatch):
    import torch

    group = object()

    def fail_comm(**kwargs):
        raise RuntimeError("communicator failed")

    monkeypatch.setenv("COMM_MASTER_ADDR", "localhost")
    monkeypatch.setenv("BATCHGEN_ENABLE_ALL_TO_ALL", "0")
    namespace = load_rank_sync({
        "torch": SimpleNamespace(
            tensor=lambda value, **kwargs: torch.tensor(value), int32=torch.int32,
            device=lambda *args: "cuda:1",
        ),
        "dist": SimpleNamespace(is_initialized=lambda: False,
                                all_reduce=lambda *args, **kwargs: None,
                                broadcast=lambda *args, **kwargs: None,
                                barrier=lambda: None,
                                ReduceOp=SimpleNamespace(MAX="max")),
        "StatelessProcessGroup": SimpleNamespace(create=lambda **kwargs: group),
        "PyNcclCommunicator": fail_comm,
    })
    engine = namespace["RankSyncEngine"](nccl_port=21000)
    context = namespace["CommunicatorContext"](1, 2, 1, "cuda:1", 21000, 4)
    with pytest.raises(RuntimeError, match="communicator failed"):
        engine._ensure_pynccl_communicator(context)
    assert engine._nccl_group is group
    assert engine._nccl_port == 21000
    assert engine.comm is None
