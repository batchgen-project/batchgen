"""CPU checks of the report-only startup decode transient probe.

Covers the measurement arithmetic and determinism gate against fake device
counters, the real worker probe method (loaded from source) with and without
CUDA, and the publication path the deployment's max_model_len already uses:
worker main loop -> scheduler -> /v1/models.
"""

import ast
import asyncio
import importlib.util
import logging
import queue
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from batchgen.worker import startup_transient as st
from tests.test_batch_in_progress_failure import bs
from tests.worker_method_source import load_worker_methods

ROOT = Path(__file__).resolve().parents[2]
MiB = 1024 * 1024


class FakeCuda:
    """Allocator and driver counters a test step mutates directly."""

    def __init__(self, total=64 * 1024 * MiB):
        self.total = total
        self.allocated = 10 * MiB
        self.peak = self.allocated
        self.reserved = 12 * MiB
        self.external = 100 * MiB
        self.syncs = 0
        self.resets = 0

    def synchronize(self, device):
        self.syncs += 1

    def reset_peak_memory_stats(self, device):
        self.resets += 1
        self.peak = self.allocated

    def mem_get_info(self, device):
        used = self.reserved + self.external
        return self.total - used, self.total

    def memory_allocated(self, device):
        return self.allocated

    def max_memory_allocated(self, device):
        return self.peak

    def memory_reserved(self, device):
        return self.reserved

    def transient(self, peak_bytes, reserved_growth=0, external_growth=0):
        """Allocate ``peak_bytes`` and free them, as one decode step would."""
        self.peak = max(self.peak, self.allocated + peak_bytes)
        self.reserved += reserved_growth
        self.external += external_growth


def _probe(**kwargs):
    defaults = dict(seam=st.SEAM_DEPLOYMENT_PREPARE, padding_bsz=8)
    defaults.update(kwargs)
    return defaults


# ---- determinism gate (pure) ------------------------------------------------

def test_runs_agree_within_exactly_one_allocator_block():
    block = st.ALLOCATOR_BLOCK_BYTES
    assert st.transient_runs_agree(5 * block, 5 * block)
    assert st.transient_runs_agree(5 * block, 6 * block)
    assert st.transient_runs_agree(6 * block, 5 * block)
    assert not st.transient_runs_agree(5 * block, 6 * block + 1)
    assert not st.transient_runs_agree(6 * block + 1, 5 * block)


# ---- measurement arithmetic ---------------------------------------------------

def test_eager_measures_exact_peak_and_external_growth_over_both_runs():
    cuda = FakeCuda()
    runs = []

    def step():
        # First run creates a lazy communicator buffer; the second finds it.
        cuda.transient(64 * MiB, reserved_growth=4 * MiB,
                       external_growth=0 if runs else 32 * MiB)
        runs.append(1)

    result = st.measure_eager(step, cuda=cuda, device=0, **_probe())
    assert len(runs) == 2 and cuda.resets == 2
    assert result.flavor == st.FLAVOR_EAGER
    assert result.allocator_gib * st.GIB == 64 * MiB
    # reserved growth is allocator memory and is not counted as external
    assert result.external_gib * st.GIB == 32 * MiB
    assert result.deterministic is True and result.unavailable is None


def test_eager_mismatch_logs_loudly_and_does_not_fail_startup(caplog):
    cuda = FakeCuda()
    peaks = iter([64 * MiB, 64 * MiB + st.ALLOCATOR_BLOCK_BYTES + 1])
    with caplog.at_level(logging.ERROR, logger=st.__name__):
        result = st.measure_eager(
            lambda: cuda.transient(next(peaks)), cuda=cuda, device=0, **_probe()
        )
    assert result.deterministic is False
    assert result.allocator_gib * st.GIB == 64 * MiB + st.ALLOCATOR_BLOCK_BYTES + 1
    assert any("not deterministic" in r.getMessage() for r in caplog.records)


def test_capture_runs_once_and_is_not_gated():
    cuda = FakeCuda()
    calls = []

    def capture():
        calls.append(1)
        cuda.transient(128 * MiB, reserved_growth=96 * MiB, external_growth=8 * MiB)

    result = st.measure_capture(capture, cuda=cuda, device=0, **_probe())
    assert calls == [1]
    assert result.allocator_gib * st.GIB == 128 * MiB
    assert result.external_gib * st.GIB == 8 * MiB
    assert result.deterministic is None


def test_struct_is_measured_or_unavailable_never_both():
    with pytest.raises(ValueError):
        st.StartupTransient(st.FLAVOR_EAGER, "s", 8)
    with pytest.raises(ValueError):
        st.StartupTransient(st.FLAVOR_EAGER, "s", 8, allocator_gib=1.0,
                            external_gib=0.0, unavailable="x")
    with pytest.raises(ValueError):
        st.StartupTransient("prefill", "s", 8, unavailable="x")


# ---- cross-rank reduction ------------------------------------------------------

def _measured(flavor, allocator, external, deterministic=None):
    return st.StartupTransient(flavor, st.SEAM_PERSISTENT_BUILD, 8,
                               allocator_gib=allocator, external_gib=external,
                               deterministic=deterministic)


def test_reduction_takes_the_tightest_rank_per_flavor():
    reduced = st.reduce_across_ranks([
        [_measured(st.FLAVOR_EAGER, 1.0, 0.5, True), _measured(st.FLAVOR_CAPTURE, 2.0, 0.1)],
        [_measured(st.FLAVOR_EAGER, 1.5, 0.25, False), _measured(st.FLAVOR_CAPTURE, 1.0, 0.3)],
    ])
    assert [(r.allocator_gib, r.external_gib, r.deterministic) for r in reduced] == [
        (1.5, 0.5, False), (2.0, 0.3, None)]


def test_reduction_is_unavailable_when_any_rank_is_and_names_it():
    missing = st.StartupTransient.missing(st.FLAVOR_EAGER, st.SEAM_PERSISTENT_BUILD, 8, "why")
    (reduced,) = st.reduce_across_ranks([
        [_measured(st.FLAVOR_EAGER, 1.0, 0.5, True)], [missing]])
    assert reduced.unavailable == "rank 1: why" and reduced.allocator_gib is None


def test_reduction_refuses_ranks_that_probed_different_configurations():
    other = st.StartupTransient(st.FLAVOR_EAGER, st.SEAM_PERSISTENT_BUILD, 16,
                                allocator_gib=1.0, external_gib=0.0)
    with pytest.raises(ValueError, match="padding_bsz"):
        st.reduce_across_ranks([[_measured(st.FLAVOR_EAGER, 1.0, 0.0)], [other]])


# ---- the real worker probe ------------------------------------------------------

def _worker_probe(*, cuda_available, cuda=None, dist=None):
    torch = SimpleNamespace(cuda=cuda if cuda is not None else SimpleNamespace())
    torch.cuda.is_available = lambda: cuda_available
    dist = dist or SimpleNamespace(is_initialized=lambda: False)
    return load_worker_methods(["_probe_startup_transient"], {
        "torch": torch, "dist": dist, "_startup_transient": st,
    })["_probe_startup_transient"]


def _worker(**kwargs):
    values = dict(rank=0, local_rank=0, world_size=1, _phase="decode",
                  _decode_padding_bsz=8, parallel_manager=SimpleNamespace(),
                  startup_transient=None)
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_probe_without_cuda_publishes_unavailable_and_still_captures(caplog):
    probe = _worker_probe(cuda_available=False)
    calls = []
    worker = _worker()
    with caplog.at_level(logging.INFO, logger=st.__name__):
        probe(worker, seam=st.SEAM_PERSISTENT_BUILD, capture=lambda: calls.append(1))
    assert calls == [1]
    assert worker.startup_transient == [
        {"flavor": "eager", "seam": st.SEAM_PERSISTENT_BUILD, "padding_bsz": 8,
         "allocator_gib": None, "external_gib": None, "deterministic": None,
         "unavailable": "CUDA is unavailable"},
        {"flavor": "capture", "seam": st.SEAM_PERSISTENT_BUILD, "padding_bsz": 8,
         "allocator_gib": None, "external_gib": None, "deterministic": None,
         "unavailable": "CUDA is unavailable"},
    ]
    lines = [r.getMessage() for r in caplog.records]
    assert any(line.startswith("[STARTUP_TRANSIENT] rank=0 ") for line in lines)
    assert any(line.startswith("[STARTUP_TRANSIENT] deployment ") for line in lines)


def test_probe_names_a_lifecycle_without_startup_capture():
    probe = _worker_probe(cuda_available=False)
    worker = _worker()
    probe(worker, seam=st.SEAM_DEPLOYMENT_PREPARE, capture=None)
    assert worker.startup_transient[1]["unavailable"] == (
        "this lifecycle captures no decode graph at startup")


def test_probe_refuses_to_run_outside_the_configured_decode_phase():
    probe = _worker_probe(cuda_available=False)
    with pytest.raises(RuntimeError, match="decode phase"):
        probe(_worker(_phase="prefill"), seam=st.SEAM_DEPLOYMENT_PREPARE, capture=None)
    with pytest.raises(RuntimeError, match="padding_bsz=None"):
        probe(_worker(_decode_padding_bsz=None), seam=st.SEAM_DEPLOYMENT_PREPARE, capture=None)


def test_probe_with_cuda_measures_the_model_step_and_reduces_over_ranks():
    cuda = FakeCuda()
    seen = []

    class Manager:
        def startup_decode_probe_step(self, padding_bsz):
            seen.append(padding_bsz)
            cuda.transient(64 * MiB)

    def all_gather_object(out, local):
        # The peer rank needs more for the same flavors.
        peer = [
            st.StartupTransient(item.flavor, item.seam, item.padding_bsz,
                                allocator_gib=item.allocator_gib * 2,
                                external_gib=item.external_gib,
                                deterministic=item.deterministic)
            if item.unavailable is None else item
            for item in local
        ]
        out[:] = [local, peer]

    dist = SimpleNamespace(is_initialized=lambda: True, all_gather_object=all_gather_object)
    probe = _worker_probe(cuda_available=True, cuda=cuda, dist=dist)
    worker = _worker(world_size=2, parallel_manager=Manager())
    probe(worker, seam=st.SEAM_DEPLOYMENT_PREPARE, capture=None)
    assert seen == [8, 8]
    eager, captured = worker.startup_transient
    assert eager["allocator_gib"] * st.GIB == 2 * 64 * MiB
    assert eager["deterministic"] is True and eager["unavailable"] is None
    assert captured["unavailable"] == "this lifecycle captures no decode graph at startup"


def test_probe_with_cuda_survives_a_failing_step_and_names_the_failure(caplog):
    # Report-only contract: a defect in one model's synthetic step must not
    # kill its deployments; the eager flavor publishes the failure by name.
    cuda = FakeCuda()

    class Manager:
        def startup_decode_probe_step(self, padding_bsz):
            raise RuntimeError("synthetic step blew up")

    dist = SimpleNamespace(is_initialized=lambda: False, all_gather_object=None)
    probe = _worker_probe(cuda_available=True, cuda=cuda, dist=dist)
    worker = _worker(world_size=1, parallel_manager=Manager())
    import logging as _logging
    with caplog.at_level(_logging.ERROR, logger=st.__name__):
        probe(worker, seam=st.SEAM_DEPLOYMENT_PREPARE, capture=None)
    eager, captured = worker.startup_transient
    assert eager["unavailable"] == "probe step failed: RuntimeError: synthetic step blew up"
    assert eager["allocator_gib"] is None
    assert captured["unavailable"] == "this lifecycle captures no decode graph at startup"
    assert any("eager probe step failed" in r.message for r in caplog.records)


def test_probe_with_cuda_names_a_manager_without_the_step():
    class GptOssManager:
        pass

    probe = _worker_probe(cuda_available=True, cuda=FakeCuda())
    worker = _worker(parallel_manager=GptOssManager())
    probe(worker, seam=st.SEAM_DEPLOYMENT_PREPARE, capture=None)
    assert worker.startup_transient[0]["unavailable"] == (
        "GptOssManager provides no cap-sized synthetic decode step")


# ---- publication: same channel as max_model_len ---------------------------------

def _published():
    return [
        st.StartupTransient.missing(flavor, st.SEAM_DEPLOYMENT_PREPARE, 8, "CUDA is unavailable").as_dict()
        for flavor in st.FLAVORS
    ]


def test_main_loop_publishes_the_transient_in_the_capacity_message():
    tree = ast.parse((ROOT / "batchgen/server_worker_main_loop.py").read_text())
    (message,) = [
        node for node in ast.walk(tree) if isinstance(node, ast.Dict)
        and any(isinstance(v, ast.Constant) and v.value == "deployment_capacity"
                for v in node.values)
    ]
    entries = {k.value: ast.unparse(v) for k, v in zip(message.keys, message.values)}
    assert entries["startup_transient"] == "worker.startup_transient"
    assert entries["max_model_len"] == "worker.model_context_length"


def test_scheduler_requires_the_transient_with_the_length_and_clears_both():
    async def run():
        worker = SimpleNamespace(response_queue=queue.Queue())
        scheduler = bs.BatchScheduler(SimpleNamespace(), worker, SimpleNamespace())
        await scheduler.start()
        try:
            worker.response_queue.put({"type": "deployment_capacity", "max_model_len": 129})
            await asyncio.sleep(0.2)
            assert not scheduler._deployment_ready.is_set()
            assert scheduler.max_model_len is None
            worker.response_queue.put({"type": "deployment_capacity", "max_model_len": 129,
                                       "startup_transient": _published()})
            await asyncio.wait_for(scheduler.wait_for_deployment(), timeout=5)
            assert scheduler.max_model_len == 129
            assert [t.model_dump() for t in scheduler.startup_transient] == _published()
            scheduler._clear_deployment()
            assert scheduler.startup_transient is None and scheduler.max_model_len is None
        finally:
            await scheduler.stop()
            worker.response_queue.put(None)

    asyncio.run(run())


def _io_struct():
    spec = importlib.util.spec_from_file_location(
        "startup_transient_io_struct", ROOT / "batchgen/server/io_struct.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_models_metadata_carries_the_published_transient():
    io = _io_struct()

    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            super().__init__(detail)
            self.status_code = status_code

    tree = ast.parse((ROOT / "batchgen/server/http_server.py").read_text())
    (node,) = [n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "published_model_object"]
    scope = dict(Request=object, HTTPException=HTTPException, ModelObject=io.ModelObject)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "http_server.py", "exec"), scope)
    scheduler = SimpleNamespace(
        max_model_len=129,
        startup_transient=[io.StartupTransientObject.model_validate(t) for t in _published()],
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        scheduler=scheduler,
        model_object=io.ModelObject(id="m", created=1, max_context_length=1),
    )))
    body = scope["published_model_object"](request).model_dump()
    assert body["max_model_len"] == 129
    assert body["startup_transient"] == _published()
    scheduler.max_model_len = None
    with pytest.raises(HTTPException) as info:
        scope["published_model_object"](request)
    assert info.value.status_code == 503
