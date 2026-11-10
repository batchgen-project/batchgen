"""The GPU KV pool is one number, measured once after the graph capture.

Sizing used to depend on WHICH decode round you were in. Round 1 measured
``total * frac - used`` before the graph warmup, so the graph footprint landed
in the ``(1 - frac)`` reserve afterwards; rounds >= 2 re-applied the fraction
on top of a steady state that already contained those graphs, charging them
twice and parking ``(1 - frac) * total`` permanently free. On the 2026-11-03
H200 GLM-5.3 server (139.81 GiB, frac 0.93, steady residency 119.58 GiB) that
was 16.27 GB of pool in round 1 and 10.44 GB from round 2 on.

The sizing here measures the steady residency once, after the capture, and
gives the pool everything above it except an explicit headroom constant. The
tests pin the arithmetic, the round-stability (the same number whatever pool
the measurement saw, and the same number every round), the headroom being the
only free remainder, and the fail-loud paths.

Torch-free by construction: ``batchgen.worker.kv_manager`` imports no torch,
and the two worker methods are extracted from the worker source and executed
against fake ``torch``/``dist``/``logging`` modules (the pattern of
tests/unit/test_phase_memory_ledger.py), so the real sizing code runs on a
plain CPU host.
"""

from __future__ import annotations

import ast
import functools
from pathlib import Path
from types import SimpleNamespace

import pytest

from batchgen.worker.kv_manager import GPU_KV_POOL_HEADROOM_GB, KVCacheManager

GIB = 1024 ** 3

# Measured anchors (H200 GLM-5.3 idx21 server, 2026-11-03, frac 0.93):
# device total, steady residency after the whole-model graph capture, and the
# round-1 provisional pool the pre-capture measurement produced.
TOTAL_GB = 139.81
STEADY_GB = 119.58
PROVISIONAL_GB = 16.27
# GLM-5.3 page: 64 tokens at 92,628 B/token (primary + page-split indexer).
BYTES_PER_PAGE = 92628 * 64


# ---------------------------------------------------------------------------
# Pure sizing arithmetic
# ---------------------------------------------------------------------------


def test_final_pool_is_the_device_minus_steady_residency_minus_headroom():
    final = KVCacheManager.compute_final_pool_gb(140.0, 132.4, 19.1)
    steady = 132.4 - 19.1
    assert abs(final - (140.0 - steady - GPU_KV_POOL_HEADROOM_GB)) < 1e-9


def test_the_headroom_is_the_only_memory_left_free():
    # Once the final pool is resident, free memory IS the headroom — no
    # unnamed (1 - frac) remainder nobody chose.
    final = KVCacheManager.compute_final_pool_gb(
        TOTAL_GB, STEADY_GB + PROVISIONAL_GB, PROVISIONAL_GB
    )
    assert abs((TOTAL_GB - STEADY_GB - final) - GPU_KV_POOL_HEADROOM_GB) < 1e-9


def test_the_memory_fraction_is_not_applied_after_the_capture():
    # frac bounds the PRE-capture provisional measurement only; re-applying it
    # to a steady state that already contains the graphs double-charges them.
    final = KVCacheManager.compute_final_pool_gb(
        TOTAL_GB, STEADY_GB + PROVISIONAL_GB, PROVISIONAL_GB
    )
    frac_based = TOTAL_GB * 0.93 - STEADY_GB
    assert final > frac_based
    recovered = TOTAL_GB * (1 - 0.93) - GPU_KV_POOL_HEADROOM_GB
    assert abs((final - frac_based) - recovered) < 1e-9


def test_the_size_does_not_depend_on_the_pool_it_was_measured_against():
    sizes = [
        KVCacheManager.compute_final_pool_gb(TOTAL_GB, STEADY_GB + pool, pool)
        for pool in (1.0, 10.44, PROVISIONAL_GB, 30.0)
    ]
    assert max(sizes) - min(sizes) < 1e-9


def test_re_measuring_with_the_final_pool_returns_the_final_pool():
    # The identity that makes the cache safe: a later round that re-derived
    # the size from its own measurement would land on the same number.
    final = KVCacheManager.compute_final_pool_gb(140.0, 113.3 + 19.1, 19.1)
    again = KVCacheManager.compute_final_pool_gb(140.0, 113.3 + final, final)
    assert abs(final - again) < 1e-9


def test_the_20261103_h200_measurement_funds_18_23_gb():
    final = KVCacheManager.compute_final_pool_gb(
        TOTAL_GB, STEADY_GB + PROVISIONAL_GB, PROVISIONAL_GB
    )
    assert abs(final - 18.23) < 0.01
    # What the two old round classes got, for the record.
    assert abs(PROVISIONAL_GB - 16.27) < 0.01
    assert abs((TOTAL_GB * 0.93 - STEADY_GB) - 10.44) < 0.01


def test_an_explicit_headroom_argument_is_honoured():
    tight = KVCacheManager.compute_final_pool_gb(
        TOTAL_GB, STEADY_GB + 5.0, 5.0, 0.0
    )
    assert abs(tight - (TOTAL_GB - STEADY_GB)) < 1e-9


def test_the_headroom_constant_is_small_and_explicit():
    assert GPU_KV_POOL_HEADROOM_GB == 2.0


@pytest.mark.parametrize(
    "total, used, provisional, headroom",
    [
        (0.0, 10.0, 1.0, 2.0),      # no device
        (140.0, 10.0, 1.0, -1.0),   # negative headroom
        (140.0, 10.0, 10.0, 2.0),   # pool == used: measurement missed the pool
        (140.0, 10.0, 20.0, 2.0),   # pool > used: impossible measurement
    ],
)
def test_impossible_sizing_inputs_fail_loud(total, used, provisional, headroom):
    with pytest.raises(RuntimeError):
        KVCacheManager.compute_final_pool_gb(total, used, provisional, headroom)


def test_pool_pages_for_gb():
    assert KVCacheManager.pool_pages_for_gb(1.0, BYTES_PER_PAGE) == int(
        GIB / BYTES_PER_PAGE
    )
    assert KVCacheManager.pool_pages_for_gb(0.0, BYTES_PER_PAGE) == 0
    assert KVCacheManager.pool_pages_for_gb(-5.0, BYTES_PER_PAGE) == 0
    with pytest.raises(RuntimeError):
        KVCacheManager.pool_pages_for_gb(1.0, 0.0)


def test_pool_floor_pages():
    assert KVCacheManager.pool_floor_pages(128) == 512
    assert KVCacheManager.pool_floor_pages(1) == 4
    assert KVCacheManager.pool_floor_pages(0) == 4  # clamped bucket


def test_cached_pool_fits_when_the_steady_state_held():
    final = KVCacheManager.compute_final_pool_gb(
        TOTAL_GB, STEADY_GB + PROVISIONAL_GB, PROVISIONAL_GB
    )
    # Free memory at the next pool creation: the device minus the steady
    # residency, i.e. the pool plus its headroom.
    KVCacheManager.check_cached_pool_fits(final, TOTAL_GB - STEADY_GB)
    KVCacheManager.check_cached_pool_fits(final, final)  # exact fit


def test_cached_pool_that_no_longer_fits_fails_loud_with_the_shortfall():
    with pytest.raises(RuntimeError) as excinfo:
        KVCacheManager.check_cached_pool_fits(18.23, 17.00, rank=3)
    message = str(excinfo.value)
    assert "Rank 3" in message
    assert "1.23 GB" in message


# ---------------------------------------------------------------------------
# The worker methods, executed against fake torch / dist / logging
# ---------------------------------------------------------------------------

WORKER = Path(__file__).resolve().parents[2] / "batchgen" / "batchgen_worker.py"


@functools.lru_cache(maxsize=1)
def _worker_class():
    tree = ast.parse(WORKER.read_text())
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )


@functools.lru_cache(maxsize=8)
def _method(name):
    return next(
        node
        for node in _worker_class().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


class _Recorder:
    """Stands in for ``logging``; keeps the formatted lines."""

    def __init__(self):
        self.lines = []

    def info(self, fmt, *args):
        self.lines.append(fmt % args if args else fmt)

    warning = info


def _fake_torch(free_gb: float, total_gb: float):
    cuda = SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda device=None: None,
        empty_cache=lambda: None,
        mem_get_info=lambda device: (int(free_gb * GIB), int(total_gb * GIB)),
    )
    return SimpleNamespace(
        cuda=cuda,
        float32="float32",
        tensor=lambda values, dtype=None, device=None: SimpleNamespace(
            item=lambda: values[0]
        ),
    )


def _callable(name, free_gb, total_gb):
    """Compile one worker method against fakes; returns (function, recorder)."""
    module = ast.fix_missing_locations(
        ast.Module(body=[_method(name)], type_ignores=[])
    )
    recorder = _Recorder()
    collections = []
    namespace = {
        "torch": _fake_torch(free_gb, total_gb),
        "gc": SimpleNamespace(collect=lambda: collections.append(1)),
        "dist": SimpleNamespace(broadcast=lambda tensor, src=0: None),
        "logging": recorder,
        "KVCacheManager": KVCacheManager,
        "GPU_KV_POOL_HEADROOM_GB": GPU_KV_POOL_HEADROOM_GB,
    }
    exec(compile(module, str(WORKER), "exec"), namespace)
    return namespace[name], recorder


def _manager(num_total_pages: int):
    return SimpleNamespace(
        is_initialized=True,
        uses_fixed_address_pool=False,
        get_stats=lambda: SimpleNamespace(num_total_pages=num_total_pages),
    )


def _finalize_worker(provisional_gb=PROVISIONAL_GB, max_bucket=128):
    pages = int(provisional_gb * GIB / BYTES_PER_PAGE)
    return SimpleNamespace(
        rank=0,
        local_rank=0,
        torch_device="cuda:0",
        gpu_memory_frac=0.93,
        gpu_kv_cache_size_gb=provisional_gb,
        _gpu_kv_sizing_provisional=True,
        _gpu_kv_cached_pool_gb=None,
        gpu_paged_kv_cache_manager=_manager(pages),
        _whole_model_bucketing=SimpleNamespace(bucket_sizes=[1, 16, 64, max_bucket]),
    )


def test_finalize_caches_the_measured_size_and_reports_the_headroom():
    used = STEADY_GB + PROVISIONAL_GB
    finalize, recorder = _callable(
        "_finalize_gpu_kv_pool_after_capture", TOTAL_GB - used, TOTAL_GB
    )
    worker = _finalize_worker()
    finalize(worker, worker.gpu_paged_kv_cache_manager)

    assert abs(worker._gpu_kv_cached_pool_gb - 18.23) < 0.01
    # The live round keeps its provisional pool (captured graphs bake its
    # addresses and this codebase refuses to recapture).
    assert worker.gpu_kv_cache_size_gb == PROVISIONAL_GB
    assert worker._gpu_kv_sizing_provisional is False

    log = "\n".join(recorder.lines)
    assert "Finalized pool size: 18.23 GB" in log
    # Pages at the live pool's measured bytes-per-page (the pool's own page
    # count divided into its size), not a re-derived profile constant.
    provisional_pages = int(PROVISIONAL_GB * GIB / BYTES_PER_PAGE)
    bytes_per_page = PROVISIONAL_GB * GIB / provisional_pages
    expected_pages = int(worker._gpu_kv_cached_pool_gb * GIB / bytes_per_page)
    assert expected_pages == 3300
    assert f"~{expected_pages} pages" in log
    assert expected_pages > provisional_pages  # round 1 was the smaller pool
    assert "steady used 119.58 GB" in log
    assert "headroom 2.00 GB" in log
    # The frac-based size the old formula would have cached, for the A/B.
    assert "10.44 GB" in log


def test_finalize_is_idempotent_so_a_later_capture_cannot_resize_the_pool():
    finalize, _ = _callable(
        "_finalize_gpu_kv_pool_after_capture",
        TOTAL_GB - (STEADY_GB + PROVISIONAL_GB),
        TOTAL_GB,
    )
    worker = _finalize_worker()
    finalize(worker, worker.gpu_paged_kv_cache_manager)
    cached = worker._gpu_kv_cached_pool_gb
    # Second call (e.g. a later warmup entry): the measure-once flag is down.
    worker.gpu_kv_cache_size_gb = 1.0
    finalize(worker, worker.gpu_paged_kv_cache_manager)
    assert worker._gpu_kv_cached_pool_gb == cached


def test_finalize_fails_loud_when_the_steady_state_leaves_no_usable_pool():
    # Residency grew to within 2.1 GB of the device: the remainder after the
    # headroom cannot even hold the largest decode bucket.
    used = TOTAL_GB - 2.1
    finalize, _ = _callable(
        "_finalize_gpu_kv_pool_after_capture", TOTAL_GB - used, TOTAL_GB
    )
    # Provisional pool of 1 GB inside that residency: the 1.1 GB remainder is
    # ~199 pages, below the 512 the largest bucket needs.
    worker = _finalize_worker(provisional_gb=1.0)
    with pytest.raises(RuntimeError) as excinfo:
        finalize(worker, worker.gpu_paged_kv_cache_manager)
    message = str(excinfo.value)
    assert "below floor" in message
    assert "512 pages" in message  # bucket 128 * 4 pages
    assert "2.00 GB pool headroom" in message
    assert worker._gpu_kv_cached_pool_gb is None


def test_finalize_does_not_pass_the_memory_fraction_to_the_sizing_helper():
    # Structural guard: re-introducing frac here is the double-charge the
    # headroom constant replaced.
    call = next(
        node
        for node in ast.walk(_method("_finalize_gpu_kv_pool_after_capture"))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "compute_final_pool_gb"
    )
    args = [ast.unparse(arg) for arg in call.args]
    assert args == ["total_mem_gb", "used_mem_gb", "provisional_gb"]


def test_the_provisional_measurement_still_bounds_itself_with_the_fraction():
    # Before the capture the graph footprint is unallocated, so (1 - frac) is
    # the only thing covering it; that is the one load-bearing use of frac.
    body = ast.unparse(_method("_init_gpu_kv_with_actual_size"))
    assert "KVCacheManager.provisional_pool_gb(total_mem_gb, used_mem_gb, self.gpu_memory_frac)" in body


def _cached_path_worker(cached_gb):
    created = []

    worker = SimpleNamespace(
        rank=0,
        local_rank=0,
        torch_device="cuda:0",
        _phase="decode",
        gpu_memory_frac=0.93,
        gpu_kv_cache_size_gb=None,
        _gpu_kv_cached_pool_gb=cached_gb,
        _gpu_kv_sizing_provisional=False,
        gpu_paged_kv_cache_manager=None,
    )

    def _init_fixed_size():
        created.append(worker.gpu_kv_cache_size_gb)
        pages = int(worker.gpu_kv_cache_size_gb * GIB / BYTES_PER_PAGE)
        worker.gpu_paged_kv_cache_manager = _manager(pages)

    worker._initialize_gpu_kv_manager_fixed_size = _init_fixed_size
    return worker, created


def test_every_later_round_creates_the_pool_at_the_one_cached_size():
    cached = KVCacheManager.compute_final_pool_gb(
        TOTAL_GB, STEADY_GB + PROVISIONAL_GB, PROVISIONAL_GB
    )
    sizes = []
    # Rounds 2 and 3, with different free memory each time (prefill leaves a
    # different allocator state behind): the pool must not follow it.
    for free_gb in (TOTAL_GB - STEADY_GB, TOTAL_GB - STEADY_GB - 1.5):
        init, recorder = _callable(
            "_init_gpu_kv_with_actual_size", free_gb, TOTAL_GB
        )
        worker, created = _cached_path_worker(cached)
        init(worker)
        sizes.append(created)
        assert "Cached pool size" in "\n".join(recorder.lines)
    assert sizes == [[cached], [cached]]


def test_a_cached_pool_that_no_longer_fits_fails_loud_instead_of_shrinking():
    cached = KVCacheManager.compute_final_pool_gb(
        TOTAL_GB, STEADY_GB + PROVISIONAL_GB, PROVISIONAL_GB
    )
    # Residency grew by 3 GB after the measurement: more than the headroom.
    init, _ = _callable(
        "_init_gpu_kv_with_actual_size", cached - 1.0, TOTAL_GB
    )
    worker, created = _cached_path_worker(cached)
    with pytest.raises(RuntimeError) as excinfo:
        init(worker)
    assert "no longer fits" in str(excinfo.value)
    assert created == []  # no pool was created at a silently smaller size
