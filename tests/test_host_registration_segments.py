"""Weights are registered with CUDA in tensor-aligned segments.

One cudaHostRegister call above (INT_MAX / 16) base pages fails on some
driver/kernel pairs (the driver's per-registration page table exceeds the
kernel's kvzalloc limit), and a copy whose source spans two registrations fails with
cudaErrorInvalidValue. So the worker registers the weight mapping in segments
of at most MAX_HOST_REGISTRATION_BYTES, cut only at page boundaries that no
tensor crosses.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
POSIX_SHM = ROOT / "core" / "Parameter_Server" / "posix_shm.cpp"
POSIX_SHM_H = ROOT / "core" / "Parameter_Server" / "posix_shm.h"
WEIGHTS_STORAGE = ROOT / "core" / "Weights_Storage" / "Weights_Storage.cpp"

PAGE = 4096
GIB = 1 << 30


def _section(source, begin, end):
    start = source.index(begin)
    return source[start:source.index(end, start)]


def _core_engine():
    try:
        from batchgen.native_loader import core_engine
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"compiled extension unavailable: {exc}")
    if not hasattr(core_engine, "plan_host_registration_segments"):
        pytest.skip("compiled extension predates plan_host_registration_segments")
    return core_engine


def _assert_valid_plan(segments, extents, total, max_segment):
    assert segments[0][0] == 0
    assert segments[-1][1] == total
    for (begin, end), (next_begin, _) in zip(segments, segments[1:]):
        assert end == next_begin
    for begin, end in segments:
        assert 0 < end - begin <= max_segment
    for begin, _ in segments[1:]:
        assert begin % PAGE == 0
        for tensor_begin, tensor_end in extents:
            assert not (tensor_begin < begin < tensor_end), (begin, tensor_begin, tensor_end)


# --------------------------------------------------------------------------
# Source wiring (runs everywhere)
# --------------------------------------------------------------------------


def test_allocator_never_registers_with_cuda():
    allocate = _section(
        POSIX_SHM.read_text(),
        "void* allocate_shared_pinned_memory(",
        "std::vector<std::pair<int64_t, int64_t>> plan_host_registration_segments(",
    )
    assert "cudaHostRegister(" not in allocate
    assert "pin_for_cuda" not in POSIX_SHM_H.read_text()


def test_ceiling_is_half_the_measured_driver_limit():
    assert "constexpr int64_t kMaxHostRegistrationBytes = int64_t{256} << 30;" in (
        POSIX_SHM_H.read_text()
    )


def test_worker_init_registers_planned_segments_and_records_each():
    source = WEIGHTS_STORAGE.read_text()
    init = _section(source, "void Weights_Storage::Init(", "// void Weights_Storage::Init(")
    plan = init.index("plan_host_registration_segments(")
    register = init.index("cudaHostRegister(address, bytes,")
    record = init.index("this->registered_ranges_.emplace_back(address, bytes);")
    assert plan < register < record
    assert "kMaxHostRegistrationBytes" in init
    assert "meta.offset + meta.byte_size" in init
    # A failed segment releases the earlier ones and the mapping before throwing.
    failure = _section(init, "if (err != cudaSuccess) {", "this->registered_ranges_.emplace_back")
    assert "cudaHostUnregister(range->first);" in failure
    assert "unmap_shared_memory(" in failure
    assert "throw std::runtime_error" in failure


def test_destructor_unregisters_every_segment_before_unmapping():
    source = WEIGHTS_STORAGE.read_text()
    destructor = _section(source, "Weights_Storage::~Weights_Storage() {", "if (this->compact_ptr_ != nullptr)")
    assert destructor.index("cudaHostUnregister(range->first);") < destructor.index(
        "unmap_shared_memory(this->log_level_, this->weight_ptr_,"
    )


# --------------------------------------------------------------------------
# Planner behaviour (needs the compiled extension)
# --------------------------------------------------------------------------


def test_region_within_ceiling_is_one_segment():
    core = _core_engine()
    extents = [(0, 100), (100, 5000)]
    assert core.plan_host_registration_segments(extents, 8192, 16 * PAGE, PAGE) == [(0, 8192)]


def test_cuts_land_on_aligned_tensor_boundaries():
    core = _core_engine()
    # 40 page-aligned tensors of 3 pages each; ceiling 10 pages -> cut at 9 pages.
    extents = [(i * 3 * PAGE, (i + 1) * 3 * PAGE) for i in range(40)]
    total = 120 * PAGE
    segments = core.plan_host_registration_segments(extents, total, 10 * PAGE, PAGE)
    _assert_valid_plan(segments, extents, total, 10 * PAGE)
    assert segments[0] == (0, 9 * PAGE)


def test_unaligned_boundaries_need_a_gap_page():
    core = _core_engine()
    # Tensors packed without padding, never touching a page boundary except at
    # the padded file start of 8 * PAGE: the only admissible interior cut.
    extents = [(1, 3 * PAGE + 7), (3 * PAGE + 7, 8 * PAGE - 5), (8 * PAGE, 13 * PAGE + 9)]
    total = 14 * PAGE
    segments = core.plan_host_registration_segments(extents, total, 10 * PAGE, PAGE)
    assert segments == [(0, 8 * PAGE), (8 * PAGE, total)]


def test_tensor_larger_than_ceiling_is_refused():
    core = _core_engine()
    with pytest.raises(RuntimeError, match="exceeds the per-registration ceiling"):
        core.plan_host_registration_segments([(0, 11 * PAGE)], 20 * PAGE, 10 * PAGE, PAGE)


def test_no_admissible_cut_is_refused():
    core = _core_engine()
    # Two tensors whose shared boundary is not page aligned: every page
    # boundary in (0, 10 pages] lies inside one of them.
    extents = [(0, 6 * PAGE + 1), (6 * PAGE + 1, 12 * PAGE)]
    with pytest.raises(RuntimeError, match="no tensor-free page boundary"):
        core.plan_host_registration_segments(extents, 12 * PAGE, 10 * PAGE, PAGE)


def test_extent_outside_region_is_refused():
    core = _core_engine()
    with pytest.raises(RuntimeError, match="lies outside"):
        core.plan_host_registration_segments([(0, 9 * PAGE)], 8 * PAGE, 4 * PAGE, PAGE)


def test_random_layouts_satisfy_the_invariants():
    core = _core_engine()
    rng = random.Random(20261110)
    for _ in range(200):
        extents = []
        cursor = 0
        for _ in range(rng.randint(1, 60)):
            if rng.random() < 0.3:  # file padding up to the next page
                cursor = -(-cursor // PAGE) * PAGE
            size = rng.randint(1, 3 * PAGE)
            extents.append((cursor, cursor + size))
            cursor += size
        total = cursor + rng.randint(0, PAGE)
        max_segment = rng.randint(3, 12) * PAGE
        rng.shuffle(extents)
        try:
            segments = core.plan_host_registration_segments(extents, total, max_segment, PAGE)
        except RuntimeError as exc:
            assert "no tensor-free page boundary" in str(exc)
            continue
        _assert_valid_plan(segments, extents, total, max_segment)


def test_k3_scale_layout_plans_without_cutting_a_tensor():
    core = _core_engine()
    # 96 page-padded files of 243 tensors each, a multi-segment layout.
    extents = []
    cursor = 0
    for _ in range(96):
        cursor = -(-cursor // PAGE) * PAGE
        for _ in range(243):
            extents.append((cursor, cursor + 64 * (1 << 20) + 3))
            cursor += 64 * (1 << 20) + 3
    segments = core.plan_host_registration_segments(
        extents, cursor, core.MAX_HOST_REGISTRATION_BYTES, PAGE
    )
    _assert_valid_plan(segments, extents, cursor, core.MAX_HOST_REGISTRATION_BYTES)
    assert core.MAX_HOST_REGISTRATION_BYTES == 256 * GIB
    assert len(segments) >= -(-cursor // core.MAX_HOST_REGISTRATION_BYTES)
