"""Compile and exercise the CUDA-free shared replica control mapping."""

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "core" / "GPU_Weight_Buffer"

DRIVER = r"""
#include "replica_control_block.h"

#include <cassert>
#include <chrono>
#include <stdexcept>
#include <sys/mman.h>
#include <thread>

using Block = ReplicaControlBlock;
using Barrier = Block::Barrier;

template <typename Ready>
void until(Ready ready) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(5);
    while (!ready()) {
        assert(std::chrono::steady_clock::now() < deadline);
        std::this_thread::yield();
    }
}

int main() {
    const size_t bytes = Block::bytes_required(2, 3, 1);
    void* mapping = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                         MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    assert(mapping != MAP_FAILED);
    const auto initialized = Block::initialize(mapping, bytes, 2, 3, 1);
    assert(initialized.members() == 2);
    assert(initialized.records_per_frame() == 3);
    for (size_t member = 0; member < 2; ++member) {
        assert(initialized.passed(member, 0, Barrier::A) == Block::kNone);
        assert(initialized.passed(member, 0, Barrier::B) == Block::kNone);
        assert(!initialized.aborted(member));
        for (uint64_t frame = 0; frame < 2; ++frame) {
            for (size_t record = 0; record < 3; ++record) {
                assert(initialized.owned_slot(member, frame, record) == Block::kNone);
            }
        }
    }

    const auto run = [mapping, bytes](size_t member) {
        auto block = Block::attach(mapping, bytes);
        const size_t peer = 1 - member;
        // Four generations exercise both parity tables and their reuse.
        for (uint64_t frame = 0; frame < 4; ++frame) {
            for (size_t record = 0; record < 3; ++record) {
                block.publish_slot(member, frame, record,
                                   frame * 100 + member * 10 + record);
            }
            block.advance(member, 0, Barrier::A, frame);
            until([&] { return block.first_waiting(0, Barrier::A, frame) == 2; });
            for (size_t record = 0; record < 3; ++record) {
                assert(block.owned_slot(peer, frame, record) ==
                       frame * 100 + peer * 10 + record);
            }
            block.advance(member, 0, Barrier::B, frame);
            until([&] { return block.first_waiting(0, Barrier::B, frame) == 2; });
        }
        if (member == 0) {
            block.abort(member);
            block.abort(member);
            until([&] { return block.aborted(peer); });
        } else {
            until([&] { return block.aborted(peer); });
            block.abort(member);
        }
    };
    std::thread first(run, 0);
    std::thread second(run, 1);
    first.join();
    second.join();
    auto attached = Block::attach(mapping, bytes);
    assert(attached.passed(0, 0, Barrier::A) == 3);
    assert(attached.passed(1, 0, Barrier::B) == 3);
    assert(attached.aborted(0) && attached.aborted(1));
    assert(attached.owned_slot(0, 2, 1) == 201);
    assert(attached.owned_slot(0, 3, 1) == 301);
    bool refused = false;
    try { attached.advance(0, 0, Barrier::A, 2); }
    catch (const std::logic_error&) { refused = true; }
    assert(refused);
    refused = false;
    try { Block::attach(mapping, bytes - 1); }
    catch (const std::invalid_argument&) { refused = true; }
    assert(refused);
    refused = false;
    try { attached.owned_slot(2, 0, 0); }
    catch (const std::out_of_range&) { refused = true; }
    assert(refused);
    refused = false;
    try { attached.publish_slot(0, 0, 3, 0); }
    catch (const std::out_of_range&) { refused = true; }
    assert(refused);
    refused = false;
    try { Block::bytes_required(0, 1, 1); }
    catch (const std::invalid_argument&) { refused = true; }
    assert(refused);
    refused = false;
    try { Block::bytes_required(1, std::numeric_limits<size_t>::max(), 1); }
    catch (const std::overflow_error&) { refused = true; }
    assert(refused);
    const uint64_t large_frame = (uint64_t{1} << 40) + 4;
    attached.advance(0, 0, Barrier::A, large_frame);
    assert(attached.passed(0, 0, Barrier::A) == large_frame);
    assert(attached.first_waiting(0, Barrier::A, large_frame) == 1);
    attached.advance(1, 0, Barrier::A, large_frame);
    assert(attached.first_waiting(0, Barrier::A, large_frame) == 2);
    refused = false;
    try { Block::bytes_required(std::numeric_limits<size_t>::max(), 1, 1); }
    catch (const std::overflow_error&) { refused = true; }
    assert(refused);
    refused = false;
    try { attached.advance(0, 0, Barrier::A, Block::kNone); }
    catch (const std::invalid_argument&) { refused = true; }
    assert(refused);
    assert(munmap(mapping, bytes) == 0);

    // A singleton uses exactly the same words, tables and barrier operations.
    const size_t one_bytes = Block::bytes_required(1, 1, 1);
    mapping = mmap(nullptr, one_bytes, PROT_READ | PROT_WRITE,
                   MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    assert(mapping != MAP_FAILED);
    auto one = Block::initialize(mapping, one_bytes, 1, 1, 1);
    one.publish_slot(0, 0, 0, 7);
    assert(one.first_waiting(0, Barrier::A, 0) == 0);
    one.advance(0, 0, Barrier::A, 0);
    assert(one.first_waiting(0, Barrier::A, 0) == 1);
    assert(one.owned_slot(0, 0, 0) == 7);
    one.advance(0, 0, Barrier::B, 0);
    assert(one.first_waiting(0, Barrier::B, 0) == 1);
    assert(munmap(mapping, one_bytes) == 0);
}
"""


def test_shared_control_block_two_threads(tmp_path):
    compiler = shutil.which("c++")
    assert compiler is not None, "system c++ is required"
    source = tmp_path / "replica_control_driver.cpp"
    source.write_text(DRIVER)
    binary = tmp_path / "replica_control_driver"
    build = subprocess.run(
        [compiler, "-std=c++17", "-pthread", "-Wall", "-Wextra", "-Werror",
         "-I", str(CONTROL), str(source),
         str(CONTROL / "replica_control_block.cpp"), "-o", str(binary)],
        capture_output=True, text=True, timeout=300,
    )
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
