// clang-format off
/* ----------------------------------------------------------------------------  *
 *  BatchGen                                                                      *
 *  Copyright (c) 2025-2026 BatchGen Team                                            *
 *                                                                               *
 *  licensed under the apache license, version 2.0 (the "license");              *
 *  you may not use this file except in compliance with the license.             *
 *                                                                               *
 *  you may obtain a copy of the license at                                      *
 *                                                                               *
 *                  http://www.apache.org/licenses/license-2.0                   *
 *                                                                               *
 *  unless required by applicable law or agreed to in writing, software          *
 *  distributed under the license is distributed on an "as is" basis,            *
 *  without warranties or conditions of any kind, either express or implied.     *
 *  see the license for the specific language governing permissions and          *
 *  limitations under the license.                                               *
 * ---------------------------------------------------------------------------- */
// clang-format on

#pragma once
#include "Parameter_Server.h"
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Upper bound on the bytes one cudaHostRegister call may cover. The NVIDIA
// kernel module keeps a 16-byte entry per 4 KiB page of a registration in one
// kvzalloc'd table, and the kernel refuses a kvzalloc above INT_MAX, so on
// driver/kernel pairs without the vmalloc fallback a single registration
// above (INT_MAX / 16) base pages fails with cudaErrorMemoryAllocation
// ("NVRM: failed to allocate page table"), whether or not THP backs the
// pages. Half that ceiling leaves margin for drivers that count the table
// differently.
constexpr int64_t kMaxHostRegistrationBytes = int64_t{256} << 30;

// Split [0, total_bytes) into consecutive registration segments of at most
// max_segment_bytes. Every interior cut is page_bytes-aligned and lies outside
// every extent in tensor_extents ([begin, end) byte ranges), so no tensor
// straddles two registrations: a copy whose source spans two registrations
// fails with cudaErrorInvalidValue. Throws std::runtime_error when an extent
// lies outside the region, is itself larger than max_segment_bytes, or when
// no admissible cut exists within max_segment_bytes of a segment start.
std::vector<std::pair<int64_t, int64_t>> plan_host_registration_segments(
    std::vector<std::pair<int64_t, int64_t>> tensor_extents,
    int64_t total_bytes, int64_t max_segment_bytes, int64_t page_bytes);

// log_level is the configured level of the component that owns this
// allocation; the shared-memory logger is created from it on first use.
// The mapping is never registered with CUDA here: the owner registers it in
// segments planned by plan_host_registration_segments.
void* allocate_shared_pinned_memory(const std::string& log_level,
                                    const std::string& shm_name, int64_t size,
                                    bool create, bool enable_hugetlbfs,
                                    bool enable_thp = false,
                                    int memfd_creator_pid = -1,
                                    int memfd_fd = -1,
                                    int* out_memfd_fd = nullptr,
                                    int64_t* out_mapped_size = nullptr);
// Unmap a region from allocate_shared_pinned_memory. The owner releases its
// CUDA registrations first.
void unmap_shared_memory(const std::string& log_level, void* ptr,
                         int64_t size);
// Returns the creator's memfd, which must stay open for attachers to reach it
// through /proc/<creator_pid>/fd/<N>.
int serialize_to_memfd(
    const std::unordered_map<std::string,
                             std::unordered_map<std::string, tensor_meta>>& map);
std::unordered_map<std::string, std::unordered_map<std::string, tensor_meta>>
deserialize_from_memfd(int memfd_creator_pid, int memfd_fd);
