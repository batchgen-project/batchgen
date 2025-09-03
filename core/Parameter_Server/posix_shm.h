// clang-format off
/* ----------------------------------------------------------------------------  *
 *  BatchGen                                                                      *
 *  copyright (c) EfficientMoE team 2025                                             *
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
void* allocate_shared_pinned_memory(const std::string& shm_name, int64_t size,
                                    bool create, bool enable_hugetlbfs);
void free_shared_pinned_memory(std::string& shm_name, void* ptr, int64_t size,
                               bool create);
void serialize_to_shared_memory(
    const std::unordered_map<std::string,
                             std::unordered_map<std::string, tensor_meta>>& map,
    const std::string& shm_name);
std::unordered_map<std::string, std::unordered_map<std::string, tensor_meta>>
deserialize_from_shared_memory(const std::string& shm_name);
