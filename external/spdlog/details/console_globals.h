/* ---------------------------------------------------------------------------- *
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

// Copyright(c) 2015-present, Gabi Melman & spdlog contributors.
// Distributed under the MIT License (http://opensource.org/licenses/MIT)

#pragma once

#include <mutex>
#include <spdlog/details/null_mutex.h>

namespace spdlog {
namespace details {

struct console_mutex {
    using mutex_t = std::mutex;
    static mutex_t &mutex() {
        static mutex_t s_mutex;
        return s_mutex;
    }
};

struct console_nullmutex {
    using mutex_t = null_mutex;
    static mutex_t &mutex() {
        static mutex_t s_mutex;
        return s_mutex;
    }
};
}  // namespace details
}  // namespace spdlog
