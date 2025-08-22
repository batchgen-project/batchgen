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

#ifndef SPDLOG_HEADER_ONLY
    #include <spdlog/sinks/stdout_color_sinks.h>
#endif

#include <spdlog/common.h>
#include <spdlog/logger.h>

namespace spdlog {

template <typename Factory>
SPDLOG_INLINE std::shared_ptr<logger> stdout_color_mt(const std::string &logger_name,
                                                      color_mode mode) {
    return Factory::template create<sinks::stdout_color_sink_mt>(logger_name, mode);
}

template <typename Factory>
SPDLOG_INLINE std::shared_ptr<logger> stdout_color_st(const std::string &logger_name,
                                                      color_mode mode) {
    return Factory::template create<sinks::stdout_color_sink_st>(logger_name, mode);
}

template <typename Factory>
SPDLOG_INLINE std::shared_ptr<logger> stderr_color_mt(const std::string &logger_name,
                                                      color_mode mode) {
    return Factory::template create<sinks::stderr_color_sink_mt>(logger_name, mode);
}

template <typename Factory>
SPDLOG_INLINE std::shared_ptr<logger> stderr_color_st(const std::string &logger_name,
                                                      color_mode mode) {
    return Factory::template create<sinks::stderr_color_sink_st>(logger_name, mode);
}
}  // namespace spdlog
