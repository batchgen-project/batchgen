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
    #include <spdlog/details/log_msg.h>
#endif

#include <spdlog/details/os.h>

namespace spdlog {
namespace details {

SPDLOG_INLINE log_msg::log_msg(spdlog::log_clock::time_point log_time,
                               spdlog::source_loc loc,
                               string_view_t a_logger_name,
                               spdlog::level::level_enum lvl,
                               spdlog::string_view_t msg)
    : logger_name(a_logger_name),
      level(lvl),
      time(log_time)
#ifndef SPDLOG_NO_THREAD_ID
      ,
      thread_id(os::thread_id())
#endif
      ,
      source(loc),
      payload(msg) {
}

SPDLOG_INLINE log_msg::log_msg(spdlog::source_loc loc,
                               string_view_t a_logger_name,
                               spdlog::level::level_enum lvl,
                               spdlog::string_view_t msg)
    : log_msg(os::now(), loc, a_logger_name, lvl, msg) {}

SPDLOG_INLINE log_msg::log_msg(string_view_t a_logger_name,
                               spdlog::level::level_enum lvl,
                               spdlog::string_view_t msg)
    : log_msg(os::now(), source_loc{}, a_logger_name, lvl, msg) {}

}  // namespace details
}  // namespace spdlog
