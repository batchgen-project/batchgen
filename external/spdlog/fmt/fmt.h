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

//
// Copyright(c) 2016-2018 Gabi Melman.
// Distributed under the MIT License (http://opensource.org/licenses/MIT)
//

#pragma once

//
// Include a bundled header-only copy of fmtlib or an external one.
// By default spdlog include its own copy.
//
#include <spdlog/tweakme.h>

#if defined(SPDLOG_USE_STD_FORMAT)  // SPDLOG_USE_STD_FORMAT is defined - use std::format
    #include <format>
#elif !defined(SPDLOG_FMT_EXTERNAL)
    #if !defined(SPDLOG_COMPILED_LIB) && !defined(FMT_HEADER_ONLY)
        #define FMT_HEADER_ONLY
    #endif
    #ifndef FMT_USE_WINDOWS_H
        #define FMT_USE_WINDOWS_H 0
    #endif

    #include <spdlog/fmt/bundled/core.h>
    #include <spdlog/fmt/bundled/format.h>

#else  // SPDLOG_FMT_EXTERNAL is defined - use external fmtlib
    #include <fmt/core.h>
    #include <fmt/format.h>
#endif
