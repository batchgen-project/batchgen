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

#include <torch/torch.h>
// #include "caching_allocator.h"
#include <cstdlib>
#include <memory>
#include <string>

#include "utils.h"

extern "C" {
void* TorchAllocate(size_t n);
void TorchFree(void* ptr);
}

struct TorchCachingAllocator : public torch::Allocator {
    // For Torch Interface
    torch::DataPtr allocate(size_t n) override {
        void* data = TorchAllocate(n);
        return {data, data, &TorchFree, torch::DeviceType::CPU};
    }

    void copy_data(void* dest, const void* src, size_t count) const override {
        // A canary for unexpected host-allocator copies: level-gated, so it
        // costs a level comparison per copy instead of an unconditional write
        // to stderr.
        CopyCanaryLogger()->debug("Copying data ({} bytes)", count);
        memcpy(dest, src, count);
    }

    // This allocator is installed by a static initializer when the extension
    // loads, long before any EngineConfig exists, so there is no configured
    // level to thread in here. Its logger therefore follows spdlog's
    // process-wide default level, which keeps the per-copy canary above
    // silent unless that default is turned down to debug.
    static const std::shared_ptr<spdlog::logger>& CopyCanaryLogger() {
        const spdlog::string_view_t level_name =
            spdlog::level::to_string_view(spdlog::default_logger()->level());
        static const std::shared_ptr<spdlog::logger> logger = init_logger(
            std::string(level_name.data(), level_name.size()),
            "TorchCachingAllocator");
        return logger;
    }

    // // Optional: Handle deallocation (if needed)
    // void deallocate(void* ptr) override {
    //   Free(ptr);  // Custom deallocation logic
    // }
};

// extern std::unique_ptr<TorchCachingAllocator> kTorchCachingAllocator;

class ReplaceTorchAllocatorOnLoad {
   public:
    ReplaceTorchAllocatorOnLoad() {
        std::call_once(flag_, [&]() {
            // InitCachingAllocator(MemoryType::PIN_SHM);
            torch_caching_allocator_ = new TorchCachingAllocator();
            torch::SetAllocator(torch::DeviceType::CPU,
                                torch_caching_allocator_);
            // std::cerr << "Set custom allocator" << std::endl;
        });
    }

   private:
    TorchCachingAllocator* torch_caching_allocator_;
    std::once_flag flag_;
};

// Create a static instance of this class
extern ReplaceTorchAllocatorOnLoad kReplaceTorchAllocatorOnLoad;
