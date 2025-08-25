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

#include <torch/torch.h>
// #include "caching_allocator.h"
#include <cstdlib>
#include <memory>

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
        std::cerr << "Copying data" << std::endl;
        memcpy(dest, src, count);
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
