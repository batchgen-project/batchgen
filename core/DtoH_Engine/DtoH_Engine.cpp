// clang-format off
/* ----------------------------------------------------------------------------  *
 *  MoE-Gen                                                                      *
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

#include "spdlog/spdlog.h"
#include <memory>
#include <string>
#include <torch/extension.h>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

#include "../KV_Storage/KV_Storage.h"
#include "../Weights_Storage/Weights_Storage.h"
#include "../data_structures.h"
#include "../threadsafe_queue.h"
#include "../utils.h"
#include "DtoH_Engine.h"

DtoH_Engine::DtoH_Engine(EngineConfig& engine_config)
    : engine_config_(engine_config) {
    cudaSetDevice(this->engine_config_.basic_config.device);
    CUDA_CHECK(
        cudaStreamCreateWithFlags(&this->DtoH_stream, cudaStreamNonBlocking));
    this->logger_ = init_logger(
        this->engine_config_.basic_config.log_level,
        "DtoH_Engine" +
            std::to_string(this->engine_config_.basic_config.device));
    this->logger_->info("DtoH_Engine Instantiated.");
};

void DtoH_Engine::Init() {
    this->terminate_flag_ = false;
    this->d2h_copy_worker_thread_ =
        std::thread(&DtoH_Engine::d2h_copy_worker_thread_func_, this);
};

void DtoH_Engine::Terminate() {
    this->terminate_flag_ = true;
    if (this->d2h_copy_worker_thread_.joinable()) {
        this->d2h_copy_worker_thread_.join();
    }
};

DtoH_Engine::~DtoH_Engine() { this->Terminate(); };

// torch::Tensor DtoH_Engine::tensor_on_demand_copy(torch::Tensor& src_tensor){
// 	auto dst_tensor = torch::empty_like(src_tensor,
// torch::kCPU).contiguous();
// 	// Create packaged task
// 	std::packaged_task<void()> task([this, dst_ptr = dst_tensor.data_ptr(),
// 									src_ptr
// =
// src_tensor.data_ptr(),
// size = src_tensor.numel() * src_tensor.element_size()]() {
// this->blocking_copy_(dst_ptr, src_ptr, size);
// 	});

// 	// Get future before moving task
// 	std::future<void> completion_future = task.get_future();

// 	// Push task to queue
// 	this->on_demand_task_queue_.push(std::move(task));

// 	// Wait for completion
// 	completion_future.wait();

// 	return dst_tensor;
// };
torch::Tensor DtoH_Engine::tensor_on_demand_copy(torch::Tensor& src_tensor) {
    auto dst_tensor = torch::empty_like(src_tensor, torch::kCPU).contiguous();
    this->blocking_copy_(dst_tensor.data_ptr(), src_tensor.data_ptr(),
                         src_tensor.numel() * src_tensor.element_size());
}

void DtoH_Engine::submit_to_queue_B(void* dst, void* src, int64_t size) {
    try {
        std::packaged_task<void()> task([this, dst, src, size](){
        	this->blocking_copy_(dst, src, size);
        });
        std::future<void> completion_future = task.get_future();
        this->queue_B_.push(std::move(task));
        completion_future.wait();

        // CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
        // // CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
        // CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost,
        //                            this->DtoH_stream));
        // CUDA_CHECK(cudaStreamSynchronize(this->DtoH_stream));
        // CUDA_CHECK(cudaStreamSynchronize(0));
    } catch (const c10::Error& e) {
        this->logger_->debug("KV_Storage: CUDA/PyTorch error: {}", e.what());
        throw std::runtime_error(e.what());
    }
    // Catch CUDA runtime errors
    catch (const cudaError_t& err) {
        this->logger_->debug("KV_Storage: CUDA runtime error: {}",
                             cudaGetErrorString(err));
        throw std::runtime_error(cudaGetErrorString(err));
    }
    // Catch standard C++ exceptions
    catch (const std::exception& e) {
        this->logger_->debug("KV_Storage: submit_to_queue_B(). Error: {}",
                             e.what());
        throw std::runtime_error("KV_Storage submit_to_queue_B().");
    }
    // Catch any other unexpected errors
    catch (...) {
        this->logger_->debug("KV_Storage: submit_to_queue_B().");
        throw std::runtime_error("KV_Storage: submit_to_queue_B().");
    }
};

void DtoH_Engine::blocking_copy_(void* dst, void* src, int64_t size) {
    try {
        CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
        if (src == nullptr) {
            throw std::invalid_argument("src null pointer detected");
        }
        if (dst == nullptr) {
            throw std::invalid_argument("dst null pointer detected");
        }

        if (this->DtoH_stream == cudaStreamDefault ||
            this->DtoH_stream == cudaStreamLegacy) {
            this->logger_->debug("Stream is default or legacy");
            throw std::invalid_argument("Stream is default or legacy");
        }

        if (this->DtoH_stream == nullptr) {
            this->logger_->debug("Stream is null");
            throw std::invalid_argument("Stream is null");
        }

        cudaError_t err = cudaStreamQuery(this->DtoH_stream);
        if (err != cudaSuccess && err != cudaErrorNotReady) {
            this->logger_->debug("Stream is invalid: {}",
                                 cudaGetErrorString(err));
            throw std::runtime_error("Invalid stream: " +
                                     std::string(cudaGetErrorString(err)));
        }

        // this->logger_->debug("DtoH_Engine: dst: {}, src: {}, size: {}",
        // (void*)dst, (void*)src, size);
        CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost,
                                   this->DtoH_stream));
        // CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaStreamSynchronize(this->DtoH_stream));
    } catch (const c10::Error& e) {
        this->logger_->debug("KV_Storage: CUDA/PyTorch error: {}", e.what());
        throw std::runtime_error(e.what());
    }
    // Catch CUDA runtime errors
    catch (const cudaError_t& err) {
        this->logger_->debug("KV_Storage: CUDA runtime error: {}",
                             cudaGetErrorString(err));
        throw std::runtime_error(cudaGetErrorString(err));
    }
    // Catch standard C++ exceptions
    catch (const std::exception& e) {
        this->logger_->debug("KV_Storage: blocking_copy_() Error: {}",
                             e.what());
        throw std::runtime_error("KV_Storage blocking_copy_() error.");
    }
    // Catch any other unexpected errors
    catch (...) {
        this->logger_->debug("KV_Storage: blocking_copy_().");
        throw std::runtime_error("KV_Storage: blocking_copy_() error.");
    }
};

void DtoH_Engine::d2h_copy_worker_thread_func_() {
    // This worker terminated.
    while (!this->terminate_flag_) {
        std::packaged_task<void()> task;
        while (on_demand_task_queue_.try_pop(task)) {
            try {
                task();
            } catch (...) {
                this->logger_->debug(
                    "DtoH_Engine: Failed to execute on_demand task.");
                throw std::runtime_error(
                    "DtoH_Engine: Failed to execute on_demand task.");
            }
        }
        if (queue_B_.try_pop(task)) {
            try {
                task();
            } catch (...) {
                this->logger_->debug("DtoH_Engine: Failed to execute task.");
                throw std::runtime_error(
                    "DtoH_Engine: Failed to execute task.");
            }
        }
    };
};

void DtoH_Engine::wait_until_queue_B_empty() {
    while (!queue_B_.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    CUDA_CHECK(cudaStreamSynchronize(this->DtoH_stream));
};
