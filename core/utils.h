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

/*
        Including infrastructure functions.
        - Model Config Parser
        - Engine Config Parser
        - cudaError Check
        - logger init
        - pinned memory allocation
*/
#pragma once
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <torch/cuda.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "data_structures.h"
template <typename Func, typename Logger>
inline void safeCall(Func&& func, Logger logger, const char* callerFunction) {
    try {
        func();
    } catch (const c10::Error& e) {
        logger->debug("{}: CUDA/PyTorch error: {}", callerFunction, e.what());
        throw std::runtime_error(e.what());
    } catch (const cudaError_t& err) {
        logger->debug("{}: CUDA runtime error: {}", callerFunction,
                      cudaGetErrorString(err));
        throw std::runtime_error(cudaGetErrorString(err));
    } catch (const std::exception& e) {
        logger->debug("{}: Error: {}", callerFunction, e.what());
        throw std::runtime_error(e.what());
    } catch (...) {
        logger->debug("{}:", callerFunction);
        throw std::runtime_error("Unknown error");
    }
}
#define SAFE_CALL(func, logger) safeCall(func, logger, __FUNCTION__)

namespace py = pybind11;
inline void throwOnCudaError(cudaError_t error, const char* file, int line,
                             const char* function, const char* call) {
    if (error != cudaSuccess) {
        std::stringstream ss;
        ss << "CUDA error " << error << " at " << file << ":" << line
           << " in function " << function << ": " << cudaGetErrorString(error)
           << "\nCall: " << call;
        throw std::runtime_error(ss.str());
    }
};

#define CUDA_CHECK(call) \
    throwOnCudaError(call, __FILE__, __LINE__, __FUNCTION__, #call)

struct sync_event {
    std::atomic<int> is_set;
    cudaEvent_t event;
    sync_event() : is_set(0) { CUDA_CHECK(cudaEventCreate(&event)); }
    ~sync_event() { cudaEventDestroy(event); }
    // Delete copy constructor and copy assignment operator
    sync_event(const sync_event&) = delete;
    sync_event& operator=(const sync_event&) = delete;
};

EngineConfig parse_engine_config(const py::object& engine_config);
ModelConfig parse_model_config(const py::object& model_config);
std::shared_ptr<spdlog::logger> init_logger(const std::string& log_level,
                                            const std::string& logger_name);
std::string get_tensor_shape(
    const torch::Tensor& t,
    bool include_dtype = true,
    bool include_device = true);

template <typename T>
constexpr T ceil_div(T numerator, T denominator) {
    return (numerator + denominator - 1) / denominator;
}


