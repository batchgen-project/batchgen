#include "host_paged_kv_worker_view.h"

#include <cuda_runtime_api.h>

#include <functional>
#include <sstream>
#include <stdexcept>

namespace batchgen::kv::worker_detail {

namespace {

std::string BuildCudaErrorMessage(const char* op, cudaError_t error) {
    std::ostringstream oss;
    oss << op << " failed with error " << static_cast<int>(error) << " ("
        << cudaGetErrorString(error) << ")";
    return oss.str();
}

template <typename Callable, typename ErrorLogger>
void InvokeCudaChecked(const char* op,
                       const std::shared_ptr<spdlog::logger>& logger,
                       Callable&& callable, ErrorLogger&& error_logger) {
    const cudaError_t result = std::invoke(std::forward<Callable>(callable));
    if (result == cudaSuccess) {
        return;
    }
    const std::string message = BuildCudaErrorMessage(op, result);
    std::invoke(std::forward<ErrorLogger>(error_logger), message);
    throw std::runtime_error(message);
}

template <typename Callable>
void InvokeCudaChecked(const char* op,
                       const std::shared_ptr<spdlog::logger>& logger,
                       Callable&& callable) {
    InvokeCudaChecked(
        op, logger, std::forward<Callable>(callable),
        [&](const std::string& message) { logger->error("{}", message); });
}

}  // namespace

void RegisterPinnedRange(void* base, std::size_t bytes, int device_index,
                         const std::shared_ptr<spdlog::logger>& logger) {
    if (base == nullptr || bytes == 0) {
        return;
    }
    if (device_index < 0) {
        throw std::invalid_argument(
            "device_index must be greater than or equal to zero");
    }
    int device_count = 0;
    InvokeCudaChecked("cudaGetDeviceCount", logger,
                      [&]() { return cudaGetDeviceCount(&device_count); });
    if (device_index >= device_count) {
        throw std::out_of_range("device_index " + std::to_string(device_index) +
                                " exceeds available devices " +
                                std::to_string(device_count));
    }
    InvokeCudaChecked(
        "cudaSetDevice", logger, [&]() { return cudaSetDevice(device_index); },
        [&](const std::string& message) {
            logger->error("{} (device_index={})", message, device_index);
        });
    
    logger->info("Attempting cudaHostRegister (ptr={}, bytes={}, device={})", 
                 base, bytes, device_index);
    
    // Check if memory is already registered in THIS process's CUDA context
    // (each process has its own CUDA runtime state)
    unsigned int existing_flags = 0;
    cudaError_t check_result = cudaHostGetFlags(&existing_flags, base);
    if (check_result == cudaSuccess) {
        // Memory is already registered in this process - skip re-registration
        logger->info("KV range already registered in this process (ptr={}, bytes={}, flags={})", 
                     base, bytes, existing_flags);
        return;
    }
    
    // Use cudaHostRegisterDefault to match Weights_Storage behavior
    // cudaHostRegisterPortable|cudaHostRegisterMapped was causing error 1 (invalid argument)
    // on some ranks, possibly due to memory alignment or size constraints
    constexpr unsigned int kFlags = cudaHostRegisterDefault;
    
    cudaError_t register_result = cudaHostRegister(base, bytes, kFlags);
    
    if (register_result == cudaSuccess) {
        logger->info("Successfully registered pinned KV range (ptr={}, bytes={}, device={})", 
                     base, bytes, device_index);
        return;
    }
    
    // Log detailed error information for debugging
    std::string err_msg = fmt::format(
        "cudaHostRegister failed with error {} ({}) "
        "(ptr={}, bytes={}, device={}, flags=cudaHostRegisterDefault)",
        static_cast<int>(register_result),
        cudaGetErrorString(register_result),
        base, bytes, device_index);
    logger->error(err_msg);
    throw std::runtime_error(err_msg);
}

void UnregisterPinnedRange(void* base, int device_index,
                           const std::shared_ptr<spdlog::logger>& logger) {
    if (base == nullptr) {
        return;
    }
    if (device_index < 0) {
        throw std::invalid_argument(
            "device_index must be greater than or equal to zero");
    }
    InvokeCudaChecked(
        "cudaSetDevice", logger, [&]() { return cudaSetDevice(device_index); },
        [&](const std::string& message) {
            logger->error("{} (device_index={})", message, device_index);
        });
    
    // Check if memory is registered in this process before unregistering
    unsigned int flags = 0;
    cudaError_t check_result = cudaHostGetFlags(&flags, base);
    if (check_result != cudaSuccess) {
        // Memory is not registered in this process - nothing to do
        logger->info("KV range not registered in this process, skipping unregister (ptr={})", base);
        return;
    }
    
    InvokeCudaChecked(
        "cudaHostUnregister", logger,
        [&]() { return cudaHostUnregister(base); },
        [&](const std::string& message) {
            logger->error("{} (ptr={})", message, base);
        });
    logger->info("Unregistered pinned KV range (ptr={})", base);
}

}  // namespace batchgen::kv::worker_detail
