// clang-format off
/* ----------------------------------------------------------------------------  *
 * BatchGen                                                                      *
 * copyright (c) EfficientMoE team 2025                                             *
 * *
 * licensed under the apache license, version 2.0 (the "license");              *
 * you may not use this file except in compliance with the license.             *
 * *
 * you may obtain a copy of the license at                                      *
 * *
 * http://www.apache.org/licenses/license-2.0                   *
 * *
 * unless required by applicable law or agreed to in writing, software          *
 * distributed under the license is distributed on an "as is" basis,            *
 * without warranties or conditions of any kind, either express or implied.     *
 * see the license for the specific language governing permissions and          *
 * limitations under the license.                                               *
 * ---------------------------------------------------------------------------- */
// clang-format on

#include "Weights_Storage.h"
#include "distributed_weights_protocol.h"
#include "../Parameter_Server/posix_shm.h"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <memory>
#include <fstream>
#include <string>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include <torch/extension.h>
#include "nlohmann_json/json.hpp"

namespace {

using batchgen::distributed_weights::Operation;
using batchgen::distributed_weights::Request;
using batchgen::distributed_weights::Response;

void send_exact(int fd, const void* data, size_t bytes) {
    const char* cursor = static_cast<const char*>(data);
    while (bytes > 0) {
        const ssize_t sent = send(fd, cursor, bytes, MSG_NOSIGNAL);
        if (sent < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error("distributed weights send failed: " +
                                     std::string(strerror(errno)));
        }
        if (sent == 0) {
            throw std::runtime_error(
                "distributed weights send returned zero");
        }
        cursor += sent;
        bytes -= static_cast<size_t>(sent);
    }
}

void recv_exact(int fd, void* data, size_t bytes) {
    char* cursor = static_cast<char*>(data);
    while (bytes > 0) {
        const ssize_t received = recv(fd, cursor, bytes, MSG_WAITALL);
        if (received < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error("distributed weights recv failed: " +
                                     std::string(strerror(errno)));
        }
        if (received == 0) {
            throw std::runtime_error(
                "distributed weights daemon closed the socket");
        }
        cursor += received;
        bytes -= static_cast<size_t>(received);
    }
}

int recv_fd_with_response(int fd, Response* response) {
    char control[CMSG_SPACE(sizeof(int))] = {};
    iovec iov{response, sizeof(*response)};
    msghdr message{};
    message.msg_iov = &iov;
    message.msg_iovlen = 1;
    message.msg_control = control;
    message.msg_controllen = sizeof(control);
    const ssize_t received = recvmsg(fd, &message, MSG_WAITALL);
    if (received != static_cast<ssize_t>(sizeof(*response))) {
        throw std::runtime_error(
            "distributed weights handshake response is truncated");
    }
    for (cmsghdr* header = CMSG_FIRSTHDR(&message); header != nullptr;
         header = CMSG_NXTHDR(&message, header)) {
        if (header->cmsg_level == SOL_SOCKET &&
            header->cmsg_type == SCM_RIGHTS &&
            header->cmsg_len >= CMSG_LEN(sizeof(int))) {
            int received_fd = -1;
            std::memcpy(&received_fd, CMSG_DATA(header), sizeof(received_fd));
            return received_fd;
        }
    }
    throw std::runtime_error(
        "distributed weights daemon did not pass the staging memfd");
}

std::vector<std::string> split_tsv(const std::string& line) {
    std::vector<std::string> fields;
    size_t begin = 0;
    while (true) {
        const size_t end = line.find('\t', begin);
        fields.emplace_back(
            line.substr(begin, end == std::string::npos
                                   ? std::string::npos
                                   : end - begin));
        if (end == std::string::npos) {
            return fields;
        }
        begin = end + 1;
    }
}

void validate_response(const Response& response) {
    if (response.magic !=
            batchgen::distributed_weights::kProtocolMagic ||
        response.version !=
            batchgen::distributed_weights::kProtocolVersion) {
        throw std::runtime_error(
            "distributed weights protocol mismatch");
    }
    if (response.status != 0) {
        throw std::runtime_error(
            "distributed weights daemon error: " +
            std::string(response.error));
    }
}

}  // namespace
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

#include "../Parameter_Server/Parameter_Server.h"
#include "../data_structures.h"
#include "../utils.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

// Weights_Storage.cpp

Weights_Storage::Weights_Storage(int device_id)
    : device_id_(device_id) {
    
    // Initialize logger with a default level (e.g., 2 for INFO) 
    // since we don't have engine_config anymore.
    this->logger = init_logger(
        "info", // Default Log Level: INFO
        "Weights_Storage_" + std::to_string(this->device_id_));
        
    this->logger->info("Weights_Storage Instantiated on device {}.", this->device_id_);
};

Weights_Storage::~Weights_Storage() {
    // free_shared_pinned_memory(this->shm_name, this->weight_ptr_,
    //                           this->byte_size_, true);
    if (this->compact_ptr_ != nullptr) {
        cudaHostUnregister(this->compact_ptr_);
        munmap(this->compact_ptr_, this->compact_bytes_);
    }
    if (this->staging_ptr_ != nullptr) {
        cudaHostUnregister(this->staging_ptr_);
        munmap(this->staging_ptr_, this->staging_bytes_);
    }
    if (this->compact_fd_ >= 0) {
        close(this->compact_fd_);
    }
    if (this->staging_fd_ >= 0) {
        close(this->staging_fd_);
    }
    if (this->daemon_socket_ >= 0) {
        close(this->daemon_socket_);
    }
    if(this->logger) {
        this->logger->info("Weights_Storage Destroyed.");
    }
};

void Weights_Storage::InitDistributed(const std::string& config_path) {
    CUDA_CHECK(cudaSetDevice(this->device_id_));
    std::ifstream config_handle(config_path);
    if (!config_handle) {
        throw std::runtime_error(
            "cannot open distributed weight config: " + config_path);
    }
    nlohmann::json config;
    config_handle >> config;

    this->local_node_rank_ = config.at("node_rank").get<int>();
    const std::string store_path =
        config.at("store_path").get<std::string>();
    const std::string metadata_path =
        config.at("metadata_path").get<std::string>();
    const std::string socket_path =
        config.at("daemon_socket").get<std::string>();
    this->compact_bytes_ =
        config.at("store_bytes").get<int64_t>();
    this->distributed_module_bytes_ =
        config.at("module_bytes").get<int64_t>();
    if (!config.contains("worker_sharded") ||
        !config.at("worker_sharded").is_boolean() ||
        !config.at("worker_sharded").get<bool>()) {
        throw std::runtime_error(
            "distributed K3 weights require worker_sharded=true");
    }

    this->compact_fd_ = open(store_path.c_str(), O_RDWR);
    if (this->compact_fd_ < 0) {
        throw std::runtime_error(
            "cannot open compact weight store O_RDWR: " + store_path +
            ": " + strerror(errno));
    }
    struct stat store_stat {};
    if (fstat(this->compact_fd_, &store_stat) != 0 ||
        store_stat.st_size != this->compact_bytes_) {
        throw std::runtime_error(
            "compact weight store size mismatch");
    }
    this->compact_ptr_ =
        mmap(nullptr, this->compact_bytes_, PROT_READ | PROT_WRITE,
             MAP_SHARED, this->compact_fd_, 0);
    if (this->compact_ptr_ == MAP_FAILED) {
        this->compact_ptr_ = nullptr;
        throw std::runtime_error(
            "mmap compact weight store failed: " +
            std::string(strerror(errno)));
    }
    CUDA_CHECK(cudaHostRegister(this->compact_ptr_, this->compact_bytes_,
                                cudaHostRegisterDefault));

    std::ifstream metadata_handle(metadata_path);
    if (!metadata_handle) {
        throw std::runtime_error(
            "cannot open distributed weight metadata: " + metadata_path);
    }
    std::string line;
    std::unordered_map<std::string, uint64_t> remote_module_bases;
    size_t local_tensors = 0;
    size_t remote_tensors = 0;
    while (std::getline(metadata_handle, line)) {
        const auto fields = split_tsv(line);
        if (fields.empty() || fields[0] == "H") {
            continue;
        }
        if (fields.size() != 9 || fields[0] != "T") {
            throw std::runtime_error(
                "invalid distributed weight metadata row");
        }
        const std::string& module_key = fields[2];
        const std::string& tensor_key = fields[3];
        const int owner = std::stoi(fields[4]);
        const uint64_t compact_offset = std::stoull(fields[5]);
        const int64_t tensor_bytes = std::stoll(fields[6]);
        const std::string& dtype = fields[7];
        const std::vector<int64_t> shape =
            nlohmann::json::parse(fields[8])
                .get<std::vector<int64_t>>();
        if (compact_offset + static_cast<uint64_t>(tensor_bytes) >
            static_cast<uint64_t>(this->compact_bytes_)) {
            throw std::runtime_error(
                "distributed tensor exceeds compact store: " +
                module_key + "/" + tensor_key);
        }
        if (owner < 0 || owner == this->local_node_rank_) {
            this->module_weights_storage_[module_key][tensor_key] =
                tensor_buffer(
                    static_cast<char*>(this->compact_ptr_) +
                        compact_offset,
                    shape, tensor_bytes, dtype);
            ++local_tensors;
        } else {
            auto base = remote_module_bases.find(module_key);
            if (base == remote_module_bases.end()) {
                remote_module_bases.emplace(module_key, compact_offset);
            } else {
                base->second = std::min(base->second, compact_offset);
            }
            this->remote_module_weights_[module_key][tensor_key] =
                distributed_tensor_meta{
                    shape, tensor_bytes, dtype, compact_offset, 0};
            ++remote_tensors;
        }
    }
    for (auto& [module_key, tensors] :
         this->remote_module_weights_) {
        const uint64_t base = remote_module_bases.at(module_key);
        uint64_t module_end = 0;
        for (auto& [tensor_key, tensor] : tensors) {
            tensor.module_offset = tensor.compact_offset - base;
            module_end = std::max(
                module_end,
                tensor.module_offset +
                    static_cast<uint64_t>(tensor.byte_size));
        }
        if (module_end !=
            static_cast<uint64_t>(this->distributed_module_bytes_)) {
            throw std::runtime_error(
                "remote module byte contract mismatch: " + module_key);
        }
    }

    this->daemon_socket_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (this->daemon_socket_ < 0) {
        throw std::runtime_error(
            "cannot create distributed weights unix socket");
    }
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    if (socket_path.size() >= sizeof(address.sun_path)) {
        throw std::runtime_error(
            "distributed weights socket path is too long");
    }
    std::strncpy(address.sun_path, socket_path.c_str(),
                 sizeof(address.sun_path) - 1);
    if (connect(this->daemon_socket_,
                reinterpret_cast<sockaddr*>(&address),
                sizeof(address)) != 0) {
        throw std::runtime_error(
            "cannot connect distributed weights daemon: " +
            std::string(strerror(errno)));
    }
    Request hello{};
    hello.operation =
        static_cast<uint32_t>(Operation::kHello);
    hello.worker_id = static_cast<uint32_t>(this->device_id_);
    send_exact(this->daemon_socket_, &hello, sizeof(hello));
    Response response{};
    this->staging_fd_ =
        recv_fd_with_response(this->daemon_socket_, &response);
    validate_response(response);
    this->staging_bytes_ =
        static_cast<int64_t>(response.staging_bytes);
    if (response.module_bytes !=
        static_cast<uint64_t>(this->distributed_module_bytes_)) {
        throw std::runtime_error(
            "distributed staging module byte mismatch");
    }
    this->staging_ptr_ =
        mmap(nullptr, this->staging_bytes_, PROT_READ | PROT_WRITE,
             MAP_SHARED, this->staging_fd_, 0);
    if (this->staging_ptr_ == MAP_FAILED) {
        this->staging_ptr_ = nullptr;
        throw std::runtime_error(
            "mmap distributed staging memfd failed");
    }
    CUDA_CHECK(cudaHostRegister(this->staging_ptr_,
                                this->staging_bytes_,
                                cudaHostRegisterDefault));
    this->distributed_ = true;
    this->byte_size_ = this->compact_bytes_;
    this->logger->info(
        "Distributed compact weights ready: node_rank={}, store={:.3f} "
        "GiB, staging={:.3f} GiB, local_tensors={}, remote_tensors={}",
        this->local_node_rank_,
        this->compact_bytes_ / (1024.0 * 1024.0 * 1024.0),
        this->staging_bytes_ / (1024.0 * 1024.0 * 1024.0),
        local_tensors, remote_tensors);
}

Weights_Storage::active_lease
Weights_Storage::acquire_remote_module(
    const std::string& module_key) {
    std::lock_guard<std::mutex> lock(this->daemon_mutex_);
    if (this->active_leases_.count(module_key) != 0) {
        throw std::runtime_error(
            "duplicate remote module acquire in one worker: " +
            module_key);
    }
    Request request{};
    request.operation =
        static_cast<uint32_t>(Operation::kAcquire);
    request.worker_id = static_cast<uint32_t>(this->device_id_);
    std::strncpy(request.module_key, module_key.c_str(),
                 sizeof(request.module_key) - 1);
    send_exact(this->daemon_socket_, &request, sizeof(request));
    Response response{};
    recv_exact(this->daemon_socket_, &response, sizeof(response));
    validate_response(response);
    if (response.slot < 0 ||
        (static_cast<uint64_t>(response.slot) + 1) *
                static_cast<uint64_t>(
                    this->distributed_module_bytes_) >
            static_cast<uint64_t>(this->staging_bytes_)) {
        throw std::runtime_error(
            "distributed weights daemon returned invalid slot");
    }
    active_lease lease{response.slot, response.generation};
    this->active_leases_[module_key] = lease;
    return lease;
}

void Weights_Storage::release_module(
    const std::string& module_key) {
    if (!this->distributed_) {
        return;
    }
    std::lock_guard<std::mutex> lock(this->daemon_mutex_);
    auto found = this->active_leases_.find(module_key);
    if (found == this->active_leases_.end()) {
        return;
    }
    Request request{};
    request.operation =
        static_cast<uint32_t>(Operation::kRelease);
    request.worker_id = static_cast<uint32_t>(this->device_id_);
    request.generation = found->second.generation;
    std::strncpy(request.module_key, module_key.c_str(),
                 sizeof(request.module_key) - 1);
    send_exact(this->daemon_socket_, &request, sizeof(request));
    Response response{};
    recv_exact(this->daemon_socket_, &response, sizeof(response));
    validate_response(response);
    this->active_leases_.erase(found);
}

/*
auto weights_map = deserialize_from_shared_memory(tensor_meta_shm_name);

*/
void Weights_Storage::Init(
    std::string& shm_name, int64_t byte_size,
    std::string& tensor_meta_shm_name, bool enable_hugetlbfs,
    bool enable_memfd, int memfd_creator_pid, int memfd_fd_arg)
{
    this->logger->info(
        "Setting CUDA device to {} for Weights_Storage initialization.",
        this->device_id_);       
    // Use the stored device_id_ member
    CUDA_CHECK(cudaSetDevice(this->device_id_));
        
    this->shm_name = shm_name;
    auto start_time = std::chrono::high_resolution_clock::now();
    this->byte_size_ = byte_size;
    auto weights_map = deserialize_from_shared_memory(tensor_meta_shm_name);
    
    this->logger->info(
        "Initializing Weights_Storage with shared memory name: {} and byte size: {}",
        shm_name, byte_size);

    // Worker process: register with CUDA for DMA access (pin_for_cuda=true)
    void* weight_ptr =
        allocate_shared_pinned_memory(shm_name, byte_size, false, enable_hugetlbfs, true,
                                      enable_memfd, memfd_creator_pid, memfd_fd_arg);
        
    // Check if weight_ptr is null
    if (weight_ptr == nullptr) {
        this->logger->error("Failed to allocate shared pinned memory.");
        throw std::runtime_error("Failed to allocate shared pinned memory.");
    }
    
    this->weight_ptr_ = weight_ptr;
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time)
            .count();
            
    this->logger->info("Shared Pinned Memory Allocation Time: {} seconds.",
                       duration);
                       
    for (auto& [module_key, tensor_map] : weights_map) {
        for (auto& [tensor_key, meta] : tensor_map) {
            this->module_weights_storage_[module_key][tensor_key] =
                tensor_buffer(static_cast<char*>(weight_ptr) + meta.offset,
                              meta.tensor_shape,
                              meta.byte_size,
                              meta.dtype);  // Pass dtype from metadata
        }
    }
}
// void Weights_Storage::Init(
//     std::string& shm_name, int64_t byte_size,
//     std::unordered_map<std::string,
//                        std::unordered_map<std::string, tensor_meta>>
//         module_weights_shm, bool enable_hugetlbfs) 
// {
//     this->logger->info(
//         "Setting CUDA device to {} for Weights_Storage initialization.",
//         this->device_id_);       
//     // Use the stored device_id_ member
//     CUDA_CHECK(cudaSetDevice(this->device_id_));
        
//     this->shm_name = shm_name;
//     auto start_time = std::chrono::high_resolution_clock::now();
//     this->byte_size_ = byte_size;
    
//     this->logger->info(
//         "Initializing Weights_Storage with shared memory name: {} and byte size: {}",
//         shm_name, byte_size);
        
//     void* weight_ptr =
//         allocate_shared_pinned_memory(shm_name, byte_size, false, enable_hugetlbfs);
        
//     // Check if weight_ptr is null
//     if (weight_ptr == nullptr) {
//         this->logger->error("Failed to allocate shared pinned memory.");
//         throw std::runtime_error("Failed to allocate shared pinned memory.");
//     }
    
//     this->weight_ptr_ = weight_ptr;
//     auto end_time = std::chrono::high_resolution_clock::now();
//     auto duration =
//         std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time)
//             .count();
            
//     this->logger->info("Shared Pinned Memory Allocation Time: {} seconds.",
//                        duration);
                       
//     for (auto& [module_key, tensor_map] : module_weights_shm) {
//         for (auto& [tensor_key, meta] : tensor_map) {
//             this->module_weights_storage_[module_key][tensor_key] =
//                 tensor_buffer(static_cast<char*>(weight_ptr) + meta.offset, 
//                               meta.tensor_shape,
//                               meta.byte_size);
//         }
//     }
// }

std::unordered_map<std::string, tensor_buffer>
Weights_Storage::get_module_weights_storage(std::string module_key) {
    /* To Facilitate Module Copy */
    auto local = this->module_weights_storage_.find(module_key);
    if (local != this->module_weights_storage_.end()) {
        return local->second;
    }
    auto remote = this->remote_module_weights_.find(module_key);
    if (remote != this->remote_module_weights_.end()) {
        const active_lease lease =
            this->acquire_remote_module(module_key);
        std::unordered_map<std::string, tensor_buffer> result;
        char* base =
            static_cast<char*>(this->staging_ptr_) +
            static_cast<int64_t>(lease.slot) *
                this->distributed_module_bytes_;
        for (const auto& [tensor_key, tensor] : remote->second) {
            result[tensor_key] = tensor_buffer(
                base + tensor.module_offset,
                tensor.tensor_shape, tensor.byte_size, tensor.dtype);
        }
        return result;
    }
    {
        this->logger->error("Module key not found in storage: {}", module_key);
        throw std::runtime_error("Module key not found in storage.");
    }
};

py::dict Weights_Storage::get_tensor(std::string module_key) {
    /* Get the tensor from the weights storage and return as Python dict.
     *
     * Uses stored dtype from tensor metadata instead of guessing from tensor names.
     * Supports: bfloat16, uint8, float8_e4m3fn, float32, float16
     */

    // Check if module key exists
    if (this->remote_module_weights_.find(module_key) !=
        this->remote_module_weights_.end()) {
        throw std::runtime_error(
            "get_tensor cannot retain a remote distributed module: " +
            module_key);
    }
    if (this->module_weights_storage_.find(module_key) ==
        this->module_weights_storage_.end()) {
        this->logger->error("Module key not found in storage: {}", module_key);
        throw std::runtime_error("Module key not found in storage.");
    }

    // Get module weights
    auto module_weights = this->module_weights_storage_[module_key];

    // Create Python dict to store tensors
    py::dict tensors;

    // Iterate through module weights and create tensors
    for (auto& [tensor_key, tb] : module_weights) {
        torch::Tensor tensor;

        // Use stored dtype instead of guessing from tensor name
        torch::Dtype torch_dtype;
        std::string resolved_dtype_name;
        if (tb.dtype == "bfloat16") {
            torch_dtype = torch::kBFloat16;
            resolved_dtype_name = "bfloat16";
        } else if (tb.dtype == "uint8") {
            torch_dtype = torch::kUInt8;
            resolved_dtype_name = "uint8";
        } else if (tb.dtype == "float8_e4m3fn") {
            torch_dtype = torch::kFloat8_e4m3fn;
            resolved_dtype_name = "float8_e4m3fn";
        } else if (tb.dtype == "float32") {
            torch_dtype = torch::kFloat32;
            resolved_dtype_name = "float32";
        } else if (tb.dtype == "float16") {
            torch_dtype = torch::kFloat16;
            resolved_dtype_name = "float16";
        } else if (tb.dtype == "int32") {
            torch_dtype = torch::kInt32;
            resolved_dtype_name = "int32";
        } else if (tb.dtype == "int64") {
            torch_dtype = torch::kInt64;
            resolved_dtype_name = "int64";
        } else if (tb.dtype == "int16") {
            torch_dtype = torch::kInt16;
            resolved_dtype_name = "int16";
        } else if (tb.dtype == "int8") {
            torch_dtype = torch::kInt8;
            resolved_dtype_name = "int8";
        } else if (tb.dtype == "float64") {
            torch_dtype = torch::kFloat64;
            resolved_dtype_name = "float64";
        } else {
            // Fallback to fp8 for backward compatibility
            this->logger->warn("Unknown dtype '{}' for tensor '{}', defaulting to fp8",
                              tb.dtype, tensor_key);
            torch_dtype = torch::kFloat8_e4m3fn;
            resolved_dtype_name = "float8_e4m3fn (fallback)";
        }

        // Log raw dtype from metadata for debugging
        this->logger->debug("[{}] tensor '{}': raw_dtype='{}' -> torch_dtype={}",
                           module_key, tensor_key, tb.dtype, resolved_dtype_name);

        auto options = torch::TensorOptions()
            .dtype(torch_dtype)
            .device(torch::kCPU)
            .requires_grad(false)
            .memory_format(torch::MemoryFormat::Contiguous);

        tensor = torch::from_blob(
            tb.data_ptr,
            tb.tensor_shape,
            options
        );

        // Add tensor to Python dict
        tensors[tensor_key.c_str()] = tensor;
    }

    return tensors;
}
