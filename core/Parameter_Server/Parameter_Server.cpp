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
#include <cuda_runtime_api.h>
#include <fcntl.h>
// #include <filesystem>
#include <future>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/serialize.h>
#include <torch/torch.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>
#include <dirent.h>
#include <fstream>

#include "../data_structures.h"
#include "../utils.h"
#include "Parameter_Server.h"
#include "posix_shm.h"
#include "tqdm.hpp"
#include "nlohmann_json/json.hpp"

// namespace fs = std::filesystem;
namespace py = pybind11;
struct TensorInfo {
    std::string dtype;
    std::vector<int64_t> shape;
    int64_t offset;
    int64_t byte_size;
};
struct FileMetadata {
    std::string file_name;
    std::unordered_map<std::string, TensorInfo> state_dict;
    int64_t total_byte_size;
};
// Parse with nlohmann/json
FileMetadata parseMetadata(const std::string& json_path) {
    std::ifstream file(json_path);
    nlohmann::json j;
    file >> j;

    FileMetadata metadata;
    metadata.file_name = j["file_name"];
    metadata.total_byte_size = j["total_byte_size"];

    for (auto& [tensor_name, tensor_data] : j["state_dict"].items()) {
        TensorInfo info;
        info.dtype = tensor_data["dtype"];
        info.shape = tensor_data["shape"].get<std::vector<int64_t>>();
        info.offset = tensor_data["offset"];
        info.byte_size = tensor_data["byte_size"];
        metadata.state_dict[tensor_name] = info;
    }

    return metadata;
};

void Parameter_Server::_load_cus_format_file_to_host_mem(
    const std::string& model_weights_path, void* weight_ptr,
    std::unordered_map<std::string,
                       std::unordered_map<std::string, std::string>>&
        state_dict_name_map) {
    /*
        First get a list of all .json file.
        Loop through each .json file and parse it.
        Load the corresponding .bin file to weight_ptr + global_offset.
        Update this->module_weights_storage_ and this->skeleton_state_dict_.
        The global_offset is the sum of all previous tensor byte sizes.
    */
    int64_t global_offset = 0;
    // Step 1: Get all .json file paths in the model_weights_path directory
    std::vector<std::string> json_paths;
    DIR* dir = opendir(model_weights_path.c_str());
    if (dir != nullptr) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            // Skip the "." and ".." directory entries
            if (strcmp(entry->d_name, ".") != 0 &&
                strcmp(entry->d_name, "..") != 0) {
                if (strstr(entry->d_name, ".json") != nullptr) {
                    std::string full_path =
                        model_weights_path + "/" + entry->d_name;
                    json_paths.push_back(full_path);
                }
            }
        }
        closedir(dir);
    } else {
        throw std::runtime_error("Failed to open directory: " +
                                 model_weights_path);
    };

    // Step 2: Loop through each .json file and parse it
    auto bar = tq::tqdm(json_paths);
    bar.set_prefix("Loading Weights");
    for (const auto& json_path : bar) {
        // Parse the JSON file
        FileMetadata metadata;
        try {
            metadata = parseMetadata(json_path);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to parse JSON file: " + json_path +
                                     " - " + e.what());
        }

        // // Step 3: Load the corresponding .bin file to weight_ptr +
        // global_offset std::string bin_file_path = model_weights_path + "/" +
        // metadata.file_name; std::ifstream weight_bin_file(bin_file_path,
        // std::ios::binary); if (!weight_bin_file) {
        //     throw std::runtime_error("Failed to open binary file: " +
        //     bin_file_path);
        // };
        // auto start_time = std::chrono::high_resolution_clock::now();
        // weight_bin_file.read(
        //     static_cast<char*>(weight_ptr) + global_offset,
        //     metadata.total_byte_size);
        // weight_bin_file.close();
        // auto end_time = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> elapsed = end_time - start_time;
        // this->logger->info("Load file {} in {:.2f} seconds",
        //                    metadata.file_name, elapsed.count());

        // Step 3: Load using memory mapping
        // std::string bin_file_path = model_weights_path + "/" +
        // metadata.file_name;

        // auto start_time = std::chrono::high_resolution_clock::now();

        // int fd = open(bin_file_path.c_str(), O_RDONLY);
        // if (fd == -1) {
        //     throw std::runtime_error("Failed to open file: " +
        //     bin_file_path);
        // }

        // // Get file size
        // struct stat st;
        // if (fstat(fd, &st) == -1) {
        //     close(fd);
        //     throw std::runtime_error("Failed to get file size");
        // }

        // // Memory map the file
        // void* mapped_data = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE,
        // fd, 0); if (mapped_data == MAP_FAILED) {
        //     close(fd);
        //     throw std::runtime_error("Failed to memory map file");
        // }

        // // Advise kernel about access pattern
        // madvise(mapped_data, st.st_size, MADV_SEQUENTIAL | MADV_WILLNEED);

        // // Copy data in large chunks
        // char* dest_ptr = static_cast<char*>(weight_ptr) + global_offset;
        // const size_t chunk_size = 32 * 1024 * 1024; // 32MB chunks
        // size_t remaining = metadata.total_byte_size;
        // size_t offset = 0;

        // while (remaining > 0) {
        //     size_t to_copy = std::min(remaining, chunk_size);
        //     std::memcpy(dest_ptr + offset, static_cast<char*>(mapped_data) +
        //     offset, to_copy); offset += to_copy; remaining -= to_copy;
        // }

        // munmap(mapped_data, st.st_size);
        // close(fd);

        // auto end_time = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> elapsed = end_time - start_time;
        // this->logger->info("Load file {} in {:.2f} seconds (mmap)",
        //                 metadata.file_name, elapsed.count());
        //

        // Step 3: Load with direct I/O and large buffer
        // std::string bin_file_path = model_weights_path + "/" +
        // metadata.file_name;

        // auto start_time = std::chrono::high_resolution_clock::now();

        // // Open with direct I/O flag
        // int fd = open(bin_file_path.c_str(), O_RDONLY | O_DIRECT);
        // if (fd == -1) {
        //     throw std::runtime_error("Failed to open file with direct I/O: "
        //     + bin_file_path);
        // }

        // // For O_DIRECT, buffer must be aligned to sector size (usually 512
        // bytes or 4KB) const size_t buffer_size = 64 * 1024 * 1024; // 64MB
        // const size_t alignment = 4096; // 4KB alignment

        // // Allocate aligned buffer
        // void* aligned_buffer;
        // if (posix_memalign(&aligned_buffer, alignment, buffer_size) != 0) {
        //     close(fd);
        //     throw std::runtime_error("Failed to allocate aligned memory");
        // }

        // char* dest_ptr = static_cast<char*>(weight_ptr) + global_offset;
        // size_t remaining = metadata.total_byte_size;
        // size_t total_read = 0;

        // while (remaining > 0) {
        //     size_t to_read = std::min(remaining, buffer_size);

        //     // For O_DIRECT, read size should be aligned
        //     size_t aligned_read_size = (to_read + alignment - 1) &
        //     ~(alignment - 1);

        //     ssize_t bytes_read = read(fd, aligned_buffer, aligned_read_size);
        //     if (bytes_read <= 0) {
        //         free(aligned_buffer);
        //         close(fd);
        //         throw std::runtime_error("Failed to read from file");
        //     }

        //     // Copy only the actual data needed
        //     size_t actual_copy = std::min(to_read,
        //     static_cast<size_t>(bytes_read)); std::memcpy(dest_ptr +
        //     total_read, aligned_buffer, actual_copy);

        //     total_read += actual_copy;
        //     remaining -= actual_copy;
        // }

        // free(aligned_buffer);
        // close(fd);

        // auto end_time = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> elapsed = end_time - start_time;
        // this->logger->info("Load file {} in {:.2f} seconds (direct I/O)",
        //                 metadata.file_name, elapsed.count());

        // Step 3: Direct I/O reading directly to destination
        std::string bin_file_path =
            model_weights_path + "/" + metadata.file_name;

        auto start_time = std::chrono::high_resolution_clock::now();

        const size_t buffer_size = 32 * 1024 * 1024;  // 32MB
        const size_t alignment = 4096;                // 4KB alignment

        char* dest_ptr = static_cast<char*>(weight_ptr) + global_offset;

        // Check alignment status
        bool dest_aligned =
            (reinterpret_cast<uintptr_t>(dest_ptr) % alignment) == 0;
        bool size_aligned = (metadata.total_byte_size % alignment) == 0;

        // Log alignment status
        std::string alignment_info = "Alignment status: ";
        if (!dest_aligned) alignment_info += "dest_ptr NOT aligned, ";
        if (!size_aligned) alignment_info += "total_size NOT aligned, ";
        if (dest_aligned && size_aligned) alignment_info += "both aligned, ";
        std::cout << std::endl;
        this->logger->info(alignment_info + "using direct I/O for {} bytes",
                           dest_aligned ? metadata.total_byte_size : 0);

        if (!dest_aligned) {
            throw std::runtime_error(
                "Destination memory not aligned for direct I/O. "
                "Global offset alignment failed.");
        }

        // Calculate aligned and tail portions
        size_t aligned_size =
            (metadata.total_byte_size / alignment) * alignment;  // Round down
        size_t tail_size = metadata.total_byte_size - aligned_size;

        off_t file_offset = 0;

        // Part 1: Direct I/O for aligned portion
        if (aligned_size > 0) {
            int fd =
                open(bin_file_path.c_str(), O_RDONLY | O_DIRECT | O_NOATIME);
            if (fd == -1) {
                throw std::runtime_error(
                    "Failed to open file with direct I/O: " + bin_file_path);
            }

            posix_fadvise(fd, 0, aligned_size, POSIX_FADV_SEQUENTIAL);

            size_t remaining = aligned_size;
            while (remaining > 0) {
                size_t to_read = std::min(remaining, buffer_size);
                // Ensure read size is aligned (should already be, but just in
                // case)
                to_read = (to_read / alignment) * alignment;

                ssize_t bytes_read =
                    pread(fd, dest_ptr + file_offset, to_read, file_offset);
                if (bytes_read <= 0) {
                    close(fd);
                    throw std::runtime_error(
                        "Failed to read aligned portion from file");
                }

                file_offset += bytes_read;
                remaining -= bytes_read;
            }

            close(fd);
            this->logger->info("Direct I/O completed for {} bytes",
                               aligned_size);
        }

        // Part 2: Regular I/O for tail portion (if any)
        if (tail_size > 0) {
            std::ifstream tail_file(bin_file_path, std::ios::binary);
            if (!tail_file) {
                throw std::runtime_error("Failed to open file for tail read: " +
                                         bin_file_path);
            }

            // Seek to the tail portion
            tail_file.seekg(aligned_size);
            if (!tail_file) {
                throw std::runtime_error("Failed to seek to tail portion");
            }

            // Read the tail directly to destination
            tail_file.read(dest_ptr + file_offset, tail_size);
            if (!tail_file ||
                tail_file.gcount() != static_cast<std::streamsize>(tail_size)) {
                throw std::runtime_error("Failed to read tail portion (" +
                                         std::to_string(tail_size) + " bytes)");
            }

            tail_file.close();
            this->logger->info("Regular I/O completed for tail {} bytes",
                               tail_size);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        // Enhanced logging with alignment details
        std::string method_info;
        if (aligned_size > 0 && tail_size > 0) {
            method_info = "hybrid (direct I/O + regular I/O)";
        } else if (aligned_size > 0) {
            method_info = "direct I/O only";
        } else {
            method_info = "regular I/O only";
        }

        this->logger->info(
            "Load file {} in {:.2f} seconds ({}, {}/{} bytes aligned/total)",
            metadata.file_name, elapsed.count(), method_info, aligned_size,
            metadata.total_byte_size);

        // Step 4: Update this->module_weights_storage_ and
        // this->skeleton_state_dict_
        start_time = std::chrono::high_resolution_clock::now();
        for (const auto& [tensor_name, tensor_info] : metadata.state_dict) {
            // Check if the tensor_name exists in state_dict_name_map
            if (state_dict_name_map.find(tensor_name) !=
                state_dict_name_map.end()) {
                // If it exists, update module_weights_storage_
                std::vector<int64_t> tensor_shape = tensor_info.shape;
                int64_t tensor_global_offset =
                    global_offset + tensor_info.offset;
                this->module_weights_storage_
                    [state_dict_name_map[tensor_name]["module_key"]]
                    [state_dict_name_map[tensor_name]["tensor_key"]] =
                    tensor_meta(tensor_global_offset, tensor_shape,
                                tensor_info.byte_size);
            } else {
                // If it does not exist, update skeleton_state_dict_
                torch::Dtype dtype;
                if (tensor_info.dtype == "float32") {
                    dtype = torch::kFloat32;
                } else if (tensor_info.dtype == "float16") {
                    dtype = torch::kFloat16;
                } else if (tensor_info.dtype == "bfloat16") {
                    dtype = torch::kBFloat16;
                } else if (tensor_info.dtype == "float8_e4m3fn") {
                    dtype = torch::kFloat8_e4m3fn;
                } else {
                    throw std::runtime_error("Unsupported dtype: " +
                                             tensor_info.dtype);
                }
                // Create a tensor from the weight_ptr + global_offset +
                // tensor_info.offset
                auto option =
                    torch::TensorOptions()
                        .dtype(dtype)
                        .device(torch::kCPU)
                        .requires_grad(false)
                        .memory_format(torch::MemoryFormat::Contiguous);
                torch::Tensor tensor =
                    torch::from_blob(static_cast<char*>(weight_ptr) +
                                         global_offset + tensor_info.offset,
                                     tensor_info.shape, option);
                this->skeleton_state_dict_[tensor_name] = tensor;
            }
        }
        end_time = std::chrono::high_resolution_clock::now();
        elapsed = end_time - start_time;
        this->logger->info(
            "Updated module_weights_storage_ and skeleton_state_dict_ in "
            "{:.2f} seconds",
            elapsed.count());
        global_offset += metadata.total_byte_size;
        global_offset =
            (global_offset + 4095) & ~4095;  // Round up to 4KB boundary
    }
};

void Parameter_Server::Init(
    std::string& weight_shm_name, std::string& tensor_meta_shm_name,
    int64_t byte_size, std::string& model_weights_path,
    std::unordered_map<std::string,
                       std::unordered_map<std::string, std::string>>&
        state_dict_name_map) {
    this->logger->info("Parameter Server Initializing...");
    this->shm_name = weight_shm_name;
    this->tensor_meta_shm_name = tensor_meta_shm_name;

    size_t free_memory = 0;
    size_t total_memory = 0;
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
    this->logger->info(
        "Before pinned memory setting, GPU Memory Usage: {} GB / {} GB",
        (total_memory - free_memory) / (1024 * 1024 * 1024),
        total_memory / (1024 * 1024 * 1024));

    // Allocate shared pinned memory for weights
    void* weight_ptr = nullptr;
    weight_ptr =
        allocate_shared_pinned_memory(weight_shm_name, byte_size, true);
    this->byte_size_ = byte_size;
    this->weight_ptr_ = weight_ptr;

    // Load weights from the model_weights_path
    _load_cus_format_file_to_host_mem(model_weights_path, weight_ptr,
                                      state_dict_name_map);

    // Serialize the module weights storage to shared memory
    serialize_to_shared_memory(this->module_weights_storage_,
                               tensor_meta_shm_name);

    std::cout << std::endl;
    this->logger->info("Parameter Server Initialized.");
};

Parameter_Server::Parameter_Server() {
    this->logger = init_logger("info", "Parameter_Server");
};
// void Parameter_Server::Init(
//     std::string& weight_shm_name, std::string& tensor_meta_shm_name,
//     int64_t byte_size, std::string& model_weights_path,
//     std::unordered_map<std::string,
//                        std::unordered_map<std::string, std::string>>&
//         state_dict_name_map) {
//     this->logger->info("Parameter Server Initializing...");
//     this->shm_name = weight_shm_name;
//     this->tensor_meta_shm_name = tensor_meta_shm_name;
//     size_t free_memory = 0;
//     size_t total_memory = 0;
//     CUDA_CHECK(cudaSetDevice(0));
//     CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
//     this->logger->info("Before pinned memory setting, GPU Memory Usage: {} GB
//     / {} GB",
//                        (total_memory - free_memory) / (1024 * 1024 * 1024),
//                        total_memory / (1024 * 1024 * 1024));

//     void* weight_ptr = nullptr;
//     weight_ptr =
//         allocate_shared_pinned_memory(weight_shm_name, byte_size, true);
//     this->byte_size_ = byte_size;
//     this->weight_ptr_ = weight_ptr;
//     // Log gpu memory usage on device 0

//     CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
//     this->logger->info("After pinned memory setting, GPU Memory Usage: {} GB
//     / {} GB",
//                        (total_memory - free_memory) / (1024 * 1024 * 1024),
//                        total_memory / (1024 * 1024 * 1024));

//     int64_t offset = 0;
//     // std::vector<fs::path> paths;
//     // std::copy(fs::directory_iterator(model_weights_path),
//     //           fs::directory_iterator(), std::back_inserter(paths));
//     std::vector<std::string> paths;
//     DIR* dir = opendir(model_weights_path.c_str());
//     if (dir != nullptr) {
//         struct dirent* entry;
//         while ((entry = readdir(dir)) != nullptr) {
//             // Skip the "." and ".." directory entries
//             if (strcmp(entry->d_name, ".") != 0 &&
//                 strcmp(entry->d_name, "..") != 0) {
//                 std::string full_path =
//                     model_weights_path + "/" + entry->d_name;
//                 paths.push_back(full_path);
//             }
//         }
//         closedir(dir);
//     }
//     auto bar = tq::tqdm(paths);
//     bar.set_prefix("Loading Weights");
//     // for(const auto& entry : fs::directory_iterator(model_weights_path)){
//     py::object torch = py::module::import("torch");
//     py::object load = torch.attr("load");
//     for (const auto& file_path : bar) {
//         // std::string file_path = entry.string();
//         this->logger->info("Loading weights from: {}", file_path);
//         // auto tmp_state_dict = load_parameters(file_path);
//         // Load the .pt file by calling torch.load() with pybind11 and then
//         cast
//         // the result to a std::unordered_map<std::string, torch::Tensor>
//         auto start_time = std::chrono::high_resolution_clock::now();
//         py::object tmp_state_dict_py = load(file_path,
//         py::arg("weights_only") = true); py::dict tmp_state_dict_dict =
//         tmp_state_dict_py.cast<py::dict>(); auto end_time =
//         std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double> elapsed =
//             end_time - start_time;
//         this->logger->info("Loaded tensors in {:.2f} seconds from {}",
//                            elapsed.count(), file_path);

//         start_time = std::chrono::high_resolution_clock::now();
//         std::unordered_map<std::string, torch::Tensor> tmp_state_dict;
//         for (auto item : tmp_state_dict_dict.attr("items")()) {
//             // Cast the item to a tuple: (key, value)
//             py::tuple pair = item.cast<py::tuple>();
//             std::string key_str = py::cast<std::string>(pair[0]);
//             torch::Tensor tensor = py::cast<torch::Tensor>(pair[1]);
//             tmp_state_dict[key_str] = tensor;
//         }
//         end_time = std::chrono::high_resolution_clock::now();
//         elapsed = end_time - start_time;
//         this->logger->info("Converted {} tensors in {:.2f} seconds from {}",
//                            tmp_state_dict.size(), elapsed.count(),
//                            file_path);

//         start_time = std::chrono::high_resolution_clock::now();
//         int64_t byte_size = 0;
//         for (auto iter = tmp_state_dict.begin();
//              iter != tmp_state_dict.end();) {
//             auto& [key, value] = *iter;
//             byte_size += value.element_size() * value.numel();
//             iter++;
//         }
//         // this->logger->info("Byte Size: {}", byte_size);
//         // Copy to pinned memory in parallel
//         std::vector<std::future<void>> futures;
//         std::unordered_map<std::string, int64_t> tensor_offset;

//         for (auto iter = tmp_state_dict.begin();
//              iter != tmp_state_dict.end();) {
//             auto& [key, value] = *iter;
//             auto size = value.element_size() * value.numel();
//             futures.push_back(std::async(
//                 std::launch::async, [weight_ptr, value, offset, size]() {
//                     memcpy(weight_ptr + offset, value.data_ptr(), size);
//                 }));
//             tensor_offset[key] = offset;
//             offset += size;
//             iter++;
//         }
//         for (auto& future : futures) {
//             future.get();
//         }
//         end_time = std::chrono::high_resolution_clock::now();
//         elapsed = end_time - start_time;
//         this->logger->info("Copied {} tensors to pinned memory in {:.2f}
//         seconds from {}",
//                             tmp_state_dict.size(), elapsed.count(),
//                             file_path);

//         for (auto iter = tmp_state_dict.begin();
//              iter != tmp_state_dict.end();) {
//             auto& [key, value] = *iter;
//             if (state_dict_name_map.find(key) != state_dict_name_map.end()) {
//                 this->module_weights_storage_
//                     [state_dict_name_map[key]["module_key"]]
//                     [state_dict_name_map[key]["tensor_key"]] =
//                     tensor_meta(tensor_offset[key], value.sizes().vec(),
//                                 value.element_size() * value.numel());
//                 iter = tmp_state_dict.erase(iter);
//             } else {
//                 this->skeleton_state_dict_[key] = value;
//                 iter++;
//             }
//         }
//         // bar.progress(1);
//     }
//     serialize_to_shared_memory(this->module_weights_storage_,
//                                tensor_meta_shm_name);
//     this->logger->info("Parameter Server Initialized.");
// };

Parameter_Server::~Parameter_Server() {
    free_shared_pinned_memory(this->shm_name, this->weight_ptr_,
                              this->byte_size_, true);
    shm_unlink(this->shm_name.c_str());
    shm_unlink(this->tensor_meta_shm_name.c_str());
    this->logger->info("Parameter Server Destroyed.");
};

std::unordered_map<std::string, torch::Tensor>
Parameter_Server::get_skeleton_state_dict() {
    return this->skeleton_state_dict_;
};

int64_t Parameter_Server::byte_size() { return this->byte_size_; };

std::unordered_map<std::string, std::unordered_map<std::string, tensor_meta>>
Parameter_Server::module_weights_shm() {
    return this->module_weights_storage_;
};
