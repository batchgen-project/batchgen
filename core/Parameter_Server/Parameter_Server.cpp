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

#include "../data_structures.h"
#include "../utils.h"
#include "Parameter_Server.h"
#include "posix_shm.h"
#include "tqdm.hpp"

// namespace fs = std::filesystem;
namespace py = pybind11;

// std::vector<char> get_the_bytes(std::string filename) {
//     int64_t filesize = fs::file_size(filename);
//     std::vector<char> bytes(filesize);

//     int fd = open(filename.c_str(), O_RDONLY);
//     if (fd == -1) {
//         throw std::runtime_error("Error opening file");
//     }

//     auto read_chunk = [&](int64_t offset, int64_t size) {
//         int64_t bytes_read = 0;
//         while (bytes_read < size) {
//             int64_t read_size = pread(fd, bytes.data() + offset + bytes_read,
//                                       size - bytes_read, offset +
//                                       bytes_read);
//             if (read_size == -1) {
//                 throw std::runtime_error("Error reading file");
//             }
//             bytes_read += read_size;
//         }
//     };

//     int num_threads = 16;

//     std::vector<std::future<void>> futures;
//     int64_t chunk_size = filesize / num_threads;
//     for (int i = 0; i < num_threads; ++i) {
//         int64_t offset = i * chunk_size;
//         int64_t size =
//             (i == num_threads - 1) ? (filesize - offset) : chunk_size;
//         futures.push_back(
//             std::async(std::launch::async, read_chunk, offset, size));
//     }

//     for (auto& future : futures) {
//         future.get();
//     }

//     close(fd);
//     return bytes;
// }

// std::unordered_map<std::string, torch::Tensor> load_parameters(
//     std::string pt_pth) {
//     std::vector<char> f = get_the_bytes(pt_pth);
//     c10::Dict<c10::IValue, c10::IValue> weights =
//         torch::pickle_load(f).toGenericDict();

//     // const torch::OrderedDict<std::string, at::Tensor>& model_params =
//     // this->named_parameters();
//     std::unordered_map<std::string, torch::Tensor> model_params;

//     torch::NoGradGuard no_grad;
//     for (auto const& w : weights) {
//         std::string name = w.key().toStringRef();
//         at::Tensor param = w.value().toTensor();

//         model_params[name] = param;
//     }
//     return model_params;
// }

Parameter_Server::Parameter_Server() {
    this->logger = init_logger("info", "Parameter_Server");
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
    this->logger->info("Before pinned memory setting, GPU Memory Usage: {} GB / {} GB",
                       (total_memory - free_memory) / (1024 * 1024 * 1024),
                       total_memory / (1024 * 1024 * 1024));
    

    void* weight_ptr = nullptr;
    weight_ptr =
        allocate_shared_pinned_memory(weight_shm_name, byte_size, true);
    this->byte_size_ = byte_size;
    this->weight_ptr_ = weight_ptr;
    // Log gpu memory usage on device 0

    
    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
    this->logger->info("After pinned memory setting, GPU Memory Usage: {} GB / {} GB",
                       (total_memory - free_memory) / (1024 * 1024 * 1024),
                       total_memory / (1024 * 1024 * 1024));
    int64_t offset = 0;
    // std::vector<fs::path> paths;
    // std::copy(fs::directory_iterator(model_weights_path),
    //           fs::directory_iterator(), std::back_inserter(paths));
    std::vector<std::string> paths;
    DIR* dir = opendir(model_weights_path.c_str());
    if (dir != nullptr) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            // Skip the "." and ".." directory entries
            if (strcmp(entry->d_name, ".") != 0 &&
                strcmp(entry->d_name, "..") != 0) {
                std::string full_path =
                    model_weights_path + "/" + entry->d_name;
                paths.push_back(full_path);
            }
        }
        closedir(dir);
    }
    auto bar = tq::tqdm(paths);
    bar.set_prefix("Loading Weights");
    // for(const auto& entry : fs::directory_iterator(model_weights_path)){
    for (const auto& file_path : bar) {
        // std::string file_path = entry.string();
        this->logger->info("Loading weights from: {}", file_path);
        // auto tmp_state_dict = load_parameters(file_path);
        // Load the .pt file by calling torch.load() with pybind11 and then cast
        // the result to a std::unordered_map<std::string, torch::Tensor>
        py::object torch = py::module::import("torch");
        py::object load = torch.attr("load");
        py::object tmp_state_dict_py = load(file_path);
        py::dict tmp_state_dict_dict = tmp_state_dict_py.cast<py::dict>();

        std::unordered_map<std::string, torch::Tensor> tmp_state_dict;
        for (auto item : tmp_state_dict_dict.attr("items")()) {
            // Cast the item to a tuple: (key, value)
            py::tuple pair = item.cast<py::tuple>();
            std::string key_str = py::cast<std::string>(pair[0]);
            torch::Tensor tensor = py::cast<torch::Tensor>(pair[1]);
            tmp_state_dict[key_str] = tensor;
        }

        int64_t byte_size = 0;
        for (auto iter = tmp_state_dict.begin();
             iter != tmp_state_dict.end();) {
            auto& [key, value] = *iter;
            byte_size += value.element_size() * value.numel();
            iter++;
        }
        // this->logger->info("Byte Size: {}", byte_size);
        // Copy to pinned memory in parallel
        std::vector<std::future<void>> futures;
        std::unordered_map<std::string, int64_t> tensor_offset;

        for (auto iter = tmp_state_dict.begin();
             iter != tmp_state_dict.end();) {
            auto& [key, value] = *iter;
            auto size = value.element_size() * value.numel();
            futures.push_back(std::async(
                std::launch::async, [weight_ptr, value, offset, size]() {
                    memcpy(weight_ptr + offset, value.data_ptr(), size);
                }));
            tensor_offset[key] = offset;
            offset += size;
            iter++;
        }
        for (auto& future : futures) {
            future.get();
        }

        for (auto iter = tmp_state_dict.begin();
             iter != tmp_state_dict.end();) {
            // this->logger->info("Key: {}", iter->first);
            auto& [key, value] = *iter;
            if (state_dict_name_map.find(key) != state_dict_name_map.end()) {
                // this->logger->info("Adding to storage: {}", key);
                // this->add_module_to_storage(
                // 	state_dict_name_map[key]["module_key"],
                // 	state_dict_name_map[key]["tensor_key"],
                // 	weight_ptr+tensor_offset[key],
                // 	value.sizes().vec(),
                // 	value.element_size() * value.numel()
                // );
                this->module_weights_storage_
                    [state_dict_name_map[key]["module_key"]]
                    [state_dict_name_map[key]["tensor_key"]] =
                    tensor_meta(tensor_offset[key], value.sizes().vec(),
                                value.element_size() * value.numel());
                iter = tmp_state_dict.erase(iter);
            } else {
                this->skeleton_state_dict_[key] = value;
                iter++;
            }
        }
        // bar.progress(1);
    }
    serialize_to_shared_memory(this->module_weights_storage_,
                               tensor_meta_shm_name);
    this->logger->info("Parameter Server Initialized.");
};

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
