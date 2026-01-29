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

#include "spdlog/spdlog.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <algorithm>
#include <array>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <condition_variable>
#include <iomanip>
#include <memory>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <torch/cuda.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

#include "data_structures.h"
#include "utils.h"

namespace py = pybind11;

torch::ScalarType str_to_torch_dtype(const std::string& dtype_str) {
    static const std::unordered_map<std::string, torch::ScalarType> dtype_map = {
        {"float16", torch::kFloat16},
        {"float32", torch::kFloat32},
        {"bfloat16", torch::kBFloat16},
        {"float8_e4m3fn", torch::kFloat8_e4m3fn},
        {"float8_e5m2", torch::kFloat8_e5m2},
        {"uint8", torch::kUInt8}
    };

    auto it = dtype_map.find(dtype_str);
    if (it != dtype_map.end()) {
        return it->second;
    }
    
    // Default to float32 if not found
    return torch::kFloat32;
}

// Basic_Config parse_basic_config(const py::object& engine_config) {
//     Basic_Config basic_config;
//     py::object basic_config_obj = engine_config.attr("Basic_Config");
//     basic_config.log_level =
//         basic_config_obj.attr("log_level").cast<std::string>();
//     basic_config.device = basic_config_obj.attr("device").cast<int64_t>();
//     basic_config.device_torch =
//         torch::Device(torch::kCUDA, basic_config.device);
//     // basic_config.dtype_str =
//     //     basic_config_obj.attr("dtype_str").cast<std::string>();
//     basic_config.weight_dtype =
//         basic_config_obj.attr("weight_dtype").cast<std::string>();
//     basic_config.weight_dtype_torch =
//         str_to_torch_dtype(basic_config.weight_dtype);
//     basic_config.kv_dtype = 
//         basic_config_obj.attr("kv_dtype").cast<std::string>();
//     basic_config.kv_dtype_torch =
//         str_to_torch_dtype(basic_config.kv_dtype);
//     basic_config.activation_dtype =
//         basic_config_obj.attr("activation_dtype").cast<std::string>();
//     basic_config.activation_dtype_torch =
//         str_to_torch_dtype(basic_config.activation_dtype);

//     basic_config.attn_mode = basic_config_obj.attr("attn_mode").cast<int64_t>();
//     basic_config.num_threads =
//         basic_config_obj.attr("num_threads").cast<int64_t>();
//     basic_config.module_types =
//         basic_config_obj.attr("module_types").cast<std::vector<std::string>>();
//     basic_config.rank = 
//         basic_config_obj.attr("rank").cast<int64_t>();
//     basic_config.world_size = 
//         basic_config_obj.attr("world_size").cast<int64_t>();    
//     return basic_config;
// };

Basic_Config parse_basic_config(const py::object& engine_config) {
    Basic_Config basic_config;
    py::object basic_config_obj = engine_config.attr("Basic_Config");

    basic_config.log_level = basic_config_obj.attr("log_level").cast<std::string>();
    basic_config.device = basic_config_obj.attr("device").cast<int64_t>();
    basic_config.device_torch = torch::Device(torch::kCUDA, basic_config.device);
    basic_config.weight_dtype = basic_config_obj.attr("weight_dtype").cast<std::string>();
    basic_config.weight_dtype_torch = str_to_torch_dtype(basic_config.weight_dtype);
    basic_config.kv_dtype = basic_config_obj.attr("kv_dtype").cast<std::string>();
    basic_config.kv_dtype_torch = str_to_torch_dtype(basic_config.kv_dtype);
    basic_config.activation_dtype = basic_config_obj.attr("activation_dtype").cast<std::string>();
    basic_config.activation_dtype_torch = str_to_torch_dtype(basic_config.activation_dtype);
    basic_config.attn_mode = basic_config_obj.attr("attn_mode").cast<int64_t>();
    basic_config.num_threads = basic_config_obj.attr("num_threads").cast<int64_t>();
    basic_config.module_types = basic_config_obj.attr("module_types").cast<std::vector<std::string>>();
    basic_config.rank = basic_config_obj.attr("rank").cast<int64_t>();
    basic_config.world_size = basic_config_obj.attr("world_size").cast<int64_t>();

    return basic_config;
};

// KV_Storage_Config parse_kv_storage_config(const py::object& engine_config) {
//     KV_Storage_Config kv_storage_config;
//     py::object kv_storage_config_obj = engine_config.attr("KV_Storage_Config");
//     kv_storage_config.num_host_slots =
//         kv_storage_config_obj.attr("num_host_slots").cast<int64_t>();
//     kv_storage_config.reserved_length =
//         kv_storage_config_obj.attr("reserved_length").cast<int64_t>();
//     kv_storage_config.slot_byte_size =
//         kv_storage_config_obj.attr("slot_byte_size").cast<int64_t>();
//     kv_storage_config.storage_byte_size =
//         kv_storage_config_obj.attr("storage_byte_size").cast<int64_t>();
//     return kv_storage_config;
// };

KV_Storage_Config parse_kv_storage_config(const py::object& engine_config) {
    KV_Storage_Config kv_storage_config;
    py::object kv_storage_config_obj = engine_config.attr("KV_Storage_Config");

    kv_storage_config.num_host_slots = kv_storage_config_obj.attr("num_host_slots").cast<int64_t>();
    kv_storage_config.reserved_length = kv_storage_config_obj.attr("reserved_length").cast<int64_t>();
    kv_storage_config.slot_byte_size = kv_storage_config_obj.attr("slot_byte_size").cast<int64_t>();
    kv_storage_config.storage_byte_size = kv_storage_config_obj.attr("storage_byte_size").cast<int64_t>();

    return kv_storage_config;
};

// GPU_Buffer_Config parse_gpu_buffer_config(const py::object& engine_config) {
//     GPU_Buffer_Config gpu_buffer_config;
//     const py::object& gpu_buffer_config_obj =
//         engine_config.attr("GPU_Buffer_Config");
//     gpu_buffer_config.num_prefill_module_buffer =
//         gpu_buffer_config_obj.attr("num_prefill_module_buffer")
//             .cast<std::unordered_map<std::string, int64_t>>();
//     gpu_buffer_config.num_decoding_module_buffer =
//         gpu_buffer_config_obj.attr("num_decoding_module_buffer")
//             .cast<std::unordered_map<std::string, int64_t>>();
//     gpu_buffer_config.num_k_buffer =
//         gpu_buffer_config_obj.attr("num_k_buffer").cast<int64_t>();
//     gpu_buffer_config.num_v_buffer =
//         gpu_buffer_config_obj.attr("num_v_buffer").cast<int64_t>();

//     gpu_buffer_config.kv_buffer_num_tokens =
//         gpu_buffer_config_obj.attr("kv_buffer_num_tokens").cast<int64_t>();
//     py::dict module_shapes_py =
//         gpu_buffer_config_obj.attr("module_shapes").cast<py::dict>();
//     for (auto item : module_shapes_py) {
//         std::string module_type = item.first.cast<std::string>();
//         py::dict module_shape_dict = item.second.cast<py::dict>();
//         std::unordered_map<std::string, std::vector<int64_t>> module_shape;
//         for (auto item : module_shape_dict) {
//             std::string module_name = item.first.cast<std::string>();
//             std::vector<int64_t> shape =
//                 item.second.cast<std::vector<int64_t>>();
//             module_shape[module_name] = shape;
//         }
//         gpu_buffer_config.module_shapes[module_type] = module_shape;
//     }
//     return gpu_buffer_config;
// };

GPU_Buffer_Config parse_gpu_buffer_config(const py::object& engine_config) {
    GPU_Buffer_Config gpu_buffer_config;
    const py::object& gpu_buffer_config_obj = engine_config.attr("GPU_Buffer_Config");

    gpu_buffer_config.num_prefill_module_buffer =
        gpu_buffer_config_obj.attr("num_prefill_module_buffer")
            .cast<std::unordered_map<std::string, int64_t>>();
    gpu_buffer_config.num_decoding_module_buffer =
        gpu_buffer_config_obj.attr("num_decoding_module_buffer")
            .cast<std::unordered_map<std::string, int64_t>>();
    gpu_buffer_config.num_k_buffer = gpu_buffer_config_obj.attr("num_k_buffer").cast<int64_t>();
    gpu_buffer_config.num_v_buffer = gpu_buffer_config_obj.attr("num_v_buffer").cast<int64_t>();
    gpu_buffer_config.kv_buffer_num_tokens = gpu_buffer_config_obj.attr("kv_buffer_num_tokens").cast<int64_t>();

    py::dict module_shapes_py = gpu_buffer_config_obj.attr("module_shapes").cast<py::dict>();
    for (auto item : module_shapes_py) {
        std::string module_type = item.first.cast<std::string>();
        py::dict module_shape_dict = item.second.cast<py::dict>();
        std::unordered_map<std::string, std::vector<int64_t>> module_shape;
        for (auto inner_item : module_shape_dict) {
            std::string module_name = inner_item.first.cast<std::string>();
            std::vector<int64_t> shape = inner_item.second.cast<std::vector<int64_t>>();
            module_shape[module_name] = shape;
        }
        gpu_buffer_config.module_shapes[module_type] = module_shape;
    }

    // Parse per-module weight dtypes (optional, for mixed-dtype models like GPT-OSS)
    if (py::hasattr(gpu_buffer_config_obj, "weight_dtypes")) {
        py::dict weight_dtypes_py = gpu_buffer_config_obj.attr("weight_dtypes").cast<py::dict>();
        for (auto item : weight_dtypes_py) {
            std::string module_type = item.first.cast<std::string>();
            // PyTorch dtype is passed as torch.dtype object, cast to ScalarType
            torch::Dtype dtype = item.second.cast<torch::Dtype>();
            gpu_buffer_config.weight_dtypes[module_type] = dtype;
        }
    }

    // Parse per-tensor dtype overrides (optional, for mixed-dtype tensors within a module)
    // Format: {module_type: {tensor_name: torch.dtype}}
    if (py::hasattr(gpu_buffer_config_obj, "tensor_dtypes")) {
        py::dict tensor_dtypes_py = gpu_buffer_config_obj.attr("tensor_dtypes").cast<py::dict>();
        for (auto module_item : tensor_dtypes_py) {
            std::string module_type = module_item.first.cast<std::string>();
            py::dict tensors_dict = module_item.second.cast<py::dict>();
            std::unordered_map<std::string, torch::Dtype> tensor_dtypes_map;
            for (auto tensor_item : tensors_dict) {
                std::string tensor_name = tensor_item.first.cast<std::string>();
                torch::Dtype dtype = tensor_item.second.cast<torch::Dtype>();
                tensor_dtypes_map[tensor_name] = dtype;
            }
            gpu_buffer_config.tensor_dtypes[module_type] = tensor_dtypes_map;
        }
    }

    return gpu_buffer_config;
};

Module_Batching_Config parse_module_batching_config(
    const py::object& engine_config) {
    Module_Batching_Config module_batching_config;
    py::object module_batching_config_obj =
        engine_config.attr("Module_Batching_Config");
    return module_batching_config;
};

namespace {

std::size_t CheckedSize(long long value, std::string_view field_name) {
    if (value < 0) {
        std::ostringstream oss;
        oss << field_name << " must be non-negative (got " << value << ')';
        throw std::invalid_argument(oss.str());
    }
    return static_cast<std::size_t>(value);
}

HostPagedKVConfig parse_host_paged_kv_config(
    const py::object& engine_config) {
    HostPagedKVConfig config;
    if (!py::hasattr(engine_config, "Host_Paged_KV_Config")) {
        return config;
    }
    const py::object cfg = engine_config.attr("Host_Paged_KV_Config");
    config.shm_name = cfg.attr("shm_name").cast<std::string>();
    config.total_byte_size = CheckedSize(
        cfg.attr("total_byte_size").cast<long long>(),
        "Host_Paged_KV_Config.total_byte_size");
    config.num_layers = CheckedSize(
        cfg.attr("num_layers").cast<long long>(),
        "Host_Paged_KV_Config.num_layers");
    config.num_pages_per_layer = CheckedSize(
        cfg.attr("num_pages_per_layer").cast<long long>(),
        "Host_Paged_KV_Config.num_pages_per_layer");
    config.page_size = CheckedSize(cfg.attr("page_size").cast<long long>(),
                                   "Host_Paged_KV_Config.page_size");
    config.num_k_heads = CheckedSize(
        cfg.attr("num_k_heads").cast<long long>(),
        "Host_Paged_KV_Config.num_k_heads");
    config.k_head_dim = CheckedSize(
        cfg.attr("k_head_dim").cast<long long>(),
        "Host_Paged_KV_Config.k_head_dim");
    config.num_v_heads = CheckedSize(
        cfg.attr("num_v_heads").cast<long long>(),
        "Host_Paged_KV_Config.num_v_heads");
    config.v_head_dim = CheckedSize(
        cfg.attr("v_head_dim").cast<long long>(),
        "Host_Paged_KV_Config.v_head_dim");
    config.kv_dtype = cfg.attr("kv_dtype").cast<std::string>();
    std::cout << "Host Paged KV Config Parsed Successfully" << std::endl;
    return config;
}

DevicePagedKVConfig parse_device_paged_kv_config(
    const py::object& engine_config) {
    DevicePagedKVConfig config;
    if (!py::hasattr(engine_config, "Device_Paged_KV_Config")) {
        return config;
    }
    const py::object cfg = engine_config.attr("Device_Paged_KV_Config");
    config.num_layers = CheckedSize(
        cfg.attr("num_layers").cast<long long>(),
        "Device_Paged_KV_Config.num_layers");
    config.num_pages_per_layer = CheckedSize(
        cfg.attr("num_pages_per_layer").cast<long long>(),
        "Device_Paged_KV_Config.num_pages_per_layer");
    config.page_size = CheckedSize(
        cfg.attr("page_size").cast<long long>(),
        "Device_Paged_KV_Config.page_size");
    config.num_k_heads = CheckedSize(
        cfg.attr("num_k_heads").cast<long long>(),
        "Device_Paged_KV_Config.num_k_heads");
    config.k_head_dim = CheckedSize(
        cfg.attr("k_head_dim").cast<long long>(),
        "Device_Paged_KV_Config.k_head_dim");
    config.num_v_heads = CheckedSize(
        cfg.attr("num_v_heads").cast<long long>(),
        "Device_Paged_KV_Config.num_v_heads");
    config.v_head_dim = CheckedSize(
        cfg.attr("v_head_dim").cast<long long>(),
        "Device_Paged_KV_Config.v_head_dim");
    config.kv_dtype = cfg.attr("kv_dtype").cast<std::string>();
    std::cout << "Device Paged KV Config Parsed Successfully" << std::endl;
    return config;
}

constexpr double kKilobyte = 1024.0;
constexpr double kMegabyte = kKilobyte * 1024.0;
constexpr double kGigabyte = kMegabyte * 1024.0;

}  // namespace

EngineConfig parse_engine_config(const py::object& engine_config) {
    // std::cerr << "Parsing EngineConfig" << std::endl;
    EngineConfig config;
    config.basic_config = parse_basic_config(engine_config);
    // std::cerr << "Basic Config Done" << std::endl;
    config.kv_storage_config = parse_kv_storage_config(engine_config);
    // std::cerr << "KV Storage Config Done" << std::endl;
    config.gpu_buffer_config = parse_gpu_buffer_config(engine_config);
    // std::cerr << "GPU Buffer Config Done" << std::endl;
    config.module_batching_config = parse_module_batching_config(engine_config);
    return config;
};

ModelConfig parse_model_config(const py::object& model_config) {
    // std::cerr << "Parsing ModelConfig" << std::endl;
    ModelConfig config;
    config.model_type = model_config.attr("model_type").cast<std::string>();
    config.num_hidden_layers =
        model_config.attr("num_hidden_layers").cast<int64_t>();
    config.num_local_experts =
        model_config.attr("num_local_experts").cast<int64_t>();
    config.num_attention_heads =
        model_config.attr("num_attention_heads").cast<int64_t>();
    config.num_key_value_heads =
        model_config.attr("num_key_value_heads").cast<int64_t>();
    // config.hidden_size = model_config.attr("hidden_size").cast<int64_t>();
    // config.intermediate_size =
    // model_config.attr("intermediate_size").cast<int64_t>();
    config.head_dim = model_config.attr("head_dim").cast<int64_t>();
    if ((config.model_type == "deepseek_v2") ||
        (config.model_type == "deepseek_v3")) {
        config.compressed_kv_dim =
            model_config.attr("compressed_kv_dim").cast<int64_t>();
    }
    // std::cerr << "Parsing ModelConfig Done" << std::endl;
    return config;
};

std::shared_ptr<spdlog::logger> init_logger(const std::string& log_level,
                                            const std::string& logger_name) {
    auto logger = spdlog::stdout_color_mt(logger_name);

    // Set colors for all five standard levels
    auto console_sink = dynamic_cast<spdlog::sinks::stdout_color_sink_mt*>(
        logger->sinks()[0].get());
    if (console_sink) {
        console_sink->set_color(spdlog::level::trace, console_sink->white);
        console_sink->set_color(spdlog::level::debug, console_sink->cyan);
        console_sink->set_color(spdlog::level::info, console_sink->green);
        console_sink->set_color(spdlog::level::warn, console_sink->yellow);
        console_sink->set_color(spdlog::level::err, console_sink->red);
    }

    // Set the log level
    logger->set_level(spdlog::level::from_str(log_level));

    // Set the pattern to match Python logging format:
    // 2026-01-10 10:51:29,379 - [LoggerName] - INFO - message
    // Note: spdlog %l gives lowercase level (info), %^%l%$ adds color
    if (logger->level() <= spdlog::level::debug) {
        logger->set_pattern("%Y-%m-%d %H:%M:%S,%e - [%n] - %^%l%$ - %v");
    } else {
        logger->set_pattern("%Y-%m-%d %H:%M:%S,%e - [%n] - %^%l%$ - %v");
    }

    // Make this logger the default logger
    // spdlog::set_default_logger(logger);

    // Optional: load log levels from the environment
    // spdlog::cfg::load_env_levels();
    logger->flush_on(spdlog::level::trace);
    return std::shared_ptr<spdlog::logger>(logger);
};
std::string get_tensor_shape(const torch::Tensor& tensor, 
                             bool include_dtype,
                             bool include_device) {
    std::ostringstream shape_str;
    
    // Get the dimensions
    auto sizes = tensor.sizes();
    shape_str << "[";
    for (size_t i = 0; i < sizes.size(); ++i) {
        shape_str << sizes[i];
        if (i < sizes.size() - 1) {
            shape_str << ", ";
        }
    }
    shape_str << "]";
    
    // Add dtype if requested
    if (include_dtype) {
        shape_str << ", dtype=" << torch::toString(tensor.scalar_type());
    }
    
    // Add device if requested
    if (include_device) {
        shape_str << ", device=";
        if (tensor.device().is_cuda()) {
            shape_str << "cuda:" << tensor.device().index();
        } else if (tensor.device().is_cpu()) {
            shape_str << "cpu";
        } else {
            shape_str << torch::toString(tensor.device().type());
        }
    }
    
    return shape_str.str();
}

double BytesToKilobytes(std::size_t bytes) {
    return static_cast<double>(bytes) / kKilobyte;
}

double BytesToMegabytes(std::size_t bytes) {
    return static_cast<double>(bytes) / kMegabyte;
}

double BytesToGigabytes(std::size_t bytes) {
    return static_cast<double>(bytes) / kGigabyte;
}
