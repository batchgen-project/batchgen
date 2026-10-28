#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Read plan of the parameter server's custom-format weight load, kept free of
// torch, CUDA, pybind and JSON so it can be compiled and tested on its own.
//
// Parameter_Server reads every <name>.json-described .bin file of a directory
// into one fixed buffer, back to back, each file starting on a 4 KiB boundary.
// plan_weight_load() computes those start offsets and rejects the whole load,
// before the first byte is read, when any file (or any tensor inside a file)
// would end past the buffer. check_weight_read_bound() guards each single read.

namespace batchgen {

// pybind11 maps this to RuntimeError; every message starts with the class name.
class WeightLoadPlanError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct WeightTensorExtent {
    std::string name;
    int64_t offset;     // byte offset inside its file
    int64_t byte_size;
};

struct WeightFileExtent {
    std::string json_path;
    std::string file_name;  // the .bin file the JSON describes
    int64_t total_byte_size;
    std::vector<WeightTensorExtent> tensors;
};

constexpr int64_t kWeightFileAlignment = 4096;

inline int64_t weight_plan_saturating_add(int64_t a, int64_t b) {
    if (b > std::numeric_limits<int64_t>::max() - a) {
        return std::numeric_limits<int64_t>::max();
    }
    return a + b;
}

inline int64_t weight_plan_align_up(int64_t value) {
    const int64_t padded = weight_plan_saturating_add(value, kWeightFileAlignment - 1);
    return padded & ~(kWeightFileAlignment - 1);
}

// Planned bytes: end of the last file, with every earlier file padded to 4 KiB.
inline int64_t planned_weight_load_bytes(const std::vector<WeightFileExtent>& files) {
    int64_t offset = 0;
    int64_t end = 0;
    for (const auto& file : files) {
        end = weight_plan_saturating_add(offset, file.total_byte_size);
        offset = weight_plan_align_up(end);
    }
    return end;
}

// Returns the buffer offset of each file, in input order. Throws
// WeightLoadPlanError before anything is read when a size is negative, a tensor
// lies outside its file, or a file would end past buffer_bytes.
inline std::vector<int64_t> plan_weight_load(const std::vector<WeightFileExtent>& files,
                                             int64_t buffer_bytes) {
    if (buffer_bytes <= 0) {
        throw WeightLoadPlanError("WeightLoadPlanError: weight buffer size must be positive, got " +
                                  std::to_string(buffer_bytes));
    }
    for (const auto& file : files) {
        if (file.total_byte_size < 0) {
            throw WeightLoadPlanError("WeightLoadPlanError: " + file.json_path +
                                      " declares a negative total_byte_size " +
                                      std::to_string(file.total_byte_size));
        }
        for (const auto& tensor : file.tensors) {
            if (tensor.offset < 0 || tensor.byte_size < 0 ||
                tensor.offset > file.total_byte_size - tensor.byte_size) {
                throw WeightLoadPlanError(
                    "WeightLoadPlanError: tensor " + tensor.name + " in " + file.json_path +
                    " spans [" + std::to_string(tensor.offset) + ", " +
                    std::to_string(weight_plan_saturating_add(tensor.offset, tensor.byte_size)) +
                    ") outside the file's total_byte_size " +
                    std::to_string(file.total_byte_size));
            }
        }
    }

    const int64_t planned_total = planned_weight_load_bytes(files);
    std::vector<int64_t> starts;
    starts.reserve(files.size());
    int64_t offset = 0;
    for (const auto& file : files) {
        const int64_t end = weight_plan_saturating_add(offset, file.total_byte_size);
        if (end > buffer_bytes) {
            throw WeightLoadPlanError(
                "WeightLoadPlanError: planned weight load of " + std::to_string(planned_total) +
                " bytes over " + std::to_string(files.size()) + " files exceeds the " +
                std::to_string(buffer_bytes) + "-byte weight buffer; first file past the bound: " +
                file.json_path + " (" + file.file_name + ", " +
                std::to_string(file.total_byte_size) + " bytes at buffer offset " +
                std::to_string(offset) + ", ending at " + std::to_string(end) + ")");
        }
        starts.push_back(offset);
        offset = weight_plan_align_up(end);
    }
    return starts;
}

// Guard for one read of nbytes into the buffer at dest_offset.
inline void check_weight_read_bound(int64_t dest_offset, int64_t nbytes, int64_t buffer_bytes,
                                    const std::string& file_path) {
    if (dest_offset < 0 || nbytes < 0 || dest_offset > buffer_bytes - nbytes) {
        throw WeightLoadPlanError(
            "WeightLoadPlanError: read of " + std::to_string(nbytes) + " bytes from " + file_path +
            " at buffer offset " + std::to_string(dest_offset) + " would pass the " +
            std::to_string(buffer_bytes) + "-byte weight buffer");
    }
}

}  // namespace batchgen
