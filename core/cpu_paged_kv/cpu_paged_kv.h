#include "spdlog/spdlog.h"
#include <c10/util/BFloat16.h>
#include <memory>
#include <string>
#include <torch/extension.h>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

#include "../data_structures.h"
#include "../utils.h"
#include "KV_Storage.h"
#include "tqdm.hpp"
#include <random>
#include <cstdint>
#include <cstring>
#include <numa.h>
#include <numaif.h>
#include <cstdlib>       // posix_memalign      
#include <cuda_runtime.h>
#include <unistd.h>      // getpagesize
#include <linux/mman.h>  // For MAP_HUGE_2MB

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
// Fallback calculation if MAP_HUGE_2MB is not directly available
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#endif


class CPUPagedKVManager