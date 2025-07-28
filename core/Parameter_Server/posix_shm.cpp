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

// **CONSOLIDATED: All required includes (duplicates removed)**
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <linux/memfd.h>
#include <linux/mman.h>
#include <numa.h>
#include <numaif.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cuda_runtime_api.h>
#include "../utils.h"
#include "posix_shm.h"
#include "spdlog/spdlog.h"
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#endif

std::shared_ptr<spdlog::logger> logger = init_logger("info", "Server");

bool check_hugepage_availability(int64_t required_size) {
    std::ifstream meminfo("/proc/meminfo");
    if (!meminfo) {
        logger->error("Failed to open /proc/meminfo");
        return false;
    }
    
    std::string line;
    long hugepage_size = 0, hugepages_free = 0, hugepages_total = 0;
    
    while (std::getline(meminfo, line)) {
        if (line.find("Hugepagesize:") == 0) {
            std::istringstream iss(line);
            std::string key, value, unit;
            iss >> key >> value >> unit;
            hugepage_size = std::stol(value) * 1024; // Convert KB to bytes
        } else if (line.find("HugePages_Free:") == 0) {
            std::istringstream iss(line);
            std::string key, value;
            iss >> key >> value;
            hugepages_free = std::stol(value);
        } else if (line.find("HugePages_Total:") == 0) {
            std::istringstream iss(line);
            std::string key, value;
            iss >> key >> value;
            hugepages_total = std::stol(value);
        }
    }
    
    long available_bytes = hugepages_free * hugepage_size;
    long total_bytes = hugepages_total * hugepage_size;
    
    logger->info("Huge page status: size={}MB, total={} ({}GB), free={} ({}GB), required={}GB",
                 hugepage_size / (1024*1024),
                 hugepages_total, total_bytes / (1024*1024*1024),
                 hugepages_free, available_bytes / (1024*1024*1024),
                 required_size / (1024*1024*1024));
    
    if (available_bytes < required_size) {
        long required_pages = (required_size + hugepage_size - 1) / hugepage_size;
        logger->warn("Insufficient huge pages: need {} pages, only {} available. "
                     "Consider: echo {} > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages",
                     required_pages, hugepages_free, 
                     hugepages_total + required_pages - hugepages_free);
        return false;
    }
    
    return true;
}

void* try_hugetlbfs_allocation(const std::string& shm_name, int64_t size, bool create) {
    std::string hugepage_path = "/dev/hugepages/" + shm_name;
    int flags = O_RDWR | (create ? O_CREAT : 0);
    
    int fd = open(hugepage_path.c_str(), flags, 0666);
    if (fd < 0) {
        return nullptr;
    }
    
    if (create) {
        if (ftruncate64(fd, size) == -1) {
            close(fd);
            unlink(hugepage_path.c_str());
            return nullptr;
        }
    } else {
        // Verify size for workers
        struct stat stat_buf;
        if (fstat(fd, &stat_buf) == -1 || stat_buf.st_size != size) {
            close(fd);
            return nullptr;
        }
    }
    
    void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    
    if (ptr == MAP_FAILED) {
        if (create) unlink(hugepage_path.c_str());
        return nullptr;
    }
    
    return ptr;
}

void* try_shm_allocation(const std::string& shm_name, int64_t size, bool create, bool use_hugepages) {
    int flags = O_RDWR | (create ? O_CREAT : 0);
    
    int fd = shm_open(shm_name.c_str(), flags, 0666);
    if (fd < 0) {
        throw std::runtime_error(fmt::format("shm_open failed for {}: {}", 
                                           shm_name, strerror(errno)));
    }
    
    if (create) {
        if (ftruncate64(fd, size) == -1) {
            close(fd);
            throw std::runtime_error(fmt::format("ftruncate failed for {}: {}",
                                               shm_name, strerror(errno)));
        }
    } else {
        // Verify size for workers
        struct stat stat_buf;
        if (fstat(fd, &stat_buf) == -1) {
            close(fd);
            throw std::runtime_error(fmt::format("fstat failed for {}: {}",
                                               shm_name, strerror(errno)));
        }
        if (stat_buf.st_size != size) {
            close(fd);
            throw std::runtime_error(fmt::format("Size mismatch: expected {} bytes, found {} bytes",
                                               size, stat_buf.st_size));
        }
    }
    
    void* ptr = nullptr;
    
    // Try huge pages if requested (server only)
    if (create && use_hugepages) {
        ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, 
                   MAP_SHARED | MAP_HUGETLB | MAP_HUGE_2MB, fd, 0);
        if (ptr != MAP_FAILED) {
            close(fd);
            return ptr;
        }
        // Fall through to regular mmap if huge pages fail
    }
    
    // Regular mmap
    ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    
    if (ptr == MAP_FAILED) {
        throw std::runtime_error(fmt::format("mmap failed for {}: {}", 
                                           shm_name, strerror(errno)));
    }
    
    return ptr;
}

void touch_pages(void* ptr, int64_t size, size_t page_size) {
    auto start = std::chrono::high_resolution_clock::now();
    
    int num_threads = std::min(16, std::max(2, (int)std::thread::hardware_concurrency() / 2));
    int64_t total_pages = size / page_size;
    int64_t pages_per_thread = total_pages / num_threads;
    
    logger->info("Touching {} pages with {} threads", total_pages, num_threads);
    
    std::vector<std::thread> threads;
    std::atomic<int64_t> completed_pages{0};
    
    auto touch_worker = [&](int thread_id) {
        int64_t start_page = thread_id * pages_per_thread;
        int64_t end_page = (thread_id == num_threads - 1) ? total_pages : start_page + pages_per_thread;
        
        volatile char* p = reinterpret_cast<volatile char*>(ptr);
        
        for (int64_t page = start_page; page < end_page; page++) {
            p[page * page_size] = 0;
            if (page % 1000 == 0) {
                completed_pages.fetch_add(1000);
            }
        }
        completed_pages.fetch_add((end_page - start_page) % 1000);
    };
    
    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back(touch_worker, i);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    // Touch last byte
    volatile char* p = reinterpret_cast<volatile char*>(ptr);
    p[size - 1] = 0;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double throughput = (double)size / (1024.0 * 1024.0 * 1024.0) / (duration.count() / 1000.0);
    
    logger->info("Page touching completed: {}GB in {}ms ({:.2f} GB/s)",
                 size / (1024*1024*1024), duration.count(), throughput);
}

void setup_numa_interleaving() {
    if (numa_available() < 0) {
        logger->debug("NUMA not available");
        return;
    }
    
    int num_nodes = numa_num_configured_nodes();
    logger->debug("NUMA available with {} nodes", num_nodes);
    
    if (num_nodes >= 2) {
        struct bitmask* nodemask = numa_allocate_nodemask();
        if (!nodemask) {
            logger->warn("Failed to allocate nodemask");
            return;
        }
        
        numa_bitmask_clearall(nodemask);
        numa_bitmask_setbit(nodemask, 0);
        numa_bitmask_setbit(nodemask, 1);
        
        if (set_mempolicy(MPOL_INTERLEAVE, nodemask->maskp, nodemask->size + 1) == 0) {
            logger->info("NUMA memory interleaving enabled for nodes 0 and 1");
        } else {
            logger->warn("Failed to set NUMA memory policy: {}", strerror(errno));
        }
        
        numa_free_nodemask(nodemask);
    }
}

void* allocate_shared_pinned_memory(const std::string& shm_name,
                                    int64_t size,
                                    bool create) {
    if (size <= 0) {
        throw std::runtime_error("Invalid size: " + std::to_string(size));
    }
    
    const size_t regular_page_size = sysconf(_SC_PAGESIZE);
    const size_t huge_page_size = 2 * 1024 * 1024; // 2MB
    
    // Determine alignment
    bool plan_huge_pages = false;
    int64_t aligned_size;
    
    if (create) {
        // Server: check huge page availability and align accordingly
        plan_huge_pages = check_hugepage_availability(size);
        aligned_size = plan_huge_pages 
            ? ((size + huge_page_size - 1) / huge_page_size) * huge_page_size
            : ((size + regular_page_size - 1) / regular_page_size) * regular_page_size;
        
        logger->info("Server: allocating {}GB (aligned to {}GB) with {} pages",
                     size / (1024.0*1024*1024), 
                     aligned_size / (1024.0*1024*1024),
                     plan_huge_pages ? "huge" : "regular");
    } else {
        // Worker: always align to huge page boundaries for compatibility
        aligned_size = ((size + huge_page_size - 1) / huge_page_size) * huge_page_size;
        logger->info("Worker: connecting to {}GB shared memory", 
                     aligned_size / (1024.0*1024*1024));
    }
    
    void* ptr = nullptr;
    bool using_huge_pages = false;
    
    // Try hugetlbfs first (both server and worker)
    ptr = try_hugetlbfs_allocation(shm_name, aligned_size, create);
    if (ptr) {
        using_huge_pages = true;
        logger->info("{} using hugetlbfs (2MB pages)", create ? "Server created" : "Worker connected to");
    } else {
        // Fall back to regular shared memory
        ptr = try_shm_allocation(shm_name, aligned_size, create, plan_huge_pages);
        
        if (create && plan_huge_pages && ((uintptr_t)ptr % huge_page_size == 0)) {
            using_huge_pages = true;
            logger->info("Server created using shm_open with MAP_HUGETLB");
        } else if (!create && ((uintptr_t)ptr % huge_page_size == 0)) {
            using_huge_pages = true;
            logger->info("Worker connected to huge page memory");
        } else {
            logger->info("{} using regular shared memory", create ? "Server created" : "Worker connected to");
            
            if (create) {
                // Try transparent huge pages
                if (madvise(ptr, aligned_size, MADV_HUGEPAGE) == 0) {
                    logger->debug("Enabled transparent huge pages hint");
                }
            }
        }
    }
    
    // Verify alignment
    size_t actual_alignment = using_huge_pages ? huge_page_size : regular_page_size;
    if ((uintptr_t)ptr % actual_alignment != 0) {
        if ((uintptr_t)ptr % regular_page_size != 0) {
            munmap(ptr, aligned_size);
            throw std::runtime_error("Memory is not page-aligned, CUDA operations will fail");
        }
        logger->warn("Memory not optimally aligned but meets minimum requirements");
    }
    
    // Server-only initialization
    if (create) {
        setup_numa_interleaving();
        touch_pages(ptr, aligned_size, using_huge_pages ? huge_page_size : regular_page_size);
    }
    
    logger->info("Successfully {} {}GB shared memory at {}",
                 create ? "allocated" : "mapped",
                 aligned_size / (1024.0*1024*1024),
                 ptr);
    
    return ptr;
}

// Helper function to verify NUMA allocation
void verify_numa_allocation(void* ptr, size_t size) {
    if (numa_available() < 0) {
        std::cout << "NUMA not available for verification" << std::endl;
        return;
    }

    // Check a few sample pages
    long page_size = sysconf(_SC_PAGESIZE);
    int num_samples = std::min(10L, (long)(size / page_size));
    
    std::cout << "Verifying NUMA allocation for " << num_samples << " sample pages:" << std::endl;
    
    for (int i = 0; i < num_samples; i++) {
        void* page_addr = (char*)ptr + (i * size / num_samples);
        int node = -1;
        
        if (get_mempolicy(&node, nullptr, 0, page_addr, MPOL_F_NODE | MPOL_F_ADDR) == 0) {
            std::cout << "Page at offset " << (i * size / num_samples) 
                      << " is on NUMA node " << node << std::endl;
        } else {
            perror("get_mempolicy failed");
        }
    }
}


void free_shared_pinned_memory(std::string& shm_name, void* ptr, int64_t size,
                               bool create) {
    cudaHostUnregister(ptr);
    munmap(ptr, size);
    shm_unlink(shm_name.c_str());
}

// -----------------------------------------------------------------------------
// Helper: Compute required serialized size
// (we write sizes and raw bytes for each string/vector)
// -----------------------------------------------------------------------------
size_t compute_serialized_size(
    const std::unordered_map<
        std::string, std::unordered_map<std::string, tensor_meta>>& map) {
    size_t total_size = 0;
    total_size += sizeof(size_t);  // outer map size

    for (const auto& outer : map) {
        total_size +=
            sizeof(size_t) + outer.first.size();  // outer key (length + bytes)
        total_size += sizeof(size_t);             // inner map size

        for (const auto& inner : outer.second) {
            total_size += sizeof(size_t) +
                          inner.first.size();  // inner key (length + bytes)
            total_size += sizeof(int64_t) *
                          2;  // tensor_meta.offset and tensor_meta.byte_size
            total_size += sizeof(size_t);  // tensor_shape vector length
            total_size += inner.second.tensor_shape.size() *
                          sizeof(int64_t);  // vector data
        }
    }
    return total_size;
}

// -----------------------------------------------------------------------------
// Simple Serialization: Write the map into a preallocated buffer.
// Throws std::runtime_error if the buffer is too small.
// -----------------------------------------------------------------------------
void serialize_map_to_buffer(
    const std::unordered_map<std::string,
                             std::unordered_map<std::string, tensor_meta>>& map,
    char* buffer, size_t buffer_size) {
    char* ptr = buffer;

    // Write outer map size.
    size_t outer_size = map.size();
    if (ptr + sizeof(size_t) > buffer + buffer_size)
        throw std::runtime_error("Buffer overflow (outer_size)");
    std::memcpy(ptr, &outer_size, sizeof(size_t));
    ptr += sizeof(size_t);

    // For each outer map element:
    for (const auto& outer : map) {
        // Write outer key: first its length, then its characters.
        size_t key_len = outer.first.size();
        if (ptr + sizeof(size_t) + key_len > buffer + buffer_size)
            throw std::runtime_error("Buffer overflow (outer key)");
        std::memcpy(ptr, &key_len, sizeof(size_t));
        ptr += sizeof(size_t);
        std::memcpy(ptr, outer.first.data(), key_len);
        ptr += key_len;

        // Write inner map size.
        size_t inner_size = outer.second.size();
        if (ptr + sizeof(size_t) > buffer + buffer_size)
            throw std::runtime_error("Buffer overflow (inner map size)");
        std::memcpy(ptr, &inner_size, sizeof(size_t));
        ptr += sizeof(size_t);

        // For each inner map element:
        for (const auto& inner : outer.second) {
            // Write inner key.
            size_t inner_key_len = inner.first.size();
            if (ptr + sizeof(size_t) + inner_key_len > buffer + buffer_size)
                throw std::runtime_error("Buffer overflow (inner key)");
            std::memcpy(ptr, &inner_key_len, sizeof(size_t));
            ptr += sizeof(size_t);
            std::memcpy(ptr, inner.first.data(), inner_key_len);
            ptr += inner_key_len;

            // Write tensor_meta.offset and tensor_meta.byte_size.
            if (ptr + sizeof(int64_t) * 2 > buffer + buffer_size)
                throw std::runtime_error(
                    "Buffer overflow (tensor_meta POD fields)");
            std::memcpy(ptr, &inner.second.offset, sizeof(int64_t));
            ptr += sizeof(int64_t);
            std::memcpy(ptr, &inner.second.byte_size, sizeof(int64_t));
            ptr += sizeof(int64_t);

            // Write tensor_meta.tensor_shape: first the number of elements...
            size_t vec_size = inner.second.tensor_shape.size();
            if (ptr + sizeof(size_t) > buffer + buffer_size)
                throw std::runtime_error("Buffer overflow (tensor_shape size)");
            std::memcpy(ptr, &vec_size, sizeof(size_t));
            ptr += sizeof(size_t);

            // ... then the raw vector data.
            if (vec_size > 0) {
                if (ptr + vec_size * sizeof(int64_t) > buffer + buffer_size)
                    throw std::runtime_error(
                        "Buffer overflow (tensor_shape data)");
                std::memcpy(ptr, inner.second.tensor_shape.data(),
                            vec_size * sizeof(int64_t));
                ptr += vec_size * sizeof(int64_t);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Simple Deserialization: Read the map from a buffer.
// Throws std::runtime_error if the buffer content is invalid.
// -----------------------------------------------------------------------------
std::unordered_map<std::string, std::unordered_map<std::string, tensor_meta>>
deserialize_map_from_buffer(const char* buffer, size_t buffer_size) {
    std::unordered_map<std::string,
                       std::unordered_map<std::string, tensor_meta>>
        result;
    const char* ptr = buffer;
    const char* end = buffer + buffer_size;

    // Read outer map size.
    if (ptr + sizeof(size_t) > end)
        throw std::runtime_error("Buffer overflow (reading outer_size)");
    size_t outer_size;
    std::memcpy(&outer_size, ptr, sizeof(size_t));
    ptr += sizeof(size_t);

    for (size_t i = 0; i < outer_size; ++i) {
        // Read outer key.
        if (ptr + sizeof(size_t) > end)
            throw std::runtime_error(
                "Buffer overflow (reading outer key length)");
        size_t key_len;
        std::memcpy(&key_len, ptr, sizeof(size_t));
        ptr += sizeof(size_t);
        if (ptr + key_len > end)
            throw std::runtime_error(
                "Buffer overflow (reading outer key data)");
        std::string outer_key(ptr, key_len);
        ptr += key_len;

        // Read inner map size.
        if (ptr + sizeof(size_t) > end)
            throw std::runtime_error(
                "Buffer overflow (reading inner map size)");
        size_t inner_size;
        std::memcpy(&inner_size, ptr, sizeof(size_t));
        ptr += sizeof(size_t);

        std::unordered_map<std::string, tensor_meta> inner_map;
        for (size_t j = 0; j < inner_size; ++j) {
            // Read inner key.
            if (ptr + sizeof(size_t) > end)
                throw std::runtime_error(
                    "Buffer overflow (reading inner key length)");
            size_t inner_key_len;
            std::memcpy(&inner_key_len, ptr, sizeof(size_t));
            ptr += sizeof(size_t);
            if (ptr + inner_key_len > end)
                throw std::runtime_error(
                    "Buffer overflow (reading inner key data)");
            std::string inner_key(ptr, inner_key_len);
            ptr += inner_key_len;

            // Read tensor_meta.offset and tensor_meta.byte_size.
            if (ptr + sizeof(int64_t) * 2 > end)
                throw std::runtime_error(
                    "Buffer overflow (reading tensor_meta POD fields)");
            int64_t offset, byte_size;
            std::memcpy(&offset, ptr, sizeof(int64_t));
            ptr += sizeof(int64_t);
            std::memcpy(&byte_size, ptr, sizeof(int64_t));
            ptr += sizeof(int64_t);

            // Read tensor_meta.tensor_shape.
            if (ptr + sizeof(size_t) > end)
                throw std::runtime_error(
                    "Buffer overflow (reading tensor_shape size)");
            size_t vec_size;
            std::memcpy(&vec_size, ptr, sizeof(size_t));
            ptr += sizeof(size_t);
            std::vector<int64_t> shape;
            if (vec_size > 0) {
                if (ptr + vec_size * sizeof(int64_t) > end)
                    throw std::runtime_error(
                        "Buffer overflow (reading tensor_shape data)");
                shape.resize(vec_size);
                std::memcpy(shape.data(), ptr, vec_size * sizeof(int64_t));
                ptr += vec_size * sizeof(int64_t);
            }

            tensor_meta meta;
            meta.offset = offset;
            meta.byte_size = byte_size;
            meta.tensor_shape = shape;
            inner_map[inner_key] = meta;
        }
        result[outer_key] = inner_map;
    }
    return result;
}

// -----------------------------------------------------------------------------
// API: Serialize the map into shared memory.
// This function computes the required size, creates (or truncates) the shared
// memory object, maps it, writes the serialized data, then unmaps/closes it.
// -----------------------------------------------------------------------------
void serialize_to_shared_memory(
    const std::unordered_map<std::string,
                             std::unordered_map<std::string, tensor_meta>>& map,
    const std::string& shm_name) {
    // Compute the buffer size required.
    size_t total_size = compute_serialized_size(map);

    // Open (or create) the shared memory region.
    int fd = shm_open(shm_name.c_str(), O_RDWR | O_CREAT, 0666);
    if (fd == -1)
        throw std::runtime_error("Failed to create shared memory: " + shm_name);

    // Set the size.
    if (ftruncate(fd, total_size) == -1) {
        close(fd);
        shm_unlink(shm_name.c_str());
        throw std::runtime_error("Failed to set shared memory size");
    }

    // Map the memory.
    void* addr =
        mmap(nullptr, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        close(fd);
        shm_unlink(shm_name.c_str());
        throw std::runtime_error("Failed to map shared memory");
    }

    // Write the data.
    serialize_map_to_buffer(map, static_cast<char*>(addr), total_size);

    // Clean up.
    munmap(addr, total_size);
    close(fd);
}

// -----------------------------------------------------------------------------
// API: Deserialize the map from shared memory.
// This function opens the shared memory region, maps it, reads the data, then
// unmaps/closes it.
// -----------------------------------------------------------------------------
std::unordered_map<std::string, std::unordered_map<std::string, tensor_meta>>
deserialize_from_shared_memory(const std::string& shm_name) {
    // Open shared memory (read-only).
    int fd = shm_open(shm_name.c_str(), O_RDONLY, 0666);
    if (fd == -1)
        throw std::runtime_error("Cannot open shared memory: " + shm_name);

    // Get the size.
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        throw std::runtime_error("Failed to get shared memory size");
    }
    size_t size = sb.st_size;

    // Map the memory.
    void* addr = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("Failed to map shared memory");
    }

    // Deserialize the map.
    auto result =
        deserialize_map_from_buffer(static_cast<const char*>(addr), size);

    // Clean up.
    munmap(addr, size);
    close(fd);
    return result;
}
