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
#include <signal.h>
#include <setjmp.h>
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

// Global for signal handling during page touch
thread_local sigjmp_buf g_page_touch_jmpbuf;
thread_local volatile bool g_in_page_touch = false;

void segv_handler(int sig, siginfo_t* info, void* context) {
    if (g_in_page_touch) {
        siglongjmp(g_page_touch_jmpbuf, 1);
    }
    // Otherwise, let the default handler deal with it
    struct sigaction sa;
    sa.sa_handler = SIG_DFL;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(sig, &sa, nullptr);
    raise(sig);
}

// Helper function to execute system command
int execute_command(const std::string& cmd) {
    int ret = system(cmd.c_str());
    if (ret == -1) {
        logger->error("Failed to execute command: {}", cmd);
        return -1;
    }
    return WEXITSTATUS(ret);
}

// Helper function for page touching with signal handling
bool touch_pages(void* ptr, int64_t size, long page_size, bool multi_threaded) {
    // Set up signal handlers
    struct sigaction sa;
    sa.sa_sigaction = segv_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_SIGINFO;
    struct sigaction old_sa_segv, old_sa_bus;
    sigaction(SIGSEGV, &sa, &old_sa_segv);
    sigaction(SIGBUS, &sa, &old_sa_bus);
    
    bool success = false;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (multi_threaded) {
        logger->info("Attempting multi-threaded memory initialization...");
        const int num_threads = std::min(16, (int)std::thread::hardware_concurrency());
        const int64_t chunk_size = size / num_threads;
        
        std::vector<std::thread> threads;
        std::atomic<int> failed_threads(0);
        
        for (int i = 0; i < num_threads; i++) {
            threads.emplace_back([=, &failed_threads]() {
                int64_t start_offset = i * chunk_size;
                int64_t end_offset = (i == num_threads - 1) ? size : start_offset + chunk_size;
                volatile char* p = reinterpret_cast<volatile char*>(ptr);
                
                g_in_page_touch = true;
                if (sigsetjmp(g_page_touch_jmpbuf, 1) == 0) {
                    for (int64_t offset = start_offset; offset < end_offset; offset += page_size) {
                        p[offset] = 0;
                    }
                } else {
                    logger->warn("Thread {} caught signal during page touch", i);
                    failed_threads++;
                }
                g_in_page_touch = false;
            });
        }
        
        for (auto& t : threads) {
            t.join();
        }
        
        success = (failed_threads == 0);
        if (!success) {
            logger->warn("{} threads failed during initialization", failed_threads.load());
        }
    } else {
        logger->info("Attempting single-threaded memory initialization...");
        volatile char* p = reinterpret_cast<volatile char*>(ptr);
        
        g_in_page_touch = true;
        if (sigsetjmp(g_page_touch_jmpbuf, 1) == 0) {
            for (int64_t offset = 0; offset < size; offset += page_size) {
                p[offset] = 0;
            }
            success = true;
        } else {
            logger->error("Single-threaded initialization failed with signal");
        }
        g_in_page_touch = false;
    }
    
    // Restore original signal handlers
    sigaction(SIGSEGV, &old_sa_segv, nullptr);
    sigaction(SIGBUS, &old_sa_bus, nullptr);
    
    if (success) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time);
        logger->info("{} initialization completed in {:.2f}s", 
                   multi_threaded ? "Multi-threaded" : "Single-threaded",
                   duration.count() / 1000.0);
    }
    
    return success;
}

void* allocate_shared_pinned_memory(const std::string& shm_name,
                                    int64_t size,
                                    bool create) {
    if (size <= 0) {
        throw std::runtime_error("Invalid size: " + std::to_string(size));
    }
    
    const size_t page_size = sysconf(_SC_PAGESIZE);
    const size_t huge_page_size = 2 * 1024 * 1024;  // 2MB
    
    logger->info("Allocating shared memory: name={}, size={}MB, mode={}", 
                shm_name, size / (1024*1024), 
                create ? "server" : "worker");
    
    void* ptr = nullptr;
    bool using_huge_pages = false;
    int64_t allocated_size = 0;
    std::string hugepage_path = "/dev/hugepages/" + shm_name;
    
    // STAGE 1: Try hugetlbfs allocation (DEFAULT CHOICE)
    logger->info("Attempting hugepage allocation...");
    int flags = O_RDWR | (create ? O_CREAT : 0);
    int fd = open(hugepage_path.c_str(), flags, 0666);
    
    if (fd >= 0) {
        // Align to huge page size
        int64_t huge_aligned_size = ((size + huge_page_size - 1) / huge_page_size) * huge_page_size;
        
        bool huge_alloc_success = false;
        if (create) {
            if (ftruncate64(fd, huge_aligned_size) == 0) {
                huge_alloc_success = true;
            } else {
                logger->warn("hugetlbfs ftruncate failed: {}", strerror(errno));
            }
        } else {
            huge_alloc_success = true;  // For workers, just try mmap
        }
        
        if (huge_alloc_success) {
            ptr = mmap(nullptr, huge_aligned_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (ptr != MAP_FAILED) {
                allocated_size = huge_aligned_size;
                logger->info("Successfully mapped {}GB using hugetlbfs", 
                           allocated_size / (1024.0*1024.0*1024.0));
                
                // For create mode, try page touching
                if (create) {
                    // Try multi-threaded page touch first
                    if (touch_pages(ptr, size, huge_page_size, true)) {
                        using_huge_pages = true;
                        logger->info("Hugepage multi-threaded initialization succeeded");
                    } else {
                        // Fallback to single-threaded page touch
                        logger->warn("Multi-threaded hugepage touch failed, trying single-threaded...");
                        if (touch_pages(ptr, size, huge_page_size, false)) {
                            using_huge_pages = true;
                            logger->info("Hugepage single-threaded initialization succeeded");
                        } else {
                            // Both failed - give up hugepages
                            logger->error("Both multi and single threaded hugepage touch failed");
                            munmap(ptr, allocated_size);
                            ptr = nullptr;
                        }
                    }
                } else {
                    // For workers, just accept the mapping
                    using_huge_pages = true;
                }
            } else {
                logger->warn("hugetlbfs mmap failed: {}", strerror(errno));
                ptr = nullptr;
            }
        }
        
        close(fd);
        
        // If hugepage approach failed and we're in create mode, clean up
        if (!using_huge_pages && create) {
            logger->info("Cleaning up failed hugepage allocation...");
            unlink(hugepage_path.c_str());
            
            // Reset hugepages to 0
            logger->info("Resetting hugepages to 0...");
            int ret = execute_command("sysctl -w vm.nr_hugepages=0");
            if (ret == 0) {
                logger->info("Successfully reset hugepages");
            } else {
                logger->warn("Failed to reset hugepages, user may need to manually clean up");
            }
        }
    } else {
        logger->debug("Could not open hugetlbfs: {}", strerror(errno));
    }
    
    // STAGE 2: Fallback to regular shared memory with single-threaded touch
    if (!ptr) {
        logger->info("Falling back to regular shared memory...");
        
        // Align to regular page size
        int64_t aligned_size = ((size + page_size - 1) / page_size) * page_size;
        
        int flags = O_RDWR | (create ? O_CREAT : 0);
        int fd = shm_open(shm_name.c_str(), flags, 0666);
        if (fd < 0) {
            throw std::runtime_error("shm_open failed: " + std::string(strerror(errno)));
        }

        if (create && ftruncate64(fd, aligned_size) == -1) {
            close(fd);
            shm_unlink(shm_name.c_str());
            throw std::runtime_error("ftruncate failed: " + std::string(strerror(errno)));
        }

        ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        
        if (ptr == MAP_FAILED) {
            if (create) shm_unlink(shm_name.c_str());
            throw std::runtime_error("mmap failed: " + std::string(strerror(errno)));
        }
        
        allocated_size = aligned_size;
        logger->info("Allocated {}GB using regular pages", 
                   allocated_size / (1024.0*1024.0*1024.0));
        
        // Try to enable transparent huge pages
        if (madvise(ptr, allocated_size, MADV_HUGEPAGE) == 0) {
            logger->debug("Enabled transparent huge pages hint");
        }
        
        // For regular pages, only use single-threaded touch
        if (create) {
            if (!touch_pages(ptr, size, page_size, false)) {
                logger->error("Regular page touch failed - memory may not be fully resident");
                // Continue anyway - the allocation succeeded even if touch failed
            }
        }
    }
    
    // STAGE 3: Register with CUDA
    try {
        logger->info("Registering {}GB with CUDA...", size / (1024.0*1024.0*1024.0));
        auto cuda_start = std::chrono::high_resolution_clock::now();
        
        cudaError_t err = cudaHostRegister(ptr, size, cudaHostRegisterDefault);
        
        auto cuda_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - cuda_start);
        
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaHostRegister failed: " + 
                                    std::string(cudaGetErrorString(err)));
        }
        
        logger->info("CUDA registration completed in {:.2f}s", cuda_duration.count() / 1000.0);
    } catch (const std::exception& e) {
        // Clean up and rethrow
        munmap(ptr, allocated_size);
        if (create) {
            if (using_huge_pages) {
                unlink(hugepage_path.c_str());
            } else {
                shm_unlink(shm_name.c_str());
            }
        }
        throw;
    }
    
    logger->info("Memory allocation completed successfully using {} pages",
               using_huge_pages ? "huge" : "regular");
    
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