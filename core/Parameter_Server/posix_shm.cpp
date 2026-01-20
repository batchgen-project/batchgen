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
#include <filesystem>

#include <cuda_runtime_api.h>
#include "../utils.h"
#include "posix_shm.h"
#include "spdlog/spdlog.h"
#include <signal.h>
#include <setjmp.h>
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#endif
namespace fs = std::filesystem; 

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

// Helper function to perform aligned mmap
// This ensures the virtual address is aligned to the specified alignment (e.g., 2MB for huge pages)
// which is often required for cudaHostRegister to work correctly with huge pages.
void* mmap_aligned(size_t length, int prot, int flags, int fd, off_t offset, size_t alignment) {
    // Allocate extra space to ensure we can find an aligned segment
    size_t total_len = length + alignment;
    
    // Reserve address space using anonymous mapping
    void* addr = mmap(nullptr, total_len, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (addr == MAP_FAILED) {
        return MAP_FAILED;
    }

    uintptr_t raw_addr = reinterpret_cast<uintptr_t>(addr);
    uintptr_t aligned_addr = (raw_addr + alignment - 1) & ~(alignment - 1);
    void* final_addr = reinterpret_cast<void*>(aligned_addr);

    // Map the file into the aligned position using MAP_FIXED
    // This replaces the anonymous mapping at that location
    void* ret = mmap(final_addr, length, prot, flags | MAP_FIXED, fd, offset);
    
    // Unmap the unused parts of the reservation
    size_t prefix_len = aligned_addr - raw_addr;
    if (prefix_len > 0) {
        munmap(addr, prefix_len);
    }
    
    size_t suffix_len = total_len - length - prefix_len;
    if (suffix_len > 0) {
        munmap(reinterpret_cast<void*>(aligned_addr + length), suffix_len);
    }

    return ret;
}

// --- End of Mocked Dependencies ---


/**
 * @brief Allocates shared memory optionally pinned for CUDA operations.
 * * This function supports two allocation strategies:
 * 1.  Hugetlbfs: Uses large 2MB pages for potentially better performance, but requires
 * root privileges and proper system configuration. Controlled by enable_hugetlbfs.
 * 2.  Regular Shared Memory: A fallback using standard 4KB pages via shm_open.
 * * In both cases, the allocated memory is "touched" to ensure it is resident in RAM.
 * If pin_for_cuda is true, memory is also registered with cudaHostRegister for DMA.
 * * @param shm_name The name for the shared memory segment.
 * @param size The desired size of the allocation in bytes.
 * @param create True if the caller is the server (creates the segment), false for workers.
 * @param enable_hugetlbfs If true, attempts to use hugetlbfs for allocation.
 * @param pin_for_cuda If true, register memory with cudaHostRegister for GPU DMA access.
 *                     Server should pass false (no GPU access needed), workers pass true.
 * @return A void pointer to the allocated shared memory.
 * @throws std::runtime_error on failure.
 */
void* allocate_shared_pinned_memory(const std::string& shm_name,
                                    int64_t size,
                                    bool create,
                                    bool enable_hugetlbfs,
                                    bool pin_for_cuda) {
    if (size <= 0) {
        throw std::runtime_error("Invalid allocation size: " + std::to_string(size));
    }

    const size_t page_size = sysconf(_SC_PAGESIZE);
    const size_t huge_page_size = 2 * 1024 * 1024; // 2MB

    logger->info("Allocating shared memory: name={}, size={}MB, mode={}",
                 shm_name, size / (1024 * 1024),
                 create ? "server" : "worker");

    void* ptr = nullptr;
    bool using_huge_pages = false;
    int64_t allocated_size = 0;
    // std::string hugepage_path = "/dev/hugepages/" + shm_name;
    std::string hugepage_path = "/dev/hugepages/" + 
        (shm_name[0] == '/' ? shm_name.substr(1) : shm_name);
        
    // STAGE 1: Attempt allocation using hugetlbfs if enabled
    if (enable_hugetlbfs) {
        logger->info("Attempting hugepage allocation...");
        int flags = O_RDWR | (create ? O_CREAT : 0);
        int fd = open(hugepage_path.c_str(), flags, 0666);

        if (fd >= 0) {
            if (create) {
                int64_t huge_aligned_size = ((size + huge_page_size - 1) / huge_page_size) * huge_page_size;
                if (ftruncate64(fd, huge_aligned_size) == 0) {
                    // Use mmap_aligned to ensure 2MB alignment (or system page size if larger)
                    size_t alignment = std::max(huge_page_size, page_size);
                    ptr = mmap_aligned(huge_aligned_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0, alignment);
                    if (ptr != MAP_FAILED) {
                        allocated_size = huge_aligned_size;
                        if (touch_pages(ptr, size, huge_page_size, true)) {
                            using_huge_pages = true;
                            logger->info("Hugepage multi-threaded initialization succeeded");
                        } else {
                            logger->error("Multi-threaded hugepage touch failed. Aborting hugepage allocation.");
                            munmap(ptr, allocated_size);
                            ptr = nullptr; // Signal failure to fallback
                        }
                    } else {
                        logger->warn("hugetlbfs mmap failed: {}", strerror(errno));
                        ptr = nullptr; // Ensure ptr is null for fallback
                    }
                } else {
                    logger->warn("hugetlbfs ftruncate failed: {}", strerror(errno));
                }
            } else { // Worker logic: wait for file to be sized by server
                struct stat sb;
                int64_t file_size = 0;
                for (int retry = 0; retry < 20; ++retry) { // Retry for up to 2 seconds
                    if (fstat(fd, &sb) == -1) {
                        close(fd);
                        throw std::runtime_error("fstat on hugepage file failed: " + std::string(strerror(errno)));
                    }
                    if (sb.st_size > 0) {
                        file_size = sb.st_size;
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }

                if (file_size > 0) {
                    // Use mmap_aligned for workers too
                    size_t alignment = std::max(huge_page_size, page_size);
                    ptr = mmap_aligned(file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0, alignment);
                    if (ptr != MAP_FAILED) {
                        allocated_size = file_size;
                        using_huge_pages = true;
                    } else {
                        logger->warn("hugetlbfs mmap failed for worker: {}", strerror(errno));
                        ptr = nullptr;
                    }
                } else {
                    logger->error("Timed out waiting for server to create hugepage file.");
                }
            }
            close(fd);

            if (!using_huge_pages && create) {
                logger->info("Cleaning up failed hugepage allocation at '{}'", hugepage_path);
                unlink(hugepage_path.c_str());
            }
        } else {
            logger->warn("Could not open hugetlbfs path '{}': {}. Check permissions and mount.", hugepage_path, strerror(errno));
        }
    }

    // STAGE 2: Fallback to regular shared memory if hugepages were disabled or failed
    if (!ptr) {
        if (enable_hugetlbfs) {
             logger->info("Falling back to regular shared memory...");
        }

        int flags = O_RDWR | (create ? O_CREAT : 0);
        int fd = shm_open(shm_name.c_str(), flags, 0666);
        if (fd < 0) {
            throw std::runtime_error("shm_open failed for '" + shm_name + "': " + strerror(errno));
        }

        if (create) {
            int64_t aligned_size = ((size + page_size - 1) / page_size) * page_size;
            if (ftruncate64(fd, aligned_size) == -1) {
                close(fd);
                shm_unlink(shm_name.c_str());
                throw std::runtime_error("ftruncate failed: " + std::string(strerror(errno)));
            }
            // Use mmap_aligned with 2MB alignment (or system page size) even for regular shm
            size_t alignment = std::max(huge_page_size, page_size);
            ptr = mmap_aligned(aligned_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0, alignment);
            allocated_size = aligned_size;
        } else { // Worker logic
            struct stat sb;
            int64_t file_size = 0;
            for (int retry = 0; retry < 20; ++retry) {
                if (fstat(fd, &sb) == -1) {
                    close(fd);
                    throw std::runtime_error("fstat on shm file failed: " + std::string(strerror(errno)));
                }
                if (sb.st_size > 0) {
                    file_size = sb.st_size;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if (file_size > 0) {
                // Use mmap_aligned for workers too
                size_t alignment = std::max(huge_page_size, page_size);
                ptr = mmap_aligned(file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0, alignment);
                allocated_size = file_size;
            } else {
                 logger->error("Timed out waiting for server to create shm file.");
            }
        }
        close(fd);

        if (ptr == MAP_FAILED) {
            if (create) shm_unlink(shm_name.c_str());
            throw std::runtime_error("mmap for regular shm failed: " + std::string(strerror(errno)));
        }
        
        logger->info("Allocated {:.3f}GB using regular pages",
                   allocated_size / (1024.0 * 1024.0 * 1024.0));

        if (madvise(ptr, allocated_size, MADV_HUGEPAGE) == 0) {
            logger->debug("Successfully enabled transparent huge pages hint.");
        }

        if (create) {
            if (!touch_pages(ptr, size, page_size, true)) {
                logger->error("Multi-threaded regular page touch failed - memory may not be fully resident.");
            }
        }
    }

    // STAGE 3: Register the successfully allocated memory with CUDA (only if pin_for_cuda is true)
    // Server process skips this to avoid GPU memory usage from page tables
    if (pin_for_cuda) {
        try {
            logger->info("Registering {:.3f}GB with CUDA...", size / (1024.0 * 1024.0 * 1024.0));
            auto cuda_start = std::chrono::high_resolution_clock::now();

            cudaError_t err = cudaHostRegister(ptr, size, cudaHostRegisterDefault);

            auto cuda_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - cuda_start);

            if (err != cudaSuccess) {
                throw std::runtime_error("cudaHostRegister failed: " + std::string(cudaGetErrorString(err)));
            }

            logger->info("CUDA registration completed in {:.2f}s", cuda_duration.count() / 1000.0);
        } catch (const std::exception& e) {
            // Clean up memory and rethrow if CUDA registration fails
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
    } else {
        logger->info("Skipping CUDA registration (server mode, no GPU access needed)");
    }

    logger->info("Memory allocation completed successfully using {} pages.",
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
            total_size += sizeof(size_t) +
                          inner.second.dtype.size();  // tensor_meta.dtype (length + bytes)
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

            // Write tensor_meta.dtype: first its length, then its characters.
            size_t dtype_len = inner.second.dtype.size();
            if (ptr + sizeof(size_t) + dtype_len > buffer + buffer_size)
                throw std::runtime_error("Buffer overflow (tensor_meta dtype)");
            std::memcpy(ptr, &dtype_len, sizeof(size_t));
            ptr += sizeof(size_t);
            if (dtype_len > 0) {
                std::memcpy(ptr, inner.second.dtype.data(), dtype_len);
                ptr += dtype_len;
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

            // Read tensor_meta.dtype.
            if (ptr + sizeof(size_t) > end)
                throw std::runtime_error(
                    "Buffer overflow (reading tensor_meta dtype length)");
            size_t dtype_len;
            std::memcpy(&dtype_len, ptr, sizeof(size_t));
            ptr += sizeof(size_t);
            std::string dtype;
            if (dtype_len > 0) {
                if (ptr + dtype_len > end)
                    throw std::runtime_error(
                        "Buffer overflow (reading tensor_meta dtype data)");
                dtype = std::string(ptr, dtype_len);
                ptr += dtype_len;
            }

            tensor_meta meta;
            meta.offset = offset;
            meta.byte_size = byte_size;
            meta.tensor_shape = shape;
            meta.dtype = dtype;
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