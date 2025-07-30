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

void* allocate_shared_pinned_memory(const std::string& shm_name,
                                    int64_t size,
                                    bool create) {
    if (size <= 0) {
        throw std::runtime_error("Invalid size: " + std::to_string(size));
    }
    
    const size_t page_size = sysconf(_SC_PAGESIZE);
    const size_t huge_page_size = 2 * 1024 * 1024;  // 2MB
    
    // Initially align to regular page size
    int64_t aligned_size = ((size + page_size - 1) / page_size) * page_size;
    
    logger->info("Allocating shared memory: name={}, size={}MB, mode={}", 
                shm_name, size / (1024*1024), 
                create ? "server" : "worker");
    
    void* ptr = nullptr;
    bool using_huge_pages = false;
    bool hugepage_fallback = false;
    
    // Try hugetlbfs first
    std::string hugepage_path = "/dev/hugepages/" + shm_name;
    int flags = O_RDWR | (create ? O_CREAT : 0);
    int fd = open(hugepage_path.c_str(), flags, 0666);
    
    if (fd >= 0) {
        // For huge pages, align to huge page size
        int64_t huge_aligned_size = ((size + huge_page_size - 1) / huge_page_size) * huge_page_size;
        
        if (create && ftruncate64(fd, huge_aligned_size) == -1) {
            logger->debug("hugetlbfs ftruncate failed: {}", strerror(errno));
            close(fd);
            unlink(hugepage_path.c_str());
            hugepage_fallback = true;
        } else {
            ptr = mmap(nullptr, huge_aligned_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            close(fd);
            
            if (ptr != MAP_FAILED) {
                using_huge_pages = true;
                aligned_size = huge_aligned_size;  // Update aligned_size for later use
                logger->info("Allocated {}GB using hugetlbfs (2MB pages)", 
                           aligned_size / (1024.0*1024.0*1024.0));
                
                // Verify huge page allocation actually worked
                if (create) {
                    logger->info("Verifying huge page allocation...");
                    bool hugepage_verified = false;
                    
                    // Method 1: Check /proc/self/smaps_rollup for our specific mapping
                    std::ifstream smaps("/proc/self/smaps");
                    std::string line;
                    uintptr_t ptr_addr = reinterpret_cast<uintptr_t>(ptr);
                    bool in_our_mapping = false;
                    
                    while (std::getline(smaps, line)) {
                        // Look for memory mapping lines (contain address ranges)
                        if (line.find("-") != std::string::npos && line.find(" ") != std::string::npos) {
                            std::istringstream iss(line);
                            std::string addr_range;
                            iss >> addr_range;
                            
                            size_t dash_pos = addr_range.find('-');
                            if (dash_pos != std::string::npos) {
                                try {
                                    uintptr_t start_addr = std::stoull(addr_range.substr(0, dash_pos), nullptr, 16);
                                    uintptr_t end_addr = std::stoull(addr_range.substr(dash_pos + 1), nullptr, 16);
                                    
                                    // Check if this is our memory mapping
                                    in_our_mapping = (ptr_addr >= start_addr && ptr_addr < end_addr);
                                    if (in_our_mapping) {
                                        logger->debug("Found our memory mapping: {}", line);
                                    }
                                } catch (...) {
                                    in_our_mapping = false;
                                }
                            }
                        }
                        // If we're in our mapping, look for huge page indicators
                        else if (in_our_mapping) {
                            if (line.find("KernelPageSize:") != std::string::npos) {
                                logger->debug("Kernel page size: {}", line);
                                // For hugetlbfs, KernelPageSize should be 2048 kB
                                if (line.find("2048 kB") != std::string::npos) {
                                    hugepage_verified = true;
                                    logger->info("Verified hugetlbfs allocation: {}", line);
                                }
                            }
                            else if (line.find("MMUPageSize:") != std::string::npos) {
                                logger->debug("MMU page size: {}", line);
                            }
                            // Reset when we exit our mapping
                            else if (line.find("-") != std::string::npos) {
                                if (hugepage_verified) break; // Found what we need
                                in_our_mapping = false;
                            }
                        }
                    }
                    
                    // Method 2: Check system huge page counters
                    if (!hugepage_verified) {
                        std::ifstream meminfo("/proc/meminfo");
                        std::string memline;
                        while (std::getline(meminfo, memline)) {
                            if (memline.find("HugePages_Free:") != std::string::npos) {
                                logger->debug("System huge page info: {}", memline);
                                break;
                            }
                        }
                    }
                    
                    // Method 3: Simple memory access test
                    volatile char* test_ptr = reinterpret_cast<volatile char*>(ptr);
                    bool access_test_passed = true;
                    
                    try {
                        // Test access at key points
                        test_ptr[0] = 1;                           // First byte
                        if (aligned_size > huge_page_size) {
                            test_ptr[huge_page_size] = 1;          // Second huge page
                        }
                        test_ptr[aligned_size - 1] = 1;           // Last byte (use aligned_size)
                        logger->debug("Basic huge page memory access test passed");
                    } catch (...) {
                        logger->warn("Huge page memory access test failed");
                        access_test_passed = false;
                    }
                    
                    // If verification fails, fall back to regular pages
                    if (!hugepage_verified || !access_test_passed) {
                        logger->warn("Huge page allocation verification failed (verified={}, access={}), falling back to regular pages", 
                                   hugepage_verified, access_test_passed);
                        munmap(ptr, aligned_size);
                        unlink(hugepage_path.c_str());
                        using_huge_pages = false;
                        hugepage_fallback = true;
                        ptr = nullptr;
                    } else {
                        logger->info("Huge page allocation successfully verified");
                    }
                }
            } else {
                logger->debug("hugetlbfs mmap failed: {}", strerror(errno));
                if (create) unlink(hugepage_path.c_str());
                hugepage_fallback = true;
            }
        }
    } else {
        logger->debug("hugetlbfs open failed: {}", strerror(errno));
        hugepage_fallback = true;
    }
    
    // Fallback to shm_open with regular pages
    if (!using_huge_pages) {
        if (hugepage_fallback) {
            logger->info("Falling back to regular pages due to huge page allocation failure");
        }
        
        // Reset aligned_size to regular page alignment
        aligned_size = ((size + page_size - 1) / page_size) * page_size;
        
        fd = shm_open(shm_name.c_str(), flags, 0666);
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
        
        logger->info("Allocated {}GB using regular pages", 
                   aligned_size / (1024.0*1024.0*1024.0));
        
        // Hint for transparent huge pages when creating
        if (create && madvise(ptr, aligned_size, MADV_HUGEPAGE) == 0) {
            logger->debug("Enabled transparent huge pages hint");
        }
    }
    
    // Server-side initialization
    if (create) {
        // Configure NUMA if available
        if (numa_available() >= 0) {
            int num_nodes = numa_num_configured_nodes();
            if (num_nodes >= 2) {
                struct bitmask* nodemask = numa_allocate_nodemask();
                if (nodemask) {
                    numa_bitmask_clearall(nodemask);
                    numa_bitmask_setbit(nodemask, 0);
                    numa_bitmask_setbit(nodemask, 1);
                    
                    if (set_mempolicy(MPOL_INTERLEAVE, nodemask->maskp, nodemask->size + 1) == 0) {
                        logger->info("NUMA memory interleaving enabled across nodes 0-1");
                    }
                    numa_free_nodemask(nodemask);
                }
            }
        }
        
        // Touch pages to ensure allocation
        logger->info("Initializing memory pages...");
        auto start_time = std::chrono::high_resolution_clock::now();
        
        const long touch_page_size = using_huge_pages ? (2 * 1024 * 1024) : page_size;
        const int num_threads = std::min(16, (int)std::thread::hardware_concurrency());
        const int64_t chunk_size = aligned_size / num_threads;  // Use aligned_size for safety
        
        logger->debug("Memory touching: aligned_size={}, touch_page_size={}, num_threads={}, chunk_size={}", 
                     aligned_size, touch_page_size, num_threads, chunk_size);
        
        std::vector<std::thread> threads;
        std::atomic<bool> touch_error{false};
        
        for (int i = 0; i < num_threads; i++) {
            threads.emplace_back([=, &touch_error]() {
                try {
                    int64_t start_offset = i * chunk_size;
                    int64_t end_offset = (i == num_threads - 1) ? aligned_size : start_offset + chunk_size;
                    volatile char* p = reinterpret_cast<volatile char*>(ptr);
                    
                    logger->debug("Thread {} touching memory from {} to {} (step={})", 
                                 i, start_offset, end_offset, touch_page_size);
                    
                    for (int64_t offset = start_offset; offset < end_offset; offset += touch_page_size) {
                        if (offset >= aligned_size) {
                            logger->warn("Thread {} hit boundary: offset={}, aligned_size={}", i, offset, aligned_size);
                            break;
                        }
                        p[offset] = 0;
                    }
                } catch (const std::exception& e) {
                    logger->error("Thread {} memory touching failed: {}", i, e.what());
                    touch_error = true;
                } catch (...) {
                    logger->error("Thread {} memory touching failed with unknown error", i);
                    touch_error = true;
                }
            });
        }
        
        for (auto& t : threads) {
            t.join();
        }
        
        if (touch_error) {
            throw std::runtime_error("Memory page touching failed");
        }
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time);
        logger->info("Memory initialization completed in {:.2f}s", duration.count() / 1000.0);
    }
    
    // Register with CUDA
    logger->info("Registering {}GB with CUDA...", size / (1024.0*1024.0*1024.0));
    auto cuda_start = std::chrono::high_resolution_clock::now();
    
    cudaError_t err = cudaHostRegister(ptr, size, cudaHostRegisterDefault);
    
    auto cuda_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - cuda_start);
    
    if (err != cudaSuccess) {
        munmap(ptr, aligned_size);
        if (create) {
            if (using_huge_pages) {
                unlink(hugepage_path.c_str());
            } else {
                shm_unlink(shm_name.c_str());
            }
        }
        throw std::runtime_error("cudaHostRegister failed: " + 
                                std::string(cudaGetErrorString(err)));
    }
    
    logger->info("CUDA registration completed in {:.2f}s", cuda_duration.count() / 1000.0);
    
    return ptr;
}

// void* allocate_shared_pinned_memory(const std::string& shm_name,
//                                     int64_t size,
//                                     bool create) {
//     if (size <= 0) {
//         throw std::runtime_error("Invalid size: " + std::to_string(size));
//     }
    
//     const size_t page_size = sysconf(_SC_PAGESIZE);
//     const size_t huge_page_size = 2 * 1024 * 1024;  // 2MB
    
//     // Initially align to regular page size
//     int64_t aligned_size = ((size + page_size - 1) / page_size) * page_size;
    
//     logger->info("Allocating shared memory: name={}, size={}MB, mode={}", 
//                 shm_name, size / (1024*1024), 
//                 create ? "server" : "worker");
    
//     void* ptr = nullptr;
//     bool using_huge_pages = false;
    
//     // Try hugetlbfs first
//     std::string hugepage_path = "/dev/hugepages/" + shm_name;
//     int flags = O_RDWR | (create ? O_CREAT : 0);
//     int fd = open(hugepage_path.c_str(), flags, 0666);
    
//     if (fd >= 0) {
//         // For huge pages, align to huge page size
//         int64_t huge_aligned_size = ((size + huge_page_size - 1) / huge_page_size) * huge_page_size;
        
//         if (create && ftruncate64(fd, huge_aligned_size) == -1) {
//             logger->debug("hugetlbfs ftruncate failed: {}", strerror(errno));
//             close(fd);
//             unlink(hugepage_path.c_str());
//         } else {
//             ptr = mmap(nullptr, huge_aligned_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
//             close(fd);
            
//             if (ptr != MAP_FAILED) {
//                 using_huge_pages = true;
//                 aligned_size = huge_aligned_size;  // Update aligned_size for later use
//                 logger->info("Allocated {}GB using hugetlbfs (2MB pages)", 
//                            aligned_size / (1024.0*1024.0*1024.0));
//             } else if (create) {
//                 unlink(hugepage_path.c_str());
//             }
//         }
//     }
    
//     // Fallback to shm_open with regular pages
//     if (!using_huge_pages) {
//         fd = shm_open(shm_name.c_str(), flags, 0666);
//         if (fd < 0) {
//             throw std::runtime_error("shm_open failed: " + std::string(strerror(errno)));
//         }

//         if (create && ftruncate64(fd, aligned_size) == -1) {
//             close(fd);
//             shm_unlink(shm_name.c_str());
//             throw std::runtime_error("ftruncate failed: " + std::string(strerror(errno)));
//         }

//         ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
//         close(fd);
        
//         if (ptr == MAP_FAILED) {
//             if (create) shm_unlink(shm_name.c_str());
//             throw std::runtime_error("mmap failed: " + std::string(strerror(errno)));
//         }
        
//         logger->info("Allocated {}GB using regular pages", 
//                    aligned_size / (1024.0*1024.0*1024.0));
        
//         // Hint for transparent huge pages when creating
//         if (create && madvise(ptr, aligned_size, MADV_HUGEPAGE) == 0) {
//             logger->debug("Enabled transparent huge pages hint");
//         }
//     }
    
//     // Server-side initialization
//     if (create) {
//         // Configure NUMA if available
//         if (numa_available() >= 0) {
//             int num_nodes = numa_num_configured_nodes();
//             if (num_nodes >= 2) {
//                 struct bitmask* nodemask = numa_allocate_nodemask();
//                 if (nodemask) {
//                     numa_bitmask_clearall(nodemask);
//                     numa_bitmask_setbit(nodemask, 0);
//                     numa_bitmask_setbit(nodemask, 1);
                    
//                     if (set_mempolicy(MPOL_INTERLEAVE, nodemask->maskp, nodemask->size + 1) == 0) {
//                         logger->info("NUMA memory interleaving enabled across nodes 0-1");
//                     }
//                     numa_free_nodemask(nodemask);
//                 }
//             }
//         }
        
//         // Touch pages to ensure allocation
//         logger->info("Initializing memory pages...");
//         auto start_time = std::chrono::high_resolution_clock::now();
        
//         const long touch_page_size = using_huge_pages ? (2 * 1024 * 1024) : page_size;
//         const int num_threads = std::min(16, (int)std::thread::hardware_concurrency());
//         const int64_t chunk_size = aligned_size / num_threads;
        
//         std::vector<std::thread> threads;
//         for (int i = 0; i < num_threads; i++) {
//             threads.emplace_back([=]() {
//                 int64_t start_offset = i * chunk_size;
//                 int64_t end_offset = (i == num_threads - 1) ? size : start_offset + chunk_size;
//                 volatile char* p = reinterpret_cast<volatile char*>(ptr);
                
//                 for (int64_t offset = start_offset; offset < end_offset; offset += touch_page_size) {
//                     p[offset] = 0;
//                 }
//             });
//         }
        
//         for (auto& t : threads) {
//             t.join();
//         }
        
//         auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
//             std::chrono::high_resolution_clock::now() - start_time);
//         logger->info("Memory initialization completed in {:.2f}s", duration.count() / 1000.0);
//     }
    
//     // Register with CUDA
//     logger->info("Registering {}GB with CUDA...", size / (1024.0*1024.0*1024.0));
//     auto cuda_start = std::chrono::high_resolution_clock::now();
    
//     cudaError_t err = cudaHostRegister(ptr, size, cudaHostRegisterDefault);
    
//     auto cuda_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
//         std::chrono::high_resolution_clock::now() - cuda_start);
    
//     if (err != cudaSuccess) {
//         munmap(ptr, aligned_size);
//         if (create) {
//             if (using_huge_pages) {
//                 unlink(hugepage_path.c_str());
//             } else {
//                 shm_unlink(shm_name.c_str());
//             }
//         }
//         throw std::runtime_error("cudaHostRegister failed: " + 
//                                 std::string(cudaGetErrorString(err)));
//     }
    
//     logger->info("CUDA registration completed in {:.2f}s", cuda_duration.count() / 1000.0);
    
//     return ptr;
// }

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
