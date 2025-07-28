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

// #ifndef MAP_HUGE_2MB
// #define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
// #endif

// spdlog::logger *logger = init_logger("info", "Server");

// bool check_hugepage_availability(int64_t required_size) {
//     std::ifstream meminfo("/proc/meminfo");
//     if (!meminfo) {
//         std::cerr << "Failed to open /proc/meminfo" << std::endl;
//         return false;
//     }    
//     std::string line;
//     long hugepage_size = 0, hugepages_free = 0, hugepages_total = 0;
    
//     while (std::getline(meminfo, line)) {
//         if (line.find("Hugepagesize:") == 0) {
//             std::istringstream iss(line);
//             std::string key, value, unit;
//             iss >> key >> value >> unit;
//             hugepage_size = std::stol(value) * 1024; // Convert KB to bytes
//         } else if (line.find("HugePages_Free:") == 0) {
//             std::istringstream iss(line);
//             std::string key, value;
//             iss >> key >> value;
//             hugepages_free = std::stol(value);
//         } else if (line.find("HugePages_Total:") == 0) {
//             std::istringstream iss(line);
//             std::string key, value;
//             iss >> key >> value;
//             hugepages_total = std::stol(value);
//         }
//     }
    
//     long available_bytes = hugepages_free * hugepage_size;
//     long total_bytes = hugepages_total * hugepage_size;
    
//     std::cout << "Huge page status:" << std::endl;
//     std::cout << "  - Page size: " << hugepage_size / (1024*1024) << "MB" << std::endl;
//     std::cout << "  - Total huge pages: " << hugepages_total 
//               << " (" << total_bytes / (1024*1024*1024) << "GB)" << std::endl;
//     std::cout << "  - Free huge pages: " << hugepages_free 
//               << " (" << available_bytes / (1024*1024*1024) << "GB)" << std::endl;
//     std::cout << "  - Required: " << required_size / (1024*1024*1024) << "GB" << std::endl;
    
//     if (available_bytes < required_size) {
//         long required_pages = (required_size + hugepage_size - 1) / hugepage_size;
//         std::cout << "WARNING: Insufficient huge pages available!" << std::endl;
//         std::cout << "  - Need " << required_pages << " pages, only " << hugepages_free << " available" << std::endl;
//         std::cout << "  - Consider running: echo " << (hugepages_total + required_pages - hugepages_free) 
//                   << " > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages" << std::endl;
//         return false;
//     }
    
//     return true;
// }

// void* allocate_shared_pinned_memory(const std::string& shm_name,
//                                     int64_t size,
//                                     bool create) {
//     void* ptr = nullptr;
//     bool using_huge_pages = false;
    
//     // Validate input parameters
//     if (size <= 0) {
//         throw std::runtime_error("Invalid size: " + std::to_string(size));
//     }
    
//     // CUDA alignment requirements: memory must be aligned to page boundaries
//     const size_t regular_page_size = sysconf(_SC_PAGESIZE);
//     const size_t huge_page_size = 2 * 1024 * 1024; // 2MB
    
//     // Determine alignment based on whether we plan to use huge pages
//     bool plan_huge_pages = false;
//     int64_t aligned_size;
    
//     if (create) {
//         // For server: check if we should plan for huge pages
//         plan_huge_pages = check_hugepage_availability(size); // Check with original size first
        
//         if (plan_huge_pages) {
//             // Align to huge page boundaries (2MB) for optimal huge page usage
//             aligned_size = ((size + huge_page_size - 1) / huge_page_size) * huge_page_size;
//             std::cout << "Planning huge pages: aligning to 2MB boundaries" << std::endl;
//         } else {
//             // Align to regular page boundaries (4KB)
//             aligned_size = ((size + regular_page_size - 1) / regular_page_size) * regular_page_size;
//             std::cout << "No huge pages available: aligning to 4KB boundaries" << std::endl;
//         }
//     } else {
//         // For workers: we don't know what the server used, so align conservatively to huge page boundaries
//         // This ensures we can handle whatever the server created
//         aligned_size = ((size + huge_page_size - 1) / huge_page_size) * huge_page_size;
//         std::cout << "Worker mode: aligning to 2MB boundaries for compatibility" << std::endl;
//     }
    
//     std::cout << "Original size: " << size / (1024*1024) << "MB, "
//               << "Aligned size: " << aligned_size / (1024*1024) << "MB" << std::endl;
    
//     // **ADDED: Verify size alignment is correct**
//     if (plan_huge_pages && create) {
//         if (aligned_size % huge_page_size != 0) {
//             throw std::runtime_error("CRITICAL: Size not properly aligned to huge page boundaries");
//         }
//         std::cout << "✓ Size properly aligned to 2MB huge page boundaries" << std::endl;
//     } else if (!create) {
//         if (aligned_size % huge_page_size != 0) {
//             throw std::runtime_error("CRITICAL: Worker size not aligned to huge page boundaries");
//         }
//         std::cout << "✓ Worker size properly aligned to 2MB boundaries" << std::endl;
//     } else {
//         if (aligned_size % regular_page_size != 0) {
//             throw std::runtime_error("CRITICAL: Size not properly aligned to page boundaries");  
//         }
//         std::cout << "✓ Size properly aligned to 4KB page boundaries" << std::endl;
//     }
    
//     // Declare variables before potential goto to avoid crossing initialization
//     std::string hugepage_path = "/dev/hugepages/" + shm_name;
//     int flags = O_RDWR | (create ? O_CREAT : 0);
//     int fd;
    
//     // **FIXED: For server, we already determined plan_huge_pages above**
//     // For workers, we don't check huge page availability
//     if (create && !plan_huge_pages) {
//         std::cout << "Huge pages not available, will use regular pages" << std::endl;
//         goto fallback_to_shm; // Skip huge page attempts for server
//     }
    
//     // **FIXED: Both server and workers try hugetlbfs first**
//     // For server: create if huge pages available
//     // For workers: always try to connect (server might have used huge pages)
//     if (create) {
//         std::cout << "Server: Trying to create hugetlbfs allocation..." << std::endl;
//     } else {
//         std::cout << "Worker: Trying to connect to hugetlbfs allocation..." << std::endl;
//     }
    
//     fd = open(hugepage_path.c_str(), flags, 0666);
    
//     if (fd >= 0) {
//         if (create) {
//             if (ftruncate64(fd, aligned_size) == -1) {
//                 std::cout << "ftruncate failed for hugetlbfs, trying fallback..." << std::endl;
//                 perror("ftruncate64 on hugetlbfs");
//                 close(fd);
//                 unlink(hugepage_path.c_str());
//                 goto fallback_to_shm;
//             }
//         } else {
//             // **ADDED: For workers, verify the hugetlbfs file has the expected size**
//             struct stat hugepage_stat;
//             if (fstat(fd, &hugepage_stat) == -1) {
//                 close(fd);
//                 std::cout << "fstat failed for hugetlbfs, trying fallback..." << std::endl;
//                 goto fallback_to_shm;
//             }
            
//             if (hugepage_stat.st_size != aligned_size) {
//                 close(fd);
//                 std::cout << "Hugetlbfs size mismatch, trying fallback..." << std::endl;
//                 goto fallback_to_shm;
//             }
            
//             std::cout << "✓ Verified hugetlbfs file size: " << hugepage_stat.st_size / (1024*1024*1024) << "GB" << std::endl;
//         }

//         // mmap the huge page file (no MAP_HUGETLB needed with hugetlbfs!)
//         ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
//         close(fd);
        
//         if (ptr != MAP_FAILED) {
//             if (create) {
//                 std::cout << "✓ Server successfully created " << aligned_size/(1024*1024*1024) 
//                           << "GB using hugetlbfs (2MB pages)!" << std::endl;
//             } else {
//                 std::cout << "✓ Worker successfully connected to " << aligned_size/(1024*1024*1024) 
//                           << "GB hugetlbfs (2MB pages)!" << std::endl;
//             }
//             using_huge_pages = true;
            
//             // Verify alignment for huge pages (should be 2MB aligned)
//             if ((uintptr_t)ptr % (2 * 1024 * 1024) != 0) {
//                 std::cout << "Warning: Huge page memory not 2MB aligned: " << ptr << std::endl;
//             } else {
//                 std::cout << "✓ Memory is properly 2MB aligned: " << ptr << std::endl;
//             }
//             goto success;
//         } else {
//             std::cout << "mmap failed on hugetlbfs, trying fallback..." << std::endl;
//             perror("mmap on hugetlbfs");
//             if (create) unlink(hugepage_path.c_str());
//         }
//     } else {
//         if (create) {
//             std::cout << "Cannot create hugetlbfs file (normal if not mounted), trying fallback..." << std::endl;
//         } else {
//             std::cout << "Cannot connect to hugetlbfs file (server might have used regular shm), trying fallback..." << std::endl;
//         }
//     }

// fallback_to_shm:
//     // **FIXED: Handle both create and connect scenarios properly**
//     if (create) {
//         std::cout << "Server: Trying shm_open with MAP_HUGETLB fallback..." << std::endl;
//     } else {
//         std::cout << "Worker: Trying to connect via regular shm_open..." << std::endl;
//     }
    
//     fd = shm_open(shm_name.c_str(), flags, 0666);
//     if (fd < 0) {
//         if (create) {
//             throw std::runtime_error("shm_open failed for " + shm_name + " (server creation failed)");
//         } else {
//             throw std::runtime_error("shm_open failed for " + shm_name + 
//                                    " (worker cannot connect - check if server process created the shared memory)");
//         }
//     }

//     if (create) {
//         if (ftruncate64(fd, aligned_size) == -1) {
//             close(fd);
//             throw std::runtime_error("ftruncate failed for " + shm_name);
//         }
//     } else {
//         // **ADDED: For workers, verify the shared memory has the expected size**
//         struct stat shm_stat;
//         if (fstat(fd, &shm_stat) == -1) {
//             close(fd);
//             throw std::runtime_error("fstat failed for " + shm_name + " (cannot verify shared memory size)");
//         }
        
//         if (shm_stat.st_size != aligned_size) {
//             close(fd);
//             throw std::runtime_error("Shared memory size mismatch: expected " + 
//                                    std::to_string(aligned_size) + " bytes, found " + 
//                                    std::to_string(shm_stat.st_size) + " bytes");
//         }
        
//         std::cout << "✓ Verified shared memory size: " << shm_stat.st_size / (1024*1024*1024) << "GB" << std::endl;
//     }

//     // **FIXED: Try MAP_HUGETLB only when creating (server) and we planned for huge pages**
//     if (create && plan_huge_pages) {
//         ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, 
//                    MAP_SHARED | MAP_HUGETLB | MAP_HUGE_2MB, fd, 0);
        
//         if (ptr != MAP_FAILED) {
//             std::cout << "✓ Server successfully allocated " << aligned_size/(1024*1024*1024) 
//                       << "GB using shm_open + MAP_HUGETLB!" << std::endl;
//             using_huge_pages = true;
//             close(fd);
            
//             // Verify 2MB alignment
//             if ((uintptr_t)ptr % (2 * 1024 * 1024) != 0) {
//                 std::cout << "Warning: Huge page memory not 2MB aligned: " << ptr << std::endl;
//             } else {
//                 std::cout << "✓ Memory is properly 2MB aligned: " << ptr << std::endl;
//             }
//             goto success;
//         } else {
//             std::cout << "shm_open + MAP_HUGETLB failed, trying regular pages..." << std::endl;
//             perror("mmap with MAP_HUGETLB on shm_open");
//         }
//     }
    
//     // **FIXED: Regular mmap for both create=false (workers) and create=true fallback**
//     ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
//     close(fd);
    
//     if (ptr == MAP_FAILED) {
//         throw std::runtime_error("All mmap methods failed for " + shm_name);
//     }
    
//     // **FIXED: Different handling for server vs workers**
//     if (create) {
//         std::cout << "✓ Server successfully created " << aligned_size/(1024*1024*1024) 
//                   << "GB using regular shared memory" << std::endl;
        
//         // Verify alignment for regular pages
//         if ((uintptr_t)ptr % regular_page_size != 0) {
//             std::cout << "Warning: Regular memory not page aligned" << std::endl;
//         } else {
//             std::cout << "✓ Memory is properly page aligned: " << ptr << std::endl;
//         }
        
//         // Use madvise hint for transparent huge pages
//         int ret = madvise(ptr, aligned_size, MADV_HUGEPAGE);
//         if (ret == 0) {
//             std::cout << "✓ Successfully hinted for transparent huge pages (may use 2MB pages)" << std::endl;
//         } else {
//             std::cout << "madvise(MADV_HUGEPAGE) failed, using regular 4KB pages" << std::endl;
//             perror("madvise");
//         }
//     } else {
//         std::cout << "✓ Worker successfully connected to existing shared memory " << aligned_size/(1024*1024*1024) 
//                   << "GB" << std::endl;
        
//         // **ADDED: Detect if the existing memory uses huge pages by checking alignment**
//         if ((uintptr_t)ptr % (2 * 1024 * 1024) == 0) {
//             std::cout << "✓ Existing memory appears to use 2MB huge pages (properly aligned)" << std::endl;
//             using_huge_pages = true;
//         } else {
//             std::cout << "✓ Existing memory uses regular pages" << std::endl;
//         }
//     }

// success:
//     // Validate pointer and alignment before proceeding
//     if (ptr == nullptr) {
//         throw std::runtime_error("Memory allocation returned null pointer");
//     }
    
//     // Check final alignment - use the appropriate alignment based on what we actually got
//     size_t required_alignment = using_huge_pages ? huge_page_size : regular_page_size;
//     if ((uintptr_t)ptr % required_alignment != 0) {
//         std::cout << "ERROR: Final memory alignment check failed!" << std::endl;
//         std::cout << "Pointer: " << ptr << ", Required alignment: " << required_alignment << std::endl;
//         std::cout << "Pointer modulo: " << ((uintptr_t)ptr % required_alignment) << std::endl;
        
//         // For CUDA, we need at least regular page alignment, so let's check that
//         if ((uintptr_t)ptr % regular_page_size != 0) {
//             munmap(ptr, aligned_size);
//             throw std::runtime_error("Memory is not page-aligned, cudaHostRegister will fail");
//         } else {
//             std::cout << "Memory is at least page-aligned, proceeding..." << std::endl;
//         }
//     } else {
//         std::cout << "✓ Memory alignment verified: " << required_alignment / 1024 << "KB alignment" << std::endl;
//         if (create) {
//             std::cout << "✓ Memory allocation and setup completed successfully" << std::endl;
//         } else {
//             std::cout << "✓ Successfully connected to existing shared memory" << std::endl;
//         }
//     }
    
//     // **FIXED: Only perform NUMA setup and page touching when creating new shared memory (server)**
//     if (create) {
//         if (numa_available() >= 0) {
//             int num_nodes = numa_num_configured_nodes();
//             std::cout << "NUMA available with " << num_nodes << " nodes." << std::endl;
            
//             if (num_nodes >= 2) {
//                 std::cout << "Interleaving memory across NUMA nodes 0 and 1." << std::endl;

//                 // Create proper nodemask using numa library functions
//                 struct bitmask* nodemask = numa_allocate_nodemask();
//                 if (!nodemask) {
//                     std::cerr << "Failed to allocate nodemask" << std::endl;
//                 } else {
//                     // Clear all bits first
//                     numa_bitmask_clearall(nodemask);
//                     // Set bits for nodes 0 and 1
//                     numa_bitmask_setbit(nodemask, 0);
//                     numa_bitmask_setbit(nodemask, 1);

//                     // Use set_mempolicy for the current process/thread
//                     int ret = set_mempolicy(MPOL_INTERLEAVE, 
//                                           nodemask->maskp, 
//                                           nodemask->size + 1);
//                     if (ret != 0) {
//                         perror("set_mempolicy(MPOL_INTERLEAVE)");
//                     } else {
//                         std::cout << "✓ NUMA memory policy set successfully" << std::endl;
//                     }

//                     numa_free_nodemask(nodemask);
//                 }
//             } else {
//                 std::cout << "Only " << num_nodes << " NUMA node(s) available, skipping interleaving." << std::endl;
//             }

//             // Multi-threaded page touching
//             std::cout << "Starting multi-threaded page touching..." << std::endl;
//             auto touch_start = std::chrono::high_resolution_clock::now();
            
//             // Use appropriate page size based on allocation method
//             long touch_page_size = using_huge_pages ? huge_page_size : regular_page_size;
//             std::cout << "Using page size: " << touch_page_size / 1024 << " KB" << std::endl;
            
//             // Determine optimal number of threads
//             int num_threads = std::min(16, std::max(2, (int)std::thread::hardware_concurrency() / 2));
//             std::cout << "Using " << num_threads << " threads for page touching" << std::endl;
            
//             // Use aligned_size consistently - since it's properly aligned, we can use exact division
//             int64_t total_pages = aligned_size / touch_page_size;
//             std::cout << "Total pages to touch: " << total_pages << std::endl;
            
//             // **ADDED: Critical safety check - ensure aligned_size is a multiple of touch_page_size**
//             if (aligned_size % touch_page_size != 0) {
//                 std::cout << "ERROR: aligned_size (" << aligned_size << ") is not a multiple of touch_page_size (" 
//                           << touch_page_size << ")" << std::endl;
//                 std::cout << "This would cause page touching to access unallocated memory!" << std::endl;
//                 throw std::runtime_error("Size/page alignment mismatch - would cause memory access violation");
//             }
//             std::cout << "✓ Size is properly aligned to page size - safe for page touching" << std::endl;
            
//             // Ensure thread boundaries are page-aligned to avoid race conditions
//             int64_t pages_per_thread = total_pages / num_threads;
            
//             std::vector<std::thread> threads;
//             std::atomic<int64_t> completed_pages{0};
//             std::atomic<bool> progress_active{true};
            
//             // Progress reporting thread
//             std::thread progress_thread([&]() {
//                 while (progress_active.load()) {
//                     int64_t current = completed_pages.load();
//                     double progress = (double)current / total_pages * 100.0;
//                     std::cout << "Page touching progress: " << std::fixed << std::setprecision(1) 
//                               << progress << "% (" << current << "/" << total_pages << " pages)" << std::endl;
//                     std::this_thread::sleep_for(std::chrono::seconds(2));
//                 }
//             });
            
//             // Worker function with proper page-aligned boundaries
//             auto touch_worker = [&](int thread_id) {
//                 // Calculate page-aligned boundaries for this thread
//                 int64_t start_page = thread_id * pages_per_thread;
//                 int64_t end_page;
                
//                 if (thread_id == num_threads - 1) {
//                     // Last thread handles any remaining pages
//                     end_page = total_pages;
//                 } else {
//                     end_page = start_page + pages_per_thread;
//                 }
                
//                 int64_t start_offset = start_page * touch_page_size;
//                 int64_t end_offset = std::min((int64_t)aligned_size, end_page * touch_page_size);
                
//                 volatile char* p = reinterpret_cast<volatile char*>(ptr);
//                 int64_t local_pages_touched = 0;
                
//                 // std::cout << "Thread " << thread_id << " touching pages " << start_page 
//                 //           << " to " << (end_page - 1) << " (offset " << start_offset 
//                 //           << " to " << end_offset << ")" << std::endl;
                
//                 // Touch pages with proper bounds checking using aligned_size
//                 for (int64_t offset = start_offset; offset < end_offset; offset += touch_page_size) {
//                     // Double-check bounds to prevent accessing beyond allocated memory
//                     if (offset >= aligned_size) {
//                         std::cout << "WARNING: Thread " << thread_id << " reached memory boundary, stopping" << std::endl;
//                         break;
//                     }
                    
//                     // Ensure we don't go beyond allocated memory
//                     if (offset < aligned_size) {
//                         p[offset] = 0;
//                         local_pages_touched++;
                        
//                         // Update global counter every 1000 pages to reduce contention
//                         if (local_pages_touched % 1000 == 0) {
//                             completed_pages.fetch_add(1000);
//                         }
//                     }
//                 }
                
//                 // Add remaining pages to counter
//                 completed_pages.fetch_add(local_pages_touched % 1000);
                
//                 // std::cout << "Thread " << thread_id << " completed " << local_pages_touched << " pages" << std::endl;
//             };
            
//             // Launch worker threads
//             for (int i = 0; i < num_threads; i++) {
//                 threads.emplace_back(touch_worker, i);
//             }
            
//             // Wait for all threads to complete
//             for (auto& t : threads) {
//                 t.join();
//             }
            
//             // Stop progress thread
//             progress_active.store(false);
//             progress_thread.join();
            
//             // Touch the last page using aligned_size bounds
//             if (aligned_size > 0) {
//                 volatile char* p = reinterpret_cast<volatile char*>(ptr);
//                 p[aligned_size - 1] = 0;
//                 std::cout << "Touched final byte at offset " << (aligned_size - 1) << std::endl;
//             }
            
//             auto touch_end = std::chrono::high_resolution_clock::now();
//             auto touch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(touch_end - touch_start);
            
//             int64_t final_pages = completed_pages.load();
//             // Use aligned_size for throughput calculation
//             double throughput = (double)aligned_size / (1024.0 * 1024.0 * 1024.0) / (touch_duration.count() / 1000.0);
            
//             std::cout << "✓ Multi-threaded page touching completed:" << std::endl;
//             std::cout << "  - Pages touched: " << final_pages << " / " << total_pages << std::endl;
//             std::cout << "  - Memory size: " << aligned_size / (1024*1024*1024) << " GB" << std::endl;
//             std::cout << "  - Time: " << touch_duration.count() << " ms (" 
//                       << touch_duration.count() / 1000.0 << " seconds)" << std::endl;
//             std::cout << "  - Throughput: " << std::fixed << std::setprecision(2) 
//                       << throughput << " GB/s" << std::endl;
//         } else {
//             std::cout << "NUMA not available on this system." << std::endl;
//         }
//     } else {
//         // Worker mode - shared memory already initialized by server
//         std::cout << "Worker mode: shared memory already initialized, skipping NUMA setup and page touching" << std::endl;
//     }
    
//     return ptr;
// }

// // **ADDED: Function to check huge page availability before allocation**
// bool check_hugepage_availability(int64_t required_size) {
//     std::ifstream meminfo("/proc/meminfo");
//     std::string line;
//     long hugepage_size = 0, hugepages_free = 0, hugepages_total = 0;
    
//     while (std::getline(meminfo, line)) {
//         if (line.find("Hugepagesize:") == 0) {
//             std::istringstream iss(line);
//             std::string key, value, unit;
//             iss >> key >> value >> unit;
//             hugepage_size = std::stol(value) * 1024; // Convert KB to bytes
//         } else if (line.find("HugePages_Free:") == 0) {
//             std::istringstream iss(line);
//             std::string key, value;
//             iss >> key >> value;
//             hugepages_free = std::stol(value);
//         } else if (line.find("HugePages_Total:") == 0) {
//             std::istringstream iss(line);
//             std::string key, value;
//             iss >> key >> value;
//             hugepages_total = std::stol(value);
//         }
//     }
    
//     long available_bytes = hugepages_free * hugepage_size;
//     long total_bytes = hugepages_total * hugepage_size;
    
//     std::cout << "Huge page status:" << std::endl;
//     std::cout << "  - Page size: " << hugepage_size / (1024*1024) << "MB" << std::endl;
//     std::cout << "  - Total huge pages: " << hugepages_total 
//               << " (" << total_bytes / (1024*1024*1024) << "GB)" << std::endl;
//     std::cout << "  - Free huge pages: " << hugepages_free 
//               << " (" << available_bytes / (1024*1024*1024) << "GB)" << std::endl;
//     std::cout << "  - Required: " << required_size / (1024*1024*1024) << "GB" << std::endl;
    
//     if (available_bytes < required_size) {
//         long required_pages = (required_size + hugepage_size - 1) / hugepage_size;
//         std::cout << "WARNING: Insufficient huge pages available!" << std::endl;
//         std::cout << "  - Need " << required_pages << " pages, only " << hugepages_free << " available" << std::endl;
//         std::cout << "  - Consider running: echo " << (hugepages_total + required_pages - hugepages_free) 
//                   << " > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages" << std::endl;
//         return false;
//     }
    
//     return true;
// }


// void* allocate_shared_pinned_memory(const std::string& shm_name,
//                                     int64_t size,
//                                     bool create) {
//     void* ptr = nullptr;
//     bool using_huge_pages = false;
    
//     // Validate input parameters
//     if (size <= 0) {
//         throw std::runtime_error("Invalid size: " + std::to_string(size));
//     }
    
//     // CUDA alignment requirements: memory must be aligned to page boundaries
//     const size_t page_size = sysconf(_SC_PAGESIZE);
    
//     // Round up size to page boundary to ensure proper alignment
//     int64_t aligned_size = ((size + page_size - 1) / page_size) * page_size;
    
//     std::cout << "Original size: " << size / (1024*1024) << "MB, "
//               << "Aligned size: " << aligned_size / (1024*1024) << "MB" << std::endl;
    
//     // Declare variables before potential goto to avoid crossing initialization
//     std::string hugepage_path = "/dev/hugepages/" + shm_name;
//     int flags = O_RDWR | (create ? O_CREAT : 0);
//     int fd;
//     bool hugepages_available = false;
    
//     // **FIXED: Only check huge page availability for server (create=true)**
//     if (create) {
//         hugepages_available = check_hugepage_availability(aligned_size);
//         if (!hugepages_available) {
//             std::cout << "Huge pages not available, will use regular pages" << std::endl;
//             goto fallback_to_shm; // Skip huge page attempts for server
//         }
//     }
    
//     // **FIXED: Both server and workers try hugetlbfs first**
//     // For server: create if huge pages available
//     // For workers: always try to connect (server might have used huge pages)
//     if (create) {
//         std::cout << "Server: Trying to create hugetlbfs allocation..." << std::endl;
//     } else {
//         std::cout << "Worker: Trying to connect to hugetlbfs allocation..." << std::endl;
//     }
    
//     fd = open(hugepage_path.c_str(), flags, 0666);
    
//     if (fd >= 0) {
//         if (create) {
//             if (ftruncate64(fd, aligned_size) == -1) {
//                 std::cout << "ftruncate failed for hugetlbfs, trying fallback..." << std::endl;
//                 perror("ftruncate64 on hugetlbfs");
//                 close(fd);
//                 unlink(hugepage_path.c_str());
//                 goto fallback_to_shm;
//             }
//         } else {
//             // **ADDED: For workers, verify the hugetlbfs file has the expected size**
//             struct stat hugepage_stat;
//             if (fstat(fd, &hugepage_stat) == -1) {
//                 close(fd);
//                 std::cout << "fstat failed for hugetlbfs, trying fallback..." << std::endl;
//                 goto fallback_to_shm;
//             }
            
//             if (hugepage_stat.st_size != aligned_size) {
//                 close(fd);
//                 std::cout << "Hugetlbfs size mismatch, trying fallback..." << std::endl;
//                 goto fallback_to_shm;
//             }
            
//             std::cout << "✓ Verified hugetlbfs file size: " << hugepage_stat.st_size / (1024*1024*1024) << "GB" << std::endl;
//         }

//         // mmap the huge page file (no MAP_HUGETLB needed with hugetlbfs!)
//         ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
//         close(fd);
        
//         if (ptr != MAP_FAILED) {
//             if (create) {
//                 std::cout << "✓ Server successfully created " << aligned_size/(1024*1024*1024) 
//                           << "GB using hugetlbfs (2MB pages)!" << std::endl;
//             } else {
//                 std::cout << "✓ Worker successfully connected to " << aligned_size/(1024*1024*1024) 
//                           << "GB hugetlbfs (2MB pages)!" << std::endl;
//             }
//             using_huge_pages = true;
            
//             // Verify alignment for huge pages (should be 2MB aligned)
//             if ((uintptr_t)ptr % (2 * 1024 * 1024) != 0) {
//                 std::cout << "Warning: Huge page memory not 2MB aligned: " << ptr << std::endl;
//             } else {
//                 std::cout << "✓ Memory is properly 2MB aligned: " << ptr << std::endl;
//             }
//             goto success;
//         } else {
//             std::cout << "mmap failed on hugetlbfs, trying fallback..." << std::endl;
//             perror("mmap on hugetlbfs");
//             if (create) unlink(hugepage_path.c_str());
//         }
//     } else {
//         if (create) {
//             std::cout << "Cannot create hugetlbfs file (normal if not mounted), trying fallback..." << std::endl;
//         } else {
//             std::cout << "Cannot connect to hugetlbfs file (server might have used regular shm), trying fallback..." << std::endl;
//         }
//     }

// fallback_to_shm:
//     // **FIXED: Handle both create and connect scenarios properly**
//     if (create) {
//         std::cout << "Server: Trying shm_open with MAP_HUGETLB fallback..." << std::endl;
//     } else {
//         std::cout << "Worker: Trying to connect via regular shm_open..." << std::endl;
//     }
    
//     fd = shm_open(shm_name.c_str(), flags, 0666);
//     if (fd < 0) {
//         if (create) {
//             throw std::runtime_error("shm_open failed for " + shm_name + " (server creation failed)");
//         } else {
//             throw std::runtime_error("shm_open failed for " + shm_name + 
//                                    " (worker cannot connect - check if server process created the shared memory)");
//         }
//     }

//     if (create) {
//         if (ftruncate64(fd, aligned_size) == -1) {
//             close(fd);
//             throw std::runtime_error("ftruncate failed for " + shm_name);
//         }
//     } else {
//         // **ADDED: For workers, verify the shared memory has the expected size**
//         struct stat shm_stat;
//         if (fstat(fd, &shm_stat) == -1) {
//             close(fd);
//             throw std::runtime_error("fstat failed for " + shm_name + " (cannot verify shared memory size)");
//         }
        
//         if (shm_stat.st_size != aligned_size) {
//             close(fd);
//             throw std::runtime_error("Shared memory size mismatch: expected " + 
//                                    std::to_string(aligned_size) + " bytes, found " + 
//                                    std::to_string(shm_stat.st_size) + " bytes");
//         }
        
//         std::cout << "✓ Verified shared memory size: " << shm_stat.st_size / (1024*1024*1024) << "GB" << std::endl;
//     }

//     // **FIXED: Try MAP_HUGETLB only when creating (server) and we're in fallback mode**
//     if (create && hugepages_available) {
//         ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, 
//                    MAP_SHARED | MAP_HUGETLB | MAP_HUGE_2MB, fd, 0);
        
//         if (ptr != MAP_FAILED) {
//             std::cout << "✓ Server successfully allocated " << aligned_size/(1024*1024*1024) 
//                       << "GB using shm_open + MAP_HUGETLB!" << std::endl;
//             using_huge_pages = true;
//             close(fd);
            
//             // Verify 2MB alignment
//             if ((uintptr_t)ptr % (2 * 1024 * 1024) != 0) {
//                 std::cout << "Warning: Huge page memory not 2MB aligned: " << ptr << std::endl;
//             } else {
//                 std::cout << "✓ Memory is properly 2MB aligned: " << ptr << std::endl;
//             }
//             goto success;
//         } else {
//             std::cout << "shm_open + MAP_HUGETLB failed, trying regular pages..." << std::endl;
//             perror("mmap with MAP_HUGETLB on shm_open");
//         }
//     }
    
//     // **FIXED: Regular mmap for both create=false (workers) and create=true fallback**
//     ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
//     close(fd);
    
//     if (ptr == MAP_FAILED) {
//         throw std::runtime_error("All mmap methods failed for " + shm_name);
//     }
    
//     // **FIXED: Different handling for server vs workers**
//     if (create) {
//         std::cout << "✓ Server successfully created " << aligned_size/(1024*1024*1024) 
//                   << "GB using regular shared memory" << std::endl;
        
//         // Verify alignment for regular pages
//         if ((uintptr_t)ptr % page_size != 0) {
//             std::cout << "Warning: Regular memory not page aligned" << std::endl;
//         } else {
//             std::cout << "✓ Memory is properly page aligned: " << ptr << std::endl;
//         }
        
//         // Use madvise hint for transparent huge pages
//         int ret = madvise(ptr, aligned_size, MADV_HUGEPAGE);
//         if (ret == 0) {
//             std::cout << "✓ Successfully hinted for transparent huge pages (may use 2MB pages)" << std::endl;
//         } else {
//             std::cout << "madvise(MADV_HUGEPAGE) failed, using regular 4KB pages" << std::endl;
//             perror("madvise");
//         }
//     } else {
//         std::cout << "✓ Worker successfully connected to existing shared memory " << aligned_size/(1024*1024*1024) 
//                   << "GB" << std::endl;
        
//         // **ADDED: Detect if the existing memory uses huge pages by checking alignment**
//         if ((uintptr_t)ptr % (2 * 1024 * 1024) == 0) {
//             std::cout << "✓ Existing memory appears to use 2MB huge pages (properly aligned)" << std::endl;
//             using_huge_pages = true;
//         } else {
//             std::cout << "✓ Existing memory uses regular pages" << std::endl;
//         }
//     }

// success:
//     // Validate pointer and alignment before proceeding
//     if (ptr == nullptr) {
//         throw std::runtime_error("Memory allocation returned null pointer");
//     }
    
//     // Check final alignment
//     size_t final_alignment = using_huge_pages ? (2 * 1024 * 1024) : page_size;
//     if ((uintptr_t)ptr % final_alignment != 0) {
//         std::cout << "ERROR: Final memory alignment check failed!" << std::endl;
//         std::cout << "Pointer: " << ptr << ", Required alignment: " << final_alignment << std::endl;
//         std::cout << "Pointer modulo: " << ((uintptr_t)ptr % final_alignment) << std::endl;
        
//         // For CUDA, we need at least page alignment, so let's check that
//         if ((uintptr_t)ptr % page_size != 0) {
//             munmap(ptr, aligned_size);
//             throw std::runtime_error("Memory is not page-aligned, cudaHostRegister will fail");
//         } else {
//             std::cout << "Memory is at least page-aligned, proceeding..." << std::endl;
//         }
//     } else {
//         if (create) {
//             std::cout << "✓ Memory allocation and setup completed successfully" << std::endl;
//         } else {
//             std::cout << "✓ Successfully connected to existing shared memory" << std::endl;
//         }
//     }
    
//     // **FIXED: Only perform NUMA setup and page touching when creating new shared memory (server)**
//     if (create) {
//         if (numa_available() >= 0) {
//             int num_nodes = numa_num_configured_nodes();
//             std::cout << "NUMA available with " << num_nodes << " nodes." << std::endl;
            
//             if (num_nodes >= 2) {
//                 std::cout << "Interleaving memory across NUMA nodes 0 and 1." << std::endl;

//                 // Create proper nodemask using numa library functions
//                 struct bitmask* nodemask = numa_allocate_nodemask();
//                 if (!nodemask) {
//                     std::cerr << "Failed to allocate nodemask" << std::endl;
//                 } else {
//                     // Clear all bits first
//                     numa_bitmask_clearall(nodemask);
//                     // Set bits for nodes 0 and 1
//                     numa_bitmask_setbit(nodemask, 0);
//                     numa_bitmask_setbit(nodemask, 1);

//                     // Use set_mempolicy for the current process/thread
//                     int ret = set_mempolicy(MPOL_INTERLEAVE, 
//                                           nodemask->maskp, 
//                                           nodemask->size + 1);
//                     if (ret != 0) {
//                         perror("set_mempolicy(MPOL_INTERLEAVE)");
//                     } else {
//                         std::cout << "✓ NUMA memory policy set successfully" << std::endl;
//                     }

//                     numa_free_nodemask(nodemask);
//                 }
//             } else {
//                 std::cout << "Only " << num_nodes << " NUMA node(s) available, skipping interleaving." << std::endl;
//             }

//             // Multi-threaded page touching
//             std::cout << "Starting multi-threaded page touching..." << std::endl;
//             auto touch_start = std::chrono::high_resolution_clock::now();
            
//             // Use appropriate page size based on allocation method
//             long touch_page_size = using_huge_pages ? (2 * 1024 * 1024) : sysconf(_SC_PAGESIZE);
//             std::cout << "Using page size: " << touch_page_size / 1024 << " KB" << std::endl;
            
//             // Determine optimal number of threads
//             int num_threads = std::min(16, std::max(2, (int)std::thread::hardware_concurrency() / 2));
//             std::cout << "Using " << num_threads << " threads for page touching" << std::endl;
            
//             // Use aligned_size consistently
//             int64_t total_pages = (aligned_size + touch_page_size - 1) / touch_page_size;
//             std::cout << "Total pages to touch: " << total_pages << std::endl;
            
//             // Ensure thread boundaries are page-aligned to avoid race conditions
//             int64_t pages_per_thread = total_pages / num_threads;
            
//             std::vector<std::thread> threads;
//             std::atomic<int64_t> completed_pages{0};
//             std::atomic<bool> progress_active{true};
            
//             // Progress reporting thread
//             std::thread progress_thread([&]() {
//                 while (progress_active.load()) {
//                     int64_t current = completed_pages.load();
//                     double progress = (double)current / total_pages * 100.0;
//                     std::cout << "Page touching progress: " << std::fixed << std::setprecision(1) 
//                               << progress << "% (" << current << "/" << total_pages << " pages)" << std::endl;
//                     std::this_thread::sleep_for(std::chrono::seconds(2));
//                 }
//             });
            
//             // Worker function with proper page-aligned boundaries
//             auto touch_worker = [&](int thread_id) {
//                 // Calculate page-aligned boundaries for this thread
//                 int64_t start_page = thread_id * pages_per_thread;
//                 int64_t end_page;
                
//                 if (thread_id == num_threads - 1) {
//                     // Last thread handles any remaining pages
//                     end_page = total_pages;
//                 } else {
//                     end_page = start_page + pages_per_thread;
//                 }
                
//                 int64_t start_offset = start_page * touch_page_size;
//                 int64_t end_offset = std::min((int64_t)aligned_size, end_page * touch_page_size);
                
//                 volatile char* p = reinterpret_cast<volatile char*>(ptr);
//                 int64_t local_pages_touched = 0;
                
//                 std::cout << "Thread " << thread_id << " touching pages " << start_page 
//                           << " to " << (end_page - 1) << " (offset " << start_offset 
//                           << " to " << end_offset << ")" << std::endl;
                
//                 // Touch pages with proper bounds checking using aligned_size
//                 for (int64_t offset = start_offset; offset < end_offset; offset += touch_page_size) {
//                     // Ensure we don't go beyond allocated memory
//                     if (offset < aligned_size) {
//                         p[offset] = 0;
//                         local_pages_touched++;
                        
//                         // Update global counter every 1000 pages to reduce contention
//                         if (local_pages_touched % 1000 == 0) {
//                             completed_pages.fetch_add(1000);
//                         }
//                     }
//                 }
                
//                 // Add remaining pages to counter
//                 completed_pages.fetch_add(local_pages_touched % 1000);
                
//                 std::cout << "Thread " << thread_id << " completed " << local_pages_touched << " pages" << std::endl;
//             };
            
//             // Launch worker threads
//             for (int i = 0; i < num_threads; i++) {
//                 threads.emplace_back(touch_worker, i);
//             }
            
//             // Wait for all threads to complete
//             for (auto& t : threads) {
//                 t.join();
//             }
            
//             // Stop progress thread
//             progress_active.store(false);
//             progress_thread.join();
            
//             // Touch the last page using aligned_size bounds
//             if (aligned_size > 0) {
//                 volatile char* p = reinterpret_cast<volatile char*>(ptr);
//                 p[aligned_size - 1] = 0;
//                 std::cout << "Touched final byte at offset " << (aligned_size - 1) << std::endl;
//             }
            
//             auto touch_end = std::chrono::high_resolution_clock::now();
//             auto touch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(touch_end - touch_start);
            
//             int64_t final_pages = completed_pages.load();
//             // Use aligned_size for throughput calculation
//             double throughput = (double)aligned_size / (1024.0 * 1024.0 * 1024.0) / (touch_duration.count() / 1000.0);
            
//             std::cout << "✓ Multi-threaded page touching completed:" << std::endl;
//             std::cout << "  - Pages touched: " << final_pages << " / " << total_pages << std::endl;
//             std::cout << "  - Memory size: " << aligned_size / (1024*1024*1024) << " GB" << std::endl;
//             std::cout << "  - Time: " << touch_duration.count() << " ms (" 
//                       << touch_duration.count() / 1000.0 << " seconds)" << std::endl;
//             std::cout << "  - Throughput: " << std::fixed << std::setprecision(2) 
//                       << throughput << " GB/s" << std::endl;
//         } else {
//             std::cout << "NUMA not available on this system." << std::endl;
//         }
//     } else {
//         // Worker mode - shared memory already initialized by server
//         std::cout << "Worker mode: shared memory already initialized, skipping NUMA setup and page touching" << std::endl;
//     }
    
//     return ptr;
// }
// void* allocate_shared_pinned_memory(const std::string& shm_name,
//                                     int64_t size,
//                                     bool create) {
//     void* ptr = nullptr;
//     bool using_huge_pages = false;
    
//     // Method 1: Try hugetlbfs first (most reliable for huge pages)
//     std::string hugepage_path = "/dev/hugepages/" + shm_name;
//     int flags = O_RDWR | (create ? O_CREAT : 0);
//     int fd = open(hugepage_path.c_str(), flags, 0666);
    
//     if (fd >= 0) {
//         std::cout << "Trying hugetlbfs allocation..." << std::endl;
        
//         if (create) {
//             if (ftruncate64(fd, size) == -1) {
//                 std::cout << "ftruncate failed for hugetlbfs, trying fallback..." << std::endl;
//                 perror("ftruncate64 on hugetlbfs");
//                 close(fd);
//                 unlink(hugepage_path.c_str());
//                 goto fallback_to_shm;
//             }
//         }

//         // mmap the huge page file (no MAP_HUGETLB needed with hugetlbfs!)
//         ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
//         close(fd);
        
//         if (ptr != MAP_FAILED) {
//             std::cout << "✓ Successfully allocated " << size/(1024*1024*1024) 
//                       << "GB using hugetlbfs (2MB pages)!" << std::endl;
//             using_huge_pages = true;
//             goto success;
//         } else {
//             std::cout << "mmap failed on hugetlbfs, trying fallback..." << std::endl;
//             perror("mmap on hugetlbfs");
//             if (create) unlink(hugepage_path.c_str());
//         }
//     } else {
//         std::cout << "Cannot open hugetlbfs file (normal if not mounted), trying fallback..." << std::endl;
//     }

// fallback_to_shm:
//     // Method 2: Try shm_open with MAP_HUGETLB (will likely fail but worth trying)
//     std::cout << "Trying shm_open with MAP_HUGETLB..." << std::endl;
//     fd = shm_open(shm_name.c_str(), flags, 0666);
//     if (fd < 0) {
//         throw std::runtime_error("shm_open failed for " + shm_name);
//     }

//     if (create) {
//         if (ftruncate64(fd, size) == -1) {
//             close(fd);
//             throw std::runtime_error("ftruncate failed for " + shm_name);
//         }
//     }

//     // Try MAP_HUGETLB with shm_open (will probably fail)
//     ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, 
//                MAP_SHARED | MAP_HUGETLB | MAP_HUGE_2MB, fd, 0);
    
//     if (ptr != MAP_FAILED) {
//         std::cout << "✓ Successfully allocated " << size/(1024*1024*1024) 
//                   << "GB using shm_open + MAP_HUGETLB!" << std::endl;
//         using_huge_pages = true;
//         close(fd);
//         goto success;
//     } else {
//         std::cout << "shm_open + MAP_HUGETLB failed (expected), trying regular pages..." << std::endl;
//         perror("mmap with MAP_HUGETLB on shm_open");
        
//         // Method 3: Fallback to regular pages with huge page hints
//         ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
//         close(fd);
        
//         if (ptr == MAP_FAILED) {
//             throw std::runtime_error("All mmap methods failed for " + shm_name);
//         }
        
//         if (create) {
//             // Use madvise hint for transparent huge pages
//             int ret = madvise(ptr, size, MADV_HUGEPAGE);
//             if (ret == 0) {
//                 std::cout << "✓ Successfully hinted for transparent huge pages (may use 2MB pages)" << std::endl;
//             } else {
//                 std::cout << "madvise(MADV_HUGEPAGE) failed, using regular 4KB pages" << std::endl;
//                 perror("madvise");
//             }
//         }
//     }

// success:
//     if (create) {
//         if (numa_available() >= 0) {
//             int num_nodes = numa_num_configured_nodes();
//             std::cout << "NUMA available with " << num_nodes << " nodes." << std::endl;
            
//             if (num_nodes >= 2) {
//                 std::cout << "Interleaving memory across NUMA nodes 0 and 1." << std::endl;

//                 // Create proper nodemask using numa library functions
//                 struct bitmask* nodemask = numa_allocate_nodemask();
//                 if (!nodemask) {
//                     std::cerr << "Failed to allocate nodemask" << std::endl;
//                 } else {
//                     // Clear all bits first
//                     numa_bitmask_clearall(nodemask);
//                     // Set bits for nodes 0 and 1
//                     numa_bitmask_setbit(nodemask, 0);
//                     numa_bitmask_setbit(nodemask, 1);

//                     // Use set_mempolicy for the current process/thread
//                     int ret = set_mempolicy(MPOL_INTERLEAVE, 
//                                           nodemask->maskp, 
//                                           nodemask->size + 1);
//                     if (ret != 0) {
//                         perror("set_mempolicy(MPOL_INTERLEAVE)");
//                     } else {
//                         std::cout << "✓ NUMA memory policy set successfully" << std::endl;
//                     }

//                     numa_free_nodemask(nodemask);
//                 }
//             } else {
//                 std::cout << "Only " << num_nodes << " NUMA node(s) available, skipping interleaving." << std::endl;
//             }

//             // Multi-threaded page touching
//             std::cout << "Starting multi-threaded page touching..." << std::endl;
//             auto touch_start = std::chrono::high_resolution_clock::now();
            
//             // Use appropriate page size based on allocation method
//             long page_size = using_huge_pages ? (2 * 1024 * 1024) : sysconf(_SC_PAGESIZE);
//             std::cout << "Using page size: " << page_size / 1024 << " KB" << std::endl;
            
//             // Determine optimal number of threads
//             int num_threads = std::min(16, std::max(2, (int)std::thread::hardware_concurrency() / 2));
//             std::cout << "Using " << num_threads << " threads for page touching" << std::endl;
            
//             // Calculate work distribution
//             int64_t total_pages = (size + page_size - 1) / page_size;
//             int64_t chunk_size = size / num_threads;
            
//             std::vector<std::thread> threads;
//             std::atomic<int64_t> completed_pages{0};
//             std::atomic<bool> progress_active{true};
            
//             // Progress reporting thread
//             std::thread progress_thread([&]() {
//                 while (progress_active.load()) {
//                     int64_t current = completed_pages.load();
//                     double progress = (double)current / total_pages * 100.0;
//                     std::cout << "Page touching progress: " << std::fixed << std::setprecision(1) 
//                               << progress << "% (" << current << "/" << total_pages << " pages)" << std::endl;
//                     std::this_thread::sleep_for(std::chrono::seconds(2));
//                 }
//             });
            
//             // Worker function for each thread
//             auto touch_worker = [&](int thread_id) {
//                 int64_t start_offset = thread_id * chunk_size;
//                 int64_t end_offset = (thread_id == num_threads - 1) ? size : start_offset + chunk_size;
                
//                 volatile char* p = reinterpret_cast<volatile char*>(ptr);
//                 int64_t local_pages_touched = 0;
                
//                 // Touch pages in this thread's range
//                 for (int64_t offset = start_offset; offset < end_offset; offset += page_size) {
//                     p[offset] = 0;
//                     local_pages_touched++;
                    
//                     // Update global counter every 1000 pages to reduce contention
//                     if (local_pages_touched % 1000 == 0) {
//                         completed_pages.fetch_add(1000);
//                     }
//                 }
                
//                 // Add remaining pages to counter
//                 completed_pages.fetch_add(local_pages_touched % 1000);
                
//                 std::cout << "Thread " << thread_id << " completed " << local_pages_touched << " pages" << std::endl;
//             };
            
//             // Launch worker threads
//             for (int i = 0; i < num_threads; i++) {
//                 threads.emplace_back(touch_worker, i);
//             }
            
//             // Wait for all threads to complete
//             for (auto& t : threads) {
//                 t.join();
//             }
            
//             // Stop progress thread
//             progress_active.store(false);
//             progress_thread.join();
            
//             // Touch the last page if size is not page-aligned
//             if (size > 0) {
//                 volatile char* p = reinterpret_cast<volatile char*>(ptr);
//                 p[size - 1] = 0;
//             }
            
//             auto touch_end = std::chrono::high_resolution_clock::now();
//             auto touch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(touch_end - touch_start);
            
//             int64_t final_pages = completed_pages.load();
//             double throughput = (double)size / (1024.0 * 1024.0 * 1024.0) / (touch_duration.count() / 1000.0);
            
//             std::cout << "✓ Multi-threaded page touching completed:" << std::endl;
//             std::cout << "  - Pages touched: " << final_pages << std::endl;
//             std::cout << "  - Time: " << touch_duration.count() << " ms (" 
//                       << touch_duration.count() / 1000.0 << " seconds)" << std::endl;
//             std::cout << "  - Throughput: " << std::fixed << std::setprecision(2) 
//                       << throughput << " GB/s" << std::endl;
//         } else {
//             std::cout << "NUMA not available on this system." << std::endl;
//         }
//     }

//     // Register with CUDA - measure performance
//     std::cout << "Starting cudaHostRegister for " << size/(1024*1024*1024) << "GB..." << std::endl;
//     // Check if ptr is null and log the size
//     if (ptr == nullptr) {
//         std::cerr << "Error: ptr is null after mmap, size requested: "
//                     << size / (1024 * 1024 * 1024) << " GB" << std::endl;
//         throw std::runtime_error("mmap failed, ptr is null");
//     }
//     std::cout << "Allocated memory at: " << ptr << std::endl;
//     std::cout << "Size of allocated memory: " << size / (1024 * 1024 * 1024) 
//               << " GB" << std::endl;
//     auto cuda_start = std::chrono::high_resolution_clock::now();
    
//     cudaError_t err = cudaHostRegister(ptr, size, cudaHostRegisterDefault);
    
//     auto cuda_end = std::chrono::high_resolution_clock::now();
//     auto cuda_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cuda_end - cuda_start);
    
//     if (err != cudaSuccess) {
//         munmap(ptr, size);
//         // Clean up hugetlbfs file if we created it
//         if (using_huge_pages && create) {
//             unlink(hugepage_path.c_str());
//         }
//         throw std::runtime_error(
//             std::string("cudaHostRegister failed: ") + cudaGetErrorString(err));
//     }
    
//     std::cout << "✓ cudaHostRegister completed in " << cuda_duration.count() 
//               << " milliseconds (" << cuda_duration.count()/1000.0 << " seconds)" << std::endl;
    
//     // Verify huge page usage
//     if (create && using_huge_pages) {
//         std::cout << "\nChecking huge page consumption..." << std::endl;
//         system("cat /proc/meminfo | grep -i 'HugePages_Free\\|HugePages_Total'");
//     }

//     return ptr;
// }

// void* allocate_shared_pinned_memory(const std::string& shm_name,
//                                     int64_t size,
//                                     bool create) {
//     void* ptr = nullptr;
//     bool using_huge_pages = false;
    
//     // Method 1: Try hugetlbfs first (most reliable for huge pages)
//     std::string hugepage_path = "/dev/hugepages/" + shm_name;
//     int flags = O_RDWR | (create ? O_CREAT : 0);
//     int fd = open(hugepage_path.c_str(), flags, 0666);
    
//     if (fd >= 0) {
//         std::cout << "Trying hugetlbfs allocation..." << std::endl;
        
//         if (create) {
//             if (ftruncate64(fd, size) == -1) {
//                 std::cout << "ftruncate failed for hugetlbfs, trying fallback..." << std::endl;
//                 perror("ftruncate64 on hugetlbfs");
//                 close(fd);
//                 unlink(hugepage_path.c_str());
//                 goto fallback_to_shm;
//             }
//         }

//         // mmap the huge page file (no MAP_HUGETLB needed with hugetlbfs!)
//         ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
//         close(fd);
        
//         if (ptr != MAP_FAILED) {
//             std::cout << "✓ Successfully allocated " << size/(1024*1024*1024) 
//                       << "GB using hugetlbfs (2MB pages)!" << std::endl;
//             using_huge_pages = true;
//             goto success;
//         } else {
//             std::cout << "mmap failed on hugetlbfs, trying fallback..." << std::endl;
//             perror("mmap on hugetlbfs");
//             if (create) unlink(hugepage_path.c_str());
//         }
//     } else {
//         std::cout << "Cannot open hugetlbfs file (normal if not mounted), trying fallback..." << std::endl;
//     }

// fallback_to_shm:
//     // Method 2: Try shm_open with MAP_HUGETLB (will likely fail but worth trying)
//     std::cout << "Trying shm_open with MAP_HUGETLB..." << std::endl;
//     fd = shm_open(shm_name.c_str(), flags, 0666);
//     if (fd < 0) {
//         throw std::runtime_error("shm_open failed for " + shm_name);
//     }

//     if (create) {
//         if (ftruncate64(fd, size) == -1) {
//             close(fd);
//             throw std::runtime_error("ftruncate failed for " + shm_name);
//         }
//     }

//     // Try MAP_HUGETLB with shm_open (will probably fail)
//     ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, 
//                MAP_SHARED | MAP_HUGETLB | MAP_HUGE_2MB, fd, 0);
    
//     if (ptr != MAP_FAILED) {
//         std::cout << "✓ Successfully allocated " << size/(1024*1024*1024) 
//                   << "GB using shm_open + MAP_HUGETLB!" << std::endl;
//         using_huge_pages = true;
//         close(fd);
//         goto success;
//     } else {
//         std::cout << "shm_open + MAP_HUGETLB failed (expected), trying regular pages..." << std::endl;
//         perror("mmap with MAP_HUGETLB on shm_open");
        
//         // Method 3: Fallback to regular pages with huge page hints
//         ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
//         close(fd);
        
//         if (ptr == MAP_FAILED) {
//             throw std::runtime_error("All mmap methods failed for " + shm_name);
//         }
        
//         if (create) {
//             // Use madvise hint for transparent huge pages
//             int ret = madvise(ptr, size, MADV_HUGEPAGE);
//             if (ret == 0) {
//                 std::cout << "✓ Successfully hinted for transparent huge pages (may use 2MB pages)" << std::endl;
//             } else {
//                 std::cout << "madvise(MADV_HUGEPAGE) failed, using regular 4KB pages" << std::endl;
//                 perror("madvise");
//             }
//         }
//     }

// success:
//     if (create) {
//         if (numa_available() >= 0) {
//             int num_nodes = numa_num_configured_nodes();
//             std::cout << "NUMA available with " << num_nodes << " nodes." << std::endl;
            
//             if (num_nodes >= 2) {
//                 std::cout << "Interleaving memory across NUMA nodes 0 and 1." << std::endl;

//                 // Create proper nodemask using numa library functions
//                 struct bitmask* nodemask = numa_allocate_nodemask();
//                 if (!nodemask) {
//                     std::cerr << "Failed to allocate nodemask" << std::endl;
//                 } else {
//                     // Clear all bits first
//                     numa_bitmask_clearall(nodemask);
//                     // Set bits for nodes 0 and 1
//                     numa_bitmask_setbit(nodemask, 0);
//                     numa_bitmask_setbit(nodemask, 1);

//                     // Use set_mempolicy for the current process/thread
//                     int ret = set_mempolicy(MPOL_INTERLEAVE, 
//                                           nodemask->maskp, 
//                                           nodemask->size + 1);
//                     if (ret != 0) {
//                         perror("set_mempolicy(MPOL_INTERLEAVE)");
//                     } else {
//                         std::cout << "✓ NUMA memory policy set successfully" << std::endl;
//                     }

//                     numa_free_nodemask(nodemask);
//                 }
//             } else {
//                 std::cout << "Only " << num_nodes << " NUMA node(s) available, skipping interleaving." << std::endl;
//             }

//             // Touch pages to enforce actual allocation
//             std::cout << "Touching pages to enforce allocation..." << std::endl;
//             auto touch_start = std::chrono::high_resolution_clock::now();
            
//             // Use appropriate page size based on allocation method
//             long page_size = using_huge_pages ? (2 * 1024 * 1024) : sysconf(_SC_PAGESIZE);
//             std::cout << "Using page size: " << page_size / 1024 << " KB" << std::endl;
            
//             volatile char* p = reinterpret_cast<volatile char*>(ptr);
//             int64_t pages_touched = 0;
            
//             for (int64_t i = 0; i < size; i += page_size) {
//                 p[i] = 0;
//                 pages_touched++;
                
//                 // Progress indicator for large allocations
//                 if (pages_touched % 10000 == 0) {
//                     double progress = (double)i / size * 100.0;
//                     std::cout << "Page touching progress: " << std::fixed << std::setprecision(1) 
//                               << progress << "%" << std::endl;
//                 }
//             }
            
//             // Touch the last page if size is not page-aligned
//             if (size > 0) {
//                 p[size - 1] = 0;
//                 pages_touched++;
//             }
            
//             auto touch_end = std::chrono::high_resolution_clock::now();
//             auto touch_duration = std::chrono::duration_cast<std::chrono::seconds>(touch_end - touch_start);
            
//             std::cout << "✓ Touched " << pages_touched << " pages in " 
//                       << touch_duration.count() << " seconds" << std::endl;
//         } else {
//             std::cout << "NUMA not available on this system." << std::endl;
//         }
//     }

//     // Register with CUDA - measure performance
//     std::cout << "Starting cudaHostRegister for " << size/(1024*1024*1024) << "GB..." << std::endl;
//     auto cuda_start = std::chrono::high_resolution_clock::now();
    
//     cudaError_t err = cudaHostRegister(ptr, size, cudaHostRegisterDefault);
    
//     auto cuda_end = std::chrono::high_resolution_clock::now();
//     auto cuda_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cuda_end - cuda_start);
    
//     if (err != cudaSuccess) {
//         munmap(ptr, size);
//         // Clean up hugetlbfs file if we created it
//         if (using_huge_pages && create) {
//             unlink(hugepage_path.c_str());
//         }
//         throw std::runtime_error(
//             std::string("cudaHostRegister failed: ") + cudaGetErrorString(err));
//     }
    
//     std::cout << "✓ cudaHostRegister completed in " << cuda_duration.count() 
//               << " milliseconds (" << cuda_duration.count()/1000.0 << " seconds)" << std::endl;
    
//     // Verify huge page usage
//     if (create && using_huge_pages) {
//         std::cout << "\nChecking huge page consumption..." << std::endl;
//         system("cat /proc/meminfo | grep -i 'HugePages_Free\\|HugePages_Total'");
//     }

//     return ptr;
// }

// void* allocate_shared_pinned_memory(const std::string& shm_name,
//                                     int64_t size,
//                                     bool create) {
//     int flags = O_RDWR | (create ? O_CREAT : 0);
//     int fd = shm_open(shm_name.c_str(), flags, 0666);
//     if (fd < 0) {
//         throw std::runtime_error("shm_open failed for " + shm_name);
//     }

//     if (create) {
//         if (ftruncate64(fd, size) == -1) {
//             close(fd);
//             throw std::runtime_error("ftruncate failed for " + shm_name);
//         }
//     }

//     // void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
//     // void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, 
//     //     MAP_SHARED | MAP_HUGETLB | MAP_HUGE_2MB, fd, 0);   
//     // Try huge pages first
//     void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, 
//                      MAP_SHARED | MAP_HUGETLB | MAP_HUGE_2MB, fd, 0);
    
//     if (ptr == MAP_FAILED) {
//         std::cout << "Huge page allocation failed, trying regular pages..." << std::endl;
//         perror("mmap with huge pages");
        
//         // Fallback to regular pages
//         ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        
//         if (ptr != MAP_FAILED && create) {
//             // Use madvise hint for regular pages
//             int ret = madvise(ptr, size, MADV_HUGEPAGE);
//             if (ret == 0) {
//                 std::cout << "Successfully hinted for huge pages" << std::endl;
//             } else {
//                 perror("madvise(MADV_HUGEPAGE) failed, continuing with regular pages");
//             }
//         }
//     } else {
//         std::cout << "Successfully allocated " << size/(1024*1024*1024) 
//                   << "GB using 2MB huge pages!" << std::endl;
//     }
//     close(fd);
//     if (ptr == MAP_FAILED) {
//         throw std::runtime_error("mmap failed for " + shm_name);
//     }

//     if (create) {
//         if (numa_available() >= 0) {
//             int num_nodes = numa_num_configured_nodes();
//             std::cout << "NUMA available with " << num_nodes << " nodes." << std::endl;
            
//             if (num_nodes >= 2) {
//                 std::cout << "Interleaving memory across NUMA nodes 0 and 1." << std::endl;

//                 // Create proper nodemask using numa library functions
//                 struct bitmask* nodemask = numa_allocate_nodemask();
//                 if (!nodemask) {
//                     std::cerr << "Failed to allocate nodemask" << std::endl;
//                 } else {
//                     // Clear all bits first
//                     numa_bitmask_clearall(nodemask);
//                     // Set bits for nodes 0 and 1
//                     numa_bitmask_setbit(nodemask, 0);
//                     numa_bitmask_setbit(nodemask, 1);

//                     // Use set_mempolicy for the current process/thread
//                     int ret = set_mempolicy(MPOL_INTERLEAVE, 
//                                           nodemask->maskp, 
//                                           nodemask->size + 1);
//                     if (ret != 0) {
//                         perror("set_mempolicy(MPOL_INTERLEAVE)");
//                     }

//                     // Alternative: use mbind with correct parameters
//                     // int ret = mbind(ptr, size, MPOL_INTERLEAVE, 
//                     //                nodemask->maskp, nodemask->size + 1, 0);
//                     // if (ret != 0) {
//                     //     perror("mbind(MPOL_INTERLEAVE)");
//                     // }

//                     numa_free_nodemask(nodemask);
//                 }
//             } else {
//                 std::cout << "Only " << num_nodes << " NUMA node(s) available, skipping interleaving." << std::endl;
//             }

//             // Touch pages to enforce actual allocation
//             // Use proper page size and ensure we don't go out of bounds
//             // long page_size = sysconf(_SC_PAGESIZE);
//             long page_size = 2 * 1024 * 1024; // Use 2MB pages for huge pages
//             volatile char* p = reinterpret_cast<volatile char*>(ptr);
//             for (int64_t i = 0; i < size; i += page_size) {
//                 p[i] = 0;
//             }
            
//             // Touch the last page if size is not page-aligned
//             if (size > 0) {
//                 p[size - 1] = 0;
//             }
//         } else {
//             std::cout << "NUMA not available on this system." << std::endl;
//         }
//     }

//     cudaError_t err = cudaHostRegister(ptr, size, cudaHostRegisterDefault);
//     if (err != cudaSuccess) {
//         munmap(ptr, size);
//         throw std::runtime_error(
//             std::string("cudaHostRegister failed: ") +
//             cudaGetErrorString(err));
//     }

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

// void* allocate_shared_pinned_memory(const std::string& shm_name,
//                                     int64_t size,
//                                     bool create) {
//     int flags = O_RDWR | (create ? O_CREAT : 0);
//     int fd = shm_open(shm_name.c_str(), flags, 0666);
//     if (fd < 0) {
//         throw std::runtime_error("shm_open failed for " + shm_name);
//     }

//     if (create) {
//         if (ftruncate64(fd, size) == -1) {
//             close(fd);
//             throw std::runtime_error("ftruncate failed for " + shm_name);
//         }
//     }

//     void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
//     close(fd);
//     if (ptr == MAP_FAILED) {
//         throw std::runtime_error("mmap failed for " + shm_name);
//     }

//     if (create) {
//         if (numa_available() >= 0) {
//             std::cout << "Interleaving memory across NUMA nodes." << std::endl;

//             // Allocate nodemask and get maxnode
//             unsigned long nodemask = (1UL << 0) | (1UL << 1);
//             long maxnode = numa_num_configured_nodes();  // Correct maxnode

//             int ret = mbind(ptr, size,
//                             MPOL_INTERLEAVE,
//                             &nodemask,
//                             maxnode,  // Should be 2, not 64
//                             0);
//             if (ret != 0) {
//                 perror("mbind(MPOL_INTERLEAVE)");
//             }

//             // Touch pages to enforce actual allocation
//             volatile char* p = reinterpret_cast<volatile char*>(ptr);
//             for (int64_t i = 0; i < size; i += 4096) {
//                 p[i] = 0;
//             }
//         }
//     }

//     cudaError_t err = cudaHostRegister(ptr, size, cudaHostRegisterDefault);
//     if (err != cudaSuccess) {
//         munmap(ptr, size);
//         throw std::runtime_error(
//             std::string("cudaHostRegister failed: ") +
//             cudaGetErrorString(err));
//     }

//     return ptr;
// }

// void* allocate_shared_pinned_memory(const std::string& shm_name,
//                                     int64_t size,
//                                     bool create) {
//     int flags = O_RDWR | (create ? O_CREAT : 0);
//     int fd = shm_open(shm_name.c_str(), flags, 0666);
//     if (fd < 0) {
//         throw std::runtime_error("shm_open failed for " + shm_name);
//     }

//     if (create) {
//         if (ftruncate64(fd, size) == -1) {
//             close(fd);
//             throw std::runtime_error("ftruncate failed for " + shm_name);
//         }
//     }

//     void* ptr = mmap(nullptr, size,
//                      PROT_READ | PROT_WRITE,
//                      MAP_SHARED,
//                      fd, 0);
//     close(fd);
//     if (ptr == MAP_FAILED) {
//         throw std::runtime_error("mmap failed for " + shm_name);
//     }

// // #if defined(__linux__)
//     if(create){
//         // Log the number of NUMA nodes
//         int num_nodes = numa_num_configured_nodes();
//         std::cout << "Number of NUMA nodes: " << num_nodes << std::endl;
//         // If NUMA is available, interleave pages between node 0 and node 1
//         if (numa_available() >= 0) {
//             std::cout << "Interleaving memory across NUMA nodes." << std::endl;
//             unsigned long nodemask = (1UL << 0) | (1UL << 1);
//             int ret = mbind(ptr, size,
//                             MPOL_INTERLEAVE,
//                             &nodemask,
//                             /* maxnode = */ 8 * sizeof(nodemask),
//                             /* flags = */ 0);
//             if (ret != 0) {
//                 // non-fatal: we'll still fall back to default if interleave fails
//                 perror("mbind(MPOL_INTERLEAVE)");
//             }
//         }
//     }
// // #endif
    
//     cudaError_t err = cudaHostRegister(ptr, size, cudaHostRegisterDefault);
//     if (err != cudaSuccess) {
//         munmap(ptr, size);
//         throw std::runtime_error(
//             std::string("cudaHostRegister failed: ") +
//             cudaGetErrorString(err));
//     }

//     return ptr;
// }

// void* allocate_shared_pinned_memory(const std::string& shm_name, int64_t size,
//                                     bool create) {
//     int flags = O_RDWR | (create ? O_CREAT : 0);
//     // Open (or create) the shared memory object
//     int fd = shm_open(shm_name.c_str(), flags, 0666);
//     // int fd = memfd_create(shm_name.c_str(), 0);

//     // int fd = shm_open(shm_name.c_str(), flags, 0666);
//     if (fd < 0) {
//         throw std::runtime_error("shm_open failed for " + shm_name);
//     }

//     if (create) {
//         // Set the shared memory object size
//         if (ftruncate64(fd, size) == -1) {
//             close(fd);
//             throw std::runtime_error("ftruncate failed for " + shm_name);
//         }
//     }

//     // First map the shared memory into the process address space
//     // void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED |
//     // MAP_LOCKED, fd, 0);
//     void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
//     close(fd);  // fd no longer needed after mapping

//     if (ptr == MAP_FAILED) {
//         throw std::runtime_error("mmap failed for " + shm_name);
//     }

//     // Now register the mapped memory with CUDA to pin it
//     // std::cerr << "Registering shared memory with CUDA" << std::endl;
//     // std::cerr << "Size: " << size << std::endl;
//     // std::cerr << "Pointer: " << ptr << std::endl;
//     // int64_t block_size = 32LL * 1024 * 1024 * 1024;
//     // // Register the memory in blocks of 32GB
//     // for (int64_t i = 0; i < size; i += block_size) {
//     //     cudaError_t err = cudaHostRegister((char*)ptr + i,
//     //     std::min(block_size, size - i), cudaHostRegisterDefault); if (err !=
//     //     cudaSuccess) {
//     //         munmap(ptr, size);
//     //         throw std::runtime_error("cudaHostRegister failed: " +
//     //         std::string(cudaGetErrorString(err)));
//     //     }
//     // }

//     cudaError_t err = cudaHostRegister(ptr, size, cudaHostRegisterDefault);
//     if (err != cudaSuccess) {
//         munmap(ptr, size);
//         throw std::runtime_error("cudaHostRegister failed: " +
//                                  std::string(cudaGetErrorString(err)));
//     }

//     return ptr;
// }

// void* allocate_shared_pinned_memory(const std::string& shm_name, size_t size,
// bool create) {
//     // Compute the system page size.
//     size_t pageSize = static_cast<size_t>(sysconf(_SC_PAGESIZE));
//     // Round up the requested size to a multiple of the page size.
//     size_t aligned_size = ((size + pageSize - 1) / pageSize) * pageSize;

//     // Open (or create) the shared memory object.
//     int flags = O_RDWR | (create ? O_CREAT : 0);
//     int fd = shm_open(shm_name.c_str(), flags, 0666);
//     if (fd < 0) {
//         throw std::runtime_error("shm_open failed for " + shm_name + ": " +
//         std::strerror(errno));
//     }

//     if (create) {
//         // Set the shared memory object size to the aligned size.
//         if (ftruncate(fd, aligned_size) == -1) {
//             close(fd);
//             throw std::runtime_error("ftruncate failed for " + shm_name + ":
//             " + std::strerror(errno));
//         }
//     }

//     // Map the shared memory into the process address space.
//     // Note: mmap with nullptr ensures that the returned pointer is page
//     aligned. void* ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE,
//     MAP_SHARED | MAP_LOCKED, fd, 0); close(fd);  // The file descriptor is no
//     longer needed after mapping.

//     if (ptr == MAP_FAILED) {
//         throw std::runtime_error("mmap failed for " + shm_name + ": " +
//         std::strerror(errno));
//     }

//     // Register the mapped memory with CUDA.
//     // Registering the entire aligned region ensures that any pointer within
//     it (even if offset)
//     // is part of a pinned, contiguous memory block.
//     if (create) {
//         // Only register the memory in the creating process
//         // Use cudaHostRegisterPortable so all CUDA contexts can access it
//         // Add cudaHostRegisterMapped if you need to get device pointers
//         cudaError_t err = cudaHostRegister(ptr, aligned_size,
//                                          cudaHostRegisterPortable |
//                                          cudaHostRegisterMapped);
//         if (err != cudaSuccess) {
//             munmap(ptr, aligned_size);
//             throw std::runtime_error("cudaHostRegister failed: " +
//                                    std::string(cudaGetErrorString(err)));
//         }
//     }

//     // cudaError_t err = cudaHostRegister(ptr, aligned_size,
//     cudaHostRegisterDefault);
//     // if (err != cudaSuccess) {
//     //     munmap(ptr, aligned_size);
//     //     throw std::runtime_error("cudaHostRegister failed: " +
//     std::string(cudaGetErrorString(err)));
//     // }

//     return ptr;
// }

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
