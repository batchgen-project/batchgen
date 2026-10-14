// clang-format off
/* ----------------------------------------------------------------------------  *
 *  BatchGen                                                                      *
 *  Copyright (c) 2025-2026 BatchGen Team                                            *
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
#include <cstdlib>
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
#include <linux/magic.h>
#include <linux/memfd.h>
#include <linux/mman.h>
#include "../numa_compat.h"
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/vfs.h>
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
#ifndef MFD_HUGETLB
#define MFD_HUGETLB 0x0004U
#endif
#ifndef HUGETLBFS_MAGIC
#define HUGETLBFS_MAGIC 0x958458f6
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

// Anonymous memfd: the region carries no /dev/shm name, so the kernel reclaims
// it as soon as the last process mapping it is gone, however that process died.
// MFD_CLOEXEC keeps an exec'd child from holding it alive; attachers reach it
// through /proc/<creator_pid>/fd/<N>, not by inheritance.
static int create_anonymous_memfd(const char* name) {
    return static_cast<int>(syscall(SYS_memfd_create, name, MFD_CLOEXEC));
}

// Same unnamed object, but backed by the host's preallocated huge page pool
// instead of base pages. No hugetlbfs mount and no path are involved, so the
// kernel returns the pages to the pool with the last mapping.
static int create_huge_anonymous_memfd(const char* name) {
    return static_cast<int>(
        syscall(SYS_memfd_create, name, MFD_CLOEXEC | MFD_HUGETLB));
}

// MFD_HUGETLB without a size flag uses the host's default huge page size, so
// the region must be rounded and mapped to exactly that size.
static size_t default_huge_page_size() {
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.find("Hugepagesize:") == 0) {
            std::istringstream iss(line);
            std::string key, value;
            iss >> key >> value;
            return static_cast<size_t>(std::stol(value)) * 1024;
        }
    }
    logger->warn(
        "Hugepagesize missing from /proc/meminfo; assuming 2MB huge pages");
    return 2 * 1024 * 1024;
}

// --fast-init requests transparent huge pages explicitly. Without it the
// mapping is not advised at all, so its page size follows the host's
// /sys/kernel/mm/transparent_hugepage/shmem_enabled policy. A failed hint is
// not fatal.
static void advise_transparent_huge_pages(void* ptr, int64_t size,
                                          bool enable_thp) {
    if (!enable_thp) {
        return;
    }
    if (madvise(ptr, size, MADV_HUGEPAGE) != 0) {
        logger->warn(
            "madvise(MADV_HUGEPAGE) failed: {}; weights page size follows the "
            "system THP setting",
            strerror(errno));
    }
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
 * * This function supports two allocation strategies, and neither gives the
 * region a name: the kernel reclaims it with the last mapping, however the
 * owners died.
 * 1.  Huge page memfd: MFD_HUGETLB takes the region from the host's huge page
 * pool, which needs that pool reserved. Controlled by enable_hugetlbfs.
 * 2.  Base page memfd: the only other strategy, and the fallback when the huge
 * page pool cannot satisfy the request.
 * * In both cases, the allocated memory is "touched" to ensure it is resident in RAM.
 * If pin_for_cuda is true, memory is also registered with cudaHostRegister for DMA.
 * * @param shm_name A label for logging; no filesystem object carries it.
 * @param size The desired size of the allocation in bytes.
 * @param create True if the caller is the server (creates the segment), false for workers.
 * @param enable_hugetlbfs If true, attempts a huge page memfd for allocation.
 * @param pin_for_cuda If true, register memory with cudaHostRegister for GPU DMA access.
 *                     Server should pass false (no GPU access needed), workers pass true.
 * @param enable_thp If true, hint transparent huge pages and prefault them.
 * @param memfd_creator_pid Creator of the memfd a worker attaches to.
 * @param memfd_fd_arg The creator's fd number for that memfd.
 * @return A void pointer to the allocated shared memory.
 * @throws std::runtime_error on failure.
 */
void* allocate_shared_pinned_memory(const std::string& shm_name,
                                    int64_t size,
                                    bool create,
                                    bool enable_hugetlbfs,
                                    bool pin_for_cuda,
                                    bool enable_thp,
                                    int memfd_creator_pid,
                                    int memfd_fd_arg,
                                    int* out_memfd_fd,
                                    int64_t* out_mapped_size) {
    if (size <= 0) {
        throw std::runtime_error("Invalid allocation size: " + std::to_string(size));
    }
    if (create && out_memfd_fd == nullptr) {
        throw std::runtime_error("memfd creator requires an output fd");
    }
    if (out_mapped_size) *out_mapped_size = 0;

    const size_t page_size = sysconf(_SC_PAGESIZE);
    const size_t huge_page_size = 2 * 1024 * 1024; // 2MB

    logger->info("Allocating shared memory: name={}, size={}MB, mode={}",
                 shm_name, size / (1024 * 1024),
                 create ? "server" : "worker");

    void* ptr = nullptr;
    bool using_huge_pages = false;
    int64_t allocated_size = 0;

    // STAGE 1: Attempt a huge page memfd if enabled. Only the creator runs
    // this: an attacher cannot tell a huge page memfd from a base page one by
    // name, because neither has a name, so it always takes the attach path in
    // STAGE 2 and discovers the page size from the fd itself.
    if (enable_hugetlbfs && create) {
        const size_t hugetlb_page_size = default_huge_page_size();
        const int64_t hugetlb_size =
            ((size + hugetlb_page_size - 1) / hugetlb_page_size) *
            hugetlb_page_size;
        logger->info("Attempting hugepage allocation ({}MB pages)...",
                     hugetlb_page_size / (1024 * 1024));

        int fd = create_huge_anonymous_memfd("batchgen_weights");
        if (fd < 0) {
            logger->warn("memfd_create(MFD_HUGETLB) failed: {}",
                         strerror(errno));
        } else if (ftruncate64(fd, hugetlb_size) != 0) {
            logger->warn("hugepage memfd ftruncate failed: {}",
                         strerror(errno));
            close(fd);
        } else {
            size_t alignment = std::max(hugetlb_page_size, page_size);
            ptr = mmap_aligned(hugetlb_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                               fd, 0, alignment);
            if (ptr == MAP_FAILED) {
                // The pool is reserved by the host, so this is where an
                // insufficient reservation surfaces.
                logger->warn("hugepage memfd mmap failed: {}", strerror(errno));
                ptr = nullptr;
                close(fd);
            } else if (!touch_pages(ptr, size, hugetlb_page_size, true)) {
                logger->error("Multi-threaded hugepage touch failed. Aborting hugepage allocation.");
                munmap(ptr, hugetlb_size);
                ptr = nullptr;
                close(fd);
            } else {
                allocated_size = hugetlb_size;
                using_huge_pages = true;
                *out_memfd_fd = fd;
                logger->info(
                    "Weights allocated via memfd_create(MFD_HUGETLB) "
                    "({:.1f} GB, {}MB pages)",
                    allocated_size / (1024.0 * 1024.0 * 1024.0),
                    hugetlb_page_size / (1024 * 1024));
            }
        }
    }

    // STAGE 2: Base page memfd. This is the only path when hugetlbfs is not
    // requested, and the fallback when it was requested but the huge page pool
    // could not satisfy it. It is also the attach path in every mode, since an
    // attacher reaches the creator's fd through /proc either way.
    if (!ptr) {
        if (enable_hugetlbfs && create) {
             logger->info("Falling back to a base page memfd allocation...");
        }

        size_t alignment = std::max(huge_page_size, page_size);
        int64_t aligned_size = ((size + huge_page_size - 1) / huge_page_size) * huge_page_size;

        if (create) {
            int fd = create_anonymous_memfd("batchgen_weights");
            if (fd < 0) {
                throw std::runtime_error(
                    "memfd_create for weights failed: " +
                    std::string(strerror(errno)));
            }
            if (ftruncate64(fd, aligned_size) != 0) {
                int err = errno;
                close(fd);
                throw std::runtime_error(
                    "ftruncate on weights memfd failed: " +
                    std::string(strerror(err)));
            }
            ptr = mmap_aligned(aligned_size, PROT_READ | PROT_WRITE,
                               MAP_SHARED, fd, 0, alignment);
            if (ptr == MAP_FAILED) {
                int err = errno;
                close(fd);
                ptr = nullptr;
                throw std::runtime_error(
                    "mmap on weights memfd failed: " +
                    std::string(strerror(err)));
            }
            allocated_size = aligned_size;
            advise_transparent_huge_pages(ptr, allocated_size, enable_thp);
            if (enable_thp) {
                // Pre-fault THP pages before weight loading to avoid
                // non-deterministic compaction stalls during direct I/O.
                if (const char* skip_touch =
                        std::getenv("BATCHGEN_FAST_INIT_SKIP_WEIGHT_TOUCH");
                    skip_touch != nullptr && std::strcmp(skip_touch, "1") == 0) {
                    logger->warn(
                        "--fast-init: Skipping weights page touching because "
                        "BATCHGEN_FAST_INIT_SKIP_WEIGHT_TOUCH=1");
                } else {
                    auto touch_start = std::chrono::high_resolution_clock::now();
                    bool ok = touch_pages(ptr, allocated_size, huge_page_size, /*multi_threaded=*/true);
                    auto touch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::high_resolution_clock::now() - touch_start).count();
                    if (ok) {
                        logger->info("--fast-init: Weights page touching completed in {:.2f}s ({:.1f} GB)",
                                     touch_ms / 1000.0, allocated_size / (1024.0 * 1024.0 * 1024.0));
                    } else {
                        logger->warn("--fast-init: Weights page touching failed after {:.2f}s, proceeding anyway",
                                     touch_ms / 1000.0);
                    }
                }
            } else if (!touch_pages(ptr, size, page_size, true)) {
                logger->error("Multi-threaded regular page touch failed - memory may not be fully resident.");
            }
            *out_memfd_fd = fd;
            logger->info("Weights allocated via memfd_create ({:.1f} GB, {} pages)",
                         allocated_size / (1024.0 * 1024.0 * 1024.0),
                         enable_thp ? "transparent huge" : "base");
        } else {
            // Worker: open the creator's memfd via /proc/<pid>/fd/<N>.
            if (memfd_creator_pid <= 0 || memfd_fd_arg < 0) {
                throw std::runtime_error(
                    "weights attach requires a valid memfd_creator_pid and "
                    "memfd_fd (got pid=" + std::to_string(memfd_creator_pid) +
                    ", fd=" + std::to_string(memfd_fd_arg) + ")");
            }
            std::string proc_path = "/proc/" + std::to_string(memfd_creator_pid) +
                                    "/fd/" + std::to_string(memfd_fd_arg);
            int fd = open(proc_path.c_str(), O_RDWR | O_CLOEXEC);
            if (fd < 0) {
                throw std::runtime_error(
                    "weights attach cannot open " + proc_path + ": " +
                    strerror(errno));
            }
            struct stat sb;
            if (fstat(fd, &sb) == -1 || sb.st_size < aligned_size) {
                close(fd);
                throw std::runtime_error(
                    "weights memfd too small or fstat failed");
            }
            // A huge page memfd only accepts a mapping aligned to its own page
            // size, which the base page alignment above does not guarantee.
            struct statfs sfs;
            const bool fd_is_hugetlb =
                fstatfs(fd, &sfs) == 0 &&
                static_cast<unsigned long>(sfs.f_type) == HUGETLBFS_MAGIC;
            if (fd_is_hugetlb) {
                alignment = std::max(default_huge_page_size(), page_size);
            }
            ptr = mmap_aligned(sb.st_size, PROT_READ | PROT_WRITE,
                               MAP_SHARED, fd, 0, alignment);
            if (ptr == MAP_FAILED) {
                int err = errno;
                close(fd);
                ptr = nullptr;
                throw std::runtime_error(
                    "mmap on weights memfd failed: " +
                    std::string(strerror(err)));
            }
            allocated_size = sb.st_size;
            using_huge_pages = fd_is_hugetlb;
            // Both sides advise, so one cannot promote the mapping while the
            // other keeps it at the base page size. Huge pages are already the
            // mapping's page size, and THP cannot apply to them.
            advise_transparent_huge_pages(ptr, allocated_size,
                                          enable_thp && !fd_is_hugetlb);
            close(fd);
            logger->info("Weights attached via memfd ({:.1f} GB, {} pages)",
                         allocated_size / (1024.0 * 1024.0 * 1024.0),
                         fd_is_hugetlb ? "huge" : "base");
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
            // Clean up memory and rethrow if CUDA registration fails. Closing
            // the creator's fd is the whole release in both modes.
            munmap(ptr, allocated_size);
            if (create && *out_memfd_fd >= 0) {
                close(*out_memfd_fd);
                *out_memfd_fd = -1;
            }
            throw;
        }
    } else {
        logger->info("Skipping CUDA registration (server mode, no GPU access needed)");
    }

    logger->info("Memory allocation completed successfully using {} pages.",
               using_huge_pages ? "huge" : "regular");

    if (out_mapped_size) *out_mapped_size = allocated_size;

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


void free_shared_pinned_memory(void* ptr, int64_t size) {
    const auto start = std::chrono::steady_clock::now();
    const cudaError_t unregister_result = cudaHostUnregister(ptr);
    const auto unregister_done = std::chrono::steady_clock::now();
    const int unmap_result = munmap(ptr, size);
    const auto unmap_done = std::chrono::steady_clock::now();
    logger->info(
        "shared memory release: cudaHostUnregister={} elapsed={:.3f}s, "
        "munmap={} elapsed={:.3f}s",
        static_cast<int>(unregister_result),
        std::chrono::duration<double>(unregister_done - start).count(),
        unmap_result,
        std::chrono::duration<double>(unmap_done - unregister_done).count());
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
// API: Serialize the map into an anonymous memfd.
// This function computes the required size, creates and sizes the memfd, maps
// it, writes the serialized data, then unmaps it. The returned fd stays open so
// workers can reach the region through /proc/<creator_pid>/fd/<N>.
// -----------------------------------------------------------------------------
int serialize_to_memfd(
    const std::unordered_map<std::string,
                             std::unordered_map<std::string, tensor_meta>>& map) {
    // Compute the buffer size required.
    size_t total_size = compute_serialized_size(map);

    int fd = create_anonymous_memfd("batchgen_tensor_meta");
    if (fd == -1)
        throw std::runtime_error(
            "memfd_create for tensor metadata failed: " +
            std::string(strerror(errno)));

    // Set the size.
    if (ftruncate(fd, total_size) == -1) {
        int err = errno;
        close(fd);
        throw std::runtime_error("ftruncate on tensor metadata memfd failed: " +
                                 std::string(strerror(err)));
    }

    // Map the memory.
    void* addr =
        mmap(nullptr, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        int err = errno;
        close(fd);
        throw std::runtime_error("mmap on tensor metadata memfd failed: " +
                                 std::string(strerror(err)));
    }

    // Write the data.
    try {
        serialize_map_to_buffer(map, static_cast<char*>(addr), total_size);
    } catch (...) {
        munmap(addr, total_size);
        close(fd);
        throw;
    }

    // Clean up; the fd is the caller's to keep and close.
    munmap(addr, total_size);
    return fd;
}

// -----------------------------------------------------------------------------
// API: Deserialize the map from the creator's anonymous memfd.
// This function opens /proc/<creator_pid>/fd/<N>, maps it, reads the data, then
// unmaps/closes it. The region is small and read exactly once per worker.
// -----------------------------------------------------------------------------
std::unordered_map<std::string, std::unordered_map<std::string, tensor_meta>>
deserialize_from_memfd(int memfd_creator_pid, int memfd_fd) {
    if (memfd_creator_pid <= 0 || memfd_fd < 0)
        throw std::runtime_error(
            "tensor metadata attach requires a valid memfd_creator_pid and "
            "memfd_fd (got pid=" + std::to_string(memfd_creator_pid) +
            ", fd=" + std::to_string(memfd_fd) + ")");

    const std::string proc_path = "/proc/" + std::to_string(memfd_creator_pid) +
                                  "/fd/" + std::to_string(memfd_fd);
    int fd = open(proc_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd == -1)
        throw std::runtime_error("Cannot open tensor metadata memfd " +
                                 proc_path + ": " + strerror(errno));

    // Get the size.
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        throw std::runtime_error("Failed to get tensor metadata memfd size");
    }
    size_t size = sb.st_size;

    // Map the memory.
    void* addr = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("Failed to map tensor metadata memfd");
    }

    // Deserialize the map.
    auto result =
        deserialize_map_from_buffer(static_cast<const char*>(addr), size);

    // Clean up.
    munmap(addr, size);
    close(fd);
    return result;
}
