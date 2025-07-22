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

#include <cstring>
#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <linux/memfd.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>
#include "../utils.h"

#include "posix_shm.h"
#include <numa.h>
#include <numaif.h>
#include <linux/mman.h>  // For MAP_HUGE_2MB

// Fallback calculation if MAP_HUGE_2MB is not directly available
#ifndef MAP_HUGE_2MB
    #define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#endif
void* allocate_shared_pinned_memory(const std::string& shm_name, int64_t size,
                                    bool create) {
    void* ptr = nullptr;
    bool using_huge_pages = false;

    // Validate input parameters
    if (size <= 0) {
        throw std::runtime_error("Invalid size: " + std::to_string(size));
    }

    // CUDA alignment requirements: memory must be aligned to page boundaries
    // For best performance, align to 64KB (or larger for huge pages)
    const size_t cuda_alignment =
        using_huge_pages ? (2 * 1024 * 1024)
                         : (64 * 1024);  // 2MB for huge pages, 64KB for regular
    const size_t page_size = sysconf(_SC_PAGESIZE);

    // Round up size to page boundary to ensure proper alignment
    int64_t aligned_size = ((size + page_size - 1) / page_size) * page_size;

    std::cout << "Original size: " << size / (1024 * 1024) << "MB, "
              << "Aligned size: " << aligned_size / (1024 * 1024) << "MB"
              << std::endl;

    // Method 1: Try hugetlbfs first (most reliable for huge pages)
    std::string hugepage_path = "/dev/hugepages/" + shm_name;
    int flags = O_RDWR | (create ? O_CREAT : 0);
    int fd = open(hugepage_path.c_str(), flags, 0666);

    if (fd >= 0) {
        std::cout << "Trying hugetlbfs allocation..." << std::endl;

        if (create) {
            if (ftruncate64(fd, aligned_size) == -1) {
                std::cout
                    << "ftruncate failed for hugetlbfs, trying fallback..."
                    << std::endl;
                perror("ftruncate64 on hugetlbfs");
                close(fd);
                unlink(hugepage_path.c_str());
                goto fallback_to_shm;
            }
        }

        // mmap the huge page file (no MAP_HUGETLB needed with hugetlbfs!)
        ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                   fd, 0);
        close(fd);

        if (ptr != MAP_FAILED) {
            std::cout << "✓ Successfully allocated "
                      << aligned_size / (1024 * 1024 * 1024)
                      << "GB using hugetlbfs (2MB pages)!" << std::endl;
            using_huge_pages = true;

            // Verify alignment for huge pages (should be 2MB aligned)
            if ((uintptr_t)ptr % (2 * 1024 * 1024) != 0) {
                std::cout << "Warning: Huge page memory not 2MB aligned: "
                          << ptr << std::endl;
            } else {
                std::cout << "✓ Memory is properly 2MB aligned: " << ptr
                          << std::endl;
            }
            goto success;
        } else {
            std::cout << "mmap failed on hugetlbfs, trying fallback..."
                      << std::endl;
            perror("mmap on hugetlbfs");
            if (create) unlink(hugepage_path.c_str());
        }
    } else {
        std::cout << "Cannot open hugetlbfs file (normal if not mounted), "
                     "trying fallback..."
                  << std::endl;
    }

fallback_to_shm:
    // Method 2: Try shm_open with MAP_HUGETLB (will likely fail but worth
    // trying)
    std::cout << "Trying shm_open with MAP_HUGETLB..." << std::endl;
    fd = shm_open(shm_name.c_str(), flags, 0666);
    if (fd < 0) {
        throw std::runtime_error("shm_open failed for " + shm_name);
    }

    if (create) {
        if (ftruncate64(fd, aligned_size) == -1) {
            close(fd);
            throw std::runtime_error("ftruncate failed for " + shm_name);
        }
    }

    // Try MAP_HUGETLB with shm_open (will probably fail)
    ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE,
               MAP_SHARED | MAP_HUGETLB | MAP_HUGE_2MB, fd, 0);

    if (ptr != MAP_FAILED) {
        std::cout << "✓ Successfully allocated "
                  << aligned_size / (1024 * 1024 * 1024)
                  << "GB using shm_open + MAP_HUGETLB!" << std::endl;
        using_huge_pages = true;
        close(fd);

        // Verify 2MB alignment
        if ((uintptr_t)ptr % (2 * 1024 * 1024) != 0) {
            std::cout << "Warning: Huge page memory not 2MB aligned: " << ptr
                      << std::endl;
        } else {
            std::cout << "✓ Memory is properly 2MB aligned: " << ptr
                      << std::endl;
        }
        goto success;
    } else {
        std::cout << "shm_open + MAP_HUGETLB failed (expected), trying regular "
                     "pages..."
                  << std::endl;
        perror("mmap with MAP_HUGETLB on shm_open");

        // Method 3: Fallback to regular pages - allocate extra space for manual
        // alignment
        size_t extra_size = aligned_size + cuda_alignment;
        ptr = mmap(nullptr, extra_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                   0);
        close(fd);

        if (ptr == MAP_FAILED) {
            throw std::runtime_error("All mmap methods failed for " + shm_name);
        }

        // Manually align the pointer to cuda_alignment boundary
        uintptr_t aligned_addr =
            ((uintptr_t)ptr + cuda_alignment - 1) & ~(cuda_alignment - 1);
        void* aligned_ptr = (void*)aligned_addr;

        // Check if we have enough space after alignment
        size_t offset = aligned_addr - (uintptr_t)ptr;
        if (offset + aligned_size > extra_size) {
            munmap(ptr, extra_size);
            throw std::runtime_error(
                "Cannot align memory within allocated space");
        }

        std::cout << "✓ Memory aligned from " << ptr << " to " << aligned_ptr
                  << " (offset: " << offset << " bytes)" << std::endl;

        // We need to keep track of the original ptr for munmap later
        // For simplicity, let's try a different approach: use posix_memalign
        // instead
        munmap(ptr, extra_size);

        // Use a simpler approach with proper alignment from the start
        void* temp_ptr = nullptr;
        if (posix_memalign(&temp_ptr, cuda_alignment, aligned_size) != 0) {
            throw std::runtime_error("posix_memalign failed for alignment");
        }

        // Copy to shared memory with proper alignment
        ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE,
                   MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED) {
            free(temp_ptr);
            throw std::runtime_error("mmap for aligned memory failed");
        }

        // Verify alignment
        if ((uintptr_t)ptr % cuda_alignment != 0) {
            std::cout << "Warning: Regular memory not " << cuda_alignment / 1024
                      << "KB aligned" << std::endl;
            // If still not aligned, we need to handle this case
            munmap(ptr, aligned_size);

            // Last resort: use the properly aligned temp_ptr and convert to
            // shared
            ptr = temp_ptr;
            std::cout << "Using posix_memaligned memory: " << ptr << std::endl;
        } else {
            std::cout << "✓ Memory is properly " << cuda_alignment / 1024
                      << "KB aligned: " << ptr << std::endl;
            free(temp_ptr);
        }

        if (create) {
            // Use madvise hint for transparent huge pages
            int ret = madvise(ptr, aligned_size, MADV_HUGEPAGE);
            if (ret == 0) {
                std::cout << "✓ Successfully hinted for transparent huge pages "
                             "(may use 2MB pages)"
                          << std::endl;
            } else {
                std::cout
                    << "madvise(MADV_HUGEPAGE) failed, using regular 4KB pages"
                    << std::endl;
                perror("madvise");
            }
        }
    }

success:
    // Validate pointer and alignment before proceeding
    if (ptr == nullptr) {
        throw std::runtime_error("Memory allocation returned null pointer");
    }

    // Check final alignment
    size_t final_alignment =
        using_huge_pages ? (2 * 1024 * 1024) : cuda_alignment;
    if ((uintptr_t)ptr % final_alignment != 0) {
        std::cout << "ERROR: Final memory alignment check failed!" << std::endl;
        std::cout << "Pointer: " << ptr
                  << ", Required alignment: " << final_alignment << std::endl;
        std::cout << "Pointer modulo: " << ((uintptr_t)ptr % final_alignment)
                  << std::endl;

        // For CUDA, we need at least page alignment, so let's check that
        if ((uintptr_t)ptr % page_size != 0) {
            munmap(ptr, aligned_size);
            throw std::runtime_error(
                "Memory is not page-aligned, cudaHostRegister will fail");
        } else {
            std::cout << "Memory is at least page-aligned, proceeding..."
                      << std::endl;
        }
    }

    if (create) {
        if (numa_available() >= 0) {
            int num_nodes = numa_num_configured_nodes();
            std::cout << "NUMA available with " << num_nodes << " nodes."
                      << std::endl;

            if (num_nodes >= 2) {
                std::cout << "Interleaving memory across NUMA nodes 0 and 1."
                          << std::endl;

                // Create proper nodemask using numa library functions
                struct bitmask* nodemask = numa_allocate_nodemask();
                if (!nodemask) {
                    std::cerr << "Failed to allocate nodemask" << std::endl;
                } else {
                    // Clear all bits first
                    numa_bitmask_clearall(nodemask);
                    // Set bits for nodes 0 and 1
                    numa_bitmask_setbit(nodemask, 0);
                    numa_bitmask_setbit(nodemask, 1);

                    // Use set_mempolicy for the current process/thread
                    int ret = set_mempolicy(MPOL_INTERLEAVE, nodemask->maskp,
                                            nodemask->size + 1);
                    if (ret != 0) {
                        perror("set_mempolicy(MPOL_INTERLEAVE)");
                    } else {
                        std::cout << "✓ NUMA memory policy set successfully"
                                  << std::endl;
                    }

                    numa_free_nodemask(nodemask);
                }
            } else {
                std::cout << "Only " << num_nodes
                          << " NUMA node(s) available, skipping interleaving."
                          << std::endl;
            }

            // Multi-threaded page touching
            std::cout << "Starting multi-threaded page touching..."
                      << std::endl;
            auto touch_start = std::chrono::high_resolution_clock::now();

            // Use appropriate page size based on allocation method
            long touch_page_size =
                using_huge_pages ? (2 * 1024 * 1024) : sysconf(_SC_PAGESIZE);
            std::cout << "Using page size: " << touch_page_size / 1024 << " KB"
                      << std::endl;

            // Determine optimal number of threads
            int num_threads = std::min(
                16, std::max(2, (int)std::thread::hardware_concurrency() / 2));
            std::cout << "Using " << num_threads << " threads for page touching"
                      << std::endl;

            // Calculate work distribution - use original size for touching, not
            // aligned_size
            int64_t total_pages =
                (size + touch_page_size - 1) / touch_page_size;
            int64_t chunk_size = size / num_threads;

            std::vector<std::thread> threads;
            std::atomic<int64_t> completed_pages{0};
            std::atomic<bool> progress_active{true};

            // Progress reporting thread
            std::thread progress_thread([&]() {
                while (progress_active.load()) {
                    int64_t current = completed_pages.load();
                    double progress = (double)current / total_pages * 100.0;
                    std::cout << "Page touching progress: " << std::fixed
                              << std::setprecision(1) << progress << "% ("
                              << current << "/" << total_pages << " pages)"
                              << std::endl;
                    std::this_thread::sleep_for(std::chrono::seconds(2));
                }
            });

            // Worker function for each thread
            auto touch_worker = [&](int thread_id) {
                int64_t start_offset = thread_id * chunk_size;
                int64_t end_offset = (thread_id == num_threads - 1)
                                         ? size
                                         : start_offset + chunk_size;

                volatile char* p = reinterpret_cast<volatile char*>(ptr);
                int64_t local_pages_touched = 0;

                // Touch pages in this thread's range
                for (int64_t offset = start_offset; offset < end_offset;
                     offset += touch_page_size) {
                    p[offset] = 0;
                    local_pages_touched++;

                    // Update global counter every 1000 pages to reduce
                    // contention
                    if (local_pages_touched % 1000 == 0) {
                        completed_pages.fetch_add(1000);
                    }
                }

                // Add remaining pages to counter
                completed_pages.fetch_add(local_pages_touched % 1000);

                // std::cout << "Thread " << thread_id << " completed " <<
                // local_pages_touched << " pages" << std::endl;
            };

            // Launch worker threads
            for (int i = 0; i < num_threads; i++) {
                threads.emplace_back(touch_worker, i);
            }

            // Wait for all threads to complete
            for (auto& t : threads) {
                t.join();
            }

            // Stop progress thread
            progress_active.store(false);
            progress_thread.join();

            // Touch the last page if size is not page-aligned
            if (size > 0) {
                volatile char* p = reinterpret_cast<volatile char*>(ptr);
                p[size - 1] = 0;
            }

            auto touch_end = std::chrono::high_resolution_clock::now();
            auto touch_duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    touch_end - touch_start);

            int64_t final_pages = completed_pages.load();
            double throughput = (double)size / (1024.0 * 1024.0 * 1024.0) /
                                (touch_duration.count() / 1000.0);

            std::cout << "✓ Multi-threaded page touching completed:"
                      << std::endl;
            std::cout << "  - Pages touched: " << final_pages << std::endl;
            std::cout << "  - Time: " << touch_duration.count() << " ms ("
                      << touch_duration.count() / 1000.0 << " seconds)"
                      << std::endl;
            std::cout << "  - Throughput: " << std::fixed
                      << std::setprecision(2) << throughput << " GB/s"
                      << std::endl;
        } else {
            std::cout << "NUMA not available on this system." << std::endl;
        }
    }

    // Additional validation before CUDA registration
    std::cout << "Pre-CUDA validation:" << std::endl;
    std::cout << "  - Pointer: " << ptr << std::endl;
    std::cout << "  - Size: " << size << " bytes ("
              << size / (1024 * 1024 * 1024) << " GB)" << std::endl;
    std::cout << "  - Aligned size: " << aligned_size << " bytes" << std::endl;
    std::cout << "  - Page alignment: "
              << ((uintptr_t)ptr % page_size == 0 ? "✓" : "✗") << std::endl;

    // Check if the memory range is valid by trying to read/write
    try {
        volatile char* test_ptr = reinterpret_cast<volatile char*>(ptr);
        char original = test_ptr[0];
        test_ptr[0] = 0x42;
        if (test_ptr[0] != 0x42) {
            throw std::runtime_error("Memory test failed - cannot write");
        }
        test_ptr[0] = original;

        // Test last byte
        original = test_ptr[size - 1];
        test_ptr[size - 1] = 0x43;
        if (test_ptr[size - 1] != 0x43) {
            throw std::runtime_error(
                "Memory test failed - cannot write to end");
        }
        test_ptr[size - 1] = original;

        std::cout << "  - Memory accessibility: ✓" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  - Memory accessibility: ✗ (" << e.what() << ")"
                  << std::endl;
        munmap(ptr, aligned_size);
        throw std::runtime_error("Memory accessibility test failed: " +
                                 std::string(e.what()));
    }

    // Register with CUDA - use original size, not aligned_size
    std::cout << "Starting cudaHostRegister for " << size / (1024 * 1024 * 1024)
              << "GB..." << std::endl;
    auto cuda_start = std::chrono::high_resolution_clock::now();

    cudaError_t err = cudaHostRegister(ptr, size, cudaHostRegisterDefault);

    auto cuda_end = std::chrono::high_resolution_clock::now();
    auto cuda_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        cuda_end - cuda_start);

    if (err != cudaSuccess) {
        std::cout << "cudaHostRegister failed with error: "
                  << cudaGetErrorString(err) << std::endl;
        std::cout << "Error details:" << std::endl;
        std::cout << "  - Pointer: " << ptr << std::endl;
        std::cout << "  - Size: " << size << std::endl;
        std::cout << "  - Alignment: " << ((uintptr_t)ptr % page_size)
                  << std::endl;

        munmap(ptr, aligned_size);
        // Clean up hugetlbfs file if we created it
        if (using_huge_pages && create) {
            unlink(hugepage_path.c_str());
        }
        throw std::runtime_error(std::string("cudaHostRegister failed: ") +
                                 cudaGetErrorString(err));
    }

    std::cout << "✓ cudaHostRegister completed in " << cuda_duration.count()
              << " milliseconds (" << cuda_duration.count() / 1000.0
              << " seconds)" << std::endl;

    // Verify huge page usage
    if (create && using_huge_pages) {
        std::cout << "\nChecking huge page consumption..." << std::endl;
        system(
            "cat /proc/meminfo | grep -i 'HugePages_Free\\|HugePages_Total'");
    }

    return ptr;
}

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
//                 std::cout << "ftruncate failed for hugetlbfs, trying
//                 fallback..." << std::endl; perror("ftruncate64 on
//                 hugetlbfs"); close(fd); unlink(hugepage_path.c_str()); goto
//                 fallback_to_shm;
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
//             std::cout << "mmap failed on hugetlbfs, trying fallback..." <<
//             std::endl; perror("mmap on hugetlbfs"); if (create)
//             unlink(hugepage_path.c_str());
//         }
//     } else {
//         std::cout << "Cannot open hugetlbfs file (normal if not mounted),
//         trying fallback..." << std::endl;
//     }

// fallback_to_shm:
//     // Method 2: Try shm_open with MAP_HUGETLB (will likely fail but worth
//     trying) std::cout << "Trying shm_open with MAP_HUGETLB..." << std::endl;
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
//         std::cout << "shm_open + MAP_HUGETLB failed (expected), trying
//         regular pages..." << std::endl; perror("mmap with MAP_HUGETLB on
//         shm_open");

//         // Method 3: Fallback to regular pages with huge page hints
//         ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
//         close(fd);

//         if (ptr == MAP_FAILED) {
//             throw std::runtime_error("All mmap methods failed for " +
//             shm_name);
//         }

//         if (create) {
//             // Use madvise hint for transparent huge pages
//             int ret = madvise(ptr, size, MADV_HUGEPAGE);
//             if (ret == 0) {
//                 std::cout << "✓ Successfully hinted for transparent huge
//                 pages (may use 2MB pages)" << std::endl;
//             } else {
//                 std::cout << "madvise(MADV_HUGEPAGE) failed, using regular
//                 4KB pages" << std::endl; perror("madvise");
//             }
//         }
//     }

// success:
//     if (create) {
//         if (numa_available() >= 0) {
//             int num_nodes = numa_num_configured_nodes();
//             std::cout << "NUMA available with " << num_nodes << " nodes." <<
//             std::endl;

//             if (num_nodes >= 2) {
//                 std::cout << "Interleaving memory across NUMA nodes 0 and 1."
//                 << std::endl;

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
//                         std::cout << "✓ NUMA memory policy set successfully"
//                         << std::endl;
//                     }

//                     numa_free_nodemask(nodemask);
//                 }
//             } else {
//                 std::cout << "Only " << num_nodes << " NUMA node(s)
//                 available, skipping interleaving." << std::endl;
//             }

//             // Multi-threaded page touching
//             std::cout << "Starting multi-threaded page touching..." <<
//             std::endl; auto touch_start =
//             std::chrono::high_resolution_clock::now();

//             // Use appropriate page size based on allocation method
//             long page_size = using_huge_pages ? (2 * 1024 * 1024) :
//             sysconf(_SC_PAGESIZE); std::cout << "Using page size: " <<
//             page_size / 1024 << " KB" << std::endl;

//             // Determine optimal number of threads
//             int num_threads = std::min(16, std::max(2,
//             (int)std::thread::hardware_concurrency() / 2)); std::cout <<
//             "Using " << num_threads << " threads for page touching" <<
//             std::endl;

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
//                     std::cout << "Page touching progress: " << std::fixed <<
//                     std::setprecision(1)
//                               << progress << "% (" << current << "/" <<
//                               total_pages << " pages)" << std::endl;
//                     std::this_thread::sleep_for(std::chrono::seconds(2));
//                 }
//             });

//             // Worker function for each thread
//             auto touch_worker = [&](int thread_id) {
//                 int64_t start_offset = thread_id * chunk_size;
//                 int64_t end_offset = (thread_id == num_threads - 1) ? size :
//                 start_offset + chunk_size;

//                 volatile char* p = reinterpret_cast<volatile char*>(ptr);
//                 int64_t local_pages_touched = 0;

//                 // Touch pages in this thread's range
//                 for (int64_t offset = start_offset; offset < end_offset;
//                 offset += page_size) {
//                     p[offset] = 0;
//                     local_pages_touched++;

//                     // Update global counter every 1000 pages to reduce
//                     contention if (local_pages_touched % 1000 == 0) {
//                         completed_pages.fetch_add(1000);
//                     }
//                 }

//                 // Add remaining pages to counter
//                 completed_pages.fetch_add(local_pages_touched % 1000);

//                 std::cout << "Thread " << thread_id << " completed " <<
//                 local_pages_touched << " pages" << std::endl;
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
//             auto touch_duration =
//             std::chrono::duration_cast<std::chrono::milliseconds>(touch_end -
//             touch_start);

//             int64_t final_pages = completed_pages.load();
//             double throughput = (double)size / (1024.0 * 1024.0 * 1024.0) /
//             (touch_duration.count() / 1000.0);

//             std::cout << "✓ Multi-threaded page touching completed:" <<
//             std::endl; std::cout << "  - Pages touched: " << final_pages <<
//             std::endl; std::cout << "  - Time: " << touch_duration.count() <<
//             " ms ("
//                       << touch_duration.count() / 1000.0 << " seconds)" <<
//                       std::endl;
//             std::cout << "  - Throughput: " << std::fixed <<
//             std::setprecision(2)
//                       << throughput << " GB/s" << std::endl;
//         } else {
//             std::cout << "NUMA not available on this system." << std::endl;
//         }
//     }

//     // Register with CUDA - measure performance
//     std::cout << "Starting cudaHostRegister for " << size/(1024*1024*1024) <<
//     "GB..." << std::endl;
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
//     auto cuda_duration =
//     std::chrono::duration_cast<std::chrono::milliseconds>(cuda_end -
//     cuda_start);

//     if (err != cudaSuccess) {
//         munmap(ptr, size);
//         // Clean up hugetlbfs file if we created it
//         if (using_huge_pages && create) {
//             unlink(hugepage_path.c_str());
//         }
//         throw std::runtime_error(
//             std::string("cudaHostRegister failed: ") +
//             cudaGetErrorString(err));
//     }

//     std::cout << "✓ cudaHostRegister completed in " << cuda_duration.count()
//               << " milliseconds (" << cuda_duration.count()/1000.0 << "
//               seconds)" << std::endl;

//     // Verify huge page usage
//     if (create && using_huge_pages) {
//         std::cout << "\nChecking huge page consumption..." << std::endl;
//         system("cat /proc/meminfo | grep -i
//         'HugePages_Free\\|HugePages_Total'");
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
//                 std::cout << "ftruncate failed for hugetlbfs, trying
//                 fallback..." << std::endl; perror("ftruncate64 on
//                 hugetlbfs"); close(fd); unlink(hugepage_path.c_str()); goto
//                 fallback_to_shm;
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
//             std::cout << "mmap failed on hugetlbfs, trying fallback..." <<
//             std::endl; perror("mmap on hugetlbfs"); if (create)
//             unlink(hugepage_path.c_str());
//         }
//     } else {
//         std::cout << "Cannot open hugetlbfs file (normal if not mounted),
//         trying fallback..." << std::endl;
//     }

// fallback_to_shm:
//     // Method 2: Try shm_open with MAP_HUGETLB (will likely fail but worth
//     trying) std::cout << "Trying shm_open with MAP_HUGETLB..." << std::endl;
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
//         std::cout << "shm_open + MAP_HUGETLB failed (expected), trying
//         regular pages..." << std::endl; perror("mmap with MAP_HUGETLB on
//         shm_open");

//         // Method 3: Fallback to regular pages with huge page hints
//         ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
//         close(fd);

//         if (ptr == MAP_FAILED) {
//             throw std::runtime_error("All mmap methods failed for " +
//             shm_name);
//         }

//         if (create) {
//             // Use madvise hint for transparent huge pages
//             int ret = madvise(ptr, size, MADV_HUGEPAGE);
//             if (ret == 0) {
//                 std::cout << "✓ Successfully hinted for transparent huge
//                 pages (may use 2MB pages)" << std::endl;
//             } else {
//                 std::cout << "madvise(MADV_HUGEPAGE) failed, using regular
//                 4KB pages" << std::endl; perror("madvise");
//             }
//         }
//     }

// success:
//     if (create) {
//         if (numa_available() >= 0) {
//             int num_nodes = numa_num_configured_nodes();
//             std::cout << "NUMA available with " << num_nodes << " nodes." <<
//             std::endl;

//             if (num_nodes >= 2) {
//                 std::cout << "Interleaving memory across NUMA nodes 0 and 1."
//                 << std::endl;

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
//                         std::cout << "✓ NUMA memory policy set successfully"
//                         << std::endl;
//                     }

//                     numa_free_nodemask(nodemask);
//                 }
//             } else {
//                 std::cout << "Only " << num_nodes << " NUMA node(s)
//                 available, skipping interleaving." << std::endl;
//             }

//             // Touch pages to enforce actual allocation
//             std::cout << "Touching pages to enforce allocation..." <<
//             std::endl; auto touch_start =
//             std::chrono::high_resolution_clock::now();

//             // Use appropriate page size based on allocation method
//             long page_size = using_huge_pages ? (2 * 1024 * 1024) :
//             sysconf(_SC_PAGESIZE); std::cout << "Using page size: " <<
//             page_size / 1024 << " KB" << std::endl;

//             volatile char* p = reinterpret_cast<volatile char*>(ptr);
//             int64_t pages_touched = 0;

//             for (int64_t i = 0; i < size; i += page_size) {
//                 p[i] = 0;
//                 pages_touched++;

//                 // Progress indicator for large allocations
//                 if (pages_touched % 10000 == 0) {
//                     double progress = (double)i / size * 100.0;
//                     std::cout << "Page touching progress: " << std::fixed <<
//                     std::setprecision(1)
//                               << progress << "%" << std::endl;
//                 }
//             }

//             // Touch the last page if size is not page-aligned
//             if (size > 0) {
//                 p[size - 1] = 0;
//                 pages_touched++;
//             }

//             auto touch_end = std::chrono::high_resolution_clock::now();
//             auto touch_duration =
//             std::chrono::duration_cast<std::chrono::seconds>(touch_end -
//             touch_start);

//             std::cout << "✓ Touched " << pages_touched << " pages in "
//                       << touch_duration.count() << " seconds" << std::endl;
//         } else {
//             std::cout << "NUMA not available on this system." << std::endl;
//         }
//     }

//     // Register with CUDA - measure performance
//     std::cout << "Starting cudaHostRegister for " << size/(1024*1024*1024) <<
//     "GB..." << std::endl; auto cuda_start =
//     std::chrono::high_resolution_clock::now();

//     cudaError_t err = cudaHostRegister(ptr, size, cudaHostRegisterDefault);

//     auto cuda_end = std::chrono::high_resolution_clock::now();
//     auto cuda_duration =
//     std::chrono::duration_cast<std::chrono::milliseconds>(cuda_end -
//     cuda_start);

//     if (err != cudaSuccess) {
//         munmap(ptr, size);
//         // Clean up hugetlbfs file if we created it
//         if (using_huge_pages && create) {
//             unlink(hugepage_path.c_str());
//         }
//         throw std::runtime_error(
//             std::string("cudaHostRegister failed: ") +
//             cudaGetErrorString(err));
//     }

//     std::cout << "✓ cudaHostRegister completed in " << cuda_duration.count()
//               << " milliseconds (" << cuda_duration.count()/1000.0 << "
//               seconds)" << std::endl;

//     // Verify huge page usage
//     if (create && using_huge_pages) {
//         std::cout << "\nChecking huge page consumption..." << std::endl;
//         system("cat /proc/meminfo | grep -i
//         'HugePages_Free\\|HugePages_Total'");
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

//     // void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED,
//     fd, 0);
//     // void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
//     //     MAP_SHARED | MAP_HUGETLB | MAP_HUGE_2MB, fd, 0);
//     // Try huge pages first
//     void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
//                      MAP_SHARED | MAP_HUGETLB | MAP_HUGE_2MB, fd, 0);

//     if (ptr == MAP_FAILED) {
//         std::cout << "Huge page allocation failed, trying regular pages..."
//         << std::endl; perror("mmap with huge pages");

//         // Fallback to regular pages
//         ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

//         if (ptr != MAP_FAILED && create) {
//             // Use madvise hint for regular pages
//             int ret = madvise(ptr, size, MADV_HUGEPAGE);
//             if (ret == 0) {
//                 std::cout << "Successfully hinted for huge pages" <<
//                 std::endl;
//             } else {
//                 perror("madvise(MADV_HUGEPAGE) failed, continuing with
//                 regular pages");
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
//             std::cout << "NUMA available with " << num_nodes << " nodes." <<
//             std::endl;

//             if (num_nodes >= 2) {
//                 std::cout << "Interleaving memory across NUMA nodes 0 and 1."
//                 << std::endl;

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
//                     //                nodemask->maskp, nodemask->size + 1,
//                     0);
//                     // if (ret != 0) {
//                     //     perror("mbind(MPOL_INTERLEAVE)");
//                     // }

//                     numa_free_nodemask(nodemask);
//                 }
//             } else {
//                 std::cout << "Only " << num_nodes << " NUMA node(s)
//                 available, skipping interleaving." << std::endl;
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

    std::cout << "Verifying NUMA allocation for " << num_samples
              << " sample pages:" << std::endl;

    for (int i = 0; i < num_samples; i++) {
        void* page_addr = (char*)ptr + (i * size / num_samples);
        int node = -1;

        if (get_mempolicy(&node, nullptr, 0, page_addr,
                          MPOL_F_NODE | MPOL_F_ADDR) == 0) {
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

//     void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
//     0); close(fd); if (ptr == MAP_FAILED) {
//         throw std::runtime_error("mmap failed for " + shm_name);
//     }

//     if (create) {
//         if (numa_available() >= 0) {
//             std::cout << "Interleaving memory across NUMA nodes." <<
//             std::endl;

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
//             std::cout << "Interleaving memory across NUMA nodes." <<
//             std::endl; unsigned long nodemask = (1UL << 0) | (1UL << 1); int
//             ret = mbind(ptr, size,
//                             MPOL_INTERLEAVE,
//                             &nodemask,
//                             /* maxnode = */ 8 * sizeof(nodemask),
//                             /* flags = */ 0);
//             if (ret != 0) {
//                 // non-fatal: we'll still fall back to default if interleave
//                 fails perror("mbind(MPOL_INTERLEAVE)");
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

// void* allocate_shared_pinned_memory(const std::string& shm_name, int64_t
// size,
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
//     void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
//     0); close(fd);  // fd no longer needed after mapping

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
//     //     std::min(block_size, size - i), cudaHostRegisterDefault); if (err
//     !=
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
