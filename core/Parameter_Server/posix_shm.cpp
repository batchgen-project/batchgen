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

void* allocate_shared_pinned_memory(const std::string& shm_name,
                                    int64_t size,
                                    bool create) {
    int flags = O_RDWR | (create ? O_CREAT : 0);
    int fd = shm_open(shm_name.c_str(), flags, 0666);
    if (fd < 0) {
        throw std::runtime_error("shm_open failed for " + shm_name);
    }

    if (create) {
        if (ftruncate64(fd, size) == -1) {
            close(fd);
            throw std::runtime_error("ftruncate failed for " + shm_name);
        }
    }

    void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (ptr == MAP_FAILED) {
        throw std::runtime_error("mmap failed for " + shm_name);
    }

    if (create) {
        if (numa_available() >= 0) {
            int num_nodes = numa_num_configured_nodes();
            std::cout << "NUMA available with " << num_nodes << " nodes." << std::endl;
            
            if (num_nodes >= 2) {
                std::cout << "Interleaving memory across NUMA nodes 0 and 1." << std::endl;

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
                    int ret = set_mempolicy(MPOL_INTERLEAVE, 
                                          nodemask->maskp, 
                                          nodemask->size + 1);
                    if (ret != 0) {
                        perror("set_mempolicy(MPOL_INTERLEAVE)");
                    }

                    // Alternative: use mbind with correct parameters
                    // int ret = mbind(ptr, size, MPOL_INTERLEAVE, 
                    //                nodemask->maskp, nodemask->size + 1, 0);
                    // if (ret != 0) {
                    //     perror("mbind(MPOL_INTERLEAVE)");
                    // }

                    numa_free_nodemask(nodemask);
                }
            } else {
                std::cout << "Only " << num_nodes << " NUMA node(s) available, skipping interleaving." << std::endl;
            }

            // Touch pages to enforce actual allocation
            // Use proper page size and ensure we don't go out of bounds
            long page_size = sysconf(_SC_PAGESIZE);
            volatile char* p = reinterpret_cast<volatile char*>(ptr);
            for (int64_t i = 0; i < size; i += page_size) {
                p[i] = 0;
            }
            
            // Touch the last page if size is not page-aligned
            if (size > 0) {
                p[size - 1] = 0;
            }
        } else {
            std::cout << "NUMA not available on this system." << std::endl;
        }
    }

    cudaError_t err = cudaHostRegister(ptr, size, cudaHostRegisterDefault);
    if (err != cudaSuccess) {
        munmap(ptr, size);
        throw std::runtime_error(
            std::string("cudaHostRegister failed: ") +
            cudaGetErrorString(err));
    }

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
