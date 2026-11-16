#pragma once

// VMM-backed prefill ring segments (slice 2c of the node replica transport).
//
// The ring's device memory is allocated with the CUDA VMM API and exported as
// a POSIX file descriptor. A node peer imports that fd and maps ONLY this
// range into its own address space with read access for its own device. The
// torch CUDA-IPC path this replaces (reduce_tensor / rebuild) went through
// cudaIpcOpenMemHandle, whose lazy peer-access enablement maps every existing
// allocation of both processes pair by pair: on a serving worker with
// hundreds of GB of pinned host weights one import took tens of seconds and
// the cohort install minutes, and each peer left a permanent foreign CUDA
// context on every device. A VMM import touches one range, costs
// milliseconds, creates no foreign context, and unmapping at uninstall
// returns everything.

#include <cuda.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace batchgen_vmm {

inline void vmm_check(CUresult result, const char* what) {
    if (result != CUDA_SUCCESS) {
        const char* name = nullptr;
        const char* message = nullptr;
        cuGetErrorName(result, &name);
        cuGetErrorString(result, &message);
        throw std::runtime_error(
            std::string(what) + " failed: " + (name ? name : "unknown") +
            " (" + (message ? message : "no detail") + ")");
    }
}

inline size_t vmm_padded_bytes(int device, int64_t bytes) {
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    size_t granularity = 0;
    vmm_check(cuMemGetAllocationGranularity(
                  &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
              "cuMemGetAllocationGranularity");
    return (static_cast<size_t>(bytes) + granularity - 1) / granularity *
           granularity;
}

// One ring segment: reserved VA + physical handle + RW mapping on the owner.
class VmmRingSegment {
  public:
    VmmRingSegment(int device, int64_t bytes) : device_(device) {
        mapped_bytes_ = vmm_padded_bytes(device, bytes);
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        vmm_check(cuMemAddressReserve(&va_, mapped_bytes_, 0, 0, 0),
                  "cuMemAddressReserve");
        try {
            vmm_check(cuMemCreate(&handle_, mapped_bytes_, &prop, 0),
                      "cuMemCreate");
        } catch (...) {
            cuMemAddressFree(va_, mapped_bytes_);
            throw;
        }
        try {
            vmm_check(cuMemMap(va_, mapped_bytes_, 0, handle_, 0), "cuMemMap");
            CUmemAccessDesc access = {};
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = device;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            vmm_check(cuMemSetAccess(va_, mapped_bytes_, &access, 1),
                      "cuMemSetAccess");
            vmm_check(cuMemsetD8(va_, 0, mapped_bytes_), "cuMemsetD8");
        } catch (...) {
            cuMemUnmap(va_, mapped_bytes_);
            cuMemRelease(handle_);
            cuMemAddressFree(va_, mapped_bytes_);
            throw;
        }
    }

    VmmRingSegment(const VmmRingSegment&) = delete;
    VmmRingSegment& operator=(const VmmRingSegment&) = delete;

    ~VmmRingSegment() {
        cuMemUnmap(va_, mapped_bytes_);
        cuMemRelease(handle_);
        cuMemAddressFree(va_, mapped_bytes_);
    }

    // The caller owns the returned fd and closes it after the peers imported.
    int export_fd() const {
        int fd = -1;
        vmm_check(cuMemExportToShareableHandle(
                      &fd, handle_, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                      0),
                  "cuMemExportToShareableHandle");
        return fd;
    }

    void* data() const { return reinterpret_cast<void*>(va_); }
    size_t mapped_bytes() const { return mapped_bytes_; }
    int device() const { return device_; }

  private:
    int device_ = -1;
    CUdeviceptr va_ = 0;
    CUmemGenericAllocationHandle handle_ = 0;
    size_t mapped_bytes_ = 0;
};

// A peer's exported segment, mapped read-only for this process's device. The
// fd stays owned by the caller; the import holds its own handle reference.
class VmmPeerImport {
  public:
    VmmPeerImport(int fd, size_t mapped_bytes, int reader_device)
        : mapped_bytes_(mapped_bytes) {
        vmm_check(cuMemImportFromShareableHandle(
                      &handle_,
                      reinterpret_cast<void*>(static_cast<intptr_t>(fd)),
                      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
                  "cuMemImportFromShareableHandle");
        try {
            vmm_check(cuMemAddressReserve(&va_, mapped_bytes_, 0, 0, 0),
                      "cuMemAddressReserve (import)");
            vmm_check(cuMemMap(va_, mapped_bytes_, 0, handle_, 0),
                      "cuMemMap (import)");
            CUmemAccessDesc access = {};
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = reader_device;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
            vmm_check(cuMemSetAccess(va_, mapped_bytes_, &access, 1),
                      "cuMemSetAccess (import)");
        } catch (...) {
            if (va_ != 0) {
                cuMemUnmap(va_, mapped_bytes_);
                cuMemAddressFree(va_, mapped_bytes_);
            }
            cuMemRelease(handle_);
            throw;
        }
    }

    VmmPeerImport(const VmmPeerImport&) = delete;
    VmmPeerImport& operator=(const VmmPeerImport&) = delete;

    ~VmmPeerImport() {
        cuMemUnmap(va_, mapped_bytes_);
        cuMemRelease(handle_);
        cuMemAddressFree(va_, mapped_bytes_);
    }

    void* data() const { return reinterpret_cast<void*>(va_); }
    size_t mapped_bytes() const { return mapped_bytes_; }

  private:
    CUdeviceptr va_ = 0;
    CUmemGenericAllocationHandle handle_ = 0;
    size_t mapped_bytes_ = 0;
};

}  // namespace batchgen_vmm
