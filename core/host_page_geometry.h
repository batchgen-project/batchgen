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

// Page geometry of the running host, the single source for every page-size
// dependent stride, alignment and rounding in core (Python mirror:
// batchgen/host_page_geometry.py, which applies the same rules):
//   base page  sysconf(_SC_PAGESIZE)          4 KiB on x86_64, 4/16/64 KiB arm64
//   THP size   transparent_hugepage/hpage_pmd_size  2 MiB on x86_64, 512 MiB
//                                              on a 64 KiB-page arm64 kernel
//   hugetlb    Hugepagesize in /proc/meminfo   the default huge page size
// A shmem mapping (memfd or tmpfs file) is backed by THP only when the policy
// that governs it allows it; its backing page size is then the THP size, else
// the base page. Pre-touch strides and the alignment of mapped extents use the
// backing size, so neither under-touches base pages nor over-aligns.

#pragma once

#include <linux/magic.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

#ifndef HUGETLBFS_MAGIC
#define HUGETLBFS_MAGIC 0x958458f6
#endif
#ifndef TMPFS_MAGIC
#define TMPFS_MAGIC 0x01021994
#endif

namespace batchgen {

// pybind11 maps this to RuntimeError; every message starts with the class name.
class HostPageGeometryError : public std::runtime_error {
   public:
    using std::runtime_error::runtime_error;
};

namespace host_page_geometry_detail {

inline std::string ReadFile(const char* path) {
    std::ifstream in(path);
    if (!in) {
        return {};
    }
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
}

}  // namespace host_page_geometry_detail

struct HostPageGeometry {
    std::size_t base_page_bytes = 0;
    std::size_t thp_pmd_bytes = 0;       // 0: the kernel has no THP
    std::size_t hugetlb_page_bytes = 0;  // 0: the kernel has no hugetlb
    std::string shmem_thp_policy;        // active shmem_enabled mode, "" without THP

    // Pure parse of the three host sources, so the rules are testable with
    // any host's text. An empty hpage_pmd_size / shmem_enabled text means the
    // kernel has no THP; meminfo without Hugepagesize means no hugetlb.
    static HostPageGeometry FromText(std::size_t base_page_bytes,
                                     const std::string& hpage_pmd_size_text,
                                     const std::string& meminfo_text,
                                     const std::string& shmem_enabled_text) {
        if (base_page_bytes == 0 ||
            (base_page_bytes & (base_page_bytes - 1)) != 0) {
            throw HostPageGeometryError(
                "HostPageGeometryError: base page size " +
                std::to_string(base_page_bytes) + " is not a power of two");
        }
        HostPageGeometry geometry;
        geometry.base_page_bytes = base_page_bytes;
        std::istringstream pmd(hpage_pmd_size_text);
        std::string pmd_value;
        if (pmd >> pmd_value) {
            if (pmd_value.find_first_not_of("0123456789") !=
                std::string::npos) {
                throw HostPageGeometryError(
                    "HostPageGeometryError: unparsable hpage_pmd_size '" +
                    hpage_pmd_size_text + "'");
            }
            geometry.thp_pmd_bytes =
                static_cast<std::size_t>(std::stoull(pmd_value));
        }
        std::istringstream meminfo(meminfo_text);
        for (std::string line; std::getline(meminfo, line);) {
            if (line.rfind("Hugepagesize:", 0) != 0) {
                continue;
            }
            std::istringstream fields(line);
            std::string key, value, unit;
            fields >> key >> value >> unit;
            if (unit != "kB" || value.empty() ||
                value.find_first_not_of("0123456789") != std::string::npos) {
                throw HostPageGeometryError(
                    "HostPageGeometryError: unparsable meminfo line '" + line +
                    "'");
            }
            geometry.hugetlb_page_bytes =
                static_cast<std::size_t>(std::stoull(value)) * 1024;
        }
        const auto open = shmem_enabled_text.find('[');
        const auto close = shmem_enabled_text.find(']', open);
        if (open != std::string::npos && close != std::string::npos) {
            geometry.shmem_thp_policy =
                shmem_enabled_text.substr(open + 1, close - open - 1);
        }
        return geometry;
    }

    static HostPageGeometry ReadHost() {
        const long page = sysconf(_SC_PAGESIZE);
        if (page <= 0) {
            throw HostPageGeometryError(
                std::string("HostPageGeometryError: sysconf(_SC_PAGESIZE) "
                            "failed: ") +
                std::strerror(errno));
        }
        using host_page_geometry_detail::ReadFile;
        return FromText(
            static_cast<std::size_t>(page),
            ReadFile("/sys/kernel/mm/transparent_hugepage/hpage_pmd_size"),
            ReadFile("/proc/meminfo"),
            ReadFile("/sys/kernel/mm/transparent_hugepage/shmem_enabled"));
    }

    // Read once per process: the sizes are fixed at boot. The shmem policy
    // can be rewritten at run time; a region records the backing size it was
    // created with where an attacher depends on it (host KV header).
    static const HostPageGeometry& Host() {
        static const HostPageGeometry geometry = ReadHost();
        return geometry;
    }

    // Whether THP backs a shmem mapping. mount_huge is the huge= option of
    // the tmpfs mount holding the object, or nullopt for the kernel's
    // internal mount (memfd, shared anonymous), which shmem_enabled governs.
    // deny/force in shmem_enabled override every mount.
    bool ShmemThpActive(bool madvised,
                        const std::optional<std::string>& mount_huge) const {
        if (thp_pmd_bytes == 0 || shmem_thp_policy == "deny") {
            return false;
        }
        if (shmem_thp_policy == "force") {
            return true;
        }
        const std::string& mode = mount_huge ? *mount_huge : shmem_thp_policy;
        return mode == "always" || mode == "within_size" ||
               (mode == "advise" && madvised);
    }

    std::size_t ShmemBackingBytes(
        bool madvised, const std::optional<std::string>& mount_huge) const {
        return ShmemThpActive(madvised, mount_huge)
                   ? std::max(thp_pmd_bytes, base_page_bytes)
                   : base_page_bytes;
    }

    // Default huge page size (MFD_HUGETLB without a size flag, nr_hugepages).
    std::size_t HugetlbPageBytes() const {
        if (hugetlb_page_bytes == 0) {
            throw HostPageGeometryError(
                "HostPageGeometryError: /proc/meminfo has no Hugepagesize; "
                "this kernel provides no hugetlb pages");
        }
        return hugetlb_page_bytes;
    }
};

// huge= option of the visible tmpfs mount that holds fd, "never" when the
// mount sets none, nullopt when fd lives on the kernel's internal shmem mount
// (memfd), which /proc/self/mountinfo does not list.
inline std::optional<std::string> TmpfsMountHugeOption(int fd) {
    struct stat st {};
    if (fstat(fd, &st) != 0) {
        throw HostPageGeometryError(
            std::string("HostPageGeometryError: fstat failed: ") +
            std::strerror(errno));
    }
    const std::string device = std::to_string(major(st.st_dev)) + ":" +
                               std::to_string(minor(st.st_dev));
    std::istringstream mountinfo(
        host_page_geometry_detail::ReadFile("/proc/self/mountinfo"));
    for (std::string line; std::getline(mountinfo, line);) {
        std::istringstream fields(line);
        std::string mount_id, parent_id, dev;
        fields >> mount_id >> parent_id >> dev;
        if (dev != device) {
            continue;
        }
        const auto separator = line.find(" - ");
        if (separator == std::string::npos) {
            continue;
        }
        std::istringstream tail(line.substr(separator + 3));
        std::string fstype, source, options;
        tail >> fstype >> source >> options;
        if (fstype != "tmpfs") {
            continue;
        }
        std::istringstream option_list(options);
        for (std::string option; std::getline(option_list, option, ',');) {
            if (option.rfind("huge=", 0) == 0) {
                return option.substr(5);
            }
        }
        return std::string("never");
    }
    return std::nullopt;
}

// Page size that backs a MAP_SHARED mapping of fd: the hugetlbfs block size
// for a hugetlb object, the THP size for a shmem object THP may back
// (madvised = MADV_HUGEPAGE is applied to the mapping), else the base page.
inline std::size_t MappingBackingBytes(int fd, bool madvised,
                                       const HostPageGeometry& geometry =
                                           HostPageGeometry::Host()) {
    struct statfs sfs {};
    if (fstatfs(fd, &sfs) != 0) {
        throw HostPageGeometryError(
            std::string("HostPageGeometryError: fstatfs failed: ") +
            std::strerror(errno));
    }
    const auto type = static_cast<unsigned long>(sfs.f_type);
    if (type == HUGETLBFS_MAGIC) {
        return static_cast<std::size_t>(sfs.f_bsize);
    }
    if (type == TMPFS_MAGIC) {
        return geometry.ShmemBackingBytes(madvised, TmpfsMountHugeOption(fd));
    }
    return geometry.base_page_bytes;
}

}  // namespace batchgen
