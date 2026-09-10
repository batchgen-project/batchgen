#pragma once

// The H20 runtime images ship libnuma.so.1 but not the libnuma-dev headers.
// Keep the core extension buildable against that runtime ABI without making
// the deployment depend on a mutable system package installation.

#include <linux/mempolicy.h>

struct bitmask;

extern "C" {
int numa_available(void);
bitmask* numa_get_interleave_mask(void);
int numa_set_interleave_mask(bitmask* mask);
void numa_bitmask_free(bitmask* mask);
extern bitmask* numa_all_nodes_ptr;
int get_mempolicy(int* policy, unsigned long* nodemask,
                  unsigned long maxnode, void* address,
                  unsigned flags);
}

// libnuma's development header provides this compatibility spelling, while
// the deployed shared object exports numa_bitmask_free.
inline void numa_free_nodemask(bitmask* mask) {
    numa_bitmask_free(mask);
}
