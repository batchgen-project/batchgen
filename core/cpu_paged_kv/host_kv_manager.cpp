// ============================================================================
// kv_cache_manager.cpp - Key Methods
// ============================================================================

#include "host_kv_manager.h"
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <numa.h>
#include <sstream>
#include <iostream>

HostKVCacheManager::