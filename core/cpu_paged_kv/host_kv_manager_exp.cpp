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

namespace kv_manager {

// ============================================================================
// KVCacheManager Implementation
// ============================================================================

KVCacheManager::KVCacheManager(const KVCacheConfig& config)
    : config_(config) {
    
    // Calculate derived values
    config_.page_size_bytes = config_.tokens_per_page * config_.token_size;
    int64_t memory_per_layer = config_.total_memory_bytes / config_.num_layers;
    config_.pages_per_layer = memory_per_layer / config_.page_size_bytes;
    
    total_pages_ = config_.pages_per_layer;
    
    // Generate shared memory names
    pid_t pid = getpid();
    k_shm_name_ = "/kv_cache_k_" + std::to_string(pid);
    if (config_.enable_v_cache) {
        v_shm_name_ = "/kv_cache_v_" + std::to_string(pid);
    }
    
    // Generate socket path if not provided
    if (config_.socket_path.empty()) {
        config_.socket_path = GetDefaultSocketPath();
    }
    socket_path_ = config_.socket_path;
}

bool KVCacheManager::Initialize() {
    std::cout << "[KVCacheManager] Initializing..." << std::endl;
    std::cout << "  Pages per layer: " << config_.pages_per_layer << std::endl;
    std::cout << "  Page size: " << config_.page_size_bytes << " bytes" << std::endl;
    std::cout << "  Tokens per page: " << config_.tokens_per_page << std::endl;
    std::cout << "  Num layers: " << config_.num_layers << std::endl;
    
    // Allocate shared memory
    if (!AllocateSharedMemory()) {
        std::cerr << "[KVCacheManager] Failed to allocate shared memory" << std::endl;
        return false;
    }
    
    // Initialize page metadata
    k_page_metadata_.resize(config_.num_layers);
    for (auto& layer_pages : k_page_metadata_) {
        layer_pages.resize(config_.pages_per_layer);
    }
    
    if (config_.enable_v_cache) {
        v_page_metadata_.resize(config_.num_layers);
        for (auto& layer_pages : v_page_metadata_) {
            layer_pages.resize(config_.pages_per_layer);
        }
    }
    
    // Initialize free page list
    for (int32_t i = 0; i < config_.pages_per_layer; ++i) {
        free_pages_.insert(i);
    }
    
    // Create socket
    if (!CreateSocket()) {
        std::cerr << "[KVCacheManager] Failed to create socket" << std::endl;
        return false;
    }
    
    std::cout << "[KVCacheManager] Initialized successfully" << std::endl;
    std::cout << "  Socket: " << socket_path_ << std::endl;
    std::cout << "  K cache shm: " << k_shm_name_ << std::endl;
    if (config_.enable_v_cache) {
        std::cout << "  V cache shm: " << v_shm_name_ << std::endl;
    }
    
    return true;
}

bool KVCacheManager::AllocateSharedMemory() {
    size_t k_total_size = config_.page_size_bytes * config_.pages_per_layer * config_.num_layers;
    
    // Create K cache shared memory
    k_shm_fd_ = shm_open(k_shm_name_.c_str(), O_CREAT | O_RDWR, 0666);
    if (k_shm_fd_ < 0) {
        perror("shm_open K cache");
        return false;
    }
    
    if (ftruncate(k_shm_fd_, k_total_size) < 0) {
        perror("ftruncate K cache");
        return false;
    }
    
    k_cache_base_ = mmap(nullptr, k_total_size, PROT_READ | PROT_WRITE,
                         MAP_SHARED, k_shm_fd_, 0);
    if (k_cache_base_ == MAP_FAILED) {
        perror("mmap K cache");
        return false;
    }
    
    // Allocate as pinned memory for CUDA DMA
    cudaError_t err = cudaHostRegister(k_cache_base_, k_total_size, cudaHostRegisterDefault);
    if (err != cudaSuccess) {
        std::cerr << "cudaHostRegister K cache failed: " << cudaGetErrorString(err) << std::endl;
        // Continue anyway - workers will register individually
    }
    
    // NUMA binding
    if (config_.enable_numa_binding && config_.numa_node >= 0) {
        if (numa_available() >= 0) {
            struct bitmask* nodemask = numa_allocate_nodemask();
            numa_bitmask_setbit(nodemask, config_.numa_node);
            
            // Move pages to NUMA node
            long ret = mbind(k_cache_base_, k_total_size, MPOL_BIND,
                           nodemask->maskp, nodemask->size + 1, MPOL_MF_MOVE | MPOL_MF_STRICT);
            if (ret < 0) {
                perror("mbind K cache");
            } else {
                std::cout << "  K cache bound to NUMA node " << config_.numa_node << std::endl;
            }
            
            numa_free_nodemask(nodemask);
        }
    }
    
    // Touch pages to commit
    memset(k_cache_base_, 0, k_total_size);
    
    // Similar for V cache
    if (config_.enable_v_cache) {
        size_t v_total_size = k_total_size;  // Same size as K
        
        v_shm_fd_ = shm_open(v_shm_name_.c_str(), O_CREAT | O_RDWR, 0666);
        if (v_shm_fd_ < 0) {
            perror("shm_open V cache");
            return false;
        }
        
        if (ftruncate(v_shm_fd_, v_total_size) < 0) {
            perror("ftruncate V cache");
            return false;
        }
        
        v_cache_base_ = mmap(nullptr, v_total_size, PROT_READ | PROT_WRITE,
                            MAP_SHARED, v_shm_fd_, 0);
        if (v_cache_base_ == MAP_FAILED) {
            perror("mmap V cache");
            return false;
        }
        
        err = cudaHostRegister(v_cache_base_, v_total_size, cudaHostRegisterDefault);
        if (err != cudaSuccess) {
            std::cerr << "cudaHostRegister V cache failed: " << cudaGetErrorString(err) << std::endl;
        }
        
        if (config_.enable_numa_binding && config_.numa_node >= 0 && numa_available() >= 0) {
            struct bitmask* nodemask = numa_allocate_nodemask();
            numa_bitmask_setbit(nodemask, config_.numa_node);
            mbind(v_cache_base_, v_total_size, MPOL_BIND,
                  nodemask->maskp, nodemask->size + 1, MPOL_MF_MOVE | MPOL_MF_STRICT);
            numa_free_nodemask(nodemask);
        }
        
        memset(v_cache_base_, 0, v_total_size);
    }
    
    return true;
}

bool KVCacheManager::AllocateSequence(int64_t uuid, int64_t prefill_length,
                                     int64_t max_decode_tokens,
                                     std::vector<int32_t>& out_page_ids) {
    int64_t total_tokens = prefill_length + max_decode_tokens;
    int64_t num_pages_needed = CalculateNumPages(total_tokens, config_.tokens_per_page);
    
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    
    // Check if UUID already exists
    {
        std::lock_guard<std::mutex> seq_lock(sequence_mutex_);
        if (sequences_.find(uuid) != sequences_.end()) {
            std::cerr << "[AllocateSequence] UUID " << uuid << " already allocated" << std::endl;
            return false;
        }
    }
    
    // Check if enough pages available
    if (free_pages_.size() < static_cast<size_t>(num_pages_needed)) {
        std::cerr << "[AllocateSequence] Not enough free pages. Need " << num_pages_needed
                  << ", have " << free_pages_.size() << std::endl;
        return false;
    }
    
    // Allocate pages
    std::vector<int32_t> page_ids;
    page_ids.reserve(num_pages_needed);
    
    auto it = free_pages_.begin();
    for (int64_t i = 0; i < num_pages_needed; ++i) {
        page_ids.push_back(*it);
        it = free_pages_.erase(it);
    }
    
    // Mark pages as owned
    for (int32_t page_id : page_ids) {
        for (int32_t layer = 0; layer < config_.num_layers; ++layer) {
            k_page_metadata_[layer][page_id].owner_uuid.store(uuid);
            k_page_metadata_[layer][page_id].token_count.store(0);
            
            if (config_.enable_v_cache) {
                v_page_metadata_[layer][page_id].owner_uuid.store(uuid);
                v_page_metadata_[layer][page_id].token_count.store(0);
            }
        }
    }
    
    // Create sequence metadata
    SequenceMetadata seq_meta;
    seq_meta.uuid = uuid;
    seq_meta.page_ids = page_ids;
    seq_meta.prefill_length = prefill_length;
    seq_meta.current_length = prefill_length;  // Will be updated during offload
    seq_meta.max_length = total_tokens;
    seq_meta.numa_node = config_.numa_node;
    seq_meta.is_allocated.store(true);
    
    {
        std::lock_guard<std::mutex> seq_lock(sequence_mutex_);
        sequences_[uuid] = std::move(seq_meta);
    }
    
    used_pages_.fetch_add(num_pages_needed);
    out_page_ids = page_ids;
    
    std::cout << "[AllocateSequence] UUID " << uuid << ": allocated " << num_pages_needed
              << " pages: [";
    for (size_t i = 0; i < page_ids.size() && i < 5; ++i) {
        std::cout << page_ids[i] << (i < page_ids.size() - 1 ? "," : "");
    }
    if (page_ids.size() > 5) std::cout << "...";
    std::cout << "]" << std::endl;
    
    return true;
}

bool KVCacheManager::EvictSequence(int64_t uuid) {
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    std::lock_guard<std::mutex> seq_lock(sequence_mutex_);
    
    auto it = sequences_.find(uuid);
    if (it == sequences_.end()) {
        std::cerr << "[EvictSequence] UUID " << uuid << " not found" << std::endl;
        return false;
    }
    
    const auto& page_ids = it->second.page_ids;
    
    // Mark pages as free
    for (int32_t page_id : page_ids) {
        for (int32_t layer = 0; layer < config_.num_layers; ++layer) {
            k_page_metadata_[layer][page_id].owner_uuid.store(-1);
            k_page_metadata_[layer][page_id].token_count.store(0);
            
            if (config_.enable_v_cache) {
                v_page_metadata_[layer][page_id].owner_uuid.store(-1);
                v_page_metadata_[layer][page_id].token_count.store(0);
            }
        }
        free_pages_.insert(page_id);
    }
    
    used_pages_.fetch_sub(page_ids.size());
    sequences_.erase(it);
    
    std::cout << "[EvictSequence] UUID " << uuid << ": freed " << page_ids.size() << " pages" << std::endl;
    
    return true;
}

std::vector<int64_t> KVCacheManager::GetPageOffsets(int64_t uuid, int32_t layer_idx, bool is_k_cache) const {
    std::lock_guard<std::mutex> seq_lock(sequence_mutex_);
    
    auto it = sequences_.find(uuid);
    if (it == sequences_.end()) {
        return {};
    }
    
    const auto& page_ids = it->second.page_ids;
    std::vector<int64_t> offsets;
    offsets.reserve(page_ids.size());
    
    int64_t layer_offset = layer_idx * config_.pages_per_layer * config_.page_size_bytes;
    
    for (int32_t page_id : page_ids) {
        int64_t page_offset = layer_offset + (page_id * config_.page_size_bytes);
        offsets.push_back(page_offset);
    }
    
    return offsets;
}

PoolStats KVCacheManager::GetPoolStats() const {
    PoolStats stats;
    stats.total_pages = total_pages_.load();
    stats.used_pages = used_pages_.load();
    stats.free_pages = stats.total_pages - stats.used_pages;
    
    {
        std::lock_guard<std::mutex> seq_lock(sequence_mutex_);
        stats.num_sequences = sequences_.size();
    }
    
    stats.numa_node = config_.numa_node;
    
    return stats;
}

// Socket handling implementation...
bool KVCacheManager::CreateSocket() {
    server_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        perror("socket");
        return false;
    }
    
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);
    
    // Remove existing socket file
    unlink(socket_path_.c_str());
    
    if (bind(server_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        return false;
    }
    
    if (listen(server_fd_, 128) < 0) {
        perror("listen");
        return false;
    }
    
    return true;
}

void KVCacheManager::Run() {
    running_.store(true);
    std::cout << "[KVCacheManager] Ready and listening on " << socket_path_ << std::endl;
    
    while (running_.load()) {
        int client_fd = accept(server_fd_, nullptr, nullptr);
        if (client_fd < 0) {
            if (running_.load()) {
                perror("accept");
            }
            continue;
        }
        
        // Handle in separate thread for simplicity (or use thread pool in production)
        std::thread(&KVCacheManager::HandleConnection, this, client_fd).detach();
    }
}

void KVCacheManager::HandleConnection(int client_fd) {
    Request req;
    while (ReceiveRequest(client_fd, req)) {
        ProcessRequest(client_fd, req);
    }
    close(client_fd);
}

void KVCacheManager::ProcessRequest(int client_fd, const Request& req) {
    Response resp;
    resp.success = false;
    
    switch (req.type) {
        case RequestType::REGISTER_WORKER: {
            resp.assigned_worker_id = next_worker_id_.fetch_add(1);
            
            // Fill memory layout
            resp.layout.k_cache_shm_name = k_shm_name_;
            resp.layout.v_cache_shm_name = v_shm_name_;
            resp.layout.k_cache_size = config_.page_size_bytes * config_.pages_per_layer * config_.num_layers;
            resp.layout.v_cache_size = config_.enable_v_cache ? resp.layout.k_cache_size : 0;
            resp.layout.num_layers = config_.num_layers;
            resp.layout.pages_per_layer = config_.pages_per_layer;
            resp.layout.page_size_bytes = config_.page_size_bytes;
            resp.layout.tokens_per_page = config_.tokens_per_page;
            resp.layout.numa_node = config_.numa_node;
            
            resp.success = true;
            std::cout << "[REGISTER_WORKER] Assigned worker ID " << resp.assigned_worker_id << std::endl;
            break;
        }
        
        case RequestType::ALLOCATE_SEQUENCE: {
            resp.success = AllocateSequence(req.uuid, req.num_tokens, req.max_tokens, resp.page_ids);
            break;
        }
        
        case RequestType::EVICT_SEQUENCE: {
            resp.success = EvictSequence(req.uuid);
            break;
        }
        
        case RequestType::GET_STATS: {
            resp.stats = GetPoolStats();
            resp.success = true;
            break;
        }
        
        case RequestType::GET_PAGE_POINTERS: {
            // Return offsets for all layers
            resp.k_page_offsets.reserve(config_.num_layers);
            resp.v_page_offsets.reserve(config_.num_layers);
            
            for (int32_t layer = 0; layer < config_.num_layers; ++layer) {
                auto k_offsets = GetPageOffsets(req.uuid, layer, true);
                auto v_offsets = GetPageOffsets(req.uuid, layer, false);
                
                // For simplicity, return first page offset per layer
                // Workers can compute others based on page IDs
                resp.k_page_offsets.push_back(k_offsets.empty() ? -1 : k_offsets[0]);
                resp.v_page_offsets.push_back(v_offsets.empty() ? -1 : v_offsets[0]);
            }
            resp.success = true;
            break;
        }
        
        case RequestType::SHUTDOWN: {
            resp.success = true;
            SendResponse(client_fd, resp);
            Shutdown();
            return;
        }
    }
    
    SendResponse(client_fd, resp);
}

// Simple binary serialization (production: use protobuf/flatbuffers)
bool KVCacheManager::SendResponse(int fd, const Response& resp) {
    // Simplified - in production use proper serialization
    ssize_t n = write(fd, &resp, sizeof(Response));
    return n == sizeof(Response);
}

bool KVCacheManager::ReceiveRequest(int fd, Request& req) {
    ssize_t n = read(fd, &req, sizeof(Request));
    return n == sizeof(Request);
}

void KVCacheManager::Shutdown() {
    running_.store(false);
    if (server_fd_ >= 0) {
        close(server_fd_);
        unlink(socket_path_.c_str());
    }
    
    // Cleanup shared memory
    if (k_cache_base_) {
        size_t size = config_.page_size_bytes * config_.pages_per_layer * config_.num_layers;
        cudaHostUnregister(k_cache_base_);
        munmap(k_cache_base_, size);
        shm_unlink(k_shm_name_.c_str());
    }
    
    if (v_cache_base_) {
        size_t size = config_.page_size_bytes * config_.pages_per_layer * config_.num_layers;
        cudaHostUnregister(v_cache_base_);
        munmap(v_cache_base_, size);
        shm_unlink(v_shm_name_.c_str());
    }
    
    std::cout << "[KVCacheManager] Shutdown complete" << std::endl;
}

// ============================================================================
// KVCacheClient Implementation (Worker Side)
// ============================================================================

KVCacheClient::KVCacheClient() = default;

KVCacheClient::~KVCacheClient() {
    if (registered_with_cuda_) {
        if (k_cache_mapped_) {
            cudaHostUnregister(k_cache_mapped_);
        }
        if (v_cache_mapped_) {
            cudaHostUnregister(v_cache_mapped_);
        }
    }
    
    if (k_cache_mapped_) {
        munmap(k_cache_mapped_, layout_.k_cache_size);
    }
    if (v_cache_mapped_) {
        munmap(v_cache_mapped_, layout_.v_cache_size);
    }
    
    if (socket_fd_ >= 0) {
        close(socket_fd_);
    }
}

bool KVCacheClient::Connect(const std::string& socket_path) {
    socket_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (socket_fd_ < 0) {
        perror("socket");
        return false;
    }
    
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);
    
    if (connect(socket_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("connect");
        return false;
    }
    
    // Register with manager
    Request req;
    req.type = RequestType::REGISTER_WORKER;
    
    Response resp;
    if (!SendRequest(req, resp) || !resp.success) {
        std::cerr << "[Client] Failed to register with manager" << std::endl;
        return false;
    }
    
    worker_id_ = resp.assigned_worker_id;
    layout_ = resp.layout;
    
    std::cout << "[Client] Registered as worker " << worker_id_ << std::endl;
    
    // Map shared memory
    return MapSharedMemory();
}

bool KVCacheClient::MapSharedMemory() {
    // Map K cache
    int k_fd = shm_open(layout_.k_cache_shm_name.c_str(), O_RDWR, 0666);
    if (k_fd < 0) {
        perror("shm_open K cache");
        return false;
    }
    
    k_cache_mapped_ = mmap(nullptr, layout_.k_cache_size, PROT_READ | PROT_WRITE,
                          MAP_SHARED, k_fd, 0);
    close(k_fd);
    
    if (k_cache_mapped_ == MAP_FAILED) {
        perror("mmap K cache");
        return false;
    }
    
    // Map V cache if enabled
    if (!layout_.v_cache_shm_name.empty()) {
        int v_fd = shm_open(layout_.v_cache_shm_name.c_str(), O_RDWR, 0666);
        if (v_fd < 0) {
            perror("shm_open V cache");
            return false;
        }
        
        v_cache_mapped_ = mmap(nullptr, layout_.v_cache_size, PROT_READ | PROT_WRITE,
                              MAP_SHARED, v_fd, 0);
        close(v_fd);
        
        if (v_cache_mapped_ == MAP_FAILED) {
            perror("mmap V cache");
            return false;
        }
    }
    
    std::cout << "[Client] Mapped shared memory successfully" << std::endl;
    return true;
}

bool KVCacheClient::RegisterWithCUDA(int device_id) {
    cudaSetDevice(device_id);
    
    cudaError_t err = cudaHostRegister(k_cache_mapped_, layout_.k_cache_size,
                                       cudaHostRegisterDefault);
    if (err != cudaSuccess) {
        std::cerr << "[Client] cudaHostRegister K failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    if (v_cache_mapped_) {
        err = cudaHostRegister(v_cache_mapped_, layout_.v_cache_size,
                              cudaHostRegisterDefault);
        if (err != cudaSuccess) {
            std::cerr << "[Client] cudaHostRegister V failed: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
    }
    
    registered_with_cuda_ = true;
    std::cout << "[Client] Registered with CUDA device " << device_id << std::endl;
    return true;
}

bool KVCacheClient::AllocateSequence(int64_t uuid, int64_t prefill_length, int64_t max_decode_tokens) {
    Request req;
    req.type = RequestType::ALLOCATE_SEQUENCE;
    req.uuid = uuid;
    req.num_tokens = prefill_length;
    req.max_tokens = max_decode_tokens;
    req.worker_id = worker_id_;
    
    Response resp;
    if (!SendRequest(req, resp) || !resp.success) {
        return false;
    }
    
    // Cache page IDs
    std::lock_guard<std::mutex> lock(cache_mutex_);
    sequence_pages_[uuid] = resp.page_ids;
    
    return true;
}

bool KVCacheClient::EvictSequence(int64_t uuid) {
    Request req;
    req.type = RequestType::EVICT_SEQUENCE;
    req.uuid = uuid;
    req.worker_id = worker_id_;
    
    Response resp;
    if (!SendRequest(req, resp) || !resp.success) {
        return false;
    }
    
    // Remove from cache
    std::lock_guard<std::mutex> lock(cache_mutex_);
    sequence_pages_.erase(uuid);
    
    return true;
}

void* KVCacheClient::GetKTokenPointer(int64_t uuid, int32_t layer_idx, int64_t token_offset) {
    auto [page_idx, offset_in_page] = GetPageAndOffset(token_offset, layout_.tokens_per_page);
    
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto it = sequence_pages_.find(uuid);
    if (it == sequence_pages_.end() || page_idx >= it->second.size()) {
        return nullptr;
    }
    
    int32_t physical_page_id = it->second[page_idx];
    
    // Calculate offset in memory
    int64_t layer_offset = layer_idx * layout_.pages_per_layer * layout_.page_size_bytes;
    int64_t page_offset = physical_page_id * layout_.page_size_bytes;
    int64_t token_offset_bytes = offset_in_page * (layout_.page_size_bytes / layout_.tokens_per_page);
    
    char* base = static_cast<char*>(k_cache_mapped_);
    return base + layer_offset + page_offset + token_offset_bytes;
}

void* KVCacheClient::GetVTokenPointer(int64_t uuid, int32_t layer_idx, int64_t token_offset) {
    if (!v_cache_mapped_) return nullptr;
    
    auto [page_idx, offset_in_page] = GetPageAndOffset(token_offset, layout_.tokens_per_page);
    
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto it = sequence_pages_.find(uuid);
    if (it == sequence_pages_.end() || page_idx >= it->second.size()) {
        return nullptr;
    }
    
    int32_t physical_page_id = it->second[page_idx];
    
    int64_t layer_offset = layer_idx * layout_.pages_per_layer * layout_.page_size_bytes;
    int64_t page_offset = physical_page_id * layout_.page_size_bytes;
    int64_t token_offset_bytes = offset_in_page * (layout_.page_size_bytes / layout_.tokens_per_page);
    
    char* base = static_cast<char*>(v_cache_mapped_);
    return base + layer_offset + page_offset + token_offset_bytes;
}

bool KVCacheClient::OffloadKV(int64_t uuid, int32_t layer_idx, const void* k_data,
                              const void* v_data, int64_t num_tokens, cudaStream_t stream) {
    // Copy token by token (or optimize with batch copy)
    for (int64_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
        void* k_dst = GetKTokenPointer(uuid, layer_idx, token_idx);
        if (!k_dst) return false;
        
        size_t token_size = layout_.page_size_bytes / layout_.tokens_per_page;
        const char* k_src = static_cast<const char*>(k_data) + token_idx * token_size;
        
        cudaMemcpyAsync(k_dst, k_src, token_size, cudaMemcpyDeviceToHost, stream);
        
        if (v_data && v_cache_mapped_) {
            void* v_dst = GetVTokenPointer(uuid, layer_idx, token_idx);
            if (!v_dst) return false;
            
            const char* v_src = static_cast<const char*>(v_data) + token_idx * token_size;
            cudaMemcpyAsync(v_dst, v_src, token_size, cudaMemcpyDeviceToHost, stream);
        }
    }
    
    return true;
}

bool KVCacheClient::LoadKV(int64_t uuid, int32_t layer_idx, void* k_dst, void* v_dst,
                           int64_t num_tokens, cudaStream_t stream) {
    for (int64_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
        void* k_src = GetKTokenPointer(uuid, layer_idx, token_idx);
        if (!k_src) return false;
        
        size_t token_size = layout_.page_size_bytes / layout_.tokens_per_page;
        char* k_dst_ptr = static_cast<char*>(k_dst) + token_idx * token_size;
        
        cudaMemcpyAsync(k_dst_ptr, k_src, token_size, cudaMemcpyHostToDevice, stream);
        
        if (v_dst && v_cache_mapped_) {
            void* v_src = GetVTokenPointer(uuid, layer_idx, token_idx);
            if (!v_src) return false;
            
            char* v_dst_ptr = static_cast<char*>(v_dst) + token_idx * token_size;
            cudaMemcpyAsync(v_dst_ptr, v_src, token_size, cudaMemcpyHostToDevice, stream);
        }
    }
    
    return true;
}

bool KVCacheClient::SendRequest(const Request& req, Response& resp) {
    ssize_t n = write(socket_fd_, &req, sizeof(Request));
    if (n != sizeof(Request)) {
        return false;
    }
    
    n = read(socket_fd_, &resp, sizeof(Response));
    return n == sizeof(Response);
}

PoolStats KVCacheClient::GetPoolStats() {
    Request req;
    req.type = RequestType::GET_STATS;
    req.worker_id = worker_id_;
    
    Response resp;
    if (!SendRequest(req, resp) || !resp.success) {
        return {};
    }
    
    return resp.stats;
}

// ============================================================================
// Utility Functions
// ============================================================================

std::string GetDefaultSocketPath() {
    pid_t pid = getpid();
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    
    return "/tmp/kv_manager_" + std::string(hostname) + "_" + std::to_string(pid) + ".sock";
}

} // namespace kv_manager