// ============================================================================
// kv_cache_manager.h
// ============================================================================

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <set>

#include <cuda_runtime.h>

namespace host_kv_manager {

// Configuration for the KV cache manager
struct KVCacheConfig {
    // Memory configuration
    int64_t tokens_per_page = 64;           // Tokens per page (64, 128, 256, 512)
    int64_t token_size = 0;                 // Bytes per token (e.g., head_dim * num_kv_heads * sizeof(dtype))
    int64_t num_layers = 0;                 // Number of transformer layers
    int64_t total_memory_bytes = 0;         // Total memory per NUMA node
    
    // NUMA configuration
    int numa_node = -1;                     // -1 = use all available, or specific node
    bool enable_numa_binding = true;        // Whether to bind memory to NUMA nodes
    
    // K/V configuration
    bool enable_v_cache = true;             // Set to false for DeepSeek-style K-only cache
    
    // Manager configuration
    std::string socket_path = "";           // Auto-generated if empty
    int max_sequences = 1024;               // Maximum concurrent sequences
    
    // Computed values (filled by manager)
    int64_t pages_per_layer = 0;            // Computed from total_memory_bytes
    int64_t page_size_bytes = 0;            // tokens_per_page * token_size
};

// Statistics for a NUMA pool
struct PoolStats {
    int64_t total_pages;
    int64_t used_pages;
    int64_t free_pages;
    int64_t num_sequences;
    int numa_node;
};

// Metadata for a sequence
struct SequenceMetadata {
    int64_t uuid;
    std::vector<int32_t> page_ids;          // Same page IDs across all layers
    int64_t prefill_length;                 // Number of prefill tokens
    int64_t current_length;                 // Current total length
    int64_t max_length;                     // Reserved capacity (prefill + max_decode)
    int numa_node;                          // Which NUMA node owns this sequence
    std::atomic<bool> is_allocated{false};  // Allocation status
};

// Page metadata (per layer)
struct PageMetadata {
    std::atomic<int64_t> owner_uuid{-1};    // -1 = free, else sequence UUID
    std::atomic<int32_t> token_count{0};    // Number of tokens in this page
    std::atomic<int32_t> ref_count{0};      // For shared reads (future use)
};

// Memory layout information for workers
struct MemoryLayout {
    // Shared memory names (for shm_open)
    std::string k_cache_shm_name;
    std::string v_cache_shm_name;  // empty if V cache disabled
    
    // Memory regions (workers will mmap these)
    size_t k_cache_size;
    size_t v_cache_size;
    
    // Layout info
    int64_t num_layers;
    int64_t pages_per_layer;
    int64_t page_size_bytes;
    int64_t tokens_per_page;
    int numa_node;
};

// Request/Response structures for socket protocol
enum class RequestType : uint8_t {
    REGISTER_WORKER,
    ALLOCATE_SEQUENCE,
    EVICT_SEQUENCE,
    GET_STATS,
    GET_PAGE_POINTERS,
    SHUTDOWN
};

struct Request {
    RequestType type;
    int64_t uuid;           // Sequence UUID
    int64_t num_tokens;     // For allocation
    int64_t max_tokens;     // For allocation
    int32_t worker_id;      // Worker identifier
};

struct Response {
    bool success;
    std::string error_msg;
    
    // For REGISTER_WORKER
    MemoryLayout layout;
    int32_t assigned_worker_id;
    
    // For ALLOCATE_SEQUENCE
    std::vector<int32_t> page_ids;
    
    // For GET_PAGE_POINTERS (returns offsets, workers compute actual pointers)
    std::vector<int64_t> k_page_offsets;  // Per layer
    std::vector<int64_t> v_page_offsets;  // Per layer
    
    // For GET_STATS
    PoolStats stats;
};

// ============================================================================
// KVCacheManager - Manager Process
// ============================================================================

class KVCacheManager {
public:
    explicit KVCacheManager(const KVCacheConfig& config);
    ~KVCacheManager();
    
    // Initialize and start the manager
    bool Initialize();
    
    // Start listening for worker connections
    void Run();
    
    // Graceful shutdown
    void Shutdown();
    
    // Core allocation APIs (called via socket from workers)
    bool AllocateSequence(int64_t uuid, int64_t prefill_length, int64_t max_decode_tokens,
                         std::vector<int32_t>& out_page_ids);
    bool EvictSequence(int64_t uuid);
    
    // Query APIs
    PoolStats GetPoolStats() const;
    bool GetSequenceMetadata(int64_t uuid, SequenceMetadata& out_metadata) const;
    std::vector<int64_t> GetPageOffsets(int64_t uuid, int32_t layer_idx, bool is_k_cache) const;
    
    // Memory accessors (workers use these offsets + base pointer)
    void* GetKCacheBasePtr() const { return k_cache_base_; }
    void* GetVCacheBasePtr() const { return v_cache_base_; }
    
    const KVCacheConfig& GetConfig() const { return config_; }
    
private:
    // Memory management
    bool AllocateSharedMemory();
    void FreeSharedMemory();
    bool BindToNUMA();
    
    // Page allocation
    bool AllocatePages(int64_t num_pages, std::vector<int32_t>& out_page_ids);
    void FreePages(const std::vector<int32_t>& page_ids);
    
    // Socket communication
    bool CreateSocket();
    void HandleConnection(int client_fd);
    void ProcessRequest(int client_fd, const Request& req);
    
    // Serialization helpers
    bool SendResponse(int fd, const Response& resp);
    bool ReceiveRequest(int fd, Request& req);
    
private:
    KVCacheConfig config_;
    
    // Memory regions
    void* k_cache_base_ = nullptr;
    void* v_cache_base_ = nullptr;
    int k_shm_fd_ = -1;
    int v_shm_fd_ = -1;
    std::string k_shm_name_;
    std::string v_shm_name_;
    
    // Page management (per layer)
    std::vector<std::vector<PageMetadata>> k_page_metadata_;  // [layer][page]
    std::vector<std::vector<PageMetadata>> v_page_metadata_;  // [layer][page]
    
    // Free page tracking
    std::set<int32_t> free_pages_;
    mutable std::mutex allocation_mutex_;
    
    // Sequence tracking
    std::unordered_map<int64_t, SequenceMetadata> sequences_;
    mutable std::mutex sequence_mutex_;
    
    // Statistics
    std::atomic<int64_t> total_pages_{0};
    std::atomic<int64_t> used_pages_{0};
    
    // Socket
    int server_fd_ = -1;
    std::string socket_path_;
    std::atomic<bool> running_{false};
    
    // Worker tracking
    std::atomic<int32_t> next_worker_id_{0};
};

// ============================================================================
// KVCacheClient - Worker Side
// ============================================================================

class KVCacheClient {
public:
    KVCacheClient();
    ~KVCacheClient();
    
    // Connect to manager and register
    bool Connect(const std::string& socket_path);
    
    // Register this worker with CUDA device
    bool RegisterWithCUDA(int device_id);
    
    // Allocation APIs
    bool AllocateSequence(int64_t uuid, int64_t prefill_length, int64_t max_decode_tokens);
    bool EvictSequence(int64_t uuid);
    
    // Get page pointers for DMA operations
    void* GetKPagePointer(int64_t uuid, int32_t layer_idx, int32_t page_idx);
    void* GetVPagePointer(int64_t uuid, int32_t layer_idx, int32_t page_idx);
    
    // Convenience: Get pointer for specific token offset
    void* GetKTokenPointer(int64_t uuid, int32_t layer_idx, int64_t token_offset);
    void* GetVTokenPointer(int64_t uuid, int32_t layer_idx, int64_t token_offset);
    
    // Query APIs
    PoolStats GetPoolStats();
    const MemoryLayout& GetMemoryLayout() const { return layout_; }
    
    // High-level operations (helpers that combine pointer getting + memcpy)
    bool OffloadKV(int64_t uuid, int32_t layer_idx, const void* k_data, 
                   const void* v_data, int64_t num_tokens, cudaStream_t stream);
    bool LoadKV(int64_t uuid, int32_t layer_idx, void* k_dst, void* v_dst,
                int64_t num_tokens, cudaStream_t stream);
    
private:
    bool MapSharedMemory();
    bool SendRequest(const Request& req, Response& resp);
    
private:
    int socket_fd_ = -1;
    int32_t worker_id_ = -1;
    
    MemoryLayout layout_;
    
    // Mapped memory
    void* k_cache_mapped_ = nullptr;
    void* v_cache_mapped_ = nullptr;
    
    // Sequence metadata cache (to avoid repeated socket calls)
    std::unordered_map<int64_t, std::vector<int32_t>> sequence_pages_;
    mutable std::mutex cache_mutex_;
    
    bool registered_with_cuda_ = false;
};

// ============================================================================
// Utility Functions
// ============================================================================

// Generate default socket path
std::string GetDefaultSocketPath();

// Calculate number of pages needed
inline int64_t CalculateNumPages(int64_t num_tokens, int64_t tokens_per_page) {
    return (num_tokens + tokens_per_page - 1) / tokens_per_page;
}

// Get page index and offset within page for a token
inline std::pair<int32_t, int32_t> GetPageAndOffset(int64_t token_idx, int64_t tokens_per_page) {
    return {static_cast<int32_t>(token_idx / tokens_per_page),
            static_cast<int32_t>(token_idx % tokens_per_page)};
}

} // namespace kv_manager