#include "distributed_weight_daemon.h"

#include "distributed_weights_protocol.h"
#include "latency_histogram.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <linux/memfd.h>
#include <netinet/in.h>
#include <pthread.h>
#include <sched.h>
#include "../numa_compat.h"
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <ucp/api/ucp.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "nlohmann_json/json.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using batchgen::distributed_weights::kAcquireNoWait;
using batchgen::distributed_weights::kHelloReleaseChannel;
using batchgen::distributed_weights::kMaxLeasesWhileAcquiring;
using batchgen::distributed_weights::kStatusRetry;
using batchgen::distributed_weights::Operation;
using batchgen::distributed_weights::Request;
using batchgen::distributed_weights::Response;

constexpr std::uint64_t kBootstrapMagic = 0x4b33523452444d41ULL;
constexpr int kRails = 8;
constexpr int kLayers = 92;
constexpr int kExperts = 896;
constexpr int kDefaultWorkers = 8;
// The summary keeps the most recent cohort swaps only.
constexpr size_t kSwapLogEntries = 32;
// One acquire request waits at most one slice, so a worker can stop between
// slices; a kAcquireNoWait request waits none. The timeout spans all requests
// of one pending acquire, waiting or not.
constexpr auto kAcquireSlice = std::chrono::milliseconds(250);
constexpr auto kAcquireTimeout = std::chrono::seconds(300);
// A pending acquire not polled for this long belongs to a worker that stopped
// asking (its producer was cancelled); the next request starts a fresh wait.
constexpr auto kAcquireAbandoned = std::chrono::seconds(5);

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void check_ucs(ucs_status_t status, const char* what) {
    if (status != UCS_OK) {
        fail(std::string(what) + ": " + ucs_status_string(status));
    }
}

void send_exact(int fd, const void* data, size_t bytes) {
    const char* cursor = static_cast<const char*>(data);
    while (bytes > 0) {
        const ssize_t sent = send(fd, cursor, bytes, MSG_NOSIGNAL);
        if (sent < 0) {
            if (errno == EINTR) {
                continue;
            }
            fail("send failed: " + std::string(strerror(errno)));
        }
        if (sent == 0) {
            fail("send returned zero");
        }
        cursor += sent;
        bytes -= static_cast<size_t>(sent);
    }
}

void recv_exact(int fd, void* data, size_t bytes) {
    char* cursor = static_cast<char*>(data);
    while (bytes > 0) {
        const ssize_t received = recv(fd, cursor, bytes, MSG_WAITALL);
        if (received < 0) {
            if (errno == EINTR) {
                continue;
            }
            fail("recv failed: " + std::string(strerror(errno)));
        }
        if (received == 0) {
            fail("peer closed the socket");
        }
        cursor += received;
        bytes -= static_cast<size_t>(received);
    }
}

void send_response_with_fd(int socket_fd, const Response& response,
                           int passed_fd) {
    char control[CMSG_SPACE(sizeof(int))] = {};
    iovec iov{const_cast<Response*>(&response), sizeof(response)};
    msghdr message{};
    message.msg_iov = &iov;
    message.msg_iovlen = 1;
    message.msg_control = control;
    message.msg_controllen = sizeof(control);
    cmsghdr* header = CMSG_FIRSTHDR(&message);
    header->cmsg_level = SOL_SOCKET;
    header->cmsg_type = SCM_RIGHTS;
    header->cmsg_len = CMSG_LEN(sizeof(int));
    std::memcpy(CMSG_DATA(header), &passed_fd, sizeof(passed_fd));
    const ssize_t sent = sendmsg(socket_fd, &message, MSG_NOSIGNAL);
    if (sent != static_cast<ssize_t>(sizeof(response))) {
        fail("sendmsg staging memfd failed");
    }
}

void set_error(Response* response, const std::string& message) {
    response->status = -1;
    std::strncpy(response->error, message.c_str(),
                 sizeof(response->error) - 1);
}

int create_listener(const std::string& ip, int port) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        fail("socket: " + std::string(strerror(errno)));
    }
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<std::uint16_t>(port));
    if (inet_pton(AF_INET, ip.c_str(), &address.sin_addr) != 1) {
        close(fd);
        fail("invalid bind IP " + ip);
    }
    if (bind(fd, reinterpret_cast<sockaddr*>(&address),
             sizeof(address)) != 0) {
        const std::string error = strerror(errno);
        close(fd);
        fail("bind " + ip + ":" + std::to_string(port) + ": " +
             error);
    }
    if (listen(fd, 1) != 0) {
        const std::string error = strerror(errno);
        close(fd);
        fail("listen: " + error);
    }
    return fd;
}

int connect_retry(const std::string& ip, int port, int timeout_s,
                  const std::atomic<bool>& stop) {
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<std::uint16_t>(port));
    if (inet_pton(AF_INET, ip.c_str(), &address.sin_addr) != 1) {
        fail("invalid peer IP " + ip);
    }
    // One attempt per 100 ms; the budget comes from the config's
    // bootstrap_timeout_s so slow-starting peers (a long host store
    // populate before the listener opens) can be given more than the
    // former hard-coded 180 s.
    const int max_attempts = timeout_s * 10;
    for (int attempt = 0; attempt < max_attempts && !stop.load(); ++attempt) {
        const int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) {
            fail("socket: " + std::string(strerror(errno)));
        }
        if (connect(fd, reinterpret_cast<sockaddr*>(&address),
                    sizeof(address)) == 0) {
            return fd;
        }
        const int error = errno;
        close(fd);
        if (error != ECONNREFUSED && error != ETIMEDOUT &&
            error != EHOSTUNREACH && error != ENETUNREACH) {
            fail("connect " + ip + ":" + std::to_string(port) +
                 ": " + strerror(error));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    fail("timed out connecting to " + ip + ":" +
         std::to_string(port));
}

ucp_ep_h create_endpoint(ucp_worker_h worker,
                         const void* remote_address) {
    ucp_ep_params_t params{};
    params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    params.address =
        static_cast<const ucp_address_t*>(remote_address);
    ucp_ep_h endpoint = nullptr;
    check_ucs(ucp_ep_create(worker, &params, &endpoint),
              "ucp_ep_create");
    return endpoint;
}

void wait_request(ucp_worker_h worker, void* request) {
    if (request == nullptr) {
        return;
    }
    if (UCS_PTR_IS_ERR(request)) {
        check_ucs(UCS_PTR_STATUS(request), "UCP request");
    }
    while (true) {
        const ucs_status_t status =
            ucp_request_check_status(request);
        if (status == UCS_INPROGRESS) {
            ucp_worker_progress(worker);
            std::this_thread::yield();
            continue;
        }
        check_ucs(status, "UCP request completion");
        ucp_request_free(request);
        return;
    }
}

void close_endpoint(ucp_worker_h worker, ucp_ep_h endpoint) {
    if (endpoint == nullptr) {
        return;
    }
    ucp_request_param_t params{};
    wait_request(worker, ucp_ep_close_nbx(endpoint, &params));
}

// Spread each owner's eight egress rails over the (num_nodes - 1) nodes that
// pull from it. With four nodes this is the validated (rail - 1) % 3 walk;
// with two nodes the single peer takes every rail.
int destination_for_owner_rail(int owner, int rail, int num_nodes) {
    return (owner + 1 + (rail - 1) % (num_nodes - 1)) % num_nodes;
}

double elapsed_seconds(Clock::time_point begin,
                       Clock::time_point end = Clock::now()) {
    return std::chrono::duration<double>(end - begin).count();
}

std::pair<int, int> parse_module_key(const std::string& key) {
    int layer = -1;
    int expert = -1;
    char tail = '\0';
    if (std::sscanf(key.c_str(), "routed_expert_%d_%d%c",
                    &layer, &expert, &tail) != 2 ||
        layer < 1 || layer > kLayers ||
        expert < 0 || expert >= kExperts) {
        fail("invalid routed expert module key: " + key);
    }
    return {layer, expert};
}

struct BootstrapHeader {
    std::uint64_t magic = kBootstrapMagic;
    std::uint64_t worker_address_bytes = 0;
    std::uint64_t rkey_bytes = 0;
    std::uint64_t source_address = 0;
    std::uint64_t source_bytes = 0;
};

struct UcpRail {
    ucp_context_h context = nullptr;
    ucp_worker_h worker = nullptr;
    ucp_mem_h source_memh = nullptr;
    ucp_mem_h staging_memh = nullptr;
    ucp_address_t* address = nullptr;
    size_t address_bytes = 0;
};

struct RemoteEndpoint {
    ucp_ep_h endpoint = nullptr;
    ucp_rkey_h rkey = nullptr;
    int control_fd = -1;
    std::uint64_t source_address = 0;
    std::uint64_t source_bytes = 0;
};

enum class SlotState {
    kFree,
    kFetching,
    kReady,
};

struct Slot {
    SlotState state = SlotState::kFree;
    std::string module_key;
    std::uint64_t generation = 0;
    std::uint32_t acquired_mask = 0;
    std::uint32_t released_mask = 0;
};

// The acquire a worker connection is retrying, kept across its slices so the
// timeout and its diagnostics see the whole wait. Empty key: none.
struct PendingAcquire {
    std::string module_key;
    Clock::time_point begin;
    Clock::time_point last_poll;
    std::uint64_t completions_at_start = 0;
};

struct Chunk {
    int slot_start = 0;
    int count = 0;
    int owner = -1;
    int rail = -1;
    void* request = nullptr;
    Clock::time_point posted;
    // When ucp_get_nbx was actually issued. `posted` is taken before the ring
    // bookkeeping (a ring_mutex acquisition and an O(depth) occupancy scan)
    // that precedes the call, so posted..submitted is daemon overhead, not
    // transport. Separating them keeps the achieved-bandwidth figure honest.
    Clock::time_point submitted;
    // Identity of the chunk's first module, carried so the trace can attribute
    // a get to a layer and to the worker that will acquire it.
    int layer = -1;
    int first_expert = -1;
    int target_worker = -1;
    // Ring census taken under ring_mutex at the moment this get was posted.
    int occupied_slots = -1;
    int ready_unacquired_slots = -1;
    int leased_slots = -1;
};

// One ucp_get_nbx, start to publication. POD and fixed size: the fetch path
// appends with one relaxed fetch_add into a preallocated ring, so tracing adds
// no allocation, no lock and no I/O to the hot path. All times are ns since the
// daemon's trace epoch.
//
// done_ns is when the polling loop OBSERVED completion, not when the NIC
// finished; prev_scan_ns is the start of the preceding completion scan, so the
// analysis can bound how much of (done - post) is detection delay rather than
// transport. Without that bound a slow single-threaded poll loop is
// indistinguishable from a slow rail.
struct GetRecord {
    std::int64_t post_ns = 0;
    std::int64_t submit_ns = 0;
    std::int64_t done_ns = 0;
    std::int64_t publish_ns = 0;
    std::int64_t prev_scan_ns = 0;
    // Earliest moment a worker was observed blocked on any module of this
    // chunk; 0 when no worker ever blocked on it.
    std::int64_t first_wait_ns = 0;
    std::uint64_t bytes = 0;
    std::int32_t rail = -1;
    std::int32_t owner = -1;
    std::int32_t layer = -1;
    std::int32_t first_expert = -1;
    std::int32_t count = 0;
    std::int32_t target_worker = -1;
    // Who held the staging ring when this get was posted. With
    // depth == outstanding * superchunk, in-flight equals the free-slot count,
    // so these three decide whether posting is gated or the ring is held.
    std::int32_t occupied_slots = -1;
    std::int32_t ready_unacquired_slots = -1;
    std::int32_t leased_slots = -1;
};

// A worker cohort is one generation of the server's worker pool. The daemon
// admits hellos for exactly one cohort epoch at a time. Draining is the window
// between BeginCohortSwap and FinishCohortSwap: the cohort is being torn down,
// so its disconnects are expected and nothing new is admitted.
enum class CohortState {
    kAdmitting,
    kDraining,
};

const char* cohort_state_name(CohortState state) {
    return state == CohortState::kAdmitting ? "admitting" : "draining";
}

// One worker control connection. The accept thread creates it and starts its
// thread; only FinishCohortSwap or Stop joins that thread. They also close the
// fd, unless CloseUnadmitted already did and left fd == -1.
// epoch/worker_id/release_only are written by the connection's own thread
// when its hello is admitted, and read by others only after joining that
// thread. A worker has one primary connection and one release-only one; a
// swap or Stop joins both like any other connection.
struct ClientConnection {
    int fd = -1;
    std::thread thread;
    std::uint64_t epoch = 0;  // 0 until the hello is admitted
    int worker_id = -1;
    bool release_only = false;  // kHelloReleaseChannel: carries kRelease only
};

// What a ring reset found and released. Built under ring_mutex.
struct RingResetReport {
    std::uint64_t outstanding_leases = 0;  // (slot, worker) acquires never released
    std::uint64_t leased_slots = 0;
    std::uint64_t discarded_ready = 0;  // fetched, never acquired
    std::uint64_t drained_fetches = 0;  // modules in flight at the reset
    std::uint64_t schedule_cursor = 0;  // prefetch position rewound to 0
    int ring_tail = 0;                  // ring position rewound to 0
    bool prefetch_handshake = false;    // reset done by the prefetch thread
};

// The counters the summary reports. Every update is a few scalar writes, and
// the summary copies the whole struct, so stats_mutex is never held for long.
struct Stats {
    std::uint64_t fetched_modules = 0;
    std::uint64_t fetched_bytes = 0;
    std::uint64_t acquire_count = 0;
    std::uint64_t release_count = 0;
    std::uint64_t cache_hits = 0;
    std::uint64_t duplicate_fetches = 0;
    std::uint64_t wait_count = 0;
    double wait_seconds = 0.0;
    std::array<std::uint64_t, kDefaultWorkers> worker_wait_count{};
    std::array<double, kDefaultWorkers> worker_wait_seconds{};
    int staging_hwm = 0;
    // Post-to-ready latency of each fetched module, in milliseconds.
    LatencyHistogram ready_latency;
};

// What the summary reads of the cohort. Copied under cohort_mutex.
struct CohortSnapshot {
    std::uint64_t epoch = 0;
    CohortState state = CohortState::kAdmitting;
    std::uint32_t admitted_mask = 0;
    bool faulted = false;
    std::string fault;
    std::uint64_t expected_disconnects = 0;
    std::uint64_t rejected_hellos = 0;
    std::string last_rejected_hello;
    std::uint64_t completed_swaps = 0;
    nlohmann::json swap_log;
};

// What the summary reads of the ring. Copied under ring_mutex.
struct RingSnapshot {
    std::vector<Slot> slots;
    std::array<PendingAcquire, kDefaultWorkers> pending_acquires;
    Clock::time_point taken;
};

}  // namespace

struct DistributedWeightDaemon::Impl {
    explicit Impl(const std::string& path) : config_path(path) {
        std::ifstream handle(path);
        if (!handle) {
            fail("cannot open distributed weight config: " + path);
        }
        nlohmann::json config;
        handle >> config;
        node_rank = config.at("node_rank").get<int>();
        node_ips = config.at("node_ips")
                       .get<std::vector<std::string>>();
        num_nodes = static_cast<int>(node_ips.size());
        store_path = config.at("store_path").get<std::string>();
        socket_path =
            config.at("daemon_socket").get<std::string>();
        summary_path =
            config.at("summary_path").get<std::string>();
        store_bytes = config.at("store_bytes").get<std::uint64_t>();
        module_bytes =
            config.at("module_bytes").get<std::uint64_t>();
        replicated_bytes =
            config.at("replicated_bytes").get<std::uint64_t>();
        base_port = config.value("base_port", 26000);
        bootstrap_timeout_s = config.value("bootstrap_timeout_s", 180);
        if (bootstrap_timeout_s <= 0) {
            fail("distributed weight bootstrap_timeout_s must be positive");
        }
        depth = config.value("depth", 32);
        superchunk = config.value("superchunk", 8);
        outstanding = config.value("outstanding", 8);
        // Per-get fetch trace. Off unless a path is configured, so an ordinary
        // run carries no buffer and no extra work on the fetch path.
        fetch_trace_path = config.value("fetch_trace_path", std::string());
        if (!fetch_trace_path.empty()) {
            const std::int64_t capacity =
                config.value("fetch_trace_capacity",
                             static_cast<std::int64_t>(2000000));
            if (capacity <= 0) {
                fail("fetch_trace_capacity must be positive");
            }
            fetch_trace.assign(static_cast<size_t>(capacity), GetRecord{});
        }
        workers = config.value("workers", kDefaultWorkers);
        worker_sharded = config.value("worker_sharded", false);
        transport = config.value("transport", "host_rdma");
        if (transport != "host_rdma" &&
            transport != "hierarchical_gdr") {
            fail("invalid distributed weight transport: " + transport);
        }
        hierarchical_gdr = transport == "hierarchical_gdr";
        if (config.contains("rail_devices")) {
            const nlohmann::json& devices = config.at("rail_devices");
            if (!devices.is_array() || devices.size() != kRails) {
                fail("distributed weight rail_devices must contain exactly "
                     "eight non-empty strings");
            }
            rail_devices.clear();
            for (const nlohmann::json& device : devices) {
                if (!device.is_string()) {
                    fail("distributed weight rail_devices must contain "
                         "exactly eight non-empty strings");
                }
                std::string value = device.get<std::string>();
                if (value.empty()) {
                    fail("distributed weight rail_devices must contain "
                         "exactly eight non-empty strings");
                }
                rail_devices.push_back(std::move(value));
            }
        }
        if (config.contains("progress_thread_cores")) {
            const nlohmann::json& cores = config.at("progress_thread_cores");
            if (!cores.is_array() ||
                cores.size() != static_cast<size_t>(kRails)) {
                fail("distributed weight progress_thread_cores must contain "
                     "exactly " + std::to_string(kRails) +
                     " distinct core ids");
            }
            for (const nlohmann::json& entry : cores) {
                if (!entry.is_number_integer()) {
                    fail("distributed weight progress_thread_cores must "
                         "contain exactly " + std::to_string(kRails) +
                         " distinct core ids");
                }
                const std::int64_t core = entry.get<std::int64_t>();
                if (core < 0 || core >= CPU_SETSIZE) {
                    fail("distributed weight progress_thread_cores entry " +
                         std::to_string(core) + " is not a valid core id");
                }
                if (std::find(progress_thread_cores.begin(),
                              progress_thread_cores.end(),
                              static_cast<int>(core)) !=
                    progress_thread_cores.end()) {
                    fail("distributed weight progress_thread_cores entry " +
                         std::to_string(core) + " is repeated");
                }
                progress_thread_cores.push_back(static_cast<int>(core));
            }
        }

        // Eight TP8 workers per node: two nodes are world16, four are world32.
        if ((num_nodes != 2 && num_nodes != 4) || node_rank < 0 ||
            node_rank >= num_nodes) {
            fail("invalid distributed weight node topology");
        }
        experts_per_owner = kExperts / num_nodes;
        if (depth <= 0 || depth % superchunk != 0 ||
            superchunk <= 0 || outstanding <= 0 ||
            workers != kDefaultWorkers ||
            !config.contains("worker_sharded") ||
            !config.at("worker_sharded").is_boolean() ||
            !worker_sharded ||
            (worker_sharded &&
             (kExperts % workers != 0 ||
              (kExperts / workers) % superchunk != 0))) {
            fail("invalid distributed weight ring configuration");
        }
        // Each worker keeps one waiting acquire outstanding while it holds
        // up to kMaxLeasesWhileAcquiring leases. A ring every worker's leases
        // can fill leaves no slot to ready any of the awaited modules.
        if (static_cast<std::int64_t>(depth) <=
            static_cast<std::int64_t>(workers) * kMaxLeasesWhileAcquiring) {
            fail("distributed weight ring too small for receive-ahead: depth " +
                 std::to_string(depth) + " must exceed workers (" +
                 std::to_string(workers) + ") * kMaxLeasesWhileAcquiring (" +
                 std::to_string(kMaxLeasesWhileAcquiring) + ")");
        }
        staging_bytes =
            static_cast<std::uint64_t>(depth) * module_bytes;
        all_workers_mask =
            (static_cast<std::uint32_t>(1) << workers) - 1;
        slots.resize(depth);
        owner_rails.resize(num_nodes);
        for (std::vector<RemoteEndpoint>& rail_remotes : remotes) {
            rail_remotes.resize(num_nodes);
        }
    }

    ~Impl() { Stop(); }

    std::string config_path;
    int node_rank = -1;
    int num_nodes = 0;
    int experts_per_owner = 0;
    std::vector<std::string> node_ips;
    std::string store_path;
    std::string socket_path;
    std::string summary_path;
    std::uint64_t store_bytes = 0;
    std::uint64_t module_bytes = 0;
    std::uint64_t replicated_bytes = 0;
    std::uint64_t staging_bytes = 0;
    int base_port = 26000;
    int bootstrap_timeout_s = 180;
    int depth = 32;
    int superchunk = 8;
    int outstanding = 8;
    // Per-get trace. Empty path disables it; the ring is sized once at config
    // time and only ever written through fetch_trace_next.
    std::string fetch_trace_path;
    std::vector<GetRecord> fetch_trace;
    std::atomic<std::uint64_t> fetch_trace_next{0};
    // Posting stalls on the positional slot gate, and how many of those had a
    // free group elsewhere in the ring. Trace-only; reported in its header.
    std::atomic<std::uint64_t> gate_stalls{0};
    std::atomic<std::uint64_t> gate_stalls_free_elsewhere{0};
    Clock::time_point trace_epoch = Clock::now();
    int workers = kDefaultWorkers;
    bool worker_sharded = false;
    std::string transport = "host_rdma";
    bool hierarchical_gdr = false;
    std::vector<std::string> rail_devices = {
        "mlx5_bond_1:1", "mlx5_bond_2:1", "mlx5_bond_3:1",
        "mlx5_bond_4:1", "mlx5_bond_5:1", "mlx5_bond_6:1",
        "mlx5_bond_7:1", "mlx5_bond_8:1"};
    std::uint32_t all_workers_mask = 0;

    int store_fd = -1;
    int staging_fd = -1;
    int unix_listener = -1;
    void* store = nullptr;
    void* staging = nullptr;
    std::array<int, kRails> tcp_listeners = [] {
        std::array<int, kRails> listeners{};
        listeners.fill(-1);
        return listeners;
    }();
    std::array<UcpRail, kRails> rails;
    std::array<std::vector<RemoteEndpoint>, kRails> remotes;
    std::array<std::vector<ucp_ep_h>, kRails> incoming_endpoints;
    std::array<std::vector<int>, kRails> incoming_controls;
    std::vector<std::vector<int>> owner_rails;

    std::atomic<bool> started{false};
    std::atomic<bool> stop{false};
    std::atomic<bool> network_ready{false};
    std::atomic<bool> failed{false};
    std::string failure_message;
    std::mutex failure_mutex;
    std::mutex ready_mutex;
    std::condition_variable ready_cv;

    std::thread bootstrap_thread;
    std::thread unix_accept_thread;
    std::thread prefetch_thread;
    std::thread summary_thread;
    std::array<std::thread, kRails> progress_threads;
    // Gap accounting for each rail's progress loop: the time between two
    // consecutive iterations. The loop otherwise spins, so a large gap means
    // the thread was off the CPU or one progress call ran long (a burst of
    // completions, an endpoint teardown) — never that the rail was idle; a
    // starved thread shows up as rare large gaps while the loop count stays
    // high. Each counter is written by its own progress thread and read by
    // the summary thread, so plain relaxed atomics suffice; the alignment
    // keeps neighbouring rails off each other's cache line, since every
    // iteration writes `loops`.
    struct alignas(64) ProgressGaps {
        std::atomic<std::uint64_t> loops{0};
        std::atomic<std::uint64_t> gap_ge_1ms{0};
        std::atomic<std::uint64_t> gap_ge_10ms{0};
        std::atomic<std::uint64_t> gap_ge_50ms{0};
        std::atomic<std::uint64_t> max_gap_us{0};
    };
    std::array<ProgressGaps, kRails> progress_gaps;
    // One core id per rail when `progress_thread_cores` is configured; empty
    // (the default) leaves every progress thread unpinned. A core the kernel
    // rejects at pin time records a daemon failure and stops the daemon: a
    // configured pin is a correctness condition, not a hint.
    std::vector<int> progress_thread_cores;
    std::vector<std::unique_ptr<ClientConnection>> clients;
    std::mutex client_mutex;

    // Lock order, outermost first. No path holds two of these at once except
    // where listed, and none is held across a thread join or WriteSummary:
    //   summary_mutex -> at most one of cohort_mutex, ring_mutex,
    //                    stats_mutex, failure_mutex at a time (WriteSummary
    //                    takes one snapshot after another)
    //   ring_mutex -> stats_mutex (MarkReady, PrefetchLoop); every
    //                 stats_mutex hold is a few scalar updates or one copy
    //                 of stats
    //   ready_mutex -> failure_mutex (WaitReady)
    //   client_mutex, cohort_mutex: leaves
    // cohort_closed mirrors (cohort_state == kDraining || cohort_faulted) so
    // the ring and ready predicates can read it without taking cohort_mutex;
    // every writer stores it and then calls WakeWaiters().
    std::mutex cohort_mutex;
    std::uint64_t cohort_epoch = 1;
    CohortState cohort_state = CohortState::kAdmitting;
    std::uint32_t cohort_admitted_mask = 0;
    // Admitted workers that also opened their release-only connection.
    std::uint32_t cohort_release_mask = 0;
    bool cohort_faulted = false;
    std::string cohort_fault;
    std::uint64_t cohort_expected_disconnects = 0;
    std::uint64_t rejected_hellos = 0;
    std::string last_rejected_hello;
    std::uint64_t completed_swaps = 0;
    nlohmann::json swap_log = nlohmann::json::array();
    std::atomic<bool> cohort_closed{false};

    std::mutex ring_mutex;
    std::condition_variable ring_cv;
    std::vector<Slot> slots;
    std::unordered_map<std::string, int> module_to_slot;
    std::uint64_t next_generation = 1;
    // Completed fetches, so an acquire timeout can tell a stalled transport
    // (no completion while slots are fetching) from a cohort that never
    // consumed what the ring holds. Under ring_mutex.
    std::uint64_t ring_completions = 0;
    // Per worker id; a cohort admits one connection per worker. Under
    // ring_mutex.
    std::array<PendingAcquire, kDefaultWorkers> pending_acquires;
    // Trace only: when a worker first blocked on a module key, so a get can be
    // split into "waited to be posted" and "in flight". Written on the acquire
    // retry path, which already holds ring_mutex, and erased when the module
    // publishes, so it holds at most the keys workers are blocked on now.
    std::unordered_map<std::string, Clock::time_point> wait_begin;
    // Ring reset handshake. Once prefetch_owns_ring is set, only the prefetch
    // thread resets slots, at a point where it has no fetch in flight. All
    // under ring_mutex.
    bool prefetch_owns_ring = false;
    std::uint64_t ring_reset_requested = 0;
    std::uint64_t ring_reset_completed = 0;
    RingResetReport last_ring_reset;

    std::mutex stats_mutex;
    std::mutex summary_mutex;
    Stats stats;  // under stats_mutex

    // Wake every condition waiter after a flag its predicate reads changed.
    // Taking each mutex once orders that store before the waiter's next
    // predicate check, so a waiter between its check and its sleep cannot
    // miss the wakeup. The caller must hold neither mutex.
    void WakeWaiters() {
        { std::lock_guard<std::mutex> lock(ring_mutex); }
        ring_cv.notify_all();
        { std::lock_guard<std::mutex> lock(ready_mutex); }
        ready_cv.notify_all();
    }

    // Transport-fatal: the daemon's own cross-node state (bootstrap, UCX
    // fetches, ring bookkeeping) is broken. This latch is permanent; no
    // cohort swap clears it. Worker-side failures go to RecordWorkerLoss.
    void RecordFailure(const std::string& message) {
        {
            std::lock_guard<std::mutex> lock(failure_mutex);
            if (!failed.exchange(true)) {
                failure_message = message;
            }
        }
        stop.store(true);
        WakeWaiters();
        WriteSummary();
    }

    std::string TransportFatalMessage() {
        return "distributed weights transport failure is not recoverable "
               "by a worker restart (cold restart required): " +
               FailureMessage();
    }

    // Cohort-recoverable: one admitted worker's control connection ended.
    // While a swap drains that cohort this is the expected teardown. Outside
    // a swap it fails the cohort (its other workers' acquires fail fast and no
    // hello of that epoch is admitted again) but leaves the transport, the
    // accept loop and the prefetch ring running for the next cohort.
    void RecordWorkerLoss(std::uint64_t epoch, int worker_id,
                          const std::string& message) {
        if (stop.load() || failed.load()) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(cohort_mutex);
            if (epoch != cohort_epoch) {
                // FinishCohortSwap joins a cohort's connections before it
                // advances the epoch, so this cannot happen.
                return;
            }
            if (cohort_state == CohortState::kDraining) {
                ++cohort_expected_disconnects;
                return;
            }
            if (!cohort_faulted) {
                cohort_faulted = true;
                cohort_fault = "worker " + std::to_string(worker_id) +
                               " control connection failed: " + message;
            }
            cohort_closed.store(true);
        }
        WakeWaiters();
        WriteSummary();
    }

    void RecordRejectedHello(const std::string& message) {
        std::lock_guard<std::mutex> lock(cohort_mutex);
        ++rejected_hellos;
        last_rejected_hello = message;
    }

    std::string CohortClosedMessage() {
        std::lock_guard<std::mutex> lock(cohort_mutex);
        if (cohort_state == CohortState::kDraining) {
            return "worker cohort " + std::to_string(cohort_epoch) +
                   " is being swapped out";
        }
        if (cohort_faulted) {
            return "worker cohort " + std::to_string(cohort_epoch) +
                   " failed (cohort-recoverable, needs a worker restart): " +
                   cohort_fault;
        }
        return "worker cohort " + std::to_string(cohort_epoch) +
               " is closed";
    }

    std::string FailureMessage() {
        std::lock_guard<std::mutex> lock(failure_mutex);
        return failure_message;
    }

    void AllocateMappings() {
        store_fd = open(store_path.c_str(), O_RDWR);
        if (store_fd < 0) {
            fail("cannot open compact store O_RDWR: " + store_path +
                 ": " + strerror(errno));
        }
        struct stat store_stat {};
        if (fstat(store_fd, &store_stat) != 0 ||
            static_cast<std::uint64_t>(store_stat.st_size) !=
                store_bytes) {
            fail("compact store byte size mismatch");
        }
        store = mmap(nullptr, store_bytes, PROT_READ | PROT_WRITE,
                     MAP_SHARED, store_fd, 0);
        if (store == MAP_FAILED) {
            store = nullptr;
            fail("compact store mmap failed: " +
                 std::string(strerror(errno)));
        }
        // Huge pages for the staged tmpfs store, as for the staging ring
        // below; failure only loses them.
        madvise(store, store_bytes, MADV_HUGEPAGE);

        staging_fd = static_cast<int>(
            syscall(SYS_memfd_create, "k3_roce_staging", 0));
        if (staging_fd < 0) {
            fail("staging memfd_create failed: " +
                 std::string(strerror(errno)));
        }
        if (ftruncate(staging_fd,
                      static_cast<off_t>(staging_bytes)) != 0) {
            fail("staging ftruncate failed: " +
                 std::string(strerror(errno)));
        }
        staging = mmap(nullptr, staging_bytes,
                       PROT_READ | PROT_WRITE, MAP_SHARED,
                       staging_fd, 0);
        if (staging == MAP_FAILED) {
            staging = nullptr;
            fail("staging mmap failed: " +
                 std::string(strerror(errno)));
        }
        madvise(staging, staging_bytes, MADV_HUGEPAGE);
        if (numa_available() < 0) {
            fail("NUMA is unavailable for distributed staging");
        }
        struct bitmask* previous = numa_get_interleave_mask();
        numa_set_interleave_mask(numa_all_nodes_ptr);
        std::memset(staging, 0, staging_bytes);
        numa_set_interleave_mask(previous);
        numa_free_nodemask(previous);
    }

    UcpRail CreateRail(int rail_index) {
        UcpRail result;
        ucp_config_t* config = nullptr;
        check_ucs(ucp_config_read(nullptr, nullptr, &config),
                  "ucp_config_read");
        const std::string& device = rail_devices.at(rail_index);
        check_ucs(ucp_config_modify(config, "TLS", "rc_x,self"),
                  "ucp_config_modify TLS");
        check_ucs(
            ucp_config_modify(config, "NET_DEVICES", device.c_str()),
            "ucp_config_modify NET_DEVICES");
        ucp_params_t params{};
        params.field_mask = UCP_PARAM_FIELD_FEATURES;
        params.features = UCP_FEATURE_RMA;
        const ucs_status_t init_status =
            ucp_init(&params, config, &result.context);
        ucp_config_release(config);
        check_ucs(init_status, "ucp_init");

        ucp_worker_params_t worker_params{};
        worker_params.field_mask =
            UCP_WORKER_PARAM_FIELD_THREAD_MODE;
        worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
        check_ucs(
            ucp_worker_create(result.context, &worker_params,
                              &result.worker),
            "ucp_worker_create");
        check_ucs(
            ucp_worker_get_address(result.worker, &result.address,
                                   &result.address_bytes),
            "ucp_worker_get_address");

        auto map = [&](void* address, std::uint64_t bytes,
                       ucp_mem_h* memory) {
            ucp_mem_map_params_t map_params{};
            map_params.field_mask =
                UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
            map_params.address = address;
            map_params.length = bytes;
            map_params.memory_type = UCS_MEMORY_TYPE_HOST;
            check_ucs(
                ucp_mem_map(result.context, &map_params, memory),
                "ucp_mem_map");
        };
        map(store, store_bytes, &result.source_memh);
        map(staging, staging_bytes, &result.staging_memh);
        return result;
    }

    void DestroyRail(UcpRail* rail) {
        if (rail->address != nullptr) {
            ucp_worker_release_address(rail->worker, rail->address);
            rail->address = nullptr;
        }
        if (rail->staging_memh != nullptr) {
            ucp_mem_unmap(rail->context, rail->staging_memh);
            rail->staging_memh = nullptr;
        }
        if (rail->source_memh != nullptr) {
            ucp_mem_unmap(rail->context, rail->source_memh);
            rail->source_memh = nullptr;
        }
        if (rail->worker != nullptr) {
            ucp_worker_destroy(rail->worker);
            rail->worker = nullptr;
        }
        if (rail->context != nullptr) {
            ucp_cleanup(rail->context);
            rail->context = nullptr;
        }
    }

    void SetupUnixListener() {
        unlink(socket_path.c_str());
        unix_listener = socket(AF_UNIX, SOCK_STREAM, 0);
        if (unix_listener < 0) {
            fail("unix socket failed: " +
                 std::string(strerror(errno)));
        }
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        if (socket_path.size() >= sizeof(address.sun_path)) {
            fail("unix socket path is too long");
        }
        std::strncpy(address.sun_path, socket_path.c_str(),
                     sizeof(address.sun_path) - 1);
        if (bind(unix_listener,
                 reinterpret_cast<sockaddr*>(&address),
                 sizeof(address)) != 0) {
            fail("unix bind failed: " +
                 std::string(strerror(errno)));
        }
        chmod(socket_path.c_str(), 0600);
        // Two connections per worker: the primary and the release-only one.
        if (listen(unix_listener, 2 * workers) != 0) {
            fail("unix listen failed: " +
                 std::string(strerror(errno)));
        }
    }

    void Start() {
        if (started.exchange(true)) {
            return;
        }
        try {
            AllocateMappings();
            SetupUnixListener();
            if (hierarchical_gdr) {
                // The compact stores remain quarter-sharded, but remote
                // shards move GPU-to-GPU through NCCL. Workers still use the
                // daemon's local Unix handshake/staging ABI; no UCP endpoint,
                // host-RDMA prefetch, or remote host registration is needed.
                network_ready.store(true);
                ready_cv.notify_all();
            } else {
                bootstrap_thread = std::thread([this]() {
                    try {
                        BootstrapNetwork();
                    } catch (const std::exception& error) {
                        if (!stop.load()) {
                            RecordFailure(error.what());
                        }
                    }
                });
            }
            unix_accept_thread = std::thread([this]() {
                try {
                    AcceptWorkers();
                } catch (const std::exception& error) {
                    if (!stop.load()) {
                        RecordFailure(error.what());
                    }
                }
            });
            summary_thread = std::thread([this]() {
                while (!stop.load()) {
                    WriteSummary();
                    for (int i = 0; i < 10 && !stop.load(); ++i) {
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(100));
                    }
                }
                WriteSummary();
            });
        } catch (...) {
            started.store(false);
            throw;
        }
    }

    void BootstrapNetwork() {
        for (int rail = 0; rail < kRails; ++rail) {
            rails[rail] = CreateRail(rail);
            tcp_listeners[rail] = create_listener(
                node_ips[node_rank],
                base_port + node_rank * kRails + rail);
        }

        std::array<std::thread, kRails> accept_threads;
        auto join_accept_threads = [&]() {
            for (std::thread& thread : accept_threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
        };
        auto close_tcp_listeners = [&]() {
            for (int& listener : tcp_listeners) {
                if (listener >= 0) {
                    shutdown(listener, SHUT_RDWR);
                    close(listener);
                    listener = -1;
                }
            }
        };
        try {
            for (int rail = 0; rail < kRails; ++rail) {
                accept_threads[rail] = std::thread([this, rail]() {
                    try {
                        const int fd = accept(tcp_listeners[rail],
                                              nullptr, nullptr);
                        if (fd < 0) {
                            if (stop.load()) {
                                return;
                            }
                            fail("owner accept failed: " +
                                 std::string(strerror(errno)));
                        }
                        std::uint64_t client_address_bytes = 0;
                        recv_exact(fd, &client_address_bytes,
                                   sizeof(client_address_bytes));
                        std::vector<unsigned char> client_address(
                            client_address_bytes);
                        recv_exact(fd, client_address.data(),
                                   client_address.size());
                        ucp_ep_h endpoint = create_endpoint(
                            rails[rail].worker, client_address.data());
                        incoming_endpoints[rail].push_back(endpoint);
                        incoming_controls[rail].push_back(fd);

                        void* packed_rkey = nullptr;
                        size_t packed_rkey_bytes = 0;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
                        check_ucs(
                            ucp_rkey_pack(
                                rails[rail].context,
                                rails[rail].source_memh,
                                &packed_rkey, &packed_rkey_bytes),
                            "ucp_rkey_pack");
#pragma GCC diagnostic pop
                        BootstrapHeader header;
                        header.worker_address_bytes =
                            rails[rail].address_bytes;
                        header.rkey_bytes = packed_rkey_bytes;
                        header.source_address =
                            reinterpret_cast<std::uint64_t>(store);
                        header.source_bytes = store_bytes;
                        send_exact(fd, &header, sizeof(header));
                        send_exact(fd, rails[rail].address,
                                   rails[rail].address_bytes);
                        send_exact(fd, packed_rkey, packed_rkey_bytes);
                        ucp_rkey_buffer_release(packed_rkey);
                    } catch (const std::exception& error) {
                        if (!stop.load()) {
                            RecordFailure(
                                "owner bootstrap failed: " +
                                std::string(error.what()));
                        }
                    }
                });
            }

            for (int rail = 0; rail < kRails; ++rail) {
                for (int owner = 0; owner < num_nodes; ++owner) {
                    if (owner == node_rank ||
                        destination_for_owner_rail(owner, rail + 1,
                                                   num_nodes) != node_rank) {
                        continue;
                    }
                    const int fd = connect_retry(
                        node_ips[owner],
                        base_port + owner * kRails + rail,
                        bootstrap_timeout_s, stop);
                    const std::uint64_t address_bytes =
                        rails[rail].address_bytes;
                    send_exact(fd, &address_bytes,
                               sizeof(address_bytes));
                    send_exact(fd, rails[rail].address,
                               rails[rail].address_bytes);

                    BootstrapHeader header{};
                    recv_exact(fd, &header, sizeof(header));
                    if (header.magic != kBootstrapMagic ||
                        header.source_bytes != store_bytes) {
                        fail("invalid distributed weight bootstrap");
                    }
                    std::vector<unsigned char> owner_address(
                        header.worker_address_bytes);
                    std::vector<unsigned char> packed_rkey(
                        header.rkey_bytes);
                    recv_exact(fd, owner_address.data(),
                               owner_address.size());
                    recv_exact(fd, packed_rkey.data(),
                               packed_rkey.size());
                    RemoteEndpoint remote;
                    remote.endpoint = create_endpoint(
                        rails[rail].worker, owner_address.data());
                    check_ucs(
                        ucp_ep_rkey_unpack(
                            remote.endpoint, packed_rkey.data(),
                            &remote.rkey),
                        "ucp_ep_rkey_unpack");
                    remote.control_fd = fd;
                    remote.source_address = header.source_address;
                    remote.source_bytes = header.source_bytes;
                    remotes[rail][owner] = remote;
                    owner_rails[owner].push_back(rail);
                }
            }

            join_accept_threads();
        } catch (const std::exception& error) {
            if (!stop.load()) {
                RecordFailure("distributed weight bootstrap failed: " +
                              std::string(error.what()));
            }
            close_tcp_listeners();
            join_accept_threads();
            throw;
        } catch (...) {
            if (!stop.load()) {
                RecordFailure("distributed weight bootstrap failed: "
                              "unknown exception");
            }
            close_tcp_listeners();
            join_accept_threads();
            throw;
        }
        if (failed.load()) {
            fail(FailureMessage());
        }
        if (stop.load()) {
            return;
        }
        // kRails rails spread over the (num_nodes - 1) peers of each owner:
        // 2/3 endpoints on four nodes, all 8 on two nodes.
        const size_t min_rails =
            static_cast<size_t>(kRails / (num_nodes - 1));
        const size_t max_rails = static_cast<size_t>(
            (kRails + num_nodes - 2) / (num_nodes - 1));
        for (int owner = 0; owner < num_nodes; ++owner) {
            if (owner == node_rank) {
                continue;
            }
            if (owner_rails[owner].size() < min_rails ||
                owner_rails[owner].size() > max_rails) {
                fail("owner rail mapping is not " +
                     std::to_string(min_rails) + "/" +
                     std::to_string(max_rails) + " endpoints");
            }
        }
        for (int rail = 0; rail < kRails; ++rail) {
            progress_threads[rail] = std::thread([this, rail]() {
                if (!progress_thread_cores.empty()) {
                    cpu_set_t cores;
                    CPU_ZERO(&cores);
                    CPU_SET(progress_thread_cores[rail], &cores);
                    const int rc = pthread_setaffinity_np(
                        pthread_self(), sizeof(cores), &cores);
                    if (rc != 0) {
                        // A configured pin that cannot be applied must not
                        // degrade silently into the unpinned behaviour the
                        // operator asked to leave.
                        RecordFailure(
                            "distributed weight progress thread for rail " +
                            std::to_string(rail) +
                            " could not be pinned to core " +
                            std::to_string(progress_thread_cores[rail]) +
                            ": " + std::string(std::strerror(rc)));
                        return;
                    }
                }
                ProgressGaps& gaps = progress_gaps[rail];
                Clock::time_point previous = Clock::now();
                while (!stop.load()) {
                    if (ucp_worker_progress(rails[rail].worker) == 0) {
                        std::this_thread::yield();
                    }
                    const Clock::time_point now = Clock::now();
                    const std::uint64_t gap_us = static_cast<std::uint64_t>(
                        std::chrono::duration_cast<std::chrono::microseconds>(
                            now - previous).count());
                    previous = now;
                    gaps.loops.fetch_add(1, std::memory_order_relaxed);
                    if (gap_us >= 1000) {
                        gaps.gap_ge_1ms.fetch_add(1, std::memory_order_relaxed);
                        if (gap_us >= 10000) {
                            gaps.gap_ge_10ms.fetch_add(
                                1, std::memory_order_relaxed);
                        }
                        if (gap_us >= 50000) {
                            gaps.gap_ge_50ms.fetch_add(
                                1, std::memory_order_relaxed);
                        }
                    }
                    std::uint64_t seen =
                        gaps.max_gap_us.load(std::memory_order_relaxed);
                    while (gap_us > seen &&
                           !gaps.max_gap_us.compare_exchange_weak(
                               seen, gap_us, std::memory_order_relaxed)) {
                    }
                }
            });
        }
        {
            // From here on only the prefetch thread may reset ring slots: it
            // is the one that knows which fetches are still in flight.
            std::lock_guard<std::mutex> lock(ring_mutex);
            prefetch_owns_ring = true;
        }
        network_ready.store(true);
        ready_cv.notify_all();
        prefetch_thread = std::thread([this]() {
            try {
                PrefetchLoop();
            } catch (const std::exception& error) {
                RecordFailure(error.what());
            }
        });
    }

    void WaitReady(double timeout_seconds) {
        std::unique_lock<std::mutex> lock(ready_mutex);
        const bool done = ready_cv.wait_for(
            lock, std::chrono::duration<double>(timeout_seconds),
            [this]() {
                return network_ready.load() || failed.load();
            });
        if (!done) {
            fail("timed out waiting for distributed weights network");
        }
        if (failed.load()) {
            fail("distributed weights daemon failed: " +
                 FailureMessage());
        }
    }

    void AcceptWorkers() {
        while (!stop.load()) {
            const int fd = accept(unix_listener, nullptr, nullptr);
            if (fd < 0) {
                if (stop.load() || errno == EBADF || errno == EINVAL) {
                    return;
                }
                if (errno == EINTR) {
                    continue;
                }
                fail("unix accept failed: " +
                     std::string(strerror(errno)));
            }
            {
                std::lock_guard<std::mutex> lock(client_mutex);
                auto connection = std::make_unique<ClientConnection>();
                connection->fd = fd;
                ClientConnection* handle = connection.get();
                clients.push_back(std::move(connection));
                handle->thread = std::thread(
                    [this, handle]() { HandleWorker(handle); });
            }
        }
    }

    // Admit a hello into the current cohort, or say why not. A refused hello
    // never joins the cohort, so refusing it is not a failure of anything.
    std::string AdmitWorker(const Request& request,
                            ClientConnection* connection) {
        if (request.magic !=
                batchgen::distributed_weights::kProtocolMagic ||
            request.version !=
                batchgen::distributed_weights::kProtocolVersion ||
            request.operation !=
                static_cast<std::uint32_t>(Operation::kHello) ||
            request.worker_id >= static_cast<std::uint32_t>(workers)) {
            return "invalid distributed weights worker hello (protocol "
                   "version " +
                   std::to_string(request.version) + ", expected " +
                   std::to_string(
                       batchgen::distributed_weights::kProtocolVersion) +
                   ")";
        }
        if ((request.flags & ~kHelloReleaseChannel) != 0) {
            return "invalid distributed weights worker hello flags " +
                   std::to_string(request.flags);
        }
        const bool release_only =
            (request.flags & kHelloReleaseChannel) != 0;
        if (failed.load()) {
            return TransportFatalMessage();
        }
        if (stop.load()) {
            return "distributed weights daemon is stopping";
        }
        std::lock_guard<std::mutex> lock(cohort_mutex);
        const std::string worker =
            "worker " + std::to_string(request.worker_id);
        if (cohort_state == CohortState::kDraining) {
            return worker + " hello refused: worker cohort " +
                   std::to_string(cohort_epoch) + " is being swapped out";
        }
        if (request.generation != cohort_epoch) {
            return worker + " hello names worker cohort " +
                   std::to_string(request.generation) +
                   " but the daemon admits cohort " +
                   std::to_string(cohort_epoch) +
                   " (a stale or misconfigured worker)";
        }
        if (cohort_faulted) {
            return worker + " hello refused: worker cohort " +
                   std::to_string(cohort_epoch) + " already failed (" +
                   cohort_fault + "); a worker restart is required";
        }
        const std::uint32_t bit = static_cast<std::uint32_t>(1)
                                  << request.worker_id;
        if (release_only) {
            // The second connection of a worker this epoch already admitted:
            // it sets no admitted bit and is sent no staging memfd.
            if ((cohort_admitted_mask & bit) == 0) {
                return worker + " release connection refused: the worker is "
                                "not admitted in worker cohort " +
                       std::to_string(cohort_epoch);
            }
            if ((cohort_release_mask & bit) != 0) {
                return worker + " already has a release connection in worker "
                                "cohort " +
                       std::to_string(cohort_epoch);
            }
            cohort_release_mask |= bit;
            connection->release_only = true;
        } else {
            if ((cohort_admitted_mask & bit) != 0) {
                return worker + " is already connected in worker cohort " +
                       std::to_string(cohort_epoch);
            }
            cohort_admitted_mask |= bit;
        }
        connection->epoch = cohort_epoch;
        connection->worker_id = static_cast<int>(request.worker_id);
        return "";
    }

    // WaitReady for a connection: also returns once the worker's cohort is
    // closed, so a swap never has to wait out the network timeout to join it.
    void WaitReadyForWorker(double timeout_seconds) {
        std::unique_lock<std::mutex> lock(ready_mutex);
        const bool done = ready_cv.wait_for(
            lock, std::chrono::duration<double>(timeout_seconds),
            [this]() {
                return network_ready.load() || failed.load() ||
                       stop.load() || cohort_closed.load();
            });
        lock.unlock();
        if (!done) {
            fail("timed out waiting for distributed weights network");
        }
        if (failed.load()) {
            fail("distributed weights daemon failed: " + FailureMessage());
        }
        if (stop.load()) {
            fail("distributed weights daemon is stopping");
        }
        if (!network_ready.load()) {
            fail(CohortClosedMessage());
        }
    }

    void HandleWorker(ClientConnection* connection) {
        const int fd = connection->fd;
        bool admitted = false;
        bool refused = false;
        std::uint64_t epoch = 0;
        int worker_id = -1;
        try {
            Request request{};
            recv_exact(fd, &request, sizeof(request));
            const std::string refusal = AdmitWorker(request, connection);
            if (!refusal.empty()) {
                refused = true;
                RecordRejectedHello(refusal);
                Response response;
                set_error(&response, refusal);
                send_exact(fd, &response, sizeof(response));
                CloseUnadmitted(connection);
                return;
            }
            admitted = true;
            epoch = connection->epoch;
            worker_id = connection->worker_id;
            const bool release_only = connection->release_only;
            Response hello;
            hello.staging_bytes = staging_bytes;
            hello.module_bytes = module_bytes;
            hello.generation = epoch;
            if (release_only) {
                // The worker's primary connection already waited for the
                // network and holds the staging memfd.
                send_exact(fd, &hello, sizeof(hello));
            } else {
                WaitReadyForWorker(600.0);
                send_response_with_fd(fd, hello, staging_fd);
            }

            while (!stop.load()) {
                Request operation{};
                recv_exact(fd, &operation, sizeof(operation));
                Response response;
                try {
                    if (operation.magic !=
                            batchgen::distributed_weights::kProtocolMagic ||
                        operation.version !=
                            batchgen::distributed_weights::kProtocolVersion ||
                        operation.worker_id != request.worker_id ||
                        (operation.flags & ~kAcquireNoWait) != 0) {
                        fail("invalid distributed weight request");
                    }
                    if (release_only &&
                        operation.operation !=
                            static_cast<std::uint32_t>(Operation::kRelease)) {
                        fail("distributed weights protocol error: the release "
                             "connection of worker " +
                             std::to_string(worker_id) +
                             " carries kRelease only, got operation " +
                             std::to_string(operation.operation));
                    }
                    const std::string module_key(
                        operation.module_key,
                        strnlen(operation.module_key,
                                sizeof(operation.module_key)));
                    if (operation.operation ==
                        static_cast<std::uint32_t>(
                            Operation::kAcquire)) {
                        Acquire(module_key, worker_id,
                                (operation.flags & kAcquireNoWait) != 0,
                                &response);
                    } else if (operation.operation ==
                               static_cast<std::uint32_t>(
                                   Operation::kRelease)) {
                        Release(module_key, worker_id,
                                operation.generation, &response);
                    } else {
                        fail("unknown distributed weight operation");
                    }
                } catch (const std::exception& error) {
                    set_error(&response, error.what());
                }
                send_exact(fd, &response, sizeof(response));
                // kStatusRetry is an acquire slice that found its module not
                // ready yet; the worker re-sends it.
                if (response.status != 0 && response.status != kStatusRetry) {
                    fail(response.error);
                }
            }
        } catch (const std::exception& error) {
            // A transport failure was already latched by whoever detected it
            // (RecordFailure); everything that ends an admitted connection
            // here is a cohort event.
            if (admitted) {
                RecordWorkerLoss(epoch, worker_id, error.what());
            } else if (!refused) {
                RecordRejectedHello("hello failed: " +
                                    std::string(error.what()));
            }
            if (!admitted) {
                CloseUnadmitted(connection);
            }
        }
    }

    // A refused or failed hello never joins a cohort, so its socket is closed
    // now instead of lingering until the next swap or Stop. Only while the
    // connection is still listed: once FinishCohortSwap or Stop has taken it,
    // they own the fd, and closing it here could hit a reused fd number.
    void CloseUnadmitted(ClientConnection* connection) {
        std::lock_guard<std::mutex> lock(client_mutex);
        for (const auto& listed : clients) {
            if (listed.get() == connection && connection->fd >= 0) {
                close(connection->fd);
                connection->fd = -1;
                return;
            }
        }
    }

    // no_wait: answer kStatusRetry at once instead of waiting one slice.
    void Acquire(const std::string& module_key, int worker_id, bool no_wait,
                 Response* response) {
        if (hierarchical_gdr) {
            fail("hierarchical_gdr forbids remote host acquire: " +
                 module_key);
        }
        const auto [layer, expert] = parse_module_key(module_key);
        const int owner = expert / experts_per_owner;
        if (owner == node_rank) {
            fail("daemon received a local module acquire: " +
                 module_key);
        }
        const std::uint32_t worker_bit =
            static_cast<std::uint32_t>(1) << worker_id;
        const std::uint32_t expected_workers =
            ExpectedWorkers(module_key);
        if ((expected_workers & worker_bit) == 0) {
            fail("worker acquired a module outside its SP shard: " +
                 module_key);
        }
        const Clock::time_point now = Clock::now();
        std::unique_lock<std::mutex> lock(ring_mutex);
        PendingAcquire& pending = pending_acquires.at(worker_id);
        if (pending.module_key != module_key ||
            now - pending.last_poll > kAcquireAbandoned) {
            pending.module_key = module_key;
            pending.begin = now;
            pending.completions_at_start = ring_completions;
        }
        const auto ready = [&]() {
            if (failed.load() || stop.load() || cohort_closed.load()) {
                return true;
            }
            const auto found = module_to_slot.find(module_key);
            return found != module_to_slot.end() &&
                   slots[found->second].state ==
                       SlotState::kReady;
        };
        // A no-wait acquire evaluates the same predicate with no slice; the
        // rest, the 300 s diagnostics included, is the same for both.
        const bool available =
            no_wait ? ready() : ring_cv.wait_for(lock, kAcquireSlice, ready);
        pending.last_poll = Clock::now();
        if (!available && pending.last_poll - pending.begin < kAcquireTimeout) {
            if (!fetch_trace.empty()) {
                // This worker is blocked on the module. Keep only the first
                // block for the key (emplace is a no-op once present); the
                // publish erases it. Already under ring_mutex, so the trace
                // adds no lock to this path.
                wait_begin.emplace(module_key, pending.begin);
            }
            response->status = kStatusRetry;
            return;
        }
        const Clock::time_point begin = pending.begin;
        const std::uint64_t completions_at_start =
            pending.completions_at_start;
        pending = PendingAcquire{};
        if (!available) {
            int fetching = 0;
            int ready = 0;
            const Slot* oldest_unacquired = nullptr;
            for (const Slot& slot : slots) {
                fetching += slot.state == SlotState::kFetching;
                ready += slot.state == SlotState::kReady;
                if (slot.state == SlotState::kReady &&
                    slot.acquired_mask == 0 &&
                    (oldest_unacquired == nullptr ||
                     slot.generation < oldest_unacquired->generation)) {
                    oldest_unacquired = &slot;
                }
            }
            const std::string oldest =
                oldest_unacquired == nullptr
                    ? ""
                    : "; oldest unacquired ready module " +
                          oldest_unacquired->module_key + " (worker " +
                          std::to_string(
                              OwningWorker(oldest_unacquired->module_key)) +
                          ")";
            const bool progressed = ring_completions != completions_at_start;
            lock.unlock();
            if (fetching > 0 && !progressed) {
                // Fetches were posted and none completed in 300 s: the
                // transport, not the cohort, stopped moving.
                const std::string message =
                    "remote fetch stalled: no fetch completed in 300 s "
                    "with " + std::to_string(fetching) +
                    " slots fetching (waiting for " + module_key + ")" +
                    oldest;
                RecordFailure(message);
                fail(message);
            }
            fail("timed out waiting for remote module " + module_key +
                 ": the ring holds " + std::to_string(ready) +
                 " ready modules this cohort did not consume" + oldest);
        }
        if (failed.load()) {
            fail(FailureMessage());
        }
        if (stop.load()) {
            fail("distributed weights daemon is stopping");
        }
        if (cohort_closed.load()) {
            lock.unlock();
            fail(CohortClosedMessage());
        }
        auto found = module_to_slot.find(module_key);
        if (found == module_to_slot.end()) {
            fail("remote module is unavailable: " + module_key);
        }
        Slot& slot = slots[found->second];
        if ((slot.acquired_mask & worker_bit) != 0) {
            fail("worker acquired remote module twice: " + module_key);
        }
        const double wait_s = elapsed_seconds(begin);
        const bool waited = wait_s > 1e-6;
        const bool cache_hit = slot.acquired_mask != 0;
        slot.acquired_mask |= worker_bit;
        response->slot = found->second;
        response->generation = slot.generation;
        response->staging_bytes = staging_bytes;
        response->module_bytes = module_bytes;
        lock.unlock();

        std::lock_guard<std::mutex> stats_lock(stats_mutex);
        ++stats.acquire_count;
        if (cache_hit) {
            ++stats.cache_hits;
        }
        if (waited) {
            ++stats.wait_count;
            stats.wait_seconds += wait_s;
            stats.worker_wait_count.at(worker_id) += 1;
            stats.worker_wait_seconds.at(worker_id) += wait_s;
        }
        (void)layer;
    }

    void Release(const std::string& module_key, int worker_id,
                 std::uint64_t generation, Response* response) {
        const std::uint32_t worker_bit =
            static_cast<std::uint32_t>(1) << worker_id;
        std::unique_lock<std::mutex> lock(ring_mutex);
        auto found = module_to_slot.find(module_key);
        if (found == module_to_slot.end()) {
            fail("release for unknown remote module: " + module_key);
        }
        Slot& slot = slots[found->second];
        if (slot.generation != generation ||
            (slot.acquired_mask & worker_bit) == 0 ||
            (slot.released_mask & worker_bit) != 0) {
            fail("invalid remote module release: " + module_key);
        }
        slot.released_mask |= worker_bit;
        response->slot = found->second;
        response->generation = generation;
        if (slot.released_mask == ExpectedWorkers(module_key)) {
            module_to_slot.erase(found);
            slot = Slot{};
            ring_cv.notify_all();
        }
        lock.unlock();
        std::lock_guard<std::mutex> stats_lock(stats_mutex);
        ++stats.release_count;
    }

    std::uint32_t ExpectedWorkers(
        const std::string& module_key) const {
        if (!worker_sharded) {
            return all_workers_mask;
        }
        const auto [layer, expert] = parse_module_key(module_key);
        const int experts_per_worker = kExperts / workers;
        const int worker = expert / experts_per_worker;
        (void)layer;
        return static_cast<std::uint32_t>(1) << worker;
    }

    // The lowest worker expected to acquire module_key.
    int OwningWorker(const std::string& module_key) const {
        const std::uint32_t expected = ExpectedWorkers(module_key);
        for (int worker = 0; worker < workers; ++worker) {
            if ((expected >> worker) & 1U) {
                return worker;
            }
        }
        return -1;
    }

    std::vector<std::string> BuildSchedule() const {
        std::vector<std::string> schedule;
        schedule.reserve(
            kLayers * (kExperts - experts_per_owner));
        if (worker_sharded) {
            const int experts_per_worker = kExperts / workers;
            // Round-robin complete superchunks across worker shards. Every
            // worker's first requested module is then within the initial ring
            // fill; a layer-major expert sweep would strand high TP ranks
            // behind get_weights()' stall timeout.
            for (int layer = 1; layer <= kLayers; ++layer) {
                for (int chunk_offset = 0;
                     chunk_offset < experts_per_worker;
                     chunk_offset += superchunk) {
                    for (int worker = 0; worker < workers; ++worker) {
                        const int first_expert =
                            worker * experts_per_worker + chunk_offset;
                        if (first_expert / experts_per_owner == node_rank) {
                            continue;
                        }
                        for (int index = 0; index < superchunk; ++index) {
                            schedule.push_back(
                                "routed_expert_" + std::to_string(layer) +
                                "_" + std::to_string(first_expert + index));
                        }
                    }
                }
            }
            return schedule;
        }
        for (int layer = 1; layer <= kLayers; ++layer) {
            for (int expert = 0; expert < kExperts; ++expert) {
                if (expert / experts_per_owner == node_rank) {
                    continue;
                }
                schedule.push_back(
                    "routed_expert_" + std::to_string(layer) + "_" +
                    std::to_string(expert));
            }
        }
        return schedule;
    }

    std::uint64_t RemoteOffset(const std::string& module_key) const {
        const auto [layer, expert] = parse_module_key(module_key);
        return replicated_bytes +
               (static_cast<std::uint64_t>(layer - 1) *
                    experts_per_owner +
                static_cast<std::uint64_t>(
                    expert % experts_per_owner)) *
                   module_bytes;
    }

    bool SlotsFree(int start, int count) {
        for (int index = 0; index < count; ++index) {
            if (slots[start + index].state != SlotState::kFree) {
                return false;
            }
        }
        return true;
    }

    std::int64_t TraceNs(Clock::time_point when) const {
        if (when == Clock::time_point{}) {
            return 0;
        }
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   when - trace_epoch)
            .count();
    }

    // Lock-free append into the preallocated ring: one relaxed fetch_add and
    // one POD store, so the fetch path never allocates, waits or does I/O for
    // the trace.
    void RecordGet(const Chunk& chunk,
                   Clock::time_point completed,
                   Clock::time_point published,
                   Clock::time_point prev_scan,
                   Clock::time_point first_wait) {
        GetRecord record;
        record.post_ns = TraceNs(chunk.posted);
        record.submit_ns = TraceNs(chunk.submitted);
        record.done_ns = TraceNs(completed);
        record.publish_ns = TraceNs(published);
        record.prev_scan_ns = TraceNs(prev_scan);
        record.first_wait_ns = TraceNs(first_wait);
        record.bytes =
            static_cast<std::uint64_t>(chunk.count) * module_bytes;
        record.rail = chunk.rail;
        record.owner = chunk.owner;
        record.layer = chunk.layer;
        record.first_expert = chunk.first_expert;
        record.count = chunk.count;
        record.target_worker = chunk.target_worker;
        record.occupied_slots = chunk.occupied_slots;
        record.ready_unacquired_slots = chunk.ready_unacquired_slots;
        record.leased_slots = chunk.leased_slots;
        const std::uint64_t index =
            fetch_trace_next.fetch_add(1, std::memory_order_relaxed);
        fetch_trace[index % fetch_trace.size()] = record;
    }

    // Dumped once, after the prefetch loop stops. CSV because a run produces
    // >100k rows and every consumer of it is columnar. A wrap is reported as a
    // count in the header, never passed off as a complete trace.
    void WriteFetchTrace() {
        if (fetch_trace.empty() || fetch_trace_path.empty()) {
            return;
        }
        const std::uint64_t written =
            fetch_trace_next.load(std::memory_order_relaxed);
        const std::uint64_t capacity = fetch_trace.size();
        const std::uint64_t start =
            written > capacity ? written - capacity : 0;
        const std::string temporary = fetch_trace_path + ".tmp";
        std::ofstream out(temporary, std::ios::trunc);
        if (!out) {
            fail("cannot open fetch trace for write: " + temporary);
        }
        out << "# node_rank=" << node_rank << " depth=" << depth
            << " superchunk=" << superchunk
            << " outstanding=" << outstanding
            << " module_bytes=" << module_bytes
            << " records=" << (written - start)
            << " dropped_by_wrap=" << start
            << " gate_stalls="
            << gate_stalls.load(std::memory_order_relaxed)
            << " gate_stalls_free_elsewhere="
            << gate_stalls_free_elsewhere.load(std::memory_order_relaxed)
            << "\n";
        out << "post_ns,submit_ns,done_ns,publish_ns,prev_scan_ns,"
               "first_wait_ns,bytes,rail,owner,layer,first_expert,count,"
               "target_worker,occupied_slots,ready_unacquired_slots,"
               "leased_slots\n";
        for (std::uint64_t index = start; index < written; ++index) {
            const GetRecord& record = fetch_trace[index % capacity];
            out << record.post_ns << ',' << record.submit_ns << ','
                << record.done_ns << ',' << record.publish_ns << ','
                << record.prev_scan_ns << ',' << record.first_wait_ns << ','
                << record.bytes << ',' << record.rail << ',' << record.owner
                << ',' << record.layer << ',' << record.first_expert << ','
                << record.count << ',' << record.target_worker << ','
                << record.occupied_slots << ','
                << record.ready_unacquired_slots << ','
                << record.leased_slots << '\n';
        }
        out.flush();
        if (!out) {
            fail("fetch trace write failed: " + temporary);
        }
        out.close();
        if (rename(temporary.c_str(), fetch_trace_path.c_str()) != 0) {
            fail("cannot publish fetch trace: " + fetch_trace_path);
        }
    }

    void MarkReady(const Chunk& chunk,
                   Clock::time_point completed,
                   Clock::time_point prev_scan = Clock::time_point{}) {
        const double latency_ms =
            elapsed_seconds(chunk.posted, completed) * 1000.0;
        const bool tracing = !fetch_trace.empty();
        Clock::time_point first_wait{};
        {
            std::lock_guard<std::mutex> lock(ring_mutex);
            for (int index = 0; index < chunk.count; ++index) {
                Slot& slot = slots[chunk.slot_start + index];
                if (tracing) {
                    // Earliest block across the chunk's modules, then forget
                    // the key so the map cannot grow with the schedule.
                    const auto waited = wait_begin.find(slot.module_key);
                    if (waited != wait_begin.end()) {
                        if (first_wait == Clock::time_point{} ||
                            waited->second < first_wait) {
                            first_wait = waited->second;
                        }
                        wait_begin.erase(waited);
                    }
                }
                if (slot.state != SlotState::kFetching) {
                    fail("fetch completed into a non-fetching slot");
                }
                slot.state = SlotState::kReady;
                const auto inserted = module_to_slot.emplace(
                    slot.module_key, chunk.slot_start + index);
                if (!inserted.second) {
                    {
                        std::lock_guard<std::mutex> stats_lock(
                            stats_mutex);
                        ++stats.duplicate_fetches;
                    }
                    fail("duplicate remote module fetch: " +
                         slot.module_key);
                }
            }
            ring_completions += chunk.count;
            ring_cv.notify_all();
        }
        if (tracing) {
            // Taken after the publish, so completion..publication (the
            // ring_mutex wait behind acquire handlers plus the slot work) is a
            // segment of its own rather than hidden inside the transfer.
            RecordGet(chunk, completed, Clock::now(), prev_scan, first_wait);
        }
        std::lock_guard<std::mutex> stats_lock(stats_mutex);
        stats.fetched_modules += chunk.count;
        stats.fetched_bytes +=
            static_cast<std::uint64_t>(chunk.count) * module_bytes;
        stats.ready_latency.record(
            latency_ms, static_cast<std::uint64_t>(chunk.count));
    }

    // Free every slot and forget every module, as at daemon start. Counts what
    // the retired cohort still held. Caller holds ring_mutex and guarantees no
    // fetch is in flight into any slot.
    RingResetReport ResetSlotsLocked() {
        RingResetReport report;
        for (Slot& slot : slots) {
            const std::uint32_t outstanding =
                slot.acquired_mask & ~slot.released_mask;
            if (outstanding != 0) {
                ++report.leased_slots;
                report.outstanding_leases +=
                    std::bitset<32>(outstanding).count();
            }
            if (slot.state == SlotState::kReady && slot.acquired_mask == 0) {
                ++report.discarded_ready;
            }
            slot = Slot{};
        }
        module_to_slot.clear();
        pending_acquires.fill(PendingAcquire{});
        // The retired cohort's blocked keys will never publish, so drop them
        // rather than let them attach to a later cohort's get.
        wait_begin.clear();
        return report;
    }

    // Wait for every in-flight fetch to land, without publishing any of
    // them. Returns the number of modules drained, or stops early once the
    // daemon stops. A failed completion throws, which the prefetch thread
    // records as a transport failure.
    std::uint64_t DrainForReset(std::deque<Chunk>* chunks) {
        std::uint64_t drained = 0;
        for (Chunk& chunk : *chunks) {
            if (chunk.request != nullptr) {
                if (UCS_PTR_IS_ERR(chunk.request)) {
                    check_ucs(UCS_PTR_STATUS(chunk.request),
                              "ucp_get_nbx");
                }
                ucs_status_t status = UCS_INPROGRESS;
                while (!stop.load() &&
                       (status = ucp_request_check_status(
                            chunk.request)) == UCS_INPROGRESS) {
                    std::this_thread::sleep_for(
                        std::chrono::microseconds(20));
                }
                if (status == UCS_INPROGRESS) {
                    return drained;
                }
                check_ucs(status, "ucp_get_nbx completion during reset");
                ucp_request_free(chunk.request);
                chunk.request = nullptr;
            }
            drained += static_cast<std::uint64_t>(chunk.count);
        }
        chunks->clear();
        return drained;
    }

    // Which rail carries the next get for `owner`. The rails that reach an
    // owner do not have equal throughput, so handing them equal numbers of
    // gets makes the group run near the rate of its slowest rail; and when an
    // owner is reached over an even number of rails, a rotating cursor against
    // a schedule that alternates that owner's workers pins each worker to one
    // rail for the life of the daemon. Choosing the rail that holds the fewest
    // bytes in flight instead lets a slow rail accumulate outstanding bytes
    // and be passed over until it drains, so the byte split follows capacity
    // without any measured bandwidth table or configuration.
    //
    // The load is derived from `in_flight`, the prefetch thread's own deque of
    // posted-but-unfinished gets, because that deque is the only account of
    // what is outstanding: a separate per-rail counter would have to be kept
    // in step with the post, both completion branches and the drain, and one
    // missed site would leave a rail permanently unselectable. A get that
    // completed immediately (its request is already nullptr) still counts here
    // until the next completion scan removes it, exactly as it still counts
    // against `outstanding`.
    //
    // Equal loads take the lowest rail index: `owner_rails[owner]` is filled
    // in ascending rail order during bootstrap and the scan only moves on a
    // strict improvement, so an empty ring assigns the same rails on every
    // run, as a zeroed cursor did. Only the prefetch thread runs this, and
    // `owner_rails` is immutable once that thread exists, so it takes no lock.
    int LeastLoadedRail(int owner, const std::deque<Chunk>& in_flight) const {
        if (owner < 0 || owner >= num_nodes) {
            fail("distributed weight rail requested for owner " +
                 std::to_string(owner) + " outside the " +
                 std::to_string(num_nodes) + " nodes of node " +
                 std::to_string(node_rank));
        }
        const std::vector<int>& available = owner_rails[owner];
        if (available.empty()) {
            // The bootstrap rail-count check rejects an owner that no rail
            // reaches, so this cannot fire in a valid topology; the rotation
            // this replaced would have divided by zero here instead.
            fail("no distributed weight rail reaches owner " +
                 std::to_string(owner) + " from node " +
                 std::to_string(node_rank));
        }
        for (const int rail : available) {
            if (rail < 0 || rail >= kRails) {
                fail("distributed weight owner " + std::to_string(owner) +
                     " offers rail " + std::to_string(rail) +
                     " outside the rails of node " +
                     std::to_string(node_rank));
            }
        }
        std::array<std::uint64_t, kRails> rail_bytes{};
        for (const Chunk& chunk : in_flight) {
            // Every posted get carries a rail this function chose, so a chunk
            // without one is a bookkeeping error, never a case to absorb.
            if (chunk.rail < 0 || chunk.rail >= kRails) {
                fail("in-flight distributed weight fetch has no rail");
            }
            rail_bytes[chunk.rail] +=
                static_cast<std::uint64_t>(chunk.count) * module_bytes;
        }
        // Compare (load, rail index) so the tie-break is the lowest rail
        // index by construction rather than by the order the bootstrap
        // happened to fill this owner's candidates in.
        int chosen = available.front();
        std::uint64_t chosen_bytes = rail_bytes[chosen];
        for (const int rail : available) {
            if (rail_bytes[rail] < chosen_bytes ||
                (rail_bytes[rail] == chosen_bytes && rail < chosen)) {
                chosen = rail;
                chosen_bytes = rail_bytes[rail];
            }
        }
        return chosen;
    }

    void PrefetchLoop() {
        const std::vector<std::string> schedule = BuildSchedule();
        size_t next = 0;
        int tail = 0;
        std::deque<Chunk> chunks;
        Clock::time_point prev_scan{};

        while (!stop.load()) {
            // A cohort swap rewinds the ring to its start-of-day state: the
            // next cohort begins at the first module of the schedule, exactly
            // like the first cohort did.
            std::uint64_t reset_request = 0;
            {
                std::lock_guard<std::mutex> lock(ring_mutex);
                if (ring_reset_requested != ring_reset_completed) {
                    reset_request = ring_reset_requested;
                }
            }
            if (reset_request != 0) {
                const std::uint64_t drained = DrainForReset(&chunks);
                if (stop.load()) {
                    break;
                }
                {
                    std::lock_guard<std::mutex> lock(ring_mutex);
                    RingResetReport report = ResetSlotsLocked();
                    report.drained_fetches = drained;
                    report.schedule_cursor = next;
                    report.ring_tail = tail;
                    report.prefetch_handshake = true;
                    last_ring_reset = report;
                    ring_reset_completed = reset_request;
                }
                ring_cv.notify_all();
                next = 0;
                tail = 0;
                // Rail selection needs no rewind of its own: the drain above
                // emptied the in-flight deque, so every rail reads as unloaded
                // and the next cohort's first get per owner takes that owner's
                // lowest rail, exactly as the first cohort's did.
                continue;
            }

            // This thread both polls completions and posts new gets, so a
            // completion can only be noticed when the scan comes round again.
            // Carrying the previous scan's start lets the trace bound how much
            // of submit..done is detection delay rather than transport.
            const Clock::time_point scan_start = Clock::now();
            for (auto iterator = chunks.begin();
                 iterator != chunks.end();) {
                if (iterator->request == nullptr) {
                    MarkReady(*iterator, Clock::now(), prev_scan);
                    iterator = chunks.erase(iterator);
                    continue;
                }
                if (UCS_PTR_IS_ERR(iterator->request)) {
                    check_ucs(UCS_PTR_STATUS(iterator->request),
                              "ucp_get_nbx");
                }
                const ucs_status_t status =
                    ucp_request_check_status(iterator->request);
                if (status == UCS_INPROGRESS) {
                    ++iterator;
                    continue;
                }
                check_ucs(status, "ucp_get_nbx completion");
                ucp_request_free(iterator->request);
                iterator->request = nullptr;
                MarkReady(*iterator, Clock::now(), prev_scan);
                iterator = chunks.erase(iterator);
            }
            prev_scan = scan_start;

            bool posted = false;
            while (static_cast<int>(chunks.size()) < outstanding) {
                if (next == schedule.size()) {
                    next = 0;
                }
                if (tail + superchunk > depth) {
                    tail = 0;
                }
                {
                    std::lock_guard<std::mutex> lock(ring_mutex);
                    if (!SlotsFree(tail, superchunk)) {
                        // Posting stops here. Record whether it stopped with a
                        // free group sitting elsewhere in the ring: that is the
                        // difference between the positional gate costing
                        // concurrency and the ring simply being full. Only on
                        // the break path, so it costs one O(depth) scan per
                        // stall, never per post.
                        if (!fetch_trace.empty()) {
                            bool free_elsewhere = false;
                            for (int group = 0; group + superchunk <= depth;
                                 group += superchunk) {
                                if (group != tail &&
                                    SlotsFree(group, superchunk)) {
                                    free_elsewhere = true;
                                    break;
                                }
                            }
                            gate_stalls.fetch_add(1,
                                                  std::memory_order_relaxed);
                            if (free_elsewhere) {
                                gate_stalls_free_elsewhere.fetch_add(
                                    1, std::memory_order_relaxed);
                            }
                        }
                        break;
                    }
                }
                const auto [layer, first_expert] =
                    parse_module_key(schedule[next]);
                const int owner =
                    first_expert / experts_per_owner;
                if (next + superchunk > schedule.size()) {
                    fail("truncated final superchunk");
                }
                for (int index = 0; index < superchunk; ++index) {
                    const auto [next_layer, next_expert] =
                        parse_module_key(schedule[next + index]);
                    if (next_layer != layer ||
                        next_expert != first_expert + index ||
                        next_expert / experts_per_owner != owner) {
                        fail("remote schedule is not superchunk-contiguous");
                    }
                }
                Chunk chunk;
                chunk.slot_start = tail;
                chunk.count = superchunk;
                chunk.owner = owner;
                chunk.rail = LeastLoadedRail(owner, chunks);
                chunk.layer = layer;
                chunk.first_expert = first_expert;
                chunk.target_worker = OwningWorker(schedule[next]);
                chunk.posted = Clock::now();

                {
                    std::lock_guard<std::mutex> lock(ring_mutex);
                    for (int index = 0; index < superchunk; ++index) {
                        Slot& slot = slots[tail + index];
                        slot.state = SlotState::kFetching;
                        slot.module_key = schedule[next + index];
                        slot.generation = next_generation++;
                    }
                    // One scan serves three purposes: the staging high-water
                    // mark, and the trace's split of who holds the ring. The
                    // split is what decides whether in-flight is low because
                    // posting is gated or because consumers hold the slots:
                    // with depth == outstanding * superchunk the two are
                    // strictly complementary, so the counts are the evidence.
                    int occupied = 0;
                    int ready_unacquired = 0;
                    int leased = 0;
                    for (const Slot& slot : slots) {
                        occupied += slot.state != SlotState::kFree;
                        if (slot.state == SlotState::kReady &&
                            slot.acquired_mask == 0) {
                            ++ready_unacquired;
                        }
                        if ((slot.acquired_mask & ~slot.released_mask) != 0) {
                            ++leased;
                        }
                    }
                    chunk.occupied_slots = occupied;
                    chunk.ready_unacquired_slots = ready_unacquired;
                    chunk.leased_slots = leased;
                    {
                        std::lock_guard<std::mutex> stats_lock(
                            stats_mutex);
                        stats.staging_hwm =
                            std::max(stats.staging_hwm, occupied);
                    }
                }

                RemoteEndpoint& remote =
                    remotes[chunk.rail][owner];
                const std::uint64_t remote_address =
                    remote.source_address +
                    RemoteOffset(schedule[next]);
                void* destination =
                    static_cast<char*>(staging) +
                    static_cast<std::uint64_t>(tail) * module_bytes;
                ucp_request_param_t params{};
                params.op_attr_mask = UCP_OP_ATTR_FIELD_MEMH;
                params.memh = rails[chunk.rail].staging_memh;
                chunk.submitted = Clock::now();
                chunk.request = ucp_get_nbx(
                    remote.endpoint, destination,
                    static_cast<size_t>(superchunk) * module_bytes,
                    remote_address, remote.rkey, &params);
                if (UCS_PTR_IS_ERR(chunk.request)) {
                    check_ucs(UCS_PTR_STATUS(chunk.request),
                              "ucp_get_nbx post");
                }
                chunks.push_back(chunk);
                next += superchunk;
                tail = (tail + superchunk) % depth;
                posted = true;
            }
            if (!posted && chunks.empty()) {
                std::unique_lock<std::mutex> lock(ring_mutex);
                ring_cv.wait_for(lock, std::chrono::milliseconds(1));
            } else {
                std::this_thread::sleep_for(
                    std::chrono::microseconds(20));
            }
        }
        WriteSummary();
        WriteFetchTrace();
    }

    // The summary's snapshots. Each takes one mutex and only copies, so a
    // request path never waits behind JSON formatting or file I/O.
    CohortSnapshot SnapshotCohort() {
        std::lock_guard<std::mutex> lock(cohort_mutex);
        CohortSnapshot snapshot;
        snapshot.epoch = cohort_epoch;
        snapshot.state = cohort_state;
        snapshot.admitted_mask = cohort_admitted_mask;
        snapshot.faulted = cohort_faulted;
        snapshot.fault = cohort_fault;
        snapshot.expected_disconnects = cohort_expected_disconnects;
        snapshot.rejected_hellos = rejected_hellos;
        snapshot.last_rejected_hello = last_rejected_hello;
        snapshot.completed_swaps = completed_swaps;
        snapshot.swap_log = swap_log;
        return snapshot;
    }

    RingSnapshot SnapshotRing() {
        std::lock_guard<std::mutex> lock(ring_mutex);
        return RingSnapshot{slots, pending_acquires, Clock::now()};
    }

    Stats SnapshotStats() {
        std::lock_guard<std::mutex> lock(stats_mutex);
        return stats;
    }

    void WriteSummary() {
        if (summary_path.empty()) {
            return;
        }
        std::lock_guard<std::mutex> summary_lock(summary_mutex);
        const CohortSnapshot cohort_snapshot = SnapshotCohort();
        const RingSnapshot ring = SnapshotRing();
        const Stats counters = SnapshotStats();
        const bool transport_failed = failed.load();
        const std::string failure = FailureMessage();

        nlohmann::json cohort;
        cohort["epoch"] = cohort_snapshot.epoch;
        cohort["state"] = cohort_state_name(cohort_snapshot.state);
        cohort["admitted_workers_mask"] = cohort_snapshot.admitted_mask;
        cohort["faulted"] = cohort_snapshot.faulted;
        cohort["fault"] = cohort_snapshot.fault;
        cohort["fault_class"] =
            cohort_snapshot.faulted ? "cohort_recoverable" : "none";
        cohort["expected_disconnects"] = cohort_snapshot.expected_disconnects;
        cohort["rejected_hellos"] = cohort_snapshot.rejected_hellos;
        cohort["last_rejected_hello"] = cohort_snapshot.last_rejected_hello;
        cohort["completed_swaps"] = cohort_snapshot.completed_swaps;
        // Keyed by worker id: ready slots no worker has acquired yet, by the
        // worker expected to acquire them, and each worker's pending acquire.
        std::map<std::string, std::uint64_t> ready_unacquired;
        nlohmann::json pending_by_worker = nlohmann::json::object();
        for (const Slot& slot : ring.slots) {
            if (slot.state == SlotState::kReady && slot.acquired_mask == 0) {
                ++ready_unacquired[std::to_string(
                    OwningWorker(slot.module_key))];
            }
        }
        for (int worker = 0; worker < kDefaultWorkers; ++worker) {
            const PendingAcquire& pending = ring.pending_acquires[worker];
            if (!pending.module_key.empty()) {
                pending_by_worker[std::to_string(worker)] = {
                    {"module_key", pending.module_key},
                    {"age_s", elapsed_seconds(pending.begin, ring.taken)},
                };
            }
        }

        nlohmann::json summary;
        summary["schema"] = "k3-distributed-weights/1";
        summary["node_rank"] = node_rank;
        summary["network_ready"] = network_ready.load();
        summary["failed"] = transport_failed;
        summary["failure"] = failure;
        summary["failure_class"] =
            transport_failed ? "transport_fatal" : "none";
        summary["cohort"] = cohort;
        summary["cohort_swaps"] = cohort_snapshot.swap_log;
        summary["depth"] = depth;
        summary["superchunk"] = superchunk;
        summary["outstanding"] = outstanding;
        summary["worker_sharded"] = worker_sharded;
        summary["transport"] = transport;
        summary["store_bytes"] = store_bytes;
        summary["staging_bytes"] = staging_bytes;
        summary["fetched_modules"] = counters.fetched_modules;
        summary["fetched_bytes"] = counters.fetched_bytes;
        summary["acquire_count"] = counters.acquire_count;
        summary["release_count"] = counters.release_count;
        summary["cache_hits"] = counters.cache_hits;
        summary["duplicate_fetches"] = counters.duplicate_fetches;
        summary["wait_count"] = counters.wait_count;
        summary["wait_seconds"] = counters.wait_seconds;
        summary["worker_wait_count"] = counters.worker_wait_count;
        summary["worker_wait_seconds"] = counters.worker_wait_seconds;
        summary["ready_unacquired_by_worker"] = ready_unacquired;
        summary["pending_acquire_by_worker"] = pending_by_worker;
        summary["staging_hwm_modules"] = counters.staging_hwm;
        // One entry per rail. The counters are monotonic over the daemon's
        // life, so two summaries subtract to the gaps of the interval.
        nlohmann::json progress_by_rail = nlohmann::json::array();
        for (int rail = 0; rail < kRails; ++rail) {
            const ProgressGaps& gaps = progress_gaps[rail];
            progress_by_rail.push_back({
                {"loops", gaps.loops.load(std::memory_order_relaxed)},
                {"gap_ge_1ms", gaps.gap_ge_1ms.load(std::memory_order_relaxed)},
                {"gap_ge_10ms",
                 gaps.gap_ge_10ms.load(std::memory_order_relaxed)},
                {"gap_ge_50ms",
                 gaps.gap_ge_50ms.load(std::memory_order_relaxed)},
                {"max_gap_us", gaps.max_gap_us.load(std::memory_order_relaxed)},
            });
        }
        summary["progress_gaps_by_rail"] = progress_by_rail;
        summary["progress_thread_cores"] = progress_thread_cores;
        const LatencyHistogram& latency = counters.ready_latency;
        summary["ready_latency_ms"] = {
            {"p50", latency.percentile(50)},
            {"p95", latency.percentile(95)},
            {"p99", latency.percentile(99)},
            {"count", latency.count()},
            {"max", latency.max()},
        };
        const std::string temporary = summary_path + ".tmp";
        {
            std::ofstream output(temporary);
            output << std::setw(2) << summary << '\n';
        }
        rename(temporary.c_str(), summary_path.c_str());
    }

    std::uint64_t CohortEpoch() {
        std::lock_guard<std::mutex> lock(cohort_mutex);
        return cohort_epoch;
    }

    std::uint64_t BeginCohortSwap() {
        if (!started.load()) {
            fail("distributed weights daemon is not started");
        }
        std::uint64_t epoch = 0;
        {
            std::lock_guard<std::mutex> lock(cohort_mutex);
            epoch = cohort_epoch;
            if (cohort_state == CohortState::kDraining) {
                return epoch;
            }
            cohort_state = CohortState::kDraining;
            cohort_closed.store(true);
        }
        // Pending acquires and hello waits of this cohort return now.
        WakeWaiters();
        WriteSummary();
        return epoch;
    }

    // Rewind the ring to its start-of-day state. With host RDMA the prefetch
    // thread does it, after draining its in-flight fetches; before that
    // thread exists (and always for hierarchical_gdr, which has none) no
    // fetch can be in flight, so it is done here.
    RingResetReport ResetRing(Clock::time_point deadline) {
        std::unique_lock<std::mutex> lock(ring_mutex);
        if (!prefetch_owns_ring) {
            return ResetSlotsLocked();
        }
        const std::uint64_t request = ++ring_reset_requested;
        ring_cv.notify_all();
        ring_cv.wait_until(lock, deadline, [&]() {
            return ring_reset_completed >= request || failed.load() ||
                   stop.load();
        });
        if (ring_reset_completed >= request) {
            return last_ring_reset;
        }
        lock.unlock();
        if (failed.load()) {
            fail(TransportFatalMessage());
        }
        if (stop.load()) {
            fail("distributed weights daemon is stopping");
        }
        const std::string message =
            "worker cohort swap: in-flight remote fetches did not drain "
            "before the swap deadline";
        RecordFailure(message);
        fail(TransportFatalMessage());
    }

    std::string FinishCohortSwap(double timeout_seconds) {
        if (!started.load()) {
            fail("distributed weights daemon is not started");
        }
        const Clock::time_point begin = Clock::now();
        std::uint64_t retiring = 0;
        {
            std::lock_guard<std::mutex> lock(cohort_mutex);
            if (cohort_state != CohortState::kDraining) {
                fail("FinishCohortSwap called without BeginCohortSwap");
            }
            retiring = cohort_epoch;
        }
        if (failed.load()) {
            fail(TransportFatalMessage());
        }
        if (stop.load()) {
            fail("distributed weights daemon is stopping");
        }

        // 1. Retire every connection accepted so far: the old cohort's and
        // any hello refused while draining. The cohort is reaped, so its
        // sockets are dead; shutdown() also frees a thread in recv/send, and
        // cohort_closed (set by Begin) frees one in an acquire or hello wait.
        std::vector<std::unique_ptr<ClientConnection>> retired;
        {
            std::lock_guard<std::mutex> lock(client_mutex);
            retired.swap(clients);
        }
        for (const auto& connection : retired) {
            if (connection->fd >= 0) {
                shutdown(connection->fd, SHUT_RDWR);
            }
        }
        WakeWaiters();
        std::uint64_t cohort_connections = 0;
        std::uint64_t other_connections = 0;
        std::uint64_t closed_fds = 0;
        for (const auto& connection : retired) {
            if (connection->thread.joinable()) {
                connection->thread.join();
            }
            if (connection->fd >= 0) {
                close(connection->fd);
                ++closed_fds;
            }
            if (connection->epoch == retiring) {
                ++cohort_connections;
            } else {
                ++other_connections;
            }
        }

        // 2. No thread of the retired cohort exists any more, and none of the
        // next one is admitted yet: slots can be released safely.
        // The timeout bounds the drain of in-flight fetches only; the joins
        // above are bounded by construction.
        const RingResetReport ring = ResetRing(
            Clock::now() + std::chrono::duration_cast<Clock::duration>(
                               std::chrono::duration<double>(
                                   timeout_seconds)));

        // 3. Admit the next cohort.
        nlohmann::json report;
        {
            std::lock_guard<std::mutex> lock(cohort_mutex);
            report["from_cohort"] = retiring;
            report["to_cohort"] = retiring + 1;
            report["closed_fds"] = closed_fds;
            report["joined_cohort_connections"] = cohort_connections;
            report["joined_refused_connections"] = other_connections;
            report["expected_disconnects"] = cohort_expected_disconnects;
            report["released_leases"] = ring.outstanding_leases;
            report["released_leased_slots"] = ring.leased_slots;
            report["discarded_ready_modules"] = ring.discarded_ready;
            report["drained_inflight_modules"] = ring.drained_fetches;
            report["prefetch_cursor_rewound_from"] = ring.schedule_cursor;
            report["ring_tail_rewound_from"] = ring.ring_tail;
            report["reset_by_prefetch_thread"] = ring.prefetch_handshake;
            report["cleared_cohort_fault"] = cohort_fault;
            report["elapsed_ms"] = elapsed_seconds(begin) * 1000.0;
            cohort_faulted = false;
            cohort_fault.clear();
            cohort_admitted_mask = 0;
            cohort_release_mask = 0;
            cohort_expected_disconnects = 0;
            cohort_epoch = retiring + 1;
            cohort_state = CohortState::kAdmitting;
            ++completed_swaps;
            swap_log.push_back(report);
            if (swap_log.size() > kSwapLogEntries) {
                swap_log.erase(swap_log.begin());
            }
            cohort_closed.store(false);
        }
        WriteSummary();
        return report.dump();
    }

    void PrepareStop() {
        if (!started.load()) {
            return;
        }
        stop.store(true);
        WakeWaiters();
    }

    void Stop() {
        if (!started.load()) {
            return;
        }
        PrepareStop();
        if (unix_listener >= 0) {
            shutdown(unix_listener, SHUT_RDWR);
            close(unix_listener);
            unix_listener = -1;
        }
        for (int& listener : tcp_listeners) {
            if (listener >= 0) {
                shutdown(listener, SHUT_RDWR);
                close(listener);
                listener = -1;
            }
        }
        {
            std::lock_guard<std::mutex> lock(client_mutex);
            for (const auto& connection : clients) {
                if (connection->fd >= 0) {
                    shutdown(connection->fd, SHUT_RDWR);
                }
            }
        }
        if (prefetch_thread.joinable()) {
            prefetch_thread.join();
        }
        if (bootstrap_thread.joinable()) {
            bootstrap_thread.join();
        }
        if (unix_accept_thread.joinable()) {
            unix_accept_thread.join();
        }
        for (std::thread& thread : progress_threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        {
            // Joined outside client_mutex: a concurrent FinishCohortSwap only
            // takes the lock to swap the list out, and joins its own share.
            std::vector<std::unique_ptr<ClientConnection>> remaining;
            {
                std::lock_guard<std::mutex> lock(client_mutex);
                remaining.swap(clients);
            }
            for (const auto& connection : remaining) {
                if (connection->thread.joinable()) {
                    connection->thread.join();
                }
                if (connection->fd >= 0) {
                    close(connection->fd);
                }
            }
        }
        if (summary_thread.joinable()) {
            summary_thread.join();
        }

        for (int rail = 0; rail < kRails; ++rail) {
            for (int owner = 0; owner < num_nodes; ++owner) {
                RemoteEndpoint& remote = remotes[rail][owner];
                if (remote.rkey != nullptr) {
                    ucp_rkey_destroy(remote.rkey);
                    remote.rkey = nullptr;
                }
                if (remote.endpoint != nullptr) {
                    close_endpoint(rails[rail].worker,
                                   remote.endpoint);
                    remote.endpoint = nullptr;
                }
                if (remote.control_fd >= 0) {
                    close(remote.control_fd);
                    remote.control_fd = -1;
                }
            }
            for (ucp_ep_h endpoint : incoming_endpoints[rail]) {
                close_endpoint(rails[rail].worker, endpoint);
            }
            for (int fd : incoming_controls[rail]) {
                close(fd);
            }
            DestroyRail(&rails[rail]);
        }
        WriteSummary();
        unlink(socket_path.c_str());
        if (staging != nullptr) {
            munmap(staging, staging_bytes);
            staging = nullptr;
        }
        if (store != nullptr) {
            munmap(store, store_bytes);
            store = nullptr;
        }
        if (staging_fd >= 0) {
            close(staging_fd);
            staging_fd = -1;
        }
        if (store_fd >= 0) {
            close(store_fd);
            store_fd = -1;
        }
        started.store(false);
    }
};

DistributedWeightDaemon::DistributedWeightDaemon(
    const std::string& config_path)
    : impl_(std::make_unique<Impl>(config_path)) {}

DistributedWeightDaemon::~DistributedWeightDaemon() = default;

void DistributedWeightDaemon::Start() { impl_->Start(); }

void DistributedWeightDaemon::WaitReady(double timeout_seconds) {
    impl_->WaitReady(timeout_seconds);
}

std::uint64_t DistributedWeightDaemon::CohortEpoch() {
    return impl_->CohortEpoch();
}

std::uint64_t DistributedWeightDaemon::BeginCohortSwap() {
    return impl_->BeginCohortSwap();
}

std::string DistributedWeightDaemon::FinishCohortSwap(
    double timeout_seconds) {
    return impl_->FinishCohortSwap(timeout_seconds);
}

void DistributedWeightDaemon::PrepareStop() { impl_->PrepareStop(); }

void DistributedWeightDaemon::Stop() { impl_->Stop(); }
