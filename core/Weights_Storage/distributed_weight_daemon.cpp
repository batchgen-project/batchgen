#include "distributed_weight_daemon.h"

#include "distributed_weights_protocol.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <linux/memfd.h>
#include <netinet/in.h>
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
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
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

double percentile(std::vector<double> values, double percent) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double index =
        (values.size() - 1) * percent / 100.0;
    const size_t lower = static_cast<size_t>(std::floor(index));
    const size_t upper = std::min(lower + 1, values.size() - 1);
    const double weight = index - lower;
    return values[lower] * (1.0 - weight) +
           values[upper] * weight;
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

struct Chunk {
    int slot_start = 0;
    int count = 0;
    int owner = -1;
    int rail = -1;
    void* request = nullptr;
    Clock::time_point posted;
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
// epoch/worker_id are written by the connection's own thread when its hello
// is admitted, and read by others only after joining that thread.
struct ClientConnection {
    int fd = -1;
    std::thread thread;
    std::uint64_t epoch = 0;  // 0 until the hello is admitted
    int worker_id = -1;
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
        staging_bytes =
            static_cast<std::uint64_t>(depth) * module_bytes;
        all_workers_mask =
            (static_cast<std::uint32_t>(1) << workers) - 1;
        slots.resize(depth);
        owner_rails.resize(num_nodes);
        owner_rail_cursor.assign(num_nodes, 0);
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
    std::vector<int> owner_rail_cursor;

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
    std::vector<std::unique_ptr<ClientConnection>> clients;
    std::mutex client_mutex;

    // Lock order, outermost first. No path holds two of these at once except
    // where listed, and none is held across a thread join or WriteSummary:
    //   summary_mutex -> cohort_mutex (snapshot), then
    //   summary_mutex -> stats_mutex -> failure_mutex
    //   ring_mutex -> stats_mutex (MarkReady, PrefetchLoop)
    //   ready_mutex -> failure_mutex (WaitReady)
    //   client_mutex, cohort_mutex: leaves
    // cohort_closed mirrors (cohort_state == kDraining || cohort_faulted) so
    // the ring and ready predicates can read it without taking cohort_mutex;
    // every writer stores it and then calls WakeWaiters().
    std::mutex cohort_mutex;
    std::uint64_t cohort_epoch = 1;
    CohortState cohort_state = CohortState::kAdmitting;
    std::uint32_t cohort_admitted_mask = 0;
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
    int staging_hwm = 0;
    // Completed fetches, so an acquire timeout can tell a stalled transport
    // (no completion while slots are fetching) from a cohort that never
    // consumed what the ring holds. Under ring_mutex.
    std::uint64_t ring_completions = 0;
    // Ring reset handshake. Once prefetch_owns_ring is set, only the prefetch
    // thread resets slots, at a point where it has no fetch in flight. All
    // under ring_mutex.
    bool prefetch_owns_ring = false;
    std::uint64_t ring_reset_requested = 0;
    std::uint64_t ring_reset_completed = 0;
    RingResetReport last_ring_reset;

    std::mutex stats_mutex;
    std::mutex summary_mutex;
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
    std::vector<double> ready_latency_ms;

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
        if (listen(unix_listener, workers) != 0) {
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
                while (!stop.load()) {
                    if (ucp_worker_progress(rails[rail].worker) == 0) {
                        std::this_thread::yield();
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
        if ((cohort_admitted_mask & bit) != 0) {
            return worker + " is already connected in worker cohort " +
                   std::to_string(cohort_epoch);
        }
        cohort_admitted_mask |= bit;
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
            WaitReadyForWorker(600.0);
            Response hello;
            hello.staging_bytes = staging_bytes;
            hello.module_bytes = module_bytes;
            hello.generation = epoch;
            send_response_with_fd(fd, hello, staging_fd);

            while (!stop.load()) {
                Request operation{};
                recv_exact(fd, &operation, sizeof(operation));
                Response response;
                try {
                    if (operation.magic !=
                            batchgen::distributed_weights::kProtocolMagic ||
                        operation.version !=
                            batchgen::distributed_weights::kProtocolVersion ||
                        operation.worker_id != request.worker_id) {
                        fail("invalid distributed weight request");
                    }
                    const std::string module_key(
                        operation.module_key,
                        strnlen(operation.module_key,
                                sizeof(operation.module_key)));
                    if (operation.operation ==
                        static_cast<std::uint32_t>(
                            Operation::kAcquire)) {
                        Acquire(module_key, worker_id, &response);
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
                if (response.status != 0) {
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

    void Acquire(const std::string& module_key, int worker_id,
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
        const Clock::time_point begin = Clock::now();
        std::unique_lock<std::mutex> lock(ring_mutex);
        const std::uint64_t completions_at_start = ring_completions;
        const bool available = ring_cv.wait_for(
            lock, std::chrono::seconds(300), [&]() {
            if (failed.load() || stop.load() || cohort_closed.load()) {
                return true;
            }
            const auto found = module_to_slot.find(module_key);
            return found != module_to_slot.end() &&
                   slots[found->second].state ==
                       SlotState::kReady;
        });
        if (!available) {
            int fetching = 0;
            int ready = 0;
            for (const Slot& slot : slots) {
                fetching += slot.state == SlotState::kFetching;
                ready += slot.state == SlotState::kReady;
            }
            const bool progressed = ring_completions != completions_at_start;
            lock.unlock();
            if (fetching > 0 && !progressed) {
                // Fetches were posted and none completed in 300 s: the
                // transport, not the cohort, stopped moving.
                const std::string message =
                    "remote fetch stalled: no fetch completed in 300 s "
                    "with " + std::to_string(fetching) +
                    " slots fetching (waiting for " + module_key + ")";
                RecordFailure(message);
                fail(message);
            }
            fail("timed out waiting for remote module " + module_key +
                 ": the ring holds " + std::to_string(ready) +
                 " ready modules this cohort did not consume");
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
        ++acquire_count;
        if (cache_hit) {
            ++cache_hits;
        }
        if (waited) {
            ++wait_count;
            wait_seconds += wait_s;
            worker_wait_count.at(worker_id) += 1;
            worker_wait_seconds.at(worker_id) += wait_s;
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
        ++release_count;
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

    std::vector<std::string> BuildSchedule() const {
        std::vector<std::string> schedule;
        schedule.reserve(
            kLayers * (kExperts - experts_per_owner));
        if (worker_sharded) {
            const int experts_per_worker = kExperts / workers;
            // Round-robin complete superchunks across worker shards. Every
            // worker's first requested module is then within the initial ring
            // fill; a layer-major expert sweep would strand high TP ranks
            // behind get_weights()' 2-second timeout.
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

    void MarkReady(const Chunk& chunk,
                   Clock::time_point completed) {
        const double latency_ms =
            elapsed_seconds(chunk.posted, completed) * 1000.0;
        {
            std::lock_guard<std::mutex> lock(ring_mutex);
            for (int index = 0; index < chunk.count; ++index) {
                Slot& slot = slots[chunk.slot_start + index];
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
                        ++duplicate_fetches;
                    }
                    fail("duplicate remote module fetch: " +
                         slot.module_key);
                }
            }
            ring_completions += chunk.count;
            ring_cv.notify_all();
        }
        std::lock_guard<std::mutex> stats_lock(stats_mutex);
        fetched_modules += chunk.count;
        fetched_bytes +=
            static_cast<std::uint64_t>(chunk.count) * module_bytes;
        for (int index = 0; index < chunk.count; ++index) {
            ready_latency_ms.push_back(latency_ms);
        }
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

    void PrefetchLoop() {
        const std::vector<std::string> schedule = BuildSchedule();
        size_t next = 0;
        int tail = 0;
        std::deque<Chunk> chunks;

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
                std::fill(owner_rail_cursor.begin(),
                          owner_rail_cursor.end(), 0);
                continue;
            }

            for (auto iterator = chunks.begin();
                 iterator != chunks.end();) {
                if (iterator->request == nullptr) {
                    MarkReady(*iterator, Clock::now());
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
                MarkReady(*iterator, Clock::now());
                iterator = chunks.erase(iterator);
            }

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
                const std::vector<int>& available =
                    owner_rails[owner];
                chunk.rail =
                    available[owner_rail_cursor[owner]++ %
                              available.size()];
                chunk.posted = Clock::now();

                {
                    std::lock_guard<std::mutex> lock(ring_mutex);
                    for (int index = 0; index < superchunk; ++index) {
                        Slot& slot = slots[tail + index];
                        slot.state = SlotState::kFetching;
                        slot.module_key = schedule[next + index];
                        slot.generation = next_generation++;
                    }
                    int occupied = 0;
                    for (const Slot& slot : slots) {
                        occupied += slot.state != SlotState::kFree;
                    }
                    {
                        std::lock_guard<std::mutex> stats_lock(
                            stats_mutex);
                        staging_hwm = std::max(staging_hwm, occupied);
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
    }

    void WriteSummary() {
        if (summary_path.empty()) {
            return;
        }
        std::lock_guard<std::mutex> summary_lock(summary_mutex);
        nlohmann::json summary;
        nlohmann::json cohort;
        nlohmann::json swaps;
        {
            std::lock_guard<std::mutex> lock(cohort_mutex);
            cohort["epoch"] = cohort_epoch;
            cohort["state"] = cohort_state_name(cohort_state);
            cohort["admitted_workers_mask"] = cohort_admitted_mask;
            cohort["faulted"] = cohort_faulted;
            cohort["fault"] = cohort_fault;
            cohort["fault_class"] =
                cohort_faulted ? "cohort_recoverable" : "none";
            cohort["expected_disconnects"] = cohort_expected_disconnects;
            cohort["rejected_hellos"] = rejected_hellos;
            cohort["last_rejected_hello"] = last_rejected_hello;
            cohort["completed_swaps"] = completed_swaps;
            swaps = swap_log;
        }
        {
            std::lock_guard<std::mutex> lock(stats_mutex);
            summary["schema"] = "k3-distributed-weights/1";
            summary["node_rank"] = node_rank;
            summary["network_ready"] = network_ready.load();
            summary["failed"] = failed.load();
            summary["failure"] = FailureMessage();
            summary["failure_class"] =
                failed.load() ? "transport_fatal" : "none";
            summary["cohort"] = cohort;
            summary["cohort_swaps"] = swaps;
            summary["depth"] = depth;
            summary["superchunk"] = superchunk;
            summary["outstanding"] = outstanding;
            summary["worker_sharded"] = worker_sharded;
            summary["transport"] = transport;
            summary["store_bytes"] = store_bytes;
            summary["staging_bytes"] = staging_bytes;
            summary["fetched_modules"] = fetched_modules;
            summary["fetched_bytes"] = fetched_bytes;
            summary["acquire_count"] = acquire_count;
            summary["release_count"] = release_count;
            summary["cache_hits"] = cache_hits;
            summary["duplicate_fetches"] = duplicate_fetches;
            summary["wait_count"] = wait_count;
            summary["wait_seconds"] = wait_seconds;
            summary["worker_wait_count"] = worker_wait_count;
            summary["worker_wait_seconds"] = worker_wait_seconds;
            summary["staging_hwm_modules"] = staging_hwm;
            summary["ready_latency_ms"] = {
                {"p50", percentile(ready_latency_ms, 50)},
                {"p95", percentile(ready_latency_ms, 95)},
                {"p99", percentile(ready_latency_ms, 99)},
            };
        }
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
