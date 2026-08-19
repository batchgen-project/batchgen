#include "distributed_weight_daemon.h"

#include "distributed_weights_protocol.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <linux/memfd.h>
#include <netinet/in.h>
#include <numa.h>
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
constexpr int kNodes = 4;
constexpr int kRails = 8;
constexpr int kLayers = 92;
constexpr int kExperts = 896;
constexpr int kExpertsPerOwner = 224;
constexpr int kDefaultWorkers = 8;

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

int connect_retry(const std::string& ip, int port,
                  const std::atomic<bool>& stop) {
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<std::uint16_t>(port));
    if (inet_pton(AF_INET, ip.c_str(), &address.sin_addr) != 1) {
        fail("invalid peer IP " + ip);
    }
    for (int attempt = 0; attempt < 1800 && !stop.load(); ++attempt) {
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

int destination_for_owner_rail(int owner, int rail) {
    return (owner + 1 + (rail - 1) % 3) % kNodes;
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
                       .get<std::array<std::string, kNodes>>();
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
        depth = config.value("depth", 32);
        superchunk = config.value("superchunk", 8);
        outstanding = config.value("outstanding", 8);
        workers = config.value("workers", kDefaultWorkers);

        if (node_rank < 0 || node_rank >= kNodes ||
            node_ips.size() != kNodes) {
            fail("invalid distributed weight node topology");
        }
        if (depth <= 0 || depth % superchunk != 0 ||
            superchunk <= 0 || outstanding <= 0 ||
            workers <= 0 || workers > 31) {
            fail("invalid distributed weight ring configuration");
        }
        staging_bytes =
            static_cast<std::uint64_t>(depth) * module_bytes;
        all_workers_mask =
            (static_cast<std::uint32_t>(1) << workers) - 1;
        slots.resize(depth);
    }

    ~Impl() { Stop(); }

    std::string config_path;
    int node_rank = -1;
    std::array<std::string, kNodes> node_ips;
    std::string store_path;
    std::string socket_path;
    std::string summary_path;
    std::uint64_t store_bytes = 0;
    std::uint64_t module_bytes = 0;
    std::uint64_t replicated_bytes = 0;
    std::uint64_t staging_bytes = 0;
    int base_port = 26000;
    int depth = 32;
    int superchunk = 8;
    int outstanding = 8;
    int workers = kDefaultWorkers;
    std::uint32_t all_workers_mask = 0;

    int store_fd = -1;
    int staging_fd = -1;
    int unix_listener = -1;
    void* store = nullptr;
    void* staging = nullptr;
    std::array<int, kRails> tcp_listeners{};
    std::array<UcpRail, kRails> rails;
    std::array<std::array<RemoteEndpoint, kNodes>, kRails> remotes;
    std::array<std::vector<ucp_ep_h>, kRails> incoming_endpoints;
    std::array<std::vector<int>, kRails> incoming_controls;
    std::array<std::vector<int>, kNodes> owner_rails;
    std::array<int, kNodes> owner_rail_cursor{};

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
    std::vector<std::thread> client_threads;
    std::vector<int> client_fds;
    std::mutex client_mutex;

    std::mutex ring_mutex;
    std::condition_variable ring_cv;
    std::vector<Slot> slots;
    std::unordered_map<std::string, int> module_to_slot;
    std::uint64_t next_generation = 1;
    int staging_hwm = 0;

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

    void RecordFailure(const std::string& message) {
        {
            std::lock_guard<std::mutex> lock(failure_mutex);
            if (!failed.exchange(true)) {
                failure_message = message;
            }
        }
        stop.store(true);
        ready_cv.notify_all();
        ring_cv.notify_all();
        WriteSummary();
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
        const std::string device =
            "mlx5_bond_" + std::to_string(rail_index + 1) + ":1";
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
            bootstrap_thread = std::thread([this]() {
                try {
                    BootstrapNetwork();
                } catch (const std::exception& error) {
                    if (!stop.load()) {
                        RecordFailure(error.what());
                    }
                }
            });
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
            for (int owner = 0; owner < kNodes; ++owner) {
                if (owner == node_rank ||
                    destination_for_owner_rail(owner, rail + 1) !=
                        node_rank) {
                    continue;
                }
                const int fd = connect_retry(
                    node_ips[owner],
                    base_port + owner * kRails + rail, stop);
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

        for (std::thread& thread : accept_threads) {
            thread.join();
        }
        if (failed.load()) {
            fail(FailureMessage());
        }
        if (stop.load()) {
            return;
        }
        for (int owner = 0; owner < kNodes; ++owner) {
            if (owner == node_rank) {
                continue;
            }
            if (owner_rails[owner].size() < 2 ||
                owner_rails[owner].size() > 3) {
                fail("owner rail mapping is not 2/3 endpoints");
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
                client_fds.push_back(fd);
                client_threads.emplace_back(
                    [this, fd]() { HandleWorker(fd); });
            }
        }
    }

    void HandleWorker(int fd) {
        try {
            Request request{};
            recv_exact(fd, &request, sizeof(request));
            if (request.magic !=
                    batchgen::distributed_weights::kProtocolMagic ||
                request.version !=
                    batchgen::distributed_weights::kProtocolVersion ||
                request.operation !=
                    static_cast<std::uint32_t>(Operation::kHello) ||
                request.worker_id >=
                    static_cast<std::uint32_t>(workers)) {
                fail("invalid distributed weights worker hello");
            }
            WaitReady(600.0);
            Response hello;
            hello.staging_bytes = staging_bytes;
            hello.module_bytes = module_bytes;
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
                        Acquire(module_key,
                                static_cast<int>(request.worker_id),
                                &response);
                    } else if (operation.operation ==
                               static_cast<std::uint32_t>(
                                   Operation::kRelease)) {
                        Release(module_key,
                                static_cast<int>(request.worker_id),
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
            if (!stop.load()) {
                RecordFailure(
                    "worker control connection failed: " +
                    std::string(error.what()));
            }
        }
    }

    void Acquire(const std::string& module_key, int worker_id,
                 Response* response) {
        const auto [layer, expert] = parse_module_key(module_key);
        const int owner = expert / kExpertsPerOwner;
        if (owner == node_rank) {
            fail("daemon received a local module acquire: " +
                 module_key);
        }
        const std::uint32_t worker_bit =
            static_cast<std::uint32_t>(1) << worker_id;
        const Clock::time_point begin = Clock::now();
        std::unique_lock<std::mutex> lock(ring_mutex);
        const bool available = ring_cv.wait_for(
            lock, std::chrono::seconds(300), [&]() {
            if (failed.load() || stop.load()) {
                return true;
            }
            const auto found = module_to_slot.find(module_key);
            return found != module_to_slot.end() &&
                   slots[found->second].state ==
                       SlotState::kReady;
        });
        if (!available) {
            fail("timed out waiting for remote module: " + module_key);
        }
        if (failed.load()) {
            fail(FailureMessage());
        }
        if (stop.load()) {
            fail("distributed weights daemon is stopping");
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
        if (slot.released_mask == all_workers_mask) {
            module_to_slot.erase(found);
            slot = Slot{};
            ring_cv.notify_all();
        }
        lock.unlock();
        std::lock_guard<std::mutex> stats_lock(stats_mutex);
        ++release_count;
    }

    std::vector<std::string> BuildSchedule() const {
        std::vector<std::string> schedule;
        schedule.reserve(
            kLayers * (kExperts - kExpertsPerOwner));
        for (int layer = 1; layer <= kLayers; ++layer) {
            for (int expert = 0; expert < kExperts; ++expert) {
                if (expert / kExpertsPerOwner == node_rank) {
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
                    kExpertsPerOwner +
                static_cast<std::uint64_t>(
                    expert % kExpertsPerOwner)) *
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

    void PrefetchLoop() {
        const std::vector<std::string> schedule = BuildSchedule();
        size_t next = 0;
        int tail = 0;
        std::deque<Chunk> chunks;

        while (!stop.load()) {
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
                    first_expert / kExpertsPerOwner;
                if (next + superchunk > schedule.size()) {
                    fail("truncated final superchunk");
                }
                for (int index = 0; index < superchunk; ++index) {
                    const auto [next_layer, next_expert] =
                        parse_module_key(schedule[next + index]);
                    if (next_layer != layer ||
                        next_expert != first_expert + index ||
                        next_expert / kExpertsPerOwner != owner) {
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
        {
            std::lock_guard<std::mutex> lock(stats_mutex);
            summary["schema"] = "k3-distributed-weights/1";
            summary["node_rank"] = node_rank;
            summary["network_ready"] = network_ready.load();
            summary["failed"] = failed.load();
            summary["failure"] = FailureMessage();
            summary["depth"] = depth;
            summary["superchunk"] = superchunk;
            summary["outstanding"] = outstanding;
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

    void Stop() {
        if (!started.load()) {
            return;
        }
        stop.store(true);
        ready_cv.notify_all();
        ring_cv.notify_all();
        if (unix_listener >= 0) {
            shutdown(unix_listener, SHUT_RDWR);
            close(unix_listener);
            unix_listener = -1;
        }
        for (int& listener : tcp_listeners) {
            if (listener > 0) {
                shutdown(listener, SHUT_RDWR);
                close(listener);
                listener = -1;
            }
        }
        {
            std::lock_guard<std::mutex> lock(client_mutex);
            for (int fd : client_fds) {
                shutdown(fd, SHUT_RDWR);
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
            std::lock_guard<std::mutex> lock(client_mutex);
            for (std::thread& thread : client_threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
            for (int fd : client_fds) {
                close(fd);
            }
            client_fds.clear();
        }
        if (summary_thread.joinable()) {
            summary_thread.join();
        }

        for (int rail = 0; rail < kRails; ++rail) {
            for (int owner = 0; owner < kNodes; ++owner) {
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

void DistributedWeightDaemon::Stop() { impl_->Stop(); }
