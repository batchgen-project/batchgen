"""CPU-only source regressions for the weight producer's receive-ahead.

With copies in flight the HtoD producer keeps one waiting acquire outstanding
on the primary node-daemon connection instead of draining the copy engine to
idle before it waits. Releases go out on a second, release-only connection
(protocol version 5), so a landing never waits behind that acquire. The
acquire's in-flight entry holds its reserved slot but no copy event until the
reply leases the module, so landing steps over it (the copies behind it are
other types' local modules and land at once) and its claim moves it to the back
so recorded entries stay in stream order; a stop or a failure drains
the reply and claims a lease in it, never releasing it at the cursor. Every
access to the worker's lease index holds lease_index_mutex_.
"""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "core/HtoD_Engine/HtoD_Engine.cu"
ENGINE_HEADER = ROOT / "core/HtoD_Engine/HtoD_Engine.h"
STORAGE = ROOT / "core/Weights_Storage/Weights_Storage.cpp"
STORAGE_HEADER = ROOT / "core/Weights_Storage/Weights_Storage.h"
DAEMON = ROOT / "core/Weights_Storage/distributed_weight_daemon.cpp"
PROTOCOL = ROOT / "core/Weights_Storage/distributed_weights_protocol.h"


def _section(source, begin, end):
    start = source.index(begin)
    return source[start:source.index(end, start)]


def _worker_body():
    return _section(
        ENGINE.read_text(),
        "void HtoD_Engine::HtoD_Worker()",
        "void HtoD_Engine::set_weight_copy_queue",
    )


def test_protocol_version_release_flag_and_ring_bound():
    protocol = PROTOCOL.read_text()
    assert "constexpr std::uint32_t kProtocolVersion = 5;" in protocol
    assert "// Version 5:" in protocol
    assert "constexpr std::uint32_t kAcquireNoWait = 1;" in protocol
    assert "constexpr std::uint32_t kHelloReleaseChannel = 2;" in protocol
    assert "constexpr std::uint32_t kMaxLeasesWhileAcquiring = 3;" in protocol

    # The producer's pipeline and the daemon's bound are one number.
    header = ENGINE_HEADER.read_text()
    assert re.search(
        r"static_assert\(\s*batchgen::distributed_weights::kMaxLeasesWhileAcquiring ==\s*"
        r"kMaxCopiesInFlight - 1,",
        header,
    )

    daemon = DAEMON.read_text()
    bound = daemon.index("static_cast<std::int64_t>(workers) * kMaxLeasesWhileAcquiring")
    assert "static_cast<std::int64_t>(depth) <=" in daemon[bound - 80:bound]
    named = daemon.index('fail("distributed weight ring too small for receive-ahead: depth "', bound)
    staging = daemon.index("staging_bytes =", named)
    assert bound < named < staging


def test_daemon_admits_a_release_connection_only_for_an_admitted_worker():
    daemon = DAEMON.read_text()
    admit = _section(daemon, "std::string AdmitWorker(", "void WaitReadyForWorker(")
    # An unknown hello flag is refused before any cohort state is read.
    flags = admit.index("(request.flags & ~kHelloReleaseChannel) != 0")
    lock = admit.index("std::lock_guard<std::mutex> lock(cohort_mutex);")
    assert flags < lock
    release = admit[admit.index("if (release_only) {"):admit.index("} else {")]
    primary = admit[admit.index("} else {"):admit.index("connection->epoch = cohort_epoch;")]
    # Same worker, same epoch: the primary bit must be set already, and the
    # release connection sets none of its own in the admitted mask.
    assert "(cohort_admitted_mask & bit) == 0" in release
    assert "cohort_release_mask |= bit;" in release
    assert "connection->release_only = true;" in release
    assert "cohort_admitted_mask |= bit" not in release
    assert "cohort_admitted_mask |= bit;" in primary
    # A swap admits a fresh cohort: both masks are cleared.
    swap = _section(daemon, "std::string FinishCohortSwap(", "void PrepareStop()")
    assert "cohort_admitted_mask = 0;" in swap
    assert "cohort_release_mask = 0;" in swap


def test_daemon_release_connection_carries_release_only():
    daemon = DAEMON.read_text()
    handler = _section(
        daemon, "void HandleWorker(ClientConnection* connection)",
        "void CloseUnadmitted(",
    )
    # No staging memfd and no network wait for the release connection.
    branch = handler.index("if (release_only) {")
    plain = handler.index("send_exact(fd, &hello, sizeof(hello));", branch)
    other = handler.index("} else {", plain)
    wait = handler.index("WaitReadyForWorker(600.0);", other)
    memfd = handler.index("send_response_with_fd(fd, hello, staging_fd);", wait)
    assert branch < plain < other < wait < memfd
    # Any other operation is a named protocol error, which ends the
    # connection; ending it counts as the worker's loss.
    loop = handler.index("while (!stop.load()) {")
    only = handler.index("if (release_only &&", loop)
    named = handler.index('"distributed weights protocol error: the release "', only)
    dispatch = handler.index("Operation::kAcquire)) {", only)
    assert loop < only < named < dispatch
    caught = handler[handler.rindex("} catch (const std::exception& error) {"):]
    assert "RecordWorkerLoss(epoch, worker_id, error.what());" in caught
    # A swap and Stop join every listed connection, the release one included.
    assert "bool release_only = false;" in daemon


def test_worker_opens_the_release_connection_after_the_primary_hello():
    storage = STORAGE.read_text()
    init = _section(
        storage,
        "void Weights_Storage::InitDistributed(",
        "void Weights_Storage::begin_acquire_locked_(",
    )
    primary = init.index("connect_daemon(&this->daemon_socket_, socket_path);")
    hello = init.index("recv_fd_with_response(this->daemon_socket_, &response);", primary)
    admitted = init.index("response.generation != cohort_epoch", hello)
    release = init.index("connect_daemon(&this->release_socket_, socket_path);", admitted)
    flag = init.index("batchgen::distributed_weights::kHelloReleaseChannel;", release)
    sent = init.index("send_exact(this->release_socket_, &release_hello,", flag)
    ready = init.index("this->distributed_ = true;", sent)
    assert primary < hello < admitted < release < flag < sent < ready

    release_module = _section(
        storage, "void Weights_Storage::release_module(", "void Weights_Storage::Init("
    )
    assert "std::lock_guard<std::mutex> lock(this->release_mutex_);" in release_module
    assert "send_exact(this->release_socket_, &request, sizeof(request));" in release_module
    assert "recv_exact(this->release_socket_, &response, sizeof(response));" in release_module
    assert "daemon_socket_" not in release_module
    assert "daemon_mutex_" not in release_module
    destructor = _section(
        storage, "Weights_Storage::~Weights_Storage()", "void Weights_Storage::InitDistributed("
    )
    assert "close(this->release_socket_);" in destructor
    header = STORAGE_HEADER.read_text()
    assert "std::mutex release_mutex_;" in header
    assert "int release_socket_ = -1;" in header


def test_two_phase_acquire_completes_a_reply_across_partial_reads():
    storage = STORAGE.read_text()
    reply = _section(
        storage,
        "Weights_Storage::AcquireReply Weights_Storage::receive_acquire_reply_locked_(",
        "Weights_Storage::acquire_remote_module(",
    )
    # One byte buffer per outstanding acquire, read without blocking unless
    # asked to; a would-block read leaves the partial frame in place.
    assert "block ? MSG_WAITALL : MSG_DONTWAIT" in reply
    assert "this->acquire_reply_bytes_ += static_cast<size_t>(received);" in reply
    would_block = reply.index("(errno == EAGAIN || errno == EWOULDBLOCK)")
    assert "return AcquireReply::kPending;" in reply[would_block:would_block + 120]

    begin = _section(
        storage,
        "void Weights_Storage::begin_acquire_locked_(",
        "void Weights_Storage::resend_acquire_locked_()",
    )
    # One request at a time on the primary connection.
    assert begin.index("if (!this->acquire_key_.empty()) {") < begin.index(
        "this->resend_acquire_locked_();"
    )

    poll = _section(
        storage,
        "acquire_poll Weights_Storage::poll_acquire(bool block)",
        "acquire_drain Weights_Storage::drain_acquire() noexcept",
    )
    retry = poll.index("case AcquireReply::kRetry:")
    resent = poll.index("this->resend_acquire_locked_();", retry)
    status = poll.index("acquire_poll::Status::kRetrySent", resent)
    assert retry < resent < status

    drain = _section(
        storage,
        "acquire_drain Weights_Storage::drain_acquire() noexcept",
        "std::vector<std::string> Weights_Storage::leased_modules() const",
    )
    # Bounded by one blocking reply; a retry is never re-sent.
    assert "this->receive_acquire_reply_locked_(true, &lease)" in drain
    assert "resend_acquire_locked_" not in drain
    assert "} catch (...) {" in drain

    header = STORAGE_HEADER.read_text()
    assert "void begin_acquire(const std::string& module_key);" in header
    assert "acquire_poll poll_acquire(bool block = false);" in header
    assert "acquire_drain drain_acquire() noexcept;" in header
    assert "bool is_remote_module(const std::string& module_key) const;" in header
    for status in ("kPending,", "kRetrySent,", "kReady,"):
        assert status in _section(header, "struct acquire_poll {", "\n};"), status


def test_producer_issues_one_acquire_ahead_and_polls_around_landing():
    body = _worker_body()
    loop = body.index("while (!terminate_flag_) {")
    before = body.index("progressed |= this->poll_acquire_(false);", loop)
    land = body.index("if (this->land_copies_(false) > 0) {", before)
    after = body.index("progressed |= this->poll_acquire_(false);", land)
    on_demand = body.index("on_demand_task_queue_.try_pop", after)
    assert loop < before < land < after < on_demand

    weights = body[body.index("for (auto& module_type :"):]
    # Slot first, then the entry, marked acquiring, then the one request.
    reserve = weights.index("acquireEmptyBuffer(module_type)")
    begin = weights.index("this->begin_in_flight_(module_type, module_name, buffer_idx,")
    marked = weights.index("this->in_flight_.back().acquiring = true;", begin)
    issued = weights.index("this->weights_storage_.begin_acquire(module_name);", marked)
    assert reserve < begin < marked < issued
    assert weights.count("begin_acquire(") == 1

    engine = ENGINE.read_text()
    poll = _section(engine, "bool HtoD_Engine::poll_acquire_(bool block)",
                    "void HtoD_Engine::drain_acquire_on_stop_()")
    # The entry holds no lease until the reply leases it.
    ready = poll.index("acquire_poll::Status::kReady")
    cleared = poll.index("this->in_flight_[index].acquiring = false;", ready)
    leased = poll.index("this->in_flight_[index].leased = true;", cleared)
    claimed = poll.index("this->claim_module_(index, polled.tensors);", leased)
    assert ready < cleared < leased < claimed


def test_landing_steps_over_the_outstanding_acquire_entry():
    engine = ENGINE.read_text()
    land = _section(
        engine,
        "size_t HtoD_Engine::land_copies_(bool wait)",
        "py::dict HtoD_Engine::weight_stream_status()",
    )
    walk = land.index("InFlightCopy& copy = this->in_flight_[index];")
    unrecorded = land.index("if (!copy.recorded) {", walk)
    step = land.index("++index;", unrecorded)
    stop = land.index("break;", step)
    synchronize = land.index("cudaEventSynchronize(copy.event)")
    query = land.index("cudaEventQuery(copy.event)")
    release = land.index("this->weights_storage_.release_module(copy.module_name);")
    complete = land.index("this->gpu_weight_buffer_.weights_copy_complete(")
    drop = land.index("this->end_in_flight_at_(index);", complete)
    # Checked before any event is waited on or queried: a pooled event keeps
    # stale state, so an unrecorded entry would land an uncopied slot. The
    # acquire's entry is stepped over, not stopped at, so a local module's
    # copy behind it is never held behind the daemon.
    assert walk < unrecorded < step < stop < synchronize < query < release < complete < drop
    assert "if (copy.acquiring) {" in land[unrecorded:step]
    # Landing never pops the front blindly: the entry stepped over stays.
    assert "this->end_in_flight_(true);" not in land
    header = ENGINE_HEADER.read_text()
    assert "bool acquiring = false;" in header
    assert "bool recorded = false;" in header

    # Recorded entries stay in stream order: an acquired-ahead entry moves to
    # the back before its copies are queued behind the ones started after it.
    claim = _section(
        engine,
        "void HtoD_Engine::claim_module_(",
        "bool HtoD_Engine::poll_acquire_(bool block)",
    )
    move = claim.index("if (index + 1 != this->in_flight_.size()) {")
    back = claim.index("this->in_flight_.push_back(std::move(moved));", move)
    first_copy = claim.index("cudaMemcpyAsync(")
    recorded = claim.index("this->in_flight_[index].recorded = true;")
    assert move < back < first_copy < recorded

    # The idle branch waits for the reply only when the acquire is the sole
    # entry; otherwise it waits for the oldest recorded copy.
    body = _worker_body()
    idle = body[body.index("if (!progressed) {"):]
    in_flight = idle.index("if (!this->in_flight_.empty()) {")
    sole = idle.index("if (this->in_flight_.size() == 1 &&", in_flight)
    reply = idle.index("this->poll_acquire_(true);", sole)
    landed = idle.index("this->land_copies_(true);", reply)
    leave = idle.index("continue;", landed)
    park = idle.index("this->producer_wake_cv_.wait(lock,")
    assert in_flight < sole < reply < landed < leave < park


def test_fail_producer_and_stop_drain_the_outstanding_acquire():
    engine = ENGINE.read_text()
    fail = _section(
        engine,
        "void HtoD_Engine::fail_producer_(",
        "size_t HtoD_Engine::land_copies_(bool wait)",
    )
    sync = fail.index("cudaStreamSynchronize(this->HtoD_stream)")
    drain = fail.index("this->drain_acquire_on_failure_(&failure);")
    walk = fail.index("for (InFlightCopy& copy : this->in_flight_) {")
    release = fail.index("this->weights_storage_.release_module(copy.module_name)")
    assert sync < drain < walk < release

    on_failure = _section(
        engine,
        "void HtoD_Engine::drain_acquire_on_failure_(std::string* failure) noexcept {",
        "void HtoD_Engine::HtoD_Worker()",
    )
    drained = on_failure.index("this->weights_storage_.drain_acquire();")
    leased = on_failure.index("copy.leased = true;", drained)
    claim = on_failure.index("this->gpu_weight_buffer_.claim_for_copy(", leased)
    pop = on_failure.index("queue.try_pop(head);", claim)
    push = on_failure.index("queue.push(copy.module_name);", pop)
    skip = on_failure.index("if (claim == WeightProduction::kSkip) {", push)
    retire = on_failure.index("this->gpu_weight_buffer_.retire_skip(", skip)
    # A lease is claimed, moving the cursor past the module, before anything
    # releases it; a retry leaves the module at its cursor.
    assert drained < leased < claim < pop < push < skip < retire
    assert on_failure.index("if (!drained.leased) {") < leased
    assert "throw" not in on_failure
    assert "} catch (...) {" in on_failure
    header = ENGINE_HEADER.read_text()
    assert "void drain_acquire_on_failure_(std::string* failure) noexcept;" in header

    on_stop = _section(
        engine,
        "void HtoD_Engine::drain_acquire_on_stop_()",
        "void HtoD_Engine::drain_acquire_on_failure_(",
    )
    drained = on_stop.index("this->weights_storage_.drain_acquire();")
    retry = on_stop.index("if (!drained.leased) {", drained)
    retry_path = on_stop[retry:on_stop.index("return;", retry)]
    claimed = on_stop.index("this->claim_module_(index, drained.tensors);", retry)
    assert drained < retry < claimed
    # A retry acquired nothing: the slot is abandoned and the entry dropped,
    # with no release and no queue change.
    assert "this->gpu_weight_buffer_.abandon(" in retry_path
    assert "this->end_in_flight_at_(index);" in retry_path
    for token in ("release_module", ".push(", ".try_pop(", "claim_for_copy"):
        assert token not in retry_path, token

    body = _worker_body()
    stop = body.index("this->drain_acquire_on_stop_();")
    landing = body.rindex("while (!this->in_flight_.empty()) {")
    assert stop < landing


def test_every_lease_index_access_holds_the_lease_index_mutex():
    storage = STORAGE.read_text()
    index_lock = "std::lock_guard<std::mutex> index_lock(this->lease_index_mutex_);"
    starts = [
        match.start()
        for match in re.finditer(
            r"^\S[^\n]*Weights_Storage::\w+\(", storage, re.MULTILINE
        )
    ]
    accesses = [match.start() for match in re.finditer(r"this->active_leases_", storage)]
    assert len(accesses) >= 6
    for access in accesses:
        start = max(begin for begin in starts if begin < access)
        assert index_lock in storage[start:access], storage[start:access][:120]
    header = STORAGE_HEADER.read_text()
    comment = header[:header.index("std::unordered_map<std::string, active_lease> active_leases_;")]
    assert "Every access, read or write, holds lease_index_mutex_" in comment
    assert "read under either" not in header
