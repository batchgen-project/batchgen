"""CPU-only regressions for the weight daemon's rail selection.

The prefetch loop posts each remote superchunk on the rail that currently holds
the fewest bytes in flight among the rails reaching that module's owner, not on
the next rail of a rotating cursor. The rails do not have equal throughput, so
an equal-count rotation makes a group of rails run near the rate of its slowest
member, and when an owner is reached over an even number of rails the cursor
also pins each of that owner's workers to one rail for the life of the daemon,
because the schedule alternates those workers within a layer.

The load is derived from the prefetch thread's own in-flight deque rather than
from a per-rail counter, so there is no second account to keep in step with the
post, the two completion branches and the drain. Ties take the lowest rail
index, which keeps the assignment reproducible and reproduces the old
behaviour on an empty ring.

The first half of this file pins that selection in the C++ source; the second
half models the rule, including why de-aliasing the cursor alone would not have
helped: with equal byte counts both rails of an owner stay limited by the
slower one.
"""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DAEMON = ROOT / "core/Weights_Storage/distributed_weight_daemon.cpp"


def _section(source, begin, end):
    start = source.index(begin)
    return source[start:source.index(end, start)]


def _posting_loop():
    """The inner posting loop of PrefetchLoop, nothing before or after it."""
    return _section(
        DAEMON.read_text(),
        "while (static_cast<int>(chunks.size()) < outstanding) {",
        "if (!posted && chunks.empty()) {",
    )


def _selection_helper():
    return _section(
        DAEMON.read_text(),
        "int LeastLoadedRail(int owner, const std::deque<Chunk>& in_flight) const {",
        "void PrefetchLoop() {",
    )


def test_the_rotating_cursor_is_gone():
    daemon = DAEMON.read_text()
    assert "owner_rail_cursor" not in daemon
    assert "% available.size()" not in daemon
    # No per-owner rail counter of any other name survives either.
    assert not re.search(r"std::vector<int>\s+owner_rail\w*;", daemon)
    assert "std::vector<std::vector<int>> owner_rails;" in daemon


def test_posting_loop_asks_the_helper_and_resolves_no_rail_itself():
    loop = _posting_loop()
    assert "chunk.rail = LeastLoadedRail(owner, chunks);" in loop
    # The candidate set is resolved inside the helper, so the loop itself must
    # not reach into the owner-to-rail map any more.
    assert "owner_rails[" not in loop
    assert loop.count("chunk.rail =") == 1


def test_selection_is_least_in_flight_bytes_over_the_in_flight_deque():
    helper = _selection_helper()
    # Candidates are exactly the rails that reach the owner.
    assert "const std::vector<int>& available = owner_rails[owner];" in helper
    # Load is derived from the deque the loop already keeps, as whole-chunk
    # bytes, and the scan moves only on a strict improvement.
    assert "for (const Chunk& chunk : in_flight) {" in helper
    assert re.search(
        r"rail_bytes\[chunk\.rail\]\s*\+=\s*"
        r"static_cast<std::uint64_t>\(chunk\.count\)\s*\*\s*module_bytes;",
        helper,
    )
    # (load, rail index), so the tie-break does not depend on the order the
    # bootstrap filled this owner's candidates in.
    assert re.search(
        r"if \(rail_bytes\[rail\] < chosen_bytes \|\|\s*"
        r"\(rail_bytes\[rail\] == chosen_bytes && rail < chosen\)\) \{",
        helper,
    )
    # A chunk without a rail is a bookkeeping error, not a case to absorb.
    assert "if (chunk.rail < 0 || chunk.rail >= kRails) {" in helper
    assert 'fail("in-flight distributed weight fetch has no rail");' in helper
    assert "for (const int rail : available) {" in helper
    assert "int chosen = available.front();" in helper
    assert "return chosen;" in helper


def test_the_load_vector_is_local_to_the_selection_and_never_decremented():
    daemon = DAEMON.read_text()
    helper = _selection_helper()
    # Declared inside the selection, so nothing else can hold stale load.
    assert "std::array<std::uint64_t, kRails> rail_bytes{};" in helper
    assert daemon.count("rail_bytes") == helper.count("rail_bytes")
    # A derived sum has no decrement site to miss; assert none was added.
    assert "rail_bytes[" in helper
    assert not re.search(r"rail_bytes\[[^\]]*\]\s*-=", daemon)
    # No member of the daemon holds per-rail load or a rail cursor under any
    # name, so the helper's local sum is the only such state in the file.
    members = _section(daemon, "    std::string config_path;", "    int store_fd = -1;")
    members += _section(daemon, "    int store_fd = -1;", "    std::thread bootstrap_thread;")
    for line in members.splitlines():
        declaration = line.strip()
        if not declaration or declaration.startswith("//"):
            continue
        assert not re.search(
            r"(?i)(cursor|rail_bytes|rail_load|next_rail|outstanding_bytes"
            r"|in_flight_bytes)",
            declaration,
        ), declaration


def test_an_owner_no_rail_reaches_fails_by_name_before_any_load_is_summed():
    helper = _selection_helper()
    empty = helper.index("if (available.empty()) {")
    named = helper.index('fail("no distributed weight rail reaches owner "', empty)
    diagnostic = helper[named:named + 260]
    assert "std::to_string(owner)" in diagnostic
    # The node rank says which side of the link map is wrong.
    assert "std::to_string(node_rank)" in diagnostic
    summed = helper.index("std::array<std::uint64_t, kRails> rail_bytes{};")
    assert empty < named < summed


def test_every_index_the_helper_uses_is_checked_before_it_indexes():
    helper = _selection_helper()
    owner_guard = helper.index("if (owner < 0 || owner >= num_nodes) {")
    candidates = helper.index("const std::vector<int>& available = owner_rails[owner];")
    rail_guard = helper.index("if (rail < 0 || rail >= kRails) {")
    summed = helper.index("std::array<std::uint64_t, kRails> rail_bytes{};")
    # The owner is checked before it indexes owner_rails, and every candidate
    # rail before any rail indexes the load array.
    assert owner_guard < candidates < rail_guard < summed
    assert 'fail("distributed weight rail requested for owner "' in helper
    assert 'fail("distributed weight owner "' in helper


def test_the_selection_sits_between_the_slot_gate_and_the_posted_fetch():
    loop = _posting_loop()
    choice = loop.index("chunk.rail = LeastLoadedRail(owner, chunks);")
    gate = loop.index("if (!SlotsFree(tail, superchunk)) {")
    # The ring lock and the fetch that matter are the ones AFTER the choice:
    # both strings also occur earlier in the loop.
    ring_lock = loop.index("std::lock_guard<std::mutex> lock(ring_mutex);", choice)
    posted = loop.index("chunk.request = ucp_get_nbx(", choice)
    assert gate < choice < ring_lock < posted


def test_the_endpoint_the_key_and_the_staging_handle_all_use_the_chosen_rail():
    # Choosing one rail and then reading another rail's endpoint, rkey or
    # memory handle is the one mistake here that could change what lands in a
    # slot, and every other assertion in this file would still pass.
    loop = _posting_loop()
    choice = loop.index("chunk.rail = LeastLoadedRail(owner, chunks);")
    endpoint = loop.index("remotes[chunk.rail][owner];", choice)
    memh = loop.index("params.memh = rails[chunk.rail].staging_memh;", choice)
    fetch = loop.index("chunk.request = ucp_get_nbx(", memh)
    assert choice < endpoint < memh < fetch
    # The key and the source address are that same endpoint's, so one rail
    # supplies the endpoint, the key, the address and the staging handle.
    assert "remote.source_address +" in loop[endpoint:memh]
    assert "remote.rkey" in loop[fetch:]
    # And the rail changes nothing about WHICH module or WHICH slot: the
    # remote offset comes from the schedule cursor, the destination and the
    # recorded slot from the ring position.
    assert "RemoteOffset(schedule[next]);" in loop[endpoint:memh]
    assert re.search(
        r"static_cast<char\*>\(staging\) \+\s*"
        r"static_cast<std::uint64_t>\(tail\) \* module_bytes;",
        loop[endpoint:memh],
    )
    assert "chunk.slot_start = tail;" in loop[:choice]


def test_completions_are_retired_before_the_posting_loop_runs():
    # The derived load is only as fresh as the scan that precedes it, so the
    # scan must stay ahead of posting, and both completion branches must erase
    # the chunk rather than leave it counted.
    body = _section(
        DAEMON.read_text(),
        "void PrefetchLoop() {",
        "    // The summary's snapshots.",
    )
    scan = body.index("for (auto iterator = chunks.begin();")
    posting = body.index("while (static_cast<int>(chunks.size()) < outstanding) {")
    assert scan < posting
    immediate = body.index("if (iterator->request == nullptr) {", scan)
    polled = body.index("const ucs_status_t status =", scan)
    assert scan < immediate < polled < posting
    assert body[immediate:polled].count("iterator = chunks.erase(iterator);") == 1
    assert body[polled:posting].count("iterator = chunks.erase(iterator);") == 1


def test_a_posted_fetch_is_in_the_deque_before_the_next_rail_is_chosen():
    # Several fetches can be posted in one pass; each must count toward the
    # next choice, which is only true if the push_back happens inside the
    # same iteration as the post.
    loop = _posting_loop()
    fetch = loop.index("chunk.request = ucp_get_nbx(")
    pushed = loop.index("chunks.push_back(chunk);", fetch)
    advance = loop.index("tail = (tail + superchunk) % depth;", pushed)
    assert fetch < pushed < advance
    # Nothing between the post and the push can leave the loop body.
    assert "continue;" not in loop[fetch:pushed]
    assert "break;" not in loop[fetch:pushed]


def test_the_reset_drain_empties_the_deque_it_accounts_from():
    drain = _section(
        DAEMON.read_text(),
        "std::uint64_t DrainForReset(std::deque<Chunk>* chunks) {",
        "int LeastLoadedRail(",
    )
    assert "chunks->clear();" in drain
    # The clear is the last thing the drain does to the deque, so no entry
    # survives a cohort swap to bias the next cohort's first choices.
    assert drain.index("chunks->clear();") > drain.index("for (Chunk& chunk : *chunks) {")


def test_a_cohort_reset_no_longer_rewinds_any_rail_state():
    daemon = DAEMON.read_text()
    reset = _section(daemon, "if (reset_request != 0) {", "const Clock::time_point scan_start")
    assert "std::fill(" not in reset
    assert "rail" in reset  # the comment that explains why nothing is rewound


# --- the rule itself, as a model ------------------------------------------
#
# Mirrors PrefetchLoop's structure rather than only its arithmetic: a scan that
# retires finished fetches, then a posting phase that runs while the number in
# flight is under the global cap, with each just-posted fetch counted before
# the next choice. Each rail is modelled as a server that moves one fetch at a
# time at its own rate, so a slow rail's queue is what grows.


def least_loaded(available, load):
    """The implemented rule: fewest in-flight bytes, lowest index on a tie.

    The comparison is on (load, rail index), as in the helper, so the result
    does not depend on the order the candidates are stored in.
    """
    chosen = available[0]
    for rail in available:
        if load[rail] < load[chosen] or (load[rail] == load[chosen] and rail < chosen):
            chosen = rail
    return chosen


class RoundRobin:
    """The rule being replaced: one cursor per owner, capacity-oblivious."""

    def __init__(self):
        self.cursor = {}

    def __call__(self, owner, available, load):
        index = self.cursor.get(owner, 0)
        self.cursor[owner] = index + 1
        return available[index % len(available)]


def run_model(schedule, owner_rails, rate, outstanding=24, chunk_bytes=1, rule=None):
    """Post `schedule` (owner per superchunk) and return what each rail carried.

    `rule(owner, available, load) -> rail`. Returns (posted_bytes_per_rail,
    choices, final_load, elapsed).
    """
    rails = sorted(rate)
    load = {rail: 0 for rail in rails}
    free_at = {rail: 0.0 for rail in rails}
    posted = {rail: 0 for rail in rails}
    choices = []
    in_flight = []  # (done_at, rail, bytes)
    now = 0.0
    index = 0
    while index < len(schedule) or in_flight:
        # Scan: retire every fetch the loop would now observe as complete.
        landed = [entry for entry in in_flight if entry[0] <= now]
        for done_at, rail, size in landed:
            load[rail] -= size
        in_flight = [entry for entry in in_flight if entry[0] > now]
        # Post while under the global in-flight cap.
        progressed = False
        while index < len(schedule) and len(in_flight) < outstanding:
            owner = schedule[index]
            available = owner_rails[owner]
            rail = rule(owner, available, load)
            assert rail in available
            start = max(now, free_at[rail])
            done_at = start + chunk_bytes / rate[rail]
            free_at[rail] = done_at
            in_flight.append((done_at, rail, chunk_bytes))
            load[rail] += chunk_bytes
            posted[rail] += chunk_bytes
            choices.append((owner, index, rail))
            index += 1
            progressed = True
        if in_flight and (not progressed or index == len(schedule)):
            now = min(entry[0] for entry in in_flight)
        elif not in_flight and index == len(schedule):
            break
    return posted, choices, load, now


def test_the_least_loaded_available_rail_wins():
    assert least_loaded([2, 5], {2: 8, 5: 3}) == 5
    assert least_loaded([2, 5], {2: 1, 5: 9}) == 2
    assert least_loaded([1, 4, 7], {1: 6, 4: 2, 7: 4}) == 4


def test_equal_load_and_a_cold_ring_both_take_the_lowest_rail():
    assert least_loaded([2, 5], {2: 4, 5: 4}) == 2
    assert least_loaded([1, 4, 7], {1: 0, 4: 0, 7: 0}) == 1
    # Lowest index, not the first stored candidate.
    assert least_loaded([3, 0, 6], {0: 0, 3: 0, 6: 0}) == 0


def test_bytes_leave_a_rail_when_its_fetch_completes_and_the_ring_ends_empty():
    owner_rails = {1: [2, 5]}
    posted, choices, load, elapsed = run_model(
        [1] * 64, owner_rails, {2: 1.0, 5: 2.0}, outstanding=8, rule=lambda o, a, l: least_loaded(a, l)
    )
    assert sum(posted.values()) == 64
    assert len(choices) == 64
    assert load == {2: 0, 5: 0}
    assert elapsed > 0


def test_a_fetch_that_lands_instantly_still_counts_until_the_next_scan():
    # A get whose request comes back already complete stays in the deque until
    # the next completion scan, so within one posting pass it still counts.
    owner_rails = {1: [2, 5]}
    rate = {2: float("inf"), 5: float("inf")}
    posted, choices, load, _ = run_model(
        [1] * 4, owner_rails, rate, outstanding=4,
        rule=lambda o, a, l: least_loaded(a, l),
    )
    assert [rail for _owner, _index, rail in choices] == [2, 5, 2, 5]
    assert posted == {2: 2, 5: 2}
    assert load == {2: 0, 5: 0}


def test_several_posts_in_one_pass_spread_without_waiting_for_a_completion():
    # No completion can be observed inside a posting pass, so the spread comes
    # only from counting each just-posted fetch.
    owner_rails = {1: [2, 5]}
    rate = {2: 1.0, 5: 2.0}
    _posted, choices, _load, _elapsed = run_model(
        [1] * 6, owner_rails, rate, outstanding=6,
        rule=lambda o, a, l: least_loaded(a, l),
    )
    assert [rail for _owner, _index, rail in choices][:6] == [2, 5, 2, 5, 2, 5]


def test_a_drain_resets_the_derived_load_and_the_next_cohorts_first_choice():
    available = [2, 5]
    load = {2: 0, 5: 0}
    chosen = []
    for _ in range(3):
        rail = least_loaded(available, load)
        chosen.append(rail)
        load[rail] += 1
    assert chosen == [2, 5, 2]
    assert load == {2: 2, 5: 1}
    # DrainForReset ends in chunks->clear(), so nothing is in flight: every
    # rail reads as unloaded and the next cohort starts at the lowest rail,
    # which is what the deleted cursor rewind used to arrange.
    load = {rail: 0 for rail in available}
    assert least_loaded(available, load) == 2


def test_only_rails_of_the_owner_are_ever_chosen():
    owner_rails = {1: [2, 5], 2: [1, 4, 7], 3: [0, 3, 6]}
    rate = {0: 2.0, 1: 2.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 2.0, 6: 2.0, 7: 1.0}
    schedule = [1, 2, 3] * 48
    posted, choices, load, _ = run_model(
        schedule, owner_rails, rate, rule=lambda o, a, l: least_loaded(a, l)
    )
    for owner, _index, rail in choices:
        assert rail in owner_rails[owner]
    assert posted[1] + posted[4] + posted[7] == 48
    assert load == {rail: 0 for rail in rate}


def test_the_byte_split_follows_rail_speed_while_rotation_splits_it_evenly():
    owner_rails = {1: [2, 5]}
    for slow, fast in ((1.0, 2.0), (1.0, 1.6)):
        rate = {2: slow, 5: fast}
        posted, _choices, _load, elapsed = run_model(
            [1] * 600, owner_rails, rate, rule=lambda o, a, l: least_loaded(a, l)
        )
        share = posted[5] / (posted[2] + posted[5])
        expected = fast / (slow + fast)
        assert abs(share - expected) < 0.05, (share, expected)

        rotation = RoundRobin()
        rr_posted, _c, _l, rr_elapsed = run_model(
            [1] * 600, owner_rails, rate, rule=rotation
        )
        # Equal counts regardless of speed: the rotation hands the slow rail
        # exactly half the bytes, which is why de-aliasing the cursor alone
        # (each worker striping over both rails) would not have helped — both
        # workers would simply be limited by the slower rail.
        assert rr_posted[2] == rr_posted[5] == 300
        assert rr_elapsed > elapsed


def test_equal_speed_rails_split_evenly_under_both_rules():
    owner_rails = {1: [2, 5]}
    rate = {2: 1.0, 5: 1.0}
    posted, _c, _l, elapsed = run_model(
        [1] * 400, owner_rails, rate, rule=lambda o, a, l: least_loaded(a, l)
    )
    assert abs(posted[2] - posted[5]) <= 2
    rr_posted, _c2, _l2, rr_elapsed = run_model(
        [1] * 400, owner_rails, rate, rule=RoundRobin()
    )
    assert rr_posted[2] == rr_posted[5] == 200
    assert abs(rr_elapsed - elapsed) / elapsed < 0.05


def test_rotation_pins_each_worker_of_a_two_rail_owner_to_one_rail():
    # The four-node shape: one owner per node is reached over two rails while
    # the schedule alternates that owner's two workers, so a mod-2 cursor gives
    # each worker the same rail every time. Least-loaded does not.
    owner_rails = {1: [2, 5]}
    rate = {2: 1.0, 5: 2.0}
    workers = [0, 1] * 100
    schedule = [1] * len(workers)

    def rails_per_worker(rule):
        _posted, choices, _load, _elapsed = run_model(
            schedule, owner_rails, rate, rule=rule
        )
        seen = {0: set(), 1: set()}
        for _owner, index, rail in choices:
            seen[workers[index]].add(rail)
        return seen

    pinned = rails_per_worker(RoundRobin())
    assert pinned == {0: {2}, 1: {5}}
    spread = rails_per_worker(lambda o, a, l: least_loaded(a, l))
    assert spread[0] == {2, 5} and spread[1] == {2, 5}
