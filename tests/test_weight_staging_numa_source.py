"""CPU-only regressions for consumer-NUMA placement of the weight staging ring.

Without `worker_numa_nodes` the daemon first-touches the staging memfd
interleaved over every NUMA node, so about half of every NIC write into staging
and every GPU HtoD read from it crosses the socket link. With it, every
superchunk group of the ring is bound before first touch to the node of the
worker that consumes it.

That works without changing the posting loop because the schedule posts one
superchunk per remote worker in ascending worker order while the posting cursor
and the ring tail advance and rewind together: ring group g always receives a
chunk for remote worker g % R. The model at the end of this file replays the
schedule and the cursor/tail walk over two passes for every node rank and checks
that claim, and the daemon re-checks the consumer of every posted chunk.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DAEMON = ROOT / "core/Weights_Storage/distributed_weight_daemon.cpp"

LAYERS, EXPERTS, WORKERS, SUPERCHUNK, DEPTH = 92, 896, 8, 8, 384
GPU_NUMA = [0, 0, 0, 0, 1, 1, 1, 1]


def _section(source, begin, end):
    start = source.index(begin)
    return source[start:source.index(end, start)]


def test_the_key_is_optional_and_validated_per_entry():
    parse = _section(DAEMON.read_text(), 'if (config.contains("worker_numa_nodes")) {',
                     'if (config.contains("rma_lanes")) {')
    assert "!nodes.is_array()" in parse
    assert "!entry.is_number_integer()" in parse
    assert "entry.get<std::int64_t>() > 63" in parse


def test_the_layout_is_built_only_when_the_key_is_present():
    daemon = DAEMON.read_text()
    assert "if (!worker_numa_nodes.empty()) {\n            BuildSlotNumaLayout();" in daemon
    layout = _section(daemon, "void BuildSlotNumaLayout() {", "UcpRail CreateRail(")
    assert "static_cast<int>(worker_numa_nodes.size()) != workers" in layout
    assert "groups % rounds != 0" in layout
    assert "schedule_groups % groups != 0" in layout
    # Rules apply only when the consumers span more than one node.
    assert layout.index("if (one_node) {") < layout.index("groups % rounds != 0")


def test_absent_key_keeps_the_interleaved_first_touch():
    alloc = _section(DAEMON.read_text(), "void AllocateMappings() {", "void BuildSlotNumaLayout() {")
    interleave = _section(alloc, "if (slot_numa.empty()) {", "return;")
    assert "numa_set_interleave_mask(numa_all_nodes_ptr);" in interleave
    assert "std::memset(staging, 0, staging_bytes);" in interleave


def test_every_run_is_bound_before_the_first_touch_and_verified_after():
    alloc = _section(DAEMON.read_text(), "void AllocateMappings() {", "void BuildSlotNumaLayout() {")
    bound = alloc[alloc.index("return;"):]
    mbind = bound.index("syscall(SYS_mbind,")
    touch = bound.index("std::memset(staging, 0, staging_bytes);")
    verify = bound.index("get_mempolicy(&node, nullptr, 0,")
    assert mbind < touch < verify
    assert "MPOL_BIND, &mask, sizeof(mask) * 8 + 1, 0)" in bound
    assert "MPOL_F_NODE | MPOL_F_ADDR" in bound
    assert "node != slot_numa[start]" in bound
    # Verification stays clear of the huge page that can straddle a boundary.
    assert "static_cast<std::uint64_t>(start) * module_bytes + kHugePage" in bound
    assert "static_cast<std::uint64_t>(end) * module_bytes - kHugePage" in bound


def test_every_posted_chunk_is_checked_against_its_slot_node():
    daemon = DAEMON.read_text()
    loop = _section(daemon, "while (static_cast<int>(chunks.size()) < outstanding) {",
                    "chunk.request = ucp_get_nbx(")
    target = loop.index("chunk.target_worker = OwningWorker(schedule[next]);")
    check = loop.index("slot_numa[tail] !=")
    assert target < check
    assert "worker_numa_nodes[chunk.target_worker]" in loop[check:]


def test_summary_and_trace_header_state_the_policy():
    daemon = DAEMON.read_text()
    assert 'summary["staging_numa"] =' in daemon
    assert '<< " staging_numa="' in daemon
    assert daemon.count('slot_numa.empty() ? "interleave" : "by_consumer"') == 2


# Model: the daemon's schedule, the cursor/tail walk and the slot layout.


def schedule_chunks(node_rank, num_nodes=4):
    per_worker = EXPERTS // WORKERS
    per_owner = EXPERTS // num_nodes
    chunks = []
    for _layer in range(1, LAYERS + 1):
        for offset in range(0, per_worker, SUPERCHUNK):
            for worker in range(WORKERS):
                first = worker * per_worker + offset
                if first // per_owner == node_rank:
                    continue
                chunks.append(worker)
    return chunks


def slot_layout(node_rank, num_nodes=4):
    per_worker = EXPERTS // WORKERS
    per_owner = EXPERTS // num_nodes
    remote = [w for w in range(WORKERS) if (w * per_worker) // per_owner != node_rank]
    return [GPU_NUMA[remote[(slot // SUPERCHUNK) % len(remote)]] for slot in range(DEPTH)]


def test_every_chunk_of_two_passes_lands_in_a_slot_on_its_consumer_node():
    for node_rank in range(4):
        chunks = schedule_chunks(node_rank)
        assert len(chunks) % (DEPTH // SUPERCHUNK) == 0
        layout = slot_layout(node_rank)
        tail = 0
        for index in range(2 * len(chunks)):
            worker = chunks[index % len(chunks)]
            if tail + SUPERCHUNK > DEPTH:
                tail = 0
            assert layout[tail] == GPU_NUMA[worker], (node_rank, index)
            tail += SUPERCHUNK


def test_the_socket_share_of_the_ring_matches_the_remote_workers():
    for node_rank, expected in ((0, (128, 256)), (1, (128, 256)), (2, (256, 128)), (3, (256, 128))):
        layout = slot_layout(node_rank)
        assert (layout.count(0), layout.count(1)) == expected


def test_two_nodes_put_every_consumer_on_one_node():
    for node_rank in range(2):
        assert len(set(slot_layout(node_rank, num_nodes=2))) == 1


def test_a_ring_that_does_not_hold_whole_rounds_breaks_the_layout():
    # Depth 32 holds 4 groups, not a multiple of the 6 remote workers: the
    # position of a consumer would drift every pass, which is why the daemon
    # rejects it.
    chunks = schedule_chunks(0)
    depth = 32
    remote = sorted(set(chunks))
    layout = [GPU_NUMA[remote[(slot // SUPERCHUNK) % len(remote)]] for slot in range(depth)]
    tail, mismatches = 0, 0
    for worker in chunks[:200]:
        mismatches += layout[tail] != GPU_NUMA[worker]
        tail = (tail + SUPERCHUNK) % depth
    assert mismatches > 0
