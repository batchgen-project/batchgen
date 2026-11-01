"""CPU-only regressions for the weight daemon's owner rail map and RMA lanes.

Under the pooled map (the default) every node connects to every owner on all
eight rails, so the least-loaded rail selection can put any owner's bytes on
any rail. The partitioned map keeps the older fixed walk, in which an owner
reaches a node over only two or three rails and each rail of a node carries a
single owner.

The bootstrap must stay balanced under either map: an owner's rail listener
accepts exactly as many connections as there are peers that pull over that
rail, or bootstrap blocks in accept until it times out. The model at the end
of this file checks that count against the connect rule for both maps.

`rma_lanes` sets UCX MAX_RMA_RAILS on every rail context, so each endpoint
splits a get over that many QPs, one per path of the bonded device.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DAEMON = ROOT / "core/Weights_Storage/distributed_weight_daemon.cpp"


def _section(source, begin, end):
    start = source.index(begin)
    return source[start:source.index(end, start)]


def _bootstrap():
    return _section(DAEMON.read_text(), "void BootstrapNetwork() {", "progress_threads[rail] = std::thread(")


def test_pooled_is_the_default_and_lanes_default_to_two():
    daemon = DAEMON.read_text()
    assert "bool pooled_owner_rails = true;" in daemon
    assert "int rma_lanes = 2;" in daemon


def test_the_owner_rail_map_accepts_only_the_two_named_values():
    daemon = DAEMON.read_text()
    parse = _section(daemon, 'if (config.contains("owner_rail_map")) {', 'if (config.contains("rma_lanes")) {')
    assert "!map.is_string()" in parse
    assert '!= "pooled"' in parse and '!= "partitioned"' in parse
    assert 'pooled_owner_rails = map.get<std::string>() == "pooled";' in parse


def test_rma_lanes_must_be_a_positive_integer():
    daemon = DAEMON.read_text()
    parse = _section(daemon, 'if (config.contains("rma_lanes")) {', 'if (config.contains("progress_thread_cores")) {')
    assert "!lanes.is_number_integer()" in parse
    assert "lanes.get<std::int64_t>() < 1" in parse


def test_every_rail_context_gets_the_configured_lane_count():
    create = _section(DAEMON.read_text(), "UcpRail CreateRail(int rail_index) {", "ucp_init(&params, config")
    assert 'ucp_config_modify(config, "MAX_RMA_RAILS",' in create
    assert "std::to_string(rma_lanes).c_str()" in create


def test_each_owner_rail_accepts_every_puller_of_that_rail():
    boot = _bootstrap()
    assert "const int pullers_per_rail = pooled_owner_rails ? num_nodes - 1 : 1;" in boot
    loop = _section(boot, "for (int peer = 0; peer < pullers_per_rail; ++peer) {", "} catch (const std::exception& error) {")
    # The whole per-connection exchange runs once per accepted peer.
    for step in ("accept(tcp_listeners[rail]", "incoming_endpoints[rail].push_back(endpoint);",
                 "incoming_controls[rail].push_back(fd);", "send_exact(fd, packed_rkey, packed_rkey_bytes);"):
        assert step in loop


def test_the_partition_walk_only_filters_connects_under_the_partitioned_map():
    boot = _bootstrap()
    assert "(!pooled_owner_rails &&" in boot
    assert "destination_for_owner_rail(owner, rail + 1," in boot


def test_the_bootstrap_check_requires_all_rails_under_the_pooled_map():
    boot = _bootstrap()
    check = _section(boot, "const size_t min_rails = pooled_owner_rails", "for (int owner = 0; owner < num_nodes; ++owner) {")
    assert check.count("? static_cast<size_t>(kRails)") == 2


def test_the_summary_states_both_settings():
    daemon = DAEMON.read_text()
    assert 'summary["owner_rail_map"] =' in daemon
    assert 'pooled_owner_rails ? "pooled" : "partitioned";' in daemon
    assert 'summary["rma_lanes"] = rma_lanes;' in daemon


# Model of the connect and accept rules.


def destination_for_owner_rail(owner, rail, num_nodes):
    return (owner + 1 + (rail - 1) % (num_nodes - 1)) % num_nodes


def connects(node, owner, rail, num_nodes, pooled):
    if owner == node:
        return False
    return pooled or destination_for_owner_rail(owner, rail + 1, num_nodes) == node


def pullers_per_rail(num_nodes, pooled):
    return num_nodes - 1 if pooled else 1


def test_connects_and_accepts_balance_for_both_maps():
    for num_nodes in (2, 3, 4, 8):
        for pooled in (True, False):
            for owner in range(num_nodes):
                for rail in range(8):
                    pullers = sum(connects(node, owner, rail, num_nodes, pooled) for node in range(num_nodes))
                    assert pullers == pullers_per_rail(num_nodes, pooled), (num_nodes, pooled, owner, rail)


def test_pooled_reaches_every_owner_over_every_rail():
    for node in range(4):
        for owner in range(4):
            if owner != node:
                assert all(connects(node, owner, rail, 4, True) for rail in range(8))


def test_partitioned_on_four_nodes_is_the_three_three_two_split():
    for node in range(4):
        counts = sorted(sum(connects(node, owner, rail, 4, False) for rail in range(8))
                        for owner in range(4) if owner != node)
        assert counts == [2, 3, 3]
