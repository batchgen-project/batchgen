"""Host memory plan at launch: the printed block, and what it refuses.

Pure CPU. MemTotal is always injected -- nothing here reads the machine the
test runs on, so the 2026-11-03 incident figures (8 local ranks, MemTotal
2113 GB = 1967 GiB, 705 GiB of resident weights) are reproducible anywhere.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import batchgen.host_memory_preflight as preflight  # noqa: E402
from batchgen.host_memory_preflight import (  # noqa: E402
    GIB,
    HostMemoryPlanError,
    build_plan,
    read_mem_total_bytes,
    resolve_weights_bytes,
    run_host_memory_preflight,
)

# The 2026-11-03 8-GPU node (bug_log.md): MemTotal 2113 GB = 1967 GiB, 8 local
# ranks, 705 GiB of resident weights (the OOM victim's shmem-rss of 1501 GiB
# minus the 796 GiB host KV). Host KV 700 GiB served; 796 GiB was OOM-killed.
INCIDENT_MEM_TOTAL = 1967 * GIB
INCIDENT_WEIGHTS = 705 * GIB
INCIDENT_RANKS = 8
# 8 x 19 GiB per-rank anon + 384 GiB page-cache slack.
INCIDENT_TRANSIENT = 536 * GIB


def _incident_plan(host_kv_gib: int, *, mem_total_bytes: int = INCIDENT_MEM_TOTAL,
                   weights_bytes: int = INCIDENT_WEIGHTS,
                   local_world_size: int = INCIDENT_RANKS,
                   weights_source: str = "checkpoint-index"):
    return build_plan(
        mem_total_bytes=mem_total_bytes,
        weights_bytes=weights_bytes,
        host_kv_bytes=host_kv_gib * GIB,
        local_world_size=local_world_size,
        weights_source=weights_source,
    )


def test_transient_term_is_per_rank_anon_plus_slack():
    plan = _incident_plan(700)
    assert plan.transient_bytes == INCIDENT_TRANSIENT
    assert plan.transient_bytes == (
        preflight._PER_RANK_TRANSIENT_BYTES * INCIDENT_RANKS
        + preflight._PAGE_CACHE_SLACK_CAP_BYTES
    )
    # One node's plan never charges a peer node's workers.
    half = _incident_plan(700, local_world_size=4)
    assert half.transient_bytes == INCIDENT_TRANSIENT - 4 * 19 * GIB


def test_page_cache_slack_is_capped_by_the_checkpoint_size():
    # The slack stands mostly for the checkpoint streaming through the page
    # cache, so a small model never inherits the 700-GiB-class calibration:
    # a 60 GiB checkpoint on a 1 TiB node must not be refused by it.
    small = preflight.build_plan(
        mem_total_bytes=1000 * GIB,
        weights_bytes=60 * GIB,
        host_kv_bytes=500 * GIB,
        local_world_size=8,
        weights_source="checkpoint-index",
    )
    assert small.transient_bytes == 8 * 19 * GIB + 60 * GIB
    assert small.fits


def test_plan_arithmetic_is_exact():
    plan = _incident_plan(700)
    assert plan.required_bytes == INCIDENT_WEIGHTS + 700 * GIB + INCIDENT_TRANSIENT
    assert plan.headroom_bytes == INCIDENT_MEM_TOTAL - plan.required_bytes
    assert plan.headroom_bytes == 26 * GIB


def test_incident_configuration_that_served_is_admitted():
    plan = _incident_plan(700)
    assert plan.fits
    assert plan.verdict == "OK"


def test_incident_configuration_that_was_oom_killed_is_refused():
    plan = _incident_plan(796)
    assert not plan.fits
    assert plan.verdict == "REFUSE"
    # Overcommit is 705 + 796 + 536 - 1967 = 70 GiB.
    assert plan.headroom_bytes == -70 * GIB


def test_plan_block_prints_every_figure_it_used():
    assert _incident_plan(700).render() == (
        "[MEM-PLAN host] MemTotal=1967GiB weights=705GiB host_kv=700GiB "
        "transient>=536GiB headroom=26GiB verdict=OK "
        "weights_src=checkpoint-index ranks=8 max_host_kv=726GiB"
    )
    assert _incident_plan(796).render() == (
        "[MEM-PLAN host] MemTotal=1967GiB weights=705GiB host_kv=796GiB "
        "transient>=536GiB headroom=-70GiB verdict=REFUSE "
        "weights_src=checkpoint-index ranks=8 max_host_kv=726GiB"
    )


def test_refusal_names_the_largest_host_kv_that_fits():
    plan = _incident_plan(796)
    assert plan.max_host_kv_gib == 726
    refusal = plan.refusal()
    assert "--host-kv-cache-size 726" in refusal
    # The operator must be able to re-derive the verdict from the refusal.
    for figure in ("weights 705GiB", "host KV 796GiB", "transient 536GiB",
                   "MemTotal 1967GiB", "8 ranks x 19GiB", "384GiB"):
        assert figure in refusal
    # The largest admissible value is admissible, the next GiB is not.
    assert _incident_plan(726).fits
    assert not _incident_plan(727).fits


def test_same_verdicts_on_the_spec_s_rounded_figures():
    # The spec documents the incident as MemTotal 2015 GiB / weights 704 GiB.
    # The transient term must separate 700 from 796 there too, or it is
    # calibrated to one decimal of one machine.
    assert _incident_plan(
        700, mem_total_bytes=2015 * GIB, weights_bytes=704 * GIB
    ).verdict == "OK"
    refused = _incident_plan(
        796, mem_total_bytes=2015 * GIB, weights_bytes=704 * GIB
    )
    assert refused.verdict == "REFUSE"
    assert refused.max_host_kv_gib == 775


def test_machine_too_small_for_any_host_kv():
    plan = _incident_plan(700, mem_total_bytes=1000 * GIB)
    assert plan.max_host_kv_gib == 0
    refusal = plan.refusal()
    assert "no --host-kv-cache-size fits this machine" in refusal
    assert "--host-kv-cache-size 0" not in refusal


def test_unreadable_machine_is_refused_not_defaulted():
    with pytest.raises(HostMemoryPlanError, match="MemTotal"):
        _incident_plan(700, mem_total_bytes=0)
    with pytest.raises(HostMemoryPlanError, match="local world size"):
        _incident_plan(700, local_world_size=0)


def test_mem_total_comes_from_the_shared_meminfo_reader(monkeypatch):
    from batchgen.scheduler import host_mem

    monkeypatch.setattr(
        host_mem,
        "get_physical_memory_info",
        lambda: {"total_physical": 2015.0, "actually_free": 147.0},
    )
    assert read_mem_total_bytes() == 2015 * GIB


def test_meminfo_reader_failure_is_a_refusal(monkeypatch):
    from batchgen.scheduler import host_mem

    # The reader reports failure by returning its message.
    monkeypatch.setattr(
        host_mem,
        "get_physical_memory_info",
        lambda: "Error reading memory info: [Errno 2] No such file",
    )
    with pytest.raises(HostMemoryPlanError, match="cannot read MemTotal"):
        read_mem_total_bytes()


def _install_process_utils(monkeypatch):
    """Make ``batchgen.server.process_utils`` importable without the server.

    ``batchgen/server/__init__.py`` imports the FastAPI app, so the real
    lookup table is loaded by path and registered under its own name; the
    table itself is the real one, so this test cannot drift from it.
    """

    package = types.ModuleType("batchgen.server")
    package.__path__ = [str(ROOT / "batchgen" / "server")]
    monkeypatch.setitem(sys.modules, "batchgen.server", package)
    path = ROOT / "batchgen" / "server" / "process_utils.py"
    spec = importlib.util.spec_from_file_location(
        "batchgen.server.process_utils", path
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "batchgen.server.process_utils", module)
    spec.loader.exec_module(module)
    return module


def test_weights_come_from_the_host_weight_envelope(monkeypatch):
    pytest.importorskip("psutil")
    process_utils = _install_process_utils(monkeypatch)
    args = SimpleNamespace(
        model="zai-org/GLM-5.2-FP8",
        cache_dir=None,
        distributed_weight_config=None,
    )
    weights, source = resolve_weights_bytes(args)
    assert source == "host-weight-envelope"
    assert weights == process_utils.get_model_byte_size("zai-org/GLM-5.2-FP8")


def test_weights_come_from_the_distributed_store_config(monkeypatch, tmp_path):
    store_config = tmp_path / "node0.json"
    store_config.write_text(
        '{"node_rank": 0, "node_ips": ["a", "b"], "workers": 8, '
        '"store_path": "/dev/shm/store.bin", "metadata_path": "m.tsv", '
        '"daemon_socket": "s", "summary_path": "sum.json", '
        '"store_bytes": 780000000000, "replicated_bytes": 1, '
        '"module_bytes": 1, "worker_sharded": true}'
    )
    args = SimpleNamespace(
        model="moonshotai/Kimi-K3",
        cache_dir=None,
        distributed_weight_config=store_config,
    )
    # No residency, default transport: mapped in place, host_rdma pins all.
    weights, source = resolve_weights_bytes(args)
    assert source == "distributed-store-mapped-host_rdma"
    assert weights == 780000000000


def test_absent_checkpoint_index_is_not_an_error(tmp_path):
    assert preflight._checkpoint_index_bytes(tmp_path) is None
    assert preflight._checkpoint_index_bytes(None) is None


def test_checkpoint_index_is_preferred_over_the_envelope(monkeypatch):
    """The resident figure wins even though it is the smaller of the two.

    The weight region is a memfd created at the envelope size but charged per
    faulted page: on 2026-11-03 the OOM victim's shmem-rss matched checkpoint
    total + host KV, not envelope + host KV. Planning on the envelope would
    refuse a 705 GiB model sized by a 760 GiB table entry.
    """

    monkeypatch.setattr(
        preflight, "_checkpoint_index_bytes", lambda cache_dir: 705 * GIB
    )
    args = SimpleNamespace(
        model="zai-org/GLM-5.3-FP8",
        cache_dir="/some/checkpoint",
        distributed_weight_config=None,
    )
    assert resolve_weights_bytes(args) == (705 * GIB, "checkpoint-index")


def test_checkpoint_index_is_read_through_the_shared_reader(tmp_path):
    pytest.importorskip("torch")
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"metadata": {"total_size": %d}, '
        '"weight_map": {"a": "model-00001.safetensors"}}' % (705 * GIB)
    )
    assert preflight._checkpoint_index_bytes(tmp_path) == 705 * GIB


def _preflight_args(host_kv_gib: int, *, world_size=8, nnodes=1):
    return SimpleNamespace(
        model="zai-org/GLM-5.2-FP8",
        cache_dir=None,
        distributed_weight_config=None,
        host_kv_cache_size=host_kv_gib,
        world_size=world_size,
        nnodes=nnodes,
        gpu_memory_frac=0.9,
    )


def _stub_sources(monkeypatch):
    monkeypatch.setattr(
        preflight, "read_mem_total_bytes", lambda: INCIDENT_MEM_TOTAL
    )
    monkeypatch.setattr(
        preflight,
        "resolve_host_weights",
        lambda args: preflight.HostWeights(
            INCIDENT_WEIGHTS, INCIDENT_WEIGHTS, "host-weight-envelope"
        ),
    )
    # The GPU block is print-only and never decides anything here.
    monkeypatch.setattr(preflight, "_log_gpu_plan", lambda args: None)


def test_preflight_prints_one_block_and_admits_the_serving_budget(
    monkeypatch, caplog
):
    _stub_sources(monkeypatch)
    with caplog.at_level(logging.INFO, logger=preflight.__name__):
        plan = run_host_memory_preflight(_preflight_args(700))
    blocks = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("[MEM-PLAN host]")
    ]
    assert blocks == [plan.render()]
    assert "verdict=OK" in blocks[0]


def test_preflight_refuses_the_budget_that_was_oom_killed(monkeypatch, caplog):
    _stub_sources(monkeypatch)
    with caplog.at_level(logging.INFO, logger=preflight.__name__):
        with pytest.raises(HostMemoryPlanError) as refusal:
            run_host_memory_preflight(_preflight_args(796))
    # The block is printed before the refusal: the operator sees the figures
    # the verdict used even though the launch ends here.
    assert any(
        "verdict=REFUSE" in record.getMessage() for record in caplog.records
    )
    assert "--host-kv-cache-size 726" in str(refusal.value)


def test_preflight_charges_only_this_node_s_ranks(monkeypatch):
    _stub_sources(monkeypatch)
    # 2 nodes x 8 ranks: the plan is still an 8-rank plan, because the peer
    # node's workers spend a different machine's memory.
    plan = run_host_memory_preflight(_preflight_args(700, world_size=16, nnodes=2))
    assert plan.local_world_size == 8
    assert plan.transient_bytes == INCIDENT_TRANSIENT


def test_preflight_requires_a_host_kv_budget(monkeypatch):
    _stub_sources(monkeypatch)
    args = _preflight_args(700)
    args.host_kv_cache_size = None
    with pytest.raises(HostMemoryPlanError, match="--host-kv-cache-size"):
        run_host_memory_preflight(args)


def test_preflight_refuses_an_undefined_per_node_rank_count(monkeypatch):
    _stub_sources(monkeypatch)
    with pytest.raises(HostMemoryPlanError, match="divisible"):
        run_host_memory_preflight(
            _preflight_args(700, world_size=8, nnodes=3)
        )


# GB300 two-tray K3.1 node store (2026-11-10, node0.json): replicated tensors
# plus the 448 experts this node owns, 92 MoE layers x 448 x 17,547,264 B.
GB300_STORE_BYTES = 837_989_344_256
GB300_REPLICATED_BYTES = 114_761_311_232
GB300_MEM_TOTAL = 1_002_291_648 * 1024


def _store_config(tmp_path, **fields):
    config = {
        "node_rank": 0, "node_ips": ["a", "b"], "workers": 4,
        "store_path": "/data/store.bin", "metadata_path": "m.tsv",
        "daemon_socket": "s", "summary_path": "sum.json",
        "store_bytes": GB300_STORE_BYTES,
        "replicated_bytes": GB300_REPLICATED_BYTES,
        "module_bytes": 17_547_264, "worker_sharded": True,
    }
    config.update(fields)
    path = tmp_path / "node0.json"
    path.write_text(__import__("json").dumps(config))
    return SimpleNamespace(
        model="moonshotai/Kimi-K3.1-Pro", cache_dir=None,
        distributed_weight_config=path,
    )


def test_an_unstaged_local_store_commits_only_its_pinned_expert_region(tmp_path):
    weights = preflight.resolve_host_weights(
        _store_config(tmp_path, transport="local", store_residency=None)
    )
    expert_region = GB300_STORE_BYTES - GB300_REPLICATED_BYTES
    assert expert_region == 92 * 448 * 17_547_264
    assert weights.resident_bytes == expert_region
    # mapped in place: nothing is copied through the page cache
    assert weights.streamed_bytes == 0
    plan = build_plan(
        mem_total_bytes=GB300_MEM_TOTAL,
        weights_bytes=weights.resident_bytes,
        host_kv_bytes=64 * GIB,
        local_world_size=4,
        weights_source=weights.source,
        streamed_bytes=weights.streamed_bytes,
    )
    assert plan.transient_bytes == 4 * 19 * GIB
    assert plan.fits


def test_a_staged_store_is_shmem_and_streams_through_the_page_cache(tmp_path):
    weights = preflight.resolve_host_weights(
        _store_config(
            tmp_path, transport="local", store_residency="run",
            store_staging_dir="/dev/shm/batchgen-weight-stores",
        )
    )
    assert weights.resident_bytes == GB300_STORE_BYTES
    assert weights.streamed_bytes == GB300_STORE_BYTES
    plan = build_plan(
        mem_total_bytes=GB300_MEM_TOTAL,
        weights_bytes=weights.resident_bytes,
        host_kv_bytes=64 * GIB,
        local_world_size=4,
        weights_source=weights.source,
        streamed_bytes=weights.streamed_bytes,
    )
    # The same store copied into a tmpfs does not fit this node.
    assert plan.transient_bytes == 4 * 19 * GIB + preflight._PAGE_CACHE_SLACK_CAP_BYTES
    assert not plan.fits


@pytest.mark.parametrize("transport, expect_whole", [
    ("host_rdma", True), ("hierarchical_gdr", False), ("local", False),
])
def test_an_unstaged_store_commits_what_its_transport_pins(
    tmp_path, transport, expect_whole,
):
    weights = preflight.resolve_host_weights(
        _store_config(tmp_path, transport=transport)
    )
    assert weights.resident_bytes == (
        GB300_STORE_BYTES if expect_whole
        else GB300_STORE_BYTES - GB300_REPLICATED_BYTES
    )
    assert weights.streamed_bytes == 0


def test_a_checkpoint_region_still_streams_every_byte(monkeypatch):
    monkeypatch.setattr(
        preflight, "_checkpoint_index_bytes", lambda cache_dir: 705 * GIB
    )
    weights = preflight.resolve_host_weights(SimpleNamespace(
        model="zai-org/GLM-5.3-FP8", cache_dir="/ckpt",
        distributed_weight_config=None,
    ))
    assert (weights.resident_bytes, weights.streamed_bytes) == (705 * GIB, 705 * GIB)
    # The incident's two configurations keep their verdicts through the
    # resolved path: 700 GiB host KV served, 796 GiB was OOM-killed.
    for host_kv, fits in ((700, True), (796, False)):
        plan = build_plan(
            mem_total_bytes=INCIDENT_MEM_TOTAL,
            weights_bytes=weights.resident_bytes,
            host_kv_bytes=host_kv * GIB,
            local_world_size=INCIDENT_RANKS,
            weights_source=weights.source,
            streamed_bytes=weights.streamed_bytes,
        )
        assert plan.fits is fits
