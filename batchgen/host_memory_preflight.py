"""Fail-closed host memory plan for the production server.

A launch commits its host memory in two large, non-swappable pieces before it
serves anything: the parameter server's weight region and the host KV pool,
both anonymous shmem.  Nothing checked whether the two fit the machine, so a
budget that overcommits MemTotal was only discovered by the kernel OOM killer
taking one worker in the middle of warmup -- after which the cohort died in a
NCCL/TCPStore cascade that names nothing about host memory (2026-11-03,
8-GPU 2 TiB node, host KV 796 GiB).

This module runs at the same stage as :mod:`batchgen.runtime_preflight`,
before any memfd, SHM region or weight byte exists, and keeps the same
contract: it prints every number it used, it refuses a plan that cannot fit,
and it has no bypass.  The refusal names the largest
``--host-kv-cache-size`` the machine can hold, which is the one thing the
operator has to change.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

GIB = 1024 ** 3

# Transient host memory one worker holds on top of the shared regions, at
# steady state: the Python/Torch runtime, CUDA host-side allocations, and the
# per-rank staging buffers.  MEASURED ~19 GiB per rank on the 2026-11-03
# 8-GPU run.  Scaled by the node's local rank count, never by world size: a
# peer node's workers spend a different machine's memory.
_PER_RANK_TRANSIENT_BYTES = 19 * GIB

# Host memory that is neither a shared region nor per-rank anon, and that the
# kernel cannot hand over fast enough while the shmem regions fault in: the
# 700+ GiB checkpoint is streamed through the page cache as the weight region
# is filled, plus the prefix-cache index region, the unidentified shm segments
# the incident measured, and kernel structures for ~1.5 TiB of shmem.
#
# Basis (2026-11-03 incident, bug_log.md; 8 local ranks, MemTotal 2113 GB =
# 1967 GiB).  Resident weights + host KV at the kill were 1501 GiB, exactly
# the OOM victim's shmem-rss (`Out of memory: Killed process ...
# shmem-rss:1573723116kB oom_score_adj:999`), leaving 466 GiB outside Shmem --
# not enough, so the kernel took one worker during warmup.  The same launch at
# host KV 700 GiB served, with 562 GiB outside Shmem and MemFree bottoming at
# ~148 GiB.  The peak transient therefore sits in (466, 562] GiB; this slack
# puts the planned total at 8 x 19 + 384 = 536 GiB, inside that window, so the
# configuration that served stays admissible and the one that died is refused.
# It is calibrated to that window, not measured on its own, and the meminfo
# sampler the bug log mentions is what would sharpen it.
#
# The slack is capped by the bytes that stream through the page cache into a
# separate resident region: the page cache this term mostly stands for is
# that copy (the checkpoint filling the parameter-server region, or a store
# source filling its tmpfs staging copy), which can never exceed the bytes
# copied, so a small model on a small-memory node is not refused by a slack
# calibrated on a 700-GiB-class checkpoint, and a store mapped in place, which
# copies nothing, carries none.
_PAGE_CACHE_SLACK_CAP_BYTES = 384 * GIB


def _page_cache_slack_bytes(streamed_bytes: int) -> int:
    return min(_PAGE_CACHE_SLACK_CAP_BYTES, max(0, streamed_bytes))

_PLAN_TAG = "[MEM-PLAN host]"
_GPU_PLAN_TAG = "[MEM-PLAN gpu]"


class HostMemoryPlanError(RuntimeError):
    """Raised when the host memory this launch commits cannot fit MemTotal."""


def _gib(byte_count: int) -> int:
    """Whole GiB, rounded down (negative headroom rounds away from zero).

    Every figure the plan prints is derived from exact byte counts; only the
    rendering is rounded, and it is rounded so that a plan can never look
    larger than it is.
    """

    return byte_count // GIB


@dataclass(frozen=True)
class HostMemoryPlan:
    """The host memory one launch commits, against one machine's MemTotal."""

    mem_total_bytes: int
    weights_bytes: int
    host_kv_bytes: int
    transient_bytes: int
    local_world_size: int
    weights_source: str

    @property
    def required_bytes(self) -> int:
        return self.weights_bytes + self.host_kv_bytes + self.transient_bytes

    @property
    def headroom_bytes(self) -> int:
        return self.mem_total_bytes - self.required_bytes

    @property
    def fits(self) -> bool:
        return self.headroom_bytes >= 0

    @property
    def verdict(self) -> str:
        return "OK" if self.fits else "REFUSE"

    @property
    def max_host_kv_gib(self) -> int:
        """Largest ``--host-kv-cache-size`` (whole GiB) this machine holds.

        Zero means the weight region plus the transient term already exceed
        MemTotal, so no host KV pool fits at all.
        """

        return max(0, _gib(self.mem_total_bytes - self.weights_bytes
                           - self.transient_bytes))

    def render(self) -> str:
        """The one-line plan block, carrying every number the verdict used."""

        return (
            f"{_PLAN_TAG} MemTotal={_gib(self.mem_total_bytes)}GiB "
            f"weights={_gib(self.weights_bytes)}GiB "
            f"host_kv={_gib(self.host_kv_bytes)}GiB "
            f"transient>={_gib(self.transient_bytes)}GiB "
            f"headroom={_gib(self.headroom_bytes)}GiB "
            f"verdict={self.verdict} "
            f"weights_src={self.weights_source} "
            f"ranks={self.local_world_size} "
            f"max_host_kv={self.max_host_kv_gib}GiB"
        )

    def refusal(self) -> str:
        """The actionable refusal: what overcommits, and the flag value that fits."""

        slack_bytes = (
            self.transient_bytes
            - _PER_RANK_TRANSIENT_BYTES * self.local_world_size
        )
        transient_terms = (
            f"{self.local_world_size} ranks x "
            f"{_gib(_PER_RANK_TRANSIENT_BYTES)}GiB + "
            f"{_gib(slack_bytes)}GiB page-cache slack"
        )
        max_host_kv = self.max_host_kv_gib
        if max_host_kv <= 0:
            return (
                f"host memory plan does not fit: weights "
                f"{_gib(self.weights_bytes)}GiB + transient "
                f"{_gib(self.transient_bytes)}GiB ({transient_terms}) already "
                f"needs more than MemTotal {_gib(self.mem_total_bytes)}GiB, so "
                f"no --host-kv-cache-size fits this machine. Run this model "
                f"on a node with more host memory. See the {_PLAN_TAG} line "
                f"above for every figure used (weights from "
                f"{self.weights_source})."
            )
        return (
            f"host memory plan does not fit: weights "
            f"{_gib(self.weights_bytes)}GiB + host KV "
            f"{_gib(self.host_kv_bytes)}GiB + transient "
            f"{_gib(self.transient_bytes)}GiB ({transient_terms}) exceeds "
            f"MemTotal {_gib(self.mem_total_bytes)}GiB by "
            f"{_gib(-self.headroom_bytes)}GiB. Relaunch with "
            f"--host-kv-cache-size {max_host_kv} or less (that is the maximum "
            f"this machine holds for these weights), or use a node with more "
            f"host memory. See the {_PLAN_TAG} line above for every figure used."
        )


def build_plan(
    *,
    mem_total_bytes: int,
    weights_bytes: int,
    host_kv_bytes: int,
    local_world_size: int,
    weights_source: str,
    streamed_bytes: Optional[int] = None,
) -> HostMemoryPlan:
    """Assemble the plan. Pure arithmetic: every input is already resolved.

    ``host_kv_bytes`` is the whole host KV budget.  A DSA model's auxiliary
    pool is carved out of that same budget by ``DualHostKVCoordinator``, and
    so is the prefix-cache index, so neither is a separate term here.
    ``weights_bytes`` is the host memory the weights hold that the kernel
    cannot reclaim; ``streamed_bytes`` the bytes copied through the page cache
    into that region at load (None: the weights themselves, as for a region
    filled from a checkpoint).
    """

    if mem_total_bytes <= 0:
        raise HostMemoryPlanError(
            f"MemTotal reported as {mem_total_bytes} bytes; refusing to plan "
            "host memory against an unreadable machine"
        )
    if local_world_size <= 0:
        raise HostMemoryPlanError(
            f"local world size reported as {local_world_size}; refusing to "
            "plan host memory for an empty cohort"
        )
    if streamed_bytes is None:
        streamed_bytes = weights_bytes
    transient_bytes = (
        _PER_RANK_TRANSIENT_BYTES * local_world_size
        + _page_cache_slack_bytes(streamed_bytes)
    )
    return HostMemoryPlan(
        mem_total_bytes=mem_total_bytes,
        weights_bytes=weights_bytes,
        host_kv_bytes=host_kv_bytes,
        transient_bytes=transient_bytes,
        local_world_size=local_world_size,
        weights_source=weights_source,
    )


def read_mem_total_bytes() -> int:
    """MemTotal of this machine, through the shared ``/proc/meminfo`` reader.

    ``host_mem.get_physical_memory_info`` reports a failure by returning its
    message instead of a mapping; a gate cannot continue on that, so the
    non-mapping case is a refusal rather than a default.
    """

    from batchgen.scheduler.host_mem import get_physical_memory_info

    info = get_physical_memory_info()
    if not isinstance(info, dict) or not info.get("total_physical"):
        raise HostMemoryPlanError(
            f"cannot read MemTotal from /proc/meminfo ({info}); the host "
            "memory plan has no machine to plan against"
        )
    # The reader divides the kB /proc/meminfo reports by 1024 twice, so its
    # "GB" are GiB and this scaling is the right one. A double holds a 2 TiB
    # MemTotal to far better than the kB the kernel itself reports, and the
    # plan's smallest term is a whole GiB, so the round-trip through the
    # reader's float is exact at the plan's resolution.
    return int(round(info["total_physical"] * GIB))


def _checkpoint_index_bytes(cache_dir: Optional[Path]) -> Optional[int]:
    """``metadata.total_size`` from the checkpoint's safetensors index, or None.

    One JSON read through the reader the model side already uses. No weight
    file is opened and no tensor is stat'ed.
    """

    if cache_dir is None:
        return None
    index_path = Path(cache_dir) / "model.safetensors.index.json"
    if not index_path.is_file():
        return None
    try:
        with open(index_path, "r", encoding="utf-8") as handle:
            return int(json.load(handle)["metadata"]["total_size"])
    except Exception as exc:  # noqa: BLE001 - a malformed index is not fatal here
        # The plan falls back to the per-model envelope, which is an upper
        # bound; say so rather than dropping the exact figure silently.
        logger.warning(
            "%s checkpoint index %s is unreadable (%s); planning from the "
            "host weight envelope instead",
            _PLAN_TAG,
            index_path,
            exc,
        )
        return None


@dataclass(frozen=True)
class HostWeights:
    """What the weights of one launch hold in this node's host memory.

    ``resident_bytes``: host memory the kernel cannot reclaim (anonymous or
    tmpfs shmem, or pinned file pages). ``streamed_bytes``: bytes copied
    through the page cache into a separate resident region at load.
    """

    resident_bytes: int
    streamed_bytes: int
    source: str


# Distributed-store transports that pin only the node's routed experts (each
# worker its own EP range, and the hierarchical source its slot, both inside
# the node's expert region); host_rdma pins the whole store.
_EXPERT_REGION_PIN_TRANSPORTS = ("hierarchical_gdr", "local")


def _distributed_store_weights(config_path) -> HostWeights:
    """Host memory of a distributed node store, by residency and transport.

    Staged (``store_residency`` set): the server copies ``store.bin`` into a
    tmpfs, so the whole store is shmem and the copy streams the source through
    the page cache. Unstaged (no residency): the store file is mapped in
    place; nothing is copied, and the non-reclaimable part is what the workers
    pin -- the whole store under host_rdma, the expert region
    (``store_bytes - replicated_bytes``) under hierarchical_gdr and local. The
    unpinned replicated prefix stays reclaimable page cache.
    """

    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    store_bytes = int(config["store_bytes"])
    if config.get("store_residency") is not None:
        return HostWeights(store_bytes, store_bytes, "distributed-store-staged")
    transport = config.get("transport", "host_rdma")
    if transport in _EXPERT_REGION_PIN_TRANSPORTS:
        pinned = store_bytes - int(config["replicated_bytes"])
    else:
        pinned = store_bytes
    return HostWeights(pinned, 0, f"distributed-store-mapped-{transport}")


def resolve_host_weights(server_args: object) -> HostWeights:
    """Host weight footprint this launch commits, and where the number is from.

    Never stats the weight files; every source is a number the loader already
    has, in the order of how exactly it describes what will be resident:

    * ``--distributed-weight-config``: from the node config's ``store_bytes``
      (this node's share of the compact store: the replicated tensors plus
      the experts this node owns), its residency and its transport, already
      validated by ``prepare_server_args`` (``_distributed_store_weights``).
    * the checkpoint's safetensors index: ``metadata.total_size``, the exact
      byte total the parameter server fills its region with.  This is the
      resident figure -- the region is an anonymous memfd created at the
      per-model envelope size but charged per faulted page, which is why the
      2026-11-03 OOM victim's shmem-rss matched checkpoint + host KV and not
      envelope + host KV.
    * otherwise the per-model host weight envelope from
      ``process_utils.MODEL_BYTE_SIZES``, the same number
      ``_config_hugepages`` reserves against.  It is an upper bound (it
      carries the table's buffer), which is the safe direction to plan in
      when the exact total is not on this machine.

    The last two fill their region from the checkpoint, so every byte also
    streams through the page cache.
    """

    config_path = getattr(server_args, "distributed_weight_config", None)
    if config_path is not None:
        # The node config is read directly so this launch-layer module does
        # not import the model package.
        return _distributed_store_weights(config_path)

    model = str(getattr(server_args, "model", ""))
    index_bytes = _checkpoint_index_bytes(
        getattr(server_args, "cache_dir", None)
        or (Path(model) if Path(model).is_dir() else None)
    )
    if index_bytes is not None:
        return HostWeights(index_bytes, index_bytes, "checkpoint-index")

    from batchgen.server.process_utils import get_model_byte_size

    envelope = int(get_model_byte_size(model))
    return HostWeights(envelope, envelope, "host-weight-envelope")


def resolve_weights_bytes(server_args: object) -> Tuple[int, str]:
    """The resident weight bytes and their source (``resolve_host_weights``)."""

    weights = resolve_host_weights(server_args)
    return weights.resident_bytes, weights.source


def _log_gpu_plan(server_args: object) -> None:
    """Print the GPU side of the budget. Print-only: this slice refuses nothing.

    The GPU arena is sized as ``total * gpu_memory_frac - model_shard``, and
    the model shard is not known until the weights are loaded, so only the
    device total and the fraction are printable here.
    """

    frac = getattr(server_args, "gpu_memory_frac", None)
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA device is visible to this process")
        device_count = torch.cuda.device_count()
        total = int(
            torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        )
    except Exception as exc:  # noqa: BLE001 - a print-only block never refuses
        logger.info(
            "%s unavailable (%s); the host plan above is unaffected",
            _GPU_PLAN_TAG,
            exc,
        )
        return
    budget = int(total * frac) if isinstance(frac, (int, float)) else None
    logger.info(
        "%s devices=%d per_dev_total=%dGiB frac=%s per_dev_arena_budget=%s",
        _GPU_PLAN_TAG,
        device_count,
        _gib(total),
        frac,
        f"{_gib(budget)}GiB" if budget is not None else "unknown",
    )


def run_host_memory_preflight(server_args: object) -> HostMemoryPlan:
    """Print this launch's host memory plan, and refuse one that cannot fit.

    Called from the server entry point before any worker, SHM region or
    weight byte exists, so a refusal costs nothing and leaves nothing behind.
    """

    world_size = int(getattr(server_args, "world_size", 1) or 1)
    nnodes = int(getattr(server_args, "nnodes", 1) or 1)
    if nnodes <= 0 or world_size % nnodes != 0:
        raise HostMemoryPlanError(
            f"world_size {world_size} is not divisible by nnodes {nnodes}; "
            "the per-node rank count the plan needs is undefined"
        )
    host_kv_gib = getattr(server_args, "host_kv_cache_size", None)
    if not host_kv_gib:
        raise HostMemoryPlanError(
            "--host-kv-cache-size is unset at the host memory plan; it is "
            "required before any host KV pool is created"
        )
    weights = resolve_host_weights(server_args)
    plan = build_plan(
        mem_total_bytes=read_mem_total_bytes(),
        weights_bytes=weights.resident_bytes,
        host_kv_bytes=int(host_kv_gib) * GIB,
        local_world_size=world_size // nnodes,
        weights_source=weights.source,
        streamed_bytes=weights.streamed_bytes,
    )
    # One block, from the server process only: every worker would print the
    # same machine's plan, and this runs before the workers exist.
    logger.info("%s", plan.render())
    _log_gpu_plan(server_args)
    if not plan.fits:
        raise HostMemoryPlanError(plan.refusal())
    return plan


__all__ = [
    "GIB",
    "HostMemoryPlan",
    "HostMemoryPlanError",
    "HostWeights",
    "build_plan",
    "read_mem_total_bytes",
    "resolve_host_weights",
    "resolve_weights_bytes",
    "run_host_memory_preflight",
]
