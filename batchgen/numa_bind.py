"""NUMA-aware CPU + memory binding for worker processes.

Called from the HTTP server worker entrypoint BEFORE `torch.cuda.set_device`
and BEFORE any pinned-host allocation so that:

  - The worker thread runs on CPUs local to its GPU (avoids cross-socket
    cudaLaunchKernel UPI hops — the 10 ms worst-case launch gaps we see
    in nsys on unbound ranks).
  - Pinned-host memory (KV offload staging, TMA descriptors, NCCL transport)
    allocates on the local NUMA node (MPOL_BIND).
  - Each rank gets a disjoint CPU slice inside its node, so the 4 ranks on
    the same socket do not preempt each other's dispatch thread.

Source of truth for the GPU-to-NUMA mapping is `/sys/bus/pci/devices/.../
numa_node` (via NVML-reported PCI bus). The old hardcoded
`{0:0, 1:0, 2:0, 3:0, 4:1, 5:1, 6:1, 7:1}` table in `batchgen_.py` is a
placeholder that never ran on the server path.

Disable with `BATCHGEN_NUMA_BIND=0`.
"""

import ctypes
import ctypes.util
import logging
import os
from typing import List, Optional


def _parse_cpulist(spec: str) -> List[int]:
    cpus: List[int] = []
    for chunk in spec.strip().split(","):
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-")
            cpus.extend(range(int(a), int(b) + 1))
        else:
            cpus.append(int(chunk))
    return cpus


def _gpu_bus_id(local_rank: int) -> Optional[str]:
    try:
        import pynvml  # lazy — avoid forcing pynvml on non-binding paths
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
        pci = pynvml.nvmlDeviceGetPciInfo(h)
        raw = pci.busId.decode() if isinstance(pci.busId, bytes) else pci.busId
    except Exception as e:
        logging.warning("[numa] nvml probe failed for rank %d: %s", local_rank, e)
        return None
    # NVML returns "00000000:1C:00.0" (uppercase, 8-hex domain).
    # sysfs uses "0000:1c:00.0".
    parts = raw.lower().split(":")
    if len(parts) == 3:
        dom, bus, devfn = parts
    elif len(parts) == 4:
        # Some drivers include an extra leading zero-domain segment.
        dom = parts[1]
        bus = parts[2]
        devfn = parts[3]
    else:
        return None
    dom = dom.zfill(8)[-4:]
    return f"{dom}:{bus}:{devfn}"


def _gpu_numa_node(local_rank: int) -> Optional[int]:
    bus = _gpu_bus_id(local_rank)
    if bus is None:
        return None
    path = f"/sys/bus/pci/devices/{bus}/numa_node"
    try:
        with open(path) as f:
            n = int(f.read().strip())
    except OSError:
        return None
    return n if n >= 0 else None


def _node_cpus(node: int) -> List[int]:
    path = f"/sys/devices/system/node/node{node}/cpulist"
    with open(path) as f:
        return _parse_cpulist(f.read())


def _set_mempolicy_bind(node: int) -> bool:
    """MPOL_BIND to the given NUMA node. Affects allocations made after
    this call in the current thread/process. Returns True on success."""
    # Preferred: libnuma.
    libnuma_name = ctypes.util.find_library("numa")
    if libnuma_name:
        try:
            nm = ctypes.CDLL(libnuma_name)
            if nm.numa_available() >= 0:
                nm.numa_set_bind_policy(ctypes.c_int(1))  # strict
                # numa_bitmask: allocate, set bit, pass to numa_set_membind
                nb = nm.numa_allocate_nodemask()
                nm.numa_bitmask_clearall(nb)
                nm.numa_bitmask_setbit(nb, ctypes.c_uint(node))
                nm.numa_set_membind(nb)
                nm.numa_free_nodemask(nb)
                return True
        except Exception as e:
            logging.debug("[numa] libnuma path failed: %s", e)
    # Fallback: direct syscall.
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        MPOL_BIND = 2
        SYS_set_mempolicy = 238  # x86_64
        maxnode_bits = 128
        nwords = (maxnode_bits + 63) // 64
        mask = (ctypes.c_ulong * nwords)()
        mask[node // 64] = 1 << (node % 64)
        rc = libc.syscall(
            ctypes.c_long(SYS_set_mempolicy),
            ctypes.c_int(MPOL_BIND),
            mask,
            ctypes.c_ulong(maxnode_bits),
        )
        return rc == 0
    except Exception as e:
        logging.debug("[numa] set_mempolicy fallback failed: %s", e)
        return False


def _nvml_device_count() -> Optional[int]:
    try:
        import pynvml
        pynvml.nvmlInit()
        return int(pynvml.nvmlDeviceGetCount())
    except Exception:
        return None


def bind_worker_to_gpu_numa(local_rank: int,
                             local_world_size: Optional[int] = None) -> None:
    """Pin the calling worker process to its GPU's local NUMA node.

    Carves a disjoint CPU slice per rank within the node (prevents the
    four ranks on the same socket from preempting each other). Also binds
    memory allocations to the local node via set_mempolicy.

    Safe to call before `torch.cuda.set_device`; no CUDA runtime calls.
    """
    if os.environ.get("BATCHGEN_NUMA_BIND", "1").lower() in ("0", "false", "no"):
        logging.info("[numa] binding disabled via BATCHGEN_NUMA_BIND=0")
        return

    node = _gpu_numa_node(local_rank)
    if node is None:
        logging.warning("[numa] could not detect NUMA node for rank %d; "
                         "leaving CPU affinity unchanged", local_rank)
        return

    # Build {rank -> node} so we can compute this rank's slot within its node.
    if local_world_size is None:
        local_world_size = _nvml_device_count() or (local_rank + 1)
    rank_to_node = {}
    for r in range(local_world_size):
        n = _gpu_numa_node(r)
        if n is not None:
            rank_to_node[r] = n
    peers_on_node = sorted(r for r, n in rank_to_node.items() if n == node)
    if local_rank in peers_on_node:
        slot = peers_on_node.index(local_rank)
        n_on_node = len(peers_on_node)
    else:
        slot, n_on_node = 0, 1

    all_cpus = _node_cpus(node)
    # Split the local CPU list into n_on_node contiguous, disjoint slices.
    # This keeps physical cores + their SMT siblings together because Linux
    # emits them in a `phys-cores..., smt-siblings...` layout in `cpulist`.
    chunk = len(all_cpus) // max(n_on_node, 1)
    if chunk == 0:
        my_cpus = all_cpus
    else:
        start = slot * chunk
        end = start + chunk if slot < n_on_node - 1 else len(all_cpus)
        my_cpus = all_cpus[start:end]

    try:
        os.sched_setaffinity(0, set(my_cpus))
    except OSError as e:
        logging.error("[numa] sched_setaffinity rank=%d node=%d failed: %s",
                       local_rank, node, e)
        return

    ok_mem = _set_mempolicy_bind(node)
    logging.info(
        "[numa] rank %d → node %d, cpus[%d:%d]=%d..%d (n=%d), membind=%s",
        local_rank, node, slot * chunk,
        (slot * chunk) + len(my_cpus), my_cpus[0], my_cpus[-1],
        len(my_cpus), "ok" if ok_mem else "failed",
    )
