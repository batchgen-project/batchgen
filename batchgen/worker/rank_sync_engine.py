"""Rank synchronization behind a worker seam (S8, RankSync).

``RankSyncEngine`` owns the cross-rank coordination a worker rank performs:
the cohort-shutdown probe and the bounded LOCAL quiesce (with its self-exit),
the gloo / PyNccl / NVSHMEM communicator handles, the ``SyncCoordinator``
facade, and the per-step MoE token-count agreement.

The engine is intentionally stateless, the S5 ``PrefillEngine`` shape: each
operation receives the live worker as its collaborator and rebinds ``self``
to it, so the moved statements keep their original state access and call
spellings. The worker keeps a same-name delegator for every moved method.
"""

from __future__ import annotations

import gc
import logging
import os
import socket
import time
from typing import List, Set, Tuple

import torch
import torch.distributed as dist

from batchgen.distributed.device_communicators.pynccl import PyNcclCommunicator
from batchgen.distributed.utils import StatelessProcessGroup
from batchgen.worker.decode_engine import BATCHGEN_MULTI_BATCH_DIAG
from batchgen.worker.sync import (
    SyncContext,
    SyncCoordinator,
    TorchDistCollectiveBackend,
)


BATCHGEN_ENABLE_ALL_TO_ALL = os.environ.get("BATCHGEN_ENABLE_ALL_TO_ALL")
if BATCHGEN_ENABLE_ALL_TO_ALL == "1":
    try:
        from pplx_kernels import nvshmem_init
    except ImportError as exc:
        logging.warning("Failed to import pplx_kernels.nvshmem_init: %s", exc)
        nvshmem_init = None
else:
    nvshmem_init = None


# Total budget for the LOCAL quiesce on a decode teardown exit.  An abandoned
# step leaves its host-KV appends and its async KV load parked behind device
# work the dead collective will never let finish, so the quiesce bounds them
# itself (see _quiesce_decode_for_cohort_shutdown) and self-exits rather than
# unwind while they still hold page-table pointers.
#
# The bound is established by NON-BLOCKING probes only -- never by running a
# blocking wait under a timeout, on this thread or any other.  MEASURED: the
# append task's wait is bound as `.def("wait", &kv::KVAsyncTask::wait)`
# (core/batchgen_Binding.cpp:968) with no `py::call_guard<gil_scoped_release>`,
# so a task parked behind the dead collective blocks in
# `std::shared_future::get` HOLDING THE GIL.  No other Python can then run --
# not the deadline, not the abort watchdog, not the supervisor's self-exit net
# -- so the rank burns the whole grace window.  `KVAsyncTask::done()`
# (core/KV_Storage/host_paged_kv_worker_view.h:93-99) is a `wait_for(0s)` status
# test, and a CUDA event query is the same class of probe for device work; both
# are safe to call under the GIL because neither can wait.
_COHORT_QUIESCE_DEADLINE_S = 5.0
_COHORT_QUIESCE_POLL_INTERVAL_S = 1e-3
_COHORT_QUIESCE_DEADLINE_REASON = "quiesce-deadline"


def _is_port_available(host: str, port: int) -> bool:
    """Check if a port is available for binding on the given host."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except (OSError, socket.error):
        return False


def _find_available_port(host: str, start_port: int, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port.

    Args:
        host: Host address to bind to
        start_port: Starting port number to check
        max_attempts: Maximum number of ports to try

    Returns:
        An available port number

    Raises:
        RuntimeError: If no available port is found within max_attempts
    """
    for offset in range(max_attempts):
        port = start_port + offset
        if _is_port_available(host, port):
            return port
    raise RuntimeError(f"No available port found in range [{start_port}, {start_port + max_attempts})")


class RankSyncEngine:
    """Stateless rank-sync operations; the worker owns all mutable state."""

    # ---- Cohort shutdown and the bounded local quiesce ---------------------

    def set_cohort_shutdown_probe(self, worker, probe) -> None:
        """Install the zero-argument cohort-teardown probe.

        ``probe()`` answers one question -- has this cohort's teardown begun --
        and the supervisor owns what backs it (a plain load from a lock-free
        shared byte the manager stores to).  Keeping it a callable is what keeps
        the flag's shape, and the no-handshake contract that goes with it, out
        of this class.
        """
        # Moved VERBATIM from batchgen_worker.py (S8); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        self._cohort_shutdown_probe = probe

    def _cohort_shutdown_requested(self, worker) -> bool:
        """Has this cohort's teardown begun?

        Called once per decode step on the hot path, so it stays a single
        attribute load and, when armed, one shared-byte load behind the probe:
        no lock, no allocation, no logging.  Unarmed (no supervisor, or a code
        path that never serves a pool) it is a ``None`` test and nothing else.
        """
        # Moved VERBATIM from batchgen_worker.py (S8); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        probe = self._cohort_shutdown_probe
        return probe is not None and probe()

    def _poll_until(self, worker, ready, deadline) -> bool:
        """Poll one NON-BLOCKING readiness probe until the shared deadline.

        ``ready()`` must answer "is this work finished?" without waiting:
        ``KVAsyncTask.done()`` (a ``wait_for(0s)`` status test) and
        ``cudaEventQuery`` behind ``torch.cuda.Event.query()`` are the two
        shapes used here.  That is the whole point of this helper.  The waits
        themselves may NOT be bounded this way, nor on a helper thread: the
        append task's ``wait`` holds the GIL while it blocks, so one such call
        anywhere in the process freezes the deadline that is supposed to bound
        it, the abort watchdog and the supervisor's self-exit net along with it.

        The probe is read once before the clock, so work that is already
        finished proves itself even if the budget is gone -- readiness is
        readiness, and self-exiting a rank whose readers have all returned would
        throw away a perfectly ordered unwind.

        Returns True only if the probe answered True.  Anything else, including
        a probe that raises (a task type with no non-blocking ``done()``), is
        reported as not-quiesced: this runs during teardown, and the caller's
        safe answer to "I cannot prove the local readers are done" is to end the
        process rather than unwind.  It never raises, for the same reason the
        quiesce itself never raises.
        """
        # Moved VERBATIM from batchgen_worker.py (S8); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        try:
            while True:
                if ready():
                    return True
                if time.perf_counter() >= deadline:
                    return False
                time.sleep(_COHORT_QUIESCE_POLL_INTERVAL_S)
        except BaseException:
            logging.warning(
                "Rank %s: probing a decode teardown quiesce wait failed; "
                "treating it as not quiesced",
                self.rank,
                exc_info=True,
            )
            return False

    def _pending_kv_appends_done(self, worker) -> bool:
        """Has every pending host-KV append task finished?  Never waits.

        ``done()`` is ``future_.wait_for(0s) == ready`` in the native task
        (``core/KV_Storage/host_paged_kv_worker_view.h:93-99``, bound at
        ``core/batchgen_Binding.cpp:969``), so this is a status read per task and
        nothing else.  ``None`` entries are skipped exactly as the real drain
        skips them, and a worker that never created the list (the drain's own
        ``hasattr`` guard admits that case) has nothing in flight.
        """
        # Moved VERBATIM from batchgen_worker.py (S8); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        for task in getattr(self, "_pending_kv_append_tasks", ()):
            if task is not None and not task.done():
                return False
        return True

    def _record_quiesce_device_event(self, worker):
        """A non-blocking probe for the device work this rank already queued.

        This replaces the ``torch.cuda.synchronize(self.torch_device)`` the load
        drain used to end in: on an abandoned step that sync is unbounded, and
        it is the last thing standing between the teardown and the unwakeable
        state one level down.  An event recorded on the worker's current stream
        (the stream the decode step records its token event on) is ready exactly
        once the work queued ahead of it has run, and ``query()`` reads that
        with ``cudaEventQuery`` -- no wait, no GIL hazard.

        Returns the event, or ``None`` if it could not even be recorded, which
        the caller treats as "the device state cannot be proven" and self-exits.
        """
        # Moved VERBATIM from batchgen_worker.py (S8); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        try:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.torch_device))
            return event
        except BaseException:
            logging.warning(
                "Rank %s: recording the decode teardown quiesce device probe "
                "failed; treating it as not quiesced",
                self.rank,
                exc_info=True,
            )
            return None

    def _self_exit_on_quiesce_deadline(self, worker) -> None:
        """End this process instead of unwinding past a quiesce that did not land.

        The ordered teardown releases the very mappings the parked waits hold
        raw pointers into -- the host-KV appends are std::async threads reading
        the worker view's page table unlocked -- so an unwind that cannot prove
        they finished is a use-after-free during teardown.  That is the one
        hazard this whole exit path exists to avoid, so when the bound expires
        the rank takes the supervisor's own bounded outcome directly.

        The marker keeps the ``path=self-exit`` grep key the supervisor's safety
        net emits, so marker accounting still sums, and adds a reason so the two
        causes stay distinguishable in the logs.  ``os._exit`` runs no atexit
        hook and no stream flush, so the marker is flushed first or it is lost.

        The status, the path marker and the pre-exit flush are the worker exit
        protocol the supervisor uses too; both take them from
        ``batchgen.eventlog.process``, so there is one definition of each.
        """
        # Moved VERBATIM from batchgen_worker.py (S8); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        from batchgen.eventlog.process import (
            EXIT_PATH_SELF_EXIT,
            WORKER_SELF_EXIT_CODE,
            flush_logs_before_exit,
            record_exit,
        )

        logging.error(
            "[shutdown] worker exit path=%s rank=%s pid=%s reason=%s",
            EXIT_PATH_SELF_EXIT,
            self.rank,
            os.getpid(),
            _COHORT_QUIESCE_DEADLINE_REASON,
        )
        flush_logs_before_exit()
        record_exit(WORKER_SELF_EXIT_CODE, _COHORT_QUIESCE_DEADLINE_REASON)
        os._exit(WORKER_SELF_EXIT_CODE)

    def _quiesce_decode_for_cohort_shutdown(self, worker, pending_async_task=None) -> None:
        """Let LOCAL in-flight decode work finish before the ordered teardown.

        Called on the decode loop's teardown exits.  Everything waited on here
        is work this rank already launched and that holds raw pointers into
        mappings the ordered teardown is about to release -- the host-KV append
        tasks (std::async threads reading the worker view's page table with no
        lock) and an async host->GPU KV load.  Returning out of decode with
        either still in flight is a use-after-free during teardown, which is
        why the exit quiesces instead of just leaving.

        Every wait here is deliberately LOCAL.  ``defer_errors=True`` keeps the
        append drain off the cross-rank error sync, which is the one thing this
        exit must never do: the peer whose death started this teardown will
        never join a collective again, so a guarded one would hang exactly
        where the unfixed code hung.

        Nothing raises out of here.  This runs on a teardown path whose whole
        purpose is to unwind normally -- an exception would escape as an error
        and lose the ordered exit -- so a failed quiesce is logged and the
        teardown continues; the process is about to drop these mappings anyway.

        The quiesce is BOUNDED, by one shared ``_COHORT_QUIESCE_DEADLINE_S``
        budget, and the bound is established by NON-BLOCKING PROBES ONLY.  On
        the exit taken when the step's token event was ABANDONED the step's
        device work is parked behind the dead collective, so every wait here is
        parked with it -- and MEASURED, the append task's ``wait`` blocks while
        HOLDING THE GIL (no ``gil_scoped_release`` in its binding), which
        freezes the deadline meant to bound it, the abort watchdog and the
        supervisor's self-exit net.  A blocking wait therefore may not be
        called, on this thread or any other, until something non-blocking has
        proven it cannot block:

          1. every pending host-KV append task reports ``done()``;
          2. the async KV load task reports ``done()`` (the dual coordinator's
             composite task answers for both of its halves);
          3. the device-work probe recorded at entry reports ``query()``, which
             is what the load drain's unbounded device sync became.

        All three inside the budget: the real drains then run with every future
        ready and the ordered unwind proceeds exactly as before.  Any of them
        expiring is the one case that may not unwind at all (the readers are
        still live, holding page-table pointers), so it self-exits with the
        reason recorded, inside ~5 s instead of the supervisor's grace window.
        On the healthy clean-restart exits all three probes answer immediately.
        """
        # Moved VERBATIM from batchgen_worker.py (S8); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        deadline = time.perf_counter() + _COHORT_QUIESCE_DEADLINE_S
        device_event = self._record_quiesce_device_event()

        quiesced = self._poll_until(self._pending_kv_appends_done, deadline)
        if quiesced and pending_async_task is not None:
            quiesced = self._poll_until(lambda: pending_async_task.done(), deadline)
        if quiesced:
            quiesced = device_event is not None and self._poll_until(
                device_event.query, deadline
            )

        if quiesced:
            try:
                if self._pending_kv_append_tasks:
                    self._wait_pending_kv_append_tasks(defer_errors=True)
            except BaseException:
                logging.warning(
                    "Rank %s: draining host-KV appends at the decode teardown "
                    "boundary failed; continuing the ordered teardown",
                    self.rank,
                    exc_info=True,
                )
            try:
                if pending_async_task is not None:
                    pending_async_task.wait()
            except BaseException:
                logging.warning(
                    "Rank %s: draining the pending async KV load at the decode "
                    "teardown boundary failed; continuing the ordered teardown",
                    self.rank,
                    exc_info=True,
                )

        self.disable_decode_watchdog()
        if not quiesced:
            self._self_exit_on_quiesce_deadline()

    # ---- Communicator handles ----------------------------------------------

    def _get_or_create_gloo_group(self, worker):
        """Get or create a Gloo process group for CPU tensor migrations.

        Gloo backend supports CPU tensors and can use RDMA if available.
        This is more memory efficient than NCCL (which requires GPU staging).

        Returns:
            The Gloo process group for CPU tensor operations.
        """
        # Moved VERBATIM from batchgen_worker.py (S8); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        if not hasattr(self, '_gloo_migration_group') or self._gloo_migration_group is None:
            logging.debug(f"Rank {self.rank}: Creating Gloo process group for CPU migrations")
            # Create a new group with Gloo backend including all ranks
            self._gloo_migration_group = dist.new_group(
                ranks=list(range(self.world_size)),
                backend="gloo"
            )
            logging.debug(f"Rank {self.rank}: Gloo process group created")
        return self._gloo_migration_group

    def _ensure_pynccl_communicator(self, worker) -> None:
        """Create the PyNccl communicator unless every rank already holds one.

        Collective and idempotent: the need-init vote, the port broadcast and
        the pre-TCPStore barrier are all torch.distributed collectives, so every
        rank must call this together. Once all ranks have a communicator the
        vote fails and nothing is created.
        """
        # Moved VERBATIM from batchgen_worker.py (S8); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        if os.getenv("BATCHGEN_ENABLE_ALL_TO_ALL", "0") != "0":
            return

        # Verify rank consistency
        if dist.is_initialized():
            assert self.rank == dist.get_rank(), \
                f"Rank mismatch: self.rank={self.rank}, dist.get_rank()={dist.get_rank()}"

        # Skip PyNccl initialization for single GPU (no inter-GPU communication needed)
        if self.world_size == 1:
            logging.debug("Single GPU mode: skipping PyNccl communicator initialization")
            return

        comm_master_addr = os.getenv("COMM_MASTER_ADDR")

        # Coordinate PyNccl initialization across all ranks
        # Use all_reduce to check if ANY rank needs to (re)init the communicator
        need_init = 1 if self.comm is None else 0
        need_init_tensor = torch.tensor([need_init], dtype=torch.int32, device=self.torch_device)
        dist.all_reduce(need_init_tensor, op=dist.ReduceOp.MAX)
        any_rank_needs_init = need_init_tensor.item() > 0

        if not any_rank_needs_init:
            return

        # All ranks must participate in init - destroy any existing comm first
        if self.comm is not None:
            logging.info(f"Rank {self.rank}: Destroying existing comm for coordinated reinit")
            try:
                self.comm.destroy()
            except Exception:
                pass
            self.comm = None
            if hasattr(self, '_nccl_group') and self._nccl_group is not None:
                del self._nccl_group
                self._nccl_group = None

        device = torch.device("cuda", self.local_rank)

        if comm_master_addr is None:
            logging.warning(f"Rank {self.rank}: COMM_MASTER_ADDR not set, skipping PyNccl init")
        elif StatelessProcessGroup is not None and PyNcclCommunicator is not None:
            # Rank 0 finds an available port, then broadcasts to all ranks
            selected_port = self._nccl_port
            if self.rank == 0:
                try:
                    selected_port = self._find_available_pynccl_port(
                        comm_master_addr,
                        self._nccl_port,
                    )
                    logging.debug(f"Rank 0: Found available port {selected_port} for PyNccl")
                except RuntimeError as e:
                    logging.error(f"Rank 0: Failed to find available port: {e}")
                    selected_port = -1

            # Broadcast the chosen port from rank 0 to all ranks
            port_tensor = torch.tensor([selected_port], dtype=torch.int32, device=self.torch_device)
            dist.broadcast(port_tensor, src=0)
            self._nccl_port = port_tensor.item()
            if self._nccl_port < 0:
                raise RuntimeError(
                    "No available port remains in this runtime's PyNccl range"
                )

            # CRITICAL: Barrier before TCPStore creation to ensure rank 0 (the server)
            # is ready before other ranks try to connect. Different ranks may reach
            # this point at very different times due to tokenization workload.
            logging.debug(f"Rank {self.rank}: Waiting for all ranks before PyNccl init...")
            dist.barrier()

            try:
                logging.debug(f"Rank {self.rank}: Creating PyNccl communicator on port {self._nccl_port}")

                # Store group separately so we can properly destroy it on reinit
                self._nccl_group = StatelessProcessGroup.create(
                    host=comm_master_addr,
                    port=self._nccl_port,
                    rank=self.rank,
                    world_size=self.world_size,
                    data_expiration_seconds=36000,  # 10 hours
                )
                self.comm = PyNcclCommunicator(
                    group=self._nccl_group,
                    device=device
                )
                # Only rank 0 logs at INFO level to reduce verbosity
                if self.rank == 0:
                    logging.info(f"PyNccl communicator initialized on port {self._nccl_port}")
                else:
                    logging.debug(f"Rank {self.rank}: PyNccl communicator initialized on port {self._nccl_port}")
            except Exception as e:
                logging.error(f"Rank {self.rank}: PyNccl communicator initialization failed - {e}")
                raise RuntimeError(f"Rank {self.rank}: PyNccl communicator initialization failed - {e}")

    def _find_available_pynccl_port(self, worker, host: str, start_port: int) -> int:
        # Moved VERBATIM from batchgen_worker.py (S8); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        end_port = self.pynccl_port_base + self.pynccl_port_span
        start_port = max(start_port, self.pynccl_port_base)
        remaining = end_port - start_port
        if remaining <= 0:
            raise RuntimeError(
                "PyNccl port range exhausted: "
                f"[{self.pynccl_port_base}, {end_port})"
            )
        return _find_available_port(host, start_port, remaining)

    def _check_and_reinit_pynccl(self, worker) -> bool:
        """
        Check if PyNccl communicator is healthy. If not, attempt to reinitialize.
        Returns True if communicator is healthy (or was successfully reinitialized).
        """
        # Moved VERBATIM from batchgen_worker.py (S8); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        if self.comm is None:
            # Will be lazily initialized in generate()
            return True

        # Skip health check if communicator is not available (e.g., single GPU)
        if not self.comm.available:
            logging.debug(f"Rank {self.rank}: PyNccl communicator not available, skipping health check")
            return True

        try:
            # Quick health check using PyNccl all_reduce
            # CRITICAL: Must enable the communicator first - it's disabled by default after init
            health_tensor = torch.ones(1, device=self.torch_device, dtype=torch.float32)
            with self.comm.change_state(enable=True):
                self.comm.all_reduce(health_tensor, op=dist.ReduceOp.SUM, stream=torch.cuda.current_stream())
            torch.cuda.synchronize(self.torch_device)

            expected = float(self.world_size)
            if abs(health_tensor.item() - expected) > 1e-6:
                raise RuntimeError(f"PyNccl health check mismatch: got {health_tensor.item()}, expected {expected}")

            logging.debug(f"Rank {self.rank}: PyNccl health check passed")
            return True
        
        except Exception as e:
            logging.warning(f"Rank {self.rank}: PyNccl health check failed: {e}")
            logging.info(f"Rank {self.rank}: Attempting to reinitialize PyNccl communicator...")

            # Destroy old communicator
            try:
                if self.comm is not None:
                    self.comm.destroy()
                    self.comm = None
                    logging.info(f"Rank {self.rank}: NCCL communicator destroyed successfully")
            except Exception as destroy_e:
                logging.warning(f"Rank {self.rank}: Error destroying PyNccl communicator: {destroy_e}")
                self.comm = None

            # Destroy old group (releases TCPStore and port)
            try:
                if self._nccl_group is not None:
                    # The group's store should be garbage collected when group is deleted
                    del self._nccl_group
                    self._nccl_group = None
                    # Force garbage collection to release TCPStore socket
                    gc.collect()
                    logging.info(f"Rank {self.rank}: NCCL group destroyed successfully")
            except Exception as group_e:
                logging.warning(f"Rank {self.rank}: Error destroying NCCL group: {group_e}")
                self._nccl_group = None

            # Synchronize all ranks before any tries to recreate (uses torch.distributed)
            # This ensures all ranks have released their connections before rank 0
            # tries to create a new TCPStore server
            try:
                if dist.is_initialized():
                    dist.barrier()
                    logging.debug(f"Rank {self.rank}: Barrier after NCCL cleanup passed")
            except Exception as barrier_e:
                logging.warning(f"Rank {self.rank}: Barrier after NCCL cleanup failed: {barrier_e}")

            # Find next available port for reinitialization
            # Rank 0 finds the port, then broadcasts to all ranks
            comm_master_addr = os.getenv("COMM_MASTER_ADDR", "127.0.0.1")
            selected_port = self._nccl_port
            if self.rank == 0:
                try:
                    selected_port = self._find_available_pynccl_port(
                        comm_master_addr,
                        self._nccl_port + 1,
                    )
                    logging.debug(f"Rank 0: Found available port {selected_port} for PyNccl reinit")
                except RuntimeError as e:
                    logging.error(f"Rank 0: Failed to find available port: {e}")
                    selected_port = -1
            # Broadcast port to all ranks
            port_tensor = torch.tensor([selected_port], dtype=torch.int32, device=self.torch_device)
            dist.broadcast(port_tensor, src=0)
            self._nccl_port = port_tensor.item()
            if self._nccl_port < 0:
                return False
            logging.debug(f"Rank {self.rank}: Next PyNccl port will be {self._nccl_port}")

            # Delay to allow OS to fully release resources
            # Rank 0 needs extra time since it's the TCPStore server
            if self.rank == 0:
                time.sleep(1.0)
            else:
                time.sleep(0.5)

            # Will be reinitialized lazily in generate()
            return True

    def init_nvshmem(self, worker):
        """Initialize NVSHMEM only once per batch, not per decode iteration."""
        # Moved VERBATIM from batchgen_worker.py (S8); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        if BATCHGEN_ENABLE_ALL_TO_ALL != "1" or nvshmem_init is None:
            if self.rank == 0:
                logging.debug("Skipping NVSHMEM initialization; BATCHGEN_ENABLE_ALL_TO_ALL is disabled")
            return
    
        # Check if already initialized this run
        if getattr(self, '_nvshmem_initialized_this_run', False):
            logging.debug(f"Rank {self.rank}: NVSHMEM already initialized this run, skipping")
            return
        
        import nvshmem.core as nvshmem
        from cuda.core.experimental import Device    
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)

        dev = Device(local_rank)
        dev.set_current()
        dist.barrier()
        nvshmem_init(
            global_rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=dev
        )
        self._nvshmem_initialized_this_run = True
        print(f"Rank {rank}: NVSHMEM initialized and Symmetric Heap allocated.")

    # ---- SyncCoordinator facade --------------------------------------------

    def _make_sync_coordinator(self, worker) -> SyncCoordinator:
        # Moved VERBATIM from batchgen_worker.py (S8); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        if self._sync_coordinator is None:
            self._sync_coordinator = SyncCoordinator(backend=TorchDistCollectiveBackend())
        return self._sync_coordinator

    def _make_sync_context(self, worker) -> SyncContext:
        # Moved VERBATIM from batchgen_worker.py (S8); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        return SyncContext(
            rank=self.rank,
            uuid_to_local=self._uuid_to_local_map,
            global_batch=self.global_batch,
            torch_device=self.torch_device,
        )

    def _sync_completion_status_tensor(
        self, worker,
        decode_uuids: List[str],
    ) -> Tuple[Set[str], List[str]]:
        # Moved VERBATIM from batchgen_worker.py (S8); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        return self._make_sync_coordinator().sync_completion_status_tensor(
            self._make_sync_context(), decode_uuids
        )

    def _sync_decode_uuids_tensor(
        self, worker,
        decode_uuids: List[str],
    ) -> List[str]:
        # Moved VERBATIM from batchgen_worker.py (S8); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        return self._make_sync_coordinator().sync_decode_uuids_tensor(
            self._make_sync_context(), decode_uuids
        )

    # ---- Rank agreement on per-step MoE token counts -----------------------

    def _sync_decode_moe_rank_counts(self, worker, batch: List[int], *, reason: str) -> int:
        """Synchronize per-rank decode row counts for 3D MoE padding masks."""
        # Moved VERBATIM from batchgen_worker.py (S8); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        local_count = int(len(batch))
        if self._decode_local_count_tensor is None:
            self._decode_local_count_tensor = torch.empty(
                (1,), dtype=torch.int64, device=self.torch_device
            )
        if (
            self._decode_all_rank_counts is None
            or self._decode_all_rank_counts.shape[0] != self.world_size
        ):
            self._decode_all_rank_counts = torch.empty(
                (self.world_size,), dtype=torch.int64, device=self.torch_device
            )
        local_count_tensor = self._decode_local_count_tensor
        all_rank_counts = self._decode_all_rank_counts
        local_count_tensor.fill_(local_count)
        dist.all_gather_into_tensor(all_rank_counts, local_count_tensor)
        max_batch_size = int(all_rank_counts.max().item())

        self._current_decode_local_batch_size = local_count
        self._current_decode_max_rank_batch_size = max_batch_size
        self._current_decode_rank_token_counts = all_rank_counts

        if max_batch_size > 0 and hasattr(self, 'parallel_manager') and self.parallel_manager is not None:
            # M2b: under DP-(world/G) x TP-G decode the 32-way max above is the
            # per-GROUP batch B_grp (the G ranks of a group hold identical
            # sequences). Decode scatters those rows across the group's G ranks
            # before the DP-32 resident MoE, so the padded all_gather/all_reduce
            # layout is sized by the POST-scatter share ceil(B_grp/G), not B_grp.
            # G==1 -> ceil(max/1)==max, byte-identical to the pure-DP path.
            from batchgen.decode_dp_group import moe_ntp_from_group_max
            _G = getattr(self.parallel_manager, "attn_tp_size", 1)
            moe_ntp = moe_ntp_from_group_max(max_batch_size, _G)
            if hasattr(self.parallel_manager, 'set_num_tokens_per_rank'):
                self.parallel_manager.set_num_tokens_per_rank(moe_ntp)
            if hasattr(self.parallel_manager, 'set_rank_token_counts'):
                self.parallel_manager.set_rank_token_counts(all_rank_counts)

        if BATCHGEN_MULTI_BATCH_DIAG:
            try:
                counts_list = all_rank_counts.detach().cpu().tolist()
            except RuntimeError:
                counts_list = ["<unavailable>"]
            logging.info(
                f"[GLM5_MOE_COUNTS] Rank {self.rank}: reason={reason} "
                f"local={local_count} max={max_batch_size} counts={counts_list}"
            )
        return max_batch_size

    def _sync_prefill_moe_rank_counts(
        self, worker, local_token_count: int, *, reason: str
    ) -> int:
        """Size resident-EP prefill collectives after TP-group row scattering."""
        # Moved VERBATIM from batchgen_worker.py (S8); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        local = torch.tensor(
            [int(local_token_count)],
            dtype=torch.int64,
            device=self.torch_device,
        )
        counts = torch.empty(
            (self.world_size,), dtype=torch.int64, device=self.torch_device
        )
        dist.all_gather_into_tensor(counts, local)
        G = self._decode_attn_tp_size()
        counts_list = [int(value) for value in counts.cpu().tolist()]
        for start in range(0, self.world_size, G):
            group = counts_list[start:start + G]
            if len(set(group)) != 1:
                raise RuntimeError(
                    "resident-EP prefill requires identical token rows within "
                    f"each TP group; ranks {start}:{start + G} reported {group}"
                )
        max_group_tokens = max(counts_list)
        from batchgen.decode_dp_group import moe_ntp_from_group_max
        moe_ntp = moe_ntp_from_group_max(max_group_tokens, G)
        self.parallel_manager.set_num_tokens_per_rank(moe_ntp)
        if self.rank == 0:
            logging.info(
                "[K3_PREFILL_MOE] reason=%s group_token_counts=%s "
                "post_scatter_ntp=%s",
                reason,
                [counts_list[start] for start in range(0, self.world_size, G)],
                moe_ntp,
            )
        return moe_ntp
