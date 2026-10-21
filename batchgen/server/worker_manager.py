"""Worker lifecycle management and inference bridging for the FastAPI server."""

from __future__ import annotations

import gc
import json
import logging
import os
import pickle
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.multiprocessing as mp

from batchgen.batchgen_worker import BatchGenWorkerArgs
from batchgen.config.model_name_utils import is_kimi_k25_backend_model
from batchgen.kv_cache.host_kv_mananger_config import build_host_kv_config
from batchgen.models.engine_loader import core_engine as bg_lib
from batchgen.parameter_server_client import ParameterServerClient
from batchgen.server.gpu_arch import detect_gpu_arch  # noqa: F401  (re-export)
from batchgen.memfd import create_memfd
from batchgen.server.process_utils import (
    cleanup_shm_files,
    get_hugepage_size,
    get_model_byte_size,
)
from batchgen.server.runtime_locks import (
    RuntimeLocks,
    hold_run_lock,
    list_run_named_objects,
    reclaim_dead_runs,
    release_run_lock,
)
from batchgen.server.runtime_lease import LaneLease
from batchgen.server.server_args import (
    ServerArgs,
    validate_shared_runtime_capability,
)
from batchgen.utils import config_torch_module_initializer

logger = logging.getLogger(__name__)

PARAMETER_SERVER_ENDPOINT_ENV = "BATCHGEN_PARAMETER_SERVER_ENDPOINT"
# A sentinel reaches the worker's ordered C++/Python teardown path.  That path
# must be allowed to finish before the signal fallback can invoke os._exit and
# skip the destructors it exists to run.  The bound still keeps a blocked
# worker from holding server shutdown indefinitely.  Releasing a large
# cudaHostRegister'd weight footprint takes on the order of minutes, so the
# grace must cover that or the fallback kills workers mid-unpin into
# uninterruptible D state.
_WORKER_SENTINEL_GRACE_S = 180.0
# A SIGKILL'd worker still cannot be reaped until the kernel finishes
# unpinning its registered pages, which again takes minutes for a large
# cohort.  Proceeding while an owned PID is alive is never safe, so the
# reap wait is long and loud rather than short and fatal.  Observed
# teardowns of a large pinned footprint finished only just inside the
# previous budget, leaving no usable margin, so the budget now covers
# several times the time such a teardown is expected to need.
_WORKER_KILL_REAP_TIMEOUT_S = 900.0
# A worker that needed SIGTERM/SIGKILL cannot name its own exit path: it is
# force-killed while a native collective holds its thread, so no Python runs in
# it again.  The marker is therefore emitted here, by the only process that
# knows a signal replaced an ordered exit.
_WORKER_EXIT_PATH_SIGNAL = "signal-fallback"


def _resolve_local_world_size(
    world_size: int,
    nnodes: int,
    visible_device_count: int,
    *,
    require_exact_visibility: bool = False,
) -> int:
    if nnodes <= 0 or world_size <= 0:
        raise ValueError("world_size and nnodes must be positive")
    if world_size % nnodes != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by nnodes ({nnodes})"
        )
    local_world_size = world_size // nnodes
    if local_world_size > visible_device_count:
        raise ValueError(
            f"world_size ({world_size}) requires {local_world_size} GPUs per "
            f"node, but only {visible_device_count} GPUs are visible"
        )
    if require_exact_visibility and local_world_size != visible_device_count:
        raise ValueError(
            f"shared mode requires exactly {local_world_size} visible GPUs, "
            f"got {visible_device_count}"
        )
    return local_world_size


def _validate_shmem_enabled() -> None:
    """Check that THP shmem is enabled for --fast-init. Raises RuntimeError if not."""
    import re
    sysfs_path = "/sys/kernel/mm/transparent_hugepage/shmem_enabled"
    try:
        with open(sysfs_path) as f:
            line = f.read().strip()
        match = re.search(r'\[(\w+)\]', line)
        active = match.group(1) if match else "unknown"
        if active not in ("always", "within_size"):
            raise RuntimeError(
                f"--fast-init requires shmem_enabled='always' or 'within_size', "
                f"got '{active}'. Fix: echo always > {sysfs_path}"
            )
        logger.info("THP shmem_enabled=%s (OK for --fast-init)", active)
    except FileNotFoundError:
        raise RuntimeError(
            f"--fast-init requires THP support. {sysfs_path} not found."
        )


class WorkerExitState:
    """Tracks worker failures that require the main process to exit."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason: Optional[str] = None
        self._exception: Optional[BaseException] = None

    def set_failure(
        self, reason: str, exc: Optional[BaseException] = None
    ) -> bool:
        with self._lock:
            if self._event.is_set():
                return False
            self._reason = reason
            self._exception = exc
            self._event.set()
            return True

    def is_failed(self) -> bool:
        return self._event.is_set()

    def clear(self) -> None:
        with self._lock:
            self._reason = None
            self._exception = None
            self._event.clear()

    @property
    def reason(self) -> Optional[str]:
        return self._reason

    @property
    def exception(self) -> Optional[BaseException]:
        return self._exception


class WorkerManager:
    """Starts GPU workers and exposes a thread-safe inference API."""

    # Bound on draining the distributed-weight daemon's in-flight remote
    # fetches during a cohort swap. A fetch that does not land within it is a
    # stalled transport, which the daemon latches as fatal.
    _DISTRIBUTED_WEIGHT_SWAP_TIMEOUT_S = 120.0

    def __init__(
        self,
        server_args: ServerArgs,
        worker_exit_state: Optional[WorkerExitState] = None,
    ):
        self.args = server_args
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            # Start method may already be set by the parent process.
            pass
        self._mp_ctx = mp.get_context("spawn")
        self.request_queue: Optional[mp.Queue] = None
        self.response_queue: Optional[mp.Queue] = None
        # Cohort-scoped teardown flag; bound with the channels below.
        self._cohort_shutdown_flag: Optional[Any] = None
        self._create_worker_channels()
        self.worker_process = None
        self.started = False
        self.model_info: Dict[str, Any] = {}
        self.args_dict: Dict[str, Any] = {}
        self.parameter_server_instance = None
        self.host_kv_manager = None
        self.host_kv_aux_manager = None
        self.distributed_weight_daemon = None
        self.distributed_weight_config = None
        # The server-owned staged copy of the distributed weight store, and
        # the config the daemon and every worker receive: the effective
        # config naming that copy, or the builder's config when it is legacy.
        self._staged_weight_store = None
        self._distributed_weight_config_path: Optional[Path] = None
        self.skeleton_state_dict = None
        # The skeleton state dict lives in an anonymous memfd this process
        # holds open; workers read it back through /proc/<our pid>/fd/<N>.
        self.skeleton_state_dict_file = None
        self.skeleton_memfd_fd = -1
        self._lock = threading.Lock()
        self._worker_exit_state = worker_exit_state or WorkerExitState()
        self._monitor_stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._join_lock = threading.Lock()
        # One owner per cohort teardown: see _handle_worker_failure.
        self._teardown_lock = threading.Lock()
        self._stopping = False
        self._monitor_interval_s = 1.0
        self._ready_event = self._mp_ctx.Event()
        self._fatal_ack_event = self._mp_ctx.Event()
        self.prefix_cache_runtime_config = None
        self.prefix_cache_coordinator = None
        self._local_world_size: Optional[int] = None
        self._runtime_dir_created = False
        self._runtime_namespace_owned = False
        self._runtime_locks: Optional[RuntimeLocks] = None
        self._lane_lease: Optional[LaneLease] = None
        # Held LOCK_SH for this process's whole life; see hold_run_lock.
        self._run_lock = None
        self._state = "serving"
        self._generation = 0
        self._first_ready = False
        self._reconciled_failure = False
        self._main_thread_dispatcher: Optional[Callable[[Callable[[], Any], float], Any]] = None
        self._quiesce_hook: Optional[Callable[[], Any]] = None
        self._resume_hook: Optional[Callable[[], Any]] = None
        self._fail_in_flight_hook: Optional[Callable[[str], int]] = None
        self._scheduler_generation_hook: Optional[Callable[[], Any]] = None
        self._pool_init_params_hook: Optional[Callable[[], Dict[str, Any]]] = None
        self._retire_channels_hook: Optional[Callable[[], Any]] = None
        self._restart_timeout_s = 600.0

    def _store_skeleton_state_dict(self, skeleton_state_dict: Dict[str, Any]) -> None:
        """Serialize the skeleton state dict into an anonymous memfd.

        A named file in the runtime dir survives a forced exit of the serving
        processes; a memfd cannot — the kernel reclaims it with its last
        reference however this process died. Workers ``torch.load`` it through
        ``/proc/<our pid>/fd/<N>``, so the descriptor stays open for this
        process's life, well past the point where the last worker has loaded
        it. MFD_CLOEXEC keeps an exec'd child from holding it open instead.
        """
        self._close_skeleton_memfd()
        fd = create_memfd("batchgen_skeleton")
        try:
            # dup so closing the wrapper does not close the descriptor the
            # workers will reach through /proc.
            with os.fdopen(os.dup(fd), "wb") as handle:
                torch.save(skeleton_state_dict, handle)
            size = os.fstat(fd).st_size
        except BaseException:
            os.close(fd)
            raise
        self.skeleton_memfd_fd = fd
        self.skeleton_state_dict_file = f"/proc/{os.getpid()}/fd/{fd}"
        logger.info(
            "Skeleton state dict held in an anonymous memfd, readable at %s "
            "(%d keys, %.2f MB)",
            self.skeleton_state_dict_file,
            len(skeleton_state_dict),
            size / (1024**2),
        )

    def _close_skeleton_memfd(self) -> None:
        """Release the skeleton memfd. Nothing is left behind to unlink."""
        if self.skeleton_memfd_fd >= 0:
            try:
                os.close(self.skeleton_memfd_fd)
            except OSError as e:
                logging.warning(f"Failed to close skeleton memfd: {e}")
            self.skeleton_memfd_fd = -1
        self.skeleton_state_dict_file = None

    def _prepare_runtime_dir(self) -> None:
        runtime_dir = self.args.runtime_identity.runtime_dir
        runtime_dir.mkdir(mode=0o700, exist_ok=False)
        self._runtime_dir_created = True
        # Create the run's liveness lock before anything else can use the
        # directory: every worker opens this exact file, and only a run whose
        # processes are all gone can be reclaimed by a later start.
        self._run_lock = hold_run_lock(runtime_dir, create=True)
        shm_dir = Path("/dev/shm")
        prefix = self.args.runtime_identity.shm_prefix
        if shm_dir.is_dir() and any(
            entry.name.startswith(prefix) for entry in shm_dir.iterdir()
        ):
            raise RuntimeError(
                f"runtime namespace {prefix!r} already has shared-memory objects"
            )
        self._runtime_namespace_owned = True

    # ---------------------- Public API ----------------------
    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def set_lifecycle_hooks(
        self,
        *,
        quiesce: Optional[Callable[[], Any]] = None,
        resume: Optional[Callable[[], Any]] = None,
        fail_in_flight: Optional[Callable[[str], int]] = None,
        generation_ready: Optional[Callable[[], Any]] = None,
        pool_init_params: Optional[Callable[[], Dict[str, Any]]] = None,
        retire_channels: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._quiesce_hook = quiesce
        self._resume_hook = resume
        self._fail_in_flight_hook = fail_in_flight
        self._scheduler_generation_hook = generation_ready
        self._pool_init_params_hook = pool_init_params
        self._retire_channels_hook = retire_channels

    def set_main_thread_dispatcher(
        self,
        dispatcher: Optional[Callable[[Callable[[], Any], float], Any]],
    ) -> None:
        """Install the event-loop bridge used when restart starts off-main-thread."""
        self._main_thread_dispatcher = dispatcher

    def status(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "generation": self.generation,
            "workers": len(getattr(self.worker_process, "processes", ()) or ()),
        }

    def _acquire_runtime_admission(self) -> None:
        if self._runtime_locks is not None:
            return
        validate_shared_runtime_capability(self.args)
        if (
            self.args.runtime_mode == "shared"
            and os.getenv(PARAMETER_SERVER_ENDPOINT_ENV)
        ):
            raise ValueError("shared mode rejects an external parameter server")

        self._local_world_size = _resolve_local_world_size(
            self.args.world_size,
            self.args.nnodes,
            torch.cuda.device_count(),
            require_exact_visibility=self.args.runtime_mode == "shared",
        )
        if self.args.runtime_mode == "shared":
            self._lane_lease = LaneLease.acquire(self.args)
        self._runtime_locks = RuntimeLocks.acquire(self.args.runtime_identity)
        # The instance lock is held and this run has no runtime directory yet, so
        # every other directory of this instance belongs to a previous run. Its
        # run id is fresh random hex, so nothing but this scan can ever remove it.
        reclaim_dead_runs(self.args.runtime_identity)

    def start(self) -> None:
        if self.started:
            return
        try:
            self._start_impl()
        except BaseException:
            try:
                self.stop()
            except BaseException:
                logger.exception("Runtime rollback failed after startup error")
            raise

    def _start_impl(self) -> None:
        import time as _time

        startup_start = _time.monotonic()
        self._acquire_runtime_admission()

        self._stopping = False
        self._monitor_stop_event.clear()
        self._ready_event.clear()
        self._fatal_ack_event.clear()
        self._prepare_runtime_dir()

        if self.args.fast_init:
            _validate_shmem_enabled()
            self._compact_memory()

        if self.args.distributed_weight_config is not None:
            self._stage_distributed_weight_store()
            self._start_distributed_weight_daemon()

        if self.args.enable_hugetlbfs:
            byte_size = get_model_byte_size(self.args.model)
            self._config_hugepages(byte_size)

        import sys as _diag_sys
        def _diag(msg):
            print(f"[DIAG {_time.time():.3f}] {msg}", flush=True)
            _diag_sys.stdout.flush()

        _diag(">>> config_torch_module_initializer")
        config_torch_module_initializer()
        _diag("<<< config_torch_module_initializer")
        if self.args.host_kv_cache_size:
            kv_start = _time.monotonic()
            try:
                _diag(">>> allocate_host_kv_cache")
                result = self.allocate_host_kv_cache(
                    self.args.host_kv_cache_size,
                    self.args.model,
                    primary_shm_name=self.args.runtime_identity.host_kv_shm_name,
                    aux_shm_name=self.args.runtime_identity.host_kv_aux_shm_name,
                    enable_thp=self.args.fast_init,
                )
                _diag("<<< allocate_host_kv_cache")
                if isinstance(result, tuple):
                    self.host_kv_manager, self.host_kv_aux_manager = result
                else:
                    self.host_kv_manager = result
                    self.host_kv_aux_manager = None
            except Exception as exc:
                if self.args.enable_prefix_cache:
                    # Prefix reuse stores shared pages in Host KV; serving
                    # without it would silently disable the feature.
                    raise RuntimeError(
                        "Host KV allocation is required for prefix cache reuse"
                    ) from exc
                logger.warning("Host KV cache allocation failed: %s", exc)
                self.host_kv_manager = None
                self.host_kv_aux_manager = None
            logger.info("[startup] Host KV cache allocated in %.2fs",
                        _time.monotonic() - kv_start)

        model_start = _time.monotonic()
        _diag(">>> _load_model_resources")
        self._load_model_resources()
        _diag("<<< _load_model_resources")
        logger.info("[startup] Model resources loaded in %.2fs",
                    _time.monotonic() - model_start)

        if self.args.enable_prefix_cache:
            self._initialize_prefix_cache_owner()

        spawn_start = _time.monotonic()
        _diag(">>> _spawn_workers")
        if self.distributed_weight_daemon is not None:
            _diag(">>> distributed_weight_daemon.wait_ready")
            self.distributed_weight_daemon.wait_ready(600.0)
            _diag("<<< distributed_weight_daemon.wait_ready")
        try:
            self._spawn_workers()
            _diag("<<< _spawn_workers")
            _diag(">>> _start_worker_monitor")
            self._start_worker_monitor()
            _diag("<<< _start_worker_monitor")
            _diag(">>> _wait_for_workers_ready")
            self._wait_for_workers_ready()
            _diag("<<< _wait_for_workers_ready")
        except Exception:
            self._shutdown_prefix_cache_owner()
            raise
        logger.info("[startup] Workers ready in %.2fs",
                    _time.monotonic() - spawn_start)

        self.started = True
        with self._lock:
            self._generation += 1
            self._state = "serving"
            self._first_ready = True
            self._reconciled_failure = False
        logger.info(
            "[startup] Total server startup: %.2fs (fast_init=%s, GPUs=%s)",
            _time.monotonic() - startup_start,
            self.args.fast_init,
            torch.cuda.device_count(),
        )

    def stop(self) -> None:
        if self._stopping:
            return
        has_partial_runtime = any(
            (
                self.started,
                self._runtime_dir_created,
                self._runtime_namespace_owned,
                self._runtime_locks is not None,
                self._lane_lease is not None,
                self._run_lock is not None,
                self.worker_process is not None,
                self.distributed_weight_daemon is not None,
                getattr(self, "_staged_weight_store", None) is not None,
                bool(self.model_info.get("shm_name")),
                bool(self.model_info.get("tensor_meta_shm_name")),
                self.skeleton_state_dict_file is not None,
            )
        )
        if not has_partial_runtime:
            return
        self._stopping = True
        worker_teardown_safe = self.worker_process is None
        artifacts_cleaned = False
        shutdown_start = time.monotonic()
        try:
            self._monitor_stop_event.set()
            logger.info("Stopping WorkerManager...")

            # Stop the monitor thread
            if self._monitor_thread is not None:
                self._monitor_thread.join(timeout=5)
                self._monitor_thread = None

            # Worker teardown closes the daemon control sockets. Mark the daemon as
            # stopping first so those expected disconnects are not recorded as a
            # transport failure; full daemon teardown still follows worker exit.
            if self.distributed_weight_daemon is not None:
                self.distributed_weight_daemon.prepare_stop()

            if self.worker_process is not None:
                try:
                    self._stop_workers()
                finally:
                    logging.getLogger("uvicorn.error").info(
                        "[shutdown] worker processes elapsed=%.3fs",
                        time.monotonic() - shutdown_start,
                    )
            worker_teardown_safe = True

            if self.distributed_weight_daemon is not None:
                self.distributed_weight_daemon.stop()
                self.distributed_weight_daemon = None
        finally:
            try:
                # Clean only resources owned by this immutable run. The model
                # regions are outside this run's SHM namespace, so they are
                # released through their C++ owner instead.
                if worker_teardown_safe:
                    # Drop the prefix-cache owner (an anonymous memfd) before
                    # the namespace sweep. Only a real owner has a runtime
                    # config; isolated test managers have neither.
                    if getattr(self, "prefix_cache_runtime_config", None) is not None:
                        self._shutdown_prefix_cache_owner()
                    if self._runtime_namespace_owned:
                        cleanup_shm_files(self.args.runtime_identity.shm_prefix)
                    if self.parameter_server_instance is not None:
                        model_release_start = time.monotonic()
                        self.parameter_server_instance = None
                        gc.collect()
                        logging.getLogger("uvicorn.error").info(
                            "[shutdown] model SHM owner release elapsed=%.3fs",
                            time.monotonic() - model_release_start,
                        )
                    # Both model regions are anonymous memfds in every mode, so
                    # dropping the owner above is the whole release: the labels
                    # name no object a supervisor could find afterwards.
                    self.model_info.pop("shm_name", None)
                    self.model_info.pop("tensor_meta_shm_name", None)

                    self._close_skeleton_memfd()
                    self._release_staged_weight_store()
                    runtime_dir = self.args.runtime_identity.runtime_dir
                    # Last holder of this run's liveness lock, and the lock file
                    # lives in the directory removed next.
                    release_run_lock(self._run_lock)
                    self._run_lock = None
                    if self._runtime_dir_created and runtime_dir.is_dir():
                        import shutil
                        shutil.rmtree(runtime_dir)
                    artifacts_cleaned = True
                else:
                    logger.error(
                        "Preserving runtime artifacts because owned worker "
                        "termination was not confirmed"
                    )
            finally:
                if artifacts_cleaned:
                    self._runtime_dir_created = False
                    self._runtime_namespace_owned = False
                    self.worker_process = None
                self.started = False
                self._stopping = False
                if artifacts_cleaned:
                    if self._lane_lease is not None:
                        self._lane_lease.close()
                        self._lane_lease = None
                    if self._runtime_locks is not None:
                        self._runtime_locks.close()
                        self._runtime_locks = None
                if artifacts_cleaned:
                    logger.info("WorkerManager stopped")
                else:
                    logger.error(
                        "WorkerManager stop incomplete; admission locks remain held"
                    )
                logging.getLogger("uvicorn.error").info(
                    "[shutdown] worker manager elapsed=%.3fs clean=%s",
                    time.monotonic() - shutdown_start,
                    artifacts_cleaned,
                )

    def _stop_workers(self) -> None:
        """Signal only the original child processes, never a reused numeric PID."""
        stop_start = time.monotonic()
        processes = self.worker_process.processes
        if not processes:
            raise RuntimeError("worker process context has no child processes")
        pidfds = []
        # Some Python builds (e.g. conda's) lack the pidfd APIs. The workers are
        # our own unreaped children, whose PIDs cannot be reused, and
        # Process.terminate()/kill() signal only a child not yet reaped, so
        # they are an equally safe fallback.
        use_pidfd = hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal")

        def send(proc, pidfd, sig):
            if pidfd is None:
                (proc.kill if sig == signal.SIGKILL else proc.terminate)()
                return
            try:
                signal.pidfd_send_signal(pidfd, sig)
            except ProcessLookupError:
                pass

        with self._join_lock:
            try:
                # An unjoined child cannot have its PID reused. Open handles
                # before joining; the handles remain bound to these children.
                for proc in processes:
                    if proc.exitcode is None:
                        pidfds.append(
                            (proc, os.pidfd_open(proc.pid) if use_pidfd else None)
                        )

                # Let an idle worker leave its main loop normally first.  This
                # gives its C++ storage objects a chance to release CUDA
                # registrations and mappings before the signal fallback uses
                # os._exit().  Workers blocked in NCCL or otherwise unable to
                # consume the sentinel still follow the existing pidfd path.
                #
                # The out-of-band flag goes first, before the sentinel: a rank
                # blocked in a collective consumes neither the sentinel (only
                # rank 0 reads the queue) nor the status broadcast (itself a
                # collective), so the flag is the only signal that reaches it.
                # Its watchdog gives the graceful path a short window and then
                # aborts this worker's communicators, which frees the blocked
                # collective so the rank can leave its loop and run the same
                # ordered teardown.  The grace and signal fallback below are
                # unchanged and remain the safety net.
                #
                # This is a plain store into lock-free shared memory, so it
                # cannot block.  That matters because this runs under
                # _teardown_lock: anything that could block here would strand
                # the teardown owner, which is exactly how the Event-based
                # version of this signal failed.
                if self._cohort_shutdown_flag is not None:
                    self._cohort_shutdown_flag.value = 1
                sentinel_sent = False
                try:
                    self.request_queue.put(None)
                    sentinel_sent = True
                except Exception:
                    logger.warning("Failed to signal worker shutdown", exc_info=True)

                deadline = time.monotonic() + (
                    _WORKER_SENTINEL_GRACE_S if sentinel_sent else 0.0
                )
                for proc in processes:
                    proc.join(timeout=max(0, deadline - time.monotonic()))

                remaining = [
                    (proc, fd) for proc, fd in pidfds if proc.exitcode is None
                ]
                # Any worker that needs a signal did not run its ordered
                # teardown, so this shutdown is abnormal and what it leaves
                # behind has to be named in the log.
                signal_fallback_used = bool(remaining)
                # Name the exit path per worker here rather than in the worker.
                # A force-killed worker is stopped while a native collective
                # owns its thread, so it never runs Python again and can never
                # log its own marker; this process is the only one that knows a
                # signal replaced an ordered exit, and for which rank.
                positions = {id(proc): index for index, proc in enumerate(processes)}
                for proc, _ in remaining:
                    logging.getLogger("uvicorn.error").info(
                        "[shutdown] worker exit path=%s rank-idx=%s pid=%s",
                        _WORKER_EXIT_PATH_SIGNAL,
                        positions.get(id(proc), "unknown"),
                        proc.pid,
                    )
                for proc, pidfd in remaining:
                    send(proc, pidfd, signal.SIGTERM)

                deadline = time.monotonic() + 5
                for proc, _ in remaining:
                    proc.join(timeout=max(0, deadline - time.monotonic()))

                # A child may finish during the SIGTERM grace period. Reap
                # only handles that are still live; sending SIGKILL to a
                # child that already exited obscures the shutdown result and
                # breaks fake-pidfd tests.
                remaining = [
                    (proc, fd) for proc, fd in remaining if proc.exitcode is None
                ]

                logging.getLogger("uvicorn.error").info(
                    "[shutdown] worker SIGTERM grace elapsed=%.3fs remaining=%d",
                    time.monotonic() - stop_start,
                    len(remaining),
                )
                if remaining:
                    logger.warning("Workers did not exit gracefully, force-killing...")
                for proc, pidfd in remaining:
                    send(proc, pidfd, signal.SIGKILL)
                deadline = time.monotonic() + _WORKER_KILL_REAP_TIMEOUT_S
                next_progress_log = time.monotonic() + 30
                while remaining and time.monotonic() < deadline:
                    for proc, _ in remaining:
                        proc.join(timeout=1)
                    remaining = [
                        (proc, fd) for proc, fd in remaining if proc.exitcode is None
                    ]
                    if remaining and time.monotonic() >= next_progress_log:
                        next_progress_log = time.monotonic() + 30
                        logging.getLogger("uvicorn.error").info(
                            "[shutdown] waiting for SIGKILL'd workers to be "
                            "reaped (page unpin in progress) elapsed=%.3fs "
                            "remaining=%d",
                            time.monotonic() - stop_start,
                            len(remaining),
                        )
                logging.getLogger("uvicorn.error").info(
                    "[shutdown] worker SIGKILL/reap elapsed=%.3fs remaining=%d",
                    time.monotonic() - stop_start,
                    sum(proc.exitcode is None for proc in processes),
                )

                if signal_fallback_used:
                    identity = self.args.runtime_identity
                    # The runtime dir is still ours here; stop() removes it once
                    # teardown is confirmed, so only name it as a leftover when
                    # something else of this run also survives.
                    leftovers = [
                        name
                        for name in list_run_named_objects(
                            identity.shm_prefix, identity.runtime_dir
                        )
                        if name != str(identity.runtime_dir)
                    ]
                    if leftovers:
                        remains = "Left behind for this run: " + ", ".join(leftovers)
                    else:
                        remains = (
                            "Nothing named remains for this run beyond its runtime "
                            "directory, which stop() removes; the model and KV "
                            "regions are anonymous memfds the kernel reclaims."
                        )
                    logger.warning(
                        "Workers exited abnormally: the SIGTERM/SIGKILL fallback "
                        "replaced their ordered teardown. %s",
                        remains,
                    )

                surviving_pids = [
                    proc.pid for proc in processes if proc.exitcode is None
                ]
                if surviving_pids:
                    raise RuntimeError(
                        "worker teardown left live owned PIDs; preserving runtime "
                        f"artifacts for investigation: {surviving_pids}"
                    )
            finally:
                for _, pidfd in pidfds:
                    if pidfd is not None:
                        os.close(pidfd)

    def get_worker_exit_state(self) -> WorkerExitState:
        return self._worker_exit_state

    def report_worker_fatal(self, reason: str, *, persisted: bool = False) -> None:
        """Enter workers_down after the existing pool_shutdown fatal path."""
        self._fatal_ack_event.set()
        if persisted:
            self._reconciled_failure = True
        self._handle_worker_failure(reason, None)

    def _set_state(self, expected: str, new: str) -> None:
        with self._lock:
            if self._state != expected:
                raise RuntimeError(
                    f"invalid worker state transition {self._state!r} -> {new!r}; "
                    f"expected {expected!r}"
                )
            self._state = new

    def _call_spawn_on_main_thread(self, fn: Callable[[], Any], timeout_s: float) -> Any:
        if threading.current_thread() is threading.main_thread():
            return fn()
        if self._main_thread_dispatcher is None:
            raise RuntimeError("worker respawn requires a main-thread dispatcher")
        return self._main_thread_dispatcher(fn, timeout_s)

    def _create_worker_channels(self) -> None:
        """Bind a fresh request/response queue pair for the next cohort.

        The channels are cohort-scoped: every worker generation gets its own
        pair, so nothing a previous cohort wrote is ever read again.

        The cohort's shutdown flag is bound here for the same reason.  It is
        the out-of-band teardown signal: the queue sentinel below reaches only
        rank 0, and the status tensor the other ranks learn shutdown from is
        itself a collective, so a rank blocked in NCCL is reachable by
        neither.  The flag reaches every local rank directly, and rotating it
        with the cohort keeps a new generation from ever starting up already
        holding the previous teardown's flag.

        It is deliberately a ``RawValue``, i.e. shared memory with NO lock and
        NO wake handshake.  ``multiprocessing.Event`` cannot be used here: its
        ``set()`` goes through ``Condition.notify``, which blocks on
        ``_woken_count`` while any registered waiter is a dead process, so the
        teardown owner itself would deadlock on the very failure this signal
        exists for.  Setting this flag is a plain store that cannot block, and
        the worker side polls it.
        """
        self.request_queue = self._mp_ctx.Queue()
        self.response_queue = self._mp_ctx.Queue()
        self._cohort_shutdown_flag = self._mp_ctx.RawValue("b", 0)

    def _abandon_worker_channels(self) -> None:
        """Release the previous cohort's queues WITHOUT reading them.

        A worker killed mid-write leaves a frame whose length header is
        readable but whose payload never arrives.  Any later read of that
        queue -- including a nominally non-blocking ``get_nowait()``, which
        polls the pipe and only then reads it -- therefore blocks forever
        inside ``recv_bytes()``, with no timeout and no exception.  Teardown
        force-kills the cohort, so a partial frame is an expected outcome and
        the only safe handling is to abandon the channel unread.

        The feeder thread may already be broken, so both steps are advisory:
        ``cancel_join_thread()`` keeps a half-written buffer from blocking
        this process, and ``close()`` releases the descriptors.
        """
        for label, channel in (
            ("request", self.request_queue),
            ("response", self.response_queue),
        ):
            if channel is None:
                continue
            for step in ("cancel_join_thread", "close"):
                action = getattr(channel, step, None)
                if action is None:
                    continue
                try:
                    action()
                except Exception:
                    logger.warning(
                        "Abandoning the %s channel: %s failed",
                        label,
                        step,
                        exc_info=True,
                    )
        self.request_queue = None
        self.response_queue = None
        # The flag belongs to the cohort that was just torn down; the next one
        # is created with its channels.
        self._cohort_shutdown_flag = None

    def _release_host_kv_sequences(self) -> None:
        for manager in (self.host_kv_manager, self.host_kv_aux_manager):
            if manager is None:
                continue
            release_all = getattr(manager, "release_all_sequences", None)
            if release_all is None:
                raise RuntimeError("host-KV manager lacks release_all_sequences")
            release_all()

    def _reconcile(self, reason: str, *, fail_in_flight: bool) -> Dict[str, Any]:
        failed_batches = 0
        if fail_in_flight and not self._reconciled_failure:
            if self._fail_in_flight_hook is not None:
                failed_batches = int(self._fail_in_flight_hook(reason) or 0)
            self._reconciled_failure = True
        self._release_host_kv_sequences()
        # Reconciliation never reads the worker queues.  The cohort it cleans
        # up after was force-killed, so a read could block forever on a
        # truncated frame; the channels are abandoned unread instead, see
        # _abandon_worker_channels.
        # The run prefix covers named QueryBook and token-store objects.  Host
        # KV, weights, tensor metadata, and skeleton state are server-owned
        # memfds and therefore intentionally remain open across this cleanup.
        cleanup_shm_files(self.args.runtime_identity.shm_prefix)
        return {"failed_batches": failed_batches}

    def _abort_restart_if_stopping(self) -> None:
        """Raise when server shutdown has begun or already finished.

        ``stop()`` sets ``_stopping`` first and clears it again when it
        returns, while ``started`` stays False from then on; checking both
        covers a restart phase that outlasted the whole of ``stop()``.
        """
        if self._stopping or not self.started:
            raise RuntimeError("worker restart aborted: server is stopping")

    def restart_workers(self, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        """Replace one worker cohort while retaining server-owned resources.

        The restart runs on a worker thread (``asyncio.to_thread``), which
        neither request cancellation nor server shutdown can interrupt.  It
        therefore checks for shutdown at its phase boundaries -- on entry,
        after teardown, and immediately before the respawn -- and aborts to
        ``workers_down`` there, so it never spawns a cohort once ``stop()``
        has begun.  Residual window: a phase already in flight when shutdown
        begins (a teardown, or a spawn already dispatched) runs to completion
        before the next boundary sees the shutdown; ``stop()`` tears down
        whatever cohort is live when it reaches the worker processes.
        """
        timeout_s = self._restart_timeout_s if timeout_s is None else float(timeout_s)
        started_at = time.monotonic()
        with self._lock:
            if self._state not in {"serving", "workers_down"}:
                raise RuntimeError(f"worker restart conflict: state={self._state}")
            self._state = "restarting"

        # Every phase runs inside one recovery scope: a restart that fails at
        # ANY point leaves the manager in the retryable workers_down state,
        # never stuck in "restarting".
        teardown_owned = False
        try:
            self._abort_restart_if_stopping()
            quiesce_start = time.monotonic()
            if self._quiesce_hook is not None:
                self._quiesce_hook()
            quiesce_s = time.monotonic() - quiesce_start
            teardown_start = time.monotonic()
            # Teardown of a cohort has one owner at a time: a monitor that
            # claimed a failure holds this lock through its reconcile, and a
            # restart from workers_down waits for it here rather than tearing
            # down beside it.  Held through the channel rotation below.
            self._teardown_lock.acquire()
            teardown_owned = True
            self._monitor_stop_event.set()
            monitor = self._monitor_thread
            if monitor is not None and monitor is not threading.current_thread():
                monitor.join(timeout=5)
                self._monitor_thread = None
            # The teardown below closes every worker's control socket to the
            # distributed-weight daemon; open the cohort swap first so those
            # disconnects are expected rather than a failure.
            self._begin_distributed_weight_swap()
            if self.worker_process is not None:
                self._stop_workers()
                self.worker_process = None
            teardown_s = time.monotonic() - teardown_start
            self._abort_restart_if_stopping()

            reconcile_start = time.monotonic()
            # In-flight batches are persisted as failed exactly once:
            # _reconciled_failure stays True from the fatal path until a
            # restart succeeds, while a restart that failed before reaching
            # reconcile leaves it False so the retry still persists them.
            reconciliation = self._reconcile(
                "Worker pool restarted; in-flight batches were failed.",
                fail_in_flight=True,
            )
            reconcile_s = time.monotonic() - reconcile_start

            # The old cohort is reaped and reconciled: release what it held in
            # the daemon and admit the next cohort's epoch. A transport failure
            # raises here, before anything is spawned.
            weight_swap_start = time.monotonic()
            weight_swap = self._finish_distributed_weight_swap()
            weight_swap_s = time.monotonic() - weight_swap_start

            # The cohort that owned the previous channels has been killed, so
            # those queues can never be read again -- a partial frame left by
            # a killed writer makes any read block forever.  Abandon them
            # unread and give the next cohort a fresh pair.  Everything that
            # reaches a worker from here on (the init message below and the
            # queues _spawn_workers hands the children) reads these
            # attributes at call time, so it picks up the new pair.
            # Tell the scheduler to retire its reader first: a pump still
            # blocked on the old response queue must see its stop event
            # before close() makes that read fail, so the failure is an
            # expected retirement rather than a reported channel error.
            if self._retire_channels_hook is not None:
                self._retire_channels_hook()
            self._abandon_worker_channels()
            self._create_worker_channels()
            teardown_owned = False
            self._teardown_lock.release()

            spawn_start = time.monotonic()
            self._monitor_stop_event.clear()
            self._ready_event.clear()
            self._fatal_ack_event.clear()
            # The previous cohort's fatal is fully handled by this point; a
            # stale failure reason must not abort the readiness wait of the
            # cohort spawned below.
            self._worker_exit_state.clear()
            self._abort_restart_if_stopping()
            self._call_spawn_on_main_thread(self._spawn_workers, timeout_s)
            self._start_worker_monitor()
            self._wait_for_workers_ready(timeout_s=timeout_s)
            # Pool capacity params live with the scheduler (it accumulated them
            # across batch submissions); re-init with the same values, never
            # with hardcoded defaults.
            if self._pool_init_params_hook is None:
                raise RuntimeError("worker restart requires a pool_init_params hook")
            init_params = self._pool_init_params_hook()
            self.request_queue.put({
                "type": "init",
                "max_output_len": init_params.get("max_output_len", 4096),
                "max_context_length": init_params.get("max_context_length"),
            })
            spawn_to_ready_s = time.monotonic() - spawn_start
            # The scheduler side of the new generation (completion pump,
            # listener re-arm, then resume) is part of the restart: neither
            # hook needs the serving state, and if either fails the restart
            # failed, so it must end in workers_down rather than in a serving
            # state with a quiesced scheduler or no completion consumer.
            if self._scheduler_generation_hook is not None:
                self._scheduler_generation_hook()
            if self._resume_hook is not None:
                self._resume_hook()
        except BaseException:
            if teardown_owned:
                self._teardown_lock.release()
            with self._lock:
                self._state = "workers_down"
            raise
        with self._lock:
            self._generation += 1
            self._state = "serving"
            self._first_ready = True
            self._worker_exit_state.clear()
            self._reconciled_failure = False
        return {
            "generation": self.generation,
            "failed_batches": reconciliation["failed_batches"],
            "elapsed_s": time.monotonic() - started_at,
            "phase_timings": {
                "quiesce_s": quiesce_s,
                "teardown_s": teardown_s,
                "reconcile_s": reconcile_s,
                "weight_swap_s": weight_swap_s,
                "spawn_to_ready_s": spawn_to_ready_s,
            },
            "distributed_weight_swap": weight_swap,
        }

    # ---------------------- Startup helpers ----------------------
    def _compact_memory(self) -> None:
        """Drop page cache and compact memory for stable THP allocation."""
        import subprocess
        import time as _time
        t0 = _time.monotonic()
        try:
            subprocess.run(["sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], check=True)
            subprocess.run(["sh", "-c", "echo 1 > /proc/sys/vm/compact_memory"], check=True)
            logger.info("[fast-init] Memory compaction completed in %.2fs (drop_caches + compact_memory)",
                        _time.monotonic() - t0)
        except (subprocess.CalledProcessError, PermissionError) as e:
            logger.warning("[fast-init] Memory compaction failed (requires root): %s", e)

    def _config_hugepages(self, byte_size: int = None) -> None:
        """Reserve the host huge page pool the weights memfd draws from.

        MFD_HUGETLB needs no hugetlbfs mount, only the reservation, so this
        configures the pool size and nothing else.

        Args:
            byte_size: Model size in bytes. If None, uses default 700GB.
        """
        hugepage_size = get_hugepage_size()

        if byte_size is not None:
            # Model byte_size already includes buffer
            num_hugepages = (byte_size + hugepage_size - 1) // hugepage_size
        else:
            # Fallback to old default (for backwards compatibility)
            num_hugepages = 350000

        logger.info(
            f"Configuring hugepages: {num_hugepages} pages "
            f"({num_hugepages * hugepage_size / (1024**3):.1f} GB, "
            f"{hugepage_size / (1024**2):.0f} MB pages)"
        )

        cmd = ["sysctl", "-w", f"vm.nr_hugepages={num_hugepages}"]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except Exception as exc:
            logger.warning(
                "Hugepages configuration failed (%s): %s", cmd, exc
            )

    def _load_model_resources(self) -> None:
        import sys as _diag_sys, time as _diag_time
        def _diag(msg):
            print(f"[DIAG {_diag_time.time():.3f}] {msg}", flush=True)
            _diag_sys.stdout.flush()

        _diag(f"  _load_model_resources entered, model={self.args.model}")
        logger.info("Loading model resources for %s", self.args.model)
        endpoint = os.getenv(PARAMETER_SERVER_ENDPOINT_ENV)
        hf_cache_dir = self.args.hf_cache_dir or Path(
            os.path.expanduser("~/.cache/huggingface")
        )
        self.args.hf_cache_dir = hf_cache_dir
        converted_ckpt_dir = (
            self.args.converted_ckpt_dir
            or Path(self.args.cache_dir or ".") / "converted_ckpt"
        )
        self.args.converted_ckpt_dir = converted_ckpt_dir
        _diag(f"  paths resolved: endpoint={endpoint!r}, cache_dir={self.args.cache_dir!r}")

        if not endpoint and self.args.cache_dir is None:
            if os.path.isdir(self.args.model):
                # --model is a local checkpoint directory: use it directly as the
                # cache dir. Avoids treating a filesystem path as an HF repo id
                # (which fails HFValidationError) and skips any network download.
                _diag(f"  local model dir -> cache_dir={self.args.model}")
                self.args.cache_dir = Path(self.args.model)
            else:
                _diag("  >>> _download_model_snapshot")
                self.args.cache_dir = self._download_model_snapshot(hf_cache_dir)
                _diag("  <<< _download_model_snapshot")

        if endpoint:
            _diag("  >>> _load_model_from_remote_server")
            self._load_model_from_remote_server(
                endpoint, hf_cache_dir, converted_ckpt_dir
            )
            _diag("  <<< _load_model_from_remote_server")
        elif self.args.distributed_weight_config is not None:
            _diag("  >>> _load_model_from_distributed_store")
            self._load_model_from_distributed_store()
            _diag("  <<< _load_model_from_distributed_store")
        else:
            _diag("  >>> _load_model_locally")
            self._load_model_locally(hf_cache_dir, converted_ckpt_dir)
            _diag("  <<< _load_model_locally")

        _diag("  >>> _configure_host_kv_cache_budget")
        self._configure_host_kv_cache_budget()
        _diag("  <<< _configure_host_kv_cache_budget")
        logger.info("Model Loaded. SHM: %s", self.model_info.get("shm_name"))

    def _spawn_workers(self) -> None:
        local_device_count = torch.cuda.device_count()
        if local_device_count == 0:
            raise RuntimeError("No CUDA devices found.")

        # Respect user-specified world_size (following vLLM/SGLang best practice)
        # Users should use CUDA_VISIBLE_DEVICES to limit visible GPUs
        world_size = self.args.world_size
        if world_size is None or world_size <= 0:
            # Auto-detect only if not explicitly specified
            world_size = local_device_count * self.args.nnodes

        local_world_size = self._local_world_size
        if local_world_size is None:
            raise RuntimeError("runtime topology was not validated before startup")

        logger.info(
            "Spawning %d DDP workers (world_size=%d, nnodes=%d)",
            local_world_size, world_size, self.args.nnodes
        )

        # Auto-detect GPU architecture if not specified
        gpu_arch = self.args.gpu_arch or detect_gpu_arch()

        args = BatchGenWorkerArgs(
            model_name=self.args.model,
            hf_cache_dir=self.args.hf_cache_dir,
            cache_dir=self.args.cache_dir,
            converted_ckpt_dir=self.args.converted_ckpt_dir,
            kv_dtype=self.args.kv_dtype,
            dist_init_addr=self.args.dist_init_addr,
            world_size=world_size,
            local_world_size=local_world_size,
            nnode_rank=self.args.node_rank,
            nnodes=self.args.nnodes,
            gpu_arch=gpu_arch,
            shm_name=self.model_info["shm_name"],
            tensor_meta_shm_name=self.model_info["tensor_meta_shm_name"],
            enable_hugetlbfs=self.args.enable_hugetlbfs,
            weight_byte_size=self.model_info["parameter_server_size"],
            host_kv_cache_size=self.args_dict.get(
                "host_kv_cache_size_per_rank"
            ),
            global_host_kv_cache_size_gb=self.args.host_kv_cache_size,
            enable_prefix_cache=self.args.enable_prefix_cache,
            prefix_cache_debug_stats=self.args.prefix_cache_debug_stats,
            skeleton_state_dict_file=self.skeleton_state_dict_file,
            # placeholders
            local_rank=-1,
            global_rank=-1,
            device=-1,
            watchdog_timeout=self.args.watchdog_timeout,
            watchdog_test_stuck_time=self.args.watchdog_test_stuck_time,
            watchdog_heartbeat_interval=self.args.watchdog_heartbeat_interval,
            decode_step_timeout=self.args.decode_step_timeout,
            enable_prepack=self.args.enable_prepack,
            host_kv_watermark=self.args.host_kv_watermark,
            enable_decode_preemption=self.args.enable_decode_preemption,
            gpu_memory_frac=self.args.gpu_memory_frac,
            initial_gpu_page_buffer=self.args.initial_gpu_page_buffer,
            extension_gpu_page_buffer=self.args.extension_gpu_page_buffer,
            decision_frequency_pages=self.args.decision_frequency_pages,
            enable_ep_with_offloading=self.args.enable_ep_with_offloading,
            ep_offloading_ratio=self.args.ep_offloading_ratio,
            pre_dequantize_weights=self.args.pre_dequantize_weights,
            enable_deepep=getattr(self.args, "enable_deepep", False),
            enable_cuda_graph=self.args.enable_cuda_graph,
            disable_cuda_graphs=self.args.disable_cuda_graphs,
            cuda_graph_max_bucket_size=self.args.cuda_graph_max_bucket_size,
            cuda_graph_num_buckets=self.args.cuda_graph_num_buckets,
            persistent_phase_instances=self.args.persistent_phase_instances,
            detokenization_include_special_tokens=self.args.detokenization_include_special_tokens,
            host_kv_chunk_size=self.args.host_kv_chunk_size,
            enable_host_kv_eviction=self.args.enable_host_kv_eviction,
            host_kv_eviction_watermark=self.args.host_kv_eviction_watermark,
            adaptive_chunk=self.args.adaptive_chunk,
            adaptive_chunk_min=self.args.adaptive_chunk_min,
            adaptive_chunk_max=self.args.adaptive_chunk_max,
            adaptive_chunk_ema_alpha=self.args.adaptive_chunk_ema_alpha,
            adaptive_chunk_multiplier=self.args.adaptive_chunk_multiplier,
            fast_init=self.args.fast_init,
            max_pool_size=self.args.max_pool_size,
            input_ids_pool_size_gb=self.args.input_ids_pool_size_gb,
            prefix_memfd_pid=self._get_prefix_memfd_pid(),
            prefix_memfd_fd=self._get_prefix_memfd_fd(),
            kv_memfd_pid=self._get_kv_memfd_pid(),
            kv_memfd_fd=self._get_kv_memfd_fd(),
            kv_aux_memfd_fd=self._get_kv_aux_memfd_fd(),
            weights_memfd_pid=self._get_weights_memfd_pid(),
            weights_memfd_fd=self._get_weights_memfd_fd(),
            tensor_meta_memfd_fd=self._get_tensor_meta_memfd_fd(),
            distributed_weight_config=(
                str(self._distributed_weight_config_path)
                if self.args.distributed_weight_config is not None
                else None
            ),
            distributed_weight_cohort=self._distributed_weight_cohort(),
            host_kv_shm_name=self.args.runtime_identity.host_kv_shm_name,
            host_kv_aux_shm_name=self.args.runtime_identity.host_kv_aux_shm_name,
            prefix_cache_shm_name=self.args.runtime_identity.prefix_cache_shm_name,
            query_book_shm_prefix=(
                self.args.runtime_identity.query_book_shm_prefix
            ),
            runtime_dir=str(self.args.runtime_identity.runtime_dir),
            pynccl_port_base=self.args.pynccl_port_base,
            pynccl_port_span=self.args.pynccl_port_span,
        )
        from batchgen.server_worker_main_loop import server_worker_main
        # Each worker arms PR_SET_PDEATHSIG, which the kernel delivers when the
        # THREAD that created it exits -- not when this process exits.  Spawning
        # from a helper thread would therefore kill every worker as soon as that
        # thread finished.
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError(
                "GPU workers must be spawned from the main thread: the worker "
                "parent-death signal is tied to the spawning thread, so workers "
                f"spawned from {threading.current_thread().name!r} would be "
                "killed when that thread exits."
            )
        self.worker_process = mp.spawn(
            server_worker_main,
            args=(
                self.request_queue,
                self.response_queue,
                args,
                self._ready_event,
                self._fatal_ack_event,
                self._cohort_shutdown_flag,
            ),
            nprocs=local_world_size,  # Use world_size-derived count, not device_count
            join=False,
            daemon=True,
        )

    # The prefix-cache coordinator region is an anonymous memfd owned by this
    # process, published to the workers like the host-KV regions below.
    def _get_prefix_memfd_pid(self) -> int:
        if getattr(self, 'prefix_cache_coordinator', None) is not None:
            return os.getpid()
        return -1

    def _get_prefix_memfd_fd(self) -> int:
        if getattr(self, 'prefix_cache_coordinator', None) is not None:
            return self.prefix_cache_coordinator.memfd_fd()
        return -1

    # Host KV regions are always anonymous memfds owned by this process, so the
    # creator pid and fds are published to the workers in every mode.
    def _get_kv_memfd_pid(self) -> int:
        if getattr(self, 'host_kv_manager', None) is not None:
            return os.getpid()
        return -1

    def _get_kv_memfd_fd(self) -> int:
        if getattr(self, 'host_kv_manager', None) is not None:
            return self.host_kv_manager.memfd_fd()
        return -1

    def _get_kv_aux_memfd_fd(self) -> int:
        if getattr(self, 'host_kv_aux_manager', None) is not None:
            return self.host_kv_aux_manager.memfd_fd()
        return -1

    # The local parameter server owns both the weights and the tensor-metadata
    # memfd, so its pid and fds are published to the workers in every mode,
    # hugetlbfs included: that mode only changes the memfd's page size.
    def _get_weights_memfd_pid(self) -> int:
        ps = getattr(self, 'parameter_server_instance', None)
        if ps is None:
            return -1
        if (
            ps.parameter_server.weights_memfd_fd() >= 0
            or ps.parameter_server.tensor_meta_memfd_fd() >= 0
        ):
            return os.getpid()
        return -1

    def _get_weights_memfd_fd(self) -> int:
        ps = getattr(self, 'parameter_server_instance', None)
        if ps is not None:
            return ps.parameter_server.weights_memfd_fd()
        return -1

    def _get_tensor_meta_memfd_fd(self) -> int:
        ps = getattr(self, 'parameter_server_instance', None)
        if ps is not None:
            return ps.parameter_server.tensor_meta_memfd_fd()
        return -1

    def _wait_for_workers_ready(self, timeout_s: Optional[float] = None) -> None:
        if self.worker_process is None:
            return
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while not self._ready_event.wait(timeout=self._monitor_interval_s):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("worker cohort did not become ready before restart timeout")
            if self._worker_exit_state.is_failed():
                reason = (
                    self._worker_exit_state.reason
                    or "Worker failed during startup"
                )
                raise RuntimeError(reason)
            exit_reason = self._collect_worker_exit_reason()
            if exit_reason:
                self._worker_exit_state.set_failure(exit_reason)
                raise RuntimeError(exit_reason)

    def _start_worker_monitor(self) -> None:
        if self.worker_process is None:
            return
        self._monitor_thread = threading.Thread(
            target=self._monitor_worker_processes,
            name="worker-monitor",
            daemon=True,
        )
        self._monitor_thread.start()

    def _monitor_worker_processes(self) -> None:
        # A failure is handed to _handle_worker_failure on every poll until
        # this monitor is stopped or the cohort is gone, not just once: the
        # handler declines while a restart owns the pool, and a cohort that
        # dies just before that restart flips to serving must still be caught
        # on the first poll after the flip.  Each reason is logged once.
        reported = None
        while not self._monitor_stop_event.is_set():
            if self.worker_process is None:
                return
            try:
                with self._join_lock:
                    for proc in self.worker_process.processes:
                        proc.join(timeout=0)
            except Exception as exc:
                if self._stopping:
                    return
                reason = f"Worker process crashed: {exc}"
                if reason != reported:
                    logger.exception("Worker process join failed")
                    reported = reason
                self._handle_worker_failure(reason, exc)
            else:
                if self._monitor_stop_event.is_set():
                    return
                exit_reason = self._collect_worker_exit_reason()
                if exit_reason:
                    if self._stopping:
                        return
                    if exit_reason != reported:
                        logger.error(exit_reason)
                        reported = exit_reason
                    self._handle_worker_failure(exit_reason, None)
            self._monitor_stop_event.wait(self._monitor_interval_s)

    def _collect_worker_exit_reason(self) -> Optional[str]:
        if self.worker_process is None:
            return None
        processes = getattr(self.worker_process, "processes", None)
        if not processes:
            return None
        exited = []
        error_files = getattr(self.worker_process, "error_files", ())
        for idx, proc in enumerate(processes):
            if proc.exitcode is None:
                continue
            detail = f"idx={idx} pid={proc.pid} exitcode={proc.exitcode}"
            if (
                idx < len(error_files)
                and error_files[idx]
                and os.path.isfile(error_files[idx])
            ):
                with open(error_files[idx], "rb") as error_file:
                    detail += " Python traceback:\n" + pickle.load(error_file)
            exited.append(detail)
        if not exited:
            return None
        return "Detected worker process exit: " + ", ".join(exited)

    def _handle_worker_failure(
        self, reason: str, exc: Optional[BaseException]
    ) -> None:
        if self._stopping:
            return
        # Teardown and reconciliation of a cohort have exactly one owner.  The
        # ownership check and the transition that claims it share one critical
        # section, so an explicit restart cannot take the cohort between them:
        # a restart that already set "restarting" owns the teardown (its own
        # reap and reconcile cover the dead cohort), and a cohort already in
        # "workers_down" is owned by whoever entered that state or by the
        # next restart.  The claim also takes the teardown lock, held through
        # the end of this reconcile: a restart requested from workers_down
        # waits on it instead of racing this teardown and its SHM cleanup.
        # Only a restart (state "restarting") or a claimed failure (state
        # "workers_down") ever holds that lock, so it is free whenever the
        # state is "serving".
        with self._lock:
            if self._state != "serving":
                return
            startup = not self._first_ready
            if not startup:
                if not self._teardown_lock.acquire(blocking=False):
                    logger.error(
                        "Worker failure not claimed: the cohort teardown is "
                        "already owned while the pool is serving (%s)",
                        reason,
                    )
                    return
                if not self._worker_exit_state.set_failure(reason, exc):
                    self._teardown_lock.release()
                    return
                self._state = "workers_down"
        if startup:
            if self._worker_exit_state.set_failure(reason, exc):
                logging.getLogger("uvicorn.error").error(
                    "[shutdown] worker failure during startup epoch=%.6f: %s",
                    time.time(),
                    reason,
                )
                self._request_server_shutdown()
            return
        logging.getLogger("uvicorn.error").error(
            "Worker cohort failed after readiness; entering workers_down: %s",
            reason,
        )
        try:
            if self._quiesce_hook is not None:
                self._quiesce_hook()
            self._monitor_stop_event.set()
            monitor = self._monitor_thread
            if monitor is not None and monitor is not threading.current_thread():
                monitor.join(timeout=5)
                self._monitor_thread = None
            # As in restart_workers: the surviving workers' sockets close in
            # the teardown, which is part of this cohort's swap. Failing to
            # open it must not skip the teardown and reconcile below; those
            # disconnects then only mark this already-failed cohort failed.
            try:
                self._begin_distributed_weight_swap()
            except Exception:
                logger.exception(
                    "Could not begin the distributed weight cohort swap"
                )
            try:
                if self.worker_process is not None:
                    self._stop_workers()
                    self.worker_process = None
                self._reconcile(reason, fail_in_flight=True)
            except Exception:
                logger.exception("Worker failure reconciliation failed")
            else:
                # The dead cohort is reaped: clear the latch its failure set in
                # the daemon now, so the next restart starts from a clean
                # cohort. If this fails the swap stays open and that restart
                # retries it.
                try:
                    self._finish_distributed_weight_swap()
                except Exception:
                    logger.exception(
                        "Distributed weight cohort swap after the worker "
                        "failure did not complete; the next restart retries it"
                    )
        finally:
            self._teardown_lock.release()

    @staticmethod
    def _request_server_shutdown() -> None:
        try:
            os.kill(os.getpid(), signal.SIGINT)
        except Exception:
            logger.error("Failed to signal server shutdown", exc_info=True)

    # ---------------------- Distributed-weight cohort swap ----------------------
    # The daemon outlives worker cohorts. Every path that tears a cohort down
    # while the server keeps running (restart_workers and the monitor's
    # failure path) brackets that teardown with a swap: begin before the
    # workers' sockets close, finish once they are reaped. stop() instead
    # tears the daemon down with the cohort (prepare_stop, then stop).
    def _distributed_weight_cohort(self) -> int:
        """The cohort epoch the next spawned workers must name in their hello."""
        if self.distributed_weight_daemon is None:
            return 0
        return int(self.distributed_weight_daemon.cohort_epoch())

    def _begin_distributed_weight_swap(self) -> None:
        if self.distributed_weight_daemon is None:
            return
        epoch = int(self.distributed_weight_daemon.begin_cohort_swap())
        logging.getLogger("uvicorn.error").info(
            "[restart] distributed weight cohort swap begun: cohort=%d", epoch
        )

    def _finish_distributed_weight_swap(self) -> Optional[Dict[str, Any]]:
        if self.distributed_weight_daemon is None:
            return None
        report = json.loads(
            self.distributed_weight_daemon.finish_cohort_swap(
                self._DISTRIBUTED_WEIGHT_SWAP_TIMEOUT_S
            )
        )
        logging.getLogger("uvicorn.error").info(
            "[restart] distributed weight cohort swap finished: %s",
            json.dumps(report, sort_keys=True),
        )
        return report

    # ---------------------- Model loading helpers ----------------------
    def _stage_distributed_weight_store(self) -> None:
        """Map a server-owned tmpfs copy of the store when the config sets
        ``store_residency``; a legacy config maps ``store_path`` directly.

        Runs before the daemon starts, so the daemon and every worker of
        every cohort receive the same effective config. Released by
        :meth:`_release_staged_weight_store` once both are gone; a worker
        restart never touches it.
        """
        from batchgen.server.weight_store_staging import (
            stage_distributed_weight_store,
        )

        staged = stage_distributed_weight_store(
            self.args.distributed_weight_config,
            self.args.runtime_identity.runtime_dir,
            fast_init=self.args.fast_init,
        )
        self._staged_weight_store = staged
        self._distributed_weight_config_path = (
            staged.effective_config_path
            if staged is not None
            else Path(self.args.distributed_weight_config)
        )

    def _release_staged_weight_store(self) -> None:
        """Drop this server's hold on the staged store (removing a ``run``
        copy). Only after the daemon is stopped and the workers are reaped:
        until then something still maps it, so an incomplete stop keeps the
        hold until this process exits."""
        staged = getattr(self, "_staged_weight_store", None)
        if staged is None or self.distributed_weight_daemon is not None:
            return
        self._staged_weight_store = None
        try:
            staged.release()
        except Exception:
            logger.exception(
                "Releasing the staged distributed weight store %s failed",
                staged.directory,
            )

    def _start_distributed_weight_daemon(self) -> None:
        if "kimi-k3" not in self.args.model.lower():
            raise ValueError(
                "--distributed-weight-config currently supports Kimi-K3 only"
            )
        from batchgen.models.moonshotai.kimi_linear.distributed_weight_store import (
            load_distributed_weight_config,
        )

        config_path = self._distributed_weight_config_path
        config = load_distributed_weight_config(config_path)
        if int(config["num_nodes"]) != int(self.args.nnodes):
            raise ValueError(
                "distributed weight config lists "
                f"{config['num_nodes']} node IPs, which does not match "
                f"--nnodes {self.args.nnodes}"
            )
        if int(config["node_rank"]) != int(self.args.node_rank):
            raise ValueError(
                "distributed weight config node_rank does not match "
                f"--node-rank: {config['node_rank']} != {self.args.node_rank}"
            )
        self.distributed_weight_config = config
        self.distributed_weight_daemon = bg_lib.DistributedWeightDaemon(
            str(config_path)
        )
        self.distributed_weight_daemon.start()
        logger.info(
            "Distributed weight daemon started for node %d: store=%s",
            self.args.node_rank,
            config["store_path"],
        )

    def _load_model_from_distributed_store(self) -> None:
        config = self.distributed_weight_config
        if config is None:
            raise RuntimeError("distributed weight config is not loaded")
        self._close_skeleton_memfd()
        self.skeleton_state_dict = None
        self.parameter_server_instance = None
        self.model_info = {
            "huggingface_ckpt_name": self.args.model,
            "shm_name": f"distributed_k3_node{self.args.node_rank}",
            "tensor_meta_shm_name": "",
            "converted_ckpt_dir": self.args.converted_ckpt_dir,
            "parameter_server_size": int(config["store_bytes"]),
        }

    def _download_model_snapshot(self, hf_cache_dir: Path) -> Path:
        logger.info("Downloading model artifacts to %s", hf_cache_dir)
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(
                self.args.model,
                cache_dir=hf_cache_dir,
                ignore_patterns=["flax*", "tf*"],
            )
        )

    def _load_model_locally(
        self, _hf_cache_dir: Path, converted_ckpt_dir: Path
    ) -> None:
        model_lower = self.args.model.lower()
        if "deepseek-v4" in model_lower:
            from batchgen.models.deepseek.deepseekv4_flash.deepseekv4_flash_parameter_server import (
                DeepSeekV4Flash_Parameter_Server,
            )

            parameter_server = DeepSeekV4Flash_Parameter_Server(
                self.args.model,
                self.args.cache_dir,
                converted_ckpt_dir,
                self.args.enable_hugetlbfs,
                enable_thp=self.args.fast_init,
            )
        elif "deepseek" in model_lower:
            from batchgen.models.deepseek.deepseek_parameter_server import (
                DeepSeek_Parameter_Server,
            )

            parameter_server = DeepSeek_Parameter_Server(
                self.args.model,
                self.args.cache_dir,
                converted_ckpt_dir,
                self.args.enable_hugetlbfs,
                enable_thp=self.args.fast_init,
            )
        elif "mixtral" in self.args.model.lower():
            from batchgen.models.mixtral.mixtral_parameter_server import (
                Mixtral_Parameter_Server,
            )

            parameter_server = Mixtral_Parameter_Server(
                self.args.model, self.args.cache_dir, converted_ckpt_dir,
                enable_thp=self.args.fast_init,
            )
        elif "gpt-oss-120b" in self.args.model.lower():
            from batchgen.models.openai.gpt_oss_120b.gpt_oss_parameter_server import (
                GptOss_Parameter_Server,
            )

            parameter_server = GptOss_Parameter_Server(
                self.args.model,
                self.args.cache_dir,
                converted_ckpt_dir,
                self.args.enable_hugetlbfs,
                enable_thp=self.args.fast_init,
            )
        elif "kimi-linear" in self.args.model.lower() or "kimi-k3" in self.args.model.lower():
            from batchgen.models.moonshotai.kimi_linear.kimi_parameter_server import (
                KimiLinear_Parameter_Server,
            )

            parameter_server = KimiLinear_Parameter_Server(
                self.args.model,
                self.args.cache_dir,
                converted_ckpt_dir,
                self.args.enable_hugetlbfs,
                enable_thp=self.args.fast_init,
            )
        elif is_kimi_k25_backend_model(self.args.model):
            from batchgen.models.moonshotai.kimi_k25.kimi_parameter_server import (
                KimiK25_Parameter_Server,
            )

            parameter_server = KimiK25_Parameter_Server(
                self.args.model,
                self.args.cache_dir,
                converted_ckpt_dir,
                self.args.enable_hugetlbfs,
                enable_thp=self.args.fast_init,
            )
        elif "minimax" in self.args.model.lower():
            from batchgen.models.minimax.minimax_m25.minimax_m25_parameter_server import (
                MiniMaxM25_Parameter_Server,
            )

            parameter_server = MiniMaxM25_Parameter_Server(
                self.args.model,
                self.args.cache_dir,
                converted_ckpt_dir,
                self.args.enable_hugetlbfs,
                enable_thp=self.args.fast_init,
            )
        elif "glm-5" in self.args.model.lower() or "glm5" in self.args.model.lower():
            import sys as _diag_sys, time as _diag_time
            def _diag(msg):
                print(f"[DIAG {_diag_time.time():.3f}] {msg}", flush=True)
                _diag_sys.stdout.flush()
            _diag("    glm5: importing GLM5_Parameter_Server")
            from batchgen.models.glm.glm5.glm5_parameter_server import (
                GLM5_Parameter_Server,
            )
            _diag("    glm5: constructing GLM5_Parameter_Server")
            parameter_server = GLM5_Parameter_Server(
                self.args.model,
                self.args.cache_dir,
                converted_ckpt_dir,
                self.args.enable_hugetlbfs,
                enable_thp=self.args.fast_init,
            )
            _diag("    glm5: GLM5_Parameter_Server constructed")
        else:
            raise NotImplementedError(
                f"Model type for {self.args.model} not supported"
            )

        import sys as _diag_sys2, time as _diag_time2
        def _diag2(msg):
            print(f"[DIAG {_diag_time2.time():.3f}] {msg}", flush=True)
            _diag_sys2.stdout.flush()
        _diag2("    >>> parameter_server.Init()")
        shm_name, tensor_meta_shm_name = parameter_server.Init()
        _diag2("    <<< parameter_server.Init() returned")
        # Publish the owner before anything else can fail: a startup rollback
        # has to be able to release the regions Init just created.
        self.parameter_server_instance = parameter_server
        ps_size = parameter_server.parameter_server.byte_size()
        _diag2(f"    ps_size={ps_size / 1024**3:.2f} GB; getting skeleton_state_dict")

        # Stash skeleton_state_dict in a memfd to avoid passing tensors through mp.spawn
        skeleton_state_dict = parameter_server.parameter_server.get_skeleton_state_dict()
        self._store_skeleton_state_dict(skeleton_state_dict)
        self.skeleton_state_dict = None  # Don't keep tensors in memory
        self.model_info = {
            "huggingface_ckpt_name": self.args.model,
            "shm_name": shm_name,
            "tensor_meta_shm_name": tensor_meta_shm_name,
            "converted_ckpt_dir": converted_ckpt_dir,
            "parameter_server_size": ps_size,
        }
        logger.info("Local parameter server initialized: %s", self.model_info)

    def _load_model_from_remote_server(
        self, endpoint: str, hf_cache_dir: Path, converted_ckpt_dir: Path
    ) -> None:
        host, port = self._parse_parameter_server_endpoint(endpoint)
        logger.info("Using external parameter server at %s:%d", host, port)
        client = ParameterServerClient(host=host, port=port)
        client.load_model(
            huggingface_ckpt_name=self.args.model,
            hf_cache_dir=hf_cache_dir,
            cache_dir=self.args.cache_dir,
            converted_ckpt_dir=converted_ckpt_dir,
        )
        info = client.get_model_info()
        for required in (
            "shm_name",
            "tensor_meta_shm_name",
            "parameter_server_size",
        ):
            if required not in info:
                raise RuntimeError(
                    f"Remote parameter server response missing '{required}'"
                )
        skeleton = info.get("skeleton_state_dict")
        if skeleton is None:
            raise RuntimeError(
                "Remote parameter server did not return a skeleton_state_dict"
            )

        # Stash skeleton_state_dict in a memfd to avoid passing tensors through mp.spawn
        self._store_skeleton_state_dict(skeleton)
        self.skeleton_state_dict = None  # Don't keep tensors in memory
        self.parameter_server_instance = None
        self.model_info = {
            "huggingface_ckpt_name": info.get(
                "huggingface_ckpt_name", self.args.model
            ),
            "shm_name": info["shm_name"],
            "tensor_meta_shm_name": info["tensor_meta_shm_name"],
            "converted_ckpt_dir": info.get("converted_ckpt_dir", converted_ckpt_dir),
            "parameter_server_size": info["parameter_server_size"],
        }
        self.args.converted_ckpt_dir = Path(self.model_info["converted_ckpt_dir"])
        if not self.args.cache_dir:
            self.args.cache_dir = info.get("cache_dir") or self.args.converted_ckpt_dir
        logger.info(
            "Fetched shared memory handles from remote parameter server"
        )

    def _configure_host_kv_cache_budget(self) -> None:
        if self.args.host_kv_cache_size is not None:
            available_mem = self.args.host_kv_cache_size
        else:
            import psutil
            import shutil

            # Calculate host memory based budget: host_mem * 0.9 - model_size
            mem = psutil.virtual_memory()
            model_size_gb = self.model_info.get("parameter_server_size", 0) / (1024**3)
            host_mem_budget = int(mem.total * 0.9 / (1024**3) - model_size_gb)

            # Check /dev/shm free space
            try:
                shm_stat = shutil.disk_usage("/dev/shm")
                shm_free_gb = shm_stat.free // (1024**3)
            except (OSError, FileNotFoundError):
                shm_free_gb = host_mem_budget  # Fallback if /dev/shm not available

            # Use minimum of host memory budget and /dev/shm free space
            available_mem = min(host_mem_budget, shm_free_gb)

            # Warn if /dev/shm is limiting
            if shm_free_gb < host_mem_budget:
                logger.warning(
                    "Host KV cache size limited by /dev/shm free space (%d GB). "
                    "For full utilization, mount /dev/shm with host memory size: "
                    "sudo mount -o remount,size=%dG /dev/shm",
                    shm_free_gb,
                    int(mem.total / (1024**3)),
                )

            logger.info(
                "Auto-detected host KV cache size: %d GB "
                "(host_mem=%.1f GB, model_size=%.1f GB, /dev/shm_free=%d GB)",
                available_mem,
                mem.total / (1024**3),
                model_size_gb,
                shm_free_gb,
            )

        if available_mem <= 0:
            raise RuntimeError("Unable to determine host KV cache budget")
        # Host KV cache is per-node shared - no division needed
        self.args_dict = vars(self.args)
        self.args_dict["host_kv_cache_size_per_rank"] = available_mem

    @staticmethod
    def allocate_host_kv_cache(
        host_kv_cache_size_gb: int, model_name: str,
        primary_shm_name: str,
        aux_shm_name: str,
        enable_thp: bool = False,
    ) -> Any:
        from batchgen.kv_cache.dual_host_kv_coordinator import DualHostKVCoordinator

        # DSA models: split budget into primary + auxiliary
        dual = DualHostKVCoordinator.create_managers(
            model_name=model_name,
            host_kv_cache_size=int(host_kv_cache_size_gb * (1024**3)),
            enable_thp=enable_thp,
            primary_shm_name=primary_shm_name,
            aux_shm_name=aux_shm_name,
        )
        if dual is not None:
            primary_mgr, aux_mgr = dual
            logger.info(
                "Allocated dual host KV cache: primary + auxiliary (DSA indexer)"
            )
            return primary_mgr, aux_mgr

        config = build_host_kv_config(
            host_kv_cache_size=host_kv_cache_size_gb * (1024**3),
            model_name=model_name,
            shm_name=primary_shm_name,
        )
        config.enable_thp = enable_thp
        # Select manager based on model's KV cache configuration
        # MLA models (num_v_heads=0) don't have V cache, GQA/MHA models (num_v_heads>0) do
        if config.num_v_heads == 0:
            host_paged_kv_manager = bg_lib.MLAHostPagedKVManager(config)
        else:
            host_paged_kv_manager = bg_lib.MHAHostPagedKVManager(config)
        host_paged_kv_manager.initialize(True)
        return host_paged_kv_manager

    def _initialize_prefix_cache_owner(self) -> None:
        from batchgen.prefix_reuse.config import (
            build_prefix_cache_runtime_config,
            create_host_prefix_cache_coordinator,
        )

        from batchgen.kv_cache.dual_host_kv_coordinator import (
            DualHostKVCoordinator,
        )

        # Size the index from the same Host-KV split the workers use.
        host_budget = int(self.args.host_kv_cache_size * (1024**3))
        identity = self.args.runtime_identity
        configs = DualHostKVCoordinator.build_host_configs(
            self.args.model, host_budget,
            identity.host_kv_shm_name, identity.host_kv_aux_shm_name,
        )
        if configs is None:
            host_config = build_host_kv_config(
                model_name=self.args.model, host_kv_cache_size=host_budget,
            )
            aux_config = None
        else:
            host_config, aux_config = configs
        self.prefix_cache_runtime_config = build_prefix_cache_runtime_config(
            model_name=self.args.model,
            kv_dtype=self.args.kv_dtype,
            host_kv_config=host_config,
            aux_host_kv_config=aux_config,
            shm_name=self.args.runtime_identity.prefix_cache_shm_name,
            debug_stats=self.args.prefix_cache_debug_stats,
        )
        self.prefix_cache_coordinator = create_host_prefix_cache_coordinator(
            core_engine_module=bg_lib,
            runtime_config=self.prefix_cache_runtime_config,
            create_region=True,
        )

    def _shutdown_prefix_cache_owner(self) -> None:
        if self.prefix_cache_runtime_config is None:
            return
        # The region is an anonymous memfd: dropping the owner coordinator
        # closes the last creator-side reference and the kernel reclaims the
        # memory once the final worker mapping disappears. Nothing is named,
        # so there is nothing to unlink.
        self.prefix_cache_coordinator = None
        self.prefix_cache_runtime_config = None

    @staticmethod
    def _parse_parameter_server_endpoint(endpoint: str) -> tuple[str, int]:
        value = (endpoint or "").strip()
        if not value:
            raise ValueError("BATCHGEN_PARAMETER_SERVER_ENDPOINT is empty")
        if ":" in value:
            host, port_str = value.rsplit(":", 1)
            host = host or "localhost"
            port = int(port_str)
        else:
            host = value
            port = 10900
        return host, port
