import gc
import json
import logging
import os
import signal
import sys
import threading
import time
import traceback
from datetime import timedelta
from typing import Any, List, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from batchgen.batchgen_worker import (
	KIMI_K3_MODEL_ID,
	BatchGenWorker,
	BatchGenWorkerArgs,
)
from batchgen.deprecation import LegacyInferenceDeprecated
from batchgen.server.process_utils import (
	die_with_parent,
	install_worker_signal_handlers,
)
from batchgen.server.runtime_locks import hold_run_lock
from batchgen.server.worker_readiness import _signal_local_worker_manager_ready
from batchgen.server.watchdog import Watchdog


_WORKER_FATAL_STORE_KEY = "batchgen_worker_fatal_v1"
_WORKER_FATAL_ACK_TIMEOUT_S = 5.0

# How long the shutdown watchdog lets the graceful path finish before it aborts
# this worker's communicators.  A worker that is NOT stuck leaves the main loop
# well inside this window (it sees the shutdown flag at the loop boundary, or
# consumes the queue sentinel) and the watchdog then does nothing at all.  A
# worker blocked inside a collective because a peer died cannot reach either,
# and only an abort makes that collective return.
_SHUTDOWN_ABORT_GRACE_S = 5.0

# How often the watchdog loads the cohort shutdown flag.  The flag carries no
# wake handshake by design (see WorkerManager._create_worker_channels), so it
# is polled rather than waited on.
_SHUTDOWN_FLAG_POLL_S = 0.25

# How long the watchdog waits for the main loop to retire on its own AFTER the
# aborts were issued, before it ends this process itself.
#
# The abort frees a rank blocked in a collective within milliseconds, but the
# release is silent: the blocked sync returns normally and, when the collective
# was captured in a CUDA graph, nothing ever raises.  A freed rank therefore
# leaves only by testing the cohort flag on its own execution path, and it has a
# FINITE window in which to do so -- a graph-mode rank measured in the W5 bench
# resumed for a few hundred replays and then re-wedged permanently with no
# exception at all.  A rank that misses that window (or was never freed, for
# example because it is parked in a sync that is not a collective) would
# otherwise hold the whole teardown until the manager's grace expires and it is
# SIGKILLed.  Ending the process here instead keeps the worst case bounded at
# this window rather than that grace, and is the only mechanism proven to end a
# rank in that state.
_SHUTDOWN_SELF_EXIT_GRACE_S = 20.0

# Exit status for that self-exit.  It must be distinguishable from every status
# this worker can otherwise produce, because the manager reports a worker's
# exitcode verbatim as the reason a cohort went down:
#   0        the clean pool-shutdown exit (batchgen_worker.py)
#   1        every fatal path here and in the worker supervisor
#   128+n    process_utils' signal translation (SIGINT 130, SIGKILL 137, ...)
#   -n       multiprocessing's own rendering of a signal death (SIGKILL -9)
# 86 is outside all of those ranges, so a self-exit can never be read back as a
# fatal error, a clean exit, or a signal.  The manager needs no change to reap
# it: it interprets no status value specially (it only formats exitcode into its
# exit reason), proc.join returns on death like any other exit, and a worker
# that is already dead when the fallback leg runs is filtered out of it before
# the signal-fallback marker is logged, so that marker correctly does not claim
# a worker that ended itself.
_WORKER_SELF_EXIT_CODE = 86

# Exit-path markers.  Every worker that reaches its ordered teardown emits
# exactly one of these, so a teardown can be read back per worker instead of
# inferred from timings.  The signal-fallback path has no marker here: a
# force-killed worker never runs Python again, so WorkerManager emits that one.
_EXIT_PATH_SENTINEL = "ordered-via-sentinel"
_EXIT_PATH_FLAG = "ordered-via-flag"
_EXIT_PATH_ABORT = "ordered-via-abort"
_EXIT_PATH_DECODE_BOUNDARY = "ordered-via-decode-boundary"

# Not an ordered teardown: this worker ended itself rather than let a wedged
# rank hold the cohort's teardown open.  It is logged by the watchdog thread,
# which owns the GIL while the main thread is stuck below Python, so unlike the
# signal fallback this marker CAN come from the worker itself.
_EXIT_PATH_SELF_EXIT = "self-exit"

# This worker's share of the run's liveness lock.  It is module state, not a
# local, because it must outlive every function here: the kernel releases the
# lock only when this process's last descriptor closes, which is what proves to
# the next run of this instance that the run is dead.
_run_lock_file = None


def _shutdown_worker_runtime(worker) -> None:
	"""Release the engine before the worker-owned weight mapping.

	``BatchGen`` keeps a non-owning C++ reference to ``Weights_Storage`` and
	its H2D/D2H workers can still hold pointers into that mapping.  Python's
	attribute destruction order is not an ownership contract, so the sentinel
	path performs the order explicitly: stop the engine, drop every Python
	owner that can retain it or a weight tensor, then release the storage last.
	Signal-driven ``os._exit`` remains the fail-closed fallback when this path
	cannot run (for example, a worker blocked in a native collective).
	"""
	if getattr(worker, "_runtime_shutdown_complete", False):
		return
	worker._runtime_shutdown_complete = True
	start = time.monotonic()

	engine = getattr(worker, "core_engine", None)
	if engine is not None:
		try:
			engine.terminate()
		except Exception:
			logging.warning(
				"Rank %s: core engine termination failed during worker shutdown",
				getattr(worker, "rank", "unknown"),
				exc_info=True,
			)
			# Do not release the mapping while a native engine may still own
			# pointers into it.  Propagating reaches the worker supervisor's
			# os._exit fail-closed path, where the kernel unmaps the process.
			raise

	# These objects may retain the BatchGen binding, model tensors, or CUDA
	# graph references.  Drop them before the storage object below.
	for attr in (
		"model",
		"parallel_manager",
		"gpu_paged_kv_cache_manager",
		"host_paged_kv_worker_view",
		"host_paged_kv_worker_view_aux",
		"_cuda_graph_adapter",
		"_cuda_graph_manager",
		"_glm5_moe_cuda_graph_manager",
		"_glm5_layer_cuda_graph_manager",
		"_whole_model_segment",
		"_buffer_pool",
		"_retired_buffer_pools",
		"skeleton_state_dict",
		"global_batch",
	):
		if hasattr(worker, attr):
			setattr(worker, attr, None)

	# The model-specific initializer retains the same C++ BatchGen object.
	worker.initializer = None
	worker.core_engine = None
	engine = None
	gc.collect()

	# This is deliberately last: the C++ engine has a non-owning reference to
	# it, and the destructor performs cudaHostUnregister/munmap/shm_unlink.
	worker.weights_storage = None
	gc.collect()
	logging.info(
		"Rank %s: ordered runtime shutdown completed in %.3fs",
		getattr(worker, "rank", "unknown"),
		time.monotonic() - start,
	)


def _format_worker_fatal(
	rank_idx: int,
	global_rank: Optional[int],
	error: BaseException,
	traceback_text: str,
) -> str:
	return json.dumps({
		"rank_idx": rank_idx,
		"global_rank": global_rank,
		"error_type": type(error).__name__,
		"error": str(error),
		"traceback": traceback_text,
	}, ensure_ascii=False)


def _publish_worker_fatal_to_store(message: str) -> None:
	"""Publish the first Python fatal without using an NCCL collective."""
	try:
		from torch.distributed.distributed_c10d import _get_default_store
		store = _get_default_store()
		store.compare_set(_WORKER_FATAL_STORE_KEY, "", message)
	except Exception:
		logging.exception("Failed to publish worker fatal through TCPStore")


def _send_worker_fatal_to_local_server(
	response_queue: mp.Queue,
	message: str,
	ready_event: Optional[mp.Event],
	fatal_ack_event: Optional[mp.Event],
) -> None:
	"""Send the exact fatal to this node's scheduler before worker exit."""
	try:
		response_queue.put({"type": "pool_shutdown", "error": message})
	except Exception:
		logging.exception("Failed to enqueue worker fatal for the local server")
		return

	# During startup the scheduler does not exist yet, so WorkerManager must
	# observe the process exit. Once ready, wait briefly for the scheduler to
	# persist active batches as failed before the worker disappears.
	if (
		ready_event is not None
		and ready_event.is_set()
		and fatal_ack_event is not None
	):
		if not fatal_ack_event.wait(timeout=_WORKER_FATAL_ACK_TIMEOUT_S):
			logging.error(
				"Timed out waiting for local server to acknowledge worker fatal"
			)


def _start_worker_fatal_monitor(
	global_rank: int,
	response_queue: mp.Queue,
	ready_event: Optional[mp.Event],
	fatal_ack_event: Optional[mp.Event],
) -> None:
	"""Relay a remote-rank TCPStore fatal to the rank-0 HTTP launcher."""
	if global_rank != 0:
		return

	from torch.distributed.distributed_c10d import _get_default_store
	store = _get_default_store()

	def _monitor() -> None:
		try:
			while True:
				if store.check([_WORKER_FATAL_STORE_KEY]):
					message = store.get(_WORKER_FATAL_STORE_KEY).decode("utf-8")
					logging.error("Remote worker Python fatal: %s", message)
					_send_worker_fatal_to_local_server(
						response_queue,
						message,
						ready_event,
						fatal_ack_event,
					)
					os._exit(1)
				time.sleep(0.1)
		except Exception:
			logging.exception("Worker fatal TCPStore monitor failed")
			os._exit(1)

	threading.Thread(
		target=_monitor,
		name="worker-fatal-monitor",
		daemon=True,
	).start()


def _shutdown_requested(shutdown_flag: Optional[Any]) -> bool:
	"""Load the cohort teardown flag.

	A plain load out of lock-free shared memory: no lock is taken and no peer
	is woken, so this is safe to call from the main loop's hot boundary and
	from the watchdog thread at the same time.
	"""
	return shutdown_flag is not None and bool(shutdown_flag.value)


def _live_process_groups() -> List[Any]:
	"""Snapshot the process groups this rank is a member of.

	Grounded in torch's own internals, which both the all-groups abort and
	``destroy_process_group`` walk the same way:
	``torch.distributed.distributed_c10d._world.pg_names`` maps each
	ProcessGroup this rank joined to its name string
	(torch 2.8.0 ``distributed_c10d.py:2271-2273``; torch 2.14.1
	``distributed_c10d.py:3136-3139``; torch 2.9.0, the production version,
	``distributed_c10d.py:2280-2283``).  Groups this rank is not a member of
	never enter it, so this is exactly "this worker's live communicators".

	The result is a list, not the live mapping: the aborts below run
	concurrently and torch mutates ``_world`` as groups go away.
	"""
	from torch.distributed.distributed_c10d import _world
	return list(_world.pg_names)


def _abort_one_process_group(group: Any) -> None:
	"""Abort exactly ONE communicator -- the group handed in, nothing else.

	``group`` is a required argument with no default, which is the point: the
	all-groups form of this operation is the bug this watchdog exists to avoid
	and it must be unreachable from here.  Asking torch to abort every group in
	one call makes it walk them in reverse *name* order inside one
	``ncclGroupStart``/``ncclGroupEnd`` region, and ``ProcessGroupNCCL::abort``
	then blocks in ``waitForFutureOrTimeout`` on the first communicator it
	reaches -- which is generally NOT the one carrying the wedged collective, so
	it never gets there and the blocked rank stays blocked.  Asking for the
	default group explicitly is the same branch by another name: torch tests
	``group is None or group == GroupMember.WORLD`` and ``GroupMember.WORLD`` IS
	the default group.

	``ProcessGroup.abort()`` is that same primitive without the ordering: it is
	the frame directly under torch's own per-group branch
	(torch 2.8.0 ``distributed_c10d.py:2298``; torch 2.14.1
	``distributed_c10d.py:3161``), reached through a pybind binding declared
	``call_guard<gil_scoped_release>``, so it does not hold the GIL while it
	blocks and the main thread can be released while this call is still in
	flight.
	"""
	if group is None:
		# Never degrade to "abort whatever torch thinks is current": that is
		# the all-groups form by another name.
		raise RuntimeError(
			"the shutdown abort needs an explicit process group; refusing to "
			"abort an unspecified group"
		)
	group.abort()


def _flush_worker_logs() -> None:
	"""Push buffered log records out before an exit that runs no handlers.

	``os._exit`` is the only way out of a process whose main thread is stuck
	below Python, and it bypasses interpreter shutdown entirely: no atexit
	hooks, no ``logging.shutdown()``, no stream flush.  Anything still sitting
	in a handler's buffer at that moment is lost, which would make the
	self-exit marker -- the one line that attributes the exit -- the most likely
	record to disappear.  Every flush is individually guarded: this runs on the
	way out of a failing teardown and must not raise there.
	"""
	for handler in list(logging.getLogger().handlers):
		try:
			handler.flush()
		except Exception:
			pass
	for stream in (sys.stdout, sys.stderr):
		try:
			stream.flush()
		except Exception:
			pass


def _start_shutdown_abort_watchdog(
	global_rank: int,
	shutdown_flag: Optional[Any],
	loop_exited: threading.Event,
	aborted: threading.Event,
	live_process_groups: Optional[Any] = None,
	abort_one_group: Optional[Any] = None,
	exit_process: Optional[Any] = None,
) -> Optional[threading.Thread]:
	"""Let a collective blocked at teardown return instead of hanging.

	The only thing this thread ever reacts to is the cohort shutdown flag,
	which nothing but ``WorkerManager._stop_workers`` sets.  It therefore
	cannot fire during healthy serving: there is no timeout and no heartbeat in
	this path.  Once teardown has set the flag the graceful path still gets
	``_SHUTDOWN_ABORT_GRACE_S`` to leave the main loop on its own, and only a
	loop still running after that window is aborted.

	The abort runs on ONE THROWAWAY DAEMON THREAD PER COMMUNICATOR, started
	concurrently and never joined.  Both halves of that are load-bearing: a
	per-group abort whose group is not the wedged one blocks indefinitely, so
	no thread may stand in front of another's call and nothing may wait on any
	of them.  Fired concurrently, the thread that happens to hold the wedged
	communicator releases the main thread regardless of what the others are
	doing, and this watchdog thread exits as soon as they are all launched.

	Having issued the aborts, this thread stays alive for one last job: it waits
	``_SHUTDOWN_SELF_EXIT_GRACE_S`` for the main loop to retire on its own and,
	if it has not, ends the process.  That wait is not a join on an abort (the
	abort never returns in the shape this exists for) -- it is a wait on the
	main loop's own exit, which is the thing being given a last chance.
	"""
	if shutdown_flag is None:
		return None
	enumerate_groups = live_process_groups or _live_process_groups
	abort_group = abort_one_group or _abort_one_process_group
	self_exit = exit_process or os._exit

	def _abort_entry(group: Any, label: str) -> None:
		started = time.monotonic()
		logging.warning(
			"[shutdown] rank %s aborting process group %s", global_rank, label
		)
		try:
			abort_group(group)
		except BaseException:
			logging.exception(
				"[shutdown] rank %s abort of process group %s raised",
				global_rank,
				label,
			)
			return
		logging.warning(
			"[shutdown] rank %s abort of process group %s returned "
			"elapsed=%.3fs",
			global_rank,
			label,
			time.monotonic() - started,
		)

	def _watch() -> None:
		# Poll the flag; the same timed wait notices a loop that left on its
		# own, so a healthy teardown costs one extra wakeup at most.
		while not _shutdown_requested(shutdown_flag):
			if loop_exited.wait(timeout=_SHUTDOWN_FLAG_POLL_S):
				return
		if loop_exited.wait(timeout=_SHUTDOWN_ABORT_GRACE_S):
			# The main loop retired at its own boundary check or consumed the
			# sentinel; there is nothing wedged to abort.
			return
		try:
			groups = list(enumerate_groups())
		except BaseException:
			logging.exception(
				"[shutdown] rank %s could not enumerate its process groups; "
				"the manager's signal fallback still covers this worker",
				global_rank,
			)
			return
		if not groups:
			# Nothing to abort, so claim nothing: the exit-path marker must not
			# say an abort freed this rank when none was issued.
			logging.warning(
				"[shutdown] rank %s main loop still running %.1fs after "
				"shutdown was requested, but it holds no live process group; "
				"the manager's signal fallback still covers this worker",
				global_rank,
				_SHUTDOWN_ABORT_GRACE_S,
			)
			return
		logging.warning(
			"[shutdown] rank %s main loop still running %.1fs after shutdown "
			"was requested; aborting %d process group(s) so the blocked "
			"collective returns",
			global_rank,
			_SHUTDOWN_ABORT_GRACE_S,
			len(groups),
		)
		# Published before the threads start: the main loop reads this to name
		# the exit path it ends up taking.
		aborted.set()
		for index, group in enumerate(groups):
			threading.Thread(
				target=_abort_entry,
				args=(group, str(index)),
				name=f"worker-shutdown-abort-{global_rank}-{index}",
				daemon=True,
			).start()
		# The aborts are away.  A rank they freed leaves through the decode-step
		# boundary check or the loop boundary within its working window; a rank
		# they did not free never will, and holding the teardown open for it
		# only trades this window for the manager's much longer grace and a
		# SIGKILL.  So give the loop that window and then end this process.
		if loop_exited.wait(timeout=_SHUTDOWN_SELF_EXIT_GRACE_S):
			return
		logging.error(
			"[shutdown] worker exit path=%s rank=%s pid=%s",
			_EXIT_PATH_SELF_EXIT,
			global_rank,
			os.getpid(),
		)
		# os._exit runs no handler and flushes nothing, so the marker above has
		# to reach the log first or the exit is unattributable.
		_flush_worker_logs()
		self_exit(_WORKER_SELF_EXIT_CODE)

	thread = threading.Thread(
		target=_watch,
		name=f"worker-shutdown-abort-{global_rank}",
		daemon=True,
	)
	thread.start()
	return thread


def _resolve_main_loop_exit(
	global_rank: int,
	shutdown_flag: Optional[Any],
	error: BaseException,
) -> str:
	"""Classify an error out of the main loop: this cohort's teardown, or fatal.

	The secondary exit.  The wedged collective itself returns NORMALLY once its
	communicator is aborted: the kernel exits, the event fires, and the rank is
	back in Python with no error at all, so the boundary check in the loop is
	what usually retires a worker.  The NEXT collective after an abort does
	raise, which is how a worker deep inside ``generate_persistent()`` gets out,
	and this is that path.

	An abort cannot be told from a real communicator failure by exception type:
	both arrive as an error from a collective.  The cohort shutdown flag is the
	only thing that separates them, and nothing but teardown sets it, so an
	error that arrives with the flag unset is re-raised untouched and keeps
	today's fatal behavior.
	"""
	if not _shutdown_requested(shutdown_flag):
		raise error
	logging.warning(
		"Rank %s: main loop left through the shutdown abort; running the "
		"ordered teardown",
		global_rank,
		exc_info=error,
	)
	return _EXIT_PATH_ABORT


def _setup_nccl_env():
	"""
	Set up NCCL environment variables for better reliability in multi-node setups.
	These should be set before any NCCL operations.
	"""
	# Increase connection timeout and retry attempts
	# NCCL_SOCKET_TIMEOUT: timeout in milliseconds for socket operations (default: varies)
	if "NCCL_SOCKET_TIMEOUT" not in os.environ:
		os.environ["NCCL_SOCKET_TIMEOUT"] = "300000"  # 5 minutes in ms

	# NCCL_NET_RETRY_COUNT: number of retries for network operations
	if "NCCL_NET_RETRY_COUNT" not in os.environ:
		os.environ["NCCL_NET_RETRY_COUNT"] = "100"  # More retries (default is ~10)

	# TORCH_NCCL_ENABLE_MONITORING: Disable NCCL HeartbeatMonitor entirely
	# The default 60s timeout can cause false positives during CPU-bound operations
	# (e.g., tokenizing large batches). Setting TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=0
	# does NOT disable it - it sets timeout to 0 which fails immediately.
	# Use TORCH_NCCL_ENABLE_MONITORING=0 to actually disable monitoring.
	if os.environ.get("TORCH_NCCL_ENABLE_MONITORING") is None:
		os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"
		logging.info("Disabled NCCL HeartbeatMonitor (TORCH_NCCL_ENABLE_MONITORING=0)")

	# NCCL_BUFFSIZE: buffer size for NCCL operations (default: 4MB)
	# Larger buffer improves throughput for multi-node communication
	if "NCCL_BUFFSIZE" not in os.environ:
		os.environ["NCCL_BUFFSIZE"] = "16777216"  # 16MB buffer size

	# NCCL_ALGO: force deterministic tree algorithm only when
	# BATCHGEN_DETERMINISTIC is set. Tree has fixed reduction order but
	# ~1.3-2× slower than ring on intra-node NVLink for the ~1.5 MiB MoE
	# allreduce payloads we issue per layer. Default (unset) lets NCCL
	# auto-select per payload size — measured 49 ms/step → expected
	# 20-30 ms/step on the MoE allreduce tail.
	# Enable BATCHGEN_DETERMINISTIC=1 to re-pin tree for repetition-
	# regression reproductions.
	_deterministic = os.environ.get("BATCHGEN_DETERMINISTIC", "0").lower() in (
		"1", "true", "yes", "on")
	if _deterministic:
		os.environ.setdefault("NCCL_ALGO", "allreduce:tree")
		logging.info(
			"BATCHGEN_DETERMINISTIC=1 → NCCL_ALGO=allreduce:tree "
			"(deterministic reductions)")
	else:
		logging.info(
			"NCCL_ALGO not forced — NCCL auto-select "
			"(ring/tree per payload; ~1.3-2× faster than pinned tree)")

	# NCCL_NVLS_ENABLE: disable NVLS (NVLink-SHARP, in-switch reduction) by
	# default. NVLS requires multicast buffer registration, and NCCL registers
	# buffers *during CUDA-graph capture*. Our decode path captures allreduce
	# across all MoE layers for every bucket (e.g. K2.5 / GLM-5 whole-model
	# graph = 16 buckets × 60 layers), so NVLS adds ~88 ms/op of multicast
	# registration → ~5.5 s/bucket, ~80 s total capture. NVLS gives no runtime
	# benefit at our small decode allreduce sizes (≤7 MB; measured replay equal
	# or slightly faster with NVLS off), so it is pure cost here. Disabling it
	# cuts whole-model capture ~80 s → ~4 s with no decode-throughput/accuracy
	# regression (validated on K2.5 L2). NVLS targets large-tensor training
	# allreduce; export NCCL_NVLS_ENABLE=1 to re-enable for such workloads.
	# Must be set before NCCL comm init (ncclCommInitRank reads it per-comm).
	if os.environ.get("NCCL_NVLS_ENABLE") is None:
		os.environ["NCCL_NVLS_ENABLE"] = "0"
		logging.info(
			"NCCL_NVLS_ENABLE=0 (NVLS off — avoids multicast registration cost "
			"during CUDA-graph capture; export NCCL_NVLS_ENABLE=1 to re-enable)")


def server_worker_main(
	rank_idx: int,
	request_queue: mp.Queue,
	response_queue: mp.Queue,
	args: BatchGenWorkerArgs,
	ready_event: Optional[mp.Event] = None,
	fatal_ack_event: Optional[mp.Event] = None,
	shutdown_flag: Optional[Any] = None,
):
	try:
		_server_worker_main_impl(
			rank_idx,
			request_queue,
			response_queue,
			args,
			ready_event,
			fatal_ack_event,
			shutdown_flag,
		)
	except Exception as e:
		global_rank = getattr(args, "global_rank", None)
		logging.error(f"[FATAL] Unhandled exception in worker "
					  f"rank_idx={rank_idx}, global_rank={global_rank}: {e}")
		traceback_text = traceback.format_exc()
		print(traceback_text, file=sys.stderr, flush=True)
		message = _format_worker_fatal(
			rank_idx,
			global_rank,
			e,
			traceback_text,
		)
		if dist.is_available() and dist.is_initialized():
			_publish_worker_fatal_to_store(message)
		_send_worker_fatal_to_local_server(
			response_queue,
			message,
			ready_event,
			fatal_ack_event,
		)
		# Do not enter distributed teardown after a rank fatal: peers may already
		# be stuck in a collective. The two launchers own process-tree cleanup.
		os._exit(1)


def _setup_worker_logging(rank_idx: int, global_rank: int = -1):
	"""
	Configure logging for worker processes.
	Spawned processes don't inherit logging config from the parent process.
	"""
	# Use global_rank if available, otherwise fall back to rank_idx
	rank_label = global_rank if global_rank >= 0 else rank_idx
	logging.basicConfig(
		level=logging.INFO,
		format=f'%(asctime)s - [BatchGenWorker-{rank_label}] - %(levelname)s - %(message)s',
		force=True,  # Override any existing configuration
	)


def _server_worker_main_impl(
	rank_idx: int,
	request_queue: mp.Queue,
	response_queue: mp.Queue,
	args: BatchGenWorkerArgs,
	ready_event: Optional[mp.Event] = None,
	fatal_ack_event: Optional[mp.Event] = None,
	shutdown_flag: Optional[Any] = None,
):
	"""
	The main loop for the GPU workers.
	Rank 0 acts as the coordinator, reading from the master process queue.
	All ranks receive the full global batch for coordinated scheduling.
	"""
	# Before anything else, including logging: tie this worker's life to the
	# server process.  SIGKILL is the only reliable parent-death signal here —
	# a Python-level handler cannot run while this thread is inside a native
	# collective, so the worker would survive as an orphan holding GPUs and SHM.
	die_with_parent()

	# Step 0.0: Join this run's liveness lock.  A spawned worker inherits no
	# descriptor, so it opens the server-created file itself and never closes it;
	# while any process of the run holds it LOCK_SH, a later run of this instance
	# cannot take it LOCK_EX and so cannot reclaim this run's resources.
	global _run_lock_file
	# args.runtime_dir IS the run directory holding the liveness lock file.
	_run_lock_file = hold_run_lock(args.runtime_dir)

	# Step 0: Configure logging for this worker process first
	_setup_worker_logging(rank_idx)

	# Step 0.1: Install signal handlers so workers respond to Ctrl+C
	# This is critical for multi-node setups where node 1 workers might be
	# blocked in NCCL operations when node 0 is killed.
	def _worker_shutdown_callback():
		"""Cleanup callback when worker receives termination signal."""
		start = time.monotonic()
		try:
			if dist.is_available() and dist.is_initialized():
				dist.destroy_process_group()
		except Exception:
			pass
		finally:
			logging.info(
				"[shutdown] worker process-group teardown elapsed=%.3fs",
				time.monotonic() - start,
			)

	install_worker_signal_handlers(_worker_shutdown_callback)

	# Step 0.5: Set up NCCL environment for reliability
	_setup_nccl_env()
	
	# Step 1: Hydrate the rank of this process
	visible_device_count = torch.cuda.device_count()
	if visible_device_count < args.local_world_size:
		raise RuntimeError(
			f"local_world_size={args.local_world_size} exceeds visible GPU count "
			f"{visible_device_count}"
		)
	args.local_rank = rank_idx
	args.global_rank = args.local_world_size * args.nnode_rank + rank_idx
	args.device = args.local_rank

	# Reconfigure logging with actual global rank for clearer log output
	_setup_worker_logging(rank_idx, args.global_rank)

	# 2. Initialize Process Group
	logging.info(f"Starting BatchGen Worker on local rank {args.local_rank}, global rank {args.global_rank}")

	# Set CUDA device before init_process_group so NCCL uses the correct device context
	torch.cuda.set_device(args.local_rank)

	try:
		pg_kwargs = dict(
			backend="nccl",
			init_method="tcp://" + args.dist_init_addr,
			world_size=args.world_size,
			rank=args.global_rank,
			timeout=timedelta(seconds=3600),
		)
		# device_id requires torch >= 2.8
		import inspect
		if 'device_id' in inspect.signature(dist.init_process_group).parameters:
			pg_kwargs['device_id'] = torch.device(f"cuda:{args.local_rank}")
		dist.init_process_group(**pg_kwargs)
	except Exception as e:
		logging.error(f"Failed to initialize process group: {e}")
		sys.exit(1)
	logging.info(f"Process group initialized for rank {args.global_rank}/{args.world_size}.")
	_start_worker_fatal_monitor(
		args.global_rank,
		response_queue,
		ready_event,
		fatal_ack_event,
	)

	# CRITICAL: Warmup NCCL connections before entering server loop
	# This ensures all inter-node NCCL connections are fully established
	# before any application-level communication happens.
	# Without this, the first collective may fail with "Connection refused"
	# if remote ranks haven't fully initialized their NCCL listeners.
	try:
		logging.debug(f"Rank {args.global_rank}: Starting NCCL connection warmup...")

		# Step 1: Simple barrier to ensure all ranks have reached this point
		dist.barrier()
		logging.debug(f"Rank {args.global_rank}: Barrier 1 passed")

		# Step 2: Small all_reduce to establish actual NCCL connections
		# NCCL lazily establishes connections, so we force it here
		warmup_tensor = torch.ones(1, device=f"cuda:{args.local_rank}")
		dist.all_reduce(warmup_tensor, op=dist.ReduceOp.SUM)
		torch.cuda.synchronize()

		expected = float(args.world_size)
		if abs(warmup_tensor.item() - expected) > 1e-6:
			raise RuntimeError(f"NCCL warmup failed: expected {expected}, got {warmup_tensor.item()}")
		logging.debug(f"Rank {args.global_rank}: all_reduce warmup passed")

		# Step 3: Test broadcast_object_list specifically since that's what fails
		test_obj = [args.global_rank] if args.global_rank == 0 else [None]
		dist.broadcast_object_list(test_obj, src=0)
		if test_obj[0] != 0:
			raise RuntimeError(f"broadcast_object_list warmup failed: got {test_obj[0]}")
		logging.debug(f"Rank {args.global_rank}: broadcast_object_list warmup passed")

		# Final barrier to ensure all warmup is complete
		dist.barrier()
		logging.info(f"Rank {args.global_rank}: NCCL warmup complete")
		
	except Exception as e:
		logging.error(f"Rank {args.global_rank}: NCCL warmup failed: {e}")
		logging.error(f"This usually indicates a network issue or startup race condition.")
		logging.error(f"Try restarting the server or check network connectivity between nodes.")
		sys.exit(1)

	# 2. Instantiate the BatchGenWorker
	worker = BatchGenWorker(args)

	# 2.5. Initialize watchdog for stuck process detection
	watchdog_timeout = getattr(args, 'watchdog_timeout', None)
	watchdog_test_stuck_time = getattr(args, 'watchdog_test_stuck_time', 0.0)
	watchdog = Watchdog.create(
		debug_name=f"worker-{args.global_rank}",
		watchdog_timeout=watchdog_timeout,
		soft=False,  # Hard mode: kill parent process on timeout
		test_stuck_time=watchdog_test_stuck_time,
	)
	if watchdog_timeout:
		logging.info(f"Rank {args.global_rank}: Watchdog initialized with timeout={watchdog_timeout}s")

	# Pass watchdog to worker for fine-grained feeding during inference
	worker.set_watchdog(watchdog)

	# Kimi-K3 finishes its workload-independent one-time work — model build,
	# resident-EP decode shards, streamed-SP8 buffers and the H2D weight
	# schedule — BEFORE the readiness signal below, so a server that reports
	# ready can serve its first request immediately. Exact model id only: the
	# sibling Kimi-Linear checkpoints share the architecture but not this
	# lifecycle. Any failure propagates and the worker never signals ready.
	if args.model_name == KIMI_K3_MODEL_ID:
		worker.prepare_kimi_k3_startup()

	# CRITICAL: Barrier after worker init to ensure all ranks complete cudaHostRegister
	# The Host KV pinned memory registration can take 200+ seconds and varies per rank.
	# Without this barrier, faster ranks will start the main loop and attempt collective
	# operations while slower ranks are still initializing, causing NCCL errors.
	logging.info(f"Rank {args.global_rank}: Worker initialized, waiting for all ranks at barrier...")
	dist.barrier()
	logging.info(f"Rank {args.global_rank}: All ranks ready, entering main loop.")

	# Each launcher owns a node-local event, so local rank 0 must signal its own
	# WorkerManager. The global barrier above proves all workers are ready first.
	_signal_local_worker_manager_ready(
		ready_event,
		local_rank=args.local_rank,
		global_rank=args.global_rank,
	)

	# 2.6. Initialize decode watchdog AFTER barrier — only monitors decode steps,
	# not worker init, CUDA graph capture, or NCCL warmup.
	decode_step_timeout = getattr(args, 'decode_step_timeout', None)
	decode_watchdog = Watchdog.create(
		debug_name=f"decode-{args.global_rank}",
		watchdog_timeout=decode_step_timeout,
		soft=False,  # Hard mode: kill parent process on timeout
	)
	if decode_step_timeout:
		logging.info(f"Rank {args.global_rank}: Decode watchdog initialized with timeout={decode_step_timeout}s")
	worker.set_decode_watchdog(decode_watchdog)

	# 2.7. Let the decode loop see this cohort's teardown.
	#
	# The loop boundary below is reached only between batches; a worker inside
	# generate_persistent() spends its life in the decode loop and would never
	# test the flag there.  That matters because the abort's release is silent
	# and finite: the freed rank resumes stepping and then re-wedges, so the
	# only code that can retire it is the decode loop itself.
	#
	# The seam is a zero-argument probe set on the worker, matching the watchdog
	# setters above, and it is deliberately a callable rather than the flag: the
	# flag's shape (lock-free shared byte, plain load, no handshake) is this
	# module's contract with WorkerManager and stays here, while the worker only
	# ever asks a yes/no question.  Set before the loop and never cleared, so
	# the probe cannot be missing on any path that reaches decode.
	worker.set_cohort_shutdown_probe(lambda: _shutdown_requested(shutdown_flag))

	# 3. Long-lived server loop
	global_rank = args.global_rank

	# NCCL timeout prevention strategy:
	# The problem: NCCL watchdog times out if a collective operation doesn't complete within timeout.
	# When Rank 0 is blocking on request_queue.get() while other ranks wait on broadcast, 
	# NCCL will timeout if no work arrives.
	#
	# Solution: Rank 0 uses non-blocking queue.get with timeout, then all ranks periodically 
	# perform a lightweight collective to keep NCCL alive (heartbeat pattern).
	QUEUE_POLL_TIMEOUT = 30.0  # Rank 0 polls queue every 30 seconds

	logging.info(f"Entering main server loop on rank {global_rank}.")

	# The abort watchdog is armed with the loop and reacts to the cohort
	# shutdown flag alone, so arming it here cannot affect serving.
	loop_exited = threading.Event()
	aborted = threading.Event()
	_start_shutdown_abort_watchdog(global_rank, shutdown_flag, loop_exited, aborted)
	exit_path = _EXIT_PATH_SENTINEL
	try:
		while True:
			# --- STEP 0: Teardown boundary check ---
			# The PRIMARY exit toward the ordered teardown.  An aborted
			# collective does not raise: the kernel exits, the event fires,
			# and the rank returns normally with no error at all.  A guard
			# that only routes exceptions therefore never fires, which is
			# exactly how the first attempt at this fix failed validation.
			# Testing the flag here is what actually retires the rank.
			if _shutdown_requested(shutdown_flag):
				exit_path = _EXIT_PATH_ABORT if aborted.is_set() else _EXIT_PATH_FLAG
				logging.info(
					"Rank %s: shutdown flag observed at the loop boundary; "
					"running the ordered teardown",
					global_rank,
				)
				break

			# --- STEP 1: Data Acquisition with heartbeat to prevent NCCL timeout ---
			task_data = None
			work_available = False

			while not work_available:
				# Same boundary, one level in.  An idle rank blocks in the
				# heartbeat broadcast below, and an aborted broadcast returns
				# to THIS loop's top, not the outer one.  Stop polling and let
				# the single check above decide and log.
				if _shutdown_requested(shutdown_flag):
					break
				if global_rank == 0:
					# Rank 0: Non-blocking poll on queue with timeout
					try:
						task_data = request_queue.get(timeout=QUEUE_POLL_TIMEOUT)
						work_available = True
					except Exception:
						# Queue.get timeout - no work yet, signal others
						work_available = False

				# All ranks synchronize on work availability status
				# This is a fast broadcast (single int) that keeps NCCL alive
				status_tensor = torch.tensor([1 if work_available else 0], dtype=torch.int32, device='cuda')
				dist.broadcast(status_tensor, src=0)
				work_available = status_tensor.item() == 1

				if not work_available:
					# No work yet - this broadcast acts as a heartbeat to prevent NCCL timeout
					# Also feed watchdog to prevent false stuck detection during idle periods
					watchdog.feed()
					# Loop back and poll again
					continue

			if not work_available:
				# The poll loop above only ends without work when the shutdown
				# flag stopped it, so go back to the boundary check and retire
				# there.  Unreachable on every pre-existing path.
				continue

			# --- STEP 2: Broadcast full task to all ranks ---
			task_container = [task_data]
			dist.broadcast_object_list(task_container, src=0)
			task_data = task_container[0]

			# --- STEP 3: Shutdown Check ---
			if task_data is None:
				logging.info(f"Rank {global_rank} received shutdown signal. Exiting worker.")
				break

			# --- STEP 3.6: Pool mode — "init" message triggers persistent generate() ---
			if isinstance(task_data, dict) and task_data.get("type") == "init":
				logging.info(f"Rank {global_rank}: Pool mode init received")
				try:
					current_max_output = task_data.get("max_output_len", 4096)
					max_context_length = task_data.get("max_context_length", None)

					# Initialize worker with pool capacity (no sequences yet)
					worker.Init(None, current_max_output, 0,
						max_context_length=max_context_length)

					# Set admission and response queues for persistent generate()
					worker.set_admission_queue(request_queue)
					worker.set_response_queue(response_queue)
					worker._publish_trajectory_pool_capacity()

					if global_rank == 0:
						logging.info(
							f"[POOL] Worker initialized (max_pool_size={args.max_pool_size}). "
							f"Entering persistent generate() loop."
						)

					# Enter persistent generate() — blocks until shutdown
					# generate() starts with empty global_batch, polls for admissions
					worker.generate_persistent()

				except Exception as e:
					logging.error(f"Error in pool mode on rank {global_rank}: {e}", exc_info=True)
					# Let the outer worker boundary publish this rank's exact traceback
					# through the TCPStore sideband and its node-local response queue.
					raise

				# The decode-step boundary exit.  A rank freed by the abort
				# leaves its decode loop through the probe set above and
				# arrives here with no exception, having issued no further
				# GPU or collective work -- an ordered exit, and the one the
				# survivors of a mid-decode cohort death take.  The worker
				# reports having taken it rather than this being inferred from
				# the flag: a pool shutdown that merely coincided with teardown
				# is a different path and must not claim this marker.  The
				# pool_shutdown sentinel below is skipped: the cohort it would
				# report to is the one being torn down.
				if worker.cohort_shutdown_observed:
					exit_path = _EXIT_PATH_DECODE_BOUNDARY
					logging.info(
						"Rank %s: shutdown flag observed at the decode-step "
						"boundary; running the ordered teardown",
						global_rank,
					)
					break

				# Pool mode exits here — send shutdown sentinel and break
				if global_rank == 0:
					response_queue.put({"type": "pool_shutdown"})
				break

			# --- STEP 4: No other message shape is served ---
			# The legacy non-pool inference path was removed; the batch API is the
			# only way in. The message was broadcast above, so every rank reaches
			# this line together and raises into the outer fatal boundary rather
			# than leaving one rank behind in a collective.
			if isinstance(task_data, dict) and "prompts" in task_data:
				raise LegacyInferenceDeprecated()
			raise RuntimeError(
				f"Rank {global_rank}: unexpected worker message: {task_data!r}"
			)
	except BaseException as exc:
		# The SECONDARY exit.  A worker deep inside generate_persistent() never
		# reaches the boundary check above, and the first collective it issues
		# AFTER its communicator was aborted raises; this is that path.  It is
		# the ONE guard -- the collective call sites inside the loop are
		# deliberately left alone -- and it re-raises anything that arrives
		# without the shutdown flag set, so a genuine failure stays fatal.
		exit_path = _resolve_main_loop_exit(global_rank, shutdown_flag, exc)
	finally:
		# Tell the watchdog the loop is done BEFORE the ordered teardown below:
		# that teardown releases a large registered mapping and can take
		# minutes, and it must never be mistaken for a loop still running and
		# aborted underneath.
		loop_exited.set()

	# Cleanup.  Keep the engine alive until its worker threads stop, then drop
	# the storage mapping only after all non-owning references are gone.  The
	# abort does not touch any of this: it only tears down communicators.
	_shutdown_worker_runtime(worker)
	# Skipped once this worker's communicators were aborted.  Its no-group form
	# shuts groups down in the SAME reverse-name order the all-groups abort
	# uses (torch 2.8.0 distributed_c10d.py:2183-2186, torch 2.14.1
	# :3017-3020), so after an abort it is both pointless and a fresh chance to
	# block on a sibling communicator -- the exact failure mode the per-group
	# abort exists to avoid.
	if dist.is_available() and dist.is_initialized() and not aborted.is_set():
		dist.destroy_process_group()
	logging.info(
		"[shutdown] worker exit path=%s rank=%s", exit_path, global_rank
	)
