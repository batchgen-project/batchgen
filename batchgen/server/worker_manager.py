"""Worker lifecycle management and inference bridging for the FastAPI server."""

from __future__ import annotations

import atexit
import logging
import os
import signal
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

import torch
import torch.multiprocessing as mp

from batchgen.batchgen_worker import BatchGenWorkerArgs
from batchgen.kv_cache.host_kv_mananger_config import build_host_kv_config
from batchgen.models.engine_loader import core_engine as bg_lib
from batchgen.parameter_server_client import ParameterServerClient
from batchgen.server.process_utils import (
    cleanup_resources,
    get_hugepage_size,
    get_model_byte_size,
)
from batchgen.server.server_args import ServerArgs
from batchgen.server_worker_main_loop import server_worker_main
from batchgen.utils import config_torch_module_initializer

logger = logging.getLogger(__name__)

PARAMETER_SERVER_ENDPOINT_ENV = "BATCHGEN_PARAMETER_SERVER_ENDPOINT"


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


def detect_gpu_arch() -> str:
    """Auto-detect GPU architecture based on CUDA compute capability.

    Returns:
        'hopper' for compute capability >= 9.0 (H100, H20, etc.)
        'ampere' for compute capability 8.x (A100, A5000, RTX 4090, etc.)

    Raises:
        RuntimeError: If no CUDA devices found or unsupported architecture
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA devices available for GPU architecture detection")

    major, minor = torch.cuda.get_device_capability(0)
    device_name = torch.cuda.get_device_name(0)

    if major >= 9:
        arch = "hopper"
    elif major == 8:
        arch = "ampere"
    else:
        raise RuntimeError(
            f"Unsupported GPU architecture: compute capability {major}.{minor} "
            f"({device_name}). BatchGen requires Hopper (sm_90+) or Ampere (sm_80+)."
        )

    logger.info(
        "Auto-detected GPU architecture: %s (compute capability %d.%d, %s)",
        arch, major, minor, device_name,
    )
    return arch


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

    @property
    def reason(self) -> Optional[str]:
        return self._reason

    @property
    def exception(self) -> Optional[BaseException]:
        return self._exception


class WorkerManager:
    """Starts GPU workers and exposes a thread-safe inference API."""

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
        self.request_queue: mp.Queue = self._mp_ctx.Queue()
        self.response_queue: mp.Queue = self._mp_ctx.Queue()
        self.worker_process = None
        self.started = False
        self.model_info: Dict[str, Any] = {}
        self.args_dict: Dict[str, Any] = {}
        self.parameter_server_instance = None
        self.skeleton_state_dict = None
        self.skeleton_state_dict_file = None
        self._lock = threading.Lock()
        self._worker_exit_state = worker_exit_state or WorkerExitState()
        self._monitor_stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._join_lock = threading.Lock()
        self._stopping = False
        self._monitor_interval_s = 1.0
        self._ready_event = self._mp_ctx.Event()

        # Register cleanup for skeleton state dict temp file
        atexit.register(self._cleanup_skeleton_state_dict_file)
        self._hugepages_enabled = False  # Track if hugepages were configured

    def _cleanup_skeleton_state_dict_file(self) -> None:
        """Clean up temporary skeleton state dict file."""
        if self.skeleton_state_dict_file and os.path.exists(self.skeleton_state_dict_file):
            try:
                logging.debug(f"Cleaning up skeleton state dict temp file: {self.skeleton_state_dict_file}")
                os.remove(self.skeleton_state_dict_file)
                self.skeleton_state_dict_file = None
            except Exception as e:
                logging.warning(f"Failed to cleanup temp file {self.skeleton_state_dict_file}: {e}")

    # ---------------------- Public API ----------------------
    def start(self) -> None:
        import time as _time

        if self.started:
            return

        startup_start = _time.monotonic()

        self._stopping = False
        self._monitor_stop_event.clear()
        self._ready_event.clear()

        if self.args.fast_init:
            _validate_shmem_enabled()
            self._compact_memory()

        if self.args.enable_hugetlbfs:
            byte_size = get_model_byte_size(self.args.model)
            self._config_hugepages(byte_size)
            self._hugepages_enabled = True

        config_torch_module_initializer()
        if self.args.host_kv_cache_size:
            kv_start = _time.monotonic()
            try:
                self.host_kv_manager = self.allocate_host_kv_cache(
                    self.args.host_kv_cache_size, self.args.model,
                    enable_memfd=self.args.fast_init,
                )
            except Exception as exc:
                logger.warning("Host KV cache allocation failed: %s", exc)
                self.host_kv_manager = None
            logger.info("[startup] Host KV cache allocated in %.2fs",
                        _time.monotonic() - kv_start)

        model_start = _time.monotonic()
        self._load_model_resources()
        logger.info("[startup] Model resources loaded in %.2fs",
                    _time.monotonic() - model_start)

        spawn_start = _time.monotonic()
        self._spawn_workers()
        self._start_worker_monitor()
        self._wait_for_workers_ready()
        logger.info("[startup] Workers ready in %.2fs",
                    _time.monotonic() - spawn_start)

        self.started = True
        logger.info(
            "[startup] Total server startup: %.2fs (fast_init=%s, GPUs=%s)",
            _time.monotonic() - startup_start,
            self.args.fast_init,
            torch.cuda.device_count(),
        )

    def stop(self) -> None:
        if not self.started:
            return
        self._stopping = True
        self._monitor_stop_event.set()
        logger.info("Stopping WorkerManager...")

        # Stop the monitor thread
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=5)

        # Collect worker PIDs before sending shutdown signal
        worker_pids = self._get_worker_pids()

        # Send SIGTERM to workers immediately for faster shutdown
        # This is critical for Node 1 workers that may be blocked in NCCL
        # waiting for Node 0 (which may already be shutting down)
        if worker_pids:
            logger.info("Sending SIGTERM to %d worker processes...", len(worker_pids))
            for pid in worker_pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                except (ProcessLookupError, OSError):
                    pass  # Process already exited

        # Also send poison pill in case workers are not blocked in NCCL
        try:
            self.request_queue.put(None)
        except Exception:
            logger.warning("Failed to signal worker shutdown", exc_info=True)

        # Wait for workers to exit (reduced timeout for interactive use)
        workers_joined = False
        if self.worker_process is not None:
            try:
                with self._join_lock:
                    self.worker_process.join(timeout=5)  # Reduced from 30s
                workers_joined = True
            except Exception:
                logger.warning("Failed to join worker process", exc_info=True)

        # Force-kill workers that didn't exit after SIGTERM
        if not workers_joined and worker_pids:
            logger.warning(
                "Workers did not exit gracefully, force-killing..."
            )
            for pid in worker_pids:
                try:
                    proc = psutil.Process(pid)
                    proc.kill()
                    logger.info(f"Force-killed worker process {pid}")
                except Exception:
                    pass

        # Get shm_name for cleanup if available
        shm_name = self.model_info.get("shm_name")
        shm_prefix = shm_name if shm_name else "batchgen"

        # Cleanup resources (shared memory, hugepages, etc.)
        cleanup_resources(
            shm_prefix=shm_prefix,
            clean_hugepages=self._hugepages_enabled,
            kill_workers=False,  # Already handled above
        )

        self.started = False
        logger.info("WorkerManager stopped")

    def _get_worker_pids(self) -> List[int]:
        """Get PIDs of all worker processes."""
        pids = []
        if self.worker_process is None:
            return pids
        processes = getattr(self.worker_process, "processes", None)
        if not processes:
            return pids
        for proc in processes:
            if proc.pid is not None:
                pids.append(proc.pid)
        return pids

    def get_worker_exit_state(self) -> WorkerExitState:
        return self._worker_exit_state

    def infer(
        self,
        prompts: List[str],
        max_input_len: Optional[int],
        max_output_len: int,
        ignore_eos: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        # Incremental writer metadata
        custom_id_map: Optional[Dict[int, str]] = None,
        request_url_map: Optional[Dict[int, str]] = None,
        prompt_text_map: Optional[Dict[int, str]] = None,
        batch_id: Optional[str] = None,
        model_name: Optional[str] = None,
        incremental_output_dir: Optional[str] = None,
        parse_thinking: bool = False,
        parse_tool_call: bool = False,
    ) -> List[Any]:
        if not self.started:
            raise RuntimeError("WorkerManager has not been started")
        payload = {
            "prompts": prompts,
            "max_input_len": max_input_len,  # None = dynamically determined
            "max_output_len": max_output_len,
            "ignore_eos": ignore_eos,
            "temperature": temperature,
            "top_p": top_p,
        }
        # Incremental writer metadata (only included when active)
        if incremental_output_dir and custom_id_map:
            payload["incremental_output_dir"] = incremental_output_dir
            payload["custom_id_map"] = custom_id_map
            payload["request_url_map"] = request_url_map or {}
            payload["prompt_text_map"] = prompt_text_map or {}
            payload["batch_id"] = batch_id
            payload["model_name"] = model_name
            payload["parse_thinking"] = parse_thinking
            payload["parse_tool_call"] = parse_tool_call
        with self._lock:
            self.request_queue.put(payload)
            result = self.response_queue.get()
        return result

    # ---------------------- Startup helpers ----------------------
    def _compact_memory(self) -> None:
        """Drop page cache and compact memory for stable THP allocation."""
        import subprocess
        import time as _time
        t0 = _time.monotonic()
        try:
            subprocess.run(["sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], check=True)
            subprocess.run(["sh", "-c", "echo 1 > /proc/sys/vm/compact_memory"], check=True)
            logging.info("[fast-init] Memory compaction completed in %.2fs (drop_caches + compact_memory)",
                         _time.monotonic() - t0)
        except (subprocess.CalledProcessError, PermissionError) as e:
            logging.warning("[fast-init] Memory compaction failed (requires root): %s", e)

    def _config_hugepages(self, byte_size: int = None) -> None:
        """Configure hugepages for shared memory.

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

        commands = [
            ["sysctl", "-w", f"vm.nr_hugepages={num_hugepages}"],
            ["mkdir", "-p", "/dev/hugepages"],
            ["mount", "-t", "hugetlbfs", "none", "/dev/hugepages"],
        ]
        for cmd in commands:
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except Exception as exc:
                logger.warning(
                    "Hugepages configuration failed (%s): %s", cmd, exc
                )

    def _load_model_resources(self) -> None:
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

        if not endpoint and self.args.cache_dir is None:
            self.args.cache_dir = self._download_model_snapshot(hf_cache_dir)

        if endpoint:
            self._load_model_from_remote_server(
                endpoint, hf_cache_dir, converted_ckpt_dir
            )
        else:
            self._load_model_locally(hf_cache_dir, converted_ckpt_dir)

        self._configure_host_kv_cache_budget()
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

        # Calculate local world size (workers per node)
        local_world_size = world_size // self.args.nnodes

        # Validate: can't spawn more workers than visible GPUs
        if local_world_size > local_device_count:
            raise ValueError(
                f"world_size ({world_size}) requires {local_world_size} GPUs per node, "
                f"but only {local_device_count} GPUs are visible. "
                f"Use CUDA_VISIBLE_DEVICES to expose more GPUs."
            )

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
            disable_cuda_graphs=self.args.disable_cuda_graphs,
            cuda_graph_max_bucket_size=self.args.cuda_graph_max_bucket_size,
            cuda_graph_num_buckets=self.args.cuda_graph_num_buckets,
            host_kv_chunk_size=self.args.host_kv_chunk_size,
            enable_host_kv_eviction=self.args.enable_host_kv_eviction,
            host_kv_eviction_watermark=self.args.host_kv_eviction_watermark,
            adaptive_chunk=self.args.adaptive_chunk,
            adaptive_chunk_min=self.args.adaptive_chunk_min,
            adaptive_chunk_max=self.args.adaptive_chunk_max,
            adaptive_chunk_ema_alpha=self.args.adaptive_chunk_ema_alpha,
            adaptive_chunk_multiplier=self.args.adaptive_chunk_multiplier,
            fast_init=self.args.fast_init,
            kv_memfd_pid=self._get_kv_memfd_pid(),
            kv_memfd_fd=self._get_kv_memfd_fd(),
            weights_memfd_pid=self._get_weights_memfd_pid(),
            weights_memfd_fd=self._get_weights_memfd_fd(),
        )
        self.worker_process = mp.spawn(
            server_worker_main,
            args=(
                self.request_queue,
                self.response_queue,
                args,
                self._ready_event,
            ),
            nprocs=local_world_size,  # Use world_size-derived count, not device_count
            join=False,
            daemon=True,
        )

    def _get_kv_memfd_pid(self) -> int:
        if self.args.fast_init and getattr(self, 'host_kv_manager', None) is not None:
            return os.getpid()
        return -1

    def _get_kv_memfd_fd(self) -> int:
        if self.args.fast_init and getattr(self, 'host_kv_manager', None) is not None:
            return self.host_kv_manager.memfd_fd()
        return -1

    def _get_weights_memfd_pid(self) -> int:
        ps = getattr(self, 'parameter_server_instance', None)
        if self.args.fast_init and ps is not None:
            fd = ps.parameter_server.weights_memfd_fd()
            if fd >= 0:
                return os.getpid()
        return -1

    def _get_weights_memfd_fd(self) -> int:
        ps = getattr(self, 'parameter_server_instance', None)
        if self.args.fast_init and ps is not None:
            return ps.parameter_server.weights_memfd_fd()
        return -1

    def _wait_for_workers_ready(self) -> None:
        if self.worker_process is None:
            return
        while not self._ready_event.wait(timeout=self._monitor_interval_s):
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
        while not self._monitor_stop_event.is_set():
            if self.worker_process is None:
                return
            try:
                with self._join_lock:
                    self.worker_process.join(timeout=self._monitor_interval_s)
            except Exception as exc:
                if self._stopping:
                    return
                logger.exception("Worker process join failed")
                self._handle_worker_failure(
                    f"Worker process crashed: {exc}", exc
                )
                return
            if self._monitor_stop_event.is_set():
                return
            exit_reason = self._collect_worker_exit_reason()
            if exit_reason:
                if self._stopping:
                    return
                logger.error(exit_reason)
                self._handle_worker_failure(exit_reason, None)
                return

    def _collect_worker_exit_reason(self) -> Optional[str]:
        if self.worker_process is None:
            return None
        processes = getattr(self.worker_process, "processes", None)
        if not processes:
            return None
        exited = []
        for idx, proc in enumerate(processes):
            if proc.exitcode is None:
                continue
            exited.append(f"idx={idx} pid={proc.pid} exitcode={proc.exitcode}")
        if not exited:
            return None
        return "Detected worker process exit: " + ", ".join(exited)

    def _handle_worker_failure(
        self, reason: str, exc: Optional[BaseException]
    ) -> None:
        if self._stopping:
            return
        if self._worker_exit_state.set_failure(reason, exc):
            self._request_server_shutdown()

    @staticmethod
    def _request_server_shutdown() -> None:
        try:
            os.kill(os.getpid(), signal.SIGINT)
        except Exception:
            logger.error("Failed to signal server shutdown", exc_info=True)

    # ---------------------- Model loading helpers ----------------------
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
        if "deepseek" in self.args.model.lower():
            from batchgen.models.deepseek.deepseek_parameter_server import (
                DeepSeek_Parameter_Server,
            )

            parameter_server = DeepSeek_Parameter_Server(
                self.args.model,
                self.args.cache_dir,
                converted_ckpt_dir,
                self.args.enable_hugetlbfs,
                enable_memfd=self.args.fast_init,
            )
        elif "mixtral" in self.args.model.lower():
            from batchgen.models.mixtral.mixtral_parameter_server import (
                Mixtral_Parameter_Server,
            )

            parameter_server = Mixtral_Parameter_Server(
                self.args.model, self.args.cache_dir, converted_ckpt_dir,
                enable_memfd=self.args.fast_init,
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
                enable_memfd=self.args.fast_init,
            )
        elif "moonshotai" in self.args.model.lower() or "kimi" in self.args.model.lower():
            from batchgen.models.moonshotai.kimi_k25.kimi_parameter_server import (
                KimiK25_Parameter_Server,
            )

            parameter_server = KimiK25_Parameter_Server(
                self.args.model,
                self.args.cache_dir,
                converted_ckpt_dir,
                self.args.enable_hugetlbfs,
                enable_memfd=self.args.fast_init,
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
                enable_memfd=self.args.fast_init,
            )
        else:
            raise NotImplementedError(
                f"Model type for {self.args.model} not supported"
            )

        shm_name, tensor_meta_shm_name = parameter_server.Init()
        ps_size = parameter_server.parameter_server.byte_size()

        # Get skeleton_state_dict and save to temp file to avoid passing tensors through mp.spawn
        skeleton_state_dict = parameter_server.parameter_server.get_skeleton_state_dict()
        logger.info(f"Saving skeleton state dict to temp file ({len(skeleton_state_dict)} keys)...")

        # Create temp file for skeleton state dict
        fd, file_path = tempfile.mkstemp(suffix='.pt', prefix='batchgen_skel_')
        os.close(fd)  # Close fd, torch.save will open its own handle

        torch.save(skeleton_state_dict, file_path)
        actual_size = os.path.getsize(file_path)
        logger.info(f"Skeleton state dict saved to {file_path} ({actual_size / (1024**2):.2f} MB)")

        self.skeleton_state_dict_file = file_path
        self.skeleton_state_dict = None  # Don't keep tensors in memory
        self.parameter_server_instance = parameter_server
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

        # Save skeleton_state_dict to temp file to avoid passing tensors through mp.spawn
        logger.info(f"Saving skeleton state dict to temp file ({len(skeleton)} keys)...")
        fd, file_path = tempfile.mkstemp(suffix='.pt', prefix='batchgen_skel_')
        os.close(fd)  # Close fd, torch.save will open its own handle

        torch.save(skeleton, file_path)
        actual_size = os.path.getsize(file_path)
        logger.info(f"Skeleton state dict saved to {file_path} ({actual_size / (1024**2):.2f} MB)")

        self.skeleton_state_dict_file = file_path
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
        enable_memfd: bool = False,
    ) -> Any:
        config = build_host_kv_config(
            host_kv_cache_size=host_kv_cache_size_gb * (1024**3),
            model_name=model_name,
        )
        if enable_memfd:
            config.enable_memfd = True
        # Select manager based on model's KV cache configuration
        # MLA models (num_v_heads=0) don't have V cache, GQA/MHA models (num_v_heads>0) do
        if config.num_v_heads == 0:
            host_paged_kv_manager = bg_lib.MLAHostPagedKVManager(config)
        else:
            host_paged_kv_manager = bg_lib.MHAHostPagedKVManager(config)
        host_paged_kv_manager.initialize(True)
        return host_paged_kv_manager

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
