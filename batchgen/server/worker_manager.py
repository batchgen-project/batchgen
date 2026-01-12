"""Worker lifecycle management and inference bridging for the FastAPI server."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
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
from batchgen.server.process_utils import cleanup_resources
from batchgen.server.server_args import ServerArgs
from batchgen.server_worker_main_loop import server_worker_main
from batchgen.utils import config_torch_module_initializer

logger = logging.getLogger(__name__)

PARAMETER_SERVER_ENDPOINT_ENV = "BATCHGEN_PARAMETER_SERVER_ENDPOINT"


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
        self._lock = threading.Lock()
        self._worker_exit_state = worker_exit_state or WorkerExitState()
        self._monitor_stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._join_lock = threading.Lock()
        self._stopping = False
        self._monitor_interval_s = 1.0
        self._ready_event = self._mp_ctx.Event()
        self._hugepages_enabled = False  # Track if hugepages were configured

    # ---------------------- Public API ----------------------
    def start(self) -> None:
        if self.started:
            return

        self._stopping = False
        self._monitor_stop_event.clear()
        self._ready_event.clear()
        if self.args.enable_hugetlbfs:
            self._config_hugepages()
            self._hugepages_enabled = True

        config_torch_module_initializer()
        if self.args.host_kv_cache_size:
            try:
                self.allocate_host_kv_cache(
                    self.args.host_kv_cache_size, self.args.model
                )
            except Exception as exc:
                logger.warning("Host KV cache allocation failed: %s", exc)
        self._load_model_resources()
        self._spawn_workers()
        self._start_worker_monitor()
        self._wait_for_workers_ready()
        self.started = True
        logger.info(
            "WorkerManager started with %s GPUs", torch.cuda.device_count()
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
        with self._lock:
            self.request_queue.put(payload)
            result = self.response_queue.get()
        return result

    # ---------------------- Startup helpers ----------------------
    def _config_hugepages(self) -> None:
        commands = [
            ["sysctl", "-w", "vm.nr_hugepages=350000"],
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
        pt_ckpt_dir = (
            self.args.pt_ckpt_dir
            or Path(self.args.cache_dir or ".") / "pt_ckpt"
        )
        self.args.pt_ckpt_dir = pt_ckpt_dir

        if not endpoint and self.args.cache_dir is None:
            self.args.cache_dir = self._download_model_snapshot(hf_cache_dir)

        if endpoint:
            self._load_model_from_remote_server(
                endpoint, hf_cache_dir, pt_ckpt_dir
            )
        else:
            self._load_model_locally(hf_cache_dir, pt_ckpt_dir)

        self._configure_host_kv_cache_budget()
        logger.info("Model Loaded. SHM: %s", self.model_info.get("shm_name"))

    def _spawn_workers(self) -> None:
        local_device_count = torch.cuda.device_count()
        if local_device_count == 0:
            raise RuntimeError("No CUDA devices found.")

        logger.info("Spawning %s DDP workers", local_device_count)
        world_size = self.args.world_size or (
            local_device_count * self.args.nnodes
        )
        if world_size == 1 and local_device_count > 1:
            world_size = local_device_count * self.args.nnodes

        # Auto-detect GPU architecture if not specified
        gpu_arch = self.args.gpu_arch or detect_gpu_arch()

        args = BatchGenWorkerArgs(
            model_name=self.args.model,
            hf_cache_dir=self.args.hf_cache_dir,
            cache_dir=self.args.cache_dir,
            pt_ckpt_dir=self.args.pt_ckpt_dir,
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
            skeleton_state_dict=self.skeleton_state_dict,
            # placeholders
            local_rank=-1,
            global_rank=-1,
            device=-1,
            watchdog_timeout=self.args.watchdog_timeout,
            watchdog_test_stuck_time=self.args.watchdog_test_stuck_time,
            watchdog_heartbeat_interval=self.args.watchdog_heartbeat_interval,
            enable_prepack=self.args.enable_prepack,
            host_kv_watermark=self.args.host_kv_watermark,
            enable_decode_preemption=self.args.enable_decode_preemption,
            gpu_memory_frac=self.args.gpu_memory_frac,
            initial_gpu_page_buffer=self.args.initial_gpu_page_buffer,
            extension_gpu_page_buffer=self.args.extension_gpu_page_buffer,
            decision_frequency_pages=self.args.decision_frequency_pages,
        )
        self.worker_process = mp.spawn(
            server_worker_main,
            args=(
                self.request_queue,
                self.response_queue,
                args,
                self._ready_event,
            ),
            nprocs=local_device_count,
            join=False,
            daemon=True,
        )

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
        self, _hf_cache_dir: Path, pt_ckpt_dir: Path
    ) -> None:
        if "deepseek" in self.args.model.lower():
            from batchgen.models.deepseek.deepseek_parameter_server import (
                DeepSeek_Parameter_Server,
            )

            parameter_server = DeepSeek_Parameter_Server(
                self.args.model,
                self.args.cache_dir,
                pt_ckpt_dir,
                self.args.enable_hugetlbfs,
            )
        elif "gpt-oss" in self.args.model.lower() or "gpt_oss" in self.args.model.lower():
            from batchgen.models.gpt_oss.gpt_oss_parameter_server import (
                GptOss_Parameter_Server,
            )

            parameter_server = GptOss_Parameter_Server(
                self.args.model,
                self.args.cache_dir,
                pt_ckpt_dir,
                self.args.enable_hugetlbfs,
            )
        elif "mixtral" in self.args.model.lower():
            from batchgen.models.mixtral.mixtral_parameter_server import (
                Mixtral_Parameter_Server,
            )

            parameter_server = Mixtral_Parameter_Server(
                self.args.model, self.args.cache_dir, pt_ckpt_dir
            )
        else:
            raise NotImplementedError(
                f"Model type for {self.args.model} not supported"
            )

        shm_name, tensor_meta_shm_name = parameter_server.Init()
        ps_size = parameter_server.parameter_server.byte_size()
        self.skeleton_state_dict = (
            parameter_server.parameter_server.get_skeleton_state_dict()
        )
        self.parameter_server_instance = parameter_server
        self.model_info = {
            "huggingface_ckpt_name": self.args.model,
            "shm_name": shm_name,
            "tensor_meta_shm_name": tensor_meta_shm_name,
            "pt_ckpt_dir": pt_ckpt_dir,
            "parameter_server_size": ps_size,
        }
        logger.info("Local parameter server initialized: %s", self.model_info)

    def _load_model_from_remote_server(
        self, endpoint: str, hf_cache_dir: Path, pt_ckpt_dir: Path
    ) -> None:
        host, port = self._parse_parameter_server_endpoint(endpoint)
        logger.info("Using external parameter server at %s:%d", host, port)
        client = ParameterServerClient(host=host, port=port)
        client.load_model(
            huggingface_ckpt_name=self.args.model,
            hf_cache_dir=hf_cache_dir,
            cache_dir=self.args.cache_dir,
            pt_ckpt_dir=pt_ckpt_dir,
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
        self.skeleton_state_dict = skeleton
        self.parameter_server_instance = None
        self.model_info = {
            "huggingface_ckpt_name": info.get(
                "huggingface_ckpt_name", self.args.model
            ),
            "shm_name": info["shm_name"],
            "tensor_meta_shm_name": info["tensor_meta_shm_name"],
            "pt_ckpt_dir": info.get("pt_ckpt_dir", pt_ckpt_dir),
            "parameter_server_size": info["parameter_server_size"],
        }
        self.args.pt_ckpt_dir = Path(self.model_info["pt_ckpt_dir"])
        if not self.args.cache_dir:
            self.args.cache_dir = info.get("cache_dir") or self.args.pt_ckpt_dir
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
        num_devices = torch.cuda.device_count()
        self.args_dict = vars(self.args)
        if num_devices > 0:
            self.args_dict["host_kv_cache_size_per_rank"] = (
                available_mem // num_devices
            )
        else:
            self.args_dict["host_kv_cache_size_per_rank"] = available_mem

    @staticmethod
    def allocate_host_kv_cache(
        host_kv_cache_size_gb: int, model_name: str
    ) -> Any:
        config = build_host_kv_config(
            host_kv_cache_size=host_kv_cache_size_gb * (1024**3),
            model_name=model_name,
        )
        host_paged_kv_manager = bg_lib.MLAHostPagedKVManager(config)
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
