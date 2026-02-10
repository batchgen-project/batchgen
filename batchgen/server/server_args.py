"""Server argument parsing and validation utilities."""

import argparse
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def is_port_available(port: int) -> bool:
    """Return whether a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", port))
            sock.listen(1)
            return True
        except (socket.error, OverflowError):
            return False


def parse_host_port(addr: str) -> tuple[str, int]:
    """Parse a host:port string."""
    if ":" not in addr:
        raise ValueError(f"Address must be in host:port format, got '{addr}'")
    host, port_str = addr.rsplit(":", 1)
    host = host or "0.0.0.0"
    try:
        port = int(port_str)
    except ValueError as exc:
        raise ValueError(f"Invalid port in address '{addr}'") from exc
    return host, port


def _validate_port_range(name: str, port: int) -> None:
    if port <= 0 or port > 65535:
        raise ValueError(f"Invalid {name}: {port}")


def _ensure_local_port_free(port: int, label: str) -> None:
    if not is_port_available(port):
        raise ValueError(f"{label} port {port} is not available on this node")


def _default_storage_path() -> Path:
    """Return default storage path under batchgen directory."""
    return Path(__file__).parent.parent / "storage"


@dataclass
class ServerArgs:
    """Server configuration."""

    model: str
    listen_ip: str = "0.0.0.0"
    listen_port: int = 10900
    hf_cache_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None
    converted_ckpt_dir: Optional[Path] = None
    enable_hugetlbfs: bool = False
    dist_init_addr: str = "localhost:12355"
    kv_dtype: str = "bfloat16"
    host_kv_cache_size: Optional[int] = None
    gpu_arch: Optional[str] = None
    nnodes: int = 1
    node_rank: int = 0
    world_size: int = 1
    storage_path: Optional[Path] = None  # Default set in __post_init__
    save_result: bool = False  # Save direct inference results to outputs/
    watchdog_timeout: Optional[float] = None  # Disabled by default
    watchdog_test_stuck_time: float = 0.0
    watchdog_heartbeat_interval: Optional[float] = None
    # Prepack optimization (default: enabled, recommended always on)
    enable_prepack: bool = True
    # Host KV watermark percentage (default: 70% free = underutilized threshold)
    host_kv_watermark: int = 70
    # Decode preemption: interrupt decode for prefill when host KV is underutilized (default: enabled)
    enable_decode_preemption: bool = True
    # GPU memory fraction for KV cache: gpu_kv_size = GPU_mem * frac - model_size (default: 0.9)
    gpu_memory_frac: float = 0.9
    # GPU page buffer settings for decode scheduling
    initial_gpu_page_buffer: int = 32  # Pages to reserve on first GPU load
    extension_gpu_page_buffer: int = 4  # Pages to add at boundaries
    decision_frequency_pages: int = 2  # How often to make scheduling decisions (in pages)
    # EP with offloading settings
    enable_ep_with_offloading: bool = False  # Enable EP with partial expert offloading
    ep_offloading_ratio: float = 0.0  # Ratio of experts to offload (0.0-1.0)
    pre_dequantize_weights: bool = False  # Pre-dequantize MoE routed expert MXFP4 weights to BF16

    def __post_init__(self):
        if self.storage_path is None:
            self.storage_path = _default_storage_path()

    def resolve_paths(self) -> None:
        """Normalize any path-like args."""
        if isinstance(self.hf_cache_dir, str):
            self.hf_cache_dir = Path(self.hf_cache_dir)
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        if isinstance(self.converted_ckpt_dir, str):
            self.converted_ckpt_dir = Path(self.converted_ckpt_dir)
        if isinstance(self.storage_path, str):
            self.storage_path = Path(self.storage_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BatchGen FastAPI server")
    parser.add_argument(
        "--model", type=str, required=True, help="HuggingFace model name"
    )
    parser.add_argument(
        "--listen-ip", type=str, default="0.0.0.0", help="Server listen IP"
    )
    parser.add_argument(
        "--listen-port", type=int, default=10900, help="Server listen port"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Path to pre-downloaded model weights",
    )
    parser.add_argument(
        "--enable-hugetlbfs",
        action="store_true",
        help="Enable hugeTLBFS for shared memory (requires system support)",
    )
    parser.add_argument(
        "--dist-init-addr",
        type=str,
        default="localhost:12355",
        help="torch.distributed init addr",
    )
    parser.add_argument(
        "--kv-dtype", type=str, default="bfloat16", help="KV cache dtype"
    )
    parser.add_argument(
        "--host-kv-cache-size",
        type=int,
        default=None,
        help="Host KV cache size (GB)",
    )
    parser.add_argument(
        "--gpu-arch", type=str, default=None, help="GPU architecture hint"
    )
    parser.add_argument(
        "--nnodes", type=int, default=1, help="Total nodes in the cluster"
    )
    parser.add_argument(
        "--node-rank", type=int, default=0, help="Rank of this node"
    )
    parser.add_argument(
        "--world-size", type=int, default=1, help="Total world size"
    )
    parser.add_argument(
        "--storage-path",
        type=Path,
        default=None,
        help="Directory for uploaded files, batches, and outputs. Default: batchgen/storage/",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Save direct inference results to {storage_path}/outputs/ as JSONL files",
    )
    parser.add_argument(
        "--watchdog-timeout",
        type=float,
        default=None,
        help="Watchdog timeout in seconds per micro-batch/decode step. Default: disabled. Recommended: 300 for production.",
    )
    parser.add_argument(
        "--no-watchdog",
        action="store_true",
        help="Disable worker watchdog (default behavior, kept for compatibility)",
    )
    parser.add_argument(
        "--watchdog-test-stuck-time",
        type=float,
        default=0.0,
        help="Deliberately sleep during watchdog feed (testing only)",
    )
    parser.add_argument(
        "--watchdog-heartbeat-interval",
        type=float,
        default=None,
        help="Idle heartbeat interval in seconds when watchdog is enabled",
    )
    parser.add_argument(
        "--enable-prepack",
        action="store_true",
        default=True,
        help="Enable prepack optimization for efficient prefill batching (default: enabled, recommended always on)",
    )
    parser.add_argument(
        "--host-kv-watermark",
        type=int,
        default=70,
        help="Host KV cache watermark percentage. When free slots exceed this threshold, prefill is prioritized (default: 70)",
    )
    parser.add_argument(
        "--enable-decode-preemption",
        action="store_true",
        default=True,
        help="Enable decode preemption. Interrupts decode to prefill new sequences when host KV is underutilized (default: enabled, recommended always on)",
    )
    parser.add_argument(
        "--gpu-memory-frac",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use for KV cache. GPU KV cache size = GPU_mem * frac - model_size (default: 0.9)",
    )
    parser.add_argument(
        "--initial-gpu-page-buffer",
        type=int,
        default=32,
        help="Pages to reserve on first GPU load (default: 32, each page = 64 tokens)",
    )
    parser.add_argument(
        "--extension-gpu-page-buffer",
        type=int,
        default=4,
        help="Pages to add at page boundaries during decode (default: 4)",
    )
    parser.add_argument(
        "--decision-frequency-pages",
        type=int,
        default=2,
        help="How often to make scheduling decisions in pages (default: 2, each page = 64 tokens)",
    )
    parser.add_argument(
        "--enable-ep-with-offloading",
        action="store_true",
        default=False,
        help="Enable Expert Parallelism with offloading mode (partial experts persistent on GPU)",
    )
    parser.add_argument(
        "--ep-offloading-ratio",
        type=float,
        default=0.0,
        help="Ratio of experts per layer to offload (0.0-1.0). E.g., 0.2 means 20%% of experts loaded/freed at runtime",
    )
    parser.add_argument(
        "--pre-dequantize-weights",
        action="store_true",
        default=False,
        help="Pre-dequantize MoE routed expert MXFP4 weights to BF16 at load time (higher HBM usage, lower compute overhead). Other weights are unaffected.",
    )
    return parser


def validate_server_args(args: ServerArgs) -> None:
    """Validate parsed arguments."""
    _validate_port_range("listen port", args.listen_port)
    _ensure_local_port_free(args.listen_port, "Listen")

    _, dist_port = parse_host_port(args.dist_init_addr)
    _validate_port_range("dist_init_addr port", dist_port)

    if args.nnodes > 1 and args.node_rank == 0:
        _ensure_local_port_free(dist_port, "dist_init_addr")
        communicator_port = 20003
        _validate_port_range("communicator port", communicator_port)
        _ensure_local_port_free(communicator_port, "COMM")

    if args.nnodes <= 0:
        raise ValueError("nnodes must be positive")
    if args.world_size <= 0:
        raise ValueError("world_size must be positive")
    if args.node_rank < 0 or args.node_rank >= args.nnodes:
        raise ValueError("node_rank must be in [0, nnodes)")
    if args.watchdog_timeout is not None and args.watchdog_timeout < 0:
        raise ValueError("watchdog_timeout must be non-negative (0 to disable)")
    if args.watchdog_heartbeat_interval is not None:
        if args.watchdog_timeout is None:
            raise ValueError(
                "watchdog_heartbeat_interval requires watchdog_timeout"
            )
        if args.watchdog_heartbeat_interval <= 0:
            raise ValueError("watchdog_heartbeat_interval must be positive")
    if args.watchdog_test_stuck_time < 0:
        raise ValueError("watchdog_test_stuck_time must be non-negative")
    if args.watchdog_test_stuck_time > 0 and args.watchdog_timeout is None:
        raise ValueError("watchdog_test_stuck_time requires watchdog_timeout")
    if args.host_kv_watermark < 0 or args.host_kv_watermark > 100:
        raise ValueError("host_kv_watermark must be between 0 and 100")
    if args.gpu_memory_frac <= 0 or args.gpu_memory_frac > 1.0:
        raise ValueError("gpu_memory_frac must be between 0 and 1.0")
    if args.initial_gpu_page_buffer <= 0:
        raise ValueError("initial_gpu_page_buffer must be positive")
    if args.extension_gpu_page_buffer <= 0:
        raise ValueError("extension_gpu_page_buffer must be positive")
    if args.decision_frequency_pages <= 0:
        raise ValueError("decision_frequency_pages must be positive")
    if args.extension_gpu_page_buffer < args.decision_frequency_pages:
        raise ValueError(
            f"extension_gpu_page_buffer ({args.extension_gpu_page_buffer}) must be >= "
            f"decision_frequency_pages ({args.decision_frequency_pages}) to prevent overflow"
        )
    if args.ep_offloading_ratio < 0.0 or args.ep_offloading_ratio > 1.0:
        raise ValueError("ep_offloading_ratio must be between 0.0 and 1.0")
    if args.ep_offloading_ratio > 0.0 and not args.enable_ep_with_offloading:
        raise ValueError(
            "ep_offloading_ratio > 0 requires --enable-ep-with-offloading"
        )
    args.storage_path.mkdir(parents=True, exist_ok=True)


def prepare_server_args(argv: Optional[list[str]] = None) -> ServerArgs:
    """Parse CLI arguments and return a validated ServerArgs."""
    parser = _build_parser()
    parsed = parser.parse_args(argv)

    # Handle watchdog disable options
    watchdog_timeout = parsed.watchdog_timeout
    if getattr(parsed, 'no_watchdog', False) or watchdog_timeout == 0:
        watchdog_timeout = None

    server_args = ServerArgs(
        model=parsed.model,
        listen_ip=parsed.listen_ip,
        listen_port=parsed.listen_port,
        cache_dir=parsed.cache_dir,
        enable_hugetlbfs=parsed.enable_hugetlbfs,
        dist_init_addr=parsed.dist_init_addr,
        kv_dtype=parsed.kv_dtype,
        host_kv_cache_size=parsed.host_kv_cache_size,
        gpu_arch=parsed.gpu_arch,
        nnodes=parsed.nnodes,
        node_rank=parsed.node_rank,
        world_size=parsed.world_size,
        storage_path=parsed.storage_path,
        save_result=parsed.save_result,
        watchdog_timeout=watchdog_timeout,
        watchdog_test_stuck_time=parsed.watchdog_test_stuck_time,
        watchdog_heartbeat_interval=parsed.watchdog_heartbeat_interval,
        enable_prepack=True,  # Always enabled, recommended for all use cases
        host_kv_watermark=parsed.host_kv_watermark,
        enable_decode_preemption=True,  # Always enabled, recommended for all use cases
        gpu_memory_frac=parsed.gpu_memory_frac,
        initial_gpu_page_buffer=parsed.initial_gpu_page_buffer,
        extension_gpu_page_buffer=parsed.extension_gpu_page_buffer,
        decision_frequency_pages=parsed.decision_frequency_pages,
        enable_ep_with_offloading=parsed.enable_ep_with_offloading,
        ep_offloading_ratio=parsed.ep_offloading_ratio,
        pre_dequantize_weights=parsed.pre_dequantize_weights,
    )
    server_args.resolve_paths()
    validate_server_args(server_args)
    return server_args
