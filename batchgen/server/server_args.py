"""Server argument parsing and validation utilities."""

import argparse
import os
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from batchgen.config.cuda_graph_flag_policy import require_cuda_graph_flag_supported
from batchgen.eventlog.process import (
    DEFAULT_LOG_LEVEL,
    add_logging_args,
    resolve_log_dir,
)
from batchgen.models.glm.glm5.cuda_graph_policy import (
    GLM5_WHOLE_MODEL_GRAPH_COMPARE_ENV,
    is_glm5_fp8_graph_default_model,
)
from batchgen.server.runtime_identity import RuntimeIdentity


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


def _is_glm_model(model_name: Optional[str]) -> bool:
    return "glm" in (model_name or "").lower()


class WeightSourceError(ValueError):
    """The launch names no usable host-weight source for its model/topology."""


#: Node counts the Kimi-K3 distributed host-weight store supports (eight TP8
#: workers per node); the store loader enforces the same set.
_K3_DISTRIBUTED_NODE_COUNTS = (2, 4)


def _is_kimi_k3_model(model_name: Optional[str]) -> bool:
    """The registered Kimi-K3 identity, not a substring guess.

    Resolved through the config registry every startup path already uses
    (``model_registry._detect_model_type_from_identifier``), so a model that
    the registry does not bind to ``kimi_k3`` is not treated as K3 here
    either.
    """
    from batchgen.config.model_registry import (
        _detect_model_type_from_identifier,
    )

    return _detect_model_type_from_identifier(model_name or "") == "kimi_k3"


def _launch_converted_ckpt_dir(args: "ServerArgs") -> Optional[Path]:
    """The converted checkpoint this launch names, as the worker resolves it.

    ``WorkerManager`` uses ``--converted-ckpt-dir``, else
    ``<--cache-dir>/converted_ckpt``. Without either it falls back to the
    working directory, which identifies nothing, so that case is ``None``.
    """
    if args.converted_ckpt_dir is not None:
        return Path(args.converted_ckpt_dir)
    if args.cache_dir is not None:
        return Path(args.cache_dir) / "converted_ckpt"
    return None


def validate_session_kv_hold_args(args: "ServerArgs") -> None:
    """Refuse --enable-session-kv-hold where the worker cannot serve it.

    The hold reuses held Host pages through the GPT-OSS prefix-hit prefill,
    keeps them in one node's Host region, and cannot share pages with the
    prefix cache yet (the worker re-checks the rest at startup).
    """
    from batchgen.prefix_reuse.config import require_session_hold_model_support

    reasons = []
    if args.enable_prefix_cache:
        reasons.append("--enable-prefix-cache is also set")
    try:
        require_session_hold_model_support(args.model)
    except ValueError as exc:
        reasons.append(str(exc))
    if int(args.nnodes) != 1:
        reasons.append(f"--nnodes={args.nnodes}; held KV lives in one node's Host region")
    if reasons:
        raise ValueError(
            "--enable-session-kv-hold is unsupported here: " + "; ".join(reasons)
        )


def _validate_distributed_weight_source(args: "ServerArgs") -> None:
    """Preflight of a ``--distributed-weight-config`` launch.

    Refuses before any server resource exists unless the config is the
    completed output of the store builder for exactly this node, built from
    the converted checkpoint this launch names, and a sample of the store
    still hashes to its completion manifest. Each refusal ends with the exact
    builder command that rebuilds the store.
    """
    if not args.distributed_weight_config.is_file():
        raise WeightSourceError(
            "distributed_weight_config does not exist: "
            f"{args.distributed_weight_config}"
        )
    if not _is_kimi_k3_model(args.model):
        raise WeightSourceError(
            "--distributed-weight-config currently supports Kimi-K3 only"
        )
    if (
        args.nnodes not in _K3_DISTRIBUTED_NODE_COUNTS
        or args.world_size != args.nnodes * 8
    ):
        raise WeightSourceError(
            "Kimi-K3 distributed host weights require "
            "--nnodes 2 --world-size 16 or --nnodes 4 --world-size 32"
        )
    from batchgen.models.moonshotai.kimi_linear.store_builder import (
        K3StoreManifestError,
        complete_staged_store,
        rebuild_command,
        verify_published_store,
    )

    converted_ckpt = _launch_converted_ckpt_dir(args)
    verified = verify_published_store(
        args.distributed_weight_config,
        converted_ckpt=converted_ckpt,
        cache_dir=args.cache_dir,
        node_rank=args.node_rank,
        num_nodes=args.nnodes,
        spot_check_store_path=complete_staged_store(args.distributed_weight_config),
    )
    config = verified["config"]
    if (config["node_rank"], config["num_nodes"]) != (args.node_rank, args.nnodes):
        raise K3StoreManifestError(
            f"{args.distributed_weight_config} is node {config['node_rank']} "
            f"of {config['num_nodes']}, but this launch is --node-rank "
            f"{args.node_rank} --nnodes {args.nnodes}: pass this node's own "
            "config, or build it with: "
            + rebuild_command(
                args.distributed_weight_config.parent
                / f"node{args.node_rank}.json",
                None,
                converted_ckpt=converted_ckpt,
                cache_dir=args.cache_dir,
                manifest=verified["manifest"],
                node_rank=args.node_rank,
                num_nodes=args.nnodes,
            )
        )


def _apply_cuda_graph_cli_env_defaults(args: "ServerArgs") -> None:
    # Phase C: mode env vars retired. Activation is gated by the
    # `--enable-cuda-graph` / `--disable-cuda-graphs` CLI flags directly,
    # consumed by the worker's `_glm5_whole_model_graph_requested_for_current_batch`
    # via `args.enable_cuda_graph` / `args.disable_cuda_graphs`. Only the
    # developer compare env var is still pinned here when graphs are
    # explicitly disabled, to keep the compare facility off.
    if args.disable_cuda_graphs and _is_glm_model(args.model):
        os.environ[GLM5_WHOLE_MODEL_GRAPH_COMPARE_ENV] = "0"


@dataclass
class ServerArgs:
    """Server configuration."""

    model: str
    instance_id: str = "default"
    runtime_mode: str = "exclusive"
    lane_lease_manifest_fd: Optional[int] = None
    listen_ip: str = "0.0.0.0"
    listen_port: int = 10900
    hf_cache_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None
    converted_ckpt_dir: Optional[Path] = None
    distributed_weight_config: Optional[Path] = None
    # Explicit opt-in to the replicated parameter server on Kimi-K3 multi-node,
    # where the distributed host-weight store is the default path.
    replicated_weights: bool = False
    # Enable the DeepEP low-latency EP exchange for the decode graph (generic across
    # EP models). Default off = NCCL all_gather + reduce_scatter; on requires the
    # DeepEP build and fails fast at startup if it is unavailable (no silent NCCL fallback).
    enable_deepep: bool = False
    enable_hugetlbfs: bool = False
    fast_init: bool = False
    dist_init_addr: str = "localhost:12355"
    # host:port of the server-level restart control store node 0 hosts. Only
    # meaningful with --nnodes > 1; derived from dist_init_addr when unset.
    restart_control_addr: Optional[str] = None
    pynccl_port_base: int = 20003
    pynccl_port_span: int = 100
    kv_dtype: str = "bfloat16"
    host_kv_cache_size: Optional[int] = None
    # DIAGNOSTIC (glm52 memcpy32_post A/B): capture GLM-5 whole-model decode
    # graphs without per-layer host-KV staging; decode tokens are then never
    # appended to host KV. Launch-scoped because graphs are captured once per
    # server lifetime, so a batch-level batchgen_debug flag cannot express it.
    glm5_disable_decode_host_kv_staging: bool = False
    enable_prefix_cache: bool = False
    prefix_cache_debug_stats: bool = False
    # /v1/turns: a finished turn of an open session keeps its Host KV for the
    # session's next turn (GPT-OSS, one node, prefix cache off).
    enable_session_kv_hold: bool = False
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
    parse_thinking: bool = False  # Extract reasoning_content from model output
    parse_tool_call: bool = False  # Extract tool_calls from model output
    enable_cuda_graph: bool = False  # Explicitly enable CUDA graph capture for supported models
    disable_cuda_graphs: bool = True  # Disable CUDA graph capture for decode attention (128K+ crash: corrupted num_tokens_per_rank)
    cuda_graph_max_bucket_size: int = 128  # Max batch size per rank for CUDA graph capture
    cuda_graph_num_buckets: int = 16  # Number of CUDA graph bucket sizes
    persistent_phase_instances: bool = False  # Keep prefill/decode model instances across phase switches
    detokenization_include_special_tokens: bool = False  # When True, include special tokens in detokenized output
    # Dynamic host KV reservation settings
    host_kv_chunk_size: int = 8192  # Initial host KV chunk size in tokens (default: 8K)
    host_kv_eviction_watermark: int = 10  # Trigger host KV eviction when free pages < this %
    enable_host_kv_eviction: bool = False  # Deprecated: eviction is always enabled when chunked host KV is active
    adaptive_chunk: bool = True  # EMA-based adaptive chunk sizing
    adaptive_chunk_min: int = 1024  # Minimum adaptive chunk size in tokens
    adaptive_chunk_max: int = 65536  # Maximum adaptive chunk size in tokens
    adaptive_chunk_ema_alpha: float = 0.1  # EMA smoothing factor
    adaptive_chunk_multiplier: float = 1.5  # Headroom multiplier on EMA
    # Incremental result saving (crash-resilient output)
    incremental_output_dir: Optional[str] = None  # Directory for incremental JSONL; None = auto
    no_incremental_save: bool = False  # Opt-out flag to disable incremental saving
    # Decode step watchdog
    decode_step_timeout: Optional[float] = None  # Max seconds per decode step (None = disabled)
    # Startup timeout
    startup_timeout: Optional[float] = None  # Max seconds from launch to server ready (None = disabled)
    # Request pool: max QueryBook capacity. Must be > 0 (see validate_server_args).
    max_pool_size: int = 10240
    # Fixed node-shared host token pool budget. Sequence capacity is derived
    # from this budget and the model maximum context; max_pool_size is kept
    # only as a deprecated CLI compatibility field.
    input_ids_pool_size_gb: float = 50.0
    # IntakePool capacity: max total requests that can be queued.
    # Prevents OOM under high-load. Default 1M. Set 0 for unlimited.
    max_intake_capacity: int = 1_000_000
    # Structured event log root (absolute once parsed) and minimum level;
    # every process of this server, workers included, writes under it.
    log_dir: Optional[str] = None
    log_level: str = DEFAULT_LOG_LEVEL
    runtime_identity: RuntimeIdentity = field(init=False)

    def __post_init__(self):
        self.runtime_identity = RuntimeIdentity.create(
            self.instance_id,
            mode=self.runtime_mode,
        )
        if self.restart_control_addr is None:
            # The control store rides beside the rendezvous: node 0's host,
            # the next port. A malformed dist_init_addr is left for
            # validate_server_args to report, rather than failing here with a
            # complaint about a derived address the operator never typed.
            try:
                host, port = parse_host_port(self.dist_init_addr)
            except ValueError:
                pass
            else:
                self.restart_control_addr = f"{host}:{port + 1}"
        if self.storage_path is None:
            self.storage_path = _default_storage_path()
        # Default incremental output dir: {storage_path}/incremental/
        if self.incremental_output_dir is None and not self.no_incremental_save:
            self.incremental_output_dir = str(self.storage_path / "incremental")

    def resolve_paths(self) -> None:
        """Normalize any path-like args."""
        if isinstance(self.hf_cache_dir, str):
            self.hf_cache_dir = Path(self.hf_cache_dir)
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        if isinstance(self.converted_ckpt_dir, str):
            self.converted_ckpt_dir = Path(self.converted_ckpt_dir)
        if isinstance(self.distributed_weight_config, str):
            self.distributed_weight_config = Path(
                self.distributed_weight_config
            )
        if isinstance(self.storage_path, str):
            self.storage_path = Path(self.storage_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BatchGen FastAPI server")
    parser.add_argument(
        "--model", type=str, required=True, help="HuggingFace model name"
    )
    parser.add_argument(
        "--instance-id",
        type=str,
        default="default",
        help=(
            "Logical server instance identifier used to namespace mutable "
            "runtime resources. Shared-host mode is not enabled by this flag."
        ),
    )
    parser.add_argument(
        "--runtime-mode",
        choices=("exclusive", "shared"),
        default="exclusive",
        help="Host admission mode (shared requires a launcher lease bundle)",
    )
    parser.add_argument(
        "--lane-lease-manifest-fd",
        type=int,
        default=None,
        help="Inherited descriptor for the shared-lane lease manifest",
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
        "--converted-ckpt-dir",
        type=Path,
        default=None,
        help="Path to pre-converted checkpoint directory (skips conversion step)",
    )
    parser.add_argument(
        "--enable-deepep",
        action="store_true",
        help=(
            "Enable the DeepEP low-latency EP exchange for the decode graph "
            "(default: off = NCCL all_gather + reduce_scatter). Requires the DeepEP "
            "build; fails fast at startup if it is unavailable (no silent NCCL "
            "fallback). Generic across EP models; on Kimi-K3 it also needs H200 TP8 "
            "and the K3 DeepEP build."
        ),
    )
    parser.add_argument(
        "--distributed-weight-config",
        type=Path,
        default=None,
        help=(
            "Node-local distributed host-weight source configuration. "
            "When set, the server skips the replicated parameter server and "
            "workers map the compact store described by this file."
        ),
    )
    parser.add_argument(
        "--replicated-weights",
        action="store_true",
        default=False,
        help=(
            "Explicit opt-in to the replicated parameter server for Kimi-K3 "
            "multi-node serving (the distributed host-weight store is the "
            "default there; see docs/deploy-kimi-k3-h200.md §6)."
        ),
    )
    parser.add_argument(
        "--enable-hugetlbfs",
        action="store_true",
        help="Enable hugeTLBFS for shared memory (requires system support)",
    )
    parser.add_argument(
        "--fast-init",
        action="store_true",
        help="Back the shared weights and host KV cache with transparent huge "
             "pages (THP) for faster memory registration. The regions are "
             "unnamed memfds in every mode; this flag only adds THP. "
             "Requires: (1) echo always > /sys/kernel/mm/transparent_hugepage/shmem_enabled, "
             "(2) root access (for pre-allocation memory compaction). "
             "Automatically runs drop_caches + compact_memory before allocation "
             "to defragment physical memory for stable 2MB THP pages.",
    )
    parser.add_argument(
        "--dist-init-addr",
        type=str,
        default="localhost:12355",
        help="torch.distributed init addr",
    )
    parser.add_argument(
        "--restart-control-addr",
        type=str,
        default=None,
        help="Address (host:port) of the restart control store node 0 hosts "
             "for coordinated worker-pool restarts; only used with "
             "--nnodes > 1. Default: --dist-init-addr host, port + 1",
    )
    parser.add_argument(
        "--pynccl-port-base",
        type=int,
        default=20003,
        help="First port in this runtime's bounded PyNccl range",
    )
    parser.add_argument(
        "--pynccl-port-span",
        type=int,
        default=100,
        help="Number of ports reserved for PyNccl initialization and recovery",
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
        "--enable-prefix-cache",
        action="store_true",
        default=False,
        help="Enable page-level Host prefix-cache reuse for qualified models",
    )
    parser.add_argument(
        "--enable-session-kv-hold",
        action="store_true",
        default=False,
        help=(
            "Hold a finished /v1/turns turn's Host KV until its session's next "
            "turn, close or abort; the next turn prefills only the suffix past "
            "its longest common prefix with the held tokens (GPT-OSS, one "
            "node, not with --enable-prefix-cache)"
        ),
    )
    parser.add_argument(
        "--prefix-cache-debug-stats",
        action="store_true",
        default=False,
        help="Log prefix-cache lifecycle statistics",
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
        "--decode-step-timeout",
        type=float,
        default=None,
        help="Maximum seconds for a single decode step. If exceeded, server is marked unhealthy and killed. Default: disabled. Recommended: 300.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=None,
        help="Maximum seconds from process launch to server ready (/health returns 200). If exceeded, server exits. Default: disabled. Recommended: 1800.",
    )
    parser.add_argument(
        "--max-pool-size",
        type=int,
        default=10240,
        help="Deprecated compatibility value; scheduling capacity is derived "
             "from --input-ids-pool-size-gb and model context. Default: 10240.",
    )
    parser.add_argument(
        "--input-ids-pool-size-gb",
        type=float,
        default=50.0,
        help="Fixed node-shared int32 input-id pool budget in GiB. "
             "Sequence capacity is derived from this budget. Default: 50.",
    )
    parser.add_argument(
        "--max-intake-capacity",
        type=int,
        default=1_000_000,
        help="Max total requests in the intake pool. Prevents OOM under high load. "
             "Default: 1000000. Set to 0 for unlimited (not recommended).",
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
    parser.add_argument(
        "--parse-thinking",
        action="store_true",
        default=False,
        help="Extract thinking/reasoning blocks from model output into reasoning_content field",
    )
    parser.add_argument(
        "--parse-tool-call",
        action="store_true",
        default=False,
        help="Extract tool call blocks from model output into tool_calls array",
    )
    cuda_graph_group = parser.add_mutually_exclusive_group()
    cuda_graph_group.add_argument(
        "--enable-cuda-graph",
        "--enable-cuda-graphs",
        action="store_true",
        dest="enable_cuda_graph",
        default=False,
        help="Enable CUDA graph capture for supported models. For GLM-5-FP8/GLM-5.1-FP8 this enables the whole-model decode graph by default.",
    )
    cuda_graph_group.add_argument(
        "--disable-cuda-graphs",
        action="store_true",
        default=False,
        help="Disable CUDA graph capture for decode attention (enabled by default for supported models)",
    )
    parser.add_argument(
        "--glm5-disable-decode-host-kv-staging",
        action="store_true",
        default=False,
        help="DIAGNOSTIC: capture GLM-5 whole-model decode graphs without the "
        "per-layer host-KV staging copies. Decode tokens are never appended to "
        "host KV, so migration/recovery of decode tails is broken; benchmark "
        "A/B use only.",
    )
    parser.add_argument(
        "--cuda-graph-max-bucket-size",
        type=int,
        default=128,
        help="Maximum batch size per rank for CUDA graph capture (default: 128). For GLM whole-model graph, this defines the top bucket; larger decode batches use eager fallback.",
    )
    parser.add_argument(
        "--cuda-graph-num-buckets",
        type=int,
        default=16,
        help="Maximum number of CUDA graph bucket sizes (default: 16). For GLM, use 7 with max bucket 64 for [1,2,4,8,16,32,64].",
    )
    parser.add_argument(
        "--persistent-phase-instances",
        action="store_true",
        default=False,
        help="Build the prefill and decode model instances once and keep them, with their "
        "decode CUDA graphs, across phase switches; a switch only releases or refills the "
        "decode routed experts (GLM-5 only).",
    )
    parser.add_argument(
        "--detokenization-include-special-tokens",
        action="store_true",
        default=False,
        help="Include special tokens in detokenized output (default: off, special tokens stripped).",
    )
    # Dynamic host KV reservation
    parser.add_argument(
        "--host-kv-chunk-size",
        type=int,
        default=8192,
        help=(
            "Initial host-KV reservation chunk in tokens (default: 8192). "
            "The first pool init caps the effective chunk with its max output "
            "length; this sizes host-KV reservations only and does not override "
            "per-request max_completion_tokens. Later larger requests grow their "
            "own reservation up to their KV budget."
        ),
    )
    parser.add_argument(
        "--host-kv-eviction-watermark",
        type=int,
        default=10,
        help="Trigger host KV eviction when free pages drop below this percentage (default: 10)",
    )
    parser.add_argument(
        "--enable-host-kv-eviction",
        action="store_true",
        default=False,
        help="[Deprecated] Host KV eviction is now always enabled. This flag is ignored.",
    )
    parser.add_argument(
        "--adaptive-chunk",
        action="store_true",
        default=True,
        help="Enable EMA-based adaptive chunk sizing (default: enabled)",
    )
    parser.add_argument(
        "--no-adaptive-chunk",
        action="store_true",
        default=False,
        help="Disable EMA-based adaptive chunk sizing (use static --host-kv-chunk-size)",
    )
    parser.add_argument(
        "--adaptive-chunk-min",
        type=int,
        default=1024,
        help="Minimum adaptive chunk size in tokens (default: 1024)",
    )
    parser.add_argument(
        "--adaptive-chunk-max",
        type=int,
        default=65536,
        help="Maximum adaptive chunk size in tokens (default: 65536)",
    )
    parser.add_argument(
        "--adaptive-chunk-ema-alpha",
        type=float,
        default=0.1,
        help="EMA smoothing factor for adaptive chunk sizing (default: 0.1)",
    )
    parser.add_argument(
        "--adaptive-chunk-multiplier",
        type=float,
        default=1.5,
        help="Headroom multiplier on EMA for adaptive chunk sizing (default: 1.5)",
    )
    # Incremental result saving (crash-resilient)
    parser.add_argument(
        "--incremental-output-dir",
        type=str,
        default=None,
        help="Directory for incremental JSONL results (crash-resilient). "
             "Default: {storage_path}/incremental/",
    )
    parser.add_argument(
        "--no-incremental-save",
        action="store_true",
        default=False,
        help="Disable incremental saving of completed sequences to disk",
    )
    add_logging_args(parser)
    return parser


def validate_server_args(args: ServerArgs) -> None:
    """Validate parsed arguments."""
    # Checked first: naming both weight sources is a contradiction whatever
    # else is wrong with the launch.
    if args.distributed_weight_config is not None and args.replicated_weights:
        raise WeightSourceError(
            "--distributed-weight-config and --replicated-weights are "
            "contradictory: pick either the distributed host-weight store or "
            "the replicated parameter server, not both"
        )
    if args.max_pool_size <= 0:
        raise ValueError("legacy non-pool mode was removed; --max-pool-size must be positive")
    if args.input_ids_pool_size_gb <= 0:
        raise ValueError("--input-ids-pool-size-gb must be positive")
    _validate_port_range("listen port", args.listen_port)
    _ensure_local_port_free(args.listen_port, "Listen")

    _, dist_port = parse_host_port(args.dist_init_addr)
    _validate_port_range("dist_init_addr port", dist_port)

    _validate_port_range("PyNccl base port", args.pynccl_port_base)
    if args.pynccl_port_span <= 0:
        raise ValueError("pynccl_port_span must be positive")
    pynccl_port_end = args.pynccl_port_base + args.pynccl_port_span
    if pynccl_port_end > 65536:
        raise ValueError("PyNccl port range must end at or before 65535")

    if args.nnodes > 1:
        # A restart is coordinated over this store, so a port that collides
        # with the rendezvous, the API, or the PyNccl range breaks the recovery
        # path only once a restart is attempted -- long after launch.
        if not args.restart_control_addr:
            raise ValueError(
                "--nnodes > 1 requires a restart_control_addr (host:port)"
            )
        _, control_port = parse_host_port(args.restart_control_addr)
        _validate_port_range(
            "restart_control_addr port (default: the dist_init_addr port + 1)",
            control_port,
        )
        if control_port in (args.listen_port, dist_port):
            raise ValueError(
                "restart_control_addr port must differ from the listen port "
                "and the dist_init_addr port"
            )
        if control_port in range(args.pynccl_port_base, pynccl_port_end):
            raise ValueError(
                "restart_control_addr port must not fall inside "
                "[pynccl_port_base, pynccl_port_base + pynccl_port_span)"
            )

    if args.nnodes > 1 and args.node_rank == 0:
        _ensure_local_port_free(dist_port, "dist_init_addr")
        _ensure_local_port_free(control_port, "restart_control_addr")
    if args.world_size > 1 and args.node_rank == 0:
        _ensure_local_port_free(args.pynccl_port_base, "PyNccl")

    if args.nnodes <= 0:
        raise ValueError("nnodes must be positive")
    if args.world_size <= 0:
        raise ValueError("world_size must be positive")
    if args.world_size % args.nnodes != 0:
        raise ValueError("world_size must be divisible by nnodes")
    if args.node_rank < 0 or args.node_rank >= args.nnodes:
        raise ValueError("node_rank must be in [0, nnodes)")
    if args.host_kv_cache_size is None:
        raise ValueError(
            "--host-kv-cache-size is required: the size (in GB) of the host-side "
            "KV-cache pool the engine uses for KV offload (e.g. --host-kv-cache-size 64)."
        )
    if args.host_kv_cache_size <= 0:
        raise ValueError("--host-kv-cache-size must be a positive number of GB")
    if args.enable_prefix_cache:
        from batchgen.prefix_reuse.config import (
            require_prefix_cache_model_support,
        )

        require_prefix_cache_model_support(args.model)
    require_cuda_graph_flag_supported(
        args.model, enable_cuda_graph=args.enable_cuda_graph
    )
    if args.enable_session_kv_hold:
        validate_session_kv_hold_args(args)
    validate_shared_runtime_capability(args)
    if args.distributed_weight_config is not None:
        _validate_distributed_weight_source(args)
    elif _is_kimi_k3_model(args.model) and args.nnodes > 1:
        if not args.replicated_weights:
            # The distributed host-weight store is the default Kimi-K3
            # multi-node path. The replicated parameter server stages the
            # whole checkpoint on every node, which does not fit shared
            # multi-tenant hosts (the multi-tenant warning in
            # docs/deploy-kimi-k3-h200.md), so it is an explicit opt-in.
            raise WeightSourceError(
                "Kimi-K3 multi-node defaults to the distributed host-weight "
                "store: pass --distributed-weight-config <node.json> (build "
                "it with python -m "
                "batchgen.models.moonshotai.kimi_linear.store_builder, see "
                "docs/deploy-kimi-k3-h200.md §6), or opt in to the replicated "
                "parameter server with --replicated-weights (it needs room "
                "for the whole checkpoint in host memory on every node and is "
                "not safe on shared nodes)"
            )
    elif args.replicated_weights:
        raise WeightSourceError(
            "--replicated-weights only selects the weight source of a "
            "multi-node Kimi-K3 launch (--nnodes > 1); every other launch "
            "already uses the replicated parameter server, so drop the flag"
        )
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
    if args.decode_step_timeout is not None and args.decode_step_timeout <= 0:
        raise ValueError("decode_step_timeout must be positive")
    if args.startup_timeout is not None and args.startup_timeout <= 0:
        raise ValueError("startup_timeout must be positive")
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
    if args.host_kv_chunk_size <= 0:
        raise ValueError("host_kv_chunk_size must be positive")
    if args.host_kv_eviction_watermark < 0 or args.host_kv_eviction_watermark > 100:
        raise ValueError("host_kv_eviction_watermark must be between 0 and 100")
    if args.adaptive_chunk_min <= 0:
        raise ValueError("adaptive_chunk_min must be positive")
    if args.adaptive_chunk_max < args.adaptive_chunk_min:
        raise ValueError("adaptive_chunk_max must be >= adaptive_chunk_min")
    if args.adaptive_chunk_ema_alpha <= 0 or args.adaptive_chunk_ema_alpha > 1.0:
        raise ValueError("adaptive_chunk_ema_alpha must be in (0, 1]")
    if args.adaptive_chunk_multiplier <= 0:
        raise ValueError("adaptive_chunk_multiplier must be positive")
    args.storage_path.mkdir(parents=True, exist_ok=True)


def validate_shared_runtime_capability(args: ServerArgs) -> None:
    """Reject any unqualified shared-host configuration before allocation."""
    if args.runtime_mode == "exclusive":
        if args.lane_lease_manifest_fd is not None:
            raise ValueError(
                "--lane-lease-manifest-fd requires --runtime-mode shared"
            )
        return
    if args.runtime_mode != "shared":
        raise ValueError("runtime_mode must be 'exclusive' or 'shared'")
    if args.lane_lease_manifest_fd is None or args.lane_lease_manifest_fd < 0:
        raise ValueError("shared mode requires --lane-lease-manifest-fd")
    if args.model != "openai/gpt-oss-120b":
        raise ValueError("shared mode is qualified only for openai/gpt-oss-120b")
    if args.nnodes != 1 or args.node_rank != 0:
        raise ValueError("shared mode requires one node with node_rank 0")
    if args.world_size not in {1, 2, 4, 8}:
        raise ValueError("shared mode world_size must be one of 1, 2, 4, or 8")
    if args.fast_init or args.enable_hugetlbfs or args.enable_deepep:
        raise ValueError(
            "shared mode rejects fast-init, hugetlbfs, and DeepEP"
        )
    if args.enable_ep_with_offloading or args.enable_cuda_graph:
        raise ValueError(
            "shared mode rejects EP offloading and CUDA graph capture"
        )
    if args.watchdog_timeout is not None or args.decode_step_timeout is not None:
        raise ValueError(
            "shared mode rejects watchdogs that signal a parent PID"
        )
    if args.distributed_weight_config is not None:
        raise ValueError("shared mode rejects distributed host weights")
    if args.cache_dir is None or args.converted_ckpt_dir is None:
        raise ValueError(
            "shared mode requires explicit cache_dir and converted_ckpt_dir"
        )
    storage_path = Path(args.storage_path).expanduser().resolve()
    if storage_path == _default_storage_path().resolve():
        raise ValueError("shared mode requires an explicit storage_path")
    _, dist_port = parse_host_port(args.dist_init_addr)
    pynccl_ports = range(
        args.pynccl_port_base,
        args.pynccl_port_base + args.pynccl_port_span,
    )
    if args.listen_port == dist_port:
        raise ValueError("shared mode listen and distributed ports must differ")
    if args.world_size > 1 and (
        args.listen_port in pynccl_ports or dist_port in pynccl_ports
    ):
        raise ValueError("shared mode communication port allocations overlap")
    incremental = args.incremental_output_dir
    if incremental is not None:
        incremental_path = Path(incremental).expanduser().resolve()
        if not incremental_path.is_relative_to(storage_path):
            raise ValueError(
                "shared mode incremental_output_dir must be within storage_path"
            )


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
        instance_id=parsed.instance_id,
        runtime_mode=parsed.runtime_mode,
        lane_lease_manifest_fd=parsed.lane_lease_manifest_fd,
        listen_ip=parsed.listen_ip,
        listen_port=parsed.listen_port,
        cache_dir=parsed.cache_dir,
        converted_ckpt_dir=parsed.converted_ckpt_dir,
        distributed_weight_config=parsed.distributed_weight_config,
        replicated_weights=parsed.replicated_weights,
        enable_hugetlbfs=parsed.enable_hugetlbfs,
        fast_init=parsed.fast_init,
        dist_init_addr=parsed.dist_init_addr,
        restart_control_addr=parsed.restart_control_addr,
        pynccl_port_base=parsed.pynccl_port_base,
        pynccl_port_span=parsed.pynccl_port_span,
        kv_dtype=parsed.kv_dtype,
        host_kv_cache_size=parsed.host_kv_cache_size,
        glm5_disable_decode_host_kv_staging=parsed.glm5_disable_decode_host_kv_staging,
        enable_prefix_cache=parsed.enable_prefix_cache,
        prefix_cache_debug_stats=parsed.prefix_cache_debug_stats,
        enable_session_kv_hold=parsed.enable_session_kv_hold,
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
        parse_thinking=parsed.parse_thinking,
        parse_tool_call=parsed.parse_tool_call,
        pre_dequantize_weights=parsed.pre_dequantize_weights,
        enable_deepep=parsed.enable_deepep,
        enable_cuda_graph=parsed.enable_cuda_graph,
        disable_cuda_graphs=parsed.disable_cuda_graphs,
        cuda_graph_max_bucket_size=parsed.cuda_graph_max_bucket_size,
        cuda_graph_num_buckets=parsed.cuda_graph_num_buckets,
        persistent_phase_instances=parsed.persistent_phase_instances,
        detokenization_include_special_tokens=parsed.detokenization_include_special_tokens,
        host_kv_chunk_size=parsed.host_kv_chunk_size,
        host_kv_eviction_watermark=parsed.host_kv_eviction_watermark,
        enable_host_kv_eviction=parsed.enable_host_kv_eviction,
        adaptive_chunk=not getattr(parsed, 'no_adaptive_chunk', False),
        adaptive_chunk_min=parsed.adaptive_chunk_min,
        adaptive_chunk_max=parsed.adaptive_chunk_max,
        adaptive_chunk_ema_alpha=parsed.adaptive_chunk_ema_alpha,
        adaptive_chunk_multiplier=parsed.adaptive_chunk_multiplier,
        incremental_output_dir=parsed.incremental_output_dir,
        no_incremental_save=parsed.no_incremental_save,
        decode_step_timeout=parsed.decode_step_timeout,
        startup_timeout=parsed.startup_timeout,
        max_pool_size=parsed.max_pool_size,
        input_ids_pool_size_gb=parsed.input_ids_pool_size_gb,
        max_intake_capacity=parsed.max_intake_capacity,
        # Resolved against the launch directory, so a spawned worker or a
        # later chdir names the same directory.
        log_dir=resolve_log_dir(parsed.log_dir),
        log_level=parsed.log_level,
    )
    server_args.resolve_paths()
    validate_server_args(server_args)
    _apply_cuda_graph_cli_env_defaults(server_args)
    return server_args
