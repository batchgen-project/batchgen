from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class V4ServingConfig:
    """Config-only edit surface for DeepSeek-V4-Flash serving experiments.

    Kernels/model code are frozen. The autonomous loop may vary only these
    serving/system knobs.
    """

    name: str
    gpu_memory_frac: float
    host_kv_cache_size_gb: int
    kv_dtype: str
    world_size: int = 4
    initial_gpu_page_buffer: int | None = None
    extension_gpu_page_buffer: int | None = None
    request_concurrency: int = 1
    numactl_node0: bool = False
    nccl_p2p_level: str | None = None
    nccl_algo: str | None = None
    nccl_min_nchannels: int | None = None
    nccl_max_nchannels: int | None = None
    nccl_buffsize_bytes: int | None = None
    nccl_shm_disable: int | None = None
    attn_prefill_mb: int | None = None
    moe_prefill_mb: int | None = None
    expert_prefill_cap: int | None = None
    prefill_token_cap: int | None = None
    attn_decode_mb: int | None = None
    moe_decode_mb: int | None = None
    expert_decode_cap: int | None = None
    watchdog_timeout_s: int = 1200
    startup_timeout_s: int = 1800
    decode_step_timeout_s: int | None = None
    server_extra_args: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def compact_json(self) -> str:
        import json

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


BASELINE_CONFIG = V4ServingConfig(
    name="baseline",
    gpu_memory_frac=0.30,
    host_kv_cache_size_gb=60,
    kv_dtype="fp8",
    world_size=4,
    request_concurrency=1,
    watchdog_timeout_s=1200,
    startup_timeout_s=1800,
)


NAMED_CONFIGS: dict[str, V4ServingConfig] = {
    BASELINE_CONFIG.name: BASELINE_CONFIG,
}


# This is the file the autonomous loop edits.
# Ranges come from docs/4xrtx6000pro-v4flash-setup.md and existing server flags.
# Keep them conservative unless a human verifies a wider range is safe.
SEARCH_SPACE: dict[str, list[Any]] = {
    "gpu_memory_frac": [0.25, 0.30, 0.40, 0.50, 0.60],
    "host_kv_cache_size_gb": [40, 60, 80, 100, 120],
    "kv_dtype": ["fp8", "bf16"],
    "request_concurrency": [1, 2, 4, 8],
    "initial_gpu_page_buffer": [None, 16, 32, 64],
    "extension_gpu_page_buffer": [None, 2, 4, 8],
    "numactl_node0": [False, True],
    "nccl_p2p_level": [None, "PIX", "NODE"],
    "nccl_algo": [None, "Ring", "Tree"],
    "nccl_min_nchannels": [None, 2, 4, 8],
    "nccl_max_nchannels": [None, 2, 4, 8],
    "nccl_buffsize_bytes": [None, 4 * 1024 * 1024, 8 * 1024 * 1024, 16 * 1024 * 1024],
    "nccl_shm_disable": [None, 0, 1],
}


EDIT_SURFACE_NOTES = {
    "frozen": [
        "batchgen kernels",
        "model code",
        "checkpoint contents",
        "bench_v4_config.py metric harness",
    ],
    "paths": {
        "checkpoint_dir": "/home/leyang/v4flash_converted_mp4",
        "hf_snapshot_dir": "/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136",
    },
    "layout_caution": (
        "The verified checkpoint is sharded for world_size=4. Alternative EP/TP layouts "
        "must only be enabled after confirming the launcher flag contract and checkpoint "
        "compatibility. Do not guess new distributed APIs."
    ),
}
