from __future__ import annotations

import importlib
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterable, Iterator, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
INFERENCE_DIR = (
    REPO_ROOT
    / "batchgen"
    / "models"
    / "deepseek"
    / "deepseekv4_flash"
    / "assets"
    / "inference"
)
ENCODING_DIR = INFERENCE_DIR.parent / "encoding"
DEFAULT_TRACE_LAYERS = (0, 2, 3)


@dataclass(frozen=True)
class RuntimeInfo:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device


@dataclass(frozen=True)
class SparseAttentionTrace:
    layer_id: int
    phase: str
    start_pos: int
    seqlen: int
    compress_ratio: int
    window_size: int
    q: torch.Tensor
    kv: torch.Tensor
    attn_sink: torch.Tensor
    topk_idxs: torch.Tensor
    softmax_scale: float
    output: torch.Tensor


@contextmanager
def prepend_sys_path(*paths: Path) -> Iterator[None]:
    old_path = list(sys.path)
    sys.path[:0] = [str(path) for path in paths]
    try:
        yield
    finally:
        sys.path = old_path


def import_reference_model_module() -> ModuleType:
    expected_path = INFERENCE_DIR / "model.py"
    existing = sys.modules.get("model")
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if (
            existing_file is not None
            and Path(existing_file).resolve() != expected_path
        ):
            del sys.modules["model"]

    with prepend_sys_path(INFERENCE_DIR, ENCODING_DIR):
        module = importlib.import_module("model")
    module_path = Path(module.__file__).resolve()
    if module_path != expected_path:
        raise RuntimeError(
            f"imported unexpected model module from {module_path}; expected "
            f"{expected_path}"
        )
    return module


def load_model_config(config_path: Path) -> dict:
    with config_path.open() as f:
        return json.load(f)


def build_model_args(
    ref_model: ModuleType,
    config: dict,
    *,
    max_batch_size: int = 1,
):
    args = ref_model.ModelArgs(**config)
    args.max_batch_size = max_batch_size
    return args


def configure_reference_globals(
    ref_model: ModuleType,
    args,
    runtime: RuntimeInfo,
) -> None:
    ref_model.world_size = runtime.world_size
    ref_model.rank = runtime.rank
    ref_model.default_dtype = (
        torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
    )
    ref_model.scale_fmt = (
        "ue8m0" if args.scale_dtype == "fp8" else args.scale_fmt
    )
    ref_model.scale_dtype = (
        torch.float8_e8m0fnu if args.scale_dtype == "fp8" else torch.float32
    )


def init_runtime() -> RuntimeInfo:
    if not torch.cuda.is_available():
        raise RuntimeError("DeepSeek-V4 Flash trace scripts require CUDA")

    import torch.distributed as dist

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return RuntimeInfo(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=torch.device(f"cuda:{local_rank}"),
    )


def destroy_runtime() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def load_reference_model(
    *,
    model_dir: Path,
    config_path: Path,
    runtime: RuntimeInfo,
    max_batch_size: int = 1,
) -> tuple[ModuleType, torch.nn.Module, dict]:
    try:
        from safetensors.torch import load_model
    except ImportError as exc:
        raise RuntimeError(
            "safetensors is required for checkpoint loading"
        ) from exc

    ref_model = import_reference_model_module()
    torch.set_default_dtype(torch.bfloat16)
    config = load_model_config(config_path)
    args = build_model_args(
        ref_model,
        config,
        max_batch_size=max_batch_size,
    )
    configure_reference_globals(ref_model, args, runtime)

    with torch.device(runtime.device):
        model = ref_model.Transformer(args)

    shard_path = (
        model_dir / f"model{runtime.rank}-mp{runtime.world_size}.safetensors"
    )
    if not shard_path.exists():
        raise FileNotFoundError(
            f"missing DeepSeek-V4 Flash MP shard {shard_path}; the reference "
            "inference loader expects model{rank}-mp{world_size}.safetensors "
            "under --model-dir"
        )
    load_model(model, str(shard_path), strict=False)
    model.eval()
    return ref_model, model, config


def load_prompt_tokens(
    *,
    model_dir: Path,
    tokenizer_path: Optional[Path],
    prompt: str,
    min_tokens: int,
) -> list[int]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for prompt tokenization"
        ) from exc

    with prepend_sys_path(ENCODING_DIR):
        encoding = importlib.import_module("encoding_dsv4")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_dir)
    text = prompt
    tokens: list[int] = []
    while len(tokens) < min_tokens:
        encoded = encoding.encode_messages(
            [{"role": "user", "content": text}],
            thinking_mode="chat",
        )
        tokens = tokenizer.encode(encoded)
        if len(tokens) < min_tokens:
            text = f"{text}\n{prompt}"
    return tokens[:min_tokens]


def clone_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().clone()


def parse_layers(value: str) -> list[int]:
    layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not layers:
        raise ValueError("at least one layer id is required")
    return layers


def validate_representative_layers(config: dict, layers: Iterable[int]) -> None:
    ratios = config.get("compress_ratios")
    if ratios is None:
        raise ValueError("config is missing compress_ratios")
    actual = {layer: int(ratios[layer]) for layer in layers}
    required = {0, 4, 128}
    missing = required - set(actual.values())
    if missing:
        raise ValueError(
            "selected layers must cover pure SWA, C4, and C128 attention; "
            f"got layer ratios {actual}, missing ratios {sorted(missing)}"
        )


def ranked_output_path(path: Path, runtime: RuntimeInfo) -> Path:
    if runtime.world_size == 1:
        return path
    suffix = "".join(path.suffixes)
    if suffix:
        stem = str(path)[: -len(suffix)]
        return Path(f"{stem}.rank{runtime.rank}{suffix}")
    return Path(f"{path}.rank{runtime.rank}")


def window_cache_in_chronological_order(
    kv_cache: torch.Tensor,
    *,
    raw_end: int,
    window_size: int,
) -> torch.Tensor:
    active_tokens = min(max(0, int(raw_end)), int(window_size))
    if active_tokens == 0:
        return kv_cache[:, :0].contiguous()
    first_token = int(raw_end) - active_tokens
    slots = torch.arange(
        first_token,
        int(raw_end),
        device=kv_cache.device,
        dtype=torch.long,
    ) % int(window_size)
    return kv_cache[:, slots].contiguous()


def page_aligned_prefill_swa_kv(
    prefill_sparse_kv: torch.Tensor,
    *,
    raw_end: int,
    window_size: int,
    page_size_tokens: int,
) -> tuple[torch.Tensor, int]:
    """Return the page-aligned SWA tail expected by BatchGen's SWA manager.

    The reference model keeps an exact token-level ring window. BatchGen's SWA
    manager keeps the page-aligned tail so old pages can be released as a unit.
    During prefill, ``sparse_attn`` receives raw KV tokens before the optional
    compressed suffix, so this helper slices the raw prefix directly.
    """

    raw_end = int(raw_end)
    window_size = int(window_size)
    page_size_tokens = int(page_size_tokens)
    if raw_end <= 0:
        return prefill_sparse_kv[:, :0].contiguous(), 0
    if page_size_tokens <= 0:
        raise ValueError("page_size_tokens must be positive")
    if raw_end > int(prefill_sparse_kv.shape[1]):
        raise ValueError(
            "prefill sparse KV does not contain the requested raw token range: "
            f"raw_end={raw_end}, kv_len={prefill_sparse_kv.shape[1]}"
        )

    first_needed_token = max(0, raw_end - window_size)
    storage_start = (first_needed_token // page_size_tokens) * page_size_tokens
    return prefill_sparse_kv[
        :, storage_start:raw_end
    ].contiguous(), storage_start


class MultiLayerSparseAttentionHook:
    """Capture sparse attention inputs and output for selected layers."""

    def __init__(
        self,
        *,
        ref_model: ModuleType,
        model: torch.nn.Module,
        layer_ids: Iterable[int],
        phases: Iterable[str] = ("decode",),
    ) -> None:
        self.ref_model = ref_model
        self.model = model
        self.layer_ids = [int(layer_id) for layer_id in layer_ids]
        self.phases = set(phases)
        self.traces: list[SparseAttentionTrace] = []
        self.indexer_traces: dict[int, dict] = {}
        self.attention_forward_traces: dict[tuple[int, str], dict] = {}
        self._original_sparse_attn: Optional[Callable[..., torch.Tensor]] = None
        self._original_forwards: dict[int, Callable[..., torch.Tensor]] = {}
        self._original_indexer_trace_hooks: dict[int, object] = {}
        self._current_context: Optional[tuple[int, str, int, int]] = None

    def __enter__(self) -> "MultiLayerSparseAttentionHook":
        self._original_sparse_attn = self.ref_model.sparse_attn
        for layer_id in self.layer_ids:
            attention = self.model.layers[layer_id].attn
            original_forward = attention.forward
            self._original_forwards[layer_id] = original_forward

            def traced_forward(
                x: torch.Tensor,
                start_pos: int,
                *,
                _layer_id: int = layer_id,
                _forward: Callable[..., torch.Tensor] = original_forward,
            ) -> torch.Tensor:
                phase = "prefill" if int(start_pos) == 0 else "decode"
                previous = self._current_context
                self._current_context = (
                    _layer_id,
                    phase,
                    int(start_pos),
                    int(x.shape[1]),
                )
                try:
                    output = _forward(x, start_pos)
                    if phase in self.phases:
                        self.attention_forward_traces[(_layer_id, phase)] = {
                            "layer_id": int(_layer_id),
                            "phase": phase,
                            "start_pos": int(start_pos),
                            "seqlen": int(x.shape[1]),
                            "input": clone_to_cpu(x),
                            "output": clone_to_cpu(output),
                        }
                    return output
                finally:
                    self._current_context = previous

            attention.forward = traced_forward
            indexer = getattr(attention, "indexer", None)
            if indexer is not None:
                self._original_indexer_trace_hooks[layer_id] = getattr(
                    indexer, "trace_hook", None
                )

                def trace_indexer_decode(
                    *,
                    _layer_id: int = layer_id,
                    **payload,
                ) -> None:
                    start_pos = int(payload["start_pos"])
                    if start_pos == 0:
                        return
                    self.indexer_traces[_layer_id] = {
                        "layer_id": int(_layer_id),
                        "start_pos": start_pos,
                        "seqlen": int(payload["seqlen"]),
                        "offset": int(payload["offset"]),
                        "compress_ratio": int(payload["compress_ratio"]),
                        "compressed_tokens_after": int(
                            payload["compressed_tokens"]
                        ),
                        "index_topk": int(payload["index_topk"]),
                        "q": clone_to_cpu(payload["q"]),
                        "weights": clone_to_cpu(payload["weights"]),
                        "topk_idxs": clone_to_cpu(payload["topk_idxs"]),
                    }

                indexer.trace_hook = trace_indexer_decode

        def traced_sparse_attn(
            q: torch.Tensor,
            kv: torch.Tensor,
            attn_sink: torch.Tensor,
            topk_idxs: torch.Tensor,
            softmax_scale: float,
        ) -> torch.Tensor:
            assert self._original_sparse_attn is not None
            output = self._original_sparse_attn(
                q,
                kv,
                attn_sink,
                topk_idxs,
                softmax_scale,
            )
            if self._current_context is None:
                return output
            layer_id, phase, start_pos, seqlen = self._current_context
            if phase not in self.phases:
                return output
            attention = self.model.layers[layer_id].attn
            self.traces.append(
                SparseAttentionTrace(
                    layer_id=layer_id,
                    phase=phase,
                    start_pos=start_pos,
                    seqlen=seqlen,
                    compress_ratio=int(attention.compress_ratio),
                    window_size=int(attention.window_size),
                    q=clone_to_cpu(q),
                    kv=clone_to_cpu(kv),
                    attn_sink=clone_to_cpu(attn_sink),
                    topk_idxs=clone_to_cpu(topk_idxs),
                    softmax_scale=float(softmax_scale),
                    output=clone_to_cpu(output),
                )
            )
            return output

        self.ref_model.sparse_attn = traced_sparse_attn
        return self

    def __exit__(self, *exc_info: object) -> None:
        for layer_id, original_forward in self._original_forwards.items():
            self.model.layers[layer_id].attn.forward = original_forward
        for (
            layer_id,
            original_hook,
        ) in self._original_indexer_trace_hooks.items():
            self.model.layers[layer_id].attn.indexer.trace_hook = original_hook
        if self._original_sparse_attn is not None:
            self.ref_model.sparse_attn = self._original_sparse_attn


def trace_to_dict(trace: SparseAttentionTrace) -> dict:
    return {
        "layer_id": trace.layer_id,
        "phase": trace.phase,
        "start_pos": trace.start_pos,
        "seqlen": trace.seqlen,
        "compress_ratio": trace.compress_ratio,
        "window_size": trace.window_size,
        "q": trace.q,
        "kv_reference": trace.kv,
        "attn_sink": trace.attn_sink,
        "topk_idxs": trace.topk_idxs,
        "softmax_scale": trace.softmax_scale,
        "output": trace.output,
    }


def replay_sparse_attention(
    ref_model: ModuleType,
    layer_export: dict,
    kv: torch.Tensor,
    *,
    topk_idxs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    decode = layer_export["decode"]
    device = kv.device
    resolved_topk = decode["topk_idxs"] if topk_idxs is None else topk_idxs
    return ref_model.sparse_attn(
        decode["q"].to(device),
        kv.to(device),
        decode["attn_sink"].to(device),
        resolved_topk.to(device),
        float(decode["softmax_scale"]),
    )


def assert_close(
    actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float
) -> None:
    torch.testing.assert_close(
        actual.float().cpu(),
        expected.float().cpu(),
        atol=atol,
        rtol=rtol,
    )
