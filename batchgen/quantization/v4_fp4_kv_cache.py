"""FP4 KV cache quantization strategies.

Ported from sglang fp4_kv_cache_quant_method.py + kvfp4_tensor.py.
Provides two methods:
  - NVFP4KVMethod: two-level scaling (global FP32 + per-block FP8 E4M3), SM90+
  - BlockFP4KVMethod: block-wise single-level scaling (exponent-only), pure PyTorch

Three-player design:
  quant_method (pure compute)  ►  Pool (buffer + batch dequant)  ►  Backend (view adaptation)
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
E2M1_MAX = 6.0
MAX_BLOCK_SCALE_FP8 = 448.0

_device = "cuda" if torch.cuda.is_available() else "cpu"

E2M1_VALUES = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=torch.float32,
    device=_device,
)
E2M1_BOUNDS = torch.tensor(
    [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], dtype=torch.float32, device=_device
)


# ---------------------------------------------------------------------------
# Hardware helpers (replaces sglang.srt.utils)
# ---------------------------------------------------------------------------
def _cuda_sm_version() -> int:
    """Return SM version (e.g. 90, 100, 120) or 0 if no CUDA."""
    if not torch.cuda.is_available():
        return 0
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _is_sm90_supported() -> bool:
    return _cuda_sm_version() >= 90


def _is_sm100_supported() -> bool:
    return _cuda_sm_version() >= 100


# ---------------------------------------------------------------------------
# Quantize utilities (from kvfp4_tensor.py)
# ---------------------------------------------------------------------------
class BlockFP4KVQuantizeUtil:
    """Block-wise FP4 (E2M1) quantization for KV cache.

    Similar to MXFP4 but uses block_size=16.
    Each block of 16 elements shares one uint8 exponent-only scale factor.
    """

    @staticmethod
    @torch.compile
    def batched_quantize(tensor: Tensor) -> tuple[Tensor, Tensor]:
        """Quantize [B, M, N] → (packed [B, M, N/2], scales [B, M*N/16])."""
        b, m, n = tensor.shape
        reshaped = tensor.view(b, m * n // 16, 16)

        block_max = reshaped.abs().max(dim=-1, keepdim=True).values
        scale_exp = torch.ceil(
            torch.log2(torch.clamp(block_max / E2M1_MAX, min=1e-10))
        )
        scale_factors = (scale_exp + 127).squeeze(-1).to(torch.uint8)

        scaled = reshaped / torch.exp2(scale_exp)
        sign_bits = (scaled < 0).to(torch.uint8) << 3
        abs_vals = scaled.abs()

        magnitude_bits = torch.sum(
            abs_vals.unsqueeze(-1) >= E2M1_BOUNDS, dim=-1
        )
        fp4_vals = sign_bits + magnitude_bits.to(torch.uint8)

        fp4_reshaped = fp4_vals.view(b, m, n)
        packed = (fp4_reshaped[..., 1::2] << 4) + fp4_reshaped[..., 0::2]
        return packed, scale_factors

    @staticmethod
    @torch.compile
    def batched_dequantize(
        quant_tensor: Tensor,
        scale_factors: Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> Tensor:
        """Dequantize (packed [B, M, N/2], scales [B, M*N/16]) → [B, M, N]."""
        b, m, n_half = quant_tensor.shape
        n = n_half * 2

        fp4_vals = torch.empty(
            b, m, n, dtype=torch.uint8, device=quant_tensor.device
        )
        fp4_vals[..., 0::2] = quant_tensor & 0x0F
        fp4_vals[..., 1::2] = (quant_tensor >> 4) & 0x0F

        sign_mask = (fp4_vals & 0x08) != 0
        magnitude_idx = fp4_vals & 0x07
        float_vals = E2M1_VALUES[magnitude_idx.long()]
        float_vals = torch.where(sign_mask, -float_vals, float_vals)

        reshaped = float_vals.view(b, m * n // 16, 16)
        scale_exp = scale_factors.float() - 127
        scaled = reshaped * torch.exp2(scale_exp.unsqueeze(-1))
        return scaled.view(b, m, n).to(dtype)


class NVFP4KVQuantizeUtil:
    """NVFP4 two-level scaling: global FP32 + block FP8 E4M3.

    - Quantize: flashinfer ``nvfp4_kv_quantize`` (SM100+) or ``fp4_quantize`` (SM90)
    - Dequantize: flashinfer ``nvfp4_kv_dequantize`` (SM100+), PyTorch fallback (SM90)
    """

    @staticmethod
    def quantize(
        tensor: Tensor, global_scale: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Quantize BF16/FP16 → NVFP4.

        Returns (fp4_data[B,M,N/2], block_scales[B,M,N/16], global_scale).
        """
        assert (
            _is_sm90_supported()
        ), "NVFP4 KV cache quantize requires SM90+ GPU"

        b, m, n = tensor.shape
        tensor_2d = tensor.reshape(b * m, n)

        if isinstance(global_scale, (int, float)):
            global_scale = torch.tensor(
                [global_scale], dtype=torch.float32, device=tensor.device
            )
        elif global_scale.dim() == 0:
            global_scale = global_scale.unsqueeze(0)

        if _is_sm100_supported():
            from flashinfer import nvfp4_kv_quantize

            fp4_2d, scales_2d = nvfp4_kv_quantize(tensor_2d, global_scale)
        else:
            from flashinfer import fp4_quantize

            global_scale_inv = 1.0 / global_scale
            fp4_2d, scales_2d = fp4_quantize(
                tensor_2d,
                global_scale_inv,
                sf_vec_size=16,
                sf_use_ue8m0=False,
                is_sf_swizzled_layout=False,
                is_sf_8x4_layout=False,
                enable_pdl=None,
            )

        fp4_data = fp4_2d.view(b, m, fp4_2d.shape[-1])
        block_scales = scales_2d.view(b, m, scales_2d.shape[-1]).view(
            torch.float8_e4m3fn
        )
        return fp4_data, block_scales, global_scale

    @staticmethod
    def dequantize(
        quant_tensor: Tensor,
        block_scales: Tensor,
        global_scale: Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> Tensor:
        """Dequantize NVFP4 → BF16/FP16."""
        b, m, n_half = quant_tensor.shape

        if isinstance(global_scale, (int, float)):
            global_scale = torch.tensor(
                [global_scale], dtype=torch.float32, device=quant_tensor.device
            )
        elif global_scale.dim() == 0:
            global_scale = global_scale.unsqueeze(0)

        if _is_sm100_supported():
            from flashinfer import nvfp4_kv_dequantize

            quant_2d = quant_tensor.view(torch.uint8).reshape(b * m, n_half)
            scales_2d = block_scales.view(torch.uint8).reshape(b * m, -1)
            output_2d = nvfp4_kv_dequantize(
                quant_2d, scales_2d, global_scale, output_dtype=dtype
            )
            return output_2d.reshape(b, m, -1)
        else:
            # Pure PyTorch fallback for SM90
            n = n_half * 2
            fp4_vals = torch.empty(
                b, m, n, dtype=torch.uint8, device=quant_tensor.device
            )
            fp4_vals[..., 0::2] = quant_tensor & 0x0F
            fp4_vals[..., 1::2] = (quant_tensor >> 4) & 0x0F
            float_vals = E2M1_VALUES[fp4_vals.long()]
            reshaped = float_vals.view(b, m * n // 16, 16)
            block_scales_float = block_scales.float().unsqueeze(-1)
            scaled = reshaped * block_scales_float
            return (scaled.view(b, m, n) * global_scale).to(dtype)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
class FP4KVCacheQuantMethod(ABC):
    """Abstract base for FP4 KV cache quantization strategies.

    Owns the quantize/dequantize computation.  The Pool owns the buffers and
    orchestrates the batch dequant loop.  Backends only do view/reshape.
    """

    name: str
    SCALE_BLOCK_SIZE: int = 1

    def needs_dequant_workspace(self) -> bool:
        return False

    def needs_global_scale(self) -> bool:
        return False

    @abstractmethod
    def create_buffers(
        self,
        size: int,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
    ) -> dict: ...

    @abstractmethod
    def quantize_and_store(
        self,
        k_buffer: Tensor,
        v_buffer: Tensor,
        k_scale_buffer: Optional[Tensor],
        v_scale_buffer: Optional[Tensor],
        loc: Tensor,
        cache_k: Tensor,
        cache_v: Tensor,
        k_scale=None,
        v_scale=None,
    ) -> None: ...

    @abstractmethod
    def dequantize_prev_kv(
        self,
        k_fp4: Tensor,
        k_scales: Tensor,
        v_fp4: Tensor,
        v_scales: Tensor,
        layer_id: int,
    ) -> tuple[Tensor, Tensor]: ...

    @abstractmethod
    def compute_cell_size(
        self, head_num: int, head_dim: int, num_layers: int, kv_size: int
    ) -> int: ...

    def load_scales_from_model(
        self, model_runner, sm_version: int = None
    ) -> None:
        """Load per-layer global scales from model weights (no-op by default)."""
        pass


# ---------------------------------------------------------------------------
# NVFP4 (two-level scaling)
# ---------------------------------------------------------------------------
class NVFP4KVMethod(FP4KVCacheQuantMethod):
    """NVFP4 two-level scaling: global FP32 + per-block FP8 E4M3.

    Supported on SM100 and SM120.
    """

    name = "nvfp4"
    SCALE_BLOCK_SIZE = 16

    def __init__(self, num_layers: int, device: str, sm_version: int = 120):
        self.num_layers = num_layers
        self.device = device
        self.sm_version = sm_version
        self.k_scales_gpu = torch.ones(
            num_layers, dtype=torch.float32, device=device
        )
        self.v_scales_gpu = torch.ones(
            num_layers, dtype=torch.float32, device=device
        )

    def needs_dequant_workspace(self) -> bool:
        return True

    def needs_global_scale(self) -> bool:
        return True

    # -- Scale management (replaces sglang model_runner integration) ----------

    def set_layer_scales(
        self, layer_id: int, k_scale: float = 1.0, v_scale: float = 1.0
    ) -> None:
        """Directly set per-layer global scales (batchgen-native API).

        For SM100, multiply by E2M1_MAX (6.0) to bridge the TRT-LLM XQA gap.
        """
        if self.sm_version == 100:
            k_scale *= E2M1_MAX
            v_scale *= E2M1_MAX
        self.k_scales_gpu[layer_id] = k_scale
        self.v_scales_gpu[layer_id] = v_scale

    def load_scales_from_model(
        self, model_runner, sm_version: int = None
    ) -> None:
        """Load per-layer scales from model.

        NOTE: sglang-specific model traversal stripped.  Use ``set_layer_scales``
        for batchgen, or override this method with model-specific logic.
        """
        if sm_version is not None:
            self.sm_version = sm_version

    # -- Buffer management ---------------------------------------------------

    def create_buffers(
        self,
        size: int,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
    ) -> dict:
        m, n, k = size, head_num, head_dim
        store_dtype = torch.uint8
        dq_dtype = torch.float8_e4m3fn

        k_buffer = [
            torch.zeros((m, n, k // 2), dtype=store_dtype, device=device)
            for _ in range(layer_num)
        ]
        v_buffer = [
            torch.zeros((m, n, k // 2), dtype=store_dtype, device=device)
            for _ in range(layer_num)
        ]
        k_scale_buffer = [
            torch.zeros(
                (m, n, k // self.SCALE_BLOCK_SIZE),
                dtype=store_dtype,
                device=device,
            )
            for _ in range(layer_num)
        ]
        v_scale_buffer = [
            torch.zeros(
                (m, n, k // self.SCALE_BLOCK_SIZE),
                dtype=store_dtype,
                device=device,
            )
            for _ in range(layer_num)
        ]
        dq_k_buffer = torch.zeros((m, n, k), dtype=dq_dtype, device=device)
        dq_v_buffer = torch.zeros((m, n, k), dtype=dq_dtype, device=device)

        return {
            "k_buffer": k_buffer,
            "v_buffer": v_buffer,
            "k_scale_buffer": k_scale_buffer,
            "v_scale_buffer": v_scale_buffer,
            "dq_k_buffer": dq_k_buffer,
            "dq_v_buffer": dq_v_buffer,
            "store_dtype": store_dtype,
        }

    def quantize_and_store(
        self,
        k_buffer: Tensor,
        v_buffer: Tensor,
        k_scale_buffer: Optional[Tensor],
        v_scale_buffer: Optional[Tensor],
        loc: Tensor,
        cache_k: Tensor,
        cache_v: Tensor,
        k_scale=None,
        v_scale=None,
    ) -> None:
        cache_k, cache_k_fp4_sf, _ = NVFP4KVQuantizeUtil.quantize(
            cache_k.contiguous(), k_scale
        )
        cache_v, cache_v_fp4_sf, _ = NVFP4KVQuantizeUtil.quantize(
            cache_v.contiguous(), v_scale
        )
        k_buffer[loc] = cache_k.view(torch.uint8)
        v_buffer[loc] = cache_v.view(torch.uint8)
        k_scale_buffer[loc] = cache_k_fp4_sf.view(torch.uint8)
        v_scale_buffer[loc] = cache_v_fp4_sf.view(torch.uint8)

    def dequantize_prev_kv(
        self,
        k_fp4: Tensor,
        k_scales: Tensor,
        v_fp4: Tensor,
        v_scales: Tensor,
        layer_id: int,
    ) -> tuple[Tensor, Tensor]:
        cur_k_scale = self.k_scales_gpu[layer_id : layer_id + 1]
        cur_v_scale = self.v_scales_gpu[layer_id : layer_id + 1]
        k_bf16 = NVFP4KVQuantizeUtil.dequantize(
            k_fp4.view(torch.uint8), k_scales, cur_k_scale
        )
        v_bf16 = NVFP4KVQuantizeUtil.dequantize(
            v_fp4.view(torch.uint8), v_scales, cur_v_scale
        )
        return k_bf16.to(torch.float8_e4m3fn), v_bf16.to(torch.float8_e4m3fn)

    def compute_cell_size(
        self, head_num: int, head_dim: int, num_layers: int, kv_size: int
    ) -> int:
        fp4_size = head_num * (head_dim // 2) * num_layers * 2 * kv_size
        scale_size = (
            head_num
            * (head_dim // self.SCALE_BLOCK_SIZE)
            * num_layers
            * 2
            * kv_size
        )
        dq_size = head_num * head_dim * 2 * kv_size
        return fp4_size + scale_size + dq_size


# ---------------------------------------------------------------------------
# BlockFP4 (single-level scaling)
# ---------------------------------------------------------------------------
class BlockFP4KVMethod(FP4KVCacheQuantMethod):
    """Block-wise FP4 single-level scaling (similar to MXFP4 but block_size=16)."""

    name = "blockfp4"
    SCALE_BLOCK_SIZE = 16

    def needs_dequant_workspace(self) -> bool:
        return True

    def create_buffers(
        self,
        size: int,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
    ) -> dict:
        m = size
        store_dtype = torch.uint8
        dq_dtype = torch.float8_e4m3fn

        k_buffer = [
            torch.zeros(
                (m, head_num, head_dim // 2), dtype=store_dtype, device=device
            )
            for _ in range(layer_num)
        ]
        v_buffer = [
            torch.zeros(
                (m, head_num, head_dim // 2), dtype=store_dtype, device=device
            )
            for _ in range(layer_num)
        ]
        k_scale_buffer = [
            torch.zeros(
                (m, (head_num * head_dim) // self.SCALE_BLOCK_SIZE),
                dtype=store_dtype,
                device=device,
            )
            for _ in range(layer_num)
        ]
        v_scale_buffer = [
            torch.zeros(
                (m, (head_num * head_dim) // self.SCALE_BLOCK_SIZE),
                dtype=store_dtype,
                device=device,
            )
            for _ in range(layer_num)
        ]
        dq_k_buffer = torch.zeros(
            (m, head_num, head_dim), dtype=dq_dtype, device=device
        )
        dq_v_buffer = torch.zeros(
            (m, head_num, head_dim), dtype=dq_dtype, device=device
        )

        return {
            "k_buffer": k_buffer,
            "v_buffer": v_buffer,
            "k_scale_buffer": k_scale_buffer,
            "v_scale_buffer": v_scale_buffer,
            "dq_k_buffer": dq_k_buffer,
            "dq_v_buffer": dq_v_buffer,
            "store_dtype": store_dtype,
        }

    def quantize_and_store(
        self,
        k_buffer,
        v_buffer,
        k_scale_buffer,
        v_scale_buffer,
        loc,
        cache_k,
        cache_v,
        k_scale=None,
        v_scale=None,
    ) -> None:
        cache_k_fp4, cache_k_sf = BlockFP4KVQuantizeUtil.batched_quantize(
            cache_k
        )
        cache_v_fp4, cache_v_sf = BlockFP4KVQuantizeUtil.batched_quantize(
            cache_v
        )
        k_buffer[loc] = cache_k_fp4
        v_buffer[loc] = cache_v_fp4
        k_scale_buffer[loc] = cache_k_sf
        v_scale_buffer[loc] = cache_v_sf

    def dequantize_prev_kv(
        self,
        k_fp4: Tensor,
        k_scales: Tensor,
        v_fp4: Tensor,
        v_scales: Tensor,
        layer_id: int,
    ) -> tuple[Tensor, Tensor]:
        k_bf16 = BlockFP4KVQuantizeUtil.batched_dequantize(k_fp4, k_scales)
        v_bf16 = BlockFP4KVQuantizeUtil.batched_dequantize(v_fp4, v_scales)
        return k_bf16.to(torch.float8_e4m3fn), v_bf16.to(torch.float8_e4m3fn)

    def compute_cell_size(
        self, head_num: int, head_dim: int, num_layers: int, kv_size: int
    ) -> int:
        fp4_size = head_num * (head_dim // 2) * num_layers * 2 * kv_size
        scale_size = (
            (head_num * head_dim // self.SCALE_BLOCK_SIZE)
            * num_layers
            * 2
            * kv_size
        )
        dq_size = head_num * head_dim * 2 * kv_size
        return fp4_size + scale_size + dq_size


# ---------------------------------------------------------------------------
# Registry + factory
# ---------------------------------------------------------------------------
FP4_KV_CACHE_QUANT_REGISTRY: dict[str, type[FP4KVCacheQuantMethod]] = {
    "nvfp4": NVFP4KVMethod,
    "blockfp4": BlockFP4KVMethod,
}


def get_fp4_kv_cache_quant_method(name: str, **kwargs) -> FP4KVCacheQuantMethod:
    """Instantiate a FP4KVCacheQuantMethod by recipe name."""
    if name not in FP4_KV_CACHE_QUANT_REGISTRY:
        raise ValueError(
            f"Unknown fp4_kv_cache_recipe: '{name}'. "
            f"Available: {list(FP4_KV_CACHE_QUANT_REGISTRY)}"
        )
    return FP4_KV_CACHE_QUANT_REGISTRY[name](**kwargs)
