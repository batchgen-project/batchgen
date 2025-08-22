# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
#                                                                               #
#  you may obtain a copy of the license at                                      #
#                                                                               #
#                  http://www.apache.org/licenses/license-2.0                   #
#                                                                               #
#  unless required by applicable law or agreed to in writing, software          #
#  distributed under the license is distributed on an "as is" basis,            #
#  without warranties or conditions of any kind, either express or implied.     #
#  see the license for the specific language governing permissions and          #
#  limitations under the license.                                               #
# ---------------------------------------------------------------------------- #

import logging
import math
import os
import time
import types
from multiprocessing import Process
from typing import Optional, Union, List

import torch
import torch.nn as nn
import triton
import triton.language as tl
from safetensors.torch import load_file
from tqdm import tqdm, trange
from transformers import AutoConfig
import torch.distributed as dist

# from batchgen.config import EngineConfig, ModelConfig
from ....config.config import EngineConfig, ModelConfig
from .configuration_deepseek_v3 import DeepseekV3Config
from datetime import timedelta

# from transformers.models.qwen2_moe.modeling_qwen2_moe import repeat_kv

try:
    from batchgen.core_engine import batchgen as core_engine
except ImportError:
    # jit compile
    from core_engine import batchgen as core_engine

from typing import Tuple

try:
    import deep_gemm
    from deep_gemm import get_col_major_tma_aligned_tensor
except ImportError:
    pass

from batchgen.models.Wrapper import Attn_Wrapper, Expert_Wrapper
# from flash_mla import get_mla_metadata, flash_mla_with_kvcache

# current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.append(current_dir)
from .modeling_deepseek_v3 import (
    DeepseekV3ForCausalLM,
    apply_rotary_pos_emb,
    rotate_half,
)

from ....config.engine_config_parser import parse_config_from_json

def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.

    Args:
        x: the dividend.
        y: the divisor.

    Returns:
        The result of the ceiling division.
    """
    return (x + y - 1) // y

def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(
        torch.float8_e4m3fn
    ).view(m, n), (x_amax / 448.0).view(m, -1)


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (ceil_div(m, 128) * 128, ceil_div(n, 128) * 128),
        dtype=x.dtype,
        device=x.device,
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (
        x_amax / 448.0
    ).view(x_view.size(0), x_view.size(2))


def w8a16_gemm(
    weight_data_fp8: torch.Tensor,
    weight_scale_inv_fp32: torch.Tensor,
    activation_bf16: torch.Tensor,
) -> torch.Tensor:
    """
    activation_bf16: [n_group, m, k]
    weight_data_fp8: [m, n]
    weight_scale_inv_fp32: [m, n]
    """
    assert weight_data_fp8.dim() == 2
    assert activation_bf16.dim() == 3
    n_group, m, k = activation_bf16.size()
    out = torch.empty_like(activation_bf16, dtype=torch.bfloat16)
    y_fp8 = (weight_data_fp8, weight_scale_inv_fp32)
    for i in range(n_group):
        x_fp8 = per_token_cast_to_fp8(activation_bf16[i])
        x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
        deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out[i])
    return out


def attn_fp8(
    Q_bf16: torch.Tensor,
    K_bf16: torch.Tensor,
    V_bf16: torch.Tensor,
):
    """
    Q_bf16: [bsz, num_heads, q_len, q_head_dim]
    K_bf16: [bsz, num_heads, kv_seq_len, q_head_dim]
    V_bf16: [bsz, num_heads, kv_seq_len, v_head_dim]
    """
    # Convert QKV to 3D tensor
    bsz, num_heads, q_len, q_head_dim = Q_bf16.size()
    _, _, kv_seq_len, _ = K_bf16.size()
    _, _, _, v_head_dim = V_bf16.size()
    Q_bf16 = Q_bf16.view(bsz * num_heads, q_len, q_head_dim)
    K_bf16 = K_bf16.view(bsz * num_heads, kv_seq_len, q_head_dim)
    V_bf16 = V_bf16.view(bsz * num_heads, kv_seq_len, v_head_dim)

    # QKT
    Q_fp8 = (
        torch.empty_like(Q_bf16, dtype=torch.float8_e4m3fn),
        torch.empty(
            (bsz * num_heads, q_len, q_head_dim // 128),
            device=Q_bf16.device,
            dtype=torch.float,
        ),
    )
    K_fp8 = (
        torch.empty_like(K_bf16, dtype=torch.float8_e4m3fn),
        torch.empty(
            (bsz * num_heads, kv_seq_len, q_head_dim // 128),
            device=K_bf16.device,
            dtype=torch.float,
        ),
    )
    attn_weights = torch.empty(
        (bsz * num_heads, q_len, kv_seq_len),
        device=Q_bf16.device,
        dtype=torch.bfloat16,
    )
    for i in range(bsz * num_heads):
        Q_fp8[0][i], Q_fp8[1][i] = per_token_cast_to_fp8(Q_bf16[i])
        K_fp8[0][i], K_fp8[1][i] = per_block_cast_to_fp8(K_bf16[i])
    K_fp8 = (K_fp8[0], get_col_major_tma_aligned_tensor(K_fp8[1]))
    m_indices = torch.arange(
        0, num_groups, device=Q_bf16.device, dtype=torch.int
    )
    m_indices = (
        m_indices.unsqueeze(-1).expand(num_groups, m).contiguous().view(-1)
    )
    deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
        Q_fp8, K_fp8, attn_weights, m_indices
    )

    attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(Q_bf16.dtype)

    # A@V
    V_fp8 = (
        torch.empty_like(V_bf16, dtype=torch.float8_e4m3fn),
        torch.empty(
            (bsz * num_heads, kv_seq_len, v_head_dim // 128),
            device=V_bf16.device,
            dtype=torch.float,
        ),
    )
    attn_weights_fp8 = (
        torch.empty_like(attn_weights, dtype=torch.float8_e4m3fn),
        torch.empty(
            (bsz * num_heads, q_len, kv_seq_len // 128),
            device=attn_weights.device,
            dtype=torch.float,
        ),
    )
    attn_output = torch.empty(
        (bsz * num_heads, q_len, v_head_dim),
        device=Q_bf16.device,
        dtype=torch.bfloat16,
    )
    for i in range(bsz * num_heads):
        V_fp8[0][i], V_fp8[1][i] = per_token_cast_to_fp8(V_bf16[i])
        attn_weights_fp8[0][i], attn_weights_fp8[1][i] = per_block_cast_to_fp8(
            attn_weights[i]
        )
    V_fp8 = (V_fp8[0], get_col_major_tma_aligned_tensor(V_fp8[1]))
    deep_gemm.gemm_fp8_fp8_bf16_nt(V_fp8, attn_weights_fp8, attn_output)

    attn_output = attn_output.view(bsz, num_heads, q_len, v_head_dim)
    return attn_output


@torch.no_grad()
def prefill_attn_fp8(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    weight_dequant_scale: dict,
):
    bsz, q_len, _ = hidden_states.size()
    if self.q_lora_rank is None:
        q = w8a16_gemm(
            self.q_proj.weight.data,
            weight_dequant_scale["q_proj.weight_scale_inv"],
            hidden_states,
        )
    else:
        q = w8a16_gemm(
            self.q_b_proj.weight.data,
            weight_dequant_scale["q_b_proj.weight_scale_inv"],
            self.q_a_layernorm(
                w8a16_gemm(
                    self.q_a_proj.weight.data,
                    weight_dequant_scale["q_a_proj.weight_scale_inv"],
                    hidden_states,
                )
            ),
        )
    q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(
        q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    )
    compressed_kv = w8a16_gemm(
        self.kv_a_proj_with_mqa.weight.data,
        weight_dequant_scale["kv_a_proj_with_mqa.weight_scale_inv"],
        hidden_states,
    )
    new_compressed_kv = compressed_kv.clone()
    kv_len = compressed_kv.size(1)
    assert kv_len == attention_mask.size(-1)
    compressed_kv, k_pe = torch.split(
        compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )
    k_pe = k_pe.view(bsz, kv_len, 1, self.qk_rope_head_dim).transpose(1, 2)
    kv = w8a16_gemm(
        self.kv_b_proj.weight.data,
        weight_dequant_scale["kv_b_proj.weight_scale_inv"],
        self.kv_a_layernorm(compressed_kv),
    )
    kv = kv.view(
        bsz, kv_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
    ).transpose(1, 2)
    k_nope, value_states = torch.split(
        kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
    )

    kv_seq_len = attention_mask.shape[-1]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

    query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
    query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

    key_states = k_pe.new_empty(
        bsz, self.num_heads, kv_seq_len, self.q_head_dim
    )
    key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
    key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

    attn_output = attn_fp8(query_states, key_states, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(
        bsz, q_len, self.num_heads * self.v_head_dim
    )
    attn_output = w8a16_gemm(
        self.o_proj.weight.data,
        weight_dequant_scale["o_proj.weight_scale_inv"],
        attn_output,
    )
    return attn_output, new_compressed_kv


def compressed_kv_fp8_to_bf16_per_token(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    Dequantize the output of bf16_to_fp8_per_token back to BF16.
    Inputs:
      q     [bsz, seq, 576]  dtype=float8_e4m3fn
      scale [bsz, seq]       dtype=float32
    Returns:
      x_bf16 [bsz, seq, 576] dtype=bfloat16
    """
    bsz, seq_len, dim = q.shape
    M = bsz * seq_len
    # flatten
    q_flat = q.view(M, dim).float()               # upcast FP8→FP32
    x_rec = q_flat * s.view(M, 1)             # rescale
    return x_rec.to(torch.bfloat16).view(bsz, seq_len, dim)	

def deepseek_v3_dequantization(
    weight_data_fp8: torch.Tensor,
    weight_scale_inv_fp32: torch.Tensor,
    block_size=(128, 128),
) -> torch.Tensor:
    """
    Vectorized dequantization that removes Python-level loops
    and leverages PyTorch's parallelism.
    """
    rows, cols = weight_data_fp8.shape
    block_rows, block_cols = block_size

    # Number of blocks in each dimension
    n_block_rows = rows // block_rows
    n_block_cols = cols // block_cols

    # 1) Reshape weight data into 4D block form and cast to float32
    #    shape becomes [n_block_rows, block_rows, n_block_cols, block_cols].
    weight_4d = weight_data_fp8.reshape(
        n_block_rows, block_rows, n_block_cols, block_cols
    ).to(torch.float32)

    # 2) Broadcast scale into 4D by unsqueezing along the second and fourth dimensions.
    #    shape becomes [n_block_rows, 1, n_block_cols, 1].
    scale_4d = weight_scale_inv_fp32.unsqueeze(1).unsqueeze(-1)

    # 3) Multiply once using broadcasting
    dequantized_4d = weight_4d * scale_4d

    # 4) Reshape back to [rows, cols] and cast to bfloat16
    dequantized_weight = dequantized_4d.reshape(rows, cols).to(torch.bfloat16)

    return dequantized_weight


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Dequantizes weights using the provided scaling factors and stores the result.

    Args:
        x_ptr (tl.pointer): Pointer to the quantized weights.
        s_ptr (tl.pointer): Pointer to the scaling factors.
        y_ptr (tl.pointer): Pointer to the output buffer for dequantized weights.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): Size of the block for tiling.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M, N).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    assert (
        x.is_contiguous() and s.is_contiguous()
    ), "Input tensors must be contiguous"
    assert x.dim() == 2 and s.dim() == 2, "Input tensors must have 2 dimensions"
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.bfloat16)
    grid = lambda meta: (  # noqa E731
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


class DeepSeekV3_Initializer:
    def __init__(
        self,
        huggingface_ckpt_name: str,
        hf_cache_dir: str,
        cache_dir: Optional[str],
        engine_config,
        skeleton_state_dict: Optional[dict],
        shm_name: str,
        tensor_meta_shm_name: str,
        pt_ckpt_dir,
        host_kv_cache_size: Optional[int] = None,
        local_rank: Optional[int] = 0,
        global_rank: Optional[int] = 0,
        world_size: Optional[int] = 1,
    ):
        self.huggingface_ckpt_name = huggingface_ckpt_name
        self.cache_dir = cache_dir
        self.pt_ckpt_dir = pt_ckpt_dir
        self.engine_config = engine_config
        self.skeleton_state_dict = skeleton_state_dict
        self.model = None
        self.hf_model_config = DeepseekV3Config.from_pretrained(
            self.cache_dir,
            # huggingface_ckpt_name,
            # cache_dir=hf_cache_dir,
            # trust_remote_code=True,
        )
        self.hf_model_config._name_or_path = huggingface_ckpt_name
        self.hf_model_config.architectures = ["DeepseekV3ForCausalLM"]

        self.host_kv_cache_size = host_kv_cache_size
        self.host_kv_cache_byte_size = host_kv_cache_size * (1024**3)

        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size    

        self.fp8_weights_IPC_handle = {}

        # TODO:
        self.model_config = self._parse_model_config()
        self._default_engine_config()
        self.state_dict_name_map = {}
        self.weight_copy_task = {}

        self.shm_name = shm_name
        self.tensor_meta_shm_name = tensor_meta_shm_name

    def _default_engine_config(self):
        props = torch.cuda.get_device_properties(
            self.engine_config.Basic_Config.device
        )
        total_memory = props.total_memory / (1024**3)
        logging.info(f"Current device total memory: {total_memory} GB")

        # self.engine_config.Basic_Config.log_level = "info"
        # self.engine_config.Basic_Config.torch_dtype = torch.float8_e4m3fn
        # self.engine_config.Basic_Config.dtype_str = "float8_e4m3fn"
        # self.engine_config.Basic_Config.attn_mode = 1
        # self.engine_config.Basic_Config.module_types = [
        #     "attn",
        #     "routed_expert",
        #     "shared_expert",
        # ]
        self.engine_config.Basic_Config.num_threads = 16

        # Determine the number of host kv slots.
        self.engine_config.KV_Storage_Config.reserved_length = (
            self.engine_config.Basic_Config.padding_length
            + self.engine_config.Basic_Config.max_decoding_length
        )
        self.engine_config.KV_Storage_Config.slot_byte_size = (
            self.engine_config.KV_Storage_Config.reserved_length
            * self.model_config.compressed_kv_dim
        )  # KV saved in FP8
        self.engine_config.KV_Storage_Config.num_host_slots = (
            self.host_kv_cache_byte_size
            // self.engine_config.KV_Storage_Config.slot_byte_size
            // self.model_config.num_hidden_layers
        )
        self.engine_config.KV_Storage_Config.storage_byte_size = (
            self.host_kv_cache_byte_size
        )
        # logging.info(
        #     f"KV storage byte size: {self.engine_config.KV_Storage_Config.storage_byte_size}"
        # )
        logging.info(
            f"Number of host kv slots: {self.engine_config.KV_Storage_Config.num_host_slots}"
        )

        # self.engine_config.Module_Batching_Config.global_batch_size = min(
        #     self.engine_config.KV_Storage_Config.num_host_slots,
        #     self.engine_config.Basic_Config.num_queries,
        # )
        # context_length = (
        #     self.engine_config.Basic_Config.max_decoding_length
        #     + self.engine_config.Basic_Config.padding_length
        # )
        # if context_length > 768:
        #     self.engine_config.Module_Batching_Config.attn_prefill_micro_batch_size = max(1, math.floor(
        #         10 * 768 / context_length)
        #     )
        #     self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size = max(1, math.floor(
        #         20 * 768 / context_length)
        #     )
        # else:
        #     self.engine_config.Module_Batching_Config.attn_prefill_micro_batch_size = 10
        #     self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size = 20
        # self.engine_config.Module_Batching_Config.attn_prefill_micro_batch_size = 4
        # self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size = 8
        # logging.info(
        #     f"attn_prefill_micro_batch_size: {self.engine_config.Module_Batching_Config.attn_prefill_micro_batch_size}"
        # )
        # logging.info(
        #     f"MoE_prefill_micro_batch_size: {self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size}"
        # )
        # self.engine_config.Module_Batching_Config.expert_prefill_batch_size_upper_bound = 2048

        # if context_length > 768:
        #     self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size = max(1, math.floor(
        #         120 * 768 / context_length)
        #     )
        # else:
        #     self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size = 100
        # self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size = 32
        # logging.info(
        #     f"attn_decoding_micro_batch_size: {self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size}"
        # )
        # self.engine_config.Module_Batching_Config.MoE_decoding_micro_batch_size = self.engine_config.Module_Batching_Config.global_batch_size
        # self.engine_config.Module_Batching_Config.expert_decoding_batch_size_upper_bound = 2048

        # L40
        # if total_memory >= 47 and total_memory < 49:
        #     self.engine_config.Module_Batching_Config.attn_prefill_micro_batch_size *= 2
        #     self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size *= 2
        #     self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size *= 2
        # if total_memory >= 78:
        #     self.engine_config.Module_Batching_Config.attn_prefill_micro_batch_size *= 3
        #     self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size *= 3
        #     self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size *= 3

        # self.engine_config.GPU_Buffer_Config.num_prefill_module_buffer = {
        #     "attn": 1,
        #     "routed_expert": 20,
        #     "shared_expert": 1,
        # }
        # self.engine_config.GPU_Buffer_Config.num_decoding_module_buffer = {
        #     "attn": 1,
        #     "routed_expert": 128,
        #     "shared_expert": 1,
        # }
        # if total_memory > 32:
        #     self.engine_config.GPU_Buffer_Config.num_decoding_module_buffer = {
        #         "attn": 1,
        #         "routed_expert": 12,
        #         "shared_expert": 1,
        #     }

        # self.engine_config.GPU_Buffer_Config.num_k_buffer = 1
        # self.engine_config.GPU_Buffer_Config.num_v_buffer = 0
        self.engine_config.GPU_Buffer_Config.kv_buffer_num_tokens = (
            self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size
            * (
                self.engine_config.Basic_Config.max_decoding_length
                + self.engine_config.Basic_Config.padding_length
            )
        )
        self.engine_config.GPU_Buffer_Config.module_shapes = {
            "attn": {
                "q_a_proj.weight": [1536, 7168],
                "q_a_layernorm.weight": [1536],
                "q_b_proj.weight": [24576, 1536],
                "kv_a_proj_with_mqa.weight": [576, 7168],
                "kv_a_layernorm.weight": [512],
                "kv_b_proj.weight": [32768, 512],
                "o_proj.weight": [7168, 16384],
            },
            "routed_expert": {
                "gate_proj.weight": [2048, 7168],
                "up_proj.weight": [2048, 7168],
                "down_proj.weight": [7168, 2048],
            },
            "shared_expert": {
                "gate_proj.weight": [2048, 7168],
                "up_proj.weight": [2048, 7168],
                "down_proj.weight": [7168, 2048],
            },
        }

    def _parse_model_config(self):
        model_config = ModelConfig()

        model_config.model_type = "deepseek_v3"
        model_config.num_hidden_layers = 61
        model_config.num_local_experts = 256
        model_config.num_attention_heads = 128
        model_config.num_key_value_heads = 128
        model_config.head_dim = 192
        model_config.compressed_kv_dim = 576
        return model_config

    def _save_safetensors_to_pt(self):
        ckpt_files = os.listdir(self.cache_dir)
        ckpt_files = [
            os.path.join(self.cache_dir, ckpt)
            for ckpt in ckpt_files
            if ckpt.endswith(".safetensors")
        ]

        def save_and_load(file_path, save_dir):
            tensor_dict = load_file(file_path)
            torch.save(tensor_dict, save_dir)
            return tensor_dict

        processes = []
        for ckpt in tqdm(
            ckpt_files, desc="Loading checkpoint files", smoothing=0
        ):
            dst_dir = os.path.join(
                self.pt_ckpt_dir,
                ckpt.split("/")[-1].replace(".safetensors", ".pt"),
            )
            if os.path.exists(dst_dir):
                continue

            p = Process(target=save_and_load, args=(ckpt, dst_dir))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            p.close()
        logging.info("All safetensor loader processes joined")

    def _extract_dequantize_scale(self):
        self.dequant_scale = {}
        for key, param in self.skeleton_state_dict.items():
            if "weight_scale_inv" in key:
                self.dequant_scale[key] = param

    def _load_model_skeleton(self):
        for key, param in self.model.named_parameters():
            if key in self.skeleton_state_dict:
                dequant_key = key + "_scale_inv"
                if dequant_key in self.dequant_scale:
                    param.data = deepseek_v3_dequantization(
                        self.skeleton_state_dict[key],
                        self.dequant_scale[dequant_key],
                    )
                else:
                    param.data = self.skeleton_state_dict[key]

        model_skeletion_byte_size = (
            sum(p.numel() * p.element_size() for p in self.model.parameters())
            / (1024**3)
        )
        logging.info(f"Model skeleton size: {model_skeletion_byte_size:.2f} GB")

        # Rank 0 print out all the tensors in the model with tensor size in MB
        # if dist.get_rank() == 0:
        #     logging.info("Model skeleton tensors:")
        #     for name, param in self.model.named_parameters():
        #         tensor_size_mb = (
        #             param.numel() * param.element_size() / (1024**2)
        #         )
        #         logging.info(
        #             f"{name}: {tensor_size_mb:.2f} MB, dtype: {param.dtype}"
        #         )
        #     for name, buffer in self.model.named_buffers():
        #         tensor_size_mb = (
        #             buffer.numel() * buffer.element_size() / (1024**2)
        #         )
        #         logging.info(
        #             f"{name}: {tensor_size_mb:.2f} MB, dtype: {buffer.dtype}"
        #         )
        # dist.barrier()

    def Init(self):
        try:
            # self._init_torch_dist_nccl()
            torch.cuda.set_device(self.local_rank)
            # self._parse_state_dict_ep()
            # logging.info("State dict parsed")
            self.core_engine = core_engine(
                self.engine_config, self.model_config
            )
            logging.info("Core engine created")
            logging.info(f"_name_or_path: {self.hf_model_config._name_or_path}")
            if (
                "deepseek-ai/DeepSeek-V3" in self.hf_model_config._name_or_path
                or "deepseek-ai/DeepSeek-R1"
                in self.hf_model_config._name_or_path
            ):
                param_byte_size = 675 * 1024 * 1024 * 1024
            else:
                raise ValueError("Unknown huggingface model card")
            # self.core_engine.Init(
            #     self.shm_name,
            #     self.tensor_meta_shm_name,
            #     param_byte_size,
            #     self.weight_copy_task,
            # )
            self.core_engine.Init(
                self.shm_name,
                self.tensor_meta_shm_name,
                param_byte_size
            )
            logging.info("Core engine initialized")
            
            # self._extract_dequantize_scale()
            # self._load_model_skeleton()
            # self._load_local_routed_experts()
            # self._config_attn_module()
            # self._config_expert_module()
            # self._config_lm_head_hook()
            
            

            # self.model.eval()
            # self.model.to(self.engine_config.Basic_Config.device_torch)
            # if dist.get_rank() == 0:    
            #     print(self.model, flush=True)
            
            # if len(self.weight_copy_task["routed_expert"]) == 0:
            #     self.weight_copy_task.pop("routed_expert")
                
            
            # self.core_engine.cuda_enable_peer_access(self.rank, self.world_size)
            # logging.info("Peer access enabled")
            # self.core_engine.set_global_routed_experts_data_ptr(self.global_routed_experts_data_ptr, self.expert_location_map)
            # logging.info("Global Expert IPC Handle Set")
            # self.core_engine.start_h2d_worker()
            logging.info("Host to Device worker started")
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e
        return (
            self.core_engine,
            self.model,
            self.engine_config,
            self.model_config,
            self.hf_model_config
        )

    def _parse_state_dict_ep(self):
        model_init_start_time = time.perf_counter()
        self.hf_model_config._attn_implementation = "eager"
        self.model = DeepseekV3ForCausalLM._from_config(
            self.hf_model_config
        ).to(self.engine_config.Basic_Config.device_torch)
        self.model.eval()
        logging.info(
            f"torch module init time: {time.perf_counter() - model_init_start_time} s"
        )

        self.weight_copy_task["attn"] = []
        self.weight_copy_task["routed_expert"] = []
        self.weight_copy_task["shared_expert"] = []

        # We have 8 devices and 256 experts per layer.
        # Each devices is responsible for 32 experts. However, the device may not be able to hold all experts.
        # In this case, we hold NUM_LOCAL_EXPERT_PER_LAYER in the GPU and the 32 - NUM_LOCAL_EXPERT_PER_LAYER in the host memory.
        # So the self.local_routed_experts in just the names of experts in each rank's GPU.

        self.local_routed_experts = []
        self.host_routed_experts = []
        # self.expert_location_map = {}

        NUM_LOCAL_EXPERT_PER_LAYER = 24  # Per requirements: 15 experts per GPU per layer
        NUM_TOTAL_EXPERTS = 256          # Total experts per layer
        


        routed_expert_gpu_start_idx = self.rank * 32
        routed_expert_gpu_end_idx = routed_expert_gpu_start_idx + NUM_LOCAL_EXPERT_PER_LAYER
        routed_expert_host_start_idx = routed_expert_gpu_end_idx
        routed_expert_host_end_idx = (self.rank + 1) * 32
        for layer_idx in range(
            self.hf_model_config.first_k_dense_replace,
            self.model_config.num_hidden_layers,
        ):
            # We split the 256 into 8 parts, each part is 32 experts.
            # The first NUM_LOCAL_EXPERT_PER_LAYER in each part associated with the corresponding rank.
            # The rest of the experts in the part are stored in the host memory.
            for expert_idx in range(routed_expert_gpu_start_idx, routed_expert_gpu_end_idx):
                self.local_routed_experts.append(
                    "routed_expert_" + str(layer_idx) + "_" + str(expert_idx)
                )
            for expert_idx in range(routed_expert_host_start_idx, routed_expert_host_end_idx):
                self.host_routed_experts.append(
                    "routed_expert_" + str(layer_idx) + "_" + str(expert_idx)
                )

        self.weight_copy_task["routed_expert"] = self.host_routed_experts
        # assert len(self.weight_copy_task["routed_expert"]) == 0


        for layer_idx in trange(self.model_config.num_hidden_layers):
            for name, _ in self.model.model.layers[
                layer_idx
            ].self_attn.named_parameters():
                tensor_full_name = (
                    "model.layers." + str(layer_idx) + ".self_attn." + name
                )
                self.state_dict_name_map[tensor_full_name] = {
                    "module_key": "attn_" + str(layer_idx),
                    "tensor_key": name,
                }
            self.weight_copy_task["attn"].append("attn_" + str(layer_idx))

            if layer_idx >= self.hf_model_config.first_k_dense_replace:
                for name, _ in self.model.model.layers[
                    layer_idx
                ].mlp.shared_experts.named_parameters():
                    tensor_full_name = (
                        "model.layers."
                        + str(layer_idx)
                        + ".mlp.shared_experts."
                        + name
                    )
                    self.state_dict_name_map[tensor_full_name] = {
                        "module_key": "shared_expert_" + str(layer_idx),
                        "tensor_key": name,
                    }
                self.weight_copy_task["shared_expert"].append(
                    "shared_expert_" + str(layer_idx)
                )

                for expert_idx in range(self.model_config.num_local_experts):
                    for name, _ in (
                        self.model.model.layers[layer_idx]
                        .mlp.experts[expert_idx]
                        .named_parameters()
                    ):
                        tensor_full_name = (
                            "model.layers."
                            + str(layer_idx)
                            + ".mlp.experts."
                            + str(expert_idx)
                            + "."
                            + name
                        )
                        self.state_dict_name_map[tensor_full_name] = {
                            "module_key": "routed_expert_"
                            + str(layer_idx)
                            + "_"
                            + str(expert_idx),
                            "tensor_key": name,
                        }
                    
                    # if "routed_expert_" + str(layer_idx) + "_" + str(expert_idx) not in self.local_routed_experts:
                    #     self.weight_copy_task["routed_expert"].append(
                    #         "routed_expert_"
                    #         + str(layer_idx)
                    #         + "_"
                    #         + str(expert_idx)
                    #     )
                    
                    # self.weight_copy_task["routed_expert"].append(
                    #     "routed_expert_"
                    #     + str(layer_idx)
                    #     + "_"
                    #     + str(expert_idx)
                    # )
        # if len(self.weight_copy_task["routed_expert"]) == 0:
        #     self.weight_copy_task.pop("routed_expert")


        
        
                