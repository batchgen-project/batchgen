# ---------------------------------------------------------------------------- #
#  MoE-Gen                                                                      #
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

import torch
import triton
import triton.language as tl
from transformers.cache_utils import DynamicCache
import torch.distributed as dist
from ..models.deepseek.deepseekv3.quantization import (
    compressed_kv_bf16_to_fp8_per_token,
    compressed_kv_fp8_to_bf16_per_token
)




def update_casual_mask(attention_mask):
    dtype = torch.bfloat16
    min_dtype = torch.finfo(dtype).min
    device = attention_mask.device
    target_length = attention_mask.size(-1)
    sequence_length = 1
    cache_position = torch.arange(
        target_length - 1, target_length, device=device
    )
    causal_mask = torch.full(
        (sequence_length, target_length),
        fill_value=min_dtype,
        dtype=dtype,
        device=device,
    )
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask *= torch.arange(
        target_length, device=device
    ) > cache_position.reshape(-1, 1)
    causal_mask = causal_mask[None, None, :, :].expand(
        attention_mask.shape[0], 1, -1, -1
    )
    if attention_mask is not None:
        causal_mask = (
            causal_mask.clone()
        )  # copy to contiguous memory for in-place edit
        mask_length = attention_mask.shape[-1]
        padding_mask = (
            causal_mask[:, :, :, :mask_length]
            + attention_mask[:, None, None, :]
        )
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[
            :, :, :, :mask_length
        ].masked_fill(padding_mask, min_dtype)
    return causal_mask


def create_position_ids_from_attention_mask(
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    attention_mask: shape [batch_size, seq_len], with values in {0, 1}.
    Returns position_ids: same shape, where
      - tokens with attention_mask=0 get position_id=1
      - tokens with attention_mask=1 get a cumsum starting at 0
    """
    position_ids = torch.zeros_like(attention_mask, dtype=torch.long)
    cumsum = torch.cumsum(attention_mask, dim=1) - 1
    position_ids[attention_mask == 1] = cumsum[attention_mask == 1]
    position_ids[attention_mask == 0] = 1
    return position_ids


# def deepseek_v3_dequantization(
# 		weight_data_fp8,
# 		weight_scale_inv_fp32,
# 		block_size = [128,128]) -> torch.Tensor:
# 	start_time = torch.cuda.Event(enable_timing=True)
# 	end_time = torch.cuda.Event(enable_timing=True)
# 	start_time.record()
# 	rows, cols = weight_data_fp8.size()
# 	n_block_rows = math.ceil(rows / block_size[0])
# 	n_block_cols = math.ceil(cols / block_size[1])
# 	assert n_block_cols == weight_scale_inv_fp32.size(1)
# 	assert n_block_rows == weight_scale_inv_fp32.size(0)

# 	dequantized_weight = weight_data_fp8.to(torch.float32)
# 	for i in range(n_block_rows):
# 		for j in range(n_block_cols):
# 			dequantized_weight[i*block_size[0]:(i+1)*block_size[0], j*block_size[1]:(j+1)*block_size[1]] *= weight_scale_inv_fp32[i, j]

# 	dequantized_weight = dequantized_weight.to(torch.bfloat16)
# 	end_time.record()
# 	torch.cuda.synchronize(7)
# 	# logging dequantize time in ms
# 	logging.info(f"Dequantization time: {start_time.elapsed_time(end_time)} ms")
# 	return dequantized_weight


def deepseek_v3_dequantization(
    weight_data_fp8, weight_scale_inv_fp32, block_size=[128, 128]
) -> torch.Tensor:
    # start_time = torch.cuda.Event(enable_timing=True)
    # end_time = torch.cuda.Event(enable_timing=True)
    # start_time.record()
    rows, cols = weight_data_fp8.size()
    n_block_rows = math.ceil(rows / block_size[0])
    n_block_cols = math.ceil(cols / block_size[1])
    assert n_block_cols == weight_scale_inv_fp32.size(1)
    assert n_block_rows == weight_scale_inv_fp32.size(0)
    # Check input are on the same device
    # logging.info(
    #     f"weight_data_fp8 device: {weight_data_fp8.device}, weight_scale_inv_fp32 device: {weight_scale_inv_fp32.device}"
    # )

    dequantized_weight = weight_data_fp8.to(torch.float32)
    expanded_scales = weight_scale_inv_fp32.repeat_interleave(
        block_size[0], dim=0
    ).repeat_interleave(block_size[1], dim=1)
    expanded_scales = expanded_scales[
        :rows, :cols
    ]  # trim if block doesn't perfectly divide
    dequantized_weight *= expanded_scales
    # for i in range(n_block_rows):
    # 	for j in range(n_block_cols):
    # 		dequantized_weight[i*block_size[0]:(i+1)*block_size[0], j*block_size[1]:(j+1)*block_size[1]] *= weight_scale_inv_fp32[i, j]

    dequantized_weight = dequantized_weight.to(torch.bfloat16)
    # end_time.record()
    # torch.cuda.synchronize(7)
    # # logging dequantize time in ms
    # logging.info(f"Dequantization time: {start_time.elapsed_time(end_time)} ms")
    return dequantized_weight


# def deepseek_v3_dequantization(
#     weight_data_fp8: torch.Tensor,
#     weight_scale_inv_fp32: torch.Tensor,
#     block_size=(128, 128)
# ) -> torch.Tensor:
#     """
#     Vectorized dequantization that removes Python-level loops
#     and leverages PyTorch's parallelism.
#     """
#     rows, cols = weight_data_fp8.shape
#     block_rows, block_cols = block_size

#     # Number of blocks in each dimension
#     n_block_rows = rows // block_rows
#     n_block_cols = cols // block_cols

#     # 1) Reshape weight data into 4D block form and cast to float32
#     #    shape becomes [n_block_rows, block_rows, n_block_cols, block_cols].
#     weight_4d = weight_data_fp8.reshape(n_block_rows, block_rows, n_block_cols, block_cols).to(torch.float32)

#     # 2) Broadcast scale into 4D by unsqueezing along the second and fourth dimensions.
#     #    shape becomes [n_block_rows, 1, n_block_cols, 1].
#     scale_4d = weight_scale_inv_fp32.unsqueeze(1).unsqueeze(-1)

#     # 3) Multiply once using broadcasting
#     dequantized_4d = weight_4d * scale_4d

#     # 4) Reshape back to [rows, cols] and cast to bfloat16
#     dequantized_weight = dequantized_4d.reshape(rows, cols).to(torch.bfloat16)

#     return dequantized_weight


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


class Attn_Wrapper(torch.nn.Module):
    phase = "prefill"
    attn_mode = 0
    cur_batch = None
    kv_quantization_factor = None

    def __init__(
        self,
        module,
        layer_idx,
        core_engine,
        engine_config,
        model_config,
        get_weights,
        weight_dequant_scale=None,
    ):
        super().__init__()
        self.module = module
        self.layer_idx = layer_idx
        self.core_engine = core_engine
        self.engine_config = engine_config
        self.model_config = model_config
        self.get_weights = get_weights
        self.attn_module_id = "attn" + "_" + str(self.layer_idx)
        self.weight_dequant_scale = weight_dequant_scale


    def forward(self, *args, **kwargs):
        logging.debug(
            f"[Layer {self.layer_idx} - Attn_Wrapper] Forward pass. Phase: {Attn_Wrapper.phase}"
        )
        self.core_engine.clear_expert_buffer(self.layer_idx, 0)
        # Step 1: Synchronize Attn Weights.
        if self.get_weights:
            weights_dict = self.core_engine.get_weights(self.attn_module_id)
            for name, param in self.module.named_parameters():
                if (
                    self.weight_dequant_scale is not None
                    and name + "_scale_inv" in self.weight_dequant_scale
                ):
                    param.data = deepseek_v3_dequantization(
                        weights_dict[name],
                        self.weight_dequant_scale[name + "_scale_inv"],
                    )
                else:
                    param.data = weights_dict[name]
        if Attn_Wrapper.phase == "prefill":
            """
				All attn Mode has the same prefill logic.
			"""
            # Get all tensor objects currently alive
            # for obj in gc.get_objects():
            # 	try:
            # 		if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            # 			if obj.numel() * obj.element_size() * 2 > 1e8:
            # 				if obj.device.type == "cuda":
            # 					logging.debug(f"Tensor object id: {id(obj)},size: {obj.numel() * obj.element_size() * 2}, shape: {obj.shape}")
            # 	except:
            # 		pass
            # Step 2: Prepare input
            arg_dict = {
                "hidden_states": kwargs["hidden_states"],
                "attention_mask": kwargs["attention_mask"],
                "position_ids": kwargs["position_ids"],
                "past_key_value": kwargs["past_key_value"],
                "output_attentions": kwargs["output_attentions"],
                "use_cache": kwargs["use_cache"],
            }
            past_key_value = DynamicCache()
            for i in range(self.layer_idx):
                past_key_value.key_cache.append(None)
                past_key_value.value_cache.append(None)
            arg_dict["past_key_value"] = past_key_value

            # Step 3: Forward pass
            num_prefill_batch = math.ceil(
                len(arg_dict["hidden_states"])
                / self.engine_config.Module_Batching_Config.attn_prefill_micro_batch_size
            )
            attn_output = torch.zeros_like(arg_dict["hidden_states"])
            for prefill_attn_batch_idx in range(num_prefill_batch):
                cur_batch_start = (
                    prefill_attn_batch_idx
                    * self.engine_config.Module_Batching_Config.attn_prefill_micro_batch_size
                )
                cur_batch_end = (
                    len(arg_dict["hidden_states"])
                    if (prefill_attn_batch_idx + 1)
                    * self.engine_config.Module_Batching_Config.attn_prefill_micro_batch_size
                    > len(arg_dict["hidden_states"])
                    else (prefill_attn_batch_idx + 1)
                    * self.engine_config.Module_Batching_Config.attn_prefill_micro_batch_size
                )
                cur_attn_batch = Attn_Wrapper.cur_batch[
                    cur_batch_start:cur_batch_end
                ]
                cur_hidden_states = arg_dict["hidden_states"][
                    cur_batch_start:cur_batch_end
                ]
                if "deepseek" in self.model_config.model_type:
                    # cur_attention_mask = arg_dict["attention_mask"][
                    #     cur_batch_start:cur_batch_end
                    # ]
                    cur_attention_mask = Attn_Wrapper.attention_mask[
                        cur_batch_start:cur_batch_end
                    ]
                else:
                    cur_attention_mask = Attn_Wrapper.attention_mask[
                        cur_batch_start:cur_batch_end
                    ]

                key_cache = None
                value_cache = None
                with torch.no_grad():
                    if "deepseek" in self.model_config.model_type:
                        position_ids = Attn_Wrapper.position_ids[
                            cur_batch_start:cur_batch_end
                        ]
                        # output = self.module.prefill_attn_fp8(
                        #     cur_hidden_states, cur_attention_mask, position_ids,
                        #     self.weight_dequant_scale
                        # )
                        # logging.info(f"cur_attention_mask_shape: {cur_attention_mask.shape}")
                        # logging.info(f"cur_attention_mask: {cur_attention_mask}")
                        # exit()
                        output = self.module.prefill_attn(
                            cur_hidden_states,
                            cur_attention_mask.to(cur_hidden_states.device),
                            position_ids.to(cur_hidden_states.device),
                        )
                        key_cache = output[1]
                        value_cache = torch.ones(
                            1,
                            dtype=torch.bfloat16,
                            device=kwargs["hidden_states"].device,
                        )
                    else:
                        kv = DynamicCache()
                        for i in range(self.layer_idx):
                            kv.key_cache.append(None)
                            kv.value_cache.append(None)
                        output = self.module(
                            hidden_states=arg_dict["hidden_states"][
                                cur_batch_start:cur_batch_end
                            ],
                            attention_mask=arg_dict["attention_mask"][
                                cur_batch_start:cur_batch_end
                            ],
                            position_ids=arg_dict["position_ids"],
                            past_key_value=kv,
                        )
                        key_cache = output[2].key_cache[self.layer_idx]
                        value_cache = output[2].value_cache[self.layer_idx]

                    torch.cuda.current_stream(self.engine_config.Basic_Config.device_torch).synchronize()
                    attn_output[cur_batch_start:cur_batch_end] = output[0]
                    torch.cuda.current_stream(self.engine_config.Basic_Config.device_torch).synchronize()


                # Quantization
                # q_key, factor = compressed_kv_bf16_to_fp8_per_token(key_cache)
                # if Attn_Wrapper.kv_quantization_factor is None:
                #     Attn_Wrapper.kv_quantization_factor = [None for _ in range(self.model_config.num_hidden_layers)]
                # if Attn_Wrapper.kv_quantization_factor[self.layer_idx] is None:
                #     Attn_Wrapper.kv_quantization_factor[self.layer_idx] = factor
                # else:
                #     Attn_Wrapper.kv_quantization_factor[self.layer_idx] = torch.cat(
                #         (Attn_Wrapper.kv_quantization_factor[self.layer_idx], factor),
                #         dim=0,
                #     )
                self.core_engine.kv_offload(
                    self.layer_idx, cur_attn_batch, key_cache, value_cache
                )

            # Step 4: Clean up
            if self.get_weights:
                self.core_engine.free_weights_buffer(self.attn_module_id)
                for name, param in self.module.named_parameters():
                    param.data = torch.tensor(
                        0.0, dtype=param.data.dtype, device=param.data.device
                    )

            logging.debug(
                f"[Layer {self.layer_idx} - Attn_Wrapper] Finish forward pass. Phase: {Attn_Wrapper.phase}"
            )

            return attn_output, None, None

        elif Attn_Wrapper.phase == "decoding":
            # logging.info(f"[Layer {self.layer_idx} - Attn_Wrapper] Decoding phase.")
            self.core_engine.clear_expert_buffer(self.layer_idx, 0)
            hidden_states = kwargs["hidden_states"]
            attention_mask = Attn_Wrapper.attention_mask
            position_ids = Attn_Wrapper.position_ids
            final_attn_result = self.core_engine.attn(
                self.module,
                self.layer_idx,
                hidden_states,
                attention_mask,
                position_ids.to(self.engine_config.Basic_Config.device_torch),
                Attn_Wrapper.cur_batch,
            )
            torch.cuda.current_stream(self.engine_config.Basic_Config.device_torch).synchronize()
            # Step 4: Clean up
            if self.get_weights:
                self.core_engine.free_weights_buffer(self.attn_module_id)
                for name, param in self.module.named_parameters():
                    param.data = torch.tensor(
                        0.0, dtype=param.data.dtype, device=param.data.device
                    )

            logging.debug(
                f"[Layer {self.layer_idx} - Attn_Wrapper] Finish forward pass. Phase: {Attn_Wrapper.phase}"
            )
            return final_attn_result, None, None


class Expert_Wrapper(torch.nn.Module):
    """
    For Mixtral and Qwen2MoE with transformers==4.42.0
    """

    phase = "prefill"

    def __init__(
        self,
        expert_module,
        layer_idx,
        expert_idx,
        core_engine,
        engine_config,
        model_config,
        get_weights,
        weight_dequant_scale=None,
    ):
        super().__init__()
        self.module = expert_module
        self.layer_idx = layer_idx
        self.expert_idx = expert_idx
        self.engine_config = engine_config
        self.model_config = model_config
        self.core_engine = core_engine
        self.get_weights = get_weights
        self.weight_dequant_scale = weight_dequant_scale
        if self.expert_idx >= 0:
            self.expert_weights_idx = (
                "routed_expert"
                + "_"
                + str(self.layer_idx)
                + "_"
                + str(self.expert_idx)
            )
        else:
            self.expert_weights_idx = (
                "shared_expert" + "_" + str(self.layer_idx)
            )

    def forward(self, *args, **kwargs):
        logging.debug(
            f"[Layer {self.layer_idx} - Expert {self.expert_idx}] Forward pass. Phase: {Expert_Wrapper.phase}"
        )
        # Step 1: Synchronize Expert Weights.
        if self.expert_idx != -1:
            self.core_engine.clear_expert_buffer(
                self.layer_idx, self.expert_idx
            )
        if self.get_weights:
            weights_dict = self.core_engine.get_weights(self.expert_weights_idx)
            for name, param in self.module.named_parameters():
                if (
                    self.weight_dequant_scale is not None
                    and name + "_scale_inv" in self.weight_dequant_scale
                ):
                    
                    param.data = deepseek_v3_dequantization(
                        weights_dict[name],
                        self.weight_dequant_scale[name + "_scale_inv"].to(
                            self.engine_config.Basic_Config.device_torch #TODO:
                        ),
                    )
                else:
                    param.data = weights_dict[name]
        else:
            self.module.gate_proj.weight.data = deepseek_v3_dequantization(
                self.fp8_gate,
                self.weight_dequant_scale[
                    "gate_proj.weight_scale_inv"
                ],
            )
            self.module.down_proj.weight.data = deepseek_v3_dequantization(
                self.fp8_down,
                self.weight_dequant_scale["down_proj.weight_scale_inv"],
            )
            self.module.up_proj.weight.data = deepseek_v3_dequantization(
                self.fp8_up,
                self.weight_dequant_scale["up_proj.weight_scale_inv"],
            )

        # Step 2: Forward pass, micro-batching in case of OOM.
        hidden_states = args[0]
        result = torch.zeros_like(hidden_states)
        token_num_upper_bound = (
            self.engine_config.Module_Batching_Config.expert_prefill_batch_size_upper_bound
            if Expert_Wrapper.phase == "prefill"
            else self.engine_config.Module_Batching_Config.expert_decoding_batch_size_upper_bound
        )
        num_micro_batch = math.ceil(len(hidden_states) / token_num_upper_bound)
        for i in range(num_micro_batch):
            start = i * token_num_upper_bound
            end = (
                len(hidden_states)
                if (i + 1) * token_num_upper_bound > len(hidden_states)
                else (i + 1) * token_num_upper_bound
            )
            micro_batch = hidden_states[start:end]
            # self.module.eval()
            with torch.no_grad():
                result[start:end].copy_(self.module(micro_batch))
            torch.cuda.current_stream(self.engine_config.Basic_Config.device_torch).synchronize()

        # Step 3: Clean up
        if self.get_weights:
            self.core_engine.free_weights_buffer(self.expert_weights_idx)
            for name, param in self.module.named_parameters():
                param.data = torch.tensor(
                    0.0, dtype=param.data.dtype, device=param.data.device
                )
        else:
            self.module.gate_proj.weight.data = self.fp8_gate
            self.module.down_proj.weight.data = self.fp8_down
            self.module.up_proj.weight.data = self.fp8_up

        logging.debug(
            f"[Layer {self.layer_idx} - Expert {self.expert_idx}] Finish forward pass. Phase: {Expert_Wrapper.phase}"
        )
        if self.expert_idx != -1:
            self.core_engine.clear_expert_buffer(
                self.layer_idx, self.expert_idx
            )
        return result

    def _register_fp8_weights(self):
        self.fp8_gate = self.module.gate_proj.weight.data
        self.fp8_down = self.module.down_proj.weight.data
        self.fp8_up = self.module.up_proj.weight.data



from collections import defaultdict
class DeepseekV3MoE(torch.nn.Module):
    """
        For DeepSeekV3 MoE and EP
        1. Pass the Gate and distribute the activations to the dst rank.
        2. Do the MLPs that each worker only call its own experts.
        3. Gather the activations back to the src rank.        
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        assert self.world_size == 8 

        self.routed_expert_start_idx = 32 * self.rank
        self.routed_expert_end_idx = 32 * (self.rank + 1)

        # Full init
        self.experts = nn.ModuleList(
                [
                    DeepseekV3MLP(
                        config, intermediate_size=config.moe_intermediate_size
                    )
                    for i in range(config.n_routed_experts)
                ]
            )
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV3MLP(
                config=config, intermediate_size=intermediate_size
            )

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight): 
        """
        x: [B, E]
        topk_ids: [B,K]
        topk_weight: [B,K]
        
        Each rank will process only experts with indices between 
        self.routed_expert_start_idx and self.routed_expert_end_idx
        """
        # 1. Send the activations and its weights to the dst rank
        # 2. Do the MLPs that each worker only call its own experts.
        # 3. Get the result activations back to the src rank.
        
        batch_size, hidden_dim = x.shape
        num_experts = 256  # Total experts across all devices
        experts_per_device = 32  # Experts per device
        num_k = self.num_experts_per_tok  # Number of experts per token
        
        # Create a structure to hold information about which tokens need processing by which experts
        # Key: destination rank, Value: [token_indices, expert_indices, weights]
        token_dispatch_map = defaultdict(lambda: [[], [], []])
        
        # For each token, determine which experts are needed and where they live
        for token_idx in range(batch_size):
            for k_idx in range(num_k):
                expert_idx = topk_ids[token_idx, k_idx].item()
                weight = topk_weight[token_idx, k_idx].item()
                
                # Calculate which rank owns this expert
                dst_rank = expert_idx // experts_per_device
                
                # Store information for this token-expert pair
                token_dispatch_map[dst_rank][0].append(token_idx)
                token_dispatch_map[dst_rank][1].append(expert_idx)  # Global expert index
                token_dispatch_map[dst_rank][2].append(weight)
        
        # Prepare tensors for all-to-all communication
        send_token_counts = torch.zeros(self.world_size, dtype=torch.int64, device=x.device)
        for dst_rank, (token_indices, _, _) in token_dispatch_map.items():
            send_token_counts[dst_rank] = len(token_indices)
        
        # All-to-all to exchange token counts
        recv_token_counts = torch.zeros_like(send_token_counts)
        dist.all_to_all_single(recv_token_counts, send_token_counts)
        
        # Prepare data to send: [token_data, expert_indices, weights]
        # For each destination rank, pack the data
        send_data = []
        send_expert_indices = []
        send_weights = []
        send_token_indices = []  # Original indices for later reconstruction
        
        for dst_rank in range(self.world_size):
            if dst_rank in token_dispatch_map:
                token_indices, expert_indices, weights = token_dispatch_map[dst_rank]
                # Get tokens that need to be sent
                send_token_data = x[token_indices]
                send_data.append(send_token_data)
                send_expert_indices.append(torch.tensor(expert_indices, device=x.device))
                send_weights.append(torch.tensor(weights, device=x.device))
                send_token_indices.append(torch.tensor(token_indices, device=x.device))
            else:
                # Empty tensors for ranks with no tokens to send
                send_data.append(torch.zeros((0, hidden_dim), device=x.device))
                send_expert_indices.append(torch.zeros((0,), dtype=torch.int64, device=x.device))
                send_weights.append(torch.zeros((0,), device=x.device))
                send_token_indices.append(torch.zeros((0,), dtype=torch.int64, device=x.device))
        
        # Concatenate send data for all-to-all
        send_data_cat = torch.cat(send_data, dim=0)
        send_expert_indices_cat = torch.cat(send_expert_indices, dim=0)
        send_weights_cat = torch.cat(send_weights, dim=0)
        send_token_indices_cat = torch.cat(send_token_indices, dim=0)
        
        # Calculate send/recv splits for all-to-all
        send_splits = [t.size(0) for t in send_data]
        recv_splits = recv_token_counts.tolist()
        
        # Prepare receive buffers
        total_recv_tokens = recv_token_counts.sum().item()
        recv_data = torch.zeros((total_recv_tokens, hidden_dim), device=x.device)
        recv_expert_indices = torch.zeros((total_recv_tokens,), dtype=torch.int64, device=x.device)
        recv_weights = torch.zeros((total_recv_tokens,), device=x.device)
        recv_src_ranks = torch.zeros((total_recv_tokens,), dtype=torch.int64, device=x.device)
        recv_token_indices = torch.zeros((total_recv_tokens,), dtype=torch.int64, device=x.device)
        
        # All-to-all for token data
        if total_recv_tokens > 0:
            # Exchange token data
            dist.all_to_all(recv_data, send_data_cat, recv_splits, send_splits)
            
            # Exchange expert indices
            dist.all_to_all_single(
                recv_expert_indices, 
                send_expert_indices_cat, 
                recv_splits, 
                send_splits
            )
            
            # Exchange weights
            dist.all_to_all_single(
                recv_weights,
                send_weights_cat,
                recv_splits,
                send_splits
            )
            
            # Exchange original token indices
            dist.all_to_all_single(
                recv_token_indices,
                send_token_indices_cat,
                recv_splits,
                send_splits
            )
            
            # Create source rank information
            offset = 0
            for src_rank, count in enumerate(recv_splits):
                if count > 0:
                    recv_src_ranks[offset:offset+count] = src_rank
                    offset += count
        
        # Compute expert outputs for tokens received by this rank
        expert_outputs = torch.zeros_like(recv_data)
        
        if total_recv_tokens > 0:
            # Group by local expert - using only experts in our assigned range
            for local_expert_idx in range(self.routed_expert_start_idx, self.routed_expert_end_idx):
                # The expert indices received are global indices, so we need to match directly
                expert_mask = (recv_expert_indices == local_expert_idx)
                if not expert_mask.any():
                    continue
                
                # Get tokens that need this expert
                expert_tokens = recv_data[expert_mask]
                
                # Run the expert - use the global expert index directly
                expert_out = self.experts[local_expert_idx](expert_tokens)
                
                # Store results
                expert_outputs[expert_mask] = expert_out
        
        # Apply weights
        expert_outputs = expert_outputs * recv_weights.unsqueeze(1)
        
        # Prepare data to send back to source
        send_back_splits = []
        send_back_data = []
        
        for src_rank in range(self.world_size):
            mask = (recv_src_ranks == src_rank)
            if mask.any():
                count = mask.sum().item()
                send_back_splits.append(count)
                # Pack [output, original_token_index]
                send_back_data.append(expert_outputs[mask])
            else:
                send_back_splits.append(0)
                send_back_data.append(torch.zeros((0, hidden_dim), device=x.device))
        
        # Concatenate for all-to-all
        send_back_data_cat = torch.cat(send_back_data, dim=0)
        
        # Prepare receive buffer for final results
        # Each token may have selected multiple experts, so we need to aggregate
        final_output = torch.zeros_like(x)
        
        # All-to-all to send results back to source ranks
        if send_back_data_cat.size(0) > 0:
            # Exchange results
            recv_back_data = torch.zeros((sum(send_splits), hidden_dim), device=x.device)
            dist.all_to_all(
                recv_back_data,
                send_back_data_cat,
                send_splits,
                send_back_splits
            )
            
            # Process received data and reconstruct output
            offset = 0
            for dst_rank in range(self.world_size):
                if dst_rank in token_dispatch_map:
                    token_indices = token_dispatch_map[dst_rank][0]
                    count = len(token_indices)
                    if count > 0:
                        # Get results for these tokens
                        token_results = recv_back_data[offset:offset+count]
                        
                        # Add to final output (accumulating results from multiple experts)
                        for i, token_idx in enumerate(token_indices):
                            final_output[token_idx] += token_results[i]
                        
                        offset += count
        
        return final_output
    
