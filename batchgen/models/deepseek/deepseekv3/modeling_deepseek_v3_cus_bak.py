""" PyTorch DeepSeek model."""
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache

from transformers.utils.import_utils import is_torch_fx_available
from .configuration_deepseek_v3 import DeepseekV3Config
import torch.distributed as dist
import numpy as np
import os
import triton
import logging

def Embedding():
	pass


def DeepseekV3Model(model_config:dict):
	"""
		DeepseekV3ForCausalLM(
		(model): DeepseekV3Model(
			(embed_tokens): Embedding(129280, 7168)
			(layers): ModuleList(
			(0-2): 3 x DeepseekV3DecoderLayer(
				(self_attn): DeepseekV3Attention(
				(q_a_proj): Linear(in_features=7168, out_features=1536, bias=False)
				(q_a_layernorm): DeepseekV3RMSNorm()
				(q_b_proj): Linear(in_features=1536, out_features=24576, bias=False)
				(kv_a_proj_with_mqa): Linear(in_features=7168, out_features=576, bias=False)
				(kv_a_layernorm): DeepseekV3RMSNorm()
				(kv_b_proj): Linear(in_features=512, out_features=32768, bias=False)
				(o_proj): Linear(in_features=16384, out_features=7168, bias=False)
				(rotary_emb): DeepseekV3YarnRotaryEmbedding()
				)
				(mlp): DeepseekV3MLP(
				(gate_proj): Linear(in_features=7168, out_features=18432, bias=False)
				(up_proj): Linear(in_features=7168, out_features=18432, bias=False)
				(down_proj): Linear(in_features=18432, out_features=7168, bias=False)
				(act_fn): SiLU()
				)
				(input_layernorm): DeepseekV3RMSNorm()
				(post_attention_layernorm): DeepseekV3RMSNorm()
			)
			(3-60): 58 x DeepseekV3DecoderLayer(
				(self_attn): DeepseekV3Attention(
				(q_a_proj): Linear(in_features=7168, out_features=1536, bias=False)
				(q_a_layernorm): DeepseekV3RMSNorm()
				(q_b_proj): Linear(in_features=1536, out_features=24576, bias=False)
				(kv_a_proj_with_mqa): Linear(in_features=7168, out_features=576, bias=False)
				(kv_a_layernorm): DeepseekV3RMSNorm()
				(kv_b_proj): Linear(in_features=512, out_features=32768, bias=False)
				(o_proj): Linear(in_features=16384, out_features=7168, bias=False)
				(rotary_emb): DeepseekV3YarnRotaryEmbedding()
				)
				(mlp): DeepseekV3MoE(
				(experts): ModuleList(
					(0-255): 256 x DeepseekV3MLP(
					(gate_proj): Linear(in_features=7168, out_features=2048, bias=False)
					(up_proj): Linear(in_features=7168, out_features=2048, bias=False)
					(down_proj): Linear(in_features=2048, out_features=7168, bias=False)
					(act_fn): SiLU()
					)
				)
				(gate): MoEGate()
				(shared_experts): DeepseekV3MLP(
					(gate_proj): Linear(in_features=7168, out_features=2048, bias=False)
					(up_proj): Linear(in_features=7168, out_features=2048, bias=False)
					(down_proj): Linear(in_features=2048, out_features=7168, bias=False)
					(act_fn): SiLU()
				)
				)
				(input_layernorm): DeepseekV3RMSNorm()
				(post_attention_layernorm): DeepseekV3RMSNorm()
			)
			)
			(norm): DeepseekV3RMSNorm()
		)
		(lm_head): Linear(in_features=7168, out_features=129280, bias=False)
		)
	"""
	def __init__(self, model_config):
		self.model_config = model_config

	



