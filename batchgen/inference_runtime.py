import torch
import math
import logging
import tqdm
from typing import Optional, List, Dict
import torch.distributed as dist
from dataclasses import dataclass

"""
	This class encapsulates the inference runtime. It provides following APIs:
	- config_prefill: configure the prefill phase
	- prefill: pass the input to self.model
	- config_decode: configure the decode phase
	- decode: pass the input to self.model and do one forward pass.
	- get_current_phase: get the current phase, prefill or decode
"""

class InferenceRuntime:
	def __init__(self, model_name: str):
		self.model_name = model_name
		current_phase = None
		pass

	def config_prefill(self, **kwargs):
		"""
		Configure the prefill phase with necessary parameters.
		"""
		pass

	def prefill(self, input_batch):
		"""
		Perform the prefill phase with the given input batch.
		"""
		pass

	def config_decode(self, **kwargs):
		"""
		Configure the decode phase with necessary parameters.
		"""
		pass

	def decode(self, decode_batch):
		"""
		Perform one step of decoding with the given decode batch.
		"""
		pass

	def get_current_phase(self) -> str:
		"""
		Return the current phase: 'prefill' or 'decode'.
		"""
		pass
