import torch
from dataclasses import dataclass
@dataclass
class ModelForwardInput:
	sequence_uuids: list[int]
	input_ids: torch.Tensor
	attention_mask: torch.Tensor
	position_ids: torch.Tensor

@dataclass
class ModelForwardOutput:
	sequence_uuids: list[int]
	new_tokens: torch.Tensor