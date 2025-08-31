import torch
def torch_gpu_mem_usage(rank):
	"""
		Return current GPU memory used by Torch in GB.
	"""
	if not torch.cuda.is_available():
		return 0.0

	mem = torch.cuda.memory_allocated(device=rank) / (1024 ** 3)
	return mem