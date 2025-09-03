import torch
# import pynvml
import logging


# def get_gpu_memory_usage(device:int):
#     pynvml.nvmlInit()
#     handle = pynvml.nvmlDeviceGetHandleByIndex(device)
#     mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
#     pynvml.nvmlShutdown()
#     # return {
#     #     "total": mem_info.total / 1024**3,
#     #     "used": mem_info.used / 1024**3,
#     #     "free": mem_info.free / 1024**3,
#     #     "usage": mem_info.used / mem_info.total * 100
#     # }
#     return mem_info.total / 1024**3, mem_info.used / 1024**3, mem_info.free / 1024**3, mem_info.used / mem_info.total * 100


# def get_gpu_memory_usage():
#     pynvml.nvmlInit()
    
#     # Get number of GPUs
#     # device_count = pynvml.nvmlDeviceGetCount()
    
#     # for i in range(device_count):
#     for i in range(1):
#         handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        
#         # Get memory info
#         mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

#         logging.info(f"GPU {i}:")
#         logging.info(f"  Total memory: {mem_info.total / 1024**3:.2f} GB")
#         logging.info(f"  Used memory: {mem_info.used / 1024**3:.2f} GB")
#         logging.info(f"  Free memory: {mem_info.free / 1024**3:.2f} GB")
#         logging.info(f"  Usage: {mem_info.used / mem_info.total * 100:.1f}%")

#         # Get additional info
#         name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
#         logging.info(f"  Device: {name}")

#     pynvml.nvmlShutdown()

	
def torch_gpu_mem_usage(rank):
	"""
		Return current GPU memory used by Torch in GB.
	"""
	if not torch.cuda.is_available():
		return 0.0

	mem = torch.cuda.memory_allocated(device=rank) / (1024 ** 3)
	return mem