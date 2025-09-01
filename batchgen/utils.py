import torch
import pynvml

def get_gpu_memory_usage():
    pynvml.nvmlInit()
    
    # Get number of GPUs
    # device_count = pynvml.nvmlDeviceGetCount()
    
    # for i in range(device_count):
    for i in range(1):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        
        # Get memory info
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        print(f"GPU {i}:")
        print(f"  Total memory: {mem_info.total / 1024**3:.2f} GB")
        print(f"  Used memory: {mem_info.used / 1024**3:.2f} GB")
        print(f"  Free memory: {mem_info.free / 1024**3:.2f} GB")
        print(f"  Usage: {mem_info.used / mem_info.total * 100:.1f}%")
        
        # Get additional info
        name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
        print(f"  Device: {name}")
    
    pynvml.nvmlShutdown()

	
def torch_gpu_mem_usage(rank):
	"""
		Return current GPU memory used by Torch in GB.
	"""
	if not torch.cuda.is_available():
		return 0.0

	mem = torch.cuda.memory_allocated(device=rank) / (1024 ** 3)
	return mem