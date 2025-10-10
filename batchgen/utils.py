import functools

# import pynvml
import logging
import os
import signal
import socket
import sys
import threading
from typing import Callable

import psutil
import torch
import zmq

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

def config_torch_module_initializer():
	def do_nothing_decorator(orig_func: Callable) -> Callable:
		@functools.wraps(orig_func)
		def do_nothing(*args, **kwargs):
			pass

		return do_nothing

	def param_init_decorator(orig_param_init: Callable) -> Callable:
		@functools.wraps(orig_param_init)
		def archer_param_init(cls, *args, **kwargs):
			orig_param_init(cls, *args, **kwargs)

			for name, param in cls.named_parameters(recurse=False):
				param.data = torch.zeros(
					1, dtype=torch.bfloat16, device=param.device
				)

			# for name, buf in cls.named_buffers(recurse=False):
			# 	buf.data = torch.zeros(1, dtype=torch.bfloat16, device=buf.device)

		return archer_param_init

	# for all the modules in torch.nn, add post_init method
	# assert False, torch.nn.modules.__dict__
	for name, module in torch.nn.modules.__dict__.items():
		if not isinstance(module, type):
			continue
		if not issubclass(module, torch.nn.modules.module.Module):
			continue
		if name in [
			"Module",
			"Sequential",
			"ModuleDict",
			"ModuleList",
			"ParameterList",
			"ParameterDict",
		]:
			continue
		module._old_init = module.__init__
		module.__init__ = param_init_decorator(module.__init__)

		if hasattr(module, "reset_parameters"):
			module._old_reset_parameters = module.reset_parameters
			module.reset_parameters = do_nothing_decorator(
				module.reset_parameters
			)
	
def torch_gpu_mem_usage(rank):
	"""
		Return current GPU memory used by Torch in GB.
	"""
	if not torch.cuda.is_available():
		return 0.0

	mem = torch.cuda.memory_allocated(device=rank) / (1024 ** 3)
	return mem

def create_position_ids_from_attention_mask(
	attention_mask: torch.Tensor,
) -> torch.Tensor:
	"""
	attention_mask: shape [batch_size, seq_len], with values in {0, 1}.
	Returns position_ids: same shape, where
	  - tokens with attention_mask=0 get position_id=1
	  - tokens with attention_mask=1 get a cumsum starting at 0
	"""
	# Cumulative sum along the sequence dimension
	cumsum = attention_mask.cumsum(dim=-1)
	# Shift by -1 and clamp at 0 so first 1-based token starts at 0
	position_ids = torch.clamp(cumsum - 1, min=0)
	# Zero out positions where mask=0, then replace those with 1
	position_ids = position_ids * attention_mask
	position_ids = position_ids + (attention_mask.eq(0) * (-1))
	return position_ids


def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = None):
    """Kill the process and all its child processes."""
    # Remove sigchld handler to avoid spammy logs.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    try:
        itself = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return

    children = itself.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    if include_parent:
        try:
            if parent_pid == os.getpid():
                itself.kill()
                sys.exit(0)

            itself.kill()

            # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
            # so we send an additional signal to kill them.
            itself.send_signal(signal.SIGQUIT)
        except psutil.NoSuchProcess:
            pass

def get_zmq_socket(
    context: zmq.Context, socket_type: zmq.SocketType, endpoint: str, bind: bool
) -> zmq.Socket:
    mem = psutil.virtual_memory()
    total_mem = mem.total / 1024**3
    available_mem = mem.available / 1024**3
    if total_mem > 32 and available_mem > 16:
        buf_size = int(0.5 * 1024**3)
    else:
        buf_size = -1

    socket = context.socket(socket_type)
    if endpoint.find("[") != -1:
        socket.setsockopt(zmq.IPV6, 1)

    def set_send_opt():
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, buf_size)

    def set_recv_opt():
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)

    if socket_type == zmq.PUSH:
        set_send_opt()
    elif socket_type == zmq.PULL:
        set_recv_opt()
    elif socket_type == zmq.DEALER:
        set_send_opt()
        set_recv_opt()
    else:
        raise ValueError(f"Unsupported socket type: {socket_type}")

    if bind:
        socket.bind(endpoint)
    else:
        socket.connect(endpoint)

    return socket

def is_port_available(port):
    """Return whether a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", port))
            s.listen(1)
            return True
        except socket.error:
            return False
        except OverflowError:
            return False
