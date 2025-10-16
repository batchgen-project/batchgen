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
    elif socket_type == zmq.PUB:
        set_send_opt()
    elif socket_type == zmq.SUB:
        set_recv_opt()
        # Subscribe to all messages by default
        socket.setsockopt(zmq.SUBSCRIBE, b"")
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


class ZMQBroadcaster:
    """
    ZMQ-based broadcast communication class.
    Rank 0 broadcasts objects to all other ranks.
    
    Uses PUB-SUB pattern where:
    - Rank 0 is the publisher (PUB socket)
    - All other ranks are subscribers (SUB socket)
    
    Uses barrier synchronization to ensure all subscribers are connected
    before broadcasting begins.
    """
    
    def __init__(self, rank: int, world_size: int, endpoint: str, context: zmq.Context = None, barrier_endpoint: str = None):
        """
        Initialize ZMQ broadcaster.
        
        Args:
            rank: Current process rank (0 to world_size-1)
            world_size: Total number of processes
            endpoint: ZMQ endpoint address (e.g., "tcp://127.0.0.1:5555")
            context: ZMQ context (creates new one if None)
            barrier_endpoint: ZMQ endpoint for barrier synchronization (auto-generated if None)
        """
        self.rank = rank
        self.world_size = world_size
        self.endpoint = endpoint
        self.context = context if context is not None else zmq.Context()
        self.socket = None
        self.barrier_socket = None
        self.ack_socket = None
        
        # Auto-generate barrier endpoint if not provided
        if barrier_endpoint is None:
            # Extract port from endpoint and use next port for barrier
            import re
            match = re.search(r':(\d+)$', endpoint)
            if match:
                base_port = int(match.group(1))
                barrier_endpoint = endpoint.rsplit(':', 1)[0] + f':{base_port + 1000}'
            else:
                barrier_endpoint = endpoint + '_barrier'
        
        self.barrier_endpoint = barrier_endpoint
        self.ack_endpoint = barrier_endpoint.rsplit(':', 1)[0] + ':' + str(int(barrier_endpoint.rsplit(':', 1)[1]) + 1)
        
        if self.rank == 0:
            # Rank 0 is the broadcaster (PUB socket)
            self.socket = get_zmq_socket(
                context=self.context,
                socket_type=zmq.PUB,
                endpoint=self.endpoint,
                bind=True
            )
            logging.info(f"Rank {self.rank}: Broadcasting on {self.endpoint}")
            
            # Setup barrier socket (PULL to collect ready signals)
            self.barrier_socket = self.context.socket(zmq.PULL)
            self.barrier_socket.bind(self.barrier_endpoint)
            logging.info(f"Rank {self.rank}: Barrier setup on {self.barrier_endpoint}")
            
            # Setup ack socket (PUB to send ready signal)
            self.ack_socket = self.context.socket(zmq.PUB)
            self.ack_socket.bind(self.ack_endpoint)
            logging.info(f"Rank {self.rank}: Ack socket on {self.ack_endpoint}")
            
            # Small delay to ensure ack socket is bound before subscribers connect
            import time
            time.sleep(0.1)
            
            # Wait for all subscribers to signal ready
            self._barrier_wait()
            
        else:
            # Setup ack socket first (SUB to receive ready signal)
            self.ack_socket = self.context.socket(zmq.SUB)
            self.ack_socket.setsockopt(zmq.SUBSCRIBE, b"")
            self.ack_socket.connect(self.ack_endpoint)
            logging.info(f"Rank {self.rank}: Connected to ack socket {self.ack_endpoint}")
            
            # Other ranks are subscribers (SUB socket)
            self.socket = get_zmq_socket(
                context=self.context,
                socket_type=zmq.SUB,
                endpoint=self.endpoint,
                bind=False
            )
            logging.info(f"Rank {self.rank}: Subscribed to {self.endpoint}")
            
            # Setup barrier socket (PUSH to signal ready)
            self.barrier_socket = self.context.socket(zmq.PUSH)
            self.barrier_socket.connect(self.barrier_endpoint)
            logging.info(f"Rank {self.rank}: Connected to barrier {self.barrier_endpoint}")
            
            # Signal that this subscriber is ready
            self._barrier_signal()
    
    def _barrier_wait(self):
        """Wait for all subscribers to be ready (rank 0 only)."""
        if self.rank != 0:
            return
        
        expected_signals = self.world_size - 1  # All ranks except rank 0
        logging.info(f"Rank {self.rank}: Waiting for {expected_signals} subscribers at barrier")
        
        for i in range(expected_signals):
            msg = self.barrier_socket.recv()
            logging.info(f"Rank {self.rank}: Barrier signal {i+1}/{expected_signals} received")
        
        logging.info(f"Rank {self.rank}: All subscribers ready, sending ack")
        
        # Send acknowledgment to all subscribers via PUB socket
        import time
        time.sleep(0.1)  # Allow subscribers to connect to ack socket
        
        for _ in range(3):  # Send multiple times to ensure delivery with PUB-SUB
            self.ack_socket.send(b'ready')
            time.sleep(0.05)
        
        logging.info(f"Rank {self.rank}: Ack sent, all ready")
    
    def _barrier_signal(self):
        """Signal that this subscriber is ready (non-zero ranks only)."""
        if self.rank == 0:
            return
        
        import time
        # Small delay to ensure connection is established
        time.sleep(0.05)
        
        self.barrier_socket.send(f"rank_{self.rank}_ready".encode('utf-8'))
        logging.info(f"Rank {self.rank}: Sent barrier signal")
        
        # Wait for acknowledgment from rank 0
        poller = zmq.Poller()
        poller.register(self.ack_socket, zmq.POLLIN)
        
        timeout = 10000  # 10 seconds
        if poller.poll(timeout):
            msg = self.ack_socket.recv()
            logging.info(f"Rank {self.rank}: Received barrier acknowledgment")
        else:
            logging.warning(f"Rank {self.rank}: Barrier ack timeout")
        
        poller.unregister(self.ack_socket)
    
    def broadcast(self, obj):
        """
        Broadcast an object from rank 0 to all other ranks.
        
        Args:
            obj: Object to broadcast (Pydantic BaseModel instance or dict with '__class__' key)
            
        Returns:
            The object itself (rank 0 returns input, others return received object)
            
        Note:
            Objects are assumed to be Pydantic BaseModel instances.
            They are serialized to JSON for efficient transmission.
            A barrier synchronization is performed after broadcast to ensure all ranks have received.
        """
        import json
        from pydantic import BaseModel
        
        if self.rank == 0:
            # Rank 0 sends the object
            if isinstance(obj, BaseModel):
                # Serialize Pydantic model to JSON with class info
                obj_dict = obj.model_dump()
                obj_dict['__class__'] = f"{obj.__class__.__module__}.{obj.__class__.__name__}"
                serialized = json.dumps(obj_dict).encode('utf-8')
            elif isinstance(obj, dict):
                # Already a dict, just serialize
                serialized = json.dumps(obj).encode('utf-8')
            else:
                # Fallback to pickle for other types
                import pickle
                serialized = pickle.dumps(obj)
                
            self.socket.send(serialized)
            logging.debug(f"Rank {self.rank}: Broadcasted object of size {len(serialized)} bytes")
            
            # Wait for all subscribers to acknowledge receipt
            self._post_broadcast_barrier()
            
            return obj
        else:
            # Other ranks receive the object
            serialized = self.socket.recv()
            
            try:
                # Try JSON deserialization first
                obj_dict = json.loads(serialized.decode('utf-8'))
                
                # Check if we need to reconstruct a Pydantic model
                if isinstance(obj_dict, dict) and '__class__' in obj_dict:
                    class_path = obj_dict.pop('__class__')
                    module_name, class_name = class_path.rsplit('.', 1)
                    
                    # Import the class dynamically
                    import importlib
                    module = importlib.import_module(module_name)
                    obj_class = getattr(module, class_name)
                    
                    # Reconstruct the Pydantic model
                    obj = obj_class(**obj_dict)
                else:
                    # Return as dict
                    obj = obj_dict
                    
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Fallback to pickle if JSON fails
                import pickle
                obj = pickle.loads(serialized)
                
            logging.debug(f"Rank {self.rank}: Received object of size {len(serialized)} bytes")
            
            # Signal that this rank has received the broadcast
            self._post_broadcast_barrier()
            
            return obj
    
    def _post_broadcast_barrier(self):
        """Barrier synchronization after broadcast to ensure all ranks have received."""
        if self.rank == 0:
            # Rank 0 waits for acknowledgments from all subscribers
            expected_acks = self.world_size - 1
            logging.debug(f"Rank {self.rank}: Waiting for {expected_acks} broadcast acknowledgments")
            
            for i in range(expected_acks):
                msg = self.barrier_socket.recv()
                logging.debug(f"Rank {self.rank}: Received ack {i+1}/{expected_acks}")
            
            logging.debug(f"Rank {self.rank}: All ranks acknowledged broadcast")
        else:
            # Other ranks send acknowledgment
            self.barrier_socket.send(f"rank_{self.rank}_ack".encode('utf-8'))
            logging.debug(f"Rank {self.rank}: Sent broadcast acknowledgment")
    
    def __len__(self):
        """Return the world size (total number of processes)."""
        return self.world_size
    
    def close(self):
        """Close the ZMQ sockets."""
        if self.socket is not None:
            self.socket.close()
            logging.info(f"Rank {self.rank}: Closed broadcaster socket")
        
        if self.barrier_socket is not None:
            self.barrier_socket.close()
            logging.info(f"Rank {self.rank}: Closed barrier socket")
        
        if self.ack_socket is not None:
            self.ack_socket.close()
            logging.info(f"Rank {self.rank}: Closed ack socket")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
