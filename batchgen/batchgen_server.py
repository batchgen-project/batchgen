import os
import torch
import logging
import argparse
import torch.multiprocessing as mp
from typing import Dict, Any, List, Union
import threading
import signal
import subprocess
import time
import socket
import struct
import pickle  # Used for both Request and Response now

# Mock imports for the sake of structure (Keep your original imports here)
from batchgen.server_worker_main_loop import server_worker_main
from batchgen.utils import config_torch_module_initializer
from batchgen.batchgen_worker import BatchGenWorkerArgs

# Configure logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - [BatchGenServer] - %(levelname)s - %(message)s'
)

class BatchGenServer:
	def __init__(self, args):
		self.args = args
		self.host = args.host
		self.port = args.port

		# Networking
		self.server_socket = None
		self.clients = []
		self.running = False

		# Process Management
		self.mp_ctx = mp.get_context('spawn')
		# Queue for sending batch data to Rank 0 worker
		self.request_queue = self.mp_ctx.Queue() 
		# Queue for receiving inference results from Rank 0 worker
		self.response_queue = self.mp_ctx.Queue()
		self.worker_process = None

		# State
		# Lock is crucial: It ensures multiple TCP clients don't push 
		# interleaved data into the queue simultaneously.
		self.inference_lock = threading.Lock() 
		self.model_info = {}
		self.parameter_server_instance = None
		self.args_dict = {}
		self.batchgen_worker_args = None

		# Signals
		signal.signal(signal.SIGINT, self.handle_shutdown)
		signal.signal(signal.SIGTERM, self.handle_shutdown)

	def config_hugepages(self):
		"""Configure hugepages for shared memory usage."""
		try:
			commands = [
				['sysctl', '-w', 'vm.nr_hugepages=350000'],
				['mkdir', '-p', '/dev/hugepages'],
				['mount', '-t', 'hugetlbfs', 'none', '/dev/hugepages']
			]
			for cmd in commands:
				subprocess.run(cmd, check=True, capture_output=True, text=True)
		except Exception as e:
			logging.warning(f"Hugepages configuration failed: {e}")
	
	def load_model_resources(self):
		"""Loads model into Parameter Server (Shared Memory)."""
		logging.info(f"Loading model: {self.args.model}")
		
		hf_cache_dir = self.args.hf_cache_dir or os.path.expanduser("~/.cache/huggingface")
		pt_ckpt_dir = self.args.pt_ckpt_dir or os.path.join(self.args.cache_dir or ".", "pt_ckpt")
		
		if self.args.cache_dir is None:
			logging.info("Downloading model from Hugging Face")
			from huggingface_hub import snapshot_download
			try:
				model_path = snapshot_download(
					self.args.model,
					cache_dir=hf_cache_dir,
					ignore_patterns=["flax*", "tf*"],
				)
				self.args.cache_dir = model_path
			except Exception as e:
				raise RuntimeError(f"Failed to download model: {e}")

			

		# --- Initialize Parameter Server Logic ---
		# (This part relies on your specific implementation modules)
		if "deepseek" in self.args.model.lower():
			from batchgen.models.deepseek.deepseek_parameter_server import DeepSeek_Parameter_Server
			ps = DeepSeek_Parameter_Server(self.args.model, self.args.cache_dir, pt_ckpt_dir, self.args.enable_hugetlbfs)
		elif "Mixtral" in self.args.model:
			from batchgen.models.mixtral.mixtral_parameter_server import Mixtral_Parameter_Server
			ps = Mixtral_Parameter_Server(self.args.model, self.args.cache_dir, pt_ckpt_dir)
		else:
			raise NotImplementedError(f"Model type for {self.args.model} not supported")

		shm_name, tensor_meta_shm_name = ps.Init()
		ps_size = ps.parameter_server.byte_size()
		
		self.parameter_server_instance = ps
		self.model_info = {
			'huggingface_ckpt_name': self.args.model,
			'shm_name': shm_name,
			'tensor_meta_shm_name': tensor_meta_shm_name,
			'pt_ckpt_dir': pt_ckpt_dir,
			'parameter_server_size': ps_size,
		}
		logging.info(self.model_info)
		# Calculate Host Memory for Workers
		import psutil
		mem = psutil.virtual_memory()
		# Reserve 20GB for OS/PS overhead
		available_mem = mem.total - (20 * 1024**3) 
		num_devices = torch.cuda.device_count()
		
		self.args_dict = vars(self.args)
		if num_devices > 0:
			self.args_dict['host_kv_cache_size_per_rank'] = available_mem // num_devices
		else:
			self.args_dict['host_kv_cache_size_per_rank'] = available_mem

		logging.info(f"Model Loaded. SHM: {shm_name}")

	def spawn_workers(self):
		"""Spawns DDP Workers via mp.spawn."""
		local_device_count = torch.cuda.device_count()
		if local_device_count == 0:
			logging.error("No CUDA devices found. Exiting.")
			exit(1)
			

		logging.info(f"Spawning {local_device_count} DDP workers...")
		
		self.batchgen_worker_args = BatchGenWorkerArgs(
			model_name=self.args.model,
			hf_cache_dir=self.args.hf_cache_dir,
			cache_dir=self.args.cache_dir,
			pt_ckpt_dir=self.args.pt_ckpt_dir,
			kv_dtype=self.args.kv_dtype,
			dist_init_addr=self.args.dist_init_addr,
			world_size=self.args.world_size,
			nnode_rank=self.args.node_rank,
			nnodes=self.args.nnodes,
			gpu_arch=self.args.gpu_arch,

			shm_name=self.model_info['shm_name'],
			tensor_meta_shm_name=self.model_info['tensor_meta_shm_name'],
			enable_hugetlbfs=self.args.enable_hugetlbfs,
			weight_byte_size=self.model_info['parameter_server_size'],

			# Place holder
			local_rank=-1,
			global_rank=-1,
			device=-1,
		)
			
		self.worker_process = mp.spawn(
			server_worker_main,
			args=(
				self.request_queue,
				self.response_queue,
				self.batchgen_worker_args,
			),
			nprocs=local_device_count,
			join=False,
			daemon=True
		)

	def start(self):
		"""Start the TCP Server loop"""
		try:
			if self.args.enable_hugetlbfs:
				self.config_hugepages()
			
			# Initialize custom torch modules if needed
			config_torch_module_initializer()
			
			# 1. Load Model & Spawn Workers
			self.load_model_resources()
			self.spawn_workers()
			
			# 2. Start TCP Listener
			self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
			self.server_socket.bind((self.host, self.port))
			self.server_socket.listen(20) # Increased backlog slightly
			
			self.running = True
			logging.info(f"BatchGenServer listening on {self.host}:{self.port}")
			
			while self.running:
				try:
					self.server_socket.settimeout(1.0)
					client_sock, addr = self.server_socket.accept()
					
					t = threading.Thread(
						target=self.handle_client,
						args=(client_sock, addr),
						daemon=True
					)
					t.start()
					self.clients.append(t)
					
				except socket.timeout:
					continue
				except Exception as e:
					if self.running:
						logging.error(f"Accept error: {e}")

		except Exception as e:
			logging.error(f"Server Startup Failed: {e}")
			import traceback
			traceback.print_exc()
		finally:
			self.cleanup()

	def handle_client(self, conn, addr):
		"""Handle individual client connection."""
		# logging.info(f"Client connected: {addr}")
		try:
			while self.running:
				# 1. Read 4-byte length header
				size_data = conn.recv(4)
				if not size_data: break
				msg_size = struct.unpack('!I', size_data)[0]
				
				# 2. Read payload
				data = b''
				while len(data) < msg_size:
					packet = conn.recv(min(4096, msg_size - len(data)))
					if not packet: break
					data += packet
				
				if len(data) < msg_size: break
				
				# 3. Deserialize (Pickle allows List[str] or List[dict] natively)
				try:
					request = pickle.loads(data)
				except Exception as e:
					logging.error(f"Deserialization error from {addr}: {e}")
					break

				# 4. Process Logic
				# print(request)
				response = self.process_request(request)
				
				# 5. Send Response (Pickled)
				resp_bytes = pickle.dumps(response)
				conn.sendall(struct.pack('!I', len(resp_bytes)))
				conn.sendall(resp_bytes)
				
		except ConnectionResetError:
			pass # Client disconnected abruptly
		except Exception as e:
			logging.error(f"Client {addr} error: {e}")
		finally:
			conn.close()

	# In BatchGenServer class
	def process_request(self, request: Any) -> Dict:
		command = None
		queries = []
		
		# Defaults
		max_input_len = 1024
		max_output_len = 128

		# --- LOGIC UPDATE START ---
		if isinstance(request, list):
			# Backwards compatibility for raw lists
			command = 'submit_inference'
			queries = request
		elif isinstance(request, dict):
			command = request.get('command')
			queries = request.get('queries', [])
			# Extract params from client request
			max_input_len = request.get('max_input_len', 1024)
			max_output_len = request.get('max_output_len', 128)
		# --- LOGIC UPDATE END ---
		else:
			return {'status': 'error', 'message': 'Invalid input type.'}
		
		if command == 'ping':
			return {'status': 'success', 'message': 'pong'}

		elif command == 'submit_inference':
			if not queries:
				return {'status': 'error', 'message': 'Empty queries list'}

			logging.info(f"Processing batch of {len(queries)} items.")
			
			with self.inference_lock:
				start_t = time.perf_counter()
				
				# --- CRITICAL FIX: Wrap data into the Dict expected by Worker ---
				worker_payload = {
					"prompts": queries,         # Remap 'queries' to 'prompts'
					"max_input_len": max_input_len,
					"max_output_len": max_output_len
				}
				
				self.request_queue.put(worker_payload)
				# ---------------------------------------------------------------
				
				result = self.response_queue.get()
				dur = time.perf_counter() - start_t
				logging.info(f"Batch finished in {dur:.2f}s")
				
				return {'status': 'success', 'results': result}

		return {'status': 'error', 'message': f'Unknown command: {command}'}

	def handle_shutdown(self, signum, frame):
		logging.info("Shutdown signal received...")
		self.running = False

	def cleanup(self):
		logging.info("Cleaning up...")
		self.running = False
		
		# Poison Pill for workers
		try:
			if self.request_queue:
				self.request_queue.put(None)
		except:
			pass

		if self.server_socket:
			try: self.server_socket.close()
			except: pass

# --- Entry Point ---

"""
	def __init__(
		self,
		huggingface_ckpt_name: str,
		hf_cache_dir: Optional[str],
		cache_dir: Optional[str],
		pt_ckpt_dir: Optional[str],
		queries: List[str],
		max_input_length: int,
		max_decoding_length: int,
		device: int,
		skeleton_state_dict,
		shm_name,
		tensor_meta_shm_name,
		engine_config_json_dir = None, # Will be deprecated in the future
		host_kv_cache_size: Optional[int] = None,
		kv_dtype: str = "bfloat16",
		dist_init_addr: str = "localhost:12355",
		local_rank: Optional[int] = 0,
		global_rank: Optional[int] = 0,
		world_size: Optional[int] = 1,
		gpu_arch: str = "hooper"
	):


"""

def parse_args():
	parser = argparse.ArgumentParser(description="BatchGen Inference Server")
	parser.add_argument("--host", type=str, default="0.0.0.0") # Listen on all interfaces
	parser.add_argument("--port", type=int, default=10900)
	parser.add_argument("--model", type=str, required=True, help="HuggingFace Model Name")
	parser.add_argument("--hf-cache-dir", type=str, default=None)
	parser.add_argument("--cache-dir", type=str, default=None)
	parser.add_argument("--pt-ckpt-dir", type=str, default=None)
	parser.add_argument("--enable-hugetlbfs", action='store_false')
	parser.add_argument("--dist-init-addr", type=str)
	parser.add_argument("--kv-dtype", type=str, default="bfloat16")
	parser.add_argument("--host-kv-cache-size", type=int, default=None)
	parser.add_argument("--gpu-arch", type=str, default=None)
	parser.add_argument("--nnodes", type=int, default=1)
	parser.add_argument("--node-rank", type=int, default=0)
	parser.add_argument("--world-size", type=int, default=1)
	
	return parser.parse_args()

if __name__ == "__main__":
	mp.set_start_method("spawn", force=True)
	args = parse_args()
	server = BatchGenServer(args)
	server.start()