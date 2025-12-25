import argparse
import logging
import os
import pickle  # Used for both Request and Response now
import signal
import socket
import struct
import subprocess
import threading
import time
from typing import Any, Dict, List, Union

import torch
import torch.multiprocessing as mp

# Mock imports for the sake of structure (Keep your original imports here)
from batchgen.server_worker_main_loop import server_worker_main
from batchgen.utils import config_torch_module_initializer
from batchgen.batchgen_worker import BatchGenWorkerArgs
from batchgen.models.engine_loader import core_engine as bg_lib

from batchgen.parameter_server_client import ParameterServerClient

from batchgen.kv_cache.host_kv_mananger_config import build_host_kv_config


# Configure logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - [BatchGenServer] - %(levelname)s - %(message)s'
)

PARAMETER_SERVER_ENDPOINT_ENV = "BATCHGEN_PARAMETER_SERVER_ENDPOINT"

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
		"""Loads model weights via local or external parameter server."""
		logging.info("Loading model resources for %s", self.args.model)
		endpoint = os.getenv(PARAMETER_SERVER_ENDPOINT_ENV)
		hf_cache_dir = self.args.hf_cache_dir or os.path.expanduser("~/.cache/huggingface")
		self.args.hf_cache_dir = hf_cache_dir
		pt_ckpt_dir = self.args.pt_ckpt_dir or os.path.join(self.args.cache_dir or ".", "pt_ckpt")
		self.args.pt_ckpt_dir = pt_ckpt_dir

		if not endpoint and self.args.cache_dir is None:
			self.args.cache_dir = self._download_model_snapshot(hf_cache_dir)

		if endpoint:
			self._load_model_from_remote_server(endpoint, hf_cache_dir, pt_ckpt_dir)
		else:
			self._load_model_locally(hf_cache_dir, pt_ckpt_dir)

		self._configure_host_kv_cache_budget()
		logging.info("Model Loaded. SHM: %s", self.model_info['shm_name'])

	def allocate_host_kv_cache(self, host_kv_cache_size_gb: int):
		"""Allocates shared host kv cache."""
		config = build_host_kv_config(
			host_kv_cache_size=host_kv_cache_size_gb * (1024**3),
			model_name=self.args.model
		)
		host_paged_kv_manager = bg_lib.MLAHostPagedKVManager(config)
		host_paged_kv_manager.initialize(True)
		return host_paged_kv_manager

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
			host_kv_cache_size=self.args_dict['host_kv_cache_size_per_rank'],
			global_host_kv_cache_size_gb=self.args.host_kv_cache_size,
			skeleton_state_dict=self.skeleton_state_dict,

			# Place holder
			local_rank=-1,
			global_rank=-1,
			device=-1,
		)
		logging.info(f"host KV cache size per rank: {self.batchgen_worker_args.host_kv_cache_size} GB")
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
			
			# 1. Allocate KV & Load Model & Spawn Workers
			self.allocate_host_kv_cache(self.args.host_kv_cache_size)
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

	def process_request(self, request: Any) -> Dict:
		command = None
		queries = []
		
		# Defaults
		max_input_len = 1024
		max_output_len = 128
		ignore_eos = False  # NEW: Default to respecting EOS

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
			ignore_eos = request.get('ignore_eos', False)  # NEW: Extract ignore_eos
		else:
			return {'status': 'error', 'message': 'Invalid input type.'}
		
		if command == 'ping':
			return {'status': 'success', 'message': 'pong'}

		elif command == 'reload':
			# Hot-reload worker code without restarting the server
			logging.info("Received reload command, sending to workers...")
			with self.inference_lock:
				self.request_queue.put({"command": "reload"})
				result = self.response_queue.get()
				return {'status': 'success', 'reload_result': result}

		elif command == 'submit_inference':
			if not queries:
				return {'status': 'error', 'message': 'Empty queries list'}

			logging.info(
				f"Processing batch of {len(queries)} items "
				f"(ignore_eos={ignore_eos}, max_output_len={max_output_len})"
			)
			
			with self.inference_lock:
				start_t = time.perf_counter()
				
				# Pass ignore_eos to worker
				worker_payload = {
					"prompts": queries,
					"max_input_len": max_input_len,
					"max_output_len": max_output_len,
					"ignore_eos": ignore_eos,  # NEW: Pass to worker
				}
				
				self.request_queue.put(worker_payload)
				
				result = self.response_queue.get()
				dur = time.perf_counter() - start_t
				logging.info(f"Batch finished in {dur:.2f}s")
				
				# Check if worker returned an error
				if isinstance(result, dict) and 'error' in result:
					logging.error(f"Worker inference failed: {result}")
					return {'status': 'error', 'message': result.get('error', 'Unknown worker error')}
				
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

	def _download_model_snapshot(self, hf_cache_dir: str) -> str:
		logging.info("Downloading model artifacts to %s", hf_cache_dir)
		from huggingface_hub import snapshot_download
		try:
			return snapshot_download(
				self.args.model,
				cache_dir=hf_cache_dir,
				ignore_patterns=["flax*", "tf*"],
			)
		except Exception as exc:  # pragma: no cover - network failure message
			raise RuntimeError(f"Failed to download model: {exc}") from exc

	def _load_model_locally(self, _hf_cache_dir: str, pt_ckpt_dir: str) -> None:
		
		if "deepseek" in self.args.model.lower():
			from batchgen.models.deepseek.deepseek_parameter_server import (
				DeepSeek_Parameter_Server,
			)
			ps = DeepSeek_Parameter_Server(
				self.args.model, self.args.cache_dir, pt_ckpt_dir, self.args.enable_hugetlbfs
			)
		elif "mixtral" in self.args.model.lower():
			from batchgen.models.mixtral.mixtral_parameter_server import (
				Mixtral_Parameter_Server,
			)
			ps = Mixtral_Parameter_Server(
				self.args.model, self.args.cache_dir, pt_ckpt_dir
			)
		else:
			raise NotImplementedError(f"Model type for {self.args.model} not supported")

		shm_name, tensor_meta_shm_name = ps.Init()
		ps_size = ps.parameter_server.byte_size()
		self.skeleton_state_dict = ps.parameter_server.get_skeleton_state_dict()
		self.parameter_server_instance = ps
		self.model_info = {
			"huggingface_ckpt_name": self.args.model,
			"shm_name": shm_name,
			"tensor_meta_shm_name": tensor_meta_shm_name,
			"pt_ckpt_dir": pt_ckpt_dir,
			"parameter_server_size": ps_size,
		}
		logging.info("Local parameter server initialized: %s", self.model_info)

	def _load_model_from_remote_server(
		self, endpoint: str, hf_cache_dir: str, pt_ckpt_dir: str
	) -> None:
		host, port = self._parse_parameter_server_endpoint(endpoint)
		logging.info(
			"Using external parameter server at %s:%d", host, port
		)
		client = ParameterServerClient(host=host, port=port)
		client.load_model(
			huggingface_ckpt_name=self.args.model,
			hf_cache_dir=hf_cache_dir,
			cache_dir=self.args.cache_dir,
			pt_ckpt_dir=pt_ckpt_dir,
		)
		info = client.get_model_info()
		for required in ("shm_name", "tensor_meta_shm_name", "parameter_server_size"):
			if required not in info:
				raise RuntimeError(
					f"Remote parameter server response missing '{required}'"
				)
		skeleton = info.get("skeleton_state_dict")
		if skeleton is None:
			raise RuntimeError(
				"Remote parameter server did not return a skeleton_state_dict"
			)
		self.skeleton_state_dict = skeleton
		self.parameter_server_instance = None
		self.model_info = {
			"huggingface_ckpt_name": info.get(
				"huggingface_ckpt_name", self.args.model
			),
			"shm_name": info["shm_name"],
			"tensor_meta_shm_name": info["tensor_meta_shm_name"],
			"pt_ckpt_dir": info.get("pt_ckpt_dir", pt_ckpt_dir),
			"parameter_server_size": info["parameter_server_size"],
		}
		self.args.pt_ckpt_dir = self.model_info["pt_ckpt_dir"]
		if not self.args.cache_dir:
			self.args.cache_dir = info.get("cache_dir") or self.args.pt_ckpt_dir
		logging.info("Fetched shared memory handles from remote parameter server")

	def _configure_host_kv_cache_budget(self) -> None:
		if self.args.host_kv_cache_size is not None:
			available_mem = self.args.host_kv_cache_size
		else:
			import psutil
			mem = psutil.virtual_memory()
			available_mem = (mem.total - (20 * 1024**3)) // (1024**3)
		if available_mem <= 0:
			raise RuntimeError("Unable to determine host KV cache budget")
		num_devices = torch.cuda.device_count()
		self.args_dict = vars(self.args)
		if num_devices > 0:
			self.args_dict["host_kv_cache_size_per_rank"] = available_mem // num_devices
		else:
			self.args_dict["host_kv_cache_size_per_rank"] = available_mem

	@staticmethod
	def _parse_parameter_server_endpoint(endpoint: str) -> tuple[str, int]:
		value = (endpoint or "").strip()
		if not value:
			raise ValueError("BATCHGEN_PARAMETER_SERVER_ENDPOINT is empty")
		if ":" in value:
			host, port_str = value.rsplit(":", 1)
			host = host or "localhost"
			try:
				port = int(port_str)
			except ValueError as exc:
				raise ValueError(
					f"Invalid port in parameter server endpoint: {value}"
				) from exc
		else:
			host = value
			port = 10900
		return host, port

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
		gpu_arch: str = "hopper"
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
	parser.add_argument("--gpu-arch", type=str)
	parser.add_argument("--nnodes", type=int, default=1)
	parser.add_argument("--node-rank", type=int, default=0)
	parser.add_argument("--world-size", type=int, default=1)
	
	return parser.parse_args()

if __name__ == "__main__":
	mp.set_start_method("spawn", force=True)
	args = parse_args()
	server = BatchGenServer(args)
	server.start()