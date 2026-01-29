"""
Standalone Parameter Server for BatchGen

This script starts a long-running parameter server that hosts model weights in shared memory.
It uses socket communication to handle requests from client processes.
"""
import atexit
import os
import sys
import tempfile
import time
import logging
import argparse
import socket
import json
import pickle
import struct
import threading
import subprocess
import signal
import multiprocessing
from multiprocessing import shared_memory
import numpy as np
from typing import Dict, Any, Optional, Tuple
import torch
import torch.distributed as dist
from batchgen.utils import config_torch_module_initializer
config_torch_module_initializer()
# Configure logging

logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class ParameterServer:
	def __init__(self, host='localhost', port=10900, model_name=None,
				 hf_cache_dir=None, cache_dir=None, converted_ckpt_dir=None, enable_hugetlbfs=False):
		"""
		Initialize the Parameter Server.
		
		Args:
			host: Host to bind the server socket to
			port: Port to listen on
			model_name: HuggingFace model name to load at startup
			hf_cache_dir: HuggingFace cache directory
			cache_dir: Model cache directory
			converted_ckpt_dir: Directory for PyTorch checkpoints
		"""
		self.host = host
		self.port = port
		self.server_socket = None
		self.clients = []
		self.running = False
		self.enable_hugetlbfs = enable_hugetlbfs

		# _init_dist_process_group(0,1)
		# Initial model parameters
		self.initial_model_name = model_name
		self.hf_cache_dir = hf_cache_dir
		self.cache_dir = cache_dir
		self.converted_ckpt_dir = converted_ckpt_dir
		# if self.converted_ckpt_dir is None:
		# 	self.converted_ckpt_dir = os.path.join(cache_dir, "converted_ckpt")
		
		# State tracking
		self.current_model = None
		self.parameter_server_instance = None
		self.model_info = {}
		
		# Shared memory for skeleton state dict
		self.skeleton_state_dict_shm_name = None
		self.skeleton_state_dict_file = None

		# Register atexit cleanup for temp files (handles normal exits, exceptions, etc.)
		atexit.register(self._cleanup_temp_files)

		# Setup signal handlers for graceful shutdown
		signal.signal(signal.SIGINT, self.handle_shutdown)
		signal.signal(signal.SIGTERM, self.handle_shutdown)

	def _cleanup_temp_files(self):
		"""Clean up temporary skeleton state dict file. Called by atexit and signal handlers."""
		if self.skeleton_state_dict_file and os.path.exists(self.skeleton_state_dict_file):
			try:
				logging.info(f"Cleaning up skeleton state dict temp file: {self.skeleton_state_dict_file}")
				os.remove(self.skeleton_state_dict_file)
				self.skeleton_state_dict_file = None
			except Exception as e:
				logging.warning(f"Failed to cleanup temp file {self.skeleton_state_dict_file}: {e}")

	def create_skeleton_state_dict_shared_memory(self, skeleton_state_dict):
		"""
		Create file-based storage for large skeleton state dict with PyTorch compatibility.

		Uses Python's tempfile module for automatic cleanup on process exit.
		The temp file is created in the system temp directory and registered
		for cleanup via atexit.

		Args:
			skeleton_state_dict: The skeleton state dict to put in shared memory

		Returns:
			Name of the file identifier
		"""
		try:
			logging.info("Starting serialization of skeleton state dict...")

			# Use torch.save instead of pickle for PyTorch tensors
			import io
			buffer = io.BytesIO()
			torch.save(skeleton_state_dict, buffer)
			serialized_dict = buffer.getvalue()
			serialized_size = len(serialized_dict)
			logging.info(f"Serialized skeleton state dict size: {serialized_size} bytes")

			# Clean up previous temp file if exists
			self._cleanup_temp_files()

			# Use tempfile.mkstemp for a secure temp file in system temp directory
			# The file persists until explicitly deleted (not auto-deleted on close)
			# This allows worker processes to read it
			fd, file_path = tempfile.mkstemp(suffix='.pt', prefix='batchgen_skel_')

			# Close the file descriptor - we'll use torch.save which opens its own handle
			os.close(fd)

			# Write the file directly with torch.save
			logging.info(f"Writing skeleton state dict to temp file: {file_path}")
			torch.save(skeleton_state_dict, file_path)

			# Verify the file was written correctly
			actual_size = os.path.getsize(file_path)
			logging.info(f"Successfully wrote state dict to temp file, size: {actual_size} bytes")

			# Store the file path for cleanup later (via atexit or signal handlers)
			self.skeleton_state_dict_file = file_path

			# Use the full path as the identifier (clients need to know where to find it)
			file_name = os.path.basename(file_path)
			self.skeleton_state_dict_shm_name = file_name

			return file_name
		except Exception as e:
			logging.error(f"Error creating skeleton state dict temp file: {e}")
			return None
	
	def create_skeleton_state_dict_shared_memory_dep(self, skeleton_state_dict):
		"""
		Create shared memory for skeleton state dict with file backup for reliability
		
		Args:
			skeleton_state_dict: The skeleton state dict to put in shared memory
			
		Returns:
			Name of the shared memory segment
		"""
		# Clean up old shared memory if it exists - but only on the server side
		if self.skeleton_state_dict_shm_name:
			try:
				# Try to clean up old shared memory
				shm = shared_memory.SharedMemory(name=self.skeleton_state_dict_shm_name)
				shm.close()
				shm.unlink()
				self.skeleton_state_dict_shm_name = None
			except Exception as e:
				logging.error(f"Error cleaning up old shared memory: {e}")
		
		# Serialize the skeleton state dict
		try:
			logging.info("Starting serialization of skeleton state dict...")
			serialized_dict = pickle.dumps(skeleton_state_dict)
			serialized_size = len(serialized_dict)
			logging.info(f"Serialized skeleton state dict size: {serialized_size} bytes")
		except Exception as e:
			logging.error(f"Error serializing state dict: {e}")
			raise
		
		# Create a new shared memory segment with a very simple name that works in Docker
		# Avoid any special characters or directory separators
		import random
		import string
		
		# Use a shorter name (Docker might have namespace limitations)
		# and ensure it's unique with timestamp + random chars
		timestamp = int(time.time()) % 10000  # Last 4 digits of current timestamp
		random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
		shm_name = f"skel_{timestamp}_{random_suffix}"
		
		# Create a backup file with the same content (for reliability)
		backup_dir = os.path.join(os.getcwd(), "shared_memory_backup")
		os.makedirs(backup_dir, exist_ok=True)
		backup_file = os.path.join(backup_dir, f"{shm_name}.bin")
		
		try:
			# Write backup file first
			logging.info(f"Writing backup file to {backup_file}")
			with open(backup_file, 'wb') as f:
				# Write size marker at the beginning (32-bit unsigned integer)
				f.write(struct.pack('!I', serialized_size))
				# Write the actual data
				f.write(serialized_dict)
			
			logging.info(f"Backup file written successfully, size: {os.path.getsize(backup_file)} bytes")
			
			# Store backup file path for clients to use
			self.skeleton_state_dict_backup_file = backup_file
			
			# Now create shared memory
			buffer_size = serialized_size + 8
			
			logging.info(f"Creating shared memory segment '{shm_name}' with size {buffer_size} bytes")
			shm = shared_memory.SharedMemory(
				create=True,
				size=buffer_size,
				name=shm_name
			)
			
			logging.info("Writing serialized data to shared memory...")
			# Write in chunks to avoid memory issues with very large dicts
			chunk_size = 100 * 1024 * 1024  # 100MB chunks
			for i in range(0, serialized_size, chunk_size):
				end_pos = min(i + chunk_size, serialized_size)
				chunk = serialized_dict[i:end_pos]
				shm.buf[i:end_pos] = chunk
			
			# Write size marker at the very end (32-bit unsigned integer)
			struct.pack_into('!I', shm.buf, buffer_size - 4, serialized_size)
			
			# Store the name for cleanup later
			self.skeleton_state_dict_shm_name = shm_name
			
			# Verify data was written correctly by reading back the size marker
			size_bytes = bytes(shm.buf[buffer_size - 4:buffer_size])
			verification_size = struct.unpack('!I', size_bytes)[0]
			if verification_size != serialized_size:
				logging.error(f"Size verification failed! Expected {serialized_size}, got {verification_size}")
			else:
				logging.info("Size verification successful")
			
			# We need to keep a reference to the shared memory to prevent automatic cleanup
			self._current_shm = shm
			
			logging.info(f"Successfully created shared memory segment '{shm_name}'")
			return shm_name
			
		except Exception as e:
			logging.error(f"Error creating shared memory: {e}")
			return None

	def config_hugepages(self, model_name: str = None):
		"""
		Configure hugepages for shared memory usage.

		Args:
			model_name: HuggingFace model name to determine required hugepages.
		"""
		from batchgen.server.process_utils import get_hugepage_size, get_model_byte_size

		hugepage_size = get_hugepage_size()

		if model_name is not None:
			byte_size = get_model_byte_size(model_name)
			num_hugepages = (byte_size + hugepage_size - 1) // hugepage_size
			logging.info(
				f"Calculating hugepages for {model_name}: "
				f"{byte_size / (1024**3):.1f} GB model, "
				f"{num_hugepages} pages ({hugepage_size / (1024**2):.0f} MB each)"
			)
		else:
			num_hugepages = 350000
			logging.warning("No model_name provided, using default 350000 hugepages")

		try:
			commands = [
				['sysctl', '-w', f'vm.nr_hugepages={num_hugepages}'],
				['mkdir', '-p', '/dev/hugepages'],
				['mount', '-t', 'hugetlbfs', 'none', '/dev/hugepages']
			]
			for cmd in commands:
				logging.info(f"Running command: {' '.join(cmd)}")
				result = subprocess.run(cmd, check=True, capture_output=True, text=True)
				if result.stdout:
					logging.info(f"Command output: {result.stdout.strip()}")
				if result.stderr:
					logging.warning(f"Command error: {result.stderr.strip()}")
		except subprocess.CalledProcessError as e:
			logging.warning(f"Error configuring hugepages: {e}")
			logging.warning(f"Command output: {e.output.strip()}")
			logging.warning(f"Command error: {e.stderr.strip()}")
			logging.warning(f"Failed to use hugepages, falling back to regular shared memory")
			
			

	def start(self):
		"""Start the parameter server"""
		start_time = time.time()
		# Preload model if specified - BEFORE starting the server socket
		if self.initial_model_name:
			logging.info(f"Preloading model: {self.initial_model_name}")
			try:
				start_time = time.time()
				if self.enable_hugetlbfs:
					self.config_hugepages(self.initial_model_name)
				result = self._preload_model(
					self.initial_model_name,
					self.hf_cache_dir,
					self.cache_dir,
					self.converted_ckpt_dir
				)
				end_time = time.time()
				if result['status'] == 'success':
					logging.info(f"Model {self.initial_model_name} loaded successfully in {end_time - start_time:.2f} seconds")
					logging.info(f"Model loaded with shared memory name: {self.model_info.get('shm_name')}")
					logging.info(f"Parameter server size: {self.model_info.get('parameter_server_size')} bytes")
					# Only start listening for connections AFTER model is loaded
					logging.info("Starting server socket...")
					
					# Start the server socket
					self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
					# Allow address reuse to avoid "address already in use" errors on restart
					self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
					self.server_socket.bind((self.host, self.port))
					self.server_socket.listen(10)  # Allow up to 10 pending connections
					
					self.running = True
					logging.info(f"Parameter server is now listening on {self.host}:{self.port}")
					end_time = time.time()
					logging.info(f"Server started in {end_time - start_time:.2f} seconds")
					
					try:
						while self.running:
							try:
								# Accept client connections with a timeout to allow checking self.running
								self.server_socket.settimeout(1.0)
								client_socket, address = self.server_socket.accept()
								logging.info(f"New client connected: {address}")
								
								# Start a new thread to handle this client
								client_thread = threading.Thread(
									target=self.handle_client,
									args=(client_socket, address),
									daemon=True
								)
								client_thread.start()
								self.clients.append((client_socket, client_thread))
								
							except socket.timeout:
								# This is expected due to the timeout, just continue
								continue
							except Exception as e:
								if self.running:  # Only log if we're not shutting down
									logging.error(f"Error accepting client connection: {e}")
					finally:
						self.cleanup()
				else:
					self.cleanup()
					raise RuntimeError(f"Failed to preload model: {result.get('message', 'Unknown error')}")
			except Exception as e:
				logging.error(f"Failed to preload model {self.initial_model_name}: {e}")
				logging.error("Server will start without a preloaded model")
				self.cleanup()
				raise RuntimeError("Failed to preload model")
				# Continue running the server even if model loading fails
		else:
			# logging.info("No initial model specified. Server starting without preloaded model.")
			self.cleanup()
			raise RuntimeError("No initial model specified.")
		
	
	def cleanup(self):
		"""Clean up resources when shutting down"""
		logging.info("Cleaning up parameter server...")
		
		# Close all client connections
		for client_socket, _ in self.clients:
			try:
				client_socket.close()
			except:
				pass
		
		# Close the server socket
		if self.server_socket:
			try:
				self.server_socket.close()
			except:
				pass
		
		# Clean up model resources
		if self.parameter_server_instance:
			logging.info("Cleaning up model resources...")
			# Any specific cleanup needed for your parameter server
		
		# Clean up shared memory - ONLY during server shutdown
		if hasattr(self, '_current_shm') and self._current_shm:
			try:
				logging.info(f"Cleaning up current shared memory {self.skeleton_state_dict_shm_name}")
				self._current_shm.close()
				self._current_shm.unlink()
				self._current_shm = None
				self.skeleton_state_dict_shm_name = None
			except Exception as e:
				logging.error(f"Error cleaning up current shared memory: {e}")
		
		# Clean up any old shared memory segments we've kept around
		if hasattr(self, '_old_shm_segments'):
			for i, old_shm in enumerate(self._old_shm_segments):
				try:
					logging.info(f"Cleaning up old shared memory segment {i+1}/{len(self._old_shm_segments)}")
					old_shm.close()
					old_shm.unlink()
				except Exception as e:
					logging.error(f"Error cleaning up old shared memory segment: {e}")
			self._old_shm_segments = []

		# clean up huge pages setting
		try:
			command = ['sysctl', '-w', 'vm.nr_hugepages=0']
			logging.info(f"Running command to clean up hugepages: {' '.join(command)}")
			result = subprocess.run(command, check=True, capture_output=True, text=True)
			if result.stdout:
				logging.info(f"Command output: {result.stdout.strip()}")
			if result.stderr:
				logging.warning(f"Command error: {result.stderr.strip()}")
		except subprocess.CalledProcessError as e:
			logging.warning(f"Error cleaning up hugepages: {e}")
			logging.warning(f"Command output: {e.output.strip()}")
			logging.warning(f"Command error: {e.stderr.strip()}")
			logging.warning("Failed to clean up hugepages, you may need to manually clear by `sysctl -w vm.nr_hugepages=0`")
		
		logging.info("Parameter server shutdown complete")
	
	# def cleanup(self):
	# 	"""Clean up resources when shutting down"""
	# 	logging.info("Cleaning up parameter server...")
		
	# 	# Close all client connections
	# 	for client_socket, _ in self.clients:
	# 		try:
	# 			client_socket.close()
	# 		except:
	# 			pass
		
	# 	# Close the server socket
	# 	if self.server_socket:
	# 		try:
	# 			self.server_socket.close()
	# 		except:
	# 			pass
		
	# 	# Clean up model resources
	# 	if self.parameter_server_instance:
	# 		logging.info("Cleaning up model resources...")
	# 		# Any specific cleanup needed for your parameter server
		
	# 	# Clean up skeleton_state_dict temporary file if it exists
	# 	if hasattr(self, 'skeleton_state_dict_file') and self.skeleton_state_dict_file:
	# 		try:
	# 			if os.path.exists(self.skeleton_state_dict_file):
	# 				logging.info(f"Removing skeleton state dict temporary file: {self.skeleton_state_dict_file}")
	# 				os.remove(self.skeleton_state_dict_file)
	# 				self.skeleton_state_dict_file = None
	# 		except Exception as e:
	# 			logging.error(f"Error removing skeleton state dict temporary file: {e}")
		
	# 	# Clean up shared memory - ONLY during server shutdown
	# 	if hasattr(self, '_current_shm') and self._current_shm:
	# 		try:
	# 			logging.info(f"Cleaning up current shared memory {self.skeleton_state_dict_shm_name}")
	# 			self._current_shm.close()
	# 			self._current_shm.unlink()
	# 			self._current_shm = None
	# 			self.skeleton_state_dict_shm_name = None
	# 		except Exception as e:
	# 			logging.error(f"Error cleaning up current shared memory: {e}")
		
	# 	# Clean up any old shared memory segments we've kept around
	# 	if hasattr(self, '_old_shm_segments'):
	# 		for i, old_shm in enumerate(self._old_shm_segments):
	# 			try:
	# 				logging.info(f"Cleaning up old shared memory segment {i+1}/{len(self._old_shm_segments)}")
	# 				old_shm.close()
	# 				old_shm.unlink()
	# 			except Exception as e:
	# 				logging.error(f"Error cleaning up old shared memory segment: {e}")
	# 		self._old_shm_segments = []
		
	# 	logging.info("Parameter server shutdown complete")
		
	def handle_shutdown(self, signum, frame):
		"""Handle shutdown signals"""
		logging.info(f"Received signal {signum}, shutting down...")
		# clean-up /dev/shm or /dev/hugepages
		# if self.model_info['shm_name'] is not None, unlink it.
		if self.model_info.get('shm_name'):
			try:
				shm_name = self.model_info.get('shm_name')
				# Remove leading slash if present to avoid double slash in path
				clean_name = shm_name.lstrip('/')
				shm_path = os.path.join("/dev/hugepages", clean_name)

				logging.info(f"Removing shared memory file {shm_path}")

				if os.path.exists(shm_path):
					os.remove(shm_path)
					logging.info(f"Successfully removed {shm_path}")
				else:
					logging.info(f"Shared memory file {shm_path} already cleaned up")
			except Exception as e:
				logging.warning(f"Shared memory cleanup not properly, you may need to manually clear by `rm -f /dev/hugepages/{shm_path}`: {e}")

		# Clean up skeleton state dict temp file
		self._cleanup_temp_files()

		# Clean up hugepages allocation - critical for releasing system memory
		# Use both methods for robustness
		if self.enable_hugetlbfs:
			# Method 1: sysctl
			try:
				command = ['sysctl', '-w', 'vm.nr_hugepages=0']
				logging.info(f"Running command to clean up hugepages: {' '.join(command)}")
				result = subprocess.run(command, check=True, capture_output=True, text=True)
				if result.stdout:
					logging.info(f"Command output: {result.stdout.strip()}")
				if result.stderr:
					logging.warning(f"Command error: {result.stderr.strip()}")
			except subprocess.CalledProcessError as e:
				logging.warning(f"Error cleaning up hugepages via sysctl: {e}")
			except Exception as e:
				logging.warning(f"Unexpected error cleaning up hugepages via sysctl: {e}")

			# Method 2: Direct /proc write (fallback)
			try:
				with open("/proc/sys/vm/nr_hugepages", "w") as f:
					f.write("0\n")
				logging.info("Reset vm.nr_hugepages to 0 via /proc")
			except PermissionError:
				logging.warning("Permission denied writing to /proc/sys/vm/nr_hugepages (need root)")
			except Exception as e:
				logging.warning(f"Failed to reset hugepages via /proc: {e}")

		self.running = False
	
	def handle_client(self, client_socket, address):
		"""Handle communication with a client"""
		try:
			while self.running:
				try:
					# Receive message size first (4 bytes for a 32-bit integer)
					size_data = client_socket.recv(4)
					if not size_data or len(size_data) < 4:
						logging.info(f"Client {address} closed connection (no size data)")
						break  # Connection closed
					
					# Unpack the size
					msg_size = struct.unpack('!I', size_data)[0]
					logging.debug(f"Received message size: {msg_size} bytes from {address}")
					
					# Check for unreasonably large message size
					if msg_size > 100 * 1024 * 1024:  # Limit to 100MB for incoming requests
						logging.warning(f"Rejecting oversized message from {address}: {msg_size} bytes")
						break
					
					# Receive the actual message
					data = b''
					remaining = msg_size
					while remaining > 0:
						chunk = client_socket.recv(min(4096, remaining))
						if not chunk:
							logging.warning(f"Connection closed by {address} during data transfer")
							break  # Connection closed
						data += chunk
						remaining -= len(chunk)
					
					if len(data) < msg_size:
						logging.warning(f"Incomplete message received from {address}: got {len(data)}/{msg_size} bytes")
						break
					
					# Parse the request
					request = json.loads(data.decode('utf-8'))
					logging.info(f"Received request from {address}: {request.get('command', 'unknown')}")
					
					# Process the request
					response = self.process_request(request)
					
					# Prepare the response for sending (pickle it)
					try:
						response_data = pickle.dumps(response)
						response_size = len(response_data)
						
						# Log large responses
						if response_size > 10 * 1024 * 1024:  # 10MB
							logging.info(f"Sending large response to {address}: {response_size} bytes")
						
						# Ensure response size fits within 32-bit unsigned int
						if response_size > 0xFFFFFFFF:  # 2^32 - 1
							logging.error(f"Response size {response_size} exceeds maximum size")
							error_response = {
								'status': 'error',
								'message': 'Response too large to send'
							}
							response_data = pickle.dumps(error_response)
							response_size = len(response_data)
						
						# Send the response size first
						client_socket.sendall(struct.pack('!I', response_size))
						
						# Send the response in chunks to handle large responses
						CHUNK_SIZE = 8192
						for i in range(0, response_size, CHUNK_SIZE):
							end = min(i + CHUNK_SIZE, response_size)
							client_socket.sendall(response_data[i:end])
							
						logging.debug(f"Sent complete response of {response_size} bytes to {address}")
							
					except Exception as e:
						logging.error(f"Error sending response to {address}: {e}")
						break
						
				except (struct.error, ValueError, json.JSONDecodeError) as e:
					logging.error(f"Protocol error with client {address}: {e}")
					break
				except Exception as e:
					logging.error(f"Unexpected error with client {address}: {e}")
					break
		
		except ConnectionResetError:
			logging.info(f"Connection reset by client {address}")
		except Exception as e:
			logging.error(f"Error handling client {address}: {e}")
		finally:
			# Clean up this client
			try:
				client_socket.close()
			except:
				pass
			logging.info(f"Client disconnected: {address}")

	def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
		"""Process a client request and return a response"""
		command = request.get('command')
		
		if command == 'ping':
			return {'status': 'success', 'message': 'pong'}
		
		elif command == 'load_model':
			return self.handle_load_model(request)
		
		elif command == 'get_model_info':
			if not self.current_model:
				return {'status': 'error', 'message': 'No model currently loaded'}
			
			# Return the file path instead of shared memory
			return {
				'status': 'success',
				'shm_name': self.model_info.get('shm_name'),
				'tensor_meta_shm_name': self.model_info.get('tensor_meta_shm_name'),
				'parameter_server_size': self.model_info.get('parameter_server_size'),
				'huggingface_ckpt_name': self.model_info.get('huggingface_ckpt_name'),
				'converted_ckpt_dir': self.model_info.get('converted_ckpt_dir'),
				'skeleton_state_dict_file': getattr(self, 'skeleton_state_dict_file', None),
				# Keep for backward compatibility, but it's just the file name now
				'skeleton_state_dict_shm_name': self.skeleton_state_dict_shm_name
			}
		
		elif command == 'exit':
			# Request to disconnect this client, not shut down the server
			return {'status': 'success', 'message': 'Disconnecting client'}
		
		else:
			return {'status': 'error', 'message': f'Unknown command: {command}'}
	
	def process_request_dep(self, request: Dict[str, Any]) -> Dict[str, Any]:
		"""Process a client request and return a response"""
		command = request.get('command')
		
		if command == 'ping':
			return {'status': 'success', 'message': 'pong'}
		
		elif command == 'load_model':
			return self.handle_load_model(request)
		
		elif command == 'get_model_info':
			if not self.current_model:
				return {'status': 'error', 'message': 'No model currently loaded'}
			
			# Create a lightweight response with shared memory name and backup file
			return {
				'status': 'success',
				'shm_name': self.model_info.get('shm_name'),
				'tensor_meta_shm_name': self.model_info.get('tensor_meta_shm_name'),
				'parameter_server_size': self.model_info.get('parameter_server_size'),
				'huggingface_ckpt_name': self.model_info.get('huggingface_ckpt_name'),
				'converted_ckpt_dir': self.model_info.get('converted_ckpt_dir'),
				'skeleton_state_dict_shm_name': self.skeleton_state_dict_shm_name,
				'skeleton_state_dict_backup_file': getattr(self, 'skeleton_state_dict_backup_file', None)
			}
		
		elif command == 'exit':
			# Request to disconnect this client, not shut down the server
			return {'status': 'success', 'message': 'Disconnecting client'}
		
		else:
			return {'status': 'error', 'message': f'Unknown command: {command}'}
	
	def handle_load_model(self, request: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle a request to load a model"""
		huggingface_ckpt_name = request.get('huggingface_ckpt_name')
		if not huggingface_ckpt_name:
			return {'status': 'error', 'message': 'Missing model name'}
		
		# Skip loading if the model is already loaded
		if self.current_model == huggingface_ckpt_name:
			logging.info(f"Model {huggingface_ckpt_name} already loaded, reusing")
			# Return a lightweight response without the skeleton_state_dict
			return {
				'status': 'success',
				'shm_name': self.model_info.get('shm_name'),
				'tensor_meta_shm_name': self.model_info.get('tensor_meta_shm_name'),
				'parameter_server_size': self.model_info.get('parameter_server_size'),
				'huggingface_ckpt_name': self.model_info.get('huggingface_ckpt_name'),
				'converted_ckpt_dir': self.model_info.get('converted_ckpt_dir'),
				'skeleton_state_dict_shm_name': self.skeleton_state_dict_shm_name
			}
		
		# Extract additional parameters
		hf_cache_dir = request.get('hf_cache_dir')
		cache_dir = request.get('cache_dir')
		converted_ckpt_dir = request.get('converted_ckpt_dir')
		
		# Handle HF cache dir - exactly as in original implementation
		if hf_cache_dir is None:
			# Use huggingface default dir
			try:
				from huggingface_hub import constants
				hf_cache_dir = constants.HF_HUB_CACHE
			except ImportError:
				hf_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
		
		# Handle cache_dir - exactly as in original implementation
		if cache_dir is None:
			# Check if model download is allowed (disabled by default for production safety)
			allow_model_download = request.get('allow_model_download', False)
			if not allow_model_download:
				error_msg = (
					"Error: Model download is disabled by default for production safety.\n"
					"Please either:\n"
					"  1. Provide cache_dir pointing to pre-downloaded model files, OR\n"
					"  2. Set allow_model_download=True in the request to enable downloading from HuggingFace Hub"
				)
				logging.error(error_msg)
				return {'status': 'error', 'message': error_msg}

			try:
				logging.info("Downloading model from Hugging Face")
				from huggingface_hub import snapshot_download
				model_path = snapshot_download(
					huggingface_ckpt_name,
					cache_dir=hf_cache_dir,
					ignore_patterns=["flax*", "tf*"],
				)
				cache_dir = model_path
			except Exception as e:
				error_msg = f"Error downloading model: {e}"
				logging.error(error_msg)
				return {'status': 'error', 'message': error_msg}
		
		# Handle converted_ckpt_dir - exactly as in original implementation
		if converted_ckpt_dir is None:
			converted_ckpt_dir = os.path.join(
				cache_dir, "converted_ckpt", huggingface_ckpt_name
			)
			if not os.path.exists(converted_ckpt_dir):
				os.makedirs(converted_ckpt_dir)
		else:
			converted_ckpt_dir = os.path.join(converted_ckpt_dir, huggingface_ckpt_name)
			if not os.path.exists(converted_ckpt_dir):
				os.makedirs(converted_ckpt_dir)
		
		logging.info(f"Will dump model parameters to: {converted_ckpt_dir}")
		
		# Create the appropriate parameter server based on model name
		try:
			if "deepseek" in huggingface_ckpt_name:
				from batchgen.models.deepseek.deepseek_parameter_server import DeepSeek_Parameter_Server
				self.parameter_server_instance = DeepSeek_Parameter_Server(
					huggingface_ckpt_name, cache_dir, converted_ckpt_dir, self.enable_hugetlbfs
				)
			elif "Mixtral" in huggingface_ckpt_name:
				from batchgen.models.mixtral.mixtral_parameter_server import Mixtral_Parameter_Server
				self.parameter_server_instance = Mixtral_Parameter_Server(
					huggingface_ckpt_name, cache_dir, converted_ckpt_dir
				)
			else:
				error_msg = f"Model architecture {huggingface_ckpt_name} not supported yet."
				logging.error(error_msg)
				return {'status': 'error', 'message': error_msg}
			
			# Initialize the parameter server and get shared memory names
			shm_name, tensor_meta_shm_name = self.parameter_server_instance.Init()
			
			# Get and store the skeleton state dict and size
			skeleton_state_dict = self.parameter_server_instance.parameter_server.get_skeleton_state_dict()
			parameter_server_size = self.parameter_server_instance.parameter_server.byte_size()
			logging.info(f"Parameter Server Size: {parameter_server_size}")
			
			# Create shared memory for the skeleton state dict
			skeleton_state_dict_shm_name = self.create_skeleton_state_dict_shared_memory(skeleton_state_dict)
			if not skeleton_state_dict_shm_name:
				error_msg = "Failed to create shared memory for skeleton state dict"
				logging.error(error_msg)
				return {'status': 'error', 'message': error_msg}
			
			# Store model info (without skeleton_state_dict to save memory)
			self.model_info = {
				'shm_name': shm_name,
				'tensor_meta_shm_name': tensor_meta_shm_name,
				'parameter_server_size': parameter_server_size,
				'huggingface_ckpt_name': huggingface_ckpt_name,
				'converted_ckpt_dir': converted_ckpt_dir
			}
			
			# Update the currently loaded model
			self.current_model = huggingface_ckpt_name
			
			logging.info(f"Successfully loaded model: {huggingface_ckpt_name}")
			
			# Return success with the model info (including skeleton_state_dict_shm_name)
			return {
				'status': 'success',
				'shm_name': shm_name,
				'tensor_meta_shm_name': tensor_meta_shm_name,
				'parameter_server_size': parameter_server_size,
				'huggingface_ckpt_name': huggingface_ckpt_name,
				'converted_ckpt_dir': converted_ckpt_dir,
				'skeleton_state_dict_shm_name': skeleton_state_dict_shm_name
			}
			
		except Exception as e:
			error_msg = f"Error initializing parameter server: {e}"
			logging.error(error_msg)
			return {'status': 'error', 'message': error_msg}
	
	def _preload_model(self, huggingface_ckpt_name, hf_cache_dir=None, cache_dir=None, converted_ckpt_dir=None):
		"""
		Preload a model at server startup
		
		Args:
			huggingface_ckpt_name: Model name on HuggingFace
			hf_cache_dir: HuggingFace cache directory
			cache_dir: Model cache directory
			converted_ckpt_dir: Directory for PyTorch checkpoints
		"""
		# Create a mock request to reuse the handle_load_model method
		request = {
			'huggingface_ckpt_name': huggingface_ckpt_name,
			'hf_cache_dir': hf_cache_dir,
			'cache_dir': cache_dir,
			'converted_ckpt_dir': converted_ckpt_dir
		}
		
		# Use the existing method to load the model
		result = self.handle_load_model(request)
		
		# if result['status'] != 'success':
		# 	raise RuntimeError(f"Failed to preload model: {result.get('message', 'Unknown error')}")
		
		return result


def parse_args():
	"""Parse command line arguments"""
	parser = argparse.ArgumentParser(description="Standalone Parameter Server for BatchGen")
	parser.add_argument(
		"--host", 
		type=str, 
		default="localhost",
		help="Host to bind the server to"
	)
	parser.add_argument(
		"--port", 
		type=int, 
		default=10900,
		help="Port to listen on"
	)
	parser.add_argument(
		"--model",
		type=str,
		default=None,
		help="HuggingFace model name to preload at server startup"
	)
	parser.add_argument(
		"--hf-cache-dir",
		type=str,
		default=None,
		help="HuggingFace cache directory"
	)
	parser.add_argument(
		"--cache-dir",
		type=str,
		default=None,
		help="Model cache directory"
	)
	parser.add_argument(
		"--pt-ckpt-dir",
		type=str,
		default=None,
		help="Directory for PyTorch checkpoints"
	)
	parser.add_argument(
		"--enable-hugetlbfs",
		action='store_true',
		default=False,
		help="Enable hugetlbfs for shared memory (requires root privileges)"
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	if args.enable_hugetlbfs:
		os.environ["BATCHGEN_ENABLE_HUGETLBFS"] = "1"
	logging.info(f"Starting Parameter Server on {args.host}:{args.port}")
	logging.info(f"Enable hugetlbfs: {os.environ.get('BATCHGEN_ENABLE_HUGETLBFS', '0')}")

	server = ParameterServer(
		host=args.host,
		port=args.port,
		model_name=args.model,
		hf_cache_dir=args.hf_cache_dir,
		cache_dir=args.cache_dir,
		converted_ckpt_dir=args.converted_ckpt_dir,
		enable_hugetlbfs=args.enable_hugetlbfs
	)
	
	try:
		server.start()
	except Exception as e:
		logging.error(f"Fatal error in parameter server: {e}")
		sys.exit(1)