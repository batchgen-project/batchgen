"""
Client library for communicating with the MoE-Gen Parameter Server.
This module provides the client-side interface to the standalone parameter server.
"""
import os
import socket
import json
import pickle
import struct
import logging
import time
from typing import Dict, Any, Optional, List, Union
from multiprocessing import shared_memory
import numpy as np
import torch

class ParameterServerClient:
    def __init__(self, host='localhost', port=9090, timeout=60):
        """
        Initialize a client connection to the parameter server.
        
        Args:
            host: The parameter server host
            port: The parameter server port
            timeout: Socket timeout in seconds
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket = None
    
    def connect(self):
        """Connect to the parameter server"""
        if self.socket is not None:
            return
        
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(self.timeout)
        try:
            self.socket.connect((self.host, self.port))
        except Exception as e:
            self.socket = None
            raise ConnectionError(f"Failed to connect to parameter server at {self.host}:{self.port}: {e}")
    
    def __del__(self):
        """Clean up when the client is deleted"""
        self.disconnect()
        
    def disconnect(self):
        """
        Disconnect from the parameter server without unlinking shared memory
        """
        if self.socket is not None:
            try:
                # Send an exit request
                self.send_request({'command': 'exit'})
            except:
                pass  # Ignore errors during exit
            
            try:
                self.socket.close()
            except:
                pass
            finally:
                self.socket = None
    
    def send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a request to the parameter server and get the response.
        
        Args:
            request: The request to send
            
        Returns:
            The server's response
        """
        if self.socket is None:
            self.connect()
        
        # Convert request to JSON and encode
        request_data = json.dumps(request).encode('utf-8')
        request_size = len(request_data)
        
        if request_size > 100 * 1024 * 1024:  # 100MB
            raise ValueError(f"Request size too large: {request_size} bytes")
        
        try:
            # Send size first (as unsigned 32-bit integer), then data
            self.socket.sendall(struct.pack('!I', request_size))
            
            # Send request data
            CHUNK_SIZE = 8192
            for i in range(0, request_size, CHUNK_SIZE):
                end = min(i + CHUNK_SIZE, request_size)
                self.socket.sendall(request_data[i:end])
            
            # Receive response size (as unsigned 32-bit integer)
            size_data = self.socket.recv(4)
            if not size_data or len(size_data) < 4:
                raise ConnectionError("Connection closed by server before receiving response size")
            
            response_size = struct.unpack('!I', size_data)[0]
            
            if response_size > 500 * 1024 * 1024:  # 500MB safety limit
                raise ValueError(f"Response size too large: {response_size} bytes")
            
            # Receive the response data in chunks
            response_data = bytearray(response_size)
            bytes_received = 0
            
            while bytes_received < response_size:
                chunk = self.socket.recv(min(8192, response_size - bytes_received))
                if not chunk:
                    raise ConnectionError(f"Connection closed by server during data transfer after receiving {bytes_received}/{response_size} bytes")
                
                # Copy chunk into the correct position in the response data buffer
                response_data[bytes_received:bytes_received+len(chunk)] = chunk
                bytes_received += len(chunk)
            
            # Unpickle the response
            try:
                response = pickle.loads(response_data)
                return response
            except Exception as e:
                raise ConnectionError(f"Error unpickling response: {e}")
            
        except socket.timeout:
            self.socket = None  # Mark as disconnected
            raise ConnectionError("Timeout while communicating with parameter server")
        except socket.error as e:
            self.socket = None  # Mark as disconnected
            raise ConnectionError(f"Socket error communicating with parameter server: {e}")
        except Exception as e:
            self.socket = None  # Mark as disconnected
            raise ConnectionError(f"Unexpected error: {e}")
    def ping(self) -> bool:
        """
        Ping the parameter server to check connection
        
        Returns:
            True if the server is reachable and responding
        """
        try:
            response = self.send_request({'command': 'ping'})
            return response.get('status') == 'success'
        except:
            return False
    
    def load_model(self, 
                 huggingface_ckpt_name: str,
                 hf_cache_dir: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 pt_ckpt_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Request the parameter server to load a model
        
        Args:
            huggingface_ckpt_name: Model name on HuggingFace
            hf_cache_dir: HuggingFace cache directory
            cache_dir: Model cache directory
            pt_ckpt_dir: Directory for PyTorch checkpoints
            
        Returns:
            Dict containing model information including shared memory names
        """
        request = {
            'command': 'load_model',
            'huggingface_ckpt_name': huggingface_ckpt_name,
        }
        
        if hf_cache_dir is not None:
            request['hf_cache_dir'] = hf_cache_dir
        if cache_dir is not None:
            request['cache_dir'] = cache_dir
        if pt_ckpt_dir is not None:
            request['pt_ckpt_dir'] = pt_ckpt_dir
        
        response = self.send_request(request)
        
        if response.get('status') != 'success':
            raise RuntimeError(f"Failed to load model: {response.get('message', 'Unknown error')}")
        
        return response

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently loaded model
        with support for PyTorch-serialized state dictionaries
        
        Returns:
            Dict containing model information
        """
        response = self.send_request({'command': 'get_model_info'})
        
        if response.get('status') != 'success':
            raise RuntimeError(f"Failed to get model info: {response.get('message', 'Unknown error')}")
        
        # Check if we need to get the skeleton state dict from file
        skeleton_state_dict_file = response.get('skeleton_state_dict_file')
        
        if skeleton_state_dict_file and os.path.exists(skeleton_state_dict_file):
            try:
                logging.info(f"Loading skeleton state dict from file: {skeleton_state_dict_file}")
                
                # Get the file size for progress reporting
                file_size = os.path.getsize(skeleton_state_dict_file)
                logging.info(f"File size: {file_size} bytes")
                
                # Load the file using torch.load
                import torch
                logging.info("Loading state dict using torch.load...")
                skeleton_state_dict = torch.load(skeleton_state_dict_file)
                logging.info(f"Successfully loaded skeleton state dict with {len(skeleton_state_dict)} keys")
                
                # Add to response
                response['skeleton_state_dict'] = skeleton_state_dict
                
            except Exception as e:
                logging.error(f"Error loading skeleton state dict from file: {e}")
                raise RuntimeError(f"Failed to load skeleton state dict: {e}")
        
        else:
            # Fall back to shared memory method if file not found (backwards compatibility)
            skeleton_state_dict_shm_name = response.get('skeleton_state_dict_shm_name')
            if skeleton_state_dict_shm_name and skeleton_state_dict_shm_name.endswith('.bin'):
                # This is actually a file name, not shared memory
                # Try looking in the backup directory
                backup_dir = os.path.join(os.getcwd(), "shared_memory_backup")
                file_path = os.path.join(backup_dir, skeleton_state_dict_shm_name)
                
                if os.path.exists(file_path):
                    try:
                        logging.info(f"Loading skeleton state dict from backup file: {file_path}")
                        
                        # Similar code as above to read the file
                        with open(file_path, 'rb') as f:
                            size_bytes = f.read(8)
                            serialized_size = struct.unpack('!Q', size_bytes)[0]
                            
                            data = bytearray(serialized_size)
                            chunk_size = 100 * 1024 * 1024
                            for i in range(0, serialized_size, chunk_size):
                                end = min(i + chunk_size, serialized_size)
                                chunk = f.read(end - i)
                                data[i:i+len(chunk)] = chunk
                        
                        skeleton_state_dict = pickle.loads(bytes(data))
                        response['skeleton_state_dict'] = skeleton_state_dict
                        
                    except Exception as e:
                        logging.error(f"Error loading from backup file: {e}")
                        raise RuntimeError(f"Failed to load skeleton state dict: {e}")
                else:
                    raise RuntimeError(f"Skeleton state dict file not found: {file_path}")
        
        return response

    # def get_model_info(self) -> Dict[str, Any]:
    #     """
    #     Get information about the currently loaded model with fallback to backup file
        
    #     Returns:
    #         Dict containing model information
    #     """
    #     response = self.send_request({'command': 'get_model_info'})
        
    #     if response.get('status') != 'success':
    #         raise RuntimeError(f"Failed to get model info: {response.get('message', 'Unknown error')}")
        
    #     # Check if we need to get the skeleton state dict 
    #     skeleton_state_dict_shm_name = response.get('skeleton_state_dict_shm_name')
    #     backup_file = response.get('skeleton_state_dict_backup_file')
        
    #     if skeleton_state_dict_shm_name or backup_file:
    #         skeleton_state_dict = None
    #         shared_mem_success = False
            
    #         # First try: shared memory (fastest)
    #         if skeleton_state_dict_shm_name:
    #             logging.info(f"Attempting to access skeleton state dict from shared memory: {skeleton_state_dict_shm_name}")
                
    #             shared_mem = None
    #             try:
    #                 # Try to access shared memory
    #                 shared_mem = shared_memory.SharedMemory(name=skeleton_state_dict_shm_name)
    #                 logging.info("Successfully accessed shared memory")
                    
    #                 # Get the size of the shared memory segment
    #                 buffer_size = shared_mem.size
                    
    #                 # Read the size marker at the end (last 4 bytes)
    #                 size_bytes = bytes(shared_mem.buf[buffer_size - 4:buffer_size])
                    
    #                 try:
    #                     data_size = struct.unpack('!I', size_bytes)[0]
    #                     logging.info(f"Found data size from marker: {data_size} bytes")
                        
    #                     if data_size <= 0 or data_size > buffer_size - 4:
    #                         logging.warning(f"Invalid data size from marker: {data_size}, buffer size: {buffer_size}")
    #                         data_size = buffer_size - 8  # Conservative estimate
    #                 except struct.error:
    #                     logging.warning("Could not unpack size marker, using conservative estimate")
    #                     data_size = buffer_size - 8  # Conservative estimate if marker is corrupted
                    
    #                 # Read the data efficiently in chunks
    #                 logging.info(f"Reading {data_size} bytes from shared memory")
                    
    #                 # Create a copy of the data to break the reference to shared memory
    #                 serialized_data = bytearray(data_size)
                    
    #                 # Use reasonably sized chunks (10MB)
    #                 chunk_size = 10 * 1024 * 1024
    #                 for i in range(0, data_size, chunk_size):
    #                     end_pos = min(i + chunk_size, data_size)
    #                     # Make an explicit copy of the chunk
    #                     chunk = bytes(shared_mem.buf[i:end_pos])  # Create a new bytes object
    #                     serialized_data[i:end_pos] = chunk  # Copy into our buffer
                    
    #                 # Close the shared memory BEFORE unpickling
    #                 if shared_mem is not None:
    #                     shared_mem.close()  # Close our access
    #                     shared_mem = None
                    
    #                 # Force a garbage collection to clean up any remaining references
    #                 import gc
    #                 gc.collect()
                    
    #                 # Try to unpickle the data from shared memory
    #                 try:
    #                     logging.info("Unpickling data from shared memory...")
    #                     skeleton_state_dict = pickle.loads(bytes(serialized_data))
    #                     logging.info(f"Successfully loaded skeleton state dict with {len(skeleton_state_dict)} keys")
    #                     shared_mem_success = True
    #                 except Exception as e:
    #                     logging.error(f"Error unpickling shared memory data: {e}")
    #                     # We'll try the backup file next
                
    #             except FileNotFoundError:
    #                 logging.error(f"Shared memory segment not found: {skeleton_state_dict_shm_name}")
    #                 # We'll try the backup file next
                
    #             except Exception as e:
    #                 logging.error(f"Error accessing shared memory: {e}")
    #                 # Always ensure we clean up
    #                 # if shared_mem is not None:
    #                 #     try:
    #                 #         shared_mem.close()
    #                 #     except Exception:
    #                 #         pass
    #                 #     shared_mem = None
    #                 # We'll try the backup file next
            
    #         # Second try: backup file (slower but more reliable)
    #         if not shared_mem_success and backup_file:
    #             try:
    #                 logging.info(f"Attempting to read skeleton state dict from backup file: {backup_file}")
                    
    #                 if os.path.exists(backup_file):
    #                     with open(backup_file, 'rb') as f:
    #                         # Read the size from the first 4 bytes
    #                         size_bytes = f.read(4)
    #                         data_size = struct.unpack('!I', size_bytes)[0]
    #                         logging.info(f"Found data size from file header: {data_size} bytes")
                            
    #                         # Read the serialized data
    #                         serialized_data = f.read(data_size)
                            
    #                         if len(serialized_data) == data_size:
    #                             # Try to unpickle
    #                             logging.info("Unpickling data from backup file...")
    #                             skeleton_state_dict = pickle.loads(serialized_data)
    #                             logging.info(f"Successfully loaded skeleton state dict from backup file with {len(skeleton_state_dict)} keys")
    #                         else:
    #                             logging.error(f"Read incomplete data: got {len(serialized_data)}, expected {data_size} bytes")
    #                 else:
    #                     logging.error(f"Backup file not found: {backup_file}")
                
    #             except Exception as e:
    #                 logging.error(f"Error reading from backup file: {e}")
            
    #         # If we got the skeleton state dict through either method, add it to the response
    #         if skeleton_state_dict is not None:
    #             response['skeleton_state_dict'] = skeleton_state_dict
    #             if shared_mem_success:
    #                 logging.info("Using skeleton state dict from shared memory")
    #             else:
    #                 logging.info("Using skeleton state dict from backup file")
    #         else:
    #             raise RuntimeError("Failed to get skeleton state dict from either shared memory or backup file")
        
    #     return response

    def __enter__(self):
        """Support for 'with' statement"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up when exiting 'with' block"""
        self.disconnect()