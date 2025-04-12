"""
Standalone Parameter Server for MoE-Gen

This script starts a long-running parameter server that hosts model weights in shared memory.
It uses socket communication to handle requests from client processes.
"""
import os
import sys
import time
import logging
import argparse
import socket
import json
import pickle
import struct
import threading
import signal
from typing import Dict, Any, Optional, Tuple
import torch
from .engine import _config_torch_module_initializer
_config_torch_module_initializer()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class ParameterServer:
    def __init__(self, host='localhost', port=9090, model_name=None,
                 hf_cache_dir=None, cache_dir=None, pt_ckpt_dir=None):
        """
        Initialize the Parameter Server.
        
        Args:
            host: Host to bind the server socket to
            port: Port to listen on
            model_name: HuggingFace model name to load at startup
            hf_cache_dir: HuggingFace cache directory
            cache_dir: Model cache directory
            pt_ckpt_dir: Directory for PyTorch checkpoints
        """
        self.host = host
        self.port = port
        self.server_socket = None
        self.clients = []
        self.running = False
        
        # Initial model parameters
        self.initial_model_name = model_name
        self.hf_cache_dir = hf_cache_dir
        self.cache_dir = cache_dir
        self.pt_ckpt_dir = pt_ckpt_dir
        
        # State tracking
        self.current_model = None
        self.parameter_server_instance = None
        self.model_info = {}
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
    
    def start(self):
        """Start the parameter server"""
        # Preload model if specified - BEFORE starting the server socket
        if self.initial_model_name:
            logging.info(f"Preloading model: {self.initial_model_name}")
            try:
                start_time = time.time()
                self._preload_model(
                    self.initial_model_name,
                    self.hf_cache_dir,
                    self.cache_dir,
                    self.pt_ckpt_dir
                )
                end_time = time.time()
                logging.info(f"Model {self.initial_model_name} loaded successfully in {end_time - start_time:.2f} seconds")
                logging.info(f"Model loaded with shared memory name: {self.model_info.get('shm_name')}")
                logging.info(f"Parameter server size: {self.model_info.get('parameter_server_size')} bytes")
            except Exception as e:
                logging.error(f"Failed to preload model {self.initial_model_name}: {e}")
                logging.error("Server will start without a preloaded model")
                # Continue running the server even if model loading fails
        else:
            logging.info("No initial model specified. Server starting without preloaded model.")
        
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
        
        logging.info("Parameter server shutdown complete")
    
    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals"""
        logging.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def handle_client(self, client_socket, address):
        """Handle communication with a client"""
        try:
            while self.running:
                # Receive message size first (4 bytes for a 32-bit integer)
                size_data = client_socket.recv(4)
                if not size_data:
                    break  # Connection closed
                
                # Unpack the size
                msg_size = struct.unpack('!I', size_data)[0]
                
                # Receive the actual message
                data = b''
                remaining = msg_size
                while remaining > 0:
                    chunk = client_socket.recv(min(4096, remaining))
                    if not chunk:
                        break  # Connection closed
                    data += chunk
                    remaining -= len(chunk)
                
                if len(data) < msg_size:
                    logging.warning(f"Incomplete message received from {address}")
                    break
                
                # Parse the request
                request = json.loads(data.decode('utf-8'))
                logging.info(f"Received request from {address}: {request.get('command', 'unknown')}")
                
                # Process the request
                response = self.process_request(request)
                
                # Send the response size first
                response_data = pickle.dumps(response)
                response_size = len(response_data)
                client_socket.sendall(struct.pack('!I', response_size))
                
                # Send the response
                client_socket.sendall(response_data)
        
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
            return {'status': 'success', **self.model_info}
        
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
            return {'status': 'success', **self.model_info}
        
        # Extract additional parameters
        hf_cache_dir = request.get('hf_cache_dir')
        cache_dir = request.get('cache_dir')
        pt_ckpt_dir = request.get('pt_ckpt_dir')
        
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
        
        # Handle pt_ckpt_dir - exactly as in original implementation
        if pt_ckpt_dir is None:
            pt_ckpt_dir = os.path.join(
                hf_cache_dir, "pt_ckpt", huggingface_ckpt_name
            )
            if not os.path.exists(pt_ckpt_dir):
                os.makedirs(pt_ckpt_dir)
        else:
            pt_ckpt_dir = os.path.join(pt_ckpt_dir, huggingface_ckpt_name)
            if not os.path.exists(pt_ckpt_dir):
                os.makedirs(pt_ckpt_dir)
        
        logging.info(f"Will dump model parameters to: {pt_ckpt_dir}")
        
        # Create the appropriate parameter server based on model name
        try:
            if "deepseek" in huggingface_ckpt_name:
                from moe_gen.models.deepseek.deepseek_parameter_server import DeepSeek_Parameter_Server
                self.parameter_server_instance = DeepSeek_Parameter_Server(
                    huggingface_ckpt_name, cache_dir, pt_ckpt_dir
                )
            elif "Mixtral" in huggingface_ckpt_name:
                from moe_gen.models.mixtral.mixtral_parameter_server import Mixtral_Parameter_Server
                self.parameter_server_instance = Mixtral_Parameter_Server(
                    huggingface_ckpt_name, cache_dir, pt_ckpt_dir
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
            
            # Store model info
            self.model_info = {
                'shm_name': shm_name,
                'tensor_meta_shm_name': tensor_meta_shm_name,
                'parameter_server_size': parameter_server_size,
                'huggingface_ckpt_name': huggingface_ckpt_name,
                'pt_ckpt_dir': pt_ckpt_dir
            }
            
            # We need to handle the skeleton_state_dict separately as it might not be
            # JSON serializable. Store it directly in the model_info but not in the response.
            self.model_info['skeleton_state_dict'] = skeleton_state_dict
            
            # Update the currently loaded model
            self.current_model = huggingface_ckpt_name
            
            logging.info(f"Successfully loaded model: {huggingface_ckpt_name}")
            
            # Return success with the model info
            # Note: skeleton_state_dict will be pickled in the response
            return {'status': 'success', **self.model_info}
            
        except Exception as e:
            error_msg = f"Error initializing parameter server: {e}"
            logging.error(error_msg)
            return {'status': 'error', 'message': error_msg}
    
    def _preload_model(self, huggingface_ckpt_name, hf_cache_dir=None, cache_dir=None, pt_ckpt_dir=None):
        """
        Preload a model at server startup
        
        Args:
            huggingface_ckpt_name: Model name on HuggingFace
            hf_cache_dir: HuggingFace cache directory
            cache_dir: Model cache directory
            pt_ckpt_dir: Directory for PyTorch checkpoints
        """
        # Create a mock request to reuse the handle_load_model method
        request = {
            'huggingface_ckpt_name': huggingface_ckpt_name,
            'hf_cache_dir': hf_cache_dir,
            'cache_dir': cache_dir,
            'pt_ckpt_dir': pt_ckpt_dir
        }
        
        # Use the existing method to load the model
        result = self.handle_load_model(request)
        
        if result['status'] != 'success':
            raise RuntimeError(f"Failed to preload model: {result.get('message', 'Unknown error')}")
        
        return result


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Standalone Parameter Server for MoE-Gen")
    parser.add_argument(
        "--host", 
        type=str, 
        default="localhost",
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=9090,
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
        help="Directory for storing checkpoint in .pt format"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    server = ParameterServer(
        host=args.host,
        port=args.port,
        model_name=args.model,
        hf_cache_dir=args.hf_cache_dir,
        cache_dir=args.cache_dir,
        pt_ckpt_dir=args.pt_ckpt_dir
    )
    
    try:
        server.start()
    except Exception as e:
        logging.error(f"Fatal error in parameter server: {e}")
        sys.exit(1)