# ---------------------------------------------------------------------------- #
#  MoE-Gen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
#                                                                               #
#  you may obtain a copy of the license at                                      #
#                                                                               #
#                  http://www.apache.org/licenses/license-2.0                   #
#                                                                               #
#  unless required by applicable law or agreed to in writing, software          #
#  distributed under the license is distributed on an "as is" basis,            #
#  without warranties or conditions of any kind, either express or implied.     #
#  see the license for the specific language governing permissions and          #
#  limitations under the license.                                               #
# ---------------------------------------------------------------------------- #

import concurrent.futures
import copy
import functools
import logging
import math
import os
import sys
import time
from typing import Callable, Dict, List, Optional

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

# import nvidia_dlprof_pytorch_nvtx as nvtx
from moe_gen.models.Wrapper import Attn_Wrapper, Expert_Wrapper

from .config.config import EngineConfig
from .models.deepseek.deepseek_parameter_server import DeepSeek_Parameter_Server
from .scheduler.host_mem import get_physical_memory_info

from moe_gen.parameter_server_client import ParameterServerClient
from .models.deepseek.deepseekv3.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from tqdm import trange

logging.basicConfig(
    level=logging.INFO,  # Set to the lowest level to capture all messages
    format="%(asctime)s - %(levelname)s - %(message)s",  # Include timestamp
    datefmt="%Y-%m-%d %H:%M:%S",  # Customize timestamp format
)

from .config.engine_config_parser import parse_config_from_json
# nvtx = False
# if nvtx:
# 	nvidia_dlprof_pytorch_nvtx.init()

import signal
import traceback
import sys
import faulthandler

def signal_handler(signum, frame):
    print(f"\n{'='*50}")
    print(f"Process {os.getpid()} (Rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 'N/A'}) received signal {signum}")
    print(f"{'='*50}")
    
    # Print current stack trace
    traceback.print_stack(frame)
    
    # For all threads
    import threading
    for thread_id, frame in sys._current_frames().items():
        print(f"\nThread {thread_id} stack:")
        traceback.print_stack(frame)
    
    # Force flush output
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Exit after some delay to allow other processes to print
    import time
    time.sleep(0.5)
    sys.exit(1)



class query:
    def __init__(
        self,
        text: str = None,
        encoded: Dict[str, torch.Tensor] = None,
        decoded_tokens: torch.Tensor = None,
    ):
        self.text = text
        self.encoded = encoded
        self.decoded_tokens = decoded_tokens


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


def _config_torch_module_initializer():
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



def run_moe_gen(moe_gen):
    moe_gen.Init()
    return moe_gen.generate()


def distribute_sequences(num_sequences, num_devices):
    """
    Distributes sequences across devices ensuring each device gets at least one sequence when possible,
    and the distribution is as even as possible.
    
    Args:
        num_sequences: Number of sequences to distribute
        num_devices: Number of available devices
    
    Returns:
        List of (start_idx, end_idx) tuples for each device
    """
    # If we have fewer sequences than devices, only use as many devices as we have sequences
    active_devices = min(num_sequences, num_devices)
    
    # Calculate base sequences per device and remainder
    base_per_device = num_sequences // active_devices
    remainder = num_sequences % active_devices
    
    distribution = []
    current_idx = 0
    
    for device_idx in range(num_devices):
        if device_idx < active_devices:
            # This device gets work
            # Add one extra sequence for the first 'remainder' devices
            device_sequences = base_per_device + (1 if device_idx < remainder else 0)
            
            start_idx = current_idx
            end_idx = start_idx + device_sequences
            current_idx = end_idx
            
            distribution.append((start_idx, end_idx))
        else:
            # This device gets no work
            distribution.append((0, 0))  # Empty range
    
    return distribution

# Entry point
def moe_gen(
    huggingface_ckpt_name: str,
    queries: List[str],
    max_input_length: int,
    max_decoding_length: int,
    device: List[int],
    engine_config_json_dir: str,
    hf_cache_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    pt_ckpt_dir: Optional[str] = None,
    host_kv_cache_size: Optional[int] = None,  # If not set, use all host memory
    parameter_server_host: str = 'localhost',
    parameter_server_port: int = 9090,
):
    """
    Run MoE_Gen using the standalone parameter server.
    
    Args:
        huggingface_ckpt_name: Model name on HuggingFace
        queries: List of queries to process
        max_input_length: Maximum input length
        max_decoding_length: Maximum decoding length
        device: List of GPU device IDs to use
        engine_config: Engine configuration
        hf_cache_dir: HuggingFace cache directory
        cache_dir: Model cache directory
        pt_ckpt_dir: Directory for PyTorch checkpoints
        host_kv_cache_size: Host KV cache size in bytes
        parameter_server_host: Host of the parameter server
        parameter_server_port: Port of the parameter server
    
    Returns:
        List of results with generated text
    """
    # Setup multiprocessing
    mp.set_start_method("spawn", force=True)
    mp.set_sharing_strategy("file_system")
    # from moe_gen.utils import _config_torch_module_initializer
    _config_torch_module_initializer()
    # Register the handler
    signal.signal(signal.SIGINT, signal_handler)  # For Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # For kill command

    # Enable faulthandler to get stack traces on segfault
    faulthandler.enable()
    
    # Get model info from the parameter server - just retrieve existing info
    logging.info(f"Connecting to parameter server at {parameter_server_host}:{parameter_server_port}")
    try:
        with ParameterServerClient(host=parameter_server_host, port=parameter_server_port) as client:
            # First check if the server has a model loaded
            model_info = client.get_model_info()
            
            # If the expected model isn't loaded, we need to request it
            if model_info.get('huggingface_ckpt_name') != huggingface_ckpt_name:
                logging.info(f"Parameter server has a different model loaded. Requesting {huggingface_ckpt_name}")
                # model_info = client.load_model(
                #     huggingface_ckpt_name=huggingface_ckpt_name,
                #     hf_cache_dir=hf_cache_dir,
                #     cache_dir=cache_dir,
                #     pt_ckpt_dir=pt_ckpt_dir
                # )

                # Terminate directly
                logging.info(f"Terminating process as the model is not loaded in the parameter server.")
                sys.exit(1)
            else:
                logging.info(f"Model {huggingface_ckpt_name} already loaded in parameter server")
    except Exception as e:
        raise RuntimeError(f"Failed to connect to parameter server: {e}")
    
    # Extract necessary data from model_info
    shm_name = model_info.get('shm_name')
    tensor_meta_shm_name = model_info.get('tensor_meta_shm_name')
    skeleton_state_dict = model_info.get('skeleton_state_dict')  # This now comes from shared memory
    parameter_server_size = model_info.get('parameter_server_size')
    pt_ckpt_dir = model_info.get('pt_ckpt_dir')
    
    if not all([shm_name, tensor_meta_shm_name, skeleton_state_dict, parameter_server_size, pt_ckpt_dir]):
        missing = []
        if not shm_name: missing.append('shm_name')
        if not tensor_meta_shm_name: missing.append('tensor_meta_shm_name')
        if not skeleton_state_dict: missing.append('skeleton_state_dict')
        if not parameter_server_size: missing.append('parameter_server_size')
        if not pt_ckpt_dir: missing.append('pt_ckpt_dir')
        raise RuntimeError(f"Missing required information from parameter server: {', '.join(missing)}")
    
    # Calculate host KV cache size if not provided
    if host_kv_cache_size is None:
        from moe_gen.utils import get_physical_memory_info
        mem_info = get_physical_memory_info()
        free_mem = mem_info["actually_free"]
        # We don't need to subtract parameter_server_size since it's in a separate process
        host_kv_cache_size = math.floor(free_mem)
        logging.info(f"Host KV Cache Size: {host_kv_cache_size}")
    
    # Distribute queries among devices
    moe_gens = []
    num_queries = len(queries)
    num_devices = len(device)
    logging.info(f"Number of devices: {num_devices}")
    logging.info(f"Total number of queries: {num_queries}")
    # queries_per_device = math.ceil(num_queries / num_devices)
    distribution = distribute_sequences(num_queries, num_devices)
    per_device_host_kv_cache_size = host_kv_cache_size // num_devices
    
    # For each device, create a MoE_Gen instance
    for device_idx in range(num_devices):
        # start_query_idx = device_idx * queries_per_device
        # end_query_idx = min((device_idx + 1) * queries_per_device, num_queries)
        start_query_idx, end_query_idx = distribution[device_idx]
        logging.info(f"Device {device_idx}: Processing queries from {start_query_idx} to {end_query_idx}")
                
        # Create MoE_Gen instance with the shared memory info
        from moe_gen.engine import MoE_Gen
        moe_gen_instance = MoE_Gen(
            huggingface_ckpt_name=huggingface_ckpt_name,
            hf_cache_dir=hf_cache_dir,
            cache_dir=cache_dir,
            queries=queries[start_query_idx:end_query_idx],
            max_input_length=max_input_length,
            max_decoding_length=max_decoding_length,
            device=device[device_idx],
            engine_config_json_dir=engine_config_json_dir,
            skeleton_state_dict=skeleton_state_dict,
            shm_name=shm_name,
            tensor_meta_shm_name=tensor_meta_shm_name,
            pt_ckpt_dir=pt_ckpt_dir,
            host_kv_cache_size=per_device_host_kv_cache_size,
            rank = device_idx,
            world_size = num_devices
        )
        moe_gens.append(moe_gen_instance)
    
    logging.info(f"Number of MoE_Gen instances: {len(moe_gens)}")
    
    # Run inference with worker processes
    
    def safe_collect_results(futures):
        all_results = []
        for future in futures:
            try:
                result = future.result()
                all_results.extend(result)
            except torch.distributed.DistBackendError as e:
                # Check if it's an out of memory error during cleanup
                if "out of memory" in str(e) and "destroy_process_group" in str(e):
                    print("Ignoring CUDA out of memory error during process cleanup")
                    # Try to extract results if they were generated before the error
                    # This assumes your function might have returned partial results
                else:
                    # Re-raise if it's a different type of DistBackendError
                    raise
            except Exception as e:
                # Handle other exceptions according to your needs
                print(f"Error in worker process: {e}")
                raise
        
        return all_results

    start_time = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit each worker and collect the future objects
        futures = [
            executor.submit(run_moe_gen, moe_gen_instance) for moe_gen_instance in moe_gens
        ]
        # Retrieve results in the same order as the futures were submitted
        all_results = safe_collect_results(futures)
    
    end_time = time.perf_counter()
    logging.info(f"Inference complete. Total time: {end_time - start_time:.2f}s")
    
    return all_results



class MoE_Gen:
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
        engine_config_json_dir,
        host_kv_cache_size: Optional[int] = None,
        rank: Optional[int] = 0,
        world_size: Optional[int] = 1,
    ):
        self.model = None
        self.hf_cache_dir = hf_cache_dir
        self.huggingface_ckpt_name = huggingface_ckpt_name
        self.cache_dir = cache_dir
        self.pt_ckpt_dir = pt_ckpt_dir
        self.queries = queries
        self.num_queries = len(queries)
        self.max_input_length = max_input_length
        self.max_decoding_length = max_decoding_length
        self.engine_config = parse_config_from_json(engine_config_json_dir)
        self.skeleton_state_dict = skeleton_state_dict
        self.rank = rank
        self.world_size = world_size

        self.engine_config.Basic_Config.device = device
        self.engine_config.Basic_Config.device_torch = torch.device(
            f"cuda:{device}"
        )
        self.engine_config.Basic_Config.max_decoding_length = (
            max_decoding_length
        )
        self.engine_config.Basic_Config.padding_length = max_input_length
        self.engine_config.Basic_Config.num_queries = self.num_queries
        self.engine_config.Basic_Config.rank = rank
        self.engine_config.Basic_Config.world_size = world_size
        
        
        
        
        self.device = device
        self.torch_device = torch.device(f"cuda:{device}")
        self.host_kv_cache_size = host_kv_cache_size

        self.attn_mode = None
        self.query_book = None
        self.model_batch_book = {}
        # TODO:
        self.token_k_cache_byte_size = 2048  # mixtral
        self.num_k_storage_tokens = math.floor(
            50 * (1024**3) / 32 / 2048
        )  # 50G k cache, 50G v cache. 192G test-bed.

        self.shm_name = shm_name
        self.tensor_meta_shm_name = tensor_meta_shm_name



    def Init(self):
        logging.info("Running with PID: %d", os.getpid())
        logging.info("Running with device: %s", self.device)
        torch.cuda.set_device(self.device)
        torch.cuda.reset_peak_memory_stats()
        logging.info(self.hf_cache_dir)
        self.model_config = AutoConfig.from_pretrained(
            self.hf_cache_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        self._config_torch_module_initializer()
        self.tokenizer = AutoTokenizer.from_pretrained(
            # self.huggingface_ckpt_name,
            self.hf_cache_dir,
            # cache_dir=self.hf_cache_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        # Use flash_attn by default thus right padding.
        self.tokenizer.padding_side = "right"


        if self.model_config.architectures[0] == "MixtralForCausalLM":
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.model_config.architectures[0] == "MixtralForCausalLM":
            from moe_gen.models.mixtral.Mixtral_Initializer import (
                Mixtral_Initializer,
            )

            self.initializer = Mixtral_Initializer(
                self.huggingface_ckpt_name,
                self.hf_cache_dir,
                self.cache_dir,
                self.engine_config,
                self.skeleton_state_dict,
                self.shm_name,
                self.tensor_meta_shm_name,
                self.pt_ckpt_dir,
                self.host_kv_cache_size,
            )
        elif self.model_config.architectures[0] == "Qwen2MoeForCausalLM":
            from moe_gen.models.Qwen_Initializer import Qwen_Initializer

            self.initializer = Qwen_Initializer(
                self.huggingface_ckpt_name,
                self.hf_cache_dir,
                self.cache_dir,
                self.engine_config,
                self.pt_ckpt_dir,
            )
        elif self.model_config.architectures[0] == "DeepseekV3ForCausalLM":
            from moe_gen.models.deepseek.deepseekv3.deepseekv3_initializer import (
                DeepSeekV3_Initializer,
            )

            self.initializer = DeepSeekV3_Initializer(
                self.huggingface_ckpt_name,
                self.hf_cache_dir,
                self.cache_dir,
                self.engine_config,
                self.skeleton_state_dict,
                self.shm_name,
                self.tensor_meta_shm_name,
                self.pt_ckpt_dir,
                self.host_kv_cache_size,
                self.rank,
                self.world_size
            )

        elif self.model_config.architectures[0] == "DeepseekV2ForCausalLM":
            from moe_gen.models.deepseek.deepseekv2.deepseekv2_initializer import (
                DeepSeek_Initializer,
            )

            self.initializer = DeepSeek_Initializer(
                self.huggingface_ckpt_name,
                self.hf_cache_dir,
                self.cache_dir,
                self.engine_config,
                self.skeleton_state_dict,
                self.shm_name,
                self.tensor_meta_shm_name,
                self.pt_ckpt_dir,
                self.host_kv_cache_size,
            )

        else:
            raise ValueError(
                f"Model architecture {self.model_config.architectures[0]} not supported yet."
            )

        self.core_engine, self.model, self.engine_config, self.model_config, self.hf_model_config = (
            self.initializer.Init()
        )
        self.vanilla_batching()
        
        
        #TODO:
        from .models.deepseek.deepseekv3.Parallel_Strategy_Manager import(  
            Parallel_Strategy_Manager,
        )
        self.parallel_manager = Parallel_Strategy_Manager(
            self.hf_model_config,
            self.engine_config,
            self.model_config,
            self.core_engine,
            self.skeleton_state_dict
        )        
        
        
        
        logging.info(f"Engine on device {self.device} initialized.")

    def _config_torch_module_initializer(self):
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

    def vanilla_batching(self):
        """
        For the input dataset, batch it to fill the host memory.
        """
        # Step 0: Create mapping from query idx to query.
        self.query_book = {
            query_idx: query(
                text=text,
                decoded_tokens=torch.zeros(
                    1, self.max_decoding_length, dtype=torch.int64
                ),
            )
            for query_idx, text in enumerate(self.queries)
        }
        # Step 1: Tokenize full dataset and pad to mad_input_length.
        for query_idx, query_instance in self.query_book.items():
            tokenized_query = self.tokenizer(
                query_instance.text,
                return_tensors="pt",
                max_length=self.max_input_length,
                truncation=True,
                padding="max_length",
            )
            query_instance.encoded = tokenized_query
            extended_size = self.max_input_length + self.max_decoding_length
            input_ids_extended = torch.zeros(
                (1, extended_size), dtype=tokenized_query["input_ids"].dtype
            )
            attention_mask_extended = torch.zeros(
                (1, extended_size),
                dtype=tokenized_query["attention_mask"].dtype,
            )

            seq_len = tokenized_query["input_ids"].size(1)
            input_ids_extended[0, :seq_len] = tokenized_query["input_ids"][0, :]
            attention_mask_extended[0, :seq_len] = tokenized_query[
                "attention_mask"
            ][0, :]

            tokenized_query["input_ids"] = input_ids_extended
            tokenized_query["attention_mask"] = attention_mask_extended
            query_instance.encoded = tokenized_query

        # Step 2: Create model batches. Batch size = self.engine_config.KV_Storage_Config.num_host_slots
        self.model_batches = []
        num_model_batch = math.ceil(
            self.num_queries
            / self.engine_config.KV_Storage_Config.num_host_slots
        )
        for model_batch_idx in range(num_model_batch):
            self.model_batches.append(
                list(
                    range(
                        model_batch_idx
                        * self.engine_config.KV_Storage_Config.num_host_slots,
                        min(
                            (model_batch_idx + 1)
                            * self.engine_config.KV_Storage_Config.num_host_slots,
                            self.num_queries,
                        ),
                    )
                )
            )

        logging.info(
            f"Number of model level batches: {len(self.model_batches)}"
        )
        logging.info(
            f"Model level batch size: {self.engine_config.KV_Storage_Config.num_host_slots}"
        )

    def initial_batching(self):
        """
        For the input dataset, batch it to fill the host memory.
        """
        # Step 0: Create mapping from query idx to query.
        self.query_book = {
            query_idx: query(
                text=text,
                decoded_tokens=torch.zeros(
                    1, self.max_decoding_length, dtype=torch.int64
                ),
            )
            for query_idx, text in enumerate(self.queries)
        }

        # Step 1: Tokenize full dataset.
        tokenized_length = []
        for query_idx, query_instance in self.query_book.items():
            tokenized_query = self.tokenizer(
                query_instance.text,
                return_tensors="pt",
                max_length=self.max_input_length,
                truncation=True,
                padding=False,
            )
            query_instance.encoded = tokenized_query
            tokenized_length.append(
                (query_idx, tokenized_query["input_ids"].shape[1])
            )

        # Step 2: Sort the tokenized queries by length
        tokenized_length = sorted(
            tokenized_length, key=lambda x: x[1], reverse=True
        )

        # Step 3: Create batches based on memory constraints
        self.model_batches = []
        current_query_num = 0
        batch_idx = 0
        while True:
            current_batch = []
            current_batch_padding_length = (
                tokenized_length[0][1] + self.max_decoding_length
            )
            num_sequences = math.floor(
                self.num_k_storage_tokens / current_batch_padding_length
            )
            if current_query_num + num_sequences > self.num_queries:
                current_batch = tokenized_length[current_query_num:]
            else:
                current_batch = tokenized_length[
                    current_query_num : current_query_num + num_sequences
                ]
            self.model_batches.append(current_batch)
            self.model_batch_book[batch_idx] = {
                "input_length": current_batch_padding_length,
                "num_new_tokens": 0,
            }
            current_query_num += num_sequences
            batch_idx += 1
            if current_query_num >= self.num_queries:
                break

        # Step 4: Complete query instances for each sequences by padding to the same length.
        for batch in self.model_batches:
            max_length = batch[0][1]
            for query_idx, _ in batch:
                self.query_book[query_idx].encoded = self.tokenizer.pad(
                    self.query_book[query_idx].encoded,
                    max_length=max_length,
                    padding="max_length",
                )

        # Step 5: clearn model_batches as list of query idx.
        self.model_batches = [
            [query_idx for query_idx, _ in batch]
            for batch in self.model_batches
        ]
        logging.debug("Initial batching done.")

    def generate(self):
        prefill_time = 0
        decoding_time = 0
        if len(self.model_batches) > 0:
            for model_batch_idx in tqdm(
                range(len(self.model_batches)), desc="Model Batch"
            ):
                self._config_prefill()
                prefill_start_time = time.perf_counter()
                with torch.no_grad():
                    new_token = self.prefill(self.model_batches[model_batch_idx])
                prefill_time += time.perf_counter() - prefill_start_time
                # self.core_engine.prefill_complete_sync()
                logging.info(f"Rank: {self.rank} prefill complete.")

                # Random create new token.
                # new_token = torch.randint(
                #     0,
                #     1000,
                #     # 129280, # self.model_config.vocab_size,
                #     (len(self.model_batches[model_batch_idx]), 1),
                #     device=self.torch_device,
                # )
                # self.update_new_token(new_token, self.model_batches[model_batch_idx], 0)
                # logging.info("Entering kv_storage creation...")
                # self.core_engine.create_fake_kv_storage()
                # self.core_engine.start_h2d_worker()
                # time.sleep(2)
                
                # dist.barrier()
                self._config_decoding()
                # nvtx.init()
                decoding_start_time = time.perf_counter()
                with torch.no_grad():
                    logging.info(
                        f"decoding batch size: {len(self.model_batches[model_batch_idx])}"
                    )
                    self.decoding(new_token, self.model_batches[model_batch_idx])
                decoding_time += time.perf_counter() - decoding_start_time
                self.core_engine.clear_kv_storage()
        else:
            # For small input batch, some worker might do not have any input.
            # In this case, it only participate in the decoding phase.
            # Todo: 
            self._config_decoding()
            decoding_start_time = time.perf_counter()
            with torch.no_grad():
                self.decoding(None, None)
            decoding_time += time.perf_counter() - decoding_start_time
            self.core_engine.clear_kv_storage()


        
        dist.barrier()
        self.model = None 
        torch.cuda.empty_cache()
        dist.destroy_process_group()

        logging.info(f"Prefill total time: {prefill_time}")
        logging.info(f"Decoding total time: {decoding_time}")
        logging.info(f"Waiting for process clean up...")

        res = [
            self.query_book[query_idx].decoded_tokens
            for query_idx in range(self.num_queries)
        ]

        # Print first 5 sequences
        # for query_idx in range(5):
        #     logging.info(
        #         f"Decoded tokens: {res[query_idx].squeeze().tolist()}"
        #     )

        # Show peak memory allocated
        # peak_memory = torch.cuda.max_memory_allocated(self.torch_device)
        # logging.info(f"Peak memory allocated: {peak_memory}")
        return res

    def _parse_state_dict_dp(self):
        model_init_start_time = time.perf_counter()
        self.hf_model_config._attn_implementation = "eager"
        self.model = DeepseekV3ForCausalLM._from_config(
            self.hf_model_config
        ).to(self.engine_config.Basic_Config.device_torch)
        self.model.eval()
        logging.info(
            f"torch module init time: {time.perf_counter() - model_init_start_time} s"
        )

        self.weight_copy_task["attn"] = []
        self.weight_copy_task["routed_expert"] = []
        self.weight_copy_task["shared_expert"] = []

        for layer_idx in trange(self.model_config.num_hidden_layers):
            for name, _ in self.model.model.layers[
                layer_idx
            ].self_attn.named_parameters():
                tensor_full_name = (
                    "model.layers." + str(layer_idx) + ".self_attn." + name
                )
                self.state_dict_name_map[tensor_full_name] = {
                    "module_key": "attn_" + str(layer_idx),
                    "tensor_key": name,
                }
            self.weight_copy_task["attn"].append("attn_" + str(layer_idx))

            if layer_idx >= self.hf_model_config.first_k_dense_replace:
                for name, _ in self.model.model.layers[
                    layer_idx
                ].mlp.shared_experts.named_parameters():
                    tensor_full_name = (
                        "model.layers."
                        + str(layer_idx)
                        + ".mlp.shared_experts."
                        + name
                    )
                    self.state_dict_name_map[tensor_full_name] = {
                        "module_key": "shared_expert_" + str(layer_idx),
                        "tensor_key": name,
                    }
                self.weight_copy_task["shared_expert"].append(
                    "shared_expert_" + str(layer_idx)
                )

                for expert_idx in range(self.model_config.num_local_experts):
                    for name, _ in (
                        self.model.model.layers[layer_idx]
                        .mlp.experts[expert_idx]
                        .named_parameters()
                    ):
                        tensor_full_name = (
                            "model.layers."
                            + str(layer_idx)
                            + ".mlp.experts."
                            + str(expert_idx)
                            + "."
                            + name
                        )
                        self.state_dict_name_map[tensor_full_name] = {
                            "module_key": "routed_expert_"
                            + str(layer_idx)
                            + "_"
                            + str(expert_idx),
                            "tensor_key": name,
                        }
                    self.weight_copy_task["routed_expert"].append(
                        "routed_expert_"
                        + str(layer_idx)
                        + "_"
                        + str(expert_idx)
                    )

    def _config_prefill(self):
        self.model, self.weight_copy_task = self.parallel_manager.configure_prefill()
        self.set_phase("prefill")
        self.core_engine.stop_h2d_worker()
        self.core_engine.clear_weight_copy_queue()
        self.core_engine.reset_prefill_buffer()
        self.core_engine.set_weight_copy_queue(self.weight_copy_task)
        self.core_engine.clear_kv_storage()
        self.core_engine.start_h2d_worker()
    
    def _config_decoding(self):
        logging.info(f"Start Config Decoding")
        self.model = self.model.to("cpu")
        # Set all model parameters to None
        for param in self.model.parameters():
            param.data = torch.zeros(
                1, dtype=torch.bfloat16, device=param.device)
        self.model = None
        torch.cuda.empty_cache()

        self.model, self.weight_copy_task = self.parallel_manager.configure_decoding()
        self.set_phase("decoding")
        self.core_engine.stop_h2d_worker()
        self.core_engine.clear_kv_copy_queue()
        self.core_engine.clear_kv_buffer()
        self.core_engine.clear_weight_copy_queue()
        self.core_engine.reset_decoding_buffer()
        self.core_engine.set_weight_copy_queue(self.weight_copy_task)
        self.core_engine.start_h2d_worker()
        logging.info(f"End Config Decoding")

    def prefill(self, batch: list[int]):
        """
        Handle the prefill for a full model batch.
        """
        if "deepseek" in self.model_config.model_type:
            self.model.model._use_flash_attention_2 = False

        input_ids = torch.cat(
            [
                self.query_book[query_idx].encoded["input_ids"][
                    :, : self.max_input_length
                ]
                for query_idx in batch
            ],
            dim=0,
        )
        attention_masks = torch.cat(
            [
                self.query_book[query_idx].encoded["attention_mask"][
                    :, : self.max_input_length
                ]
                for query_idx in batch
            ],
            dim=0,
        )

        num_prefill_micro_batches = math.ceil(
            len(batch)
            / self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size
        )
        prefill_micro_batch_input_ids = torch.split(
            input_ids,
            self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size,
        )
        Prefill_micro_batch_attention_masks = torch.split(
            attention_masks,
            self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size,
        )
        logging.info(
            f"Number of prefill micro batches: {num_prefill_micro_batches}"
        )
        cur_batch_start = 0
        output_tokens = []
        for micro_batch_idx in tqdm(
            range(num_prefill_micro_batches), desc="Prefill Micro Batch"
        ):
            with torch.no_grad():
                Attn_Wrapper.attention_mask = (
                    Prefill_micro_batch_attention_masks[micro_batch_idx]
                )
                if "deepseek" in self.model_config.model_type:
                    Attn_Wrapper.position_ids = (
                        create_position_ids_from_attention_mask(
                            Prefill_micro_batch_attention_masks[micro_batch_idx]
                        )
                    )
                else:
                    Attn_Wrapper.position_ids = (
                        create_position_ids_from_attention_mask(
                            Prefill_micro_batch_attention_masks[micro_batch_idx]
                        )
                    )
                cur_batch_size = prefill_micro_batch_input_ids[
                    micro_batch_idx
                ].shape[0]
                cur_batch = batch[
                    cur_batch_start : cur_batch_start + cur_batch_size
                ]
                Attn_Wrapper.cur_batch = cur_batch
                cur_batch_start += cur_batch_size
                assert len(cur_batch) == cur_batch_size

                outputs = self.model(
                    prefill_micro_batch_input_ids[micro_batch_idx].to(
                        self.torch_device
                    ),
                    attention_mask=Prefill_micro_batch_attention_masks[
                        micro_batch_idx
                    ].to(self.torch_device),
                    # position_ids=micro_batch_position_ids[micro_batch_idx].to(self.torch_device),
                    use_cache=False,
                )
                # Greedy
                new_tokens = torch.argmax(
                    outputs.logits[:, -1, :], dim=-1
                ).view(-1, 1)
                output_tokens.append(new_tokens)

        new_tokens = torch.cat(output_tokens, dim=0)
        self.update_new_token(new_tokens, batch, 0)
        return new_tokens

    def decoding(self, new_tokens: torch.Tensor, batch: list[int]):
        """
        Handle the decoding for a full model batch.
        All the queries reach <EOS> or the max decoding length.

        return
                - answer_set: dict[query_idx, decoded_tokens]
        """
        if "deepseek" in self.model_config.model_type:
            self.model.model._use_flash_attention_2 = True
        new_token_idx = 1
        # attention_mask = torch.cat([self.query_book[query_idx].encoded["attention_mask"][:,:self.max_max_input_length + new_token_idx] for query_idx in batch], dim=0)
        # if attention_mask.dim() == 2 and (self.model_config.model_type not in ["Qwen2"]):
        #  	attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # 	attention_mask = torch.where(attention_mask == 0, torch.finfo(torch.bfloat16).min, torch.tensor(0.0, dtype=torch.bfloat16, device=attention_mask.device))
        # Attn_Wrapper.attention_mask = attention_mask

        # Log device memory usage
        logging.info(f"{self.rank} Device memory usage: {torch.cuda.memory_allocated(self.torch_device) / (1024**3)} GB")

        while new_token_idx < self.max_decoding_length and len(batch) > 0:
            if self.rank == 0:
                logging.info(f"Decoding new token idx: {new_token_idx}")
            # Step 1: Before each round of decoding, review the attention mode and batching plan.
            # TODO: review attention mode. Current fixing attention mode.
            RUNTIME_ATTN_MODE = self.engine_config.Basic_Config.attn_mode
            if RUNTIME_ATTN_MODE == 2:
                w = float(os.getenv("SPLIT_RATIO_W", None))
                if w is None:
                    logging.info(
                        f"CPU compute ratio not set. Default setting applied."
                    )
                    w = 0.6
                logging.info(f"Split ratio: {w}")

            if RUNTIME_ATTN_MODE == 0:
                """
					CPU ATTN MODE
						- NO ATTN MICRO BATCH
				"""
                # self.set_attn_mode(0)
                # self.core_engine.set_attn_mode(0)
                with torch.no_grad():
                    Attn_Wrapper.cur_batch = [batch]
                    attention_mask = torch.cat(
                        [
                            self.query_book[query_idx].encoded[
                                "attention_mask"
                            ][:, : self.max_input_length + new_token_idx]
                            for query_idx in batch
                        ],
                        dim=0,
                    )
                    if "deepseek" not in self.model_config.model_type:
                        position_ids = create_position_ids_from_attention_mask(
                            attention_mask
                        )[:, -1].unsqueeze(-1)
                    else:
                        position_ids = create_position_ids_from_attention_mask(
                            attention_mask
                        )
                    # DeepSeek use flash-attn by default
                    
                    # if attention_mask.dim() == 2 and (
                    #     self.model_config.model_type not in ["Qwen2", "deepseek"]
                    # ):
                    #     attention_mask = attention_mask.unsqueeze(1).unsqueeze(
                    #         2
                    #     )
                    #     attention_mask = torch.where(
                    #         attention_mask == 0,
                    #         torch.finfo(torch.bfloat16).min,
                    #         torch.tensor(
                    #             0.0,
                    #             dtype=torch.bfloat16,
                    #             device=attention_mask.device,
                    #         ),
                    #     )

                    Attn_Wrapper.attention_mask = attention_mask
                    Attn_Wrapper.position_ids = position_ids
                    new_tokens = self.model(
                        new_tokens.to(self.torch_device),
                        attention_mask=attention_mask.to(self.torch_device),
                        # position_ids=position_ids.to(self.torch_device),
                        use_cache=False,
                    )
                    new_tokens = torch.argmax(new_tokens.logits, dim=-1).view(
                        -1, 1
                    )
                    # logging.info(f"New tokens: {new_tokens}")
                    # start = time.perf_counter()
                    self.update_new_token(new_tokens, batch, new_token_idx)
                    # logging.info(
                    #     f"Update new token time is ms: {(time.perf_counter() - start) * 1000} ms"
                    # )

                # TODO: Temporally remove.
                # Check <EOS>, if <EOS>, remove from batch.
                # for idx, query_idx in enumerate(batch):
                # 	if new_tokens[idx] == self.tokenizer.eos_token_id:
                # 		batch.remove(query_idx)
                new_token_idx += 1

            elif RUNTIME_ATTN_MODE == 1:
                """
					GPU ATTN MODE
						- ATTN MICRO BATCH
				"""
                # Submit KV copy task to the core engine.
                micro_batch_size = self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size
                num_micro_batches = math.ceil(len(batch) / micro_batch_size)
                # logging.info(f"num_micro_batches: {num_micro_batches}")
                micro_batches = [
                    batch[
                        micro_batch_idx * micro_batch_size : (
                            micro_batch_idx + 1
                        )
                        * micro_batch_size
                    ]
                    for micro_batch_idx in range(num_micro_batches)
                ]
                Attn_Wrapper.cur_batch = micro_batches
                # TODO: init ModelConfig in the initializer.
                # Resub every 32 new tokens.
                if (new_token_idx - 1) % 32 == 0:
                    for idx in range(new_token_idx - 1, new_token_idx + 31):
                        # Note: DeepSeek use fp8 kv.
                        if "deepseek" in self.model_config.model_type:
                            # past_kv_byte_size = (
                            #     (self.max_input_length + idx)
                            #     * self.model_config.compressed_kv_dim
                            #     * 2
                            # )
                            # past_kv_byte_size = (
                            #     (self.max_input_length + idx)
                            #     * self.model_config.compressed_kv_dim
                            # )

                            # Copy one more token to avoid torch::cat in attention forward.
                            past_kv_byte_size = (
                                (self.max_input_length + idx + 1)
                                * self.model_config.compressed_kv_dim
                            )

                        elif "mixtral" in self.model_config.model_type:
                            past_kv_byte_size = (
                                (self.max_input_length + idx)
                                * self.model_config.num_key_value_heads
                                * self.model_config.head_dim
                                * 2
                            )
                        else:
                            raise ValueError(
                                f"Model architecture {self.model_config.model_type} not supported yet."
                            )

                        for layer_idx in range(
                            self.model_config.num_hidden_layers
                        ):
                            for micro_batch_idx in range(num_micro_batches):
                                cur_batch = micro_batches[micro_batch_idx]
                                # logging.info(f"token idx: {idx}, layer idx: {layer_idx}, micro_batch_idx: {micro_batch_idx} current batch: {cur_batch}")
                                self.core_engine.submit_to_KV_queue(
                                    cur_batch,
                                    micro_batch_idx,
                                    layer_idx,
                                    past_kv_byte_size,
                                )

                with torch.no_grad():
                    attention_mask = torch.cat(
                        [
                            self.query_book[query_idx].encoded[
                                "attention_mask"
                            ][:, : self.max_input_length + new_token_idx]
                            for query_idx in batch
                        ],
                        dim=0,
                    ).to(self.torch_device)
                    # logging.info(f"engine.py attention_mask: {attention_mask}")   
                    if "deepseek" in self.model_config.model_type:
                        position_ids = create_position_ids_from_attention_mask(
                            attention_mask
                        )
                    else:
                        position_ids = create_position_ids_from_attention_mask(
                            attention_mask
                        )[:, -1].unsqueeze(-1)

                    # if attention_mask.dim() == 2 and (
                    #     self.model_config.model_type not in ["Qwen2", "deepseek"]
                    # ):
                    #     attention_mask = attention_mask.unsqueeze(1).unsqueeze(
                    #         2
                    #     )
                    #     attention_mask = torch.where(
                    #         attention_mask == 0,
                    #         torch.finfo(torch.bfloat16).min,
                    #         torch.tensor(
                    #             0.0,
                    #             dtype=torch.bfloat16,
                    #             device=attention_mask.device,
                    #         ),
                    #     )

                    Attn_Wrapper.attention_mask = attention_mask
                    Attn_Wrapper.position_ids = position_ids
                    # logging.info(f"rank: {self.rank} attention_mask: {attention_mask}")
                    # logging.info(f"rank: {self.rank} position_ids: {position_ids}")
                    new_tokens = self.model(
                        new_tokens.to(self.torch_device),
                        attention_mask=attention_mask.to(self.torch_device),
                        # position_ids=position_ids.to(self.torch_device),
                        use_cache=False,
                    )
                    # torch.cuda.synchronize(self.engine_config.Basic_Config.device_torch)
                    # torch.cuda.current_stream(self.engine_config.Basic_Config.device_torch).synchronize()
                    new_tokens = torch.argmax(new_tokens.logits, dim=-1).view(
                        -1, 1
                    )
                    self.update_new_token(new_tokens, batch, new_token_idx)
                    # torch.cuda.synchronize(self.engine_config.Basic_Config.device_torch)
                    # torch.cuda.current_stream(self.engine_config.Basic_Config.device_torch).synchronize()
                    # logging.info(f"New tokens: {new_tokens}")
                new_token_idx += 1

                # Step 1.1 Config new micro_batch size. Magic Number change every 32 new tokens.
                # seq_len = self.query_book[batch[0]].encoded["input_ids"].shape[1] + self.query_book[batch[0]].num_decoded_tokens
                # ATTN_DECODING_MICRO_BATCH_SIZE = self.engine_config.GPU_Buffer_Config.k_buffer_num_tokens // seq_len
            elif RUNTIME_ATTN_MODE == 2:
                """
					CPU-GPU Parallel ATTN
				"""
                # TODO: wordload partitioning.
                CPU_batch = batch[: math.ceil(len(batch) * w)]
                GPU_batch = batch[math.ceil(len(batch) * w) :]
                logging.info(
                    f"CPU batch size: {len(CPU_batch)}, GPU batch size: {len(GPU_batch)}"
                )

                GPU_micro_batch_size = self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size
                num_GPU_micro_batches = math.ceil(
                    len(GPU_batch) / GPU_micro_batch_size
                )
                GPU_micro_batches = [
                    GPU_batch[
                        micro_batch_idx * GPU_micro_batch_size : (
                            micro_batch_idx + 1
                        )
                        * GPU_micro_batch_size
                    ]
                    for micro_batch_idx in range(num_GPU_micro_batches)
                ]
                Attn_Wrapper.cur_batch = [CPU_batch] + GPU_micro_batches
                # TODO:
                if (new_token_idx - 1) % 32 == 0:
                    for idx in range(new_token_idx - 1, new_token_idx + 31):
                        if "deepseek" in self.model_config.model_type:
                            past_kv_byte_size = (
                                (self.max_input_length + idx)
                                * self.model_config.compressed_kv_dim
                                * self.engine_config.Basic_Config.torch_dtype.itemsize
                            )
                        else:
                            past_kv_byte_size = (
                                (self.max_input_length + idx)
                                * self.model_config.num_key_value_heads
                                * self.model_config.head_dim
                                * self.engine_config.Basic_Config.torch_dtype.itemsize
                            )

                        if "deepseek" in self.model_config.model_type:
                            for layer_idx in range(
                                self.model_config.num_hidden_layers
                            ):
                                self.core_engine.submit_to_KV_queue(
                                    cur_batch, 0, layer_idx, past_kv_byte_size
                                )

                        else:
                            for layer_idx in range(
                                self.model_config.num_hidden_layers
                            ):
                                for micro_batch_idx in range(
                                    num_GPU_micro_batches
                                ):
                                    cur_batch = GPU_micro_batches[
                                        micro_batch_idx
                                    ]
                                    self.core_engine.submit_to_KV_queue(
                                        cur_batch,
                                        micro_batch_idx,
                                        layer_idx,
                                        past_kv_byte_size,
                                    )

                with torch.no_grad():
                    attention_mask = torch.cat(
                        [
                            self.query_book[query_idx].encoded[
                                "attention_mask"
                            ][:, : self.max_input_length + new_token_idx]
                            for query_idx in batch
                        ],
                        dim=0,
                    )
                    if "deepseek" not in self.model_config.model_type:
                        position_ids = create_position_ids_from_attention_mask(
                            attention_mask
                        )[:, -1].unsqueeze(-1)
                    else:
                        position_ids = create_position_ids_from_attention_mask(
                            attention_mask
                        )
                    if attention_mask.dim() == 2 and (
                        self.model_config.model_type not in ["Qwen2"]
                    ):
                        attention_mask = attention_mask.unsqueeze(1).unsqueeze(
                            2
                        )
                        attention_mask = torch.where(
                            attention_mask == 0,
                            torch.finfo(torch.bfloat16).min,
                            torch.tensor(
                                0.0,
                                dtype=torch.bfloat16,
                                device=attention_mask.device,
                            ),
                        )

                    Attn_Wrapper.attention_mask = attention_mask
                    Attn_Wrapper.position_ids = position_ids
                    new_tokens = self.model(
                        new_tokens.to(self.torch_device),
                        attention_mask=attention_mask.to(self.torch_device),
                        # position_ids=position_ids,
                        use_cache=False,
                    )
                    # torch.cuda.synchronize(self.engine_config.Basic_Config.device_torch)
                    # torch.cuda.current_stream(self.engine_config.Basic_Config.device_torch).synchronize()
                    new_tokens = torch.argmax(new_tokens.logits, dim=-1).view(
                        -1, 1
                    )
                    # start = time.perf_counter()
                    self.update_new_token(new_tokens, batch, new_token_idx)
                    # torch.cuda.synchronize(self.engine_config.Basic_Config.device_torch)
                    # torch.cuda.current_stream(self.engine_config.Basic_Config.device_torch).synchronize()
                    print(f"New tokens: {new_tokens}")
                    # logging.info(f"Update new token time is ms: {(time.perf_counter() - start) * 1000} ms")
                new_token_idx += 1

    def set_phase(self, phase: str):
        """
        Control different behavior of the engine in different phases.
        """
        torch.cuda.empty_cache()
        self.core_engine.set_phase(phase)
        Attn_Wrapper.phase = phase
        Expert_Wrapper.phase = phase

    def set_mode(self, mode: str):
        """
        Control different behavior of the engine in different phases.
        """
        pass

    # def update_new_token(
    #     self, new_tokens: torch.Tensor, query_idx: List[int], new_token_idx: int
    # ):
    #     new_tokens = new_tokens.to("cpu")
    #     for idx, q_idx in enumerate(query_idx):
    #         self.query_book[q_idx].decoded_tokens[:, new_token_idx] = (
    #             new_tokens[idx]
    #         )
    #         self.query_book[q_idx].encoded["input_ids"][
    #             0, new_token_idx + self.max_input_length
    #         ] = new_tokens[idx]
    #         self.query_book[q_idx].encoded["attention_mask"][
    #             0, new_token_idx + self.max_input_length
    #         ] = torch.tensor(1, dtype=torch.int64)


    def update_new_token(
        self, new_tokens: torch.Tensor, query_idx: List[int], new_token_idx: int
    ):
        new_tokens = new_tokens.to("cpu")
        for idx, q_idx in enumerate(query_idx):
            # Update decoded tokens
            self.query_book[q_idx].decoded_tokens[:, new_token_idx] = new_tokens[idx]
            
            # Update encoded input_ids
            # self.query_book[q_idx].encoded["input_ids"][
            #     0, new_token_idx + self.max_input_length
            # ] = new_tokens[idx]
            
            # Get the current attention mask
            attention_mask = self.query_book[q_idx].encoded["attention_mask"][0]
            
            # Find the first 0 in the attention mask
            zeros_positions = (attention_mask == 0).nonzero(as_tuple=True)[0]
            # logging.info(f"zeros_positions: {zeros_positions}")
            if len(zeros_positions) > 0:
                # If a 0 is found, change the first one to 1
                first_zero_pos = zeros_positions[0].item()
                self.query_book[q_idx].encoded["attention_mask"][0, first_zero_pos] = torch.tensor(1, dtype=attention_mask.dtype)
                # self.query_book[q_idx].encoded["input_ids"][0, first_zero_pos] = new_tokens[idx]
            else:
                raise ValueError("No 0 found in the attention mask.")
