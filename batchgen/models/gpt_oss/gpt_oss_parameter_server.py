# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
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

"""Parameter server for GPT-OSS-120B model.

Handles model weight loading and shared memory allocation for GPT-OSS.
Model specs:
- 36 layers, 128 experts, 117B total params (5.1B active)
- MXFP4 quantized weights (~55 GB for MoE, ~3 GB for attention)
"""

import gc
import logging
import os
import shutil
import uuid
from multiprocessing import Process

import torch
from safetensors.torch import load_file
from tqdm import tqdm, trange
from transformers import AutoConfig

from .configuration_gpt_oss import GptOssConfig
from .modeling_gpt_oss import GptOssForCausalLM
from ...ckpt_converter.ckpt_converter import ckpt_converter

try:
    from batchgen.core_engine import Parameter_Server
except ImportError:
    from batchgen.models.engine_loader import core_engine
    Parameter_Server = core_engine.Parameter_Server


class GptOss_Parameter_Server:
    """Parameter server for GPT-OSS-120B model weights.

    Manages shared memory allocation for model parameters, enabling
    efficient weight loading across multiple GPU workers.

    Args:
        huggingface_ckpt_name: HuggingFace model identifier (e.g., "openai/gpt-oss-120b")
        cache_dir: Directory containing model checkpoint files
        pt_ckpt_dir: Directory for converted PyTorch checkpoint files
        enable_hugetlbfs: Whether to use hugepages for shared memory
    """

    def __init__(
        self,
        huggingface_ckpt_name: str,
        cache_dir: str,
        pt_ckpt_dir: str,
        enable_hugetlbfs: bool,
    ):
        self.cache_dir = cache_dir
        self.huggingface_ckpt_name = huggingface_ckpt_name
        self.pt_ckpt_dir = pt_ckpt_dir
        self.weight_copy_task = {}
        self.state_dict_name_map = {}
        self.enable_hugetlbfs = enable_hugetlbfs

        # Load model configuration
        self.hf_model_config = GptOssConfig.from_pretrained(
            huggingface_ckpt_name,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.hf_model_config._name_or_path = huggingface_ckpt_name
        self.hf_model_config.architectures = ["GptOssForCausalLM"]

        free_memory, total_memory = torch.cuda.mem_get_info()
        gpu0_memory = free_memory / 1024 / 1024 / 1024
        total_memory = total_memory / 1024 / 1024 / 1024
        logging.info(
            f"GPT-OSS Parameter Server instantiation: GPU 0 free memory: {gpu0_memory:.2f} GB / {total_memory:.2f} GB"
        )

    def Init(self):
        """Initialize the parameter server and allocate shared memory."""
        free_memory, total_memory = torch.cuda.mem_get_info()
        gpu0_memory = free_memory / 1024 / 1024 / 1024
        total_memory = total_memory / 1024 / 1024 / 1024
        logging.info(
            f"GPU 0 free mem at Init start: {gpu0_memory:.2f} GB / {total_memory:.2f} GB"
        )

        self._parse_state_dict()

        free_memory, total_memory = torch.cuda.mem_get_info()
        gpu0_memory = free_memory / 1024 / 1024 / 1024
        total_memory = total_memory / 1024 / 1024 / 1024
        logging.info(
            f"GPU 0 free mem before cpp PM instantiate: {gpu0_memory:.2f} GB / {total_memory:.2f} GB"
        )

        self.parameter_server = Parameter_Server(self.enable_hugetlbfs)

        # Calculate required shared memory size for GPT-OSS-120B
        # MXFP4 quantized: ~55 GB for experts + ~3 GB for attention + ~2 GB for embeddings
        # Total estimated: ~60 GB
        byte_size = 65 * 1024 * 1024 * 1024  # 65 GB with buffer

        total, used, free = shutil.disk_usage("/dev/shm")
        logging.info(f"Freespace in /dev/shm: {free / 1024 / 1024 / 1024:.2f} GB")
        if free < byte_size:
            raise ValueError(
                f"Shared memory size is not enough. Required: {byte_size / 1024 / 1024 / 1024:.2f} GB, "
                f"Available: {free / 1024 / 1024 / 1024:.2f} GB. "
                f"Please clear /dev/shm or increase the size by running "
                f"'sudo mount -o remount,size=<size>G /dev/shm'"
            )

        self.shm_name = "/shm_" + str(uuid.uuid4())
        self.tensor_meta_shm_name = "/shm_" + str(uuid.uuid4())
        logging.info(f"Model parameters shared memory name: {self.shm_name}")
        logging.info(f"Tensor meta shared memory name: {self.tensor_meta_shm_name}")
        logging.info(f"Byte size: {byte_size / 1024 / 1024 / 1024:.2f} GB")

        free_memory, total_memory = torch.cuda.mem_get_info()
        gpu0_memory = free_memory / 1024 / 1024 / 1024
        total_memory = total_memory / 1024 / 1024 / 1024
        logging.info(
            f"GPU 0 free memory before cpp PM Init: {gpu0_memory:.2f} GB / {total_memory:.2f} GB"
        )

        # Convert checkpoint files to BatchGen format (or validate existing conversion)
        converter = ckpt_converter()
        self.pt_ckpt_dir = converter.convert_model_directory(self.cache_dir)

        self.parameter_server.Init(
            self.shm_name,
            self.tensor_meta_shm_name,
            byte_size,
            self.pt_ckpt_dir,
            self.state_dict_name_map,
        )
        return self.shm_name, self.tensor_meta_shm_name

    def _parse_state_dict(self):
        """Parse model state dict to build weight copy tasks and name mapping."""
        self.hf_model_config._attn_implementation = "eager"

        model = GptOssForCausalLM._from_config(self.hf_model_config).to("cpu")
        model.eval()

        self.weight_copy_task["attn"] = []
        self.weight_copy_task["routed_expert"] = []

        num_layers = self.hf_model_config.num_hidden_layers
        num_experts = self.hf_model_config.num_local_experts

        for layer_idx in trange(num_layers, desc="Parsing GPT-OSS state_dict"):
            # Attention weights
            for name, _ in model.model.layers[layer_idx].self_attn.named_parameters():
                tensor_full_name = f"model.layers.{layer_idx}.self_attn.{name}"
                self.state_dict_name_map[tensor_full_name] = {
                    "module_key": f"attn_{layer_idx}",
                    "tensor_key": name,
                }
            self.weight_copy_task["attn"].append(f"attn_{layer_idx}")

            # Expert weights (MoE)
            for expert_idx in range(num_experts):
                for name, _ in (
                    model.model.layers[layer_idx].mlp.experts[expert_idx].named_parameters()
                ):
                    tensor_full_name = (
                        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{name}"
                    )
                    self.state_dict_name_map[tensor_full_name] = {
                        "module_key": f"routed_expert_{layer_idx}_{expert_idx}",
                        "tensor_key": name,
                    }
                self.weight_copy_task["routed_expert"].append(
                    f"routed_expert_{layer_idx}_{expert_idx}"
                )

            # Router weights (gate)
            for name, _ in model.model.layers[layer_idx].mlp.router.named_parameters():
                tensor_full_name = f"model.layers.{layer_idx}.mlp.router.{name}"
                self.state_dict_name_map[tensor_full_name] = {
                    "module_key": f"router_{layer_idx}",
                    "tensor_key": name,
                }

            # Layer norms
            for name, _ in model.model.layers[layer_idx].input_layernorm.named_parameters():
                tensor_full_name = f"model.layers.{layer_idx}.input_layernorm.{name}"
                self.state_dict_name_map[tensor_full_name] = {
                    "module_key": f"input_layernorm_{layer_idx}",
                    "tensor_key": name,
                }

            for name, _ in model.model.layers[layer_idx].post_attention_layernorm.named_parameters():
                tensor_full_name = f"model.layers.{layer_idx}.post_attention_layernorm.{name}"
                self.state_dict_name_map[tensor_full_name] = {
                    "module_key": f"post_attention_layernorm_{layer_idx}",
                    "tensor_key": name,
                }

        # Embedding and final norm
        for name, _ in model.model.embed_tokens.named_parameters():
            tensor_full_name = f"model.embed_tokens.{name}"
            self.state_dict_name_map[tensor_full_name] = {
                "module_key": "embed_tokens",
                "tensor_key": name,
            }

        for name, _ in model.model.norm.named_parameters():
            tensor_full_name = f"model.norm.{name}"
            self.state_dict_name_map[tensor_full_name] = {
                "module_key": "final_norm",
                "tensor_key": name,
            }

        # LM head
        for name, _ in model.lm_head.named_parameters():
            tensor_full_name = f"lm_head.{name}"
            self.state_dict_name_map[tensor_full_name] = {
                "module_key": "lm_head",
                "tensor_key": name,
            }

        del model
        gc.collect()
        torch.cuda.empty_cache()

    def save_and_load(self, file_path, save_dir):
        """Convert safetensors file to PyTorch format."""
        tensor_dict = load_file(file_path)
        torch.save(tensor_dict, save_dir)
        return tensor_dict

    def _save_safetensors_to_pt(self):
        """Convert safetensors checkpoint files to PyTorch format."""
        logging.info(f"cache_dir: {self.cache_dir}")
        ckpt_files = os.listdir(self.cache_dir)
        ckpt_files = [
            os.path.join(self.cache_dir, ckpt)
            for ckpt in ckpt_files
            if ckpt.endswith(".safetensors")
        ]
        processes = []
        for ckpt in tqdm(ckpt_files, desc="Loading checkpoint files", smoothing=0):
            dst_dir = os.path.join(
                self.pt_ckpt_dir,
                ckpt.split("/")[-1].replace(".safetensors", ".pt"),
            )
            if os.path.exists(dst_dir):
                continue

            logging.info(
                f"Checkpoint file: {ckpt} not found in {self.pt_ckpt_dir}. "
                f"Converting now. Will skip this step next time."
            )
            p = Process(target=self.save_and_load, args=(ckpt, dst_dir))
            p.start()
            processes.append(p)

        try:
            for p in processes:
                p.join()
                if p.exitcode != 0:
                    logging.error(f"Process terminated with exit code {p.exitcode}")
                    import sys
                    sys.exit(1)
                p.close()
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            import sys
            sys.exit(1)
        logging.info("All safetensor loader processes joined")
