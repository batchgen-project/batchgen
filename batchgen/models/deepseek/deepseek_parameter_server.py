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

import logging
import os
import shutil
from multiprocessing import Process

import torch
from safetensors.torch import load_file
from tqdm import tqdm, trange
from transformers import AutoConfig
import gc

from .deepseekv2.configuration_deepseek_v2 import DeepseekV2Config
from .deepseekv2.modeling_deepseek_v2 import DeepseekV2ForCausalLM
from .deepseekv3.configuration_deepseek_v3 import DeepseekV3Config
from .deepseekv3.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from ...ckpt_converter.ckpt_converter import ckpt_converter

try:
    from batchgen.core_engine import Parameter_Server
except ImportError:
    from batchgen.models.engine_loader import core_engine  # noqa
    from core_engine import Parameter_Server  # noqa


class DeepSeek_Parameter_Server:
    def __init__(self, huggingface_ckpt_name, cache_dir, pt_ckpt_dir, enable_hugetlbfs):
        self.cache_dir = cache_dir
        self.huggingface_ckpt_name = huggingface_ckpt_name
        self.pt_ckpt_dir = pt_ckpt_dir
        self.weight_copy_task = {}
        self.state_dict_name_map = {}
        self.enable_hugetlbfs = enable_hugetlbfs
        config_cls = (
            DeepseekV2Config
            if "V2" in huggingface_ckpt_name
            else DeepseekV3Config
        )
        self.hf_model_config = config_cls.from_pretrained(
            huggingface_ckpt_name, trust_remote_code=True
        )
        self.hf_model_config._name_or_path = huggingface_ckpt_name
        self.hf_model_config.architectures = ["DeepseekV3ForCausalLM"]
        free_memory, total_memory = torch.cuda.mem_get_info()
        gpu0_memory = free_memory / 1024 / 1024 / 1024
        total_memory = total_memory / 1024 / 1024 / 1024
        logging.info(f"Python PM instantiation: GPU 0 free memory: {gpu0_memory} GB / {total_memory} GB")




        # self.hf_model_config = AutoConfig.from_pretrained(huggingface_ckpt_name, cache_dir=cache_dir, trust_remote_code=True)

    def Init(self):
        # log gpu 0 memory usage before init
        free_memory, total_memory = torch.cuda.mem_get_info()
        gpu0_memory = free_memory / 1024 / 1024 / 1024
        total_memory = total_memory / 1024 / 1024 / 1024
        logging.info(f"GPU 0 free mem pm start Init: {gpu0_memory} GB / {total_memory} GB")
        
        self._save_safetensors_to_pt()
        free_memory, total_memory = torch.cuda.mem_get_info()
        gpu0_memory = free_memory / 1024 / 1024 / 1024
        total_memory = total_memory / 1024 / 1024 / 1024
        logging.info(f"GPU 0 free mem after save safetensors to pt: {gpu0_memory} GB / {total_memory} GB")

        self._parse_state_dict()
        free_memory, total_memory = torch.cuda.mem_get_info()
        gpu0_memory = free_memory / 1024 / 1024 / 1024
        total_memory = total_memory / 1024 / 1024 / 1024
        logging.info(f"GPU 0 free mem before cpp pm instantiate: {gpu0_memory} GB / {total_memory} GB")

        self.parameter_server = Parameter_Server(self.enable_hugetlbfs)
        logging.info(f"architectures: {self.hf_model_config.architectures[0]}")
        if self.hf_model_config.architectures[0] == "DeepseekV2ForCausalLM":
            if "DeepSeek-V2-Lite" in self.huggingface_ckpt_name:
                byte_size = 32 * 1024 * 1024 * 1024
            else:
                byte_size = 236 * 1024 * 1024 * 1024 * 2
        elif self.hf_model_config.architectures[0] == "DeepseekV3ForCausalLM":
            byte_size = 675 * 1024 * 1024 * 1024
        else:
            raise ValueError("Unknown huggingface model card")

        total, used, free = shutil.disk_usage("/dev/shm")
        logging.info(f"Freespace in /dev/shm: {free/1024/1024/1024} GB")
        if free < byte_size:
            raise ValueError(
                f"Shared memory size is not enough. Required: {byte_size}, Available: {free}. \
                Please clear /dev/shm or increase the size by running 'sudo mount -o remount,size=<size>G /dev/shm'"
            )

        import uuid

        self.shm_name = "/shm_" + str(uuid.uuid4())
        self.tensor_meta_shm_name = "/shm_" + str(uuid.uuid4())
        logging.info(f"Model parameters shared memory name: {self.shm_name}")
        logging.info(
            f"Tensor meta shared memory name: {self.tensor_meta_shm_name}"
        )
        logging.info(f"Byte size: {byte_size}")
        # logging gpu 0 free memory
        free_memory, total_memory = torch.cuda.mem_get_info()
        gpu0_memory = free_memory / 1024 / 1024 / 1024
        total_memory = total_memory / 1024 / 1024 / 1024
        logging.info(f"GPU 0 free memory before cpp pm Init: {gpu0_memory} GB / {total_memory} GB")
        # Todo: Fix this
        self.pt_ckpt_dir = os.path.join(self.cache_dir, "converted_ckpt")
        file_list = []
        for file_name in os.listdir(self.cache_dir):
            if file_name.endswith(".safetensors") or file_name.endswith(".pt"):
                file_list.append(os.path.join(self.cache_dir, file_name))
        if not os.path.exists(self.pt_ckpt_dir):
            os.makedirs(self.pt_ckpt_dir, exist_ok=True)
            logging.info(f"First time run this model, converting the checkpoint files to BatchGen compatible format.")
            converter = ckpt_converter()
            for file_path in tqdm(file_list, desc="Converting checkpoint files", smoothing=0):
                logging.debug(f"Converting {file_path} to {self.pt_ckpt_dir}")
                converter.convert(file_path, self.pt_ckpt_dir)
        else:
            # Check if files are corrupted
            # Num of metadata(.json) and bin(.bin) files should be the same as file_list
            # Also they should be in the same name exclusing the extension
            metadata_files = []
            bin_files = []
            for file_name in os.listdir(self.pt_ckpt_dir):
                if file_name.endswith(".json"):
                    metadata_files.append(os.path.join(self.pt_ckpt_dir, file_name))
                elif file_name.endswith(".bin"):
                    bin_files.append(os.path.join(self.pt_ckpt_dir, file_name))
            if len(metadata_files) != len(bin_files):
                raise ValueError(
                    f"Metadata files {len(metadata_files)} and bin files {len(bin_files)} do not match. \
                    Please clean the converted_ckpt dir in {self.cache_dir} and run again."
                )
            if len(metadata_files) != len(file_list):
                raise ValueError(
                    f"Metadata files {len(metadata_files)} and original checkpoint files {len(file_list)} do not match. \
                    Please clean the converted_ckpt dir in {self.cache_dir} and run again."
                )
            for files in file_list:
                file_name = os.path.basename(files)
                metadata_file = os.path.join(
                    self.pt_ckpt_dir, file_name.replace(".safetensors", ".json").replace(".pt", ".json")
                )
                bin_file = os.path.join(
                    self.pt_ckpt_dir, file_name.replace(".safetensors", ".bin").replace(".pt", ".bin")
                )
                if not os.path.exists(metadata_file):
                    raise FileNotFoundError(f"Metadata file {metadata_file} does not exist. \
                    Please clean the converted_ckpt dir in {self.cache_dir} and run again.")    
                if not os.path.exists(bin_file):
                    raise FileNotFoundError(f"Bin file {bin_file} does not exist. \
                    Please clean the converted_ckpt dir in {self.cache_dir} and run again.")
         

        self.parameter_server.Init(
            self.shm_name,
            self.tensor_meta_shm_name,
            byte_size,
            self.pt_ckpt_dir,
            self.state_dict_name_map,
        )
        return self.shm_name, self.tensor_meta_shm_name

    def _parse_state_dict(self):
        self.hf_model_config._attn_implementation = "eager"
        if self.hf_model_config.architectures[0] == "DeepseekV2ForCausalLM":
            model = DeepseekV2ForCausalLM._from_config(self.hf_model_config).to('cpu')
        elif self.hf_model_config.architectures[0] == "DeepseekV3ForCausalLM":
            model = DeepseekV3ForCausalLM._from_config(self.hf_model_config).to('cpu')
        else:
            raise ValueError("Unknown model architecture")
        model.eval()

        self.weight_copy_task["attn"] = []
        self.weight_copy_task["routed_expert"] = []
        self.weight_copy_task["shared_expert"] = []

        for layer_idx in trange(
            len(model.model.layers), desc="Parsing state_dict"
        ):
            for name, _ in model.model.layers[
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
                for name, _ in model.model.layers[
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

                for expert_idx in range(
                    len(model.model.layers[layer_idx].mlp.experts)
                ):
                    for name, _ in (
                        model.model.layers[layer_idx]
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
        del model
        gc.collect()  
        torch.cuda.empty_cache() 



    def save_and_load(self, file_path, save_dir):
        tensor_dict = load_file(file_path)
        torch.save(tensor_dict, save_dir)
        return tensor_dict

    def _save_safetensors_to_pt(self):
        logging.info(f"cache_dir: {self.cache_dir}")
        ckpt_files = os.listdir(self.cache_dir)
        ckpt_files = [
            os.path.join(self.cache_dir, ckpt)
            for ckpt in ckpt_files
            if ckpt.endswith(".safetensors")
        ]
        processes = []
        for ckpt in tqdm(
            ckpt_files, desc="Loading checkpoint files", smoothing=0
        ):
            dst_dir = os.path.join(
                self.pt_ckpt_dir,
                ckpt.split("/")[-1].replace(".safetensors", ".pt"),
            )
            # logging.info(f"dst_dir: {dst_dir}")
            if os.path.exists(dst_dir):
                continue

            logging.info(
                f"Checkpoint file: {ckpt} not found in {self.pt_ckpt_dir}. Dump it now. Will omit this step next time run this model."
            )
            p = Process(target=self.save_and_load, args=(ckpt, dst_dir))
            p.start()
            processes.append(p)

        try:
            for p in processes:
                p.join()
                if (
                    p.exitcode != 0
                ):  # Check if process terminated with non-zero exit code
                    print(f"Process terminated with exit code {p.exitcode}")
                    import sys

                    sys.exit(1)  # Terminate the program with error code
                p.close()
        except Exception as e:
            print(f"Error occurred: {e}")
            import sys

            sys.exit(1)  # Terminate the program with error code
        logging.info("All safetensor loader processes joined")
