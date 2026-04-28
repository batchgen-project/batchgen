# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
# ---------------------------------------------------------------------------- #

"""DeepSeek-V4-Flash parameter server integration.

This class enumerates V4 checkpoint keys directly from the V4 assets contract.
It is intentionally separate from ``deepseek_parameter_server.py`` so V4 never
instantiates the DeepSeek-V3 skeleton as a side effect of startup.
"""

from __future__ import annotations

import gc
import logging
import os
import shutil
import uuid

import torch

from batchgen.ckpt_converter.ckpt_converter import ckpt_converter
from batchgen.config.model_registry import load_config
from batchgen.server.process_utils import get_model_byte_size

from .tensor_contract import build_v4_weight_contract

try:
    from batchgen.core_engine import Parameter_Server
except ImportError:
    from batchgen.models.engine_loader import core_engine  # noqa

    Parameter_Server = core_engine.Parameter_Server  # noqa


class DeepSeekV4Flash_Parameter_Server:
    def __init__(
        self,
        huggingface_ckpt_name,
        cache_dir,
        converted_ckpt_dir,
        enable_hugetlbfs,
        enable_memfd=False,
    ):
        self.cache_dir = cache_dir
        self.huggingface_ckpt_name = huggingface_ckpt_name
        self.converted_ckpt_dir = converted_ckpt_dir
        self.weight_copy_task = {}
        self.state_dict_name_map = {}
        self.enable_hugetlbfs = enable_hugetlbfs
        self.enable_memfd = enable_memfd
        self.model_config = load_config(huggingface_ckpt_name)

    def Init(self):
        self._parse_state_dict()
        self.parameter_server = Parameter_Server(
            self.enable_hugetlbfs,
            self.enable_memfd,
        )

        byte_size = get_model_byte_size(self.huggingface_ckpt_name)
        total, used, free = shutil.disk_usage("/dev/shm")
        logging.info("Freespace in /dev/shm: %.2f GB", free / 1024**3)
        if free < byte_size:
            raise ValueError(
                "Shared memory size is not enough. "
                f"Required: {byte_size}, Available: {free}. "
                "Please clear /dev/shm or increase the size."
            )

        self.shm_name = "/shm_" + str(uuid.uuid4())
        self.tensor_meta_shm_name = "/shm_" + str(uuid.uuid4())
        logging.info("V4 parameter shared memory name: %s", self.shm_name)
        logging.info("V4 tensor meta shared memory name: %s", self.tensor_meta_shm_name)
        logging.info("V4 parameter-server byte size: %.2f GB", byte_size / 1024**3)

        if self.converted_ckpt_dir is None:
            converter = ckpt_converter()
            self.converted_ckpt_dir = converter.convert_model_directory(self.cache_dir)
        else:
            logging.info("Using pre-converted V4 checkpoint: %s", self.converted_ckpt_dir)

        self.parameter_server.Init(
            self.shm_name,
            self.tensor_meta_shm_name,
            byte_size,
            str(self.converted_ckpt_dir),
            self.state_dict_name_map,
        )
        return self.shm_name, self.tensor_meta_shm_name

    def _parse_state_dict(self):
        self.state_dict_name_map, self.weight_copy_task = build_v4_weight_contract(
            self.model_config
        )
        logging.info(
            "DeepSeek-V4 weight contract: %d mapped tensors, %d attn modules, "
            "%d routed expert modules, %d shared expert modules",
            len(self.state_dict_name_map),
            len(self.weight_copy_task["attn"]),
            len(self.weight_copy_task["routed_expert"]),
            len(self.weight_copy_task["shared_expert"]),
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
