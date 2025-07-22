import ctypes
import json
import logging
import os

import safetensors
import torch
from safetensors.torch import load_file


class ckpt_converter:
    """
    Convert .safetesors or .pt checkpoints to a format compatible with MoE-Gen.
    The main purpose is to achieve peak read performance of SSD.

    This converter provide functionality to read in a file and store the tensors in contiguous format one after one.
    We use a dict to store the metadata of this file.
            "file_name": str,
            "state_dict": dict{
                    "tensor_name": {
                            ”dtype": str,
                            "shape": tuple,
                            "offset": int64,
                            "byte_size": int64
                    }
            }
            "total_byte_size": int64

    We save the tensors in a file. And the metadata in a json file.
    """

    def __init__(self):
        pass

    def _dtype_to_str(self, dtype):
        """
        Convert torch dtype to string.
        """
        if dtype == torch.float32:
            return "float32"
        elif dtype == torch.float16:
            return "float16"
        elif dtype == torch.bfloat16:
            return "bfloat16"
        elif dtype == torch.float8_e4m3fn:
            return "float8_e4m3fn"
        elif dtype == torch.int64:
            return "int64"
        elif dtype == torch.int32:
            return "int32"
        elif dtype == torch.uint8:
            return "uint8"
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def convert(self, ckpt_path, output_dir):
        # Check if the file dir exists
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint file path {ckpt_path} does not exist."
            )

        # Check if in compatible format
        if not (
            ckpt_path.endswith(".safetensors") or ckpt_path.endswith(".pt")
        ):
            raise ValueError(
                f"Checkpoint file {ckpt_path} is not in .safetensors or .pt format."
            )

        # Create output directory if not exists
        if not os.path.exists(output_dir):
            logging.info(
                f"Output directory {output_dir} does not exist. Creating it."
            )
            os.makedirs(output_dir, exist_ok=True)

        # Load the checkpoint
        logging.debug(f"Loading checkpoint from {ckpt_path}.")
        if ckpt_path.endswith(".safetensors"):
            ckpt = load_file(ckpt_path)
        else:
            ckpt = torch.load(ckpt_path, weights_only=True)

        out_file_name = os.path.join(
            output_dir,
            os.path.basename(ckpt_path).replace(".safetensors", ".bin"),
        ).replace(".pt", ".bin")
        out_metadata_name = os.path.join(
            output_dir,
            os.path.basename(ckpt_path)
            .replace(".safetensors", ".json")
            .replace(".pt", ".json"),
        )

        metadata = {}
        metadata["file_name"] = os.path.basename(out_file_name)
        metadata["state_dict"] = {}
        total_byte_size = 0

        with open(out_file_name, "wb") as out_file:
            for tensor_name, tensor in ckpt.items():
                if not isinstance(tensor, torch.Tensor):
                    logging.warning(
                        f"Skipping non-tensor {tensor_name} in checkpoint."
                    )
                    continue
                # Ensure tensor is contiguous
                tensor = tensor.contiguous()
                # Get tensor metadata
                dtype_str = self._dtype_to_str(tensor.dtype)
                shape = tuple(tensor.shape)
                offset = total_byte_size
                tensor_byte_size = tensor.element_size() * tensor.numel()
                total_byte_size += tensor_byte_size

                metadata["state_dict"][tensor_name] = {
                    "dtype": dtype_str,
                    "shape": shape,
                    "offset": offset,
                    "byte_size": tensor_byte_size,
                }
                # Write tensor to file
                data_ptr = tensor.data_ptr()
                buf = (ctypes.c_char * tensor_byte_size).from_address(data_ptr)
                # Write tensor data to file
                out_file.write(buf)
            metadata["total_byte_size"] = total_byte_size

        # Save metadata to json file
        with open(out_metadata_name, "w") as metadata_file:
            json.dump(metadata, metadata_file, indent=4)
