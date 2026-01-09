import json
import safetensors
from safetensors.torch import load_file
import torch
import os
import logging
import ctypes	
class ckpt_converter:
	"""
		Convert .safetesors or .pt checkpoints to a format compatible with BatchGen.
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
			raise FileNotFoundError(f"Checkpoint file path {ckpt_path} does not exist.")
	
		# Check if in compatible format
		if not (ckpt_path.endswith(".safetensors") or ckpt_path.endswith(".pt")):
			raise ValueError(f"Checkpoint file {ckpt_path} is not in .safetensors or .pt format.")
		
		# Create output directory if not exists
		if not os.path.exists(output_dir):
			logging.info(f"Output directory {output_dir} does not exist. Creating it.")
			os.makedirs(output_dir, exist_ok=True)
		
		# Load the checkpoint
		logging.debug(f"Loading checkpoint from {ckpt_path}.")
		if ckpt_path.endswith(".safetensors"):
			ckpt = load_file(ckpt_path)
		else:
			ckpt = torch.load(ckpt_path, weights_only=True)
		
		out_file_name = os.path.join(output_dir, os.path.basename(ckpt_path).replace(".safetensors", ".bin")).replace(".pt", ".bin")
		out_metadata_name = os.path.join(output_dir, os.path.basename(ckpt_path).replace(".safetensors", ".json").replace(".pt", ".json"))	
	
		metadata = {}
		metadata["file_name"] = os.path.basename(out_file_name)
		metadata["state_dict"] = {}
		total_byte_size = 0
		

		with open(out_file_name, "wb") as out_file:
			for tensor_name, tensor in ckpt.items():
				if not isinstance(tensor, torch.Tensor):
					logging.warning(f"Skipping non-tensor {tensor_name} in checkpoint.")
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
					"byte_size": tensor_byte_size
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

	def _get_checkpoint_files(self, input_dir):
		"""
		Get list of checkpoint files (.safetensors or .pt) in a directory.

		Args:
			input_dir: Directory to scan for checkpoint files

		Returns:
			List of full paths to checkpoint files
		"""
		file_list = []
		for file_name in os.listdir(input_dir):
			if file_name.endswith(".safetensors") or file_name.endswith(".pt"):
				file_list.append(os.path.join(input_dir, file_name))
		return sorted(file_list)

	def validate_converted_directory(self, input_dir, output_dir):
		"""
		Validate that converted checkpoint files are consistent with source files.

		Args:
			input_dir: Directory containing source .safetensors or .pt files
			output_dir: Directory containing converted .bin and .json files

		Returns:
			Tuple of (is_valid: bool, error_message: str or None)
		"""
		file_list = self._get_checkpoint_files(input_dir)

		if not file_list:
			return False, f"No checkpoint files (.safetensors or .pt) found in {input_dir}"

		# Count metadata and bin files
		metadata_files = []
		bin_files = []
		for file_name in os.listdir(output_dir):
			if file_name.endswith(".json"):
				metadata_files.append(os.path.join(output_dir, file_name))
			elif file_name.endswith(".bin"):
				bin_files.append(os.path.join(output_dir, file_name))

		# Check counts match
		if len(metadata_files) != len(bin_files):
			return False, (
				f"Metadata files ({len(metadata_files)}) and bin files ({len(bin_files)}) count mismatch. "
				f"Please clean {output_dir} and reconvert."
			)

		if len(metadata_files) != len(file_list):
			return False, (
				f"Converted files ({len(metadata_files)}) and source checkpoint files ({len(file_list)}) count mismatch. "
				f"Please clean {output_dir} and reconvert."
			)

		# Check each source file has corresponding converted files
		for src_file in file_list:
			file_name = os.path.basename(src_file)
			metadata_file = os.path.join(
				output_dir,
				file_name.replace(".safetensors", ".json").replace(".pt", ".json")
			)
			bin_file = os.path.join(
				output_dir,
				file_name.replace(".safetensors", ".bin").replace(".pt", ".bin")
			)

			if not os.path.exists(metadata_file):
				return False, (
					f"Metadata file {metadata_file} does not exist. "
					f"Please clean {output_dir} and reconvert."
				)
			if not os.path.exists(bin_file):
				return False, (
					f"Bin file {bin_file} does not exist. "
					f"Please clean {output_dir} and reconvert."
				)

		return True, None

	def convert_model_directory(self, input_dir, output_dir=None, force=False):
		"""
		Convert all checkpoint files in a directory to BatchGen format.

		This method converts all .safetensors and .pt files in the input directory
		to the BatchGen format (.bin + .json metadata files) for peak SSD read performance.

		Args:
			input_dir: Directory containing .safetensors or .pt checkpoint files
			output_dir: Output directory for converted files.
			           If None, defaults to <input_dir>/converted_ckpt
			force: If True, reconvert even if output directory already exists
			       and contains valid converted files

		Returns:
			Path to the directory containing converted checkpoint files

		Raises:
			FileNotFoundError: If input_dir does not exist
			ValueError: If no checkpoint files found in input_dir
			RuntimeError: If validation fails and force=False
		"""
		# Validate input directory
		if not os.path.exists(input_dir):
			raise FileNotFoundError(f"Input directory {input_dir} does not exist.")

		if not os.path.isdir(input_dir):
			raise ValueError(f"{input_dir} is not a directory.")

		# Set default output directory
		if output_dir is None:
			output_dir = os.path.join(input_dir, "converted_ckpt")

		# Get list of checkpoint files
		file_list = self._get_checkpoint_files(input_dir)

		if not file_list:
			raise ValueError(f"No checkpoint files (.safetensors or .pt) found in {input_dir}")

		logging.info(f"Found {len(file_list)} checkpoint files to convert")

		# Check if already converted
		if os.path.exists(output_dir) and not force:
			is_valid, error_msg = self.validate_converted_directory(input_dir, output_dir)
			if is_valid:
				logging.info(f"Converted checkpoint files already exist and are valid in {output_dir}")
				return output_dir
			else:
				raise RuntimeError(
					f"Converted directory exists but validation failed: {error_msg}\n"
					f"Use force=True to reconvert, or manually clean {output_dir}"
				)

		# Create output directory
		os.makedirs(output_dir, exist_ok=True)
		logging.info(f"Converting {len(file_list)} checkpoint files to BatchGen format...")

		# Convert each file with progress
		try:
			from tqdm import tqdm
			file_iterator = tqdm(file_list, desc="Converting checkpoint files", smoothing=0)
		except ImportError:
			file_iterator = file_list
			logging.info("Install tqdm for progress bar: pip install tqdm")

		for file_path in file_iterator:
			logging.debug(f"Converting {file_path} to {output_dir}")
			self.convert(file_path, output_dir)

		logging.info(f"Conversion complete. Output directory: {output_dir}")
		return output_dir

	

		



			
		




			



		




