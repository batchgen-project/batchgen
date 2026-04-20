import json
import safetensors
from safetensors.torch import load_file
import torch
import os
import logging
import shutil
import ctypes

KNOWN_TOKENIZER_ASSET_FILES = (
	"tokenizer.json",
	"tokenizer_config.json",
	"tiktoken.model",
	"chat_template.jinja",
)

MODEL_NAME_PATTERNS = (
	("moonshotai/Kimi-K2.5", "kimi_k25"),
	("Kimi-K2.5", "kimi_k25"),
	("openai/gpt-oss-120b", "gpt_oss"),
	("gpt-oss", "gpt_oss"),
	("THUDM/GLM-5", "glm5"),
	("GLM-5-FP8", "glm5"),
	("GLM-5", "glm5"),
	("MiniMaxAI/MiniMax-M2.5", "minimax_m25"),
	("MiniMax-M2.5", "minimax_m25"),
	("DeepSeek-R1", "deepseek"),
	("DeepSeek-V3", "deepseek"),
	("DeepSeek-V2-Lite", "deepseek"),
	("DeepSeek-V2", "deepseek"),
)

MODEL_TYPE_ALIASES = {
	"gpt_oss": "gpt_oss",
	"deepseek_v3": "deepseek",
	"deepseek_v2": "deepseek",
	"minimax_m25": "minimax_m25",
	"kimi_k25": "kimi_k25",
}

ARCHITECTURE_PATTERNS = (
	("GptOss", "gpt_oss"),
	("DeepseekV3", "deepseek"),
	("DeepseekV2", "deepseek"),
	("MiniMaxM2", "minimax_m25"),
	("GLM", "glm5"),
	("ChatGLM", "glm5"),
)

REQUIRED_TOKENIZER_ASSETS_BY_MODEL = {
	"deepseek": ("tokenizer.json",),
	"glm5": ("tokenizer.json",),
	"minimax_m25": ("tokenizer.json",),
	"kimi_k25": ("tiktoken.model", "tokenizer_config.json", "chat_template.jinja"),
	"gpt_oss": ("chat_template.jinja",),
}


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
		self._copied_tokenizer_assets = set()

	def _get_tokenizer_asset_files(self, source_dir):
		"""Return tokenizer asset files present in the source checkpoint dir."""
		asset_files = []
		for file_name in KNOWN_TOKENIZER_ASSET_FILES:
			if os.path.exists(os.path.join(source_dir, file_name)):
				asset_files.append(file_name)
		return asset_files

	def _detect_model_family(self, input_dir, model_identifier=None):
		"""Infer model family for tokenizer asset validation."""
		if model_identifier:
			for pattern, model_family in MODEL_NAME_PATTERNS:
				if pattern in str(model_identifier):
					return model_family

		config_path = os.path.join(input_dir, "config.json")
		if os.path.exists(config_path):
			try:
				with open(config_path, "r") as f:
					config = json.load(f)
			except Exception:
				config = {}
			model_type = config.get("model_type")
			if model_type in MODEL_TYPE_ALIASES:
				return MODEL_TYPE_ALIASES[model_type]
			for arch in config.get("architectures", []):
				for pattern, model_family in ARCHITECTURE_PATTERNS:
					if pattern in arch:
						return model_family

		for pattern, model_family in MODEL_NAME_PATTERNS:
			if pattern in str(input_dir):
				return model_family

		return None

	def _get_required_tokenizer_asset_files(self, input_dir, model_identifier=None):
		"""Return required tokenizer assets for the detected model family."""
		model_family = self._detect_model_family(input_dir, model_identifier=model_identifier)
		if model_family is None:
			return []
		return list(REQUIRED_TOKENIZER_ASSETS_BY_MODEL.get(model_family, ()))

	def _copy_tokenizer_assets(self, source_dir, output_dir, model_identifier=None):
		"""Copy tokenizer assets from source checkpoint dir into converted output dir."""
		copy_key = (os.path.abspath(source_dir), os.path.abspath(output_dir))
		if copy_key in self._copied_tokenizer_assets:
			return
		self._copied_tokenizer_assets.add(copy_key)

		present_assets = set(self._get_tokenizer_asset_files(source_dir))
		for file_name in present_assets:
			source_file = os.path.join(source_dir, file_name)
			target_file = os.path.join(output_dir, file_name)
			shutil.copyfile(source_file, target_file)
			logging.info(f"Copied {file_name} to converted checkpoint dir: {target_file}")

		required_assets = set(self._get_required_tokenizer_asset_files(source_dir, model_identifier=model_identifier))
		missing_required_assets = sorted(required_assets - present_assets)
		if missing_required_assets:
			logging.warning(
				f"Missing required tokenizer assets for source checkpoint directory {source_dir}: {missing_required_assets}. "
				"Conversion will continue, but runtime tokenizer loading may fail."
			)

	def _backfill_missing_tokenizer_assets(self, source_dir, output_dir):
		"""Copy missing tokenizer assets into an existing converted checkpoint dir."""
		copied_files = []
		for file_name in self._get_tokenizer_asset_files(source_dir):
			source_file = os.path.join(source_dir, file_name)
			target_file = os.path.join(output_dir, file_name)
			if os.path.exists(target_file):
				continue

			shutil.copyfile(source_file, target_file)
			copied_files.append(file_name)

		if copied_files:
			logging.info(
				f"Backfilled tokenizer assets into existing converted checkpoint dir {output_dir}: "
				f"{copied_files}"
			)

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
		
	def _apply_marlin_repack(self, ckpt):
		"""Replace INT4 expert weights with Marlin tile layout IN-PLACE using GPU kernel.

		Finds paired weight_packed + weight_scale tensors for routed expert
		projections (gate/up/down) and REPLACES them with Marlin format.
		Same tensor names, different data layout. Same total bytes.
		Uses fused CUDA kernel (~52 us/projection).

		Returns: modified ckpt dict with weight_packed/weight_scale replaced by Marlin layout.
		"""
		from batchgen.moe.marlin_transform import raw_to_marlin_fused_gpu
		import re

		# Pattern: *.mlp.experts.*.{gate,up,down}_proj.weight_packed
		packed_pattern = re.compile(
			r'(.+\.mlp\.experts\.\d+\.(gate|up|down)_proj)\.weight_packed$')

		device = "cuda" if torch.cuda.is_available() else None
		if device is None:
			logging.warning("[ckpt_converter] No GPU available, skipping Marlin repack")
			return ckpt

		count = 0
		for name in list(ckpt.keys()):
			m = packed_pattern.match(name)
			if not m:
				continue
			prefix = m.group(1)
			scale_name = f"{prefix}.weight_scale"
			if scale_name not in ckpt:
				continue

			packed = ckpt[name]       # [N, K//8] int32 or uint8
			scale = ckpt[scale_name]  # [N, K//32] bf16

			# Convert uint8 → int32 if needed
			if packed.dtype == torch.uint8:
				N_dim = scale.shape[0]
				K_dim = scale.shape[1] * 32
				packed = packed.view(N_dim, K_dim // 8, 4).contiguous().view(torch.int32).squeeze(-1)

			if packed.dtype != torch.int32:
				continue

			N = packed.shape[0]
			K = packed.shape[1] * 8

			# GPU transform: H2D → kernel → D2H
			packed_gpu = packed.to(device)
			scale_gpu = scale.to(device=device, dtype=torch.bfloat16)
			marlin_qw, marlin_s = raw_to_marlin_fused_gpu(packed_gpu, scale_gpu, K, N)
			torch.cuda.synchronize()

			# REPLACE in-place: same tensor names, Marlin layout
			ckpt[name] = marlin_qw.cpu()
			ckpt[scale_name] = marlin_s.cpu()
			count += 1

		if count > 0:
			logging.info(f"[ckpt_converter] Marlin GPU repack: {count} projections replaced in-place")
		return ckpt

	def convert(self, ckpt_path, output_dir, marlin=False, model_identifier=None):
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

		# Optional: repack INT4 expert weights to Marlin tile layout
		if marlin:
			ckpt = self._apply_marlin_repack(ckpt)

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

		self._copy_tokenizer_assets(os.path.dirname(os.path.abspath(ckpt_path)), output_dir, model_identifier=model_identifier)

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

	def _get_expected_output_files(self, input_dir):
		"""Return the allowed set of files in the converted output dir."""
		expected_files = set()
		for src_file in self._get_checkpoint_files(input_dir):
			file_name = os.path.basename(src_file)
			expected_files.add(
				file_name.replace(".safetensors", ".json").replace(".pt", ".json")
			)
			expected_files.add(
				file_name.replace(".safetensors", ".bin").replace(".pt", ".bin")
			)

		expected_files.update(KNOWN_TOKENIZER_ASSET_FILES)

		return expected_files

	def validate_converted_directory(self, input_dir, output_dir, model_identifier=None):
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

		if not os.path.isdir(output_dir):
			return False, f"Output directory {output_dir} does not exist or is not a directory."

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

		required_assets = self._get_required_tokenizer_asset_files(
			input_dir, model_identifier=model_identifier
		)
		for file_name in required_assets:
			output_file = os.path.join(output_dir, file_name)
			if not os.path.exists(output_file):
				return False, (
					f"Required tokenizer asset {output_file} does not exist for model validation. "
					f"Please clean {output_dir} and reconvert."
				)

		expected_files = self._get_expected_output_files(input_dir)
		actual_files = {entry.name for entry in os.scandir(output_dir)}
		unexpected_files = sorted(actual_files - expected_files)
		if unexpected_files:
			return False, (
				f"Unexpected files or directories in {output_dir}: {unexpected_files}. "
				f"Please clean {output_dir} and reconvert."
			)

		return True, None

	def convert_model_directory(self, input_dir, output_dir=None, force=False, marlin=False, model_identifier=None):
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
			self._backfill_missing_tokenizer_assets(input_dir, output_dir)
			is_valid, error_msg = self.validate_converted_directory(input_dir, output_dir, model_identifier=model_identifier)
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
			self.convert(file_path, output_dir, marlin=marlin, model_identifier=model_identifier)

		logging.info(f"Conversion complete. Output directory: {output_dir}"
		             f"{' (with Marlin repack)' if marlin else ''}")
		return output_dir

	

		



			
		




			



		

