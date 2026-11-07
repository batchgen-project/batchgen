import itertools
import json
import safetensors
from safetensors.torch import load_file
import torch
import os
import logging
import ctypes
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

#: Output .bin files are written through a buffer of this size, so the many
#: small per-tensor writes reach a network filesystem as large sequential
#: ones instead of thousands of tiny requests per shard.
WRITE_BUFFER_BYTES = 64 * 1024 * 1024


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
		# One repack lock PER GPU: parallel shard workers each repack on
		# their round-robin device, so all local GPUs work concurrently
		# while no two workers contend on one device (or its memory).
		self._repack_locks = {}
		self._locks_guard = threading.Lock()
		self._device_rr = itertools.count()

	def _next_repack_device(self):
		"""Round-robin CUDA device for one shard's repack; None without GPUs."""
		if not torch.cuda.is_available():
			return None
		count = torch.cuda.device_count()
		if count < 1:
			return None
		return f"cuda:{next(self._device_rr) % count}"

	def _repack_lock(self, device):
		with self._locks_guard:
			lock = self._repack_locks.get(device)
			if lock is None:
				lock = self._repack_locks[device] = threading.Lock()
			return lock

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
		elif hasattr(torch, "float8_e8m0fnu") and dtype == torch.float8_e8m0fnu:
			return "float8_e8m0fnu"
		elif hasattr(torch, "float4_e2m1fn_x2") and dtype == torch.float4_e2m1fn_x2:
			return "float4_e2m1fn_x2"
		elif dtype == torch.int64:
			return "int64"
		elif dtype == torch.int32:
			return "int32"
		elif dtype == torch.int8:
			return "int8"
		elif dtype == torch.uint8:
			return "uint8"
		else:
			raise ValueError(f"Unsupported dtype: {dtype}")
		
	def _apply_marlin_repack(self, ckpt, device=None):
		"""Replace INT4 expert weights with Marlin tile layout IN-PLACE using GPU kernel.

		Finds paired weight_packed + weight_scale tensors for routed expert
		projections (gate/up/down) and REPLACES them with Marlin format.
		Same tensor names, different data layout. Same total bytes.
		Uses fused CUDA kernel (~52 us/projection).

		Returns: modified ckpt dict with weight_packed/weight_scale replaced by Marlin layout.
		"""
		# NOTE: raw_to_marlin_fused_gpu (INT4 path) imported lazily below so the
		# K3 MXFP4 branch does not require the _C_marlin_transform kernel.
		import re

		# Pattern: *.mlp.experts.*.{gate,up,down}_proj.weight_packed
		packed_pattern = re.compile(
			r'(.+\.mlp\.experts\.\d+\.(gate|up|down)_proj)\.weight_packed$')
		# K3 routed experts: block_sparse_moe.experts.N.w{1,2,3} (MXFP4 uint8 E8M0).
		k3_pattern = re.compile(
			r'(.+\.block_sparse_moe\.experts\.\d+\.w[123])\.weight_packed$')

		if device is None:
			device = "cuda" if torch.cuda.is_available() else None
		if device is None:
			logging.warning("[ckpt_converter] No GPU available, skipping Marlin repack")
			return ckpt

		count = 0
		for name in list(ckpt.keys()):
			m = packed_pattern.match(name)
			mk = k3_pattern.match(name)
			if not m and not mk:
				continue
			prefix = (m or mk).group(1)
			scale_name = f"{prefix}.weight_scale"
			if scale_name not in ckpt:
				continue

			packed = ckpt[name]       # [N, K//8] int32 or uint8
			scale = ckpt[scale_name]  # [N, K//32] bf16

			# K3 MXFP4 (task #53, V1): uint8 E2M1 packed + uint8 E8M0 scale ->
			# marlin_qw int32 + marlin-order uint8 E8M0 scale; decode dequants
			# per-forward. Bit-exact split of repack_mxfp4_to_marlin_device.
			if mk and packed.dtype == torch.uint8 and scale.dtype == torch.uint8:
				from batchgen.models.moonshotai.kimi_linear.k3.mxfp4_expert import (
					repack_mxfp4_to_marlin_device,
				)
				N = packed.shape[0]
				K = packed.shape[1] * 2  # MXFP4 pack factor 2
				marlin_qw, marlin_s = repack_mxfp4_to_marlin_device(
					packed.to(device), scale.to(device), K, N, scale_bf16=False)
				torch.cuda.synchronize()
				ckpt[name] = marlin_qw.cpu()
				ckpt[scale_name] = marlin_s.cpu()
				count += 1
				continue

			# Refuse MXFP4. This path is uniform-INT4-only: it reinterprets the
			# uint8 buffer as packed uint4b8 below, and converts the SCALE
			# TENSOR VALUE to bfloat16 — an E8M0 exponent byte of 121 would
			# become 121.0 instead of 2**(121-127) = 0.015625. Both corruptions
			# are silent. Scoped to tensors this function would actually touch:
			# an unscoped check would reject every MXFP4 checkpoint the repack
			# never matches, and `batchgen/tools/convert_checkpoint.py:144`
			# passes marlin=True BY DEFAULT.
			if scale.dtype == torch.uint8:
				raise ValueError(
					f"{scale_name}: uint8 (E8M0) scale on a tensor the Marlin "
					"repack matches. Marlin handles uniform INT4 with BF16 "
					"scales only; MXFP4 needs an E2M1 nibble decode and E8M0 "
					"scale handling. Re-run with --no-marlin."
				)

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
			from batchgen.moe.marlin_transform import raw_to_marlin_fused_gpu
			marlin_qw, marlin_s = raw_to_marlin_fused_gpu(packed_gpu, scale_gpu, K, N)
			torch.cuda.synchronize()

			# REPLACE in-place: same tensor names, Marlin layout
			ckpt[name] = marlin_qw.cpu()
			ckpt[scale_name] = marlin_s.cpu()
			count += 1

		if count > 0:
			logging.info(f"[ckpt_converter] Marlin GPU repack: {count} projections replaced in-place")
		return ckpt

	def convert(self, ckpt_path, output_dir, marlin=False, repack_device=None):
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

		# Optional: repack INT4 expert weights to Marlin tile layout, on this
		# shard's device, serialized only against other shards on the SAME
		# device.
		if marlin:
			device = repack_device or self._next_repack_device()
			if device is None:
				ckpt = self._apply_marlin_repack(ckpt, None)
			else:
				with self._repack_lock(device):
					with torch.cuda.device(device):
						ckpt = self._apply_marlin_repack(ckpt, device)

		out_file_name = os.path.join(output_dir, os.path.basename(ckpt_path).replace(".safetensors", ".bin")).replace(".pt", ".bin")
		out_metadata_name = os.path.join(output_dir, os.path.basename(ckpt_path).replace(".safetensors", ".json").replace(".pt", ".json"))	
	
		metadata = {}
		metadata["file_name"] = os.path.basename(out_file_name)
		metadata["state_dict"] = {}
		total_byte_size = 0
		

		with open(out_file_name, "wb", buffering=WRITE_BUFFER_BYTES) as out_file:
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

	def convert_model_directory(self, input_dir, output_dir=None, force=False, marlin=False, max_workers=None, partition_index=0, partition_count=1):
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
			partition_index/partition_count: convert only every
			       partition_count-th file starting at partition_index (of the
			       sorted file list), so several nodes convert one checkpoint
			       concurrently into a shared output directory. Each converted
			       shard depends only on its source file, so partitions are
			       disjoint and complete. With partition_count > 1 the
			       existing-output shortcut and the final whole-directory
			       validation are skipped (other partitions may still be
			       running); run --validate once after every partition
			       finished.

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

		if partition_count < 1 or not 0 <= partition_index < partition_count:
			raise ValueError(
				f"partition_index/partition_count must satisfy 0 <= index < count; "
				f"got {partition_index}/{partition_count}"
			)

		# Get list of checkpoint files
		file_list = self._get_checkpoint_files(input_dir)

		if not file_list:
			raise ValueError(f"No checkpoint files (.safetensors or .pt) found in {input_dir}")

		if partition_count > 1:
			file_list = file_list[partition_index::partition_count]
			logging.info(
				f"Partition {partition_index}/{partition_count}: converting "
				f"{len(file_list)} of the checkpoint files"
			)
			if not file_list:
				return output_dir
		else:
			logging.info(f"Found {len(file_list)} checkpoint files to convert")

		# Check if already converted (whole-directory shortcut: only
		# meaningful when this call owns the whole conversion)
		if partition_count == 1 and os.path.exists(output_dir) and not force:
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

		# Convert shards in PARALLEL. Per-shard cost is network-filesystem
		# I/O (safetensors read + buffered .bin write), which overlaps across
		# threads, plus the GPU marlin repack, which runs on a round-robin
		# device with one lock per device, so every local GPU repacks
		# concurrently with no per-device contention or transient-HBM OOM.
		# A failed shard re-raises (fail-fast; no partial-output silent
		# success).
		if max_workers is None:
			workers = min(8, len(file_list))
			if marlin and torch.cuda.is_available():
				workers = min(max(workers, torch.cuda.device_count()), len(file_list))
		else:
			workers = int(max_workers)
		workers = max(1, workers)
		try:
			from tqdm import tqdm
			progress = tqdm(total=len(file_list), desc="Converting checkpoint files", smoothing=0)
		except ImportError:
			progress = None
			logging.info("Install tqdm for progress bar: pip install tqdm")

		logging.info(f"Converting {len(file_list)} shards with {workers} worker thread(s)")
		if workers == 1:
			for file_path in file_list:
				self.convert(file_path, output_dir, marlin=marlin)
				if progress is not None:
					progress.update(1)
		else:
			with ThreadPoolExecutor(max_workers=workers) as pool:
				futures = {pool.submit(self.convert, fp, output_dir, marlin): fp for fp in file_list}
				for fut in as_completed(futures):
					fut.result()  # re-raise any shard error (fail-fast)
					if progress is not None:
						progress.update(1)
		if progress is not None:
			progress.close()

		logging.info(f"Conversion complete. Output directory: {output_dir}"
		             f"{' (with Marlin repack)' if marlin else ''}")
		return output_dir

	

		



			
		




			



		



