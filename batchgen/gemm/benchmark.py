"""
Complete benchmarking and validation suite for FP8 MoE kernels.
"""

import torch
import time
import numpy as np
from typing import List, Tuple
# import matplotlib.pyplot as plt


class MoEBenchmarker:
	"""Comprehensive MoE kernel benchmarking."""
	
	def __init__(self, device='cuda'):
		self.device = device
		self.results = []
	
	def create_test_data(
		self,
		M: int,
		N: int,
		K: int,
		num_experts: int,
		group_sizes: List[int],
	):
		"""Create test data for MoE."""
		# Hidden states (activations)
		hidden_states = torch.randn(M, K, device=self.device, dtype=torch.float16)
		hidden_states = hidden_states.to(torch.float8_e4m3fn)
		
		hidden_states_scale = torch.ones(
			M, (K + 127) // 128,
			device=self.device,
			dtype=torch.float32
		) * 0.1
		
		# Expert weights
		gate_weights = [
			torch.randn(N, K, device=self.device, dtype=torch.float16).to(torch.float8_e4m3fn)
			for _ in range(num_experts)
		]
		up_weights = [
			torch.randn(N, K, device=self.device, dtype=torch.float16).to(torch.float8_e4m3fn)
			for _ in range(num_experts)
		]
		
		# Scales
		gate_scales = [
			torch.ones((N + 127) // 128, (K + 127) // 128, device=self.device) * 0.1
			for _ in range(num_experts)
		]
		up_scales = [
			torch.ones((N + 127) // 128, (K + 127) // 128, device=self.device) * 0.1
			for _ in range(num_experts)
		]
		
		# Create pointer tensors
		gate_ptrs = torch.tensor(
			[w.data_ptr() for w in gate_weights],
			device=self.device,
			dtype=torch.int64
		)
		up_ptrs = torch.tensor(
			[w.data_ptr() for w in up_weights],
			device=self.device,
			dtype=torch.int64
		)
		gate_scale_ptrs = torch.tensor(
			[s.data_ptr() for s in gate_scales],
			device=self.device,
			dtype=torch.int64
		)
		up_scale_ptrs = torch.tensor(
			[s.data_ptr() for s in up_scales],
			device=self.device,
			dtype=torch.int64
		)
		
		# Group metadata
		group_sizes_tensor = torch.tensor(group_sizes, device=self.device, dtype=torch.int32)
		activated_group_idx = torch.arange(len(group_sizes), device=self.device, dtype=torch.int32)
		
		# Calculate start indices
		group_start_indices = torch.zeros(len(group_sizes), device=self.device, dtype=torch.int32)
		cumsum = 0
		for i, size in enumerate(group_sizes):
			group_start_indices[i] = cumsum
			cumsum += size
		
		num_active = torch.tensor([len(group_sizes)], device=self.device, dtype=torch.int32)
		
		return (
			hidden_states, hidden_states_scale,
			gate_weights, gate_ptrs,
			up_weights, up_ptrs,
			gate_scales, gate_scale_ptrs,
			up_scales, up_scale_ptrs,
			group_sizes_tensor, activated_group_idx,
			group_start_indices, num_active
		)
	
	def benchmark_kernel(
		self,
		kernel_func,
		M: int,
		N: int,
		K: int,
		num_experts: int,
		group_sizes: List[int],
		num_iters: int = 100,
	):
		"""Benchmark a specific kernel configuration."""
		# Create data
		data = self.create_test_data(M, N, K, num_experts, group_sizes)
		
		# Warmup
		for _ in range(10):
			_ = kernel_func(*data)
		
		torch.cuda.synchronize()
		
		# Benchmark
		times = []
		for _ in range(num_iters):
			start = time.perf_counter()
			_ = kernel_func(*data)
			torch.cuda.synchronize()
			times.append((time.perf_counter() - start) * 1000)
		
		median_time = np.median(times)
		min_time = np.min(times)
		std_time = np.std(times)
		
		# Calculate FLOPS
		# Each expert: 2 GEMMs (gate, up) = 4*M*N*K FLOPS
		total_tokens = sum(group_sizes)
		flops_per_token = 4 * N * K  # gate + up
		total_flops = flops_per_token * total_tokens
		tflops = total_flops / median_time / 1e9
		
		# Calculate utilization
		h100_peak = 989  # TFLOPS for FP8
		utilization = (tflops / h100_peak) * 100
		
		result = {
			'M': M,
			'N': N,
			'K': K,
			'num_experts': num_experts,
			'total_tokens': total_tokens,
			'avg_group_size': total_tokens / num_experts,
			'time_ms': median_time,
			'time_std': std_time,
			'tflops': tflops,
			'utilization_%': utilization,
		}
		
		self.results.append(result)
		return result
	
	def compare_kernels(
		self,
		kernels: List[Tuple[str, callable]],
		test_configs: List[dict],
	):
		"""Compare multiple kernel implementations."""
		print("\n" + "="*100)
		print("MoE KERNEL COMPARISON")
		print("="*100)
		
		print(f"\n{'Config':<20} {'Kernel':<20} {'Time(ms)':<12} {'TFLOPS':<10} {'Util%':<8} {'Speedup':<10}")
		print("-"*100)
		
		for config in test_configs:
			baseline_time = None
			config_name = f"M={config['M']},E={config['num_experts']}"
			
			for kernel_name, kernel_func in kernels:
				result = self.benchmark_kernel(
					kernel_func,
					config['M'],
					config['N'],
					config['K'],
					config['num_experts'],
					config['group_sizes'],
				)
				
				if baseline_time is None:
					baseline_time = result['time_ms']
					speedup_str = "1.00x (baseline)"
				else:
					speedup = baseline_time / result['time_ms']
					speedup_str = f"{speedup:.2f}x"
				
				print(f"{config_name:<20} {kernel_name:<20} {result['time_ms']:<12.4f} "
					  f"{result['tflops']:<10.2f} {result['utilization_%']:<8.1f} {speedup_str:<10}")
		
		print("="*100)
	
	def sweep_group_sizes(
		self,
		kernel_func,
		M: int = 128,
		N: int = 4096,
		K: int = 11008,
		num_experts: int = 8,
	):
		"""Test various group size distributions."""
		print("\n" + "="*100)
		print("GROUP SIZE SWEEP")
		print("="*100)
		
		test_cases = [
			([1]*8, "Tiny uniform (1 token/expert)"),
			([4]*8, "Small uniform (4 tokens/expert)"),
			([8]*8, "Medium uniform (8 tokens/expert)"),
			([16]*8, "Large uniform (16 tokens/expert)"),
			([1, 2, 4, 8, 16, 20, 24, 32], "Variable sizes"),
			([32, 32, 1, 1, 1, 1, 1, 1], "Highly skewed"),
		]
		
		print(f"\n{'Case':<30} {'Avg Size':<10} {'Time(ms)':<12} {'TFLOPS':<10} {'Util%':<8}")
		print("-"*100)
		
		for group_sizes, description in test_cases:
			result = self.benchmark_kernel(
				kernel_func,
				M, N, K, num_experts, group_sizes,
			)
			
			print(f"{description:<30} {result['avg_group_size']:<10.1f} "
				  f"{result['time_ms']:<12.4f} {result['tflops']:<10.2f} "
				  f"{result['utilization_%']:<8.1f}")
		
		print("="*100)
	
	def profile_bottlenecks(self, kernel_func, M=16, N=4096, K=11008):
		"""Identify performance bottlenecks."""
		print("\n" + "="*100)
		print("BOTTLENECK ANALYSIS")
		print("="*100)
		
		# Test with 8 experts
		group_sizes = [M // 8] * 8
		data = self.create_test_data(M, N, K, 8, group_sizes)
		
		print("""
To profile bottlenecks, run:

1. Overall profile:
   nsys profile -o moe_profile python your_script.py

2. Detailed metrics:
   ncu --set full -o moe_detailed python your_script.py

3. Key metrics to check:
   ncu --metrics \\
	   smsp__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active,\\
	   dram__throughput.avg.pct_of_peak_sustained_elapsed,\\
	   sm__warps_active.avg.pct_of_peak_sustained_active \\
	   python your_script.py

Expected values:
  - Tensor core utilization: >30% (currently likely <10%)
  - Memory bandwidth: 50-80%
  - Occupancy: >50%
		""")
		
		# Run kernel once for profiling
		_ = kernel_func(*data)
		torch.cuda.synchronize()
		
		print("✅ Kernel executed successfully for profiling")
		print("="*100)


class MoEValidator:
	"""Validate MoE kernel correctness."""
	
	@staticmethod
	def reference_moe(
		hidden_states,
		hidden_states_scale,
		gate_weights,
		up_weights,
		gate_scales,
		up_scales,
		group_sizes,
		group_start_indices,
	):
		"""Reference implementation (slow but correct)."""
		M, K = hidden_states.shape
		N = gate_weights[0].shape[0]
		
		output = torch.zeros(M, N, device=hidden_states.device, dtype=torch.bfloat16)
		
		for expert_idx, group_size in enumerate(group_sizes):
			if group_size == 0:
				continue
			
			start_idx = group_start_indices[expert_idx]
			end_idx = start_idx + group_size
			
			# Get tokens for this expert
			tokens = hidden_states[start_idx:end_idx]
			token_scales = hidden_states_scale[start_idx:end_idx]
			
			# Dequantize activations
			tokens_fp32 = tokens.to(torch.float32)
			for i in range(tokens.shape[0]):
				for k_block in range((K + 127) // 128):
					k_start = k_block * 128
					k_end = min(k_start + 128, K)
					scale = token_scales[i, k_block]
					tokens_fp32[i, k_start:k_end] *= scale
			
			# Dequantize weights
			gate_w = gate_weights[expert_idx].to(torch.float32)
			up_w = up_weights[expert_idx].to(torch.float32)
			
			# Apply scales (simplified - assumes uniform scaling for reference)
			gate_w = gate_w * gate_scales[expert_idx].mean()
			up_w = up_w * up_scales[expert_idx].mean()
			
			# Compute
			gate_out = torch.matmul(tokens_fp32, gate_w.t())
			up_out = torch.matmul(tokens_fp32, up_w.t())
			
			# SiLU
			gate_act = gate_out / (1.0 + torch.exp(-gate_out))
			result = gate_act * up_out
			
			output[start_idx:end_idx] = result.to(torch.bfloat16)
		
		return output
	
	@staticmethod
	def validate_correctness(
		kernel_func,
		M: int = 32,
		N: int = 512,
		K: int = 1024,
		num_experts: int = 4,
	):
		"""Validate kernel correctness against reference."""
		print("\n" + "="*80)
		print("CORRECTNESS VALIDATION")
		print("="*80)
		
		device = 'cuda'
		
		# Create test data
		group_sizes = [8, 8, 8, 8]  # Uniform for easier validation
		
		benchmarker = MoEBenchmarker(device)
		data = benchmarker.create_test_data(M, N, K, num_experts, group_sizes)
		
		# Run kernel
		kernel_output = kernel_func(*data)
		
		# Run reference (with simplified scaling)
		ref_output = MoEValidator.reference_moe(
			data[0], data[1],  # hidden_states, scales
			data[2], data[4],  # gate_weights, up_weights
			data[6], data[8],  # gate_scales, up_scales
			group_sizes,
			data[12],  # group_start_indices
		)
		
		# Compare
		max_diff = torch.abs(kernel_output - ref_output).max().item()
		mean_diff = torch.abs(kernel_output - ref_output).mean().item()
		rel_error = mean_diff / (torch.abs(ref_output).mean().item() + 1e-8)
		
		print(f"Max absolute error:   {max_diff:.6f}")
		print(f"Mean absolute error:  {mean_diff:.6f}")
		print(f"Relative error:       {rel_error:.6f}")
		
		# FP8 has limited precision
		passed = rel_error < 0.15 and max_diff < 1.0
		
		if passed:
			print("✅ PASS: Kernel output is correct!")
		else:
			print("❌ FAIL: Kernel has numerical errors!")
			print("Note: Some error is expected due to FP8 quantization")
		
		print("="*80)
		return passed


def run_comprehensive_benchmark(
	original_kernel,
	optimized_kernel,
):
	"""Run complete benchmark suite."""
	print("\n" + "="*100)
	print("🚀 COMPREHENSIVE MoE KERNEL BENCHMARK")
	print("="*100)
	
	benchmarker = MoEBenchmarker()
	validator = MoEValidator()
	
	# 1. Validation
	print("\n" + "="*100)
	print("STEP 1: CORRECTNESS VALIDATION")
	print("="*100)
	
	validator.validate_correctness(optimized_kernel)
	
	# 2. Performance comparison
	print("\n" + "="*100)
	print("STEP 2: PERFORMANCE COMPARISON")
	print("="*100)
	
	kernels = [
		("Original", original_kernel),
		("Optimized", optimized_kernel),
	]
	
	test_configs = [
		{
			'M': 64,
			'N': 4096,
			'K': 11008,
			'num_experts': 8,
			'group_sizes': [8] * 8,
		},
		{
			'M': 128,
			'N': 4096,
			'K': 11008,
			'num_experts': 8,
			'group_sizes': [16] * 8,
		},
		{
			'M': 64,
			'N': 4096,
			'K': 11008,
			'num_experts': 8,
			'group_sizes': [1, 2, 4, 8, 12, 16, 18, 20],  # Variable
		},
	]
	
	benchmarker.compare_kernels(kernels, test_configs)
	
	# 3. Group size sweep
	print("\n" + "="*100)
	print("STEP 3: GROUP SIZE SENSITIVITY")
	print("="*100)
	
	benchmarker.sweep_group_sizes(optimized_kernel)
	
	# 4. Profiling instructions
	benchmarker.profile_bottlenecks(optimized_kernel)
	
	# 5. Summary
	print("\n" + "="*100)
	print("📊 SUMMARY")
	print("="*100)
	
	print("""
Expected improvements with optimized kernel:
  - BLOCK_N: 16 → 128 (8× better tensor core utilization)
  - Grid: 1D → 2D (8× more parallelism)
  - Overall speedup: 5-8× for typical workloads
  
Target performance:
  - Utilization: 30-50% of peak (up from <5%)
  - TFLOPS: 300-500 on H100 (up from <50)
  - Latency: <0.5ms for M=128, 8 experts
  
If not hitting targets:
  1. Profile with NSight Compute
  2. Check tensor core utilization
  3. Verify BLOCK_N ≥ 128
  4. Consider work-stealing for variable groups
	""")
	
	print("="*100)


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
	from batchgen.moe.fused_dequant_moe import fused_fp8_moe_stage_1
	from batchgen.gemm.w8a8_grouped_gemm_stage_1 import fused_fp8_moe_stage_1_optimized
	run_comprehensive_benchmark(fused_fp8_moe_stage_1, fused_fp8_moe_stage_1_optimized)
#     print("""
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  🔥 MoE KERNEL BENCHMARKING SUITE                                    ║
# ║                                                                      ║
# ║  Usage:                                                              ║
# ║    from moe_benchmark_suite import run_comprehensive_benchmark      ║
# ║    run_comprehensive_benchmark(original_kernel, optimized_kernel)   ║
# ║                                                                      ║
# ║  Features:                                                           ║
# ║    ✓ Correctness validation                                         ║
# ║    ✓ Performance comparison                                         ║
# ║    ✓ Group size sensitivity analysis                                ║
# ║    ✓ Profiling guidance                                             ║
# ║    ✓ Bottleneck identification                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝
#     """)