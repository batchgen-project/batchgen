"""Debug: compare CPU vs fused GPU transform on small tensors."""
import torch
from batchgen.moe.marlin_weight_prep import repack_int4_to_marlin_gs32, get_weight_perm, GPTQ_MARLIN_TILE

K, N = 7168, 2048  # production shapes
device = "cuda"

# Create simple sequential nibbles
q_raw = torch.arange(N * K, device=device, dtype=torch.int32) % 16
q_raw = q_raw.view(N, K)

# Pack as K2.5 int32 [N, K//8]
raw_packed = torch.zeros(N, K // 8, dtype=torch.int32, device=device)
for i in range(8):
    raw_packed |= (q_raw[:, i::8] & 0xF) << (i * 4)

scales = torch.ones(N, K // 32, dtype=torch.bfloat16, device=device)

# Forward: K2.5 → Marlin
marlin_qw, marlin_s = repack_int4_to_marlin_gs32(raw_packed, scales, K, N)
print(f"marlin_qw shape: {marlin_qw.shape}")  # [K//16, N*2] = [4, 64]
print(f"raw_packed shape: {raw_packed.shape}")  # [N, K//8] = [32, 8]

# CPU reference
from batchgen.moe.marlin_transform import marlin_to_wgmma_cpu, marlin_to_wgmma_fused_gpu
cpu_out, cpu_s = marlin_to_wgmma_cpu(marlin_qw, marlin_s, K, N)

# GPU fused
gpu_out, gpu_s = marlin_to_wgmma_fused_gpu(marlin_qw, marlin_s, K, N)
torch.cuda.synchronize()

print(f"\nraw_packed[0,:4] = {raw_packed[0,:4].tolist()}")
print(f"cpu_out[0,:4]    = {cpu_out[0,:4].tolist()}")
print(f"gpu_out[0,:4]    = {gpu_out[0,:4].tolist()}")
print(f"CPU match: {torch.equal(raw_packed, cpu_out)}")
print(f"GPU match: {torch.equal(raw_packed, gpu_out)}")

# Unpack first int32 to nibbles for comparison
def unpack_nibbles(val):
    return [(val >> (i*4)) & 0xF for i in range(8)]

print(f"\nFirst int32 nibbles (n=0, k=0..7):")
print(f"  raw:  {unpack_nibbles(raw_packed[0,0].item())}")
print(f"  cpu:  {unpack_nibbles(cpu_out[0,0].item())}")
print(f"  gpu:  {unpack_nibbles(gpu_out[0,0].item())}")

# Check what the GPU kernel reads from marlin_qw
perm = get_weight_perm(4).numpy()
print(f"\nGPU kernel address trace for output (n=0, k=0..7):")
input_stride = N * 2  # marlin row stride
for k in range(8):
    k_tile = k // 16
    k_in_tile = k % 16
    n_tile = 0 // 16
    n_in_tile = 0 % 16
    raw_col = n_tile * 256 + k_in_tile * 16 + n_in_tile
    chunk = raw_col // 1024
    pos = raw_col % 1024
    marlin_col = chunk * 1024 + int(perm[pos])
    marlin_int32 = marlin_col // 8
    marlin_shift = (marlin_col % 8) * 4
    val = (marlin_qw[k_tile, marlin_int32].item() >> marlin_shift) & 0xF
    print(f"  k={k}: raw_col={raw_col}, perm[{pos}]={perm[pos]}, marlin_col={marlin_col}, "
          f"int32_idx={marlin_int32}, shift={marlin_shift}, val={val}")
