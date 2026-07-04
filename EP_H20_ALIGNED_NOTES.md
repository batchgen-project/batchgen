# H20 aligned EP comparison

## Objective

Re-run the H20 EP benchmark in a **native** environment aligned as closely as possible to gala2's local `sm120` stack, so the NVLink-vs-PCIe comparison is not confounded by the earlier mismatched Docker image.

## Environment match table

| Item | gala2 local target | TencentNode0 H20 actual | Notes |
|---|---:|---:|---|
| GPU | RTX PRO 6000 Blackwell Server Edition (`sm_120`) | NVIDIA H20 (`sm_90`) | Different GPU generation/topology by design |
| Interconnect | PCIe baseline | NV18 NVLink mesh | This is the hardware variable of interest |
| Driver | n/a in this note | `550.144.03` | Exposes CUDA 12.4 runtime on node |
| nvcc | n/a in this note | `/usr/local/cuda-12.8/bin/nvcc` `12.8.93` | Toolkit present, but driver still gates runtime compatibility |
| torch | `2.12.0+cu130` | attempted: `2.12.0+cu130` -> **failed**; used `2.10.0+cu128` | `2.12.0+cu130` cannot initialize CUDA on this H20 host (`driver too old`, found `12040`) |
| triton | `3.7.0` | `3.7.0` | Matched |
| NCCL | `2.29.7` | `2.27.5` | Comes from the nearest working torch wheel |
| Verification command | `python -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.nccl.version())"` | same | gala2: `2.12.0+cu130 3.7.0 (2, 29, 7)`; H20: `2.10.0+cu128 3.7.0 (2, 27, 5)` |
| Install source | n/a | Tencent mirror only | `https://mirrors.cloud.tencent.com/pypi/simple/` |

## What was run

- Native venv on `TencentNode0`: `/data3/leyangxue/venvs/batchgen_gala2_align`
- Synced code to: `/data3/leyangxue/gmoe`
- Generated fixtures on H20 for:
  - decode `B={8,32,64,128,256}`
  - prefill `M={512,2048}`
- Ran real `torchrun --standalone --nproc_per_node=4` NCCL jobs for both:
  - `all_gather` + `all_reduce`
  - `all_to_all`
- All measured cells passed correctness gate (`max_rel_diff=0`, `recall=1.0`)

## Result files

- gala2 baseline: `benchmarks/results/grouped_moe/sm_120/ep_collective_compare_v4_flash.jsonl`
- H20 aligned run: `benchmarks/results/grouped_moe/h20_sm90/ep_aligned_comparison.jsonl`

## Comparable V4-Flash results (`median_us`)

| Phase | Size | gala2 all_gather | gala2 all_to_all | gala2 winner | H20 all_gather | H20 all_to_all | H20 winner |
|---|---:|---:|---:|---|---:|---:|---|
| decode | 8 | 1105.195 | 827.087 | all_to_all | 1042.779 | 1088.666 | all_gather |
| decode | 32 | 896.862 | 857.010 | all_to_all | 1434.703 | 1449.743 | all_gather |
| decode | 64 | 944.588 | 1010.582 | all_gather | 1610.540 | 2616.620 | all_gather |
| decode | 128 | 2481.456 | 4284.596 | all_gather | 1892.841 | 3449.863 | all_gather |
| decode | 256 | 1529.401 | 2126.448 | all_gather | 1969.646 | 2425.553 | all_gather |
| prefill | 512 | 1668.171 | 3605.165 | all_gather | 1513.131 | 2541.926 | all_gather |
| prefill | 2048 | 5287.723 | 5856.596 | all_gather | 4639.548 | 4905.827 | all_gather |

## `all_to_all / all_gather` ratio

| Phase | Size | gala2 ratio | H20 ratio |
|---|---:|---:|---:|
| decode | 8 | 0.748 | 1.044 |
| decode | 32 | 0.956 | 1.010 |
| decode | 64 | 1.070 | 1.625 |
| decode | 128 | 1.727 | 1.823 |
| decode | 256 | 1.390 | 1.231 |
| prefill | 512 | 2.161 | 1.680 |
| prefill | 2048 | 1.108 | 1.057 |

## Authoritative verdict

With the **closest working native alignment** that the H20 driver allows, **NVLink does not make `all_to_all` beat `all_gather`** for this ws=4 V4-Flash EP benchmark.

More specifically:

- On gala2, `all_to_all` won only the two smallest decode points (`B=8,32`).
- On the aligned H20 native run, those two small-batch wins **disappeared**; `all_gather` won **all 7 comparable cells**.
- For larger decode (`B>=64`) and both prefill points, the ranking remains `all_gather <= all_to_all`, often by a wide margin.

So the headline answer is:

> **After removing the mismatched Docker image and rerunning natively, NVLink does not improve the EP verdict in favor of `all_to_all`; if anything, the small-batch `all_to_all` advantage seen on gala2 PCIe disappears on H20 NVLink.**

## Caveat

This is the strongest honest conclusion available **without changing the H20 driver**. It is still not a perfect apples-to-apples software match, because:

- gala2 runs `torch 2.12.0+cu130` + `NCCL 2.29.7`
- H20 can only run `torch 2.10.0+cu128` + `NCCL 2.27.5` under the current driver

Therefore this run is **much cleaner than the earlier Docker comparison**, but it is still a **closest-working alignment**, not an exact software clone.
