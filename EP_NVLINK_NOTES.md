# EP NVLink comparison on H20

## Run setup
- Host: `TencentNode0` (`node0`), 8x NVIDIA H20
- Topology: full `NV18` GPU↔GPU mesh (`nvidia-smi topo -m`)
- GPUs used: `0,1,2,3`
- Runtime: `docker run --gpus all lmsysorg/sglang:dev-cu13`
- Why Docker: remote `v4venv` was missing both `numpy` and `triton.tools.mxfp`; benchmark ran cleanly in the CUDA 13 container without code changes
- Benchmark: `benchmarks/grouped_moe_probes/ep_collective_compare.py`
- Grid: V4-Flash decode `B={8,32,64,128,256}` and prefill `M={512,2048}`
- Result file: `benchmarks/results/grouped_moe/h20_sm90/ep_nvlink_comparison.jsonl`

## Short answer
No. On this EP collective harness, NVLink did **not** turn BatchGen's owner-grouped `all_to_all` path into a universal end-to-end win.

Using `median_us` (= dispatch + local_gemm + combine kernel time), H20 `all_to_all` wins only **2/7** requested cells:
- decode 128
- prefill 512

For the same 7-cell grid on gala2 PCIe, `all_to_all` wins **3/7** cells.

So the earlier “collectives dominate” story was **not just a PCIe artifact** in this harness.

## H20 per-cell comparison

| phase | size | all_gather dispatch | all_gather gemm | all_gather combine | all_gather total | all_to_all dispatch | all_to_all gemm | all_to_all combine | all_to_all total | winner |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| decode | 8 | 255.5 | 389.7 | 414.6 | 1028.6 | 613.8 | 386.2 | 147.8 | 1149.6 | all_gather |
| decode | 32 | 62.8 | 760.0 | 12.7 | 835.5 | 386.3 | 761.0 | 218.6 | 1519.6 | all_gather |
| decode | 64 | 443.9 | 1101.4 | 561.1 | 2259.6 | 601.1 | 1093.5 | 357.3 | 2495.5 | all_gather |
| decode | 128 | 470.1 | 1193.9 | 332.9 | 2785.8 | 488.9 | 1191.9 | 39.0 | 1744.8 | all_to_all |
| decode | 256 | 99.0 | 1254.6 | 189.9 | 2236.3 | 584.9 | 1259.5 | 433.6 | 2420.2 | all_gather |
| prefill | 512 | 703.7 | 1380.4 | 84.4 | 2170.4 | 188.1 | 1379.5 | 49.7 | 1629.5 | all_to_all |
| prefill | 2048 | 142.8 | 3874.7 | 476.1 | 4564.3 | 353.6 | 3872.9 | 868.1 | 5377.7 | all_gather |

## H20 vs gala2 (requested 7-cell grid)

### Winner count by total kernel time
- gala2 PCIe: `all_to_all` wins **3/7**
- H20 NVLink: `all_to_all` wins **2/7**

### What improved on H20
- Some collective-heavy cells improved materially, especially:
  - decode 128 `all_to_all`: total `4284.6 -> 1744.8 us`
  - prefill 512 `all_to_all`: total `3605.2 -> 1629.5 us`
  - prefill 2048 `all_gather`: total `5287.7 -> 4564.3 us`

### What did **not** happen
- H20 did **not** consistently reduce `all_to_all` dispatch enough to beat `all_gather`
- Small decode cells stayed unfavorable for `all_to_all` on H20:
  - decode 8: dispatch `613.8 us` vs `255.5 us`
  - decode 32: dispatch `386.3 us` vs `62.8 us`
  - decode 64: dispatch `601.1 us` vs `443.9 us`
  - decode 256: dispatch `584.9 us` vs `99.0 us`
- Local GEMM stayed essentially tied across collective patterns, so outcomes were still dominated by communication behavior

## Interpretation
- NVLink helps, but it is **not sufficient by itself** to restore an across-the-board EP win in this benchmark
- The owner-grouped `all_to_all` path still has meaningful dispatch/combine overhead on several cells even on NV18 H20
- The hypothesis “gala2 lost mainly because it was PCIe-only” is too weak; communication pattern costs remain shape-dependent on NVLink too

## Files
- H20 results: `benchmarks/results/grouped_moe/h20_sm90/ep_nvlink_comparison.jsonl`
- gala2 baseline: `benchmarks/results/grouped_moe/sm_120/ep_collective_compare_v4_flash.jsonl`
