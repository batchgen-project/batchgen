# RAGGED VALIDATION B200

## Outcome

Validated the ragged MoE decode kernel on datacenter Blackwell (`sm_100`, NVIDIA B200) with `profiling_tier=1` and a small V4-Flash decode sweep (`B={1,8,64,256}`).

- Ragged kernel: **passes numerics at all 4 points** (`max_rel_diff=0`, `recall=1.0`)
- Upstream `deep_gemm` was already installed on the B200 image
- DeepGEMM decode comparisons ran for both legs:
  - FP8 / UE8M0 masked grouped kernel
  - NVFP4 masked grouped kernel
- Both DeepGEMM legs were **numerically invalid** against the BF16 eager reference at all 4 points on this setup

Primary artifact:

- `benchmarks/results/grouped_moe/b200_sm100/ragged_vs_deepgemm.jsonl`

## Remote procedure followed

1. Read `/etc/vast-agents-guide.md`
2. Synced:
   - `batchgen/moe/`
   - `benchmarks/grouped_moe_probes/`
   - `benchmarks/shared/` (required dependency of the probe harness)
3. Checked environment with `/venv/main/bin/python3`
4. Ran V4-Flash **decode only** with `GROUPED_MOE_FORCE_PROFILING_TIER=1`
5. Synced results back locally

## Remote environment

- GPU: `NVIDIA B200`
- Arch: `sm_100`
- Python: `/venv/main`
- Torch: `2.12.1+cu130`
- DeepGEMM: `/venv/main/lib/python3.12/site-packages/deep_gemm/__init__.py`

## Results summary

| B | Ragged us | Ragged status | DeepGEMM FP8 us | FP8 status | DeepGEMM NVFP4 us | NVFP4 status |
|---|---:|---|---:|---|---:|---|
| 1 | 268.094 | ok | 2901.106 | INVALID | 8354.183 | INVALID |
| 8 | 834.296 | ok | 20381.538 | INVALID | 60393.069 | INVALID |
| 64 | 3153.269 | ok | 91902.363 | INVALID | 274799.050 | INVALID |
| 256 | 3998.928 | ok | 118927.253 | INVALID | 350850.472 | INVALID |

## Numeric validation details

Ragged path:

- `B=1`: pass, `max_rel_diff=0`, `recall=1.0`
- `B=8`: pass, `max_rel_diff=0`, `recall=1.0`
- `B=64`: pass, `max_rel_diff=0`, `recall=1.0`
- `B=256`: pass, `max_rel_diff=0`, `recall=1.0`

DeepGEMM path:

- FP8 decode was invalid at all 4 points; recalls stayed very low (`0.0` to `0.0433`)
- NVFP4 decode was also invalid at all 4 points; recalls stayed very low (`0.0078` to `0.0402`)
- During the first `B=256` FP8 attempt, DeepGEMM hit a transient OOM because other remote processes were holding large allocations; I reran that point successfully and kept the successful measurement in the final JSONL

## Notes

- I made the ragged probe path self-contained for remote use by removing an unnecessary import dependency on the full DeepSeek V4 model package; this let the B200 run from the synced probe tree without installing the whole BatchGen stack remotely.
- No `ncu` was used; `profiling_tier=1` was forced exactly as requested.
- This run validates the ragged kernel on B200, but it does **not** validate DeepGEMM numerics on `sm_100`; the comparison currently shows a large correctness gap on this setup.
