# autoresearch_v4

Autonomous serving-config optimization loop for **DeepSeek-V4-Flash on 4x RTX PRO 6000 Blackwell Server GPUs (0-3 only)**.

This org is adapted from karpathy/autoresearch, but the edit surface is **serving/system config only**. The kernels and model code are already correct and must remain frozen.

## In-scope files

Read these first:

1. `docs/4xrtx6000pro-v4flash-setup.md` — verified setup, baseline, sweep levers.
2. `benchmarks/grouped_moe_probes/autoresearch_v4/bench_v4_config.py` — **fixed, read-only metric harness**.
3. `benchmarks/grouped_moe_probes/autoresearch_v4/config_space.py` — the edit surface.
4. `benchmarks/grouped_moe_probes/autoresearch_v4/results.tsv` — append-only experiment log.

## What you CAN edit

- `config_space.py`
- temporary JSON/dict configs that feed `bench_v4_config.py`

Allowed knobs are only serving/system configuration:

- `--gpu-memory-frac`
- `--host-kv-cache-size`
- `--kv-dtype`
- `--world-size` / verified EP layout flags only
- `--initial-gpu-page-buffer`
- `--extension-gpu-page-buffer`
- NCCL env (`NCCL_P2P_LEVEL`, `NCCL_ALGO`, `NCCL_MIN_NCHANNELS`, `NCCL_MAX_NCHANNELS`, `NCCL_BUFFSIZE`, `NCCL_SHM_DISABLE`)
- NUMA pinning (`numactl --cpunodebind=0 --membind=0`)
- fixed-harness request concurrency/batch size

## What you MUST NOT edit

- `bench_v4_config.py` — fixed metric, fixed warmup, fixed cleanup, fixed accuracy guardrail
- any model/kernels source
- checkpoint files
- test datasets
- any benchmark output number by hand

## Goal

Maximize **decode tokens/sec** on the fixed harness.

Primary metric:

- `decode_tok_s` from the worker log during the fixed decode benchmark batch

Secondary constraints:

- `prefill_ttft_s` should not regress badly
- `accuracy_guard` must pass (small MMLU/coherence sanity check)
- configs that leak GPU memory, containers, or `/dev/shm` are failures

## First run

The first run is always the known-good baseline config. Do not start by mutating anything.

## Experiment loop

LOOP FOREVER:

1. Read the current best row in `results.tsv`.
2. Propose exactly **one** config change.
3. Apply that one change in `config_space.py` or via a one-off config payload.
4. Run the fixed harness:

   ```bash
   python benchmarks/grouped_moe_probes/autoresearch_v4/bench_v4_config.py --config-name <name> --tag <tag>
   ```

5. Inspect the new TSV row.
6. Keep the change only if all of the following are true:
   - status is `ok`
   - accuracy guard passes
   - decode tok/s improves meaningfully, or is flat with lower complexity / lower risk
7. If the new row is worse, revert the config change.
8. Log every experiment, including crashes and cleanup failures.

## Keep / discard rule

- **keep**: higher `decode_tok_s`, with guardrail pass and no cleanup leak
- **discard**: lower/equal `decode_tok_s` without a simplicity win
- **crash**: server fails, benchmark fails, or cleanup fails

Simplicity bias:

- prefer simpler configs when gains are within noise
- avoid coupled multi-knob jumps unless you are recovering from a known broken config

## Non-negotiable ops contract

Every experiment must leave the machine clean **before the next launch**:

- kill `launch_http_server` in-container
- remove the container
- kill leftover compute PIDs on GPUs 0-3
- clear leaked `/dev/shm/shm_*` and `/dev/shm/batchgen_host_kv_cache`
- verify GPUs 0-3 are back to `0 MiB`
- verify `/dev/shm` is clean enough for the next run

If cleanup does not complete, treat the experiment as a failure. Never continue from a poisoned machine state.

## Anti-gaming rules

- Do not change the benchmark prompts, warmup, decode length, or guardrail
- Do not fake numbers
- Do not use cache tricks, precomputed outputs, or harness edits
- Do not optimize for the first JIT request; warmup is discarded by design
- Do not write large artifacts under `/mnt`; use `/tmp` or `/dev/shm`

## Never stop

Once the loop starts, do not ask whether to continue. Continue iterating until the human stops you.
