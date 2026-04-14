# GLM-5.1-FP8 Support

GLM-5.1 (754B params, ~40B active) was open-sourced by Z.ai on 2026-04-07
via HuggingFace (`zai-org/GLM-5.1` and `zai-org/GLM-5.1-FP8`).

**GLM-5.1 is architecturally identical to GLM-5.** Every field in
`config.json` — `model_type`, `architectures`, all layer/head/expert
dimensions, RoPE parameters, DSA indexer settings, MTP layer 78, FP8
`quantization_config` (including the full `modules_to_not_convert` list) —
matches byte-for-byte. The only delta is the informational
`transformers_version` string. Verified by direct `config.json` diff
between `zai-org/GLM-5-FP8` and `zai-org/GLM-5.1-FP8` on 2026-04-14.

As a result **no new model code, initializer, PSM, parameter server, KV
profile, or tokenizer is needed** — the existing
`batchgen/models/glm/glm5/` implementation serves both checkpoints. GLM-5.1
is a weight refresh, not a new model family.

## What was wired up

Registry entries were added so `GLM-5.1` / `GLM-5.1-FP8` resolve to the
same `glm_moe_dsa` path as `GLM-5` / `GLM-5-FP8`:

| File | Addition |
|---|---|
| `batchgen/config/model_registry.py` | `MODEL_NAME_PATTERNS`: `"GLM-5.1-FP8"`, `"GLM-5.1"` → `glm_moe_dsa` (placed before `GLM-5` so substring match picks the longer pattern) |
| `batchgen/config/tokenizer_registry.py` | `TOKENIZER_NAME_PATTERNS`: same two patterns → `glm_moe_dsa` tokenizer |
| `batchgen/kv_cache/host_kv_mananger_config.py` | Both `glm5_mla` and `glm5_indexer` alias tuples extended with `zai-org/glm-5.1-fp8`, `zai-org/glm-5.1`, `glm-5.1-fp8`, `glm-5.1` |
| `batchgen/server/process_utils.py` | `MODEL_BYTE_SIZES`: `"zai-org/GLM-5.1-FP8"` → 760 GB, `"zai-org/GLM-5.1"` → 1400 GB (same as GLM-5 — identical param count) |

No changes were needed in `get_initializer.py`, `get_parallel_strategy_manager.py`,
or the parameter-server dispatch in `batchgen_server.py` /
`batchgen_server_dev.py` / `server/worker_manager.py` — those branches
already substring-match `"glm-5"` (case-insensitive), which `GLM-5.1-FP8`
satisfies.

## Running GLM-5.1-FP8

```bash
python -m batchgen.batchgen_server \
  --model zai-org/GLM-5.1-FP8 \
  --enable-thinking \
  ...
```

Everything downstream (tokenizer, chat template, config, PSM, initializer,
WP2/WP4/WP5 DSA kernels, dual host KV cache, MMLU-Pro harness) is the
same code path that serves GLM-5-FP8.

## Regression checklist

1. `load_config("zai-org/GLM-5.1-FP8")` returns `Glm5Config` with the
   expected shapes.
2. `build_host_kv_config` / `build_host_kv_config_aux` resolve both the
   MLA profile and the DSA indexer profile for `zai-org/GLM-5.1-FP8`.
3. `get_model_byte_size("zai-org/GLM-5.1-FP8")` returns 760 GB (exact
   match, no fallback warning).
4. An MMLU-Pro smoke on GLM-5.1-FP8 should match the GLM-5-FP8 baseline
   (≥78% with WP2/WP4/WP5 enabled). The weights differ, but the forward
   graph is identical.
