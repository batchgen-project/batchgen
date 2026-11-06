# Deploy Kimi-K3.1-Pro on 4 H20 Nodes

Kimi-K3.1-Pro is Kimi-K3 with a gated sparse-attention (DSA) indexer on its 24
full-attention layers: each of those layers keeps its gated-MLA geometry and adds a
per-layer MQA indexer (one 128-dimension key per token, LayerNorm key norm, no rotary
embedding on the index path, 32 query heads read from the MLA query latent, per-head
gates, top-2048 selection). Everything else — the 896-expert MXFP4 MoE, the KDA layers,
the tokenizer and the chat template — is unchanged from Kimi-K3, so this guide only
describes the deltas; the workflow is the [Kimi-K3 on 4 H20 nodes guide](deploy-kimi-k3-h20.md)
with the substitutions below.

## 1. What changes vs Kimi-K3

- **Model id**: use the registered id `moonshotai/Kimi-K3.1-Pro` for `--model` (a local
  path is refused: the config registry, the host-KV profile table and the runtime
  preflight all key on the id). Pass `--cache-dir` pointing at the downloaded checkpoint
  so its `config.json` (with `dsa_config`) is read; there is no built-in K3.1 default
  config.
- **Checkpoint**: download `moonshotai/Kimi-K3.1-Pro` and convert it with the same
  converter and flags as Kimi-K3 (§2 of the [H200 guide](deploy-kimi-k3-h200.md)); the
  converter passes the indexer tensors through unquantized. The converted size is
  essentially Kimi-K3's.
- **Hopper only.** The DSA index-key writer and selector extensions are built for
  `sm90a`; the runtime preflight refuses a build without them (`kimi_k31` contract,
  which also requires DeepGEMM). There is no SM100 DSA path in this release.
- **KV cache is dual-region.** Next to the MLA latent cache, each of the 24 DSA layers
  stores one FP8 index key per token (page-split layout: 128 one-byte values plus one
  fp32 scale per token, 132 bytes per token per layer). The host-KV and GPU-KV pools
  split the budget automatically; size `--host-kv-cache-size` as for Kimi-K3 and the
  index-key region rides along.
- **Prefix cache is not supported for Kimi-K3.1 in this release**: a launch with
  `--enable-prefix-cache` is refused at argument validation. Launch without it.
- **Long prompts take the sparse path automatically.** Prompts longer than the model's
  `index_topk` (2,048 tokens) attend only to their indexer-selected tokens during
  prefill; shorter prompts stay dense. Decode scores every step through the paged FP8
  indexer inside the captured CUDA graph. No flag controls this; it is the model's
  architecture.

## 2. Prepare the Distributed-Weight Store

Identical to §2 of the [Kimi-K3 H20 guide](deploy-kimi-k3-h20.md), with the K3.1
converted checkpoint as `--converted-ckpt` and `--cache-dir` pointing at the K3.1
checkpoint directory. The builder validates the checkpoint against the full K3.1 tensor
map (the 24 indexer tensor groups included) before writing anything.

## 3. Start the Server (4 × 8 H20)

As §3 of the [Kimi-K3 H20 guide](deploy-kimi-k3-h20.md), substituting the model id and
store paths:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_GID_INDEX=3 UCX_IB_GID_INDEX=3

python -m batchgen.launch_http_server \
    --model moonshotai/Kimi-K3.1-Pro \
    --cache-dir /shared/models/Kimi-K3.1-Pro \
    --converted-ckpt-dir /shared/models/Kimi-K3.1-Pro/converted_ckpt \
    --distributed-weight-config /shared/k31-dwconfig/node${RANK}.json \
    --listen-port 10900 \
    --world-size 32 --nnodes 4 --node-rank ${RANK} \
    --dist-init-addr node0-ip:12355 \
    --kv-dtype bf16 \
    --host-kv-cache-size 512 \
    --gpu-memory-frac 0.90 \
    --parse-thinking \
    --storage-path /shared/storage
```

Notes:

- `--gpu-memory-frac 0.90` is the validated bring-up setting for K3.1 on H20; raise it
  only if the resident-expert repack, both KV regions and the capture headroom still fit.
- Do not pass `--enable-deepep` or `--enable-prefix-cache` (unsupported for this model in
  this release).
- Before readiness every worker preloads the family extensions **and** the DSA indexer
  extensions; a build missing them fails at startup, not at the first request.
- All other H20 notes (watermark, `/dev/shm` sizing with `huge=within_size`, port
  reservations, cluster fail-fast, Triton prewarm) apply unchanged from the
  [Kimi-K3 H20 guide](deploy-kimi-k3-h20.md).

## 4. Submit Jobs

Identical to Kimi-K3: set `"model": "kimi-k3"`-family request bodies to
`moonshotai/Kimi-K3.1-Pro` (the tokenizer and chat template are shared with Kimi-K3,
including `enable_thinking` and the thinking-effort levels) and use the
[Batch API Guide](batch-api-guide.md).
