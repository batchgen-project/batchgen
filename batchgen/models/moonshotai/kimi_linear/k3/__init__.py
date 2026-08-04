# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3 divergences from the shared Kimi-Linear stack             #
#  copyright (c) EfficientMoE team 2025                                         #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""Kimi-K3 (2.8T) specifics, contained in one subpackage.

K3 and Kimi-Linear-48B share `config.py`, `model.py`, the parameter server, the
initializer and the model-side dispatch sites (`get_initializer.py`,
`worker_manager.py`, `model_registry.py`, `host_kv_mananger_config.py` already
route "kimi-k3" here). What genuinely differs — the checkpoint name prefix, the
MXFP4 routed-expert format, the full-rank KDA gate, q-LoRA MLA, LatentMoE and
AttnRes tensors — lives here rather than as `if is_k3:` branches sprayed through
the shared files, so the containment stays reviewable and a later fork is a
`git mv`.

The TOKENIZER is the one exception, and it costs exactly one core PR. K3 does
NOT share the 48B's tokenizer: the two ship different `added_tokens_decoder`
tables over the same BPE merge file (163586 is `<|end_of_msg|>` in K3 and
`<|im_end|>` in the 48B), and K3 has no Jinja chat template at all. It is served
by `batchgen/models/moonshotai/kimi_k3/tokenizer.py`, and pointing
`tokenizer_registry.py`'s "Kimi-K3" pattern at tokenizer type "kimi_k3" is a
change to `batchgen/config/`, which is outside MODEL_ALLOW_RE. Until that core
PR lands, "Kimi-K3" still resolves to "kimi_linear" and K3 is served by the
wrong tokenizer — silently (bug_log.md 2026-07-31).

Nothing is exported eagerly: the shared files import from this package inside
their K3 branches only.
"""
