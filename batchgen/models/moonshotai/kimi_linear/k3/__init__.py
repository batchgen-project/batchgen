# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3 divergences from the shared Kimi-Linear stack             #
#  copyright (c) EfficientMoE team 2025                                         #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""Kimi-K3 (2.8T) specifics, contained in one subpackage.

K3 and Kimi-Linear-48B share `config.py`, `model.py`, the parameter server, the
initializer and every dispatch site (`get_initializer.py`, `worker_manager.py`,
`model_registry.py`, `tokenizer_registry.py`, `host_kv_mananger_config.py` all
already route "kimi-k3" here), so K3 costs ZERO core registration PRs. What
genuinely differs — the checkpoint name prefix, the MXFP4 routed-expert format,
the full-rank KDA gate, q-LoRA MLA, LatentMoE and AttnRes tensors — lives here
rather than as `if is_k3:` branches sprayed through the shared files, so the
containment stays reviewable and a later fork is a `git mv`.

Nothing is exported eagerly: the shared files import from this package inside
their K3 branches only.
"""
