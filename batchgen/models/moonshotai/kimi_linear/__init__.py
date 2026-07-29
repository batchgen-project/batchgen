# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-Linear / Kimi-K3 family                                      #
#  copyright (c) EfficientMoE team 2025                                         #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""Kimi-Linear / Kimi-K3 model family for BatchGen.

Hybrid KDA (linear/recurrent attention) + NoPE-MLA MoE. Built fresh for this family;
shared plumbing (tokenizer, MLA math, MoE, vision stub, engine seams) is imported from
`batchgen.models.moonshotai.kimi_k25` where it is identical, rather than copied.

Exports are added incrementally as each component lands (config first, then
parameter server / initializer / parallel-strategy manager).
"""
