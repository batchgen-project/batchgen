# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3                                                           #
#  copyright (c) EfficientMoE team 2025                                         #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""Kimi-K3 model package for BatchGen.

Currently holds the K3 tokenizer and its vendored, md5-verified checkpoint
assets. The K3 *model* (weights / layers / parallel strategy) lives in
``batchgen.models.moonshotai.kimi_linear`` because K3's text tower is the
Kimi-Linear architecture; only the tokenizer is genuinely K3-specific.

This directory MUST remain an importable package (not a bare data folder):
``setup.py`` ships ``**/*.json`` / ``**/*.model`` as package data, but the
``assets/*.py`` chat renderer only makes it into the wheel because
``find_packages()`` sees ``assets`` as a package. See ``assets/__init__.py``.
"""

__all__ = ["tokenizer"]
