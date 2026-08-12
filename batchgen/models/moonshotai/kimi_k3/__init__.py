# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3                                                           #
#  copyright (c) EfficientMoE team 2025                                         #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""Kimi-K3 model package for BatchGen.

Contents:
  * ``tokenizer``       — the K3 tokenizer + vendored, md5-verified assets.
  * ``config``          — strict K3 config parser (:func:`parse_k3_config`);
                          hard-fails on unknown keys and violated invariants.
  * ``model``           — the K3 decoder (M2: prefill-only, eager).
  * ``kda_reference``   — vendored pure-torch KDA core; parity-oracle only,
                          never a serving path.
  * ``assets/``         — checkpoint config/tokenizer files (byte-pinned).

The load path / parameter server / tensor map still live under
``batchgen.models.moonshotai.kimi_linear`` (``k3/tensor_map.py``): K3's
checkpoint layout is owned there, the nn.Module lives here.

Submodules are intentionally NOT imported eagerly: tokenizer users must not
pull torch, and ``model`` must stay importable on CPU (its fla import is lazy,
inside the KDA forward).

This directory MUST remain an importable package (not a bare data folder):
``setup.py`` ships ``**/*.json`` / ``**/*.model`` as package data, but the
``assets/*.py`` chat renderer only makes it into the wheel because
``find_packages()`` sees ``assets`` as a package. See ``assets/__init__.py``.
"""

__all__ = ["tokenizer", "config", "model", "kda_reference"]
