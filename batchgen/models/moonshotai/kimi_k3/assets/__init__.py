# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3 vendored checkpoint assets                                #
#  copyright (c) EfficientMoE team 2025                                         #
# ---------------------------------------------------------------------------- #
"""Kimi-K3 tokenizer assets, vendored verbatim from the served checkpoint.

Every file in this directory is a byte-for-byte copy of the corresponding file
in the released Kimi-K3 checkpoint and is md5-verified against it. Nothing here
is BatchGen-authored, and nothing here may be edited: ``tokenizer.py`` imports
``tokenization_kimi`` / ``encoding_k3`` from this package so that BatchGen's
chat rendering is bit-exact with HuggingFace *by construction* rather than by
a hand-port that can drift.

This file exists so ``find_packages()`` treats the directory as a package and
``setup.py`` ships the ``.py`` assets into the wheel — without it the wheel
installs but ``encoding_k3`` is missing at runtime (``tokenization_kimi.py:14``
does ``from .encoding_k3 import ...``). The same reasoning applies to
``kimi_k25/assets/__init__.py``.

Re-vendoring procedure: replace files wholesale, then run
``tests/test_kimi_k3_tokenizer.py``. The pinned tables in
``kimi_k3/tokenizer.py`` (``KIMI_K3_ADDED_TOKENS`` and friends) will fail the
suite if the special-token layout moved, which is exactly when a human must
re-decide rather than a fallback silently absorb the change.
"""
