# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
#                                                                               #
#  you may obtain a copy of the license at                                      #
#                                                                               #
#                  http://www.apache.org/licenses/license-2.0                   #
#                                                                               #
#  unless required by applicable law or agreed to in writing, software          #
#  distributed under the license is distributed on an "as is" basis,            #
#  without warranties or conditions of any kind, either express or implied.     #
#  see the license for the specific language governing permissions and          #
#  limitations under the license.                                               #
# ---------------------------------------------------------------------------- #

"""Single, decoupled model-config resolver for BatchGen.

Historically GLM config was resolved in three inconsistent places and the
engine ran on values hardcoded in ``glm5_initializer._parse_model_config`` that
silently drifted from the checkpoint. This module is the ONE producer of a
rich, checkpoint-backed :class:`BaseModelConfig` subclass; the GLM initializer
then *projects* that rich config into the minimal engine ``ModelConfig``.

Design constraints (do not break):

* This module's OWN body is PURE PYTHON — no ``torch`` / engine / distributed
  imports execute at module load. Family config classes are imported *lazily*
  inside :meth:`BatchGenModelConfig.resolve` (they live under ``batchgen.models``
  and only pull heavier deps when actually needed).

  Caveat on import path: the enclosing ``batchgen.config`` package ``__init__``
  eagerly imports the tokenizer stack (torch) and the model registry (whose
  auto-import triggers an engine JIT build), so a plain
  ``import batchgen.config.batchgen_model_config`` is NOT torch-free — that cost
  comes from the package ``__init__``, not this module. In the real engine the
  GLM initializer already lives behind torch, so this is a non-issue there.
  A genuinely headless caller (e.g. the unit tests) must exec this file directly
  via ``importlib`` — see ``tests/test_batchgen_model_config.py`` — rather than
  going through the package import.
* Supported variants are matched by an ordered name->class registry. Insertion
  order == iteration order, so more-specific patterns (``GLM-5.2``) MUST precede
  broader ones (``GLM-5``) — otherwise ``GLM-5.2-FP8`` would substring-match
  ``GLM-5`` first.
* Only GLM routes through this resolver today. Other families keep their
  existing ``load_config`` behaviour untouched.
"""

from __future__ import annotations

import json
import logging
import re
from importlib import import_module
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from batchgen.config.model_config import BaseModelConfig

logger = logging.getLogger(__name__)


# Ordered registry: substring pattern -> (module path, class name) of the rich
# family config. ORDER MATTERS — dict insertion order is iteration order, and a
# more-specific pattern must come before a broader one it would substring-match.
#
# GLM-5 / GLM-5.1 share the architecturally-identical glm_moe_dsa graph and use
# GLM5Config. GLM-5.2 gets its OWN config identity (GLM52Config, model_type
# "glm_moe_dsa_5_2"); the model *code* stays shared, only the config differs.
_SUPPORTED_VARIANTS: Dict[str, Tuple[str, str]] = {
    "GLM-5.2-FP8": ("batchgen.models.glm.glm5.config", "GLM52Config"),
    "GLM-5.2": ("batchgen.models.glm.glm5.config", "GLM52Config"),
    "GLM-5.1-FP8": ("batchgen.models.glm.glm5.config", "GLM5Config"),
    "GLM-5.1": ("batchgen.models.glm.glm5.config", "GLM5Config"),
    "GLM-5-FP8": ("batchgen.models.glm.glm5.config", "GLM5Config"),
    "GLM-5": ("batchgen.models.glm.glm5.config", "GLM5Config"),
}

# Best-effort fallback for an unlisted GLM variant (see resolve()).
_FALLBACK_VARIANT: Tuple[str, str] = ("batchgen.models.glm.glm5.config", "GLM5Config")


class BatchGenModelConfig:
    """Namespace for the config-resolution entry point.

    This is intentionally a thin static facade rather than a dataclass: it holds
    no state and exists only so callers get a single, discoverable API
    (``BatchGenModelConfig.resolve(...)``) that returns the rich family config.
    """

    @staticmethod
    def _match_variant(model_name: str) -> Optional[Tuple[str, str]]:
        """Return the (module, class) for the first matching name pattern.

        When ``model_name`` matches only the broad ``GLM-5`` / ``GLM-5-FP8``
        pattern but carries an unlisted minor/patch token (e.g. ``GLM-5.3``,
        ``GLM-50``), emit a loud warning before binding it to GLM5Config — a
        silently mis-resolved future variant would build the engine with wrong
        dims (mirrors model_registry._warn_if_unlisted_glm5_variant).
        """
        for pattern, target in _SUPPORTED_VARIANTS.items():
            if pattern in model_name:
                if pattern in ("GLM-5", "GLM-5-FP8"):
                    BatchGenModelConfig._warn_unlisted_glm5(model_name)
                return target
        return None

    @staticmethod
    def _warn_unlisted_glm5(model_name: str) -> None:
        """Loudly warn for a GLM-5.x identifier caught by the broad GLM-5 rule."""
        listed_minors = set()
        for pat in _SUPPORTED_VARIANTS:
            m = re.search(r"GLM-5(?:\.(\d+))?", pat)
            if m and pat.startswith("GLM-5"):
                listed_minors.add(m.group(1))  # None for bare GLM-5
        m = re.search(r"GLM-5(\d*)(?:\.(\d+))?", model_name)
        if m is None:
            return
        glued, minor = m.group(1), m.group(2)
        if glued:
            logger.warning(
                "BatchGenModelConfig.resolve: model_name=%r matched the broad "
                "'GLM-5' pattern as a superstring (GLM-5%s...). This is almost "
                "certainly NOT GLM-5; resolving to GLM5Config anyway. Add an "
                "explicit variant if this is real.",
                model_name, glued,
            )
        elif minor not in listed_minors:
            logger.warning(
                "BatchGenModelConfig.resolve: model_name=%r is an unlisted "
                "GLM-5.%s variant; it matched the broad 'GLM-5' pattern and is "
                "resolving to GLM5Config (GLM-5 base). If its config.json "
                "diverges the engine will be built with wrong dims. Add an "
                "explicit 'GLM-5.%s' entry to _SUPPORTED_VARIANTS.",
                model_name, minor, minor,
            )

    @staticmethod
    def _read_hf_config(checkpoint_path: Optional[str]) -> Optional[Dict]:
        """Load ``config.json`` from a checkpoint dir (or a direct json path).

        Returns None when no checkpoint path is given or no config.json exists
        (e.g. a bare HuggingFace model id) — the caller then falls back to the
        family config's built-in defaults.
        """
        if not checkpoint_path:
            return None
        p = Path(checkpoint_path)
        if p.is_file() and p.suffix == ".json":
            config_file = p
        else:
            config_file = p / "config.json"
        if not config_file.exists():
            return None
        with open(config_file, "r") as f:
            return json.load(f)

    @staticmethod
    def resolve(
        model_name: str,
        checkpoint_path: Optional[str] = None,
    ) -> "BaseModelConfig":
        """Resolve a rich model config for ``model_name``.

        Steps:
          1. Match ``model_name`` against the supported-variant registry to pick
             the family config class. An unlisted variant proceeds best-effort
             (GLM5Config) but logs a loud warning.
          2. Read ``<checkpoint_path>/config.json`` when available.
          3. Build the rich config via the class's ``from_hf`` classmethod when a
             checkpoint config is present; otherwise use the class defaults.
          4. ``validate()`` — FAIL LOUD on missing required fields or
             self-inconsistency.
          5. Return the rich subclass instance.

        Args:
            model_name: Model identifier / checkpoint name used for pattern
                matching (e.g. "zai-org/GLM-5.2-FP8", "GLM-5-FP8").
            checkpoint_path: Local path to the checkpoint dir (or its
                config.json). None / non-local ids fall back to defaults.

        Returns:
            A rich :class:`BaseModelConfig` subclass instance.
        """
        target = BatchGenModelConfig._match_variant(model_name)
        if target is None:
            logger.warning(
                "BatchGenModelConfig.resolve: model_name=%r matched no supported "
                "variant %s. Proceeding best-effort with %s defaults; verify the "
                "resolved config is correct for this checkpoint.",
                model_name,
                list(_SUPPORTED_VARIANTS.keys()),
                _FALLBACK_VARIANT[1],
            )
            target = _FALLBACK_VARIANT

        module_path, class_name = target
        # Lazy import: keeps this module free of torch / engine deps at load.
        config_module = import_module(module_path)
        config_cls = getattr(config_module, class_name)

        hf_dict = BatchGenModelConfig._read_hf_config(checkpoint_path)
        if hf_dict is not None:
            logger.info(
                "Resolving %s from checkpoint config for model_name=%r",
                class_name,
                model_name,
            )
            config = config_cls.from_hf(hf_dict)
        else:
            logger.warning(
                "BatchGenModelConfig.resolve: no config.json found for "
                "model_name=%r (checkpoint_path=%r). Using %s built-in defaults.",
                model_name,
                checkpoint_path,
                class_name,
            )
            config = config_cls()

        config._name_or_path = model_name
        config.validate()
        return config
