# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                            #
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
* The release a ``--model`` value names is resolved by
  :mod:`batchgen.config.model_identity`, the rule every per-model table (host-KV
  profiles, host weight envelope) uses: a registered release name, a local
  checkpoint directory named after it, or a Base/Instruct/dated variant of it.
  A value that names no registered release is refused, never bound to a
  neighbouring release, and a checkpoint whose config.json carries another
  release's geometry is refused too, so a local path cannot build an engine
  that disagrees with the host-KV layout allocated for the same value.
* Only GLM routes through this resolver today. Other families keep their
  existing ``load_config`` behaviour untouched.
"""

from __future__ import annotations

import json
import logging
from importlib import import_module
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from batchgen.config import model_identity
from batchgen.models.glm.glm5 import releases as glm5_releases

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from batchgen.config.model_config import BaseModelConfig

logger = logging.getLogger(__name__)


# Released name -> (module path, class name) of the rich family config.
#
# GLM-5 / GLM-5.1 share the architecturally-identical glm_moe_dsa graph and use
# GLM5Config. GLM-5.2 and GLM-5.3 get OWN config identities; the model code
# stays shared, while checkpoint geometry and chat contracts remain explicit.
# The rows are generated from the release registry in
# batchgen/models/glm/glm5/releases.py.
_SUPPORTED_VARIANTS: Dict[str, Tuple[str, str]] = glm5_releases.config_variants()

# The same released names keyed the way model_identity resolves any --model
# value, so this resolver and the host-KV profile tables agree on the release a
# value names.
_RELEASE_INDEX = model_identity.build_index(glm5_releases.releases_by_name())


class BatchGenModelConfig:
    """Namespace for the config-resolution entry point.

    This is intentionally a thin static facade rather than a dataclass: it holds
    no state and exists only so callers get a single, discoverable API
    (``BatchGenModelConfig.resolve(...)``) that returns the rich family config.
    """

    @staticmethod
    def _match_release(model_name: str) -> Optional["glm5_releases.Glm5Release"]:
        """The registered GLM-5 release ``model_name`` names, or None."""
        return model_identity.lookup(_RELEASE_INDEX, model_name)

    @staticmethod
    def _match_variant(model_name: str) -> Optional[Tuple[str, str]]:
        """Return the (module, class) of the release ``model_name`` names."""
        release = BatchGenModelConfig._match_release(model_name)
        if release is None:
            return None
        return glm5_releases.GLM5_CONFIG_MODULE, release.config_class

    @staticmethod
    def _require_checkpoint_matches_release(
        release: "glm5_releases.Glm5Release",
        hf_dict: Dict,
        model_name: str,
        checkpoint_path: Optional[str],
    ) -> None:
        """Refuse a checkpoint whose config.json carries another release's
        geometry.

        The geometry (the DSA top-k schedule) selects the host-KV indexer
        profile allocated for ``model_name``; building the engine from a
        checkpoint with a different schedule would run it against a host-KV
        layout of another release.
        """
        freq = hf_dict.get("index_topk_freq")
        found = (1 if freq is None else freq, hf_dict.get("index_skip_topk_offset"))
        geometry = release.geometry
        expected = (geometry.index_topk_freq, geometry.index_skip_topk_offset)
        if found != expected:
            raise ValueError(
                f"BatchGenModelConfig.resolve: model_name={model_name!r} names "
                f"GLM-{release.version}, but the checkpoint at "
                f"{checkpoint_path!r} has (index_topk_freq, "
                f"index_skip_topk_offset)={found}, not GLM-{release.version}'s "
                f"{expected}. Name the checkpoint directory after the release "
                "it holds."
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
          1. Resolve the registered release ``model_name`` names (see the
             module docstring); a value that names none raises ValueError.
          2. Read ``<checkpoint_path>/config.json`` when available.
          3. Build the rich config via the class's ``from_hf`` classmethod when a
             checkpoint config is present, and refuse a checkpoint whose
             geometry is another release's; otherwise use the class defaults.
          4. ``validate()`` — FAIL LOUD on missing required fields or
             self-inconsistency.
          5. Return the rich subclass instance.

        Args:
            model_name: The ``--model`` value: a released id, a local
                checkpoint directory named after one, or a variant of one
                (e.g. "zai-org/GLM-5.2-FP8", "/models/GLM-5-FP8").
            checkpoint_path: Local path to the checkpoint dir (or its
                config.json). None / non-local ids fall back to defaults.

        Returns:
            A rich :class:`BaseModelConfig` subclass instance.

        Raises:
            ValueError: ``model_name`` names no registered GLM-5 release, or
                the checkpoint holds a release of another geometry.
        """
        release = BatchGenModelConfig._match_release(model_name)
        if release is None:
            raise ValueError(
                f"BatchGenModelConfig.resolve: model_name={model_name!r} names "
                f"no registered GLM-5 release (registered: "
                f"{list(_SUPPORTED_VARIANTS)}). A local checkpoint must be a "
                "directory named after its released id, e.g. "
                ".../<org>/<released-name>; a new release is added to "
                "batchgen/models/glm/glm5/releases.py."
            )

        module_path = glm5_releases.GLM5_CONFIG_MODULE
        class_name = release.config_class
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
            BatchGenModelConfig._require_checkpoint_matches_release(
                release, hf_dict, model_name, checkpoint_path
            )
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
