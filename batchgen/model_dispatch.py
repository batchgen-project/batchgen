"""Single source of truth for model-name -> (Initializer, PSM) dispatch.

The runtime core must never branch on a model name or import a model package;
all such mapping lives here. (Design: batchgen_design/model_architecture_spec.md
section 2.1 -- model->implementation dispatch lives only in the registry layer.)

This is a behavior-preserving consolidation of the former duplicated if/elif
chains in `get_initializer.py` and `get_parallel_strategy_manager.py`: the same
matching semantics (substring / exact, same order) resolve every name to the
same classes as before. Migrating the key to the exact canonical HuggingFace id
is a follow-up that depends on standardizing launch identifiers.

Class imports stay lazy (per-entry loader closures) so importing this module
does not pull in every model package.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from batchgen.config.model_name_utils import KIMI_K25_BACKEND_NAME_PATTERNS


@dataclass(frozen=True)
class ModelEntry:
    """One registered model: how to match its name and how to load its classes."""

    key: str
    substrings: Tuple[str, ...] = ()
    exact_names: Tuple[str, ...] = ()
    initializer_loader: Optional[Callable[[], type]] = None
    psm_loader: Optional[Callable[[], type]] = None

    def matches(self, name_lower: str) -> bool:
        if name_lower in self.exact_names:
            return True
        return any(pattern in name_lower for pattern in self.substrings)


# --- lazy loaders: keep model-package imports out of module import time -------
def _minimax_initializer():
    from batchgen.models.minimax.minimax_m25.minimax_m25_initializer import MiniMaxM25Initializer
    return MiniMaxM25Initializer


def _minimax_psm():
    from batchgen.models.minimax.minimax_m25.Parallel_Strategy_Manager import MiniMaxM25ParallelStrategyManager
    return MiniMaxM25ParallelStrategyManager


def _kimi_k25_initializer():
    from batchgen.models.moonshotai.kimi_k25.kimi_initializer import KimiK25Initializer
    return KimiK25Initializer


def _kimi_k25_psm():
    from batchgen.models.moonshotai.kimi_k25.Parallel_Strategy_Manager import KimiK25ParallelStrategyManager
    return KimiK25ParallelStrategyManager


def _deepseek_v4_flash_initializer():
    from batchgen.models.deepseek.deepseekv4_flash.deepseekv4_flash_initializer import DeepSeekV4FlashInitializer
    return DeepSeekV4FlashInitializer


def _deepseek_v4_flash_psm():
    from batchgen.models.deepseek.deepseekv4_flash.Parallel_Strategy_Manager import DeepSeekV4FlashParallelStrategyManager
    return DeepSeekV4FlashParallelStrategyManager


def _deepseek_v3_initializer():
    from batchgen.models.deepseek.deepseekv3.deepseekv3_initializer import DeepseekV3Initializer
    return DeepseekV3Initializer


def _deepseek_v3_psm():
    from batchgen.models.deepseek.deepseekv3.Parallel_Strategy_Manager import DeepseekV3ParallelStrategyManager
    return DeepseekV3ParallelStrategyManager


def _gpt_oss_initializer():
    from batchgen.models.openai.gpt_oss_120b.gpt_oss_initializer import GptOssInitializer
    return GptOssInitializer


def _gpt_oss_psm():
    from batchgen.models.openai.gpt_oss_120b.Parallel_Strategy_Manager import GptOssParallelStrategyManager
    return GptOssParallelStrategyManager


def _glm5_initializer():
    from batchgen.models.glm.glm5.glm5_initializer import GLM5Initializer
    return GLM5Initializer


def _glm5_psm():
    from batchgen.models.glm.glm5.Parallel_Strategy_Manager import GLM5ParallelStrategyManager
    return GLM5ParallelStrategyManager


# Order matters: first match wins. This reproduces the original if/elif order in
# get_initializer.py / get_parallel_strategy_manager.py exactly.
MODEL_REGISTRY: Tuple[ModelEntry, ...] = (
    ModelEntry("minimax_m25", substrings=("minimax",),
               initializer_loader=_minimax_initializer, psm_loader=_minimax_psm),
    ModelEntry("kimi_k25", substrings=KIMI_K25_BACKEND_NAME_PATTERNS,
               initializer_loader=_kimi_k25_initializer, psm_loader=_kimi_k25_psm),
    ModelEntry("deepseek_v4_flash", substrings=("deepseek-v4",),
               initializer_loader=_deepseek_v4_flash_initializer, psm_loader=_deepseek_v4_flash_psm),
    ModelEntry("deepseek_v3", exact_names=("deepseek-ai/deepseek-r1", "deepseek-ai/deepseek-v3"),
               initializer_loader=_deepseek_v3_initializer, psm_loader=_deepseek_v3_psm),
    ModelEntry("gpt_oss", substrings=("gpt-oss-120b",),
               initializer_loader=_gpt_oss_initializer, psm_loader=_gpt_oss_psm),
    ModelEntry("glm5", substrings=("glm-5",),
               initializer_loader=_glm5_initializer, psm_loader=_glm5_psm),
)


def resolve_model(model_name: str) -> ModelEntry:
    """Return the registry entry for `model_name`, or raise ValueError if unsupported."""
    name_lower = (model_name or "").strip().lower()
    for entry in MODEL_REGISTRY:
        if entry.matches(name_lower):
            return entry
    raise ValueError(f"Unsupported model name: {model_name}")
