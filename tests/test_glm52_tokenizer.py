"""CPU-only GLM-5.2 tokenizer and chat-template regressions."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GLM5_DIR = REPO_ROOT / "batchgen" / "models" / "glm" / "glm5"
TEMPLATE_PATH = GLM5_DIR / "chat_template_5_2.jinja"
GLM53_TEMPLATE_PATH = GLM5_DIR / "chat_template_5_3.jinja"


def _namespace(name: str, path: Path | None = None):
    module = types.ModuleType(name)
    module.__path__ = [] if path is None else [str(path)]
    sys.modules[name] = module
    return module


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _bootstrap_glm52_tokenizer():
    _namespace("batchgen", REPO_ROOT / "batchgen")
    _namespace("batchgen.config", REPO_ROOT / "batchgen" / "config")
    _load_module(
        "batchgen.config.model_name_utils",
        REPO_ROOT / "batchgen" / "config" / "model_name_utils.py",
    )
    _load_module(
        "batchgen.config.base_tokenizer",
        REPO_ROOT / "batchgen" / "config" / "base_tokenizer.py",
    )
    _load_module(
        "batchgen.config.fast_tokenizer",
        REPO_ROOT / "batchgen" / "config" / "fast_tokenizer.py",
    )

    # Load the registry without importing every model package, then register
    # only the GLM tokenizer module under test.
    _namespace("batchgen.models")
    registry = _load_module(
        "batchgen.config.tokenizer_registry",
        REPO_ROOT / "batchgen" / "config" / "tokenizer_registry.py",
    )
    _namespace("batchgen.models.glm", REPO_ROOT / "batchgen" / "models" / "glm")
    _namespace("batchgen.models.glm.glm5", GLM5_DIR)
    tokenizer_module = _load_module(
        "batchgen.models.glm.glm5.tokenizer",
        GLM5_DIR / "tokenizer.py",
    )
    return registry, tokenizer_module


def test_glm52_chat_template_matches_released_bytes():
    assert hashlib.md5(TEMPLATE_PATH.read_bytes()).hexdigest() == (
        "42994f78b64752fe472149dd7e20410d"
    )


def test_glm53_chat_template_matches_released_bytes():
    assert hashlib.sha256(GLM53_TEMPLATE_PATH.read_bytes()).hexdigest() == (
        "3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c"
    )


def test_glm53_routes_to_dedicated_template_and_supports_loop_controls():
    registry, tokenizer_module = _bootstrap_glm52_tokenizer()

    tokenizer = registry.load_tokenizer("zai-org/GLM-5.3-FP8")
    assert isinstance(tokenizer, tokenizer_module.GLM53Tokenizer)
    assert tokenizer.CHAT_TEMPLATE_FILENAME == "chat_template_5_3.jinja"
    for effort, label in (("low", "Low"), ("high", "High"), ("max", "Max")):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=effort,
        )
        assert f"<|system|>Reasoning Effort: {label}" in rendered
        assert rendered.endswith("<|assistant|><think>")


def test_glm52_routes_to_dedicated_template_and_renders_reasoning_directive():
    registry, tokenizer_module = _bootstrap_glm52_tokenizer()

    tokenizer = registry.load_tokenizer("zai-org/GLM-5.2-FP8")
    assert isinstance(tokenizer, tokenizer_module.GLM52Tokenizer)
    assert tokenizer.CHAT_TEMPLATE_FILENAME == "chat_template_5_2.jinja"

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert rendered == (
        "[gMASK]<sop><|system|>Reasoning Effort: Max"
        "<|user|>Hello<|assistant|><think>"
    )
