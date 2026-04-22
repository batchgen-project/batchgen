"""Focused non-GPU regressions for Kimi-K2.6 support wiring."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_process_utils():
    return _load_module(
        "batchgen.server.process_utils",
        REPO_ROOT / "batchgen" / "server" / "process_utils.py",
    )


def _load_model_name_utils():
    return _load_module(
        "batchgen.config.model_name_utils",
        REPO_ROOT / "batchgen" / "config" / "model_name_utils.py",
    )


def _load_scheduler_modules():
    if "batchgen" not in sys.modules:
        pkg = types.ModuleType("batchgen")
        pkg.__path__ = [str(REPO_ROOT / "batchgen")]
        sys.modules["batchgen"] = pkg
    if "batchgen.server" not in sys.modules:
        pkg = types.ModuleType("batchgen.server")
        pkg.__path__ = [str(REPO_ROOT / "batchgen" / "server")]
        sys.modules["batchgen.server"] = pkg

    io_struct = _load_module(
        "batchgen.server.io_struct",
        REPO_ROOT / "batchgen" / "server" / "io_struct.py",
    )

    stub_specs = {
        "batchgen.server.intake_pool": {"IntakeEntry": object, "IntakePool": object, "Priority": object},
        "batchgen.server.scheduling_pool": {"SchedulingPool": object},
        "batchgen.server.server_args": {"ServerArgs": object},
        "batchgen.server.storage": {"StorageManager": object},
        "batchgen.server.worker_manager": {"WorkerManager": object},
    }
    for module_name, attrs in stub_specs.items():
        module = types.ModuleType(module_name)
        for name, value in attrs.items():
            setattr(module, name, value)
        sys.modules[module_name] = module

    batch_scheduler = _load_module(
        "batchgen.server.batch_scheduler",
        REPO_ROOT / "batchgen" / "server" / "batch_scheduler.py",
    )
    return io_struct, batch_scheduler


def test_kimi_k26_backend_model_name_helper_and_byte_size():
    model_name_utils = _load_model_name_utils()
    process_utils = _load_process_utils()

    assert model_name_utils.is_kimi_k25_backend_model("moonshotai/Kimi-K2.5")
    assert model_name_utils.is_kimi_k25_backend_model("moonshotai/Kimi-K2.6")
    assert model_name_utils.is_kimi_k25_backend_model("Kimi-K2.6")
    assert not model_name_utils.is_kimi_k25_backend_model("moonshotai/Kimi-K3")
    assert process_utils.get_model_byte_size("moonshotai/Kimi-K2.6") == 650 * 1024**3


def test_routing_sources_include_kimi_k26():
    shared_text = (
        REPO_ROOT / "batchgen" / "config" / "model_name_utils.py"
    ).read_text()
    assert "Kimi-K2.6" in shared_text or "kimi-k2.6" in shared_text

    direct_paths = [
        REPO_ROOT / "batchgen" / "server" / "process_utils.py",
        REPO_ROOT / "batchgen" / "kv_cache" / "host_kv_mananger_config.py",
    ]
    for path in direct_paths:
        text = path.read_text()
        assert "Kimi-K2.6" in text or "kimi-k2.6" in text

    helper_paths = [
        REPO_ROOT / "batchgen" / "config" / "model_registry.py",
        REPO_ROOT / "batchgen" / "config" / "tokenizer_registry.py",
        REPO_ROOT / "batchgen" / "get_initializer.py",
        REPO_ROOT / "batchgen" / "get_parallel_strategy_manager.py",
        REPO_ROOT / "batchgen" / "server" / "worker_manager.py",
        REPO_ROOT / "batchgen" / "batchgen_worker.py",
    ]
    for path in helper_paths:
        text = path.read_text()
        assert "is_kimi_k25_backend_model" in text or "KIMI_K25_BACKEND_MODEL_IDS" in text


def test_kimi_assets_include_preserve_thinking_updates():
    template_text = (
        REPO_ROOT
        / "batchgen"
        / "models"
        / "moonshotai"
        / "kimi_k25"
        / "assets"
        / "chat_template.jinja"
    ).read_text()
    tokenizer_text = (
        REPO_ROOT
        / "batchgen"
        / "models"
        / "moonshotai"
        / "kimi_k25"
        / "assets"
        / "tokenization_kimi.py"
    ).read_text()

    assert "preserve_thinking" in template_text
    assert "message.get('reasoning'" in template_text
    assert "preserve_thinking: bool = False" in tokenizer_text


def test_chat_request_accepts_preserve_thinking_and_reasoning_content():
    io_struct, _ = _load_scheduler_modules()
    ChatCompletionRequest = io_struct.ChatCompletionRequest

    req = ChatCompletionRequest(
        model="moonshotai/Kimi-K2.6",
        messages=[
            {"role": "assistant", "content": "visible", "reasoning_content": "hidden"},
            {"role": "user", "content": "next"},
        ],
        preserve_thinking=True,
    )

    assert req.preserve_thinking is True
    assert req.messages[0].reasoning_content == "hidden"


def test_scheduler_forwards_preserve_thinking_and_reasoning_content():
    io_struct, batch_scheduler = _load_scheduler_modules()
    ChatCompletionRequest = io_struct.ChatCompletionRequest
    BatchScheduler = batch_scheduler.BatchScheduler

    class _FakeTokenizer:
        def __init__(self):
            self.messages = None
            self.kwargs = None

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
            self.messages = messages
            self.kwargs = kwargs
            return "formatted-prompt"

    tokenizer = _FakeTokenizer()
    scheduler = BatchScheduler.__new__(BatchScheduler)
    scheduler._get_tokenizer = lambda model: tokenizer

    request = ChatCompletionRequest(
        model="moonshotai/Kimi-K2.6",
        messages=[
            {"role": "assistant", "content": "visible", "reasoning_content": "hidden"},
            {"role": "user", "content": "next"},
        ],
        thinking=False,
        preserve_thinking=True,
        tools=[{"type": "function", "function": {"name": "ping", "parameters": {"type": "object"}}}],
    )

    prompts, max_tokens, sampling_params = scheduler._convert_requests_to_worker_inputs(
        [SimpleNamespace(body=request)],
        None,
    )

    assert prompts == ["formatted-prompt"]
    assert max_tokens == [None]
    assert sampling_params == [{"temperature": None, "top_p": 1.0, "top_k": None}]
    assert tokenizer.kwargs["thinking"] is False
    assert tokenizer.kwargs["preserve_thinking"] is True
    assert "tools" in tokenizer.kwargs
    assert tokenizer.messages[0]["reasoning_content"] == "hidden"


def test_trim_tokens_honors_all_eos_ids():
    _, batch_scheduler = _load_scheduler_modules()
    BatchScheduler = batch_scheduler.BatchScheduler

    scheduler = BatchScheduler.__new__(BatchScheduler)
    tokenizer = SimpleNamespace(eos_token_id=163586, eos_token_ids={163585, 163586}, pad_token_id=163839)

    trimmed = scheduler._trim_tokens([11, 163585, 42, 163839], tokenizer)

    assert trimmed == [11]
