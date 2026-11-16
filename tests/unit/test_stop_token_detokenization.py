"""The worker decodes a completion through its stop token, for every tokenizer.

``BatchGenWorker._decode_tokens_to_string`` hands the stop token a sequence
ended on to ``decode(skip_special_tokens=True)`` together with the completion,
and the tokenizer decides whether it shows. Before, the worker sliced the stop
token off unless ``--detokenization-include-special-tokens`` was set, which
hid GPT-OSS ``<|call|>`` / ``<|return|>`` from the harmony parser (it could
not tell a finished tool call from a ``max_tokens`` truncation).

The change is a no-op for a tokenizer whose decode drops every one of its stop
tokens under ``skip_special_tokens=True``. This file checks that, through the
real worker method, for every registered tokenizer and every stop id the
worker uses (``eos_token_ids``, else ``{eos_token_id}``): only the stop
tokens in ``KEPT_STOP_TOKENS`` (output-grammar terminators) may appear in the
text. A newly registered tokenizer is covered automatically.
"""

import pytest
import tiktoken
import torch

from batchgen.config import tokenizer_assets as ta
from batchgen.config import tokenizer_registry as registry
from batchgen.models.openai.gpt_oss_120b import tokenizer as gpt_oss_tok
from tests.worker_method_source import load_worker_methods

_DECODE = load_worker_methods(
    ["_decode_tokens_to_string"], {"torch": torch}
)["_decode_tokens_to_string"]

#: Stop tokens that are part of a model's output grammar and so stay in the
#: completion text. Every other stop token must decode to nothing.
KEPT_STOP_TOKENS = {
    "gpt_oss": {"<|return|>", "<|call|>"},
}

BODIES = (
    "Hello there, 42 is the answer.",
    "Mixed text: 中文, émoji 🙂, and code `x = 1`.",
)


def _gpt_oss_byte_level_encoding() -> tiktoken.Encoding:
    """Byte-level ranks with the real GPT-OSS special tokens.

    Decode filtering is by token id, so the real o200k ranks (an installer
    staged asset) are not needed to check which stop ids survive.
    """
    return tiktoken.Encoding(
        name="gpt_oss_test",
        pat_str=ta.O200K_BASE.pat_str,
        mergeable_ranks={bytes([b]): b for b in range(256)},
        special_tokens={
            **gpt_oss_tok.GPT_OSS_SPECIAL_TOKENS,
            **{f"<|reserved_{i}|>": i for i in range(200013, 201088)},
        },
    )


def _load(tokenizer_type, monkeypatch):
    if tokenizer_type == "gpt_oss":
        monkeypatch.setattr(
            gpt_oss_tok, "_get_tiktoken_tokenizer", _gpt_oss_byte_level_encoding
        )
    return registry.TOKENIZER_REGISTRY[tokenizer_type]()


def _worker(tok):
    """The fields the worker sets from its tokenizer (load_model_and_tokenizer)."""

    class _Worker:
        pass

    worker = _Worker()
    worker.tokenizer = tok
    worker.eos_token_ids = getattr(tok, "eos_token_ids", {tok.eos_token_id})
    worker.pad_token_id = getattr(tok, "pad_token_id", 0)
    worker.detokenization_include_special_tokens = False
    return worker


@pytest.mark.parametrize("retired", ["DeepSeek-V2-Lite", "DeepSeek-V2",
                                     "Mixtral-8x22B", "Mixtral-8x7B"])
def test_retired_tokenizer_patterns_no_longer_resolve(retired):
    # Their modules never existed: resolution must fail loud at
    # load_tokenizer's clear no-tokenizer-registered message, and the
    # registry warm-up must not retry (and ERROR-log) a dead import.
    assert registry.resolve_tokenizer_type(retired) is None


def test_every_declared_tokenizer_type_is_registered():
    # Every declared module must exist and register its type: a declared
    # module that cannot import puts an ERROR line into every server start.
    declared = set(registry._TOKENIZER_MODULES)
    registered = set(registry.TOKENIZER_REGISTRY)
    assert declared - registered == set()
    assert set(KEPT_STOP_TOKENS) <= registered


@pytest.mark.parametrize("tokenizer_type", sorted(registry.TOKENIZER_REGISTRY))
def test_stop_token_shows_only_when_it_is_output_grammar(tokenizer_type, monkeypatch):
    tok = _load(tokenizer_type, monkeypatch)
    worker = _worker(tok)
    stop_ids = sorted(worker.eos_token_ids)
    assert stop_ids, tokenizer_type
    kept = KEPT_STOP_TOKENS.get(tokenizer_type, set())
    seen_kept = set()
    for body_text in BODIES:
        body = tok.encode(body_text, add_special_tokens=False)
        assert not set(body) & set(stop_ids), (tokenizer_type, body_text)
        without_stop = tok.decode(body, skip_special_tokens=True)
        for stop_id in stop_ids:
            # Trajectory buffer: completion, the stop token, then padding.
            buffer = torch.tensor(
                body + [stop_id] + [worker.pad_token_id] * 3, dtype=torch.int64
            )
            text = _DECODE(worker, buffer)
            stop_text = tok.decode([stop_id], skip_special_tokens=False)
            if stop_text in kept:
                seen_kept.add(stop_text)
                assert text == without_stop + stop_text, (tokenizer_type, stop_id)
            else:
                # The old worker sliced the stop token off: same text.
                assert text == without_stop, (tokenizer_type, stop_id, text)
    assert seen_kept == kept, tokenizer_type


@pytest.mark.parametrize("tokenizer_type", sorted(registry.TOKENIZER_REGISTRY))
def test_include_special_tokens_shows_every_stop_token(tokenizer_type, monkeypatch):
    tok = _load(tokenizer_type, monkeypatch)
    worker = _worker(tok)
    worker.detokenization_include_special_tokens = True
    body = tok.encode(BODIES[0], add_special_tokens=False)
    for stop_id in sorted(worker.eos_token_ids):
        buffer = torch.tensor(body + [stop_id, worker.pad_token_id], dtype=torch.int64)
        assert _DECODE(worker, buffer) == tok.decode(
            body + [stop_id], skip_special_tokens=False
        ), (tokenizer_type, stop_id)
