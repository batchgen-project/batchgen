"""M2 phase B: per-model runtime adapters reproduce the exact legacy inline
formulas (decode.py / prefill.py) they replace. GPU-free.
"""
import torch

from batchgen.contracts.runtime_adapter import RuntimePhase, RuntimeState
from batchgen.models.deepseek.deepseekv3.runtime_adapter import DeepseekV3RuntimeAdapter
from batchgen.models.openai.gpt_oss_120b.runtime_adapter import GptOssRuntimeAdapter


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Model:
    class model:
        _use_flash_attention_2 = None


# ---- DeepSeek-V3 (MLA): all three behaviors -------------------------------
def test_deepseek_past_kv_byte_size_matches_legacy():
    a = DeepseekV3RuntimeAdapter(_Cfg(compressed_kv_dim=576))
    s = RuntimeState(RuntimePhase.DECODE, None, max_input_length=100, token_idx=5)
    assert a.past_kv_byte_size(s) == (100 + 5 + 1) * 576  # decode.py:388-391


def test_deepseek_position_ids_are_full():
    a = DeepseekV3RuntimeAdapter(_Cfg())
    mask = torch.ones(2, 5, dtype=torch.long)
    out = a.compute_position_ids(RuntimeState(RuntimePhase.DECODE, mask, 5, 0))
    assert tuple(out.shape) == (2, 5)  # full, not last-token (2,1)


def test_deepseek_flash_attention_toggle():
    a = DeepseekV3RuntimeAdapter(_Cfg())
    m = _Model()
    a.configure_attention_backend(m, phase=RuntimePhase.DECODE)
    assert m.model._use_flash_attention_2 is True   # decode.py:230
    a.configure_attention_backend(m, phase=RuntimePhase.PREFILL)
    assert m.model._use_flash_attention_2 is False  # prefill.py:89/261


# ---- GPT-OSS (GQA): KV byte size only -------------------------------------
def test_gpt_oss_past_kv_byte_size_matches_legacy():
    a = GptOssRuntimeAdapter(_Cfg(num_key_value_heads=8, head_dim=64))
    s = RuntimeState(RuntimePhase.DECODE, None, max_input_length=100, token_idx=5)
    assert a.past_kv_byte_size(s) == (100 + 5 + 1) * 8 * 64 * 2  # decode.py:400-407


def test_gpt_oss_position_ids_default_last_token():
    a = GptOssRuntimeAdapter(_Cfg())
    mask = torch.ones(2, 5, dtype=torch.long)
    out = a.compute_position_ids(RuntimeState(RuntimePhase.DECODE, mask, 5, 0))
    assert tuple(out.shape) == (2, 1)  # GQA default: last token only
