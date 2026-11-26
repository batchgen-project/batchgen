"""K3 fused-KDA-decode coverage is explained and recorded, never silent.

Pure-CPU: `coverage_gap` is shape/dtype/stride arithmetic, and the
serving-module switch is driven with SimpleNamespace modules; no CUDA or
extension load is needed (the extension-missing path on CPU is itself one of
the recorded outcomes).
"""
import logging
from types import SimpleNamespace

import pytest
import torch

from batchgen_kernels.attention.kda_fused_decode import coverage_gap, covered
from batchgen.models.moonshotai.kimi_linear import serving_modules

HEADS = 3            # smallest supported local head count (TP32)
HEAD_DIM = 128
SEGMENT = HEADS * HEAD_DIM
BATCH = 2
SLOTS = 4


def _covered_args():
    mixed_qkv = torch.zeros(BATCH, 3 * SEGMENT, dtype=torch.bfloat16)
    forget_gate = torch.zeros(BATCH, SEGMENT, dtype=torch.bfloat16)
    beta = torch.zeros(BATCH, HEADS, dtype=torch.bfloat16)
    conv = lambda: torch.zeros(SLOTS, SEGMENT, 3, dtype=torch.bfloat16)
    weight = lambda: torch.zeros(SEGMENT, 4, dtype=torch.float32)
    a_log = torch.zeros(HEADS, dtype=torch.float32)
    dt_bias = torch.zeros(SEGMENT, dtype=torch.float32)
    onorm_gate = torch.zeros(BATCH, SEGMENT, dtype=torch.bfloat16)
    onorm_weight = torch.zeros(HEAD_DIM, dtype=torch.float32)
    state = torch.zeros(SLOTS, HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.float32)
    state_indices = torch.zeros(BATCH, dtype=torch.int32)
    return dict(
        mixed_qkv=mixed_qkv, forget_gate=forget_gate, beta=beta,
        conv_q=conv(), conv_k=conv(), conv_v=conv(),
        weight_q=weight(), weight_k=weight(), weight_v=weight(),
        bias_q=None, bias_k=None, bias_v=None,
        a_log=a_log, dt_bias=dt_bias, onorm_gate=onorm_gate,
        onorm_weight=onorm_weight, state=state, state_indices=state_indices,
    )


def test_native_k3_tensors_are_covered():
    args = _covered_args()
    assert coverage_gap(**args) is None
    assert covered(**args) is True


@pytest.mark.parametrize("mutate,expect", [
    (lambda a: a.update(mixed_qkv=a["mixed_qkv"].float()),
     "mixed_qkv dtype"),
    (lambda a: a.update(forget_gate=a["forget_gate"].float()),
     "forget_gate dtype"),
    (lambda a: a.update(weight_q=a["weight_q"].to(torch.bfloat16)),
     "weight_q dtype"),
    (lambda a: a.update(state_indices=a["state_indices"].long()),
     "state_indices dtype"),
    (lambda a: a.update(beta=torch.zeros(BATCH, HEADS + 1,
                                         dtype=torch.bfloat16)),
     "beta shape"),
    (lambda a: a.update(
        mixed_qkv=torch.zeros(BATCH, 2 * 3 * SEGMENT,
                              dtype=torch.bfloat16)[:, ::2]),
     "mixed_qkv last-axis stride"),
    (lambda a: a.update(
        state=torch.zeros(SLOTS, HEADS, HEAD_DIM, 2 * HEAD_DIM,
                          dtype=torch.float32)[..., :HEAD_DIM]),
     "state inner strides"),
])
def test_coverage_gap_names_the_failing_check(mutate, expect):
    args = _covered_args()
    mutate(args)
    gap = coverage_gap(**args)
    assert gap is not None and expect in gap, gap
    assert covered(**args) is False


# ------------------------------------------------ serving-module switch (C5)

def _fresh_status(monkeypatch):
    monkeypatch.setattr(
        serving_modules, "_KDA_FUSED_DECODE_STATUS",
        {"outcome": None, "reason": None},
    )
    monkeypatch.setattr(serving_modules, "_KDA_FUSED_COVERAGE_WARNED", False)
    monkeypatch.setattr(
        serving_modules, "_KDA_FUSED_DECODE_IMPORT_FAILED", False)


def _k3_module(conv_dtype=torch.float32):
    conv = lambda: SimpleNamespace(
        weight=torch.zeros(SEGMENT, 1, 4, dtype=conv_dtype), bias=None)
    return SimpleNamespace(
        use_full_rank_gate=True,
        q_conv1d=conv(), k_conv1d=conv(), v_conv1d=conv(),
        A_log=torch.zeros(HEADS, dtype=torch.float32),
        dt_bias=torch.zeros(SEGMENT, dtype=torch.float32),
        o_norm=SimpleNamespace(
            weight=torch.zeros(HEAD_DIM, dtype=torch.float32),
            variance_epsilon=1e-6,
        ),
        head_dim=HEAD_DIM,
        gate_lower_bound=0.0,
    )


def _kda_state():
    return SimpleNamespace(
        conv_q=torch.zeros(SLOTS, SEGMENT, 3, dtype=torch.bfloat16),
        conv_k=torch.zeros(SLOTS, SEGMENT, 3, dtype=torch.bfloat16),
        conv_v=torch.zeros(SLOTS, SEGMENT, 3, dtype=torch.bfloat16),
        recurrent_pool=torch.zeros(
            SLOTS, HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.float32),
    )


def test_non_k3_records_the_fla_outcome_without_warning(monkeypatch, caplog):
    _fresh_status(monkeypatch)
    module = _k3_module()
    module.use_full_rank_gate = False
    with caplog.at_level(logging.WARNING, logger=serving_modules.__name__):
        out = serving_modules._kda_fused_decode(
            module, None, None, None, None, _kda_state(),
            torch.zeros(BATCH, dtype=torch.int32),
        )
    assert out is None
    assert serving_modules._KDA_FUSED_DECODE_STATUS["outcome"] == "fla-non-k3"
    assert not caplog.records


def test_coverage_miss_warns_once_naming_the_check(monkeypatch, caplog):
    _fresh_status(monkeypatch)
    module = _k3_module()
    state = _kda_state()
    slot_ids = torch.zeros(BATCH, dtype=torch.int32)
    mixed_qkv = torch.zeros(BATCH, 3 * SEGMENT, dtype=torch.bfloat16)
    # fp32 forget gate: covered() requires bf16, so the switch must miss.
    f = torch.zeros(BATCH, HEADS, HEAD_DIM, dtype=torch.float32)
    beta = torch.zeros(BATCH, HEADS, dtype=torch.bfloat16)
    z = torch.zeros(BATCH, HEADS, HEAD_DIM, dtype=torch.bfloat16)
    with caplog.at_level(logging.WARNING, logger=serving_modules.__name__):
        for _ in range(2):
            out = serving_modules._kda_fused_decode(
                module, mixed_qkv, f, beta, z, state, slot_ids)
            assert out is None
    warnings = [r for r in caplog.records if "coverage miss" in r.getMessage()]
    assert len(warnings) == 1
    assert "forget_gate dtype" in warnings[0].getMessage()
    status = serving_modules._KDA_FUSED_DECODE_STATUS
    assert status["outcome"] == "fla-coverage-miss"
    assert "forget_gate dtype" in status["reason"]


def test_conv_layout_miss_names_the_convs(monkeypatch, caplog):
    _fresh_status(monkeypatch)
    module = _k3_module(conv_dtype=torch.bfloat16)  # kernel needs fp32
    mixed_qkv = torch.zeros(BATCH, 3 * SEGMENT, dtype=torch.bfloat16)
    f = torch.zeros(BATCH, HEADS, HEAD_DIM, dtype=torch.bfloat16)
    beta = torch.zeros(BATCH, HEADS, dtype=torch.bfloat16)
    z = torch.zeros(BATCH, HEADS, HEAD_DIM, dtype=torch.bfloat16)
    with caplog.at_level(logging.WARNING, logger=serving_modules.__name__):
        out = serving_modules._kda_fused_decode(
            module, mixed_qkv, f, beta, z, _kda_state(),
            torch.zeros(BATCH, dtype=torch.int32))
    assert out is None
    warnings = [r for r in caplog.records if "coverage miss" in r.getMessage()]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "q_conv1d" in message and "v_conv1d" in message


@pytest.mark.parametrize("outcome,reason,level,text", [
    ("fused", None, logging.INFO, "coverage: PASS"),
    ("fla-non-k3", "no full-rank gate", logging.INFO, "FLA chain"),
    (None, None, logging.WARNING, "not exercised"),
    ("fla-coverage-miss", "beta shape (2, 4) != (2, 3)", logging.WARNING,
     "coverage: MISS"),
])
def test_startup_coverage_log_states_the_outcome(
    monkeypatch, caplog, outcome, reason, level, text
):
    _fresh_status(monkeypatch)
    serving_modules._KDA_FUSED_DECODE_STATUS.update(
        outcome=outcome, reason=reason)
    with caplog.at_level(logging.INFO, logger=serving_modules.__name__):
        serving_modules.log_kda_fused_decode_startup_coverage(rank=3)
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == level
    assert text in record.getMessage()
    assert "rank=3" in record.getMessage()


def test_the_first_outcome_wins(monkeypatch):
    _fresh_status(monkeypatch)
    serving_modules._record_kda_fused_decode_outcome("fused", None)
    serving_modules._record_kda_fused_decode_outcome("fla-coverage-miss", "x")
    assert serving_modules._KDA_FUSED_DECODE_STATUS["outcome"] == "fused"
