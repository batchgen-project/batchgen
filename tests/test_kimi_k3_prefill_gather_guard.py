"""Regression for the K3 prefill last-token gather canary.

`KimiLinearParallelStrategyManager.gather_prefill_last_token_hidden` fills the
rows it *owns* into a per-sequence ``selected`` tensor, then checks that those
rows carry a real hidden state (a canary for the s4/s5 gather bug, an int32
row-offset wrap since fixed). The check must compare against the ``owned`` mask
directly. The previous form sampled a fixed trailing window with
``range(-8, 0)``, which missed non-tail drops and, worse, raised ``IndexError``
when a prefill microbatch carried fewer than 8 sequences — and because the check
runs on every rank, one raise brought the whole distributed server down.
"""
import torch


def _count_owned_all_zero(selected: torch.Tensor, owned: torch.Tensor) -> int:
    # Mirrors the canary in gather_prefill_last_token_hidden.
    if not bool(owned.any()):
        return 0
    return int((selected[owned].abs().sum(dim=-1) == 0).sum().item())


def test_canary_counts_only_all_zero_owned_rows():
    selected = torch.zeros(4, 3)
    selected[1] = 1.0
    selected[3] = 2.0
    owned = torch.tensor([True, True, False, True])  # rows 0,1,3 owned; row 0 is all-zero
    assert _count_owned_all_zero(selected, owned) == 1  # only owned row 0


def test_canary_zero_when_all_owned_rows_nonzero():
    selected = torch.arange(1, 13, dtype=torch.float32).reshape(4, 3)
    owned = torch.ones(4, dtype=torch.bool)
    assert _count_owned_all_zero(selected, owned) == 0


def test_canary_handles_small_and_empty_batches():
    # <8 sequences (the case the old range(-8, 0) form crashed on) and no owned rows.
    for n in (1, 2, 7):
        selected = torch.zeros(n, 3)
        owned = torch.zeros(n, dtype=torch.bool)
        assert _count_owned_all_zero(selected, owned) == 0  # nothing owned -> no anomaly


def test_old_range_form_would_have_failed_below_eight():
    # Documents the exact previous failure: range(-8, 0) indexing a 2-row tensor.
    selected = torch.zeros(2, 3)
    with __import__("pytest").raises(IndexError):
        [selected[i] for i in range(-8, 0)]
