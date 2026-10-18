"""Bitwise guard on `sample_tokens` across the per-request sampling matrix.

The decode step's host path is being cleaned up in slices; several of those
slices move work around the sampled-token readback. The sampled tokens
themselves are the one thing that must never move, so this pins the whole
per-request matrix — all-greedy, all-sampling, mixed greedy/sampling,
per-row ``top_k > 0``, per-row ``top_p < 1``, and the empty batch — in two
independent ways:

* a fixed-seed golden token vector per case, recorded from the build before
  the host-path slices started (``GOLDEN`` below), and
* an admissible-set check that recomputes each row's candidate set from the
  parameters and asserts the sampled token is in it, which also catches a
  golden vector that was regenerated to match a behaviour change.

The golden vectors depend on the RNG stream, so they are only asserted on the
torch version they were recorded with; the structural checks always run.
"""
import pytest
import torch
import torch.nn.functional as F

from batchgen.sampling import sample_tokens
from batchgen.worker.sampling_residency import host_decisions_from_cast

SEED = 20261017
GOLDEN_TORCH_VERSION = "2.13.0"

# Recorded on the slice-I base commit (origin/main 47c40d82) with
# torch 2.13.0, CPU: `sample_tokens` is unmodified by that slice, so these are
# the pre-slice tokens.
GOLDEN = {
    "greedy_temperature_none": [9, 13, 23, 12],
    "greedy_temperature_zero": [9, 13, 23, 12],
    "greedy_per_row_zero": [9, 13, 23, 12],
    "sampling_temp1_scalar": [28, 19, 1, 13],
    "sampling_temp_per_row": [28, 19, 1, 13],
    "mixed_greedy_sampling": [9, 8, 23, 19],
    "top_k_scalar": [28, 13, 1, 12],
    "top_k_per_row": [9, 13, 1, 13],
    "top_p_scalar": [28, 13, 1, 13],
    "top_p_per_row": [9, 13, 1, 13],
    "top_k_and_top_p_mixed": [9, 26, 23, 4],
}

B, V = 4, 32


def _logits():
    generator = torch.Generator().manual_seed(4242)
    return torch.randn(B, V, generator=generator, dtype=torch.float32)


def _cases():
    """Name -> kwargs for sample_tokens, over the whole parameter matrix."""
    return {
        "greedy_temperature_none": dict(temperature=None),
        "greedy_temperature_zero": dict(temperature=0.0),
        "greedy_per_row_zero": dict(temperature=torch.zeros(B)),
        "sampling_temp1_scalar": dict(temperature=1.0),
        "sampling_temp_per_row": dict(
            temperature=torch.tensor([0.5, 1.0, 1.5, 2.0])
        ),
        "mixed_greedy_sampling": dict(
            temperature=torch.tensor([0.0, 1.0, 0.0, 0.8])
        ),
        "top_k_scalar": dict(temperature=1.0, top_k=3),
        "top_k_per_row": dict(
            temperature=1.0, top_k=torch.tensor([1, 4, 0, 8])
        ),
        "top_p_scalar": dict(temperature=1.0, top_p=0.5),
        "top_p_per_row": dict(
            temperature=1.0, top_p=torch.tensor([0.1, 0.5, 0.9, 1.0])
        ),
        "top_k_and_top_p_mixed": dict(
            temperature=torch.tensor([0.0, 1.0, 0.7, 1.3]),
            top_p=torch.tensor([1.0, 0.6, 0.95, 0.3]),
            top_k=torch.tensor([0, 5, 2, 0]),
        ),
    }


def compute_matrix():
    """Sampled tokens for every case, each under the same fixed seed."""
    logits = _logits()
    out = {}
    for name, kwargs in _cases().items():
        torch.manual_seed(SEED)
        out[name] = sample_tokens(logits, **kwargs).reshape(-1).tolist()
    return out


def _row_params(kwargs, row):
    temperature = kwargs.get("temperature")
    if temperature is None:
        temp = 0.0
    elif isinstance(temperature, torch.Tensor):
        temp = float(temperature[row])
    else:
        temp = float(temperature)

    top_p = kwargs.get("top_p")
    if top_p is None:
        p = 1.0
    elif isinstance(top_p, torch.Tensor):
        p = float(top_p[row])
    else:
        p = float(top_p)

    top_k = kwargs.get("top_k", 0)
    if isinstance(top_k, torch.Tensor):
        k = int(top_k[row])
    else:
        k = int(top_k)
    return temp, p, k


def _admissible(logits_row, temp, p, k):
    """Token ids `sample_tokens` is allowed to return for one row.

    Mirrors the implementation's op order so float32 ties resolve the same
    way: temperature scale, rank filter, then the shifted nucleus mask that
    always keeps the top token.
    """
    scaled = logits_row / temp
    sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
    keep = torch.ones(scaled.numel(), dtype=torch.bool)
    if k > 0:
        keep &= torch.arange(scaled.numel()) < min(k, scaled.numel())
    if p < 1.0:
        filtered = sorted_logits.clone()
        if k > 0:
            filtered[min(k, scaled.numel()):] = float("-inf")
        cumulative = F.softmax(filtered, dim=-1).cumsum(dim=-1)
        drop = cumulative > p
        drop[1:] = drop[:-1].clone()
        drop[0] = False
        keep &= ~drop
    return {int(sorted_indices[i]) for i in range(scaled.numel()) if keep[i]}


def test_matrix_is_reproducible_under_a_fixed_seed():
    assert compute_matrix() == compute_matrix()


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_matrix_matches_the_recorded_pre_slice_tokens(name):
    if torch.__version__.split("+")[0] != GOLDEN_TORCH_VERSION:
        pytest.skip(
            f"golden tokens were recorded on torch {GOLDEN_TORCH_VERSION}, "
            f"running {torch.__version__}"
        )
    assert compute_matrix()[name] == GOLDEN[name]


def test_the_matrix_covers_every_case_that_has_a_golden():
    assert sorted(_cases()) == sorted(GOLDEN)


def test_greedy_rows_are_argmax_rows():
    logits = _logits()
    expected = logits.float().argmax(dim=-1).tolist()
    for name, kwargs in _cases().items():
        torch.manual_seed(SEED)
        tokens = sample_tokens(logits, **kwargs).reshape(-1).tolist()
        for row in range(B):
            temp, _, _ = _row_params(kwargs, row)
            if temp <= 0:
                assert tokens[row] == expected[row], (
                    f"{name} row {row} is greedy and must be argmax"
                )


def test_sampled_rows_stay_inside_their_admissible_set():
    logits = _logits()
    for name, kwargs in _cases().items():
        torch.manual_seed(SEED)
        tokens = sample_tokens(logits, **kwargs).reshape(-1).tolist()
        for row in range(B):
            temp, p, k = _row_params(kwargs, row)
            if temp <= 0:
                continue
            allowed = _admissible(logits[row], temp, p, k)
            assert tokens[row] in allowed, (
                f"{name} row {row} sampled {tokens[row]} outside "
                f"top_k={k} top_p={p}"
            )


def test_shape_and_dtype_hold_for_every_case():
    logits = _logits()
    for name, kwargs in _cases().items():
        torch.manual_seed(SEED)
        tokens = sample_tokens(logits, **kwargs)
        assert tokens.shape == (B, 1), name
        assert tokens.dtype == torch.long, name


def test_an_empty_batch_samples_no_tokens():
    empty = torch.zeros(0, V, dtype=torch.float32)
    for kwargs in (
        dict(temperature=None),
        dict(temperature=1.0),
        dict(temperature=1.0, top_k=4, top_p=0.5),
        dict(
            temperature=torch.zeros(0),
            top_k=torch.zeros(0, dtype=torch.int64),
            top_p=torch.zeros(0),
        ),
    ):
        torch.manual_seed(SEED)
        tokens = sample_tokens(empty, **kwargs)
        assert tokens.shape == (0, 1)
        assert tokens.dtype == torch.long


def _host_decisions_for(kwargs):
    """The four branch decisions `sample_tokens` would reduce on device.

    The logits in this file are fp32, so `.to(logits.dtype)` is the identity
    and the fp32 row values are already the cast values. (Where the cast is
    lossy -- bf16 logits -- the agreement between the host predicate and the
    device mask is pinned in `test_sampling_residency.py`.)
    """
    temps, top_ps, top_ks = [], [], []
    for row in range(B):
        temp, p, k = _row_params(kwargs, row)
        temps.append(temp)
        top_ps.append(p)
        top_ks.append(k)
    return host_decisions_from_cast(temps, top_ps, top_ks)


def _reaches_the_branches(kwargs):
    """False for the scalar greedy short-circuit, which returns before them."""
    temperature = kwargs.get("temperature")
    if temperature is None:
        return False
    if isinstance(temperature, torch.Tensor):
        return True
    return temperature > 0


def test_host_decided_branches_reproduce_the_matrix_token_for_token():
    """The decode step path hands `sample_tokens` host-side branch decisions
    instead of letting it reduce `greedy_mask`, `sampling_mask`,
    `s_top_ks.max()` and `(s_top_ps < 1.0)` on device, and selects the
    invalid-probability fallback on device instead of branching on host. For
    every case in this matrix that must be the same token vector.
    """
    logits = _logits()
    for name, kwargs in _cases().items():
        if not _reaches_the_branches(kwargs):
            continue
        torch.manual_seed(SEED)
        expected = sample_tokens(logits, **kwargs).reshape(-1).tolist()
        torch.manual_seed(SEED)
        got = sample_tokens(
            logits, host_decisions=_host_decisions_for(kwargs), **kwargs
        ).reshape(-1).tolist()
        assert got == expected, (
            f"{name}: host-decided {got} != device-reduced {expected}"
        )


def test_the_host_decided_path_agrees_with_the_recorded_pre_slice_tokens():
    if torch.__version__.split("+")[0] != GOLDEN_TORCH_VERSION:
        pytest.skip(
            f"golden tokens were recorded on torch {GOLDEN_TORCH_VERSION}, "
            f"running {torch.__version__}"
        )
    logits = _logits()
    for name, kwargs in _cases().items():
        if not _reaches_the_branches(kwargs):
            continue
        torch.manual_seed(SEED)
        got = sample_tokens(
            logits, host_decisions=_host_decisions_for(kwargs), **kwargs
        ).reshape(-1).tolist()
        assert got == GOLDEN[name], name


def test_a_single_row_batch_is_sampled_like_its_row_in_a_batch():
    # Per-row params must not leak across rows: a one-row batch carrying row
    # 1's params must land in row 1's admissible set.
    logits = _logits()
    kwargs = _cases()["top_k_per_row"]
    temp, p, k = _row_params(kwargs, 1)
    torch.manual_seed(SEED)
    token = sample_tokens(
        logits[1:2],
        temperature=temp,
        top_k=k,
        top_p=p,
    ).reshape(-1).tolist()[0]
    assert token in _admissible(logits[1], temp, p, k)
