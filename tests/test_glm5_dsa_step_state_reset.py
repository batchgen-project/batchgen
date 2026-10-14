"""The per-step DSA prev-topk clear must run regardless of the index_topk hint.

Regression for the eager (second) decode path: with ``index_topk`` present in
the model config (GLM-5.2), a clear nested under the hint's else-branch never
executes, so shared reuse-topk layers can consume a stale top-k tensor from
the previous decode step.
"""

import torch

from batchgen.batchgen_worker import _reset_glm5_dsa_step_state


class _Wrapper:
    _dsa_short_count = "sentinel"
    _dsa_prev_topk_indices = "stale"


def test_clear_runs_with_index_topk_set():
    _Wrapper._dsa_prev_topk_indices = "stale"
    cache_seqlens = torch.tensor([100, 3000, 2048], dtype=torch.int32)
    _reset_glm5_dsa_step_state(_Wrapper, cache_seqlens, 2048)
    assert _Wrapper._dsa_short_count == 2  # 100 and 2048 are <= 2048
    assert _Wrapper._dsa_prev_topk_indices is None


def test_clear_runs_without_index_topk():
    _Wrapper._dsa_prev_topk_indices = "stale"
    cache_seqlens = torch.tensor([100], dtype=torch.int32)
    _reset_glm5_dsa_step_state(_Wrapper, cache_seqlens, None)
    assert _Wrapper._dsa_short_count is None
    assert _Wrapper._dsa_prev_topk_indices is None
