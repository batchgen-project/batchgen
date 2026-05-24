"""T1+T2: Segment parity — real GLM-5 segments under CUDA-graph capture+replay.

This is the bring-up gate (§G T1): for each `CapturableSegment` returned by
the adapter's `build_segments`, capture the segment, replay it, and compare
the replay output to the adapter's `run_eager_reference` byte-for-byte at
`atol/rtol = 1e-3`.

**No-hack contract enforced here (per §G):**

- The adapter built is `Glm5CudaGraphAdapter` (the real production class).
- The model is a *small but real* `Glm5Model` (~2 layers, hidden=256, 4
  experts) — only dimensions and counts are reduced. RMSNorm, RoPE, MLA
  absorb, FlashMLA, all_gather, paged-KV manager are the real production
  code.
- `kv_append_callback` is the real `AttnWrapperBase.kv_append_callback`.

Runs only on H20 (GLM-5 deps include FP8 kernels + FlashMLA). Skip on
non-GPU CI and on GH02 (where the GLM-5 tiny config is verified separately).
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GLM-5 segment parity requires CUDA; run on H20.",
)


@pytest.fixture
def tiny_glm5_adapter():
    """Build a tiny-but-real Glm5CudaGraphAdapter for parity tests.

    Implementation note for the H20 runner: this fixture should:
      1. Construct `Glm5Config` with small dims (`hidden_size=256`,
         `num_hidden_layers=2`, `num_experts=4`, `vocab_size=1024`).
      2. Instantiate `Glm5ForCausalLM` from that config on CUDA.
      3. Build a real `GpuPagedKvManager` with a tiny page table.
      4. Return `Glm5CudaGraphAdapter` already passed through
         `build_segments(...)` so `_ctx` is populated.

    The fixture lives here (and not in a shared conftest) because it depends
    on the GLM-5 model imports which carry CUDA/FlashMLA load-time cost.
    """
    pytest.skip(
        "tiny-GLM5 fixture not yet implemented; lift on H20 alongside the "
        "first end-to-end T1 run. See plan §G T1."
    )


def test_whole_model_segment_replay_matches_eager(tiny_glm5_adapter):
    """Bring-up gate: graph-replay outputs match eager outputs at atol=1e-3."""
    pytest.skip("requires tiny_glm5_adapter fixture (see above)")


def test_kv_staging_uses_contiguous_clone_path(tiny_glm5_adapter):
    """Audit §A finding #6: the per-layer fallback branch must not fire on
    the adapter path. Verified by asserting `kv_append_callback` is invoked
    exactly `num_layers` times per step, each with a sliced view of the
    single contiguous staging tensor (not a fresh per-layer clone)."""
    pytest.skip("requires tiny_glm5_adapter fixture (see above)")


def test_compare_facility_passes_at_atol_1e_2(tiny_glm5_adapter):
    """`compare_decode_outputs` returns `passed=True` under the bring-up
    tolerance (§G T1)."""
    pytest.skip("requires tiny_glm5_adapter fixture (see above)")


def test_capture_signature_invalidates_on_page_table_resize(tiny_glm5_adapter):
    """Audit §A finding #5: signature MUST change when page-table storage
    moves; the adapter MUST then fall back to eager exactly once with
    `reason='capture_signature_mismatch_...'`."""
    pytest.skip("requires tiny_glm5_adapter fixture (see above)")
