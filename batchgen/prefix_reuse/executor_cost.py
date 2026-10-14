"""Offline linear cost model for one DP rank's prefill executor time.

A profile turns the four features of ``wave_plan.prefill_plan_work`` into
predicted executor seconds. The coefficients are a measured fit, so a profile
is only valid for the exact model and machine it was fitted on; lookup is
therefore explicit and returns ``None`` rather than falling back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ExecutorCostProfile:
    model: str
    machine: str  # label only; matching uses the device properties below
    compute_capability: tuple[int, int]
    memory_gib: tuple[float, float]  # inclusive range of total device memory
    # The fit holds only for the parallel layout it was measured on: a rank's
    # executor seconds change with its expert/attention shard sizes.
    world_size: int
    intercept: float
    per_1e5_computed_tokens: float
    per_1e9_attention_pairs: float
    per_100_item_rows: float
    per_chunk: float

    def seconds(
        self,
        computed_tokens: int,
        attention_pairs: int,
        item_rows: int,
        chunks: int,
    ) -> float:
        return (
            self.intercept
            + self.per_1e5_computed_tokens * computed_tokens / 100_000
            + self.per_1e9_attention_pairs * attention_pairs / 1_000_000_000
            + self.per_100_item_rows * item_rows / 100
            + self.per_chunk * chunks
        )


# Empirical GPT-OSS-120B prefill executor fit (sm_90, ~141 GB device,
# 4-rank world). The pair term
# encodes full causal attention, so this profile must not be transferred
# to another model: GLM-5.2 DSA runs a sparse indexer instead and its cost has
# no full-attention pair term at all.
PROFILES: tuple[ExecutorCostProfile, ...] = (
    ExecutorCostProfile(
        model="gpt-oss-120b",
        machine="sm90-141g",
        world_size=4,
        # sm_90 with ~141 GB device memory. Other sm_90 parts carry 80 or
        # 96 GB and fall outside this memory range, so they never match.
        compute_capability=(9, 0),
        memory_gib=(130.0, 150.0),
        intercept=-0.14581837148589716,
        per_1e5_computed_tokens=6.761851967927209,
        per_1e9_attention_pairs=0.6507025875903313,
        per_100_item_rows=0.3697124295421248,
        per_chunk=0.1405919218614613,
    ),
)


def find_executor_cost_profile(
    model_name: Optional[str],
    compute_capability: tuple[int, int],
    total_memory_bytes: int,
    world_size: int,
) -> Optional[ExecutorCostProfile]:
    """Match a profile by exact model spec, device properties, and world size.

    Device marketing names vary by vendor SKU and firmware, so the machine is
    identified by compute capability and total memory instead of by name. A
    profile fitted on one parallel layout must not score another, and a
    smaller model sharing a name prefix must not inherit a larger model's
    coefficients, so both match exactly; no match returns ``None`` and the
    packer balances computed tokens instead.
    """

    # Checkpoint ids carry an org prefix ("openai/gpt-oss-120b"); the profile
    # key is the exact model basename.
    model = (model_name or "").lower().rsplit("/", 1)[-1]
    memory_gib = total_memory_bytes / (1 << 30)
    for profile in PROFILES:
        low, high = profile.memory_gib
        if (
            profile.model.lower() == model
            and tuple(compute_capability) == profile.compute_capability
            and low <= memory_gib <= high
            and world_size == profile.world_size
        ):
            return profile
    return None
