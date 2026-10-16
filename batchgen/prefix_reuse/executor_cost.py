"""Offline linear cost model for one prefill executor's time.

An executor is one serve group: a single rank under pure DP, or the G ranks
that prefill a group's sequences in TP-G lockstep. A profile turns the four
features of ``wave_plan.prefill_plan_work`` into that executor's predicted
seconds. The coefficients are a measured fit, so a profile is only valid for
the exact model, machine and group it was fitted on; lookup is therefore
explicit and returns ``None`` rather than falling back.
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
    # Ranks per executor: 1 is a pure-DP rank, 8 one K3 serve group. A group's
    # seconds change with its TP width, so this always matches exactly.
    group_size: int
    # The world the fit holds for. Set, it matches exactly: an executor whose
    # expert shards change with the world size (GPT-OSS) must not transfer.
    # ``None`` matches any world: the fit was validated group-local.
    world_size: Optional[int]
    fitted_world_size: int  # provenance only; never matched
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
        group_size=1,
        world_size=4,
        fitted_world_size=4,
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
    # Kimi-Linear-48B-A3B, one 8-rank serve group of a two-node world, fitted
    # from the dual-node gate ledger run; see internal records for the samples,
    # the residual report and the identifiability analysis.
    #
    # ``per_chunk`` is pinned to zero rather than fitted: every pooled wave the
    # fit could observe ran as a single chunk, so that column was an exact
    # multiple of the intercept and no coefficient is identifiable for it. A
    # multi-chunk pooled wave needs computed tokens above the planner's prefill
    # token cap, which prefix sharing prevents — a wave with enough sharing to
    # enter the pool saves most of its prompt tokens, and a wave without
    # sharing declines the pool altogether. So the profile's validity domain is
    # the single-chunk pooled wave, which is this deployment's actual domain,
    # and a deployment that does cut pooled waves into chunks must re-fit
    # before trusting it.
    #
    # ``world_size`` is exact: the fit saw one group count only, so the
    # per-group-cost-independence check that a ``None`` world claims was not
    # run. The computed-tokens and attention-pairs columns are strongly
    # correlated in the fitted samples, so neither coefficient is trustworthy
    # alone; the packer consumes only their joint prediction.
    ExecutorCostProfile(
        model="kimi-linear-48b-a3b",
        machine="sm90-141g",
        group_size=8,
        world_size=16,
        fitted_world_size=16,
        compute_capability=(9, 0),
        memory_gib=(130.0, 150.0),
        intercept=5.186165427267378,
        per_1e5_computed_tokens=1.7899117634244257,
        per_1e9_attention_pairs=1.5830601489530896,
        per_100_item_rows=0.9683259814970133,
        per_chunk=0.0,
    ),
)


def find_executor_cost_profile(
    model_name: Optional[str],
    compute_capability: tuple[int, int],
    total_memory_bytes: int,
    world_size: int,
    group_size: int,
) -> Optional[ExecutorCostProfile]:
    """Match a profile by exact model spec, device properties, and layout.

    Device marketing names vary by vendor SKU and firmware, so the machine is
    identified by compute capability and total memory instead of by name. A
    profile fitted on one parallel layout must not score another, and a
    smaller model sharing a name prefix must not inherit a larger model's
    coefficients, so both match exactly; no match returns ``None`` and the
    packer balances computed tokens instead.

    ``group_size`` always matches exactly. ``world_size`` matches exactly
    unless the profile leaves it ``None``, which claims its group's seconds do
    not depend on how many groups the world holds. Such a group-local profile
    may only land together with the residual check that shows per-group cost
    independent of group count, and any scale-out beyond the group counts that
    check covered re-runs it before trusting the profile.
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
            and group_size == profile.group_size
            and (profile.world_size is None or world_size == profile.world_size)
        ):
            return profile
    return None
