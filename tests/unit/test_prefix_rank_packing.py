"""CPU-only tests for the per-wave DP rank packer.

The prefix modules are pure Python, but ``batchgen/__init__`` is not, so each
module is loaded from its file under its canonical dotted name (rank_packing
imports its siblings absolutely) and ``sys.modules`` is restored afterwards.
"""

import importlib.util
import os
import random
import subprocess
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[2]
_NAMES = (
    "batchgen",
    "batchgen.prefix_reuse",
    "batchgen.prefix_reuse.wave_plan",
    "batchgen.prefix_reuse.executor_cost",
    "batchgen.prefix_reuse.rank_packing",
)


def _load_prefix_modules():
    saved = {name: sys.modules.get(name) for name in _NAMES}
    try:
        for name in ("batchgen", "batchgen.prefix_reuse"):
            if sys.modules.get(name) is None:
                package = types.ModuleType(name)
                package.__path__ = [str(_ROOT.joinpath(*name.split(".")))]
                sys.modules[name] = package
        loaded = {}
        for stem in ("wave_plan", "executor_cost", "rank_packing"):
            name = f"batchgen.prefix_reuse.{stem}"
            path = _ROOT / "batchgen" / "prefix_reuse" / f"{stem}.py"
            spec = importlib.util.spec_from_file_location(name, path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[stem] = module
        return loaded
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


_MODULES = _load_prefix_modules()

plan_wave_prefix_sharing = _MODULES["wave_plan"].plan_wave_prefix_sharing
validate_wave_prefix_plan = _MODULES["wave_plan"].validate_wave_prefix_plan
prefill_plan_work = _MODULES["wave_plan"].prefill_plan_work
assign_ranks_for_sharing = _MODULES["wave_plan"].assign_ranks_for_sharing
build_tree = _MODULES["wave_plan"]._build_tree
find_executor_cost_profile = _MODULES["executor_cost"].find_executor_cost_profile
pack_wave_ranks = _MODULES["rank_packing"].pack_wave_ranks
prompt_score = _MODULES["rank_packing"].prompt_score
unit_features = _MODULES["rank_packing"].unit_features
RankPacking = _MODULES["rank_packing"].RankPacking
seg_info = _MODULES["rank_packing"]._seg_info
unit_segs = _MODULES["rank_packing"]._unit_segs
RankState = _MODULES["rank_packing"]._RankState
feature_score = _MODULES["rank_packing"]._feature_score

B = 16  # block tokens
POOL = 100_000  # effectively unlimited for these waves
CHUNK = 131_072  # the server default; never binds at this scale
GIB = 1 << 30
GPTOSS = find_executor_cost_profile("gpt-oss-120b", (9, 0), 141 * GIB, 4)
assert GPTOSS is not None


def _tokens(uid, blocks):
    """``blocks`` blocks of tokens, unique to ``uid`` (blocks * B < 4096)."""
    return [uid * 4096 + k for k in range(blocks * B)]


def _family(uid, size, prefix_blocks=16, branch_blocks=8):
    """A long shared prefix, a two-way sub-prefix, then private suffixes.

    The sub-prefix is what makes a family halvable: without it the split rule
    would jump straight from the whole family to one unit per prompt.
    """
    head = _tokens(uid, prefix_blocks)
    prompts = []
    for i in range(size):
        prompts.append(
            head
            + _tokens(uid * 100 + 1 + i % 2, branch_blocks)
            + _tokens(uid * 10000 + i, 1)
            + [7]
        )
    return prompts


def _few_shot(uid, size, prefix_blocks=32):
    """MMLU-like: one shared few-shot prefix, one distinct question each."""
    head = _tokens(uid, prefix_blocks)
    return [head + _tokens(uid * 10000 + i, 2) + [7] for i in range(size)]


def _nested(uid, groups, per_group):
    """Long heterogeneous prompts under nested prefixes of uneven tail length."""
    root = _tokens(uid, 8)
    prompts = []
    for g in range(groups):
        mid = _tokens(uid * 100 + g, 6)
        for j in range(per_group):
            prompts.append(root + mid + _tokens(uid * 10000 + g * 100 + j, 1 + j % 4) + [5])
    return prompts


def _hot_family(uid, size, prefix_blocks=40):
    """One long shared document, then a distinct continuation per prompt.

    This is the shape a per-unit scorer gets wrong: split one level the family
    becomes ``size`` single-prompt units, and charging each of them its own copy
    of the shared document makes every split look worse than one busy rank.
    """
    head = _tokens(uid, prefix_blocks)
    return [head + _tokens(uid * 10000 + i, 1 + i % 4) + [7] for i in range(size)]


TIE = _family(1, 100) + _family(2, 100) + _family(3, 100)
HOT_FAMILY = _hot_family(11, 120)
HOT = HOT_FAMILY + _family(12, 12) + _family(13, 12) + _family(14, 12)


def plan(prompts, chunk_tokens=CHUNK):
    result = plan_wave_prefix_sharing(
        prompts, block_tokens=B, pool_pages=POOL, chunk_tokens=chunk_tokens
    )
    validate_wave_prefix_plan(result, prompts)
    return result


def exact_seconds(prompts, profile=GPTOSS):
    """Exact cost of one rank group: plan it, then score its measured work."""
    if not prompts:
        return 0.0
    work = prefill_plan_work(plan(prompts))
    if profile is None:
        return float(work.computed_tokens)
    return max(
        profile.seconds(
            work.computed_tokens, work.full_attention_pairs, work.item_rows, work.chunks
        ),
        0.0,
    )


def exact_rank_seconds(prompts, ranks, world, profile=GPTOSS):
    return [
        exact_seconds([p for p, r in zip(prompts, ranks) if r == rank], profile)
        for rank in range(world)
    ]


def pack(prompts, world=4, profile=GPTOSS, fixed=()):
    return pack_wave_ranks(
        prompts,
        world_size=world,
        block_tokens=B,
        chunk_tokens=CHUNK,
        profile=profile,
        fixed=fixed,
    )


def test_tie_case_fills_every_rank_and_lowers_the_exact_max():
    legacy = assign_ranks_for_sharing(TIE, world_size=4, block_tokens=B)
    # Splitting a family ties the busiest rank at 2212 computed tokens, so the
    # legacy planner rejects it and parks one family per rank: [100, 100, 100, 0].
    assert [legacy.count(r) for r in range(4)] == [100, 100, 100, 0]

    packed = pack(TIE)
    counts = [packed.ranks.count(r) for r in range(4)]
    assert all(count > 0 for count in counts), counts
    assert max(exact_rank_seconds(TIE, packed.ranks, 4)) < max(
        exact_rank_seconds(TIE, legacy, 4)
    )


def test_hot_family_split_beats_keeping_it_whole():
    packed = pack(HOT)
    hot = packed.ranks[:len(HOT_FAMILY)]
    assert len(set(hot)) >= 2, hot

    # What a per-unit scorer settles on: the hot family whole on one rank and
    # one small family per remaining rank, so three ranks stay nearly idle.
    whole = [0] * len(HOT_FAMILY) + [1] * 12 + [2] * 12 + [3] * 12
    legacy = assign_ranks_for_sharing(HOT, world_size=4, block_tokens=B)
    assert len(set(legacy[:len(HOT_FAMILY)])) == 1, "legacy keeps it whole"
    busiest = max(exact_rank_seconds(HOT, packed.ranks, 4))
    assert busiest < max(exact_rank_seconds(HOT, whole, 4))
    assert busiest < max(exact_rank_seconds(HOT, legacy, 4))


def union_features(prompts, groups):
    """Feed one rank's units through ``_RankState`` as ``_place`` does."""
    tree = build_tree(prompts, B)
    lengths = [len(p) for p in prompts]
    state = RankState(seg_info(tree, B))
    for members in groups:
        state.add(unit_segs(tree, members, lengths, B))
    return state.features()


def test_rank_features_are_exact_however_the_members_are_grouped():
    rng = random.Random(11)
    for _ in range(25):
        vocab = [_tokens(v, 1) for v in range(rng.randint(2, 5))]
        prompts = [
            [t for _ in range(rng.randint(0, 5)) for t in rng.choice(vocab)]
            + [rng.randint(0, 3) for _ in range(rng.randint(1, 2 * B))]
            for _ in range(rng.randint(2, 12))
        ]
        chosen = rng.sample(range(len(prompts)), rng.randint(1, len(prompts)))
        grouped = {}
        for index in chosen:
            grouped.setdefault(rng.randint(0, 2), []).append(index)

        feats = union_features(prompts, list(grouped.values()))
        assert feats == union_features(prompts, [[i] for i in chosen])
        work = prefill_plan_work(plan([prompts[i] for i in sorted(chosen)]))
        assert feats.computed_tokens == work.computed_tokens
        assert feats.attention_pairs == work.full_attention_pairs
        assert feats.item_rows == work.item_rows
        # depth is the one bound left: it may over-count merged pooled runs.
        lengths = [len(p) for p in prompts]
        exact = unit_features(build_tree(prompts, B), sorted(chosen), lengths, B)
        assert feats.depth >= exact.depth


def test_packed_rank_predictions_match_each_rank_group_plan():
    for wave in (TIE, HOT, _few_shot(4, 120)):
        packed = pack(wave)
        for rank in range(4):
            members = [i for i, r in enumerate(packed.ranks) if r == rank]
            feats = union_features(wave, [[i] for i in members])
            assert packed.predicted[rank] == feature_score(GPTOSS, feats, CHUNK)
            group = [wave[i] for i in members]
            if not group:
                continue
            work = prefill_plan_work(plan(group))
            assert feats.computed_tokens == work.computed_tokens
            assert feats.attention_pairs == work.full_attention_pairs
            assert feats.item_rows == work.item_rows


def test_unit_features_match_prefill_plan_work():
    rng = random.Random(7)
    for _ in range(25):
        vocab = [_tokens(v, 1) for v in range(rng.randint(2, 5))]
        prompts = [
            [t for _ in range(rng.randint(0, 5)) for t in rng.choice(vocab)]
            + [rng.randint(0, 3) for _ in range(rng.randint(1, 2 * B))]
            for _ in range(rng.randint(2, 12))
        ]
        tree = build_tree(prompts, B)
        lengths = [len(p) for p in prompts]
        subsets = [list(range(len(prompts)))]
        for _ in range(3):
            subsets.append(
                sorted(rng.sample(range(len(prompts)), rng.randint(1, len(prompts))))
            )
        for members in subsets:
            feats = unit_features(tree, members, lengths, B)
            work = prefill_plan_work(plan([prompts[i] for i in members]))
            assert feats.computed_tokens == work.computed_tokens
            assert feats.attention_pairs == work.full_attention_pairs
            assert feats.item_rows == work.item_rows


def test_random_packings_never_worse_than_one_rank():
    rng = random.Random(1)
    for profile in (None, GPTOSS):
        for _ in range(40):
            vocab = [_tokens(v, 1) for v in range(rng.randint(2, 5))]
            prompts = [
                [t for _ in range(rng.randint(0, 4)) for t in rng.choice(vocab)]
                + [rng.randint(0, 3)]
                for _ in range(rng.randint(1, 10))
            ]
            world = rng.randint(1, 4)
            packed = pack(prompts, world=world, profile=profile)
            assert len(packed.ranks) == len(prompts)
            assert all(0 <= r < world for r in packed.ranks)
            loads = exact_rank_seconds(prompts, packed.ranks, world, profile)
            assert max(loads) <= exact_seconds(prompts, profile) + 1e-12


def test_every_rank_group_validates():
    packed = pack(TIE)
    for rank in range(4):
        group = [p for p, r in zip(TIE, packed.ranks) if r == rank]
        assert group
        plan(group)  # validates as a side effect


_REPLAY_SCRIPT = """
import sys
sys.path.insert(0, {tests!r})
import test_prefix_rank_packing as mod
print(mod.pack(mod.TIE).ranks)
"""


def test_packing_is_deterministic_across_hash_seeds():
    baseline = pack(TIE).ranks
    assert pack(TIE).ranks == baseline
    env = dict(os.environ, PYTHONHASHSEED="424242")
    done = subprocess.run(
        [sys.executable, "-c", _REPLAY_SCRIPT.format(tests=str(Path(__file__).parent))],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert done.stdout.strip() == repr(baseline), done.stderr


def score_one(tokens, profile=GPTOSS):
    return prompt_score(tokens, block_tokens=B, chunk_tokens=CHUNK, profile=profile)


def test_prompt_score_is_the_packers_score_of_a_one_prompt_wave():
    for profile in (None, GPTOSS):
        for tokens in (_tokens(1, 1) + [7], _few_shot(6, 1)[0], _family(7, 1)[0]):
            solo = pack([tokens], world=1, profile=profile)
            assert score_one(tokens, profile) == solo.predicted[0]
    # Pinning that same score is what a re-entering sequence contributes.
    tokens = _few_shot(8, 1)[0]
    pinned = pack([_tokens(9, 2) + [7]], world=2, profile=GPTOSS,
                  fixed=((1, score_one(tokens)),))
    assert pinned.predicted[1] == score_one(tokens)


def test_prompt_score_depends_only_on_length():
    # Non-owner ranks score a re-entry from stale tokens of the right length.
    tokens = _few_shot(10, 1)[0]
    stale = [0] * len(tokens)
    for profile in (None, GPTOSS):
        assert score_one(tokens, profile) == score_one(stale, profile)


def test_prompt_score_rejects_bad_arguments():
    with pytest.raises(ValueError):
        score_one([])
    with pytest.raises(ValueError):
        prompt_score([1], block_tokens=0, chunk_tokens=CHUNK, profile=None)


def test_pinned_load_shifts_prompts_off_the_busy_rank():
    prompts = _family(1, 40) + _family(2, 40) + _family(3, 40) + _family(4, 40)
    packed = pack(prompts, fixed=((2, 0.5),))
    counts = [packed.ranks.count(r) for r in range(4)]
    assert counts[2] < min(counts[r] for r in (0, 1, 3)), counts


def test_profile_lookup_matches_device_properties():
    assert find_executor_cost_profile("openai/gpt-oss-120b", (9, 0), 140 * GIB, 4) is not None
    # Same sm_90, different total memory (80 GB and 96 GB parts).
    assert find_executor_cost_profile("gpt-oss-120b", (9, 0), 80 * GIB, 4) is None
    assert find_executor_cost_profile("gpt-oss-120b", (9, 0), 96 * GIB, 4) is None
    assert find_executor_cost_profile("gpt-oss-120b", (10, 0), 141 * GIB, 4) is None
    # The fit is layout-specific: another world size must not match.
    assert find_executor_cost_profile("gpt-oss-120b", (9, 0), 141 * GIB, 8) is None
    # A smaller model sharing the name prefix must not inherit coefficients.
    assert find_executor_cost_profile("gpt-oss-20b", (9, 0), 141 * GIB, 4) is None
    assert find_executor_cost_profile("GLM-5.2", (9, 0), 141 * GIB, 4) is None


def _replay(prompts, world=4):
    """Exact busiest-rank seconds for the legacy planner and for the packer."""
    legacy = assign_ranks_for_sharing(prompts, world_size=world, block_tokens=B)
    packed = pack(prompts, world=world)
    return (
        max(exact_rank_seconds(prompts, legacy, world)),
        max(exact_rank_seconds(prompts, packed.ranks, world)),
    )


def test_replay_of_synthetic_waves_is_never_worse_in_sum():
    cases = {
        "three-family forest": TIE,
        "shared few-shot prefix": _few_shot(4, 120),
        "nested heterogeneous": _nested(5, 4, 15),
    }
    observed = {name: _replay(wave) for name, wave in cases.items()}
    legacy_total = sum(pair[0] for pair in observed.values())
    packed_total = sum(pair[1] for pair in observed.values())
    assert packed_total <= legacy_total + 1e-9, observed


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        pack_wave_ranks(
            [[1]], world_size=0, block_tokens=B, chunk_tokens=CHUNK, profile=None
        )
    with pytest.raises(ValueError):
        pack_wave_ranks(
            [[]], world_size=2, block_tokens=B, chunk_tokens=CHUNK, profile=None
        )
    empty = pack_wave_ranks(
        [], world_size=2, block_tokens=B, chunk_tokens=CHUNK, profile=None,
        fixed=((1, 3.0),),
    )
    assert empty == RankPacking((), (0.0, 3.0), 0)
