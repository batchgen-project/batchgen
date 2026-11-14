"""Both prefill [METRICS] emissions carry the ledger-contract fields.

The prefill report tooling binds a run's FLOPs model through seq_len_min/max,
proves determinism through first_sampled_token_ids, and attributes ranks
through world_size. The prepacked and the pooled wave paths each emit one
structured record; a path that drops one of these keys silently falls out of
the ledger (the pooled path shipped that way), so this test pins the contract
at the source level for both emission sites.
"""

import pathlib
import re

# S5b: both [METRICS] emission sites (prefill_prepacked, _prefill_prepacked_pool)
# moved verbatim with their bodies to PrefillEngine.
ENGINE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "batchgen" / "worker" / "prefill_engine.py"
)

CONTRACT_KEYS = (
    '"sequences"',
    '"prefill_s"',
    '"seq_len_min"',
    '"seq_len_max"',
    '"world_size"',
    '"first_sampled_token_ids"',
)


def _emission_block(source: str, component: str) -> str:
    starts = [m.start() for m in re.finditer(re.escape(component), source)]
    assert starts, f"no METRICS emission mentions {component!r}"
    blocks = []
    for start in starts:
        window = source[start:start + 4000]
        if '"phase"' in window[:200] and '"prefill"' in window[:300]:
            blocks.append(window)
    assert blocks, f"no phase=prefill METRICS block for {component!r}"
    return "\n".join(blocks)


def test_prepacked_emission_carries_contract_keys():
    source = ENGINE.read_text()
    block = _emission_block(source, '"component": "prefill_full_attention"')
    for key in CONTRACT_KEYS:
        assert key in block, f"prepacked METRICS block lost {key}"


def test_pooled_wave_emission_carries_contract_keys():
    source = ENGINE.read_text()
    block = _emission_block(source, '"component": "prefix_pool"')
    for key in CONTRACT_KEYS:
        assert key in block, f"pooled wave METRICS block lost {key}"
