import hashlib
import importlib.util
import json
import logging
import sys
import uuid
from multiprocessing import shared_memory
from pathlib import Path

import pytest
import torch


_ROOT = Path(__file__).parents[2]


def _load(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_INTEGRITY = _load(
    "test_prefix_integrity_module", "batchgen/prefix_reuse/integrity.py"
)
IntegrityCounters = _INTEGRITY.IntegrityCounters
IntegrityLedger = _INTEGRITY.IntegrityLedger
PrefixIntegrityError = _INTEGRITY.PrefixIntegrityError
page_hashes = _INTEGRITY.page_hashes
page_identity = _INTEGRITY.page_identity
token_chain_hash = _INTEGRITY.token_chain_hash

_PAGE_SHAPE = (64, 8, 64)  # GPT-OSS host page: tokens x KV heads x head dim
_LAYERS = 3


def _pages(n, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((n, *_PAGE_SHAPE), generator=generator).to(torch.bfloat16)


def _kv_hashes(n, seed):
    layers = range(_LAYERS)
    k = torch.stack([page_hashes(_pages(n, seed + i)) for i in layers], 1)
    v = torch.stack([page_hashes(_pages(n, seed + 100 + i)) for i in layers], 1)
    return k, v


@pytest.fixture
def ledger():
    ledger = IntegrityLedger(f"bgi_{uuid.uuid4().hex[:8]}", 16, _LAYERS)
    yield ledger
    ledger.close()
    ledger.unlink()


def test_identical_bytes_give_identical_hashes():
    pages = _pages(3)
    hashes = page_hashes(pages)
    assert hashes.shape == (3, 2) and hashes.dtype == torch.int64
    assert torch.equal(hashes, page_hashes(pages.clone()))


def test_single_bf16_element_change_changes_only_that_page():
    pages = _pages(3)
    before = page_hashes(pages)
    pages.view(torch.int16)[1, 5, 3, 7] ^= 1
    after = page_hashes(pages)
    assert torch.equal(after[[0, 2]], before[[0, 2]])
    assert (after[1] != before[1]).all()


def test_swapping_two_tokens_within_a_page_changes_the_hash():
    pages = _pages(1)
    swapped = pages.clone()
    swapped[0, [3, 10]] = pages[0, [10, 3]]
    assert not torch.equal(swapped, pages)
    assert (page_hashes(swapped) != page_hashes(pages)).all()


def test_swapping_pages_moves_their_hashes():
    pages = _pages(3)
    order = [2, 0, 1]
    assert torch.equal(page_hashes(pages[order]), page_hashes(pages)[order])


def test_extreme_int16_patterns_do_not_overflow():
    shape = (1, 64, 16, 64)  # 2**16 elements: the exactness bound
    low = torch.full(shape, -(2**15), dtype=torch.int16).view(torch.bfloat16)
    high = torch.full(shape, 2**15 - 1, dtype=torch.int16).view(torch.bfloat16)
    h_low, h_high = page_hashes(low), page_hashes(high)
    assert torch.equal(h_low, page_hashes(low.clone()))
    assert torch.equal(h_high, page_hashes(high.clone()))
    # Exact sums: h_low = -2**15 * sum(w), h_high = (2**15 - 1) * sum(w).
    assert (h_low < 0).all() and (h_high > 0).all()
    for column in range(2):
        low_value, high_value = int(h_low[0, column]), int(h_high[0, column])
        assert high_value * 2**15 == -low_value * (2**15 - 1)


def test_page_hashes_reject_unsupported_inputs():
    with pytest.raises(ValueError, match="2-byte dtype"):
        page_hashes(torch.zeros((1, 4), dtype=torch.float32))
    with pytest.raises(ValueError, match="exact hashing"):
        page_hashes(torch.zeros((1, 2**16 + 1), dtype=torch.bfloat16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cpu_and_cuda_hashes_match():
    pages = _pages(4)
    assert torch.equal(page_hashes(pages.cuda()).cpu(), page_hashes(pages))


def test_token_chain_hash_is_exact_order_sensitive_and_63_bit():
    tokens = list(range(1000, 1128))
    value = token_chain_hash(tokens)
    reference = hashlib.blake2b(
        b"".join(t.to_bytes(4, "little", signed=True) for t in tokens),
        digest_size=8,
    ).digest()
    assert value == int.from_bytes(reference, "little") & ((1 << 63) - 1)
    assert 0 <= value < 2**63
    changed = list(tokens)
    changed[77] += 1
    swapped = list(tokens)
    swapped[3], swapped[4] = swapped[4], swapped[3]
    assert token_chain_hash(changed) != value
    assert token_chain_hash(swapped) != value


def test_page_identity_matches_prefix_chain_hashes():
    tokens = list(range(5, 5 + 200))
    chains, raw_ends = page_identity(tokens, 3)
    assert chains == [token_chain_hash(tokens[: (i + 1) * 64]) for i in range(3)]
    assert raw_ends == [tokens[63], tokens[127], tokens[191]]
    with pytest.raises(PrefixIntegrityError, match="cannot fill"):
        page_identity(tokens, 4)


def test_second_attach_sees_the_same_rows(ledger):
    tokens = list(range(128))
    chains, raw_ends = page_identity(tokens, 2)
    k, v = _kv_hashes(2, seed=1)
    ledger.record([4, 9], k, v, chains, raw_ends, rank=2)
    peer = IntegrityLedger(ledger.name[: -len("_integrity")], 16, _LAYERS)
    try:
        assert peer.verify([4, 9], k, v, context="peer") == 2
        assert peer.verify_identity([4, 9], tokens, context="peer") == 2
    finally:
        peer.close()


def test_verify_rejects_missing_rows_and_kv_mismatches(ledger):
    k, v = _kv_hashes(2, seed=3)
    ledger.record([5, 6], k, v, [11, 12], [0, 0], rank=1)
    assert ledger.verify([5, 6], k, v, context="h2d") == 2

    with pytest.raises(PrefixIntegrityError, match=r"h2d: page 7 has no ledger row"):
        ledger.verify([5, 7], k, v, context="h2d")

    bad_k = k.clone()
    bad_k[1, 2, 0] += 1
    with pytest.raises(PrefixIntegrityError, match=r"page 6 layer 2 K hash mismatch"):
        ledger.verify([5, 6], bad_k, v, context="h2d")

    bad_v = v.clone()
    bad_v[0, 1, 1] -= 1
    with pytest.raises(
        PrefixIntegrityError, match=r"page 5 layer 1 V hash mismatch.*rank 1"
    ):
        ledger.verify([5, 6], k, bad_v, context="h2d")


def test_verify_identity_rejects_wrong_tokens_and_raw_end(ledger):
    tokens = list(range(300, 300 + 192))
    chains, raw_ends = page_identity(tokens, 3)
    k, v = _kv_hashes(3, seed=5)
    ledger.record([1, 2, 3], k, v, chains, raw_ends, rank=0)
    assert ledger.verify_identity([1, 2, 3], tokens, context="lookup") == 3

    wrong = list(tokens)
    wrong[100] += 1  # inside request page 1
    with pytest.raises(
        PrefixIntegrityError,
        match=r"lookup: page 2 \(request page 1\) token_chain_hash mismatch",
    ):
        ledger.verify_identity([1, 2, 3], wrong, context="lookup")

    ledger.record([8], k[:1], v[:1], chains[:1], [raw_ends[0] + 1], rank=0)
    with pytest.raises(PrefixIntegrityError, match="raw_end_token mismatch"):
        ledger.verify_identity([8], tokens, context="lookup")


def test_ledger_rejects_bad_ids_and_shapes(ledger):
    k, v = _kv_hashes(1, seed=7)
    with pytest.raises(ValueError, match="outside"):
        ledger.record([-1], k, v, [0], [0], rank=0)
    with pytest.raises(ValueError, match="outside"):
        ledger.verify([16], k, v, context="x")
    with pytest.raises(ValueError, match="K/V hashes must be"):
        ledger.record([0], k[:, :2], v[:, :2], [0], [0], rank=0)
    with pytest.raises(ValueError, match="chain hashes"):
        ledger.record([0, 1], torch.cat([k, k]), torch.cat([v, v]), [0], [0], rank=0)


def test_attach_to_mismatched_geometry_fails_loud(ledger, monkeypatch):
    monkeypatch.setattr(_INTEGRITY, "_ATTACH_TIMEOUT_S", 0.2)
    base = ledger.name[: -len("_integrity")]
    with pytest.raises(RuntimeError, match="expected"):
        IntegrityLedger(base, 4096, _LAYERS)


def test_unlink_removes_the_segment():
    ledger = IntegrityLedger(f"bgi_{uuid.uuid4().hex[:8]}", 4, 1)
    name = ledger.name
    ledger.close()
    ledger.unlink()
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name)


def test_counters_emit_prefix_integrity_metric(caplog):
    metrics = _load("test_prefix_integrity_metrics", "batchgen/prefix_reuse/metrics.py")
    counters = IntegrityCounters()
    counters.pages_ledgered += 4
    counters.pages_verified["h2d"] += 3
    counters.identity_checks += 1
    with caplog.at_level(logging.INFO):
        metrics.emit_prefix_cache_metric(rank=2, **counters.metric_record("summary"))
    record = next(
        item.message for item in caplog.records if item.message.startswith("[METRICS]")
    )
    assert json.loads(record[len("[METRICS] "):]) == {
        "component": "prefix_integrity",
        "identity_checks": 1,
        "pages_ledgered": 4,
        "pages_verified": {"h2d": 3},
        "phase": "summary",
        "rank": 2,
        "schema_version": 1,
    }


def test_integrity_flag_requires_prefix_cache(tmp_path, monkeypatch):
    server_args = _load(
        "test_prefix_integrity_server_args", "batchgen/server/server_args.py"
    )
    monkeypatch.setattr(server_args, "_ensure_local_port_free", lambda *_: None)
    argv = [
        "--model", "openai/gpt-oss-120b",
        "--host-kv-cache-size", "1",
        "--storage-path", str(tmp_path),
        "--prefix-cache-integrity-check",
    ]
    with pytest.raises(ValueError, match="requires --enable-prefix-cache"):
        server_args.prepare_server_args(argv)
    parsed = server_args.prepare_server_args(argv + ["--enable-prefix-cache"])
    assert parsed.prefix_cache_integrity_check is True
    default = server_args.prepare_server_args(argv[:-1])
    assert default.prefix_cache_integrity_check is False
