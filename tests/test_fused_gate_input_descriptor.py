"""The fused gate serves a stored input descriptor only for its pinned buffer.

A TMA descriptor declares an address and an extent. Remembering the one built
for a transient input and reusing it for a later, smaller tensor allocated at
the same address makes the tile loads read past that tensor (an illegal memory
access once the neighbouring memory is gone).
"""

import re
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "batchgen_kernels" / "src" / "moe" / "routing" / "fused_gate.cu"


def _function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[brace:index + 1]
    raise AssertionError(f"unterminated body of {signature}")


def test_forward_never_stores_a_per_call_input_in_the_context():
    source = SOURCE.read_text()
    forward = _function_body(source, "torch::Tensor fused_router_forward(")
    for field in ("tma_desc_a", "cached_input_ptr", "cached_input_rows", "has_cached_tma_a"):
        assert not re.search(rf"ctx->{field}\s*=[^=]", forward), (
            f"fused_router_forward assigns ctx->{field}: an input descriptor "
            "may only be stored by fused_gate_warmup, which pins its buffer"
        )


def test_warmup_keeps_the_pinned_buffer_alive():
    source = SOURCE.read_text()
    warmup = _function_body(source, "void fused_gate_warmup(")
    assert re.search(r"ctx->pinned_input\s*=\s*base_buffer\s*;", warmup)


def _context(e=32, k_dim=64):
    from batchgen.moe.routing.cuda_routing import FusedGateContext

    torch.manual_seed(20261122)
    weight = (torch.randn(e, k_dim, device="cuda") * 0.1).to(torch.bfloat16)
    context = FusedGateContext(weight, None, topk=4)
    if context._fallback:
        pytest.skip("fused gate extension unavailable on this device")
    return context, weight


def _reference(hidden, weight):
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return hidden.float().matmul(weight.float().t())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def _close(actual, reference):
    tolerance = 1e-5 + 1.6e-2 * reference.abs()
    assert bool(((actual.float() - reference).abs() <= tolerance).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_smaller_input_at_a_reused_address_is_computed_from_its_own_extent():
    context, weight = _context()
    big = (torch.randn(1024, 64, device="cuda") * 0.1).to(torch.bfloat16)
    _close(context.router_forward(big), _reference(big, weight))
    address = big.data_ptr()
    del big
    small = (torch.randn(8, 64, device="cuda") * 0.1).to(torch.bfloat16)
    if small.data_ptr() != address:
        pytest.skip("the allocator did not hand the freed address back")
    _close(context.router_forward(small), _reference(small, weight))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_views_of_the_warmed_buffer_and_other_inputs_both_match_reference():
    context, weight = _context()
    base = (torch.randn(256, 64, device="cuda") * 0.1).to(torch.bfloat16)
    context.warmup(base)
    for rows in (256, 128, 3):
        view = base[:rows]
        _close(context.router_forward(view), _reference(view, weight))
    other = (torch.randn(40, 64, device="cuda") * 0.1).to(torch.bfloat16)
    _close(context.router_forward(other), _reference(other, weight))
    # The pinned buffer is still served after an unrelated input.
    _close(context.router_forward(base[:17]), _reference(base[:17], weight))
