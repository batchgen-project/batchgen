# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-Linear                                                       #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""World-size-1 resident-EP decode runs without collectives (CPU regression).

At world size one the resident-EP dispatch is the identity: the gather of a
single rank is its own padded rows and the SUM-combine over one rank is a
no-op. ``ResidentEPMoELayer.forward`` therefore must not touch the
communicator at all — a single-GPU serve runs the resident decode without a
process group. This test drives ``forward`` on CPU with a spy communicator
and a stubbed fused kernel and asserts:

  * world_size == 1: no ``change_state`` / ``all_gather`` / ``all_reduce``
    call reaches the communicator (regression — the pre-fix path issued
    both unconditionally and crashed without an initialized group);
  * the identity dispatch is exact: the router sees the padded local rows
    themselves and the returned slice equals the kernel output on those
    rows, including an empty rank (0 local tokens) and a partially padded
    one (num_tokens < ntp);
  * world_size > 1 still issues all_gather + all_reduce (the guard must not
    over-skip on the multi-rank schedule).

NON-VACUITY: on the pre-fix module the spy records the two collectives at
world size one and the no-collective assertion fails.

Run: python -m pytest tests/test_resident_ep_world_size_1.py -x -q -rA
"""

from __future__ import annotations

import contextlib
import importlib.util
import pathlib
import sys
import types

import pytest
import torch

_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "batchgen"
    / "moe"
    / "fused_moe_bf16_resident.py"
)

H = 8          # hidden size
I_SZ = 4       # expert intermediate size
NUM_LOCAL = 4  # experts on this rank's shard


def _load_module_with_stubbed_kernels() -> types.ModuleType:
    """Load fused_moe_bf16_resident.py by path with batchgen_kernels stubbed.

    The module imports ``fused_moe_bf16`` from the Triton package at import
    time and ``triton`` is not installed on a CPU-only box; the stub only has
    to carry that one symbol. Tests monkeypatch the loaded module's
    ``fused_moe_bf16`` binding per case — nothing here executes a kernel.
    """
    def _unpatched_kernel(*args, **kwargs):
        raise AssertionError("test must monkeypatch fused_moe_bf16")

    pkg = types.ModuleType("batchgen_kernels")
    triton_pkg = types.ModuleType("batchgen_kernels.triton")
    kernel_mod = types.ModuleType("batchgen_kernels.triton.fused_moe_bf16")
    kernel_mod.fused_moe_bf16 = _unpatched_kernel
    triton_pkg.fused_moe_bf16 = kernel_mod
    pkg.triton = triton_pkg

    names = (
        "batchgen_kernels",
        "batchgen_kernels.triton",
        "batchgen_kernels.triton.fused_moe_bf16",
    )
    saved = {name: sys.modules.get(name) for name in names}
    sys.modules["batchgen_kernels"] = pkg
    sys.modules["batchgen_kernels.triton"] = triton_pkg
    sys.modules["batchgen_kernels.triton.fused_moe_bf16"] = kernel_mod
    try:
        spec = importlib.util.spec_from_file_location(
            "_fused_moe_bf16_resident_under_test", _MODULE_PATH
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


@pytest.fixture(scope="module")
def resident_mod() -> types.ModuleType:
    return _load_module_with_stubbed_kernels()


@pytest.fixture(autouse=True)
def _reset_ntp(resident_mod):
    """num_tokens_per_rank is class state; never leak it across tests."""
    yield
    resident_mod.ResidentEPMoELayer.num_tokens_per_rank = None


class SpyComm:
    """Records every communicator touch; services the calls so the pre-fix
    path still completes and the no-collective assertion is the only
    discriminator."""

    def __init__(self):
        self.calls = []

    @contextlib.contextmanager
    def change_state(self, enable):
        self.calls.append(("change_state", enable))
        yield

    def all_gather(self, out, inp):
        self.calls.append(("all_gather", tuple(out.shape), tuple(inp.shape)))
        world = out.shape[0] // inp.shape[0]
        out.copy_(inp.repeat(world, 1))

    def all_reduce(self, tensor, op=None):
        self.calls.append(("all_reduce", tuple(tensor.shape)))

    @property
    def collective_calls(self):
        return [c for c in self.calls if c[0] != "change_state"]


class RecordingGate:
    """Top-k=1 router onto local expert 0; records what it was given."""

    def __init__(self):
        self.seen = []

    def __call__(self, x):
        num_global = x.shape[0]
        self.seen.append(x.detach().clone())
        topk_idx = torch.zeros((num_global, 1), dtype=torch.int64)
        topk_weight = torch.ones((num_global, 1), dtype=torch.float32)
        return topk_idx, topk_weight


def _affine_kernel_stub(all_tokens, w13, w2, masked_weight, local_ids):
    """Row-preserving stand-in for the fused kernel: a distinct affine map,
    so slicing/padding mistakes change the output."""
    return all_tokens * 2.0 + 1.0


def _make_layer(resident_mod, comm, world_size, rank):
    w13 = torch.zeros((NUM_LOCAL, 2 * I_SZ, H))
    w2 = torch.zeros((NUM_LOCAL, H, I_SZ))
    return resident_mod.ResidentEPMoELayer(
        layer_idx=0, w13=w13, w2=w2, comm=comm,
        world_size=world_size, rank=rank, expert_start=0,
    )


@pytest.mark.parametrize("num_tokens,ntp", [(3, 3), (2, 5), (0, 4)])
def test_world_size_1_forward_never_enters_collectives(
        resident_mod, monkeypatch, num_tokens, ntp):
    monkeypatch.setattr(resident_mod, "fused_moe_bf16", _affine_kernel_stub)
    spy = SpyComm()
    layer = _make_layer(resident_mod, spy, world_size=1, rank=0)
    resident_mod.ResidentEPMoELayer.set_num_tokens_per_rank(ntp)
    gate = RecordingGate()
    torch.manual_seed(7 + num_tokens)
    x_local = torch.randn(num_tokens, H)

    out = layer.forward(x_local, gate)

    # Regression: a single rank exchanges nothing — the communicator is
    # never touched (pre-fix: all_gather + all_reduce recorded here).
    assert spy.calls == [], (
        f"world-size-1 resident decode entered the communicator: {spy.calls}"
    )

    # Identity dispatch: the router saw exactly the zero-padded local rows.
    assert len(gate.seen) == 1
    expected_global = torch.zeros(ntp, H)
    expected_global[:num_tokens] = x_local
    assert torch.equal(gate.seen[0].view(ntp, H), expected_global)

    # The returned slice is the kernel output on the local rows.
    assert out.shape == (num_tokens, H)
    assert torch.equal(out, x_local * 2.0 + 1.0)


def test_world_size_2_forward_still_issues_collectives(
        resident_mod, monkeypatch):
    monkeypatch.setattr(resident_mod, "fused_moe_bf16", _affine_kernel_stub)
    spy = SpyComm()
    layer = _make_layer(resident_mod, spy, world_size=2, rank=0)
    resident_mod.ResidentEPMoELayer.set_num_tokens_per_rank(3)
    torch.manual_seed(11)
    x_local = torch.randn(2, H)

    out = layer.forward(x_local, RecordingGate())

    names = [c[0] for c in spy.collective_calls]
    assert names == ["all_gather", "all_reduce"], (
        f"multi-rank schedule must keep both collectives, saw {spy.calls}"
    )
    assert out.shape == (2, H)
