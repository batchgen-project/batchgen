# ---------------------------------------------------------------------------- #
#  BatchGen — DSA kernels                                                       #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""The DSA page-table-width kernels compile one binary for every width.

Under test: kernel revision r2 of ``_transform_selected_positions_kernel``
(``selected_page_table.py``) and ``_dense_paged_gather_kernel``
(``fused_paged_gather.py``). Eager decode hands both the ACTIVE page-table
width, which follows the longest live sequence; r1 baked it as a
``tl.constexpr``, so every new width was a new binary compiled mid-request.
r2 passes it as a runtime int listed in ``do_not_specialize``.

The check is on the specialisation key Triton derives from the kernel
signature (see ``batchgen_kernels/triton_prewarm.py``: constexpr values,
argument dtypes, and the "== 1" / "divisible by 16" class of every
specialised runtime int), modelled from the kernel source, so neither Triton
nor CUDA is needed. Bit-identity of the outputs is pinned on GPU by
``tests/test_selected_page_table.py`` and ``tests/test_dsa_valid_token_kernels.py``.

Run: python -m pytest tests/unit/test_dsa_width_runtime_keys.py -q
"""

import ast
import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_DSA = _ROOT / "batchgen_kernels" / "attention" / "dsa"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORE = _load(_ROOT / "batchgen_kernels" / "triton_prewarm.py",
             "_triton_prewarm_core_for_width_keys")

WIDTHS = (1, 2, 16, 17, 64, 1584, 3169, 4096, 4097, 50_000)

# ``(file, kernel, width argument, structural constexprs it must keep)``.
KERNELS = (
    ("selected_page_table.py", "_transform_selected_positions_kernel",
     "max_pages_per_seq",
     ("index_topk", "page_size", "has_slot_indices", "has_valid_tokens",
      "block")),
    ("fused_paged_gather.py", "_dense_paged_gather_kernel",
     "max_pages_per_seq",
     ("D", "page_size", "BLOCK_T", "BLOCK_D")),
)


class KernelSignature:
    """What the specialisation key of one ``@triton.jit`` kernel reads."""

    def __init__(self, params, constexprs, do_not_specialize):
        self.params = tuple(params)
        self.constexprs = frozenset(constexprs)
        self.do_not_specialize = frozenset(do_not_specialize)

    @classmethod
    def from_source(cls, path: Path, name: str) -> "KernelSignature":
        tree = ast.parse(path.read_text())
        kernel, = [node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == name]
        params = [arg.arg for arg in kernel.args.args]
        constexprs = [arg.arg for arg in kernel.args.args
                      if arg.annotation is not None
                      and ast.unparse(arg.annotation) == "tl.constexpr"]
        dns = []
        for decorator in kernel.decorator_list:
            if isinstance(decorator, ast.Call) and ast.unparse(
                    decorator.func) == "triton.jit":
                for keyword in decorator.keywords:
                    if keyword.arg == "do_not_specialize":
                        dns = list(ast.literal_eval(keyword.value))
        return cls(params, constexprs, dns)

    def key(self, ints: dict) -> tuple:
        """The key of one launch whose int arguments are ``ints``; the
        pointer arguments are fixed (same dtypes, aligned allocations)."""
        parts = []
        for name in self.params:
            if name not in ints:
                parts.append((name, "ptr"))
            elif name in self.constexprs:
                parts.append((name, "constexpr", ints[name]))
            elif name in self.do_not_specialize:
                parts.append((name, "i32"))
            else:
                parts.append((name, "i32", CORE.int_class(ints[name])))
        return tuple(parts)

    def r1(self, width_arg: str) -> "KernelSignature":
        """The pre-revision signature: the width a constexpr."""
        return KernelSignature(self.params, self.constexprs | {width_arg},
                               self.do_not_specialize - {width_arg})


def _launch_ints(params, width_arg, width):
    """Every non-pointer argument of the kernel at fixed structural values."""
    fixed = {"index_topk": 2048, "page_size": 64, "has_slot_indices": True,
             "has_valid_tokens": False, "block": 256, "D": 128,
             "BLOCK_T": 8, "BLOCK_D": 128, "num_tokens": 4100}
    ints = {name: value for name, value in fixed.items() if name in params}
    ints[width_arg] = width
    return ints


@pytest.mark.parametrize("filename,kernel,width_arg,structural", KERNELS)
def test_width_is_a_runtime_int_outside_the_key(filename, kernel, width_arg,
                                                structural):
    sig = KernelSignature.from_source(_DSA / filename, kernel)
    assert width_arg in sig.params
    assert width_arg not in sig.constexprs
    assert width_arg in sig.do_not_specialize
    keys = {sig.key(_launch_ints(sig.params, width_arg, w)) for w in WIDTHS}
    assert len(keys) == 1


@pytest.mark.parametrize("filename,kernel,width_arg,structural", KERNELS)
def test_r1_keyed_every_width(filename, kernel, width_arg, structural):
    """The bug r2 removes: as a constexpr each width was its own binary."""
    r1 = KernelSignature.from_source(_DSA / filename, kernel).r1(width_arg)
    keys = {r1.key(_launch_ints(r1.params, width_arg, w)) for w in WIDTHS}
    assert len(keys) == len(WIDTHS)


@pytest.mark.parametrize("filename,kernel,width_arg,structural", KERNELS)
def test_structural_constexprs_are_kept(filename, kernel, width_arg,
                                        structural):
    sig = KernelSignature.from_source(_DSA / filename, kernel)
    assert set(structural) <= sig.constexprs
    # Only the width left the key: no other argument was de-specialised.
    assert sig.do_not_specialize == {width_arg}


@pytest.mark.parametrize("filename,kernel,width_arg,structural", KERNELS)
def test_revision_is_recorded_in_the_source(filename, kernel, width_arg,
                                            structural):
    assert "Kernel revision r2" in (_DSA / filename).read_text()
