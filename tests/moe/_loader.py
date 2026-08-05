"""Import batchgen.moe MXFP4/marlin modules for CPU tests.

On GPU machines with the full package installed, a plain import works. On
CPU-only dev machines, executing batchgen/__init__.py fails (it imports the
client and checks compiled batchgen_kernels), so we fall back to lightweight
namespace-package stubs pointing at the real source directories — the target
modules themselves (mxfp4_oracle_vector, marlin_weight_prep) import only
torch/numpy/logging and are fully CPU-safe.
"""

import importlib
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_moe_modules():
    """Return (mxfp4_oracle_vector, marlin_weight_prep) modules."""
    try:
        oracle = importlib.import_module("batchgen.moe.mxfp4_oracle_vector")
        mwp = importlib.import_module("batchgen.moe.marlin_weight_prep")
        return oracle, mwp
    except Exception:
        pass

    for pkg, rel in (("batchgen", "batchgen"), ("batchgen.moe", "batchgen/moe")):
        if pkg not in sys.modules:
            mod = types.ModuleType(pkg)
            mod.__path__ = [str(REPO_ROOT / rel)]
            sys.modules[pkg] = mod
    oracle = importlib.import_module("batchgen.moe.mxfp4_oracle_vector")
    mwp = importlib.import_module("batchgen.moe.marlin_weight_prep")
    return oracle, mwp
