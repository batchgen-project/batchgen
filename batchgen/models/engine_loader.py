import logging
import os

# Preload the pip libucx-cu12 UCX libraries (if installed) so core_engine's
# distributed-weight daemon resolves libucp/libuct/libucs/libucm at load time,
# independent of LD_LIBRARY_PATH. No-op if the wheel is absent (a system UCX or
# the linker rpath baked at build time then satisfies the .so).
try:
    import libucx

    libucx.load_library()
except Exception:
    pass

from batchgen.op_builder.core_engine import CoreEngineBuilder

use_jit = False
try:
    from batchgen import core_engine
except ImportError:
    logging.debug("Do not detect pre-installed core engine, use JIT mode.")
    use_jit = True

# print("run core engine importer")

core_engine = CoreEngineBuilder().load() if use_jit else core_engine
